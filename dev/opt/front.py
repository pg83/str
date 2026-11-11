"""Parse and type-check the string library's C source into a typed AST."""
from dataclasses import dataclass, field, replace

from . import ctype as T
from .builtins import BUILTINS
from .source import Token, integer as parse_integer, literal


@dataclass(eq=False)
class Decl:
    name: str
    type: T.Type
    token: Token
    storage: frozenset = frozenset()
    attrs: dict = field(default_factory=dict)
    params: list = field(default_factory=list)
    init: object = None
    body: object = None
    builtin: object = None


@dataclass
class Expr:
    kind: str
    type: T.Type
    token: Token
    args: tuple = ()
    value: object = None

    @property
    def lvalue(self):
        return self.kind in ('var', 'deref', 'member', 'index') and self.type.kind != 'function'


@dataclass
class Stmt:
    kind: str
    token: Token
    args: tuple = ()
    value: object = None


@dataclass
class Init:
    token: Token
    expr: Expr | None = None
    entries: list | None = None  # [(designators: tuple of field names / array indexes, Init)]
    type: T.Type | None = None


@dataclass
class Unit:
    path: str
    declarations: list
    records: dict
    dependencies: set = field(default_factory=set)
    definitions: list = field(default_factory=list)

    @property
    def functions(self):
        return [d for d in self.definitions if d.body is not None]


def const(expr):
    """A scalar integer constant expression, or None for a runtime value."""
    if expr.kind == 'int':
        return expr.value
    if expr.kind == 'cast':
        v = const(expr.args[0])
        return T.convert_integer(v, expr.type) if v is not None and expr.type.kind in ('int', 'ptr') else None
    if expr.kind == 'unary':
        v = const(expr.args[0])
        if v is None:
            return None
        r = {'+': lambda: v, '-': lambda: -v, '~': lambda: ~v, '!': lambda: int(not v)}[expr.value]()
        return T.convert_integer(r, expr.type)
    if expr.kind == 'conditional':
        c = const(expr.args[0])
        return const(expr.args[1] if c else expr.args[2]) if c is not None else None
    if expr.kind == 'binary':
        a = const(expr.args[0])
        if a is None:
            return None
        if expr.value == '&&' and not a:
            return 0
        if expr.value == '||' and a:
            return 1
        b = const(expr.args[1])
        if b is None:
            return None
        op = expr.value
        if op in ('/', '%') and b == 0:
            expr.token.error('division by zero in a constant expression')
        if op in ('<<', '>>') and not 0 <= b < expr.args[0].type.bits:
            expr.token.error('shift count outside the type width')
        div = lambda: (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
        r = {'+': lambda: a + b, '-': lambda: a - b, '*': lambda: a * b, '/': div, '%': lambda: a - div() * b,
             '&': lambda: a & b, '|': lambda: a | b, '^': lambda: a ^ b, '<<': lambda: a << b, '>>': lambda: a >> b,
             '==': lambda: int(a == b), '!=': lambda: int(a != b), '<': lambda: int(a < b), '>': lambda: int(a > b),
             '<=': lambda: int(a <= b), '>=': lambda: int(a >= b), '&&': lambda: int(bool(a) and bool(b)),
             '||': lambda: int(bool(a) or bool(b)), ',': lambda: b}[op]()
        return T.convert_integer(r, expr.type)
    return None


def cast(expr, type_, explicit=False):
    source = expr.type.decay()
    if source == type_:
        return decay(expr)
    ok = source.kind == type_.kind == 'int' or T.compatible(source, type_)
    if type_.kind == 'ptr':
        ok |= source.kind == 'ptr' and (explicit or source.base.kind == 'void' or type_.base.kind == 'void' or T.compatible(source.base, type_.base))
        ok |= source.kind == 'int' and (explicit or const(expr) == 0)
    if explicit:
        ok |= type_.kind == 'void' or source.scalar and type_.scalar
    if not ok:
        expr.token.error('cannot convert %s to %s' % (source, type_))
    return Expr('cast', type_, expr.token, (decay(expr),))


def decay(expr):
    type_ = expr.type.decay()
    return Expr('decay', type_, expr.token, (expr,)) if expr.type.kind in ('array', 'function') else expr


_STORAGE = {'static', 'extern', 'typedef', 'inline', '__inline__'}
_QUALIFIERS = {'const', 'volatile', 'restrict', '__restrict__'}
_BASIC = {'void', 'char', 'short', 'int', 'long', 'signed', 'unsigned'}
_ASSIGN = {'=', '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^='}
_PREC = {',': 1, **{s: 2 for s in _ASSIGN}, '?': 3, '||': 4, '&&': 5, '|': 6, '^': 7, '&': 8,
         '==': 9, '!=': 9, '<': 10, '>': 10, '<=': 10, '>=': 10, '<<': 11, '>>': 11,
         '+': 12, '-': 12, '*': 13, '/': 13, '%': 13}


class Parser:
    def __init__(self, tokens, path='<string>'):
        self.tokens = [*tokens, Token('eof', '', path)]
        self.i, self.path, self.records, self.declarations, self.definitions = 0, path, {}, [], []
        self.scopes = [{name: Decl(name, type_, Token('ident', name), frozenset({'typedef'})) for name, type_ in T.ALIASES.items()}]
        self.func, self.loops, self.labels, self.gotos = None, 0, set(), []
        for name, b in BUILTINS.items():
            type_ = T.function(b.result if isinstance(b.result, T.Type) else T.INT, [p or T.INT for p in b.params])
            self.scopes[0][name] = Decl(name, type_, Token('ident', name), builtin=b)

    @property
    def token(self):
        return self.tokens[self.i]

    def take(self, text):
        if self.token.text == text:
            token = self.token
            self.i += 1
            return token
        return None

    def expect(self, text):
        t = self.take(text)
        if t is None:
            self.token.error('expected %r, got %r' % (text, self.token.text))
        return t

    def ident(self):
        t = self.token
        if t.kind != 'ident':
            t.error('expected an identifier')
        self.i += 1
        return t

    def lookup(self, name):
        return next((s[name] for s in reversed(self.scopes) if name in s), None)

    def type_start(self, t=None):
        t = t or self.token
        decl = self.lookup(t.text)
        return t.text in _STORAGE | _QUALIFIERS | _BASIC | {'struct', 'enum', '__attribute__'} or decl is not None and 'typedef' in decl.storage

    def attributes(self):
        attrs = {}
        while self.take('__attribute__'):
            self.expect('(')
            self.expect('(')
            while not self.take(')'):
                name = self.ident()
                args = []
                if self.take('('):
                    if not self.take(')'):
                        while True:
                            args.append(self.expr(2))
                            if self.take(')'):
                                break
                            self.expect(',')
                if name.text not in ('aligned', 'noinline', 'always_inline', 'constructor'):
                    name.error('unsupported attribute ' + name.text)
                if name.text == 'aligned':
                    if len(args) != 1 or const(args[0]) is None:
                        name.error('aligned needs one constant')
                    value = const(args[0])
                    if value <= 0 or value & (value - 1):
                        name.error('alignment must be a positive power of two')
                    attrs[name.text] = value
                else:
                    if args:
                        name.error('attribute does not take arguments')
                    attrs[name.text] = True
                if self.token.text != ')':
                    self.expect(',')
            self.expect(')')
        return attrs

    def spec(self):
        start = self.token
        words, storage, quals, attrs, base = [], set(), set(), {}, None
        while True:
            word = self.token.text
            if word in _STORAGE:
                storage.add(word)
                self.i += 1
            elif word in _QUALIFIERS:
                quals.add(word)
                self.i += 1
            elif word == '__attribute__':
                attrs.update(self.attributes())
            elif word in _BASIC and base is None:
                words.append(word)
                self.i += 1
            elif word == 'struct' and base is None and not words:
                self.i += 1
                tag = self.ident() if self.token.kind == 'ident' else None
                name = tag.text if tag else '.anonymous%d' % len(self.records)
                record = self.records.setdefault(name, T.Record(name))
                base = T.Type('struct', base=record)
                if self.take('{'):
                    fields = []
                    while not self.take('}'):
                        fspec, fstorage, fattrs = self.spec()
                        if fstorage or fattrs:
                            self.token.error('storage or attributes on struct fields are unsupported')
                        while True:
                            field_, ftype, _ = self.declarator(fspec)
                            if field_ is None:
                                self.token.error('struct member needs a name')
                            fields.append((field_.text, ftype))
                            if not self.take(','):
                                break
                        self.expect(';')
                    try:
                        record.complete(fields)
                    except ValueError as e:
                        start.error(str(e))
                elif tag is None:
                    start.error('struct needs a tag or definition')
            elif word == 'enum' and base is None and not words:
                self.i += 1
                if self.token.kind == 'ident':
                    self.i += 1
                self.expect('{')
                value = 0
                while not self.take('}'):
                    name = self.ident()
                    if self.take('='):
                        e = self.expr(2)
                        value = const(e)
                        if value is None:
                            name.error('enum value is not constant')
                    d = Decl(name.text, T.INT, name, frozenset({'enum'}), init=value)
                    self.bind(d)
                    value += 1
                    if self.token.text != '}':
                        self.expect(',')
                base = T.INT
            elif base is None and not words:
                decl = self.lookup(word)
                if decl is None or 'typedef' not in decl.storage:
                    break
                base = decl.type
                self.i += 1
            else:
                break
        if base is None:
            if not words:
                start.error('expected a type')
            unsigned = 'unsigned' in words
            if 'signed' in words and unsigned or words.count('long') > 2:
                start.error('invalid integer type')
            if 'void' in words:
                if len(words) != 1:
                    start.error('invalid void type')
                base = T.VOID
            elif 'char' in words:
                base = T.UCHAR if unsigned else T.CHAR
            elif 'short' in words:
                base = T.USHORT if unsigned else T.SHORT
            elif words.count('long') == 2:
                base = T.ULLONG if unsigned else T.LLONG
            elif 'long' in words:
                base = T.ULONG if unsigned else T.LONG
            else:
                base = T.UINT if unsigned else T.INT
        return base.qualify(*quals), frozenset(storage), attrs

    def declarator(self, base, abstract=False):
        def tree():
            pointers = []
            while self.take('*'):
                quals = []
                while self.token.text in _QUALIFIERS:
                    quals.append(self.token.text)
                    self.i += 1
                pointers.append(quals)
            if self.token.kind == 'ident' and not self.type_start():
                node = ('name', self.ident())
            elif self.token.text == '(' and (not abstract or not self.type_start(self.tokens[self.i + 1])):
                self.i += 1
                node = tree()
                self.expect(')')
            else:
                node = ('name', None)
            while True:
                if self.take('['):
                    count = None
                    if not self.take(']'):
                        e = self.expr(2)
                        count = const(e)
                        if count is None or count <= 0:
                            e.token.error('array bound must be a positive integer constant')
                        self.expect(']')
                    node = ('array', node, count)
                elif self.take('('):
                    params = []
                    if not self.take(')'):
                        if self.token.text == 'void' and self.tokens[self.i + 1].text == ')':
                            self.i += 2
                        else:
                            while True:
                                token = self.token
                                pbase, storage, attrs = self.spec()
                                name, type_, _ = self.declarator(pbase, abstract=True)
                                params.append(Decl(name.text if name else '', type_.decay(), name or token, storage, attrs))
                                if self.take(')'):
                                    break
                                self.expect(',')
                    node = ('function', node, params)
                else:
                    break
            for quals in reversed(pointers):
                node = ('ptr', node, quals)
            return node

        def resolve(node, type_, params=()):
            kind = node[0]
            if kind == 'name':
                return node[1], type_, list(params) if type_.kind == 'function' else []
            if kind == 'ptr':
                return resolve(node[1], T.pointer(type_).qualify(*node[2]), params)
            if kind == 'array':
                return resolve(node[1], T.array(type_, node[2]), params)
            return resolve(node[1], T.function(type_, [p.type for p in node[2]]), node[2])

        name, type_, params = resolve(tree(), base)
        if not abstract and name is None:
            self.token.error('declaration needs a name')
        return name, type_, params

    def typename(self):
        base, storage, attrs = self.spec()
        if storage or attrs:
            self.token.error('storage or attributes in a type name')
        name, type_, _ = self.declarator(base, abstract=True)
        if name is not None:
            name.error('unexpected identifier in type name')
        return type_

    def bind(self, d):
        scope = self.scopes[-1]
        old = scope.get(d.name)
        if old is not None:
            if len(self.scopes) != 1 or not T.compatible(old.type, d.type) or old.body is not None and d.body is not None:
                d.token.error('conflicting declaration of ' + d.name)
            if d.type.kind == 'array' and d.type.count is not None:
                old.type = d.type
            old.attrs.update(d.attrs)
            return old
        scope[d.name] = d
        return d

    def declaration(self, global_=False):
        base, storage, attrs = self.spec()
        out = []
        if self.take(';'):
            return out
        while True:
            name, type_, params = self.declarator(base)
            more = self.attributes()
            d = self.bind(Decl(name.text, type_, name, storage, attrs | more, params))
            if global_ and d not in self.declarations and 'typedef' not in storage:
                self.declarations.append(d)
            if self.token.text == '{' and type_.kind == 'function':
                if not global_ or d.body is not None:
                    name.error('unexpected function definition')
                d.params = params
                self.func, self.labels, self.gotos = d, set(), []
                self.scopes.append({})
                for p in params:
                    if not p.name:
                        p.token.error('function definition needs named parameters')
                    self.bind(p)
                d.body = self.block(new_scope=False)
                self.definitions.append(d)
                for g in self.gotos:
                    if g.text not in self.labels:
                        g.error('undefined label ' + g.text)
                self.scopes.pop()
                self.func = None
                return [d]
            if self.take('='):
                if d.init is not None:
                    name.error('duplicate initializer of ' + d.name)
                d.init = self.initializer()
                d.type = self.check_init(d.type, d.init)
                if global_ and d not in self.definitions:
                    self.definitions.append(d)
            elif global_ and type_.kind != 'function' and 'extern' not in storage and 'typedef' not in storage and d not in self.definitions:
                self.definitions.append(d)
            out.append(d)
            if not self.take(','):
                break
        self.expect(';')
        return out

    def initializer(self):
        token = self.token
        if not self.take('{'):
            return Init(token, expr=self.expr(2))
        entries = []
        while not self.take('}'):
            path = []
            while self.token.text in ('.', '['):
                if self.take('.'):
                    path.append(self.ident().text)
                else:
                    self.expect('[')
                    e = self.expr(2)
                    index = const(e)
                    if index is None or index < 0:
                        e.token.error('initializer index must be a nonnegative constant')
                    path.append(index)
                    self.expect(']')
            if path:
                self.expect('=')
            entries.append((tuple(path), self.initializer()))
            if self.token.text != '}':
                self.expect(',')
        return Init(token, entries=entries)

    def check_init(self, type_, init):
        if init.expr is not None:
            if type_.kind == 'array' and type_.base.bits == 8 and init.expr.kind == 'string':
                n = len(init.expr.value) + 1
                if type_.count is None:
                    type_ = T.array(type_.base, n)
                elif type_.count < n - 1:
                    init.token.error('string initializer is too long')
            else:
                init.expr = cast(init.expr, type_.plain)
        elif type_.kind in ('array', 'struct'):
            cursor, count = 0, type_.count or 0
            for path, value in init.entries:
                current = type_
                if not path:
                    if type_.kind == 'array':
                        path = (cursor,)
                    else:
                        if cursor >= len(type_.base.fields):
                            value.token.error('too many struct initializers')
                        path = (type_.base.fields[cursor][0],)
                for designator in path:
                    if current.kind == 'array' and isinstance(designator, int):
                        if current.count is not None and designator >= current.count:
                            value.token.error('initializer index outside array')
                        current = current.base
                    elif current.kind == 'struct' and isinstance(designator, str):
                        try:
                            current = current.base.member(designator)[1]
                        except ValueError as e:
                            value.token.error(str(e))
                    else:
                        value.token.error('designator does not match ' + str(current))
                self.check_init(current, value)
                if type_.kind == 'array':
                    cursor = path[0] + 1
                    count = max(count, cursor)
                else:
                    cursor = next(i for i, f in enumerate(type_.base.fields) if f[0] == path[0]) + 1
            if type_.kind == 'array' and type_.count is None:
                type_ = T.array(type_.base, count)
        elif len(init.entries) == 1 and not init.entries[0][0]:
            self.check_init(type_, init.entries[0][1])
        else:
            init.token.error('expected a scalar initializer')
        init.type = type_
        return type_

    def parse(self):
        while self.token.kind != 'eof':
            self.declaration(global_=True)
        return Unit(self.path, self.declarations, self.records, definitions=self.definitions)

    def block(self, new_scope=True):
        token = self.expect('{')
        if new_scope:
            self.scopes.append({})
        body = []
        while not self.take('}'):
            if self.token.kind == 'eof':
                self.token.error('unfinished block')
            body.append(self.stmt())
        if new_scope:
            self.scopes.pop()
        return Stmt('block', token, tuple(body))

    def condition(self):
        self.expect('(')
        e = self.expr()
        self.expect(')')
        if not e.type.decay().scalar:
            e.token.error('condition must be scalar')
        return e

    def loop_body(self):
        self.loops += 1
        s = self.stmt()
        self.loops -= 1
        return s

    def stmt(self):
        t = self.token
        if t.text == '{':
            return self.block()
        if self.take(';'):
            return Stmt('empty', t)
        if self.take('if'):
            cond, yes = self.condition(), self.stmt()
            no = self.stmt() if self.take('else') else None
            return Stmt('if', t, (cond, yes, no))
        if self.take('while'):
            return Stmt('while', t, (self.condition(), self.loop_body()))
        if self.take('do'):
            body = self.loop_body()
            self.expect('while')
            cond = self.condition()
            self.expect(';')
            return Stmt('do', t, (body, cond))
        if self.take('for'):
            self.scopes.append({})
            self.expect('(')
            if self.type_start():
                init = Stmt('decl', self.token, value=self.declaration())
            else:
                init = Stmt('expr', self.token, (self.expr(),)) if self.token.text != ';' else None
                self.expect(';')
            cond = self.expr() if self.token.text != ';' else None
            self.expect(';')
            step = self.expr() if self.token.text != ')' else None
            self.expect(')')
            body = self.loop_body()
            self.scopes.pop()
            return Stmt('for', t, (init, cond, step, body))
        if self.take('return'):
            value = self.expr() if self.token.text != ';' else None
            if self.func is None:
                t.error('return outside a function')
            result = self.func.type.base
            if value is not None:
                value = cast(value, result)
            elif result.kind != 'void':
                t.error('return needs a value')
            self.expect(';')
            return Stmt('return', t, (value,))
        if t.text in ('break', 'continue'):
            self.i += 1
            if not self.loops:
                t.error(t.text + ' outside a loop')
            self.expect(';')
            return Stmt(t.text, t)
        if self.take('goto'):
            label = self.ident()
            self.gotos.append(label)
            self.expect(';')
            return Stmt('goto', t, value=label.text)
        if t.kind == 'ident' and self.tokens[self.i + 1].text == ':':
            self.i += 2
            if t.text in self.labels:
                t.error('duplicate label ' + t.text)
            self.labels.add(t.text)
            return Stmt('label', t, (self.stmt(),), t.text)
        if self.take('_Pragma'):
            self.expect('(')
            text = self.string()
            self.expect(')')
            return Stmt('pragma', t, value=text.value.decode())
        if t.text in ('__asm__', 'asm'):
            return self.asm()
        if self.type_start():
            return Stmt('decl', t, value=self.declaration())
        e = self.expr()
        self.expect(';')
        return Stmt('expr', t, (e,))

    def asm(self):
        t = self.token
        self.i += 1
        volatile = bool(self.take('volatile') or self.take('__volatile__'))
        self.expect('(')
        code = self.string().value.decode()
        if code not in ('', 'cpuid', 'xgetbv', 'rep movsb', 'rep stosb', 'sfence'):
            t.error('unsupported asm operation ' + repr(code))
        groups = []
        while self.take(':'):
            operands = []
            while self.token.text not in (':', ')'):
                constraint = self.string().value.decode()
                if self.take('('):
                    e = self.expr()
                    self.expect(')')
                    if not groups and not e.lvalue:
                        e.token.error('asm output needs an lvalue')
                    operands.append((constraint, e))
                else:
                    operands.append((constraint, None))
                if self.token.text not in (':', ')'):
                    self.expect(',')
            groups.append(operands)
        self.expect(')')
        self.expect(';')
        if len(groups) > 3:
            t.error('asm supports outputs, inputs and clobbers')
        return Stmt('asm', t, value=(code, volatile, groups))

    def string(self):
        t, data = self.token, bytearray()
        if t.kind != 'string':
            t.error('expected string literal')
        while self.token.kind == 'string':
            data += literal(self.token)
            self.i += 1
        return Expr('string', T.array(T.CHAR, len(data) + 1), t, value=bytes(data))

    def expr(self, prec=1):
        t = self.token
        self.i += 1
        if t.kind == 'number':
            value = parse_integer(t.text)
            lhs = Expr('int', T.integer_type(t.text, value), t, value=value)
        elif t.kind == 'char':
            data = literal(t)
            if len(data) != 1:
                t.error('expected a one-byte character literal')
            lhs = Expr('int', T.INT, t, value=data[0])
        elif t.kind == 'string':
            self.i -= 1
            lhs = self.string()
        elif t.text == 'sizeof':
            if self.token.text == '(' and self.type_start(self.tokens[self.i + 1]):
                self.i += 1
                type_ = self.typename()
                self.expect(')')
            else:
                type_ = self.expr(14).type
            try:
                size = type_.size
            except ValueError as e:
                t.error(str(e))
            lhs = Expr('int', T.ULONG, t, value=size)
        elif t.text == '(':
            if self.type_start():
                type_ = self.typename()
                self.expect(')')
                lhs = cast(self.expr(14), type_, explicit=True)
            else:
                lhs = self.expr()
                self.expect(')')
        elif t.text in ('+', '-', '~', '!', '*', '&', '++', '--'):
            value = self.expr(14)
            if t.text == '&':
                if not value.lvalue and value.type.kind != 'function':
                    t.error('address needs an lvalue or function')
                lhs = Expr('address', T.pointer(value.type), t, (value,))
            elif t.text == '*':
                value = decay(value)
                if value.type.kind != 'ptr':
                    t.error('dereference needs a pointer')
                lhs = Expr('deref', value.type.base, t, (value,))
            elif t.text in ('++', '--'):
                self.modifiable(value)
                if not value.type.scalar:
                    t.error('increment needs an integer or pointer')
                lhs = Expr('preinc', value.type.plain, t, (value,), 1 if t.text == '++' else -1)
            elif t.text == '!':
                value = decay(value)
                if not value.type.scalar:
                    t.error('! needs a scalar')
                lhs = Expr('unary', T.INT, t, (value,), '!')
            else:
                if value.type.kind != 'int':
                    t.error('unary arithmetic needs an integer')
                type_ = T.promote(value.type)
                lhs = Expr('unary', type_, t, (cast(value, type_),), t.text)
        elif t.kind == 'ident':
            decl = self.lookup(t.text)
            if decl is None or 'typedef' in decl.storage:
                t.error('undeclared value ' + t.text)
            lhs = Expr('int', T.INT, t, value=decl.init) if 'enum' in decl.storage else Expr('var', decl.type, t, value=decl)
        else:
            t.error('expected expression, got ' + repr(t.text))
        while True:
            op = self.token
            if op.text == '[':
                self.i += 1
                idx = self.expr()
                self.expect(']')
                base = decay(lhs)
                if base.type.kind != 'ptr' or idx.type.kind != 'int':
                    op.error('index needs a pointer and integer')
                lhs = Expr('index', base.type.base, op, (base, cast(idx, T.promote(idx.type))))
            elif op.text == '(':
                self.i += 1
                args = []
                if not self.take(')'):
                    while True:
                        args.append(self.expr(2))
                        if self.take(')'):
                            break
                        self.expect(',')
                callee = decay(lhs)
                if callee.type.kind != 'ptr' or callee.type.base.kind != 'function':
                    op.error('call needs a function')
                builtin = lhs.value.builtin if lhs.kind == 'var' else None
                params = builtin.params if builtin else callee.type.base.params
                if len(args) != len(params):
                    op.error('expected %d arguments, got %d' % (len(params), len(args)))
                args = [cast(a, p) if p is not None else decay(a) for a, p in zip(args, params)]
                if builtin:
                    for i, lo, hi in builtin.immediates:
                        value = const(args[i])
                        if value is None or not lo <= value <= hi:
                            args[i].token.error('intrinsic argument %d must be constant in [%d, %d]' % (i + 1, lo, hi))
                    result = args[0].type if builtin.result == 'arg0' else builtin.result
                else:
                    result = callee.type.base.base
                lhs = Expr('call', result, op, (callee, *args), builtin)
            elif op.text in ('.', '->'):
                self.i += 1
                name = self.ident()
                if op.text == '->':
                    base = decay(lhs)
                    if base.type.kind != 'ptr':
                        op.error('-> needs a pointer')
                    lhs = Expr('deref', base.type.base, op, (base,))
                if lhs.type.kind != 'struct':
                    op.error('member access needs a struct')
                try:
                    member = lhs.type.base.member(name.text)
                except ValueError as e:
                    name.error(str(e))
                lhs = Expr('member', member[1].qualify(*lhs.type.qualifiers), op, (lhs,), member)
            elif op.text in ('++', '--'):
                self.i += 1
                self.modifiable(lhs)
                if not lhs.type.scalar:
                    op.error('increment needs an integer or pointer')
                lhs = Expr('postinc', lhs.type.plain, op, (lhs,), 1 if op.text == '++' else -1)
            elif _PREC.get(op.text, -1) >= prec:
                self.i += 1
                if op.text == '?':
                    yes = self.expr()
                    self.expect(':')
                    no = self.expr(3)
                    if not lhs.type.decay().scalar:
                        lhs.token.error('conditional needs a scalar condition')
                    a, b = yes.type.decay(), no.type.decay()
                    if a.kind == b.kind == 'int':
                        type_ = T.arithmetic(a, b)
                    elif a == b or a.kind == 'ptr' and (b.kind == 'ptr' or const(no) == 0):
                        type_ = a
                    elif b.kind == 'ptr' and const(yes) == 0:
                        type_ = b
                    else:
                        op.error('incompatible conditional arms: %s and %s' % (a, b))
                    lhs = Expr('conditional', type_, op, (lhs, cast(yes, type_), cast(no, type_)))
                else:
                    rhs = self.expr(_PREC[op.text] + (0 if op.text in _ASSIGN else 1))
                    lhs = self.binary(op, lhs, rhs)
            else:
                break
        return lhs

    def modifiable(self, e):
        if not e.lvalue or 'const' in e.type.qualifiers or e.type.kind == 'array':
            e.token.error('expected a modifiable lvalue')

    def binary(self, op, lhs, rhs):
        if op.text in _ASSIGN:
            self.modifiable(lhs)
            if op.text != '=':
                rhs = self.binary(replace(op, text=op.text[:-1]), lhs, rhs)
                return Expr('compound_assign', lhs.type.plain, op, (lhs, cast(rhs, lhs.type.plain)))
            return Expr('assign', lhs.type.plain, op, (lhs, cast(rhs, lhs.type.plain)))
        lhs, rhs = decay(lhs), decay(rhs)
        a, b, result = lhs.type.plain, rhs.type.plain, None
        if op.text == ',':
            result = b
        elif op.text in ('&&', '||'):
            if not a.scalar or not b.scalar:
                op.error('logical operands must be scalar')
            result = T.INT
        elif op.text in ('+', '-') and (a.kind == 'ptr' or b.kind == 'ptr'):
            if a.kind == 'ptr' and b.kind == 'int':
                result, rhs = a, cast(rhs, T.LONG)
            elif op.text == '+' and a.kind == 'int' and b.kind == 'ptr':
                result, lhs = b, cast(lhs, T.LONG)
            elif op.text == '-' and a.kind == b.kind == 'ptr' and T.compatible(a.base, b.base):
                result = T.LONG
            else:
                op.error('invalid pointer arithmetic')
        elif op.text in ('==', '!=', '<', '>', '<=', '>=') and (a.kind == 'ptr' or b.kind == 'ptr'):
            if a.kind == 'ptr':
                rhs = cast(rhs, a)
            else:
                lhs = cast(lhs, b)
            result = T.INT
        else:
            try:
                type_ = T.arithmetic(a, b)
            except ValueError as e:
                op.error(str(e))
            if op.text in ('<<', '>>'):
                result = T.promote(a)
                lhs, rhs = cast(lhs, result), cast(rhs, T.promote(b))
            else:
                lhs, rhs = cast(lhs, type_), cast(rhs, type_)
                result = T.INT if op.text in ('==', '!=', '<', '>', '<=', '>=') else type_
        return Expr('binary', result, op, (lhs, rhs), op.text)
