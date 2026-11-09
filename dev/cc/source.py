"""Tokens, source locations and preprocessing for the library's C dialect.

System includes name language builtins; they never open platform headers.
Local includes and ordinary object/function macros are expanded here, with
token hide sets so recursive expansion has the same boundaries as C.
"""
import re
from dataclasses import dataclass, replace
from pathlib import Path


class Error(Exception):
    def __init__(self, token, message):
        self.token, self.message = token, message
        super().__init__('%s:%d:%d: %s' % (token.path, token.line, token.column, message))


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    path: str = '<builtin>'
    line: int = 1
    column: int = 1
    hide: frozenset = frozenset()

    def error(self, message):
        raise Error(self, message)


_TOKEN = re.compile(
    r'(?P<space>[^\S\n]+)|(?P<newline>\n)|(?P<splice>\\\n)|'
    r'(?P<comment>/\*[\s\S]*?\*/|//[^\n]*)|'
    r'(?P<ident>[A-Za-z_][A-Za-z_0-9]*)|'
    r'(?P<number>0[xX][0-9a-fA-F]+[uUlL]*|[0-9]+[uUlL]*)|'
    r'(?P<string>"(?:[^"\\\n]|\\[^\n])*")|'
    r"(?P<char>'(?:[^'\\\n]|\\[^\n])*')|"
    r'(?P<punct>>>=|<<=|\.\.\.|->|\+\+|--|&&|\|\||<=|>=|==|!=|<<|>>|'
    r'\+=|-=|\*=|/=|%=|&=|\|=|\^=|##|[{}\[\]();,:?~!+*/%&|^<>=.#\\-])'
)


def lex(text, path='<string>', newlines=False):
    tokens = []
    line, column, pos = 1, 1, 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            Token('', text[pos], str(path), line, column).error('unrecognized character %r' % text[pos])
        kind, value = m.lastgroup, m.group()
        if kind == 'newline' and newlines:
            tokens.append(Token(kind, value, str(path), line, column))
        elif kind not in ('space', 'newline', 'splice', 'comment'):
            tokens.append(Token(kind, value, str(path), line, column))
        elif kind == 'comment' and newlines:
            tokens.extend(Token('newline', '\n', str(path), line + n, 1) for n in range(value.count('\n')))
        count = value.count('\n')
        line += count
        column = len(value.rsplit('\n', 1)[-1]) + 1 if count else column + len(value)
        pos = m.end()
    return tokens


def integer(text):
    value = re.sub('[uUlL]+$', '', text)
    base = 16 if value.lower().startswith('0x') else 8 if len(value) > 1 and value.startswith('0') else 10
    return int(value, base)


def literal(token):
    """The bytes of a narrow C string or character literal."""
    text, out, i = token.text[1:-1], bytearray(), 0
    escapes = {'a': 7, 'b': 8, 'f': 12, 'n': 10, 'r': 13, 't': 9, 'v': 11, '\\': 92, "'": 39, '"': 34, '?': 63}
    while i < len(text):
        ch = text[i]
        i += 1
        if ch != '\\':
            out.extend(ch.encode('utf-8'))
            continue
        if i == len(text):
            token.error('unfinished escape')
        ch = text[i]
        i += 1
        if ch in escapes:
            out.append(escapes[ch])
        elif ch in '01234567':
            end = i
            while end < min(i + 2, len(text)) and text[end] in '01234567':
                end += 1
            value = int(ch + text[i:end], 8)
            if value > 255:
                token.error('escape does not fit in one byte')
            out.append(value)
            i = end
        elif ch == 'x':
            end = i
            while end < len(text) and text[end] in '0123456789abcdefABCDEF':
                end += 1
            if end == i:
                token.error('hex escape needs a digit')
            value = int(text[i:end], 16)
            if value > 255:
                token.error('escape does not fit in one byte')
            out.append(value)
            i = end
        else:
            token.error('unsupported escape \\' + ch)
    return bytes(out)


@dataclass
class Macro:
    params: tuple | None
    body: list


class Preprocessor:
    headers = {'stddef.h', 'stdint.h', 'immintrin.h', 'emmintrin.h'}

    def __init__(self, defines=None, include_dirs=()):
        self.macros, self.dependencies = {}, set()
        self.include_dirs = [Path(p).resolve() for p in include_dirs]
        builtins = {'__STR_CC__': '1', '__x86_64__': '1', 'SIZE_MAX': '0xffffffffffffffffUL',
                    'PTRDIFF_MAX': '0x7fffffffffffffffL', 'NULL': '((void *)0)',
                    '_SIDD_UBYTE_OPS': '0', '_SIDD_CMP_EQUAL_ANY': '0', '_SIDD_NEGATIVE_POLARITY': '16',
                    '_SIDD_POSITIVE_POLARITY': '0', '_SIDD_LEAST_SIGNIFICANT': '0', '_SIDD_BIT_MASK': '0'}
        for name, value in (builtins | (defines or {})).items():
            self.macros[name] = Macro(None, lex(str(value)))
        self.macros['__has_builtin'] = Macro(('x',), lex('1'))

    def expand(self, tokens):
        tokens, out, i, steps = list(tokens), [], 0, 0
        while i < len(tokens):
            t = tokens[i]
            macro = self.macros.get(t.text) if t.kind == 'ident' and t.text not in t.hide else None
            if macro is None:
                out.append(t)
                i += 1
                continue
            end, args = i + 1, {}
            if macro.params is not None:
                if end == len(tokens) or tokens[end].text != '(':
                    out.append(t)
                    i += 1
                    continue
                groups, group, depth, end = [], [], 0, end + 1
                while end < len(tokens):
                    v = tokens[end].text
                    if v == ')' and depth == 0:
                        if group or groups or macro.params:
                            groups.append(group)
                        break
                    if v == ',' and depth == 0:
                        groups.append(group)
                        group = []
                    else:
                        group.append(tokens[end])
                        depth += (v == '(') - (v == ')')
                    end += 1
                if end == len(tokens):
                    t.error('unfinished macro invocation')
                end += 1
                if len(groups) != len(macro.params):
                    t.error('%s expects %d macro arguments, got %d' % (t.text, len(macro.params), len(groups)))
                args = {p: self.expand(g) for p, g in zip(macro.params, groups)}
            body = []
            for b in macro.body:
                if b.text in ('#', '##'):
                    b.error('macro stringification and token pasting are not supported')
                for a in args.get(b.text, [b]):
                    body.append(replace(a, path=t.path, line=t.line, column=t.column, hide=a.hide | t.hide | {t.text}))
            tokens[i:end] = body
            steps += 1
            if steps > 100000:
                t.error('macro expansion limit exceeded')
        return out

    def condition(self, tokens):
        out, i = [], 0
        while i < len(tokens):
            t = tokens[i]
            if t.text != 'defined':
                out.append(t)
                i += 1
                continue
            i += 1
            paren = i < len(tokens) and tokens[i].text == '('
            i += paren
            if i == len(tokens) or tokens[i].kind != 'ident':
                t.error('defined needs a macro name')
            out.append(replace(t, kind='number', text=str(int(tokens[i].text in self.macros))))
            i += 1
            if paren:
                if i == len(tokens) or tokens[i].text != ')':
                    t.error('expected ) after defined')
                i += 1
        out = [replace(t, kind='number', text='0') if t.kind == 'ident' else t for t in self.expand(out)]
        return Constant(out).parse() != 0

    def process(self, path):
        path = Path(path).resolve()
        self.dependencies.add(path)
        return self.text(path.read_text(), str(path), path.parent)

    def text(self, text, path='<string>', directory=None):
        tokens = lex(text, path, newlines=True)
        lines, current = [], []
        for t in tokens + [Token('newline', '\n', str(path))]:
            if t.kind == 'newline':
                lines.append(current)
                current = []
            else:
                current.append(t)
        out, pending, stack, active = [], [], [], True

        def flush():
            out.extend(self.expand(pending))
            pending.clear()

        for line in lines:
            if not line or line[0].text != '#':
                if active:
                    pending.extend(line)
                continue
            flush()
            if len(line) < 2:
                continue
            directive, args = line[1], line[2:]
            name = directive.text
            if name in ('if', 'ifdef', 'ifndef'):
                if name != 'if' and (len(args) != 1 or args[0].kind != 'ident'):
                    directive.error('expected one macro name')
                yes = active and (self.condition(args) if name == 'if' else (args[0].text in self.macros) == (name == 'ifdef'))
                stack.append([active, yes, False])
                active = yes
            elif name in ('else', 'elif'):
                if not stack or stack[-1][2]:
                    directive.error('unexpected #' + name)
                parent, taken, _ = stack[-1]
                active = parent and not taken and (name == 'else' or self.condition(args))
                stack[-1] = [parent, taken or active, name == 'else']
            elif name == 'endif':
                if not stack:
                    directive.error('unexpected #endif')
                active = stack.pop()[0]
            elif not active:
                continue
            elif name == 'define':
                if not args or args[0].kind != 'ident':
                    directive.error('expected macro name')
                ident, params, k = args[0], None, 1
                if k < len(args) and args[k].text == '(' and args[k].line == ident.line and args[k].column == ident.column + len(ident.text):
                    params, k = [], k + 1
                    while k < len(args) and args[k].text != ')':
                        if args[k].kind != 'ident':
                            args[k].error('expected macro parameter')
                        params.append(args[k].text)
                        k += 1
                        if k < len(args) and args[k].text != ')':
                            if args[k].text != ',':
                                args[k].error('expected , in macro parameters')
                            k += 1
                    if k == len(args):
                        ident.error('unfinished macro parameters')
                    params, k = tuple(params), k + 1
                self.macros[ident.text] = Macro(params, args[k:])
            elif name == 'undef':
                if len(args) != 1:
                    directive.error('expected one macro name')
                self.macros.pop(args[0].text, None)
            elif name == 'include':
                args = self.expand(args)
                if len(args) == 1 and args[0].kind == 'string':
                    target = literal(args[0]).decode()
                    paths = ([Path(directory)] if directory is not None else []) + self.include_dirs
                    found = next((p / target for p in paths if (p / target).is_file()), None)
                    if found is None:
                        directive.error('local include not found: ' + target)
                    out.extend(self.process(found))
                elif args and args[0].text == '<' and args[-1].text == '>':
                    target = ''.join(t.text for t in args[1:-1])
                    if target not in self.headers:
                        directive.error('unsupported builtin header: ' + target)
                else:
                    directive.error('expected header name')
            elif name == 'pragma':
                if args and args[0].text != 'once':
                    directive.error('unsupported pragma')
            elif name == 'error':
                directive.error('#error ' + ' '.join(t.text for t in args))
            else:
                directive.error('unsupported directive #' + name)
        flush()
        if stack:
            Token('', '', str(path)).error('unterminated conditional directive')
        return out


_PREC = {'||': 1, '&&': 2, '|': 3, '^': 4, '&': 5, '==': 6, '!=': 6, '<': 7, '>': 7, '<=': 7, '>=': 7,
         '<<': 8, '>>': 8, '+': 9, '-': 9, '*': 10, '/': 10, '%': 10}


class Constant:
    """Integer preprocessor expressions, including lazy &&, || and ?: ."""
    def __init__(self, tokens):
        self.tokens, self.i = tokens + [Token('eof', '')], 0

    def take(self, text):
        if self.tokens[self.i].text == text:
            self.i += 1
            return True
        return False

    def parse(self):
        result = self.expr(0, True)
        if self.tokens[self.i].kind != 'eof':
            self.tokens[self.i].error('unexpected token in preprocessor expression')
        return result

    def expr(self, prec, evaluate):
        t = self.tokens[self.i]
        self.i += 1
        if t.text in ('+', '-', '!', '~'):
            v = self.expr(11, evaluate)
            lhs = {'+': lambda: v, '-': lambda: -v, '!': lambda: int(not v), '~': lambda: ~v}[t.text]()
        elif t.text == '(':
            lhs = self.expr(0, evaluate)
            if not self.take(')'):
                self.tokens[self.i].error('expected )')
        elif t.kind == 'number':
            lhs = integer(t.text)
        elif t.kind == 'char':
            value = literal(t)
            if len(value) != 1:
                t.error('expected a one-byte character constant')
            lhs = value[0]
        else:
            t.error('expected preprocessor expression')
        while _PREC.get(self.tokens[self.i].text, -1) >= prec:
            op = self.tokens[self.i]
            self.i += 1
            rhs = self.expr(_PREC[op.text] + 1, evaluate and not (op.text == '&&' and not lhs or op.text == '||' and lhs))
            if not evaluate:
                lhs = 0
                continue
            if op.text in ('/', '%') and rhs == 0:
                op.error('division by zero')
            f = {'+': lambda: lhs + rhs, '-': lambda: lhs - rhs, '*': lambda: lhs * rhs,
                 '/': lambda: (abs(lhs) // abs(rhs)) * (-1 if (lhs < 0) != (rhs < 0) else 1),
                 '%': lambda: lhs - ((abs(lhs) // abs(rhs)) * (-1 if (lhs < 0) != (rhs < 0) else 1)) * rhs,
                 '<<': lambda: lhs << rhs, '>>': lambda: lhs >> rhs, '&': lambda: lhs & rhs, '|': lambda: lhs | rhs,
                 '^': lambda: lhs ^ rhs, '==': lambda: int(lhs == rhs), '!=': lambda: int(lhs != rhs),
                 '<': lambda: int(lhs < rhs), '>': lambda: int(lhs > rhs), '<=': lambda: int(lhs <= rhs), '>=': lambda: int(lhs >= rhs),
                 '&&': lambda: int(bool(lhs) and bool(rhs)), '||': lambda: int(bool(lhs) or bool(rhs))}
            if op.text in ('&&', '||'):
                lhs = int(bool(lhs) and bool(rhs)) if op.text == '&&' else int(bool(lhs) or bool(rhs))
            else:
                lhs = f[op.text]()
        if prec == 0 and self.take('?'):
            yes = self.expr(0, evaluate and bool(lhs))
            if not self.take(':'):
                self.tokens[self.i].error('expected :')
            no = self.expr(0, evaluate and not lhs)
            lhs = yes if lhs else no
        return lhs
