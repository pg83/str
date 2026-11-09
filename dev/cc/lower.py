"""Typed C AST to explicit control flow and SSA values."""
from . import types as T
from .front import const
from .ir import Function, Term, promote
from .loops import fixed_for, contains_loop


class Lower:
    def __init__(self, decl):
        self.fn = Function(decl)
        self.block = self.fn.entry
        self.locals, self.labels, self.loops, self.saved_values = {}, {}, [], {}
        self.no_unroll = False
        for p, v in zip(decl.params, self.fn.params):
            self.store(self.local(p), v)

    def emit(self, kind, type_=None, args=(), **attrs):
        op = self.fn.make(kind, [type_] if type_ is not None and type_.kind != 'void' else [], args, attrs)
        self.block.ops.append(op)
        return op.results[0] if op.results else None

    def local(self, d):
        if d not in self.locals:
            op = self.fn.make('alloca', [T.pointer(d.type)], attrs={'name': d.name, 'size': d.type.size,
                              'alignment': max(d.type.alignment, d.attrs.get('aligned', 1))}, source=d.token)
            self.fn.allocas.append(op)
            self.locals[d] = op.results[0]
        return self.locals[d]

    def jump(self, target):
        if self.block.term is None:
            self.block.term = Term('jmp', targets=(target,))

    def boolean(self, value):
        return self.emit('ne', T.INT, [value, self.fn.constant(value.type, 0)])

    def branch(self, value, yes, no):
        self.block.term = Term('br', (value,), (yes, no))

    def offset(self, address, value):
        if value == 0:
            return address
        return self.emit('ptradd', address.type, [address, self.fn.constant(T.LONG, value)])

    def load(self, address, type_, volatile=False):
        if type_.kind in ('array', 'struct', 'function'):
            return address
        return self.emit('load', type_, [address], memory='read', width=type_.size, alignment=1, volatile=volatile)

    def store(self, address, value, volatile=False):
        if address.type.base.kind == 'struct':
            self.emit('copy', args=[address, value, self.fn.constant(T.ULONG, address.type.base.size)], memory='readwrite')
        else:
            self.emit('store', args=[address, value], memory='write', width=value.type.size, alignment=1, volatile=volatile)

    def address(self, e):
        if e.kind == 'var':
            d = e.value
            if d in self.locals:
                return self.locals[d]
            if d.type.kind != 'function' and d not in self.fn.decl.params and 'extern' not in d.storage:
                # Global objects are identified before lowering by lower_unit.
                if not getattr(d, 'global_', False):
                    return self.local(d)
            return self.emit('symbol', T.pointer(d.type), name=d.name, declaration=d)
        if e.kind == 'deref':
            return self.expr(e.args[0])
        if e.kind == 'index':
            base, idx = map(self.expr, e.args)
            scale = base.type.base.size
            idx = self.convert(idx, T.LONG)
            if scale != 1:
                idx = self.emit('mul', T.LONG, [idx, self.fn.constant(T.LONG, scale)])
            return self.emit('ptradd', T.pointer(e.type), [base, idx])
        if e.kind == 'member':
            address = self.offset(self.address(e.args[0]), e.value[2])
            return self.convert(address, T.pointer(e.type))
        if e.kind == 'string':
            return self.emit('string', T.pointer(e.type), data=e.value + b'\0')
        e.token.error('cannot take address of ' + e.kind)

    def convert(self, value, type_):
        return value if value.type == type_.plain else self.emit('cast', type_.plain, [value])

    def expr(self, e):
        if e is None:
            return None
        if id(e) in self.saved_values:
            return self.saved_values[id(e)]
        constant = const(e)
        if constant is not None:
            return self.fn.constant(e.type, constant)
        kind = e.kind
        if kind in ('var', 'deref', 'index', 'member'):
            return self.load(self.address(e), e.type, 'volatile' in e.type.qualifiers)
        if kind == 'string':
            return self.address(e)
        if kind == 'address':
            return self.address(e.args[0])
        if kind == 'decay':
            return self.convert(self.address(e.args[0]), e.type)
        if kind == 'cast':
            value = self.expr(e.args[0])
            return self.convert(value, e.type) if e.type.kind != 'void' else None
        if kind == 'assign':
            addr, value = self.address(e.args[0]), self.expr(e.args[1])
            self.store(addr, value, 'volatile' in e.args[0].type.qualifiers)
            return value
        if kind == 'compound_assign':
            left = e.args[0]
            address = self.address(left)
            self.saved_values[id(left)] = self.load(address, left.type, 'volatile' in left.type.qualifiers)
            value = self.expr(e.args[1])
            self.saved_values.pop(id(left))
            self.store(address, value, 'volatile' in left.type.qualifiers)
            return value
        if kind in ('preinc', 'postinc'):
            addr = self.address(e.args[0])
            old = self.load(addr, e.type)
            step = e.value * (e.type.base.size if e.type.kind == 'ptr' else 1)
            new = self.emit('ptradd' if e.type.kind == 'ptr' else 'add', e.type, [old, self.fn.constant(T.LONG if e.type.kind == 'ptr' else e.type, step)])
            self.store(addr, new)
            return old if kind == 'postinc' else new
        if kind == 'unary':
            value = self.expr(e.args[0])
            if e.value == '+':
                return value
            if e.value == '!':
                return self.emit('eq', T.INT, [value, self.fn.constant(value.type, 0)])
            return self.emit('neg' if e.value == '-' else 'not', e.type, [value])
        if kind == 'binary':
            if e.value == ',':
                self.expr(e.args[0])
                return self.expr(e.args[1])
            if e.value in ('&&', '||'):
                lhs = self.expr(e.args[0])
                left, right, done = self.block, self.fn.block('logical.rhs'), self.fn.block('logical.end')
                initial = self.fn.constant(T.INT, int(e.value == '||'))
                self.branch(lhs, right if e.value == '&&' else done, done if e.value == '&&' else right)
                self.block = right
                rhs = self.boolean(self.expr(e.args[1]))
                right_end = self.block
                self.jump(done)
                self.block = done
                return self.emit('phi', T.INT, [initial, rhs], blocks=[left, right_end])
            lhs, rhs = map(self.expr, e.args)
            if e.value in ('+', '-') and (lhs.type.kind == 'ptr' or rhs.type.kind == 'ptr'):
                if lhs.type.kind == rhs.type.kind == 'ptr':
                    value = self.emit('sub', T.LONG, [self.convert(lhs, T.LONG), self.convert(rhs, T.LONG)])
                    scale = lhs.type.base.size
                    return self.emit('div', T.LONG, [value, self.fn.constant(T.LONG, scale)]) if scale != 1 else value
                base, offset = (lhs, rhs) if lhs.type.kind == 'ptr' else (rhs, lhs)
                scale = base.type.base.size * (-1 if e.value == '-' else 1)
                if scale != 1:
                    offset = self.emit('mul', T.LONG, [offset, self.fn.constant(T.LONG, scale)])
                return self.emit('ptradd', e.type, [base, offset])
            op = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '%': 'rem', '&': 'and', '|': 'or', '^': 'xor',
                  '<<': 'shl', '>>': 'shr', '==': 'eq', '!=': 'ne', '<': 'lt', '>': 'gt', '<=': 'le', '>=': 'ge'}[e.value]
            return self.emit(op, e.type, [lhs, rhs])
        if kind == 'conditional':
            condition = self.expr(e.args[0])
            yes, no, done = self.fn.block('select.yes'), self.fn.block('select.no'), self.fn.block('select.end')
            self.branch(condition, yes, no)
            self.block = yes
            a = self.expr(e.args[1])
            a_end = self.block
            self.jump(done)
            self.block = no
            b = self.expr(e.args[2])
            b_end = self.block
            self.jump(done)
            self.block = done
            return self.emit('phi', e.type, [a, b], blocks=[a_end, b_end]) if e.type.kind != 'void' else None
        if kind == 'call':
            callee = self.expr(e.args[0])
            args = [self.expr(a) for a in e.args[1:]]
            if e.value is None:
                return self.emit('call', e.type, [callee, *args], memory='readwrite')
            builtin = e.value
            if builtin.op == 'copy':
                n = self.fn.defs[args[2]].attrs.get('value')
                if n in (1, 2, 4, 8, 16, 32, 64):
                    type_ = {1: T.UCHAR, 2: T.USHORT, 4: T.UINT, 8: T.ULONG, 16: T.V128, 32: T.V256, 64: T.V512}[n]
                    value = self.load(args[1], type_)
                    self.store(args[0], value)
                else:
                    self.emit('copy', args=args, memory='readwrite')
                return args[0]
            return self.emit('intrinsic', e.type, args, builtin=builtin, memory=builtin.memory, width=builtin.width,
                             alignment=builtin.alignment, mask=builtin.mask)
        e.token.error('unsupported lowering of ' + kind)

    def initialize(self, address, type_, init):
        if init.expr is not None:
            if type_.kind == 'array' and init.expr.kind == 'string':
                data = (init.expr.value + b'\0')[:type_.count]
                for i, byte in enumerate(data):
                    self.store(self.convert(self.offset(address, i), T.pointer(type_.base)), self.fn.constant(type_.base, byte))
            else:
                self.store(address, self.expr(init.expr))
            return
        if type_.kind not in ('struct', 'array'):
            self.initialize(address, type_, init.entries[0][1])
            return
        # C fills every omitted member with zero, including local aggregates.
        self.emit('fill', args=[address, self.fn.constant(T.UCHAR, 0), self.fn.constant(T.ULONG, type_.size)], memory='write')
        cursor = 0
        for path, value in init.entries:
            if not path:
                path = (cursor,) if type_.kind == 'array' else (type_.base.fields[cursor][0],)
            target, current = address, type_
            for designator in path:
                if current.kind == 'array':
                    offset, current = designator * current.base.size, current.base
                else:
                    _, current, offset = current.base.member(designator)
                target = self.convert(self.offset(target, offset), T.pointer(current))
            self.initialize(target, current, value)
            cursor = path[0] + 1 if type_.kind == 'array' else next(i for i, f in enumerate(type_.base.fields) if f[0] == path[0]) + 1

    def stmt(self, s):
        if s is None:
            return
        if self.block.term is not None and s.kind != 'label':
            self.block = self.fn.block('dead')
        kind = s.kind
        if kind == 'block':
            for child in s.args:
                self.stmt(child)
        elif kind == 'decl':
            for d in s.value:
                if 'typedef' in d.storage:
                    continue
                address = self.local(d)
                if d.init is not None:
                    self.initialize(address, d.type, d.init)
        elif kind == 'expr':
            self.expr(s.args[0])
        elif kind == 'return':
            v = self.expr(s.args[0])
            self.block.term = Term('ret', (v,) if v is not None else (), source=s.token)
        elif kind == 'if':
            cond, a, b = s.args
            v = self.expr(cond)
            yes, no, done = self.fn.block('if.yes'), self.fn.block('if.no'), self.fn.block('if.end')
            self.branch(v, yes, no)
            self.block = yes
            self.stmt(a)
            self.jump(done)
            self.block = no
            self.stmt(b)
            self.jump(done)
            self.block = done
        elif kind in ('while', 'do', 'for'):
            count = fixed_for(s) if kind == 'for' and not self.no_unroll else None
            defer_unroll = count is not None and count > 1 and contains_loop(s.args[3])
            disabled = self.no_unroll
            self.no_unroll = False
            if count is not None and not defer_unroll:
                init, _, step, body = s.args
                self.stmt(init)
                done = self.fn.block('unroll.end')
                for i in range(count):
                    advance = self.fn.block('unroll.step')
                    self.loops.append((done, advance))
                    self.stmt(body)
                    self.jump(advance)
                    self.block = advance
                    self.expr(step)
                    self.loops.pop()
                self.jump(done)
                self.block = done
                return
            if kind == 'for':
                init, cond, step, body = s.args
                self.stmt(init)
            elif kind == 'while':
                cond, body = s.args
                step = None
            else:
                body, cond = s.args
                step = None
            head, loop, advance, done = (self.fn.block('loop.' + n) for n in ('head', 'body', 'step', 'end'))
            head.unroll_disabled = disabled
            head.fixed_trip_count = count if defer_unroll else None
            self.jump(loop if kind == 'do' else head)
            self.block = head
            if cond is not None:
                self.branch(self.expr(cond), loop, done)
            else:
                self.jump(loop)
            self.loops.append((done, advance))
            self.block = loop
            self.stmt(body)
            self.jump(advance)
            self.block = advance
            self.expr(step)
            self.jump(head)
            self.loops.pop()
            self.block = done
        elif kind in ('break', 'continue'):
            self.jump(self.loops[-1][kind == 'continue'])
        elif kind == 'goto':
            if s.value not in self.labels:
                self.labels[s.value] = self.fn.block('label.' + s.value)
            self.jump(self.labels[s.value])
        elif kind == 'label':
            if s.value not in self.labels:
                self.labels[s.value] = self.fn.block('label.' + s.value)
            self.jump(self.labels[s.value])
            self.block = self.labels[s.value]
            self.stmt(s.args[0])
        elif kind == 'asm':
            code, volatile, groups = s.value
            outputs = groups[0] if groups else []
            inputs = groups[1] if len(groups) > 1 else []
            args = [self.expr(e) for constraint, e in outputs if '+' in constraint] + [self.expr(e) for _, e in inputs]
            op = self.fn.make('asm', [e.type.plain for _, e in outputs], args,
                              {'code': code, 'volatile': volatile, 'constraints': [[c for c, _ in g] for g in groups],
                               'memory': 'readwrite' if code.startswith('rep ') or any(c == 'memory' for g in groups for c, _ in g) else ''}, s.token)
            self.block.ops.append(op)
            for (_, e), value in zip(outputs, op.results):
                self.store(self.address(e), value)
        elif kind == 'pragma':
            self.no_unroll = 'unroll' in s.value
        elif kind != 'empty':
            s.token.error('unsupported statement ' + kind)

    def run(self):
        self.stmt(self.fn.decl.body)
        if self.block.term is None:
            self.block.term = Term('ret' if self.fn.type.base.kind == 'void' else 'unreachable')
        self.fn.blocks = self.fn.reachable()
        promote(self.fn)
        self.fn.verify()
        return self.fn


def lower_unit(unit):
    for d in unit.declarations:
        d.global_ = True
    from objfile import Object
    from .data import emit_data
    data = Object()
    emit_data(data, unit)
    immutable = {}
    for symbol in data.symbols:
        section = data.sections.get(symbol.section)
        if symbol.type == 'STT_OBJECT' and section is not None and not section.flags & 1:
            immutable[symbol.name] = section.data, section.relocs, symbol.value, symbol.size
    functions = [Lower(d).run() for d in unit.functions]
    referenced = {rel.symbol for section in data.sections.values() for rel in section.relocs}
    for fn in functions:
        fn.immutable = immutable
        fn.global_address = fn.name in referenced or '.text.'+fn.name in referenced
    return functions
