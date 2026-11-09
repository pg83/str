"""The typed C AST of front.py to the IR of ir.py.

Locals are stack slots (alloca); ssa.py promotes them. Conditions are
lowered as jumps (&&, || and ! become control flow, a comparison a branch on
it), __builtin_expect becomes the branch's weights and __builtin_unpredictable
its 'unpredictable' mark. The C types decide the operations: a signed
operand's division, shift, widening and comparison are the signed ones.
Intrinsics are vops named by their builtin, except the plain loads and stores
of vectors, the bitwise ops on them and the constants, which are the IR's own.
"""
from . import ctype as C
from .front import const
from .ir import Function, Ty, I1, I8, I16, I32, I64, PTR, VOID, int_ty

LIKELY = (2000, 1)
UNLIKELY = (1, 2000)


def ir_ty(t):
    t = t.decay() if t.kind in ('array', 'function') else t
    if t.kind == 'int':
        return int_ty(t.bits)
    if t.kind == 'ptr':
        return PTR
    if t.kind == 'vector':
        return Ty('v', t.bits)
    if t.kind == 'void':
        return VOID
    raise ValueError('no IR type for ' + str(t))


def strip_casts(e):
    """The operand of value-preserving conversions (widening, same width)."""
    while e.kind == 'cast' and e.type.kind in ('int', 'ptr') and e.args[0].type.decay().kind in ('int', 'ptr') \
            and (e.type.kind == 'ptr' and e.args[0].type.decay().kind == 'ptr' or ir_ty(e.type).bits >= ir_ty(e.args[0].type.decay()).bits):
        e = e.args[0]
    return e


class Lower:
    def __init__(self, module, decl):
        self.module, self.decl = module, decl
        attrs = dict(decl.attrs)
        attrs['static'] = 'static' in decl.storage
        attrs['inline'] = 'inline' in decl.storage or '__inline__' in decl.storage
        self.fn = Function(decl.name, [ir_ty(p.type) for p in decl.params], ir_ty(decl.type.base), attrs)
        self.block = self.fn.entry
        self.locals, self.labels, self.loops, self.saved = {}, {}, [], {}
        self.no_unroll = False
        for p, v in zip(decl.params, self.fn.params):
            self.store(self.local(p), v, p.type)

    # ------------------------------------------------------------ helpers

    def emit(self, kind, args=(), ty=None, **attrs):
        op = self.fn.new(kind, args, ty, attrs)
        self.block.append(op)
        return op.result

    def const(self, ty, value):
        return self.fn.const(ty, value)

    def local(self, d):
        v = self.locals.get(d)
        if v is None:
            op = self.fn.new('alloca', (), PTR, {'size': d.type.size, 'align': max(d.type.alignment, d.attrs.get('aligned', 1)),
                                                 'name': d.name})
            entry = self.fn.entry
            at = next((i for i, o in enumerate(entry.ops) if o.kind != 'alloca'), len(entry.ops))
            entry.insert(at, op)
            v = self.locals[d] = op.result
        return v

    def jump(self, target):
        if self.block.term is None:
            self.block.set_term(self.fn.new('jmp'))
            self.block.term.targets = [target]

    def branch(self, cond, yes, no, weights=None, unpredictable=False):
        if yes is no:
            self.jump(yes)
            return
        op = self.fn.new('br', [cond])
        op.targets = [yes, no]
        if weights:
            op.attrs['weights'] = weights
        if unpredictable:
            op.attrs['unpredictable'] = True
        self.block.set_term(op)

    def new_block(self, name):
        return self.fn.block(name)

    def load(self, address, ctype, volatile=False):
        if ctype.kind in ('array', 'struct', 'function'):
            return address
        return self.emit('load', [address], ir_ty(ctype), align=ctype.alignment, volatile=volatile)

    def store(self, address, value, ctype, volatile=False):
        if ctype.kind == 'struct':
            self.emit('memcpy', [address, value, self.const(I64, ctype.size)], align=ctype.alignment)
        else:
            self.emit('store', [address, value], align=ctype.alignment, volatile=volatile)

    def convert(self, v, source, target):
        """C conversion of v from type source to type target."""
        source, target = source.decay(), target.decay() if target.kind != 'void' else target
        if target.kind == 'void':
            return None
        a, b = ir_ty(source), ir_ty(target)
        if a is b:
            return v
        if a.is_int and b.is_int:
            if b.bits < a.bits:
                return self.emit('trunc', [v], b)
            return self.emit('sext' if source.signed else 'zext', [v], b)
        if a.is_ptr and b.is_int:
            v = self.emit('p2i', [v], I64)
            return v if b.bits == 64 else self.emit('trunc', [v], b)
        if a.is_int and b.is_ptr:
            if a.bits < 64:
                v = self.emit('sext' if source.signed else 'zext', [v], I64)
            return self.emit('i2p', [v], PTR)
        if a.kind == b.kind == 'v' and a.bits == b.bits:
            return v
        raise ValueError('%s: cannot convert %s to %s' % (self.fn.name, source, target))

    def bool_value(self, cond):
        return self.emit('zext', [cond], I32)

    def ptradd(self, base, offset):
        if offset.const == 0:
            return base
        return self.emit('ptradd', [base, offset], PTR)

    def offset(self, base, n):
        return base if n == 0 else self.ptradd(base, self.const(I64, n))

    # ---------------------------------------------------------- addresses

    def address(self, e):
        k = e.kind
        if k == 'var':
            d = e.value
            if d in self.locals:
                return self.locals[d]
            if d.type.kind == 'function' or getattr(d, 'global_', False) or 'extern' in d.storage:
                return self.fn.sym(d.name)
            return self.local(d)
        if k == 'deref':
            return self.expr(e.args[0])
        if k == 'index':
            base = self.expr(e.args[0])
            index = self.convert(self.expr(e.args[1]), e.args[1].type, C.LONG)
            size = e.args[0].type.base.size
            if size != 1:
                index = self.emit('mul', [index, self.const(I64, size)], I64)
            return self.ptradd(base, index)
        if k == 'member':
            return self.offset(self.address(e.args[0]), e.value[2])
        if k == 'string':
            return self.fn.sym(self.module.string(e.value))
        e.token.error('cannot take the address of ' + k)

    # -------------------------------------------------------- expressions

    def expr(self, e):
        if e is None:
            return None
        if id(e) in self.saved:
            return self.saved[id(e)]
        k = e.kind
        if e.type.kind in ('int', 'ptr'):
            c = const(e)
            if c is not None:
                return self.const(ir_ty(e.type), c)
        if k in ('var', 'deref', 'index', 'member'):
            if k == 'var' and 'enum' in e.value.storage:
                return self.const(I32, e.value.init)
            return self.load(self.address(e), e.type, 'volatile' in e.type.qualifiers)
        if k == 'string':
            return self.address(e)
        if k == 'address':
            return self.address(e.args[0])
        if k == 'decay':
            return self.address(e.args[0])
        if k == 'cast':
            inner = e.args[0]
            v = self.expr(inner)
            return self.convert(v, inner.type, e.type) if e.type.kind != 'void' else None
        if k == 'assign':
            lhs = e.args[0]
            address, value = self.address(lhs), self.expr(e.args[1])
            self.store(address, value, lhs.type, 'volatile' in lhs.type.qualifiers)
            return value
        if k == 'compound_assign':
            lhs = e.args[0]
            address = self.address(lhs)
            self.saved[id(lhs)] = self.load(address, lhs.type, 'volatile' in lhs.type.qualifiers)
            value = self.expr(e.args[1])
            del self.saved[id(lhs)]
            self.store(address, value, lhs.type, 'volatile' in lhs.type.qualifiers)
            return value
        if k in ('preinc', 'postinc'):
            lhs = e.args[0]
            address = self.address(lhs)
            old = self.load(address, e.type)
            if e.type.kind == 'ptr':
                new = self.offset(old, e.value * e.type.base.size)
            else:
                new = self.emit('add', [old, self.const(ir_ty(e.type), e.value)], ir_ty(e.type))
            self.store(address, new, e.type)
            return old if k == 'postinc' else new
        if k == 'unary':
            op = e.value
            if op == '!':
                return self.bool_value(self.test(e.args[0], zero=True))
            v = self.expr(e.args[0])
            t = ir_ty(e.type)
            if op == '+':
                return v
            if op == '-':
                return self.emit('sub', [self.const(t, 0), v], t)
            return self.emit('xor', [v, self.const(t, -1)], t)
        if k == 'binary':
            return self.binary(e)
        if k == 'conditional':
            return self.conditional(e)
        if k == 'call':
            return self.call(e)
        e.token.error('cannot lower ' + k)

    def test(self, e, zero=False):
        """An i1: e != 0 (or e == 0)."""
        v = self.expr(e)
        t = e.type.decay()
        return self.emit('icmp', [v, self.const(ir_ty(t), 0)], I1, pred='eq' if zero else 'ne')

    def compare(self, e):
        """An i1 for a comparison expression."""
        a, b = e.args
        va, vb = self.expr(a), self.expr(b)
        t = a.type.decay()
        signed_ = t.kind == 'int' and t.signed
        pred = {'==': 'eq', '!=': 'ne'}.get(e.value)
        if pred is None:
            pred = ('s' if signed_ else 'u') + {'<': 'lt', '>': 'gt', '<=': 'le', '>=': 'ge'}[e.value]
        return self.emit('icmp', [va, vb], I1, pred=pred)

    def binary(self, e):
        op = e.value
        if op == ',':
            self.expr(e.args[0])
            return self.expr(e.args[1])
        if op in ('&&', '||', '==', '!=', '<', '>', '<=', '>='):
            if op in ('&&', '||'):
                yes, no, done = self.new_block('logic.true'), self.new_block('logic.false'), self.new_block('logic.end')
                self.cond(e, yes, no)
                phi = self.fn.phi(I1)
                for b, value in ((yes, 1), (no, 0)):
                    self.block = b
                    self.jump(done)
                    phi.add_incoming(b, self.const(I1, value))
                self.block = done
                done.append(phi)
                return self.bool_value(phi.result)
            return self.bool_value(self.compare(e))
        a, b = e.args
        ta, tb = a.type.decay(), b.type.decay()
        if op in ('+', '-') and (ta.kind == 'ptr' or tb.kind == 'ptr'):
            va, vb = self.expr(a), self.expr(b)
            if ta.kind == 'ptr' and tb.kind == 'ptr':
                d = self.emit('sub', [self.emit('p2i', [va], I64), self.emit('p2i', [vb], I64)], I64)
                size = ta.base.size
                return d if size == 1 else self.emit('sdiv', [d, self.const(I64, size)], I64, exact=True)
            base, base_t, offset, offset_t = (va, ta, vb, tb) if ta.kind == 'ptr' else (vb, tb, va, ta)
            offset = self.convert(offset, offset_t, C.LONG)
            size = base_t.base.size
            if op == '-':
                offset = self.emit('sub', [self.const(I64, 0), offset], I64)
            if size != 1:
                offset = self.emit('mul', [offset, self.const(I64, size)], I64)
            return self.ptradd(base, offset)
        t = ir_ty(e.type)
        va, vb = self.expr(a), self.expr(b)
        if op in ('<<', '>>'):
            vb = self.convert(vb, tb, e.type)
            kind = 'shl' if op == '<<' else ('ashr' if e.type.signed else 'lshr')
            return self.emit(kind, [va, vb], t)
        if op in ('/', '%'):
            kind = ('s' if e.type.signed else 'u') + ('div' if op == '/' else 'rem')
            return self.emit(kind, [va, vb], t)
        kind = {'+': 'add', '-': 'sub', '*': 'mul', '&': 'and', '|': 'or', '^': 'xor'}[op]
        return self.emit(kind, [va, vb], t)

    def conditional(self, e):
        yes, no, done = self.new_block('sel.true'), self.new_block('sel.false'), self.new_block('sel.end')
        self.cond(e.args[0], yes, no)
        values = []
        for b, arm in ((yes, e.args[1]), (no, e.args[2])):
            self.block = b
            v = self.expr(arm)
            values.append((self.block, v))
            self.jump(done)
        self.block = done
        if e.type.kind == 'void':
            return None
        phi = self.fn.phi(ir_ty(e.type))
        for b, v in values:
            phi.add_incoming(b, v)
        done.append(phi)
        return phi.result

    # ---------------------------------------------------------- conditions

    def cond(self, e, yes, no, weights=None, unpredictable=False):
        """Branch to yes when e is true, else to no; the current block ends here."""
        e0 = e
        e = strip_casts(e)
        k = e.kind
        if k == 'binary' and e.value == '&&':
            mid = self.new_block('and.rhs')
            self.cond(e.args[0], mid, no, weights=LIKELY if weights == LIKELY else None, unpredictable=unpredictable)
            self.block = mid
            self.cond(e.args[1], yes, no, weights, unpredictable)
            return
        if k == 'binary' and e.value == '||':
            mid = self.new_block('or.rhs')
            self.cond(e.args[0], yes, mid, weights=UNLIKELY if weights == UNLIKELY else None, unpredictable=unpredictable)
            self.block = mid
            self.cond(e.args[1], yes, no, weights, unpredictable)
            return
        if k == 'unary' and e.value == '!':
            self.cond(e.args[0], no, yes, weights[::-1] if weights else None, unpredictable)
            return
        if k == 'binary' and e.value == ',':
            self.expr(e.args[0])
            self.cond(e.args[1], yes, no, weights, unpredictable)
            return
        if k == 'call' and e.value is not None and e.value.op in ('expect', 'unpredictable'):
            if e.value.op == 'unpredictable':
                self.cond(e.args[1], yes, no, weights, True)
                return
            expected = const(e.args[2])
            inner = strip_casts(e.args[1])
            if expected is None:
                self.cond(inner, yes, no, weights, unpredictable)
            else:
                self.cond(inner, yes, no, LIKELY if expected else UNLIKELY, unpredictable)
            return
        if k == 'conditional':
            a, b = self.new_block('cond.true'), self.new_block('cond.false')
            self.cond(e.args[0], a, b)
            self.block = a
            self.cond(e.args[1], yes, no, weights, unpredictable)
            self.block = b
            self.cond(e.args[2], yes, no, weights, unpredictable)
            return
        if e.type.kind in ('int', 'ptr'):
            c = const(e0)
            if c is not None:
                self.jump(yes if c else no)
                return
        if k == 'binary' and e.value in ('==', '!=', '<', '>', '<=', '>='):
            self.branch(self.compare(e), yes, no, weights, unpredictable)
            return
        self.branch(self.test(e0), yes, no, weights, unpredictable)

    # --------------------------------------------------------------- calls

    def call(self, e):
        builtin = e.value
        args = e.args[1:]
        if builtin is None:
            callee = self.expr(e.args[0])
            values = [self.expr(a) for a in args]
            t = ir_ty(e.type)
            return self.emit('call', [callee, *values], t if t is not VOID else None)
        return self.intrinsic(builtin, args, e)

    def intrinsic(self, b, args, e):
        op = b.op
        if op in ('expect', 'unpredictable'):
            v = self.expr(args[0])
            for a in args[1:]:
                self.expr(a)
            return v
        if op == 'copy':
            dst, src = self.expr(args[0]), self.expr(args[1])
            n = const(args[2])
            types = {1: I8, 2: I16, 4: I32, 8: I64, 16: Ty('v', 128), 32: Ty('v', 256), 64: Ty('v', 512)}
            if n in types:
                v = self.emit('load', [src], types[n], align=1)
                self.emit('store', [dst, v], align=1)
            else:
                self.emit('memcpy', [dst, src, self.expr(args[2])], align=1)
            return dst
        values = [self.expr(a) for a in args]
        t = ir_ty(e.type)
        if op == 'load':
            return self.emit('load', values, t, align=b.alignment)
        if op == 'store':
            self.emit('store', values, align=b.alignment)
            return None
        if op == 'zero':
            return self.const(t, bytes(t.size))
        if op in ('constant8', 'constant64_reversed') and all(v.const is not None for v in values):
            if op == 'constant8':
                data = bytes(v.const & 0xff for v in values)
            else:
                data = (values[1].const & (1 << 64) - 1).to_bytes(8, 'little') + (values[0].const & (1 << 64) - 1).to_bytes(8, 'little')
            return self.const(t, data)
        if op in ('and', 'or', 'xor'):
            return self.emit(op, values, t)
        if op in ('ctz_nonzero', 'tzcnt'):
            v = self.emit('cttz', values, values[0].ty, zero_undef=op == 'ctz_nonzero')
            return v if t is values[0].ty else self.convert(v, args[0].type, e.type)
        if op == 'bzhi':
            return self.emit('bzhi', [values[0], self.convert(values[1], args[1].type, args[0].type)], t)
        if op == 'splat8' and values[0].const is not None:
            return self.const(t, bytes([values[0].const & 0xff]) * t.size)
        attrs = {'name': op}
        if b.memory:
            attrs['memory'] = b.memory
            attrs['effects'] = 'write' in b.memory
        if b.width:
            attrs['width'] = b.width
        return self.emit('vop', values, t if t is not VOID else None, **attrs)

    # ---------------------------------------------------------- statements

    def initialize(self, address, ctype, init):
        if init.expr is not None:
            if ctype.kind == 'array' and init.expr.kind == 'string':
                data = (init.expr.value + b'\0')[:ctype.count]
                for i, byte in enumerate(data):
                    self.store(self.offset(address, i), self.const(I8, byte), C.CHAR)
            else:
                self.store(address, self.expr(init.expr), ctype)
            return
        if ctype.kind not in ('struct', 'array'):
            self.initialize(address, ctype, init.entries[0][1])
            return
        self.emit('memset', [address, self.const(I8, 0), self.const(I64, ctype.size)], align=ctype.alignment)
        cursor = 0
        for path, value in init.entries:
            if not path:
                path = (cursor,) if ctype.kind == 'array' else (ctype.base.fields[cursor][0],)
            target, current = address, ctype
            for designator in path:
                if current.kind == 'array':
                    target, current = self.offset(target, designator * current.base.size), current.base
                else:
                    _, current, off = current.base.member(designator)
                    target = self.offset(target, off)
            self.initialize(target, current, value)
            cursor = path[0] + 1 if ctype.kind == 'array' else next(i for i, f in enumerate(ctype.base.fields) if f[0] == path[0]) + 1

    def stmt(self, s):
        if s is None:
            return
        if self.block.term is not None and s.kind != 'label':
            self.block = self.new_block('dead')
        k = s.kind
        if k == 'block':
            for child in s.args:
                self.stmt(child)
        elif k == 'decl':
            for d in s.value:
                if 'typedef' in d.storage:
                    continue
                address = self.local(d)
                if d.init is not None:
                    self.initialize(address, d.type, d.init)
        elif k == 'expr':
            self.expr(s.args[0])
        elif k == 'return':
            v = self.expr(s.args[0])
            op = self.fn.new('ret', [v] if v is not None else [])
            self.block.set_term(op)
        elif k == 'if':
            c, a, b = s.args
            yes, done = self.new_block('if.then'), self.new_block('if.end')
            no = self.new_block('if.else') if b is not None else done
            self.cond(c, yes, no)
            self.block = yes
            self.stmt(a)
            self.jump(done)
            if b is not None:
                self.block = no
                self.stmt(b)
                self.jump(done)
            self.block = done
        elif k in ('while', 'do', 'for'):
            self.loop(s)
        elif k in ('break', 'continue'):
            self.jump(self.loops[-1][k == 'continue'])
        elif k == 'goto':
            self.jump(self.label(s.value))
        elif k == 'label':
            target = self.label(s.value)
            self.jump(target)
            self.block = target
            self.stmt(s.args[0])
        elif k == 'asm':
            self.asm(s)
        elif k == 'pragma':
            self.no_unroll = 'unroll' in s.value
        elif k != 'empty':
            s.token.error('unsupported statement ' + k)

    def label(self, name):
        if name not in self.labels:
            self.labels[name] = self.new_block('label.' + name)
        return self.labels[name]

    def loop(self, s):
        k = s.kind
        no_unroll, self.no_unroll = self.no_unroll, False
        if k == 'for':
            init, c, step, body = s.args
            self.stmt(init)
        elif k == 'while':
            c, body = s.args
            step = None
        else:
            body, c = s.args
            step = None
        loop_body = self.new_block('loop.body')
        head = loop_body if k == 'do' else self.new_block('loop.head')
        latch, done = self.new_block('loop.latch'), self.new_block('loop.end')
        if no_unroll:
            head.attrs['nounroll'] = True
        self.jump(head)
        if k != 'do':
            self.block = head
            if c is not None:
                self.cond(c, loop_body, done)
            else:
                self.jump(loop_body)
        self.loops.append((done, latch))
        self.block = loop_body
        self.stmt(body)
        self.jump(latch)
        self.loops.pop()
        self.block = latch
        if k == 'do':
            self.cond(c, loop_body, done)
        else:
            self.expr(step)
            self.jump(head)
        self.block = done

    def asm(self, s):
        code, volatile, groups = s.value
        outputs = groups[0] if groups else []
        inputs = groups[1] if len(groups) > 1 else []
        clobbers = [c for c, _ in groups[2]] if len(groups) > 2 else []
        args = [self.expr(e) for c, e in outputs if '+' in c] + [self.expr(e) for _, e in inputs]
        tys = [ir_ty(e.type) for _, e in outputs]
        op = self.fn.new('asm', args, tys=tys, attrs={
            'code': code, 'volatile': volatile, 'outputs': [c for c, _ in outputs], 'inputs': [c for c, _ in inputs],
            'clobbers': clobbers, 'memory': code.startswith('rep ') or 'memory' in clobbers})
        self.block.append(op)
        for (_, e), r in zip(outputs, op.results):
            self.store(self.address(e), r, e.type)

    def run(self):
        self.stmt(self.decl.body)
        if self.block.term is None:
            self.block.set_term(self.fn.new('ret' if self.fn.ret_ty is VOID else 'unreachable'))
        fn = self.fn
        fn.remove_unreachable()
        return fn


def lower_function(module, decl):
    return Lower(module, decl).run()
