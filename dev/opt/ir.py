"""The optimizer's IR: SSA values of signless types, ops in basic blocks, an
explicit CFG.

Types are i1 (a branch condition), i8..i64, ptr and v128/v256/v512; how an
operation treats its operands (signed or not, which shift) is the op's kind,
not the type's. Every value has one defining op and a list of its uses (the
ops that read it, once per operand), kept by the methods here; a pass changes
operands only through them. Constants, parameters, undefined values and the
addresses of globals are ops outside any block; the rest live in blocks, phis
first, a terminator last (jmp, br, ret, unreachable). A phi's incoming values
are parallel to its preds, one per predecessor; a br never names the same
block twice, so the predecessors of a block are a set.
"""


class Ty:
    __slots__ = ('kind', 'bits')
    _interned = {}

    def __new__(cls, kind, bits):
        t = cls._interned.get((kind, bits))
        if t is None:
            t = object.__new__(cls)
            t.kind, t.bits = kind, bits
            cls._interned[(kind, bits)] = t
        return t

    @property
    def size(self):
        return max(self.bits // 8, 1)

    @property
    def is_int(self):
        return self.kind == 'i'

    @property
    def is_vec(self):
        return self.kind == 'v'

    @property
    def is_ptr(self):
        return self.kind == 'p'

    @property
    def scalar(self):
        return self.kind in ('i', 'p')

    @property
    def mask(self):
        return (1 << self.bits) - 1

    def __repr__(self):
        return {'i': 'i%d' % self.bits, 'p': 'ptr', 'v': 'v%d' % self.bits}.get(self.kind, self.kind)


I1, I8, I16, I32, I64 = (Ty('i', n) for n in (1, 8, 16, 32, 64))
PTR = Ty('p', 64)
V128, V256, V512 = (Ty('v', n) for n in (128, 256, 512))
VOID = Ty('void', 0)


def int_ty(bits):
    return Ty('i', bits)


def signed(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


TERMINATORS = {'jmp', 'br', 'ret', 'unreachable'}
# no effect but their result: free to remove when unused, to move, to merge
PURE = {'add', 'sub', 'mul', 'and', 'or', 'xor', 'shl', 'lshr', 'ashr', 'icmp', 'zext', 'sext', 'trunc',
        'p2i', 'i2p', 'ptradd', 'select', 'cttz', 'bzhi', 'phi', 'udiv', 'urem', 'sdiv', 'srem'}
# may trap or read memory: not moved past a store, not speculated
DIVISIONS = {'udiv', 'urem', 'sdiv', 'srem'}
COMMUTATIVE = {'add', 'mul', 'and', 'or', 'xor'}
SWAP = {'eq': 'eq', 'ne': 'ne', 'ult': 'ugt', 'ugt': 'ult', 'ule': 'uge', 'uge': 'ule',
        'slt': 'sgt', 'sgt': 'slt', 'sle': 'sge', 'sge': 'sle'}
INVERT = {'eq': 'ne', 'ne': 'eq', 'ult': 'uge', 'uge': 'ult', 'ugt': 'ule', 'ule': 'ugt',
          'slt': 'sge', 'sge': 'slt', 'sgt': 'sle', 'sle': 'sgt'}


class Value:
    __slots__ = ('id', 'ty', 'op', 'uses')

    def __init__(self, id_, ty, op):
        self.id, self.ty, self.op, self.uses = id_, ty, op, []

    def __repr__(self):
        return '%%%d' % self.id

    @property
    def const(self):
        """The integer (or bytes) of a constant, else None."""
        return self.op.attrs['value'] if self.op.kind == 'const' else None


class Op:
    __slots__ = ('kind', 'args', 'results', 'attrs', 'block', 'targets', 'preds')

    def __init__(self, kind, args=(), attrs=None, targets=(), preds=None):
        self.kind = kind
        self.args = list(args)
        for a in self.args:
            a.uses.append(self)
        self.results = []
        self.attrs = attrs if attrs is not None else {}
        self.block = None
        self.targets = list(targets)
        self.preds = list(preds) if preds is not None else None

    @property
    def result(self):
        return self.results[0] if self.results else None

    @property
    def ty(self):
        return self.results[0].ty if self.results else VOID

    def set_arg(self, i, v):
        old = self.args[i]
        if old is v:
            return
        old.uses.remove(self)
        self.args[i] = v
        v.uses.append(self)

    def replace_arg(self, old, new):
        for i, a in enumerate(self.args):
            if a is old:
                self.set_arg(i, new)

    def detach(self):
        for a in self.args:
            a.uses.remove(self)
        self.args = []

    # phis
    def incoming(self, pred):
        return self.args[self.preds.index(pred)]

    def add_incoming(self, pred, v):
        self.preds.append(pred)
        self.args.append(v)
        v.uses.append(self)

    def remove_incoming(self, pred):
        i = self.preds.index(pred)
        self.args[i].uses.remove(self)
        del self.args[i]
        del self.preds[i]

    @property
    def is_term(self):
        return self.kind in TERMINATORS

    @property
    def effects(self):
        """Writes memory, may trap, or does something besides its result."""
        k = self.kind
        if k in PURE:
            return k in DIVISIONS
        if k == 'load':
            return self.attrs.get('volatile', False)
        if k == 'vop':
            return bool(self.attrs.get('effects'))
        return k not in ('const', 'param', 'undef', 'sym')

    @property
    def reads_memory(self):
        k = self.kind
        if k in ('load', 'memcpy', 'call', 'asm'):
            return k != 'asm' or bool(self.attrs.get('memory'))
        return k == 'vop' and 'read' in self.attrs.get('memory', '')

    @property
    def writes_memory(self):
        k = self.kind
        if k in ('store', 'memcpy', 'memset', 'call'):
            return True
        if k == 'asm':
            return bool(self.attrs.get('memory'))
        return k == 'vop' and 'write' in self.attrs.get('memory', '')

    def __repr__(self):
        return format_op(self)


class Block:
    __slots__ = ('id', 'name', 'ops', 'term', 'attrs')

    def __init__(self, id_, name):
        self.id, self.name, self.ops, self.term, self.attrs = id_, name, [], None, {}

    def __repr__(self):
        return '%s.%d' % (self.name, self.id)

    @property
    def succs(self):
        return self.term.targets if self.term is not None else []

    def phis(self):
        out = []
        for op in self.ops:
            if op.kind != 'phi':
                break
            out.append(op)
        return out

    def append(self, op):
        op.block = self
        self.ops.append(op)
        return op

    def insert(self, index, op):
        op.block = self
        self.ops.insert(index, op)
        return op

    def insert_before(self, anchor, op):
        return self.insert(self.ops.index(anchor), op)

    def remove(self, op):
        self.ops.remove(op)
        op.block = None

    def set_term(self, op):
        if self.term is not None:
            self.term.detach()
            self.term.block = None
        op.block = self
        self.term = op


class Function:
    def __init__(self, name, param_tys, ret_ty, attrs=None):
        self.name, self.ret_ty = name, ret_ty
        self.attrs = attrs if attrs is not None else {}
        self.blocks, self.next_id, self.next_block = [], 0, 0
        self.consts = {}
        self.params = [self._value(Op('param', attrs={'index': n}), t) for n, t in enumerate(param_tys)]
        self.entry = self.block('entry')

    def _value(self, op, ty):
        v = Value(self.next_id, ty, op)
        self.next_id += 1
        op.results.append(v)
        return v

    def new(self, kind, args=(), ty=None, attrs=None, tys=None):
        """An op not yet in a block, with one result of ty (none for None or VOID), or one per tys."""
        op = Op(kind, args, attrs)
        for t in (tys if tys is not None else [ty] if ty is not None and ty is not VOID else []):
            self._value(op, t)
        return op

    def phi(self, ty):
        op = Op('phi', preds=[])
        self._value(op, ty)
        return op

    def const(self, ty, value):
        if ty.kind in ('i', 'p'):
            value &= ty.mask
        key = (ty, value)
        v = self.consts.get(key)
        if v is None:
            v = self._value(Op('const', attrs={'value': value}), ty)
            self.consts[key] = v
        return v

    def undef(self, ty):
        key = (ty, 'undef')
        v = self.consts.get(key)
        if v is None:
            v = self._value(Op('undef'), ty)
            self.consts[key] = v
        return v

    def sym(self, name):
        key = (PTR, 'sym', name)
        v = self.consts.get(key)
        if v is None:
            v = self._value(Op('sym', attrs={'name': name}), PTR)
            self.consts[key] = v
        return v

    def block(self, name='b'):
        b = Block(self.next_block, name)
        self.next_block += 1
        self.blocks.append(b)
        return b

    # ---------------------------------------------------------------- CFG

    def preds(self):
        out = {b: [] for b in self.blocks}
        for b in self.blocks:
            for t in b.succs:
                out[t].append(b)
        return out

    def rpo(self):
        """The blocks reachable from the entry, in reverse postorder."""
        seen, post, stack = {self.entry}, [], [(self.entry, iter(self.entry.succs))]
        while stack:
            b, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                post.append(b)
                stack.pop()
            elif nxt not in seen:
                seen.add(nxt)
                stack.append((nxt, iter(nxt.succs)))
        return post[::-1]

    def remove_unreachable(self):
        live = set(self.rpo())
        dead = [b for b in self.blocks if b not in live]
        if not dead:
            return False
        for b in dead:
            for t in b.succs:
                if t in live:
                    for phi in t.phis():
                        while b in phi.preds:
                            phi.remove_incoming(b)
        for b in dead:
            for op in b.ops + ([b.term] if b.term else []):
                op.detach()
        for b in dead:
            for op in b.ops:
                for r in op.results:
                    for user in list(r.uses):
                        if user.block in live:
                            raise ValueError('%s: %s defined in an unreachable block is used in %s' % (self.name, r, user.block))
        self.blocks = [b for b in self.blocks if b in live]
        return True

    def ops(self):
        for b in self.blocks:
            yield from b.ops
            yield b.term

    def replace_all_uses(self, old, new):
        if old is new:
            return
        for user in list(old.uses):
            user.replace_arg(old, new)

    def redirect(self, block, old, new):
        """Make block's edge to old go to new (the phis of both follow)."""
        t = block.term
        t.targets = [new if x is old else x for x in t.targets]
        for phi in old.phis():
            if block in phi.preds:
                phi.remove_incoming(block)

    # ------------------------------------------------------------- checks

    def verify(self):
        dom = Dominators(self)
        preds = self.preds()
        where = {}
        for b in self.blocks:
            if b.term is None:
                raise ValueError('%s: %s has no terminator' % (self.name, b))
            if b.term.kind == 'br' and (len(b.term.targets) != 2 or b.term.targets[0] is b.term.targets[1]):
                raise ValueError('%s: %s branches twice to one block' % (self.name, b))
            seen_body = False
            for i, op in enumerate(b.ops):
                if op.block is not b:
                    raise ValueError('%s: %s in %s claims %s' % (self.name, op, b, op.block))
                if op.is_term:
                    raise ValueError('%s: terminator %s inside %s' % (self.name, op, b))
                if op.kind == 'phi':
                    if seen_body:
                        raise ValueError('%s: phi %s after other ops in %s' % (self.name, op, b))
                    if sorted(p.id for p in op.preds) != sorted(p.id for p in preds[b]) or len(op.args) != len(op.preds):
                        raise ValueError('%s: phi %s of %s has preds %s, the block %s' % (self.name, op, b, op.preds, preds[b]))
                else:
                    seen_body = True
                for r in op.results:
                    if r in where:
                        raise ValueError('%s: %s defined twice' % (self.name, r))
                    where[r] = (b, i)
        for b in self.blocks:
            for i, op in enumerate(b.ops + [b.term]):
                for n, a in enumerate(op.args):
                    if op not in a.uses:
                        raise ValueError('%s: %s does not list %s among its uses' % (self.name, a, op))
                    if a.op.block is None:
                        if a.op.kind not in ('const', 'param', 'undef', 'sym'):
                            raise ValueError('%s: %s of a removed op %s used by %s' % (self.name, a, a.op.kind, op))
                        continue
                    if a not in where:
                        raise ValueError('%s: %s used by %s is not defined' % (self.name, a, op))
                    db, di = where[a]
                    if op.kind == 'phi':
                        ub, ui = op.preds[n], len(op.preds[n].ops)
                    else:
                        ub, ui = b, i
                    if not dom.dominates(db, ub) or db is ub and di >= ui:
                        raise ValueError('%s: %s (%s) does not dominate its use in %s' % (self.name, a, db, op))
        for b in self.blocks:
            for op in b.ops + [b.term]:
                for r in op.results:
                    for u in r.uses:
                        if u.block is None or r not in u.args:
                            raise ValueError('%s: stale use of %s by %s' % (self.name, r, u))

    def dump(self):
        lines = ['function %s(%s) -> %r' % (self.name, ', '.join('%r:%r' % (v, v.ty) for v in self.params), self.ret_ty)]
        for b in self.blocks:
            lines.append('%r:%s' % (b, ''.join(' %s=%s' % kv for kv in sorted(b.attrs.items()))))
            for op in b.ops:
                lines.append('  ' + format_op(op))
            lines.append('  ' + (format_op(b.term) if b.term else '<no terminator>'))
        return '\n'.join(lines)


def value_text(v):
    op = v.op
    if op.kind == 'const':
        x = op.attrs['value']
        return ('%s' % x.hex() if isinstance(x, bytes) else '%d' % x) + ':%r' % v.ty
    if op.kind == 'sym':
        return '@' + op.attrs['name']
    if op.kind == 'undef':
        return 'undef:%r' % v.ty
    return repr(v)


def format_op(op):
    res = ', '.join('%r:%r' % (r, r.ty) for r in op.results)
    if op.kind == 'phi':
        args = ', '.join('[%r: %s]' % (p, value_text(a)) for p, a in zip(op.preds, op.args))
    else:
        args = ', '.join(value_text(a) for a in op.args)
    if op.targets:
        args += (' -> ' if args else '-> ') + ', '.join(map(repr, op.targets))
    attrs = {k: v for k, v in op.attrs.items() if k not in ('builtin',)}
    if 'builtin' in op.attrs:
        attrs['builtin'] = op.attrs['builtin'].name
    text = ' {%s}' % ', '.join('%s=%s' % (k, v.hex() if isinstance(v, bytes) else v) for k, v in attrs.items()) if attrs else ''
    return (res + ' = ' if res else '') + op.kind + (' ' + args if args else '') + text


# ------------------------------------------------------------- dominators

class Dominators:
    """Immediate dominators by Cooper, Harvey and Kennedy; the tree's
    pre/post numbers answer dominates() in constant time."""

    def __init__(self, fn):
        order = fn.rpo()
        index = {b: i for i, b in enumerate(order)}
        preds = fn.preds()
        idom = {order[0]: order[0]}

        def intersect(a, b):
            while a is not b:
                while index[a] > index[b]:
                    a = idom[a]
                while index[b] > index[a]:
                    b = idom[b]
            return a

        changed = True
        while changed:
            changed = False
            for b in order[1:]:
                new = None
                for p in preds[b]:
                    if p in idom:
                        new = p if new is None else intersect(p, new)
                if idom.get(b) is not new:
                    idom[b] = new
                    changed = True
        self.order, self.idom = order, idom
        self.children = {b: [] for b in order}
        for b in order[1:]:
            self.children[idom[b]].append(b)
        self.pre, self.post = {}, {}
        n, stack = 0, [(order[0], False)]
        while stack:
            b, done = stack.pop()
            if done:
                self.post[b] = n
                n += 1
                continue
            self.pre[b] = n
            n += 1
            stack.append((b, True))
            stack.extend((c, False) for c in reversed(self.children[b]))

    def dominates(self, a, b):
        return a in self.pre and b in self.pre and self.pre[a] <= self.pre[b] and self.post[b] <= self.post[a]

    def parent(self, b):
        p = self.idom.get(b)
        return None if p is b else p

    def lca(self, a, b):
        while not self.dominates(a, b):
            a = self.idom[a]
        return a

    def depth(self, b):
        d = 0
        while self.idom[b] is not b:
            b = self.idom[b]
            d += 1
        return d


# ------------------------------------------------------------------ loops

class Loop:
    def __init__(self, header):
        self.header, self.blocks, self.latches, self.children, self.parent = header, {header}, [], [], None

    @property
    def depth(self):
        d, loop = 1, self.parent
        while loop is not None:
            d += 1
            loop = loop.parent
        return d

    def exits(self, fn=None):
        """(inside, outside) edges leaving the loop."""
        return [(b, t) for b in self.blocks for t in b.succs if t not in self.blocks]

    def __repr__(self):
        return 'Loop(%r, %d blocks)' % (self.header, len(self.blocks))


def find_loops(fn, dom=None):
    """The natural loops, innermost last in the list; a block's innermost loop in .of."""
    dom = dom or Dominators(fn)
    preds = fn.preds()
    loops = {}
    for b in dom.order:
        for t in b.succs:
            if dom.dominates(t, b):
                loop = loops.setdefault(t, Loop(t))
                loop.latches.append(b)
                todo = [b]
                while todo:
                    x = todo.pop()
                    if x in loop.blocks:
                        continue
                    loop.blocks.add(x)
                    todo.extend(p for p in preds[x] if p in dom.pre)
    ordered = sorted(loops.values(), key=lambda l: len(l.blocks), reverse=True)
    for i, inner in enumerate(ordered):
        for outer in reversed(ordered[:i]):
            if inner.header in outer.blocks and inner is not outer:
                inner.parent = outer
                outer.children.append(inner)
                break
    innermost = {}
    for loop in ordered:
        for b in loop.blocks:
            innermost[b] = loop
    result = LoopInfo(ordered, innermost)
    return result


class LoopInfo:
    def __init__(self, loops, innermost):
        self.loops, self.of = loops, innermost

    def depth(self, b):
        loop = self.of.get(b)
        return loop.depth if loop else 0
