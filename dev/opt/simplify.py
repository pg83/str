"""Local simplification: constants folded, identities applied, operands put
in a canonical order and form, comparisons decided or narrowed by what is
known of their operands (facts.py); and dead code removed.

simplify(fn) runs a worklist over every op until nothing changes. A rule
returns the value that replaces the op's result, or rewrites the op in
place; new ops go in front of the op they serve."""
from .facts import Facts
from .ir import I1, I64, PTR, SWAP, INVERT, COMMUTATIVE, DIVISIONS, Dominators, signed


def removable(op):
    """Nothing but its result: gone when nobody uses that (a division too: by zero it is undefined)."""
    if op.kind in ('alloca',) or op.kind in DIVISIONS:
        return True
    return not op.effects and op.kind not in ('asm', 'call')


def dead_code(fn):
    """Remove every op whose result nobody uses and that has no effect."""
    changed = False
    work = [op for b in fn.blocks for op in b.ops]
    while work:
        op = work.pop()
        if op.block is None or any(r.uses for r in op.results):
            continue
        if removable(op):
            args = [a.op for a in op.args]
            op.detach()
            op.block.remove(op)
            work.extend(a for a in args if a.block is not None)
            changed = True
    return changed


def fold_int(kind, a, b, ty):
    """Constant folding of a binary integer op, or None."""
    n, mask = ty.bits, ty.mask
    if kind == 'add':
        return (a + b) & mask
    if kind == 'sub':
        return (a - b) & mask
    if kind == 'mul':
        return (a * b) & mask
    if kind == 'and':
        return a & b
    if kind == 'or':
        return a | b
    if kind == 'xor':
        return a ^ b
    if kind == 'shl':
        return (a << b) & mask if b < n else None
    if kind == 'lshr':
        return a >> b if b < n else None
    if kind == 'ashr':
        return (signed(a, n) >> b) & mask if b < n else None
    if b == 0:
        return None
    if kind == 'udiv':
        return a // b
    if kind == 'urem':
        return a % b
    sa, sb = signed(a, n), signed(b, n)
    q = abs(sa) // abs(sb) * (1 if (sa < 0) == (sb < 0) else -1)
    if kind == 'sdiv':
        return q & mask
    if kind == 'srem':
        return (sa - q * sb) & mask
    return None


def compare(pred, a, b, bits):
    if pred in ('slt', 'sle', 'sgt', 'sge'):
        a, b = signed(a, bits), signed(b, bits)
    return {'eq': a == b, 'ne': a != b, 'ult': a < b, 'ule': a <= b, 'ugt': a > b, 'uge': a >= b,
            'slt': a < b, 'sle': a <= b, 'sgt': a > b, 'sge': a >= b}[pred]


def decide(pred, lo1, hi1, lo2, hi2):
    """The truth of an unsigned comparison of values in two ranges, or None."""
    if pred == 'ult':
        return True if hi1 < lo2 else False if lo1 >= hi2 else None
    if pred == 'ule':
        return True if hi1 <= lo2 else False if lo1 > hi2 else None
    if pred == 'ugt':
        return True if lo1 > hi2 else False if hi1 <= lo2 else None
    if pred == 'uge':
        return True if lo1 >= hi2 else False if hi1 < lo2 else None
    if pred == 'eq':
        return False if hi1 < lo2 or hi2 < lo1 else (True if lo1 == hi1 == lo2 == hi2 else None)
    if pred == 'ne':
        return True if hi1 < lo2 or hi2 < lo1 else (False if lo1 == hi1 == lo2 == hi2 else None)
    return None


def power_of_two(c):
    return c > 0 and c & (c - 1) == 0


def bit_test(v):
    """(x, mask, kind) when v tests bits of x: 'all' (x & mask == mask), 'none'
    (x & mask == 0), 'any' (!= 0) or 'notall' (!= mask); else None."""
    op = v.op
    if op.kind != 'icmp' or op.attrs['pred'] not in ('eq', 'ne'):
        return None
    a, b = op.args
    c = b.const
    if c is None:
        return None
    if a.op.kind == 'and' and a.op.args[1].const is not None:
        x, m = a.op.args[0], a.op.args[1].const
    else:
        x, m = a, a.ty.mask
    eq = op.attrs['pred'] == 'eq'
    if c == 0:
        return x, m, 'none' if eq else 'any'
    if c == m:
        return x, m, 'all' if eq else 'notall'
    return None


class Simplifier:
    def __init__(self, fn):
        self.fn = fn
        self.facts = Facts(fn)
        self.changed = False
        self.dom = None

    def dominators(self):
        if self.dom is None:
            self.dom = Dominators(self.fn)
        return self.dom

    def new(self, before, kind, args, ty, **attrs):
        op = self.fn.new(kind, args, ty, attrs)
        before.block.insert_before(before, op)
        self.work.append(op)
        return op.result

    def run(self):
        fn = self.fn
        self.work = [op for b in reversed(fn.rpo()) for op in reversed(b.ops + [b.term])]
        while self.work:
            op = self.work.pop()
            if op.block is None:
                continue
            new = self.visit(op)
            if new is None or op.result is None or new is op.result:
                continue
            self.changed = True
            self.facts = Facts(fn)
            users = list(op.result.uses)
            fn.replace_all_uses(op.result, new)
            self.work.extend(users)
            if removable(op) or op.kind == 'load':
                args = [a.op for a in op.args]
                op.detach()
                op.block.remove(op)
                self.work.extend(a for a in args if a.block is not None)
        if dead_code(fn):
            self.changed = True
        return self.changed

    def requeue(self, op):
        self.changed = True
        self.facts = Facts(self.fn)
        self.work.append(op)
        for r in op.results:
            self.work.extend(r.uses)

    # --------------------------------------------------------------- rules

    def visit(self, op):
        k = op.kind
        if k == 'phi':
            others = {a for a in op.args if a is not op.result and a.op.kind != 'undef'}
            if len(others) == 1:
                v = others.pop()
                if len(others) + 1 == len({a for a in op.args if a is not op.result}):
                    return v
                # undefined on the other ways: v stands for the phi where v is defined
                if v.op.block is None or self.dominators().dominates(v.op.block, op.block):
                    return v
                return None
            if not others:
                return self.fn.undef(op.ty)
            return None
        if k in ('add', 'sub', 'mul', 'and', 'or', 'xor', 'shl', 'lshr', 'ashr', 'udiv', 'sdiv', 'urem', 'srem'):
            if op.ty.kind == 'v':
                return self.vector_logic(op)
            return self.binary(op)
        if k == 'icmp':
            return self.icmp(op)
        if k in ('zext', 'sext', 'trunc'):
            return self.cast(op)
        if k == 'p2i':
            a = op.args[0]
            if a.op.kind == 'i2p':
                return a.op.args[0]
            if a.op.kind == 'ptradd' and (len(a.uses) == 1 or a.op.args[1].const is None):
                return self.new(op, 'add', [self.new(op, 'p2i', [a.op.args[0]], I64), a.op.args[1]], I64)
            if a.op.kind == 'const':
                return self.fn.const(I64, a.const)
            return None
        if k == 'i2p':
            a = op.args[0]
            if a.op.kind == 'p2i':
                return a.op.args[0]
            if a.op.kind == 'const':
                return self.fn.const(PTR, a.const)
            return None
        if k == 'ptradd':
            p, off = op.args
            if off.const == 0:
                return p
            if p.op.kind == 'ptradd' and off.const is not None and p.op.args[1].const is not None:
                return self.new(op, 'ptradd', [p.op.args[0], self.fn.const(I64, p.op.args[1].const + off.const)], PTR)
            if p.op.kind == 'const' and off.const is not None:
                return self.fn.const(PTR, p.const + off.const)
            return None
        if k == 'select':
            c, a, b = op.args
            if c.const is not None:
                return a if c.const else b
            if a is b:
                return a
            if op.ty is I1 and a.const is not None and b.const is not None:
                return c if a.const else self.new(op, 'xor', [c, self.fn.const(I1, 1)], I1)
            if c.op.kind == 'xor' and c.op.args[1].const == 1 and c.ty is I1:
                op.set_arg(0, c.op.args[0])
                op.set_arg(1, b)
                op.set_arg(2, a)
                self.requeue(op)
                return None
            if c.op.kind == 'icmp' and c.op.attrs['pred'] in ('eq', 'ne') and c.op.args[1].const == 0 and a.const is not None and b.const is not None \
                    and c.op.args[0].ty is op.ty and self.facts.range(c.op.args[0]) == (0, 1):
                # x in {0, 1} picks between two constants: arithmetic on x
                x = c.op.args[0]
                hi, lo = (a.const, b.const) if c.op.attrs['pred'] == 'ne' else (b.const, a.const)   # x = 1 gives hi
                mask = op.ty.mask
                if hi == (lo + 1) & mask:
                    return x if lo == 0 else self.new(op, 'add', [x, self.fn.const(op.ty, lo)], op.ty)
                if lo == (hi + 1) & mask:
                    if hi == 0:
                        return self.new(op, 'xor', [x, self.fn.const(op.ty, 1)], op.ty)
                    return self.new(op, 'sub', [self.fn.const(op.ty, lo), x], op.ty)
            t = bit_test(c)
            if t is not None and power_of_two(t[1]) and t[0].ty is op.ty and a.const is not None and b.const is not None:
                # (x & 2^n) ? 2^k : 0 is the bit shifted into place
                x, m, kind = t
                hi, lo = (a.const, b.const) if kind in ('all', 'any') else (b.const, a.const)
                if lo == 0 and power_of_two(hi):
                    n, k = m.bit_length() - 1, hi.bit_length() - 1
                    if n == k:
                        return self.new(op, 'and', [x, self.fn.const(op.ty, hi)], op.ty)
                    shifted = self.new(op, 'lshr' if n > k else 'shl', [x, self.fn.const(op.ty, abs(n - k))], op.ty)
                    return self.new(op, 'and', [shifted, self.fn.const(op.ty, hi)], op.ty)
            return None
        if k == 'cttz':
            a = op.args[0]
            if a.const is not None:
                if a.const == 0:
                    return None if op.attrs.get('zero_undef') else self.fn.const(op.ty, a.ty.bits)
                return self.fn.const(op.ty, (a.const & -a.const).bit_length() - 1)
            if not op.attrs.get('zero_undef') and self.facts.nonzero(a):
                op.attrs['zero_undef'] = True
            return None
        if k == 'bzhi':
            x, n = op.args
            if n.const is not None:
                count = n.const & 0xff
                if count >= op.ty.bits:
                    return x
                return self.new(op, 'and', [x, self.fn.const(op.ty, (1 << count) - 1)], op.ty)
            return None
        if k == 'br':
            c = op.args[0]
            if c.op.kind == 'xor' and c.op.args[1].const == 1:
                op.set_arg(0, c.op.args[0])
                op.targets.reverse()
                if 'weights' in op.attrs:
                    op.attrs['weights'] = op.attrs['weights'][::-1]
                self.changed = True
                self.work.append(op)
            return None
        if k == 'vop':
            return self.vop(op)
        return None

    def binary(self, op):
        k, ty = op.kind, op.ty
        a, b = op.args
        ca, cb = a.const, b.const
        if ca is not None and cb is not None:
            v = fold_int(k, ca, cb, ty)
            return self.fn.const(ty, v) if v is not None else None
        if k in COMMUTATIVE and ca is not None:
            op.set_arg(0, b)
            op.set_arg(1, a)
            self.requeue(op)
            return None
        mask = ty.mask
        if cb is not None:
            if cb == 0:
                if k in ('add', 'sub', 'or', 'xor', 'shl', 'lshr', 'ashr'):
                    return a
                if k in ('and', 'mul'):
                    return self.fn.const(ty, 0)
            if cb == 1 and k in ('mul', 'udiv', 'sdiv'):
                return a
            if cb == mask:
                if k == 'and':
                    return a
                if k == 'or':
                    return self.fn.const(ty, mask)
            if k == 'sub':
                return self.new(op, 'add', [a, self.fn.const(ty, -cb)], ty)
            if k == 'mul' and power_of_two(cb):
                return self.new(op, 'shl', [a, self.fn.const(ty, cb.bit_length() - 1)], ty)
            if k == 'udiv' and power_of_two(cb):
                return self.new(op, 'lshr', [a, self.fn.const(ty, cb.bit_length() - 1)], ty)
            if k == 'urem' and power_of_two(cb):
                return self.new(op, 'and', [a, self.fn.const(ty, cb - 1)], ty)
            if k in ('shl', 'lshr', 'ashr') and cb >= ty.bits:
                return None
            # reassociate the constants of a chain
            inner = a.op
            if inner.kind == k and k in ('add', 'and', 'or', 'xor', 'mul') and inner.args[1].const is not None:
                c = fold_int(k, inner.args[1].const, cb, ty)
                return self.new(op, k, [inner.args[0], self.fn.const(ty, c)], ty)
            if k in ('shl', 'lshr') and inner.kind == k and inner.args[1].const is not None:
                total = inner.args[1].const + cb
                if total >= ty.bits:
                    return self.fn.const(ty, 0)
                return self.new(op, k, [inner.args[0], self.fn.const(ty, total)], ty)
            if k == 'and':
                z, o = self.facts.bits(a)
                if (~cb & mask) & ~z == 0:
                    return a                       # the bits cleared are known zero
                if cb & ~z == 0:
                    return self.fn.const(ty, 0)
            if k == 'or':
                z, o = self.facts.bits(a)
                if cb & ~o == 0:
                    return a
            if k in ('lshr', 'ashr'):
                z, o = self.facts.bits(a)
                if (mask >> cb) & ~z == 0:
                    return self.fn.const(ty, 0)
            if k == 'add' and inner.kind == 'sub' and inner.args[0].const is not None:
                return self.new(op, 'sub', [self.fn.const(ty, inner.args[0].const + cb), inner.args[1]], ty)
        if a is b:
            if k in ('and', 'or'):
                return a
            if k in ('sub', 'xor'):
                return self.fn.const(ty, 0)
        if ca == 0 and k in ('shl', 'lshr', 'ashr', 'udiv', 'sdiv', 'urem', 'srem'):
            return a
        if k == 'sub':
            pa, pb = a.op, b.op
            # (x + y) - x, (p + y) - p through p2i
            if pa.kind == 'add' and pa.args[0] is b:
                return pa.args[1]
            if pa.kind == 'add' and pa.args[1] is b:
                return pa.args[0]
            if pa.kind == 'add' and pb.kind == 'add' and pa.args[0] is pb.args[0]:
                return self.new(op, 'sub', [pa.args[1], pb.args[1]], ty)
            if pb.kind == 'add' and pb.args[0] is a and pb.args[1].const is not None:
                return self.fn.const(ty, -pb.args[1].const)
        if k == 'add':
            # x + (0 - y) -> x - y
            if b.op.kind == 'sub' and b.op.args[0].const == 0:
                return self.new(op, 'sub', [a, b.op.args[1]], ty)
        if k == 'and' and ty.kind == 'i':
            za, _ = self.facts.bits(a)
            zb, _ = self.facts.bits(b)
            if za | zb == mask:
                return self.fn.const(ty, 0)
        if k == 'or' and ty.kind == 'i' and cb is None and ca is None:
            za, _ = self.facts.bits(a)
            zb, _ = self.facts.bits(b)
            if za | zb == mask:
                return self.new(op, 'add', [a, b], ty)       # no bit in both: a sum, which a lea can take
        if k in ('and', 'or') and ty is I1:
            return self.combine_tests(op, k, a, b)
        return None

    def combine_tests(self, op, k, a, b):
        """(x & m1) == m1 && (x & m2) == m2 is (x & (m1|m2)) == (m1|m2); the zero
        tests alike, and the two kinds' negations under ||."""
        ta, tb = bit_test(a), bit_test(b)
        if ta is None or tb is None or ta[0] is not tb[0]:
            return None
        x, m1, kind1 = ta
        _, m2, kind2 = tb
        m = m1 | m2
        if k == 'and':
            kind1 = {'any': 'all', 'notall': 'none'}.get(kind1, kind1) if power_of_two(m1) else kind1
            kind2 = {'any': 'all', 'notall': 'none'}.get(kind2, kind2) if power_of_two(m2) else kind2
            if kind1 != kind2 or kind1 not in ('all', 'none'):
                return None
            masked = self.new(op, 'and', [x, self.fn.const(x.ty, m)], x.ty)
            return self.new(op, 'icmp', [masked, self.fn.const(x.ty, m if kind1 == 'all' else 0)], I1, pred='eq')
        kind1 = {'all': 'any', 'none': 'notall'}.get(kind1, kind1) if power_of_two(m1) else kind1
        kind2 = {'all': 'any', 'none': 'notall'}.get(kind2, kind2) if power_of_two(m2) else kind2
        if kind1 != kind2 or kind1 not in ('any', 'notall'):
            return None
        masked = self.new(op, 'and', [x, self.fn.const(x.ty, m)], x.ty)
        return self.new(op, 'icmp', [masked, self.fn.const(x.ty, 0 if kind1 == 'any' else m)], I1, pred='ne')

    def vector_logic(self, op):
        k = op.kind
        a, b = op.args
        if k not in ('and', 'or', 'xor'):
            return None
        if a.const is not None and b.const is None:
            op.set_arg(0, b)
            op.set_arg(1, a)
            self.requeue(op)
            return None
        cb = b.const
        size = op.ty.size
        if cb is not None:
            if cb == bytes(size):
                return a if k in ('or', 'xor') else b
            if cb == b'\xff' * size and k == 'and':
                return a
            if a.const is not None:
                return self.fn.const(op.ty, bytes((x & y) if k == 'and' else (x | y) if k == 'or' else (x ^ y) for x, y in zip(a.const, cb)))
        if a is b:
            return a if k != 'xor' else self.fn.const(op.ty, bytes(size))
        return None

    def cast(self, op):
        k, ty = op.kind, op.ty
        a = op.args[0]
        if a.ty is ty:
            return a
        if a.const is not None:
            if k == 'sext':
                return self.fn.const(ty, signed(a.const, a.ty.bits))
            return self.fn.const(ty, a.const)
        inner = a.op
        if k == 'trunc':
            if inner.kind in ('zext', 'sext'):
                src = inner.args[0]
                if src.ty is ty:
                    return src
                if src.ty.bits < ty.bits:
                    return self.new(op, inner.kind, [src], ty)
                return self.new(op, 'trunc', [src], ty)
            if inner.kind == 'trunc':
                return self.new(op, 'trunc', [inner.args[0]], ty)
            return None
        if k == 'zext' and inner.kind == 'zext':
            return self.new(op, 'zext', [inner.args[0]], ty)
        if k in ('zext', 'sext') and inner.kind == 'trunc' and inner.args[0].ty is ty:
            # the bits the truncation dropped are known zero (and the sign bit, for sext): the original
            z, _ = self.facts.bits(inner.args[0])
            dropped = ty.mask ^ (a.ty.mask if k == 'zext' else a.ty.mask >> 1)
            if dropped & ~z == 0:
                return inner.args[0]
        if k == 'sext' and inner.kind in ('sext', 'zext'):
            return self.new(op, inner.kind, [inner.args[0]], ty)
        if k == 'sext':
            z, _ = self.facts.bits(a)
            if z >> (a.ty.bits - 1) & 1:
                return self.new(op, 'zext', [a], ty)
        return None

    def icmp(self, op):
        pred = op.attrs['pred']
        a, b = op.args
        ty = a.ty
        if a.const is not None and b.const is not None:
            return self.fn.const(I1, int(compare(pred, a.const, b.const, ty.bits)))
        if a.const is not None:
            op.set_arg(0, b)
            op.set_arg(1, a)
            op.attrs['pred'] = SWAP[pred]
            self.requeue(op)
            return None
        if a is b:
            return self.fn.const(I1, int(pred in ('eq', 'ule', 'uge', 'sle', 'sge')))
        mask = ty.mask
        lo1, hi1 = self.facts.range(a)
        lo2, hi2 = self.facts.range(b)
        verdict = decide(pred, lo1, hi1, lo2, hi2)
        if verdict is not None:
            return self.fn.const(I1, int(verdict))
        cb = b.const
        inner = a.op
        if cb is not None:
            # unsigned comparisons with the ends of the range to equality
            if pred == 'ult' and cb == 1 or pred == 'ule' and cb == 0:
                return self.new(op, 'icmp', [a, b if cb == 0 else self.fn.const(ty, 0)], I1, pred='eq')
            if pred == 'ugt' and cb == 0 or pred == 'uge' and cb == 1:
                return self.new(op, 'icmp', [a, self.fn.const(ty, 0)], I1, pred='ne')
            # signed comparisons of values known nonnegative
            if pred in ('slt', 'sle', 'sgt', 'sge') and hi1 < 1 << (ty.bits - 1) and cb < 1 << (ty.bits - 1):
                op.attrs['pred'] = 'u' + pred[1:]
                self.requeue(op)
                return None
            if pred in ('eq', 'ne') and inner.kind == 'and' and inner.args[1].const is not None:
                m = inner.args[1].const
                x = inner.args[0]
                if cb == m and power_of_two(m):
                    # the bit set: the bit not clear
                    return self.new(op, 'icmp', [a, self.fn.const(ty, 0)], I1, pred=INVERT[pred])
                if cb == 0 and x.op.kind in ('lshr', 'shl') and x.op.args[1].const is not None and len(x.uses) == 1:
                    # a test of shifted bits: the bits themselves
                    n = x.op.args[1].const
                    if x.op.kind == 'lshr' and (m << n) <= mask:
                        return self.new(op, 'icmp', [self.new(op, 'and', [x.op.args[0], self.fn.const(ty, m << n)], ty), b], I1, pred=pred)
                    if x.op.kind == 'shl' and m & ((1 << n) - 1) == 0:
                        return self.new(op, 'icmp', [self.new(op, 'and', [x.op.args[0], self.fn.const(ty, m >> n)], ty), b], I1, pred=pred)
            if pred in ('eq', 'ne') and cb == 0 and inner.kind == 'xor' and inner.args[1].const is None and len(a.uses) == 1:
                return self.new(op, 'icmp', [inner.args[0], inner.args[1]], I1, pred=pred)   # x ^ y == 0: x == y
            if pred in ('eq', 'ne') and cb == 0 and inner.kind == 'lshr' and inner.args[1].const is not None and inner.args[1].const < ty.bits:
                # x >> n against zero: x against 2^n
                return self.new(op, 'icmp', [inner.args[0], self.fn.const(ty, 1 << inner.args[1].const)], I1, pred='ult' if pred == 'eq' else 'uge')
            if pred in ('eq', 'ne'):
                if inner.kind == 'zext':
                    src = inner.args[0]
                    if cb > src.ty.mask:
                        return self.fn.const(I1, int(pred == 'ne'))
                    if src.ty is I1:
                        if (cb == 0) == (pred == 'ne'):
                            return src
                        return self.new(op, 'xor', [src, self.fn.const(I1, 1)], I1)
                    return self.new(op, 'icmp', [src, self.fn.const(src.ty, cb)], I1, pred=pred)
                # (the op stays for its other uses otherwise, and its flags serve the compare)
                if inner.kind == 'xor' and inner.args[1].const is not None and len(a.uses) == 1:
                    return self.new(op, 'icmp', [inner.args[0], self.fn.const(ty, inner.args[1].const ^ cb)], I1, pred=pred)
                if inner.kind == 'add' and inner.args[1].const is not None and len(a.uses) == 1:
                    return self.new(op, 'icmp', [inner.args[0], self.fn.const(ty, cb - inner.args[1].const)], I1, pred=pred)
                if inner.kind == 'sub' and cb == 0 and len(a.uses) == 1:
                    return self.new(op, 'icmp', [inner.args[0], inner.args[1]], I1, pred=pred)
                z, o = self.facts.bits(a)
                if cb & z or ~cb & mask & o:
                    return self.fn.const(I1, int(pred == 'ne'))
            # C - x against K where x <= C: x against C - K, the other way round
            if inner.kind == 'sub' and inner.args[0].const is not None and pred in ('ult', 'ule', 'ugt', 'uge', 'eq', 'ne'):
                c = inner.args[0].const
                x = inner.args[1]
                if self.facts.range(x)[1] <= c and cb <= c:
                    flip = {'ult': 'ugt', 'ule': 'uge', 'ugt': 'ult', 'uge': 'ule', 'eq': 'eq', 'ne': 'ne'}[pred]
                    return self.new(op, 'icmp', [x, self.fn.const(ty, c - cb)], I1, pred=flip)
            # x + c against K with no wrap either way: x against K - c
            if inner.kind == 'add' and inner.args[1].const is not None and pred in ('ult', 'ule', 'ugt', 'uge'):
                c = inner.args[1].const
                x = inner.args[0]
                xlo, xhi = self.facts.range(x)
                if xhi + c <= mask and cb >= c:
                    return self.new(op, 'icmp', [x, self.fn.const(ty, cb - c)], I1, pred=pred)
            # zext x against K below x's width: x against K
            if inner.kind == 'zext' and cb <= inner.args[0].ty.mask and pred not in ('slt', 'sle', 'sgt', 'sge'):
                src = inner.args[0]
                return self.new(op, 'icmp', [src, self.fn.const(src.ty, cb)], I1, pred=pred)
        if pred in ('eq', 'ne') and a.op.kind == 'zext' and b.op.kind == 'zext' and a.op.args[0].ty is b.op.args[0].ty:
            return self.new(op, 'icmp', [a.op.args[0], b.op.args[0]], I1, pred=pred)
        if pred in ('ult', 'ule', 'ugt', 'uge') and a.op.kind == 'zext' and b.op.kind == 'zext' and a.op.args[0].ty is b.op.args[0].ty:
            return self.new(op, 'icmp', [a.op.args[0], b.op.args[0]], I1, pred=pred)
        # a not of a comparison feeding nothing else
        return None

    def vop(self, op):
        name = op.attrs['name']
        args = op.args
        if name == 'andnot' and args[0] is args[1]:
            return self.fn.const(op.ty, bytes(op.ty.size))
        if name == 'andnot' and args[0].const is not None and args[0].const == bytes(op.ty.size):
            return args[1]
        return None


def not_of_compare(fn):
    """xor (icmp p a b), 1 with one use -> icmp (not p) a b."""
    changed = False
    for b in fn.blocks:
        for op in list(b.ops):
            if op.kind == 'xor' and op.ty is I1 and op.args[1].const == 1:
                inner = op.args[0].op
                if inner.kind == 'icmp' and len(inner.result.uses) == 1:
                    inner.attrs['pred'] = INVERT[inner.attrs['pred']]
                    fn.replace_all_uses(op.result, inner.result)
                    op.detach()
                    b.remove(op)
                    changed = True
    return changed


def simplify(fn):
    changed = False
    while True:
        s = Simplifier(fn)
        step = s.run()
        step = not_of_compare(fn) or step
        if not step:
            return changed
        changed = True
