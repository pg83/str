"""What is known of a value without running the code: its known zero and one
bits, and its unsigned range. Computed on demand per value and memoized for
one pass; a phi that depends on itself is taken as unknown on that path."""
from .ir import PTR


def trailing_zeros(mask, bits):
    """How many low bits the known-zero mask covers."""
    n = 0
    while n < bits and mask >> n & 1:
        n += 1
    return n


class Facts:
    def __init__(self, fn, assumptions=None):
        self.fn = fn
        self.assume = assumptions or {}      # value -> (lo, hi) known on the path considered
        self.memo_bits, self.memo_range = {}, {}

    # ------------------------------------------------------------- bits

    def bits(self, v):
        """(known zero, known one) masks of v's bits."""
        r = self.memo_bits.get(v)
        if r is None:
            self.memo_bits[v] = (0, 0)
            r = self._bits(v)
            mask = v.ty.mask if v.ty.kind in ('i', 'p') else 0
            r = (r[0] & mask, r[1] & mask)
            self.memo_bits[v] = r
        return r

    def _bits(self, v):
        op, ty = v.op, v.ty
        if ty.kind not in ('i', 'p'):
            return 0, 0
        mask, n = ty.mask, ty.bits
        k = op.kind
        if v in self.assume:
            lo, hi = self.range(v)
            return mask ^ ((1 << hi.bit_length()) - 1), 0
        if k == 'const':
            c = op.attrs['value']
            return ~c & mask, c
        if k == 'alloca':
            a = op.attrs.get('align', 1)
            return a - 1, 0
        if k in ('and', 'or', 'xor'):
            z1, o1 = self.bits(op.args[0])
            z2, o2 = self.bits(op.args[1])
            if k == 'and':
                return z1 | z2, o1 & o2
            if k == 'or':
                return z1 & z2, o1 | o2
            return (z1 & z2) | (o1 & o2), (z1 & o2) | (o1 & z2)
        if k in ('shl', 'lshr', 'ashr'):
            c = op.args[1].const
            z, o = self.bits(op.args[0])
            if c is None or c >= n:
                if k == 'shl':
                    # x << s: as many low zeros as x has, at least
                    return (1 << trailing_zeros(z, n)) - 1, 0
                return 0, 0
            if k == 'shl':
                return ((z << c) | ((1 << c) - 1)) & mask, (o << c) & mask
            if k == 'lshr':
                high = mask ^ (mask >> c)
                return (z >> c) | high, o >> c
            sign = 1 << (n - 1)
            if z & sign:
                high = mask ^ (mask >> c)
                return (z >> c) | high, o >> c
            if o & sign:
                high = mask ^ (mask >> c)
                return z >> c, (o >> c) | high
            return z >> c, o >> c
        if k == 'zext':
            src = op.args[0]
            z, o = self.bits(src)
            return z | (mask ^ src.ty.mask), o
        if k == 'sext':
            src = op.args[0]
            z, o = self.bits(src)
            sign = 1 << (src.ty.bits - 1)
            ext = mask ^ src.ty.mask
            if z & sign:
                return z | ext, o
            if o & sign:
                return z, o | ext
            return z, o
        if k == 'trunc':
            return self.bits(op.args[0])
        if k in ('p2i', 'i2p'):
            return self.bits(op.args[0])
        if k in ('add', 'sub', 'ptradd'):
            z1, o1 = self.bits(op.args[0])
            z2, o2 = self.bits(op.args[1])
            tz = min(trailing_zeros(z1, n), trailing_zeros(z2, op.args[1].ty.bits))
            low = (1 << tz) - 1
            if k in ('add', 'ptradd'):
                lo, hi = self.range(v)
                if lo <= hi and hi != mask:
                    return low | (mask ^ ((1 << hi.bit_length()) - 1)), 0
            return low, 0
        if k == 'mul':
            z1, _ = self.bits(op.args[0])
            z2, _ = self.bits(op.args[1])
            tz = min(trailing_zeros(z1, n) + trailing_zeros(z2, n), n)
            return (1 << tz) - 1, 0
        if k in ('select', 'phi'):
            values = op.args[1:] if k == 'select' else op.args
            z, o = mask, mask
            for a in values:
                za, oa = self.bits(a)
                z, o = z & za, o & oa
            return z, o
        if k in ('cttz', 'urem', 'udiv') or k == 'vop':
            lo, hi = self.range(v)
            return mask ^ ((1 << hi.bit_length()) - 1), 0
        return 0, 0

    # ------------------------------------------------------------ range

    def range(self, v):
        """(lo, hi), unsigned, inclusive."""
        r = self.memo_range.get(v)
        if r is None:
            ty = v.ty
            full = (0, ty.mask) if ty.kind in ('i', 'p') else (0, 0)
            self.memo_range[v] = full
            r = self._range(v, full)
            a = self.assume.get(v)
            if a is not None:
                lo, hi = max(r[0], a[0]), min(r[1], a[1])
                if lo <= hi:
                    r = (lo, hi)
            self.memo_range[v] = r
        return r

    def _range(self, v, full):
        op, ty = v.op, v.ty
        if ty.kind not in ('i', 'p'):
            return full
        mask = ty.mask
        k = op.kind
        if k == 'const':
            c = op.attrs['value']
            return c, c
        lo, hi = full
        if k == 'param' and 'range' in op.attrs:
            lo, hi = op.attrs['range']
        if k == 'urem' and op.args[1].const:
            hi = op.args[1].const - 1
        elif k == 'cttz':
            hi = op.args[0].ty.bits - (1 if op.attrs.get('zero_undef') else 0)
        elif k == 'vop':
            name = op.attrs['name']
            if name in ('cmpistri',):
                hi = 16
            elif name in ('cmpistra', 'cmpistrc', 'cmpistrz', 'testz'):
                hi = 1
            elif name == 'movemask8':
                hi = (1 << (op.args[0].ty.bits // 8)) - 1
        elif k in ('zext',):
            lo, hi = self.range(op.args[0])
        elif k == 'trunc':
            a, b = self.range(op.args[0])
            if b <= mask:
                lo, hi = a, b
        elif k == 'and':
            hi = min(self.range(op.args[0])[1], self.range(op.args[1])[1])
        elif k == 'lshr' and op.args[1].const is not None and op.args[1].const < ty.bits:
            a, b = self.range(op.args[0])
            c = op.args[1].const
            lo, hi = a >> c, b >> c
        elif k == 'udiv' and op.args[1].const:
            a, b = self.range(op.args[0])
            lo, hi = a // op.args[1].const, b // op.args[1].const
        elif k in ('add', 'ptradd'):
            a1, b1 = self.range(op.args[0])
            a2, b2 = self.range(op.args[1])
            if b1 + b2 <= mask:
                lo, hi = a1 + a2, b1 + b2
        elif k == 'sub':
            a1, b1 = self.range(op.args[0])
            a2, b2 = self.range(op.args[1])
            if a1 >= b2:
                lo, hi = a1 - b2, b1 - a2
        elif k in ('select', 'phi'):
            values = op.args[1:] if k == 'select' else op.args
            ranges = [self.range(a) for a in values]
            lo, hi = min(r[0] for r in ranges), max(r[1] for r in ranges)
        elif k == 'or':
            a1, b1 = self.range(op.args[0])
            a2, b2 = self.range(op.args[1])
            lo, hi = max(a1, a2), (1 << max(b1, b2).bit_length()) - 1
        z, o = self.bits(v)
        lo, hi = max(lo, o), min(hi, mask & ~z)
        if lo > hi:
            return full
        return lo, hi

    def nonzero(self, v):
        return self.range(v)[0] > 0 or self.bits(v)[1] != 0

    def align(self, v):
        """The largest power of two v is known to be a multiple of (pointers)."""
        z, _ = self.bits(v)
        return 1 << trailing_zeros(z, v.ty.bits if v.ty is not PTR else 64)
