"""Bit-vector terms and a z3 session over SMT-LIB 2.

Terms are hash-consed (one object per distinct term), folded when their
arguments are constants, and sent to z3 once, each as a define-fun, the
first time an assertion needs them. Widths: n > 0 a bit-vector of n bits,
0 a Bool, -1 memory (an array from 64-bit addresses to bytes).
"""
import atexit
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

BOOL, MEM = 0, -1
SLOW = float(os.environ['PROVE_SLOW']) if os.environ.get('PROVE_SLOW') else None
PROFILE = os.environ.get('PROVE_PROFILE')  # a file: a line per query asked (seconds, answer, what for, where from)
INCREMENTAL = os.environ.get('PROVE_INCREMENTAL') == '1'
MEMORY = int(os.environ.get('PROVE_MEMORY', '6000'))  # megabytes z3 may take
TIMEOUT = int(os.environ.get('PROVE_TIMEOUT', '900'))  # seconds a query may take: then unknown, which no proof accepts
QTIMEOUT = int(os.environ.get('PROVE_QTIMEOUT', '120'))  # a query with quantifiers whose part without them was not unsat


class T:
    __slots__ = ('op', 'w', 'args', 'val', 'id')

    def __repr__(self):
        return 'T(%s/%d%s)' % (self.op, self.w, '=%#x' % self.val if self.op in ('const', 'bconst') else '')


_table = {}
_next = [0]
_made = threading.Lock()  # the prover's checking threads build terms too: one term per key, whoever asks


def _mk(op, w, args=(), val=None):
    key = (op, w, args, val)
    t = _table.get(key)
    if t is None:
        with _made:
            t = _table.get(key)
            if t is None:
                t = T()
                t.op, t.w, t.args, t.val = op, w, args, val
                t.id = _next[0]
                _next[0] += 1
                _table[key] = t
    return t


def mask(w):
    return (1 << w) - 1


def const(w, v):
    return _mk('const', w, (), v & mask(w))


def var(name, w):
    return _mk('var', w, (name,))


TRUE = _mk('bconst', BOOL, (), 1)
FALSE = _mk('bconst', BOOL, (), 0)


def boolean(b):
    return TRUE if b else FALSE


def is_const(t):
    return t.op in ('const', 'bconst')


def signed(t):
    v = t.val
    return v - (1 << t.w) if v >> (t.w - 1) else v


# ------------------------------------------------------------- known bits

def low_zeros(t):
    """How many low bits of t are known to be zero."""
    if t.op == 'const':
        v = t.val
        return t.w if v == 0 else (v & -v).bit_length() - 1
    if t.op == 'concat':
        n = 0
        for p in reversed(t.args):
            z = low_zeros(p)
            n += z
            if z < p.w:
                break
        return n
    if t.op == 'add':
        return min(low_zeros(t.args[0]), low_zeros(t.args[1]))
    if t.op == 'and':
        return max(low_zeros(t.args[0]), low_zeros(t.args[1]))
    if t.op == 'mul':
        return min(t.w, low_zeros(t.args[0]) + low_zeros(t.args[1]))
    if t.op == 'neg':
        return low_zeros(t.args[0])
    if t.op == 'shl' and t.args[1].op == 'const':
        return min(t.w, low_zeros(t.args[0]) + t.args[1].val)
    if t.op == 'extract':
        return max(0, min(t.w, low_zeros(t.args[0]) - t.args[2]))
    if t.op in ('zext', 'sext'):
        return low_zeros(t.args[0])
    return 0


def low_bits(t, j):
    """t's low j bits as a number when they are known, else None."""
    if j == 0:
        return 0
    if t.op == 'const':
        return t.val & mask(j)
    if t.op == 'add':
        a, b = t.args
        la, lb = low_bits(a, j), low_bits(b, j)
        return None if la is None or lb is None else (la + lb) & mask(j)
    if t.op == 'concat':
        last = t.args[-1]
        if j <= last.w:
            return low_bits(last, j)
        lo = low_bits(last, last.w)
        hi = low_bits(_mk('concat', t.w - last.w, t.args[:-1]) if len(t.args) > 2 else t.args[0], j - last.w)
        return None if lo is None or hi is None else hi << last.w | lo
    if t.op in ('extract',) and t.args[2] == 0 and j <= t.w:
        return low_bits(t.args[0], j)
    if t.op in ('zext', 'sext') and j <= t.args[0].w:
        return low_bits(t.args[0], j)
    if t.op == 'and':
        la, lb = low_bits(t.args[0], j), low_bits(t.args[1], j)
        if la is not None and lb is not None:
            return la & lb
        if la == 0 or lb == 0:
            return 0
        return None
    if t.op == 'neg':
        la = low_bits(t.args[0], j)
        return None if la is None else -la & mask(j)
    if t.op == 'sub':
        la, lb = low_bits(t.args[0], j), low_bits(t.args[1], j)
        return None if la is None or lb is None else (la - lb) & mask(j)
    z = low_zeros(t)
    return 0 if z >= j else None


def addends_of(t):
    if t.op == 'add':
        return addends_of(t.args[0]) + addends_of(t.args[1])
    return [t]


def split(t):
    """t as (base, constant): a sum's constant part apart."""
    if t.op == 'add' and t.args[1].op == 'const':
        return t.args[0], t.args[1].val
    if t.op == 'const':
        return None, t.val
    return t, 0


# ------------------------------------------------------------- bit-vectors

_lin = {}


def linear(t):
    """t as a linear form: ({atom: coefficient}, constant), modulo 2^w;
    sums, differences, negations and products with constants unfolded."""
    r = _lin.get(t.id)
    if r is not None:
        return r
    m = mask(t.w)
    if t.op == 'const':
        r = ({}, t.val)
    elif t.op in ('add', 'sub'):
        (fa, ca), (fb, cb) = linear(t.args[0]), linear(t.args[1])
        k = 1 if t.op == 'add' else m
        f = dict(fa)
        for x, v in fb.items():
            f[x] = (f.get(x, 0) + k * v) & m
        r = ({x: v for x, v in f.items() if v}, (ca + k * cb) & m)
    elif t.op == 'neg':
        fa, ca = linear(t.args[0])
        r = ({x: (-v) & m for x, v in fa.items()}, (-ca) & m)
    elif t.op == 'mul' and is_const(t.args[1]):
        fa, ca = linear(t.args[0])
        k = t.args[1].val
        f = {x: (v * k) & m for x, v in fa.items()}
        r = ({x: v for x, v in f.items() if v}, (ca * k) & m)
    else:
        r = ({t: 1}, 0)
    _lin[t.id] = r
    return r


def from_linear(w, f, c):
    """The canonical term of a linear form: its atoms by id, each times its
    coefficient (a negation for -1), the constant last."""
    m = mask(w)
    terms = []
    for x in sorted(f, key=lambda x: x.id):
        v = f[x] & m
        if v == 1:
            terms.append(x)
        elif v == m:
            terms.append(_mk('neg', w, (x,)))
        else:
            terms.append(_mk('mul', w, (x, const(w, v))))
    if not terms:
        return const(w, c)
    r = terms[0]
    for x in terms[1:]:
        r = _mk('add', w, (r, x))
    return _mk('add', w, (r, const(w, c))) if c & m else r


def _combine(a, b, k):
    (fa, ca), (fb, cb) = linear(a), linear(b)
    m = mask(a.w)
    f = dict(fa)
    for x, v in fb.items():
        f[x] = (f.get(x, 0) + k * v) & m
    return from_linear(a.w, {x: v for x, v in f.items() if v}, (ca + k * cb) & m)


def add(a, b):
    assert a.w == b.w > 0
    if is_const(a) and is_const(b):
        return const(a.w, a.val + b.val)
    return _combine(a, b, 1)


def neg(a):
    if is_const(a):
        return const(a.w, -a.val)
    f, c = linear(a)
    m = mask(a.w)
    return from_linear(a.w, {x: (-v) & m for x, v in f.items()}, (-c) & m)


def sub(a, b):
    assert a.w == b.w > 0
    if a is b:
        return const(a.w, 0)
    if is_const(a) and is_const(b):
        return const(a.w, a.val - b.val)
    return _combine(a, b, mask(a.w))


def mul(a, b):
    assert a.w == b.w > 0
    if is_const(a) and is_const(b):
        return const(a.w, a.val * b.val)
    if is_const(a):
        a, b = b, a
    if is_const(b) and b.val == 1:
        return a
    if is_const(b) and b.val == 0:
        return b
    if is_const(b):  # a linear form scaled
        f, c = linear(a)
        m = mask(a.w)
        f = {x: (v * b.val) & m for x, v in f.items()}
        return from_linear(a.w, {x: v for x, v in f.items() if v}, (c * b.val) & m)
    return _mk('mul', a.w, (a, b))


def low_part(t, j):
    """A term with t's low j bits, without the parts that can not touch
    them: the addends with j low zeros (an aligned base: (B + n) & 15 is
    n & 15), through extracts and zero extensions of the low end."""
    if j <= 0:
        return const(t.w, 0)
    if t.op in ('add', 'sub', 'neg') or t.op == 'mul' and is_const(t.args[1]):
        f, c = linear(t)
        keep = {x: v for x, v in f.items() if (v & -v).bit_length() - 1 + low_zeros(x) < j}
        if len(keep) < len(f):
            return from_linear(t.w, keep, c)
        return t
    if t.op == 'extract' and t.args[2] == 0 and j <= t.w:
        x = low_part(t.args[0], j)
        return t if x is t.args[0] else extract(x, t.args[1], 0)
    if t.op == 'zext' and j <= t.args[0].w:
        x = low_part(t.args[0], j)
        return t if x is t.args[0] else zext(x, t.w)
    return t


def band(a, b):
    assert a.w == b.w > 0, (a, b)
    if is_const(a) and is_const(b):
        return const(a.w, a.val & b.val)
    if is_const(a):
        a, b = b, a
    if is_const(b):
        if b.val == 0:
            return b
        if b.val == mask(a.w):
            return a
        base, c = split(a)
        if base is not None:
            k = low_zeros(base)
            tz = (b.val & -b.val).bit_length() - 1  # the mask's trailing zeros
            if (~b.val & mask(a.w)) == mask(tz) and tz <= k:
                return add(base, const(a.w, c & b.val))  # (B + c) & ~(2^j - 1), B's low j bits zero
            j = b.val.bit_length()
            if b.val == mask(j) and j <= k:
                return const(a.w, c & b.val)  # (B + c) & (2^j - 1): c's low bits
        j = b.val.bit_length()
        if b.val == mask(j) and a.op == 'concat':  # the parts above the mask's bits go
            return concat(const(a.w - j, 0), extract(a, j - 1, 0))
        if b.val == mask(j):
            lb = low_bits(a, j)
            if lb is not None:
                return const(a.w, lb)  # the low bits known
            lp = low_part(a, j)
            if lp is not a:
                return band(lp, b)
    if a is b:
        return a
    return _mk('and', a.w, (a, b))


def bor(a, b):
    assert a.w == b.w > 0
    if is_const(a) and is_const(b):
        return const(a.w, a.val | b.val)
    if is_const(a):
        a, b = b, a
    if is_const(b):
        if b.val == 0:
            return a
        if b.val == mask(a.w):
            return b
    if a is b:
        return a
    return _mk('or', a.w, (a, b))


def bxor(a, b):
    assert a.w == b.w > 0
    if is_const(a) and is_const(b):
        return const(a.w, a.val ^ b.val)
    if a is b:
        return const(a.w, 0)
    if is_const(a):
        a, b = b, a
    if is_const(b) and b.val == 0:
        return a
    return _mk('xor', a.w, (a, b))


def bnot(a):
    if is_const(a):
        return const(a.w, ~a.val)
    if a.op == 'not':
        return a.args[0]
    return _mk('not', a.w, (a,))


def _shift(op, a, b, f):
    assert a.w == b.w > 0
    if is_const(b):
        if b.val >= a.w:
            return const(a.w, 0) if op != 'ashr' else ite(slt(a, const(a.w, 0)), const(a.w, -1), const(a.w, 0))
        if b.val == 0:
            return a
        if is_const(a):
            return const(a.w, f(a, b.val))
    return _mk(op, a.w, (a, b))


def shl(a, b):
    return _shift('shl', a, b, lambda a, n: a.val << n)


def lshr(a, b):
    return _shift('lshr', a, b, lambda a, n: a.val >> n)


def ashr(a, b):
    return _shift('ashr', a, b, lambda a, n: signed(a) >> n)


def extract(a, hi, lo):
    assert 0 <= lo <= hi < a.w, (a, hi, lo)
    if lo == 0 and hi == a.w - 1:
        return a
    if is_const(a):
        return const(hi - lo + 1, a.val >> lo)
    if a.op == 'extract':
        return extract(a.args[0], hi + a.args[2], lo + a.args[2])
    if a.op == 'concat':  # the pieces of the parts it spans, most significant first
        pos, pieces = a.w, []
        for p in a.args:
            pos -= p.w
            plo, phi = max(lo, pos), min(hi, pos + p.w - 1)
            if plo <= phi:
                pieces.append(extract(p, phi - pos, plo - pos))
        return concat(*pieces)
    if a.op in ('lshr', 'shl') and is_const(a.args[1]):  # a shift by a constant: the bits it moves there (psrlw's
        x, c = a.args[0], a.args[1].val                  # bytes then each of their own byte, not of the word)
        if a.op == 'lshr':
            if lo + c >= a.w:
                return const(hi - lo + 1, 0)
            if hi + c < a.w:
                return extract(x, hi + c, lo + c)
            return zext(extract(x, a.w - 1, lo + c), hi - lo + 1)
        if hi < c:
            return const(hi - lo + 1, 0)
        if lo >= c:
            return extract(x, hi - c, lo - c)
        return concat(extract(x, hi - c, 0), const(c - lo, 0))
    if a.op == 'zext':
        x = a.args[0]
        if hi < x.w:
            return extract(x, hi, lo)
        if lo >= x.w:
            return const(hi - lo + 1, 0)
    base, c = split(a)
    if base is not None and hi < low_zeros(base):
        return const(hi - lo + 1, c >> lo)  # the bits of B + c below B's known zeros are c's
    lb = low_bits(a, hi + 1)
    if lb is not None:
        return const(hi - lo + 1, lb >> lo)
    return _mk('extract', hi - lo + 1, (a, hi, lo))


def concat(*parts):
    """Most significant part first."""
    flat = []
    for p in parts:
        assert p.w > 0
        if p.op == 'concat':
            flat.extend(p.args)
        else:
            flat.append(p)
    merged = []
    for p in flat:
        if merged and is_const(merged[-1]) and is_const(p):
            q = merged.pop()
            merged.append(const(q.w + p.w, q.val << p.w | p.val))
        elif merged and p.op == 'extract' and merged[-1].op == 'extract' and merged[-1].args[0] is p.args[0] and merged[-1].args[2] == p.args[1] + 1:
            q = merged.pop()
            merged.append(extract(p.args[0], q.args[1], p.args[2]))
        else:
            merged.append(p)
    if len(merged) == 1:
        return merged[0]
    return _mk('concat', sum(p.w for p in merged), tuple(merged))


def zext(a, w):
    assert w >= a.w
    if w == a.w:
        return a
    if is_const(a):
        return const(w, a.val)
    if a.op == 'zext':
        return zext(a.args[0], w)
    return _mk('zext', w, (a,))


def sext(a, w):
    assert w >= a.w
    if w == a.w:
        return a
    if is_const(a):
        return const(w, signed(a))
    return _mk('sext', w, (a,))


def ite(c, a, b):
    assert c.w == BOOL and a.w == b.w
    if c is TRUE:
        return a
    if c is FALSE:
        return b
    if a is b:
        return a
    return _mk('ite', a.w, (c, a, b))


def bit(c):
    """A Bool as a 1-bit vector."""
    return ite(c, const(1, 1), const(1, 0))


# ------------------------------------------------------------- predicates

def eq(a, b):
    assert a.w == b.w, (a, b)
    if a is b:
        return TRUE
    if is_const(a) and is_const(b):
        return boolean(a.val == b.val)
    if is_const(a):
        a, b = b, a
    if a.op == 'ite' and is_const(b) and b.val == 0 and a.args[0].op in ('ult', 'ule') and a.args[0].args == (a.args[1], a.args[2]):
        return or_(eq(a.args[1], b), eq(a.args[2], b))  # an unsigned minimum is zero when one of the two is
    if a.op == 'ite' and is_const(b) and is_const(a.args[1]) and is_const(a.args[2]) and a.args[1].val != a.args[2].val:
        if a.args[1].val == b.val:
            return a.args[0]
        if a.args[2].val == b.val:
            return not_(a.args[0])
        return FALSE
    if a.op == 'ite' and is_const(b) and is_const(a.args[1]):  # a lane ite(g, 0, y) tested against a constant
        if a.args[1].val == b.val:
            return or_(a.args[0], eq(a.args[2], b))
        return and_(not_(a.args[0]), eq(a.args[2], b))
    return _mk('eq', BOOL, (a, b))


def ult(a, b):
    if is_const(a) and is_const(b):
        return boolean(a.val < b.val)
    if a is b or is_const(b) and b.val == 0:
        return FALSE
    return _mk('ult', BOOL, (a, b))


def ule(a, b):
    if is_const(a) and is_const(b):
        return boolean(a.val <= b.val)
    if a is b or is_const(a) and a.val == 0:
        return TRUE
    return _mk('ule', BOOL, (a, b))


def slt(a, b):
    if is_const(a) and is_const(b):
        return boolean(signed(a) < signed(b))
    if a is b:
        return FALSE
    return _mk('slt', BOOL, (a, b))


def sle(a, b):
    if is_const(a) and is_const(b):
        return boolean(signed(a) <= signed(b))
    if a is b:
        return TRUE
    return _mk('sle', BOOL, (a, b))


def msb(a):
    return eq(extract(a, a.w - 1, a.w - 1), const(1, 1))


def is_zero(a):
    return eq(a, const(a.w, 0))


def not_(a):
    if a is TRUE:
        return FALSE
    if a is FALSE:
        return TRUE
    if a.op == 'not_b':
        return a.args[0]
    return _mk('not_b', BOOL, (a,))


def and_(*xs):
    out = []
    for x in xs:
        if x is FALSE:
            return FALSE
        if x is TRUE:
            continue
        if x.op == 'and_b':
            out.extend(x.args)
        else:
            out.append(x)
    if not out:
        return TRUE
    if len(out) == 1:
        return out[0]
    return _mk('and_b', BOOL, tuple(out))


def or_(*xs):
    out = []
    for x in xs:
        if x is TRUE:
            return TRUE
        if x is FALSE:
            continue
        if x.op == 'or_b':
            out.extend(x.args)
        else:
            out.append(x)
    if not out:
        return FALSE
    if len(out) == 1:
        return out[0]
    return _mk('or_b', BOOL, tuple(out))


# ------------------------------------------------------------- memory

_iv = {}


def interval(t):
    """(lo, hi): t's value as a signed number lies there, or None. Constants,
    zero extensions, masks and shifts by constants bound their values; sums
    add the bounds times the coefficients (a table's index c + a byte: [c,
    c + 255]; a bit table's qword c + 8 (x >> 7 << 1 | x >> 3 & 1): [c, c + 24])."""
    r = _iv.get(t.id, 0)
    if r != 0:
        return r
    r = None
    lim = 1 << (t.w - 2)
    if t.op == 'const':
        r = (signed(t), signed(t))
    elif t.op == 'zext':
        ix = interval(t.args[0]) if t.args[0].w > 1 else None
        r = ix if ix is not None and ix[0] >= 0 else (0, (1 << t.args[0].w) - 1)
    elif t.op == 'and' and is_const(t.args[1]) and t.args[1].val < lim:
        r = (0, t.args[1].val)
    elif t.op == 'extract' and t.args[2] == 0:  # the low bits: the value when it fits them
        ix = interval(t.args[0])
        if ix is not None and 0 <= ix[0] and ix[1] < lim:
            r = ix
    elif t.op == 'mul' and is_const(t.args[1]):
        ix, k = interval(t.args[0]), signed(t.args[1])
        if ix is not None:
            lo, hi = sorted((ix[0] * k, ix[1] * k))
            r = (lo, hi) if -lim < lo <= hi < lim else None
    elif t.op == 'lshr' and is_const(t.args[1]):
        ix, s = interval(t.args[0]), t.args[1].val
        if ix is not None and ix[0] >= 0:
            r = (ix[0] >> s, ix[1] >> s)
        elif s >= 2:
            r = (0, ((1 << t.w) - 1) >> s)
    elif t.op == 'shl' and is_const(t.args[1]):
        ix, s = interval(t.args[0]), t.args[1].val
        if ix is not None and -lim < ix[0] << s and ix[1] << s < lim:
            r = (ix[0] << s, ix[1] << s)
    elif t.op in ('add', 'sub', 'neg'):
        f, c = linear(t)
        lo = hi = signed(const(t.w, c))
        for x, v in f.items():
            k = signed(const(t.w, v))
            ix = interval(x)
            if ix is None:
                break
            lo, hi = (lo + k * ix[0], hi + k * ix[1]) if k > 0 else (lo + k * ix[1], hi + k * ix[0])
        else:
            r = (lo, hi) if -lim < lo <= hi < lim else None
    _iv[t.id] = r
    return r


def select(m, addr):
    assert m.w == MEM and addr.w == 64
    while m.op == 'store':  # past the stores at a known distance from addr, or bounded clear of it
        base, a, v = m.args
        if a is addr:
            return v
        d = sub(a, addr)
        if is_const(d):
            m = base
            continue
        ia, ib = interval(a), interval(addr)
        if ia is not None and ib is not None and (ia[1] < ib[0] or ib[1] < ia[0]):
            m = base
            continue
        under = base
        while under.op == 'store':
            under = under.args[0]
        if under.op in ('win', 'fill'):  # over a window: the stores as ites, so no lambda reaches the solver
            return ite(eq(a, addr), v, select(base, addr))
        break
    if m.op in ('win', 'fill'):  # read through: no lambda reaches the solver for it
        inside = and_(ule(m.args[1], addr), ult(addr, m.args[2]))
        return ite(inside, select(m.args[3], addr) if m.op == 'win' else m.args[3], select(m.args[0], addr))
    return _mk('select', 8, (m, addr))


def store(m, addr, v):
    assert m.w == MEM and addr.w == 64 and v.w == 8
    return _mk('store', MEM, (m, addr, v))


def mem_var(name):
    return _mk('var', MEM, (name,))


def lam(x, body):
    """The array whose element at index x (a 64-bit var) is body."""
    assert x.op == 'var' and x.w == 64 and body.w == 8
    return _mk('lam', MEM, (x, body))


def forall_(x, body):
    """For every value of the variable x, body (a Bool)."""
    assert x.op == 'var' and body.w == BOOL
    if x.args[0] not in variables(body):
        return body
    return _mk('forall', BOOL, (x, body))


def exists_(x, body):
    assert x.op == 'var' and body.w == BOOL
    if x.args[0] not in variables(body):
        return body
    return _mk('exists', BOOL, (x, body))


BINDERS = ('lam', 'forall', 'exists')


def win(m, lo, hi, h):
    """m but in [lo, hi), there h's bytes (a loop's general memory)."""
    assert m.w == MEM and h.w == MEM
    return _mk('win', MEM, (m, lo, hi, h))


def fill(m, lo, hi, v):
    """m with every index in [lo, hi) holding v (rep stosb)."""
    assert m.w == MEM and lo.w == hi.w == 64 and v.w == 8
    return _mk('fill', MEM, (m, lo, hi, v))


def subst(t, old, new):
    """t with the term old in it replaced by new, node for node (the same
    shape: no folding, so two lanes the same code computed come out the
    same term)."""
    memo = {}

    def go(x):
        if x is old:
            return new
        r = memo.get(x.id)
        if r is None:
            args = tuple(go(y) if isinstance(y, T) else y for y in x.args)
            r = x if all(a is b for a, b in zip(args, x.args)) else _mk(x.op, x.w, args, x.val)
            memo[x.id] = r
        return r
    return go(t)


def replace(terms, mapping):
    """The terms with every subterm whose id mapping has replaced by its
    term there, node for node; the same list (each term itself) when none
    is in them."""
    memo = {}
    out = []
    for t in terms:
        stack = [t]
        while stack:
            x = stack[-1]
            if x.id in memo:
                stack.pop()
                continue
            if x.id in mapping:
                memo[x.id] = mapping[x.id]
                stack.pop()
                continue
            pending = [y for y in x.args if isinstance(y, T) and y.id not in memo]
            if pending:
                stack.extend(pending)
                continue
            stack.pop()
            args = tuple(memo[y.id] if isinstance(y, T) else y for y in x.args)
            memo[x.id] = x if all(a is b for a, b in zip(args, x.args)) else _mk(x.op, x.w, args, x.val)
        out.append(memo[t.id])
    return out


_vars = {}


def variables(t):
    """The names of the variables a term depends on."""
    r = _vars.get(t.id)
    if r is None:
        if t.op == 'var':
            r = frozenset(t.args[:1])
        else:
            r = frozenset()
            for x in t.args:
                if isinstance(x, T):
                    r = r | variables(x)
        _vars[t.id] = r
    return r


# ------------------------------------------------------------- SMT-LIB

def sort(w):
    if w == BOOL:
        return 'Bool'
    if w == MEM:
        return '(Array (_ BitVec 64) (_ BitVec 8))'
    return '(_ BitVec %d)' % w


BIN = {'add': 'bvadd', 'sub': 'bvsub', 'mul': 'bvmul', 'and': 'bvand', 'or': 'bvor', 'xor': 'bvxor', 'shl': 'bvshl', 'lshr': 'bvlshr',
       'ashr': 'bvashr', 'ult': 'bvult', 'ule': 'bvule', 'slt': 'bvslt', 'sle': 'bvsle', 'eq': '='}


def _die_with_parent():
    """z3 goes when the prover does (a killed run left it on a long query)."""
    try:
        import ctypes
        ctypes.CDLL(None).prctl(1, 9)  # PR_SET_PDEATHSIG, SIGKILL
    except Exception:
        pass


_children = []


def _reap(*_):
    for p in _children:
        try:
            p.kill()
        except Exception:
            pass


def _on_term(sig, frame):
    """A static python has no prctl: the solvers killed on SIGTERM (kill,
    timeout) and at exit."""
    _reap()
    os._exit(128 + sig)


_lam = {}


def has_lambda(t):
    """Whether t has an array written as a lambda (rep movs and stos, a
    loop's window, a contract's copy): bitwuzla does not read those."""
    r = _lam.get(t.id)
    if r is None:
        r = t.op in ('lam', 'fill', 'win', 'forall', 'exists') or any(isinstance(x, T) and has_lambda(x) for x in t.args)
        _lam[t.id] = r
    return r


_quant = {}


def has_quantifier(t):
    r = _quant.get(t.id)
    if r is None:
        r = t.op in ('forall', 'exists') or any(isinstance(x, T) and has_quantifier(x) for x in t.args)
        _quant[t.id] = r
    return r


_arr = {}


def has_array(t):
    """Whether t reads memory."""
    r = _arr.get(t.id)
    if r is None:
        r = t.w == MEM or any(isinstance(x, T) and has_array(x) for x in t.args)
        _arr[t.id] = r
    return r


_digests = {}


def digest(t):
    """A term by its structure: the same in any run, whatever its id there."""
    stack = [t]
    while stack:
        x = stack[-1]
        if x.id in _digests:
            stack.pop()
            continue
        pending = [a for a in x.args if isinstance(a, T) and a.id not in _digests]
        if pending:
            stack.extend(pending)
            continue
        stack.pop()
        word = repr((x.op, x.w, x.val, [_digests[a.id] if isinstance(a, T) else a for a in x.args]))
        _digests[x.id] = hashlib.blake2b(word.encode(), digest_size=16).digest()
    return _digests[t.id]


class Answers:
    """PROVE_CACHE=dir: what the solvers answered, by the query (the
    structure of its conditions, in any order), kept across runs in
    dir/answers.txt, a line each: a proof run again (a connection lost, a
    change to the prover that leaves these queries as they were) asks the
    solvers only what they were not asked. sat and unsat are answers to the
    query whatever the timeout; unknown with the seconds it was given, the
    answer again to a query given no more (a run again does not wait out
    each way that did not settle the first time); a sat whose model is
    wanted is asked again. Terms hold their variables' names, the
    assumptions are among the conditions: a line says nothing beyond its
    query, in any function, on any ELF."""

    def __init__(self, path):
        os.makedirs(path, exist_ok=True)
        self.path = os.path.join(path, 'answers.txt')
        self.known, self.hits = {}, 0
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    word = line.split()
                    if len(word) == 2 and word[1] in ('sat', 'unsat'):  # a last line cut short: nothing
                        self.known[word[0]] = word[1]
                    elif len(word) == 3 and word[1] == 'unknown' and word[2].isdigit() and self.known.get(word[0], 'unknown').startswith('unknown'):
                        self.known[word[0]] = 'unknown %s' % max(int(word[2]), int(self.known.get(word[0], 'unknown 0').split()[1]))
        self.out = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)

    def key(self, conds):
        return hashlib.blake2b(b''.join(sorted({digest(c) for c in conds})), digest_size=16).hexdigest()

    def add(self, key, answer, limit):
        if answer == 'unknown':
            old = self.known.get(key)
            if old is None or old.startswith('unknown') and int(old.split()[1]) < limit:
                self.known[key] = 'unknown %d' % limit
                os.write(self.out, ('%s unknown %d\n' % (key, limit)).encode())
        elif answer in ('sat', 'unsat') and self.known.get(key, 'unknown').startswith('unknown'):
            self.known[key] = answer
            os.write(self.out, ('%s %s\n' % (key, answer)).encode())  # one write: whole among the other processes'


ANSWERS = None  # opened with the first solver: run.py imports this for its own work, with the hosts' PROVE_CACHE


class Solver:
    """bitwuzla, and z3 for what bitwuzla does not read (lambdas): each query
    in a context of its own (PROVE_INCREMENTAL=1: one z3 session, the
    definitions at the base level, assertions under push/pop). bitwuzla is
    what makes the proofs go: on the queries z3 took 40 to over 300 s over
    it took 1.4 to 69 (PROVE_BITWUZLA: its path; without it, z3 alone)."""

    def __init__(self, z3='z3', log=None):
        global ANSWERS
        if ANSWERS is None and os.environ.get('PROVE_CACHE') and not INCREMENTAL:
            try:
                ANSWERS = Answers(os.environ['PROVE_CACHE'])
            except OSError as e:  # a place of no writes: every query asked
                print('PROVE_CACHE: %s' % e, file=sys.stderr, flush=True)
                os.environ.pop('PROVE_CACHE')
        # a memory limit: a query that would take more answers unknown (16 parallel jobs on a lab host fit)
        self.zp = subprocess.Popen([z3, '-in', '-memory:%d' % MEMORY], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                   preexec_fn=_die_with_parent)
        bw = os.environ.get('PROVE_BITWUZLA') or shutil.which('bitwuzla')
        self.bp = None
        if bw and not INCREMENTAL:
            self.bp = subprocess.Popen([bw, '-M', str(MEMORY)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                       preexec_fn=_die_with_parent)
        self.p = self.zp
        if not _children:
            signal.signal(signal.SIGTERM, _on_term)
            atexit.register(_reap)
        _children.extend(p for p in (self.zp, self.bp) if p is not None)
        self.done = set()
        self.log = log
        self.queries = 0
        self.assumed, self.slow = [], 0
        self.since, self.doing = None, None  # the running query's start, and what the prover asks it for (its status lines)
        self.send('(set-option :print-success false)')
        self.send('(set-option :timeout %d)' % (TIMEOUT * 1000))

    def send(self, s):
        if self.log:
            self.log.write(s + '\n')
        self.p.stdin.write(s + '\n')

    def name(self, t):
        if t.op == 'const':
            return '(_ bv%d %d)' % (t.val, t.w)
        if t.op == 'bconst':
            return 'true' if t.val else 'false'
        if t.op == 'var':
            return t.args[0]
        return 't%d' % t.id

    def free_parts(self, t, xs, out, seen):
        """The largest subterms of t that depend on none of the bound
        variables xs (a binder inside adds its own)."""
        key = (t.id, xs)
        if key in seen:
            return
        seen.add(key)
        if not xs & variables(t):
            out.append(t)
            return
        if t.op in BINDERS:
            self.free_parts(t.args[1], xs | {t.args[0].args[0]}, out, seen)
            return
        for a in t.args:
            if isinstance(a, T):
                self.free_parts(a, xs, out, seen)

    def inline(self, t, xs):
        if not xs & variables(t):
            return self.name(t)
        if t.op == 'var':
            return t.args[0]
        if t.op in BINDERS:
            return self.binder(t, xs)
        return self.expr(t, lambda a: self.inline(a, xs))

    def binder(self, t, xs):
        x = t.args[0]
        body = self.inline(t.args[1], xs | {x.args[0]})
        word = {'lam': 'lambda', 'forall': 'forall', 'exists': 'exists'}[t.op]
        return '(%s ((%s %s)) %s)' % (word, x.args[0], sort(x.w), body)

    def define(self, root):
        stack = [root]
        while stack:
            t = stack[-1]
            if t.id in self.done or t.op in ('const', 'bconst'):
                stack.pop()
                continue
            if t.op in BINDERS:  # the body's parts free of the bound variables first; the rest goes inline
                parts = []
                self.free_parts(t.args[1], frozenset({t.args[0].args[0]}), parts, set())
                pending = [a for a in parts if a.id not in self.done and a.op not in ('const', 'bconst')]
                if pending:
                    stack.extend(pending)
                    continue
                stack.pop()
                self.done.add(t.id)
                self.send('(define-fun t%d () %s %s)' % (t.id, sort(t.w), self.binder(t, frozenset())))
                continue
            pending = [a for a in t.args if isinstance(a, T) and a.id not in self.done and a.op not in ('const', 'bconst')]
            if pending:
                stack.extend(pending)
                continue
            stack.pop()
            self.done.add(t.id)
            if t.op == 'var':
                self.send('(declare-const %s %s)' % (t.args[0], sort(t.w)))
            else:
                self.send('(define-fun t%d () %s %s)' % (t.id, sort(t.w), self.expr(t)))

    def expr(self, t, n=None):
        n = n or self.name
        a = t.args
        if t.op in BIN:
            return '(%s %s %s)' % (BIN[t.op], n(a[0]), n(a[1]))
        if t.op == 'neg':
            return '(bvneg %s)' % n(a[0])
        if t.op == 'not':
            return '(bvnot %s)' % n(a[0])
        if t.op == 'extract':
            return '((_ extract %d %d) %s)' % (a[1], a[2], n(a[0]))
        if t.op == 'concat':
            return '(concat %s)' % ' '.join(n(p) for p in a)
        if t.op == 'zext':
            return '((_ zero_extend %d) %s)' % (t.w - a[0].w, n(a[0]))
        if t.op == 'sext':
            return '((_ sign_extend %d) %s)' % (t.w - a[0].w, n(a[0]))
        if t.op == 'ite':
            return '(ite %s %s %s)' % (n(a[0]), n(a[1]), n(a[2]))
        if t.op == 'not_b':
            return '(not %s)' % n(a[0])
        if t.op == 'and_b':
            return '(and %s)' % ' '.join(n(x) for x in a)
        if t.op == 'or_b':
            return '(or %s)' % ' '.join(n(x) for x in a)
        if t.op == 'select':
            return '(select %s %s)' % (n(a[0]), n(a[1]))
        if t.op == 'store':
            return '(store %s %s %s)' % (n(a[0]), n(a[1]), n(a[2]))
        if t.op == 'fill':
            return '(lambda ((x (_ BitVec 64))) (ite (and (bvule %s x) (bvult x %s)) %s (select %s x)))' % (n(a[1]), n(a[2]), n(a[3]), n(a[0]))
        if t.op == 'win':
            return '(lambda ((x (_ BitVec 64))) (ite (and (bvule %s x) (bvult x %s)) (select %s x) (select %s x)))' % (n(a[1]), n(a[2]), n(a[3]), n(a[0]))
        raise ValueError(t.op)

    def check(self, *conds, keep=False, timeout=None, bare=False, loose=False, qtimeout=QTIMEOUT, abstract=None):
        """abstract: {id: term}, subterms the query is first asked with
        replaced (a spec's predicate by an uninterpreted one: any
        interpretation, so unsat there is unsat here), the query as it is
        only when that is not unsat. With quantifiers among the conditions (strspn's long sets): first
        without them (fewer hypotheses: unsat holds for the whole), the full
        query only when that is not unsat, for QTIMEOUT at most (z3 may never
        settle a query with quantifiers that is sat: unknown, no proof);
        loose: a sat or unknown answer need not be the full query's (a
        branch kept that might be dead)."""
        if abstract and not INCREMENTAL:
            allc = ([] if bare else self.assumed) + [c for c in conds if c is not TRUE]
            new = replace(allc, abstract)
            if any(a is not b for a, b in zip(new, allc)):
                self.abstracting = True
                try:
                    r = self._check(*new, timeout=timeout, bare=True)
                    if r == 'unsat':
                        return 'unsat'
                    if r == 'sat' and os.environ.get('PROVE_DUMP_ABSTRACT'):  # what the abstraction loses: the query, to read its model
                        self.slow += 1
                        name = os.path.join(os.environ['PROVE_DUMP_ABSTRACT'], 'abstract%d-%d.smt2' % (os.getpid(), self.slow))
                        self.dump(new, name)
                        with open(name, 'a') as f:
                            f.write('; %s\n; %s\n' % (self.doing, getattr(self, 'note', '')))
                finally:
                    self.abstracting = False
        if not INCREMENTAL:
            allc = ([] if bare else self.assumed) + list(conds)
            if any(has_quantifier(c) for c in allc):
                plain = [c for c in allc if not has_quantifier(c)]
                r = self._check(*plain, keep=keep and loose, timeout=timeout, bare=True)
                if r == 'unsat' or loose:
                    return r
                return self._check(*conds, keep=keep, timeout=min(timeout or TIMEOUT, qtimeout), bare=bare)
        return self._check(*conds, keep=keep, timeout=timeout, bare=bare)

    def _check(self, *conds, keep=False, timeout=None, bare=False):
        """sat / unsat / unknown of the conjunction and the assumptions, with
        keep a sat answer left for values(). Each query in a context of its
        own, with only the definitions it needs: one z3 session through a
        whole proof slows down as its terms pile up (a memchr query 0.5 s
        alone, 45 s there); PROVE_INCREMENTAL=1 has the session, under a push."""
        t0 = time.time()
        self.queries += 1
        if INCREMENTAL:
            for c in conds:
                self.define(c)
            self.send('(push 1)')
            for c in conds:
                if c is not TRUE:
                    self.send('(assert %s)' % self.name(c))
        else:
            conds = ([] if bare else self.assumed) + [c for c in conds if c is not TRUE]  # bare: without the assumptions (fewer hypotheses: still sound)
            limit = min(TIMEOUT, timeout or TIMEOUT) * 1000
            if ANSWERS is not None:
                asked = ANSWERS.key(conds)
                known = ANSWERS.known.get(asked, '')
                if known == 'unsat' or known == 'sat' and not keep or known.startswith('unknown') and int(known.split()[1]) >= limit // 1000:
                    ANSWERS.hits += 1
                    self.result = known.split()[0]
                    return self.result
            if self.bp is not None and not any(has_lambda(c) for c in conds):
                self.p = self.bp
                self.send('(reset)')
                self.send('(set-option :print-success false)')
                self.send('(set-option :produce-models true)')
                self.send('(set-option :time-limit-per %d)' % limit)
            else:
                self.p = self.zp
                self.send('(reset)')
                self.send('(set-option :print-success false)')
                self.send('(set-option :timeout %d)' % limit)
            self.done = set()
            for c in conds:
                self.define(c)
            for c in conds:
                self.send('(assert %s)' % self.name(c))
        self.send('(check-sat)')
        self.p.stdin.flush()
        self.since = time.time()
        watch = None
        if SLOW is not None:  # a query still running after SLOW seconds says where it is
            where = ' < '.join('%s:%d' % (f.name, f.lineno) for f in traceback.extract_stack()[-5:-1][::-1])
            def running(conds=[c for c in conds if c is not TRUE] if not INCREMENTAL else []):
                print('    query running for %d s: %s' % (max(SLOW, 60), where), file=sys.stderr, flush=True)
                if os.environ.get('PROVE_DUMP') and conds:  # the query that is slow, while it is (PROVE_DUMP: each one standalone)
                    self.slow += 1
                    self.dump(conds, os.path.join(os.environ['PROVE_DUMP'], 'running-%d-%d-%d.smt2' % (os.getpid(), id(self) % 100000, self.slow)))
            watch = threading.Timer(max(SLOW, 60), running)
            watch.daemon = True
            watch.start()
        while True:
            line = self.p.stdout.readline().strip()
            if line in ('sat', 'unsat', 'unknown'):
                break
            if line.startswith(('(error', '[error')) or not line:
                raise RuntimeError('%s: %s' % ('bitwuzla' if self.p is self.bp else 'z3', line))
        if watch is not None:
            watch.cancel()
        self.since = None
        self.result = line
        if INCREMENTAL and not (keep and line == 'sat'):
            self.send('(pop 1)')
        if ANSWERS is not None:
            ANSWERS.add(asked, line, limit // 1000)
        dt = time.time() - t0
        if PROFILE:
            where = ' < '.join('%s:%d' % (f.name, f.lineno) for f in traceback.extract_stack()[-7:-1][::-1] if f.name not in ('check', '_check'))
            with open(PROFILE, 'a') as f:
                f.write('%.3f\t%s\t%s%s\t%s\n' % (dt, line, self.doing, ' [abstract]' if getattr(self, 'abstracting', False) else '', where))
        if SLOW is not None and dt > SLOW:  # PROVE_SLOW=seconds: where the slow queries come from (PROVE_DUMP=dir: each one standalone)
            where = ' < '.join('%s:%d' % (f.name, f.lineno) for f in traceback.extract_stack()[-5:-1][::-1])
            print('    slow query %.1f s, %s: %s' % (dt, line, where), file=sys.stderr, flush=True)
            if os.environ.get('PROVE_DUMP'):
                self.slow += 1
                self.dump((self.assumed if INCREMENTAL else []) + [c for c in conds if c is not TRUE], os.path.join(os.environ['PROVE_DUMP'], 'slow%d.smt2' % self.slow))
        return line

    def assume(self, c):
        """An assertion for every query (outside the pushes)."""
        self.define(c)
        self.send('(assert %s)' % self.name(c))
        self.assumed.append(c)

    def dump(self, conds, path):
        """The conjunction as a script of its own (written apart from this
        solver's session: a query may be running)."""
        b = object.__new__(Solver)
        lines = []
        b.done, b.send, b.log = set(), lines.append, None
        for c in conds:
            b.define(c)
        with open(path, 'w') as f:
            f.write('\n'.join(lines + ['(assert %s)' % self.name(c) for c in conds] + ['(check-sat)']) + '\n')

    def values(self, terms):
        """After a sat check: the model's values of terms, then the pop."""
        for t in terms:
            self.define(t)
        self.send('(get-value (%s))' % ' '.join(self.name(t) for t in terms))
        self.p.stdin.flush()
        text, depth = '', 0
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('z3 closed')
            text += line
            depth += line.count('(') - line.count(')')
            if depth <= 0 and text.strip():
                break
        if INCREMENTAL:
            self.send('(pop 1)')
        return text

    def close(self):
        for p in (self.zp, self.bp):
            if p is not None:
                p.stdin.close()
                p.wait()
