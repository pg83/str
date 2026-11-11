#!/usr/bin/env python3
"""Proofs of the library's functions on their machine code.

    ix run bin/z3 -- python3 dev/proof/prove.py --elf .build/proof/libstr.elf --func evex512.strlen_evex512 --spec strlen --induct

The linked library (./build proof) is executed symbolically from the
function's entry: a pointer is its low bits (one alignment class per run,
--align, 64 in all) and an unknown ALIGN-aligned base, so every place in a
page at once; the memory is an array of unknown bytes; every other register
is unknown. Every feasible path is followed (z3 decides each branch both
ways), and at the return the spec's obligations are proved for every input
the path admits: no input exists that breaks one.

Two modes for the loops:
  --max N    bounded: inputs up to N bytes, the loops unrolled as far as
             they go (time grows with the square of N);
  --induct   unbounded: at a loop's head, the second time a path reaches
             it, the registers that changed by one constant step become
             their first value plus K, any multiple of the step, and the rest
             of the state unknown. The spec's invariant is proved at the
             first arrival (the base) and one iteration from the general
             state either leaves with the obligations met or comes back to
             the head in the same general form with K + step, the invariant
             kept (the step). The lengths are then unbounded.
"""
import argparse
import itertools
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import smt as S  # noqa: E402
from smt import const, var  # noqa: E402
import machine as X  # noqa: E402

USER_LO, USER_HI = 0x10000, (1 << 47) - (1 << 32)  # canonical user addresses, with room for any object
ALIGN = int(os.environ.get('PROVE_ALIGN', '64'))  # the proof splits on a pointer's low bits: at least this when the code aligns at all (align_of)
K_LIMIT = 1 << 47  # a loop's distance from its first iteration: below this no address wraps, and the next one's below it is proved


def page(a):
    return S.lshr(a, const(64, 12))


def prefix_count(lanes):
    """A masked access whose lane i is bit i of bzhi(-1, idx), its lanes'
    addresses the first's plus i times the lane's bytes: the count of the
    lanes it touches, min(idx, its lanes) as 64 bits; else None. (strcmp
    evex512: 64 lanes each its own sum and pages, more than 20 s to tell
    they keep to lane 0's page; the count's last lane alone, at once.)"""
    k, n, ln = None, 0, lanes[0][1]
    for i, (la, nb, _, c) in enumerate(lanes):
        if c is S.FALSE:  # past the mask's width
            continue
        if nb != ln or c.op != 'eq' or c.args[1] is not const(1, 1) or c.args[0].op != 'extract' or c.args[0].args[1:] != (i, i):
            return None
        if k is None:
            k = c.args[0].args[0]
        if c.args[0].args[0] is not k or not S.is_const(S.sub(la, lanes[0][0])) or S.sub(la, lanes[0][0]).val != i * ln:
            return None
        n = i + 1
    if k is None or k.op != 'ite':
        return None
    big, ones, low = k.args
    w = k.w
    if not (S.is_const(ones) and ones.val == S.mask(w) and big.op == 'ule' and S.is_const(big.args[0]) and big.args[0].val == w):
        return None
    idx = big.args[1]
    if low is not S.sub(S.shl(const(w, 1), idx), const(w, 1)) or n > w:
        return None
    return S.ite(S.ult(idx, const(w, n)), S.zext(idx, 64), const(64, n))


def last_of(a, n):
    """The last byte of an access of n bytes (a number, or a term: rep movs)."""
    return S.add(a, S.sub(n, const(64, 1)) if isinstance(n, S.T) else const(64, n - 1))


def nonempty(n):
    return S.not_(S.is_zero(n)) if isinstance(n, S.T) else S.TRUE


class Failure(Exception):
    pass


class Counterexample(Exception):
    """A check's sat, found in a checking thread: its detail for the main one to report."""


class Restart(Exception):
    """Candidates of a loop's invariant the step did not keep: prove the
    class again without them (Houdini)."""

    def __init__(self, head, names):
        self.head, self.names = head, names


def align_of(img, entry):
    """The classes a function's proof splits a pointer into: 64, or the
    largest alignment its code rounds an address down to (an and with -2^j),
    so that every such address stays the base plus a known offset; 1, every
    pointer in one proof, for code that rounds none (it reads its
    misalignment with an and with 2^j - 1: a term like any other). The
    split is a case split: any number of classes covers every pointer; the
    fewest that keep the proofs fast (memset sse2: one class in 284 s, 64
    of a minute each; memchr sse2, which rounds to 16: 16 classes of 525 s,
    64 of 111). A 32-bit and is a count's (strcmp's page distance to 128),
    rsp's the frame's: neither a pointer the classes are of."""
    best, seen, todo = 1, set(), [entry]
    while todo:
        a = todo.pop()
        while a not in seen:
            seen.add(a)
            j = img.decode(a)
            mn = j['mnemonic']
            if mn == 'int3':
                break
            if mn == 'and' and j['ops'][0]['kind'] == 'Register' and j['ops'][0]['size'] == 64 and j['ops'][0]['reg'] != 'rsp':  # not a count's, not a frame's
                for o in j['ops']:
                    if o['kind'] == 'Immediate':
                        v = S.signed(S.sext(const(o['size'], o['imm']), 64)) if o.get('signed') else o['imm']
                        if v < 0 and (-v) & (-v - 1) == 0 and -v <= 4096:
                            best = max(best, -v)
            if j.get('branch_target') is not None:
                todo.append(j['branch_target'])
            elif mn in ('call', 'jmp') and j['ops'][0]['kind'] == 'Memory' and j['ops'][0]['mem'].get('rip_rel'):  # through the table
                at = j['offset'] + j['len'] + j['ops'][0]['mem']['disp']
                todo.append(sum(img.byte(at + i) << 8 * i for i in range(8)))
            if mn in ('ret', 'jmp'):
                break
            a += j['len']
    return max(best, ALIGN) if best > 1 else 1


def aligned_pointer(name, a, align):
    """A pointer with its low bits a and the rest unknown: base + a, the base
    known to be align-aligned, and the memory around it an array of its own."""
    sh = align.bit_length() - 1
    base = S.concat(var(name + '_hi', 64 - sh), const(sh, 0)) if sh else var(name + '_hi', 64)
    return base, S.add(base, const(64, a)), S.mem_var(name + '_mem')


SPANS = (16, 32, 64, 128, 256, 512, 1024)
AUX = 20  # seconds for a query that only picks a cheaper form of an obligation (unknown: the plain form)


CHUNK = 32  # a long disjunction in an obligation goes to the solver in parts of this many disjuncts


def chunked(part):
    """A part (conditions that must be unsat together) whose disjunction is
    long, as parts of CHUNK disjuncts each: all unsat exactly when it is.
    The solver does them far faster apart (strcspn avx2's exit past an
    unrolled loop, 128 offsets: 268 s whole, 48 in four parts)."""
    for i, c in enumerate(part):
        if c.op == 'or_b' and len(c.args) > 3 * CHUNK // 2:
            rest = part[:i] + part[i + 1:]
            # each part an or_b of its own disjuncts as they are (or_ would flatten one that is a disjunction
            # itself, strcspn's member(b), out of the term the abstraction replaces)
            return [rest + [c.args[j] if j + 1 == len(c.args) else S._mk('or_b', S.BOOL, tuple(c.args[j:j + CHUNK]))]
                    for j in range(0, len(c.args), CHUNK)]
    return [part]


class Alt:
    """An obligation written in several equivalent ways: alternatives, each
    a list of parts (queries) that must all be unsat; the prover tries them
    in turn (a short timeout for all but the last), sat in any part is a
    counterexample."""

    def __init__(self, alts):
        self.alts = alts

    def __radd__(self, prefix):
        return Alt([[list(prefix) + part for part in alt] for alt in self.alts])


def split_length(spec, st, n, given=()):
    """(c, r, b): n = c + r with c a constant (the whole blocks an unrolled
    loop's exit passed: 16 k) and r a rest the path keeps below b (an index
    in the block it stopped in), else None."""
    if st is None or S.is_const(n):
        return None
    f, c = S.linear(n)
    if not f or not 0 < c <= 4096:
        return None
    r = S.from_linear(n.w, f, 0)
    for b in SPANS[:3]:
        if spec.solver.check(*st.pc, *given, S.ule(const(64, b), r), timeout=AUX) == 'unsat':  # r < b: the forms over [0, b) cover it
            return c, r, b
    return None


def span(spec, st, lo, hi, pred, given=(), down=False, anchor=None, by_loads=True, cut=True):
    """pred at an offset in [lo, hi), as conditions for a query: the offsets
    one by one when the span is a known number, or when the path bounds it
    (a span up to an index from a mask's trailing zeros: offsets past lo by
    constants, the very terms of the bytes the path loaded, so the solver
    need not match a free offset against each of them), else through a
    free offset; and when the path loaded bytes, by them (loaded), the way
    the prover tries last (strcpy sse2's copy end: 0.8 s from lo, 44 by
    the loads; strcmp evex past its loops the other way round)."""
    n = S.sub(hi, lo)
    if S.is_const(n) and n.val <= 4096:
        return [S.or_(*[pred(S.add(lo, const(64, j))) for j in range(n.val)])]
    # an unrolled loop's exit: the c bytes of the blocks it passed with no bound to compare (they are all in
    # the span), then the rest r in the block it stopped in: the same offsets, fewer of them against a term
    cut = split_length(spec, st, n, given) if by_loads and cut else None
    if cut is not None:
        c, r, b = cut
        whole = S.or_(*[pred(S.add(lo, const(64, j))) for j in range(c)],
                      *[S.and_(S.ult(const(64, j), r), pred(S.add(lo, const(64, c + j)))) for j in range(b)])
        rest = span(spec, st, lo, hi, pred, given, down, anchor, by_loads, cut=False)
        return Alt([[[whole]]] + (rest.alts if isinstance(rest, Alt) else [[rest]]))
    way = loaded(spec, st, lo, hi, pred) if st is not None and by_loads else None
    if way is not None:
        rest = span(spec, st, lo, hi, pred, given, down, anchor, False)
        return Alt((rest.alts if isinstance(rest, Alt) else [[rest]]) + [way])  # the last: tried with the full timeout
    if st is not None:
        for b in SPANS:
            if spec.solver.check(*st.pc, *given, S.ult(const(64, b), n), timeout=AUX) == 'unsat':
                # the offsets from both ends: the path's loads past the frontier
                # and those back from an end (memcmp's last blocks at n - 64)
                # match one of them term for term; either is the whole span
                up = S.or_(*[S.and_(S.ult(const(64, j), n), pred(S.add(lo, const(64, j)))) for j in range(b)])
                if not down:
                    return [up]
                # the span split where the code's blocks from lo meet its last
                # ones back from the anchor (n: memcmp's last pair, memchr's last
                # block): offsets from lo below anchor - c, back from it above
                a = hi if anchor is None else anchor
                inside = lambda x: S.and_(S.ule(lo, x), S.ult(x, hi))  # noqa: E731
                alts = []
                for c in (16, 32, 64, 128):
                    if c < b:
                        upc = S.or_(*[S.and_(S.ult(const(64, j), n), S.ult(S.add(lo, const(64, j)), S.sub(a, const(64, c))), pred(S.add(lo, const(64, j))))
                                      for j in range(b)])
                        dnc = S.or_(*[S.and_(inside(S.sub(a, const(64, j + 1))), pred(S.sub(a, const(64, j + 1)))) for j in range(c)])
                        alts.append([[upc], [dnc]])
                return Alt(alts + [[[up]]])
    return [S.ule(lo, spec.i), S.ult(spec.i, hi), pred(spec.i)]


def loaded(spec, st, lo, hi, pred):
    """The span's offsets the path loaded, each byte of each load as the
    path's own term (its address less its object's pointer), and any other
    offset in it through a free one outside them all: two queries, together
    the whole span. A loop's blocks from an address rounded down are then
    matched term for term, not against offsets from the frontier (strcmp
    evex past its entry loop: 960 of the 1088 bytes, 554 s)."""
    own, i = getattr(spec, 'owner', lambda a: spec.s), getattr(spec, 'i', None)
    if i is None:
        return None
    offs, ranges = {}, []
    for a, nb, _, c in st.loads:
        if not isinstance(nb, int):
            return None
        try:
            o = S.sub(a, own(a))
        except Failure:  # not the objects' (the stack): the free offset's
            continue
        ranges.append((o, nb))
        for j in range(nb):
            x = S.add(o, const(64, j))
            offs[x.id] = x
    if not ranges or len(offs) > 2048:
        return None
    inside = lambda x: S.and_(S.ule(lo, x), S.ult(x, hi))  # noqa: E731
    hit = S.or_(*[S.and_(inside(x), pred(x)) for x in offs.values()])
    gap = [inside(i)] + [S.not_(S.ult(S.sub(i, o), const(64, nb))) for o, nb in ranges] + [pred(i)]
    return [[hit], gap]


def pinned(spec, st, idx, pred, ends, given=()):
    """pred(idx) for an index the path bounds near one of the ends (lo for
    an index past it, hi for one before it): the same written out over the
    offsets it can be, one form per end it is near, any of them equivalent
    to pred(idx) where the bound holds (so a query may take them all, and
    their negations all for not pred(idx)). The bytes are then the path's
    own terms, not a select at a computed index."""
    direct = pred(idx)
    if direct is S.TRUE or direct is S.FALSE:  # the answer built from those very bytes: settled already
        return [direct]
    forms = []
    for lo, hi in ends:
        cut = split_length(spec, st, S.sub(idx, lo), given) if lo is not None else None
        if cut is not None:  # lo + c + r, r in the block the scan stopped in
            c, r, b = cut
            forms.append(S.or_(*[S.and_(S.eq(r, const(64, j)), pred(S.add(lo, const(64, c + j)))) for j in range(b)]))
        for b in SPANS[:4]:
            if lo is not None and spec.solver.check(*st.pc, *given, S.ule(const(64, b), S.sub(idx, lo)), timeout=AUX) == 'unsat':
                forms.append(S.or_(*[S.and_(S.eq(idx, S.add(lo, const(64, j))), pred(S.add(lo, const(64, j)))) for j in range(b)]))
                break
            if hi is not None and spec.solver.check(*st.pc, *given, S.ule(const(64, b), S.sub(S.sub(hi, idx), const(64, 1))), timeout=AUX) == 'unsat':  # hi - b <= idx < hi
                forms.append(S.or_(*[S.and_(S.eq(idx, S.sub(hi, const(64, j + 1))), pred(S.sub(hi, const(64, j + 1)))) for j in range(b)]))
                break
    return forms + [direct]


def owner(addr, objects):
    """The object (pointer, base) of objects an address is in, as the machine
    finds its region: the one whose base is a term of the sum, else the one
    whose base's variables it alone has (an address rounded down). A's
    pointer plus an amount computed from b's (a distance to b's page end)
    is a's."""
    for p, base in objects:
        if X.offset_from(addr, base) is not None:
            return p
    for x, rest in X.rounded(addr):  # rounded down, plus an offset of no other object's: the pointer's
        p = owner(x, objects)
        if not any(X.offset_from(rest, b) is not None for q, b in objects if q is not p):
            return p
    hits = [p for p, base in objects if S.variables(base) & S.variables(addr)]
    if len(hits) != 1:
        raise Failure('an address of no single object: %r' % addr)
    return hits[0]


class Scan:
    """A scan from s for the first byte that matches: strlen (a zero byte,
    unbounded) and memchr (c within n bytes, as C11 has it: read as if one
    byte at a time, stopping at the match, so the object may end there).
    The answer's offset r; the last byte that must be readable: the match,
    or for memchr without one the n-th. Every load lies in the pages of
    [s, s + last]; no stores; aligned loads aligned.
    A loop's invariant at its frontier F (a register pointing into the
    object): no match in [s, F), the frontier within the object, and every
    load so far either in the pages up to F - 1's (B) or ending by F + step
    with F aligned to the step (A: [F, F + step) is then F's page, and F's
    byte must be in the object, which the return checks)."""
    bounded = False

    def __init__(self, maxlen, induct):
        self.maxlen, self.induct = maxlen, induct

    def setup(self, a, align):
        self.base, self.s, arr = aligned_pointer('s', a, align)
        self.regions = [(self.base, arr)]
        self.z = var('z', 64)  # where the scan must stop at the latest: a match, or n
        self.i = var('i', 64)
        params = self.params()
        stop = self.match(self.byte(S.add(self.s, self.z)))
        if self.bounded:
            stop = S.or_(S.and_(S.ult(self.z, self.n), stop), S.eq(self.z, self.n))
        # the object, up to where the scan must stop, inside user space
        assume = [S.ule(const(64, USER_LO), self.s), S.ult(self.s, const(64, USER_HI)), stop, S.ult(self.z, const(64, USER_HI - USER_LO)),
                  S.ule(S.add(self.s, self.z), const(64, USER_HI))]
        if not self.induct:
            assume.append(S.ule(self.z, const(64, self.maxlen)))
        return params, assume

    def byte(self, addr):
        st = X.State()
        st.regions, st.mem = self.regions, S.mem_var('M')
        return X.byte(st, addr)

    def at(self, off):
        return self.match(self.byte(S.add(self.s, off)))

    def some_match(self, lo, hi, st=None, given=(), down=False, anchor=None):
        return span(self, st, lo, hi, self.at, given, down, anchor)

    def last_readable(self):
        """The last offset the scan may need, given the stop z."""
        return S.ite(S.eq(self.z, self.n), S.sub(self.n, const(64, 1)), self.z) if self.bounded else self.z

    def stop_end(self, a, n):
        """For a load of the scanned bytes: ends E, so that only a stop below
        E puts the load past the stop's page (early): its last byte's offset
        first (a load that ends at the stop a mask found, strcpy_long's
        nul - 31: no stop below it is what the mask says, the page's start
        is that and arithmetic on it), then the offset of its last byte's
        page (a load that reaches past the stop in its page); None for a
        load whose bound is not the stop's."""
        if self.bounded:
            return None
        p = self.owner(a) if hasattr(self, 'owner') else self.s
        if p is not self.s and p is not getattr(self, 'b', None):
            return None
        start = S.shl(page(last_of(a, n)), const(64, 12))
        return [S.sub(last_of(a, n), p), S.ite(S.ult(p, start), S.sub(start, p), const(64, 0))]  # a load in p's page: past no stop at all

    def early(self, a, n):
        """A load past the last page the stop allows, or before s's page, is
        wrong whatever the answer."""
        out = [S.ult(page(S.add(self.s, self.last_readable())), page(last_of(a, n))), S.ult(page(a), page(self.s))]
        if self.bounded:
            out.append(S.is_zero(self.n))
        return S.and_(nonempty(n), S.or_(*out))

    def early_store(self, e):
        return S.and_(e[3], S.TRUE)  # a scan writes nothing

    def loads_unsafe(self, st, last):
        out = []
        s = self.s
        unsafe = [S.and_(c, S.or_(S.ult(page(a), page(s)), S.ult(page(S.add(s, last)), page(S.add(a, const(64, n - 1)))))) for a, n, _, c in st.loads]
        if unsafe:
            if self.bounded:
                unsafe.append(S.is_zero(self.n))
            out.append(('every load in the pages of the object', [S.or_(*unsafe)]))
        mis = [S.not_(S.is_zero(S.band(a, const(64, n - 1)))) for a, n, _ in st.aligned]
        if mis:
            out.append(('aligned loads aligned', [S.or_(*mis)]))
        if st.stores:
            raise Failure('stores at %s' % ', '.join('%#x' % e[2] for e in st.stores))
        return out

    def obligations(self, st):
        """At the return. Past a loop the bytes before its frontier are the
        invariant's."""
        s = self.s
        found, idx = self.answer(st)
        lo = S.sub(st.front_lo, s) if st.front_lo is not None else const(64, 0)
        ends = [(lo, None)] + ([(None, self.n)] if self.bounded else [])
        out = [('the answer is a match', [found] + Alt([[[S.not_(f)]] for f in pinned(self, st, idx, self.at, ends, [found])]))]
        if self.bounded:
            out.append(('the answer within the n bytes', [found, S.not_(S.ult(idx, self.n))]))
        out.append(('no match before the answer', [found] + self.some_match(lo, idx, st, [found], down=self.bounded, anchor=self.n if self.bounded else None)))
        if st.front_lo is not None:
            out.append(('the answer at or past the frontier', [found, S.ult(idx, lo)]))
        last = idx
        if self.bounded:
            out.append(('no match in the n bytes when there is no answer', [S.not_(found)] + self.some_match(lo, self.n, st, [S.not_(found)], down=True)))
            last = S.ite(found, idx, S.sub(self.n, const(64, 1)))
        for f in st.front_bytes:
            out.append(('the byte at a loop\'s frontier in the object', [S.ult(S.add(s, last), f)]))
        return out + self.loads_unsafe(st, last), [s, idx, self.z] + ([self.n] if self.bounded else [])

    def frontier_checks(self, st, f, lo):
        s = self.s
        out = [('the frontier past s', [S.ule(f, s)])]
        if lo is not None:
            out.append(('the frontier at or past the previous loop\'s', [S.ult(f, lo)]))
        if self.bounded:
            out.append(('the frontier within the object', [S.ult(self.n, S.sub(f, s))]))
        return out

    def loads_by(self, st, f, mode, step, what):
        out = []
        s = self.s
        if st.loads:
            if mode == 'B':
                bad = [S.and_(c, S.ult(page(S.sub(f, const(64, 1))), page(S.add(a, const(64, n - 1))))) for a, n, _, c in st.loads]
                out.append(('%s in the pages up to the frontier\'s' % what, [S.or_(*bad)]))
            else:
                bad = [S.and_(c, S.ult(S.add(f, const(64, step)), S.add(a, const(64, n)))) for a, n, _, c in st.loads]
                out.append(('%s end by the frontier and a step' % what, [S.or_(*bad)]))
            out.append(('%s after s\'s page' % what, [S.or_(*[S.and_(c, S.ult(page(a), page(s))) for a, _, _, c in st.loads])]))
        mis = [S.not_(S.is_zero(S.band(a, const(64, n - 1)))) for a, n, _ in st.aligned]
        if mis:
            out.append(('%s: aligned ones aligned' % what, [S.or_(*mis)]))
        return out

    def base_case(self, st, f, mode, step):
        s = self.s
        lo = S.sub(st.front_lo, s) if st.front_lo is not None else const(64, 0)
        out = self.frontier_checks(st, f, st.front_lo)
        out.append(('no match before the loop', self.some_match(lo, S.sub(f, s), st)))
        return out + self.loads_by(st, f, mode, step, 'the loads before the loop')

    def general(self, f):
        """What the invariant at frontier f gives without a quantifier: the
        stop lies at or past it."""
        return [S.ule(S.sub(f, self.s), self.z)]

    def candidates(self, step):
        """For the invariant: no stop in the c bytes from the frontier (a loop
        that checked them ahead of it), byte by byte (a bound on z alone
        would say nothing of the bytes); bounds on the frontier, the next
        iteration's bytes inside the object (a loop that checks at its
        bottom), the frontier inside it (one that checks at its top)."""
        out = [('no stop in the %d bytes from F' % c,
                (lambda c: lambda st, f: S.and_(*[S.not_(self.at(S.add(S.sub(f, self.s), const(64, j)))) for j in range(c)]))(c))
               for c in sorted({32, 64, 128, step, 2 * step}) if c <= 256]
        if not self.bounded:
            return out
        return out + [('the next iteration inside the object', lambda st, f: S.ule(S.add(S.sub(f, self.s), const(64, step)), self.n)),
                      ('the frontier inside the object', lambda st, f: S.ule(S.sub(f, self.s), self.n))]

    def step_case(self, st, f, f2, mode, step, cov=None):
        s = self.s
        out = [('no match in the iteration', self.some_match(S.sub(st.front_lo, s), S.sub(f2, s), st))]  # front_lo: f, or an inner loop's
        if self.bounded:
            out.append(('the next frontier within the object', [S.ult(self.n, S.sub(f2, s))]))
        return out + self.loads_by(st, f2, mode, step, 'the iteration\'s loads')

    def mode(self, solver, st, f, step):
        """The loads invariant this loop can have: B if the loads so far lie in
        the pages up to F - 1's, else A if F is aligned to the step and the
        loads end by F + step."""
        if not st.loads or solver.check(*st.pc, S.or_(*[S.and_(c, S.ult(page(S.sub(f, const(64, 1))), page(S.add(a, const(64, n - 1)))))
                                                         for a, n, _, c in st.loads])) == 'unsat':
            return 'B'
        if step <= 4096 and solver.check(*st.pc, S.not_(S.is_zero(S.band(f, const(64, step - 1))))) == 'unsat':
            return 'A'
        return 'B'  # the base case then fails with a counterexample

    def show(self):
        return [self.s, self.z] + ([self.n] if self.bounded else [])


class Strlen(Scan):
    """strlen(s): the offset of the first zero byte."""

    def params(self):
        return [self.s]

    def match(self, b):
        return S.is_zero(b)

    def answer(self, st):
        return S.TRUE, st.gpr['rax']

    def contracts(self, img, entry):
        return {t: self.c_tail for t in named_targets(img, entry, ('strlen_tail_',))}

    def c_tail(self, prover, st):
        """strlen_tail(a, s), proved on its own (StrlenTail): under its
        precondition (a 16-aligned, s < a, no NUL in [s, a)) strlen(s)."""
        a = st.gpr['rdi']
        lo = S.sub(st.front_lo, self.s) if st.front_lo is not None else const(64, 0)
        out = [('strlen_tail is given s', [S.not_(S.eq(st.gpr['rsi'], self.s))]),
               ('strlen_tail\'s a 16-aligned', [S.not_(S.is_zero(S.band(a, const(64, 15))))]),
               ('strlen_tail\'s a past s', [S.ule(a, self.s)]),
               ('no NUL before strlen_tail\'s a', self.some_match(lo, S.sub(a, self.s), st))]
        return tail_call(self, prover, st, out)


class StrlenTail(Strlen):
    """strlen_tail(a, s): strlen's kernel, under its contract's precondition
    (a 16-aligned, s < a, no NUL in [s, a)): strlen(s). Its frontier a
    from the start (z, any NUL, at or past it)."""

    def setup(self, al, align):
        params, assume = Strlen.setup(self, al, align)
        delta = var('adelta', 64)
        self.a = self.front0 = S.add(self.s, delta)
        return [self.a, self.s], assume + [S.is_zero(S.band(self.a, const(64, 15))), S.ult(const(64, 0), delta), S.ule(delta, self.z)]

    def contracts(self, img, entry):
        return {}


class Memchr(Scan):
    """memchr(s, c, n): s + r for the first r < n with s[r] == (u8)c, else
    null; n == 0 reads nothing (s may be unreadable)."""
    bounded = True

    def params(self):
        self.c = var('c', 64)
        self.n = var('n', 64)
        return [self.s, self.c, self.n]

    def match(self, b):
        return S.eq(b, S.extract(self.c, 7, 0))

    def answer(self, st):
        r = st.gpr['rax']
        return S.not_(S.is_zero(r)), S.sub(r, self.s)

    def contracts(self, img, entry):
        return {t: self.c_large for t in named_targets(img, entry, ('memchr_large_',))}

    def c_large(self, prover, st):
        """memchr_large(from, c, n - F), F = from - s, proved on its own
        (MemchrLarge): under its precondition (no match in [from, A), A =
        from's 16-byte block plus 32, the n bytes past A) memchr itself from
        from on; with no match in [s, from) either its answer and loads are
        this spec's."""
        frm = st.gpr['rdi']
        f, d = S.sub(frm, self.s), large_head(frm)
        out = [('memchr_large is given a pointer at or past s', [S.ult(frm, self.s)]),
               ('memchr_large is given c', [S.not_(S.eq(S.extract(st.gpr['rsi'], 7, 0), S.extract(self.c, 7, 0)))]),
               ('memchr_large is given the rest of n', [S.not_(S.eq(st.gpr['rdx'], S.sub(self.n, f)))]),
               ('memchr_large\'s n past its head', [S.ule(st.gpr['rdx'], d)])]
        lo = S.sub(st.front_lo, self.s) if st.front_lo is not None else const(64, 0)
        out.append(('no match before memchr_large\'s head ends', self.some_match(lo, S.add(f, d), st)))
        return tail_call(self, prover, st, out)


def large_head(s):
    """The bytes memchr_large's caller has checked: to its 16-byte block plus 32 (17 to 32 of them)."""
    return S.sub(const(64, 32), S.band(s, const(64, 15)))


class MemchrLarge(Memchr):
    """memchr_large(s, c, n): memchr's kernel, under the precondition its
    callers meet (no match in [s, A), A the 16-byte block of s plus 32, n
    past A): memchr itself. The scan's loads in the pages up to the match
    or of the n bytes, as memchr's: C11 lets the object end at the match."""

    def setup(self, a, align):
        params, assume = Memchr.setup(self, a, align)
        d = large_head(self.s)
        return params, assume + [S.ult(d, self.n)] + [S.or_(S.ule(d, const(64, j)), S.not_(self.at(const(64, j)))) for j in range(32)]

    def contracts(self, img, entry):
        return {}


class Strnlen(Scan):
    """strnlen(s, n): the offset of the first NUL before n, else n; n == 0
    reads nothing (s may be unreadable). memchr's blocks for the NUL."""
    bounded = True

    def params(self):
        self.n = var('n', 64)
        return [self.s, self.n]

    def match(self, b):
        return S.is_zero(b)

    def answer(self, st):
        r = st.gpr['rax']
        return S.ult(r, self.n), r

    def contracts(self, img, entry):
        return {t: self.c_large for t in named_targets(img, entry, ('strnlen_large_',))}

    def c_large(self, prover, st):
        """strnlen_large(s, from, n), proved on its own (StrnlenLarge): under
        its precondition (no NUL in [s, A), A = from's 16-byte block plus 32,
        within 256 bytes of s, n past A - s) strnlen(s, n) itself."""
        frm = st.gpr['rsi']
        d = S.add(S.sub(frm, self.s), large_head(frm))  # A - s
        out = [('strnlen_large is given s', [S.not_(S.eq(st.gpr['rdi'], self.s))]),
               ('strnlen_large is given from at or past s', [S.ult(frm, self.s)]),
               ('strnlen_large\'s head within 256 bytes of s', [S.ult(const(64, 256), d)]),
               ('strnlen_large is given n', [S.not_(S.eq(st.gpr['rdx'], self.n))]),
               ('strnlen_large\'s n past its head', [S.ule(self.n, d)])]
        lo = S.sub(st.front_lo, self.s) if st.front_lo is not None else const(64, 0)
        out.append(('no NUL before strnlen_large\'s head ends', self.some_match(lo, d, st)))
        return tail_call(self, prover, st, out)


class StrnlenLarge(Strnlen):
    """strnlen_large(s, from, n): strnlen's kernel, under the precondition
    its callers meet (from at or past s, no NUL in [s, A), A = from's 16-byte
    block plus 32, within 256 bytes of s, n past A - s): strnlen(s, n). The
    loads as memchr_large's: aligned, in the pages up to the NUL or of the n
    bytes."""

    def setup(self, a, align):
        params, assume = Strnlen.setup(self, a, align)
        delta = var('fdelta', 64)
        self.frm = S.add(self.s, delta)
        d = S.add(delta, large_head(self.frm))
        return [self.s, self.frm, self.n], assume + [S.ule(d, const(64, 256)), S.ult(d, self.n)] + \
            [S.or_(S.ule(d, const(64, j)), S.not_(self.at(const(64, j)))) for j in range(256)]

    def contracts(self, img, entry):
        return {}


class StrCompare(Scan):
    """strcmp(a, b): the sign of a[r] - b[r] at the first r where they differ
    or a's is NUL (C11 reads both as if one byte at a time up to there: the
    objects may end there). The answer is the difference of two bytes it
    loaded, read off its term as memcmp's; the scan's obligations with a
    stop at an offset of both strings, and every load of a string in its
    pages up to the stop. a's low bits are the alignment class; b any
    pointer. Loops: the frontier F into a, its twin b + (F - a); the loads
    so far in the pages up to the twins' bytes before them (B)."""

    def setup(self, a, align):
        self.base, self.s, self.A = aligned_pointer('a', a, align)
        self.bbase, self.b, self.B = aligned_pointer('b', 0, 1)
        self.regions = [(self.base, self.A), (self.bbase, self.B)]
        self.z, self.i = var('z', 64), var('i', 64)
        s, b, z = self.s, self.b, self.z
        assume = [S.ule(const(64, USER_LO), s), S.ult(s, const(64, USER_HI)), S.ule(const(64, USER_LO), b), S.ult(b, const(64, USER_HI)),
                  self.at(z), S.ult(z, const(64, USER_HI - USER_LO)), S.ule(S.add(s, z), const(64, USER_HI)), S.ule(S.add(b, z), const(64, USER_HI))]
        if not self.induct:
            assume.append(S.ule(z, const(64, self.maxlen)))
        return [s, b], assume

    def bytes_at(self, off):
        return S.select(self.A, S.add(S.sub(self.s, self.base), off)), S.select(self.B, S.add(S.sub(self.b, self.bbase), off))

    def at(self, off):
        x, y = self.bytes_at(off)
        return S.or_(S.not_(S.eq(x, y)), S.is_zero(x))

    def owner(self, addr):
        return owner(addr, ((self.s, self.base), (self.b, self.bbase)))

    def twin(self, p, f):
        """Frontier f (into a) for the string at p."""
        return f if p is self.s else S.add(self.b, S.sub(f, self.s))

    def early(self, a, n):
        p = self.owner(a)
        return S.and_(nonempty(n), S.or_(S.ult(page(S.add(p, self.z)), page(last_of(a, n))), S.ult(page(a), page(p))))

    def loads_unsafe(self, st, last):
        out = []
        unsafe = [S.and_(c, S.or_(S.ult(page(a), page(self.owner(a))), S.ult(page(S.add(self.owner(a), last)), page(last_of(a, n)))))
                  for a, n, _, c in st.loads]
        if unsafe:
            out.append(('every load in the pages of its string up to the answer', [S.or_(*unsafe)]))
        if st.stores:
            raise Failure('stores at %s' % ', '.join('%#x' % e[2] for e in st.stores))
        return out

    def loads_by(self, st, f, mode, step, what):
        """Nothing for the loops: every load is checked as it runs against
        its string's pages up to any stop z, the general state's z at or
        past the frontier."""
        return []

    def mode(self, solver, st, f, step):
        return 'B'

    def frontier_checks(self, st, f, lo):
        out = [('the frontier at or past a', [S.ult(f, self.s)])]
        if lo is not None:
            out.append(('the frontier at or past the previous loop\'s', [S.ult(f, lo)]))
        return out

    def contracts(self, img, entry):
        return {t: self.c_tail for t in named_targets(img, entry, ('strcmp_tail_',))}

    def c_tail(self, prover, st):
        """strcmp_tail(a + F, b + F), any pointers: strcmp itself, proved as
        such (spec strcmp_tail); with no stop in [0, F) the first stop from
        there is this spec's, the difference there its answer, its loads in
        the pages up to it."""
        f = st.gpr['rdi']
        out = [('strcmp_tail is given b as far on as a', [S.not_(S.eq(S.sub(st.gpr['rsi'], self.b), S.sub(f, self.s)))])]
        return tail_call(self, prover, st, out + self.base_case(st, f, 'B', 1))

    def answer(self, st):
        """The index the answer is the difference at: of a's byte in it; for a
        constant one (0: both NUL), a witness, the offset of a byte load from
        a on the path, the latest at which the stop and the difference hold."""
        r = S.extract(st.gpr['rax'], 31, 0)
        sa = {x.args[1].id: x for x in selects(r) if x.args[0] is self.A}
        if len(sa) == 1:
            return S.TRUE, S.sub(next(iter(sa.values())).args[1], S.sub(self.s, self.base))
        if sa or not S.is_const(r):
            raise Failure('an answer from %d of a\'s bytes' % len(sa))
        for a, n, _, c in reversed(st.loads):
            if n == 1 and c is S.TRUE and self.owner(a) is self.s:
                w = S.sub(a, self.s)
                x, y = self.bytes_at(w)
                if self.solver.check(*st.pc, S.not_(S.and_(self.at(w), S.eq(r, S.sub(S.zext(x, 32), S.zext(y, 32)))))) == 'unsat':
                    return S.TRUE, w
        raise Failure('a constant answer with no byte of a to witness it')

    def obligations(self, st):
        out, show = Scan.obligations(self, st)
        r = S.extract(st.gpr['rax'], 31, 0)
        lo = S.sub(st.front_lo, self.s) if st.front_lo is not None else const(64, 0)

        def right(j):
            x, y = self.bytes_at(j)
            return S.eq(r, S.sub(S.zext(x, 32), S.zext(y, 32)))
        forms = pinned(self, st, self.answer(st)[1], right, [(lo, None)])
        return [('the answer the difference of the bytes at its index', [S.not_(f) for f in forms])] + out, show

    def show(self):
        return [self.s, self.b, self.z, self.i]


class Writer:
    """A function that writes the n bytes of an object at d: each store
    writes, lane by lane, the byte the spec expects at its index (expected),
    and every index in [0, n) is covered by some store; stores outside the
    object are refused as they happen. A loop's invariant: the iterations so
    far covered [F1, F) (F1 its first frontier), each store right. The
    memory is not needed for this: written as the loop goes, it is unknown in
    the general state, which over-approximates."""

    def __init__(self, maxlen, induct):
        self.maxlen, self.induct = maxlen, induct

    def inside(self, a, nb):
        return S.and_(S.ule(self.d, a), S.ule(S.add(a, nb), S.add(self.d, self.n)))

    def early_store(self, e):
        a, nb, rip, c, lanes = e
        if isinstance(nb, S.T):  # rep stos, rep movs: nothing for a count of 0
            return S.and_(c, S.not_(S.is_zero(nb)), S.not_(self.inside(a, nb)))
        return S.and_(c, S.not_(self.inside(a, const(64, nb))))

    def intervals(self, stores):
        out = []
        for a, nb, rip, c, lanes in stores:
            lo = S.sub(a, self.d)
            out.append((lo, S.add(lo, nb if isinstance(nb, S.T) else const(64, nb)), c))
        return out

    def wrong_values(self, stores):
        bad = []
        for a, nb, rip, c, lanes in stores:
            idx = S.sub(a, self.d)
            if isinstance(lanes, tuple):  # rep movs: from the source at the same index
                bad.append(S.and_(c, S.not_(S.is_zero(nb)), self.movs_wrong(idx, nb, *lanes[1:])))
                continue
            for j, v in enumerate(lanes):
                cond = S.and_(c, S.not_(S.eq(v, self.expected_at(S.add(idx, const(64, j))))))
                if isinstance(nb, S.T):
                    cond = S.and_(cond, S.not_(S.is_zero(nb)))
                bad.append(cond)
        return S.or_(*bad)

    def uncovered(self, ivs, lo, hi):
        """An index in [lo, hi) no interval covers."""
        i = self.i
        return [S.ule(lo, i), S.ult(i, hi)] + [S.not_(S.and_(c, S.ule(a, i), S.ult(i, b))) for a, b, c in ivs]

    def obligations(self, st):
        out = [('the answer is d', [S.not_(S.eq(st.gpr['rax'], self.d))])]
        wv = self.wrong_values(st.stores)
        if wv is not S.FALSE:
            out.append(('every store writes the expected bytes', [wv]))
        out.append(('every byte of the object written', self.uncovered(st.cover + self.intervals(st.stores), const(64, 0), self.n)))
        return out, self.show()  # the loads: each checked as it ran (early), under fewer of the path's conditions

    def base_case(self, st, f, mode, step, down=False):
        out = [('the frontier at or past d', [S.ult(f, self.d)])]
        if st.front_lo is not None and not down:
            out.append(('the frontier at or past the previous loop\'s', [S.ult(f, st.front_lo)]))
        wv = self.wrong_values(st.stores)
        if wv is not S.FALSE:
            out.append(('the stores before the loop write the expected bytes', [wv]))
        return out

    def cover_general(self, st, lo, hi):
        """The general state's coverage: what was before the loop and what
        the iterations wrote, d's indexes [lo, hi) (from the first one's
        lowest store up, or down to the next one's), none for a loop whose
        first iteration stored nothing."""
        return st.cover + self.intervals(st.stores) + ([(lo, hi, S.TRUE)] if lo is not None else [])

    def step_case(self, st, f, f2, mode, step, cov=None):
        out = []
        wv = self.wrong_values(st.stores)
        if wv is not S.FALSE:
            out.append(('the iteration\'s stores write the expected bytes', [wv]))
        if cov is not None:
            out.append(('the iteration covers its part of d', self.uncovered(st.cover + self.intervals(st.stores), *cov)))
        return out

    def candidates(self, step):
        return [('the next iteration inside the object', lambda st, f: S.ule(S.add(S.sub(f, self.d), const(64, step)), self.n)),
                ('the frontier inside the object', lambda st, f: S.ule(S.sub(f, self.d), self.n)),
                ('the frontier at or past d', lambda st, f: S.ule(self.d, f))]

    def general(self, f):
        return []

    def mode(self, solver, st, f, step):
        return 'B'


class Fill(Writer):
    """memset(d, c, n): d; the n bytes at d all (u8)c, nothing else written,
    nothing read outside them."""

    def setup(self, a, align):
        self.base, self.d, arr = aligned_pointer('d', a, align)
        self.regions = [(self.base, arr)]
        self.c, self.n, self.i = var('c', 64), var('n', 64), var('i', 64)
        assume = [S.ule(const(64, USER_LO), self.d), S.ult(self.d, const(64, USER_HI)), S.ult(self.n, const(64, USER_HI)),
                  S.ule(S.add(self.d, self.n), const(64, USER_HI))]
        if not self.induct:
            assume.append(S.ule(self.n, const(64, self.maxlen)))
        return [self.d, self.c, self.n], assume

    def expected_at(self, idx):
        return S.extract(self.c, 7, 0)

    def contracts(self, img, entry):
        return {t: self.c_large for t in named_targets(img, entry, ('fill_large_',))}

    def c_large(self, prover, st):
        """fill_large(d, c, n), proved on its own (FillLarge): for n past 128
        memset itself, every one of the n bytes its store (any before it
        within them, checked as they ran)."""
        out = [('fill_large is given d', [S.not_(S.eq(st.gpr['rdi'], self.d))]),
               ('fill_large is given c', [S.not_(S.eq(S.extract(st.gpr['rsi'], 7, 0), S.extract(self.c, 7, 0)))]),
               ('fill_large is given n', [S.not_(S.eq(st.gpr['rdx'], self.n))]),
               ('fill_large\'s n past 128', [S.ule(self.n, const(64, 128))])]
        return tail_call(self, prover, st, out)

    def movs_wrong(self, idx, count, src, arr, off):
        return S.TRUE

    def early(self, a, n):
        return S.and_(nonempty(n), S.not_(self.inside(a, n if isinstance(n, S.T) else const(64, n))))

    def show(self):
        return [self.d, self.n, self.c, self.i]


class FillLarge(Fill):
    """fill_large(d, c, n): memset's kernel, n > 128: memset itself."""

    def setup(self, a, align):
        params, assume = Fill.setup(self, a, align)
        return params, assume + [S.ult(const(64, 128), self.n)]

    def contracts(self, img, entry):
        return {}


class Copy(Writer):
    """memcpy(d, s, n): d; the n bytes at d those at s, nothing else
    written, no load outside the pages of [s, s + n); the objects do not
    overlap. d's low bits are the alignment class; s any pointer."""

    def setup(self, a, align):
        self.base, self.d, darr = aligned_pointer('d', a, align)
        self.sbase, self.s, sarr = aligned_pointer('s', 0, 1)
        self.regions = [(self.base, darr), (self.sbase, sarr)]
        self.src = sarr
        self.n, self.i = var('n', 64), var('i', 64)
        d, s, n = self.d, self.s, self.n
        assume = [S.ule(const(64, USER_LO), d), S.ult(d, const(64, USER_HI)), S.ule(const(64, USER_LO), s), S.ult(s, const(64, USER_HI)),
                  S.ult(n, const(64, USER_HI)), S.ule(S.add(d, n), const(64, USER_HI)), S.ule(S.add(s, n), const(64, USER_HI)),
                  S.or_(S.ule(S.add(d, n), s), S.ule(S.add(s, n), d))]
        if not self.induct:
            assume.append(S.ule(n, const(64, self.maxlen)))
        return [d, s, n], assume

    def expected_at(self, idx):
        return S.select(self.src, S.add(S.sub(self.s, self.sbase), idx))

    def contracts(self, img, entry):
        return {t: self.c_large for t in named_targets(img, entry, ('copy_large_',))}

    def c_large(self, prover, st):
        """copy_large(d, s, n), proved on its own (CopyLarge, memmove's
        spec): for n past 64 the copy itself, with the objects overlapping
        or not, from the source as the kernel finds it: for memmove nothing
        written before it."""
        if getattr(self, 'window', False) and st.stores:
            raise Failure('stores before copy_large: the source it copies may have changed')
        out = [('copy_large is given d', [S.not_(S.eq(st.gpr['rdi'], self.d))]),
               ('copy_large is given s', [S.not_(S.eq(st.gpr['rsi'], self.s))]),
               ('copy_large is given n', [S.not_(S.eq(st.gpr['rdx'], self.n))]),
               ('copy_large\'s n past 64', [S.ule(self.n, const(64, 64))])]
        return tail_call(self, prover, st, out)

    def movs_wrong(self, idx, count, src, arr, off):
        """The source's region is never written (the stores stay in d's)."""
        return S.not_(S.eq(idx, S.sub(src, self.s)))

    def early(self, a, n):
        s = self.s
        return S.and_(nonempty(n), S.or_(S.is_zero(self.n), S.ult(page(a), page(s)), S.ult(page(S.add(s, S.sub(self.n, const(64, 1)))),
                                                                                         page(last_of(a, n)))))

    def show(self):
        return [self.d, self.s, self.n, self.i]


class Move(Copy):
    """memmove(d, s, n): memcpy's, but the objects may overlap: one memory,
    s anywhere relative to d (the base plus an unknown offset t), every
    byte written the source's as it was before the call. A loop's general
    state keeps the memory of its first arrival but for what the iterations
    covered: the source may lie there, and a load from it then is unknown,
    as a store's value then fails."""
    window = True

    def setup(self, a, align):
        self.base, self.d, arr = aligned_pointer('d', a, align)
        self.s = var('s', 64)  # a pointer of its own (its pages' arithmetic simple), into d's memory at s - base
        self.sbase = self.base
        self.regions = [(self.base, arr)]
        self.aliases = [(self.s, 0, S.sub(self.s, self.base))]
        self.src = arr
        self.foreign = S.variables(self.s)  # what makes an address the source's
        self.n, self.i = var('n', 64), var('i', 64)
        d, s, n = self.d, self.s, self.n
        assume = [S.ule(const(64, USER_LO), d), S.ult(d, const(64, USER_HI)), S.ule(const(64, USER_LO), s), S.ult(s, const(64, USER_HI)),
                  S.ult(n, const(64, USER_HI)), S.ule(S.add(d, n), const(64, USER_HI)), S.ule(S.add(s, n), const(64, USER_HI))]
        if not self.induct:
            assume.append(S.ule(n, const(64, self.maxlen)))
        return [d, s, n], assume

    def movs_wrong(self, idx, count, src, arr, off):
        """From the source at the index, its bytes as they were at the call
        (some j below the count otherwise)."""
        j = var('j', 64)
        return S.or_(S.not_(S.eq(idx, S.sub(src, self.s))),
                     S.and_(S.ult(j, count), S.not_(S.eq(S.select(arr, S.add(off, j)), S.select(self.src, S.add(off, j))))))


class CopyLarge(Move):
    """copy_large(d, s, n): memcpy's and memmove's kernel, n > 64, any
    overlap: memmove itself."""

    def setup(self, a, align):
        params, assume = Move.setup(self, a, align)
        return params, assume + [S.ult(const(64, 64), self.n)]

    def contracts(self, img, entry):
        return {}


def selects(t, out=None, seen=None):
    """The outermost select subterms of t (the bytes it was computed from,
    not those its addresses were)."""
    if out is None:
        out, seen = [], set()
    if t.id in seen:
        return out
    seen.add(t.id)
    if t.op == 'select':
        out.append(t)
        return out
    for x in t.args:
        if isinstance(x, S.T) and x.w != S.MEM:
            selects(x, out, seen)
    return out


class Compare:
    """memcmp(a, b, n): the sign of a[r] - b[r] at the first r < n where they
    differ, 0 without one; the loads in the pages of [a, a + n) and of
    [b, b + n) (it may read all n bytes of both), nothing for n == 0.
    The answer is the difference of two bytes it loaded (at an index of its
    own), or 0: the index is read off the answer's term, and the proof is
    that the bytes there are a's and b's at one index r < n, with no
    difference before it, and none at all when they are equal. a's low bits
    are the alignment class; b any pointer. A loop's invariant at its
    frontier F (into a): no difference before F - a."""
    bounded = True

    def __init__(self, maxlen, induct):
        self.maxlen, self.induct = maxlen, induct

    def setup(self, a, align):
        self.base, self.a, self.A = aligned_pointer('a', a, align)
        self.bbase, self.b, self.B = aligned_pointer('b', 0, 1)
        self.regions = [(self.base, self.A), (self.bbase, self.B)]
        self.n, self.i, self.z = var('n', 64), var('i', 64), var('z', 64)
        a, b, n = self.a, self.b, self.n
        # z: a difference or n, any (so the invariants can say no difference before F)
        stop = S.or_(S.and_(S.ult(self.z, n), self.differ(self.z)), S.eq(self.z, n))
        assume = [S.ule(const(64, USER_LO), a), S.ult(a, const(64, USER_HI)), S.ule(const(64, USER_LO), b), S.ult(b, const(64, USER_HI)),
                  S.ult(n, const(64, USER_HI)), S.ule(S.add(a, n), const(64, USER_HI)), S.ule(S.add(b, n), const(64, USER_HI)), stop,
                  S.ule(self.z, n)]
        if not self.induct:
            assume.append(S.ule(n, const(64, self.maxlen)))
        return [a, b, n], assume

    def at(self, arr, base, p, off):
        return S.select(arr, S.add(S.sub(p, base), off))

    def differ(self, off):
        return S.not_(S.eq(self.at(self.A, self.base, self.a, off), self.at(self.B, self.bbase, self.b, off)))

    def some_diff(self, lo, hi, st=None, given=(), down=False, anchor=None):
        return span(self, st, lo, hi, self.differ, given, down, anchor)

    @staticmethod
    def alt_parts(x):
        """A span's ways as alternatives' parts: one alternative for a plain one."""
        return [p for alt in x.alts for p in [alt]] if isinstance(x, Alt) else [[x]]

    def owner(self, addr):
        return owner(addr, ((self.a, self.base), (self.b, self.bbase)))

    def early(self, addr, nb):
        p = self.owner(addr)
        return S.and_(nonempty(nb), S.or_(S.is_zero(self.n), S.ult(page(addr), page(p)),
                                          S.ult(page(S.add(p, S.sub(self.n, const(64, 1)))), page(last_of(addr, nb)))))

    def early_store(self, e):
        return e[3]

    def contracts(self, img, entry):
        return {t: self.c_large for t in named_targets(img, entry, ('memcmp_large_',))}

    def c_large(self, prover, st):
        """memcmp_large(a, b, n), proved on its own (MemcmpLarge): for n past
        128 with the first 64 bytes equal, memcmp itself."""
        lo = S.sub(st.front_lo, self.a) if st.front_lo is not None else const(64, 0)
        out = [('memcmp_large is given a', [S.not_(S.eq(st.gpr['rdi'], self.a))]),
               ('memcmp_large is given b', [S.not_(S.eq(st.gpr['rsi'], self.b))]),
               ('memcmp_large is given n', [S.not_(S.eq(st.gpr['rdx'], self.n))]),
               ('memcmp_large\'s n past 128', [S.ule(self.n, const(64, 128))]),
               ('no difference in memcmp_large\'s first 64 bytes', self.some_diff(lo, const(64, 64), st))]
        return tail_call(self, prover, st, out)

    def answer_index(self, r):
        """r's index into the objects, or None for a constant answer."""
        if S.is_const(r):
            return None
        sa = {x.args[1].id: x for x in selects(r) if x.args[0] is self.A}
        if len(sa) != 1:
            raise Failure('an answer not from one byte of a: %d of its bytes' % len(sa))
        return S.sub(next(iter(sa.values())).args[1], S.sub(self.a, self.base))

    def obligations(self, st):
        lo = S.sub(st.front_lo, self.a) if st.front_lo is not None else const(64, 0)
        r = S.extract(st.gpr['rax'], 31, 0)
        idx = self.answer_index(r)
        if idx is None:
            out = [('a constant answer is 0', [S.not_(S.is_zero(r))]),
                   ('no difference in the n bytes when the answer is 0', self.some_diff(lo, self.n, st, down=True))]
        else:
            ends = [(lo, None), (None, self.n)]
            right = lambda j: S.eq(r, S.sub(S.zext(self.at(self.A, self.base, self.a, j), 32), S.zext(self.at(self.B, self.bbase, self.b, j), 32)))  # noqa: E731
            same = pinned(self, st, idx, lambda j: S.not_(self.differ(j)), ends)
            out = [('the answer the difference of the bytes at its index', Alt([[[S.not_(f)]] for f in pinned(self, st, idx, right, ends)])),
                   ('the index within n', [S.not_(S.ult(idx, self.n))]),
                   ('no difference before the index', self.some_diff(lo, idx, st, down=True, anchor=self.n)),
                   ('no difference at all when the bytes at the index are equal', Alt([[[f] + x for x in part] for f in same for part in
                                                                                     self.alt_parts(self.some_diff(lo, self.n, st, [f], down=True))]))]
        return out, self.show()  # the loads: each checked as it ran (early)

    def general(self, f):
        return [S.ule(S.sub(f, self.a), self.z)]

    def base_case(self, st, f, mode, step):
        lo = S.sub(st.front_lo, self.a) if st.front_lo is not None else const(64, 0)
        out = [('the frontier at or past a', [S.ult(f, self.a)])]
        if st.front_lo is not None:
            out.append(('the frontier at or past the previous loop\'s', [S.ult(f, st.front_lo)]))
        out.append(('no difference before the loop', self.some_diff(lo, S.sub(f, self.a), st)))
        return out

    def step_case(self, st, f, f2, mode, step, cov=None):
        return [('no difference in the iteration', self.some_diff(S.sub(st.front_lo, self.a), S.sub(f2, self.a), st))]

    def candidates(self, step):
        return [('the next iteration inside the object', lambda st, f: S.ule(S.add(S.sub(f, self.a), const(64, step)), self.n)),
                ('the frontier inside the object', lambda st, f: S.ule(S.sub(f, self.a), self.n))]

    def mode(self, solver, st, f, step):
        return 'B'

    def show(self):
        return [self.a, self.b, self.n, self.i, self.z]


class MemcmpLarge(Compare):
    """memcmp_large(a, b, n): memcmp's kernel, n > 128 and the first 64
    bytes equal: memcmp itself."""

    def setup(self, a, align):
        params, assume = Compare.setup(self, a, align)
        return params, assume + [S.ult(const(64, 128), self.n)] + [S.not_(self.differ(const(64, j))) for j in range(64)]

    def contracts(self, img, entry):
        return {}


class StrCopy(Scan, Writer):
    """strcpy(d, s): d; the bytes of s up to its NUL (L + 1 of them) at d,
    nothing else written, s read as if one byte at a time up to the NUL (its
    object may end there). z: any NUL of s (so the stores stay within
    [d, d + z] and the loads in the pages of [s, s + z] for the first
    one). At a return the copy's end E: the largest end of a store (or of
    what a loop covered), with s[E - 1] NUL, no NUL before it, [0, E) of d
    covered, every store's bytes s's at its index. s's low bits are the
    alignment class (its scan aligns its loads); d any pointer; the
    objects apart. A loop's frontier F scans s: no NUL in [s, F), and the
    stores so far cover d's indexes up to F - s from the first's."""
    bounded = False

    def setup(self, a, align):
        self.base, self.s, self.src = aligned_pointer('s', a, align)
        self.dbase, self.d, self.D = aligned_pointer('d', 0, 1)
        self.regions = [(self.base, self.src), (self.dbase, self.D)]
        self.z, self.i = var('z', 64), var('i', 64)
        s, d, z = self.s, self.d, self.z
        assume = [S.ule(const(64, USER_LO), s), S.ult(s, const(64, USER_HI)), S.ule(const(64, USER_LO), d), S.ult(d, const(64, USER_HI)),
                  self.at(z), S.ult(z, const(64, USER_HI - USER_LO)), S.ule(S.add(s, z), const(64, USER_HI)),
                  S.ult(S.add(d, z), const(64, USER_HI)), S.or_(S.ult(S.add(d, z), s), S.ult(S.add(s, z), d))]
        if not self.induct:
            assume.append(S.ule(z, const(64, self.maxlen)))
        return self.params(), assume

    def params(self):
        return [self.d, self.s]

    def contracts(self, img, entry):
        return {t: self.c_long for t in named_targets(img, entry, ('strcpy_long_',))}

    def c_long(self, prover, st):
        """strcpy_long(d, s, a), proved on its own (StrCopyLong): under its
        precondition it is strcpy(d, s), the whole copy its own, d its
        answer; the stores before it s's bytes within the copy (checked as
        they ran, and here their values)."""
        a = st.gpr['rdx']
        out = [('strcpy_long is given d', [S.not_(S.eq(st.gpr['rdi'], self.d))]),
               ('strcpy_long is given s', [S.not_(S.eq(st.gpr['rsi'], self.s))]),
               ('strcpy_long\'s a 16-aligned', [S.not_(S.is_zero(S.band(a, const(64, 15))))]),
               ('strcpy_long\'s a past s by 1..128', [S.not_(S.and_(S.ult(self.s, a), S.ule(a, S.add(self.s, const(64, 128)))))]),
               ('strcpy_long\'s a 32-aligned or past s by 16 at most', [S.not_(S.or_(S.is_zero(S.band(a, const(64, 31))), S.ule(S.sub(a, self.s), const(64, 16))))])]
        return tail_call(self, prover, st, out + self.base_case(st, a, 'B', 1))

    def at(self, off):
        return S.is_zero(S.select(self.src, S.add(S.sub(self.s, self.base), off)))

    def expected_at(self, idx):
        return S.select(self.src, S.add(S.sub(self.s, self.base), idx))

    def movs_wrong(self, idx, count, src, arr, off):
        return S.not_(S.eq(idx, S.sub(src, self.s)))

    def owner(self, addr):
        return owner(addr, ((self.s, self.base), (self.d, self.dbase)))

    def last_readable(self):
        return self.z

    def early(self, a, n):
        if self.owner(a) is self.d:
            return nonempty(n)  # d is not read
        return Scan.early(self, a, n)

    def early_store(self, e):
        a, nb, rip, c, lanes = e
        nb = nb if isinstance(nb, S.T) else const(64, nb)
        return S.and_(c, S.not_(S.is_zero(nb)), S.not_(S.and_(S.ule(self.d, a), S.ule(S.add(a, nb), S.add(S.add(self.d, self.z), const(64, 1))))))

    def store_stop_end(self, e):
        """For a store: E, so that only a NUL of s below E puts its end past
        d + z (early_store; the copy's last store, its end at the NUL a mask
        found); None for a count not a number."""
        a, nb, rip, c, lanes = e
        if not isinstance(nb, int):
            return None
        last = S.add(a, const(64, nb - 1))
        return S.ite(S.ult(self.d, last), S.sub(last, self.d), const(64, 0))

    def extent(self, st, ivs):
        """The end of the copy: the largest end of a store that ran (a masked
        store's lanes each their own), as a term."""
        e = const(64, 0)
        for _, b, c in ivs:
            e = S.ite(S.and_(c, S.ult(e, b)), b, e)
        return e

    def obligations(self, st):
        ivs = st.cover + self.intervals(st.stores)
        if not ivs:
            raise Failure('nothing written')
        e = self.extent(st, ivs)
        last = S.sub(e, const(64, 1))
        lo = S.sub(st.front_lo, self.s) if st.front_lo is not None else const(64, 0)
        out = [('the answer is d', [S.not_(S.eq(st.gpr['rax'], self.d))]),
               ('the copy ends with a NUL', [S.not_(self.at(last))]),
               ('no NUL before the end of the copy', self.some_match(lo, last, st))]
        if st.front_lo is not None:
            out.append(('the end at or past the frontier', [S.ult(last, lo)]))
        wv = self.wrong_values(st.stores)
        if wv is not S.FALSE:
            out.append(('every store writes the bytes of s', [wv]))
        out.append(('every byte of the copy written', self.uncovered(ivs, const(64, 0), e)))
        for f in st.front_bytes:
            out.append(('the byte at a loop\'s frontier in the string', [S.ult(S.add(self.s, last), f)]))
        return out + self.loads_unsafe(st, last), self.show()

    def loads_unsafe(self, st, last):
        out = []
        s = self.s
        unsafe = [S.and_(c, S.or_(S.ult(page(a), page(s)), S.ult(page(S.add(s, last)), page(S.add(a, const(64, n - 1))))))
                  for a, n, _, c in st.loads if self.owner(a) is s]
        if unsafe:
            out.append(('every load in the pages of the string', [S.or_(*unsafe)]))
        return out

    def loads_by(self, st, f, mode, step, what):
        """Nothing for the loops: every load is checked as it runs against
        the pages up to any NUL z, the general state's z at or past the
        frontier (the copy loops look ahead of it)."""
        return []

    def mode(self, solver, st, f, step):
        return 'B'

    def frontier_checks(self, st, f, lo):
        out = [('the frontier at or past s', [S.ult(f, self.s)])]
        if lo is not None:
            out.append(('the frontier at or past the previous loop\'s', [S.ult(f, lo)]))
        return out

    def base_case(self, st, f, mode, step):
        out = Scan.base_case(self, st, f, mode, step)
        wv = self.wrong_values(st.stores)
        if wv is not S.FALSE:
            out.append(('the stores before the loop write the bytes of s', [wv]))
        return out

    def step_case(self, st, f, f2, mode, step, cov=None):
        return Scan.step_case(self, st, f, f2, mode, step) + Writer.step_case(self, st, f, f2, mode, step, cov)

    def show(self):
        return [self.d, self.s, self.z, self.i]


class StrCopyLong(StrCopy):
    """strcpy_long(d, s, a): strcpy's kernel past its short path, the
    contract its callers rely on: a 16-aligned, s < a <= s + 128, a <= s +
    16 unless a is 32-aligned (avx2's kernel copies a NUL in the 16 bytes
    from a by copy_small, 32 bytes at most: every caller passes align_down(s,
    16) + 16 or a 32-aligned a), no NUL in [s, a) (a precondition: z, any
    NUL, at or past a - s, and the scan's frontier there from the start);
    then strcpy(d, s), d its answer."""

    def contracts(self, img, entry):
        return {}

    def setup(self, a, align):
        params, assume = StrCopy.setup(self, a, align)
        delta = var('adelta', 64)  # a = s + delta: an address in s's region
        self.a = S.add(self.s, delta)
        assume += [S.is_zero(S.band(self.a, const(64, 15))), S.ult(const(64, 0), delta), S.ule(delta, const(64, 128)), S.ule(delta, self.z),
                   S.or_(S.is_zero(S.band(self.a, const(64, 31))), S.ule(delta, const(64, 16)))]  # past s by 16 at most unless 32-aligned
        assume += [S.or_(S.ule(delta, const(64, j)), S.not_(self.at(const(64, j)))) for j in range(128)]  # no NUL in [s, a)
        return [self.d, self.s, self.a], assume


def contract_targets(img, entry):
    """The functions a function calls (direct calls and jumps out of it)."""
    out, seen, todo = set(), set(), [entry]
    while todo:
        a = todo.pop()
        while a not in seen:
            seen.add(a)
            j = img.decode(a)
            mn = j['mnemonic']
            if mn == 'int3':
                break
            tgt = j.get('branch_target')
            if mn == 'call' and tgt is not None:
                out.add(tgt)
            elif tgt is not None:
                todo.append(tgt)
            if mn in ('ret', 'jmp'):
                break
            a += j['len']
    return out


def named_targets(img, entry, fns):
    """{address: the function's name past its configuration} of the functions
    of those names (a name ending in _: any it begins) the code from entry
    reaches by a call or a jump (a tail call), entry itself never; not
    followed into them."""
    names = {a: n for n, a in img.symbols.items()}
    out, seen, todo = {}, set(), [entry]
    while todo:
        a = todo.pop()
        while a not in seen:
            seen.add(a)
            j = img.decode(a)
            if j['mnemonic'] == 'int3':
                break
            tgt = j.get('branch_target')
            if tgt is not None:
                fn = names.get(tgt, '').split('.', 1)[-1]
                if tgt != entry and (fn in fns or any(fn.startswith(x) for x in fns if x.endswith('_'))):
                    out[tgt] = fn
                else:
                    todo.append(tgt)
            if j['mnemonic'] in ('ret', 'jmp'):
                break
            a += j['len']
    return out


class StrCat(Writer):
    """strcat(d, s): d; s with its NUL copied over d's NUL on, nothing else
    written: d's bytes before its NUL kept, d[P + j] = s[j] for j up to s's
    NUL, P the length of d. d and s read as if one byte at a time up to
    their NULs (z0 and z: any NULs). The callees are taken by their
    contracts (strlen, strcpy and strcpy_long, each proved on its own): a
    call's answer a fresh length with its NUL and no NUL before it (a fact
    the obligations use at the indexes they need), its stores one copy of
    s. The loops of strcat itself are short: unrolled. At the return the
    copy's start P (the lowest store) and end E (the highest): d's NUL at P
    and none before it, s's at E - P - 1 and none before it, [P, E) covered
    with s's bytes. d's low bits are the alignment class; s any pointer;
    the objects apart."""
    unroll = 4

    def setup(self, a, align):
        self.base, self.d, self.D = aligned_pointer('d', a, align)
        self.sbase, self.s, self.S = aligned_pointer('s', 0, 1)
        self.regions = [(self.base, self.D), (self.sbase, self.S)]
        self.z0, self.z, self.i = var('z0', 64), var('z', 64), var('i', 64)
        d, s, z0, z = self.d, self.s, self.z0, self.z
        top = const(64, USER_HI)
        assume = [S.ule(const(64, USER_LO), d), S.ult(d, top), S.ule(const(64, USER_LO), s), S.ult(s, top),
                  self.dnul(z0), self.snul(z), S.ult(z0, const(64, USER_HI - USER_LO)), S.ult(z, const(64, USER_HI - USER_LO)),
                  S.ult(S.add(S.add(d, z0), z), top), S.ule(S.add(s, z), top),
                  S.or_(S.ult(S.add(S.add(d, z0), z), s), S.ult(S.add(s, z), d))]
        if not self.induct:
            assume += [S.ule(z0, const(64, self.maxlen)), S.ule(z, const(64, self.maxlen))]
        self.calls = 0
        return [d, s], assume

    def dbyte(self, j):
        return S.select(self.D, S.add(S.sub(self.d, self.base), j))

    def sbyte(self, j):
        return S.select(self.S, S.add(S.sub(self.s, self.sbase), j))

    def dnul(self, j):
        return S.is_zero(self.dbyte(j))

    def snul(self, j):
        return S.is_zero(self.sbyte(j))

    def owner(self, addr):
        return owner(addr, ((self.d, self.base), (self.s, self.sbase)))

    def early(self, a, n):
        p = self.owner(a)
        last = self.z0 if p is self.d else self.z
        return S.and_(nonempty(n), S.or_(S.ult(page(a), page(p)), S.ult(page(S.add(p, last)), page(last_of(a, n)))))

    def early_store(self, e):
        a, nb, rip, c, lanes = e
        nb = nb if isinstance(nb, S.T) else const(64, nb)
        end = S.add(S.add(S.add(self.d, self.z0), self.z), const(64, 1))
        return S.and_(c, S.not_(S.is_zero(nb)), S.not_(S.and_(S.ule(self.d, a), S.ule(S.add(a, nb), end))))

    # the callees by their contracts
    def contracts(self, img, entry):
        names = {a: n for n, a in img.symbols.items()}
        out = {}
        for tgt in contract_targets(img, entry):
            n = names.get(tgt, '')
            fn = n.split('.', 1)[-1]
            if fn.startswith('strlen'):
                out[tgt] = self.c_strlen
            elif fn.startswith('strcpy_long'):
                out[tgt] = self.c_strcpy_long
            elif fn.startswith('strcpy'):
                out[tgt] = self.c_strcpy
        return out

    def c_strlen(self, prover, st):
        """strlen(d): R, d[R] NUL, none before (a fact), R <= z0."""
        prover.prove(st, [('strlen is called on d', [S.not_(S.eq(st.gpr['rdi'], self.d))])], self.show())
        if st.stores:
            raise Failure('strlen of d after stores')
        self.calls += 1
        r = var('len%d' % self.calls, 64)
        st.pc += [self.dnul(r), S.ule(r, self.z0)]
        st.facts = getattr(st, 'facts', []) + [('d', const(64, 0), r)]
        st.dfront = r
        X.call_return(st, r, 'c%d_' % self.calls)

    def copy_call(self, prover, st, what):
        """The copy of s to dst = d + d's length: L, s[L] NUL, none before,
        L <= z; d's array the copy over it, one store of L + 1 bytes."""
        dst, src = st.gpr['rdi'], st.gpr['rsi']
        front = getattr(st, 'dfront', None)
        q = S.sub(dst, self.d)
        if front is None:  # d's length found in line: dst at a NUL of d, none before it
            dpred = lambda j: S.and_(self.dnul(j), *self.instances(st, 'd', j))  # noqa: E731
            prover.prove(st, [('%s copies to a NUL of d' % what, [S.not_(self.dnul(q))]),
                              ('no NUL of d before where %s copies to' % what, span(self, st, const(64, 0), q, dpred))], self.show())
            st.facts = getattr(st, 'facts', []) + [('d', const(64, 0), q)]
            st.dfront = front = q
        prover.prove(st, [('%s copies from s' % what, [S.not_(S.eq(src, self.s))]),
                          ('%s copies to d\'s end' % what, [S.not_(S.eq(q, front))])], self.show())
        self.calls += 1
        L = var('len%d' % self.calls, 64)
        st.pc += [self.snul(L), S.ule(L, self.z)]
        st.facts = getattr(st, 'facts', []) + [('s', const(64, 0), L)]
        st.sfront = L
        x = var('xc%d' % self.calls, 64)
        q = S.sub(dst, self.base)
        n = S.add(L, const(64, 1))
        old = st.regions[0][1]
        st.regions[0] = (self.base, S.lam(x, S.ite(S.and_(S.ule(q, x), S.ult(x, S.add(q, n))),
                                                  S.select(self.S, S.add(S.sub(self.s, self.sbase), S.sub(x, q))), S.select(old, x))))
        st.stores.append((dst, n, st.rip, S.TRUE, ('movs', src, self.S, S.sub(src, self.sbase))))
        X.call_return(st, dst, 'c%d_' % self.calls)
        return L

    def c_strcpy(self, prover, st):
        self.copy_call(prover, st, 'strcpy')

    def c_strcpy_long(self, prover, st):
        """strcpy_long(dst, s, a): its precondition first (StrCopyLong's)."""
        a = st.gpr['rdx']
        prover.prove(st, [('strcpy_long\'s a 16-aligned', [S.not_(S.is_zero(S.band(a, const(64, 15))))]),
                          ('strcpy_long\'s a past s by 1..128', [S.not_(S.and_(S.ult(self.s, a), S.ule(a, S.add(self.s, const(64, 128)))))]),
                          ('strcpy_long\'s a 32-aligned or past s by 16 at most',
                           [S.not_(S.or_(S.is_zero(S.band(a, const(64, 31))), S.ule(S.sub(a, self.s), const(64, 16))))]),
                          ('no NUL of s before strcpy_long\'s a', span(self, st, const(64, 0), S.sub(a, self.s), self.snul))], self.show())
        L = self.copy_call(prover, st, 'strcpy_long')
        st.pc.append(S.ule(S.sub(a, self.s), L))

    def instances(self, st, which, j):
        """The facts (no NUL in [lo, hi) of d or s) at index j."""
        out = []
        for w, lo, hi in getattr(st, 'facts', []):
            if w == which:
                out.append(S.not_(S.and_(S.ule(lo, j), S.ult(j, hi), (self.dnul if w == 'd' else self.snul)(j))))
        return out

    def expected_at(self, idx):
        return self.sbyte(S.sub(idx, self.P))

    def movs_wrong(self, idx, count, src, arr, off):
        if arr is not self.S:
            raise Failure('a copy from other than s')
        return S.not_(S.eq(S.sub(idx, self.P), S.sub(src, self.s)))

    def obligations(self, st):
        ivs = self.intervals(st.stores)
        if not ivs:
            raise Failure('nothing written')
        P = self.first(st, ivs)
        self.P = P
        E = StrCopy.extent(self, st, ivs)
        last = S.sub(S.sub(E, P), const(64, 1))  # s's NUL
        # no NUL before them: the bytes the path checked, the callees' facts
        dpred = lambda j: S.and_(self.dnul(j), *self.instances(st, 'd', j))  # noqa: E731
        spred = lambda j: S.and_(self.snul(j), *self.instances(st, 's', j))  # noqa: E731
        out = [('the answer is d', [S.not_(S.eq(st.gpr['rax'], self.d))]),
               ('the copy starts at a NUL of d', [S.not_(self.dnul(P))]),
               ('no NUL of d before the copy', span(self, st, const(64, 0), P, dpred)),
               ('the copy ends with a NUL of s', [S.not_(self.snul(last))]),
               ('no NUL of s before the end of the copy', span(self, st, const(64, 0), last, spred))]
        wv = self.wrong_values(st.stores)
        if wv is not S.FALSE:
            out.append(('every store writes the bytes of s', [wv]))
        out.append(('every byte of the copy written', self.uncovered(ivs, P, E)))
        return out, self.show()

    def first(self, st, ivs):
        """The start of the copy: the smallest start of a store that ran."""
        p = const(64, (1 << 64) - 1)
        for a, _, c in ivs:
            p = S.ite(S.and_(c, S.ult(a, p)), a, p)
        return p

    def base_case(self, *a):
        raise Failure('a loop in strcat itself past the unrolling')

    def show(self):
        return [self.d, self.s, self.z0, self.z, self.i]


SETMAX = 32  # strspn and strcspn: sets of up to this many bytes


class Span(Scan):
    """strspn(s, set) (in = True: the length of s's prefix of set bytes) and
    strcspn (in = False: of bytes not in it, the NUL ending it). The set a
    string of L <= SETMAX bytes (L its first NUL, written out byte by byte),
    a byte in it when it equals one of the L: no quantifier anywhere. A stop
    of s: for strspn a byte not in the set (the NUL among them), for strcspn
    one in it or the NUL; the scan's obligations for the answer, s read up
    to the stop's page, the set up to its NUL's. Loops over the set and the
    stack (tables built from it) run out unrolled; the scan of s by
    induction. s's low bits are the alignment class; set any pointer. Where
    the code hands the rest to truffle_scan32 (and an empty set to strlen)
    those are taken by their contracts, proved on their own: its registers
    and where it starts checked at the jump, the rest of the path theirs."""
    bounded = False
    spn = True
    lane_terms = True
    setmax = SETMAX  # the short spec's longest set, the long spec's shortest less one (--setmax: sse2's table scan from 2 bytes)

    def __init__(self, maxlen, induct):
        Scan.__init__(self, maxlen, induct)
        self.abstraction = {}  # member(b) -> an uninterpreted predicate's: obligations with the lane facts need no more of it
        self.canon = 0

    def setup(self, a, align):
        self.base, self.s, self.S = aligned_pointer('s', a, align)
        self.tbase, self.t, self.T = aligned_pointer('set', 0, 1)
        self.regions = [(self.base, self.S), (self.tbase, self.T)]
        self.z, self.i, self.L = var('z', 64), var('i', 64), var('L', 64)
        s, t, z, L = self.s, self.t, self.z, self.L
        top = const(64, USER_HI)
        assume = [S.ule(const(64, USER_LO), s), S.ult(s, top), S.ule(const(64, USER_LO), t), S.ult(t, top),
                  S.ule(L, const(64, self.setmax)), S.is_zero(self.tbyte(L)), S.ule(S.add(t, L), top)]
        assume += [S.or_(S.ule(L, const(64, m)), S.not_(S.is_zero(self.tbyte(const(64, m))))) for m in range(self.setmax)]  # L the first NUL
        assume += [self.at(z), S.ult(z, const(64, USER_HI - USER_LO)), S.ule(S.add(s, z), top)]
        if not self.induct:
            assume.append(S.ule(z, const(64, self.maxlen)))
        return [s, t], assume

    def tbyte(self, j):
        return S.select(self.T, S.add(S.sub(self.t, self.tbase), j))

    def canonical(self, prover, st):
        """At a loop's first arrival: a vector register holding the set
        (16 of its bytes from j on, as loaded: shuffled near a page end by
        set & 15) becomes, where the path proves its lanes are those bytes
        up to the set's NUL, those bytes there and unknown bytes past L: the
        same lanes up to the NUL, anything after it (more states: sound;
        pcmpistri reads a set up to its NUL). The loop and its exits then
        meet the set as the spec's terms, not the shuffle's."""
        if not self.lane_terms:  # SpanLong's: proved as it was
            return
        for n, lanes in list(st.vec.items()):
            lo = lanes[:16]
            if not any('set_mem' in S.variables(x) for x in lo):
                continue
            for j in (0, 16):
                want = [self.tbyte(const(64, j + k)) for k in range(16)]
                if all(x is w or x.op == 'ite' and x.args[1] is w for x, w in zip(lo, want)):
                    break  # as the spec has them already
                # the window must hold a byte of the set before its NUL: past it the claim is empty
                # (a 1-byte set's needle, its byte 16 times, is "the set's bytes from 16" for any set shorter)
                if j and prover.solver.check(*st.pc, S.ult(self.L, const(64, j)), timeout=6 * AUX) != 'unsat':
                    continue
                bad = S.or_(*[S.and_(S.ule(const(64, j + k), self.L), S.not_(S.eq(x, w))) for k, (x, w) in enumerate(zip(lo, want))])
                if prover.solver.check(*st.pc, bad, timeout=6 * AUX) == 'unsat':
                    self.canon += 1
                    st.vec[n] = [S.ite(S.ule(const(64, j + k), self.L), w, var('canon%d_%d' % (self.canon, k), 8)) for k, w in enumerate(want)] + lanes[16:]
                    if prover.args.verbose:
                        print('    zmm%d: the set\'s bytes from %d up to its NUL, unknown past it' % (n, j), flush=True)
                    break

    def contracts(self, img, entry):
        got = named_targets(img, entry, ('truffle_scan32', 'strlen_'))
        if 'truffle_scan32' not in got.values():  # no shuffle scan in this unit: its code walked as it is
            return {}
        return {t: self.c_scan32 if fn == 'truffle_scan32' else self.c_strlen for t, fn in got.items()}

    def c_scan32(self, prover, st):
        """truffle_scan32(from, s, set1, set2, count, members): its contract
        (TruffleScan, TruffleScan32) says the answer is from - s plus the
        offset from `from` of the first byte that stops it, its loads in
        the pages from from's up to that byte's. Here that byte stops this
        spec too when count and the two registers are the set, for every
        byte; and with no stop in [s, from), it is s's first: the answer
        and the loads this spec asks for, the rest of the path."""
        frm, x = st.gpr['rdi'], var('cx', 8)
        lanes = X.read_vec(st, 'xmm0')[:16] + X.read_vec(st, 'xmm1')[:16]
        count = st.gpr['rdx']
        want = S.or_(*[S.and_(S.ult(const(64, k), count), S.eq(lanes[k], x)) for k in range(SETMAX)])
        out = [('truffle_scan32 is given s', [S.not_(S.eq(st.gpr['rsi'], self.s))]),
               ('truffle_scan32 stops at what the spec does', [S.not_(S.eq(S.extract(st.gpr['rcx'], 31, 0), const(32, 0 if self.spn else 1)))]),
               ('truffle_scan32\'s count at most its two registers', [S.ult(const(64, SETMAX), count)]),
               ('truffle_scan32\'s registers and count the set', [S.not_(S.eq(self.member(x), want))])]
        out += self.base_case(st, frm, 'B', 1)
        st2 = st.copy()
        st2.pc = st.pc + self.general(frm)
        prover.later(prover.prove, st2, [('truffle_scan32\'s from in user space', [S.ult(const(64, USER_HI), frm)])], self.show())
        return tail_call(self, prover, st, out)

    def c_strlen(self, prover, st):
        """strcspn with an empty set: the first NUL of s."""
        out = [('strlen is given s', [S.not_(S.eq(st.gpr['rdi'], self.s))]),
               ('strlen\'s stop the spec\'s: the set empty', [self.member(var('cx', 8))])]
        if self.spn:
            raise Failure('strlen from strspn')
        return tail_call(self, prover, st, out)

    def lane_facts(self, prover, st):
        """A byte b of s the path tests by a computation of b alone (truffle:
        a nibble table's lane, the high nibble's bit) has a term c(b) of one
        bit: with c at a free byte x proved, once, to say x is in the set (or
        is not) whatever x, the fact c(b) = member(b) for each such b; when
        no one term of b does, their disjunction (pcmpistrm on the set's two
        halves of 16: "equals a valid byte of this half"). The
        obligations meet the path's terms through those, not by matching
        the table against the set for every byte (strspn avx2 past 2048
        bytes: 466 s locally, unknown under load). For checks: sound, an
        instance of what was proved for every x."""
        if not self.lane_terms:  # SpanLong's member is its mask M's
            return []
        if not hasattr(self, 'lanes'):
            self.lanes = {'one': {}, 'cands': {}, 'node': {}, 'lemma': {}, 'f': {}, 'parts': {}, 'x': var('lane_x', 8)}
        L = self.lanes
        if 'scope' not in L:  # what a membership test may read besides the byte: the set (its bytes, its length, the lanes past its NUL)
            L['scope'] = frozenset().union(*[S.variables(m) for m in self.lane_forms(L['x'])]) - S.variables(L['x']) | {'L', 'set_hi'}
        in_scope = lambda names: all(v in L['scope'] or v.startswith('canon') for v in names)  # noqa: E731
        found = set()
        for c in st.pc:
            got = L['cands'].get(c.id)
            if got is None:
                got = L['cands'][c.id] = self.lane_scan(c)
            found |= got
        if not found:
            return []
        setpc = []  # the path's conditions on the set alone (a table built under them): when what holds for every path does not

        def sign_of(f):
            # a term that reads anything but the byte and the set (a lane of a block shuffled by s & 15)
            # says nothing of the byte alone: no lemma, no query
            sign = L['lemma'].get(f.id)
            if sign is None:
                sign = L['lemma'][f.id] = self.lane_lemma(prover, f, []) if in_scope(S.variables(f) - S.variables(L['x'])) else -1
            if sign == 0:
                if not setpc:
                    setpc.extend(c for c in st.pc if in_scope(S.variables(c)))
                key = ('given', f.id, frozenset(c.id for c in setpc))  # the same few for every path the set's loading takes one way
                sign = L['lemma'].get(key)
                if sign is None:
                    sign = L['lemma'][key] = self.lane_lemma(prover, f, setpc)
            return max(sign, 0)

        out, per_byte = [], {}
        subst = L['last_subst'] = {}  # node -> the abstraction of what its lemma says (the facts' own terms, for abstract_of)
        for n in found:
            node, b = L['node'][n], L['node'][L['one'][n]]
            f = L['f'].get(n)
            if f is None:
                f = L['f'][n] = S.subst(node, b, L['x'])
            per_byte.setdefault(b.id, (b, []))[1].append((f, node))
        for b, terms in per_byte.values():
            terms.sort(key=lambda t: t[0].id)  # the same order for every byte the same code tests
            done = False
            for f, node in terms:
                sign = sign_of(f)
                if sign:
                    form = self.lane_forms(b)[sign - 1]
                    out.append(S.eq(self.lane_bool(node), form))
                    done = True
                    ab = self.abstraction.get(self.member(b).id) if sign <= 2 else None
                    if ab is not None:
                        subst[node.id] = (ab if sign == 1 else S.not_(ab)) if node.w == S.BOOL else S.ite(ab if sign == 1 else S.not_(ab), const(1, 1), const(1, 0))
            if not done and len(terms) > 1:  # a set in parts, each tested on its own (pcmpistrm on its two halves of 16)
                # the outermost terms alone: one inside another (a lane's compare with one set byte, under the
                # test of the half) is not a part of its own
                key = tuple(node.id for _, node in terms)
                parts = L['parts'].get(key)
                if parts is None:
                    inner = set()
                    for f, node in terms:
                        stack = [node]
                        while stack:
                            t = stack.pop()
                            for y in t.args:
                                if isinstance(y, S.T) and y.id not in inner:
                                    inner.add(y.id)
                                    stack.append(y)
                    parts = L['parts'][key] = [(f, node) for f, node in terms if node.id not in inner]
                # the halves: a pair of them first (the byte's NUL test is a part too, of no half)
                for size in range(2, len(parts) + 1):
                    for group in itertools.combinations(parts, size):
                        f = S.or_(*[self.lane_bool(f) for f, _ in group])
                        sign = sign_of(f)
                        if sign:
                            c = S.or_(*[self.lane_bool(node) for _, node in group])
                            out.append(S.eq(c, self.lane_forms(b)[sign - 1]))
                            break
                    else:
                        continue
                    break
        return out

    @staticmethod
    def lane_bool(t):
        return t if t.w == S.BOOL else S.eq(t, const(1, 1))

    def lane_forms(self, b):
        """What a one-bit term of b may say of it, for every b."""
        return [self.member(b), S.not_(self.member(b))]

    def lane_lemma(self, prover, f, given):
        """The place in lane_forms (from 1) of what f(x) says of x for every
        x, 0 when none is proved (in 120 s: once a term, the tables of a
        32-byte set take more than 20)."""
        x, fb, t0 = self.lanes['x'], self.lane_bool(f), time.time()
        rs, sign = [], 0
        for k, m in enumerate(self.lane_forms(x)):
            rs.append(prover.solver.check(*given, S.not_(S.eq(fb, m)), timeout=6 * AUX))
            if rs[-1] == 'unsat':
                sign = k + 1
                break
        if prover.args.verbose and (sign or 'unknown' in rs):
            print('   lane lemma (%d variables%s): %s, form %d %.1f s' % (len(S.variables(f)), ', under the path' if given else '', '/'.join(rs), sign, time.time() - t0),
                  flush=True)
        return sign

    def lane_scan(self, root):
        """The largest terms under root that read one byte of s and no other,
        those of one bit: their ids (the byte's in lanes['one'])."""
        L = self.lanes
        one, node, out, seen, stack = L['one'], L['node'], set(), set(), [root]
        while stack:
            t = stack[-1]
            if t.id in one:
                stack.pop()
                if t.id not in seen:
                    seen.add(t.id)
                    if one[t.id] == -1:  # more than one byte: the largest ones of one are below
                        stack.extend(y for y in t.args if isinstance(y, S.T) and y.id not in seen)
                    elif one[t.id] is not None and one[t.id] != t.id and t.w in (S.BOOL, 1) and not S.is_const(t):
                        out.add(t.id)
                continue
            kids = [y for y in t.args if isinstance(y, S.T) and y.id not in one]
            if kids:
                stack.extend(kids)
                continue
            if t.op == 'select' and t.args[0] is self.S:
                b = t.id
            elif t.op == 'var' and not (t.args[0] in L['scope'] or t.args[0].startswith('canon')):
                b = -1  # a variable no membership test reads (a shuffled lane's guard of s_hi): the test is below it
            else:
                bs = {one[y.id] for y in t.args if isinstance(y, S.T)} - {None}
                b = -1 if -1 in bs or len(bs) > 1 else next(iter(bs)) if bs else None
            one[t.id], node[t.id] = b, t
        return frozenset(out)

    def member(self, b):
        """b one of the set's bytes before its NUL: written as the machine's
        pcmpistri writes it (a set byte valid when no NUL is at or before
        it), so that the path's terms and the spec's are the same terms."""
        tb = [self.tbyte(const(64, k)) for k in range(self.setmax)]
        m = S.or_(*[S.and_(S.not_(S.or_(*[S.is_zero(x) for x in tb[:k + 1]])), S.eq(tb[k], b)) for k in range(self.setmax)])
        if self.lane_terms and m.id not in self.abstraction:  # what an obligation is first asked with in its place
            self.abstraction[m.id] = S.not_(S.is_zero(S.select(S.mem_var('member_of'), S.zext(b, 64))))
        return m

    def at(self, off):
        b = S.select(self.S, S.add(S.sub(self.s, self.base), off))
        if self.spn:
            return S.not_(self.member(b))
        # member(b) kept a term of its own (or_ would flatten its disjuncts into this one): the abstraction
        # replaces it by the uninterpreted predicate, as it does strspn's not member(b)
        return S._mk('or_b', S.BOOL, (self.member(b), S.is_zero(b)))

    def owner(self, addr):
        return owner(addr, ((self.s, self.base), (self.t, self.tbase)))

    def early(self, a, n):
        p = self.owner(a)
        last = self.z if p is self.s else self.L
        return S.and_(nonempty(n), S.or_(S.ult(page(a), page(p)), S.ult(page(S.add(p, last)), page(last_of(a, n)))))

    def loads_unsafe(self, st, last):
        bad = []
        for a, n, _, c in st.loads:
            p = self.owner(a)
            end = last if p is self.s else self.L
            bad.append(S.and_(c, S.or_(S.ult(page(a), page(p)), S.ult(page(S.add(p, end)), page(last_of(a, n))))))
        if st.stores:
            raise Failure('stores at %s' % ', '.join('%#x' % e[2] for e in st.stores))
        return [('every load in the pages of its string up to the answer or the set\'s NUL', [S.or_(*bad)])] if bad else []

    def loads_by(self, st, f, mode, step, what):
        return []  # each load checked as it ran against the pages up to any stop z

    def mode(self, solver, st, f, step):
        return 'B'

    def frontier_checks(self, st, f, lo):
        out = [('the frontier at or past s', [S.ult(f, self.s)])]
        if lo is not None:
            out.append(('the frontier at or past the previous loop\'s', [S.ult(f, lo)]))
        return out

    def params(self):
        return [self.s, self.t]

    def answer(self, st):
        return S.TRUE, st.gpr['rax']

    def keep_unrolling(self, prev, now):
        """A loop that read nothing of s since its last arrival: over the set
        or the stack, bounded by the set's length; it runs out unrolled."""
        return not any(self.owner(a) is self.s for a, _, _, _ in now.loads[len(prev.loads):])

    def show(self):
        return [self.s, self.t, self.z, self.L, self.i]


class Cspan(Span):
    spn = False


class SpanLong(Span):
    """strspn and strcspn for sets longer than SETMAX bytes, any length: the
    set's length L its first NUL and a byte in the set when it equals one
    of the L, both with quantifiers (z3 takes those queries). The loop that
    builds the table from the set is generalized as a side loop: its
    invariant, the table's bit for every byte x set exactly when x is one of
    the set's bytes before the frontier (and the NUL for strcspn when the
    table has it), is one of the candidates read off the first iteration's
    stores to the stack (a byte table, sse2's; a bit table at
    16 (x >> 7) + (x & 15), bit (x >> 4) & 7, the avx2 units' truffle), kept
    Houdini's way like the others. A byte in the set is a bit of M, 256 bits
    defined by one axiom (bit x set exactly when x is one of the set's L
    bytes): the spec's terms have no quantifier. Past the loop (the first
    load of s) the table's invariant, its frontier at L, is the table
    holding M: 256 facts without a quantifier, what the scan's lookups
    read."""
    lane_terms = False

    def contracts(self, img, entry):
        return {}

    def setup(self, a, align):
        self.M = var('inset', 256)
        params, assume = Span.setup(self, a, align)
        L, t, top = self.L, self.t, const(64, USER_HI)
        qk, qm = var('qk0', 64), var('qm', 8)
        assume = [c for c in assume if c is not S.ule(L, const(64, self.setmax))]  # the short form's bound (its first bytes' facts stay: instances)
        assume += [S.ult(const(64, self.setmax), L), S.ult(L, const(64, USER_HI - USER_LO)),
                   S.forall_(qk, S.or_(S.ule(L, qk), S.not_(S.is_zero(self.tbyte(qk))))),  # no NUL before L
                   *[S.or_(S.ule(L, const(64, m)), S.not_(S.is_zero(self.tbyte(const(64, m))))) for m in range(self.setmax, 2 * self.setmax)],  # instances
                   S.forall_(qm, S.eq(self.member(qm), self.in_first(qm, L)))]  # M's definition
        self.side_base = self.tbase
        self.table_forms, self.table_v = {}, {}
        self.tables = {}  # a table's invariant (a term) -> its memory, kind, offset, NUL, count
        return params, assume

    def member(self, b):
        if S.is_const(b):
            return S.eq(S.extract(self.M, b.val, b.val), const(1, 1))
        return S.eq(S.extract(S.lshr(self.M, S.zext(b, 256)), 0, 0), const(1, 1))

    def settle(self, prover, st, new):
        """At the first load of s past the set loop, a table's invariant among
        the path's facts (about the general state's memory, its count n):
        L is n + j for a small j (one instance of no NUL before L says it),
        and then the table now (the last bytes stored since) holds M, for
        any byte x and any truth p for "x one of the first n" (the
        invariant's instance and M's definition with p): written out for
        the 256 bytes in the invariant's place, no quantifier left. One
        that does not hold so is dropped (fewer facts: sound), and its loop
        explored again without it (a candidate the first arrival kept that
        a step would drop: the empty table is every table)."""
        if not any(self.owner(a) is self.s for a, _, _, _ in new):
            return
        for i, c in enumerate(st.pc):
            if c not in self.tables:
                continue
            gmem, kind, off, d, nul, n = self.tables[c]
            plain = [y for y in prover.solver.assumed + st.pc if not S.has_quantifier(y)]
            for j in range(3):
                m = S.add(n, const(64, j))
                inst = S.or_(S.ule(self.L, m), S.not_(S.is_zero(self.tbyte(m))))
                if prover.solver._check(*plain, inst, S.not_(S.eq(m, self.L)), bare=True, timeout=AUX) == 'unsat':
                    break
            else:
                continue
            x, p = var('sx', 8), var('sp', S.BOOL)
            cur = st.regions[-1][1]
            ih = self.table_is(gmem, kind, off, x, S.or_(p, S.and_(S.boolean(nul), S.is_zero(x))))
            mdef = S.eq(self.member(x), S.or_(p, *[S.eq(self.tbyte(S.add(n, const(64, e))), x) for e in range(j)]))
            for now in (nul, not nul):  # the NUL's entry as in the loop, or as a store since set it (strcspn's)
                goal = self.table_is(cur, kind, off, x, S.or_(self.member(x), S.and_(S.boolean(now), S.is_zero(x))))
                if prover.solver._check(*plain, ih, mdef, S.not_(goal), bare=True, timeout=6 * AUX) == 'unsat':
                    break
            else:
                del st.pc[i]
                name = next(nm for nm, form in self.table_forms.items() if form == (kind, off, d, nul))
                g = st.gen
                while g is not None and name not in [nm for nm, _ in g['kept']]:
                    g = g['parent']
                if g is not None:
                    prover.lost.setdefault(g['head'], set()).add(name)
                if prover.args.verbose:
                    print('    past the set loop: not kept: %s' % name, flush=True)
                return self.settle(prover, st, new)
            st.pc[i:i + 1] = [self.table_is(cur, kind, off, const(8, y), S.or_(self.member(const(8, y)), S.boolean(now and y == 0)))
                              for y in range(256)]
            if prover.args.verbose:
                print('    past the set loop: the %s table at rsp0%+d holds the set%s (L = n + %d; 256 facts)' % (kind, off, ' and the NUL' if now else '', j),
                      flush=True)
            return self.settle(prover, st, new)

    def keep_unrolling(self, prev, now):
        """Only a loop that reads neither string (the table's zeroing)."""
        return not now.loads[len(prev.loads):]

    def in_first(self, x, n):
        """x one of the set's first n bytes (written out for a known small n)."""
        if S.is_const(n) and n.val <= 64:
            return S.or_(*[S.eq(self.tbyte(const(64, j)), x) for j in range(n.val)])
        k = var('qk', 64)
        return S.exists_(k, S.and_(S.ult(k, n), S.eq(self.tbyte(k), x)))

    def table_holds(self, st, name, f):
        """A table's invariant where the frontier's count is known (the first
        arrival): for any byte x, no quantifier. None otherwise."""
        kind, c, d, nul = self.table_forms[name]
        n = S.add(S.sub(f, self.t), const(64, d & S.mask(64)))
        if not S.is_const(n):
            return None
        x = var('bx', 8)
        want = self.in_first(x, n)
        if nul:
            want = S.or_(want, S.is_zero(x))
        plain = [c for c in st.pc if not S.has_quantifier(c)]
        return self.solver._check(*plain, S.not_(self.table_is(st.regions[-1][1], kind, c, x, want)), bare=True,
                                  timeout=6 * AUX) == 'unsat'

    def side_base_case(self, st, f, step):
        return [('the set loop\'s frontier in the set', [S.ult(f, self.t)])]

    def side_step_case(self, st, f, f2, step):
        return []

    def side_candidates(self, s1, s2, a1, step):
        """The frontier within the set; the table's invariants: a table the
        code cleared before the loop (a run of stack bytes at constant
        offsets, 256 of them for a byte table, 32 for a bit table), holding
        the set's bytes before the frontier (or the one before it: a loop
        that stores the byte it loaded the time before), and the NUL or not."""
        t, L = self.t, self.L
        out = [('the set frontier at most its length', lambda st, f: S.ule(S.sub(f, t), L))]
        tb = S.sub(t, self.tbase)
        for r, v in s1.gpr.items():  # a register holding a set byte near the frontier (loaded the time before, stored this time)
            if v.op == 'zext' and v.args[0].op == 'select' and v.args[0].args[0] is self.T:
                d = S.sub(v.args[0].args[1], S.add(tb, S.sub(a1, t)))
                if S.is_const(d):
                    out.append(('%s the set byte at F%+d' % (r, S.signed(d)),
                                (lambda r, d, w: lambda st, f: S.eq(st.gpr[r], S.zext(self.tbyte(S.add(S.sub(f, t), d)), w)))(r, d, v.w)))
        offs, m = set(), s1.regions[-1][1]
        while m.op == 'store':
            if S.is_const(m.args[1]):
                offs.add(S.signed(m.args[1]))
            m = m.args[0]
        runs, start = [], None
        for o in sorted(offs):
            if start is None or o != prev + 1:
                if start is not None:
                    runs.append((start, prev + 1))
                start = o
            prev = o
        if start is not None:
            runs.append((start, prev + 1))
        forms = []
        for lo, hi in runs:
            for kind, size in (('byte', 256), ('bit', 32)):
                if hi - lo >= size:
                    forms += [(kind, lo), (kind, hi - size)]
        vals, m2 = set(), s2.regions[-1][1]  # the values the iteration stores at computed places: one constant, a byte table's v
        while m2.op == 'store' and m2 is not s1.regions[-1][1]:
            if not S.is_const(m2.args[1]):
                vals.add(m2.args[2])
            m2 = m2.args[0]
        v = next(iter(vals)) if len(vals) == 1 and S.is_const(next(iter(vals))) and next(iter(vals)).val else None
        for kind, c in sorted(set(forms)):
            if kind == 'byte' and v is not None:
                self.table_v[(kind, c)] = v
            for d in (-1, 0):
                for nul in (False, True):
                    name = 'the %s table at rsp0%+d, the set before F%+d%s' % (kind, c, d, ', the NUL' if nul else '')
                    self.table_forms[name] = (kind, c, d, nul)
                    out.append((name, (lambda kind, c, d, nul: lambda st, f: self.table_inv(st, kind, c, d, nul, f))(kind, c, d, nul)))
        return out

    def table_step(self, st, gen, name, f, f2):
        """The step of a table's invariant without a quantifier: for any byte
        x and any truth p for "x one of the set's first n bytes", the
        general state's table having x exactly when p (or the NUL) gives the
        table after the iteration having x exactly when p or x is the n-th
        byte (or the NUL): that is the invariant one byte on, p being what
        it stands for. None: not of that form (the frontier moved by other
        than 1)."""
        if S.sub(f2, f) is not const(64, 1):
            return None
        kind, c, d, nul = self.table_forms[name]
        x, p = var('sx', 8), var('sp', S.BOOL)
        n = S.add(S.sub(f, self.t), const(64, d & S.mask(64)))
        nulx = S.and_(S.boolean(nul), S.is_zero(x))
        ih = self.table_is(gen['mem'][-1], kind, c, x, S.or_(p, nulx))
        goal = self.table_is(st.regions[-1][1], kind, c, x, S.or_(p, S.eq(self.tbyte(n), x), nulx))
        plain = [c for c in st.pc if not S.has_quantifier(c)]  # the step needs no quantified fact (fewer hypotheses: sound)
        return self.solver._check(*plain, ih, S.not_(goal), bare=True, timeout=6 * AUX) == 'unsat'

    def table_is(self, mem, kind, c, x, want):
        """The table's entry for x says want: a byte table whose loop stores
        one value v, the byte v or 0 (a scan may and the entries); else
        the entry nonzero, the bit set."""
        v = self.table_v.get((kind, c))
        if v is not None:
            return S.eq(S.select(mem, S.add(const(64, c), S.zext(x, 64))), S.ite(want, v, const(8, 0)))
        return S.eq(self.table_bit(mem, kind, c, x), want)

    def table_bit(self, mem, kind, c, x):
        if kind == 'byte':
            return S.not_(S.is_zero(S.select(mem, S.add(const(64, c), S.zext(x, 64)))))
        hi = S.zext(S.lshr(x, const(8, 7)), 64)
        lo = S.zext(S.band(x, const(8, 15)), 64)
        byte = S.select(mem, S.add(S.add(const(64, c), S.mul(hi, const(64, 16))), lo))
        return S.not_(S.is_zero(S.band(S.lshr(byte, S.band(S.lshr(x, const(8, 4)), const(8, 7))), const(8, 1))))

    def table_inv(self, st, kind, c, d, nul, f):
        x = var('qx', 8)
        n = S.add(S.sub(f, self.t), const(64, d & S.mask(64)))
        want = self.in_first(x, n)
        if nul:
            want = S.or_(want, S.is_zero(x))
        mem = st.regions[-1][1]
        inv = S.forall_(x, self.table_is(mem, kind, c, x, want))
        self.tables[inv] = (mem, kind, c, d, nul, n)
        return inv


class CspanLong(SpanLong):
    spn = False


def truffle_bit(lanes, x):
    """What two 16-byte tables say of byte x: bit (x >> 4) & 7 of lane
    (x >> 7) * 16 + (x & 15), a bit of their 256 by a permutation of x's."""
    idx = S.concat(S.extract(x, 7, 7), S.extract(x, 3, 0), S.extract(x, 6, 4))
    return S.eq(S.extract(S.lshr(X.to_bv(lanes), S.zext(idx, 256)), 0, 0), const(1, 1))


def members_arg(name, members):
    """An int argument 0 or 1, as the ABI has it: the upper half of the register unknown."""
    return S.concat(var(name + '_hi', 32), S.zext(members, 32))


class TruffleScan(Span):
    """truffle_scan(from, s, low, high, members), for any two tables (a 256-bit
    set, any) and members 0 or 1: a stop is a byte the tables say yes of
    when members, no when not (nothing of NULs: a caller's tables have it
    or not), and any stop z at or past from is taken to exist. The answer
    F + r, F = from - s (any number: s is only subtracted), r the offset
    of the first stop from `from`; its loads in from's page up to r's, no
    stores. A byte's lane term (this code's per lane mask bit) is the
    tables' bit by the lane lemma. from any pointer."""

    def setup(self, a, align):
        self.base, self.s, self.S = aligned_pointer('s', a, align)
        self.regions = [(self.base, self.S)]
        self.z, self.i, self.F, self.members = var('z', 64), var('i', 64), var('F', 64), var('members', 1)
        self.tables = [var('init_zmm0_%d' % j, 8) for j in range(16)] + [var('init_zmm1_%d' % j, 8) for j in range(16)]
        s, z, top = self.s, self.z, const(64, USER_HI)
        assume = [S.ule(const(64, USER_LO), s), S.ule(s, top), self.at(z), S.ult(z, const(64, USER_HI - USER_LO)), S.ule(S.add(s, z), top)]
        return [s, S.sub(s, self.F), members_arg('members', self.members)], assume

    def member(self, b):
        return truffle_bit(self.tables, b)

    def stop(self, b):
        return S.eq(S.bit(self.member(b)), self.members)

    def lane_forms(self, b):
        return [self.member(b), S.not_(self.member(b)), self.stop(b), S.not_(self.stop(b))]

    def at(self, off):
        return self.stop(S.select(self.S, S.add(S.sub(self.s, self.base), off)))

    def owner(self, addr):
        return owner(addr, ((self.s, self.base),))

    def answer(self, st):
        return S.TRUE, S.sub(st.gpr['rax'], self.F)

    def contracts(self, img, entry):
        return {}

    def keep_unrolling(self, prev, now):
        return False

    def show(self):
        return [self.s, self.F, self.members, self.z, self.i]


class TruffleScan32:
    """truffle_scan32(from, s, set1, set2, count, members), count at most 32,
    the set the first count of the two registers' lanes, members 0 or 1: the
    tables it builds on its stack say of every byte whether it is one of
    them (members: or the NUL), and it hands truffle_scan its from, s and
    members and those tables; its answer that one's. truffle_scan by its
    contract. It reads nothing but its frame. The loop it builds them in runs
    out unrolled, a way per count."""

    def __init__(self, maxlen, induct):
        pass

    def setup(self, a, align):
        self.regions = []
        self.frm, self.sp = var('from', 64), var('s', 64)
        self.count, self.members, self.r = var('count', 64), var('members', 1), var('scan_r', 64)
        self.set = [var('init_zmm0_%d' % j, 8) for j in range(16)] + [var('init_zmm1_%d' % j, 8) for j in range(16)]
        return [self.frm, self.sp, self.count, members_arg('members', self.members)], [S.ule(self.count, const(64, SETMAX))]

    def want(self, x):
        """x one of the first count lanes, or the NUL for members."""
        return S.or_(S.and_(S.eq(self.members, const(1, 1)), S.is_zero(x)),
                     *[S.and_(S.ult(const(64, k), self.count), S.eq(self.set[k], x)) for k in range(SETMAX)])

    def contracts(self, img, entry):
        return {t: self.c_scan for t in named_targets(img, entry, ('truffle_scan',))}

    def c_scan(self, prover, st):
        x = var('cx', 8)
        tables = X.read_vec(st, 'xmm0')[:16] + X.read_vec(st, 'xmm1')[:16]
        pre = [('truffle_scan is given from', [S.not_(S.eq(st.gpr['rdi'], self.frm))]),
               ('truffle_scan is given s', [S.not_(S.eq(st.gpr['rsi'], self.sp))]),
               ('truffle_scan is given members', [S.not_(S.eq(S.extract(st.gpr['rdx'], 31, 0), S.zext(self.members, 32)))]),
               ('the stack aligned at the call', [S.not_(S.eq(S.band(st.gpr['rsp'], const(64, 15)), const(64, 8)))]),
               ('the tables say which bytes are in the set', [S.not_(S.eq(truffle_bit(tables, x), self.want(x)))])]
        if st.gpr['rsp'] is X.RSP0:  # a jump (the tables in the red zone, dead by then): truffle_scan returns to the caller, its answer this one's
            return tail_call(self, prover, st, pre)
        prover.later(prover.prove, st, pre, self.show())
        X.call_return(st, self.r, 'c_')

    def early(self, a, n):
        return nonempty(n)  # nothing but the frame

    def early_store(self, e):
        return e[3]

    def keep_unrolling(self, prev, now):
        return True

    def obligations(self, st):
        return [('the answer truffle_scan\'s', [S.not_(S.eq(st.gpr['rax'], self.r))])], self.show()

    def show(self):
        return [self.count, self.members, self.r]


# ------------------------------------------------------------ derived.c

DERIVED_ENTRIES = ('str_memchr', 'str_strnlen', 'str_strlen', 'str_strspn', 'str_strcspn', 'str_memcpy', 'str_memmove', 'str_memset', 'str_memcmp', 'malloc')


class Obj:
    """An object of a derived function's spec: a region of its own (base,
    the memory at its entry, its index among the regions), the bytes
    [0, read) from the base that may be read and [0, write) that may be
    written; stop: a string's NUL (z, any) or a bounded object's end
    (memchr's z: a NUL before n, or n), where its callees' contracts are
    instanced."""

    def __init__(self, name, base, arr, k):
        self.name, self.base, self.arr, self.k = name, base, arr, k
        self.read = self.write = const(64, 0)
        self.stop = self.n = None


def member(inset, x):
    """The byte x in the set: bit x of the set's 256 (the NUL's never)."""
    return S.eq(S.extract(S.lshr(inset, S.zext(x, 256)), 0, 0), const(1, 1))


def fits(off, n, limit):
    """[off, off + n) inside [0, limit), no wrap."""
    return S.and_(S.ule(n, limit), S.ule(off, S.sub(limit, n)))


class Over:
    """derived.c: functions made of calls of the entry points (str_memchr and
    so on, each a jump through its slot of str_table, which holds one of the
    configurations' implementations, every one proved by the slot's spec)
    and a few loads and stores of their own, no loop. A call is taken by its
    slot's contract: its arguments checked, its answer a fresh term with
    what the spec says of it instanced at the objects' stops (strlen's a NUL
    at or before any NUL of the string; memchr's at or before the bounded
    object's stop, and without one the stop at n), memcpy's bytes the
    destination's memory from then on; malloc's answer null or an object of
    the size asked, apart. A jump to one (a tail call) is checked for its
    arguments and the state it leaves; the rest is the callee's. The objects
    are regions of their own, apart, in user space; the function's own loads
    and stores inside what each allows; at the return the answer and every
    object's memory byte by byte (at a free offset) what the function's
    definition says."""
    unroll = 1

    @property
    def base(self):  # what a loop's generalization asks first: a derived function has none
        raise Failure('a loop in a function of derived.c')

    def __init__(self, maxlen, induct):
        self.maxlen, self.induct = maxlen, induct
        self.objs, self.regions, self.assume = [], [], []
        self.calls = 0
        self.alloc = self.msize = None
        self.inset = var('inset', 256)

    def setup(self, a, align):
        self.case = a
        params = self.objects(a)
        self.assume.append(S.not_(member(self.inset, const(8, 0))))
        return params, self.assume

    # ---- the objects

    def obj(self, name, base=None):
        o = Obj(name, var(name, 64) if base is None else base, S.mem_var(name + '_mem'), len(self.regions))
        self.objs.append(o)
        self.regions.append((o.base, o.arr))
        return o

    def place(self, o, extent, nullable=False):
        """o's bytes [0, extent) in user space (nullable: or o null)."""
        top = const(64, USER_HI)
        inside = S.and_(S.ule(const(64, USER_LO), o.base), S.ult(o.base, top), S.ult(extent, const(64, USER_HI - USER_LO)), S.ule(S.add(o.base, extent), top))
        self.assume.append(S.or_(S.is_zero(o.base), inside) if nullable else inside)

    def string(self, name, writable=False, base=None, nullable=False):
        """A string: a NUL at its stop (any NUL), the bytes up to it readable (and writable)."""
        o = self.obj(name, base)
        o.stop = var(name + '_z', 64)
        o.read = S.add(o.stop, const(64, 1))
        o.write = o.read if writable else const(64, 0)
        self.assume += [S.is_zero(S.select(o.arr, o.stop)), S.ult(o.stop, const(64, USER_HI - USER_LO))]
        self.place(o, o.read, nullable)
        return o

    def bounded(self, name, n, write=None):
        """memchr's object: readable up to its stop, a NUL before n (any) or n; write: the bytes that may be written."""
        o = self.obj(name)
        o.n, o.stop = n, var(name + '_z', 64)
        o.read = S.ite(S.eq(o.stop, n), n, S.add(o.stop, const(64, 1)))
        o.write = const(64, 0) if write is None else write
        self.assume.append(S.or_(S.and_(S.ult(o.stop, n), S.is_zero(S.select(o.arr, o.stop))), S.eq(o.stop, n)))
        self.place(o, o.read if write is None else S.ite(S.ult(o.read, write), write, o.read))
        return o

    def buffer(self, name, n, readable=False):
        """n bytes, written (or read)."""
        o = self.obj(name)
        o.write = n
        o.read = n if readable else const(64, 0)
        self.place(o, n)
        return o

    def cell(self, name):
        """A char *, read and written: its value the 8 bytes of its memory as a
        load has them (the term a string's base can be)."""
        o = self.obj(name)
        o.value = X.to_bv([S.select(o.arr, const(64, i)) for i in range(8)])
        o.read = o.write = const(64, 8)
        self.place(o, o.read)
        return o

    def allocation(self):
        """What malloc gives: null, or the size asked of its own (the region set aside)."""
        self.alloc = self.obj('m')
        return self.alloc

    def owner_of(self, addr):
        for o in self.objs:
            off = X.offset_from(addr, o.base)
            if off is not None:
                return o, off
        return None, None

    def write_of(self, o):
        return self.msize if o is self.alloc else o.write

    def early(self, a, n):
        """The function's own load past what its object lets be read."""
        o, off = self.owner_of(a)
        if o is None:
            raise Failure('a load of no object: %r' % a)
        return S.not_(fits(off, const(64, n), o.read))

    def early_store(self, e):
        a, nb, rip, c, lanes = e
        nb = nb if isinstance(nb, S.T) else const(64, nb)
        o, off = self.owner_of(a)
        if o is None:
            raise Failure('a store to no object at %#x' % rip)
        w = self.write_of(o)
        return S.and_(c, S.not_(fits(off, nb, w if w is not None else const(64, 0))))

    # ---- the callees by their contracts

    def contracts(self, img, entry):
        return {t: getattr(self, 'c_' + n.replace('str_', '', 1)) for t, n in named_targets(img, entry, DERIVED_ENTRIES).items()}

    def log(self, st, *entry):
        st.log = getattr(st, 'log', ()) + (entry,)

    def calls_of(self, st, name):
        return [e for e in getattr(st, 'log', ()) if e[0] == name]

    def one(self, st, name):
        got = self.calls_of(st, name)
        if len(got) != 1:
            raise Failure('%d calls of %s on a path' % (len(got), name))
        return got[0]

    def returns(self, st, rax):
        """The state after a call: rax the answer, the caller-saved registers unknown."""
        self.calls += 1
        X.call_return(st, rax, 'c%d_' % self.calls)

    def tail(self, st):
        return st.gpr['rsp'] is X.RSP0

    def untouched(self, st, o, what):
        if st.regions[o.k][1] is not o.arr:
            raise Failure('%s of %s after stores to it' % (what, o.name))

    def c_memchr(self, prover, st):
        """memchr(p, 0, n) of a bounded object from its start, a way each: p +
        r, r < n, a NUL there, none before (r at or before the stop); null, no
        NUL in the n bytes (the stop at n)."""
        p, c, n = st.gpr['rdi'], S.extract(st.gpr['rsi'], 7, 0), st.gpr['rdx']
        o, off = self.owner_of(p)
        if o is None or o.n is None or self.tail(st):
            raise Failure('memchr of no bounded object, or as a tail call: %r' % p)
        self.untouched(st, o, 'memchr')
        prover.prove(st, [('memchr is given %s, a NUL and its n' % o.name, [S.or_(S.not_(S.is_zero(off)), S.not_(S.is_zero(c)), S.not_(S.eq(n, o.n)))])], self.show())
        r = var('at%d' % (self.calls + 1), 64)
        none = st.copy()  # the other way, walked later: no NUL in the n bytes, null
        none.pc.append(S.eq(o.stop, n))
        self.log(none, 'memchr', o, S.FALSE, r)
        st.pc += [S.ult(r, n), S.is_zero(S.select(o.arr, r)), S.ule(r, o.stop)]
        self.log(st, 'memchr', o, S.TRUE, r)
        self.returns(st, S.add(p, r))
        self.returns(none, const(64, 0))
        prover.works[-1].append(none)

    def c_strnlen(self, prover, st):
        """strnlen(p, n) of a bounded object from its start, a way each: r < n,
        a NUL there, none before (r at or before the stop); n, no NUL in the n
        bytes (the stop at n)."""
        p, n = st.gpr['rdi'], st.gpr['rsi']
        o, off = self.owner_of(p)
        if o is None or o.n is None or self.tail(st):
            raise Failure('strnlen of no bounded object, or as a tail call: %r' % p)
        self.untouched(st, o, 'strnlen')
        prover.prove(st, [('strnlen is given %s and its n' % o.name, [S.or_(S.not_(S.is_zero(off)), S.not_(S.eq(n, o.n)))])], self.show())
        r = var('at%d' % (self.calls + 1), 64)
        none = st.copy()  # the other way, walked later: no NUL in the n bytes, n
        none.pc.append(S.eq(o.stop, n))
        self.log(none, 'strnlen', o, S.FALSE, r)
        st.pc += [S.ult(r, n), S.is_zero(S.select(o.arr, r)), S.ule(r, o.stop)]
        self.log(st, 'strnlen', o, S.TRUE, r)
        self.returns(st, r)
        self.returns(none, n)
        prover.works[-1].append(none)

    def c_strlen(self, prover, st):
        """strlen(p), p at or before its string's NUL: L, a NUL at p + L, none
        before it (at or before that NUL)."""
        p = st.gpr['rdi']
        o, off = self.owner_of(p)
        if o is None or o.stop is None or o.n is not None or self.tail(st):
            raise Failure('strlen of no string, or as a tail call: %r' % p)
        self.untouched(st, o, 'strlen')
        prover.prove(st, [('strlen is given %s at or before its NUL' % o.name, [S.not_(S.ule(off, o.stop))])], self.show())
        L = var('len%d' % (self.calls + 1), 64)
        st.pc += [S.is_zero(S.select(o.arr, S.add(off, L))), S.ule(L, S.sub(o.stop, off))]
        self.log(st, 'strlen', o, off, L)
        self.returns(st, L)

    def span_call(self, prover, st, name, stops):
        p, setp = st.gpr['rdi'], st.gpr['rsi']
        o, off = self.owner_of(p)
        if o is None or o.stop is None or o.n is not None or self.tail(st):
            raise Failure('%s of no string, or as a tail call: %r' % (name, p))
        self.untouched(st, o, name)
        prover.prove(st, [('%s is given %s at or before its NUL' % (name, o.name), [S.not_(S.ule(off, o.stop))]),
                          ('%s is given the set' % name, [S.not_(S.eq(setp, self.set.base))])], self.show())
        r = var('span%d' % (self.calls + 1), 64)
        st.pc += [S.ule(r, S.sub(o.stop, off)), stops(S.select(o.arr, S.add(off, r)))]
        self.log(st, name, o, off, r)
        self.returns(st, r)

    def c_strspn(self, prover, st):
        """strspn(p, set): r, p's bytes before p + r in the set (no NUL among
        them: r at or before the NUL), the one there not."""
        self.span_call(prover, st, 'strspn', lambda b: S.not_(member(self.inset, b)))

    def c_strcspn(self, prover, st):
        """strcspn(p, set): r, p's bytes before p + r neither in the set nor
        the NUL, the one there either."""
        self.span_call(prover, st, 'strcspn', lambda b: S.or_(S.is_zero(b), member(self.inset, b)))

    def c_memcpy(self, prover, st):
        """memcpy(d, s, n), d's object and s's apart: d's n bytes from then
        on s's (as s's memory has them at the call); d."""
        d, s, n = st.gpr['rdi'], st.gpr['rsi'], st.gpr['rdx']
        od, xd = self.owner_of(d)
        os_, xs = self.owner_of(s)
        if od is None or os_ is None or od is os_:
            raise Failure('memcpy between %r and %r' % (d, s))
        w = self.write_of(od)
        out = [('memcpy writes inside what %s allows' % od.name, [S.not_(fits(xd, n, w if w is not None else const(64, 0)))]),
               ('memcpy reads inside what %s allows' % os_.name, [S.not_(fits(xs, n, os_.read))])]
        if self.tail(st):
            self.log(st, 'memcpy', od, xd, os_, xs, n)
            return tail_call(self, prover, st, out + self.tail_args(st, 'memcpy'))
        prover.prove(st, out, self.show())
        x = var('xm%d' % (self.calls + 1), 64)
        old, src = st.regions[od.k][1], st.regions[os_.k][1]
        st.regions[od.k] = (od.base, S.lam(x, S.ite(X.within(x, xd, n), S.select(src, S.add(xs, S.sub(x, xd))), S.select(old, x))))
        self.log(st, 'memcpy', od, xd, os_, xs, n)
        self.returns(st, d)

    def tail_only(self, prover, st, name):
        if not self.tail(st):
            raise Failure('%s called, not jumped to' % name)
        return tail_call(self, prover, st, self.tail_args(st, name))

    def c_memmove(self, prover, st):
        return self.tail_only(prover, st, 'memmove')

    def c_memset(self, prover, st):
        return self.tail_only(prover, st, 'memset')

    def c_memcmp(self, prover, st):
        return self.tail_only(prover, st, 'memcmp')

    def tail_args(self, st, name):
        raise Failure('a jump to %s' % name)

    def c_malloc(self, prover, st):
        """malloc(n): null, or an object of n bytes of its own."""
        if self.alloc is None or self.tail(st) or self.calls_of(st, 'malloc'):
            raise Failure('malloc where the spec has none')
        n, m = st.gpr['rdi'], self.alloc.base
        self.msize = n
        st.pc.append(S.or_(S.is_zero(m), S.and_(S.ule(const(64, USER_LO), m), S.ult(n, const(64, USER_HI - USER_LO)), S.ule(S.add(m, n), const(64, USER_HI)))))
        self.log(st, 'malloc', n)
        self.returns(st, m)

    # ---- at the return

    def final(self, st, o):
        return st.regions[o.k][1]

    def memory_is(self, st, o, want, what, upto=None):
        """o's bytes at the return want(x) at every offset x (below upto)."""
        x = var('x_' + o.name, 64)
        bad = S.not_(S.eq(S.select(self.final(st, o), x), want(x)))
        return [(what, [S.and_(S.ult(x, upto), bad) if upto is not None else bad])]

    def unchanged(self, st, *changed):
        out = []
        for o in self.objs:
            if o not in changed and o is not self.alloc and self.final(st, o) is not o.arr:
                out += self.memory_is(st, o, lambda x, o=o: S.select(o.arr, x), '%s unchanged' % o.name)
        return out

    def answer(self, st, want, what):
        return [(what, [S.not_(S.eq(st.gpr['rax'], want))])]

    def show(self):
        return [o.base for o in self.objs] + [o.stop for o in self.objs if o.stop is not None]


class OverTail(Over):
    """bcmp, bcopy, bzero, explicit_bzero: a jump to an entry point with the
    arguments the definition gives it (bcmp(a, b, n) memcmp's, its sign
    unasked; bcopy(s, d, n) memmove(d, s, n); bzero(d, n) memset(d, 0, n))."""
    callee = None
    arity = 3

    def objects(self, a):
        self.p = [var('a%d' % i, 64) for i in range(self.arity)]
        return self.p

    def jump_ok(self, st):
        return S.and_(*[S.eq(st.gpr[r], w) for r, w in zip(('rdi', 'rsi', 'rdx'), self.p)])

    def tail_args(self, st, name):
        if name != self.callee:
            raise Failure('a jump to %s, not %s' % (name, self.callee))
        return [('the jump to %s with the arguments of the definition' % name, [S.not_(self.jump_ok(st))])]

    def obligations(self, st):
        raise Failure('a return, not a jump to %s' % self.callee)

    def show(self):
        return self.p


class OverBcmp(OverTail):
    callee = 'memcmp'


class OverBcopy(OverTail):
    callee = 'memmove'

    def jump_ok(self, st):
        return S.and_(S.eq(st.gpr['rdi'], self.p[1]), S.eq(st.gpr['rsi'], self.p[0]), S.eq(st.gpr['rdx'], self.p[2]))


class OverBzero(OverTail):
    callee = 'memset'
    arity = 2

    def jump_ok(self, st):
        return S.and_(S.eq(st.gpr['rdi'], self.p[0]), S.is_zero(S.extract(st.gpr['rsi'], 7, 0)), S.eq(st.gpr['rdx'], self.p[1]))


class OverMempcpy(Over):
    """mempcpy(d, s, n): d + n; d's n bytes s's, nothing else written."""

    def objects(self, a):
        self.n = var('n', 64)
        self.d, self.s = self.buffer('d', self.n), self.buffer('s', self.n, readable=True)
        return [self.d.base, self.s.base, self.n]

    def obligations(self, st):
        self.one(st, 'memcpy')
        want = lambda x: S.ite(S.ult(x, self.n), S.select(self.s.arr, x), S.select(self.d.arr, x))  # noqa: E731
        return (self.answer(st, S.add(self.d.base, self.n), 'the answer d + n') + self.memory_is(st, self.d, want, 'd: s\'s n bytes, nothing else')
                + self.unchanged(st, self.d)), self.show()


class OverStrncat(Over):
    """strncat(d, s, n): d; s's bytes before its first NUL, n at most (k:
    strnlen's answer from s), copied to d's first NUL (L: strlen's
    answer) and a NUL after them, nothing else written. d a string with room
    past any NUL of its for any stop of s and a NUL (its capacity cap), s
    bounded by n."""

    def objects(self, a):
        self.n, self.cap = var('n', 64), var('cap', 64)
        self.d, self.s = self.string('d'), self.bounded('s', self.n)
        self.d.write = self.cap
        self.assume.append(S.ule(S.add(S.add(self.d.stop, self.s.stop), const(64, 1)), self.cap))
        self.place(self.d, self.cap)
        return [self.d.base, self.s.base, self.n]

    def obligations(self, st):
        L = self.one(st, 'strlen')[3]
        _, _, f, r = self.one(st, 'strnlen')
        k = S.ite(f, r, self.n)
        want = lambda x: S.ite(X.within(x, L, k), S.select(self.s.arr, S.sub(x, L)),  # noqa: E731
                               S.ite(S.eq(x, S.add(L, k)), const(8, 0), S.select(self.d.arr, x)))
        return (self.answer(st, self.d.base, 'the answer d') + self.memory_is(st, self.d, want, 'd: s\'s k bytes at its NUL and a NUL, nothing else')
                + self.unchanged(st, self.d)), self.show()


class OverStrlcpy(Over):
    """strlcpy(d, s, n): s's length (L: strlen's answer); with n > 0 d's
    first k = min(L, n - 1) bytes s's and a NUL after them, nothing else
    written. d n bytes."""

    def objects(self, a):
        self.n = var('n', 64)
        self.d, self.s = self.buffer('d', self.n), self.string('s')
        return [self.d.base, self.s.base, self.n]

    def obligations(self, st):
        L = self.one(st, 'strlen')[3]
        last = S.sub(self.n, const(64, 1))
        k = S.ite(S.ult(L, last), L, last)
        want = lambda x: S.ite(S.is_zero(self.n), S.select(self.d.arr, x), S.ite(S.ult(x, k), S.select(self.s.arr, x),  # noqa: E731
                               S.ite(S.eq(x, k), const(8, 0), S.select(self.d.arr, x))))
        return (self.answer(st, L, 'the answer s\'s length') + self.memory_is(st, self.d, want, 'd: s\'s first k bytes and a NUL, nothing else')
                + self.unchanged(st, self.d)), self.show()


class OverStrlcat(Over):
    """strlcat(d, s, n): without a NUL in d's n bytes (strnlen's answer n)
    n + s's length, nothing written; else at d's first NUL r (strnlen's) s's
    first k = min(L, n - r - 1) bytes and a NUL, nothing else written, and
    r + L (L: strlen's answer). d bounded by n, its n bytes writable."""

    def objects(self, a):
        self.n = var('n', 64)
        self.d, self.s = self.bounded('d', self.n, write=self.n), self.string('s')
        return [self.d.base, self.s.base, self.n]

    def obligations(self, st):
        _, _, f, r = self.one(st, 'strnlen')
        L = self.one(st, 'strlen')[3]
        room = S.sub(S.sub(self.n, r), const(64, 1))
        k = S.ite(S.ult(L, room), L, room)
        want = lambda x: S.ite(f, S.ite(X.within(x, r, k), S.select(self.s.arr, S.sub(x, r)),  # noqa: E731
                                        S.ite(S.eq(x, S.add(r, k)), const(8, 0), S.select(self.d.arr, x))), S.select(self.d.arr, x))
        return (self.answer(st, S.add(S.ite(f, r, self.n), L), 'the answer d\'s length (n at most) and s\'s')
                + self.memory_is(st, self.d, want, 'd: s\'s first k bytes and a NUL at its NUL, nothing else') + self.unchanged(st, self.d)), self.show()


class OverStrpbrk(Over):
    """strpbrk(s, set): s + r at the first byte of s in the set (r: strcspn's
    answer), null where that is s's NUL; nothing written."""

    def objects(self, a):
        self.s, self.set = self.string('s'), self.string('set')
        return [self.s.base, self.set.base]

    def obligations(self, st):
        r = self.one(st, 'strcspn')[3]
        at = S.select(self.s.arr, r)
        return self.answer(st, S.ite(S.is_zero(at), const(64, 0), S.add(self.s.base, r)), 'the answer the first byte in the set, or null') + \
            self.unchanged(st), self.show()


class OverStrdup(Over):
    """strdup(s): null when malloc's answer is; else a jump to memcpy(m, s,
    L + 1), m malloc's answer for L + 1 bytes (L: strlen's), which copies s
    and its NUL there and gives m."""

    def objects(self, a):
        self.s = self.string('s')
        self.allocation()
        return [self.s.base]

    def tail_args(self, st, name):
        if name != 'memcpy':
            raise Failure('a jump to %s' % name)
        n1 = S.add(self.one(st, 'strlen')[3], const(64, 1))
        size = self.one(st, 'malloc')[1]
        ok = S.and_(S.eq(st.gpr['rdi'], self.alloc.base), S.eq(st.gpr['rsi'], self.s.base), S.eq(st.gpr['rdx'], n1))
        return [('malloc asked for s and its NUL', [S.not_(S.eq(size, n1))]), ('the jump to memcpy(m, s, L + 1)', [S.not_(ok)]),
                ('m not null there', [S.is_zero(self.alloc.base)])]

    def obligations(self, st):
        self.one(st, 'malloc')
        return (self.answer(st, const(64, 0), 'the answer null') + [('malloc\'s answer null there', [S.not_(S.is_zero(self.alloc.base))])]
                + self.unchanged(st)), self.show()


class OverStrndup(Over):
    """strndup(s, n): l strnlen's answer from s; null when malloc's
    answer (for l + 1 bytes) is; else m with s's l bytes and a NUL."""

    def objects(self, a):
        self.n = var('n', 64)
        self.s = self.bounded('s', self.n)
        self.allocation()
        return [self.s.base, self.n]

    def obligations(self, st):
        _, _, f, r = self.one(st, 'strnlen')
        l1 = S.add(S.ite(f, r, self.n), const(64, 1))
        out = [('malloc asked for l + 1 bytes', [S.not_(S.eq(self.one(st, 'malloc')[1], l1))])]
        if not self.calls_of(st, 'memcpy'):
            return (out + self.answer(st, const(64, 0), 'the answer null') + [('malloc\'s answer null there', [S.not_(S.is_zero(self.alloc.base))])]
                    + self.unchanged(st)), self.show()
        l = S.sub(l1, const(64, 1))
        want = lambda x: S.ite(S.ult(x, l), S.select(self.s.arr, x), const(8, 0))  # noqa: E731
        return (out + self.answer(st, self.alloc.base, 'the answer m') + self.memory_is(st, self.alloc, want, 'm: s\'s l bytes and a NUL', upto=l1)
                + self.unchanged(st)), self.show()


class TokenSpec(Over):
    """strsep, strtok_r and strtok: past a token's start t in string o,
    strcspn's answer r2 from there: at t + r2 a NUL and the char * cell
    t + r2 + 1 when the byte there is in the set; else nothing written and
    the cell null."""

    def token_end(self, st, o, t, cellv):
        _, _, off, r2 = self.one(st, 'strcspn')
        e = S.add(off, r2)
        cut = S.not_(S.is_zero(S.select(o.arr, e)))
        want = lambda x: S.ite(S.and_(cut, S.eq(x, e)), const(8, 0), S.select(o.arr, x))  # noqa: E731
        return ([('strcspn from the token\'s start', [S.not_(S.eq(off, t))])]
                + self.memory_is(st, o, want, 'the token\'s end a NUL when a byte of the set is there, nothing else')
                + [('the cell past the token\'s end, else null', [S.not_(S.eq(cellv, S.ite(cut, S.add(S.add(o.base, e), const(64, 1)), const(64, 0))))])])

    def cell_now(self, st, c):
        return X.to_bv([S.select(self.final(st, c), const(64, i)) for i in range(8)])


class OverStrsep(TokenSpec):
    """strsep(sp, sep): *sp (v); null and nothing written when v is null;
    else the token from v ends where TokenSpec says, the cell *sp."""

    def objects(self, a):
        self.sp = self.cell('sp')
        self.v = self.sp.value
        self.s, self.set = self.string('s', writable=True, base=self.v, nullable=True), self.string('set')
        return [self.sp.base, self.set.base]

    def obligations(self, st):
        out = self.answer(st, self.v, 'the answer *sp')
        if not self.calls_of(st, 'strcspn'):
            return out + [('*sp null there', [S.not_(S.is_zero(self.v))])] + self.unchanged(st), self.show()
        return out + self.token_end(st, self.s, const(64, 0), self.cell_now(st, self.sp)) + self.unchanged(st, self.s, self.sp), self.show()


class OverStrtokR(TokenSpec):
    """strtok_r(s, sep, last) in two cases (the class): 0, s a string; 1, s
    null and the string *last's (v; null: none). With none: null, nothing
    written. Past strspn's answer r1 from the string's start (its bytes the
    set's) the NUL: null and the cell null, nothing else written; else the
    token there, its end and the cell as TokenSpec says."""

    def objects(self, a):
        self.last = self.state_cell()
        self.v = self.state_value()
        if a == 0:
            self.s = self.string('s', writable=True)
            first = self.s.base
        else:
            self.s = self.string('s', writable=True, base=self.v, nullable=True)
            first = const(64, 0)
        self.set = self.string('set')
        return [first, self.set.base] + ([self.last.base] if self.last is not None else [])

    def state_cell(self):
        return self.cell('last')

    def state_value(self):
        return self.last.value

    def state_now(self, st):
        return self.cell_now(st, self.last)

    def obligations(self, st):
        cellv = self.state_now(st)
        if not self.calls_of(st, 'strspn'):
            if self.case == 0:
                raise Failure('a string and no strspn')
            return (self.answer(st, const(64, 0), 'the answer null') + [('*last null there', [S.not_(S.is_zero(self.v))]),
                                                                       ('the cell unchanged', [S.not_(S.eq(cellv, self.v))])]
                    + self.unchanged(st)), self.show()
        _, _, off1, r1 = self.one(st, 'strspn')
        out = [('strspn from the string\'s start', [S.not_(S.is_zero(off1))])]
        if not self.calls_of(st, 'strcspn'):
            return (out + self.answer(st, const(64, 0), 'the answer null past the set\'s bytes at the NUL')
                    + [('the NUL there', [S.not_(S.is_zero(S.select(self.s.arr, r1)))]), ('the cell null', [S.not_(S.is_zero(cellv))])]
                    + self.unchanged(st, self.last)), self.show()
        return (out + self.answer(st, S.add(self.s.base, r1), 'the answer the token past the set\'s bytes')
                + [('not at the NUL', [S.is_zero(S.select(self.s.arr, r1))])] + self.token_end(st, self.s, r1, cellv)
                + self.unchanged(st, self.s, self.last)), self.show()


class OverStrtok(OverStrtokR):
    """strtok(s, sep): strtok_r's with strtok's own cell, the static
    strtok_last of its object (its bytes the memory's at that address)."""

    def state_cell(self):
        self.addr = self.image.symbols['strtok_last']
        self.cellobj = Obj('strtok_last', const(64, self.addr), None, None)
        self.cellobj.read = self.cellobj.write = const(64, 8)
        return None

    def state_value(self):
        return X.to_bv([S.select(S.mem_var('M'), self.at(i)) for i in range(8)])

    def at(self, i):
        return const(64, self.addr + i)

    def owner_of(self, addr):
        o, off = Over.owner_of(self, addr)
        if o is None and S.is_const(addr) and self.addr <= addr.val < self.addr + 8:
            return self.cellobj, const(64, addr.val - self.addr)
        return o, off

    def state_now(self, st):
        return X.to_bv([S.select(st.mem, self.at(i)) for i in range(8)])


SPECS = {'strlen': Strlen, 'strlen_tail': StrlenTail, 'memchr': Memchr, 'memchr_large': MemchrLarge, 'strnlen': Strnlen, 'strnlen_large': StrnlenLarge,
         'memset': Fill, 'fill_large': FillLarge, 'memcpy': Copy,
         'memmove': Move, 'copy_large': CopyLarge, 'memcmp': Compare, 'memcmp_large': MemcmpLarge, 'strcmp': StrCompare, 'strcmp_tail': StrCompare, 'strcpy': StrCopy, 'strcpy_long': StrCopyLong, 'strcat': StrCat, 'strspn': Span, 'strcspn': Cspan, 'strspn_long': SpanLong, 'strcspn_long': CspanLong, 'truffle_scan': TruffleScan, 'truffle_scan32': TruffleScan32,
         'bcmp': OverBcmp, 'bcopy': OverBcopy, 'bzero': OverBzero, 'explicit_bzero': OverBzero, 'mempcpy': OverMempcpy, 'strncat': OverStrncat, 'strlcpy': OverStrlcpy, 'strlcat': OverStrlcat, 'strpbrk': OverStrpbrk, 'strsep': OverStrsep, 'strtok_r': OverStrtokR,
         'strtok': OverStrtok, 'strdup': OverStrdup, 'strndup': OverStrndup}


def contract_stops(spec, img, entry):
    """The callees this spec takes by their contracts, in the code from entry."""
    x = SPECS[spec](0, True)
    return set(x.contracts(img, entry)) if hasattr(x, 'contracts') else set()


def contract_callees(spec, img, entry):
    """The callees whose proofs this function's rests on: strcat's every call
    (nothing of it walked as if it were strcat's); else those it takes by contract."""
    return contract_stops(spec, img, entry) | (contract_targets(img, entry) if spec == 'strcat' else set())


def moved_by(v, d, step, k):
    """v moved by d an iteration, after k / step of them (k a multiple of the
    step, so the division is exact)."""
    if d == step:
        return S.add(v, k)
    if d == -step:
        return S.sub(v, k)
    if d % step == 0:
        return S.add(v, S.mul(k, const(64, d // step)))
    return S.add(v, S.mul(S.lshr(k, const(64, step.bit_length() - 1)), const(64, d)))


def result(args, a, what, paths, obligations, queries, seconds, detail):
    """One line per alignment class for dev/proof/run.py."""
    print('RESULT ' + json.dumps({'func': args.func, 'spec': args.spec, 'max': 'induct' if args.induct else args.max, 'classes': args.classes,
                                  'align': a, 'prefix': getattr(args, 'prefix', '') or '', 'setmax': getattr(args, 'setmax', SETMAX), 'result': what,
                                  'paths': paths, 'obligations': obligations, 'queries': queries, 'seconds': round(seconds, 1),
                                  'detail': detail}), flush=True)


class Current:
    """The solver of the thread that asks (a spec's span() and pinned())."""

    def __init__(self, prover):
        self.prover = prover

    def __getattr__(self, name):
        return getattr(self.prover.solver, name)


def check_abi(st, solver):
    """The callee-saved registers and rsp as at the entry: the same term, or
    (read back past stores at computed places: a frame aligned down) a
    query that they can not differ."""
    for r in X.CALLEE_SAVED:
        v, v0 = st.gpr[r], var('init_' + r, 64)
        if v is not v0 and solver.check(*st.pc, S.not_(S.eq(v, v0))) != 'unsat':
            raise Failure('%s not preserved' % r)
    if st.gpr['rsp'] is not S.add(X.RSP0, const(64, 8)):
        raise Failure('rsp at the return: %r' % st.gpr['rsp'])


def tail_call(spec, prover, st, out):
    """A jump to a callee in place of the return: its precondition (out) and
    the state it leaves the caller's checked, the rest of the path its
    contract's."""
    prover.later(prover.prove, st, out, spec.show())
    check_tail(st, prover.solver)
    return True


def check_tail(st, solver):
    """A jump to a callee as the return: the callee-saved registers as at the
    entry (the callee keeps them, by its contract), rsp at the return address."""
    for r in X.CALLEE_SAVED:
        v, v0 = st.gpr[r], var('init_' + r, 64)
        if v is not v0 and solver.check(*st.pc, S.not_(S.eq(v, v0))) != 'unsat':
            raise Failure('%s not preserved at the tail call' % r)
    if st.gpr['rsp'] is not X.RSP0:
        raise Failure('a tail call with rsp at %r' % st.gpr['rsp'])
    ret = X.pop64(st.copy())
    if ret is not const(64, X.RET_MARK):
        # stores at computed places below rsp since (a frame in the red zone, aligned down): the slot's
        # bytes read past them are the return address, and the path says the slot is as it was
        if X.slot_guess(st, st.gpr['rsp']) is not const(64, X.RET_MARK) or \
                solver.check(*st.pc, S.not_(S.eq(ret, const(64, X.RET_MARK)))) != 'unsat':
            raise Failure('a tail call with the return address not in its place')


class Pool:
    """Checking threads, each with its own solver processes (the work is
    theirs: the prover's Python is a percent of a proof's time): the checks
    that do not steer the walk go to them, the walk goes on; a queue of a
    few per thread holds it back when they fall behind. The first failure
    stops the proof, reported as the walk would have."""

    def __init__(self, prover, n):
        self.prover, self.jobs, self.error, self.solvers = prover, queue.Queue(maxsize=2 * n), None, []
        for _ in range(n):  # started here: a fork from a thread with preexec_fn may hang
            solver = S.Solver()
            for c in prover.main_solver.assumed:
                solver.assume(c)
            self.solvers.append(solver)
        self.threads = [threading.Thread(target=self.work, args=(x,), daemon=True) for x in self.solvers]
        for t in self.threads:
            t.start()

    def work(self, solver):
        self.prover.tls.solver = solver
        while True:
            job = self.jobs.get()
            if job is None:
                break
            check, st, args, kw = job
            try:
                if self.error is None:
                    check(st, *args, **kw)
            except Counterexample as e:
                self.error = self.error or ('fail', str(e))
            except Failure as e:  # unknown
                self.error = self.error or ('error', str(e))
            except Exception as e:  # a bug: the proof stops too
                self.error = self.error or ('error', '%s in a checking thread: %s' % (type(e).__name__, e))
            finally:
                self.jobs.task_done()

    def submit(self, check, st, args, kw):
        self.raise_error()
        self.jobs.put((check, st, args, kw))

    def raise_error(self):
        if self.error is None:
            return
        kind, detail = self.error
        if kind == 'fail':
            self.prover.report(detail)
        raise Failure(detail)

    def finish(self):
        self.jobs.join()
        for _ in self.threads:
            self.jobs.put(None)
        for t in self.threads:
            t.join()
        self.prover.queries = sum(x.queries for x in self.solvers)
        for x in self.solvers:
            x.close()
        self.raise_error()


class Prover:
    def __init__(self, args, mach, spec, solver, a):
        self.args, self.mach, self.spec, self.main_solver, self.a = args, mach, spec, solver, a
        self.abstract = getattr(spec, 'abstraction', None) if os.environ.get('PROVE_ABSTRACT') != '0' else None  # the spec's terms an obligation is first asked with uninterpreted
        self.tls, self.pool, self.works, self.status_thread = threading.local(), None, [], None
        self.paths = self.proved = self.loops = 0
        self.dropped = {}  # loop head -> the invariant candidates a step did not keep
        self.lost = {}     # this run's: dropped at its end
        self.widen = {}    # loop head -> registers, vectors and masks some way round changed that the general state kept
        self.best_way = {}  # an obligation written several ways -> the one that proved it last
        self.best_stop = {}  # (an access, its ways) -> the way no_stop_below proved it by last
        self.covered = {}  # id -> a first arrival at a loop's head whose general state's subtree is explored
        self.few = {}  # (rip, a shift count's term) -> its few values on the path, or None
        self.loops_ahead = {}  # rip -> whether a loop's head can be reached from it
        self.prefix = getattr(args, 'prefix', '') or ''  # the first two-way branches' ways this run takes (a split of one proof)
        self.backs = set()
        self.t0 = time.time()

    @property
    def solver(self):
        """The main thread's solver, or a checking thread's own."""
        return getattr(self.tls, 'solver', None) or self.main_solver

    def fail(self, what, st, show):
        if st.gen is not None:  # in a loop's general state: its distance and frontier too
            show = list(show) + [st.gen['k'], st.gen['front']]
        detail = '%s; branches at %s; %s' % (what, ' '.join('%#x' % x for x in st.trace), ' '.join(self.solver.values(show).split()))
        if threading.current_thread() is not threading.main_thread():
            raise Counterexample(detail)
        self.report(detail)

    def report(self, detail):
        print('FAIL %s, s %% %d == %d: %s' % (self.args.func, self.align, self.a, detail))
        result(self.args, self.a, 'fail', self.paths, self.proved, self.solver.queries, time.time() - self.t0, detail)
        sys.exit(1)

    SHIFTS = ('shrx', 'shlx', 'sarx', 'shr', 'shl', 'sar')

    def shift_cases(self, st, most=4, span_most=32):
        """A variable shift whose count register takes at most `most` values on
        the path (or an interval of up to span_most where no loop lies ahead): the path once per value,
        that value its condition and the register's: a case split, every way
        the count can be. strlen_tail evex512's first block, its NUL mask
        shifted by a & 63 (0, 16, 32 or 48: a 16-aligned): each bit then the
        byte at a constant offset from a, not one of 64 a symbolic shift may
        bring (a query of more than 200 s each way)."""
        j = self.mach.image.decode(st.rip)
        mn = j['mnemonic']
        if mn not in self.SHIFTS:
            return None
        cnt = j['ops'][2] if mn.endswith('x') else j['ops'][1] if len(j['ops']) > 1 else None
        if cnt is None or cnt['kind'] != 'Register' or cnt['reg'] not in X.GPR:
            return None
        full = X.GPR[cnt['reg']][0]
        t = st.gpr[full]
        if S.is_const(t):
            return None
        key = (st.rip, t.id)
        if key not in self.few:
            # the values one by one when there are few; else an interval the path bounds the count to
            # (memchr_large's last block, its mask shifted by 32 - (end - a): 1 to 31), every value in it
            # a way (one the path rules out is a way with no input: its checks hold)
            # (only where no loop lies ahead: before one, a way per value is that loop's subtree per value:
            # truffle_scan's first block, shifted by s & 31 past strspn_long's table loop, made 32 of them)
            ahead = self.loops_ahead.get(st.rip)
            if ahead is None:
                ahead = self.loops_ahead[st.rip] = bool(X.backward_targets(self.mach.image, st.rip))
            self.few[key] = self.few_values(st, t, most) or (None if ahead else self.count_range(st, t, span_most))
        vals = self.few[key]
        if not vals or len(vals) < 2:
            return None
        out = []
        for v in vals:
            s = st.copy()
            s.pc.append(S.eq(t, const(64, v)))
            s.gpr[full] = const(64, v)
            s.trace.append(st.rip)
            out.append(s)
        if self.args.verbose:
            print('    the count %s at %#x: %d cases (%s)' % (full, st.rip, len(vals), ', '.join(str(v) for v in vals)), flush=True)
        return out

    def count_range(self, st, t, most):
        """[lo, hi] when the path bounds t to at most `most` values there (hi, lo
        proved; the first values below the path's lowest dropped), else None."""
        for hi in (most - 1, 2 * most - 1):
            if self.solver.check(*st.pc, S.ult(const(t.w, hi), t), timeout=AUX) == 'unsat':
                break
        else:
            return None
        lo = max(0, hi - most + 1)
        if lo and self.solver.check(*st.pc, S.ult(t, const(t.w, lo)), timeout=AUX) != 'unsat':
            return None
        for _ in range(2):
            if lo < hi and self.solver.check(*st.pc, S.eq(t, const(t.w, lo)), timeout=AUX) == 'unsat':
                lo += 1
        return list(range(lo, hi + 1))

    def few_values(self, st, t, most):
        """The values of t on the path, when there are at most `most`; else None."""
        vals = []
        for _ in range(most + 1):
            r = self.solver.check(*st.pc, *[S.not_(S.eq(t, const(t.w, v))) for v in vals], keep=True, timeout=AUX)
            if r == 'unsat':
                return vals
            if r != 'sat' or len(vals) == most:
                if r == 'sat':
                    self.solver.values([t])  # the model's: taken, so the session goes on
                return None
            text = self.solver.values([t])
            m = re.search(r'#b([01]+)|#x([0-9a-fA-F]+)|\(_ bv(\d+) \d+\)', text)
            if m is None:
                return None
            vals.append(int(m.group(1), 2) if m.group(1) else int(m.group(2), 16) if m.group(2) else int(m.group(3)))
        return None

    def abstract_of(self, st):
        """The terms a check's first try replaces: the spec's predicates by
        uninterpreted ones, and a lane's test of its byte that the lane lemma
        says is the predicate (its instance, true for every byte) by the same
        uninterpreted one: the 16 x 16 compares of a pcmpistri block, say,
        never bit-blasted where the facts settle them."""
        extra = getattr(st, 'lane_abstract', None)
        if not extra or self.abstract is None:
            return self.abstract
        return {**self.abstract, **extra}

    def later(self, check, st, *args, **kw):
        """A check that does not steer the walk (a return's obligations, an
        access's bounds, a loop's base and step cases): with checking
        threads, on a copy of the state in one of them, the walk going on."""
        facts = self.spec.lane_facts(self, st) if hasattr(self.spec, 'lane_facts') else []
        if facts:
            st = st.copy()
            st.pc = st.pc + facts
            st.lane_abstract = self.spec.lanes.get('last_subst')
        if self.pool is None:
            return check(st, *args, **kw)
        self.pool.submit(check, st if facts else st.copy(), args, kw)

    def plain(self, st):
        """The assumptions and the path's conditions that read no memory."""
        return [c for c in self.solver.assumed + st.pc if not S.has_array(c)]

    def no_stop_below(self, st, end):
        """z (any stop) not in [lo, end), lo the frontier's offset: proved by
        the span's ways, as a condition, None when not proved. For a load's
        bound or a store's (strcpy's last, its end at the NUL a mask found).
        Every way for 20 s first, then the one that did it last time at this
        access, then the path's loads' with the full timeout (on a loaded
        host it took more than 120 s where the bound without it took 1800),
        the others for 120 (strcpy_long's last store: the loads' way more
        than 3000 s, the other one less than 120)."""
        spec, t0 = self.spec, time.time()
        lo = S.sub(st.front_lo, spec.s) if st.front_lo is not None else const(64, 0)
        sp = span(spec, st, lo, end, spec.at)
        ways = sp.alts if isinstance(sp, Alt) else [[sp]]
        order = list(reversed(range(len(ways))))  # the loads' way first
        best = self.best_stop.get((st.rip, len(ways)))
        rounds = [(k, AUX) for k in order] + ([(best, None)] if best is not None else []) + [(order[0], None)] + [(k, 6 * AUX) for k in order[1:]]
        done = set()
        for k, limit in rounds:
            rs = []
            for i, part in enumerate(ways[k]):
                r = 'unsat' if (k, i) in done else self.solver.check(*st.pc, *part, timeout=limit, abstract=self.abstract_of(st))
                if r == 'unsat':
                    done.add((k, i))
                rs.append(r)
                if r != 'unsat':
                    break
            if self.args.verbose:
                print('   no stop below the access before %#x: %s (way %d of %d, %s s) %.1f s' % (st.rip, '/'.join(rs), k + 1, len(ways), limit or 'all',
                                                                                          time.time() - t0), flush=True)
            if len(rs) == len(ways[k]) and all(r == 'unsat' for r in rs):
                self.best_stop[(st.rip, len(ways))] = k
                return S.not_(S.and_(S.ule(lo, spec.z), S.ult(spec.z, end)))
        return None

    def never(self, st, conds, what, lanes=True, end=None):
        """conds can not hold on the path: a counterexample when they can,
        a failure when z3 can not tell. A disjunction among them goes one
        disjunct at a time (a load's page bounds, below and above: two
        queries of 6 and 17 s where the one took 72); a masked access's
        lanes all at once, one at a time when that does not settle soon."""
        self.solver.doing = what
        flat = []
        for c in conds:
            flat.extend(c.args if c.op == 'and_b' else [c])
        ors = [c for c in flat if c.op == 'or_b' and len(c.args) <= 4]
        if ors:
            rest = [c for c in flat if c is not ors[0]]
            for x in ors[0].args:
                self.never(st, rest + [x], what, lanes, end)
            return
        # first on what the path says of the addresses alone (fewer hypotheses: unsat holds for the whole): the
        # bytes compared before, reads at the stop's offset among them, can take the solver half an hour
        plain = self.plain(st)
        if len(plain) < len(self.solver.assumed) + len(st.pc) and self.solver.check(*plain, *flat, bare=True, timeout=AUX, abstract=self.abstract_of(st)) == 'unsat':
            return
        # then with this way round's conditions on memory too (past a loop's head: the masks of the block it
        # tests, "a byte found", not the bytes the path compared before the loop): a tzcnt's index below 64
        recent = [c for c in st.pc[st.gen['pc']:] if S.has_array(c)] if st.gen is not None and 'pc' in st.gen else []
        if recent:  # and the assumptions on memory (z a stop: the found byte's NUL against the block's masks)
            recent = [c for c in self.solver.assumed if S.has_array(c)] + recent
        if recent and len(plain) + len(recent) < len(self.solver.assumed) + len(st.pc) and \
                self.solver.check(*plain, *recent, *flat, bare=True, timeout=6 * AUX, abstract=self.abstract_of(st)) == 'unsat':  # strcpy_long's: 83 s, the path's 133
            return
        each = next((c for c in flat if c.op == 'or_b'), None) if lanes else None
        self.solver.doing = what + ': the path'
        r = self.solver.check(*st.pc, *flat, keep=True, timeout=AUX if each is not None else 3 * AUX if end is not None else None, abstract=self.abstract_of(st))
        if r == 'unknown' and end is not None:  # a load past a stop the bytes compared before rule out (strcmp evex
            # past its loops: a byte at a mask's first set bit, 870 s): the stop not below its page first, by the loads
            lemma = None
            for k, e in enumerate(end if isinstance(end, list) else [end]):  # the ends a lemma may take, the strongest first
                self.solver.doing = what + ': no stop below it (%d)' % (k + 1)
                got = self.no_stop_below(st, e)
                if got is not None and self.solver.check(*plain, got, *flat, bare=True, timeout=AUX) == 'unsat':
                    return
                lemma = lemma or got
            self.solver.doing = what + ': the path%s' % (' and the lemma' if lemma is not None else ', no lemma')
            r = self.solver.check(*st.pc, *([lemma] if lemma is not None else []), *flat, keep=True, abstract=self.abstract_of(st))
        if r == 'unknown' and each is not None:  # strcmp evex past three loops: 431 s, a lane alone 1 or 2 s
            rest = [c for c in flat if c is not each]
            for x in each.args:
                self.never(st, rest + [x], what, False)
            return
        if r == 'sat':
            self.fail(what, st, self.spec.show())
        if r != 'unsat':
            raise Failure('%s: z3 answered %s' % (what, r))

    def prove(self, st, obligations, show):
        for what, conds in obligations:
            if not isinstance(conds, Alt) and len(chunked(conds)) > 1:
                conds = Alt([[conds]])
            if isinstance(conds, Alt):
                self.prove_alt(st, what, conds, show)
                continue
            t1 = time.time()
            self.solver.doing = what
            r = self.solver.check(*st.pc, *conds, keep=True, abstract=self.abstract_of(st))
            if self.args.verbose:
                print('   %-50s %s %.1f s' % (what, r, time.time() - t1), flush=True)
            if r == 'sat':
                self.fail(what, st, show)
            if r == 'unknown':
                raise Failure('%s: z3 answered unknown; branches at %s' % (what, ' '.join('%#x' % x for x in st.trace)))
            self.proved += 1

    def prove_alt(self, st, what, alt, show):
        """The ways in rounds: each but the last for 10 s, again for 120,
        then the last with the full timeout, then the others with it."""
        ways = [[q for part in way for q in chunked(part)] for way in alt.alts]
        rounds = [(k, AUX // 2) for k in range(len(ways) - 1)] + [(k, 6 * AUX) for k in range(len(ways) - 1)] + [(len(ways) - 1, None)] + \
            [(k, None) for k in range(len(ways) - 1)]
        best = self.best_way.get((what, len(ways)))
        if best is not None:  # the way that did it last time first
            rounds.insert(0, (best, 6 * AUX))
        done = set()  # the parts proved in an earlier round
        self.solver.doing = what
        self.solver.note = 'at %#x, branches at %s' % (st.rip, ' '.join('%#x' % x for x in st.trace[-12:]))
        for k, limit in rounds:
            t1 = time.time()
            rs = []
            for i, conds in enumerate(ways[k]):
                r = 'unsat' if (k, i) in done else self.solver.check(*st.pc, *conds, keep=True, timeout=limit, abstract=self.abstract_of(st))
                if r == 'unsat':
                    done.add((k, i))
                if r == 'sat':
                    self.fail(what, st, show)
                rs.append(r)
                if r != 'unsat':
                    break
            if self.args.verbose:
                print('   %-50s %s (way %d of %d, %s s) %.1f s' % (what, '/'.join(rs), k + 1, len(ways), limit or 'all', time.time() - t1), flush=True)
            if len(rs) == len(ways[k]) and all(r == 'unsat' for r in rs):
                self.proved += 1
                self.best_way[(what, len(ways))] = k
                return
        raise Failure('%s: z3 answered unknown every way; branches at %s' % (what, ' '.join('%#x' % x for x in st.trace)))

    def generalize(self, s1, s2, dropped=frozenset(), widen=frozenset()):
        """s1 and s2: the same path at a loop's head, the first and the second
        time. The state for every iteration, after the base case is proved.
        The registers that moved by constants move by the multiples of K, a
        multiple of the frontier's step; a loop whose pointers moved by an
        amount that is not a constant (another loop's, inside it) gets K any
        number (step 1), its pointers that moved alike the first plus K."""
        moved, havoc, deltas = {}, [], {}
        for r in s1.gpr:
            if s2.gpr[r] is s1.gpr[r]:
                continue
            d = S.sub(s2.gpr[r], s1.gpr[r])
            if S.is_const(d) and r not in widen:
                moved[r] = S.signed(d)
            else:
                havoc.append(r)  # recomputed in the loop: any value (sound; a proof that needs it fails)
                deltas[r] = d
        for r in widen:  # changed on another way round the loop
            if r in s1.gpr and r not in havoc:
                havoc.append(r)
        deltas = {r: d for r, d in deltas.items() if r not in widen}
        def detect(base, moved, havoc):
            """The frontier into the object at base: (front, step, a1, down,
            variable, moved, havoc), or a Failure."""
            moved, havoc = dict(moved), list(havoc)
            bv = S.variables(base)
            others = frozenset().union(*[S.variables(b) for b, _ in s1.regions if b is not base], getattr(self.spec, 'foreign', frozenset())) - bv

            foreign = getattr(self.spec, 'foreign', frozenset())

            def inside(x):  # an address into the object: a sum with its base, a pointer to it rounded down, else a term of its base's variables and no other's (not the source's of a memmove)
                if foreign & S.variables(x):
                    return False
                if X.offset_from(x, base) is not None or any(inside(y) and not any(X.offset_from(r, b) is not None for b, _ in s1.regions
                                                                                              if b is not base) for y, r in X.rounded(x)):
                    return True
                return bv & S.variables(x) and not others & S.variables(x) and S.interval(x) is None  # bounded: an index (memcmp's 0xa0 - (a & 31)), not an address

            def into(r):
                return inside(s1.gpr[r])

            front = [r for r in moved if X.offset_from(s1.gpr[r], base) is not None and moved[r] > 0 and not others & S.variables(s1.gpr[r])]
            if not front:  # a pointer rounded down: one that depends on the object's base, not on another region's
                front = [r for r in moved if moved[r] > 0 and into(r)]
                if len(front) > 1:  # the one the iteration's accesses are at
                    acc = [e[0] for e in s2.loads[len(s1.loads):] + s2.stores[len(s1.stores):]]
                    front = [r for r in front if any(X.offset_from(x, s1.gpr[r]) is not None and S.is_const(X.offset_from(x, s1.gpr[r])) for x in acc)]
            if len(front) > 1 and all(s1.gpr[r] is s1.gpr[front[0]] and s2.gpr[r] is s2.gpr[front[0]] for r in front):
                front = front[:1]  # copies of one pointer (an unrolled loop's exits read r9 = rax): one frontier, not a base and an index
            down = variable = False
            if len(front) == 1:
                step = moved[front[0]]  # the frontier's step; the other registers move by multiples of it (a count down, say)
                a1 = s1.gpr[front[0]]
                # a register past the bytes it reads (loads at rcx - 3 .. rcx), or
                # behind them (truffle's rax rounded down, its loads at rax + 32
                # .. rax + 63 through rcx): the frontier is the lowest of them
                acc = [e[0] for e in s2.loads[len(s1.loads):] + s2.stores[len(s1.stores):] if inside(e[0])]
                below = [S.signed(S.sub(x, a1)) for x in acc if S.is_const(S.sub(x, a1))]
                if below and (min(below) < 0 or len(below) == len(acc) and min(below) > 0):
                    a1 = S.add(a1, const(64, min(below)))
            elif not front and not [d for d in moved.values() if d > 0] and [r for r in deltas if into(r)]:
                # moved by what an inner loop did: the pointers into the object
                # and every register that moved by the same amount
                cand = [r for r in deltas if into(r)]
                # the most registers moved alike (a's and b's pointers), a sum with the base first
                r0 = max(cand, key=lambda r: (sum(deltas[x] is deltas[r] for x in deltas), X.offset_from(s1.gpr[r], base) is not None))
                front, step, a1, variable = [r0], 1, s1.gpr[r0], True
                for r in list(havoc):
                    if deltas.get(r) is deltas[r0]:
                        havoc.remove(r)
                        moved[r] = 1
                for r in [r for r in moved if r not in deltas or deltas[r] is not deltas[r0]]:
                    del moved[r]  # the constant movers: their counts unknown
                    havoc.append(r)
            else:
                # no register points into the object (a base and an index): the
                # lowest address the first iteration touched, the smallest step;
                # nothing moves up: a loop that runs down, the frontier the lowest
                # address of the iteration it is about to run
                steps = [d for d in moved.values() if d > 0]
                if not steps and not front:
                    steps, down = [-d for d in moved.values() if d < 0], True
                acc = [e[0] for e in s2.loads[len(s1.loads):] + s2.stores[len(s1.stores):] if inside(e[0])]
                low = [x for x in acc if all(S.is_const(S.sub(y, x)) and S.signed(S.sub(y, x)) >= 0 for y in acc)]
                if not steps or not low:
                    raise Failure('loop at %#x: no frontier (registers %s, %d accesses into the object, none the lowest)' % (s1.rip, front, len(acc)))
                step = min(steps)
                a1 = low[0]
            return front, step, a1, down, variable, moved, havoc

        side = False
        try:
            front, step, a1, down, variable, moved, havoc = detect(self.spec.base, moved, havoc)
        except Failure:
            if getattr(self.spec, 'side_base', None) is None:
                raise
            # a loop over the spec's other object (strspn's set, building its
            # table): the spec's side invariants, the scan's frontier kept
            front, step, a1, down, variable, moved, havoc = detect(self.spec.side_base, moved, havoc)
            side = True
        if down and not hasattr(self.spec, 'cover_general'):
            raise Failure('loop at %#x runs down: only the writers have that' % s1.rip)
        lane_step = step  # a vector's lanes move by multiples of the first iteration's step
        if '#step' in widen and step > 1 and not variable:  # another way round moved by other than a multiple: step 1, K any
            for r in list(moved):
                if moved[r] == step:
                    moved[r] = 1
                elif moved[r] == -step:
                    moved[r] = -1
                else:
                    del moved[r]
                    havoc.append(r)
            step = 1
        if step & (step - 1):
            raise Failure('loop at %#x: the frontier\'s step %d, not a power of two' % (s1.rip, step))
        mode = 'B' if variable or side else self.spec.mode(self.solver, s1, a1, step)
        if side:
            self.later(self.prove, s1, self.spec.side_base_case(s1, a1, step), self.spec.show())
        else:
            self.later(self.prove, s1, self.spec.base_case(s1, a1, mode, step, down) if down else self.spec.base_case(s1, a1, mode, step), self.spec.show())
        kept = [(n, f) for n, f in self.candidates(s1, moved, havoc, step, a1, side, s2) if n not in dropped and self.holds(s1, n, f, a1)]  # true at the first arrival
        self.proved += len(kept)
        self.loops += 1
        sh = step.bit_length() - 1
        k = S.concat(var('k%d_hi' % self.loops, 64 - sh), const(sh, 0)) if sh else var('k%d' % self.loops, 64)  # any multiple of the step
        g = s1.copy()
        g.fresh = 'g%d_' % self.loops
        for r, d in moved.items():  # d each iteration: d times K / step (a count down by 1 beside a pointer by 128, say)
            g.gpr[r] = moved_by(s1.gpr[r], d, step, k)
        for r in havoc:
            g.gpr[r] = var(g.fresh + r, 64)
        g.regions = [(b, a if a is s2.regions[i][1] else S.mem_var('%smem%d' % (g.fresh, i))) for i, (b, a) in enumerate(s1.regions)]  # written: unknown
        swin = None
        if s1.regions[-1][1] is not s2.regions[-1][1] and '#stack' not in widen:
            # the stack the iteration wrote (spills, a call's return address):
            # unknown only there, the return address and the rest kept
            offs, m = [], s2.regions[-1][1]
            while m is not s1.regions[-1][1] and m.op == 'store':
                offs.append(m.args[1])
                m = m.args[0]
            ivs = [S.interval(o) for o in offs]  # a table's c + a byte: [c, c + 255]
            if m is s1.regions[-1][1] and offs and all(iv is not None for iv in ivs):
                lo, hi = min(iv[0] for iv in ivs), max(iv[1] for iv in ivs) + 1
                swin = (lo, hi)
                g.regions[-1] = (X.RSP0, S.win(s1.regions[-1][1], const(64, lo & S.mask(64)), const(64, hi & S.mask(64)), S.mem_var('%sstack' % g.fresh)))
        windows = []
        a = S.sub(a1, k) if down else S.add(a1, k)  # the frontier: up, where the next iteration starts; down, the lowest address it touches
        lo, hi = (S.add(a, const(64, step)), S.add(a1, const(64, step))) if down else (a1, a)  # what the iterations so far covered
        w = None
        if hasattr(self.spec, 'cover_general'):
            # a writer: the coverage so far from the stores' frontier, the
            # lowest store of the first iteration (behind the scan's in a
            # pipelined loop); the stores' values checked in the base case
            sts = [e[0] for e in s2.stores[len(s1.stores):]]
            low = [x for x in sts if all(S.is_const(S.sub(y, x)) and S.signed(S.sub(y, x)) >= 0 for y in sts)]
            if sts and not low:
                raise Failure('loop at %#x: no store of the first iteration the lowest' % s1.rip)
            w1 = S.sub(low[0], self.spec.d) if sts else None
            if w1 is not None:
                w = S.sub(w1, k) if down else S.add(w1, k)
                g.cover = self.spec.cover_general(s1, S.add(w, const(64, step)), S.add(w1, const(64, step))) if down else self.spec.cover_general(s1, w1, w)
            else:
                g.cover = self.spec.cover_general(s1, None, None)
        if getattr(self.spec, 'window', False):
            # the first arrival's memory but for what the iterations so far
            # covered, there unknown: a memmove's source may lie in it
            for i, (b, arr) in enumerate(s1.regions):
                if arr is not s2.regions[i][1]:
                    windows.append((i, b, arr))
                    g.regions[i] = (b, self.window(arr, S.sub(lo, b), S.sub(hi, b), S.mem_var('%swin%d' % (g.fresh, i))))
                    g.win = dict(g.win)
                    g.win[i] = (S.sub(lo, b), S.sub(hi, b), arr)
        g.vec, vmoved = {}, {}
        for n, v1 in s1.vec.items():
            v2 = s2.vec.get(n)
            if v2 is None:
                continue
            if 'zmm%d' % n in widen:
                continue
            if all(x is y for x, y in zip(v1, v2)):
                g.vec[n] = v1  # the same lanes: kept
                continue
            if variable:
                continue
            # lanes loaded from memory at offsets that move with the frontier
            # (a group loaded ahead of its store): moved
            lanes = []
            for x, y in zip(v1, v2):
                if x is y:
                    lanes.append((None, x, 0))  # a lane the loop leaves alone (the zeroed upper ones)
                    continue
                if x.op == 'select' and y.op == 'select':  # the same array (memory the loop leaves alone), or proved at the step
                    dlt = S.sub(y.args[1], x.args[1])
                    if S.is_const(dlt) and S.signed(dlt) % lane_step == 0:
                        lanes.append((x.args[0], x.args[1], S.signed(dlt) // lane_step))
                        continue
                lanes = None
                break
            if lanes:
                g.vec[n] = [o if a is None else S.select(a, S.add(o, k if m == 1 else S.mul(k, const(64, m)))) for a, o, m in lanes]
                vmoved[n] = [(a, o, m) for a, o, m in lanes]
            elif self.args.verbose:
                x, y = v1[0], v2[0]
                print('    zmm%d not moved: %s(%s) / %s(%s)' % (n, x.op, ','.join(getattr(z, 'op', str(z)) for z in x.args), y.op,
                                                           ','.join(getattr(z, 'op', str(z)) for z in y.args)), flush=True)
        g.k = {n: v for n, v in s1.k.items() if s2.k.get(n) is v and n not in widen}
        g.flags = {}
        g.loads, g.aligned, g.stores, g.requires = [], [], [], []
        g.snaps = dict(s1.snaps)  # the heads met before: an enclosing loop's, generalized when this one's exit gets back to it
        if not side:  # a side loop leaves the scan's frontier as it was
            g.front_lo = a
            g.front_bytes = list(s1.front_bytes) + ([a] if mode == 'A' else [])
        g.pc = list(s1.pc) + [S.ult(k, const(64, K_LIMIT))] + ([] if side else self.spec.general(a)) + [f(g, a) for n, f in kept]
        g.gen = {'head': s1.rip, 'front': a, 'moved': moved, 'havoc': havoc, 'step': step, 'mode': mode, 'regs': dict(g.gpr), 'pc': len(g.pc),
                 'steps': g.steps, 'kept': kept, 'k': k, 'vec': dict(g.vec), 'vmoved': vmoved, 'kregs': dict(g.k),
                 'mem': [arr for _, arr in g.regions], 'down': down, 'a1': a1, 'windows': windows, 'parent': s1.gen, 'fresh': g.fresh, 'w': w,
                 'side': side, 'swin': swin}
        if self.args.verbose:
            print('  loop at %#x: step %s, frontier %s, loads %s, %s move, %s recomputed, %d invariant candidates; vectors kept %s, moved %s' % (
                s1.rip, 'any' if variable else ('-' if down else '') + str(step), front[0] if len(front) == 1 else 'from the accesses', mode,
                sorted(moved), sorted(havoc), len(kept), sorted(set(g.vec) - set(vmoved)), sorted(vmoved)), flush=True)
            if side:
                print('    side loop, kept: %s' % '; '.join(n for n, f in kept), flush=True)
            print('    arrived by %s (%d branches, %d conditions)' % (' '.join('%#x' % x for x in s1.trace[-14:]), len(s1.trace), len(s1.pc)), flush=True)
        return g

    def window(self, arr, lo, hi, h):
        return S.win(arr, lo, hi, h)

    def holds(self, st, name, f, front):
        """Whether candidate f holds at a head state with that frontier. A
        spec may know a cheaper way to the same (no stop below an offset: the
        bytes one by one)."""
        way = getattr(self.spec, 'candidate_holds', None)
        if way is not None:
            r = way(st, name, front)
            if r is not None:
                return r
        # a table's invariant needs no assumption (they have quantifiers: z3 slows down on them)
        table = ' table ' in name  # quantified: a false one may take z3 long to refute; unknown drops it (sound)
        if table and hasattr(self.spec, 'table_holds'):
            r = self.spec.table_holds(st, name, front)
            if r is not None:
                return r
        return self.solver.check(*st.pc, S.not_(f(st, front)), bare=table, timeout=6 * AUX if table else None, qtimeout=AUX) == 'unsat'

    def candidates(self, s1, moved, havoc, step, a1, side=False, s2=None):
        """Invariant candidates, as functions of a head state and its frontier:
        the spec's bounds on the frontier; every moving register against every
        register the loop leaves alone and against the constants its body
        compares with (<, <= signed and unsigned, both ways)."""
        out = list(self.spec.side_candidates(s1, s2, a1, step) if side else self.spec.candidates(step))
        rels = [('<u', S.ult), ('<=u', S.ule), ('<s', S.slt), ('<=s', S.sle)]
        pairs, consts = set(), {}
        a = s1.rip
        for _ in range(400):  # the body: from the head to the jump back to it; its comparisons
            j = self.mach.image.decode(a)
            if j['mnemonic'] in ('cmp', 'test'):
                regs = [X.GPR[o['reg']][0] for o in j['ops'] if o['kind'] == 'Register' and o['reg'] in X.GPR]
                imms = [o for o in j['ops'] if o['kind'] == 'Immediate']
                if len(regs) == 2 and regs[0] != regs[1]:
                    pairs.add(tuple(regs))
                for o in imms:
                    v = S.sext(const(o['size'], o['imm']), 64) if o.get('signed') else const(64, o['imm'])
                    for r in regs:
                        consts.setdefault(r, set()).update({v.val, (v.val + 1) & S.mask(64), (v.val - 1) & S.mask(64)})
            if j.get('branch_target') == s1.rip:
                break
            a += j['len']
        for r, d in moved.items():  # counters: down ones stay at or above 1 and below where they began (no wrap), up ones above
            r1 = s1.gpr[r]
            if d < 0:
                out.append(('%s >=u 1' % r, (lambda r: lambda st, f: S.ule(const(64, 1), st.gpr[r]))(r)))
                out.append(('%s <=u its first' % r, (lambda r, r1: lambda st, f: S.ule(st.gpr[r], r1))(r, r1)))
            else:
                out.append(('%s >=u its first' % r, (lambda r, r1: lambda st, f: S.ule(r1, st.gpr[r]))(r, r1)))
        for x, y in pairs:
            if x not in moved and y not in moved:
                continue
            if x in havoc or y in havoc:
                continue
            for nm, op in rels:
                out.append(('%s %s %s' % (x, nm, y), (lambda x, y, op: lambda st, f: op(st.gpr[x], st.gpr[y]))(x, y, op)))
                out.append(('%s %s %s' % (y, nm, x), (lambda x, y, op: lambda st, f: op(st.gpr[y], st.gpr[x]))(x, y, op)))
        for m, cs in consts.items():
            if m not in moved:
                continue
            for cv in sorted(cs):
                for nm, op in rels:
                    out.append(('%s %s %#x' % (m, nm, cv), (lambda m, cv, op: lambda st, f: op(st.gpr[m], const(64, cv)))(m, cv, op)))
                    out.append(('%#x %s %s' % (cv, nm, m), (lambda m, cv, op: lambda st, f: op(const(64, cv), st.gpr[m]))(m, cv, op)))
        return out

    def back_at_head(self, st, gen):
        """A general state's path back at its loop's head (from inside an
        inner loop too): the same form with K + D, D what the frontier moved
        this time (a multiple of the step; in a loop that runs down, the
        step), K + D below the limit, the invariant kept."""
        step, regs, moved = gen['step'], gen['regs'], gen['moved']
        ref = [r for r in moved if moved[r] in (step, -step)]
        if not ref:
            raise Failure('loop at %#x: no register moves by the frontier\'s step' % st.rip)
        dd = S.sub(st.gpr[ref[0]], regs[ref[0]])
        D = dd if moved[ref[0]] > 0 else S.neg(dd)
        off_form, changed = [], set()
        for r, v in regs.items():
            if r in gen['havoc']:
                continue
            if r in moved:
                want = moved_by(v, moved[r], step, D)
                if st.gpr[r] is not want:
                    off_form.append(S.not_(S.eq(st.gpr[r], want)))
            elif st.gpr[r] is not v:
                changed.add(r)
        # the rest of the state too: what the general state kept must come back
        # the same, what moved D on, the memory not written as it was
        k1 = S.add(gen['k'], D)
        lanes_differ = []
        for n, lanes in gen['vec'].items():
            now = X.read_vec(st, 'zmm%d' % n)
            if n in gen['vmoved']:
                want = [o if a is None else S.select(a, S.add(o, k1 if m == 1 else S.mul(k1, const(64, m)))) for a, o, m in gen['vmoved'][n]]
                mine = []
                for x, y in zip(now, want):  # loaded from the memory as it is now: the same byte, the solver decides
                    if x is not y:
                        if x.op != 'select' or y.op != 'select':
                            changed.add('zmm%d' % n)
                            break
                        mine.append(S.not_(S.eq(x, y)))
                if mine and 'zmm%d' % n not in changed:
                    if self.solver.check(*st.pc, S.or_(*mine)) == 'unsat':
                        lanes_differ += mine
                    else:
                        changed.add('zmm%d' % n)  # not of that form after all (reloaded each time round): unknown
            elif any(x is not y for x, y in zip(now, lanes)):
                changed.add('zmm%d' % n)
        for n, v in gen['kregs'].items():
            if st.k.get(n) is not v:
                changed.add(n)
        if changed:  # kept in the general state, changed on this way round: unknown there, the loop again
            self.widen.setdefault(gen['head'], set()).update(changed)
            return
        win = {i for i, _, _ in gen['windows']}
        if gen['swin'] is not None:  # the stack: written only inside its window (else the loop again with the stack unknown)
            win.add(len(gen['mem']) - 1)
            m = st.regions[-1][1]
            while m is not gen['mem'][-1] and m.op == 'store':
                iv = S.interval(m.args[1])
                if iv is None or not gen['swin'][0] <= iv[0] <= iv[1] < gen['swin'][1]:
                    self.widen.setdefault(gen['head'], set()).add('#stack')
                    return
                m = m.args[0]
            if m is not gen['mem'][-1]:
                self.widen.setdefault(gen['head'], set()).add('#stack')
                return
        for i, arr in enumerate(gen['mem']):
            if i in win or arr.op == 'var' and arr.args[0].startswith(gen['fresh']):
                continue  # the window's, checked below; unknown in the general state: anything
            if st.regions[i][1] is not arr:
                raise Failure('loop at %#x: region %d written though the general state keeps it' % (st.rip, i))
        a = gen['front']
        a2 = S.sub(a, D) if gen['down'] else S.add(a, D)  # the next iteration's frontier
        plain = [(n, f) for n, f in gen['kept'] if ' table ' not in n]  # tables one by one, without the assumptions (holds)
        if plain and self.solver.check(*st.pc, S.or_(*[S.not_(f(st, a2)) for n, f in plain])) == 'unsat':
            lost = []  # all kept, one query
        else:
            lost = [n for n, f in plain if not self.holds(st, n, f, a2)]
        for n, f in gen['kept']:  # a table's invariant: its step without quantifiers when the spec has one
            if ' table ' in n:
                way = getattr(self.spec, 'table_step', None)
                r = way(st, gen, n, a, a2) if way is not None else None
                if r is None:
                    r = self.holds(st, n, f, a2)
                if not r:
                    lost.append(n)
        if lost:  # gathered over the loop's subtree, then the loop again without them
            self.lost.setdefault(gen['head'], set()).update(lost)
            return
        self.proved += len(gen['kept'])
        st = st.copy()
        st.pc += [f(st, a2) for n, f in gen['kept']]  # proved: what the rest may use
        out = []
        if off_form:
            out.append(('the registers come back in the general form', [S.or_(*off_form)]))
        if gen['down']:
            out.append(('a loop down moves by its step', [S.not_(S.eq(D, const(64, step)))]))
        elif step > 1 and (not S.is_const(D) or D.val % step) and \
                self.solver.check(*st.pc, S.not_(S.is_zero(S.band(D, const(64, step - 1))))) != 'unsat':
            self.widen.setdefault(gen['head'], set()).add('#step')  # the loop again with step 1
            return
        out.append(('the distance from the first iteration below the limit', [S.not_(S.ult(k1, const(64, K_LIMIT)))]))
        if lanes_differ:
            out.append(('the vectors loaded ahead hold the bytes of the general state\'s form', [S.or_(*lanes_differ)]))
        xo = var('xo', 64)
        for i, b, arr in gen['windows']:  # every byte the iteration wrote inside the window grown by it: the same form one step on
            lo, hi = (S.sub(S.add(gen['a1'], const(64, step)), k1), S.add(gen['a1'], const(64, step))) if gen['down'] else (gen['a1'], S.add(gen['a1'], k1))
            lo, hi = S.sub(lo, b), S.sub(hi, b)
            bad = []
            for sa, nb, _, c, _ in st.stores:
                r = X.region_of(st, sa)
                if r is not None and r[0] == i:
                    x = S.add(r[1], xo)
                    bad.append(S.and_(c, S.ult(xo, nb if isinstance(nb, S.T) else const(64, nb)), S.not_(S.and_(S.ule(lo, x), S.ult(x, hi)))))
            if bad:
                out.append(('the iteration writes only inside the window grown by it', [S.or_(*bad)]))
        self.later(self.prove, st, out, self.spec.show())
        w = gen['w']
        cov = None if w is None else (w, S.add(w, const(64, step)) if gen['down'] else S.add(w, D))  # the stores' part of this iteration
        if gen['side']:
            self.later(self.prove, st, self.spec.side_step_case(st, a, a2, step), self.spec.show())
            return
        self.later(self.prove, st, self.spec.step_case(st, a, S.add(a, const(64, step)) if gen['down'] else a2, gen['mode'], step, cov), self.spec.show())

    def run(self, entry, params):
        self.lost = {}
        self.unroll = getattr(self.spec, 'unroll', 1)
        self.contracts = self.spec.contracts(self.mach.image, entry) if hasattr(self.spec, 'contracts') else {}
        st = X.fresh_state(entry, params, self.spec.regions, getattr(self.spec, 'aliases', ()))
        if hasattr(self.spec, 'memory'):  # what the spec has in memory outside its regions (strtok's state)
            st.mem = self.spec.memory(st.mem)
        st.front_lo = getattr(self.spec, 'front0', None)  # a precondition's: no stop before it
        if self.prefix:
            self.backs = X.backward_targets(self.mach.image, entry)
        if self.args.workers > 1:
            self.pool = Pool(self, self.args.workers)
        self.explore(st)
        if self.pool is not None:
            self.pool.finish()

    def explore_loop(self, s1, s2):
        """A loop met the second time on a path: generalized, its subtree (the
        rest of the path for every iteration count) explored on its own;
        again without the invariant candidates some step did not keep, and
        with what another way round the loop changed unknown."""
        dropped, widen = set(), set()
        while True:
            g = self.generalize(s1, s2, frozenset(dropped), frozenset(widen))
            self.lost.pop(s1.rip, None)
            self.widen.pop(s1.rip, None)
            self.explore(g)
            lost = self.lost.pop(s1.rip, set())
            wider = self.widen.pop(s1.rip, set()) - widen
            if not lost and not wider:
                return
            dropped |= lost
            widen |= wider
            if self.args.verbose:
                print('  the loop at %#x: %d candidates not kept (%s), %s changed elsewhere; its subtree again' % (
                    s1.rip, len(lost), ', '.join(sorted(lost)[:4]), ', '.join(sorted(wider)) or 'nothing'), flush=True)

    def status(self, st):
        """PROVE_STATUS=dir: a line a minute in dir/<func>-<classes>-<align>.txt
        on how far the proof is: the paths closed, the ways still to walk in
        each walk under way (the outermost first), the loops generalized,
        where it is, the checking threads' backlog, and the queries running
        longer than a minute, their seconds and what they are for. Written
        by a thread of its own: the walk may wait on a full backlog."""
        self.where = st
        if self.status_thread is None:
            self.status_thread = threading.Thread(target=self.status_loop, daemon=True)
            self.status_thread.start()

    def status_loop(self):
        name = '%s-%s-%s-%s%s.txt' % (self.args.func, self.args.spec, self.args.classes or self.align, self.a, '-p' + self.prefix if self.prefix else '')
        while True:
            time.sleep(60)
            now, st = time.time(), self.where
            solvers = [self.main_solver] + (self.pool.solvers if self.pool else [])
            busy = sorted(([round(now - x.since), x.doing] for x in solvers if x.since and now - x.since > 60), reverse=True)
            line = {'s': round(now - self.t0), 'paths': self.paths, 'obligations': self.proved, 'loops': self.loops,
                    'ways': [len(w) for w in self.works], 'rip': '%#x' % st.rip, 'loop': None if st.gen is None else '%#x' % st.gen['head'],
                    'queries': sum(x.queries for x in solvers), 'backlog': self.pool.jobs.qsize() if self.pool else 0, 'busy': busy}
            try:
                with open(os.path.join(os.environ['PROVE_STATUS'], name), 'a') as f:
                    f.write(json.dumps(line) + '\n')
            except OSError:
                pass

    def explore(self, st0):
        """--prefix: a proof split by its paths. At the first two-way branches
        of a path (before it meets a loop or a contract: the walk's top level)
        this run takes the way the prefix says, 0 the taken one, 1 the other;
        a path that meets a loop, a contract or its end before the prefix is
        used up goes on only where the rest of the prefix is zeros. The
        prefixes of a length are then a partition of the paths: each
        explored by one run, every one by some run."""
        work = [st0]
        self.works.append(work)
        targets = self.mach.targets
        report = 'PROVE_STATUS' in os.environ
        top = len(self.works) == 1 and bool(self.prefix)
        while work:
            st = work.pop()
            closed = dropped = False
            while not st.done:
                if report:
                    self.status(st)
                if top and not st.free and (st.rip in self.backs or st.rip in self.contracts):
                    if '1' in self.prefix[st.fork:]:  # another run's
                        dropped = closed = True
                        break
                    st.free = True
                if st.steps > self.args.steps:
                    raise Failure('a path past %d instructions at %#x' % (self.args.steps, st.rip))
                contract = self.contracts.get(st.rip)
                if contract is not None:  # a callee the spec takes by its contract (proved on its own): no loop's head
                    if contract(self, st):  # a tail call: the path ends with it
                        closed = True
                        break
                    continue
                if self.args.induct and st.rip in targets:
                    g = st.gen  # this loop's general state, or an enclosing one's (back from inside an inner loop)
                    while g is not None and g['head'] != st.rip:
                        g = g['parent']
                    if g is not None and (g is not st.gen or st.steps > g['steps']):
                        self.back_at_head(st, g)
                        closed = True
                        break
                    seen_n, snap = st.snaps.get(st.rip, (0, None))
                    if seen_n < self.unroll or (snap is not None and seen_n < 4 * SETMAX and hasattr(self.spec, 'keep_unrolling')
                                                and self.spec.keep_unrolling(snap, st)):  # the first arrivals run on (a short loop runs out in them)
                        if seen_n == 0 and hasattr(self.spec, 'canonical') and os.environ.get('PROVE_CANON') != '0':
                            self.spec.canonical(self, st)
                        st.snaps[st.rip] = (seen_n + 1, st.copy())
                    else:
                        if id(snap) not in self.covered:  # a loop; after another one on this path, the next in a row
                            self.explore_loop(snap, st)
                            self.covered[id(snap)] = snap
                        # else another way round from the same first arrival: in the general state, whose
                        # subtree (every way round, every way out) is explored, from that arrival (the base case)
                        closed = True
                        break
                forks = self.shift_cases(st)
                if forks:  # a shift by a register of a few values: a way per value, each shift by a constant
                    st = forks[0]
                    work.extend(forks[1:])
                seen, seen_st, seen_al, seen_rq = len(st.loads), len(st.stores), len(st.aligned), len(st.requires)
                succ = self.mach.step(st)
                for bad, what, rrip in st.requires[seen_rq:]:
                    self.later(self.never, st, [bad], '%s at %#x' % (what, rrip))
                for aa, an, arip in st.aligned[seen_al:]:  # a misaligned movaps, movdqa or legacy SSE operand faults, its mask whatever
                    self.later(self.never, st, [S.not_(S.is_zero(S.band(aa, const(64, an - 1))))], 'the access at %#x needs %d-byte alignment and may not have it' % (arip, an))
                new = st.loads[seen:]
                if new and hasattr(self.spec, 'settle'):
                    self.spec.settle(self, st, new)
                stop_end = getattr(self.spec, 'stop_end', lambda a, n: None)
                if len(new) == 1:
                    la, ln, lrip, lc = new[0]
                    self.later(self.never, st, [lc, self.spec.early(la, ln)], 'the load at %#x reaches past what the object allows' % lrip,
                               end=stop_end(la, ln))
                elif new:  # a masked load's lanes
                    what = 'the load at %#x reaches past what the object allows' % new[0][2]
                    la0, p0 = new[0][0], page(new[0][0])
                    cnt = prefix_count(new)
                    if cnt is not None:  # the lanes below a count (bzhi's mask): only the last of them can leave lane 0's page
                        some = S.ult(const(64, 0), cnt)
                        last = S.sub(S.add(la0, S.mul(cnt, const(64, new[0][1]))), const(64, 1))
                        off = S.and_(some, S.or_(S.not_(S.eq(page(last), p0)), S.ult(last, la0)))
                    else:
                        some = S.or_(*[lc for _, _, _, lc in new])
                        off = S.or_(*[S.and_(lc, S.not_(S.and_(S.eq(page(la), p0), S.eq(page(last_of(la, ln)), p0)))) for la, ln, _, lc in new])
                    if self.solver.check(*self.plain(st), off, bare=True, timeout=AUX) == 'unsat':
                        # every lane it may load in lane 0's page (a mask up to the page's end): lane 0's bounds are
                        # theirs (strcmp evex: 0.8 and 1.1 s where the lanes took 431); one query whatever the
                        # lanes (one at a time, each would be as long: past a loop, the bytes compared before)
                        self.later(self.never, st, [some, self.spec.early(la0, 1)], what, False, stop_end(la0, 1))
                    else:
                        self.later(self.never, st, [S.or_(*[S.and_(lc, self.spec.early(la, ln)) for la, ln, _, lc in new])], what)
                bads = [b for b in (self.spec.early_store(e) for e in st.stores[seen_st:]) if b is not S.FALSE]
                if bads:  # a masked store's lanes: one query for them all
                    one = st.stores[seen_st] if len(st.stores) == seen_st + 1 else None
                    self.later(self.never, st, [S.or_(*bads)], 'the store at %#x lands outside the object' % st.stores[seen_st][2],
                               end=getattr(self.spec, 'store_stop_end', lambda e: None)(one) if one is not None else None)
                if len(succ) == 1:
                    st = succ[0][0]
                    continue
                if top and not st.free and len(succ) == 2 and not any(c is S.TRUE or c is S.FALSE for _, c in succ):
                    s, c = succ[int(self.prefix[st.fork])]
                    s.fork = st.fork + 1
                    s.free = s.fork == len(self.prefix)
                    if self.solver.check(*s.pc, c, loose=True) == 'unsat':  # the prefix's way is dead: no path of this run here
                        dropped = closed = True
                        break
                    s.pc.append(c)
                    s.trace.append(s.rip)
                    st = s
                    continue
                live, dead = [], 0
                for s, c in succ:
                    if c is S.FALSE:
                        continue
                    if c is S.TRUE:
                        live.append(s)
                        continue
                    # a two-way branch whose one way is dead: the other is the path's own (no query; a dead
                    # branch may stay: sound)
                    if (dead and len(succ) == 2) or self.solver.check(*s.pc, c, loose=True) != 'unsat':
                        s.pc.append(c)
                        s.trace.append(s.rip)
                        live.append(s)
                    else:
                        dead += 1
                if not live:  # the ways' conditions cover every case: the path's own facts contradict (a dead way kept)
                    if self.solver.check(*st.pc) != 'unsat':
                        raise Failure('no feasible way on at %#x' % st.rip)
                    closed = True
                    break
                st = live[0]
                work.extend(live[1:])
            if dropped or (top and not st.free and '1' in self.prefix[st.fork:]):  # another run's path (or none)
                continue
            self.paths += 1
            if closed:
                continue
            check_abi(st, self.solver)
            obligations, show = self.spec.obligations(st)
            self.later(self.prove, st, obligations, show)
        self.works.pop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--elf', required=True)
    ap.add_argument('--func', required=True)
    ap.add_argument('--spec', required=True, choices=sorted(SPECS))
    ap.add_argument('--max', type=int, default=256, help='bounded: the bound on the length or size')
    ap.add_argument('--induct', action='store_true', help='unbounded: the loops by induction')
    ap.add_argument('--steps', type=int, default=20000, help='instructions on one path at most')
    ap.add_argument('--log', help='the SMT-LIB sent to z3')
    ap.add_argument('--verbose', action='store_true', help='each obligation and its time')
    ap.add_argument('--align', default='all', help="the pointer's low bits to prove for: all, or a list")
    ap.add_argument('--classes', type=int, help='the number of alignment classes (a power of two), not align_of\'s')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('PROVE_WORKERS', '1')),
                    help='checking threads, each with its own solvers (1: the walk checks as it goes)')
    ap.add_argument('--prefix', default='', help='0s and 1s: the ways this run takes at a path\'s first two-way branches (the runs of every prefix of a length: one proof)')
    ap.add_argument('--setmax', type=int, default=SETMAX, help='strspn, strcspn: the short spec takes the sets of up to this many bytes, '
                    'the long one the longer ones (the same for both: together every set; sse2\'s table scan, one code from 2 bytes on: 1)')
    args = ap.parse_args()
    img = X.Image(args.elf)
    mach = X.Machine(img, bmi=not args.func.startswith(('sse2.', 'sse42.')))
    mach.targets = X.branch_targets(img) if args.induct else set()
    entry = img.symbols[args.func]
    align = args.classes or align_of(img, entry)
    classes = range(align) if args.align == 'all' else [int(x) for x in args.align.split(',')]
    log = open(args.log, 'w') if args.log else None
    t0 = time.time()
    total = [0, 0, 0]
    for a in classes:
        spec = SPECS[args.spec](args.max, args.induct)
        spec.image = img
        if isinstance(spec, Span):
            spec.setmax = args.setmax
        solver = S.Solver(log=log)
        params, assume = spec.setup(a, align)
        for c in assume:
            solver.assume(c)
        if solver.check() != 'sat':
            raise Failure('the assumptions contradict each other: the proof would be empty')
        p = Prover(args, mach, spec, solver, a)
        p.align = align
        spec.solver = Current(p)  # span() and pinned() ask the solver of the thread they run in
        mach.decide = lambda st, bad, solver=solver: solver.check(*st.pc, bad, keep=True) == 'unsat'
        p.run(entry, params)
        queries = solver.queries + getattr(p, 'queries', 0)
        print('s %% %d == %2d: %d paths, %d loops by induction, %d obligations, %d queries (%.1f s)' % (
            align, a, p.paths, p.loops, p.proved, queries, time.time() - t0), flush=True)
        result(args, a, 'proved', p.paths, p.proved, queries, time.time() - p.t0, '')
        total[0] += p.paths
        total[1] += p.proved
        total[2] += queries
        solver.close()
    print('%s: %s proved %s, %d alignment classes: %d paths, %d obligations, %d solver queries, %.1f s' % (
        args.func, args.spec, 'for every length (loops by induction)' if args.induct else 'for lengths up to %d' % args.max, len(classes),
        total[0], total[1], total[2], time.time() - t0))


if __name__ == '__main__':
    main()
