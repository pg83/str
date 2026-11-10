"""x86-64 semantics over smt terms, for the instructions of the library.

A State holds the registers as terms (vectors as lists of byte terms, lane 0
the lowest address), the flags as Bool terms (None where an instruction left
them undefined), memory as an array term, the path condition, and every
load and store with its address and width: the prover's obligations. The
stack lives apart, at known offsets from the entry's rsp (rsp0), where the
library keeps it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import x86  # noqa: E402
import smt as S  # noqa: E402
from smt import const, var  # noqa: E402

GPR = {}
for _i in range(16):
    GPR[x86.R64[_i]] = (x86.R64[_i], 64, 0)
    GPR[x86.R32[_i]] = (x86.R64[_i], 32, 0)
    GPR[x86.R16[_i]] = (x86.R64[_i], 16, 0)
    GPR[x86.R8[_i]] = (x86.R64[_i], 8, 0)
for _i, _n in enumerate(('ah', 'ch', 'dh', 'bh')):
    GPR[_n] = (x86.R64[_i], 8, 8)

RET_MARK = 0xdead0000beef0000
RSP0 = S.concat(var('rsp0_hi', 60), const(4, 8))  # the entry's rsp: 8 past a 16-byte boundary (SysV)
CALLEE_SAVED = ('rbx', 'rbp', 'r12', 'r13', 'r14', 'r15')


class Unsupported(Exception):
    pass


class Fault(Exception):
    """Replay: an access outside the mapped pages (a guard page natively)."""


class State:
    def __init__(self):
        self.rip = 0
        self.gpr = {}
        self.vec = {}
        self.k = {}
        self.flags = {}
        self.mem = None
        self.pc = []
        self.loads = []    # (addr, bytes, rip, cond): cond the lane's mask bit for a masked access, else TRUE
        self.aligned = []  # (addr, alignment, rip)
        self.requires = []  # (bad, what, rip): conditions an instruction's semantics here needs false (rep movs: no backward overlap)
        self.stores = []   # (addr, bytes, rip, cond, lanes): lanes the values; rep stos: bytes a term, lanes [the byte]
        self.stack = {}    # the replay's stack: offset from rsp0 -> byte
        self.regions = []  # (base, array): memory at base + offset as array[offset]
        self.cmem = None   # replay: (address, bytearray), the only memory there is
        self.steps = 0
        self.done = False
        self.trace = []
        self.fresh = 'init_'  # the prefix of the variables a register read before any write gets
        self.snaps = {}       # branch target -> the state at the path's first visit (loop heads, for induction)
        self.gen = None       # the loop this state stands for every iteration of
        self.front_lo = None  # the latest loop frontier: the bytes before it are its invariant's
        self.front_bytes = []  # frontiers whose byte must be in the object (loads invariant A)
        self.cover = []       # writers: (lo, hi, cond) index ranges of the object written right before the stores since
        self.win = {}         # region -> (lo, hi, array): a loop's general memory, the array but in [lo, hi) (offsets)
        self.aliases = []     # (pointer, region, offset): addresses from the pointer are in the region at offset + their distance from it
        self.fork = 0         # --prefix: the decisions this path took at the first two-way branches
        self.free = False     # past them (a loop's head, a contract, the prefix used up): every way its own

    def copy(self):
        s = State.__new__(State)
        s.rip, s.mem, s.steps, s.done = self.rip, self.mem, self.steps, self.done
        s.gpr, s.vec, s.k, s.flags, s.stack = dict(self.gpr), dict(self.vec), dict(self.k), dict(self.flags), dict(self.stack)
        s.regions = list(self.regions)
        s.cmem = None if self.cmem is None else (self.cmem[0], bytearray(self.cmem[1]))
        s.fresh, s.snaps, s.gen = self.fresh, dict(self.snaps), self.gen
        s.front_lo, s.front_bytes = self.front_lo, list(self.front_bytes)
        s.cover = list(self.cover)
        s.win = self.win
        s.aliases = self.aliases
        s.pc, s.loads, s.aligned, s.stores, s.trace = list(self.pc), list(self.loads), list(self.aligned), list(self.stores), list(self.trace)
        s.requires = list(self.requires)
        s.fork, s.free = self.fork, self.free
        for k, v in self.__dict__.items():  # what a spec keeps on the path (its callees' answers): the copy's too
            if k not in s.__dict__:
                s.__dict__[k] = v
        return s


def fresh_state(entry, args, regions=(), aliases=()):
    """The state at a function's entry: args are terms for rdi, rsi, rdx, rcx,
    r8, r9 in order; every other register a variable of its own."""
    st = State()
    st.rip = entry
    for r in x86.R64:
        st.gpr[r] = var('init_' + r, 64)
    for r, t in zip(('rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9'), args):
        st.gpr[r] = t
    st.gpr['rsp'] = RSP0
    st.stack = {i: const(8, RET_MARK >> 8 * i & 0xff) for i in range(8)}  # the replay's stack (concrete runs)
    st.mem = S.mem_var('M')
    stack = S.mem_var('stack')
    for i in range(8):
        stack = S.store(stack, const(64, i), const(8, RET_MARK >> 8 * i & 0xff))
    st.regions = list(regions) + [(RSP0, stack)]  # the stack: bytes like any memory, the return address at rsp0
    st.aliases = list(aliases)
    return st


def on_stack(st, addr):
    """Whether addr is in the stack's region (the last one)."""
    if not st.regions or st.regions[-1][0] is not RSP0:
        return False
    r = region_of(st, addr)
    return r is not None and r[0] == len(st.regions) - 1


def push64(st, v):
    rsp = S.sub(st.gpr['rsp'], const(64, 8))
    if st.cmem is not None:
        off = stack_offset(st, rsp)
        if off is None:
            raise Unsupported('push with an unknown stack')
        for i in range(8):
            st.stack[off + i] = S.extract(v, 8 * i + 7, 8 * i)
    else:
        if not on_stack(st, rsp):
            raise Unsupported('push off the stack')
        st.requires.append((S.sle(const(64, 0), S.sub(S.add(rsp, const(64, 7)), RSP0)), 'a push at or past the return address', st.rip))
        for i in range(8):
            write_byte(st, S.add(rsp, const(64, i)), S.extract(v, 8 * i + 7, 8 * i))
    st.gpr['rsp'] = rsp


def pop64(st):
    rsp = st.gpr['rsp']
    if st.cmem is not None:
        off = stack_offset(st, rsp)
        if off is None or any(off + i not in st.stack for i in range(8)):
            raise Unsupported('pop of an unknown stack slot')
        v = to_bv([st.stack[off + i] for i in range(8)])
    else:
        if not on_stack(st, rsp):
            raise Unsupported('pop off the stack')
        v = to_bv([byte(st, S.add(rsp, const(64, i))) for i in range(8)])
    st.gpr['rsp'] = S.add(rsp, const(64, 8))
    return v


# ------------------------------------------------------------- registers

def vec_index(name):
    for p, n in (('xmm', 16), ('ymm', 32), ('zmm', 64)):
        if name.startswith(p):
            return int(name[3:]), n
    return None


def read_reg(st, name):
    if name in GPR:
        full, w, lo = GPR[name]
        return S.extract(st.gpr[full], lo + w - 1, lo)
    if name.startswith('k') and name[1:].isdigit():
        if name not in st.k:
            st.k[name] = var(st.fresh + name, 64)
        return st.k[name]
    raise Unsupported('register %s' % name)


def write_reg(st, name, v):
    full, w, lo = GPR[name]
    assert v.w == w, (name, v)
    if w == 64:
        st.gpr[full] = v
    elif w == 32:
        st.gpr[full] = S.zext(v, 64)
    else:
        old = st.gpr[full]
        parts = []
        if lo + w < 64:
            parts.append(S.extract(old, 63, lo + w))
        parts.append(v)
        if lo:
            parts.append(S.extract(old, lo - 1, 0))
        st.gpr[full] = S.concat(*parts)


def read_vec(st, name):
    n, width = vec_index(name)
    lanes = st.vec.get(n)
    if lanes is None:
        lanes = [var('%szmm%d_%d' % (st.fresh, n, i), 8) for i in range(64)]
        st.vec[n] = lanes
    return lanes[:width]


def write_vec(st, name, lanes, legacy=False):
    """VEX and EVEX writes zero the lanes above the operand; legacy SSE keeps them."""
    n, width = vec_index(name)
    assert len(lanes) == width
    if legacy:
        old = read_vec(st, 'zmm%d' % n)
        st.vec[n] = list(lanes) + old[width:]
    else:
        st.vec[n] = list(lanes) + [const(8, 0)] * (64 - width)


def write_k(st, name, v, width):
    """A mask register written with width bits: the rest zeroed."""
    st.k[name] = S.zext(v, 64) if v.w < 64 else v


def to_bv(lanes):
    """Byte lanes as one vector, lane 0 least significant."""
    return S.concat(*reversed(lanes))


def bits(bools):
    """Bools as a bit-vector, the first least significant."""
    return S.concat(*[S.bit(b) for b in reversed(bools)])


# ------------------------------------------------------------- memory

class Image:
    """The linked library: its sections by address, its symbols."""

    def __init__(self, path):
        import elf
        e = elf.read(path, executable=True)
        self.sections = [s for s in e.sections if s.addr and s.type in ('SHT_PROGBITS', 'SHT_NOBITS')]
        self.symbols = {s.name: s.value for s in e.symbols if s.name and s.type in ('STT_FUNC', 'STT_OBJECT', 'STT_NOTYPE')}
        self.sizes = {s.name: s.size for s in e.symbols if s.name and s.type == 'STT_FUNC'}
        self.names = {}
        for s in e.symbols:
            if s.name and s.type in ('STT_FUNC', 'STT_NOTYPE') and s.value:
                self.names.setdefault(s.value, s.name)
        self.text = next(s for s in self.sections if s.name == '.text')
        self.cache = {}
        self.overrides = {}  # address -> byte: the table a configuration installs

    def section_at(self, a):
        for s in self.sections:
            if s.addr <= a < s.addr + s.size:
                return s
        return None

    def byte(self, a):
        if a in self.overrides:
            return self.overrides[a]
        s = self.section_at(a)
        return 0 if s.type == 'SHT_NOBITS' else s.data[a - s.addr]

    def decode(self, a):
        j = self.cache.get(a)
        if j is None:
            t = self.text
            if not t.addr <= a < t.addr + t.size:
                raise Unsupported('jump outside .text: %#x' % a)
            j = x86.decode_one(t.data, t.addr, a - t.addr)
            self.cache[a] = j
        return j


def addends(t):
    if t.op == 'add':
        return addends(t.args[0]) + addends(t.args[1])
    return [t]


def offset_from(addr, base):
    """addr - base when addr is a sum with base as one of its terms, else None."""
    terms = addends(addr)
    if base not in terms:
        return None
    terms.remove(base)
    off = const(64, 0)
    for x in terms:
        off = S.add(off, x)
    return off


def rounded(addr):
    """The pointers addr rounds down, each with the rest of the sum: (x, rest)
    for addr = (x & mask) + rest."""
    f, c = S.linear(addr)
    out = []
    for x, k in f.items():
        if k == 1 and x.op == 'and' and S.is_const(x.args[1]):
            out.append((x.args[0], S.from_linear(addr.w, {y: v for y, v in f.items() if y is not x}, c)))
    return out


def region_of(st, addr):
    """(index, offset) of addr's region, or None. An address that is no sum
    with a base in it (rounded down, say) belongs to the one region whose
    base it depends on, at addr - base."""
    for k, (base, arr) in enumerate(st.regions):
        off = offset_from(addr, base)
        if off is not None:
            return k, off
    for p, k, poff in st.aliases:  # another pointer into a region (memmove's source: d's memory at s - d's base)
        off = offset_from(addr, p)
        if off is not None:
            return k, S.add(poff, off)
    for x, rest in rounded(addr):  # a pointer rounded down, plus an offset (no other region's base added in): its region
        r = region_of(st, x)
        if r is not None and not any(offset_from(rest, b) is not None for k, (b, _) in enumerate(st.regions) if k != r[0]):
            return r[0], S.sub(addr, st.regions[r[0]][0])
    vs = S.variables(addr)
    hits = [k for k, (base, arr) in enumerate(st.regions) if S.variables(base) & vs]
    if len(hits) == 1:
        k = hits[0]
        return k, S.sub(addr, st.regions[k][0])
    if hits or (st.regions and not S.is_const(addr) and st.cmem is None):
        raise Unsupported('an address of no single region: %r' % addr)
    return None


def byte(st, addr):
    """The byte at addr: in a region, an element of the region's array at the
    offset from its base (the solver then compares offsets, not addresses)."""
    r = region_of(st, addr)
    if r is not None:
        return S.select(st.regions[r[0]][1], r[1])
    return S.select(st.mem, addr)


def write_byte(st, addr, v):
    r = region_of(st, addr)
    if r is not None:
        k, off = r
        st.regions[k] = (st.regions[k][0], S.store(st.regions[k][1], off, v))
    else:
        st.mem = S.store(st.mem, addr, v)


def call_return(st, rax, prefix):
    """The state after a call whose effect the caller's proof takes from
    the callee's contract: the caller-saved registers, vectors, masks and
    flags unknown (a new prefix for their variables), rax the answer, the
    return address popped."""
    st.fresh = prefix
    for r in ('rcx', 'rdx', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11'):
        st.gpr[r] = var(prefix + r, 64)
    st.gpr['rax'] = rax
    st.vec, st.k, st.flags = {}, {}, {}
    target = pop64(st)
    if not S.is_const(target):
        raise Unsupported('a contract\'s return to a computed address')
    st.rip = target.val


def slot_guess(st, addr):
    """The 8 bytes at addr read past every store at another or a computed
    place and every window: a constant to check, or None."""
    out = []
    for i in range(8):
        r = region_of(st, S.add(addr, const(64, i)))
        if r is None:
            return None
        arr, off, v = st.regions[r[0]][1], r[1], None
        while arr.op in ('store', 'win', 'fill'):
            if arr.op == 'store' and arr.args[1] is off:
                v = arr.args[2]
                break
            arr = arr.args[0]
        if v is None or not S.is_const(v):
            return None
        out.append(v)
    return to_bv(out)


def stack_offset(st, addr):
    """addr as rsp0 + offset, or None (the replay's stack)."""
    off = offset_from(addr, RSP0)
    return S.signed(off) if off is not None and S.is_const(off) else None


def load(m, st, addr, n, rip, conds=None):
    """n bytes at addr as lanes; conds: a masked load's lane conditions (a
    lane masked off is not read: no fault, no obligation)."""
    off = stack_offset(st, addr) if st.cmem is not None else None
    if off is not None:
        if any(off + i not in st.stack for i in range(n)):
            raise Fault('a load of the stack not written at rsp0%+d' % off)
        return [st.stack[off + i] for i in range(n)]
    sec = m.image.section_at(addr.val) if S.is_const(addr) else None
    if sec is not None and not sec.flags & 1:  # the image's constants; a writable one (strtok's state) is memory like any other
        return [const(8, m.image.byte(addr.val + i)) for i in range(n)]
    if st.cmem is not None:
        base, buf = st.cmem
        if not S.is_const(addr) or not base <= addr.val <= addr.val + n <= base + len(buf):
            raise Fault('load of %d bytes at %r from %#x' % (n, addr, rip))
        o = addr.val - base
        return [const(8, b) for b in buf[o:o + n]]
    if on_stack(st, addr):  # the function's own frame: no obligation
        return [byte(st, S.add(addr, const(64, i))) for i in range(n)]
    if conds is None:
        st.loads.append((addr, n, rip, S.TRUE))
    else:
        st.loads.extend((S.add(addr, const(64, i)), 1, rip, c) for i, c in enumerate(conds) if c is not S.FALSE)
    if st.win or any(st.regions[k][1].op == 'store' for k in range(len(st.regions) - 1)):
        lanes = window_load(m, st, addr, n)
        if lanes is not None:
            return lanes
    return [byte(st, S.add(addr, const(64, i))) for i in range(n)]


def within(x, o, n):
    """x one of the n offsets from o (modulo 2^64, as the memory has them; n a
    number or a term)."""
    return S.ult(S.sub(x, o), n if isinstance(n, S.T) else const(64, n))


def window_load(m, st, addr, n):
    """A load clear of the stores since the region's memory was last set (at
    the entry, or at a loop's general state; and clear of that state's
    window): the bytes of the memory under them, one query instead of a
    case split in every lane (memmove's source read after stores that may
    overlap it). Deeper first: clear of the stores and windows under that
    too (an earlier loop's, the memory at the entry under them all), then
    the memory under this state's own. j: any byte of the load."""
    r = region_of(st, addr)
    if r is None or m.decide is None:
        return None
    k, off = r
    j = var('jw', 64)
    x = S.add(off, j)
    if k in st.win:
        lo, hi, arr = st.win[k]
        hit = [S.and_(S.ule(lo, x), S.ult(x, hi))]
    else:
        arr = st.regions[k][1]
        while arr.op == 'store':  # the memory before the stores since
            arr = arr.args[0]
        if arr is st.regions[k][1]:
            return None
        hit = []
    for a, nb, _, c, _ in st.stores:
        rs = region_of(st, a)
        if rs is not None and rs[0] == k:
            hit.append(S.and_(c, within(x, rs[1], nb)))
    ways, deep, more = [(arr, hit)], arr, list(hit)
    while deep.op in ('store', 'win', 'fill'):
        if deep.op == 'store':
            more.append(S.eq(deep.args[1], x))
        else:
            more.append(S.and_(S.ule(deep.args[1], x), S.ult(x, deep.args[2])))
        deep = deep.args[0]
    if deep is not arr:
        ways.insert(0, (deep, more))
    for arr, hit in ways:
        if m.decide(st, S.and_(S.ult(j, const(64, n)), S.or_(*hit))):
            return [S.select(arr, S.add(off, const(64, i))) for i in range(n)]
    return None


def store_bytes(m, st, addr, lanes, rip, conds=None):
    off = stack_offset(st, addr) if st.cmem is not None else None
    if off is not None:
        for i, b in enumerate(lanes):
            if conds is None or conds[i] is S.TRUE:
                st.stack[off + i] = b
            elif conds[i] is not S.FALSE:
                raise Fault('a masked store to the stack with an unknown mask')
        return
    if st.cmem is not None:
        base, buf = st.cmem
        for i, b in enumerate(lanes):
            c = S.TRUE if conds is None else conds[i]
            if c is S.FALSE:
                continue
            a = S.add(addr, const(64, i))
            if c is not S.TRUE or not S.is_const(b) or not S.is_const(a):
                raise Fault('a store of an unknown byte or mask or address at %#x' % rip)
            if not base <= a.val < base + len(buf):
                raise Fault('store at %#x from %#x' % (a.val, rip))
            buf[a.val - base] = b.val
        return
    if on_stack(st, addr):  # the function's own frame: written, nothing past it (the return address, the caller's frame)
        last = S.sub(S.add(addr, const(64, len(lanes) - 1)), RSP0)
        st.requires.append((S.sle(const(64, 0), last), 'a store to the stack at or past the return address', rip))
        for i, b in enumerate(lanes):
            a = S.add(addr, const(64, i))
            c = S.TRUE if conds is None else conds[i]
            if c is not S.FALSE:
                write_byte(st, a, b if c is S.TRUE else S.ite(c, b, byte(st, a)))
        return
    if conds is None:
        st.stores.append((addr, len(lanes), rip, S.TRUE, list(lanes)))
        for i, b in enumerate(lanes):
            write_byte(st, S.add(addr, const(64, i)), b)
        return
    for i, (b, c) in enumerate(zip(lanes, conds)):
        if c is S.FALSE:
            continue
        a = S.add(addr, const(64, i))
        st.stores.append((a, 1, rip, c, [b]))
        write_byte(st, a, S.ite(c, b, byte(st, a)))


def copy_bytes(m, st, dst, src, count, rip):
    """rep movsb forward: count (a term) bytes from src to dst. Within one
    region too, where the destination must not start inside the source
    (byte by byte forward, it would read bytes it wrote): then every byte
    written is the source's as it was."""
    if st.cmem is not None:
        base, buf = st.cmem
        if not all(S.is_const(x) for x in (dst, src, count)):
            raise Fault('rep movs with unknown operands at %#x' % rip)
        for i in range(count.val):  # byte by byte, forward, as the instruction does
            s, d = src.val + i, dst.val + i
            if not (base <= s < base + len(buf) and base <= d < base + len(buf)):
                raise Fault('rep movs out of the pages at %#x' % rip)
            buf[d - base] = buf[s - base]
        return
    rd, rs = region_of(st, dst), region_of(st, src)
    if rd is None or rs is None:
        raise Unsupported('rep movs outside the regions at %#x' % rip)
    if rd[0] == rs[0]:
        st.requires.append((S.and_(S.not_(S.is_zero(count)), S.ult(src, dst), S.ult(dst, S.add(src, count))), 'rep movs with its destination inside its source', rip))
    (kd, dlo), (ks, slo) = rd, rs
    old, srcarr = st.regions[kd][1], st.regions[ks][1]
    st.loads.append((src, count, rip, S.TRUE))
    st.stores.append((dst, count, rip, S.TRUE, ('movs', src, srcarr, slo)))  # the source's memory as the copy found it
    x = var('xm%d' % len(st.stores), 64)
    body = S.ite(S.and_(S.ule(dlo, x), S.ult(x, S.add(dlo, count))), S.select(srcarr, S.add(S.sub(x, dlo), slo)), S.select(old, x))
    st.regions[kd] = (st.regions[kd][0], S.lam(x, body))


def fill_bytes(m, st, addr, count, v, rip):
    """rep stosb: count (a term) bytes v from addr."""
    if st.cmem is not None:
        base, buf = st.cmem
        if not (S.is_const(addr) and S.is_const(count) and S.is_const(v)) or not base <= addr.val <= addr.val + count.val <= base + len(buf):
            raise Fault('rep stos out of the pages or unknown at %#x' % rip)
        buf[addr.val - base:addr.val - base + count.val] = bytes([v.val]) * count.val
        return
    st.stores.append((addr, count, rip, S.TRUE, [v]))
    r = region_of(st, addr)
    if r is None:
        raise Unsupported('rep stos outside a region at %#x' % rip)
    k, off = r
    st.regions[k] = (st.regions[k][0], S.fill(st.regions[k][1], off, S.add(off, count), v))


def mem_addr(st, j, mem):
    if mem['rip_rel']:
        return const(64, j['offset'] + j['len'] + mem['disp'])
    a = const(64, mem['disp'])
    if mem['base'] not in (None, 'none'):
        a = S.add(read_reg(st, mem['base']), a)
    if mem['index'] not in (None, 'none'):
        a = S.add(a, S.mul(read_reg(st, mem['index']), const(64, mem['scale'])))
    return a


# ------------------------------------------------------------- operands

def opw(o):
    """An operand's width in bits."""
    if o['kind'] == 'Memory':
        return o['mem']['size'] * 8
    return o['size']


def op_value(m, st, j, o, w):
    """A general-purpose operand as a w-bit term."""
    if o['kind'] == 'Register':
        return read_reg(st, o['reg'])
    if o['kind'] == 'Immediate':
        v = const(o['size'], o['imm'])
        return S.sext(v, w) if o.get('signed') else S.zext(v, w)
    if o['kind'] == 'Memory':
        return to_bv(load(m, st, mem_addr(st, j, o['mem']), w // 8, j['offset']))
    raise Unsupported(o['kind'])


def op_store(m, st, j, o, v):
    if o['kind'] == 'Register':
        write_reg(st, o['reg'], v)
    elif o['kind'] == 'Memory':
        store_bytes(m, st, mem_addr(st, j, o['mem']), [S.extract(v, 8 * i + 7, 8 * i) for i in range(v.w // 8)], j['offset'])
    else:
        raise Unsupported(o['kind'])


def vec_operand(m, st, j, o, nbytes, aligned=False):
    if o['kind'] == 'Register':
        if o['reg'] in GPR:
            raise Unsupported('gpr as vector')
        return read_vec(st, o['reg'])[:nbytes]
    addr = mem_addr(st, j, o['mem'])
    if aligned:
        st.aligned.append((addr, nbytes, j['offset']))
    return load(m, st, addr, nbytes, j['offset'])


# ------------------------------------------------------------- flags

def flags_logic(st, r):
    st.flags = {'CF': S.FALSE, 'OF': S.FALSE, 'ZF': S.is_zero(r), 'SF': S.msb(r), 'PF': None}


def flags_add(st, a, b, r):
    st.flags = {'CF': S.ult(r, a), 'ZF': S.is_zero(r), 'SF': S.msb(r), 'OF': S.and_(S.eq(S.msb(a), S.msb(b)), S.not_(S.eq(S.msb(r), S.msb(a)))),
                'PF': None}


def flags_sub(st, a, b, r):
    st.flags = {'CF': S.ult(a, b), 'ZF': S.eq(a, b), 'SF': S.msb(r), 'OF': S.and_(S.not_(S.eq(S.msb(a), S.msb(b))), S.not_(S.eq(S.msb(r), S.msb(a)))),
                'PF': None}


def flag(st, name):
    f = st.flags.get(name)
    if f is None:
        raise Unsupported('flag %s undefined here' % name)
    return f


def condition(st, cc):
    f = lambda n: flag(st, n)
    return {
        'o': lambda: f('OF'), 'no': lambda: S.not_(f('OF')), 'b': lambda: f('CF'), 'ae': lambda: S.not_(f('CF')), 'e': lambda: f('ZF'),
        'ne': lambda: S.not_(f('ZF')), 'be': lambda: S.or_(f('CF'), f('ZF')), 'a': lambda: S.and_(S.not_(f('CF')), S.not_(f('ZF'))),
        's': lambda: f('SF'), 'ns': lambda: S.not_(f('SF')), 'l': lambda: S.not_(S.eq(S.bit(f('SF')), S.bit(f('OF')))),
        'ge': lambda: S.eq(S.bit(f('SF')), S.bit(f('OF'))),
        'le': lambda: S.or_(f('ZF'), S.not_(S.eq(S.bit(f('SF')), S.bit(f('OF'))))),
        'g': lambda: S.and_(S.not_(f('ZF')), S.eq(S.bit(f('SF')), S.bit(f('OF')))),
    }[cc]()


def tzcnt(x):
    r = const(x.w, x.w)
    for i in reversed(range(x.w)):
        r = S.ite(S.eq(S.extract(x, i, i), const(1, 1)), const(x.w, i), r)
    return r


# ------------------------------------------------------------- execution

CC = set(x86.CC)
MASKED = {'vmovdqu8', 'vmovdqu64', 'vmovdqu32', 'vmovdqa64', 'vmovdqa32', 'vpcmpb', 'vpcmpub', 'vpcmpeqb', 'vptestnmb', 'vptestmb'}


def consecutive(lanes):
    """(array, offset) when lane c is the array's byte at offset + c for
    every c (a block loaded from one region), else None: a shuffle by an
    index the code computes then reads its byte at offset + index, one
    select, the same byte as the 16-way choice among the lanes."""
    x = lanes[0]
    if x.op != 'select':
        return None
    arr, off = x.args
    for c, y in enumerate(lanes[1:], 1):
        if y.op != 'select' or y.args[0] is not arr or S.sub(y.args[1], off) is not const(64, c):
            return None
    return arr, off


def shuffle_index(x):
    """A pshufb index lane as (guard, k): the lane is 0 where guard, else the
    half's byte k. The page-safe load's index a | (a > 15 ? 0xff : 0)
    (pcmpgtb and por: the lanes past the block's end made negative) is 0
    exactly where a is not in [0, 15] (a negative a has the top bit, one
    past 15 the mask's): guard a > 15 unsigned, k = a itself there."""
    if x.op == 'or':
        for a, m in (x.args, x.args[::-1]):
            if m.op == 'ite' and m.args[1] is const(8, 0xff) and m.args[2] is const(8, 0) and m.args[0].op == 'slt' and \
                    m.args[0].args[0] is const(8, 15) and m.args[0].args[1] is a:
                return S.ult(const(8, 15), a), a
    return S.msb(x), S.band(x, const(8, 15))


def low_of(t):
    """(p, j): t is p's low j bits, zero above (p & 15 through the 32-bit
    and of its low half and an extract to a byte), else None."""
    if t.op == 'and' and S.is_const(t.args[1]):
        m = t.args[1].val
        j = m.bit_length()
        if j and m == S.mask(j):
            y = t.args[0]
            while y.op == 'extract' and y.args[2] == 0 and y.args[1] + 1 >= j:
                y = y.args[0]
            return y, j
    if t.op == 'extract' and t.args[2] == 0:
        r = low_of(t.args[0])
        if r is not None and r[1] <= t.w:
            return r
    return None


def index64(k):
    """zext(k) to 64 bits for a shuffle's byte index k in [0, 15]: a pointer's
    low bits plus a constant (the page-safe load's iota + (p & 15)) as
    (p & 15) + the constant, both below 16, so no carry; the block's
    address plus it then folds to p plus the constant (merge_masks)."""
    f, c = S.linear(k)
    if len(f) == 1:
        (x, v), = f.items()
        r = low_of(x)
        if v == 1 and r is not None and r[1] <= 4 and c + S.mask(r[1]) <= S.mask(k.w):
            p, n = r
            p64 = p if p.w == 64 else S.zext(p, 64) if p.w < 64 else S.extract(p, 63, 0)
            return S.add(S.band(p64, const(64, S.mask(n))), const(64, c))
    return S.zext(k, 64)


def merge_masks(t):
    """A sum with p & m1 and p & m2 at the same coefficient, m1 and m2
    disjoint, as one p & (m1 | m2), p itself when that is all of p's bits:
    an aligned block's address plus the pointer's low bits is the pointer."""
    f, c = S.linear(t)
    groups = {}
    for x, v in f.items():
        if x.op == 'and' and S.is_const(x.args[1]):
            groups.setdefault((x.args[0].id, v), []).append(x)
    f, changed = dict(f), False
    for (_, v), xs in groups.items():
        masks = [x.args[1].val for x in xs]
        if len(xs) < 2 or any(masks[i] & masks[j] for i in range(len(xs)) for j in range(i)):
            continue
        union = 0
        for m in masks:
            union |= m
        p = xs[0].args[0]
        for x in xs:
            del f[x]
        y = p if union == S.mask(t.w) else S.band(p, const(t.w, union))
        f[y] = (f.get(y, 0) + v) & S.mask(t.w)
        if not f[y]:
            del f[y]
        changed = True
    return S.from_linear(t.w, f, c) if changed else t


def zero_or(t):
    """(g, y) for a lane ite(g, 0, y) (a shuffle's lane past its block), else (FALSE, t)."""
    if t.op == 'ite' and t.args[1] is const(t.w, 0):
        return t.args[0], t.args[2]
    return S.FALSE, t


def lane_conds(st, j, n):
    """A masked instruction's lane conditions, or None unmasked."""
    k = j.get('op_mask')
    if k in (None, 'none'):
        return None
    km = read_reg(st, k)
    return [S.eq(S.extract(km, i, i), const(1, 1)) for i in range(n)]


def backward_targets(image, entry):
    """The targets of the backward branches within a function in the code
    reachable from entry: every cycle of a linear layout has one, so a path
    meets one of them before it goes round any loop."""
    spans = sorted((a, a + image.sizes.get(n, 0)) for n, a in image.symbols.items() if image.sizes.get(n))
    def inside(a):
        lo = max((s for s in spans if s[0] <= a), default=None)
        return lo if lo is not None and a < lo[1] else None
    out, seen, todo = set(), set(), [entry]
    while todo:
        a = todo.pop()
        while a not in seen:
            seen.add(a)
            j = image.decode(a)
            if j['mnemonic'] == 'int3':
                break
            t = j.get('branch_target')
            if t is not None:
                todo.append(t)
                f = inside(a)
                if t <= a and f is not None and f[0] <= t:
                    out.add(t)
            if j['mnemonic'] in ('ret', 'jmp'):
                break
            a += j['len']
    return out


def branch_targets(image):
    """Every direct branch target in the code: a linear sweep of each function."""
    out = set()
    t = image.text
    for name, start in image.symbols.items():
        size = image.sizes.get(name, 0)
        if not (t.addr <= start < t.addr + t.size) or not size:
            continue
        a = start
        while a < start + size:
            try:
                j = image.decode(a)
            except Exception:
                a += 1
                continue
            if j.get('branch_target') is not None:
                out.add(j['branch_target'])
            a += j['len']
    return out


class Machine:
    def __init__(self, image, bmi=True):
        self.image = image
        self.bmi = bmi  # False: tzcnt runs as bsf (a CPU without BMI1, the sse2 and sse42 configurations' targets)
        self.fresh_n = 0
        self.decide = None  # (state, bad) -> whether the path's conditions rule bad out: the prover's solver

    def step(self, st):
        """Execute one instruction: [(state, condition)] successors, where a
        conditional branch gives two with their conditions (TRUE otherwise)."""
        j = self.image.decode(st.rip)
        st.steps += 1
        mn = j['mnemonic']
        ops = j['ops']
        nxt = j['offset'] + j['len']
        if j.get('op_mask') not in (None, 'none') and mn not in MASKED:
            raise Unsupported('%s with a mask at %#x' % (mn, st.rip))
        if mn.startswith('nop') or mn in ('endbr64',):
            st.rip = nxt
            return [(st, S.TRUE)]
        if mn == 'ret':
            target = pop64(st)
            if S.is_const(target) and target.val == RET_MARK:
                st.done = True
                return [(st, S.TRUE)]
            if not S.is_const(target) and self.decide is not None:  # stores at computed places since (a frame aligned down): the slot's bytes past them, if they are all
                guess = slot_guess(st, S.sub(st.gpr['rsp'], const(64, 8)))
                if guess is not None and self.decide(st, S.not_(S.eq(target, guess))):
                    target = guess
            if not S.is_const(target):
                raise Unsupported('ret to a computed address')
            if target.val == RET_MARK:
                st.done = True
                return [(st, S.TRUE)]
            st.rip = target.val
            return [(st, S.TRUE)]
        if mn == 'jmp':
            o = ops[0]
            if o['kind'] == 'NearBranch':
                st.rip = j['branch_target']
                return [(st, S.TRUE)]
            t = op_value(self, st, j, o, 64)
            if not S.is_const(t):
                raise Unsupported('indirect jump to a computed address at %#x' % st.rip)
            st.rip = t.val
            return [(st, S.TRUE)]
        if mn == 'call':
            o = ops[0]
            target = j['branch_target'] if o['kind'] == 'NearBranch' else None
            if target is None:
                t = op_value(self, st, j, o, 64)
                if not S.is_const(t):
                    raise Unsupported('indirect call to a computed address')
                target = t.val
            push64(st, const(64, nxt))
            st.rip = target
            return [(st, S.TRUE)]
        if mn.startswith('j') and mn[1:] in CC:
            c = condition(st, mn[1:])
            taken, fall = st, st.copy()
            taken.rip, fall.rip = j['branch_target'], nxt
            return [(taken, c), (fall, S.not_(c))]
        f = getattr(self, 'i_' + mn, None)
        if f is None:
            raise Unsupported('%s (%s) at %#x' % (mn, j['gas'], st.rip))
        f(st, j, ops)
        st.rip = nxt
        return [(st, S.TRUE)]

    # --- general purpose
    def i_mov(self, st, j, ops):
        w = ops[0]['size'] if ops[0]['kind'] == 'Register' else ops[0]['mem']['size'] * 8
        op_store(self, st, j, ops[0], op_value(self, st, j, ops[1], w))

    def i_inc(self, st, j, ops):
        w = opw(ops[0])
        a = op_value(self, st, j, ops[0], w)
        r = S.add(a, const(w, 1))
        cf = st.flags.get('CF')
        flags_add(st, a, const(w, 1), r)
        st.flags['CF'] = cf
        op_store(self, st, j, ops[0], r)

    def i_dec(self, st, j, ops):
        w = opw(ops[0])
        a = op_value(self, st, j, ops[0], w)
        r = S.sub(a, const(w, 1))
        cf = st.flags.get('CF')
        flags_sub(st, a, const(w, 1), r)
        st.flags['CF'] = cf
        op_store(self, st, j, ops[0], r)

    def i_bzhi(self, st, j, ops):
        w = opw(ops[0])
        x = op_value(self, st, j, ops[1], w)
        idx = S.zext(S.extract(op_value(self, st, j, ops[2], w), 7, 0), w)
        big = S.ule(const(w, w), idx)
        r = S.ite(big, x, S.band(x, S.sub(S.shl(const(w, 1), idx), const(w, 1))))
        st.flags = {'ZF': S.is_zero(r), 'SF': S.msb(r), 'CF': S.ult(const(w, w - 1), idx), 'OF': S.FALSE, 'PF': None}
        write_reg(st, ops[0]['reg'], r)

    def _bt(self, st, j, ops, fn):
        """bt/bts/btr/btc on a register: the bit (offset mod the width) into
        CF, then the bit set, cleared or flipped."""
        if ops[0]['kind'] != 'Register':
            raise Unsupported('%s on memory at %#x' % (j['mnemonic'], st.rip))
        w = opw(ops[0])
        x = op_value(self, st, j, ops[0], w)
        idx = S.band(op_value(self, st, j, ops[1], w) if ops[1]['kind'] != 'Immediate' else const(w, ops[1]['imm']), const(w, w - 1))
        bit = S.shl(const(w, 1), idx)
        st.flags = dict(st.flags, CF=S.not_(S.is_zero(S.band(x, bit))))
        if fn is not None:
            write_reg(st, ops[0]['reg'], fn(x, bit))

    def i_bt(self, st, j, ops):
        self._bt(st, j, ops, None)

    def i_bts(self, st, j, ops):
        self._bt(st, j, ops, S.bor)

    def i_btr(self, st, j, ops):
        self._bt(st, j, ops, lambda x, b: S.band(x, S.bnot(b)))

    def i_btc(self, st, j, ops):
        self._bt(st, j, ops, S.bxor)

    def i_push(self, st, j, ops):
        push64(st, op_value(self, st, j, ops[0], 64))

    def i_pop(self, st, j, ops):
        write_reg(st, ops[0]['reg'], pop64(st))

    def i_andn(self, st, j, ops):
        """andn r1, r2, r3: r1 = ~r2 & r3."""
        w = opw(ops[0])
        r = S.band(S.bnot(op_value(self, st, j, ops[1], w)), op_value(self, st, j, ops[2], w))
        st.flags = {'ZF': S.is_zero(r), 'SF': S.msb(r), 'CF': S.FALSE, 'OF': S.FALSE, 'PF': None}
        write_reg(st, ops[0]['reg'], r)

    def _bls(self, st, j, ops, fn, cf):
        w = opw(ops[0])
        x = op_value(self, st, j, ops[1], w)
        r = fn(x, S.sub(x, const(w, 1)))
        st.flags = {'ZF': S.is_zero(r), 'SF': S.msb(r), 'CF': cf(x), 'OF': S.FALSE, 'PF': None}
        write_reg(st, ops[0]['reg'], r)

    def i_blsr(self, st, j, ops):  # x & (x - 1): the lowest set bit cleared
        self._bls(st, j, ops, S.band, S.is_zero)

    def i_blsmsk(self, st, j, ops):  # x ^ (x - 1): up to the lowest set bit
        self._bls(st, j, ops, S.bxor, S.is_zero)

    def i_blsi(self, st, j, ops):  # x & -x: the lowest set bit
        self._bls(st, j, ops, lambda x, xm1: S.band(x, S.neg(x)), lambda x: S.not_(S.is_zero(x)))

    def _movs(self, st, j, ops):
        if j.get('rep') != 'rep':
            raise Unsupported('movsb without rep at %#x' % st.rip)
        rdi, rsi, rcx = st.gpr['rdi'], st.gpr['rsi'], st.gpr['rcx']
        copy_bytes(self, st, rdi, rsi, rcx, st.rip)
        st.gpr['rdi'] = S.add(rdi, rcx)
        st.gpr['rsi'] = S.add(rsi, rcx)
        st.gpr['rcx'] = const(64, 0)

    def i_stosb(self, st, j, ops):
        if j.get('rep') != 'rep':
            raise Unsupported('stosb without rep at %#x' % st.rip)
        rdi, rcx = st.gpr['rdi'], st.gpr['rcx']
        fill_bytes(self, st, rdi, rcx, S.extract(st.gpr['rax'], 7, 0), st.rip)
        st.gpr['rdi'] = S.add(rdi, rcx)
        st.gpr['rcx'] = const(64, 0)

    def i_neg(self, st, j, ops):
        w = opw(ops[0])
        a = op_value(self, st, j, ops[0], w)
        r = S.neg(a)
        st.flags = {'CF': S.not_(S.is_zero(a)), 'ZF': S.is_zero(r), 'SF': S.msb(r), 'OF': S.eq(a, const(w, 1 << (w - 1))), 'PF': None}
        op_store(self, st, j, ops[0], r)

    def i_not(self, st, j, ops):
        w = opw(ops[0])
        op_store(self, st, j, ops[0], S.bnot(op_value(self, st, j, ops[0], w)))

    def _cmov(self, st, j, ops, cc):
        w = opw(ops[0])
        src = op_value(self, st, j, ops[1], w)  # read (and loaded from memory) whatever the condition
        old = read_reg(st, ops[0]['reg'])
        write_reg(st, ops[0]['reg'], S.ite(condition(st, cc), src, old))  # a 32-bit cmov zero-extends either way: write_reg does

    def i_movzx(self, st, j, ops):
        w = opw(ops[0])
        src = op_value(self, st, j, ops[1], opw(ops[1]))
        write_reg(st, ops[0]['reg'], S.zext(src, w))

    def i_movsx(self, st, j, ops):
        w = opw(ops[0])
        write_reg(st, ops[0]['reg'], S.sext(op_value(self, st, j, ops[1], opw(ops[1])), w))

    i_movabs = i_mov
    i_movsxd = i_movsx
    i_movzb = i_movzw = i_movzbl = i_movzwl = i_movzx
    i_movsw = i_movsx

    def i_movsb(self, st, j, ops):  # the AT&T name of both movsx from a byte and the string move
        (self.i_movsx if ops else self._movs)(st, j, ops)

    def _alu(self, st, j, ops, fn, flags, store=True):
        w = opw(ops[0])
        a = op_value(self, st, j, ops[0], w)
        b = op_value(self, st, j, ops[1], w)
        r = fn(a, b)
        flags(st, a, b, r)
        if store:
            op_store(self, st, j, ops[0], r)

    def i_add(self, st, j, ops):
        self._alu(st, j, ops, S.add, flags_add)

    def i_sub(self, st, j, ops):
        self._alu(st, j, ops, S.sub, flags_sub)

    def i_cmp(self, st, j, ops):
        self._alu(st, j, ops, S.sub, flags_sub, store=False)

    def i_and(self, st, j, ops):
        self._alu(st, j, ops, S.band, lambda st, a, b, r: flags_logic(st, r))

    def i_or(self, st, j, ops):
        self._alu(st, j, ops, S.bor, lambda st, a, b, r: flags_logic(st, r))

    def i_xor(self, st, j, ops):
        self._alu(st, j, ops, S.bxor, lambda st, a, b, r: flags_logic(st, r))

    def i_test(self, st, j, ops):
        self._alu(st, j, ops, S.band, lambda st, a, b, r: flags_logic(st, r), store=False)

    def i_lea(self, st, j, ops):
        a = mem_addr(st, j, ops[1]['mem'])
        w = opw(ops[0])
        write_reg(st, ops[0]['reg'], S.extract(a, w - 1, 0))

    def i_tzcnt(self, st, j, ops):
        if not self.bmi:
            return self.i_bsf(st, j, ops)
        w = opw(ops[0])
        x = op_value(self, st, j, ops[1], w)
        r = tzcnt(x)
        st.flags = {'CF': S.is_zero(x), 'ZF': S.is_zero(r), 'SF': None, 'OF': None, 'PF': None}
        write_reg(st, ops[0]['reg'], r)

    def i_bsf(self, st, j, ops):
        """The lowest set bit; with a zero source the destination is left
        undefined (an unknown value here) and ZF set."""
        w = opw(ops[0])
        x = op_value(self, st, j, ops[1], w)
        self.fresh_n += 1
        r = S.ite(S.is_zero(x), var('%sbsf%d' % (st.fresh, self.fresh_n), w), tzcnt(x))
        st.flags = {'ZF': S.is_zero(x), 'CF': None, 'SF': None, 'OF': None, 'PF': None}
        write_reg(st, ops[0]['reg'], r)

    def _shift(self, st, j, ops, fn, out_bit):
        w = opw(ops[0])
        a = op_value(self, st, j, ops[0], w)
        cnt = ops[1] if len(ops) > 1 else {'kind': 'Immediate', 'imm': 1, 'size': 8}
        if cnt['kind'] == 'Immediate':
            n = cnt['imm'] & (63 if w == 64 else 31)
            if n == 0:
                return
            r = fn(a, const(w, n))
            cf = S.eq(S.extract(a, out_bit(w, n), out_bit(w, n)), const(1, 1))
            st.flags = {'CF': cf, 'ZF': S.is_zero(r), 'SF': S.msb(r), 'OF': None, 'PF': None}
        else:
            n = S.band(S.zext(read_reg(st, cnt['reg']), w), const(w, 63 if w == 64 else 31))
            r = fn(a, n)
            st.flags = {'CF': None, 'ZF': None, 'SF': None, 'OF': None, 'PF': None}  # unchanged for a count of 0: not modelled
        op_store(self, st, j, ops[0], r)

    def i_shr(self, st, j, ops):
        self._shift(st, j, ops, S.lshr, lambda w, n: n - 1)

    def i_shl(self, st, j, ops):
        self._shift(st, j, ops, S.shl, lambda w, n: w - n)

    def i_sar(self, st, j, ops):
        self._shift(st, j, ops, S.ashr, lambda w, n: n - 1)

    def _shiftx(self, st, j, ops, fn):
        w = opw(ops[0])
        x = op_value(self, st, j, ops[1], w)
        n = S.band(op_value(self, st, j, ops[2], w), const(w, w - 1))
        write_reg(st, ops[0]['reg'], fn(x, n))

    def i_shrx(self, st, j, ops):
        self._shiftx(st, j, ops, S.lshr)

    def i_shlx(self, st, j, ops):
        self._shiftx(st, j, ops, S.shl)

    def i_sarx(self, st, j, ops):
        self._shiftx(st, j, ops, S.ashr)

    # --- mask registers
    def _kmov(self, st, j, ops, width):
        d, s = ops
        if s['kind'] == 'Register' and s['reg'].startswith('k') and s['reg'][1:].isdigit():
            v = S.extract(read_reg(st, s['reg']), width - 1, 0)
        else:
            v = S.extract(op_value(self, st, j, s, opw(s)), width - 1, 0)
        if d['kind'] == 'Register' and d['reg'] in GPR:
            write_reg(st, d['reg'], S.zext(v, d['size']))
        elif d['kind'] == 'Register':
            write_k(st, d['reg'], v, width)
        else:
            op_store(self, st, j, d, v)

    def i_kmovq(self, st, j, ops):
        self._kmov(st, j, ops, 64)

    def i_kmovd(self, st, j, ops):
        self._kmov(st, j, ops, 32)

    def i_kmovw(self, st, j, ops):
        self._kmov(st, j, ops, 16)

    def i_kmovb(self, st, j, ops):
        self._kmov(st, j, ops, 8)

    def _kortest(self, st, j, ops, width):
        r = S.extract(S.bor(read_reg(st, ops[0]['reg']), read_reg(st, ops[1]['reg'])), width - 1, 0)
        st.flags = {'ZF': S.is_zero(r), 'CF': S.eq(r, const(width, -1)), 'SF': S.FALSE, 'OF': S.FALSE, 'PF': S.FALSE}

    def i_kortestq(self, st, j, ops):
        self._kortest(st, j, ops, 64)

    def i_kortestd(self, st, j, ops):
        self._kortest(st, j, ops, 32)

    def i_kortestw(self, st, j, ops):
        self._kortest(st, j, ops, 16)

    # --- vectors
    def _width(self, ops):
        for o in ops:
            if o['kind'] == 'Register' and vec_index(o['reg']):
                return vec_index(o['reg'])[1]
        raise Unsupported('no vector operand')

    def _vmov(self, st, j, ops, aligned):
        d, s = ops
        n = self._width(ops)
        legacy = j.get('encoding') == 'Legacy'
        conds = lane_conds(st, j, n)
        if d['kind'] == 'Register':
            if conds is None:
                write_vec(st, d['reg'], vec_operand(self, st, j, s, n, aligned), legacy)
                return
            if s['kind'] == 'Memory':
                addr = mem_addr(st, j, s['mem'])
                src = load(self, st, addr, n, j['offset'], conds)
            else:
                src = read_vec(st, s['reg'])[:n]
            old = read_vec(st, d['reg'])[:n]
            zero = const(8, 0)
            write_vec(st, d['reg'], [S.ite(c, x, zero if j.get('zeroing') else o) for c, x, o in zip(conds, src, old)])
        else:
            lanes = read_vec(st, s['reg'])[:n]
            addr = mem_addr(st, j, d['mem'])
            if aligned:
                st.aligned.append((addr, n, j['offset']))
            store_bytes(self, st, addr, lanes, j['offset'], conds)

    def i_vmovdqu64(self, st, j, ops):
        self._vmov(st, j, ops, False)

    i_vmovdqu32 = i_vmovdqu8 = i_vmovdqu16 = i_vmovdqu = i_vmovups = i_movdqu = i_movups = i_vmovdqu64

    def i_vmovdqa64(self, st, j, ops):
        self._vmov(st, j, ops, True)

    i_vmovdqa32 = i_vmovdqa = i_vmovaps = i_movdqa = i_movaps = i_vmovdqa64

    def _vtestnm(self, st, j, ops, want_zero):
        n = self._width(ops)
        a = vec_operand(self, st, j, ops[1], n)
        b = vec_operand(self, st, j, ops[2], n)
        bs = [S.is_zero(S.band(x, y)) for x, y in zip(a, b)]
        if not want_zero:
            bs = [S.not_(x) for x in bs]
        conds = lane_conds(st, j, n)
        if conds is not None:
            bs = [S.and_(c, x) for c, x in zip(conds, bs)]
        write_k(st, ops[0]['reg'], bits(bs), n)

    def _vpcmp_k(self, st, j, ops, pred):
        n = self._width(ops)
        conds = lane_conds(st, j, n)
        a = vec_operand(self, st, j, ops[1], n)
        if ops[2]['kind'] == 'Memory' and conds is not None:
            b = load(self, st, mem_addr(st, j, ops[2]['mem']), n, j['offset'], conds)  # EVEX: no fault for a lane masked off
        else:
            b = vec_operand(self, st, j, ops[2], n)
        bs = [pred(x, y) for x, y in zip(a, b)]
        if conds is not None:
            bs = [S.and_(c, x) for c, x in zip(conds, bs)]
        write_k(st, ops[0]['reg'], bits(bs), n)

    def _vpcmp_pred(self, st, j, ops, signed):
        lt, le = (S.slt, S.sle) if signed else (S.ult, S.ule)
        p = ops[3]['imm'] & 7
        f = {0: S.eq, 1: lt, 2: le, 3: lambda x, y: S.FALSE, 4: lambda x, y: S.not_(S.eq(x, y)), 5: lambda x, y: S.not_(lt(x, y)),
             6: lambda x, y: S.not_(le(x, y)), 7: lambda x, y: S.TRUE}[p]
        self._vpcmp_k(st, j, ops[:3], f)

    def i_vpcmpb(self, st, j, ops):
        self._vpcmp_pred(st, j, ops, True)

    def i_vpcmpub(self, st, j, ops):
        self._vpcmp_pred(st, j, ops, False)

    def i_vptestnmb(self, st, j, ops):
        self._vtestnm(st, j, ops, True)

    def i_vptestmb(self, st, j, ops):
        self._vtestnm(st, j, ops, False)

    def _lanewise(self, st, j, ops, fn):
        n = self._width(ops)
        if len(ops) == 3:
            a = vec_operand(self, st, j, ops[1], n)
            b = vec_operand(self, st, j, ops[2], n)
            legacy = False
        else:
            a = vec_operand(self, st, j, ops[0], n)
            b = vec_operand(self, st, j, ops[1], n, aligned=True)
            legacy = True
        write_vec(st, ops[0]['reg'], [fn(x, y) for x, y in zip(a, b)], legacy)

    def i_vpminub(self, st, j, ops):
        self._lanewise(st, j, ops, lambda x, y: S.ite(S.ult(x, y), x, y))

    i_pminub = i_vpminub

    def i_movd(self, st, j, ops):
        d, s = ops
        if d['kind'] == 'Register' and vec_index(d['reg']):
            w = s['size'] if s['kind'] == 'Register' else 32 if j['mnemonic'].endswith('d') else 64
            v = op_value(self, st, j, s, w)
            n = vec_index(d['reg'])[1]
            lanes = [S.extract(v, 8 * i + 7, 8 * i) for i in range(w // 8)] + [const(8, 0)] * (16 - w // 8)
            if j.get('encoding') == 'Legacy':
                write_vec(st, 'xmm%d' % vec_index(d['reg'])[0], lanes, legacy=True)
            else:
                write_vec(st, 'xmm%d' % vec_index(d['reg'])[0], lanes)
        else:
            lanes = read_vec(st, s['reg'])
            w = d['size'] if d['kind'] == 'Register' else 32 if j['mnemonic'].endswith('d') else 64
            op_store(self, st, j, d, to_bv(lanes[:w // 8]))

    i_vmovd = i_movq = i_vmovq = i_movd

    def i_punpcklbw(self, st, j, ops):
        n = self._width(ops)
        legacy = len(ops) == 2
        a = vec_operand(self, st, j, ops[0 if legacy else 1], n)
        b = vec_operand(self, st, j, ops[1 if legacy else 2], n, aligned=legacy)
        out = []
        for lane in range(0, n, 16):  # per 128-bit lane: the low 8 bytes of each, interleaved
            for i in range(8):
                out += [a[lane + i], b[lane + i]]
        write_vec(st, ops[0]['reg'], out, legacy)

    i_vpunpcklbw = i_punpcklbw

    def i_pshuflw(self, st, j, ops):
        n = self._width(ops)
        legacy = j.get('encoding') == 'Legacy'
        src = vec_operand(self, st, j, ops[1], n, aligned=legacy)
        imm = ops[2]['imm']
        out = []
        for lane in range(0, n, 16):
            words = [src[lane + 2 * w:lane + 2 * w + 2] for w in range(8)]
            lo = [words[(imm >> (2 * k)) & 3] for k in range(4)]
            out += [b for w in lo + words[4:] for b in w]
        write_vec(st, ops[0]['reg'], out, legacy)

    def i_pshufd(self, st, j, ops):
        n = self._width(ops)
        legacy = j.get('encoding') == 'Legacy'
        src = vec_operand(self, st, j, ops[1], n, aligned=legacy)
        imm = ops[2]['imm']
        out = []
        for lane in range(0, n, 16):
            dw = [src[lane + 4 * w:lane + 4 * w + 4] for w in range(4)]
            out += [b for k in range(4) for b in dw[(imm >> (2 * k)) & 3]]
        write_vec(st, ops[0]['reg'], out, legacy)

    i_vpshufd = i_pshufd

    def i_vpbroadcastb(self, st, j, ops):
        n = self._width(ops[:1])
        s = ops[1]
        if s['kind'] == 'Register' and s['reg'] in GPR:
            b = S.extract(read_reg(st, s['reg']), 7, 0)
        elif s['kind'] == 'Register':
            b = read_vec(st, s['reg'])[0]
        else:
            b = load(self, st, mem_addr(st, j, s['mem']), 1, j['offset'])[0]
        write_vec(st, ops[0]['reg'], [b] * n)

    # --- byte arithmetic, shuffles, halves, the SSE4.2 string compare (strspn and strcspn)
    def i_paddb(self, st, j, ops):
        self._lanewise(st, j, ops, S.add)

    def i_psubb(self, st, j, ops):
        self._lanewise(st, j, ops, S.sub)

    def i_pcmpgtb(self, st, j, ops):
        self._lanewise(st, j, ops, lambda x, y: S.ite(S.slt(y, x), const(8, 0xff), const(8, 0)))

    i_vpaddb, i_vpsubb, i_vpcmpgtb = i_paddb, i_psubb, i_pcmpgtb

    def i_pblendvb(self, st, j, ops):
        """A lane from the second source where the mask's lane has its top
        bit (legacy: the mask is xmm0)."""
        n = self._width(ops)
        if len(ops) == 4:
            a, b, m, legacy = vec_operand(self, st, j, ops[1], n), vec_operand(self, st, j, ops[2], n), vec_operand(self, st, j, ops[3], n), False
        else:
            a, b, m, legacy = vec_operand(self, st, j, ops[0], n), vec_operand(self, st, j, ops[1], n, aligned=True), read_vec(st, 'xmm0'), True
        write_vec(st, ops[0]['reg'], [S.ite(S.msb(c), y, x) for x, y, c in zip(a, b, m)], legacy)

    i_vpblendvb = i_pblendvb

    def i_pshufb(self, st, j, ops):
        """Each lane the byte of its 128-bit half the index's low four bits
        name, 0 where the index has its top bit."""
        n = self._width(ops)
        legacy = len(ops) == 2
        a = vec_operand(self, st, j, ops[0 if legacy else 1], n)
        idx = vec_operand(self, st, j, ops[1 if legacy else 2], n, aligned=legacy)
        out = []
        for i, x in enumerate(idx):
            half = a[i // 16 * 16:i // 16 * 16 + 16]
            zero, k = shuffle_index(x)
            run = None if S.is_const(k) else consecutive(half)
            if run is not None:  # the half one array's 16 bytes in a row (a block loaded): the lane its byte at k
                v = S.select(run[0], merge_masks(S.add(run[1], index64(k))))
            else:
                v = half[15]
                for c in range(14, -1, -1):
                    v = S.ite(S.eq(k, const(8, c)), half[c], v)
            out.append(S.ite(zero, const(8, 0), v))
        write_vec(st, ops[0]['reg'], out, legacy)

    i_vpshufb = i_pshufb

    def i_psrlw(self, st, j, ops):
        n = self._width(ops)
        legacy = len(ops) == 2
        a = vec_operand(self, st, j, ops[0 if legacy else 1], n)
        c = ops[-1]
        if c['kind'] != 'Immediate':
            raise Unsupported('psrlw by a register at %#x' % st.rip)
        out = []
        for i in range(0, n, 2):
            w = S.lshr(S.concat(a[i + 1], a[i]), const(16, min(c['imm'], 16))) if c['imm'] < 16 else const(16, 0)
            out += [S.extract(w, 7, 0), S.extract(w, 15, 8)]
        write_vec(st, ops[0]['reg'], out, legacy)

    i_vpsrlw = i_psrlw

    def i_vbroadcasti128(self, st, j, ops):
        x = vec_operand(self, st, j, ops[1], 16)
        write_vec(st, ops[0]['reg'], x + x)

    def i_vextracti128(self, st, j, ops):
        if ops[0]['kind'] != 'Register':
            raise Unsupported('vextracti128 to memory at %#x' % st.rip)
        x = read_vec(st, ops[1]['reg'])[:32]
        h = ops[2]['imm'] & 1
        write_vec(st, ops[0]['reg'], x[16 * h:16 * h + 16])

    def i_vinserti128(self, st, j, ops):
        a = list(read_vec(st, ops[1]['reg'])[:32])
        b = vec_operand(self, st, j, ops[2], 16)
        h = ops[3]['imm'] & 1
        a[16 * h:16 * h + 16] = b
        write_vec(st, ops[0]['reg'], a)

    def _vpbroadcast(self, st, j, ops, e):
        n = self._width(ops[:1])
        s = ops[1]
        x = read_vec(st, s['reg'])[:e] if s['kind'] == 'Register' else load(self, st, mem_addr(st, j, s['mem']), e, j['offset'])
        write_vec(st, ops[0]['reg'], list(x) * (n // e))

    def i_vpbroadcastd(self, st, j, ops):
        self._vpbroadcast(st, j, ops, 4)

    def i_vpbroadcastq(self, st, j, ops):
        self._vpbroadcast(st, j, ops, 8)

    def i_vpinsrq(self, st, j, ops):
        a = list(read_vec(st, ops[1]['reg'])[:16])
        v = op_value(self, st, j, ops[2], 64)
        q = ops[3]['imm'] & 1
        a[8 * q:8 * q + 8] = [S.extract(v, 8 * i + 7, 8 * i) for i in range(8)]
        write_vec(st, ops[0]['reg'], a)

    def _pcmpistr(self, st, j, ops):
        """pcmpistri/m with unsigned bytes and 'equal any' (Intel SDM 4.1):
        the implicit lengths (up to the first NUL of each), bit i of the
        result where text byte i (i before the text's length) equals a set
        byte before the set's length; the polarity; the flags: CF a bit set,
        ZF the text has a NUL, SF the set has one, OF bit 0."""
        imm = ops[2]['imm']
        if imm & 3 or imm >> 2 & 3:
            raise Unsupported('pcmpistr mode %#x at %#x' % (imm, st.rip))
        a = read_vec(st, ops[0]['reg'])[:16]
        b = vec_operand(self, st, j, ops[1], 16)  # no alignment needed
        nul_before = lambda xs, i: S.or_(*[S.is_zero(x) for x in xs[:i]])  # noqa: E731
        va = [S.not_(nul_before(a, i + 1)) for i in range(16)]  # byte i before the first NUL
        # a text lane ite(g, 0, y) (the page-safe load's lanes past its block): NUL where g, else y's
        # tests, so that "y is a valid byte of the set" is a term of y alone (the lane lemma's)
        gb = [zero_or(x) for x in b]
        zb = [S.or_(g, S.is_zero(y)) for g, y in gb]
        vb = [S.not_(S.or_(*zb[:i + 1])) for i in range(16)]
        r1 = [S.and_(vb[i], S.not_(g), S.or_(*[S.and_(va[k], S.eq(a[k], y)) for k in range(16)])) for i, (g, y) in enumerate(gb)]
        pol = imm >> 4 & 3
        r2 = [S.not_(x) for x in r1] if pol == 1 else [S.ite(vb[i], S.not_(r1[i]), r1[i]) for i in range(16)] if pol == 3 else r1
        st.flags = {'CF': S.or_(*r2), 'ZF': S.not_(vb[15]), 'SF': S.not_(va[15]), 'OF': r2[0], 'PF': S.FALSE}
        return r2, imm

    def i_pcmpistri(self, st, j, ops):
        r2, imm = self._pcmpistr(st, j, ops)
        order = range(15, -1, -1) if imm & 0x40 else range(16)
        idx = const(32, 16)
        for i in reversed(list(order)):
            idx = S.ite(r2[i], const(32, i), idx)
        write_reg(st, 'ecx', idx)

    def i_pcmpistrm(self, st, j, ops):
        r2, imm = self._pcmpistr(st, j, ops)
        if imm & 0x40:
            lanes = [S.ite(x, const(8, 0xff), const(8, 0)) for x in r2]
        else:
            m = bits(r2)
            lanes = [S.extract(m, 7, 0), S.extract(m, 15, 8)] + [const(8, 0)] * 14
        write_vec(st, 'xmm0', lanes, j.get('encoding') == 'Legacy')

    i_vpcmpistri, i_vpcmpistrm = i_pcmpistri, i_pcmpistrm

    def i_pcmpeqb(self, st, j, ops):
        self._lanewise(st, j, ops, lambda x, y: S.ite(S.eq(x, y), const(8, 0xff), const(8, 0)))

    def i_vpcmpeqb(self, st, j, ops):
        if ops[0]['kind'] == 'Register' and ops[0]['reg'].startswith('k'):
            self._vpcmp_k(st, j, ops, S.eq)
        else:
            self.i_pcmpeqb(st, j, ops)

    def _pcmpeq_wide(self, st, j, ops, e):
        """pcmpeq of e-byte elements: an element's lanes all ones where the
        elements are equal (vpcmpeqd x, x, x: all ones)."""
        if ops[0]['kind'] == 'Register' and ops[0]['reg'].startswith('k'):
            raise Unsupported('%s into a mask register' % j['mnemonic'])
        n = self._width(ops)
        legacy = len(ops) == 2
        a = vec_operand(self, st, j, ops[0] if legacy else ops[1], n)
        b = vec_operand(self, st, j, ops[1] if legacy else ops[2], n, aligned=legacy and j.get('encoding') == 'Legacy')
        out = []
        for g in range(0, n, e):
            c = S.and_(*[S.eq(x, y) for x, y in zip(a[g:g + e], b[g:g + e])])
            out += [S.ite(c, const(8, 0xff), const(8, 0))] * e
        write_vec(st, ops[0]['reg'], out, legacy)

    def i_pcmpeqw(self, st, j, ops):
        self._pcmpeq_wide(st, j, ops, 2)

    def i_pcmpeqd(self, st, j, ops):
        self._pcmpeq_wide(st, j, ops, 4)

    def i_pcmpeqq(self, st, j, ops):
        self._pcmpeq_wide(st, j, ops, 8)

    i_vpcmpeqw, i_vpcmpeqd, i_vpcmpeqq = i_pcmpeqw, i_pcmpeqd, i_pcmpeqq

    def i_vpternlogd(self, st, j, ops):
        """Each bit the imm8 entry its three source bits index: a (the
        destination's old value) the high one, then b, then c."""
        if j.get('op_mask') not in (None, 'none'):
            raise Unsupported('vpternlog with a mask at %#x' % st.rip)
        n = self._width(ops)
        a = read_vec(st, ops[0]['reg'])[:n]
        b = vec_operand(self, st, j, ops[1], n)
        c = vec_operand(self, st, j, ops[2], n)
        imm = ops[3]['imm'] & 0xff
        out = []
        for x, y, z in zip(a, b, c):
            terms = []
            for i in range(8):
                if imm >> i & 1:
                    terms.append(S.band(S.band(x if i & 4 else S.bnot(x), y if i & 2 else S.bnot(y)), z if i & 1 else S.bnot(z)))
            r = const(8, 0)
            for tm in terms:
                r = S.bor(r, tm)
            out.append(r)
        write_vec(st, ops[0]['reg'], out)

    i_vpternlogq = i_vpternlogd

    def i_pxor(self, st, j, ops):
        self._lanewise(st, j, ops, S.bxor)

    i_vpxor = i_vpxord = i_vpxorq = i_xorps = i_vxorps = i_pxor

    def i_por(self, st, j, ops):
        self._lanewise(st, j, ops, S.bor)

    i_vpor = i_vpord = i_vporq = i_por

    def i_pand(self, st, j, ops):
        self._lanewise(st, j, ops, S.band)

    i_vpand = i_vpandd = i_vpandq = i_pand

    def i_pandn(self, st, j, ops):
        self._lanewise(st, j, ops, lambda x, y: S.band(S.bnot(x), y))

    i_vpandn = i_vpandnd = i_vpandnq = i_pandn

    def i_ptest(self, st, j, ops):
        """ZF: no bit set in both; CF: none in the source alone."""
        n = self._width(ops)
        legacy = j.get('encoding') == 'Legacy'
        a = vec_operand(self, st, j, ops[0], n)
        b = vec_operand(self, st, j, ops[1], n, aligned=legacy)
        st.flags = {'ZF': S.and_(*[S.is_zero(S.band(x, y)) for x, y in zip(a, b)]),
                    'CF': S.and_(*[S.is_zero(S.band(S.bnot(x), y)) for x, y in zip(a, b)]),
                    'OF': S.FALSE, 'SF': S.FALSE, 'PF': S.FALSE}

    i_vptest = i_ptest

    def i_pmovmskb(self, st, j, ops):
        n = self._width(ops)
        lanes = vec_operand(self, st, j, ops[1], n)
        m = bits([S.msb(x) for x in lanes])
        write_reg(st, ops[0]['reg'], S.zext(m, ops[0]['size']) if m.w < ops[0]['size'] else S.extract(m, ops[0]['size'] - 1, 0))

    i_vpmovmskb = i_pmovmskb

    def i_vzeroupper(self, st, j, ops):
        for n in range(16):
            lanes = read_vec(st, 'zmm%d' % n)
            st.vec[n] = lanes[:16] + [const(8, 0)] * 48


for _cc in x86.CC:
    setattr(Machine, 'i_cmov' + _cc, (lambda cc: lambda self, st, j, ops: self._cmov(st, j, ops, cc))(_cc))
    setattr(Machine, 'i_set' + _cc, (lambda cc: lambda self, st, j, ops: op_store(self, st, j, ops[0], S.zext(S.bit(condition(st, cc)), 8)))(_cc))
