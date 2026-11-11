"""x86-64 encoder: a structured instruction (as x86.decode produces it, or as
opt.py rewrites it) -> machine code bytes.

The opcode tables of x86.py are indexed the other way round: for a mnemonic,
every form that could encode it. The operands are bound to each form's
operand specs; forms that fit are encoded and the shortest wins, unless the
instruction carries a 'form' hint (the encoding it was decoded from) that
still fits: then that one wins, so an untouched object round-trips byte for
byte. Branch displacements are supplied by the caller (rel = target - end).

encode(j, rel=None, branch_size=None) -> (bytes, fields) where fields maps
'disp', 'imm' and 'branch' to (position, size in bytes) inside the bytes, for
relocations.
"""
import x86
from x86 import R64, R32, R16, R8, R8_LEGACY, CC

GPR = {}
for n in range(16):
    GPR[R64[n]] = (n, 64)
    GPR[R32[n]] = (n, 32)
    GPR[R16[n]] = (n, 16)
    GPR[R8[n]] = (n, 8)
for n in range(4):
    GPR[R8_LEGACY[4 + n]] = (4 + n, 8)   # ah ch dh bh: no REX allowed
HIGH8 = {'ah', 'ch', 'dh', 'bh'}
NEEDS_REX8 = {'spl', 'bpl', 'sil', 'dil'}
VEC_WIDTH = {'xmm': 128, 'ymm': 256, 'zmm': 512}
SEG_BYTE = {'es': 0x26, 'cs': 0x2e, 'ss': 0x36, 'ds': 0x3e, 'fs': 0x64, 'gs': 0x65}

# mnemonic as the decoder names it -> (table mnemonic, forced operand size or None)
ALIASES = {'cbtw': ('cbw', 16), 'cwtl': ('cbw', 32), 'cltq': ('cbw', 64), 'cwtd': ('cwd', 16), 'cltd': ('cwd', 32), 'cqto': ('cwd', 64),
           'movsw': ('movs', 16), 'movsl': ('movs', 32), 'movsq': ('movs', 64), 'stosw': ('stos', 16), 'stosl': ('stos', 32), 'stosq': ('stos', 64),
           'movabs': ('mov', None), 'pextrq': ('pextrd', None), 'pinsrq': ('pinsrd', None)}


class NoMatch(Exception):
    pass


class Candidate:
    __slots__ = ('mnemonic', 'map', 'opcode', 'pfx', 'w', 'specs', 'attrs', 'group', 'kind', 'cc')

    def __init__(self, mnemonic, map_, opcode, pfx, w, specs, attrs, group=None, kind='legacy', cc=False):
        self.mnemonic, self.map, self.opcode, self.pfx, self.w = mnemonic, map_, opcode, pfx, w
        self.specs, self.attrs, self.group, self.kind, self.cc = specs, attrs.split(), group, kind, cc

    def __repr__(self):
        return '<%s map%d %02x %s %s %s>' % (self.mnemonic, self.map, self.opcode, self.pfx, self.kind, self.specs)


INDEX = {}


def add(c):
    INDEX.setdefault(c.mnemonic, []).append(c)


def build_index():
    if INDEX:
        return
    for op, e in x86.ONE.items():
        if e is None:
            continue
        m, specs, attrs = e
        if 'cc' in attrs.split():
            if op & 15 == 0:
                add(Candidate(m, 0, op, '', '*', specs, attrs, cc=True))
            continue
        add(Candidate(m, 0, op, '', '*', specs, attrs))
    for op, entries in x86.GROUPS.items():
        for reg, e in enumerate(entries):
            if e is not None:
                add(Candidate(e[0], 0, op, '', '*', e[1], e[2], group=reg))
    for (op, pfx), e in x86.TWO.items():
        m, specs, attrs = e
        if m is None:
            continue
        if 'cc' in attrs.split():
            if op & 15 == 0:
                add(Candidate(m, 1, op, '' if pfx == '*' else pfx, '*', specs, attrs, cc=True))
            continue
        add(Candidate(m, 1, op, '' if pfx == '*' else pfx, '*', specs, attrs))
    for table, map_ in ((x86.THREE8, 2), (x86.THREE_A, 3)):
        for (op, pfx), e in table.items():
            add(Candidate(e[0], map_, op, '' if pfx == '*' else pfx, '*', e[1], e[2]))
    for op, entries in x86.TWO_GROUPS.items():
        for reg, m in enumerate(entries):
            if m:
                add(Candidate(m, 1, op, '66', '*', ['Udq', 'Ib'], 'acc=RW,R', group=reg))
    for reg, m in enumerate([None, None, None, None, 'bt', 'bts', 'btr', 'btc']):
        if m:
            add(Candidate(m, 1, 0xba, '', '*', ['Ev', 'Ib'], '', group=reg))
    add(Candidate('xgetbv', 1, 0x01d0, '', '*', [], 'nosuffix'))
    add(Candidate('sfence', 1, 0xaef8, '', '*', [], 'nosuffix'))
    add(Candidate('endbr64', 1, 0x1efa, 'f3', '*', [], 'nosuffix'))
    add(Candidate('endbr32', 1, 0x1efb, 'f3', '*', [], 'nosuffix'))
    for (map_, op, pfx, w), (m, specs, attrs, evex) in x86.VEX.items():
        a = attrs.split()
        if evex != 'only' and 'evexonly' not in a:
            add(Candidate(m, map_, op, pfx, w, specs, attrs, kind='vex'))
        if evex != 'no':
            if isinstance(evex, tuple):
                add(Candidate(evex[0], map_, op, pfx, w, evex[1], evex[2], kind='evex'))
            else:
                add(Candidate(m, map_, op, pfx, w, specs, attrs, kind='evex'))
    for op, entries in x86.VEX_GROUPS.items():
        for reg, m in enumerate(entries):
            if m:
                add(Candidate(m, 1, op, '66', '*', ['Hx', 'Ux', 'Ib'], 'vex', group=reg, kind='vex'))
                add(Candidate(m, 1, op, '66', '*', ['Hx', 'Ux', 'Ib'], 'vex', group=reg, kind='evex'))
    for reg, m in enumerate(x86.BMI_GROUP17):
        if m:
            add(Candidate(m, 2, 0xf3, '', '*', ['By', 'Ey'], 'vex', group=reg, kind='vex'))


# ---------------------------------------------------------------- binding

class Plan:
    """What one candidate needs to be encoded with these operands."""

    def __init__(self):
        self.osz = None        # operand size of 'v' specs
        self.ysz = None        # 32/64 of 'y' specs
        self.vw = None         # vector width (L)
        self.reg = None        # ModRM.reg: (number, is_vector)
        self.rm = None         # ('reg', number, is_vector) or ('mem', mem dict)
        self.vvvv = None       # (number, is_vector)
        self.imms = []         # (value, size bits, signed)
        self.branch = None     # size bits
        self.opreg = None      # register in the opcode byte
        self.needs_rex = False
        self.no_rex = False
        self.evex_only = False
        self.mem_size = 0
        self.lreg = None       # register in imm8[7:4]


def vec(name):
    if name[:3] in VEC_WIDTH and name[3:].isdigit():
        return int(name[3:]), VEC_WIDTH[name[:3]]
    return None


def kreg(name):
    return int(name[1:]) if name[0] == 'k' and name[1:].isdigit() else None


def bind(cand, ops, plan):
    specs = cand.specs
    if len(specs) != len(ops):
        raise NoMatch
    # vector width from any vector register operand, before the memory operands need it
    for o in ops:
        if o['kind'] == 'Register' and vec(o['reg']):
            plan.vw = max(plan.vw or 0, vec(o['reg'])[1])
    for spec, o in zip(specs, ops):
        kind = o['kind']
        if spec in ('Eb', 'Ew', 'Ed', 'Eq', 'Ev', 'Ey', 'M', 'Mq', 'Mdq', 'Mv', 'Mk'):
            size = {'Eb': 8, 'Ew': 16, 'Ed': 32, 'Eq': 64, 'Ev': 0, 'Ey': 0, 'M': 0, 'Mq': 64, 'Mdq': 128, 'Mv': 0, 'Mk': 0}[spec]
            msize = next((int(a[6:]) for a in cand.attrs if a.startswith('msize=')), size)
            if kind == 'Register':
                if spec.startswith('M') or o['reg'] not in GPR:
                    raise NoMatch
                n, rs = GPR[o['reg']]
                if spec == 'Ev':
                    fit_v(plan, rs)
                elif spec == 'Ey':
                    fit_y(plan, rs)
                elif rs != size:
                    raise NoMatch
                gpr8(plan, o['reg'])
                plan.rm = ('reg', n, False)
            elif kind == 'Memory':
                ms = o['mem']['size'] * 8
                if spec == 'Ev' or spec == 'Mv':
                    fit_v(plan, ms)
                elif spec == 'Ey':
                    fit_y(plan, ms)
                elif spec in ('M', 'Mk'):
                    pass
                elif ms and ms != msize:
                    raise NoMatch
                plan.rm = ('mem', o['mem'])
                plan.mem_size = ms
            else:
                raise NoMatch
        elif spec in ('Gb', 'Gw', 'Gd', 'Gq', 'Gv', 'Gy'):
            if kind != 'Register' or o['reg'] not in GPR:
                raise NoMatch
            n, rs = GPR[o['reg']]
            size = {'Gb': 8, 'Gw': 16, 'Gd': 32, 'Gq': 64, 'Gv': 0, 'Gy': 0}[spec]
            if spec == 'Gv':
                fit_v(plan, rs)
            elif spec == 'Gy':
                fit_y(plan, rs)
            elif rs != size:
                raise NoMatch
            gpr8(plan, o['reg'])
            plan.reg = (n, False)
        elif spec in ('Rd', 'Rq', 'Ry'):
            if kind != 'Register' or o['reg'] not in GPR:
                raise NoMatch
            n, rs = GPR[o['reg']]
            if spec == 'Ry':
                fit_y(plan, rs)
            elif rs != {'Rd': 32, 'Rq': 64}[spec]:
                raise NoMatch
            plan.rm = ('reg', n, False)
        elif spec == 'By':
            if kind != 'Register' or o['reg'] not in GPR:
                raise NoMatch
            n, rs = GPR[o['reg']]
            fit_y(plan, rs)
            plan.vvvv = (n, False)
        elif spec in ('Zb', 'Zv', 'Zq', 'Zy'):
            if kind != 'Register' or o['reg'] not in GPR:
                raise NoMatch
            n, rs = GPR[o['reg']]
            if spec == 'Zv':
                fit_v(plan, rs)
            elif spec == 'Zy':
                fit_y(plan, rs)
            elif rs != {'Zb': 8, 'Zq': 64}[spec]:
                raise NoMatch
            gpr8(plan, o['reg'])
            plan.opreg = n
        elif spec == 'AL':
            if kind != 'Register' or o['reg'] != 'al':
                raise NoMatch
        elif spec == 'rAX':
            if kind != 'Register' or o['reg'] not in ('ax', 'eax', 'rax'):
                raise NoMatch
            fit_v(plan, GPR[o['reg']][1])
        elif spec == 'CL':
            if kind != 'Register' or o['reg'] != 'cl':
                raise NoMatch
        elif spec == 'ONE':
            if kind != 'Immediate' or o['imm'] != 1:
                raise NoMatch
        elif spec in ('Ib', 'Iw', 'Id', 'Iz', 'Iv'):
            if kind != 'Immediate':
                raise NoMatch
            plan.imms.append((o['imm'], spec))
        elif spec in ('Jb', 'Jz'):
            if kind != 'NearBranch':
                raise NoMatch
            plan.branch = 8 if spec == 'Jb' else 32
        elif spec in ('Vx', 'Vdq', 'Vqq'):
            v = vec(o['reg']) if kind == 'Register' else None
            if v is None or not fit_vec(plan, spec[1:], v[1]):
                raise NoMatch
            plan.reg = (v[0], True)
        elif spec in ('Hx', 'Hdq', 'Hqq'):
            v = vec(o['reg']) if kind == 'Register' else None
            if v is None or not fit_vec(plan, spec[1:], v[1]):
                raise NoMatch
            plan.vvvv = (v[0], True)
        elif spec in ('Wx', 'Wdq', 'Wqq', 'Wq', 'Wd', 'Ww', 'Wb', 'Whalf', 'Ux', 'Udq'):
            if kind == 'Register':
                v = vec(o['reg'])
                if v is None:
                    raise NoMatch
                want = {'Wx': 'x', 'Wdq': 'dq', 'Wqq': 'qq', 'Ux': 'x', 'Udq': 'dq', 'Wq': 'dq', 'Wd': 'dq', 'Ww': 'dq', 'Wb': 'dq', 'Whalf': 'half'}[spec]
                if want == 'half':
                    if plan.vw is None or v[1] != (128 if plan.vw <= 256 else 256):
                        raise NoMatch
                elif not fit_vec(plan, want, v[1]):
                    raise NoMatch
                plan.rm = ('reg', v[0], True)
            elif kind == 'Memory' and not spec.startswith('U'):
                ms = o['mem']['size'] * 8
                want = {'Wx': None, 'Wdq': 128, 'Wqq': 256, 'Wq': 64, 'Wd': 32, 'Ww': 16, 'Wb': 8, 'Whalf': None}[spec]
                if spec == 'Wx':
                    if plan.vw is None:
                        plan.vw = ms
                    elif ms and ms != plan.vw:
                        raise NoMatch
                elif spec == 'Whalf':
                    if plan.vw is None or (ms and ms != plan.vw // 2):
                        raise NoMatch
                elif ms and ms != want:
                    raise NoMatch
                plan.rm = ('mem', o['mem'])
                plan.mem_size = ms
            else:
                raise NoMatch
        elif spec == 'Lx':
            v = vec(o['reg']) if kind == 'Register' else None
            if v is None or not fit_vec(plan, 'x', v[1]):
                raise NoMatch
            plan.lreg = v[0]
        elif spec == 'Kr':
            k = kreg(o['reg']) if kind == 'Register' else None
            if k is None:
                raise NoMatch
            plan.reg = (k, False)
        elif spec == 'Km':
            if kind == 'Register':
                k = kreg(o['reg'])
                if k is None:
                    raise NoMatch
                plan.rm = ('reg', k, False)
            elif kind == 'Memory':
                plan.rm = ('mem', o['mem'])
            else:
                raise NoMatch
        elif spec == 'Kv':
            k = kreg(o['reg']) if kind == 'Register' else None
            if k is None:
                raise NoMatch
            plan.vvvv = (k, False)
        else:
            raise NoMatch
    if plan.vw is None:
        plan.vw = 128
    if plan.vw == 512:
        plan.evex_only = True


def fit_v(plan, size):
    if size not in (16, 32, 64) or (plan.osz is not None and plan.osz != size):
        raise NoMatch
    plan.osz = size


def fit_y(plan, size):
    if size not in (32, 64) or (plan.ysz is not None and plan.ysz != size):
        raise NoMatch
    plan.ysz = size


def fit_vec(plan, want, width):
    if want == 'dq':
        return width == 128
    if want == 'qq':
        return width == 256
    if plan.vw is not None and plan.vw != width:
        return False
    plan.vw = width
    return True


def gpr8(plan, name):
    if name in NEEDS_REX8:
        plan.needs_rex = True
    if name in HIGH8:
        plan.no_rex = True


# ---------------------------------------------------------------- bytes

def imm_fits(value, spec, osz, signed_hint):
    """Returns the immediate size in bits for the spec, or None."""
    if spec == 'Ib':
        return 8 if -128 <= value <= (127 if signed_hint else 255) else None
    if spec == 'Iw':
        return 16 if -32768 <= value <= 65535 else None
    if spec == 'Id':
        return 32 if -2 ** 31 <= value < 2 ** 32 else None
    if spec == 'Iz':
        if osz == 16:
            return 16 if -32768 <= value <= 65535 else None
        if osz == 64:
            return 32 if -2 ** 31 <= value < 2 ** 31 else None
        return 32 if -2 ** 31 <= value < 2 ** 32 else None
    if spec == 'Iv':
        return osz if osz is not None and -2 ** (osz - 1) <= value < 2 ** osz else None
    return None


def encode_mem(mem, disp_n, hint_disp):
    """ModRM mod/rm, SIB, displacement bytes and REX.X/B for a memory operand.
    disp_n: EVEX disp8*N scale (1 otherwise). Returns (mod, rm, sib, disp_bytes, X, B, disp_size)."""
    base, index, scale, disp = mem['base'], mem['index'], mem['scale'], mem['disp']
    base = None if base in (None, 'none') else base
    index = None if index in (None, 'none') else index
    if mem['rip_rel']:
        return 0, 5, None, (disp & 0xffffffff).to_bytes(4, 'little'), 0, 0, 32
    X = B = 0
    sib = None
    if index is not None:
        i = GPR[index][0]
        if i == 4:
            raise NoMatch
        X = i >> 3
        ss = {1: 0, 2: 1, 4: 2, 8: 3}[scale]
    if base is None:
        if index is None:
            return 0, 4, 0x25, (disp & 0xffffffff).to_bytes(4, 'little'), 0, 0, 32
        return 0, 4, (ss << 6) | ((i & 7) << 3) | 5, (disp & 0xffffffff).to_bytes(4, 'little'), X, 0, 32
    b = GPR[base][0]
    B = b >> 3
    if index is not None:
        sib = (ss << 6) | ((i & 7) << 3) | (b & 7)
        rm = 4
    elif b & 7 == 4:
        sib = 0x24 | (b & 7)
        rm = 4
    else:
        rm = b & 7
    # displacement size
    if hint_disp == 0 and disp == 0 and b & 7 != 5:
        mod, dbytes, dsize = 0, b'', 0
    elif hint_disp == 8 or (hint_disp is None and disp % disp_n == 0 and -128 <= disp // disp_n <= 127 and not (disp == 0 and b & 7 != 5 and hint_disp is None)):
        if disp % disp_n or not -128 <= disp // disp_n <= 127:
            raise NoMatch
        mod, dbytes, dsize = 1, ((disp // disp_n) & 0xff).to_bytes(1, 'little'), 8
    elif disp == 0 and b & 7 != 5 and hint_disp is None:
        mod, dbytes, dsize = 0, b'', 0
    else:
        if not -2 ** 31 <= disp < 2 ** 31:
            raise NoMatch
        mod, dbytes, dsize = 2, (disp & 0xffffffff).to_bytes(4, 'little'), 32
    return mod, rm, sib, dbytes, X, B, dsize


def try_encode(cand, j, rel, branch_size, hint):
    ops = j['ops']
    plan = Plan()
    bind(cand, ops, plan)
    mnemonic = j['mnemonic']
    alias = ALIASES.get(mnemonic)
    forced = alias[1] if alias else None
    if forced:
        if plan.osz is not None and plan.osz != forced:
            raise NoMatch
        plan.osz = forced
    masked = j.get('op_mask', 'none') != 'none'
    hi_regs = any(x[0] >= 16 for x in (plan.reg, plan.vvvv) if x and x[1]) or (plan.rm and plan.rm[0] == 'reg' and plan.rm[2] and plan.rm[1] >= 16)
    if cand.kind == 'legacy':
        if masked or j.get('zeroing') or hi_regs:
            raise NoMatch
    elif cand.kind == 'vex':
        if masked or j.get('zeroing') or hi_regs or plan.evex_only or plan.vw == 512:
            raise NoMatch
    use_hint = hint is not None and hint['map'] == cand.map and hint['pfx'] == cand.pfx and hint['encoding'] == {'legacy': 'Legacy', 'vex': 'VEX', 'evex': 'EVEX'}[cand.kind] \
        and (hint['opcode'] == cand.opcode or (cand.cc and hint['opcode'] & 0xf0 == cand.opcode))
    out = bytearray()
    fields = {}
    # immediates
    imm_bytes = bytearray()
    imm_pos = None
    for value, spec in plan.imms:
        size = imm_fits(value, spec, plan.osz, 'imm8s' in cand.attrs)
        if size is None:
            raise NoMatch
        if use_hint and hint['imm'] and hint['imm'] != size and len(plan.imms) == 1 and plan.branch is None:
            raise NoMatch
        if imm_pos is None:
            imm_pos = len(imm_bytes)
        imm_bytes += (value & ((1 << size) - 1)).to_bytes(size // 8, 'little')
    if plan.lreg is not None:
        imm_bytes += bytes([plan.lreg << 4])
    if plan.branch:
        bsize = branch_size or (hint['imm'] if use_hint and hint['imm'] else None) or plan.branch
        if bsize != plan.branch:
            raise NoMatch
        if rel is None:
            rel = 0
        if plan.branch == 8 and not -128 <= rel <= 127:
            raise NoMatch
        if plan.branch == 32 and not -2 ** 31 <= rel < 2 ** 31:
            raise NoMatch
        imm_pos = len(imm_bytes)
        imm_bytes += (rel & ((1 << plan.branch) - 1)).to_bytes(plan.branch // 8, 'little')
    # ModRM / SIB / displacement
    modrm = None
    tail = b''
    X = B = 0
    disp_pos = disp_size = None
    needs_modrm = plan.reg is not None or plan.rm is not None or cand.group is not None
    W = 0
    if cand.kind == 'legacy':
        if plan.osz == 64 or plan.ysz == 64:
            W = 1
    else:
        W = cand.w if cand.w in (0, 1) else (1 if plan.ysz == 64 else 0)
    if needs_modrm:
        regfield = cand.group if cand.group is not None else (plan.reg[0] & 7 if plan.reg else 0)
        R = (plan.reg[0] >> 3) & 1 if plan.reg else 0
        R2 = (plan.reg[0] >> 4) & 1 if plan.reg and plan.reg[1] else 0
        if plan.rm is None:
            raise NoMatch
        if plan.rm[0] == 'reg':
            n = plan.rm[1]
            modrm = 0xc0 | (regfield << 3) | (n & 7)
            B = (n >> 3) & 1
            X = (n >> 4) & 1 if plan.rm[2] else 0
        else:
            mem = plan.rm[1]
            disp_n = 1
            if cand.kind == 'evex':
                disp_n = plan.vw // 8
                if 't1' in cand.attrs:
                    disp_n = max(1, plan.mem_size // 8)
                for a in cand.attrs:
                    if a.startswith('N='):
                        disp_n = int(a[2:])
            hint_disp = hint['disp'] if use_hint else None
            mod, rm, sib, dbytes, X, B, disp_size = encode_mem(mem, disp_n, hint_disp)
            modrm = (mod << 6) | (regfield << 3) | rm
            tail = (bytes([sib]) if sib is not None else b'') + dbytes
            if dbytes:
                disp_pos = 1 + (1 if sib is not None else 0)
    else:
        R = R2 = 0
    # prefixes and opcode
    seg = next((o['mem'].get('seg') for o in ops if o['kind'] == 'Memory'), None)
    if j.get('notrack'):
        seg = 'ds'
    if cand.kind != 'legacy':
        opcode_bytes = bytes([cand.opcode & 0xff])   # the map lives in the VEX/EVEX prefix
    else:
        opcode_bytes = {0: bytes([cand.opcode & 0xff]), 1: bytes([0x0f, cand.opcode & 0xff]), 2: bytes([0x0f, 0x38, cand.opcode & 0xff]),
                        3: bytes([0x0f, 0x3a, cand.opcode & 0xff])}[cand.map]
        if cand.opcode > 0xff:  # two-byte opcode after 0F (xgetbv, endbr)
            opcode_bytes = bytes([0x0f, cand.opcode >> 8, cand.opcode & 0xff])
    if cand.cc:
        opcode_bytes = opcode_bytes[:-1] + bytes([(cand.opcode & 0xf0) | CC.index(j['cc'])])
    if plan.opreg is not None:
        opcode_bytes = opcode_bytes[:-1] + bytes([(cand.opcode & 0xf8) | (plan.opreg & 7)])
        B = (plan.opreg >> 3) & 1
    if cand.kind == 'legacy':
        if j.get('lock'):
            out.append(0xf0)
        if plan.osz == 16 and cand.pfx != '66':
            out.append(0x66)
        if cand.pfx:
            out.append({'66': 0x66, 'f3': 0xf3, 'f2': 0xf2}[cand.pfx])
        if j.get('rep'):
            out.append(0xf3 if j['rep'] == 'rep' else 0xf2)
        if seg:
            out.append(SEG_BYTE[seg])
        rex = (W << 3) | (R << 2) | (X << 1) | B
        if rex or plan.needs_rex or (use_hint and hint.get('rex')):
            if plan.no_rex:
                raise NoMatch
            out.append(0x40 | rex)
        out += opcode_bytes
    else:
        if seg:
            out.append(SEG_BYTE[seg])
        pp = {'': 0, '66': 1, 'f3': 2, 'f2': 3}[cand.pfx]
        vvvv = plan.vvvv[0] if plan.vvvv else 0
        V2 = (vvvv >> 4) & 1
        L = {128: 0, 256: 1, 512: 2}[plan.vw]
        if cand.kind == 'vex':
            if use_hint and hint.get('L') is not None and not any(o['kind'] == 'Register' and vec(o['reg']) for o in ops):
                L = hint['L']
            if 'vlen=1' in cand.attrs:
                L = 1
            if cand.map == 1 and W == 0 and X == 0 and B == 0 and not (use_hint and not hint.get('vex2', True)):
                out += bytes([0xc5, ((R ^ 1) << 7) | ((~vvvv & 15) << 3) | (L << 2) | pp])
            else:
                out += bytes([0xc4, ((R ^ 1) << 7) | ((X ^ 1) << 6) | ((B ^ 1) << 5) | cand.map,
                              (W << 7) | ((~vvvv & 15) << 3) | (L << 2) | pp])
        else:
            aaa = int(j['op_mask'][1:]) if masked else 0
            if use_hint and hint.get('L') is not None and not any(o['kind'] == 'Register' and vec(o['reg']) for o in ops):
                L = hint['L']
            out += bytes([0x62, ((R ^ 1) << 7) | ((X ^ 1) << 6) | ((B ^ 1) << 5) | ((R2 ^ 1) << 4) | cand.map,
                          (W << 7) | ((~vvvv & 15) << 3) | 4 | pp,
                          ((1 if j.get('zeroing') else 0) << 7) | (L << 5) | ((1 if j.get('bcast') else 0) << 4) | ((V2 ^ 1) << 3) | aaa])
        out += opcode_bytes
    if modrm is not None:
        modrm_at = len(out)
        out.append(modrm)
        out += tail
        if disp_pos is not None:
            fields['disp'] = (modrm_at + disp_pos, disp_size // 8)
    if imm_bytes:
        at = len(out)
        if plan.branch:
            fields['branch'] = (at + imm_pos, plan.branch // 8)
        elif imm_pos is not None:
            fields['imm'] = (at + imm_pos, len(imm_bytes) - imm_pos - (1 if plan.lreg is not None else 0))
        out += imm_bytes
    return bytes(out), fields, use_hint


def encode(j, rel=None, branch_size=None):
    b, fields, _ = encode_with_form(j, rel, branch_size)
    return b, fields


def encode_with_form(j, rel=None, branch_size=None):
    """Also return the selected table form, for a compiler emitting fresh IR."""
    build_index()
    mnemonic = j['mnemonic']
    if j.get('cc'):
        base = mnemonic[:-len(j['cc'])]
    else:
        base = ALIASES.get(mnemonic, (mnemonic, None))[0]
    cands = INDEX.get(base, [])
    if base == 'movq':
        cands = cands + INDEX['movd']   # 66 REX.W 0F 6E/7E: movd's form with a 64-bit register
    if not cands:
        raise NoMatch('no encoding for ' + mnemonic)
    best = None
    for cand in cands:
        # Rewrites retain the instruction's encoding family (and explicitly
        # update it when converting a function to EVEX). A shorter compressed
        # displacement cannot introduce AVX-512 into an SSE/AVX2 function.
        if cand.kind == 'evex' and j.get('encoding') in ('Legacy', 'VEX'):
            continue
        try:
            b, fields, hinted = try_encode(cand, j, rel, branch_size, j.get('form'))
        except NoMatch:
            continue
        key = (0 if hinted else 1, len(b))
        if best is None or key < best[0]:
            best = (key, b, fields, cand)
    if best is None:
        raise NoMatch('no form fits %s %s' % (mnemonic, [o.get('reg') or o.get('mem') or o.get('imm') or o.get('branch') for o in j['ops']]))
    return best[1], best[2], best[3]
