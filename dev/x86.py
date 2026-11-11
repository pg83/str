"""x86-64 decoder for compiler output (clang/gcc, -O0..-O3) of this library.

decode(code, base) -> list of dicts, one per instruction:
  offset, len, bytes, mnemonic, gas (AT&T text, iced style: operands joined by
  ',' without spaces, masks as {%k1}{z}), encoding ('Legacy'|'VEX'|'EVEX'),
  op_mask, zeroing, flow, branch_target (direct jumps/calls), ops (kind:
  Register/Memory/Immediate/NearBranch), regs ([{name, full, access}]),
  mems ([{base, index, scale, disp, size, access}]).

Operand specs follow the SDM letters: E = r/m, G = reg, M = memory only,
I = immediate, J = relative branch, Z = register in the opcode byte, V = vector
reg in ModRM.reg, W = vector r/m, H = vector in VEX.vvvv, U = vector r/m that
must be a register, K = mask register (Kr: ModRM.reg, Km: r/m, Kv: vvvv),
B = GPR in VEX.vvvv. Size letters: b 8, w 16, d 32, q 64, v 16/32/64 by
operand size, y 32/64 by REX.W, x = vector width by VEX.L / EVEX.L'L, dq =
xmm, qq = ymm, z = 16/32 immediate (32 sign-extended for 64-bit operands).

The gas text for rip-relative memory operands prints the absolute address
(end of instruction + displacement), as iced did; emit.py replaces it by the
relocation symbol. Jump/call targets print as 0x%016X for the same reason.

Access kinds: Read, Write, ReadWrite. uses()/defs() in opt.py key on these.
"""
import struct

R64 = ['rax', 'rcx', 'rdx', 'rbx', 'rsp', 'rbp', 'rsi', 'rdi'] + ['r%d' % i for i in range(8, 16)]
R32 = ['eax', 'ecx', 'edx', 'ebx', 'esp', 'ebp', 'esi', 'edi'] + ['r%dd' % i for i in range(8, 16)]
R16 = ['ax', 'cx', 'dx', 'bx', 'sp', 'bp', 'si', 'di'] + ['r%dw' % i for i in range(8, 16)]
R8 = ['al', 'cl', 'dl', 'bl', 'spl', 'bpl', 'sil', 'dil'] + ['r%db' % i for i in range(8, 16)]
R8_LEGACY = ['al', 'cl', 'dl', 'bl', 'ah', 'ch', 'dh', 'bh']
SEG = {0x26: 'es', 0x2e: 'cs', 0x36: 'ss', 0x3e: 'ds', 0x64: 'fs', 0x65: 'gs'}
CC = ['o', 'no', 'b', 'ae', 'e', 'ne', 'be', 'a', 's', 'ns', 'p', 'np', 'l', 'ge', 'le', 'g']
SUFFIX = {8: 'b', 16: 'w', 32: 'l', 64: 'q'}
PRED = ['eq', 'lt', 'le', 'false', 'neq', 'nlt', 'nle', 'true']

R, W, RW = 'Read', 'Write', 'ReadWrite'
# dst = f(x, x) that does not depend on x: a write only, as iced and the hardware treat it
ZERO_IDIOMS = {'xor', 'sub', 'pxor', 'vpxor', 'vpxord', 'vpxorq', 'xorps', 'xorpd', 'vxorps', 'vxorpd', 'psubb', 'psubw', 'psubd', 'psubq',
               'vpsubb', 'vpsubw', 'vpsubd', 'vpsubq', 'pcmpgtb', 'pcmpgtw', 'pcmpgtd', 'pandn', 'vpandn', 'vpandnd', 'vpandnq', 'andnps', 'andnpd'}


class DecodeError(Exception):
    pass


def gpr(n, size, rex):
    if size == 64: return R64[n]
    if size == 32: return R32[n]
    if size == 16: return R16[n]
    return R8[n] if rex is not None else R8_LEGACY[n] if n < 8 else R8[n]


def gpr_full(name):
    if name in ('ah', 'ch', 'dh', 'bh'):
        return R64[R8_LEGACY.index(name) - 4]
    for i in range(16):
        if name in (R64[i], R32[i], R16[i], R8[i]) or (i < 8 and name == R8_LEGACY[i]):
            return R64[i]
    return name


def vec(n, size):
    return {128: 'xmm', 256: 'ymm', 512: 'zmm'}[size] + str(n)


def full_of(name):
    if name.startswith(('xmm', 'ymm', 'zmm')):
        return 'zmm' + name[3:]
    if name.startswith('k') and name[1:].isdigit():
        return name
    return gpr_full(name)


# ---------------------------------------------------------------- tables
# Entry: (mnemonic, operand specs, attributes). Attributes (space separated):
#   acc=W,R,...   access per operand (default: first RW, rest R)
#   flow=...      ConditionalBranch / UnconditionalBranch / Call / Return / IndirectBranch / IndirectCall / Exception
#   cc            conditional: mnemonic gets CC[opcode & 15]
#   nosuffix      integer instruction printed without size suffix
#   vex           VEX/EVEX form (printed with v prefix already in mnemonic)
#   imm8s         8-bit immediate sign-extended to the operand size
#   evexonly      only valid as EVEX
#   t1            EVEX tuple1 scalar: disp8*N uses the element size
#   N=<n>         explicit disp8 scale for EVEX
#   regs=...      extra implicit registers: name:R|W|RW,...
#   mem=...       implicit memory accesses: rsi:R,rdi:W (base register:access, size = operand size)

ARITH = ['add', 'or', 'adc', 'sbb', 'and', 'sub', 'xor', 'cmp']
SHIFT = ['rol', 'ror', 'rcl', 'rcr', 'shl', 'shr', 'sal', 'sar']

ONE = {}
for i, m in enumerate(ARITH):
    acc = 'acc=R,R' if m == 'cmp' else 'acc=RW,R'
    ONE[i * 8 + 0] = (m, ['Eb', 'Gb'], acc)
    ONE[i * 8 + 1] = (m, ['Ev', 'Gv'], acc)
    ONE[i * 8 + 2] = (m, ['Gb', 'Eb'], acc)
    ONE[i * 8 + 3] = (m, ['Gv', 'Ev'], acc)
    ONE[i * 8 + 4] = (m, ['AL', 'Ib'], acc)
    ONE[i * 8 + 5] = (m, ['rAX', 'Iz'], acc)
for i in range(8):
    ONE[0x50 + i] = ('push', ['Zq'], 'acc=R regs=rsp:RW stack=push')
    ONE[0x58 + i] = ('pop', ['Zq'], 'acc=W regs=rsp:RW stack=pop')
    ONE[0x91 + i] = ('xchg', ['Zv', 'rAX'], 'acc=RW,RW') if i else None
    ONE[0xb0 + i] = ('mov', ['Zb', 'Ib'], 'acc=W,R')
    ONE[0xb8 + i] = ('mov', ['Zv', 'Iv'], 'acc=W,R')
ONE[0x63] = ('movslq', ['Gv', 'Ed'], 'acc=W,R nosuffix')
ONE[0x68] = ('push', ['Iz'], 'acc=R regs=rsp:RW stack=push')
ONE[0x69] = ('imul', ['Gv', 'Ev', 'Iz'], 'acc=W,R,R')
ONE[0x6a] = ('push', ['Ib'], 'acc=R regs=rsp:RW imm8s stack=push')
ONE[0x6b] = ('imul', ['Gv', 'Ev', 'Ib'], 'acc=W,R,R imm8s')
for i in range(16):
    ONE[0x70 + i] = ('j', ['Jb'], 'cc flow=ConditionalBranch nosuffix')
ONE[0x84] = ('test', ['Eb', 'Gb'], 'acc=R,R')
ONE[0x85] = ('test', ['Ev', 'Gv'], 'acc=R,R')
ONE[0x86] = ('xchg', ['Eb', 'Gb'], 'acc=RW,RW')
ONE[0x87] = ('xchg', ['Ev', 'Gv'], 'acc=RW,RW')
ONE[0x88] = ('mov', ['Eb', 'Gb'], 'acc=W,R')
ONE[0x89] = ('mov', ['Ev', 'Gv'], 'acc=W,R')
ONE[0x8a] = ('mov', ['Gb', 'Eb'], 'acc=W,R')
ONE[0x8b] = ('mov', ['Gv', 'Ev'], 'acc=W,R')
ONE[0x8d] = ('lea', ['Gv', 'M'], 'acc=W,A')
ONE[0x90] = ('nop', [], 'nosuffix')
ONE[0x98] = ('cbw', [], 'nosuffix regs=rax:RW')
ONE[0x99] = ('cwd', [], 'nosuffix regs=rax:R,rdx:W')
ONE[0xa4] = ('movsb', [], 'nosuffix regs=rsi:RW,rdi:RW mem=rsi:R,rdi:W msize=8')
ONE[0xa5] = ('movs', [], 'nosuffix regs=rsi:RW,rdi:RW mem=rsi:R,rdi:W')
ONE[0xa8] = ('test', ['AL', 'Ib'], 'acc=R,R')
ONE[0xa9] = ('test', ['rAX', 'Iz'], 'acc=R,R')
ONE[0xaa] = ('stosb', [], 'nosuffix regs=rdi:RW,rax:R mem=rdi:W msize=8')
ONE[0xab] = ('stos', [], 'nosuffix regs=rdi:RW,rax:R mem=rdi:W')
ONE[0xc2] = ('ret', ['Iw'], 'acc=R flow=Return nosuffix regs=rsp:RW stack=pop')
ONE[0xc3] = ('ret', [], 'flow=Return nosuffix regs=rsp:RW stack=pop')
ONE[0xc6] = ('mov', ['Eb', 'Ib'], 'acc=W,R')
ONE[0xc7] = ('mov', ['Ev', 'Iz'], 'acc=W,R')
ONE[0xc9] = ('leave', [], 'nosuffix regs=rsp:W,rbp:RW mem=rbp:R msize=64')
ONE[0xcc] = ('int3', [], 'nosuffix flow=Exception')
ONE[0xe8] = ('call', ['Jz'], 'flow=Call nosuffix regs=rsp:RW stack=push')
ONE[0xe9] = ('jmp', ['Jz'], 'flow=UnconditionalBranch nosuffix')
ONE[0xeb] = ('jmp', ['Jb'], 'flow=UnconditionalBranch nosuffix')
ONE[0xf4] = ('hlt', [], 'nosuffix')

# ModRM.reg groups in the one-byte map: opcode -> [entries by reg field]
GROUPS = {
    0x80: [(m, ['Eb', 'Ib'], 'acc=R,R' if m == 'cmp' else 'acc=RW,R') for m in ARITH],
    0x81: [(m, ['Ev', 'Iz'], 'acc=R,R' if m == 'cmp' else 'acc=RW,R') for m in ARITH],
    0x83: [(m, ['Ev', 'Ib'], ('acc=R,R' if m == 'cmp' else 'acc=RW,R') + ' imm8s') for m in ARITH],
    0xc0: [(m, ['Eb', 'Ib'], 'acc=RW,R') for m in SHIFT],
    0xc1: [(m, ['Ev', 'Ib'], 'acc=RW,R') for m in SHIFT],
    0xd0: [(m, ['Eb', 'ONE'], 'acc=RW,R') for m in SHIFT],
    0xd1: [(m, ['Ev', 'ONE'], 'acc=RW,R') for m in SHIFT],
    0xd2: [(m, ['Eb', 'CL'], 'acc=RW,R') for m in SHIFT],
    0xd3: [(m, ['Ev', 'CL'], 'acc=RW,R') for m in SHIFT],
    0xf6: [('test', ['Eb', 'Ib'], 'acc=R,R'), ('test', ['Eb', 'Ib'], 'acc=R,R'), ('not', ['Eb'], 'acc=RW'), ('neg', ['Eb'], 'acc=RW'),
           ('mul', ['Eb'], 'acc=R regs=rax:RW'), ('imul', ['Eb'], 'acc=R regs=rax:RW'), ('div', ['Eb'], 'acc=R regs=rax:RW'), ('idiv', ['Eb'], 'acc=R regs=rax:RW')],
    0xf7: [('test', ['Ev', 'Iz'], 'acc=R,R'), ('test', ['Ev', 'Iz'], 'acc=R,R'), ('not', ['Ev'], 'acc=RW'), ('neg', ['Ev'], 'acc=RW'),
           ('mul', ['Ev'], 'acc=R regs=rax:RW,rdx:W'), ('imul', ['Ev'], 'acc=R regs=rax:RW,rdx:W'), ('div', ['Ev'], 'acc=R regs=rax:RW,rdx:RW'), ('idiv', ['Ev'], 'acc=R regs=rax:RW,rdx:RW')],
    0xfe: [('inc', ['Eb'], 'acc=RW'), ('dec', ['Eb'], 'acc=RW')],
    0xff: [('inc', ['Ev'], 'acc=RW'), ('dec', ['Ev'], 'acc=RW'), ('call', ['Eq'], 'acc=R flow=IndirectCall regs=rsp:RW stack=push'), None,
           ('jmp', ['Eq'], 'acc=R flow=IndirectBranch'), None, ('push', ['Eq'], 'acc=R regs=rsp:RW stack=push'), None],
}

# 0F map. Key: (opcode, mandatory prefix) with prefix in ('', '66', 'f3', 'f2'); '*' = any (prefix ignored).
TWO = {}


def two(op, pfx, mnemonic, ops, attrs=''):
    TWO[(op, pfx)] = (mnemonic, ops, attrs)


two(0x05, '*', 'syscall', [], 'nosuffix')
two(0x0b, '*', 'ud2', [], 'nosuffix flow=Exception')
two(0x1f, '*', 'nop', ['Ev'], 'acc=A')
two(0x18, '*', 'prefetch', ['M'], 'acc=A nosuffix')
two(0xa2, '*', 'cpuid', [], 'nosuffix regs=rax:RW,rcx:RW,rbx:W,rdx:W')
for i in range(16):
    two(0x40 + i, '*', 'cmov', ['Gv', 'Ev'], 'cc acc=RW,R')
    two(0x80 + i, '*', 'j', ['Jz'], 'cc flow=ConditionalBranch nosuffix')
    two(0x90 + i, '*', 'set', ['Eb'], 'cc acc=W nosuffix')
two(0xa3, '*', 'bt', ['Ev', 'Gv'], 'acc=R,R')
two(0xab, '*', 'bts', ['Ev', 'Gv'], 'acc=RW,R')
two(0xb3, '*', 'btr', ['Ev', 'Gv'], 'acc=RW,R')
two(0xbb, '*', 'btc', ['Ev', 'Gv'], 'acc=RW,R')
two(0xa4, '*', 'shld', ['Ev', 'Gv', 'Ib'], 'acc=RW,R,R')
two(0xa5, '*', 'shld', ['Ev', 'Gv', 'CL'], 'acc=RW,R,R')
two(0xac, '*', 'shrd', ['Ev', 'Gv', 'Ib'], 'acc=RW,R,R')
two(0xad, '*', 'shrd', ['Ev', 'Gv', 'CL'], 'acc=RW,R,R')
two(0xaf, '*', 'imul', ['Gv', 'Ev'], 'acc=RW,R')
two(0xb0, '*', 'cmpxchg', ['Eb', 'Gb'], 'acc=RW,R regs=rax:RW')
two(0xb1, '*', 'cmpxchg', ['Ev', 'Gv'], 'acc=RW,R regs=rax:RW')
two(0xb6, '*', 'movzb', ['Gv', 'Eb'], 'acc=W,R')
two(0xb7, '*', 'movzw', ['Gv', 'Ew'], 'acc=W,R')
two(0xbe, '*', 'movsb', ['Gv', 'Eb'], 'acc=W,R')
two(0xbf, '*', 'movsw', ['Gv', 'Ew'], 'acc=W,R')
two(0xbc, '', 'bsf', ['Gv', 'Ev'], 'acc=W,R')
two(0xbc, 'f3', 'tzcnt', ['Gv', 'Ev'], 'acc=W,R')
two(0xbd, '', 'bsr', ['Gv', 'Ev'], 'acc=W,R')
two(0xbd, 'f3', 'lzcnt', ['Gv', 'Ev'], 'acc=W,R')
two(0xb8, 'f3', 'popcnt', ['Gv', 'Ev'], 'acc=W,R')
two(0xc0, '*', 'xadd', ['Eb', 'Gb'], 'acc=RW,RW')
two(0xc1, '*', 'xadd', ['Ev', 'Gv'], 'acc=RW,RW')
for i in range(8):
    two(0xc8 + i, '*', 'bswap', ['Zy'], 'acc=RW')
# SSE moves
two(0x10, '', 'movups', ['Vdq', 'Wdq'], 'acc=W,R')
two(0x10, '66', 'movupd', ['Vdq', 'Wdq'], 'acc=W,R')
two(0x10, 'f3', 'movss', ['Vdq', 'Wd'], 'acc=RW,R')
two(0x10, 'f2', 'movsd', ['Vdq', 'Wq'], 'acc=RW,R')
two(0x11, '', 'movups', ['Wdq', 'Vdq'], 'acc=W,R')
two(0x11, '66', 'movupd', ['Wdq', 'Vdq'], 'acc=W,R')
two(0x11, 'f3', 'movss', ['Wd', 'Vdq'], 'acc=W,R')
two(0x11, 'f2', 'movsd', ['Wq', 'Vdq'], 'acc=W,R')
two(0x12, '', 'movlps', ['Vdq', 'Mq'], 'acc=RW,R')
two(0x13, '', 'movlps', ['Mq', 'Vdq'], 'acc=W,R')
two(0x14, '', 'unpcklps', ['Vdq', 'Wdq'], 'acc=RW,R')
two(0x14, '66', 'unpcklpd', ['Vdq', 'Wdq'], 'acc=RW,R')
two(0x16, '', 'movhps', ['Vdq', 'Mq'], 'acc=RW,R')
two(0x17, '', 'movhps', ['Mq', 'Vdq'], 'acc=W,R')
two(0x28, '', 'movaps', ['Vdq', 'Wdq'], 'acc=W,R')
two(0x28, '66', 'movapd', ['Vdq', 'Wdq'], 'acc=W,R')
two(0x29, '', 'movaps', ['Wdq', 'Vdq'], 'acc=W,R')
two(0x29, '66', 'movapd', ['Wdq', 'Vdq'], 'acc=W,R')
two(0x2e, '', 'ucomiss', ['Vdq', 'Wd'], 'acc=R,R')
two(0x2e, '66', 'ucomisd', ['Vdq', 'Wq'], 'acc=R,R')
two(0x54, '', 'andps', ['Vdq', 'Wdq'], 'acc=RW,R')
two(0x54, '66', 'andpd', ['Vdq', 'Wdq'], 'acc=RW,R')
two(0x55, '', 'andnps', ['Vdq', 'Wdq'], 'acc=RW,R')
two(0x55, '66', 'andnpd', ['Vdq', 'Wdq'], 'acc=RW,R')
two(0x56, '', 'orps', ['Vdq', 'Wdq'], 'acc=RW,R')
two(0x56, '66', 'orpd', ['Vdq', 'Wdq'], 'acc=RW,R')
two(0x57, '', 'xorps', ['Vdq', 'Wdq'], 'acc=RW,R')
two(0x57, '66', 'xorpd', ['Vdq', 'Wdq'], 'acc=RW,R')
two(0x2b, '', 'movntps', ['Mdq', 'Vdq'], 'acc=W,R')
two(0xe7, '66', 'movntdq', ['Mdq', 'Vdq'], 'acc=W,R')
SSE2_66 = {0x60: 'punpcklbw', 0x61: 'punpcklwd', 0x62: 'punpckldq', 0x63: 'packsswb', 0x64: 'pcmpgtb', 0x65: 'pcmpgtw', 0x66: 'pcmpgtd',
           0x67: 'packuswb', 0x68: 'punpckhbw', 0x69: 'punpckhwd', 0x6a: 'punpckhdq', 0x6b: 'packssdw', 0x6c: 'punpcklqdq', 0x6d: 'punpckhqdq',
           0x74: 'pcmpeqb', 0x75: 'pcmpeqw', 0x76: 'pcmpeqd', 0xd1: 'psrlw', 0xd2: 'psrld', 0xd3: 'psrlq', 0xd4: 'paddq', 0xd5: 'pmullw',
           0xd8: 'psubusb', 0xd9: 'psubusw', 0xda: 'pminub', 0xdb: 'pand', 0xdc: 'paddusb', 0xdd: 'paddusw', 0xde: 'pmaxub', 0xdf: 'pandn',
           0xe0: 'pavgb', 0xe1: 'psraw', 0xe2: 'psrad', 0xe3: 'pavgw', 0xe4: 'pmulhuw', 0xe5: 'pmulhw', 0xe8: 'psubsb', 0xe9: 'psubsw',
           0xea: 'pminsw', 0xeb: 'por', 0xec: 'paddsb', 0xed: 'paddsw', 0xee: 'pmaxsw', 0xef: 'pxor', 0xf1: 'psllw', 0xf2: 'pslld', 0xf3: 'psllq',
           0xf4: 'pmuludq', 0xf5: 'pmaddwd', 0xf6: 'psadbw', 0xf8: 'psubb', 0xf9: 'psubw', 0xfa: 'psubd', 0xfb: 'psubq', 0xfc: 'paddb',
           0xfd: 'paddw', 0xfe: 'paddd'}
for op, m in SSE2_66.items():
    two(op, '66', m, ['Vdq', 'Wdq'], 'acc=RW,R')
two(0x6e, '66', 'movd', ['Vdq', 'Ey'], 'acc=W,R')      # movq with REX.W
two(0x7e, '66', 'movd', ['Ey', 'Vdq'], 'acc=W,R')
two(0x7e, 'f3', 'movq', ['Vdq', 'Wq'], 'acc=W,R')
two(0xd6, '66', 'movq', ['Wq', 'Vdq'], 'acc=W,R')
two(0x6f, '66', 'movdqa', ['Vdq', 'Wdq'], 'acc=W,R')
two(0x6f, 'f3', 'movdqu', ['Vdq', 'Wdq'], 'acc=W,R')
two(0x7f, '66', 'movdqa', ['Wdq', 'Vdq'], 'acc=W,R')
two(0x7f, 'f3', 'movdqu', ['Wdq', 'Vdq'], 'acc=W,R')
two(0x70, '66', 'pshufd', ['Vdq', 'Wdq', 'Ib'], 'acc=W,R,R')
two(0x70, 'f2', 'pshuflw', ['Vdq', 'Wdq', 'Ib'], 'acc=W,R,R')
two(0x70, 'f3', 'pshufhw', ['Vdq', 'Wdq', 'Ib'], 'acc=W,R,R')
two(0xc6, '', 'shufps', ['Vdq', 'Wdq', 'Ib'], 'acc=RW,R,R')
two(0xd7, '66', 'pmovmskb', ['Gd', 'Udq'], 'acc=W,R nosuffix')
two(0xc2, '', 'cmpps', ['Vdq', 'Wdq', 'Ib'], 'acc=RW,R,R')
# 0F 71/72/73 shift-by-immediate groups (66 prefix)
TWO_GROUPS = {
    0x71: [None, None, 'psrlw', None, 'psraw', None, 'psllw', None],
    0x72: [None, None, 'psrld', None, 'psrad', None, 'pslld', None],
    0x73: [None, None, 'psrlq', 'psrldq', None, None, 'psllq', 'pslldq'],
}
two(0x01, '*', None, [], '')  # handled specially (xgetbv etc.)
two(0xba, '*', None, [], '')  # group 8: bt/bts/btr/btc Ev,Ib
# 0F 38 map
THREE8 = {}


def t38(op, pfx, mnemonic, ops, attrs=''):
    THREE8[(op, pfx)] = (mnemonic, ops, attrs)


for op, m in {0x00: 'pshufb', 0x01: 'phaddw', 0x02: 'phaddd', 0x04: 'pmaddubsw', 0x08: 'psignb', 0x0b: 'pmulhrsw', 0x10: 'pblendvb', 0x1c: 'pabsb', 0x1d: 'pabsw', 0x1e: 'pabsd',
              0x28: 'pmuldq', 0x29: 'pcmpeqq', 0x2b: 'packusdw', 0x37: 'pcmpgtq', 0x38: 'pminsb', 0x39: 'pminsd', 0x3a: 'pminuw', 0x3b: 'pminud',
              0x3c: 'pmaxsb', 0x3d: 'pmaxsd', 0x3e: 'pmaxuw', 0x3f: 'pmaxud', 0x40: 'pmulld'}.items():
    t38(op, '66', m, ['Vdq', 'Wdq'], 'acc=RW,R')
t38(0x17, '66', 'ptest', ['Vdq', 'Wdq'], 'acc=R,R')
for op, m in {0x20: 'pmovsxbw', 0x21: 'pmovsxbd', 0x22: 'pmovsxbq', 0x23: 'pmovsxwd', 0x24: 'pmovsxwq', 0x25: 'pmovsxdq',
              0x30: 'pmovzxbw', 0x31: 'pmovzxbd', 0x32: 'pmovzxbq', 0x33: 'pmovzxwd', 0x34: 'pmovzxwq', 0x35: 'pmovzxdq'}.items():
    t38(op, '66', m, ['Vdq', 'Wq'], 'acc=W,R')
t38(0xf0, '', 'movbe', ['Gv', 'Mv'], 'acc=W,R')
t38(0xf1, '', 'movbe', ['Mv', 'Gv'], 'acc=W,R')
t38(0xf0, 'f2', 'crc32', ['Gy', 'Eb'], 'acc=RW,R')
t38(0xf1, 'f2', 'crc32', ['Gy', 'Ev'], 'acc=RW,R')
# 0F 3A map
THREE_A = {}


def t3a(op, pfx, mnemonic, ops, attrs=''):
    THREE_A[(op, pfx)] = (mnemonic, ops, attrs)


t3a(0x0f, '66', 'palignr', ['Vdq', 'Wdq', 'Ib'], 'acc=RW,R,R')
t3a(0x0e, '66', 'pblendw', ['Vdq', 'Wdq', 'Ib'], 'acc=RW,R,R')
t3a(0x14, '66', 'pextrb', ['Ed', 'Vdq', 'Ib'], 'acc=W,R,R nosuffix msize=8')
t3a(0x15, '66', 'pextrw', ['Ed', 'Vdq', 'Ib'], 'acc=W,R,R nosuffix msize=16')
t3a(0x16, '66', 'pextrd', ['Ey', 'Vdq', 'Ib'], 'acc=W,R,R nosuffix')   # pextrq with REX.W
t3a(0x17, '66', 'extractps', ['Ed', 'Vdq', 'Ib'], 'acc=W,R,R nosuffix')
t3a(0x20, '66', 'pinsrb', ['Vdq', 'Ed', 'Ib'], 'acc=RW,R,R nosuffix msize=8')
t3a(0x21, '66', 'insertps', ['Vdq', 'Wd', 'Ib'], 'acc=RW,R,R')
t3a(0x22, '66', 'pinsrd', ['Vdq', 'Ey', 'Ib'], 'acc=RW,R,R nosuffix')  # pinsrq with REX.W
t3a(0x60, '66', 'pcmpestrm', ['Vdq', 'Wdq', 'Ib'], 'acc=R,R,R regs=rax:R,rdx:R,xmm0:W')
t3a(0x61, '66', 'pcmpestri', ['Vdq', 'Wdq', 'Ib'], 'acc=R,R,R regs=rax:R,rdx:R,rcx:W')
t3a(0x62, '66', 'pcmpistrm', ['Vdq', 'Wdq', 'Ib'], 'acc=R,R,R regs=xmm0:W')
t3a(0x63, '66', 'pcmpistri', ['Vdq', 'Wdq', 'Ib'], 'acc=R,R,R regs=rcx:W')
t3a(0x44, '66', 'pclmulqdq', ['Vdq', 'Wdq', 'Ib'], 'acc=RW,R,R')

# VEX / EVEX maps. Key: (map, opcode, prefix, W) with W in ('*', 0, 1).
# Most AVX ops: dst = ModRM.reg, src1 = vvvv, src2 = r/m. Attribute 'kdst' marks EVEX forms writing a mask.
VEX = {}


def v(map_, op, pfx, w, mnemonic, ops, attrs='', evex=None):
    """evex: None = same entry for VEX and EVEX; 'only' = EVEX only; 'no' = VEX only; otherwise a (mnemonic, ops, attrs) for the EVEX form."""
    VEX[(map_, op, pfx, w)] = (mnemonic, ops, attrs + ' vex', evex)


AVX3 = {0x60: 'vpunpcklbw', 0x61: 'vpunpcklwd', 0x62: 'vpunpckldq', 0x63: 'vpacksswb', 0x67: 'vpackuswb', 0x68: 'vpunpckhbw',
        0x69: 'vpunpckhwd', 0x6a: 'vpunpckhdq', 0x6b: 'vpackssdw', 0x6c: 'vpunpcklqdq', 0x6d: 'vpunpckhqdq', 0xd4: 'vpaddq', 0xd5: 'vpmullw',
        0xd8: 'vpsubusb', 0xd9: 'vpsubusw', 0xda: 'vpminub', 0xdc: 'vpaddusb', 0xdd: 'vpaddusw', 0xde: 'vpmaxub', 0xe0: 'vpavgb',
        0xe3: 'vpavgw', 0xe4: 'vpmulhuw', 0xe5: 'vpmulhw', 0xe8: 'vpsubsb', 0xe9: 'vpsubsw', 0xea: 'vpminsw', 0xec: 'vpaddsb',
        0xed: 'vpaddsw', 0xee: 'vpmaxsw', 0xf4: 'vpmuludq', 0xf5: 'vpmaddwd', 0xf6: 'vpsadbw', 0xf8: 'vpsubb', 0xf9: 'vpsubw',
        0xfa: 'vpsubd', 0xfb: 'vpsubq', 0xfc: 'vpaddb', 0xfd: 'vpaddw', 0xfe: 'vpaddd', 0xd1: 'vpsrlw', 0xd2: 'vpsrld', 0xd3: 'vpsrlq',
        0xe1: 'vpsraw', 0xe2: 'vpsrad', 0xf1: 'vpsllw', 0xf2: 'vpslld', 0xf3: 'vpsllq'}
for op, m in AVX3.items():
    v(1, op, '66', '*', m, ['Vx', 'Hx', 'Wx'], 'acc=W,R,R')
# compares: VEX writes a vector, EVEX writes a mask
for op, m in {0x64: 'vpcmpgtb', 0x65: 'vpcmpgtw', 0x66: 'vpcmpgtd', 0x74: 'vpcmpeqb', 0x75: 'vpcmpeqw', 0x76: 'vpcmpeqd'}.items():
    v(1, op, '66', '*', m, ['Vx', 'Hx', 'Wx'], 'acc=W,R,R', evex=(m, ['Kr', 'Hx', 'Wx'], 'acc=W,R,R vex kdst'))
# logical: VEX vpxor etc; EVEX vpxord/q by W
for op, m in {0xdb: 'vpand', 0xdf: 'vpandn', 0xeb: 'vpor', 0xef: 'vpxor'}.items():
    v(1, op, '66', 0, m, ['Vx', 'Hx', 'Wx'], 'acc=W,R,R', evex=(m + 'd', ['Vx', 'Hx', 'Wx'], 'acc=W,R,R vex'))
    v(1, op, '66', 1, m, ['Vx', 'Hx', 'Wx'], 'acc=W,R,R', evex=(m + 'q', ['Vx', 'Hx', 'Wx'], 'acc=W,R,R vex'))
for pfx, m in (('', 'vmovups'), ('66', 'vmovupd')):
    v(1, 0x10, pfx, '*', m, ['Vx', 'Wx'], 'acc=W,R')
    v(1, 0x11, pfx, '*', m, ['Wx', 'Vx'], 'acc=W,R')
for pfx, m in (('', 'vmovaps'), ('66', 'vmovapd')):
    v(1, 0x28, pfx, '*', m, ['Vx', 'Wx'], 'acc=W,R')
    v(1, 0x29, pfx, '*', m, ['Wx', 'Vx'], 'acc=W,R')
v(1, 0x10, 'f3', '*', 'vmovss', ['Vdq', 'Hdq', 'Udq'], 'acc=W,R,R')
v(1, 0x10, 'f2', '*', 'vmovsd', ['Vdq', 'Hdq', 'Udq'], 'acc=W,R,R')
for pfx, m in (('', 'vxorps'), ('66', 'vxorpd')):
    v(1, 0x57, pfx, '*', m, ['Vx', 'Hx', 'Wx'], 'acc=W,R,R')
for pfx, m in (('', 'vandps'), ('66', 'vandpd')):
    v(1, 0x54, pfx, '*', m, ['Vx', 'Hx', 'Wx'], 'acc=W,R,R')
for pfx, m in (('', 'vorps'), ('66', 'vorpd')):
    v(1, 0x56, pfx, '*', m, ['Vx', 'Hx', 'Wx'], 'acc=W,R,R')
v(1, 0x6f, '66', '*', 'vmovdqa', ['Vx', 'Wx'], 'acc=W,R', evex='no')
v(1, 0x7f, '66', '*', 'vmovdqa', ['Wx', 'Vx'], 'acc=W,R', evex='no')
v(1, 0xe7, '66', '*', 'vmovntdq', ['Wx', 'Vx'], 'acc=W,R', evex='no')   # memory only: a non-temporal store
v(1, 0x6f, 'f3', '*', 'vmovdqu', ['Vx', 'Wx'], 'acc=W,R', evex='no')
v(1, 0x7f, 'f3', '*', 'vmovdqu', ['Wx', 'Vx'], 'acc=W,R', evex='no')
for pfx, w, m in (('66', 0, 'vmovdqa32'), ('66', 1, 'vmovdqa64'), ('f3', 0, 'vmovdqu32'), ('f3', 1, 'vmovdqu64'), ('f2', 0, 'vmovdqu8'), ('f2', 1, 'vmovdqu16')):
    v(1, 0x6f, pfx, w, m, ['Vx', 'Wx'], 'acc=W,R evexonly')
    v(1, 0x7f, pfx, w, m, ['Wx', 'Vx'], 'acc=W,R evexonly')
v(1, 0x6e, '66', 0, 'vmovd', ['Vdq', 'Ed'], 'acc=W,R t1')
v(1, 0x6e, '66', 1, 'vmovq', ['Vdq', 'Eq'], 'acc=W,R t1')
v(1, 0x7e, '66', 0, 'vmovd', ['Ed', 'Vdq'], 'acc=W,R t1')
v(1, 0x7e, '66', 1, 'vmovq', ['Eq', 'Vdq'], 'acc=W,R t1')
v(1, 0x7e, 'f3', '*', 'vmovq', ['Vdq', 'Wq'], 'acc=W,R t1')
v(1, 0xd6, '66', '*', 'vmovq', ['Wq', 'Vdq'], 'acc=W,R t1')
v(1, 0x70, '66', '*', 'vpshufd', ['Vx', 'Wx', 'Ib'], 'acc=W,R,R')
v(1, 0x70, 'f2', '*', 'vpshuflw', ['Vx', 'Wx', 'Ib'], 'acc=W,R,R')
v(1, 0x70, 'f3', '*', 'vpshufhw', ['Vx', 'Wx', 'Ib'], 'acc=W,R,R')
v(1, 0xd7, '66', '*', 'vpmovmskb', ['Gd', 'Ux'], 'acc=W,R nosuffix', evex='no')
v(1, 0x77, '', '*', 'vzeroupper', [], 'regs=' + ','.join('zmm%d:RW' % i for i in range(16)), evex='no')
v(1, 0xc6, '', '*', 'vshufps', ['Vx', 'Hx', 'Wx', 'Ib'], 'acc=W,R,R,R')
v(1, 0x14, '', '*', 'vunpcklps', ['Vx', 'Hx', 'Wx'], 'acc=W,R,R')
v(1, 0x12, '', '*', 'vmovlps', ['Vdq', 'Hdq', 'Mq'], 'acc=W,R,R t1')
v(1, 0x13, '', '*', 'vmovlps', ['Mq', 'Vdq'], 'acc=W,R t1')
v(1, 0x16, '', '*', 'vmovhps', ['Vdq', 'Hdq', 'Mq'], 'acc=W,R,R t1')
v(1, 0x17, '', '*', 'vmovhps', ['Mq', 'Vdq'], 'acc=W,R t1')
# VEX shift-by-immediate groups: 0F 71/72/73 /n with vvvv as destination
VEX_GROUPS = {
    0x71: [None, None, 'vpsrlw', None, 'vpsraw', None, 'vpsllw', None],
    0x72: [None, None, 'vpsrld', None, 'vpsrad', None, 'vpslld', None],
    0x73: [None, None, 'vpsrlq', 'vpsrldq', None, None, 'vpsllq', 'vpslldq'],
}
# mask instructions (VEX map 1)
for op, m in {0x41: 'kand', 0x42: 'kandn', 0x45: 'kor', 0x46: 'kxnor', 0x47: 'kxor', 0x4a: 'kadd'}.items():
    for pfx, w, sfx in (('', 0, 'w'), ('', 1, 'q'), ('66', 0, 'b'), ('66', 1, 'd')):
        v(1, op, pfx, w, m + sfx, ['Kr', 'Kv', 'Km'], 'acc=W,R,R nosuffix', evex='no')
for pfx, w, m in (('66', 0, 'kunpckbw'), ('', 0, 'kunpckwd'), ('', 1, 'kunpckdq')):  # the low halves of two masks into one
    v(1, 0x4b, pfx, w, m, ['Kr', 'Kv', 'Km'], 'acc=W,R,R nosuffix vlen=1', evex='no')
for op, m in {0x44: 'knot', 0x98: 'kortest', 0x99: 'ktest'}.items():
    for pfx, w, sfx in (('', 0, 'w'), ('', 1, 'q'), ('66', 0, 'b'), ('66', 1, 'd')):
        v(1, op, pfx, w, m + sfx, ['Kr', 'Km'], ('acc=W,R' if m == 'knot' else 'acc=R,R') + ' nosuffix', evex='no')
for pfx, w, sfx, size in (('', 0, 'w', 16), ('', 1, 'q', 64), ('66', 0, 'b', 8), ('66', 1, 'd', 32)):
    v(1, 0x90, pfx, w, 'kmov' + sfx, ['Kr', 'Km'], 'acc=W,R nosuffix msize=%d' % size, evex='no')
    v(1, 0x91, pfx, w, 'kmov' + sfx, ['Mk', 'Kr'], 'acc=W,R nosuffix msize=%d' % size, evex='no')
for pfx, w, sfx in (('', 0, 'w'), ('66', 0, 'b'), ('f2', 0, 'd'), ('f2', 1, 'q')):
    v(1, 0x92, pfx, w, 'kmov' + sfx, ['Kr', 'Ry'], 'acc=W,R nosuffix', evex='no')
    v(1, 0x93, pfx, w, 'kmov' + sfx, ['Gy', 'Km'], 'acc=W,R nosuffix', evex='no')
# map 2 (0F 38)
for op, m in {0x00: 'vpshufb', 0x04: 'vpmaddubsw', 0x0b: 'vpmulhrsw', 0x28: 'vpmuldq', 0x2b: 'vpackusdw', 0x38: 'vpminsb', 0x39: 'vpminsd',
              0x3a: 'vpminuw', 0x3b: 'vpminud', 0x3c: 'vpmaxsb', 0x3d: 'vpmaxsd', 0x3e: 'vpmaxuw', 0x3f: 'vpmaxud', 0x40: 'vpmulld',
              0x36: 'vpermd', 0x16: 'vpermps', 0x45: 'vpsrlvd', 0x47: 'vpsllvd', 0x46: 'vpsravd', 0x8c: 'vpmaskmovd'}.items():
    v(2, op, '66', '*', m, ['Vx', 'Hx', 'Wx'], 'acc=W,R,R')
v(2, 0x29, '66', '*', 'vpcmpeqq', ['Vx', 'Hx', 'Wx'], 'acc=W,R,R', evex=('vpcmpeqq', ['Kr', 'Hx', 'Wx'], 'acc=W,R,R vex kdst'))
v(2, 0x37, '66', '*', 'vpcmpgtq', ['Vx', 'Hx', 'Wx'], 'acc=W,R,R', evex=('vpcmpgtq', ['Kr', 'Hx', 'Wx'], 'acc=W,R,R vex kdst'))
v(2, 0x17, '66', '*', 'vptest', ['Vx', 'Wx'], 'acc=R,R', evex='no')
for op, m in {0x20: 'vpmovsxbw', 0x21: 'vpmovsxbd', 0x22: 'vpmovsxbq', 0x23: 'vpmovsxwd', 0x24: 'vpmovsxwq', 0x25: 'vpmovsxdq',
              0x30: 'vpmovzxbw', 0x31: 'vpmovzxbd', 0x32: 'vpmovzxbq', 0x33: 'vpmovzxwd', 0x34: 'vpmovzxwq', 0x35: 'vpmovzxdq'}.items():
    v(2, op, '66', '*', m, ['Vx', 'Whalf'], 'acc=W,R')
v(2, 0x78, '66', '*', 'vpbroadcastb', ['Vx', 'Wb'], 'acc=W,R t1')
v(2, 0x79, '66', '*', 'vpbroadcastw', ['Vx', 'Ww'], 'acc=W,R t1')
v(2, 0x58, '66', '*', 'vpbroadcastd', ['Vx', 'Wd'], 'acc=W,R t1')
v(2, 0x59, '66', '*', 'vpbroadcastq', ['Vx', 'Wq'], 'acc=W,R t1')
v(2, 0x7a, '66', '*', 'vpbroadcastb', ['Vx', 'Rd'], 'acc=W,R evexonly')
v(2, 0x7b, '66', '*', 'vpbroadcastw', ['Vx', 'Rd'], 'acc=W,R evexonly')
v(2, 0x7c, '66', 0, 'vpbroadcastd', ['Vx', 'Rd'], 'acc=W,R evexonly')
v(2, 0x7c, '66', 1, 'vpbroadcastq', ['Vx', 'Rq'], 'acc=W,R evexonly')
v(2, 0x1a, '66', '*', 'vbroadcastf128', ['Vqq', 'Mdq'], 'acc=W,R', evex='no')
v(2, 0x5a, '66', '*', 'vbroadcasti128', ['Vqq', 'Mdq'], 'acc=W,R', evex='no')
v(2, 0x26, '66', 0, 'vptestmb', ['Kr', 'Hx', 'Wx'], 'acc=W,R,R evexonly kdst')
v(2, 0x26, '66', 1, 'vptestmw', ['Kr', 'Hx', 'Wx'], 'acc=W,R,R evexonly kdst')
v(2, 0x26, 'f3', 0, 'vptestnmb', ['Kr', 'Hx', 'Wx'], 'acc=W,R,R evexonly kdst')
v(2, 0x26, 'f3', 1, 'vptestnmw', ['Kr', 'Hx', 'Wx'], 'acc=W,R,R evexonly kdst')
v(2, 0x27, '66', 0, 'vptestmd', ['Kr', 'Hx', 'Wx'], 'acc=W,R,R evexonly kdst')
v(2, 0x27, '66', 1, 'vptestmq', ['Kr', 'Hx', 'Wx'], 'acc=W,R,R evexonly kdst')
v(2, 0x27, 'f3', 0, 'vptestnmd', ['Kr', 'Hx', 'Wx'], 'acc=W,R,R evexonly kdst')
v(2, 0x27, 'f3', 1, 'vptestnmq', ['Kr', 'Hx', 'Wx'], 'acc=W,R,R evexonly kdst')
v(2, 0x28, 'f3', 0, 'vpmovm2b', ['Vx', 'Km'], 'acc=W,R evexonly')
v(2, 0x28, 'f3', 1, 'vpmovm2w', ['Vx', 'Km'], 'acc=W,R evexonly')
v(2, 0x38, 'f3', 0, 'vpmovm2d', ['Vx', 'Km'], 'acc=W,R evexonly')
v(2, 0x38, 'f3', 1, 'vpmovm2q', ['Vx', 'Km'], 'acc=W,R evexonly')
v(2, 0x29, 'f3', 0, 'vpmovb2m', ['Kr', 'Ux'], 'acc=W,R evexonly kdst')
v(2, 0x29, 'f3', 1, 'vpmovw2m', ['Kr', 'Ux'], 'acc=W,R evexonly kdst')
v(2, 0x39, 'f3', 0, 'vpmovd2m', ['Kr', 'Ux'], 'acc=W,R evexonly kdst')
v(2, 0x39, 'f3', 1, 'vpmovq2m', ['Kr', 'Ux'], 'acc=W,R evexonly kdst')
# BMI (VEX map 2, GPR operands)
v(2, 0xf2, '', '*', 'andn', ['Gy', 'By', 'Ey'], 'acc=W,R,R', evex='no')
v(2, 0xf5, '', '*', 'bzhi', ['Gy', 'Ey', 'By'], 'acc=W,R,R', evex='no')
v(2, 0xf5, 'f2', '*', 'pdep', ['Gy', 'By', 'Ey'], 'acc=W,R,R', evex='no')
v(2, 0xf5, 'f3', '*', 'pext', ['Gy', 'By', 'Ey'], 'acc=W,R,R', evex='no')
v(2, 0xf6, 'f2', '*', 'mulx', ['Gy', 'By', 'Ey'], 'acc=W,W,R regs=rdx:R', evex='no')
v(2, 0xf7, '', '*', 'bextr', ['Gy', 'Ey', 'By'], 'acc=W,R,R', evex='no')
v(2, 0xf7, '66', '*', 'shlx', ['Gy', 'Ey', 'By'], 'acc=W,R,R', evex='no')
v(2, 0xf7, 'f3', '*', 'sarx', ['Gy', 'Ey', 'By'], 'acc=W,R,R', evex='no')
v(2, 0xf7, 'f2', '*', 'shrx', ['Gy', 'Ey', 'By'], 'acc=W,R,R', evex='no')
BMI_GROUP17 = [None, 'blsr', 'blsmsk', 'blsi', None, None, None, None]  # VEX map 2 F3 /n: By, Ey
# map 3 (0F 3A)
v(3, 0x0f, '66', '*', 'vpalignr', ['Vx', 'Hx', 'Wx', 'Ib'], 'acc=W,R,R,R')
v(3, 0x0e, '66', '*', 'vpblendw', ['Vx', 'Hx', 'Wx', 'Ib'], 'acc=W,R,R,R', evex='no')
v(3, 0x02, '66', '*', 'vpblendd', ['Vx', 'Hx', 'Wx', 'Ib'], 'acc=W,R,R,R', evex='no')
v(3, 0x4c, '66', '*', 'vpblendvb', ['Vx', 'Hx', 'Wx', 'Lx'], 'acc=W,R,R,R', evex='no')
v(3, 0x00, '66', 1, 'vpermq', ['Vqq', 'Wqq', 'Ib'], 'acc=W,R,R')   # W1 only: the encoder keeps it
v(3, 0x01, '66', 1, 'vpermpd', ['Vqq', 'Wqq', 'Ib'], 'acc=W,R,R')
v(3, 0x14, '66', '*', 'vpextrb', ['Ed', 'Vdq', 'Ib'], 'acc=W,R,R nosuffix msize=8 t1')
v(3, 0x15, '66', '*', 'vpextrw', ['Ed', 'Vdq', 'Ib'], 'acc=W,R,R nosuffix msize=16 t1')
v(3, 0x16, '66', 0, 'vpextrd', ['Ed', 'Vdq', 'Ib'], 'acc=W,R,R nosuffix t1')
v(3, 0x16, '66', 1, 'vpextrq', ['Eq', 'Vdq', 'Ib'], 'acc=W,R,R nosuffix t1')
v(3, 0x20, '66', '*', 'vpinsrb', ['Vdq', 'Hdq', 'Ed', 'Ib'], 'acc=W,R,R,R nosuffix msize=8 t1')
v(3, 0x22, '66', 0, 'vpinsrd', ['Vdq', 'Hdq', 'Ed', 'Ib'], 'acc=W,R,R,R nosuffix t1')
v(3, 0x22, '66', 1, 'vpinsrq', ['Vdq', 'Hdq', 'Eq', 'Ib'], 'acc=W,R,R,R nosuffix t1')
v(3, 0x38, '66', '*', 'vinserti128', ['Vqq', 'Hqq', 'Wdq', 'Ib'], 'acc=W,R,R,R', evex=('vinserti32x4', ['Vx', 'Hx', 'Wdq', 'Ib'], 'acc=W,R,R,R vex N=16'))
v(3, 0x39, '66', '*', 'vextracti128', ['Wdq', 'Vqq', 'Ib'], 'acc=W,R,R', evex=('vextracti32x4', ['Wdq', 'Vx', 'Ib'], 'acc=W,R,R vex N=16'))
v(3, 0x18, '66', '*', 'vinsertf128', ['Vqq', 'Hqq', 'Wdq', 'Ib'], 'acc=W,R,R,R', evex=('vinsertf32x4', ['Vx', 'Hx', 'Wdq', 'Ib'], 'acc=W,R,R,R vex N=16'))
v(3, 0x19, '66', '*', 'vextractf128', ['Wdq', 'Vqq', 'Ib'], 'acc=W,R,R', evex=('vextractf32x4', ['Wdq', 'Vx', 'Ib'], 'acc=W,R,R vex N=16'))
v(3, 0x46, '66', '*', 'vperm2i128', ['Vqq', 'Hqq', 'Wqq', 'Ib'], 'acc=W,R,R,R', evex='no')
v(3, 0x3e, '66', 0, 'vpcmpub', ['Kr', 'Hx', 'Wx', 'Ib'], 'acc=W,R,R,R evexonly kdst pred')
v(3, 0x3e, '66', 1, 'vpcmpuw', ['Kr', 'Hx', 'Wx', 'Ib'], 'acc=W,R,R,R evexonly kdst pred')
v(3, 0x3f, '66', 0, 'vpcmpb', ['Kr', 'Hx', 'Wx', 'Ib'], 'acc=W,R,R,R evexonly kdst pred')
v(3, 0x3f, '66', 1, 'vpcmpw', ['Kr', 'Hx', 'Wx', 'Ib'], 'acc=W,R,R,R evexonly kdst pred')
v(3, 0x1e, '66', 0, 'vpcmpud', ['Kr', 'Hx', 'Wx', 'Ib'], 'acc=W,R,R,R evexonly kdst pred')
v(3, 0x1e, '66', 1, 'vpcmpuq', ['Kr', 'Hx', 'Wx', 'Ib'], 'acc=W,R,R,R evexonly kdst pred')
v(3, 0x1f, '66', 0, 'vpcmpd', ['Kr', 'Hx', 'Wx', 'Ib'], 'acc=W,R,R,R evexonly kdst pred')
v(3, 0x1f, '66', 1, 'vpcmpq', ['Kr', 'Hx', 'Wx', 'Ib'], 'acc=W,R,R,R evexonly kdst pred')
v(3, 0x25, '66', 0, 'vpternlogd', ['Vx', 'Hx', 'Wx', 'Ib'], 'acc=RW,R,R,R evexonly')
v(3, 0x25, '66', 1, 'vpternlogq', ['Vx', 'Hx', 'Wx', 'Ib'], 'acc=RW,R,R,R evexonly')
v(3, 0x60, '66', '*', 'vpcmpestrm', ['Vdq', 'Wdq', 'Ib'], 'acc=R,R,R regs=rax:R,rdx:R,xmm0:W', evex='no')
v(3, 0x61, '66', '*', 'vpcmpestri', ['Vdq', 'Wdq', 'Ib'], 'acc=R,R,R regs=rax:R,rdx:R,rcx:W', evex='no')
v(3, 0x62, '66', '*', 'vpcmpistrm', ['Vdq', 'Wdq', 'Ib'], 'acc=R,R,R regs=xmm0:W', evex='no')
v(3, 0x63, '66', '*', 'vpcmpistri', ['Vdq', 'Wdq', 'Ib'], 'acc=R,R,R regs=rcx:W', evex='no')
# mask shifts (VEX map 3)
for op, m in {0x30: 'kshiftr', 0x32: 'kshiftl'}.items():
    for w, sfx in ((0, 'b'), (1, 'w')):
        v(3, op, '66', w, m + sfx, ['Kr', 'Km', 'Ib'], 'acc=W,R,R nosuffix', evex='no')
    for w, sfx in ((0, 'd'), (1, 'q')):
        v(3, op + 1, '66', w, m + sfx, ['Kr', 'Km', 'Ib'], 'acc=W,R,R nosuffix', evex='no')


# ---------------------------------------------------------------- decoder

class Ctx:
    __slots__ = ('code', 'base', 'pos', 'start', 'prefixes', 'rex', 'seg', 'opsize', 'rep', 'vex', 'evex', 'W', 'vvvv', 'L', 'pp',
                 'mask', 'zero', 'bcast', 'R', 'X', 'B', 'R2', 'V2', 'modrm', 'map', 'opcode', 'mnemonic', 'attrs', 'pfx', 'vex2', 'lock')

    def __init__(self, code, base, pos):
        self.code, self.base, self.pos, self.start = code, base, pos, pos
        self.prefixes, self.rex, self.seg, self.opsize, self.rep = [], None, None, False, None
        self.vex = self.evex = False
        self.W = self.L = self.pp = 0
        self.vvvv = 0
        self.mask, self.zero, self.bcast = 0, False, False
        self.R = self.X = self.B = self.R2 = self.V2 = 0
        self.modrm = None
        self.pfx = ''      # mandatory prefix of the table entry ('' , '66', 'f3', 'f2')
        self.vex2 = False
        self.lock = False

    def byte(self):
        if self.pos >= len(self.code):
            raise DecodeError('truncated instruction at %#x' % (self.base + self.start))
        b = self.code[self.pos]
        self.pos += 1
        return b

    def imm(self, size):
        n = size // 8
        if self.pos + n > len(self.code):
            raise DecodeError('truncated immediate at %#x' % (self.base + self.start))
        val = int.from_bytes(self.code[self.pos:self.pos + n], 'little', signed=True)
        self.pos += n
        return val


def parse_prefixes(c):
    while True:
        b = c.code[c.pos] if c.pos < len(c.code) else None
        if b is None:
            raise DecodeError('truncated')
        if b == 0x66:
            c.opsize = True
        elif b in (0xf2, 0xf3):
            c.rep = b
        elif b == 0xf0:
            c.lock = True
        elif b in SEG:
            c.seg = SEG[b]
        elif b == 0x67:
            raise DecodeError('address-size prefix')
        else:
            break
        c.prefixes.append(b)
        c.pos += 1
    if 0x40 <= b <= 0x4f:
        c.rex = b
        c.W, c.R, c.X, c.B = (b >> 3) & 1, (b >> 2) & 1, (b >> 1) & 1, b & 1
        c.pos += 1


def mandatory_prefix(c):
    if c.rep == 0xf3: return 'f3'
    if c.rep == 0xf2: return 'f2'
    if c.opsize: return '66'
    return ''


def opsize_bits(c):
    if c.W: return 64
    if c.opsize: return 16
    return 32


class Operand:
    __slots__ = ('kind', 'reg', 'imm', 'target', 'mem', 'size', 'access', 'text', 'full', 'implicit', 'field_pos', 'literal')

    def __init__(self, kind, **kw):
        self.kind = kind
        self.reg = self.imm = self.target = self.mem = None
        self.size = 0
        self.access = R
        self.text = ''
        self.implicit = False
        self.field_pos = None
        self.literal = None
        for k, val in kw.items():
            setattr(self, k, val)


def modrm_fields(c):
    if c.modrm is None:
        c.modrm = c.byte()
    m = c.modrm
    return m >> 6, (m >> 3) & 7, m & 7


def decode_memory(c, mod, rm, size, disp8_scale=1):
    """Memory operand after ModRM (mod != 3). Returns Operand(kind='Memory')."""
    base = index = None
    scale = 1
    disp = 0
    rip_rel = False
    disp_pos = None
    if rm == 4:
        sib = c.byte()
        ss, idx, bs = sib >> 6, (sib >> 3) & 7, sib & 7
        idx |= c.X << 3
        if idx != 4:
            index = R64[idx]
            scale = 1 << ss
        if bs == 5 and mod == 0:
            disp_pos = c.pos - c.start
            disp = c.imm(32)
        else:
            base = R64[bs | (c.B << 3)]
    elif rm == 5 and mod == 0:
        rip_rel = True
        disp_pos = c.pos - c.start
        disp = c.imm(32)
    else:
        base = R64[rm | (c.B << 3)]
    if mod == 1:
        disp_pos = c.pos - c.start
        disp = c.imm(8) * disp8_scale
    elif mod == 2:
        disp_pos = c.pos - c.start
        disp = c.imm(32)
    o = Operand('Memory', size=size)
    o.mem = {'base': base, 'index': index, 'scale': scale, 'disp': disp, 'rip_rel': rip_rel, 'disp_size': 8 if mod == 1 else 32 if (mod == 2 or disp_pos is not None) else 0}
    o.field_pos = disp_pos
    return o


def vec_width(c):
    return 512 if c.L == 2 else 256 if c.L == 1 else 128


def decode_operands(c, specs, attrs, end_hint):
    """Build Operand objects from specs. end_hint is unused (text needs the final end; done later)."""
    ops = []
    osz = opsize_bits(c)
    ysz = 64 if c.W else 32
    vw = vec_width(c)
    disp8 = 1
    if c.evex:
        disp8 = vw // 8
        if 't1' in attrs:
            disp8 = None  # set from the memory operand size below
        for a in attrs.split():
            if a.startswith('N='):
                disp8 = int(a[2:])
    for spec in specs:
        o = None
        if spec in ('Eb', 'Ew', 'Ed', 'Eq', 'Ev', 'Ey', 'M', 'Mq', 'Mdq', 'Mv', 'Mk'):
            size = {'Eb': 8, 'Ew': 16, 'Ed': 32, 'Eq': 64, 'Ev': osz, 'Ey': ysz, 'M': 0, 'Mq': 64, 'Mdq': 128, 'Mv': osz, 'Mk': 0}[spec]
            msize = size  # pinsrb & co: a 32-bit register but an 8-bit memory operand
            for a in attrs.split():
                if a.startswith('msize='):
                    msize = int(a[6:])
            mod, reg, rm = modrm_fields(c)
            if mod == 3:
                if spec.startswith('M'):
                    raise DecodeError('register form of memory-only operand')
                o = Operand('Register', reg=gpr(rm | (c.B << 3), size, c.rex if not (c.vex or c.evex) else 0), size=size)
            else:
                o = decode_memory(c, mod, rm, msize, disp8 if disp8 else max(1, msize // 8))
        elif spec in ('Gb', 'Gw', 'Gd', 'Gq', 'Gv', 'Gy'):
            size = {'Gb': 8, 'Gw': 16, 'Gd': 32, 'Gq': 64, 'Gv': osz, 'Gy': ysz}[spec]
            mod, reg, rm = modrm_fields(c)
            o = Operand('Register', reg=gpr(reg | (c.R << 3), size, c.rex if not (c.vex or c.evex) else 0), size=size)
        elif spec in ('Rd', 'Rq', 'Ry'):  # GPR in r/m, register only
            size = {'Rd': 32, 'Rq': 64, 'Ry': ysz}[spec]
            mod, reg, rm = modrm_fields(c)
            o = Operand('Register', reg=gpr(rm | (c.B << 3), size, 0), size=size)
        elif spec in ('By',):
            o = Operand('Register', reg=gpr(c.vvvv, ysz, 0), size=ysz)
        elif spec in ('Zb', 'Zv', 'Zq', 'Zy'):
            size = {'Zb': 8, 'Zv': osz, 'Zq': 64, 'Zy': ysz}[spec]
            o = Operand('Register', reg=gpr((c.opcode & 7) | (c.B << 3), size, c.rex), size=size)
        elif spec == 'AL':
            o = Operand('Register', reg='al', size=8)
        elif spec == 'rAX':
            o = Operand('Register', reg=gpr(0, osz, c.rex), size=osz)
        elif spec == 'CL':
            o = Operand('Register', reg='cl', size=8)
        elif spec == 'ONE':
            o = Operand('Immediate', imm=1, size=8, implicit=True)
        elif spec in ('Ib', 'Iw', 'Id', 'Iz', 'Iv'):
            size = {'Ib': 8, 'Iw': 16, 'Id': 32, 'Iz': 16 if osz == 16 else 32, 'Iv': osz}[spec]
            pos = c.pos - c.start
            val = c.imm(size)
            o = Operand('Immediate', imm=val, size=size, field_pos=pos)
        elif spec in ('Jb', 'Jz'):
            size = 8 if spec == 'Jb' else 32
            pos = c.pos - c.start
            o = Operand('NearBranch', imm=c.imm(size), size=size, field_pos=pos)
        elif spec in ('Vx', 'Vdq', 'Vqq'):
            size = {'Vx': vw, 'Vdq': 128, 'Vqq': 256}[spec]
            mod, reg, rm = modrm_fields(c)
            o = Operand('Register', reg=vec(reg | (c.R << 3) | (c.R2 << 4), size), size=size)
        elif spec in ('Hx', 'Hdq', 'Hqq'):
            size = {'Hx': vw, 'Hdq': 128, 'Hqq': 256}[spec]
            o = Operand('Register', reg=vec(c.vvvv | (c.V2 << 4), size), size=size)
        elif spec in ('Wx', 'Wdq', 'Wqq', 'Wq', 'Wd', 'Ww', 'Wb', 'Whalf', 'Ux', 'Udq'):
            size = {'Wx': vw, 'Wdq': 128, 'Wqq': 256, 'Wq': 64, 'Wd': 32, 'Ww': 16, 'Wb': 8, 'Whalf': vw // 2, 'Ux': vw, 'Udq': 128}[spec]
            mod, reg, rm = modrm_fields(c)
            if mod == 3:
                rsize = size if size >= 128 else 128
                if spec == 'Whalf':
                    rsize = 128 if vw <= 256 else 256
                o = Operand('Register', reg=vec(rm | (c.B << 3) | (c.X << 4), rsize), size=rsize)
            else:
                if spec.startswith('U'):
                    raise DecodeError('memory form of register-only operand')
                scale = disp8 if disp8 else max(1, size // 8)
                if spec == 'Whalf' and c.evex:
                    scale = size // 8
                o = decode_memory(c, mod, rm, size, scale)
        elif spec == 'Lx':  # register in imm8[7:4]
            val = c.imm(8)
            o = Operand('Register', reg=vec((val >> 4) & 15, vw), size=vw)
        elif spec == 'Kr':
            mod, reg, rm = modrm_fields(c)
            o = Operand('Register', reg='k%d' % reg, size=64)
        elif spec == 'Km':
            mod, reg, rm = modrm_fields(c)
            if mod == 3:
                o = Operand('Register', reg='k%d' % rm, size=64)
            else:
                size = next((int(a[6:]) for a in attrs.split() if a.startswith('msize=')), 0)
                o = decode_memory(c, mod, rm, size, 1)
        elif spec == 'Kv':
            o = Operand('Register', reg='k%d' % (c.vvvv & 7), size=64)
        else:
            raise DecodeError('operand spec ' + spec)
        ops.append(o)
    return ops


def lookup(c):
    """Find the table entry; consumes opcode bytes (and ModRM for groups). Returns (mnemonic, specs, attrs)."""
    if c.vex or c.evex:
        pfx = ['', '66', 'f3', 'f2'][c.pp]
        c.pfx = pfx
        for w in (c.W, '*'):
            e = VEX.get((c.map, c.opcode, pfx, w))
            if e is None:
                continue
            mnemonic, specs, attrs, evex = e
            if c.evex:
                if evex == 'no':
                    continue
                if isinstance(evex, tuple):
                    mnemonic, specs, attrs = evex
            else:
                if 'evexonly' in attrs or evex == 'only':
                    continue
            return mnemonic, specs, attrs
        if c.map == 1 and c.opcode in VEX_GROUPS and pfx == '66':
            mod, reg, rm = modrm_fields(c)
            m = VEX_GROUPS[c.opcode][reg]
            if m:
                return m, ['Hx', 'Ux', 'Ib'], 'acc=W,R,R vex'
        if c.map == 2 and c.opcode == 0xf3 and pfx == '' and not c.evex:
            mod, reg, rm = modrm_fields(c)
            m = BMI_GROUP17[reg]
            if m:
                return m, ['By', 'Ey'], 'acc=W,R vex'
        raise DecodeError('unknown %s map %d opcode %02x pfx %r W%d' % ('EVEX' if c.evex else 'VEX', c.map, c.opcode, pfx, c.W))
    op = c.opcode
    if c.map == 0:
        if op in GROUPS:
            mod, reg, rm = modrm_fields(c)
            e = GROUPS[op][reg]
            if e is None:
                raise DecodeError('unknown group opcode %02x /%d' % (op, reg))
            return e
        e = ONE.get(op)
        if e is None:
            raise DecodeError('unknown opcode %02x' % op)
        return e
    pfx = mandatory_prefix(c)
    table = {1: TWO, 2: THREE8, 3: THREE_A}[c.map]
    if c.map == 1:
        if op == 0x01:
            b = c.byte()
            if b == 0xd0:
                c.opcode = 0x01d0   # two-byte opcode 0F 01 D0
                return 'xgetbv', [], 'nosuffix regs=rcx:R,rax:W,rdx:W'
            raise DecodeError('unknown 0F 01 %02x' % b)
        if op == 0xae and c.pos < len(c.code) and c.code[c.pos] == 0xf8:
            c.byte()
            c.opcode = 0xaef8   # 0F AE F8: sfence
            return 'sfence', [], 'nosuffix'
        if op == 0x1e:
            # F3 0F 1E FA/FB: endbr64/endbr32 (-fcf-protection); anything else is a reserved nop
            if pfx == 'f3' and c.pos < len(c.code) and c.code[c.pos] in (0xfa, 0xfb):
                c.rep = None
                c.pfx = 'f3'
                return ('endbr64' if c.byte() == 0xfa else 'endbr32'), [], 'nosuffix'
            return 'nop', ['Ev'], 'acc=A'
        if op == 0xba:
            mod, reg, rm = modrm_fields(c)
            m = [None, None, None, None, 'bt', 'bts', 'btr', 'btc'][reg]
            if m:
                return m, ['Ev', 'Ib'], 'acc=R,R' if m == 'bt' else 'acc=RW,R'
            raise DecodeError('unknown group 8 /%d' % reg)
        if op in TWO_GROUPS and pfx == '66':
            mod, reg, rm = modrm_fields(c)
            m = TWO_GROUPS[op][reg]
            if m:
                c.pfx = '66'
                c.opsize = False
                return m, ['Udq', 'Ib'], 'acc=RW,R'
    for key in ((op, pfx), (op, '*')):
        e = table.get(key)
        if e is not None and e[0] is not None:
            # a 66 prefix used as a mandatory prefix is not an operand-size prefix
            if key[1] == '66':
                c.opsize = False
            if key[1] in ('f3', 'f2'):
                c.rep = None
            if key[1] != '*':
                c.pfx = key[1]
            return e
    raise DecodeError('unknown opcode 0F %s%02x' % ({1: '', 2: '38 ', 3: '3A '}[c.map], op))


def decode_one(code, base, pos):
    c = Ctx(code, base, pos)
    parse_prefixes(c)
    b = c.byte()
    if b in (0xc4, 0xc5) :
        c.vex = True
        if b == 0xc5:
            c.vex2 = True
            p = c.byte()
            c.R = (~p >> 7) & 1
            c.vvvv = (~p >> 3) & 15
            c.L = (p >> 2) & 1
            c.pp = p & 3
            c.map = 1
        else:
            p1, p2 = c.byte(), c.byte()
            c.R, c.X, c.B = (~p1 >> 7) & 1, (~p1 >> 6) & 1, (~p1 >> 5) & 1
            c.map = p1 & 31
            c.W = p2 >> 7
            c.vvvv = (~p2 >> 3) & 15
            c.L = (p2 >> 2) & 1
            c.pp = p2 & 3
        c.opcode = c.byte()
    elif b == 0x62:
        c.evex = True
        p0, p1, p2 = c.byte(), c.byte(), c.byte()
        c.R, c.X, c.B, c.R2 = (~p0 >> 7) & 1, (~p0 >> 6) & 1, (~p0 >> 5) & 1, (~p0 >> 4) & 1
        c.map = p0 & 7
        c.W = p1 >> 7
        c.vvvv = (~p1 >> 3) & 15
        c.pp = p1 & 3
        c.zero = bool(p2 >> 7)
        c.L = (p2 >> 5) & 3
        c.bcast = bool((p2 >> 4) & 1)
        c.V2 = (~p2 >> 3) & 1
        c.mask = p2 & 7
        c.opcode = c.byte()
    else:
        c.map = 0
        c.opcode = b
        if b == 0x0f:
            b2 = c.byte()
            if b2 == 0x38:
                c.map, c.opcode = 2, c.byte()
            elif b2 == 0x3a:
                c.map, c.opcode = 3, c.byte()
            else:
                c.map, c.opcode = 1, b2
    mnemonic, specs, attrs = lookup(c)
    if c.vex and c.X and all(not s.startswith(('Wx', 'Wdq', 'Wqq', 'Ux', 'Udq')) for s in specs):
        pass
    ops = decode_operands(c, specs, attrs, None)
    return finish(c, mnemonic, specs, attrs, ops)


def finish(c, mnemonic, specs, attrs, ops):
    attr = attrs.split()
    acc = next((a[4:].split(',') for a in attr if a.startswith('acc=')), None)
    end = c.base + c.pos
    off = c.base + c.start
    length = c.pos - c.start
    osz = opsize_bits(c)
    cc = CC[c.opcode & 15] if 'cc' in attr else None
    if cc:
        mnemonic += cc
    # mnemonic aliases by operand size / W (the encoder maps them back, see ALIASES)
    if mnemonic == 'cbw':
        mnemonic = {16: 'cbtw', 32: 'cwtl', 64: 'cltq'}[osz]
    elif mnemonic == 'cwd':
        mnemonic = {16: 'cwtd', 32: 'cltd', 64: 'cqto'}[osz]
    elif mnemonic in ('movs', 'stos'):
        mnemonic += SUFFIX[osz]
    elif mnemonic == 'movd' and c.W and not c.vex:
        mnemonic = 'movq'
    elif mnemonic in ('pextrd', 'pinsrd') and c.W:
        mnemonic = mnemonic[:-1] + 'q'
    elif mnemonic == 'mov' and ops and ops[1].kind == 'Immediate' and ops[1].size == 64:
        mnemonic = 'movabs'
    # accesses
    for k, o in enumerate(ops):
        a = acc[k] if acc and k < len(acc) else ('RW' if k == 0 else 'R')
        o.access = {'R': R, 'W': W, 'RW': RW, 'A': 'A'}[a]
    # EVEX merge-masking makes the destination read-modify-write
    if c.evex and c.mask and not c.zero and ops and ops[0].kind == 'Register' and 'kdst' not in attr and ops[0].access == W:
        ops[0].access = RW
    regs = []
    mems = []

    def add_reg(name, access):
        regs.append({'name': name, 'full': full_of(name), 'access': access})

    zero_idiom = (mnemonic in ZERO_IDIOMS and len(ops) >= 2 and all(o.kind == 'Register' for o in ops)
                  and len({o.reg for o in ops}) == 1)
    for k, o in enumerate(ops):
        if o.kind == 'Register':
            if zero_idiom:
                if k == 0:
                    add_reg(o.reg, W)
                continue
            add_reg(o.reg, o.access)
        elif o.kind == 'Memory':
            m = o.mem
            if mnemonic != 'nop':
                if m['base']:
                    add_reg(m['base'], R)
                if m['index']:
                    add_reg(m['index'], R)
            if o.access != 'A':
                disp = (end + m['disp']) & 0xffffffffffffffff if m['rip_rel'] else m['disp']
                mems.append({'base': m['base'] or 'none', 'index': m['index'] or 'none', 'scale': m['scale'], 'disp': disp, 'size': o.size // 8, 'access': o.access})
    for a in attr:
        if a.startswith('regs='):
            for item in a[5:].split(','):
                name, access = item.split(':')
                add_reg(name, {'R': R, 'W': W, 'RW': RW}[access])
        if a.startswith('mem='):
            msize = osz
            for b in attr:
                if b.startswith('msize='):
                    msize = int(b[6:])
            for item in a[4:].split(','):
                name, access = item.split(':')
                mems.append({'base': name, 'index': 'none', 'scale': 1, 'disp': 0, 'size': msize // 8, 'access': {'R': R, 'W': W, 'RW': RW}[access]})
        if a == 'stack=push':
            mems.append({'base': 'rsp', 'index': 'none', 'scale': 1, 'disp': -8, 'size': 8, 'access': W})
        if a == 'stack=pop':
            mems.append({'base': 'rsp', 'index': 'none', 'scale': 1, 'disp': 0, 'size': 8, 'access': R})
    if c.evex and c.mask:
        add_reg('k%d' % c.mask, R)
    string_op = mnemonic in ('movsb', 'movsw', 'movsl', 'movsq', 'stosb', 'stosw', 'stosl', 'stosq')
    if c.rep and string_op:
        add_reg('rcx', RW)
    flow = next((a[5:] for a in attr if a.startswith('flow=')), 'Next')
    target = None
    for o in ops:
        if o.kind == 'NearBranch':
            target = (end + o.imm) & 0xffffffffffffffff
            o.target = target
    indirect = mnemonic in ('call', 'jmp') and ops and ops[0].kind != 'NearBranch'
    # the structured operands
    jops = []
    imm_size = disp_size = 0
    for o in ops:
        if o.kind == 'Register':
            jops.append({'kind': 'Register', 'reg': o.reg, 'size': o.size, 'implicit': o.implicit})
        elif o.kind == 'Memory':
            m = o.mem
            jops.append({'kind': 'Memory', 'mem': {'base': m['base'] or 'none', 'index': m['index'] or 'none', 'scale': m['scale'], 'disp': m['disp'],
                                                    'size': o.size // 8, 'rip_rel': m['rip_rel'], 'seg': c.seg if not (indirect and c.seg == 'ds') else None},
                         'field_pos': o.field_pos})
            disp_size = 0 if o.field_pos is None else 32 if (m['rip_rel'] or (m['base'] is None and m['index'] is None)) else m['disp_size']
        elif o.kind == 'Immediate':
            signed = 'imm8s' in attr or (o.size == 32 and osz == 64) or o.size == 64
            jops.append({'kind': 'Immediate', 'imm': o.imm, 'size': o.size, 'signed': signed, 'implicit': o.implicit, 'field_pos': o.field_pos})
            if not o.implicit:
                imm_size = o.size
        elif o.kind == 'NearBranch':
            jops.append({'kind': 'NearBranch', 'branch': o.target, 'size': o.size, 'field_pos': o.field_pos})
            imm_size = o.size
    j = {'offset': off, 'len': length, 'bytes': c.code[c.start:c.pos].hex(), 'mnemonic': mnemonic, 'cc': cc,
         'encoding': 'EVEX' if c.evex else 'VEX' if c.vex else 'Legacy',
         'op_mask': 'k%d' % c.mask if c.evex and c.mask else 'none', 'zeroing': bool(c.evex and c.zero), 'bcast': bool(c.evex and c.bcast),
         'rep': ('repne' if c.rep == 0xf2 else 'rep') if c.rep and string_op else None,
         'lock': c.lock, 'notrack': bool(indirect and c.seg == 'ds'), 'flow': flow,
         'form': {'map': c.map, 'opcode': c.opcode, 'pfx': c.pfx, 'W': c.W, 'L': c.L, 'vex2': c.vex2,
                  'encoding': 'EVEX' if c.evex else 'VEX' if c.vex else 'Legacy', 'imm': imm_size, 'disp': disp_size, 'rex': c.rex is not None},
         'ops': jops, 'regs': regs, 'mems': mems}
    if target is not None:
        j['branch_target'] = target
    j['gas'] = format(j)
    return j


# ---------------------------------------------------------------- text

NOSUFFIX = {'cbtw', 'cwtl', 'cltq', 'cwtd', 'cltd', 'cqto', 'cpuid', 'xgetbv', 'sfence', 'ret', 'leave', 'int3', 'ud2', 'hlt', 'syscall',
            'endbr64', 'endbr32', 'movslq', 'prefetch', 'nop', 'pause', 'vzeroupper',
            'movsb', 'movsw', 'movsl', 'movsq', 'stosb', 'stosw', 'stosl', 'stosq'}


def is_vector_reg(name):
    return name.startswith(('xmm', 'ymm', 'zmm')) or (name.startswith('k') and name[1:].isdigit())


def reg_text(o):
    return '%' + o['reg']


def mem_text(m, end=None):
    """AT&T text of a memory operand. end: the instruction's end offset, to print a rip-relative target as a number."""
    seg = ('%%%s:' % m['seg']) if m.get('seg') else ''
    d = m['disp']
    if m['rip_rel']:
        return seg + ('0x%X(%%rip)' % ((end + d) & 0xffffffffffffffff) if end is not None else ('%d(%%rip)' % d))
    if m['base'] in (None, 'none') and m['index'] in (None, 'none'):
        return seg + ('0x%X' % (d & 0xffffffff) if d >= 0 else '-0x%X' % -d)
    s = seg + (('0x%X' % d) if d > 0 else ('-0x%X' % -d) if d < 0 else '')
    s += '('
    if m['base'] not in (None, 'none'):
        s += '%' + m['base']
    if m['index'] not in (None, 'none'):
        s += ',%%%s,%d' % (m['index'], m['scale'])
    return s + ')'


def imm_text(o):
    val, size = o['imm'], o['size']
    if o.get('signed') or val < 0:
        return ('$0x%X' % val) if val >= 0 else ('$-0x%X' % -val)
    return '$0x%X' % (val & ((1 << size) - 1))


def format(j):
    """AT&T text from the structured instruction."""
    mnemonic, ops = j['mnemonic'], j['ops']
    name = mnemonic
    pred = None
    if mnemonic.startswith('vpcmp') and len(ops) == 4 and ops[-1]['kind'] == 'Immediate' and 0 <= ops[-1]['imm'] < 8 and mnemonic[5:] in ('b', 'ub', 'w', 'uw', 'd', 'ud', 'q', 'uq'):
        pred = ops[-1]['imm']
        name = 'vpcmp' + PRED[pred] + mnemonic[5:]
    texts = []
    for k, o in enumerate(ops):
        if o.get('implicit'):
            continue
        if pred is not None and k == len(ops) - 1:
            continue
        if o['kind'] == 'Register':
            t = reg_text(o)
        elif o['kind'] == 'Memory':
            t = mem_text(o['mem'], j['offset'] + j['len'])
            if j.get('bcast'):
                t += '{1to%d}' % (o['mem']['size'] * 8 // (64 if j['form'] and j['form']['W'] else 32))
        elif o['kind'] == 'Immediate':
            t = imm_text(o)
        else:
            t = '0x%X' % o['branch']
        if k == 0 and j['op_mask'] != 'none':
            t += '{%%%s}' % j['op_mask']
            if j['zeroing']:
                t += '{z}'
        texts.append(t)
    if mnemonic in ('call', 'jmp') and ops and ops[0]['kind'] != 'NearBranch':
        texts = ['*' + texts[0]]
    suffix = ''
    vector = any(o['kind'] == 'Register' and is_vector_reg(o['reg']) for o in ops) or mnemonic.startswith('v')
    if not vector and mnemonic not in NOSUFFIX and not j.get('cc') and ops:
        sizes = [o['size'] if o['kind'] == 'Register' else o['mem']['size'] * 8 for o in ops if o['kind'] in ('Register', 'Memory') and not o.get('implicit')]
        if mnemonic in ('movzb', 'movzw', 'movsb', 'movsw') and len(ops) == 2 and ops[0]['kind'] == 'Register':
            suffix = SUFFIX[ops[0]['size']]
        elif mnemonic in ('call', 'jmp', 'push', 'pop'):
            suffix = 'q'
        elif sizes and sizes[0]:
            suffix = SUFFIX[sizes[0]]
    if mnemonic == 'nop' and ops:
        suffix = SUFFIX[ops[0]['mem']['size'] * 8 if ops[0]['kind'] == 'Memory' else ops[0]['size']]
    prefix = ('lock ' if j.get('lock') else '') + ('notrack ' if j.get('notrack') else '') + ((j['rep'] + ' ') if j.get('rep') else '')
    text = prefix + name + suffix
    if texts:
        text += ' ' + ','.join(reversed(texts))
    return text


def decode(code, base=0):
    out = []
    pos = 0
    while pos < len(code):
        j = decode_one(code, base, pos)
        out.append(j)
        pos += j['len']
    return out


if __name__ == '__main__':
    import sys
    data = open(sys.argv[1], 'rb').read()
    off, ln, base = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    for j in decode(data[off:off + ln], base):
        print('%6x: %-30s %s' % (j['offset'], j['bytes'], j['gas']))
