"""Machine code before registers are assigned: x86 instructions on virtual
registers (VReg) or physical ones (names: rax..r15, v0..v31, k0..k7), in
basic blocks that keep the IR's control flow.

An instruction's operands are in Intel order (destination first), each
with its access ('r', 'w', 'rw'); implicit registers, the flags among them,
are listed apart. Branch targets are blocks; calls name symbols."""

GPR = ['rax', 'rcx', 'rdx', 'rbx', 'rsp', 'rbp', 'rsi', 'rdi'] + ['r%d' % n for n in range(8, 16)]
GPR8 = ['al', 'cl', 'dl', 'bl', 'spl', 'bpl', 'sil', 'dil'] + ['r%db' % n for n in range(8, 16)]
GPR16 = ['ax', 'cx', 'dx', 'bx', 'sp', 'bp', 'si', 'di'] + ['r%dw' % n for n in range(8, 16)]
GPR32 = ['eax', 'ecx', 'edx', 'ebx', 'esp', 'ebp', 'esi', 'edi'] + ['r%dd' % n for n in range(8, 16)]
VEC = ['v%d' % n for n in range(32)]
MASK = ['k%d' % n for n in range(8)]
CALLER_SAVED = ['rax', 'rcx', 'rdx', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11']
CALLEE_SAVED = ['rbx', 'rbp', 'r12', 'r13', 'r14', 'r15']
ARG_GPR = ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9']
FLAGS = 'flags'
# instructions that write the flags (besides those listing FLAGS as an implicit definition)
FLAG_WRITERS = {'add', 'sub', 'and', 'or', 'xor', 'neg', 'inc', 'dec', 'cmp', 'test', 'shl', 'shr', 'sar', 'bsf', 'tzcnt',
                'popcnt', 'imul', 'bt', 'bts', 'btr', 'btc', 'ptest', 'vptest', 'kortestb', 'kortestw', 'kortestd', 'kortestq',
                'pcmpistri', 'vpcmpistri', 'pcmpistrm', 'vpcmpistrm', 'bzhi', 'cpuid', 'xgetbv'}
# those whose zero flag tells whether their result is zero
ZERO_FLAG = {'add', 'sub', 'and', 'or', 'xor', 'neg', 'inc', 'dec'}
CLOBBERED_BY_CALL = frozenset(CALLER_SAVED + VEC + MASK + [FLAGS])


def bank(preg):
    if preg in GPR:
        return 'gpr'
    if preg[0] == 'v':
        return 'vec'
    if preg[0] == 'k':
        return 'mask'
    return 'flags'


class VReg:
    __slots__ = ('id', 'cls', 'lo16', 'hint')

    def __init__(self, id_, cls):
        self.id, self.cls, self.lo16, self.hint = id_, cls, False, None

    def __repr__(self):
        return '%%%s%d' % ({'gpr': 'r', 'vec': 'x', 'mask': 'k'}[self.cls], self.id)


class Reg:
    __slots__ = ('r', 'bits')

    def __init__(self, r, bits):
        self.r, self.bits = r, bits

    def __repr__(self):
        return '%r:%d' % (self.r, self.bits) if not isinstance(self.r, str) else name(self.r, self.bits)


class Imm:
    __slots__ = ('value', 'bits')

    def __init__(self, value, bits=32):
        self.value, self.bits = value, bits

    def __repr__(self):
        return '$%d' % self.value


class Mem:
    __slots__ = ('base', 'index', 'scale', 'disp', 'size', 'sym', 'slot')

    def __init__(self, size, base=None, index=None, scale=1, disp=0, sym=None, slot=None):
        self.size, self.base, self.index, self.scale, self.disp, self.sym, self.slot = size, base, index, scale, disp, sym, slot

    def regs(self):
        return [r for r in (self.base, self.index) if r is not None]

    def copy(self):
        return Mem(self.size, self.base, self.index, self.scale, self.disp, self.sym, self.slot)

    def __repr__(self):
        where = ''.join([self.sym or '', '+slot%d' % self.slot.id if self.slot else ''])
        inner = ','.join(repr(r) for r in self.regs())
        return '%s%d(%s%s)%s' % (where, self.disp, inner, ',%d' % self.scale if self.index is not None else '', '[%d]' % self.size)


class Label:
    __slots__ = ('block',)

    def __init__(self, block):
        self.block = block

    def __repr__(self):
        return '.%s' % self.block.name


class Sym:
    """A call or jump target: a function by name; slot names the table slot of a str_* entry."""
    __slots__ = ('name', 'slot')

    def __init__(self, name, slot=None):
        self.name, self.slot = name, slot

    def __repr__(self):
        return self.name


class Slot:
    """A stack object: a local of the IR (alloca) or a spill."""
    __slots__ = ('id', 'size', 'align', 'offset')

    def __init__(self, id_, size, align):
        self.id, self.size, self.align, self.offset = id_, size, align, None


class Ins:
    __slots__ = ('op', 'ops', 'acc', 'cc', 'imp_def', 'imp_use', 'mask', 'zero', 'kind', 'rep')

    def __init__(self, op, ops=(), acc='', cc=None, imp_def=(), imp_use=(), mask=None, zero=False, kind=None, rep=None):
        self.op, self.ops, self.cc = op, list(ops), cc
        self.acc = acc.split() if isinstance(acc, str) else list(acc)
        self.imp_def, self.imp_use = set(imp_def), set(imp_use)
        self.mask, self.zero, self.kind, self.rep = mask, zero, kind, rep

    def uses(self):
        out = []
        for o, a in zip(self.ops, self.acc):
            if isinstance(o, Reg):
                if 'r' in a:
                    out.append(o.r)
            elif isinstance(o, Mem):
                out.extend(o.regs())
        if self.mask is not None:
            out.append(self.mask)
        out.extend(self.imp_use)
        return out

    def defs(self):
        out = [o.r for o, a in zip(self.ops, self.acc) if isinstance(o, Reg) and 'w' in a]
        out.extend(self.imp_def)
        return out

    def regs(self):
        """Every register operand (explicit, memory and mask), for renaming."""
        out = []
        for o in self.ops:
            if isinstance(o, Reg):
                out.append(o.r)
            elif isinstance(o, Mem):
                out.extend(o.regs())
        if self.mask is not None:
            out.append(self.mask)
        return out

    def map(self, f):
        for o in self.ops:
            if isinstance(o, Reg):
                o.r = f(o.r)
            elif isinstance(o, Mem):
                if o.base is not None:
                    o.base = f(o.base)
                if o.index is not None:
                    o.index = f(o.index)
        if self.mask is not None:
            self.mask = f(self.mask)

    @property
    def is_copy(self):
        return self.kind == 'copy'

    @property
    def is_branch(self):
        return self.kind in ('jmp', 'jcc')

    @property
    def is_exit(self):
        return self.kind in ('ret', 'tail')

    def target(self):
        return self.ops[0].block if self.kind in ('jmp', 'jcc') else None

    def __repr__(self):
        text = self.op + (self.cc or '')
        ops = ', '.join(repr(o) for o in self.ops)
        extra = ''
        if self.mask is not None:
            extra += ' {%r}%s' % (self.mask, '{z}' if self.zero else '')
        return '%s %s%s' % (text, ops, extra)


class MBlock:
    def __init__(self, name, source=None):
        self.name, self.source = name, source
        self.ins, self.succs = [], []
        self.freq, self.loop_depth, self.loop_header, self.align = 1.0, 0, False, 0
        self.loops = frozenset()      # the headers of the loops the block is in

    def __repr__(self):
        return self.name


class MFunction:
    def __init__(self, name, isa):
        self.name, self.isa = name, isa
        self.blocks, self.next_vreg, self.slots = [], 0, []
        self.attrs = {}
        self.frame = None

    def vreg(self, cls):
        v = VReg(self.next_vreg, cls)
        self.next_vreg += 1
        return v

    def block(self, name, source=None):
        b = MBlock(name, source)
        self.blocks.append(b)
        return b

    def slot(self, size, align):
        s = Slot(len(self.slots), size, align)
        self.slots.append(s)
        return s

    def dump(self):
        lines = ['mfunction %s' % self.name]
        for b in self.blocks:
            lines.append('%s: -> %s%s' % (b.name, ', '.join(s.name for s in b.succs), '  (loop header)' if b.loop_header else ''))
            lines.extend('    %r' % i for i in b.ins)
        return '\n'.join(lines)


def name(preg, bits):
    """The assembler's name of a physical register at a width."""
    if preg in GPR:
        n = GPR.index(preg)
        return {8: GPR8, 16: GPR16, 32: GPR32, 64: GPR}[bits][n]
    if preg[0] == 'v':
        return {128: 'xmm', 256: 'ymm', 512: 'zmm'}[bits] + preg[1:]
    return preg
