"""Instruction selection: the IR of a function to x86 instructions on
virtual registers (mir.py).

Each block's ops are selected in order; an op whose only uses can absorb it
is folded into them instead: a comparison into the branch, select or setcc
that reads it (the flags live only between the two), address arithmetic
into the memory operands that use it, a load with one use into that
instruction's memory operand when nothing writes memory in between, a
zero-extension of a 32-bit value into the write that produced it. Phis
become copies at the ends of the predecessors (critical edges with phis are
split first); the register allocator removes most of them.

Mask-register values (AVX-512 compares) live in k registers; a copy to a
general register is made where one is needed (kmov) and dropped when it is
not. The ISA of the unit decides the forms: legacy SSE, VEX or EVEX, tzcnt
or bsf, shlx or shifts by cl."""
from . import ir
from .ir import I1, I8, I16, I32, I64, PTR, INVERT, SWAP
from .mir import (MFunction, Ins, Reg, Imm, Mem, Label, Sym, VReg, ARG_GPR, FLAGS, CALLER_SAVED, VEC, MASK, FLAG_WRITERS, ZERO_FLAG)

CC_OF = {'eq': 'e', 'ne': 'ne', 'ult': 'b', 'ule': 'be', 'ugt': 'a', 'uge': 'ae', 'slt': 'l', 'sle': 'le', 'sgt': 'g', 'sge': 'ge'}
ARITH = {'add': 'add', 'sub': 'sub', 'and': 'and', 'or': 'or', 'xor': 'xor'}
# two-input vector ops: (legacy, vex) mnemonics
VOP2 = {'cmpeq_epi8': ('pcmpeqb', 'vpcmpeqb'), 'cmpgt_epi8': ('pcmpgtb', 'vpcmpgtb'), 'min_epu8': ('pminub', 'vpminub'),
        'add_epi8': ('paddb', 'vpaddb'), 'sub_epi8': ('psubb', 'vpsubb'), 'shuffle_epi8': ('pshufb', 'vpshufb'),
        'and': ('pand', 'vpand'), 'or': ('por', 'vpor'), 'xor': ('pxor', 'vpxor'), 'andnot': ('pandn', 'vpandn')}
MASK_CMP = {'cmpeq_epi8_mask': ('vpcmpeqb', None), 'cmpneq_epu8_mask': ('vpcmpub', 4), 'test_epi8_mask': ('vptestmb', None),
            'testn_epi8_mask': ('vptestnmb', None)}
KSUFFIX = {8: 'b', 16: 'w', 32: 'd', 64: 'q'}


class Unsupported(Exception):
    pass


def split_critical_edges(fn):
    preds = fn.preds()
    for b in list(fn.blocks):
        if not b.phis() or len(preds[b]) < 2:
            continue
        for p in list(preds[b]):
            if len(p.succs) < 2:
                continue
            mid = fn.block('edge')
            mid.attrs['edge'] = True
            jmp = fn.new('jmp')
            jmp.targets = [b]
            mid.set_term(jmp)
            p.term.targets = [mid if t is b else t for t in p.term.targets]
            for phi in b.phis():
                i = phi.preds.index(p)
                phi.preds[i] = mid


def is_pow2(c):
    return c > 0 and c & (c - 1) == 0


def cold_blocks(fn):
    """The blocks every way into which is the unlikely side of a weighted branch (or comes from such a block)."""
    preds = fn.preds()
    cold = set()
    changed = True
    while changed:
        changed = False
        for b in fn.blocks:
            if b is fn.entry or b in cold or not preds[b]:
                continue
            ways = []
            for p in preds[b]:
                t = p.term
                unlikely = False
                if t.kind == 'br' and 'weights' in t.attrs and t.targets[0] is not t.targets[1]:
                    w, i = t.attrs['weights'], t.targets.index(b)
                    unlikely = w[i] * 10 < w[1 - i]
                ways.append(p in cold or unlikely)
            if all(ways):
                cold.add(b)
                changed = True
    return cold


def permute_ternary(imm, perm):
    """The imm8 of vpternlog with its operands reordered: new operand i is old operand perm[i]."""
    out = 0
    for i in range(8):
        bits = [(i >> 2) & 1, (i >> 1) & 1, i & 1]      # the bits of the new operands 0, 1, 2
        old = [0, 0, 0]
        for n in range(3):
            old[perm[n]] = bits[n]
        if imm >> (old[0] << 2 | old[1] << 1 | old[2]) & 1:
            out |= 1 << i
    return out


class ISel:
    def __init__(self, fn, isa):
        self.fn, self.isa = fn, isa
        self.vex = isa in ('avx2', 'evex', 'evex512')
        self.evex = isa in ('evex', 'evex512')
        self.bmi = isa in ('avx2', 'evex', 'evex512')
        self.mf = MFunction(fn.name, isa)
        self.mf.attrs = dict(fn.attrs)
        self.mf.ret_ty = fn.ret_ty
        self.vregs = {}          # Value -> VReg
        self.clean32 = set()     # VRegs whose 32-bit write zero-extended them
        self.kview = {}          # Value -> VReg in a mask register (AVX-512 compares)
        self.kcache = {}         # (Value, block) -> the mask register it was moved into there
        self.const_here = {}     # (constant, block) -> the register it was put in there
        self.zext_ok = set()     # VRegs whose narrow (8/16-bit) value is zero-extended in them (movzx, kmov)
        self.pending_late = set()   # ops of the block held back to its end (they feed phis alone)
        self.cmp_ins, self.cmp_cc = {}, {}   # a compare kept as a value: the instruction that set the flags, its condition
        self.block_of = {}
        self.skip = set()        # ops folded into their users
        self.addr_only = set()   # ops that only feed memory operands
        self.constants = {}      # bytes -> label
        self.block = None
        self.slots = {}          # alloca op -> Slot
        self.preds = fn.preds()
        self.uses_at = {}

    # ------------------------------------------------------------ helpers

    def emit(self, op, *ops, acc='', **kw):
        ins = Ins(op, ops, acc, **kw)
        self.block.ins.append(ins)
        return ins

    def new(self, cls):
        return self.mf.vreg(cls)

    def gpr(self, bits=64):
        return Reg(self.new('gpr'), bits)

    def vreg(self, v):
        """The register holding v, materializing constants and mask views."""
        r = self.vregs.get(v)
        if r is not None:
            return r
        op = v.op
        if op.kind == 'const':
            return self.materialize(v)      # at the use; vector constants were placed by place_constants
        elif op.kind == 'undef':
            r = self.new('vec' if v.ty.is_vec else 'gpr')
        elif op.kind == 'sym':
            r = self.new('gpr')
            self.emit('lea', Reg(r, 64), Mem(8, sym=op.attrs['name']), acc='w r')
        elif v in self.kview:
            r = self.new('gpr')
            k = self.kview[v]
            self.emit('kmov' + KSUFFIX[v.ty.bits], Reg(r, 64 if v.ty.bits > 32 else 32), Reg(k, v.ty.bits), acc='w r', kind='kmov')
            if v.ty.bits <= 32:
                self.clean32.add(r)
                self.zext_ok.add(r)
        elif op in self.skip and op.kind == 'load':
            # a load folded somewhere but also needed as a value: reload
            r = self.select_load(op)
        elif op in self.addr_only:
            r = self.new('gpr')
            self.emit('lea', Reg(r, 64), self.mem_operand(v, 8), acc='w r')
        elif op in self.pending_late:
            # wanted before the end of the block after all: selected here
            self.pending_late.discard(op)
            self.select(op)
            return self.vregs[v]
        else:
            raise Unsupported('%s: %s has no register (op %s)' % (self.fn.name, v, op.kind))
        self.vregs[v] = r
        return r

    def reg(self, v, bits=None):
        bits = bits or v.ty.bits
        if v.ty.is_vec:
            return Reg(self.vreg(v), bits)
        if v.ty is I1:
            return Reg(self.vreg(v), 8)
        return Reg(self.vreg(v), max(bits, 8))

    def define(self, v, cls=None):
        r = self.new(cls or ('vec' if v.ty.is_vec else 'gpr'))
        self.vregs[v] = r
        if v.ty.kind == 'i' and v.ty.bits == 32:
            self.clean32.add(r)
        return Reg(r, 8 if v.ty is I1 else v.ty.bits)

    def imm_ok(self, v, bits=32):
        """An operand usable as a sign-extended imm32."""
        c = v.const
        return c is not None and not v.ty.is_vec and (v.ty.bits <= 32 or -(1 << 31) <= ir.signed(c, 64) < (1 << 31))

    def imm(self, v):
        c = v.const
        if v.ty.bits <= 32:
            c = ir.signed(c, v.ty.bits) if v.ty.bits < 32 else ir.signed(c, 32)
        else:
            c = ir.signed(c, 64)
        return Imm(c, min(v.ty.bits, 32) if v.ty.bits >= 8 else 8)

    def materialize(self, v):
        c = v.const
        if v.ty.is_vec:
            r = self.const_reg(v)
            if r is None:
                r = self.const_here.get((v, self.block))
            if r is not None:
                return r
            r = self.const_here[(v, self.block)] = self.new('vec')
            size = v.ty.size
            if c == bytes(size):
                self.emit('vpxor' if self.vex else 'pxor', Reg(r, 128), Reg(r, 128), Reg(r, 128), acc='w r r', kind='zero') if self.vex \
                    else self.emit('pxor', Reg(r, 128), Reg(r, 128), acc='w r', kind='zero')
            elif c == b'\xff' * size:
                if self.vex:
                    self.emit('vpcmpeqd', Reg(r, v.ty.bits), Reg(r, v.ty.bits), Reg(r, v.ty.bits), acc='w r r', kind='ones')
                else:
                    self.emit('pcmpeqd', Reg(r, 128), Reg(r, 128), acc='w r', kind='ones')
            else:
                self.emit(self.vmov_load(v.ty.bits, True), Reg(r, v.ty.bits), self.constant_operand(c), acc='w r')
            return r
        r = self.const_here.get((v, self.block))
        if r is not None:
            return r
        pre = self.loop_entry(self.fn_block) if self.fn_block not in self.cold else None
        if pre is not None:
            # inside a loop: the register is set once, before the loop
            r = self.const_here.get((v, pre))
            if r is None:
                r = self.const_here[(v, pre)] = self.new('gpr')
                ins = Ins('mov', [Reg(r, 32), Imm(c & 0xffffffff, 32)], 'w r') if c < (1 << 32) else \
                    Ins('mov', [Reg(r, 64), Imm(ir.signed(c, 64), 32)], 'w r') if -(1 << 31) <= ir.signed(c, 64) < 0 else \
                    Ins('movabs', [Reg(r, 64), Imm(ir.signed(c, 64), 64)], 'w r')
                if c < (1 << 32):
                    self.clean32.add(r)
                self.insert_before_end(pre, ins)
            return r
        r = self.new('gpr')
        if c == 0:
            self.emit('xor', Reg(r, 32), Reg(r, 32), acc='w r', kind='zero')
            self.clean32.add(r)
        else:
            self.materialize_into(r, v)
        self.const_here[(v, self.block)] = r
        return r

    def loop_entry(self, b):
        """The block before the outermost loop around b (its header's dominator), or None when b is in no loop."""
        loop = self.loops.of.get(b)
        pre = None
        while loop is not None:
            parent = self.dom.parent(loop.header)
            if parent is None:
                break
            pre, loop = parent, self.loops.of.get(parent)
        return pre

    def insert_before_end(self, b, ins):
        """ins into the (already selected) block b, before its phi copies, compare and jump."""
        mb = self.block_of[b]
        at = len(mb.ins)
        while at > 0 and (mb.ins[at - 1].kind in ('jmp', 'jcc', 'cmp', 'copy', 'copy32', 'kmov', 'tail') or mb.ins[at - 1].op in ('set',)):
            at -= 1
        mb.ins.insert(at, ins)

    def constant_operand(self, data):
        label = self.constants.get(data)
        if label is None:
            label = self.constants[data] = '.LCPI%d_%d' % (self.mf.attrs.setdefault('cp_index', 0), len(self.constants))
        return Mem(len(data), sym=label)

    def vmov_load(self, bits, aligned):
        if not self.vex:
            return 'movdqa' if aligned else 'movdqu'
        if bits == 512:
            return 'vmovdqa64' if aligned else 'vmovdqu64'
        return 'vmovdqa' if aligned else 'vmovdqu'

    # ---------------------------------------------------------- analysis

    def analyze(self):
        fn = self.fn
        for b in fn.blocks:
            for i, op in enumerate(b.ops + [b.term]):
                self.uses_at[op] = (b, i)
        # address-only ops: every use is a memory operand (or another address-only op)
        changed = True
        candidates = set()
        for b in fn.blocks:
            for op in b.ops:
                if op.kind in ('ptradd', 'i2p', 'p2i') or op.kind == 'add' and op.ty is I64 \
                        or op.kind == 'shl' and op.ty is I64 and op.args[1].const in (1, 2, 3) \
                        or op.kind == 'mul' and op.ty is I64 and op.args[1].const in (2, 4, 8):
                    candidates.add(op)
        addr_only = set(candidates)
        while changed:
            changed = False
            for op in list(addr_only):
                for u in op.result.uses:
                    ok = (u.kind == 'load' and u.args[0] is op.result or u.kind == 'store' and u.args[0] is op.result
                          or u.kind == 'vop' and u.attrs.get('memory') and u.args[-1 if 'store' in u.attrs['name'] else 0] is op.result
                          and self.vop_address_index(u) is not None and u.args[self.vop_address_index(u)] is op.result
                          or u in addr_only and u.kind != 'p2i')
                    if not ok:
                        addr_only.discard(op)
                        changed = True
                        break
        # a p2i feeding adds that feed i2p: fine as address-only only in that chain
        self.addr_only = addr_only
        # loads folded into their single user
        for b in fn.blocks:
            for i, op in enumerate(b.ops):
                if op.kind != 'load' or op.attrs.get('volatile') or len(op.result.uses) != 1:
                    continue
                user = op.result.uses[0]
                ub, ui = self.uses_at.get(user, (None, None))
                if ub is not b or any(x.writes_memory or x.kind in ('call', 'asm') for x in b.ops[i + 1:ui]):
                    continue
                if self.can_fold_load(op, user):
                    self.skip.add(op)
        # store(p, op(load(p), x)): the operation on the memory itself
        self.rmw = {}
        for b in fn.blocks:
            for i, op in enumerate(b.ops):
                if op.kind != 'store':
                    continue
                v = op.args[1]
                o = v.op
                if o.kind not in ('or', 'and', 'xor', 'add', 'sub') or o.block is not b or len(v.uses) != 1 or v.ty.kind != 'i' or v.ty.bits < 8:
                    continue
                which = next((n for n, a in enumerate(o.args) if a.op.kind == 'load' and a.op in self.skip and a.op.args[0] is op.args[0]
                              and len(a.uses) == 1 and not a.op.attrs.get('volatile')), None)
                if which is None or (o.kind == 'sub' and which != 0):
                    continue
                oi = self.uses_at[o][1]
                if any(x.writes_memory or x.kind in ('call', 'asm') for x in b.ops[oi + 1:i]):
                    continue
                self.rmw[op] = (o, o.args[1 - which])
                self.skip.add(o)
        # a masked load feeding one masked compare under the same mask: the compare's memory operand
        for b in fn.blocks:
            for i, op in enumerate(b.ops):
                if op.kind != 'vop' or op.attrs['name'] != 'masked_load' or len(op.result.uses) != 1:
                    continue
                user = op.result.uses[0]
                ub, ui = self.uses_at.get(user, (None, None))
                if ub is not b or user.kind != 'vop' or not user.attrs['name'].endswith('_masked') or user.args[0] is not op.args[0]:
                    continue
                if any(x.writes_memory or x.kind in ('call', 'asm') for x in b.ops[i + 1:ui]):
                    continue
                if user.args[1] is op.result and user.args[2] is not op.result and not (user.args[2].op.kind == 'vop' and user.args[2].op in self.skip):
                    user.args[1], user.args[2] = user.args[2], user.args[1]     # (symmetric compares)
                if user.args[2] is op.result and user.args[1] is not op.result:
                    self.skip.add(op)
        # comparisons folded into their users
        for b in fn.blocks:
            for op in b.ops:
                if op.kind == 'icmp' and op.result.uses and all(
                        u.kind in ('br', 'select') and self.uses_at[u][0] is b or u.kind == 'zext' and len(op.result.uses) == 1 and self.uses_at[u][0] is b
                        for u in op.result.uses):
                    self.skip.add(op)
            # c1 && c2 under a branch: a branch per condition
            if b.term.kind == 'br':
                self.logic_tree(b.term.args[0], b)
        # 1 << n feeding one or/xor: bts/btc
        for b in fn.blocks:
            for op in b.ops:
                if op.kind == 'shl' and op.args[0].const == 1 and op.ty.bits >= 32 and len(op.result.uses) == 1:
                    u = op.result.uses[0]
                    if u.kind in ('or', 'xor') and self.uses_at[u][0] is b and u.args[1] is op.result and u.args[0] is not op.result:
                        self.skip.add(op)
                if op.kind == 'vop' and op.attrs['name'] in ('cmpistra', 'cmpistrc', 'cmpistrz', 'testz') and op.result.uses and all(
                        u.kind == 'icmp' and self.uses_at[u][0] is b and u.args[1].const is not None and u in self.skip for u in op.result.uses):
                    self.skip.add(op)     # the flag is read where the comparison is

    def logic_tree(self, v, b):
        """v is an and/or of conditions (each used by it alone, in b) the branch
        can take one at a time: mark the tree as folded. True when so."""
        op = v.op
        if op.kind in ('and', 'or') and v.ty is I1 and len(v.uses) == 1 and self.uses_at.get(op, (None,))[0] is b:
            inner = [self.logic_tree(a, b) for a in op.args]
            if all(inner):
                self.skip.add(op)
                return True
            return False
        if op.kind == 'xor' and v.ty is I1 and op.args[1].const == 1 and len(v.uses) == 1 and self.uses_at.get(op, (None,))[0] is b:
            if self.logic_tree(op.args[0], b):
                self.skip.add(op)
                return True
            return False
        if op.kind == 'icmp' and len(v.uses) == 1 and self.uses_at.get(op, (None,))[0] is b:
            self.skip.add(op)
            return True
        return op.kind != 'icmp'      # any other i1 value: a register to test

    def can_fold_load(self, load, user):
        """Whether the load's value may be the user's memory operand."""
        v = load.result
        ty = v.ty
        if ty.is_vec:
            if not self.vex and load.attrs.get('align', 1) < ty.size:
                return False      # legacy SSE: a memory operand must be aligned
            if user.kind in ('and', 'or', 'xor'):
                if user.args[0] is v and user.args[1] is not v:
                    user.args.reverse()
                return user.args[1] is v
            if user.kind != 'vop':
                return False
            name, args = user.attrs['name'], user.args
            if name in VOP2:
                if name in ('cmpeq_epi8', 'min_epu8', 'add_epi8', 'and', 'or', 'xor') and args[0] is v and args[1] is not v:
                    args.reverse()
                return args[1] is v and args[0] is not v
            if name in MASK_CMP:
                if args[0] is v and args[1] is not v:
                    args.reverse()
                return args[1] is v and args[0] is not v
            if name.endswith('_masked'):
                if args[1] is v and args[2] is not v:
                    args[1], args[2] = args[2], args[1]
                return args[2] is v and args[1] is not v
            if name == 'ternary32':
                return args[2] is v and args[0] is not v and args[1] is not v
            if name in ('insert128', 'testz'):
                return args[1] is v and args[0] is not v
            return False
        if user.kind == 'vop' and user.attrs['name'] == 'splat8' and self.vex:
            return user.args[0] is v
        if ty.bits < 32 and user.kind not in ('zext', 'sext'):
            return False
        if user.kind in ('add', 'and', 'or', 'xor', 'mul'):
            if user.args[0] is v and user.args[1] is not v:
                user.args.reverse()
            return user.args[1] is v and user.args[0] is not v
        if user.kind in ('sub', 'icmp'):
            return user.args[0] is not v or user.args[1] is not v
        if user.kind in ('cttz', 'zext', 'sext'):
            return True
        if user.kind == 'call':
            return user.args[0] is v and len(v.uses) == 1   # jmp/call *mem
        return False

    def vop_address_index(self, op):
        name = op.attrs['name']
        if name in ('masked_load',):
            return 1
        if name in ('masked_store', 'store_low'):
            return 0
        return None

    # ----------------------------------------------------------- addresses

    def mem_operand(self, v, size, index_ok=True):
        """A memory operand for the address v."""
        base = None
        index = None
        scale = 1
        disp = 0
        sym = None
        slot = None

        def add_part(x):
            nonlocal base, index, scale, disp, sym, slot
            op = x.op
            if op.kind == 'const':
                disp += ir.signed(op.attrs['value'], 64)
                return
            if op.kind == 'sym' and sym is None and base is None:
                sym = op.attrs['name']
                return
            if op.kind == 'alloca' and x not in self.vregs and slot is None and base is None:
                slot = self.slot_of(op)
                return
            if op.kind in ('ptradd', 'add') and op.args[1].const is not None and x.ty.bits == 64 and op not in self.skip and x not in self.vregs \
                    and (op in self.addr_only or op.args[0] in self.vregs or op.args[0].op.kind == 'param'):
                # p + c: the displacement takes c even when p + c itself is wanted elsewhere
                add_part(op.args[0])
                add_part(op.args[1])
                return
            if op in self.addr_only or op.kind in ('i2p', 'p2i') and x not in self.vregs:
                if op.kind in ('i2p', 'p2i'):
                    add_part(op.args[0])
                    return
                if op.kind in ('ptradd', 'add'):
                    add_part(op.args[0])
                    add_part(op.args[1])
                    return
                if op.kind == 'shl' and index is None and index_ok:
                    index, scale = self.vreg(op.args[0]), 1 << op.args[1].const
                    return
                if op.kind == 'mul' and index is None and index_ok:
                    index, scale = self.vreg(op.args[0]), op.args[1].const
                    return
            r = self.vreg(x)
            if base is None:
                base = r
            elif index is None and index_ok:
                index = r
            else:
                t = self.new('gpr')
                self.emit('lea', Reg(t, 64), Mem(8, base=base, index=index, scale=scale), acc='w r')
                base, index, scale = t, r, 1

        add_part(v)
        if sym is not None and (base is not None or index is not None):
            t = self.new('gpr')
            self.emit('lea', Reg(t, 64), Mem(8, sym=sym), acc='w r')
            sym = None
            if base is None:
                base = t
            elif index is None:
                index = t
            else:
                t2 = self.new('gpr')
                self.emit('lea', Reg(t2, 64), Mem(8, base=base, index=index, scale=scale), acc='w r')
                base, index, scale = t2, t, 1
        if slot is not None and (index is not None and base is not None):
            t = self.new('gpr')
            self.emit('lea', Reg(t, 64), Mem(8, base=base, index=index, scale=scale), acc='w r')
            base, index, scale = t, None, 1
        if slot is not None and base is not None and index is None:
            index, base = base, None
        if not -(1 << 31) <= disp < (1 << 31):
            t = self.new('gpr')
            self.emit('movabs', Reg(t, 64), Imm(disp, 64), acc='w r')
            if base is None:
                base = t
            elif index is None:
                index = t
            else:
                raise Unsupported('address with a 64-bit displacement and two registers')
            disp = 0
        if base is None and index is None and sym is None and slot is None:
            t = self.new('gpr')
            self.emit('mov', Reg(t, 64), Imm(disp, 32), acc='w r')
            base, disp = t, 0
        return Mem(size, base=base, index=index, scale=scale, disp=disp, sym=sym, slot=slot)

    def slot_of(self, op):
        s = self.slots.get(op)
        if s is None:
            s = self.slots[op] = self.mf.slot(op.attrs['size'], op.attrs.get('align', 1))
        return s

    def operand(self, v, bits=None, allow_imm=True, allow_mem=True):
        """A read operand: an immediate, a folded load, or a register."""
        if allow_imm and self.imm_ok(v):
            return self.imm(v) if bits is None else Imm(ir.signed(v.const, min(bits, 32)) if bits <= 32 else ir.signed(v.const, 64), 32)
        if allow_mem and v.op in self.skip and v.op.kind == 'load' and v not in self.vregs:
            return self.mem_operand(v.op.args[0], v.ty.size)
        return self.reg(v, bits)

    # ------------------------------------------------------------- blocks

    def run(self):
        fn = self.fn
        split_critical_edges(fn)
        self.preds = fn.preds()
        self.analyze()
        order = fn.rpo()
        for b in order:
            self.block_of[b] = self.mf.block(b.name.replace('.', '_') + '_%d' % b.id, b)
        self.mf.entry = self.block_of[fn.entry]
        loops = ir.find_loops(fn)
        for loop in loops.loops:
            self.block_of[loop.header].loop_header = True
        for b in order:
            self.block_of[b].loop_depth = loops.depth(b)
            loop = loops.of.get(b)
            headers = set()
            while loop is not None:
                headers.add(self.block_of[loop.header])
                loop = loop.parent
            self.block_of[b].loops = frozenset(headers)
        for b in order:
            for phi in b.phis():
                self.define(phi.result)
        self.select_params()
        self.place_constants()
        for b in order:
            self.block = self.block_of[b]
            self.fn_block = b
            self.block.succs = [self.block_of[s] for s in b.succs]
            for v in self.const_at.get(b, ()):
                self.const_targets.setdefault(v, {})[b] = self.materialize(v)
            ops = [op for op in b.ops if op.kind != 'phi' and op not in self.skip and op not in self.addr_only]
            late = [op for op in ops if op.kind in ir.PURE and op.kind not in ir.DIVISIONS and op.result is not None
                    and op.result.uses and all(u.kind == 'phi' or u in self.addr_only for u in op.result.uses)
                    and any(u.kind == 'phi' for u in op.result.uses)]
            self.pending_late = set(late)
            for op in ops:
                if op not in late:
                    self.select(op)
            for op in late:
                if op in self.pending_late:
                    self.pending_late.discard(op)
                    self.select(op)      # (a loop's next pointer after the last use of this one: one register, no copy)
            self.phi_copies(b)
            self.terminator(b.term)
        self.mf.constants = self.constants
        self.mf.vreg_clean32 = self.clean32
        return self.mf

    def foldable_constant_use(self, v, u):
        """The constant v can be u's memory operand (so needs no register there)."""
        if u.kind in ('and', 'or', 'xor') and u.ty.is_vec:
            return u.args[1] is v and u.args[0] is not v
        if u.kind != 'vop':
            return False
        name, args = u.attrs['name'], u.args
        if name in VOP2 or name in MASK_CMP or name in ('insert128', 'testz'):
            return args[1] is v and args[0] is not v
        if name.endswith('_masked') or name == 'ternary32':
            return args[2] is v and v not in args[:2]
        return False

    def place_constants(self):
        """Where each vector constant is materialized: a use outside any loop
        that can take it as a memory operand does; the others get a register
        in their block, or before the outermost loop around it. Several
        regions of uses get several registers, each where it dominates its
        uses (const_reg finds the one in force)."""
        fn = self.fn
        dom = self.dom = ir.Dominators(fn)
        loops = self.loops = ir.find_loops(fn, dom)
        self.cold = cold_blocks(fn)
        self.const_at = {}
        self.const_targets = {}
        for v in list(fn.consts.values()):
            if v.op.kind != 'const' or not v.ty.is_vec or not v.uses:
                continue
            idiom = v.const == bytes(v.ty.size) or v.const == b'\xff' * v.ty.size
            for u in v.uses:
                blocks = [p for p, a in zip(u.preds, u.args) if a is v] if u.kind == 'phi' else [u.block] if u.block is not None else []
                for ub in blocks:
                    if ub not in dom.pre:
                        continue
                    if not idiom and (loops.depth(ub) == 0 or ub in self.cold) and self.foldable_constant_use(v, u):
                        continue
                    t = ub
                    while ub not in self.cold:
                        loop = loops.of.get(t)
                        if loop is None or dom.parent(loop.header) is None:
                            break
                        t = dom.parent(loop.header)
                    at = self.const_at.setdefault(t, [])
                    if v not in at:
                        at.append(v)

    def const_reg(self, v):
        """The register holding the vector constant v where the block being selected is, or None."""
        targets = self.const_targets.get(v)
        if not targets:
            return None
        b = self.fn_block
        while b is not None:
            r = targets.get(b)
            if r is not None:
                return r
            b = self.dom.parent(b)
        return None

    def select_params(self):
        self.block = self.mf.entry
        self.param_reg = {}
        gi = vi = 0
        for p in self.fn.params:
            if p.ty.is_vec:
                r = self.define(p)
                self.emit('vmovdqa' if self.vex else 'movdqa', Reg(r.r, p.ty.bits), Reg('v%d' % vi, p.ty.bits), acc='w r', kind='copy')
                vi += 1
            else:
                r = self.define(p)
                self.param_reg[p] = ARG_GPR[gi]
                if p.ty.bits <= 32:
                    self.emit('mov', Reg(r.r, 32), Reg(ARG_GPR[gi], 32), acc='w r', kind='copy')
                else:
                    self.emit('mov', Reg(r.r, 64), Reg(ARG_GPR[gi], 64), acc='w r', kind='copy')
                gi += 1

    def phi_copies(self, b):
        """Copies for the phis of b's successors, as a parallel assignment."""
        moves = []
        for s in b.succs:
            for phi in s.phis():
                v = phi.incoming(b)
                dst = self.vregs[phi.result]
                if v.op.kind == 'undef':
                    continue
                if v.const is not None and not v.ty.is_vec:
                    moves.append((dst, ('imm', v), phi.ty))
                else:
                    src = self.vreg(v)
                    if src is not dst:
                        moves.append((dst, ('reg', src), phi.ty))
        if not moves:
            return
        pending = list(moves)
        done = []
        while pending:
            srcs = {s[1] for _, s, _ in pending if s[0] == 'reg'}
            ready = [m for m in pending if m[0] not in srcs]
            if not ready:
                # a cycle: break it through a temporary
                dst, src, ty = pending[0]
                tmp = self.new('vec' if ty.is_vec else 'gpr')
                self.copy(tmp, src, ty)
                pending[0] = (dst, ('reg', tmp), ty)
                continue
            for m in ready:
                self.copy(m[0], m[1], m[2])
                pending.remove(m)

    def copy(self, dst, src, ty):
        if src[0] == 'imm':
            r = self.materialize_into(dst, src[1])
            return
        if ty.is_vec:
            self.emit('vmovdqa' if self.vex else 'movdqa', Reg(dst, ty.bits), Reg(src[1], ty.bits), acc='w r', kind='copy')
        else:
            bits = 64 if ty.bits > 32 else 32
            self.emit('mov', Reg(dst, bits), Reg(src[1], bits), acc='w r', kind='copy')
            if bits == 32:
                self.clean32.add(dst)

    def materialize_into(self, dst, v):
        """A constant into the register dst: a 32-bit move zero-extends, a
        negative one sign-extends an imm32, the rest take movabs."""
        c, bits = v.const, v.ty.bits
        if bits <= 32 or c < (1 << 32):
            self.emit('mov', Reg(dst, 32), Imm(c & 0xffffffff, 32), acc='w r')
            self.clean32.add(dst)
        elif -(1 << 31) <= ir.signed(c, 64) < 0:
            self.emit('mov', Reg(dst, 64), Imm(ir.signed(c, 64), 32), acc='w r')
        else:
            self.emit('movabs', Reg(dst, 64), Imm(ir.signed(c, 64), 64), acc='w r')

    # ---------------------------------------------------------- terminators

    def terminator(self, t):
        k = t.kind
        if k == 'jmp':
            self.emit('jmp', Label(self.block_of[t.targets[0]]), kind='jmp')
        elif k == 'br':
            yes, no = (self.block_of[x] for x in t.targets)
            self.branch(t.args[0], yes, no, t.attrs.get('weights'))
        elif k == 'ret':
            if getattr(self.block, 'tail', False):
                return
            if t.args:
                v = t.args[0]
                if v.ty.is_vec:
                    self.emit('vmovdqa' if self.vex else 'movdqa', Reg('v0', v.ty.bits), self.reg(v), acc='w r', kind='copy')
                else:
                    bits = 64 if v.ty.bits > 32 else 32
                    if self.imm_ok(v):
                        self.materialize_into('rax', v)
                    else:
                        self.emit('mov', Reg('rax', bits), self.reg(v, bits), acc='w r', kind='copy')
            self.emit('ret', kind='ret')
        elif k == 'unreachable':
            self.emit('ud2', kind='trap')
        else:
            raise Unsupported('terminator ' + k)

    def branch(self, v, yes, no, weights):
        """A branch on v to yes or no: one jump per condition of an and/or tree."""
        op = v.op
        if op in self.skip and op.kind == 'xor' and v.ty is I1:
            self.branch(op.args[0], no, yes, weights[::-1] if weights else None)
            return
        if op in self.skip and op.kind in ('and', 'or') and v.ty is I1:
            a, b = op.args
            mid = self.mf.block(self.block.name + '.c')
            if op.kind == 'and':
                self.branch(a, mid, no, weights)
            else:
                self.branch(a, yes, mid, weights)
            self.block = mid
            self.branch(b, yes, no, weights)
            return
        cc = self.condition(v)
        ins = self.emit('j', Label(yes), cc=cc, kind='jcc')
        ins.ops.append(Label(no))
        self.block.succs = [yes, no]
        self.block.weights = weights

    def condition(self, v):
        """Emit what sets the flags for the i1 v; returns the condition code for v true."""
        op = v.op
        if op.kind == 'icmp' and op.args[1].const is not None and op.args[0].op.kind == 'vop' and op.args[0].op in self.skip:
            cc = self.string_flags(op.args[0].op)
            want = op.args[1].const
            if op.attrs['pred'] == 'eq':
                return cc if want == 1 else INVERT_CC[cc]
            return INVERT_CC[cc] if want == 1 else cc
        if op.kind == 'icmp' and op in self.skip:
            return self.compare(op)
        if op.kind == 'icmp' and op.block is self.fn_block and op in self.cmp_ins:
            ins = self.cmp_ins[op]
            for later in reversed(self.block.ins):
                if later is ins:
                    return self.cmp_cc[op]        # the flags are still the compare's
                if later.op in FLAG_WRITERS or FLAGS in later.imp_def or later.kind in ('call', 'cmp'):
                    break
            return self.compare(op)               # cheaper than testing the byte
        if op.kind == 'vop' and op in self.skip:
            return self.string_flags(op)
        r = self.reg(v, 8)
        self.emit('test', r, r, acc='r r', kind='cmp')
        return 'ne'

    def compare(self, op):
        a, b = op.args
        pred = op.attrs['pred']
        bits = max(a.ty.bits, 8)
        if a in self.kview and pred in ('eq', 'ne') and b.const == 0 and a in self.vregs:
            r = Reg(self.vregs[a], 64 if a.ty.bits > 32 else 32)
            self.emit('test', r, r, acc='r r', kind='cmp')
            return 'e' if pred == 'eq' else 'ne'
        if a in self.kview and pred in ('eq', 'ne') and b.const in (0, a.ty.mask):
            # kortest: ZF for an all-zero mask, CF for an all-one one
            k = Reg(self.kview[a], a.ty.bits)
            self.emit('kortest' + KSUFFIX[a.ty.bits], k, k, acc='r r', kind='cmp')
            if b.const == 0:
                return 'e' if pred == 'eq' else 'ne'
            return 'b' if pred == 'eq' else 'ae'
        if b.const == 0 and pred in ('eq', 'ne') and self.zero_flag_set(a):
            return 'e' if pred == 'eq' else 'ne'
        if a.op in self.skip and a.op.kind == 'load' and a not in self.vregs and not self.imm_ok(b):
            a, b, pred = b, a, SWAP[pred]
        if pred in ('eq', 'ne') and a.op.kind == 'and' and len(a.uses) == 1 and a.op.block is not None and a.op not in self.skip \
                and b.const is not None and b.const == a.op.args[1].const and self.imm_ok(b) and a.op.args[1].const != 0:
            # (x & m) == m: no bit of m clear in x
            x = a.op.args[0]
            t = Reg(self.new('gpr'), bits)
            self.emit('mov', t, self.reg(x, bits), acc='w r', kind='copy')
            self.emit('not', t, acc='rw')
            self.emit('test', t, self.imm(b), acc='r r', kind='cmp')
            return 'e' if pred == 'eq' else 'ne'
        if b.const == 0 and pred in ('eq', 'ne'):
            inner = a.op
            if inner.kind == 'and' and len(a.uses) == 1 and inner.block is not None and inner not in self.skip:
                # the and feeds only this compare: test
                x, y = inner.args
                if self.imm_ok(x) and not self.imm_ok(y):
                    x, y = y, x
                if self.imm_ok(y):
                    self.emit('test', self.operand(x, bits, allow_imm=False), self.imm(y), acc='r r', kind='cmp')
                else:
                    other = self.operand(y, bits, allow_imm=False)
                    if isinstance(other, Mem):
                        self.emit('test', other, self.reg(x, bits), acc='r r', kind='cmp')   # test's memory operand comes first
                    else:
                        self.emit('test', self.reg(x, bits), other, acc='r r', kind='cmp')
                return 'e' if pred == 'eq' else 'ne'
            r = self.operand(a, bits, allow_imm=False, allow_mem=False)
            self.emit('test', r, r, acc='r r', kind='cmp')
            return 'e' if pred == 'eq' else 'ne'
        if self.imm_ok(b):
            if bits == 64 and 0 <= b.const < (1 << 31) and a in self.vregs and self.vregs[a] in self.clean32 and pred in ('eq', 'ne', 'ult', 'ule', 'ugt', 'uge'):
                bits = 32      # the upper half is zero: the shorter compare
            self.emit('cmp', self.operand(a, bits, allow_imm=False), self.imm(b) if bits == 64 else Imm(b.const, 32), acc='r r', kind='cmp')
        else:
            self.emit('cmp', self.operand(a, bits, allow_imm=False, allow_mem=False), self.operand(b, bits, allow_imm=False), acc='r r', kind='cmp')
        return CC_OF[pred]

    def zero_flag_set(self, v):
        """The flags still hold the result of the instruction that computed v (in this block)."""
        r = self.vregs.get(v)
        if r is None or v.op.block is not self.fn_block:
            return False
        for ins in reversed(self.block.ins):
            if any(isinstance(o, Reg) and o.r is r and 'w' in a for o, a in zip(ins.ops, ins.acc)):
                return ins.op in ZERO_FLAG and (ins.op not in ('shl', 'shr', 'sar'))
            if ins.op in FLAG_WRITERS or FLAGS in ins.imp_def or ins.kind in ('call', 'cmp'):
                return False
        return False

    def dies_at(self, v, op):
        """Every use of v is op or comes before it in op's block."""
        at = self.uses_at[op]
        return all(u is op or self.uses_at.get(u, (None, 0))[0] is at[0] and self.uses_at[u][1] < at[1] for u in v.uses)

    def string_flags(self, op):
        """pcmpistri/vptest for a flag the branch reads."""
        name = op.attrs['name']
        if name == 'testz':
            a, b = op.args
            self.emit('vptest' if self.vex else 'ptest', self.reg(a), self.operand(b, allow_imm=False), acc='r r', kind='cmp')
            return 'e'
        self.pcmpistr(op, 'pcmpistri', op.args)
        return {'cmpistra': 'a', 'cmpistrc': 'b', 'cmpistrz': 'e'}[name]

    def pcmpistr(self, op, mnemonic, args):
        a, b, imm = args
        m = ('v' if self.vex else '') + mnemonic
        ins = self.emit(m, self.reg(a), self.operand(b, allow_imm=False), Imm(imm.const & 0xff, 8), acc='r r r')
        ins.imp_def = {'rcx'} if mnemonic == 'pcmpistri' else {'v0'}
        ins.imp_def.add(FLAGS)
        return ins

    # -------------------------------------------------------------- ops

    def select(self, op):
        k = op.kind
        method = getattr(self, 'sel_' + k, None)
        if method is None:
            raise Unsupported('%s: no selection for %s' % (self.fn.name, k))
        method(op)

    def sel_alloca(self, op):
        align = op.attrs.get('align', 1)
        if align > 16:
            # the frame is 16-aligned: a wider alignment is made in a register
            # over a slot padded by the difference
            slot = self.slots[op] = self.mf.slot(op.attrs['size'] + align - 16, 16)
            t = self.new('gpr')
            self.emit('lea', Reg(t, 64), Mem(8, slot=slot, disp=align - 1), acc='w r')
            self.emit('and', Reg(t, 64), Imm(-align, 32), acc='rw r')
            self.vregs[op.result] = t
            return
        self.slot_of(op)
        # its address is a memory operand wherever it is used; a register only on demand
        self.addr_only.add(op)

    def sel_load(self, op):
        self.vregs[op.result] = self.select_load(op)

    def select_load(self, op):
        v = op.result
        ty = v.ty
        mem = self.mem_operand(op.args[0], ty.size)
        if ty.is_vec:
            r = self.new('vec')
            aligned = op.attrs.get('align', 1) >= ty.size
            self.emit(self.vmov_load(ty.bits, aligned), Reg(r, ty.bits), mem, acc='w r')
            return r
        r = self.new('gpr')
        if ty.bits == 8:
            self.emit('movzb', Reg(r, 32), mem, acc='w r')
            self.clean32.add(r)
        elif ty.bits == 16:
            self.emit('movzw', Reg(r, 32), mem, acc='w r')
            self.clean32.add(r)
        else:
            self.emit('mov', Reg(r, ty.bits), mem, acc='w r')
            if ty.bits == 32:
                self.clean32.add(r)
        return r

    def sel_store(self, op):
        addr, v = op.args
        ty = v.ty
        mem = self.mem_operand(addr, ty.size)
        if op in self.rmw:
            o, x = self.rmw[op]
            bits = ty.bits
            other = self.imm(x) if self.imm_ok(x) and bits <= 32 or x.const is not None and bits == 64 and self.imm_ok(x) else self.reg(x, bits)
            self.emit(ARITH[o.kind], mem, other, acc='rw r')
            return
        if ty.is_vec:
            aligned = op.attrs.get('align', 1) >= ty.size
            self.emit(self.vmov_load(ty.bits, aligned), mem, self.reg(v), acc='w r')
            return
        bits = max(ty.bits, 8)
        if self.imm_ok(v) and bits <= 64:
            self.emit('mov', mem, Imm(ir.signed(v.const, min(bits, 32)) if bits <= 32 else ir.signed(v.const, 64), min(bits, 32)), acc='w r')
        else:
            self.emit('mov', mem, self.reg(v, bits), acc='w r')

    def sel_memset(self, op):
        addr, value, n = op.args
        if n.const is None or value.const is None:
            raise Unsupported('memset with a variable size or value')
        size, byte = n.const, value.const & 0xff
        done = 0
        width = 32 if self.vex and size >= 32 else 16
        while done < size:
            chunk = min(width, size - done)
            if chunk >= 16:
                chunk = width if size - done >= width else 16
                vbytes = bytes([byte]) * chunk
                vt = ir.Ty('v', chunk * 8)
                r = self.materialize(self.fn.const(vt, vbytes))
                mem = self.mem_operand(addr, chunk)
                mem.disp += done
                self.emit(self.vmov_load(chunk * 8, False), mem, Reg(r, chunk * 8), acc='w r')
            else:
                chunk = 8 if chunk >= 8 else 4 if chunk >= 4 else 2 if chunk >= 2 else 1
                value64 = int.from_bytes(bytes([byte]) * chunk, 'little')
                mem = self.mem_operand(addr, chunk)
                mem.disp += done
                if chunk == 8 and not -(1 << 31) <= ir.signed(value64, 64) < (1 << 31):
                    t = self.new('gpr')
                    self.emit('movabs', Reg(t, 64), Imm(ir.signed(value64, 64), 64), acc='w r')
                    self.emit('mov', mem, Reg(t, 64), acc='w r')
                else:
                    self.emit('mov', mem, Imm(ir.signed(value64, chunk * 8) if chunk < 8 else ir.signed(value64, 64), min(chunk * 8, 32)), acc='w r')
            done += chunk

    def sel_memcpy(self, op):
        dst, src, n = op.args
        if n.const is None:
            raise Unsupported('memcpy with a variable size')
        size, done = n.const, 0
        if size >= 128 and size % 8 == 0:
            self.emit('mov', Reg('rdi', 64), Reg(self.vreg(dst), 64), acc='w r', kind='copy')
            self.emit('mov', Reg('rsi', 64), Reg(self.vreg(src), 64), acc='w r', kind='copy')
            self.emit('mov', Reg('rcx', 32), Imm(size // 8, 32), acc='w r')
            ins = self.emit('movsq', rep='rep')
            ins.imp_use, ins.imp_def = {'rdi', 'rsi', 'rcx'}, {'rdi', 'rsi', 'rcx'}
            ins.kind = 'string'
            return
        while done < size:
            chunk = 32 if self.vex and size - done >= 32 else 16 if size - done >= 16 else 8 if size - done >= 8 else 4 if size - done >= 4 else 2 if size - done >= 2 else 1
            s, d = self.mem_operand(src, chunk), self.mem_operand(dst, chunk)
            s.disp += done
            d.disp += done
            if chunk >= 16:
                r = self.new('vec')
                self.emit(self.vmov_load(chunk * 8, False), Reg(r, chunk * 8), s, acc='w r')
                self.emit(self.vmov_load(chunk * 8, False), d, Reg(r, chunk * 8), acc='w r')
            else:
                r = self.new('gpr')
                bits = chunk * 8
                self.emit('movzb' if bits == 8 else 'movzw' if bits == 16 else 'mov', Reg(r, max(bits, 32)), s, acc='w r')
                self.emit('mov', d, Reg(r, bits), acc='w r')
            done += chunk

    # scalar arithmetic ---------------------------------------------------

    def sel_add(self, op):
        self.arith(op)

    def sel_sub(self, op):
        self.arith(op)

    def sel_and(self, op):
        self.arith(op)

    def sel_or(self, op):
        self.arith(op)

    def sel_xor(self, op):
        self.arith(op)

    def arith(self, op):
        if op.ty.is_vec:
            self.vector_logic(op)
            return
        a, b = op.args
        k = op.kind
        bits = 8 if op.ty is I1 else 64 if op.ty.bits == 64 else 32
        dst = self.define(op.result)
        dst = Reg(dst.r, bits)
        if k == 'sub' and a.const == 0:
            self.emit('mov', dst, self.reg(b, bits), acc='w r', kind='copy')
            self.emit('neg', dst, acc='rw')
            return
        if k == 'xor' and b.const is not None and b.const == op.ty.mask and op.ty.bits >= 8:
            self.emit('mov', dst, self.reg(a, bits), acc='w r', kind='copy')
            self.emit('not', dst, acc='rw')
            return
        if k in ('xor', 'and') and self.bmi and bits >= 32 and b.op.kind == 'add' and b.op.args[1].const == op.ty.mask and b.op.args[0] is a \
                and len(b.uses) == 1:
            self.emit('blsmsk' if k == 'xor' else 'blsr', dst, self.reg(a, bits), acc='w r')     # the bits up to the lowest set one; that one cleared
            if bits == 32:
                self.clean32.add(dst.r)
            return
        if k in ('or', 'xor') and bits == 64 and b.const is not None and is_pow2(b.const) and not self.imm_ok(b):
            # a bit beyond an imm32: bts/btc with the bit's index
            self.emit('mov', dst, self.reg(a, bits), acc='w r', kind='copy')
            self.emit('bts' if k == 'or' else 'btc', dst, Imm(b.const.bit_length() - 1, 8), acc='rw r')
            return
        if k in ('or', 'xor') and b.op in self.skip and b.op.kind == 'shl':
            # x | 1 << n, x ^ 1 << n
            self.emit('mov', dst, self.reg(a, bits), acc='w r', kind='copy')
            self.emit('bts' if k == 'or' else 'btc', dst, self.count(b.op.args[1], bits), acc='rw r')
            return
        if k == 'and' and bits == 64 and b.const is not None and b.const < (1 << 31):
            # the result fits 32 bits: the shorter form, which zero-extends
            dst = Reg(dst.r, 32)
            self.emit('mov', dst, self.reg(a, 32), acc='w r', kind='copy')
            self.emit('and', dst, Imm(b.const, 32), acc='rw r')
            self.clean32.add(dst.r)
            return
        tested = any(u.kind == 'icmp' and u.args[0] is op.result and u.args[1].const == 0 and u.attrs['pred'] in ('eq', 'ne')
                     and self.uses_at[u][0] is op.block for u in op.result.uses)
        if k == 'add' and bits >= 32 and not tested:
            for x, y in ((a, b), (b, a)):
                if y.op.kind == 'shl' and y.op.args[1].const in (1, 2, 3) and len(y.uses) == 1 and y.op.block is op.block and x.const is None:
                    self.emit('lea', dst, Mem(bits // 8, base=self.vreg(x), index=self.vreg(y.op.args[0]), scale=1 << y.op.args[1].const), acc='w r', kind='lea')
                    return
        # add with two registers: lea keeps both operands
        if k == 'add' and b.const is None and bits >= 32 and not (b.op in self.skip and b.op.kind == 'load'):
            self.emit('lea', dst, Mem(bits // 8, base=self.vreg(a), index=self.vreg(b)), acc='w r', kind='lea')
            return
        if k == 'add' and self.imm_ok(b) and bits >= 32 and b.const != 0 and b.op.kind == 'const' and not tested:
            # lea dst, [a + c]: no copy and no flags
            self.emit('lea', dst, Mem(bits // 8, base=self.vreg(a), disp=ir.signed(b.const, bits)), acc='w r', kind='lea')
            return
        if k in ('add', 'and', 'or', 'xor') and b.const is None and not (b.op in self.skip and b.op.kind == 'load') \
                and self.dies_at(b, op) and not self.dies_at(a, op):
            a, b = b, a      # the operand that dies here is overwritten: no copy for the other
        self.emit('mov', dst, self.reg(a, bits), acc='w r', kind='copy')
        self.emit(ARITH[k], dst, self.operand(b, bits), acc='rw r')

    def vector_logic(self, op):
        a, b = op.args
        name = op.kind
        legacy, vex = VOP2[name]
        dst = self.define(op.result)
        if self.vex:
            self.emit(vex, dst, self.reg(a), self.vop_operand(b), acc='w r r')
        else:
            if not (b.op in self.skip and b.op.kind == 'load') and b.const is None and self.dies_at(b, op) and not self.dies_at(a, op):
                a, b = b, a
            self.emit('movdqa', dst, self.reg(a), acc='w r', kind='copy')
            self.emit(legacy, dst, self.vop_operand(b), acc='rw r')

    def sel_mul(self, op):
        a, b = op.args
        bits = 64 if op.ty.bits == 64 else 32
        dst = Reg(self.define(op.result).r, bits)
        if self.imm_ok(b):
            self.emit('imul', dst, self.reg(a, bits), self.imm(b), acc='w r r')
        else:
            self.emit('mov', dst, self.reg(a, bits), acc='w r', kind='copy')
            self.emit('imul', dst, self.operand(b, bits, allow_imm=False), acc='rw r')

    def count(self, b, bits, keep=None):
        """The register for a shift count: the hardware keeps its low bits only
        (keep, bits - 1 by default), so a mask of them and the extensions go."""
        keep = bits - 1 if keep is None else keep
        v = b
        while v.op.kind in ('trunc', 'zext') and v.ty.bits >= 8 and v.op.args[0].ty.bits >= 8 and not v.op.args[0].ty.is_vec:
            v = v.op.args[0]
        if v.op.kind == 'and' and v.op.args[1].const is not None and v.op.args[1].const & keep == keep:
            v = v.op.args[0]
        if v.op.kind == 'shl' and v.op.args[1].const in (1, 2, 3) and v.op.args[0].op.kind == 'and' and v.op.args[0].op.args[1].const is not None \
                and len(v.uses) == 1 and len(v.op.args[0].uses) == 1:
            k, m = v.op.args[1].const, v.op.args[0].op.args[1].const
            if (m << k) & keep == keep & ~((1 << k) - 1):
                t = self.new('gpr')
                self.emit('lea', Reg(t, bits), Mem(bits // 8, index=self.vreg(v.op.args[0].op.args[0]), scale=1 << k), acc='w r', kind='lea')
                return Reg(t, bits)
        return self.reg(v, bits)

    def shift(self, op, legacy, bmi):
        a, b = op.args
        bits = 64 if op.ty.bits == 64 else 32
        dst = Reg(self.define(op.result).r, bits)
        if b.const is not None:
            self.emit('mov', dst, self.reg(a, bits), acc='w r', kind='copy')
            self.emit(legacy, dst, Imm(b.const & (bits - 1), 8), acc='rw r')
            return
        if self.bmi:
            self.emit(bmi, dst, self.operand(a, bits, allow_imm=False), self.count(b, bits), acc='w r r')
            return
        self.emit('mov', Reg('rcx', 32), self.count(b, 32), acc='w r', kind='copy')
        self.emit('mov', dst, self.reg(a, bits), acc='w r', kind='copy')
        ins = self.emit(legacy, dst, Reg('rcx', 8), acc='rw r')

    def sel_shl(self, op):
        self.shift(op, 'shl', 'shlx')

    def sel_lshr(self, op):
        if op.ty.bits < 32:
            # the upper bits of the register may hold anything: shift a zero-extended copy
            a, b = op.args
            t = self.new('gpr')
            self.emit('movzb' if op.ty.bits == 8 else 'movzw', Reg(t, 32), self.reg(a, op.ty.bits), acc='w r')
            self.clean32.add(t)
            dst = Reg(self.define(op.result).r, 32)
            if b.const is not None:
                self.emit('mov', dst, Reg(t, 32), acc='w r', kind='copy')
                self.emit('shr', dst, Imm(b.const & 31, 8), acc='rw r')
            elif self.bmi:
                self.emit('shrx', dst, Reg(t, 32), self.reg(b, 32), acc='w r r')
            else:
                self.emit('mov', Reg('rcx', 32), self.reg(b, 32), acc='w r', kind='copy')
                self.emit('mov', dst, Reg(t, 32), acc='w r', kind='copy')
                self.emit('shr', dst, Reg('rcx', 8), acc='rw r')
            return
        self.shift(op, 'shr', 'shrx')

    def sel_ashr(self, op):
        if op.ty.bits < 32:
            a, b = op.args
            t = self.new('gpr')
            self.emit('movsb' if op.ty.bits == 8 else 'movsw', Reg(t, 32), self.reg(a, op.ty.bits), acc='w r')
            dst = Reg(self.define(op.result).r, 32)
            if b.const is not None:
                self.emit('mov', dst, Reg(t, 32), acc='w r', kind='copy')
                self.emit('sar', dst, Imm(b.const & 31, 8), acc='rw r')
            elif self.bmi:
                self.emit('sarx', dst, Reg(t, 32), self.reg(b, 32), acc='w r r')
            else:
                self.emit('mov', Reg('rcx', 32), self.reg(b, 32), acc='w r', kind='copy')
                self.emit('mov', dst, Reg(t, 32), acc='w r', kind='copy')
                self.emit('sar', dst, Reg('rcx', 8), acc='rw r')
            return
        self.shift(op, 'sar', 'sarx')

    def division(self, op):
        a, b = op.args
        bits = 64 if op.ty.bits == 64 else 32
        signed_ = op.kind in ('sdiv', 'srem')
        self.emit('mov', Reg('rax', bits), self.reg(a, bits), acc='w r', kind='copy')
        if signed_:
            self.emit('cqto' if bits == 64 else 'cltd', imp_def={'rdx'}, imp_use={'rax'})
        else:
            self.emit('xor', Reg('rdx', 32), Reg('rdx', 32), acc='w r', kind='zero')
        d = self.reg(b, bits)
        ins = self.emit('idiv' if signed_ else 'div', d, acc='r')
        ins.imp_use, ins.imp_def = {'rax', 'rdx'}, {'rax', 'rdx', FLAGS}
        dst = Reg(self.define(op.result).r, bits)
        self.emit('mov', dst, Reg('rax' if op.kind in ('udiv', 'sdiv') else 'rdx', bits), acc='w r', kind='copy')

    sel_udiv = sel_sdiv = sel_urem = sel_srem = division

    def sel_icmp(self, op):
        # not folded: a byte value from the flags
        cc = self.compare(op)
        self.cmp_ins[op], self.cmp_cc[op] = self.block.ins[-1], cc
        dst = self.define(op.result)
        self.emit('set', Reg(dst.r, 8), cc=cc, acc='w', kind='setcc')
        self.mf.setcc_defs = getattr(self.mf, 'setcc_defs', set()) | {dst.r}

    def sel_zext(self, op):
        a = op.args[0]
        src_bits, dst_bits = a.ty.bits, op.ty.bits
        if a.ty is I1:
            if a.op.kind == 'icmp' and a.op in self.skip:
                dst = Reg(self.define(op.result).r, 32)
                self.emit('xor', dst, dst, acc='w r', kind='zero')
                cc = self.condition(a)
                self.emit('set', Reg(dst.r, 8), cc=cc, acc='w', kind='setcc')
                return
            dst = Reg(self.define(op.result).r, 32)
            self.emit('movzb', dst, self.reg(a, 8), acc='w r')
            return
        if a.op in self.skip and a.op.kind == 'load' and a not in self.vregs:
            mem = self.mem_operand(a.op.args[0], a.ty.size)
            dst = self.define(op.result)
            if src_bits == 8:
                self.emit('movzb', Reg(dst.r, 32), mem, acc='w r')
            elif src_bits == 16:
                self.emit('movzw', Reg(dst.r, 32), mem, acc='w r')
            else:
                self.emit('mov', Reg(dst.r, 32), mem, acc='w r')
            self.clean32.add(dst.r)
            return
        src = self.vreg(a)
        if src_bits == 32:
            if src in self.clean32:
                self.vregs[op.result] = src
            else:
                dst = Reg(self.define(op.result).r, 32)
                self.emit('mov', dst, Reg(src, 32), acc='w r', kind='copy32')
            return
        if src in self.zext_ok and a.op.kind not in ('trunc', 'phi'):
            self.vregs[op.result] = src
            if dst_bits == 32:
                self.clean32.add(src)
            return
        dst = self.define(op.result)
        self.emit('movzb' if src_bits == 8 else 'movzw', Reg(dst.r, 32), Reg(src, src_bits), acc='w r')
        self.clean32.add(dst.r)

    def sel_sext(self, op):
        a = op.args[0]
        src_bits, dst_bits = a.ty.bits, op.ty.bits
        dst = self.define(op.result)
        if a.op in self.skip and a.op.kind == 'load' and a not in self.vregs:
            mem = self.mem_operand(a.op.args[0], a.ty.size)
            m = {8: 'movsb', 16: 'movsw', 32: 'movslq'}[src_bits]
            self.emit(m, Reg(dst.r, dst_bits if dst_bits == 64 else 32), mem, acc='w r')
            return
        m = {8: 'movsb', 16: 'movsw', 32: 'movslq'}[src_bits]
        self.emit(m, Reg(dst.r, dst_bits if dst_bits == 64 else 32), self.reg(a, src_bits), acc='w r')

    def sel_trunc(self, op):
        self.vregs[op.result] = self.vreg(op.args[0])

    def sel_p2i(self, op):
        self.vregs[op.result] = self.vreg(op.args[0])

    def sel_i2p(self, op):
        self.vregs[op.result] = self.vreg(op.args[0])

    def sel_ptradd(self, op):
        dst = self.define(op.result)
        self.emit('lea', dst, self.address_of_ptradd(op), acc='w r', kind='lea')

    def address_of_ptradd(self, op):
        a, b = op.args
        base = self.vreg(a)
        if b.const is not None:
            return Mem(8, base=base, disp=ir.signed(b.const, 64))
        if b.op in self.addr_only and b.op.kind == 'shl':
            return Mem(8, base=base, index=self.vreg(b.op.args[0]), scale=1 << b.op.args[1].const)
        if b.op in self.addr_only and b.op.kind == 'mul':
            return Mem(8, base=base, index=self.vreg(b.op.args[0]), scale=b.op.args[1].const)
        return Mem(8, base=base, index=self.vreg(b))

    def sel_select(self, op):
        c, a, b = op.args
        bits = 64 if op.ty.bits == 64 else 32
        dst = Reg(self.define(op.result).r, bits)
        # cmov reads the false value into the destination first
        if self.imm_ok(b):
            self.materialize_into(dst.r, b)
        else:
            self.emit('mov', dst, self.reg(b, bits), acc='w r', kind='copy')
        if self.imm_ok(a):
            t = self.materialize(a)
            true_reg = Reg(t, bits)
        else:
            true_reg = self.reg(a, bits)
        cc = self.condition(c)
        self.emit('cmov', dst, true_reg, cc=cc, acc='rw r')

    def sel_cttz(self, op):
        a = op.args[0]
        bits = 64 if a.ty.bits == 64 else 32
        dst = Reg(self.define(op.result).r, bits)
        src = self.operand(a, bits, allow_imm=False)
        if self.bmi or op.attrs.get('zero_undef'):
            self.emit('tzcnt', dst, src, acc='w r')      # rep bsf on hardware without BMI: the same for a nonzero input, and no false dependency
        else:
            self.emit('bsf', dst, src, acc='w r')
            if not op.attrs.get('zero_undef'):
                # bsf leaves the destination undefined for zero: the width then
                t = self.new('gpr')
                self.emit('mov', Reg(t, 32), Imm(bits, 32), acc='w r')
                self.emit('cmov', dst, Reg(t, bits), cc='e', acc='rw r')
        if bits == 32:
            self.clean32.add(dst.r)

    def sel_bzhi(self, op):
        a, n = op.args
        bits = 64 if op.ty.bits == 64 else 32
        dst = Reg(self.define(op.result).r, bits)
        self.emit('bzhi', dst, self.operand(a, bits, allow_imm=False), self.count(n, bits, 255), acc='w r r')
        if bits == 32:
            self.clean32.add(dst.r)

    def sel_call(self, op):
        callee = op.args[0]
        args = op.args[1:]
        gi = vi = 0
        moves = []
        for a in args:
            if a.ty.is_vec:
                moves.append(('v%d' % vi, a))
                vi += 1
            else:
                moves.append((ARG_GPR[gi], a))
                gi += 1
        if gi > 6 or vi > 8:
            raise Unsupported('%s: too many arguments in a call' % self.fn.name)
        # a parameter passed on in another register leaves its own before that one is written
        # (so it can keep its register from the entry to here)
        ordered, waiting = [], list(moves)
        while waiting:
            # a copy into a register some waiting copy still reads a parameter from waits
            step = [m for m in waiting if not any(self.param_reg.get(m2[1]) == m[0] and m2 is not m for m2 in waiting)] or waiting[:1]
            for m in step:
                waiting.remove(m)
            ordered.extend(step)
        moves = ordered
        for preg, a in moves:
            if a.ty.is_vec:
                self.emit('vmovdqa' if self.vex else 'movdqa', Reg(preg, a.ty.bits), self.reg(a), acc='w r', kind='copy')
            elif self.imm_ok(a):
                self.materialize_into(preg, a)
            else:
                bits = 64 if a.ty.bits > 32 else 32
                self.emit('mov', Reg(preg, bits), self.reg(a, bits), acc='w r', kind='copy')
        # is this a tail call? the next op is the return of its result
        term = op.block.term
        body = op.block.ops
        tail = (body[-1] is op and term.kind == 'ret' and (not term.args and op.result is None or term.args and term.args[0] is op.result))
        if callee.op.kind == 'sym':
            target = Sym(callee.op.attrs['name'])
        elif callee.op.kind == 'load' and callee.op in self.skip and callee not in self.vregs:
            target = self.mem_operand(callee.op.args[0], 8)
        else:
            target = Reg(self.vreg(callee), 64)
        ins = self.emit('jmp' if tail else 'call', target, acc='r', kind='tail' if tail else 'call')
        ins.imp_use = {preg for preg, _ in moves}
        if tail:
            self.tail_done = True
            self.block.tail = True
            return
        ins.imp_def = set(CALLER_SAVED) | set(VEC) | set(MASK) | {FLAGS}
        if op.result is not None:
            v = op.result
            if v.ty.is_vec:
                dst = self.define(v)
                self.emit('vmovdqa' if self.vex else 'movdqa', dst, Reg('v0', v.ty.bits), acc='w r', kind='copy')
            else:
                bits = 64 if v.ty.bits > 32 else 32
                dst = Reg(self.define(v).r, bits)
                self.emit('mov', dst, Reg('rax', bits), acc='w r', kind='copy')

    def sel_asm(self, op):
        code = op.attrs['code']
        outputs, inputs = op.attrs['outputs'], op.attrs['inputs']
        if code == '':
            # a barrier for the compiler: the outputs are the inputs
            n_inout = sum('+' in c for c in outputs)
            for r, (c, a) in zip(op.results, zip(outputs, op.args[:n_inout])):
                self.vregs[r] = self.vreg(a)
            return
        fixed = {'a': 'rax', 'b': 'rbx', 'c': 'rcx', 'd': 'rdx', 'D': 'rdi', 'S': 'rsi'}
        inout = [c for c in outputs if '+' in c]
        values = list(op.args)
        regs_in = []
        for c, a in zip(inout + inputs, values):
            preg = fixed[c.strip('+=&')]
            bits = 64 if a.ty.bits > 32 else 32
            if self.imm_ok(a):
                self.materialize_into(preg, a)
            else:
                self.emit('mov', Reg(preg, bits), self.reg(a, bits), acc='w r', kind='copy')
            regs_in.append(preg)
        if code == 'cpuid':
            ins = self.emit('cpuid')
            ins.imp_use, ins.imp_def = {'rax', 'rcx'}, {'rax', 'rbx', 'rcx', 'rdx'}
        elif code == 'xgetbv':
            ins = self.emit('xgetbv')
            ins.imp_use, ins.imp_def = {'rcx'}, {'rax', 'rdx'}
        elif code == 'rep movsb':
            ins = self.emit('movsb', rep='rep')
            ins.imp_use, ins.imp_def = {'rdi', 'rsi', 'rcx'}, {'rdi', 'rsi', 'rcx'}
            ins.kind = 'string'
        elif code == 'rep stosb':
            ins = self.emit('stosb', rep='rep')
            ins.imp_use, ins.imp_def = {'rdi', 'rcx', 'rax'}, {'rdi', 'rcx'}
            ins.kind = 'string'
        else:
            raise Unsupported('asm ' + code)
        for r, c in zip(op.results, outputs):
            preg = fixed[c.strip('+=&')]
            bits = 64 if r.ty.bits > 32 else 32
            dst = Reg(self.define(r).r, bits)
            self.emit('mov', dst, Reg(preg, bits), acc='w r', kind='copy')

    # ------------------------------------------------------------ vectors

    def vop_operand(self, v):
        if v.op.kind == 'const' and v.ty.is_vec:
            r = self.const_reg(v)
            return Reg(r, v.ty.bits) if r is not None else self.constant_operand(v.const)
        if v.op.kind == 'vop' and v.op in self.skip and v.op.attrs['name'] == 'masked_load' and v not in self.vregs:
            return self.mem_operand(v.op.args[1], v.ty.size)
        return self.operand(v, allow_imm=False)

    def sel_vop(self, op):
        name = op.attrs['name']
        args = op.args
        if name in VOP2:
            legacy, vex = VOP2[name]
            a, b = args
            dst = self.define(op.result)
            if self.vex:
                self.emit(vex, dst, self.reg(a), self.vop_operand(b), acc='w r r')
            else:
                if name in ('cmpeq_epi8', 'min_epu8', 'add_epi8', 'and', 'or', 'xor') and not (b.op in self.skip and b.op.kind == 'load') \
                        and b.const is None and self.dies_at(b, op) and not self.dies_at(a, op):
                    a, b = b, a      # the operand that dies here is overwritten
                self.emit('movdqa', dst, self.reg(a), acc='w r', kind='copy')
                self.emit(legacy, dst, self.vop_operand(b), acc='rw r')
            return
        if name == 'movemask8':
            dst = Reg(self.define(op.result).r, 32)
            self.emit('vpmovmskb' if self.vex else 'pmovmskb', dst, self.reg(args[0]), acc='w r')
            self.clean32.add(dst.r)
            return
        if name == 'lshr16':
            a, n = args
            dst = self.define(op.result)
            if self.vex:
                self.emit('vpsrlw', dst, self.reg(a), Imm(n.const & 0xff, 8), acc='w r r')
            else:
                self.emit('movdqa', dst, self.reg(a), acc='w r', kind='copy')
                self.emit('psrlw', dst, Imm(n.const & 0xff, 8), acc='rw r')
            return
        if name == 'splat8':
            self.splat(op, args[0])
            return
        if name in MASK_CMP:
            self.mask_compare(op, name, args[0], args[1], None)
            return
        if name.endswith('_masked'):
            base = name[:-len('_masked')] + '_mask'
            self.mask_compare(op, base, args[1], args[2], args[0])
            return
        if name == 'masked_load':
            k, addr = args
            mask = self.mask_reg(k)       # before the load: the kmov must precede it
            dst = self.define(op.result)
            self.emit('vmovdqu8', dst, self.mem_operand(addr, op.ty.size), acc='w r', mask=mask, zero=True)
            return
        if name == 'masked_store':
            addr, k, v = args
            mask = self.mask_reg(k)
            self.emit('vmovdqu8', self.mem_operand(addr, v.ty.size), self.reg(v), acc='w r', mask=mask)
            return
        if name == 'ternary32':
            a, b, c, imm = args
            imm = imm.const & 0xff
            # the destination is overwritten: the operand that dies here, the truth table permuted to match
            if not self.dies_at(a, op):
                if b.const is None and self.dies_at(b, op):
                    a, b, imm = b, a, permute_ternary(imm, (1, 0, 2))
                elif c.const is None and not (c.op in self.skip and c.op.kind == 'load') and self.dies_at(c, op):
                    a, c, imm = c, a, permute_ternary(imm, (2, 1, 0))
            dst = self.define(op.result)
            self.emit('vmovdqa', dst, self.reg(a), acc='w r', kind='copy')
            self.emit('vpternlogd', dst, self.reg(b), self.vop_operand(c), Imm(imm, 8), acc='rw r r r')
            return
        if name == 'constant64_reversed':
            hi, lo = args
            dst = self.define(op.result)
            pair = self.adjacent_loads(lo, hi)
            if pair is not None:
                self.emit(self.vmov_load(128, False), Reg(dst.r, 128), pair, acc='w r')
                return
            if self.vex:
                self.emit('vmovq', Reg(dst.r, 128), self.reg(lo, 64), acc='w r')
                self.emit('vpinsrq', Reg(dst.r, 128), Reg(dst.r, 128), self.reg(hi, 64), Imm(1, 8), acc='w r r r')
            else:
                t = self.new('vec')
                self.emit('movq', Reg(dst.r, 128), self.reg(lo, 64), acc='w r')
                self.emit('movq', Reg(t, 128), self.reg(hi, 64), acc='w r')
                self.emit('punpcklqdq', Reg(dst.r, 128), Reg(t, 128), acc='rw r')
            return
        if name == 'insert64':
            v, x, imm = args
            dst = self.define(op.result)
            if self.vex:
                self.emit('vpinsrq', Reg(dst.r, 128), self.reg(v, 128), self.reg(x, 64), Imm(imm.const & 1, 8), acc='w r r r')
            else:
                self.emit('movdqa', Reg(dst.r, 128), self.reg(v, 128), acc='w r', kind='copy')
                self.emit('pinsrq', Reg(dst.r, 128), self.reg(x, 64), Imm(imm.const & 1, 8), acc='rw r r')
            return
        if name == 'extract_low32':
            dst = Reg(self.define(op.result).r, 32)
            self.emit('vmovd' if self.vex else 'movd', dst, self.reg(args[0], 128), acc='w r')
            self.clean32.add(dst.r)
            return
        if name == 'insert_low32':
            dst = self.define(op.result)
            self.emit('vmovd' if self.vex else 'movd', Reg(dst.r, 128), self.reg(args[0], 32), acc='w r')
            return
        if name == 'store_low':
            addr, v = args
            width = op.attrs['width']
            mem = self.mem_operand(addr, width)
            self.emit(('v' if self.vex else '') + ('movq' if width == 8 else 'movd'), mem, self.reg(v, 128), acc='w r')
            return
        if name in ('cmpistri', 'cmpistra', 'cmpistrc', 'cmpistrz'):
            ins = self.pcmpistr(op, 'pcmpistri', args)
            if name == 'cmpistri':
                dst = Reg(self.define(op.result).r, 32)
                self.emit('mov', dst, Reg('rcx', 32), acc='w r', kind='copy')
                self.clean32.add(dst.r)
            else:
                dst = Reg(self.define(op.result).r, 32)
                cc = {'cmpistra': 'a', 'cmpistrc': 'b', 'cmpistrz': 'e'}[name]
                self.emit('set', Reg(dst.r, 8), cc=cc, acc='w', kind='setcc')
                self.emit('movzb', dst, Reg(dst.r, 8), acc='w r')
                self.clean32.add(dst.r)
            return
        if name == 'cmpistrm':
            self.pcmpistr(op, 'pcmpistrm', args)
            dst = self.define(op.result)
            self.emit('vmovdqa' if self.vex else 'movdqa', Reg(dst.r, 128), Reg('v0', 128), acc='w r', kind='copy')
            return
        if name == 'broadcast128':
            x = args[0]
            dst = self.define(op.result)
            if x.const is not None and self.const_reg(x) is None:
                self.emit('vbroadcasti128', Reg(dst.r, 256), self.constant_operand(x.const), acc='w r')
                return
            src = self.vreg(x)
            self.emit('vinserti128', Reg(dst.r, 256), Reg(src, 256), Reg(src, 128), Imm(1, 8), acc='w r r r')
            return
        if name in ('widen_undef', 'truncate'):
            self.vregs[op.result] = self.vreg(args[0])
            return
        if name == 'insert128':
            a, b, imm = args
            dst = self.define(op.result)
            self.emit('vinserti128', Reg(dst.r, 256), self.reg(a, 256), self.vop_operand(b), Imm(imm.const & 1, 8), acc='w r r r')
            return
        if name == 'extract128':
            a, imm = args
            dst = self.define(op.result)
            self.emit('vextracti128', Reg(dst.r, 128), self.reg(a, 256), Imm(imm.const & 1, 8), acc='w r r')
            return
        if name == 'testz':
            a, b = args
            self.emit('vptest' if self.vex else 'ptest', self.reg(a), self.vop_operand(b), acc='r r', kind='cmp')
            dst = Reg(self.define(op.result).r, 32)
            self.emit('set', Reg(dst.r, 8), cc='e', acc='w', kind='setcc')
            self.emit('movzb', dst, Reg(dst.r, 8), acc='w r')
            self.clean32.add(dst.r)
            return
        if name in ('constant8',):
            raise Unsupported('%s: a vector built from variable bytes' % self.fn.name)
        raise Unsupported('%s: vector operation %s' % (self.fn.name, name))

    def adjacent_loads(self, lo, hi):
        """lo and hi loaded from p and p + 8 with nothing written between: one 16-byte load."""
        a, b = lo.op, hi.op
        if a.kind != 'load' or b.kind != 'load' or a.block is not b.block or lo.ty.bits != 64:
            return None
        if any(u is not lo.op and u.kind != 'vop' for u in lo.uses) or len(lo.uses) != 1 or len(hi.uses) != 1:
            return None
        ma, mb = self.mem_operand(a.args[0], 8), self.mem_operand(b.args[0], 8)
        if (ma.base, ma.index, ma.scale, ma.sym, ma.slot) != (mb.base, mb.index, mb.scale, mb.sym, mb.slot) or mb.disp != ma.disp + 8:
            return None
        ops = a.block.ops
        i, j = sorted((ops.index(a), ops.index(b)))
        if any(x.writes_memory for x in ops[i:j]):
            return None
        return Mem(16, base=ma.base, index=ma.index, scale=ma.scale, disp=ma.disp, sym=ma.sym, slot=ma.slot)

    def splat(self, op, x):
        dst = self.define(op.result)
        bits = op.ty.bits
        if x.const is not None:
            self.vregs[op.result] = self.materialize(self.fn.const(op.ty, bytes([x.const & 0xff]) * op.ty.size))
            return
        if self.vex and x.op in self.skip and x.op.kind == 'load' and x not in self.vregs:
            self.emit('vpbroadcastb', Reg(dst.r, bits), self.mem_operand(x.op.args[0], 1), acc='w r')
            return
        if self.evex:
            self.emit('vpbroadcastb', Reg(dst.r, bits), self.reg(x, 32), acc='w r')
            return
        if self.vex:
            self.emit('vmovd', Reg(dst.r, 128), self.reg(x, 32), acc='w r')
            self.emit('vpbroadcastb', Reg(dst.r, bits), Reg(dst.r, 128), acc='w r')
            return
        if self.isa == 'sse42':
            # SSSE3: shuffle the byte everywhere by an all-zero index
            zero = self.materialize(self.fn.const(op.ty, bytes(16)))
            self.emit('movd', Reg(dst.r, 128), self.reg(x, 32), acc='w r')
            self.emit('pshufb', Reg(dst.r, 128), Reg(zero, 128), acc='rw r')
            return
        # SSE2: the byte in every lane by unpack and shuffles
        self.emit('movd', Reg(dst.r, 128), self.reg(x, 32), acc='w r')
        self.emit('punpcklbw', Reg(dst.r, 128), Reg(dst.r, 128), acc='rw r')
        self.emit('pshuflw', Reg(dst.r, 128), Reg(dst.r, 128), Imm(0, 8), acc='w r r')
        self.emit('pshufd', Reg(dst.r, 128), Reg(dst.r, 128), Imm(0, 8), acc='w r r')

    def mask_reg(self, v):
        """v (an integer) in a mask register: the compare's own, or one moved in (once per block)."""
        k = self.kview.get(v)
        if k is not None:
            return k
        k = self.kcache.get((v, self.block))
        if k is None:
            k = self.new('mask')
            self.emit('kmov' + KSUFFIX[v.ty.bits if v.ty.bits in (8, 16, 32, 64) else 32], Reg(k, v.ty.bits), self.reg(v, 64 if v.ty.bits > 32 else 32), acc='w r', kind='kmov')
            self.kcache[(v, self.block)] = k
        return k

    def mask_compare(self, op, name, a, b, mask):
        mnemonic, pred = MASK_CMP[name]
        write_mask = self.mask_reg(mask) if mask is not None else None
        k = self.new('mask')
        self.kview[op.result] = k
        zero = bytes(a.ty.size)
        if name in ('cmpeq_epi8_mask', 'cmpneq_epu8_mask') and (a.const == zero or b.const == zero):
            # against zero: the test of the value with itself, no constant
            x = b if a.const == zero else a
            r = self.reg(x)
            self.emit('vptestnmb' if name == 'cmpeq_epi8_mask' else 'vptestmb', Reg(k, op.ty.bits), r, r, acc='w r r', mask=write_mask)
        else:
            ops = [Reg(k, op.ty.bits), self.reg(a), self.vop_operand(b)]
            acc = 'w r r'
            if pred is not None:
                ops.append(Imm(pred, 8))
                acc += ' r'
            self.emit(mnemonic, *ops, acc=acc, mask=write_mask)
        # the integer view, for the uses that need it; dropped when none does
        gpr_uses = [u for u in op.result.uses if not (
            u.kind == 'icmp' and u.args[1].const == 0 and u.attrs['pred'] in ('eq', 'ne') and u in self.skip
            or u.kind == 'vop' and (u.attrs['name'].endswith('_masked') and u.args[0] is op.result
                                    or u.attrs['name'] in ('masked_load',) and u.args[0] is op.result
                                    or u.attrs['name'] == 'masked_store' and u.args[1] is op.result))]
        if gpr_uses:
            r = self.new('gpr')
            bits = op.ty.bits
            self.emit('kmov' + KSUFFIX[bits], Reg(r, 64 if bits > 32 else 32), Reg(k, bits), acc='w r', kind='kmov')
            if bits <= 32:
                self.clean32.add(r)
                self.zext_ok.add(r)
            self.vregs[op.result] = r


INVERT_CC = {'e': 'ne', 'ne': 'e', 'b': 'ae', 'ae': 'b', 'a': 'be', 'be': 'a', 'l': 'ge', 'ge': 'l', 'g': 'le', 'le': 'g'}


def select(fn, isa):
    return ISel(fn, isa).run()
