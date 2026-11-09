"""Instruction scheduling and shared vector exits for long AVX2 copies."""
from . import machine as M
from . import types as T
from .select import Select
from .stringselect import resolve_block


class LongCopySelect(Select):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for b in self.source.blocks:
            for op in b.ops:
                if op.attrs.get('cold_offset'):
                    self.folded.add(self.definition(op.args[0]))

    def operation(self, op):
        if op.attrs.get('cold_offset'):
            previous = self.definition(op.args[0])
            a, c = (self.operand(v, True) for v in previous.args)
            self.ins('lea', [self.value(op.results[0]), M.Mem(8, base=a.reg, index=c.reg, disp=op.attrs['cold_offset'])], ['W', 'A'])
            return
        if op.attrs.get('counter_initial'):
            self.move(M.R(self.value(op.results[0]).reg, 32), M.Imm(32, 32))
            return
        if op.attrs.get('copy_count'):
            a, c = (self.operand(v, True) for v in op.args)
            self.ins('lea', [self.value(op.results[0]), M.Mem(8, base=a.reg, index=c.reg)], ['W', 'A'])
            return
        if op.attrs.get('direct_address'):
            a = self.operand(op.args[0], True)
            d = self.definition(op.args[1])
            c = self.operand(op.args[1], True) if d.kind != 'constant' else None
            address = M.Mem(8, base=a.reg, index=c.reg) if c else M.Mem(8, base=a.reg, disp=d.attrs['value'])
            self.ins('lea', [self.value(op.results[0]), address], ['W', 'A'])
            return
        return super().operation(op)

    def localize_vex_zero_compares(self):
        zeros = [(b, i) for b in self.fn.blocks for i in b.ins if i.mnemonic == 'vpxor' and len({o.reg for o in i.operands if isinstance(o, M.R)}) == 1]
        old = {i.operands[0].reg for b, i in zeros}
        for b, i in zeros:
            b.ins.remove(i)
        for values, color, before in self.source.avx_long_zero_groups:
            zero = self.fn.reg(values[0].type)
            self.fn.preferred[zero.reg] = M.physical('xmm' + str(color))
            for n, value in enumerate(values):
                block = self.owners[value]
                cmp = next((i for i in block.ins if i.mnemonic == 'vpcmpeqb' and self.value(value).reg in i.defs))
                source = next((o for o in cmp.operands[1:] if not isinstance(o, M.R) or o.reg not in old))
                cmp.operands = (cmp.operands[0], zero, source) if isinstance(source, M.Mem) else (cmp.operands[0], source, zero)
                if not n:
                    target = self.blocks[before] if before is not None else block
                    at = next((k for k, i in enumerate(target.ins) if i.target is not None), len(target.ins)) if before else target.ins.index(cmp)
                    narrow = M.R(zero.reg, 128)
                    target.ins.insert(at, M.Instr('vpxor', (narrow, narrow, narrow), ('W', 'R', 'R')))

    def run(self):
        fn, pool = super().run()
        fn.ssa_blocks = self.blocks.copy()
        breaks = {self.value(v).reg for v in self.source.avx_long_breaks}
        for block in fn.blocks:
            for ins in list(block.ins):
                if ins.mnemonic == 'tzcnt' and ins.operands[0].reg in breaks:
                    d = M.R(ins.operands[0].reg, 32)
                    block.ins.insert(block.ins.index(ins), M.Instr('xor', (d, d), ('W', 'R')))
        setup, head, later, pointer, count, cursor = self.source.avx_long_lookahead
        setup, head, later = (self.blocks[q] for q in (setup, head, later))
        progress = self.value(self.source.avx_long_progress)
        for ins in setup.ins:
            if ins.mnemonic == 'parallel':
                selected = [(d, s) for d, s in ins.copies if d.reg == progress.reg and isinstance(s, M.Imm) and (s.value == 0)]
                if selected:
                    ins.copies = tuple((pair for pair in ins.copies if pair not in selected))
                    d = M.R(progress.reg, 32)
                    setup.ins.insert(0, M.Instr('xor', (d, d), ('W', 'R')))
                    break
        lea = next((i for i in head.ins if i.mnemonic == 'lea' and self.value(pointer).reg in i.defs))
        update = next((i for i in head.ins if self.value(count).reg in i.defs))
        if head.ins.index(lea) > head.ins.index(update):
            head.ins.remove(lea)
            head.ins.insert(head.ins.index(update), lea)
        for block in fn.blocks:
            for ins in block.ins:
                if ins.mnemonic != 'parallel':
                    continue
                pairs = [(d, s) for d, s in ins.copies if d.reg == self.value(cursor).reg and isinstance(s, M.R) and (s.reg == self.value(pointer).reg)]
                if not pairs:
                    continue
                ins.copies = tuple((pair for pair in ins.copies if pair not in pairs))
                d, s = pairs[0]
                test = next((i for i in later.ins if i.mnemonic == 'test'))
                later.ins.insert(later.ins.index(test), M.Instr('mov', (d, s), ('W', 'R')))
        branch, jump = later.ins[-2:]
        from cc.select import INVERSE
        branch.target, jump.target = (jump.target, branch.target)
        branch.cc = INVERSE[branch.cc]
        branch.mnemonic = 'j' + branch.cc
        kernel, edges, first, second = self.source.avx_long_kernel
        first, second = (self.blocks[first], self.blocks[second])
        m1, z2, m2 = (self.value(v).reg for v in self.source.avx_long_masks)
        stores = [i for i in first.ins if i.mnemonic in ('vmovdqu', 'vmovups') and isinstance(i.operands[0], M.Mem)]
        mask = next((i for i in first.ins if i.mnemonic == 'vpmovmskb' and m1 in i.defs))
        first.ins.remove(mask)
        first.ins.insert(first.ins.index(stores[-1]), mask)
        stores = [i for i in second.ins if i.mnemonic in ('vmovdqu', 'vmovups') and isinstance(i.operands[0], M.Mem)]
        pair = [next((i for i in second.ins if i.mnemonic == name and reg in i.defs)) for name, reg in (('vpcmpeqb', z2), ('vpmovmskb', m2))]
        for ins in pair:
            second.ins.remove(ins)
        at = second.ins.index(stores[-1]) + 1
        second.ins[at:at] = pair
        return (fn, pool)

def share_epilogue(fn):
    if not getattr(fn.source, 'avx_copy_long', False):
        return False
    first = resolve_block(fn, fn.source.avx_long_kernel[1][0])
    last = resolve_block(fn, fn.source.avx_long_kernel[1][2])

    def redundant(i):
        return i.mnemonic.startswith(('mov', 'vmov')) and len(i.operands) == 2 and all((isinstance(o, M.R) for o in i.operands)) and (i.operands[0] == i.operands[1]) and (not i.preserve)
    first.ins = [i for i in first.ins if not redundant(i)]
    last.ins = [i for i in last.ins if not redundant(i)]

    def signature(i):
        return (i.mnemonic, i.operands, i.access, i.target)
    other = last.ins[-1].target
    if other is not None:
        other.ins = [i for i in other.ins if not redundant(i)]
        if [signature(i) for i in other.ins] == [signature(i) for i in first.ins]:
            last.ins[-1].target = first
            last.successors = [first]
            if not any((other in b.successors for b in fn.blocks)):
                fn.blocks.remove(other)
    if [signature(i) for i in last.ins[-len(first.ins):]] == [signature(i) for i in first.ins]:
        last.ins[-len(first.ins):] = [M.Instr('jmp', target=first)]
        last.successors = [first]
    return True

def parallel_copies(fn):
    if not getattr(fn.source, 'avx_copy_long', False):
        return
    for block in fn.blocks:
        scalar = next((i for i in block.ins if i.mnemonic == 'xor' and all((isinstance(o, M.R) and o.reg == M.physical('r8') for o in i.operands))), None)
        zero = next((i for i in block.ins if i.mnemonic == 'vpxor'), None)
        if scalar is not None and zero is not None and (block.ins.index(scalar) > block.ins.index(zero)):
            block.ins.remove(scalar)
            block.ins.insert(block.ins.index(zero), scalar)
    scalar = next(((b, i) for b in fn.blocks for i in b.ins if i.mnemonic == 'xor' and all((isinstance(o, M.R) and o.reg == M.physical('r8') for o in i.operands))), None)
    setup = fn.ssa_blocks[fn.source.avx_long_lookahead[0]]
    zero = next((i for i in setup.ins if i.mnemonic == 'vpxor'), None)
    if scalar is not None and zero is not None and (scalar[0] is not setup):
        setup.ins.remove(zero)
        scalar[0].ins.insert(scalar[0].ins.index(scalar[1]) + 1, zero)
    final = fn.ssa_blocks[fn.source.avx_long_final]
    address = next((i for i in final.ins if i.mnemonic == 'lea' and i.operands[0].reg == M.physical('rdi')), None)
    store = next((i for i in final.ins if i.mnemonic in ('vmovdqu', 'vmovups') and isinstance(i.operands[0], M.Mem)), None)
    if address is not None and store is not None and (final.ins.index(address) < final.ins.index(store)):
        final.ins.remove(address)
        final.ins.insert(final.ins.index(store) + 1, address)

def layout_blocks(fn):
    if getattr(fn.source, 'avx_copy_long', False):
        order = []
        for source in fn.source.avx_long_order:
            block = resolve_block(fn, source)
            if block is not None and block not in order:
                order.append(block)
            order.extend((q for q in fn.blocks if getattr(q, 'source_parent', None) is source and q not in order))
        fn.block_order = order
    return

def upper_state(fn):
    if getattr(fn.source, 'avx_copy_long', False):
        block = resolve_block(fn, fn.source.avx_long_clean_return)
        block.ins = [i for i in block.ins if i.mnemonic != 'vzeroupper']
