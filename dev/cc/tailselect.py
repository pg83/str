"""Instruction selection for the AVX2 strcmp tail's scan recipe."""
from . import machine as M
from .select import Select
from .stringselect import resolve_block


class TailSelect(Select):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for block in self.source.blocks:
            for op in block.ops:
                if op.attrs.get('scalar_andnot'):
                    self.folded.add(self.definition(op.args[0]))

    def compare(self, op, invert=False):
        if op.attrs.get('page_compare'):
            self.ins('cmp', [M.R(self.operand(op.args[0], True).reg, 32), self.operand(op.args[1])],
                     ['R', 'R'], keep_immediate=True)
            return 'be' if invert else 'a'
        return super().compare(op, invert)

    def operation(self, op):
        if op.attrs.get('materialize_index') or op.attrs.get('scalar_lea'):
            a, b = (self.operand(v, True) for v in op.args)
            self.ins('lea', [self.value(op.results[0]), M.Mem(8, base=a.reg, index=b.reg)], ['W', 'A'])
            return
        if op.attrs.get('scalar_andnot'):
            inverse = self.definition(op.args[0])
            a, b = self.operand(inverse.args[0], True), self.operand(op.args[1], True)
            self.ins('andn', [self.value(op.results[0]), a, b], ['W', 'R', 'R'])
            return
        return super().operation(op)

    def run(self):
        fn, pool = super().run()
        fn.ssa_blocks = self.blocks.copy()
        counts = {self.value(v).reg for v in self.source.avx_tail_count_breaks}
        for block in fn.blocks:
            for ins in list(block.ins):
                if ins.mnemonic == 'tzcnt' and ins.operands[0].reg in counts:
                    dest = M.R(ins.operands[0].reg, 32)
                    block.ins.insert(block.ins.index(ins), M.Instr('xor', (dest, dest), ('W', 'R')))
        head, result = self.source.avx_tail_root
        compare = next(ins for ins in self.blocks[head].ins
                       if ins.mnemonic == 'vpcmpeqb' and self.value(result).reg in ins.defs)
        zero = compare.operands[2].reg
        owner, instruction = next((b, ins) for b in fn.blocks for ins in b.ins
                                  if ins.mnemonic == 'vpxor' and zero in ins.defs)
        fn.preferred[zero] = M.physical('xmm0')
        owner.ins.remove(instruction)
        fn.entry.ins.insert(0, instruction)

        # Preserve the all-ones comparison on the continuing path and only
        # complement the word on the edge that needs its trailing-zero count.
        source, value, returned = self.source.avx_tail_inversion
        block = self.blocks[source]
        inverse = self.value(value).reg
        instruction = next(ins for ins in block.ins if ins.mnemonic == 'not' and inverse in ins.defs)
        block.ins.remove(instruction)
        test = next(ins for ins in block.ins if ins.mnemonic == 'test')
        test.mnemonic = 'cmp'
        test.operands = test.operands[0], M.Imm(-1, 32)
        self.blocks[returned].ins.insert(0, instruction)
        return fn, pool


def layout_blocks(fn):
    if not getattr(fn.source, 'avx_cmp_tail', False):
        return
    ordered = []
    for source in fn.source.avx_tail_order:
        block = resolve_block(fn, source)
        if block is not None and block not in ordered:
            ordered.append(block)
        ordered.extend(b for b in fn.blocks if getattr(b, 'source_parent', None) is source and b not in ordered)
    fn.block_order = ordered
