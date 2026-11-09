"""Selection and scheduling for bounded AVX2 string copies."""
from . import machine as M
from . import types as T
from .select import Select
from .stringselect import resolve_block


class CopySelect(Select):
    def intrinsic(self, op):
        if op.attrs['builtin'].op == 'load':
            value = op.args[0]
            while self.source.defs[value].kind == 'cast':
                value = self.source.defs[value].args[0]
            if value in self.source.avx_copy_bases:
                retained = self.source.word_return_bases
                self.source.word_return_bases = retained - {value}
                try:
                    return super().intrinsic(op)
                finally:
                    self.source.word_return_bases = retained
        return super().intrinsic(op)

    def operation(self, op):
        if op.attrs.get('copy_count'):
            a, b = (self.operand(v, True) for v in op.args)
            self.ins('lea', [self.value(op.results[0]), M.Mem(8, base=a.reg, index=b.reg)], ['W', 'A'])
            return
        return super().operation(op)

    def localize_vex_zero_compares(self):
        zeros = [(b, ins) for b in self.fn.blocks for ins in b.ins if ins.mnemonic == 'vpxor'
                 and len({o.reg for o in ins.operands if isinstance(o, M.R)}) == 1]
        old = {ins.operands[0].reg for _, ins in zeros}
        for block, ins in zeros:
            block.ins.remove(ins)
        for values, color in self.source.avx_copy_zero_groups:
            zero = self.fn.reg(T.V256)
            self.fn.preferred[zero.reg] = M.physical('xmm%d' % color)
            for n, value in enumerate(values):
                block = self.owners[value]
                compare = next(ins for ins in block.ins
                               if ins.mnemonic == 'vpcmpeqb' and self.value(value).reg in ins.defs)
                source = next(o for o in compare.operands[1:] if not isinstance(o, M.R) or o.reg not in old)
                compare.operands = ((compare.operands[0], zero, source) if isinstance(source, M.Mem)
                                    else (compare.operands[0], source, zero))
                if n == 0:
                    narrow = M.R(zero.reg, 128)
                    block.ins.insert(block.ins.index(compare),
                                     M.Instr('vpxor', (narrow, narrow, narrow), ('W', 'R', 'R')))

    def run(self):
        fn, pool = super().run()
        fn.ssa_blocks = self.blocks.copy()
        counts = {self.value(v).reg for v in self.source.avx_copy_breaks}
        for block in fn.blocks:
            for ins in list(block.ins):
                if ins.mnemonic == 'tzcnt' and ins.operands[0].reg in counts:
                    dest = M.R(ins.operands[0].reg, 32)
                    block.ins.insert(block.ins.index(ins), M.Instr('xor', (dest, dest), ('W', 'R')))
        for pointer, data in self.source.avx_copy_address_loads:
            cursor, loaded = self.value(pointer).reg, self.value(data).reg
            for block in fn.blocks:
                address = next((ins for ins in block.ins if ins.mnemonic == 'lea' and cursor in ins.defs), None)
                load = next((ins for ins in block.ins if ins.mnemonic == 'vmovdqa' and loaded in ins.defs), None)
                if address is not None and load is not None and block.ins.index(address) > block.ins.index(load):
                    block.ins.remove(address)
                    block.ins.insert(block.ins.index(load), address)
        tail = self.blocks[self.source.avx_copy_tail]
        count = next(ins for ins in tail.ins if ins.mnemonic == 'tzcnt')
        difference = next(ins for ins in tail.ins if ins.mnemonic == 'sub')
        if tail.ins.index(count) > tail.ins.index(difference):
            tail.ins.remove(count)
            tail.ins.insert(tail.ins.index(difference), count)
        return fn, pool


def layout_blocks(fn):
    if not getattr(fn.source, 'avx_copy_short', False):
        return
    ordered = []
    for source in fn.source.avx_copy_order:
        block = resolve_block(fn, source)
        if block is not None and block not in ordered:
            ordered.append(block)
        ordered.extend(b for b in fn.blocks if getattr(b, 'source_parent', None) is source and b not in ordered)
    fn.block_order = ordered
