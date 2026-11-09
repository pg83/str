"""Selection and return layout for the AVX2 strcmp entry recipe."""
from . import machine as M
from . import types as T
from .select import Select
from .stringselect import resolve_block


class CmpSelect(Select):
    def compare(self, op, invert=False):
        if self.definition(op.args[0]).results == (self.source.avx_cmp_page,):
            lhs = M.R(self.operand(op.args[0], True).reg, 32)
            self.ins('cmp', [lhs, self.operand(op.args[1])], ['R', 'R'], keep_immediate=True)
            return 'a' if invert else 'be'
        return super().compare(op, invert)

    def operand(self, value, register=False):
        op = self.source.defs[value]
        if op.kind == 'constant' and op.attrs.get('value') == b'\xff' * 16:
            result = self.fn.reg(T.V128)
            self.fn.preferred[result.reg] = M.physical('xmm2')
            self.ins('vpcmpeqd', [result, result, result], ['W', 'R', 'R'])
            return result
        return super().operand(value, register)

    def localize_vex_zero_compares(self):
        zeros = [(b, ins) for b in self.fn.blocks for ins in b.ins
                 if ins.mnemonic == 'vpxor'
                 and len({o.reg for o in ins.operands if isinstance(o, M.R)}) == 1]
        old = {ins.operands[0].reg for _, ins in zeros}
        for block, ins in zeros:
            block.ins.remove(ins)
        for values, color in self.source.avx_cmp_zero_groups:
            zero = self.fn.reg(values[0].type)
            self.fn.preferred[zero.reg] = M.physical('xmm%d' % color)
            for n, value in enumerate(values):
                block = self.owners[value]
                compare = next(ins for ins in block.ins
                               if ins.mnemonic == 'vpcmpeqb' and self.value(value).reg in ins.defs)
                source = next(o for o in compare.operands[1:] if o.reg not in old)
                compare.operands = compare.operands[0], source, zero
                if n == 0:
                    at = block.ins.index(compare)
                    if value.type.bits == 256:
                        at = next(k for k, ins in enumerate(block.ins) if ins.mnemonic == 'vpcmpeqb')
                    narrow = M.R(zero.reg, 128)
                    block.ins.insert(at, M.Instr('vpxor', (narrow, narrow, narrow), ('W', 'R', 'R')))

    def run(self):
        fn, pool = super().run()
        fn.ssa_blocks = self.blocks.copy()
        counts = {self.value(v).reg for v in self.source.avx_cmp_break_counts}
        for block in fn.blocks:
            for ins in list(block.ins):
                if ins.mnemonic == 'tzcnt' and ins.operands[0].reg in counts:
                    dest = M.R(ins.operands[0].reg, 32)
                    block.ins.insert(block.ins.index(ins), M.Instr('xor', (dest, dest), ('W', 'R')))
        return fn, pool


def share_epilogue(fn):
    if not getattr(fn.source, 'avx_cmp', False):
        return False
    first, second = (resolve_block(fn, b) for b in fn.source.avx_cmp_shared_returns)

    def signature(ins):
        return ins.mnemonic, ins.operands, ins.access, ins.implicit_uses, ins.implicit_defs

    if [signature(ins) for ins in first.ins] != [signature(ins) for ins in second.ins]:
        raise ValueError('strcmp recipe expected identical byte-difference returns')
    for block in fn.blocks:
        block.successors = [first if b is second else b for b in block.successors]
        for ins in block.ins:
            if ins.target is second:
                ins.target = first
    fn.blocks.remove(second)
    return True


def upper_state(fn):
    if not getattr(fn.source, 'avx_cmp', False):
        return
    # Scalar-only callers enter the tail without changing YMM state. The
    # two wide paths have their own short tail calls and explicit boundary.
    for source in fn.source.avx_cmp_dirty_tails:
        block = resolve_block(fn, source)
        if block is None or not block.ins:
            raise ValueError('strcmp recipe lost a wide tail edge')
        block.ins.insert(len(block.ins) - 1, M.Instr('vzeroupper'))


def layout_blocks(fn):
    if not getattr(fn.source, 'avx_cmp', False):
        return
    ordered = []
    for source in fn.source.avx_cmp_order:
        block = resolve_block(fn, source)
        if block is not None and block not in ordered:
            ordered.append(block)
        ordered.extend(b for b in fn.blocks if getattr(b, 'source_parent', None) is source and b not in ordered)
    fn.block_order = ordered
