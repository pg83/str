"""Instruction scheduling and shared suffixes for AVX2 strcspn."""
from dataclasses import replace

from . import machine as M
from .select import Select, INVERSE
from .spanselect import SpanSelect
from .stringselect import resolve_block


class CspnSelect(SpanSelect):
    def intrinsic(self, op):
        if (op.attrs['builtin'].op == 'or'
                and all(self.definition(v).kind == 'intrinsic'
                        and self.definition(v).attrs['builtin'].op == 'cmpistrm' for v in op.args)):
            self.binary_args[op] = op.args
            return Select.intrinsic(self, op)
        return super().intrinsic(op)

    def compare(self, op, invert=False):
        if op.attrs.get('cspn_index_compare'):
            self.ins('cmp', [M.R(self.operand(v, True).reg, 32) for v in op.args[::-1]], ['R', 'R'])
            return 'be' if invert else 'a'
        return super().compare(op, invert)

    def run(self):
        fn, pool = super().run()
        for source in self.source.avx_cspn_heads:
            for ins in self.blocks[source].ins:
                if (ins.mnemonic == 'mov'
                        and all(isinstance(o, M.R) and o.reg.bank == 'gpr' for o in ins.operands)):
                    ins.operands = tuple(M.R(o.reg, 32) for o in ins.operands)
        first_zero = self.value(self.source.avx_head1_zero_result).reg
        cursors = {self.value(v).reg for v in self.source.avx_pair_cursors}
        for block in fn.blocks:
            for ins in block.ins:
                if ins.mnemonic == 'vpcmpeqb' and isinstance(ins.operands[2], M.R):
                    zero = ins.operands[2].reg
                    physical = (M.physical('xmm2') if ins.operands[0].reg == first_zero
                                else fn.preferred.get(ins.operands[0].reg))
                    if physical is not None:
                        fn.preferred.setdefault(zero, physical)
            # The returned cursor is needed after both comparisons. Keep
            # that independent LEA between PCMPISTRM and the combined mask.
            for ins in list(block.ins):
                if (ins.mnemonic != 'lea' or not isinstance(ins.operands[1], M.Mem)
                        or ins.operands[0].reg not in cursors):
                    continue
                compares = [i for i in block.ins if i.mnemonic == 'vpcmpistrm']
                if len(compares) == 2 and block.ins.index(ins) < block.ins.index(compares[-1]):
                    block.ins.remove(ins)
                    block.ins.insert(block.ins.index(compares[-1]) + 1, ins)
        for source in self.source.avx_invert_pairs:
            branch, jump = self.blocks[source].ins[-2:]
            branch.target, jump.target = jump.target, branch.target
            branch.cc = INVERSE[branch.cc]
            branch.mnemonic = 'j' + branch.cc
        return fn, pool


def share_epilogue(fn):
    if not getattr(fn.source, 'avx_cspn', False):
        return False
    source, joined, sentinel = (resolve_block(fn, b) for b in fn.source.avx_loop_returns)
    # Keep an independent copy after the single-table sentinel, so the
    # non-NUL match can enter it directly after the constant assignment.
    duplicate = fn.block('span.cspn.single.other.return')
    duplicate.ins = [replace(ins) for ins in joined.ins]
    duplicate.source_parent = fn.source.avx_loop_returns[2]
    for block in (source, sentinel):
        block.successors = [duplicate if b is joined else b for b in block.successors]
        for ins in block.ins:
            if ins.target is joined:
                ins.target = duplicate
    first, second = (resolve_block(fn, b) for b in fn.source.avx_cspn_heads)
    for block in fn.blocks:
        block.successors = [first if b is second else b for b in block.successors]
        for ins in block.ins:
            if ins.target is second:
                ins.target = first
    fn.blocks.remove(second)
    for source in fn.source.avx_sentinels:
        edge = resolve_block(fn, source)
        joined = edge.successors[0]
        edge.ins[-1:] = [replace(ins) for ins in joined.ins]
        edge.successors = joined.successors[:]
    return True
