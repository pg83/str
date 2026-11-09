"""Instruction selection and layout for AVX2 set-scan recipes."""
from collections import Counter
from dataclasses import replace

from . import machine as M
from . import types as T
from .prefixes import string_bare
from .select import Select
from .stringselect import resolve_block


class SpanSelect(Select):
    """Instruction choices for the AVX2 span recipe."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_truth = {}
        for block in self.source.blocks:
            if block.term.kind != 'br':
                continue
            value = block.term.args[0]
            current, casts = value, []
            while self.source.defs[current].kind == 'cast':
                cast = self.source.defs[current]
                if (self.uses[current] != 1 or cast.args[0].type.kind != 'int'
                        or current.type.kind != 'int' or current.type.bits < cast.args[0].type.bits):
                    break
                casts.append(cast)
                current = cast.args[0]
            load = self.source.defs[current]
            if (casts and load.kind == 'load' and load.attrs['width'] <= 2 and load in block.ops
                    and not load.attrs.get('volatile') and self.uses[current] == 1):
                self.folded.update([*casts, load])
                self.memory_truth[value] = load
            compare = self.source.defs[value]
            if (block.term.targets[0].term.kind == 'ret' and compare.kind in ('lt', 'le', 'gt', 'ge')
                    and any(self.source.defs[string_bare(self.source, v)].kind == 'intrinsic'
                            and self.source.defs[string_bare(self.source, v)].attrs['builtin'].op == 'cmpistri'
                            for v in compare.args)):
                self.valid_return_tests.add(self.blocks[block])

    def bit_count(self, value):
        op = self.source.defs[string_bare(self.source, value)]
        return super().bit_count(value) or op.kind == 'intrinsic' and op.attrs['builtin'].op == 'cmpistri'

    def zero_extension(self, op):
        return False if op.attrs.get('force_zero_extension') else super().zero_extension(op)

    def move(self, dest, source, preserve=False):
        if all(isinstance(o, M.R) and o.reg.bank == 'vec' for o in (dest, source)):
            return self.ins('vmovaps', [dest, source], ['W', 'R'], preserve=preserve)
        return super().move(dest, source, preserve)

    def operation(self, op):
        if op.attrs.get('force_zero_extension'):
            self.ins('mov', [M.R(self.value(op.results[0]).reg, 32),
                             M.R(self.operand(op.args[0], True).reg, 32)], ['W', 'R'], preserve=True)
            return
        if op.attrs.get('parameter_copy'):
            self.move(self.value(op.results[0]), self.operand(op.args[0], True))
            return
        return super().operation(op)

    def compare(self, op, invert=False):
        if op.attrs.get('narrow_compare'):
            self.ins('cmp', [M.R(self.operand(op.args[0], True).reg, 32), self.operand(op.args[1])], ['R', 'R'])
            return 'e' if invert else 'ne'
        if op.attrs.get('test_width'):
            value = M.R(self.operand(op.args[0], True).reg, op.attrs['test_width'])
            self.ins('test', [value, value], ['R', 'R'])
            return 'e' if invert else 'ne'
        return super().compare(op, invert)

    def condition(self, value):
        if value in self.memory_truth:
            op = self.memory_truth[value]
            self.ins('cmp', [self.address(op.args[0], op.attrs['width']), M.Imm(0, 8)], ['R', 'R'])
            return 'ne'
        op = self.source.defs[value]
        if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'bzhi':
            return 'ne'
        return super().condition(value)

    def intrinsic(self, op):
        name = op.attrs['builtin'].op
        if name == 'cmpistri':
            self.fn.preferred[self.value(op.results[0]).reg] = M.physical('rcx')
        if name == 'cmpeq_epi8':
            self.binary_args[op] = op.args
        if name == 'add_epi8' and self.source.defs[op.args[0]].kind == 'constant':
            self.binary_args[op] = op.args[::-1]
        if name == 'or':
            self.binary_args[op] = op.args[::-1]
            for value, mask in (op.args, op.args[::-1]):
                definition = self.definition(mask)
                if (definition.kind == 'intrinsic' and definition.attrs['builtin'].op == 'cmpgt_epi8'
                        and definition.args[0] == value):
                    memory = M.Mem(16, symbol=self.constant_name(bytes([0x8f]) * 16))
                    self.ins('vpblendvb', [self.value(op.results[0]), self.operand(value, True),
                                          memory, self.operand(mask, True)], ['W', 'R', 'R', 'R'])
                    return
        return super().intrinsic(op)

    def run(self):
        fn, pool = super().run()
        fn.ssa_blocks = self.blocks.copy()
        known = {self.value(op.results[0]).reg for b in self.source.blocks for op in b.ops
                 if op.kind == 'intrinsic' and op.results
                 and op.attrs['builtin'].op in ('cmpistri', 'ctz_nonzero', 'tzcnt')}
        for block in fn.blocks:
            for ins in block.ins:
                if (ins.mnemonic == 'mov' and len(ins.operands) == 2
                        and all(isinstance(o, M.R) and o.bits == 64 and o.reg.bank == 'gpr' for o in ins.operands)
                        and ins.operands[1].reg in known):
                    ins.operands = tuple(M.R(o.reg, 32) for o in ins.operands)
                if ins.mnemonic == 'parallel':
                    ins.copies = tuple((M.R(d.reg, 32), M.R(s.reg, 32))
                                       if all(isinstance(o, M.R) and o.reg.bank == 'gpr' for o in (d, s))
                                       and s.reg in known else (d, s) for d, s in ins.copies)
        # AVX broadcasts need no zero register. Recreate comparison zeroes
        # in their consuming block instead of keeping a loop-wide live value.
        counts = Counter(r for b in fn.blocks for i in b.ins for r in i.defs)
        for owner in fn.blocks:
            for zero in list(owner.ins):
                if (zero.mnemonic not in ('vpxor', 'vpxord')
                        or len({o.reg for o in zero.operands if isinstance(o, M.R)}) != 1):
                    continue
                old = zero.operands[0].reg
                if counts[old] != 1:
                    continue
                owner.ins.remove(zero)
                for block in fn.blocks:
                    users = [i for i in block.ins if old in i.uses]
                    if not users:
                        continue
                    reg = fn.reg(T.V128)
                    for ins in users:
                        ins.map_regs({old: reg.reg})
                    block.ins.insert(block.ins.index(users[0]), M.Instr('vpxor', (reg, reg, reg), ('W', 'R', 'R')))
        newset = self.value(self.source.avx_parameters[self.source.params[1]]).reg
        for block in fn.blocks:
            for ins in list(block.ins):
                if (ins.mnemonic == 'lea' and isinstance(ins.operands[1], M.Mem)
                        and ins.operands[1].base == newset and ins.operands[1].disp == 16
                        and len(block.successors) == 2):
                    reg = ins.operands[0].reg
                    targets = [s for s in block.successors if any(reg in i.uses for i in s.ins)]
                    if (len(targets) == 1
                            and not any(reg in i.uses for i in block.ins[block.ins.index(ins) + 1:])):
                        block.ins.remove(ins)
                        targets[0].ins.insert(0, ins)
        retained = {self.value(v).reg for v in self.source.word_return_bases}
        for block in fn.blocks:
            for ins in block.ins:
                ins.operands = tuple(replace(o, base=o.index, index=o.base)
                                     if isinstance(o, M.Mem) and o.index in retained and o.scale == 1 else o
                                     for o in ins.operands)
        dependency_breaks = {self.value(v).reg for v in self.source.avx_break_counts}
        for block in fn.blocks:
            for ins in list(block.ins):
                if ins.mnemonic == 'tzcnt' and ins.operands[0].reg in dependency_breaks:
                    dest = M.R(ins.operands[0].reg, 32)
                    block.ins.insert(block.ins.index(ins), M.Instr('xor', (dest, dest), ('W', 'R')))
        self.source.shared_index_return = True
        return fn, pool


def parallel_copies(fn):
    if not getattr(fn.source, 'avx_span', False):
        return
    for block in fn.blocks:
        for ins in block.ins:
            if (ins.mnemonic in ('movups', 'movaps', 'movdqu', 'movdqa',
                                 'vmovups', 'vmovaps', 'vmovdqu', 'vmovdqa')
                    and all(isinstance(o, M.R) and o.reg.bank == 'vec' for o in ins.operands)):
                ins.mnemonic = 'vmovaps'


def share_epilogue(fn):
    if not getattr(fn.source, 'avx_span', False):
        return False
    first, second, shared, edges = fn.source.avx_returns
    fast, loop = resolve_block(fn, second), resolve_block(fn, shared)
    if fast is None or loop is None or fast.ins[-1].mnemonic != 'ret':
        raise ValueError('span recipe lost its common return')
    if not all(ins.mnemonic == 'mov'
               and all(isinstance(o, M.R) and o.reg == M.physical('rax') for o in ins.operands)
               for ins in fast.ins[:-1]):
        raise ValueError('span head result must already be in EAX')
    fast.ins = fast.ins[-1:]
    loop.ins[-1] = M.Instr('jmp', target=fast)
    loop.successors = [fast]
    fn.retained_returns = {fast}
    return True


def layout_blocks(fn):
    if not getattr(fn.source, 'avx_span', False):
        return
    ordered = []
    for source in fn.source.avx_block_order:
        block = resolve_block(fn, source)
        if block is not None and block not in ordered:
            ordered.append(block)
        ordered.extend(b for b in fn.blocks if getattr(b, 'source_parent', None) is source and b not in ordered)
    fn.block_order = ordered
