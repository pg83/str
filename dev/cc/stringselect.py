"""Instruction selection and scheduling for the concrete SSE4.2 string recipe."""
from collections import Counter
from dataclasses import replace

from . import machine as M
from . import types as T
from .facts import possible_bits
from .prefixes import string_bare
from .regalloc import liveness
from .select import Select, INVERSE

def local_zeros(fn):
    if fn.isa != 'sse42':
        return
    definitions = Counter((r for b in fn.blocks for i in b.ins for r in i.defs))
    constants = {}
    for owner in fn.blocks:
        for zero in list(owner.ins):
            if zero.mnemonic != 'pxor' or len(zero.operands) != 2 or zero.operands[0] != zero.operands[1] or (not isinstance(zero.operands[0], M.R)) or (definitions[zero.operands[0].reg] != 1):
                continue
            old = zero.operands[0].reg
            owner.ins.remove(zero)
            grouped = set()
            groups = []
            for root, region, color in getattr(fn, 'string_regions', []):
                if any((old in i.uses for b in region for i in b.ins)):
                    groups.append((root, region, color))
                    grouped.update(region)
            groups.extend(((b, {b}, None) for b in fn.blocks if b not in grouped and any((old in i.uses for i in b.ins))))
            for root, region, color in groups:
                reg = fn.reg(T.V128)
                if color is not None:
                    fn.preferred[reg.reg] = M.physical('xmm' + str(color))
                for block in region:
                    users = [i for i in block.ins if old in i.uses]
                    for i in users:
                        i.map_regs({old: reg.reg})
                    constants[block] = reg
                first = next((n for n, i in enumerate(root.ins) if reg.reg in i.uses or i.target is not None), len(root.ins))
                root.ins.insert(first, M.Instr('pxor', (reg, reg), ('W', 'R')))
    _, out = liveness(fn)
    for block in fn.blocks:
        live = set(out[block])
        after = {}
        for i in reversed(block.ins):
            after[i] = set(live)
            live = live - i.defs | i.uses
        for zero, compare in list(zip(block.ins, block.ins[1:])):
            if zero.mnemonic != 'pxor' or zero.operands[0] != zero.operands[1] or compare.mnemonic != 'pcmpeqb' or (compare.operands[0] != zero.operands[0]):
                continue
            result, data = compare.operands
            local = constants.get(block)
            fn.preferred.setdefault(result.reg, M.physical('xmm0'))
            if local is None and (not isinstance(data, M.R) or data.reg in after[compare]):
                continue
            if local is None:
                local = fn.reg(T.V128)
                zero.operands = (local, local)
                block.ins.insert(block.ins.index(compare), M.Instr('movaps', (result, data), ('W', 'R')))
            else:
                zero.mnemonic = 'movaps' if isinstance(data, M.R) else 'movdqa'
                zero.operands = (result, data)
                zero.access = ('W', 'R')
            compare.operands = (result, local)

class StringSelect(Select):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_truth = {}
        for block in self.source.blocks:
            positions = {op: n for n, op in enumerate(block.ops)}
            for load in block.ops:
                if load.kind != 'intrinsic' or load.attrs['builtin'].op != 'load' or load.results[0].type != T.V128:
                    continue
                value = load.results[0]
                users = self.users.get(value, [])
                if not users or self.uses[value] != len(users):
                    continue
                if any((user not in positions or user.kind != 'intrinsic' or (not user.attrs['builtin'].op.startswith('cmpistr')) or (user.args[1] != value) or (user.args[0] == value) for user in users)):
                    continue
                builtin = load.attrs['builtin']
                if self.fn.isa in ('sse2', 'sse42') and max(builtin.alignment, self.alignments.get(load.args[0], 1)) < 16:
                    continue
                last = max((positions[u] for u in users))
                if any((op.kind in ('call', 'asm') or 'write' in op.attrs.get('memory', '') or op.attrs.get('volatile') for op in block.ops[positions[load] + 1:last])):
                    continue
                self.folded.add(load)
                self.folded_vectors[value] = (load, 16, load.args[0])
            if block.term.kind != 'br':
                continue
            value = block.term.args[0]
            current = value
            casts = []
            while self.source.defs[current].kind == 'cast':
                cast = self.source.defs[current]
                if self.uses[current] != 1 or cast.args[0].type.kind != 'int' or current.type.kind != 'int' or (current.type.bits < cast.args[0].type.bits):
                    break
                casts.append(cast)
                current = cast.args[0]
            load = self.source.defs[current]
            if casts and load.kind == 'load' and (load.attrs['width'] <= 2) and (load in block.ops) and (not load.attrs.get('volatile')) and (self.uses[current] == 1) and (not any((op.kind in ('call', 'asm') or 'write' in op.attrs.get('memory', '') or op.attrs.get('volatile') for op in block.ops[block.ops.index(load) + 1:]))):
                self.folded.update([*casts, load])
                self.memory_truth[value] = load
        self.fn.string_regions = [(self.blocks[root], {self.blocks[b] for b in region}, color) for root, region, color in getattr(self.source, 'string_regions', [])]
        for value in list(self.folded_vectors):
            users = self.users.get(value, [])
            if len(users) > 1 and any((u.kind == 'intrinsic' and u.attrs['builtin'].op == 'cmpistrm' for u in users)):
                load = self.folded_vectors.pop(value)[0]
                self.folded.discard(load)
        for b in self.source.blocks:
            if b.term.kind != 'br' or b.term.targets[0].term.kind != 'ret':
                continue
            op = self.source.defs[b.term.args[0]]
            if op.kind not in ('lt', 'le', 'gt', 'ge'):
                continue
            if op.attrs.get('index_compare') or any((self.source.defs[string_bare(self.source, v)].kind == 'intrinsic' and self.source.defs[string_bare(self.source, v)].attrs['builtin'].op == 'cmpistri' for v in op.args)):
                self.valid_return_tests.add(self.blocks[b])

    def bit_count(self, value):
        op = self.source.defs[string_bare(self.source, value)]
        return (super().bit_count(value) or op.kind == "intrinsic" and op.attrs["builtin"].op == "cmpistri"
                or string_bare(self.source, value) in getattr(self.source, 'string_indices', ()))

    def zero_extension(self, op):
        return False if op.attrs.get("force_zero_extension") else super().zero_extension(op)

    def operation(self, op):
        if op.attrs.get('force_zero_extension'):
            self.ins('mov', [M.R(self.value(op.results[0]).reg, 32),
                             M.R(self.operand(op.args[0], True).reg, 32)], ['W', 'R'], preserve=True)
            return
        if op.results and op.results[0] in getattr(self.source, 'mask_branches', set()):
            return
        return super().operation(op)

    def condition(self, value):
        if value in self.memory_truth:
            op = self.memory_truth[value]
            self.ins('cmp', [self.address(op.args[0], op.attrs['width']), M.Imm(0, 8)], ['R', 'R'])
            return 'ne'
        if value in getattr(self.source, 'mask_branches', set()):
            op = self.source.defs[value]
            self.ins('test', [self.operand(v, True) for v in op.args], ['R', 'R'])
            return 'ne'
        return super().condition(value)

    def compare(self, op, invert=False):
        if op.attrs.get('existing_flags'):
            return 'ae' if invert else 'b'
        if op.attrs.get('index_compare'):
            self.ins('cmp', [M.R(self.operand(v, True).reg, 32) for v in op.args[::-1]], ['R', 'R'])
            return 'be' if invert else 'a'
        if op.attrs.get('wide_compare'):
            self.ins('cmp', [self.operand(v, True) for v in op.args], ['R', 'R'])
            return 'be' if invert else 'a'
        return super().compare(op, invert)

    def intrinsic(self, op):
        if op.attrs['builtin'].op == 'cmpistri':
            self.fn.preferred[self.value(op.results[0]).reg] = M.physical('rcx')
        if self.fn.isa == 'sse42':
            b = op.attrs['builtin']
            result = self.value(op.results[0]) if op.results else None
            if b.op == 'splat8':
                source = op.args[0]
                while self.source.defs[source].kind == 'cast':
                    source = self.source.defs[source].args[0]
                value = self.operand(source, True)
                zero = self.operand(self.source.constant(T.V128, bytes(16)), True)
                self.ins('movd', [result, M.R(value.reg, 32)], ['W', 'R'])
                self.ins('pshufb', [result, zero], ['RW', 'R'])
                return
            if b.op == 'add_epi8' and self.source.defs[op.args[0]].kind == 'constant':
                self.binary_args[op] = op.args[::-1]
            if b.op == 'or':
                for value, mask in (op.args, op.args[::-1]):
                    definition = self.definition(mask)
                    if definition.kind == 'intrinsic' and definition.attrs['builtin'].op == 'cmpgt_epi8' and (definition.args[0] == value):
                        fixed = M.R(M.physical('xmm0'), 128)
                        m = self.operand(mask, True)
                        self.fn.preferred[m.reg] = fixed.reg
                        self.move(fixed, m)
                        self.move(result, self.operand(value, True))
                        users = self.users.get(op.results[0], ())
                        shuffle_control = users and all(
                            user.kind == 'intrinsic' and user.attrs['builtin'].op == 'shuffle_epi8'
                            and user.args[1] == op.results[0] for user in users)
                        # PSHUFB consumes the high bit and low nibble only.
                        padding = 0x8f if shuffle_control else 0xff
                        memory = M.Mem(16, symbol=self.constant_name(bytes([padding]) * 16))
                        self.ins('pblendvb', [result, memory], ['RW', 'R'], implicit_uses=frozenset({fixed.reg}))
                        return
        return super().intrinsic(op)

    def run(self):
        fn, pool = super().run()
        for block in self.source.blocks:
            if block.term.kind == 'ret' and block.term.args and (block.term.args[0].type.kind == 'int') and (block.term.args[0].type.bits == 64) and (possible_bits(self.source, block.term.args[0]) < 1 << 32):
                ins = self.blocks[block].ins
                if len(ins) >= 2 and ins[-1].mnemonic == 'ret' and (ins[-2].mnemonic == 'mov') and all((isinstance(o, M.R) and o.reg.bank == 'gpr' for o in ins[-2].operands)) and (ins[-2].operands[0].reg == M.physical('rax')):
                    ins[-2].operands = tuple((M.R(o.reg, 32) for o in ins[-2].operands))
        local_zeros(fn)
        known = {self.value(op.results[0]).reg for b in self.source.blocks for op in b.ops if op.kind == 'intrinsic' and op.results and (op.attrs['builtin'].op in ('cmpistri', 'ctz_nonzero', 'tzcnt'))}
        known.update(self.value(v).reg for v in getattr(self.source, 'string_indices', ()))
        for b in fn.blocks:
            for ins in b.ins:
                if ins.mnemonic == 'mov' and len(ins.operands) == 2 and all((isinstance(o, M.R) and o.bits == 64 and (o.reg.bank == 'gpr') for o in ins.operands)) and (ins.operands[1].reg in known):
                    ins.operands = tuple((M.R(o.reg, 32) for o in ins.operands))
        for source, shift, compare in getattr(self.source, 'mask_schedules', []):
            block = self.blocks[source]
            result = self.value(compare).reg
            copy = next((i for i in block.ins if i.mnemonic in ('movaps', 'movdqa') and result in i.defs and all((isinstance(o, M.R) for o in i.operands))), None)
            start = next((i for i in block.ins if i.mnemonic == 'mov' and len(i.operands) == 2 and isinstance(i.operands[1], M.Imm) and (i.operands[1].value == -1)), None)
            if copy is not None and start is not None and (block.ins.index(copy) > block.ins.index(start)):
                block.ins.remove(copy)
                block.ins.insert(block.ins.index(start), copy)
        for block in fn.blocks:
            for ins in list(block.ins):
                if ins.mnemonic != 'lea' or not isinstance(ins.operands[1], M.Mem) or ins.operands[1].base != self.value(self.source.params[1]).reg or (ins.operands[1].disp != 16) or (len(block.successors) != 2):
                    continue
                reg = ins.operands[0].reg
                targets = [s for s in block.successors if any((reg in i.uses for i in s.ins))]
                if len(targets) == 1 and (not any((reg in i.uses for i in block.ins[block.ins.index(ins) + 1:]))):
                    block.ins.remove(ins)
                    targets[0].ins.insert(0, ins)
        retained = {self.value(v).reg for v in getattr(self.source, 'word_return_bases', ())}
        for block in fn.blocks:
            for i in block.ins:
                i.operands = tuple((replace(o, base=o.index, index=o.base) if isinstance(o, M.Mem) and o.index in retained and (o.scale == 1) else o for o in i.operands))
        if hasattr(self.source, 'string_spn_mask_return'):
            fn.string_fast_return = self.blocks[self.source.string_spn_mask_return]
        if getattr(self.source, 'string_cspn', None):
            fn.ssa_blocks = self.blocks.copy()
            block = self.blocks[self.source.string_cspn['invert']]
            branch, jump = block.ins[-2:]
            branch.target, jump.target = jump.target, branch.target
            branch.cc = INVERSE[branch.cc]
            branch.mnemonic = 'j' + branch.cc
        return fn, pool


def share_epilogue(fn):
    """Let a precomputed mask return enter the same suffix as its loop."""
    recipe = getattr(fn.source, 'string_cspn', None)
    if recipe is not None:
        first, second = (resolve_block(fn, source) for source in recipe['returns'])
        def signature(ins):
            return ins.mnemonic, ins.operands, ins.access, ins.implicit_uses, ins.implicit_defs
        if (first is not None and second is not None and first is not second
                and [signature(i) for i in first.ins] == [signature(i) for i in second.ins]):
            for block in fn.blocks:
                block.successors = [first if b is second else b for b in block.successors]
                for ins in block.ins:
                    if ins.target is second:
                        ins.target = first
            fn.blocks.remove(second)
        for source in recipe['sentinels']:
            edge = resolve_block(fn, source)
            if edge is None or len(edge.successors) != 1:
                continue
            joined = edge.successors[0]
            edge.ins[-1:] = [replace(ins) for ins in joined.ins]
            edge.successors = joined.successors[:]
        return True
    if not hasattr(fn, 'string_fast_return'):
        return False
    fast = fn.string_fast_return
    loops = [b for b in fn.blocks if b is not fast and b.ins and b.ins[-1].mnemonic == 'ret'
             and any(i.mnemonic == 'tzcnt' for i in b.ins) and any(i.mnemonic == 'add' for i in b.ins)]
    if len(loops) != 1:
        return False
    loop = loops[0]
    loop.ins[-1] = M.Instr('jmp', target=fast)
    loop.successors = [fast]
    fn.layout_owners[fast] = loop
    fn.retained_returns = set(getattr(fn, 'retained_returns', ())) | {fast}
    return True


def resolve_block(fn, source):
    """Follow empty SSA/phi edge aliases removed by machine cleanup."""
    block, seen = fn.ssa_blocks[source], set()
    while block not in fn.blocks and block not in seen:
        seen.add(block)
        if not block.ins or block.ins[-1].mnemonic != 'jmp' or block.ins[-1].target is None:
            return None
        block = block.ins[-1].target
    return block if block in fn.blocks else None


def layout_blocks(fn):
    recipe = getattr(fn.source, 'string_cspn', None)
    if recipe is None:
        return
    ordered = []
    for source in fn.source.string_block_order:
        block = resolve_block(fn, source)
        if block is not None and block not in ordered:
            ordered.append(block)
        ordered.extend(b for b in fn.blocks if getattr(b, 'source_parent', None) is source and b not in ordered)
    fn.block_order = ordered
    fn.loop_alignments = {resolve_block(fn, head): 0 for head in recipe['heads']}
