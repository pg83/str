"""Select x86 operations from typed SSA, retaining virtual registers."""
from collections import Counter
from dataclasses import replace

from . import types as T
from . import machine as M
from .regalloc import CALLER_SAVED, liveness, pointer_parameters
from .facts import boolean_values
from .predicates import bit_test
from .ir import dominators
from .facts import possible_bits
from .alignment import alignments
from .stringops import fold_flag_branches, retain_string_indices
from .recipes import schedule_function


ARGS = ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9']
INVERSE = {'e': 'ne', 'ne': 'e', 'b': 'ae', 'ae': 'b', 'a': 'be', 'be': 'a',
           'l': 'ge', 'ge': 'l', 'g': 'le', 'le': 'g'}


class Select:
    def __init__(self, fn, isa='sse2', trampolines=None):
        self.source, self.fn = fn, M.Function(fn, isa)
        self.blocks = {b: self.fn.block(b.name) for b in fn.blocks}
        self.valid_return_tests = set()
        for block in fn.blocks:
            if self.fn.isa not in ('sse2', 'sse42') or block.term.kind != 'br':
                continue
            test = fn.defs[block.term.args[0]]
            if test.kind == 'ne' and fn.defs[test.args[1]].kind == 'constant':
                constant = fn.defs[test.args[1]].attrs['value']
                returned = block.term.targets[0].term
                if constant == (1 << test.args[0].type.bits) - 1 and returned.kind == 'ret' and returned.args == (test.args[0],):
                    self.valid_return_tests.add(self.blocks[block])
        self.direct_return_blocks = {self.blocks[b] for b in getattr(fn, 'direct_return_paths', ())}
        self.block = self.fn.entry
        self.fn.cold_blocks = {self.blocks[b] for b in getattr(fn, 'cold', ())}
        self.fn.preferred.update((M.Reg('gpr', v.id), M.physical(reg)) for v, reg in getattr(fn, 'induction_registers', {}).items())
        self.fn.preferred.update((M.Reg('gpr', v.id), M.physical(reg)) for v, reg in getattr(fn, 'scan_registers', {}).items())
        self.fn.preferred.update((M.Reg('vec', v.id), M.physical(reg)) for v, reg in getattr(fn, 'vector_registers', {}).items())
        self.fn.layout_owners = {self.blocks[b]: self.blocks[p] for b, p in getattr(fn, 'layout_owners', {}).items()
                                 if b in self.blocks and p in self.blocks and b is not p}
        for b, value in getattr(fn, 'layout_conditions', {}).items():
            owner = next((p for p in fn.blocks if p.term.kind == 'br' and p.term.args == (value,)), None)
            if b in self.blocks and owner is not None:
                self.fn.layout_owners[self.blocks[b]] = self.blocks[owner]
        dom, pred = dominators(fn), fn.predecessors()
        self.inline_zero_guards, self.table_constant_returns = set(), {}
        for head in getattr(fn, 'indexed_byte_scans', {}):
            returned = next(b for b in head.term.targets if b is not head)
            rejected = fn.defs[head.term.args[0]].kind == 'or'
            if returned.term.kind == 'ret':
                for block in fn.blocks:
                    if block.term.kind == 'ret' and block.term.args and fn.defs[block.term.args[0]].kind == 'constant':
                        value = fn.defs[block.term.args[0]].attrs['value']
                        if rejected and len(pred[block]) == 1 and pred[block][0].term.kind == 'br':
                            probe = pred[block][0]
                            if value == 0:
                                self.inline_zero_guards.add(self.blocks[probe])
                                continue
                            self.table_constant_returns[self.blocks[block]] = self.blocks[probe]
                            test = fn.defs[probe.term.args[0]]
                            for op in probe.ops:
                                if op.kind == 'load' and op is not test and op.attrs['width'] == 1:
                                    self.fn.preferred[M.Reg('gpr', op.results[0].id)] = M.physical('rcx')
                        self.fn.layout_owners[self.blocks[block]] = self.blocks[returned]
        if getattr(fn, 'pipelined_copies', {}):
            for block in fn.blocks:
                if block.term.kind != 'jmp' or block.term.targets[0].term.kind != 'ret':
                    continue
                memory = [op for op in block.ops if op.attrs.get('memory')]
                if (len(memory) == 2 and memory[0].kind == 'load' and memory[1].kind == 'store'
                        and memory[0].attrs['width'] == memory[1].attrs['width'] == 1):
                    self.fn.layout_owners[self.blocks[block.term.targets[0]]] = self.blocks[block]
        zero_returns = [b for b in fn.blocks if b.term.kind == 'ret' and b.term.args
                        and fn.defs[b.term.args[0]].kind == 'constant' and fn.defs[b.term.args[0]].attrs['value'] == 0]
        for head in getattr(fn, 'preincrement_scans', {}):
            incoming = [b for b in pred[head] if b is not head]
            if len(incoming) != 1 or len(pred[incoming[0]]) != 1:
                continue
            before, probe = incoming[0], pred[incoming[0]][0]
            if probe.term.kind != 'br' or before not in probe.term.targets:
                continue
            returned = next(b for b in probe.term.targets if b is not before)
            if returned.term.kind == 'ret' and returned not in zero_returns:
                for zero in zero_returns:
                    self.fn.layout_owners[self.blocks[zero]] = self.blocks[returned]
                    self.fn.layout_owners[self.blocks[before]] = self.blocks[zero]
        self.scalar_loop_exits = {}
        for entry, tail, exit_, before in getattr(fn, 'scalar_read_loops', ()):
            if any(b not in self.blocks for b in (entry, tail, exit_, before)) or before.term.kind != 'br' or entry not in before.term.targets:
                continue
            alternative = next(b for b in before.term.targets if b is not entry)
            if alternative.term.kind != 'ret' and (alternative.term.kind != 'jmp' or alternative.term.targets != (exit_,)):
                continue
            self.fn.layout_owners[self.blocks[alternative]] = self.blocks[exit_]
            self.fn.layout_owners[self.blocks[exit_]] = self.blocks[tail]
            self.scalar_loop_exits[exit_] = tail
            for target in entry.term.targets:
                if target is not tail and (target.term.kind == 'ret' or target.term.kind == 'jmp' and target.term.targets == (exit_,)):
                    self.fn.layout_owners[self.blocks[target]] = self.blocks[alternative]
        displaced = sorted((b for b in getattr(fn, 'displaced_returns', ()) if b in self.blocks), key=lambda b: (len(dom[b]), b.id))
        if displaced and all(len(pred[b]) == 1 and pred[b][0].term.kind == 'br' for b in displaced):
            last = pred[displaced[-1]][0]
            following = next((b for b in last.term.targets if b is not displaced[-1]), None)
            if following is not None:
                for block in displaced:
                    self.fn.layout_owners[self.blocks[block]] = self.blocks[following]
            for block in getattr(fn, 'scalar_return_paths', ()):
                if block in self.blocks and block not in displaced:
                    self.fn.layout_owners[self.blocks[block]] = self.blocks[displaced[-1]]
            first = pred[displaced[0]][0]
            if len(pred[first]) == 1 and pred[first][0].term.kind == 'br':
                partial = next((b for b in pred[first][0].term.targets if b is not first), None)
                if partial is not None:
                    self.fn.layout_owners[self.blocks[first]] = self.blocks[partial]
        probes = list(dict.fromkeys(b for chain in getattr(fn, 'distance_probes', ()) for b in chain if b in self.blocks))
        starts = [b for b in probes if not any(other is not b and other in dom[b] for other in probes)]
        chains = [sorted((b for b in probes if start in dom[b]), key=lambda b: len(dom[b])) for start in starts]
        if len(chains) > 1 and all(len(chain) >= 3 for chain in chains):
            # Keep two probes of each scan inline, then share the remaining
            # probes and return computations below the alternative entries.
            if fn.entry.term.kind == 'br':
                fallback = [b for b in fn.entry.term.targets if b not in dom[chains[0][0]]]
                if len(fallback) == 1:
                    self.fn.layout_owners[self.blocks[fallback[0]]] = self.blocks[chains[0][1]]
            for previous, chain in zip(chains, chains[1:]):
                self.fn.layout_owners[self.blocks[chain[0]]] = self.blocks[previous[1]]
            for chain in chains:
                for block in chain[2:]:
                    self.fn.layout_owners[self.blocks[block]] = self.blocks[chains[-1][1]]
        for entry in getattr(fn, 'fixed_loop_entries', ()):
            if entry not in self.blocks:
                continue
            fallbacks = [b for b in fn.blocks if b.term.kind == 'ret' and entry not in dom[b]
                         and any(op.kind == 'call' for op in b.ops)]
            if len(fallbacks) == 1:
                self.fn.layout_owners[self.blocks[entry]] = self.blocks[fallbacks[0]]
            reachable, pending = set(), list(getattr(fn, 'fixed_loop_exits', ()))
            while pending:
                block = pending.pop()
                if block not in reachable:
                    reachable.add(block)
                    pending.extend(block.term.targets)
            returns = [b for b in fn.blocks if b in reachable and b.term.kind == 'ret'
                       and not any(op.kind == 'call' for op in b.ops)]
            tails = [b for b in fn.blocks if b not in reachable and b.term.kind == 'ret' and entry in dom[b]
                     and any(op.kind == 'call' for op in b.ops)]
            if len(returns) == len(tails) == 1:
                self.fn.layout_owners[self.blocks[tails[0]]] = self.blocks[returns[0]]
        self.dominators = {self.blocks[b]: {self.blocks[p] for p in ancestors} for b, ancestors in dom.items()}
        self.loop_blocks = set()
        self.loops = {}
        self.loop_vectors = set()
        self.fn.loop_headers = set()
        self.fn.unaligned_loops = {self.blocks[b] for b in getattr(fn, 'unaligned_loops', ()) if b in self.blocks}
        self.fn.loop_alignments = {self.blocks[b]: align for b, align in getattr(fn, 'loop_alignments', {}).items() if b in self.blocks}
        for tail in fn.blocks:
            for head in tail.term.targets:
                if head not in dom[tail]:
                    continue
                self.fn.loop_headers.add(self.blocks[head])
                members, todo = {head}, [tail]
                while todo:
                    b = todo.pop()
                    if b not in members:
                        members.add(b)
                        todo.extend(pred[b])
                self.loop_blocks.update(self.blocks[b] for b in members)
                self.loops.setdefault(self.blocks[head], set()).update(self.blocks[b] for b in members)
                defined = {v for b in members for op in b.ops for v in op.results if v.type.kind == 'vector'}
                self.loop_vectors.update(v for b in fn.blocks if b not in members
                                         for v in (*b.term.args, *(v for op in b.ops for v in op.args)) if v in defined)
        self.loop_dependencies = set()
        pending = [v for b in fn.blocks if self.blocks[b] in self.loop_blocks
                   for v in (*b.term.args, *(v for op in b.ops for v in op.args))]
        while pending:
            value = pending.pop()
            if value in self.loop_dependencies or value.type.kind not in ('int', 'ptr') or value.type.bits != 64:
                continue
            self.loop_dependencies.add(value)
            op = fn.defs[value]
            if op.kind in ('phi', 'cast', 'ptradd', 'add', 'sub', 'mul', 'and', 'or', 'xor', 'shl', 'shr'):
                pending.extend(op.args)
        self.trampolines = trampolines or {}
        self.fn.loop_blocks = self.loop_blocks
        self.values, self.stack, self.constants = {}, {}, {}
        self.primary_constants = set()
        self.owners = {v: self.blocks[b] for b in fn.blocks for op in b.ops for v in op.results}
        self.uses = Counter(v for b in fn.blocks for op in b.ops for v in op.args)
        self.uses.update(v for b in fn.blocks for v in b.term.args)
        self.predicate_values = {v for b in fn.blocks for op in b.ops for v in op.results
                                 if op.kind == 'intrinsic' and op.attrs['builtin'].name.endswith('_mask')}
        self.vector_mask_compares, self.mask_widths = {}, {}
        if isa in ('evex', 'evex512'):
            for block in fn.blocks:
                for op in block.ops:
                    if op.kind != 'intrinsic' or op.attrs['builtin'].op != 'movemask8':
                        continue
                    compare = self.definition(op.args[0])
                    if (compare.kind != 'intrinsic' or compare.attrs['builtin'].op != 'cmpeq_epi8'
                            or self.uses[compare.results[0]] != 1):
                        continue
                    for n, arg in enumerate(compare.args):
                        zero = self.definition(arg)
                        if zero.kind == 'constant' and isinstance(zero.attrs['value'], bytes) and not any(zero.attrs['value']):
                            value = compare.args[1 - n]
                            self.vector_mask_compares[op] = value, compare
                            self.predicate_values.add(op.results[0])
                            self.mask_widths[op.results[0]] = value.type.size
                            break
        candidates = {v for b in fn.blocks for op in b.ops for v in op.results if v.type.kind == 'int'
                      and (op.kind == 'phi' or op.kind == 'cast' and op.args[0].type.bits == v.type.bits)}
        while True:
            rejected = {v for v in candidates if any(a not in self.predicate_values | candidates for a in fn.defs[v].args)}
            if not rejected:
                break
            candidates -= rejected
        self.predicate_values |= candidates
        scalar_phis = {v for b in fn.blocks for op in b.ops if op.kind == 'phi' for v in op.results
                       if v.type.bits <= 32 and v in self.predicate_values}
        self.scalar_search_masks = bool(fn.type.base.kind == 'ptr' and getattr(fn, 'return_joins', ()) and scalar_phis)
        if self.scalar_search_masks:
            self.predicate_values -= scalar_phis
        self.mask_phi_blocks = {M.Reg('mask', v.id): self.blocks[b] for b in fn.blocks for op in b.ops
                                if op.kind == 'phi' for v in op.results if v in self.predicate_values}
        self.booleans = boolean_values(fn)
        self.hoisted, self.constant_registers = [], {}
        self.scalar_constants = {}
        self.address_tests = {}
        self.mask_values = {}
        self.scalar_masks = {}
        self.mask_tests = []
        self.fn.tail_pointers = {self.value(v).reg for v in getattr(fn, 'pointer_scan_steps', ())}
        if fn.type.base.kind == 'ptr' and getattr(fn, 'return_joins', ()):
            def pointer(value):
                while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
                    value = fn.defs[value].args[0]
                return value
            reads = {pointer(op.args[0]) for b in fn.blocks for op in b.ops if op.kind == 'intrinsic'
                     and op.attrs['builtin'].op == 'load'}
            for block in fn.blocks:
                if (block.term.kind != 'ret' or not block.ops or block.ops[-1].kind != 'call'
                        or len(block.ops[-1].args) < 2):
                    continue
                value = pointer(block.ops[-1].args[1])
                definition = fn.defs[value]
                if value in reads and definition.kind == 'ptradd' and fn.defs[definition.args[1]].kind == 'constant':
                    reg = self.value(value).reg
                    self.fn.tail_pointers.add(reg)
                    self.fn.preferred[reg] = M.physical('rcx')
        self.users = {}
        for b in fn.blocks:
            for op in b.ops:
                for v in op.args:
                    self.users.setdefault(v, []).append(op)
        for join in getattr(fn, 'guarded_pointer_returns', ()):
            for op in join.ops:
                if op.kind != 'phi':
                    continue
                self.fn.preferred[self.value(op.results[0]).reg] = M.physical('rcx')
                for value in op.args:
                    scan = self.definition(value)
                    if scan.kind == 'cast':
                        scan = self.definition(scan.args[0])
                    if scan.kind == 'intrinsic' and scan.attrs['builtin'].op == 'ctz_nonzero':
                        mask = self.definition(scan.args[0])
                        if mask.kind == 'intrinsic' and mask.attrs['builtin'].op == 'movemask8':
                            self.fn.preferred[self.value(mask.results[0]).reg] = M.physical('rax')
        self.folded = {compare for _, compare in self.vector_mask_compares.values()}
        for body, _, difference, *_ in getattr(fn, 'word_scans', {}).values():
            load = self.definition(difference.args[1])
            if load.kind == 'load' and load in body.ops and self.register_uses(load.results[0]) == 1:
                self.folded.add(load)
        self.address_sums = set()
        if getattr(fn, 'indexed_byte_scans', {}):
            for block in fn.blocks:
                for op in block.ops:
                    if op.kind != 'add' or op.results[0].type.bits != 64:
                        continue
                    inner = fn.defs[op.args[0]]
                    if inner.kind == 'add' and self.uses[op.args[0]] == 1 and fn.defs[inner.args[1]].kind == 'constant':
                        self.folded.add(inner)
                        self.address_sums.add(op)
        for head in getattr(fn, 'indexed_byte_scans', {}):
            condition = fn.defs[head.term.args[0]]
            if condition.kind == 'and' and self.uses[head.term.args[0]] == 1:
                self.folded.add(condition)
        self.branch_tests = {}
        self.compare_tests = {}
        self.shift_counts = {}
        for block in fn.blocks:
            for op in block.ops:
                if op.kind not in ('eq', 'ne'):
                    continue
                right = fn.defs[op.args[1]]
                test = bit_test(fn, op.args[0])
                if test is None and getattr(fn, 'aligned_vector_search', False):
                    masked = self.definition(op.args[0])
                    if (masked.kind == 'and' and fn.defs[masked.args[1]].kind == 'constant'
                            and 0 < fn.defs[masked.args[1]].attrs['value'] < 1 << 32):
                        test = masked.args[0], fn.defs[masked.args[1]].attrs['value']
                if right.kind != 'constant' or right.attrs['value'] != 0 or test is None or test[1] >= 1 << 32:
                    continue
                self.compare_tests[op] = test
                value = op.args[0]
                while value != test[0] and self.uses[value] == 1:
                    folded = fn.defs[value]
                    self.folded.add(folded)
                    value = folded.args[0]
        pure_memory = not any(op.kind == 'call' or 'write' in op.attrs.get('memory', '') for b in fn.blocks for op in b.ops)
        if pure_memory:
            for b in fn.blocks:
                for op in b.ops:
                    if op.kind in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                        left, right = self.definition(op.args[0]), self.definition(op.args[1])
                        chosen = right if right.kind == 'load' else left
                        if chosen.kind == 'load' and not chosen.attrs.get('volatile') and self.uses[chosen.results[0]] == 1:
                            self.folded.add(chosen)
        self.condition_flags = {}
        conditions = [b.term.args[0] for b in fn.blocks if b.term.kind == 'br']
        for block in fn.blocks:
            for op in block.ops:
                if (getattr(fn, 'large_byte_compares', False) and op.kind in ('eq', 'ne', 'lt', 'gt', 'le', 'ge')
                        and self.users.get(op.results[0]) and self.uses[op.results[0]] == len(self.users[op.results[0]])
                        and all(user.kind == 'select' and user.args[0] == op.results[0] for user in self.users[op.results[0]])):
                    self.folded.add(op)
        for block in fn.blocks:
            if block.term.kind != 'br':
                continue
            load = self.definition(block.term.args[0])
            if (load.kind == 'load' and load in block.ops and not load.attrs.get('volatile')
                    and load.attrs['width'] <= 8 and self.uses[load.results[0]] == 1
                    and not any(op.kind in ('call', 'asm') or 'write' in op.attrs.get('memory', '') or op.attrs.get('volatile')
                                for op in block.ops[block.ops.index(load)+1:])):
                self.folded.add(load)
        conditions.extend(op.args[0] for b in fn.blocks for op in b.ops if op.kind == 'select')
        for value in conditions:
            op = fn.defs[value]
            if (op.kind == 'intrinsic' and op.attrs['builtin'].op == 'testz' and self.uses[value] == 1):
                self.folded.add(op)
            while op.kind == 'cast' and self.uses[op.results[0]] == 1 and op.args[0].type.bits == op.results[0].type.bits:
                op = fn.defs[op.args[0]]
            if op.kind == 'and' and self.uses[value] == self.uses[op.results[0]] == 1:
                if op.attrs.get('boolean_test'):
                    self.folded.add(op)
                mask = fn.defs[op.args[1]]
                if mask.kind == 'constant' and 0 < mask.attrs['value'] < 1 << 32:
                    self.folded.add(op)
        for block in fn.blocks:
            for op in block.ops:
                if op.kind not in ('shl', 'shr'):
                    continue
                count = fn.defs[op.args[1]]
                if count.kind == 'and' and self.uses[count.results[0]] == 1:
                    mask = fn.defs[count.args[1]]
                    if mask.kind == 'constant' and mask.attrs['value'] == op.args[0].type.bits - 1:
                        self.shift_counts[op] = count.args[0]
                        self.folded.add(count)
                value = self.shift_counts.get(op, op.args[1])
                while True:
                    count = fn.defs[value]
                    if count.kind == 'cast' and count.args[0].type.bits >= (op.args[0].type.bits - 1).bit_length():
                        value = count.args[0]
                    elif count.kind in ('add', 'sub') and not possible_bits(fn, count.args[1]) & (op.args[0].type.bits - 1):
                        value = count.args[0]
                    else:
                        break
                if value != op.args[1]:
                    self.shift_counts[op] = value
        for op in fn.allocas:
            self.stack[op.results[0]] = self.fn.slot(op.attrs['size'], op.attrs['alignment'])
        for b in fn.blocks:
            for i, op in enumerate(b.ops):
                if op.kind != 'call':
                    continue
                target = self.definition(op.args[0])
                if target.kind == 'load' and self.uses[target.results[0]] == 1 and target in b.ops[:i]:
                    after = b.ops[b.ops.index(target) + 1:i]
                    if not any('write' in x.attrs.get('memory', '') for x in after):
                        self.folded.add(target)
        for block in fn.blocks:
            if block.term.kind != 'br':
                continue
            value, chain = block.term.args[0], []
            op = fn.defs[value]
            while op.kind == 'cast' and self.uses[op.results[0]] == 1:
                chain.append(op)
                op = fn.defs[op.args[0]]
            if op.kind != 'and' or self.uses[op.results[0]] != 1:
                continue
            mask = fn.defs[op.args[1]]
            if mask.kind != 'constant' or mask.attrs['value'] != 1:
                continue
            chain.append(op)
            shift = fn.defs[op.args[0]]
            if shift.kind != 'shr' or self.uses[shift.results[0]] != 1:
                continue
            count = fn.defs[shift.args[1]]
            if count.kind != 'constant' or not 0 <= count.attrs['value'] < shift.args[0].type.bits:
                continue
            chain.append(shift)
            source, bit = shift.args[0], count.attrs['value']
            load = fn.defs[source]
            memory = load.kind == 'load' and not load.attrs.get('volatile') and self.uses[source] == 1 and load in block.ops
            if memory:
                after = block.ops[block.ops.index(load) + 1:]
                memory = not any(x.kind in ('call', 'asm') or 'write' in x.attrs.get('memory', '') for x in after)
            self.branch_tests[value] = source, bit, memory
            self.folded.update(chain)
            if memory:
                self.folded.add(load)
        self.binary_args, self.mask_args, self.folded_vectors = {}, {}, {}
        self.zero_mask_compares = set()
        self.alignments = alignments(fn)
        self.prepare_binary_operands()
        self.mask_packs = {}
        def packed_input(value):
            casts = []
            while fn.defs[value].kind == 'cast' and self.uses[value] == 1:
                op = fn.defs[value]
                casts.append(op)
                value = op.args[0]
            return (value, casts) if value in self.predicate_values and value.type.bits == 32 and self.uses[value] == 1 else None
        for block in fn.blocks:
            for op in block.ops:
                if op.kind != 'or' or op.results[0].type.bits != 64:
                    continue
                for low, high in (op.args, op.args[::-1]):
                    shift = fn.defs[high]
                    if shift.kind != 'shl' or self.uses[high] != 1 or fn.defs[shift.args[1]].kind != 'constant' or fn.defs[shift.args[1]].attrs['value'] != 32:
                        continue
                    a, b = packed_input(low), packed_input(shift.args[0])
                    if a is not None and b is not None:
                        self.mask_packs[op] = a[0], b[0]
                        self.predicate_values.add(op.results[0])
                        self.folded.update([shift, *a[1], *b[1]])
                        break
        self.mask_intersections, self.mask_unions = {}, {}
        for block in fn.blocks:
            for op in block.ops:
                if op.kind != 'intrinsic' or '_mask_' not in op.attrs['builtin'].name:
                    continue
                builtin = op.attrs['builtin']
                if builtin.op not in ('cmpeq_epi8_masked', 'testn_epi8_masked'):
                    continue
                predicate, a, b = op.args
                test = self.definition(predicate)
                if (test.kind != 'intrinsic' or test.attrs['builtin'].op != 'test_epi8_mask'
                        or test.args[0] != test.args[1] or self.uses[predicate] != 1):
                    continue
                if builtin.op == 'cmpeq_epi8_masked' and not all(self.definition(v).kind == 'intrinsic'
                        and self.definition(v).attrs['builtin'].op == 'masked_load' for v in (a, b)):
                    continue
                self.mask_intersections[op] = test.args[0], a, b
                self.folded.add(test)
        for value in conditions:
            comparison = self.definition(value)
            if comparison.kind not in ('eq', 'ne'):
                continue
            a, b = comparison.args
            constant = fn.defs[b]
            mask = self.definition(a)
            if (mask not in self.mask_intersections or mask.attrs['builtin'].op != 'testn_epi8_masked'
                    or self.uses[a] != 1 or constant.kind != 'constant'
                    or constant.attrs['value'] != (1 << a.type.bits) - 1):
                continue
            x, d, other = self.mask_intersections[mask]
            if d == other:
                self.mask_unions[comparison] = x, d
                self.folded.add(mask)
        self.bitsets, self.narrow_counts = {}, set()
        self.low_bit_masks = {}
        if self.fn.isa in ('avx2', 'evex', 'evex512'):
            for block in fn.blocks:
                for op in block.ops:
                    if op.kind != 'xor' or op.results[0].type.bits not in (32, 64):
                        continue
                    for value, decremented in (op.args, op.args[::-1]):
                        decrement = self.definition(decremented)
                        if (decrement.kind == 'sub' and self.definition(decrement.args[0]) is self.definition(value)
                                and self.definition(decrement.args[1]).kind == 'constant'
                                and self.definition(decrement.args[1]).attrs['value'] == 1):
                            self.low_bit_masks[op] = value
                            break
        self.narrow_shifts = {}
        self.byte_counts = set()
        def count_uses_only(value, seen=frozenset()):
            users = self.users.get(value, ())
            if value in seen or not users or self.uses[value] != len(users):
                return False
            return all(user.kind in ('shl', 'shr') and user.args[1] == value
                       or user.kind == 'cast' and user.results[0].type.bits >= 8
                       and count_uses_only(user.results[0], seen | {value}) for user in users)
        for block in fn.blocks:
            for op in block.ops:
                if (op.kind == 'and' and op.results[0].type.bits in (32, 64)
                        and fn.defs[op.args[1]].kind == 'constant'
                        and 0 < fn.defs[op.args[1]].attrs['value'] < 256 and count_uses_only(op.results[0])):
                    self.byte_counts.add(op)
                if op.kind not in ('shl', 'shr'):
                    continue
                count = fn.defs[self.shift_counts.get(op, op.args[1])]
                if count.kind not in ('add', 'sub') or count.results[0].type.bits != 64 or self.uses[count.results[0]] != 1:
                    continue
                amount = fn.defs[count.args[1]]
                if amount.kind == 'constant' and -(1 << 31) <= amount.attrs['value'] < 1 << 31:
                    self.narrow_shifts[count] = count.args[0], amount.attrs['value'] * (-1 if count.kind == 'sub' else 1)
        for block in fn.blocks:
            for op in block.ops:
                if op.kind != 'or' or op.results[0].type.bits not in (32, 64):
                    continue
                for value, base in ((op.args[1], op.args[0]), (op.args[0], op.args[1])):
                    shift = fn.defs[value]
                    if shift.kind != 'shl' or self.uses[value] != 1:
                        continue
                    one = fn.defs[shift.args[0]]
                    if one.kind != 'constant' or one.attrs['value'] != 1:
                        continue
                    extra, casts, current = None, [], base
                    while fn.defs[current].kind == 'cast' and self.uses[current] == 1:
                        cast = fn.defs[current]
                        if current.type.bits != cast.args[0].type.bits:
                            break
                        casts.append(cast)
                        current = cast.args[0]
                    combined = fn.defs[current]
                    if combined.kind == 'or' and self.uses[current] == 1:
                        for low, shifted in (combined.args, combined.args[::-1]):
                            mask = low
                            while fn.defs[mask].kind == 'cast':
                                mask = fn.defs[mask].args[0]
                            scalar = fn.defs[mask]
                            if scalar.kind == 'xor' and fn.defs[scalar.args[1]].kind == 'constant':
                                mask = scalar.args[0]
                                while fn.defs[mask].kind == 'cast':
                                    mask = fn.defs[mask].args[0]
                                scalar = fn.defs[mask]
                            legacy_mask = (self.fn.isa in ('sse2', 'sse42') and scalar.kind == 'intrinsic'
                                           and scalar.attrs['builtin'].op == 'movemask8')
                            if ((mask in self.predicate_values and mask.type.bits == 32 or legacy_mask)
                                    and fn.defs[shifted].kind == 'shl'):
                                base, extra = (low, shifted) if legacy_mask else (shifted, low)
                                self.folded.update([combined, *casts])
                                break
                    self.bitsets[op] = base, shift.args[1], extra
                    self.folded.add(shift)
                    count = fn.defs[shift.args[1]]
                    if count.kind == 'sub' and self.uses[shift.args[1]] == 1:
                        amount = fn.defs[count.args[1]]
                        if amount.kind == 'constant' and amount.attrs['value'] == 1:
                            self.narrow_counts.add(count)
                    break

    def prepare_binary_operands(self):
        fn = self.source
        aligned = self.alignments
        commutative = {'and', 'or', 'xor', 'cmpeq_epi8', 'min_epu8', 'add_epi8'}
        binary = commutative | {'andnot', 'cmpgt_epi8', 'sub_epi8', 'shuffle_epi8'}
        vex = self.fn.isa not in ('sse2', 'sse42')
        for block in fn.blocks:
            positions = {op: n for n, op in enumerate(block.ops)}
            def load(value, consumer, mask=None):
                op = fn.defs[value]
                while op.kind == 'cast' and op.args[0].type.bits == value.type.bits and self.uses[value] == 1:
                    value = op.args[0]
                    op = fn.defs[value]
                width, address = 0, None
                if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'load':
                    builtin = op.attrs['builtin']
                    width, alignment = builtin.width, builtin.alignment
                    address = op.args[0]
                elif (op.kind == 'intrinsic' and op.attrs['builtin'].op == 'masked_load' and mask is not None
                      and self.definition(mask) is self.definition(op.args[0])):
                    width, alignment, address = op.attrs['builtin'].width, 1, op.args[1]
                elif op.kind == 'load' and value.type.kind == 'vector' and not op.attrs.get('volatile'):
                    width, alignment = op.attrs['width'], op.attrs.get('alignment', 1)
                    address = op.args[0]
                if not width or self.uses[value] != 1 or op not in positions or positions[op] >= positions[consumer]:
                    return None
                if not vex and max(alignment, aligned.get(address, 1)) < width:
                    return None
                between = block.ops[positions[op] + 1:positions[consumer]]
                if any(x.kind in ('call', 'asm') or 'write' in x.attrs.get('memory', '') or x.attrs.get('volatile') for x in between):
                    return None
                return op, width, address
            for op in block.ops:
                if op.kind != 'intrinsic' or op in self.folded:
                    continue
                builtin = op.attrs['builtin']
                if builtin.name.endswith('_mask'):
                    masked = '_mask_' in builtin.name
                    predicate, args = (op.args[0], op.args[1:]) if masked else (None, op.args)
                    if builtin.op.startswith('cmpeq_epi8'):
                        zero = next((n for n, v in enumerate(args) if self.definition(v).kind == 'constant'
                                     and isinstance(self.definition(v).attrs['value'], bytes)
                                     and not any(self.definition(v).attrs['value'])), None)
                        if zero is not None:
                            value = args[1 - zero]
                            self.mask_args[op] = (predicate, value, value) if masked else (value, value)
                            self.zero_mask_compares.add(op)
                            continue
                    left, right = load(args[0], op, predicate), load(args[1], op, predicate)
                    if left is not None and (right is None or masked and builtin.op.startswith('cmpneq_epu8')):
                        args = args[::-1]
                    self.mask_args[op] = (predicate, *args) if masked else args
                    candidate = load(args[1], op, predicate)
                    if candidate is not None:
                        self.folded.add(candidate[0])
                        self.folded_vectors[args[1]] = candidate
                    continue
                if builtin.op not in binary:
                    continue
                kind, args = op.attrs['builtin'].op, op.args
                right = fn.defs[args[1]]
                zero = kind == 'cmpeq_epi8' and right.kind == 'constant' and isinstance(right.attrs['value'], bytes) and not any(right.attrs['value'])
                ones = (not vex and kind == 'xor' and right.kind == 'constant'
                        and isinstance(right.attrs['value'], bytes) and all(byte == 255 for byte in right.attrs['value'])
                        and not getattr(fn, 'word_scans', {}))
                if (zero or ones or kind in commutative and load(args[0], op) is not None and load(args[1], op) is None
                        and (vex or self.register_uses(args[1]) <= 1
                             or any(self.owners.get(v) is self.blocks[block] for v in getattr(fn, 'pointer_scan_steps', ())))):
                    args = args[::-1]
                elif (not vex and kind in commutative and all(self.definition(v).kind == 'intrinsic'
                        and self.definition(v).attrs['builtin'].op == 'load' for v in args)
                        and all(load(v, op) is None for v in args)):
                    self.fn.preferred.setdefault(self.value(args[0]).reg, M.physical('xmm2' if getattr(fn, 'word_scans', {}) else 'xmm0'))
                    args = args[::-1]
                self.binary_args[op] = args
                candidate = load(args[1], op)
                if candidate is not None:
                    self.folded.add(candidate[0])
                    self.folded_vectors[args[1]] = candidate

    def definition(self, v):
        op = self.source.defs[v]
        while op.kind == 'cast' and op.args[0].type.bits == v.type.bits:
            v = op.args[0]
            op = self.source.defs[v]
        return op

    def value(self, v):
        if v not in self.values:
            op = self.source.defs[v]
            if op.kind == 'cast' and op.args[0].type.bits == v.type.bits:
                base = self.value(op.args[0])
                self.values[v] = M.R(base.reg, base.bits if base.reg.bank == 'mask' else 8 if v in self.booleans else v.type.bits)
            elif self.zero_extension(op):
                base = self.value(op.args[0])
                if base.reg.bank == 'mask':
                    base = self.operand(op.args[0], True)
                self.values[v] = M.R(base.reg, 8 if v in self.booleans else v.type.bits)
                self.fn.sizes[base.reg] = max(v.type.bits, self.fn.sizes[base.reg])
            elif op.kind == 'intrinsic' and op.attrs['builtin'].op in ('widen_undef', 'truncate'):
                base = self.value(op.args[0])
                self.values[v] = M.R(base.reg, v.type.bits)
                self.fn.sizes[base.reg] = max(self.fn.sizes.get(base.reg, 0), v.type.bits)
            elif v in self.predicate_values:
                reg = M.Reg('mask', v.id)
                bits = self.mask_widths.get(v, v.type.bits)
                self.fn.sizes[reg] = bits
                self.values[v] = M.R(reg, bits)
            else:
                self.values[v] = self.fn.reg(v.type, v.id)
                if v in self.booleans:
                    self.values[v] = M.R(self.values[v].reg, 8)
        return self.values[v]

    def zero_extension(self, op):
        if op.kind != 'cast':
            return False
        source, result = op.args[0], op.results[0]
        definition = self.definition(source)
        if source in self.booleans and result not in self.booleans:
            return False
        if source.type.bits == 32 and result.type.bits == 64:
            nonnegative = not source.type.signed or possible_bits(self.source, source) < 1 << 31
            return nonnegative and definition.kind not in ('param', 'constant', 'undef')
        if source in self.predicate_values and source.type.bits in (8, 16) and result.type.bits >= 32 and not source.type.signed:
            return True
        return (source.type.bits in (8, 16) and result.type.bits >= 32 and not source.type.signed
                and definition.kind == 'load')

    def bit_count(self, value):
        return bool(self.bit_count_width(value))

    def bit_count_width(self, value):
        op = self.source.defs[value]
        while op.kind == 'cast':
            op = self.source.defs[op.args[0]]
        return op.args[0].type.bits if op.kind == 'intrinsic' and op.attrs['builtin'].op in ('ctz_nonzero', 'tzcnt') else 0

    def register_uses(self, value):
        op = self.source.defs[value]
        while op.kind == 'cast' and (op.args[0].type.bits == value.type.bits or self.zero_extension(op)):
            value = op.args[0]
            op = self.source.defs[value]
        def count(v):
            total = self.uses[v]
            for user in self.users.get(v, ()):
                if user.kind == 'cast' and (user.args[0].type.bits == user.results[0].type.bits or self.zero_extension(user)):
                    total += count(user.results[0]) - 1
            return total
        return count(value)

    def symbolic(self, v):
        op = self.definition(v)
        if op.kind == 'symbol':
            return op.attrs['name'], 0
        if op.kind == 'ptradd':
            base, offset = self.symbolic(op.args[0]), self.source.defs[op.args[1]]
            if base is not None and offset.kind == 'constant':
                return base[0], base[1] + T.convert_integer(offset.attrs['value'], T.LONG)
        return None

    def stack_argument(self, v, seen=frozenset()):
        if v in seen:
            return False
        op = self.source.defs[v]
        if op.kind == 'alloca':
            return True
        if op.kind == 'asm' and not op.attrs['code'] and len(op.results) == len(op.args):
            return self.stack_argument(op.args[op.results.index(v)], seen | {v})
        return op.kind in ('cast', 'ptradd', 'add', 'sub', 'phi', 'select') and any(
            self.stack_argument(a, seen | {v}) for a in op.args)

    def ins(self, name, operands=(), access=(), **kwargs):
        i = M.Instr(name, tuple(operands), tuple(access), **kwargs)
        self.block.ins.append(i)
        if name == 'test' and len(i.operands) == 2 and isinstance(i.operands[0], M.R) and isinstance(i.operands[1], M.Imm):
            self.address_tests.setdefault(i.operands[0].reg, set()).add(self.block)
        return i

    def move(self, d, s, preserve=False):
        mask = next((o for o in (d, s) if isinstance(o, M.R) and o.reg.bank == 'mask'), None)
        if mask is not None:
            return self.ins('kmov' + {8: 'b', 16: 'w', 32: 'd', 64: 'q'}[mask.bits], [d, s], ['W', 'R'], preserve=preserve)
        vec = isinstance(d, M.R) and d.reg.bank == 'vec' or isinstance(s, M.R) and s.reg.bank == 'vec'
        bits = d.bits if isinstance(d, M.R) else s.bits if isinstance(s, M.R) else d.size * 8
        if isinstance(d, M.R) and isinstance(s, M.R) and d.bits < s.bits:
            s = M.R(s.reg, d.bits)
        if not vec and isinstance(s, M.Imm) and isinstance(d, M.R) and bits == 64:
            if s.symbol is not None or 0 <= s.value < 1 << 32:
                d, s = M.R(d.reg, 32), replace(s, bits=32)
        if not vec and isinstance(s, M.Imm) and isinstance(d, M.R) and s.value == 0 and s.symbol is None:
            out = M.R(d.reg, max(32, d.bits))
            return self.ins('xor', [out, out], ['W', 'R'])
        opcode = ('movaps' if isinstance(d, M.R) and isinstance(s, M.R) else 'movups') if vec and bits == 128 and self.fn.isa in ('sse2', 'sse42') else 'vmovups' if vec else 'mov'
        return self.ins(opcode, [d, s], ['W', 'R'], preserve=preserve)

    def constant_name(self, data, derived=False):
        if not derived:
            self.primary_constants.add(data)
        if data not in self.constants:
            self.constants[data] = '.LCPI%d_%d' % (getattr(self.source, 'machine_order', 0), len(self.constants))
        return self.constants[data]

    def materialize_vector(self, result, data):
        if not any(data):
            name = 'pxor' if self.fn.isa in ('sse2', 'sse42') else 'vpxord' if result.bits == 512 else 'vpxor'
            self.ins(name, [result, result] if name == 'pxor' else [result, result, result],
                     ['W', 'R'] if name == 'pxor' else ['W', 'R', 'R'])
            return
        self.constant_name(data)
        if self.fn.isa == 'avx2' and result.bits == 256:
            for width, name in ((4, 'vpbroadcastd'), (8, 'vpbroadcastq'), (16, 'vbroadcasti128')):
                if data == data[:width] * (32//width):
                    self.ins(name, [result, M.Mem(width, symbol=self.constant_name(data[:width], derived=True))], ['W', 'R'])
                    return
        memory = M.Mem(len(data), symbol=self.constant_name(data))
        name = ('movdqa' if self.fn.isa in ('sse2', 'sse42') else 'vmovdqa64' if result.bits == 512 else 'vmovdqa')
        self.ins(name, [result, memory], ['W', 'R'])

    def operand(self, v, register=False):
        if not register and v in self.folded_vectors:
            op, width, address = self.folded_vectors[v]
            return self.address(address, width)
        op = self.source.defs[v]
        if not register and getattr(self.source, 'word_scans', {}) and op.kind == 'load' and op in self.folded:
            return self.address(op.args[0], op.attrs['width'])
        symbol = self.symbolic(v)
        if symbol is not None and not register:
            return M.Imm(symbol[1], 32, symbol[0])
        if op.kind in ('constant', 'undef'):
            value = op.attrs.get('value', bytes(v.type.size) if v.type.kind == 'vector' else 0)
            if isinstance(value, bytes):
                if self.fn.isa in ('sse2', 'sse42') and all(byte == 255 for byte in value):
                    if getattr(self.source, 'word_scans', {}):
                        key = v.type, value
                        if key not in self.constant_registers:
                            out = self.fn.reg(v.type)
                            self.fn.preferred[out.reg] = M.physical('xmm0')
                            self.hoisted.append(M.Instr('pcmpeqd', (out, out), ('W', 'R')))
                            self.constant_registers[key] = out
                        return self.constant_registers[key]
                    out = self.fn.reg(v.type)
                    self.ins('pcmpeqd', [out, out], ['W', 'R'])
                    return out
                if not any(value):
                    key = v.type, value
                    if key not in self.constant_registers:
                        out = self.fn.reg(v.type)
                        if getattr(self.source, 'word_scans', {}):
                            self.fn.preferred[out.reg] = M.physical('xmm1')
                        name = 'pxor' if self.fn.isa in ('sse2', 'sse42') else 'vpxord' if v.type.bits == 512 else 'vpxor'
                        self.hoisted.append(M.Instr(name, (out, out) if name == 'pxor' else (out, out, out),
                                                    ('W', 'R') if name == 'pxor' else ('W', 'R', 'R')))
                        self.constant_registers[key] = out
                    return self.constant_registers[key]
                self.constant_name(value)
                if not register:
                    return M.Mem(v.type.size, symbol=self.constants[value])
                out = self.fn.reg(v.type)
                if not any(value):
                    name = 'pxor' if self.fn.isa in ('sse2', 'sse42') else 'vpxord' if v.type.bits == 512 else 'vpxor'
                    self.ins(name, [out, out] if name == 'pxor' else [out, out, out], ['W', 'R'] if name == 'pxor' else ['W', 'R', 'R'])
                else:
                    self.materialize_vector(out, value)
                return out
            number = T.convert_integer(value, T.LONG if v.type.bits == 64 else T.INT) if v.type.bits in (32, 64) else value
            if not register:
                return M.Imm(number, v.type.bits)
            cache = v.type.bits == 64 and not -(1 << 31) <= number < 1 << 32
            key = self.block, v.type, number
            if cache and key in self.scalar_constants:
                return self.scalar_constants[key]
            out = self.fn.reg(v.type)
            if v in getattr(self.source, 'scalar_constant_registers', {}):
                self.fn.preferred[out.reg] = M.physical(self.source.scalar_constant_registers[v])
            self.move(out, M.Imm(number, v.type.bits))
            if cache:
                self.scalar_constants[key] = out
            return out
        result = self.value(v)
        if result.reg.bank == 'mask':
            if result.reg not in self.scalar_masks:
                out = self.fn.reg(v.type, result.reg.id)
                self.scalar_masks[result.reg] = result, M.R(out.reg, max(32, result.bits))
            return M.R(self.scalar_masks[result.reg][1].reg, v.type.bits)
        return result

    def mask(self, value, bits):
        key = self.block, value, bits
        if key in self.mask_values:
            return self.mask_values[key]
        original = self.value(value) if self.source.defs[value].kind not in ('constant', 'undef') else None
        if original is not None and original.reg.bank == 'mask':
            return M.R(original.reg, bits)
        source = self.operand(value, True)
        reg = M.Reg('mask', self.fn.next_reg)
        self.fn.next_reg += 1
        self.fn.sizes[reg] = bits
        result = M.R(reg, bits)
        transfer = max(32, bits)
        self.ins('kmov' + ('q' if transfer == 64 else 'd'), [M.R(reg, transfer), M.R(source.reg, transfer)], ['W', 'R'])
        self.mask_values[key] = result
        return result

    def address(self, v, size):
        terms, displacement, symbol, slot = [], 0, None, None

        def term(value, scale=1):
            nonlocal displacement, symbol, slot
            op = self.source.defs[value]
            if op.kind == 'cast' and op.args[0].type.bits == 64:
                term(op.args[0], scale)
            elif op.kind == 'constant' and isinstance(op.attrs['value'], int):
                displacement += op.attrs['value'] * scale
            elif value in getattr(self.source, 'retained_offsets', ()):
                terms.append((self.operand(value, register=True).reg, scale))
            elif (op.kind == 'ptradd' and value != v and value in self.values
                  and self.address_tests.get(self.values[value].reg, set()) & self.dominators.get(self.block, {self.block})):
                terms.append((self.operand(value, register=True).reg, scale))
            elif (op.kind == 'ptradd' and (value in self.loop_dependencies or self.block in self.loop_blocks or self.owners.get(value) in self.loop_blocks)
                  and value != v and self.owners.get(value, self.block) is not self.block
                  and self.source.defs[op.args[1]].kind != 'constant'):
                terms.append((self.operand(value, register=True).reg, scale))
            elif (op.kind == 'ptradd' and (self.fn.unaligned_loops or getattr(self.source, 'copy_loops', False)
                                          or getattr(self.source, 'pipelined_copies', {})
                                          and self.owners.get(value) in self.loop_blocks) and value in self.values
                  and self.owners.get(value, self.block) is not self.block
                  and self.source.defs[op.args[1]].kind == 'constant'):
                terms.append((self.operand(value, register=True).reg, scale))
            elif (op.kind == 'add' and value.type.kind == 'int' and value.type.bits == 64
                  and ((getattr(self.fn, 'returned_parameter', None) is not None or getattr(self.source, 'fixed_loop_exits', ())
                        or getattr(self.source, 'counted_offsets', ()))
                       and self.uses[value] > 1 or getattr(self.source, 'indexed_byte_scans', {})
                       and self.owners.get(value) in self.loop_blocks and self.owners.get(value) is not self.block)):
                terms.append((self.operand(value, register=True).reg, scale))
            elif op.kind == 'ptradd':
                if value != v and value in getattr(self.source, 'word_return_bases', ()):
                    terms.append((self.operand(value, register=True).reg, scale))
                else:
                    term(op.args[0], scale)
                    term(op.args[1], scale)
            elif op.kind == 'add':
                term(op.args[0], scale)
                term(op.args[1], scale)
            elif op.kind == 'sub' and self.source.defs[op.args[1]].kind == 'constant':
                term(op.args[0], scale)
                displacement -= self.source.defs[op.args[1]].attrs['value'] * scale
            elif op.kind == 'mul' and self.source.defs[op.args[1]].kind == 'constant' and self.source.defs[op.args[1]].attrs['value'] in (1, 2, 4, 8):
                term(op.args[0], scale * self.source.defs[op.args[1]].attrs['value'])
            elif op.kind == 'symbol' and symbol is None and scale == 1:
                symbol = op.attrs['name']
            elif value in self.stack and slot is None and scale == 1:
                slot = self.stack[value]
            else:
                terms.append((self.operand(value, register=True).reg, scale))

        term(v)
        def fallback():
            value = v
            op = self.source.defs[value]
            while op.kind == 'cast' and op.args[0].type.bits == 64:
                value = op.args[0]
                op = self.source.defs[value]
            if op.kind == 'ptradd':
                base = self.operand(op.args[0], True).reg
                other = self.source.defs[op.args[1]]
                if other.kind == 'constant' and -(1 << 31) <= other.attrs['value'] < 1 << 31:
                    return M.Mem(size, base=base, disp=other.attrs['value'])
                return M.Mem(size, base=base, index=self.operand(op.args[1], True).reg)
            return M.Mem(size, base=self.value(v).reg)
        if symbol is not None and terms or len(terms) > (1 if slot is not None else 2) or any(s not in (1, 2, 4, 8) for _, s in terms) or not -(1 << 31) <= displacement < 1 << 31:
            return fallback()
        base = next((r for r, s in terms if s == 1), None) if slot is None else None
        if base is not None:
            terms.remove((base, 1))
        if len(terms) > 1:
            return fallback()
        index, scale = terms[0] if terms else (None, 1)
        source = self.definition(v)
        if (source.kind == 'ptradd' and source.args[0] in getattr(self.source, 'word_return_bases', ())
                and base is not None and index is not None and scale == 1):
            base, index = index, base
        return M.Mem(size, base, index, scale, displacement, symbol, slot)

    def compare(self, op, invert=False):
        if op in self.mask_unions:
            x, d = self.mask_unions[op]
            masks = []
            for name, value in (('vptestmb', d), ('vptestnmb', x)):
                reg = M.Reg('mask', self.fn.next_reg)
                self.fn.next_reg += 1
                result = M.R(reg, op.args[0].type.bits)
                self.fn.sizes[reg] = result.bits
                vector = self.operand(value, True)
                self.ins(name, [result, vector, vector], ['W', 'R', 'R'])
                masks.append(result)
            self.ins('kortest' + {8: 'b', 16: 'w', 32: 'd', 64: 'q'}[masks[0].bits], masks[::-1], ['R', 'R'])
            cc = 'e' if op.kind == 'eq' else 'ne'
            return INVERSE[cc] if invert else cc
        if op in self.compare_tests:
            word, mask = self.compare_tests[op]
            value = self.operand(word, True)
            value = M.R(value.reg, 8 if mask < 256 else 32)
            self.ins('test', [value, M.Imm(mask)], ['R', 'R'])
            cc = 'e' if op.kind == 'eq' else 'ne'
            return INVERSE[cc] if invert else cc
        a, b = op.args
        constant = self.source.defs[b]
        if (op.kind in ('eq', 'ne') and a in self.predicate_values and constant.kind == 'constant'
                and constant.attrs['value'] == (1 << a.type.bits) - 1):
            mask = self.value(a)
            self.ins('kortest' + {8: 'b', 16: 'w', 32: 'd', 64: 'q'}[mask.bits], [mask, mask], ['R', 'R'])
            cc = 'b' if op.kind == 'eq' else 'ae'
            return INVERSE[cc] if invert else cc
        count_compare = None
        swapped = False
        left, right = self.definition(a), self.definition(b)
        lhs = self.address(left.args[0], left.attrs['width']) if left in self.folded and left.kind == 'load' else self.operand(a, True)
        rhs = self.address(right.args[0], right.attrs['width']) if right in self.folded and right.kind == 'load' else self.operand(b)
        if op in getattr(self.source, 'narrow_word_comparisons', ()) and isinstance(lhs, M.R) and isinstance(rhs, M.R):
            lhs, rhs = M.R(lhs.reg, 32), M.R(rhs.reg, 32)
        if (op.kind in ('eq', 'ne') and isinstance(lhs, M.R) and isinstance(rhs, M.R)
                and left.kind == right.kind == 'cast' and left.args[0].type == right.args[0].type
                and left.args[0].type.bits in (8, 16)
                and all(self.source.defs[item.args[0]].kind == 'load' for item in (left, right))):
            lhs, rhs = M.R(lhs.reg, left.args[0].type.bits), M.R(rhs.reg, right.args[0].type.bits)
        if (a.type.kind == b.type.kind == 'ptr' and right.kind == 'param'
                and getattr(self.fn, 'returned_parameter', None) is not None):
            lhs, rhs, swapped = rhs, lhs, True
        if (isinstance(lhs, M.R) and isinstance(rhs, M.R) and lhs.bits == rhs.bits == 64
                and self.bit_count(a) and possible_bits(self.source, b) < 1 << 31):
            lhs, rhs = M.R(rhs.reg, 32), M.R(lhs.reg, 32)
            swapped = True
        elif (isinstance(lhs, M.R) and isinstance(rhs, M.R) and lhs.bits == rhs.bits == 64
                and (self.bit_count_width(a) == 32
                     or self.block in {self.blocks[b] for b in getattr(self.source, 'guarded_pointer_returns', ())})):
            lhs, rhs, swapped = rhs, lhs, True
        if isinstance(rhs, M.Imm) and a.type.bits == 64 and not -(1 << 31) <= rhs.value < 1 << 31:
            rhs = self.operand(b, True)
        if isinstance(rhs, M.Imm):
            rhs = replace(rhs, bits=32)
            definition = self.definition(a)
            page_guard = (definition.kind == 'and' and self.definition(definition.args[0]).kind == 'or'
                          and self.definition(definition.args[1]).kind == 'constant'
                          and self.definition(definition.args[1]).attrs['value'] == 4095 and self.fn.loop_headers)
            if page_guard and isinstance(lhs, M.R):
                lhs = M.R(lhs.reg, 32)
                if op.kind == 'le':
                    if self.block in self.loop_blocks:
                        count_compare = 'be'
                    else:
                        rhs, count_compare = replace(rhs, value=rhs.value+1), 'b'
            if (definition.kind == 'phi' and isinstance(lhs, M.R) and lhs.bits == 64
                    and all(self.definition(v).kind == 'and' and self.definition(self.definition(v).args[1]).kind == 'constant'
                            and self.definition(self.definition(v).args[1]).attrs['value'] == 4095 for v in definition.args)):
                lhs = M.R(lhs.reg, 32)
                if op.kind == 'le':
                    count_compare = 'be'
            if (isinstance(lhs, M.R) and lhs.bits == 64 and definition.kind == 'and' and 0 <= rhs.value < 1 << 32
                    and (self.uses[a] == self.uses[definition.results[0]] == 1
                         or any(user is not op and user.kind in ('eq', 'ne', 'lt', 'le', 'gt', 'ge')
                                and self.owners.get(user.results[0]) in self.dominators[self.block] - {self.block}
                                for user in self.users.get(a, ())))):
                mask = self.source.defs[definition.args[1]]
                if mask.kind == 'constant' and 0 <= mask.attrs['value'] < 1 << 32:
                    lhs = M.R(lhs.reg, 32)
            if isinstance(lhs, M.R) and self.bit_count(a) and 0 < rhs.value <= 64:
                lhs = M.R(lhs.reg, 32)
                if op.kind == 'lt' and not op.attrs.get('biased_count'):
                    rhs, count_compare = replace(rhs, value=rhs.value - 1), 'be'
                elif op.kind == 'ge' and not op.attrs.get('biased_count'):
                    rhs, count_compare = replace(rhs, value=rhs.value - 1), 'a'
        if isinstance(lhs, M.R) and isinstance(rhs, M.Imm) and rhs.value == 0 and op.kind in ('eq', 'ne'):
            self.ins('test', [lhs, lhs], ['R', 'R'])
        else:
            self.ins('cmp', [lhs, rhs], ['R', 'R'], keep_immediate=count_compare is not None)
        signed = a.type.signed
        cc = count_compare or {'eq': 'e', 'ne': 'ne', 'lt': 'l' if signed else 'b', 'le': 'le' if signed else 'be',
              'gt': 'g' if signed else 'a', 'ge': 'ge' if signed else 'ae'}[op.kind]
        if swapped:
            cc = {'b': 'a', 'a': 'b', 'be': 'ae', 'ae': 'be', 'l': 'g', 'g': 'l', 'le': 'ge', 'ge': 'le'}.get(cc, cc)
        return INVERSE[cc] if invert else cc

    def condition(self, value):
        op = self.definition(value)
        if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'testz':
            self.ins('vptest', [self.operand(v, True) for v in op.args], ['R', 'R'])
            return 'e'
        if op.kind == 'load' and op in self.folded:
            self.ins('cmp', [self.address(op.args[0], op.attrs['width']), M.Imm(0, 8)], ['R', 'R'])
            return 'ne'
        if self.value(value).reg.bank == 'mask':
            mask = self.value(value)
            ins = self.ins('masktest', [mask, mask], ['R', 'R'])
            self.mask_tests.append((self.block, ins, mask))
            return 'ne'
        if op.kind in ('eq', 'ne', 'lt', 'gt', 'le', 'ge'):
            key = self.block, op
            previous = self.condition_flags.get(key)
            flags = M.Reg('flags', 'rflags')
            if (getattr(self.source, 'large_byte_compares', False) and previous is not None
                    and not any(flags in ins.defs for ins in self.block.ins[previous[0]:])):
                return previous[1]
            cc = self.compare(op)
            self.condition_flags[key] = len(self.block.ins), cc
            return cc
        if op.kind == 'and':
            if op.attrs.get('boolean_test'):
                a, b = (self.operand(v, True) for v in op.args)
                self.ins('test', [M.R(a.reg, 8), M.R(b.reg, 8)], ['R', 'R'])
                return 'ne'
            if (getattr(self.source, 'indexed_byte_scans', {}) and op in self.folded
                    and all(self.definition(v).kind != 'constant' and possible_bits(self.source, v) <= 255 for v in op.args)):
                a, c = (self.operand(v, True) for v in op.args)
                self.ins('test', [M.R(a.reg, 8), M.R(c.reg, 8)], ['R', 'R'])
                return 'ne'
            if op not in self.folded and op.results[0] not in self.booleans and self.block.ins:
                last = self.block.ins[-1]
                if (last.mnemonic == 'and' and isinstance(last.operands[0], M.R)
                        and last.operands[0].reg == self.value(value).reg):
                    return 'ne'
            if all(v in self.booleans for v in op.args):
                operand = self.operand(value, True)
                self.ins('cmp', [operand, M.Imm(1, 8)], ['R', 'R'])
                return 'e'
            mask = self.source.defs[op.args[1]]
            if mask.kind == 'constant' and 0 < mask.attrs['value'] < 1 << 32:
                n = mask.attrs['value']
                if n == 255 and getattr(self.fn, 'returned_parameter', None) is not None:
                    lhs = self.operand(op.args[0], True)
                    lhs = M.R(lhs.reg, 8)
                    self.ins('test', [lhs, lhs], ['R', 'R'])
                    return 'ne'
                n &= -self.alignments.get(op.args[0], 1)
                lhs = self.operand(op.args[0], True)
                if (n == 1 and lhs.bits == 32 and self.definition(op.args[0]).kind == 'load'
                        and self.uses[op.args[0]] > 1 and getattr(self.source, 'shared_return', None) is not None):
                    scratch = self.fn.reg(T.UINT)
                    self.move(scratch, lhs)
                    self.ins('and', [scratch, M.Imm(1, 32)], ['RW', 'R'])
                    return 'ne'
                if n < 256:
                    lhs = M.R(lhs.reg, 8)
                elif lhs.bits > 32:
                    lhs = M.R(lhs.reg, 32)
                other = self.definition(op.args[0])
                if n & (n - 1) == 0 and other.kind == 'not':
                    source = self.operand(other.args[0], True)
                    source = M.R(source.reg, lhs.bits)
                    test = M.Instr('test', (source, M.Imm(n)), ('R', 'R'))
                    flags = M.Reg('flags', 'rflags')
                    at = next((i for i, ins in enumerate(self.block.ins) if ins.mnemonic == 'not' and ins.operands[0].reg == lhs.reg), None)
                    if at is not None and self.uses[other.args[0]] == 1:
                        if at and self.block.ins[at - 1].mnemonic == 'mov' and self.block.ins[at - 1].operands[0].reg == lhs.reg:
                            at -= 1
                        if not any(flags in ins.defs for ins in self.block.ins[at:]):
                            self.block.ins.insert(at, test)
                            return 'e'
                    self.block.ins.append(test)
                    return 'e'
                self.ins('test', [lhs, M.Imm(n)], ['R', 'R'])
                return 'ne'
        operand = self.operand(value, True)
        if getattr(self.source, 'word_scans', {}) and op.kind == 'load' and value.type.bits == 8:
            operand = M.R(operand.reg, 32)
        if getattr(self.source, 'indexed_byte_scans', {}) and possible_bits(self.source, value) <= 255:
            operand = M.R(operand.reg, 8)
        if (self.block.ins and self.block.ins[-1].mnemonic in ('add', 'sub', 'inc', 'dec', 'and', 'or', 'xor')
                and self.block.ins[-1].operands[0] == operand):
            return 'ne'
        self.ins('test', [operand, operand], ['R', 'R'])
        return 'ne'

    def call(self, op, tail=False):
        gp = vp = 0
        argument_regs = set()
        copies = []
        for v in op.args[1:]:
            if v.type.kind == 'vector':
                destination = M.R(M.physical('xmm%d' % vp), v.type.bits)
                vp += 1
            else:
                if gp == len(ARGS):
                    raise ValueError(self.fn.name + ': stack arguments not implemented yet')
                destination = M.R(M.physical(ARGS[gp]), max(32, v.type.bits))
                gp += 1
            if tail and v.type.kind == 'ptr':
                pointer = self.definition(v)
                if pointer.kind == 'ptradd' and self.definition(pointer.args[1]).kind == 'constant':
                    base = self.definition(pointer.args[0])
                    if (base.kind == 'and' and self.definition(base.args[1]).kind == 'constant'
                            and self.definition(base.args[1]).attrs['value'] & (1 << 63)):
                        self.fn.preferred.setdefault(self.value(base.results[0]).reg, destination.reg)
            source = self.operand(v)
            if isinstance(source, M.R) and source.bits < destination.bits:
                definition = self.definition(v)
                known_extension = (definition.kind == 'param' and definition.results[0].type == v.type
                                   or definition.kind == 'load' and not v.type.signed)
                if v.type.kind == 'int' and v.type.bits in (8, 16) and not known_extension:
                    extended = self.fn.reg(T.INT if v.type.signed else T.UINT)
                    name = ('movsb' if v.type.bits == 8 else 'movsw') if v.type.signed else ('movzb' if v.type.bits == 8 else 'movzw')
                    self.ins(name, [extended, source], ['W', 'R'])
                    source = extended
                else:
                    source = M.R(source.reg, destination.bits)
            copies.append((destination, source))
            argument_regs.add(destination.reg)
        if copies:
            self.ins('parallel', copies=tuple(copies))
        target = self.definition(op.args[0])
        if target.kind == 'symbol':
            name = target.attrs['name']
            if name in self.trampolines:
                table, offset = self.trampolines[name]
                operand = M.Mem(8, disp=offset, symbol=table)
            else:
                operand = M.Target(name)
        elif target in self.folded:
            operand = self.address(target.args[0], 8)
        else:
            operand = self.operand(op.args[0], True)
        clobbers = {M.physical(r) for r in CALLER_SAVED}
        clobbers |= {M.physical('zmm%d' % n) for n in range(32)}
        clobbers |= {M.physical('k%d' % n) for n in range(8)}
        self.ins('jmp' if tail else 'call', [operand], ['R'], implicit_uses=frozenset(argument_regs),
                 implicit_defs=frozenset() if tail else frozenset(clobbers),
                 callee=target.attrs['name'] if target.kind == 'symbol' else None)
        self.scalar_constants = {key: value for key, value in self.scalar_constants.items() if key[0] is not self.block}
        if op.results and not tail:
            result = op.results[0]
            source = M.R(M.physical('xmm0' if result.type.kind == 'vector' else 'rax'), result.type.bits)
            self.move(self.value(result), source)

    def intrinsic(self, op):
        b, args = op.attrs['builtin'], op.args
        result = self.value(op.results[0]) if op.results else None
        vector = next((v.type for v in (*op.results, *args) if v.type.kind == 'vector'), None)
        vex = self.fn.isa not in ('sse2', 'sse42')
        if 'memory_broadcast' in op.attrs:
            self.ins('vpbroadcastb', [result, self.address(op.attrs['memory_broadcast'], 1)], ['W', 'R'])
        elif b.op == 'load':
            address = self.address(args[0], b.width)
            if b.width == 64 and (b.alignment == 64 or self.definition(args[0]).kind == 'param'):
                self.ins('vmovdqa64' if b.alignment == 64 else 'vmovdqu64', [result, address], ['W', 'R'])
            elif b.alignment >= b.width:
                self.ins(('v' if vex else '') + 'movdqa', [result, address], ['W', 'R'])
            else:
                self.move(result, address)
            if op.results[0] in getattr(self.source, 'floating_copies', ()):
                self.block.ins[-1].domain = 'fp'
        elif b.op == 'masked_load':
            mask = self.mask(args[0], b.width)
            self.ins('vmovdqu8', [result, self.address(args[1], b.width)], ['W', 'R'], mask=mask.reg, zeroing=True)
        elif b.op == 'masked_store':
            mask = self.mask(args[1], b.width)
            self.ins('vmovdqu8', [self.address(args[0], b.width), self.operand(args[2], True)], ['W', 'R'], mask=mask.reg)
        elif b.op in ('store', 'store_low'):
            source = self.operand(args[1], True)
            if b.op == 'store_low':
                self.ins(('v' if vex else '') + ('movq' if b.width == 8 else 'movd'), [self.address(args[0], b.width), source], ['W', 'R'])
            else:
                address = self.address(args[0], b.width)
                value_op = self.definition(args[1])
                integer_value = value_op.kind == 'intrinsic' and value_op.attrs['builtin'].op not in ('load', 'masked_load', 'widen_undef', 'truncate')
                if b.width == 64 and (b.alignment == 64 or self.definition(args[0]).kind == 'param'
                                     or integer_value):
                    self.ins('vmovdqa64' if b.alignment == 64 else 'vmovdqu64', [address, source], ['W', 'R'])
                elif integer_value:
                    name = ('v' if vex else '') + ('movdqa' if b.alignment >= b.width else 'movdqu')
                    self.ins(name, [address, source], ['W', 'R'])
                elif b.alignment >= b.width:
                    self.ins(('v' if vex else '') + 'movaps', [address, source], ['W', 'R'])
                else:
                    self.move(address, source)
                if args[1] in getattr(self.source, 'floating_copies', ()):
                    self.block.ins[-1].domain = 'fp'
        elif b.name.endswith('_mask'):
            if op in self.mask_intersections:
                x, a, c = self.mask_intersections[op]
                mask = M.Reg('mask', self.fn.next_reg)
                self.fn.next_reg += 1
                self.fn.sizes[mask] = result.bits
                name = 'vpcmpeqb' if b.op == 'cmpeq_epi8_masked' else 'vptestnmb'
                self.ins(name, [M.R(mask, result.bits), self.operand(a, True), self.operand(c)], ['W', 'R', 'R'])
                vector = self.operand(x, True)
                self.ins('vptestmb', [result, vector, vector], ['W', 'R', 'R'], mask=mask)
                return
            args = self.mask_args.get(op, args)
            masked = '_mask_' in b.name
            mask = self.mask(args[0], result.bits).reg if masked else None
            inputs = args[1:] if masked else args
            a, c = self.operand(inputs[0], True), self.operand(inputs[1])
            if b.op.startswith('cmpneq_epu8'):
                self.ins('vpcmpb', [result, a, c, M.Imm(4, 8)], ['W', 'R', 'R', 'R'], mask=mask)
            else:
                name = 'vptestnmb' if op in self.zero_mask_compares else 'vpcmpeqb' if b.op.startswith('cmpeq_epi8') else 'vptestnmb' if b.op.startswith('testn_epi8') else 'vptestmb'
                self.ins(name, [result, a, c], ['W', 'R', 'R'], mask=mask)
        elif b.op == 'ternary32':
            last = self.definition(args[2])
            immediate = self.source.defs[args[3]]
            if (immediate.kind == 'constant' and immediate.attrs['value'] == 0xfe
                    and last.kind == 'intrinsic' and last.attrs['builtin'].op == 'ternary32'
                    and self.source.defs[last.args[3]].kind == 'constant'
                    and self.source.defs[last.args[3]].attrs['value'] == 0xfe):
                args = args[2], args[1], args[0], args[3]
            self.ins('vmovdqa64' if result.bits == 512 else 'vmovdqa', [result, self.operand(args[0], True)], ['W', 'R'])
            self.ins('vpternlogd', [result, self.operand(args[1], True), self.operand(args[2]), replace(self.operand(args[3]), bits=8)], ['RW', 'R', 'R', 'R'])
        elif b.op in ('and', 'andnot', 'or', 'xor', 'cmpeq_epi8', 'cmpgt_epi8', 'min_epu8', 'add_epi8', 'sub_epi8', 'shuffle_epi8'):
            args = self.binary_args.get(op, args)
            name = {'and': 'pand', 'andnot': 'pandn', 'or': 'por', 'xor': 'pxor', 'cmpeq_epi8': 'pcmpeqb', 'cmpgt_epi8': 'pcmpgtb',
                    'min_epu8': 'pminub', 'add_epi8': 'paddb', 'sub_epi8': 'psubb', 'shuffle_epi8': 'pshufb'}[b.op]
            first = self.source.defs[args[0]]
            zero = not vex and b.op == 'cmpeq_epi8' and first.kind == 'constant' and isinstance(first.attrs['value'], bytes) and not any(first.attrs['value'])
            if zero:
                if (self.block in self.loop_blocks or getattr(self.source, 'preincrement_scans', {})
                        or getattr(self.source, 'pipelined_copies', {})):
                    constant = self.operand(args[0], True)
                    loaded = self.operand(args[1])
                    name = 'movdqa' if isinstance(loaded, M.Mem) else 'movaps'
                    self.ins(name, [result, loaded], ['W', 'R'])
                    self.ins('pcmpeqb', [result, constant], ['RW', 'R'])
                    return
                self.ins('pxor', [result, result], ['W', 'R'])
                self.ins(name, [result, self.operand(args[1])], ['RW', 'R'])
                return
            a, c = self.operand(args[0], True), self.operand(args[1])
            if vex:
                self.ins('vpxorq' if result.bits == 512 and b.op == 'xor' else 'v' + name, [result, a, c], ['W', 'R', 'R'])
            else:
                self.move(result, a)
                self.ins(name, [result, c], ['RW', 'R'])
        elif b.op == 'movemask8':
            if op in self.vector_mask_compares:
                source = self.operand(self.vector_mask_compares[op][0], True)
                self.ins('vptestnmb', [result, source, source], ['W', 'R', 'R'])
            else:
                self.ins('vpmovmskb' if vex else 'pmovmskb', [M.R(result.reg, 32), self.operand(args[0], True)], ['W', 'R'])
        elif b.op == 'lshr16':
            if vex:
                self.ins('vpsrlw', [result, self.operand(args[0], True), self.operand(args[1])], ['W', 'R', 'R'])
            else:
                self.move(result, self.operand(args[0], True))
                self.ins('psrlw', [result, self.operand(args[1])], ['RW', 'R'])
        elif b.op == 'testz':
            self.ins('vptest', [self.operand(v, True) for v in args], ['R', 'R'])
            self.ins('sete', [M.R(result.reg, 8)], ['W'], cc='e')
            if result.bits != 8:
                self.ins('movzb', [M.R(result.reg, 32), M.R(result.reg, 8)], ['W', 'R'])
        elif b.op in ('ctz_nonzero', 'tzcnt'):
            bits = max(32, args[0].type.bits)
            self.ins('tzcnt', [M.R(result.reg, bits), M.R(self.operand(args[0], True).reg, bits)], ['W', 'R'])
        elif b.op.startswith('cmpistr'):
            mode = self.operand(args[2])
            inputs = [self.operand(args[0], True), self.operand(args[1]), replace(mode, bits=8)]
            masked = b.op == 'cmpistrm'
            name = ('v' if vex else '') + ('pcmpistrm' if masked else 'pcmpistri')
            fixed = M.physical('xmm0' if masked else 'rcx')
            self.ins(name, inputs, ['R', 'R', 'R'], implicit_defs=frozenset({fixed, M.Reg('flags', 'rflags')}))
            if b.op in ('cmpistri', 'cmpistrm'):
                self.move(result, M.R(fixed, 128 if masked else 32))
            else:
                cc = {'cmpistra': 'a', 'cmpistrc': 'b', 'cmpistrz': 'e'}[b.op]
                self.ins('set' + cc, [M.R(result.reg, 8)], ['W'], cc=cc)
                if result.bits != 8:
                    self.ins('movzb', [M.R(result.reg, 32), M.R(result.reg, 8)], ['W', 'R'])
        elif b.op in ('extract_low32', 'insert_low32'):
            self.ins('vmovd' if vex else 'movd', [result, self.operand(args[0], True)], ['W', 'R'])
        elif b.op in ('widen_undef', 'truncate'):
            return
        elif b.op == 'broadcast128':
            source = self.operand(args[0], True)
            self.ins('vinserti128', [result, M.R(source.reg, 256), source, M.Imm(1, 8)], ['W', 'R', 'R', 'R'])
        elif b.op == 'insert128':
            self.ins('vinserti128', [result, self.operand(args[0], True), self.operand(args[1], True), replace(self.operand(args[2]), bits=8)], ['W', 'R', 'R', 'R'])
        elif b.op == 'insert64':
            source, word = self.operand(args[0], True), self.operand(args[1], True)
            lane = replace(self.operand(args[2]), bits=8)
            if vex:
                self.ins('vpinsrq', [result, source, word, lane], ['W', 'R', 'R', 'R'])
            else:
                self.move(result, source)
                self.ins('pinsrq', [result, word, lane], ['RW', 'R', 'R'])
        elif b.op == 'extract128':
            self.ins('vextracti128', [result, self.operand(args[0], True), replace(self.operand(args[1]), bits=8)], ['W', 'R', 'R'])
        elif b.op == 'constant64_reversed':
            low, high = self.operand(args[1], True), self.operand(args[0], True)
            self.ins('vmovq' if vex else 'movq', [result, low], ['W', 'R'])
            if vex:
                self.ins('vpinsrq', [result, result, high, M.Imm(1, 8)], ['W', 'R', 'R', 'R'])
            else:
                tmp = self.fn.reg(T.V128)
                self.ins('movq', [tmp, high], ['W', 'R'])
                self.ins('punpcklqdq', [result, tmp], ['RW', 'R'])
        elif b.op == 'bzhi':
            source, count = self.operand(args[0], True), self.operand(args[1], True)
            self.ins('bzhi', [result, source, M.R(count.reg, result.bits)], ['W', 'R', 'R'])
        elif b.op == 'splat8':
            if self.fn.isa in ('sse2', 'sse42') and getattr(self.source, 'preincrement_scans', {}):
                self.fn.preferred.setdefault(result.reg, M.physical('xmm0'))
            value = args[0]
            source = self.source.defs[value]
            while source.kind == 'cast' and source.args[0].type.bits >= 8:
                value = source.args[0]
                source = self.source.defs[value]
            widen = (not vex and source.kind == 'load' and source.results[0].type.bits == 8
                     and self.owners.get(value) is not self.block)
            source = self.operand(value, True)
            if widen:
                extended = self.fn.reg(T.UINT)
                self.ins('movzb', [extended, M.R(source.reg, 8)], ['W', 'R'])
                source = extended
            if self.fn.isa in ('evex', 'evex512'):
                self.ins('vpbroadcastb', [result, M.R(source.reg, 32)], ['W', 'R'])
                return
            tmp = self.fn.reg(T.V128)
            self.ins('vmovd' if vex else 'movd', [tmp, M.R(source.reg, 32)], ['W', 'R'])
            if vex:
                self.ins('vpbroadcastb', [result, tmp], ['W', 'R'])
            else:
                self.move(result, tmp)
                self.ins('punpcklbw', [result, result], ['RW', 'R'])
                self.ins('pshuflw', [result, result, M.Imm(0, 8)], ['W', 'R', 'R'])
                self.ins('pshufd', [result, result, M.Imm(0x44, 8)], ['W', 'R', 'R'])
        elif b.op in ('expect', 'unpredictable'):
            self.move(result, self.operand(args[0]))
        else:
            raise ValueError(self.fn.name + ': intrinsic selection not implemented: ' + b.name)

    def operation(self, op):
        if op in self.folded or op.attrs.get('selection_folded'):
            return
        args = op.args
        result = self.value(op.results[0]) if op.results else None
        kind = op.kind
        if op.attrs.get('byte_shift'):
            self.move(M.R(result.reg, 32), self.operand(args[0], True))
            self.ins('shr', [M.R(result.reg, 8), self.operand(args[1])], ['RW', 'R'])
            return
        if 'scaled_shift_count' in op.attrs:
            source = self.operand(op.attrs['scaled_shift_count'], True)
            self.ins('lea', [result, M.Mem(4, index=source.reg, scale=8)], ['W', 'A'])
            return
        if 'scaled_sum' in op.attrs:
            a, b, scale = op.attrs['scaled_sum']
            a, b = self.operand(a, True), self.operand(b, True)
            self.ins('lea', [result, M.Mem(result.bits//8, base=a.reg, index=b.reg, scale=scale)], ['W', 'A'])
            return
        if 'rmw_or' in op.attrs:
            self.ins('or', [self.address(args[0], op.attrs['width']), self.operand(op.attrs['rmw_or'], True)], ['RW', 'R'])
            return
        if kind == 'phi' or op in self.folded:
            return
        if op in self.address_sums:
            address = self.address(op.results[0], 8)
            if (address.base is not None and address.index is not None and address.scale == 1
                    and self.definition(op.args[1]).kind != 'neg'):
                address = replace(address, base=address.index, index=address.base)
            self.ins('lea', [result, address], ['W', 'A'])
            return
        if op in self.mask_packs:
            low, high = self.mask_packs[op]
            self.ins('kunpckdq', [result, self.value(high), self.value(low)], ['W', 'R', 'R'])
            return
        if op in self.low_bit_masks:
            self.ins('blsmsk', [result, self.operand(self.low_bit_masks[op], True)], ['W', 'R'])
            return
        if op in self.byte_counts:
            source = self.operand(args[0], True)
            self.move(M.R(result.reg, 32), M.R(source.reg, 32))
            self.ins('and', [M.R(result.reg, 8), M.Imm(self.source.defs[args[1]].attrs['value'], 8)], ['RW', 'R'])
            return
        if op in self.narrow_counts:
            return
        if op in self.bitsets:
            base, count, extra = self.bitsets[op]
            self.move(result, self.operand(base, True))
            count_op = self.source.defs[count]
            if count_op in self.narrow_counts:
                source = self.operand(count_op.args[0], True)
                out = M.R(self.value(count).reg, 8)
                self.move(out, M.R(source.reg, 8))
                self.ins('dec', [out], ['RW'])
            index = self.operand(count, True)
            self.ins('bts', [result, M.R(index.reg, result.bits)], ['RW', 'R'])
            if extra is not None:
                self.ins('or', [result, self.operand(extra)], ['RW', 'R'])
            return
        if op in self.narrow_shifts:
            value, offset = self.narrow_shifts[op]
            source = self.operand(value, True)
            self.ins('lea', [M.R(result.reg, 32), M.Mem(4, base=source.reg, disp=offset)], ['W', 'A'])
            return
        if kind == 'scalar_constant':
            self.move(result, self.operand(args[0]))
        elif kind == 'vector_constant':
            data = self.source.defs[args[0]].attrs['value']
            if op.attrs.get('splat64'):
                self.ins('vpbroadcastq', [result, M.Mem(8, symbol=self.constant_name(data[:8]))], ['W', 'R'])
            else:
                self.materialize_vector(result, data)
        elif kind == 'cast':
            if self.zero_extension(op):
                return
            source = self.operand(args[0], True)
            a, c = args[0].type, op.results[0].type
            if source.bits < min(a.bits, c.bits):
                self.ins('movzb', [M.R(result.reg, 32), source], ['W', 'R'])
                return
            if a.bits == c.bits:
                if source.bits < result.bits:
                    self.ins('movzb', [M.R(result.reg, 32), source], ['W', 'R'])
                return
            if a.kind == c.kind == 'vector':
                self.values[op.results[0]] = M.R(source.reg, c.bits)
            elif a.bits < c.bits and a.bits < 32:
                name = ('movsb' if a.bits == 8 else 'movsw') if a.signed else ('movzb' if a.bits == 8 else 'movzw')
                self.ins(name, [M.R(result.reg, max(32, c.bits) if a.signed else 32), source], ['W', 'R'])
            elif a.bits == 32 and c.bits == 64:
                if a.signed:
                    self.ins('movslq', [result, source], ['W', 'R'])
                else:
                    self.move(M.R(result.reg, 32), source, preserve=True)
            else:
                self.move(result, M.R(source.reg, c.bits))
        elif kind in ('symbol', 'string'):
            if kind == 'string':
                data = op.attrs['data']
                self.constants.setdefault(data, '.LC.%s.%d' % (self.fn.name, len(self.constants)))
                symbol = self.constants[data]
            else:
                symbol = op.attrs['name']
            self.move(result, M.Imm(0, 32, symbol))
        elif kind == 'ptradd':
            offset = self.definition(args[1])
            if getattr(self.source, 'shared_load_return', None) is not None and offset.kind == 'sub':
                total, low = self.definition(offset.args[0]), self.definition(offset.args[1])
                if (total.kind == 'constant' and low.kind == 'and'
                        and self.definition(low.args[0]) is self.definition(args[0])):
                    self.move(result, self.operand(args[0], True))
                    negative = self.fn.reg(T.LONG)
                    self.move(negative, self.operand(offset.args[1], True))
                    self.ins('neg', [negative], ['RW'])
                    self.ins('add', [result, negative], ['RW', 'R'])
                    self.ins('add', [result, self.operand(offset.args[0])], ['RW', 'R'], keep_immediate=True)
                    return
            symbol = self.symbolic(op.results[0])
            if symbol is not None:
                self.move(result, M.Imm(symbol[1], 32, symbol[0]))
            else:
                self.ins('lea', [result, self.address(op.results[0], 8)], ['W', 'A'])
        elif kind == 'load':
            mem = self.address(args[0], op.attrs['width'])
            if result.bits in (8, 16):
                self.ins('movzb' if result.bits == 8 else 'movzw', [M.R(result.reg, 32), mem], ['W', 'R'], preserve=op.attrs.get('volatile', False))
            else:
                self.move(result, mem, preserve=op.attrs.get('volatile', False))
        elif kind == 'store':
            value = self.operand(args[1])
            if isinstance(value, M.Imm) and op.attrs['width'] == 8 and not -(1 << 31) <= value.value < 1 << 31:
                value = self.operand(args[1], True)
            self.move(self.address(args[0], op.attrs['width']), value)
        elif kind in ('add', 'sub', 'mul', 'and', 'or', 'xor'):
            if (getattr(self.source, 'indexed_byte_scans', {}) and kind in ('and', 'or', 'xor')
                    and op.results[0].type.bits == 32
                    and self.register_uses(args[0]) > self.register_uses(args[1])):
                args = args[::-1]
            if (kind == 'and' and self.fn.loop_headers and self.definition(args[0]).kind == 'or'
                    and self.definition(args[1]).kind == 'constant' and self.definition(args[1]).attrs['value'] == 4095
                    and self.uses[op.results[0]] > 1):
                self.fn.preferred.setdefault(result.reg, M.physical('rcx'))
            if op.attrs.get('page_guard'):
                args = args[::-1]
            users = self.users.get(op.results[0], ())
            narrow_logic = (kind in ('or', 'xor') and result.bits == 64 and users
                            and self.uses[op.results[0]] == len(users)
                            and all(user.kind == 'and' and user.args[0] == op.results[0]
                                    and self.definition(user.args[1]).kind == 'constant'
                                    and 0 <= self.definition(user.args[1]).attrs['value'] < 1 << 32 for user in users))
            if narrow_logic and not self.fn.unaligned_loops and all(self.definition(v).kind == 'param' for v in args):
                args = tuple(sorted(args, key=lambda v: self.definition(v).attrs['index'], reverse=True))
            if (kind == 'add' and (self.bit_count_width(args[1]) == 64
                                  or (getattr(self.source, 'fixed_loop_exits', ()) or getattr(self.source, 'distance_returns', False))
                                  and self.bit_count(args[1])) and not self.bit_count(args[0])
                    and self.register_uses(args[1]) <= self.register_uses(args[0])):
                args = args[::-1]
            if (kind == 'add' and self.bit_count(args[0]) and self.definition(args[1]).kind == 'param'
                    and self.block in self.direct_return_blocks and not getattr(self.source, 'byte_compares', False)):
                args = args[::-1]
            if kind == 'or' and getattr(self.source, 'return_joins', ()) and self.definition(args[1]).kind == 'shl':
                left = args[0]
                while self.source.defs[left].kind == 'cast':
                    left = self.source.defs[left].args[0]
                if left in self.predicate_values and left.type.bits == 32:
                    args = args[::-1]
            if (kind in ('add', 'mul', 'and', 'or', 'xor') and getattr(self.source, 'shared_return', None) is not None
                    and self.definition(args[1]).kind != 'constant'
                    and (self.uses[args[0]] > self.uses[args[1]]
                         or self.definition(args[0]).kind == self.definition(args[1]).kind == 'and'
                         and all(self.definition(self.definition(v).args[0]).kind == 'shr' for v in args)
                         or self.definition(args[1]).kind in ('eq', 'ne', 'lt', 'le', 'gt', 'ge')
                         and any(user.kind == 'cast' for user in self.users.get(op.results[0], ())))):
                args = args[::-1]
            a, c = self.operand(args[0], True), self.operand(args[1])
            if op in getattr(self.source, 'narrow_word_values', ()):
                result, a, c = M.R(result.reg, 32), M.R(a.reg, 32), replace(c, bits=32, value=T.convert_integer(c.value, T.INT))
            byte_logic = (getattr(self.source, 'indexed_byte_scans', {}) and kind in ('and', 'or', 'xor')
                          and result.bits == 32 and all(possible_bits(self.source, v) <= 255 for v in args))
            if narrow_logic:
                result, a = M.R(result.reg, 32), M.R(a.reg, 32)
                c = M.R(c.reg, 32) if isinstance(c, M.R) else replace(c, bits=32)
            elif byte_logic:
                result, a = M.R(result.reg, 8), M.R(a.reg, 8)
                c = M.R(c.reg, 8) if isinstance(c, M.R) else replace(c, bits=8)
            if result.bits < a.bits:
                a = M.R(a.reg, result.bits)
            if isinstance(c, M.R) and result.bits < c.bits:
                c = M.R(c.reg, result.bits)
            if isinstance(c, M.Imm) and result.bits == 64 and not -(1 << 31) <= c.value < 1 << 31:
                if kind != 'and' or not 0 <= c.value < 1 << 32:
                    c = self.operand(args[1], True)
            branch_value = any(b.term.kind == 'br' and b.term.args == op.results for b in self.source.blocks)
            if (kind == 'and' and isinstance(c, M.Imm) and result.bits == 64 and 0 <= c.value < 1 << 32
                    and not (branch_value and self.uses[op.results[0]] > 1)):
                result, a, c = M.R(result.reg, 32), M.R(a.reg, 32), replace(c, bits=32)
            if (kind == 'add' and result.bits == 64 and isinstance(c, M.Imm) and c.symbol is None
                    and -(1 << 31) <= c.value < 1 << 31
                    and (op.results[0] in getattr(self.source, 'byte_compare_offsets', ())
                         or op.results[0] in getattr(self.source, 'large_compare_offsets', ())
                         or self.source.type.base.kind == 'ptr' and (getattr(self.source, 'return_joins', ())
                             or op.results[0] in getattr(self.source, 'counted_offsets', ())))):
                self.ins('lea', [result, M.Mem(8, base=a.reg, disp=c.value)], ['W', 'A'])
            elif (kind == 'add' and result.bits == 64 and isinstance(a, M.R) and isinstance(c, M.R)
                    and self.fn.isa in ('sse2', 'sse42') and (getattr(self.source, 'shared_index_return', None) is not None
                        or getattr(self.source, 'pipelined_copies', {}) or getattr(self.source, 'indexed_byte_scans', {})
                        or getattr(self.source, 'word_scans', {}))):
                address = M.Mem(8, base=a.reg, index=c.reg) if (getattr(self.source, 'pipelined_copies', {})
                         or getattr(self.source, 'indexed_byte_scans', {}) or getattr(self.source, 'word_scans', {})) else M.Mem(8, base=c.reg, index=a.reg)
                self.ins('lea', [result, address], ['W', 'A'])
            elif kind == 'mul' and isinstance(c, M.Imm):
                self.ins('imul', [result, a, c], ['W', 'R', 'R'])
            else:
                self.move(M.R(result.reg, 32) if byte_logic else result, M.R(a.reg, 32) if byte_logic else a)
                self.ins('imul' if kind == 'mul' else kind, [result, c], ['RW', 'R'])
        elif kind in ('shl', 'shr'):
            source, count = self.operand(args[0], True), self.operand(self.shift_counts.get(op, args[1]))
            if isinstance(count, M.R) and self.fn.isa in ('avx2', 'evex', 'evex512'):
                name = 'sarx' if kind == 'shr' and args[0].type.signed else 'shrx' if kind == 'shr' else 'shlx'
                self.ins(name, [result, source, M.R(count.reg, result.bits)], ['W', 'R', 'R'])
                return
            self.move(result, source)
            if isinstance(count, M.R):
                self.fn.preferred.setdefault(count.reg, M.physical('rcx'))
                self.move(M.R(M.physical('rcx'), 32), M.R(count.reg, 32))
                count = M.R(M.physical('rcx'), 8)
            self.ins('sar' if kind == 'shr' and args[0].type.signed else kind, [result, count], ['RW', 'R'])
        elif kind in ('div', 'rem'):
            divisor = self.source.defs[args[1]]
            c = divisor.attrs.get('value') if divisor.kind == 'constant' else None
            if c is not None and c > 0 and c & (c - 1) == 0 and not args[0].type.signed:
                source = self.operand(args[0], True)
                if result.bits == 64 and possible_bits(self.source, args[0]) < 1 << 32:
                    result, source = M.R(result.reg, 32), M.R(source.reg, 32)
                self.move(result, source)
                self.ins('shr' if kind == 'div' else 'and', [result, M.Imm(c.bit_length() - 1 if kind == 'div' else c - 1)], ['RW', 'R'])
            else:
                source, denominator = self.operand(args[0], True), self.operand(args[1], True)
                bits = result.bits
                a, d = M.R(M.physical('rax'), bits), M.R(M.physical('rdx'), bits)
                self.move(a, source)
                if args[0].type.signed:
                    self.ins('cqto' if bits == 64 else 'cltd', implicit_uses=frozenset({a.reg}), implicit_defs=frozenset({d.reg}))
                else:
                    self.ins('xor', [M.R(d.reg, 32), M.R(d.reg, 32)], ['W', 'R'])
                self.ins('idiv' if args[0].type.signed else 'div', [denominator], ['R'],
                         implicit_uses=frozenset({a.reg, d.reg}), implicit_defs=frozenset({a.reg, d.reg}))
                self.move(result, a if kind == 'div' else d)
        elif kind in ('neg', 'not'):
            self.move(result, self.operand(args[0], True))
            self.ins(kind, [result], ['RW'])
        elif kind in ('eq', 'ne', 'lt', 'gt', 'le', 'ge'):
            if self.uses[op.results[0]] == 1 and self.source.defs[op.results[0]] is op and any(b.term.kind == 'br' and b.term.args[0] == op.results[0] for b in self.source.blocks):
                return
            cc = self.compare(op)
            self.ins('set' + cc, [M.R(result.reg, 8)], ['W'], cc=cc)
            if result.bits != 8:
                self.ins('movzb', [M.R(result.reg, 32), M.R(result.reg, 8)], ['W', 'R'])
        elif kind == 'select':
            condition, a, c = args
            yes, no = self.source.defs[a], self.source.defs[c]
            bit = self.definition(condition)
            if yes.kind == no.kind == 'constant' and no.attrs['value'] == 0:
                n = yes.attrs['value']
                if n > 0 and n & (n - 1) == 0 and bit.kind == 'and':
                    mask = self.source.defs[bit.args[1]]
                    source = self.source.defs[bit.args[0]]
                    if mask.kind == 'constant' and mask.attrs['value'] == 1 and source.kind == 'shr':
                        shift = self.source.defs[source.args[1]]
                        if shift.kind == 'constant' and source.args[0].type.bits == result.bits:
                            delta = shift.attrs['value'] - (n.bit_length() - 1)
                            if -result.bits < delta < result.bits:
                                value = self.operand(source.args[0], True)
                                self.move(result, M.R(value.reg, result.bits))
                                if delta:
                                    self.ins('shr' if delta > 0 else 'shl', [result, M.Imm(abs(delta), 8)], ['RW', 'R'])
                                self.ins('and', [result, M.Imm(n)], ['RW', 'R'])
                                return
            true, false = self.operand(a, True), self.operand(c)
            if isinstance(false, M.Imm) and false.symbol is None:
                false = self.operand(c, True)
            if result.bits < 32:
                result, true, false = M.R(result.reg, 32), M.R(true.reg, 32), M.R(false.reg, 32)
            cc = self.condition(condition)
            table_return = op in getattr(self.source, 'table_return_selects', ())
            table_difference = op in getattr(self.source, 'table_difference_selects', ())
            if ((table_return or table_difference) and len(self.block.ins) >= 2 and self.block.ins[-2].mnemonic == 'lea'
                    and self.block.ins[-2].operands[0] == (true if table_return else false) and self.block.ins[-1].mnemonic == 'test'
                    and not self.block.ins[-2].defs & self.block.ins[-1].uses):
                self.block.ins[-2:] = self.block.ins[-1], self.block.ins[-2]
            prefer_first = (getattr(self.source, 'shared_load_return', None) is not None
                            and self.register_uses(a) < self.register_uses(c)) or table_return
            if op in getattr(self.source, 'large_compare_select_first', ()):
                prefer_first = True
            elif op in getattr(self.source, 'large_compare_select_second', ()):
                prefer_first = False
            constant_minimum = (bit.kind in ('gt', 'ge') and bit.args == (c, a)
                                and yes.kind == 'constant' and no.kind != 'constant' and self.uses[c] > 1)
            if constant_minimum:
                prefer_first = True
            if (prefer_first or result.bits == 64 and op.results[0].type.kind == 'ptr'
                    and self.owners.get(a) is not self.block and self.uses[a] > 1 and no.kind == 'constant'):
                self.move(result, true)
                cc = ('l' if c.type.signed else 'b') if constant_minimum else INVERSE[cc]
                self.ins('cmov' + cc, [result, false], ['RW', 'R'], cc=cc)
            else:
                comparison = self.definition(condition)
                if (op.results[0].type.kind == 'ptr' and no.kind == 'constant'
                        and comparison.kind == 'lt' and self.bit_count(comparison.args[0])
                        and cc in ('b', 'l') and self.block.ins[-1].mnemonic == 'cmp'
                        and all(isinstance(o, M.R) for o in self.block.ins[-1].operands)):
                    self.block.ins[-1].operands = self.block.ins[-1].operands[::-1]
                    cc = 'a' if cc == 'b' else 'g'
                self.move(result, false)
                self.ins('cmov' + cc, [result, true], ['RW', 'R'], cc=cc)
        elif kind == 'call':
            self.call(op)
        elif kind == 'intrinsic':
            self.intrinsic(op)
        elif kind == 'asm':
            self.asm(op)
        elif kind in ('copy', 'fill'):
            self.memory_operation(op)
        else:
            raise ValueError(self.fn.name + ': operation selection not implemented: ' + kind)

    def asm(self, op):
        code, constraints = op.attrs['code'], op.attrs['constraints']
        if not code:
            if any(c not in ('+r', '+x') for c in constraints[0]) or len(op.args) != len(op.results):
                raise ValueError(self.fn.name + ': unsupported opaque asm constraints')
            for result, value in zip(op.results, op.args):
                self.move(self.value(result), self.operand(value, True))
            return
        names = {'a': 'rax', 'b': 'rbx', 'c': 'rcx', 'd': 'rdx', 'D': 'rdi', 'S': 'rsi'}
        outputs = constraints[0] if constraints else []
        inputs = constraints[1] if len(constraints) > 1 else []
        read, written, cursor = set(), set(), 0
        assignments = []
        for c, result in zip(outputs, op.results):
            if c[-1] not in names:
                raise ValueError(self.fn.name + ': unsupported asm constraint ' + c)
            fixed = M.physical(names[c[-1]])
            written.add(fixed)
            if '+' in c:
                assignments.append((M.R(fixed, result.type.bits), self.operand(op.args[cursor], True)))
                cursor += 1
                read.add(fixed)
        for c in inputs:
            if c not in names:
                raise ValueError(self.fn.name + ': unsupported asm input constraint ' + c)
            value = op.args[cursor]
            cursor += 1
            fixed = M.physical(names[c])
            assignments.append((M.R(fixed, max(32, value.type.bits)), self.operand(value)))
            read.add(fixed)
        order = {'rax': 0, 'rcx': 1, 'rdx': 2, 'rdi': 3, 'rsi': 4}
        for dst, src in sorted(assignments, key=lambda pair: order.get(pair[0].reg.id, 5)):
            self.move(dst, src)
        rep = code.startswith('rep ')
        self.ins(code[4:] if rep else code, implicit_uses=frozenset(read), implicit_defs=frozenset(written),
                 rep='rep' if rep else None, preserve=True)
        for c, result in zip(outputs, op.results):
            self.move(self.value(result), M.R(M.physical(names[c[-1]]), result.type.bits))

    def memory_operation(self, op):
        args = op.args
        count = self.source.defs[args[2]]
        n = count.attrs.get('value') if count.kind == 'constant' else None
        if n is not None and 0 <= n <= 4096:
            offset, loaded = 0, []
            width = 32 if self.fn.isa in ('avx2', 'evex', 'evex512') else 16
            while offset < n:
                k = min(width, 1 << ((n - offset).bit_length() - 1))
                type_ = {1: T.UCHAR, 2: T.USHORT, 4: T.UINT, 8: T.ULONG, 16: T.V128, 32: T.V256}[k]
                reg = self.fn.reg(type_)
                if op.kind == 'copy':
                    src = replace(self.address(args[1], k), disp=self.address(args[1], k).disp + offset)
                    if k in (1, 2):
                        self.ins('movzb' if k == 1 else 'movzw', [M.R(reg.reg, 32), src], ['W', 'R'])
                    else:
                        self.move(reg, src)
                else:
                    value = self.source.defs[args[1]]
                    if value.kind != 'constant' or value.attrs['value'] != 0:
                        raise ValueError(self.fn.name + ': nonzero aggregate initialization not implemented yet')
                    if k >= 16:
                        legacy = k == 16 and self.fn.isa in ('sse2', 'sse42')
                        floating = op.attrs.get('domain') == 'fp'
                        name = ('xorps' if legacy else 'vxorps') if floating else 'pxor' if legacy else 'vpxor'
                        zero = M.R(reg.reg, 128) if floating and not legacy else reg
                        self.ins(name, [zero, zero] if legacy else [zero, zero, zero],
                                 ['W', 'R'] if legacy else ['W', 'R', 'R'])
                    else:
                        self.move(reg, M.Imm(0))
                loaded.append((offset, k, reg))
                offset += k
            for offset, k, reg in loaded:
                dst = self.address(args[0], k)
                self.move(replace(dst, disp=dst.disp + offset), reg)
                self.block.ins[-1].domain = op.attrs.get('domain')
            return
        raise ValueError(self.fn.name + ': variable aggregate memory operation not implemented yet')

    def run(self):
        self.block = self.fn.entry
        for op in self.source.allocas:
            self.ins('lea', [self.value(op.results[0]), M.Mem(8, slot=self.stack[op.results[0]])], ['W', 'A'])
        gp = vp = 0
        parameter_moves = []
        for v in self.source.params:
            if v.type.kind == 'vector':
                source = M.R(M.physical('xmm%d' % vp), v.type.bits)
                vp += 1
            else:
                if gp == len(ARGS):
                    raise ValueError(self.fn.name + ': stack parameters not implemented yet')
                source = M.R(M.physical(ARGS[gp]), v.type.bits)
                gp += 1
            parameter_moves.append(self.move(self.value(v), source))
            self.fn.preferred[self.value(v).reg] = source.reg
            if getattr(self.source, 'distance_returns', False) and len(self.source.params) == 1 and v.type.kind == 'ptr':
                self.fn.preferred[self.value(v).reg] = M.physical('rsi')
        returned = self.opaque_return_parameter()
        if getattr(self.source, 'hot_return_paths', ()) and self.fn.unaligned_loops:
            for block in self.source.blocks:
                for op in block.ops:
                    if op.kind == 'phi' and op.results[0].type.kind == 'ptr' and not any(item.kind == 'call' for item in block.ops):
                        origins = pointer_parameters(self.source, op.results[0])
                        if origins in ({0}, {1}):
                            preferred = M.physical('rdi' if origins == {0} else 'rsi')
                            self.fn.preferred[self.value(op.results[0]).reg] = preferred
                            for value in op.args:
                                if self.blocks[block] in self.fn.unaligned_loops and self.definition(value).kind == 'ptradd':
                                    self.fn.preferred[self.value(value).reg] = preferred
                if self.blocks[block] not in self.fn.unaligned_loops:
                    continue
                for op in block.ops:
                    if op.kind == 'phi' and op.results[0].type.kind == 'int' and op.results[0].type.bits == 64:
                        origins = pointer_parameters(self.source, op.results[0])
                        if origins in ({0}, {1}):
                            self.fn.preferred[self.value(op.results[0]).reg] = M.physical('r8' if origins == {0} else 'rdx')
        if returned is not None:
            self.fn.returned_parameter = returned
            self.fn.preferred[self.value(returned).reg] = M.physical('rax')
            reused = M.physical(ARGS[self.source.params.index(returned)])
            for op in self.source.entry.ops:
                if op.kind != 'and' or op.results[0].type.bits != 64:
                    continue
                source, mask = self.definition(op.args[0]), self.definition(op.args[1])
                if (source.kind == 'param' and source.results[0].type.kind == 'ptr' and mask.kind == 'constant'
                        and mask.attrs['value'] & (1 << 63) and (~mask.attrs['value']) & ((1 << 64) - 1) < 4096):
                    self.fn.preferred[self.value(op.results[0]).reg] = (M.physical('rdx')
                        if getattr(self.source, 'pipelined_copies', {}) else reused)
                    break
        entry_prefix = len(self.fn.entry.ins)
        if getattr(self.source, 'pipelined_copies', {}):
            if self.source.entry.term.kind == 'br':
                self.fn.preferred[self.value(self.source.entry.term.args[0]).reg] = M.physical('rdi')
            for block in self.source.blocks:
                for op in block.ops:
                    if (op.kind == 'add' and op.results[0].type.bits == 64 and any(self.bit_count(v) for v in op.args)
                            and any(user.kind == 'add' and self.definition(user.args[1]).kind == 'constant'
                                    and self.definition(user.args[1]).attrs['value'] == 1 for user in self.users.get(op.results[0], ()))):
                        self.fn.preferred[self.value(op.results[0]).reg] = M.physical('rdx')
                        for user in self.users.get(op.results[0], ()):
                            if user.kind == 'add' and self.definition(user.args[1]).kind == 'constant':
                                self.fn.preferred[self.value(user.results[0]).reg] = M.physical('rdx')
        for b in self.source.blocks:
            self.block = self.blocks[b]
            at, casts = len(b.ops)-1, set()
            while at >= 0 and b.ops[at].kind == 'cast' and b.ops[at].results[0].type.bits == b.ops[at].args[0].type.bits:
                casts.add(b.ops[at])
                at -= 1
            returned = []
            for value in b.term.args:
                while self.source.defs[value] in casts:
                    value = self.source.defs[value].args[0]
                returned.append(value)
            tail = b.ops[at] if (at >= 0 and b.ops[at].kind == 'call' and b.term.kind == 'ret'
                                 and b.ops[at].results == tuple(returned)
                                 and not any(op.attrs['alignment'] > 16 for op in self.source.allocas)
                                 and not any(self.stack_argument(v) for v in b.ops[at].args[1:])) else None
            for op in b.ops:
                if op is not tail and not (tail is not None and op in casts):
                    self.operation(op)
            term = b.term
            self.block.successors = [self.blocks[t] for t in term.targets]
            if tail is not None:
                self.call(tail, tail=True)
            elif term.kind == 'jmp':
                self.ins('jmp', target=self.blocks[term.targets[0]])
            elif term.kind == 'br':
                defining = self.source.defs[term.args[0]]
                if term.args[0] in self.branch_tests:
                    value, bit, memory = self.branch_tests[term.args[0]]
                    if memory:
                        load = self.source.defs[value]
                        address = self.address(load.args[0], 1)
                        operand = replace(address, disp=address.disp + bit // 8)
                        mask = 1 << (bit % 8)
                    else:
                        operand, mask = self.operand(value, True), 1 << bit
                    self.ins('test', [operand, M.Imm(mask)], ['R', 'R'])
                    cc = 'ne'
                elif defining.kind in ('eq', 'ne', 'lt', 'gt', 'le', 'ge'):
                    cc = self.compare(defining)
                else:
                    cc = self.condition(term.args[0])
                self.ins('j' + cc, target=self.blocks[term.targets[0]], cc=cc)
                self.ins('jmp', target=self.blocks[term.targets[1]])
            elif term.kind == 'ret':
                if term.args:
                    v = term.args[0]
                    destination = M.R(M.physical('xmm0' if v.type.kind == 'vector' else 'rax'), max(32, v.type.bits) if v.type.kind != 'vector' else v.type.bits)
                    source = self.operand(v)
                    if isinstance(source, M.R) and source.bits < destination.bits:
                        source = M.R(source.reg, destination.bits)
                    self.move(destination, source)
                    self.ins('ret', implicit_uses=frozenset({destination.reg}))
                else:
                    self.ins('ret')
            else:
                self.ins('ud2')
        # Phi assignments execute on their incoming edges, in parallel.
        for b in self.source.blocks:
            phis = [op for op in b.ops if op.kind == 'phi']
            if not phis:
                continue
            for pred in self.source.predecessors()[b]:
                edge = self.fn.block('edge.%d.%d' % (pred.id, b.id))
                edge.source_parent = pred
                edge.successors = [self.blocks[b]]
                self.block = edge
                copies = []
                for phi in phis:
                    at = phi.attrs['blocks'].index(pred)
                    destination = self.value(phi.results[0])
                    source = self.value(phi.args[at]) if destination.reg.bank == 'mask' else self.operand(phi.args[at])
                    copies.append((destination, source))
                self.ins('parallel', copies=tuple(copies))
                self.ins('jmp', target=self.blocks[b])
                if b in getattr(self.source, 'deferred_phi_blocks', ()):
                    self.fn.layout_owners[edge] = self.blocks[self.source.layout_owners[b]]
                if b in self.scalar_loop_exits and all(isinstance(value, M.Imm) and value.value == 0 for _, value in copies):
                    self.fn.layout_owners[edge] = self.blocks[self.scalar_loop_exits[b]]
                previous = self.blocks[pred]
                previous.successors = [edge if t is self.blocks[b] else t for t in previous.successors]
                for ins in previous.ins:
                    if ins.target is self.blocks[b]:
                        ins.target = edge
        # A scalar view is needed only on paths that consume it. In loops the
        # common zero-mask path can test the mask bank directly.
        predecessors = {b: [p for p in self.fn.blocks if b in p.successors] for b in self.fn.blocks}
        dom = {b: {b} if b is self.fn.entry else set(self.fn.blocks) for b in self.fn.blocks}
        changed = True
        while changed:
            changed = False
            for b in self.fn.blocks:
                if b is self.fn.entry or not predecessors[b]:
                    continue
                current = {b} | set.intersection(*(dom[p] for p in predecessors[b]))
                if dom[b] != current:
                    dom[b], changed = current, True
        for block, ins, mask in self.mask_tests:
            pair = self.scalar_masks.get(mask.reg)
            users = [b for b in self.fn.blocks if pair is not None and any(pair[1].reg in i.uses for i in b.ins)]
            after = any(all(s in dom[b] for b in users) for s in block.successors)
            short_return = any(b.ins[-1].mnemonic == 'ret'
                               and sum(i.mnemonic not in ('mov', 'parallel', 'jmp', 'ret') for i in b.ins) <= 1 for b in users)
            if users and (not after or short_return or self.scalar_search_masks and mask.bits <= 32) and block not in self.loop_blocks:
                ins.mnemonic, ins.operands = 'test', (pair[1], pair[1])
            else:
                ins.mnemonic = 'kortest' + {8: 'b', 16: 'w', 32: 'd', 64: 'q'}[mask.bits]
        for reg, (source, result) in self.scalar_masks.items():
            owner = next((b for b in self.fn.blocks if any(reg in i.defs for i in b.ins)), None)
            if owner is None:
                raise ValueError(self.fn.name + ': mask without a definition')
            users = [b for b in self.fn.blocks if any(result.reg in i.uses for i in b.ins)]
            if not users:
                continue
            target = max(set.intersection(*(dom[b] for b in users)), key=lambda b: len(dom[b]))
            if self.scalar_search_masks and source.bits <= 32:
                target = owner
            at = next((n for n, i in enumerate(target.ins) if result.reg in i.uses or i.target is not None
                       or i.mnemonic in ('ret', 'jmp')), len(target.ins))
            if target is owner:
                at = next(n for n, i in enumerate(owner.ins) if reg in i.defs) + 1
            elif self.mask_phi_blocks.get(reg) is target:
                at = 0
            target.ins.insert(at, M.Instr('kmov' + {8: 'b', 16: 'w', 32: 'd', 64: 'q'}[source.bits],
                                        (result, source), ('W', 'R')))
        if self.low_bit_masks:
            for block in self.fn.blocks:
                reads = [ins for ins in block.ins if ins.mnemonic.startswith('kmov')
                         and all(isinstance(o, M.R) for o in ins.operands)
                         and ins.operands[0].reg.bank == 'gpr' and ins.operands[1].reg.bank == 'mask']
                if len(reads) < 2:
                    continue
                first = block.ins.index(reads[0])
                for ins in reads[1:]:
                    at = block.ins.index(ins)
                    if not any(ins.defs & (other.uses | other.defs) or ins.uses & other.defs
                               for other in block.ins[first:at]):
                        block.ins.pop(at)
                        block.ins.insert(first + 1, ins)
                        first += 1
        self.fn.entry.ins[entry_prefix:entry_prefix] = self.hoisted
        for returned, probe in self.table_constant_returns.items():
            if (len(returned.ins) == 2 and returned.ins[0].mnemonic == 'mov'
                    and returned.ins[-1].mnemonic == 'ret' and isinstance(returned.ins[0].operands[1], M.Imm)):
                at = next((n for n, ins in enumerate(probe.ins) if ins.mnemonic in ('cmp', 'test')), None)
                if at is not None:
                    probe.ins.insert(at, returned.ins.pop(0))
        for head, (phi, read) in getattr(self.source, 'pipelined_copies', {}).items():
            block = self.blocks[head]
            carrier = self.value(phi.results[0])
            previous = self.fn.reg(phi.results[0].type)
            for ins in block.ins:
                ins.map_regs({carrier.reg: previous.reg})
            name = 'movdqa' if self.fn.isa in ('sse2', 'sse42') else 'vmovdqa'
            block.ins.insert(0, M.Instr(name, (previous, carrier), ('W', 'R')))
            self.fn.preferred[carrier.reg] = M.physical('xmm1')
            self.fn.preferred[self.value(read.results[0]).reg] = M.physical('xmm1')
            for parent, value in zip(phi.attrs['blocks'], phi.args):
                if parent is not head:
                    self.fn.preferred[self.value(value).reg] = M.physical('xmm1')
            self.fn.preferred[previous.reg] = M.physical('xmm2')
            self.fn.preferred[self.value(head.term.args[0]).reg] = M.physical('rsi')
        if any(i.rep or i.mnemonic in ('cpuid', 'xgetbv') for b in self.fn.blocks for i in b.ins):
            self.sink_parameters(parameter_moves)
        self.prune_moves()
        self.sink_constants(dom)
        self.hoist_word_step_constants()
        self.hoist_final_pointer_updates()
        self.hoist_loop_constants(predecessors, dom)
        self.sink_induction_updates()
        self.merge_phi_edges()
        self.hoist_loop_phi_copies()
        self.hoist_search_return_copies()
        self.hoist_exit_constants()
        self.schedule_page_guards()
        self.schedule_broadcasts()
        self.schedule_vectors()
        self.schedule_packed_loads()
        self.schedule_indexed_byte_scans()
        self.localize_vex_zero_compares()
        self.pair_zero_compares()
        self.fn.vector_tail_values = {self.value(v).reg for v in self.loop_vectors if v in self.values}
        def bare_return(block, seen=frozenset()):
            if block in seen or not block.ins or len(block.ins) > 3:
                return False
            if any(i.mnemonic not in ('mov', 'xor', 'parallel', 'jmp', 'ret')
                   or any(isinstance(o, M.Mem) for o in i.operands) for i in block.ins):
                return False
            return block.ins[-1].mnemonic == 'ret' or len(block.successors) == 1 and bare_return(block.successors[0], seen | {block})
        for block in self.fn.blocks:
            if block in self.valid_return_tests and len(block.successors) == 2 and bare_return(block.successors[0]):
                returned, failure = block.successors
                if failure.ins and not failure.successors and failure.ins[-1].callee:
                    self.fn.layout_owners[failure] = returned
                    for other in self.fn.blocks:
                        if (other is not failure and other.ins and not other.successors
                                and other.ins[-1].callee == failure.ins[-1].callee):
                            self.fn.layout_owners[other] = failure
            if getattr(self.source, 'shared_store_return', None) is not None and len(block.successors) == 2:
                def to_loop(b, seen=frozenset()):
                    return (b in self.loop_blocks or b in self.fn.loop_headers
                            or b not in seen and len(b.successors) == 1 and to_loop(b.successors[0], seen | {b}))
                if not to_loop(block.successors[0]) and to_loop(block.successors[1]):
                    block.successors.reverse()
            if (getattr(self.source, 'shared_load_return', None) is not None and block in self.loop_blocks
                    and len(block.successors) == 2 and block.successors[0] not in self.loop_blocks
                    and block.successors[1] in self.loop_blocks):
                fallbacks = getattr(self.source, 'return_address_fallbacks', ())
                if fallbacks and any(i.mnemonic.startswith(('vptest', 'vpcmp')) for i in block.successors[0].ins):
                    owner, cold, seen = self.blocks[fallbacks[-1]], block.successors[0], set()
                    while cold not in seen:
                        seen.add(cold)
                        self.fn.layout_owners[cold] = owner
                        owner = cold
                        cold = next((b for b in cold.successors if b not in self.loop_blocks
                                     and any(i.mnemonic.startswith(('vptest', 'vpcmp')) for i in b.ins)), None)
                        if cold is None:
                            break
                block.successors.reverse()
            if (block not in self.valid_return_tests and block not in self.inline_zero_guards and len(block.successors) == 2
                    and bare_return(block.successors[0]) and not bare_return(block.successors[1])):
                block.successors.reverse()
        self.schedule_table_builds()
        self.finalize_constants()
        fold_flag_branches(self.fn)
        retain_string_indices(self.fn)
        schedule_function(self)
        return self.fn, self.constants

    def finalize_constants(self):
        """Number logical pool entries before fragments introduced by selection."""
        entries = [data for data, name in self.constants.items() if name.startswith('.LCPI')]
        entries.sort(key=lambda data: data not in self.primary_constants)
        renamed = {self.constants[data]: '.LCPI%d_%d' % (getattr(self.source, 'machine_order', 0), n)
                   for n, data in enumerate(entries)}
        for block in self.fn.blocks:
            for ins in block.ins:
                ins.operands = tuple(replace(o, symbol=renamed[o.symbol])
                                     if isinstance(o, M.Mem) and o.symbol in renamed else o for o in ins.operands)
        other = [data for data in self.constants if data not in entries]
        self.constants = {data: renamed.get(self.constants[data], self.constants[data]) for data in [*entries, *other]}

    def schedule_table_builds(self):
        if not getattr(self.source, 'vector_table_build', False):
            return
        rest = self.blocks[self.source.table_build_remainder]
        addresses = []
        for value in self.source.table_build_pointers:
            reg = self.value(value).reg
            for owner in self.fn.blocks:
                found = next((ins for ins in owner.ins if ins.mnemonic == 'lea' and reg in ins.defs), None)
                if found is not None:
                    owner.ins.remove(found)
                    addresses.append(found)
                    break
        rest.ins[:0] = addresses
        self.fn.table_build_remainder = rest
        head = self.blocks[self.source.table_build_group]
        position = self.value(self.source.table_build_early_position).reg
        bits = self.value(self.source.table_build_pair_bits).reg
        broadcast = next(ins for ins in head.ins if position in ins.defs)
        target = next(ins for ins in head.ins if bits in ins.defs and ins.mnemonic == 'vpbroadcastb')
        head.ins.remove(broadcast)
        head.ins.insert(head.ins.index(target), broadcast)

    def hoist_word_step_constants(self):
        vector = self.blocks.get(getattr(self.source, 'word_scan_vector', None))
        target = self.blocks.get(getattr(self.source, 'word_scan_advance', None))
        if vector is None or target is None or len(vector.successors) != 2:
            return
        incoming, _ = liveness(self.fn)
        for start in vector.successors:
            edge, seen = start, set()
            while edge not in seen and len(edge.ins) == 1 and edge.ins[0].mnemonic == 'jmp' and len(edge.successors) == 1:
                seen.add(edge)
                edge = edge.successors[0]
            if (len(edge.ins) != 2 or edge.ins[0].mnemonic != 'parallel' or edge.successors != [target]
                    or not edge.ins[0].copies or not all(isinstance(src, M.Imm) for _, src in edge.ins[0].copies)):
                continue
            other = next(block for block in vector.successors if block is not start)
            if edge.ins[0].defs & incoming[other]:
                continue
            at = next((n for n, ins in enumerate(vector.ins) if ins.mnemonic in ('test', 'cmp') or ins.target is not None), len(vector.ins))
            old = self.block
            self.block = vector
            copied = []
            for dst, src in edge.ins[0].copies:
                copied.append(self.move(dst, src))
                vector.ins.pop()
            self.block = old
            vector.ins[at:at] = copied
            edge.ins.pop(0)

    def schedule_indexed_byte_scans(self):
        for head, (primary, lookups) in getattr(self.source, 'indexed_byte_scans', {}).items():
            condition = self.source.defs[head.term.args[0]]
            if len(primary) != 4 or condition.kind not in ('and', 'or'):
                continue
            block = self.blocks[head]
            first = self.source.defs[condition.args[0]]
            shared = self.source.defs[first.args[0]]
            delayed = next((ins for ins in block.ins if self.value(lookups[1].results[0]).reg in ins.defs), None)
            last = next((ins for ins in block.ins if self.value(lookups[3].results[0]).reg in ins.defs), None)
            if delayed is None or last is None or shared.kind != condition.kind:
                continue
            reductions = [ins for ins in block.ins if self.value(shared.results[0]).reg in ins.defs]
            block.ins.remove(delayed)
            for ins in reductions:
                block.ins.remove(ins)
            at = block.ins.index(last)
            block.ins[at:at] = [delayed, *reductions]
            if condition.kind == 'or':
                reg = self.value(condition.results[0]).reg
                final = [ins for ins in block.ins if reg in ins.defs]
                test = next((ins for ins in reversed(block.ins) if ins.mnemonic == 'test' and reg in ins.uses), None)
                if test is not None:
                    for ins in final:
                        block.ins.remove(ins)
                    at = block.ins.index(test)
                    block.ins[at:at+1] = final

    def localize_vex_zero_compares(self):
        """Rematerialize cheap zeros along short chains of byte probes."""
        if self.fn.isa != 'avx2':
            return
        pred = {b: [p for p in self.fn.blocks if b in p.successors] for b in self.fn.blocks}
        zeros = [(b, ins) for b in self.fn.blocks for ins in b.ins if ins.mnemonic == 'vpxor']
        if getattr(self.source, 'vector_classification', False):
            for owner, zero in zeros:
                if not all(isinstance(o, M.R) for o in zero.operands) or len({o.reg for o in zero.operands}) != 1:
                    continue
                reg = zero.operands[0].reg
                users = [(b, ins) for b in self.fn.blocks for ins in b.ins if reg in ins.uses]
                if (len(users) != 2 or any(ins.mnemonic != 'vpcmpeqb' for _, ins in users)
                        or not any(b is self.fn.entry for b, _ in users)):
                    continue
                compare = next(ins for b, ins in users if b is self.fn.entry)
                owner.ins.remove(zero)
                zero.operands = (M.R(reg, 128),)*3
                self.fn.entry.ins.insert(self.fn.entry.ins.index(compare), zero)
                self.fn.preferred[reg] = M.physical('xmm3')
                for _, ins in users:
                    result, constant, source = ins.operands
                    ins.operands = result, source, constant
            return
        for owner, zero in zeros:
            if (len(zero.operands) != 3 or not all(isinstance(o, M.R) for o in zero.operands)
                    or zero.operands[0].bits not in (128, 256)
                    or len({o.reg for o in zero.operands}) != 1):
                continue
            reg = zero.operands[0].reg
            users = [(b, ins) for b in self.fn.blocks for ins in b.ins if reg in ins.uses]
            if not users or any(ins.mnemonic != 'vpcmpeqb' or len(ins.operands) != 3
                                or not isinstance(ins.operands[1], M.R)
                                or ins.operands[1].reg != reg for _, ins in users):
                continue
            local = [(b, ins) for b, ins in users if b not in self.loop_blocks]
            probes = {b: ins for b, ins in local if sum(p is b for p, _ in local) == 1}
            if len(probes) != len(local):
                continue
            loops = [(head, members) for head, members in self.loops.items()
                     if all(b in members for b, _ in users if b in self.loop_blocks)]
            remaining = [(b, ins) for b, ins in users if b in self.loop_blocks]
            before = None
            if remaining:
                if not loops:
                    continue
                head, members = min(loops, key=lambda pair: len(pair[1]))
                incoming = [p for p in pred[head] if p not in members
                            and not (len(p.successors) == 1 and any(parent in members for parent in pred[p]))]
                if len(incoming) != 1:
                    continue
                before = incoming[0]
            visited = set()
            for block in self.fn.order():
                if block not in probes or block in visited:
                    continue
                following = next((b for b in block.successors if b in probes and b not in visited
                                  and pred[b] == [block]), None)
                constant = self.fn.reg(T.V128 if zero.operands[0].bits == 128 else T.V256)
                if remaining and isinstance(probes[block].operands[2], M.R):
                    self.fn.preferred[constant.reg] = M.physical('xmm1')
                narrow = M.R(constant.reg, 128)
                at = block.ins.index(probes[block])
                block.ins.insert(at, M.Instr('vpxor', (narrow, narrow, narrow), ('W', 'R', 'R')))
                for member in (block,) if following is None else (block, following):
                    compare = probes[member]
                    result, _, source = compare.operands
                    compare.operands = (result, constant, source) if isinstance(source, M.Mem) else (result, source, constant)
                    visited.add(member)
            owner.ins.remove(zero)
            if remaining:
                zero.operands = tuple(M.R(reg, 128) for _ in zero.operands)
                at = next((n for n, ins in enumerate(before.ins) if ins.target is not None), len(before.ins))
                before.ins.insert(at, zero)
                self.fn.preferred[reg] = M.physical('xmm1')
                for _, compare in remaining:
                    result, constant, source = compare.operands
                    if isinstance(source, M.R):
                        compare.operands = result, source, constant

    def pair_zero_compares(self):
        """Retain a zero vector for one following compare before consuming it."""
        if self.fn.isa not in ('sse2', 'sse42'):
            return
        pred = {b: [p for p in self.fn.blocks if b in p.successors] for b in self.fn.blocks}
        def candidate(block):
            for n, (zero, compare) in enumerate(zip(block.ins, block.ins[1:])):
                if (zero.mnemonic == 'pxor' and len(zero.operands) == 2 and zero.operands[0] == zero.operands[1]
                        and compare.mnemonic == 'pcmpeqb' and len(compare.operands) == 2
                        and compare.operands[0] == zero.operands[0] and isinstance(compare.operands[1], M.Mem)):
                    return n, zero, compare
            return None
        paired = set()
        for block in self.fn.order():
            first = candidate(block)
            if block in paired or first is None:
                continue
            for following in block.successors:
                second = candidate(following)
                if following in paired or pred[following] != [block] or second is None:
                    continue
                at, zero, compare = first
                constant = self.fn.reg(T.V128)
                zero.operands = constant, constant
                result, memory = compare.operands
                block.ins.insert(at + 1, M.Instr('movdqa', (result, memory), ('W', 'R')))
                compare.operands = result, constant
                _, other, _ = second
                other.mnemonic, other.operands, other.access = 'movaps', (other.operands[0], constant), ('W', 'R')
                paired.update((block, following))
                break

    def schedule_packed_loads(self):
        if not self.mask_packs:
            return
        compares = {'vpcmpb', 'vpcmpeqb', 'vptestmb', 'vptestnmb'}
        for block in self.fn.blocks:
            if not any(ins.mnemonic == 'kunpckdq' for ins in block.ins):
                continue
            for ins in list(block.ins):
                if (not ins.mnemonic.startswith('vmov') or len(ins.operands) != 2
                        or not isinstance(ins.operands[0], M.R) or ins.operands[0].reg.bank != 'vec'
                        or not isinstance(ins.operands[1], M.Mem) or ins.preserve or ins.mask is not None):
                    continue
                old = at = block.ins.index(ins)
                while at:
                    previous = block.ins[at - 1]
                    if (previous.mnemonic not in compares or previous.preserve
                            or previous.defs & (ins.defs | ins.uses) or previous.uses & ins.defs):
                        break
                    at -= 1
                if old != at:
                    block.ins.pop(old)
                    block.ins.insert(at, ins)

    def hoist_loop_phi_copies(self):
        if not self.fn.unaligned_loops:
            return
        incoming, _ = liveness(self.fn)
        for block in self.fn.blocks:
            if block not in self.loop_blocks or len(block.successors) != 2:
                continue
            for edge in block.successors:
                if (len(edge.ins) != 2 or edge.ins[0].mnemonic != 'parallel'
                        or len(edge.successors) != 1 or edge.successors[0] not in self.fn.unaligned_loops):
                    continue
                copies = edge.ins[0]
                if not all(isinstance(o, M.R) and o.reg.bank == 'gpr' and o.bits == 64 for pair in copies.copies for o in pair):
                    continue
                other = next(b for b in block.successors if b is not edge)
                if copies.defs & incoming[other]:
                    continue
                at = next((n for n, ins in enumerate(block.ins) if ins.target is not None), len(block.ins))
                if at and block.ins[at-1].mnemonic in ('cmp', 'test'):
                    at -= 1
                block.ins.insert(at, edge.ins.pop(0))

    def hoist_search_return_copies(self):
        if self.fn.isa != 'avx2' or getattr(self.source, 'aligned_search_width', None) != 32:
            return
        target = self.blocks.get(self.source.aligned_search_prefix_return)
        incoming, _ = liveness(self.fn)
        for block in self.fn.blocks:
            if len(block.successors) != 2:
                continue
            for edge in block.successors:
                if (edge.successors != [target] or len(edge.ins) != 2
                        or edge.ins[0].mnemonic != 'parallel'):
                    continue
                copies = edge.ins[0]
                other = next(b for b in block.successors if b is not edge)
                if copies.defs & incoming[other]:
                    continue
                at = next((n for n, ins in enumerate(block.ins) if ins.target is not None), len(block.ins))
                if at and block.ins[at-1].mnemonic in ('cmp', 'test'):
                    at -= 1
                block.ins.insert(at, edge.ins.pop(0))

    def hoist_exit_constants(self):
        length = getattr(self.source, 'vector_length_loop', None) is not None
        if not self.fn.unaligned_loops and not length:
            return
        incoming, _ = liveness(self.fn)
        for block in self.fn.blocks:
            if len(block.successors) != 2:
                continue
            for edge in block.successors:
                start = edge
                if length:
                    seen = set()
                    while (edge not in seen and len(edge.ins) == 1 and edge.ins[0].mnemonic == 'jmp'
                           and len(edge.successors) == 1):
                        seen.add(edge)
                        edge = edge.successors[0]
                if (len(edge.ins) != 2 or edge.ins[0].mnemonic != 'parallel' or len(edge.successors) != 1
                        or not edge.successors[0].ins):
                    continue
                exit_ = edge.successors[0].ins[-1]
                if (length and exit_.mnemonic != 'ret' or not length
                        and (exit_.mnemonic != 'jmp' or exit_.target is not None)):
                    continue
                other = next(b for b in block.successors if b is not start)
                remaining = []
                for destination, source in edge.ins[0].copies:
                    if not isinstance(source, M.Imm) or not isinstance(destination, M.R) or destination.reg in incoming[other]:
                        remaining.append((destination, source))
                        continue
                    if length and source.value == 0:
                        remaining.append((destination, source))
                        continue
                    old = self.block
                    self.block = block
                    ins = self.move(destination, source)
                    block.ins.pop()
                    self.block = old
                    at = next((n for n, item in enumerate(block.ins) if item.target is not None), len(block.ins))
                    if at and (block.ins[at-1].mnemonic == 'cmp' or length and block.ins[at-1].mnemonic == 'test'):
                        at -= 1
                    block.ins.insert(at, ins)
                edge.ins[0].copies = tuple(remaining)

    def sink_induction_updates(self):
        if not getattr(self.source, 'induction_registers', None) and not self.fn.tail_pointers:
            return
        for block in self.fn.blocks:
            if getattr(self.source, 'pipelined_copies', {}) and block not in self.loop_blocks:
                continue
            for ins in list(block.ins):
                if not getattr(self.source, 'induction_registers', None) and ins.operands and isinstance(ins.operands[0], M.R) and ins.operands[0].reg not in self.fn.tail_pointers:
                    continue
                if (ins.mnemonic != 'lea' or not isinstance(ins.operands[1], M.Mem)
                        or ins.operands[1].base is None or ins.operands[1].index is not None or ins.operands[1].disp <= 0):
                    continue
                old, reg, base = block.ins.index(ins), ins.operands[0].reg, ins.operands[1].base
                end = next((n for n in range(old + 1, len(block.ins)) if reg in block.ins[n].uses
                            or base in block.ins[n].defs or block.ins[n].target is not None), len(block.ins))
                reads = [n for n in range(old + 1, end) if any(isinstance(o, M.Mem) and base in M.regs(o) for o in block.ins[n].operands)]
                if reads:
                    at = reads[-1]
                    block.ins.pop(old)
                    block.ins.insert(at, ins)

    def hoist_final_pointer_updates(self):
        target = self.blocks.get(getattr(self.source, 'shared_store_return', None))
        if target is None:
            return
        def returns(block, seen=frozenset()):
            return block is target or block not in seen and len(block.successors) == 1 and returns(block.successors[0], seen | {block})
        for block in self.fn.blocks:
            if len(block.successors) != 1 or not returns(block.successors[0]):
                continue
            for ins in list(block.ins):
                if (ins.mnemonic != 'lea' or not isinstance(ins.operands[1], M.Mem)
                        or ins.operands[1].base is None or ins.operands[1].index is not None):
                    continue
                old = block.ins.index(ins)
                if any(previous.defs & ins.uses or previous.uses & ins.defs for previous in block.ins[:old]):
                    continue
                block.ins.pop(old)
                block.ins.insert(0, ins)

    def opaque_return_parameter(self):
        if (self.source.type.base.kind != 'ptr' or not self.fn.loop_headers
                or any(op.kind == 'call' for b in self.source.blocks for op in b.ops)):
            return None
        returns = [b.term.args[0] for b in self.source.blocks if b.term.kind == 'ret' and b.term.args]
        if not returns:
            return None
        roots, opaque = set(), False
        for value in returns:
            while True:
                op = self.source.defs[value]
                if op.kind == 'cast' and value.type.bits == op.args[0].type.bits:
                    value = op.args[0]
                elif op.kind == 'asm' and not op.attrs['code'] and len(op.args) == len(op.results) == 1:
                    value, opaque = op.args[0], True
                else:
                    break
            roots.add(value)
        return next(iter(roots)) if opaque and len(roots) == 1 and roots <= set(self.source.params) else None

    def hoist_loop_constants(self, predecessors, dom):
        def before_branch(block):
            return next((n for n, item in enumerate(block.ins)
                         if item.target is not None or item.mnemonic in ('jmp', 'ret', 'ud2')), len(block.ins))
        for ins in list(self.fn.entry.ins):
            if (ins.mnemonic != 'pxor' or not ins.operands
                    or not all(isinstance(o, M.R) for o in ins.operands)
                    or len({o.reg for o in ins.operands}) != 1):
                continue
            value = ins.operands[0].reg
            users = {b for b in self.fn.blocks if any(value in item.uses for item in b.ins if item is not ins)}
            loops = [(head, members) for head, members in self.loops.items() if users and users <= members]
            if loops:
                head, _ = max(loops, key=lambda pair: len(pair[1]))
                incoming = [p for p in predecessors[head] if head not in dom[p]]
                if len(incoming) == 1:
                    self.fn.entry.ins.remove(ins)
                    incoming[0].ins.insert(before_branch(incoming[0]), ins)
            elif users:
                common = set.intersection(*(dom[b] for b in users))
                target = max(common, key=lambda b: len(dom[b]))
                self.fn.entry.ins.remove(ins)
                at = next((n for n, item in enumerate(target.ins) if value in item.uses), len(target.ins))
                at = min(at, before_branch(target))
                if at and len(target.ins[at].operands) >= 2:
                    previous, first = target.ins[at-1], target.ins[at]
                    if (previous.mnemonic in ('movups', 'movaps', 'movdqu', 'movdqa')
                            and len(previous.operands) == 2 and all(isinstance(o, M.R) for o in previous.operands)
                            and previous.operands[0] == first.operands[0]):
                        at -= 1
                target.ins.insert(at, ins)
        if getattr(self.source, 'word_scans', {}):
            constants = []
            for block in self.loop_blocks:
                for ins in list(block.ins):
                    if (ins.mnemonic == 'mov' and len(ins.operands) == 2 and isinstance(ins.operands[0], M.R)
                            and ins.operands[0].bits == 64 and isinstance(ins.operands[1], M.Imm)
                            and ins.operands[1].symbol is None and not -(1 << 31) <= ins.operands[1].value < 1 << 32):
                        block.ins.remove(ins)
                        constants.append(ins)
            constants.sort(key=lambda ins: self.fn.preferred.get(ins.operands[0].reg) != M.physical('rcx'))
            self.fn.entry.ins[0:0] = constants
        hot = bool(getattr(self.source, 'hot_return_paths', ()))
        if getattr(self.source, 'shared_load_return', None) is None and not hot:
            return
        for block in self.fn.blocks:
            for ins in list(block.ins):
                if (ins.mnemonic != 'mov' or len(ins.operands) != 2
                        or not isinstance(ins.operands[0], M.R)
                        or not isinstance(ins.operands[1], M.Imm) or ins.operands[1].symbol is not None):
                    continue
                if ins.operands[0].bits != 64 and not (ins.operands[0].bits == 32 and ins.operands[1].value == -1):
                    continue
                loops = [(head, members) for head, members in self.loops.items() if block in members]
                if not loops:
                    continue
                head, _ = max(loops, key=lambda pair: len(pair[1]))
                incoming = [p for p in predecessors[head] if head not in dom[p]]
                if len(incoming) != 1:
                    continue
                target = incoming[0]
                block.ins.remove(ins)
                target.ins.insert(0 if ins.operands[0].bits == 32 else before_branch(target), ins)
                if ins.operands[1].value == -1:
                    self.fn.preferred[ins.operands[0].reg] = M.physical('rax')

    def schedule_page_guards(self):
        for source, block in self.blocks.items():
            for op in source.ops:
                if not op.attrs.get('page_guard'):
                    continue
                for value in op.args:
                    reg = self.value(value).reg
                    at = next((n for n, ins in enumerate(block.ins) if ins.mnemonic.startswith('set')
                               and ins.operands[0].reg == reg), None)
                    if at is None or at == 0 or block.ins[at - 1].mnemonic != 'cmp':
                        continue
                    comparison, result = block.ins[at - 1:at + 1]
                    operands = {r for r in comparison.uses if r.bank != 'flags'}
                    first = max((n + 1 for n, ins in enumerate(block.ins[:at - 1]) if ins.defs & operands), default=0)
                    del block.ins[at - 1:at + 1]
                    block.ins[first:first] = [comparison, result]

    def schedule_broadcasts(self):
        if any(op.kind != 'call' and 'write' in op.attrs.get('memory', '') for b in self.source.blocks for op in b.ops):
            return
        if self.fn.isa in ('sse2', 'sse42') and getattr(self.source, 'guarded_pointer_returns', ()):
            for block in self.fn.blocks:
                for first in list(block.ins):
                    at = block.ins.index(first)
                    if first.mnemonic != 'movd':
                        continue
                    end = at+1
                    if end < len(block.ins) and block.ins[end].mnemonic in ('movaps', 'movups'):
                        end += 1
                    if [ins.mnemonic for ins in block.ins[end:end+3]] != ['punpcklbw', 'pshuflw', 'pshufd']:
                        continue
                    group = block.ins[at:end+3]
                    uses = set().union(*(ins.uses for ins in group))
                    defs = set().union(*(ins.defs for ins in group))
                    old = at
                    while at:
                        previous = block.ins[at-1]
                        if (previous.target is not None or previous.preserve or previous.mnemonic in ('call', 'jmp', 'ret')
                                or any(isinstance(o, M.Mem) for o in previous.operands)
                                or previous.defs & (uses | defs) or previous.uses & defs):
                            break
                        at -= 1
                    if old != at:
                        del block.ins[old:old+len(group)]
                        block.ins[at:at] = group
        for block in self.fn.blocks:
            for ins in list(block.ins):
                if ins.mnemonic != 'vpbroadcastb' or not all(isinstance(o, M.R) for o in ins.operands):
                    continue
                old = at = block.ins.index(ins)
                if self.fn.isa == 'avx2' and not self.fn.loop_headers and old:
                    scalar = block.ins[old-1]
                    if (scalar.mnemonic == 'vmovd' and len(scalar.operands) == 2
                            and isinstance(scalar.operands[0], M.R)
                            and scalar.operands[0].reg == ins.operands[1].reg):
                        group = [scalar, ins]
                        uses = scalar.uses | ins.uses
                        defs = scalar.defs | ins.defs
                        first = at = old-1
                        while at:
                            previous = block.ins[at-1]
                            if (previous.target is not None or previous.preserve
                                    or previous.mnemonic in ('call', 'jmp', 'ret')
                                    or previous.mnemonic.startswith('cmov')
                                    or any(isinstance(o, M.Mem) for o in previous.operands)
                                    or any(r.bank == 'vec' for r in previous.defs | previous.uses)
                                    or previous.defs & (uses | defs) or previous.uses & defs):
                                break
                            at -= 1
                        if first != at:
                            del block.ins[first:first+2]
                            block.ins[at:at] = group
                        continue
                while at:
                    previous = block.ins[at - 1]
                    if (previous.target is not None or previous.preserve or previous.mnemonic in ('call', 'jmp', 'ret')
                            or any(isinstance(o, M.Mem) for o in previous.operands)
                            or any(r.bank == 'vec' for r in previous.defs | previous.uses)
                            or previous.defs & (ins.uses | ins.defs) or ins.defs & previous.uses):
                        break
                    at -= 1
                if at != old:
                    block.ins.pop(old)
                    block.ins.insert(at, ins)

    def schedule_vectors(self):
        """Order independent reduction pairs by the inputs' definition order."""
        self.schedule_vector_streams()
        if getattr(self.source, 'aligned_search_width', None) == 32 or getattr(self.source, 'large_byte_compares', False):
            return
        pure = {'vpminub', 'vpand', 'vpor', 'vpxor', 'vpxord', 'vpaddb'}
        for block in self.fn.blocks:
            if any(ins.mnemonic == 'vpternlogd' for ins in block.ins):
                continue
            positions = {r: n for n, ins in enumerate(block.ins) for r in ins.defs}
            start = 0
            while start < len(block.ins):
                end = start
                while (end < len(block.ins) and block.ins[end].mnemonic in pure
                       and len(block.ins[end].operands) == 3
                       and all(isinstance(o, M.R) and o.reg.bank == 'vec' for o in block.ins[end].operands)):
                    end += 1
                if end - start > 1:
                    pending, ordered = list(block.ins[start:end]), []
                    original = {ins: n for n, ins in enumerate(pending)}
                    while pending:
                        definitions = {r for ins in pending for r in ins.defs}
                        ready = [ins for ins in pending if not ins.uses & definitions]
                        if not ready:
                            ordered.extend(pending)
                            break
                        chosen = min(ready, key=lambda ins: (min((positions.get(r, -1) for r in ins.uses), default=-1), original[ins]))
                        ordered.append(chosen)
                        pending.remove(chosen)
                    block.ins[start:end] = ordered
                start = end + 1

    def schedule_vector_streams(self):
        """Keep independent loads and differences ahead of ternary reductions."""
        priority = {'vpxor': 1, 'vpxord': 1, 'vpxorq': 1, 'vpminub': 2, 'vpternlogd': 4}
        moves = {'vmovups', 'vmovaps', 'vmovdqu', 'vmovdqa', 'vmovdqu64', 'vmovdqa64'}
        for block in self.fn.blocks:
            if not any(i.mnemonic == 'vpternlogd' for i in block.ins):
                continue
            positions = {r: n for n, i in enumerate(block.ins) for r in i.defs if r.bank == 'vec'}
            def rank(ins):
                if ins.mnemonic in moves:
                    return 0 if isinstance(ins.operands[1], M.Mem) else 3
                return priority.get(ins.mnemonic)
            start = 0
            while start < len(block.ins):
                end = start
                while end < len(block.ins) and rank(block.ins[end]) is not None and not block.ins[end].preserve:
                    end += 1
                if end - start > 1:
                    region = block.ins[start:end]
                    before = {ins: {old for old in region[:n]
                                    if old.defs & (ins.uses | ins.defs) or old.uses & ins.defs}
                              for n, ins in enumerate(region)}
                    pending, ordered, done = list(region), [], set()
                    while pending:
                        ready = [ins for ins in pending if before[ins] <= done]
                        chosen = min(ready, key=lambda ins: (rank(ins), min((positions.get(r, -1) for r in ins.uses if r.bank == 'vec'), default=-1), region.index(ins)))
                        ordered.append(chosen)
                        done.add(chosen)
                        pending.remove(chosen)
                    block.ins[start:end] = ordered
                start = end + 1

    def merge_phi_edges(self):
        canonical, aliases = {}, {}
        return_block = self.blocks.get(getattr(self.source, 'shared_return', None))
        for b in self.fn.blocks:
            if len(b.ins) == 2 and b.ins[0].mnemonic == 'parallel' and b.ins[1].mnemonic == 'jmp' and b.ins[1].target is not None:
                if b.ins[1].target is return_block:
                    continue
                key = b.ins[0].copies, b.ins[1].target
                if key in canonical:
                    aliases[b] = canonical[key]
                else:
                    canonical[key] = b
        for b in self.fn.blocks:
            b.successors = [aliases.get(s, s) for s in b.successors]
            for ins in b.ins:
                if ins.target in aliases:
                    ins.target = aliases[ins.target]
        self.fn.blocks = [b for b in self.fn.blocks if b not in aliases]

    def sink_constants(self, dom):
        for block in self.fn.blocks:
            for ins in list(block.ins):
                if not (ins.mnemonic == 'lea' and isinstance(ins.operands[1], M.Mem)):
                    continue
                stack = ins.operands[1].slot is not None
                reg = ins.operands[0].reg
                users = [b for b in self.fn.blocks if any(reg in i.uses for i in b.ins)]
                common_return = self.blocks.get(getattr(self.source, 'shared_store_return', None))
                return_edge = (len(block.successors) > 1 and len(users) == 1 and users[0].name.startswith('edge.')
                               and (users[0].successors == [common_return]
                                    or any(users[0].successors == [self.blocks.get(b)] for b in getattr(self.source, 'fixed_loop_exits', ()))))
                if not stack and not return_edge and (self.fn.loop_headers or any(i.mnemonic == 'call' for b in self.fn.blocks for i in b.ins)):
                    continue
                if len(users) == 1 and users[0] is not block:
                    target = users[0]
                    if block not in dom[target]:
                        continue
                    at = next(n for n, i in enumerate(target.ins) if reg in i.uses)
                    block.ins.remove(ins)
                    target.ins.insert(at, ins)
        for block in self.fn.blocks:
            constants = [ins for ins in block.ins if len(ins.operands) == 2
                         and isinstance(ins.operands[0], M.R) and not ins.operands[0].reg.physical
                         and (ins.mnemonic == 'mov' and isinstance(ins.operands[1], M.Imm)
                              or ins.mnemonic == 'lea' and isinstance(ins.operands[1], M.Mem) and ins.operands[1].slot is not None)]
            if not constants:
                continue
            body = [ins for ins in block.ins if ins not in constants]
            deadline, pending = len(body), {}
            for ins in reversed(constants):
                reg = ins.operands[0].reg
                first = next((n for n, other in enumerate(body) if reg in other.uses or other.target is not None
                              or other.mnemonic in ('ret', 'jmp')), len(body))
                if (ins.mnemonic == 'mov' and isinstance(ins.operands[1], M.Imm)
                        and ins.operands[0].bits == 64 and not -(1 << 31) <= ins.operands[1].value < 1 << 32
                        and 0 < first < len(body) and body[first].operands
                        and body[first - 1].mnemonic == 'mov'
                        and body[first - 1].operands[0] == body[first].operands[0]):
                    first -= 1
                deadline = min(deadline, first)
                pending.setdefault(deadline, []).insert(0, ins)
            block.ins = []
            for n in range(len(body) + 1):
                block.ins.extend(pending.get(n, ()))
                if n < len(body):
                    block.ins.append(body[n])

    def sink_parameters(self, moves):
        """Keep input registers until fixed-register asm requires a saved copy."""
        entry = self.fn.entry
        for move in moves:
            dst, src = move.operands
            valid = True
            for ins in entry.ins[entry.ins.index(move) + 1:]:
                if valid and ins.mnemonic == 'mov' and ins.operands == (src, dst):
                    entry.ins.remove(ins)
                    continue
                if src.reg in ins.defs:
                    valid = False
            entry.ins.remove(move)
            at = next((i for i, ins in enumerate(entry.ins) if dst.reg in ins.uses or src.reg in ins.defs
                       or ins.target is not None or ins.mnemonic in ('ret', 'jmp')), len(entry.ins))
            entry.ins.insert(at, move)

    def prune_moves(self):
        changed = True
        while changed:
            changed = False
            _, outgoing = liveness(self.fn)
            for b in self.fn.blocks:
                out, live = [], set(outgoing[b])
                for i in reversed(b.ins):
                    writes = any(isinstance(o, M.Mem) and 'W' in a for o, a in zip(i.operands, i.access))
                    side_effect = writes or i.mnemonic in ('call', 'jmp', 'ret', 'ud2', 'cpuid', 'xgetbv') or i.target is not None or i.preserve
                    if i.defs and not i.defs & live and not side_effect:
                        changed = True
                        continue
                    out.append(i)
                    live = (live - i.defs) | i.uses
                b.ins = list(reversed(out))
