"""SSA recipes for the SSE4.2 set-scanning kernels.

Page recovery keeps its zero and shuffle controls local. The scalar mask
operation remains a branch test, and paired string masks stay in SIMD until
only their low word is needed.
"""
from collections import Counter

from . import types as T
from .ir import dominators, Term
from .builtins import BUILTINS

def prepare_recovery(fn):
    """Keep page recovery and the set-size probes beside their consumers."""
    if fn.type.base != T.ULONG or len(fn.params) != 2 or (not any((op.kind == 'intrinsic' and op.attrs['builtin'].op.startswith('cmpistr') for b in fn.blocks for op in b.ops))):
        return
    pred = fn.predecessors()
    dom = dominators(fn)
    owners = fn.layout_owners = dict(getattr(fn, 'layout_owners', {}))
    fn.recovery_regions = []
    for block in fn.blocks:
        if block.term.kind != 'br':
            continue
        compare = fn.defs[block.term.args[0]]
        if compare.kind != 'le' or fn.defs[compare.args[1]].kind != 'constant' or fn.defs[compare.args[1]].attrs['value'] != 4080:
            continue
        masked = fn.defs[compare.args[0]]
        if masked.kind != 'and' or fn.defs[masked.args[1]].kind != 'constant' or fn.defs[masked.args[1]].attrs['value'] != 4095:
            continue
        fast, cold = block.term.targets
        if fast.term.kind != 'jmp':
            continue
        join = fast.term.targets[0]
        if not any((op.kind == 'phi' and op.results[0].type == T.V128 for op in join.ops)):
            continue
        region = set()
        todo = [cold]
        valid = True
        while todo:
            b = todo.pop()
            if b is join or b in region:
                continue
            region.add(b)
            if len(region) > 4 or b is block or b is fast or (b.term.kind == 'ret') or any((op.kind in ('call', 'asm') or 'write' in op.attrs.get('memory', '') for op in b.ops)):
                valid = False
                break
            todo.extend(b.term.targets)
        incoming = [p for p in pred[join] if p in region]
        if not valid or len(incoming) != 1 or any((cold not in dom[b] for b in region)):
            continue
        fn.recovery_regions.append((cold, region))
        owners[join] = incoming[0]
        owners[cold] = fast
        for b in region:
            parents = [p for p in pred[b] if p in region and p.term.kind == 'jmp']
            if len(parents) == 1:
                owners[b] = parents[0]

    def resolve(b):
        seen = set()
        while b not in seen and b.term.kind == 'jmp' and (not b.ops):
            seen.add(b)
            b = b.term.targets[0]
        return b
    heads = {h for p in fn.blocks for h in p.term.targets if h in dom[p]}
    for block in fn.blocks:
        if block.term.kind != 'br':
            continue
        condition = fn.defs[block.term.args[0]]
        while condition.kind == 'cast':
            condition = fn.defs[condition.args[0]]
        if condition.kind != 'intrinsic' or condition.attrs['builtin'].op != 'movemask8':
            continue
        success = resolve(block.term.targets[0])
        failure = resolve(block.term.targets[1])
        if failure.term.kind != 'br' or not any((op.kind == 'load' and op.attrs['width'] == 1 for op in failure.ops)):
            continue
        if sum((resolve(t) is success for t in failure.term.targets)) != 1:
            continue
        region = set()
        todo = [success]
        while todo:
            b = todo.pop()
            if b in region:
                continue
            region.add(b)
            todo.extend(b.term.targets)
        if failure in region:
            continue
        returns = [b for b in region if b.term.kind == 'ret' and any((h in dom[b] for h in heads & region))]
        if returns:
            owners[failure] = max(returns, key=lambda b: (sum((h in dom[b] for h in heads & region)), b.id))
    return True


def prepare_prefixes(fn):
    if not prepare_recovery(fn):
        return
    pred = fn.predecessors()
    dom = dominators(fn)
    heads = {h for p in fn.blocks for h in p.term.targets if h in dom[p]}
    uses = Counter(v for b in fn.blocks for v in (*b.term.args, *(v for op in b.ops for v in op.args)))
    fn.mask_branches = set()
    fn.mask_schedules = []
    post = {b: {b} if not b.term.targets else set(fn.blocks) for b in fn.blocks}
    changed = True
    while changed:
        changed = False
        for b in reversed(fn.blocks):
            if not b.term.targets:
                continue
            value = {b} | set.intersection(*(post[s] for s in b.term.targets))
            if value != post[b]:
                post[b] = value
                changed = True
    use = {}
    for b in fn.blocks:
        for op in b.ops:
            for n, value in enumerate(op.args):
                use.setdefault(value, []).append(op.attrs['blocks'][n] if op.kind == 'phi' else b)
        for value in b.term.args:
            use.setdefault(value, []).append(b)
    for old in fn.blocks:
        for op in list(old.ops):
            if op.kind != 'intrinsic' or op.attrs['builtin'].op != 'load' or op.results[0].type != T.V128:
                continue
            value = op.args[0]
            while fn.defs[value].kind == 'cast':
                value = fn.defs[value].args[0]
            if value != fn.params[1] or not use.get(op.results[0]):
                continue
            targets = [b for b in fn.blocks if b is not old and b in post[old] and (old in dom[b]) and all((b in dom[u] for u in use[op.results[0]])) and (b not in heads) and any((phi.kind == 'phi' and phi.results[0].type == T.V128 for phi in b.ops))]
            if not targets:
                continue
            target = min(targets, key=lambda b: len(dom[b]))
            region = set()
            todo = [old]
            valid = True
            while todo:
                b = todo.pop()
                if b is target or b in region:
                    continue
                region.add(b)
                ops = b.ops[b.ops.index(op) + 1:] if b is old else b.ops
                if any((x.kind in ('call', 'asm') or 'write' in x.attrs.get('memory', '') or x.attrs.get('volatile') for x in ops)):
                    valid = False
                    break
                todo.extend(b.term.targets)
            if not valid:
                continue
            old.ops.remove(op)
            at = next((n for n, x in enumerate(target.ops) if x.kind != 'phi'), len(target.ops))
            target.ops.insert(at, op)
    for block in fn.blocks:
        if block.term.kind != 'br':
            continue
        test = fn.defs[block.term.args[0]]
        if test.kind != 'and' or test.results[0].type != T.UINT or uses[test.results[0]] != 1:
            continue
        for mask, other in (test.args, test.args[::-1]):
            masked = fn.defs[mask]
            if masked.kind != 'sub' or fn.defs[masked.args[1]].kind != 'constant' or fn.defs[masked.args[1]].attrs['value'] != 1:
                continue
            shift = fn.defs[masked.args[0]]
            if shift.kind != 'shl' or fn.defs[shift.args[0]].kind != 'constant' or fn.defs[shift.args[0]].attrs['value'] != 1:
                continue
            source = fn.defs[other]
            while source.kind == 'cast':
                source = fn.defs[source.args[0]]
            if source.kind != 'intrinsic' or source.attrs['builtin'].op != 'movemask8':
                continue
            compare = fn.defs[source.args[0]]
            new = fn.make('shl', [T.UINT], [fn.constant(T.UINT, -1), shift.args[1]])
            at = block.ops.index(masked)
            if compare in block.ops and (fn.defs[shift.args[1]] not in block.ops or block.ops.index(fn.defs[shift.args[1]]) < block.ops.index(compare)):
                at = block.ops.index(compare)
            block.ops.insert(at, new)
            masked.kind = 'not'
            masked.args = new.results
            masked.attrs = {}
            fn.mask_branches.add(test.results[0])
            fn.mask_schedules.append((block, new.results[0], compare.results[0]))
            break
    return True

def string_bare(fn, value):
    while fn.defs[value].kind == 'cast':
        value = fn.defs[value].args[0]
    return value

def configure_prefixes(fn):
    if fn.name not in ('strspn_sse42', 'strcspn_sse42'):
        return
    registers = fn.induction_registers = dict(getattr(fn, 'induction_registers', {}))
    vectors = fn.vector_registers = dict(getattr(fn, 'vector_registers', {}))
    fn.string_regions = []
    pred = fn.predecessors()
    for stage, (root, region) in enumerate(fn.recovery_regions):
        data, control, zero = ((1, 3, 2), (0, 2, 0), (2, 3, 1), (4, 1, 0))[stage]
        fn.string_regions.append((root, region, zero))
        join = next((b for b, owner in fn.layout_owners.items() if owner in region and b not in region and any((op.kind == 'phi' and op.results[0].type == T.V128 for op in b.ops))))
        for block in region | {join}:
            for op in block.ops:
                if op.kind == 'phi' and op.results[0].type == T.V128:
                    vectors[op.results[0]] = 'xmm' + str(data)
                if op.kind == 'intrinsic':
                    name = op.attrs['builtin'].op
                    if op.results and op.results[0].type == T.V128:
                        if block is root:
                            color = 0 if name in ('cmpgt_epi8', 'cmpeq_epi8') else data if name in ('load', 'shuffle_epi8') else control
                        else:
                            color = zero if name == 'sub_epi8' else data if name == 'or' else 0
                        vectors[op.results[0]] = 'xmm' + str(color)
                    if name == 'movemask8' and block is root:
                        registers[op.results[0]] = 'r8' if stage == 2 else 'rdx'
                if block is root and op.results and (op.results[0].type.kind == 'int'):
                    if op.kind == 'and' and fn.defs[op.args[1]].kind == 'constant':
                        mask = fn.defs[op.args[1]].attrs['value']
                        if mask in (15, (1 << 64) - 16):
                            registers[op.results[0]] = 'rdx' if stage == 2 else 'rax'
                    if op.kind == 'sub' and fn.defs[op.args[0]].kind == 'constant' and (fn.defs[op.args[0]].attrs['value'] == 16):
                        registers[op.results[0]] = 'rdx' if stage == 1 else 'rcx'
                    if op.kind in ('shl', 'not') and op.results[0].type == T.UINT:
                        registers[op.results[0]] = 'rdx' if stage == 2 else 'rax'
    for block in fn.blocks:
        for op in list(block.ops):
            if op.kind == 'ptradd' and string_bare(fn, op.args[0]) == fn.params[1] and (fn.defs[op.args[1]].kind == 'constant') and (fn.defs[op.args[1]].attrs['value'] == 16):
                registers[op.results[0]] = 'rax'
                fn.word_return_bases = set(getattr(fn, 'word_return_bases', ())) | set(op.results)
            if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'load' and (string_bare(fn, op.args[0]) == fn.params[1]):
                if any((u.kind == 'intrinsic' and u.attrs['builtin'].op == 'cmpistrm' and (u.args[0] == op.results[0]) for b in fn.blocks for u in b.ops)):
                    vectors[op.results[0]] = 'xmm3'
            if op.kind not in ('and', 'or') or not op.results or op.results[0].type != T.UINT:
                continue
            inputs = []
            for value in op.args:
                extract = fn.defs[string_bare(fn, value)]
                if extract.kind != 'intrinsic' or extract.attrs['builtin'].op != 'extract_low32':
                    break
                cmp = fn.defs[extract.args[0]]
                if cmp.kind != 'intrinsic' or cmp.attrs['builtin'].op != 'cmpistrm':
                    break
                inputs.append(extract.args[0])
            if len(inputs) != 2:
                continue
            combined = fn.make('intrinsic', [T.V128], inputs[::-1], {'builtin': BUILTINS['_mm_' + op.kind + '_si128'], 'memory': '', 'width': 0, 'alignment': 1, 'mask': None})
            extract = fn.make('intrinsic', [T.INT], combined.results, {'builtin': BUILTINS['_mm_cvtsi128_si32'], 'memory': '', 'width': 0, 'alignment': 1, 'mask': None})
            at = block.ops.index(op)
            block.ops[at:at] = [combined, extract]
            op.kind = 'cast'
            op.args = extract.results
            op.attrs = {}
            vectors[inputs[0]] = 'xmm1'
            vectors[inputs[1]] = 'xmm0'
            vectors[combined.results[0]] = 'xmm0'
            for cmpvalue in inputs:
                text = fn.defs[cmpvalue].args[1]
                vectors[text] = 'xmm4'

def combine_guarded_return(fn):
    if fn.name != 'strspn_sse42':
        return
    registers = fn.induction_registers
    pred = fn.predecessors()
    for b in list(fn.blocks):
        if b.term.kind != 'br':
            continue
        value = b.term.args[0]
        op = fn.defs[value]
        if op.kind != 'cast' or fn.defs[op.args[0]].kind != 'intrinsic' or fn.defs[op.args[0]].attrs['builtin'].op != 'extract_low32':
            continue
        vector = fn.defs[fn.defs[op.args[0]].args[0]]
        if vector.kind != 'intrinsic' or vector.attrs['builtin'].op != 'and':
            continue
        rhs, failure = b.term.targets
        if pred[rhs] != [b] or rhs.term.kind != 'br' or rhs.term.targets[1] is not failure:
            continue
        if len(rhs.ops) != 3 or rhs.ops[0].kind != 'intrinsic' or rhs.ops[0].attrs['builtin'].op != 'ctz_nonzero':
            continue
        count, cast, compare = rhs.ops
        if cast.kind != 'cast' or compare.kind != 'lt':
            continue
        present = fn.make('ne', [T.INT], [value, fn.constant(value.type, 0)])
        b.ops.append(present)
        b.ops.extend(rhs.ops)
        compare.kind = 'gt'
        compare.args = compare.args[::-1]
        compare.attrs['wide_compare'] = True
        cast.attrs['force_zero_extension'] = True
        both = fn.make('and', [T.INT], [present.results[0], compare.results[0]])
        b.ops.append(both)
        fn.mask_branches.add(both.results[0])
        registers[present.results[0]] = 'rdx'
        registers[compare.results[0]] = 'rsi'
        registers[count.results[0]] = registers[cast.results[0]] = 'rax'
        b.term = Term('br', both.results, rhs.term.targets)
        for target in rhs.term.targets:
            for phi in target.ops:
                if phi.kind == 'phi':
                    phi.attrs['blocks'] = [b if p is rhs else p for p in phi.attrs['blocks']]
        fn.blocks.remove(rhs)
        fn.string_spn_mask_return = rhs.term.targets[0]
        break


def prepare_string(fn):
    if fn.name not in ('strspn_sse42', 'strcspn_sse42'):
        return False
    if not prepare_prefixes(fn) or len(fn.recovery_regions) != 4:
        return False
    configure_prefixes(fn)
    combine_guarded_return(fn)
    fn.string_prefixes = True
    return True
