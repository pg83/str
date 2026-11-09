"""Carried indices and shared mask counts for grouped byte comparisons."""
from . import types as T
from .ir import Term
from .builtins import BUILTINS


def prepare_large_byte_compares(fn):
    if (getattr(fn, 'large_byte_compares', False) or fn.type.base != T.INT or len(fn.params) != 3
            or any(v.type.kind != 'ptr' for v in fn.params[:2]) or fn.params[2].type != T.ULONG
            or any(op.kind in ('call', 'asm') or op.attrs.get('volatile')
                   or 'write' in op.attrs.get('memory', '') for b in fn.blocks for op in b.ops)):
        return False

    def constant(value):
        op = fn.defs[value]
        return T.convert_integer(op.attrs['value'], T.LONG) if op.kind == 'constant' and isinstance(op.attrs['value'], int) else None

    def bare(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value

    def terms(value):
        value = bare(value)
        op = fn.defs[value]
        if op.kind == 'add' and value.type.bits == 64:
            return terms(op.args[0]) + terms(op.args[1])
        return [value]

    heads = []
    for block in fn.blocks:
        compares = [op for op in block.ops if op.kind == 'intrinsic'
                    and op.attrs['builtin'].op == 'cmpeq_epi8' and op.results[0].type == T.V256]
        phis = [op for op in block.ops if op.kind == 'phi']
        if block.term.kind == 'br' and len(compares) == 4 and len(phis) == 1:
            heads.append((block, compares, phis[0]))
    if len(heads) != 1:
        return False
    head, compares, index = heads[0]
    groups = [op for op in head.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'and']
    roots = [op for op in groups if all(fn.defs[v] in groups for v in op.args)]
    if len(groups) != 3 or len(roots) != 1:
        return False
    root = roots[0]
    left, right = (fn.defs[v] for v in root.args)
    pred = fn.predecessors()
    if len(pred[head]) != 2 or index.results[0].type != T.ULONG:
        return False
    latch = next((b for b in head.term.targets if b in pred[head]), None)
    if latch is None or latch.term.kind != 'br' or head not in latch.term.targets:
        return False
    before = next(b for b in pred[head] if b is not latch)
    exit_ = next(b for b in latch.term.targets if b is not head)
    if before.term.kind != 'br' or set(before.term.targets) != {head, exit_}:
        return False
    initial = fn.defs[index.args[index.attrs['blocks'].index(before)]]
    update = fn.defs[index.args[index.attrs['blocks'].index(latch)]]
    if (initial.kind != 'sub' or constant(initial.args[0]) != 64 or update.kind != 'add'
            or update.args[0] != index.results[0] or constant(update.args[1]) != 128):
        return False
    tests = [fn.defs[b.term.args[0]] for b in (before, latch)]
    distances = []
    for test, pointer in zip(tests, (initial.results[0], update.results[0])):
        if test.kind != 'gt' or constant(test.args[1]) != 128:
            return False
        distance = fn.defs[test.args[0]]
        if distance.kind != 'sub' or distance.args != (fn.params[2], pointer):
            return False
        distances.append(distance)
    alignment = fn.defs[initial.args[1]]
    if alignment.kind != 'and' or constant(alignment.args[1]) != 31:
        return False
    joins = [b for b in getattr(fn, 'return_joins', ()) if b in fn.blocks
             and len([op for op in b.ops if op.kind == 'phi']) == 1
             and sum(op.kind == 'load' and op.attrs['width'] == 1 for op in b.ops) == 2]
    if len(joins) != 1:
        return False
    join = joins[0]
    phi = next(op for op in join.ops if op.kind == 'phi')
    records = []
    for block, value in zip(phi.attrs['blocks'], phi.args):
        if block.term.kind != 'jmp' or block.term.targets != (join,):
            return False
        count, base, delta = None, None, 0
        for term in terms(value):
            op = fn.defs[term]
            if op.kind == 'cast' and op.args[0].type.bits == 32:
                scan = fn.defs[op.args[0]]
                if (count is None and scan in block.ops and scan.kind == 'intrinsic'
                        and scan.attrs['builtin'].op == 'ctz_nonzero' and scan.args[0].type == T.UINT):
                    count = scan
                    continue
            if constant(term) is not None:
                delta += constant(term)
            elif base is None:
                base = term
            else:
                return False
        if count is None or base is None:
            return False
        records.append((block, base, delta, count))
    if (len(records) != 8 or sum(base == index.results[0] for _, base, _, _ in records) != 4
            or any(count.attrs != records[0][3].attrs for _, _, _, count in records)):
        return False

    registers = fn.induction_registers = dict(getattr(fn, 'induction_registers', {}))
    vectors = fn.vector_registers = dict(getattr(fn, 'vector_registers', {}))
    owners = fn.layout_owners = dict(getattr(fn, 'layout_owners', {}))
    fn.large_byte_compares = True
    fn.shared_load_return = join

    def prefer(value, register):
        registers[value] = register
        while fn.defs[value].kind == 'cast':
            value = fn.defs[value].args[0]
            registers[value] = register

    def make(block, kind, type_, args, attrs=None):
        op = fn.make(kind, [type_], args, attrs or {})
        block.ops.append(op)
        return op.results[0]

    # Carry the last vector's index. Its negative displacements stay in the
    # loads, while the loop's bound becomes a descending remainder.
    pre = fn.block('compare.preheader')
    start = make(pre, 'sub', T.ULONG, [fn.constant(T.ULONG, 160), initial.args[1]])
    pre.term = Term('jmp', targets=(head,))
    before.term.targets = tuple(pre if b is head else b for b in before.term.targets)
    carried = fn.make('phi', [T.ULONG], attrs={'blocks': [pre, latch]})
    remaining = fn.make('phi', [T.ULONG], attrs={'blocks': [pre, latch]})
    next_index = make(latch, 'add', T.ULONG, [carried.results[0], fn.constant(T.ULONG, 128)])
    next_remaining = make(latch, 'add', T.ULONG, [remaining.results[0], fn.constant(T.ULONG, -128)])
    carried.args = start, next_index
    remaining.args = distances[0].results[0], next_remaining
    index.kind, index.args, index.attrs = 'sub', (carried.results[0], fn.constant(T.ULONG, 96)), {}
    head.ops[0:0] = [carried, remaining]
    distances[1].kind, distances[1].args, distances[1].attrs = 'cast', (next_remaining,), {}
    latch.ops.remove(update)
    adjusted = fn.block('compare.remainder')
    update.kind, update.args, update.attrs = 'add', (next_index, fn.constant(T.ULONG, -96)), {}
    adjusted.ops = [update]
    adjusted.term = Term('jmp', targets=(exit_,))
    owners[adjusted], owners[exit_] = latch, adjusted
    latch.term.targets = tuple(adjusted if b is exit_ else b for b in latch.term.targets)
    for op in exit_.ops:
        if op.kind == 'phi':
            op.attrs['blocks'] = [adjusted if b is latch else b for b in op.attrs['blocks']]
    ordered(latch, [fn.defs[next_index], fn.defs[next_remaining], tests[1]], fn)
    for value in (carried.results[0], start, next_index, initial.results[0], update.results[0]):
        prefer(value, 'rax')
    for value in (remaining.results[0], distances[0].results[0], distances[1].results[0], next_remaining):
        prefer(value, 'rcx')
    prefer(alignment.results[0], 'r8')
    prefer(alignment.args[0], 'r8')

    # Each exit supplies an index base and one nonzero 32-bit mask. Count
    # only after these choices, before the common pair of byte loads.
    incoming, masks, blocks = [], [], []
    loop_returns = []
    for block, base, delta, count in records:
        if base == index.results[0]:
            base, delta = carried.results[0], delta-96
            loop_returns.append(block)
        if delta:
            base = make(block, 'add', T.ULONG, [base, fn.constant(T.ULONG, delta)])
        incoming.append(base)
        masks.append(count.args[0])
        blocks.append(block)
        prefer(base, 'rax')
        if count.args[0] != head.term.args[0]:
            prefer(count.args[0], 'rcx')
    base = fn.make('phi', [T.ULONG], incoming, {'blocks': blocks})
    mask = fn.make('phi', [T.UINT], masks, {'blocks': blocks})
    count = fn.make('intrinsic', [T.INT], mask.results, dict(records[0][3].attrs))
    extended = fn.make('cast', [T.ULONG], count.results)
    phi.kind, phi.args, phi.attrs = 'add', (extended.results[0], base.results[0]), {}
    join.ops[0:0] = [base, mask, count, extended]
    prefer(phi.results[0], 'rcx')
    prefer(base.results[0], 'rax')
    prefer(mask.results[0], 'rcx')
    prefer(count.results[0], 'rcx')
    prefer(head.term.args[0], 'r8')
    owners[join] = loop_returns[2]
    for b in fn.blocks:
        if b.term.kind == 'ret' and b is not join:
            owners[b] = loop_returns[3]
            owners[loop_returns[0]], owners[loop_returns[1]], owners[loop_returns[2]] = b, loop_returns[0], loop_returns[1]
    fn.large_compare_parts = head, base, mask

    # Start the three independent unaligned reads before their compares.
    # The final aligned read can then reuse the third read's register.
    for compare in compares[:3]:
        compare.args = compare.args[::-1]
    roots = [op for op in head.ops if op.kind == 'phi']
    roots.extend(fn.defs[op.args[0]] for op in compares[:3])
    roots.extend(compares[:3])
    roots.extend([fn.defs[compares[3].args[0]], compares[3], right, left, root])
    ordered(head, roots, fn)
    for compare, read_reg, result_reg in zip(compares, ('xmm0', 'xmm1', 'xmm3', 'xmm3'), ('xmm2', 'xmm1', 'xmm0', 'xmm3')):
        vectors[compare.args[0]] = read_reg
        vectors[compare.results[0]] = result_reg
    vectors[right.results[0]], vectors[left.results[0]], vectors[root.results[0]] = 'xmm3', 'xmm4', 'xmm3'
    return True


def finish_large_byte_compares(fn):
    if not getattr(fn, 'large_byte_compares', False) or getattr(fn, 'large_compare_pairs', False):
        return False
    head, base_phi, mask_phi = fn.large_compare_parts
    join = fn.shared_load_return
    pred = fn.predecessors()

    def bare(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value

    def constant(value):
        op = fn.defs[value]
        return T.convert_integer(op.attrs['value'], T.LONG) if op.kind == 'constant' and isinstance(op.attrs['value'], int) else None

    def affine(value):
        value, delta = bare(value), 0
        while fn.defs[value].kind in ('add', 'sub') and constant(fn.defs[value].args[1]) is not None:
            op = fn.defs[value]
            delta += constant(op.args[1]) * (-1 if op.kind == 'sub' else 1)
            value = bare(op.args[0])
        return value, delta

    pairs = []
    for block in fn.blocks:
        if block.term.kind != 'br':
            continue
        test = fn.defs[block.term.args[0]]
        if test.kind != 'eq' or constant(test.args[1]) != 0xffffffff:
            continue
        move = fn.defs[bare(test.args[0])]
        if move.kind != 'intrinsic' or move.attrs['builtin'].op != 'movemask8':
            continue
        intersection = fn.defs[move.args[0]]
        if intersection.kind != 'intrinsic' or intersection.attrs['builtin'].op != 'and':
            continue
        compares = [fn.defs[v] for v in intersection.args]
        if any(op not in block.ops or op.kind != 'intrinsic' or op.attrs['builtin'].op != 'cmpeq_epi8'
               or op.results[0].type != T.V256 for op in compares):
            continue
        cold = block.term.targets[1]
        if pred[cold] != [block] or cold.term.kind != 'br':
            continue
        yes, no = cold.term.targets
        if any(pred[b] != [cold] or b.term.kind != 'jmp' or b.term.targets != (join,)
               or b not in base_phi.attrs['blocks'] or b not in mask_phi.attrs['blocks'] for b in (yes, no)):
            continue
        first = fn.defs[bare(cold.term.args[0])]
        if first.kind != 'xor' or constant(first.args[1]) != 0xffffffff:
            continue
        extracted = fn.defs[bare(first.args[0])]
        if extracted.kind != 'intrinsic' or extracted.attrs['builtin'].op != 'movemask8':
            continue
        compare = fn.defs[extracted.args[0]]
        if compare not in compares:
            continue
        compares = [compare, next(op for op in compares if op is not compare)]
        bases = [affine(base_phi.args[base_phi.attrs['blocks'].index(b)]) for b in (yes, no)]
        if bases[0][0] != bases[1][0] or bases[1][1] - bases[0][1] != 32:
            continue
        pairs.append((block, test, cold, yes, no, compares, bases))
    if len(pairs) != 2:
        return False

    registers, vectors = fn.induction_registers, fn.vector_registers
    offsets = fn.large_compare_offsets = set()
    prefer_first = fn.large_compare_select_first = set()
    prefer_second = fn.large_compare_select_second = set()

    def prefer(value, register):
        registers[value] = register
        while fn.defs[value].kind == 'cast':
            value = fn.defs[value].args[0]
            registers[value] = register

    def operation(kind, type_, args, attrs=None):
        return fn.make(kind, [type_], args, attrs or {})

    def intrinsic(name, args):
        builtin = BUILTINS[name]
        return operation('intrinsic', builtin.result, args, {'builtin': builtin, 'memory': '',
                         'width': 0, 'alignment': 1, 'mask': None})

    removed, last_cold = set(), None
    for block, test, cold, yes, no, compares, bases in pairs:
        last = bases[0][0] == fn.params[2]
        if last:
            last_cold = cold
        # Testing the XOR reduction keeps the four original vectors available
        # for the uncommon path that has to identify the differing block.
        x = intrinsic('_mm256_xor_si256', compares[0].args)
        y = intrinsic('_mm256_xor_si256', compares[1].args)
        reduced = intrinsic('_mm256_or_si256', [x.results[0], y.results[0]])
        block.ops[block.ops.index(test):block.ops.index(test)] = [x, y, reduced]
        test.kind, test.args = 'intrinsic', (reduced.results[0], reduced.results[0])
        builtin = BUILTINS['_mm256_testz_si256']
        test.attrs = {'builtin': builtin, 'memory': '', 'width': 0, 'alignment': 1, 'mask': None}
        for op in compares:
            block.ops.remove(op)
        ordered(block, [fn.defs[compares[0].args[0]], fn.defs[compares[1].args[0]],
                        fn.defs[compares[0].args[1]], fn.defs[compares[1].args[1]], x, y, reduced, test], fn)
        vectors[x.results[0]], vectors[y.results[0]], vectors[reduced.results[0]] = 'xmm4', 'xmm5', 'xmm4'
        for n, compare in enumerate(compares):
            vectors[compare.args[0]] = 'xmm1' if n == 0 else 'xmm0'
            vectors[compare.args[1]] = 'xmm3' if n == 0 else 'xmm2'
            vectors[compare.results[0]] = 'xmm1' if n == 0 else 'xmm0'

        early, late, selected_bases = [], [], []
        for n, (root, delta) in enumerate(bases):
            value = root
            if delta:
                op = operation('add', T.ULONG, [root, fn.constant(T.ULONG, delta)])
                (early if last else late).append(op)
                value = op.results[0]
                offsets.add(value)
                prefer(value, 'rax' if n == 0 else 'rdx')
            selected_bases.append(value)
        first_mask = intrinsic('_mm256_movemask_epi8', compares[0].results)
        first_word = operation('cast', T.UINT, first_mask.results)
        condition = operation('ne', T.INT, [first_word.results[0], fn.constant(T.UINT, 0xffffffff)])
        second_mask = intrinsic('_mm256_movemask_epi8', compares[1].results)
        second_word = operation('cast', T.UINT, second_mask.results)
        selected = operation('select', T.UINT, [condition.results[0], first_word.results[0], second_word.results[0]])
        base = operation('select', T.ULONG, [condition.results[0], *selected_bases])
        inverted = operation('not', T.UINT, selected.results)
        cold.ops = [*early, compares[0], first_mask, first_word, condition,
                    compares[1], second_mask, second_word, selected, *late, base, inverted]
        cold.term = Term('jmp', targets=(join,))
        if not last:
            fn.layout_owners[block.term.targets[0]] = cold
        for phi, value in ((base_phi, base.results[0]), (mask_phi, inverted.results[0])):
            keep = [(b, v) for b, v in zip(phi.attrs['blocks'], phi.args) if b not in (yes, no)]
            keep.append((cold, value))
            phi.attrs['blocks'], phi.args = [b for b, _ in keep], tuple(v for _, v in keep)
        prefer(first_word.results[0], 'r8' if last else 'rdx')
        prefer(second_word.results[0], 'rcx')
        prefer(selected.results[0], 'rcx')
        prefer(inverted.results[0], 'rcx')
        prefer(base.results[0], 'rdx' if last else 'rax')
        (prefer_second if last else prefer_first).add(base)
        prefer_second.add(selected)
        removed.update((yes, no))
    fn.blocks = [b for b in fn.blocks if b not in removed]
    fn.layout_owners = {b: p for b, p in fn.layout_owners.items() if b not in removed and p not in removed}
    if last_cold is not None:
        fn.layout_owners[head.term.targets[0]] = last_cold
    fn.large_compare_pairs = True
    return True


def ordered(block, roots, fn):
    original, seen, result = set(block.ops), set(), []
    def visit(op):
        if op not in original or op in seen:
            return
        for value in op.args:
            visit(fn.defs[value])
        seen.add(op)
        result.append(op)
    for op in roots:
        visit(op)
    for op in block.ops:
        visit(op)
    block.ops = result
