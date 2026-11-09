"""Unroll short vector-table reductions and retain their scalar remainder."""
from . import types as T
from .builtins import BUILTINS
from .interproc import callee
from .ir import Term


def prepare_vector_tables(fn, functions):
    if (fn.type.base != T.ULONG or len(fn.params) != 6 or len(fn.blocks) != 3
            or any(v.type.kind != 'ptr' for v in fn.params[:2])
            or any(v.type != T.V128 for v in fn.params[2:4])
            or fn.params[4].type != T.ULONG or fn.params[5].type != T.INT
            or getattr(fn, 'parameter_masks', {}).get(fn.params[4], 64) > 63
            or getattr(fn, 'parameter_masks', {}).get(fn.params[5], 2) > 1
            or len(fn.allocas) != 2
            or any(op.attrs['size'] != 32 or op.attrs['alignment'] != 32 for op in fn.allocas)
            or any(op.kind == 'asm' or op.attrs.get('volatile') for b in fn.blocks for op in b.ops)):
        return False

    def bare(value):
        while fn.defs[value].kind == 'cast':
            value = fn.defs[value].args[0]
        return value

    def constant(value):
        op = fn.defs[value]
        return op.attrs.get('value') if op.kind == 'constant' else None

    def builtin(value, kind):
        op = fn.defs[value]
        return op if op.kind == 'intrinsic' and op.attrs['builtin'].op == kind else None

    entry = fn.entry
    if entry.term.kind != 'br' or entry.term.args != (fn.params[4],):
        return False
    old_head, end = entry.term.targets
    if (old_head.term.kind != 'br' or old_head.term.targets != (old_head, end)
            or end.term.kind != 'ret' or len(end.term.args) != 1):
        return False
    call = fn.defs[end.term.args[0]]
    if (call.kind != 'call' or len(call.args) != 6 or call not in end.ops
            or tuple(bare(v) for v in (*call.args[1:3], call.args[-1])) != (*fn.params[:2], fn.params[5])):
        return False
    target = functions.get(callee(fn, call.args[0]))
    if target is None or not getattr(target, 'vector_classification', False):
        return False
    low = builtin(call.args[3], 'truncate')
    high = builtin(call.args[4], 'extract128')
    if low is None or high is None or low.args != high.args[:1] or constant(high.args[1]) != 1:
        return False
    returned = fn.defs[low.args[0]]
    if returned.kind != 'phi' or returned.attrs['blocks'] != [entry, old_head]:
        return False
    phis = [op for op in old_head.ops if op.kind == 'phi']
    index = next((op for op in phis if op.results[0].type == T.ULONG), None)
    accum = next((op for op in phis if op.results[0].type == T.V256), None)
    if len(phis) != 2 or index is None or accum is None:
        return False
    if (any(op.attrs['blocks'] != [entry, old_head] for op in phis) or constant(index.args[0]) != 0
            or returned.args != accum.args):
        return False
    step = fn.defs[index.args[1]]
    test = fn.defs[old_head.term.args[0]]
    if (step.kind != 'add' or step.args[0] != index.results[0] or constant(step.args[1]) != 1
            or test.kind != 'lt' or test.args != (step.results[0], fn.params[4])):
        return False
    reduction = builtin(accum.args[1], 'or')
    if reduction is None or accum.results[0] not in reduction.args:
        return False
    term = next(v for v in reduction.args if v != accum.results[0])
    contribution = builtin(term, 'and')
    if contribution is None:
        return False
    compare = next((builtin(v, 'cmpeq_epi8') for v in contribution.args if builtin(v, 'cmpeq_epi8') is not None), None)
    if compare is None:
        return False
    value_splat = builtin(next(v for v in contribution.args if v != compare.results[0]), 'splat8')
    lookup = next((v for v in compare.args if isinstance(constant(v), bytes)), None)
    if lookup is None or lookup.type != T.V256 or value_splat is None:
        return False
    position_splat = builtin(next(v for v in compare.args if v != lookup), 'splat8')
    if position_splat is None:
        return False
    loads = [fn.defs[bare(op.args[0])] for op in (position_splat, value_splat)]
    if any(op.kind != 'load' or op.attrs['width'] != 1 or op.results[0].type != T.UCHAR for op in loads):
        return False
    addresses = [fn.defs[op.args[0]] for op in loads]
    roots = {op.results[0] for op in fn.allocas}
    if (any(op.kind != 'ptradd' or bare(op.args[0]) not in roots
            or bare(op.args[1]) != index.results[0] for op in addresses)
            or {bare(op.args[0]) for op in addresses} != roots):
        return False
    # Only these two byte loads and the pure reduction may occur in the
    # loop. In particular no write may separate a load from its broadcast.
    expected = {*phis, step, test, reduction, contribution, compare, position_splat, value_splat,
                *loads, *addresses, *(fn.defs[op.args[0]] for op in (position_splat, value_splat)),
                *(fn.defs[op.args[1]] for op in addresses)}
    if set(old_head.ops) != expected:
        return False
    stores = [op for op in entry.ops if op.attrs.get('memory') == 'write']
    if (len(stores) != 2 or any(op.kind != 'intrinsic' or op.attrs['builtin'].op != 'store'
            or op.attrs['width'] != 32 for op in stores)
            or {bare(op.args[0]) for op in stores} != roots
            or any(op.kind == 'call' or op.attrs.get('memory') for b in fn.blocks for op in b.ops
                   if op not in [*stores, *loads, call])):
        return False
    position_store, value_store = [next(op for op in stores if bare(op.args[0]) == bare(address.args[0]))
                                   for address in addresses]
    positions = builtin(position_store.args[1], 'or')
    values = builtin(value_store.args[1], 'shuffle_epi8')
    if positions is None or values is None:
        return False
    mask_low, mask_high = [builtin(v, 'and') for v in positions.args]
    if mask_low is None or mask_high is None:
        return False
    shift_high = builtin(mask_high.args[0], 'lshr16')
    combined = builtin(mask_low.args[0], 'insert128')
    if (combined is None or shift_high is None or shift_high.args[0] != combined.results[0]
            or mask_low.args[0] != combined.results[0]
            or any(not isinstance(constant(op.args[1]), bytes) or len(constant(op.args[1])) != 32
                   or constant(op.args[1]) != constant(op.args[1])[:8]*4 for op in (mask_low, mask_high))):
        return False
    value_index = builtin(values.args[1], 'and')
    shift_value = builtin(value_index.args[0], 'lshr16') if value_index is not None else None
    initial = builtin(accum.args[0], 'widen_undef')
    inserted = builtin(initial.args[0], 'insert_low32') if initial is not None else None
    if (value_index is None or shift_value is None or shift_value.args[0] != combined.results[0]
            or inserted is None or inserted.args != (fn.params[5],)
            or any(op not in entry.ops for op in (combined, mask_low, shift_high, mask_high, positions,
                                                  shift_value, value_index, values, inserted, initial))):
        return False
    if fn.defs[call.args[0]] not in end.ops or fn.defs[call.args[0]].kind != 'symbol':
        return False

    initial_value = initial.results[0]
    p, v = (op.args[0] for op in addresses)
    # Put the most recently consumed table first in the aligned frame.
    fn.allocas = [next(op for op in fn.allocas if op.results[0] == bare(base)) for base in (v, p)]
    fn.vector_table_build = True
    fn.retain_vector_constants = True
    scalars = fn.induction_registers = dict(getattr(fn, 'induction_registers', {}))
    vectors = fn.vector_registers = dict(getattr(fn, 'vector_registers', {}))

    def prefer(value, register):
        (vectors if value.type.kind == 'vector' else scalars)[value] = register
        return value

    def add(block, kind, type_, args=(), attrs=None, register=None):
        op = fn.make(kind, [type_], args, attrs)
        block.ops.append(op)
        return prefer(op.results[0], register) if register else op.results[0]

    def intrinsic(block, name, args, register):
        builtin_ = BUILTINS[name]
        return add(block, 'intrinsic', builtin_.result, args, {'builtin': builtin_}, register)

    def phi(block, type_, blocks, args, register):
        return add(block, 'phi', type_, args, {'blocks': blocks}, register)

    def jump(block, target_):
        block.term = Term('jmp', targets=(target_,))

    def branch(block, value, yes, no):
        block.term = Term('br', (value,), (yes, no))

    def scalar(block, number, register):
        return add(block, 'scalar_constant', T.ULONG, [fn.constant(T.ULONG, number)], register=register)

    def table(block):
        return add(block, 'vector_constant', T.V256, [lookup], register='xmm1')

    for op, register in ((combined, 'xmm0'), (mask_low, 'xmm1'), (shift_high, 'xmm2'),
                         (mask_high, 'xmm2'), (positions, 'xmm1'), (shift_value, 'xmm0'),
                         (value_index, 'xmm0'), (values, 'xmm2'), (inserted, 'xmm0'), (initial, 'xmm0')):
        prefer(op.results[0], register)
    for op, register in ((mask_low, 'xmm1'), (mask_high, 'xmm3')):
        materialized = fn.make('vector_constant', [T.V256], [op.args[1]], {'splat64': True})
        prefer(materialized.results[0], register)
        entry.ops.insert(entry.ops.index(op), materialized)
        op.args = op.args[0], materialized.results[0]
    positions.args = positions.args[::-1]
    entry.ops.remove(inserted)
    entry.ops.remove(initial)
    at = entry.ops.index(position_store)+1
    entry.ops[at:at] = [inserted, initial]

    count = fn.block('tables.count')
    small = fn.block('tables.small')
    big = fn.block('tables.big')
    head = fn.block('tables.group')
    done = fn.block('tables.group.done')
    rest = fn.block('tables.rest')
    tail = fn.block('tables.tail')
    branch(entry, fn.params[4], count, end)
    truncated = add(count, 'cast', T.UINT, [fn.params[4]], register='rax')
    rem32 = add(count, 'and', T.UINT, [truncated, fn.constant(T.UINT, 3)], register='rax')
    remainder = add(count, 'cast', T.ULONG, [rem32], register='rax')
    enough = add(count, 'ge', T.INT, [fn.params[4], fn.constant(T.ULONG, 4)])
    branch(count, enough, big, small)
    small_k = scalar(small, 0, 'r8')
    jump(small, rest)
    bound = add(big, 'and', T.ULONG, [fn.params[4], fn.constant(T.ULONG, 60)], register='rdx')
    first = scalar(big, 0, 'r8')
    iota = table(big)
    jump(big, head)
    k = phi(head, T.ULONG, [big, head], [first, first], 'r8')
    acc = phi(head, T.V256, [big, head], [initial_value, initial_value], 'xmm0')
    records = []

    def lane(block, index_, offset_, lookup_, base_p, base_v, compare_reg='xmm2', bits_reg='xmm3', byte_reg='r9'):
        item = add(block, 'add', T.ULONG, [index_, fn.constant(T.ULONG, offset_)]) if offset_ else index_
        pp = add(block, 'ptradd', p.type, [base_p, item])
        pos_byte = add(block, 'load', T.UCHAR, [pp],
                       {'memory': 'read', 'width': 1, 'alignment': 1, 'selection_folded': True})
        signed = add(block, 'cast', T.CHAR, [pos_byte], {'selection_folded': True})
        pos = add(block, 'intrinsic', T.V256, [signed],
                  {'builtin': BUILTINS['_mm256_set1_epi8'], 'memory_broadcast': pp}, compare_reg)
        eq = intrinsic(block, '_mm256_cmpeq_epi8', [pos, lookup_], compare_reg)
        vp = add(block, 'ptradd', v.type, [base_v, item])
        byte_ = add(block, 'load', T.UCHAR, [vp], {'memory': 'read', 'width': 1, 'alignment': 1}, byte_reg)
        cast = add(block, 'cast', T.CHAR, [byte_])
        bits = intrinsic(block, '_mm256_set1_epi8', [cast], bits_reg)
        contribution_ = intrinsic(block, '_mm256_and_si256', [eq, bits], compare_reg)
        records.append({'block': block, 'position': pos, 'byte': byte_, 'byte_address': vp, 'bits': bits})
        return contribution_

    a = lane(head, k, 0, iota, p, v)
    first_acc = intrinsic(head, '_mm256_or_si256', [acc, a], 'xmm0')
    b = lane(head, k, 1, iota, p, v, byte_reg='r10')
    c = lane(head, k, 2, iota, p, v, 'xmm3', 'xmm4')
    prefer(records[2]['position'], 'xmm4')
    pair = intrinsic(head, '_mm256_or_si256', [b, c], 'xmm2')
    both = intrinsic(head, '_mm256_or_si256', [first_acc, pair], 'xmm0')
    d = lane(head, k, 3, iota, p, v)
    result = intrinsic(head, '_mm256_or_si256', [both, d], 'xmm0')
    next_k = add(head, 'add', T.ULONG, [k, fn.constant(T.ULONG, 4)], register='r8')
    condition = add(head, 'ne', T.INT, [bound, next_k])
    branch(head, condition, head, done)
    fn.defs[k].args = first, next_k
    fn.defs[acc].args = initial_value, result
    branch(done, remainder, rest, end)

    start = phi(rest, T.ULONG, [small, done], [small_k, next_k], 'r8')
    initial_acc = phi(rest, T.V256, [small, done], [initial_value, result], 'xmm0')
    vp = add(rest, 'ptradd', v.type, [v, start], register='rdx')
    pp = add(rest, 'ptradd', p.type, [p, start], register='r8')
    first_tail = scalar(rest, 0, 'r9')
    tail_table = table(rest)
    jump(rest, tail)
    j = phi(tail, T.ULONG, [rest, tail], [first_tail, first_tail], 'r9')
    accumulated = phi(tail, T.V256, [rest, tail], [initial_acc, initial_acc], 'xmm0')
    # Retain the two invariant addresses as loop-carried values so their
    # stack bases are materialized once before the scalar remainder.
    pbase = phi(tail, p.type, [rest, tail], [pp, pp], 'r8')
    vbase = phi(tail, v.type, [rest, tail], [vp, vp], 'rdx')
    fn.defs[pbase].args = pp, pbase
    fn.defs[vbase].args = vp, vbase
    last = lane(tail, j, 0, tail_table, pbase, vbase, byte_reg='r10')
    tail_result = intrinsic(tail, '_mm256_or_si256', [accumulated, last], 'xmm0')
    next_tail = add(tail, 'add', T.ULONG, [j, fn.constant(T.ULONG, 1)], register='r9')
    again = add(tail, 'ne', T.INT, [remainder, next_tail])
    branch(tail, again, tail, end)
    fn.defs[j].args = first_tail, next_tail
    fn.defs[accumulated].args = initial_acc, tail_result
    returned.args = initial_value, result, tail_result
    returned.attrs['blocks'] = [entry, done, tail]
    prefer(returned.results[0], 'xmm0')
    prefer(low.results[0], 'xmm0')
    prefer(high.results[0], 'xmm1')
    fn.blocks.remove(old_head)
    fn.layout_owners = {**getattr(fn, 'layout_owners', {}), end: tail, big: small, head: big,
                        done: head, rest: done, tail: rest}

    # Load the next independent value byte while the first vector update
    # is in flight. Its pure address computations move with it.
    early = fn.defs[records[1]['byte']]
    address = fn.defs[records[1]['byte_address']]
    item = fn.defs[address.args[1]]
    for op in (item, address, early):
        head.ops.remove(op)
    at = head.ops.index(fn.defs[records[0]['byte']])+1
    head.ops[at:at] = [item, address, early]
    fn.table_build_group = head
    fn.table_build_early_position = records[2]['position']
    fn.table_build_pair_bits = records[1]['bits']
    fn.table_build_remainder = rest
    fn.table_build_pointers = [vp, pp]
    return True
