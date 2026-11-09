"""Function recipe for the AVX2 strcpy entry's bounded copy region."""
from types import SimpleNamespace

from . import types as T
from .ir import Term
from .prefixes import string_bare


def intrinsic(block, name):
    return next(op for op in block.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == name)


def aligned_load(block):
    return next(op for op in block.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'load'
                and op.attrs.get('alignment', 1) >= 32)


def match_copy(fn, returned=None):
    if len(fn.params) != 2 or fn.type.base.kind != 'ptr' or fn.entry.term.kind != 'br':
        return None
    hubs = [(b, fn.defs[b.term.args[0]]) for b in fn.blocks if b.term.kind == 'br'
            and fn.defs[b.term.args[0]].kind == 'phi' and b.term.args[0].type.kind == 'ptr']
    if len(hubs) != 1:
        return None
    hub, status = hubs[0]
    continuing = [(p, v) for p, v in zip(status.attrs['blocks'], status.args)
                  if fn.defs[v].kind != 'constant']
    if len(continuing) != 1:
        return None
    last, end = continuing[0]
    call_block, done = hub.term.targets
    if done.term.kind != 'ret' or done.term.args != (fn.params[0] if returned is None else returned,):
        return None
    fast, cold = fn.entry.term.targets
    if fast.term.kind != cold.term.kind or fast.term.kind != 'jmp' or fast.term.targets != cold.term.targets:
        return None
    joined = fast.term.targets[0]
    mask = next(op for op in joined.ops if op.kind == 'phi' and op.results[0].type == T.UINT)
    hit, aligned = joined.term.targets
    first16, first8_guard = hit.term.targets
    first8, first4_guard = first8_guard.term.targets
    first4, first2_guard = first4_guard.term.targets
    first2, first1 = first2_guard.term.targets
    second_hit, miss = aligned.term.targets
    second_small, second32 = second_hit.term.targets
    second16, second8_guard = second_small.term.targets
    second8, second4_guard = second8_guard.term.targets
    second4, second2_guard = second4_guard.term.targets
    second2, second1_guard = second2_guard.term.targets
    second1 = second1_guard.term.targets[0]
    third_hit, third_miss = miss.term.targets
    fourth_hit, fourth_miss = third_miss.term.targets
    if fourth_miss is not last or last.term.targets != (hub,):
        return None
    load = intrinsic(aligned, 'load')
    cursor = string_bare(fn, load.args[0])
    cursor_op = fn.defs[cursor]
    if (cursor_op.kind != 'ptradd' or fn.defs[cursor_op.args[1]].kind != 'constant'
            or fn.defs[cursor_op.args[1]].attrs['value'] != 32):
        return None
    base = string_bare(fn, cursor_op.args[0])
    pointers = [cursor, string_bare(fn, aligned_load(miss).args[0]),
                string_bare(fn, aligned_load(third_miss).args[0])]
    for pointer, offset in zip(pointers, (32, 64, 96)):
        op = fn.defs[pointer]
        if (op.kind != 'ptradd' or string_bare(fn, op.args[0]) != base
                or fn.defs[op.args[1]].kind != 'constant'
                or fn.defs[op.args[1]].attrs['value'] != offset):
            return None
    return SimpleNamespace(hub=hub, status=status, last=last, end=end, call_block=call_block, done=done,
                           fast=fast, cold=cold, joined=joined, mask=mask, hit=hit, aligned=aligned,
                           first16=first16, first8_guard=first8_guard, first8=first8,
                           first4_guard=first4_guard, first4=first4, first2_guard=first2_guard,
                           first2=first2, first1=first1, second_hit=second_hit, miss=miss,
                           second_small=second_small, second32=second32, second16=second16,
                           second8_guard=second8_guard, second8=second8, second4_guard=second4_guard,
                           second4=second4, second2_guard=second2_guard, second2=second2,
                           second1_guard=second1_guard, second1=second1,
                           third_hit=third_hit, third_miss=third_miss,
                           fourth_hit=fourth_hit, fourth_miss=fourth_miss,
                           base=base, pointers=pointers)


def prepare_copy(fn, region=None):
    r = match_copy(fn) if region is None else region
    if r is None:
        return False
    registers = fn.induction_registers = dict(getattr(fn, 'induction_registers', {}))
    vectors = fn.vector_registers = dict(getattr(fn, 'vector_registers', {}))

    def prefer(value, physical):
        registers[value] = physical
        while fn.defs[value].kind == 'cast':
            value = fn.defs[value].args[0]
            registers[value] = physical

    def add(block, kind, type_, args=(), attrs=None, physical=None):
        op = fn.make(kind, [type_], args, attrs)
        block.ops.append(op)
        if physical:
            (vectors if type_.kind == 'vector' else registers)[op.results[0]] = physical
        return op.results[0]

    # Every completed bounded copy goes straight to the public return.
    # Only the last clean vector can supply the nonzero continuation cursor.
    for parent, value in zip(r.status.attrs['blocks'], r.status.args):
        target = r.done if fn.defs[value].kind == 'constant' else r.call_block
        parent.term.targets = tuple(target if b is r.hub else b for b in parent.term.targets)
    for op in r.call_block.ops:
        op.args = tuple(r.end if v == r.status.results[0] else v for v in op.args)

    # The fast and recovered masks test locally. Their phi belongs to the
    # hit block, so the no-hit path needs no join or register copy.
    r.joined.ops.remove(r.mask)
    fn.entry.ops.extend(r.joined.ops)
    r.joined.ops = []
    r.hit.ops.insert(0, r.mask)
    for block in (r.fast, r.cold):
        incoming = r.mask.args[r.mask.attrs['blocks'].index(block)]
        block.term = Term('br', (incoming,), (r.hit, r.aligned))
    fn.blocks = fn.reachable()
    prefer(r.base, 'rdx')
    for pointer in r.pointers:
        prefer(pointer, 'rax')
    prefer(r.end, 'rdx')
    first_count = intrinsic(r.hit, 'ctz_nonzero')
    for value in (intrinsic(r.fast, 'movemask8').results[0], intrinsic(r.cold, 'movemask8').results[0],
                  r.mask.results[0], first_count.results[0]):
        prefer(value, 'rax')
    for block in (r.aligned, r.miss, r.third_miss):
        prefer(intrinsic(block, 'movemask8').results[0], 'rcx')
    second_count = intrinsic(r.second_hit, 'ctz_nonzero')
    distance = next(op for op in r.second_hit.ops if op.kind == 'sub')
    length = fn.defs[fn.defs[r.second_hit.term.args[0]].args[0]]
    sum_ = fn.defs[length.args[0]]
    prefer(second_count.results[0], 'rdx')
    prefer(sum_.results[0], 'rcx')
    prefer(length.results[0], 'rcx')
    prefer(distance.results[0], 'rax')
    sum_.attrs['copy_count'] = True
    fn.retained_offsets = {length.results[0]}
    fn.word_return_bases = set(r.pointers)
    vector_colors = [(intrinsic(r.fast, 'load'), 0), (intrinsic(r.fast, 'cmpeq_epi8'), 0),
                     (intrinsic(r.cold, 'cmpeq_epi8'), 0),
                     (intrinsic(r.aligned, 'load'), 1), (intrinsic(r.aligned, 'cmpeq_epi8'), 2),
                     (aligned_load(r.miss), 1), (intrinsic(r.miss, 'cmpeq_epi8'), 0),
                     (aligned_load(r.third_miss), 0), (intrinsic(r.third_miss, 'cmpeq_epi8'), 1)]
    prefix_load = next(op for op in r.miss.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'load'
                       and string_bare(fn, op.args[0]) == fn.params[1])
    vector_colors.append((prefix_load, 2))
    for op, color in vector_colors:
        vectors[op.results[0]] = 'ymm%d' % color

    # Hits in the final two vectors share their cursor/mask and final store.
    tail = fn.block('copy.short.tail')
    masks = [block.term.args[0] for block in (r.miss, r.third_miss)]
    pointer = add(tail, 'phi', r.pointers[1].type, r.pointers[1:],
                  {'blocks': [r.third_hit, r.fourth_hit]}, 'rax')
    mask = add(tail, 'phi', T.UINT, masks, {'blocks': [r.third_hit, r.fourth_hit]}, 'rcx')
    old_count = intrinsic(r.third_hit, 'ctz_nonzero')
    count = add(tail, 'intrinsic', T.INT, [mask], dict(old_count.attrs), 'rcx')
    p = add(tail, 'cast', T.LONG, [pointer])
    s = add(tail, 'cast', T.LONG, [fn.params[1]])
    delta = add(tail, 'sub', T.LONG, [p, s], physical='rax')
    extended = add(tail, 'cast', T.LONG, [count])
    offset = add(tail, 'add', T.LONG, [delta, extended], {'copy_count': True}, 'rax')
    old_load = intrinsic(r.third_hit, 'load')
    old_store = next(op for op in r.third_hit.ops if 'write' in op.attrs.get('memory', ''))
    src = add(tail, 'ptradd', r.pointers[1].type, [fn.params[1], offset])
    src = add(tail, 'ptradd', r.pointers[1].type, [src, fn.constant(T.LONG, -31)])
    cast = add(tail, 'cast', old_load.args[0].type, [src])
    loaded = add(tail, 'intrinsic', T.V256, [cast], dict(old_load.attrs), 'ymm0')
    dst = add(tail, 'ptradd', old_store.args[0].type, [fn.params[0], offset])
    dst = add(tail, 'ptradd', old_store.args[0].type, [dst, fn.constant(T.LONG, -31)])
    cast = add(tail, 'cast', old_store.args[0].type, [dst])
    tail.ops.append(fn.make('intrinsic', args=[cast, loaded], attrs=dict(old_store.attrs)))
    tail.term = Term('ret', (fn.params[0],))
    for block in (r.third_hit, r.fourth_hit):
        block.ops = []
        block.term = Term('jmp', targets=(tail,))
    fn.retained_offsets.add(offset)
    fn.avx_copy_zero_groups = [([intrinsic(r.fast, 'cmpeq_epi8').results[0]], 0),
                               ([intrinsic(r.cold, 'cmpeq_epi8').results[0]], 0),
                               ([intrinsic(r.aligned, 'cmpeq_epi8').results[0],
                                 intrinsic(r.miss, 'cmpeq_epi8').results[0]], 0),
                               ([intrinsic(r.third_miss, 'cmpeq_epi8').results[0]], 1)]
    fn.avx_copy_breaks = {second_count.results[0]}
    for block in (r.second8, r.second4, r.second1):
        prefer(next(op for op in block.ops if op.kind == 'load').results[0], 'rax')
    for block in (r.second8, r.second4):
        prefer([op for op in block.ops if op.kind == 'load'][1].results[0], 'rdx')
    for block in (r.first8, r.first4, r.first2):
        prefer(next(op for op in block.ops if op.kind == 'load').results[0], 'rcx')

    # The two small-copy families can share their two-byte stores. One has
    # a direct CTZ index; the other adds the aligned block's distance first.
    old_index = next(op.args[1] for op in r.first2.ops if op.kind == 'ptradd'
                     and string_bare(fn, op.args[0]) == fn.params[1])
    r.second2.ops = []
    count_value = next(op for op in r.second_hit.ops if op.kind == 'cast' and op.args == second_count.results)
    c = add(r.second2, 'cast', T.LONG, count_value.results)
    total = add(r.second2, 'add', T.LONG, [distance.results[0], c], {'copy_count': True}, 'rax')
    r.second2.term = Term('jmp', targets=(r.first2,))
    early_count = next(op for op in r.hit.ops if op.kind == 'cast' and op.args == first_count.results)
    direct = add(r.first2_guard, 'cast', T.LONG, early_count.results)
    index = fn.make('phi', [T.LONG], [direct, total], {'blocks': [r.first2_guard, r.second2]})
    r.first2.ops.insert(0, index)
    prefer(index.results[0], 'rax')
    for op in r.first2.ops:
        if op is not index:
            op.args = tuple(index.results[0] if v == old_index else v for v in op.args)
    r.second1_guard.term.targets = r.first1, r.done
    r.second1.ops = []
    r.second1.term = Term('jmp', targets=(r.first1,))
    fn.shared_index_return = True
    fn.avx_copy_order = [fn.entry, r.fast, r.joined, r.hit, r.first16, r.cold, r.aligned,
                         r.second_hit, r.second_small, r.second16, r.first8_guard, r.first8,
                         r.first4_guard, r.first4, r.miss, r.third_miss, tail, r.second32,
                         r.first2_guard, r.first2, r.second8_guard, r.second8, r.second4_guard, r.second4,
                         r.last, r.call_block, r.second2_guard, r.second2, r.second1_guard,
                         r.first1, r.done, r.third_hit, r.fourth_hit, r.second1]
    fn.avx_copy_bases = set(r.pointers)
    fn.avx_copy_address_loads = [(pointer, aligned_load(block).results[0])
                                 for pointer, block in zip(r.pointers, (r.aligned, r.miss, r.third_miss))]
    fn.avx_copy_tail = tail
    fn.count_first_blocks = (tail,)
    fn.avx_copy_short = True
    fn.blocks = fn.reachable()
    fn.avx_copy_order = [b for b in fn.avx_copy_order if b in fn.blocks]
    return True
