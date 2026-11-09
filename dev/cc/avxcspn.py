"""SSA recipe for the two AVX2 strcspn kernels.

Both scans distinguish an explicit set match from the text's terminating NUL.
The one-table loop is unrolled eight times. Four two-table blocks share a
single NUL count and a return suffix, retaining the independent match tests.
"""
from types import SimpleNamespace

from . import types as T
from .avxspan import recovery_prefixes, configure_prefixes, intrinsic, builtin
from .ir import Term
from .prefixes import string_bare
from .tails import converging_unary


def index_phi(block):
    return next(op for op in block.ops if op.kind == 'phi' and op.results[0].type == T.UINT)


def index_paths(head, paired):
    """The explicit match, NUL flag, NUL count and resulting index phi."""
    if paired:
        count, flag = head.term.targets
        zero, joined = flag.term.targets
    else:
        flag, joined = head.term.targets
        zero = flag.term.targets[0]
        count = None
    return SimpleNamespace(head=head, count=count, flag=flag, zero=zero,
                           joined=joined, index=index_phi(joined))


def match_cspn(fn):
    prefixes = recovery_prefixes(fn)
    if prefixes is None:
        return None
    first_head = index_paths(prefixes[1]['join'], False)
    second_head = index_paths(prefixes[3]['join'], True)
    public, setup = first_head.joined.term.targets
    if setup.term.kind != 'jmp':
        return None
    body = setup.term.targets[0]
    step, stopped = body.term.targets
    if step.term.kind != 'br' or step.term.targets[0] is not body:
        return None
    single = index_paths(stopped, False)
    pointer = next(op for op in body.ops if op.kind == 'phi' and op.results[0].type.kind == 'ptr')
    base = pointer.args[pointer.attrs['blocks'].index(setup)]
    after = pointer.args[pointer.attrs['blocks'].index(step)]
    limit = fn.defs[fn.defs[step.term.args[0]].args[1]]
    if limit.kind != 'constant' or limit.attrs['value'] % 8:
        return None
    done = step.term.targets[1]
    first_call = next(b for b in done.term.targets if any(op.kind == 'call' for op in b.ops))
    call = next(op for op in first_call.ops if op.kind == 'call')
    count_phi = fn.defs[fn.defs[call.args[5]].args[0]]
    old_count = next(fn.defs[string_bare(fn, v)] for v in count_phi.args
                     if fn.defs[string_bare(fn, v)].kind == 'intrinsic')
    pair_blocks, pair_pointers = [], []
    block = second_head.joined.term.targets[1]
    for offset in range(4):
        paths = index_paths(block, True)
        returned, following = paths.joined.term.targets
        compare = fn.defs[paths.joined.term.args[0]]
        if (returned.term.kind != 'ret' or compare.kind != 'lt'
                or fn.defs[compare.args[1]].kind != 'constant'
                or fn.defs[compare.args[1]].attrs['value'] != 16):
            return None
        pointer_value = string_bare(fn, intrinsic(block, 'load').args[0])
        if offset:
            op = fn.defs[pointer_value]
            if (op.kind != 'ptradd' or op.args[0] != pair_pointers[0]
                    or fn.defs[op.args[1]].kind != 'constant'
                    or fn.defs[op.args[1]].attrs['value'] != 16 * offset):
                return None
        paths.returned = returned
        pair_blocks.append(paths)
        pair_pointers.append(pointer_value)
        block = following
    fallback = block
    pair_call = next(b for b in fallback.term.targets if any(op.kind == 'call' for op in b.ops))
    call = next(op for op in pair_call.ops if op.kind == 'call')
    endpoint, final_count_cast = fn.defs[call.args[1]], fn.defs[call.args[5]]
    if (endpoint.kind != 'ptradd' or endpoint.args[0] != pair_pointers[0]
            or fn.defs[endpoint.args[1]].kind != 'constant'
            or fn.defs[endpoint.args[1]].attrs['value'] != 64):
        return None
    probe1, probe2 = (prefixes[n]['join'].term.targets[1] for n in (0, 2))
    long_call = next(b for b in probe2.term.targets if any(op.kind == 'call' for op in b.ops))
    return SimpleNamespace(prefixes=prefixes, first_head=first_head, second_head=second_head,
                           public=public, setup=setup, body=body, step=step, single=single,
                           pointer=pointer.results[0], base=base, after=after, limit=limit.attrs['value'],
                           loaded=intrinsic(body, 'load'), flag=intrinsic(body, 'cmpistra'),
                           done=done, first_call=first_call, count_phi=count_phi, old_count=old_count,
                           pairs=pair_blocks, pair_pointers=pair_pointers,
                           fallback=fallback, pair_call=pair_call, endpoint=endpoint,
                           final_count_cast=final_count_cast, probe1=probe1, probe2=probe2,
                           long_call=long_call)


def prepare_cspn(fn):
    r = match_cspn(fn)
    if r is None:
        return False
    configure_prefixes(fn, r)
    fn.avx_span = fn.avx_cspn = True
    registers, vectors = fn.induction_registers, fn.vector_registers

    def prefer(value, physical):
        registers[value] = registers[string_bare(fn, value)] = physical

    def add(block, kind, type_, args=(), attrs=None, physical=None):
        op = fn.make(kind, [type_], args, attrs)
        block.ops.append(op)
        if physical:
            (vectors if type_.kind == 'vector' else registers)[op.results[0]] = physical
        return op.results[0]

    set1, text1, set2, text2 = r.prefixes
    # The empty-set path also calls a continuation, so copy the ABI inputs
    # before testing the set's first byte.
    guard = set1['guard']
    copies = [op for op in guard.ops if op.attrs.get('parameter_copy')]
    for op in copies:
        guard.ops.remove(op)
    for block in fn.blocks:
        for op in block.ops:
            op.args = tuple(fn.avx_parameters.get(v, v) for v in op.args)
        block.term.args = tuple(fn.avx_parameters.get(v, v) for v in block.term.args)
    fn.entry.ops[0:0] = copies
    for op in text1['join'].ops:
        if op.kind == 'phi' and op.results[0].type == T.ULONG:
            prefer(op.results[0], 'rdx')
    for op in text1['root'].ops:
        if op.kind == 'sub':
            prefer(op.results[0], 'rdx')
    for block in (text2['guard'], text2['root']):
        for op in block.ops:
            if op.results and op.results[0].type.kind == 'int' and op.kind in ('and', 'sub'):
                prefer(op.results[0], 'rdx' if op.kind == 'sub' else 'rcx')
    for op in text2['join'].ops:
        if op.kind == 'phi' and op.results[0].type == T.ULONG:
            prefer(op.results[0], 'rdx')
    prefer(intrinsic(set2['join'], 'movemask8').results[0], 'rax')
    for block in fn.blocks:
        for op in block.ops:
            if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'extract_low32':
                prefer(op.results[0], 'rcx')
    for paths in (r.first_head, r.second_head):
        prefer(paths.index.results[0], 'rcx')
        prefer(intrinsic(paths.zero, 'ctz_nonzero').results[0], 'rcx')
        if paths.count is not None:
            prefer(intrinsic(paths.count, 'ctz_nonzero').results[0], 'rcx')
        else:
            prefer(intrinsic(paths.head, 'cmpistri').results[0], 'rcx')
    prefer(intrinsic(r.first_head.zero, 'movemask8').results[0], 'rax')
    first_zero_result = intrinsic(r.first_head.zero, 'cmpeq_epi8').results[0]
    vectors[first_zero_result] = 'xmm1'
    vectors[intrinsic(r.second_head.zero, 'cmpeq_epi8').results[0]] = 'xmm0'
    for setup in (r.setup, r.pairs[0].head):
        for op in setup.ops:
            if (op.kind == 'ptradd' and string_bare(fn, op.args[0]) == fn.avx_parameters[fn.params[0]]
                    and fn.defs[op.args[1]].kind == 'constant' and fn.defs[op.args[1]].attrs['value'] == 16):
                prefer(op.results[0], 'rcx')

    def sentinel(paths):
        edge = fn.block('span.cspn.sentinel')
        edge.term = Term('jmp', targets=(paths.joined,))
        paths.flag.term.targets = paths.flag.term.targets[0], edge
        index = paths.index
        at = index.attrs['blocks'].index(paths.flag)
        index.attrs['blocks'][at] = edge
        index.args = tuple(fn.constant(T.UINT, 16) if n == at else v for n, v in enumerate(index.args))
        return edge

    sentinel1, sentinel2 = sentinel(r.first_head), sentinel(r.second_head)
    old_joins, old_blocks = getattr(fn, 'return_joins', ()), set(fn.blocks)
    fn.return_joins = (r.second_head.joined,)
    converging_unary(fn)
    fn.return_joins = old_joins
    common_head = next(b for b in fn.blocks if b not in old_blocks)
    for op in common_head.ops:
        if op.results:
            prefer(op.results[0], 'rcx')
    for paths in (r.first_head, r.second_head):
        fn.defs[paths.joined.term.args[0]].attrs['cspn_index_compare'] = True
    first, second = fn.block('span.cspn.head1.return'), fn.block('span.cspn.head2.return')
    public_phi = next(op for op in r.public.ops if op.kind == 'phi')
    for paths, returned in ((r.first_head, first), (r.second_head, second)):
        value = public_phi.args[public_phi.attrs['blocks'].index(paths.joined)]
        returned.term = Term('ret', (value,))
        paths.joined.term.targets = returned, paths.joined.term.targets[1]

    # Use an offset in RDX and retain the aligned scan origin in RAX.
    limit = r.limit * 16
    end = add(r.setup, 'ptradd', r.base.type, [r.base, fn.constant(T.LONG, limit)], physical='rdi')
    prefer(r.base, 'rax')
    bodies = [fn.block('span.cspn.single.body.%d' % n) for n in range(8)]
    exits = [fn.block('span.cspn.single.exit.%d' % n) for n in range(8)]
    step = fn.block('span.cspn.single.step')
    index = add(bodies[0], 'phi', T.ULONG, attrs={'blocks': [r.setup, step]}, physical='rdx')
    next_index = add(step, 'add', T.ULONG, [index, fn.constant(T.ULONG, 128)], physical='rdx')
    fn.defs[index].args = fn.constant(T.ULONG, 0), next_index
    compare = add(step, 'ne', T.INT, [next_index, fn.constant(T.ULONG, limit)], {'narrow_compare': True})
    step.term = Term('br', (compare,), (bodies[0], r.done))
    pointers, data = [], []
    bases = set(getattr(fn, 'word_return_bases', ()))
    for n, (body, edge) in enumerate(zip(bodies, exits)):
        point = add(body, 'ptradd', r.base.type, [r.base, index])
        if n:
            point = add(body, 'ptradd', r.base.type, [point, fn.constant(T.LONG, 16 * n)])
        cast = add(body, 'cast', r.loaded.args[0].type, [point])
        loaded = add(body, 'intrinsic', T.V128, [cast], dict(r.loaded.attrs), 'xmm1')
        data.append(loaded)
        flag = add(body, 'intrinsic', T.INT, [r.flag.args[0], loaded, r.flag.args[2]], dict(r.flag.attrs))
        body.term = Term('br', (flag,), (bodies[n + 1] if n < 7 else step, edge))
        point = add(edge, 'ptradd', r.base.type, [r.base, index], physical='rax')
        bases.add(point)
        if n:
            point = add(edge, 'ptradd', r.base.type, [point, fn.constant(T.LONG, 16 * n)], physical='rax')
            bases.add(point)
        pointers.append(point)
        edge.term = Term('jmp', targets=(r.single.head,))
    pointer_phi = fn.make('phi', [r.base.type], pointers, {'blocks': exits})
    vector_phi = fn.make('phi', [T.V128], data, {'blocks': exits})
    r.single.head.ops[0:0] = [pointer_phi, vector_phi]
    prefer(pointer_phi.results[0], 'rax')
    vectors[vector_phi.results[0]] = 'xmm1'
    replacements = {r.pointer: pointer_phi.results[0], r.loaded.results[0]: vector_phi.results[0], r.after: end}
    for block in fn.blocks:
        if block in (r.body, r.step, step) or block in bodies or block in exits:
            continue
        for op in block.ops:
            op.args = tuple(replacements.get(a, a) for a in op.args)
            if op.kind == 'phi':
                op.attrs['blocks'] = [step if p is r.step else p for p in op.attrs['blocks']]
    r.setup.term = Term('jmp', targets=(bodies[0],))
    fn.blocks = [b for b in fn.blocks if b not in (r.body, r.step)]

    # Compute the pointer difference once before the flag/index suffix.
    returned = r.single.joined
    distance = next(op for op in returned.ops if op.kind == 'sub')
    operands = [fn.defs[v] for v in distance.args]
    for op in [operands[1], operands[0], distance]:
        returned.ops.remove(op)
    r.single.head.ops[2:2] = [operands[1], operands[0], distance]
    prefer(distance.results[0], 'rax')
    for op in (intrinsic(r.single.head, 'cmpistri'), intrinsic(r.single.zero, 'movemask8'),
               intrinsic(r.single.zero, 'ctz_nonzero'), r.single.index):
        prefer(op.results[0], 'rcx')
    vectors[intrinsic(r.single.zero, 'cmpeq_epi8').results[0]] = 'xmm0'
    index_cast = next(op for op in returned.ops if op.kind == 'cast' and op.args == r.single.index.results)
    index_cast.attrs['force_zero_extension'] = True
    prefer(index_cast.results[0], 'rcx')
    loop_sentinel = sentinel(r.single)

    masked = add(r.done, 'or', T.UINT,
                 [set1['join'].term.args[0], fn.constant(T.UINT, 65536)], physical='r8')
    first_count = add(r.done, 'intrinsic', T.INT, [masked], dict(r.old_count.attrs), 'rdx')
    r.done.term = Term('jmp', targets=(r.first_call,))
    r.count_phi.kind, r.count_phi.args, r.count_phi.attrs = 'cast', (first_count,), {}

    # Explicit matches and NULs meet at a common pointer/index return.
    prefer(r.pair_pointers[0], 'rdi')
    for value in r.pair_pointers[1:]:
        prefer(value, 'rdx')
    flags = [p.flag for p in r.pairs]
    joins = [p.joined for p in r.pairs]
    common_zero, common_return = fn.block('span.cspn.pair.zero'), fn.block('span.cspn.pair.return')
    zero_p = add(common_zero, 'phi', r.base.type, r.pair_pointers, {'blocks': flags}, 'rdx')
    zero_vec = add(common_zero, 'phi', T.V128,
                   [intrinsic(p.head, 'load').results[0] for p in r.pairs], {'blocks': flags}, 'xmm4')
    original_zero = r.pairs[0].zero
    eq = add(common_zero, 'intrinsic', T.V128, [zero_vec, fn.constant(T.V128, bytes(16))],
             dict(intrinsic(original_zero, 'cmpeq_epi8').attrs), 'xmm0')
    mask = add(common_zero, 'intrinsic', T.INT, [eq], dict(intrinsic(original_zero, 'movemask8').attrs), 'rax')
    count = add(common_zero, 'intrinsic', T.INT, [mask], dict(intrinsic(original_zero, 'ctz_nonzero').attrs), 'rcx')
    common_zero.term = Term('jmp', targets=(common_return,))
    ret_p = add(common_return, 'phi', r.base.type, [*r.pair_pointers, zero_p],
                {'blocks': [*joins, common_zero]}, 'rdx')
    ret_i = add(common_return, 'phi', T.UINT,
                [*[p.index.results[0] for p in r.pairs], add(common_zero, 'cast', T.UINT, [count])],
                {'blocks': [*joins, common_zero]}, 'rcx')
    signed = add(common_return, 'cast', T.LONG, [ret_p])
    start = add(common_return, 'cast', T.LONG, [fn.avx_parameters[fn.params[0]]])
    distance = add(common_return, 'sub', T.LONG, [signed, start], physical='rdx')
    index = add(common_return, 'cast', T.ULONG, [ret_i], {'force_zero_extension': True}, 'rax')
    delta = add(common_return, 'cast', T.ULONG, [distance])
    result = add(common_return, 'add', T.ULONG, [index, delta], physical='rax')
    common_return.term = Term('ret', (result,))
    for paths, following in zip(r.pairs, [*[p.head for p in r.pairs[1:]], r.fallback]):
        paths.flag.term.targets = common_zero, following
        paths.joined.term.targets = common_return, following
        phi = paths.index
        incoming = [(p, v) for p, v in zip(phi.attrs['blocks'], phi.args) if p not in (paths.flag, paths.zero)]
        phi.attrs['blocks'], phi.args = [p for p, _ in incoming], tuple(v for _, v in incoming)
        prefer(phi.results[0], 'rcx')
        fn.defs[paths.joined.term.args[0]].attrs['biased_count'] = True

    r.pair_call.ops.remove(r.endpoint)
    r.fallback.ops.append(r.endpoint)
    prefer(r.endpoint.results[0], 'rdi')
    second_mask = set2['join'].term.args[0]
    tz = add(r.fallback, 'intrinsic', T.UINT, [second_mask], builtin('_tzcnt_u32'), 'rcx')
    candidate = add(r.fallback, 'or', T.UINT, [tz, fn.constant(T.UINT, 16)], physical='rcx')
    condition = add(r.fallback, 'ne', T.INT, [second_mask, fn.constant(T.UINT, 0)], {'test_width': 16})
    selected = add(r.fallback, 'select', T.UINT,
                   [condition, candidate, fn.constant(T.UINT, 32)], physical='rdx')
    r.fallback.term = Term('jmp', targets=(r.pair_call,))
    r.final_count_cast.args = (selected,)
    fn.word_return_bases = bases
    fn.avx_break_counts = {first_count, tz, intrinsic(r.first_head.zero, 'ctz_nonzero').results[0], count}
    fn.avx_sentinels = sentinel1, sentinel2
    fn.avx_loop_returns = r.single.head, r.single.joined, loop_sentinel
    fn.avx_cspn_heads = first, second
    fn.avx_head1_zero_result = first_zero_result
    fn.avx_pair_cursors = r.pair_pointers[1:]
    fn.avx_invert_pairs = [p.joined for p in r.pairs[1:3]]

    fn.avx_block_order = [fn.entry, set1['guard'], set1['fast'], fn.entry.term.targets[1],
                          *set1['region'], set1['join'], text1['guard'], text1['fast'],
                          *text1['region'], text1['join'], r.first_head.flag, r.first_head.zero,
                          r.first_head.joined, first, r.probe1, set2['guard'], set2['fast'], sentinel1,
                          r.setup, *bodies, step, r.done, r.first_call,
                          *set2['region'], set2['join'], text2['guard'], text2['fast'],
                          *text2['region'], text2['join'], r.second_head.flag, r.second_head.zero,
                          common_head, r.second_head.count, r.second_head.joined,
                          r.pairs[0].head, r.pairs[0].count, r.pairs[0].joined, r.probe2, r.long_call,
                          *exits, r.single.head, r.single.flag, r.single.zero, r.single.joined,
                          loop_sentinel, r.pairs[0].flag, r.pairs[1].head, r.pairs[1].count,
                          r.pairs[1].joined, sentinel2, r.pairs[1].flag,
                          r.pairs[2].head, r.pairs[2].count, r.pairs[2].joined, r.pairs[2].flag,
                          r.pairs[3].head, r.pairs[3].count, r.pairs[3].joined, r.pairs[3].flag,
                          common_zero, common_return, r.fallback, r.pair_call, second]
    fn.loop_alignments = {bodies[0]: 0}
    fn.keep_tail_calls = ('str_strlen',)
    fn.blocks = fn.reachable()
    return True
