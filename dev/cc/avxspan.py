"""Per-function SSA and scheduling recipes for the AVX2 set scans.

The single-table kernel scans eight vectors per iteration. The four paired
mask blocks share their return arithmetic, and table continuations count the
set bytes without branching. The recipe matches source operations and edges;
registers and block order are explicit policy for these library functions.
"""
from types import SimpleNamespace

from . import types as T
from .builtins import BUILTINS
from .ir import Term, dominators
from .prefixes import prepare_recovery, string_bare


def intrinsic(block, name):
    return next(op for op in block.ops
                if op.kind == 'intrinsic' and op.attrs['builtin'].op == name)


def recovery_prefixes(fn):
    """Find the set/text page recovery regions in source order."""
    if not prepare_recovery(fn) or len(fn.recovery_regions) != 4:
        return None
    pred = fn.predecessors()
    prefixes = []
    for root, region in fn.recovery_regions:
        guard = next(p for p in pred[root] if p not in region)
        fast = next(b for b in guard.term.targets if b is not root)
        join = next(b for b, owner in fn.layout_owners.items()
                    if owner in region and b not in region
                    and any(op.kind == 'phi' and op.results[0].type == T.V128 for op in b.ops))
        prefixes.append(dict(root=root, region=[b for b in fn.blocks if b in region],
                             guard=guard, fast=fast, join=join))
    return prefixes


def match_span(fn):
    """Find the page prefixes, counted scan and four paired mask blocks."""
    prefixes = recovery_prefixes(fn)
    if prefixes is None:
        return None
    set1, text1, set2, text2 = prefixes
    public, setup = text1['join'].term.targets
    if setup.term.kind != 'jmp':
        return None
    body = setup.term.targets[0]
    if body.term.kind != 'br':
        return None
    returned, step = body.term.targets
    if step.term.kind != 'br' or step.term.targets[0] is not body or returned.term.kind != 'ret':
        return None
    done = step.term.targets[1]
    ptr = next(op for op in body.ops if op.kind == 'phi' and op.results[0].type.kind == 'ptr')
    before = ptr.args[ptr.attrs['blocks'].index(setup)]
    after = ptr.args[ptr.attrs['blocks'].index(step)]
    limit = fn.defs[fn.defs[step.term.args[0]].args[1]]
    if limit.kind != 'constant' or limit.attrs['value'] % 8:
        return None
    loaded, flag = intrinsic(body, 'load'), intrinsic(body, 'cmpistrc')
    count = intrinsic(returned, 'cmpistri')
    cast = next(op for op in returned.ops if op.kind == 'cast' and op.args == count.results)
    distance = next(op for op in returned.ops if op.kind == 'sub')
    first_call = next(b for b in done.term.targets if any(op.kind == 'call' for op in b.ops))
    call = next(op for op in first_call.ops if op.kind == 'call')
    count_phi = fn.defs[fn.defs[call.args[5]].args[0]]
    old_count = next(fn.defs[string_bare(fn, v)] for v in count_phi.args
                     if fn.defs[string_bare(fn, v)].kind == 'intrinsic')
    pair_setup = text2['join'].term.targets[1]
    pair_blocks, pair_returns, pair_pointers, pair_counts, pair_compares = [], [], [], [], []
    block = pair_setup
    for offset in range(4):
        if block.term.kind != 'br':
            return None
        edge, following = block.term.targets
        compare = fn.defs[block.term.args[0]]
        if (edge.term.kind != 'ret' or compare.kind != 'lt'
                or fn.defs[compare.args[1]].kind != 'constant'
                or fn.defs[compare.args[1]].attrs['value'] != 16):
            return None
        load = intrinsic(block, 'load')
        pointer = string_bare(fn, load.args[0])
        if offset:
            address = fn.defs[pointer]
            if (address.kind != 'ptradd' or address.args[0] != pair_pointers[0]
                    or fn.defs[address.args[1]].kind != 'constant'
                    or fn.defs[address.args[1]].attrs['value'] != 16 * offset):
                return None
        pair_blocks.append(block)
        pair_returns.append(edge)
        pair_pointers.append(pointer)
        pair_counts.append(intrinsic(block, 'ctz_nonzero').results[0])
        pair_compares.append(compare)
        block = following
    fallback = block
    pair_call = next(b for b in fallback.term.targets if any(op.kind == 'call' for op in b.ops))
    last_call = next(op for op in pair_call.ops if op.kind == 'call')
    final_count_cast = fn.defs[last_call.args[5]]
    endpoint = fn.defs[last_call.args[1]]
    if (endpoint.kind != 'ptradd' or endpoint.args[0] != pair_pointers[0]
            or fn.defs[endpoint.args[1]].kind != 'constant'
            or fn.defs[endpoint.args[1]].attrs['value'] != 64):
        return None
    probe1, probe2 = set1['join'].term.targets[1], set2['join'].term.targets[1]
    long_call = next(b for b in probe2.term.targets if any(op.kind == 'call' for op in b.ops))
    return SimpleNamespace(prefixes=prefixes, public=public, setup=setup, body=body, step=step,
                           done=done, returned=returned, pointer=ptr.results[0], base=before,
                           after=after, limit=limit.attrs['value'], loaded=loaded, flag=flag,
                           count=count, count_cast=cast, distance=distance,
                           first_call=first_call, first_count_phi=count_phi, old_count=old_count,
                           pair_blocks=pair_blocks, pair_returns=pair_returns,
                           pair_pointers=pair_pointers, pair_counts=pair_counts,
                           pair_compares=pair_compares, fallback=fallback, pair_call=pair_call,
                           final_count_cast=final_count_cast, endpoint=endpoint,
                           probe1=probe1, probe2=probe2, long_call=long_call)


def configure_prefixes(fn, recipe):
    registers = fn.induction_registers = dict(getattr(fn, 'induction_registers', {}))
    vectors = fn.vector_registers = dict(getattr(fn, 'vector_registers', {}))

    def prefer(value, physical):
        registers[value] = registers[string_bare(fn, value)] = physical

    for stage, prefix in enumerate(recipe.prefixes):
        root, join = prefix['root'], prefix['join']
        region = set(prefix['region'])
        data, control, mask, loaded, low, valid = (
            (0, 0, 1, 1, 'rcx', 'rax'), (1, 1, 2, 2, 'rax', 'rdi'),
            (1, 0, 1, 1, 'rdi', 'rcx'), (4, 0, 3, 3, 'rax', 'rdx'))[stage]
        for block in region | {join}:
            for op in block.ops:
                if op.kind == 'phi' and op.results[0].type == T.V128:
                    vectors[op.results[0]] = 'xmm%d' % data
                if op.kind == 'phi' and op.results[0].type == T.ULONG:
                    prefer(op.results[0], valid)
                if op.kind == 'intrinsic':
                    name = op.attrs['builtin'].op
                    if op.results and op.results[0].type == T.V128:
                        if block is root:
                            color = (mask if name in ('cmpgt_epi8', 'cmpeq_epi8') else
                                     loaded if name == 'load' else data if name == 'shuffle_epi8' else control)
                        elif block is join:
                            color = mask if stage == 0 else 0 if stage == 2 else data
                        else:
                            color = data if name == 'or' else 2 if name == 'load' else 1 if stage == 0 else 0
                        vectors[op.results[0]] = 'xmm%d' % color
                    if name == 'movemask8':
                        prefer(op.results[0], 'r8' if stage == 0 and block is join else
                               'rcx' if stage == 0 else 'rdi' if stage == 2 and block is root else 'rcx')
                if block is root and op.results and op.results[0].type.kind == 'int':
                    if (op.kind == 'and' and fn.defs[op.args[1]].kind == 'constant'
                            and fn.defs[op.args[1]].attrs['value'] in (15, (1 << 64) - 16)):
                        prefer(op.results[0], low)
                    if (op.kind == 'sub' and fn.defs[op.args[0]].kind == 'constant'
                            and fn.defs[op.args[0]].attrs['value'] == 16):
                        prefer(op.results[0], valid)
        if stage in (0, 2):
            test = fn.defs[root.term.args[0]]
            for value, other in (test.args, test.args[::-1]):
                source, definition = fn.defs[string_bare(fn, value)], fn.defs[other]
                if (source.kind == 'intrinsic' and source.attrs['builtin'].op == 'movemask8'
                        and definition.kind == 'sub'):
                    shift = fn.defs[definition.args[0]]
                    if shift.kind == 'shl':
                        test.kind, test.args = 'intrinsic', (value, shift.args[1])
                        test.attrs = builtin('_bzhi_u32')
                        prefer(test.results[0], 'rcx' if stage == 0 else 'rdi')
                        break
    for block in fn.blocks:
        for op in list(block.ops):
            if (op.kind == 'ptradd' and string_bare(fn, op.args[0]) == fn.params[1]
                    and fn.defs[op.args[1]].kind == 'constant'
                    and fn.defs[op.args[1]].attrs['value'] == 16):
                prefer(op.results[0], 'rax')
                fn.word_return_bases = set(getattr(fn, 'word_return_bases', ())) | set(op.results)
            if (op.kind == 'intrinsic' and op.attrs['builtin'].op == 'load'
                    and string_bare(fn, op.args[0]) == fn.params[1]
                    and block is not recipe.prefixes[0]['join']):
                if any(u.kind == 'intrinsic' and u.attrs['builtin'].op == 'cmpistrm'
                       and u.args[0] == op.results[0] for b in fn.blocks for u in b.ops):
                    vectors[op.results[0]] = 'xmm2'
            if op.kind not in ('and', 'or') or not op.results or op.results[0].type != T.UINT:
                continue
            inputs = []
            for value in op.args:
                extract = fn.defs[string_bare(fn, value)]
                if extract.kind != 'intrinsic' or extract.attrs['builtin'].op != 'extract_low32':
                    break
                compare = fn.defs[extract.args[0]]
                if compare.kind != 'intrinsic' or compare.attrs['builtin'].op != 'cmpistrm':
                    break
                inputs.append(extract.args[0])
            if len(inputs) != 2:
                continue
            combined = fn.make('intrinsic', [T.V128], inputs[::-1], builtin('_mm_' + op.kind + '_si128'))
            extract = fn.make('intrinsic', [T.INT], combined.results, builtin('_mm_cvtsi128_si32'))
            at = block.ops.index(op)
            block.ops[at:at] = [combined, extract]
            op.kind, op.args, op.attrs = 'cast', extract.results, {}
            vectors[inputs[0]], vectors[inputs[1]], vectors[combined.results[0]] = 'xmm3', 'xmm0', 'xmm0'
            prefer(extract.results[0], 'rax')
            for value in inputs:
                vectors[fn.defs[value].args[1]] = 'xmm4'

    # Switch from ABI parameters after the empty-set test. Both table
    # continuations use RSI for the original string and RDX for the set.
    guard, dom = recipe.prefixes[0]['guard'], dominators(fn)
    copied, copies = {}, []
    for param, physical in ((fn.params[1], 'rdx'), (fn.params[0], 'rsi')):
        copy = fn.make('ptradd', [param.type], [param, fn.constant(T.LONG, 0)], {'parameter_copy': True})
        copies.append(copy)
        copied[param] = copy.results[0]
        prefer(copy.results[0], physical)
        fn.word_return_bases = set(getattr(fn, 'word_return_bases', ())) | set(copy.results)
    for block in fn.blocks:
        if guard not in dom[block]:
            continue
        for op in block.ops:
            op.args = tuple(copied.get(v, v) for v in op.args)
        block.term.args = tuple(copied.get(v, v) for v in block.term.args)
    guard.ops[0:0] = copies
    fn.avx_parameters = copied


def builtin(name):
    return dict(builtin=BUILTINS[name], memory='', width=0, alignment=1, mask=None)


def prepare_span(fn):
    recipe = match_span(fn)
    if recipe is None:
        return False
    configure_prefixes(fn, recipe)
    fn.avx_span = True
    reg, vec = fn.induction_registers, fn.vector_registers

    def add(block, kind, type_, args=(), attrs=None, physical=None):
        op = fn.make(kind, [type_], args, attrs)
        block.ops.append(op)
        if physical:
            (vec if type_.kind == 'vector' else reg)[op.results[0]] = physical
        return op.results[0]

    r = recipe
    limit = r.limit * 16
    end = add(r.setup, 'ptradd', r.base.type, [r.base, fn.constant(T.LONG, limit)], physical='rdi')
    reg[r.base] = 'rdx'
    bodies = [fn.block('span.single.body.%d' % n) for n in range(8)]
    exits = [fn.block('span.single.exit.%d' % n) for n in range(8)]
    step = fn.block('span.single.step')
    index = add(bodies[0], 'phi', T.ULONG, attrs={'blocks': [r.setup, step]}, physical='rax')
    next_index = add(step, 'add', T.ULONG, [index, fn.constant(T.ULONG, 128)], physical='rax')
    fn.defs[index].args = fn.constant(T.ULONG, 0), next_index
    compare = add(step, 'ne', T.INT, [next_index, fn.constant(T.ULONG, limit)], {'narrow_compare': True})
    step.term = Term('br', (compare,), (bodies[0], r.done))
    pointers, data = [], []
    bases = set(getattr(fn, 'word_return_bases', ()))
    for n, (body, edge) in enumerate(zip(bodies, exits)):
        pointer = add(body, 'ptradd', r.base.type, [r.base, index])
        if n:
            pointer = add(body, 'ptradd', r.base.type, [pointer, fn.constant(T.LONG, 16 * n)])
        cast = add(body, 'cast', r.loaded.args[0].type, [pointer])
        loaded = add(body, 'intrinsic', T.V128, [cast], dict(r.loaded.attrs), 'xmm1')
        data.append(loaded)
        flag = add(body, 'intrinsic', T.INT, [r.flag.args[0], loaded, r.flag.args[2]], dict(r.flag.attrs))
        body.term = Term('br', (flag,), (edge, bodies[n + 1] if n < 7 else step))
        point = add(edge, 'ptradd', r.base.type, [r.base, index], physical='rdx')
        bases.add(point)
        if n:
            point = add(edge, 'ptradd', r.base.type, [point, fn.constant(T.LONG, 16 * n)], physical='rdx')
            bases.add(point)
        pointers.append(point)
        edge.term = Term('jmp', targets=(r.returned,))
    pointer_phi = fn.make('phi', [r.pointer.type], pointers, {'blocks': exits})
    vector_phi = fn.make('phi', [T.V128], data, {'blocks': exits})
    r.returned.ops[0:0] = [pointer_phi, vector_phi]
    reg[pointer_phi.results[0]], vec[vector_phi.results[0]] = 'rdx', 'xmm1'
    replacements = {r.pointer: pointer_phi.results[0], r.loaded.results[0]: vector_phi.results[0], r.after: end}
    for block in fn.blocks:
        if block in (r.body, r.step, step) or block in bodies or block in exits:
            continue
        for op in block.ops:
            op.args = tuple(replacements.get(a, a) for a in op.args)
            if op.kind == 'phi':
                op.attrs['blocks'] = [step if parent is r.step else parent for parent in op.attrs['blocks']]
    r.setup.term = Term('jmp', targets=(bodies[0],))
    fn.blocks = [b for b in fn.blocks if b not in (r.body, r.step)]
    r.returned.ops.remove(r.count)
    r.returned.ops.remove(r.count_cast)
    result = fn.defs[r.returned.term.args[0]]
    at = r.returned.ops.index(result)
    r.returned.ops[at:at] = [r.count, r.count_cast]
    result.args = result.args[::-1]
    reg[r.distance.results[0]] = 'rdx'

    # A nonempty low-word sentinel makes CTZ defined for the full table too.
    masked = add(r.done, 'or', T.UINT,
                 [r.prefixes[0]['join'].term.args[0], fn.constant(T.UINT, 65536)], physical='r8')
    first_count = add(r.done, 'intrinsic', T.INT, [masked], dict(r.old_count.attrs), 'rdx')
    r.done.term = Term('jmp', targets=(r.first_call,))
    r.first_count_phi.kind, r.first_count_phi.args, r.first_count_phi.attrs = 'cast', (first_count,), {}

    # The hot pair blocks only form memory addresses. Materialize their
    # updated cursor on an exit, where the common return actually needs it.
    shared = fn.block('span.pair.return')
    for pointer, count in zip(r.pair_pointers, r.pair_counts):
        reg[pointer], reg[count] = 'rdi', 'rax'
    for block, edge, pointer in zip(r.pair_blocks[1:], r.pair_returns[1:], r.pair_pointers[1:]):
        op = fn.defs[pointer]
        local = fn.make(op.kind, [pointer.type], op.args, dict(op.attrs))
        at = block.ops.index(op)
        block.ops[at] = local
        fn.defs[intrinsic(block, 'load').args[0]].args = local.results
        edge.ops.insert(0, op)
    p = add(shared, 'phi', r.pair_pointers[0].type, r.pair_pointers,
            {'blocks': r.pair_returns}, 'rdi')
    c = add(shared, 'phi', T.INT, r.pair_counts, {'blocks': r.pair_returns}, 'rax')
    for edge in r.pair_returns:
        edge.term = Term('jmp', targets=(shared,))
    signed_p = add(shared, 'cast', T.LONG, [p])
    signed_s = add(shared, 'cast', T.LONG, [fn.avx_parameters[fn.params[0]]])
    distance = add(shared, 'sub', T.LONG, [signed_p, signed_s], physical='rdi')
    count = add(shared, 'cast', T.ULONG, [c], {'force_zero_extension': True}, 'rax')
    wide_distance = add(shared, 'cast', T.ULONG, [distance])
    value = add(shared, 'add', T.ULONG, [count, wide_distance], physical='rax')
    shared.term = Term('ret', (value,))
    for compare in r.pair_compares:
        compare.attrs['biased_count'] = True

    first = fn.block('span.head1.return')
    second = fn.block('span.head2.return')
    for prefix, returned in ((r.prefixes[1], first), (r.prefixes[3], second)):
        head = prefix['join']
        incoming = next(op for op in r.public.ops if op.kind == 'phi')
        returned.term = Term('ret', (incoming.args[incoming.attrs['blocks'].index(head)],))
        head.term.targets = tuple(returned if t is r.public else t for t in head.term.targets)

    # A zero second mask means a full table. TZCNT is defined on zero, and
    # only the selected nonzero result contributes to the shorter count.
    r.pair_call.ops.remove(r.endpoint)
    r.fallback.ops.append(r.endpoint)
    reg[r.endpoint.results[0]] = 'rdi'
    second_mask = r.prefixes[2]['join'].term.args[0]
    tz = add(r.fallback, 'intrinsic', T.UINT, [second_mask], builtin('_tzcnt_u32'), 'rax')
    candidate = add(r.fallback, 'or', T.UINT, [tz, fn.constant(T.UINT, 16)], physical='rax')
    test = add(r.fallback, 'ne', T.INT, [second_mask, fn.constant(T.UINT, 0)], {'test_width': 16})
    selected = add(r.fallback, 'select', T.UINT,
                   [test, candidate, fn.constant(T.UINT, 32)], physical='rdx')
    r.fallback.term = Term('jmp', targets=(r.pair_call,))
    r.final_count_cast.args = (selected,)
    fn.avx_break_counts = {tz, first_count}
    fn.word_return_bases = bases
    fn.avx_returns = first, second, shared, r.pair_returns
    set1, text1, set2, text2 = r.prefixes

    def prefix_order(p):
        return [p['guard'], p['fast'], *p['region'], p['join']]

    fn.avx_block_order = [fn.entry, set1['guard'], set1['fast'],
                          fn.entry.term.targets[1], *set1['region'], set1['join'],
                          *prefix_order(text1), first, r.setup, *bodies, step, r.done, r.first_call,
                          r.probe1, *prefix_order(set2), *prefix_order(text2),
                          r.pair_blocks[0], r.pair_blocks[1], r.pair_returns[1],
                          r.probe2, r.long_call, *exits, r.returned,
                          r.pair_blocks[2], r.pair_returns[2], r.pair_blocks[3], r.pair_returns[3],
                          r.pair_returns[0], shared, second, r.fallback, r.pair_call]
    fn.blocks = fn.reachable()
    return True
