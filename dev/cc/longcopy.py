"""Source roles and the pass recipe for AVX2 pipelined string copying."""
from types import SimpleNamespace

from . import types as T
from .ir import Term
from .prefixes import string_bare


def intrinsic(block, name):
    return next(op for op in block.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == name)


def reads(block):
    return [op for op in block.ops if 'read' in op.attrs.get('memory', '')]


def small_copy(header):
    vector, word_guard = header.term.targets
    word, dword_guard = word_guard.term.targets
    dword, half_guard = dword_guard.term.targets
    half, byte_guard = half_guard.term.targets
    byte, returned = byte_guard.term.targets
    if vector.term.kind != 'jmp' or vector.term.targets != (returned,):
        return None
    return SimpleNamespace(header=header, vector=vector, word_guard=word_guard, word=word,
                           dword_guard=dword_guard, dword=dword, half_guard=half_guard,
                           half=half, byte_guard=byte_guard, byte=byte, returned=returned)


def match_copy(fn):
    if len(fn.params) != 3 or fn.type.base.kind != 'ptr' or fn.entry.term.kind != 'br':
        return None
    prefix, head = fn.entry.term.targets
    if prefix.term.kind != 'br':
        return None
    prefix_hit, prefix_advance = prefix.term.targets
    first = small_copy(prefix_hit)
    if first is None:
        return None
    head_hit, prefix_guard = head.term.targets
    second_header, known = head_hit.term.targets
    second = small_copy(second_header)
    if second is None:
        return None
    if known.term.kind != 'br' or known.term.targets[0] is not known:
        return None
    known_tail = known.term.targets[1]
    known_return = known_tail.term.targets[0]
    prefix_copy, single = prefix_guard.term.targets
    if prefix_copy.term.kind != 'br' or prefix_copy.term.targets != (prefix_copy, single):
        return None
    single_test, initial = single.term.targets
    if single_test.term.kind != 'br' or single_test.term.targets[1] is not single:
        return None
    last = single_test.term.targets[0]
    if initial.term.kind != 'jmp':
        return None
    group_head = initial.term.targets[0]
    exit_, half = group_head.term.targets
    if half.term.kind != 'br' or half.term.targets[0] is not exit_:
        return None
    next_ = half.term.targets[1]
    if next_.term.kind != 'jmp' or next_.term.targets != (group_head,):
        return None
    if exit_.term.targets[0] is not last:
        return None
    exit_second = exit_.term.targets[1]
    exit_third = exit_second.term.targets[1]
    exit_fourth = exit_third.term.targets[1]
    if (exit_second.term.targets[0] is not last or exit_third.term.targets[0] is not last
            or exit_fourth.term.targets != (last,)):
        return None

    b = dict(entry=fn.entry, prefix=prefix, prefix_hit=prefix_hit, prefix_advance=prefix_advance,
             head=head, head_hit=head_hit, prefix_guard=prefix_guard, prefix_copy=prefix_copy,
             single=single, single_test=single_test, initial=initial, group_head=group_head,
             half=half, next=next_, exit=exit_, exit_second=exit_second,
             exit_third=exit_third, exit_fourth=exit_fourth, last=last,
             known=known, known_tail=known_tail, known_return=known_return, head_small=second_header)
    for label, region in (('prefix', first), ('head', second)):
        for key, block in vars(region).items():
            if key != 'header':
                b[label + '_' + key] = block
    v = {}

    def result(key, op):
        v[key] = op.results[0]
        return op

    def phis(block):
        return [op for op in block.ops if op.kind == 'phi']

    source = next(op for op in phis(head) if op.results[0].type.kind == 'ptr')
    result('source_cursor', source)
    result('source_distance', next(op for op in head.ops if op.kind == 'sub'))
    result('source_distance_unsigned', fn.defs[prefix_guard.term.args[0]])
    for label, block, hit in (('prefix', prefix, prefix_hit), ('head', head, head_hit)):
        result(label + '_vector', intrinsic(block, 'load'))
        result(label + '_zero', intrinsic(block, 'cmpeq_epi8'))
        result(label + '_mask', intrinsic(block, 'movemask8'))
        result(label + '_status', fn.defs[block.term.args[0]])
        result(label + '_count', intrinsic(hit, 'ctz_nonzero'))
        length = fn.defs[fn.defs[hit.term.args[0]].args[0]]
        result(label + '_length', length)
        result(label + '_last', fn.defs[length.args[0]])
    result('prefix_distance', next(op for op in prefix_hit.ops if op.kind == 'sub'))
    result('head_bound', fn.defs[head_hit.term.args[0]])
    result('known_copy_index', next(op for op in phis(known) if op.results[0].type == T.ULONG))
    result('prefix_copy_index', next(op for op in phis(prefix_copy) if op.results[0].type == T.ULONG))
    result('prefix_copy_next', fn.defs[fn.defs[prefix_copy.term.args[0]].args[0]])
    result('prefix_copy_vector', intrinsic(prefix_copy, 'load'))
    for label, region in (('prefix', first), ('head', second)):
        for kind, block in (('word', region.word), ('dword', region.dword)):
            left, right = reads(block)
            result(label + '_' + kind, left)
            result(label + '_' + kind + '_tail', right)
        word, byte = reads(region.half)
        result(label + '_halfword', word)
        result(label + '_last_pointer', fn.defs[byte.args[0]])
        store = next(op for op in region.half.ops if op.kind == 'store' and op.attrs['width'] == 1)
        address = fn.defs[store.args[0]]
        index_cast = fn.defs[address.args[1]]
        result(label + '_last_index_signed', index_cast)
        result(label + '_last_index', fn.defs[index_cast.args[0]])
    result('head_vector_tail', reads(second.vector)[1])
    for op in phis(single):
        result('single_cursor' if op.results[0].type.kind == 'ptr' else
               'single_vector' if op.results[0].type.kind == 'vector' else 'single_offset', op)
    load = intrinsic(single_test, 'load')
    result('single_next_vector', load)
    cast = result('single_next_cast', fn.defs[load.args[0]])
    result('single_next_cursor', fn.defs[cast.args[0]])
    result('single_zero', intrinsic(single_test, 'cmpeq_epi8'))
    result('single_mask', intrinsic(single_test, 'movemask8'))
    result('single_status', fn.defs[single_test.term.args[0]])
    result('single_alignment', fn.defs[single.term.args[0]])
    result('single_next_offset', next(op for op in single.ops if op.kind == 'add'))

    initial_reads, half_reads, next_reads = reads(initial), reads(half), reads(next_)
    if any(len(ops) != 4 for ops in (initial_reads, half_reads, next_reads)):
        return None
    group_vectors = []
    for n, op in enumerate(initial_reads):
        result('initial_vector_%d' % n, op)
        group = next(phi for phi in phis(group_head)
                     if phi.args[phi.attrs['blocks'].index(initial)] == op.results[0])
        result('group_vector_%d' % n, group)
        group_vectors.append(group)
    mask_phi = next(phi for phi in phis(group_head) if phi.results[0].type == T.UINT)
    for label, block, values in (('initial', initial, initial_reads), ('half', half, half_reads), ('next', next_, next_reads)):
        if label != 'initial':
            for n, op in enumerate(values):
                result(label + '_vector_%d' % n, op)
        zero = result(label + '_zero', intrinsic(block, 'cmpeq_epi8'))
        result(label + '_mask', intrinsic(block, 'movemask8'))
        if label == 'initial':
            result(label + '_status', fn.defs[mask_phi.args[mask_phi.attrs['blocks'].index(initial)]])
        elif label == 'half':
            result(label + '_status', fn.defs[block.term.args[0]])
        else:
            result(label + '_status', fn.defs[mask_phi.args[mask_phi.attrs['blocks'].index(next_)]])
        min_ = result(label + '_min', fn.defs[zero.args[0]])
        result(label + '_min_low', fn.defs[min_.args[0]])
        result(label + '_min_high', fn.defs[min_.args[1]])
    for op in phis(group_head):
        if op not in group_vectors:
            result('group_cursor' if op.results[0].type.kind == 'ptr' else
                   'group_status' if op.results[0].type == T.UINT else 'group_offset', op)
    for n, group in enumerate(group_vectors[:3]):
        result('exit_vector_%d' % n, next(phi for phi in phis(exit_)
               if phi.args[phi.attrs['blocks'].index(group_head)] == group.results[0]))
    for op in phis(exit_):
        if op.results[0].type.kind != 'vector':
            result('exit_cursor' if op.results[0].type.kind == 'ptr' else
                   'exit_status' if op.results[0].type == T.UINT else 'exit_offset', op)
    for label, block in (('exit', exit_), ('exit_second', exit_second), ('exit_third', exit_third)):
        result(label + '_zero', intrinsic(block, 'cmpeq_epi8'))
        result(label + '_mask', intrinsic(block, 'movemask8'))
    for op in phis(last):
        result('last_cursor' if op.results[0].type.kind == 'ptr' else 'last_status', op)
    result('last_count', intrinsic(last, 'ctz_nonzero'))
    result('last_distance', next(op for op in last.ops if op.kind == 'sub'))
    result('last_vector', intrinsic(last, 'load'))
    return SimpleNamespace(values=v, blocks=b)


def prepare_copy(fn):
    r = match_copy(fn)
    if r is None:
        return False
    v, b = (r.values, r.blocks)
    fn.avx_copy_long = True
    fn.extra_gprs = ('r11',)
    reg = fn.induction_registers = dict(getattr(fn, 'induction_registers', {}))
    vec = fn.vector_registers = dict(getattr(fn, 'vector_registers', {}))

    def prefer(value, physical):
        reg[value] = physical
        while fn.defs[value].kind == 'cast':
            value = fn.defs[value].args[0]
            reg[value] = physical

    def add(block, kind, type_, args=(), attrs=None, physical=None):
        op = fn.make(kind, [type_], args, attrs)
        block.ops.append(op)
        if physical:
            (vec if type_.kind == 'vector' else reg)[op.results[0]] = physical
        return op.results[0]
    prefer(fn.params[0], 'rax')
    prefer(v['source_cursor'], 'rdx')
    prefer(v['source_distance'], 'rcx')
    prefer(v['single_cursor'], 'r10')
    prefer(v['single_offset'], 'r9')
    prefer(v['single_next_cursor'], 'rdi')
    prefer(v['single_next_offset'], 'r9')
    prefer(v['prefix_copy_index'], 'rdi')
    prefer(v['prefix_copy_next'], 'rdi')
    vec[v['prefix_copy_vector']] = 'ymm1'
    setup = fn.block('copy.lookahead.setup')
    setup.term = Term('jmp', targets=(b['single'],))
    for parent in (b['prefix_guard'], b['prefix_copy']):
        parent.term.targets = tuple((setup if q is b['single'] else q for q in parent.term.targets))
    for n in ('single_offset', 'single_vector', 'single_cursor'):
        phi = fn.defs[v[n]]
        initial = phi.args[0]
        back = phi.args[-1]
        phi.attrs['blocks'] = [setup, b['single_test']]
        phi.args = (initial, back)
    progress = add(b['single'], 'phi', T.ULONG, attrs={'blocks': [setup, b['single_test']]}, physical='r8')
    progress_next = add(b['single_test'], 'add', T.ULONG, [progress, fn.constant(T.ULONG, 32)], physical='r8')
    fn.defs[progress].args = (fn.constant(T.ULONG, 0), progress_next)
    phi = fn.defs[progress]
    b['single'].ops.remove(phi)
    b['single'].ops.insert(0, phi)
    load = fn.defs[v['single_next_vector']]
    cast = fn.defs[v['single_next_cast']]
    b['single_test'].ops.remove(cast)
    b['single_test'].ops.remove(load)
    at = b['single'].ops.index(fn.defs[v['single_alignment']])
    b['single'].ops[at:at] = [cast, load]
    for block in fn.blocks:
        for op in block.ops:
            op.args = tuple((v['single_next_vector'] if a == v['initial_vector_0'] else a for a in op.args))
    kernel = fn.block('copy.groups.setup')
    edges = [fn.block('copy.groups.exit.' + str(n)) for n in range(3)]
    for edge in edges:
        edge.term = Term('jmp', targets=(b['exit'],))
    p = add(kernel, 'cast', T.LONG, [v['source_cursor']])
    s = add(kernel, 'cast', T.LONG, [fn.params[1]])
    distance = add(kernel, 'sub', T.LONG, [p, s], physical='rdi')
    biased = add(kernel, 'ptradd', fn.params[0].type, [fn.params[0], distance], {'direct_address': True}, 'rdi')
    dest = add(kernel, 'ptradd', fn.params[0].type, [biased, fn.constant(T.LONG, 256)], {'direct_address': True}, 'rdi')
    kernel.term = Term('jmp', targets=(b['half'],))
    b['initial'].term = Term('br', (v['initial_status'],), (edges[0], kernel))
    b['half'].term = Term('br', (v['half_status'],), (edges[1], b['next']))
    b['next'].term = Term('br', (v['next_status'],), (edges[2], b['half']))
    carried = [fn.defs[v[n]] for n in ('group_vector_0', 'group_vector_1', 'group_vector_2', 'group_vector_3')]
    for op in carried:
        b['group_head'].ops.remove(op)
        op.attrs['blocks'] = [kernel, b['next']]
    index = fn.make('phi', [T.ULONG], attrs={'blocks': [kernel, b['next']]})
    nextindex = add(b['next'], 'add', T.ULONG, [index.results[0], fn.constant(T.ULONG, 256)], physical='r8')
    index.args = (progress, nextindex)
    prefer(index.results[0], 'r8')
    b['half'].ops[0:0] = [*carried, index]
    retained = {nextindex, v['prefix_length'], v['head_length']}
    for block, reads, writes, source_start, dest_start in ((b['half'], ['half_vector_0', 'half_vector_1', 'half_vector_2', 'half_vector_3'], ['group_vector_0', 'group_vector_1', 'group_vector_2', 'group_vector_3'], 160, -224), (b['next'], ['next_vector_0', 'next_vector_1', 'next_vector_2', 'next_vector_3'], ['half_vector_0', 'half_vector_1', 'half_vector_2', 'half_vector_3'], 288, -96)):
        stores = [op for op in block.ops if op.kind == 'intrinsic' and 'write' in op.attrs.get('memory', '')]
        for k, n in enumerate(reads):
            op = fn.defs[v[n]]
            new = []

            def make(kind, type_, args):
                q = fn.make(kind, [type_], args)
                new.append(q)
                return q.results[0]
            address = make('ptradd', v['source_cursor'].type, [v['source_cursor'], index.results[0]])
            address = make('ptradd', v['source_cursor'].type, [address, fn.constant(T.LONG, source_start + 32 * k)])
            converted = make('cast', op.args[0].type, [address])
            op.args = (converted,)
            at = block.ops.index(op)
            block.ops[at:at] = new
        for k, op in enumerate(stores):
            address = fn.make('ptradd', [dest.type], [dest, index.results[0]])
            offset = fn.make('ptradd', [dest.type], [address.results[0], fn.constant(T.LONG, dest_start + 32 * k)])
            cast = fn.make('cast', [op.args[0].type], offset.results)
            at = block.ops.index(op)
            block.ops[at:at] = [address, offset, cast]
            op.args = (cast.results[0], op.args[1])
    points = [v['single_next_cursor']]
    offsets = [v['single_next_offset']]
    for edge, cursor, delta in ((edges[1], index.results[0], 160), (edges[2], nextindex, 32)):
        p = add(edge, 'ptradd', v['source_cursor'].type, [v['source_cursor'], cursor])
        p = add(edge, 'ptradd', v['source_cursor'].type, [p, fn.constant(T.LONG, delta)], physical='rdi')
        offset = add(edge, 'add', T.ULONG, [v['source_distance_unsigned'], cursor])
        offset = add(edge, 'add', T.ULONG, [offset, fn.constant(T.ULONG, delta)], physical='r9')
        points.append(p)
        offsets.append(offset)
        retained.add(offset)
    for n, args in (('exit_vector_0', ['single_next_vector', 'half_vector_0', 'next_vector_0']), ('exit_vector_1', ['initial_vector_1', 'half_vector_1', 'next_vector_1']), ('exit_vector_2', ['initial_vector_2', 'half_vector_2', 'next_vector_2'])):
        op = fn.defs[v[n]]
        op.args = tuple((v[x] for x in args))
        op.attrs['blocks'] = edges[:]
    for n, args in (('exit_offset', offsets), ('exit_status', [v['initial_status'], v['half_status'], v['next_status']]), ('exit_cursor', points)):
        op = fn.defs[v[n]]
        op.args = tuple(args)
        op.attrs['blocks'] = edges[:]
    fn.blocks = [q for q in fn.blocks if q is not b['group_head']]
    colors = {'prefix_vector': 0, 'prefix_zero': 0, 'head_vector': 0, 'head_zero': 1, 'single_vector': 0, 'single_next_vector': 0, 'single_zero': 2, 'initial_vector_1': 4, 'initial_vector_2': 3, 'initial_vector_3': 7, 'initial_min_low': 1, 'initial_min_high': 2, 'initial_min': 2, 'initial_zero': 2, 'group_vector_0': 0, 'group_vector_1': 4, 'group_vector_2': 3, 'group_vector_3': 7, 'half_vector_0': 6, 'half_vector_1': 5, 'half_vector_2': 2, 'half_vector_3': 8, 'half_min_low': 9, 'half_min_high': 10, 'half_min': 9, 'half_zero': 9, 'next_vector_0': 0, 'next_vector_1': 4, 'next_vector_2': 3, 'next_vector_3': 7, 'next_min_low': 9, 'next_min_high': 10, 'next_min': 9, 'next_zero': 2, 'exit_vector_0': 6, 'exit_vector_1': 5, 'exit_vector_2': 2, 'exit_zero': 0, 'exit_second_zero': 1, 'exit_third_zero': 0, 'last_vector': 0}
    for n, color in colors.items():
        vec[v[n]] = 'xmm' + str(color) if v[n].type.bits == 128 else 'ymm' + str(color)
    for n in ('initial_mask', 'half_mask', 'next_mask', 'exit_status'):
        prefer(v[n], 'r10')
    for n in ('single_mask', 'exit_mask', 'exit_second_mask', 'exit_third_mask', 'last_status'):
        prefer(v[n], 'r11')
    prefer(v['exit_offset'], 'r9')
    prefer(v['exit_cursor'], 'rdi')
    prefer(v['last_cursor'], 'rdi')
    prefer(v['last_count'], 'rcx')
    prefer(v['last_distance'], 'rdx')
    prefer(v['prefix_mask'], 'rcx')
    prefer(v['prefix_count'], 'rdi')
    prefer(v['prefix_distance'], 'rdx')
    prefer(v['prefix_last'], 'rcx')
    prefer(v['prefix_length'], 'rcx')
    prefer(v['head_mask'], 'rdi')
    prefer(v['head_count'], 'rdi')
    prefer(v['head_last'], 'rdx')
    prefer(v['head_length'], 'rdx')
    for n in ('prefix_last', 'head_last'):
        fn.defs[v[n]].attrs['copy_count'] = True
    for block, total, base_distance, count, last_pointer, index_cast, word_value, physical in ((b['prefix_half'], 'prefix_last_index', 'prefix_distance', 'prefix_count', 'prefix_last_pointer', 'prefix_last_index_signed', 'prefix_halfword', 'rdx'), (b['head_half'], 'head_last_index', 'source_distance', 'head_count', 'head_last_pointer', 'head_last_index_signed', 'head_halfword', 'rcx')):
        op = fn.defs[v[total]]
        op.kind = 'add'
        op.args = (v[base_distance], v[count])
        op.attrs = {'copy_count': True}
        prefer(v[total], physical)
        fn.defs[v[last_pointer]].args = (fn.params[1], v[index_cast])
        prefer(v[word_value], 'rcx' if block is b['prefix_half'] else 'rdx')
    fn.floating_copies = set(getattr(fn, 'floating_copies', ())) - {v['head_vector_tail']}
    test = fn.defs[v['head_bound']]
    test.kind = 'lt'
    test.args = (v['head_length'], fn.constant(T.ULONG, 33))
    fn.defs[v['half_min_high']].args = fn.defs[v['half_min_high']].args[::-1]
    fn.aligned_vector_copies = True
    phis = [fn.defs[v[n]] for n in ('exit_vector_0', 'exit_vector_1', 'exit_vector_2', 'exit_offset', 'exit_status', 'exit_cursor')]
    b['exit'].ops = phis + [op for op in b['exit'].ops if op not in phis]
    for edge in edges[1:]:
        sums = [op for op in edge.ops if op.kind == 'add']
        sums[0].attrs['copy_count'] = True
        prefer(sums[0].results[0], 'r9')
        if edge is edges[1]:
            sums[1].attrs['cold_offset'] = 160
        else:
            retained.add(sums[0].results[0])
    for block, total, index_cast, load_value in ((b['prefix_half'], 'prefix_last_index', 'prefix_last_index_signed', 'prefix_halfword'), (b['head_half'], 'head_last_index', 'head_last_index_signed', 'head_halfword')):
        moved = [fn.defs[v[total]], fn.defs[v[index_cast]]]
        for op in moved:
            block.ops.remove(op)
        at = block.ops.index(fn.defs[v[load_value]])
        block.ops[at:at] = moved
    copy_setup = fn.block('copy.known.setup')
    initial = add(copy_setup, 'add', T.ULONG, [fn.constant(T.ULONG, 32), fn.constant(T.ULONG, 0)], {'counter_initial': True}, 'rcx')
    copy_setup.term = Term('jmp', targets=(b['known'],))
    b['head_hit'].term.targets = tuple((copy_setup if q is b['known'] else q for q in b['head_hit'].term.targets))
    phi = fn.defs[v['known_copy_index']]
    at = phi.attrs['blocks'].index(b['head_hit'])
    phi.attrs['blocks'][at] = copy_setup
    phi.args = tuple((initial if n == at else q for n, q in enumerate(phi.args)))
    for n in ('prefix_word', 'prefix_dword'):
        prefer(v[n], 'rdx')
    prefer(v['prefix_word_tail'], 'rsi')
    prefer(v['prefix_dword_tail'], 'rsi')
    for n in ('head_word', 'head_dword'):
        prefer(v[n], 'rcx')
    prefer(v['head_word_tail'], 'rsi')
    prefer(v['head_dword_tail'], 'rsi')
    fn.retained_offsets = retained
    fn.word_return_bases = set(getattr(fn, 'word_return_bases', ())) | {v['source_cursor'], dest, biased}
    fn.avx_long_zero_groups = [([v['prefix_zero']], 0, None), ([v['head_zero']], 1, None), ([v['single_zero']], 1, setup), ([v['initial_zero'], v['half_zero'], v['next_zero'], v['exit_zero']], 1, None), ([v['exit_second_zero'], v['exit_third_zero']], 0, None)]
    fn.avx_long_breaks = {v['prefix_count'], v['last_count']}
    fn.avx_long_clean_return = b['prefix_returned']
    fn.avx_long_progress = progress
    fn.avx_long_lookahead = (setup, b['single'], b['single_test'], v['single_next_cursor'], v['single_next_offset'], v['single_cursor'])
    fn.avx_long_masks = (v['half_mask'], v['next_zero'], v['next_mask'])
    fn.avx_long_final = b['exit_fourth']
    fn.avx_long_initial = (edges[0], b['exit'])
    fn.avx_long_kernel = (kernel, edges, b['half'], b['next'])
    fn.loop_alignments = dict(getattr(fn, 'loop_alignments', {}))
    fn.loop_alignments[b['single']] = 0
    fn.shared_index_return = True
    fn.avx_long_order = [b[n] for n in ('entry', 'prefix', 'prefix_hit', 'prefix_vector', 'prefix_advance', 'head', 'head_hit', 'head_small', 'head_vector', 'prefix_guard', 'prefix_copy')] + [setup, b['single'], b['single_test'], b['prefix_word_guard'], b['prefix_word'], copy_setup, b['known'], b['known_tail'], b['known_return'], b['initial'], kernel, b['half'], b['next'], edges[2], edges[0], b['exit'], b['exit_second'], b['prefix_dword_guard'], b['prefix_dword'], b['head_word_guard'], b['head_word'], b['prefix_half_guard'], b['prefix_half'], b['head_dword_guard'], b['head_dword'], b['prefix_byte_guard'], b['prefix_byte'], b['prefix_returned'], b['head_half_guard'], b['head_half'], edges[1], b['exit_third'], b['head_byte_guard'], b['head_byte'], b['exit_fourth'], b['last']]
    fn.blocks = fn.reachable()
    fn.avx_long_order = [q for q in fn.avx_long_order if q in fn.blocks]
    return True
