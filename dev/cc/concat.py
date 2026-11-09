"""AVX2 strcat: three destination probes followed by a bounded string copy."""
from copy import copy
from types import SimpleNamespace

from . import types as T
from .ir import Term
from .prefixes import string_bare
from .shortcopy import intrinsic, match_copy, prepare_copy


def match_concat(fn):
    if len(fn.params) != 2 or fn.type.base.kind != 'ptr' or fn.entry.term.kind != 'br':
        return None
    first, head = fn.entry.term.targets
    if first.term.kind != 'jmp' or head.term.kind != 'br':
        return None
    root = first.term.targets[0]
    pointers = [op for op in root.ops if op.kind == 'phi' and op.results[0].type.kind == 'ptr']
    if len(pointers) != 1 or root.term.kind != 'br':
        return None
    destination = pointers[0].results[0]
    guard = fn.defs[root.term.args[0]]
    if guard.kind != 'le':
        return None

    def constant(v):
        op = fn.defs[v]
        return op.attrs.get('value') if op.kind == 'constant' else None

    page = fn.defs[guard.args[0]]
    if (page.kind != 'and' or constant(page.args[1]) != 4095 or constant(guard.args[1]) != 4064
            or string_bare(fn, page.args[0]) != fn.params[1]):
        return None
    view = copy(fn)
    view.entry, view.params = root, [destination, fn.params[1]]
    region = match_copy(view, returned=fn.params[0])
    if region is None:
        return None
    count_test = fn.defs[head.term.args[0]]
    if count_test.kind != 'lt' or constant(count_test.args[1]) != 3:
        return None
    counter = fn.defs[count_test.args[0]]
    logical, end = head.term.targets
    if (counter.kind != 'phi' or logical.term.kind != 'br' or end.term.kind != 'br'
            or logical.term.targets[0] is not end):
        return None
    body = logical.term.targets[1]
    if body.term.kind != 'jmp' or body.term.targets != (head,):
        return None
    if counter.attrs['blocks'] != [fn.entry, body] or constant(counter.args[0]) != 0:
        return None
    update = fn.defs[counter.args[1]]
    if update.kind != 'add' or update.args[0] != counter.results[0] or constant(update.args[1]) != 1:
        return None
    cursor = [op for op in head.ops if op.kind == 'phi' and op.results[0].type.kind == 'ptr']
    if len(cursor) != 1 or cursor[0].attrs['blocks'] != [fn.entry, body]:
        return None
    base, next_cursor = cursor[0].args
    step = fn.defs[next_cursor]
    if step.kind != 'ptradd' or step.args[0] != cursor[0].results[0] or constant(step.args[1]) != 32:
        return None
    aligned = fn.defs[string_bare(fn, base)]
    if (aligned.kind != 'and' or constant(aligned.args[1]) != (1 << 64)-32
            or string_bare(fn, aligned.args[0]) != fn.params[0]):
        return None
    mask = fn.defs[end.term.args[0]]
    if (mask.kind != 'phi' or mask.attrs['blocks'] != [fn.entry, body]
            or mask.args[0] != fn.entry.term.args[0] or logical.term.args != mask.results):
        return None
    found, fallback = end.term.targets
    if found.term.kind != 'jmp' or found.term.targets != (root,) or fallback.term.kind != 'ret':
        return None
    calls = [op for op in fallback.ops if op.kind == 'call']
    if len(calls) != 1 or calls[0].args[1:] != tuple(fn.params):
        return None
    if set(pointers[0].attrs['blocks']) != {first, found}:
        return None
    initial_load = intrinsic(fn.entry, 'load')
    initial_compare = intrinsic(fn.entry, 'cmpeq_epi8')
    initial_mask = intrinsic(fn.entry, 'movemask8')
    if (initial_load.results[0].type != T.V256 or initial_load.attrs.get('alignment', 1) < 32
            or initial_compare.args[0] != initial_load.results[0]
            or constant(initial_compare.args[1]) != bytes(32)
            or initial_mask.args != initial_compare.results):
        return None
    return SimpleNamespace(root=root, destination=destination, view=view, copy=region,
                           base=base, aligned=aligned.results[0], fallback=fallback,
                           load=initial_load, compare=initial_compare, mask=initial_mask,
                           count=intrinsic(first, 'ctz_nonzero'), page=page, guard=guard)


def prepare_concat(fn):
    c = match_concat(fn)
    if c is None:
        return False
    r, dst = c.copy, c.destination
    entry, params = fn.entry, fn.params
    reachable, todo = set(), [c.root]
    while todo:
        block = todo.pop()
        if block not in reachable:
            reachable.add(block)
            todo.extend(block.term.targets)
    search = [b for b in fn.blocks if b not in reachable]
    prepare_copy(c.view, r)
    fn.__dict__.update(c.view.__dict__)
    fn.entry, fn.params = entry, params
    fn.blocks = search + c.view.blocks
    for block in fn.blocks:
        if block.term.kind == 'ret' and block.term.args == (dst,):
            block.term.args = (params[0],)
    registers, vectors = fn.induction_registers, fn.vector_registers

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

    # Save the source in its copy ABI register before reusing RSI for the
    # destination scan. Keep the public destination in RDI for every return.
    src_copy = fn.make('ptradd', [params[1].type], [params[1], fn.constant(T.LONG, 0)],
                       {'cat_parameter_copy': True})
    dst_copy = fn.make('ptradd', [c.base.type], [params[0], fn.constant(T.LONG, 0)],
                       {'cat_parameter_copy': True})
    src, initial_dst = src_copy.results[0], dst_copy.results[0]
    prefer(src, 'rdx')
    prefer(initial_dst, 'rsi')
    prefer(params[0], 'rdi')
    for block in fn.blocks:
        for op in block.ops:
            op.args = tuple(src if a == params[1] else a for a in op.args)
        block.term.args = tuple(src if a == params[1] else a for a in block.term.args)
    entry.ops[:0] = [src_copy, dst_copy]

    # Both source-mask paths reconverge while the destination CTZ remains
    # live. Complete the destination address in that common mask block.
    r.hit.ops.remove(r.mask)
    r.joined.ops = [r.mask]
    r.fast.term = Term('jmp', targets=(r.joined,))
    r.cold.term = Term('jmp', targets=(r.joined,))
    r.joined.term = Term('br', r.mask.results, (r.hit, r.aligned))
    fn.blocks.append(r.joined)
    end_op = fn.defs[dst]
    c.root.ops.remove(end_op)
    scans = [fn.block('cat.scan.' + str(n)) for n in (32, 64, 96)]
    hits = [fn.block('cat.hit.' + str(n)) for n in (32, 64)]
    check, found_move, found = [fn.block('cat.' + name) for name in ('checked.flag', 'found.move', 'found')]
    masks, pointers, compares = [], [], []
    zero = fn.constant(T.V256, bytes(32))
    prefer(c.aligned, 'rax')
    prefer(c.mask.results[0], 'rcx')
    prefer(entry.term.args[0], 'rcx')
    vectors[c.compare.results[0]] = 'ymm1'
    for n, (scan, delta) in enumerate(zip(scans, (32, 64, 96))):
        offset = fn.constant(T.LONG, delta)
        pointer = add(scan, 'ptradd', c.base.type, [c.base, offset])
        cast = add(scan, 'cast', c.load.args[0].type, [pointer])
        data = add(scan, 'intrinsic', T.V256, [cast], dict(c.load.attrs))
        compare = add(scan, 'intrinsic', T.V256, [data, zero], dict(c.compare.attrs),
                      'ymm1' if n == 1 else 'ymm0')
        mask = add(scan, 'intrinsic', T.INT, [compare], dict(c.mask.attrs), 'rcx')
        status = add(scan, 'cast', T.UINT, [mask])
        prefer(status, 'rcx')
        masks.append(status)
        compares.append(compare)
        if n < 2:
            pointers.append(add(hits[n], 'ptradd', c.base.type, [c.base, offset], physical='rax'))
            hits[n].term = Term('jmp', targets=(check,))
            scan.term = Term('br', (status,), (hits[n], scans[n+1]))
        else:
            pointers.append(add(scan, 'ptradd', c.base.type, [c.base, offset], physical='rax'))
            stop = add(scan, 'eq', T.UCHAR, [status, fn.constant(T.UINT, 0)],
                       {'cat_materialize_status': True}, 'rsi')
            scan.term = Term('br', (stop,), (c.fallback, found_move))
    pointer = add(check, 'phi', c.base.type, pointers[:2], {'blocks': hits}, 'rax')
    mask = add(check, 'phi', T.UINT, masks[:2], {'blocks': hits}, 'rcx')
    flag = add(check, 'add', T.UCHAR, [fn.constant(T.UCHAR, 0)]*2, {'cat_clear_status': True}, 'rsi')
    check.term = Term('br', (flag,), (c.fallback, found_move))
    origin = add(found_move, 'phi', c.base.type, [pointer, pointers[2]],
                 {'blocks': [check, scans[2]]}, 'rsi')
    mask = add(found_move, 'phi', T.UINT, [mask, masks[2]], {'blocks': [check, scans[2]]}, 'rcx')
    found_move.term = Term('jmp', targets=(found,))
    origin = add(found, 'phi', c.base.type, [initial_dst, origin], {'blocks': [entry, found_move]}, 'rsi')
    mask = add(found, 'phi', T.UINT, [entry.term.args[0], mask], {'blocks': [entry, found_move]}, 'rcx')
    count = add(found, 'intrinsic', T.INT, [mask], dict(c.count.attrs), 'rax')
    extended = add(found, 'cast', T.LONG, [count])
    found.term = Term('jmp', targets=(c.root,))
    end_op.kind, end_op.args, end_op.attrs = 'ptradd', (origin, extended), {'cat_end': True}
    r.joined.ops.append(end_op)
    prefer(dst, 'rsi')
    entry.term = Term('br', entry.term.args, (found, scans[0]))

    # The copy region has an additional live public return pointer. Its
    # aligned source and vector masks use RCX/R8 instead of RDX/RCX.
    prefer(r.base, 'rcx')
    prefer(r.end, 'rcx')
    for block in (r.fast, r.cold):
        prefer(intrinsic(block, 'movemask8').results[0], 'rcx')
    prefer(r.mask.args[1], 'rcx')
    prefer(r.mask.results[0], 'rcx')
    first_count = intrinsic(r.hit, 'ctz_nonzero').results[0]
    prefer(first_count, 'rax')
    for block in (r.aligned, r.miss, r.third_miss):
        prefer(intrinsic(block, 'movemask8').results[0], 'r8')
    prefer(intrinsic(r.second_hit, 'ctz_nonzero').results[0], 'r8')
    tail = fn.avx_copy_tail
    tail_mask = next(op for op in tail.ops if op.kind == 'phi' and op.results[0].type == T.UINT)
    tail_count = intrinsic(tail, 'ctz_nonzero').results[0]
    prefer(tail_mask.results[0], 'r8')
    prefer(tail_count, 'rcx')
    prefer(c.page.results[0], 'rcx')
    prefer(next(op for op in r.cold.ops if op.kind == 'and').results[0], 'rcx')
    fn.avx_copy_breaks = {count, first_count, tail_count}
    fn.avx_copy_zero_groups[:0] = [([c.compare.results[0], compares[0]], 0), ([compares[1], compares[2]], 0)]
    c.guard.attrs['cat_page_compare'] = True
    fn.word_return_bases |= {src, initial_dst, pointers[2]}
    fn.retained_offsets |= {dst, src, initial_dst}
    fn.avx_cat_parameters = src, initial_dst
    fn.avx_cat_entry = c.compare.results[0], entry.term.args[0]
    fn.avx_cat_third = scans[2], pointers[2], masks[2]
    prefix = [entry, scans[0], hits[0], scans[1], hits[1], check, c.fallback, scans[2],
              found_move, found, c.root, r.fast, r.cold, r.joined]
    fn.avx_copy_order = prefix + [b for b in fn.avx_copy_order if b not in prefix]
    fn.blocks = fn.reachable()
    fn.avx_copy_order = [b for b in fn.avx_copy_order if b in fn.blocks]
    fn.post_skip_passes = ('copy',)
    fn.avx_cat = True
    return True
