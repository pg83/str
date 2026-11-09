"""AVX2 strcmp tail recipe: indexed vector groups and page-bounded words."""
from types import SimpleNamespace

from . import types as T
from .ir import Term
from .prefixes import string_bare
from .words import (prepare_word_scans, split_word_scan_returns,
                    normalize_word_indices, share_word_scan_steps)


def intrinsics(block, name):
    return [op for op in block.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == name]


def match_tail(fn):
    outer = next((b for b in fn.blocks
                  if sum(op.kind == 'phi' and op.results[0].type.kind == 'ptr' for op in b.ops) == 2
                  and any(op.kind == 'select' for op in b.ops)), None)
    if outer is None:
        return None
    setup, short = outer.term.targets
    if setup.term.kind != 'jmp':
        return None
    head = setup.term.targets[0]
    asm = next((op for op in head.ops if op.kind == 'asm' and len(op.results) == 4
                and op.attrs['code'] == ''), None)
    if asm is None:
        return None
    groups = [fn.defs[v] for v in asm.args]
    if any(op.kind != 'intrinsic' or op.attrs['builtin'].op != 'and' for op in groups):
        return None
    data = [fn.defs[op.args[0]] for op in groups]
    equal = [fn.defs[op.args[1]] for op in groups]
    if any(op.kind != 'intrinsic' or op.attrs['builtin'].op != 'load'
           or op.results[0].type != T.V256 for op in data):
        return None
    left = string_bare(fn, data[0].args[0])
    right = string_bare(fn, fn.defs[equal[0].args[1]].args[0])
    pointer_phis = [fn.defs[left], fn.defs[right]]
    if any(op.kind != 'phi' or setup not in op.attrs['blocks'] for op in pointer_phis):
        return None
    latch = head.term.targets[1]
    if latch.term.kind != 'br' or latch.term.targets != (head, outer):
        return None
    bases = [op.args[op.attrs['blocks'].index(setup)] for op in pointer_phis]
    updates = [fn.defs[op.args[op.attrs['blocks'].index(latch)]] for op in pointer_phis]
    counter = next(op for op in head.ops if op.kind == 'phi' and op.results[0].type == T.ULONG)
    limit = fn.defs[counter.args[counter.attrs['blocks'].index(setup)]]
    span = next(op for op in outer.ops if op.kind == 'select')
    if (limit.kind != 'div' or limit.args[0] != span.results[0]
            or fn.defs[limit.args[1]].kind != 'constant' or fn.defs[limit.args[1]].attrs['value'] != 128):
        return None
    checks, returns, block = [], [], head.term.targets[0]
    for _ in range(3):
        checks.append(block)
        returned, block = block.term.targets
        returns.append(returned)
    returns.append(block)
    counts = [intrinsics(b, 'ctz_nonzero')[0] for b in returns]
    root_zero = intrinsics(head, 'cmpeq_epi8')[-1]
    root_min = fn.defs[root_zero.args[0]]
    lower_min, upper_min = (fn.defs[v] for v in root_min.args)
    root_mask = intrinsics(head, 'movemask8')[0]
    return SimpleNamespace(outer=outer, setup=setup, short=short, head=head, latch=latch,
                           asm=asm, groups=groups, data=data, equal=equal, pointers=(left, right),
                           pointer_phis=pointer_phis, bases=bases, updates=updates, counter=counter,
                           limit=limit, span=span, checks=checks, returns=returns, counts=counts,
                           root_zero=root_zero, root_min=root_min,
                           lower_min=lower_min, upper_min=upper_min, root_mask=root_mask)


def prepare_tail(fn):
    r = match_tail(fn)
    if r is None or not prepare_word_scans(fn, page_width=32):
        return False
    split_word_scan_returns(fn)
    normalize_word_indices(fn)
    share_word_scan_steps(fn)
    registers, vectors = fn.induction_registers, dict(getattr(fn, 'vector_registers', {}))
    fn.vector_registers = vectors

    def prefer(value, physical):
        registers[value] = physical
        while fn.defs[value].kind == 'cast':
            value = fn.defs[value].args[0]
            registers[value] = physical

    def add(block, kind, type_, args=(), attrs=None, physical=None):
        op = fn.make(kind, [type_], args, attrs)
        block.ops.append(op)
        if physical:
            registers[op.results[0]] = physical
        return op.results[0]

    word, (_, loaded, difference, pair, subtract, mask, stop) = next(iter(fn.word_scans.items()))
    word_step = next(b for b in word.term.targets if b.term.kind == 'br' and word in b.term.targets)
    word_return = word.term.targets[0]
    word_done = word_step.term.targets[1]
    byte = word_done.term.targets[0]
    mismatch, nul = byte.term.targets
    byte_step, zero = nul.term.targets
    page_advance = word_done.term.targets[1]
    pred = fn.predecessors()
    word_guard = next(p for p in pred[word] if p is not word_step)
    word_entry, vector = r.short.term.targets
    word_bound = fn.defs[fn.defs[word_guard.term.args[0]].args[0]]
    first_distance, second_distance = (fn.defs[v] for v in fn.defs[r.span.args[0]].args)
    second_page_mask = fn.defs[second_distance.args[1]]
    both = fn.defs[r.short.term.args[0]]
    page_checks = [fn.defs[v] for v in both.args]
    page_checks.sort(key=lambda op: op.args[0] != first_distance.args[1])
    single_mask = intrinsics(vector, 'movemask8')[0]
    inversion = fn.defs[vector.term.args[0]]
    single_return, vector_advance = vector.term.targets
    single_count = intrinsics(single_return, 'ctz_nonzero')[0]
    single_good = fn.defs[single_mask.args[0]]
    single_zero, single_equal = (fn.defs[v] for v in single_good.args)
    single_data = fn.defs[single_zero.args[0]]
    word_count = intrinsics(word_return, 'ctz_nonzero')[0]

    # Reuse the minimum page distance already computed before the outer
    # choice; the scalar fallback need not form the same minimum again.
    for block in fn.blocks:
        for op in block.ops:
            op.args = tuple(r.span.results[0] if v == word_bound.results[0] else v for v in op.args)
        block.term.args = tuple(r.span.results[0] if v == word_bound.results[0] else v for v in block.term.args)
    prefer(r.span.results[0], 'r8')
    guard2 = fn.block('compare.page.second')
    second = page_checks[1]
    r.short.ops.remove(second)
    guard2.ops = [second]
    for op in page_checks:
        op.kind, op.args = 'gt', (op.args[0], fn.constant(T.ULONG, 4064))
        op.attrs['page_compare'] = True
    r.short.term = Term('br', page_checks[0].results, (word_entry, guard2))
    guard2.term = Term('br', page_checks[1].results, (word_entry, vector))

    # Retain both page origins, carrying one byte index through each group.
    r.limit.kind, r.limit.args = 'and', (r.span.results[0], fn.constant(T.ULONG, -128))
    prefer(r.limit.results[0], 'r8')
    index = add(r.head, 'phi', T.ULONG, attrs={'blocks': [r.setup, r.latch]}, physical='r9')
    next_index = add(r.latch, 'add', T.ULONG, [index, fn.constant(T.ULONG, 128)], physical='r9')
    fn.defs[index].args = fn.constant(T.ULONG, 0), next_index
    fn.word_return_bases = set(getattr(fn, 'word_return_bases', ())) | {next_index}
    fn.narrow_word_values = set(getattr(fn, 'narrow_word_values', ())) | {r.limit}
    phi = fn.defs[index]
    r.head.ops.remove(phi)
    r.head.ops.insert(0, phi)
    for old, base in zip(r.pointers, r.bases):
        for block in fn.blocks:
            if block is r.head:
                continue
            for op in list(block.ops):
                if old not in op.args:
                    continue
                if block is r.latch:
                    op.args = tuple(base if v == old else v for v in op.args)
                    if op.kind == 'ptradd':
                        op.args = base, next_index
                elif op.kind == 'ptradd' and op.args[0] == old:
                    total = fn.defs[op.args[1]]
                    while total.kind == 'cast':
                        total = fn.defs[total.args[0]]
                    delta = 0
                    if total.kind == 'add':
                        count, constant = total.args
                        delta = fn.defs[constant].attrs['value']
                        total = fn.defs[count]
                        while total.kind == 'cast':
                            total = fn.defs[total.args[0]]
                    cast = fn.make('cast', [T.LONG], total.results)
                    start = fn.make('ptradd', [old.type], [base, cast.results[0]])
                    offset = fn.make('add', [T.ULONG], [index, fn.constant(T.ULONG, delta)])
                    at = block.ops.index(op)
                    block.ops[at:at] = [cast, start, offset]
                    op.args = start.results[0], offset.results[0]
                    registers[start.results[0]] = 'rdi' if base == r.bases[0] else 'rsi'
                    fn.word_return_bases |= set(start.results)
                else:
                    op.args = tuple(base if v == old else v for v in op.args)
        local = fn.make('ptradd', [old.type], [base, index])
        at = next(n for n, op in enumerate(r.head.ops) if op.kind != 'phi')
        r.head.ops.insert(at, local)
        for op in r.head.ops:
            if op is not local:
                op.args = tuple(local.results[0] if v == old else v for v in op.args)
    compare = add(r.latch, 'ne', T.INT, [r.limit.results[0], next_index])
    advance = fn.block('compare.vector.advance')
    for op in r.updates:
        r.latch.ops.remove(op)
        op.attrs['materialize_index'] = True
    advance.ops = r.updates
    advance.term = Term('jmp', targets=(r.outer,))
    for op in r.outer.ops:
        if op.kind == 'phi':
            op.attrs['blocks'] = [advance if p is r.latch else p for p in op.attrs['blocks']]
    r.latch.term = Term('br', (compare,), (r.head, advance))
    for op in [*r.pointer_phis, r.counter]:
        r.head.ops.remove(op)

    # Pipeline independent loads and comparisons while retaining the source
    # min tree and the vectors needed by each mismatch return.
    ordered = [*r.data[:3], r.equal[0], r.equal[1], r.groups[0], r.groups[1],
               r.equal[2], r.data[3], r.groups[2], r.equal[3], r.groups[3],
               r.asm, r.lower_min, r.upper_min, r.root_min, r.root_zero, r.root_mask,
               fn.defs[r.head.term.args[0]]]
    original, seen, output = list(r.head.ops), set(), []

    def visit(op):
        if op not in original or op in seen:
            return
        for value in op.args:
            visit(fn.defs[value])
        seen.add(op)
        output.append(op)

    for op in original:
        if op.kind == 'phi':
            visit(op)
    for op in ordered + original:
        visit(op)
    r.head.ops = output
    colors = [(op.results[0], n) for ops, ns in (
        (r.data, (1, 2, 4, 5)), (r.equal, (3, 5, 1, 4)), (r.groups, (3, 2, 1, 4)))
              for op, n in zip(ops, ns)]
    colors += list(zip(r.asm.results, (3, 2, 1, 4)))
    colors += [(op.results[0], n) for op, n in (
        (r.lower_min, 5), (r.upper_min, 4), (r.root_min, 4), (r.root_zero, 4),
        *zip([intrinsics(b, 'cmpeq_epi8')[0] for b in r.checks], (3, 0, 0)),
        (single_data, 1), (single_equal, 2), (single_zero, 1), (single_good, 1))]
    for value, color in colors:
        vectors[value] = 'ymm%d' % color
    for block in r.checks:
        prefer(intrinsics(block, 'movemask8')[0].results[0], 'rcx')
    for op in [*r.counts, single_count]:
        prefer(op.results[0], 'rcx')
    prefer(r.root_mask.results[0], 'rax')
    prefer(single_mask.results[0], 'rax')
    prefer(inversion.results[0], 'rax')
    fn.avx_tail_count_breaks = {word_count.results[0], r.counts[-1].results[0], single_count.results[0]}
    prefer(subtract.results[0], 'r10')
    prefer(pair.results[0], 'r11')
    prefer(mask.results[0], 'r11')
    subtract.attrs['scalar_lea'] = True
    pair.attrs['scalar_andnot'] = True
    mask.args = mask.args[::-1]
    cast = fn.defs[subtract.args[0]]
    for op in (cast, subtract):
        word.ops.remove(op)
    at = word.ops.index(pair)
    word.ops[at:at] = [cast, subtract]
    inversion.kind, inversion.args = 'not', (inversion.args[0],)
    for op in list(fn.defs.values()):
        if op.kind == 'and' and op is not r.limit and op in fn.narrow_word_values:
            op.args = op.args[0], fn.constant(T.ULONG, 0x78)
    r.outer.ops.remove(first_distance)
    r.outer.ops.insert(r.outer.ops.index(second_page_mask), first_distance)
    fn.avx_tail_order = [fn.entry, r.outer, r.setup, r.head, r.latch, advance,
                         r.short, guard2, vector, vector_advance, word_entry, word_guard, word, word_step,
                         word_done, byte, nul, byte_step, page_advance,
                         r.checks[0], r.returns[0], r.checks[1], r.returns[1], word_return,
                         r.checks[2], r.returns[2], mismatch, zero, r.returns[3], single_return]
    fn.loop_alignments = {b: 0 for b in fn.blocks}
    fn.loop_alignments.update({r.head: 4, word: 4, byte: 4})
    fn.avx_cmp_tail = True
    fn.avx_tail_root = r.head, r.root_zero.results[0]
    fn.avx_tail_inversion = vector, inversion.results[0], single_return
    return True
