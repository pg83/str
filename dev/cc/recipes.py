"""Explicit optimization pipelines for the library's concrete kernels.

Recipes operate on the parsed program and check the relevant loop shape before
rewriting it. Register choices and scheduling are policy; loads, stores, tests
and returned values still come from the C function's SSA.
"""
from . import types as T
from .ir import Term
from .prefixes import prepare_string
from .cspn import prepare_cspn
from .avxspan import prepare_span
from .avxcspn import prepare_cspn as prepare_avx_cspn
from .avxstrcmp import prepare_strcmp
from .strcmptail import prepare_tail as prepare_strcmp_tail
from .shortcopy import prepare_copy as prepare_short_copy
from .longcopy import prepare_copy as prepare_long_copy
from .concat import prepare_concat


def copy_pipeline(fn):
    """Use two biased cursors for the alternating four-vector forward copy."""
    if len(fn.params) != 3 or fn.type.base.kind != 'ptr':
        return False

    def bare(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value

    def constant(value):
        op = fn.defs[value]
        return op.attrs['value'] if op.kind == 'constant' else None

    def address(value):
        value, offset = bare(value), 0
        while fn.defs[value].kind == 'ptradd' and constant(fn.defs[value].args[1]) is not None:
            op = fn.defs[value]
            offset += T.convert_integer(constant(op.args[1]), T.LONG)
            value = bare(op.args[0])
        return value, offset

    def memory(block, effect):
        return [op for op in block.ops if effect in op.attrs.get('memory', '')]

    pred = fn.predecessors()
    for head in fn.blocks:
        phis = [op for op in head.ops if op.kind == 'phi']
        vectors = [op for op in phis if op.results[0].type == T.V256]
        pointers = [op for op in phis if op.results[0].type.kind == 'ptr']
        counters = [op for op in phis if op.results[0].type == T.ULONG]
        if len(vectors) != 4 or len(pointers) != 2 or len(counters) != 1 or head.term.kind != 'br':
            continue
        join, body = head.term.targets
        if body.term.kind != 'br' or body.term.targets[1] is not head or len(pred[head]) != 2 or body not in pred[head]:
            continue
        before = next(b for b in pred[head] if b is not body)
        merged = body.term.targets[0]
        if (before.term.kind != 'br' or before.term.targets != (merged, head)
                or merged.term.kind != 'jmp' or merged.term.targets != (join,)
                or pred[body] != [head] or set(pred[join]) != {merged, head}):
            continue
        reads, writes = memory(head, 'read'), memory(head, 'write')
        later_reads, later_writes = memory(body, 'read'), memory(body, 'write')
        if (any(len(ops) != 4 for ops in (reads, writes, later_reads, later_writes))
                or any(op.kind != 'intrinsic' or op.attrs.get('width') != 32 or op.attrs.get('volatile')
                       for op in reads + writes + later_reads + later_writes)
                or any(op.kind in ('call', 'asm') for block in (head, body) for op in block.ops)):
            continue
        source, destination = address(reads[0].args[0])[0], address(writes[0].args[0])[0]
        if {source, destination} != {op.results[0] for op in pointers}:
            continue
        if ([(address(op.args[0])) for op in reads] != [(source, 128 + 32*k) for k in range(4)]
                or [address(op.args[0]) for op in later_reads] != [(source, 256 + 32*k) for k in range(4)]
                or [address(op.args[0]) for op in writes] != [(destination, 32*k) for k in range(4)]
                or [address(op.args[0]) for op in later_writes] != [(destination, 128 + 32*k) for k in range(4)]):
            continue
        src_phi, dst_phi, counter = fn.defs[source], fn.defs[destination], counters[0]
        entry_of = lambda op: op.args[op.attrs['blocks'].index(before)]
        back_of = lambda op: op.args[op.attrs['blocks'].index(body)]
        if address(back_of(src_phi)) != (source, 256) or address(back_of(dst_phi)) != (destination, 256):
            continue
        initial_src, initial_dst = fn.defs[bare(entry_of(src_phi))], fn.defs[bare(entry_of(dst_phi))]
        if (initial_src.kind != 'ptradd' or initial_dst.kind != 'ptradd'
                or bare(initial_src.args[0]) != fn.params[1] or bare(initial_dst.args[0]) != fn.params[0]
                or initial_src.args[1] != initial_dst.args[1]):
            continue
        skew = fn.defs[bare(initial_dst.args[1])]
        if skew.kind != 'sub' or constant(skew.args[0]) != 32:
            continue
        low = fn.defs[bare(skew.args[1])]
        if low.kind != 'and' or constant(low.args[1]) != 31 or bare(low.args[0]) != fn.params[0]:
            continue
        initial_count = fn.defs[bare(entry_of(counter))]
        intermediate = fn.defs[fn.defs[head.term.args[0]].args[0]]
        count_update = fn.defs[bare(back_of(counter))]
        tests = [fn.defs[b.term.args[0]] for b in (before, head, body)]
        if (initial_count.kind != 'sub' or bare(initial_count.args[0]) != fn.params[2]
                or bare(initial_count.args[1]) != skew.results[0]
                or any(op.kind != 'lt' or constant(op.args[1]) != 256 for op in tests)
                or tests[0].args[0] != initial_count.results[0]
                or intermediate.kind != 'sub' or intermediate.args[0] != counter.results[0]
                or constant(intermediate.args[1]) != 128
                or count_update.kind != 'sub' or count_update.args[0] != counter.results[0]
                or constant(count_update.args[1]) != 256):
            continue
        carriers = [fn.defs[op.args[1]] for op in writes]
        if (set(carriers) != set(vectors)
                or any(back_of(phi) != read.results[0] for phi, read in zip(carriers, later_reads))):
            continue
        preloads = memory(before, 'read')
        initial_vectors = [entry_of(phi) for phi in carriers]
        if (len(preloads) != 9 or [op.results[0] for op in preloads[5:]] != initial_vectors
                or any(op.kind != 'intrinsic' or op.attrs.get('width') != 32 for op in preloads)):
            continue
        final_stores = memory(join, 'write')
        if len(final_stores) != 9:
            continue
        final_vectors = [fn.defs[op.args[1]] for op in final_stores[:4]]
        final_destination = fn.defs[address(final_stores[0].args[0])[0]]
        if (final_destination.kind != 'phi' or final_destination not in join.ops
                or any(phi.kind != 'phi' or phi not in join.ops for phi in final_vectors)):
            continue
        valid = True
        for k, phi in enumerate(final_vectors):
            values = dict(zip(phi.attrs['blocks'], phi.args))
            middle = fn.defs[values.get(merged)] if merged in values else None
            if (values.get(head) != reads[k].results[0] or middle is None or middle.kind != 'phi'
                    or dict(zip(middle.attrs['blocks'], middle.args)) != {
                        before: initial_vectors[k], body: later_reads[k].results[0]}):
                valid = False
        if not valid or initial_dst not in before.ops:
            continue
        break
    else:
        return False

    setup, initial, normal, partial = (fn.block('copy.pipeline.' + name)
                                       for name in ('setup', 'initial', 'done', 'half'))
    registers = fn.induction_registers = dict(getattr(fn, 'induction_registers', {}))
    # A no-overlap entry can release the original destination in RDI; the
    # overlap-dispatching wrapper must preserve it until the final return.
    opaque = fn.name == 'copy_huge_avx2'
    low_reg, cursor_reg = ('rdi', 'rcx') if opaque else ('rcx', 'rax')

    def make(block, kind, type_, args, register=None):
        op = fn.make(kind, [type_], args)
        block.ops.append(op)
        if register:
            registers[op.results[0]] = register
        return op.results[0]

    destination_start = make(setup, 'ptradd', destination.type, [fn.params[0], fn.constant(T.LONG, 32)], 'r9')
    negative = make(setup, 'neg', T.ULONG, low.results, 'r10')
    second = make(setup, 'ptradd', destination.type, [fn.params[0], fn.constant(T.LONG, 160)], cursor_reg)
    source_start = make(setup, 'ptradd', source.type, [fn.params[1], fn.constant(T.LONG, 384)], 'rsi')
    setup.term = Term('jmp', targets=(head,))
    for phi in phis:
        phi.attrs['blocks'] = [setup if p is before else p for p in phi.attrs['blocks']]
    new_phis = []

    def carry(value, register):
        phi = fn.make('phi', [value.type], attrs={'blocks': [setup, body]})
        new_phis.append(phi)
        update = fn.make('ptradd', [value.type], [phi.results[0], fn.constant(T.LONG, 256)])
        body.ops.insert(body.ops.index(tests[2]), update)
        phi.args = value, update.results[0]
        registers[phi.results[0]] = registers[update.results[0]] = register
        return phi.results[0], update.results[0]

    current_dst, next_dst = carry(destination_start, 'r9')
    second_dst, _ = carry(second, cursor_reg)
    current_src, _ = carry(source_start, 'rsi')
    # Keep each sum local to its memory operations, so addressing-mode
    # selection can fold both registers and the displacement on both halves.
    for block in (head, body):
        src_sum = fn.make('ptradd', [source.type], [current_src, negative])
        src_base = fn.make('ptradd', [source.type], [src_sum.results[0], fn.constant(T.LONG, -352)])
        dst_base = fn.make('ptradd', [destination.type], [current_dst, negative])
        prefix = [src_sum, src_base, dst_base]
        replacements = {source: src_base.results[0], destination: dst_base.results[0]}
        if block is body:
            half = fn.make('ptradd', [destination.type], [dst_base.results[0], fn.constant(T.LONG, 128)])
            prefix.append(half)
            half_destination = bare(later_writes[0].args[0])
            replacements[half_destination] = half.results[0]
        for op in block.ops:
            if op.kind != 'phi':
                op.args = tuple(replacements.get(v, v) for v in op.args)
        at = next(n for n, op in enumerate(block.ops) if op.kind != 'phi')
        block.ops[at:at] = prefix
    head.ops.remove(src_phi)
    head.ops.remove(dst_phi)
    head.ops[0:0] = new_phis

    before.ops.remove(initial_dst)
    initial.ops.append(initial_dst)
    registers[initial_dst.results[0]] = cursor_reg

    def adjusted(block, pointer, register):
        integer = make(block, 'cast', T.ULONG, [pointer])
        result = make(block, 'sub', T.ULONG, [integer, low.results[0]], register)
        return make(block, 'cast', destination.type, [result], register)

    final_ptrs = [initial_dst.results[0], adjusted(normal, next_dst, 'r9'), adjusted(partial, second_dst, cursor_reg)]
    for k, phi in enumerate(final_vectors):
        phi.args = initial_vectors[k], later_reads[k].results[0], reads[k].results[0]
        phi.attrs['blocks'] = [initial, normal, partial]
    final_destination.args, final_destination.attrs['blocks'] = tuple(final_ptrs), [initial, normal, partial]
    for block in (initial, normal, partial):
        block.term = Term('jmp', targets=(join,))
    before.term.targets, head.term.targets, body.term.targets = (initial, setup), (partial, body), (normal, head)
    fn.blocks.remove(merged)
    for op, register in ((low, low_reg), (skew, cursor_reg), (initial_count, 'r8'), (counter, 'r8'),
                         (intermediate, 'r11'), (count_update, 'r8'), (final_destination, cursor_reg)):
        registers[op.results[0]] = register
    fn.scalar_constant_registers = {**getattr(fn, 'scalar_constant_registers', {}), skew.args[0]: cursor_reg}
    intermediate.kind, intermediate.args = 'add', (counter.results[0], fn.constant(T.ULONG, -128))
    count_update.kind, count_update.args = 'add', (counter.results[0], fn.constant(T.ULONG, -256))
    fn.counted_offsets = set(getattr(fn, 'counted_offsets', ())) | set(intermediate.results)
    # Order the exceptional parallel assignments by their result registers.
    other_phis = [op for op in join.ops if op.kind == 'phi' and op not in final_vectors]
    join.ops = final_vectors[1:] + final_vectors[:1] + other_phis + [op for op in join.ops if op.kind != 'phi']
    vectors = fn.vector_registers = dict(getattr(fn, 'vector_registers', {}))
    vectors[preloads[0].results[0]] = 'ymm0'
    for k, load in enumerate(preloads[1:5], 1):
        vectors[load.results[0]] = 'ymm%d' % k
    for k, number in enumerate((8, 5, 6, 7)):
        for value in (initial_vectors[k], carriers[k].results[0], later_reads[k].results[0], final_vectors[k].results[0]):
            vectors[value] = 'ymm%d' % number
    for load, number in zip(reads, (9, 12, 11, 10)):
        vectors[load.results[0]] = 'ymm%d' % number
    fn.layout_owners = {**getattr(fn, 'layout_owners', {}), setup: initial, head: setup, body: head,
                        normal: body, partial: normal, join: partial}
    fn.floating_copies = set(getattr(fn, 'floating_copies', ())) | {
        op.results[0] for block in fn.blocks for op in block.ops
        if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'load'}
    fn.copy_pipeline = setup, destination_start
    fn.extra_gprs = ('r11',)
    fn.aligned_vector_copies = True
    fn.verify()
    return True


def backward_copy(fn):
    """Retain the descending count in RDX and fold its decrement into addresses."""
    if not getattr(fn, 'copy_pipeline', None) or fn.entry.term.kind != 'br':
        return False

    def bare(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value

    def constant(value):
        op = fn.defs[value]
        return op.attrs['value'] if op.kind == 'constant' else None

    both = fn.defs[fn.entry.term.args[0]]
    if both.kind != 'and':
        return False
    left, right = (fn.defs[value] for value in both.args)
    if (left.kind != 'gt' or right.kind != 'lt' or bare(left.args[0]) != fn.params[0]
            or bare(left.args[1]) != fn.params[1] or bare(right.args[0]) != fn.params[0]):
        return False
    source_end = fn.defs[bare(right.args[1])]
    if (source_end.kind != 'ptradd' or bare(source_end.args[0]) != fn.params[1]
            or bare(source_end.args[1]) != fn.params[2]):
        return False
    backward, forward = fn.entry.term.targets
    if backward.term.kind != 'br' or forward.term.kind != 'br':
        return False
    head, returned = backward.term.targets
    if head.term.kind != 'br' or head.term.targets != (head, returned) or returned.term.kind != 'ret':
        return False
    phis = [op for op in head.ops if op.kind == 'phi']
    if len(phis) != 1 or phis[0].results[0].type != T.ULONG or set(phis[0].attrs['blocks']) != {head, backward}:
        return False
    phi = phis[0]
    initial = fn.defs[bare(phi.args[phi.attrs['blocks'].index(backward)])]
    update = fn.defs[bare(phi.args[phi.attrs['blocks'].index(head)])]
    test = fn.defs[head.term.args[0]]
    if (update.kind != 'sub' or update.args[0] != phi.results[0] or constant(update.args[1]) != 128
            or test.kind != 'gt' or test.args != (update.results[0], update.args[1])
            or initial.kind != 'sub' or bare(initial.args[0]) != fn.params[2]):
        return False
    low = fn.defs[bare(initial.args[1])]
    end = fn.defs[bare(low.args[0])] if low.kind == 'and' else None
    if (end is None or constant(low.args[1]) != 31 or end.kind != 'ptradd'
            or bare(end.args[0]) != fn.params[0] or bare(end.args[1]) != fn.params[2]):
        return False
    loads = [op for op in head.ops if 'read' in op.attrs.get('memory', '')]
    stores = [op for op in head.ops if 'write' in op.attrs.get('memory', '')]
    if (len(loads) != 4 or len(stores) != 4 or any(op.attrs.get('width') != 32 for op in loads + stores)
            or any(op.kind in ('call', 'asm') or op.attrs.get('volatile') for op in head.ops)):
        return False
    addresses = [op for op in head.ops if op.kind == 'ptradd' and bare(op.args[1]) == update.results[0]]
    if len(addresses) != 2 or {bare(op.args[0]) for op in addresses} != set(fn.params[:2]):
        return False
    tails = [b for b in forward.term.targets if b.term.kind == 'ret' and any(op.kind == 'call' for op in b.ops)]
    if len(tails) != 1:
        return False
    for op in addresses:
        cast = fn.make('cast', [T.LONG], phi.results)
        base = fn.make('ptradd', [op.results[0].type], [op.args[0], cast.results[0]])
        at = head.ops.index(op)
        head.ops[at:at] = [cast, base]
        op.args = base.results[0], fn.constant(T.LONG, -128)
    head.ops.remove(update)
    head.ops.insert(head.ops.index(test), update)
    update.kind, update.args = 'add', (phi.results[0], fn.constant(T.ULONG, -128))
    left.kind, right.kind, both.kind = 'le', 'ge', 'or'
    both.args = both.args[::-1]
    fn.entry.term.targets = forward, backward
    fn.induction_registers.update({left.results[0]: 'rax', right.results[0]: 'rcx', both.results[0]: 'rcx',
                                   end.results[0]: 'rax', low.results[0]: 'rcx', initial.results[0]: 'rdx',
                                   phi.results[0]: 'rdx', update.results[0]: 'rdx'})
    fn.layout_owners.update({backward: tails[0], returned: head,
                            next(b for b in forward.term.targets if b is not tails[0]): returned})
    retained = [op for op in backward.ops if 'read' in op.attrs.get('memory', '')]
    for k, op in enumerate(retained):
        fn.vector_registers[op.results[0]] = 'ymm%d' % k
    for k, op in enumerate(loads, 5):
        fn.vector_registers[op.results[0]] = 'ymm%d' % k
    return True


FUNCTION_PASSES = {
    'copy_huge_avx2': (copy_pipeline,),
    'copy_loop_avx2': (copy_pipeline, backward_copy),
    'strspn_sse42': (prepare_string,),
    'strcspn_sse42': (prepare_string, prepare_cspn),
    'strspn_avx2': (prepare_span,),
    'strspn_avx2_zen': (prepare_span,),
    'strcspn_avx2': (prepare_avx_cspn,),
    'strcspn_avx2_zen': (prepare_avx_cspn,),
    'strcmp_avx2': (prepare_strcmp,),
    'strcmp_tail_avx2': (prepare_strcmp_tail,),
    'strcpy_avx2': (prepare_short_copy,),
    'strcpy_long_avx2': (prepare_long_copy,),
    'strcat_avx2': (prepare_concat,),
}


def optimize_function(fn):
    from .opt import eliminate_dead
    for run in FUNCTION_PASSES.get(fn.name, ()):
        if run(fn):
            eliminate_dead(fn)
            fn.verify()


def schedule_function(selector):
    pipeline = getattr(selector.source, 'copy_pipeline', None)
    if pipeline is not None:
        setup, first = pipeline
        block, reg = selector.blocks[setup], selector.value(first).reg
        at = next(n for n, ins in enumerate(block.ins) if reg in ins.defs)
        block.ins.insert(0, block.ins.pop(at))
