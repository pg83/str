"""Control-flow and layout policy for the concrete SSE4.2 strcspn kernel."""
from . import types as T
from .ir import Term, dominators
from .prefixes import string_bare
from .tails import converging_unary


def prepare_cspn(fn):
    if not getattr(fn, 'string_prefixes', False) or fn.name != 'strcspn_sse42':
        return False
    pred, dom = fn.predecessors(), dominators(fn)
    registers, vectors = fn.induction_registers, fn.vector_registers

    def prefer(value, register):
        registers[value] = registers[string_bare(fn, value)] = register

    def constant(value):
        op = fn.defs[value]
        return op.attrs['value'] if op.kind == 'constant' else None

    def intrinsic(block, name):
        return next(op for op in block.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == name)

    def index_phi(block):
        return next(op for op in block.ops if op.kind == 'phi' and op.results[0].type == T.UINT)

    def count_path(block, register, vector_register, mask_register=None):
        count = intrinsic(block, 'ctz_nonzero')
        prefer(count.results[0], register)
        mask = intrinsic(block, 'movemask8')
        prefer(mask.results[0], mask_register or register)
        compare = intrinsic(block, 'cmpeq_epi8')
        vectors[compare.results[0]] = 'xmm%d' % vector_register
        return count

    prefixes = []
    for root, region in fn.recovery_regions:
        guard = next(p for p in pred[root] if p not in region)
        fast = next(b for b in guard.term.targets if b is not root)
        join = next(b for b, owner in fn.layout_owners.items() if owner in region and b not in region
                    and any(op.kind == 'phi' and op.results[0].type == T.V128 for op in b.ops))
        prefixes.append(dict(root=root, region=[b for b in fn.blocks if b in region], guard=guard, fast=fast, join=join))
    if len(prefixes) != 4:
        return False
    set1, text1, set2, text2 = prefixes
    for prefix in (text1, text2):
        for op in prefix['root'].ops:
            if op.kind == 'and' and constant(op.args[1]) in (15, (1 << 64)-16):
                prefer(op.results[0], 'rcx')
            elif op.kind == 'sub' and constant(op.args[0]) == 16:
                prefer(op.results[0], 'rax')
        valid = next(op for op in prefix['join'].ops if op.kind == 'phi' and op.results[0].type == T.ULONG)
        prefer(valid.results[0], 'rax')

    first_head = text1['join']
    first_zero_flag, first_join = first_head.term.targets
    first_zero = first_zero_flag.term.targets[0]
    first_index = index_phi(first_join)
    first_return, first_setup = first_join.term.targets
    prefer(first_index.results[0], 'rcx')
    count_path(first_zero, 'rcx', 0)
    fn.defs[first_join.term.args[0]].attrs['index_compare'] = True

    second_head = text2['join']
    second_count, second_zero_flag = second_head.term.targets
    second_zero, second_join = second_zero_flag.term.targets
    second_index = index_phi(second_join)
    second_return, second_setup = second_join.term.targets
    prefer(second_index.results[0], 'rcx')
    prefer(intrinsic(second_count, 'ctz_nonzero').results[0], 'rcx')
    prefer(second_head.term.args[0], 'rcx')
    count_path(second_zero, 'rcx', 4)
    fn.defs[second_join.term.args[0]].attrs['index_compare'] = True

    def sentinel(flag, join, phi, name):
        edge = fn.block('string.index.sentinel.' + name)
        edge.term = Term('jmp', targets=(join,))
        flag.term.targets = flag.term.targets[0], edge
        at = phi.attrs['blocks'].index(flag)
        phi.attrs['blocks'][at] = edge
        phi.args = tuple(fn.constant(T.UINT, 16) if n == at else value for n, value in enumerate(phi.args))
        return edge

    first_sentinel = sentinel(first_zero_flag, first_join, first_index, 'first')
    second_sentinel = sentinel(second_zero_flag, second_join, second_index, 'second')

    # Share the two head-mask counts, retaining the independent sentinel edge.
    old_joins, old_blocks = getattr(fn, 'return_joins', ()), set(fn.blocks)
    fn.return_joins = (second_join,)
    converging_unary(fn)
    fn.return_joins = old_joins
    common_counts = [b for b in fn.blocks if b not in old_blocks]
    if len(common_counts) != 1:
        raise ValueError(fn.name + ': expected a common head-mask count')
    common_count = common_counts[0]
    for op in common_count.ops:
        if op.results and op.results[0].type.kind == 'int':
            prefer(op.results[0], 'rcx')

    def loop(setup, paired):
        head = setup.term.targets[0]
        pointer = next(op for op in head.ops if op.kind == 'phi' and op.results[0].type.kind == 'ptr')
        latch = next(b for b in pointer.attrs['blocks'] if b is not setup)
        joins = [b for b in pred[latch] if b.term.kind == 'br' and b.term.targets[1] is latch]
        if len(joins) != 1:
            raise ValueError(fn.name + ': expected an index-tested loop latch')
        joined = joins[0]
        index = index_phi(joined)
        returned = joined.term.targets[0]
        if paired:
            count, zero_flag = head.term.targets
        else:
            zero_flag = head.term.targets[0]
            count = None
        zero = zero_flag.term.targets[0]
        scalar = 'rax' if paired else 'rcx'
        prefer(index.results[0], scalar)
        count_path(zero, scalar, 4 if paired else 0, None if paired else 'rax')
        loaded = intrinsic(head, 'load')
        vectors[loaded.results[0]] = 'xmm%d' % (4 if paired else 0)
        if count is not None:
            prefer(intrinsic(count, 'ctz_nonzero').results[0], scalar)
            prefer(head.term.args[0], scalar)

        # The cursor and its distance advance together on the existing latch.
        initial = pointer.args[pointer.attrs['blocks'].index(setup)]
        a = fn.make('cast', [T.LONG], [initial])
        origin = fn.make('cast', [T.LONG], [fn.params[0]])
        distance = fn.make('sub', [T.LONG], [a.results[0], origin.results[0]])
        setup.ops.extend([a, origin, distance])
        offset = fn.make('phi', [T.LONG], attrs={'blocks': [setup, latch]})
        head.ops.insert(0, offset)
        advance = fn.make('add', [T.LONG], [offset.results[0], fn.constant(T.LONG, 16)])
        latch.ops.append(advance)
        offset.args = distance.results[0], advance.results[0]
        for value in (distance.results[0], offset.results[0], advance.results[0]):
            prefer(value, 'rdx')
        for value in (*pointer.results, *pointer.args):
            prefer(value, 'rsi')
        old_distance = next(op for op in returned.ops if op.kind == 'sub'
                            and string_bare(fn, op.args[0]) == pointer.results[0]
                            and string_bare(fn, op.args[1]) == fn.params[0])
        old_distance.kind, old_distance.args, old_distance.attrs = 'cast', offset.results, {}
        addition = next(op for op in returned.ops if op.kind == 'add' and op.results[0].type == T.ULONG)
        addition.args = addition.args[::-1]
        extended_index = fn.defs[addition.args[0]]

        # ZF-clear plus index 16 advances directly. A located NUL returns
        # directly; only the match-index edge needs the index<16 test.
        zero_flag.term.targets = zero, latch
        output = fn.make('phi', [T.UINT], [index.results[0], index.args[index.attrs['blocks'].index(zero)]],
                         {'blocks': [joined, zero]})
        returned.ops.insert(0, output)
        prefer(output.results[0], scalar)
        extended_index.args = output.results
        if paired:
            extended_index.attrs['force_zero_extension'] = True
            prefer(extended_index.results[0], 'rax')
        zero.term = Term('jmp', targets=(returned,))
        pairs = [(parent, value) for parent, value in zip(index.attrs['blocks'], index.args)
                 if parent not in (zero_flag, zero)]
        index.attrs['blocks'], index.args = [b for b, _ in pairs], tuple(v for _, v in pairs)
        if not paired:
            fn.defs[joined.term.args[0]].attrs['existing_flags'] = True
        return dict(head=head, latch=latch, check=joined, returned=returned,
                    zero_flag=zero_flag, zero=zero, count=count)

    first_loop, second_loop = loop(first_setup, False), loop(second_setup, True)
    fn.string_indices = tuple(op.results[0] for b in fn.blocks for op in b.ops
                              if op.kind == 'phi' and op.results[0].type == T.UINT)
    first_probe, second_probe = set1['join'].term.targets[1], set2['join'].term.targets[1]
    table = next(b for b in second_probe.term.targets if b.term.kind == 'ret')
    empty = next(b for b in fn.entry.term.targets if b.term.kind == 'ret')

    def recovery(prefix):
        return prefix['region'] + [prefix['join']]

    fn.string_block_order = (
        [fn.entry, set1['guard'], set1['fast'], empty] + recovery(set1)
        + [text1['guard'], text1['fast']] + recovery(text1)
        + [first_zero_flag, first_zero, first_join, first_return, first_probe, set2['guard'], set2['fast'], first_sentinel,
           first_setup, first_loop['check'], first_loop['latch'], first_loop['head'], first_loop['zero_flag'],
           first_loop['zero'], first_loop['returned']]
        + recovery(set2) + [text2['guard'], text2['fast']] + recovery(text2)
        + [second_zero_flag, second_zero, common_count, second_count, second_join, second_setup,
           second_loop['zero_flag'], second_loop['latch'], second_loop['head'], second_loop['count'],
           second_loop['check'], second_probe, table, second_loop['zero'], second_loop['returned'], second_sentinel])
    fn.string_cspn = dict(returns=(first_return, second_return),
                          sentinels=(first_sentinel, second_sentinel),
                          heads=(first_loop['head'], second_loop['head']),
                          invert=second_loop['check'])
    fn.preserve_address_order = True
    fn.keep_tail_calls = ('str_strlen',)
    fn.tail_limit = 3
    return True
