"""Keep the invariant tables and return state of a byte classification loop."""
from . import types as T
from .ir import Term
from .builtins import BUILTINS
from .interproc import callee


def prepare_classification_scan(fn):
    if (fn.type.base != T.ULONG or len(fn.params) != 5
            or any(v.type.kind != 'ptr' for v in fn.params[:2])
            or any(v.type != T.V128 for v in fn.params[2:4])
            or getattr(fn, 'parameter_masks', {}).get(fn.params[4], 2) > 1
            or any(op.kind in ('call', 'asm') or op.attrs.get('volatile')
                   or 'write' in op.attrs.get('memory', '') for b in fn.blocks for op in b.ops)):
        return False

    def bare(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value

    def constant(value):
        op = fn.defs[value]
        return op.attrs.get('value') if op.kind == 'constant' else None

    def graph(block):
        by_kind = {}
        for op in block.ops:
            if op.kind == 'intrinsic':
                by_kind.setdefault(op.attrs['builtin'].op, []).append(op)
        required = {'load': 1, 'shuffle_epi8': 3, 'xor': 1, 'or': 1, 'lshr16': 1,
                    'and': 2, 'cmpeq_epi8': 1, 'movemask8': 1}
        if any(len(by_kind.get(kind, ())) != count for kind, count in required.items()):
            return None
        nodes = {kind: ops[0] for kind, ops in by_kind.items() if len(ops) == 1}
        if nodes['load'].results[0].type != T.V256:
            return None
        shift = nodes['lshr16']
        indexed = next((op for op in by_kind['and'] if shift.results[0] in op.args), None)
        lookup = next((op for op in by_kind['shuffle_epi8'] if isinstance(constant(op.args[0]), bytes)), None)
        if indexed is None or lookup is None:
            return None
        final = next(op for op in by_kind['and'] if op is not indexed)
        nodes.update(indexed=indexed, lookup=lookup, final=final, shuffles=by_kind['shuffle_epi8'])
        mask = nodes['movemask8'].results[0]
        inverse = next((op for op in block.ops if op.kind == 'not' and bare(op.args[0]) == mask), None)
        if inverse is None:
            return None
        flipped = next((op for op in block.ops if op.kind == 'xor' and inverse.results[0] in op.args), None)
        if flipped is None:
            return None
        nodes.update(inverse=inverse, flipped=flipped)
        return nodes

    pred = fn.predecessors()
    heads = [b for b in fn.blocks if b in b.term.targets and graph(b) is not None]
    if len(heads) != 1:
        return False
    head = heads[0]
    loop, prefix = graph(head), graph(fn.entry)
    phis = [op for op in head.ops if op.kind == 'phi']
    if prefix is None or len(phis) != 1 or phis[0].results[0].type.kind != 'ptr' or len(pred[head]) != 2:
        return False
    pointer = phis[0]
    before = next(b for b in pred[head] if b is not head)
    initial = fn.defs[pointer.args[pointer.attrs['blocks'].index(before)]]
    update = fn.defs[pointer.args[pointer.attrs['blocks'].index(head)]]
    if (initial.kind != 'ptradd' or constant(initial.args[1]) != 32 or update.kind != 'ptradd'
            or update.args[0] != pointer.results[0] or constant(update.args[1]) != 32):
        return False
    a, b = prefix['flipped'], loop['flipped']
    flip = next(v for v in b.args if v != loop['inverse'].results[0])
    if flip not in a.args or head.term.args != b.results:
        return False
    flipped = fn.defs[bare(flip)]
    if (flipped.kind != 'add' or bare(flipped.args[0]) != fn.params[4]
            or constant(flipped.args[1]) not in (-1, 0xffffffff)):
        return False
    joins = [block for block in getattr(fn, 'return_joins', ()) if block in fn.blocks
             and sum(op.kind == 'phi' for op in block.ops) == 2]
    if len(joins) != 1:
        return False
    join = joins[0]
    address = next((op for op in join.ops if op.kind == 'phi' and op.results[0].type.kind == 'ptr'), None)
    mask = next((op for op in join.ops if op.kind == 'phi' and op.results[0].type == T.UINT), None)
    if address is None or mask is None or b.results[0] not in mask.args:
        return False
    returned = mask.attrs['blocks'][mask.args.index(b.results[0])]
    if (returned.term.kind != 'jmp' or returned.term.targets != (join,)
            or any(b.results[0] in op.args for block in fn.blocks for op in block.ops if op is not mask)):
        return False
    count = next((op for op in join.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'ctz_nonzero'), None)
    difference = next((op for op in join.ops if op.kind == 'sub' and bare(op.args[0]) == address.results[0]
                       and bare(op.args[1]) == fn.params[1]), None)
    if count is None or difference is None:
        return False
    broadcasts = [op for op in fn.entry.ops if op.kind == 'intrinsic'
                  and op.attrs['builtin'].op == 'broadcast128' and op.args[0] in fn.params[2:4]]
    if len(broadcasts) != 2:
        return False
    constants = [next((v for v in op.args if isinstance(constant(v), bytes)), None)
                 for op in (loop['xor'], loop['indexed'])]
    bits = loop['lookup'].args[0]
    if (any(v is None or v.type != T.V256 for v in constants) or bits != prefix['lookup'].args[0]
            or not isinstance(constant(bits), bytes)):
        return False

    fn.vector_classification = True
    fn.retain_vector_constants = True
    registers = fn.induction_registers = dict(getattr(fn, 'induction_registers', {}))
    vectors = fn.vector_registers = dict(getattr(fn, 'vector_registers', {}))

    def prefer(value, register):
        registers[value] = register
        while fn.defs[value].kind == 'cast':
            value = fn.defs[value].args[0]
            registers[value] = register

    # The loaded cursor and the preceding cursor advance together. The
    # latter provides the return address without keeping an old cursor live.
    previous = fn.make('phi', [pointer.results[0].type], attrs={'blocks': [before, head]})
    advance = fn.make('ptradd', [pointer.results[0].type], [previous.results[0], update.args[1]])
    previous.args = initial.args[0], advance.results[0]
    head.ops.insert(0, previous)
    head.ops.remove(update)
    head.ops.remove(b)
    b.args = flip, loop['inverse'].results[0]
    returned.ops.insert(0, b)
    converted = fn.make('cast', [address.results[0].type], advance.results)
    returned.ops.append(converted)
    at = address.attrs['blocks'].index(returned)
    address.args = tuple(converted.results[0] if n == at else value for n, value in enumerate(address.args))
    condition = fn.make('ne', [T.INT], [flip, loop['inverse'].results[0]])
    head.ops.extend([advance, update, condition])
    head.term = Term('br', condition.results, head.term.targets)
    prefer(initial.args[0], 'rax')
    for value in (previous.results[0], advance.results[0]):
        prefer(value, 'rax')
    for value in (pointer.results[0], initial.results[0], update.results[0]):
        prefer(value, 'rcx')
    for value in (flip, b.results[0]):
        prefer(value, 'rdx')
    for value in (loop['movemask8'].results[0], loop['inverse'].results[0]):
        prefer(value, 'rdi')
    prefer(address.results[0], 'rdi')
    prefer(mask.results[0], 'rcx')
    prefer(count.results[0], 'rax')
    prefer(difference.results[0], 'rdi')
    prefer(join.term.args[0], 'rax')

    # Complementing after the prefix's XOR avoids extending the mask's live
    # range. The loop observes equality first and applies its XOR on exit.
    raw = prefix['inverse'].args[0]
    combined = fn.make('xor', [T.UINT], [raw, flip])
    fn.entry.ops.insert(fn.entry.ops.index(a), combined)
    a.kind, a.args = 'not', combined.results
    for value in (prefix['movemask8'].results[0], combined.results[0], a.results[0], fn.entry.term.args[0]):
        prefer(value, 'rcx')

    for broadcast in broadcasts:
        user = next(op for op in prefix['shuffles'] if broadcast.results[0] in op.args)
        fn.entry.ops.remove(broadcast)
        fn.entry.ops.insert(fn.entry.ops.index(user), broadcast)
        vectors[broadcast.results[0]] = 'xmm%d' % fn.params[2:4].index(broadcast.args[0])
    table = fn.make('vector_constant', [T.V256], [bits])
    fn.entry.ops.insert(fn.entry.ops.index(prefix['lookup']), table)
    for op in (prefix['lookup'], loop['lookup']):
        op.args = tuple(table.results[0] if v == bits else v for v in op.args)
    vectors[table.results[0]] = 'xmm2'
    for n, value in enumerate(constants):
        materialized = fn.make('vector_constant', [T.V256], [value])
        before.ops.append(materialized)
        vectors[materialized.results[0]] = 'xmm%d' % (n+4)
        for op in head.ops:
            op.args = tuple(materialized.results[0] if v == value else v for v in op.args)
    for nodes, data_reg, low_reg, temporary in ((prefix, 'xmm2', 'xmm3', 'xmm4'), (loop, 'xmm6', 'xmm7', 'xmm8')):
        nodes['or'].args = nodes['or'].args[::-1]
        low = next(op for op in nodes['shuffles'] if op is not nodes['lookup'] and op.args[1] == nodes['load'].results[0])
        high = next(op for op in nodes['shuffles'] if op not in (low, nodes['lookup']))
        vectors[nodes['load'].results[0]] = data_reg
        vectors[low.results[0]] = low_reg
        vectors[nodes['xor'].results[0]] = temporary
        vectors[high.results[0]] = temporary
        vectors[nodes['or'].results[0]] = low_reg
        vectors[nodes['lshr16'].results[0]] = data_reg
        last = 'xmm4' if nodes is prefix else 'xmm6'
        for op in (nodes['indexed'], nodes['lookup'], nodes['final'], nodes['cmpeq_epi8']):
            vectors[op.results[0]] = last
    return True


def prepare_classification_tables(fn, functions):
    """Build four initialized words and pass their vectors to a byte scan.

    The empty input keeps the three known-zero words in registers. On the
    loop edge the independent one-hot bit is computed before the indexed
    read-modify-write, reducing the number of simultaneously live values.
    """
    if (fn.type.base != T.ULONG or len(fn.params) != 4
            or any(v.type.kind != 'ptr' for v in fn.params[:3])
            or fn.params[3].type != T.INT
            or getattr(fn, 'parameter_masks', {}).get(fn.params[3], 2) > 1
            or len(fn.allocas) != 1 or fn.allocas[0].attrs['size'] != 32
            or fn.allocas[0].attrs['alignment'] > 16
            or any(op.kind == 'asm' or op.attrs.get('volatile') for b in fn.blocks for op in b.ops)):
        return False

    def bare(value):
        while fn.defs[value].kind == 'cast':
            value = fn.defs[value].args[0]
        return value

    def constant(value):
        op = fn.defs[value]
        return op.attrs.get('value') if op.kind == 'constant' else None

    def binary(value, kind, right=None):
        op = fn.defs[value]
        return op if op.kind == kind and (right is None or constant(op.args[1]) == right) else None

    root = fn.allocas[0].results[0]

    def offset(value):
        if value == root:
            return 0
        op = fn.defs[value]
        if op.kind == 'cast' and op.args[0].type.kind == 'ptr':
            return offset(op.args[0])
        if op.kind == 'ptradd':
            base, step = offset(op.args[0]), constant(op.args[1])
            if base is not None and isinstance(step, int):
                return base+T.convert_integer(step, T.LONG)
        return None

    calls = [(b, op) for b in fn.blocks for op in b.ops if op.kind == 'call']
    if len(calls) != 1:
        return False
    loaded, call = calls[0]
    target = functions.get(callee(fn, call.args[0]))
    if (target is None or not getattr(target, 'vector_classification', False)
            or len(call.args) != 6 or loaded.term.kind != 'ret' or loaded.term.args != call.results
            or fn.defs[call.args[0]].kind != 'symbol' or fn.defs[call.args[0]] not in loaded.ops
            or tuple(bare(v) for v in (*call.args[1:3], call.args[-1])) != (*fn.params[:2], fn.params[3])):
        return False
    vectors = [fn.defs[bare(v)] for v in call.args[3:5]]
    if any(op.kind != 'intrinsic' or op.attrs['builtin'].op != 'constant64_reversed' for op in vectors):
        return False
    words = [fn.defs[bare(v)] for vector in vectors for v in vector.args]
    if (any(op.kind != 'load' or op.attrs['width'] != 8 or op.results[0].type != T.ULONG
            or op not in loaded.ops for op in words)
            or [offset(op.args[0]) for op in words] != [8, 0, 24, 16]):
        return False
    high, low, top, upper_low = words
    if any(fn.defs[op.args[0]].kind != 'ptradd' or fn.defs[op.args[0]] not in loaded.ops
           or fn.defs[fn.defs[op.args[0]].args[0]] in loaded.ops for op in (high, top, upper_low)):
        return False

    pred = fn.predecessors()
    heads = [b for b in fn.blocks if b in b.term.targets]
    if len(heads) != 1:
        return False
    head = heads[0]
    if len(pred[head]) != 2 or set(head.term.targets) != {head, loaded}:
        return False
    before = next(b for b in pred[head] if b is not head)
    if (len(fn.blocks) != 4 or before.term.kind != 'jmp' or pred[before] != [fn.entry]
            or len(before.ops) != 1 or fn.entry.term.kind != 'br'
            or fn.entry.term.targets != (before, loaded)):
        return False
    phis = [op for op in head.ops if op.kind == 'phi']
    pointers = [op for op in phis if op.results[0].type.kind == 'ptr']
    bytes_ = [op for op in phis if op.results[0].type == T.UCHAR]
    if len(phis) != 2 or len(pointers) != 1 or len(bytes_) != 1:
        return False
    pointer, byte = pointers[0], bytes_[0]
    initial = fn.defs[byte.args[byte.attrs['blocks'].index(before)]]
    next_byte = fn.defs[byte.args[byte.attrs['blocks'].index(head)]]
    step = fn.defs[pointer.args[pointer.attrs['blocks'].index(head)]]
    first_step = fn.defs[pointer.args[pointer.attrs['blocks'].index(before)]]
    if (initial.kind != 'load' or initial.attrs['width'] != 1 or initial.args != (fn.params[2],)
            or fn.entry.term.args != initial.results or next_byte.kind != 'load' or next_byte.attrs['width'] != 1
            or next_byte.args != pointer.results or head.term.args != next_byte.results
            or step.kind != 'ptradd' or step.args[0] != pointer.results[0] or constant(step.args[1]) != 1
            or first_step is not before.ops[0] or first_step.kind != 'ptradd'
            or first_step.args[0] != fn.params[2] or constant(first_step.args[1]) != 1):
        return False
    stores = [op for op in head.ops if op.kind == 'store']
    if len(stores) != 1 or stores[0].attrs['width'] != 8:
        return False
    store = stores[0]
    merge = binary(store.args[1], 'or')
    if merge is None:
        return False
    read, shifted = (fn.defs[v] for v in merge.args)
    if read.kind != 'load':
        read, shifted = shifted, read
    if (read.kind != 'load' or read.attrs['width'] != 8 or read.args != store.args[:1]
            or shifted.kind != 'shl' or shifted.results[0].type != T.ULONG):
        return False
    wide = fn.defs[shifted.args[0]]
    bit = fn.defs[bare(shifted.args[0])]
    count = binary(shifted.args[1], 'mul', 8)
    if wide.kind != 'cast' or bit.kind != 'shl' or constant(bit.args[0]) != 1 or count is None:
        return False
    nibble = binary(bit.args[1], 'and', 7)
    lane = binary(count.args[0], 'and', 7)
    if nibble is None or lane is None:
        return False
    shr = binary(nibble.args[0], 'shr', 4)
    byte_value = lane.args[0]
    byte_cast = fn.defs[byte_value]
    if (shr is None or shr.args[0] != byte_value or byte_cast.kind != 'cast'
            or byte_cast.args != byte.results or byte_value.type != T.UINT):
        return False
    address = fn.defs[store.args[0]]
    if address.kind != 'ptradd' or offset(address.args[0]) != 0:
        return False
    scale = binary(address.args[1], 'mul', 8)
    index_cast = fn.defs[scale.args[0]] if scale is not None else None
    if index_cast is None or index_cast.kind != 'cast':
        return False
    index = binary(index_cast.args[0], 'or')
    if index is None:
        return False
    doubled, bottom = (fn.defs[v] for v in index.args)
    if doubled.kind != 'shl':
        doubled, bottom = bottom, doubled
    if (doubled.kind != 'shl' or constant(doubled.args[1]) != 1
            or bottom.kind != 'and' or constant(bottom.args[1]) != 1):
        return False
    highest = binary(doubled.args[0], 'shr', 7)
    bottom_shr = binary(bottom.args[0], 'shr', 3)
    if highest is None or bottom_shr is None or highest.args[0] != byte_value or bottom_shr.args[0] != byte_value:
        return False
    ordered = [pointer, byte, byte_cast, shr, nibble, bit, wide, lane, count, shifted,
               highest, doubled, bottom_shr, bottom, index, index_cast, scale, address, read, merge,
               store, next_byte, step]
    if set(ordered) != set(head.ops) or len(ordered) != len(head.ops):
        return False
    fills = [op for op in fn.entry.ops if op.kind == 'fill']
    initial_stores = [op for op in fn.entry.ops if op.kind == 'store']
    if (len(fills) != 1 or offset(fills[0].args[0]) != 0 or constant(fills[0].args[1]) != 0
            or constant(fills[0].args[2]) != 32 or len(initial_stores) != 4
            or {offset(op.args[0]) for op in initial_stores} != {0, 8, 16, 24}
            or any(op.attrs['width'] != 8 for op in initial_stores)
            or any(bare(op.args[1]) != fn.params[3] if offset(op.args[0]) == 0 else constant(op.args[1]) != 0
                   for op in initial_stores)
            or any(op.kind in ('call', 'copy') or op.attrs.get('memory')
                   for op in fn.entry.ops if op not in [*fills, *initial_stores, initial])
            or fn.entry.ops.index(fills[0]) > min(fn.entry.ops.index(op) for op in initial_stores)
            or any(op.attrs.get('memory') for op in loaded.ops if op not in [*words, call])):
        return False

    # All reads in the original exit are initialized words of this stack
    # object. Increasing its alignment makes the paired vector reads valid.
    fn.allocas[0].attrs['alignment'] = 16
    fn.scalar_classification = True
    fn.ymm_tail_callees = {target.name}
    fills[0].attrs['domain'] = 'fp'
    fn.entry.ops = [op for op in fn.entry.ops if op not in initial_stores or offset(op.args[0]) == 0]
    head.ops = ordered
    registers = fn.induction_registers = dict(getattr(fn, 'induction_registers', {}))

    def prefer(op, register):
        value = op.results[0]
        registers[value] = register
        while fn.defs[value].kind == 'cast':
            value = fn.defs[value].args[0]
            if value.type.bits != op.results[0].type.bits:
                break
            registers[value] = register

    initialized = next(op for op in initial_stores if offset(op.args[0]) == 0)
    registers[initialized.args[1]] = 'rax'
    for ops, register in (([initial, next_byte, byte, byte_cast, bottom_shr, bottom, index, index_cast], 'r8'),
                          ([shr, nibble, bit, wide, shifted], 'r9'),
                          ([lane, count, highest, doubled], 'r10'),
                          ([pointer, step, first_step, high], 'rdx'), ([top], 'rax')):
        for op in ops:
            prefer(op, register)
    one = fn.make('scalar_constant', [T.UINT], [bit.args[0]])
    before.ops.append(one)
    registers[one.results[0]] = 'rax'
    bit.args = one.results[0], bit.args[1]
    shr.attrs['byte_shift'] = True
    # SHLX observes only the low six count bits. Here 8*(byte & 7)
    # and 8*byte have the same count; the latter is one scaled LEA.
    count.attrs['scaled_shift_count'] = byte_value
    index.attrs['scaled_sum'] = (bottom.results[0], highest.results[0], 2)
    store.attrs['rmw_or'] = shifted.results[0]
    for op in (lane, doubled, read, merge):
        op.attrs['selection_folded'] = True

    empty = fn.block('classification.empty')
    join = fn.block('classification.return')
    vectors = fn.vector_registers = dict(getattr(fn, 'vector_registers', {}))

    def vector_load(block, pointer):
        builtin = BUILTINS['_mm_load_si128']
        op = fn.make('intrinsic', [T.V128], [pointer],
                     {'builtin': builtin, 'memory': 'read', 'width': 16, 'alignment': 16})
        block.ops.append(op)
        return op.results[0]

    # Keep the two upper scalar words to preserve their insertion order;
    # the remaining low halves can be loaded directly into vector registers.
    loaded.ops = [fn.defs[high.args[0]], high, fn.defs[top.args[0]], top, fn.defs[upper_low.args[0]]]
    upper = vector_load(loaded, upper_low.args[0])
    loaded.term = Term('jmp', targets=(join,))
    zero = fn.make('vector_constant', [T.V128], [fn.constant(T.V128, bytes(16))])
    empty.ops.append(zero)
    zeros = []
    for register in ('rax', 'rdx'):
        op = fn.make('scalar_constant', [T.ULONG], [fn.constant(T.ULONG, 0)])
        empty.ops.append(op)
        registers[op.results[0]] = register
        zeros.append(op.results[0])
    empty.term = Term('jmp', targets=(join,))

    def phi(type_, a, b, register):
        op = fn.make('phi', [type_], [a, b], {'blocks': [loaded, empty]})
        join.ops.append(op)
        (vectors if type_.kind == 'vector' else registers)[op.results[0]] = register
        return op.results[0]

    hi = phi(T.ULONG, high.results[0], zeros[1], 'rdx')
    last = phi(T.ULONG, top.results[0], zeros[0], 'rax')
    base = phi(T.V128, upper, zero.results[0], 'xmm1')
    vectors[upper] = vectors[zero.results[0]] = 'xmm1'
    low = vector_load(join, root)
    table_values = []
    for vector, word, register in ((low, hi, 'xmm0'), (base, last, 'xmm1')):
        builtin = BUILTINS['_mm_insert_epi64']
        insert = fn.make('intrinsic', [T.V128], [vector, word, fn.constant(T.INT, 1)], {'builtin': builtin})
        join.ops.append(insert)
        vectors[vector] = vectors[insert.results[0]] = register
        table_values.append(insert.results[0])
    call.args = call.args[:3]+tuple(table_values)+call.args[-1:]
    join.ops.extend([fn.defs[call.args[0]], call])
    join.term = Term('ret', call.results)
    fn.entry.term.targets = before, empty
    fn.layout_owners = {**getattr(fn, 'layout_owners', {}), loaded: head, empty: loaded, join: empty}
    return True
