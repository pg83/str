"""Common prefixes and indexed return loads for bounded byte comparisons."""
from . import types as T
from .ir import Term, dominators
from .facts import possible_bits
from .vexcompares import share_byte_counts, pack_byte_masks


def prepare_byte_compares(fn):
    if (fn.type.base != T.INT or len(fn.params) != 3
            or any(v.type.kind != 'ptr' for v in fn.params[:2])
            or fn.params[2].type != T.ULONG):
        return False
    calls = [(b, op) for b in fn.blocks for op in b.ops if op.kind == 'call']
    memory = [op for b in fn.blocks for op in b.ops if op.kind != 'call' and op.attrs.get('memory')]
    if (not calls or any(b.term.kind != 'ret' for b, _ in calls)
            or any('write' in op.attrs['memory'] or op.attrs.get('volatile') for op in memory)
            or any(op.attrs['width'] not in (1, 16, 32) for op in memory)):
        return False
    width = max((op.attrs['width'] for op in memory), default=0)
    if width == 32 and not any(op.kind == 'intrinsic' and op.attrs['builtin'].op == 'movemask8'
                               and op.args[0].type.bits == 256 for b in fn.blocks for op in b.ops):
        return False
    dom = dominators(fn)
    if any(head in dom[tail] for tail in fn.blocks for head in tail.term.targets):
        return False

    def bare(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value

    def constant(value):
        op = fn.defs[value]
        return op.attrs['value'] if op.kind == 'constant' else None

    def difference(block):
        if block.term.kind != 'ret' or len(block.term.args) != 1:
            return None
        result = fn.defs[block.term.args[0]]
        if result.kind != 'sub':
            return None
        loads, addresses = [], []
        for n, value in enumerate(result.args):
            cast = fn.defs[value]
            if cast.kind != 'cast' or cast.results[0].type != T.INT:
                return None
            load = fn.defs[cast.args[0]]
            if (load.kind != 'load' or load not in block.ops or load.attrs['width'] != 1
                    or load.results[0].type.signed or load.attrs.get('volatile')):
                return None
            address = fn.defs[bare(load.args[0])]
            if address.kind != 'ptradd' or bare(address.args[0]) != fn.params[n]:
                return None
            loads.append(load)
            addresses.append(address)
        if addresses[0].args[1] != addresses[1].args[1] or addresses[0].args[1].type != T.LONG:
            return None
        return block, addresses[0].args[1], loads, addresses

    returns = [part for block in fn.blocks if (part := difference(block)) is not None]
    if len(returns) < 3:
        return False
    # A common, unconditional read prefix of both arms may precede the size
    # test. Keep the rest of each arm where its bounds have been established.
    pred = fn.predecessors()
    for block in fn.blocks:
        if block.term.kind != 'br':
            continue
        left, right = block.term.targets
        if pred[left] != [block] or pred[right] != [block]:
            continue
        replacements, pairs = {}, []
        for a, b in zip(left.ops, right.ops):
            if (a.kind != b.kind or a.kind in ('phi', 'call', 'asm') or a.attrs != b.attrs
                    or len(a.results) != len(b.results) or len(a.results) != 1
                    or a.results[0].type != b.results[0].type
                    or a.attrs.get('volatile') or 'write' in a.attrs.get('memory', '')):
                break
            arguments = tuple(replacements.get(v, v) for v in b.args)
            if arguments != a.args:
                if (a.kind not in ('shl', 'shr') or arguments[0] != a.args[0]
                        or constant(arguments[1]) is None or constant(arguments[1]) != constant(a.args[1])):
                    break
            replacements[b.results[0]] = a.results[0]
            pairs.append((a, b))
        if sum('read' in a.attrs.get('memory', '') for a, _ in pairs) < 4:
            continue
        left.ops = left.ops[len(pairs):]
        right.ops = right.ops[len(pairs):]
        at = block.ops.index(fn.defs[block.term.args[0]])
        block.ops[at:at] = [a for a, _ in pairs]
        for other in fn.blocks:
            for op in other.ops:
                op.args = tuple(replacements.get(v, v) for v in op.args)
            other.term.args = tuple(replacements.get(v, v) for v in other.term.args)

    join = fn.block('return.byte.difference')
    phi = fn.make('phi', [T.LONG], [index for _, index, _, _ in returns],
                  {'blocks': [b for b, _, _, _ in returns]})
    join.ops.append(phi)
    values = []
    for load, address in zip(returns[0][2], returns[0][3]):
        pointer = fn.make('ptradd', [address.results[0].type], [address.args[0], phi.results[0]])
        byte = fn.make('load', [load.results[0].type], pointer.results, dict(load.attrs))
        extended = fn.make('cast', [T.INT], byte.results)
        join.ops.extend([pointer, byte, extended])
        values.append(extended.results[0])
    result = fn.make('sub', [T.INT], values)
    join.ops.append(result)
    join.term = Term('ret', result.results)
    for block, _, _, _ in returns:
        block.term = Term('jmp', targets=(join,))

    # A previous mask phi shared one trailing-zero count. With other returned
    # indices now joining here, put that pure count on its original edges.
    expanded = []
    removed = set()
    for block, index, _, _ in returns:
        cast = fn.defs[index]
        scan = fn.defs[cast.args[0]] if cast.kind == 'cast' else None
        mask = fn.defs[scan.args[0]] if scan is not None and scan.kind == 'intrinsic' else None
        if (scan is None or scan.kind != 'intrinsic' or scan.attrs['builtin'].op not in ('ctz_nonzero', 'tzcnt')
                or mask.kind != 'phi' or mask not in block.ops
                or any(parent.term.kind != 'jmp' or parent.term.targets != (block,) for parent in mask.attrs['blocks'])):
            expanded.append((block, index))
            continue
        internal = {v for op in block.ops for v in op.results}
        if any(v in internal for other in fn.blocks if other not in (block, join)
               for v in (*other.term.args, *(v for op in other.ops for v in op.args))):
            expanded.append((block, index))
            continue
        for parent, incoming in zip(mask.attrs['blocks'], mask.args):
            counted = fn.make(scan.kind, [scan.results[0].type], [incoming], dict(scan.attrs))
            converted = fn.make('cast', [T.LONG], counted.results)
            parent.ops.extend([counted, converted])
            parent.term = Term('jmp', targets=(join,))
            expanded.append((parent, converted.results[0]))
        removed.add(block)
    count_join = None
    if width == 32:
        expanded, count_join = share_byte_counts(fn, expanded)
    phi.attrs['blocks'], phi.args = [b for b, _ in expanded], tuple(v for _, v in expanded)
    fn.blocks = [b for b in fn.blocks if b not in removed]
    registers = fn.induction_registers = dict(getattr(fn, 'induction_registers', {}))
    registers[phi.results[0]] = 'rcx'
    for _, index in expanded:
        registers[index] = 'rcx'
        op = fn.defs[index]
        if op.kind == 'cast':
            registers[op.args[0]] = 'rcx'
    owners = fn.layout_owners = {b: p for b, p in getattr(fn, 'layout_owners', {}).items()
                               if b not in removed and p not in removed}
    owner = min(expanded, key=lambda pair: sum(bool(op.attrs.get('memory')) for op in pair[0].ops))[0]
    owners[join] = owner
    fn.return_joins = [b for b in getattr(fn, 'return_joins', ()) if b not in removed] + ([count_join] if count_join else []) + [join]
    fn.byte_compares, fn.byte_compare_return = True, join
    fn.byte_compare_width = width
    fn.byte_compare_offsets = set()
    for op in fn.entry.ops:
        if op.kind == 'sub' and op.args[0] == fn.params[2] and constant(op.args[1]) == 1:
            op.kind, op.args = 'add', (op.args[0], fn.constant(T.ULONG, -1))
            fn.byte_compare_offsets.add(op.results[0])
    for block in fn.blocks:
        if block.term.kind != 'br':
            continue
        test = fn.defs[block.term.args[0]]
        if test.kind == 'lt' and test.args[0] in fn.byte_compare_offsets and constant(test.args[1]) == 32:
            test.kind, test.args = 'le', (test.args[0], fn.constant(T.ULONG, 31))
        elif width == 16 and test.kind == 'le' and test.args[0] == fn.params[2] and constant(test.args[1]) in (64, 128):
            test.kind, test.args = 'lt', (test.args[0], fn.constant(T.ULONG, constant(test.args[1])+1))
    return True


def pack_byte_compare_masks(fn):
    if not getattr(fn, 'byte_compares', False):
        return False
    if fn.byte_compare_width == 32:
        return pack_byte_masks(fn)
    registers, vectors = fn.induction_registers, {}
    owners = {v: b for b in fn.blocks for op in b.ops for v in op.results}

    def constant(value):
        op = fn.defs[value]
        return op.attrs['value'] if op.kind == 'constant' else None

    def defined(value):
        op = fn.defs[value]
        while op.kind == 'cast' and op.args[0].type.bits == value.type.bits:
            value, op = op.args[0], fn.defs[op.args[0]]
        return op

    def inverse(value):
        op = fn.defs[value]
        while op.kind == 'cast' and min(value.type.bits, op.args[0].type.bits) >= 32:
            value, op = op.args[0], fn.defs[op.args[0]]
        if op.kind != 'xor' or constant(op.args[1]) != 65535 or possible_bits(fn, op.args[0]) > 65535:
            return None
        return op

    def prefer(value, register):
        registers[value] = register
        op = fn.defs[value]
        while op.kind == 'cast':
            value = op.args[0]
            registers[value] = register
            op = fn.defs[value]

    def pair(block, before, low, high, register):
        # The high mask's upper bits disappear in a 32-bit shift by 16.
        # Complementing the entire word therefore replaces xor 0xffff.
        inverted = fn.make('not', [T.UINT], [high.args[0]])
        shifted = fn.make('shl', [T.UINT], [inverted.results[0], fn.constant(T.INT, 16)])
        combined = fn.make('or', [T.UINT], [shifted.results[0], low.results[0]])
        block.ops[block.ops.index(before):block.ops.index(before)] = [inverted, shifted, combined]
        for value in (inverted.results[0], shifted.results[0], combined.results[0], high.args[0]):
            prefer(value, register)
        prefer(low.results[0], 'rcx')
        prefer(low.args[0], 'rcx')
        return combined.results[0]

    for block in fn.blocks:
        compares = [op for op in block.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'cmpeq_epi8']
        # A prefix defined before the size branch has already packed its low
        # two masks. Pack the remaining two locally before shifting by 32.
        for op in list(block.ops):
            if op.kind != 'or' or op.results[0].type.bits != 64:
                continue
            lower, last = (fn.defs[v] for v in op.args)
            if lower.kind != 'or' or last.kind != 'shl' or constant(last.args[1]) != 48:
                continue
            prefix, third = lower.args[0], fn.defs[lower.args[1]]
            if owners.get(prefix) is block or possible_bits(fn, prefix) >= 1 << 32 or third.kind != 'shl' or constant(third.args[1]) != 32:
                continue
            low, high = inverse(third.args[0]), inverse(last.args[0])
            if low is None or high is None:
                continue
            packed = pair(block, op, low, high, 'rax')
            wide = fn.make('cast', [T.ULONG], [packed])
            shifted = fn.make('shl', [T.ULONG], [wide.results[0], fn.constant(T.INT, 32)])
            at = block.ops.index(op)
            block.ops[at:at] = [wide, shifted]
            op.args = shifted.results[0], prefix
            for value in (wide.results[0], shifted.results[0], op.results[0]):
                prefer(value, 'rax')
        for op in list(block.ops):
            if op.kind != 'or' or op.results[0].type.bits != 64:
                continue
            low, shifted = inverse(op.args[0]), fn.defs[op.args[1]]
            if low is None or shifted.kind != 'shl' or constant(shifted.args[1]) != 16:
                continue
            high = inverse(shifted.args[0])
            if high is None:
                continue
            register = 'r8' if len(compares) >= 4 or block.term.kind == 'br' else 'rdx'
            packed = pair(block, op, low, high, register)
            op.kind, op.args = 'cast', (packed,)
            prefer(op.results[0], register)
        used = {v for b in fn.blocks for v in (*b.term.args, *(v for item in b.ops for v in item.args))}
        for op in list(block.ops):
            if op.kind == 'shl' and op.results[0].type.bits == 64 and constant(op.args[1]) == 48:
                if op.results[0] not in used:
                    continue
                old = inverse(op.args[0])
                if old is not None:
                    negative = fn.make('not', [T.UINT], [old.args[0]])
                    wide = fn.make('cast', [T.ULONG], negative.results)
                    at = block.ops.index(op)
                    block.ops[at:at] = [negative, wide]
                    op.args = wide.results[0], op.args[1]
                    for value in (old.args[0], negative.results[0], wide.results[0], op.results[0]):
                        prefer(value, 'r8')
            if (len(compares) >= 4 and op.kind == 'or' and op.results[0].type.bits == 64
                    and fn.defs[op.args[1]].kind == 'shl'):
                shift = fn.defs[op.args[1]]
                amount = constant(shift.args[1])
                if amount in (32, 48):
                    register = 'rcx' if amount == 32 else 'r8'
                    prefer(shift.results[0], register)
                    mask = inverse(shift.args[0])
                    if mask is not None:
                        prefer(mask.args[0], register)
                        prefer(mask.results[0], register)
                    op.args = op.args[::-1]
                    prefer(op.results[0], register)
        # Combine a variable-position tail with its sentinel independently
        # of the first 32-byte mask, so that the two shifts stay parallel.
        for op in list(block.ops):
            if op.kind != 'or' or op.results[0].type.bits != 64:
                continue
            combined, sentinel = (defined(v) for v in op.args)
            if (sentinel.kind == 'shl' and constant(sentinel.args[0]) == 1
                    and combined.kind == 'cast' and combined.args[0].type == T.UINT):
                prefer(op.results[0], 'rdx')
            if (combined.kind != 'or' or sentinel.kind != 'shl' or constant(sentinel.args[0]) != 1
                    or constant(sentinel.args[1]) is not None):
                continue
            earlier, second = (fn.defs[v] for v in combined.args)
            if earlier.kind != 'or' or second.kind != 'shl' or constant(second.args[1]) is not None:
                continue
            prefix, first = earlier.args[0], fn.defs[earlier.args[1]]
            if first.kind != 'shl' or constant(first.args[1]) is not None:
                continue
            marked = fn.make('or', [T.ULONG], [first.results[0], sentinel.results[0]])
            other = fn.make('or', [T.ULONG], [second.results[0], prefix])
            at = block.ops.index(op)
            block.ops[at:at] = [marked, other]
            op.args = other.results[0], marked.results[0]
            for shift, register in ((first, 'r9'), (second, 'r10')):
                prefer(shift.results[0], register)
                mask = inverse(shift.args[0])
                if mask is not None:
                    prefer(mask.results[0], register)
                    prefer(mask.args[0], register)
            prefer(marked.results[0], 'r9')
            prefer(other.results[0], 'r10')
            prefer(op.results[0], 'r10')
            for zero in fn.blocks:
                if zero.term.kind == 'ret' and zero.term.args and constant(zero.term.args[0]) == 0:
                    fn.layout_owners[zero] = block
        for op in block.ops:
            if op.kind != 'select' or op.results[0].type.bits != 64:
                continue
            first = fn.defs[op.args[1]]
            if first.kind == 'cast':
                first = fn.defs[first.args[0]]
            if first.kind == 'intrinsic' and first.attrs['builtin'].op in ('ctz_nonzero', 'tzcnt'):
                prefer(op.args[1], 'r9')
                prefer(first.args[0], 'r9')
            tail = fn.defs[op.args[2]]
            while tail.kind == 'add' and constant(tail.args[1]) is not None:
                tail = fn.defs[tail.args[0]]
            if tail.kind == 'add' and tail.args[1] == fn.params[2]:
                prefer(tail.args[0], 'rcx')
                last = fn.defs[tail.args[0]]
                if last.kind == 'cast':
                    last = fn.defs[last.args[0]]
                if last.kind == 'intrinsic' and last.attrs['builtin'].op in ('ctz_nonzero', 'tzcnt'):
                    prefer(last.args[0], 'r8')
        for op in block.ops:
            if op.kind == 'or' and constant(op.args[1]) == 1 << 63:
                fn.scalar_constant_registers = dict(getattr(fn, 'scalar_constant_registers', {}))
                fn.scalar_constant_registers[op.args[1]] = 'rcx'
        if len(compares) >= 2:
            original, seen, output = set(block.ops), set(), []
            def visit(op):
                if op not in original or op in seen:
                    return
                for value in op.args:
                    visit(fn.defs[value])
                seen.add(op)
                output.append(op)
            for op in compares:
                visit(fn.defs[op.args[0]])
            variable = any(op.kind == 'shl' and constant(op.args[1]) is None
                           and constant(op.args[0]) is None for op in block.ops)
            for n, op in enumerate(compares):
                visit(fn.defs[op.args[1]])
                if not variable or n == 0:
                    visit(op)
                vectors[op.args[0]] = 'xmm%d' % n
                vectors[op.args[1]] = vectors[op.results[0]] = 'xmm%d' % (len(compares) if n == 0 else n-1)
            for op in block.ops:
                visit(op)
            block.ops = output
    fn.vector_registers = vectors
    return True
