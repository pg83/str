"""Share full-width byte masks before scalar comparison return loads."""
from . import types as T
from .ir import Term


def share_byte_counts(fn, incoming):
    counts = []
    for block, value in incoming:
        cast = fn.defs[value]
        scan = fn.defs[cast.args[0]] if cast.kind == 'cast' else None
        if (scan is not None and scan in block.ops and scan.kind == 'intrinsic'
                and scan.attrs['builtin'].op == 'ctz_nonzero' and scan.args[0].type.bits == 64):
            counts.append((block, scan))
    if len(counts) < 2 or any(scan.attrs != counts[0][1].attrs for _, scan in counts):
        return incoming, None
    target = fn.block('return.byte.count')
    mask = fn.make('phi', [counts[0][1].args[0].type], [scan.args[0] for _, scan in counts],
                   {'blocks': [b for b, _ in counts]})
    count = fn.make('intrinsic', [counts[0][1].results[0].type], mask.results, dict(counts[0][1].attrs))
    index = fn.make('cast', [T.LONG], count.results)
    target.ops = [mask, count, index]
    target.term = counts[0][0].term
    blocks = {b for b, _ in counts}
    for block in blocks:
        block.term = Term('jmp', targets=(target,))
    fn.layout_owners = dict(getattr(fn, 'layout_owners', {}))
    predecessors = fn.predecessors()[counts[-1][0]]
    fn.layout_owners[target] = predecessors[0] if len(predecessors) == 1 else counts[-1][0]
    fn.byte_compare_count_return = target
    return [(b, v) for b, v in incoming if b not in blocks] + [(target, index.results[0])], target


def pack_byte_masks(fn):
    registers, vectors = fn.induction_registers, dict(getattr(fn, 'vector_registers', {}))
    owners = {v: b for b in fn.blocks for op in b.ops for v in op.results}

    def bare(value):
        while fn.defs[value].kind == 'cast' and min(value.type.bits, fn.defs[value].args[0].type.bits) >= 32:
            value = fn.defs[value].args[0]
        return value

    def constant(value):
        op = fn.defs[value]
        return op.attrs['value'] if op.kind == 'constant' else None

    def prefer(value, register):
        registers[value] = register
        while fn.defs[value].kind == 'cast':
            value = fn.defs[value].args[0]
            registers[value] = register

    def inverse(value):
        op = fn.defs[bare(value)]
        return op if op.kind == 'not' and op.results[0].type.bits == 32 else None

    def ordered(block, roots):
        original, seen, output = set(block.ops), set(), []
        def visit(op):
            if op not in original or op in seen:
                return
            for value in op.args:
                visit(fn.defs[value])
            seen.add(op)
            output.append(op)
        for root in roots:
            visit(root)
        for op in block.ops:
            visit(op)
        block.ops = output

    for value in fn.byte_compare_offsets:
        prefer(value, 'rcx')
    count_join = getattr(fn, 'byte_compare_count_return', None)
    if count_join is not None:
        for op in count_join.ops:
            if op.kind == 'phi':
                prefer(op.results[0], 'rax')
                for value in op.args:
                    prefer(value, 'rax')
    for block in fn.blocks:
        compares = [op for op in block.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'cmpeq_epi8']
        for op in block.ops:
            if op.kind == 'and' and constant(op.args[1]) == 4095:
                prefer(op.results[0], 'rax')
        # A complemented low word retained across the size branch is the
        # prefix. The high word supplies the accumulator of the packed mask.
        for op in list(block.ops):
            if op.kind != 'or' or op.results[0].type.bits != 64:
                continue
            low, shifted = inverse(op.args[0]), fn.defs[op.args[1]]
            if low is None or shifted.kind != 'shl' or constant(shifted.args[1]) != 32:
                continue
            high = inverse(shifted.args[0])
            if high is None:
                continue
            if owners.get(low.results[0]) is not block:
                prefer(low.args[0], 'r8')
                prefer(low.results[0], 'r8')
                prefer(op.args[0], 'r8')
                prefer(high.args[0], 'rax')
                prefer(high.results[0], 'rax')
                prefer(shifted.args[0], 'rax')
                prefer(shifted.results[0], 'rax')
                prefer(op.results[0], 'rax')
                op.args = op.args[::-1]
                continue
            left = fn.make('cast', [T.ULONG], low.args)
            right = fn.make('cast', [T.ULONG], high.args)
            shift = fn.make('shl', [T.ULONG], [right.results[0], shifted.args[1]])
            packed = fn.make('or', [T.ULONG], [shift.results[0], left.results[0]])
            at = block.ops.index(op)
            block.ops[at:at] = [left, right, shift, packed]
            op.kind, op.args = 'not', packed.results
            prefer(low.args[0], 'rcx')
            for value in (high.args[0], right.results[0], shift.results[0], packed.results[0], op.results[0]):
                prefer(value, 'r8')
        for op in list(block.ops):
            if op.kind != 'or' or op.results[0].type.bits != 64:
                continue
            combined, sentinel = (fn.defs[bare(v)] for v in op.args)
            if (combined.kind != 'or' or sentinel.kind != 'shl' or constant(sentinel.args[0]) != 1
                    or constant(sentinel.args[1]) is not None):
                continue
            prefix, shift = combined.args[0], fn.defs[combined.args[1]]
            if shift.kind != 'shl' or constant(shift.args[1]) is not None or constant(shift.args[0]) is not None:
                continue
            marked = fn.make('or', [T.ULONG], [shift.results[0], sentinel.results[0]])
            block.ops.insert(block.ops.index(op), marked)
            op.args = marked.results[0], prefix
            prefer(marked.results[0], 'rax')
            prefer(shift.results[0], 'rax')
            prefer(shift.args[0], 'rax')
            inverse_ = inverse(shift.args[0])
            if inverse_ is not None:
                prefer(inverse_.args[0], 'rax')
                prefer(inverse_.results[0], 'rax')
            prefer(prefix, 'r8')
            inverse_ = inverse(prefix)
            if inverse_ is not None:
                prefer(inverse_.args[0], 'r8')
                prefer(inverse_.results[0], 'r8')
        for op in block.ops:
            if op.kind == 'select' and op.results[0].type.bits == 64:
                first = fn.defs[bare(op.args[1])]
                if first.kind == 'cast':
                    first = fn.defs[first.args[0]]
                if first.kind == 'intrinsic' and first.attrs['builtin'].op == 'ctz_nonzero':
                    prefer(op.args[1], 'r9')
                    prefer(first.args[0], 'r9')
                tail = fn.defs[bare(op.args[2])]
                while tail.kind == 'add' and constant(tail.args[1]) is not None:
                    tail = fn.defs[bare(tail.args[0])]
                if tail.kind == 'add' and tail.args[1] == fn.params[2]:
                    prefer(tail.args[0], 'rcx')
            if op.kind == 'or' and constant(op.args[1]) == 1 << 63:
                fn.scalar_constant_registers = dict(getattr(fn, 'scalar_constant_registers', {}))
                fn.scalar_constant_registers[op.args[1]] = 'rcx'
        if len(compares) == 2:
            roots = [fn.defs[op.args[0]] for op in compares]
            for n, op in enumerate(compares):
                roots.extend([fn.defs[op.args[1]], op])
                roots.extend(user for user in block.ops if user.kind == 'intrinsic'
                             and user.attrs['builtin'].op == 'movemask8' and user.args == op.results)
                vectors[op.args[0]] = 'xmm%d' % n
                vectors[op.results[0]] = 'xmm0'
            ordered(block, roots)
    fn.vector_registers = vectors
    return True
