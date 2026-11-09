"""Concrete pass recipe for the AVX2 strcmp entry point."""
from . import types as T
from .prefixes import string_bare


def prepare_strcmp(fn):
    def operations(block, name):
        return [op for op in block.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == name]

    stages = []
    for block in fn.blocks:
        tests = [op for op in operations(block, 'andnot') if op.results[0].type == T.V256]
        if len(tests) != 1:
            continue
        test = tests[0]
        zero, equal = (fn.defs[v] for v in test.args)
        if (zero.kind != 'intrinsic' or equal.kind != 'intrinsic'
                or zero.attrs['builtin'].op != equal.attrs['builtin'].op or zero.attrs['builtin'].op != 'cmpeq_epi8'
                or zero.args[0] != equal.args[0]
                or fn.defs[zero.args[1]].kind != 'constant' or fn.defs[zero.args[1]].attrs['value'] != bytes(32)):
            return False
        mask = next(op for op in operations(block, 'movemask8') if op.args == test.results)
        stages.append((block, fn.defs[zero.args[0]], equal, zero, test, mask))
    if len(stages) != 4:
        return False
    scalar = next((b for b in fn.blocks if operations(b, 'xor')
                   and any(op.results[0].type == T.V128 for op in operations(b, 'xor'))), None)
    if scalar is None:
        return False
    tail = next((b for b in fn.blocks if any(op.kind == 'call' for op in b.ops)), None)
    if tail is None or sum(op.kind == 'call' for b in fn.blocks for op in b.ops) != 1:
        return False
    page = fn.defs[fn.defs[fn.entry.term.args[0]].args[0]]
    if (page.kind != 'and' or fn.defs[page.args[1]].kind != 'constant'
            or fn.defs[page.args[1]].attrs['value'] != 4095):
        return False
    both = stages[0][0].term.targets[1]
    short = both.term.targets[1]
    call = next(op for op in tail.ops if op.kind == 'call')
    argument = fn.defs[call.args[1]]
    if argument.kind != 'phi' or short not in argument.attrs['blocks']:
        return False
    origin = fn.defs[string_bare(fn, stages[1][1].args[0])]
    if (origin.kind != 'ptradd' or fn.defs[origin.args[1]].kind != 'constant'
            or fn.defs[origin.args[1]].attrs['value'] != 32):
        return False
    before = argument.args[argument.attrs['blocks'].index(short)]
    first_return = stages[0][0].term.targets[0]
    scalar_return, scalar_continue = scalar.term.targets
    scalar_guard = fn.entry.term.targets[1]
    aligned_tail = stages[-1][0].term.targets[1]
    registers = fn.induction_registers = dict(getattr(fn, 'induction_registers', {}))
    vectors = fn.vector_registers = dict(getattr(fn, 'vector_registers', {}))

    def prefer(value, physical):
        registers[value] = physical
        while fn.defs[value].kind == 'cast':
            value = fn.defs[value].args[0]
            registers[value] = physical

    prefer(page.results[0], 'rcx')
    prefer(argument.results[0], 'rax')
    scalar_mask = operations(scalar, 'movemask8')[0]
    prefer(scalar_mask.results[0], 'rax')
    for op in scalar_continue.ops:
        if op.kind == 'ptradd' and string_bare(fn, op.args[0]) == fn.params[0]:
            prefer(op.results[0], 'rdi')
    stages[1][0].ops.remove(origin)
    both.ops.insert(0, origin)
    for op in short.ops:
        op.args = tuple(origin.results[0] if v == before else v for v in op.args)
    argument.args = tuple(origin.results[0] if v == before else v for v in argument.args)
    registers[origin.results[0]] = 'rax'
    fn.word_return_bases = {origin.results[0]}
    for (block, data, equal, zero, test, mask), colors in zip(stages, (
            (0, 2, 0, 0), (1, 2, 1, 1), (1, 2, 0, 0), (0, 2, 0, 0))):
        for op, color in zip((data, equal, zero, test), colors):
            vectors[op.results[0]] = 'ymm%d' % color
        block.ops.remove(zero)
        block.ops.insert(block.ops.index(equal) + 1, zero)
        prefer(mask.results[0], 'rax')
        prefer(block.term.args[0], 'rax')
    scalar_data = next(op for op in operations(scalar, 'load')
                       if string_bare(fn, op.args[0]) == fn.params[0])
    inversion = operations(scalar, 'xor')[0]
    equal = fn.defs[inversion.args[0]]
    zero = next(op for op in operations(scalar, 'cmpeq_epi8') if op is not equal)
    union = operations(scalar, 'or')[0]
    for op, color in ((scalar_data, 0), (equal, 1), (inversion, 1), (zero, 0), (union, 0)):
        vectors[op.results[0]] = 'xmm%d' % color
    counts = [op.results[0] for b in fn.blocks for op in operations(b, 'ctz_nonzero')]
    for value in counts:
        prefer(value, 'rcx')
    fn.avx_cmp = True
    fn.avx_cmp_page = page.results[0]
    fn.avx_cmp_zero_groups = [([stages[0][3].results[0]], 1),
                              ([stages[1][3].results[0], stages[2][3].results[0]], 0),
                              ([stages[3][3].results[0]], 1), ([zero.results[0]], 2)]
    fn.avx_cmp_break_counts = set(counts)
    fn.avx_cmp_shared_returns = first_return, scalar_return
    fn.avx_cmp_dirty_tails = short, aligned_tail
    fn.avx_cmp_order = [fn.entry, stages[0][0], first_return, scalar_guard, tail,
                        both, short, scalar, scalar_continue,
                        *[s[0] for s in stages[1:]], aligned_tail,
                        *[s[0].term.targets[0] for s in stages[1:]], scalar_return]
    fn.shared_index_return = True
    fn.ymm_tail_callees = ('str_strcmp_tail',)
    fn.duplicate_short_tail_calls = True
    return True
