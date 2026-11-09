"""SSA inlining, constant propagation and control-flow cleanup."""
from . import types as T
from .ir import Term, dominators, promote, simplify_phis, copy_function
from .facts import possible_bits
from .predicates import combine_tests, combine_failures, mask_conditions, sink_mask_inversions, combine_pointer_guards
from .counted import unroll_counted_loops, normalize_counted_loops
from .copyloops import normalize_copy_loops
from .bytecompares import prepare_byte_compares, pack_byte_compare_masks
from .searches import prepare_aligned_searches, finish_aligned_searches
from .lengths import prepare_vector_lengths
from .largecompares import prepare_large_byte_compares, finish_large_byte_compares
from .interproc import infer_parameter_masks, simplify_boolean_selections
from .classifications import prepare_classification_scan, prepare_classification_tables
from .tableloops import prepare_vector_tables
from .threading import thread_status_edge
from .recipes import optimize_function
from .ranges import propagate_ranges
from .vectors import combine_movemasks
from .words import prepare_word_scans, split_word_scan_returns, normalize_word_indices, share_word_scan_steps
from .rotation import fold_read_loops, rotate_loops, rotate_guarded_regions
from .invariants import hoist_invariants, loop_preheaders
from .bounded import unroll_bounded_loops
from .induction import correlated_scalar_phis, normalize_guarded_indices, normalize_preincrement_scans, normalize_indexed_byte_scans, normalize_pipelined_copies, reduce_pointer_differences, reduce_store_addresses, reduce_alignment_stores
from .tails import return_expressions, thread_sentinel_exits, converging_unary, constant_returns, share_returned_address_counts, shared_store_returns, share_guarded_pointer_returns
from .addresses import cancel_offsets, factor_pointer_phis, factor_base_pointer_phis, correlated_pointer_phis, hoist_returned_addresses, tail_distances, return_loop_addresses, reuse_masked_addresses, guarded_pointer_distances


def replace_values(fn, replacements):
    def value(v):
        seen = set()
        while v in replacements:
            if v in seen:
                raise ValueError('cyclic SSA replacement')
            seen.add(v)
            v = replacements[v]
        return v
    for b in fn.blocks:
        for op in b.ops:
            op.args = tuple(map(value, op.args))
        b.term.args = tuple(map(value, b.term.args))


def callee_of(fn, value):
    op = fn.defs[value]
    while op.kind == 'cast':
        op = fn.defs[op.args[0]]
    return op.attrs['name'] if op.kind == 'symbol' else None


def inline_call(fn, block, index, callee):
    call = block.ops[index]
    continuation = fn.block('after.' + callee.name)
    continuation.ops, continuation.term = block.ops[index + 1:], block.term
    block.ops = block.ops[:index]
    # The old successor now receives its incoming edge from the continuation.
    for target in set(continuation.term.targets):
        for op in target.ops:
            if op.kind == 'phi':
                op.attrs['blocks'] = [continuation if p is block else p for p in op.attrs['blocks']]
    values = dict(zip(callee.params, call.args[1:]))
    blocks = {b: fn.block(callee.name + '.' + b.name) for b in callee.blocks}
    for b, clone in blocks.items():
        clone.unroll_disabled = b.unroll_disabled
        clone.fixed_trip_count = b.fixed_trip_count
    for op in callee.constants:
        value = fn.constant(op.results[0].type, op.attrs['value']) if op.kind == 'constant' else fn.undef(op.results[0].type)
        values[op.results[0]] = value
    for op in callee.allocas:
        new = fn.make('alloca', [v.type for v in op.results], attrs=dict(op.attrs), source=op.source)
        new.attrs['name'] = callee.name + '.' + new.attrs['name']
        fn.allocas.append(new)
        values.update(zip(op.results, new.results))
    pairs = []
    for b in callee.blocks:
        for op in b.ops:
            new = fn.make(op.kind, [v.type for v in op.results], attrs=dict(op.attrs), source=op.source)
            values.update(zip(op.results, new.results))
            blocks[b].ops.append(new)
            pairs.append((op, new))
    for old, new in pairs:
        new.args = tuple(values[v] for v in old.args)
        if new.kind == 'phi':
            new.attrs['blocks'] = [blocks[b] for b in old.attrs['blocks']]
    returns = []
    for old, b in blocks.items():
        t = old.term
        if t.kind == 'ret':
            b.term = Term('jmp', targets=(continuation,), source=t.source)
            if t.args:
                returns.append((b, values[t.args[0]]))
        else:
            b.term = Term(t.kind, tuple(values[v] for v in t.args), tuple(blocks[b] for b in t.targets), t.source, t.weights)
    block.term = Term('jmp', targets=(blocks[callee.entry],))
    inserted = [blocks[b] for b in callee.blocks] + [continuation]
    fn.blocks = [b for b in fn.blocks if b not in inserted]
    at = fn.blocks.index(block) + 1
    fn.blocks[at:at] = inserted
    if call.results:
        if not returns:
            # A nonreturning callee makes the continuation unreachable.
            result = fn.undef(call.results[0].type)
        elif len(returns) == 1:
            result = returns[0][1]
        else:
            phi = fn.make('phi', [call.results[0].type], [v for _, v in returns], {'blocks': [b for b, _ in returns]})
            continuation.ops.insert(0, phi)
            result = phi.results[0]
        replace_values(fn, {call.results[0]: result})


def fold(fn):
    replacements = {}
    uses, conditions = {}, set()
    for block in fn.blocks:
        if block.term.kind == 'br':
            conditions.add(block.term.args[0])
            value = block.term.args[0]
            origin = fn.defs[value]
            while origin.kind == 'cast':
                origin = fn.defs[origin.args[0]]
            if origin.kind == 'intrinsic' and origin.attrs['builtin'].op == 'expect':
                expected = fn.defs[origin.args[1]]
                if expected.kind == 'constant':
                    block.term.weights = (2000, 1) if expected.attrs['value'] else (1, 2000)
        for v in block.term.args:
            uses[v] = uses.get(v, 0) + 1
        for op in block.ops:
            for v in op.args:
                uses[v] = uses.get(v, 0) + 1
            if op.kind == 'select':
                conditions.add(op.args[0])
            if op.kind == 'call' and op.results:
                callee = getattr(fn, 'callees', {}).get(callee_of(fn, op.args[0]))
                if callee is not None and hasattr(callee, 'return_mask'):
                    op.attrs['result_mask'] = callee.return_mask

    def constant(v):
        while v in replacements:
            v = replacements[v]
        op = fn.defs[v]
        return op.attrs['value'] if op.kind == 'constant' else None

    def resolve(v):
        while v in replacements:
            v = replacements[v]
        return v

    def boolean(v, seen=frozenset()):
        v = resolve(v)
        if v in seen:
            return False
        op = fn.defs[v]
        if op.kind == 'constant':
            return op.attrs['value'] in (0, 1)
        if op.kind in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            return True
        if op.kind == 'and' and constant(op.args[1]) == 1:
            return True
        if op.kind in ('cast', 'phi', 'and', 'or', 'xor', 'select'):
            args = op.args[1:] if op.kind == 'select' else op.args
            return all(boolean(a, seen | {v}) for a in args)
        return False

    def address(v):
        op = fn.defs[resolve(v)]
        if op.kind == 'symbol':
            return op.attrs['name'], 0
        if op.kind == 'cast' and op.args[0].type.bits == 64:
            return address(op.args[0])
        if op.kind == 'ptradd':
            base, offset = address(op.args[0]), constant(op.args[1])
            if base is not None and offset is not None:
                return base[0], base[1] + T.convert_integer(offset, T.LONG)
        return None

    def before(block, current, kind, type_, args):
        generated = fn.make(kind, [type_], args)
        block.ops.insert(block.ops.index(current), generated)
        return generated.results[0]

    for b in fn.blocks:
        for op in b.ops:
            op.args = tuple(resolve(v) for v in op.args)
            if len(op.results) != 1:
                continue
            result, args = op.results[0], op.args
            if (op.kind in ('add', 'mul', 'and', 'or', 'xor', 'eq', 'ne') and constant(args[0]) is not None
                    and constant(args[1]) is None):
                op.args = args = args[::-1]
            if (op.kind in ('lt', 'le', 'gt', 'ge') and constant(args[0]) is not None
                    and constant(args[1]) is None):
                op.kind = {'lt': 'gt', 'le': 'ge', 'gt': 'lt', 'ge': 'le'}[op.kind]
                op.args = args = args[::-1]
            if op.kind in ('add', 'and', 'or', 'xor') and constant(args[1]) is not None:
                left = fn.defs[args[0]]
                if left.kind == op.kind and constant(left.args[1]) is not None:
                    a, c = constant(left.args[1]), constant(args[1])
                    combined = {'add': lambda: a + c, 'and': lambda: a & c, 'or': lambda: a | c, 'xor': lambda: a ^ c}[op.kind]()
                    op.args = args = left.args[0], fn.constant(result.type, combined)
            if op.kind in ('sub', 'ptradd') and constant(args[1]) is not None:
                left = fn.defs[args[0]]
                if left.kind == op.kind and constant(left.args[1]) is not None:
                    combined = constant(left.args[1]) + constant(args[1])
                    op.args = args = left.args[0], fn.constant(args[1].type, combined)
            if op.kind == 'add':
                left = fn.defs[args[0]]
                if (left.kind == 'sub' and constant(left.args[1]) is not None and constant(args[1]) is None
                        and uses.get(args[0]) == 1 and args[0].type == args[1].type == result.type):
                    total = before(b, op, 'add', result.type, [args[1], left.args[0]])
                    op.args = total, fn.constant(result.type, -constant(left.args[1]))
                    args = op.args
            if op.kind == 'or' and constant(args[1]) is None:
                right = fn.defs[args[1]]
                if (right.kind == 'or' and (constant(right.args[1]) is not None
                                           or boolean(args[0]) and all(boolean(v) for v in right.args))):
                    merged = before(b, op, 'or', result.type, [args[0], right.args[0]])
                    op.args = args = merged, right.args[1]
            if op.kind == 'and' and result in conditions and uses.get(result) == 1:
                inverse, mask = fn.defs[args[0]], constant(args[1])
                if inverse.kind == 'not' and mask is not None and possible_bits(fn, inverse.args[0]) & ~mask == 0:
                    op.kind, op.args = 'ne', (inverse.args[0], args[1])
                    args = op.args
            if op.kind in ('eq', 'ne'):
                left, wanted = fn.defs[args[0]], constant(args[1])
                if left.kind == 'and' and wanted is not None:
                    mask = constant(left.args[1])
                    shift = fn.defs[left.args[0]]
                    if mask is not None and 0 <= wanted <= mask and shift.kind == 'shr' and uses.get(left.results[0]) == 1:
                        count = constant(shift.args[1])
                        if count is not None and 0 <= count < result.type.bits and mask << count < 1 << shift.args[0].type.bits:
                            left.args = shift.args[0], fn.constant(left.results[0].type, mask << count)
                            op.args = args = args[0], fn.constant(args[1].type, wanted << count)
                            wanted, mask = wanted << count, mask << count
                    if mask is not None and wanted == mask and mask > 0 and mask & (mask - 1):
                        inverse = before(b, op, 'not', left.args[0].type, [left.args[0]])
                        missing = before(b, op, 'and', left.results[0].type, [inverse, left.args[1]])
                        op.args = args = missing, fn.constant(missing.type, 0)
            if op.kind in ('lt', 'le', 'gt', 'ge') and not args[0].type.signed:
                left, bound = fn.defs[args[0]], constant(args[1])
                capped = fn.defs[args[1]]
                if capped.kind == 'select':
                    choice = fn.defs[capped.args[0]]
                    minimum = (choice.kind in ('lt', 'le') and tuple(capped.args[1:]) == choice.args
                               or choice.kind in ('gt', 'ge') and tuple(capped.args[1:]) == choice.args[::-1])
                    if minimum:
                        for variable, cap in (choice.args, choice.args[::-1]):
                            limit = constant(cap)
                            maximum = possible_bits(fn, args[0])
                            if limit is not None and 0 <= maximum <= limit and (maximum < limit or op.kind in ('le', 'gt')):
                                op.args = args = args[0], variable
                                break
                biased = left
                while (biased.kind == 'cast' and biased.results[0].type.kind == biased.args[0].type.kind == 'int'
                       and biased.results[0].type.bits >= biased.args[0].type.bits):
                    biased = fn.defs[biased.args[0]]
                if biased.kind == 'add' and bound is not None:
                    base, delta = biased.args[0], constant(biased.args[1])
                    source = fn.defs[base]
                    while source.kind == 'cast':
                        source = fn.defs[source.args[0]]
                    if (source.kind == 'intrinsic' and source.attrs['builtin'].op in ('ctz_nonzero', 'tzcnt')
                            and delta is not None and 0 < delta <= bound
                            and possible_bits(fn, base) + delta < 1 << (biased.results[0].type.bits - biased.results[0].type.signed)):
                        if base.type != args[0].type:
                            base = before(b, op, 'cast', args[0].type, [base])
                        op.args = args = base, fn.constant(args[1].type, bound - delta)
                        op.attrs['biased_count'] = True
                        left, bound = fn.defs[base], bound - delta
                        if bound == 1 and op.kind in ('ge', 'lt'):
                            op.kind = 'ne' if op.kind == 'ge' else 'eq'
                            op.args = args = base, fn.constant(args[1].type, 0)
                if left.kind == 'sub' and bound is not None:
                    total, part = constant(left.args[0]), fn.defs[left.args[1]]
                    mask = constant(part.args[1]) if part.kind == 'and' else None
                    if total is not None and mask is not None and 0 <= mask <= total and 0 <= bound <= total:
                        op.kind = {'lt': 'gt', 'le': 'ge', 'gt': 'lt', 'ge': 'le'}[op.kind]
                        op.args = args = left.args[1], fn.constant(args[0].type, total - bound)
            values = [constant(v) for v in args]
            if op.kind == 'lt' and values[0] == 0 and args[1].type.kind == 'int' and not args[1].type.signed:
                op.kind, op.args = 'ne', (args[1], args[0])
                args, values = op.args, values[::-1]
            if op.kind == 'gt' and values[1] == 0 and args[0].type.kind == 'int' and not args[0].type.signed:
                op.kind = 'ne'
            r = None
            if op.kind in ('eq', 'ne') and values[1] in (0, 1) and boolean(args[0]):
                keep = op.kind == ('eq' if values[1] else 'ne')
                original = fn.defs[args[0]]
                uncast = original
                while uncast.kind == 'cast':
                    uncast = fn.defs[uncast.args[0]]
                if keep:
                    if args[0].type == result.type:
                        replacements[result] = args[0]
                    else:
                        op.kind, op.args = 'cast', (args[0],)
                    continue
                if original.kind in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                    op.kind = {'eq': 'ne', 'ne': 'eq', 'lt': 'ge', 'le': 'gt', 'gt': 'le', 'ge': 'lt'}[original.kind]
                    op.args = args = original.args
                    values = [constant(v) for v in args]
                elif result not in conditions and original.kind in ('and', 'or') and all(boolean(a) for a in original.args):
                    a = before(b, op, 'eq', result.type, [original.args[0], fn.constant(original.args[0].type, 0)])
                    c = before(b, op, 'eq', result.type, [original.args[1], fn.constant(original.args[1].type, 0)])
                    op.kind, op.args = ('or' if original.kind == 'and' else 'and'), (a, c)
                    continue
                elif result not in conditions and not (uncast.kind == 'and' and constant(uncast.args[1]) == 1):
                    op.kind, op.args = 'xor', (args[0], fn.constant(args[0].type, 1))
                    continue
            if op.kind == 'load' and not op.attrs.get('volatile'):
                at = address(args[0])
                if at is not None and at[0] in getattr(fn, 'immutable', {}):
                    data, relocs, start, size = fn.immutable[at[0]]
                    offset, width = at[1], op.attrs['width']
                    if 0 <= offset <= size - width and not any(offset + start < rel.offset + 8 and rel.offset < offset + start + width for rel in relocs):
                        raw = data[start + offset:start + offset + width]
                        r = raw if result.type.kind == 'vector' else int.from_bytes(raw, 'little')
            elif op.kind == 'select':
                if values[0] is not None:
                    replacements[result] = args[1 if values[0] else 2]
                elif args[1] == args[2]:
                    replacements[result] = args[1]
                else:
                    yes, no = fn.defs[args[1]], fn.defs[args[2]]
                    if yes.kind == 'select' and yes.args[2] == args[2] and boolean(args[0]) and boolean(yes.args[0]):
                        condition = before(b, op, 'and', args[0].type, [args[0], yes.args[0]])
                        op.args = condition, yes.args[1], args[2]
                    elif (yes.kind == no.kind == 'or' and yes.args[0] == no.args[0]):
                        selected = before(b, op, 'select', result.type, [args[0], yes.args[1], no.args[1]])
                        op.kind, op.args = 'or', (yes.args[0], selected)
                    elif boolean(args[0]) and values[2] == 0 and boolean(args[1]):
                        op.kind, op.args = 'and', (args[0], args[1])
                    elif boolean(args[0]) and values[1] == 1 and boolean(args[2]):
                        op.kind, op.args = 'or', (args[0], args[2])
            elif op.kind == 'cast' and args[0].type == result.type:
                replacements[result] = args[0]
            elif op.kind == 'cast' and values[0] is not None:
                r = values[0]
            elif op.kind == 'cast' and result.type.kind in ('int', 'ptr'):
                inner = fn.defs[args[0]]
                if inner.kind == 'cast' and inner.args[0].type.kind in ('int', 'ptr'):
                    original, middle = inner.args[0], args[0].type
                    if (result.type.bits <= middle.bits
                            or possible_bits(fn, original) < 1 << (middle.bits - 1)):
                        op.args = (original,)
                elif inner.kind == 'add' and result.type.kind == 'int' and result.type.bits == 64 and args[0].type.bits == 32:
                    base, delta = inner.args[0], constant(inner.args[1])
                    source = fn.defs[base]
                    while source.kind == 'cast':
                        source = fn.defs[source.args[0]]
                    if (source.kind == 'intrinsic' and source.attrs['builtin'].op in ('ctz_nonzero', 'tzcnt')
                            and delta is not None and 0 <= delta
                            and possible_bits(fn, base) + delta < 1 << (args[0].type.bits - args[0].type.signed)):
                        widened = before(b, op, 'cast', result.type, [base])
                        op.kind, op.args = 'add', (widened, fn.constant(result.type, delta))
            elif op.kind in ('neg', 'not') and values[0] is not None:
                r = -values[0] if op.kind == 'neg' else ~values[0]
            elif op.kind in ('add', 'ptradd', 'sub', 'mul', 'div', 'rem', 'and', 'or', 'xor', 'shl', 'shr', 'eq', 'ne', 'lt', 'gt', 'le', 'ge'):
                a, c = values
                if a is not None and c is not None:
                    div = lambda: (abs(a) // abs(c)) * (-1 if (a < 0) != (c < 0) else 1)
                    if op.kind in ('div', 'rem') and c == 0 or op.kind in ('shl', 'shr') and not 0 <= c < args[0].type.bits:
                        continue
                    r = {'add': lambda: a + c, 'ptradd': lambda: a + c, 'sub': lambda: a - c, 'mul': lambda: a * c,
                         'div': div, 'rem': lambda: a - div() * c, 'and': lambda: a & c, 'or': lambda: a | c, 'xor': lambda: a ^ c,
                         'shl': lambda: a << c, 'shr': lambda: a >> c, 'eq': lambda: int(a == c), 'ne': lambda: int(a != c),
                         'lt': lambda: int(a < c), 'gt': lambda: int(a > c), 'le': lambda: int(a <= c), 'ge': lambda: int(a >= c)}[op.kind]()
                elif c == 0 and op.kind in ('add', 'ptradd', 'sub', 'or', 'xor', 'shl', 'shr') and args[0].type == result.type:
                    replacements[result] = args[0]
                elif c == 1 and op.kind in ('mul', 'div') and args[0].type == result.type:
                    replacements[result] = args[0]
                elif c == 0 and op.kind in ('mul', 'and'):
                    r = 0
                elif args[0] == args[1]:
                    if op.kind in ('sub', 'xor', 'ne', 'lt', 'gt'):
                        r = 0
                    elif op.kind in ('eq', 'le', 'ge'):
                        r = 1
                    elif op.kind in ('and', 'or'):
                        replacements[result] = args[0]
            elif op.kind == 'intrinsic':
                builtin = op.attrs['builtin']
                if builtin.op in ('expect', 'unpredictable'):
                    replacements[result] = args[0]
                elif builtin.op == 'zero':
                    r = bytes(result.type.size)
                elif builtin.op == 'splat8' and values[0] is not None:
                    r = bytes([values[0] & 255]) * result.type.size
                elif builtin.op == 'constant8' and all(v is not None for v in values):
                    r = bytes(v & 255 for v in values)
                elif builtin.op == 'constant64_reversed' and all(v is not None for v in values):
                    r = b''.join((v & ((1 << 64) - 1)).to_bytes(8, 'little') for v in reversed(values))
                elif builtin.op == 'broadcast128' and isinstance(values[0], bytes):
                    r = values[0] * 2
                elif builtin.op in ('ctz_nonzero', 'tzcnt') and values[0] is not None:
                    v = values[0] & ((1 << args[0].type.bits) - 1)
                    if v or builtin.op == 'tzcnt':
                        r = (v & -v).bit_length() - 1 if v else args[0].type.bits
            if r is not None:
                replacements[result] = fn.constant(result.type, r)
    replace_values(fn, replacements)
    for b in fn.blocks:
        if b.term.kind == 'br':
            condition = fn.defs[b.term.args[0]]
            if condition.kind in ('eq', 'ne') and constant(condition.args[1]) == 0:
                targets = b.term.targets if condition.kind == 'ne' else b.term.targets[::-1]
                weights = b.term.weights
                if weights is not None and condition.kind == 'eq':
                    weights = weights[::-1]
                b.term = Term('br', (condition.args[0],), targets, b.term.source, weights)
            value = constant(b.term.args[0])
            if value is not None or b.term.targets[0] is b.term.targets[1]:
                b.term = Term('jmp', targets=(b.term.targets[0 if value else 1],), source=b.term.source)
    fn.blocks = fn.reachable()
    pred = fn.predecessors()
    for b in fn.blocks:
        for op in b.ops:
            if op.kind == 'phi':
                pairs = [(p, v) for p, v in zip(op.attrs['blocks'], op.args) if p in pred[b]]
                op.attrs['blocks'] = [p for p, _ in pairs]
                op.args = tuple(v for _, v in pairs)
    simplify_phis(fn)


def eliminate_dead(fn):
    live, todo = set(), [v for b in fn.blocks for v in b.term.args]
    for b in fn.blocks:
        for op in b.ops:
            effect = op.attrs.get('memory', '')
            if op.kind == 'call' or 'write' in effect or op.attrs.get('volatile'):
                live.add(op)
                todo.extend(op.args)
    while todo:
        op = fn.defs[todo.pop()]
        if op in live:
            continue
        live.add(op)
        todo.extend(op.args)
    for b in fn.blocks:
        b.ops = [op for op in b.ops if op in live]
    fn.allocas = [op for op in fn.allocas if op in live]
    fn.constants = [op for op in fn.constants if op in live]
    fn.constant_values = {k: v for k, v in fn.constant_values.items() if fn.defs[v] in live}


def common_values(fn):
    replacements = {}
    dom, pred = dominators(fn), fn.predecessors()
    children = {b: [] for b in fn.blocks}
    stable_memory = not any(op.kind == 'call' or 'write' in op.attrs.get('memory', '') for b in fn.blocks for op in b.ops)
    for b in fn.blocks:
        if b is not fn.entry:
            parent = max(dom[b] - {b}, key=lambda p: len(dom[p]))
            children[parent].append(b)

    def visit(b, inherited, inherited_loads):
        pure, loads = dict(inherited), dict(inherited_loads) if stable_memory or len(pred[b]) <= 1 else {}
        for op in b.ops:
            op.args = tuple(replacements.get(v, v) for v in op.args)
            if op.kind == 'phi' or len(op.results) != 1:
                if op.kind in ('call', 'asm') or 'write' in op.attrs.get('memory', ''):
                    loads.clear()
                continue
            memory = op.attrs.get('memory', '')
            if op.kind in ('call', 'asm') or 'write' in memory or op.attrs.get('volatile'):
                loads.clear()
                continue
            if op.kind in ('alloca', 'undef'):
                continue
            attrs = tuple((k, v) for k, v in op.attrs.items() if k not in ('declaration',))
            key = (op.kind, op.results[0].type, op.args, attrs)
            table = loads if memory else pure
            if key in table:
                replacements[op.results[0]] = table[key]
            else:
                table[key] = op.results[0]
        for child in children[b]:
            visit(child, pure, loads)
    visit(fn.entry, {}, {})
    replace_values(fn, replacements)


def sink_values(fn):
    """Delay pure work until the common dominator of its uses."""
    dom, pred = dominators(fn), fn.predecessors()
    post = {b: {b} if not b.term.targets else set(fn.blocks) for b in fn.blocks}
    changed = True
    while changed:
        changed = False
        for b in reversed(fn.blocks):
            if not b.term.targets:
                continue
            value = {b} | set.intersection(*(post[s] for s in b.term.targets))
            if value != post[b]:
                post[b], changed = value, True
    loops = {b: 0 for b in fn.blocks}
    for tail in fn.blocks:
        for head in tail.term.targets:
            if head not in dom[tail]:
                continue
            members, todo = {head}, [tail]
            while todo:
                b = todo.pop()
                if b not in members:
                    members.add(b)
                    todo.extend(pred[b])
            for b in members:
                loops[b] += 1
    pure_memory = not any(op.kind == 'call' or 'write' in op.attrs.get('memory', '') for b in fn.blocks for op in b.ops)
    place = {op: b for b in fn.blocks for op in b.ops}
    users = {}
    for b in fn.blocks:
        for op in b.ops:
            for n, v in enumerate(op.args):
                edge = op.attrs['blocks'][n] if op.kind == 'phi' else None
                users.setdefault(v, []).append((op, edge))
        for v in b.term.args:
            users.setdefault(v, []).append((None, b))
    movable = {'cast', 'symbol', 'ptradd', 'add', 'sub', 'mul', 'and', 'or', 'xor', 'neg', 'not',
               'shl', 'shr', 'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'select', 'intrinsic', 'load'}
    returned = {v for b in fn.blocks if b.term.kind == 'ret' for v in b.term.args}
    changed = True
    while changed:
        changed = False
        for op in reversed(list(place)):
            if op.kind not in movable or len(op.results) != 1 or op.attrs.get('volatile'):
                continue
            if op.kind == 'or' and op.results[0] in returned and fn.defs[op.args[1]].kind == 'constant':
                continue
            if op.attrs.get('memory') and not pure_memory:
                continue
            use = users.get(op.results[0], ())
            if not use:
                continue
            blocks = [edge if edge is not None else place[user] for user, edge in use]
            target = max(set.intersection(*(dom[b] for b in blocks)), key=lambda b: len(dom[b]))
            old = place[op]
            if target is old or loops[target] > loops[old]:
                continue
            if op.kind == 'load' and target in post[old]:
                continue
            if old not in dom[target]:
                continue
            old.ops.remove(op)
            at = min((target.ops.index(user) for user, edge in use if edge is None and place[user] is target), default=len(target.ops))
            target.ops.insert(at, op)
            place[op] = target
            changed = True
    if getattr(fn, 'shared_return', None) is not None:
        for b in fn.blocks:
            for op in reversed(list(b.ops)):
                if op.kind not in movable or op.attrs.get('memory') or op.attrs.get('volatile') or len(op.results) != 1:
                    continue
                local = [user for user, edge in users.get(op.results[0], ()) if edge is None and place[user] is b]
                if not local:
                    continue
                if op.kind in ('eq', 'ne', 'lt', 'le', 'gt', 'ge') and len(local) == 1 and local[0].kind == 'or':
                    other = next((v for v in local[0].args if v != op.results[0]), None)
                    if other is not None and fn.defs[other].kind in ('phi', 'xor'):
                        continue
                at, old = min(b.ops.index(user) for user in local), b.ops.index(op)
                if at > old + 1:
                    b.ops.pop(old)
                    b.ops.insert(at - 1, op)


def lower_constant_selects(fn):
    for b in fn.blocks:
        out = []
        for op in b.ops:
            if op.kind == 'select' and op.results[0].type.kind == 'int':
                condition, yes, no = op.args
                a, c = fn.defs[yes], fn.defs[no]
                if a.kind == c.kind == 'constant':
                    a, c = a.attrs['value'], c.attrs['value']
                    delta = a ^ c
                    if a >= 0 and c >= 0 and delta and delta & (delta - 1) == 0 and not a & delta and possible_bits(fn, condition) <= 1:
                        result = op.results[0]
                        inverse = fn.make('eq', [T.INT], [condition, fn.constant(condition.type, 0)])
                        widened = fn.make('cast', [result.type], inverse.results)
                        shift = fn.make('shl', [result.type], [widened.results[0], fn.constant(T.INT, delta.bit_length() - 1)])
                        out.extend([inverse, widened, shift])
                        op.kind, op.args = 'or', (shift.results[0], yes)
            out.append(op)
        b.ops = out


def schedule_predicates(fn):
    """Overlap independent integer computations before materializing predicates."""
    if getattr(fn, 'shared_return', None) is None:
        return
    comparisons = {'eq', 'ne', 'lt', 'le', 'gt', 'ge'}
    arithmetic = {'cast', 'ptradd', 'add', 'sub', 'mul', 'and', 'or', 'xor', 'neg', 'not', 'shl', 'shr'}
    for b in fn.blocks:
        tests = [op for op in b.ops if op.kind in comparisons]
        if len(tests) < 2 or any(possible_bits(fn, op.args[0]) <= 1 for op in tests):
            continue
        # Memory and other effects delimit regions whose operations can commute.
        start = 0
        for end in range(len(b.ops) + 1):
            if end < len(b.ops) and b.ops[end].kind in arithmetic | comparisons:
                continue
            region = b.ops[start:end]
            numeric = {op for op in region if op.kind in arithmetic and possible_bits(fn, op.results[0]) > 1}
            # Boolean-to-integer conversions belong after their predicates.
            changed = True
            while changed:
                changed = False
                for op in list(numeric):
                    if any(fn.defs[v] in region and fn.defs[v] not in numeric for v in op.args):
                        numeric.remove(op)
                        changed = True
            users = {op: {use for use in numeric if any(fn.defs[v] is op for v in use.args)} for op in numeric}
            heights = {}
            def height(op):
                if op not in heights:
                    heights[op] = 1 + max((height(use) for use in users[op]), default=0)
                return heights[op]
            pending, ordered = set(numeric), []
            while pending:
                ready = [op for op in pending if all(fn.defs[v] not in pending for v in op.args)]
                chosen = max(ready, key=lambda op: (height(op), sum(fn.defs[v] in numeric for v in op.args), -op.results[0].id))
                ordered.append(chosen)
                pending.remove(chosen)
            b.ops[start:end] = ordered + [op for op in region if op not in numeric]
            start = end + 1


def empty_blocks(fn):
    changed = True
    while changed:
        changed = False
        pred = fn.predecessors()
        for b in fn.blocks:
            if b is fn.entry or b.ops or b.term.kind != 'jmp':
                continue
            target = b.term.targets[0]
            if target is b:
                continue
            incoming = pred[b]
            # Retain an edge block when the other arm already enters the
            # target: its phi values can depend on which arm was taken.
            if any(target in p.term.targets for p in incoming):
                continue
            for p in incoming:
                p.term.targets = tuple(target if t is b else t for t in p.term.targets)
            for op in target.ops:
                if op.kind == 'phi':
                    pairs = []
                    for p, v in zip(op.attrs['blocks'], op.args):
                        pairs.extend((i, v) for i in incoming) if p is b else pairs.append((p, v))
                    op.attrs['blocks'] = [p for p, _ in pairs]
                    op.args = tuple(v for _, v in pairs)
            fn.blocks.remove(b)
            changed = True
            break


def merge_blocks(fn):
    changed = True
    while changed:
        changed = False
        pred = fn.predecessors()
        for b in fn.blocks:
            if b.term.kind != 'jmp':
                continue
            target = b.term.targets[0]
            if target is fn.entry or target is b or pred[target] != [b] or any(op.kind == 'phi' for op in target.ops):
                continue
            b.ops += target.ops
            b.term = target.term
            for other in fn.blocks:
                for op in other.ops:
                    if op.kind == 'phi':
                        op.attrs['blocks'] = [b if p is target else p for p in op.attrs['blocks']]
            fn.blocks.remove(target)
            changed = True
            break


def memory_addresses(fn):
    for b in fn.blocks:
        for op in b.ops:
            if op.kind not in ('load', 'store'):
                continue
            address = op.args[0]
            defining = fn.defs[address]
            while defining.kind == 'cast' and defining.args[0].type.kind == 'ptr':
                address = defining.args[0]
                defining = fn.defs[address]
            op.args = (address, *op.args[1:])


def scalarize(fn):
    """Split nonescaping stack aggregates into disjoint typed scalar slots."""
    roots = {op.results[0]: op for op in fn.allocas}
    aliases = {}

    def address(v):
        if v in aliases:
            return aliases[v]
        result = None
        op = fn.defs[v]
        if v in roots:
            result = v, 0
        elif op.kind == 'cast' and v.type.kind == 'ptr' and op.args[0].type.kind == 'ptr':
            result = address(op.args[0])
        elif op.kind == 'ptradd':
            base, offset = address(op.args[0]), fn.defs[op.args[1]]
            if base is not None and offset.kind == 'constant':
                result = base[0], base[1] + T.convert_integer(offset.attrs['value'], T.LONG)
        aliases[v] = result
        return result

    accesses, escaped = {v: [] for v in roots}, set()
    for b in fn.blocks:
        for op in b.ops:
            derived = op.kind in ('cast', 'ptradd') and op.results and address(op.results[0]) is not None
            for i, v in enumerate(op.args):
                at = address(v)
                if at is None:
                    continue
                root, offset = at
                if derived and i == 0:
                    continue
                if i == 0 and op.kind in ('load', 'store') and not op.attrs.get('volatile'):
                    type_ = op.results[0].type if op.kind == 'load' else op.args[1].type
                    width = op.attrs['width']
                    if type_.size == width and 0 <= offset <= roots[root].attrs['size'] - width:
                        accesses[root].append((op, offset, type_))
                        continue
                escaped.add(root)
        escaped.update(at[0] for v in b.term.args if (at := address(v)) is not None)
    for root, uses in accesses.items():
        if root in escaped or not uses:
            continue
        parts = sorted({(offset, type_) for _, offset, type_ in uses}, key=lambda p: p[0])
        if any(a + t.size > c for (a, t), (c, _) in zip(parts, parts[1:])):
            continue
        if len(parts) == 1 and parts[0] == (0, root.type.base.plain):
            continue
        slots = {}
        old = roots[root]
        for offset, type_ in parts:
            align = min(old.attrs['alignment'], type_.alignment, offset & -offset if offset else type_.alignment)
            op = fn.make('alloca', [T.pointer(type_)], attrs={
                'name': '%s.%d' % (old.attrs['name'], offset), 'size': type_.size, 'alignment': align}, source=old.source)
            fn.allocas.append(op)
            slots[offset, type_] = op.results[0]
        for op, offset, type_ in uses:
            op.args = (slots[offset, type_], *op.args[1:])


def forward_stack_loads(fn):
    roots = {op.results[0] for op in fn.allocas}

    def address(v):
        if v in roots:
            return v, 0
        op = fn.defs[v]
        if op.kind == 'cast' and op.args[0].type.bits == 64:
            return address(op.args[0])
        if op.kind == 'ptradd':
            base, offset = address(op.args[0]), fn.defs[op.args[1]]
            if base is not None and offset.kind == 'constant':
                return base[0], base[1] + T.convert_integer(offset.attrs['value'], T.LONG)
        return None

    replacements = {}
    for b in fn.blocks:
        known = {}
        for op in b.ops:
            if op.kind in ('load', 'store') and not op.attrs.get('volatile'):
                at, width = address(op.args[0]), op.attrs['width']
                if at is not None:
                    key = at[0], at[1], width
                    if op.kind == 'store':
                        known = {k: v for k, v in known.items() if k[0] != at[0] or k[1] + k[2] <= at[1] or at[1] + width <= k[1]}
                        known[key] = op.args[1]
                    elif key in known and known[key].type == op.results[0].type:
                        replacements[op.results[0]] = known[key]
                    continue
            if (op.kind == 'call' or 'write' in op.attrs.get('memory', '')
                    or op.kind in ('load', 'store') and op.attrs.get('volatile')):
                known.clear()
    replace_values(fn, replacements)


def return_tails(fn):
    """Expose returns on their incoming paths, including tail-call paths."""
    pred = fn.predecessors()
    for b in fn.blocks:
        if b is getattr(fn, 'shared_return', None):
            continue
        if b.term.kind != 'ret' or any(op.kind != 'phi' for op in b.ops):
            continue
        local = {v for op in b.ops for v in op.results}
        for p in pred[b]:
            if p.term.kind != 'jmp':
                continue
            values = {op.results[0]: op.args[op.attrs['blocks'].index(p)] for op in b.ops}
            args = tuple(values.get(v, v) for v in b.term.args)
            if not any(v in local for v in args):
                p.term = Term('ret', args, source=b.term.source)


def duplicate_return_tests(fn):
    """Place a small return after each incoming value of a shared test."""
    pred = fn.predecessors()
    uses = {}
    for b in fn.blocks:
        for op in b.ops:
            for v in op.args:
                uses.setdefault(v, set()).add(b)
        for v in b.term.args:
            uses.setdefault(v, set()).add(b)
    for b in list(fn.blocks):
        if len(b.ops) != 1 or b.ops[0].kind != 'phi' or b.term.kind != 'br':
            continue
        phi = b.ops[0]
        if b.term.args != phi.results or len(pred[b]) < 2 or not all(p.term.kind == 'jmp' for p in pred[b]):
            continue
        ret = next((t for t in b.term.targets if t.term.kind == 'ret' and len(t.ops) <= 4 and pred[t] == [b]), None)
        if ret is None or not uses.get(phi.results[0], set()) <= {b, ret}:
            continue
        if any(op.kind in ('phi', 'call', 'asm', 'load', 'store', 'copy', 'fill') or op.attrs.get('memory') for op in ret.ops):
            continue
        if any(any(op.kind == 'phi' for op in t.ops) for t in b.term.targets if t is not ret):
            continue
        for p, value in zip(phi.attrs['blocks'], phi.args):
            new = fn.block('return.' + p.name)
            replacements = {phi.results[0]: value}
            for op in ret.ops:
                clone = fn.make(op.kind, [v.type for v in op.results],
                                [replacements.get(v, v) for v in op.args], dict(op.attrs), op.source)
                new.ops.append(clone)
                replacements.update(zip(op.results, clone.results))
            new.term = Term('ret', tuple(replacements.get(v, v) for v in ret.term.args), source=ret.term.source)
            p.term = Term('br', (value,), tuple(new if t is ret else t for t in b.term.targets), b.term.source, b.term.weights)


def thread_constant_returns(fn):
    """Resolve a returned status flag together with its companion values."""
    pred, dom = fn.predecessors(), dominators(fn)
    for block in list(fn.blocks):
        if block.term.kind != 'br' or len(block.ops) < 2 or any(op.kind != 'phi' for op in block.ops):
            continue
        condition = fn.defs[block.term.args[0]]
        leaves = {target for target in block.term.targets if target.term.kind == 'ret' and pred[target] == [block]
                  and len(target.ops) <= 4 and not any(op.kind == 'phi' for op in target.ops)}
        loop_exit = len(leaves) == 1 and any(target not in leaves and target in dom[block] for target in block.term.targets)
        if condition not in block.ops or len(leaves) != 2 and not loop_exit:
            continue
        values = {v for op in block.ops for v in op.results}
        if not loop_exit and any(v in values for other in fn.blocks if other is not block and other not in block.term.targets
                for v in (*other.term.args, *(v for op in other.ops for v in op.args))):
            continue
        for parent, value in zip(condition.attrs['blocks'], condition.args):
            constant = fn.defs[value]
            if constant.kind != 'constant':
                continue
            target = block.term.targets[0 if constant.attrs['value'] else 1]
            if target not in leaves:
                continue
            replacements = {op.results[0]: op.args[op.attrs['blocks'].index(parent)] for op in block.ops}
            clone = fn.block('return.' + parent.name)
            for op in target.ops:
                copied = fn.make(op.kind, [v.type for v in op.results], [replacements.get(v, v) for v in op.args],
                                 dict(op.attrs), op.source)
                clone.ops.append(copied)
                replacements.update(zip(op.results, copied.results))
            clone.term = Term('ret', tuple(replacements.get(v, v) for v in target.term.args), source=target.term.source)
            parent.term.targets = tuple(clone if t is block else t for t in parent.term.targets)


def thread_boolean_tests(fn):
    counts = {}
    for block in fn.blocks:
        for op in block.ops:
            for v in op.args:
                counts[v] = counts.get(v, 0) + 1
        for v in block.term.args:
            counts[v] = counts.get(v, 0) + 1
    for b in list(fn.blocks):
        if len(b.ops) != 1 or b.ops[0].kind != 'phi' or b.term.kind != 'br':
            continue
        phi = b.ops[0]
        if b.term.args != phi.results or counts.get(phi.results[0]) != 1:
            continue
        for p, value in zip(phi.attrs['blocks'], phi.args):
            op = fn.defs[value]
            if op.kind == 'constant':
                target = b.term.targets[0 if op.attrs['value'] else 1]
                targets = (target,)
                term = Term(p.term.kind, p.term.args, tuple(target if t is b else t for t in p.term.targets), p.term.source, p.term.weights)
            elif p.term.kind == 'jmp':
                targets = b.term.targets
                term = Term('br', (value,), targets, b.term.source, b.term.weights)
            else:
                continue
            if b in targets or any(t in p.term.targets for t in targets):
                continue
            for target in set(targets):
                for other in target.ops:
                    if other.kind == 'phi':
                        at = other.attrs['blocks'].index(b)
                        other.attrs['blocks'].append(p)
                        other.args += (other.args[at],)
            p.term = term


def if_convert(fn):
    """Turn small, nontrapping scalar diamonds into value selections."""
    pred = fn.predecessors()
    pure = {'cast', 'add', 'sub', 'mul', 'and', 'or', 'xor', 'shl', 'shr', 'neg', 'not',
            'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'ptradd', 'symbol', 'select'}
    for root in list(fn.blocks):
        if root not in fn.blocks or root.term.kind != 'br':
            continue
        yes, no = root.term.targets
        if (yes is not no and yes.term.kind == no.term.kind == 'ret' and yes.term.args and no.term.args
                and pred[yes] == pred[no] == [root] and yes.term.args[0].type.scalar
                and all(len(a.ops) <= 4 and all(op.kind in pure for op in a.ops) for a in (yes, no))):
            root.ops.extend([*yes.ops, *no.ops])
            yes.ops, no.ops = [], []
            op = fn.make('select', [yes.term.args[0].type], [root.term.args[0], yes.term.args[0], no.term.args[0]])
            root.ops.append(op)
            root.term = Term('ret', op.results)
            fn.blocks = fn.reachable()
            pred = fn.predecessors()
            continue
        joins = []
        if yes.term.kind == 'jmp' and no.term.kind == 'jmp' and yes.term.targets == no.term.targets:
            joins.append(yes.term.targets[0])
        if yes.term.kind == 'jmp' and yes.term.targets == (no,):
            joins.append(no)
        if no.term.kind == 'jmp' and no.term.targets == (yes,):
            joins.append(yes)
        for join in joins:
            if join is root:
                continue
            arms = [b for b in (yes, no) if b is not join]
            phis = [op for op in join.ops if op.kind == 'phi']
            if not phis or any(not op.results[0].type.scalar for op in phis):
                continue
            if any(len(a.ops) > 8 or any(op.kind not in pure for op in a.ops) for a in arms):
                continue
            if (join is getattr(fn, 'shared_return', None) and possible_bits(fn, root.term.args[0]) > 1
                    and any(len(a.ops) > 1 for a in arms)):
                continue
            if (any(pred[a] != [root] for a in arms) and join is not getattr(fn, 'shared_return', None)
                    and any(possible_bits(fn, op.results[0]) > 1 for op in phis)):
                continue
            expected = set(arms) | ({root} if join in (yes, no) else set())
            if not expected <= set(pred[join]):
                continue
            cloned = {}
            removed = {a for a in arms if pred[a] == [root]} | ({root} if join in (yes, no) else set())
            for arm in arms:
                values = {}
                for old in arm.ops:
                    op = fn.make(old.kind, [v.type for v in old.results], [values.get(v, v) for v in old.args], dict(old.attrs), old.source)
                    root.ops.append(op)
                    values.update(zip(old.results, op.results))
                cloned[arm] = values
            for phi in phis:
                values = {p: v for p, v in zip(phi.attrs['blocks'], phi.args)}
                a, c = values[root if yes is join else yes], values[root if no is join else no]
                a, c = cloned.get(yes, {}).get(a, a), cloned.get(no, {}).get(c, c)
                op = fn.make('select', [phi.results[0].type], [root.term.args[0], a, c], source=phi.source)
                root.ops.append(op)
                keep = [(p, v) for p, v in values.items() if p not in removed]
                phi.attrs['blocks'] = [p for p, _ in keep] + [root]
                phi.args = tuple(v for _, v in keep) + op.results
            root.term = Term('jmp', targets=(join,))
            fn.blocks = fn.reachable()
            pred = fn.predecessors()
            break


def share_returns(fn):
    if not fn.type.base.scalar or any(op.kind == 'call' or 'write' in op.attrs.get('memory', '') for b in fn.blocks for op in b.ops):
        return
    blocks = [b for b in fn.blocks if b.term.kind == 'ret' and b.term.args]
    if len({b.term.args[0] for b in blocks}) < 3:
        return
    join = fn.block('return')
    phi = fn.make('phi', [fn.type.base], [b.term.args[0] for b in blocks], {'blocks': blocks})
    join.ops = [phi]
    join.term = Term('ret', phi.results)
    for b in blocks:
        b.term = Term('jmp', targets=(join,))
    fn.shared_return = join


def memory_libcalls(fn):
    for b in fn.blocks:
        for op in list(b.ops):
            if op.kind != 'copy':
                continue
            count = fn.defs[op.args[2]]
            if count.kind != 'constant' or count.attrs['value'] <= 128:
                continue
            signature = T.function(T.VOID, [v.type for v in op.args])
            symbol = fn.make('symbol', [T.pointer(signature)], attrs={'name': 'memcpy'})
            b.ops.insert(b.ops.index(op), symbol)
            op.kind, op.args = 'call', (symbol.results[0], *op.args)
            op.attrs = {'memory': 'readwrite'}


def inline_cost(fn):
    cost = 0
    for b in fn.blocks:
        cost += b.term.kind == 'br'
        for op in b.ops:
            if op.kind in ('cast', 'symbol', 'ptradd', 'phi'):
                continue
            cost += 8 if op.kind == 'call' else 2 if op.kind == 'asm' else 1
    return cost


def module_order(functions):
    """Public definitions enter the module immediately; static bodies are deferred."""
    by_name, ordered, queued = {fn.name: fn for fn in functions}, [], []
    seen = set()
    def declare(fn):
        if fn not in seen:
            seen.add(fn)
            ordered.append(fn)
            queued.append(fn)
    def references(fn):
        for b in fn.blocks:
            for op in b.ops:
                if op.kind == 'symbol' and op.attrs['name'] in by_name:
                    declare(by_name[op.attrs['name']])
    for fn in functions:
        if 'static' not in fn.decl.storage or fn.decl.attrs.get('constructor'):
            declare(fn)
            references(fn)
    for fn in queued:
        references(fn)
    for fn in functions:
        declare(fn)
    for n, fn in enumerate(ordered):
        fn.module_order = n


def cold_blocks(fn):
    dom, pred = dominators(fn), fn.predecessors()
    roots = set()
    for b in fn.blocks:
        if b.term.kind != 'br' or b.term.weights is None:
            continue
        weights = b.term.weights
        for n, target in enumerate(b.term.targets):
            if weights[n] * 100 < weights[1 - n] and pred[target] == [b]:
                roots.add(target)
    return {b for b in fn.blocks if roots & dom[b]}


def should_inline(callee, call, cold, caller=None):
    if callee.decl.attrs.get('always_inline'):
        return True
    cost = inline_cost(callee)
    if cold and cost > 18:
        ranges = call.attrs.get('arg_ranges', ())
        entry = {p: bounds for p, bounds in zip(callee.params, ranges) if bounds is not None}
        if entry:
            probe = copy_function(callee)
            probe.entry_ranges = entry
            simplify(probe)
            cost = inline_cost(probe)
        stack_arguments = 0
        if caller is not None:
            for value in call.args[1:]:
                op = caller.defs[value]
                while op.kind == 'cast':
                    op = caller.defs[op.args[0]]
                if op.kind == 'alloca':
                    stack_arguments += 1
        return cost <= 18 + 2*stack_arguments
    return 'static' in callee.decl.storage or 'inline' in callee.decl.storage or cost <= 42


def simplify(fn):
    memory_libcalls(fn)
    for _ in range(3):
        propagate_ranges(fn)
        fold(fn)
        factor_pointer_phis(fn)
        factor_base_pointer_phis(fn)
        correlated_pointer_phis(fn)
        cancel_offsets(fn)
        common_values(fn)
        combine_movemasks(fn)
        eliminate_dead(fn)
        memory_addresses(fn)
        forward_stack_loads(fn)
        eliminate_dead(fn)
        if fn.allocas:
            scalarize(fn)
            eliminate_dead(fn)
            promote(fn)
        empty_blocks(fn)
        merge_blocks(fn)
        return_tails(fn)
        duplicate_return_tests(fn)
        while thread_status_edge(fn):
            eliminate_dead(fn)
            fn.verify()
        thread_constant_returns(fn)
        thread_boolean_tests(fn)
        if_convert(fn)
        fold(fn)
    fn.verify()


def optimize(functions):
    by_name = {f.name: f for f in functions}
    for fn in functions:
        fn.callees = by_name
        simplify(fn)
    module_order(functions)
    for fn in functions:
        count = 0
        while True:
            selected = None
            cold = cold_blocks(fn)
            for b in fn.blocks:
                for i, op in enumerate(b.ops):
                    if op.kind != 'call':
                        continue
                    callee = by_name.get(callee_of(fn, op.args[0]))
                    if callee is None or callee is fn or callee.decl.attrs.get('noinline'):
                        continue
                    if should_inline(callee, op, b in cold, fn):
                        selected = b, i, callee
                        break
                if selected:
                    break
            if selected is None:
                break
            inline_call(fn, *selected)
            simplify(fn)
            count += 1
            if count > 10000:
                raise ValueError(fn.name + ': recursive inlining limit')
        simplify(fn)
        if combine_pointer_guards(fn):
            simplify(fn)
        combine_tests(fn)
        simplify(fn)
        share_returns(fn)
        simplify(fn)
        combine_failures(fn)
        simplify(fn)
        sink_values(fn)
        simplify(fn)
        lower_constant_selects(fn)
        simplify(fn)
        sink_values(fn)
        simplify(fn)
        schedule_predicates(fn)
        fn.verify()
        fn.cold = cold_blocks(fn)
        if fn.type.base.scalar:
            fn.return_mask = 0
            for b in fn.blocks:
                if b.term.kind == 'ret' and b.term.args:
                    fn.return_mask |= possible_bits(fn, b.term.args[0])
    # Callees remain compact until all callers have been inlined. Runtime
    # unrolling then uses the caller's size budget, rather than copying an
    # already expanded loop into every caller.
    for fn in functions:
        changed = thread_sentinel_exits(fn)
        if changed:
            simplify(fn)
        changed = unroll_bounded_loops(fn, allow_alignment=inline_cost(fn) <= 64) or changed
        changed = loop_preheaders(fn) or changed
        changed = hoist_invariants(fn) or changed
        if changed:
            simplify(fn)
        changed = inline_cost(fn) <= 64 and unroll_counted_loops(fn) or changed
        changed = fold_read_loops(fn) or changed
        changed = rotate_loops(fn) or changed
        changed = rotate_guarded_regions(fn) or changed
        changed = correlated_scalar_phis(fn) or changed
        changed = hoist_returned_addresses(fn) or changed
        changed = mask_conditions(fn) or changed
        changed = return_expressions(fn) or changed
        for block in list(fn.blocks):
            phis = [op for op in block.ops if op.kind == 'phi']
            if (fn.type.base.kind == 'int' and fn.type.base.bits == 64 and block.term.kind == 'br'
                    and len(phis) == 1 and phis[0].results[0].type.kind == 'int'):
                changed = return_expressions(fn, phis[0]) or changed
        if changed:
            simplify(fn)
            if converging_unary(fn):
                simplify(fn)
            sink_values(fn)
            simplify(fn)
            schedule_predicates(fn)
            fn.cold = cold_blocks(fn)
            fn.verify()
        if return_loop_addresses(fn):
            eliminate_dead(fn)
            fn.verify()
        if share_returned_address_counts(fn):
            eliminate_dead(fn)
            fn.verify()
        if normalize_guarded_indices(fn):
            eliminate_dead(fn)
            fn.verify()
        if prepare_word_scans(fn):
            split_word_scan_returns(fn)
            normalize_word_indices(fn)
            share_word_scan_steps(fn)
            eliminate_dead(fn)
            fn.verify()
        if normalize_preincrement_scans(fn):
            eliminate_dead(fn)
            fn.verify()
        if normalize_indexed_byte_scans(fn):
            eliminate_dead(fn)
            fn.verify()
        if normalize_pipelined_copies(fn):
            eliminate_dead(fn)
            fn.verify()
        if prepare_vector_lengths(fn):
            eliminate_dead(fn)
            fn.verify()
        if reduce_pointer_differences(fn):
            eliminate_dead(fn)
            fn.verify()
        if reduce_store_addresses(fn):
            eliminate_dead(fn)
            fn.verify()
        if reduce_alignment_stores(fn):
            eliminate_dead(fn)
            fn.verify()
        if sink_mask_inversions(fn):
            eliminate_dead(fn)
            fn.verify()
        if normalize_counted_loops(fn):
            eliminate_dead(fn)
            fn.verify()
        if normalize_copy_loops(fn):
            eliminate_dead(fn)
            fn.verify()
        if prepare_byte_compares(fn):
            eliminate_dead(fn)
            pack_byte_compare_masks(fn)
            eliminate_dead(fn)
            fn.verify()
        if prepare_large_byte_compares(fn):
            eliminate_dead(fn)
            finish_large_byte_compares(fn)
            eliminate_dead(fn)
            fn.verify()
        if prepare_aligned_searches(fn):
            eliminate_dead(fn)
            finish_aligned_searches(fn)
            eliminate_dead(fn)
            fn.verify()
        tail_distances(fn)
        eliminate_dead(fn)
        constant_returns(fn)
        shared_store_returns(fn)
        if share_guarded_pointer_returns(fn):
            eliminate_dead(fn)
            reuse_masked_addresses(fn)
            guarded_pointer_distances(fn)
            eliminate_dead(fn)
            fn.verify()
    live = emitted_functions(functions)
    infer_parameter_masks(live)
    for fn in live:
        if simplify_boolean_selections(fn):
            eliminate_dead(fn)
            fn.verify()
        if prepare_classification_scan(fn):
            eliminate_dead(fn)
            fn.verify()
    by_name = {fn.name: fn for fn in live}
    for fn in live:
        if prepare_classification_tables(fn, by_name):
            eliminate_dead(fn)
            fn.verify()
        if prepare_vector_tables(fn, by_name):
            eliminate_dead(fn)
            fn.verify()
    for n, fn in enumerate(live):
        optimize_function(fn)
        fn.machine_order = n
    return functions


def emitted_functions(functions):
    by_name = {f.name: f for f in functions}
    wanted = {f.name for f in functions if 'static' not in f.decl.storage or f.decl.attrs.get('constructor')
              or getattr(f, 'global_address', False)}
    todo = list(wanted)
    while todo:
        f = by_name[todo.pop()]
        for b in f.blocks:
            for op in b.ops:
                if op.kind == 'symbol':
                    name = op.attrs['name']
                    if name in by_name and name not in wanted:
                        wanted.add(name)
                        todo.append(name)
    return sorted((f for f in functions if f.name in wanted), key=lambda f: getattr(f, 'module_order', functions.index(f)))
