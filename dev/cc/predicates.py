"""Combine short-circuit tests of bits in the same scalar value."""
from .ir import Term


def bit_test(fn, value):
    op = fn.defs[value]
    while op.kind == 'cast' and op.args[0].type.bits == value.type.bits:
        value = op.args[0]
        op = fn.defs[value]
    if op.kind != 'and':
        return None
    mask = fn.defs[op.args[1]]
    if mask.kind != 'constant':
        return None
    mask = mask.attrs['value']
    if mask <= 0 or mask & (mask - 1):
        return None
    value = op.args[0]
    source = fn.defs[value]
    if source.kind == 'shr':
        shift = fn.defs[source.args[1]]
        if shift.kind != 'constant':
            return None
        count = shift.attrs['value']
        if not 0 <= count < source.args[0].type.bits or mask << count >= 1 << source.args[0].type.bits:
            return None
        value, mask = source.args[0], mask << count
    return value, mask


def missing_test(fn, value):
    op = fn.defs[value]
    if op.kind != 'and':
        return None
    source, mask = fn.defs[op.args[0]], fn.defs[op.args[1]]
    if source.kind == 'not' and mask.kind == 'constant' and mask.attrs['value'] > 0:
        return source.args[0], mask.attrs['value']
    return None


def combine_tests(fn, limit=5):
    pred = fn.predecessors()
    pure = {'cast', 'ptradd', 'symbol', 'add', 'sub', 'mul', 'and', 'or', 'xor', 'neg', 'not',
            'shl', 'shr', 'eq', 'ne', 'lt', 'le', 'gt', 'ge'}
    for head in list(fn.blocks):
        if head not in fn.blocks or head.term.kind != 'br':
            continue
        first = bit_test(fn, head.term.args[0])
        if first is None:
            continue
        word, mask = first
        current, chain, fail = head, [], head.term.targets[1]
        probe, count = head, 1
        while True:
            next_ = probe.term.targets[0]
            if next_.term.kind != 'br' or pred[next_] != [probe]:
                break
            test = bit_test(fn, next_.term.args[0])
            if test is None or test[0] != word or next_.term.targets[1] is not fail:
                break
            probe, count = next_, count + 1
        tail = probe.term.targets[0]
        suffix = missing_test(fn, tail.term.args[0]) if tail.term.kind == 'br' else None
        budget = min(limit, 3) if count > 3 and suffix is not None and suffix[0] != word and tail.term.targets[0] is fail else limit
        while len(chain) + 1 < budget:
            next_ = current.term.targets[0]
            if (next_ is head or pred[next_] != [current] or next_.term.kind != 'br'
                    or next_.term.targets[1] is not fail or any(op.kind not in pure for op in next_.ops)):
                break
            test = bit_test(fn, next_.term.args[0])
            if test is None or test[0] != word:
                break
            same = True
            for op in fail.ops:
                if op.kind == 'phi':
                    values = dict(zip(op.attrs['blocks'], op.args))
                    same &= values.get(head) == values.get(next_)
            if not same:
                break
            chain.append(next_)
            current, mask = next_, mask | test[1]
        if not chain or current.term.targets[0] in (head, *chain):
            continue
        success = current.term.targets[0]
        for op in success.ops:
            if op.kind == 'phi':
                op.attrs['blocks'] = [head if b is current else b for b in op.attrs['blocks']]
        for block in chain:
            head.ops.extend(block.ops)
            block.ops = []
        inverse = fn.make('not', [word.type], [word])
        missing = fn.make('and', [word.type], [inverse.results[0], fn.constant(word.type, mask)])
        head.ops.extend([inverse, missing])
        head.term = Term('br', missing.results, (fail, success))
        fn.blocks = fn.reachable()
        pred = fn.predecessors()


def combine_failures(fn):
    """Evaluate a short group of nontrapping tests before one failure edge."""
    from . import types as T
    pure = {'cast', 'ptradd', 'symbol', 'add', 'sub', 'mul', 'and', 'or', 'xor', 'neg', 'not',
            'shl', 'shr', 'eq', 'ne', 'lt', 'le', 'gt', 'ge'}
    pred = fn.predecessors()
    for head in list(fn.blocks):
        if head not in fn.blocks or head.term.kind != 'br':
            continue
        initial = missing_test(fn, head.term.args[0])
        if initial is None:
            continue
        fail, next_ = head.term.targets
        parts, words, chain = [(head.term.args[0], False)], {initial[0]}, []
        while len(parts) < 3 and next_.term.kind == 'br' and pred[next_] == [head if not chain else chain[-1]]:
            if any(op.kind not in pure for op in next_.ops):
                break
            missing = missing_test(fn, next_.term.args[0])
            bit = bit_test(fn, next_.term.args[0])
            scalar = missing is not None and next_.term.targets[0] is fail
            single = bit is not None and next_.term.targets[1] is fail
            word = missing[0] if scalar else bit[0] if single else None
            if word is None or word in words:
                break
            if any(op.kind == 'phi' and op.args[op.attrs['blocks'].index(head)] != op.args[op.attrs['blocks'].index(next_)] for op in fail.ops):
                break
            parts.append((next_.term.args[0], single))
            words.add(word)
            chain.append(next_)
            next_ = next_.term.targets[1 if scalar else 0]
        if len(parts) < 2:
            continue
        for b in chain:
            head.ops.extend(b.ops)
            b.ops = []
        scalar, booleans = [], []
        for value, single in parts:
            if single:
                test = fn.make('eq', [T.INT], [value, fn.constant(value.type, 0)])
                booleans.append(test)
            else:
                scalar.append(value)
        union = scalar[0]
        for value in scalar[1:]:
            type_ = T.arithmetic(value.type, union.type)
            operands = []
            for v in (value, union):
                if v.type != type_:
                    cast = fn.make('cast', [type_], [v])
                    head.ops.append(cast)
                    v = cast.results[0]
                operands.append(v)
            op = fn.make('or', [type_], operands)
            head.ops.append(op)
            union = op.results[0]
        if booleans:
            test = fn.make('ne', [T.INT], [union, fn.constant(union.type, 0)])
            head.ops.append(test)
            union = test.results[0]
            for test in booleans:
                head.ops.append(test)
                op = fn.make('or', [T.INT], [test.results[0], union])
                head.ops.append(op)
                union = op.results[0]
        for op in next_.ops:
            if op.kind == 'phi':
                op.attrs['blocks'] = [head if b is chain[-1] else b for b in op.attrs['blocks']]
        head.term = Term('br', (union,), (fail, next_))
        fn.blocks = fn.reachable()
        pred = fn.predecessors()


def mask_conditions(fn):
    """Test complemented masks and wrapping increments in the mask bank."""
    from collections import Counter
    from . import types as T
    uses = Counter(v for b in fn.blocks for op in b.ops for v in op.args)
    uses.update(v for b in fn.blocks for v in b.term.args)
    changed = False
    for block in fn.blocks:
        if block.term.kind != 'br':
            continue
        value = block.term.args[0]
        op = fn.defs[value]
        if value.type.kind != 'int' or value.type.bits not in (16, 32, 64):
            continue
        if op.kind == 'not':
            source = op.args[0]
        elif (op.kind == 'add' and uses[value] == 1 and fn.defs[op.args[1]].kind == 'constant'
              and fn.defs[op.args[1]].attrs['value'] == 1):
            source = op.args[0]
        else:
            continue
        mask = fn.defs[source]
        if mask.kind != 'intrinsic' or not mask.attrs['builtin'].name.endswith('_mask'):
            continue
        comparison = fn.make('ne', [T.INT], [source, fn.constant(source.type, (1 << source.type.bits) - 1)])
        block.ops.append(comparison)
        block.term.args = comparison.results
        changed = True
    return changed


def sink_mask_inversions(fn):
    """Test a complete equality mask in the loop and invert only on exit."""
    from . import types as T
    from .facts import possible_bits
    from .ir import dominators
    owners = {op: b for b in fn.blocks for op in b.ops}
    dom, changed = dominators(fn), False
    for head in fn.blocks:
        if head.term.kind != 'br' or head.term.targets[1] is not head or head.term.targets[0] is head:
            continue
        value = head.term.args[0]
        op = fn.defs[value]
        if op.kind != 'xor' or value.type.bits != 32 or op not in head.ops:
            continue
        source, constant = op.args[0], fn.defs[op.args[1]]
        raw = fn.defs[source]
        while raw.kind == 'cast':
            raw = fn.defs[raw.args[0]]
        if (constant.kind != 'constant' or constant.attrs['value'] != 65535 or possible_bits(fn, source) > 65535
                or raw.kind != 'intrinsic' or raw.attrs['builtin'].op != 'movemask8' or raw.args[0].type.bits != 128):
            continue
        exit_ = head.term.targets[0]
        users = [item for b in fn.blocks for item in b.ops if value in item.args]
        if (not users or any(item.kind != 'intrinsic' or item.attrs['builtin'].op != 'ctz_nonzero'
                            or exit_ not in dom[owners[item]] for item in users)
                or any(value in b.term.args for b in fn.blocks if b is not head)):
            continue
        test = fn.make('ne', [T.INT], [source, op.args[1]])
        head.ops.remove(op)
        head.ops.append(test)
        head.term.args = test.results
        op.kind, op.args, op.attrs = 'not', (source,), {}
        at = next((n for n, item in enumerate(exit_.ops) if item.kind != 'phi'), len(exit_.ops))
        exit_.ops.insert(at, op)
        if (any(item.kind == 'call' for b in fn.blocks for item in b.ops)
                and any(b.term.kind == 'ret' and b.term.args and fn.defs[b.term.args[0]].kind == 'constant'
                        and fn.defs[b.term.args[0]].attrs['value'] == 0 for b in fn.blocks)):
            fn.loop_alignments = {**getattr(fn, 'loop_alignments', {}), head: 0}
        changed = True
    return changed


def combine_pointer_guards(fn):
    """Evaluate two independent page-offset bounds before one branch."""
    from . import types as T
    pure = {'cast', 'and', 'sub', 'lt', 'le'}
    pred, changed = fn.predecessors(), False
    def comparison(value):
        op = fn.defs[value]
        if op.kind not in ('lt', 'le') or op.args[0].type.signed:
            return None
        offset, bound = (fn.defs[v] for v in op.args)
        if offset.kind != 'and' or bound.kind != 'constant':
            return None
        mask = fn.defs[offset.args[1]]
        if mask.kind != 'constant':
            return None
        mask = mask.attrs['value']
        if mask < 255 or mask & (mask + 1) or not mask // 2 < bound.attrs['value'] < mask:
            return None
        source = offset.args[0]
        while fn.defs[source].kind == 'cast':
            source = fn.defs[source].args[0]
        return (op, mask, bound.attrs['value']) if source.type.kind == 'ptr' else None
    for head in list(fn.blocks):
        if head not in fn.blocks or head.term.kind != 'br':
            continue
        first = comparison(head.term.args[0])
        if first is None:
            continue
        next_, fail = head.term.targets
        if (next_ is head or pred[next_] != [head] or next_.term.kind != 'br'
                or next_.term.targets[1] is not fail or any(op.kind not in pure for op in next_.ops)):
            continue
        second = comparison(next_.term.args[0])
        if second is None or first[1:] != second[1:]:
            continue
        if any(op.kind == 'phi' and op.args[op.attrs['blocks'].index(head)] != op.args[op.attrs['blocks'].index(next_)] for op in fail.ops):
            continue
        success = next_.term.targets[0]
        for item in success.ops:
            if item.kind == 'phi':
                item.attrs['blocks'] = [head if b is next_ else b for b in item.attrs['blocks']]
        head.ops.extend(next_.ops)
        next_.ops = []
        for op, _, bound in (first, second):
            if op.kind == 'le':
                op.kind, op.args = 'lt', (op.args[0], fn.constant(op.args[1].type, bound + 1))
        combined = fn.make('and', [T.INT], [head.term.args[0], next_.term.args[0]], {'page_guard': True})
        head.ops.append(combined)
        head.term = Term('br', combined.results, (success, fail))
        fn.blocks = fn.reachable()
        pred = fn.predecessors()
        changed = True
    return changed
