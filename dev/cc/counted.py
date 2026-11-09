"""Runtime unrolling of small memory loops with a descending byte count."""
from . import types as T
from .ir import Term


def unroll_counted_loops(fn):
    changed = False
    def constant(value):
        op = fn.defs[value]
        return op.attrs['value'] if op.kind == 'constant' else None

    def delta(phi, body):
        at = phi.attrs['blocks'].index(body)
        op = fn.defs[phi.args[at]]
        while op.kind == 'cast' and op.args[0].type.bits == op.results[0].type.bits:
            op = fn.defs[op.args[0]]
        if op.kind not in ('add', 'ptradd', 'sub') or op.args[0] != phi.results[0]:
            return None
        n = constant(op.args[1])
        return None if n is None else n * (-1 if op.kind == 'sub' else 1)

    allowed = {'cast', 'ptradd', 'load', 'store', 'intrinsic', 'add', 'sub', 'mul', 'and', 'or', 'xor', 'shl', 'shr', 'not', 'neg'}
    for head in list(fn.blocks):
        if head not in fn.blocks or head.unroll_disabled or head.term.kind != 'br':
            continue
        body, exit_ = head.term.targets
        pred = fn.predecessors()
        if (body is head or exit_ in (body, head) or body.term.kind != 'jmp' or body.term.targets != (head,)
                or pred[body] != [head] or len(pred[head]) != 2 or body not in pred[head]):
            continue
        incoming = next(b for b in pred[head] if b is not body)
        phis = [op for op in head.ops if op.kind == 'phi']
        tests = [op for op in head.ops if op.kind != 'phi']
        if len(tests) != 1 or not phis:
            continue
        test = tests[0]
        if test.kind != 'ge' or test.results != head.term.args or test.args[0].type.signed:
            continue
        step = constant(test.args[1])
        counter = next((p for p in phis if p.results[0] == test.args[0]), None)
        if counter is None or counter.results[0].type.kind != 'int' or step is None or step <= 0 or step & (step - 1):
            continue
        if any(p.results[0].type.kind not in ('int', 'ptr') for p in phis):
            continue
        deltas = {p: delta(p, body) for p in phis}
        if deltas[counter] != -step or any(n not in (-step, step) for n in deltas.values()):
            continue
        if any(op.kind not in allowed or op.attrs.get('volatile') for op in body.ops):
            continue
        # Only the header phis carry values out of this two-block loop.
        internal = {v for op in body.ops for v in op.results} | set(test.results)
        if any(v in internal for b in fn.blocks if b not in (head, body)
               for v in (*b.term.args, *(v for op in b.ops for v in op.args))):
            continue
        memory = sum(bool(op.attrs.get('memory')) for op in body.ops)
        if not memory:
            continue
        factor = 8
        while factor * memory > 16:
            factor //= 2
        if factor < 2 or factor * step >= 1 << counter.results[0].type.bits:
            continue
        initial = {p.results[0]: p.args[p.attrs['blocks'].index(incoming)] for p in phis}
        count = initial[counter.results[0]]
        ctype = count.type
        prelude, maincheck, main, done = (fn.block('unroll.' + n) for n in ('prelude', 'check', 'body', 'exit'))
        prebody = fn.block('unroll.remainder')
        preafter = fn.block('unroll.remainder.end') if factor > 2 else prebody
        main.unroll_disabled = True

        def make(block, kind, type_, args):
            op = fn.make(kind, [type_], args)
            block.ops.append(op)
            return op.results[0]

        def advance(block, phi, amount):
            start = initial[phi.results[0]]
            if start.type.kind == 'ptr':
                offset = make(block, 'cast', T.LONG, [amount]) if amount.type != T.LONG else amount
                if deltas[phi] < 0:
                    offset = make(block, 'neg', T.LONG, [offset])
                return make(block, 'ptradd', start.type, [start, offset])
            offset = make(block, 'cast', start.type, [amount]) if amount.type != start.type else amount
            return make(block, 'add' if deltas[phi] > 0 else 'sub', start.type, [start, offset])

        def clone_body(block, values):
            values = dict(values)
            for old in body.ops:
                op = fn.make(old.kind, [v.type for v in old.results], [values.get(v, v) for v in old.args], dict(old.attrs), old.source)
                block.ops.append(op)
                values.update(zip(old.results, op.results))
            return {p.results[0]: values[p.args[p.attrs['blocks'].index(body)]] for p in phis}

        # The original guard prevents underflow in (count - step).
        head.ops = []
        condition = make(head, 'ge', T.INT, [count, fn.constant(ctype, step)])
        head.term = Term('br', (condition,), (prelude, done))
        offset = make(prelude, 'sub', ctype, [count, fn.constant(ctype, step)])
        trips = make(prelude, 'shr', ctype, [offset, fn.constant(T.INT, step.bit_length() - 1)])
        trips = make(prelude, 'add', ctype, [trips, fn.constant(ctype, 1)])
        remainder = make(prelude, 'and', ctype, [trips, fn.constant(ctype, factor - 1)])
        prelude.term = Term('br', (remainder,), (prebody, maincheck))
        if factor == 2:
            adjusted = clone_body(prebody, initial)
            prebody.term = Term('jmp', targets=(maincheck,))
        else:
            bytes_ = make(prelude, 'shl', ctype, [remainder, fn.constant(T.INT, step.bit_length() - 1)])
            index = fn.make('phi', [ctype], attrs={'blocks': [prelude, prebody]})
            prebody.ops.append(index)
            values = {p.results[0]: advance(prebody, p, index.results[0]) for p in phis}
            clone_body(prebody, values)
            next_ = make(prebody, 'add', ctype, [index.results[0], fn.constant(ctype, step)])
            index.args = (fn.constant(ctype, 0), next_)
            again = make(prebody, 'ne', T.INT, [next_, bytes_])
            prebody.term = Term('br', (again,), (prebody, preafter))
            adjusted = {p.results[0]: advance(preafter, p, next_) for p in phis}
            preafter.term = Term('jmp', targets=(maincheck,))
        current = {}
        for p in phis:
            op = fn.make('phi', [p.results[0].type], [initial[p.results[0]], adjusted[p.results[0]]], {'blocks': [prelude, preafter]})
            maincheck.ops.append(op)
            current[p.results[0]] = op.results[0]
        enough = make(maincheck, 'ge', T.INT, [offset, fn.constant(ctype, (factor - 1) * step)])
        maincheck.term = Term('br', (enough,), (main, done))
        main_phis, values = {}, {}
        for p in phis:
            op = fn.make('phi', [p.results[0].type], attrs={'blocks': [maincheck, main]})
            main.ops.append(op)
            main_phis[p] = op
            values[p.results[0]] = op.results[0]
        for _ in range(factor):
            values = clone_body(main, values)
        for p, op in main_phis.items():
            op.args = current[p.results[0]], values[p.results[0]]
        again = make(main, 'gt', T.INT, [values[counter.results[0]], fn.constant(ctype, step - 1)])
        main.term = Term('br', (again,), (main, done))
        replacements = {}
        for p in phis:
            op = fn.make('phi', [p.results[0].type], [initial[p.results[0]], current[p.results[0]], values[p.results[0]]],
                         {'blocks': [head, maincheck, main]})
            done.ops.append(op)
            replacements[p.results[0]] = op.results[0]
        done.term = Term('jmp', targets=(exit_,))
        old_blocks = set(fn.blocks) - {head, body, prelude, prebody, preafter, maincheck, main, done}
        for b in old_blocks:
            for op in b.ops:
                op.args = tuple(replacements.get(v, v) for v in op.args)
                if op.kind == 'phi' and b is exit_:
                    op.attrs['blocks'] = [done if p is head else p for p in op.attrs['blocks']]
            b.term.args = tuple(replacements.get(v, v) for v in b.term.args)
        added = [prelude, prebody] + ([preafter] if preafter is not prebody else []) + [maincheck, main, done]
        fn.blocks = [b for b in fn.blocks if b is not body and b not in added]
        at = fn.blocks.index(head) + 1
        fn.blocks[at:at] = added
        if not hasattr(fn, 'counted_loops'):
            fn.counted_loops = []
        fn.counted_loops.append(dict(prelude=prelude, body=prebody, after=preafter,
                                    check=maincheck, main=main, exit=done, count=count,
                                    offset=offset, remainder=remainder, factor=factor, step=step))
        fn.verify()
        changed = True
    return changed


def normalize_counted_loops(fn):
    """Keep the small unroll remainder separate from full-width loop counts."""
    from .regalloc import pointer_parameters

    loops = getattr(fn, 'counted_loops', ())
    if not loops or fn.type.base.kind != 'ptr':
        return False

    def bare(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value

    calls = [(b, op) for b in fn.blocks for op in b.ops if op.kind == 'call']
    if any(b.term.kind != 'ret' or tuple(map(bare, b.term.args)) != op.results for b, op in calls):
        return False
    # These small vector loops share one destination and retain an opaque
    # copy of the original pointer for their return value.
    opaque = any(op.kind == 'asm' and not op.attrs['code'] and len(op.args) == len(op.results) == 1
                 and pointer_parameters(fn, op.args[0]) == {0} for b in fn.blocks for op in b.ops)
    memory = [op for b in fn.blocks for op in b.ops if op.kind != 'call' and op.attrs.get('memory')]
    if (not opaque or not memory or any(op.attrs.get('width') != 16 for op in memory)
            or any(op.attrs.get('volatile') for op in memory)):
        return False
    copying = any('read' in op.attrs['memory'] for op in memory)
    count_register, offset_register = (('rdi', 'r8') if copying else ('rsi', 'rdi')) if not calls else ('rcx', 'rsi')
    destination_register = 'rax' if calls else 'rcx'
    remainder_register, index_register = ('r9', 'r10') if copying else ('r8', 'r9')
    registers = fn.induction_registers = dict(getattr(fn, 'induction_registers', {}))
    owners = fn.layout_owners = dict(getattr(fn, 'layout_owners', {}))
    changed = False

    def constant(value):
        op = fn.defs[value]
        return op.attrs['value'] if op.kind == 'constant' else None

    for loop in loops:
        prelude, body, after, check, main = (loop[n] for n in ('prelude', 'body', 'after', 'check', 'main'))
        offset = fn.defs[loop['offset']]
        if (any(b not in fn.blocks for b in (prelude, body, after, check, main))
                or offset not in prelude.ops or prelude.term.kind != 'br'
                or prelude.term.args != (loop['remainder'],)):
            continue
        registers[loop['count']] = count_register
        registers[loop['offset']] = offset_register
        fn.counted_offsets = set(getattr(fn, 'counted_offsets', ())) | {loop['offset']}
        count = fn.defs[bare(loop['count'])]
        skew = fn.defs[bare(count.args[1])] if count.kind == 'sub' else None
        low = fn.defs[bare(skew.args[1])] if skew is not None and skew.kind == 'sub' else None
        if (skew is not None and low is not None and low.kind == 'and'
                and constant(skew.args[0]) == 16 and constant(low.args[1]) == 15
                and pointer_parameters(fn, low.args[0]) == {0}):
            # n - (16 - low_address_bits) - step. Retaining the low bits
            # avoids copying the live, descending byte count in the prelude.
            total = fn.make('add', [loop['count'].type], [low.results[0], count.args[0]])
            prelude.ops.insert(prelude.ops.index(offset), total)
            offset.kind, offset.args = 'add', (total.results[0], fn.constant(loop['count'].type, -16-loop['step']))
            registers[low.results[0]] = offset_register
            registers[total.results[0]] = offset_register
            registers[skew.results[0]] = remainder_register
            fn.scalar_constant_registers = dict(getattr(fn, 'scalar_constant_registers', {}))
            fn.scalar_constant_registers[skew.args[0]] = remainder_register
        elif offset.kind == 'sub' and constant(offset.args[1]) is not None:
            offset.kind, offset.args = 'add', (offset.args[0], fn.constant(offset.results[0].type, -constant(offset.args[1])))
        for block in (check, main, loop['exit']):
            if block not in fn.blocks:
                continue
            for phi in block.ops:
                if phi.kind != 'phi':
                    continue
                if phi.results[0].type.kind == 'int' and phi.results[0].type.bits == 64:
                    registers[phi.results[0]] = count_register
                    for incoming in phi.args:
                        registers.setdefault(bare(incoming), count_register)
                elif phi.results[0].type.kind == 'ptr':
                    origins = pointer_parameters(fn, phi.results[0])
                    if origins in ({0}, {1}):
                        register = destination_register if origins == {0} else 'rsi'
                        registers[phi.results[0]] = register
                        for incoming in phi.args:
                            registers[bare(incoming)] = register
        # A descending count uses add-negative after stores, keeping the
        # pointer and counter updates in the same order as the source loop.
        for op in main.ops:
            if op.kind == 'sub' and op.results[0].type.bits == 64 and constant(op.args[1]) is not None:
                op.kind, op.args = 'add', (op.args[0], fn.constant(op.results[0].type, -constant(op.args[1])))
        changed = True
        if loop['factor'] <= 2:
            # ((offset >> log2(step)) + 1) & 1 is the inverted step bit.
            bit = fn.make('and', [loop['count'].type], [loop['offset'], fn.constant(loop['count'].type, loop['step'])])
            prelude.ops.append(bit)
            prelude.term = Term('br', bit.results, (check, body))
            for phi in check.ops:
                if phi.kind != 'phi' or body not in phi.attrs['blocks'] or phi.results[0].type.kind != 'int':
                    continue
                at = phi.attrs['blocks'].index(body)
                if phi.args[at] == loop['offset']:
                    update = fn.make('add', [loop['count'].type], [loop['count'], fn.constant(loop['count'].type, -loop['step'])])
                    body.ops.append(update)
                    phi.args = tuple(update.results[0] if n == at else v for n, v in enumerate(phi.args))
                    registers[update.results[0]] = count_register
            owners[check], owners[main] = body, check
            for target in main.term.targets:
                if target is not main:
                    owners[target] = main
            continue
        remainder = fn.defs[loop['remainder']]
        increment = fn.defs[remainder.args[0]]
        shift = fn.defs[increment.args[0]]
        if (remainder.kind != 'and' or increment.kind != 'add' or shift.kind != 'shr'
                or shift.args[0] != loop['offset'] or loop['step'] * loop['factor'] >= 1 << 32
                or check not in prelude.term.targets or body not in prelude.term.targets
                or after.term.targets != (check,) or body not in body.term.targets):
            continue
        # Only low bits enter ((count-step)>>log2(step)+1)&(factor-1).
        def make(block, kind, type_, args):
            op = fn.make(kind, [type_], args)
            block.ops.append(op)
            return op.results[0]

        small = make(prelude, 'cast', T.UINT, [loop['offset']])
        trips = make(prelude, 'shr', T.UINT, [small, shift.args[1]])
        trips = make(prelude, 'add', T.UINT, [trips, fn.constant(T.UINT, 1)])
        rem = make(prelude, 'and', T.UINT, [trips, fn.constant(T.UINT, loop['factor'] - 1)])
        start = fn.block('unroll.start')
        bytes32 = make(start, 'shl', T.UINT, [rem, shift.args[1]])
        bytes_ = make(start, 'cast', loop['count'].type, [bytes32])
        start.term = Term('jmp', targets=(body,))
        prelude.term = Term('br', (rem,), (start, check))
        index = next(op for op in body.ops if op.kind == 'phi')
        index.attrs['blocks'] = [start if b is prelude else b for b in index.attrs['blocks']]
        comparison = fn.defs[body.term.args[0]]
        comparison.args = bytes_, comparison.args[0]
        registers.update({v: remainder_register for v in (small, trips, rem, bytes32, bytes_)})
        registers[fn.defs[trips].args[0]] = remainder_register
        registers[index.results[0]] = index_register
        for v in index.args:
            if fn.defs[v].kind != 'constant':
                registers[v] = index_register
        owners[after], owners[check], owners[main] = body, after, check
        if loop['exit'] in fn.blocks:
            owners[loop['exit']] = main
        for target in main.term.targets:
            if target is not main:
                owners[target] = main
        # Addresses can advance before the independent remaining-byte count.
        original, output, seen = set(after.ops), [], set()
        def visit(op):
            if op not in original or op in seen:
                return
            for value in op.args:
                visit(fn.defs[value])
            seen.add(op)
            output.append(op)
        pointers = [op for op in after.ops if op.kind == 'ptradd']
        pointers.sort(key=lambda op: pointer_parameters(fn, op.args[0]) != {0})
        for op in pointers:
            visit(op)
        for op in after.ops:
            visit(op)
        after.ops = output
        if copying and loop['step'] == 16:
            index_copy_loop(fn, main, check, loop['step']*loop['factor'], offset_register)
    if changed:
        fn.verify()
    return changed


def index_copy_loop(fn, main, before, step, register):
    """Share one byte index when both copied pointers stay inside the loop."""
    phis = [op for op in main.ops if op.kind == 'phi' and op.results[0].type.kind == 'ptr']
    if len(phis) != 2 or any(set(phi.attrs['blocks']) != {before, main} for phi in phis):
        return False
    carried = {v for phi in phis for v in (phi.results[0], phi.args[phi.attrs['blocks'].index(main)])}
    if any(v in carried for b in fn.blocks if b is not main
           for v in (*b.term.args, *(v for op in b.ops for v in op.args))):
        return False
    for phi in phis:
        update = fn.defs[phi.args[phi.attrs['blocks'].index(main)]]
        amount = fn.defs[update.args[1]] if update.kind == 'ptradd' else None
        if (amount is None or amount.kind != 'constant' or amount.attrs['value'] != step
                or update.args[0] != phi.results[0]):
            return False
    index = fn.make('phi', [T.ULONG], attrs={'blocks': [before, main]})
    next_ = fn.make('add', [T.ULONG], [index.results[0], fn.constant(T.ULONG, step)])
    index.args = fn.constant(T.ULONG, 0), next_.results[0]
    main.ops.insert(0, index)
    for phi in phis:
        base = phi.args[phi.attrs['blocks'].index(before)]
        phi.kind, phi.args, phi.attrs = 'ptradd', (base, index.results[0]), {}
    main.ops.insert(main.ops.index(fn.defs[main.term.args[0]]), next_)
    fn.induction_registers[index.results[0]] = fn.induction_registers[next_.results[0]] = register
    return True
