"""Rotate guarded loops into an entry test and a conditional latch.

The header may read memory. Its entry and latch copies execute exactly once
per original header execution; phis carry both its inputs and its read results
to the body and to the exit.
"""
from .ir import Term, dominators
from .facts import possible_bits
from . import types as T


def fold_read_loops(fn):
    """Speculate an affine, side-effect-free latch into a reading header."""
    changed = False
    pure = {'cast', 'ptradd', 'add', 'sub', 'mul', 'and', 'or', 'xor', 'neg', 'not', 'shl', 'shr'}
    for head in list(fn.blocks):
        if head not in fn.blocks or head.term.kind != 'br' or not any('read' in op.attrs.get('memory', '') for op in head.ops):
            continue
        pred = fn.predecessors()
        choices = [(n, b) for n, b in enumerate(head.term.targets) if b is not head
                   and b.term.kind == 'jmp' and b.term.targets == (head,) and pred[b] == [head]]
        if len(choices) != 1 or len(pred[head]) != 2:
            continue
        keep, body = choices[0]
        exit_ = head.term.targets[1 - keep]
        if exit_ in (head, body) or any(op.kind not in pure or op.attrs.get('volatile') or op.attrs.get('memory') for op in body.ops):
            continue
        phis = [op for op in head.ops if op.kind == 'phi']
        if not phis:
            continue
        outside = {v for b in fn.blocks if b not in (head, body)
                   for v in (*b.term.args, *(v for op in b.ops for v in op.args))}
        inverse = {}
        valid = True
        for phi in phis:
            value = phi.results[0]
            if value not in outside:
                continue
            back = phi.args[phi.attrs['blocks'].index(body)]
            update = fn.defs[back]
            if update.kind not in ('add', 'sub', 'ptradd') or update.args[0] != value:
                valid = False
                break
            count = fn.defs[update.args[1]]
            if count.kind != 'constant' or not isinstance(count.attrs['value'], int):
                valid = False
                break
            inverse[value] = (back, count.attrs['value'] * (-1 if update.kind == 'sub' else 1))
        if not valid:
            continue
        done = fn.block('latch.exit')
        replacements = {}
        for value, (next_, delta) in inverse.items():
            if value.type.kind == 'ptr':
                op = fn.make('ptradd', [value.type], [next_, fn.constant(T.LONG, -delta)])
            else:
                op = fn.make('sub', [value.type], [next_, fn.constant(value.type, delta)])
            done.ops.append(op)
            replacements[value] = op.results[0]
        done.term = Term('jmp', targets=(exit_,))
        for block in fn.blocks:
            if block in (head, body, done):
                continue
            for op in block.ops:
                op.args = tuple(replacements.get(v, v) for v in op.args)
                if block is exit_ and op.kind == 'phi':
                    op.attrs['blocks'] = [done if b is head else b for b in op.attrs['blocks']]
            block.term.args = tuple(replacements.get(v, v) for v in block.term.args)
        head.ops.extend(body.ops)
        for phi in phis:
            phi.attrs['blocks'] = [head if b is body else b for b in phi.attrs['blocks']]
        head.term.targets = (head, done) if keep == 0 else (done, head)
        fn.blocks.remove(body)
        fn.blocks.remove(done)
        fn.blocks.insert(fn.blocks.index(head) + 1, done)
        fn.verify()
        changed = True
    return changed


def rotate_loops(fn):
    changed = False
    for head in list(fn.blocks):
        if head not in fn.blocks or head.term.kind != 'br':
            continue
        pred = fn.predecessors()
        choices = [(n, b) for n, b in enumerate(head.term.targets) if b is not head
                   and b.term.kind == 'jmp' and b.term.targets == (head,) and pred[b] == [head]]
        if len(choices) != 1 or len(pred[head]) != 2:
            continue
        keep, body = choices[0]
        exit_ = head.term.targets[1 - keep]
        if exit_ in (body, head) or body not in pred[head] or any(op.kind == 'phi' for op in body.ops):
            continue
        phis = [op for op in head.ops if op.kind == 'phi']
        tests = [op for op in head.ops if op.kind != 'phi']
        if not phis or not tests or sum(op.kind not in ('cast', 'ptradd', 'symbol') for op in tests) > 32:
            continue
        if any(op.kind in ('call', 'asm') or op.attrs.get('volatile') or 'write' in op.attrs.get('memory', '') for op in tests):
            continue
        incoming = next(b for b in pred[head] if b is not body)
        condition = head.term.args[0]
        boolean = possible_bits(fn, condition) <= 1
        initial = {p.results[0]: p.args[p.attrs['blocks'].index(incoming)] for p in phis}
        header = [v for op in phis + tests for v in op.results]
        # Values at body entry are the results of either header instance.
        current, new_phis = {}, []
        for v in header:
            op = fn.make('phi', [v.type], attrs={'blocks': [head, body]})
            new_phis.append(op)
            current[v] = op.results[0]
        if condition in header and (keep == 1 or boolean):
            current[condition] = fn.constant(condition.type, 1 - keep)
        back = {p.results[0]: current.get(p.args[p.attrs['blocks'].index(body)], p.args[p.attrs['blocks'].index(body)]) for p in phis}
        for op in body.ops:
            op.args = tuple(current.get(v, v) for v in op.args)
        body.ops[:0] = new_phis
        # Read/update the next iteration's header after completing this body.
        latch = dict(back)
        for old in tests:
            clone = fn.make(old.kind, [v.type for v in old.results], [latch.get(v, v) for v in old.args], dict(old.attrs), old.source)
            body.ops.append(clone)
            latch.update(zip(old.results, clone.results))
        for old in tests:
            old.args = tuple(initial.get(v, v) for v in old.args)
            initial.update((v, v) for v in old.results)
        for old, new in zip(header, new_phis):
            new.args = initial[old], latch[old]
        weights, source = head.term.weights, head.term.source
        done = fn.block('rotate.exit')
        outside = {}
        for v in header:
            op = fn.make('phi', [v.type], [initial[v], latch[v]], {'blocks': [head, body]})
            done.ops.append(op)
            outside[v] = op.results[0]
        if condition in header and (keep == 0 or boolean):
            outside[condition] = fn.constant(condition.type, keep)
        done.term = Term('jmp', targets=(exit_,))
        targets = (body, done) if keep == 0 else (done, body)
        body.term = Term('br', (latch.get(condition, condition),), targets, source, weights)
        body.unroll_disabled = head.unroll_disabled
        if len(tests) == 1:
            test = tests[0]
            if test.kind in ('lt', 'le', 'gt', 'ge') and all(fn.defs[v].kind != 'constant' for v in test.args):
                test.kind = {'lt': 'gt', 'le': 'ge', 'gt': 'lt', 'ge': 'le'}[test.kind]
                test.args = test.args[::-1]
        head.ops = tests
        head.term = Term('br', (condition,), targets, source, weights)
        # Every edge leaving the original loop now passes through done.
        for block in fn.blocks:
            if block in (head, body, done):
                continue
            for op in block.ops:
                op.args = tuple(outside.get(v, v) for v in op.args)
                if block is exit_ and op.kind == 'phi':
                    op.attrs['blocks'] = [done if b is head else b for b in op.attrs['blocks']]
            block.term.args = tuple(outside.get(v, v) for v in block.term.args)
        fn.blocks.remove(done)
        fn.blocks.insert(fn.blocks.index(body) + 1, done)
        fn.verify()
        changed = True
    return changed


def rotate_guarded_regions(fn):
    """Move a pure loop guard to the latch across a small body with exits."""
    pure = {'cast', 'ptradd', 'add', 'sub', 'and', 'or', 'xor', 'neg', 'not', 'shl', 'shr', 'eq', 'ne', 'lt', 'le', 'gt', 'ge'}
    changed = False
    for head in list(fn.blocks):
        if head.term.kind != 'br' or head.unroll_disabled:
            continue
        pred, dom = fn.predecessors(), dominators(fn)
        tails = [b for b in pred[head] if head in dom[b]]
        incoming = [b for b in pred[head] if b not in tails]
        if len(tails) != 1 or len(incoming) != 1:
            continue
        tail, before = tails[0], incoming[0]
        if tail is head or tail.term.kind != 'jmp' or tail.term.targets != (head,):
            continue
        members, pending = {head}, [tail]
        while pending:
            block = pending.pop()
            if block not in members:
                members.add(block)
                pending.extend(pred[block])
        body = [b for b in fn.blocks if b in members and b is not head]
        if len(body) < 2 or len(body) > 4:
            continue
        inside = [n for n, target in enumerate(head.term.targets) if target in members]
        if len(inside) != 1:
            continue
        keep = inside[0]
        entry, exit_ = head.term.targets[keep], head.term.targets[1 - keep]
        phis = [op for op in head.ops if op.kind == 'phi']
        tests = [op for op in head.ops if op.kind != 'phi']
        if (not phis or not tests or len(tests) > 8 or pred[entry] != [head]
                or any(op.kind == 'phi' for op in entry.ops)
                or any(op.kind not in pure or op.attrs.get('memory') or op.attrs.get('volatile') for op in tests)):
            continue
        header = [v for op in phis + tests for v in op.results]
        if pred[exit_] != [head] and any(entry not in dom[b]
                and any(v in header for v in (*b.term.args, *(v for op in b.ops if op.kind != 'phi' for v in op.args)))
                for b in fn.blocks if b not in members):
            continue
        # Early exits carry header values on their existing phi edges. Other
        # uses must pass through the ordinary exit so its new phis dominate.
        if any(exit_ not in dom[b] and entry not in dom[b]
               and any(v in header for v in (*b.term.args, *(v for op in b.ops if op.kind != 'phi' for v in op.args)))
               or any(v in header and parent not in members and exit_ not in dom[parent] and entry not in dom[parent]
                      for op in b.ops if op.kind == 'phi' for parent, v in zip(op.attrs['blocks'], op.args))
               for b in fn.blocks if b not in members):
            continue
        condition, source, weights = head.term.args[0], head.term.source, head.term.weights
        initial = {op.results[0]: op.args[op.attrs['blocks'].index(before)] for op in phis}
        current, new_phis = {}, []
        for value in header:
            phi = fn.make('phi', [value.type], attrs={'blocks': [head, tail]})
            new_phis.append(phi)
            current[value] = phi.results[0]
        if keep == 1 or possible_bits(fn, condition) <= 1:
            current[condition] = fn.constant(condition.type, 1 - keep)
        back = {op.results[0]: current.get(op.args[op.attrs['blocks'].index(tail)], op.args[op.attrs['blocks'].index(tail)]) for op in phis}
        for block in body:
            for op in block.ops:
                op.args = tuple(current.get(v, v) for v in op.args)
            block.term.args = tuple(current.get(v, v) for v in block.term.args)
        entry.ops[:0] = new_phis
        if any(op.kind == 'intrinsic' and op.attrs['builtin'].op == 'masked_load' for op in entry.ops):
            fn.unaligned_loops = set(getattr(fn, 'unaligned_loops', ())) | {entry}
        latch = dict(back)
        for old in tests:
            clone = fn.make(old.kind, [v.type for v in old.results], [latch.get(v, v) for v in old.args], dict(old.attrs), old.source)
            tail.ops.append(clone)
            latch.update(zip(old.results, clone.results))
            if old.kind == 'lt' and not old.args[0].type.signed and old.args[0] in back and old.args[1] not in header:
                initial_count, update = fn.defs[initial[old.args[0]]], fn.defs[back[old.args[0]]]
                if (initial_count.kind == 'constant' and initial_count.attrs['value'] == 0
                        and update.kind == 'add' and update.args[0] == current[old.args[0]]
                        and fn.defs[update.args[1]].kind == 'constant'
                        and fn.defs[update.args[1]].attrs['value'] == 1):
                    clone.kind, clone.args = 'ne', clone.args[::-1]
                    if all(op.kind != 'intrinsic' and 'write' not in op.attrs.get('memory', '') for b in body for op in b.ops):
                        fn.scalar_read_loops = [*getattr(fn, 'scalar_read_loops', ()), (entry, tail, exit_, before)]
                        fn.induction_registers = {**getattr(fn, 'induction_registers', {}),
                                                  current[old.args[0]]: 'rcx', back[old.args[0]]: 'rcx'}
        for old in tests:
            old.args = tuple(initial.get(v, v) for v in old.args)
            initial.update((v, v) for v in old.results)
        for value, phi in zip(header, new_phis):
            phi.args = initial[value], latch[value]
        done, outside = fn.block('rotate.exit'), {}
        for value in header:
            phi = fn.make('phi', [value.type], [initial[value], latch[value]], {'blocks': [head, tail]})
            done.ops.append(phi)
            outside[value] = phi.results[0]
        done.term = Term('jmp', targets=(exit_,))
        if keep == 0 or possible_bits(fn, condition) <= 1:
            outside[condition] = fn.constant(condition.type, keep)
        for block in fn.blocks:
            if block in members or block is done:
                continue
            for op in block.ops:
                if op.kind == 'phi':
                    op.args = tuple((current if parent is not head and (parent in members or entry in dom[parent]) else outside).get(v, v)
                                    for parent, v in zip(op.attrs['blocks'], op.args))
                    op.attrs['blocks'] = [done if parent is head else parent for parent in op.attrs['blocks']]
                else:
                    values = current if entry in dom[block] else outside
                    op.args = tuple(values.get(v, v) for v in op.args)
            values = current if entry in dom[block] else outside
            block.term.args = tuple(values.get(v, v) for v in block.term.args)
        targets = (entry, done) if keep == 0 else (done, entry)
        head.ops, head.term = tests, Term('br', (condition,), targets, source, weights)
        tail.term = Term('br', (latch[condition],), targets, source, weights)
        fn.verify()
        changed = True
    return changed
