"""Late strength reduction of pointer differences in reading loops.

This runs after algebraic simplification: post-increment exit values must
keep using the updated induction variable rather than its previous value.
"""
from .ir import dominators, Term
from . import types as T


def correlated_scalar_phis(fn):
    """Keep one induction value when incoming edges give a fixed difference."""
    changed = False
    def affine(value):
        offset = 0
        while True:
            op = fn.defs[value]
            if op.kind == 'cast' and op.results[0].type.bits == op.args[0].type.bits:
                value = op.args[0]
            elif op.kind in ('add', 'sub') and fn.defs[op.args[1]].kind == 'constant':
                offset += fn.defs[op.args[1]].attrs['value'] * (-1 if op.kind == 'sub' else 1)
                value = op.args[0]
            elif op.kind == 'constant':
                return None, offset + op.attrs['value']
            else:
                return value, offset
    for block in fn.blocks:
        phis = [op for op in block.ops if op.kind == 'phi' and op.results[0].type.kind in ('int', 'ptr')]
        for n, current in enumerate(phis):
            if current.kind != 'phi':
                continue
            for other in phis[n+1:]:
                if (other.kind != 'phi' or other.results[0].type.bits != current.results[0].type.bits
                        or other.results[0].type.kind != current.results[0].type.kind
                        or set(other.attrs['blocks']) != set(current.attrs['blocks'])):
                    continue
                before = dict(zip(current.attrs['blocks'], current.args))
                differences = set()
                for pred, value in zip(other.attrs['blocks'], other.args):
                    a, b = affine(before[pred]), affine(value)
                    if a[0] != b[0]:
                        break
                    differences.add((b[1] - a[1]) % (1 << value.type.bits))
                else:
                    if len(differences) != 1:
                        continue
                    delta = differences.pop()
                    if delta and (other.results[0].type != current.results[0].type or current.results[0].type.kind != 'int'):
                        continue
                    target, base = current, other
                    target.kind = 'cast' if delta == 0 else 'sub'
                    target.args = (base.results[0],) if delta == 0 else (base.results[0], fn.constant(target.results[0].type, delta))
                    target.attrs = {}
                    block.ops.remove(target)
                    at = next((n for n, op in enumerate(block.ops) if op.kind != 'phi'), len(block.ops))
                    block.ops.insert(at, target)
                    changed = True
                    if target is current:
                        break
    return changed


def normalize_guarded_indices(fn):
    """Carry an index already biased by the loop's look-ahead guard."""
    changed = False
    for head in list(fn.blocks):
        pred, dom = fn.predecessors(), dominators(fn)
        if head.term.kind != 'br' or len(pred[head]) != 2:
            continue
        test = fn.defs[head.term.args[0]]
        if test.kind != 'lt' or test.args[0].type.kind != 'int' or test.args[0].type.signed or test.args[0].type.bits != 64:
            continue
        guard = fn.defs[test.args[0]]
        if guard.kind != 'add' or guard not in head.ops:
            continue
        phi, amount = (fn.defs[v] for v in guard.args)
        if phi.kind != 'phi' or phi not in head.ops or set(head.ops) != {phi, guard, test} or amount.kind != 'constant':
            continue
        step = amount.attrs['value']
        if not isinstance(step, int) or step <= 0 or step & (step-1):
            continue
        body = head.term.targets[0]
        if (body is head or body.term.kind != 'br' or head not in body.term.targets or pred[body] != [head]
                or body not in pred[head] or not any('read' in op.attrs.get('memory', '') for op in body.ops)
                or any(op.kind in ('call', 'asm') or op.attrs.get('volatile') or 'write' in op.attrs.get('memory', '') for op in body.ops)):
            continue
        before = next(b for b in pred[head] if b is not body)
        initial = fn.defs[phi.args[phi.attrs['blocks'].index(before)]]
        if initial.kind != 'constant' or phi.args[phi.attrs['blocks'].index(body)] != guard.results[0]:
            continue
        value = phi.results[0]
        outside = [(b, op) for b in fn.blocks if b not in (head, body) for op in b.ops if value in op.args]
        if (any(body not in dom[b] or op.kind != 'add' or op.results[0].type.bits != 64 or op.args.count(value) != 1 for b, op in outside)
                or any(value in b.term.args for b in fn.blocks if b not in (head, body))):
            continue
        advanced = fn.make('add', [value.type], [guard.results[0], guard.args[1]])
        body.ops.append(advanced)
        parents = list(phi.attrs['blocks'])
        initial_value = fn.constant(value.type, initial.attrs['value'] + step)
        phi.kind, phi.args, phi.attrs = 'sub', (guard.results[0], guard.args[1]), {}
        guard.kind = 'phi'
        guard.args = tuple(initial_value if b is before else advanced.results[0] for b in parents)
        guard.attrs = {'blocks': parents}
        head.ops = [guard, phi, test]
        for block, op in outside:
            other = op.args[1 if op.args[0] == value else 0]
            total = fn.make('add', [op.results[0].type], [other, advanced.results[0]])
            block.ops.insert(block.ops.index(op), total)
            op.args = total.results[0], fn.constant(op.results[0].type, -2*step)
            op.attrs = {}
        fn.induction_registers = {**getattr(fn, 'induction_registers', {}),
                                  guard.results[0]: 'rax', advanced.results[0]: 'rax'}
        changed = True
    if changed:
        fn.verify()
    return changed


def normalize_preincrement_scans(fn):
    """Carry the address being read, then advance it on the loop's back edge."""
    from collections import Counter
    uses = Counter(v for b in fn.blocks for op in b.ops for v in op.args)
    uses.update(v for b in fn.blocks for v in b.term.args)
    def root(value):
        while fn.defs[value].kind in ('cast', 'ptradd'):
            value = fn.defs[value].args[0]
        return fn.defs[value]
    changed = False
    for head in list(fn.blocks):
        pred = fn.predecessors()
        if (head not in head.term.targets or len(pred[head]) != 2
                or not any('read' in op.attrs.get('memory', '') for op in head.ops)
                or any(op.kind in ('call', 'asm')
                       or op.attrs.get('volatile') for op in head.ops)):
            continue
        writes = [op for op in head.ops if 'write' in op.attrs.get('memory', '')]
        if any(op.kind not in ('store', 'intrinsic') or op.kind == 'intrinsic' and op.attrs['builtin'].op != 'store'
               or root(op.args[0]).kind != 'alloca' for op in writes):
            continue
        before = next(b for b in pred[head] if b is not head)
        phis = [op for op in head.ops if op.kind == 'phi']
        for phi in phis:
            pointer = phi.results[0]
            if pointer.type.kind != 'ptr' or uses[pointer] != 1:
                continue
            back = phi.args[phi.attrs['blocks'].index(head)]
            step = fn.defs[back]
            if (step.kind != 'ptradd' or step not in head.ops or step.args[0] != pointer
                    or back.type != pointer.type):
                continue
            amount = fn.defs[step.args[1]]
            if (amount.kind != 'constant' or not isinstance(amount.attrs['value'], int)
                    or amount.attrs['value'] <= 0 or amount.attrs['value'] & (amount.attrs['value'] - 1)):
                continue
            initial = phi.args[phi.attrs['blocks'].index(before)]
            if writes and root(initial).kind != 'param':
                continue
            pre = fn.block('scan.preheader')
            address = fn.make('ptradd', [pointer.type], [initial, step.args[1]])
            pre.ops.append(address)
            pre.term = Term('jmp', targets=(head,))
            before.term.targets = tuple(pre if target is head else target for target in before.term.targets)
            for block in fn.blocks:
                for op in block.ops:
                    if op is not phi and op is not step:
                        op.args = tuple(pointer if v == back else v for v in op.args)
                block.term.args = tuple(pointer if v == back else v for v in block.term.args)
            phi.args = tuple(address.results[0] if parent is before else value
                             for parent, value in zip(phi.attrs['blocks'], phi.args))
            for other in phis:
                other.attrs['blocks'] = [pre if b is before else b for b in other.attrs['blocks']]
            head.ops.remove(step)
            head.ops.append(step)
            fn.preincrement_scans = {**getattr(fn, 'preincrement_scans', {}), head: step}
            if writes:
                index = root(initial).attrs['index']
                fn.induction_registers = dict(getattr(fn, 'induction_registers', {}))
                if index < 6:
                    preferred = ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9'][index]
                    fn.induction_registers = {**getattr(fn, 'induction_registers', {}),
                                              pointer: preferred, back: preferred, address.results[0]: preferred}
                for other in phis:
                    if other.results[0].type.kind == 'int' and other.results[0].type.bits == 8:
                        for value in (*other.results, *other.args):
                            fn.induction_registers[value] = 'rax'
            changed = True
            break
    if changed:
        fn.verify()
    return changed


def normalize_indexed_byte_scans(fn):
    """Carry the last byte of a group and its distance through table scans."""
    changed = False

    def uncast(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value

    def local(value):
        value = uncast(value)
        while fn.defs[value].kind == 'ptradd':
            value = uncast(fn.defs[value].args[0])
        return fn.defs[value].kind == 'alloca'

    for head in list(fn.blocks):
        pred, dom = fn.predecessors(), dominators(fn)
        if (head.term.kind != 'br' or head not in head.term.targets or len(pred[head]) != 2
                or any(op.kind in ('call', 'asm') or op.attrs.get('volatile')
                       or 'write' in op.attrs.get('memory', '') for op in head.ops)):
            continue
        before = next(b for b in pred[head] if b is not head)
        if before.term.kind != 'jmp':
            continue
        for phi in list(head.ops):
            if phi.kind != 'phi' or phi.results[0].type.kind != 'ptr':
                continue
            pointer, back = phi.results[0], phi.args[phi.attrs['blocks'].index(head)]
            update = fn.defs[back]
            if update.kind != 'ptradd' or update not in head.ops or update.args[0] != pointer:
                continue
            amount = fn.defs[update.args[1]]
            if amount.kind != 'constant' or amount.attrs['value'] not in (2, 4, 8):
                continue
            step = amount.attrs['value']
            addresses = [op for op in head.ops if op.kind == 'ptradd' and op.args[0] == pointer]
            if (len(addresses) != step or any(fn.defs[op.args[1]].kind != 'constant' for op in addresses)
                    or sorted(fn.defs[op.args[1]].attrs['value'] for op in addresses) != list(range(step, 2*step))
                    or any(op not in addresses and pointer in op.args for b in fn.blocks for op in b.ops)
                    or any(pointer in b.term.args or back in b.term.args for b in fn.blocks)):
                continue
            addresses.sort(key=lambda op: fn.defs[op.args[1]].attrs['value'])
            primary, lookups = [], []
            for address in addresses:
                readers = [op for op in head.ops if op.kind == 'load' and op.attrs['width'] == 1
                           and not op.results[0].type.signed and uncast(op.args[0]) == address.results[0]]
                if len(readers) != 1:
                    break
                byte = readers[0]
                secondary = []
                for op in head.ops:
                    if op.kind != 'load' or op.attrs['width'] != 1 or op.results[0].type.signed:
                        continue
                    location = fn.defs[uncast(op.args[0])]
                    if location.kind != 'ptradd' or not local(location.args[0]):
                        continue
                    index = fn.defs[location.args[1]]
                    if index.kind == 'cast' and index.args == byte.results:
                        secondary.append(op)
                if len(secondary) != 1:
                    break
                primary.append(byte)
                lookups.append(secondary[0])
            if len(primary) != step:
                continue
            roots = [(b, op) for b in fn.blocks if b is not head for op in b.ops
                     if op.kind == 'sub' and op.results[0].type.bits == 64 and uncast(op.args[0]) == back
                     and fn.defs[uncast(op.args[1])].kind == 'param' and op.args[1].type.bits == 64]
            if not roots or len({uncast(op.args[1]) for _, op in roots}) != 1:
                continue
            if any(back in op.args and (head not in dom[b] or op.kind == 'phi')
                   for b in fn.blocks if b is not head for op in b.ops):
                continue
            initial = phi.args[phi.attrs['blocks'].index(before)]
            base, type_ = uncast(roots[0][1].args[1]), roots[0][1].results[0].type
            cursor = fn.make('ptradd', [pointer.type], [initial, fn.constant(T.LONG, 2*step-1)])
            left, right = (fn.make('cast', [type_], [v]) for v in (initial, base))
            distance = fn.make('sub', [type_], [left.results[0], right.results[0]])
            before.ops.extend([cursor, left, right, distance])
            bias = int(fn.defs[head.term.args[0]].kind == 'or')
            initial_distance = distance.results[0]
            if bias:
                biased = fn.make('add', [type_], [initial_distance, fn.constant(type_, bias)])
                before.ops.append(biased)
                initial_distance = biased.results[0]
            next_pointer = fn.make('ptradd', [pointer.type], [pointer, update.args[1]])
            offset = fn.make('phi', [type_], attrs={'blocks': [before, head]})
            next_offset = fn.make('add', [type_], [offset.results[0], fn.constant(type_, step)])
            offset.args = initial_distance, next_offset.results[0]
            head.ops.insert(0, offset)
            phi.args = tuple(cursor.results[0] if b is before else next_pointer.results[0] for b in phi.attrs['blocks'])
            for address in addresses:
                delta = fn.defs[address.args[1]].attrs['value']-(2*step-1)
                address.args = pointer, fn.constant(T.LONG, delta)
            head.ops.extend([next_pointer, next_offset])
            for block in fn.blocks:
                if block is head:
                    continue
                out, adjusted = [], None
                for op in block.ops:
                    if (block, op) in roots:
                        op.kind = 'add' if bias else 'cast'
                        op.args = (next_offset.results[0], fn.constant(op.results[0].type, -bias)) if bias else next_offset.results
                        op.attrs = {}
                    elif back in op.args:
                        if adjusted is None:
                            biased = fn.make('ptradd', [pointer.type], [next_pointer.results[0], fn.constant(T.LONG, -(2*step-1))])
                            out.append(biased)
                            adjusted = biased.results[0]
                        op.args = tuple(adjusted if value == back else value for value in op.args)
                    out.append(op)
                block.ops = out
            fn.induction_registers = {**getattr(fn, 'induction_registers', {}), uncast(initial): 'rax',
                                      cursor.results[0]: 'rcx', pointer: 'rcx', next_pointer.results[0]: 'rcx',
                                      distance.results[0]: 'rax', initial_distance: 'rax', offset.results[0]: 'rax', next_offset.results[0]: 'rax'}
            fn.indexed_byte_scans = {**getattr(fn, 'indexed_byte_scans', {}), head: (tuple(primary), tuple(lookups))}
            if step == 4:
                for register, first, second in zip(('rdx', 'rdi', 'rsi', 'r8'), primary, lookups):
                    fn.induction_registers[first.results[0]] = register
                    fn.induction_registers[second.results[0]] = register
                condition = fn.defs[head.term.args[0]]
                middle = fn.defs[condition.args[0]] if condition.kind in ('and', 'or') else None
                first = fn.defs[middle.args[0]] if middle is not None and middle.kind == condition.kind else None
                if first is not None and first.kind == condition.kind:
                    old = middle.args[1]
                    middle.args = first.results[0], condition.args[1]
                    condition.args = middle.results[0], old
                    fn.induction_registers[first.results[0]] = 'rdi'
                    fn.induction_registers[middle.results[0]] = 'r8'
                    if condition.kind == 'or':
                        fn.induction_registers[condition.results[0]] = 'r8'
            returned = next(b for b in head.term.targets if b is not head)
            if returned.term.kind == 'ret' and len(returned.term.args) == 1:
                selected = fn.defs[returned.term.args[0]]
                local_ops = set(returned.ops)
                def constant_offset(value):
                    delta = 0
                    while fn.defs[value] in local_ops:
                        item = fn.defs[value]
                        if item.kind == 'cast' and value.type.bits == item.args[0].type.bits:
                            value = item.args[0]
                        elif item.kind == 'add' and fn.defs[item.args[1]].kind == 'constant':
                            delta += T.convert_integer(fn.defs[item.args[1]].attrs['value'], T.LONG)
                            value = item.args[0]
                        else:
                            break
                    return value, delta
                if bias:
                    for op in returned.ops:
                        if op.kind == 'add' and op.results[0].type.bits == 64 and fn.defs[op.args[1]].kind == 'constant':
                            value, delta = constant_offset(op.results[0])
                            op.kind = 'add' if delta else 'cast'
                            op.args = (value, fn.constant(op.results[0].type, delta)) if delta else (value,)
                if selected.kind == 'select' and all(op.kind in ('cast', 'ptradd', 'add', 'sub', 'select') for op in returned.ops):
                    yes, no = (fn.defs[v] for v in selected.args[1:])
                    difference = bias and yes.kind == no.kind == 'sub'
                    if difference:
                        negative = fn.make('neg', [no.args[1].type], [no.args[1]])
                        returned.ops.insert(returned.ops.index(no), negative)
                        no.kind, no.args = 'add', (no.args[0], negative.results[0])
                    if yes.kind == no.kind == 'add' or difference:
                        original, ordered, seen = set(returned.ops), [], set()
                        def visit(op):
                            if op not in original or op in seen:
                                return
                            for value in op.args:
                                visit(fn.defs[value])
                            ordered.append(op)
                            seen.add(op)
                        visit(yes if difference else no)
                        visit(no if difference else yes)
                        visit(selected)
                        returned.ops = ordered
                        fn.induction_registers[no.results[0]] = 'rax' if difference else 'rdx'
                        fn.induction_registers[yes.results[0]] = 'rcx' if difference else 'rax'
                        fn.induction_registers[selected.results[0]] = 'rax'
                        name = 'table_difference_selects' if difference else 'table_return_selects'
                        setattr(fn, name, {*getattr(fn, name, ()), selected})
            changed = True
            break
    if changed:
        fn.verify()
    return changed


def normalize_pipelined_copies(fn):
    """Carry source and destination cursors through a one-block look-ahead copy."""
    from collections import Counter
    changed = False

    def uncast(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value

    def affine(value, stop=frozenset()):
        value = uncast(value)
        if value in stop:
            return Counter({value: 1}), 0
        op = fn.defs[value]
        if op.kind == 'constant' and isinstance(op.attrs['value'], int):
            return Counter(), T.convert_integer(op.attrs['value'], T.LONG)
        if op.kind in ('ptradd', 'add', 'sub') and value.type.bits == 64:
            terms, offset = affine(op.args[0], stop)
            other, delta = affine(op.args[1], stop)
            sign = -1 if op.kind == 'sub' else 1
            for part, coefficient in other.items():
                terms[part] += coefficient * sign
                if not terms[part]:
                    del terms[part]
            return terms, offset + sign * delta
        return Counter({value: 1}), 0

    def make(block, kind, type_, args):
        op = fn.make(kind, [type_], args)
        block.ops.append(op)
        return op.results[0]

    for head in list(fn.blocks):
        pred, dom = fn.predecessors(), dominators(fn)
        if head.term.kind != 'br' or head not in head.term.targets or len(pred[head]) != 2:
            continue
        before = next(b for b in pred[head] if b is not head)
        if before.term.kind != 'jmp' or any(op.kind in ('call', 'asm') or op.attrs.get('volatile') for op in head.ops):
            continue
        reads = [op for op in head.ops if 'read' in op.attrs.get('memory', '')]
        writes = [op for op in head.ops if 'write' in op.attrs.get('memory', '')]
        if (len(reads) != 1 or len(writes) != 1 or reads[0].kind != 'intrinsic' or writes[0].kind != 'intrinsic'
                or reads[0].attrs['builtin'].op != 'load' or writes[0].attrs['builtin'].op != 'store'):
            continue
        read, store = reads[0], writes[0]
        vector = fn.defs[uncast(store.args[1])]
        address = fn.defs[uncast(read.args[0])]
        if (vector.kind != 'phi' or vector not in head.ops or vector.results[0].type.kind != 'vector'
                or vector.args[vector.attrs['blocks'].index(head)] != read.results[0]
                or address.kind != 'ptradd' or address not in head.ops):
            continue
        pointer = fn.defs[uncast(address.args[0])]
        amount = fn.defs[address.args[1]]
        if (pointer.kind != 'phi' or pointer not in head.ops or pointer.results[0].type.kind != 'ptr'
                or amount.kind != 'constant' or pointer.args[pointer.attrs['blocks'].index(head)] != address.results[0]):
            continue
        if (any(op is not address and pointer.results[0] in op.args for op in head.ops)
                or any(address.results[0] in b.term.args or any(address.results[0] in op.args
                           and (head not in dom[b] or op.kind == 'phi') for op in b.ops)
                       for b in fn.blocks if b is not head)):
            continue
        step = amount.attrs['value']
        if not isinstance(step, int) or step != read.attrs['width'] or step != store.attrs['width'] or step <= 0 or step & (step-1):
            continue
        target = fn.defs[uncast(store.args[0])]
        if target.kind != 'ptradd' or target not in head.ops:
            continue
        counter = fn.defs[uncast(target.args[1])]
        if counter.kind != 'phi' or counter not in head.ops or counter.results[0].type.bits != 64:
            continue
        update = fn.defs[counter.args[counter.attrs['blocks'].index(head)]]
        if update.kind != 'add' or uncast(update.args[0]) != counter.results[0] or affine(update.args[1]) != (Counter(), step):
            continue
        initial = pointer.args[pointer.attrs['blocks'].index(before)]
        distance = affine(counter.args[counter.attrs['blocks'].index(before)])
        initial_terms, initial_offset = affine(initial)
        sources = [(value, coefficient) for value, coefficient in distance[0].items() if coefficient == -1]
        if len(sources) != 1:
            continue
        source = sources[0][0]
        wanted = Counter(initial_terms)
        wanted[source] -= 1
        if not wanted[source]:
            del wanted[source]
        if distance != (wanted, initial_offset):
            continue
        base = uncast(target.args[0])
        owners = {v: b for b in fn.blocks for op in b.ops for v in op.results}
        if any(owners.get(v) is not None and owners[v] not in dom[before] for v in (base, source)):
            continue
        if any(pointer.results[0] in (*b.term.args, *(v for op in b.ops for v in op.args))
               for b in fn.blocks if b is not head):
            continue
        # Replace the integer offset with a destination cursor. The source
        # cursor names the block read now, and both updates precede the test.
        destination = make(before, 'ptradd', target.results[0].type,
                           [base, counter.args[counter.attrs['blocks'].index(before)]])
        initial_read = make(before, 'ptradd', initial.type, [initial, address.args[1]])
        next_read = fn.make('ptradd', [initial.type], [pointer.results[0], address.args[1]])
        next_destination = fn.make('ptradd', [destination.type], [target.results[0], address.args[1]])
        old_address = address.results[0]
        for op in head.ops:
            if op not in (pointer, address):
                op.args = tuple(pointer.results[0] if v == old_address else v for v in op.args)
        pointer.args = tuple(initial_read if parent is before else next_read.results[0]
                             for parent in pointer.attrs['blocks'])
        target.kind, target.args, target.attrs = 'phi', (destination, next_destination.results[0]), {'blocks': [before, head]}
        head.ops.remove(target)
        head.ops.insert(0, target)
        head.ops.extend([next_destination, next_read])
        # A copied tail has the same source delta as the loop's destination.
        # Keep exit addresses in terms of the advanced cursors so allocation
        # can reuse their registers on both the back edge and the exit edge.
        for block in fn.blocks:
            if block is head or head not in dom[block]:
                continue
            tails = {}
            expected = {base: 1, old_address: 1, source: -1}
            for op in block.ops:
                if op.kind != 'intrinsic' or op.attrs['builtin'].op != 'store':
                    continue
                terms, offset = affine(op.args[0], {old_address})
                if all(terms.get(v) == coefficient for v, coefficient in expected.items()):
                    extra = {v: n for v, n in terms.items() if v not in expected}
                    if len(extra) == 1 and next(iter(extra.values())) == 1:
                        tails[op] = next(iter(extra)), offset
            out = []
            adjusted = None
            for op in block.ops:
                if op in tails:
                    index, offset = tails[op]
                    total = fn.make('ptradd', [destination.type], [next_destination.results[0], index])
                    final = fn.make('ptradd', [op.args[0].type], [total.results[0], fn.constant(T.LONG, offset)])
                    out.extend([total, final])
                    op.args = (final.results[0], *op.args[1:])
                if old_address in op.args:
                    if adjusted is None:
                        biased = fn.make('ptradd', [initial.type], [next_read.results[0], fn.constant(T.LONG, -step)])
                        out.append(biased)
                        adjusted = biased.results[0]
                    op.args = tuple(adjusted if v == old_address else v for v in op.args)
                out.append(op)
            block.ops = out
        # Keep a short copy's final byte indexed by its shared distance.
        # Re-form this sum locally: the full byte count is already one larger.
        for block in fn.blocks:
            scalar_reads = [op for op in block.ops if op.kind == 'load']
            scalar_writes = [op for op in block.ops if op.kind == 'store']
            if ([op.attrs['width'] for op in scalar_reads] != [2, 1]
                    or [op.attrs['width'] for op in scalar_writes] != [2, 1]):
                continue
            terms, offset = affine(scalar_reads[1].args[0], {uncast(initial)})
            rest = {v: n for v, n in terms.items() if v != uncast(initial)}
            if offset or terms.get(uncast(initial)) != 1 or len(rest) != 1 or next(iter(rest.values())) != 1:
                continue
            index = next(iter(rest))
            index_op = fn.defs[index]
            if index.type.bits != 64 or (owners.get(index) is block
                    and (index_op.kind != 'cast' or owners.get(index_op.args[0]) is block)):
                continue
            distances = [op.results[0] for owner in dom[block] for op in owner.ops if op.kind == 'sub'
                         and affine(op.results[0], {uncast(initial)}) == (Counter({uncast(initial): 1, source: -1}), 0)]
            if len(distances) != 1:
                continue
            prefix = []
            if owners.get(index) is block:
                extended = fn.make('cast', [index.type], index_op.args)
                prefix.append(extended)
                index = extended.results[0]
            total = fn.make('add', [T.LONG], [distances[0], index])
            prefix.append(total)
            for memory, origin in ((scalar_reads[1], source), (scalar_writes[1], base)):
                location = fn.make('ptradd', [memory.args[0].type], [origin, total.results[0]])
                prefix.append(location)
                memory.args = (location.results[0], *memory.args[1:])
            block.ops[0:0] = prefix
        floating = {op.results[0] for block in fn.blocks if block not in (head, before) for op in block.ops
                    if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'load'
                    and any(user.kind == 'intrinsic' and user.attrs['builtin'].op == 'store'
                            and user.args[1] == op.results[0] for user in block.ops)}
        fn.floating_copies = set(getattr(fn, 'floating_copies', ())) | floating
        fn.induction_registers = {**getattr(fn, 'induction_registers', {}), initial: 'rcx', initial_read: 'rdx',
                                  destination: 'rcx', pointer.results[0]: 'rdx',
                                  next_read.results[0]: 'rdx', target.results[0]: 'rcx', next_destination.results[0]: 'rcx'}
        fn.pipelined_copies = {**getattr(fn, 'pipelined_copies', {}), head: (vector, read)}
        changed = True
    if changed:
        fn.verify()
    return changed


def reduce_pointer_differences(fn):
    dom, pred = dominators(fn), fn.predecessors()
    owners = {v: b for b in fn.blocks for op in b.ops for v in op.results}
    changed = False

    def uncast(value):
        op = fn.defs[value]
        while op.kind == 'cast' and value.type.bits == op.args[0].type.bits:
            value = op.args[0]
            op = fn.defs[value]
        return value

    def constant(value):
        op = fn.defs[value]
        return op.attrs['value'] if op.kind == 'constant' and isinstance(op.attrs['value'], int) else None

    for head in fn.blocks:
        if (head not in head.term.targets or len(pred[head]) != 2
                or not any('read' in op.attrs.get('memory', '') for op in head.ops)
                or any('write' in op.attrs.get('memory', '') or op.kind in ('call', 'asm') for op in head.ops)):
            continue
        before = next(b for b in pred[head] if b is not head)
        for phi in list(head.ops):
            if phi.kind != 'phi' or phi.results[0].type.kind != 'ptr':
                continue
            pointer = phi.results[0]
            next_pointer = phi.args[phi.attrs['blocks'].index(head)]
            step_op = fn.defs[uncast(next_pointer)]
            if step_op.kind != 'ptradd' or uncast(step_op.args[0]) != pointer:
                continue
            step = constant(step_op.args[1])
            if step is None or step <= 0 or step & (step - 1):
                continue
            matches = {}
            for block in fn.blocks:
                if block is head:
                    continue
                for op in block.ops:
                    if op.kind != 'sub' or op.results[0].type.kind != 'int' or op.results[0].type.bits != 64:
                        continue
                    if uncast(op.args[0]) != pointer:
                        continue
                    base = uncast(op.args[1])
                    owner = owners.get(base)
                    if owner is head or owner is not None and owner not in dom[before]:
                        continue
                    matches.setdefault((base, op.results[0].type), []).append(op)
            for (base, type_), roots in matches.items():
                initial_pointer = phi.args[phi.attrs['blocks'].index(before)]
                reads = [op for op in head.ops if 'read' in op.attrs.get('memory', '')]
                bias = step if len(reads) == 1 or head is getattr(fn, 'vector_length_loop', None) else step // 2
                initial_base, initial_bias = initial_pointer, bias
                advanced = fn.defs[uncast(initial_pointer)]
                if (head in getattr(fn, 'preincrement_scans', {}) and advanced.kind == 'ptradd'
                        and advanced in before.ops and constant(advanced.args[1]) == bias):
                    initial_base, initial_bias = advanced.args[0], 0

                def make(block, kind, result_type, args):
                    op = fn.make(kind, [result_type], args)
                    block.ops.append(op)
                    return op.results[0]

                if uncast(initial_base) == base:
                    initial = fn.constant(type_, -initial_bias)
                else:
                    a = make(before, 'cast', type_, [initial_base])
                    b = make(before, 'cast', type_, [base])
                    initial = make(before, 'sub', type_, [a, b])
                    if initial_bias:
                        initial = make(before, 'add', type_, [initial, fn.constant(type_, -initial_bias)])
                if initial_base != initial_pointer:
                    before.ops.remove(advanced)
                    before.ops.append(advanced)
                offset = fn.make('phi', [type_], attrs={'blocks': [before, head]})
                head.ops.insert(0, offset)
                next_ = make(head, 'add', type_, [offset.results[0], fn.constant(type_, step)])
                if head in getattr(fn, 'preincrement_scans', {}):
                    head.ops.remove(step_op)
                    head.ops.append(step_op)
                offset.args = initial, next_
                # Recognize constant offsets around each distance, including
                # casts that only change signedness. Keep only their final use.
                offsets = {op.results[0]: 0 for op in roots}
                pending = True
                while pending:
                    pending = False
                    for block in fn.blocks:
                        if block is head:
                            continue
                        for op in block.ops:
                            if len(op.results) != 1 or op.results[0] in offsets or op.results[0].type.bits != 64:
                                continue
                            value = None
                            if op.kind == 'cast' and op.args[0] in offsets and op.args[0].type.bits == 64:
                                value = offsets[op.args[0]]
                            elif op.kind in ('add', 'sub') and op.args[0] in offsets:
                                n = constant(op.args[1])
                                if n is not None:
                                    value = offsets[op.args[0]] + n * (-1 if op.kind == 'sub' else 1)
                            if value is not None:
                                offsets[op.results[0]] = value
                                pending = True
                for block in fn.blocks:
                    if block in (head, before):
                        continue
                    out = []
                    for op in block.ops:
                        if len(op.results) == 1 and op.results[0] in offsets:
                            delta = offsets[op.results[0]] - (step - bias)
                            op.kind, op.attrs = ('add' if delta else 'cast'), {}
                            op.args = (next_, fn.constant(op.results[0].type, delta)) if delta else (next_,)
                        elif op.kind == 'add' and op.results[0].type.bits == 64:
                            known = next((n for n, v in enumerate(op.args) if v in offsets), None)
                            if known is not None:
                                part, other = op.args[known], op.args[1 - known]
                                delta = offsets[part] - (step - bias)
                                value = next_
                                if value.type != op.results[0].type:
                                    cast = fn.make('cast', [op.results[0].type], [value])
                                    out.append(cast)
                                    value = cast.results[0]
                                total = fn.make('add', [op.results[0].type], [other, value])
                                out.append(total)
                                op.args = (total.results[0], fn.constant(op.results[0].type, delta)) if delta else total.results
                                op.kind = 'add' if delta else 'cast'
                                op.attrs = {}
                        out.append(op)
                    block.ops = out
                if not hasattr(fn, 'induction_registers'):
                    fn.induction_registers = {}
                comparisons = {op.attrs['builtin'].op for op in head.ops if op.kind == 'intrinsic'
                               and op.attrs['builtin'].op.startswith('cmpistr')}
                pointer_reg, offset_reg = ('rax', 'rdx' if 'cmpistri' in comparisons else 'rcx') if comparisons else (
                    ('rdi' if fn.params and uncast(initial_base) == fn.params[0] else 'rax', 'rcx') if len(reads) == 1 else ('rcx', 'rdx'))
                fn.induction_registers.update({pointer: pointer_reg, next_pointer: pointer_reg,
                                               offset.results[0]: offset_reg, next_: offset_reg})
                changed = True
    if changed:
        fn.verify()
    return changed


def reduce_store_addresses(fn):
    """Carry an affine destination pointer through a copying loop."""
    changed = False
    def uncast(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value
    def constant(value):
        op = fn.defs[value]
        return T.convert_integer(op.attrs['value'], T.LONG) if op.kind == 'constant' and isinstance(op.attrs['value'], int) else None
    for head in list(fn.blocks):
        pred, dom = fn.predecessors(), dominators(fn)
        if (head not in head.term.targets or len(pred[head]) != 2
                or not any('read' in op.attrs.get('memory', '') for op in head.ops)
                or not any('write' in op.attrs.get('memory', '') for op in head.ops)
                or any(op.kind in ('call', 'asm') or op.attrs.get('volatile') for op in head.ops)):
            continue
        owners = {v: b for b in fn.blocks for op in b.ops for v in op.results}
        before = next(b for b in pred[head] if b is not head)
        for root in list(head.ops):
            if root.kind != 'ptradd':
                continue
            distance = fn.defs[uncast(root.args[1])]
            if distance.kind != 'sub':
                continue
            pointer, base = (uncast(v) for v in distance.args)
            phi = fn.defs[pointer]
            if phi.kind != 'phi' or phi not in head.ops or pointer.type.kind != 'ptr':
                continue
            next_ = phi.args[phi.attrs['blocks'].index(head)]
            update = fn.defs[uncast(next_)]
            if update.kind != 'ptradd' or uncast(update.args[0]) != pointer:
                continue
            step = constant(update.args[1])
            if step is None or step <= 0 or step & (step - 1):
                continue
            if any(owners.get(v) is not None and owners[v] not in dom[before] for v in (base, root.args[0])):
                continue
            if any(root.results[0] in (*b.term.args, *(v for op in b.ops for v in op.args)) for b in fn.blocks if b is not head):
                continue
            stores = []
            for op in head.ops:
                if op.kind != 'intrinsic' or op.attrs['builtin'].op != 'store':
                    continue
                address, offset = uncast(op.args[0]), 0
                while address != root.results[0] and fn.defs[address].kind == 'ptradd':
                    item = fn.defs[address]
                    n = constant(item.args[1])
                    if n is None:
                        break
                    offset += n
                    address = uncast(item.args[0])
                if address == root.results[0]:
                    stores.append((offset, op.attrs['width']))
            if len(stores) < 2 or any(width != stores[0][1] for _, width in stores):
                continue
            bias = max(offset for offset, _ in stores)
            if sorted(offset for offset, _ in stores) != list(range(0, step, stores[0][1])):
                continue
            initial_pointer = phi.args[phi.attrs['blocks'].index(before)]
            entry = before
            if before.term.kind != 'jmp':
                entry = fn.block('loop.address')
                entry.term = Term('jmp', targets=(head,))
                before.term.targets = tuple(entry if b is head else b for b in before.term.targets)
                for op in head.ops:
                    if op.kind == 'phi':
                        op.attrs['blocks'] = [entry if b is before else b for b in op.attrs['blocks']]
                fn.blocks.remove(entry)
                fn.blocks.insert(fn.blocks.index(head), entry)
            def make(block, kind, type_, args):
                op = fn.make(kind, [type_], args)
                block.ops.append(op)
                return op.results[0]
            a = make(entry, 'cast', T.LONG, [initial_pointer])
            b = make(entry, 'cast', T.LONG, [base])
            index = make(entry, 'sub', T.LONG, [a, b])
            initial = make(entry, 'ptradd', root.results[0].type, [root.args[0], index])
            if bias:
                initial = make(entry, 'ptradd', root.results[0].type, [initial, fn.constant(T.LONG, bias)])
            target = fn.make('phi', [root.results[0].type], attrs={'blocks': [entry, head]})
            head.ops.insert(0, target)
            target_next = make(head, 'ptradd', root.results[0].type, [target.results[0], fn.constant(T.LONG, step)])
            target.args = initial, target_next
            root.args = target.results[0], fn.constant(T.LONG, -bias)
            if not hasattr(fn, 'induction_registers'):
                fn.induction_registers = {}
            fn.induction_registers.update({pointer: 'rcx', next_: 'rcx', target.results[0]: 'rdx', target_next: 'rdx'})
            changed = True
            break
    if changed:
        fn.verify()
    return changed


def reduce_alignment_stores(fn):
    """Use an invariant destination delta in an alignment-copy loop."""
    def uncast(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value
    def constant(value):
        op = fn.defs[value]
        return T.convert_integer(op.attrs['value'], T.LONG) if op.kind == 'constant' and isinstance(op.attrs['value'], int) else None
    changed = False
    for head in list(fn.blocks):
        pred, dom = fn.predecessors(), dominators(fn)
        if head.term.kind != 'br' or not any('read' in op.attrs.get('memory', '') for op in head.ops):
            continue
        tails = [b for b in pred[head] if head in dom[b]]
        incoming = [b for b in pred[head] if b not in tails]
        if len(tails) != 1 or len(incoming) != 1:
            continue
        tail, before = tails[0], incoming[0]
        if tail is head or tail not in head.term.targets or pred[tail] != [head] or tail.term.kind != 'br':
            continue
        test = fn.defs[tail.term.args[0]]
        if test.kind != 'and' or constant(test.args[1]) is None:
            continue
        span = constant(test.args[1]) + 1
        if span <= 1 or span & (span - 1):
            continue
        stores = [op for op in tail.ops if 'write' in op.attrs.get('memory', '')]
        if len(stores) != 1 or any(op.kind in ('call', 'asm') or op.attrs.get('volatile') for op in head.ops + tail.ops):
            continue
        store = stores[0]
        if store.kind != 'intrinsic' or store.attrs['builtin'].op != 'store':
            continue
        root = fn.defs[uncast(store.args[0])]
        if root.kind != 'ptradd':
            continue
        distance = fn.defs[uncast(root.args[1])]
        if distance.kind != 'sub':
            continue
        pointer, source = (uncast(v) for v in distance.args)
        phi = fn.defs[pointer]
        if phi.kind != 'phi' or phi not in head.ops or pointer.type.kind != 'ptr':
            continue
        next_ = phi.args[phi.attrs['blocks'].index(tail)]
        update = fn.defs[uncast(next_)]
        if update.kind != 'ptradd' or uncast(update.args[0]) != pointer or uncast(test.args[0]) != next_:
            continue
        step = constant(update.args[1])
        if step is None or step != store.attrs['width'] or step >= span or span % step:
            continue
        owners = {v: b for b in fn.blocks for op in b.ops for v in op.results}
        if any(owners.get(v) is not None and owners[v] not in dom[before] for v in (source, root.args[0])):
            continue
        initial_pointer = phi.args[phi.attrs['blocks'].index(before)]
        entry = fn.block('loop.delta')
        entry.term = Term('jmp', targets=(head,))
        before.term.targets = tuple(entry if b is head else b for b in before.term.targets)
        for op in head.ops:
            if op.kind == 'phi':
                op.attrs['blocks'] = [entry if b is before else b for b in op.attrs['blocks']]
        fn.blocks.remove(entry)
        fn.blocks.insert(fn.blocks.index(head), entry)
        def make(block, kind, type_, args):
            op = fn.make(kind, [type_], args)
            block.ops.append(op)
            return op.results[0]
        dst = make(entry, 'cast', T.LONG, [root.args[0]])
        src = make(entry, 'cast', T.LONG, [source])
        delta = make(entry, 'sub', T.LONG, [dst, src])
        base = make(entry, 'cast', root.args[0].type, [delta])
        initial = make(entry, 'ptradd', pointer.type, [initial_pointer, fn.constant(T.LONG, step)])
        cursor = fn.make('phi', [pointer.type], attrs={'blocks': [entry, tail]})
        head.ops.insert(0, cursor)
        cursor_next = make(tail, 'ptradd', pointer.type, [cursor.results[0], fn.constant(T.LONG, step)])
        cursor.args = initial, cursor_next
        value = fn.make('cast', [T.LONG], cursor.results)
        offset = fn.make('add', [T.LONG], [value.results[0], fn.constant(T.LONG, -step)])
        at = tail.ops.index(root)
        tail.ops[at:at] = [value, offset]
        root.args = base, offset.results[0]
        low = fn.make('cast', [T.UINT], cursor.results)
        widened = fn.make('cast', [test.args[0].type], low.results)
        at = tail.ops.index(test)
        tail.ops[at:at] = [low, widened]
        test.args = widened.results[0], test.args[1]
        if not hasattr(fn, 'induction_registers'):
            fn.induction_registers = {}
        fn.induction_registers.update({pointer: 'rcx', next_: 'rcx', initial_pointer: 'rcx', delta: 'rdx',
                                       cursor.results[0]: 'rdi', cursor_next: 'rdi', initial: 'rdi'})
        changed = True
    if changed:
        fn.verify()
    return changed
