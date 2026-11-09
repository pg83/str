"""Share returned addresses and simplify pointer-distance arithmetic."""
from collections import Counter

from . import types as T
from .ir import dominators, Term


def uncast(fn, value):
    op = fn.defs[value]
    while op.kind == 'cast' and value.type.bits == op.args[0].type.bits:
        value = op.args[0]
        op = fn.defs[value]
    return value


def factor_pointer_phis(fn):
    """Move a common constant pointer offset below an acyclic phi."""
    dom = dominators(fn)
    loop_phis = {v for b in fn.blocks for op in b.ops if op.kind == 'phi'
                 and any(b in dom[p] for p in op.attrs['blocks']) for v in op.results}
    for block in fn.blocks:
        for op in list(block.ops):
            if (op.kind != 'phi' or op.results[0].type.kind != 'ptr'
                    or any(block in dom[p] for p in op.attrs['blocks'])):
                continue
            inputs = []
            for value in op.args:
                source = fn.defs[value]
                while source.kind == 'cast' and value.type.bits == source.args[0].type.bits:
                    value = source.args[0]
                    source = fn.defs[value]
                if source.kind != 'ptradd' or source.args[0].type != op.results[0].type:
                    break
                offset = fn.defs[source.args[1]]
                if offset.kind != 'constant' or not isinstance(offset.attrs['value'], int):
                    break
                inputs.append((source.args[0], T.convert_integer(offset.attrs['value'], T.LONG)))
            if len(inputs) != len(op.args) or not inputs or len({n for _, n in inputs}) != 1 or not inputs[0][1]:
                continue
            if any(uncast(fn, value) in loop_phis for value, _ in inputs):
                continue
            merged = fn.make('phi', [op.results[0].type], [v for v, _ in inputs], {'blocks': list(op.attrs['blocks'])})
            block.ops[block.ops.index(op)] = merged
            op.kind, op.args, op.attrs = 'ptradd', (merged.results[0], fn.constant(T.LONG, inputs[0][1])), {}
            at = next((n for n, item in enumerate(block.ops) if item.kind != 'phi'), len(block.ops))
            block.ops.insert(at, op)


def factor_base_pointer_phis(fn):
    """Carry an offset through a join before passing a shared base to a tail call."""
    arguments = {uncast(fn, value) for b in fn.blocks if b.term.kind == 'ret'
                 for op in b.ops if op.kind == 'call' for value in op.args[1:]}
    dom = dominators(fn)
    for block in fn.blocks:
        for op in list(block.ops):
            if (op.kind != 'phi' or op.results[0].type.kind != 'ptr'
                    or op.results[0] not in arguments
                    or any(block in dom[p] for p in op.attrs['blocks'])):
                continue
            parts = []
            for value in op.args:
                source = fn.defs[uncast(fn, value)]
                if source.kind != 'ptradd':
                    break
                parts.append((uncast(fn, source.args[0]), source.args[1]))
            if len(parts) != len(op.args) or not parts or len({base for base, _ in parts}) != 1:
                continue
            offsets = []
            for pred, (_, value) in zip(op.attrs['blocks'], parts):
                if value.type != T.LONG:
                    cast = fn.make('cast', [T.LONG], [value])
                    pred.ops.append(cast)
                    value = cast.results[0]
                offsets.append(value)
            phi = fn.make('phi', [T.LONG], offsets, {'blocks': list(op.attrs['blocks'])})
            block.ops[block.ops.index(op)] = phi
            at = next((n for n, item in enumerate(block.ops) if item.kind != 'phi'), len(block.ops))
            base = parts[0][0]
            if base.type != op.results[0].type:
                cast = fn.make('cast', [op.results[0].type], [base])
                block.ops.insert(at, cast)
                at += 1
                base = cast.results[0]
            op.kind, op.args, op.attrs = 'ptradd', (base, phi.results[0]), {}
            block.ops.insert(at, op)


def correlated_pointer_phis(fn):
    """Carry one pointer through a join when the others have fixed offsets."""
    dom = dominators(fn)
    def address(value):
        offset = 0
        while True:
            value = uncast(fn, value)
            op = fn.defs[value]
            if op.kind != 'ptradd':
                return value, offset
            count = fn.defs[op.args[1]]
            if count.kind != 'constant' or not isinstance(count.attrs['value'], int):
                return value, offset
            offset += T.convert_integer(count.attrs['value'], T.LONG)
            value = op.args[0]
    for block in fn.blocks:
        if not any(parent in parent.term.targets for parent in fn.predecessors()[block]):
            continue
        phis = [op for op in block.ops if op.kind == 'phi' and op.results[0].type.kind == 'ptr'
                and not any(block in dom[p] for p in op.attrs['blocks'])]
        available = []
        for op in phis:
            parts = {b: address(v) for b, v in zip(op.attrs['blocks'], op.args)}
            for base, previous in available:
                if op.results[0].type != base.results[0].type or parts.keys() != previous.keys():
                    continue
                if any(parts[b][0] != previous[b][0] for b in parts):
                    continue
                offsets = {parts[b][1] - previous[b][1] for b in parts}
                if len(offsets) != 1:
                    continue
                offset = offsets.pop()
                op.kind, op.args, op.attrs = 'ptradd', (base.results[0], fn.constant(T.LONG, offset)), {}
                block.ops.remove(op)
                at = next((n for n, item in enumerate(block.ops) if item.kind != 'phi'), len(block.ops))
                block.ops.insert(at, op)
                break
            else:
                available.append((op, parts))


def cancel_offsets(fn):
    """s + ((a - s) + i) is a + i in the target's pointer-width arithmetic."""
    uses = Counter(v for b in fn.blocks for item in b.ops for v in item.args)
    uses.update(v for b in fn.blocks for v in b.term.args)
    writes, checked, pending = False, set(), [fn]
    while pending and not writes:
        callee = pending.pop()
        if callee in checked:
            continue
        checked.add(callee)
        for b in callee.blocks:
            for item in b.ops:
                if item.kind != 'call':
                    writes |= 'write' in item.attrs.get('memory', '')
                    continue
                target = callee.defs[uncast(callee, item.args[0])]
                if target.kind == 'symbol' and target.attrs['name'] in getattr(fn, 'callees', {}):
                    pending.append(fn.callees[target.attrs['name']])
    for block in fn.blocks:
        for op in list(block.ops):
            if op.kind != 'ptradd':
                continue
            index = uncast(fn, op.args[1])
            if writes and fn.defs[index].kind == 'add' and uses[index] > 1:
                continue
            terms, signs, delta = Counter(), {}, [0]

            def collect(value, coefficient=1):
                value = uncast(fn, value)
                source = fn.defs[value]
                if source.kind == 'constant' and isinstance(source.attrs['value'], int):
                    delta[0] += coefficient * T.convert_integer(source.attrs['value'], T.LONG)
                elif source.kind in ('add', 'sub') and value.type.bits == 64:
                    collect(source.args[0], coefficient)
                    collect(source.args[1], coefficient * (-1 if source.kind == 'sub' else 1))
                else:
                    terms[value] += coefficient
                    signs.setdefault(value, set()).add(coefficient > 0)

            collect(op.args[0])
            collect(op.args[1])
            original = uncast(fn, op.args[0])
            if terms[original] or signs.get(original) != {False, True}:
                continue
            pointers = [(v, n) for v, n in terms.items() if n and v.type.kind == 'ptr']
            scalars = [(v, n) for v, n in terms.items() if n and v.type.kind != 'ptr']
            if len(pointers) != 1 or pointers[0][1] != 1 or any(v.type.bits != 64 or abs(n) != 1 for v, n in scalars):
                continue
            inserted = []

            def make(kind, type_, args):
                new = fn.make(kind, [type_], args)
                inserted.append(new)
                return new.results[0]

            offset = None
            for value, coefficient in scalars:
                if value.type != T.LONG:
                    value = make('cast', T.LONG, [value])
                if offset is None:
                    offset = value if coefficient == 1 else make('neg', T.LONG, [value])
                else:
                    offset = make('add' if coefficient == 1 else 'sub', T.LONG, [offset, value])
            if delta[0] or offset is None:
                literal = fn.constant(T.LONG, delta[0])
                offset = make('add', T.LONG, [offset, literal]) if offset is not None else literal
            base = pointers[0][0]
            if base.type != op.args[0].type:
                base = make('cast', op.args[0].type, [base])
            op.args = base, offset
            at = block.ops.index(op)
            block.ops[at:at] = inserted


def hoist_returned_addresses(fn):
    """Common pure address work in separate return arms can precede the fork."""
    tails = fn.type.base.kind != 'ptr' and bool(getattr(fn, 'unaligned_loops', ()))
    if fn.type.base.kind != 'ptr' and not tails:
        return False
    changed, again = False, True
    while again:
        again = False
        needed, pending = set(), [v for b in fn.blocks if b.term.kind == 'ret' for v in b.term.args]
        if tails:
            pending.extend(v for b in fn.blocks if b.term.kind == 'ret' for op in b.ops if op.kind == 'call' for v in op.args[1:])
        while pending:
            value = pending.pop()
            if value in needed:
                continue
            needed.add(value)
            op = fn.defs[value]
            if op.kind == 'select':
                pending.extend(op.args)
            elif op.kind in ('phi', 'cast', 'ptradd', 'add', 'sub'):
                pending.extend(op.args)
        dom = dominators(fn)
        owners = {v: b for b in fn.blocks for op in b.ops for v in op.results}
        groups = {}
        for block in fn.blocks:
            for op in block.ops:
                if (op.kind not in ('cast', 'ptradd') or not tails and op.results[0] not in needed
                        or op.results[0].type.kind not in ('int', 'ptr')):
                    continue
                key = op.kind, op.results[0].type, op.args
                groups.setdefault(key, []).append((block, op))
        for pairs in groups.values():
            if (len(pairs) < 2 or not any(op.results[0] in needed for _, op in pairs)
                    or any(op.args != pairs[0][1].args for _, op in pairs)):
                continue
            target = max(set.intersection(*(dom[b] for b, _ in pairs)), key=lambda b: len(dom[b]))
            if any(owners.get(v) is not None and owners[v] not in dom[target] for v in pairs[0][1].args):
                continue
            local = [(b, op) for b, op in pairs if b is target]
            block, chosen = min(local, key=lambda pair: target.ops.index(pair[1])) if local else pairs[0]
            if block is not target:
                block.ops.remove(chosen)
                target.ops.append(chosen)
                owners[chosen.results[0]] = target
            replacements = {}
            for block, op in pairs:
                if op is not chosen:
                    replacements[op.results[0]] = chosen.results[0]
                    block.ops.remove(op)
            for block in fn.blocks:
                for op in block.ops:
                    op.args = tuple(replacements.get(v, v) for v in op.args)
                block.term.args = tuple(replacements.get(v, v) for v in block.term.args)
            changed = again = True
    if changed:
        fn.verify()
    return changed


def reuse_masked_addresses(fn):
    """Reuse an aligned address available on some incoming paths."""
    if not getattr(fn, 'guarded_pointer_returns', ()):
        return False
    pred, dom = fn.predecessors(), dominators(fn)
    changed = False
    for block in list(fn.blocks):
        if len(pred[block]) != 2:
            continue
        for op in list(block.ops):
            if op.kind != 'and' or op.results[0].type.bits != 64:
                continue
            mask = fn.defs[op.args[1]]
            if (mask.kind != 'constant' or not mask.attrs['value'] & (1 << 63)
                    or (~mask.attrs['value']) & ((1 << 64)-1) >= 4096):
                continue
            available = []
            for parent in pred[block]:
                candidates = [other for owner in dom[parent] for other in owner.ops if other is not op
                              and other.kind == op.kind and other.args == op.args and other.results[0].type == op.results[0].type]
                available.append(candidates[0] if candidates else None)
            if sum(other is not None for other in available) != 1:
                continue
            missing = available.index(None)
            parent = pred[block][missing]
            locations = {v: b for b in fn.blocks for item in b.ops for v in item.results}
            if any(locations.get(v) is not None and locations[v] not in dom[parent] for v in op.args):
                continue
            incoming = list(pred[block])
            if parent.term.kind != 'jmp':
                entry = fn.block('partial.alignment')
                entry.term = Term('jmp', targets=(block,))
                parent.term.targets = tuple(entry if b is block else b for b in parent.term.targets)
                for phi in block.ops:
                    if phi.kind == 'phi':
                        phi.attrs['blocks'] = [entry if b is parent else b for b in phi.attrs['blocks']]
                incoming[missing] = entry
                parent = entry
            inserted = fn.make(op.kind, [op.results[0].type], op.args, dict(op.attrs), op.source)
            parent.ops.append(inserted)
            available[missing] = inserted
            op.kind, op.args, op.attrs = 'phi', tuple(other.results[0] for other in available), {'blocks': incoming}
            block.ops.remove(op)
            block.ops.insert(0, op)
            fn.layout_owners[block] = pred[block][1-missing]
            fn.scan_registers = {**getattr(fn, 'scan_registers', {}), op.results[0]: 'r8',
                                  **{other.results[0]: 'r8' for other in available}}
            changed = True
    return changed


def guarded_pointer_distances(fn):
    """Compute a shared scan distance before testing its match mask."""
    if not getattr(fn, 'guarded_pointer_returns', ()):
        return False
    def count(value):
        while fn.defs[value].kind == 'cast' and value.type.bits >= fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value
    dom, groups = dominators(fn), {}
    for block in fn.blocks:
        for op in block.ops:
            if op.kind != 'sub' or op.results[0].type.bits != 64:
                continue
            left, right = (uncast(fn, value) for value in op.args)
            if left.type.kind == right.type.kind == 'ptr' and fn.defs[right].kind == 'param':
                groups.setdefault((left, right, op.results[0].type), []).append((block, op))
    changed = False
    for (pointer, base, type_), pairs in groups.items():
        if len(pairs) < 2:
            continue
        owner = max(set.intersection(*(dom[block] for block, _ in pairs)), key=lambda b: len(dom[b]))
        locations = {value: block for block in fn.blocks for op in block.ops for value in op.results}
        if any(locations.get(value) is not None and locations[value] not in dom[owner] for value in (pointer, base)):
            continue
        left, right = (fn.make('cast', [type_], [value]) for value in (pointer, base))
        distance = fn.make('sub', [type_], [left.results[0], right.results[0]])
        owner.ops.extend([left, right, distance])
        replacements = {op.results[0]: distance.results[0] for _, op in pairs}
        for block in fn.blocks:
            for op in block.ops:
                op.args = tuple(replacements.get(value, value) for value in op.args)
            block.term.args = tuple(replacements.get(value, value) for value in block.term.args)
        fn.scan_registers = {**getattr(fn, 'scan_registers', {}), pointer: 'r8', distance.results[0]: 'r8'}
        fn.pointer_scan_steps = {*getattr(fn, 'pointer_scan_steps', ()), pointer}
        # Keep the selected address in terms of the same checked index.
        for block in fn.blocks:
            if block.term.kind != 'ret' or len(block.term.args) != 1:
                continue
            selected = fn.defs[block.term.args[0]]
            if selected.kind != 'select':
                continue
            test = fn.defs[selected.args[0]]
            address = fn.defs[uncast(fn, selected.args[1])]
            if test.kind != 'lt' or address.kind != 'ptradd' or uncast(fn, address.args[0]) != pointer:
                continue
            index = fn.defs[uncast(fn, test.args[0])]
            if (index.kind != 'add' or uncast(fn, index.args[0]) != distance.results[0]
                    or count(index.args[1]) != count(address.args[1])):
                continue
            cast = fn.make('cast', [T.LONG], [test.args[0]])
            block.ops.insert(block.ops.index(address), cast)
            address.args = base, cast.results[0]
            fn.scan_registers[index.results[0]] = 'r8'
        changed = True
    return changed


def tail_distances(fn):
    """Avoid an intermediate distance in n - (from - start) on read-only paths."""
    if any(op.kind != 'call' and 'write' in op.attrs.get('memory', '') for b in fn.blocks for op in b.ops):
        return
    arguments = {v for b in fn.blocks if b.term.kind == 'ret' and b.ops and b.ops[-1].kind == 'call'
                 for v in b.ops[-1].args[1:]}
    for block in fn.blocks:
        for op in list(block.ops):
            if op.kind != 'sub' or op.results[0] not in arguments or op.results[0].type.bits != 64:
                continue
            distance = fn.defs[uncast(fn, op.args[1])]
            if distance.kind != 'sub' or any(uncast(fn, v).type.kind != 'ptr' for v in distance.args):
                continue
            type_ = op.results[0].type
            start = fn.make('cast', [type_], [uncast(fn, distance.args[1])])
            end = fn.make('cast', [type_], [uncast(fn, distance.args[0])])
            total = fn.make('add', [type_], [op.args[0], start.results[0]])
            at = block.ops.index(op)
            block.ops[at:at] = [start, end, total]
            op.args = total.results[0], end.results[0]


def return_loop_addresses(fn):
    """Reuse page distances on incoming edges of a masked scan step."""
    if getattr(fn, 'shared_load_return', None) is None:
        return False
    fn.scan_registers = {}
    changed = False
    # (align_down(p, A) + C) - p is C - (p & (A - 1)).
    for block in fn.blocks:
        for op in list(block.ops):
            if op.kind != 'sub' or op.results[0].type.bits != 64:
                continue
            left, right = (uncast(fn, v) for v in op.args)
            address = fn.defs[left]
            if address.kind != 'ptradd' or address not in block.ops:
                continue
            base, count = (fn.defs[uncast(fn, v)] for v in address.args)
            if base.kind != 'and' or count.kind != 'constant' or uncast(fn, base.args[0]) != right:
                continue
            mask = fn.defs[base.args[1]]
            if mask.kind != 'constant':
                continue
            low = (~mask.attrs['value']) & ((1 << 64) - 1)
            if not low or low & (low + 1):
                continue
            offset = fn.make('and', [T.ULONG], [base.args[0], fn.constant(T.ULONG, low)])
            cast = fn.make('cast', [op.results[0].type], offset.results)
            step = fn.make('sub', [op.results[0].type], [fn.constant(op.results[0].type, count.attrs['value']), cast.results[0]])
            at = block.ops.index(address)
            block.ops[at:at] = [offset, cast, step]
            address.args = right, step.results[0]
            for other in fn.blocks:
                for item in other.ops:
                    item.args = tuple(step.results[0] if v == op.results[0] else v for v in item.args)
                other.term.args = tuple(step.results[0] if v == op.results[0] else v for v in other.term.args)
            block.ops.remove(op)
            changed = True
    dom = dominators(fn)
    for head in fn.blocks:
        if head.term.kind != 'br':
            continue
        test = fn.defs[head.term.args[0]]
        if test.kind != 'and' or fn.defs[test.args[1]].kind != 'constant':
            continue
        clear = fn.defs[test.args[1]].attrs['value']
        fn.scan_registers[test.results[0]] = 'rdx'
        subject, zero, other = uncast(fn, test.args[0]), head.term.targets[1], head.term.targets[0]
        for block in fn.blocks:
            if zero not in dom[block] or other in dom[block]:
                continue
            for op in block.ops:
                if op.kind != 'and' or uncast(fn, op.args[0]) != subject:
                    continue
                mask = fn.defs[op.args[1]]
                if mask.kind == 'constant' and mask.attrs['value'] & clear:
                    op.args = op.args[0], fn.constant(op.args[1].type, mask.attrs['value'] & ~clear)
                    changed = True
    owners = {v: b for b in fn.blocks for op in b.ops for v in op.results}
    def key(value):
        source = fn.defs[value]
        if source.kind == 'cast' and value.type.bits == source.args[0].type.bits:
            return key(source.args[0])
        if source.kind == 'constant':
            return 'constant', value.type.bits, source.attrs['value']
        if source.kind in ('and', 'sub'):
            return source.kind, value.type.bits, tuple(key(v) for v in source.args)
        return value
    pred = fn.predecessors()
    for block in list(fn.blocks):
        if len(pred[block]) != 2:
            continue
        for op in list(block.ops):
            if op.kind != 'sub' or op.results[0].type.bits != 64:
                continue
            total, part = (fn.defs[v] for v in op.args)
            if total.kind != 'constant' or part.kind != 'and':
                continue
            n, mask = total.attrs['value'], fn.defs[part.args[1]]
            if not isinstance(n, int) or n <= 0 or n & (n - 1) or mask.kind != 'constant' or mask.attrs['value'] != n - 1:
                continue
            available = []
            for parent in pred[block]:
                candidates = [v for v, owner in owners.items() if owner in dom[parent]
                              and v.type == op.results[0].type and key(v) == key(op.results[0])]
                available.append(candidates[0] if candidates else None)
            if sum(v is not None for v in available) != 1:
                continue
            missing = available.index(None)
            parent = pred[block][missing]
            inserted = fn.block('partial.address')
            generated = {}
            def clone(value):
                if owners.get(value) is not block:
                    return value
                if value in generated:
                    return generated[value]
                old = fn.defs[value]
                if old.kind not in ('cast', 'and', 'sub'):
                    raise ValueError('non-arithmetic page distance')
                args = [clone(v) for v in old.args]
                new = fn.make(old.kind, [value.type], args, dict(old.attrs), old.source)
                inserted.ops.append(new)
                generated[value] = new.results[0]
                return new.results[0]
            available[missing] = clone(op.results[0])
            inserted.term = Term('jmp', targets=(block,))
            parent.term.targets = tuple(inserted if b is block else b for b in parent.term.targets)
            for phi in block.ops:
                if phi.kind == 'phi':
                    phi.attrs['blocks'] = [inserted if b is parent else b for b in phi.attrs['blocks']]
            incoming = [inserted if b is parent else b for b in pred[block]]
            phi = fn.make('phi', [op.results[0].type], available, {'blocks': incoming})
            fn.scan_registers[phi.results[0]] = 'r8'
            block.ops.insert(0, phi)
            for other in fn.blocks:
                for item in other.ops:
                    item.args = tuple(phi.results[0] if v == op.results[0] else v for v in item.args)
                other.term.args = tuple(phi.results[0] if v == op.results[0] else v for v in other.term.args)
            block.ops.remove(op)
            changed = True
            break
    users = {}
    for block in fn.blocks:
        for op in block.ops:
            for value in op.args:
                users.setdefault(value, []).append(op)
    for block in fn.blocks:
        for op in list(block.ops):
            if op.kind == 'div' and op.results[0].type.bits == 64:
                fn.scan_registers[op.results[0]] = 'rcx'
            if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'ctz_nonzero':
                if any(item.kind in ('lt', 'le', 'gt', 'ge')
                       or item.kind == 'cast' and any(use.kind in ('lt', 'le', 'gt', 'ge')
                                                    for use in users.get(item.results[0], ()))
                       for item in users.get(op.results[0], ())):
                    fn.scan_registers[op.results[0]] = 'rdx'
            if (op.kind == 'asm' and not op.attrs['code'] and len(op.args) == 2
                    and all(v.type.kind == 'ptr' for v in op.args)):
                inputs = [fn.defs[v] for v in op.args]
                if (all(item.kind == 'ptradd' and item in block.ops and fn.defs[item.args[1]].kind == 'constant' for item in inputs)
                        and not any(item.results[0] in other.args for item in inputs for other in inputs)):
                    start = min(block.ops.index(item) for item in inputs)
                    block.ops = [item for item in block.ops if item not in inputs]
                    block.ops[start:start] = inputs
            if op.kind != 'select' or fn.defs[op.args[0]].kind not in ('lt', 'le'):
                continue
            fn.scan_registers[op.results[0]] = 'rcx'
            inputs = [fn.defs[v] for v in op.args[1:]]
            if any(item.kind != 'sub' or item not in block.ops or fn.defs[item.args[0]].kind != 'constant' for item in inputs):
                continue
            required, ordered = set(), []
            def visit(item):
                if item in required or item not in block.ops:
                    return
                if item.kind not in ('cast', 'and', 'sub'):
                    return
                for value in item.args:
                    visit(fn.defs[value])
                required.add(item)
                ordered.append(item)
            for item in inputs:
                visit(item)
            start = min(block.ops.index(item) for item in required)
            block.ops = [item for item in block.ops if item not in required]
            block.ops[start:start] = ordered
    return changed
