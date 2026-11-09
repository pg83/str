"""Expand small bounded loops while keeping their exit regions shared."""
from . import types as T
from .alignment import alignments
from .ir import Term, dominators, promote


def unroll_bounded_loops(fn, allow_alignment=True):
    changed = False
    for head in list(fn.blocks):
        if head not in fn.blocks or head.unroll_disabled or head.term.kind != 'br':
            continue
        pred, dom = fn.predecessors(), dominators(fn)
        tails = [b for b in pred[head] if head in dom[b]]
        incoming = [b for b in pred[head] if b not in tails]
        if len(tails) != 1 or len(incoming) != 1:
            continue
        tail, before = tails[0], incoming[0]
        if tail is head or tail.term.kind != 'jmp' or tail.term.targets != (head,):
            continue
        members, todo = {head}, [tail]
        while todo:
            block = todo.pop()
            if block not in members:
                members.add(block)
                todo.extend(pred[block])
        phis = [op for op in head.ops if op.kind == 'phi']
        initial = {p.results[0]: p.args[p.attrs['blocks'].index(before)] for p in phis}
        count = head.fixed_trip_count
        keep = 0
        if count is None:
            if not allow_alignment:
                continue
            condition = fn.defs[head.term.args[0]]
            if condition.kind in ('eq', 'ne') and fn.defs[condition.args[1]].kind == 'constant' and fn.defs[condition.args[1]].attrs['value'] == 0:
                keep = condition.kind == 'eq'
                condition = fn.defs[condition.args[0]]
            if condition.kind != 'and' or head.term.targets[keep] not in members or head.term.targets[1 - keep] in members:
                continue
            mask = fn.defs[condition.args[1]]
            if mask.kind != 'constant' or not isinstance(mask.attrs['value'], int):
                continue
            span = mask.attrs['value'] + 1
            if span <= 1 or span & (span - 1):
                continue

            def uncast(value):
                op = fn.defs[value]
                while op.kind == 'cast' and value.type.bits == op.args[0].type.bits:
                    value = op.args[0]
                    op = fn.defs[value]
                return value

            induction = uncast(condition.args[0])
            phi = fn.defs[induction]
            if phi.kind != 'phi' or phi not in head.ops:
                continue
            update = fn.defs[uncast(phi.args[phi.attrs['blocks'].index(tail)])]
            if update.kind not in ('ptradd', 'add') or uncast(update.args[0]) != induction:
                continue
            step = fn.defs[update.args[1]]
            if step.kind != 'constant' or not isinstance(step.attrs['value'], int):
                continue
            step = step.attrs['value']
            if step <= 0 or step & (step - 1) or span % step or not 2 <= span // step <= 8:
                continue
            if alignments(fn).get(initial[induction], 1) < step:
                continue
            count = span // step
        elif head.term.targets[0] not in members or head.term.targets[1] in members:
            continue
        body = [b for b in fn.blocks if b in members]
        if count * sum(op.kind not in ('phi', 'cast', 'ptradd') for b in body for op in b.ops) > 160:
            continue
        if any(op.kind in ('call', 'asm') or op.attrs.get('volatile') for op in head.ops):
            continue
        # Hidden scalar slots repair values used beyond several different
        # loop exits. Promotion then reconstructs the necessary exit phis.
        owners = {v: b for b in body for op in b.ops for v in op.results}
        liveout = {v for b in fn.blocks if b not in members
                   for v in (*b.term.args, *(v for op in b.ops for v in op.args)) if v in owners}
        slots = {}
        for v in sorted(liveout, key=lambda v: v.id):
            slot = fn.make('alloca', [T.pointer(v.type)], attrs={'name': 'unroll.live.%d' % v.id,
                           'size': v.type.size, 'alignment': v.type.alignment})
            fn.allocas.append(slot)
            slots[v] = slot.results[0]
        exits, inserted, copies = {}, [], []
        current = initial
        for n in range(count):
            blocks = {b: fn.block('align.%d.%s' % (n, b.name)) for b in body}
            inserted.extend(blocks.values())
            values, pairs = dict(current), []
            for b in body:
                blocks[b].unroll_disabled = b.unroll_disabled
                for op in b.ops:
                    if b is head and op.kind == 'phi':
                        continue
                    clone = fn.make(op.kind, [v.type for v in op.results], attrs=dict(op.attrs), source=op.source)
                    blocks[b].ops.append(clone)
                    values.update(zip(op.results, clone.results))
                    pairs.append((op, clone))
            for old, clone in pairs:
                clone.args = tuple(values.get(v, v) for v in old.args)
                if old.kind == 'phi':
                    clone.attrs['blocks'] = [blocks[b] for b in old.attrs['blocks']]
            current = {p.results[0]: values.get(p.args[p.attrs['blocks'].index(tail)], p.args[p.attrs['blocks'].index(tail)]) for p in phis}
            copies.append((blocks, values, dict(current)))
        exit_ = head.term.targets[1 - keep]
        for n, (blocks, values, next_) in enumerate(copies):
            for old, block in blocks.items():
                t = old.term
                targets = []
                for target in t.targets:
                    mapping = values
                    source = old
                    if target is head:
                        if n + 1 < count:
                            target = copies[n + 1][0][head]
                        else:
                            target, mapping, source = exit_, {**values, **next_}, head
                    elif target in members:
                        target = blocks[target]
                    if target not in inserted:
                        exits.setdefault((source, target), []).append((block, mapping))
                    targets.append(target)
                block.term = Term(t.kind, tuple(values.get(v, v) for v in t.args), tuple(targets), t.source, t.weights)
                if any(target not in inserted for target in targets):
                    mapping = {**values, **next_} if old is tail and n + 1 == count else values
                    for value, slot in slots.items():
                        if owners[value] in dom[old]:
                            block.ops.append(fn.make('store', args=[slot, mapping[value]], attrs={
                                'memory': 'write', 'width': value.type.size, 'alignment': value.type.alignment, 'volatile': False}))
        if not hasattr(fn, 'layout_conditions'):
            fn.layout_conditions = {}
        for (source, destination), edges in exits.items():
            if source is not head and source.term.kind == 'br':
                owner = copies[-1] if head.fixed_trip_count is not None else copies[0]
                fn.layout_conditions[destination] = owner[1].get(source.term.args[0], source.term.args[0])
                if head.fixed_trip_count is not None:
                    fn.fixed_loop_exits = set(getattr(fn, 'fixed_loop_exits', ())) | {destination}
                    fn.fixed_loop_entries = set(getattr(fn, 'fixed_loop_entries', ())) | {before}
        outside = [b for b in fn.blocks if b not in members and b not in inserted]
        for b in outside:
            b.term.targets = tuple(copies[0][0][head] if t is head else t for t in b.term.targets)
            for op in b.ops:
                if op.kind != 'phi':
                    continue
                pairs = []
                for p, v in zip(op.attrs['blocks'], op.args):
                    if p not in members:
                        pairs.append((p, v))
                    else:
                        pairs.extend((block, mapping.get(v, v)) for block, mapping in exits.get((p, b), ()))
                op.attrs['blocks'], op.args = [p for p, _ in pairs], tuple(v for _, v in pairs)
        # Replace surviving non-edge uses by loads from the temporary slots.
        for b in outside:
            loaded = {}

            def read(value, block, out):
                if value not in slots:
                    return value
                key = block, value, id(out)
                if key not in loaded:
                    op = fn.make('load', [value.type], [slots[value]], {'memory': 'read', 'width': value.type.size,
                                 'alignment': value.type.alignment, 'volatile': False})
                    out.append(op)
                    loaded[key] = op.results[0]
                return loaded[key]

            out = []
            for op in b.ops:
                if op.kind == 'phi':
                    op.args = tuple(read(v, p, p.ops) for p, v in zip(op.attrs['blocks'], op.args))
                else:
                    op.args = tuple(read(v, b, out) for v in op.args)
                out.append(op)
            b.term.args = tuple(read(v, b, out) for v in b.term.args)
            b.ops = out
        fn.blocks = [b for b in fn.blocks if b not in members and b not in inserted]
        at = fn.blocks.index(before) + 1
        fn.blocks[at:at] = inserted
        promote(fn)
        fn.verify()
        changed = True
    return changed
