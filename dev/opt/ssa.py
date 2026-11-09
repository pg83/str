"""Stack slots to SSA values: small aggregates accessed at constant offsets
are split into one slot per scalar (sroa), then every slot that is only
loaded and stored whole becomes values and phis (mem2reg, Cytron's
algorithm over the dominance frontiers)."""
from .ir import Dominators


def accesses(alloca):
    """[(op, offset, ty)] for every use of the slot, or None if it escapes."""
    out = []
    todo = [(alloca.result, 0)]
    while todo:
        v, offset = todo.pop()
        for u in v.uses:
            if u.kind == 'load' and u.args[0] is v and not u.attrs.get('volatile'):
                out.append((u, offset, u.result.ty))
            elif u.kind == 'store' and u.args[0] is v and u.args[1] is not v and not u.attrs.get('volatile'):
                out.append((u, offset, u.args[1].ty))
            elif u.kind == 'ptradd' and u.args[0] is v and u.args[1].const is not None:
                todo.append((u.result, offset + u.args[1].const))
            else:
                return None
    return out


def sroa(fn):
    changed = False
    for a in [op for op in fn.entry.ops if op.kind == 'alloca']:
        uses = accesses(a)
        if not uses:
            continue
        slots = {}
        for op, offset, ty in uses:
            slots.setdefault(offset, set()).add(ty)
        if len(slots) < 2 or any(len(tys) != 1 for tys in slots.values()):
            continue
        spans = sorted((offset, next(iter(tys)).size) for offset, tys in slots.items())
        if any(o1 + s1 > o2 for (o1, s1), (o2, _) in zip(spans, spans[1:])) or spans[0][0] < 0 or spans[-1][0] + spans[-1][1] > a.attrs['size']:
            continue
        new = {}
        at = fn.entry.ops.index(a)
        for offset, size in spans:
            op = fn.new('alloca', (), a.result.ty, {'size': size, 'align': size, 'name': '%s.%d' % (a.attrs.get('name', ''), offset)})
            fn.entry.insert(at, op)
            new[offset] = op.result
        for op, offset, ty in uses:
            op.set_arg(0, new[offset])
        changed = True
    if changed:
        from .simplify import dead_code
        dead_code(fn)
    return changed


def promotable(a):
    ty = None
    for u in a.result.uses:
        if u.kind == 'load' and u.args[0] is a.result and not u.attrs.get('volatile'):
            t = u.result.ty
        elif u.kind == 'store' and u.args[0] is a.result and u.args[1] is not a.result and not u.attrs.get('volatile'):
            t = u.args[1].ty
        else:
            return None
        if t.size != a.attrs['size'] or ty is not None and t is not ty:
            return None
        ty = t
    return ty


def mem2reg(fn):
    slots = {}
    for op in fn.entry.ops:
        if op.kind == 'alloca':
            ty = promotable(op)
            if ty is not None:
                slots[op] = ty
    if not slots:
        return False
    dom = Dominators(fn)
    preds = fn.preds()
    frontier = {b: set() for b in dom.order}
    for b in dom.order:
        if len(preds[b]) >= 2:
            for p in preds[b]:
                runner = p
                while runner is not dom.idom[b]:
                    frontier[runner].add(b)
                    runner = dom.idom[runner]
    placed = {}   # (block, slot) -> phi
    for slot, ty in slots.items():
        defs = {u.block for u in slot.result.uses if u.kind == 'store'}
        work, seen = list(defs), set()
        while work:
            b = work.pop()
            for d in frontier[b]:
                if d in seen:
                    continue
                seen.add(d)
                phi = fn.phi(ty)
                d.insert(0, phi)
                placed[(d, slot)] = phi
                if d not in defs:
                    work.append(d)
    by_block = {}
    for (b, slot), phi in placed.items():
        by_block.setdefault(b, []).append((slot, phi))
    stacks = {slot: [fn.undef(ty)] for slot, ty in slots.items()}
    stack = [(dom.order[0], False)]
    pushed_at = {}
    while stack:
        b, done = stack.pop()
        if done:
            for slot in pushed_at.pop(b):
                stacks[slot].pop()
            continue
        pushed = []
        for slot, phi in by_block.get(b, ()):
            stacks[slot].append(phi.result)
            pushed.append(slot)
        for op in list(b.ops):
            if op.kind == 'load' and op.args and op.args[0].op in slots:
                fn.replace_all_uses(op.result, stacks[op.args[0].op][-1])
                op.detach()
                b.remove(op)
            elif op.kind == 'store' and op.args[0].op in slots:
                slot = op.args[0].op
                stacks[slot].append(op.args[1])
                pushed.append(slot)
                op.detach()
                b.remove(op)
        for s in b.succs:
            for slot, phi in by_block.get(s, ()):
                phi.add_incoming(b, stacks[slot][-1])
        pushed_at[b] = pushed
        stack.append((b, True))
        stack.extend((c, False) for c in reversed(dom.children[b]))
    for slot in slots:
        slot.detach()
        fn.entry.remove(slot)
    return True
