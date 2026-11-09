"""If-conversion: a branch whose arms only compute a few values for a phi
becomes selects (cmov) when the arms are cheap and cannot trap, unless the
branch is known to go one way (__builtin_expect); a branch marked
unpredictable is converted whenever it can be."""
from .ir import DIVISIONS

ARM_COST = 2


def speculable(op):
    return (not op.effects and op.kind not in ('load', 'phi', 'alloca', 'vop') and op.kind not in DIVISIONS
            and op.ty.kind in ('i', 'p'))


def arm(block, preds, join, origin):
    """The ops of an arm block that only computes and jumps to join, or None."""
    if block is join:
        return []
    if preds[block] != [origin] or block.term.kind != 'jmp' or block.term.targets[0] is not join:
        return None
    if any(op.kind == 'phi' for op in block.ops):
        return None
    if not all(speculable(op) for op in block.ops):
        return None
    return list(block.ops)


def ifconv(fn):
    changed = False
    preds = fn.preds()
    for b in list(fn.blocks):
        t = b.term
        if b not in fn.blocks or t.kind != 'br':
            continue
        w = t.attrs.get('weights')
        unpredictable = t.attrs.get('unpredictable')
        if w and not unpredictable and max(w) > 10 * min(w):
            continue
        yes, no = t.targets
        join = None
        for cand in (yes.succs[0] if yes.term.kind == 'jmp' else None, no.succs[0] if no.term.kind == 'jmp' else None, yes, no):
            if cand is not None and cand is not b:
                a1, a2 = arm(yes, preds, cand, b), arm(no, preds, cand, b)
                if a1 is not None and a2 is not None and (a1 or a2 or yes is cand or no is cand):
                    join = cand
                    break
        if join is None or join is b:
            continue
        ops_yes, ops_no = arm(yes, preds, join, b), arm(no, preds, join, b)
        phis = join.phis()
        if not phis or any(p.ty.kind not in ('i', 'p') for p in phis):
            continue
        limit = ARM_COST * 2 if unpredictable else ARM_COST
        if len(ops_yes) > limit or len(ops_no) > limit:
            continue
        # the edge (or arm block) through which each way reaches join
        via_yes = b if yes is join else yes
        via_no = b if no is join else no
        if via_yes is via_no:
            continue
        for arm_ops in (ops_yes, ops_no):
            for op in arm_ops:
                op.block.remove(op)
                b.insert(len(b.ops), op)
        cond = t.args[0]
        selects = []
        for phi in phis:
            vy, vn = phi.incoming(via_yes), phi.incoming(via_no)
            if vy is vn:
                selects.append((phi, vy))
                continue
            sel = fn.new('select', [cond, vy, vn], phi.ty)
            b.append(sel)
            selects.append((phi, sel.result))
        for phi, v in selects:
            for p in (via_yes, via_no):
                if p in phi.preds:
                    phi.remove_incoming(p)
            phi.add_incoming(b, v)
        jmp = fn.new('jmp')
        jmp.targets = [join]
        b.set_term(jmp)
        for arm_block in (yes, no):
            if arm_block is not join:
                arm_block.term.detach()
                fn.blocks.remove(arm_block)
        preds = fn.preds()
        changed = True
    return changed
