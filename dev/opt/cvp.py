"""Correlated value propagation: what a branch decided holds in the blocks
its edge dominates. Walking the dominator tree with the conditions taken so
far, a comparison whose operands' ranges decide it becomes a constant, and
a select on one picks its value."""
from .facts import Facts
from .ir import Dominators, I1, INVERT
from .simplify import decide

SIGNED = {'slt': 'ult', 'sle': 'ule', 'sgt': 'ugt', 'sge': 'uge'}


def bound(pred, c, bits):
    """x pred c as an unsigned interval of x, or None when it is not one."""
    mask = (1 << bits) - 1
    if pred == 'eq':
        return c, c
    if pred == 'ult':
        return (0, c - 1) if c > 0 else (1, 0)
    if pred == 'ule':
        return 0, c
    if pred == 'ugt':
        return (c + 1, mask) if c < mask else (1, 0)
    if pred == 'uge':
        return c, mask
    if pred == 'ne':
        if c == 0:
            return 1, mask
        if c == mask:
            return 0, mask - 1
    return None


def cvp(fn):
    dom = Dominators(fn)
    preds = fn.preds()
    changed = False
    stack = [(fn.entry, {})]
    while stack:
        b, assume = stack.pop()
        facts = Facts(fn, assume)
        for op in list(b.ops):
            if op.kind == 'icmp':
                a, c = op.args
                pred = op.attrs['pred']
                lo1, hi1 = facts.range(a)
                lo2, hi2 = facts.range(c)
                half = 1 << (a.ty.bits - 1)
                if pred in SIGNED and hi1 < half and hi2 < half:
                    pred = SIGNED[pred]
                verdict = decide(pred, lo1, hi1, lo2, hi2)
                if verdict is not None:
                    fn.replace_all_uses(op.result, fn.const(I1, int(verdict)))
                    op.detach()
                    b.remove(op)
                    changed = True
            elif op.kind == 'select' and op.args[0].const is not None:
                fn.replace_all_uses(op.result, op.args[1] if op.args[0].const else op.args[2])
                op.detach()
                b.remove(op)
                changed = True
        for child in dom.children[b]:
            extra = assume
            if preds[child] == [b] and b.term.kind == 'br' and b.term.args[0].op.kind == 'icmp':
                cmp = b.term.args[0].op
                x, c = cmp.args
                if c.const is not None and x.ty.kind in ('i', 'p'):
                    pred = cmp.attrs['pred']
                    if child is b.term.targets[1]:
                        pred = INVERT[pred]
                    if pred in SIGNED:
                        lo, hi = facts.range(x)
                        pred = SIGNED[pred] if hi < 1 << (x.ty.bits - 1) and c.const < 1 << (x.ty.bits - 1) else None
                    interval = bound(pred, c.const, x.ty.bits) if pred else None
                    if interval is not None:
                        extra = dict(assume)
                        old = extra.get(x, (0, x.ty.mask))
                        extra[x] = (max(old[0], interval[0]), min(old[1], interval[1]))
                        if x.op.kind in ('add', 'sub') and x.op.args[1].const is not None:
                            # x = y + k in [lo, hi]: y in [lo - k, hi - k] when that does not wrap
                            y, k = x.op.args[0], x.op.args[1].const
                            k = k if x.op.kind == 'add' else -k
                            lo, hi = (interval[0] - k) & x.ty.mask, (interval[1] - k) & x.ty.mask
                            if lo <= hi:
                                old = extra.get(y, (0, y.ty.mask))
                                extra[y] = (max(old[0], lo), min(old[1], hi))
            stack.append((child, extra))
    return changed
