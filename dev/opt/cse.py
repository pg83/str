"""Common subexpressions: a pure op computed again where an equal one
dominates it takes that one's value (a scoped table over the dominator
tree). Loads are merged within a block, up to the next op that may write
memory."""
from .ir import Dominators, COMMUTATIVE


def key(op):
    if op.kind in ('phi', 'alloca') or op.effects or op.kind == 'load' or not op.results:
        return None
    if op.kind == 'vop' and op.attrs.get('memory'):
        return None
    args = [a.id for a in op.args]
    if op.kind in COMMUTATIVE:
        args.sort()
    attrs = tuple(sorted((k, v if not isinstance(v, (list, dict)) else repr(v)) for k, v in op.attrs.items() if k != 'builtin'))
    return op.kind, tuple(args), attrs, tuple(r.ty for r in op.results)


def cse(fn):
    dom = Dominators(fn)
    changed = False
    table = {}
    stack = [(dom.order[0], None)]
    while stack:
        b, undo = stack.pop()
        if undo is not None:
            for k in undo:
                del table[k]
            continue
        added = []
        loads = {}
        for op in list(b.ops):
            if op.kind == 'load' and not op.attrs.get('volatile'):
                lk = (op.args[0].id, op.result.ty, op.attrs.get('align'))
                prior = loads.get(lk)
                if prior is not None:
                    fn.replace_all_uses(op.result, prior)
                    op.detach()
                    b.remove(op)
                    changed = True
                else:
                    loads[lk] = op.result
                continue
            if op.writes_memory or op.kind in ('call', 'asm'):
                loads.clear()
                continue
            k = key(op)
            if k is None:
                continue
            prior = table.get(k)
            if prior is not None:
                for r, p in zip(op.results, prior.results):
                    fn.replace_all_uses(r, p)
                op.detach()
                b.remove(op)
                changed = True
            else:
                table[k] = op
                added.append(k)
        stack.append((b, added))
        stack.extend((c, None) for c in reversed(dom.children[b]))
    return changed
