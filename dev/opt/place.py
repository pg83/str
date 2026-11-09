"""Where pure ops go: out of loops into the preheader when their operands
are all defined outside (loop-invariant code motion), and down into the one
successor whose paths use them, so that a value needed only on one way is
not computed on the other (sinking)."""
from .ir import Dominators, find_loops


def movable(op):
    return not op.effects and op.kind not in ('phi', 'load', 'alloca') and not (op.kind == 'vop' and op.attrs.get('memory'))


def preheader(fn, loop, preds):
    outside = [p for p in preds[loop.header] if p not in loop.blocks]
    if len(outside) == 1 and outside[0].term.kind == 'jmp':
        return outside[0]
    pre = fn.block('preheader')
    jmp = fn.new('jmp')
    jmp.targets = [loop.header]
    pre.set_term(jmp)
    for p in outside:
        p.term.targets = [pre if x is loop.header else x for x in p.term.targets]
    for phi in loop.header.phis():
        incoming = [(p, phi.incoming(p)) for p in outside]
        for p, _ in incoming:
            phi.remove_incoming(p)
        values = {v for _, v in incoming}
        if len(values) == 1:
            phi.add_incoming(pre, values.pop())
        else:
            new = fn.phi(phi.ty)
            for p, v in incoming:
                new.add_incoming(p, v)
            pre.insert(0, new)
            phi.add_incoming(pre, new.result)
    if 'nounroll' in loop.header.attrs:
        pass
    return pre


def hoist(fn):
    changed = False
    dom = Dominators(fn)
    info = find_loops(fn, dom)
    preds = fn.preds()
    for loop in reversed(info.loops):
        pre = None
        inside = loop.blocks
        for b in [b for b in dom.order if b in inside]:
            for op in list(b.ops):
                if not movable(op):
                    continue
                if any(a.op.block is not None and a.op.block in inside for a in op.args):
                    continue
                if pre is None:
                    pre = preheader(fn, loop, preds)
                    preds = fn.preds()
                b.remove(op)
                pre.append(op)
                changed = True
    return changed


def sink(fn):
    changed = False
    for _ in range(4):
        moved = False
        dom = Dominators(fn)
        preds = fn.preds()
        for b in dom.order:
            for op in reversed(list(b.ops)):
                if not movable(op) or not op.results:
                    continue
                uses = [u for r in op.results for u in r.uses]
                if not uses:
                    continue
                places = set()
                for u in uses:
                    if u.kind == 'phi':
                        places.update(p for p, a in zip(u.preds, u.args) if a in op.results)
                    else:
                        places.add(u.block)
                if b in places:
                    continue
                target = None
                for s in b.succs:
                    if preds[s] == [b] and all(dom.dominates(s, p) for p in places):
                        target = s
                        break
                if target is None:
                    continue
                b.remove(op)
                target.insert(len(target.phis()), op)
                moved = changed = True
        if not moved:
            break
    return changed
