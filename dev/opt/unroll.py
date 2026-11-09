"""Full unrolling of loops that run a constant number of times.

A loop qualifies when a branch it takes on every iteration (its block
dominates the latch) leaves on a comparison of an induction variable
(a header phi from a constant start, stepped by a constant) with a constant:
the iteration that leaves is found by running the comparison. The loop is
then copied once per iteration, iteration i's latch jumping to iteration
i + 1's header; the copies' comparisons fold to constants afterwards and the
paths they rule out disappear. Data-dependent exits stay in each copy. Values
the loop defines and code after it uses get the copy that reaches the use
(ssaupdate.py)."""
from .ir import Dominators, find_loops, I1
from .simplify import compare
from .ssaupdate import SSAUpdater

MAX_COPIES = 16
MAX_OPS = 600


def induction(phi, pre, latch):
    """(start, step) of a header phi stepped by a constant, else None."""
    start = phi.incoming(pre).const
    nxt = phi.incoming(latch)
    if start is None or nxt.op.kind != 'add' or nxt.op.args[0] is not phi.result or nxt.op.args[1].const is None:
        return None
    return start, nxt.op.args[1].const


def conditions(v, exit_when):
    """The comparisons whose truth (exit_when) alone leaves the loop: v itself,
    the conjuncts of an and the loop leaves on being false, the disjuncts of
    an or it leaves on being true, through negations."""
    op = v.op
    if op.kind == 'icmp':
        return [(op, exit_when)]
    if v.ty is I1 and op.kind == 'xor' and op.args[1].const == 1:
        return conditions(op.args[0], not exit_when)
    if v.ty is I1 and (op.kind == 'and' and not exit_when or op.kind == 'or' and exit_when):
        return conditions(op.args[0], exit_when) + conditions(op.args[1], exit_when)
    return []


def trip_count(fn, loop, preds, dom):
    """(iteration that leaves, preheader, latch) or None."""
    header = loop.header
    if header.attrs.get('nounroll'):
        return None
    outside = [p for p in preds[header] if p not in loop.blocks]
    if len(outside) != 1 or len(loop.latches) != 1:
        return None
    pre, latch = outside[0], loop.latches[0]
    best = None
    for b in loop.blocks:
        t = b.term
        if t.kind != 'br' or not dom.dominates(b, latch):
            continue
        inside = [x in loop.blocks for x in t.targets]
        if inside.count(False) != 1:
            continue
        for c, exit_when in conditions(t.args[0], not inside[0]):
            if c.args[1].const is None:
                continue
            x, bound, pred = c.args[0], c.args[1].const, c.attrs['pred']
            width = x.ty.bits
            stepped = 0
            if x.op.kind == 'add' and x.op.args[1].const is not None and x.op.args[0].op.kind == 'phi':
                phi, stepped = x.op.args[0].op, 1
            elif x.op.kind == 'phi':
                phi = x.op
            else:
                continue
            if phi.block is not header:
                continue
            iv = induction(phi, pre, latch)
            if iv is None or stepped and x is not phi.incoming(latch):
                continue
            start, step = iv
            for i in range(MAX_COPIES + 1):
                value = (start + step * (i + stepped)) & x.ty.mask
                if compare(pred, value, bound, width) == exit_when:
                    if best is None or i < best:
                        best = i
                    break
    if best is None:
        return None
    return best, pre, latch


def unroll_loop(fn, loop, exit_at, pre, latch):
    order = [b for b in fn.rpo() if b in loop.blocks]
    header = loop.header
    copies = []
    unreachable = fn.block('unrolled.end')
    unreachable.set_term(fn.new('unreachable'))
    prev_vm = None
    for i in range(exit_at + 1):
        bm = {b: fn.block('%s.u%d' % (b.name, i)) for b in order}
        vm = {}
        for phi in header.phis():
            v = phi.incoming(pre) if i == 0 else phi.incoming(latch)
            vm[phi.result] = v if i == 0 else prev_vm.get(v, v)
        pending = []
        for b in order:
            nb = bm[b]
            for op in b.ops:
                if op.kind == 'phi':
                    if b is header:
                        continue
                    new = fn.phi(op.ty)
                    nb.append(new)
                    pending.append((op, new))
                else:
                    new = fn.new(op.kind, [vm.get(a, a) for a in op.args], tys=[r.ty for r in op.results], attrs=dict(op.attrs))
                    nb.append(new)
                for r, nr in zip(op.results, new.results):
                    vm[r] = nr
        for old, new in pending:
            for p, a in zip(old.preds, old.args):
                new.add_incoming(bm[p], vm.get(a, a))
        copies.append((bm, vm))
        prev_vm = vm
    for i, (bm, vm) in enumerate(copies):
        for b in order:
            t = b.term
            new = fn.new(t.kind, [vm.get(a, a) for a in t.args], attrs=dict(t.attrs))
            targets = []
            for x in t.targets:
                if x is header:
                    targets.append(copies[i + 1][0][header] if i + 1 < len(copies) else unreachable)
                elif x in loop.blocks:
                    targets.append(bm[x])
                else:
                    targets.append(x)
                    for phi in x.phis():
                        v = phi.incoming(b)
                        phi.add_incoming(bm[b], vm.get(v, v))
            new.targets = targets
            bm[b].set_term(new)
    pre.term.targets = [copies[0][0][header] if x is header else x for x in pre.term.targets]
    # the originals: no edges out, then the uses of their values after the loop repaired
    for b in order:
        for x in b.term.targets:
            if x not in loop.blocks:
                for phi in x.phis():
                    if b in phi.preds:
                        phi.remove_incoming(b)
        b.term.detach()
        b.term.targets = []
    for b in order:
        for phi in b.phis():
            phi.detach()
    preds = fn.preds()
    for b in order:
        for op in b.ops:
            for r in op.results:
                outside = [u for u in r.uses if u.block is not None and u.block not in loop.blocks]
                if not outside:
                    continue
                up = SSAUpdater(fn, r.ty, preds)
                for bm, vm in copies:
                    up.add(bm[b], vm[r])
                up.rewrite(r, set(loop.blocks))
    for b in order:
        for op in b.ops:
            op.detach()
        fn.blocks.remove(b)
    fn.remove_unreachable()


def unroll(fn):
    changed = False
    for _ in range(64):
        dom = Dominators(fn)
        info = find_loops(fn, dom)
        preds = fn.preds()
        done = False
        for loop in reversed(info.loops):      # innermost first
            if loop.children:
                continue
            tc = trip_count(fn, loop, preds, dom)
            if tc is None:
                continue
            exit_at, pre, latch = tc
            size = sum(len(b.ops) + 1 for b in loop.blocks)
            if exit_at + 1 > MAX_COPIES or size * (exit_at + 1) > MAX_OPS:
                continue
            unroll_loop(fn, loop, exit_at, pre, latch)
            changed = done = True
            break
        if not done:
            return changed
    return changed
