"""Full unrolling of loops that run a constant number of times.

A loop qualifies when a branch it takes on every iteration (its block
dominates the latch) leaves on a comparison of an induction variable
(a header phi from a constant start, stepped by a constant) with a constant:
the iteration that leaves is found by running the comparison. The loop is
then copied once per iteration, iteration i's latch jumping to iteration
i + 1's header; the copies' comparisons fold to constants afterwards and the
paths they rule out disappear. Data-dependent exits stay in each copy. Values
the loop defines and code after it uses get the copy that reaches the use
(ssaupdate.py).

A small loop whose count is not known (a header test of an induction
variable against a bound from outside, no other exit) is unrolled four
times at run time: four copies in a row run while four more iterations fit,
the loop itself takes the rest. A loop with a known count too large to copy
out whole runs eight (or four) iterations per round when the count divides,
the count test once per round, its other exits in every copy."""
from .ir import Dominators, find_loops, I1
from .simplify import compare
from .ssaupdate import SSAUpdater

MAX_COPIES = 16
MAX_OPS = 600
MAX_TRIP = 4096       # counts looked for (beyond MAX_COPIES: partial unrolling)


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
            for i in range(MAX_TRIP + 1):
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


RUNTIME_UNROLL = 4
RUNTIME_MAX_OPS = 16


def runtime_candidate(fn, loop, preds):
    """A small loop testing an induction variable against a bound known
    outside it at its header, leaving nowhere else: (pre, latch, phi, step,
    bound, body target) or None."""
    header = loop.header
    if header.attrs.get('nounroll') or header.attrs.get('unrolled'):
        return None
    outside = [p for p in preds[header] if p not in loop.blocks]
    if len(outside) != 1 or len(loop.latches) != 1:
        return None
    pre, latch = outside[0], loop.latches[0]
    if pre.term.kind != 'jmp':
        return None
    t = header.term
    if t.kind != 'br' or t.targets[0] is t.targets[1]:
        return None
    inside = [x in loop.blocks for x in t.targets]
    if inside != [True, False]:
        return None
    for b in loop.blocks:
        if b is not header and any(x not in loop.blocks for x in b.term.targets):
            return None
        if any(op.kind == 'call' for op in b.ops):
            return None
    if sum(len(b.ops) for b in loop.blocks) > RUNTIME_MAX_OPS:
        return None
    c = t.args[0].op
    if c.kind != 'icmp' or c.attrs['pred'] != 'ult':
        return None
    i, n = c.args
    if i.op.kind != 'phi' or i.op.block is not header:
        return None
    if n.const is None and n.op.block in loop.blocks:
        return None
    nxt = i.op.incoming(latch)
    if nxt.op.kind != 'add' or nxt.op.args[0] is not i or nxt.op.args[1].const is None or nxt.op.args[1].const == 0:
        return None
    return pre, latch, i.op, nxt.op.args[1].const, n, t.targets[0]


def unroll_runtime(fn, loop, pre, latch, phi, step, bound, body, count=RUNTIME_UNROLL):
    """The loop copied count times in a row, entered while count more
    iterations fit (i + step * (count - 1) < bound); the original loop takes
    what is left. The copies' headers jump straight into their bodies."""
    header = loop.header
    order = [b for b in fn.rpo() if b in loop.blocks]
    main = fn.block(header.name + '.main')
    main.attrs['unrolled'] = True
    header.attrs['unrolled'] = True
    main_phis = {}
    for old in header.phis():
        np = fn.phi(old.ty)
        main.append(np)
        np.add_incoming(pre, old.incoming(pre))
        main_phis[old] = np
    copies = []
    prev_vm = None
    for k in range(count):
        bm = {b: (main if k == 0 and b is header else fn.block('%s.u%d' % (b.name, k))) for b in order}
        vm = {}
        for old in header.phis():
            vm[old.result] = main_phis[old].result if k == 0 else prev_vm.get(old.incoming(latch), old.incoming(latch))
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
    for k, (bm, vm) in enumerate(copies):
        for b in order:
            t = b.term
            if b is header:
                if k == 0:
                    ahead = fn.new('add', [main_phis[phi].result, fn.const(phi.ty, step * (count - 1))], phi.ty)
                    main.append(ahead)
                    cond = fn.new('icmp', [ahead.result, bound], I1, {'pred': 'ult'})
                    main.append(cond)
                    new = fn.new('br', [cond.result])
                    new.targets = [bm[body], header]
                else:
                    new = fn.new('jmp')
                    new.targets = [bm[body]]
            else:
                new = fn.new(t.kind, [vm.get(a, a) for a in t.args], attrs=dict(t.attrs))
                new.targets = [(copies[k + 1][0][header] if k + 1 < count else main) if x is header else bm[x] for x in t.targets]
            bm[b].set_term(new)
    last_bm, last_vm = copies[-1]
    for old, np in main_phis.items():
        v = old.incoming(latch)
        np.add_incoming(last_bm[latch], last_vm.get(v, v))
        old.add_incoming(main, np.result)
        old.remove_incoming(pre)
    pre.term.targets = [main if x is header else x for x in pre.term.targets]


PARTIAL_MAX_OPS = 120


def unroll_partial(fn, loop, pre, latch, count):
    """count copies of the loop in a row as the loop: the header's count test
    stays in the first copy alone (the count is a multiple of count), the
    data-dependent exits stay in every copy."""
    header = loop.header
    order = [b for b in fn.rpo() if b in loop.blocks]
    body = next(x for x in header.term.targets if x in loop.blocks)
    copies = []
    main_phis = {}
    prev_vm = None
    for k in range(count):
        bm = {b: fn.block('%s.p%d' % (b.name, k)) for b in order}
        vm = {}
        for phi in header.phis():
            if k == 0:
                np = fn.phi(phi.ty)
                bm[header].append(np)
                np.add_incoming(pre, phi.incoming(pre))
                main_phis[phi] = np
                vm[phi.result] = np.result
            else:
                v = phi.incoming(latch)
                vm[phi.result] = prev_vm.get(v, v)
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
    for k, (bm, vm) in enumerate(copies):
        for b in order:
            t = b.term
            if b is header and k > 0:
                new = fn.new('jmp')
                new.targets = [bm[body]]
            else:
                new = fn.new(t.kind, [vm.get(a, a) for a in t.args], attrs=dict(t.attrs))
                targets = []
                for x in t.targets:
                    if x is header:
                        targets.append(copies[k + 1][0][header] if k + 1 < count else copies[0][0][header])
                    elif x in loop.blocks:
                        targets.append(bm[x])
                    else:
                        targets.append(x)
                        for phi in x.phis():
                            v = phi.incoming(b)
                            phi.add_incoming(bm[b], vm.get(v, v))
                new.targets = targets
            bm[b].set_term(new)
    last_bm, last_vm = copies[-1]
    for phi, np in main_phis.items():
        v = phi.incoming(latch)
        np.add_incoming(last_bm[latch], last_vm.get(v, v))
    copies[0][0][header].attrs['nounroll'] = True
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
                cand = runtime_candidate(fn, loop, preds)
                if cand is None:
                    continue
                pre, latch, phi, step, bound, body = cand
                unroll_runtime(fn, loop, pre, latch, phi, step, bound, body)
                changed = done = True
                break
            exit_at, pre, latch = tc
            size = sum(len(b.ops) + 1 for b in loop.blocks)
            if exit_at + 1 > MAX_COPIES or size * (exit_at + 1) > MAX_OPS:
                # too many iterations for copies of each: several per round when the count divides
                header = loop.header
                counted = header.term.kind == 'br' and sum(x in loop.blocks for x in header.term.targets) == 1
                for count in (8, 4):
                    if counted and exit_at % count == 0 and exit_at > count and size * count <= PARTIAL_MAX_OPS:
                        unroll_partial(fn, loop, pre, latch, count)
                        changed = done = True
                        break
                if done:
                    break
                continue
            unroll_loop(fn, loop, exit_at, pre, latch)
            changed = done = True
            break
        if not done:
            return changed
    return changed
