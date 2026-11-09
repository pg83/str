"""Code the paths share. The identical leading ops of a branch's two arms
are hoisted above the branch; the identical trailing ops of blocks that end
alike (the same return, a jump to the same block) are merged into one block
the paths jump to, with phis for the inputs that differ; and a cheap
condition that only picks between a block's other way and its own successor
is folded into that block's branch (c1 && c2 decided by one branch: the bit
tests of one word then combine, simplify.py, and what remains is split
back into jumps by the selection)."""
from .ir import I1, PURE, DIVISIONS, Dominators, find_loops
from .ifconv import speculable
from .ssaupdate import repair


# ------------------------------------------------------------------ hoist

def same_op(a, b):
    return (a.kind == b.kind and a.kind not in ('phi', 'alloca') and a.attrs == b.attrs
            and len(a.results) == len(b.results) and all(x.ty is y.ty for x, y in zip(a.results, b.results))
            and len(a.args) == len(b.args) and all(x is y for x, y in zip(a.args, b.args)))


def hoist_common(fn):
    """The leading ops both arms of a branch compute alike run before the branch."""
    changed = False
    preds = fn.preds()
    for b in fn.blocks:
        t = b.term
        if t.kind != 'br':
            continue
        s1, s2 = t.targets
        if s1 is s2 or preds[s1] != [b] or preds[s2] != [b]:
            continue
        while s1.ops and s2.ops and same_op(s1.ops[0], s2.ops[0]):
            o1, o2 = s1.ops[0], s2.ops[0]
            s1.remove(o1)
            b.append(o1)
            for r, s in zip(o1.results, o2.results):
                fn.replace_all_uses(s, r)
            o2.detach()
            s2.remove(o2)
            changed = True
    return changed


# ------------------------------------------------------------------ tails

def body(block):
    return [op for op in block.ops if op.kind != 'phi']


def common_tail(a, b, ends):
    """How many trailing ops of a and b are the same computation (n), and the
    pairs of inputs (x from a, y from b) that differ, given ends, the pairs
    the two blocks' ends use."""
    inputs = {}

    def match(x, y):
        if x is y:
            return x.op.block is not a and x.op.block is not b
        if x.ty is not y.ty or x.const is not None or y.const is not None or x.op.kind in ('sym', 'undef') or y.op.kind in ('sym', 'undef'):
            return False
        if recomputable(x, y, a, b, match):
            inputs[(x, y)] = 'recompute'      # the same expression on both sides: computed once, in the tail
            return True
        if address_only(x) or address_only(y):
            return False      # a phi would cost the address its folding
        inputs[(x, y)] = 'phi'
        return True

    if not all(match(x, y) for x, y in ends):
        return 0, {}
    ops_a, ops_b = body(a), body(b)
    n = 0
    while n < len(ops_a) and n < len(ops_b):
        oa, ob = ops_a[-1 - n], ops_b[-1 - n]
        if not (oa.kind == ob.kind and oa.kind != 'alloca' and oa.attrs == ob.attrs and len(oa.results) == len(ob.results)
                and all(r.ty is s.ty for r, s in zip(oa.results, ob.results)) and len(oa.args) == len(ob.args)):
            break
        # a result used by the later ops: the same position on both sides, or no match
        if any((x is r) != (y is s) for r, s in zip(oa.results, ob.results) for x, y in inputs):
            break
        trial = dict(inputs)
        if not all(match(x, y) for x, y in zip(oa.args, ob.args)):
            inputs = trial
            break
        for r, s in zip(oa.results, ob.results):
            inputs.pop((r, s), None)
        n += 1
    return n, inputs


def address_only(v):
    """A pointer sum that only feeds memory operands (folded into them, a lea as a phi)."""
    return v.op.kind in ('ptradd', 'i2p') and all(
        u.kind in ('load', 'store') and u.args[0] is v or u.kind == 'ptradd' or u.kind == 'vop' and u.attrs.get('memory') for u in v.uses)


def recomputable(x, y, a, b, match):
    """x (of block a) and y (of b) are the same pure expression, each used once,
    of operands that match in turn (the same value, or inputs of the tail)."""
    ox, oy = x.op, y.op
    return (ox.kind == oy.kind and ox.kind in PURE and ox.kind not in DIVISIONS and ox.kind != 'phi' and ox.attrs == oy.attrs
            and ox.block is a and oy.block is b and len(x.uses) == 1 and len(y.uses) == 1 and len(ox.results) == 1
            and len(ox.args) == len(oy.args) and all(match(p, q) for p, q in zip(ox.args, oy.args)))


def ends_of(block):
    """The values the block's end passes on, paired by position: the return's, or its successor's phi entries."""
    t = block.term
    if t.kind == 'ret':
        return list(t.args)
    return [phi.incoming(block) for phi in t.targets[0].phis()]


def merge_tail(fn, a, b, n, inputs):
    m = fn.block(a.name + '.tail')
    phis = {}
    for (x, y), how in inputs.items():
        if how == 'phi':
            phi = fn.phi(x.ty)
            m.append(phi)
            phi.add_incoming(a, x)
            phi.add_incoming(b, y)
            phis[(x, y)] = phi.result
    mapping = {}

    def value(x, y):
        if x is y:
            return x
        v = phis.get((x, y))
        if v is None and inputs.get((x, y)) == 'recompute':
            args = [value(p, q) for p, q in zip(x.op.args, y.op.args)]
            v = phis[(x, y)] = m.append(fn.new(x.op.kind, args, x.ty, dict(x.op.attrs))).result
        return v if v is not None else mapping[x]

    ops_a, ops_b = body(a)[len(body(a)) - n:], body(b)[len(body(b)) - n:]
    for oa, ob in zip(ops_a, ops_b):
        new = fn.new(oa.kind, [value(x, y) for x, y in zip(oa.args, ob.args)], tys=[r.ty for r in oa.results], attrs=dict(oa.attrs))
        m.append(new)
        for r, s, nr in zip(oa.results, ob.results, new.results):
            mapping[r] = nr
    t = a.term
    if t.kind == 'ret':
        m.set_term(fn.new('ret', [value(x, y) for x, y in zip(a.term.args, b.term.args)]))
    else:
        target = t.targets[0]
        for phi in target.phis():
            v = value(phi.incoming(a), phi.incoming(b))
            phi.remove_incoming(a)
            phi.remove_incoming(b)
            phi.add_incoming(m, v)
        jmp = fn.new('jmp')
        jmp.targets = [target]
        m.set_term(jmp)
    for block, ops in ((a, ops_a), (b, ops_b)):
        for op in reversed(ops):
            op.detach()
            block.remove(op)
        jmp = fn.new('jmp')
        jmp.targets = [m]
        block.set_term(jmp)


def canonical_order(block):
    """The block's ops in a dependence order (operands first, in argument
    order, from its end): the same computation then reads the same whatever
    order it was written in.  Only where nothing writes memory."""
    ops = body(block)
    if any(op.effects or op.kind == 'alloca' for op in ops):
        return
    here = set(ops)
    order, seen = [], set()

    def visit(op):
        if op in seen:
            return
        seen.add(op)
        for a in op.args:
            if a.op in here:
                visit(a.op)
        order.append(op)

    for a in block.term.args:
        if a.op in here:
            visit(a.op)
    if block.term.kind == 'jmp':
        for phi in block.term.targets[0].phis():
            a = phi.incoming(block)
            if a.op in here:
                visit(a.op)
    for op in ops:
        visit(op)
    block.ops = [op for op in block.ops if op.kind == 'phi'] + order


def merge_tails(fn, min_ops=2):
    """Blocks that end alike and compute the same last ops share one copy of them."""
    changed = False
    while True:
        groups = {}
        for b in fn.blocks:
            t = b.term
            if t.kind == 'ret':
                groups.setdefault('ret', []).append(b)
            elif t.kind == 'jmp' and t.targets[0] is not b and body(b):
                groups.setdefault(t.targets[0], []).append(b)
        for blocks in groups.values():
            if len(blocks) > 1:
                for b in blocks:
                    canonical_order(b)
        best = None
        for blocks in groups.values():
            for i, a in enumerate(blocks):
                ends_a = ends_of(a)
                for b in blocks[i + 1:]:
                    n, inputs = common_tail(a, b, list(zip(ends_a, ends_of(b))))
                    if n >= min_ops and n >= sum(1 for how in inputs.values() if how == 'phi') and (best is None or n > best[0]):
                        best = (n, inputs, a, b)
        if best is None:
            return changed
        n, inputs, a, b = best
        merge_tail(fn, a, b, n, inputs)
        changed = True


# ---------------------------------------------------------- duplication

def duplicate_tails(fn, max_ops=1):
    """A small block that joins values (phis) and decides or returns on them
    is copied into each predecessor that only jumps to it: the join and the
    jump go, the phis resolve to what each way brought."""
    changed = False
    preds = fn.preds()
    dom = Dominators(fn)
    for b in list(fn.blocks):
        if b is fn.entry or b not in fn.blocks or b.term.kind not in ('ret', 'br'):
            continue
        phis = b.phis()
        ops = body(b)
        if not phis or len(ops) > max_ops or any(op.effects or op.kind in ('load', 'alloca', 'call') for op in ops) or b in b.term.targets:
            continue
        if any(dom.dominates(b, p) for p in preds[b]):
            continue       # a loop header: its copy in the latch would make the loop irreducible
        defined = [phi.result for phi in phis] + [op.result for op in ops if op.result is not None]
        jumpers = [p for p in preds[b] if p.term.kind == 'jmp' and p is not b]
        if not jumpers or len(preds[b]) < 2:
            continue
        copies = {}
        for p in jumpers:
            mapping = {phi.result: phi.incoming(p) for phi in phis}
            copies[p] = mapping
            for op in ops:
                new = fn.new(op.kind, [mapping.get(a, a) for a in op.args], tys=[r.ty for r in op.results], attrs=dict(op.attrs))
                p.append(new)
                for r, nr in zip(op.results, new.results):
                    mapping[r] = nr
            t = fn.new(b.term.kind, [mapping.get(a, a) for a in b.term.args], attrs=dict(b.term.attrs))
            t.targets = list(b.term.targets)
            for phi in phis:
                phi.remove_incoming(p)
            p.set_term(t)
            for s in b.term.targets:
                for phi in s.phis():
                    v = phi.incoming(b)
                    phi.add_incoming(p, mapping.get(v, v))
        preds = fn.preds()
        gone = not preds[b]
        if gone:
            # out of the graph before the repair looks at the predecessors
            for s in b.term.targets:
                for phi in s.phis():
                    if b in phi.preds:
                        phi.remove_incoming(b)
            b.term.targets = []
            preds = fn.preds()
        # the values the block defined, used further on: the copy that reaches each use
        defs = {}
        for v in defined:
            # (the successors' phi entries were given each copy's value above)
            if any(u.block is not b and not (u.kind == 'phi' and u.block in b.term.targets) for u in v.uses):
                defs[v] = [(p, copies[p][v]) for p in jumpers] + ([] if gone else [(b, v)])
        if defs:
            repair(fn, defs, preds)
        if gone:
            for op in reversed(b.ops):
                op.detach()
            b.term.detach()
            fn.blocks.remove(b)
            preds = fn.preds()
        dom = Dominators(fn)
        changed = True
    return changed


# --------------------------------------------------------------- rotation

def rotate_loops(fn, max_ops=4):
    """A loop that tests its exit at the header (while) is made to test it at
    its latch (do-while): the header's compare is copied into the preheader,
    which then guards the first iteration, and into the latch, which then
    branches back itself; the entry falls into the body (no jump to a header
    laid out below), the body's first block takes the phis."""
    changed = False
    for _ in range(16):
        dom = Dominators(fn)
        info = find_loops(fn, dom)
        preds = fn.preds()
        done = False
        for loop in info.loops:
            h = loop.header
            if h.term.kind != 'br' or h.attrs.get('rotated') or len(preds[h]) != 2 or len(loop.latches) != 1:
                continue
            outside = [p for p in preds[h] if p not in loop.blocks]
            if len(outside) != 1:
                continue
            pre, latch = outside[0], loop.latches[0]
            if pre.term.kind != 'jmp' or latch.term.kind != 'jmp' or latch is h or h.term.targets[0] is h.term.targets[1]:
                continue
            inside = [x in loop.blocks for x in h.term.targets]
            if inside.count(True) != 1:
                continue
            ops = body(h)
            if len(ops) > max_ops or not all(speculable(op) for op in ops):
                continue
            copies = {}
            for p in (pre, latch):
                mapping = {phi.result: phi.incoming(p) for phi in h.phis()}
                for op in ops:
                    new = fn.new(op.kind, [mapping.get(a, a) for a in op.args], tys=[r.ty for r in op.results], attrs=dict(op.attrs))
                    p.append(new)
                    for r, nr in zip(op.results, new.results):
                        mapping[r] = nr
                t = fn.new('br', [mapping.get(h.term.args[0], h.term.args[0])], attrs=dict(h.term.attrs))
                t.targets = list(h.term.targets)
                p.set_term(t)
                for s in h.term.targets:
                    for phi in s.phis():
                        v = phi.incoming(h)
                        phi.add_incoming(p, mapping.get(v, v))
                copies[p] = mapping
            for s in h.term.targets:
                for phi in s.phis():
                    if h in phi.preds:
                        phi.remove_incoming(h)
            entry = h.term.targets[inside.index(True)]
            h.term.targets = []
            preds = fn.preds()
            inner = loop.blocks - {h}
            outside_defs = {}
            for v in [phi.result for phi in h.phis()] + [op.result for op in ops if op.result is not None]:
                users = [u for u in v.uses if u.block is not h]
                if any(u.block in inner for u in users):
                    # inside the loop the value is the phi at its new head (a use in the latch
                    # itself reads the start of the block, which the updater does not model)
                    phi = fn.phi(v.ty)
                    entry.insert(0, phi)
                    phi.add_incoming(pre, copies[pre][v])
                    phi.add_incoming(latch, copies[latch][v])
                    for u in users:
                        if u.block in inner:
                            u.replace_arg(v, phi.result)      # (at the end of any block of the loop the value is this iteration's)
                if any(u.block not in inner for u in v.uses if u.block is not h):
                    outside_defs[v] = [(pre, copies[pre][v]), (latch, copies[latch][v])]
            if outside_defs:
                repair(fn, outside_defs, preds)
            for op in reversed(h.ops):
                op.detach()
            h.term.detach()
            fn.blocks.remove(h)
            entry.attrs['rotated'] = True
            changed = done = True
            break
        if not done:
            break
    return changed


# ------------------------------------------------------------- conditions

def probability(term, index):
    w = term.attrs.get('weights')
    return None if not w else w[index] / (w[0] + w[1])


def fold_conditions(fn, budget=2):
    """b: br c1 -> s, other; s (reached from b alone): a few pure ops, br c2 ->
    cont, other.  The two branches become one in b on c1 && c2 (with the
    negations the directions ask for); s's ops run in b."""
    changed = False
    preds = fn.preds()
    for b in list(fn.blocks):
        if b not in fn.blocks or b.term.kind != 'br':
            continue
        for side in (0, 1):
            s, other = b.term.targets[side], b.term.targets[1 - side]
            if s is other or s is b or preds[s] != [b] or s.term.kind != 'br' or s.phis():
                continue
            if other not in s.term.targets or s.term.targets[0] is s.term.targets[1]:
                continue
            if len(s.ops) > budget + 1 or not all(speculable(op) for op in s.ops):
                continue
            if any(phi.incoming(b) is not phi.incoming(s) for phi in other.phis()):
                continue
            s_side = s.term.targets.index(other)
            cont = s.term.targets[1 - s_side]
            if cont is b:
                continue
            c1, c2 = b.term.args[0], s.term.args[0]
            for op in list(s.ops):
                s.remove(op)
                b.append(op)
            one = fn.const(I1, 1)
            if side == 1:
                c1 = b.append(fn.new('xor', [c1, one], I1)).result
            if s_side == 0:
                c2 = b.append(fn.new('xor', [c2, one], I1)).result
            cond = b.append(fn.new('and', [c1, c2], I1)).result
            br = fn.new('br', [cond])
            br.targets = [cont, other]
            p1, p2 = probability(b.term, side), probability(s.term, 1 - s_side)
            if p1 is not None or p2 is not None:
                p = (p1 if p1 is not None else 1) * (p2 if p2 is not None else 1)
                br.attrs['weights'] = (int(p * 2000) + 1, int((1 - p) * 2000) + 1)
            if b.term.attrs.get('unpredictable') or s.term.attrs.get('unpredictable'):
                br.attrs['unpredictable'] = True
            b.set_term(br)
            for phi in cont.phis():
                phi.preds = [b if p is s else p for p in phi.preds]
            for phi in other.phis():
                if s in phi.preds:
                    phi.remove_incoming(s)
            s.term.detach()
            s.term = None
            fn.blocks.remove(s)
            preds = fn.preds()
            changed = True
            break
    return changed
