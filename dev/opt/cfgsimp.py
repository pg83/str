"""Control-flow simplification: branches on constants folded, unreachable
blocks removed, a block merged into its only predecessor, empty blocks
bypassed, and jumps threaded: a block whose branch is decided by the values
a predecessor brings (a phi of constants after inlining, say) is copied for
that predecessor, the copy jumping straight to the known target."""
from .ir import Op, Dominators, I1, INVERT
from .common import fold_conditions, hoist_common
from .cvp import bound
from .facts import Facts
from .simplify import fold_int, compare, decide
from .ssaupdate import repair


def to_jmp(fn, block, target):
    """Replace block's terminator with a jump; the edges dropped lose their phi entries."""
    old = block.term
    for t in old.targets:
        if t is not target:
            for phi in t.phis():
                if block in phi.preds:
                    phi.remove_incoming(block)
    op = fn.new('jmp')
    op.targets = [target]
    block.set_term(op)


def fold_branches(fn):
    changed = False
    for b in fn.blocks:
        t = b.term
        if t.kind == 'br':
            c = t.args[0].const
            if c is not None:
                to_jmp(fn, b, t.targets[0] if c else t.targets[1])
                changed = True
    return changed


def merge_blocks(fn):
    """B -> S where S has no other predecessor: S's ops move into B."""
    changed = False
    preds = fn.preds()
    for b in list(fn.blocks):
        if b.term is None or b.term.kind != 'jmp':
            continue
        s = b.term.targets[0]
        if s is b or s is fn.entry or len(preds[s]) != 1 or b not in fn.blocks:
            continue
        for phi in s.phis():
            fn.replace_all_uses(phi.result, phi.args[0])
            phi.detach()
            s.remove(phi)
        for op in list(s.ops):
            s.ops.remove(op)
            b.append(op)
        term = s.term
        s.term = None
        b.term.detach()
        term.block = b
        b.term = term
        for t in term.targets:
            for phi in t.phis():
                phi.preds = [b if p is s else p for p in phi.preds]
        fn.blocks.remove(s)
        preds = fn.preds()
        if 'nounroll' in s.attrs:
            b.attrs['nounroll'] = True
        changed = True
    return changed


def skip_empty(fn):
    """An empty block that only jumps on: its predecessors jump there directly."""
    changed = False
    preds = fn.preds()
    for e in list(fn.blocks):
        if e is fn.entry or e.ops or e.term.kind != 'jmp':
            continue
        t = e.term.targets[0]
        if t is e:
            continue
        for p in list(preds[e]):
            if p.term.kind == 'br' and t in p.term.targets:
                # p reaches t both ways: fine only when t's phis agree
                if any(phi.incoming(p) is not phi.incoming(e) for phi in t.phis()):
                    continue
                to_jmp(fn, p, t)
                changed = True
                continue
            if p in preds[t] and any(phi.incoming(p) is not phi.incoming(e) for phi in t.phis()):
                continue
            p.term.targets = [t if x is e else x for x in p.term.targets]
            if p not in preds[t]:
                for phi in t.phis():
                    phi.add_incoming(p, phi.incoming(e))
            changed = True
            preds = fn.preds()
        preds = fn.preds()
        if not preds[e]:
            for phi in t.phis():
                if e in phi.preds:
                    phi.remove_incoming(e)
            e.term.detach()
            fn.blocks.remove(e)
            preds = fn.preds()
    return changed


def evaluate(op, env):
    """The constant result of a pure op given constants for some values, or None."""
    def val(v):
        c = env.get(v)
        return c if c is not None else v.const
    args = [val(a) for a in op.args]
    if any(a is None for a in args):
        return None
    k = op.kind
    if k == 'icmp':
        return int(compare(op.attrs['pred'], args[0], args[1], op.args[0].ty.bits))
    if k in ('add', 'sub', 'mul', 'and', 'or', 'xor', 'shl', 'lshr', 'ashr', 'udiv', 'urem', 'sdiv', 'srem'):
        if op.ty.kind != 'i':
            return None
        return fold_int(k, args[0], args[1], op.ty)
    if k in ('zext', 'trunc'):
        return args[0] & op.ty.mask
    if k == 'select':
        return args[1] if args[0] else args[2]
    return None


def edge_range(facts, v, p, b):
    """The range of v on the edge p -> b: its own, narrowed by p's branch when that compares v with a constant."""
    lo, hi = facts.range(v)
    t = p.term
    if t.kind == 'br' and t.targets[0] is not t.targets[1] and v.ty.kind in ('i', 'p'):
        c = t.args[0].op
        true_edge = t.targets[0] is b
        if c.kind == 'icmp' and c.args[0] is v and c.args[1].const is not None:
            pred = c.attrs['pred'] if true_edge else INVERT[c.attrs['pred']]
            if pred in ('slt', 'sle', 'sgt', 'sge'):
                half = 1 << (v.ty.bits - 1)
                pred = 'u' + pred[1:] if hi < half and c.args[1].const < half else None
            interval = bound(pred, c.args[1].const, v.ty.bits) if pred else None
            if interval is not None:
                lo, hi = max(lo, interval[0]), min(hi, interval[1])
        elif t.args[0] is v:
            lo, hi = (1, 1) if true_edge else (0, 0)
    return (lo, hi) if lo <= hi else facts.range(v)


def thread(fn, limit=8):
    """Copy a small block for each predecessor that decides its branch."""
    changed = False
    dom = Dominators(fn)
    preds = fn.preds()
    for b in list(fn.blocks):
        if b is fn.entry or b.term.kind != 'br' or len(preds[b]) < 2 or b not in fn.blocks:
            continue
        if any(dom.dominates(b, p) for p in preds[b]):
            continue   # a loop header: threading would make the loop irreducible
        body = [op for op in b.ops if op.kind != 'phi']
        if len(body) > limit or any(op.effects or op.kind in ('load', 'vop', 'alloca') for op in body):
            continue
        phis = b.phis()
        base = Facts(fn)
        cond = b.term.args[0]
        for p in list(preds[b]):
            env = {phi.result: phi.incoming(p).const for phi in phis if phi.incoming(p).const is not None}
            for phi in phis:
                v = phi.incoming(p)
                if v.const is None and v.ty.kind == 'i':
                    lo, hi = edge_range(base, v, p, b)
                    if lo == hi:
                        env[phi.result] = lo       # p's own branch fixes the value it passes
            c = None
            if env:
                for op in body:
                    v = evaluate(op, env)
                    if v is not None:
                        env[op.result] = v
                c = env.get(cond, cond.const)
            if c is None and cond.op.kind == 'icmp' and phis:
                # what the incoming values' ranges decide, the predecessor's own branch included
                facts = Facts(fn, {phi.result: edge_range(base, phi.incoming(p), p, b) for phi in phis})
                x, y = cond.op.args
                pred = cond.op.attrs['pred']
                lo1, hi1 = facts.range(x)
                lo2, hi2 = facts.range(y)
                half = 1 << (x.ty.bits - 1)
                if pred in ('slt', 'sle', 'sgt', 'sge') and hi1 < half and hi2 < half:
                    pred = 'u' + pred[1:]
                verdict = decide(pred, lo1, hi1, lo2, hi2)
                c = None if verdict is None else int(verdict)
            if c is None:
                continue
            target = b.term.targets[0] if c else b.term.targets[1]
            if target is b:
                continue
            if p in preds[target]:
                continue
            # the copy of b for p
            copy = fn.block(b.name + '.thread')
            mapping = {phi.result: phi.incoming(p) for phi in phis}
            for op in body:
                new = fn.new(op.kind, [mapping.get(a, a) for a in op.args], op.ty, dict(op.attrs))
                copy.append(new)
                mapping[op.result] = new.result
            jmp = fn.new('jmp')
            jmp.targets = [target]
            copy.set_term(jmp)
            p.term.targets = [copy if x is b else x for x in p.term.targets]
            for phi in phis:
                phi.remove_incoming(p)
            for phi in target.phis():
                v = phi.incoming(b)
                phi.add_incoming(copy, mapping.get(v, v))
            new_preds = fn.preds()
            defs = {}
            for original in [phi.result for phi in phis] + [op.result for op in body]:
                if any(u.block is not b for u in original.uses if u.block is not None):
                    defs[original] = [(b, original), (copy, mapping[original])]
            repair(fn, defs, new_preds)
            preds = fn.preds()
            dom = Dominators(fn)
            changed = True
            if len(preds[b]) < 2:
                break
    return changed


def split_returns(fn):
    """A return block that only joins values gets a copy per predecessor:
    each path returns its own value where it ends (no jump to a shared
    return, and a call that returns straight away becomes a tail call)."""
    changed = False
    preds = fn.preds()
    for b in list(fn.blocks):
        if b.term.kind != 'ret' or len(preds[b]) < 2 or any(op.kind != 'phi' for op in b.ops):
            continue
        phis = b.phis()
        value = b.term.args[0] if b.term.args else None
        if value is not None and value.op.kind == 'phi' and value.op.block is not b:
            continue
        if value is not None and value.op.block is b and any(u.block is not b for u in value.uses):
            continue
        for p in list(preds[b]):
            v = value
            if v is not None and v.op in phis:
                v = v.op.incoming(p)
            ret = fn.new('ret', [v] if v is not None else [])
            for phi in phis:
                phi.remove_incoming(p)
            if p.term.kind == 'jmp':
                p.set_term(ret)      # the path returns where it ends
            else:
                tail = fn.block('ret')
                tail.set_term(ret)
                p.term.targets = [tail if t is b else t for t in p.term.targets]
        changed = True
    if changed:
        fn.remove_unreachable()
    return changed


def simplify_cfg(fn):
    changed = False
    while True:
        step = fold_branches(fn)
        step = fn.remove_unreachable() or step
        step = merge_blocks(fn) or step
        step = skip_empty(fn) or step
        step = hoist_common(fn) or step
        step = fold_conditions(fn) or step
        if not step:
            step = thread(fn)
        if not step:
            return changed
        changed = True
