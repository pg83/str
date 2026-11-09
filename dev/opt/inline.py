"""Inlining: a callee's body copied into the caller at the call.

Callees are inlined bottom-up, each already optimized, so that a caller sees
what its callees came to. always_inline functions always, noinline never;
the rest by size: a function declared inline up to INLINE_COST ops, a static
one up to STATIC_COST, the last call of a static function whatever its size,
an exported one up to EXTERN_COST."""
from .ir import Op

INLINE_COST = 150      # declared inline (common.h's helpers)
STATIC_COST = 60
EXTERN_COST = 30


def cost(fn):
    return sum(1 for b in fn.blocks for op in b.ops if op.kind not in ('phi', 'alloca')) + len(fn.blocks)


def callee_of(op, module):
    c = op.args[0]
    if c.op.kind == 'sym':
        return module.functions.get(c.op.attrs['name'])
    return None


def map_value(caller, v, vm):
    m = vm.get(v)
    if m is not None:
        return m
    k = v.op.kind
    if k == 'const':
        return caller.const(v.ty, v.op.attrs['value'])
    if k == 'sym':
        return caller.sym(v.op.attrs['name'])
    if k == 'undef':
        return caller.undef(v.ty)
    raise ValueError('inline: %s of %s has no copy' % (v, k))


def inline_call(caller, callee, call):
    b = call.block
    at = b.ops.index(call)
    cont = caller.block(b.name + '.after')
    for op in b.ops[at + 1:]:
        cont.append(op)
    del b.ops[at:]
    call.block = None
    term = b.term
    b.term = None
    term.block = cont
    cont.term = term
    for t in term.targets:
        for phi in t.phis():
            phi.preds = [cont if p is b else p for p in phi.preds]
    vm = dict(zip(callee.params, call.args[1:]))
    order = callee.rpo()
    blocks = {cb: caller.block(cb.name) for cb in order}
    returns = []
    phis = []
    entry_allocas = [op for op in caller.entry.ops if op.kind == 'alloca']
    alloca_at = len(entry_allocas)
    for cb in order:
        nb = blocks[cb]
        for op in cb.ops:
            if op.kind == 'phi':
                new = caller.phi(op.ty)
                nb.append(new)
                phis.append((op, new))
            else:
                new = caller.new(op.kind, [map_value(caller, a, vm) for a in op.args], tys=[r.ty for r in op.results],
                                 attrs=dict(op.attrs))
                if op.kind == 'alloca':
                    caller.entry.insert(alloca_at, new)
                    alloca_at += 1
                else:
                    nb.append(new)
            for r, nr in zip(op.results, new.results):
                vm[r] = nr
    for old, new in phis:
        for p, a in zip(old.preds, old.args):
            if p in blocks:
                new.add_incoming(blocks[p], map_value(caller, a, vm))
    for cb in order:
        nb, t = blocks[cb], cb.term
        if t.kind == 'ret':
            op = caller.new('jmp')
            op.targets = [cont]
            nb.set_term(op)
            if t.args:
                returns.append((nb, map_value(caller, t.args[0], vm)))
        else:
            op = caller.new(t.kind, [map_value(caller, a, vm) for a in t.args], attrs=dict(t.attrs))
            op.targets = [blocks[x] for x in t.targets]
            nb.set_term(op)
    jmp = caller.new('jmp')
    jmp.targets = [blocks[callee.entry]]
    b.set_term(jmp)
    if call.result is not None:
        if not returns:
            value = caller.undef(call.result.ty)
        elif len(returns) == 1:
            value = returns[0][1]
        else:
            phi = caller.phi(call.result.ty)
            for nb, v in returns:
                phi.add_incoming(nb, v)
            cont.insert(0, phi)
            value = phi.result
        caller.replace_all_uses(call.result, value)
    call.detach()
    caller.remove_unreachable()


def inline_calls(module, fn, callers_left):
    """Inline what the policy allows into fn; callers_left: {callee name: calls not yet inlined}."""
    changed = False
    while True:
        site = None
        for b in fn.blocks:
            for op in b.ops:
                if op.kind != 'call':
                    continue
                callee = callee_of(op, module)
                if callee is None or callee is fn or callee.attrs.get('noinline') or not callee.blocks:
                    continue
                c = cost(callee)
                static = callee.attrs.get('static')
                if (callee.attrs.get('always_inline')
                        or static and callers_left.get(callee.name, 0) == 1 and not callee.attrs.get('address_taken')
                        or callee.attrs.get('inline') and c <= INLINE_COST
                        or static and c <= STATIC_COST
                        or c <= EXTERN_COST):
                    site = op, callee
                    break
            if site:
                break
        if site is None:
            return changed
        op, callee = site
        callers_left[callee.name] = callers_left.get(callee.name, 1) - 1
        inline_call(fn, callee, op)
        changed = True


def call_graph(module):
    calls = {}
    for fn in module.functions.values():
        out = []
        for b in fn.blocks:
            for op in b.ops:
                if op.kind == 'call':
                    c = callee_of(op, module)
                    if c is not None:
                        out.append(c.name)
        calls[fn.name] = out
    return calls


def bottom_up(module):
    """Function names, every callee before its callers (cycles broken anywhere)."""
    calls = call_graph(module)
    order, state = [], {}

    def visit(name):
        stack = [(name, iter(calls.get(name, ())))]
        state[name] = 1
        while stack:
            n, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                order.append(n)
                state[n] = 2
                stack.pop()
            elif nxt not in state:
                state[nxt] = 1
                stack.append((nxt, iter(calls.get(nxt, ()))))

    for name in module.functions:
        if name not in state:
            visit(name)
    return order
