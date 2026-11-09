"""Recognize dispatch entry points in source SSA, before machine encoding."""


def original(fn, v):
    op = fn.defs[v]
    while op.kind == 'cast' and op.args[0].type.bits == v.type.bits:
        v = op.args[0]
        op = fn.defs[v]
    return v


def address(fn, v):
    op = fn.defs[original(fn, v)]
    if op.kind == 'symbol':
        return op.attrs['name'], 0
    if op.kind == 'ptradd':
        a = address(fn, op.args[0])
        b = fn.defs[op.args[1]]
        if a is not None and b.kind == 'constant':
            return a[0], a[1] + b.attrs['value']
    return None


def trampolines(functions):
    out = {}
    for fn in functions:
        if len(fn.blocks) != 1 or 'static' in fn.decl.storage:
            continue
        b = fn.entry
        calls, loads = [o for o in b.ops if o.kind == 'call'], [o for o in b.ops if o.kind == 'load']
        if len(calls) != 1 or len(loads) != 1 or any(o.kind not in ('call', 'load', 'symbol', 'ptradd', 'cast') for o in b.ops):
            continue
        call, load = calls[0], loads[0]
        if original(fn, call.args[0]) != load.results[0] or load.attrs.get('volatile'):
            continue
        if tuple(original(fn, v) for v in call.args[1:]) != tuple(fn.params):
            continue
        if b.term.kind != 'ret' or tuple(original(fn, v) for v in b.term.args) != call.results:
            continue
        target = address(fn, load.args[0])
        if target is not None:
            out[fn.name] = target
    return out
