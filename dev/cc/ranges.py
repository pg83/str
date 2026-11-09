"""Conservative integer intervals along CFG edges."""
from collections import deque

from .ir import Term


def domain(value):
    bits = value.type.bits
    return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if value.type.signed else (0, (1 << bits) - 1)


def propagate_ranges(fn):
    def canonical(value):
        op = fn.defs[value]
        while (op.kind == 'cast' and value.type.bits == op.args[0].type.bits
               and value.type.signed == op.args[0].type.signed):
            value = op.args[0]
            op = fn.defs[value]
        return value

    def interval(value, facts, seen=frozenset()):
        value = canonical(value)
        full = domain(value)
        if value in facts:
            return facts[value]
        if value in seen:
            return full
        seen = seen | {value}
        op = fn.defs[value]
        bounds = lambda v: interval(v, facts, seen)
        result = full
        if op.kind == 'constant' and isinstance(op.attrs['value'], int):
            result = op.attrs['value'], op.attrs['value']
        elif op.kind == 'cast':
            result = bounds(op.args[0])
        elif op.kind in ('add', 'sub', 'mul', 'ptradd'):
            a, c = bounds(op.args[0]), bounds(op.args[1])
            if op.kind in ('add', 'ptradd'):
                result = a[0] + c[0], a[1] + c[1]
            elif op.kind == 'sub':
                result = a[0] - c[1], a[1] - c[0]
            else:
                products = [x * y for x in a for y in c]
                result = min(products), max(products)
        elif op.kind == 'and':
            a, c = bounds(op.args[0]), bounds(op.args[1])
            if c[0] == c[1] and c[0] >= 0:
                result = 0, c[0]
            elif a[0] == a[1] and a[0] >= 0:
                result = 0, a[0]
        elif op.kind in ('shl', 'shr'):
            a, c = bounds(op.args[0]), bounds(op.args[1])
            if c[0] == c[1] and 0 <= c[0] < value.type.bits:
                n = c[0]
                result = (a[0] << n, a[1] << n) if op.kind == 'shl' else (a[0] >> n, a[1] >> n)
        elif op.kind in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            a, c = bounds(op.args[0]), bounds(op.args[1])
            yes = {'eq': a[0] == a[1] == c[0] == c[1], 'ne': a[1] < c[0] or c[1] < a[0],
                   'lt': a[1] < c[0], 'le': a[1] <= c[0], 'gt': a[0] > c[1], 'ge': a[0] >= c[1]}[op.kind]
            no = {'ne': a[0] == a[1] == c[0] == c[1], 'eq': a[1] < c[0] or c[1] < a[0],
                  'ge': a[1] < c[0], 'gt': a[1] <= c[0], 'le': a[0] > c[1], 'lt': a[0] >= c[1]}[op.kind]
            result = (1, 1) if yes else (0, 0) if no else (0, 1)
        elif op.kind == 'select':
            condition, a, c = (bounds(v) for v in op.args)
            result = c if condition == (0, 0) else a if condition[0] > 0 or condition[1] < 0 else (min(a[0], c[0]), max(a[1], c[1]))
        elif op.kind == 'call' and 'result_mask' in op.attrs:
            result = 0, op.attrs['result_mask']
        elif op.kind == 'intrinsic':
            kind = op.attrs['builtin'].op
            if kind in ('ctz_nonzero', 'tzcnt'):
                result = 0, op.args[0].type.bits - (kind == 'ctz_nonzero')
            elif kind.startswith('cmpistr'):
                if kind == 'cmpistri':
                    result = 0, 16
                elif kind != 'cmpistrm':
                    result = 0, 1
        return result if full[0] <= result[0] <= result[1] <= full[1] else full

    def restrict(facts, value, low, high, seen=frozenset()):
        value = canonical(value)
        before = interval(value, facts)
        current = max(low, before[0]), min(high, before[1])
        if current[0] > current[1]:
            return False
        if current != domain(value):
            facts[value] = current
        op = fn.defs[value]
        if value not in seen and op.kind in ('add', 'sub', 'ptradd'):
            rhs = fn.defs[op.args[1]]
            if rhs.kind == 'constant' and op.args[0].type.bits == value.type.bits:
                shift = rhs.attrs['value'] * (1 if op.kind == 'sub' else -1)
                full = domain(op.args[0])
                modulus = full[1] - full[0] + 1
                low, high = current[0] + shift, current[1] + shift
                quotient = (low - full[0]) // modulus
                low, high = low - quotient * modulus, high - quotient * modulus
                if high <= full[1] and not restrict(facts, op.args[0], low, high, seen | {value}):
                    return False
        return True

    def assume(facts, value, truth):
        value = canonical(value)
        low, high = interval(value, facts)
        if not truth:
            if not restrict(facts, value, 0, 0):
                return None
        elif low == 0:
            if not restrict(facts, value, 1, high):
                return None
        elif high == 0:
            if not restrict(facts, value, low, -1):
                return None
        op = fn.defs[value]
        if op.kind not in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            return facts
        a, c = op.args
        ar, cr = interval(a, facts), interval(c, facts)
        kind = op.kind
        if cr[0] != cr[1] and ar[0] == ar[1]:
            a, c, ar, cr = c, a, cr, ar
            kind = {'eq': 'eq', 'ne': 'ne', 'lt': 'gt', 'le': 'ge', 'gt': 'lt', 'ge': 'le'}[kind]
        if cr[0] != cr[1]:
            return facts
        if not truth:
            kind = {'eq': 'ne', 'ne': 'eq', 'lt': 'ge', 'le': 'gt', 'gt': 'le', 'ge': 'lt'}[kind]
        c, (low, high) = cr[0], ar
        if kind == 'eq':
            low = high = c
        elif kind == 'ne':
            low += low == c
            high -= high == c
        elif kind in ('lt', 'le'):
            high = min(high, c - (kind == 'lt'))
        else:
            low = max(low, c + (kind == 'gt'))
        return facts if restrict(facts, a, low, high) else None

    pred = fn.predecessors()
    incoming = {b: None for b in fn.blocks}
    outgoing = {(b, target): None for b in fn.blocks for target in b.term.targets}
    pending, queued = deque([fn.entry]), {fn.entry}
    while pending:
        block = pending.popleft()
        queued.remove(block)
        inputs = [outgoing[p, block] for p in pred[block] if outgoing[p, block] is not None]
        facts = dict(getattr(fn, 'entry_ranges', {})) if block is fn.entry else None
        if block is not fn.entry and inputs:
            keys = set.intersection(*(set(f) for f in inputs))
            facts = {v: (min(f[v][0] for f in inputs), max(f[v][1] for f in inputs)) for v in keys}
        incoming[block] = facts
        for n, target in enumerate(block.term.targets):
            result = None if facts is None else dict(facts)
            if (result is not None and block.term.kind == 'br'
                    and block.term.targets[0] is not block.term.targets[1]):
                result = assume(result, block.term.args[0], n == 0)
            if outgoing[block, target] != result:
                outgoing[block, target] = result
                if target not in queued:
                    queued.add(target)
                    pending.append(target)
    replacements = {}
    for block, facts in incoming.items():
        if facts is None:
            continue
        for op in block.ops:
            if op.kind == 'call':
                op.attrs['arg_ranges'] = [interval(v, facts) if v.type.kind in ('int', 'ptr') else None for v in op.args[1:]]
            if op.kind not in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                continue
            low, high = interval(op.results[0], facts)
            if low == high:
                replacements[op.results[0]] = fn.constant(op.results[0].type, low)
        if block.term.kind == 'br':
            low, high = interval(block.term.args[0], facts)
            if low == high == 0 or low > 0 or high < 0:
                block.term = Term('jmp', targets=(block.term.targets[low == high == 0],), source=block.term.source)
    if replacements:
        for block in fn.blocks:
            for op in block.ops:
                op.args = tuple(replacements.get(v, v) for v in op.args)
            block.term.args = tuple(replacements.get(v, v) for v in block.term.args)
