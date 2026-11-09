"""Conservative possible-one-bit facts for scalar SSA values."""


def possible_bits(fn, value, seen=frozenset()):
    full = (1 << value.type.bits) - 1
    if value in seen:
        return full
    seen = seen | {value}
    op = fn.defs[value]
    bits = lambda v: possible_bits(fn, v, seen)
    if op.kind == 'param':
        return getattr(fn, 'parameter_masks', {}).get(value, full)
    if op.kind == 'constant' and isinstance(op.attrs['value'], int):
        return op.attrs['value'] & full
    if op.kind in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
        return 1
    if op.kind == 'cast':
        source = op.args[0]
        mask = bits(source)
        if value.type.bits > source.type.bits and source.type.signed and mask & (1 << (source.type.bits - 1)):
            mask |= full ^ ((1 << source.type.bits) - 1)
        return mask & full
    if op.kind == 'and':
        return bits(op.args[0]) & bits(op.args[1]) & full
    if op.kind == 'add':
        upper = bits(op.args[0])+bits(op.args[1])
        if upper <= full:
            return (1 << upper.bit_length())-1
    if op.kind == 'sub':
        constant = fn.defs[op.args[0]]
        if constant.kind == 'constant' and 0 <= bits(op.args[1]) <= constant.attrs['value'] < 1 << (value.type.bits - value.type.signed):
            return (1 << constant.attrs['value'].bit_length()) - 1
    if op.kind == 'select':
        comparison = fn.defs[op.args[0]]
        if comparison.kind in ('lt', 'le') and tuple(op.args[1:]) == comparison.args:
            upper = min(bits(op.args[1]), bits(op.args[2]))
            return (1 << upper.bit_length()) - 1
    if op.kind in ('or', 'xor', 'phi', 'select'):
        mask = 0
        for arg in op.args[1:] if op.kind == 'select' else op.args:
            mask |= bits(arg)
        return mask & full
    if op.kind in ('shl', 'shr'):
        count = fn.defs[op.args[1]]
        if count.kind == 'constant' and 0 <= count.attrs['value'] < value.type.bits:
            n, mask = count.attrs['value'], bits(op.args[0])
            if op.kind == 'shl':
                return (mask << n) & full
            if not op.args[0].type.signed or not mask & (1 << (value.type.bits - 1)):
                return mask >> n
    if op.kind == 'call':
        return op.attrs.get('result_mask', full) & full
    if op.kind == 'intrinsic':
        builtin = op.attrs['builtin']
        if builtin.op in ('ctz_nonzero', 'tzcnt'):
            limit = op.args[0].type.bits - (builtin.op == 'ctz_nonzero')
            return (1 << limit.bit_length()) - 1
        if builtin.op in ('cmpistra', 'cmpistrc', 'cmpistrz', 'testz'):
            return 1
        if builtin.op == 'cmpistri':
            return 31
        if builtin.op == 'movemask8':
            return (1 << op.args[0].type.size) - 1
    return full


def boolean_values(fn):
    """Values whose consumers only observe their truth value or low bit."""
    cache = {}

    def boolean(v, seen=frozenset()):
        if v in cache:
            return cache[v]
        if v in seen or v.type.kind != 'int':
            return False
        op = fn.defs[v]
        if op.kind == 'constant':
            result = op.attrs['value'] in (0, 1)
        elif op.kind in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            result = True
        elif op.kind == 'and' and fn.defs[op.args[1]].kind == 'constant' and fn.defs[op.args[1]].attrs['value'] == 1:
            result = True
        elif op.kind in ('cast', 'phi', 'select', 'and', 'or', 'xor'):
            args = op.args[1:] if op.kind == 'select' else op.args
            result = all(boolean(a, seen | {v}) for a in args)
        elif op.kind == 'call':
            result = op.attrs.get('result_mask', -1) in (0, 1)
        elif op.kind == 'intrinsic':
            result = op.attrs['builtin'].op in ('cmpistra', 'cmpistrc', 'cmpistrz', 'testz')
        else:
            result = False
        cache[v] = result
        return result

    values = {v for b in fn.blocks for op in b.ops for v in (*op.results, *op.args)}
    values.update(v for b in fn.blocks for v in b.term.args)
    candidates = {v for v in values if boolean(v)}
    changed = True
    while changed:
        changed = False
        bad = {v for b in fn.blocks if b.term.kind != 'br' for v in b.term.args}
        for b in fn.blocks:
            for op in b.ops:
                for n, v in enumerate(op.args):
                    safe = op.kind == 'select' and n == 0
                    if op.kind == 'cast' and op.results[0].type.kind == 'int':
                        safe = True
                    if op.kind in ('cast', 'phi', 'and', 'or', 'xor', 'select'):
                        safe |= op.results[0] in candidates
                    if op.kind in ('eq', 'ne'):
                        other = fn.defs[op.args[1 - n]]
                        safe |= other.kind == 'constant' and other.attrs['value'] in (0, 1)
                    if not safe:
                        bad.add(v)
        if bad & candidates:
            candidates -= bad
            changed = True
    return candidates
