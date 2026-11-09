"""Guaranteed power-of-two alignment, including induction-variable cycles."""


def alignments(fn, maximum=4096):
    ops = [*fn.constants, *fn.allocas, *(fn.defs[v] for v in fn.params), *(op for b in fn.blocks for op in b.ops)]
    values = {v: maximum for op in ops for v in op.results}
    def constant(value):
        op = fn.defs[value]
        return op.attrs['value'] if op.kind == 'constant' and isinstance(op.attrs['value'], int) else None
    changed = True
    while changed:
        changed = False
        for op in ops:
            if not op.results:
                continue
            args = [values.get(v, 1) for v in op.args]
            result = 1
            if op.kind == 'constant' and isinstance(op.attrs['value'], int):
                n = op.attrs['value']
                result = min(maximum, abs(n) & -abs(n)) if n else maximum
            elif op.kind == 'alloca':
                result = min(maximum, op.attrs['alignment'])
            elif op.kind in ('cast', 'neg'):
                result = args[0]
            elif op.kind in ('add', 'sub', 'ptradd', 'or', 'xor', 'phi', 'select'):
                result = min(args[1:] if op.kind == 'select' else args, default=1)
            elif op.kind == 'and':
                result = max(args)
            elif op.kind == 'mul':
                result = min(maximum, args[0] * args[1])
            elif op.kind in ('shl', 'shr'):
                count = constant(op.args[1])
                if count is not None and 0 <= count < op.args[0].type.bits:
                    result = min(maximum, args[0] << count) if op.kind == 'shl' else max(1, args[0] >> count)
            for v in op.results:
                if result < values[v]:
                    values[v], changed = result, True
    return values
