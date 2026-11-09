"""Share offsets and bit scans in a grouped vector length search."""
from . import types as T
from .ir import Term


def prepare_vector_lengths(fn):
    if (fn.type.base != T.ULONG or len(fn.params) != 2
            or any(value.type.kind != 'ptr' for value in fn.params)
            or any(op.kind in ('call', 'asm') or op.attrs.get('volatile')
                   or 'write' in op.attrs.get('memory', '') for b in fn.blocks for op in b.ops)):
        return False
    trees = {b for b in fn.blocks if b in b.term.targets
             and sum(op.kind == 'intrinsic' and op.attrs['builtin'].op == 'min_epu8'
                     and op.results[0].type.bits == 256 for op in b.ops) == 3}
    join = getattr(fn, 'shared_return', None)
    if (len(trees) != 1 or join not in fn.blocks or len(join.ops) != 1
            or join.ops[0].kind != 'phi' or join.term.kind != 'ret'
            or join.term.args != join.ops[0].results):
        return False
    head = next(iter(trees))
    reductions = [op for op in head.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'min_epu8']
    roots = [op for op in reductions if all(fn.defs[v] in reductions for v in op.args)]
    if len(roots) != 1:
        return False
    left, right = (fn.defs[v] for v in roots[0].args)

    def bare(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value

    def constant(value):
        op = fn.defs[value]
        return T.convert_integer(op.attrs['value'], T.LONG) if op.kind == 'constant' and isinstance(op.attrs['value'], int) else None

    def parts(value):
        value = bare(value)
        op = fn.defs[value]
        if op.kind == 'add' and value.type.bits == 64:
            return parts(op.args[0]) + parts(op.args[1])
        return [value]

    def probe(mask):
        op = fn.defs[bare(mask)]
        if op.kind != 'intrinsic' or op.attrs['builtin'].op != 'movemask8':
            return False
        compare = fn.defs[op.args[0]]
        if compare.kind != 'intrinsic' or compare.attrs['builtin'].op != 'cmpeq_epi8':
            return False
        return any(fn.defs[v].kind == 'intrinsic' and fn.defs[v].attrs.get('memory') == 'read'
                   and fn.defs[v].attrs.get('width') == 32 for v in compare.args)

    incoming = join.ops[0]
    records = []
    for block, value in zip(incoming.attrs['blocks'], incoming.args):
        if block.term.kind != 'jmp' or block.term.targets != (join,):
            return False
        count, distance, delta = None, None, 0
        for term in parts(value):
            op = fn.defs[term]
            if op.kind == 'cast' and op.args[0].type.bits == 32:
                scan = fn.defs[op.args[0]]
                if scan.kind == 'intrinsic' and scan.attrs['builtin'].op == 'ctz_nonzero' and count is None:
                    count = scan
                    continue
            if op.kind == 'sub' and bare(op.args[1]) == fn.params[1] and distance is None:
                distance = bare(op.args[0])
            elif constant(term) is not None:
                delta += constant(term)
            else:
                return False
        if count is None or distance is None:
            return False
        while fn.defs[distance].kind == 'ptradd':
            address = fn.defs[distance]
            offset = constant(address.args[1])
            if offset is None:
                return False
            delta += offset
            distance = bare(address.args[0])
        records.append((block, value, count, distance, delta))
    groups = {}
    for record in records:
        if probe(record[2].args[0]):
            groups.setdefault(record[3], []).append(record)
    if (len(groups) != 2 or any(len(group) != 3 for group in groups.values())
            or not any(fn.defs[base].kind == 'phi' and fn.defs[base] in head.ops for base in groups)):
        return False

    fn.vector_length_loop = head
    fn.shared_return = None
    registers = fn.induction_registers = dict(getattr(fn, 'induction_registers', {}))
    owners = fn.layout_owners = dict(getattr(fn, 'layout_owners', {}))
    joins = fn.return_joins = list(getattr(fn, 'return_joins', ()))

    def prefer(value, register):
        registers[value] = register
        while fn.defs[value].kind == 'cast':
            value = fn.defs[value].args[0]
            registers[value] = register

    def make(block, kind, type_, args, attrs=None):
        op = fn.make(kind, [type_], args, attrs or {})
        block.ops.append(op)
        return op.results[0]

    grouped = {record[0] for group in groups.values() for record in group}
    for block, value, count, base, delta in records:
        if block not in grouped:
            block.term = Term('ret', (value,))
            prefer(count.results[0], 'rax')
            if base == fn.params[0]:
                distance = next(fn.defs[term] for term in parts(value) if fn.defs[term].kind == 'sub')
                prefer(distance.results[0], 'rdi')
                fn.defs[value].args = fn.defs[value].args[::-1]
            else:
                prefer(count.args[0], 'rsi')

    pred = fn.predecessors()
    for base, group in groups.items():
        loop = fn.defs[base].kind == 'phi' and fn.defs[base] in head.ops
        group.sort(key=lambda record: record[4])
        target = fn.block('return.vector.offset')
        blocks = [record[0] for record in group]
        mask = make(target, 'phi', group[0][2].args[0].type,
                    [record[2].args[0] for record in group], {'blocks': blocks})
        offset = make(target, 'phi', T.LONG, [fn.constant(T.LONG, record[4]) for record in group], {'blocks': blocks})
        original = group[0][2]
        address = make(target, 'cast', T.LONG, [base])
        if not loop:
            address = make(target, 'add', T.LONG, [address, offset])
        start = make(target, 'cast', T.LONG, [fn.params[1]])
        distance = make(target, 'sub', T.LONG, [address, start])
        unsigned = make(target, 'cast', T.ULONG, [distance])
        count = make(target, 'intrinsic', original.results[0].type, [mask], dict(original.attrs))
        extended = make(target, 'cast', T.ULONG, [count])
        if loop:
            unsigned = make(target, 'add', T.ULONG, [offset, unsigned])
            prefer(unsigned, 'rax')
        result = make(target, 'add', T.ULONG, [unsigned, extended] if loop else [extended, unsigned])
        target.term = Term('ret', (result,))
        for block, _, _, _, _ in group:
            block.term = Term('jmp', targets=(target,))
        prefer(mask, 'rcx' if loop else 'rax')
        prefer(count, 'rcx' if loop else 'rax')
        prefer(offset, 'rax' if loop else 'rdx')
        prefer(address, 'rcx')
        if not loop:
            prefer(distance, 'rcx')
        for record in group:
            prefer(record[2].args[0], 'rcx' if loop else 'rax')
        owners[target] = pred[group[-1][0]][0]
        joins.append(target)
    if head.ops.index(left) > head.ops.index(right):
        head.ops.remove(left)
        head.ops.insert(head.ops.index(right), left)
    fn.blocks.remove(join)
    return True
