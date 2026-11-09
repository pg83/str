"""Infer integer argument bits for internal functions with known callers."""
from collections import defaultdict

from .facts import possible_bits


def callee(fn, value):
    op = fn.defs[value]
    while op.kind == 'cast':
        op = fn.defs[op.args[0]]
    return op.attrs['name'] if op.kind == 'symbol' else None


def infer_parameter_masks(functions):
    by_name = {fn.name: fn for fn in functions}
    escaped = {fn.name for fn in functions if getattr(fn, 'global_address', False)
               or 'static' not in fn.decl.storage or fn.decl.attrs.get('constructor')}
    calls = []
    for fn in functions:
        uses = defaultdict(list)
        for block in fn.blocks:
            for value in block.term.args:
                uses[value].append((None, 0))
            for op in block.ops:
                for n, value in enumerate(op.args):
                    uses[value].append((op, n))
                if op.kind == 'call':
                    target = by_name.get(callee(fn, op.args[0]))
                    if target is not None:
                        calls.append((fn, op, target))

        def escapes(value, name, seen=frozenset()):
            if value in seen:
                return True
            for user, n in uses[value]:
                if user is not None and user.kind == 'cast':
                    if any(escapes(v, name, seen | {value}) for v in user.results):
                        return True
                elif user is None or user.kind != 'call' or n != 0 or callee(fn, user.args[0]) != name:
                    return True
            return False

        for block in fn.blocks:
            for op in block.ops:
                if (op.kind == 'symbol' and op.attrs['name'] in by_name
                        and escapes(op.results[0], op.attrs['name'])):
                    escaped.add(op.attrs['name'])
    for fn in functions:
        fn.parameter_masks = {p: (1 << p.type.bits)-1 if fn.name in escaped else 0
                              for p in fn.params if p.type.kind == 'int'}
    changed = True
    while changed:
        changed = False
        for fn, call, target in calls:
            if target.name in escaped:
                continue
            for arg, param in zip(call.args[1:], target.params):
                if param not in target.parameter_masks:
                    continue
                mask = target.parameter_masks[param] | (possible_bits(fn, arg) & ((1 << param.type.bits)-1))
                if mask != target.parameter_masks[param]:
                    target.parameter_masks[param] = mask
                    changed = True


def simplify_boolean_selections(fn):
    changed = False
    for block in fn.blocks:
        for op in list(block.ops):
            if op.kind != 'select' or op.results[0].type.kind != 'int' or possible_bits(fn, op.args[0]) > 1:
                continue
            yes, no = (fn.defs[v] for v in op.args[1:])
            if yes.kind != 'constant' or no.kind != 'constant':
                continue
            type_ = op.results[0].type
            all_ones = (1 << type_.bits)-1
            a, b = yes.attrs['value'] & all_ones, no.attrs['value'] & all_ones
            if (a, b) not in ((1, 0), (0, all_ones), (all_ones, 0)):
                continue
            value = op.args[0]
            if value.type != type_:
                cast = fn.make('cast', [type_], [value])
                block.ops.insert(block.ops.index(op), cast)
                value = cast.results[0]
            if (a, b) == (1, 0):
                op.kind, op.args = 'cast', (value,)
            elif a == 0:
                op.kind, op.args = 'add', (value, fn.constant(type_, -1))
            else:
                op.kind, op.args = 'neg', (value,)
            op.attrs = {}
            changed = True
    return changed
