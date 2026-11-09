"""Combine scalar masks before extracting their vector sign bits."""
from collections import Counter

from .builtins import BUILTINS
from . import types as T


def combine_movemasks(fn):
    uses = Counter(v for b in fn.blocks for op in b.ops for v in op.args)
    uses.update(v for b in fn.blocks for v in b.term.args)
    for block in fn.blocks:
        before = {}
        local = set(block.ops)

        def mask(value):
            moved, inverse = set(), None
            while True:
                op = fn.defs[value]
                if op not in local:
                    return None
                moved.add(op)
                if op.kind == 'cast' and value.type.bits >= 32 and op.args[0].type.bits >= 32:
                    value = op.args[0]
                elif (op.kind == 'xor' and inverse is None and fn.defs[op.args[1]].kind == 'constant'
                      and fn.defs[op.args[1]].attrs['value'] == 65535):
                    value, inverse = op.args[0], op
                elif op.kind == 'intrinsic' and op.attrs['builtin'].op == 'movemask8' and op.args[0].type == T.V128:
                    return op.args[0], inverse, moved
                else:
                    return None

        def intrinsic(name, args):
            builtin = BUILTINS[name]
            return fn.make('intrinsic', [builtin.result], args, {'builtin': builtin, 'memory': '',
                           'width': 0, 'alignment': 1, 'mask': None})

        for op in block.ops:
            if op.kind not in ('and', 'or', 'xor'):
                continue
            inputs = [mask(v) for v in op.args]
            if any(part is None for part in inputs):
                continue
            moved = set.union(*(part[2] for part in inputs))
            expected = Counter(v for item in moved for v in item.args)
            expected.update(op.args)
            if any(uses[v] != expected[v] for item in moved for v in item.results):
                continue
            vectors = []
            for vector, inverse, _ in inputs:
                if inverse is not None:
                    ones = fn.constant(T.V128, b'\xff'*16)
                    invert = intrinsic('_mm_xor_si128', [vector, ones])
                    before.setdefault(inverse, []).append(invert)
                    vector = invert.results[0]
                vectors.append(vector)
            combined = intrinsic('_mm_' + op.kind + '_si128', vectors[::-1])
            extracted = intrinsic('_mm_movemask_epi8', combined.results)
            before.setdefault(op, []).extend([combined, extracted])
            op.kind, op.args, op.attrs = 'cast', extracted.results, {}
        block.ops = [item for op in block.ops for item in (*before.get(op, ()), op)]
