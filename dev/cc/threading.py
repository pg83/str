"""Bypass known status branches while preserving their companion values."""
from collections import defaultdict

from .ir import Term, dominators


def thread_status_edge(fn):
    """Thread one constant incoming status through a multi-value merge.

    A phi operand is used on its incoming edge, rather than in its phi's
    block. This distinction lets existing successor phis accept the new
    edge directly. Other companion uses get a new successor phi when that
    successor dominates every use affected by the bypass.
    """
    candidates = []
    for block in fn.blocks:
        if block is fn.entry or block.term.kind != 'br':
            continue
        phis = [op for op in block.ops if op.kind == 'phi']
        extra = [op for op in block.ops if op.kind != 'phi']
        condition = fn.defs[block.term.args[0]]
        if len(phis) < 2 or condition not in phis:
            continue
        if any(op.kind != 'cast' or any(fn.defs[v].kind not in ('param', 'constant', 'alloca', 'undef')
                                       for v in op.args) for op in extra):
            continue
        candidates.append((block, phis, extra, condition))
    if not candidates:
        return False
    pred, dom = fn.predecessors(), dominators(fn)
    uses = defaultdict(list)
    for block in fn.blocks:
        for op in block.ops:
            for n, value in enumerate(op.args):
                uses[value].append(op.attrs['blocks'][n] if op.kind == 'phi' else block)
        for value in block.term.args:
            uses[value].append(block)
    for block, phis, extra, condition in candidates:
        if len(uses[condition.results[0]]) != 1:
            continue
        for parent, incoming in zip(condition.attrs['blocks'], condition.args):
            value = fn.defs[incoming]
            if value.kind != 'constant' or parent is block:
                continue
            target = block.term.targets[0 if value.attrs['value'] else 1]
            reached, pending = set(), [target]
            while pending:
                current = pending.pop()
                if current not in reached:
                    reached.add(current)
                    pending.extend(current.term.targets)
            # Keep loop-carried state at its original merge. Rejoins outside
            # the successor's dominance region also need a broader SSA repair.
            if block in reached:
                continue
            repairs, valid = [], True
            for phi in phis:
                value = phi.results[0]
                affected = [owner for owner in uses[value] if owner in reached]
                if affected and (block not in dom[target] or any(target not in dom[owner] for owner in affected)):
                    valid = False
                    break
                if affected:
                    repairs.append(value)
            if not valid:
                continue

            # These casts depend only on entry definitions and cannot trap.
            for op in extra:
                block.ops.remove(op)
                fn.entry.ops.append(op)
            incoming_values = {op.results[0]: op.args[op.attrs['blocks'].index(parent)] for op in phis}
            edge = fn.block('thread.' + parent.name)
            edge.term = Term('jmp', targets=(target,))
            for op in target.ops:
                if op.kind == 'phi':
                    old = op.args[op.attrs['blocks'].index(block)]
                    op.args += (incoming_values.get(old, old),)
                    op.attrs['blocks'].append(edge)
            replacements, added = {}, []
            for value in repairs:
                phi = fn.make('phi', [value.type], [*[value for _ in pred[target]], incoming_values[value]],
                              {'blocks': [*pred[target], edge]})
                added.append(phi)
                replacements[value] = phi.results[0]
            for other in fn.blocks:
                for op in other.ops:
                    op.args = tuple(replacements.get(v, v) if target in dom.get(
                        op.attrs['blocks'][n] if op.kind == 'phi' else other, set()) else v
                                    for n, v in enumerate(op.args))
                if target in dom.get(other, set()):
                    other.term.args = tuple(replacements.get(v, v) for v in other.term.args)
            target.ops[:0] = added
            if parent.term.kind == 'jmp' and any(op.kind == 'call' for op in target.ops):
                # Prefer fallthrough from the argument-update block over a
                # conditional cold edge into the same shared call.
                fn.layout_owners = {**getattr(fn, 'layout_owners', {}), target: parent}
            parent.term.targets = tuple(edge if t is block else t for t in parent.term.targets)
            fn.blocks = fn.reachable()
            current_pred = fn.predecessors()
            for other in fn.blocks:
                for op in other.ops:
                    if op.kind == 'phi':
                        pairs = [(p, v) for p, v in zip(op.attrs['blocks'], op.args) if p in current_pred[other]]
                        op.attrs['blocks'] = [p for p, _ in pairs]
                        op.args = tuple(v for _, v in pairs)
            return True
    return False
