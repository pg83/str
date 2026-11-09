"""Hoist nontrapping loop-invariant arithmetic into a unique entry block."""
from .ir import Term, dominators


PURE = {'cast', 'symbol', 'ptradd', 'add', 'sub', 'mul', 'and', 'or', 'xor',
        'neg', 'not', 'shl', 'shr', 'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'select'}


def loop_preheaders(fn):
    """Merge all entries to a natural loop before moving invariant work."""
    changed = False
    for head in list(fn.blocks):
        pred, dom = fn.predecessors(), dominators(fn)
        tails = [b for b in pred[head] if head in dom[b]]
        if not tails:
            continue
        members, todo = {head}, list(tails)
        while todo:
            block = todo.pop()
            if block not in members:
                members.add(block)
                todo.extend(pred[block])
        incoming = [b for b in pred[head] if b not in members]
        if len(incoming) <= 1:
            continue
        entry = fn.block('loop.preheader')
        for phi in head.ops:
            if phi.kind != 'phi':
                continue
            pairs = list(zip(phi.attrs['blocks'], phi.args))
            initial = [(b, v) for b, v in pairs if b in incoming]
            if len({v for _, v in initial}) == 1:
                value = initial[0][1]
            else:
                merged = fn.make('phi', [phi.results[0].type], [v for _, v in initial], {'blocks': [b for b, _ in initial]})
                entry.ops.append(merged)
                value = merged.results[0]
            keep = [(b, v) for b, v in pairs if b not in incoming]
            phi.attrs['blocks'], phi.args = [entry, *(b for b, _ in keep)], (value, *(v for _, v in keep))
        entry.term = Term('jmp', targets=(head,))
        for block in incoming:
            block.term.targets = tuple(entry if target is head else target for target in block.term.targets)
        fn.blocks.remove(entry)
        fn.blocks.insert(fn.blocks.index(head), entry)
        fn.verify()
        changed = True
    return changed


def hoist_invariants(fn):
    dom, pred = dominators(fn), fn.predecessors()
    loops = {}
    for tail in fn.blocks:
        for head in tail.term.targets:
            if head not in dom[tail]:
                continue
            members, todo = loops.setdefault(head, {head}), [tail]
            while todo:
                block = todo.pop()
                if block not in members:
                    members.add(block)
                    todo.extend(pred[block])
    owners = {v: b for b in fn.blocks for op in b.ops for v in op.results}
    changed = False
    for head, members in sorted(loops.items(), key=lambda pair: len(pair[1])):
        entry = [b for b in pred[head] if b not in members]
        if len(entry) != 1:
            continue
        before = entry[0]
        pending = True
        while pending:
            pending = False
            for block in fn.blocks:
                if block not in members:
                    continue
                for op in list(block.ops):
                    if op.kind not in PURE or op.attrs.get('memory') or op.attrs.get('volatile'):
                        continue
                    if any(owners.get(v) in members for v in op.args):
                        continue
                    block.ops.remove(op)
                    before.ops.append(op)
                    owners.update((v, before) for v in op.results)
                    changed = pending = True
    if changed:
        fn.verify()
    return changed
