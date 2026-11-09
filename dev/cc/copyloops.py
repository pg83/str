"""Induction variables for aligned forward and backward vector copies."""
from . import types as T
from .counted import index_copy_loop
from .regalloc import pointer_parameters


def normalize_copy_loops(fn):
    def bare(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value

    def constant(value):
        op = fn.defs[value]
        return op.attrs['value'] if op.kind == 'constant' else None

    calls = [(b, op) for b in fn.blocks for op in b.ops if op.kind == 'call']
    if (fn.type.base.kind != 'ptr' or len(fn.params) != 3 or not calls
            or any(b.term.kind != 'ret' or tuple(map(bare, b.term.args)) != op.results for b, op in calls)):
        return False
    memory = [op for b in fn.blocks for op in b.ops if op.kind != 'call' and op.attrs.get('memory')]
    if (not memory or any(op.attrs.get('width') != 16 or op.attrs.get('volatile') for op in memory)
            or not any('read' in op.attrs['memory'] for op in memory)
            or not any('write' in op.attrs['memory'] for op in memory)):
        return False
    registers = fn.induction_registers = dict(getattr(fn, 'induction_registers', {}))
    owners = fn.layout_owners = dict(getattr(fn, 'layout_owners', {}))
    pred, changed, forward = fn.predecessors(), False, []
    for head in reversed(list(fn.blocks)):
        if head.term.kind != 'br' or head not in head.term.targets or len(pred[head]) != 2:
            continue
        phis = [op for op in head.ops if op.kind == 'phi']
        counters = [op for op in phis if op.results[0].type.kind == 'int' and op.results[0].type.bits == 64]
        if len(counters) != 1:
            continue
        phi = counters[0]
        update = fn.defs[bare(phi.args[phi.attrs['blocks'].index(head)])]
        test = fn.defs[head.term.args[0]]
        if (update.kind != 'sub' or update.args[0] != phi.results[0]
                or constant(update.args[1]) not in (16, 64) or test.kind != 'ge'
                or test.args != (update.results[0], update.args[1])):
            continue
        step = constant(update.args[1])
        reads = [op for op in head.ops if 'read' in op.attrs.get('memory', '')]
        writes = [op for op in head.ops if 'write' in op.attrs.get('memory', '')]
        if (len(reads) != len(writes) or len(reads) != step//16
                or any(op.kind in ('call', 'asm') or op.attrs.get('volatile') for op in head.ops)):
            continue
        before = next(b for b in pred[head] if b is not head)
        pointers = [op for op in phis if op.results[0].type.kind == 'ptr']
        if len(pointers) not in (0, 2):
            continue
        if pointers and {frozenset(pointer_parameters(fn, p.results[0]) or ()) for p in pointers} != {frozenset({0}), frozenset({1})}:
            continue
        register = 'rcx' if pointers else 'rdx'
        registers[phi.results[0]] = registers[update.results[0]] = register
        initial = bare(phi.args[phi.attrs['blocks'].index(before)])
        registers[initial] = register
        for pointer in pointers:
            target = 'rax' if pointer_parameters(fn, pointer.results[0]) == {0} else 'rsi'
            registers[pointer.results[0]] = target
            for value in pointer.args:
                registers[bare(value)] = target
        if not pointers:
            # Address the current iteration as base+old_count-step and delay
            # decrementing the carried count until all stores have completed.
            for op in list(head.ops):
                if op.kind != 'ptradd' or bare(op.args[1]) != update.results[0]:
                    continue
                cast = fn.make('cast', [op.args[1].type], phi.results)
                base = fn.make('ptradd', [op.results[0].type], [op.args[0], cast.results[0]])
                at = head.ops.index(op)
                head.ops[at:at] = [cast, base]
                op.args = base.results[0], fn.constant(op.args[1].type, -step)
            head.ops.remove(update)
            head.ops.insert(head.ops.index(test), update)
            if step == 64:
                initial_op = fn.defs[initial]
                if initial_op.kind == 'sub':
                    low = fn.defs[bare(initial_op.args[1])]
                    if low.kind == 'and' and constant(low.args[1]) == 15:
                        registers[low.results[0]] = 'rcx'
                        registers[bare(low.args[0])] = 'rax'
        else:
            forward.append((head, before))
            if step == 64:
                start = fn.defs[initial]
                if start.kind == 'sub':
                    skew = fn.defs[bare(start.args[1])]
                    low = fn.defs[bare(skew.args[1])] if skew.kind == 'sub' else None
                    if low is not None and low.kind == 'and' and constant(skew.args[0]) == 16 and constant(low.args[1]) == 15:
                        registers[low.results[0]], registers[skew.results[0]] = 'rax', 'r8'
                        fn.scalar_constant_registers = dict(getattr(fn, 'scalar_constant_registers', {}))
                        fn.scalar_constant_registers[skew.args[0]] = 'r8'
                        for pointer in pointers:
                            if pointer_parameters(fn, pointer.results[0]) != {0}:
                                continue
                            address = fn.defs[bare(pointer.args[pointer.attrs['blocks'].index(before)])]
                            if address.kind != 'ptradd' or address not in before.ops or bare(address.args[1]) != skew.results[0]:
                                continue
                            negative = fn.make('neg', [T.ULONG], low.results)
                            base = fn.make('cast', [T.ULONG], [address.args[0]])
                            total = fn.make('add', [T.ULONG], [negative.results[0], base.results[0]])
                            cast = fn.make('cast', [address.results[0].type], total.results)
                            at = before.ops.index(address)
                            before.ops[at:at] = [negative, base, total, cast]
                            address.args = cast.results[0], fn.constant(T.LONG, 16)
                            registers[negative.results[0]] = registers[total.results[0]] = 'rax'
            else:
                index_copy_loop(fn, head, before, step, 'r8')
        update.kind, update.args = 'add', (update.args[0], fn.constant(update.results[0].type, -step))
        test.kind, test.args = 'gt', (test.args[0], fn.constant(test.args[0].type, step-1))
        for target in head.term.targets:
            if target is not head:
                owners[target] = head
        changed = True
    if not changed:
        return False
    fn.copy_loops = True
    # Keep the no-overlap forward path inline and place the backward body
    # after its return, using the complement of the overlap predicate.
    for block in fn.blocks:
        if block.term.kind != 'br':
            continue
        both = fn.defs[block.term.args[0]]
        if both.kind != 'and':
            continue
        left, right = (fn.defs[v] for v in both.args)
        if (left.kind != 'gt' or right.kind != 'lt' or bare(left.args[0]) != fn.params[0]
                or bare(left.args[1]) != fn.params[1] or bare(right.args[0]) != fn.params[0]):
            continue
        end = fn.defs[bare(right.args[1])]
        if end.kind != 'ptradd' or end.args[0] != fn.params[1] or bare(end.args[1]) != fn.params[2]:
            continue
        left.kind, right.kind, both.kind = 'le', 'ge', 'or'
        both.args = both.args[::-1]
        registers[left.results[0]], registers[right.results[0]], registers[both.results[0]] = 'rax', 'rcx', 'rcx'
        backward, ahead = block.term.targets
        block.term.targets = ahead, backward
        exits = [target for head, _ in forward for target in head.term.targets if target is not head and target.term.kind == 'ret']
        if len(exits) == 1:
            owners[backward] = exits[0]
        break
    return True
