"""Guard fusion and descending remainders for aligned vector searches."""
from . import types as T
from .ir import Term, dominators
from .alignment import alignments


def prepare_aligned_searches(fn):
    if (fn.type.base.kind != 'ptr' or len(fn.params) != 3
            or fn.params[0].type.kind != 'ptr' or fn.params[1].type.bits != 8
            or fn.params[2].type != T.ULONG
            or any(op.kind in ('call', 'asm') or op.attrs.get('volatile') or 'write' in op.attrs.get('memory', '')
                   for b in fn.blocks for op in b.ops)):
        return False

    def bare(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value

    def defined(value):
        return fn.defs[bare(value)]

    def constant(value):
        op = fn.defs[value]
        return op.attrs['value'] if op.kind == 'constant' else None

    clamp = next((op for op in fn.entry.ops if op.kind == 'select'
                  and op.args[2] == fn.params[2] and constant(op.args[1]) == (1 << 63)-1), None)
    splat = next((op for op in fn.entry.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'splat8'), None)
    if clamp is None or splat is None:
        return False
    width = splat.results[0].type.size
    if width not in (16, 32):
        return False
    aligned_reg, cursor_reg, remaining_reg = ('rcx', 'rdx', 'rsi') if width == 16 else ('rsi', 'rcx', 'rdx')
    pred, alignment = fn.predecessors(), alignments(fn)
    guards = []
    for head in list(fn.blocks):
        if head.term.kind != 'br':
            continue
        mask = defined(head.term.args[0])
        if mask.kind != 'and' or constant(mask.args[1]) not in (3*width, 4*width-1):
            continue
        rhs, done = head.term.targets
        if rhs.term.kind != 'br' or pred[rhs] != [head] or rhs.term.targets[1] is not done:
            continue
        test = defined(rhs.term.args[0])
        if test.kind != 'ge' or constant(test.args[1]) != width or not test.args[0].type.signed:
            continue
        distance = defined(test.args[0])
        if distance.kind != 'sub' or any(op.kind not in ('cast', 'sub', 'ge') for op in rhs.ops):
            continue
        if any(phi.kind == 'phi' and phi.args[phi.attrs['blocks'].index(head)] != phi.args[phi.attrs['blocks'].index(rhs)]
               for phi in done.ops):
            continue
        guards.append((head, rhs, done, mask, test, distance))
    if len(guards) != 4 or len({done for _, _, done, *_ in guards}) != 1:
        return False
    fn.aligned_vector_search = True
    fn.aligned_search_width = width
    registers = fn.induction_registers = dict(getattr(fn, 'induction_registers', {}))
    vectors = fn.vector_registers = dict(getattr(fn, 'vector_registers', {}))
    owners = fn.layout_owners = dict(getattr(fn, 'layout_owners', {}))
    vectors[splat.results[0]] = 'xmm0'

    def prefer(value, register):
        registers[value] = register
        while fn.defs[value].kind == 'cast':
            value = fn.defs[value].args[0]
            registers[value] = register

    def ordered(block, roots):
        original, seen, output = set(block.ops), set(), []
        def visit(op):
            if op not in original or op in seen:
                return
            for value in op.args:
                visit(fn.defs[value])
            seen.add(op)
            output.append(op)
        for op in block.ops:
            if op.kind == 'phi':
                visit(op)
        for root in roots:
            visit(root)
        for op in block.ops:
            visit(op)
        block.ops = output

    distances, bodies = {}, []
    for n, (head, rhs, done, mask, test, distance) in enumerate(guards):
        if alignment.get(mask.args[0], 1) >= width:
            mask.args = mask.args[0], fn.constant(mask.args[1].type, 3*width)
        boolean = fn.make('ne', [T.INT], [mask.results[0], fn.constant(mask.results[0].type, 0)])
        combined = fn.make('and', [T.INT], [boolean.results[0], test.results[0]] if n == 0
                           else [test.results[0], boolean.results[0]], {'boolean_test': n == 0})
        head.ops.extend([boolean, *rhs.ops, combined])
        body = rhs.term.targets[0]
        head.term = Term('br', combined.results, (body, done))
        for phi in done.ops:
            if phi.kind == 'phi':
                pairs = [(b, v) for b, v in zip(phi.attrs['blocks'], phi.args) if b is not rhs]
                phi.attrs['blocks'], phi.args = [b for b, _ in pairs], tuple(v for _, v in pairs)
        for phi in body.ops:
            if phi.kind == 'phi':
                phi.attrs['blocks'] = [head if b is rhs else b for b in phi.attrs['blocks']]
        fn.blocks.remove(rhs)
        distances[head] = distance.results[0]
        bodies.append(body)
        prefer(distance.results[0], remaining_reg)
        prefer(boolean.results[0], cursor_reg if n == 0 else 'rdi')
        prefer(test.results[0], 'rdi' if n == 0 else 'r8')
        prefer(combined.results[0], 'r8')
        prefer(distance.args[1], aligned_reg if n == 0 else cursor_reg)
    end = bare(guards[0][-1].args[0])
    prefer(end, 'rax')
    prefer(clamp.results[0], 'rax')
    initial_pointer = bare(guards[0][-1].args[1])
    root = defined(initial_pointer)
    advanced = defined(root.args[0])
    if root.kind == 'and' and advanced.kind == 'ptradd':
        prefer(advanced.results[0], 'rdi')
        ordered(fn.entry, [clamp, splat, defined(end), advanced, root])
    done = guards[0][2]
    pointer = next(op for op in done.ops if op.kind == 'phi' and op.results[0].type.kind == 'ptr')
    incoming = []
    for parent, value in zip(pointer.attrs['blocks'], pointer.args):
        if parent in distances:
            incoming.append(distances[parent])
            continue
        cast = fn.make('cast', [T.LONG], [value])
        difference = fn.make('sub', [T.LONG], [guards[0][-1].args[0], cast.results[0]])
        parent.ops.extend([cast, difference])
        incoming.append(difference.results[0])
        prefer(value, aligned_reg)
        prefer(difference.results[0], remaining_reg)
    remaining = fn.make('phi', [T.LONG], incoming, {'blocks': list(pointer.attrs['blocks'])})
    done.ops.insert(0, remaining)
    replace = {}
    for op in done.ops:
        if op.kind == 'sub' and bare(op.args[0]) == end and bare(op.args[1]) == pointer.results[0]:
            replace[op.results[0]] = remaining.results[0]
    prefer(pointer.results[0], cursor_reg)
    prefer(remaining.results[0], remaining_reg)
    def rewrite(replacements):
        for block in fn.blocks:
            for op in block.ops:
                op.args = tuple(replacements.get(v, v) for v in op.args)
            block.term.args = tuple(replacements.get(v, v) for v in block.term.args)
    rewrite(replace)

    # Rotate end-cursor into a carried remainder, but initialize each guarded
    # vector loop on its taken edge, where its signed bound is established.
    pred, dom = fn.predecessors(), dominators(fn)
    loops = []
    for head in list(fn.blocks):
        for phi in list(head.ops):
            if phi.kind != 'phi' or phi.results[0].type.kind != 'ptr' or len(phi.args) != 2:
                continue
            tails = [b for b in pred[head] if head in dom[b]]
            if len(tails) != 1 or tails[0] is head:
                continue
            tail = tails[0]
            before = next(b for b in pred[head] if b is not tail)
            if tail.term.kind != 'br' or head not in tail.term.targets or before.term.kind != 'br' or head not in before.term.targets:
                continue
            next_pointer = phi.args[phi.attrs['blocks'].index(tail)]
            update = defined(next_pointer)
            test = defined(tail.term.args[0])
            if update.kind != 'ptradd' or update.args[0] != phi.results[0] or constant(update.args[1]) not in (width, 4*width):
                continue
            old_distance = defined(test.args[0]) if test.kind == 'ge' else None
            if (old_distance is None or old_distance.kind != 'sub' or bare(old_distance.args[0]) != end
                    or bare(old_distance.args[1]) != next_pointer or test.args[1] != update.args[1]):
                continue
            loops.append((head, phi, before, tail, update, test, old_distance))
    for head, pointer_phi, before, tail, update, test, old_distance in loops:
        step = constant(update.args[1])
        exit_ = next(b for b in tail.term.targets if b is not head)
        initial = pointer_phi.args[pointer_phi.attrs['blocks'].index(before)]
        entry = fn.block('search.count')
        cast = fn.make('cast', [T.LONG], [initial])
        start = fn.make('sub', [T.LONG], [guards[0][-1].args[0], cast.results[0]])
        entry.ops = [cast, start]
        entry.term = Term('jmp', targets=(head,))
        before.term.targets = tuple(entry if b is head else b for b in before.term.targets)
        for phi in head.ops:
            if phi.kind == 'phi':
                phi.attrs['blocks'] = [entry if b is before else b for b in phi.attrs['blocks']]
        count = fn.make('phi', [T.LONG], attrs={'blocks': [entry, tail]})
        next_ = fn.make('add', [T.LONG], [count.results[0], fn.constant(T.LONG, -step)])
        count.args = start.results[0], next_.results[0]
        head.ops.insert(0, count)
        tail.ops.insert(tail.ops.index(test), next_)
        rewrite({old_distance.results[0]: next_.results[0]})
        test.kind, test.args = 'gt', (next_.results[0], fn.constant(T.LONG, step-1))
        for op in list(exit_.ops):
            if op.kind != 'sub' or bare(op.args[0]) != end:
                continue
            exit_pointer = defined(op.args[1])
            if exit_pointer.kind != 'phi' or exit_pointer not in exit_.ops:
                continue
            bound = fn.defs[before.term.args[0]].args[0]
            rest = fn.make('phi', [T.LONG], [bound if b is before else next_.results[0] for b in exit_pointer.attrs['blocks']],
                           {'blocks': list(exit_pointer.attrs['blocks'])})
            exit_.ops.insert(0, rest)
            rewrite({op.results[0]: rest.results[0]})
            prefer(rest.results[0], remaining_reg)
        for value in (start.results[0], count.results[0], next_.results[0]):
            prefer(value, remaining_reg)
        for value in (initial, pointer_phi.results[0], update.results[0]):
            prefer(value, cursor_reg)
        owners[exit_] = tail
    fn.aligned_search_bodies = tuple(bodies)
    fn.aligned_search_loops = tuple(head for head, *_ in loops)
    owners[done] = bodies[-1].term.targets[1]
    fn.deferred_phi_blocks = set(getattr(fn, 'deferred_phi_blocks', ())) | {done}
    return True


def finish_aligned_searches(fn):
    if not getattr(fn, 'aligned_vector_search', False):
        return False
    registers, owners = fn.induction_registers, fn.layout_owners
    bodies, loops = fn.aligned_search_bodies, fn.aligned_search_loops
    width = fn.aligned_search_width
    aligned_reg, cursor_reg, remaining_reg = ('rcx', 'rdx', 'rsi') if width == 16 else ('rsi', 'rcx', 'rdx')

    def bare(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value

    def prefer(value, register):
        registers[value] = register
        while fn.defs[value].kind == 'cast':
            value = fn.defs[value].args[0]
            registers[value] = register

    old = getattr(fn, 'shared_return', None)
    if old is None or old not in fn.blocks or len(old.ops) != 1 or old.ops[0].kind != 'phi':
        return False
    phi = old.ops[0]
    if any(not (parent.term.kind == 'jmp' and parent.term.targets == (old,))
           and (fn.defs[value].kind != 'constant' or fn.defs[value].attrs['value'] != 0)
           for parent, value in zip(phi.attrs['blocks'], phi.args)):
        return False
    zero = fn.block('search.not.found')
    zero.term = Term('ret', (fn.constant(fn.type.base, 0),))
    for parent, value in zip(phi.attrs['blocks'], phi.args):
        if parent.term.kind == 'jmp' and parent.term.targets == (old,):
            parent.term = Term('ret', (value,))
        else:
            parent.term.targets = tuple(zero if b is old else b for b in parent.term.targets)
    fn.blocks.remove(old)
    fn.shared_return = None
    fn.return_joins = [b for b in getattr(fn, 'return_joins', ()) if b is not old]
    fn.cold = set(getattr(fn, 'cold', ())) - {old}
    fn.layout_owners = owners = {b: p for b, p in owners.items() if b is not old and p is not old}
    prefix_return = bodies[0].term.targets[0]
    fn.aligned_search_prefix_return = prefix_return
    for block in fn.blocks:
        for op in block.ops:
            if op.kind == 'phi' and op.results[0].type.kind == 'ptr':
                prefer(op.results[0], cursor_reg)
            if op.kind == 'intrinsic' and op.attrs['builtin'].op in ('ctz_nonzero', 'tzcnt'):
                prefer(op.results[0], 'rax')
            if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'movemask8':
                prefer(op.results[0], remaining_reg if block in bodies else aligned_reg if block in loops else 'rax')
        if block is prefix_return:
            for op in block.ops:
                if op.kind == 'phi' and op.results[0].type.kind == 'int':
                    prefer(op.results[0], remaining_reg)
        if block.term.kind == 'ret' and block.term.args and block is not prefix_return:
            address = fn.defs[bare(block.term.args[0])]
            if address.kind == 'ptradd' and fn.defs[bare(address.args[0])].kind == 'phi':
                prefer(address.results[0], cursor_reg)
    bulk = next((head for head in loops if sum('read' in op.attrs.get('memory', '') for op in head.ops) == 4), None)
    small = next((head for head in loops if sum('read' in op.attrs.get('memory', '') for op in head.ops) == 1), None)
    if bulk is None or small is None:
        return True
    # The four comparisons keep their masks for the cold hit path. Schedule
    # each aligned load next to its comparison before reducing the pairs.
    compares = [op for op in bulk.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'cmpeq_epi8']
    positions = {op: n for n, op in enumerate(bulk.ops)}
    compares.sort(key=lambda op: positions[fn.defs[op.args[0]]])
    original, seen, output = set(bulk.ops), set(), []
    def visit(op):
        if op not in original or op in seen:
            return
        for value in op.args:
            visit(fn.defs[value])
        seen.add(op)
        output.append(op)
    for op in bulk.ops:
        if op.kind == 'phi':
            visit(op)
    for op in compares:
        visit(op)
    for op in bulk.ops:
        visit(op)
    bulk.ops = output
    if width == 32:
        for op in bulk.ops:
            if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'or':
                fn.vector_registers[op.results[0]] = 'xmm6' if op.args == (compares[0].results[0], compares[1].results[0]) else 'xmm5'
    hit, single = bulk.term.targets[0], small.term.targets[0]
    first = hit.term.targets[0]
    tail = next((block for block in fn.blocks if block.term.kind == 'br'
                 and any(op.kind == 'intrinsic' and op.attrs['builtin'].op == 'load'
                         and op.attrs['alignment'] == 1 for op in block.ops)), None)
    if tail is None:
        return True
    last = tail.term.targets[0]
    if width == 32:
        for op in tail.ops:
            if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'cmpeq_epi8':
                fn.vector_registers[op.results[0]] = 'xmm0'
    def returned_parts(block):
        if block.term.kind != 'ret' or len(block.term.args) != 1:
            return None
        address = fn.defs[bare(block.term.args[0])]
        if address.kind != 'ptradd':
            return None
        index = fn.defs[address.args[1]]
        if index.kind != 'cast':
            return None
        scan = fn.defs[index.args[0]]
        if scan.kind != 'intrinsic' or scan.attrs['builtin'].op not in ('ctz_nonzero', 'tzcnt'):
            return None
        return address, scan
    a, b = returned_parts(first), returned_parts(last)
    if a is not None and b is not None:
        shared = fn.block('search.return')
        mask = fn.make('phi', [a[1].args[0].type], [a[1].args[0], b[1].args[0]], {'blocks': [first, last]})
        pointer = fn.make('phi', [a[0].args[0].type], [a[0].args[0], b[0].args[0]], {'blocks': [first, last]})
        count = fn.make('intrinsic', [a[1].results[0].type], mask.results, dict(a[1].attrs))
        index = fn.make('cast', [T.LONG], count.results)
        address = fn.make('ptradd', [a[0].results[0].type], [pointer.results[0], index.results[0]])
        value = fn.make('cast', [fn.type.base], address.results)
        shared.ops = [mask, pointer, count, index, address, value]
        shared.term = Term('ret', value.results)
        first.term = Term('jmp', targets=(shared,))
        last.term = Term('jmp', targets=(shared,))
        prefer(mask.results[0], 'rax')
        prefer(count.results[0], 'rax')
        prefer(pointer.results[0], cursor_reg)
        prefer(address.results[0], cursor_reg)
        owners[shared] = hit
        owners[single] = shared
        owners[hit.term.targets[1]] = single
    owners[hit] = zero
    # Shift counts observe only their low five bits. Keep 16-remainder in
    # byte registers, after computing the final vector's equality mask.
    for op in list(tail.ops):
        if op.kind != 'shr':
            continue
        amount = fn.defs[op.args[1]]
        if amount.kind != 'sub' or fn.defs[amount.args[0]].kind != 'constant' or fn.defs[amount.args[0]].attrs['value'] != width:
            continue
        low = fn.make('cast', [T.UCHAR], [amount.args[1]])
        difference = (fn.make('sub', [T.UCHAR], [fn.constant(T.UCHAR, width), low.results[0]]) if width == 16
                      else fn.make('neg', [T.UCHAR], low.results))
        at = tail.ops.index(amount)
        tail.ops[at:at] = [low, difference]
        amount.kind, amount.args = 'cast', difference.results
        prefer(low.results[0], remaining_reg)
        prefer(difference.results[0], 'rcx' if width == 16 else remaining_reg)
        fn.scalar_constant_registers = dict(getattr(fn, 'scalar_constant_registers', {}))
        fn.scalar_constant_registers[fn.constant(T.UCHAR, width)] = 'rcx'
        # These arithmetic operations are independent of the vector compare.
        # Place them just before the shift, after its scalar mask is ready.
        ordered = [item for item in tail.ops if item not in (low, difference, amount)]
        at = ordered.index(op)
        ordered[at:at] = [low, difference, amount]
        tail.ops = ordered
    return True
