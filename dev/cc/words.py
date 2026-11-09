"""Canonical forms for page-bounded scalar string comparisons."""
from . import types as T
from .ir import Term, dominators
from .facts import possible_bits


def prepare_word_scans(fn, page_width=16):
    def bare(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value

    def defined(value):
        return fn.defs[bare(value)]

    def constant(value):
        op = defined(value)
        return op.attrs['value'] if op.kind == 'constant' else None

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
        for op in roots:
            visit(op)
        for op in block.ops:
            visit(op)
        block.ops = output

    pred, dom, changed = fn.predecessors(), dominators(fn), False
    for head in list(fn.blocks):
        if head.term.kind != 'br' or len(pred[head]) != 2:
            continue
        test = defined(head.term.args[0])
        if test.kind != 'le' or test.args[0].type.bits != 64 or test.args[0].type.signed:
            continue
        guard, limit = defined(test.args[0]), defined(test.args[1])
        if guard.kind != 'add' or guard not in head.ops or constant(guard.args[1]) != 8 or limit.kind != 'select':
            continue
        index = defined(guard.args[0])
        if index.kind != 'phi' or index not in head.ops or set(head.ops) != {index, guard, test}:
            continue
        n1, n2 = defined(limit.args[2]), defined(limit.args[1])
        comparison = defined(limit.args[0])
        if (n1.kind != n2.kind or n1.kind != 'sub' or constant(n1.args[0]) != 4096 or n1.args[0] != n2.args[0]
                or comparison.kind != 'gt' or comparison.args != (n1.results[0], n2.results[0])):
            continue
        body, done = head.term.targets
        if (body.term.kind != 'br' or head not in body.term.targets or pred[body] != [head]
                or any(op.kind in ('call', 'asm') or op.attrs.get('volatile') or 'write' in op.attrs.get('memory', '') for op in body.ops)):
            continue
        stop = defined(body.term.args[0])
        if stop.kind != 'or':
            continue
        mask, difference = (defined(v) for v in stop.args)
        if mask.kind != 'and' or difference.kind != 'xor':
            continue
        pair, high = defined(mask.args[0]), mask.args[1]
        if pair.kind != 'and' or constant(high) != 0x8080808080808080:
            continue
        subtract, inverse = (defined(v) for v in pair.args)
        if (subtract.kind != 'sub' or inverse.kind != 'not' or constant(subtract.args[1]) != 0x0101010101010101
                or bare(subtract.args[0]) != inverse.args[0] or bare(difference.args[0]) != inverse.args[0]):
            continue
        x, y = defined(difference.args[0]), defined(difference.args[1])
        if (x.kind != y.kind or x.kind != 'load' or x.attrs['width'] != y.attrs['width'] or x.attrs['width'] != 8
                or {op for op in body.ops if 'read' in op.attrs.get('memory', '')} != {x, y}):
            continue
        page_guard = None
        for block in dom[head]:
            if block.term.kind != 'br':
                continue
            both = defined(block.term.args[0])
            if both.kind != 'and':
                continue
            checks = [defined(value) for value in both.args]
            if (all(op.kind == 'le' and constant(op.args[1]) == 4096 - page_width for op in checks)
                    and {bare(op.args[0]) for op in checks} == {bare(n1.args[1]), bare(n2.args[1])}):
                page_guard = block, both, checks
                break
        if page_guard is None:
            continue
        # The word is retained for xor; form the zero-byte part in two
        # independent registers and fold the second word into that xor.
        negative = fn.constant(subtract.results[0].type, -0x0101010101010101)
        subtract.kind, subtract.args = 'add', (subtract.args[0], negative)
        pair.args = inverse.results[0], high
        mask.args = subtract.results[0], pair.results[0]
        stop.args = difference.results[0], mask.results[0]
        ordered(body, [inverse, pair, subtract, mask, difference, stop, fn.defs[body.term.args[0]]])
        # Keep the second page distance as the default value of the minimum.
        comparison.kind = 'lt'
        limit.args = comparison.results[0], n1.results[0], n2.results[0]
        before = pred[head][0] if pred[head][1] is body else pred[head][1]
        ordered(before, [n1, n2, comparison, limit])
        outer, both, checks = page_guard
        checks.sort(key=lambda op: bare(op.args[0]) != bare(n1.args[1]))
        for check in checks:
            check.kind, check.args = 'ge', (check.args[0], fn.constant(check.args[1].type, 4097 - page_width))
        both.kind, both.args = 'or', tuple(op.results[0] for op in checks[::-1])
        outer.term.targets = outer.term.targets[::-1]
        if outer.term.weights is not None:
            outer.term.weights = outer.term.weights[::-1]
        ordered(outer, [*checks, both])
        fn.loop_alignments = {**getattr(fn, 'loop_alignments', {}), outer: 0}
        fn.word_scans = {**getattr(fn, 'word_scans', {}), head: (body, x, difference, pair, subtract, mask, stop)}
        fn.word_scan_outer = outer
        fn.scalar_constant_registers = {high: 'rcx', negative: 'rdx'}
        fn.induction_registers = {**getattr(fn, 'induction_registers', {}), x.results[0]: 'rax',
                                  difference.results[0]: 'rax', stop.results[0]: 'rax', inverse.results[0]: 'r10',
                                  pair.results[0]: 'r10', subtract.results[0]: 'r11', mask.results[0]: 'r11',
                                  bare(n1.args[1]): 'rax', bare(n2.args[1]): 'r9',
                                  n1.results[0]: 'r10', n2.results[0]: 'r8', limit.results[0]: 'r8',
                                  checks[0].results[0]: 'r8', checks[1].results[0]: 'r10', both.results[0]: 'r10'}
        # Compare vectors in the same order: difference, inversion, NUL test.
        vector = outer.term.targets[1]
        candidates = [op for op in vector.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'xor']
        for inversion in candidates:
            equal, ones = defined(inversion.args[0]), defined(inversion.args[1])
            if (equal.kind != 'intrinsic' or equal.attrs['builtin'].op != 'cmpeq_epi8'
                    or ones.kind != 'constant' or ones.attrs['value'] != b'\xff'*16):
                continue
            zero = next((op for op in vector.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'cmpeq_epi8'
                         and any(constant(arg) == bytes(16) for arg in op.args)), None)
            if zero is not None:
                ordered(vector, [equal, inversion, zero, fn.defs[vector.term.args[0]]])
        changed = True
    return changed


def split_word_scan_returns(fn):
    if not getattr(fn, 'word_scans', {}):
        return False
    join = getattr(fn, 'shared_return', None)
    if join is None or join not in fn.blocks or len(join.ops) != 1 or join.ops[0].kind != 'phi':
        return False
    phi = join.ops[0]
    if join.term.kind != 'ret' or join.term.args != phi.results:
        return False
    for parent, value in zip(phi.attrs['blocks'], phi.args):
        if parent.term.kind == 'jmp':
            parent.term = Term('ret', (value,))
        else:
            returned = fn.block('scan.return')
            returned.term = Term('ret', (value,))
            parent.term.targets = tuple(returned if target is join else target for target in parent.term.targets)
    fn.blocks.remove(join)
    fn.shared_return = None
    fn.return_joins = [block for block in getattr(fn, 'return_joins', ()) if block is not join]
    return True


def normalize_word_indices(fn):
    def bare(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value

    changed = False
    for head, (body, *_) in list(getattr(fn, 'word_scans', {}).items()):
        pred, dom = fn.predecessors(), dominators(fn)
        test = fn.defs[head.term.args[0]]
        guard, bound = fn.defs[test.args[0]], test.args[1]
        phi = fn.defs[guard.args[0]]
        before = next(b for b in pred[head] if b is not body)
        done = head.term.targets[1]
        if (phi.kind != 'phi' or fn.defs[phi.args[phi.attrs['blocks'].index(before)]].kind != 'constant'
                or fn.defs[phi.args[phi.attrs['blocks'].index(before)]].attrs['value'] != 0
                or possible_bits(fn, bound) >= 1 << 31):
            continue
        old = phi.results[0]
        if any(old in op.args and done not in dom[b] and body not in dom[b]
               for b in fn.blocks if b not in (head, body) for op in b.ops):
            continue
        remainder = fn.make('and', [old.type], [bound, fn.constant(old.type, -8)])
        done.ops.insert(0, remainder)
        for block in fn.blocks:
            if done not in dom[block]:
                continue
            for op in block.ops:
                op.args = tuple(remainder.results[0] if value == old else value for value in op.args)
            block.term.args = tuple(remainder.results[0] if value == old else value for value in block.term.args)
        # Only a continuing iteration advances the look-ahead index.
        step = fn.block('word.step')
        advanced = fn.make('add', [old.type], [guard.results[0], guard.args[1]])
        again = fn.make('le', [T.INT], [advanced.results[0], bound])
        step.ops = [advanced, again]
        step.term = Term('br', again.results, (body, done))
        fn.layout_owners = {**getattr(fn, 'layout_owners', {}), done: step}
        body.term.targets = tuple(step if target is head else target for target in body.term.targets)
        phi.kind, phi.args, phi.attrs = 'sub', (guard.results[0], guard.args[1]), {}
        guard.kind, guard.args, guard.attrs = 'phi', (fn.constant(old.type, 8), advanced.results[0]), {'blocks': [head, step]}
        test.kind, test.args = 'ge', (bound, fn.constant(bound.type, 8))
        head.ops = [test]
        body.ops[0:0] = [guard, phi]
        head.term.targets = body, done
        fn.induction_registers.update({guard.results[0]: 'r9', advanced.results[0]: 'r9', remainder.results[0]: 'r9'})
        fn.narrow_word_values = {*getattr(fn, 'narrow_word_values', ()), remainder}
        # The byte loop begins at floor(n / 8) * 8, which cannot exceed n.
        for op in done.ops:
            if op.kind == 'lt' and op.args == (remainder.results[0], bound):
                op.kind = 'ne'
                fn.narrow_word_comparisons = {*getattr(fn, 'narrow_word_comparisons', ()), op}
        for block in fn.blocks:
            for op in block.ops:
                if op.kind != 'phi' or op.results[0].type != old.type or remainder.results[0] not in op.args:
                    continue
                byte_index = op.results[0]
                fn.induction_registers[byte_index] = 'r9'
                for value in op.args:
                    update = fn.defs[value]
                    if update.kind != 'add' or update.args[0] != byte_index:
                        continue
                    fn.induction_registers[value] = 'r9'
                    for b in fn.blocks:
                        for comparison in b.ops:
                            if comparison.kind == 'lt' and comparison.args == (value, bound):
                                comparison.kind, comparison.args = 'ne', (bound, value)
                loaded = [item for item in block.ops if item.kind == 'load' and item.attrs['width'] == 1]
                if len(loaded) == 2:
                    fn.induction_registers[loaded[0].results[0]] = 'rax'
                    fn.induction_registers[loaded[1].results[0]] = 'r10'
                    fn.word_byte_blocks = {*getattr(fn, 'word_byte_blocks', ()), block}
        # On the word's return edge, add the local byte offset to each base
        # before using the carried word index in the two byte loads.
        for block in fn.blocks:
            if body not in dom.get(block, ()) or block.term.kind != 'ret':
                continue
            totals = [op for op in block.ops if op.kind == 'add' and old in op.args]
            for total in totals:
                offset = total.args[1 if total.args[0] == old else 0]
                roots = [op for op in block.ops if op.kind == 'ptradd' and bare(op.args[1]) == total.results[0]]
                if len(roots) != 2:
                    continue
                cast = fn.make('cast', [T.LONG], [offset])
                index = fn.make('cast', [T.LONG], guard.results)
                relative = fn.make('add', [T.LONG], [index.results[0], fn.constant(T.LONG, -8)])
                at = min(block.ops.index(op) for op in roots)
                block.ops[at:at] = [cast, index, relative]
                for reg, op in zip(('rdi', 'rsi'), roots):
                    base = fn.make('ptradd', [op.results[0].type], [op.args[0], cast.results[0]])
                    block.ops.insert(block.ops.index(op), base)
                    op.args = base.results[0], relative.results[0]
                    fn.induction_registers[base.results[0]] = reg
                    fn.word_return_bases = {*getattr(fn, 'word_return_bases', ()), base.results[0]}
                scan = fn.defs[offset]
                fn.induction_registers[offset] = 'rcx'
                while scan.kind in ('cast', 'div', 'shr'):
                    fn.induction_registers[scan.args[0]] = 'rcx'
                    scan = fn.defs[scan.args[0]]
        fn.word_scans.pop(head)
        fn.word_scans[body] = (body, *_[0:])
        changed = True
    return changed


def share_word_scan_steps(fn):
    outer = getattr(fn, 'word_scan_outer', None)
    if outer is None:
        return False
    phis = [op for op in outer.ops if op.kind == 'phi']
    if len(phis) != 2 or any(op.results[0].type.kind != 'ptr' for op in phis):
        return False
    def bare(value):
        while fn.defs[value].kind == 'cast' and value.type.bits == fn.defs[value].args[0].type.bits:
            value = fn.defs[value].args[0]
        return value
    choices = []
    for parent in phis[0].attrs['blocks']:
        if parent.term.kind != 'jmp' or parent.term.targets != (outer,):
            continue
        roots = [fn.defs[phi.args[phi.attrs['blocks'].index(parent)]] for phi in phis]
        if (all(op.kind == 'ptradd' and bare(op.args[0]) == phi.results[0] for op, phi in zip(roots, phis))
                and bare(roots[0].args[1]) == bare(roots[1].args[1])):
            choices.append((parent, roots[0].args[1]))
    if len(choices) != 2:
        return False
    join = fn.block('scan.advance')
    count = fn.make('phi', [T.LONG], [value for _, value in choices], {'blocks': [block for block, _ in choices]})
    updates = [fn.make('ptradd', [phi.results[0].type], [phi.results[0], count.results[0]]) for phi in phis]
    join.ops = [count, *updates]
    join.term = Term('jmp', targets=(outer,))
    replaced = {block for block, _ in choices}
    for phi, update in zip(phis, updates):
        incoming = [(block, value) for block, value in zip(phi.attrs['blocks'], phi.args) if block not in replaced]
        incoming.append((join, update.results[0]))
        phi.attrs['blocks'], phi.args = [block for block, _ in incoming], tuple(value for _, value in incoming)
    for parent, _ in choices:
        parent.term.targets = (join,)
    fn.induction_registers[count.results[0]] = 'r8'
    for phi, update in zip(phis, updates):
        origin = bare(phi.args[0])
        param = fn.defs[origin]
        if param.kind == 'param' and param.attrs['index'] < 2:
            fn.induction_registers[update.results[0]] = ('rdi', 'rsi')[param.attrs['index']]
    join.ops[1:] = sorted(updates, key=lambda op: fn.induction_registers.get(op.results[0]) != 'rdi')
    vector = outer.term.targets[1]
    fn.word_scan_advance = join
    fn.word_scan_vector = vector
    owners = dict(getattr(fn, 'layout_owners', {}))
    owners[join] = vector
    word_returns = [body.term.targets[0] for body, *_ in fn.word_scans.values() if body.term.targets[0].term.kind == 'ret']
    byte_returns, zero_returns = [], []
    for body in getattr(fn, 'word_byte_blocks', ()):
        mismatch, null = body.term.targets
        if mismatch.term.kind == 'ret' and null.term.kind == 'br':
            step, zero = null.term.targets
            if zero.term.kind == 'ret':
                byte_returns.append(mismatch)
                zero_returns.append(zero)
                owners[vector] = step
    tails = [*word_returns, *byte_returns, *zero_returns]
    if vector.term.kind == 'br' and vector.term.targets[0].term.kind == 'ret':
        tails.append(vector.term.targets[0])
    previous = join
    for returned in tails:
        owners[returned] = previous
        previous = returned
    fn.layout_owners = owners
    return True
