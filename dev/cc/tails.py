"""Share nonvolatile scalar load expressions at converging return paths."""
from collections import Counter
from dataclasses import dataclass

from .ir import Term, dominators
from .facts import possible_bits
from . import types as T


PURE = {'cast', 'ptradd', 'add', 'sub', 'mul', 'and', 'or', 'xor', 'neg', 'not',
        'shl', 'shr', 'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'select', 'intrinsic', 'load'}


@dataclass
class Pattern:
    values: tuple
    op: object = None
    args: tuple = ()
    distance: bool = False
    counts: tuple = ()


def barrier(op):
    return op.kind in ('call', 'asm') or op.attrs.get('volatile') or 'write' in op.attrs.get('memory', '')


def masked32(fn, value, seen=frozenset()):
    if value in seen:
        return False
    op = fn.defs[value]
    if op.kind == 'intrinsic':
        return value.type.bits == 32 and op.attrs['builtin'].op.endswith('_masked')
    return op.kind in ('cast', 'or', 'and', 'shl', 'shr') and any(masked32(fn, v, seen | {value}) for v in op.args)


def constant_returns(fn):
    """Keep equal constant exits together after the last SSA simplification."""
    groups = {}
    for block in fn.blocks:
        if block.ops or block.term.kind != 'ret' or len(block.term.args) != 1:
            continue
        value = block.term.args[0]
        op = fn.defs[value]
        if op.kind == 'constant':
            groups.setdefault((value.type, op.attrs['value']), []).append(block)
    for blocks in groups.values():
        if len(blocks) < 2:
            continue
        join = fn.block('return.constant')
        join.term = blocks[0].term
        for block in blocks:
            block.term = Term('jmp', targets=(join,))
        fn.cold = set(getattr(fn, 'cold', ())) | {join}
    fn.verify()


def shared_store_returns(fn):
    """Place a common bit-scan and final copy after its incoming paths."""
    if fn.type.base.kind != 'ptr':
        return
    pred = fn.predecessors()
    for block in fn.blocks:
        if block.term.kind != 'ret' or len(pred[block]) < 2:
            continue
        if not any(op.kind == 'intrinsic' and op.attrs['builtin'].op == 'ctz_nonzero' for op in block.ops):
            continue
        if (sum('read' in op.attrs.get('memory', '') for op in block.ops) != 1
                or sum('write' in op.attrs.get('memory', '') for op in block.ops) != 1):
            continue
        if not hasattr(fn, 'layout_owners'):
            fn.layout_owners = {}
        fn.layout_owners[block] = pred[block][-1]
        fn.shared_store_return = block
        wide = [b for b in fn.blocks if b.term.kind == 'jmp'
                and sum('read' in op.attrs.get('memory', '') for op in b.ops) == 2
                and sum('write' in op.attrs.get('memory', '') for op in b.ops) == 2]
        copy_loops = [b for b in fn.blocks if b in b.term.targets
                      and any(op.kind == 'phi' and op.results[0].type.kind == 'int' for op in b.ops)
                      and sum('read' in op.attrs.get('memory', '') for op in b.ops) == 1
                      and sum('write' in op.attrs.get('memory', '') for op in b.ops) == 1]
        if wide and len(copy_loops) == 1:
            fn.layout_owners[wide[0]] = copy_loops[0]
        partial = [b for b in pred[block] if b.term.kind == 'br'
                   and any('write' in op.attrs.get('memory', '') for op in b.ops)]
        if len(wide) > 1 and partial:
            fn.layout_owners[wide[-1]] = partial[0]
        # Bulk copies and the overlapping final copy only transport bits.
        # Keep their move domain independent of the scanning registers.
        floating = {op.results[0] for op in block.ops if op.kind == 'intrinsic'
                    and op.attrs['builtin'].op == 'load'}
        for arm in wide:
            reads = [op for op in arm.ops if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'load']
            if len(reads) == 2:
                floating.add(reads[-1].results[0])
        def address(value):
            offset = 0
            while True:
                op = fn.defs[value]
                if op.kind == 'cast' and value.type.bits == op.args[0].type.bits:
                    value = op.args[0]
                elif op.kind == 'ptradd' and fn.defs[op.args[1]].kind == 'constant':
                    n = fn.defs[op.args[1]].attrs['value']
                    offset += n if n < 1 << 63 else n - (1 << 64)
                    value = op.args[0]
                else:
                    return (value, offset) if op.kind == 'param' else None
        for arm in fn.blocks:
            memory = [op for op in arm.ops if op.attrs.get('memory')]
            group, previous = [], None
            for n in range(0, len(memory), 2):
                pair = memory[n:n + 2]
                current = None
                if (len(pair) == 2 and all(op.kind == 'intrinsic' for op in pair)
                        and pair[0].attrs['builtin'].op == 'load' and pair[1].attrs['builtin'].op == 'store'
                        and pair[0].results[0] == pair[1].args[1]):
                    src, dst = address(pair[0].args[0]), address(pair[1].args[0])
                    if src is not None and dst is not None:
                        current = src, dst, pair[0].attrs['width']
                if (current is None or previous is not None and (current[0] != (previous[0][0], previous[0][1] + previous[2])
                        or current[1] != (previous[1][0], previous[1][1] + previous[2]) or current[2] != previous[2])):
                    if len(group) >= 2:
                        floating.update(group)
                    group = []
                if current is not None:
                    group.append(pair[0].results[0])
                previous = current
            if len(group) >= 2:
                floating.update(group)
        fn.floating_copies = floating


def share_guarded_pointer_returns(fn):
    """Share base + index, or null, after independent bit scans."""
    if fn.type.base.kind != 'ptr' or getattr(fn, 'return_joins', ()):
        return False

    def uncast(value):
        while fn.defs[value].kind == 'cast':
            value = fn.defs[value].args[0]
        return value

    groups = {}
    for block in fn.blocks:
        if block.term.kind != 'ret' or len(block.term.args) != 1 or any(barrier(op) for op in block.ops):
            continue
        selected = fn.defs[block.term.args[0]]
        if selected.kind != 'select':
            continue
        test, zero = fn.defs[selected.args[0]], fn.defs[selected.args[2]]
        address = fn.defs[uncast(selected.args[1])]
        if test.kind != 'lt' or address.kind != 'ptradd' or zero.kind != 'constant' or zero.attrs['value'] != 0:
            continue
        count = fn.defs[uncast(test.args[0])]
        if (count.kind != 'intrinsic' or count not in block.ops or count.attrs['builtin'].op != 'ctz_nonzero'
                or uncast(address.args[1]) != count.results[0] or test.args[0].type != T.ULONG):
            continue
        base = uncast(address.args[0])
        if fn.defs[base].kind != 'param' or base.type.kind != 'ptr':
            continue
        key = base, test.args[1], selected.args[2], selected.results[0].type
        groups.setdefault(key, []).append((block, test.args[0]))
    changed, pred = False, fn.predecessors()
    for (base, bound, zero, type_), members in groups.items():
        if len(members) < 2:
            continue
        blocks, counts = zip(*members)
        join = fn.block('return.guarded.address')
        phi = fn.make('phi', [T.ULONG], counts, {'blocks': list(blocks)})
        index = fn.make('cast', [T.LONG], phi.results)
        address = fn.make('ptradd', [type_], [base, index.results[0]])
        test = fn.make('lt', [T.INT], [phi.results[0], bound])
        result = fn.make('select', [type_], [test.results[0], address.results[0], zero])
        join.ops = [phi, index, address, test, result]
        join.term = Term('ret', result.results)
        fn.layout_owners = {**getattr(fn, 'layout_owners', {}), join: blocks[-1]}
        for block in blocks:
            block.term = Term('jmp', targets=(join,))
            for probe in pred[block]:
                if probe.term.kind == 'br':
                    for fallback in probe.term.targets:
                        if fallback is not block:
                            fn.layout_owners[fallback] = join
        fn.return_joins = [*getattr(fn, 'return_joins', ()), join]
        fn.guarded_pointer_returns = [*getattr(fn, 'guarded_pointer_returns', ()), join]
        changed = True
    return changed


def expose_returned_loads(fn):
    """Expose load expressions hidden behind a phi of returned differences."""
    join = getattr(fn, 'shared_return', None)
    if join is None or join not in fn.blocks or len(join.ops) != 1 or join.ops[0].kind != 'phi':
        return False
    phi = join.ops[0]
    if join.term.kind != 'ret' or join.term.args != phi.results:
        return False
    loads, constants = [], []
    for block, value in zip(phi.attrs['blocks'], phi.args):
        op = fn.defs[value]
        if op.kind == 'constant' and join in block.term.targets:
            constants.append((block, value))
            continue
        if block.term.kind != 'jmp' or block.term.targets != (join,):
            return False
        if op.kind != 'sub':
            return False
        for arg in op.args:
            source = fn.defs[arg]
            if source.kind == 'cast':
                source = fn.defs[source.args[0]]
            if source.kind != 'load' or source not in block.ops or source.attrs.get('volatile'):
                return False
        loads.append((block, value))
    if len(loads) < 2:
        return False
    for block, value in loads:
        block.term = Term('ret', (value,), source=join.term.source)
    returned = {}
    for block, value in constants:
        if value not in returned:
            target = fn.block('return.constant')
            target.term = Term('ret', (value,), source=join.term.source)
            returned[value] = target
        block.term.targets = tuple(returned[value] if b is join else b for b in block.term.targets)
    fn.blocks.remove(join)
    fn.shared_return = None
    return True


def thread_sentinel_exits(fn):
    """Bypass a merged sentinel check on edges with a known outcome."""
    changed = False
    for join in list(fn.blocks):
        tests = [op for op in join.ops if op.kind != 'phi']
        if join.term.kind != 'br' or len(tests) != 1:
            continue
        test = tests[0]
        if test.kind not in ('eq', 'ne') or test.results != join.term.args:
            continue
        value, wanted = test.args
        phi, constant = fn.defs[value], fn.defs[wanted]
        if (phi.kind != 'phi' or phi not in join.ops or not value.type.scalar
                or constant.kind != 'constant' or constant.attrs['value'] != (1 << value.type.bits)-1):
            continue
        phis = [op for op in join.ops if op.kind == 'phi']
        removed = set()
        for pred, incoming in zip(phi.attrs['blocks'], phi.args):
            if pred.term.kind != 'jmp' or pred.term.targets != (join,):
                continue
            source = fn.defs[incoming]
            equal = source.attrs['value'] == constant.attrs['value'] if source.kind == 'constant' else False if possible_bits(fn, incoming) != (1 << value.type.bits)-1 else None
            if equal is None:
                continue
            truth = equal if test.kind == 'eq' else not equal
            target = join.term.targets[0 if truth else 1]
            if target.term.kind != 'ret' or len(target.ops) > 4 or any(op.kind == 'phi' for op in target.ops):
                continue
            values = {op.results[0]: op.args[op.attrs['blocks'].index(pred)] for op in phis}
            values[test.results[0]] = fn.constant(test.results[0].type, int(truth))
            block = fn.block('sentinel.exit')
            for op in target.ops:
                clone = fn.make(op.kind, [v.type for v in op.results], [values.get(v, v) for v in op.args], dict(op.attrs), op.source)
                block.ops.append(clone)
                values.update(zip(op.results, clone.results))
            block.term = Term('ret', tuple(values.get(v, v) for v in target.term.args), source=target.term.source)
            pred.term = Term('jmp', targets=(block,))
            removed.add(pred)
        if removed:
            for op in phis:
                pairs = [(b, v) for b, v in zip(op.attrs['blocks'], op.args) if b not in removed]
                op.attrs['blocks'], op.args = [b for b, _ in pairs], tuple(v for _, v in pairs)
            changed = True
    return changed


def return_expressions(fn, source_phi=None):
    """Share byte-difference tails and pointer-distance bit scans at exits.

    Only expressions local to incoming blocks move. A source phi also lets
    exits share a computation before testing its sentinel value. Bare
    returns and tail calls retain their own exits.
    """
    exposed = expose_returned_loads(fn) if source_phi is None else False
    owners = {v: b for b in fn.blocks for op in b.ops for v in op.results}
    uses = Counter(v for b in fn.blocks for op in b.ops for v in op.args)
    uses.update(v for b in fn.blocks for v in b.term.args)
    phi_join = owners.get(source_phi.results[0]) if source_phi is not None else None
    incoming = dict(zip(source_phi.attrs['blocks'], source_phi.args)) if source_phi is not None else {}
    candidates = (list(incoming) if source_phi is not None else
                  [b for b in fn.blocks if b.term.kind == 'ret' and len(b.term.args) == 1
                   and b.term.args[0].type.kind in ('int', 'ptr')])
    if source_phi is not None and any(b.term.kind != 'jmp' or b.term.targets != (phi_join,) for b in candidates):
        return False
    def returned(block):
        return incoming[block] if source_phi is not None else block.term.args[0]
    if source_phi is None and not exposed:
        pred = fn.predecessors()
        hot = [b for b in candidates if any(p.term.weights is not None
               and p.term.weights[p.term.targets.index(b)] > 100 * p.term.weights[1-p.term.targets.index(b)]
               for p in pred[b] if p.term.kind == 'br' and p.term.targets[0] is not p.term.targets[1])]
        if hot:
            fn.hot_return_paths = hot
            candidates = [b for b in candidates if b not in hot]
    direct = [b for b in candidates if any(op.kind == 'select' for op in b.ops)
              and sum(any(v.type.kind == 'vector' for v in op.results) and 'read' in op.attrs.get('memory', '') for op in b.ops) >= 4]
    if direct:
        fn.direct_return_paths = direct
        candidates = [b for b in candidates if b not in direct]

    def pointer(value):
        while True:
            op = fn.defs[value]
            if op.kind == 'cast' and op.args[0].type.bits == value.type.bits:
                value = op.args[0]
            elif (fn.type.base.kind == 'int' and op.kind == 'and' and value.type.bits == 64 and fn.defs[op.args[1]].kind == 'constant'
                    and fn.defs[op.args[1]].attrs['value'] & (1 << 63)):
                value = op.args[0]
            else:
                return value.type.kind == 'ptr'

    def plan(blocks):
        nodes = {}

        def match(values):
            if values in nodes:
                return nodes[values]
            node = nodes[values] = Pattern(values)
            if len(set(values)) == 1:
                return node
            ops = [fn.defs[v] for v in values]
            first = ops[0]
            # When the bases differ, pass the completed addresses to the
            # shared loads. This also lets the incoming paths update their
            # pointer registers before joining the return.
            if (exposed and first.kind == 'ptradd'
                    and len({op.args[0] for op in ops if op.kind == 'ptradd'}) > 1):
                return node
            if (first.kind not in PURE or barrier(first) or len(first.results) != 1
                    or any(owners.get(v) is not b for b, v in zip(blocks, values))
                    or any(op.kind != first.kind or len(op.results) != 1 or op.results[0].type != first.results[0].type
                           or op.attrs != first.attrs or tuple(v.type for v in op.args) != tuple(v.type for v in first.args) for op in ops[1:])):
                return node
            if (first.kind == 'intrinsic' and first.attrs['builtin'].op in ('ctz_nonzero', 'tzcnt')
                    and any(masked32(fn, op.args[0]) for op in ops)):
                return node
            if exposed and first.kind == 'add' and first.results[0].type.kind == 'int':
                # Keep a completed loop index on each edge rather than
                # passing its carried and local parts through separate phis.
                for block, op in zip(blocks, ops):
                    for arg in op.args:
                        while fn.defs[arg].kind == 'cast' and arg.type.bits == fn.defs[arg].args[0].type.bits:
                            arg = fn.defs[arg].args[0]
                        if fn.defs[arg].kind == 'phi' and owners.get(arg) is not block:
                            return node
            node.op = first
            node.args = tuple(match(tuple(op.args[n] for op in ops)) for n in range(len(first.args)))
            return node

        root = match(tuple(returned(b) for b in blocks))
        common = [node for node in nodes.values() if node.op is not None]
        loads = [node for node in common if node.op.kind == 'load' and node.values[0].type.size <= 8]
        if loads and not exposed:
            def count_offset(value):
                delta = 0
                while True:
                    op = fn.defs[value]
                    if op.kind == 'cast':
                        value = op.args[0]
                    elif op.kind in ('add', 'sub') and fn.defs[op.args[1]].kind == 'constant':
                        delta += fn.defs[op.args[1]].attrs['value'] * (-1 if op.kind == 'sub' else 1)
                        value = op.args[0]
                    elif op.kind == 'intrinsic' and op.attrs['builtin'].op in ('ctz_nonzero', 'tzcnt'):
                        return delta
                    else:
                        return 0
            # Keep a bit scan's fixed displacement in each memory operand,
            # rather than adding it to an index before a shared pair of loads.
            if any(any(count_offset(v) for v in node.values) for node in nodes.values()):
                displaced = {b for b in blocks if any(count_offset(v) for op in b.ops for v in op.results)}
                fn.displaced_returns = set(getattr(fn, 'displaced_returns', ())) | displaced
                fn.scalar_return_paths = {b for b in candidates if sum(op.kind == 'load' and op.attrs['width'] == 1 for op in b.ops) == 2}
                return None
        search = (root.values[0].type.kind in ('int', 'ptr')
                  and any(node.op.kind == 'intrinsic' and node.op.attrs['builtin'].op in ('ctz_nonzero', 'tzcnt') for node in common)
                  and any(node.op.kind == 'sub' and all(pointer(v) for v in node.op.args) for node in common))
        if len(common) < 4 or len(loads) < 2 and not search:
            return None
        root.distance = search and root.values[0].type.kind == 'int' and not loads
        if root.distance:
            count = next(node for node in common if node.op.kind == 'intrinsic'
                         and node.op.attrs['builtin'].op in ('ctz_nonzero', 'tzcnt'))
            root.counts = tuple(fn.defs[v].args[0] for v in count.values)
        for n, block in enumerate(blocks):
            moved = {fn.defs[node.values[n]] for node in common}
            expected = Counter(v for op in moved for v in op.args)
            expected.update((returned(block),))
            if any(uses[v] != expected[v] for op in moved for v in op.results):
                return None
            start = min(block.ops.index(op) for op in moved)
            if any(barrier(op) for op in block.ops[start:] if op not in moved):
                return None
        return root

    used, changed = set(), False
    for first in candidates:
        if first in used:
            continue
        group, root = [first], None
        for other in candidates:
            if other in used or other in group or returned(other).type != returned(first).type:
                continue
            proposed = plan(group + [other])
            if proposed is not None:
                group.append(other)
                root = proposed
        if root is None:
            continue
        join = fn.block('return.expression')
        phis, ops, emitted = [], [], {}

        def emit(node):
            if node.values in emitted:
                return emitted[node.values]
            if len(set(node.values)) == 1:
                value = node.values[0]
            elif node.op is None:
                type_, values = node.values[0].type, node.values
                addresses = root.distance and type_.kind == 'int' and type_.bits == 64 and all(pointer(v) for v in values)
                if addresses:
                    type_, casted = T.pointer(T.VOID), []
                    for block, value in zip(group, values):
                        cast = fn.make('cast', [type_], [value])
                        block.ops.append(cast)
                        casted.append(cast.results[0])
                    values = tuple(casted)
                phi = fn.make('phi', [type_], values, {'blocks': list(group)})
                phis.append(phi)
                value = phi.results[0]
                if addresses:
                    cast = fn.make('cast', [node.values[0].type], [value])
                    ops.append(cast)
                    value = cast.results[0]
            else:
                args = tuple(emit(arg) for arg in node.args)
                clone = fn.make(node.op.kind, [node.values[0].type], args, dict(node.op.attrs), node.op.source)
                ops.append(clone)
                value = clone.results[0]
            emitted[node.values] = value
            return value

        result = emit(root)
        if root.distance:
            fn.distance_returns = True
        join.ops = phis + ops
        join.term = Term('ret', (result,), source=first.term.source) if source_phi is None else Term('jmp', targets=(phi_join,))
        if (exposed and source_phi is None and len(phis) == 1 and phis[0].results[0].type.kind == 'int'
                and phis[0].results[0].type.bits == 64 and sum(op.kind == 'load' for op in ops) == 2
                and any(b not in group and b.term.kind == 'ret' and fn.defs[returned(b)].kind == 'constant' for b in candidates)):
            fn.shared_index_return = join
        if source_phi is not None:
            pairs = []
            for block, value in zip(source_phi.attrs['blocks'], source_phi.args):
                if block is group[0]:
                    pairs.append((join, result))
                elif block not in group:
                    pairs.append((block, value))
            source_phi.attrs['blocks'], source_phi.args = [b for b, _ in pairs], tuple(v for _, v in pairs)
        for block in group:
            block.term = Term('jmp', targets=(join,))
        if not hasattr(fn, 'layout_owners'):
            fn.layout_owners = {}
        fn.layout_owners[join] = group[0] if fn.type.base.kind == 'ptr' else group[-1]
        if root.distance:
            widths = []
            for value in root.counts:
                op = fn.defs[value]
                while op.kind == 'cast':
                    value = op.args[0]
                    op = fn.defs[value]
                widths.append(op.args[0].type.size if op.kind == 'intrinsic' and op.attrs['builtin'].op == 'movemask8' else value.type.bits)
            probes = [owners.get(value, block) for value, block in zip(root.counts, group)]
            dom = dominators(fn)
            fn.layout_owners[join] = max((b for b, width in zip(probes, widths) if width == widths[0]), key=lambda b: len(dom[b]))
            fn.distance_probes = [*getattr(fn, 'distance_probes', ()), probes]
        if source_phi is not None:
            fn.layout_owners[phi_join] = join
        for block in candidates:
            if block not in group and block.term.kind == 'ret' and not block.ops and fn.defs[block.term.args[0]].kind == 'constant':
                fn.layout_owners[block] = direct[-1] if direct else join
        if not hasattr(fn, 'return_joins'):
            fn.return_joins = []
        fn.return_joins.append(join)
        used.update(group)
        changed = True
        if exposed and used == set(candidates):
            fn.shared_return = join
            fn.shared_load_return = join
    return changed or exposed


def converging_unary(fn):
    """Factor matching unary computations on a subset of an index phi's edges."""
    owners = {v: b for b in fn.blocks for op in b.ops for v in op.results}
    uses = Counter(v for b in fn.blocks for op in b.ops for v in op.args)
    uses.update(v for b in fn.blocks for v in b.term.args)
    pred, changed = fn.predecessors(), False
    for join in getattr(fn, 'return_joins', ()):
        if join not in fn.blocks:
            continue
        phis = [op for op in join.ops if op.kind == 'phi']
        if len(phis) != 1:
            continue
        phi = phis[0]
        chains = []
        for block, value in zip(phi.attrs['blocks'], phi.args):
            chain = []
            while owners.get(value) is block and uses[value] == 1:
                op = fn.defs[value]
                if op.kind not in PURE or len(op.results) != 1 or len(op.args) != 1 or op.attrs.get('memory') or barrier(op):
                    break
                if (chain and any(item.kind != 'cast' for item in chain) and op.kind == 'cast'
                        and op.results[0].type.bits == op.args[0].type.bits):
                    # Keep the input's integer rank conversion on its edge;
                    # it need not prevent sharing the actual computation.
                    break
                chain.append(op)
                value = op.args[0]
            if any(op.kind == 'intrinsic' and op.attrs['builtin'].op in ('ctz_nonzero', 'tzcnt') for op in chain) and masked32(fn, value):
                if not hasattr(fn, 'unshared_bit_counts'):
                    fn.unshared_bit_counts = set()
                fn.unshared_bit_counts.update(v for op in chain if op.kind == 'intrinsic' for v in op.results)
                chain = []
            chains.append((block, value, chain))
        selected = None
        for n, (_, _, chain) in enumerate(chains):
            if not chain or all(op.kind == 'cast' for op in chain):
                continue
            group = [n]
            for m in range(n + 1, len(chains)):
                other = chains[m][2]
                if chains[m][1].type == chains[n][1].type and len(other) == len(chain) and all(a.kind == b.kind and a.results[0].type == b.results[0].type
                                                   and a.attrs == b.attrs for a, b in zip(chain, other)):
                    group.append(m)
            if len(group) > 1:
                selected = group
                break
        if selected is None:
            continue
        group = [chains[n] for n in selected]
        if any(block.term.kind != 'jmp' or block.term.targets != (join,) for block, _, _ in group):
            continue
        common = fn.block('return.index')
        incoming = fn.make('phi', [group[0][1].type], [v for _, v, _ in group], {'blocks': [b for b, _, _ in group]})
        common.ops = [incoming]
        value = incoming.results[0]
        for op in reversed(group[0][2]):
            clone = fn.make(op.kind, [op.results[0].type], [value], dict(op.attrs), op.source)
            common.ops.append(clone)
            value = clone.results[0]
        common.term = Term('jmp', targets=(join,))
        pairs = []
        for n, pair in enumerate(zip(phi.attrs['blocks'], phi.args)):
            if n == selected[0]:
                pairs.append((common, value))
            elif n not in selected:
                pairs.append(pair)
        phi.attrs['blocks'], phi.args = [b for b, _ in pairs], tuple(v for _, v in pairs)
        for block, _, _ in group:
            block.term = Term('jmp', targets=(common,))
        owner = group[-1][0]
        moved = set(group[-1][2])
        while owner is not fn.entry and len(pred[owner]) == 1 and all(op in moved
                or op.kind == 'cast' and op.results[0].type.bits == op.args[0].type.bits for op in owner.ops):
            owner = pred[owner][0]
            if owner.term.kind != 'jmp':
                break
        fn.layout_owners[join], fn.layout_owners[common] = common, owner
        changed = True
    return changed


def share_returned_address_counts(fn):
    """Share equal bit-scan offsets before a common pair of byte loads."""
    from . import types as T
    from .facts import possible_bits
    join = getattr(fn, 'shared_load_return', None)
    if join is None or join not in fn.blocks:
        return False
    phis = [op for op in join.ops if op.kind == 'phi']
    if len(phis) != 2 or any(op.results[0].type.kind != 'ptr' for op in phis):
        return False
    uses = Counter(v for b in fn.blocks for op in b.ops for v in op.args)
    uses.update(v for b in fn.blocks for v in b.term.args)
    groups = {}
    for block in phis[0].attrs['blocks']:
        if block.term.kind != 'jmp' or block.term.targets != (join,):
            continue
        addresses = [op.args[op.attrs['blocks'].index(block)] for op in phis]
        roots = [fn.defs[v] for v in addresses]
        if any(op.kind != 'ptradd' or op not in block.ops for op in roots) or roots[0].args[1] != roots[1].args[1]:
            continue
        value, offset, moved = roots[0].args[1], 0, set(roots)
        while fn.defs[value].kind == 'cast' and fn.defs[value] in block.ops:
            moved.add(fn.defs[value])
            value = fn.defs[value].args[0]
        op = fn.defs[value]
        if op.kind == 'add' and op in block.ops:
            constant = fn.defs[op.args[1]]
            if constant.kind != 'constant' or not isinstance(constant.attrs['value'], int):
                continue
            offset = constant.attrs['value']
            if offset < 0 or offset + possible_bits(fn, op.args[0]) >= 1 << value.type.bits:
                continue
            moved.add(op)
            value = op.args[0]
            while fn.defs[value].kind == 'cast' and fn.defs[value] in block.ops:
                moved.add(fn.defs[value])
                value = fn.defs[value].args[0]
            op = fn.defs[value]
        if op.kind != 'intrinsic' or op not in block.ops or op.attrs['builtin'].op != 'ctz_nonzero':
            continue
        moved.add(op)
        expected = Counter(v for item in moved for v in item.args)
        expected.update(addresses)
        if any(uses[v] != expected[v] for item in moved for v in item.results):
            continue
        key = offset, op.results[0].type, op.args[0].type, op.attrs['builtin'].name
        groups.setdefault(key, []).append((block, op, tuple(root.args[0] for root in roots)))
    if not groups:
        return False
    narrow = sorted(groups.items(), key=lambda pair: pair[0][0])
    if (len(narrow) > 1 and all(len(incoming) == 1 and key[1].bits == 32
            and key[2].bits == 32 and key[1:] == narrow[0][0][1:]
            and incoming[0][2] == narrow[0][1][0][2]
            and key[0] + possible_bits(fn, incoming[0][1].results[0]) < 1 << 31
            for key, incoming in narrow)):
        # A narrow index can carry the fixed vector displacement once, before
        # the shared address updates, instead of updating both pointers twice.
        target = fn.block('return.offset')
        blocks, counts = [], []
        for (offset, _, _, _), incoming in narrow:
            block, original, _ = incoming[0]
            count = fn.make('intrinsic', [original.results[0].type], original.args,
                            dict(original.attrs), original.source)
            block.ops.append(count)
            value = count.results[0]
            if offset:
                kind = 'or' if not offset & possible_bits(fn, value) else 'add'
                displaced = fn.make(kind, [value.type], [value, fn.constant(value.type, offset)])
                block.ops.append(displaced)
                value = displaced.results[0]
            blocks.append(block)
            counts.append(value)
            block.term = Term('jmp', targets=(target,))
        count = fn.make('phi', [counts[0].type], counts, {'blocks': blocks})
        index = fn.make('cast', [T.LONG], count.results)
        target.ops.extend([count, index])
        for phi, base in zip(phis, narrow[0][1][0][2]):
            address = fn.make('ptradd', [base.type], [base, index.results[0]])
            target.ops.append(address)
            pairs = [(b, value) for b, value in zip(phi.attrs['blocks'], phi.args) if b not in blocks]
            pairs.append((target, address.results[0]))
            phi.attrs['blocks'], phi.args = [b for b, _ in pairs], tuple(value for _, value in pairs)
        target.term = Term('jmp', targets=(join,))
        owner = blocks[-1]
        for block in blocks[:-1]:
            fn.layout_owners[block] = owner
            owner = block
        fn.layout_owners[target], fn.layout_owners[join] = owner, target
        fn.return_address_joins = [target]
        fn.return_address_fallbacks = [b for b in fn.predecessors()[join] if b is not target]
        return True
    joins, single = [], {}
    for (offset, _, _, _), incoming in sorted(groups.items(), key=lambda pair: pair[0][0]):
        blocks = [b for b, _, _ in incoming]
        target = blocks[0] if len(blocks) == 1 else fn.block('return.address')
        if len(blocks) > 1:
            joins.append(target)
        def merge(values):
            if len(set(values)) == 1:
                return values[0]
            phi = fn.make('phi', [values[0].type], values, {'blocks': blocks})
            target.ops.append(phi)
            return phi.results[0]
        mask = merge([op.args[0] for _, op, _ in incoming])
        bases = [merge([base[n] for _, _, base in incoming]) for n in range(2)]
        original = incoming[0][1]
        count = fn.make('intrinsic', [original.results[0].type], [mask], dict(original.attrs), original.source)
        index = fn.make('cast', [T.LONG], count.results)
        target.ops.extend([count, index])
        results = []
        for base in bases:
            address = fn.make('ptradd', [base.type], [base, index.results[0]])
            target.ops.append(address)
            if offset:
                address = fn.make('ptradd', [base.type], [address.results[0], fn.constant(T.LONG, offset)])
                target.ops.append(address)
            results.append(address.results[0])
        for phi, result in zip(phis, results):
            pairs, inserted = [], False
            for old, value in zip(phi.attrs['blocks'], phi.args):
                if old not in blocks:
                    pairs.append((old, value))
                elif not inserted:
                    pairs.append((target, result))
                    inserted = True
            phi.attrs['blocks'], phi.args = [b for b, _ in pairs], tuple(v for _, v in pairs)
        for block in blocks:
            if block is not target:
                block.term = Term('jmp', targets=(target,))
        target.term = Term('jmp', targets=(join,))
        if len(blocks) == 1:
            single[offset] = target
    if joins and single:
        owner = single[max(single)]
        for block in joins:
            fn.layout_owners[block] = owner
            owner = block
        for offset, block in sorted(single.items()):
            if offset != max(single):
                fn.layout_owners[block] = owner
                owner = block
        fn.layout_owners[join] = owner
    fn.return_address_joins = joins
    paths = set(joins) | set(single.values())
    fn.return_address_fallbacks = [b for b in fn.predecessors()[join] if b not in paths]
    return True
