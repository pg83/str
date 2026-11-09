"""Typed values and explicit basic blocks before x86 instruction selection.

Values have one definition. Memory operations remain ordered in a block;
their effect and access properties survive until selection. Branches name
blocks rather than byte offsets.
"""
from dataclasses import dataclass, field
from copy import copy

from . import types as T


@dataclass(frozen=True)
class Value:
    id: int
    type: T.Type

    def __str__(self):
        return '%%%d' % self.id


@dataclass(eq=False)
class Op:
    kind: str
    results: tuple = ()
    args: tuple = ()
    attrs: dict = field(default_factory=dict)
    source: object = None


@dataclass(eq=False)
class Term:
    kind: str
    args: tuple = ()
    targets: tuple = ()
    source: object = None
    weights: tuple | None = None


@dataclass(eq=False)
class Block:
    id: int
    name: str
    ops: list = field(default_factory=list)
    term: Term | None = None
    unroll_disabled: bool = False
    fixed_trip_count: int | None = None

    def __str__(self):
        return '%s.%d' % (self.name, self.id)


class Function:
    def __init__(self, decl):
        self.decl = decl
        self.name, self.type = decl.name, decl.type
        self.blocks, self.params, self.allocas, self.constants = [], [], [], []
        self.defs, self.next_value, self.next_block, self.constant_values = {}, 0, 0, {}
        self.entry = self.block('entry')
        for n, p in enumerate(decl.params):
            op = self.make('param', [p.type], attrs={'index': n, 'name': p.name}, source=p.token)
            self.params.append(op.results[0])

    def block(self, name):
        b = Block(self.next_block, name)
        self.next_block += 1
        self.blocks.append(b)
        return b

    def make(self, kind, types=(), args=(), attrs=None, source=None):
        values = tuple(Value(self.next_value + i, t.plain) for i, t in enumerate(types))
        self.next_value += len(values)
        op = Op(kind, values, tuple(args), attrs or {}, source)
        self.defs.update((v, op) for v in values)
        return op

    def constant(self, type_, value):
        if type_.kind in ('int', 'ptr'):
            value = T.convert_integer(value, type_)
        key = (type_.plain, value)
        if key not in self.constant_values:
            op = self.make('constant', [type_], attrs={'value': value})
            self.constants.append(op)
            self.constant_values[key] = op.results[0]
        return self.constant_values[key]

    def undef(self, type_):
        op = self.make('undef', [type_])
        self.constants.append(op)
        return op.results[0]

    def predecessors(self):
        pred = {b: [] for b in self.blocks}
        for b in self.blocks:
            for target in set(b.term.targets if b.term else ()):
                pred[target].append(b)
        return pred

    def reachable(self):
        seen, todo = set(), [self.entry]
        while todo:
            b = todo.pop()
            if b in seen:
                continue
            seen.add(b)
            if b.term:
                todo.extend(b.term.targets)
        return [b for b in self.blocks if b in seen]

    def verify(self):
        blocks = self.reachable()
        pred = self.predecessors()
        dom = dominators(self, blocks, pred)
        locations = {}
        for op in self.constants + self.allocas:
            locations.update((v, (self.entry, -1)) for v in op.results)
        for v in self.params:
            locations[v] = (self.entry, -1)
        for b in blocks:
            if b.term is None:
                raise ValueError('%s: unterminated block %s' % (self.name, b))
            for i, op in enumerate(b.ops):
                for v in op.results:
                    if v in locations:
                        raise ValueError('%s: duplicate definition of %s' % (self.name, v))
                    locations[v] = (b, i)

        def use(v, block, pos):
            if v not in locations:
                raise ValueError('%s: undefined value %s in %s' % (self.name, v, block))
            owner, at = locations[v]
            if owner not in dom[block] or owner is block and at >= pos:
                raise ValueError('%s: %s does not dominate its use in %s' % (self.name, v, block))

        for b in blocks:
            for i, op in enumerate(b.ops):
                if op.kind == 'phi':
                    incoming = op.attrs['blocks']
                    if len(incoming) != len(op.args) or set(incoming) != set(pred[b]):
                        raise ValueError('%s: phi predecessors differ in %s' % (self.name, b))
                    for p, v in zip(incoming, op.args):
                        if v.type != op.results[0].type:
                            raise ValueError('%s: phi type differs in %s' % (self.name, b))
                        use(v, p, len(p.ops))
                else:
                    for v in op.args:
                        use(v, b, i)
            for v in b.term.args:
                use(v, b, len(b.ops))

    def dump(self):
        lines = ['function %s(%s) -> %s' % (self.name, ', '.join('%s:%s' % (v, v.type) for v in self.params), self.type.base)]
        for op in self.constants + self.allocas:
            lines.append('  ' + format_op(op))
        for b in self.blocks:
            lines.append(str(b) + ':')
            lines.extend('  ' + format_op(op) for op in b.ops)
            t = b.term
            lines.append('  ' + (t.kind + ' ' + ', '.join(map(str, (*t.args, *t.targets))) if t else '<unterminated>'))
        return '\n'.join(lines)


def format_op(op):
    values = ', '.join('%s:%s' % (v, v.type) for v in op.results)
    def fmt(key, value):
        if key == 'blocks':
            return ','.join(map(str, value))
        if key in ('declaration', 'builtin'):
            return value.name
        if isinstance(value, bytes):
            return value.hex()
        return str(value)
    attrs = ', '.join('%s=%s' % (k, fmt(k, v)) for k, v in op.attrs.items())
    return (values + ' = ' if values else '') + op.kind + ' ' + ', '.join(map(str, op.args)) + (' {' + attrs + '}' if attrs else '')


def copy_function(fn):
    """An independent SSA graph for call-site cost and specialization probes."""
    result = copy(fn)
    blocks = {b: copy(b) for b in fn.blocks}
    ops = {op: Op(op.kind, op.results, op.args, dict(op.attrs), op.source) for op in fn.defs.values()}
    for op in ops.values():
        if op.kind == 'phi':
            op.attrs['blocks'] = [blocks.get(b, b) for b in op.attrs['blocks']]
    for old, block in blocks.items():
        block.ops = [ops[op] if op in ops else Op(op.kind, op.results, op.args, dict(op.attrs), op.source) for op in old.ops]
        block.term = Term(old.term.kind, old.term.args, tuple(blocks[b] for b in old.term.targets), old.term.source, old.term.weights)
    result.blocks, result.entry = [blocks[b] for b in fn.blocks], blocks[fn.entry]
    result.defs = {v: ops[op] for v, op in fn.defs.items()}
    result.constants, result.allocas = [ops[op] for op in fn.constants], [ops[op] for op in fn.allocas]
    result.constant_values = dict(fn.constant_values)
    if hasattr(fn, 'shared_return'):
        result.shared_return = blocks.get(fn.shared_return)
    return result


def dominators(fn, blocks=None, pred=None):
    blocks = blocks or fn.reachable()
    pred = pred or fn.predecessors()
    dom = {b: {b} if b is fn.entry else set(blocks) for b in blocks}
    changed = True
    while changed:
        changed = False
        for b in blocks:
            if b is fn.entry:
                continue
            incoming = [dom[p] for p in pred[b] if p in dom]
            new = {b} | (set.intersection(*incoming) if incoming else set())
            if new != dom[b]:
                dom[b], changed = new, True
    return dom


def promote(fn):
    """Promote nonescaping scalar/vector stack slots to SSA, including loops."""
    fn.blocks = fn.reachable()
    pred = fn.predecessors()
    dom = dominators(fn, fn.blocks, pred)
    idom = {b: max(dom[b] - {b}, key=lambda d: len(dom[d])) for b in fn.blocks if b is not fn.entry}
    children, frontier = {b: [] for b in fn.blocks}, {b: set() for b in fn.blocks}
    for b, parent in idom.items():
        children[parent].append(b)
    for b in fn.blocks:
        if len(pred[b]) < 2:
            continue
        for p in pred[b]:
            while p is not idom.get(b):
                frontier[p].add(b)
                p = idom.get(p)
                if p is None:
                    break
    slots = {op.results[0]: op for op in fn.allocas if op.results[0].type.base.kind in ('int', 'ptr', 'vector')}
    writes = {s: set() for s in slots}
    for b in fn.blocks:
        for op in b.ops:
            for i, v in enumerate(op.args):
                if v in slots:
                    if i != 0 or op.kind not in ('load', 'store') or op.attrs.get('volatile'):
                        slots.pop(v)
                    elif (op.results[0].type if op.kind == 'load' else op.args[1].type) != v.type.base.plain:
                        slots.pop(v)
                    elif op.kind == 'store':
                        writes[v].add(b)
        for v in b.term.args:
            slots.pop(v, None)
    phis = {}
    for slot in slots:
        todo, seen = list(writes[slot]), set()
        while todo:
            b = todo.pop()
            for target in sorted(frontier[b], key=lambda b: b.id):
                if target in seen:
                    continue
                seen.add(target)
                phi = fn.make('phi', [slot.type.base], attrs={'blocks': [], 'slot': slot.id})
                target.ops.insert(0, phi)
                phis[(target, slot)] = phi
                if target not in writes[slot]:
                    todo.append(target)
    stacks = {s: [fn.undef(s.type.base)] for s in slots}
    replacements = {}

    def resolve(v):
        while v in replacements:
            v = replacements[v]
        return v

    def rename(b):
        pushed = []
        ops = []
        for s in slots:
            if (b, s) in phis:
                stacks[s].append(phis[(b, s)].results[0])
                pushed.append(s)
        for op in b.ops:
            op.args = tuple(resolve(v) for v in op.args)
            s = op.args[0] if op.args else None
            if op.kind == 'load' and s in slots:
                replacements[op.results[0]] = stacks[s][-1]
            elif op.kind == 'store' and s in slots:
                stacks[s].append(op.args[1])
                pushed.append(s)
            else:
                ops.append(op)
        b.ops = ops
        b.term.args = tuple(resolve(v) for v in b.term.args)
        for target in set(b.term.targets):
            for s in slots:
                if (target, s) in phis:
                    phi = phis[(target, s)]
                    phi.args += (resolve(stacks[s][-1]),)
                    phi.attrs['blocks'].append(b)
        for child in children[b]:
            rename(child)
        for s in reversed(pushed):
            stacks[s].pop()

    rename(fn.entry)
    for b in fn.blocks:
        for op in b.ops:
            op.args = tuple(resolve(v) for v in op.args)
        b.term.args = tuple(resolve(v) for v in b.term.args)
    fn.allocas = [op for op in fn.allocas if op.results[0] not in slots]
    simplify_phis(fn)


def simplify_phis(fn):
    replacements, changed = {}, True

    def resolve(v):
        while v in replacements:
            v = replacements[v]
        return v

    while changed:
        changed = False
        for b in fn.blocks:
            ops = []
            for op in b.ops:
                op.args = tuple(resolve(v) for v in op.args)
                if op.kind == 'phi':
                    other = {v for v in op.args if v != op.results[0]}
                    if len(other) == 1:
                        replacements[op.results[0]] = other.pop()
                        changed = True
                        continue
                ops.append(op)
            b.ops = ops
    for b in fn.blocks:
        for op in b.ops:
            op.args = tuple(resolve(v) for v in op.args)
        b.term.args = tuple(resolve(v) for v in b.term.args)
