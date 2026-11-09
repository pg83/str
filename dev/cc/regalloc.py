"""Register liveness, interference coloring and stack spill insertion."""
from collections import Counter, defaultdict
from dataclasses import replace

from .machine import Imm, Instr, Mem, R, Reg, physical, regs


GPRS = ['rax', 'rcx', 'rdx', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'rbx', 'r14', 'r15', 'r12', 'r13']
CALLEE_SAVED = {'rbx', 'rbp', 'r12', 'r13', 'r14', 'r15'}
CALLER_SAVED = {'rax', 'rcx', 'rdx', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11'}


def pointer_parameters(fn, value, seen=frozenset()):
    if value in seen:
        return set()
    op = fn.defs[value]
    if op.kind == 'param':
        return {op.attrs['index']}
    if (op.kind in ('cast', 'ptradd')
            or op.kind == 'and' and fn.defs[op.args[1]].kind == 'constant'):
        args = op.args[:1]
    elif op.kind == 'phi':
        args = op.args
    elif op.kind == 'asm' and not op.attrs['code'] and len(op.results) == len(op.args):
        args = (op.args[op.results.index(value)],)
    else:
        return None
    parts = [pointer_parameters(fn, v, seen | {value}) for v in args]
    return None if any(part is None for part in parts) else set().union(*parts)


def pools(fn):
    gprs = GPRS[:8]+['r11']+GPRS[8:] if getattr(fn.source, 'word_scans', {}) and not fn.slots else GPRS
    extra = [reg for reg in getattr(fn.source, 'extra_gprs', ()) if reg not in gprs]
    gprs = gprs[:8] + extra + gprs[8:]
    return {'gpr': [physical(r) for r in gprs],
            'vec': [physical('zmm%d' % n) for n in range(31 if fn.isa in ('evex', 'evex512') else 15)],
            'mask': [physical('k%d' % n) for n in range(7)]}


def liveness(fn):
    incoming, outgoing = {b: set() for b in fn.blocks}, {b: set() for b in fn.blocks}
    changed = True
    while changed:
        changed = False
        for b in reversed(fn.blocks):
            out = set().union(*(incoming[s] for s in b.successors))
            live = set(out)
            for i in reversed(b.ins):
                live = (live - i.defs) | i.uses
            if live != incoming[b] or out != outgoing[b]:
                incoming[b], outgoing[b], changed = live, out, True
    return incoming, outgoing


def interference(fn):
    _, outgoing = liveness(fn)
    graph, affinity, frequency = defaultdict(set), defaultdict(Counter), Counter()
    fn.destructive_copies = {move for b in fn.blocks for move, arithmetic in zip(b.ins, b.ins[1:])
                             if getattr(fn.source, 'preincrement_scans', {}) and move.mnemonic == 'mov'
                             and len(move.operands) == 2 and all(isinstance(o, R) for o in move.operands)
                             and move.operands[0].reg in arithmetic.uses & arithmetic.defs
                             and arithmetic.mnemonic not in ('mov', 'parallel')}

    def edge(a, b):
        if a != b and a.bank == b.bank:
            graph[a].add(b)
            graph[b].add(a)

    widths = {}
    for b in fn.blocks:
        for i in b.ins:
            for o in i.operands:
                if isinstance(o, R):
                    widths[o.reg] = max(widths.get(o.reg, 0), o.bits)
    for b in fn.blocks:
        live = set(outgoing[b])
        for i in reversed(b.ins):
            copies = i.copies
            if i.mnemonic in ('mov', 'movups', 'movaps', 'vmovups', 'vmovaps') and len(i.operands) == 2 and all(isinstance(o, R) for o in i.operands):
                copies = (i.operands,)
            if (i.mnemonic == 'movzb' and len(i.operands) == 2 and all(isinstance(o, R) for o in i.operands)
                    and widths[i.operands[1].reg] == 8):
                copies = (i.operands,)
            pairs = {(d.reg, s.reg) for d, s in copies if isinstance(d, R) and isinstance(s, R)}
            for a, c in pairs:
                weight = 2 if i in fn.destructive_copies else 1
                affinity[a][c] += weight
                affinity[c][a] += weight
            if (i.mnemonic in ('tzcnt', 'lzcnt', 'bsf', 'bsr') and len(i.operands) == 2
                    and all(isinstance(o, R) for o in i.operands)):
                a, c = (o.reg for o in i.operands)
                affinity[a][c] += 1
                affinity[c][a] += 1
            if i.mnemonic in ('shlx', 'shrx', 'sarx') and all(isinstance(o, R) for o in i.operands):
                a, c = (o.reg for o in i.operands[:2])
                affinity[a][c] += 1
                affinity[c][a] += 1
            search = fn.source.type.base.kind == 'ptr' and getattr(fn.source, 'return_joins', ())
            if ((getattr(fn.source, 'induction_registers', None) or search or getattr(fn.source, 'fixed_loop_exits', ())) and i.mnemonic == 'lea'
                    and len(i.operands) == 2 and isinstance(i.operands[0], R)
                    and isinstance(i.operands[1], Mem) and i.operands[1].base is not None
                    and (i.operands[1].index is None or search) and i.operands[1].slot is None and i.operands[1].symbol is None):
                a, c = i.operands[0].reg, i.operands[1].base
                affinity[a][c] += 1
                affinity[c][a] += 1
            for d in i.defs:
                graph[d]
                for value in live:
                    if (d, value) not in pairs:
                        edge(d, value)
            # Multiple results are simultaneous, even if one dies here.
            for d in i.defs:
                for other in i.defs:
                    edge(d, other)
            frequency.update(i.defs | i.uses)
            for r in i.uses:
                graph[r]
            live = (live - i.defs) | i.uses
    return graph, affinity, frequency


def color(fn):
    graph, affinity, frequency = interference(fn)
    position = {}
    for b in fn.order():
        for ins in b.ins:
            for r in sorted(ins.defs | ins.uses, key=lambda r: (r.bank, str(r.id))):
                position.setdefault(r, len(position))
    available = pools(fn)
    predicates = {i.mask for b in fn.blocks for i in b.ins if i.mask is not None}
    assigned = {r: r for r in graph if r.physical}
    return_phis = []
    source_defs = {v.id: op for v, op in fn.source.defs.items()}
    def seed(reg, preferred):
        if (reg in assigned or preferred not in available[reg.bank] or preferred in graph[reg]
                or any(assigned.get(n) == preferred for n in graph[reg])):
            return
        assigned[reg] = preferred

    for reg, preferred in fn.preferred.items():
        if reg in graph:
            seed(reg, preferred)
    if getattr(fn.source, 'shared_load_return', None) is not None:
        for block in fn.source.blocks:
            for op in block.ops:
                if op.kind != 'phi' or op.results[0].type.kind != 'ptr':
                    continue
                parameters = pointer_parameters(fn.source, op.results[0])
                if parameters is not None and len(parameters) == 1:
                    index = next(iter(parameters))
                    if index < 6:
                        seed(Reg('gpr', op.results[0].id), physical(['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9'][index]))
    for block in fn.blocks:
        for ins in block.ins:
            if (ins.mnemonic in ('pxor', 'vpxor', 'vpxord') and ins.operands
                    and all(isinstance(o, R) for o in ins.operands) and len({o.reg for o in ins.operands}) == 1):
                reg = ins.operands[0].reg
                for preferred in available['vec']:
                    seed(reg, preferred)
                    if reg in assigned:
                        break
    tail_values = set(getattr(fn, 'vector_tail_values', ())) & graph.keys()
    def stream_order(reg):
        op, parent = source_defs.get(reg.id), reg
        if op is not None and op.kind == 'intrinsic' and op.attrs['builtin'].op == 'xor':
            parent = next((Reg('vec', v.id) for v in op.args if Reg('vec', v.id) in tail_values), reg)
        return -position[parent], parent != reg, -position[reg]
    for reg in sorted(tail_values, key=stream_order):
        for preferred in available['vec']:
            seed(reg, preferred)
            if reg in assigned:
                break
    pending = sorted((r for r in tail_values if r in assigned), key=stream_order, reverse=True)
    while pending:
        reg = pending.pop()
        for neighbor in sorted(affinity[reg], key=lambda r: (-affinity[reg][r], position.get(r, -1))):
            if neighbor.bank == 'vec' and not neighbor.physical and neighbor not in assigned:
                seed(neighbor, assigned[reg])
                if neighbor in assigned:
                    pending.append(neighbor)
    if any(i.mnemonic == 'call' for b in fn.blocks for i in b.ins):
        callers = {physical(r) for r in CALLER_SAVED} & set(available['gpr'])
        for value in reversed(fn.source.params):
            reg = Reg('gpr', value.id)
            if reg not in graph or reg in assigned or not callers <= graph[reg]:
                continue
            for preferred in available['gpr']:
                if preferred.id in CALLEE_SAVED:
                    seed(reg, preferred)
                    if reg in assigned:
                        break
    for b in fn.order():
        for a, c in zip(b.ins, b.ins[1:]):
            if (c.mnemonic == 'ret' and a.mnemonic == 'mov' and len(a.operands) == 2
                    and all(isinstance(o, R) for o in a.operands) and a.operands[0].reg.physical):
                seed(a.operands[1].reg, a.operands[0].reg)
                if getattr(fn.source, 'shared_return', None) is not None:
                    result = a.operands[1].reg
                    for neighbor, weight in sorted(affinity[result].items(), key=lambda pair: (
                            source_defs.get(pair[0].id) is not None and source_defs[pair[0].id].kind == 'phi',
                            -pair[1], position.get(pair[0], 0))):
                        if not neighbor.physical:
                            seed(neighbor, a.operands[0].reg)
                            if neighbor not in assigned and source_defs.get(neighbor.id) is not None and source_defs[neighbor.id].kind == 'phi':
                                return_phis.append(neighbor)
    factored = getattr(fn.source, 'return_joins', ())
    copy_chains = bool(factored or getattr(fn.source, 'induction_registers', None))
    for block in factored:
        for op in block.ops:
            if op.kind != 'phi':
                continue
            if fn.source.type.base.kind == 'int' and op.results[0].type.kind == 'ptr':
                parameters = pointer_parameters(fn.source, op.results[0])
                if parameters is not None and len(parameters) == 1:
                    index = next(iter(parameters))
                    if index < 6:
                        seed(Reg('gpr', op.results[0].id), physical(['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9'][index]))
                continue
            if fn.source.type.base.kind == 'ptr' and op.results[0].type.kind != 'ptr':
                continue
            reg = Reg('gpr', op.results[0].id)
            if getattr(fn.source, 'distance_returns', False) and Reg('mask', op.results[0].id) in graph:
                continue
            if reg in graph:
                choices = available['gpr'][1:] + available['gpr'][:1]
                if (fn.source.type.base.kind == 'int' and fn.source.type.base.bits == 64 and op.results[0].type.bits <= 32
                        and any(user.kind == 'intrinsic' and user.attrs['builtin'].op in ('ctz_nonzero', 'tzcnt')
                                and op.results[0] in user.args for b in fn.source.blocks for user in b.ops)):
                    choices = available['gpr']
                if fn.source.type.base.kind == 'ptr':
                    choices = sorted(choices, key=lambda r: r.id != 'r8')
                for preferred in choices:
                    seed(reg, preferred)
                    if reg in assigned:
                        break
    # Preserve return and common-tail registers through their copy chains.
    changed = True
    while changed:
        changed = False
        for b in fn.blocks:
            for ins in b.ins:
                copies = ins.copies if copy_chains else ()
                if (ins.mnemonic in (('mov', 'movzb', 'tzcnt', 'lzcnt', 'bsf', 'bsr') if factored else ('mov', 'movzb'))
                        and len(ins.operands) == 2 and all(isinstance(o, R) for o in ins.operands)):
                    copies = (ins.operands,)
                for d, s in copies:
                    if not isinstance(d, R) or not isinstance(s, R):
                        continue
                    source = source_defs.get(s.reg.id)
                    if d.reg.physical or s.reg.physical or not copy_chains and source is not None and source.kind == 'phi':
                        continue
                    if (ins.mnemonic == 'parallel' and getattr(fn.source, 'induction_registers', None)
                            and source is not None and source.kind == 'ptradd' and frequency[s.reg] == 2):
                        continue
                    if (ins.mnemonic in ('tzcnt', 'lzcnt', 'bsf', 'bsr') and source is not None and source.kind == 'phi'
                            and not getattr(fn.source, 'distance_returns', False)):
                        continue
                    if (ins.mnemonic in ('tzcnt', 'lzcnt', 'bsf', 'bsr') and source is not None and source.kind == 'not'
                            and getattr(fn.source, 'preincrement_scans', {})):
                        continue
                    if (ins.mnemonic in ('tzcnt', 'lzcnt', 'bsf', 'bsr')
                            and any(v.id == d.reg.id for v in getattr(fn.source, 'unshared_bit_counts', ()))):
                        continue
                    preferred = assigned.get(d.reg)
                    if preferred is not None and preferred.bank == 'gpr' and (copy_chains or preferred == physical('rax')) and s.reg not in assigned:
                        seed(s.reg, preferred)
                        changed |= s.reg in assigned
    if getattr(fn.source, 'shared_return', None) is not None:
        widths = {}
        for b in fn.blocks:
            for ins in b.ins:
                for o in (*ins.operands, *(o for pair in ins.copies for o in pair)):
                    if isinstance(o, R):
                        widths[o.reg] = max(widths.get(o.reg, 0), o.bits)
        for r in sorted((r for r in graph if not r.physical and r.bank == 'gpr' and widths.get(r) == 8
                         and source_defs.get(r.id) is not None and source_defs[r.id].kind == 'phi'),
                        key=lambda r: (-frequency[r], position[r])):
            choices = [p for p in available[r.bank] if p not in graph[r] and not any(assigned.get(n) == p for n in graph[r])]
            if choices:
                preferred = max(choices, key=lambda p: (sum(w for n, w in affinity[r].items() if assigned.get(n) == p), -available[r.bank].index(p)))
                seed(r, preferred)
        for r in return_phis:
            choices = [p for p in available[r.bank] if p not in graph[r] and not any(assigned.get(n) == p for n in graph[r])]
            if choices:
                seed(r, choices[0])
    for b in fn.order():
        for ins in b.ins:
            if (ins.mnemonic == 'mov' and len(ins.operands) == 2 and all(isinstance(o, R) for o in ins.operands)
                    and not ins.operands[0].reg.physical and ins.operands[1].reg.physical and ins.operands[0].reg in graph):
                seed(ins.operands[0].reg, ins.operands[1].reg)
    first = next((i for b in fn.order() for i in b.ins if i.mnemonic == 'mov' and len(i.operands) == 2
                  and isinstance(i.operands[0], R) and isinstance(i.operands[1], Imm) and i.operands[1].symbol), None)
    if first is not None:
        reg = first.operands[0].reg
        neighbors = [n for n in graph[reg] if n not in assigned and frequency[n] > frequency[reg]]
        if neighbors:
            live = max(neighbors, key=lambda n: (frequency[n], -position[n]))
            choices = [p for p in available[live.bank] if p not in graph[live]
                       and not any(assigned.get(n) == p for n in graph[live])]
            if choices:
                preferred = max(choices, key=lambda p: (sum(weight for n, weight in affinity[live].items() if assigned.get(n) == p),
                                                         -available[live.bank].index(p)))
                seed(live, preferred)
        for preferred in available[reg.bank]:
            seed(reg, preferred)
            if reg in assigned:
                break
    todo = {r for r in graph if r not in assigned}
    copy_neighbors = defaultdict(Counter)
    for ins in fn.destructive_copies:
        d, s = ins.operands
        if d.reg.bank == s.reg.bank == 'gpr' and not d.reg.physical and not s.reg.physical:
            copy_neighbors[d.reg][s.reg] += 2
            copy_neighbors[s.reg][d.reg] += 2
    if fn.source.type.base.kind == 'ptr' and factored:
        for b in fn.blocks:
            for ins in b.ins:
                pairs = ins.copies
                if ins.mnemonic == 'mov' and len(ins.operands) == 2 and all(isinstance(o, R) for o in ins.operands):
                    pairs = (ins.operands,)
                for d, s in pairs:
                    if isinstance(d, R) and isinstance(s, R) and d.reg.bank == s.reg.bank:
                        source = source_defs.get(s.reg.id)
                        if (source is not None and source.kind == 'intrinsic' and source.attrs['builtin'].op in ('ctz_nonzero', 'tzcnt')
                                and source.args[0].type.bits == 32):
                            continue
                        copy_neighbors[d.reg][s.reg] += 1
                        copy_neighbors[s.reg][d.reg] += 1
    order = []
    chronological = (getattr(fn.source, 'shared_return', None) is not None or bool(factored)
                     or getattr(fn, 'returned_parameter', None) is not None
                     or fn.source.type.base.kind == 'ptr' and not any(i.mnemonic == 'call' for b in fn.blocks for i in b.ins)
                     and any(i.mnemonic.startswith('kmov') and isinstance(i.operands[0], R)
                             and i.operands[0].reg.bank == 'gpr' for b in fn.blocks for i in b.ins)
                     or len(getattr(fn, 'loop_headers', ())) == 1
                     and not any(i.mnemonic == 'call' for b in fn.blocks for i in b.ins))
    if chronological:
        order = sorted(todo, key=lambda r: position[r], reverse=True)
        todo.clear()
    while todo:
        easy = [r for r in todo if len(graph[r] & todo) + sum(n.physical and n in available[r.bank] for n in graph[r]) < len(available[r.bank])]
        if easy:
            r = min(easy, key=lambda r: (frequency[r], -position[r]))
        else:
            r = min(todo, key=lambda r: (frequency[r] / (len(graph[r]) + 1), -position[r]))
        order.append(r)
        todo.remove(r)
    spills = []
    for r in reversed(order):
        excluded = {assigned[n] for n in graph[r] if n in assigned}
        if r in predicates:
            excluded.add(physical('k0'))
        choices = [p for p in available[r.bank] if p not in excluded]
        if not choices:
            spills.append(r)
            continue
        preferences = Counter()
        hints = Counter()
        future = Counter()
        for neighbor, weight in copy_neighbors[r].items():
            if neighbor not in assigned and neighbor not in graph[r]:
                forbidden = {assigned[n] for n in graph[neighbor] if n in assigned}
                for p in choices:
                    if p not in forbidden:
                        future[p] += weight
        for neighbor, weight in affinity[r].items():
            if neighbor in assigned:
                preferences[assigned[neighbor]] += weight
            if neighbor in fn.preferred and neighbor not in graph[r]:
                hints[fn.preferred[neighbor]] += weight
        assigned[r] = max(choices, key=lambda p: (p == fn.preferred.get(r), hints[p], future[p], preferences[p], -available[r.bank].index(p)))
    return assigned, spills


def spill(fn, values):
    slots = {}
    for r in values:
        bits = max(64, fn.sizes[r]) if r.bank == 'gpr' else fn.sizes[r]
        size = bits // 8
        slots[r] = (fn.slot(size, min(size, 64)), bits)
    for b in fn.blocks:
        out = []
        for ins in b.ins:
            if ins.mnemonic == 'parallel':
                def spilled(o):
                    if isinstance(o, R) and o.reg in slots:
                        slot, bits = slots[o.reg]
                        return Mem(o.bits // 8, slot=slot)
                    return o
                ins.copies = tuple((spilled(d), spilled(s)) for d, s in ins.copies)
                out.append(ins)
                continue
            used = (ins.uses | ins.defs) & slots.keys()
            mapping, before, after = {}, [], []
            for r in sorted(used, key=lambda r: r.id):
                slot, bits = slots[r]
                tmp = Reg(r.bank, fn.next_reg)
                fn.next_reg += 1
                fn.sizes[tmp] = bits
                mapping[r] = tmp
                mem, reg = Mem(bits // 8, slot=slot), R(tmp, bits)
                move = 'mov' if r.bank == 'gpr' else 'kmov' + {8: 'b', 16: 'w', 32: 'd', 64: 'q'}[bits] if r.bank == 'mask' else 'movups' if bits == 128 else 'vmovups'
                if r in ins.uses:
                    before.append(Instr(move, (reg, mem), ('W', 'R')))
                if r in ins.defs:
                    after.append(Instr(move, (mem, reg), ('W', 'R')))
            ins.map_regs(mapping)
            out.extend([*before, ins, *after])
        b.ins = out


def allocate(fn):
    for _ in range(50):
        assigned, spills = color(fn)
        if not spills:
            for b in fn.blocks:
                for ins in b.ins:
                    ins.map_regs(assigned)
            return assigned
        spill(fn, spills)
    raise ValueError(fn.name + ': register allocation did not converge')
