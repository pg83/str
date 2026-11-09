"""Keep string-comparison flags and repeated fixed-register results live."""
from collections import Counter

from . import machine as M
from .regalloc import liveness


def fold_flag_branches(fn):
    """Use a flag predicate directly when its materialized byte dies at a test."""
    if not any(ins.mnemonic.startswith('set') for block in fn.blocks for ins in block.ins):
        return
    inverse = {'e': 'ne', 'ne': 'e', 'b': 'ae', 'ae': 'b', 'a': 'be', 'be': 'a',
               'l': 'ge', 'ge': 'l', 'g': 'le', 'le': 'g'}
    flags = M.Reg('flags', 'rflags')
    _, outgoing = liveness(fn)

    def deferred_zero_copy(ins):
        # Parallel copies are expanded after allocation; a scalar zero may
        # become XOR and overwrite the predicate's flags at that point.
        return ins.mnemonic == 'parallel' and any(
            isinstance(d, M.R) and d.reg.bank == 'gpr' and isinstance(s, M.Imm)
            and s.value == 0 and s.symbol is None for d, s in ins.copies)
    while True:
        uses = Counter(r for block in fn.blocks for ins in block.ins for r in ins.uses)
        changed = False
        for block in fn.blocks:
            if flags in outgoing[block]:
                continue
            for n, test in enumerate(block.ins[:-1]):
                if (test.mnemonic != 'test' or len(test.operands) != 2 or test.operands[0] != test.operands[1]
                        or not isinstance(test.operands[0], M.R)):
                    continue
                branch = block.ins[n+1]
                if branch.target is None or branch.cc not in ('e', 'ne'):
                    continue
                reg = test.operands[0].reg
                if uses[reg] != 1:
                    continue
                setter = next((ins for ins in reversed(block.ins[:n]) if reg in ins.defs), None)
                if (setter is None or setter.preserve or not setter.mnemonic.startswith('set')
                        or setter.cc not in inverse or setter.operands[0] != test.operands[0]):
                    continue
                between = block.ins[block.ins.index(setter)+1:n]
                if any(flags in ins.defs or reg in ins.uses or ins.preserve or deferred_zero_copy(ins) for ins in between):
                    continue
                branch.cc = setter.cc if branch.cc == 'ne' else inverse[setter.cc]
                branch.mnemonic = 'j' + branch.cc
                block.ins.remove(test)
                block.ins.remove(setter)
                changed = True
                break
            if changed:
                break
        if not changed:
            return


def retain_string_indices(fn):
    """Model an identical PCMPISTRI result as retaining ECX's existing value.

    The instruction still executes to provide fresh flags. An implicit ECX
    use keeps the previous result live, allowing the allocator to coalesce
    the scalar index without letting an unrelated value occupy ECX.
    """
    rcx = M.physical('rcx')
    names = {'pcmpistri', 'vpcmpistri'}
    if not any(ins.mnemonic in names for block in fn.blocks for ins in block.ins):
        return
    pred = {block: [p for p in fn.blocks if block in p.successors] for block in fn.blocks}

    def key(ins):
        if ins.mnemonic not in names:
            return None
        if any(isinstance(o, M.Mem) and (not o.symbol or not o.symbol.startswith(('.LCPI', '.LC.')))
               for o in ins.operands):
            return None
        return ins.mnemonic, ins.operands

    def transfer(ins, available):
        if ins.mnemonic in names:
            return key(ins)
        if available is not None:
            dependencies = set().union(*(M.regs(o) for o in available[1]))
            if rcx in ins.defs or ins.defs & dependencies:
                return None
        return available

    def incoming(block, outgoing):
        states = [outgoing[p] for p in pred[block]]
        return states[0] if block is not fn.entry and states and all(k == states[0] for k in states) else None

    outgoing = {block: None for block in fn.blocks}
    changed = True
    while changed:
        changed = False
        for block in fn.blocks:
            value = incoming(block, outgoing)
            for ins in block.ins:
                value = transfer(ins, value)
            if value != outgoing[block]:
                outgoing[block], changed = value, True
    for block in fn.blocks:
        value = incoming(block, outgoing)
        for ins in block.ins:
            signature = key(ins)
            if signature is not None and signature == value and rcx in ins.implicit_defs:
                ins.implicit_defs = ins.implicit_defs - {rcx}
                ins.implicit_uses = ins.implicit_uses | {rcx}
            value = transfer(ins, value)
