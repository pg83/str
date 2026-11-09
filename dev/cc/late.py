"""Native machine simplifications after the shared frame and vector passes."""
import opt

from . import machine as M
from .select import INVERSE


def conditional_tails(obj, code):
    report = []
    for section, instructions in code.items():
        functions = obj.functions(section)
        if len(functions) != 1:
            raise ValueError('native code requires one function per section')
        symbol = functions[0]
        fn = opt.Function(symbol.name, instructions, symbol.value, symbol.value + symbol.size)
        changed = True
        while changed:
            changed = False
            opt.analyze(fn)
            pred = opt.predecessors(fn)
            offsets = fn.index_of_offset()
            block_at = {b.start: b for b in fn.blocks}
            for block in fn.blocks:
                live = [i for i in fn.ins[block.start:block.end] if i.gas]
                if not live:
                    continue
                branch = live[-1]
                if branch.flow != 'ConditionalBranch' or not fn.internal(branch):
                    continue
                taken = block_at[offsets[branch.target]]
                fall = block_at.get(block.end)
                for target, invert in ((fall, True), (taken, False)):
                    if target is None or pred[id(target)] != [block]:
                        continue
                    body = [i for i in fn.ins[target.start:target.end] if i.gas]
                    if len(body) != 1:
                        continue
                    tail = body[0]
                    callee = tail.j.get('callee')
                    if tail.mnemonic != 'jmp' or tail.reloc is None or not callee or not tail.j.get('shrinkwrap_tail'):
                        continue
                    if invert:
                        following = next((n for n in range(target.end, len(fn.ins)) if fn.ins[n].gas), len(fn.ins))
                        if not target.end <= offsets[branch.target] <= following:
                            continue
                    cc = branch.j['cc']
                    cc = INVERSE[cc] if invert else cc
                    relocation = tail.reloc
                    direct = relocation.type == 'R_X86_64_PLT32'
                    name, addend = (relocation.symbol, relocation.addend + 4) if direct else (callee, 0)
                    arguments = frozenset(M.physical(r['full']) for r in tail.j['regs'] if r['access'] == 'Read')
                    replacement = M.structured(M.Instr('j' + cc, (M.Target(name, addend),), ('R',), cc=cc,
                                                       implicit_uses=arguments), branch.offset)
                    replacement.reloc.offset += branch.home - branch.offset
                    replacement.reloc.symbol_is_section = direct and relocation.symbol_is_section
                    replacement.j['home'] = branch.home
                    if branch.j.get('before'):
                        replacement.j['before'] = True
                    branch.j, branch.reloc = replacement.j, replacement.reloc
                    opt.delete(tail)
                    report.append('%s: conditional tail call to %s' % (fn.name, callee))
                    changed = True
                    break
                if changed:
                    break
    return report
