"""Allocate registers, lay out native instructions and emit an ELF object."""
from dataclasses import replace

import elfout
import layout
import encode
from objfile import Object, Section, Symbol

from . import machine as M
from .regalloc import CALLEE_SAVED, GPRS, allocate, liveness
from .select import INVERSE, Select
from .stringselect import StringSelect, share_epilogue as share_string_epilogue, layout_blocks as layout_string_blocks
from .spanselect import (SpanSelect, parallel_copies as span_parallel_copies,
                      share_epilogue as share_span_epilogue, layout_blocks as layout_span_blocks)
from .cspnselect import CspnSelect, share_epilogue as share_cspn_epilogue
from .cmpselect import (CmpSelect, share_epilogue as share_cmp_epilogue,
                        upper_state as cmp_upper_state, layout_blocks as layout_cmp_blocks)
from .tailselect import TailSelect, layout_blocks as layout_tail_blocks
from .copyselect import CopySelect, layout_blocks as layout_copy_blocks
from .catselect import CatSelect
from .longcopyselect import (LongCopySelect, parallel_copies as long_parallel_copies,
                             share_epilogue as share_long_epilogue, upper_state as long_upper_state,
                             layout_blocks as layout_long_blocks)
from .types import align_up
from .data import emit_data


def move(d, s):
    mask = next((o for o in (d, s) if isinstance(o, M.R) and o.reg.bank == 'mask'), None)
    if mask is not None:
        return M.Instr('kmov' + {8: 'b', 16: 'w', 32: 'd', 64: 'q'}[mask.bits], (d, s), ('W', 'R'))
    vec = isinstance(d, M.R) and d.reg.bank == 'vec' or isinstance(s, M.R) and s.reg.bank == 'vec'
    bits = d.bits if isinstance(d, M.R) else s.bits if isinstance(s, M.R) else d.size * 8
    if isinstance(d, M.R) and isinstance(s, M.R) and d.bits < s.bits:
        s = M.R(s.reg, d.bits)
    if not vec and isinstance(d, M.R) and isinstance(s, M.Imm) and s.value == 0 and s.symbol is None:
        zero = M.R(d.reg, 32)
        return M.Instr('xor', (zero, zero), ('W', 'R'))
    if not vec and isinstance(d, M.R) and isinstance(s, M.Imm) and bits == 64 and (s.symbol is not None or 0 <= s.value < 1 << 32):
        d, s = M.R(d.reg, 32), replace(s, bits=32)
    name = 'movups' if vec and bits == 128 else 'vmovups' if vec else 'mov'
    return M.Instr(name, (d, s), ('W', 'R'))


def parallel_copies(fn):
    for b in fn.blocks:
        out = []
        for ins in b.ins:
            if ins.mnemonic != 'parallel':
                out.append(ins)
                continue
            copies = [(d, s) for d, s in ins.copies if d != s and not (
                isinstance(d, M.R) and isinstance(s, M.R) and d.reg == s.reg and d.reg.bank == 'mask' and d.bits >= s.bits)]

            def copy(d, s):
                if isinstance(d, M.Mem) and isinstance(s, M.Mem):
                    size = d.size
                    name = 'r11' if size <= 8 else 'zmm31' if fn.isa in ('evex', 'evex512') else 'xmm15'
                    tmp = M.R(M.physical(name), size * 8)
                    out.extend([move(tmp, s), move(d, tmp)])
                else:
                    out.append(move(d, s))

            while copies:
                sources = set().union(*(M.regs(s) for _, s in copies))
                available = [i for i, (d, _) in enumerate(copies) if not M.regs(d) & sources]
                ready = (min(available, key=lambda n: GPRS.index(copies[n][0].reg.id)
                             if isinstance(copies[n][0], M.R) and copies[n][0].reg.id in GPRS else len(GPRS))
                         if available and getattr(fn, 'unaligned_loops', ()) else next(iter(available), None))
                if ready is not None:
                    d, s = copies.pop(ready)
                    copy(d, s)
                    continue
                d = copies[0][0]
                if not isinstance(d, M.R):
                    raise ValueError('parallel copy memory cycle')
                bits = 64 if d.reg.bank == 'gpr' else max(o.bits for pair in copies for o in pair if isinstance(o, M.R) and o.reg == d.reg)
                slot = fn.slot(bits // 8, min(bits // 8, 64))
                copy(M.Mem(bits // 8, slot=slot), M.R(d.reg, bits))
                copies = [(a, M.Mem(s.bits // 8, slot=slot) if isinstance(s, M.R) and s.reg == d.reg else s) for a, s in copies]
        if getattr(fn.source, 'aligned_vector_copies', False):
            for ins in out:
                if ins.mnemonic == 'vmovups' and all(isinstance(o, M.R) for o in ins.operands):
                    ins.mnemonic = 'vmovaps'
        b.ins = out
        # ABI argument copies can restore an input register that still holds
        # the saved value. Keep value identities across full-width GPR copies.
        values, out = {}, []
        for n, ins in enumerate(b.ins):
            copy = (ins.mnemonic == 'mov' and len(ins.operands) == 2 and not ins.preserve
                    and all(isinstance(o, M.R) and o.bits == 64 and o.reg.bank == 'gpr' for o in ins.operands))
            if copy:
                d, s = (o.reg for o in ins.operands)
                source = values.get(s, ('input', s))
                if values.get(d, ('input', d)) == source:
                    continue
            for r in ins.defs:
                values[r] = ('definition', n, r)
            if copy:
                values[d] = source
            out.append(ins)
        b.ins = out


def cleanup_edges(fn):
    for b in fn.blocks:
        b.ins = [i for i in b.ins if not (i.mnemonic in ('mov', 'movups', 'movaps', 'vmovups', 'vmovaps', 'movdqu', 'movdqa', 'vmovdqu', 'vmovdqa', 'vmovdqu64', 'vmovdqa64', 'kmovb', 'kmovw', 'kmovd', 'kmovq') and not i.preserve
                 and len(i.operands) == 2 and isinstance(i.operands[0], M.R) and i.operands[0] == i.operands[1])]
        out = []
        for ins in b.ins:
            if (ins.mnemonic == 'test' and len(ins.operands) == 2 and ins.operands[0] == ins.operands[1]
                    and out and out[-1].mnemonic in ('and', 'or', 'xor') and out[-1].operands[0] == ins.operands[0]):
                continue
            out.append(ins)
        b.ins = out
    tails, aliases = {}, {}
    for b in fn.blocks:
        if b.name.startswith('edge.') and len(b.ins) == 2 and b.ins[-1].mnemonic == 'jmp':
            move_ = b.ins[0]
            if (not move_.preserve and (move_.mnemonic == 'mov' and all(isinstance(o, M.R) for o in move_.operands)
                                       or move_.mnemonic == 'xor' and len(move_.operands) == 2 and move_.operands[0] == move_.operands[1]
                                       or move_.mnemonic == 'lea')):
                key = move_.mnemonic, move_.operands, b.ins[-1].target
                if key in tails:
                    aliases[b] = tails[key]
                else:
                    tails[key] = b
    if aliases:
        for b in fn.blocks:
            b.successors = [aliases.get(s, s) for s in b.successors]
            for ins in b.ins:
                if ins.target in aliases:
                    ins.target = aliases[ins.target]
        fn.layout_owners = {aliases.get(b, b): aliases.get(p, p) for b, p in getattr(fn, 'layout_owners', {}).items()
                            if aliases.get(b, b) is not aliases.get(p, p)}
        fn.blocks = [b for b in fn.blocks if b not in aliases]
    pred = {b: [p for p in fn.blocks if b in p.successors] for b in fn.blocks}
    rax = M.physical('rax')
    if not getattr(fn.source, 'scalar_read_loops', ()) and not any(rax in i.uses | i.defs for i in fn.entry.ins):
        for b in fn.blocks:
            if pred[b] != [fn.entry] or len(b.ins) != 2 or b.ins[-1].mnemonic != 'jmp':
                continue
            zero = b.ins[0]
            if zero.mnemonic == 'xor' and zero.operands == (M.R(rax, 32), M.R(rax, 32)):
                fn.entry.ins.insert(0, zero)
                b.ins.pop(0)
                break
    while True:
        aliases = {b: b.ins[0].target for b in fn.blocks if b is not fn.entry and len(b.ins) == 1
                   and b.ins[0].mnemonic == 'jmp' and b.ins[0].target is not None and b.ins[0].target is not b}
        if not aliases:
            break
        # Do not collapse an empty cycle: it remains an infinite loop.
        for b in list(aliases):
            seen, target = {b}, aliases[b]
            while target in aliases and target not in seen:
                seen.add(target)
                target = aliases[target]
            if target in seen:
                aliases.pop(b)
        if not aliases:
            break
        def resolve(b):
            while b in aliases:
                b = aliases[b]
            return b
        for b in fn.blocks:
            b.successors = [resolve(s) for s in b.successors]
            for i in b.ins:
                if i.target is not None:
                    i.target = resolve(i.target)
        fn.loop_headers = {resolve(b) for b in getattr(fn, 'loop_headers', ())}
        fn.unaligned_loops = {resolve(b) for b in getattr(fn, 'unaligned_loops', ())}
        fn.loop_alignments = {resolve(b): align for b, align in getattr(fn, 'loop_alignments', {}).items()}
        fn.layout_owners = {resolve(b): resolve(p) for b, p in getattr(fn, 'layout_owners', {}).items()
                            if resolve(b) is not resolve(p)}
        fn.blocks = [b for b in fn.blocks if b not in aliases]
    fn.cold_blocks = getattr(fn, 'cold_blocks', set()) & set(fn.blocks)


def execution_domains(fn):
    """Choose integer move forms for vector registers pinned to that domain."""
    for block in fn.blocks:
        for zero in block.ins:
            if (zero.mnemonic != 'pxor' or len(zero.operands) != 2 or zero.operands[0] != zero.operands[1]
                    or not isinstance(zero.operands[0], M.R)):
                continue
            reg = zero.operands[0].reg
            users = [ins for b in fn.blocks for ins in b.ins if ins is not zero and reg in ins.uses]
            if not users or any(ins.mnemonic not in ('movaps', 'movups', 'movdqa', 'movdqu')
                                or len(ins.operands) != 2 or not isinstance(ins.operands[0], M.Mem)
                                or ins.operands[0].slot is None or ins.operands[1] != zero.operands[0] for ins in users):
                continue
            zero.mnemonic = 'xorps'
            for ins in users:
                ins.domain = 'fp'
    for b in fn.blocks:
        for ins in b.ins:
            if ins.domain == 'fp':
                aligned = ins.mnemonic in ('movdqa', 'vmovdqa', 'vmovdqa64', 'movaps', 'vmovaps')
                ins.mnemonic = ('v' if ins.mnemonic.startswith('v') else '') + ('movaps' if aligned else 'movups')
    flexible = {'movups', 'movaps', 'vmovups', 'vmovaps'}
    integer = {o.reg for b in fn.blocks for i in b.ins if i.mnemonic not in flexible
               for o in i.operands if isinstance(o, M.R) and o.reg.bank == 'vec'}
    changed = True
    while changed:
        changed = False
        for b in fn.blocks:
            for ins in b.ins:
                if ins.mnemonic not in flexible or not all(isinstance(o, M.R) and o.reg.bank == 'vec' for o in ins.operands):
                    continue
                registers = {o.reg for o in ins.operands}
                if registers & integer and not registers <= integer:
                    integer |= registers
                    changed = True
    for b in fn.blocks:
        for ins in b.ins:
            if ins.mnemonic not in flexible or ins.domain == 'fp':
                continue
            operands = [o for o in ins.operands if isinstance(o, M.R) and o.reg.bank == 'vec']
            if not any(o.reg in integer for o in operands):
                continue
            bits = max(o.bits for o in operands)
            aligned = ins.mnemonic.endswith('aps')
            ins.mnemonic = ('v' if ins.mnemonic.startswith('v') else '') + ('movdqa' if aligned else 'movdqu') + ('64' if bits == 512 else '')


def schedule_counts(fn):
    """Keep independent pointer differences ahead of scalar bit scans."""
    flags = M.Reg('flags', 'rflags')
    _, outgoing = liveness(fn)
    count_first = {fn.ssa_blocks[b] for b in getattr(fn.source, 'count_first_blocks', ())}
    for block in fn.blocks:
        live, after = set(outgoing[block]), {}
        for ins in reversed(block.ins):
            after[ins] = flags in live
            live = (live - ins.defs) | ins.uses
        if (getattr(fn, 'returned_parameter', None) is not None and not getattr(fn.source, 'pipelined_copies', {})
                or fn.source.type.base.kind == 'ptr' and getattr(fn.source, 'fixed_loop_exits', ())
                or block in count_first):
            for count in list(block.ins):
                if count.mnemonic not in ('tzcnt', 'lzcnt', 'bsf', 'bsr') or after[count]:
                    continue
                at = block.ins.index(count)
                previous = at - 1
                while previous >= 0 and block.ins[previous].mnemonic.startswith('kmov'):
                    previous -= 1
                if previous < 0 or block.ins[previous].mnemonic != 'sub':
                    continue
                difference = block.ins[previous]
                if not all(isinstance(o, M.R) for o in (*count.operands, *difference.operands)):
                    continue
                between = block.ins[previous + 1:at + 1]
                written = {r for r in difference.defs if r.bank == 'gpr'}
                used = {r for r in difference.uses if r.bank == 'gpr'}
                if any(written & (ins.uses | ins.defs) or used & ins.defs for ins in between):
                    continue
                block.ins.pop(previous)
                block.ins.insert(block.ins.index(count) + 1, difference)
            continue
        for n in range(len(block.ins) - 1):
            count, difference = block.ins[n:n + 2]
            if (count.mnemonic not in ('tzcnt', 'lzcnt', 'bsf', 'bsr') or difference.mnemonic != 'sub'
                    or after[difference] or not all(isinstance(o, M.R) for o in (*count.operands, *difference.operands))):
                continue
            written = {r for r in count.defs if r.bank == 'gpr'}
            changed = {r for r in difference.defs if r.bank == 'gpr'}
            if written & (difference.uses | difference.defs) or changed & count.uses:
                continue
            block.ins[n:n + 2] = [difference, count]


def arithmetic_forms(fn):
    """Use destructive arithmetic when address arithmetic has no live flags."""
    flags = M.Reg('flags', 'rflags')
    _, outgoing = liveness(fn)
    for b in fn.blocks:
        if getattr(fn.source, 'large_byte_compares', False):
            for address, mask in zip(b.ins, b.ins[1:]):
                if (address.mnemonic == 'lea' and len(address.operands) == 2
                        and isinstance(address.operands[0], M.R) and address.operands[0].bits == 64
                        and isinstance(address.operands[1], M.Mem) and mask.mnemonic == 'and'
                        and isinstance(mask.operands[0], M.R) and mask.operands[0].bits == 32
                        and address.operands[0].reg == mask.operands[0].reg):
                    address.operands = mask.operands[0], replace(address.operands[1], size=4)
        live = set(outgoing[b])
        for n, ins in reversed(list(enumerate(b.ins))):
            only_zero = (n + 1 < len(b.ins) and b.ins[n + 1].target is not None
                         and b.ins[n + 1].cc in ('e', 'ne') and flags not in outgoing[b])
            if (flags not in live and ins.mnemonic == 'lea' and len(ins.operands) == 2
                    and isinstance(ins.operands[0], M.R) and isinstance(ins.operands[1], M.Mem)):
                d, source = ins.operands
                if (getattr(fn, 'unaligned_loops', ()) and source.base is not None and source.base != d.reg
                        and source.index is None and source.slot is None and source.symbol is None and source.disp == 128):
                    b.ins.insert(b.ins.index(ins), M.Instr('mov', (d, M.R(source.base, d.bits)), ('W', 'R')))
                    ins.mnemonic, ins.operands, ins.access = 'add', (d, M.Imm(source.disp)), ('RW', 'R')
                elif source.base == d.reg and source.index is None and source.slot is None and source.symbol is None and source.disp:
                    ins.mnemonic, ins.operands, ins.access = 'add', (d, M.Imm(source.disp)), ('RW', 'R')
                elif ((fn.source.type.base.kind == 'ptr' and getattr(fn.source, 'return_joins', ())
                       or getattr(fn.source, 'shared_load_return', None) is not None
                       or getattr(fn.source, 'shared_index_return', None) is not None
                       or getattr(fn.source, 'indexed_byte_scans', {})
                       or getattr(fn.source, 'word_scans', {})
                       or getattr(fn.source, 'counted_offsets', ())
                       or getattr(fn.source, 'copy_loops', False)
                       or getattr(fn.source, 'aligned_vector_search', False)
                       or getattr(fn, 'unaligned_loops', ())
                       or getattr(fn, 'returned_parameter', None) is not None and getattr(fn.source, 'induction_registers', None))
                      and source.base is not None and source.index is not None and source.scale == 1
                      and d.reg in (source.base, source.index)
                      and (not source.disp or getattr(fn.source, 'shared_load_return', None) is not None
                           or getattr(fn.source, 'copy_loops', False)
                           or getattr(fn.source, 'aligned_vector_search', False)
                           or getattr(fn, 'returned_parameter', None) is not None and getattr(fn.source, 'induction_registers', None))
                      and source.slot is None and source.symbol is None):
                    other = source.index if source.base == d.reg else source.base
                    ins.mnemonic, ins.operands, ins.access = 'add', (d, M.R(other, d.bits)), ('RW', 'R')
                    if source.disp:
                        b.ins.insert(b.ins.index(ins) + 1, M.Instr('add', (d, M.Imm(source.disp)), ('RW', 'R')))
            if ((flags not in live or only_zero) and ins.mnemonic in ('add', 'sub') and len(ins.operands) == 2
                    and isinstance(ins.operands[0], M.R) and isinstance(ins.operands[1], M.Imm) and ins.operands[1].symbol is None):
                d, c = ins.operands
                if c.value in (1, -1):
                    positive = (ins.mnemonic == 'add') == (c.value == 1)
                    ins.mnemonic, ins.operands, ins.access = 'inc' if positive else 'dec', (d,), ('RW',)
                elif not ins.keep_immediate and not -128 <= c.value <= 127 and -128 <= -c.value <= 127:
                    ins.mnemonic = 'sub' if ins.mnemonic == 'add' else 'add'
                    ins.operands = d, replace(c, value=-c.value)
            live = (live - ins.defs) | ins.uses


def compact_byte_shift_counts(fn):
    """Reuse a dead input's low byte for a sole CL shift-count use."""
    if not getattr(fn.source, 'byte_compares', False):
        return
    _, outgoing = liveness(fn)
    flags = M.Reg('flags', 'rflags')
    for block in fn.blocks:
        live, after = set(outgoing[block]), {}
        for ins in reversed(block.ins):
            after[ins] = set(live)
            live = (live - ins.defs) | ins.uses
        for n, ins in reversed(list(enumerate(block.ins[:-1]))):
            shift = block.ins[n+1]
            if (getattr(fn.source, 'byte_compare_width', None) == 32
                    and ins.mnemonic in ('add', 'sub') and len(ins.operands) == 2
                    and isinstance(ins.operands[0], M.R) and ins.operands[0].bits == 32
                    and isinstance(ins.operands[1], M.Imm) and ins.operands[1].symbol is None
                    and -128 <= ins.operands[1].value <= 127 and flags not in after[ins]
                    and shift.mnemonic in ('shlx', 'shrx', 'sarx') and len(shift.operands) == 3
                    and isinstance(shift.operands[2], M.R)
                    and shift.operands[2].reg == ins.operands[0].reg
                    and ins.operands[0].reg not in after[shift]
                    and ins.operands[0].reg not in M.regs(shift.operands[1])):
                ins.operands = M.R(ins.operands[0].reg, 8), replace(ins.operands[1], bits=8)
                continue
            if (ins.mnemonic != 'lea' or len(ins.operands) != 2
                    or not isinstance(ins.operands[0], M.R) or ins.operands[0].bits != 32
                    or not isinstance(ins.operands[1], M.Mem)):
                continue
            destination, source = ins.operands
            if (source.base is None or source.index is not None or source.slot is not None or source.symbol is not None
                    or not -128 <= source.disp <= 127 or source.base == destination.reg
                    or source.base in after[ins] or flags in after[ins]
                    or shift.mnemonic not in ('shl', 'shr', 'sar') or len(shift.operands) != 2
                    or shift.operands[1] != M.R(destination.reg, 8) or destination.reg in after[shift]):
                continue
            block.ins[n:n+1] = [M.Instr('add', (M.R(source.base, 8), M.Imm(source.disp, 8)), ('RW', 'R')),
                                M.Instr('mov', (destination, M.R(source.base, 32)), ('W', 'R'))]


def schedule_short_vector_compare(fn):
    """Overlap two VEX byte masks and keep the bit scan independent."""
    if (fn.isa != 'avx2' or len(fn.blocks) != 1 or fn.source.type.base.kind != 'int'
            or fn.source.type.base.bits != 32 or len(fn.source.params) != 3):
        return
    block = fn.blocks[0]
    compares = [ins for ins in block.ins if ins.mnemonic == 'vpcmpeqb']
    masks = [ins for ins in block.ins if ins.mnemonic == 'vpmovmskb']
    scans = [ins for ins in block.ins if ins.mnemonic == 'tzcnt' and ins.operands[0].bits == 64]
    marked = [ins for ins in block.ins if ins.mnemonic == 'bts']
    if (len(compares) != 2 or len(masks) != 2 or len(scans) != 1 or len(marked) != 1
            or any(ins.mnemonic == 'call' or ins.target is not None for ins in block.ins)):
        return
    flags = M.Reg('flags', 'rflags')
    live, after = set(), {}
    for ins in reversed(block.ins):
        after[ins] = set(live)
        live = (live - ins.defs) | ins.uses
    first = masks[0].operands[0]
    for ins in list(block.ins):
        if (ins.mnemonic != 'xor' or ins.operands != (first, M.Imm(65535, 32))):
            continue
        at, end = block.ins.index(ins), block.ins.index(compares[1])
        if at >= end or any((ins.defs - {flags}) & (other.defs | other.uses)
                            or ins.uses & other.defs or flags in other.uses | other.defs
                            for other in block.ins[at+1:end+1]):
            continue
        block.ins.pop(at)
        block.ins.insert(block.ins.index(compares[1])+1, ins)
    bit = marked[0]
    for ins in list(block.ins):
        if (ins.mnemonic != 'or' or ins.operands[0] != bit.operands[0]
                or not isinstance(ins.operands[1], M.R)):
            continue
        at, end = block.ins.index(ins), block.ins.index(bit)
        if at >= end or flags in after[bit] or any(((ins.uses | ins.defs) - {flags}) & other.defs or flags in other.uses
                            for other in block.ins[at+1:end]):
            continue
        if ins.operands[1].reg in bit.defs:
            continue
        block.ins.pop(at)
        block.ins.insert(block.ins.index(bit)+1, ins)
    scan = scans[0]
    d, s = scan.operands
    if isinstance(d, M.R) and isinstance(s, M.R) and d.reg != s.reg:
        out = M.R(d.reg, 32)
        block.ins.insert(block.ins.index(scan), M.Instr('xor', (out, out), ('W', 'R')))


def schedule_large_compare_tests(fn):
    """Start a scalar mask test while the second vector compare is pending."""
    if not getattr(fn.source, 'large_byte_compares', False):
        return
    flags = M.Reg('flags', 'rflags')
    for block in fn.blocks:
        if sum(ins.mnemonic == 'vpcmpeqb' for ins in block.ins) != 2:
            continue
        for test in list(block.ins):
            if (test.mnemonic != 'cmp' or len(test.operands) != 2 or not isinstance(test.operands[0], M.R)
                    or not isinstance(test.operands[1], M.Imm) or test.operands[1].value != -1):
                continue
            mask = next((n for n, ins in enumerate(block.ins) if ins.mnemonic == 'vpmovmskb'
                         and ins.operands[0] == test.operands[0]), None)
            if mask is None:
                continue
            at = next((n for n in range(mask+1, len(block.ins)) if block.ins[n].mnemonic == 'vpcmpeqb'), None)
            old = block.ins.index(test)
            if at is None or at >= old or any(flags in ins.uses | ins.defs or test.uses & ins.defs for ins in block.ins[at:old]):
                continue
            block.ins.pop(old)
            block.ins.insert(at, test)


def share_tail_calls(fn):
    """Share an identical terminal jump when one path needs no argument work."""
    for tail in list(fn.blocks):
        if (len(tail.ins) != 1 or tail.successors or tail.ins[0].mnemonic != 'jmp'
                or tail.ins[0].target is not None or not tail.ins[0].callee):
            continue
        call = tail.ins[0]
        matches = [b for b in fn.blocks if b is not tail and len(b.ins) > 1 and not b.successors
                   and b.ins[-1].mnemonic == call.mnemonic and b.ins[-1].operands == call.operands
                   and b.ins[-1].callee == call.callee and b.ins[-1].implicit_uses == call.implicit_uses]
        for block in matches:
            block.ins[-1] = M.Instr('jmp', target=tail)
            block.successors = [tail]
        if matches:
            fn.layout_owners[tail] = matches[-1]


def duplicate_tail_calls(fn):
    """Place a short tail call on incoming edges before adding epilogues."""
    if not (getattr(fn, 'unaligned_loops', ()) or getattr(fn.source, 'duplicate_short_tail_calls', False)):
        return
    for block in list(fn.blocks):
        if (not 1 <= len(block.ins) <= 3 or block.successors or block.ins[-1].mnemonic != 'jmp'
                or block.ins[-1].target is not None or not block.ins[-1].callee
                or any(ins.target is not None or ins.preserve for ins in block.ins)
                or any(isinstance(o, M.Mem) for ins in block.ins[:-1] for o in ins.operands)):
            continue
        copied = []
        for pred in fn.blocks:
            if pred.successors != [block] or not pred.ins or pred.ins[-1].mnemonic != 'jmp' or pred.ins[-1].target is not block:
                continue
            pred.ins[-1:] = [replace(ins) for ins in block.ins]
            pred.successors = []
            copied.append(pred)
        if copied and not any(block in b.successors for b in fn.blocks):
            fn.blocks.remove(block)
            fn.layout_owners = {b: copied[-1] if owner is block else owner for b, owner in getattr(fn, 'layout_owners', {}).items() if b is not block}


def search_return_counts(fn):
    """Break bit-scan dependencies and share the cursor return suffix."""
    length = getattr(fn.source, 'vector_length_loop', None) is not None
    compare = getattr(fn.source, 'byte_compare_width', None) == 32
    classification = getattr(fn.source, 'vector_classification', False)
    if fn.isa != 'avx2' or not (getattr(fn.source, 'aligned_search_width', None) == 32 or length or compare or classification):
        return
    for block in fn.blocks:
        for ins in list(block.ins):
            if (ins.mnemonic == 'tzcnt' and len(ins.operands) == 2
                    and all(isinstance(o, M.R) for o in ins.operands)
                    and (ins.operands[0].bits == 32 or compare and ins.operands[0].bits == 64)
                    and ins.operands[0].reg != ins.operands[1].reg):
                out = M.R(ins.operands[0].reg, 32)
                block.ins.insert(block.ins.index(ins), M.Instr('xor', (out, out), ('W', 'R')))
    if length or compare or classification:
        return
    returns = [block for block in fn.blocks if len(block.ins) >= 4 and block.ins[-1].mnemonic == 'ret'
               and block.ins[-4].mnemonic == 'tzcnt' and block.ins[-3].mnemonic == 'add'
               and block.ins[-2].mnemonic == 'mov']
    def signature(ins):
        return ins.mnemonic, ins.operands, ins.access, ins.implicit_uses, ins.implicit_defs
    for n, a in enumerate(returns):
        for b in returns[n+1:]:
            if [signature(i) for i in a.ins[-3:]] != [signature(i) for i in b.ins[-3:]]:
                continue
            owner = b if b.ins[-4].operands[0].reg != b.ins[-4].operands[1].reg else a
            tail = fn.block('return.cursor')
            tail.ins = a.ins[-3:]
            fn.layout_owners[tail] = owner
            for block in (a, b):
                block.ins[-3:] = [M.Instr('jmp', target=tail)]
                block.successors = [tail]
            return


def frame(fn, force=False):
    offsets, size, alignment = {}, 0, 16
    slot_alignment = max((a for _, a in fn.slots), default=1)
    for n, (bytes_, align) in enumerate(fn.slots):
        size = align_up(size, align)
        offsets[n], size, alignment = size, size + bytes_, max(alignment, align)
    physical_defs = {r.id for b in fn.blocks for ins in b.ins for r in ins.defs if r.physical}
    saved = [r for r in ('r15', 'r14', 'r13', 'r12', 'rbx') if r in physical_defs & CALLEE_SAVED]
    dynamic = alignment > 16
    calls = any(ins.mnemonic == 'call' for b in fn.blocks for ins in b.ins)
    use_fp = bool(force or saved or calls or size or dynamic)
    fn.frame_pointer = use_fp
    base_offset = -align_up(len(saved) * 8 + align_up(size, slot_alignment), slot_alignment)
    size = align_up(size + 16 if calls else size, alignment) if dynamic else -base_offset - len(saved) * 8
    if (calls or size) and not dynamic and ((len(saved) + use_fp) * 8 + size) % 16 != 8:
        size += 8
    if not calls and not saved and not dynamic:
        size = max(0, size-128)
    padding_push = calls and size == 8 and not fn.slots and not dynamic
    rsp, rbp = M.R(M.physical('rsp')), M.R(M.physical('rbp'))
    prologue = [M.Instr('push', (rbp,), ('R',)), move(rbp, rsp)] if use_fp else []
    prologue.extend(M.Instr('push', (M.R(M.physical(r)),), ('R',)) for r in saved)
    if dynamic:
        prologue.append(M.Instr('and', (rsp, M.Imm(-alignment)), ('RW', 'R')))
    if size:
        prologue.append(M.Instr('push', (M.R(M.physical('rax')),), ('R',)) if padding_push
                        else M.Instr('sub', (rsp, M.Imm(size)), ('RW', 'R')))
    epilogue = []
    if dynamic:
        if saved:
            epilogue.append(M.Instr('lea', (rsp, M.Mem(8, base=rbp.reg, disp=-8 * len(saved))), ('W', 'A')))
        else:
            epilogue.append(move(rsp, rbp))
    elif size:
        epilogue.append(M.Instr('add', (rsp, M.Imm(size)), ('RW', 'R')))
    epilogue.extend(M.Instr('pop', (M.R(M.physical(r)),), ('W',)) for r in reversed(saved))
    if use_fp:
        epilogue.append(M.Instr('pop', (rbp,), ('W',)))

    def memory(o):
        if isinstance(o, M.Mem) and o.slot is not None:
            return replace(o, base=rsp.reg if dynamic else rbp.reg,
                           disp=o.disp + offsets[o.slot] + (0 if dynamic else base_offset), slot=None)
        return o

    for b in fn.blocks:
        out = []
        for ins in b.ins:
            ins.operands = tuple(map(memory, ins.operands))
            if ins.mnemonic == 'ret' or ins.mnemonic == 'jmp' and ins.target is None:
                if epilogue and ins.operands and isinstance(ins.operands[0], M.R) and ins.operands[0].reg.id in saved:
                    target = M.R(M.physical('r11'), ins.operands[0].bits)
                    out.append(move(target, ins.operands[0]))
                    ins.operands = (target,)
                result_move = None
                if (ins.mnemonic == 'ret' and out and out[-1].mnemonic == 'mov'
                        and len(out[-1].operands) == 2 and all(isinstance(o, M.R) for o in out[-1].operands)
                        and out[-1].operands[0].reg.id == 'rax' and out[-1].operands[1].reg.id in saved):
                    result_move = out.pop()
                if padding_push and result_move is not None:
                    out.append(result_move)
                    result_move = None
                for i in epilogue:
                    if result_move is not None and i.mnemonic == 'pop' and i.operands[0].reg == result_move.operands[1].reg:
                        out.append(result_move)
                        result_move = None
                    out.append(replace(i))
                if result_move is not None:
                    out.append(result_move)
            out.append(ins)
        b.ins = out
    count = 0
    while count < len(fn.entry.ins):
        i = fn.entry.ins[count]
        if (i.mnemonic != 'mov' or len(i.operands) != 2 or not all(isinstance(o, M.R) for o in i.operands)
                or i.operands[0].reg.id not in CALLEE_SAVED or i.operands[1].reg.id not in ('rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9')):
            break
        count += 1
    fn.entry.ins[:count] = sorted(fn.entry.ins[:count], key=lambda i: ('rbx', 'r14', 'r15', 'r12', 'r13').index(i.operands[0].reg.id))
    fn.entry.ins[:0] = prologue


def fold_tail_addresses(fn):
    """Compute an aligned tail-call argument directly in its ABI register."""
    for b in fn.blocks:
        if not b.ins or b.ins[-1].mnemonic != 'jmp' or b.ins[-1].target is not None:
            continue
        tail = b.ins[-1]
        for n in range(len(b.ins) - 4):
            copy, mask, address = b.ins[n:n + 3]
            if (copy.mnemonic != 'mov' or len(copy.operands) != 2 or not all(isinstance(o, M.R) for o in copy.operands)
                    or copy.preserve or copy.operands[0].bits != 64 or copy.operands[1].bits != 64):
                continue
            tmp, original = copy.operands
            if tmp == original or mask.mnemonic != 'and' or mask.operands[0] != tmp or not isinstance(mask.operands[1], M.Imm):
                continue
            if address.mnemonic != 'lea' or address.operands[0] != tmp:
                continue
            mem = address.operands[1]
            if not isinstance(mem, M.Mem) or mem.base != tmp.reg or mem.index is not None or mem.symbol or mem.slot is not None:
                continue
            copies = b.ins[n + 3:-1]
            final = copies[-1] if copies else None
            if final is None or final.mnemonic != 'mov' or final.operands != (original, tmp) or final.preserve:
                continue
            saved = copies[:-1]
            if any(i.mnemonic != 'mov' or i.preserve or len(i.operands) != 2 or i.operands[1] != original
                   or not isinstance(i.operands[0], M.R) or i.operands[0].reg in (tmp.reg, original.reg) for i in saved):
                continue
            if tmp.reg in tail.uses or original.reg in set().union(*(M.regs(o) for o in tail.operands)):
                continue
            b.ins[n:-1] = [*saved, M.Instr('and', (original, mask.operands[1]), ('RW', 'R')),
                            M.Instr('add', (original, M.Imm(mem.disp)), ('RW', 'R'))]
            break


def table_stack_addresses(fn):
    block = getattr(fn, 'table_build_remainder', None)
    if block is None:
        return
    for ins in list(block.ins):
        if ins.mnemonic != 'lea' or len(ins.operands) != 2 or not isinstance(ins.operands[1], M.Mem):
            continue
        dst, source = ins.operands
        if (dst.reg != source.index or source.base != M.physical('rsp') or source.scale != 1
                or source.symbol is not None or not source.disp):
            continue
        ins.mnemonic, ins.operands, ins.access = 'add', (dst, M.R(source.base)), ('RW', 'R')
        block.ins.insert(block.ins.index(ins)+1, M.Instr('add', (dst, M.Imm(source.disp)), ('RW', 'R')))


def upper_state(fn):
    dirty = any(isinstance(o, M.R) and o.bits >= 256 and o.reg in ins.defs and int(o.reg.id[3:]) < 16
                for b in fn.blocks for ins in b.ins for o in ins.operands if isinstance(o, M.R) and o.reg.bank == 'vec')
    if not dirty:
        return
    for b in fn.blocks:
        out = []
        local_dirty = dirty
        for i in b.ins:
            if ((i.mnemonic in ('ret', 'call') or i.mnemonic == 'jmp' and i.target is None)
                    and local_dirty and not (i.mnemonic == 'jmp' and i.callee in getattr(fn.source, 'ymm_tail_callees', ()))):
                out.append(M.Instr('vzeroupper'))
            out.append(i)
            if i.mnemonic == 'call':
                local_dirty = False
            elif any(isinstance(o, M.R) and o.bits >= 256 and o.reg in i.defs
                     and int(o.reg.id[3:]) < 16 for o in i.operands if isinstance(o, M.R) and o.reg.bank == 'vec'):
                local_dirty = True
        b.ins = out
    cmp_upper_state(fn)
    long_upper_state(fn)


def merge_store_tails(fn):
    """Share identical store sequences at the ends of scalar size branches."""
    def same(a, b):
        return (a.mnemonic, a.operands, a.access, a.implicit_uses, a.implicit_defs, a.cc, a.rep, a.mask, a.zeroing) == (
            b.mnemonic, b.operands, b.access, b.implicit_uses, b.implicit_defs, b.cc, b.rep, b.mask, b.zeroing)
    fn.tail_owners = {}
    while True:
        selected = None
        returns = [b for b in fn.blocks if b.ins and b.ins[-1].mnemonic == 'ret']
        for n, a in enumerate(returns):
            for b in returns[n + 1:]:
                count = 0
                for x, y in zip(reversed(a.ins), reversed(b.ins)):
                    if x.preserve or y.preserve or x.target is not None or y.target is not None or x.mnemonic == 'call' or not same(x, y):
                        break
                    count += 1
                owner, other = (a, b) if len(a.ins) >= len(b.ins) else (b, a)
                if count < 4 or count == len(owner.ins):
                    continue
                stores = sum(any(isinstance(o, M.Mem) and 'W' in access for o, access in zip(i.operands, i.access)) for i in owner.ins[-count:])
                if stores >= 2 and (selected is None or count > selected[0]):
                    selected = count, owner, other
        if selected is None:
            break
        count, owner, other = selected
        tail = fn.block('tail.%d' % owner.id)
        tail.ins = owner.ins[-count:]
        fn.tail_owners[tail] = owner
        for b in (owner, other):
            b.ins[-count:] = [M.Instr('jmp', target=tail)]
            b.successors = [tail]
    # A branch that just returns can enter the return suffix of a shared
    # store tail, without keeping another copy of the same ABI move and ret.
    for owner in list(fn.tail_owners):
        if len(owner.ins) < 3 or owner.ins[-1].mnemonic != 'ret':
            continue
        suffix = owner.ins[-2:]
        if suffix[0].mnemonic != 'mov' or not all(isinstance(o, M.R) for o in suffix[0].operands):
            continue
        aliases = [b for b in fn.blocks if b is not owner and len(b.ins) == 2
                   and all(same(a, c) for a, c in zip(b.ins, suffix))]
        if not aliases:
            continue
        tail = fn.block('return.tail.%d' % owner.id)
        tail.ins = suffix
        fn.tail_owners[tail] = owner
        owner.ins[-2:] = [M.Instr('jmp', target=tail)]
        owner.successors = [tail]
        for b in fn.blocks:
            b.successors = [tail if s in aliases else s for s in b.successors]
            for i in b.ins:
                if i.target in aliases:
                    i.target = tail
        fn.blocks = [b for b in fn.blocks if b not in aliases]


def share_constant_epilogue(fn):
    """A precomputed return value can skip the zeroing of a constant exit."""
    if (share_long_epilogue(fn) or share_cmp_epilogue(fn) or share_cspn_epilogue(fn)
            or share_string_epilogue(fn) or share_span_epilogue(fn)):
        return
    if getattr(fn.source, 'indexed_byte_scans', {}):
        return
    bare = [b for b in fn.blocks if len(b.ins) == 1 and b.ins[0].mnemonic == 'ret']
    zero = next((b for b in fn.blocks if len(b.ins) == 2 and b.ins[1].mnemonic == 'ret'
                 and b.ins[0].mnemonic == 'xor' and b.ins[0].operands == (M.R(M.physical('rax'), 32),)*2), None)
    if not bare or zero is None:
        return
    tail = bare[0]
    zero.ins[-1:] = [M.Instr('jmp', target=tail)]
    zero.successors = [tail]
    fn.tail_owners[tail] = zero
    aliases = set(bare[1:])
    for block in fn.blocks:
        block.successors = [tail if b in aliases else b for b in block.successors]
        for ins in block.ins:
            if ins.target in aliases:
                ins.target = tail
    fn.blocks = [b for b in fn.blocks if b not in aliases]


def share_byte_compare_epilogue(fn):
    """Let the zero-length exit enter the vector comparison's return."""
    if fn.isa != 'avx2' or getattr(fn.source, 'byte_compare_width', None) != 32:
        return
    a = M.R(M.physical('rax'), 32)
    zero = next((b for b in fn.blocks if len(b.ins) == 2 and b.ins[-1].mnemonic == 'ret'
                 and b.ins[0].mnemonic == 'xor' and b.ins[0].operands == (a, a)), None)
    returns = [b for b in fn.blocks if b.ins and b.ins[-1].mnemonic == 'ret'
               and sum(ins.mnemonic == 'movzb' and any(isinstance(o, M.Mem) and o.size == 1 for o in ins.operands)
                       for ins in b.ins) == 2]
    if zero is None or len(returns) != 1:
        return
    pred = [b for b in fn.blocks if zero in b.successors]
    if len(pred) != 1 or len(pred[0].successors) != 2:
        return
    before = pred[0]
    other = next(b for b in before.successors if b is not zero)
    incoming, _ = liveness(fn)
    at = next((n for n, ins in enumerate(before.ins) if ins.mnemonic in ('cmp', 'test') or ins.target is not None), len(before.ins))
    if a.reg in incoming[other] or any(a.reg in ins.uses for ins in before.ins[at:]):
        return
    before.ins.insert(at, zero.ins[0])
    owner = returns[0]
    tail = fn.block('return.compare')
    tail.ins = [owner.ins[-1]]
    owner.ins[-1:] = [M.Instr('jmp', target=tail)]
    owner.successors = [tail]
    fn.layout_owners[tail] = owner
    before.successors = [tail if b is zero else b for b in before.successors]
    for ins in before.ins:
        if ins.target is zero:
            ins.target = tail
    fn.layout_owners = {b: tail if owner is zero else owner for b, owner in fn.layout_owners.items() if b is not zero}
    fn.blocks.remove(zero)


def share_mixed_width_return(fn):
    """Retain vector-state cleanup at a scalar and wide-vector join."""
    if fn.isa != 'avx2' or fn.source.type.base.kind != 'ptr':
        return
    stores = {o.size for b in fn.blocks for ins in b.ins for o, access in zip(ins.operands, ins.access)
              if isinstance(o, M.Mem) and 'W' in access}
    if not {16, 32} <= stores or any(isinstance(o, M.Mem) and 'R' in access
                                    for b in fn.blocks for ins in b.ins if ins.callee is None
                                    for o, access in zip(ins.operands, ins.access)):
        return
    bare = [b for b in fn.blocks if len(b.ins) == 2 and b.ins[-1].mnemonic == 'ret'
            and b.ins[0].mnemonic == 'mov' and all(isinstance(o, M.R) for o in b.ins[0].operands)]
    if len(bare) < 2 or any(b.ins[0].operands != bare[0].ins[0].operands for b in bare[1:]):
        return
    scalar = [b for b in fn.blocks if b.ins and b.ins[-1].mnemonic == 'ret'
              and [o.size for ins in b.ins for o, access in zip(ins.operands, ins.access)
                   if isinstance(o, M.Mem) and 'W' in access] == [1]]
    if len(scalar) != 1:
        return
    target, aliases = bare[-1], set(bare[:-1])
    for block in fn.blocks:
        block.successors = [target if b in aliases else b for b in block.successors]
        for ins in block.ins:
            if ins.target in aliases:
                ins.target = target
    fn.layout_owners = {target if b in aliases else b: target if owner in aliases else owner
                        for b, owner in fn.layout_owners.items()}
    fn.layout_owners[target] = scalar[0]
    fn.retained_returns = {target}
    fn.blocks = [b for b in fn.blocks if b not in aliases]


def share_sibling_returns(fn):
    """Let a bare return enter the same suffix as its sibling path."""
    def signature(i):
        return i.mnemonic, i.operands, i.access, i.implicit_uses, i.implicit_defs, i.cc, i.rep, i.mask, i.zeroing, i.preserve
    for bare in list(fn.blocks):
        if (bare in fn.tail_owners or bare in getattr(fn, 'retained_returns', ())
                or len(bare.ins) != 2 or bare.ins[-1].mnemonic != 'ret' or bare.ins[0].mnemonic != 'mov'
                or not all(isinstance(o, M.R) for o in bare.ins[0].operands)):
            continue
        siblings = {s for p in fn.blocks if bare in p.successors for s in p.successors if s is not bare}
        owners = [b for b in siblings if len(b.ins) > 2 and [signature(i) for i in b.ins[-2:]] == [signature(i) for i in bare.ins]
                  and sum(any(isinstance(o, M.Mem) and 'W' in access for o, access in zip(i.operands, i.access)) for i in b.ins[:-2]) >= 2]
        if not owners:
            continue
        owner = max(owners, key=lambda b: (len(b.ins), -fn.blocks.index(b)))
        tail = fn.block('return.shared.%d' % owner.id)
        tail.ins = owner.ins[-2:]
        owner.ins[-2:] = [M.Instr('jmp', target=tail)]
        owner.successors = [tail]
        fn.tail_owners[tail] = owner
        for b in fn.blocks:
            b.successors = [tail if s is bare else s for s in b.successors]
            for i in b.ins:
                if i.target is bare:
                    i.target = tail
        fn.blocks.remove(bare)


def linearize(fn):
    layout_string_blocks(fn)
    layout_span_blocks(fn)
    layout_cmp_blocks(fn)
    layout_tail_blocks(fn)
    layout_copy_blocks(fn)
    layout_long_blocks(fn)
    order = fn.order()
    for n, b in enumerate(order):
        next_ = order[n + 1] if n + 1 < len(order) else None
        ins = []
        for i in b.ins:
            if (i.mnemonic == 'lea' and len(i.operands) == 2 and isinstance(i.operands[1], M.Mem)
                    and i.operands[1].index == M.physical('rax') and i.operands[1].scale == 1
                    and getattr(fn.source, 'shared_index_return', None) is None
                    and not getattr(fn.source, 'preserve_address_order', False)
                    and not getattr(fn.source, 'indexed_byte_scans', {})
                    and i.operands[1].base is not None and i.operands[1].base.id != 'rsp'):
                mem = i.operands[1]
                i.operands = (i.operands[0], replace(mem, base=mem.index, index=mem.base))
            if i.mnemonic in ('mov', 'movups', 'movaps', 'vmovups', 'vmovaps', 'movdqu', 'movdqa', 'vmovdqu', 'vmovdqa', 'vmovdqu64', 'vmovdqa64') and not i.preserve and len(i.operands) == 2:
                d, s = i.operands
                if isinstance(d, M.R) and isinstance(s, M.R) and d.reg == s.reg and d.bits == s.bits:
                    continue
            ins.append(i)
        aliases = {}
        for i in ins:
            if (i.mnemonic == 'mov' and len(i.operands) == 2 and isinstance(i.operands[0], M.Mem)
                    and isinstance(i.operands[1], M.R) and i.operands[1].reg in aliases):
                old = i.operands[1]
                reg, bits = aliases[old.reg]
                if old.bits <= bits:
                    i.operands = (i.operands[0], M.R(reg, old.bits))
            aliases = {d: pair for d, pair in aliases.items() if d not in i.defs and pair[0] not in i.defs}
            if (i.mnemonic == 'mov' and len(i.operands) == 2 and all(isinstance(o, M.R) for o in i.operands)
                    and i.operands[0].reg.bank == 'gpr' and i.operands[0].bits == i.operands[1].bits
                    and not any(o.reg.id in ('rsp', 'rbp') for o in i.operands)):
                d, s = i.operands
                aliases[d.reg] = s.reg, d.bits
        for at in range(len(ins) - 1):
            move_, test = ins[at:at + 2]
            if (move_.mnemonic == 'mov' and len(move_.operands) == 2
                    and isinstance(move_.operands[0], M.R) and isinstance(move_.operands[1], M.Imm)
                    and move_.operands[1].symbol and test.mnemonic in ('cmp', 'test')
                    and move_.operands[0].reg not in test.uses):
                ins[at:at + 2] = [test, move_]
        if (ins and ins[-1].mnemonic == 'jmp' and ins[-1].target is not None
                and ins[-1].target is not next_ and len(ins[-1].target.ins) == 1
                and ins[-1].target.ins[0].mnemonic == 'ret' and not b.name.startswith('edge.')):
            ins[-1] = replace(ins[-1].target.ins[0])
        if (len(ins) >= 2 and ins[-1].mnemonic == 'jmp' and ins[-1].target is not None
                and ins[-2].cc and ins[-2].target is not None):
            branch, other = ins[-2], ins[-1]
            if branch.target is next_:
                cc = INVERSE[branch.cc]
                ins[-2:] = [replace(branch, mnemonic='j' + cc, cc=cc, target=other.target)]
            elif other.target is next_:
                ins.pop()
        elif ins and ins[-1].mnemonic == 'jmp' and ins[-1].target is next_ and next_ is not None:
            ins.pop()
        b.ins = ins
        for n, branch in enumerate(ins):
            if branch.target is None or branch.cc != 'be' or n == 0:
                continue
            comparison = ins[n - 1]
            if comparison.mnemonic != 'cmp' or comparison.keep_immediate or len(comparison.operands) != 2:
                continue
            lhs, rhs = comparison.operands
            if not isinstance(lhs, M.R) or not isinstance(rhs, M.Imm) or rhs.symbol is not None:
                continue
            c = rhs.value
            if not 0 <= c < min((1 << lhs.bits) - 1, 0x7fffffff) or c == 127 and lhs.bits > 8:
                continue
            comparison.operands = lhs, replace(rhs, value=c + 1)
            branch.mnemonic, branch.cc = 'jb', 'b'
    aliases, previous = {}, None
    def signature(i):
        return i.mnemonic, i.operands, i.cc, i.rep, i.target
    for b in order:
        if (previous is not None and b not in getattr(fn, 'retained_returns', ())
                and b.ins and len(b.ins) <= 3 + bool(getattr(fn, 'frame_pointer', False))
                and b.ins[-1].mnemonic == 'ret' and len(previous.ins) >= len(b.ins)):
            start = len(previous.ins) - len(b.ins)
            if [signature(i) for i in previous.ins[start:]] == [signature(i) for i in b.ins]:
                aliases[b] = previous, start
                b.ins = []
        if b.ins:
            previous = b
    # Empty edge blocks name the address of the next emitted instruction.
    starts, position = {}, 0
    for b in order:
        starts[b] = starts[aliases[b][0]] + aliases[b][1] * 16 if b in aliases else position
        position += len(b.ins) * 16
    result, offset = [], 0
    for b in order:
        for n, i in enumerate(b.ins):
            try:
                instruction = M.structured(i, offset, starts.get(i.target), isa=fn.isa)
                if (n == 0 and b in getattr(fn, 'loop_headers', ()) and b not in getattr(fn, 'unaligned_loops', ()) and b is not fn.entry
                        and (getattr(fn.source, 'shared_load_return', None) is None
                             or any(isinstance(o, M.Mem) for ins in b.ins for o in ins.operands))):
                    instruction.j['align_before'] = getattr(fn, 'loop_alignments', {}).get(b, 4)
                result.append(instruction)
            except encode.NoMatch as e:
                raise ValueError('%s: %s; operands %s' % (fn.name, e, i.operands)) from e
            offset += 16
    return result, max(offset, 1)


def object_model(functions, path, isa='sse2', trampolines=None, unit=None, frame_pointer=False):
    obj, code, constants = Object(), {}, {}
    obj.sections['.text'] = Section('.text', 'SHT_PROGBITS', 6, 4, b'')
    locals_ = [Symbol(path.name, 0, 0, 'STT_FILE', 'STB_LOCAL', 'ABS')]
    globals_ = []
    for ordinal, fn in enumerate(functions):
        if not hasattr(fn, 'machine_order'):
            fn.machine_order = ordinal
        selector = (CatSelect if getattr(fn, 'avx_cat', False) else
                    LongCopySelect if getattr(fn, 'avx_copy_long', False) else
                    CopySelect if getattr(fn, 'avx_copy_short', False) else
                    TailSelect if getattr(fn, 'avx_cmp_tail', False) else
                    CmpSelect if getattr(fn, 'avx_cmp', False) else
                    CspnSelect if getattr(fn, 'avx_cspn', False) else
                    SpanSelect if getattr(fn, 'avx_span', False) else
                    StringSelect if getattr(fn, 'string_prefixes', False) else Select)
        selected, pool = selector(fn, isa, trampolines).run()
        allocate(selected)
        parallel_copies(selected)
        span_parallel_copies(selected)
        long_parallel_copies(selected)
        execution_domains(selected)
        cleanup_edges(selected)
        schedule_counts(selected)
        fold_tail_addresses(selected)
        arithmetic_forms(selected)
        compact_byte_shift_counts(selected)
        schedule_short_vector_compare(selected)
        schedule_large_compare_tests(selected)
        share_tail_calls(selected)
        duplicate_tail_calls(selected)
        merge_store_tails(selected)
        share_mixed_width_return(selected)
        share_sibling_returns(selected)
        share_constant_epilogue(selected)
        share_byte_compare_epilogue(selected)
        search_return_counts(selected)
        frame(selected, frame_pointer)
        table_stack_addresses(selected)
        upper_state(selected)
        instructions, extent = linearize(selected)
        if instructions and hasattr(fn, 'tail_limit'):
            instructions[0].j['tail_limit'] = fn.tail_limit
        if instructions and hasattr(fn, 'post_skip_passes'):
            instructions[0].j['post_skip_passes'] = fn.post_skip_passes
        for instruction in instructions:
            if instruction.j.get('callee') and instruction.mnemonic == 'jmp':
                instruction.j['shrinkwrap_tail'] = (bool(getattr(selected, 'loop_headers', ()))
                                                   and instruction.j['callee'] not in getattr(fn, 'keep_tail_calls', ()))
        name = '.text.' + fn.name
        referenced = {o.symbol for b in selected.blocks for i in b.ins for o in i.operands if isinstance(o, M.Mem) and o.symbol}
        constant_symbols = []
        for data, constant in pool.items():
            used = constant in referenced
            if not used and not getattr(fn, 'retain_vector_constants', False):
                continue
            if constant.startswith('.LCPI'):
                size = len(data)
                section = '.rodata.cst%d' % size
                if section not in obj.sections:
                    obj.sections[section] = Section(section, 'SHT_PROGBITS', 0x12, size, b'', entsize=size)
                at = len(obj.sections[section].data)
                obj.sections[section].data += data
                # A broadcast may use a shorter pool fragment. Preserve the
                # original literal's bytes without an unused local label.
                if used:
                    locals_.append(Symbol(constant, at, 0, 'STT_NOTYPE', 'STB_LOCAL', section))
                    constant_symbols.append(constant)
            else:
                constants[constant] = data
        if instructions and constant_symbols:
            instructions[0].j['constant_symbols'] = constant_symbols
        obj.sections[name] = Section(name, 'SHT_PROGBITS', 6, 16, bytes(extent))
        locals_.append(Symbol(name, 0, 0, 'STT_SECTION', 'STB_LOCAL', name))
        symbol = Symbol(fn.name, 0, extent, 'STT_FUNC', 'STB_LOCAL' if 'static' in fn.decl.storage else 'STB_GLOBAL', name)
        (locals_ if symbol.bind == 'STB_LOCAL' else globals_).append(symbol)
        code[name] = instructions
    for name, data in constants.items():
        section = '.rodata.' + name[1:]
        alignment = len(data) if len(data) in (16, 32, 64) else 1
        obj.sections[section] = Section(section, 'SHT_PROGBITS', 2, alignment, data)
        locals_.append(Symbol(name, 0, len(data), 'STT_OBJECT', 'STB_LOCAL', section))
    obj.sections['.note.GNU-stack'] = Section('.note.GNU-stack', 'SHT_PROGBITS', 0, 1, b'')
    obj.symbols = [*locals_, *globals_]
    if unit is not None:
        emit_data(obj, unit)
    return obj, code


def emit_object(functions, path, isa='sse2', trampolines=None, unit=None):
    obj, code = object_model(functions, path, isa, trampolines, unit)
    layouts = {name: layout.layout(obj, name, ins, func_align=6, jcc_align=False) for name, ins in code.items()}
    remap = layout.resolve_relocs(obj, layouts)
    return elfout.write_model(obj, None, layouts, remap)
