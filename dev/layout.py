"""Lay a code section out again from its (rewritten) instructions: encode each
one, relax branches to their short form where the distance allows, keep the
alignment padding the compiler chose, map the old offsets (labels, symbols,
relocation targets) to the new ones and regenerate the relocations.

layout(obj, section, instructions) -> Layout with
  data        the section's bytes
  offsets     {old offset: new offset} for every original instruction start
              (deleted ones map to what follows) and for the old section end
then resolve_relocs(obj, {section: Layout}) fills
  relocs      [(new offset, type, symbol name, symbol_is_section, addend)]
and returns the remap(section, old offset) function for symbols and data.
"""
import os

import encode
from objfile import Section

# The nop sequences the assemblers use for padding, by length.
NOPS = {1: '90', 2: '6690', 3: '0f1f00', 4: '0f1f4000', 5: '0f1f440000', 6: '660f1f440000', 7: '0f1f8000000000',
        8: '0f1f840000000000', 9: '660f1f840000000000', 10: '662e0f1f840000000000', 11: '66662e0f1f840000000000',
        12: '6666662e0f1f840000000000', 13: '666666662e0f1f840000000000', 14: '66666666662e0f1f840000000000',
        15: '6666666666662e0f1f840000000000'}


def nops(n):
    out = b''
    while n > 15:
        out += bytes.fromhex(NOPS[15])
        n -= 15
    return out + (bytes.fromhex(NOPS[n]) if n else b'')


FUSIBLE = {'cmp', 'test', 'add', 'sub', 'and', 'inc', 'dec'}


def branch_length(i, size):
    if size == 8:
        return 2
    return 5 if i.mnemonic in ('jmp', 'call') else 6


class Layout:
    def __init__(self):
        self.data = b''
        self.offsets = {}
        self.relocs = []
        self.pending = []
        self.branch_sizes = {}


def padding_runs(obj, section, instructions):
    """{offset of a padding nop: (alignment or None, original bytes of the run)} — the run's last nop
    carries the alignment, the others None. Same rule as before: a run of nops ending on an
    8-byte boundary at a branch target, a symbol or outside any function."""
    covered = set()
    for f in obj.functions(section):
        covered.update(range(f.value, f.value + f.size))
    starts = {s.value for s in obj.symbols if s.section == section}
    targets = {i.target for i in instructions if i.target is not None and i.reloc is None}
    for sec in obj.sections.values():
        for r in sec.relocs:
            if r.symbol_is_section and r.symbol == section:
                targets.add(r.addend)
    for i in instructions:
        r = i.reloc
        if r is not None and r.symbol_is_section and r.symbol == section and r.type in ('R_X86_64_PC32', 'R_X86_64_PLT32'):
            targets.add(r.addend + i.home_end - r.offset)
    padding = {}
    run = []

    def alignment(run):
        length = run[-1].end - run[0].offset
        k = max(length.bit_length(), 4)
        return min(k, (run[-1].end & -run[-1].end).bit_length() - 1, 6)

    def flush(run):
        if run and run[-1].end % 8 == 0 and (run[-1].end in targets or run[-1].end in starts or run[-1].end not in covered):
            for n in run:
                padding[n.offset] = (None, b'')
            padding[run[-1].offset] = (alignment(run), b''.join(bytes.fromhex(n.j['bytes']) for n in run))
        elif run and run[-1].end not in covered:
            for n in run:
                padding[n.offset] = (None, b'')
    for i in instructions:
        if i.mnemonic == 'nop' and i.gas and i.offset == i.home:
            run.append(i)
            continue
        flush(run)
        run = []
    flush(run)
    return padding


def layout(obj, section, instructions, func_align=None, jcc_align=None, source=None, phase=0):
    """jcc_align: pad branches off 32-byte boundaries (the Skylake JCC erratum); None: POST_JCC_ALIGN (default on).
    source: the object's section the instructions came from, when they are written as a new one (a clone).
    phase: the function entry's offset within its aligned line (dev/phases.json, measured per host class)."""
    section = source or section
    sec = obj.sections[section]
    old_end = len(sec.data)
    padding = padding_runs(obj, section, instructions)
    func_starts = {f.value for f in obj.functions(section)}
    func_align = int(func_align) if func_align else (int(os.environ['POST_FUNC_ALIGN']) if os.environ.get('POST_FUNC_ALIGN') else None)
    internal = {}
    for i in instructions:
        if i.target is not None and i.reloc is None and 0 <= i.target < old_end and i.gas:
            internal[id(i)] = i.target
    # branch sizes: the decoded form where there is one, short otherwise (grown as needed);
    # a branch or call with a relocation is rel32 whatever its text says: the field the
    # linker fills is 4 bytes (a pass may have given it a new target and no form)
    bsize = {}
    for i in instructions:
        if id(i) in internal:
            form = i.j.get('form')
            bsize[id(i)] = form['imm'] if form and form.get('imm') in (8, 32) else i.j.get('branch_size', 8)
        elif i.reloc is not None and i.flow in ('ConditionalBranch', 'UnconditionalBranch', 'Call'):
            bsize[id(i)] = 32
    # new instructions marked 'before' belong to the original instruction after them: its
    # label (and a function symbol) must point at the first of them
    leading = set()
    run = []
    for i in instructions:
        if i.offset != i.home and i.gas and i.j.get('before'):
            run.append(i)
        elif i.offset == i.home:
            if run:
                leading.add(id(run[0]))
            run = []
        else:
            run = []
    if jcc_align is None:
        jcc_align = os.environ.get('POST_JCC_ALIGN', '1') == '1'
    offsets = {}
    encoded = {}
    while True:
        off = 0
        offsets = {}
        items = []
        placed_as_pair = set()
        pending = None   # where the function started, when new instructions came first
        live = [i for i in instructions if i.gas]
        following = {id(a): b for a, b in zip(live, live[1:])}
        leads_function = set()
        run_start = None
        for i in instructions:
            if id(i) in leading:
                run_start = id(i)
            if i.offset == i.home:
                if i.offset in func_starts and run_start is not None:
                    leads_function.add(run_start)
                run_start = None
        for i in instructions:
            if id(i) in placed_as_pair:
                continue
            starts_function = (i.offset in func_starts and i.offset == i.home and id(i) not in leading) or id(i) in leads_function
            if starts_function and func_align:
                pad = (int(os.environ.get('POST_FUNC_PHASE', phase)) - off) % (1 << func_align)
                if pad:
                    items.append(('pad', nops(pad)))
                    off += pad
            if i.j.get('align_before') and not starts_function:
                pad = (-off) % (1 << i.j['align_before'])
                if pad:
                    items.append(('pad', nops(pad)))
                    off += pad
            if id(i) in leading:
                pending = off
            if i.offset == i.home:
                offsets.setdefault(i.offset, pending if pending is not None else off)
                pending = None
            if not i.gas:
                continue
            if i.offset in padding:
                k, original = padding[i.offset]
                if k is None:
                    continue
                pad = (-off) % (1 << k)
                blob = original if len(original) == pad else nops(pad)
                items.append(('pad', blob))
                off += pad
                continue
            # sizing pass: lengths only, the displacements come once every target is placed
            size = bsize.get(id(i))
            b, fields = encode.encode(i.j, None, size)
            # Skylake-era JCC erratum: a branch, or a flag-setting instruction fused with the
            # branch after it, must not cross or end on a 32-byte boundary, else the decoded
            # instruction cache cannot hold it. Pad up to the boundary in front of it.
            unit = [(i, b, fields)]
            nxt = following.get(id(i))
            if jcc_align and i.mnemonic in FUSIBLE and nxt is not None and nxt.flow == 'ConditionalBranch' and id(nxt) not in padding:
                nb, nf = encode.encode(nxt.j, None, bsize.get(id(nxt)))
                unit.append((nxt, nb, nf))
            if jcc_align and (unit[-1][0].flow in ('ConditionalBranch', 'UnconditionalBranch', 'Call', 'Return', 'IndirectBranch', 'IndirectCall')):
                length = sum(len(x[1]) for x in unit)
                if length <= 32 and (off // 32 != (off + length - 1) // 32 or (off + length) % 32 == 0):
                    pad = (-off) % 32
                    items.append(('pad', nops(pad)))
                    off += pad
            for u, ub, uf in unit:
                encoded[id(u)] = (off, ub, uf)
                items.append(('ins', u))
                off += len(ub)
                if u is not i:
                    placed_as_pair.add(id(u))
        offsets.setdefault(old_end, off)
        # every target known now: check the short branches and the final displacements
        grown = False
        for i in instructions:
            if id(i) in internal and bsize[id(i)] == 8:
                off_i, b, _ = encoded[id(i)]
                rel = offsets[internal[id(i)]] - (off_i + len(b))
                if not -128 <= rel <= 127:
                    bsize[id(i)] = 32
                    grown = True
        if not grown:
            break
    # final pass: encode with the settled offsets
    out = Layout()
    out.offsets = offsets
    out.branch_sizes = {i.offset: bsize[id(i)] for i in instructions if id(i) in bsize}
    data = bytearray()
    for kind, x in items:
        if kind == 'pad':
            data += x
            continue
        i = x
        off = len(data)
        rel = None
        size = None
        if id(i) in internal:
            size = bsize[id(i)]
            rel = offsets[internal[id(i)]] - (off + branch_length(i, size))
        else:
            size = bsize.get(id(i))
        b, fields = encode.encode(i.j, rel, size)
        assert encoded[id(i)][0] == off, (i, encoded[id(i)][0], off)
        data += b
        if i.reloc is not None:
            out.pending.append((off, len(b), fields, i))
    out.data = bytes(data)
    return out


def resolve_relocs(obj, layouts):
    """Second pass, once every code section is laid out: relocations with the
    new field offsets and addends; targets in code sections mapped to their
    new offsets."""

    def remap(section, old):
        lay = layouts.get(section)
        if lay is None:
            return old
        if old in lay.offsets:
            return lay.offsets[old]
        raise ValueError('%s+%#x is not an instruction boundary' % (section, old))

    for lay in layouts.values():
        lay.relocs = []
        for off, length, fields, i in lay.pending:
            r = i.reloc
            pos = r.offset - i.home
            op = next((o for o in i.j['ops'] if o.get('field_pos') == pos), None)
            if op is None:
                raise ValueError('relocation %s at %d does not match an operand of %s' % (r.type, pos, i))
            field = {'NearBranch': 'branch', 'Memory': 'disp', 'Immediate': 'imm'}[op['kind']]
            fpos, fsize = fields[field]
            code_symbol = r.symbol_is_section and obj.sections[r.symbol].is_code
            if r.type in ('R_X86_64_PC32', 'R_X86_64_PLT32'):
                effective = r.addend + (i.home_end - r.offset)   # what the old instruction pointed at, relative to the symbol
                if code_symbol:
                    effective = remap(r.symbol, effective)
                addend = effective - (length - fpos)
            else:
                addend = remap(r.symbol, r.addend) if code_symbol else r.addend
            lay.relocs.append((off + fpos, r.type, r.symbol, r.symbol_is_section, addend))
    return remap
