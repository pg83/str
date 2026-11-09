"""Machine code: the blocks in an order that makes the likely paths fall
through, every instruction encoded (dev/encode.py), branches relaxed to
their short forms, the function's entry on its 64-byte line (at the measured
phase, dev/phases.json), loop headers on 16-byte boundaries, and, for the
code of the Skylake-SP configuration, no branch crossing or ending on a
32-byte boundary (the JCC erratum). Relocations against other functions, the
data and the constant pool are collected for the object."""
import encode

from .mir import Reg, Imm, Mem, Label, Sym, FLAGS, name as reg_name

NOPS = {1: '90', 2: '6690', 3: '0f1f00', 4: '0f1f4000', 5: '0f1f440000', 6: '660f1f440000', 7: '0f1f8000000000',
        8: '0f1f840000000000', 9: '660f1f840000000000', 10: '662e0f1f840000000000', 11: '66662e0f1f840000000000',
        12: '6666662e0f1f840000000000', 13: '666666662e0f1f840000000000', 14: '66666666662e0f1f840000000000',
        15: '6666666666662e0f1f840000000000'}
FUSIBLE = {'cmp', 'test', 'add', 'sub', 'and', 'inc', 'dec'}
INVERT = {'e': 'ne', 'ne': 'e', 'b': 'ae', 'ae': 'b', 'a': 'be', 'be': 'a', 'l': 'ge', 'ge': 'l', 'g': 'le', 'le': 'g',
          's': 'ns', 'ns': 's', 'o': 'no', 'no': 'o', 'p': 'np', 'np': 'p'}
# the EVEX names of VEX instructions when a register above 15 or a 512-bit one is in play
EVEX_NAMES = {'vpxor': 'vpxord', 'vpand': 'vpandd', 'vpor': 'vpord', 'vpandn': 'vpandnd', 'vmovdqa': 'vmovdqa64', 'vmovdqu': 'vmovdqu64',
              'vinserti128': 'vinserti32x4', 'vextracti128': 'vextracti32x4'}


def nops(n):
    out = b''
    while n > 15:
        out += bytes.fromhex(NOPS[15])
        n -= 15
    return out + (bytes.fromhex(NOPS[n]) if n else b'')


# ---------------------------------------------------------------- order

def order_blocks(mf):
    """Chains of likely successors; cold blocks (those ending in a trap, or
    reached only through unlikely branches) last."""
    preds = {b: [] for b in mf.blocks}
    for b in mf.blocks:
        for s in b.succs:
            preds[s].append(b)
    index = {b: n for n, b in enumerate(mf.blocks)}

    def likely(b):
        """b's successors, the likelier first."""
        succs = b.succs
        if len(succs) < 2:
            return list(succs)
        w = getattr(b, 'weights', None)
        if w and w[0] != w[1]:
            return [succs[0], succs[1]] if w[0] > w[1] else [succs[1], succs[0]]
        # a back edge is taken, an exit is not: stay in the loop
        for s in succs:
            if index[s] <= index[b] or b.loops and b.loops <= s.loops and not all(b.loops <= x.loops for x in succs):
                return [s] + [x for x in succs if x is not s]
        return list(succs)

    cold = set()
    for b in mf.blocks:
        if b.ins and b.ins[-1].kind == 'trap':
            cold.add(b)
    changed = True
    while changed:
        changed = False
        for b in mf.blocks:
            if b in cold or b is mf.entry:
                continue
            ps = preds[b]
            if ps and all(p in cold or (getattr(p, 'weights', None) and p.succs[p.succs.index(b)] is b
                                        and p.weights[p.succs.index(b)] * 10 < p.weights[1 - p.succs.index(b)]) for p in ps):
                cold.add(b)
                changed = True
    # a loop whose header tests the exit and whose latches only jump back:
    # the header goes after the body (the back edge its own branch, the exit
    # the fall-through), the entry jumping over the body to it
    rotate = set()
    for h in mf.blocks:
        if not h.loop_header or len(h.succs) != 2 or h is mf.entry:
            continue
        latches = [p for p in preds[h] if h in p.loops]
        if len(latches) != len(preds[h]) - 1 or not latches:
            continue
        if not all(len(p.succs) == 1 and p.ins and p.ins[-1].kind == 'jmp' for p in latches):
            continue
        inside = [s for s in h.succs if h in s.loops]
        if len(inside) == 1 and likely(h)[0] is inside[0]:
            rotate.add(h)
    placed, out = set(), []
    pending = [mf.entry]
    deferred = set()
    while len(out) < len(mf.blocks):
        if not pending:
            rest = [b for b in mf.blocks if b not in placed]
            hot = [b for b in rest if b not in cold]
            pending.append(min(hot or rest, key=lambda b: index[b]))
        b = pending.pop(0)
        if b in placed:
            continue
        prev = None
        while b is not None and b not in placed:
            if b in rotate and b not in deferred and (prev is None or b not in prev.loops):
                deferred.add(b)
                pending.append(b)
                prev, b = b, [s for s in likely(b) if b in s.loops][0]
                continue
            placed.add(b)
            out.append(b)
            nxt = None
            for s in likely(b):
                if s not in placed:
                    if nxt is None and (s not in cold or all(x in cold for x in b.succs)):
                        nxt = s
                    else:
                        pending.append(s)
            prev, b = b, nxt
    mf.blocks = out


# -------------------------------------------------------------- encoding

class Item:
    """One instruction to encode: its structured form, its branch target block, its relocation."""
    __slots__ = ('j', 'target', 'reloc', 'size', 'offset', 'fields', 'kind', 'fused', 'ins')

    def __init__(self, j, target=None, reloc=None, kind=None, ins=None):
        self.j, self.target, self.reloc, self.kind, self.ins = j, target, reloc, kind, ins
        self.size, self.offset, self.fields, self.fused = 0, 0, {}, False


def structured(ins, isa):
    """The encoder's instruction from a MIR instruction; (j, target block, relocation request)."""
    ops, target, reloc = [], None, None
    evex_regs = any(isinstance(o, Reg) and isinstance(o.r, str) and o.r[0] == 'v' and (int(o.r[1:]) >= 16 or o.bits == 512) for o in ins.ops)
    mnemonic = ins.op
    if evex_regs and mnemonic in EVEX_NAMES:
        mnemonic = EVEX_NAMES[mnemonic]
    for o in ins.ops:
        if isinstance(o, Reg):
            ops.append({'kind': 'Register', 'reg': reg_name(o.r, o.bits), 'size': o.bits, 'implicit': False})
        elif isinstance(o, Imm):
            ops.append({'kind': 'Immediate', 'imm': o.value, 'size': o.bits, 'signed': o.value < 0, 'implicit': False})
        elif isinstance(o, Mem):
            if o.slot is not None:
                raise ValueError('a stack slot was not laid out')
            ops.append({'kind': 'Memory', 'mem': {'base': o.base if o.base is not None else 'none', 'index': o.index if o.index is not None else 'none',
                                                  'scale': o.scale, 'disp': 0 if o.sym else o.disp, 'size': o.size,
                                                  'rip_rel': o.sym is not None, 'seg': None}, 'field_pos': None})
            if o.sym is not None:
                reloc = ('disp', 'R_X86_64_PC32', o.sym, o.disp)
        elif isinstance(o, Label):
            ops.append({'kind': 'NearBranch', 'branch': 0, 'size': 32, 'field_pos': None})
            target = o.block
        elif isinstance(o, Sym):
            ops.append({'kind': 'NearBranch', 'branch': 0, 'size': 32, 'field_pos': None})
            reloc = ('branch', 'R_X86_64_PLT32', o.name, 0)
        else:
            raise ValueError('operand %r' % (o,))
    j = {'mnemonic': mnemonic + (ins.cc or ''), 'cc': ins.cc, 'ops': ops, 'form': None, 'offset': 0,
         'op_mask': ins.mask if ins.mask is not None else 'none', 'zeroing': ins.zero, 'bcast': False,
         'rep': ins.rep, 'lock': False, 'notrack': False}
    if isa in ('sse2', 'sse42'):
        j['encoding'] = 'Legacy'
    elif isa == 'avx2':
        j['encoding'] = 'VEX'
    return j, target, reloc


def assemble(mf, phase=0, jcc_align=False, loop_align=16):
    """mf's blocks (already ordered) to bytes; returns (code, relocs, labels) with
    relocs [(offset, type, symbol, addend)] and labels {block: offset}."""
    order = mf.blocks
    position = {b: n for n, b in enumerate(order)}
    headers = {b for b in order if b.loop_header}
    items = []
    for n, b in enumerate(order):
        following = order[n + 1] if n + 1 < len(order) else None
        items.append(('block', b))
        for i in b.ins:
            if i.kind == 'jmp':
                if i.ops[0].block is following:
                    continue
                j, target, reloc = structured(i, mf.isa)
                items.append(('ins', Item(j, target, reloc, 'jmp', i)))
            elif i.kind == 'jcc':
                yes, no = i.ops[0].block, i.ops[1].block
                if no is following:
                    j, _, _ = structured(i, mf.isa)
                    j['ops'] = j['ops'][:1]
                    items.append(('ins', Item(j, yes, None, 'jcc', i)))
                elif yes is following:
                    j, _, _ = structured(i, mf.isa)
                    j['ops'] = j['ops'][:1]
                    j['cc'] = INVERT[i.cc]
                    j['mnemonic'] = 'j' + j['cc']
                    items.append(('ins', Item(j, no, None, 'jcc', i)))
                else:
                    j, _, _ = structured(i, mf.isa)
                    j['ops'] = j['ops'][:1]
                    items.append(('ins', Item(j, yes, None, 'jcc', i)))
                    jj = {'mnemonic': 'jmp', 'cc': None, 'ops': [{'kind': 'NearBranch', 'branch': 0, 'size': 32, 'field_pos': None}],
                          'form': None, 'offset': 0, 'op_mask': 'none', 'zeroing': False, 'bcast': False, 'rep': None, 'lock': False, 'notrack': False}
                    items.append(('ins', Item(jj, no, None, 'jmp', None)))
            else:
                j, target, reloc = structured(i, mf.isa)
                items.append(('ins', Item(j, target, reloc, i.kind, i)))
    # sizes: short branches first, grown where the distance needs rel32
    short = {id(it): True for kind, it in items if kind == 'ins' and it.target is not None}
    for kind, it in items:
        if kind == 'ins' and it.target is None:
            size = 32 if it.reloc and it.reloc[0] == 'branch' else None
            b, fields = encode.encode(it.j, None, size)
            it.size, it.fields = len(b), fields
    while True:
        labels = {}
        offset = 0
        pads = {}
        for n, (kind, x) in enumerate(items):
            if kind == 'block':
                pad = 0
                if x is mf.entry:
                    pad = (phase - offset) % 64
                elif x in headers and loop_align:
                    pad = (-offset) % loop_align
                pads[id(x)] = pad
                offset += pad
                labels[x] = offset
                continue
            it = x
            if it.target is not None:
                it.size = 2 if short[id(it)] else (5 if it.kind == 'jmp' else 6)
            pad = 0
            if jcc_align:
                unit = it.size
                fused = False
                if it.j['mnemonic'] in FUSIBLE and n + 1 < len(items) and items[n + 1][0] == 'ins' and items[n + 1][1].kind == 'jcc':
                    nxt = items[n + 1][1]
                    unit += 2 if short[id(nxt)] else 6
                    fused = True
                if it.kind in ('jmp', 'jcc', 'call', 'tail', 'ret') or fused:
                    if unit <= 32 and (offset // 32 != (offset + unit - 1) // 32 or (offset + unit) % 32 == 0):
                        pad = (-offset) % 32
            pads[id(it)] = pad
            offset += pad
            it.offset = offset
            offset += it.size
        end = offset
        grown = False
        for kind, it in items:
            if kind == 'ins' and it.target is not None and short[id(it)]:
                rel = labels[it.target] - (it.offset + it.size)
                if not -128 <= rel <= 127:
                    short[id(it)] = False
                    grown = True
        if not grown:
            break
    code = bytearray()
    relocs = []
    for kind, x in items:
        pad = pads[id(x)]
        if pad:
            code += nops(pad)
        if kind == 'block':
            continue
        it = x
        assert len(code) == it.offset, (mf.name, len(code), it.offset)
        if it.target is not None:
            rel = labels[it.target] - (it.offset + it.size)
            b, fields = encode.encode(it.j, rel, 8 if short[id(it)] else 32)
        else:
            b, fields = encode.encode(it.j, None, 32 if it.reloc and it.reloc[0] == 'branch' else None)
        assert len(b) == it.size, (mf.name, it.j['mnemonic'], len(b), it.size)
        if it.reloc:
            field, rtype, symbol, addend = it.reloc
            pos, width = fields[field]
            relocs.append((it.offset + pos, rtype, symbol, addend - (len(b) - pos)))
        code += b
    return bytes(code), relocs, labels
