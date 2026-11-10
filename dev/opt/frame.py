"""The stack frame and the AVX state: after the registers are assigned.

The callee-saved registers the function writes are pushed at the entry and
popped before every return or tail jump; the stack slots get their offsets
from rsp. A leaf function keeps its slots in the red zone below rsp; one
that calls, or needs more than the zone holds, lowers rsp by a multiple of
16 (so that calls see an aligned stack). A function of an AVX unit that
dirtied the upper half of a ymm0-15 register executes vzeroupper before it
returns or calls."""
from . import mir
from .mir import Reg, Mem, Imm, Ins, VEC

RED_ZONE = 128


def layout_slots(mf):
    """Offsets of the slots within their region; (total, alignment)."""
    slots = sorted(mf.slots, key=lambda s: -s.align)
    offset = 0
    for s in slots:
        offset = (offset + s.align - 1) & -s.align
        s.offset = offset
        offset += s.size
    align = max([s.align for s in slots] + [8])
    return (offset + align - 1) & -align, align


def frame(mf):
    saved = [r for r in mir.abi.callee_saved if r in mf.used]
    calls = any(i.kind == 'call' for b in mf.blocks for i in b.ins)
    total, align = layout_slots(mf)
    rsp_mod16 = (8 + 8 * len(saved)) % 16          # rsp after the pushes
    leaf = not calls and total + rsp_mod16 <= RED_ZONE
    if leaf:
        adjust = 0
        if total:
            # the region ends at the 16-aligned address below rsp
            base_disp = -(rsp_mod16 + total)
        else:
            base_disp = 0
    else:
        # rsp ends 16-aligned: the region's size (a multiple of 16) plus what the pushes left over
        adjust = total + rsp_mod16
        base_disp = 0
    for b in mf.blocks:
        for i in b.ins:
            for o in i.ops:
                if isinstance(o, Mem) and o.slot is not None:
                    o.disp += base_disp + o.slot.offset
                    o.slot = None
                    if o.base is None:
                        o.base = 'rsp'
                    elif o.index is None:
                        o.index, o.base = o.base, 'rsp'
                    else:
                        raise ValueError('%s: a stack slot addressed with two registers' % mf.name)
    prologue = [Ins('push', [Reg(r, 64)], 'r', kind='push') for r in saved]
    if adjust:
        prologue.append(Ins('sub', [Reg('rsp', 64), Imm(adjust, 32)], 'rw r'))
    epilogue = []
    if adjust:
        epilogue.append(Ins('add', [Reg('rsp', 64), Imm(adjust, 32)], 'rw r'))
    epilogue.extend(Ins('pop', [Reg(r, 64)], 'w', kind='pop') for r in reversed(saved))
    mf.entry.ins[0:0] = prologue
    for b in mf.blocks:
        out = []
        for i in b.ins:
            if i.kind in ('ret', 'tail'):
                out.extend(Ins(e.op, list(e.ops), list(e.acc), kind=e.kind) for e in epilogue)
            out.append(i)
        b.ins = out
    mf.frame_size = adjust
    mf.saved = saved


def stack_changes(mf):
    """How far rsp is below its value at the entry, from each instruction that changes it on,
    as [(offset of its end, bytes)] (from mf.ends, after assemble): the prologue's pushes and
    subtraction, an epilogue's addition and pops, and the body's frame again after a return
    or a tail jump, since what follows one is reached from the body. The unwind tables of an
    object for wasm2o, which runs the code on goroutine stacks, are made of it."""
    body = 8 * len(mf.saved) + mf.frame_size
    below, out = 0, []
    for end, i in mf.ends:
        now = below
        if i is None:
            pass
        elif i.op == 'push':
            now += 8
        elif i.op == 'pop':
            now -= 8
        elif i.op in ('sub', 'add') and isinstance(i.ops[0], Reg) and i.ops[0].r == 'rsp':
            if not isinstance(i.ops[1], Imm):
                raise ValueError('%s: rsp changed by a register' % mf.name)
            now += i.ops[1].value if i.op == 'sub' else -i.ops[1].value
        elif i.kind in ('ret', 'tail'):
            now = body
        elif 'rsp' in i.defs():
            raise ValueError('%s: %s writes rsp' % (mf.name, i))
        if now != below:
            out.append((end, now))
            below = now
    return out


def dirties(i):
    """Writes the upper half of one of ymm0-15."""
    for o, a in zip(i.ops, i.acc):
        if isinstance(o, Reg) and 'w' in a and isinstance(o.r, str) and o.r[0] == 'v' and int(o.r[1:]) < 16 and o.bits >= 256:
            return True
    return False


def vzeroupper(mf):
    if mf.isa not in ('avx2', 'evex', 'evex512'):
        return
    # forward dataflow: dirty at the start of a block if dirty at the end of a predecessor
    dirty_out = {b: False for b in mf.blocks}
    dirty_in = {b: False for b in mf.blocks}
    preds = {b: [] for b in mf.blocks}
    for b in mf.blocks:
        for s in b.succs:
            preds[s].append(b)
    changed = True
    while changed:
        changed = False
        for b in mf.blocks:
            d = any(dirty_out[p] for p in preds[b])
            for i in b.ins:
                if i.kind in ('call', 'ret', 'tail'):
                    d = False
                elif dirties(i):
                    d = True
            if d != dirty_out[b] or any(dirty_out[p] for p in preds[b]) != dirty_in[b]:
                dirty_out[b], dirty_in[b] = d, any(dirty_out[p] for p in preds[b])
                changed = True
    for b in mf.blocks:
        d = dirty_in[b]
        out = []
        for i in b.ins:
            if i.kind in ('call', 'ret', 'tail'):
                if d:
                    # before the epilogue's pops, not between them and the return
                    at = len(out)
                    while at and out[at - 1].kind in ('pop',) or at and out[at - 1].op == 'add' and out[at - 1].ops and isinstance(out[at - 1].ops[0], Reg) and out[at - 1].ops[0].r == 'rsp':
                        at -= 1
                    out.insert(at, Ins('vzeroupper', kind='vzeroupper'))
                d = False
            elif dirties(i):
                d = True
            out.append(i)
        b.ins = out
