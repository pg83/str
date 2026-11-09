"""One function from the IR to bytes: selection, registers, frame, order, encoding."""
from .asm import assemble, order_blocks
from .frame import frame, vzeroupper
from .isel import select
from .mir import Label, Reg, Imm, Mem, FLAGS, FLAG_WRITERS
from .regalloc import allocate, dead_code, liveness

FLAG_READERS = {'cmov', 'set', 'adc', 'sbb', 'j'}


def peephole(mf):
    """Where the flags are dead: a zero moved into a 32-bit register is an xor
    (shorter, no immediate); lea r, [r + c] is add r, c (lea runs on the
    ports the shuffles want on Intel, add on any)."""
    for b in mf.blocks:
        live = False      # the flags, walking backwards; dead at a block's end (a compare sits with its branch)
        for i in reversed(b.ins):
            if i.op in FLAG_READERS or i.cc is not None and i.op != 'j' and i.kind != 'jcc':
                live = True
            if i.kind == 'jcc':
                live = True
            if i.op == 'mov' and not live and len(i.ops) == 2 and isinstance(i.ops[0], Reg) and isinstance(i.ops[1], Imm) \
                    and i.ops[1].value == 0 and i.ops[0].bits == 32:
                i.op, i.ops, i.acc, i.kind = 'xor', [i.ops[0], Reg(i.ops[0].r, 32)], ['w', 'r'], 'zero'
            if i.op == 'lea' and not live and len(i.ops) == 2 and isinstance(i.ops[0], Reg) and isinstance(i.ops[1], Mem):
                m = i.ops[1]
                if m.index is None and m.sym is None and m.slot is None and m.base == i.ops[0].r and -(1 << 31) <= m.disp < (1 << 31) and m.disp != 0:
                    i.op, i.ops, i.acc, i.kind = 'add', [i.ops[0], Imm(m.disp, 32)], ['rw', 'r'], None
            if i.op in FLAG_WRITERS or FLAGS in i.imp_def:
                live = False


def thread_jumps(mf):
    """A block that only jumps on (its copies coalesced away) is skipped by its predecessors."""
    target = {}
    for b in mf.blocks:
        if b is not mf.entry and len(b.ins) == 1 and b.ins[0].kind == 'jmp':
            target[b] = b.ins[0].ops[0].block

    def final(b):
        seen = set()
        while b in target and b not in seen:
            seen.add(b)
            b = target[b]
        return b

    for b in mf.blocks:
        for i in b.ins:
            for o in i.ops:
                if isinstance(o, Label):
                    o.block = final(o.block)
        b.succs = [final(s) for s in b.succs]
    for b in target:
        if b.loop_header:
            final(b).loop_header = True
    reached, todo = {mf.entry}, [mf.entry]
    while todo:
        b = todo.pop()
        for s in b.succs:
            if s not in reached:
                reached.add(s)
                todo.append(s)
    mf.blocks = [b for b in mf.blocks if b in reached]


def hoist_latch_copies(mf):
    """A loop's latch that only copies (the phis' values, left after the
    allocation) and jumps to the header: the copies go into the body before
    its exit branch when they write nothing the exit reads, so the branch
    back is the body's own and the latch disappears."""
    preds = {b: 0 for b in mf.blocks}
    for b in mf.blocks:
        for s in b.succs:
            preds[s] += 1
    live_in, _ = liveness(mf)
    for b in mf.blocks:
        if len(b.ins) < 2 or b.ins[-1].kind != 'jcc':
            continue
        jcc = b.ins[-1]
        targets = [o.block for o in jcc.ops if isinstance(o, Label)]
        for latch in targets:
            other = targets[1] if latch is targets[0] else targets[0]
            if latch is b or latch is other or preds[latch] != 1 or not latch.ins or latch.ins[-1].kind != 'jmp':
                continue
            copies = latch.ins[:-1]
            if not copies or not all(i.kind in ('copy', 'copy32') and all(isinstance(o, Reg) for o in i.ops) for i in copies):
                continue
            defs = {d for i in copies for d in i.defs()}
            if defs & live_in[other]:
                continue
            # before the compare unless it reads what the copies write (then between it and the jump)
            at = len(b.ins) - 1
            if b.ins[at - 1].kind == 'cmp' and not defs & set(b.ins[at - 1].uses()):
                at -= 1
            b.ins[at:at] = copies
            latch.ins = latch.ins[-1:]
            break


def compile_function(fn, isa, phase=0, jcc_align=False):
    mf = select(fn, isa)
    allocate(mf)
    dead_code(mf)
    hoist_latch_copies(mf)
    thread_jumps(mf)
    peephole(mf)
    frame(mf)
    vzeroupper(mf)
    order_blocks(mf)
    code, relocs, labels = assemble(mf, phase=phase, jcc_align=jcc_align)
    return mf, code, relocs
