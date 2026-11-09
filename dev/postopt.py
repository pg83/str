"""Post-compiler passes over iced-decoded .text.

The compiler cannot be told from C to use ymm16-31 (so every function that
touches a ymm register ends in vzeroupper) and never folds a masked load into
the memory operand of a masked compare. Both are done here, on the decoded
instructions, function by function, with a small CFG and register liveness to
prove the rewrites safe:

  rename   in a function where no vector register is live at entry and every
           instruction touching a 256- or 512-bit register has an EVEX form,
           rename the register webs (def-use connected components) that
           contain a ymm or zmm access to 16-31 and drop vzeroupper: nothing dirties the upper
           halves of ymm0-15 any more, which is exactly the condition under
           which the compiler would not have emitted it. Webs that only ever
           use xmm stay VEX-encoded (shorter) - xmm writes do not dirty state.
  fold     vmovdqu8 (p),X{M}{z}; vmovdqu8 (q),Y{M}{z}; vpcmp<pred>b Y,X,K  ->
           vmovdqu8 (p),X{M}{z}; vpcmp<pred>b (q),X,K{M}
           when X is dead afterwards and pred(0,0) is false, so masked-off lanes
           give 0 either way. The masked memory operand suppresses faults on
           masked-off bytes just as the masked load did. K is then a subset of
           M, so a later kand M,K,K is dropped.
  tail     jmp L where L starts a block of at most 3 instructions ending in
           ret or a tail jump: the block is copied over the jmp (one taken
           branch less on the path); the original is dropped if that jmp was
           its only way in.

  hot      jcc L where L starts a short tail ending in ret and the fall-through
           does not: invert the condition, put the tail in line and branch to
           the old fall-through. Short inputs (the bulk of the weight) return
           early, so the early return should be the fall-through path.
  ktest    kortest k,k; je/jne L  ->  kmov k,r; test r,r; je/jne L with a free
           GPR r, and when L begins with the same kmov k,r on every way in,
           that kmov goes away: test+jcc fuse, kortest+jcc do not.
  kor      kortest kA,kB; je/jne, where the only way into one successor is
           this branch and it starts with kor kA,kB,kD; kmov kD,r: the kor and
           kmov move in front of the branch and the kortest becomes test r,r.
           The compiler tests the or of two masks with kortest and then builds
           it again for the index; on the path that needs the index (the
           common one in strcmp) that was three mask ops after the load.
  kfirst   vpcmp (mem),X,kA; vptestm X,X,kB{kA}  ->  vptestm X,X,kA; vpcmp (mem),X,kB{kA}
           (any two mask-producing compares, the first with a memory operand
           and the second masked by its result and free of memory): the same
           and of the two masks, but the compare that waits for the load no
           longer has the other one behind it. The critical path from the
           loads to the mask is one operation shorter (strcmp, memcmp).
  tramp    jmp/call F where F, in another object of the same run, is nothing
           but jmp *slot(%rip) (the str_* entry points of dispatch.c): go
           through the slot directly, jmp/call *slot(%rip). The sources call
           the entry points; the code jumps through the table.
  lea      lea (a,b),c; add $imm,c -> lea imm(a,b),c.
  copy     mov A,B at the entry (the ABI shuffle): reads of B become reads of A
           where both are unchanged, the copy moves to the places that still
           need B. Measured neutral (the copy is eliminated at rename anyway).
  zcmp     load + vptestnmb -> vpcmpeqb against a zeroed register; off, see
           zero_compares.
  frame    push %rbp; mov %rsp,%rbp at the entry and every pop %rbp or leave,
           in a function that calls nothing and touches rbp and rsp nowhere
           else: a build whose environment asks for frame pointers everywhere
           (ix: -fno-omit-frame-pointer, after the library's own flags) gives
           every leaf of the library three instructions it does not need; the
           post pass drops .eh_frame anyway.
  vzu      vzeroupper where no path from the entry writes a 256- or 512-bit
           register (calls count as writes): the compiler puts one before every
           ret of a function that touches ymm anywhere, also on the paths of a
           few bytes that only use xmm and general registers.
The layout (layout.py) keeps branches and fused compare+branch pairs off
32-byte boundaries (the Skylake JCC erratum); POST_JCC_ALIGN=0 turns it off.

Passes are selected by POST_PASSES (comma list, default all) for A/B runs.

Library convention relied on: no function takes or returns vector registers,
so calls and tail jumps neither read nor keep them.
"""
import copy
import os
import re
from dataclasses import dataclass, field

import x86
from encode import HIGH8
from objfile import Ins, Reloc


def refresh(i):
    """After a rewrite of the structured operands: no encoding hint any more, text regenerated."""
    i.j['form'] = None
    i.j['gas'] = x86.format(i.j)


def register(name, size):
    return {'kind': 'Register', 'reg': name, 'size': size, 'implicit': False}


def full_vec(name):
    return 'zmm' + name[3:]

VEC = re.compile(r'%([xy])mm(\d+)\b')
# VEX mnemonics whose EVEX form has another name.
EVEX_NAME = {'vmovdqu': 'vmovdqu64', 'vmovdqa': 'vmovdqa64', 'vpxor': 'vpxord', 'vpor': 'vpord',
             'vpand': 'vpandd', 'vpandn': 'vpandnd', 'vinserti128': 'vinserti32x4', 'vextracti128': 'vextracti32x4'}
# VEX mnemonics that keep their name in EVEX (the assembler picks EVEX for ymm16-31).
EVEX_SAME = {'vmovups', 'vmovaps', 'vmovupd', 'vmovapd', 'vmovd', 'vmovq', 'vpbroadcastb', 'vpbroadcastw',
             'vpbroadcastd', 'vpbroadcastq', 'vpminub', 'vpmaxub', 'vpsubb', 'vpaddb', 'vpshufb', 'vpalignr',
             'vpsrldq', 'vpslldq', 'vpsrlw', 'vpsllw', 'vpsrld', 'vpslld', 'vpsrlq', 'vpsllq', 'vpunpcklbw',
             'vpunpckhbw', 'vpermq', 'vpshufd'}
PRED = {0: 'eq', 1: 'lt', 2: 'le', 3: 'false', 4: 'neq', 5: 'nlt', 6: 'nle', 7: 'true'}
SWAPPED = {0: 0, 1: 6, 2: 5, 3: 3, 4: 4, 5: 2, 6: 1, 7: 7}
FALSE_ON_ZERO = {1, 3, 4, 6}  # pred(0, 0) == false

READS = ('Read', 'CondRead', 'ReadWrite', 'ReadCondWrite')
WRITES = ('Write', 'CondWrite', 'ReadWrite', 'ReadCondWrite')
GPR32 = {'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp'} | {'r%dd' % n for n in range(8, 16)}


def is_vec(full):
    return full.startswith('zmm')


def uses(i):
    if i.mnemonic == 'vzeroupper':  # iced lists xmm0-15 as read; nothing depends on their values
        return set()
    return {r['full'] for r in i.j['regs'] if r['access'] in READS}


def defs(i):
    return {r['full'] for r in i.j['regs'] if r['access'] in WRITES}


def kills(i):
    """Registers fully overwritten (no merge with the old value)."""
    out = set()
    for r in i.j['regs']:
        if r['access'] != 'Write':
            continue
        if r['name'] == r['full'] or r['name'] in GPR32 or (is_vec(r['full']) and i.j['encoding'] in ('VEX', 'EVEX')):
            out.add(r['full'])
    return out


def writes_memory(i):
    return any(m['access'] in WRITES for m in i.j['mems'])


@dataclass
class Block:
    start: int          # index into the function's instruction list
    end: int            # exclusive
    succ: list = field(default_factory=list)
    live_in: set = field(default_factory=set)
    live_out: set = field(default_factory=set)


@dataclass
class Function:
    name: str
    ins: list           # Ins objects, in order
    lo: int = 0         # original offset range of the body
    hi: int = 0
    blocks: list = field(default_factory=list)
    live_after: list = field(default_factory=list)  # per instruction, registers live after it

    def index_of_offset(self):
        return {i.offset: k for k, i in enumerate(self.ins)}

    def internal(self, i):
        return i.target is not None and self.lo <= i.target < self.hi and i.reloc is None


def split_functions(obj, ins, section='.text'):
    out = []
    for f in obj.functions(section):
        lo, hi = f.value, f.value + f.size
        body = [i for i in ins if lo <= i.offset < hi]
        assert body and body[0].offset == lo, f.name
        out.append(Function(f.name, body, lo, hi))
    return out


def analyze(fn):
    build_cfg(fn)
    liveness(fn)


def build_cfg(fn):
    offsets = fn.index_of_offset()
    lo, hi = fn.lo, fn.hi
    leaders = {0}
    for k, i in enumerate(fn.ins):
        if i.flow in ('UnconditionalBranch', 'ConditionalBranch', 'Return', 'IndirectBranch'):
            leaders.add(k + 1)
        if i.target is not None and lo <= i.target < hi and i.reloc is None:
            leaders.add(offsets[i.target])
    leaders = sorted(l for l in leaders if l < len(fn.ins))
    fn.blocks = [Block(a, b) for a, b in zip(leaders, leaders[1:] + [len(fn.ins)])]
    block_at = {b.start: b for b in fn.blocks}
    for b in fn.blocks:
        last = fn.ins[b.end - 1]
        internal = last.target is not None and lo <= last.target < hi and last.reloc is None
        if last.flow in ('UnconditionalBranch',):
            if internal:
                b.succ.append(block_at[offsets[last.target]])
        elif last.flow == 'ConditionalBranch':
            if internal:
                b.succ.append(block_at[offsets[last.target]])
            if b.end < len(fn.ins):
                b.succ.append(block_at[b.end])
        elif last.flow in ('Return', 'IndirectBranch'):
            pass
        elif b.end < len(fn.ins):
            b.succ.append(block_at[b.end])


ARG_GPRS = {'rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9'}


def exit_live(i):
    """What an instruction leaving the function keeps alive: integer argument
    registers at tail jumps and indirect jumps (callee arguments), rax at ret.
    Nothing vector (library convention)."""
    if i.flow == 'Return':
        return {'rax'}
    if i.flow in ('UnconditionalBranch', 'IndirectBranch') and (i.reloc is not None or i.flow == 'IndirectBranch'):
        return set(ARG_GPRS)
    return set()


def liveness(fn):
    """Backward liveness of full registers. Calls and exits keep nothing vector
    alive (library convention); at calls and tail jumps the integer argument
    registers are live."""
    changed = True
    while changed:
        changed = False
        for b in reversed(fn.blocks):
            out = set()
            for s in b.succ:
                out |= s.live_in
            live = set(out)
            for k in range(b.end - 1, b.start - 1, -1):
                i = fn.ins[k]
                live |= exit_live(i)
                if i.flow in ('Call', 'IndirectCall'):
                    live = {r for r in live if not is_vec(r)} | ARG_GPRS
                live = (live - kills(i)) | uses(i)
            if live != b.live_in or out != b.live_out:
                b.live_in, b.live_out = live, out
                changed = True
    fn.live_after = [None] * len(fn.ins)
    for b in fn.blocks:
        live = set(b.live_out)
        for k in range(b.end - 1, b.start - 1, -1):
            i = fn.ins[k]
            live |= exit_live(i)
            fn.live_after[k] = set(live)
            if i.flow in ('Call', 'IndirectCall'):
                live = {r for r in live if not is_vec(r)} | ARG_GPRS
            live = (live - kills(i)) | uses(i)


def delete(i):
    i.j['gas'] = ''
    i.j['deleted'] = True
    i.j['regs'] = []
    i.j['mems'] = []
    i.j['flow'] = 'Next'
    i.j.pop('branch_target', None)
    i.reloc = None


# ---------------------------------------------------------------- fold

def masked_load(i):
    """vmovdqu8 (mem),%xN{%k}{z}: (memory operand, full register, mask, width letter) or None."""
    ops = i.j['ops']
    if i.mnemonic != 'vmovdqu8' or i.reloc is not None or len(ops) != 2 or ops[0]['kind'] != 'Register' or ops[1]['kind'] != 'Memory' \
            or i.j['op_mask'] == 'none' or not i.j['zeroing'] or not i.gas:
        return None
    return ops[1], full_vec(ops[0]['reg']), i.j['op_mask'], ops[0]['reg'][0]


def fold_masked_compares(fn):
    done = []
    for b in fn.blocks:
        loads = {}  # vector reg -> (index, memory text, mask)
        for k in range(b.start, b.end):
            i = fn.ins[k]
            for d in defs(i):
                loads.pop(d, None)
            for reg in [r for r, (_, _, mask, _, _) in loads.items() if mask in defs(i)]:
                loads.pop(reg)
            if writes_memory(i):
                loads.clear()
            ml = masked_load(i)
            if ml:
                mem, reg, mask, width = ml
                loads[reg] = (k, mem, mask, width, {r['full'] for r in i.j['regs'] if r['full'] != reg and r['full'] != mask})
                continue
            if i.mnemonic in ('vpcmpb', 'vpcmpub') and i.j['op_mask'] == 'none' and i.reloc is None and i.gas:
                ops = i.j['ops']
                if len(ops) != 4 or ops[1]['kind'] != 'Register' or ops[2]['kind'] != 'Register':
                    continue
                kreg, first, rm, pred = ops[0]['reg'], full_vec(ops[1]['reg']), full_vec(ops[2]['reg']), ops[3]['imm']
                if first == rm or first not in loads or rm not in loads:
                    continue
                if loads[first][2] != loads[rm][2]:
                    continue
                # Fold whichever is dead after the compare into the memory operand (the rm slot).
                if rm not in fn.live_after[k]:
                    victim, keep = rm, first
                elif first not in fn.live_after[k]:
                    victim, keep, pred = first, rm, SWAPPED[pred]
                else:
                    continue
                if pred not in FALSE_ON_ZERO:
                    continue
                lk, mem, mask, width, addr_regs = loads[victim]
                # The victim must be untouched between its load and the compare, and its address stable.
                if any(victim in uses(fn.ins[m]) or victim in defs(fn.ins[m]) or addr_regs & defs(fn.ins[m]) for m in range(lk + 1, k)):
                    continue
                keep_op = ops[1] if keep == first else ops[2]
                i.j['ops'] = [ops[0], keep_op, dict(mem), {'kind': 'Immediate', 'imm': pred, 'size': 8, 'signed': False, 'implicit': False}]
                i.j['ops'][2]['mem'] = dict(mem['mem'])
                i.j['op_mask'] = mask
                i.j['zeroing'] = False
                i.j['regs'] = [r for r in i.j['regs'] if r['full'] != victim] + [{'name': mask, 'full': mask, 'access': 'Read'}]
                refresh(i)
                done.append('%s: folded %s into %s' % (fn.name, fn.ins[lk], i.gas))
                delete(fn.ins[lk])
                loads.pop(victim)
                done += drop_subset_ands(fn, b, k + 1, kreg, mask)
    return done


def kregs(i, prefix, count):
    """The mask registers of a k-instruction with that mnemonic prefix and operand count, or None."""
    ops = i.j['ops']
    if not i.gas or not i.mnemonic.startswith(prefix) or len(i.mnemonic) != len(prefix) + 1 or len(ops) != count \
            or any(o['kind'] != 'Register' or o['reg'][0] != 'k' for o in ops):
        return None
    return [o['reg'] for o in ops]


def drop_subset_ands(fn, b, start, kreg, mask):
    """kreg is a subset of mask from index start on: kand mask,kreg,kreg is a
    no-op there, and in successors reachable only through this block."""
    done = []
    preds = predecessors(fn)
    for k in range(start, b.end):
        i = fn.ins[k]
        m = kregs(i, 'kand', 3)
        if m and {m[1], m[2]} == {kreg, mask} and m[0] == kreg:   # AT&T order in the text; ops are dst, src1, src2
            done.append('%s: dropped %s' % (fn.name, i))
            delete(i)
            continue
        mt = kregs(i, 'ktest', 2)
        if mt and set(mt) == {kreg, mask}:
            i.j['mnemonic'] = 'kortest' + i.mnemonic[-1]   # ZF is the same; CF differs, only je/jne follow
            i.j['ops'] = [register(kreg, 64), register(kreg, 64)]
            i.j['regs'] = [{'name': kreg, 'full': kreg, 'access': 'Read'}]
            refresh(i)
            done.append('%s: ktest -> %s' % (fn.name, i.gas))
            continue
        if kreg in defs(i) or mask in defs(i):
            return done
    for s in b.succ:
        if preds[id(s)] == [b]:
            done += drop_subset_ands(fn, s, s.start, kreg, mask)
    return done


def predecessors(fn):
    preds = {id(b): [] for b in fn.blocks}
    for b in fn.blocks:
        for s in b.succ:
            preds[id(s)].append(b)
    return preds


# ---------------------------------------------------------------- tail

TAIL_MAX = 4
clone_counter = [0]


def clone(i, before=False):
    """A new instruction made from i. before=True: it is inserted in front of an original
    instruction and belongs to it, so a label on that instruction moves to the clone
    (a copy restored before a use, a zero register at the entry). Otherwise the clone
    belongs to what precedes it (a duplicated tail after a jump)."""
    j = copy.deepcopy(i.j)
    j['home'] = i.home
    j['before'] = before
    clone_counter[0] += 1
    j['offset'] = (1 << 40) + clone_counter[0] * 16  # unique, never a label
    return Ins(j, i.reloc)


def tail_chain(fn, tb):
    """Blocks from tb through fall-throughs to a ret or tail jump, if at most TAIL_MAX live instructions."""
    chain, tail, b = [], [], tb
    while True:
        chain.append(b)
        tail += [fn.ins[k] for k in range(b.start, b.end) if fn.ins[k].gas]
        if len(tail) > getattr(fn, 'tail_limit', TAIL_MAX):
            return None, None
        last = fn.ins[b.end - 1]
        if last.flow == 'Return' or (last.flow == 'UnconditionalBranch' and last.reloc is not None):
            return chain, tail
        if last.flow == 'Next' and len(b.succ) == 1:
            b = b.succ[0]
            continue
        return None, None


def duplicate_tails(fn):
    done = []
    preds = predecessors(fn)
    block_at = {b.start: b for b in fn.blocks}
    offsets = fn.index_of_offset()
    new_ins = []
    for b in fn.blocks:
        last = fn.ins[b.end - 1]
        if not (last.flow == 'UnconditionalBranch' and fn.internal(last)):
            continue
        tb = block_at[offsets[last.target]]
        if tb is b or tb.start == 0:
            continue
        chain, tail = tail_chain(fn, tb)
        if not chain or b in chain:
            continue
        done.append('%s: copied %d-instruction tail at %s over %s' % (fn.name, len(tail), label_of(tail[0]), last))
        delete(last)
        new_ins.append((b.end, [clone(i) for i in tail]))
        preds[id(tb)].remove(b)
    # Drop blocks of the chains nothing reaches any more (the ones only that jmp led to).
    for b in fn.blocks:
        if not preds[id(b)] and b.start != 0 and any(fn.ins[k].gas for k in range(b.start, b.end)):
            for k in range(b.start, b.end):
                delete(fn.ins[k])
            for s in b.succ:
                # A pending tail copy may already have removed this edge.
                if b in preds[id(s)]:
                    preds[id(s)].remove(b)
            done.append('%s: dropped unreachable block at %s' % (fn.name, label_of(fn.ins[b.start])))
    for k, clones in sorted(new_ins, reverse=True):
        fn.ins[k:k] = clones
    return done


def label_of(i):
    return '.L_%x' % i.home


# ---------------------------------------------------------------- rename

def renameable(i):
    if not any(is_vec(r['full']) for r in i.j['regs']):
        return True
    if i.mnemonic == 'vzeroupper' or i.j['encoding'] == 'EVEX':
        return True
    return i.j['encoding'] == 'VEX' and (i.mnemonic in EVEX_NAME or i.mnemonic in EVEX_SAME)


def vector_webs(fn):
    """Def-use webs of vector registers: union-find over (instruction, register)
    accesses connected by reaching definitions. Returns {(k, full): web id}."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    def accesses(k):
        i = fn.ins[k]
        if i.mnemonic == 'vzeroupper':
            return set(), set()
        d = {r['full'] for r in i.j['regs'] if is_vec(r['full']) and r['access'] in WRITES}
        u = {r['full'] for r in i.j['regs'] if is_vec(r['full']) and r['access'] in READS}
        return d, u

    # reaching definitions per block: {reg: set of def indices}
    gen, kill = [], []
    for b in fn.blocks:
        g, kl = {}, set()
        for k in range(b.start, b.end):
            d, u = accesses(k)
            for reg in d:
                g[reg] = {k}
                kl.add(reg)
        gen.append(g)
        kill.append(kl)
    rin = [dict() for _ in fn.blocks]
    rout = [dict() for _ in fn.blocks]
    index = {id(b): n for n, b in enumerate(fn.blocks)}
    changed = True
    while changed:
        changed = False
        for n, b in enumerate(fn.blocks):
            inn = {}
            for p in fn.blocks:
                if b in p.succ:
                    for reg, defs in rout[index[id(p)]].items():
                        inn.setdefault(reg, set()).update(defs)
            out = {reg: set(defs) for reg, defs in inn.items() if reg not in kill[n]}
            for reg, defs in gen[n].items():
                out[reg] = set(defs)
            if inn != rin[n] or out != rout[n]:
                rin[n], rout[n] = inn, out
                changed = True
    for n, b in enumerate(fn.blocks):
        cur = {reg: set(defs) for reg, defs in rin[n].items()}
        for k in range(b.start, b.end):
            d, u = accesses(k)
            for reg in u:
                for dk in cur.get(reg, ()):
                    union((k, reg), (dk, reg))
            for reg in d:
                if reg in u:  # read-modify-write: the new value continues the web
                    pass
                cur[reg] = {k}
    webs = {}
    for k in range(len(fn.ins)):
        d, u = accesses(k)
        for reg in d | u:
            webs[(k, reg)] = find((k, reg))
    return webs


def rename_vectors(fn):
    if not any(is_vec(r['full']) for i in fn.ins for r in i.j['regs']):
        return []
    entry_live = {r for r in fn.blocks[0].live_in if is_vec(r)}
    if entry_live:
        return ['%s: not renamed, live at entry: %s' % (fn.name, sorted(entry_live))]
    if any(int(m.group(2)) >= 16 for i in fn.ins for m in VEC.finditer(i.gas)):
        return ['%s: not renamed, already uses ymm16-31' % fn.name]
    webs = vector_webs(fn)
    # webs with a 256-bit access are the ones that dirty the upper state
    wide = set()
    for (k, reg), w in webs.items():
        if any(r['full'] == reg and r['name'].startswith(('ymm', 'zmm')) for r in fn.ins[k].j['regs']):
            wide.add(w)
    if not wide:
        return []
    for (k, reg), w in webs.items():
        if w in wide and not renameable(fn.ins[k]):
            return ['%s: not renamed, no EVEX form for %s' % (fn.name, fn.ins[k])]
    removed = 0
    for k, i in enumerate(fn.ins):
        if not i.gas:
            continue
        if i.mnemonic == 'vzeroupper':
            delete(i)
            removed += 1
            continue
        regs = {reg for (kk, reg), w in webs.items() if kk == k and w in wide}
        if not regs:
            continue
        for o in i.j['ops']:
            if o['kind'] == 'Register' and o['reg'][:3] in ('xmm', 'ymm', 'zmm') and full_vec(o['reg']) in regs:
                o['reg'] = o['reg'][:3] + str(int(o['reg'][3:]) + 16)
        if i.mnemonic in EVEX_NAME:
            i.j['mnemonic'] = EVEX_NAME[i.mnemonic]
        i.j['encoding'] = 'EVEX'
        for r in i.j['regs']:
            if r['full'] in regs:
                n = int(re.search(r'\d+', r['name']).group()) + 16
                r['name'] = re.sub(r'\d+', str(n), r['name'])
                r['full'] = 'zmm%d' % n
        refresh(i)
    return ['%s: renamed %d webs to ymm16-31, %d vzeroupper removed' % (fn.name, len(wide), removed)]


# ---------------------------------------------------------------- hot

INVERT = {'e': 'ne', 'ne': 'e', 'b': 'ae', 'ae': 'b', 'be': 'a', 'a': 'be', 'l': 'ge', 'ge': 'l', 'le': 'g', 'g': 'le',
          's': 'ns', 'ns': 's', 'o': 'no', 'no': 'o', 'p': 'np', 'np': 'p'}


VECTOR_RESULTS = {'vpmovmskb', 'pmovmskb', 'kmovw', 'kmovd', 'kmovq', 'kortestw', 'kortestd', 'kortestq', 'ktestw', 'ktestd', 'ktestq'}


def hot_tails(fn):
    """A conditional branch to a short return tail becomes fall-through code; the
    old fall-through is reached by the inverted branch. Only where the branch
    decides on a vector compare's result (a movemask, a mask moved to a
    register, a kortest): the string that ends in the block is the common
    case, as the first blocks are sized. A branch on an argument (n == 0 at
    an entry) keeps the compiler's layout: its return is rare, and making it
    fall through would cost every other call a taken branch."""
    done = []
    preds = predecessors(fn)
    block_at = {b.start: b for b in fn.blocks}
    offsets = fn.index_of_offset()
    new_ins = []
    for b in fn.blocks:
        last = fn.ins[b.end - 1]
        if not (last.flow == 'ConditionalBranch' and fn.internal(last)) or b.end >= len(fn.ins):
            continue
        tb = block_at[offsets[last.target]]
        fb = block_at.get(b.end)
        if tb is b or tb.start == 0 or fb is None:
            continue
        chain, tail = tail_chain(fn, tb)
        if not chain or b in chain or tail[-1].flow != 'Return':
            continue
        fchain, ftail = tail_chain(fn, fb)
        if fchain and ftail[-1].flow == 'Return':
            continue  # both ways return quickly: leave the layout alone
        if not any(fn.ins[k].mnemonic in VECTOR_RESULTS for k in range(b.start, b.end - 1)):
            continue  # not a vector compare's verdict: the compiler's layout stands
        cc = last.mnemonic[1:]
        if cc not in INVERT:
            continue
        fall = fn.ins[fb.start]
        last.j['mnemonic'] = 'j' + INVERT[cc]
        last.j['cc'] = INVERT[cc]
        last.j['branch_target'] = fall.offset
        last.j['ops'][0]['branch'] = fall.offset
        refresh(last)
        new_ins.append((b.end, [clone(i) for i in tail]))
        preds[id(tb)].remove(b)
        done.append('%s: %s now falls through into the %d-instruction return tail at %s' % (fn.name, last.j['mnemonic'], len(tail), label_of(tail[0])))
    for b in fn.blocks:
        if not preds[id(b)] and b.start != 0 and any(fn.ins[k].gas for k in range(b.start, b.end)):
            for k in range(b.start, b.end):
                delete(fn.ins[k])
            for s in b.succ:
                preds[id(s)].remove(b)
            done.append('%s: dropped unreachable block at %s' % (fn.name, label_of(fn.ins[b.start])))
    for k, clones in sorted(new_ins, reverse=True):
        fn.ins[k:k] = clones
    return done


# ---------------------------------------------------------------- zcmp

LOADS = {'vmovdqu', 'vmovdqa', 'vmovdqu8', 'vmovdqu16', 'vmovdqu32', 'vmovdqu64', 'vmovdqa32', 'vmovdqa64', 'vmovups', 'vmovaps'}


def zero_compares(fn):
    """vmovdqu (m),X; vptestnmb X,X,K with X dead afterwards -> vpcmpeqb (m),Z,K with Z a
    register zeroed once at the entry (a zero idiom, free): one fused uop instead of two.
    Z is a ymm16-31 register the function never touches. Off by default: measured
    neutral on Zen 5 and 20% slower on Skylake-SP (strlen at 5 bytes, 5 runs), so the
    memory-operand compare does not fuse the way the separate load does there."""
    done = []
    used = {int(o['reg'][3:]) for i in fn.ins for o in i.j['ops'] if i.gas and o['kind'] == 'Register' and o['reg'][:3] in ('xmm', 'ymm', 'zmm')}
    z = next((n for n in range(16, 32) if n not in used), None)
    if z is None:
        return done
    sites = []
    for b in fn.blocks:
        live = [k for k in range(b.start, b.end) if fn.ins[k].gas]
        for k1, k2 in zip(live, live[1:]):
            load, cmp = fn.ins[k1], fn.ins[k2]
            lo, co = load.j['ops'], cmp.j['ops']
            if not (load.mnemonic in LOADS and len(lo) == 2 and lo[0]['kind'] == 'Register' and lo[1]['kind'] == 'Memory' and load.reloc is None
                    and load.j['op_mask'] == 'none' and cmp.mnemonic == 'vptestnmb' and cmp.j['op_mask'] == 'none' and cmp.reloc is None
                    and len(co) == 3 and co[1]['kind'] == 'Register' and co[2]['kind'] == 'Register' and co[1]['reg'] == co[2]['reg'] == lo[0]['reg']):
                continue
            if full_vec(lo[0]['reg']) in fn.live_after[k2]:
                continue
            sites.append((load, cmp))
    if not sites:
        return done
    width = {'xmm': 128, 'ymm': 256}
    for load, cmp in sites:
        x = load.j['ops'][0]['reg']
        zreg = x[:3] + str(z)
        mem = dict(load.j['ops'][1])
        mem['mem'] = dict(mem['mem'])
        cmp.j['mnemonic'] = 'vpcmpb'
        cmp.j['ops'] = [cmp.j['ops'][0], register(zreg, width[x[:3]]), mem, {'kind': 'Immediate', 'imm': 0, 'size': 8, 'signed': False, 'implicit': False}]
        cmp.j['regs'] = [r for r in cmp.j['regs'] if r['full'] != full_vec(x)] + [{'name': zreg, 'full': 'zmm%d' % z, 'access': 'Read'}] \
            + [{'name': r['name'], 'full': r['full'], 'access': 'Read'} for r in load.j['regs'] if r['access'] == 'Read' and not is_vec(r['full'])]
        cmp.j['mems'] = list(load.j['mems'])
        cmp.j['encoding'] = 'EVEX'
        refresh(cmp)
        done.append('%s: %s + vptestnmb -> %s' % (fn.name, load.gas, cmp.gas))
        delete(load)
    zero = clone(fn.ins[0], before=True)
    xz = 'xmm%d' % z
    zero.j.update({'mnemonic': 'vpxord', 'cc': None, 'ops': [register(xz, 128), register(xz, 128), register(xz, 128)], 'op_mask': 'none', 'zeroing': False,
                   'bcast': False, 'rep': None, 'lock': False, 'notrack': False, 'flow': 'Next', 'encoding': 'EVEX', 'mems': [],
                   'regs': [{'name': xz, 'full': 'zmm%d' % z, 'access': 'Write'}]})
    zero.j.pop('branch_target', None)
    zero.reloc = None
    refresh(zero)
    # at the entry, after any endbr64
    at = 1 if fn.ins[0].mnemonic == 'endbr64' else 0
    fn.ins[at:at] = [zero]
    done.append('%s: %s at the entry' % (fn.name, zero.gas))
    return done


# ---------------------------------------------------------------- copy

def gpr_alias(name, full):
    """The register of the same width as name within the full register full."""
    for table in (x86.R64, x86.R32, x86.R16, x86.R8):
        if name in table:
            return table[x86.R64.index(full)]
    return None


def sink_copies(fn):
    """mov A,B at the entry block: B's reads on the paths where neither changes
    become reads of A, the copy itself moves to the points that still need B
    (A overwritten while B is live, B read implicitly, B live into a block the
    copy does not dominate). The hot path usually needs none."""
    done = []
    entry = fn.blocks[0]
    block_of = {}
    for b in fn.blocks:
        for k in range(b.start, b.end):
            block_of[k] = b
    preds = predecessors(fn)
    for m in range(entry.start, entry.end):
        mov = fn.ins[m]
        o = mov.j['ops']
        if not (mov.gas and mov.mnemonic == 'mov' and mov.reloc is None and len(o) == 2 and o[0]['kind'] == 'Register' and o[1]['kind'] == 'Register'
                and o[0]['size'] == 64 and o[1]['size'] == 64 and o[0]['reg'] in x86.R64 and o[1]['reg'] in x86.R64 and o[0]['reg'] != o[1]['reg']):
            continue
        B, A = o[0]['reg'], o[1]['reg']
        if A in ('rsp', 'rbp') or B in ('rsp', 'rbp'):
            continue   # the frame: rsp moves at every push and pop
        # the region: instructions after the copy on every path until A or B changes, in blocks the copy dominates
        region = set()      # instruction indices
        ends = {}           # index -> 'A' (A written; B may need restoring before) | 'B' (B written)
        implicit = set()    # indices that read B without naming it: restore before
        bad = False
        work = [(entry, m + 1)]
        seen_blocks = set()
        while work and not bad:
            b, start = work.pop()
            k = start
            stop = None
            while k < b.end:
                i = fn.ins[k]
                if not i.gas:
                    k += 1
                    continue
                reads_b = [r for r in i.j['regs'] if r['full'] == B and r['access'] in READS]
                # renamable only when every access to B is an explicit pure read (register or address)
                pure = all(r['access'] == 'Read' for r in i.j['regs'] if r['full'] == B) \
                    and all(not op.get('implicit') for op in i.j['ops'] if op['kind'] == 'Register' and x86.gpr_full(op['reg']) == B) \
                    and not any(r['name'] == B for r in i.j['regs'] if r['full'] == B and r['name'] not in [op['reg'] for op in i.j['ops'] if op['kind'] == 'Register']
                                and not any(op['kind'] == 'Memory' and B in (op['mem']['base'], op['mem']['index']) for op in i.j['ops']))
                if reads_b and not pure:
                    implicit.add(k)
                if any(op['kind'] == 'Register' and op['reg'] in HIGH8 and x86.gpr_full(op['reg']) in (A, B) for op in i.j['ops']):
                    bad = True
                    break
                writes_a = any(r['full'] == A and r['access'] in WRITES for r in i.j['regs'])
                writes_b = any(r['full'] == B and r['access'] in WRITES for r in i.j['regs'])
                region.add(k)
                if i.flow in ('Call', 'IndirectCall'):
                    bad = True   # calls clobber both
                    break
                if writes_a or writes_b:
                    ends[k] = 'B' if writes_b else 'A'
                    stop = k
                    break
                k += 1
            if bad or stop is not None:
                continue
            if b.end < len(fn.ins) or True:
                for s in b.succ:
                    if id(s) in seen_blocks:
                        continue
                    if any(p is not entry and (id(p) not in seen_blocks and p is not b) for p in preds[id(s)]):
                        # a way in that does not come through the region: B may not equal A there
                        continue
                    seen_blocks.add(id(s))
                    work.append((s, s.start))
        if bad or not region:
            continue
        region_blocks = {id(block_of[k]) for k in region}
        # where must B be restored? {index: belongs to the instruction there (its label
        # moves to the copy) or to the edge before it (the label stays: a loop head
        # entered from the region's fall-through must not run the copy on its back edges)}
        inserts = {}
        for k in region:
            i = fn.ins[k]
            if k in implicit:
                inserts[k] = True
            if ends.get(k) == 'A' and B in fn.live_after[k]:
                inserts[k] = True
            if B in exit_live(i):
                inserts[k] = True
        for b in fn.blocks:
            if id(b) not in region_blocks:
                continue
            last = b.end - 1
            if last in region and last not in ends:
                for s in b.succ:
                    if id(s) not in region_blocks and B in s.live_in:
                        if fn.ins[last].flow != 'Next':
                            inserts[last] = True
                        else:
                            inserts.setdefault(b.end, False)
        if len(inserts) > 3:
            continue
        # rewrite the reads
        for k in sorted(region):
            i = fn.ins[k]
            if k in implicit:
                continue   # B is restored right before it
            if any(r['full'] == B and r['access'] != 'Read' for r in i.j['regs']):
                continue   # writes B: nothing to rename
            changed = False
            for op in i.j['ops']:
                if op['kind'] == 'Register' and x86.gpr_full(op['reg']) == B and not op.get('implicit'):
                    op['reg'] = gpr_alias(op['reg'], A)
                    changed = True
                elif op['kind'] == 'Memory':
                    if op['mem']['base'] == B:
                        op['mem']['base'] = A
                        changed = True
                    if op['mem']['index'] == B:
                        op['mem']['index'] = A
                        changed = True
            if changed:
                for r in i.j['regs']:
                    if r['full'] == B and r['access'] == 'Read':
                        r['name'] = gpr_alias(r['name'], A)
                        r['full'] = A
                refresh(i)
        # self-moves left by the renaming (mov %rsi,%rdi with rsi now rdi) go away
        for k in sorted(region):
            i = fn.ins[k]
            if i.gas and i.mnemonic == 'mov' and len(i.j['ops']) == 2 and all(o['kind'] == 'Register' for o in i.j['ops']) and i.j['ops'][0]['reg'] == i.j['ops'][1]['reg']:
                delete(i)
        new_ins = []
        for k in sorted(inserts, reverse=True):
            copy_ = clone(mov, before=inserts[k])
            copy_.j.pop('branch_target', None)
            refresh(copy_)
            new_ins.append((k, [copy_]))
        for k, clones in new_ins:
            fn.ins[k:k] = clones
        done.append('%s: %s moved from the entry to %d place(s)' % (fn.name, mov.gas, len(inserts)))
        delete(mov)
        return done   # one copy per function per pass; the next pass run sees a fresh CFG
    return done


# ---------------------------------------------------------------- lea

def fold_lea_add(fn):
    """leaq (a,b),c; addq $imm,c -> leaq imm(a,b),c when the flags of the add are dead."""
    done = []
    for b in fn.blocks:
        live = [k for k in range(b.start, b.end) if fn.ins[k].gas]
        for k1, k2 in zip(live, live[1:]):
            lea, add = fn.ins[k1], fn.ins[k2]
            lo, ao = lea.j['ops'], add.j['ops']
            if not (lea.mnemonic == 'lea' and add.mnemonic == 'add' and lo[0]['size'] == 64 and lo[1]['kind'] == 'Memory' and not lo[1]['mem']['rip_rel']
                    and ao[0]['kind'] == 'Register' and ao[0]['reg'] == lo[0]['reg'] and ao[1]['kind'] == 'Immediate' and lea.reloc is None and add.reloc is None):
                continue
            disp = lo[1]['mem']['disp'] + ao[1]['imm']
            if not -2 ** 31 <= disp < 2 ** 31:
                continue
            # flags: the add's flags must not be read before being written (next instruction only)
            nxt = next((fn.ins[k] for k in live if k > k2), None)
            if nxt is not None and (nxt.flow == 'ConditionalBranch' or nxt.mnemonic.startswith(('set', 'cmov', 'adc', 'sbb'))):
                continue
            if k2 == live[-1]:
                continue
            before = '%s + %s' % (lea.gas, add.gas)
            lo[1]['mem']['disp'] = disp
            refresh(lea)
            done.append('%s: %s -> %s' % (fn.name, before, lea.gas))
            delete(add)
    return done


# ---------------------------------------------------------------- thread

def thread_jumps(fn):
    """jcc/jmp L where L is just 'jmp X': go to X directly (X may be a tail call)."""
    done = []
    block_at = {b.start: b for b in fn.blocks}
    offsets = fn.index_of_offset()
    for b in fn.blocks:
        last = fn.ins[b.end - 1]
        if last.flow not in ('ConditionalBranch', 'UnconditionalBranch') or not fn.internal(last):
            continue
        tb = block_at[offsets[last.target]]
        body = [fn.ins[k] for k in range(tb.start, tb.end) if fn.ins[k].gas]
        if len(body) != 1 or body[0].flow != 'UnconditionalBranch' or body[0] is last:
            continue
        jmp = body[0]
        if jmp.reloc is not None:
            last.reloc = jmp.reloc
            last.j['home'] = jmp.home  # the relocation field is measured from the jmp's bytes
            last.j['len'] = jmp.j['len']
            last.j['ops'] = [dict(o) for o in jmp.j['ops']]
        last.j['branch_target'] = jmp.target
        last.j['ops'][0]['branch'] = jmp.target
        refresh(last)
        done.append('%s: %s now goes straight to %s' % (fn.name, last.mnemonic, label_of(jmp) if jmp.reloc is None else jmp.reloc.symbol))
    analyze(fn)
    preds = predecessors(fn)
    for b in fn.blocks:
        if not preds[id(b)] and b.start != 0 and any(fn.ins[k].gas for k in range(b.start, b.end)):
            for k in range(b.start, b.end):
                delete(fn.ins[k])
            done.append('%s: dropped unreachable block at %s' % (fn.name, label_of(fn.ins[b.start])))
    return done


# ---------------------------------------------------------------- ktest

FREE_GPRS = ['eax', 'r10d', 'r11d']  # never arguments: safe even where a tail call follows


def kortest_of(i):
    """kortestw/d %k,%k -> (width letter, k) or None."""
    m = kregs(i, 'kortest', 2)
    return (i.mnemonic[-1], m[0]) if m and m[0] == m[1] and i.mnemonic[-1] in 'wd' else None


def kmov_of(i):
    """kmovw/d %k,%r32 -> (width letter, k, r32) or None."""
    ops = i.j['ops']
    if i.gas and i.mnemonic in ('kmovw', 'kmovd') and len(ops) == 2 and ops[0]['kind'] == 'Register' and ops[1]['kind'] == 'Register' \
            and ops[0]['size'] == 32 and ops[1]['reg'][0] == 'k':
        return i.mnemonic[-1], ops[1]['reg'], ops[0]['reg']
    return None


def kortest_to_test(fn):
    """Only where it pays: a block L starting with kmov k,r whose every way in ends
    with kortest k,k; je/jne (or already kmov k,r; test r,r; jcc). Those kortests
    become kmov k,r; test r,r (test+jcc fuse, kortest+jcc do not) and L's kmov
    goes away."""
    done = []
    preds = predecessors(fn)
    new_ins = []
    # (kor hoisting - kortest kA,kB; jcc; kor kA,kB,kC -> kor; kortest kC,kC; jcc - was tried and
    # costs +40% on strcmp 5 on Skylake-SP: the branch then waits for the kor.)
    # kmov sinks
    new_ins = []
    for L in fn.blocks:
        first = next((fn.ins[k] for k in range(L.start, L.end) if fn.ins[k].gas), None)
        tm = kmov_of(first) if first is not None else None
        if not tm or not preds[id(L)]:
            continue
        width, kreg, reg = tm
        plan = []
        for p in preds[id(L)]:
            seq = [fn.ins[k] for k in range(p.start, p.end) if fn.ins[k].gas]
            if len(seq) >= 3 and kmov_of(seq[-3]) and kmov_of(seq[-3])[1:] == (kreg, reg) \
                    and seq[-2].mnemonic == 'test' and [o['reg'] for o in seq[-2].j['ops']] == [reg, reg] and seq[-1].flow == 'ConditionalBranch':
                continue  # already in the wanted shape
            if len(seq) >= 2 and seq[-1].flow == 'ConditionalBranch' and seq[-1].mnemonic in ('je', 'jne'):
                m = kortest_of(seq[-2])
                if m and m[1] == kreg and full_gpr(reg) not in fn.live_after[fn.ins.index(seq[-1])]:
                    plan.append((seq[-2], seq[-1], m[0]))
                    continue
            plan = None
            break
        if plan is None:
            continue
        for kt, jcc, kw in plan:
            kt.j['mnemonic'] = 'kmov' + kw
            kt.j['ops'] = [register(reg, 32), register(kreg, 64)]
            kt.j['encoding'] = 'VEX'
            kt.j['regs'] = [{'name': reg, 'full': full_gpr(reg), 'access': 'Write'}, {'name': kreg, 'full': kreg, 'access': 'Read'}]
            refresh(kt)
            test = clone(kt)
            test.j.update({'mnemonic': 'test', 'cc': None, 'ops': [register(reg, 32), register(reg, 32)],
                           'regs': [{'name': reg, 'full': full_gpr(reg), 'access': 'Read'}], 'encoding': 'Legacy'})
            refresh(test)
            new_ins.append((fn.ins.index(jcc), [test]))
            done.append('%s: kortest%s %s -> kmov %s; test, for %s' % (fn.name, kw, kreg, reg, label_of(first)))
        done.append('%s: dropped %s, every way in now does it' % (fn.name, first))
        delete(first)
    for k, clones in sorted(new_ins, reverse=True):
        fn.ins[k:k] = clones
    return done


# Instructions that leave the flags alone; anything else is taken to write them.
FLAG_NEUTRAL = re.compile(r'^(mov|lea|push|pop|nop|k[a-z]+|v[a-z0-9]+|cmov[a-z]+|set[a-z]+|j[a-z]+)$')
FLAG_READERS = re.compile(r'^(j(?!mp$)[a-z]+|set[a-z]+|cmov[a-z]+|adc|sbb)$')


def reads_flags_first(fn, b, start):
    """Whether the flags are read in block b from index start on before anything writes them."""
    for k in range(start, b.end):
        i = fn.ins[k]
        if not i.gas:
            continue
        if FLAG_READERS.match(i.mnemonic):
            return True
        if not FLAG_NEUTRAL.match(i.mnemonic):
            return False
    return any(reads_flags_first(fn, s, s.start) for s in b.succ)


def kor_before_branch(fn):
    """kortest kA,kB; jcc with a successor, entered only from here, that starts
    with kor kA,kB,kD; kmov kD,r: kor; kmov; test r,r; jcc. The kor and kmov
    move up; kD and r must be dead on the other edge, and nothing may read the
    flags of the kortest (test sets CF differently)."""
    done = []
    preds = predecessors(fn)
    inserts = []
    for b in fn.blocks:
        seq = [k for k in range(b.start, b.end) if fn.ins[k].gas]
        if len(seq) < 2:
            continue
        kt, jcc = fn.ins[seq[-2]], fn.ins[seq[-1]]
        m = kregs(kt, 'kortest', 2)
        if not m or jcc.flow != 'ConditionalBranch' or jcc.mnemonic not in ('je', 'jne'):
            continue
        for s in b.succ:
            if s is b or preds[id(s)] != [b]:
                continue
            body = [k for k in range(s.start, s.end) if fn.ins[k].gas]
            if len(body) < 2:
                continue
            kor, km = kregs(fn.ins[body[0]], 'kor', 3), kmov_of(fn.ins[body[1]])
            if not kor or sorted(kor[1:]) != sorted(m) or not km or km[1] != kor[0]:
                continue
            kd, reg = kor[0], km[2]
            others = [x for x in b.succ if x is not s]
            if any(kd in x.live_in or full_gpr(reg) in x.live_in for x in others):
                continue
            if reads_flags_first(fn, s, body[2] if len(body) > 2 else s.end) or any(reads_flags_first(fn, x, x.start) for x in others):
                continue
            inserts.append((seq[-2], [clone(fn.ins[body[0]], before=True), clone(fn.ins[body[1]], before=True)]))
            kt.j.update({'mnemonic': 'test', 'cc': None, 'ops': [register(reg, 32), register(reg, 32)],
                         'regs': [{'name': reg, 'full': full_gpr(reg), 'access': 'Read'}], 'encoding': 'Legacy'})
            refresh(kt)
            done.append('%s: %s and %s moved in front of the branch, kortest -> test %s' % (fn.name, fn.ins[body[0]].gas, fn.ins[body[1]].gas, reg))
            delete(fn.ins[body[0]])
            delete(fn.ins[body[1]])
            break
    for k, clones in sorted(inserts, reverse=True):
        fn.ins[k:k] = clones
    return done


# ---------------------------------------------------------------- kfirst

def kcompare(i):
    """A mask-producing compare: (destination k, [source register names]) or None."""
    ops = i.j['ops']
    if not i.gas or i.j['encoding'] != 'EVEX' or not (i.mnemonic.startswith('vpcmp') or i.mnemonic.startswith('vptest')) \
            or not ops or ops[0]['kind'] != 'Register' or ops[0]['reg'][0] != 'k' or i.reloc is not None:
        return None
    return ops[0]['reg'], [o['reg'] for o in ops[1:] if o['kind'] == 'Register']


def kreg_entries(name, access):
    return [{'name': name, 'full': name, 'access': access}]


def mask_op_without_memory_first(fn):
    done = []
    for b in fn.blocks:
        idx = [k for k in range(b.start, b.end) if fn.ins[k].gas]
        for p, q in zip(idx, idx[1:]):
            A, B = fn.ins[p], fn.ins[q]
            ca, cb = kcompare(A), kcompare(B)
            if not ca or not cb or p == b.start:   # a label on A would be left behind by the swap
                continue
            ka, kb = ca[0], cb[0]
            if A.j['op_mask'] != 'none' or B.j['op_mask'] != ka or ka in cb[1] or ka in fn.live_after[q]:
                continue
            if not any(o['kind'] == 'Memory' for o in A.j['ops']) or any(o['kind'] == 'Memory' for o in B.j['ops']):
                continue
            if kb in ca[1] or A.j['zeroing'] or B.j['zeroing']:
                continue
            # B computes its mask into ka, unmasked; A, masked by it, delivers the and into kb
            B.j['ops'][0]['reg'] = ka
            B.j['op_mask'] = 'none'
            B.j['regs'] = [r for r in B.j['regs'] if r['name'] not in (ka, kb)] + kreg_entries(ka, 'Write')
            A.j['ops'][0]['reg'] = kb
            A.j['op_mask'] = ka
            A.j['regs'] = [r for r in A.j['regs'] if r['name'] not in (ka, kb)] + kreg_entries(kb, 'Write') + kreg_entries(ka, 'Read')
            refresh(A)
            refresh(B)
            fn.ins[p], fn.ins[q] = B, A
            done.append('%s: %s before %s, the load-bearing compare under its mask' % (fn.name, B.gas, A.gas))
    if done:
        analyze(fn)
    return done


# ---------------------------------------------------------------- tramp

def trampolines_of(obj, code):
    """{function name: (symbol, offset)} for every global function of the object whose
    body is one jmp *symbol+offset(%rip): the str_* entry points of dispatch.c."""
    out = {}
    exported = {s.name for s in obj.symbols if s.type == 'STT_FUNC' and s.bind == 'STB_GLOBAL'}
    for section, ins in code.items():
        for f in obj.functions(section):
            body = [i for i in ins if f.value <= i.offset < f.value + f.size and i.gas]
            # a frame-pointer build wraps the jump in push %rbp; mov %rsp,%rbp; ...; pop %rbp
            if len(body) == 4 and [i.gas for i in body[:3]] == ['pushq %rbp', 'movq %rsp,%rbp', 'popq %rbp']:
                body = body[3:]
            if f.name in exported and len(body) == 1 and body[0].mnemonic == 'jmp' and body[0].flow == 'IndirectBranch' \
                    and body[0].reloc is not None and not body[0].reloc.symbol_is_section and body[0].reloc.type == 'R_X86_64_PC32':
                i, r = body[0], body[0].reloc
                out[f.name] = (r.symbol, r.addend + (i.home_end - r.offset))
    return out


def through_trampolines(fn, trampolines):
    """jmp/call F with F a trampoline: jmp/call *slot(%rip)."""
    done = []
    for i in fn.ins:
        if not i.gas or i.reloc is None or i.reloc.type != 'R_X86_64_PLT32' or i.mnemonic not in ('jmp', 'call'):
            continue
        t = trampolines.get(i.reloc.symbol)
        if t is None:
            continue
        symbol, slot = t
        i.j['ops'] = [{'kind': 'Memory', 'mem': {'base': 'none', 'index': 'none', 'scale': 1, 'disp': 0, 'size': 8, 'rip_rel': True, 'seg': None}, 'field_pos': 2}]
        i.j['mems'] = [{'base': 'none', 'index': 'none', 'scale': 1, 'disp': 0, 'size': 8, 'access': 'Read'}]
        i.j['flow'] = 'IndirectBranch' if i.mnemonic == 'jmp' else 'IndirectCall'
        i.j['encoding'] = 'Legacy'
        i.j['len'] = 6   # ff 25/15 disp32: the relocation field is at byte 2
        i.j.pop('branch_target', None)
        done.append('%s: %s %s now goes through %s+%#x' % (fn.name, i.mnemonic, i.reloc.symbol, symbol, slot))
        i.reloc = Reloc(i.home + 2, 'R_X86_64_PC32', symbol, slot - 4)
        refresh(i)
    return done


# ---------------------------------------------------------------- frame

def reg_op(i, k):
    ops = i.j['ops']
    return ops[k]['reg'] if k < len(ops) and ops[k]['kind'] == 'Register' else None


def drop_frame(fn):
    """push %rbp; mov %rsp,%rbp, at the entry or shrink-wrapped around a loop
    further in, pop %rbp / leave at the ends, nothing else that reads or writes
    rbp or rsp and no call: all of it goes."""
    live = [i for i in fn.ins if i.gas and not i.mnemonic.startswith('nop')]
    frame = []
    k = 0
    while k < len(live):
        i = live[k]
        if i.flow in ('Call', 'IndirectCall'):
            return []
        if i.mnemonic == 'push' and reg_op(i, 0) == 'rbp':
            mov = live[k + 1] if k + 1 < len(live) else None
            if not (mov and mov.mnemonic == 'mov' and reg_op(mov, 0) == 'rbp' and reg_op(mov, 1) == 'rsp'):
                return []
            frame += [i, mov]
            k += 2
            continue
        if (i.mnemonic == 'pop' and reg_op(i, 0) == 'rbp') or i.mnemonic == 'leave':
            frame.append(i)
        elif i.flow in ('Return', 'UnconditionalBranch', 'ConditionalBranch', 'IndirectBranch') and not ({'rbp'} & (uses(i) | defs(i))):
            pass  # ret reads rsp to pop the return address: unchanged once the frame is gone
        elif {'rbp', 'rsp'} & (uses(i) | defs(i)):
            return []
        k += 1
    if not any(i.mnemonic == 'push' for i in frame):
        return []
    for i in frame:
        delete(i)
    where = '' if frame[0] is live[0] else ' (around a loop)'
    return ['%s: the frame (%d instructions)%s dropped' % (fn.name, len(frame), where)]


# ---------------------------------------------------------------- vzu

def dirties(i):
    """Whether i may leave an upper half of ymm0-15 / zmm0-15 nonzero."""
    if i.flow in ('Call', 'IndirectCall'):
        return True
    return any(r['name'][:3] in ('ymm', 'zmm') and r['access'] in WRITES for r in i.j['regs'])


def clean_vzeroupper(fn):
    """Forward: may the upper state be dirty at each point? vzeroupper where it cannot goes."""
    # a block with no way in on the graph but the entry (a jump table's target) may be dirty
    entered = {id(s) for b in fn.blocks for s in b.succ}
    dirty_in = {id(b): b is not fn.blocks[0] and id(b) not in entered for b in fn.blocks}
    changed = True
    while changed:
        changed = False
        for b in fn.blocks:
            d = dirty_in[id(b)]
            for k in range(b.start, b.end):
                i = fn.ins[k]
                if i.mnemonic == 'vzeroupper':
                    d = False
                elif i.gas and dirties(i):
                    d = True
            for s in b.succ:
                if d and not dirty_in[id(s)]:
                    dirty_in[id(s)] = True
                    changed = True
    done = []
    dirty_out = {}
    for b in fn.blocks:
        d = dirty_in[id(b)]
        for k in range(b.start, b.end):
            i = fn.ins[k]
            if i.mnemonic == 'vzeroupper':
                if not d:
                    delete(i)
                    done.append('%s: vzeroupper at %04x on a clean path dropped' % (fn.name, i.offset))
                d = False
            elif i.gas and dirties(i):
                d = True
        dirty_out[id(b)] = d
    # an exit shared by clean and dirty paths (the compiler merges the returns):
    # the clean ones that fall into it or jump to it get a copy without vzeroupper
    preds = predecessors(fn)
    block_at = {b.start: b for b in fn.blocks}
    new_ins = []
    for x in fn.blocks:
        if not dirty_in[id(x)] or x.start == 0:
            continue
        chain, tail = tail_chain(fn, x)
        if not chain or not any(i.mnemonic == 'vzeroupper' for i in tail):
            continue
        for p in list(preds[id(x)]):
            if dirty_out[id(p)] or p in chain:
                continue
            last = fn.ins[p.end - 1]
            falls = last.flow in ('Next', 'ConditionalBranch') and block_at.get(p.end) is x and not (last.flow == 'ConditionalBranch' and fn.internal(last) and last.target == fn.ins[x.start].offset)
            jumps = last.flow == 'UnconditionalBranch' and fn.internal(last)
            if not (falls or jumps):
                continue
            if jumps:
                delete(last)
            new_ins.append((p.end, [clone(i) for i in tail if i.mnemonic != 'vzeroupper']))
            preds[id(x)].remove(p)
            done.append('%s: the exit at %s copied without vzeroupper for a clean way in' % (fn.name, label_of(tail[0])))
    for k, clones in sorted(new_ins, key=lambda e: e[0], reverse=True):
        fn.ins[k:k] = clones
    return done


def full_gpr(name):
    return {'eax': 'rax', 'ecx': 'rcx', 'edx': 'rdx', 'esi': 'rsi', 'edi': 'rdi'}.get(name, 'r' + name[1:-1] if name.endswith('d') else name)


def optimize(obj, ins, evex, section='.text', trampolines=None):
    """evex: the unit is compiled for AVX-512VL, so ymm16-31 and masked memory operands exist.
    section: the code section the instructions came from (-ffunction-sections puts each function in its own).
    trampolines: the one-jump entry points of the whole run, see trampolines_of."""
    report = []
    passes = os.environ.get('POST_PASSES', 'fold,rename,tail,hot,thread,ktest,kor,kfirst,lea,copy,tramp,frame,vzu').split(',')
    if not evex:  # layout passes apply to every unit; the mask/EVEX ones only to the AVX-512 ones
        passes = [p for p in passes if p in ('tail', 'hot', 'thread', 'lea', 'copy', 'tramp', 'frame', 'vzu')]
    functions = split_functions(obj, ins, section)
    for fn in functions:
        fn.tail_limit = next((i.j['tail_limit'] for i in fn.ins if 'tail_limit' in i.j), TAIL_MAX)
        skipped = next((i.j['post_skip_passes'] for i in fn.ins if 'post_skip_passes' in i.j), ())
        analyze(fn)
        for name in passes:
            if name in skipped:
                continue
            if name == 'fold':
                report += fold_masked_compares(fn)
            elif name == 'tail':
                report += duplicate_tails(fn)
            elif name == 'hot':
                report += hot_tails(fn)
            elif name == 'ktest':
                report += kortest_to_test(fn)
            elif name == 'kor':
                report += kor_before_branch(fn)
            elif name == 'kfirst':
                report += mask_op_without_memory_first(fn)
            elif name == 'tramp':
                report += through_trampolines(fn, trampolines or {})
            elif name == 'thread':
                report += thread_jumps(fn)
            elif name == 'lea':
                report += fold_lea_add(fn)
            elif name == 'zcmp':
                report += zero_compares(fn)
            elif name == 'copy':
                report += sink_copies(fn)
            elif name == 'rename':
                report += rename_vectors(fn)
            elif name == 'frame':
                report += drop_frame(fn)
            elif name == 'vzu':
                report += clean_vzeroupper(fn)
            elif name:
                raise ValueError('unknown pass ' + name)
            analyze(fn)
    # Rebuild the instruction list: functions may have grown or shrunk.
    starts = {fn.lo: fn for fn in functions}   # keyed by the original start: a pass may have put a new instruction first
    inside = {i.offset for fn in functions for i in fn.ins}
    out = []
    for i in ins:
        if i.offset in starts:
            out += starts[i.offset].ins
        elif i.offset not in inside:
            out.append(i)
    return out, report
