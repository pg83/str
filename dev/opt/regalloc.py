"""Register allocation by graph coloring with coalescing (iterated register
coalescing, George and Appel), one class at a time: general registers, vector
registers, mask registers.

Liveness is computed over the blocks; every register interferes with the
registers live where it is defined (the physical ones a call or an
instruction clobbers among them). A copy between two registers that do not
interfere is coalesced when the merged node still has fewer than K neighbors
of significant degree (Briggs) or, against a physical register, every
neighbor already interferes with it or has a small degree (George). The
first attempt skips both tests and coalesces every copy that does not
interfere (the functions are small, the tests too timid for their unrolled
blocks); when that leaves something uncolorable the allocation is redone
with the tests. What cannot be colored then is spilled to a stack slot: its
uses and definitions get a short-lived register around them, and the
allocation runs again."""
from .mir import Reg, Mem, Ins, VReg, GPR, VEC, MASK, CALLER_SAVED, CALLEE_SAVED, FLAGS, FLAG_WRITERS, bank

GPR_POOL = CALLER_SAVED + CALLEE_SAVED      # the cheap ones first: no save and restore


def liveness(mf):
    """live-in and live-out sets of virtual and physical registers per block."""
    use_def = {}
    for b in mf.blocks:
        use, de = set(), set()
        for i in b.ins:
            for r in i.uses():
                if r not in de:
                    use.add(r)
            de.update(i.defs())
        use_def[b] = (use, de)
    live_in = {b: set() for b in mf.blocks}
    live_out = {b: set() for b in mf.blocks}
    changed = True
    while changed:
        changed = False
        for b in reversed(mf.blocks):
            out = set()
            for s in b.succs:
                out |= live_in[s]
            use, de = use_def[b]
            inn = use | (out - de)
            if out != live_out[b] or inn != live_in[b]:
                live_out[b], live_in[b] = out, inn
                changed = True
    return live_in, live_out


def is_vreg(r):
    return isinstance(r, VReg)


class Allocator:
    def __init__(self, mf, cls, prefer, aggressive=True):
        self.mf, self.cls, self.aggressive = mf, cls, aggressive
        if cls == 'gpr':
            self.pool = GPR_POOL
        elif cls == 'vec':
            self.pool = VEC if mf.isa in ('evex', 'evex512') else VEC[:16]
        else:
            self.pool = MASK
        self.prefer = prefer      # vreg -> ordered register names to try first
        self.K = len(self.pool)

    def build(self):
        mf = self.mf
        live_in, live_out = liveness(mf)
        adj = {}
        moves = []
        self.nodes = set()
        self.forbidden = {}
        self.cost = {}            # vreg -> how often it is read or written (what a spill would cost)

        def node_of(r):
            if is_vreg(r):
                return r if r.cls == self.cls else None
            return r if bank(r) == self.cls else None

        def link(a, b):
            if a is b or a is None or b is None:
                return
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        for b in mf.blocks:
            live = {node_of(r) for r in live_out[b]} - {None}
            for i in reversed(b.ins):
                defs = [n for n in (node_of(r) for r in i.defs()) if n is not None]
                uses = [n for n in (node_of(r) for r in i.uses()) if n is not None]
                copy_src = None
                if i.is_copy and isinstance(i.ops[0], Reg) and isinstance(i.ops[1], Reg):
                    d, s = node_of(i.ops[0].r), node_of(i.ops[1].r)
                    if d is not None and s is not None and d is not s:
                        moves.append((i, d, s))
                        copy_src = s
                for r in defs + uses:
                    if is_vreg(r):
                        self.cost[r] = self.cost.get(r, 0) + 1 + 4 * b.loop_depth
                for d in defs:
                    if is_vreg(d):
                        self.nodes.add(d)
                    adj.setdefault(d, set())
                    for l in live:
                        if l is not copy_src:
                            link(d, l)
                for d in defs:
                    live.discard(d)
                for u in uses:
                    live.add(u)
                    if is_vreg(u):
                        self.nodes.add(u)
                        adj.setdefault(u, set())
                if i.mask is not None and is_vreg(i.mask) and i.mask.cls == self.cls:
                    self.forbidden.setdefault(i.mask, set()).add('k0')   # k0 cannot be a write mask
        for p in self.pool:
            adj.setdefault(p, set())
        self.adj, self.moves = adj, moves

    def coalesce_and_color(self):
        adj = {n: set(ns) for n, ns in self.adj.items()}
        nodes = set(self.nodes)
        alias = {}
        color = {p: p for p in self.pool}
        precolored = set(self.pool)
        forbidden = {n: set(f) for n, f in self.forbidden.items()}

        def find(n):
            while n in alias:
                n = alias[n]
            return n

        def degree(n):
            return len(adj[n])

        progress = True
        while progress:
            progress = False
            for i, d0, s0 in self.moves:
                d, s = find(d0), find(s0)
                if d is s or (d in precolored and s in precolored) or s in adj[d]:
                    continue
                if d in precolored:
                    d, s = s, d       # the virtual one is merged into the other
                if s in precolored:
                    if s in forbidden.get(d, ()) or s not in self.pool:
                        continue
                    ok = all(degree(t) < self.K or t in precolored or t in adj[s] for t in adj[d])
                else:
                    merged = adj[d] | adj[s]
                    ok = sum(1 for t in merged if degree(t) >= self.K) < self.K
                if not ok and not self.aggressive:
                    continue
                # (aggressive: every copy goes; what cannot be colored then is a spill the caller undoes)
                for t in adj[d]:
                    adj[t].discard(d)
                    adj[t].add(s)
                    adj[s].add(t)
                adj[s].discard(s)
                del adj[d]
                alias[d] = s
                nodes.discard(d)
                if d in forbidden:
                    forbidden.setdefault(s, set()).update(forbidden.pop(d))
                progress = True
        # simplify: nodes of low degree first, then the spill candidates (the
        # most neighbors for the least use)
        stack = []
        work = set(nodes)
        deg = {n: len(adj[n]) for n in work}
        cost = {n: 0 for n in work}
        for m in self.nodes:
            n = find(m)
            if n in cost:
                cost[n] += self.cost.get(m, 0)
        while work:
            low = [n for n in work if deg[n] < self.K]
            n = min(low, key=lambda x: (deg[x], x.id)) if low else max(work, key=lambda x: (deg[x] / (cost[x] or 1), -x.id))
            work.discard(n)
            stack.append(n)
            for t in adj[n]:
                if t in deg and t in work:
                    deg[t] -= 1
        # the copies left: a node takes the register of the other side when it can
        partners = {}
        for i, d0, s0 in self.moves:
            d, s = find(d0), find(s0)
            if d is not s:
                partners.setdefault(d, []).append(s)
                partners.setdefault(s, []).append(d)
        spilled = []
        while stack:
            n = stack.pop()
            used = {color[t] for t in adj[n] if t in color}
            bad = forbidden.get(n, set())
            first = [color[p] for p in partners.get(n, ()) if p in color] + list(self.prefer.get(n, ()))
            order = first + [p for p in self.pool if p not in first]
            choice = next((p for p in order if p not in used and p not in bad), None)
            if choice is None:
                spilled.append(n)
            else:
                color[n] = choice
        return alias, color, spilled

    def run(self):
        self.build()
        alias, color, spilled = self.coalesce_and_color()
        if spilled:
            return spilled

        def find(n):
            while n in alias:
                n = alias[n]
            return n

        mapping = {n: color[find(n)] for n in self.nodes}
        cls = self.cls
        for b in self.mf.blocks:
            for i in b.ins:
                i.map(lambda r: mapping.get(r, r) if is_vreg(r) and r.cls == cls else r)
        return []


def spill(mf, vregs):
    """Give each spilled virtual register a stack slot; every use reads it into
    a fresh register just before, every definition stores from one after."""
    slots = {v: mf.slot(64 if v.cls == 'vec' else 8, 64 if v.cls == 'vec' else 8) for v in vregs}
    for b in mf.blocks:
        out = []
        for i in b.ins:
            pre, post, renames = [], [], {}
            for r in set(i.regs()):
                if r not in slots:
                    continue
                t = mf.vreg(r.cls)
                renames[r] = t
                bits = 512 if r.cls == 'vec' else 64
                mem = Mem(bits // 8, slot=slots[r])
                if r in i.uses():
                    pre.append(load_ins(mf, Reg(t, bits), mem))
                if r in i.defs():
                    post.append(store_ins(mf, mem, Reg(t, bits)))
            i.map(lambda r: renames.get(r, r))
            out.extend(pre)
            out.append(i)
            out.extend(post)
        b.ins = out


def load_ins(mf, reg, mem):
    if reg.r.cls == 'vec':
        evex = mf.isa in ('evex', 'evex512')
        bits = 512 if evex else 256 if mf.isa == 'avx2' else 128
        return Ins('vmovdqu64' if evex else 'vmovdqu' if mf.isa == 'avx2' else 'movdqu', [Reg(reg.r, bits), mem], 'w r')
    if reg.r.cls == 'mask':
        return Ins('kmovq', [Reg(reg.r, 64), mem], 'w r')
    return Ins('mov', [Reg(reg.r, 64), mem], 'w r')


def store_ins(mf, mem, reg):
    i = load_ins(mf, reg, mem)
    i.ops.reverse()
    return i


def dead_code(mf):
    """Instructions whose results nobody reads and that do nothing else (by
    liveness: a two-address chain that feeds nothing dies whole)."""
    changed = True
    while changed:
        changed = False
        _, live_out = liveness(mf)
        for b in mf.blocks:
            live = set(live_out[b])
            flags = False       # read by a later instruction (a branch on the result of an and, say)
            kept = []
            for i in reversed(b.ins):
                defs = i.defs()
                writes_flags = i.op in FLAG_WRITERS or FLAGS in i.imp_def
                removable = (defs and all(is_vreg(r) and r not in live for r in defs)
                             and not any(isinstance(o, Mem) and 'w' in a for o, a in zip(i.ops, i.acc))
                             and i.kind not in ('call', 'tail', 'jmp', 'jcc', 'ret', 'cmp', 'string', 'trap')
                             and not i.imp_def - {'flags'} and not (writes_flags and flags))
                if removable:
                    changed = True
                    continue
                if writes_flags:
                    flags = False
                if i.kind == 'jcc' or i.op in ('cmov', 'set', 'adc', 'sbb') or FLAGS in i.imp_use:
                    flags = True
                live.difference_update(defs)
                live.update(i.uses())
                kept.append(i)
            kept.reverse()
            b.ins = kept


def widths(mf):
    """The widest use of each virtual vector register."""
    out = {}
    for b in mf.blocks:
        for i in b.ins:
            for o in i.ops:
                if isinstance(o, Reg) and is_vreg(o.r):
                    out[o.r] = max(out.get(o.r, 0), o.bits)
    return out


def allocate(mf):
    dead_code(mf)
    prefer = {}
    if mf.isa in ('evex', 'evex512'):
        # 256- and 512-bit values in the upper sixteen: nothing dirties the
        # upper halves of ymm0-15, so no vzeroupper is needed; 128-bit values
        # in the lower sixteen keep their shorter VEX encodings
        for v, bits in widths(mf).items():
            if v.cls == 'vec':
                prefer[v] = VEC[16:] + VEC[:16] if bits >= 256 else VEC[:16] + VEC[16:]
    for cls in ('mask', 'vec', 'gpr'):
        for attempt in range(8):
            spilled = Allocator(mf, cls, prefer).run()
            if spilled:
                spilled = Allocator(mf, cls, prefer, aggressive=False).run()
            if not spilled:
                break
            spill(mf, spilled)
        else:
            raise ValueError('%s: register allocation of the %s class did not converge' % (mf.name, cls))
    mf.used = set()
    for b in mf.blocks:
        b.ins = [i for i in b.ins if not (i.kind == 'copy' and isinstance(i.ops[0], Reg) and isinstance(i.ops[1], Reg) and i.ops[0].r == i.ops[1].r)]
        for i in b.ins:
            mf.used.update(r for r in i.defs() if isinstance(r, str))
