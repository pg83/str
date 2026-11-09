"""SSA repair after a pass gives a value more than one definition (a block
cloned for a path, the copies of an unrolled loop): every use reads the
definition that reaches it, with phis where definitions meet. The algorithm
of Braun et al., "Simple and Efficient Construction of SSA Form" (2013)."""


class SSAUpdater:
    def __init__(self, fn, ty, preds):
        self.fn, self.ty, self.preds = fn, ty, preds
        self.defs = {}          # block -> the value at its end
        self.replaced = {}      # trivial phi value -> what replaced it

    def add(self, block, value):
        self.defs[block] = value

    def resolve(self, v):
        while v in self.replaced:
            v = self.replaced[v]
        return v

    def at_end(self, block):
        v = self.defs.get(block)
        if v is not None:
            return self.resolve(v)
        return self.at_start(block)

    def at_start(self, block):
        preds = self.preds[block]
        if len(preds) == 1:
            v = self.at_end(preds[0])
        elif not preds:
            v = self.fn.undef(self.ty)
        else:
            phi = self.fn.phi(self.ty)
            block.insert(0, phi)
            self.defs[block] = phi.result
            for p in preds:
                phi.add_incoming(p, self.at_end(p))
            v = self.trivial(phi)
        # only blocks without a definition get here: the value at the start is the one at the end
        self.defs[block] = v
        return self.resolve(v)

    def trivial(self, phi):
        same = None
        for a in phi.args:
            a = self.resolve(a)
            if a is same or a is phi.result:
                continue
            if same is not None:
                return phi.result
            same = a
        if same is None:
            same = self.fn.undef(self.ty)
        users = [u for u in phi.result.uses if u is not phi]
        self.fn.replace_all_uses(phi.result, same)
        phi.detach()
        if phi.block is not None:
            phi.block.remove(phi)
        self.replaced[phi.result] = same
        for u in users:
            if u.kind == 'phi' and u.block is not None and u.result is not None:
                self.trivial(u)
        return same

    def rewrite(self, value, skip):
        """Every use of value outside the blocks in skip reads the definition that reaches it."""
        for user in list(value.uses):
            if user.block is None:
                continue
            if user.kind == 'phi':
                for i, (p, a) in enumerate(zip(list(user.preds), list(user.args))):
                    if a is value and p not in skip:
                        user.set_arg(i, self.at_end(p))
            elif user.block not in skip:
                user.replace_arg(value, self.at_start(user.block))


def repair(fn, defs, preds=None):
    """defs: {original value: [(block, value)]}, the original's own block among them.
    The uses outside those blocks get the reaching definition."""
    preds = preds or fn.preds()
    for original, pairs in defs.items():
        up = SSAUpdater(fn, original.ty, preds)
        for block, v in pairs:
            up.add(block, v)
        up.rewrite(original, {b for b, _ in pairs})
