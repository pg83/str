"""The IR passes, in the order they run over a unit.

Every function first gets its locals into SSA and a cleanup (early); then,
bottom-up over the calls, each function inlines its callees and runs the
rest: cleanup to a fixed point, full unrolling, if-conversion, loop-invariant
motion and sinking."""
import os

from .cfgsimp import simplify_cfg, split_returns
from .common import duplicate_tails, merge_tails
from .cse import cse
from .cvp import cvp
from .ifconv import ifconv
from .inline import bottom_up, callee_of, inline_calls
from .place import hoist, sink
from .simplify import dead_code, simplify
from .ssa import mem2reg, sroa
from .unroll import unroll

VERIFY = os.environ.get('OPT_VERIFY', '1') == '1'


def check(fn, where):
    if VERIFY:
        try:
            fn.verify()
        except ValueError as e:
            raise ValueError('%s, after %s:\n%s' % (e, where, fn.dump())) from None


def cleanup(fn):
    for _ in range(16):
        changed = sroa(fn)
        changed = mem2reg(fn) or changed
        check(fn, 'mem2reg')
        changed = simplify(fn) or changed
        check(fn, 'simplify')
        changed = cvp(fn) or changed
        check(fn, 'cvp')
        changed = simplify_cfg(fn) or changed
        check(fn, 'simplify_cfg')
        changed = cse(fn) or changed
        check(fn, 'cse')
        if not changed:
            return


def early(fn):
    cleanup(fn)


def late(fn):
    for _ in range(8):
        cleanup(fn)
        changed = unroll(fn)
        check(fn, 'unroll')
        if not changed:
            break
    if ifconv(fn):
        check(fn, 'ifconv')
        cleanup(fn)
    hoist(fn)
    check(fn, 'hoist')
    sink(fn)
    check(fn, 'sink')
    cleanup(fn)
    split_returns(fn)
    check(fn, 'split_returns')
    dead_code(fn)
    if merge_tails(fn):
        check(fn, 'merge_tails')
        dead_code(fn)
        simplify_cfg(fn)
        check(fn, 'simplify_cfg')
    if duplicate_tails(fn):
        check(fn, 'duplicate_tails')
        simplify(fn)
        simplify_cfg(fn)
        check(fn, 'simplify_cfg')


def param_ranges(module):
    """A static function called with constants alone in some argument position
    knows that parameter's range (truffle_scan's members, 0 or 1)."""
    known = {}       # (function, parameter index) -> (lo, hi), False when some call passes anything
    for _ in range(8):
        seen = {}
        for fn in module.functions.values():
            for op in fn.ops():
                if op.kind != 'call':
                    continue
                c = callee_of(op, module)
                if c is None:
                    continue
                ranges = seen.setdefault(c.name, [None] * len(c.params))
                for i, a in enumerate(op.args[1:len(c.params) + 1]):
                    if ranges[i] is False:
                        continue
                    if a.const is not None:
                        r = (a.const, a.const)
                    elif a.op.kind == 'param' and known.get((fn.name, a.op.attrs['index'])):
                        r = known[(fn.name, a.op.attrs['index'])]      # a parameter passed on, its range known so far
                    elif a.op.kind == 'param' and known.get((fn.name, a.op.attrs['index'])) is None and fn.attrs.get('static'):
                        continue                                        # not known yet: another round may tell
                    else:
                        ranges[i] = False
                        continue
                    lo, hi = ranges[i] if ranges[i] else r
                    ranges[i] = (min(lo, r[0]), max(hi, r[1]))
        new = {}
        for name, ranges in seen.items():
            fn = module.functions[name]
            if not fn.attrs.get('static') or fn.attrs.get('address_taken'):
                continue
            for i, (p, r) in enumerate(zip(fn.params, ranges)):
                if p.ty.kind == 'i':
                    new[(name, i)] = r
        if new == known:
            break
        known = new
    for (name, i), r in known.items():
        if r:
            module.functions[name].params[i].op.attrs['range'] = r


def optimize(module):
    for fn in module.functions.values():
        early(fn)
    param_ranges(module)
    calls = {}
    for fn in module.functions.values():
        for b in fn.blocks:
            for op in b.ops:
                if op.kind == 'call':
                    c = callee_of(op, module)
                    if c is not None:
                        calls[c.name] = calls.get(c.name, 0) + 1
    for name in bottom_up(module):
        fn = module.functions[name]
        inline_calls(module, fn, calls)
        check(fn, 'inline')
        late(fn)
    return module
