"""The IR passes, in the order they run over a unit.

Every function first gets its locals into SSA and a cleanup (early); then,
bottom-up over the calls, each function inlines its callees and runs the
rest: cleanup to a fixed point, full unrolling, if-conversion, loop-invariant
motion and sinking."""
import os

from .cfgsimp import simplify_cfg, split_returns
from .common import merge_tails
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


def optimize(module):
    for fn in module.functions.values():
        early(fn)
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
