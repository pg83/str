"""Recognize small constant-trip C loops before constructing their CFG."""
from .front import Expr, Stmt, const
from . import types as T


def variable(e):
    while e.kind == 'cast':
        e = e.args[0]
    return e.value if e.kind == 'var' else None


def contains_loop(node):
    return isinstance(node, Stmt) and (node.kind in ('while', 'do', 'for') or any(contains_loop(a) for a in node.args))


def fixed_for(s, limit=16):
    init, cond, step, body = s.args
    if init is None or init.kind != 'decl' or len(init.value) != 1 or cond is None or step is None:
        return None
    d = init.value[0]
    if d.type.kind != 'int' or d.init is None or d.init.expr is None:
        return None
    start = const(d.init.expr)
    if start is None or cond.kind != 'binary' or cond.value not in ('<', '<=') or variable(cond.args[0]) is not d:
        return None
    bound = cond.args[1]
    end = const(bound)
    conversions = []
    while end is None and bound.kind == 'cast':
        conversions.append(bound.type)
        bound = bound.args[0]
        end = const(bound)
    if end is None and bound.kind == 'var':
        declaration = bound.value
        if 'const' in declaration.type.qualifiers and declaration.init is not None and declaration.init.expr is not None:
            end = const(declaration.init.expr)
    if end is not None:
        for type_ in reversed(conversions):
            end = T.convert_integer(end, type_)
    if end is None:
        return None
    def increments(expr):
        return increments(expr.args[0]) + increments(expr.args[1]) if expr.kind == 'binary' and expr.value == ',' else [expr]
    steps, induction = increments(step), []
    for part in steps:
        if part.kind in ('preinc', 'postinc') and variable(part.args[0]) is d:
            induction.append((part, part.value))
        elif part.kind == 'compound_assign' and variable(part.args[0]) is d and part.token.text == '+=':
            rhs = part.args[1]
            while rhs.kind == 'cast':
                rhs = rhs.args[0]
            induction.append((part, const(rhs.args[1]) if rhs.kind == 'binary' and rhs.value == '+' else None))
    if len(induction) != 1:
        return None
    increment, delta = induction[0]
    if delta is None or delta <= 0:
        return None
    stop = end + (cond.value == '<=')
    count = max(0, (stop - start + delta - 1) // delta)
    if count > limit or start < 0 or start + count * delta >= 1 << (d.type.bits - d.type.signed):
        return None

    def safe(node):
        if isinstance(node, Expr):
            if node.kind in ('address', 'assign', 'compound_assign', 'preinc', 'postinc') and variable(node.args[0]) is d:
                return False
            return all(safe(a) for a in node.args)
        if isinstance(node, Stmt):
            if node.kind in ('label', 'goto', 'asm'):
                return False
            if node.kind == 'decl':
                return all(x.init is None or x.init.expr is None or safe(x.init.expr) for x in node.value)
            return all(safe(a) for a in node.args)
        return True
    return count if safe(body) and all(safe(part) for part in steps if part is not increment) else None
