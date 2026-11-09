"""Machine operations with virtual registers and symbolic block targets."""
from dataclasses import dataclass, field, replace
from collections import deque

import encode
import x86
from objfile import Ins, Reloc


@dataclass(frozen=True)
class Reg:
    bank: str
    id: int | str

    @property
    def physical(self):
        return isinstance(self.id, str)


@dataclass(frozen=True)
class R:
    reg: Reg
    bits: int = 64


@dataclass(frozen=True)
class Imm:
    value: int
    bits: int = 32
    symbol: str | None = None


@dataclass(frozen=True)
class Mem:
    size: int
    base: Reg | None = None
    index: Reg | None = None
    scale: int = 1
    disp: int = 0
    symbol: str | None = None
    slot: int | None = None


@dataclass(frozen=True)
class Target:
    symbol: str
    addend: int = 0


def regs(operand):
    if isinstance(operand, R):
        return {operand.reg}
    if isinstance(operand, Mem):
        return {r for r in (operand.base, operand.index) if r is not None}
    return set()


@dataclass(eq=False)
class Instr:
    mnemonic: str
    operands: tuple = ()
    access: tuple = ()
    implicit_uses: frozenset = frozenset()
    implicit_defs: frozenset = frozenset()
    target: object = None
    cc: str | None = None
    copies: tuple = ()
    rep: str | None = None
    preserve: bool = False
    mask: Reg | None = None
    zeroing: bool = False
    callee: str | None = None
    keep_immediate: bool = False
    domain: str | None = None

    @property
    def uses(self):
        out = set(self.implicit_uses)
        if self.mask is not None:
            out.add(self.mask)
        if self.cc:
            out.add(Reg('flags', 'rflags'))
        if self.mnemonic in ('xor', 'pxor', 'xorps', 'vpxor', 'vxorps', 'vpxord', 'pcmpeqd') and self.operands and all(isinstance(o, R) for o in self.operands) and len({o.reg for o in self.operands}) == 1:
            return out
        for operand, access in zip(self.operands, self.access):
            if isinstance(operand, Mem) or 'R' in access:
                out |= regs(operand)
        for _, source in self.copies:
            out |= regs(source)
        return out

    @property
    def defs(self):
        out = set(self.implicit_defs)
        if self.mnemonic in ('add', 'sub', 'imul', 'and', 'or', 'xor', 'neg', 'shl', 'shr', 'sar', 'cmp', 'test', 'inc', 'dec',
                             'tzcnt', 'lzcnt', 'bsf', 'bsr', 'bt', 'bts', 'btr', 'btc', 'bzhi', 'div', 'idiv',
                             'kortestb', 'kortestw', 'kortestd', 'kortestq', 'ptest', 'vptest'):
            out.add(Reg('flags', 'rflags'))
        out |= {o.reg for o, a in zip(self.operands, self.access) if isinstance(o, R) and 'W' in a}
        out |= {d.reg for d, _ in self.copies if isinstance(d, R)}
        return out

    def map_regs(self, mapping):
        def operand(o):
            if isinstance(o, R):
                return replace(o, reg=mapping.get(o.reg, o.reg))
            if isinstance(o, Mem):
                return replace(o, base=mapping.get(o.base, o.base), index=mapping.get(o.index, o.index))
            return o
        self.operands = tuple(map(operand, self.operands))
        self.copies = tuple((operand(d), operand(s)) for d, s in self.copies)
        self.implicit_uses = frozenset(mapping.get(r, r) for r in self.implicit_uses)
        self.implicit_defs = frozenset(mapping.get(r, r) for r in self.implicit_defs)
        self.mask = mapping.get(self.mask, self.mask)


@dataclass(eq=False)
class Block:
    id: int
    name: str
    ins: list = field(default_factory=list)
    successors: list = field(default_factory=list)


class Function:
    def __init__(self, source, isa='sse2'):
        self.source, self.name, self.isa = source, source.name, isa
        self.blocks, self.sizes, self.slots = [], {}, []
        self.preferred = {}
        self.next_reg = source.next_value
        self.entry = None

    def block(self, name):
        b = Block(len(self.blocks), name)
        self.blocks.append(b)
        if self.entry is None:
            self.entry = b
        return b

    def reg(self, type_, id_=None):
        bank = 'vec' if type_.kind == 'vector' else 'gpr'
        if id_ is None:
            id_, self.next_reg = self.next_reg, self.next_reg + 1
        reg = Reg(bank, id_)
        self.sizes[reg] = max(self.sizes.get(reg, 0), type_.bits)
        return R(reg, type_.bits)

    def slot(self, size, alignment):
        self.slots.append((size, alignment))
        return len(self.slots) - 1

    def order(self):
        if hasattr(self, 'block_order'):
            ordered = [b for b in self.block_order if b in self.blocks]
            return ordered + [b for b in self.blocks if b not in ordered]
        if getattr(self.source, 'shared_return', None) is not None:
            seen, post, stack = set(), [], [(self.entry, False)]
            while stack:
                b, done = stack.pop()
                if done:
                    post.append(b)
                elif b not in seen:
                    seen.add(b)
                    stack.append((b, True))
                    stack.extend((s, False) for s in b.successors)
            ordered = post[::-1]
            owners = getattr(self, 'layout_owners', {})
            items = list(owners.items())
            ordered_returns = getattr(self.source, 'shared_load_return', None) is not None or bool(getattr(self.source, 'scalar_read_loops', ()))
            if ordered_returns:
                def depth(b, seen=frozenset()):
                    return 0 if b not in owners or b in seen else 1 + depth(owners[b], seen | {b})
                items.sort(key=lambda pair: depth(pair[0]))
            for b, owner in items:
                if (b is not owner and b in ordered and owner in ordered and (ordered.index(b) > ordered.index(owner) + 1
                        or ordered_returns and ordered.index(b) != ordered.index(owner) + 1)):
                    ordered.remove(b)
                    ordered.insert(ordered.index(owner) + 1, b)
            if getattr(self.source, 'shared_load_return', None) is not None:
                for head in getattr(self, 'loop_headers', ()):
                    if any(isinstance(o, Mem) for ins in head.ins for o in ins.operands):
                        continue
                    latches = [b for b in ordered if b in getattr(self, 'loop_blocks', ()) and b is not head
                               and b.successors == [head] and 1 < len(b.ins) <= 4]
                    if len(latches) == 1:
                        latch = latches[0]
                        ordered.remove(latch)
                        ordered.insert(ordered.index(head), latch)
            return ordered
        seen, out, pending = set(), [], deque([self.entry])
        owners = {**getattr(self, 'layout_owners', {}), **getattr(self, 'tail_owners', {})}
        cold = getattr(self, 'cold_blocks', set())
        while pending:
            ready = [n for n, b in enumerate(pending) if b in seen or b not in owners or owners[b] in seen]
            hot = [n for n in ready if pending[n] not in cold]
            at = (hot or ready or [0])[0]
            b = pending[at]
            del pending[at]
            while b not in seen:
                if b in owners and owners[b] not in seen:
                    pending.append(b)
                    break
                if b in cold and any(p not in seen and p not in cold and (p not in owners or owners[p] in seen) for p in pending):
                    pending.append(b)
                    break
                seen.add(b)
                out.append(b)
                if not b.successors:
                    break
                successors = b.successors
                if ((self.source.type.base.kind == 'ptr' or getattr(self.source, 'distance_returns', False)) and getattr(self.source, 'return_joins', ())
                        and len(successors) == 2 and successors[0] in seen and successors[1] not in seen):
                    successors = successors[::-1]
                pending.extend(successors[1:])
                b = successors[0]
        return out


def physical(name):
    return Reg('vec' if name.startswith(('xmm', 'ymm', 'zmm')) else 'mask' if name.startswith('k') else 'gpr',
               'zmm' + name[3:] if name.startswith(('xmm', 'ymm')) else name)


def register_name(r):
    if not r.reg.physical:
        raise ValueError('unallocated virtual register %s' % (r,))
    name = r.reg.id
    if r.reg.bank == 'gpr':
        return {8: x86.R8, 16: x86.R16, 32: x86.R32, 64: x86.R64}[r.bits][x86.R64.index(name)]
    if r.reg.bank == 'vec':
        return {128: 'xmm', 256: 'ymm', 512: 'zmm'}[r.bits] + name[3:]
    return name


def structured(ins, offset=0, target=None, isa=None):
    """Construct the encoder's instruction model directly, without decoding."""
    ops, relocation = [], None
    for o in ins.operands:
        if isinstance(o, R):
            ops.append({'kind': 'Register', 'reg': register_name(o), 'size': o.bits, 'implicit': False})
        elif isinstance(o, Imm):
            ops.append({'kind': 'Immediate', 'imm': 0 if o.symbol else o.value, 'size': o.bits, 'signed': o.value < 0, 'implicit': False})
            if o.symbol:
                relocation = ('imm', 'R_X86_64_32', o.symbol, o.value)
        elif isinstance(o, Mem):
            if o.slot is not None:
                raise ValueError('stack slot not laid out')
            ops.append({'kind': 'Memory', 'mem': {'base': o.base.id if o.base else 'none', 'index': o.index.id if o.index else 'none',
                        'scale': o.scale, 'disp': 0 if o.symbol else o.disp, 'size': o.size,
                        'rip_rel': o.symbol is not None, 'seg': None}, 'field_pos': None})
            if o.symbol:
                relocation = ('disp', 'R_X86_64_PC32', o.symbol, o.disp)
        elif isinstance(o, Target):
            ops.append({'kind': 'NearBranch', 'branch': 0, 'size': 32, 'field_pos': None})
            relocation = ('branch', 'R_X86_64_PLT32', o.symbol, o.addend)
        else:
            raise ValueError('unknown machine operand: ' + repr(o))
    if ins.target is not None:
        ops.append({'kind': 'NearBranch', 'branch': target, 'size': 32, 'field_pos': None})
    name = ins.mnemonic
    if any(isinstance(o, R) and o.reg.bank == 'vec' and (int(o.reg.id[3:]) >= 16 or o.bits == 512) for o in ins.operands):
        name = {'vpxor': 'vpxord', 'vpand': 'vpandd', 'vpor': 'vpord', 'vpandn': 'vpandnd',
                'vmovdqa': 'vmovdqa64', 'vmovdqu': 'vmovdqu64', 'vinserti128': 'vinserti32x4', 'vextracti128': 'vextracti32x4'}.get(name, name)
    j = {'offset': offset, 'mnemonic': name, 'cc': ins.cc, 'ops': ops, 'form': None,
         'op_mask': ins.mask.id if ins.mask is not None else 'none', 'zeroing': ins.zeroing, 'bcast': False, 'rep': ins.rep,
         'lock': False, 'notrack': False, 'regs': [], 'mems': []}
    if isa in ('sse2', 'sse42', 'avx2'):
        j['encoding'] = 'VEX' if isa == 'avx2' else 'Legacy'
    if ins.callee is not None:
        j['callee'] = ins.callee
    external_branch = relocation is not None and relocation[0] == 'branch'
    b, fields, form = encode.encode_with_form(j, branch_size=32 if ins.target is not None or external_branch else None)
    j.update(len=len(b), bytes=b.hex(), encoding={'legacy': 'Legacy', 'vex': 'VEX', 'evex': 'EVEX'}[form.kind])
    attrs = form.attrs
    flow = next((a[5:] for a in attrs if a.startswith('flow=')), 'Next')
    if ins.mnemonic in ('call', 'jmp') and ops[0]['kind'] != 'NearBranch':
        flow = 'IndirectCall' if ins.mnemonic == 'call' else 'IndirectBranch'
    j['flow'] = flow
    if ins.target is not None:
        j['branch_target'] = target
    elif ops and ops[0]['kind'] == 'NearBranch':
        j['branch_target'] = offset + len(b)
        ops[0]['branch'] = offset + len(b)
    zero_idiom = name in x86.ZERO_IDIOMS | {'pcmpeqd'} and len(ops) >= 2 and all(o['kind'] == 'Register' for o in ops) and len({o['reg'] for o in ops}) == 1
    for n, (operand, access) in enumerate(zip(ops, ins.access)):
        if operand['kind'] == 'Register':
            if zero_idiom:
                if n:
                    continue
                access = 'W'
            j['regs'].append({'name': operand['reg'], 'full': x86.full_of(operand['reg']), 'access': {'R': 'Read', 'W': 'Write', 'RW': 'ReadWrite'}[access]})
        elif operand['kind'] == 'Memory':
            m = operand['mem']
            for reg in (m['base'], m['index']):
                if reg != 'none':
                    j['regs'].append({'name': reg, 'full': reg, 'access': 'Read'})
            if access != 'A':
                j['mems'].append({**m, 'access': {'R': 'Read', 'W': 'Write', 'RW': 'ReadWrite'}[access]})
    for reg in ins.implicit_uses | ins.implicit_defs:
        access = 'ReadWrite' if reg in ins.implicit_uses and reg in ins.implicit_defs else 'Read' if reg in ins.implicit_uses else 'Write'
        j['regs'].append({'name': reg.id, 'full': reg.id, 'access': access})
    if ins.mask is not None:
        j['regs'].append({'name': ins.mask.id, 'full': ins.mask.id, 'access': 'Read'})
    access_names = {'R': 'Read', 'W': 'Write', 'RW': 'ReadWrite'}
    for attr in attrs:
        if attr.startswith('regs='):
            for item in attr[5:].split(','):
                reg, access = item.split(':')
                effect = {'name': reg, 'full': x86.full_of(reg), 'access': access_names[access]}
                if effect not in j['regs']:
                    j['regs'].append(effect)
        elif attr.startswith('mem='):
            width = next((int(a[6:]) for a in attrs if a.startswith('msize=')), 64)
            for item in attr[4:].split(','):
                reg, access = item.split(':')
                j['mems'].append({'base': reg, 'index': 'none', 'scale': 1, 'disp': 0,
                                  'size': width // 8, 'access': access_names[access]})
        elif attr in ('stack=push', 'stack=pop'):
            push = attr == 'stack=push'
            j['mems'].append({'base': 'rsp', 'index': 'none', 'scale': 1, 'disp': -8 if push else 0,
                              'size': 8, 'access': 'Write' if push else 'Read'})
    for operand in ops:
        field = 'branch' if operand['kind'] == 'NearBranch' else 'disp' if operand['kind'] == 'Memory' else 'imm' if operand['kind'] == 'Immediate' else None
        if field in fields:
            operand['field_pos'] = fields[field][0]
    reloc = None
    if relocation:
        field, type_, symbol, addend = relocation
        pos = fields[field][0]
        reloc = Reloc(offset + pos, type_, symbol, addend - (len(b) - pos) if type_ in ('R_X86_64_PC32', 'R_X86_64_PLT32') else addend)
    j['gas'] = x86.format(j)
    return Ins(j, reloc)
