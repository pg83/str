"""Specialize the code per configuration.

dispatch.c names, for every class of host, a complete table: which
implementation fills each slot. The compiler sees only data, so every call
from one primitive to another goes through the table, jmp *slot(%rip), and
the code of five configurations lies interleaved, unit by unit. Here each
configuration gets its own copy of every function it reaches: the copies
call each other directly (a conditional tail call included), lie together in
call order, and are laid out as the configuration asks: a configuration whose
slots name the *_jcc twins gets its copies with the Skylake JCC padding. dispatch.c is untouched: the
relocations of str_configs, and of the initial str_table, are pointed at the
copies, so the table a host gets holds code that never looks at the table
again. The copies live in their units' objects as sections
.text.<configuration>.<function>, their global symbols
<configuration>.<function>.

Requires one function per section (-ffunction-sections).
"""
import copy

from objfile import Ins, Reloc
from postopt import refresh

# struct str_config { const char *name; unsigned needs; struct str_table table; }
SLOTS = ['memcpy', 'memmove', 'memset', 'memcmp', 'strlen', 'strcpy', 'strcat', 'strcmp', 'strspn', 'strcspn', 'memchr',
         'copy_large', 'fill_large', 'memcmp_large', 'strlen_tail', 'strcmp_tail', 'strcpy_long', 'copy_huge', 'fill_huge', 'memchr_large']
SLOT = 8
TABLE_OFFSET = 16
CONFIG_SIZE = TABLE_OFFSET + SLOT * len(SLOTS)
JCC = '_jcc'   # a slot naming the twin laid out with the JCC padding: this configuration's copies get it


class Config:
    def __init__(self, name, needs, slots):
        self.name, self.needs = name, needs
        self.jcc = any(s.endswith(JCC) for s in slots)
        self.slots = [s[:-len(JCC)] if s.endswith(JCC) else s for s in slots]   # the functions themselves

    def __repr__(self):
        return 'Config(%s, needs=%#x%s)' % (self.name, self.needs, ', jcc' if self.jcc else '')


def symbol_at(obj, reloc):
    """The function a data relocation names (a symbol, or a section plus offset)."""
    if not reloc.symbol_is_section:
        return reloc.symbol
    for s in obj.symbols:
        if s.section == reloc.symbol and s.value == reloc.addend and s.type == 'STT_FUNC':
            return s.name
    raise ValueError('no function at %s+%#x' % (reloc.symbol, reloc.addend))


def read_configs(obj):
    """The configurations of dispatch.o, from str_configs and its relocations."""
    sym = next(s for s in obj.symbols if s.name == 'str_configs')
    sec = obj.sections[sym.section]
    relocs = {r.offset: r for r in sec.relocs}
    out = []
    for i in range(sym.size // CONFIG_SIZE):
        base = sym.value + i * CONFIG_SIZE
        r = relocs[base]
        if r.symbol_is_section:
            data = obj.sections[r.symbol].data
            name = data[r.addend:data.index(b'\0', r.addend)].decode()
        else:
            s = next(s for s in obj.symbols if s.name == r.symbol)
            data = obj.sections[s.section].data
            start = s.value + r.addend
            name = data[start:data.index(b'\0', start)].decode()
        needs = int.from_bytes(sec.data[base + 8:base + 12], 'little')
        slots = [symbol_at(obj, relocs[base + TABLE_OFFSET + j * SLOT]) for j in range(len(SLOTS))]
        out.append(Config(name, needs, slots))
    return out


def initial_table(obj):
    """The functions the initial str_table (in .data) holds, by slot."""
    sym = next(s for s in obj.symbols if s.name == 'str_table')
    sec = obj.sections[sym.section]
    relocs = {r.offset: r for r in sec.relocs}
    return [symbol_at(obj, relocs[sym.value + j * SLOT]) for j in range(len(SLOTS))]


class Units:
    """The decoded objects of one run and where every function is."""

    def __init__(self, units):
        self.units = units   # [(path, obj, code)]
        self.global_fn = {}  # name -> (unit index, section)
        self.local_fn = []   # per unit: name -> section
        for u, (path, obj, code) in enumerate(units):
            local = {}
            for section in code:
                for f in obj.functions(section):
                    if f.value != 0:
                        raise ValueError('%s: %s is not at the start of its section; compile with -ffunction-sections' % (path, f.name))
                    if f.bind == 'STB_GLOBAL':
                        self.global_fn[f.name] = (u, section)
                    else:
                        local[f.name] = section
            self.local_fn.append(local)

    def function_name(self, u, section):
        obj = self.units[u][1]
        return obj.functions(section)[0].name

    def resolve(self, u, reloc, config):
        """The function a relocation of unit u refers to under the configuration: (unit, section),
        or None for anything but code (data, the ERMS flag)."""
        obj = self.units[u][1]
        if reloc.symbol_is_section:
            return (u, reloc.symbol) if reloc.symbol in self.units[u][2] else None
        name = reloc.symbol
        if name == 'str_table':
            return None
        if name.startswith('str_') and name[4:] in SLOTS:
            name = config.slots[SLOTS.index(name[4:])]
        if name.endswith(JCC):
            name = name[:-len(JCC)]
        if name in self.global_fn:
            return self.global_fn[name]
        if name in self.local_fn[u]:
            return (u, self.local_fn[u][name])
        return None


def slot_of_indirect(i):
    """jmp/call *str_table+off(%rip): the slot index, else None."""
    r = i.reloc
    if r is None or r.symbol != 'str_table' or i.mnemonic not in ('jmp', 'call') or i.j['ops'][0]['kind'] != 'Memory':
        return None
    return (r.addend + (i.home_end - r.offset)) // SLOT


def direct_branch(i, target):
    """Turn jmp/call *slot(%rip) into jmp/call target (rel32, PLT32)."""
    i.j['ops'] = [{'kind': 'NearBranch', 'branch': 0, 'size': 32, 'field_pos': 1}]
    i.j['mems'] = []
    i.j['flow'] = 'UnconditionalBranch' if i.mnemonic == 'jmp' else 'Call'
    i.j['encoding'] = 'Legacy'
    i.j['len'] = 5
    i.j.pop('branch_target', None)
    i.reloc = Reloc(i.home + 1, 'R_X86_64_PLT32', target, -4)
    refresh(i)


def specialize(units_list, configs, initial):
    """Clone every function each configuration reaches, into its unit's object.
    Returns (clones, overrides):
      clones:    per unit, [(new section name, instructions, symbol name, jcc layout, source section)] in call order
      overrides: {(section, offset): (symbol, addend)} for the data relocations of dispatch.o
    initial: the slot functions of the initial str_table, pointed at the matching configuration's clones."""
    units = Units(units_list)
    clones = [[] for _ in units_list]
    overrides = {}
    dispatch = next(u for u, (path, obj, code) in enumerate(units_list) if any(s.name == 'str_configs' for s in obj.symbols))
    dobj = units_list[dispatch][1]
    configs_sym = next(s for s in dobj.symbols if s.name == 'str_configs')
    table_sym = next(s for s in dobj.symbols if s.name == 'str_table')
    for ci, config in enumerate(configs):
        done = {}        # (unit, section) -> clone symbol name
        order = []
        work = []
        for slot in config.slots:
            work.append(units.global_fn[slot])
        while work:
            f = work.pop(0)
            if f in done:
                continue
            u, section = f
            fname = units.function_name(u, section)
            done[f] = '%s.%s' % (config.name, fname)
            order.append(f)
            for i in units_list[u][2][section]:
                if i.reloc is None:
                    continue
                j = slot_of_indirect(i)
                target = units.global_fn[config.slots[j]] if j is not None else units.resolve(u, i.reloc, config)
                if target is not None and target not in done:
                    work.append(target)
        jcc = config.jcc
        for f in order:
            u, section = f
            fname = units.function_name(u, section)
            ins = [Ins(copy.deepcopy(i.j), copy.copy(i.reloc)) for i in units_list[u][2][section]]
            for i in ins:
                if i.reloc is None:
                    continue
                j = slot_of_indirect(i)
                if j is not None:
                    direct_branch(i, done[units.global_fn[config.slots[j]]])
                    continue
                target = units.resolve(u, i.reloc, config)
                if target is not None:
                    # a function symbol, or a section plus offset: the clone's symbol, same addend
                    i.reloc = Reloc(i.reloc.offset, i.reloc.type, done[target], i.reloc.addend, False)
            clones[u].append(('.text.%s.%s' % (config.name, fname), ins, done[f], jcc, section))
        # the configuration's table slots and, for the matching one, the initial table
        base = configs_sym.value + ci * CONFIG_SIZE + TABLE_OFFSET
        for j, slot in enumerate(config.slots):
            overrides[(configs_sym.section, base + j * SLOT)] = (done[units.global_fn[slot]], 0)
        if config.slots == initial:
            for j, slot in enumerate(config.slots):
                overrides[(table_sym.section, table_sym.value + j * SLOT)] = (done[units.global_fn[slot]], 0)
    return clones, overrides
