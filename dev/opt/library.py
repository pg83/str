"""The library: every unit compiled, and every configuration of dispatch.c
given its own copy of the code it reaches.

The sources call each other through the table (str_memcpy and the like, in
dispatch.c, jump through str_table); a configuration is a complete table, so
its copy of a function calls the configuration's implementations directly,
and the table a host gets points at these copies: <configuration>.<function>,
laid out as the configuration asks (the Skylake-SP one with the JCC
padding). The plain functions stay as well, calling through the table.

The archive is one object with one .text: every function at its 64-byte
line, its entry at its phase, calls and jumps between them resolved here.
Its API is global (the public slots' entry points, the host's choice,
the table and the configurations, derived.c); every other name local.

Built for wasm2o (abi='wasm2o', mir.WASM2O), the code keeps the registers of
the code wasm2o compiles and the object is one its loader links (pizdos,
bin/wasm2o): the read-only data in one .rodata, no constructor, an .eh_frame
with the stack pointer at every instruction, nothing outside it; derived.c is
left out (the loader binds only the table's slots). The loader calls str_choose
and binds the public slots of the table it returns."""
import json
import struct
from pathlib import Path

from . import mir
from .codegen import compile_function
from .elfout import write_archive, write_object
from .ir import Function
from .module import Module
from .objmodel import Object, Reloc, Section, Symbol
from .pipeline import optimize

UNITS = [('sse2', 'sse2'), ('sse42', 'sse42'), ('avx2', 'avx2'), ('evex', 'evex'), ('evex512', 'evex512'), ('dispatch', 'sse2'), ('derived', 'sse2')]
JCC = '_jcc'
PUBLIC = ('memcpy', 'memmove', 'memset', 'memcmp', 'strlen', 'strcpy', 'strcat', 'strcmp', 'strspn', 'strcspn', 'memchr', 'strnlen')  # the rest are kernels


class Config:
    def __init__(self, name, needs, slots, jcc):
        self.name, self.needs, self.slots, self.jcc = name, needs, slots, jcc

    def __repr__(self):
        return 'Config(%s%s)' % (self.name, ', jcc' if self.jcc else '')


def copy_function(fn, name, rename):
    """An independent copy of fn called name; rename maps symbol names."""
    new = Function(name, [p.ty for p in fn.params], fn.ret_ty, dict(fn.attrs))
    vm = dict(zip(fn.params, new.params))
    order = fn.rpo()              # a definition before its uses, phis apart
    bm = {fn.entry: new.entry}
    for b in order:
        if b is not fn.entry:
            bm[b] = new.block(b.name)
        bm[b].attrs = dict(b.attrs)
    for b in order:
        for op in b.phis():
            nop = new.phi(op.ty)
            bm[b].append(nop)
            vm[op.result] = nop.result

    def value(v):
        m = vm.get(v)
        if m is not None:
            return m
        op = v.op
        if op.kind == 'const':
            m = new.const(v.ty, op.attrs['value'])
        elif op.kind == 'sym':
            m = new.sym(rename.get(op.attrs['name'], op.attrs['name']))
        elif op.kind == 'undef':
            m = new.undef(v.ty)
        else:
            raise ValueError('%s: %s of %s is not defined before its use' % (fn.name, v, op.kind))
        vm[v] = m
        return m

    for b in order:
        nb = bm[b]
        for op in b.ops:
            if op.kind == 'phi':
                continue
            nop = new.new(op.kind, [value(a) for a in op.args], tys=[r.ty for r in op.results], attrs=dict(op.attrs))
            nb.append(nop)
            for r, nr in zip(op.results, nop.results):
                vm[r] = nr
    for b in order:
        for op in b.phis():
            nop = vm[op.result].op
            for p, a in zip(op.preds, op.args):
                nop.add_incoming(bm[p], value(a))
    for b in order:
        t = b.term
        nt = new.new(t.kind, [value(a) for a in t.args], attrs=dict(t.attrs))
        nt.targets = [bm[x] for x in t.targets]
        bm[b].set_term(nt)
    return new


def read_configs(dispatch):
    """The configurations from dispatch.c's str_configs initializer."""
    record = dispatch.ast.records['str_table']
    slot_names = [f[0] for f in record.fields]
    config_record = dispatch.ast.records['str_config']
    table_offset = config_record.member('table')[2]
    size = config_record.size
    section = dispatch.data.sections['.rodata.str_configs']
    relocs = {r.offset: r for r in section.relocs}
    strings = dispatch.data.sections.get('.rodata.str1.1')
    out = []
    for i in range(len(section.data) // size):
        base = i * size
        r = relocs[base]
        data = strings.data if r.symbol_is_section else None
        name = data[r.addend:data.index(b'\0', r.addend)].decode()
        needs = int.from_bytes(section.data[base + 8:base + 12], 'little')
        slots, jcc = {}, False
        for j, slot in enumerate(slot_names):
            sym = relocs[base + table_offset + 8 * j].symbol
            if sym.endswith(JCC):
                sym, jcc = sym[:-len(JCC)], True
            slots[slot] = sym
        out.append(Config(name, needs, slots, jcc))
    initial = {}
    table = dispatch.data.sections['.data.str_table']
    for r in table.relocs:
        initial[slot_names[r.offset // 8]] = r.symbol
    return out, initial, slot_names


def retarget_tables(dispatch, configs, initial, slot_names):
    """str_configs and the initial str_table point at the configurations' copies."""
    record = dispatch.ast.records['str_config']
    size, table_offset = record.size, record.member('table')[2]
    section = dispatch.data.sections['.rodata.str_configs']
    relocs = {r.offset: r for r in section.relocs}
    for ci, config in enumerate(configs):
        for j, slot in enumerate(slot_names):
            r = relocs[ci * size + table_offset + 8 * j]
            r.symbol, r.symbol_is_section = '%s.%s' % (config.name, config.slots[slot]), False
    matching = next((c for c in configs if c.slots == initial), None)
    if matching is not None:
        for r in dispatch.data.sections['.data.str_table'].relocs:
            r.symbol, r.symbol_is_section = '%s.%s' % (matching.name, initial[slot_names[r.offset // 8]]), False


def compile_library(source, archive, out=None, defines=None, phases=None, abi='sysv'):
    mir.use_abi(abi)
    try:
        return build_library(source, archive, out, defines, phases, abi)
    finally:
        mir.use_abi('sysv')


def build_library(source, archive, out, defines, phases, abi):
    units = [(u, isa) for u, isa in UNITS if abi != 'wasm2o' or u != 'derived']
    source, archive = Path(source), Path(archive)
    out = Path(out) if out is not None else archive.parent / 'post'
    out.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if phases is None:
        path = source / 'dev/phases.json'
        phases = json.loads(path.read_text()) if path.exists() else {}
    modules = {}
    for unit, isa in units:
        m = Module(source / (unit + '.c'), isa, defines)
        optimize(m)
        modules[unit] = m
    owner = {}        # function name -> unit
    for unit, m in modules.items():
        for name in m.functions:
            owner.setdefault(name, unit)
    configs, initial, slot_names = read_configs(modules['dispatch'])
    entries = {'str_' + s: s for s in slot_names}
    retarget_tables(modules['dispatch'], configs, initial, slot_names)
    report = []
    # the copies: every function each configuration reaches from its slots
    clones = {unit: [] for unit, _ in units}
    for config in configs:
        rename = {'str_' + s: '%s.%s' % (config.name, f) for s, f in config.slots.items()}
        todo = [(owner[name], name) for name in dict.fromkeys(config.slots.values())]
        done = set()
        while todo:
            unit, name = todo.pop(0)
            if (unit, name) in done:
                continue
            done.add((unit, name))
            fn = modules[unit].functions[name]
            local = {}
            for (ty, *key), v in fn.consts.items():
                if key and key[0] == 'sym' and v.uses:
                    target = v.op.attrs['name']
                    if target in entries:
                        continue
                    if target in modules[unit].functions:
                        # a function of the same unit (a static one, for all the clone knows): its own copy
                        local[target] = '%s.%s' % (config.name, target)
                        todo.append((unit, target))
            clones[unit].append((config, copy_function(fn, '%s.%s' % (config.name, name), {**rename, **local})))
        report.append('%s: %d functions, %s' % (config.name, len(done), 'the JCC padding' if config.jcc else 'no padding'))
    functions = []
    for unit, isa in units:
        m = modules[unit]
        mine = [(unit, isa, fn, phases.get(fn.name, 0), False) for fn in m.emitted()]
        mine += [(unit, isa, fn, phases.get(fn.name, 0), config.jcc) for config, fn in clones[unit]]
        functions += mine
        report.append('%s: %d functions' % (unit, len(mine)))
    data, exported = build_object(modules, functions, api_names(modules, slot_names), abi)
    name = 'libstr.o'
    (out / name).write_bytes(data)
    write_archive(archive, [(name, data, exported)])
    report.append('%s: %d functions in one .text, %d names exported' % (name, len(functions), len(exported)))
    (out / 'report.txt').write_text('\n'.join(report) + '\n')
    return {'archive': archive, 'objects': [out / name]}


def referenced(fn):
    """The symbols a function's code refers to."""
    return [v.op.attrs['name'] for (_, *k), v in fn.consts.items() if k and k[0] == 'sym' and v.uses]


def api_names(modules, slot_names):
    """What the object exports: the public slots' entry points and dispatch.c's other functions (the
    host's choice and its parts), the table and the configurations, derived.c's functions."""
    kernels = {'str_' + x for x in slot_names if x not in PUBLIC}
    out = set()
    for unit in ('dispatch', 'derived'):
        m = modules.get(unit)
        if m is None:
            continue
        out |= {f.name for f in m.functions.values() if not f.attrs.get('static') and f.name not in kernels}
        out |= {x.name for x in m.data.symbols if x.type == 'STT_OBJECT' and x.bind == 'STB_GLOBAL'}
    return out


def build_object(modules, functions, api, abi='sysv'):
    """functions [(unit, isa, fn, phase, jcc)] in one .text, each at a 64-byte line with the pad
    its code starts with (the entry at its phase) and int3 between them: a function's symbol is
    its entry, the pad never runs. A call or jump of one to another is resolved here (the
    unit's own function first: statics of two units may share a name, the later gets its
    unit's in front, <unit>.c.<name>); the constants pooled by size, the strings, the data
    (dispatch's and derived's, their sections as they are) and what is outside relocations. The
    API global, every other name local."""
    obj = Object()
    for unit, m in modules.items():
        for name, section in m.data.sections.items():
            if name in obj.sections:
                raise ValueError('%s: data section %s in two units' % (unit, name))
            obj.sections[name] = section
        obj.symbols += m.data.symbols
    data_names = {x.name for x in obj.symbols if x.type == 'STT_OBJECT'}
    text = bytearray()
    placed = {}        # (unit, name) -> the entry's offset
    by_name = {}       # name -> [(unit, offset)]
    sites = []         # (offset in .text, type, symbol, addend, unit)
    code_symbols = []
    pools, pool_offsets, pool_symbols = {}, {}, []
    strings = {}       # (unit, label) -> the local symbol of the string
    regions = []       # (entry, size, stack changes from the entry): the unwind tables
    for unit, isa, fn, phase, jcc in functions:
        mf, code, relocs = compile_function(fn, isa, phase=phase, jcc_align=jcc)
        start = -len(text) % 64 + len(text)
        text += b'\xcc' * (start - len(text)) + code
        entry = start + mf.entry_offset
        if (unit, fn.name) in placed:
            raise ValueError('%s: %s twice' % (unit, fn.name))
        placed[(unit, fn.name)] = entry
        by_name.setdefault(fn.name, []).append((unit, entry))
        code_symbols.append((unit, fn, entry, len(code) - mf.entry_offset))
        regions.append((entry, len(code) - mf.entry_offset, [(at - mf.entry_offset, below) for at, below in mf.stack]))
        for offset, rtype, symbol, addend in relocs:
            sites.append((start + offset, rtype, symbol, addend, unit, mf))
    relocs = []
    for at, rtype, symbol, addend, unit, mf in sites:
        if symbol in mf.constants.values():
            value = next(d for d, label in mf.constants.items() if label == symbol)
            csec = '.rodata.cst%d' % len(value)
            if csec not in pools:
                pools[csec] = Section(csec, 'SHT_PROGBITS', 0x12, len(value), b'', entsize=len(value))
            key = (csec, value)
            if key not in pool_offsets:
                label = '.LCPI%d' % len(pool_symbols)
                pool_offsets[key] = label
                pool_symbols.append(Symbol(label, len(pools[csec].data), 0, 'STT_NOTYPE', 'STB_LOCAL', csec))
                pools[csec].data += value
            relocs.append(Reloc(at, rtype, pool_offsets[key], addend))
        elif symbol.startswith('.L.str.'):
            if (unit, symbol) not in strings:
                label = '.L.str.%s.%s' % (unit, symbol[len('.L.str.'):])
                strings[(unit, symbol)] = label
                pool_symbols.append(Symbol(label, string_offset(modules[unit], obj, symbol), 0, 'STT_NOTYPE', 'STB_LOCAL', '.rodata.str1.1'))
            relocs.append(Reloc(at, rtype, strings[(unit, symbol)], addend))
        elif symbol in by_name and rtype in ('R_X86_64_PLT32', 'R_X86_64_PC32'):
            target = placed.get((unit, symbol))
            if target is None:
                if len(by_name[symbol]) != 1:
                    raise ValueError('%s: %s is in %s' % (unit, symbol, ', '.join(u for u, _ in by_name[symbol])))
                target = by_name[symbol][0][1]
            d = target + addend - at
            if not -(1 << 31) <= d < (1 << 31):
                raise ValueError('a displacement past 32 bits at %#x' % at)
            text[at:at + 4] = (d & 0xffffffff).to_bytes(4, 'little')
        elif symbol in by_name or symbol in data_names or not any(symbol == f.name for m in modules.values() for f in m.functions.values()):
            relocs.append(Reloc(at, rtype, symbol, addend))  # an address taken, the data, or outside
        else:
            raise ValueError('%s: %s refers to %s, which is not emitted' % (unit, mf.name, symbol))
    obj.sections = {'.text': Section('.text', 'SHT_PROGBITS', 6, 64, bytes(text), relocs=relocs), **obj.sections}
    # the data's references to a static function by its old section (the constructor in .init_array)
    for section in obj.sections.values():
        for r in section.relocs:
            if r.symbol_is_section and r.symbol.startswith('.text.'):
                unit, name = next((u, n) for (u, n) in placed if n == r.symbol[len('.text.'):])
                r.symbol, r.addend = '.text', r.addend + placed[(unit, name)]
    for csec, section in pools.items():
        obj.sections[csec] = section
    obj.sections['.note.GNU-stack'] = Section('.note.GNU-stack', 'SHT_PROGBITS', 0, 1, b'')
    names, func_symbols = {}, []
    for unit, fn, entry, size in code_symbols:  # a name a second unit has too: its unit in front
        name = fn.name if fn.name not in names else '%s.c.%s' % (unit, fn.name)
        names[name] = unit
        func_symbols.append(Symbol(name, entry, size, 'STT_FUNC', 'STB_GLOBAL' if name in api else 'STB_LOCAL', '.text'))
    data_locals = [x for x in obj.symbols if x.bind == 'STB_LOCAL']
    data_globals = [x for x in obj.symbols if x.bind != 'STB_LOCAL']
    file_symbol = Symbol('libstr', 0, 0, 'STT_FILE', 'STB_LOCAL', 'ABS')
    obj.symbols = ([file_symbol, Symbol('.text', 0, 0, 'STT_SECTION', 'STB_LOCAL', '.text')] + data_locals + pool_symbols
                   + [x for x in func_symbols if x.bind == 'STB_LOCAL'] + data_globals + [x for x in func_symbols if x.bind != 'STB_LOCAL'])
    if abi == 'wasm2o':
        for_wasm2o(obj, regions)
    return write_object(obj)


def for_wasm2o(obj, regions):
    """The object as wasm2o's loader takes it: the data, read-only, in one .rodata (the table
    str_dispatch would write is not written there: the loader calls str_choose), the
    constructor and the stack note dropped, and an .eh_frame of the regions. Nothing may
    be outside: the loader resolves no symbol of the library."""
    merged, data, relocs, align = {}, bytearray(), [], 16
    for name, section in obj.sections.items():
        if name in ('.text', '.init_array', '.note.GNU-stack'):
            continue
        at = -len(data) % section.align + len(data)
        data += bytes(at - len(data)) + (section.data if section.type != 'SHT_NOBITS' else bytes(section.nobits_size))
        merged[name] = at
        align = max(align, section.align)
    for name, at in merged.items():
        for r in obj.sections[name].relocs:
            relocs.append(Reloc(r.offset + at, r.type, r.symbol, r.addend, r.symbol_is_section))
    for section in [obj.sections['.text']] + [None]:
        for r in (section.relocs if section is not None else relocs):
            if r.symbol_is_section and r.symbol in merged:
                r.symbol, r.addend = '.rodata', r.addend + merged[r.symbol]
            elif r.symbol_is_section and r.symbol != '.text':
                raise ValueError('a relocation against %s' % r.symbol)
    for x in obj.symbols:
        if x.section in merged:
            x.section, x.value = '.rodata', x.value + merged[x.section]
    defined = {x.name for x in obj.symbols if x.section not in ('UND',)}
    outside = sorted({r.symbol for s in (obj.sections['.text'].relocs, relocs) for r in s if not r.symbol_is_section and r.symbol not in defined})
    if outside:
        raise ValueError('the library for wasm2o refers to %s' % ', '.join(outside))
    text = obj.sections['.text']
    frames, frame_relocs = eh_frame(regions)
    obj.sections = {'.text': text, '.rodata': Section('.rodata', 'SHT_PROGBITS', 2, align, bytes(data), relocs=relocs),
                    '.eh_frame': Section('.eh_frame', 'SHT_PROGBITS', 2, 8, frames, relocs=frame_relocs)}
    obj.symbols = [x for x in obj.symbols if x.section not in ('.init_array', '.note.GNU-stack')
                   and not (x.type == 'STT_SECTION' and x.section not in obj.sections)]


def uleb(x):
    out = bytearray()
    while True:
        b, x = x & 0x7f, x >> 7
        out.append(b | (0x80 if x else 0))
        if not x:
            return bytes(out)


def eh_frame(regions):
    """An .eh_frame for regions [(start in .text, size, [(offset, bytes below the entry's rsp)])]
    and its relocations: a CIE (the CFA rsp + 8 at an entry, the return address under it) and
    an FDE per region, its CFA offset changing where the stack pointer does."""
    def entry(body):
        body += bytes(-(len(body) + 4) % 8)          # DW_CFA_nop to the address size
        return struct.pack('<I', len(body)) + body

    # version 1, augmentation zR, code alignment 1, data alignment -8, return address in r16
    # (rip), FDE addresses pc-relative 4-byte; def_cfa rsp+8, rip at cfa-8
    out = bytearray(entry(struct.pack('<I', 0) + b'\x01zR\x00' + uleb(1) + b'\x78' + uleb(16) + uleb(1) + b'\x1b' + b'\x0c\x07\x08\x90\x01'))
    relocs = []
    for start, size, changes in regions:
        program, loc = bytearray(), 0
        for at, below in changes:
            if at >= size:
                continue
            d = at - loc
            if d < 64:
                program.append(0x40 | d)
            elif d < 256:
                program += bytes([0x02, d])
            elif d < 65536:
                program += b'\x03' + struct.pack('<H', d)
            else:
                program += b'\x04' + struct.pack('<I', d)
            program += b'\x0e' + uleb(8 + below)
            loc = at
        here = len(out)
        body = struct.pack('<I', here + 4) + struct.pack('<iI', 0, size) + uleb(0) + program
        relocs.append(Reloc(here + 8, 'R_X86_64_PC32', '.text', start, True))
        out += entry(body)
    return bytes(out), relocs


def string_offset(module, obj, label):
    """The offset of a string literal the code refers to by label, placed into .rodata.str1.1 on demand."""
    data = next(d for d, name in module.strings.items() if name == label)
    section = obj.sections.get('.rodata.str1.1')
    if section is None:
        section = obj.sections['.rodata.str1.1'] = Section('.rodata.str1.1', 'SHT_PROGBITS', 0x32, 1, b'', entsize=1)
    offsets = getattr(module, 'string_offsets', None)
    if offsets is None:
        offsets = module.string_offsets = {}
    if data not in offsets:
        offsets[data] = len(section.data)
        section.data += data + b'\0'
    return offsets[data]
