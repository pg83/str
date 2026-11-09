"""The library: every unit compiled, and every configuration of dispatch.c
given its own copy of the code it reaches.

The sources call each other through the table (str_memcpy and the like, in
dispatch.c, jump through str_table); a configuration is a complete table, so
its copy of a function calls the configuration's implementations directly,
and the table a host gets points at these copies: <configuration>.<function>
in the unit's object, laid out as the configuration asks (the Skylake-SP one
with the JCC padding). The plain functions stay as well, calling through the
table."""
import json
from pathlib import Path

from .codegen import compile_function
from .elfout import write_archive, write_object
from .ir import Function
from .module import Module
from .objmodel import Object, Reloc, Section, Symbol
from .pipeline import optimize

UNITS = [('sse2', 'sse2'), ('sse42', 'sse42'), ('avx2', 'avx2'), ('evex', 'evex'), ('evex512', 'evex512'), ('dispatch', 'sse2')]
JCC = '_jcc'


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


def compile_library(source, archive, out=None, defines=None, phases=None):
    source, archive = Path(source), Path(archive)
    out = Path(out) if out is not None else archive.parent / 'post'
    out.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if phases is None:
        path = source / 'dev/phases.json'
        phases = json.loads(path.read_text()) if path.exists() else {}
    modules = {}
    for unit, isa in UNITS:
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
    clones = {unit: [] for unit, _ in UNITS}
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
    objects = []
    for unit, isa in UNITS:
        m = modules[unit]
        functions = [(fn, phases.get(fn.name, 0), False) for fn in m.emitted()]
        functions += [(fn, phases.get(fn.name, 0), config.jcc) for config, fn in clones[unit]]
        obj = m.data
        data, exported = build_object(m, isa, functions, obj, unit)
        path = out / (unit + '.c.o')
        path.write_bytes(data)
        objects.append((path.name, data, exported))
        report.append('%s: %d functions' % (unit, len(functions)))
        if unit == 'dispatch':
            pass
    write_archive(archive, objects)
    (out / 'report.txt').write_text('\n'.join(report) + '\n')
    return {'archive': archive, 'objects': [out / n for n, _, _ in objects]}


def build_object(module, isa, functions, obj, unit):
    pools = {}      # constant size -> Section
    pool_offsets = {}
    obj.sections = {'.text': Section('.text', 'SHT_PROGBITS', 6, 4, b''), **obj.sections}
    file_symbol = Symbol(module.path.name, 0, 0, 'STT_FILE', 'STB_LOCAL', 'ABS')
    code_symbols, section_symbols = [], []
    sizes = {}
    pool_symbols = []     # the constants and strings the code refers to: local symbols, as the
    # linker wants for mergeable sections (a section-relative addend before the piece is an error)
    emitted = {f.name for f, _, _ in functions}
    statics = {f.name for f, _, _ in functions if f.attrs.get('static')}
    for fn, phase, jcc in functions:
        mf, code, relocs = compile_function(fn, isa, phase=phase, jcc_align=jcc)
        name = '.text.' + fn.name
        section = Section(name, 'SHT_PROGBITS', 6, 64, code)
        for offset, rtype, symbol, addend in relocs:
            if symbol in mf.constants.values():
                data = next(d for d, label in mf.constants.items() if label == symbol)
                csec = '.rodata.cst%d' % len(data)
                if csec not in pools:
                    pools[csec] = Section(csec, 'SHT_PROGBITS', 0x12, len(data), b'', entsize=len(data))
                key = (csec, data)
                if key not in pool_offsets:
                    label = '.LCPI%d' % len(pool_symbols)
                    pool_offsets[key] = label
                    pool_symbols.append(Symbol(label, len(pools[csec].data), 0, 'STT_NOTYPE', 'STB_LOCAL', csec))
                    pools[csec].data += data
                section.relocs.append(Reloc(offset, rtype, pool_offsets[key], addend))
            elif symbol.startswith('.L.str.'):
                if symbol not in pool_offsets:
                    pool_symbols.append(Symbol(symbol, string_offset(module, obj, symbol), 0, 'STT_NOTYPE', 'STB_LOCAL', '.rodata.str1.1'))
                    pool_offsets[symbol] = symbol
                section.relocs.append(Reloc(offset, rtype, symbol, addend))
            elif symbol in module.functions and module.functions[symbol].attrs.get('static') and symbol not in emitted:
                raise ValueError('%s: call to %s, which is not emitted' % (fn.name, symbol))
            elif '.' in symbol and symbol not in emitted and symbol.split('.', 1)[1] in module.functions \
                    and module.functions[symbol.split('.', 1)[1]].attrs.get('static'):
                raise ValueError('%s: call to %s, a copy this object does not hold' % (fn.name, symbol))
            else:
                section.relocs.append(Reloc(offset, rtype, symbol, addend))
        obj.sections[name] = section
        sizes[name] = len(code)
        bind = 'STB_LOCAL' if fn.attrs.get('static') else 'STB_GLOBAL'
        code_symbols.append(Symbol(fn.name, 0, len(code), 'STT_FUNC', bind, name))
        section_symbols.append(Symbol(name, 0, 0, 'STT_SECTION', 'STB_LOCAL', name))
    for csec, section in pools.items():
        obj.sections[csec] = section
    obj.sections['.note.GNU-stack'] = Section('.note.GNU-stack', 'SHT_PROGBITS', 0, 1, b'')
    data_locals = [s for s in obj.symbols if s.bind == 'STB_LOCAL']
    data_globals = [s for s in obj.symbols if s.bind != 'STB_LOCAL']
    obj.symbols = ([file_symbol] + section_symbols + data_locals + pool_symbols + [s for s in code_symbols if s.bind == 'STB_LOCAL']
                   + data_globals + [s for s in code_symbols if s.bind != 'STB_LOCAL'])
    return write_object(obj)


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
