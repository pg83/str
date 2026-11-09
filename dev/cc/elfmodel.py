"""ELF compatibility details of the pinned library build.

The output contract includes symbol ordering and the old producer's .comment
record, as well as code and relocations. The compatibility record below is
preserved metadata; it does not identify the compiler running this build.
No reference object or executable compiler is needed to construct the model.
"""
from objfile import Section, Symbol


BASELINE_IDENT = b'\0clang version 23.1.5\0'


def compatibility_model(obj, code):
    definitions = {s.name: s for s in obj.symbols if s.name and s.type != 'STT_SECTION'}
    relocations = [r for s in obj.sections.values() for r in s.relocs]
    relocations.extend(i.reloc for ins in code.values() for i in ins if i.reloc is not None)
    for rel in relocations:
        sym = definitions.get(rel.symbol)
        if (not rel.symbol_is_section and sym is not None and sym.bind == 'STB_LOCAL'
                and sym.type != 'STT_NOTYPE' and sym.section in obj.sections):
            rel.symbol, rel.addend, rel.symbol_is_section = sym.section, rel.addend + sym.value, True
    globals_, seen, local_events = [], set(), []

    def symbol(name):
        if name in seen:
            return
        seen.add(name)
        sym = definitions.get(name)
        if sym is None:
            sym = Symbol(name, 0, 0, 'STT_NOTYPE', 'STB_GLOBAL', 'UND')
        if sym.bind != 'STB_LOCAL':
            globals_.append(sym)
        elif sym.type == 'STT_FUNC':
            local_events.append((sym.type, sym.name))

    # A symbol first enters the table when the assembler encounters its
    # definition or first relocation, including references to later globals.
    for name, instructions in code.items():
        for ins in instructions:
            for label in ins.j.get('constant_symbols', ()):
                local_events.append(('STT_NOTYPE', label))
        local_events.append(('STT_SECTION', name))
        for s in obj.symbols:
            if s.section == name and s.type == 'STT_FUNC':
                symbol(s.name)
        for ins in instructions:
            if ins.j.get('callee'):
                symbol(ins.j['callee'])
            if ins.reloc is not None and not ins.reloc.symbol_is_section:
                if ins.reloc.symbol != 'str_table' or not ins.j.get('callee'):
                    symbol(ins.reloc.symbol)
    for section in obj.sections.values():
        if section.name in code:
            continue
        for s in obj.symbols:
            if s.section == section.name and s.type == 'STT_OBJECT':
                symbol(s.name)
        for rel in sorted(section.relocs, key=lambda r: r.offset):
            if not rel.symbol_is_section:
                symbol(rel.symbol)

    referenced_sections = {r.symbol for s in obj.sections.values() for r in s.relocs if r.symbol_is_section}
    referenced_sections.update(i.reloc.symbol for ins in code.values() for i in ins
                               if i.reloc is not None and i.reloc.symbol_is_section)
    locals_ = [s for s in obj.symbols if s.bind == 'STB_LOCAL'
               and (s.type != 'STT_SECTION' or s.section in code or s.section in referenced_sections)]
    available = {s.section for s in locals_ if s.type == 'STT_SECTION'}
    for name in obj.sections:
        if name in referenced_sections and name not in available:
            at = next((n for n, s in enumerate(locals_) if s.section == name), len(locals_))
            locals_.insert(at, Symbol(name, 0, 0, 'STT_SECTION', 'STB_LOCAL', name))
    ranks = {event: n for n, event in enumerate(local_events)}
    positions = [n for n, s in enumerate(locals_) if (s.type, s.name) in ranks]
    ordered = sorted((locals_[n] for n in positions), key=lambda s: ranks[(s.type, s.name)])
    for n, sym in zip(positions, ordered):
        locals_[n] = sym
    obj.symbols = locals_ + globals_
    note = obj.sections.pop('.note.GNU-stack', None)
    obj.sections['.comment'] = Section('.comment', 'SHT_PROGBITS', 0x30, 1, BASELINE_IDENT, entsize=1)
    if note is not None:
        obj.sections[note.name] = note
