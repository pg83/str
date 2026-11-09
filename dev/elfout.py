"""Emit relocatable ELF64 objects and GNU ar archives from an object model.

The compiler can build the model directly; the post pass can also read an
existing object and supply replacement code layouts. Neither the object
emitter nor the archive emitter needs to parse an input object."""
import struct

import elf
from objfile import Object

DROP = {'.eh_frame', '.llvm_addrsig', '.symtab', '.strtab', '.shstrtab'}


def write_object(src_path, dst_path, layouts, remap, rename=None, new_sections=(), overrides=None):
    """Compatibility entry for the post pass: src_path can also be an Object."""
    obj = src_path if isinstance(src_path, Object) else Object(src_path)
    return write_model(obj, dst_path, layouts, remap, rename, new_sections, overrides)


def write_model(obj, dst_path, layouts=None, remap=None, rename=None, new_sections=(), overrides=None, defined=None):
    """layouts: {code section name: Layout}; remap(section, old offset) -> new offset;
    rename: {symbol name: new name} for the symbols defined here and the relocations against them;
    new_sections: [(name, Layout, symbol name)] code sections that did not exist in the source, each
    with one global function symbol at its start; overrides: {(section, offset): (symbol, addend)}
    for data relocations that now point elsewhere."""
    layouts = layouts or {}
    remap = remap or (lambda section, offset: offset)
    rename = rename or {}
    overrides = overrides or {}
    kept = []   # old sections that stay, in order
    for s in obj.sections.values():
        if s.type == 'SHT_RELA' or s.name in DROP or s.name.startswith('.rela'):
            continue
        kept.append(s)
    # new section list: null, kept..., .rela.X for each X with relocations, .symtab, .strtab, .shstrtab
    names = ['']
    contents = [b'']
    headers = [dict(type=0, flags=0, link=0, info=0, align=0, entsize=0)]
    new_index = {}
    for s in kept:
        new_index[s.name] = len(names)
        names.append(s.name)
        if s.name in layouts:
            contents.append(layouts[s.name].data)
        else:
            contents.append(s.data if s.type != 'SHT_NOBITS' else b'')
        headers.append(dict(type=s, flags=s.flags, link=0, info=0, align=s.align, entsize=s.entsize, nobits=s.nobits_size if s.type == 'SHT_NOBITS' else None))
    for name, lay, symname in new_sections:
        new_index[name] = len(names)
        names.append(name)
        contents.append(lay.data)
        headers.append(dict(type=1, flags=0x6, link=0, info=0, align=64, entsize=0))
    # symbols: drop the ones of dropped sections, renumber, update code values and sizes;
    # a new section gets a section symbol among the locals and its function among the globals
    symbols = []
    sym_index = {}
    new_locals = [('', 0, 0, 'STT_SECTION', 'STB_LOCAL', new_index[name]) for name, lay, symname in new_sections]
    new_globals = [(symname, 0, len(lay.data), 'STT_FUNC', 'STB_GLOBAL', new_index[name]) for name, lay, symname in new_sections]
    source_symbols = obj.symbols
    if not source_symbols or source_symbols[0].name or source_symbols[0].section != 'UND':
        from objfile import Symbol
        source_symbols = [Symbol('', 0, 0, 'STT_NOTYPE', 'STB_LOCAL', 'UND'), *source_symbols]
    for n, sym in enumerate(source_symbols):
        if sym.section not in ('UND', 'ABS', 'COMMON'):
            secname = sym.section
            if secname not in new_index:
                continue
            shndx = new_index[secname]
            value, size = sym.value, sym.size
            if secname in layouts and sym.type != 'STT_SECTION':   # a section symbol stays at 0: relocations add the mapped offset
                value = remap(secname, sym.value)
                if sym.type == 'STT_FUNC' or size:
                    size = remap(secname, sym.value + sym.size) - value
        else:
            shndx = {'UND': 0, 'ABS': 0xfff1, 'COMMON': 0xfff2}[sym.section]
            value, size = sym.value, sym.size
        if sym.bind != 'STB_LOCAL' and new_locals:
            symbols += new_locals
            new_locals = []
        sym_index[n] = len(symbols)
        name = '' if sym.type == 'STT_SECTION' else rename.get(sym.name, sym.name)
        symbols.append((name, value, size, sym.type, sym.bind, shndx))
    symbols += new_locals + new_globals
    # a pass may relocate against a symbol the object did not reference (the table behind a trampoline)
    needed = {symname for lay in layouts.values() for _, _, symname, is_section, _ in lay.relocs if not is_section}
    needed |= {r.symbol for s in kept if s.name not in layouts for r in s.relocs if not r.symbol_is_section}
    needed |= {symname for symname, _ in overrides.values()}
    for name in sorted(needed - {name for name, *_ in symbols}):
        symbols.append((name, 0, 0, 'STT_NOTYPE', 'STB_GLOBAL', 0))
    strtab = bytearray(b'\0')
    name_offset = {}
    for name, *_ in symbols:
        if name and name not in name_offset:
            name_offset[name] = len(strtab)
            strtab += name.encode() + b'\0'
    symtab = bytearray()
    first_global = len(symbols)
    for n, (name, value, size, type_, bind, shndx) in enumerate(symbols):
        t = {'STT_NOTYPE': 0, 'STT_OBJECT': 1, 'STT_FUNC': 2, 'STT_SECTION': 3, 'STT_FILE': 4}.get(type_, int(type_[4:]) if type_.startswith('STT_') and type_[4:].isdigit() else 0)
        b = {'STB_LOCAL': 0, 'STB_GLOBAL': 1, 'STB_WEAK': 2}[bind]
        if b != 0 and n < first_global:
            first_global = n
        symtab += struct.pack('<IBBHQQ', name_offset.get(name, 0), (b << 4) | t, 0, shndx, value, size)
    # relocations
    rela = {}
    for s in kept:
        entries = []
        if s.name in layouts:
            for off, type_, symname, is_section, addend in layouts[s.name].relocs:
                entries.append((off, type_, symname if is_section else rename.get(symname, symname), is_section, addend))
        else:
            for r in s.relocs:
                is_section = r.symbol_is_section
                symname = r.symbol
                addend = r.addend
                if is_section and symname in layouts:
                    addend = remap(symname, addend)
                if (s.name, r.offset) in overrides:
                    symname, addend = overrides[(s.name, r.offset)]
                    is_section = False
                entries.append((r.offset, r.type, symname if is_section else rename.get(symname, symname), is_section, addend))
        if entries:
            rela[s.name] = entries
    for name, lay, symname in new_sections:
        if lay.relocs:
            rela[name] = list(lay.relocs)
    by_name = {}
    for n, (name, value, size, type_, bind, shndx) in enumerate(symbols):
        if type_ == 'STT_SECTION':
            by_name[('section', names[shndx])] = n
        elif name:
            by_name.setdefault(name, n)
    type_codes = {v: k for k, v in elf.RELOC_TYPES.items()}
    symtab_index = len(names) + len(rela)
    for secname, entries in rela.items():
        body = bytearray()
        for off, type_, symname, is_section, addend in sorted(entries):
            symi = by_name[('section', symname)] if is_section else by_name[symname]
            body += struct.pack('<QQq', off, (symi << 32) | type_codes[type_], addend)
        names.append('.rela' + secname)
        contents.append(bytes(body))
        headers.append(dict(type=4, flags=0x40, link=symtab_index, info=new_index[secname], align=8, entsize=24))
    names.append('.symtab'); contents.append(bytes(symtab)); headers.append(dict(type=2, flags=0, link=symtab_index + 1, info=first_global, align=8, entsize=24))
    names.append('.strtab'); contents.append(bytes(strtab)); headers.append(dict(type=3, flags=0, link=0, info=0, align=1, entsize=0))
    shstrtab = bytearray(b'\0')
    name_pos = []
    for name in names:
        name_pos.append(len(shstrtab) if name else 0)
        if name:
            shstrtab += name.encode() + b'\0'
    shstr_pos = len(shstrtab)
    shstrtab += b'.shstrtab\0'
    names.append('.shstrtab'); contents.append(bytes(shstrtab)); headers.append(dict(type=3, flags=0, link=0, info=0, align=1, entsize=0))
    name_pos.append(shstr_pos)
    # file layout
    out = bytearray(64)
    offsets = []
    for n, (content, h) in enumerate(zip(contents, headers)):
        align = max(h['align'], 1) if n else 1
        while len(out) % align:
            out.append(0)
        offsets.append(len(out))
        if h.get('nobits') is None:
            out += content
    while len(out) % 8:
        out.append(0)
    shoff = len(out)
    for n, h in enumerate(headers):
        type_ = h['type']
        if not isinstance(type_, int):   # a kept original section
            s = type_
            type_ = {'SHT_PROGBITS': 1, 'SHT_NOBITS': 8, 'SHT_NOTE': 7, 'SHT_INIT_ARRAY': 14, 'SHT_FINI_ARRAY': 15, 'SHT_X86_64_UNWIND': 0x70000001}.get(s.type, int(s.type[4:]) if s.type[4:].isdigit() else 1)
        size = h['nobits'] if h.get('nobits') is not None else len(contents[n])
        out += struct.pack('<IIQQQQIIQQ', name_pos[n], type_, h['flags'], 0, offsets[n] if n else 0, size, h['link'], h['info'], h['align'], h['entsize'])
    ident = b'\x7fELF' + bytes([2, 1, 1, 0]) + bytes(8)
    struct.pack_into('<16sHHIQQQIHHHHHH', out, 0, ident, 1, 62, 1, 0, 0, shoff, 0, 64, 0, 0, 64, len(headers), len(headers) - 1)
    data = bytes(out)
    if defined is not None:
        defined.extend(name for name, _, _, _, bind, shndx in symbols if name and bind in ('STB_GLOBAL', 'STB_WEAK') and shndx != 0)
    if dst_path is not None:
        with open(dst_path, 'wb') as f:
            f.write(data)
    return data


def write_archive(path, members, defined=None):
    """GNU ar with a symbol index. defined: {member name: [global symbol names]}.
    With no symbol lists the legacy post pass reads them from its ELF members;
    the compiler supplies them from the models it has just emitted."""
    symbols = []   # (name, member index)
    for n, (name, data) in enumerate(members):
        if defined is not None:
            symbols.extend((symbol, n) for symbol in defined[name])
        else:
            import tempfile
            with tempfile.TemporaryDirectory() as where:
                from pathlib import Path
                tmp = Path(where) / 'member.o'
                tmp.write_bytes(data)
                e = elf.read(tmp)
            symbols.extend((s.name, n) for s in e.symbols if s.bind in ('STB_GLOBAL', 'STB_WEAK') and s.shndx != 0 and s.name)
    long_names = bytearray()
    member_names = []
    for name, _ in members:
        if len(name) > 15:
            member_names.append('/%d' % len(long_names))
            long_names += name.encode() + b'/\n'
        else:
            member_names.append(name + '/')

    def header(name, size):
        return ('%-16s%-12d%-6d%-6d%-8s%-10d`\n' % (name, 0, 0, 0, '644', size)).encode()

    def padded(data):
        return data + (b'\n' if len(data) % 2 else b'')

    # offsets: magic, symbol table member, long names member, then the members
    body_members = [padded(data) for _, data in members]
    symtab_size = 4 + 4 * len(symbols) + sum(len(n) + 1 for n, _ in symbols)
    pos = 8 + 60 + (symtab_size + symtab_size % 2)
    if long_names:
        pos += 60 + len(long_names) + len(long_names) % 2
    member_pos = []
    for data in body_members:
        member_pos.append(pos)
        pos += 60 + len(data)
    symtab = struct.pack('>I', len(symbols)) + b''.join(struct.pack('>I', member_pos[m]) for _, m in symbols) + b''.join(n.encode() + b'\0' for n, _ in symbols)
    out = bytearray(b'!<arch>\n')
    out += header('/', len(symtab)) + padded(symtab)
    if long_names:
        out += header('//', len(long_names)) + padded(bytes(long_names))
    for mname, data, (_, raw) in zip(member_names, body_members, members):
        out += header(mname, len(raw)) + data
    with open(path, 'wb') as f:
        f.write(out)
