"""Relocatable ELF64 objects and GNU ar archives from the object model."""
import struct

from .objmodel import Symbol

RELOC_TYPES = {'R_X86_64_64': 1, 'R_X86_64_PC32': 2, 'R_X86_64_PLT32': 4, 'R_X86_64_32': 10, 'R_X86_64_32S': 11}
SECTION_TYPES = {'SHT_PROGBITS': 1, 'SHT_NOBITS': 8, 'SHT_NOTE': 7, 'SHT_INIT_ARRAY': 14, 'SHT_FINI_ARRAY': 15}


def write_object(obj):
    """The bytes of obj as an ELF relocatable file; the global symbols it defines are returned too."""
    sections = list(obj.sections.values())
    names = ['']
    contents = [b'']
    headers = [dict(type=0, flags=0, link=0, info=0, align=0, entsize=0, nobits=None)]
    index = {}
    for s in sections:
        index[s.name] = len(names)
        names.append(s.name)
        contents.append(s.data if s.type != 'SHT_NOBITS' else b'')
        headers.append(dict(type=SECTION_TYPES[s.type], flags=s.flags, link=0, info=0, align=s.align, entsize=s.entsize,
                            nobits=s.nobits_size if s.type == 'SHT_NOBITS' else None))
    # symbols: the null one, the locals (a section symbol for every section with relocations against it), the globals
    symbols = [Symbol('', 0, 0, 'STT_NOTYPE', 'STB_LOCAL', 'UND')]
    locals_ = [s for s in obj.symbols if s.bind == 'STB_LOCAL']
    globals_ = [s for s in obj.symbols if s.bind != 'STB_LOCAL']
    referenced_sections = {r.symbol for s in sections for r in s.relocs if r.symbol_is_section}
    have_section_symbol = {s.section for s in locals_ if s.type == 'STT_SECTION'}
    for name in names[1:]:
        if name in referenced_sections and name not in have_section_symbol:
            locals_.append(Symbol(name, 0, 0, 'STT_SECTION', 'STB_LOCAL', name))
    symbols.extend(locals_)
    first_global = len(symbols)
    symbols.extend(globals_)
    defined = {s.name for s in symbols if s.name}
    needed = sorted({r.symbol for s in sections for r in s.relocs if not r.symbol_is_section} - defined)
    symbols.extend(Symbol(n, 0, 0, 'STT_NOTYPE', 'STB_GLOBAL', 'UND') for n in needed)
    strtab = bytearray(b'\0')
    name_offset = {}
    for s in symbols:
        if s.name and s.type != 'STT_SECTION' and s.name not in name_offset:
            name_offset[s.name] = len(strtab)
            strtab += s.name.encode() + b'\0'
    symtab = bytearray()
    by_name, by_section = {}, {}
    for n, s in enumerate(symbols):
        if s.type == 'STT_SECTION':
            by_section.setdefault(s.section, n)
        elif s.name:
            by_name.setdefault(s.name, n)
        shndx = {'UND': 0, 'ABS': 0xfff1}.get(s.section, index.get(s.section))
        if shndx is None:
            raise ValueError('symbol %s in an unknown section %s' % (s.name, s.section))
        t = {'STT_NOTYPE': 0, 'STT_OBJECT': 1, 'STT_FUNC': 2, 'STT_SECTION': 3, 'STT_FILE': 4}[s.type]
        b = {'STB_LOCAL': 0, 'STB_GLOBAL': 1, 'STB_WEAK': 2}[s.bind]
        symtab += struct.pack('<IBBHQQ', name_offset.get(s.name, 0) if s.type != 'STT_SECTION' else 0, (b << 4) | t, 0, shndx, s.value, s.size)
    symtab_index = len(names) + sum(1 for s in sections if s.relocs)
    for s in sections:
        if not s.relocs:
            continue
        body = bytearray()
        for r in sorted(s.relocs, key=lambda r: r.offset):
            symi = by_section[r.symbol] if r.symbol_is_section else by_name[r.symbol]
            body += struct.pack('<QQq', r.offset, (symi << 32) | RELOC_TYPES[r.type], r.addend)
        names.append('.rela' + s.name)
        contents.append(bytes(body))
        headers.append(dict(type=4, flags=0x40, link=symtab_index, info=index[s.name], align=8, entsize=24, nobits=None))
    names.append('.symtab')
    contents.append(bytes(symtab))
    headers.append(dict(type=2, flags=0, link=symtab_index + 1, info=first_global, align=8, entsize=24, nobits=None))
    names.append('.strtab')
    contents.append(bytes(strtab))
    headers.append(dict(type=3, flags=0, link=0, info=0, align=1, entsize=0, nobits=None))
    shstrtab = bytearray(b'\0')
    name_pos = []
    for n in names:
        name_pos.append(len(shstrtab) if n else 0)
        if n:
            shstrtab += n.encode() + b'\0'
    name_pos.append(len(shstrtab))
    shstrtab += b'.shstrtab\0'
    names.append('.shstrtab')
    contents.append(bytes(shstrtab))
    headers.append(dict(type=3, flags=0, link=0, info=0, align=1, entsize=0, nobits=None))
    out = bytearray(64)
    offsets = []
    for n, (content, h) in enumerate(zip(contents, headers)):
        align = max(h['align'], 1) if n else 1
        while len(out) % align:
            out.append(0)
        offsets.append(len(out))
        if h['nobits'] is None:
            out += content
    while len(out) % 8:
        out.append(0)
    shoff = len(out)
    for n, h in enumerate(headers):
        size = h['nobits'] if h['nobits'] is not None else len(contents[n])
        out += struct.pack('<IIQQQQIIQQ', name_pos[n], h['type'], h['flags'], 0, offsets[n] if n else 0, size, h['link'], h['info'], h['align'], h['entsize'])
    ident = b'\x7fELF' + bytes([2, 1, 1, 0]) + bytes(8)
    struct.pack_into('<16sHHIQQQIHHHHHH', out, 0, ident, 1, 62, 1, 0, 0, shoff, 0, 64, 0, 0, 64, len(headers), len(headers) - 1)
    exported = [s.name for s in globals_ if s.section not in ('UND', 'ABS')]
    return bytes(out), exported


def write_archive(path, members):
    """GNU ar with a symbol index; members: [(name, bytes, [defined global symbols])]."""
    symbols = [(symbol, n) for n, (_, _, defined) in enumerate(members) for symbol in defined]
    long_names = bytearray()
    member_names = []
    for name, _, _ in members:
        if len(name) > 15:
            member_names.append('/%d' % len(long_names))
            long_names += name.encode() + b'/\n'
        else:
            member_names.append(name + '/')

    def header(name, size):
        return ('%-16s%-12d%-6d%-6d%-8s%-10d`\n' % (name, 0, 0, 0, '644', size)).encode()

    def padded(data):
        return data + (b'\n' if len(data) % 2 else b'')

    bodies = [padded(data) for _, data, _ in members]
    symtab_size = 4 + 4 * len(symbols) + sum(len(n) + 1 for n, _ in symbols)
    pos = 8 + 60 + symtab_size + symtab_size % 2
    if long_names:
        pos += 60 + len(long_names) + len(long_names) % 2
    member_pos = []
    for data in bodies:
        member_pos.append(pos)
        pos += 60 + len(data)
    symtab = struct.pack('>I', len(symbols)) + b''.join(struct.pack('>I', member_pos[m]) for _, m in symbols) + b''.join(n.encode() + b'\0' for n, _ in symbols)
    out = bytearray(b'!<arch>\n')
    out += header('/', len(symtab)) + padded(symtab)
    if long_names:
        out += header('//', len(long_names)) + padded(bytes(long_names))
    for mname, data, (_, raw, _) in zip(member_names, bodies, members):
        out += header(mname, len(raw)) + data
    with open(path, 'wb') as f:
        f.write(out)
