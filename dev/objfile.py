"""Relocatable object model for the post passes: sections, symbols,
relocations, and decoded instructions of every code section (via x86.py)."""
from dataclasses import dataclass, field
from pathlib import Path

import elf
import x86


@dataclass
class Symbol:
    name: str
    value: int
    size: int
    type: str   # STT_FUNC, STT_OBJECT, STT_SECTION, STT_NOTYPE
    bind: str   # STB_LOCAL, STB_GLOBAL, STB_WEAK
    section: str  # section name, or 'UND' / 'ABS'


@dataclass
class Reloc:
    offset: int
    type: str   # R_X86_64_PLT32, R_X86_64_PC32, R_X86_64_64, R_X86_64_32, R_X86_64_32S
    symbol: str
    addend: int
    symbol_is_section: bool = False


@dataclass
class Section:
    name: str
    type: str
    flags: int
    align: int
    data: bytes
    offset: int = 0
    nobits_size: int = 0
    relocs: list = field(default_factory=list)
    entsize: int = 0

    @property
    def is_code(self):
        return bool(self.flags & 0x4) and self.type == 'SHT_PROGBITS'


@dataclass
class Ins:
    """One decoded instruction (fields from x86.decode) plus its relocation, if any."""
    j: dict
    reloc: Reloc = None

    @property
    def offset(self): return self.j['offset']
    @property
    def end(self): return self.j['offset'] + self.j['len']
    @property
    def mnemonic(self): return self.j['mnemonic']
    @property
    def gas(self): return self.j['gas']
    @property
    def flow(self): return self.j['flow']
    @property
    def target(self): return self.j.get('branch_target')
    @property
    def home(self): return self.j.get('home', self.j['offset'])  # where the bytes came from (clones differ from offset)
    @property
    def home_end(self): return self.home + self.j['len']
    def __repr__(self): return '%04x: %s' % (self.offset, self.gas)


class Object:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self.sections = {}
        self.symbols = []
        if path is None:
            return
        e = elf.read(self.path)
        for s in e.sections:
            if s.type in ('SHT_PROGBITS', 'SHT_NOBITS', 'SHT_NOTE', 'SHT_INIT_ARRAY', 'SHT_FINI_ARRAY', 'SHT_X86_64_UNWIND') and s.name:
                self.sections[s.name] = Section(s.name, s.type, s.flags, s.align, s.data, s.offset, s.size, entsize=s.entsize)
        for sym in e.symbols:
            sec = e.section_name(sym.shndx)
            name = sym.name or (sec if sym.type == 'STT_SECTION' else '')
            self.symbols.append(Symbol(name, sym.value, sym.size, sym.type, sym.bind, sec))
        for index, relocs in e.relocs.items():
            target = e.sections[index].name
            if target not in self.sections:
                continue
            for r in relocs:
                sym = e.symbols[r.symbol]
                is_sec = sym.type == 'STT_SECTION'
                self.sections[target].relocs.append(Reloc(r.offset, r.type, e.section_name(sym.shndx) if is_sec else sym.name, r.addend, is_sec))
        for sec in self.sections.values():
            sec.relocs.sort(key=lambda r: r.offset)

    def code_sections(self):
        return [s.name for s in self.sections.values() if s.is_code]

    def functions(self, section='.text'):
        return sorted((s for s in self.symbols if s.type == 'STT_FUNC' and s.section == section), key=lambda s: s.value)

    def decode(self, section='.text'):
        """Decode a code section; attach relocations by instruction range."""
        sec = self.sections[section]
        ins = [Ins(j) for j in x86.decode(sec.data, 0)]
        relocs = iter(sec.relocs)
        pending = next(relocs, None)
        for i in ins:
            while pending and pending.offset < i.offset:
                pending = next(relocs, None)
            if pending and i.offset <= pending.offset < i.end:
                i.reloc = pending
                pending = next(relocs, None)
        return ins

    def decode_text(self):
        return self.decode('.text')

    def decode_all(self):
        return {name: self.decode(name) for name in self.code_sections()}
