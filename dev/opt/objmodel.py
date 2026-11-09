"""A relocatable object as the compiler builds it: sections with their bytes
and relocations, symbols."""
from dataclasses import dataclass, field


@dataclass
class Symbol:
    name: str
    value: int
    size: int
    type: str       # STT_FUNC, STT_OBJECT, STT_SECTION, STT_NOTYPE, STT_FILE
    bind: str       # STB_LOCAL, STB_GLOBAL
    section: str    # a section's name, or 'UND' / 'ABS'


@dataclass
class Reloc:
    offset: int
    type: str       # R_X86_64_PLT32, R_X86_64_PC32, R_X86_64_64
    symbol: str     # a symbol's name, or a section's when symbol_is_section
    addend: int
    symbol_is_section: bool = False


@dataclass
class Section:
    name: str
    type: str       # SHT_PROGBITS, SHT_NOBITS, SHT_INIT_ARRAY
    flags: int      # SHF_WRITE 1, SHF_ALLOC 2, SHF_EXECINSTR 4, SHF_MERGE 0x10, SHF_STRINGS 0x20
    align: int
    data: bytes = b''
    nobits_size: int = 0
    relocs: list = field(default_factory=list)
    entsize: int = 0


class Object:
    def __init__(self):
        self.sections = {}
        self.symbols = []

    def add_section(self, section):
        self.sections[section.name] = section
        return section
