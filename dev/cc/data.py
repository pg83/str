"""C static initializers to bytes, symbols and relocations."""
from objfile import Reloc, Section, Symbol

from .front import const
from .types import align_up


class Data:
    def __init__(self, obj):
        self.obj = obj
        self.locals, self.globals, self.strings = [], [], {}

    def string(self, value):
        data = value + b'\0'
        name = '.rodata.str1.1'
        if name not in self.obj.sections:
            self.obj.sections[name] = Section(name, 'SHT_PROGBITS', 0x32, 1, b'', entsize=1)
            self.locals.append(Symbol(name, 0, 0, 'STT_SECTION', 'STB_LOCAL', name))
        if data not in self.strings:
            sec = self.obj.sections[name]
            self.strings[data] = len(sec.data)
            sec.data += data
        return name, self.strings[data], True

    def address(self, expr):
        if expr.kind in ('cast', 'decay', 'address'):
            return self.address(expr.args[0])
        if expr.kind == 'string':
            return self.string(expr.value)
        if expr.kind == 'var':
            decl = expr.value
            if decl.type.kind == 'function' and 'static' in decl.storage:
                return '.text.' + decl.name, 0, True
            return decl.name, 0, False
        if expr.kind == 'member':
            symbol, offset, section = self.address(expr.args[0])
            return symbol, offset + expr.value[2], section
        if expr.kind == 'index':
            symbol, offset, section = self.address(expr.args[0])
            index = const(expr.args[1])
            if index is not None:
                return symbol, offset + index * expr.type.size, section
        if expr.kind == 'binary' and expr.value in ('+', '-'):
            delta = const(expr.args[1])
            if delta is not None and expr.args[0].type.kind == 'ptr':
                symbol, offset, section = self.address(expr.args[0])
                return symbol, offset + delta * expr.args[0].type.base.size * (1 if expr.value == '+' else -1), section
        expr.token.error('initializer is not a constant address')

    def initializer(self, data, relocs, offset, type_, init):
        if init.expr is not None:
            e = init.expr
            if type_.kind == 'array' and e.kind == 'string':
                value = (e.value + b'\0')[:type_.count]
                data[offset:offset + len(value)] = value
                return
            value = const(e)
            if value is not None:
                if type_.kind not in ('int', 'ptr'):
                    e.token.error('integer initializer of a non-scalar')
                value &= (1 << type_.bits) - 1
                data[offset:offset + type_.size] = value.to_bytes(type_.size, 'little')
            elif type_.kind == 'ptr':
                name, addend, is_section = self.address(e)
                relocs.append(Reloc(offset, 'R_X86_64_64', name, addend, is_section))
            else:
                e.token.error('static initializer is not constant')
            return
        if type_.kind not in ('array', 'struct'):
            self.initializer(data, relocs, offset, type_, init.entries[0][1])
            return
        cursor = 0
        for path, value in init.entries:
            if not path:
                path = (cursor,) if type_.kind == 'array' else (type_.base.fields[cursor][0],)
            current, at = type_, offset
            for index in path:
                if current.kind == 'array':
                    at += index * current.base.size
                    current = current.base
                else:
                    _, current, field_offset = current.base.member(index)
                    at += field_offset
            self.initializer(data, relocs, at, current, value)
            cursor = path[0] + 1 if type_.kind == 'array' else next(i for i, f in enumerate(type_.base.fields) if f[0] == path[0]) + 1

    def add(self, decl):
        type_ = decl.type
        readonly = type_
        while readonly.kind == 'array':
            readonly = readonly.base
        readonly = 'const' in readonly.qualifiers
        size, alignment = type_.size, max(type_.alignment, decl.attrs.get('aligned', 1))
        if type_.kind == 'array' and size >= 16:
            alignment = max(16, alignment)
        data, relocs = bytearray(size), []
        if decl.init is not None:
            self.initializer(data, relocs, 0, type_, decl.init)
        bss = not readonly and not relocs and not any(data)
        name = ('.bss.' if bss else '.rodata.' if readonly else '.data.') + decl.name
        self.obj.sections[name] = Section(name, 'SHT_NOBITS' if bss else 'SHT_PROGBITS', 2 if readonly else 3,
                                          alignment, b'' if bss else bytes(data), nobits_size=size, relocs=relocs)
        self.locals.append(Symbol(name, 0, 0, 'STT_SECTION', 'STB_LOCAL', name))
        sym = Symbol(decl.name, 0, size, 'STT_OBJECT', 'STB_LOCAL' if 'static' in decl.storage else 'STB_GLOBAL', name)
        (self.locals if sym.bind == 'STB_LOCAL' else self.globals).append(sym)

    def run(self, unit):
        constructors = []
        for d in unit.definitions:
            if d.type.kind == 'function':
                if d.attrs.get('constructor'):
                    constructors.append(d)
            else:
                self.add(d)
        if constructors:
            relocs = []
            for n, d in enumerate(constructors):
                relocs.append(Reloc(n * 8, 'R_X86_64_64', '.text.' + d.name, 0, True))
            self.obj.sections['.init_array'] = Section('.init_array', 'SHT_INIT_ARRAY', 3, 8, bytes(len(constructors) * 8), relocs=relocs)
        old = self.obj.symbols
        self.obj.symbols = [s for s in old if s.bind == 'STB_LOCAL'] + self.locals + [s for s in old if s.bind != 'STB_LOCAL'] + self.globals


def emit_data(obj, unit):
    Data(obj).run(unit)
