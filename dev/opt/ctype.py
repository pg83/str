"""The library's LP64 types, layout and integer conversions."""
from dataclasses import dataclass, replace


def align_up(value, alignment):
    return (value + alignment - 1) & -alignment


@dataclass(eq=False)
class Record:
    name: str
    fields: list | None = None
    size: int = 0
    alignment: int = 1

    def complete(self, fields):
        if self.fields is not None:
            raise ValueError('struct %s is already defined' % self.name)
        offset, alignment, out = 0, 1, []
        for name, type_ in fields:
            if any(f[0] == name for f in out):
                raise ValueError('duplicate struct member ' + name)
            alignment = max(alignment, type_.alignment)
            offset = align_up(offset, type_.alignment)
            out.append((name, type_, offset))
            offset += type_.size
        self.fields, self.size, self.alignment = out, align_up(offset, alignment), alignment

    def member(self, name):
        for field_ in self.fields or ():
            if field_[0] == name:
                return field_
        raise ValueError('struct %s has no member %s' % (self.name, name))


@dataclass(frozen=True)
class Type:
    kind: str
    bits: int = 0
    signed: bool = False
    rank: int = 0
    base: object = None
    count: int | None = None
    params: tuple = ()
    qualifiers: frozenset = frozenset()

    @property
    def size(self):
        if self.kind in ('int', 'ptr', 'vector'):
            return self.bits // 8
        if self.kind == 'array':
            if self.count is None:
                raise ValueError('sizeof an incomplete array')
            return self.base.size * self.count
        if self.kind == 'struct' and self.base.fields is not None:
            return self.base.size
        raise ValueError('sizeof an incomplete type: ' + str(self))

    @property
    def alignment(self):
        if self.kind == 'array':
            return self.base.alignment
        if self.kind == 'struct':
            if self.base.fields is None:
                raise ValueError('alignment of an incomplete struct')
            return self.base.alignment
        return self.size

    @property
    def scalar(self):
        return self.kind in ('int', 'ptr')

    @property
    def plain(self):
        return replace(self, qualifiers=frozenset()) if self.qualifiers else self

    def qualify(self, *qualifiers):
        return replace(self, qualifiers=self.qualifiers | set(qualifiers))

    def decay(self):
        return pointer(self.base) if self.kind == 'array' else pointer(self) if self.kind == 'function' else self.plain

    def __str__(self):
        if self.kind == 'int':
            text = ('i' if self.signed else 'u') + str(self.bits)
        elif self.kind == 'ptr':
            text = str(self.base) + '*'
        elif self.kind == 'array':
            text = '%s[%s]' % (self.base, self.count if self.count is not None else '')
        elif self.kind == 'function':
            text = '%s(%s)' % (self.base, ', '.join(map(str, self.params)))
        elif self.kind == 'struct':
            text = 'struct ' + self.base.name
        elif self.kind == 'vector':
            text = 'v' + str(self.bits)
        else:
            text = self.kind
        return (' '.join(sorted(self.qualifiers)) + ' ' if self.qualifiers else '') + text


VOID = Type('void')
CHAR = Type('int', 8, True, 1)
UCHAR = Type('int', 8, False, 1)
SHORT = Type('int', 16, True, 2)
USHORT = Type('int', 16, False, 2)
INT = Type('int', 32, True, 3)
UINT = Type('int', 32, False, 3)
LONG = Type('int', 64, True, 4)
ULONG = Type('int', 64, False, 4)
LLONG = Type('int', 64, True, 5)
ULLONG = Type('int', 64, False, 5)
V128, V256, V512 = (Type('vector', bits) for bits in (128, 256, 512))


def pointer(base):
    return Type('ptr', 64, base=base)


def array(base, count):
    return Type('array', base=base, count=count)


def function(base, params):
    return Type('function', base=base, params=tuple(p.decay() for p in params))


def promote(type_):
    type_ = type_.decay()
    return INT if type_.kind == 'int' and type_.rank < INT.rank else type_


def arithmetic(a, b):
    a, b = promote(a), promote(b)
    if a.kind != 'int' or b.kind != 'int':
        raise ValueError('expected integer operands, got %s and %s' % (a, b))
    if a == b:
        return a
    if a.signed == b.signed:
        return a if a.rank >= b.rank else b
    u, s = (b, a) if a.signed else (a, b)
    if u.rank >= s.rank:
        return u
    return s if s.bits > u.bits else replace(s, signed=False)


def integer_type(text, value):
    suffix = text.lower().rstrip('ul')
    suffix = text[len(suffix):].lower()
    rank = 5 if 'll' in suffix else 4 if 'l' in suffix else 3
    unsigned = 'u' in suffix
    decimal = not text.lower().startswith('0x') and not (len(text) > 1 and text[0] == '0')
    candidates = [INT, UINT, LONG, ULONG, LLONG, ULLONG]
    for t in candidates:
        if t.rank < rank or unsigned and t.signed or decimal and not unsigned and not t.signed:
            continue
        if value < 1 << (t.bits - t.signed):
            return t
    raise ValueError('integer literal does not fit in 64 bits: ' + text)


def convert_integer(value, type_):
    value &= (1 << type_.bits) - 1
    return value - (1 << type_.bits) if type_.signed and value >= 1 << (type_.bits - 1) else value


def compatible(a, b):
    a, b = a.plain, b.plain
    if a == b:
        return True
    if a.kind == b.kind == 'array':
        return compatible(a.base, b.base) and (a.count is None or b.count is None or a.count == b.count)
    if a.kind == b.kind == 'ptr':
        return compatible(a.base, b.base)
    if a.kind == b.kind == 'function':
        return compatible(a.base, b.base) and len(a.params) == len(b.params) and all(compatible(x, y) for x, y in zip(a.params, b.params))
    return False


ALIASES = {'size_t': ULONG, 'ptrdiff_t': LONG, 'uintptr_t': ULONG, 'intptr_t': LONG,
           'int8_t': CHAR, 'uint8_t': UCHAR, 'int16_t': SHORT, 'uint16_t': USHORT,
           'int32_t': INT, 'uint32_t': UINT, 'int64_t': LONG, 'uint64_t': ULONG,
           '__m128i': V128, '__m256i': V256, '__m512i': V512,
           '__mmask8': UCHAR, '__mmask16': USHORT, '__mmask32': UINT, '__mmask64': ULLONG}
