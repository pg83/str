"""Typed builtin operations. No intrinsic headers or compiler declarations.

Memory operations name their access width, alignment and optional mask;
selection must preserve their fault suppression. Immediate operands are
checked by the frontend, before instruction selection.
"""
from dataclasses import dataclass

from .ctype import VOID, CHAR, INT, UINT, ULONG, LLONG, ULLONG, V128, V256, V512, USHORT, pointer


@dataclass(frozen=True)
class Builtin:
    name: str
    result: object
    params: tuple
    op: str
    isa: str = 'sse2'
    memory: str = ''
    width: int = 0
    alignment: int = 1
    mask: int | None = None
    immediates: tuple = ()


BUILTINS = {}


def add(name, result, params, op, isa='sse2', **attrs):
    assert name not in BUILTINS, name
    BUILTINS[name] = Builtin(name, result, tuple(params), op, isa, **attrs)


VP, CVP = pointer(VOID), pointer(VOID.qualify('const'))
for prefix, vec, mask, isa in (('_mm', V128, USHORT, 'sse2'), ('_mm256', V256, UINT, 'avx2'), ('_mm512', V512, ULLONG, 'avx512')):
    suffix = 'si' + str(vec.bits)
    width = vec.bits // 8
    for unaligned in (False, True):
        add(prefix + ('_loadu_' if unaligned else '_load_') + suffix, vec, [CVP], 'load', isa,
            memory='read', width=width, alignment=1 if unaligned else width)
        add(prefix + ('_storeu_' if unaligned else '_store_') + suffix, VOID, [VP, vec], 'store', isa,
            memory='write', width=width, alignment=1 if unaligned else width)
    add(prefix + '_setzero_' + suffix, vec, [], 'zero', isa)
    add(prefix + '_set1_epi8', vec, [CHAR], 'splat8', isa)
    add(prefix + '_setr_epi8', vec, [CHAR] * width, 'constant8', isa)
    for stem in ('and', 'andnot', 'or', 'xor'):
        add(prefix + '_' + stem + '_' + suffix, vec, [vec, vec], stem, isa)
    for stem in ('cmpeq_epi8', 'cmpgt_epi8', 'min_epu8', 'add_epi8', 'sub_epi8', 'shuffle_epi8'):
        feature = 'ssse3' if vec == V128 and stem == 'shuffle_epi8' else isa
        add(prefix + '_' + stem, vec, [vec, vec], stem, feature)
    if vec != V512:
        add(prefix + '_movemask_epi8', INT, [vec], 'movemask8', isa)
    add(prefix + '_srli_epi16', vec, [vec, INT], 'lshr16', isa, immediates=((1, 0, 255),))
    add(prefix + '_maskz_loadu_epi8', vec, [mask, CVP], 'masked_load', 'avx512', memory='read', width=width, mask=0)
    add(prefix + '_mask_storeu_epi8', VOID, [VP, mask, vec], 'masked_store', 'avx512', memory='write', width=width, mask=1)
    for stem in ('cmpeq_epi8', 'cmpneq_epu8', 'test_epi8', 'testn_epi8'):
        add(prefix + '_' + stem + '_mask', mask, [vec, vec], stem + '_mask', 'avx512')
        add(prefix + '_mask_' + stem + '_mask', mask, [mask, vec, vec], stem + '_masked', 'avx512')
    add(prefix + '_ternarylogic_epi32', vec, [vec, vec, vec, INT], 'ternary32', 'avx512', immediates=((3, 0, 255),))

add('_mm_set_epi64x', V128, [LLONG, LLONG], 'constant64_reversed')
add('_mm_insert_epi64', V128, [V128, LLONG, INT], 'insert64', 'sse4.1', immediates=((2, 0, 1),))
add('_mm_cvtsi128_si32', INT, [V128], 'extract_low32')
add('_mm_cvtsi32_si128', V128, [INT], 'insert_low32')
add('_mm_storel_epi64', VOID, [VP, V128], 'store_low', memory='write', width=8)
add('_mm_storeu_si32', VOID, [VP, V128], 'store_low', memory='write', width=4)
add('_mm256_stream_si256', VOID, [VP, V256], 'stream', 'avx2', memory='write', width=32, alignment=32)
for stem in ('cmpistri', 'cmpistra', 'cmpistrc', 'cmpistrz', 'cmpistrm'):
    add('_mm_' + stem, V128 if stem == 'cmpistrm' else INT, [V128, V128, INT], stem, 'sse42', immediates=((2, 0, 255),))
add('_mm256_broadcastsi128_si256', V256, [V128], 'broadcast128', 'avx2')
add('_mm256_castsi128_si256', V256, [V128], 'widen_undef', 'avx2')
add('_mm256_castsi256_si128', V128, [V256], 'truncate', 'avx2')
add('_mm256_inserti128_si256', V256, [V256, V128, INT], 'insert128', 'avx2', immediates=((2, 0, 1),))
add('_mm256_extracti128_si256', V128, [V256, INT], 'extract128', 'avx2', immediates=((1, 0, 1),))
add('_mm256_testz_si256', INT, [V256, V256], 'testz', 'avx2')
for bits, type_ in ((32, UINT), (64, ULLONG)):
    add('_bzhi_u' + str(bits), type_, [type_, UINT], 'bzhi', 'avx2')
    add('_tzcnt_u' + str(bits), type_, [type_], 'tzcnt', 'avx2')
add('__builtin_ctz', INT, [UINT], 'ctz_nonzero')
add('__builtin_ctzll', INT, [ULLONG], 'ctz_nonzero')
add('__builtin_memcpy', VP, [VP, CVP, ULONG], 'copy', memory='readwrite')
add('__builtin_expect', 'arg0', [None, None], 'expect')
add('__builtin_unpredictable', 'arg0', [None], 'unpredictable')
