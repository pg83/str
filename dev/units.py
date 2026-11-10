"""The translation units of the library and the -march each is compiled with.
build.py and dev/post.py share this; an object is matched to its unit by the
stem of its file name (evex.o, libstr_evex.a, ours-evex.o)."""

UNITS = {
    'sse2': ['-march=x86-64'],
    'sse42': ['-march=x86-64-v2'],
    'avx2': ['-march=x86-64-v3'],
    'evex': ['-march=x86-64-v4', '-mprefer-vector-width=256'],
    'evex512': ['-march=x86-64-v4', '-mprefer-vector-width=512'],
    'dispatch': ['-march=x86-64'],
    'derived': ['-march=x86-64'],
}


def unit_of(path):
    stem = str(path).rsplit('/', 1)[-1].split('.', 1)[0]   # evex.o, evex.c.o, libstr_evex.a, ours-evex.o
    for sep in ('-', '_'):
        stem = stem.rsplit(sep, 1)[-1]
    if stem not in UNITS:
        raise ValueError('%s: cannot tell which unit this is (expected one of %s)' % (path, ', '.join(UNITS)))
    return stem
