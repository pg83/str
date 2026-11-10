"""What ./build proof proves: every configuration's copy of every function,
by its spec, and how (build.py makes one node per entry, dev/proof/node.py
runs it).

A node is <spec>_<config> (strspn_sse2). It proves the function in one
class (every pointer in one proof) unless SPLIT gives it a number of
classes: then each class is a node of its own, <spec>_<config>_<k>, and the
function's node stands for all of them. SAME names the copies whose code is
another's (the post pass gives each configuration its own): their node
checks that, by the prover's fingerprint, and takes that one's proof.
DEPS: the specs a spec's proof takes its callees from (strcat: strlen's,
strcpy's and strcpy_long's contracts, each proved on its own; strspn's and
strcspn's the shuffle scan's, in the units that have it). A function not
every configuration has (truffle_scan) has no entry for the others."""

CONFIGS = ('sse2', 'sse42', 'avx2', 'evex', 'evex512', 'evex512r')

# spec -> the symbol of each configuration's copy
FUNCS = {
    'strlen': {'sse2': 'sse2.strlen_sse2', 'sse42': 'sse42.strlen_sse2', 'avx2': 'avx2.strlen_avx2', 'evex': 'evex.strlen_evex512',
               'evex512': 'evex512.strlen_evex512', 'evex512r': 'evex512r.strlen_evex512'},
    'memchr': {'sse2': 'sse2.memchr_sse2', 'sse42': 'sse42.memchr_sse2', 'avx2': 'avx2.memchr_avx2', 'evex': 'evex.memchr_evex',
               'evex512': 'evex512.memchr_evex512', 'evex512r': 'evex512r.memchr_evex512'},
    'memset': {'sse2': 'sse2.memset_sse2', 'sse42': 'sse42.memset_sse2', 'avx2': 'avx2.memset_avx2', 'evex': 'evex.memset_evex512',
               'evex512': 'evex512.memset_evex512', 'evex512r': 'evex512r.memset_rep'},
    'memcpy': {'sse2': 'sse2.memcpy_sse2', 'sse42': 'sse42.memcpy_sse2', 'avx2': 'avx2.memcpy_avx2', 'evex': 'evex.memcpy_evex512',
               'evex512': 'evex512.memcpy_evex512', 'evex512r': 'evex512r.memcpy_evex512'},
    'memmove': {'sse2': 'sse2.memmove_sse2', 'sse42': 'sse42.memmove_sse2', 'avx2': 'avx2.memmove_avx2', 'evex': 'evex.memmove_evex512',
                'evex512': 'evex512.memmove_evex512', 'evex512r': 'evex512r.memmove_evex512'},
    'memcmp': {'sse2': 'sse2.memcmp_sse2', 'sse42': 'sse42.memcmp_sse2', 'avx2': 'avx2.memcmp_avx2', 'evex': 'evex.memcmp_evex512',
               'evex512': 'evex512.memcmp_evex512', 'evex512r': 'evex512r.memcmp_evex512'},
    'strcmp': {'sse2': 'sse2.strcmp_sse2', 'sse42': 'sse42.strcmp_sse2', 'avx2': 'avx2.strcmp_avx2', 'evex': 'evex.strcmp_evex',
               'evex512': 'evex512.strcmp_evex', 'evex512r': 'evex512r.strcmp_evex'},
    'strcpy': {'sse2': 'sse2.strcpy_sse2', 'sse42': 'sse42.strcpy_sse2', 'avx2': 'avx2.strcpy_avx2', 'evex': 'evex.strcpy_evex',
               'evex512': 'evex512.strcpy_evex', 'evex512r': 'evex512r.strcpy_evex'},
    'strcpy_long': {'sse2': 'sse2.strcpy_long_sse2', 'sse42': 'sse42.strcpy_long_sse2', 'avx2': 'avx2.strcpy_long_avx2',
                    'evex': 'evex.strcpy_long_avx2', 'evex512': 'evex512.strcpy_long_evex512', 'evex512r': 'evex512r.strcpy_long_evex512'},
    'strcat': {'sse2': 'sse2.strcat_sse2', 'sse42': 'sse42.strcat_sse2', 'avx2': 'avx2.strcat_avx2', 'evex': 'evex.strcat_evex',
               'evex512': 'evex512.strcat_evex', 'evex512r': 'evex512r.strcat_evex'},
    'strspn': {'sse2': 'sse2.strspn_sse2', 'sse42': 'sse42.strspn_sse42', 'avx2': 'avx2.strspn_avx2', 'evex': 'evex.strspn_avx2',
               'evex512': 'evex512.strspn_avx2_zen', 'evex512r': 'evex512r.strspn_avx2_zen'},
    'strcspn': {'sse2': 'sse2.strcspn_sse2', 'sse42': 'sse42.strcspn_sse42', 'avx2': 'avx2.strcspn_avx2', 'evex': 'evex.strcspn_avx2',
                'evex512': 'evex512.strcspn_avx2_zen', 'evex512r': 'evex512r.strcspn_avx2_zen'},
}
# the long sets (more than 32 bytes) of the same functions, a spec of their own
FUNCS['strspn_long'] = dict(FUNCS['strspn'])
FUNCS['strcspn_long'] = dict(FUNCS['strcspn'])

# the kernels the public functions hand their long cases to (str_table's), each proved apart under its contract's
# precondition, the caller taking it by that contract
_K = {'strlen_tail': ('strlen_tail_sse2', 'strlen_tail_avx2', 'strlen_tail_evex512'), 'memchr_large': ('memchr_large_sse2', 'memchr_large_avx2'),
      'fill_large': ('fill_large_sse2', 'fill_large_avx2', 'fill_large_evex512'), 'copy_large': ('copy_large_sse2', 'copy_large_avx2'),
      'memcmp_large': ('memcmp_large_sse2', 'memcmp_large_avx2')}
_KCONF = {'sse2': 0, 'sse42': 0, 'avx2': 1, 'evex': 1, 'evex512': -1, 'evex512r': -1}  # which of a kernel's copies each configuration names
for _spec, _fns in _K.items():
    FUNCS[_spec] = {c: '%s.%s' % (c, _fns[i] if i >= 0 else _fns[-1]) for c, i in _KCONF.items()}
FUNCS['copy_large']['evex512'] = 'evex512.copy_large_avx2'
FUNCS['copy_large']['evex512r'] = 'evex512r.copy_large_avx2'
FUNCS['memchr_large']['evex512'] = 'evex512.memchr_large_avx2'
FUNCS['memchr_large']['evex512r'] = 'evex512r.memchr_large_avx2'
FUNCS['memcmp_large']['evex512'] = 'evex512.memcmp_large_avx2'
FUNCS['memcmp_large']['evex512r'] = 'evex512r.memcmp_large_avx2'

# strcmp's kernel past its first blocks, any pointers: strcmp itself, proved apart (strcmp takes it by its contract)
FUNCS['strcmp_tail'] = {'sse2': 'sse2.strcmp_tail_sse2', 'sse42': 'sse42.strcmp_tail_sse2', 'avx2': 'avx2.strcmp_tail_avx2', 'evex': 'evex.strcmp_tail_evex',
                        'evex512': 'evex512.strcmp_tail_evex512', 'evex512r': 'evex512r.strcmp_tail_evex512'}

# the shuffle scan of the avx2 family's strspn and strcspn, proved apart: the tables' build, the scan for any tables
TRUFFLE = ('avx2', 'evex', 'evex512', 'evex512r')
for _fn in ('truffle_scan', 'truffle_scan32'):
    FUNCS[_fn] = {c: '%s.%s' % (c, _fn) for c in TRUFFLE}

# (spec, config) -> classes: those not proved in one class (it did not end
# in hours, or failed where the classes do not: a shuffle by s & 15)
SPLIT = {
    # the head block shuffled by s & 15 near a page end, the blocks after it at 16 - (s & 15): a constant in
    # each class, the spec's offsets the loads' own terms (one class: the solver matched them, an hour and more)
    **{(s, c): 16 for s in ('strspn', 'strcspn') for c in ('sse42', 'avx2', 'evex', 'evex512')},
}

# (spec, config) -> the configuration whose copy has the same code
SAME = {
    **{(s, 'sse42'): 'sse2' for s in ('strlen', 'memchr', 'memset', 'memcpy', 'memmove', 'memcmp', 'strcmp', 'strcmp_tail', 'strcpy', 'strcpy_long',
                                      'strcat')},
    **{(s, 'evex512r'): 'evex512' for s in ('memchr', 'memcpy', 'memmove', 'memcmp', 'strcmp', 'strcmp_tail', 'strspn', 'strspn_long')},
    **{(s, c): 'avx2' for s in ('truffle_scan', 'truffle_scan32') for c in ('evex512', 'evex512r')},
    ('strcspn', 'evex512r'): 'evex512',  # the same past their contracts' callees
    ('strcat', 'evex512r'): 'evex512',
    **{(s, 'sse42'): 'sse2' for s in _K},
    **{(s, c): 'evex' for s in ('memcpy', 'memmove', 'memcmp') for c in ('evex512', 'evex512r')},
    ('memchr_large', 'evex512'): 'avx2', ('memchr_large', 'evex512r'): 'avx2', ('memcmp_large', 'evex512'): 'avx2', ('memcmp_large', 'evex512r'): 'avx2',
    ('fill_large', 'evex'): 'avx2', ('fill_large', 'evex512r'): 'evex512', ('copy_large', 'evex512r'): 'evex512',
}

# (spec, config) -> the set length strspn's and strcspn's short spec ends at, the long one taking the longer
# sets (both specs of a function the same: together every set). sse2's table scan is one code for every set
# from 2 bytes on: proved once by the long spec's invariant of the table, not once per length by the short
# one (its table loop run out 31 times, each a scan loop of its own: hours).
SETMAX = {(s, 'sse2'): 1 for s in ('strspn', 'strcspn', 'strspn_long', 'strcspn_long')}

# derived.c: one copy for every configuration (it calls the entry points, so the table), its spec ->
# its symbol; the slots' specs it takes by their contracts, each slot's implementation in every
# configuration proved by them (strspn's and strcspn's: their short and their long sets)
DERIVED = {s: 'str_' + s for s in ('bcmp', 'bcopy', 'bzero', 'explicit_bzero', 'mempcpy', 'strnlen', 'strncat', 'strlcpy', 'strlcat',
                                   'strpbrk', 'strsep', 'strtok_r', 'strtok', 'strdup', 'strndup')}
DERIVED_DEPS = {'bcmp': ('memcmp',), 'bcopy': ('memmove',), 'bzero': ('memset',), 'explicit_bzero': ('memset',), 'mempcpy': ('memcpy',),
                'strnlen': ('memchr',), 'strncat': ('strlen', 'memchr', 'memcpy'), 'strlcpy': ('strlen', 'memcpy'),
                'strlcat': ('memchr', 'strlen', 'memcpy'), 'strpbrk': ('strcspn', 'strcspn_long'), 'strsep': ('strcspn', 'strcspn_long'),
                'strtok_r': ('strspn', 'strspn_long', 'strcspn', 'strcspn_long'), 'strtok': ('strspn', 'strspn_long', 'strcspn', 'strcspn_long'),
                'strdup': ('strlen', 'memcpy'), 'strndup': ('memchr', 'memcpy')}
DERIVED_CASES = {'strtok_r': 2, 'strtok': 2}  # the classes: s a string, or s null and the saved pointer's
SLOT_SPECS = {'strspn': ('strspn', 'strspn_long'), 'strcspn': ('strcspn', 'strcspn_long')}  # a slot's specs, all of them its implementations'

DEPS = {'strcat': ('strlen', 'strcpy', 'strcpy_long'), 'strcmp': ('strcmp_tail',), 'strcpy': ('strcpy_long',), 'truffle_scan32': ('truffle_scan',),
        'strlen': ('strlen_tail',), 'memchr': ('memchr_large',), 'memset': ('fill_large',), 'memcpy': ('copy_large',), 'memmove': ('copy_large',),
        'memcmp': ('memcmp_large',)}
NO_KERNEL = {('memset', 'evex512r')}  # memset_rep: rep stosb, no kernel
TRUFFLE_DEPS = {'strspn': ('truffle_scan32',), 'strcspn': ('truffle_scan32', 'strlen')}

# the order the ready proofs start in, the longest first (hours to tens of
# hours on the lab, the classes of a split each a part of their function's)
SLOW = ('strspn', 'strcspn', 'strcpy_long', 'strcmp_tail', 'strcmp', 'strcpy', 'memchr', 'memcmp', 'strspn_long', 'strcspn_long', 'strcat',
        'memmove', 'memcpy', 'memset', 'strlen', 'truffle_scan', 'truffle_scan32', 'memchr_large', 'memcmp_large', 'copy_large', 'strlen_tail',
        'fill_large')


def deps(spec, config):
    """The specs whose proofs this configuration's copy of spec takes its callees from."""
    if config in TRUFFLE and spec in TRUFFLE_DEPS:
        return TRUFFLE_DEPS[spec]
    return () if (spec, config) in NO_KERNEL else DEPS.get(spec, ())


def order():
    """The specs, each after those its proofs take contracts from."""
    out, todo = [], list(FUNCS)
    while todo:
        spec = next(x for x in todo if all(d in out for c in configs(x) for d in deps(x, c)))
        out.append(spec)
        todo.remove(spec)
    return out


def configs(spec):
    """The configurations that have the function."""
    return [c for c in CONFIGS if c in FUNCS[spec]]


def priority(spec, config):
    """Higher first: by spec, the wider configurations (more lanes, more
    paths) ahead within one."""
    return 10 * (len(SLOW) - SLOW.index(spec)) + CONFIGS.index(config)


def name(spec, config, k=None):
    return '%s_%s' % (spec, config) if k is None else '%s_%s_%d' % (spec, config, k)
