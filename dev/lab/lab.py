#!/usr/bin/env python3
"""Build kbench, the kernel lab's binary: ./build lab runs this.

    lab.py --deps DIR --libstr libstr.a --work DIR --binary PATH [--canary] [--only GLOBS]

One static binary for every host (kernels the host cannot run are skipped at
run time), linked against Debian's static glibc, with:
  glibc     every IFUNC variant of the eleven primitives (libc.a as it is);
  ours      libstr.a as ./build makes it (the post pass's objects);
  freebsd   lib/libc/amd64/string, every variant (scalar, baseline, x86_64_v2);
  musl      its string files (C, x86_64 memcpy/memmove/memset in asm);
  llvm      LLVM libc's string functions, built for x86-64-v2, -v3 and -v4;
  asmlib    Agner Fog's ELF64 archive, its dispatchers and every variant;
  kl        dev/lab/kernels/*.c, new kernels (canary/*.c with --canary);
  base      with --base, the libstr.a of another revision (base.py): an A/B
            of two revisions of ours in one binary, the stand and the rest of
            the layout the same for both.
Every library's symbols get a prefix (fbsd_, musl_, llvm_v3_, asm_, base_), so they
link side by side. kernels-gen.c, the registry kbench reads, is written from
nm of all of it: a symbol is a kernel by the rules in classify().
"""
import argparse
import io
import json
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PRIMS = ['memcpy', 'memmove', 'memset', 'memcmp', 'strlen', 'strcpy', 'strcat', 'strcmp', 'strspn', 'strcspn', 'memchr']
SLOTS = PRIMS + ['copy_large', 'fill_large', 'memcmp_large', 'strlen_tail', 'strcmp_tail', 'strcpy_long', 'copy_huge', 'fill_huge', 'memchr_large']
ISA_FLAGS = {'sse2': ['-march=x86-64'], 'ssse3': ['-march=x86-64', '-mssse3'], 'sse42': ['-march=x86-64-v2'],
             'avx2': ['-march=x86-64-v3'], 'evex': ['-march=x86-64-v4', '-mprefer-vector-width=256'],
             'evex512': ['-march=x86-64-v4', '-mprefer-vector-width=512']}
ISA_NEEDS = {'sse2': [], 'ssse3': ['F_SSSE3'], 'sse42': ['F_SSE42'], 'avx2': ['F_AVX2'], 'evex': ['F_AVX512'], 'evex512': ['F_AVX512'], 'rep': []}
LLVM_LEVELS = {'v2': ['F_SSE42'], 'v3': ['F_AVX2'], 'v4': ['F_AVX512']}
FREEBSD_UNITS = ['memcpy.S', 'memmove.S', 'memset.S', 'memcmp.S', 'strlen.S', 'strcmp.S', 'stpcpy.S', 'strcat.S', 'strcpy.c', 'strspn.S',
                 'strcspn.S', 'strchrnul.S', 'memchr.S']
FREEBSD_SYMBOLS = ['memcpy', 'memmove', 'memset', 'memcmp', 'strlen', 'strcmp', 'stpcpy', '__stpcpy', 'strcat', 'strcpy', 'strspn', 'strcspn',
                   '__strcspn', 'strchrnul', '__strchrnul', 'memchr', '__memchr']
MUSL_UNITS = ['src/string/' + n for n in ('memcmp.c', 'strlen.c', 'strcpy.c', 'stpcpy.c', 'strcat.c', 'strcmp.c', 'strspn.c', 'strcspn.c',
                                          'strchrnul.c', 'memchr.c', 'x86_64/memcpy.s', 'x86_64/memmove.s', 'x86_64/memset.s')]


def run(cmd, work, name):
    r = subprocess.run([str(x) for x in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (work / (name + '.log')).write_text(' '.join(str(x) for x in cmd) + '\n' + r.stdout)
    if r.returncode:
        raise SystemExit('lab.py: %s failed:\n%s' % (name, r.stdout[-4000:]))
    return r.stdout


def needs_of(variant):
    """Features a variant needs, from the tokens of its name (glibc's and asmlib's conventions)."""
    v = variant.lower()
    out = set()
    if re.search(r'avx512|evex|zmm', v):
        out.add('F_AVX512')
    elif re.search(r'avx2|u256', v):
        out.add('F_AVX2')
    elif 'avx' in v:
        out.add('F_AVX')
    if re.search(r'sse4|sse42|x86_64_v2', v):
        out.add('F_SSE42')
    if 'ssse3' in v:
        out.add('F_SSSE3')
    if 'erms' in v:
        out.add('F_ERMS')
    if 'movbe' in v:
        out.add('F_MOVBE')
    if 'rtm' in v:
        out.add('F_RTM')
    return sorted(out)


def classify(sym, source):
    """(origin, prim, needs) when sym is a kernel; source: which input it comes from."""
    P = '|'.join(PRIMS)
    if source == 'glibc':
        m = re.match(r'^__(%s)_(.+)$' % P, sym)
        if m and not re.search(r'chk|explicit', m.group(2)):  # _rtm too: glibc runs them where the CPU has RTM
            return 'glibc', m.group(1), needs_of(m.group(2))
    elif source == 'asmlib':
        m = re.match(r'^asm_A_(%s)$' % P, sym)
        if m:
            return 'asmlib', m.group(1), []
        m = re.match(r'^asm_(memcpy|memmove|memset|memcmp)(AVX512BW|AVX512F|U256|U|SSE2|SSSE3|AVX2|AVX)$', sym)
        if m:
            return 'asmlib', m.group(1), needs_of(m.group(2))
    elif source == 'freebsd':
        m = re.match(r'^fbsd_(?:__)?(%s)(_baseline|_scalar|_x86_64_v2)?$' % P, sym)
        if m:
            return 'freebsd', m.group(1), needs_of(m.group(2) or '')
    elif source == 'musl':
        m = re.match(r'^musl_(%s)$' % P, sym)
        if m:
            return 'musl', m.group(1), []
    elif source == 'llvm':
        m = re.match(r'^llvm_(v[234])_(%s)$' % P, sym)
        if m:
            return 'llvm', m.group(2), LLVM_LEVELS[m.group(1)]
    return None


def slot_of(sym):
    """(slot, needs, spec) for one of our implementations: plain names call each other
    through str_table, <unit>.<name> are the post pass's copies for one configuration."""
    unit, _, plain = sym.rpartition('.')
    if unit not in ('', 'sse2', 'sse42', 'avx2', 'evex', 'evex512'):
        return None
    for s in sorted(SLOTS, key=len, reverse=True):
        m = re.match(r'^%s_(sse2|sse42|avx2|evex|evex512|rep)(_zen)?(_jcc)?$' % s, plain)
        if m:
            return s, ISA_NEEDS[m.group(1)], 1 if unit else 0
    return None


def nm(path, *flags):
    out = subprocess.run(['llvm-nm', *flags, str(path)], capture_output=True, text=True, check=True).stdout
    return [f[2] for f in (line.split() for line in out.splitlines()) if len(f) == 3]


def isolate(objects, prefix, work, name):
    """Link objects into one relocatable, prefix every symbol: their calls to each other
    resolve to the prefixed names; nothing may stay undefined."""
    combined = work / (name + '-combined.o')
    run(['clang', '--no-default-config', '-fuse-ld=lld', '-nostdlib', '-no-pie', '-Wl,-r', *objects, '-o', combined], work, name + '-link')
    out = work / (name + '.o')
    run(['llvm-objcopy', '--prefix-symbols=' + prefix, combined, out], work, name + '-prefix')
    undefined = nm(out, '--undefined-only')
    defined = set(nm(out, '--defined-only'))
    missing = [u for u in undefined if u not in defined]
    assert not missing, '%s needs %s' % (name, missing)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--deps', required=True, type=Path)
    ap.add_argument('--libstr', required=True, type=Path)
    ap.add_argument('--post-report', type=Path, help="the post pass's report: which configurations it lays out with the JCC padding")
    ap.add_argument('--work', required=True, type=Path)
    ap.add_argument('--binary', required=True, type=Path)
    ap.add_argument('--canary', action='store_true', help='add canary/*.c, kernels broken on purpose: validation must reject every one')
    ap.add_argument('--only', help='comma list of globs under kernels/ to compile instead of all')
    ap.add_argument('--base', type=Path, help="another revision's libstr.a (base.py), linked beside ours with its symbols prefixed base_")
    ap.add_argument('--base-rev-file', type=Path, help='a file with its revision, for the meta')
    args = ap.parse_args()
    work = args.work.resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    # clang's own headers (stddef.h, immintrin.h): the resource directory, or share/include
    # of the toolchain, where ix keeps them (its clang has no default search path)
    resource = subprocess.run(['clang', '-print-resource-dir'], capture_output=True, text=True, check=True).stdout.strip()
    clang_bin = Path(shutil.which('clang')).resolve().parent
    headers = next(d for d in (Path(resource) / 'include', clang_bin.parent / 'share/include') if (d / 'immintrin.h').exists())
    builtin = ['-nostdinc', '-isystem', str(headers)]

    # Debian's glibc and gcc runtime: headers, startup objects, libc.a, libgcc
    sysroot = work / 'sysroot'
    for deb in ('libc6-dev.deb', 'libgcc-dev.deb'):
        members = subprocess.run(['llvm-ar', 't', str(args.deps / deb)], capture_output=True, text=True, check=True).stdout.split()
        data, = [m for m in members if m.startswith('data.tar')]
        blob = subprocess.run(['llvm-ar', 'p', str(args.deps / deb), data], capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            tar.extractall(sysroot, filter='data')
    glibc = sysroot / 'usr/lib/x86_64-linux-gnu'
    gcc = next((sysroot / 'usr/lib/gcc/x86_64-linux-gnu').iterdir())
    glibc_inc = [*builtin, '-isystem', str(sysroot / 'usr/include'),
                 '-isystem', str(sysroot / 'usr/include/x86_64-linux-gnu')]
    base = ['clang', '--no-default-config', '-O3', '-fno-builtin', '-fno-lto', '-fno-stack-protector', '-fno-pic', '-fno-pie']
    inputs = {}  # path -> source tag for the registry

    # FreeBSD
    fsrc = work / 'freebsd'
    with tarfile.open(args.deps / 'freebsd.tar') as tar:
        tar.extractall(fsrc, filter='data')
    fobjs = []
    for unit in FREEBSD_UNITS:
        obj = work / ('fbsd-' + unit + '.o')
        variant = ['-DBENCH_VARIANT=x86_64_v2'] if unit in ('strspn.S', 'strcspn.S') else []
        run([*base, '-march=x86-64', *glibc_inc, '-I', HERE / 'freebsd', *['-D%s=fbsd_%s' % (s, s) for s in FREEBSD_SYMBOLS], *variant,
             '-c', fsrc / 'lib/libc/amd64/string' / unit, '-o', obj], work, 'fbsd-' + unit)
        fobjs.append(obj)
    farchive = work / 'freebsd.a'
    run(['llvm-ar', 'rcs', farchive, *fobjs], work, 'fbsd-ar')
    inputs[farchive] = 'freebsd'

    # musl: its string files, against its own headers
    with tarfile.open(args.deps / 'musl.tar.gz') as tar:
        tar.extractall(work, filter='data')
    msrc = work / 'musl-1.2.5'
    (msrc / 'obj/include/bits').mkdir(parents=True)
    alltypes = run(['sed', '-f', msrc / 'tools/mkalltypes.sed', msrc / 'arch/x86_64/bits/alltypes.h.in', msrc / 'include/alltypes.h.in'], work, 'musl-alltypes')
    (msrc / 'obj/include/bits/alltypes.h').write_text(alltypes)
    minc = [*builtin, '-std=c99', '-ffreestanding', '-D_XOPEN_SOURCE=700', *['-I' + str(msrc / d) for d in ('arch/x86_64', 'arch/generic', 'obj/src/internal',
            'src/include', 'src/internal', 'obj/include', 'include')]]
    mobjs = []
    for unit in MUSL_UNITS:
        obj = work / ('musl-' + unit.replace('/', '-') + '.o')
        run([*base, '-march=x86-64', *minc, '-c', msrc / unit, '-o', obj], work, 'musl-' + Path(unit).name)
        mobjs.append(obj)
    inputs[isolate(mobjs, 'musl_', work, 'musl')] = 'musl'

    # LLVM libc, three ISA levels, the Linux string configuration
    lsrc = work / 'llvm'
    with tarfile.open(args.deps / 'llvm-libc.tar') as tar:
        tar.extractall(lsrc, filter='data')
    config = json.loads((lsrc / 'libc/config/config.json').read_text())
    platform = json.loads((lsrc / 'libc/config/linux/config.json').read_text())
    sconf = {k: v['value'] for k, v in config['string'].items()}
    sconf.update({k: v['value'] for k, v in platform['string'].items()})
    lflags = ['clang++', '--no-default-config', '-O3', '-std=c++17', '-fno-builtin', '-fno-lto', '-fno-exceptions', '-fno-rtti',
              '-fno-lax-vector-conversions', '-fno-unwind-tables', '-fno-asynchronous-unwind-tables', '-fno-stack-protector', '-fno-pic', '-fno-pie',
              '-nostdinc++', *builtin, *['-isystem' + str(msrc / d) for d in ('arch/x86_64', 'arch/generic', 'obj/include', 'include')],
              '-I', str(lsrc / 'libc'), '-DLIBC_NAMESPACE=llvm_libc_lab', '-DLIBC_COPT_PUBLIC_PACKAGING',
              '-DLIBC_COPT_STRING_LENGTH_IMPL=' + sconf['LIBC_CONF_STRING_LENGTH_IMPL'],
              '-DLIBC_COPT_FIND_FIRST_CHARACTER_IMPL=' + sconf['LIBC_CONF_FIND_FIRST_CHARACTER_IMPL']]
    for flag in ('MEMCPY_X86_USE_NTA_STORES', 'MEMSET_X86_USE_SOFTWARE_PREFETCHING', 'USE_MEM_BUILTINS'):
        if sconf['LIBC_CONF_' + flag]:
            lflags.append('-DLIBC_COPT_' + flag)
    for level in LLVM_LEVELS:
        lobjs = []
        for f in PRIMS:
            obj = work / ('llvm-%s-%s.o' % (level, f))
            run([*lflags, '-march=x86-64-' + level, '-c', lsrc / 'libc/src/string' / (f + '.cpp'), '-o', obj], work, 'llvm-%s-%s' % (level, f))
            lobjs.append(obj)
        inputs[isolate(lobjs, 'llvm_%s_' % level, work, 'llvm-' + level)] = 'llvm'

    # asmlib
    with zipfile.ZipFile(args.deps / 'asmlib.zip') as z:
        (work / 'libaelf64.a').write_bytes(z.read('libaelf64.a'))
    run(['llvm-objcopy', '--prefix-symbols=asm_', work / 'libaelf64.a', work / 'asmlib.a'], work, 'asmlib-prefix')
    inputs[work / 'asmlib.a'] = 'asmlib'

    # new kernels
    kl = []
    sources = (sorted({p for g in args.only.split(',') for p in (HERE / 'kernels').glob(g)}) if args.only else sorted((HERE / 'kernels').glob('*.c')))
    for src in sources + (sorted((HERE / 'canary').glob('*.c')) if args.canary else []):
        isa = src.name.split('_')[0]
        assert isa in ISA_FLAGS, '%s: the file name starts with the ISA (%s)' % (src.name, ', '.join(ISA_FLAGS))
        obj = work / ('kl-' + src.stem + '.o')
        run([*base, *ISA_FLAGS[isa], *glibc_inc, '-I', ROOT, '-Wall', '-Wextra', '-c', src, '-o', obj], work, 'kl-' + src.stem)
        kl.append((obj, isa))

    # ours with every function global: the library exports its API only, kbench names the implementations
    ours = work / 'ours.a'
    local = sorted({f[2] for f in (line.split() for line in run(['llvm-nm', '--defined-only', args.libstr], work, 'ours-nm').splitlines())
                    if len(f) == 3 and f[1] == 't'})
    (work / 'ours.globalize').write_text(''.join(n + '\n' for n in local))
    run(['llvm-objcopy', '--globalize-symbols=' + str(work / 'ours.globalize'), args.libstr, ours], work, 'ours-globalize')

    # the baseline: every symbol prefixed base_ but its calls out of the library (memcpy)
    base_lib, base_gen = [], ['const struct str_config *const kl_base_configs = 0;', 'const unsigned kl_base_nconfigs = 0;',
                              'struct str_table *const kl_base_table = 0;', 'const char kl_base_rev[] = "";']
    if args.base:
        a = work / 'base.a'
        run(['llvm-objcopy', '--prefix-symbols=base_', args.base, a], work, 'base-prefix')
        listing = subprocess.run(['llvm-nm', '-S', str(args.base)], capture_output=True, text=True, check=True).stdout.splitlines()
        defined = {f[-1]: f for f in (line.split() for line in listing) if len(f) >= 3 and f[-2] != 'U'}
        external = sorted({f[-1] for f in (line.split() for line in listing) if len(f) == 2 and f[0] == 'U'} - set(defined))
        if external:
            run(['llvm-objcopy', *['--redefine-sym=base_%s=%s' % (s, s) for s in external], a], work, 'base-externals')
        size = {s: int(defined[s][1], 16) for s in ('str_configs', 'str_table')}
        base_lib = [a]
        # the tables must have this revision's layout: kbench reads its configurations as ours
        base_gen = ['extern const struct str_config base_str_configs[];', 'extern struct str_table base_str_table;',
                    '_Static_assert(%d == sizeof(struct str_table), "the base revision\'s struct str_table differs from this one\'s");' % size['str_table'],
                    '_Static_assert(%d %% sizeof(struct str_config) == 0, "the base revision\'s struct str_config differs from this one\'s");'
                    % size['str_configs'],
                    'const struct str_config *const kl_base_configs = base_str_configs;',
                    'const unsigned kl_base_nconfigs = %d / sizeof(struct str_config);' % size['str_configs'],
                    'struct str_table *const kl_base_table = &base_str_table;', 'const char kl_base_rev[] = "%s";' % (args.base_rev_file.read_text().strip() if args.base_rev_file else '')]

    # the registry
    defs, slots = [], []
    for path, source in inputs.items():
        for sym in nm(path, '--defined-only', '-g'):
            c = classify(sym, source)
            if c:
                defs.append((c[0], sym, c[1], c[2]))
    glibc_syms = nm(glibc / 'libc.a', '--defined-only', '-g')
    variants = sorted({s for s in glibc_syms if classify(s, 'glibc')})
    for sym in variants:
        c = classify(sym, 'glibc')
        defs.append((c[0], sym, c[1], c[2]))
    for sym in sorted(set(nm(ours, '--defined-only', '-g'))):
        s = slot_of(sym)
        if s:
            slots.append((s[0], sym, s[1], s[2]))
    for obj, isa in kl:
        for sym in nm(obj, '--defined-only', '-g'):
            m = re.match(r'^kl_(%s)_(%s)_\w+$' % ('|'.join(PRIMS), '|'.join(ISA_FLAGS)), sym)
            if m:
                assert m.group(2) == isa, '%s in a %s file' % (sym, isa)
                defs.append(('kl', sym, m.group(1), ISA_NEEDS[isa]))
    # a variant before its alias: kbench keeps the first name of each address
    defs.sort(key=lambda d: (d[0], PRIMS.index(d[2]), not re.search(r'_(baseline|scalar|x86_64_v2)$', d[1]), d[1]))
    gen = ['/* generated by dev/lab/lab.py */', '#include <stdint.h>', '#include <string.h>', '#include "common.h"', '#include "dev/lab/kl.h"']
    for i, d in enumerate(defs):  # by assembler name: common.h declares some of the C names otherwise
        gen.append('extern char kl_d%d[] __asm__("%s");' % (i, d[1]))
    for i, s in enumerate(slots):
        gen.append('extern char kl_s%d[] __asm__("%s");' % (i, s[1]))
    gen.append('const KlDef kl_defs[] = {')
    gen += ['    {"%s", "%s", P_%s, kl_d%d, %s},' % (o, s, p.upper(), i, '|'.join(n) or '0') for i, (o, s, p, n) in enumerate(defs)]
    gen += ['};', 'const unsigned kl_ndefs = %d;' % len(defs), 'const KlSlot kl_slots[] = {']
    gen += ['    {(int)(offsetof(struct str_table, %s) / sizeof(void *)), "%s", kl_s%d, %s, %d},' % (sl, s, i, '|'.join(n) or '0', spec)
            for i, (sl, s, n, spec) in enumerate(slots)]
    gen += ['};', 'const unsigned kl_nslots = %d;' % len(slots)]
    # the configurations whose copies the post pass lays out with the JCC padding: their
    # <config>.<name> copies are the code of <name>_jcc, not of <name>
    jcc = re.findall(r'^(\w+): \d+ functions, the JCC padding$', args.post_report.read_text(), re.M) if args.post_report else []
    gen += ['const char *const kl_jcc_configs[] = {%s};' % ', '.join(['"%s"' % c for c in jcc] + ['0'])]
    gen += base_gen
    # glibc's choice on this host: the IPLT stub of each primitive jumps through the slot IFUNC filled
    gen.append('const char *glibc_variant(void *p) {')
    gen += ['    if (p == (void *)kl_d%d) return "%s";' % (i, d[1]) for i, d in enumerate(defs) if d[0] == 'glibc']
    gen += ['    return "?";', '}']
    gen.append('void *resolved_glibc(void *entry) {')
    gen += ['    unsigned char *p = entry;', '    int32_t disp;', '    if (p[0] != 0xff || p[1] != 0x25) return entry;',
            '    memcpy(&disp, p + 2, 4);', '    return *(void **)(p + 6 + disp);', '}']
    (work / 'kernels-gen.c').write_text('\n'.join(gen) + '\n')
    counts = {}
    for d in defs:
        counts[d[0]] = counts.get(d[0], 0) + 1
    print('registry: %d kernels (%s), %d slot implementations of ours' % (len(defs), ', '.join('%s %d' % kv for kv in sorted(counts.items())), len(slots)))

    # kbench. The stand (its loops over a trace) is part of every measurement and
    # must keep one shape wherever it lands: no branch cut by or ending at a
    # 32-byte boundary (Skylake's JCC erratum: such a branch leaves the uop
    # cache; moving the stand by 16 bytes changed its own legacy-decoded uops
    # from 14 to 36 a call on Skylake-SP and ours' strcmp by 25%), and every
    # block (the loop's head, the switch's cases: the first case right after
    # the switch's jump cost the empty stand 5.4 legacy-decoded uops a call
    # of the memcpy trace on Broadwell) at the start of a 64-byte line of its own (Broadwell: the empty
    # stand at 9 or 11 cycles a call by the same 16 bytes; Skylake-SP, the
    # branches clear of the boundaries and the blocks 32-aligned: 8 or 12
    # cycles a call on the strcat trace by the half of the line the loop
    # starts in).
    stand = ['-ffunction-sections', '-mllvm', '-x86-align-branch-boundary=32', '-mllvm', '-x86-align-branch=fused+jcc+jmp+call+ret+indirect', '-mllvm',
             '-align-all-blocks=6', '-mllvm', '-align-all-functions=6']
    objs = []
    for src, name in ((HERE / 'kbench.c', 'kbench.o'), (work / 'kernels-gen.c', 'kernels-gen.o')):
        run([*base[:2], '-O2', '-march=x86-64', '-fno-pic', '-fno-pie', '-fno-builtin', '-fno-vectorize', '-fno-slp-vectorize', *stand, *glibc_inc,
             '-I', ROOT, '-Wall', '-Wextra', '-c', src, '-o', work / name], work, name)
        objs.append(work / name)
    # the stand, the empty functions and glibc's variants first in .text, in a fixed
    # order: their addresses then do not move with a change of ours, of the lab's
    # kernels or of the rest of kbench (glibc is not laid out against the JCC
    # erratum, and its time on Skylake-SP follows its place)
    order = work / 'order.txt'
    order.write_text('\n'.join(['trace_run_%d' % k for k in range(4)] + ['mixed_run_%d' % k for k in range(4)] +
                               ['noop_cpy', 'noop_set', 'noop_cmp', 'noop_len', 'noop_scpy', 'noop_scmp', 'noop_spn', 'noop_chr'] + variants) + '\n')
    run(['clang', '--no-default-config', '-fuse-ld=lld', '-nostdlib', '-static', '-no-pie', '-Wl,-z,noexecstack', '-Wl,--symbol-ordering-file=' + str(order),
         '-Wl,--no-warn-symbol-ordering', glibc / 'crt1.o', glibc / 'crti.o',
         gcc / 'crtbeginT.o', *objs, *[o for o, _ in kl], ours, *base_lib, *inputs, '-Wl,--start-group', glibc / 'libc.a', gcc / 'libgcc.a',
         gcc / 'libgcc_eh.a', '-Wl,--end-group', gcc / 'crtend.o', glibc / 'crtn.o', '-o', args.binary], work, 'link')
    print(args.binary)


if __name__ == '__main__':
    main()
