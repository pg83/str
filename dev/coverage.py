#!/usr/bin/env python3
"""Line and branch coverage of the library's C by the test, with gcc and gcov.

    coverage.py [--out DIR] [--cc gcc] [--min 100] [--cppflags ...] [--ldflags ...]

The units are compiled with --coverage at -O1 (every source branch keeps
its own counter; the post pass is not involved, it is checked by the normal
build), the test is linked against them and run, and gcov's JSON gives the
lines and branches each unit executed. Prints a summary per file and the
uncovered lines and branches with their source, writes DIR/summary.txt and
DIR/uncovered.txt, and fails when the total line or branch coverage is below
--min percent. A line commented "uncoverable: <why>" keeps its branches out
of the count (xgetbv behind the OSXSAVE check, the compiler's own branch in
ctz for an argument that is never zero). The test exercises every configuration the host can run and
the decisions for other hosts from cpuid snapshots, so one AVX-512 machine
covers everything; a host without AVX-512 cannot run the EVEX code and
reports less."""
import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from units import UNITS  # noqa: E402

SOURCES = ['sse2.c', 'sse42.c', 'avx2.c', 'evex.c', 'evex512.c', 'dispatch.c', 'derived.c', 'common.h', 'pcmpistri.h']
CFLAGS = ['-O1', '-fno-builtin', '-fno-stack-protector', '-fno-pic', '-fno-pie', '-ffunction-sections', '--coverage',
          '-fprofile-update=atomic', '-Wall', '-Wextra']


def run(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(ROOT / '.build/coverage'))
    p.add_argument('--cc', default=os.environ.get('CC', 'gcc'))
    p.add_argument('--gcov', default=os.environ.get('GCOV', 'gcov'))
    p.add_argument('--nm', default=os.environ.get('NM', 'nm'))
    p.add_argument('--glibc-shims', action='store_true', default=bool(os.environ.get('COVERAGE_GLIBC_SHIMS')),
                   help='the gcov runtime of a glibc toolchain linked against musl: define the glibc-only names it wants')
    p.add_argument('--cppflags', default=os.environ.get('COVERAGE_CPPFLAGS', ''))
    p.add_argument('--ldflags', default=os.environ.get('COVERAGE_LDFLAGS', ''))
    p.add_argument('--min', type=float, default=100.0, help='required percent of lines and of branches')
    p.add_argument('--no-run', action='store_true', help='report from the profiles of an earlier run')
    a = p.parse_args()
    out = Path(a.out).resolve()
    obj = out / 'obj'
    obj.mkdir(parents=True, exist_ok=True)
    cc = shlex.split(a.cc)
    cppflags = shlex.split(a.cppflags)
    ldflags = shlex.split(a.ldflags)
    objects = []
    if not a.no_run:
        for unit, march in UNITS.items():
            o = obj / (unit + '.o')
            run([*cc, *cppflags, '-I', str(ROOT), *CFLAGS, *march, '-c', str(ROOT / (unit + '.c')), '-o', str(o)])
            objects.append(o)
            if unit in ('dispatch', 'derived'):   # no configuration names these: no twins
                continue
            # the *_jcc twins the Skylake-SP configuration names come from the post pass,
            # which has no part here: the unit compiled once more with its global
            # functions renamed; the counters of both copies add up line by line
            names = [line.split()[-1] for line in subprocess.check_output([a.nm, '-g', '--defined-only', str(o)], text=True).splitlines()
                     if len(line.split()) == 3 and line.split()[1] == 'T']
            twin = obj / (unit + '-jcc.o')
            run([*cc, *cppflags, '-I', str(ROOT), *CFLAGS, *march, *['-D%s=%s_jcc' % (n, n) for n in names],
                 '-c', str(ROOT / (unit + '.c')), '-o', str(twin)])
            objects.append(twin)
        test = obj / 'test'
        if a.glibc_shims:
            (obj / 'shims.c').write_text('#include <stdlib.h>\nlong __isoc23_strtol(const char *s, char **e, int b) { return strtol(s, e, b); }\n')
            run([*cc, *cppflags, '-O1', '-c', str(obj / 'shims.c'), '-o', str(obj / 'shims.o')])
            objects.append(obj / 'shims.o')
        run([*cc, *cppflags, '-I', str(ROOT), '-O1', '-fno-pic', '-fno-pie', '-Wall', '-c', str(ROOT / 'tst/test.c'), '-o', str(obj / 'test.o')])
        run([*cc, '-no-pie', '-o', str(test), str(obj / 'test.o'), *map(str, objects), '--coverage', *ldflags])
        for f in obj.glob('*.gcda'):
            f.unlink()
        run([str(test)], stdout=open(out / 'test.txt', 'w'))
    # gcov, JSON: one record per source file the object includes
    lines = defaultdict(dict)      # file -> line -> count
    branches = defaultdict(dict)   # file -> (line, index) -> count
    for o in sorted(obj.glob('*.o')):
        if not o.with_suffix('.gcda').exists():   # the test, the shims
            continue
        run([a.gcov, '--json-format', '--branch-probabilities', '--stdout', '-o', str(obj), str(o)], stdout=open(o.with_suffix('.json'), 'w'), stderr=subprocess.DEVNULL)
        data = json.load(open(o.with_suffix('.json')))
        for f in data['files']:
            name = Path(f['file']).name
            if name not in SOURCES:
                continue
            for ln in f['lines']:
                n = ln['line_number']
                lines[name][n] = lines[name].get(n, 0) + ln['count']
                for i, br in enumerate(ln.get('branches', [])):
                    key = (n, i)
                    branches[name][key] = branches[name].get(key, 0) + br['count']
    src = {name: (ROOT / name).read_text().split('\n') for name in SOURCES if (ROOT / name).exists()}
    # a line marked "uncoverable:" with its reason is left out of the branch count: a
    # branch the hardware of the test host cannot take, or the compiler's own
    excluded = {name: {n + 1 for n, line in enumerate(text) if 'uncoverable:' in line} for name, text in src.items()}
    for name in branches:
        for key in [k for k in branches[name] if k[0] in excluded.get(name, ())]:
            del branches[name][key]
    summary = []
    uncovered = []
    tl = tc = tb = tbc = 0
    for name in SOURCES:
        if name not in lines:
            continue
        L = lines[name]
        Bm = branches[name]
        lc = sum(1 for v in L.values() if v > 0)
        bc = sum(1 for v in Bm.values() if v > 0)
        tl += len(L); tc += lc; tb += len(Bm); tbc += bc
        summary.append('%-12s lines %4d/%-4d %6.2f%%   branches %4d/%-4d %6.2f%%' % (
            name, lc, len(L), 100.0 * lc / max(len(L), 1), bc, len(Bm), 100.0 * bc / max(len(Bm), 1)))
        for n in sorted(k for k, v in L.items() if v == 0):
            uncovered.append('%s:%d: line never run: %s' % (name, n, src[name][n - 1].strip()))
        for (n, i) in sorted(k for k, v in Bm.items() if v == 0):
            if L.get(n, 0) > 0:
                uncovered.append('%s:%d: branch %d never taken: %s' % (name, n, i, src[name][n - 1].strip()))
    total = 'total        lines %4d/%-4d %6.2f%%   branches %4d/%-4d %6.2f%%' % (
        tc, tl, 100.0 * tc / max(tl, 1), tbc, tb, 100.0 * tbc / max(tb, 1))
    summary.append(total)
    (out / 'summary.txt').write_text('\n'.join(summary) + '\n')
    (out / 'uncovered.txt').write_text('\n'.join(uncovered) + '\n')
    print('\n'.join(summary))
    if uncovered:
        print('\nuncovered (%d):' % len(uncovered))
        print('\n'.join(uncovered[:200]))
    ok = 100.0 * tc / max(tl, 1) >= a.min and 100.0 * tbc / max(tb, 1) >= a.min
    print('\n%s: %.2f%% of lines and %.2f%% of branches, %s %.0f%%' % ('OK' if ok else 'FAIL', 100.0 * tc / max(tl, 1), 100.0 * tbc / max(tb, 1), 'at least' if ok else 'below', a.min))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
