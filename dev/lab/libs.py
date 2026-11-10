#!/usr/bin/env python3
"""Whole libraries on the workload: the platform x library table, glibc = 1.

    dev/lab/libs.py 'TAG*' [--hosts local,lab2,lab3]

Reads the runs .lab/<TAG*>/<host>/ of kbench --libs (run.py --tag ... -- --libs
--no-matrix --prims none). Per run and trace a library's time is its minimum
over the passes less noop's (the stand's own cost per call: the record, the
switch, the call); a library's figure is the minimum over the runs, and the
range is its ratio to glibc in each run on its own. "ours" is the
configuration the host's dispatch installs, "base" the same configuration of
another revision linked into the same binary (base.py), for an A/B of two
revisions with the stand and the rest of the layout the same. A library runs a trace only with
a function for every primitive in it (asmlib has no strcmp, strspn, strcspn
or memchr that passes validation: no mixed trace).
"""
import argparse
import collections
import csv
import glob
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LIBS = ['ours', 'base', 'glibc', 'freebsd', 'llvm', 'musl', 'asmlib']
TRACES = ['mixed', 'memcpy', 'memmove', 'memset', 'memcmp', 'strlen', 'strcpy', 'strcat', 'strcmp', 'strspn', 'strcspn', 'memchr', 'strspn_long',
          'strcspn_long']


def host_data(pattern, host):
    runs, meta = [], None
    for d in sorted(glob.glob(str(ROOT / '.lab' / pattern / host))):
        p = Path(d)
        if not (p / 'out.csv.gz').exists():
            continue
        meta = json.loads((p / 'meta.json').read_text())
        ours, base = 'ours/' + meta['ours_config'], 'base/' + meta['ours_config']
        t = {}
        with gzip.open(p / 'out.csv.gz', 'rt') as f:
            for r in csv.DictReader(f):
                if r['prim'] != 'lib':
                    continue
                name = {ours: 'ours', base: 'base', 'glibc/dispatch': 'glibc'}.get(r['kernel'], r['kernel'])
                t[(name, r['scenario'][6:])] = float(r['min_ns'])
        net = {(lib, tr): max(v - t.get(('noop', tr), 0), 0.01) for (lib, tr), v in t.items() if lib != 'noop'}
        runs.append(net)
    return runs, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pattern')
    ap.add_argument('--hosts', default='local,lab2,lab3')
    args = ap.parse_args()
    head, detail = [], []
    for host in args.hosts.split(','):
        runs, meta = host_data(args.pattern, host)
        if not runs:
            continue
        best = collections.defaultdict(lambda: float('inf'))
        for net in runs:
            for k, v in net.items():
                best[k] = min(best[k], v)

        def ratio(lib, tr):
            return best[(lib, tr)] / best[('glibc', tr)] if (lib, tr) in best and ('glibc', tr) in best else None

        def spread(lib, tr):
            rs = [net[(lib, tr)] / net[('glibc', tr)] for net in runs if (lib, tr) in net and ('glibc', tr) in net]
            return (min(rs), max(rs)) if rs else None

        cpu = '%s (%s)' % (host, meta['brand'].strip())
        clock = meta.get('clock_ghz', [0, 0])
        head.append((cpu, meta['ours_config'], clock, len(runs), {lib: (ratio(lib, 'mixed'), spread(lib, 'mixed')) for lib in LIBS},
                     best.get(('glibc', 'mixed'))))
        # the workload's traces, then any other a trace file had (pizdos.py trace: a program's calls)
        traces = [t for t in TRACES if ('glibc', t) in best] + sorted({t for (_, t) in best} - set(TRACES))
        lines = ['-- %s: ours = %s, %d runs, clock %.2f..%.2f GHz; ratio to glibc per trace' % (cpu, meta['ours_config'], len(runs), clock[0], clock[1]),
                 '   %-8s' % '' + ''.join('%13s' % t for t in traces)]
        lines.append('   %-8s' % 'glibc ns' + ''.join('%13s' % ('%.2f' % best[('glibc', t)] if ('glibc', t) in best else '-') for t in traces))
        for lib in LIBS:
            if lib == 'glibc':
                continue
            lines.append('   %-8s' % lib + ''.join('%13s' % ('%.2f' % ratio(lib, t) if ratio(lib, t) else '-') for t in traces))
        detail.append('\n'.join(lines))
    print('the mixed workload, ns per call net of the stand, ratio to glibc (the range: each run on its own)')
    print('%-52s %-9s' % ('platform', 'config') + ''.join('%17s' % l for l in LIBS))
    for cpu, cfg, clock, n, cells, gl in head:
        row = '%-52s %-9s' % (cpu[:52], cfg)
        for lib in LIBS:
            r, s = cells[lib]
            row += '%17s' % (('%.2f (%.2f-%.2f)' % (r, s[0], s[1])) if r and s and lib != 'glibc' else ('1 (%.2f ns)' % gl if lib == 'glibc' and gl else '-'))
        print(row)
    print()
    print('\n\n'.join(detail))


if __name__ == '__main__':
    main()
