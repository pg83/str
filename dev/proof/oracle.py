#!/usr/bin/env python3
"""Build the oracle (oracle.c): a table of every function libstr.a defines
by its symbol (the configurations' <config>.<name> copies included, local
in the library: a copy of it with them global), linked with it, static.

    oracle.py --libstr .build/libstr.a --out .build/proof/oracle
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--libstr', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = subprocess.run(['llvm-nm', '--defined-only', args.libstr], capture_output=True, text=True, check=True).stdout
    names = sorted({f[2] for f in (line.split() for line in out.splitlines()) if len(f) == 3 and f[1] in 'Tt'})
    table = ['#include <stddef.h>', 'struct oracle_fn { const char *name; void *fn; };']
    table += ['extern char f%d[] __asm__("%s");' % (i, n) for i, n in enumerate(names)]
    table += ['const struct oracle_fn oracle_fns[] = {'] + ['    {"%s", f%d},' % (n, i) for i, n in enumerate(names)] + ['    {0, 0},', '};']
    with tempfile.TemporaryDirectory() as d:
        gen = Path(d) / 'table.c'
        gen.write_text('\n'.join(table) + '\n')
        # every function global in a copy: the library exports its API only, the table names them all
        (Path(d) / 'globalize').write_text(''.join(n + '\n' for n in names))
        lib = Path(d) / 'libstr.a'
        subprocess.run(['llvm-objcopy', '--globalize-symbols=' + str(Path(d) / 'globalize'), args.libstr, lib], check=True)
        env = lambda k: os.environ.get(k, '').split()  # the toolchain's headers and libraries (ix passes them so)
        subprocess.run(['clang', *env('CPPFLAGS'), '-O1', '-static', HERE / 'oracle.c', gen, lib, *env('LDFLAGS'), '-o', args.out], check=True)


if __name__ == '__main__':
    main()
