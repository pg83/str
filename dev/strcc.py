#!/usr/bin/env python3
"""The library's compiler (dev/opt): the C dialect to the objects and the archive.

    strcc.py SRC_DIR -o libstr.a [--out DIR] [--abi wasm2o]   the library: DIR/libstr.o, DIR/report.txt
    strcc.py FILE.c --isa avx2 --dump ir|mir|asm [--function NAME]   one unit's code, for reading
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from opt.source import Error  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source', type=Path)
    ap.add_argument('-o', dest='output', type=Path)
    ap.add_argument('--out', type=Path, help='where the objects and the report go (default: next to the archive, post/)')
    ap.add_argument('-D', dest='defines', action='append', default=[])
    ap.add_argument('--isa', default=None, choices=['sse2', 'sse42', 'avx2', 'evex', 'evex512'])
    ap.add_argument('--abi', default='sysv', choices=['sysv', 'wasm2o'], help="the library's calling convention (dev/opt/mir.py)")
    ap.add_argument('--dump', choices=['ir', 'mir', 'asm'])
    ap.add_argument('--function')
    ap.add_argument('--no-opt', action='store_true', help='dump the IR as lowered, before the passes')
    args = ap.parse_args()
    defines = dict(d.split('=', 1) if '=' in d else (d, '1') for d in args.defines)
    if args.source.is_dir():
        if args.output is None:
            ap.error('a library build needs -o ARCHIVE')
        from opt.library import compile_library
        result = compile_library(args.source, args.output, args.out, defines, abi=args.abi)
        print('%s: %d objects' % (result['archive'], len(result['objects'])))
        return
    from opt import mir
    from opt.module import Module
    from opt.pipeline import optimize
    mir.use_abi(args.abi)
    isa = args.isa or (args.source.stem if args.source.stem in ('sse42', 'avx2', 'evex', 'evex512') else 'sse2')
    module = Module(args.source, isa, defines)
    if not args.no_opt:
        optimize(module)
    functions = module.emitted() if not args.function else [module.functions[args.function]]
    for fn in functions:
        if args.dump == 'mir' or args.dump == 'asm':
            from opt.codegen import compile_function
            from opt.isel import select
            if args.dump == 'mir':
                print(select(fn, isa).dump())
            else:
                mf, code, relocs = compile_function(fn, isa)
                print(mf.dump())
                import x86
                for j in x86.decode(code, 0):
                    print('%6x: %-28s %s' % (j['offset'], j['bytes'], j['gas']))
                for r in relocs:
                    print('    reloc %#x %s %s%+d' % r)
        else:
            print(fn.dump())
        print()


if __name__ == '__main__':
    try:
        main()
    except (Error, ValueError) as e:
        sys.exit(str(e))
