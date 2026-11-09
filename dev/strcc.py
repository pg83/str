#!/usr/bin/env python3
"""Compile the library's C dialect with builtin SIMD operations."""
import argparse
import json
from pathlib import Path
import sys

from cc.front import Parser
from cc.lower import lower_unit
from cc.source import Error, Preprocessor
from cc.opt import optimize, emitted_functions
from cc.emit import emit_object
from cc.abi import trampolines


def read(path, defines=None, include_dirs=()):
    pp = Preprocessor(defines, include_dirs)
    unit = Parser(pp.process(path), str(path)).parse()
    unit.dependencies = pp.dependencies
    return unit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('-I', dest='include_dirs', action='append', default=[])
    ap.add_argument('-D', dest='defines', action='append', default=[])
    ap.add_argument('-o', dest='output', type=Path)
    ap.add_argument('--emit', choices=['ast', 'ssa', 'obj'], default='ssa')
    ap.add_argument('--isa', choices=['sse2', 'sse42', 'avx2', 'evex', 'evex512'])
    ap.add_argument('--function', help='emit only this function')
    ap.add_argument('-O', dest='optimize', action='store_true', help='inline and simplify SSA')
    args = ap.parse_args()
    defines = dict(d.split('=', 1) if '=' in d else (d, '1') for d in args.defines)
    if args.source.is_dir():
        if args.emit != 'obj' or args.output is None or args.function:
            ap.error('a library build needs --emit obj -o ARCHIVE and no --function')
        from cc.library import compile_library
        result = compile_library(args.source, args.output, defines=defines)
        print('%s: %d native objects' % (result['archive'], len(result['objects'])))
        return
    unit = read(args.source, defines, args.include_dirs)
    if args.emit == 'ast':
        text = json.dumps({'source': unit.path, 'dependencies': sorted(map(str, unit.dependencies)),
                           'declarations': [{'name': d.name, 'type': str(d.type), 'definition': d.body is not None,
                                             'attributes': d.attrs} for d in unit.declarations]}, indent=2) + '\n'
    else:
        functions = lower_unit(unit)
        if args.optimize:
            optimize(functions)
        selected = [f for f in functions if not args.function or f.name == args.function]
        if args.emit == 'obj' and not args.function:
            selected = emitted_functions(functions)
        if args.function and not selected:
            ap.error('no definition of ' + args.function)
        if args.emit == 'obj':
            dispatch = args.source.resolve().parent / 'dispatch.c'
            entries = trampolines(functions)
            if dispatch.is_file() and dispatch != args.source.resolve():
                entries.update(trampolines(optimize(lower_unit(read(dispatch, defines, args.include_dirs)))))
            isa = args.isa or (args.source.stem if args.source.stem in ('sse42', 'avx2', 'evex', 'evex512') else 'sse2')
            text = emit_object(selected, args.source, isa, entries, unit if not args.function else None)
        else:
            text = '\n\n'.join(f.dump() for f in selected) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(text) if isinstance(text, bytes) else args.output.write_text(text)
    else:
        sys.stdout.buffer.write(text) if isinstance(text, bytes) else sys.stdout.write(text)


if __name__ == '__main__':
    try:
        main()
    except (Error, ValueError) as e:
        sys.exit(str(e))
