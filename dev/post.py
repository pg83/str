#!/usr/bin/env python3
"""Post-compiler passes over the library objects, pure Python: decode, rewrite,
encode, write the objects and the archive.

    post.py [--out DIR] [--archive LIB.a] [--report FILE] [--roundtrip] [--check]
            [--passes a,b,c] [--func-align N] OBJECT.o|ARCHIVE.a ...

Each object is decoded with x86.py and rewritten by opt.py (nothing with
--roundtrip); then cfg.py gives every configuration of dispatch.c its own
copies of the functions it reaches, calling each other directly and laid
out as the configuration asks (the sources call through the table; the code
a host runs does not); layout.py lays the sections out and elfout.py writes
the objects, every unit but dispatch also as a twin with the Skylake JCC
padding and its functions renamed name_jcc, the implementations the
Skylake-SP configuration names. --check compares the result with the
original: with --roundtrip the code bytes, relocations and symbols must be
identical; otherwise it reports the code sizes.

dev/phases.json gives functions, by configuration, the offset of their entry
within its 64-byte line: measured per host class on the workload (the same
code runs up to 20% faster or slower by that offset alone, differently per
function and per host, and no model of 32-byte windows predicted it), so
the numbers are the stand's, and a function whose code changed should be
measured again."""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cfg  # noqa: E402
import elf  # noqa: E402
import elfout  # noqa: E402
import layout  # noqa: E402
import postopt as opt  # noqa: E402
from objfile import Object  # noqa: E402
from units import UNITS, unit_of  # noqa: E402


def optimize_all(path, obj, code, args, trampolines):
    unit = unit_of(path)
    march = UNITS[unit]
    report = []
    if not args.roundtrip:
        for name in code:
            code[name], lines = opt.optimize(obj, code[name], evex='-march=x86-64-v4' in march, section=name, trampolines=trampolines)
            report += lines
    return report


def write(path, out, args, obj, code, clones, overrides, maps, exports):
    """Lay the object out again, with the configuration clones as new sections, and write it;
    for every unit but dispatch also the twin laid out with the JCC padding, its global functions
    renamed name_jcc (the implementations the Skylake-SP configuration names). Returns the paths.
    maps collects {object: {section: {new offset: original offset}}} for --layout-map."""
    jcc = None if args.roundtrip else False
    phases = args.phase_map   # {configuration.function or function: the entry's offset in its 64-byte line}
    layouts = {name: layout.layout(obj, name, ins, args.func_align, jcc_align=jcc,
                                   phase=max([phases.get(f.name, 0) for f in obj.functions(name)] or [0])) for name, ins in code.items()}
    new_sections = []
    for section, ins, symname, jcc_cfg, source in clones:
        layouts[section] = layout.layout(obj, section, ins, args.func_align, jcc_align=jcc_cfg, source=source, phase=phases.get(symname, 0))
        new_sections.append((section, layouts[section], symname))
    remap = layout.resolve_relocs(obj, layouts)
    new = out / Path(path).name
    elfout.write_model(obj, new, layouts, remap, None, new_sections, overrides, defined=exports.setdefault(new.name, []))
    maps[new.name] = {name: {str(v): k for k, v in lay.offsets.items()} for name, lay in layouts.items()}
    out_paths = [new]
    if not args.roundtrip and unit_of(path) != 'dispatch':
        exported = {s.name for s in obj.symbols if s.type == 'STT_FUNC' and s.bind == 'STB_GLOBAL' and s.section != 'UND'}
        data = sorted(s.name for s in obj.symbols if s.type == 'STT_OBJECT' and s.bind == 'STB_GLOBAL' and s.section != 'UND')
        if data:   # a unit is code only: its twin would define the data a second time
            raise ValueError('%s defines data (%s); data belongs to dispatch.c' % (Path(path).name, ', '.join(data)))
        twin_layouts = {name: layout.layout(obj, name, ins, args.func_align, jcc_align=True) for name, ins in code.items()}
        remap = layout.resolve_relocs(obj, twin_layouts)
        twin = out / ('jcc-' + Path(path).name)
        elfout.write_model(obj, twin, twin_layouts, remap, {name: name + '_jcc' for name in exported}, defined=exports.setdefault(twin.name, []))
        out_paths.append(twin)
    return out_paths


def describe(path):
    """Code bytes, relocations and symbols of an object, for comparison."""
    e = elf.read(path)
    code = {s.name: s.data for s in e.sections if s.flags & 4 and s.type == 'SHT_PROGBITS'}
    relocs = {}
    for index, rs in e.relocs.items():
        name = e.sections[index].name
        if name == '.eh_frame':
            continue
        out = []
        for r in rs:
            sym = e.symbols[r.symbol]
            out.append((r.offset, r.type, e.sections[sym.shndx].name if sym.type == 'STT_SECTION' else sym.name, r.addend))
        relocs[name] = sorted(out)
    symbols = sorted((s.name, s.value, s.size, s.type, s.bind, e.section_name(s.shndx)) for s in e.symbols
                     if s.type != 'STT_SECTION' and not (0 < s.shndx < 0xff00 and e.sections[s.shndx].name in ('.eh_frame',)))
    return code, relocs, symbols


def compare(a, b):
    """First difference between two described objects, or None."""
    ca, ra, sa = a
    cb, rb, sb = b
    for name in ca:
        if ca[name] != cb.get(name):
            x, y = ca[name], cb.get(name, b'')
            k = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
            return 'code %s differs at %#x (%d vs %d bytes)' % (name, k, len(x), len(y))
    if ra != rb:
        for name in set(ra) | set(rb):
            if ra.get(name) != rb.get(name):
                d = [(x, y) for x, y in zip(ra.get(name, []), rb.get(name, [])) if x != y][:1]
                return 'relocations of %s differ: %s' % (name, d or (len(ra.get(name, [])), len(rb.get(name, []))))
    if sa != sb:
        d = [(x, y) for x, y in zip(sa, sb) if x != y][:1]
        return 'symbols differ: %s' % (d or (len(sa), len(sb)))
    return None


def inputs_of(paths, out):
    """Objects to process: .o files as given, members of .a archives extracted (our own ar reader)."""
    objects = []
    for p in paths:
        p = Path(p)
        if p.suffix == '.a':
            where = out / 'members' / p.stem
            where.mkdir(parents=True, exist_ok=True)
            for name, data in read_archive(p):
                if name.endswith('.o'):
                    (where / name).write_bytes(data)
                    objects.append(where / name)
        else:
            objects.append(p)
    return objects


def read_archive(path):
    data = path.read_bytes()
    assert data[:8] == b'!<arch>\n', path
    pos = 8
    long_names = b''
    members = []
    while pos + 60 <= len(data):
        name = data[pos:pos + 16].decode().rstrip()
        size = int(data[pos + 48:pos + 58])
        body = data[pos + 60:pos + 60 + size]
        pos += 60 + size + (size % 2)
        if name == '/':
            continue
        if name == '//':
            long_names = body
            continue
        if name.startswith('/'):
            start = int(name[1:])
            name = long_names[start:long_names.index(b'/\n', start)].decode()
        else:
            name = name.rstrip('/')
        members.append((name, body))
    return members


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='post', help='output directory')
    p.add_argument('--archive', help='also collect the new objects into this archive')
    p.add_argument('--report', help='write the pass report here (default: stdout)')
    p.add_argument('--roundtrip', action='store_true', help='no passes: decode, lay out, write')
    p.add_argument('--check', action='store_true', help='compare the result with the original')
    p.add_argument('--passes', help='comma-separated pass list (default: all), see opt.py')
    p.add_argument('--layout-map', help='write {object: {section: {new offset: original offset}}} as JSON: where each instruction came from, for reading perf samples of the result against the sources')
    p.add_argument('--phases', default=str(HERE / 'phases.json'), help='{function or configuration.function: entry offset within its 64-byte line} (default dev/phases.json; absent: 0)')
    p.add_argument('--func-align', default='6', help='align every function to 2^N bytes (default 6: entries on 64-byte lines, where a 16-byte entry at the end of a 32-byte window costs up to a third on Skylake)')
    p.add_argument('inputs', nargs='+')
    args = p.parse_args()
    if args.passes:
        os.environ['POST_PASSES'] = args.passes
    if args.roundtrip:
        os.environ['POST_JCC_ALIGN'] = '0'   # reproduce the object, do not improve its layout
        args.func_align = None
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    failed = False
    objects = []
    # every object decoded first: a pass may look across them (the trampolines of dispatch.c)
    decoded = [(path, Object(path)) for path in inputs_of(args.inputs, out)]
    decoded = [(path, obj, obj.decode_all()) for path, obj in decoded]
    trampolines = {}
    for path, obj, code in decoded:
        trampolines.update(opt.trampolines_of(obj, code))
    args.phase_map = {}
    if not args.roundtrip and args.phases and Path(args.phases).exists():
        args.phase_map = json.load(open(args.phases))
        lines.append('phases: %s, %d entries' % (args.phases, len(args.phase_map)))
    reports = {}
    for path, obj, code in decoded:
        reports[path] = optimize_all(path, obj, code, args, trampolines)
    # the configurations: every function each one reaches, cloned into its unit, calling directly
    clones = [[] for _ in decoded]
    overrides = {}
    dispatch = [obj for _, obj, _ in decoded if any(s.name == 'str_configs' for s in obj.symbols)]
    if not args.roundtrip and dispatch:
        configs = cfg.read_configs(dispatch[0])
        clones, overrides = cfg.specialize(decoded, configs, cfg.initial_table(dispatch[0]))
        for c in configs:
            n = sum(1 for cl in clones for section, *_ in cl if section.startswith('.text.%s.' % c.name))
            lines.append('%s: %d functions, %s' % (c.name, n, 'the JCC padding' if c.jcc else 'no padding'))
    maps, exports = {}, {}
    for (path, obj, code), unit_clones in zip(decoded, clones):
        unit = unit_of(path)
        paths = write(path, out, args, obj, code, unit_clones, overrides, maps, exports)
        objects += paths
        new = paths[0]
        lines += ['%s: %s' % (unit, line) for line in reports[path]]
        if args.check:
            diff = compare(describe(path), describe(new))
            if args.roundtrip:
                lines.append('%s: %s' % (unit, 'identical' if diff is None else diff))
                failed |= diff is not None
            else:
                a, b = describe(path)[0], describe(new)[0]
                lines.append('%s: %d -> %d bytes of code' % (unit, sum(map(len, a.values())), sum(map(len, b.values()))))
    if args.archive:
        elfout.write_archive(args.archive, [(o.name, o.read_bytes()) for o in objects], defined=exports)
    if args.layout_map:
        Path(args.layout_map).write_text(json.dumps(maps))
    text = '\n'.join(lines) + '\n'
    if args.report:
        Path(args.report).write_text(text)
    else:
        sys.stdout.write(text)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
