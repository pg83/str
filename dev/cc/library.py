"""Build the library from C to an archive entirely in memory until emission."""
import json
from pathlib import Path
from types import SimpleNamespace

import cfg
import elfout
import opt as machine_opt
import post
import layout
from units import UNITS

from .abi import trampolines
from .emit import object_model
from .elfmodel import compatibility_model
from .front import Parser
from .lower import lower_unit
from .opt import emitted_functions, optimize
from .source import Preprocessor
from .late import conditional_tails


def compile_library(source, archive, out=None, defines=None):
    source, archive = Path(source).resolve(), Path(archive).resolve()
    out = Path(out).resolve() if out is not None else archive.parent / 'post'
    out.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    parsed = []
    entries, dependencies = {}, set()
    for unit in UNITS:
        pp = Preprocessor(defines)
        ast = Parser(pp.process(source / (unit + '.c')), str(source / (unit + '.c'))).parse()
        ast.dependencies = pp.dependencies
        functions = optimize(lower_unit(ast))
        entries.update(trampolines(functions))
        parsed.append((unit, ast, functions))
        dependencies.update(pp.dependencies)
    decoded = []
    report = []
    for unit, ast, functions in parsed:
        isa = 'sse2' if unit == 'dispatch' else unit
        # Match the reference's frame policy before the shared machine passes:
        # their tail-duplication cost includes epilogues that drop_frame removes.
        selected = emitted_functions(functions)
        obj, code = object_model(selected, source / (unit + '.c'), isa, entries, ast, frame_pointer=True)
        # Keep branch forms chosen with the original epilogues in shared
        # search tails. Frame/vzeroupper removal may later shorten the span.
        for fn in selected:
            if fn.type.base.kind != 'ptr' or not getattr(fn, 'return_joins', ()):
                continue
            name = '.text.' + fn.name
            before = layout.layout(obj, name, code[name], func_align=6, jcc_align=False)
            for ins in code[name]:
                if ins.offset in before.branch_sizes:
                    ins.j['branch_size'] = before.branch_sizes[ins.offset]
        compatibility_model(obj, code)
        path = Path(unit + '.c.o')
        decoded.append((path, obj, code))
        report.append('%s: native C frontend, %d functions' % (unit, len(code)))
    entry_points = {}
    for _, obj, code in decoded:
        entry_points.update(machine_opt.trampolines_of(obj, code))
    args = SimpleNamespace(roundtrip=False, func_align='6', phase_map=json.loads((source / 'dev/phases.json').read_text()))
    for path, obj, code in decoded:
        report.extend('%s: %s' % (path.stem, line) for line in post.optimize_all(path, obj, code, args, entry_points))
        report.extend('%s: %s' % (path.stem, line) for line in conditional_tails(obj, code))
    dispatch = next(obj for path, obj, _ in decoded if path.name == 'dispatch.c.o')
    configs = cfg.read_configs(dispatch)
    clones, overrides = cfg.specialize(decoded, configs, cfg.initial_table(dispatch))
    maps, exports, objects = {}, {}, []
    for (path, obj, code), extra in zip(decoded, clones):
        objects.extend(post.write(path, out, args, obj, code, extra, overrides, maps, exports))
    elfout.write_archive(archive, [(p.name, p.read_bytes()) for p in objects], defined=exports)
    (out / 'report.txt').write_text('\n'.join(report) + '\n')
    return {'archive': archive, 'objects': objects, 'dependencies': dependencies}
