"""One translation unit: its AST, its functions in the IR, its static data."""
from pathlib import Path

from .data import emit_data
from .front import Parser
from .lower import lower_function
from .objmodel import Object
from .source import Preprocessor


class Module:
    def __init__(self, path, isa, defines=None):
        self.path, self.isa = Path(path), isa
        self.name = self.path.stem
        pp = Preprocessor(defines)
        self.ast = Parser(pp.process(self.path), str(self.path)).parse()
        self.dependencies = pp.dependencies
        self.strings = {}
        for d in self.ast.declarations:
            d.global_ = True
        self.functions = {}
        for d in self.ast.functions:
            self.functions[d.name] = lower_function(self, d)
        self.data = Object()
        emit_data(self.data, self.ast)
        referenced = {r.symbol for s in self.data.sections.values() for r in s.relocs}
        for fn in self.functions.values():
            fn.attrs['isa'] = isa
            if fn.name in referenced or '.text.' + fn.name in referenced:
                fn.attrs['address_taken'] = True
        for fn in self.functions.values():
            for sym in [v for (t, *k), v in fn.consts.items() if k and k[0] == 'sym']:
                target = self.functions.get(sym.op.attrs['name'])
                if target is not None and any(u.kind != 'call' or u.args[0] is not sym or sym in u.args[1:] for u in sym.uses):
                    target.attrs['address_taken'] = True

    def string(self, data):
        name = self.strings.get(data)
        if name is None:
            name = self.strings[data] = '.L.str.%d' % len(self.strings)
        return name

    def emitted(self):
        """The functions that get code: exported ones, constructors, and the static
        ones these still call or take the address of."""
        wanted = [f for f in self.functions.values()
                  if not f.attrs.get('static') or f.attrs.get('constructor') or f.attrs.get('address_taken')]
        names = {f.name for f in wanted}
        todo = list(wanted)
        while todo:
            fn = todo.pop()
            for (t, *k), v in fn.consts.items():
                if k and k[0] == 'sym' and v.uses:
                    name = v.op.attrs['name']
                    if name in self.functions and name not in names:
                        names.add(name)
                        todo.append(self.functions[name])
        return [f for f in self.functions.values() if f.name in names]
