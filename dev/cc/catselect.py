"""Keep the destination scan and source-copy handoff in their ABI registers."""
from . import machine as M
from .copyselect import CopySelect


class CatSelect(CopySelect):
    def operation(self, op):
        if op.attrs.get('cat_parameter_copy'):
            self.move(self.value(op.results[0]), self.operand(op.args[0], True))
            return
        if op.attrs.get('cat_materialize_status'):
            cc = super().compare(op)
            self.ins('set' + cc, [M.R(self.value(op.results[0]).reg, 8)], ['W'], cc=cc, preserve=True)
            return
        if op.attrs.get('cat_clear_status'):
            dest = M.R(self.value(op.results[0]).reg, 32)
            self.ins('xor', [dest, dest], ['W', 'R'])
            return
        if op.attrs.get('cat_end'):
            a, c = (self.operand(v, True) for v in op.args)
            self.ins('lea', [self.value(op.results[0]), M.Mem(8, base=a.reg, index=c.reg)], ['W', 'A'])
            return
        super().operation(op)

    def compare(self, op, invert=False):
        if op.attrs.get('cat_materialize_status'):
            result = M.R(self.value(op.results[0]).reg, 8)
            self.ins('test', [result, result], ['R', 'R'])
            return 'e' if invert else 'ne'
        if op.attrs.get('cat_page_compare'):
            self.ins('cmp', [M.R(self.operand(op.args[0], True).reg, 32), self.operand(op.args[1])],
                     ['R', 'R'], keep_immediate=True)
            return 'a' if invert else 'be'
        return super().compare(op, invert)

    def run(self):
        fn, pool = super().run()
        src, dst = (self.value(v).reg for v in self.source.avx_cat_parameters)
        compare, shift = (self.value(v).reg for v in self.source.avx_cat_entry)
        source_move = next(i for i in fn.entry.ins if i.mnemonic == 'mov' and src in i.defs)
        destination_move = next(i for i in fn.entry.ins if i.mnemonic == 'mov' and dst in i.defs)
        fn.entry.ins.remove(source_move)
        fn.entry.ins.remove(destination_move)
        compare = next(i for i in fn.entry.ins if compare in i.defs)
        fn.entry.ins.insert(fn.entry.ins.index(compare)+1, source_move)
        shift = next(i for i in fn.entry.ins if i.mnemonic == 'shrx' and shift in i.defs)
        fn.entry.ins.insert(fn.entry.ins.index(shift)+1, destination_move)
        block, pointer, mask = self.source.avx_cat_third
        block, pointer, mask = self.blocks[block], self.value(pointer).reg, self.value(mask).reg
        owner, address = next((b, i) for b in fn.blocks for i in b.ins if i.mnemonic == 'lea' and pointer in i.defs)
        owner.ins.remove(address)
        extract = next(i for i in block.ins if i.mnemonic == 'vpmovmskb' and mask in i.defs)
        block.ins.insert(block.ins.index(extract), address)
        return fn, pool
