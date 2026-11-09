"""The library's compiler: its C dialect to x86-64 objects, as a pipeline of passes.

source, front      preprocessor, parser and type checker: a typed AST
lower, ssa         the AST to the IR (ir.py), locals to SSA values
inline, simplify, cse, cfgsimp, unroll, ifconv, place
                   the IR passes (pipeline.py runs them)
isel, regalloc, frame, layout, asm
                   machine code: instruction selection, registers, the
                   frame, block order, encoding
library            the library: every unit, a copy of the code per
                   configuration of dispatch.c, the objects and the archive
"""
