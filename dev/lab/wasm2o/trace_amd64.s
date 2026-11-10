#include "textflag.h"

// Measurement only: the wrappers append a record of a call of lib/str's
// memcpy, memmove or memset (trace.go) and jump to the function in
// traceTargets. SI, DI and R8 are free: the code calls lib/str as it calls
// its own functions, which may change them. R15 is the memory's base.
TEXT ·w2oTraceMemcpy(SB), NOSPLIT|NOFRAME, $0-0
	MOVQ	$0, SI
	JMP	w2oTrace<>(SB)

TEXT ·w2oTraceMemmove(SB), NOSPLIT|NOFRAME, $0-0
	MOVQ	$1, SI
	JMP	w2oTrace<>(SB)

TEXT ·w2oTraceMemset(SB), NOSPLIT|NOFRAME, $0-0
	MOVQ	$2, SI
	JMP	w2oTrace<>(SB)

TEXT w2oTrace<>(SB), NOSPLIT|NOFRAME, $0-0
	MOVQ	·traceNext(SB), DI
	LEAQ	32(DI), R8
	CMPQ	R8, ·traceEnd(SB)
	JHI	full
	MOVQ	R8, ·traceNext(SB)
	MOVL	SI, 0(DI)
	MOVL	DX, 4(DI)
	MOVQ	AX, R8
	SUBQ	R15, R8
	MOVQ	R8, 8(DI)
	MOVQ	CX, R8
	CMPQ	SI, $2
	JEQ	isset
	SUBQ	R15, R8
isset:
	MOVQ	R8, 16(DI)
	MOVQ	(SP), R8
	MOVQ	R8, 24(DI)
full:
	LEAQ	·traceTargets(SB), DI
	MOVQ	(DI)(SI*8), DI
	JMP	DI

TEXT ·traceStubs(SB), NOSPLIT, $0-24
	LEAQ	·w2oTraceMemcpy(SB), AX
	MOVQ	AX, memcpy+0(FP)
	LEAQ	·w2oTraceMemmove(SB), AX
	MOVQ	AX, memmove+8(FP)
	LEAQ	·w2oTraceMemset(SB), AX
	MOVQ	AX, memset+16(FP)
	RET
