//go:build linux && amd64

package main

import (
	"os"
	"unsafe"
)

// Measurement only (str's dev/lab/pizdos.py trace): with W2O_STR_TRACE=FILE
// the GOT's str.memcpy, str.memmove and str.memset are wrappers that append a
// record of each call and jump to the function; the records go to FILE at
// exit. A record, 32 bytes: the function (0 memcpy, 1 memmove, 2 memset) in
// a word, the length in a word, the destination's offset in the memory, the
// source's offset (memset: the byte), the return address.
var (
	traceBuf     []byte
	traceNext    uintptr
	traceEnd     uintptr
	traceTargets [3]uintptr
)

var traceNames = []string{"memcpy", "memmove", "memset"}

func traceStubs() (memcpy, memmove, memset uintptr)

func strTraced(sym string) uintptr {
	if os.Getenv("W2O_STR_TRACE") == "" {
		return 0
	}

	if traceBuf == nil {
		traceBuf = make([]byte, 1<<31)
		traceNext = uintptr(unsafe.Pointer(&traceBuf[0]))
		traceEnd = traceNext + uintptr(len(traceBuf))
	}

	var stubs [3]uintptr
	stubs[0], stubs[1], stubs[2] = traceStubs()

	for k, n := range traceNames {
		if sym == symStr+n {
			traceTargets[k] = strFunction(sym)

			return stubs[k]
		}
	}

	return 0
}

func dumpStrTrace() {
	path := os.Getenv("W2O_STR_TRACE")

	if path == "" || traceBuf == nil {
		return
	}

	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)

	if err != nil {
		return
	}

	f.Write(traceBuf[:traceNext-uintptr(unsafe.Pointer(&traceBuf[0]))])
	f.Close()
}
