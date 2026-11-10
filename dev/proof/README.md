# Proofs on the machine code

The library's functions are proved correct on the linked machine code
(`./build proof` links `libstr.a` alone into `.build/proof/libstr.elf`; the
post pass's copies `<config>.<function>` are what each configuration runs),
not on the C: what the compiler and the post pass made is what is checked.

```
ix run lib/c bld/compiler bin/z3 bin/bitwuzla -- ./build proof        # the whole chain: every proof below
ix run lib/c bld/compiler bin/z3 bin/bitwuzla -- ./build --dist=root@lab1.mesh:20,root@lab2.mesh:25,root@lab3.mesh:18 -j 100 \
    -Dproof_workers=4 -Dproof_timeout=5400 strspn_avx2           # one function's proof (its 16 classes) on the lab
ix run lib/c bld/compiler bin/z3 bin/bitwuzla -- ./build strspn_avx2_3 strcat_sse2   # a class; a proof and its callees'
ix run lib/c bld/compiler -- ./build proof_elf             # .build/proof/libstr.elf, .build/proof/oracle
ix run bin/z3 bin/bitwuzla -- python3 dev/proof/prove.py --elf .build/proof/libstr.elf \
    --func avx2.memchr_avx2 --spec memchr --induct --align 49 [--verbose]
dev/proof/run.py --z3 .../bin/z3 --bitwuzla .../bin/bitwuzla --func avx2.memchr_avx2,evex.memchr_evex \
    --spec memchr --induct                                 # every alignment class, spread over the lab
dev/proof/run.py ... --func memchr:sse2.memchr_sse2,strlen:sse42.strlen_sse2 --induct --hosts lab1,lab3
python3 dev/proof/replay.py --func avx2.strspn_avx2 --spec strspn --n 2000
ix run bin/z3 -- python3 dev/proof/check_smt.py
```

`./build proof` is the chain (dev/proof/targets.py): a node per configuration's
copy of each function by its spec, `<spec>_<config>` (strspn_sse2), and per
class where the function is proved in classes, `<spec>_<config>_<k>`; a copy
whose code is another's (by run.py's fingerprint, checked when it runs)
takes that one's proof; strcat's node depends on the proofs of the strlen,
strcpy and strcpy_long its contracts stand for, every public function's on
its kernel's (str_table's: strlen_tail, memchr_large, fill_large,
copy_large, memcmp_large, strcmp_tail, strcpy_long), strspn's and strcspn's in the avx2 family on truffle_scan32's
(and strlen's), truffle_scan32's on truffle_scan's, and each checks every
callee is one of them. Each node is
dev/proof/node.py; its output, under
`.build/proof/`, the proof's RESULT line. Under `--dist` the proof nodes run
on the hosts only (the runner's `remote` nodes: with no host connected they
wait, so a host given as `root@labN.mesh`), z3 and bitwuzla sent there
as the proof_solvers node's outputs; -Dproof_workers, -Dproof_timeout,
-Dproof_memory are prove.py's --workers, PROVE_TIMEOUT and PROVE_MEMORY,
-Dproof_node_timeout the seconds a node may take before it fails. A host's
`:N` is how many proofs it runs at once, each of them -Dproof_workers + 1
solvers busy: lab1's 88 cores with 4 workers, `root@lab1.mesh:17`. The
longest proofs start first (targets.priority). Each running proof writes a
line a minute to ~/proof-status/<func>-<spec>-<classes>-<align>.txt on its host
(PROVE_STATUS); dev/proof/report.py lists what .build/proof/ holds proved
and what not yet.

## How

`machine.py` executes the code symbolically: registers, vector lanes, mask
registers and flags are terms (`smt.py`), memory is arrays of bytes, one per
object (a region: the object's base and the memory around it), the stack one
more. A pointer the spec classes is an unknown base aligned to 64 (or to
what the code rounds to) plus its low bits, one alignment class per run:
the page position is then symbolic, every place in a page at once. The
classes are a case split, any number of them covers every pointer; one
class, the low bits symbolic too, is every pointer in one proof: `run.py`
tries that first (strlen sse2 in 8 s, strcmp avx2 in 1102 s, strspn
avx2's long sets in 1079 s), for three hours at most, the classes queued
behind every such try (the slots the tries leave take them; those still
queued when the try proves are dropped). At each
branch z3 decides which ways are feasible; every feasible path is followed.

Loops (`--induct`): the second time a path reaches a loop's head, the
registers that moved become their first value plus K (any multiple of the
step; any number for a loop whose pointers moved by what an inner loop
did), the rest unknown; the base case is proved at the first arrival, and
every way round from the general state either leaves (and is followed to
the return) or comes back to the head in the same form with K + D, the
invariant kept. Invariant candidates (bounds on the frontier, the bytes
checked ahead of it, relations of the registers the loop compares) are
kept or dropped Houdini's way; what another way round changes is widened.
Loops that run down (memmove backwards), loops in a row and nested loops
are handled; a writer's coverage is tracked from its stores. A loop is
generalized once per first arrival: another path that comes round to its
head from the same arrival is in that general state, whose every way round
and every way out are explored (strcmp sse2: 323 s to 24 s).

A loop that builds a table on the stack from strspn's set (a side loop:
its frontier in the set, not in s) has the table's invariant as a
candidate read off its first iteration's stores: for every byte x, its
entry (a byte table's v or 0, a bit table's bit) says whether x is one of
the set's bytes before the frontier. That one has a quantifier; its step is
proved without one (for any x and any truth p for "x one of the first n",
the entry after the iteration says p or x the n-th byte). A byte of the set
is a bit of M, 256 bits defined by one axiom; past the loop (the first load
of s) the invariant, its count L, is the table holding M, written out as 256
facts without a quantifier: the scan's queries go to bitwuzla.

strspn's and strcspn's sets of up to 32 bytes: a byte in the set when it
equals one of them, written as pcmpistri computes it, so that those paths'
terms and the spec's are the same. Past 2048 bytes the avx2 and evex code
tests bytes through truffle tables (a nibble's lane, the high nibble's
bit), terms the spec's are not: every byte matched again, 466 s an
obligation, unknown under load. A one-bit term of the path that reads one
byte b of s and no other, c(b), is taken at a free byte x and proved, once,
to say x in the set (or not) for every x; each such c(b) = member(b) then
goes with the path's checks, instances of that lemma. strspn avx2's first
178 obligations: 3591 s to 581 s, none over 46 s. A mutant whose truffle
table is wrong (one byte of its xor constant 0) is refuted in the loop's
step: the lemma does not hold for it, so no fact is added. A set of 17 to
32 bytes the avx2 code tests with pcmpistrm on its two halves of 16, each a
one-bit term of b ("equals a valid byte of this half") that is not the set
alone: their disjunction is, proved the same way; a mutant that ors the
halves' "not in it" masks where the code ands them is refuted there (the
answer not a match). The term layer
also keeps psrlw's bytes their own: (w >> 4) & 15 of a word's low byte no
longer reads its high byte.

The lemma is asked once per term: a term that reads anything besides the
byte and the set (a lane of a block shuffled by s & 15) gets none, and the
retry under the path's conditions on the set takes those alone (it took
every condition that did not read s, a new key on every path: 6000 lemma
queries in sse42 strspn's first ten minutes, 200 now). A lane's term with
a lemma is also what a check's first try replaces by the uninterpreted
predicate (member_of, as the spec's member(b) is): the 16 x 16 byte
compares of a pcmpistri block are not bit-blasted where the facts settle
them. strcspn's stop, member(b) or the NUL, keeps member(b) a term of its
own for that (or_ flattened its disjuncts into the stop's, and no
abstraction applied: strcspn ran two to three times strspn's time).

The page-safe head load (load16_from: pshufb of the aligned block by
iota + (p & 15), the lanes past 15 made negative) is read as what it is:
lane i is 0 where i + (p & 15) > 15, else the byte at p + i (the index
a | (a > 15 ? 0xff : 0) gives the guard and the byte; the block's address
plus p's low bits folds to p), and pcmpistri takes a lane ite(g, 0, y) as
the NUL where g and y's tests elsewhere, so the lane lemma applies to y.
With every pointer in one class those lanes were the bytes at the block's
address plus a symbolic index, matched against the spec's offsets by the
solver: "no match before the answer" 41 to 74 s on the head paths, now 3.
Both rewrites are checked against the plain forms by the solver for any p.
The blocks after a shuffled head start at 16 - (s & 15), with every pointer
in one class an offset the solver still matches against the spec's (strspn
avx2: more than an hour); targets.SPLIT proves strspn and strcspn of sets up
to 32 bytes in 16 classes in sse42 and the avx2 family, the offsets
constants there: a class of strcspn avx2, 525 paths in 1292 s with six
checking threads, 82% of it the walk's branch queries (a two-way branch
whose one way is dead takes the other with no query).

Past 64 bytes of text (2048 for a set of up to 16) the avx2 family hands
the rest to truffle_scan32: walked as part of strspn, its tables' loop is a
way per count and the scan's loop is explored again on each of them, in
every way the code got there (strspn avx2: more than a hundred arrivals, in
each of 16 classes; not done in days). So it is proved apart, in three
tiers, each a node with the next one's proof behind it. truffle_scan, for
any tables: the tables' 32 bytes unknown, a byte's bit in them what "in the
set" means, members 0 or 1 unknown; one class, one loop by induction: 96 s.
truffle_scan32: its 33 ways, in each the tables it hands on say of a free
byte x exactly "x is one of the first count lanes" (or the NUL, for
members), its other arguments passed on; 5 s (its tables in the red zone
and a jump to truffle_scan: a tail call, rsp at the return address and the
path proving the slot's bytes it, the stores having gone below). And the caller: at its jump
to truffle_scan32, count and the two registers say of a free byte exactly
what the spec's set does, members the spec's, no stop before from, the
callee-saved registers as at the entry and rsp at the return address; the
path ends there. Mutants of each tier (a table's position or bit, the NUL's
entry, a step of the scan, members or the mask's not) are refuted at the
check they break.

strcmp likewise: its first blocks hand the rest to strcmp_tail, whose
contract (any pointers) is strcmp's own; walked as part of strcmp it was
explored again from each way the first blocks leave (strcmp evex: 6.4
hours, evex512's more than 13). Proved apart with the strcmp spec: avx2's
in 72 s, evex's in 102; the first blocks then check at the jump that b
moved as far as a, that no stop lies before, the callee-saved registers
and rsp (avx2: 6.7 s). Mutants that move b another way, skip 16 bytes or
round the next block wrong are refuted there.

The other kernels likewise, each under its contract's precondition, the
first blocks checking it at the jump: strlen_tail (a 16-aligned, s < a, no
NUL in [s, a); the table said s <= a, no caller passes a = s, and the
loop's frontier past s is what the proof has), memchr_large (no match in
[s, A), n > A - s, A s's 16-byte block plus 32: what sse2's first blocks
give, from 18 bytes on; the table said n > 32 and 32 bytes, which they do
not), fill_large (n > 128), copy_large (n > 64, any overlap: memmove's spec;
memmove writes nothing before it), memcmp_large (n > 128, the first 64
bytes equal). The first blocks are then one class of seconds to minutes
(memchr avx2: 64 classes of an hour each, now 202 s), the kernels one of
seconds to minutes. Mutants: a size test moved (memset 112, memmove 48,
memcmp 112), strlen's a off its 16-byte boundary, memchr's n not less
what it scanned.

strcpy hands a string past its first blocks to strcpy_long, whose
contract (no NUL in [s, a), a 16-aligned and 1 to 128 past s: then strcpy
itself) strcat already took: at the jump d, s and a are checked against it
and the stores before are s's bytes; the avx2 family's strcpy, 64 classes
with the kernel in each, is one class of 28 to 53 s. Mutants: a not
16-aligned, d moved, a block's NUL test skipped.

The prover's Python is under 1% of a proof's time, the solvers the rest,
and about 80% of it goes to checks that do not steer the walk: a return's
obligations, an access's bounds, a loop's base and step cases. With
`--workers N` (PROVE_WORKERS) those go to N threads, each with solver
processes of its own, on a copy of the path's state, while the walk goes
on with its branch queries; the first failure stops the proof as before.
8 workers: strcmp sse2 21 to 9.5 s, strlen avx2 355 to 116 s, strcmp avx2
859 to 316 s, strcpy_long sse2 84 to 48 s.

An unrolled loop's exit at its k-th block leaves a span of 16 k + r bytes,
r the index in the block: written as the 16 k offsets with no bound to
compare and r's at most 16 (or 32, 64), not 128 offsets each against a
symbolic length; a long disjunction in an obligation goes to the solver in
parts of 32 disjuncts (strcspn avx2's exit past its x8 loop: 268 s whole,
48 in parts). A loop's pointer and its copy (the exits read r9 = rax) are
one frontier: the generalization took the block counter's step, 8, and a
frontier that lagged the bytes the loop had tested (strcspn avx2's
counterexample in both triage runs, refuted natively on 91800 cases).

A shift by a register that takes a few values on the path (at most four,
the solver asked for each) splits the path by them, the shift then by a
constant: strlen_tail evex512's first block, its NUL mask shifted by a & 63
(0, 16, 32 or 48), each bit then a byte at a constant offset from a, not
one of 64 a symbolic shift may bring (killed at an hour on 4 paths; 140
paths in 961 s with four threads).

strspn and strcspn split the sets at a length, --setmax N (32): the short
spec takes those of up to N bytes, the long one longer ones, both proved
at the same N, so together every set. sse2's code is one table scan for
every set from 2 bytes on: at N = 32 the short spec ran its table loop
out once per length, 31 scan loops of hours; at N = 1 (targets.SETMAX) the
short spec is the 0- and 1-byte paths (6 and 15 s) and the long one the
table scan once, by its table invariant (332 and 479 s).

PROVE_PROFILE=file writes a line per solver query (seconds, answer, the
obligation, [abstract] for a first try, the prover's frames that asked):
where a proof's time goes, before guessing. PROVE_SLOW=s with PROVE_DUMP=dir
writes a query slower than s seconds as a script of its own, and
PROVE_DUMP_ABSTRACT=dir a first try that came out sat (what the
abstraction lost: its model says which byte's predicate was free).

`--prefix 0110` splits one proof by its paths: at a path's first two-way
branches (before it meets a loop's head or a contract) the run takes the
ways the prefix says, 0 the taken one, 1 the other, and a path that meets
a loop, a contract or its end before the prefix is used up goes on only
where the rest of it is zeros. The 2^k prefixes of a length k are a
partition of the paths, so their runs together are the proof (memcmp avx2:
14 paths and 61 obligations whole, the sum of its eight prefixes').

PROVE_CACHE=dir keeps the solvers' answers across runs, dir/answers.txt, by
the query's structure (ids aside, the conditions in any order, the
assumptions among them): a proof cut off by a lost connection, or run again
after a change to the prover that leaves its queries as they were, asks
only what was not asked (truffle_scan again: 146 s to 0.3). `./build`'s
nodes keep them in ~/proof-cache on the host; -Dproof_cache=0 asks every
query, for the last word on a chain.

PROVE_STATUS=dir has a proof write a line a minute to
dir/<func>-<spec>-<classes>-<align>[-p<prefix>].txt: its time, the paths closed, the
obligations, the loops generalized, the ways each walk under way has
still to take, where it is, the checking threads' backlog.

`run.py` proves a configuration's copy of a function once: the same
instructions, branch targets, callees' names and constants read (and the
same tzcnt) are the same machine to the prover. It hands the jobs out as
the hosts' slots free up, the longest specs first.

A load's or a store's bounds are asked first on what the path says of the
addresses alone, its conditions that read memory (the bytes compared
before) left out: fewer hypotheses, so unsat holds for the whole path; the
whole path only when that is not unsat (strcmp avx2 1213 s to 1102 s). A
masked load whose lanes all lie in lane 0's page (a mask up to the page's
end) has lane 0's bounds, one address; else its lanes go all at once, one
at a time when that does not settle in 20 s (strcmp evex past three loops:
a query of 431 s, of 30 minutes on a loaded host, is 0.8 and 1.1 s so).

At a return the spec's obligations are proved for every input the path
admits, quantifier-free: the stop of a scan is a free variable z ("any
stop"), so a property for all stops is one query. "No match in [lo, hi)"
is written out over offsets from lo when the path bounds the span; past a
loop whose blocks start at an address rounded down those are not the
terms of the bytes it loaded, and the solver matched 960 of them against
the path's (strcmp evex: 554 s, unknown under load). Its last way is by
the loads: a match at one of the path's loaded bytes, as their own terms,
or at an offset outside them all (two queries, together the span; 20 s
there); a way that did not do in 130 s gets the full timeout after it.
A load whose bound needs the bytes compared before (a byte at a mask's
first set bit past strcmp evex's loops: 870 s, unknown under load) has
first a lemma proved the same ways: no stop below the page of its last
byte; the bound then follows from the addresses. strcpy's last store, its
end at the NUL a mask found, likewise: no NUL below its last byte. Both
when the full path has not settled the bound in 60 s; the lemma's ways each
for 20 s first, then the one that did it last at that access, then the
loads' with the full timeout (strcpy_long's last store: the loads' way more
than 3000 s, the other less than 120; strcpy_long sse2 84 s to 39).

strspn's set near its page end is loaded shuffled by set & 15; at a loop's
first arrival a register the path proves holds the set's bytes up to its
NUL becomes those bytes there and unknown ones past it (pcmpistri reads a
set up to its NUL), so the loop and its exits meet the spec's terms; with
16 classes of s (its head's shuffle by s & 15 then a constant) a class of
strspn avx2 went from 57 paths in 240 s to 91, the one-class tries having
failed on exactly those exits. A shuffle of a loaded block by a computed
index is one select at the block's offset plus the index. The window from
16 (the set's second half) is taken only where the path proves the set
reaches it: a 1-byte set's needle, its byte 16 times, is "the set's bytes
from 16" for every shorter set, and the needle's lanes became unknown (sse2
strspn refuted "no match in the iteration" on that).

## The specs

| spec | what is proved |
|------|----------------|
| `strlen` | the answer is the offset of the first NUL; every load in the pages of `[s, s + len]` (C11: the string may end at its NUL next to an unmapped page) |
| `memchr` | the first `(u8)c` in `[s, s + n)` or null; loads within the pages up to the match (C11 lets the object end at it) or of the n bytes; nothing read for n = 0 |
| `memset` | every byte of `[d, d + n)` written with `(u8)c`, nothing else written, the answer `d` |
| `memcpy` | every byte of `d` the corresponding byte of `s`, nothing else written, loads in the pages of `[s, s + n)`; the objects apart |
| `memmove` | the same with the objects overlapping in any way: one memory, every byte written the source's as it was before the call |
| `memcmp` | the sign (in fact the value) of the first differing byte pair, 0 without one; loads in the pages of both objects |
| `strcmp` | the difference of the first pair that differs or ends; loads of each string in its pages up to that byte; strcmp_tail by its contract |
| `strcmp_tail` | strcmp's, for the kernel (any pointers) |
| `strlen_tail`, `memchr_large`, `fill_large`, `copy_large`, `memcmp_large` | strlen's, memchr's, memset's, memmove's and memcmp's under the kernel's precondition (str_table's, as above) |
| `strcpy` | the string with its NUL at `d`, nothing else written, the source read up to its NUL's page; the objects apart; strcpy_long by its contract |
| `strcpy_long` | the kernel past strcpy's short path, under its precondition: no NUL in [s, a), a 16-aligned and 1 to 128 past s, 16 at most unless 32-aligned (avx2's kernel copies a NUL in the 16 bytes at a by copy_small, 32 bytes at most; the table's contract said any a to 128, which that kernel does not keep: found by its proof, every caller passes one of the two) |
| `strcat` | `d`'s bytes before its NUL kept, `s` with its NUL copied from there; the callees by their contracts |
| `strspn`, `strcspn` | the length of the prefix in (not in) the set; sets of up to 32 bytes; in the avx2 family with truffle_scan32 and, for an empty set, strlen by their contracts |
| `strspn_long`, `strcspn_long` | the same for sets of more than 32 bytes, any length |
| `truffle_scan` | for any two 16-byte tables and members 0 or 1, from any pointer, any stop past it: the answer `from - s` plus the offset of the first byte the tables say yes of (members) or no of; loads from from's page up to that byte's; nothing written |
| `truffle_scan32` | count at most 32: the tables say of every byte whether it is one of the first count lanes of the two registers (or the NUL, for members), truffle_scan is handed them and from, s, members; its answer returned; nothing read but the frame |

Every function and configuration is also checked for: aligned accesses
(`movaps`, `movdqa`, legacy SSE memory operands) aligned; callee-saved
registers and `rsp` restored; rep movs with no backward overlap; every
store to the stack and every push below the return address (the caller's
frame is not the function's to write: a frame that reached it was
refuted there).

## What the proofs rest on

- the solvers, bitwuzla for every query it reads and z3 for those with an
  array written as a lambda (each query's `unsat`; `unknown` is a failure,
  never a pass); bitwuzla is what makes the slow proofs go (6 to 50 times
  faster than z3 on their queries);
- the instruction semantics in `machine.py`, checked against the hardware by
  `replay.py` (random inputs run natively by the oracle and through the
  machine, results and memory compared);
- the term layer's rewrites (`smt.py`), checked by `check_smt.py` (random
  expressions folded and raw, z3 asked for a difference);
- the specs as written in `prove.py`;
- mutants of the code (patched instructions) are refuted with
  counterexamples: a proof that cannot fail proves nothing.

Limits: lengths are unbounded up to the 2^47 bytes of user space; strspn's
and strcspn's sets any length (up to 32 bytes written out byte by byte,
longer through the mask M and its axiom).
