<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Independent owner roots for Gemma C4 attention

The checked owner-root launcher reproduces all **32,768 D256 attention outputs
byte for byte** against the existing packed four-stream MMA on retained real
Gemma31 operands. Eager execution, graph replay, fresh queries, restoration and
independent process repeats pass. This is an operator proof; production
selection and whole-model qualification remain open.

The [earlier attention comparison](../gemma-attention-c4/README.md) established
that this packed result also matches the original public backend on the same
native-origin inputs. These are first-local dense31 inputs at query positions
67–70, with four independent owners, one query each, 32 query heads, 16 KV heads,
256 readable cells, F32 Q and F16 K/V/masks. No new model or original run was
needed for this proof. [Aggregate identities](results.json) retain the five
input hashes and complete output hashes.

The distinct manual API accepts eight real K/V descriptors, not a fictitious
contiguous allocation. Each span, stride, alignment, view lineage and alias is
checked before launch. Different owners may not overlap; read-only K-as-V within
an owner is allowed. The replay deliberately uses unequal physical root pitches.
All roots, output and scratch belong to the funded held execution closure.
Pinned upload buffers are reused only after proven completion; an unproven
teardown retains the complete ownership bundle.

Only the 207-line GGML outer controller is ported. Its K/V sequence-base patterns at two tile sites (four expressions) select
the corresponding owner pointers from a 64-byte by-value argument. The pinned tile processing, F32-to-F16 query conversion, MMA,
softmax/reduction and fixups remain included unchanged. The packed compiled
kernel supplies the grid and stream-K partitioning; owner-kernel resource checks
may refuse but cannot select a different grid. Both actual launchers use 48
blocks, occupancy one block per SM and 399,616 bytes of scratch in this proof.
No production selector calls the new launcher.

| Check | Result |
| --- | --- |
| Metadata refusals/admissions, without opening a device | 27/27 pass |
| Locked narrow build and source checks | 7/7 official steps pass |
| Four independent replay processes and nine full-file comparisons | 17/17 official steps pass |
| First/restored/last-paid output, owners and packed | Same 131,072 bytes; SHA `724ee828…` |
| Fresh Q multiplied by −0.5, owners and packed | Same 131,072 bytes; SHA `9dc7d424…` |
| Complete Q/K/V/mask and eight real K/V byte witnesses | Exact before, fresh, restored and after paid work |

Each process pays 32 graph replay waves, full output publication, finite scans
and four attention-coordinate argmax scans. Owner-first/repeat take
0.003130963/0.002994692 s; packed-first/repeat take
0.013006954/0.002807605 s. The packed-first outlier is retained. **No speed claim
is made:** both modes use preuploaded operands, so these timers do not measure
removing whole-model KV packing. Only first, fresh and last paid complete outputs
are retained; intermediate paid outputs are finite-scanned. Input witnesses and
output file writes are outside the timer.

The measured nine-source frame is `ee3838f0…`, helper `3d1420d8…`, and locked
SDK receipt `fbf84c74…`. The isolated diagnostic base is `756ab6e`, not a claim
about current production. Checksum sync omitted `.git`; the receipt therefore
has commit null and origin none. Explicit base/source identities qualify the
build. The declared SDK is `aarch64-c09daba6ac31edee` (NVCC 13.4.92, CUDA toolkit
13.4.2, Clang 22.1.8); the actual sm_121a, fast-math compile command is retained
externally. The port and launcher retain GGML's MIT notice alongside Apache-2.0;
the existing packaged notice covers them. Source-lock scope/obligation metadata
changes, while archive, revision, patches and compile policy stay fixed.

Build1 failed before compilation because the supervised nonlogin shell could
not find mise. Build2 rejected signed-to-unsigned warnings. The measured v2 fixes
only explicit host/fixture casts; build3 and proof1 complete successfully.
Failures, official retirement records and raw outputs remain external. No full
suite was run for this unselected diagnostic.

Reproduce with the manual target `llmp_gemma_attention_owner_replay`: first
`--metadata`, then authenticated retained inputs and a new private output
directory, selecting `owners` or `packed`. Use the installed Spark supervisor
and retain official completion before reading outputs. The fixed input hashes
in `results.json` identify the required Task20 carrier.

Actual D512 arithmetic, other readable widths, model-level copy removal and
paid C4 latency remain owed before a consumer is qualified. Metadata accepts
the closed D512/GQA8 and 16-head cases but supplies no arithmetic evidence for
them. Gemma26 and assistant transfers likewise require actual operand controls.
TensorFold task entry observed `609ca419…`, version 0.6.5; its Gemma recipe is
MLX26 only and provides no31 CUDA recipe.
