<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# First global Gemma31 attention over independent roots

All **65,536 D512 attention outputs match packed four-stream MMA byte for byte**
on freshly captured native Gemma31 first-global inputs. Independent processes,
eager execution, graph replay, changed queries and restoration pass. This extends
the [earlier D256 operator proof](../gemma-owner-root-attention/README.md);
production selection and whole-model copy-removal performance remain open.

Four owners have query positions 67–70 and initialized endpoints 68–71. The
captured layer is 5: F32 Q, 32 query heads, four F16 KV heads, GQA8, one query
per owner and 256 readable cells. The capture uses context 256 / max rows 128,
ordinary products, normRoPE and normADD, with normmul explicitly off. All other
experimental policies are off. These are native-origin operands; no new original
engine inputs or outputs were acquired.

The separate capture wrapper snapshots only the first four one-query D512
operations, authenticates `blk.5.q_rope` lineage and each actual root span, and
queues pinned D2H copies on the held target stream. Pinned snapshots and their
metadata charge survive graph replay and teardown; uncertain completion retains
the complete ownership bundle. The adapter checks canonical 4096-byte IDs in all
four arms before output reads, all 16 repeated operand payloads, zero cold KV
padding and the complete padded masks. Capture and control retain identical
finite four-owner model heads: 4,194,304 bytes, SHA `7d673475…`.

The existing owner-root kernel is unchanged. Eight independent K/V addresses
replace packed sequence-base expressions while retaining the pinned MMA tile,
query conversion and reduction/fixup arithmetic. The original packed compiled
kernel supplies the grid and stream-K partitions. Both actual kernels admit two
blocks per SM and use the preserved 96-block grid with 1,585,408 bytes of scratch.
Independent roots deliberately have unequal physical pitches; no contiguous
allocation or shared virtual-address pitch is assumed.

| Check | Result |
| --- | --- |
| Metadata controls, without opening a device | 27/27 pass |
| Four capture/control processes and carrier authentication | 8/8 official steps pass |
| Four replay processes and nine full-file comparisons | 17/17 official steps pass |
| First, restored and last paid complete output | Same 262,144 bytes; SHA `373cc3d7…` |
| Fresh Q multiplied by −0.5 | Same 262,144 bytes; SHA `04c7c2ad…` |
| Complete GPU operand witnesses | Exact before, fresh, restored and after paid work |

Each process pays 32 replay waves, complete output publication, finite scans and
four attention-coordinate argmax scans. Owner-first/repeat take
0.004805109/0.004748838 s; packed-first/repeat take 0.004917349/0.004818166 s.
**These are operator diagnostics, not a copy-removal speed result:** both modes
use preuploaded operands. Upload, packing, input witnesses and output file writes
are outside the timer. First, fresh and last paid full outputs are retained;
intermediate paid outputs are finite-scanned.

The measured eight-source frame is `cd7c09b3…`, capture helper `b6dd3940…`, replay
helper `437c7542…`, and locked SDK receipt `fbf84c74…`. The isolated base is
`ab88b9d`; checksum sync excludes `.git`, so the receipt has commit null/origin
none. This identifies the diagnostic source rather than latest production.
The declared SDK is `aarch64-c09daba6ac31edee` (NVCC 13.4.92, CUDA toolkit 13.4.2,
Clang 22.1.8). Private helpers explicitly load the canonical native SDK cuBLAS
directory. New diagnostic code is Apache-2.0; the already committed launcher
retains its MIT AND Apache-2.0 attribution. Kernel sources and source-lock policy
are unchanged. [Aggregate identities and completion records](results.json)
identify inputs, outputs, sources, binaries and the five official jobs; raw
payloads and logs remain external.

Build1 failed on a missing validator declaration header. Build2 compiled both
targets and passed metadata controls, then stopped because Spark A lacked `rg`.
Separate compiled1 completed the symbol, source and receipt checks with `grep`.
The actual real/wrapped symbol has the third bool parameter, forwarded unchanged.
All checks completed before capture. No full suite was run for this unselected
diagnostic.

Reproduce with `jitllm_gemma_attention_global_capture` using the canonical IDs
and four new private capture/control directories described in
[the protocol](PROTOCOL.md), then `pack_inputs.py`. Run
`jitllm_gemma_attention_global_owner_replay --metadata`, followed by `owners`
and `packed` on the authenticated carrier. Use the installed Spark supervisor
and read outputs only after official completion. TensorFold task entry resolved
`609ca419…` / 0.6.5; its actual Gemma recipe is MLX26 only, with no matching
Gemma31 CUDA recipe.
