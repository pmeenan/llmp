<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Original-image Gemma assistant oracle seam

The unchanged pinned llama.cpp image executes the approved 26B-A4B Q8_0
assistant against its target's frozen K/V and post-finalnorm feature.
Complete one- and three-step heads and recurrent postprojections repeat
byte-for-byte at C1, serial C2 and genuine physical batch two. Entire borrowed
local/global cache payloads, cell positions/membership and opaque target
sequence-state bytes remain unchanged after every step.

This establishes a bounded same-input oracle seam. It does not qualify the
target's math, assistant quality, speculative verification/rollback, serving
or performance. Native cross-analysis has not been performed. Different
physical shapes are independently repeated; equality between C1 and joined
C2 is not required.

| Original arm | Owners | Physical draft rows/call | Completed head rows, including repeats | Decode calls | Own full-byte repeat | Target state unchanged |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| C1 | 1 | 1 | 8 | 8 | Pass | Pass |
| Serial C2 | 2 | 1 | 16 | 16 | Pass | Pass |
| Genuine batch two | 2 | 2 | 16 | 8 | Pass | Pass |

Each arm independently runs one and three recurrent steps twice. These are
original C-API decode calls and complete published rows, not CUDA launch
counts or a batching speed result. No timings from this untimed client are
performance measurements. [Results](results.json) retains aggregate counts,
identities and the two preserved client failures.

## Exact original seams and physical inputs

The image is `sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`,
source `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`. Two no-weight probes proved
its nextn, `ctx_other`, split-cache getters/RTTI and backend tensor-read exports.
An exact 19-header, 410,815-byte closure links to the original libraries;
no private-layout reinterpretation or rebuilt math is used. Context parameters
are 160 bytes with `ctx_other` at offset 152, and GGML tensors are 336 bytes.
The header and four library hashes are in [provenance](provenance.json).

The target uses serial owner prefills of 64/65 canonical IDs, BOS once,
context 4,096 per owner, ubatch 128, separate streams, F16 K/V and Flash
Attention. `swa_full=false` gives local capacity 1,280 and global capacity
4,096. Original fusion and graph policies remain enabled. Unmasked nextn
exports the post-finalnorm feature at P−1 before head row selection, as in the
[original target graph](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/src/models/gemma4.cpp).
The pending anchor at P has not entered target KV.

Original descriptor and cell getters authenticate local layer 28
`[256,8,256,1]` and global layer 29 `[512,2,256,1]`, F16 pitch and owner
stream offsets. Global V is captured independently from rotated K. Physical
occupancy and membership determine the read width; masks do not prove
initialization. Complete physical padding is checked zero against the
[original allocation clear](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/src/llama-kv-cache.cpp).
Opaque public sequence state is an equality witness only, not a parsed or
translated native format.

The [original assistant graph](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/src/models/gemma4-assistant.cpp)
borrows these target caches. Both token and input feature arrays are supplied;
query P remains constant through recurrence. Each step exports all 262,144
F32 heads and the 2,816-value F32 postprojection. Both contexts are constructed
before target prefill: original shared-cell construction calls `resize`, which
resets occupancy. That ordering follows the original speculative setup.

Only stage-zero target cache/feature/anchor may be supplied before the native
assistant freezes source and endogenous one/three-step own repeats. Later
stock incoming features and anchors are prior stock outputs and remain
withheld together with heads and projections. Subsequent unchanged-source
same-input replay is labelled posthoc, separately from endogenous recurrence.
[Protocol](PROTOCOL.md) defines this calibration boundary and exact raw schema.

## Reproduction and limits

Use the approved target and assistant paths and full hashes in provenance,
plus the canonical 1,024-ID file with SHA
`b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610`.
Its preparation is the [existing tokenizer/corpus recipe](../gemma-quality/PROTOCOL.md).
Populate the external `headers` directory with the 19 paths in provenance
using read-only `git show PIN:path` or pinned primary raw URLs, checking every
listed digest. Copy the reusable client and wrapper into the diagnostic tree,
then run [reference.sh](reference.sh) `build`, `acquire1 NEW_NAME`,
`acquire-serial2 NEW_NAME` or `acquire-batch2 NEW_NAME` through installed
Spark GPU admission. All outputs use new owner-only directories.

Freeze client/source/header/input/model identities before acquisition. After
official successful retirement, run [own_freeze.py](own_freeze.py) with output,
new receipt, work/source paths, the pre-acquisition receipt and official job
directory. It validates exact lengths, finite values, recurrence alignment,
signed-zero byte repeats, full physical padding and equality to pre-run source
identities. Raw arrays, tokens and logs stay external; aggregates and the
reusable reproduction code are retained here.

Known client vector capacity is conservatively bounded by 256 MiB for at most
two owners/three steps, including at most 32 MiB opaque state per owner;
the descriptor-only arena is 1 MiB. Original model/backend allocations are
separate. No whole-process or physical peak is measured, and this is no
jitLLM memory qualification.

Two failed acquisitions remain preserved. The first refused the all-MTP
assistant because the client confused public trunk count zero with total
GGUF block count four. The second detected occupancy reset when the client
created the borrowed context after prefill. Corrected guards independently
check trunk zero, NextN four and metadata block count four; setup now precedes
prefill. Both failures occurred before assistant recurrent output acquisition.

Task-entry TensorFold refresh at 2026-10-05 08:48:40 UTC pinned
`609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5. Its
[Gemma recipe](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/docs/recipes/gemma-4.md)
remains 26B MLX only, without this CUDA assistant comparator.

The original-image build and all three acquisitions plus separate
retirement/authentication jobs passed on Spark-b. No production files changed.
The final four measured client/protocol/wrapper/freezer source hashes match
the pre-acquisition freeze exactly; formatting preceded the measured build.
Local light checks passed 1,410 headers/REUSE files, 386 boundaries,
changed-format/diff checks and five bounded parser controls.
