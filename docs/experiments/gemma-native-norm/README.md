<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Checked native Gemma norm chains

Two explicit, default-off policies now reuse the original GGML norm/rotation
and norm/residual launchers. On the measured dense31 128-row teacher-forced
shape, the complete candidate and its repeat match all 1,024 pinned stock
heads byte for byte. The ordinary31 output and its failed baseline remain
unchanged. This is a bounded optional implementation; full model support,
optimized batching, long context and production default selection remain open.

## Contracts and selection

`gemma_norm.h` declares CPU checks and zero-scratch launch wrappers for named
three-descriptor RMS_NORM/MUL/ROPE and RMS_NORM/MUL/ADD operations. The rotation
contract is F32 full-width D256/D512 NEOX with checked I32 positions and optional
F32 factors. Residual addition is F32 at approved widths 2,816/5,376, with the
actual normalized operand authenticated in either ADD order. Norm/product
intermediates must be nonviews. Metadata, extents, view chains, parameters,
strides and actual-input overlaps are checked before launch. Finite payloads,
positive frequency-factor contents and protected operand residency through
completion/replay remain caller obligations; descriptors do not prove leases.

`DeviceChoices::fuse_norm_rope` and `fuse_norm_add` are separate false-by-default
choices, also exposed as explicit runner diagnostic options. A kept norm or
product, an external reader, or an ineligible chain retains primitives. This
also tightens the existing two-node norm fallback to preserve kept/view-read
norm values when policy combinations or generic fusion request that fallback.
Scoped bounded view preflight rejects cyclic unrelated graph/keep descriptors
before these new policies inspect readers. The generic executor binds the
named kernels through its existing checked operation table.

At the final frontier, GET_ROWS can intervene between norm/MUL and ADD. The
checked four-node selector defers only norm/MUL, executes GET_ROWS as its
original paid/materialized primitive, then emits the fused three-descriptor
step at ADD. It refuses dependencies on elided intermediates and a gather
output overlapping actual norm inputs/weights. All three descriptors extend
actual-input lifetimes through the late step. Placement/rebinding keeps the
same checked selection. No placement ABI, cache-store fusion, intermediate
allocation elision, scheduler, state layout or lifecycle is added. Raw K-as-V
and every required independent consumer remain readable.

## Complete-head numerical evidence

The [baseline protocol](../gemma31-runner/PROTOCOL.md) supplies the same approved
31B source/prepared artifact and complete corpus. Eight 128-row calls use
context 4,096 and F16 KV, with BOS 2 once, 1,024 IDs and 1,023 next-ID scores.
Each retained output is 1 GiB: 1,024 × 262,144 F32 values. All candidate heads
are finite. The candidate-only repeat freeze precedes reading the stock oracle;
its zero repeat movement does not replace or widen ordinary31's calibration.

| Arm | Selected norm/RoPE + norm/ADD per call | Exact stock rows | Strict argmax differences | Maximum raw difference |
| --- | --- | --- | --- | --- |
| Ordinary historical/current control | 0 + 0 | Historical quality failure | 330 in the historical complete comparison | 27.1256761551 historically |
| First coverage candidate | 120 + 119 | 865 / 1,024 | 0 | 0.8426780701 |
| Complete candidate and own repeat | 120 + 120 | 1,024 / 1,024 | 0 | 0 |

The first candidate missed the intervening final residual GET_ROWS pattern.
Its complete acquisition, independent repeat and comparison remain retained;
zero argmax differences did not qualify it or permit score inheritance. The
second candidate includes the checked deferred pattern. Both its full files
and the authenticated retained stock file have SHA-256
`c4b9b73ac7d9c62c62ff1daabc09b36bc716df5a5b124076d0b888bfc42fb0c1`.
Complete byte identity at the same IDs, chunk and scoring alignment therefore
inherits the stock mean NLL **3.5773256064214842** / PPL **35.77772905610885**;
no redundant full NLL scan was run. This does not establish other shapes,
joined requests, context depths or 26B reference quality.

The current ordinary control remains byte exact to historical ordinary SHA
`87d2274ad1420412885cd118a33dc79e614cc743989f96eb0c953e5f8ffc3720`.
Its recorded PPL **41.00513611237161**, **14.61%** above stock, remains a failed
ordinary baseline. Neither that large policy movement nor the first candidate's
remaining differences are treated as benign noise.

## Paid screen

The paid screen retains common corpus IDs, loaded weights, fresh state after
six discarded warm rows, three seeded untimed frontier calls, then 32 completed
forced-prefix calls from common position 8,195. Native pays all 64 128-row
prefill calls and their 63 intermediate full-vocabulary publications; reference
uses the previously fastest screened ubatch 256 among 128 through 8,192,
with batch 8,192. Full vocabulary publication and completion waits remain in
the timers. This is fixed-prefix paid work, not generated-history throughput.

| Arm, acquisition order | Paid 8K prefill (s) | Paid 32 calls (s) |
| --- | --- | --- |
| Reference A1, ubatch 256 | 12.4954 | 4.04232 |
| Ordinary B1 | 13.8454 | 3.31425 |
| Candidate C1 | 13.7583 | 3.3072 |
| Candidate C2 | 13.8033 | 3.31465 |
| Ordinary B2 | 13.8409 | 3.36774 |
| Reference A2, ubatch 256 | 12.4201 | 4.06692 |

Candidate mean latency is 0.4504% lower for prefill and 0.9000% lower for the
32 calls; ordinary decode bookends move 1.6139%. This short screen establishes
no stable speed winner. Reference prefill remains faster. Candidate, ordinary
and reference each repeat both retained heads and all 32 own argmax IDs exactly.
Candidate has three strict argmax differences from reference over those calls;
ordinary has one. Candidate versus reference retained prefill head has maximum
raw delta 1.50895548, TV 0.09096524 and reference-argmax NLL delta 0.19821212;
final head has raw delta 7.25693965, TV 0.0001081621 and NLL delta 0.0001050442.
The competitive reference uses ubatch 256 while native uses 128: the exact
1,024-row quality result does not extend to this paid shape.

Each native process has a conservative charged capacity budget
**31,460,337,344 bytes**: fixed 538,969,088 + weight extents 18,895,339,520 + two
state capacities of 2,390,753,280 + publication 1,048,576 + plan/graph retention
7,243,473,600. The retention term funds 100 possible calls × plan floor
2,194,992 × 33; actual native capture/replay counts are 33/33. Neither managed
charges nor this capacity bound measure process/system physical peak.

Two retained heads and per-call argmax IDs are diagnostic observations only;
selected likelihood changes use the reference head's argmax rather than corpus
PPL. The 8K screen is not a full quality, physical peak-memory or support gate.
No arithmetic default is adopted.

## Controls and provenance

Locked Spark-native focused build/tests passed **24/24**, no skips; the final
integrated suite passed **1,649/1,649** (294 GPU, 38 model tests, no skips)
in 200.74 s on physical Spark-b `spark-56f5`, 2026-10-05 UTC. CPU controls cover parameters, strides,
logical elided dependencies including in-place/view intermediates, physical
aliases, stale/cyclic/overflowed metadata, default choices, combined legacy
flags, keeps/external views, late gather placement and primitive fallback.
GPU controls cover D256/D512 factors, both residual orders/widths, source
preservation, zero scratch, exact capture repeats and fresh inputs/positions.
Norm/learned scaling uses an independent FP64 reference. Complete FP64 rotation
is controlled at zero/small positions; large fresh positions compare with the
checked original primitive RoPE after independent FP64 norm/scale, at the
unchanged 0.0005 bound. This does not claim universal FP64 large-angle agreement.
Earlier CPU phase constructions failed two historical control runs; the final
oracle isolates the pinned GPU phase contract without widening that tolerance
or discarding any model acquisition.

Complete 26B local/global layer controls preserve raw sources, independent
slots, state and capture behavior. Dense31 owns same-shape 1/2/4-request heads,
checkpoint/spill/replay and repeat state controls; these do not establish
scalar-versus-joined reference agreement or optimized batching.

Implementation entered at `a9b5a92`. Measured native source/test manifest 20
files has SHA **`d31bc5c4bdd00621f28a02b65de59d972ea571094430198226909c5b793b3972`**
on `4f9be31`; final integration rebases to `ed274f3`, preserving measured source
bytes except the two additive matcher CMake inventories. SDK is
`aarch64-c09daba6ac31edee`, native CUDA 13.4.92 with Clang host/GCC 16 runtime;
original reference math is its pinned CUDA 13.3 image. Prepared GGML tree
`026f1ac94af98011933e71cee240d0a405ac4220cb208356acf61aa2bcea6207`
is declared and unmodified. Original norm.cu and rope.cu math remain unchanged;
no source-lock or upstream arithmetic patch is introduced. Complete binary,
source, calibration and aggregate identities are in [results.json](results.json).

Task-entry TensorFold refresh **2026-10-05 03:40:04 UTC** observes HEAD
`609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5, still MLX Gemma26
without a dense31/CUDA comparator; [protocol](PROTOCOL.md) retains its pinned
sources. Same-format llama remains b29 at digest
`837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`.

## Reproduction

Use the approved raw/prepared model, matching corpus and dense31 tokenizer IDs
from the baseline protocol. Build locked sources and the existing benchmarks;
run every build, test and scan with installed `spark-job --gpu`, timeout at
most 600 seconds and an official successful wait. Local heavy checks use
`hostlock shared`. Raw vectors and logs stay outside the tree.

`run.sh quality NAME` selects both policies; `ordinary NAME` retains defaults.
Choose a fresh external `JITLLM_GEMMA_NORM_ROOT`; outputs refuse existing paths.
Retain actual source/build/binary identities before acquisition. Acquire
`both-first`, `both-repeat`, then run `analyze.py freeze ROOT IDS IDENTITIES`.
Authenticate that immutable candidate-only freeze SHA before running
`analyze.py compare ROOT QUALITY_ROOT FREEZE_SHA`; the script authenticates
all candidate row bytes, complete stock hash and unchanged ordinary calibration.
`analyze.py selftest` checks signed zero, first-tie argmax and nonfinite refusal.

For the six paid arms use the existing dense31 `reference.sh prefill screen`
with names `norm-reference-a1` and `norm-reference-a2`, ubatch 256; interleave
`run.sh paid-ordinary paid-ordinary1`, `paid paid-both1`, `paid paid-both2`,
`paid-ordinary paid-ordinary2`. Supply the official completed six-step log to
`compare_paid.py ROOT REFERENCE_ROOT LOG`. It authenticates forced IDs,
prefix/completion counts, paid budgets, policy counts, retained-repeat bytes
and all 32 argmax histories, and emits labelled two-head metrics.
