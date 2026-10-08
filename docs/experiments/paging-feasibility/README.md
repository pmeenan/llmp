<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M0 paging feasibility — first cut, 2026-09-21

**Historical first-cut report. The bounded [full study](full-study.md) is now
complete (2026-09-22).** The measurements below retain their original scope.
Real Gemma routes show useful decode locality, but global LRU can reread an
expert working set many times during long-prompt prefill. This supports
keeping M6's switching/retention work ahead of M7's expert paging, and makes
prefill scheduling and phase-specific retention explicit follow-up measurements.
It establishes neither a switching speedup nor an end-to-end paging latency.

## What ran

`capture.cc` links to the **unmodified**, digest-pinned llama.cpp reference
image from [reference setup](../reference-setup/README.md). Its evaluation
callback reads `ffn_moe_topk-N`, the actual selected expert indices, after
backend synchronization. No checkpoint code, new engine patch, or host SDK
installation is involved. The callback respects tensor strides, requires
8 distinct expert IDs in 0–127 for every token, and requires all 30 layers
exactly once in each decode call. Capture deliberately synchronizes GPU work;
its elapsed time is **not** a generation-performance measurement.

Input is the frozen [A→B→A](../reference-aba/README.md) Gemma continuation:
18,339 prompt tokens followed by 117 teacher-forced output tokens from the
saved 118-token reference response. Each of the 118 resulting next-token
predictions matched that response. This is one synthetic notebook workload,
not a representative multi-request corpus, a freely sampled new conversation,
or a minutes-to-hours switching session. No B routes are captured here.

Two prefill batch/microbatch sizes, **64 and 512**, each have a traced run and
an untraced control. Decode has one sequence and one token per call.
All four final runs produced identical 118-token prediction files. Capture
and controls repeated during validation; the two route files were byte-for-byte
identical across the initial, corrected, and final captures. This checks top-1
prediction preservation, not bitwise logits or general numerical equivalence.

Configuration: all 31/31 layers offloaded to GPU, 8 CPU threads, 32,768 context,
f16 K/V, Flash Attention on, full SWA, no speculative decoding, no prompt cache
reuse, no warmup, fresh process/context per run. Full prompt prefill is intentional
so both prefill and decode routes are available; this is not the retained-prefix
execution path timed in A→B→A. No concurrent request batching was tested.

Provenance is in `pins.json`: image digest, full source revision, verified
header hashes, model and frozen input SHA-256, extracted token-input identity,
build command, harness/binary hashes, route and prediction identities.
The source archive is pinned there too. Execution host is `spark-c4e2`,
NVIDIA GB10, driver **580.178.04**, kernel **7.0.0-1019-nvidia**.
CUDA libraries remain those of the existing CUDA 13.3 image (cudart 13.3.29-1,
cuBLAS 13.5.1.27-1). PTX JIT was disabled. The standalone reference harness uses
the image's default `g++`, GCC 13.3.0 (upstream libraries were built with
GCC 14.2.0), C++23, explicit `-march=armv8-a`, and warnings as errors;
it does not change D-032's Clang-first runtime toolchain.

## Replay contract and budget

`replay.py` models a **hypothetical prepared layout**, not the raw GGUF's file
ranges or an executed llmpalooza artifact. Each expert's projection/scale closure
is concatenated, then padded to whole 2 MiB extents, isolated from other experts.
Layers 0–28 need two extents per expert; layer 29 needs three. This deliberately
simple layout has 14,353,054,720 payload bytes and **16,374,562,816 physical
bytes (15.25 GiB)**, a 14.1% padding tax. Actual GGML view compatibility and a
layout sharing boundary extents remain M2/artifact-proof work. Expert IDs are
layer-qualified, and shared FFNs and routers belong to the non-expert budget.

Every selected expert in a layer/chunk's token union is leased together.
Replay loads every missing extent of that union, protects all its resident
hits during reclamation, executes conceptually, then releases eligibility.
Global LRU reclaims complete expert closures; all extents of one closure have
the same lifetime in this experiment. There is no substitution or prefetch.
Simultaneously accessed experts use increasing ID as the deterministic LRU
tie-breaker. This is one policy, not an optimal retention bound.

The total simulated budget includes:

| Component | GiB | Basis |
| --- | ---: | --- |
| Non-expert weights, rounded to 2 MiB | 2.402344 | Verified GGUF tensor accounting |
| Full 32,768-cell K/V | 6.875000 | Logged 640 + 6,400 MiB GPU allocations |
| Scratch/output allowance | 1.000000 | Scenario allowance, above logged compute/output below |
| Four I/O destination slots | 0.007812 | Conservative extra allowance; no separate staging copy |
| Headroom | 2.000000 | Scenario allowance, not a validated admission margin |
| **Fixed total** | **12.285156** | Subtracted before expert capacity |

The observed batch-512 compute buffers were 576.91 MiB GPU + 75.02 MiB host;
batch-64 buffers were 72.11 + 9.38 MiB. Both used 1 MiB host output. The same
1 GiB scratch allowance is kept for both sweeps for a matched budget. The
reference also logs a 748 MiB CUDA-host model buffer in addition to its
16,147.43 MiB CUDA model buffer (the latter equals all stored tensor payloads).
The 2 GiB headroom allowance covers this extra reference buffer and leaves
1.27 GiB for other runtime overhead; whether llmpalooza needs that duplicate
buffer is unproved. This is not measured peak physical
occupancy or an admission guarantee. The real capture ran resident without
physical pressure; the budgets below exist only in offline replay.

Cache starts empty, non-expert weights are assumed present, and state stays
resident throughout. Table read bytes include expert payload **and padding**,
not non-expert initial loading or state traffic. Whole-model residency of this
layout/envelope needs **27.535156 GiB**. Below that, whole-model residency is
infeasible under the same assumptions, so there is no valid resident-generation
speedup ratio to claim. At 28 GiB, cold demand loading still first-touches a few
new experts during decode; a fully preloaded control would have zero misses.

## Miss-byte curves and storage-time scenarios

All figures in this table are **offline estimates from captured routes**.
The 12 GiB point is rejected because its fixed envelope alone does not fit.
Prefill/decode bytes cover 18,339/117 tokens respectively. Decode p95 uses
`floor(0.95*(n-1))` over the 117 token totals, including zero-miss tokens.

| Total GiB | Batch | Prefill reads GiB | Decode reads GiB | Decode mean MiB/token | Decode p95 MiB/token | Bulk read ms/token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 512 | 332.156 | 55.049 | 481.79 | 704 | 33.765 |
| 14 | 64 | 1924.939 | 54.758 | 479.25 | 688 | 33.587 |
| 16 | 512 | 332.156 | 7.068 | 61.86 | 180 | 4.335 |
| 16 | 64 | 1924.939 | 6.475 | 56.67 | 172 | 3.971 |
| 18 | 512 | 332.156 | 3.402 | 29.78 | 104 | 2.087 |
| 18 | 64 | 1924.939 | 2.254 | 19.73 | 60 | 1.382 |
| 20 | 512 | 332.156 | 1.703 | 14.91 | 50 | 1.045 |
| 20 | 64 | 182.273 | 0.854 | 7.47 | 24 | 0.524 |
| 22 | 512 | 37.629 | 0.336 | 2.94 | 12 | 0.206 |
| 22 | 64 | 27.074 | 0.207 | 1.81 | 8 | 0.127 |
| 24 | 512 | 13.732 | 0.084 | 0.74 | 4 | 0.052 |
| 24 | 64 | 13.670 | 0.068 | 0.60 | 4 | 0.042 |
| 28 | 512 | 12.949 | 0.045 | 0.39 | 4 | 0.028 |
| 28 | 64 | 12.922 | 0.039 | 0.34 | 4 | 0.024 |

Bulk read time divides padded miss bytes by the earlier **14.962131 GB/s**
[180-second direct host-VMM measurement](../io-path/README.md), with zero
compute/I/O overlap. `aggregates.json` also reports a serial-read scenario
using **177 µs per 2 MiB read**, the prior queue-depth-one p50 upper run endpoint.
These are service-time scenarios, not predicted end-to-end latency bounds:
small scattered dependency groups may miss bulk throughput; queue startup,
filesystem layout, map/registration costs, scheduler work, and concurrent
state writes are omitted. No inference compute timing is added and no overlap
benefit is assumed. Fully hidden I/O would expose zero storage stall, but this
capture supplies no evidence that such overlap is achievable.

Batch 512 averages 77.74 selected experts per layer/chunk (p95 97, maximum 112);
batch 64 averages 56.47 (p95 73, maximum 95). Smaller chunks touch fewer experts
at once but revisit layers many more times. At 14 GiB, their prefill reads
are 332.156 versus 1,924.939 GiB: about 23.84 versus 138.14 seconds of the bulk
service-time scenario, before any compute or per-group overhead. At 24 GiB,
those reads fall to 13.732/13.670 GiB. Decode uses 8 experts per layer/token and
shows a much smaller retained working set on this particular short response.

As an arithmetic reference only, the original 16,947,541,728-byte GGUF divided
by that bandwidth is **1.133 seconds**. That is not a full-swap floor: it omits
state, lifecycle, and compute and uses a different layout. The measured
A→B→A retained-state return remains **18.304 seconds median** under that
experiment's forced-displacement configuration. This one-model cold-prefill
replay is not matched to that switch and cannot claim to beat it.

## Reuse distance and next-layer prediction

`aggregates.json` contains all 30 layers' prefill and decode reuse quantiles
for both captures. Distance counts distinct experts accessed in **strictly
intervening token groups**, excluding both endpoint groups; top-k peers have
equal access time. Cold first uses are counted separately; the histories carry
from prefill into decode. This avoids inventing temporal order among a token's
simultaneous expert selections. It is a locality diagnostic, not the global
byte-LRU cache size directly.

Across layers, median distances range from **0–27** in prefill and **0–20** in
decode; per-layer p95 ranges are **33–77** and **20–48**, respectively, for both
batch sizes. The full per-layer values and cold counts are retained in the
aggregate file, without raw routes or histograms.

A small causal predictor trains only on the first **58 decode tokens** and
is then frozen for the remaining **59**. For layers 1–29 it ranks the next
layer's experts by co-occurrence with the current layer's selected experts,
choosing eight, with deterministic ID ties. Each test has 13,688 actual
selected contributions. Baselines use the training half's eight most frequent
experts per layer, or the immediately preceding token's same-layer selections.

| Prefill batch | Next-layer co-occurrence recall | Static layer recall | Previous-token same-layer recall |
| ---: | ---: | ---: | ---: |
| 512 | 64.79% | 52.51% | 37.73% |
| 64 | 65.01% | 53.32% | 38.57% |

This is held-out recall on one short synthetic continuation, not cross-request
prediction accuracy. Expert indices in different layers have independent
identities; the learned association maps between them, rather than assuming
that equal IDs mean equal experts. Predicted hits may already be resident,
and late/wrong prefetches consume capacity and bandwidth. Neither bandwidth
cost nor lead time was simulated, so these numbers do not establish hidden
misses or justify enabling prefetch.

## Reproduction and retained evidence

Use a new private external output directory on the chosen Spark. Supply the
hash-verified Gemma GGUF and frozen `final-trace.json` from reference-aba;
the latter remains external at the recorded SHA-256. The installed copy used
here is under `/home/pmeenan/.local/share/llmp/aba`. If unavailable, regenerate
with the A→B→A preparation harness and verify its identity before using this
exact experiment; a different trace needs new provenance and results.

Download the upstream source archive for revision
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` from
[the pinned archive](https://github.com/ggml-org/llama.cpp/archive/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4.tar.gz),
verify its `source_archive_sha256` in `pins.json`, and extract it outside Git.
Run from a copy of this harness directory, supplying absolute paths:

```sh
DOCKER='sudo -n docker' python3 run.py /path/to/model.gguf \
  /path/to/final-trace.json /path/to/llama.cpp-b29c606e28a01b1bc8c1351026a0fa6e616bf6c4 \
  /path/to/new-capture-directory
python3 replay.py /path/to/new-capture-directory /path/to/new-aggregate.json
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -p 'test_*.py'
```

Replay may run on the workstation after copying the capture directory. It
verifies route/prediction identities, pinned model/engine/input provenance,
both distinct batch captures and controls, every layer/step, token counts,
expert ranges/uniqueness, and complete trace length before accepting results.
The capture wrapper verifies model and input hashes and all upstream public
headers before building; it binds no service and disables container networking.
Containers are uniquely named and removal is attempted on success, failure,
SIGTERM, and Ctrl-C. A forced process kill or host failure can still require
manual cleanup. Deliberately captured synthetic tokens, predictions, logs,
and routes remain in private external directories; no production prompts are
logged and no service or retention job is installed.

Licenses/categories follow reference-setup: llmpalooza-authored harness/replay are
Apache-2.0; llama.cpp/GGML and their headers are external MIT reference tools;
Gemma is external Apache-2.0 benchmark data. The image's GCC and CUDA/platform
libraries retain their separately recorded tool/runtime terms. No model,
upstream implementation, image, or captured token data is redistributed here.
This does not admit a backend into llmpalooza's implementation dependency profile.

## First-cut handoff and subsequent work

Spark verification: final native harness compiled with warnings as errors;
both GPU captures and both controls passed with PTX JIT disabled and exact
prediction matches. A signed-overflow count input was rejected before model
load by the final Spark executable. All 30 routed layers and all 18,456 processed tokens
validated at each batch size. No experiment containers remain running.
Workstation verification: replay completed both eight-point budget sweeps;
unit checks cover physical padding, atomic leases, LRU/layer identity,
impossible groups/envelopes, full-residency cold misses, route corruption and
truncation, provenance and paired batches, simultaneous-access distances and
ID relabeling, and cleanup success/failure/interruption. No llmpalooza runtime,
physical paging, Spark-b experiment, or numerical-logit comparison ran.

At this first-cut handoff the full plan checkbox remained open. The subsequent
[full study](full-study.md) completes the bounded multi-model, larger-library,
spill/recompute, switching, decode-batch, and prediction investigation. It
records conditional Qwen trajectory estimates after failed numerical
equivalence, and conservative recomputation for unvalidated spill continuations.
It also supersedes any assumption that a successful short Gemma continuation
proves complete saved SWA-window coverage (RE-007). These are still offline
feasibility results, not implemented paging or measured llmpalooza speedups.
The owner subsequently accepted switching/stall criteria in D-036
(2026-09-22); implementation validation remains ahead. The original first-cut
verification and review below are historical records.

## Independent review

A separate agent reviewed the complete change and challenged input bounds,
callback lifetimes/strides, atomic dependency protection, physical accounting,
reuse metrics, provenance, cleanup, and report claims. Initial findings were
fixed: signed count overflow, ID-dependent simultaneous-access distances, and
acceptance of duplicate batch/provenance metadata. No unresolved defects were
found in the final first-cut scope.

On the workstation, the reviewer reran all **17 tests**, checked **1,000
randomized replays** against an independent list-based atomic-LRU calculation,
verified every miss-table cell against the aggregate, and checked locality and
prediction summaries. Final capture/source/binary/input identities match the
pins; all four prediction files match the frozen reference response, and
initial/final route identities agree. Logged GPU, KV, compute, output, and host
model buffers support the reported allocation figures. The checked-in aggregate
matches the final replay output; whitespace checks passed. GPU execution was
performed by the builder on Spark and reviewed from its external evidence,
not independently repeated by the reviewer. Actual paging, admission safety,
multi-model switching, and numerical-logit equivalence remain unvalidated.
