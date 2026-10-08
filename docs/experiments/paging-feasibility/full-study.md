<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M0 paging feasibility — full study

The bounded M0 feasibility study is complete. This report extends the historical [first cut](README.md); it does not describe an
implemented llmpalooza pager.

## Evidence and comparison contract

The study separates three kinds of evidence: measured normal-reference
switching, route captures from a matched instrumentable reference, and offline
storage-service estimates. Estimated seconds exclude kernel execution,
allocation, graph construction, driver overhead, and scheduling. They cannot
be subtracted from measured switching latency to claim a speedup.

The exact Gemma→Ornith→Gemma trace connects replay to the existing
[27-trial measured reference cycle](../reference-aba/README.md). A second
corpus alternates four requests per model: planning, implementation, failure
review, and handoff. Gemma prompts grow from 7,339 to 9,758 tokens and Ornith
from 1,865 to 4,298; every response supplies 768 teacher-forced tokens.
Their declared arrivals span 91 minutes. These are **synthetic arrival times**,
not a 91-minute hardware soak; no retention-expiry policy is evaluated.
The large-model corpus alternates three 512-token responses per model over
a declared 21-minute schedule, with actual prompts growing from 1,699 to
2,790 for DeepSeek and 1,865 to 2,974 for Qwen.

Captures use the same digest-pinned llama.cpp image/revision as the reference
cycle, C++23, the image's GCC 13.3.0 for harnesses, explicit ARMv8-A, f16 state, Flash
Attention, all primary layers on GPU, and no checkpoint code. Gemma/Ornith
run on `spark-c4e2`; DeepSeek runs on `spark-56f5` and Qwen on
`spark-c4e2`, both GB10 with driver
580.178.04. This external reference tooling does not change the Clang-first
runtime toolchain. Model bytes, source headers, inputs, binaries, events,
predictions, and resource-accounting logs are pinned in the evidence records.

### Instrumentation changes the execution configuration

The longer Gemma workload exposed a problem invisible in the short first cut:
the routing callback changed 51/3,072 top-1 predictions with ordinary fusion.
A repeated untraced control matched all 3,072. The pinned CUDA backend can fuse
across the routed-index node; callback synchronization changes graph partitioning.
Both controls and captures therefore disable **CUDA fusion and CUDA graphs**.
See [RE-006](../../rough-edges.md). This validates routing within that matched
configuration, not numerical equivalence to the normal optimized engine. A
later [fusion-preserving capture](../fused-routes/README.md) keeps upstream
optimized logits bit-exact and shows the optimized plan and the plan with
fusion/graphs disabled select different expert sets in 40–43% of token-layer
rows; these captures were not re-recorded.
Teacher-forced inputs originate from the optimized reference. Top-1 agreement
is not bitwise logit equality, and capture call times are not throughput evidence.
Some small captures overlapped on the GPU, so those times are diagnostic only.

Qwen adds a separate limitation: an identical untraced binary repeated on the
same host differs on **6/1,536** predictions (3, 2, 1 by turn). Its restore probe
differs on 10 predictions, including five before any restoration, so those
differences cannot be assigned to restoration alone. The cause of baseline
non-repeatability is unresolved. Its capture explicitly records numerical
equivalence failures; replay requires explicit opt-in and compares policies
against the **same sampled traced trajectory**. These are conditional traffic
estimates, not a claim of equivalent untraced/normal-model behavior or an
accepted numerical tolerance. Qwen spill returns conservatively recompute.
Across its four profiles, traced/control differences are **10, 15, 6, and 4
out of 1,536** predictions (full retained, full recompute, default recompute,
default retained). All four equivalence checks are recorded as failed.

Qwen also prunes its last prefill layer to the output row. The first capture
correctly stopped at the harness's overly restrictive all-rows assertion.
The corrected callback records the actual one-row final-layer dependency,
explicitly marks it as output-only, and still accounts for the full consumed
input batch. It rejects reduced row counts in other layers or decode.

## Models and licenses

| Model | Quantization | Primary routed layers / experts / selected | Tensor payload GiB | Non-expert GiB | Isolated prepared weights GiB |
| --- | --- | --- | ---: | ---: | ---: |
| Gemma 4 26B A4B | UD-Q4_K_M | 30 / 128 / 8 | 15.77 | 2.402 | 17.652 |
| Ornith 1.5 35B | Q4_K_M | 40 / 256 / 8 | 20.21 | 2.048 | 22.049 |
| DeepSeek V4 Flash | UD-Q2_K_XL | 43 / 256 / 6 | 90.177 | 6.568 | 94.068 |
| Qwen3.8 Flash Next | UD-IQ3_XXS | 48 / 512 / 10 | 76.323 | 31.035 | 80.035 |

Exact tensor accounting and artifact hashes take precedence over rounded table
values. Stored unused MTP tensors are conservatively charged to non-expert
weights. Qwen's n-gram table is also charged in full, and native capture turns
lazy loading off. The normal-reference server uses its normal lazy setting;
its configuration is consequently a separate comparison view.

`full-pins.json` pins all three shards of each large model, the quantizer
revision, base-model revision, and actual LICENSE hash. DeepSeek uses MIT.
Qwen uses Qwen Community License 1.0, **not Apache-2.0**; its internal-use
exception permits this internal benchmark, while its separate commercial
service conditions must not be generalized into redistribution permission.
Weights and their preserved licenses remain external. These are reference
model artifacts, not incorporated implementation dependencies. Gemma/Ornith
retain the license classification in the existing reference artifact ledger.

## Replay assumptions

All policies consume the same requests and memory budget. `partial` acquires
the selected layer/chunk's entire dependency closure and keeps unused extents
in global deterministic LRU. `retained_eager` loads the active model completely
but reclaims individual extents from inactive weights. `whole` evicts complete
inactive weight sets only when necessary; it keeps both models if they fit.
This avoids handicapping the full-swap comparator at large budgets.

Active non-expert weights are fully leased. Inactive non-expert weights and
expert weights are reclaimable in 2 MiB extents. The replay represents
non-expert pages compactly because they share an access age; independent
expanded-page simulations verify identical reclamation and refill accounting.
Two hypothetical layouts are evaluated:
each expert closure padded independently to 2 MiB, and closures packed within
each layer with shared boundary extents charged once. The latter is a padding
sensitivity, not a validated executable artifact layout or adopted ABI.

Live state is charged using observed context allocations, including recurrent,
compressed-attention, and indexer state. Fixed overhead adds 1 GiB scratch,
2 GiB headroom, and four 2 MiB I/O destinations; observed workspace above
1 GiB adds to that allowance. Each node remains one physical memory budget.
Large-study points above 121.6877 GiB are hypothetical larger single-domain
capacity sensitivities. They are not Spark measurements and do not combine
two Sparks into shared memory; configured placement remains a separate path.
These overheads are scenario allowances, not measured physical peaks or an
admission guarantee. The modeled direct state write does not add a second
payload-sized serialization buffer; the diagnostic native restore probe does
make host copies, whose cost is excluded from the storage-service estimates.
Inactive live-state allocations remain charged in the resident scenario.
The spill scenario writes the outgoing snapshot before releasing it, restores
the incoming one, and enforces an 8 GiB spill ceiling including both files.
Snapshot sizing includes the native sequence-file header and token IDs.
The recompute scenario drops state and uses full-prompt captures. Gemma reuse
requires sufficient window coverage (RE-004/RE-007). Corrected live captures
record the coverage check; older default-SWA captures receive conservative
recomputation fallback, and affected sequence-spill returns use full-prompt
captures. No state is expired within these traces.

When the first full model fits, all policies start with those weights warm;
otherwise they share a cold start. An execution closure that exceeds the
budget is infeasible, never silently evicted while leased. Initial model load
is excluded in the warm scenario; subsequent misses are counted.

The bulk scenario uses measured 14.962131 GB/s direct-to-host-VMM reads.
The serial sensitivity charges 177 microseconds per 2 MiB weight extent.
Writes use the measured 11.641 GB/s burst result plus 3.838 ms flush allowance;
this is not a sustained-drive write guarantee. A hypothetical 1 GB/s case
shows sensitivity to slower storage. Decode exposure is shown with 0% and
hypothetical 50% overlap. There is no measured compute lead time establishing
that 50% is achievable. See the [I/O spike](../io-path/README.md).

## Findings from the longer small-model workload

With full SWA and resident state, retaining inactive weight extents provides
most of the modeled benefit near the reference pressure budget without demand
paging during generation. At **41.6877 GiB**, isolated closures read **70.740 GiB**
over eight requests under retained-eager switching, versus **141.152 GiB** under
whole-model switching and **70.004 GiB** under routed demand paging. Eager
loading has zero modeled decode misses; demand paging's largest per-request
p95 decode service estimate is **2.383 ms/token** at the bulk read rate.
This is evidence for M6's partial retention before M7's expert paging, not a
measured end-to-end speedup or a universal choice of cache policy.

| Total budget GiB | Routed demand reads GiB | Retained-eager reads GiB | Whole-model reads GiB | Demand decode service: largest request p95 ms |
| ---: | ---: | ---: | ---: | ---: |
| 28 | 139.404 | infeasible with both live states | infeasible with both live states | 4.205 |
| 36 | 102.338 | 104.865 | 141.152 | 4.065 |
| 40 | 82.451 | 80.865 | 141.152 | 3.224 |
| 41.6877 | 70.004 | 70.740 | 141.152 | 2.383 |
| 44 | 46.453 | 56.865 | 141.152 | 1.261 |
| 48 | 22.580 | 32.865 | 141.152 | 0.701 |
| 64 | 21.971 | 22.049 | 22.049 | 0.701 |

These totals include first access to Ornith; Gemma starts warm. At 64 GiB,
keeping both complete models removes repeated swapping in the whole-model
comparator too. The first Ornith request still brings in weights. Tiny
remaining cold demand misses explain a nonzero maximum p95 even when the
eventual working set fits.

Packing and state policy materially change the fit threshold. At 41.6877 GiB,
the hypothetical packed layout with default-window allocations lets both small
models fit: all three policies read approximately **20.213 GiB** over the run,
essentially the initial Ornith load. That result does not include the compute
cost of Gemma's required prompt recomputation. Under full-SWA resident state,
packing reduces retained-eager reads from 70.740 to **46.639 GiB**. A validated
import layout can therefore matter as much as more elaborate demand policy.

For the first Gemma return in the longer corpus, full-SWA resident-state
retained-eager loading estimates **0.582 s** bulk storage service, **0.735 s**
with serialized extent reads, and **8.714 s** in the hypothetical 1 GB/s case.
Whole-model switching estimates **1.267**, **1.600**, and **18.954 s** respectively.
These storage sensitivities preserve the byte savings while showing why a
single bulk-bandwidth conversion is not a switching-latency prediction.
The small corpus's spill peak is **551.75 MiB**, below the explicit 8 GiB cap;
the cap is exercised independently in regression tests rather than claimed
as a stress point reached by this workload.

### Matched A→B→A and the full-swap floor

At the exact reference experiment's 41.6877 GiB remaining budget, isolated
closures produce the following **estimated storage service**, with full-SWA
allocations. These are the original outward B request and return A request.

| Policy / state | Outward weight reads through first token GiB | Return weight reads through first token GiB | Outward / return storage seconds |
| --- | ---: | ---: | --- |
| Demand / resident | 8.621 | 0 | 0.619 / 0 |
| Retained eager / resident | 22.049 | 8.115 | 1.582 / 0.582 |
| Whole model / resident | 22.049 | 17.652 | 1.582 / 1.267 |
| Demand / spill | 8.621 | 0 | 0.673 / 0.049 |
| Retained eager / spill | 22.049 | 1.240 | 1.636 / 0.138 |
| Whole model / spill | 22.049 | 17.652 | 1.636 / 1.316 |

The measured pressure reference took **21.232 s outward and 18.304 s back**
with sequence restore, and **25.236 s back** with recomputation. Those measured
times include process/model lifecycle, real GGUF loading, kernels, and actual
state operations. The modeled prepared extents and bulk service do not capture
those costs. In particular, the replay's safe Gemma spill return recomputes
18,339 prompt tokens after detecting insufficient saved-window coverage;
resident return reuses 18,297 tokens and processes 42. Zero modeled reads on
a return with resident weights says nothing about the time to recompute it.
The earlier reference's short matching restored output does not prove complete
window coverage after prefix rollback. Keep its measured timing as historical
evidence, not a validation of the broader restore contract.

### Restoration requires valid context coverage

The extended negative probe restored sequence snapshots into newly created
contexts and reserialized identical bytes, but Gemma then differed from live
continuation on **58/3,072** top-1 predictions. Ornith matched **3,072/3,072**.
The stored Gemma window omitted **770 positions** needed when the next rendered
chat prompt rewound past the prior response. Full-SWA allocation does not make
the reference's sequence serializer save all historical window cells.
Default-SWA live caches can have the same coverage gap.

The corrected harness checks actual attention-window coverage before reuse
and recomputes when insufficient. Corrected Gemma restore detected all three
insufficient windows and matched **3,072/3,072** predictions from the full
recomputation control. Replay never credits the old invalid
default-SWA reuse and chooses captured full-prompt routes for affected Gemma
spill returns. It still charges the unsuccessful restore attempt's I/O.
See [RE-007](../../rough-edges.md). This is a concrete requirement for M6:
retention metadata must describe restore coverage, and a failed coverage test
must select an earlier valid checkpoint or recomputation. Byte equality alone
is not a continuation-correctness test.

DeepSeek's append-only restored probe also differs on **16/1,536** predictions
(0, 6, 10 by turn), despite identical serialized round trips. It reuses the
entire prior prefix, so the Gemma rollback explanation does not apply. Source
inspection found the expected raw/compressed KV and compressor state in the
serializer; physical cache compaction can change attention shapes, but the
cause has not been isolated. The initial comparison also used different Spark
hosts; a repeat on the control's host reproduced the same 0/6/10 differences,
and both restored prediction files match exactly across hosts. Host choice
does not explain this observed difference. Neither harmless rounding nor
upstream corruption is established.
The large-model replay therefore explicitly sets `--spill-recompute D`:
DeepSeek returns recompute after attempted restoration, with that attempt's
I/O still charged. This conservative scenario avoids crediting unvalidated
restored-state reuse.

### Batch size and predictability

For batch-512 prefill in the longer retained-context corpus, per-layer token
unions average **83.63/128** Gemma experts and **154.55/256** Ornith experts,
with observed maxima 121 and 229. Decode reuse distances count distinct other
experts used since the previous selection, treating a token's selected set as
a group. Across Gemma layers, median reuse distance ranges **0–25** and p95
**44–93**; Ornith ranges **8–84** and **106–218**. Cold selections are counted
separately. The large layer-to-layer spread argues against a single locality
assumption for every layer; full per-layer counts remain in the aggregates.

Four independent sequences decoding together select a mean **24.56** distinct
Gemma experts per layer and **27.96** Ornith experts, against eight for one
sequence. All **3,068 decode predictions per model** match their untraced
four-sequence controls. This check starts after prefill; it does not compare
the first output prediction of each prompt. Total contexts are 49,152 tokens
for Gemma and 32,768 for Ornith, with 10.3125 and 0.8711 GiB of allocated state.
This experiment measures the actual union of routed dependencies, not four
copies of a single route.

At 24 GiB total budget, isolated demand paging's estimated p95 storage service
per four-token batch is **24.949 ms** for Gemma and **1.402 ms** for Ornith.
At 28 GiB it is **1.121 ms** and **0 ms**, respectively. These are separate
single-model budget sweeps, not the alternating-session experiment. Aggregate
throughput cannot be inferred by dividing these service estimates by four.

With prefill batch 512, held-out next-layer co-occurrence recalls **46.78%**
of selected Gemma contributions and **37.88%** for Ornith. Static popularity
recalls 30.40% and 18.51%; previous-token reuse recalls 42.05% and 34.75%.
Prefill batch 64 produces closely similar held-out recalls (46.78%, 37.94%).
The original narrow A→B→A corpus does not establish the same gain: its Gemma
next-layer recall is 29.67%, below previous-token reuse at 40.62%.
The large-model held-out cases also favor previous-token reuse over this
particular next-layer predictor: DeepSeek recalls **23.03%** next-layer versus
**31.39%** previous-token, and Qwen's conditional trajectory **25.85%** versus
**37.15%**. Each trains on one request and tests two later requests; this is a
small exploratory split, not evidence that learned prefetch will generalize.

Bringing in a completely cold next-layer prediction requires roughly
2.24 ms for a typical Gemma layer and 1.12 ms for Ornith at the measured bulk
rate. None of those cold prediction sets fits a hypothetical 0.1, 0.5, or
1 ms lead time. There is useful predictability, but this evidence does not
support promising that one-layer-ahead prefetch hides all misses. Per-layer
reuse-distance distributions and held-out counters are retained in aggregates.

## Larger-than-memory library

The DeepSeek/Qwen pair exercises a library whose prepared weights exceed one
Spark's physical memory. The following full-SWA, resident-state curves use
isolated 2 MiB closures and the same captured requests for every policy.
**All pair-level results are conditional on Qwen's sampled traced trajectory**;
its failed equivalence checks prevent claims about equivalent normal execution.

| Total budget GiB | Routed demand reads GiB | Retained-eager reads GiB | Whole-model reads GiB | Demand decode service: largest request p95 ms |
| ---: | ---: | ---: | ---: | ---: |
| 96 | 380.951 | infeasible | infeasible | 21.585 |
| 112 | 251.244 | 343.387 | 428.242 | 20.744 |
| 121.6877 | 230.586 | 304.637 | 428.242 | 18.221 |
| 144 | 126.012 | 215.387 | 428.242 | 8.970 |
| 192 | 76.635 | 80.035 | 80.035 | 3.224 |
| 256 | 76.635 | 80.035 | 80.035 | 3.224 |

At 121.6877 GiB, demand paging reads fewer bytes than retained eager, but its
largest request p95 decode service is 18.221 ms/token before overlap; eager
loading has no modeled generation-time misses. This illustrates the tradeoff
between switching traffic and decode exposure, not a measured throughput gain.
At 96 GiB the full DeepSeek execution envelope does not fit, so all policies
share a cold start; demand can execute smaller closures. At 192/256 GiB both
complete models fit and the whole-model comparator retains them. The 144 GiB
and larger points describe hypothetical larger single memory domains, not
available capacity on one Spark or combined memory across two nodes.

The conservative spill scenario recomputes both large models on return after
charging attempted restoration. At 121.6877 GiB its total reads are 304.906 GiB
for demand, 303.184 GiB for retained eager, and 428.242 GiB for whole-model
switching. Recomputed prefill changes the route workload and nearly removes
the resident scenario's demand advantage over retained eager. The largest
large-workload spill occupancy is 228.763 MiB, below the 8 GiB ceiling.
Snapshot correctness and prefill policy therefore matter alongside locality.

## Normal-reference large-model switching

The pinned optimized server ran the canonical two-large-model pair with its
normal cache settings and one loaded model at a time. Each model executes on
one Spark; their combined tensor payload exceeds its physical memory budget.
Aggregating the six load/switch-to-first-token observations:

| Incoming model / phase | Count | Median seconds | Observed range seconds |
| --- | ---: | ---: | --- |
| DeepSeek / initial load | 1 | 99.520 | 99.520 |
| Qwen / first switch | 1 | 92.714 | 92.714 |
| DeepSeek / returns | 2 | 90.384 | 88.925–91.843 |
| Qwen / returns | 2 | 75.207 | 75.193–75.221 |

Each switch clock starts before saving the previous result and includes file
flush, harness bookkeeping, unload, load, restore, prompt construction, and
the first token. This is **one six-request run**, with no controlled cache
conditioning, locked-memory pressure, or repeated-trial confidence interval.
It is not comparable as a performance ranking against another implementation's
reported 46 seconds. The reference loaded/saved actual conversation state;
subsequent reuse is reported separately from a successful restore API call.
The small-model 27-trial pressure experiment's validated recomputation arm
(25.236 s return) remains the stronger measured correctness floor for its
named workload; its 18.304 s restore result is historical timing with the
newly identified coverage limitation.

## Reproduction and limits

Raw token inputs, routes, logs, snapshots, model weights, and source archives
stay outside Git. Keep the frozen external inputs by hash; rerunning generation
creates a new workload identity, especially timestamps and sampled outputs.
`full-evidence.json` records each native receipt, including distinct hosts and
builds, input identities, and positive/negative prediction comparisons.
`full-aggregates.json` contains the budget curves, per-layer locality, and
parallel-batch summaries. Capture specifications carry provisional allocation
fields; replay replaces them with allocations measured in the pinned logs.
The aggregate's derived `kv_bytes`/`normal_kv_bytes` govern accounting, while
`input_spec_kv_bytes` retains the original field for audit.
`prepare_sessions.py` generates the varied corpora; `run_sessions.py` builds
and checks controls/captures; `switch_replay.py` validates inputs and replays
policies; `locality_sessions.py` reports per-layer reuse and held-out next-layer
prediction; `parallel_replay.py` handles four simultaneous decode sequences.
`restore_probe.cc` checks snapshot serialization across context destruction and
recreation. External manifests and inputs are explicit arguments, never
downloaded or trusted implicitly by the replay.

Supply the hash-matching private inputs and model directories, plus the pinned
llama.cpp source headers from the first-cut `pins.json`. Capture specifications
can be extracted from the relevant receipt's `spec` in `full-evidence.json`.
The commands below use external paths and create new output directories/files:

```sh
python3 run_sessions.py SPEC MODELS SOURCE INPUTS NEW_CAPTURE
python3 run_sessions.py Q_SPEC MODELS SOURCE INPUTS NEW_Q_CAPTURE --record-prediction-drift Q
python3 run_sessions.py RESTORE_SPEC MODELS SOURCE INPUTS NEW_PROBE --restore-probe
python3 run_sessions.py PARALLEL_SPEC MODELS SOURCE INPUTS NEW_PARALLEL --parallel
python3 merge_captures.py MERGED_CAPTURE DEEPSEEK_CAPTURE QWEN_CAPTURE
python3 switch_replay.py MERGED_CAPTURE LARGE_SESSIONS_JSON NEW_REPLAY_JSON --allow-prediction-drift Q --spill-recompute D --spill-recompute Q
python3 locality_sessions.py MERGED_CAPTURE NEW_LOCALITY_JSON --allow-prediction-drift Q
python3 parallel_replay.py PARALLEL_CAPTURE NEW_PARALLEL_JSON --inputs INPUTS
python3 -m unittest discover -s . -p 'test_*.py'
```

Run native capture on a Spark with the existing pinned container runtime
(`DOCKER="sudo -n docker"` in this environment); replay/tests run on the
workstation. The old invalid normal-SWA and unsafe restore captures are
negative evidence, not recommended reproduction configurations. Current
sources enforce coverage checks and record output-only final-layer routing.

The held-out predictor trains on the first half of requests and freezes before
testing later requests. It compares next-layer co-occurrence against static
popularity and the previous token. Its byte estimate assumes all predicted
closures are cold; it is not a cache-aware prefetch simulation. The synthetic
notebook corpus, fixed teacher forcing, single quantization per model, and
small request count limit generalization. Runtime admission, actual paging,
VMM completion safety, executable artifact layout, cache policy tuning, and
end-to-end generation stalls remain later implementation evidence.

## Consequences for the implementation plan

Keep M6's partial weight retention and state management ahead of M7's routed
expert paging. Measure an eager active-model load that preserves inactive
extents as the first policy; the small-workload curves show it can capture
most of the demand policy's byte savings without generation-time misses.
Keep the measured full-swap/recompute path as a fallback comparison.
On two nodes, M6a must compare configured placement before paying these paging
costs. Capturing models on different Sparks is not a distributed-serving or
placement benchmark.

M6 state tests must cover rendered-prompt rollback, insufficient sliding-window
coverage, recurrent state, expiry, and recovery from an unusable snapshot.
Round-trip bytes and one successful short continuation are insufficient exit
checks. Preserve or restore a valid earlier prefix, or explicitly recompute.

M7 must distinguish prefill from decode and test batch unions. Small-budget
global LRU can repeatedly fetch nearly the entire prefill working set; paging
must not be assumed beneficial merely because one routed layer fits. Evaluate
prefill admission and loading policy with the existing full-swap floor.
Measure useful prefetch lead time and cache-aware extra traffic before adding
prediction complexity. No selected contribution may be replaced or skipped.

Following this study, the owner accepted workload-specific switching-benefit
and generation-stall targets in D-036 (2026-09-22). This study supplies the
supporting evidence; it does not demonstrate that an implementation meets
those targets.

## Build and verification handoff

The full study completed **28 traced/control pairs**. Twenty-four pairs matched
all **43,792** compared top-1 records; Qwen's four pairs compared 6,144 records
with 35 differences in total and remain explicitly failed equivalence checks.
Safe Gemma restoration matched its recomputation control on 3,072 records;
Ornith restoration matched retained continuation on 3,072. DeepSeek's repeated
negative restore result and Qwen's baseline drift are retained as evidence.

Native builds/captures ran on the two named Sparks using the pinned image.
Workstation checks passed **71 unit/regression tests**, GCC 13.3
warnings-as-errors C++23 syntax checks against pinned headers, and an independent protected-LRU
comparison over **2,000 randomized layouts / 160,000 transitions**, plus
**3,600 complete switching simulations** across all policies, state modes,
and layouts. Prediction
files are checked against complete workload request/batch identities, not just
against each other. Current-source hashes and per-run source/binary/log hashes
distinguish revised harnesses from historical evidence. Aggregate generation
checks original event, prediction, model, input, and resource-log identities.

Actual llmpalooza paging, prepared-layout execution, distributed placement, sustained
state-write performance, and end-to-end stall acceptance were not implemented
or benchmarked by this experiment. They remain their named later milestone
checks. All repository changes remain in the working tree for human review
and commit.

## Independent final review

A separate reviewer examined the complete change and challenged native capture
bounds, routing completeness, protected dependency closures, state and spill
accounting, prediction validation, licenses, provenance, and report claims.
Review fixes bind prediction files to complete request/batch identities even
when both comparison files are identically truncated, and preserve extent-level
reclamation of inactive non-expert weights. No unresolved implementation or
report defects were found in the final bounded experiment scope.

The reviewer reran all **71 tests** on the workstation. An independent explicit
2 MiB page oracle matched **160,000 cache transitions** and **3,600 complete
switching simulations**, including partial reclamation/refill, protected active
weights, whole-model eviction, and infeasible budgets. Final aggregate contents
match all **792 replay cases**, three locality results, and the parallel result.
Reported budget tables, return sensitivities, packing comparisons, and spill
totals agree with those outputs. All **25 current source hashes**, **81 recorded
run entries** (including merged duplicates), and five input identities match
the external evidence; large-model shard identities match the pinned manifest.

GPU execution and native compilation were performed by the builder and reviewed
from its evidence, not independently repeated by this reviewer. Qwen numerical
drift, unvalidated restored continuations, hypothetical layouts, and the absence
of actual llmpalooza paging remain explicit experiment limitations, not passing
runtime correctness or performance claims.
