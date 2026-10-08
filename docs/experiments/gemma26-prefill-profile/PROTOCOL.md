<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 paid-prefill performance diagnosis — compact first trace

Benchmark-only prefill diagnosis. New task entry
2026-10-05T14:59:25.284078Z refreshed primary TensorFold HEAD
609ca419abecebdc5a059498a613680bd3aa847f, version 0.6.5; four primary files
are retained with hashes. Gemma26 remains MLX-only, not a same-format CUDA
comparator. Source base is committed Task40
36811f4ac93975c490e9118b93506aa3880df298.

## Question and fixed experiment

Locate the work behind native26 all1024 prefill3.80551s versus corrected ring
reference2.41568/2.41232s (+57.6433% native latency). These are the existing
untraced Task40 measurements, not predictions from profiling. Only the
8192-row prefill block is the primary attribution window; preserve the
remaining three anchors/32 incoming-head choices and two heads/state as
completion and mathematical-fidelity controls.

Keep target4ddb/rawf2c28, canonical8227 IDs6b6567, C1/context16384,
physical ubatch/max_rows1024, local2048/global16384, F16KV, falseSWA/unified,
six warm rows+clear, checked norm60/90 and uniform route/reduce30/30,
ordinary products and all other optional native policies0. Original image837fc
and b29 math libraries/header ABI remain immutable, fusion/graphs allowed.
No cache, masks, product/head/fusion selection changes, no eval callback,
kept graph intermediate or operation-observer that could change eligibility.
No additional policy/profile/context ladder.

Native pays eight Chunk calls and eight complete vocabulary publications;
reference pays eight physical1024 ubatches but requests only one final
head. Do not remove the extra seven native publications or assert that
they explain the whole gap. Attribute the actual head projection kernels
and transfers where uniquely identifiable, and keep ambiguity explicit.

## Minimal phase annotation and observation

The installed public NVTX interface is header-only, Apache-2.0 WITH
LLVM-exception. Headers remain external under the Nsight2025.3.2 tool root;
no production dependency or header vendoring. Model-free build and runtime
trace checks must establish the exact range in SQLite before model loading.

Use separate copied manual native helper and copied ring client, preserving
historical measured helpers. Add only one constant host NVTX range around
the paid prefill: push immediately before the existing start-clock read,
pop after the existing elapsed-clock read (native includes its unchanged
last_built_policy lookup). No ranges/counters inside the paid loop, no kernel
argument/plan instrumentation, no production source edits. The existing
stopwatches retain their boundaries, but any times from profiled runs are
diagnostic and never replace the untraced bookend.

First build-only feasibility verifies the already-installed Nsight2025.3.2
public NVTX header/injection interface, records exact tool/header/license
identity, and tests a tiny range without models. No guessed ABI, invented
pin, new library math build or owner checkout mutation. If that public
interface is unavailable, stop for a revised protocol; do not guess a paid
window from sleeps or buffered stdout timestamps. The profiler range is
observer metadata only; it must not introduce an original backend callback.

Trace the whole short application with CUDA/NVTX/OS-runtime records,
GPU graph-node activity when supported, CPU sampling disabled initially.
Extract only the explicit paid-prefill NVTX interval. Source/model loading,
warm/reset, anchors, decode, file/state copies and teardown remain visible
for control but excluded from prefill attribution. No delay/duration cut-off
that can leave the model alive after the profiler exits.

Use actual GPU activity records/correlation IDs (including executed graph
nodes), not cudaGraphLaunch event counts, to group products, routed/shared
expert operations, attention, norms/RoPE, cache/mask/gather/scatter and
host↔device transfers. Retain observed symbol, grid/block, stream,
correlation/graph identity, bytes/duration and phase boundaries externally.
Ambiguous roles stay unknown. A large vocabulary projection is labelled
as head work only when its actual launch geometry plus pinned source/shape
and position uniquely identify it; no assumed whole-model operation list.

Check for eight native1MiB head D2H publications versus one reference1MiB
publication inside prefill. Their source-verified sequence can delimit
native chunk completion, but it does not identify every preceding kernel.
Preserve separate actual kernel counts versus graph/API launches and source
predicted operation counts. Node-level graph tracing may expose many events;
each is still diagnostic, not a production timing cost.

## Host, GPU and resource accounting

Report paid-range wall span, union of GPU active intervals, CUDA API/capture/
instantiate/launch/copy/synchronization spans, and observed OS-runtime blocking.
Report overlap instead of adding durations across concurrent streams/threads.
The residual is host/unattributed time, not automatically scheduler or planner
CPU time. Without CPU samples or internal ranges, CheckPlaces/node.Call and
descriptor/plan construction remain hypotheses. If the residual is material,
request one separate bounded passive CPU-sampling diagnosis rather than
adding broad runtime instrumentation in this first trace.

Verified source leads, not measured attribution yet:
- benchmark gemma_prefill.cc123–133 pays eight separate Chunk outputs and
  last_built_policy inside the prefill timer;
- runner CheckPlaces visits weight/state extents through node.Call;
- each new shape includes global/local read widths, so Planned can miss
  as the initialized endpoint grows;
- state growth, host input staging, capture/instantiation and publication
  belong to the actual paid Chunk work.
Do not remove checks, state growth, graph funding or head publications.
CUDA activity can show copies; it does not alone establish disk-page-in bytes.
Warm residency and actual observed copy/file-read evidence must be distinguished.

## Fidelity, retirement and bounded run sequence

Freeze copied annotated source/binaries, exact math/build receipt/source
manifest, original four libraries/headers, tool identities and Task40 own
source25e5/nativea1f7 linkage before model profiling. An untraced annotated
control must match Task40 native two heads/state592445440/layout2048/32
choices, and original two heads/32 choices must match its own Task40 ring
result. Annotation-source timing is not a new performance screen.

Then one native and one original trace, separate installed --gpu jobs,
timeout600 each, expected10–30s each. Keep the exact same model workload;
no warm graph reuse outside the established recipe. Every traced result
must again match the authenticated untraced complete output/state witness.
If tracing changes math/fusion/graph completion, stop instead of interpreting
its kernel breakdown. No extra whole-engine unit suite is needed for this
benchmark diagnostic.

A separate small child-wait executable runs the actual helper without creating
another process group. It flushes/closes an exclusive mode0600 temporary receipt only
after waitpid proves child exit0, then publishes by exclusive hardlink. A
write/fsync/close failure leaves no final receipt. The frozen native helper returns0 only after
completed file flushes and proven whole-owner teardown; the original helper
returns0 only after batch/context/model/backend release. Output identities are
then verified separately after official supervisor retirement. Preserve existing
native closure/admission, funded pinned snapshot and unproven-retirement
quarantine. Do not use output-file existence as proof of completion.

Profiler and application success are separate (RE-039). Require installed
supervisor final/job/log records, explicit successful child termination and
the final completion receipt; then verify no surviving profiler/application
GPU processes before handing B over. Use whole-run collection/child wait,
not a bounded profiler capture whose target can outlive it. On timeout or
unknown retirement, retire the owned process group through spark-job and
report failure; never count a valid .nsys-rep alone as an application pass.

Docker daemon children are outside the local supervisor process group. Each
original acquisition has a new CID file, unique name and task ownership label.
EXIT/INT/TERM traps retire only that recorded, label-checked container. The
retirement receipt requires a successful Docker query proving that exact CID
absent. Preserve CID files; if the wrapper is SIGKILLed before its trap runs,
use reference.sh cleanup OUTPUT_NAME under the installed supervisor before
another model. Require the recorded container absence and Spark busy check;
process-group retirement alone does not prove container/app retirement.

Nsight output/SQLite/raw events stay external. Git receives aggregate
category/overlap/count tables, measured identities and limitations. No
physical peak ratio, kernel adoption, quality qualification or claim that
profiled timings establish the untraced performance gap. Dense31 transfer
is a later separately approved trace, not an inferred result from26.

## Interface proof and reproducible identity gates

The no-model probe compiled the public header in the original image, then used
exactly the intended `--trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
--cuda-graph-trace=node --wait=all --stop-on-exit=true --kill=none --export=sqlite`
flags. SQLite contains one positive `llmp.gemma26.paid_prefill` range. The
child-wait controls refuse a nonzero child and an existing completion file;
child, profiler and installed supervisor all retired successfully. Header/license
texts and SQLite stay external; final provenance keeps hashes and the actual
Apache-2.0 WITH LLVM-exception license, not complete header copies.

Before building, make an external `production-source-manifest.json` containing
all tracked `src/`, `third_party/` and `toolchains/` paths with byte length and
SHA256, plus sorted path NUL content NUL bundle SHA256. Verify those files on the
checksum-synchronized Spark tree before the locked manual-target build. Make
`source-ancestry.json` with base36811f4, empty tracked_production_diff, exact
changed/new dirty_paths and source_files identities for validate.py SOURCES.
The dirty paths must be confined to that allowlist. Copy the authenticated
canonical IDs, eight original headers and model-free nvtx-interface.json to the
external scratch; the original build writes libraries.sha256.

Run validate.py source ROOT SCRATCH BUILD_JOB after successful build retirement.
It exclusively creates source-identities.json after rechecking current sources,
math/header/tool identities, artifact metadata, production manifest, actual
build receipt and both binaries plus child-wait binary. The retained Task40
source/own receipt and bookend log are authenticated by fixed SHA256. Every
later gate reauthenticates the complete environment against this source freeze.

Run native.sh control native-control and reference.sh control original-control
in separate supervised jobs. After retirement, run validate.py result ROOT
SCRATCH JOB SOURCE_SHA OUTPUT_NAME ENGINE; it requires finite complete two-head
identity, all32 incoming choices, native initialized-state identity/layout and
unchanged non-timing policy/count fields against Task40. Actual local2048 and
global16384 ring creation logs are also required. The resulting exclusive
control receipt is the fidelity gate, not a fresh competitive measurement.

Only after both controls pass run each wrapper trace with native-trace or
original-trace. Run trace.py ROOT SCRATCH SOURCE_SHA OUTPUT_NAME ENGINE JOB
CONTROL_RECEIPT_SHA after retirement. It authenticates the exact annotated
control, repeated complete outputs, child completion and source environment,
then reads the actual paid NVTX interval and CUDA graph-node activity. Raw
SQLite/event records remain external. Overlap unions are clipped to the range;
category summed durations are labelled as overlapping. No guessed kernel-role
classification or planner CPU attribution is made by the extractor.

The optional manual CMake target is excluded from the default build and requires
LLMP_BENCHMARK_NVTX_INCLUDE_DIR to name the recorded external public headers.
No production compile or runtime target depends on NVTX. The public observer
probe and immutable model math are separate identities.

Additional head-cost source caveat: frontier_head narrows the final block's
projected/residual rows before FFN. Thus eight native publications versus one
reference publication can change final-block FFN/product work as well as D2H
copies. Disabling the head would expand hidden output and keep an unnarrowed
FFN, so no head suppression is assumed cheaper or state-only in this task.
