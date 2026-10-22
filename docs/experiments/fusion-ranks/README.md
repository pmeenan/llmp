<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fusion versus its own decomposition: first U6 measurement slice

Status: [fresh W5 results](FRESH-W5-RESULTS.md) independently qualify all twelve
registered exact F32 RMSNormMul tuples: eleven fusion wins and one neutral cell.
The W5 native live-registry/planner verification recognized all twelve rows on
compiler identity `4c2b14cf0a66eebb8f433daefff9b58b4cfc0afdc082d6ba0e5a1ec3ac200cee`.
The U8 spatial-definition hook build produces identity
`8148b25f8c488058047940556525d80736d9d5a4fd4b4401c7bd0fa323146830`:
34 of the 195 compiler-owned object receipts change, including the component
objects and their planner/backend dependents. The twelve W5 rows remain exact
historical evidence and are stale on this build; selection retains the incumbent
until newly source-matched comparisons are measured. The subsequent
[U7a GDN gate spike](../gdn-gates/RESULTS.md) produces compiler identity
`374364310fbe78618b0f332a189bb8d9db18fde6eea5ea646892aa088429c50e`;
its narrow GDN screen does not refresh these RMSNormMul rows. Full U6 remains open.
The [original results](RESULTS.md) and
[historical data](historical-v11-v12.json) retain unchanged measurements and
identities; they are stale for current source. The registrations below preserve
the original protocol and pre-execution status. The current-source rerun followed
[FRESH-W5-REGISTRATION.md](FRESH-W5-REGISTRATION.md), using the corrected
[compiler-owned source identity](SOURCE-IDENTITY.md).

## Prospective protocol (2026-10-22, before measurement)

The first screen is F32 RMSNormMul, width 1024 and one row on GB10. This is a
synthetic operator shape, not a checkpoint-size eligibility condition. The
standalone `llmp_fusion_ranks [width [rows]]` constructs one fused IR node and
uses `ir::Expand` to obtain **that node's own decomposition**. Both arms lower
through the production planner/backend. The primitive arm disables only the
RMSNormMul rewrite so it cannot reassemble the fusion under test. The harness
requires exactly the fused norm/multiply versus separate norm and multiply
implementation inventory. No checkpoint, source-quality claim, numerical
proposal, or alias mode is involved: both arms compute the same F32 contract,
with identical input data, epsilon, result encoding and complete output shape.

Before timing, every output must be finite, relative RMS difference must be at
most 1e-6, and maximum absolute difference at most 2e-5. These are prospective
arithmetic tolerances, not measured errors. Both arms use deterministic,
nonzero bounded inputs and per-channel scales. The same complete output is
read after one execution; the timed arms replay the captured complete chains.
Inputs, intermediate storage and output remain distinct between arms.

Four untimed paired warmups precede a primitive bookend, four additional paired
warmups and nine alternating AB/BA paired samples, then a primitive bookend.
Each event sample replays one graph containing 128 complete chains, avoiding
CPU submission gaps between individual kernels. Captured-node inventory and
replayed output are checked against the eagerly executed arm before timing. Report median nanoseconds,
all selected implementation identities, graph-node counts, activation bytes,
query key, selection-environment digest, device/model/architecture/multiprocessors, driver/runtime versions,
output differences and both bookends. Preserve all nine unsorted pairs with
first-arm order; compute medians from copies. Flush a JSONL receipt after every
warmup arm, timing arm and bookend, and a refusal receipt before a detected
failure, so a late failure retains completed samples rather than discarding
unfavorable evidence. Refuse timing evidence if bookends differ
by more than 10%; do not adjust this bound after observing results. No clocks,
profiling or kernel counters are needed for the first screen. Each case runs
as an exclusive supervised Spark job, at most 600 seconds; expected runtime is
seconds after the narrow native build.

Selection follows D-085's coarse band: a fusion over 10% slower than its own
primitive chain is excluded for this exact query, and the existing planner
fallback decomposes it. A fusion with latency more than 10% below its
primitive chain is measured-positive. Both bounds use the decomposition latency
as the denominator; exactly 90% and 110% are neutral and retain the incumbent.
Production selection and harness reporting use the same overflow-safe classifier. Numerical eligibility
is checked independently before this preference; speed evidence cannot grant
an arithmetic change. Unmeasured and source-stale records retain the incumbent.

If the first screen validates the mechanism, the next small matrix is widths
256, 1024 and 4096 at rows 1 (decode), 4 (short wave), 32 and 256 (prefill).
Measure each tuple; do not infer an interval from endpoints or extrapolate
1024 to other widths. These are descriptions of measured shape classes, not
backend capability restrictions. Expand to other selected fusions only after
reviewing this first class and its paid decomposition. No whole-model speed
claim follows from isolated operation timings.

## Original v8 registration (closed without timing)

The original v8 registration below is historical. Its launch attempts produced
no timing; the v11 registration in the next section governs execution.

The reviewed v8 source checkpoint and its native binary were frozen before this
registration. This registration adds their execution identities; the frozen
source snapshot retains its original protocol text. No measurement has run.

| Identity | Value |
| --- | --- |
| Source base | `dfba7a78cd6f6a821148d3cc434ab182153ece7c` |
| Explicit wave-J dependency patch SHA-256 | `2cfe12da558abb5c84fe2b3b4d1a0641c5edcebe739e3bd847545c093054e775` |
| Reviewed U6 v8 patch SHA-256 | `a710843e8001c5844412af83cf25e3198425c2a2f04dbdc21e71a82bc9434125` |
| Complete 2,778-file source manifest SHA-256 | `d4fc63f59700340efd9737f6fffed32f69bd62d3fcef84a8ac55355e0e34e642` |
| Native benchmark SHA-256 | `7e50e9411abaf714bf1d8a5c8fc99228b2515a63ac4045001c4b95621feda5b6` |
| Launch helper SHA-256 | `6b035d31bf8e0ff9b754c1efd534f30d5c8aa6751026d907bbb6c45cda982f69` |
| Host / GPU / driver | `spark` (`spark-c4e2`), NVIDIA GB10, `580.178.04` |
| SDK | `aarch64-2c4eb5b2f68f3fa5` |

The source base, dependency and slice patch identify the compiled source. The
retained remote checkout's `61f8426f.dirty` label in its build receipt describes
that checkout, not this compilation. All six narrow native targets and the
focused rank, planner, execution, wrapper-launch, source-digest and generator
checks passed before registration; independent whole and adversarial reviews
accepted v8 after repairing the latency denominator and repeated-root proof.

Run only after this registration is committed. The installed Spark supervisor
runs one exclusive job, `m35-u6-first-1024x1-v8-r1`, with a 600-second limit and
30-second termination grace. Its command is pinned Python with bytecode disabled,
`/tmp/u6-first-screen-v8.py --prereg-commit COMMITTED_REGISTRATION`, in
`/home/pmeenan/src/llmp`. The helper authenticates all 2,778 source files and the
benchmark before launching the exact `llmp_fusion_ranks 1024 1` command through
mise; it authenticates the binary again after retirement. No build or source sync
may change this snapshot before execution.

The helper and manifest are staged at the exact paths above and
`/tmp/u6-v8-first-screen-source-manifest.json`. The private, fresh output directory
is `~/.local/share/llmp/m35-u6-fusion-ranks/first-1024x1-v8-r1`: `launch.json`
retains identities, validation and process completion; `samples.jsonl` retains
all flushed harness events; `stderr.txt` retains diagnostics. The wrapper gives
the harness 540 seconds, signals its owned process group on cancellation or
failure, and reaps its child with one 20-second TERM wait and at most five further seconds after
KILL inside the supervisor's 30-second grace. Cancellation during launch waits
for child ownership; repeated signals cannot restart cleanup. Final receipts
retain retirement and binary-postcheck failures, and only successful completion
sets `completed=true`. The supervisor's post-job GPU-process check remains
required. Failed or refused evidence is retained. Review
this first result before expanding the prospectively listed matrix or installing
rank data. This operator screen makes no model-speed or whole-U6 coverage claim.

## Repaired v11 first-screen registration

**Prospective:** run only after this registration is committed and reviewed.
No performance samples exist. The first A supervisor job failed before the
helper because its invocation used a relative mise path and omitted the
registered working directory. The corrected invocation authenticated all 2,778
source files and the v8 binary, then the harness refused a zero-byte backend
workspace during setup. No warmup, timing or correctness result was collected.
Both failures remain in the external first-screen receipt directories.

The repair uses the production `DeviceBackend` factory's full declaration,
workspace binding and launch sequence. It obtains the actual library and selected
persistent backing requirements. The fixture has one owner and both planner
arms start from its own expanded IR: normal rewriting reconstructs the fused
arm with production provenance; disabling only `rms_norm_mul` leaves its
primitive arm. The actual final fused query must equal the reported typed
fixture query. These are harness corrections, with the registered arithmetic,
shape, inputs, epsilon, correctness, sampling, drift and decision bands unchanged.

Before this registration, native job `u6-fusion-v11-validate` on spark-b passed
`--validate-only 1024 1` through complete setup, eager output, capture, replay,
comparison and teardown. Both output differences were zero; captured graphs
contained 128 fused and 256 primitive nodes. The mode creates no timing events,
collects no samples and emits no measurement row. Its final completion receipt
followed teardown. Earlier v9/v10 validation refusals for missing rewrite
provenance and joined input ports remain preserved.

| Frozen execution identity | Value |
| --- | --- |
| Source base | `dfba7a78cd6f6a821148d3cc434ab182153ece7c` |
| Explicit wave-J dependency patch SHA-256 | `2cfe12da558abb5c84fe2b3b4d1a0641c5edcebe739e3bd847545c093054e775` |
| Reviewed U6 v11 patch SHA-256 | `ed7439d4e71f83698e3723a1508a5ec4f09186336c36f713b258348dc19e1756` |
| Complete 2,778-file source manifest SHA-256 | `b01ba89bdd4689caccca6c5aa5d20e12efbc375bd9bbd567f537082148b0c978` |
| Native benchmark SHA-256 | `6e9bcdebd527fdf2ea4dd63fe685d2254a064725704a6b2a58430e33bc4c3b46` |
| Launch helper SHA-256 | `667bdf81939430fe247bb8f04a79775339b67b4846ab1eefd7e911573d952cda` |
| Host / GPU | `spark-b` (`spark-56f5`), NVIDIA GB10 |
| SDK / driver API / runtime API | `aarch64-2c4eb5b2f68f3fa5` / `13000` / `13040` |
| Library / selected persistent backing | 33,554,432 / 21,015,040 bytes |

This source snapshot retains its pre-registration protocol text. Its exact
source chain and manifest identify the compiled source; the inherited remote
`61f8426f.dirty` stamp does not identify this build. The production rank code
remains the whole/adversarial-reviewed v8 implementation. All six v8 native
targets and focused checks passed; v11 rebuilt the changed harness and passed
its complete functional validation.

Stage only the authenticated helper and manifest at
`/tmp/u6-first-screen-v11.py` and
`/tmp/u6-v11-first-screen-source-manifest.json`. Preserve the frozen source and
binary. Run exactly one exclusive 600-second job with 30-second termination
grace, using absolute installed tool paths and the registered working directory:

```sh
~/.local/bin/spark-job start --gpu --name m35-u6-first-1024x1-v11-r1 \
  --timeout 600 --grace 30 -C /home/pmeenan/src/llmp -- \
  /home/pmeenan/.local/bin/mise exec -- python3 -B /tmp/u6-first-screen-v11.py \
  --prereg-commit COMMITTED_V11_REGISTRATION
```

The helper retains the reviewed v8 bounded lifecycle and source/binary
validation, substituting only the v11 identities and fresh output path
`~/.local/share/llmp/m35-u6-fusion-ranks/first-1024x1-v11-r1`. Keep `launch.json`,
all `samples.jsonl` events and `stderr.txt`, including any refusal, and check
that the completed job leaves no GPU process. Review the first result before
expanding the already listed matrix or installing any measured rank. No model
speed, arithmetic qualification or full-U6 completion follows from this screen.

## Prospective v12 capacity amendment for the last matrix case

The reviewed v11 first screen and ten additional matrix cases produced eleven
valid measurements. The twelfth case, width 4096 and 256 rows, refused its
argument guard before backend setup and produced no timing. Preserve that
refusal and every earlier attempt. No measured rank has been installed yet;
the first eleven results are being audited separately.

The v12 harness replaces its blanket factor-16 input estimate with overflow-safe
per-input F32 bounds. Before lowering, each arm checks its actual production
placement extent against the unchanged 32 MiB activation arena. Before binding
or launching, it also checks the lowered activation and kernel-pool requirements
against that arena and the unchanged 16 MiB pool. No kernel, implementation
identity, input, arithmetic, sampling order, drift bound or decision band changes.
The other 2,776 source-manifest entries are byte-identical to v11.

Before timing, supervised shared jobs `u6-fusion-v12-build`,
`u6-fusion-v12-validate` and `u6-fusion-v12-negative-controls` on spark-b passed.
Full `--validate-only 4096 256` reported logical placement 8,404,992 bytes for
both arms, actual lowered activation 8,404,992 bytes fused and 12,599,296 bytes
primitive, and zero kernel-pool bytes. Output differences were zero, captured
nodes 128/256 and replay equality true; completion followed teardown. No timing
event or sample was created. Two oversized individual inputs refused before
backend setup; a third case whose inputs fit separately refused its 67,125,248
byte production placement before lowering. These retained controls validate
resource refusal, not performance.

| Frozen v12 execution identity | Value |
| --- | --- |
| Source base | `dfba7a78cd6f6a821148d3cc434ab182153ece7c` |
| Explicit wave-J dependency patch SHA-256 | `2cfe12da558abb5c84fe2b3b4d1a0641c5edcebe739e3bd847545c093054e775` |
| Full U6 v12 patch SHA-256 | `43373c288f15490f21e32325212f9a593775e2c09d1531324a718d3dbd94db8a` |
| Narrow v11-to-v12 repair SHA-256 | `c67f5451dfbf786b2bda35f6379aa1f3db1cdcf80dca206c248ee751b99e5856` |
| Complete 2,778-file source manifest SHA-256 | `acfb562360004d70ba11d80e09a0cde9dc594aee6925520587c7d89276456b98` |
| Native benchmark SHA-256 | `ecb80b29f4df1271feaeba9ba0f950b61c5e6c6837207385370cabcca74b610b` |
| Last-case launch helper SHA-256 | `38a2b379c9d9f99428f4c3b5b3e720120cb480b41e598699e38d9f98b54e570a` |
| Host / GPU / SDK | `spark-b` / NVIDIA GB10 / `aarch64-2c4eb5b2f68f3fa5` |

Run only after independent review and this amendment's commit. Stage the
manifest at `/tmp/u6-v12-last-screen-source-manifest.json` and the helper at
`/tmp/u6-last-screen-v12.py`. The helper differs from the reviewed v11 helper
only in frozen identities, output directory and the exact last-case arguments.
It authenticates every manifest entry before execution and the binary before
and after retirement. The inherited remote dirty Git label is not build identity.

```sh
~/.local/bin/spark-job start --gpu --name m35-u6-matrix-4096x256-v12-r1 \
  --timeout 600 --grace 30 -C /home/pmeenan/src/llmp -- \
  /home/pmeenan/.local/bin/mise exec -- python3 -B /tmp/u6-last-screen-v12.py \
  --prereg-commit COMMITTED_V12_REGISTRATION
```

The fresh output is
`~/.local/share/llmp/m35-u6-fusion-ranks/matrix-4096x256-v12-r1`. Preserve its
launch, flushed samples and diagnostics, including refusal or cleanup failure;
check post-job absence of GPU processes. Existing v11 measurements are retained
and are not rerun. This amendment authorizes only the remaining prospective
matrix tuple; it establishes no model-speed or full-U6 conclusion.

## Data and source identity

`execution/fusion_ranks.h` distinguishes unmeasured, stale, neutral, fused-win
and decomposition-win records. Its exact query identity covers all source and
output encodings, layouts, encoding parameters, dimensions, strides, offsets,
alignment/alias facts, operation parameters, immutable device facts and
selection requirements. Automatic-rewrite bookkeeping is not an arithmetic
fact. Operand ordering is canonicalized; unknown/duplicate facts do not match.
No model name, checkpoint shape predicate or vendor name lives in execution.

The first comparator contract is deliberately nonrecursive: only immediate
own-decomposition children that are terminal, stateless single-node operations
with distinct natural source roots qualify. Repeated source roots remain
unmeasured: cloning them must not erase an alias relation absent from the query key. The planner constructs that chain and checks its
current ordered selections at **final selection**, after rewrites, input
preparation, layout changes and pruning. The fallback expansion does not rerun
those passes. Nested fused/composite children, views/layout operations, state,
batching anchors, implementation groups, expansion and pitched outputs remain
unmeasured. Each selected child must also declare `standalone_comparator`: its
backend guarantees no neighbor can fold or replace its launched work. This is
a reusable capability contract, not a fusion or checkpoint whitelist.

Initially GGML RMSNorm and Mul carry that audited contract. Lowering may wrap
those adjacent launches in `ggml.rms_norm_mul.unfused`; its recipe still calls
RMSNorm then Mul as two GPU launches with unchanged arithmetic. The harness
records the logical chain and this physical bound recipe separately, checks
the exact recipe, and requires two captured nodes per complete primitive
chain. That recipe is included in the selection-environment identity too.
The launch audit is `Lowering::Build`'s ordered wrapper, `CheckRmsNormThenMul`
in `validate.cc` (exact dependency and disjoint intermediate checks), and
`RmsNormThenMul` in `ops.cu`: it passes the original norm descriptor/epsilon,
then the same `ShiftBinaryLeft` normalization as standalone Mul, to the same
upstream launchers on one context/stream. This slice aligns that normalization
for width-one tensors too; no shape exemption is used.
Other primitive implementations require their own launch-contract audit before
comparisons qualify; all selected-fusion/shape-class coverage remains open.

The selection-environment digest includes every registered implementation
identity/capability and every ordinary device-rank record. Registry ordering is
canonical by name, matching the candidate tie-break; priorities and eligibility
metadata are in each identity. Query identity includes semantic policy options,
explicit rank order and all tuning facts, including `all_rows`. It excludes
checkpoint/spec/source-quality evidence handles. Comparison data is excluded
from this environment to avoid a self-cycle: qualifying children are terminal,
so changing a nested fusion's preference cannot influence their selection.
The current ordered chain must independently match the recorded identities;
without that proof, a query is explicitly unmeasured.

Each declarative JSON comparison owns the fused implementation identity and
**every implementation in the ordered primitive chain**, measured medians,
sample count, selection-environment digest, descriptive shape label and report provenance. Source changes
make a comparison stale; missing evidence does not mean measured-neutral.
Complete comparison contents participate in `RankDataDigest`, registry digest
and the existing plan-cache identity. The registry rejects malformed, unbound
or overlapping comparisons.

The data has a deliberately separate identity from implementation sources:
`tools/fusion_rank_data.py` accepts only the documented JSON fields and emits
C++ values, never source text supplied by the data. This generator, the IR
own-decomposition/builder/inference sources, execution selection/serialization
logic, planner final-selection/pattern sources and all existing backend code
remain in the backend module source digest. The exact inventory is `module_files`
in `src/kernels/ggml/CMakeLists.txt`. Only `data.json` is excluded from that source digest;
its complete semantic contents enter the registry rank digest. Thus inserting
measured implementation identities into the data does not create a circular
hash, and changing measurements invalidates plans without changing the
implementations that were measured. No runtime file reads or new dependencies
are introduced.

Before accepting a measurement, record the Git base, dependency patch and
slice patch SHA-256, native build receipt, supervised job receipt, host,
driver/toolkit, and the verified harness result in this report. Raw samples and
logs remain in external scratch. Replace the harness's `UNREVIEWED` provenance
with the reviewed report reference before inserting any comparison in
`data.json`. A syntax-valid record is not evidence that its measurements ran.
