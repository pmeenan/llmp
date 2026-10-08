<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prepare fresh Gemma state beside the current prefill chunk

Gemma2 and Gemma3 now prepare eligible fresh zero-backed state for the next known
chunk beside current model work, without evicting cache or publishing a future cursor.
Gemma2 ordinary 128-row C2 at configured context 8192 reduces paid prefill by
4.126% and combined prefill/decode by 2.905%; exact default serving and real
joined-wrap lifecycle controls qualify its adapter.
The Gemma3 ordinary 128-row C2 screen reduces paid prefill by 2.617% and combined
prefill/decode by 1.941%. A separate configured 256-row screen reduces them by
2.081% and 1.390%. Each has two samples per mode, exact complete outputs and
initialized states, and identical planning/kernel work. These are bounded
same-native measurements, not a reference-parity or larger-context speed claim.

The shared mechanism and ordinary Gemma2/Gemma3 fresh-zero policies are adopted.
T67 remains composite OPEN: preparation from kept restart files or fresh sparse
file reads, larger-context qualification and other growing LLM adapters remain
compatible work. Public chunk/context/cohort defaults are unchanged. There is
no new public setting.

## Gemma3 paid comparisons

Order is preparation off / on / on / off, with a fresh process per arm and the
same actual binary within each comparison. Values are seconds.

| Recipe / endpoint | O1 | A1 | A2 | O2 | Mean change | Off bookend movement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary 128-row prefill | 1.133000 | 1.093040 | 1.104590 | 1.123680 | −2.617% | −0.823% |
| Ordinary 128-row decode | 0.487855 | 0.484128 | 0.483558 | 0.483447 | −0.372% | −0.904% |
| Ordinary 128-row combined | 1.620855 | 1.577168 | 1.588148 | 1.607127 | −1.941% | −0.847% |
| Configured 256-row prefill | 1.056950 | 1.043370 | 1.033160 | 1.063720 | −2.081% | +0.641% |
| Configured 256-row decode | 0.491900 | 0.492138 | 0.492159 | 0.491395 | +0.102% | −0.103% |
| Configured 256-row combined | 1.548850 | 1.535508 | 1.525319 | 1.555115 | −1.390% | +0.404% |

Ordinary 128-row prefill ranges are off 1.123680–1.133000 and on
1.093040–1.104590; configured 256-row ranges are off 1.056950–1.063720 and on
1.033160–1.043370. Both prefill and combined ranges do not overlap. Candidate
prefill movement is +1.057% at 128 rows and −0.979% at 256 rows. The ordinary decode
change is inside off movement and is **not a decode gain**. Both sample counts
are small; [results.json](results.json) retains all endpoints, means, ranges,
movement, exact-output identities and necessary provenance.
`python3 compare.py` recomputes the aggregates and checks their stored values.

Both comparisons use Gemma3 4B QAT Q4_0, context 4096, two owners, prefixes
3072/3584 and 32 joined decode units/64 chosen IDs. The chosen execution budget
is the same 34359738368 bytes (32 GiB) in every arm, with actual derived startup minimum
checked below it. Both modes select root attention, device masks, shared-Q8,
capacity-two lookahead and capture-ahead. The only difference is state
preparation. The 128 recipe has per-owner state_max_rows=128, wave capacity 256,
local ring 1280, 23 joined groups/5888 rows and six scalar groups/768 rows. The
256 recipe explicitly uses stock-ring, state_max_rows=512, wave capacity 512,
local ring 1536, 11 joined groups/5632 rows and four scalar groups/1024 rows.
Neither comparison changes public admission.

Each process loads the model, seeds three rows per owner, then Clear+DropPlans
before paid prompt execution; zero cold plan bytes/graphs are checked. Startup,
model loading and the common fixed three-step decode warmup are excluded.
State growth/zero-fill, optional submission and scoped retirement, CPU planning,
graph capture and current cursor publication are included in prefill time.
There is no extra preparation warmup. Final heads and state hashes are captured
outside the paid interval for both modes.

All non-preparation, non-timing summary fields match across each factor. At 128 rows,
13 future plans are built and cached, 13 capture-ahead operations and 25 prefill
replays occur, with 338 Q8-preparation and 880 prepared-product selections across bound plans.
Those bound-plan selection counts are not executed-kernel counts.
Pair count is zero because adjacent hints can share a complete padded shape;
this is valid deduplication, not missing work. At 256 rows the respective future
counts are 11 built/cached, one pair, 10 capture-ahead and 10 prefill replays,
with the same Q8/product counts. Each candidate completes and later adopts 166
extents in five submitted acquisitions; neither mode has a preparation failure
or capacity refusal. Ordinary 128-row attempts 47 and configured 256-row attempts 21;
these counts include predictions needing no new extents. Off modes prepare none.

Each arm returns 64 valid IDs and two complete finite 262208-vocabulary rows,
with byte-identical choices/heads and both complete initialized-state hashes
across its four arms. Final positions are 3107/3619. These are per-factor exact
identities; different ring layouts are not compared across the 128/256 factors.
Every process exits 0, the installed supervisors positively retire, boot remains
stable, GPUs are empty after retirement and observed kernel messages are empty.

## Gemma2 transfer: ordinary 128-row at configured context 8192

The same shared ticket/drain protocol now serves Gemma2. This is one fixed
preparation off/on/on/off comparison with two samples per mode, using the same
actual ELF and exact non-preparation planning/kernel work. Values are seconds.

| Endpoint | O1 | A1 | A2 | O2 | Mean change | Off bookend movement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill | 1.286450 | 1.240780 | 1.236820 | 1.297780 | −4.126% | +0.881% |
| Decode | 0.581019 | 0.581126 | 0.579140 | 0.581472 | −0.191% | +0.078% |
| Combined paid | 1.867469 | 1.821906 | 1.815960 | 1.879252 | −2.905% | +0.631% |

Prefill ranges are off 1.286450–1.297780 and on 1.236820–1.240780;
combined ranges are off 1.867469–1.879252 and on 1.815960–1.821906. They do
not overlap. Candidate prefill movement is −0.319%; combined movement is
−0.326%. Decode ranges overlap, so this is **not a decode speed claim**.
The small n=2 result is bounded to this own-engine recipe; it is neither a
stock/TensorFold parity result nor a claim about every admitted context.

Both modes use the approved Gemma2 2B Q8_0 artifact, context 8192, ordinary
per-owner chunk/state_max_rows 128, wave capacity 256 and local ring 4352.
Standing input lengths are 4391/4903; paid prompt prefixes are 4352/4864,
followed by the same fixed three-step decode warmup and 32 joined paid decode
units/64 IDs. The chosen common execution budget is 34359738368 bytes (32 GiB),
above the actual derived minimum 8604565504 bytes. Root attention, device
masks, shared-Q8, capacity-two lookahead and capture-ahead match; only
prepare-state differs. Startup/model load and common warmup are excluded;
state growth/zero-fill, optional preparation submission/drain, CPU planning,
capture and current publication are paid. Off-paid final payload checks are
identical. Cold seed-three/Clear/DropPlans verifies zero plans/graphs before
paid prefill; there is no extra preparation warmup.

Actual work is 33 joined state-only groups/8448 rows and six scalar groups/768
rows, 18 built/cached future plans, 18 capture-ahead operations, 35 prefill
replays in every arm. Separately, bound-plan counters record 258 Q8-preparation
selections and 672 prepared-MMVQ-product selections; these are not executed
kernel counts.
Each candidate attempts 67 predictions and submits eight acquisitions, then
completes and adopts 416 extents with zero refusals/failures; off counters are
zero. All other non-timing summary fields and actual budget dictionaries match.
The last joined wave ends at 4224, below local ring 4352; ring crossing in this
factor occurs during scalar owner departure. The separate lifetime control,
not this timing factor, proves a real joined wrapped wave at 4608 positions.

All four arms have byte-identical complete finite 256000-vocabulary heads,
all 64 choices and both initialized-state hashes, ending at 4387/4899.
Processes and installed supervisor retire positively; boot is stable, observed
kernel messages are empty and GPU is empty after retirement. The measured
source is 799a378f… and probe c8f2d85e…; complete identities and aggregates are
in results.json. No failed heavy attempt is pooled into this factor.

The recipient lifetime control compares actual packed/unhinted/off against
root/hinted/on for 36 joined 128-row waves, authenticates local ring 4352 and
real wrap at 4608, and preserves complete heads/state, checkpoint advance and
rewind, protected peer, spill, fresh kept restart and full next-head identity.
It completes/adopts 416 extents; checked receipt is 15754ac6… . Kept restart is
read normally, rather than initialized by optional preparation.

The final actual Server/PromptSession control compares explicit off with a
fresh unset/default Server. Prompt extra_rows=3072 gives owner lengths
4352/4480 at context 8192/local ring 4352, ensuring real backing growth.
Both runner and internal ServingOptions defaults are true; OFF has zero
preparation counters, while default completes/adopts 416 extents with no errors,
33 joined groups/8448 rows, exact full finite heads/history/initialized state
and current cursors, healthy owners and a drained registry. Both Servers
explicitly tear down successfully. Its checked receipt is a530d06d… and final
qualified source 8c719c9f… . This uses the ordinary startup budget; the timing
factor's chosen 32 GiB is separate. The two new G2 cases are additional to the
historical G3 19 executions/18 unique cases; unchanged shared controls were not
rerun. Default/wiring changes preserve the explicit measured OFF/ON semantics;
the final probe ELF is still c8f2d85e… . No further timing/reference ladder was run.

## Ownership and retirement

A bounded LiveState-owned ticket authenticates all requested fresh ranges,
current extent generations and original zero sources before submitting a
no-victim AcquireProgram **before** the current Job. Its descriptor allowance
is `4096 + 256N` bytes, charged before optional allocation without AskReclaim.
Backing remains charged to the node budget. There is no new thread, nested
node call in meanwhile, useful-cache eviction, future token execution or
logical-state publication. The existing meanwhile callback stays host-only.

After the current fence, Acquire's ProgramDone.gone proves only task destruction:
withdrawn scheduler-owned page-ins may still be active. A separate scoped drain
joins existing loads for the ticket's exact typed IDs, including queued and
mailbox-blocked stages, without restarting missing loads or waiting for unrelated
loads. Absence from the scheduler load map is safe only with authenticated
Resident/Nonresident catalog state; quarantined/unknown completion is not proof.
Cancelled zeroing may legitimately finish resident. All partial completions,
the whole ticket, host charge, spill source and LiveState owner remain retained
until positive scoped retirement. Completed unused zeros are reclaimable and
unpublished; actual Use alone marks logical ranges used and advances the caller.

Clear and other mutators drain first. Node teardown drains linked owners before
collecting managed extents. On unproven preparation it refuses while retaining
the complete owner; the stack-owned Runtime Server aborts before returning or
running any member destructor. The failstop guard is qualified with actual
mapped backing and registry ownership in an isolated child, rather than relying
on a later LiveState destructor assertion. Ordinary successful teardown is
unchanged. Kept restart sources are never converted to zero or discarded as
fresh preparation, and repeated RegisterSpill refuses before changing the
original source or settling an outstanding ticket.

## Focused qualification and limits

Fifteen prerequisite cases cover unpublished/adopted/reclaimed zero backing,
whole-range/source validation, no-victim/headroom/overflow refusal, partial and
cancelled loads, queued scoped drains/unrelated loads, generation/source changes,
held invalidation, repeated registration and actual nonzero kept-file reads.
A real wrapped C2 current-write/future-refusal control quarantines **both** current
owners before cursor publication. Cancellation exposed a genuine task-gone versus
page-in-retirement gap; its failed attempt remains unpooled and the scoped drain
repairs it. A later guard fixture failed to compile on private state access and
ran no tests; its public-API recovery remains separately recorded. An earlier
launcher path error ran no controller, build or tests. No failed attempt is
pooled into the positive checks or timing samples.

Three further checks prove normal registry drain, actual Server failstop on a
synthetic unknown catalog outcome with mapped backing retained, and a prepared
128-row packed/root comparison crossing local ring 1280 at 1536 positions. That
control matches complete heads and initialized state, same-owner checkpoint
advance/rewind, protected peer, spill, fresh kept restart and full next head;
136 completed extents are adopted. Its synthetic unknown case issues no actual
provider eviction and claims no false physical-retirement proof. Existing strict
256 and 512 fixtures retain their original assertions.

The final actual PromptSession/RunPromptWave control compares explicit off with
fresh ordinary defaults, checks both runner and ServingOptions defaults, positive
preparation/adoption, exact histories/full heads/full state, equal joined work,
current cursors and healthy/drained owners. It completes and adopts 136 extents,
with 9 joined groups / 2304 rows and no pending preparation owner. Its checked receipt is
`7f48d9ef6a29bc1dadd02bf628a83258c7c1f0b460b443bd4a99b9d85075318f`;
final source manifest is
`b78ad0e89eea45ff3d14d532471dba6c5422a3f9f078cb685aff9df9d7efc5e7`.
The actual default test uses the runtime startup budget; the chosen 32 GiB
budget belongs to the timing comparisons. Positive executions total 19 with 18
unique cases because the normal teardown case is intentionally rechecked after
adding the registry assertions. Full suites, package/shipment checks, other
family/context profiles and new reference measurements are deferred.

DeepSeek, native/GGUF Qwen and both Gemma4 profiles can reuse the arbitrary
range/source protocol but have no adapter or measured T67 gain in these slices.
Image has fixed generation buffers rather than increasing token-bound state.
Gemma3 larger admitted contexts are not artificially blocked by a 4096 selector;
performance/context controls remain open. Fresh sparse-file initialization can
be extended with explicit provenance/authentication; source.zero=false alone is
not incompatibility. Kept restart requires preserving its authoritative bytes.

## Reproduction and identity chain

Use the retained `jitllm_gemma3_joint_prefill_probe`, fresh output directories,
and the installed supervised Spark GPU job protocol. Prepare or retrieve the
standing 3111/3623 ID files using the frozen source texts and native-tokenizer
recipe in [prefill transfer](../prefill-transfer/README.md#two-distinct-future-shapes-2026-10-08).
The source texts are engine.md/async-model.md from commit 46cfd5c7; that report
records exact truncation, tokenizer metadata and ID hashes. Approved artifact
manifest SHA is 8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb;
index SHA is 4422d9c5a115e9a5091a471eba1db5a232f825956c5e8e77be2c77c815c093e6.
Input hashes are e550a181088bc3d4924de3d45091067a38cd75c7c9ee676947e8880334a83c7c
(12444 bytes) and c8e853512376e0148c7c5798974bd91e42849c918f8f23017fb3e22eef23cfee
(14492 bytes). No disposable raw bundle is required to recreate the workload.

For ordinary 128-row, run the following argv in off/on/on/off order, appending
`prepare-state` only to the on arms. Absence of the flag explicitly sets the
benchmark's control false despite the final runner default true:

```text
jitllm_gemma3_joint_prefill_probe ARTIFACT IDS0 IDS1 FRESH_OUT first-cycle
  bounded-roots device-masks prefill-ahead owner-prefill flexible-owner-prefill
  chunk=128 lookahead-capacity=2 shared-q8 budget-bytes=34359738368
```

For the separate configured 256-row replay, replace chunk=128 with `chunk=256
stock-ring`. Enforce the actual summary geometry/work counts above, equal
actual budget dictionaries, positive on preparation/adoption, zero off counters,
complete finite full heads, all 64 IDs and both initialized-state hashes exact.
Require source/receipt/actual ELF/first-resolved pinned cuBLAS/input/artifact
metadata checks before and after, MemAvailable >= 48 GiB and empty GPUs, stable
boot/kernel guards, bounded 145-second children, owned cancellation cleanup and
positive supervisor retirement. A failed guard is retained and investigated,
not silently retried or omitted.

Configured 256-row used source manifest ba7277e8…/probe d0dd38d7…;
ordinary 128-row used 5c73b712…/probe 1594165e…; both actual official SDK receipts are
0296e41b…. Complete identities, all work counts and output hashes are retained
in results.json. Final defaults/wiring change only four source paths from the
ordinary 128-row proof; the benchmark explicitly assigns its flag in both modes,
so measured option semantics remain unchanged. Final actual Runtime qualification
binds the final ELF/source and proves the unset ordinary path. No factor is
relabelled as using that later binary.

Task-entry TensorFold main f8fe17d2 and retained python-0.6 ed78d6fc README pins
were checked on 2026-10-08. Native Gemma4/DeepSeek and retained Python CUDA support
are applicable references, but neither lists this Gemma3 recipe. No new
TensorFold or stock competitive-parity claim follows from these own controls.

## Gemma2 reproduction and source bridge

Use `jitllm_gemma2_joint_prefill_probe` in fresh supervised output directories,
with the approved artifact and the standing two distinct ID files from
[prefill copies](../gemma-prefill-copies/README.md). Artifact manifest SHA is
eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870 and
index SHA is f1773c66b42084e0f497ed406716639800aaaccef0db49bd5992962f137add62.
ID hashes are 102c7b555b1caed5aed3d9880a173aae153f8f8dc1534e6a4c58685c243b6c0c
(17564 bytes/4391 IDs) and
a930726bd964ae88ef0448f50a51d2e376ce2487313f26063ab84c1b0b41d77f
(19612 bytes/4903 IDs). The linked report records frozen source text and native
tokenizer preparation; no disposable raw evidence is needed for replay.

Run off/on/on/off with the following argv, appending `prepare-state` only to
on arms. Its absence explicitly assigns false despite the adopted true runner
default. There is no stock-ring flag.

```text
jitllm_gemma2_joint_prefill_probe ARTIFACT IDS0 IDS1 FRESH_OUT first-cycle
  bounded-roots device-masks prefill-ahead owner-prefill flexible-owner-prefill
  chunk=128 lookahead-capacity=2 shared-q8 budget-bytes=34359738368
```

Enforce exact geometry/work and equal actual budget dictionaries above, positive
on and zero off preparation, complete finite full heads/all 64 IDs/both initialized
states exact, and the same source/receipt/actual ELF/first-resolved pinned cuBLAS,
input and artifact metadata pre/post checks. Use bounded 145-second children,
MemAvailable >= 48 GiB/empty GPUs, stable boot/kernel guards, owned cancellation
cleanup and positive installed supervisor retirement. All preparation costs are
paid in prefill; final payload validation is outside the timer for both modes.
The shared `compare.py` replays all three retained factors.

The Gemma2 task-entry TensorFold main f8fe17d2 and retained python-0.6 ed78d6fc
were checked again at 2026-10-08T13:47:48Z. Their native Gemma4/DeepSeek and
retained Python CUDA support are applicable references; neither advertises this
approved Gemma2 serving recipe. The check adds no competitive-parity claim.
