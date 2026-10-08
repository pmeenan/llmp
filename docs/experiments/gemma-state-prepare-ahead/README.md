<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prepare fresh Gemma3 state beside the current prefill chunk

Gemma3 now prepares eligible fresh zero-backed state for the next known chunk
beside current model work, without evicting cache or publishing a future cursor.
The ordinary 128-row C2 screen reduces paid prefill by 2.617% and combined
prefill/decode by 1.941%. A separate configured 256-row screen reduces them by
2.081% and 1.390%. Each has two samples per mode, exact complete outputs and
initialized states, and identical planning/kernel work. These are bounded
same-native measurements, not a reference-parity or larger-context speed claim.

The shared mechanism and ordinary Gemma3 fresh-zero policy are adopted.
T67 remains composite OPEN: preparation from kept restart files or fresh sparse
file reads, larger-context qualification and other growing LLM adapters remain
compatible work. Public chunk/context/cohort defaults are unchanged. There is
no new public setting.

## Paid comparisons

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
replays occur, with 338 Q8 preparations and 880 prepared-product selections.
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

DeepSeek, native/GGUF Qwen, Gemma2 and both Gemma4 profiles can reuse the arbitrary
range/source protocol but have no adapter or measured T67 gain in this slice.
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
