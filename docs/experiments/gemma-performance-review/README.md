<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma implementation handoff — 2026-10-06

The owner requested a stopping point for Opus to close the remaining gaps.
The tested bounded Gemma31 serving recipe is committed as `ee6beb1`; M3.5
implementation and measurement are paused at this handoff. Both `spark` and
`spark-b` were checked free with the installed `spark-job busy`: no running or
waiting GPU jobs and no GPU compute processes. No new capture-policy build or
inference started. The partial experiment described below remains external.

## Current competitive evidence

Use the [current serving bridge](../gemma31-serving-bridge/README.md) and its
[aggregate](../gemma31-serving-bridge/results.json) as the latest whole-serving
Gemma31 comparison. The reference is official llama.cpp v0.6.0/d81235049384534c167caea52b85a694f6103d14,
with its authenticated ARM64 CUDA image and actual initialized-driver proof.
Older b29c606/b10964 results remain historical controls, not the current target.

| Approved model / workload | Native mean | Reference mean | Native elapsed excess |
| --- | ---: | ---: | ---: |
| Gemma31 C1, 8K continuation / 128 completed decode waves | 23.927816 s | 23.197650 s | 3.147588% / 730.166 ms |
| Gemma31 C4, four independent 8K continuations / 128 completed joined waves | 61.316722 s | 59.234650 s | 3.514957% / 2.082072 s |
| Gemma26 current-release C2 representative short screen | 0.778420 s | 0.762009 s | 2.153583% |

The Gemma31 cycles include Clear, prefill, frontend selection/sampling,
publication, completion and request retirement. Order is reference/native/
native/reference in fresh processes, with preceding warm cycles. C1 prefill
means are 11.218646/10.763300 s and decode/finish 12.709170/12.434350 s;
C4 means are 44.903326/43.275300 s and 16.413396/15.959400 s. These are short
matched observations, not sustained parity or kernel-only timing.

Gemma31 C1/C4 have zero predicted-ID differences and zero positive-margin or
tied-choice differences; 128/129 and 512/516 complete heads are byte-exact.
The aggregate does not assign the nonexact vectors to a phase. Independent
128/512-target generated-history conditional-loss gates pass. The current
scalar corpus matches all 1,024 full heads exactly over 1,023 authentic
next-token targets. Its high absolute perplexity is diagnostic; parity is not
broad semantic-quality qualification. Complete native own repeats include
initialized state, final head, layout, history and generated IDs.

The natural HTTP controls pass literal continuation, chat/SSE agreement,
stop suppression, disconnect/recovery and client-observed peer progress after
one client departs. They do not prove precise backend cancellation chronology
or a four-owner GPU shape for every wave. Four settings and three calibration
controls pass without skips; the ordinary default smoke naturally stops after
three tokens, with no candidate forcing or prefill override.

The production selection is deliberately bounded: approved dense31, resolved
context<=8192, max_slots<=4, prefill fallback/cap256, both checked norm chains
and eligible owner attention. Smaller explicit prefill overrides survive.
Gemma26, larger configurations and explicit arithmetic diagnostics retain
their prior recipe and calibration identities. Default context 262144 is not
silently reduced. Thinking/tools, assistants, broader contexts/cohorts,
sustained performance, memory and full shipment gates remain open.

Gemma26's [current-release C2 transfer](../gemma-release-c2-26/README.md)
passes strict choices, its separate conditional-loss gate and complete own
repeats, with 64/66 complete heads byte-exact. It uses matched 992-column
prefill, all30 routing/reduction and F16 rings. It is not a current whole-serving
C1/C4/corpus or default-adoption gate. Historical scalar/corpus and C5 failures
must not be declared resolved by this short C2 result.

## What the current timing diagnosis actually shows

The [Clear/state-growth attribution and third-cycle control](../gemma-state-phase-attribution/README.md)
are committed as `3a0be49` and `302e0c2`. They measure the same private C1
serving recipe, retain complete fixed-own outputs, and keep the original paid
cycle boundaries. They are native diagnostics, not fresh reference bookends.

- The paid phase control has Clear 57.452 ms, state growth 162.182 ms and required
  planning 0.286 ms, with 160 plan hits and zero misses. Repeated planning is not
  the dominant paid cost in this workload. Cold planning remains a separate
  concern for genuinely new shapes.
- The three-cycle control observes warm 33 eager / 1 captured / 126 replayed,
  second 0 eager / 32 captured / 128 replayed, third 0 eager / 0 captured / 160 replayed.
  Retained graphs rise from 1 to 33. The second cycle captures 32 prefill plans,
  paying 86.959 ms capture and 178.513 ms instantiate/upload inside its timer.
- Second/third whole cycles are 23.922937/23.657850 s; prefills
  11.210460/10.932725 s. Both complete paid outputs match the fixed own proof.
  This establishes capture acquisition in the paid second cycle. It does not
  establish that suppressing capture improves acquisition-plus-later-reuse
  cost, or replace the original matched comparison.

Nested counters and device-stream elapsed spans include overlapping work and
host submission gaps. Do not subtract them to invent a residual kernel cost.
Stock's selector waits for consecutive stable graph properties, but its actual
prefill capture counts have not been measured here. Do not assert that stock
never captures prefill.

## First experiment for Opus

Compare fresh capture-ON and capture-OFF native processes using the same
helper and exact C1 recipe. Each arm runs warm → second paid → third paid.
Retain all six cycle times and graph counters; compare the sum of second and
third paid cycles as well as each cycle. Charge all acquisition inside the
existing clocks. Check warm IDs/history and both paid complete initialized
state, head, layout, tokens and history against the fixed own proof in each
arm. This tests both avoided acquisition and lost prefill replay benefit.
A useful native result then warrants a fresh matched reference/native/native/
reference comparison before adopting a policy or claiming gap closure.

The unrun draft is preserved at
`/tmp/jitllm-m35-coordination/gemma-prefill-capture-policy`, branch
`m35/gemma-prefill-capture-policy`, base `302e0c2`; external caller/diff are in
its sibling `gemma-prefill-capture-policy-raw`. Nothing in this experiment is
committed, compiled or qualified. Its two prospective native files add a
capture option defaulting to current behavior, a pre-bind setter that refuses
bound/released runners, and an explicit prefill marker. Suppression requires
an actual multirow segment. Total rows>1 is insufficient: four-owner decode
also has multiple total rows and must retain graph capture/replay. Verify and
existing graph replay remain on their established paths.

Seven runtime files in that scratch tree are borrowed private serving-recipe
bytes, not owned experiment changes. Do not copy its full diff onto main:
main now contains the reviewed bounded default and newer documentation.
Transplant only the intended native delta onto current main and preserve its
ordinary recipe. The external `gemma31_capture_policy.cc` is authoritative;
`benchmark1.diff` predates formatting. Known unfinished work: include
capture-off in the C1-only admission guard, regenerate the caller diff, finish
the arm-aware checker/aggregate and supervised queues, then obtain fresh
source/method review. No queue or source-ready packet is frozen yet.

## Next material lead and backward transfer

Clear currently discards growing state backing. The following growth acquires
fresh extents from a registered sparse-zero source. Its measured Clear/growth
cost makes completion-aware state reuse a concrete next lead if capture policy
leaves a material gap. Simply skipping eviction is unsafe: content generations,
write-back sources, reclaim eligibility and padded initialized bytes must
remain correct. A retained-state design must preserve node-wide charges,
lease/registration retirement, zero reconstruction after reclaim, stale-closure
rejection and quarantine on an unproven reset. Measure its complete reset/growth
cycle; moving allocation out of the benchmark clock is not an optimization.
No retention implementation or provider/IO attribution has been performed.

Apply eligible changes to Gemma26 as requested by the owner. State-only prefill,
lookahead, graph/source indexing, wrapper caching, weight/state placement
memoization, plain norm fusion and slot-sized publication are already shared.
The checked norm chains and owner attention are available, but current whole
serving qualification/default selection remain owed. Gemma26 also uses the
MoE route/reduce policies; dense31 does not. Preserve Gemma26's matched 1024-row
cache geometry when parameterizing the current native/public serving callers;
do not inherit dense31's 256-row recipe silently. Start with representative C1
quality and a PASS-only short whole-serving comparison, then broaden to C4,
corpus and HTTP if warranted. Old C5/scalar failures remain independent.

The [exact MMVQ preparation screen](../gemma31-mmvq-shared-prep/README.md)
reported only about 1% chain improvement with overlapping ranges, not a model
gain. Broad shared-Q8 changes to vector arithmetic were rejected. Do not
revive those changes as if already qualified or prioritize another small
operator screen over a measured whole-engine cost.

## Operating instructions and scope still owed

Start from main at `ee6beb1` plus this handoff commit. Current source/reference
pins are documented above and in the bridge aggregate; recheck latest
TensorFold at each new task entry. The previous task-entry snapshot was
0.6.6/cb2ebf0540f42604e2759b2ddef497861e928248, with no matching dense31
CUDA GB10/GGUF comparator found. This is recipe eligibility, not a claim about
all TensorFold model support. Prefer it where a model/format/platform matches.
The owner also requested a latest llama.cpp/docs refresh after the in-flight
Gemma work is finished, including checking Clef support and performance updates.

Use hostlock shared for workstation builds/tests/inference. Spark heavy work
uses the installed `~/.local/bin/spark-job start --gpu` and official wait,
bounded to 600 seconds by default with stop-on-fail. Recheck busy before handover.
Never touch the owner's TensorFold work. Synchronize owned sources with
`rsync -rlpc --exclude=.git --exclude=/build`; unanchored `--exclude=build`
once omitted `tools/build` before compilation. Failed official records and
raw payloads remain external; the bridge retains 40 successful and 6 failed
records without waiving a numerical failure.

The original serving evidence/raw receipts are at
`/tmp/jitllm-m35-coordination/gemma31-production-bridge-raw`; phase diagnostic
support is at `gemma-state-phase-attribution-raw` in the same parent. Reusable
native/public callers, analysis and HTTP method are committed in the bridge.
Inspect the relevant current receipts rather than revalidating every historical
archive. No full regression suite should run until the gap-closing changes
are settled; use focused correctness, lifetime and matched performance checks.
The final suite/shipment checks remain owed.

M3.5 is incomplete. Its other models/quants, EXL3, media inputs/generation,
Jev/Clef, prefix reuse and final quality/performance/memory/swap gates remain
in [the plan](../../plan.md). DeepSeek 4.1 Flash is on the dual-Spark list.
EmbeddingGemma 2 is recorded as a research/API-assessment candidate (`9970314`),
not executed model support. No M3 speed exception is extended to new models.
