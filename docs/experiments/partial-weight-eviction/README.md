<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Partial weight eviction and exact switch recovery

Status: the internal partial-weight foundation and focused recovery controls
pass, but ordinary partial eviction remains disabled and its adoption gate
stays open. The final current-source full/partial comparison saves 18.3%
prepared first-output time and 19.5% read bytes. Cold first-arrival is 11.9%
slower with nonoverlapping ranges, and one partial cold return also regresses.
That fails the ordinary never-worse-than-full floor. Shared WakeFlag and bounded
completion harvesting retain narrow work-removal benefits; neither is claimed
to repair the intermittent long retirement tail. Historical diagnostic and
candidate factors below remain separate. No further unchanged attribution
ladder or automatic default change follows this decision.

The runtime can keep clean inactive prepared weights at their existing
addresses, make room only for the incoming closure's missing extents, and
reclaim from every inactive model through the existing GreedyDual order.
Whole live-state spill remains the first bounded slice. Releasing the old
request at a completed unit is required even when the incoming model is fully
cached and no extent needs eviction.

Selection uses actual catalog occupancy and missing closure bytes. Source,
generation, lease, registration and protected-closure eligibility are checked
before work and again at execution. Consecutive selected weights are evicted
in a batch; pending selection does not count as physical memory freed. Cache
charges for optional plans and graphs continue to displace only plans/graphs.
Runtime-managed state acquisition now uses the same global reclaim preflight
and cannot silently fall back to the standalone harness's catalog LRU.

Mixed allocation classes need extra care: a parked GPU backing cannot fund
an incoming host allocation. The selected victim policy stays unchanged, but
only the needed incompatible/surplus selected subset is released before the
remaining donors are handed off. Release demand is max(0, occupancy + missing
bytes - compatible donor bytes - budget), including an already-over-budget
host charge. Resident incoming extents create no load demand.

Failed partial activation cleans only incoming additions and restores only
missing outgoing extents. It preserves incoming preexisting weights and
unrelated caches. Spill bookkeeping commits only after readiness checks.
If a successful outgoing spill is followed by both an incoming readiness
failure and a failed rollback load, saved generations are proven and any
undo-restored state is released unchanged before durable spill publication.

read_bytes reports completed positive READ results through the loaded
boundary, including actual storage padding and state reads. It excludes
writes, failed/cancelled/held-unsubmitted reads and device zero-fill. The
reload-rate estimate uses aggregate measured page reads, rather than claiming
a pure weight rate. Later demand reads remain separately reported.

## Focused checks

SparkA final merged integration4, installed GPU-exclusive supervisor,
DONE0/101 seconds:34 checks passed, no skipped/disabled/errors. Memory18,
swap-room6, node4, state1, serving4, cap1. The partial serving controls run
the Gemma2 /0 fixture with a real peer adapter and an unrelated Gemma26
cache; the parameterized Gemma3 /1 partial-recovery case was not run. A
following narrow2-test rebuild/check (integration5,DONE0/36sec) verifies
post-join counter access and actual cap/default/refusal behavior. They cover fully cached switches, old-request
retirement, exact heads, scoped missing-read rollback, durable-record
recreation, late readiness failure, failed undo, and exact recovered full
initialized-state bytes. Managed acquisition covers stale/foreign closure
refusal before reclaim, zero/reentrant callbacks and in-flight refusal.
Mixed host/device handoff, actual credit trimming, batch failure and completed
read accounting have dedicated controls. This is a focused check set, not the
full regression suite.

A separate synthetic CPU selection control ranked100K eligible weight
extents and selected5000 deterministic victims in8605 microseconds in the
integration2 run. This measures selection CPU time only, not a model swap.

The independent challenge found and closed two adjacent late-failure defects:
successful undo initially left stale spill membership/records; failed undo
then lacked outgoing durable publication. Integration3 proves both recovery
paths. No ordinary adoption or all-family performance claim is made.

## Matched comparison: prepared gain, cold regression

Frozen method partial-weight-factor2 uses one existing production
DeepSeek V4+DSpark / Qwen3.8+MTP ordered pair, same binary
full/partial/partial/full, fresh process/data/calibration per arm. Each arm
has8192 context tokens,16 continuation tokens and two swap cycles; zero
context off, handoff on, configured4096 prefill chunks. Setup-ready and first
output stay separate, first use and prepared rows stay separate. The existing
state digest is excluded from endpoints; complete continuation logits are
hashed after the endpoint. Remaining release-after wait is reported as such,
since the excluded hashing can overlap background release.

Cross-arm diagnostics authenticate source text, actual int32 context IDs,
full initialized pre/restored state bytes and every finite full-vocabulary
continuation logit row. Each arm independently checks actual draft execution,
target device-mask selection, replay and exact unswapped continuations.
Results must retain individual arms/ranges and n=2 limitations. This is not a
fresh reference-engine qualification or full32-row swap table.


The completed factor runs on merged main698a077 plus reviewed candidate,
SparkA, installed GPU-exclusive supervisor DONE0/306sec. All four arms select
108GiB via an internal wrapper-only diagnostic cap; ordinary dynamic budgets
are109.141486/109.229377/108.786017/109.217658GiB. The common cap is floorGiB
of the smallest prior unprofiled dynamic budget108.328986GiB, retaining all
physical/startup-footprint guards. Unknown physical availability, cap above
ordinary dynamic budget or below required footprint refuses; production has
no public cap option and ordinary unset behavior is unchanged.

| Arm | Prepared ready, s (two legs) | Prepared first output, s | Completed reads, GiB | Known recovered warnings |
| --- | ---: | ---: | ---: | ---: |
| F1 | 14.032095 | 14.369218 | 172.409103 | 14 |
| P1 | 11.425823 | 11.748842 | 138.706371 | 11 |
| P2 | 11.406445 | 11.708675 | 138.706371 | 17 |
| F2 | 14.081499 | 14.413911 | 172.409103 | 0 |

Prepared means: ready14.056797→11.416134s(-18.786%), first output
14.391565→11.728759s(-18.503%), completed reads172.409103→138.706371GiB
(-19.548%). Full bookends and partial repetitions stay narrowly grouped.
Prepared A→B first output is6.123→5.939s(-2.99%); B→A8.269→5.789s(-29.99%).
Returns retain31.445GiB of DeepSeek weights and2.256GiB of Qwen weights.
This is n=2 per policy and one ordered pair, not statistical confidence,
a full swap table, joined batching or per-family performance qualification.

First-use regressions block adoption:

| Leg | Full first output F1/F2, s | Partial P1/P2, s | Mean change |
| --- | ---: | ---: | ---: |
| A→B |6.306760 /6.263321 |11.094470 /9.517335 |+63.975% |
| B→A |8.262766 /8.278566 |10.040703 /8.947772 |+14.794% |

The first Qwen arrival has1327 non-handoff loads in each partial arm, versus
zero in full, and longer page-in7.52–8.61s versus5.76s. Prior55–80us normal
Create measurements bound1327 ordinary calls at73–106ms; they do not alone
explain the whole gap, while the successful long-call attribution shows
6–59ms tails can occur. The first return has2.70–3.09s before the evicted
boundary, versus0.04–0.06s full, even though it has only174 non-handoff loads
versus14946 full. Existing untimestamped logs show one 2 MiB plan reclaim, followed by
the existing `malloc_trim(0)` call. Its association with the cold return was
unsupported: the later phase diagnostic puts the same reclaim at prepared
return. Selection/gather, early host work, eviction-lane waits and physical
backing pressure remain leads, not measured causes. Aggregate Create counters do not assign calls
by phase; no further tracing or capacity/default change is selected.

All16 rows are exact. Source text, actual int32 context IDs, full initialized
state and every full16-row finite continuation logit digest agree across
all arms. DSpark actually executes, target device masks select and graphs
replay. Both Table and final post-joined-teardown Create counters record zero
failures. Known recovered memdesc warning counts14/11/17/0 are included in
timings, not a clean-kernel claim; no unexpected kernel error occurs. Minimum
MemAvailable5.015/4.027/4.208/5.825GiB. Managed requested/released occupancy
stays within commonB. The initial invalid control and diagnostic profiles
below remain distinct.

## Rejected first control and attribution

Factor1 stopped after its firstFULL arm: runtime completed all4 rows exact,
but the mandatory kernel cursor gate found7 newNVRM NV_ERR_NO_MEMORY messages
at _memdescAllocInternal/mem_desc.c1359. No partial arm ran and none of this
control's timings qualify for comparison/adoption. ActualB109.164923GiB,
fixed4.131720GiB, MemAvailable start116.226482GiB/low5.122353GiB. Messages
clustered within12milliseconds38.3seconds after process start. Lazy parked
backing stays charged; the reserve's64MiB post-guard charge alone cannot
explain a multi-GiB discrepancy. No catalog-over-budget defect is established.

Source shows idle reserve refills swallow a known Create refusal and retry
only after the next command, whereas ordinary page-in Create refusals fail
work. An attribution-only counter now records reserve/ordinary actual
provider attempts/failures and the last returned error/detail/steady timestamp;
it does not classify a driver-internal failed attempt whose API succeeds.
The counter build and5focusedVmmLane controls pass. The first attribution
control completed4exact rows, reproducing3 allocation messages with zero
returned failures (reserve80467attempts, ordinary1084). Actual catalog
occupancy stayed within budget. Nearest MemAvailable was6.54GiB with no
large Normal-zone buddy blocks: a fragmentation lead, not proof of its cause.

A CUDA-only whole-run Nsight capture then reproduced13 allocation messages
and one distinct early refcnt0x56 warning. The native process independently
returned0, retired, and completed every exact state/continuation check. Nsight
and observer overhead makes every timing ineligible. All83314 traced
cuMemCreate calls returnedSuccess; backing-service counters likewise recorded
79336reserve and2215ordinary attempts without returned failures.

The installed Nsight serializes these driver names in its RUNTIME table.
Actual child PID/globalTid ownership and normalized epoch/session offset were
verified before analysis. The observed RAW/REAL drift spread was459us; kernel
source-to-receipt disagreement adds a few milliseconds, so exact call IDs are
not claimed. In the union of receipt REAL and mapped source times expanded
10ms in both directions, every allocation warning has only cuMemCreate as an
allocator/create/module/library API candidate. Those adjacent calls all
succeed; narrow mapped overlaps last6.05–59.07ms. This identifies successful
VMM creation as the allocator-family lead for the reproduced cluster, rather
than cuBLAS first use or module loading. Driver-internal fallback details and
the earlier unprofiled clusters remain unproven. The refcnt warning occurs
near startup before the allocation cluster and is preserved separately.

The capture target retired successfully; the first automatic analysis refused
the unexpected table naming, and bounded offline analyses then passed in8s
and2s. A preceding4s preparation attempt stopped before weight positioning
because cuBLAS tracing injected an environment variable the runtime guards
against; CUDA-only tracing required no guard bypass. A copied stale data path
was also corrected before the completed capture. Failed job records remain
part of the evidence. No capacity/default change or assumption of outside-host
pressure was made. Those diagnostic/profile runs are not pooled with the later valid factor.

## Reproduction and identities

The valid factor2 uses main698a077 plus this candidate, source-manifest
SHA256`9a4438a9e9ed155f173c30503af9569e8484cb49a1171bf351b653308e91ab3e`,
method`3844beef9a8fca84cc9648bede832f7cc79d47300f8c900939cadbed621d253e`,
packet`243be7e108a7f1f2008425b2ed4b665f44177c6ffc6e0174d71dfe6c0c0c9e2d`.
Spark-native pinned SDKaarch64-c09daba6ac31edee, official core receipt,
CUDA13.4.2/cuBLAS13.8.0.4, GB10 driver580.178.04. Wrapper SHA256
`22b0c9b4576c50ce3589ac14216d2199ce12d6b7f197c13520c488ff1df6ae23`;
runtime`72de809f81a717b7968d90d244389228d1e951fc7b4c3f609d21daf82fa45314`.
Both first-resolved `lib/jitllm` cuBLAS payloads equal the authenticated
`cublas` copies (`libcublas.so.13` SHA256ee7c1657a03695c0de790aa79e34cef9c9649756b1846b11dd44caca20ba656b,
`libcublasLt.so.13` ba3b942f4ea43433b65e8c492a7b73de887534dc20146506ddaa4a78c79c5d30).

Retrieve context text with `git show 4655685:docs/decisions.md > CONTEXT`;
368882bytes, SHA2566b159ff20d198a3ff825d78b5edc0c2bd8c9cf5af7222875fb88301635e14db5.
The resulting8192 DeepSeekint32 IDs hash
9f4c81443188649554a331e0172672704d276f5909e942bbdc5d97c4ed60fa6c.
Complete initialized state387072000bytes hash
3933641e8fc0b818dec6e0934fc3f8c3bf12f3a2904622f5620d1af799d88f7d;
complete16×129280float continuation logits8273920bytes (hashed with a
uint64 row length before each row) hash
0b1b657441a1f686357cccc4290362f0a71b241302015abd66ab1af8643b26a5.

Use the existing prepared artifact store. ConfigureDeepSeek target
8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234
andDSparkdd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5,
context262144/prefill4096/maxslots4/wave_formspeculative/speculationtrue.
Qwen3.8 targetc4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93
andMTP8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40,
context33792/prefill4096/maxslots4/speculationtrue/draft_vocab65536, tokenizer
andtemplate from Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6 as recorded
in model-support.md. For each fresharm use a new owner-only data/state/spill,
calibration and enrollment namespace. No reused settings/calibration.

Run the built wrapper in orderfull/partial/partial/full through the installed
Spark GPU-exclusive supervisor,600sectotal/stop-on-fail,145secpernativeprocess:

```
jitllm_swap_pager ARM --budget-bytes 115964116992 \
  --config FRESH_CONFIG --anchor FRESH_ENROLLMENT swap-table \
  --pairs deepseek:qwen3.8 --context-text CONTEXT --context-tokens 8192 \
  --continue 16 --cycles 2 --zero-context off --handoff on \
  --short-prompt 'What is the capital of France? Answer in one sentence.' \
  --report NEW_REPORT
```

Preserve kernel journal cursor/boot before eacharm and new messages afterward,
including failedarms. The only accepted recovered warning is the exact
`NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY] (0x00000051) returned from _memdescAllocInternal(pMemDesc) @ mem_desc.c:1359`;
record its count and retain raw clock fields. Refuse other NVIDIA/NVRM/Xid/UVM,
refcnt/driver or host allocation errors, actualCreate failures, budgetabove
108GiB or currentdynamicbelowcap, missingexactrows/digests, and unprovennative
retirement. Require exactlyone final post-joined-teardown counter record with
retiredtrue/failureszero and counters no smaller than Table. Raw outputs and
scripts stay external and may be deleted at milestoneclose; the context is
reconstructible from Git, and artifacts/pins/IDs and command above suffice
to repeat without the raw bundle.

Latest follow-up: cap-only host attribution source b425822bcd61bcb1e3b7afac14ce4d745e63243ce3e18d363173b49e5934b2e2 passes the existing actual cap/default/refusal and post-join accessor controls (integration6,DONE0/25sec). It preserves all policies and adds no ordinary clock reads or new scheduler calls. The one FULL/one PARTIAL diagnostic retired positively: all eight rows exact, no returned Create failures or kernel warnings. It did not reproduce the repeated seconds-scale cold regression. The same logical occupancy/retention/read/graph/draft lifecycle and the same six reclaim events were preserved; CPU frequency endpoints match. Diagnostic partial began with less MemAvailable and reached a lower minimum, so scalar availability does not explain the disappearance. Cold Qwen planning fell124ms→29ms and DS return planning64–75ms→36ms, while other cold host intervals also fell; cause remains unresolved. The original untimestamped2MiBPLAN log cannot be assigned to cold return: the diagnostic places its same PLAN at prepared return, so the earlier malloc_trim association was unsupported. Ordinary partial remains off. The complete read-only comparison and existing scheduler.started endpoint gap are recorded in external partial-weight-lifecycle-comparison1. The bounded subtraction-only endpoint follow-up and the lane-interval qualification below preserve the existing policies; no broad trace follows from the ambiguous result.


## Cold-return retirement attribution

The second bounded diagnostic ran one FULL and up to three fresh PARTIAL
processes, stopping after the third PARTIAL reproduced the cold-return tail.
All 16 rows retained exact initialized states and complete continuation heads;
actual provider Create failures and unexpected kernel errors remained zero.
This is attribution evidence, not a bookended performance comparison.

| Arm | Preparation | Retirement | Total eviction | Accepted recovered warnings |
| --- | ---: | ---: | ---: | ---: |
| FULL | 6.971 ms | 30.756 ms | 37.727 ms | 0 |
| PARTIAL 1 | 41.801 ms | 42.395 ms | 84.196 ms | 0 |
| PARTIAL 2 | 40.309 ms | 41.602 ms | 81.911 ms | 0 |
| PARTIAL 3 | 90.533 ms | 1651.353 ms | 1741.886 ms | 19 |

The reproduced tail lies after the scheduler's validated Swap started and
before eviction completed. Room preparation was 53.270 ms and release
partitioning 17.338 ms; no heap release or victim take occurred through ready.
This excludes those measured host phases as the dominant cause of this tail.
No new reserve Create began during this activation, but the attempt-only
counters could not exclude a Create already running on the sole backing lane.
Lazy parking itself performs metadata lookup, stash allocation and board
publication; it does not invoke CUDA unmap. Neither the metadata lookup nor
the provider's atomic exclusivity assertion is a waiting mutex.

Integration8 adds cap-only coherent actual lane-call intervals: one current
Create and per-kind first/last/longest completed serial intervals, plus
aggregate lazy-handler elapsed intervals. These durations include
descheduling and bookkeeping. They do not measure pure CUDA driver time.
Its three fake and two actual Gemma2 serving controls passed on Spark A
(DONE rc0, 39 seconds), including busy/completion/refusal, reserve completion,
cap admission/default refusal and a retained cached switch. No ordinary
per-wave scheduler call or clock read was added. Source SHA256
`df295a22548ae7dcfcc9b05b3a141e0f075d56b524b7d3afcf067600ee18937c`.
The next fixed bounded acquisition is diagnostic only; ordinary partial
retention remains off pending its result and a resolved cold-path policy.

## Handler attribution and the wake improvement

The next bounded acquisition reproduced a 1.346 s cold-return retirement.
Successful lazy handlers accounted for 1.297 s (96.4% before boundary
clipping), spread across 35,359 handlers at a mean36.693µs. Fast controls
averaged0.933–1.009µs. No Create was active at the requested boundary and no
new Create started during retirement; authenticated completion intervals did
not overlap it. This rejects the in-flight reserve-Create explanation for
that observed tail. The signed residual was positive48.780ms. There were
35,514 logical victims but only35,359 lazy handoffs:155 explicit releases
are distinct operations and do not enter the lazy-handler count.

Splitting the handler into metadata, stash insertion and completion
publication did not reproduce the tail in one FULL and three PARTIAL
processes. All16 rows were exact with zero returned failures or recovered
warnings. Cold retirement ranged35.289–65.867ms. These new milestone clocks
and timing locks may perturb synchronization, so the absence is not evidence
that the historical tail is resolved. Metadata lookups are ordinary table
lookups; stash insertion allocates list and map nodes. The successful lazy
path announces acceptance and terminal completion separately. The scheduler
can harvest between those announcements, requiring another queue insertion
and wake.

A small shared improvement moves WakeFlag's notification outside its
handshake mutex. The mutex still closes the predicate-to-wait race: a waiter
either sees the pending predicate or reaches its wait before the signaler
acquires and releases the handshake mutex. Producers must finish Signal
before the WakeFlag is destroyed; completion observation alone is not a
lifetime proof. PagedNode joins its scheduler and lanes before those members
are destroyed. Existing coalescing, many-publisher, sleeping-lane and shutdown
controls all pass (eight exact checks). No notification policy or public
interface changes.

The equally instrumented old/new/new/old comparison ran PARTIAL in all four
arms at the same108GiB; only WakeFlag differed. It retained exact IDs,
initialized states and16 complete continuation heads in all16 rows. Cold
return publication elapsed fell20.099ms[19.166,21.032] to8.154ms[8.046,8.263],
59.4%; retirement fell67.662ms[65.679,69.645] to43.362ms[42.358,44.365],35.9%.
Cold-return ready fell0.341% and first output0.255%, with nonoverlapping n2
ranges. Prepared endpoints overlap, and all-paid improvement0.106% is smaller
than the full old/new bookend movement1.151%/1.429%. This qualifies a narrow
publication benefit, not a broad swap speedup or historical-tail repair.
Recovered warning counts were1/6/0/0; actual Create failures remained zero.
The shared base mechanism applies to every current family; actual model
performance was measured only on DeepSeek/Qwen. No unmeasured family speed
claim follows from inheritance.

Server no longer enables backing handler timers for the internal budget cap.
An ordinary full/partial/partial/full comparison uses the same qualified wake
implementation in both policies and requires every positioning/swap timing
snapshot to be null. Count-only Create diagnostics, exact outputs, fixed
admissible108GiB budget, kernel classifications and post-join retirement proof
remain. The actual cap/default and zero-victim retained-switch controls pass
with timing disabled (two exact serving tests,26s). This final confirmation
is the pending adoption decision; partial remains off until it is reviewed.

### Replaying the narrow wake comparison

The wake comparison is separate from the historical FULL/PARTIAL factor and
the ordinary confirmation. All four O1/N1/N2/O2 arms use the `partial` wrapper
mode,108GiB and the same command/artifacts/context reconstructed above.
Each process gets a fresh configuration data directory and enrollment path.
Both variants deliberately enable handler timing by setting
`.diagnostic_backing_timing = options.diagnostic_budget_cap_bytes.has_value()`
in `ServerNodeSettings`; ordinary serving and the final ordinary factor
leave that initializer absent. This is an internal diagnostic replay edit,
not a public configuration surface. Both variants must have identical stage
instrumentation, binaries built with the pinned Spark SDK and identical
first-resolved/fallback cuBLAS payloads.

The old variant calls `wake_.notify_one()` while its `std::scoped_lock
lock(mutex_)` is still in scope. The candidate closes that lock scope first,
then calls the same notification. Keep the pending exchange, predicate,
waiter handshake and every other source byte identical between variants.
The measured snapshots use basec382965 plus the partial pager candidate.
Old source-manifest SHA256 is
`271c69a2f0314ebefd6171eae771f3c8a4477e2165e4d1556cc742bac51dc39f`;
new53-file manifest SHA256 is
`bc4268fb704e8bb85c5ffd95340927246e1b01ea7d7c5984d2a0018387e75e1e`.
The latter adds unchanged completion/test bindings as well as WakeFlag;
its common47 bindings are unchanged. The original wake header is retrieved
with `git show c382965:src/base/wake.h`, SHA256
`5f7cf85f2695cfadd8692e2597eadae14901cd297ea0b2f4fe6d70305c9f076d`.
The candidate header SHA256 is
`34f1cc025330d7142268843e3b89587839f004b95dffb55c9fbd6e60ea216b48`.
The source-manifest digests are not Git commits.

| Measured payload | Old | Candidate |
| --- | --- | --- |
| runtime ELF SHA256 | `3d27136abf562efe2a743dd9cb87ab91bf0cb8829fffd4a181635e7bf41a5b74` | `413cea59a1d12982f3a48a31358e47a0f769cb0f809443a1110b3f6497341975` |
| swap wrapper ELF SHA256 | `6fe1beffa4280a3532cce5719a1acad122766f682aab9fa569ead374683997ba` | `009e7353e2c73913cb44117dee112dfd18d7f303212a423103207e9b4b619499` |

Both official Spark receipts have SHA256
`0296e41b77f3db5f50dfc06b68ebd87906a683d45350ea2d5586f929c8333ecd`.
The receipt binds dependencies/toolchain; it is not the modified source proof.
The source manifests and actual ELF identities provide that separate proof.
The same copied private cuBLAS files resolved first in all four processes:
libcublas SHA256`ee7c1657a03695c0de790aa79e34cef9c9649756b1846b11dd44caca20ba656b`,
libcublasLt SHA256`ba3b942f4ea43433b65e8c492a7b73de887534dc20146506ddaa4a78c79c5d30`.
Replay requires the source edit and command above, not disposable raw outputs
or the retained private binary directory. Fresh ELF hashes can differ after
subsequent independent source integration and are recorded for the new run.

## Ordinary confirmation: prepared gain survives; first arrival remains slower

The ordinary FULL/PARTIAL/PARTIAL/FULL confirmation completed positively in
about4minutes on SparkA. Both policies use the qualified wake implementation;
all phase handler-timing payloads are null. All16 rows preserve exact actual
IDs, initialized states and complete continuation heads. Returned Create
failures and accepted recovered warnings are zero in every arm. Every arm
selects the same admissible108GiB; dynamic budgets remain above it.

| Paid endpoint | FULL F1 | PARTIAL P1 | PARTIAL P2 | FULL F2 | Partial mean change |
| --- | ---: | ---: | ---: | ---: | ---: |
| First A→B ready | 5.864410s | 6.099423s | 5.909583s | 5.862250s | +2.408% |
| First A→B first output | 6.227492s | 6.642062s | 6.281990s | 6.236074s | +3.695% |
| First B→A ready | 8.205476s | 5.889917s | 5.698839s | 8.145461s | −29.125% |
| First B→A first output | 8.348435s | 6.094783s | 5.831038s | 8.255351s | −28.174% |
| Prepared two-leg ready | 14.021966s | 11.589693s | 11.373592s | 14.016917s | −18.102% |
| Prepared two-leg first output | 14.362099s | 11.971196s | 11.682601s | 14.355682s | −17.634% |
| Prepared completed storage reads | 185122865152B | 148934832128B | 148934832128B | 185122865152B | −19.548% |
| All four first-output endpoints | 28.938026s | 24.708041s | 23.795629s | 28.847107s | −16.062% |

Prepared first-output ranges do not overlap. Full bookends move−0.045% and
partial prepared arms−2.411%; n2 is still limited. The cold-return retirement
is64.063–80.765ms in partial, without the historical seconds-scale tail.
Two ordinary samples without that intermittent event do not prove it cannot
recur. The first A→B still regresses in both partial arms: full bookend
movement+0.138%, partial movement−5.421%, with both partial values above the
full range. The cycle's substantial net gain does not erase that first-arrival
regression. Ordinary partial remains disabled pending the adoption decision.

First A→B has identical completed reads in every arm (76840689664B). Partial
retains34126954496B of outgoing weights but hands off35342 of36669 incoming
extents; full hands off all36669. Partial's room phase is24.0–24.3ms plus
13.8–14.4ms partitioning, versus full room6.9ms and no partitioning. Room
contains10.8ms gather and5.2–5.4ms selection; these nested spans must not be
summed twice. There is no heap trim. Partial page-in is5.964632/5.761244s,
versus full5.756547/5.754440s: most of the extra page-in appears only in P1.
P1 also spends an extra roughly170ms between ready and first output.
Count-only snapshots show fresh Create attempts in partial and none in full,
but do not measure API duration or prove they explain the complete mean gap.
There is no new policy heuristic, reserve change, trace or automatic repeat.

Measured ordinary source-manifest SHA256:
`fb75504ed0757032b9b5366b87b4be26f3c880e46ec7226d410df7b1ba200bda`.
Runtime ELF:
`ab779e0905b6244cb99076fe0e66267d792e9f4b51118adec9b96754b8030d0b`;
wrapper ELF:
`99afe07f1b6db59fee5343cdaa153d9b9850307e492e87de6e3f682adf1e3a84`.
The official receipt and pinned cuBLAS identities match the wake comparison.
The frozen method-manifest SHA256 is
`3d70ee615832c9e74efbb58bafa2f2b9dfe63fc1f21df260b9e7ed609076501f`;
checked aggregate SHA256 is
`86d5ca5025c9cb3b8e03336db8cbf90194409bd522f1595dd4df37e84ac3f8f0`.
Reproduce with the command/context/artifacts above, current ordinary no-timer
source, fresh configurations and FULL/PARTIAL/PARTIAL/FULL order. Raw scripts
and output bundles are not required inputs and may be removed at milestone
close.

## Contiguous donor identity validation

ReleaseForHandoff previously allocated a std::set node for each missing and
selected typed extent identity. At the actual36669missing/35342selected
counts that creates roughly72K short-lived tree nodes. A narrow replacement
reserves contiguous identity scratch, sorts it once and refuses adjacent
aliases. Identity generations remain part of the comparison; class/domain/
size/overflow and mixed-donor arithmetic stay unchanged. Sorting the scratch
never sorts the actual selected release sequence. The new contract control
covers all three duplicate placements, reused indices at distinct generations
and shuffled deterministic release order. All seven SwapRoom tests pass.

A same-process CPU helper comparison uses the frozen old helper and current
candidate, same actual counts/O/B, fixed O/N/N/O32calls per shape. Compatible
sorted input falls6.943ms[6.907,6.980]→0.791ms[0.789,0.792],88.615%. A synthetic
mixed-class/shuffled control at the same counts falls7.485ms[7.482,7.489]→
1.614ms[1.609,1.618],78.441%. All complete outputs and refusal controls agree.
Per-call brackets cover the helper allocation/sort/local cleanup, while
result equality and returned-result cleanup stay outside. This is roughly6ms
of deterministic CPU helper savings, not an end-to-end model speed claim.
It can remove part of the14ms partition span, but cannot explain or erase the
full first-arrival mean gap by itself. The subsequent ordinary PARTIAL O/N/N/O model comparison kept the108GiB
cap, WakeFlag candidate and count-only/no-handler-timer path in all arms.
Only the contiguous donor-identity helper changed. All16 complete states,
continuation heads and outputs were exact; final Create failures were zero.
Known recovered allocation warning counts were0/1/3/2. Authenticated kernel
clock brackets place the warnings in first A→B page-in, rather than the
later B→A retirement; this temporal placement does not establish a cause.

Cold first-arrival partitioning fell13.361ms[13.312,13.409]→5.529ms[5.409,5.649],
58.615%; cold-return partitioning fell13.827→7.775ms,43.768%. First-arrival
evict endpoints fell7.823%. Prepared two-leg first-output means moved
11.717205s[11.701454,11.732956]→11.700784s[11.688718,11.712850],−0.140%,
with overlapping ranges and old bookend movement0.269%. These are useful
host preparation savings, without a demonstrated broad model gain.

The ordinary intermittent tail genuinely recurred in N2: cold-return
retirement1.409748490s, versus41.560–42.860ms in the other three arms.
N2 cold-return prepare was59.840ms, so the large delay remains in retirement.
The old/new/new/old first-return outputs were5.806032/5.816686/7.433837/
5.812442s; first-arrival outputs were6.285804/6.597847/8.300789/6.315646s.
All-paid new means regress8.176%, dominated by the anomalous arm. The
previous tree-helper factor also had intermittent long retirement, so this
screen establishes neither helper causation nor a tail repair. Partial's
ordinary default remains off. The next candidate coalesces only synchronous
successful lazy-Park acceptance and terminal announcements, preserving the
independent generation/proof state machines and logical progress counts.
No further unchanged diagnostic repetition is planned.

Source-manifest SHA256:
`d06dd565bd4ca97de28e02b78b5850a1a43722f9b54444d1270f9f7540a2fe0b`;
method-manifest SHA256:
`c30da37090475c5b208ee54d3424c86f6104a8e82b8fdd4cbf4859d7c298ec8f`;
checked aggregate SHA256:
`518bb7867c17f990df9e2cbad04d9c4569a36f8c19fed54262a668e2b7126b57`.
Replay uses the existing context/artifacts and ordinary108GiB partial route
in all four arms, replacing only ReleaseForHandoff's contiguous identity
scratch with the prior set-based validation for the two old bookends.
Actual private cuBLAS resolution and payloads were equal in all arms.


## Paired announcement for synchronous lazy Park

The next concrete source candidate changes only successful synchronous lazy
Park. Previously acceptance and the proven success terminal each announced
news independently; if the scheduler harvested between them, one already
completed Park could incur two news insertions and wakes. AcceptedSuccess
runs the existing independent acceptance and terminal state machines with a
stack-local sticky announcement flag. Each recorded observation still
increments logical progress. Contradiction and later-proof paths use the
same flag, and any earlier change is announced once after both calls settle,
including when the second call is duplicate or stale.

The pair is not atomic against other publishers, Harvest or Close. Every
existing generation, duplicate, contradiction and proof check remains;
late news after owner retirement is harmless as for the independent API.
Refused/unknown operations, map reuse, other backing successes and all
asynchronous operations retain their existing publication routes. Producer
threads must finish their Signal calls before board/wake destruction;
observing completion alone never proves publisher retirement.

The exact26 focused prerequisites pass:19 board/wake controls,3 actual lazy
Park/refusal compatibility controls,3 lane wake/shutdown controls and1 many-
publisher scheduler control. Four new tests exercise complete paired success,
logical counts/duplicate/stale/proof; acceptance/terminal contradictions and
partial-change flushes;8000 operations with concurrent owner harvest/Close/
reuse and three publishers; and200 last-generation retirement races.
Publisher threads join before board/wake destruction. The actual Park control
also checks two logical publications and accepted, proven success before
Close, with unchanged mapping/stash behavior.

An initial method preflight failed before retention or build because the
method checksum file used relative paths from the wrong working directory.
Only those checksum paths were corrected to installed absolute paths; the
bounded retry completed all26 checks. The failed supervisor record is retained.
Source-manifest SHA256:
`a2c1eef24aa235e1e4933a060b5d8cef37a775ad6c06472db42e55a5bc3d16ef`.
Checked prerequisite aggregate SHA256:
`374f55f634cc151ce54a68620b58f9e0e657000b574aa19e8994f9c0a0713a1c`.
The separately reviewed ordinary PARTIAL O/N/N/O comparison uses the same
108GiB cap, ordinary handler timing off and equal actual first-resolved
private cuBLAS payloads. It changes only this announcement candidate.
All16 initialized states, complete continuation heads and outputs are exact;
returned Create failures are zero, and recovered warning counts are0/0/0/12.
All12 O2 warnings fall around first A→B page-in and setup under the recorded
RAW/MONOTONIC brackets plus10ms uncertainty: seven overlap page-in; five
follow its loaded endpoint and precede paid-ready. None maps to later B→A
retirement. This is
qualified temporal placement, without an internal allocation-cause claim.

| Endpoint | O1 | N1 | N2 | O2 |
| --- | ---: | ---: | ---: | ---: |
| First A→B first output |6.334160s|6.386659s|6.288601s|7.484897s|
| First B→A first output |5.821974s|5.812210s|5.834331s|7.466203s|
| First A→B evict |103.436ms|129.389ms|123.298ms|105.813ms|
| First B→A retirement |70.102ms|46.074ms|75.565ms|1529.655ms|
| Prepared two-leg first output |11.730160s|11.694087s|11.723971s|11.676247s|

Prepared first-output means are11.703204s[11.676247,11.730160] old versus
11.709029s[11.694087,11.723971] new,+0.050%, with overlapping ranges.
Old prepared bookend movement is−0.460%; candidate movement+0.256%.
Cold first-arrival means fall8.277% and cold-return means12.354%, but both
are dominated by the slow O2 bookend. First-arrival evict time instead
increases20.759%, from104.625ms[103.436,105.813] to126.344ms[123.298,129.389],
with nonoverlapping ranges. O2 again reproduces the seconds-scale ordinary
cold-return retirement tail; candidate samples46.074/75.565ms do not prove
it cannot recur. All-paid endpoints fall5.491%, below old bookend movement
11.475%, so there is no demonstrated broad speedup. Ordinary partial remains
off. The paired-announcement candidate is not adopted; its API/caller/tests
were reverted to the authenticated prior source. No automatic repeat follows,
and no historical-tail repair is claimed.

Candidate runtime SHA256:
`35d05dc44f1c1c2c275cb3558a29aea4ee92c1f65554e395c281e3ba40b2029e`;
wrapper SHA256:
`6b763b33b500d2d9dc79a309fe7f7142190e3bdfb4db1315a604acc2a97697c0`.
Model method SHA256:
`4af02e882cabd968a263394415c6337ed62751e8dce1723919b91a059f4a4f32`;
checked aggregate SHA256:
`2ca83a46e8059b3a4c07859fbc652d0496dffa74de0850c1183b294b4aebc131`.
Replay uses the same context/artifacts, ordinary handler timing off and
PARTIAL in all four arms; old bookends call independent Accept/Complete
only at the successful lazy-Park callsite, candidate calls AcceptedSuccess.
Existing generation/proof/contradiction logic remains the same in both.


## Bounded owner removal of completion news

The rejected paired-publication paths were restored exactly to the qualified
53-file d06dd565 manifest. The next source candidate removes only currently
queued completion indices into a fixed64-element local array under one
news-mutex hold, then releases the mutex before queued-clear, mailbox reads,
deduplication or scheduler observation. It never waits to fill a batch.
The remaining result limit bounds each removal; stale/duplicate holes permit
another fill, and remaining news retains the existing owner re-signal.
Generation/proof/Close rules and provider publication remain unchanged.

Owner-only clockless counters record removed indices and successful nonempty
pop locks, excluding empty/final-more checks. The node exposes them only after
scheduler/lane joins; the existing cap-only final record includes the optional
snapshot. There are no per-handler clocks, provider hooks or per-wave calls.
All26 focused controls pass:18 board/wake,3 actual Park/refusal,3 lane/shutdown,
1 scheduler many-publisher and1 actual mixed-donor GPU transaction. Three new
board controls exercise bounded multi-index/partial-limit draining across
closed holes and generation reuse, retired news, and concurrent acceptance/
terminal/later-proof publication with owner harvest/Close/reuse/dedup.
The actualGPU control refuses the snapshot before join and checks it afterward.

A separately reviewed candidate-only four-row process completes with all
initialized states, complete continuation heads and outputs exact; zero
Create failures and zero recovered warnings. The selected108GiB is admissible
against117384813356B dynamic capacity. Its final whole-node counters are
1,243,072 removed indices and1,089,413 nonempty pop locks,1.1410475 indices
per lock. Actual batching therefore occurs, with modest whole-node amortization.
These totals include all phases and teardown; they establish neither eviction-
only nor Park-specific amortization or historical-tail causality.

This n1 run is explicitly diagnostic and performance_eligible=false.
Cold first-arrival/return first outputs are6.336500/5.800813s, with retirement
70.169/42.560ms; prepared two-leg first output is11.676345s. Absence of the
intermittent tail in one sample is not a repair proof. No comparison, automatic
expanded factor or ordinary default change follows without a separate review.
Source-manifest SHA256:
`2eabcc1416232c6ccb9108948496a24959dd3c69de48e245a95834bb742d49e3`;
probe-method SHA256:
`1799fa1bda9e561aeab0020fca55674d75c331335f3f677ea280cbc59a08a6e0`;
checked aggregate SHA256:
`ff944f2eae7dd280ab8f0ae70002f6f2e941d400de058f9d01da2a3542573861`.

The separately reviewed ordinary comparison then runs four fresh PARTIAL
processes in old/new/new/old order. Both versions keep the same WakeFlag,
contiguous donor scratch, 108 GiB capacity, private pinned cuBLAS payloads and
ordinary handler timing disabled. The only candidate changes are bounded
consumer news removal, owner counters, their retired snapshot and cap-only
final record. Actual library resolution is authenticated for both wrappers
and runtimes before paid work. All 16 rows retain identical initialized
state, complete finite continuation heads, actual input IDs and outputs.
All four arms report zero Create failures and zero kernel warnings.

| Ordinary PARTIAL arm | Cold A→B first (s) | Cold B→A first (s) | Cold B→A retirement (ms) | Prepared two-leg first (s) |
| --- | ---: | ---: | ---: | ---: |
| O1 | 6.301011 | 5.795759 | 42.194 | 11.691545 |
| N1 | 6.345467 | 5.805997 | 45.295 | 11.709082 |
| N2 | 6.266610 | 5.822784 | 66.128 | 11.671587 |
| O2 | 6.262012 | 5.819350 | 69.503 | 11.743839 |

Cold first-arrival eviction falls from 134.544 ms [133.122, 135.965] to
111.322 ms [94.583, 128.060], −17.260%, with nonoverlapping ranges.
Its first-output mean increases 0.390%, with overlapping ranges.
Cold-return retirement is unchanged within the observed spread:
55.848 ms [42.194, 69.503] versus 55.711 ms [45.295, 66.128].
Prepared two-leg first output changes from 11.717692 s [11.691545, 11.743839]
to 11.690335 s [11.671587, 11.709082], −0.233%; the old bookends move
+0.447%, and the ranges overlap. All-paid first output changes +0.0168%.
These n2 measurements support a narrow first-arrival eviction improvement,
with no demonstrated broad swap speedup. No seconds-scale retirement tail
appears in either version; its absence establishes no repair.

New arms retire with whole-node index/pop-lock totals of
1,215,408/1,075,511 and 1,226,649/1,065,689: 1.130075 and 1.151038 indices
per nonempty pop lock, or 11.510% and 13.122% fewer such locks than one
per index. These totals include all phases and teardown; they do not establish
Park-specific or eviction-only amortization. Ordinary partial remains off
pending the coordinator's reviewed disposition.

The old runtime SHA256 is
`3007876fac556589d2169b09679d6b9f35bdbd31268a29eefa15b69cfd3f515e`,
and its wrapper SHA256 is
`41f77aa77945ae61c6456203a62eca27546dcc46366462d920db09d13a286a53`.
The new runtime SHA256 is
`cfb53c18a5b573bce3878d2f2e91be6a959649704d40440d36f35048b2303840`,
and its wrapper SHA256 is
`b391864feeb54f0a885cfe2b59bd0637df9f21e764fcf02c4062344b1fd4dab9`.
The source-manifest SHA256 remains
`2eabcc1416232c6ccb9108948496a24959dd3c69de48e245a95834bb742d49e3`;
the model-method SHA256 is
`faa3f9699ac6737c62cfa3754b94899a26b51b3c56ea8939c4a9c14650805d3f`;
the checked aggregate SHA256 is
`2a1260671796dedaff9739e897fd2a53aa1af9de8e4ad1bc5f53df93ee628eb4`.
Replay uses the same prepared artifacts and context retrieval above, PARTIAL
in all four arms, 8192 context tokens, 16 continuation rows, two cycles,
zero-context off and handoff on. Each arm uses a fresh data/calibration
namespace. The old helper takes one currently queued news index under each
news-mutex hold; the candidate takes at most 64, bounded by the remaining
result limit, and unlocks before mailbox reads. Neither version enables
per-handler diagnostic timing in the ordinary runtime.

## Final current-source selection

A final fresh same-binary FULL/PARTIAL/PARTIAL/FULL comparison includes the
retained WakeFlag, contiguous donor identity scratch and bounded consumer
news removal in BOTH policies. Only the internal full/partial override changes.
All four processes use the same 108 GiB capacity, authentic current runtime/
wrapper and private pinned cuBLAS payloads, ordinary handler clocks disabled,
8K initialized state, 16 complete finite continuation heads and two cycles.
Fresh owned data/calibration namespaces prevent calibration inheritance.
Actual ldd path/inode/bytes are checked for both policy invocation identities.
All 16 rows retain exact initialized state, full heads, actual IDs and outputs;
Create failures are zero through final joined teardown. Recovered known
memdesc warning counts are 0/29/1/0, accepted and included in paid time.

| Arm | Cold A→B ready / first (s) | Cold B→A ready / first (s) | Prepared two-leg ready / first (s) | Recovered warnings |
| --- | ---: | ---: | ---: | ---: |
| F1 | 5.872710 / 6.222469 | 8.150193 / 8.258982 | 14.070960 / 14.409738 | 0 |
| P1 | 6.823827 / 7.434467 | 8.652286 / 8.815839 | 11.524465 / 11.890037 | 29 |
| P2 | 6.064144 / 6.531613 | 5.774880 / 5.941655 | 11.363492 / 11.672593 | 1 |
| F2 | 5.894415 / 6.263430 | 8.181304 / 8.290709 | 14.082194 / 14.417312 | 0 |

Prepared first-output mean improves 18.262%, from 14.413525 s
[14.409738, 14.417312] to 11.781315 s [11.672593, 11.890037]. Ready improves
18.702%; completed storage read bytes fall from 185,122,865,152 to
148,934,832,128 per prepared cycle, −19.548%. The full first-output bookends
move +0.053%, while partial arms move −1.829%. These separated ranges
support the bounded prepared benefit, without statistical certainty at n2.

Cold first-arrival first output regresses 11.855%, from 6.242950 s
[6.222469, 6.263430] to 6.983040 s [6.531613, 7.434467], with nonoverlapping
ranges; ready regresses 9.525%. Full bookends move +0.658%, while partial
arms move −12.144%. Cold-return mean improves 10.829%, but its wide partial
range [5.941655, 8.815839] includes P1 slower than both full controls
[8.258982, 8.290709]. All-paid first output improves 9.637%, with full
bookends moving +0.278% and partial arms −14.195%; that aggregate does not
erase the cold first-arrival regression.

Cold-return retirement remains 30.651–59.622 ms across the four arms; the
historical seconds-scale retirement tail does not reproduce, without proof
it is repaired. P1 cold-return preparation is 139.766 ms, including 60.432 ms
release partition work, while P2 preparation is 35.671 ms. Other paid phases
also rise in P1; no single-cause attribution follows this factor, and known
recovered warning counts are not proof of the exact slow phase. Both full
arms reach about 6.36–6.41 GB minimum MemAvailable; partial arms about
3.33–3.42 GB. Selected capacity is exactly 115,964,116,992 bytes in every arm,
below each independently recorded dynamic budget. The physical differences
are recorded conditions, not grounds for changing guards or discarding P1.

Ordinary partial stays false. The shared internal mechanism, exact recovery,
missing-read accounting and qualified Wake/identity/Harvest work can land;
ordinary partial adoption remains explicitly open for every family. There is
no public partial-mode configuration, CLI or HTTP capability. Actual model
performance evidence is DeepSeek/Qwen; actual transaction/recovery fixtures
exercise Gemma2 and retain unrelated Gemma26 cache. Other families inherit
shared code without unmeasured performance claims. The fixture is also
parameterized for Gemma3, but its partial recovery /1 arm was not run.

Final method SHA256:
`2466ac915ce4bd0a12c75772a88aa92c84821a6f7f095d76d5464ad7a428e35a`;
source manifest SHA256:
`2eabcc1416232c6ccb9108948496a24959dd3c69de48e245a95834bb742d49e3`;
checked aggregate SHA256:
`f8cecd1e6f7abc12c08cb370b3a036784679327a2d69f2f62487a7f408a961d5`.
Runtime and wrapper are the new cfb53c18/b391864f identities recorded above,
identical in all four arms. The private library digests and build receipt are
unchanged. Final source integration onto a8bf2f0 preserves its direct256
fixture and optional-node StateHash; all A production source bytes remain
identical to the measured source, with only committed Gemma dependencies and
the combined test fixture changing.

To reproduce after raw cleanup, retrieve the context with
`git show 4655685:docs/decisions.md > CONTEXT.md` and verify its recorded
368882-byte/SHA256 identity above. Use the approved prepared DeepSeek/Qwen
artifacts, pinned SDK libraries and a freshly built `jitllm_swap_pager`.
For F1, P1, P2, F2 in that exact order, create four otherwise identical runtime
TOML files whose only difference is a fresh absolute `data_dir` namespace,
and invoke the benchmark as follows with MODE full, partial, partial, full:

```sh
jitllm_swap_pager MODE --budget-bytes 115964116992 \
  --config ARM.toml --anchor ARM/enrollment swap-table \
  --pairs deepseek:qwen3.8 --context-text CONTEXT.md \
  --context-tokens 8192 --continue 16 --cycles 2 --zero-context off \
  --handoff on --short-prompt 'What is the capital of France? Answer in one sentence.' \
  --report ARM/table.json
```

Run through the installed Spark GPU-exclusive supervisor, stop on first
failure, at most 145 seconds per native process and 600 seconds for the fixed
four arms. Authenticate source/ELF/receipt/library identities and actual ldd
resolution before/after; require native success, no remaining GPU process,
all state/ID/full-head digests exact within/across arms, actual target/drafter
execution and replay, completed reads equal submitted bytes, managed occupancy
within selected capacity and selected capacity within actual startup guards.
Require exactly one joined final Create/harvest record with no actual provider
failure. Preserve per-arm kernel cursor/boot and count only the exact recovered
message documented above; refuse every other NVIDIA/Xid/UVM/refcnt or host
allocation error. Do not enable per-handler timing, move paid work outside the
endpoints or subtract whole-continuation planning from first-output latency.
Ready ends at incoming readiness; first output ends before diagnostic hashing;
remaining release wait is reported separately and may overlap hashing.
Compute cold legs separately, sum the two prepared rows for each arm, then
compare two full versus two partial means/ranges/bookend movement. No raw
result bundle or disposable acquisition script is needed to reconstruct the
inputs, paid work, endpoint definitions or aggregate formulas.

Final integration on Spark A compiles the combined a8bf2f0 fixture and passes
exactly 17 focused checks: seven Wake/bounded-Harvest controls, seven SwapRoom
controls and three actual Gemma2 ordinary-default/cap/partial-cache/recovery
controls. No failures, errors, disabled or skipped cases occur. The combined
direct256 fixture is preserved byte-for-byte and compile-checked; it is not
claimed as run by this check set. The final integration runtime/wrapper hashes
are `7b71a574e85d0b7ae6e77832607fab1a16d3e8bc82812f93c6bf9f705884ca30` /
`585d9aba10608d5451b5f244f0cacac227614c18e7270c7d0a5e0d3e96089364`;
checked aggregate SHA256 is
`514931394afd1de33684bec7c80fdfad1e76da954bea4deb173b073f74d07928`.
Local changed-source formatting, REUSE, embedded headers, portability
boundaries and diff whitespace checks pass. Full regression/shipment tiers
and the all-pair swap table remain deferred under the owner's override;
this slice does not claim those gates passed.
