<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Matched-input production DeepSeek prefill attribution

One representative 8K screen on Spark A records the production fast plan's
complete original steps, with one ordinary pass before, one observed pass,
and one ordinary pass after. All six complete frontier heads are byte equal
within their chunk. The ordinary bookends differ by 0.486%. This selects a
small next factor; it does not qualify a default, model quality, concurrency,
or a whole-model improvement.

The fixed 8192 I32 input is the literal study's input
`60329b1e4ff5d19d40082666e8c08b86655d4b1173486aecbc4a752bb6aa3ac7`.
The prepared community artifact is
`cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac`.
Context is 8192, two chunks of 4096, all 43 layers, and one complete frontier
head per chunk. Native fast arithmetic, F16 KV, F32 hyper-connections,
compact experts and wide sparse attention are retained. Optional Q2-D2R and
literal HCA aliases are off. The literal pipeline has different cache/math
contracts and final-head cadence; these results do not directly subtract
from its whole-pipeline time. The older single-4096 native trace uses a
different input and remains descriptive history.

| Pass | Paid prefill wall, seconds |
| --- | ---: |
| Ordinary before | 13.380417334 |
| Observed | 13.308826627 |
| Ordinary after | 13.315705974 |

Wall includes both complete chunks and their result copies. Weight loading
is separate, 6.981657113 seconds; the full supervised child took
52.732588619 seconds. CPU assembly was 44.654/49.718 ms for the two observed
chunks, submission 6631.253/6572.907 ms, and existing fence waits
0.078/0.079 ms. Submission overlaps device work and includes the existing
result-copy behavior: it is not a measurement of dispatch overhead.

Every complete original PlanStep is rebound against the same registry
outside the measured wall. Its implementation identity and original node
group match the corresponding whole BoundGraph step; concatenating the
steps reproduces the complete plan. The ordinary controls retain
BoundGraph::Run. Distinct precreated events bracket all 3785 steps of each
chunk, plus input and result copies. No new completion fence is introduced;
event elapsed times are read after the existing completed chunk fence.
There are 15148 event objects. CUDA current-free drops by 57,139,200 bytes
after their creation; this observation is not a memory-qualification gate.

| Dependency chain | Step intervals | Event milliseconds |
| --- | ---: | ---: |
| Routed FFN | 774 | 4283.788467 |
| Attention | 126 | 3024.521674 |
| Attention output and HC expansion | 430 | 2012.545824 |
| Q/KV | 604 | 1362.917024 |
| Shared FFN and HC expansion | 516 | 719.079584 |
| FFN input and routing | 1972 | 630.258080 |
| HC attention input | 1118 | 559.763808 |
| Compression and indexer | 1204 | 531.895072 |
| Mixed Q/KV and compression dependencies | 84 | 45.067552 |
| Unclassified | 706 | 6.610144 |
| Embedding and HC initialization | 2 | 3.318720 |
| Frontier head | 34 | 5.502624 |
| Input copies | 2 | 9.729664 |
| Result copies | 2 | 0.070688 |

These ordered event intervals total 13,195.068925 ms. They are not a
kernel-busy trace; host enqueue gaps can fall inside an event pair. The
remaining difference from whole wall is not assigned to a glue-gap cause.
Classification follows actual `src` dependencies and external graph-named
frontiers. Normalized `view_src` aliases are used for state destinations,
not as dependency edges that bypass a frontier. All ordered rows remain;
mixed and unclassified intervals are explicit rather than reassigned by
shape or kernel name.

The exact post-down weighting and six-slot sum account for 516 steps,
665.472256 ms: one MUL and five ascending ADDs in every layer/chunk. This is
about 5% of observed wall and 15.5% of routed FFN intervals, sufficient for
one real-operand 4096-row reduction screen. The proposed candidate changes
only that materialization/reduction sequence; it must calibrate against
complete original output bytes before a performance inference. The stage
budget is not a predicted whole-prefill gain.

The operation transfers to route-weighted MoE reductions with the same
post-down weighting, slot order, contiguous F32 operands and separate
rounding. The current candidate is bounded to width4096/six slots; other
widths, slots or layouts need an explicit contract. Qwen's expert scales,
shared addition and alternative weight placement, and EXL3's differing
layouts/precision, block blanket substitution. Multiplexed state and paid
packing require separate controls; this screen is one request on stream0.

Provenance: actual model receipt
`5ec146e8c98818f85da50f9ffde1f98873be87666e964649b0cf8176653abe7c`,
native receipt `bacace0351ecaa1d88b6481a2ab45d3469a3b3819bad30a70282ca08077787b1`,
build receipt `64ba681e8ee646a0d06a4384bd42d05d65953a3ffd46210b5c30de95e5dc1e0a`,
external binary `6b566a20c7a228e24ed07451045c53ccbcef9477bbbd73c2360a216e52403ae1`,
and classification-r3 `a72e05f06971a0405cfe9f408318bada3cda884beb87e84468b7086f7ea32506`.
The selected native source map is `5553813b…`, qualified by `86b8b9db…`,
using actual A SDK `f38891fc…` and unchanged resolved SDK cuBLAS payloads.
Chunk0 head is `d7481e856eaf08617cc325452a109f4abccffc6bf272372183eb17d2883829b9`;
chunk1 is `fa7df546fa88bd1523d1a332da4d7028d1db95a55bdfecadb31790b5c593c786`.
All three copies per chunk retain 129280 finite F32 values; no head is all zero.
The child and supervisor completed rc0, inherited process group was reaped,
and final strong retirement was 117.306 GiB with no GPU/container/native
work. Raw plans, ordered rows, heads, logs and controllers remain outside
Git at `/home/pmeenan/scratch/m3-ds4-prefill-attribution-records/`.

## First captured-operand reduction screen

One short O/C/C/O operator screen on Spark A replaces the ordinary F32 MUL
and five ascending ADD launches with one ordered kernel. GPU phase medians
are 8.084000/1.883136/1.877888/7.684096 ms, with one warmup and three paid
samples per phase. Pooled phase medians give **4.19250× operator rate**
(76.15% less elapsed time); the corresponding host-wall ratio is 4.13373×.
Ordinary bookends move 5.20%, candidate bookends 0.28%. This is one actual
layer/chunk's resident operands, not a whole-prefill speedup.

The unchanged production graph captured F32 down `[4096,6,4096,1]`, scaled
weights `[1,6,4096,1]` and the final `[4096,4096,1,1]` result on its existing
stream. Every warm and timed result contains 16,777,216 finite floats and
matches all captured bytes, SHA-256
`f2eee85f7b60aa59e5ab79080c285d8b14c93af96bea5c4504f40dfb58bf53ea`.
All inputs remain unchanged and ten allocation tail guards match. Four
complete frontier heads match the preceding production heads. Candidate
scratch is zero; the capture/replay diagnostics own 1,275,169,280 device
bytes including guards, which is not a sampled model-memory result.
Retained SASS has six separate multiplies and five ordered additions,
without FMA; the observed host CUDA `cuobjdump` is recorded separately
from the SDK compiler.

Actual screen receipt is
`1772695114e626b8ea6430ad962a2044bf69cf0c6f9d465e8687c033b0bb8a39`,
build `451f3cb2…`, binary `2ee7116e…`, and operator receipt `fb91cd50…`.
The supervised native child completed zero and was reaped after 42.088 s;
independent retirement recorded 117.348 GiB free with model probes clear.
Failed compile/inspection-path attempts are retained and excluded. Raw
operands, complete outputs and receipts remain outside Git under
`/home/pmeenan/scratch/m3-ds4-weighted-reduction-records/` and the corresponding
Spark A scratch directory. Routine suites were deferred for this diagnostic.

This signal selected one complete 8K before/candidate/after prefill screen.
Its placement must keep down, weights and final output simultaneously live:
the old plan can reuse down storage before the last ADD is born. Keep all
intermediate allocations charged for the first substitution. The earlier
665 ms budget limits expectations; no default, quality, concurrency,
memory or whole-model gain follows from this operator result.

## Complete 8K reduction screen

One ordinary/candidate/ordinary pass on the same input and native contracts
takes **13.303123922/12.705059177/13.200088719 s**. Against the ordinary
bookend mean of 13.2516063205 s, the candidate supplies **4.30181% more
tokens/s**, or 4.12438% less time. Ordinary bookends move 0.78056%. This
single candidate pass is a useful adoption signal, without a repeat-stability
qualification or a context ladder.

Each chunk retains all 3785 original steps and its original complete
signature. The private candidate replaces only 43 six-step reductions per
chunk with the screened ordered kernel. Placement keeps all three operands
simultaneously live and retains every original intermediate allocation;
removed intermediate values are neither computed nor exposed. Both arms
pay the same maximum activation allocation, **1,891,631,104 bytes**.
Ordinary/candidate extents are 1,462,430,464/1,512,794,880 bytes. Both also
pay 190,840,832 scratch, 207,618,048 pinned staging, 264,126,464 state and
33,554,432 cuBLAS workspace bytes. These are declared capacities, not a
sampled model-memory gate. Input assembly/copies, all launches, head copies
and existing completion fences remain inside whole-prefill wall; the
6.980053 s load, clears, prebinding and report I/O are outside.

All six heads retain 129280 finite F32 values, with no head all zero, and
match every byte of the corresponding production head above. The same
numerical kernel and production compiler/device-math flags retain six
separate multiplies and five ascending additions without FMA. Main and the
warm source/build remain unchanged; this private substitution does not
enable a production default or qualify concurrency.

Screen receipt is
`8f435e3f51a6dd1bb49f75472cee13cbbe2dabd087292f1b0235f207e34a8bb9`,
native `e80f08e6…`, plans `a7f0a123…`, build `ff9091e6…`, and binary
`75e5a1e9…`. The native child completes zero in 51.381099 s and is reaped;
terminal admission records 117.352 GiB free with model probes clear. Raw
records and all six heads remain under
`/home/pmeenan/scratch/m3-ds4-weighted-prefill-records/`. Routine suites
were deferred for this diagnostic. The chosen next implementation is a
normal native graph/registry operation with the same bounded F32 contract,
preserving Exact, the existing small-row combine and ordinary fallback.

## Native graph adoption

The selected operation is now a normal two-input native graph node. Its
direct dependencies keep down outputs, weights and its distinct result live
together. Fast wide rows 9–4096 select it only for canonical F32 width 4096
and six experts. Exact, the existing small-row MoE combine and unsupported
shapes keep their original paths. The compiled kernel has six FMUL and five
FADD instructions, no FFMA, 24 registers and no local/shared/stack storage.

Three normal production 8K confirmation passes take
**13.042181402/12.871286353/12.910588913 s**, mean 12.941352223 s.
They contain no event marks and use the normal bound graph. Every one of
the six complete heads still matches the original production head bytes.
These passes confirm the selected implementation; the causal speed result
remains the preceding 4.30% bookended substitution. Each chunk now has
3570 steps, including 43 ordered reductions. The measured graph extent is
1,462,430,464 bytes; allocated activation capacity is 1,828,716,544 bytes,
scratch 190,840,832, staging 207,618,048 and state 264,126,464. These
capacities do not substitute for a sampled model-memory comparison.

On Spark A, the final locked check passes 1245 tests, including 258 GPU
tests. They cover full output equality at rows 1/9/4096, short/wrapping
bounds, aliases, alignment, strides and graph selection/fallback. SDK
format/tidy, boundary, REUSE and header checks pass. Check receipt is
`cab2e5f5…`, target `06762afe…`, and the 1266-file source map
`5e3664ab…`. Production confirmation receipt is `2b95ae89…`, native
`84f203f9…`, build `fa63a149…` and binary `4887d405…`; its child exits
zero in 50.913893 s and is reaped, with 116.574 GiB free at terminal
admission. Raw records and all six heads remain under
`/home/pmeenan/scratch/m3-ds4-ordered-reduce-final-records/`.
Workstation checks remain deferred to the settled optimization run.

The transferable mechanism is eliminating intermediate route-weighted
expert arrays while preserving weighting placement and ordered summation.
Qwen's ten-expert sorted NVFP4 combine already implements its own combine;
this six-expert F32 kernel does not match it. Another MoE consumer needs
its own matching layout/precision/order contract and measured screen.

## Wide HC first screen: keep the current path

Extending the existing small-row HC mix/pre kernels to 4096-row prefill is
**2.18% slower** in one complete 8K ordinary/candidate/ordinary screen on
Spark A. Paid times are 12.757279643/13.045291677/12.763420386 s; ordinary
bookends move by 0.0481%. This negative selects no production change,
quality ladder or broader timing matrix.

The community artifact's 86 HC matrices are F16. Each candidate chunk pays
86 existing F16-to-F32 graph conversions, with 135266304 logical output
bytes, before 86 native HC mix and 86 HC pre operations. The existing CUDA
bodies and flags are unchanged. A narrow private planner mapping connects
CPY to the existing packed F16/F32 converter and bound validator. Both
chunks drop from 3570 to 2710 steps, but fewer steps do not yield a speed
gain. Expert products and all 43 ordered reductions per chunk remain the
same. Allocated activation rises by 46137344 bytes to 1874853888; scratch,
staging and state capacities are unchanged.

All four ordinary full heads match the selected production bytes. Both
candidate heads are finite and retain the same argmax, but all 129280
values differ: maximum absolute differences are 0.886101/0.841692, RMS
0.160280/0.159910 and NMSE 0.001125/0.000772. These two heads establish no
task-quality approval. Preserve the ordinary projection and investigate
the mixing/output-normalization suffix separately if its own paid screen
warrants it; this whole screen attributes no phase cost.

The initial attempt fell back because the HC weight-type guard excluded
F16. The paid-conversion retry then refused the missing CPY planner route
before candidate prefill. Both attempts and their completed outputs are
retained. The completed screen binds source graph 974a3aa4, planner
406f505b, study f1f87215, controller 63086f39, build 5d2e92ed and executable
a3f9b79b to SDK f38891fc and unchanged resolved cuBLAS. Actual outer receipt
is `1b8e5df488b95ba186001ac4f1aa882567ce4602b961ad2f85edceb551776881`;
all eleven commands complete zero and are reaped, with 117.242 GiB free
and model probes clear at retirement. Six full heads and small receipts
remain outside Git under
`/home/pmeenan/scratch/m3-ds4-wide-hc-short-records/model-r3/`.
