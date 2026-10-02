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
