<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Concurrent generation comparisons

The owner requires both one request and concurrent generation requests in
engine comparisons (2026-10-01). CUDA execution streams are a separate
implementation detail. Existing single-request/operator results establish
only their stated scope; they do not qualify concurrent serving.

Measure one resident engine process with 1, 2 and 4 independent requests for
the same model. Submit each burst from a common client barrier with separate
connections. Start at 8K, then 32K; extend the matched long-context ladder
within each engine's real admission budget. Keep prompt token IDs, output
limit, greedy sampling, cache precision, speculative/plain mode and model
weights explicit. Report unsupported or refused combinations, never remove
them from the report. Different weight formats remain separate comparisons.

For Qwen, compare llmpalooza, Mia and TensorFold. TensorFold's existing
71377a53 image supports Flash Next's shared CUDA forwards with explicit
`--parallel 4`; its default `auto` selects one request. Test both the default
single-request deployment and the explicit concurrent deployment. The pinned
MLX affine 4-bit checkpoint is cross-quantization. Current TensorFold
c4646171139ee8a3c38103eaa1699dad226ec12b also documents support for the exact
Mia NVFP4 checkpoint; that is a separately pinned follow-up, with its
actual arithmetic and quality evidence recorded before treating it as an
equivalent oracle. GLM on CUDA is currently documented as serial, including
two ranks; its claim must be checked again at M4 entry.

Use literal completions with identical prepared strings and independently
prove the original and affine tokenizer produce identical packed IDs.
Record server prompt counts too. Unique beginning and end markers prevent
accidental shared-prefix reuse during fresh-burst comparisons. Reused prompts
are a separate measured condition. Every member receives the same solo
output control; TensorFold's returned token IDs/usage authenticate output
counts and exact solo/concurrent equality. Require actual `cached == 0`
telemetry for every fresh timed TensorFold request. Standard API engines without
token-ID extensions retain complete output text, usage and finish records;
do not claim token identity by counting SSE events or re-tokenizing output.

Publish burst throughput as the sum of actual completion tokens divided by
first submission to last terminal event. Report each request's time to first
visible text, terminal latency and end-to-end completion-token rate, plus the
median and worst request. Stream chunks can contain several tokens; stream
gap statistics are chunk-delivery latency, not per-token latency. With this
small sample, publish raw per-request values and maxima rather than claiming
a stable p99. Include sampled node memory, process memory where attributable,
errors, refusals and completed output lengths. Early EOS is retained as a
short result; it is not a fixed-length throughput comparison.

First isolate fresh prompt-plus-generation bursts. Warm conversation branches
and requests arriving during an existing decode are separate follow-ups:
their prefix policy and prefill interference can materially change latency.
Keep cold engine startup separate from warm serving throughput. A single
request on the concurrent deployment is also measured, because shared
forwards may disable its single-request CUDA graphs.

Llmpalooza's Qwen serving path now retains four independent branches and funds
two active requests, alternating prefill chunks and sharing eligible decode
products. A third request waits for a retirement before entering; other
families still serve one active request. Report these actual admission limits
and queueing. A concurrent-request client alone does not establish batched
execution or scaling, and solo results do not establish concurrent parity.
Shared-weight changes retain quality, state isolation, cancellation and
memory admission checks.

All Spark jobs are exclusive per node and supervised by `tools/spark-job`,
with strong memory/process admission and terminal retirement. Run only the
needed benchmark/controller checks while isolating performance. No unit
suite or swap table is part of each comparison; the final implementation
receives its production checks. Raw responses, timings and source inventories
stay in external scratch, with aggregate results and provenance in Git.

## First bounded TensorFold matrix

On Spark B, use the existing `71377a53` image
`sha256:1a2afff2bd001cdea746d96bdb7614a4f937caa28e1646babd9acb951ea3aa48`
and the pinned Vontra `dadefa80` MLX affine 4-bit/group-32 checkpoint. One
CUDA rank, `--parallel 4 --context 33792`, BF16 KV, greedy requests, MTP
default cap 6 and confidence 0.30. Each literal input is exactly 8,192 IDs;
reply cap is 256. Six fresh bursts cover plain/speculative × 1/2/4 requests,
one observation per cell. Every one of the fourteen requests receives a
separate matching solo output control on the same deployment (twenty-eight
requests total). Solo controls may resume their repeated prompt; their
timings are excluded from the fresh-burst performance table. Early stop is
retained and marked as a shorter workload. This initial screen estimates
scaling; repeat decision-relevant cells before establishing a gate.

Allow approximately 6–8 minutes including the historical 143-second warm
startup. The default single-request graph deployment and the 32K matrix
are separate next runs. No TensorFold/native parity claim follows from
this TensorFold-only initial baseline.
