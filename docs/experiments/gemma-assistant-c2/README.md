<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma assistant C2 identical-input and short timing screen

Native serial and joined assistant queries reproduce all six original serial-C2
heads and postprojection features byte for byte, including endogenous three-step
recurrence. Both differ from the original physical batch-two arithmetic despite
matching its six greedy choices. The short joined timer is faster than the
original physical batch-two timer; this does not qualify assistant quality,
optimized batching, end-to-end speculation, or serving.

The unchanged production graph executes the approved Q8 26B assistant against
separate frozen inputs from the original target. Owners have distinct histories:
P=64 and P=65, normalized target feature at P−1 and pending anchor at P. Each
owner reads local layer 28 (D256, KV8) and global layer 29 (D512, KV2), with
physical capacities 1,280/4,096 and read width 256. Query position stays P for
all assistant steps. Neither assistant path writes target caches.

## Math and immutable inputs

The new manual helper binds the exact previously vocabulary-qualified target
and assistant artifacts; it rechecks their tensor/target contracts but performs
no new vocabulary-parser scan. Its ordinary primitive policy and production
kernels remain unchanged. Serial executes two single-owner plans; joined
executes an actual two-segment plan. Each segment retains the existing local
attention selector and Q sample dimension one. No forced MMA, norm fusion,
shared-Q8 writer, MoE policy, or generic fusion flag was introduced.

A source/input/binary receipt preceded native endogenous one-step and three-step
first/repeat controls. Each policy repeated exactly, and one-step output equaled
step zero of its three-step chain. The exclusive native freeze
`be500f4abca28f0fe7fad1eecc9a9242161a53e871944e822a4644cccedcde26`
was created after official retirement, before any original C2 recurrent inputs
or outputs were released. The original serial and physical batch-two packages
were then admitted separately. Posthoc runs use that arm's incoming feature and
anchor at every step, preserving identical inputs even if recurrence differs.
The native helper, graph, kernels, and original timing client were already
compiled and remained unchanged throughout this exposure.

| Original arm | Native policy | Exact heads/features | Strict greedy differences | Max head delta | Max feature delta | Max TV | Max absolute chosen-NLL delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Serial C2 | Serial | 6/6 and 6/6 | 0/6 | 0 | 0 | 0 | 0 |
| Serial C2 | Joined | 6/6 and 6/6 | 0/6 | 0 | 0 | 0 | 0 |
| Physical batch two | Serial | 0/6 and 0/6 | 0/6 | 0.40007496 | 0.41434097 | 0.01985614 | 0.03046731 |
| Physical batch two | Joined | 0/6 and 0/6 | 0/6 | 0.40007496 | 0.41434097 | 0.01985614 | 0.03046731 |

These are six fixed full-vocabulary rows, not corpus PPL. TV uses FP64-normalized
complete heads. Chosen-NLL delta is native minus reference NLL at the reference
winner; the table reports its maximum absolute magnitude, not a signed mean.
Own byte-repeat movement is zero for each native policy. There are no strict
or positive-margin winner differences in these six rows, but distribution drift
is retained. The sample does not establish a near-tie or distribution allowance.
The distinct physical batch-two execution policy may affect arithmetic; this
screen does not isolate its first changed operator or prove a cause.

All borrowed physical K/V cells were initialized: canonical stage0 read bytes
plus explicit zero tails. Funded, completion-ordered copies compared every byte
of both owners' complete local/global physical arrays before and after each
chain. The original client separately checked complete physical caches,
positions/membership and opaque target state. Both remained unchanged. These
native witnesses cover the borrowed layers, not a fabricated whole-model target
checkpoint. The helper never retags original cache arrays as native state.

## Paid 32-wave screen

All arms ran on the same Spark (`spark-c4e2`, GB10). A wave publishes two complete
heads of 262,144 F32 values and two features of 2,816 F32 values. Thirty-two waves
cycle the same three admitted incoming fixtures, yielding 64 completed rows per
arm. Three untimed fixed-shape waves precede timing. Timers include host input
staging, completed execution, full publication, finite scans and lower-index
argmax. Setup, target prefill, disk writes and full-cache witnesses are outside
the timer. Cache/state witnesses bracket the paid block in both engines.

| Fixed incoming arm | Original first (s) | Native serial (s) | Native joined (s) | Original repeat (s) |
| --- | --- | --- | --- | --- |
| Serial C2, original serial execution | 0.182074716 | 0.162831199 | 0.099268342 | 0.184292833 |
| Physical batch two, original joined execution | 0.120501433 | 0.163458747 | 0.101180635 | 0.119398257 |

Native capture delta is zero in paid blocks; replay delta is 64 for serial and
32 for joined. Original GGML graph-reuse delta is zero for serial and 32 for
physical batch two. This counter does not prove CUDA replay; original CUDA
graphs and fusion stayed enabled, with its default library logging retained.
Native joining reduces the number of completed jobs and full-head publication
boundaries from two per wave to one. No production selection follows from a
single approximately 0.1–0.2 s paid block.

All 64 paid winner IDs agree in every arm. Each original bookend's final full
heads/features repeat exactly and match the independently retained step-one
fixture (the final wave index is 31 modulo 3). With serial-C2 incoming tensors,
native final paid vectors match original serial reference bytes; the physical
batch-two differences remain. Intermediate
paid vectors are published and finite-checked but are not retained or claimed
as a whole paid-vector comparison. Only correctness-chain complete vectors,
all paid winners, and final paid full vectors are retained externally.

This manual helper pays host feature staging, whereas the production component
has dedicated device recurrence. It loads only the target embedding dependency
plus assistant weights and maps its own frozen source arrays; the original
client sets up the full target before timing. No target setup, peak-memory,
whole-engine performance or accepted-proposal rate comparison is claimed.
Target numerical qualification and native component C2 optimized policy,
device-mask transfer and end-to-end verification remain open. The unchanged C1
helper and its calibrated history remain in
[the native foundation](../gemma-assistant-execution/README.md).

## Reproduction and provenance

Base `a6dd81f`; the new helper is
[gemma_assistant_wave_fixture.cc](../../../benchmarks/gemma_assistant_wave_fixture.cc).
[PROTOCOL.md](PROTOCOL.md), [reference.sh](reference.sh),
[llama_assistant_timing.cc](llama_assistant_timing.cc), [compare.py](compare.py),
[provenance.json](provenance.json) and [results.json](results.json) record the
closed protocol, pins, source/build identities and aggregate results. The
original source is `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, image digest
`837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`.
Only the thin client is compiled; original `/app` math libraries are unchanged.
Task-entry TensorFold at 2026-10-05 10:29:07 UTC was
`609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5: its Gemma26 recipe
was MLX-only, with no same-format CUDA Q8 assistant comparator.

The canonical target input is the first 1,024 War-and-Peace IDs from the
[representative fixture](../gemma-quality/README.md), 4,096 bytes, SHA
`b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610`.
The original target contexts and assistant context are constructed before
prefill; owner prefixes contain the first 64/65 IDs. New reference timing
reconstruction verifies all 22 admitted stage0 files byte for byte, including
exact source descriptor pitches/offsets and physical cell maps, before timing.

Supply the authenticated external input-only owner directories and manifests
as `INPUT_ROOT/owner-{0,1}` and `owner-{0,1}-manifest.json`; do not include original
later data before the own freeze. Run the new manual target with
`INPUT_ROOT NEW_OUTPUT`. After the exclusive own freeze, separately supply
original serial/batch `incoming-feature.f32` (3×2×2,816 F32) and
`incoming-anchor.i32` (3×2 I32); append their directory for posthoc replay, and
`time` for the bounded paid block. Frozen admission receipts authenticate exact source, binary and input
sizes/hashes; post-run verification proves unchanged identities and official
retirement. The pinned-image
wrapper uses the same stage0 root and incoming files. Raw vectors, winner IDs,
logs and receipts remain outside Git under
`~/.local/share/jitllm/gemma-assistant-c2` on Spark and the external coordination
scratch. Aggregate results contain no per-wave token arrays. Raw supervisor
timestamps are EDT; provenance also records UTC.

The locked Spark build, native own repeats, posthoc controls, six-invocation/eight-block bookend,
final-byte/64-winner verification and six analyzer controls passed; light
format, REUSE/header and portability checks are recorded with the final bundle.

An analyzer-only review correction compares the reference winner against the
native chosen ID for positive-margin differences, while retaining the reference
top-two margin separately. A tied-top control and all four retained comparisons
passed in the final six-control analysis; measured aggregates and helper binaries
are unchanged. Earlier acquisition/analyzer identities remain in provenance.
