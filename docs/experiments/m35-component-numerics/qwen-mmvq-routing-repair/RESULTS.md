<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen ordinary MMVQ routing qualification

The repaired five-ID Qwen head/state control passes exactly against the frozen
upstream reference. Its actual full-plan comparison and all six registered speed arms also pass.
All three median slowdown metrics are within the unchanged 1.10 bound.

The [original speed screen](../remaining-c1-speed/RESULTS.md) retains all six
valid Qwen samples and its failed 1.121499 first-token median against the 1.10
bound. The repair changes shared GGML selection: GB10 ordinary Q6_K/Q8_0 products
in the five-through-eight physical-column class favor existing upstream MMVQ,
sharing input preparation where eligible and keeping automatic GLUs separate.
The new performance observation is five-column work; this is not a measured
speed claim for every shape in that schedule class. Explicit product arithmetic,
accumulation and hints, directly supplied fused IR, other devices/encodings and
routed/grouped operations retain their contracts. No kernel arithmetic changes.

## Registered head/state and actual plans

The [protocol](PROTOCOL.md), disabled pins and exact identity map were committed
before collection at `1ef62b6a07a8634b3adc20e5db43107690bb6f4f`.
The unchanged runners bind a new candidate runtime
`8f1f22eaa2901d575dab00ccc2a2979b1ae533615f40982e3173267e83026081`
and Qwen test executable
`710a7c3c208a605b0d8418ffd47e4df3b268aaf62fe3fb667084ea2fe28e060d`.
The source/compiler ledger authenticates the changed capability predicate;
192/196 primary object identities match the frozen reference. The other four
are that capability source and the three previously reviewed reference planner
objects. Configuration and constant identities agree; the declaration ledger
changes only the authenticated capability source hash. All 168 reconstructed
reference implementation identities match exactly, without wildcard removal.

One exclusive supervised arm on Spark B (`spark-56f5`) completed in 10.06 seconds,
with one selected test, zero failures/skips, complete exports and unchanged
immutable inputs. All seven semantic records are byte-identical to the retained
reference: full-head and kept-row results, generated/history tokens, saved state,
exact spill/restore and continuation. This is a plain target control, so it emits
no speculative observer trace.

Independent relationships cover all three completed plans and 16 completed uses.
The five-row prefill contains 494 upstream prepared products (246 Q8_0 and
248 Q6_K), 388 Q8 preparations and 48 separate ordinary GLUs. Its three remaining
VecQ products have actual scalar gathered-head inputs and correctly remain
outside the selection change. The 4,299 reference descriptors match through the
proved operand/storage relationships, allowing only the intended shared input
preparation. Routed products, GDN, HC and attention remain unchanged; scalar
decode and continuation retain their previous route. Independent actual-gate
review SHA-256 is `5909b564358122931f4dcfb34928f2e6de023d9e24304b66a351d277b89315bd`;
relational review `9cdd34bc988fc8be3cb118547bce0575ac7bffa440879af96703478f7d9f0c9d`.

A preliminary invocation on an unintended host failed before opening the staged
runner; no model or enrolled output was created. Its setup-only receipt remains
separate from the single successful B arm. No measured or semantic sample was
retried or discarded.

## Speed and reused qualifications

The six speed arms retain the original corpus, 104/111 warmup/measured prompt
tokens, context 512, prefill chunk 5, one slot, temperature zero and 128 output
tokens per request. Pair order is A/R, R/A, A/R. A is the same frozen upstream
reference; R is the new candidate. Each arm must pass input, work and retirement
checks before its raw timestamps enter the unchanged median slowdown verdict.

All six arms complete validly with clean TERM/return-code-zero retirement,
unchanged inputs and no competing GPU process. All three pairs remain included.
Root and independent review recompute the same verdict from raw client times.

| Pair | Arm | TTFT (s) | Decode (tok/s) | Wall (tok/s) | Peak GiB |
|---:|:---:|---:|---:|---:|---:|
| 1 | A | 1.010695 | 33.138875 | 26.419005 | 52.549 |
| 1 | R | 1.008851 | 34.075746 | 27.021008 | 52.540 |
| 2 | R | 1.005996 | 34.123943 | 27.056306 | 52.526 |
| 2 | A | 1.008511 | 33.328680 | 26.553152 | 52.551 |
| 3 | A | 1.019172 | 32.514972 | 25.979941 | 52.531 |
| 3 | R | 1.014592 | 34.074988 | 26.979977 | 52.538 |

Median slower-candidate ratios are **0.997506 TTFT**, **0.972506 decode**
and **0.977721 wall**, all at or below **1.10**. Peak GiB is whole-node
`MemAvailable` loss, not GPU memory. The [aggregate](results.json) retains every
absolute observation, pair ratio and binding, including the original speed
failure. Independent speed review SHA-256 is
`3a95ae156975683469cf443273b5c867b3ae72a98d500897e697c2bc3a9dcb7e`;
100-file retention receipt `37f5b91980176cdb4c90bea2dcb67d465a783fb6b3085754c1741588a0bc646c`.

Source/domain review `f520d243a9b5e607d3adccceb4860d72f0052108d7e7bff3d1025c71f37c7125`
supports reuse of the other three families' head/state and C1 results, Gemma31 C2,
Qwen's two current bounded replies and its 996-target wide teacher result.
This joins actual prompt tails, configured draft bounds, component source and
prior complete plans; it is source-backed inference, not a new per-operation C1
trace. The two Qwen replies have 66-token prefill at chunk 4096 and scalar decode;
wide scoring has 997 rows. None enters the changed ordinary column class. No new
reply or teacher was collected or claimed.

Final native builds pass on both Sparks. Spark B passes all 174 related checks,
including three new routing controls. Spark A's final shared suite passes 3,696
cases, zero failures/errors/skips in 553.64 seconds, with clean source guards
before/after the build and tests. The exact static checks and final reviewed
source identities accompany closeout. Raw logs remain outside Git under
`/tmp/llmp-m35-orchestration/u4-qwen-mmvq-repair-{head,speed}-r1/` and their original
Spark experiment paths.

Frozen numerical differences, strict Mia failures and incomplete bounded replies
retain their original classification. This grants no new device, encoding,
maximum context, general batching or complete-answer qualification.
