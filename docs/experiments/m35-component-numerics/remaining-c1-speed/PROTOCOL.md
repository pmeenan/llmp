<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Remaining U4 matched C1 speed qualification

This registration fixes the remaining affected-path speed screen before collection.
Collection requires a committed, independently reviewed protocol, static disabled
[pins](pins.json), the bound [executor](run-c1.py), and an external enabled receipt
naming that commit and their hashes. The positive head/state controls and bounded
reply checks must already pass. No timing is collected by the default preview.

Compare A, the retained current-source upstream-MMVQ reference, with R, the
repaired default candidate. Both use identical artifacts, configuration, input,
libraries, hardware and normal decoding policies. The scoped reference changes
eligible ordinary block products with implementation-chosen arithmetic and
one through eight columns to upstream MMVQ and separate GLU; explicit arithmetic
contracts, routed, grouped and wide work remain unchanged. The
[head/state registration](../remaining-domain-transfer/PROTOCOL.md) and its exact
implementation-identity crosswalk bind this contrast. This measures competitive
speed against the qualified upstream arithmetic; it does not compare against
the rejected inaccurate reduction schedule.

Run three pairs per family, in fixed order A/R, R/A, A/R. Each arm is a fresh
runtime process with one warmup C1 request (corpus prompt 0), then one measured
C1 request (prompt 1), temperature zero and exactly 128 output tokens. The
existing pinned `speed_gate.py` collector supplies the workload and computes
metrics; the executor overrides only the registered family prompt size, cells,
repeat count and output bound. There is no additional priming session. This is
a bounded C1 comparison, separate from the retained C2/C4 evidence.

| Family | Host | Corpus characters | Context | Prefill chunk | Slots | Decode mode |
|---|---|---:|---:|---:|---:|---|
| Gemma31 | spark-c4e2 | 26000 | 8192 | 256 | 4 | current assistant enabled |
| Gemma26 | spark-56f5 | 26000 | 8192 | 1024 | 4 | current assistant enabled |
| Qwen GGUF | spark-56f5 | 128 | 512 | 5 | 1 | target only |
| DeepSeek | spark-c4e2 | 30000 | 262144 | 4096 | 4 | current DSpark, plain wave |

The Qwen input is deliberately small so the five-column ordinary prefill path
is exercised without thousands of tiny chunks. Its recorded prompt must exceed
five tokens and fit the context with the generated continuation. Prompt identity,
usage and completion work must match between arms of every pair. The fixed
corpus, current artifact IDs and metadata, tokenizer/template, runtime binaries,
source manifests, actual resolved libraries and configurations are in the pins.

Run only through installed `spark-job`, one exclusive job per arm, timeout at
most 600 seconds with supervised retirement. Independent hosts run in parallel;
no build or other measurement overlaps on either host. Each arm preserves its
launch, raw collector record and runtime log, supervisor outcome, immutable
pre/post checks, request audit and recomputed metrics outside Git. Existing arm
outputs are refused. An invalid input, environment, lifecycle or request result
stops that family's collection without automatic retry. Preserve failures; a
changed protocol requires a new prospective registration and fresh namespace.

Every warmup and measured request must complete all 128 tokens, finish by the
length bound, have zero cached prompt tokens, finite positive timing and valid
accounting. No competing GPU process may be present. The service must retire
with TERM and return code zero; pinned inputs and resolved dependencies must
remain unchanged. Distinct generated wording is allowed; numerical and bounded
semantic qualification precedes this timing screen. These capped requests do
not establish complete answers.

Recompute C1 TTFT, decode throughput and wall throughput from the raw client
chunk timestamps using the pinned collector. For every pair report absolute
observations and slower-candidate ratios: R/A for TTFT, A/R for decode and wall
rates. A family passes when all six arms are valid and the median of its three
ratios is at most **1.10 for each metric**. No threshold or exclusion changes
after collection. Report peak whole-node `MemAvailable` loss in GiB separately;
it is not GPU memory and does not enter the speed verdict. Review actual work,
retirement and all ratios independently before adopting the result.

Reuse the unchanged native Qwen/MTP, image, teacher-score and prior scalar/C2
controls at their documented scope. This screen grants no new device, encoding,
maximum context, assistant accuracy beyond the registered controls, general
batching or full-task-completion qualification. Existing frozen-result and strict
Mia failures remain recorded and their expected values are unchanged.
