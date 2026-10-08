<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma state-only intermediate prefill

Non-final, non-scoring Gemma prompt chunks now keep all KV writes and omit
unused final-layer query, attention, output, FFN and vocabulary work. The
explicit output mode belongs to the plan-cache key. Successful state-only
chunks clear their published logits; clean pre-dispatch refusals preserve the
completed prefix. Unknown completion retains the existing quarantine rules.
Final chunks, scoring and retained-feature requests use the full path. Startup measurement funds both graph modes before execution.

| 8K prefill recipe | Fresh full, s | State-only repeats, s | Change versus full |
| --- | ---: | ---: | ---: |
| Gemma 26 all1024, normmul off | 2.72628 | 2.64598 / 2.64089 | −3.04% |
| Gemma 31 both256, normmul off | 12.0496 | 11.7857 / 11.7303 | −2.42% |

These percentages describe the existing optional all/both research policies.
Ordinary serving has separate correctness controls, without a claimed ordinary
policy performance percentage. Every native arm preserves its prior complete
prefill/final heads, initialized state and 32 choices. The state footprints are
592,445,440 bytes for 26B and 1,740,636,160 bytes for 31B, including local-ring
wraparound. Each state-only repeat is exact within its policy.

The matched ring-cache references keep SWA/unified false, F16 KV, context
16,384 and physical ubatch 1,024/256. Local capacities are 2,048/1,280; global
capacity is 16,384. Gemma 31 reference prefill 10.8658/10.8510 s gives a remaining 8.28%
latency gap; decode remains 2.38% above its reference mean. Gemma 26 references
2.42188/2.65986 s move 9.83%, so their mean does not establish a resolved
competitive gap. A short follow-up unchanged reference 2.42238/full-native
2.71430 s returned to the initial range. All follow-up heads, choices and native
state match the fixed baselines. The outlying reference remains in the record.
An idle 40°C/P8 clock query does not explain its earlier paid timing.

Both arms use the same 8,227 canonical IDs, six discarded warm rows, Clear,
8K prefill, three anchors and 32 forced units. The native full arm publishes
8/32 heads during prefill, while state-only and reference publish one. The
32 forced units still pay one lower-index incoming-head argmax scan and full
publication each. Timers, graphs, masks, cache policy and arithmetic choices
remain unchanged. The native reported prefill counters describe the last built
full final-chunk plan; intermediate state-only plans intentionally omit the tail.

The explicit graph cut supports both approved Gemmas. Generic runtime callers
use a default virtual prefill hook that preserves existing full-head behavior
for Qwen and DeepSeek. Their MTP/features, hyper-connections and recurrent state
need separate dependency-cut qualification before adopting head omission.
No production fusion flag or optimized batching qualification changes here.

Graph controls reject malformed modes, partial-layer/hidden-input/no-head
combinations and feature outputs, and verify every KV store with no hidden
output. Placed plans accept empty frontier sources. Runtime controls cover
partial cancel/resume, scoring and unchanged default families. Four actual
GPU policy controls compare scalar and same-shape unequal-row/past waves,
initialized KV bytes, full continuation heads/state, whole-wave refusal and
captured replay. Feature retention is checked under its held request lease.
Local REUSE/header checks pass for 1,595 files; 394 portability boundaries,
changed formatting, diff and JSON checks pass. The final production Spark
suite passed 1,708/1,708 targets without skips (725.19 s; 316 GPU, 62 model).

The initial build refused a signed fixture argument; its retry passed. A first
feature test attempted a borrow after the automatic request lease ended;
putting the test's borrow inside a held request passed. Production math was
unchanged across these fixture corrections. The expanded tests changed only
test bytes after timing; the measured helper identity remains retained.

Reproduce with manual `llmp_gemma_prefill ARTIFACT IDS NEW_OUT PROFILE POLICY
ROWS normmul-off full|state-only`, using 26/all/1024 or 31/both/256. The existing
owned-container ring launcher supplies the original bookends, retaining the
[26B](../gemma26-swa-ring-transfer/README.md) and
[31B](../gemma-swa-ring-h1/README.md) baseline recipes. [Results](results.json)
retain source, binary, input, aggregate and fixed-state identities. Raw job
logs, token telemetry, vectors and snapshots stay external. Source base is
59cba05; native NVCC is 13.4.92 (toolkit 13.4.2), and original-image math
uses CUDA 13.3. Task-entry TensorFold remained 609ca419/0.6.5 with a Gemma 26 MLX recipe, supplying no matching
CUDA comparison. Whole model quality and optimized batching gates remain open.
