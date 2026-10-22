<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared GDN gates: first U7a result, 2026-10-22

The shared GDN pointwise gate kernel now accepts positive head and row counts
whose two F32 output planes fit GGML's signed byte-offset range. Its capability,
constructor and bound-tensor check agree on operand shapes and encodings; the
constructor rejects unrepresentable products before allocating GGML metadata.
The bound check retains alignment, disjoint-output and current-view checks and
checks strides before calling GGML's contiguous-layout helpers. CUDA uses wide
indices and a grid-stride loop, retaining the original rounded addition,
softplus threshold, multiplication and sigmoid expressions. The first row
requires no head-index division. No model dimension selects this implementation.

## Registered first screen

Registration: `963997c3422d531e6728424ed06c02463ae8d3fa`, [protocol](README.md).
One exclusive Spark B job ran all three A/B/A triplets, without an intervening
build or retry. Every process completed, paid 128 captured kernel nodes per
replay and 128 replays per sample, and produced the same output hash. Each
median uses seven samples after four warmups. The table reports captured GPU
execution per kernel; it does not measure host construction/checking or model
throughput.

| Triplet | A before ns | B ns | A after ns | B / mean A | A bookend movement |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 751.785 | 736.297 | 750.816 | 0.980029 | 0.12894% |
| 2 | 752.289 | 736.289 | 755.641 | 0.976556 | 0.44452% |
| 3 | 752.277 | 736.283 | 751.775 | 0.979066 | 0.06674% |

Median ratio **0.979066** passes the prospectively fixed maximum of 1.10;
every baseline bookend movement is below the fixed 10% instability bound.
The observation is about 2.09% lower captured time at 48 heads and one row.
It supports retaining competitive speed for that existing shape, not a
whole-model speed claim or a fusion-rank update. Other shapes have correctness
coverage, not newly measured speed claims.

## Correctness and provenance

The expanded GGML-chain test covers `(heads, rows)` pairs `(48, 1/2/3/4/8/16)`,
`(1,17)`, `(7,33)`, `(31,17)`, `(37,19)`, `(65,17)`, `(257,17)` and `(513,1)`.
It checks both output planes, their views and saved-row consumers bit for bit,
including the original FTZ and threshold edge cases. Negative tests cover
mismatched and extreme extents, doubled-storage overflow, malformed strides,
output aliases, misalignment and stale views. Fifty focused gate, lowering and
component tests passed with no skipped entries on Spark B. After rebuilding the
181 selected consumer targets against the final module, including every shared
test consumer and the affected model target, the
fresh CTest inventory selected 3,624 shared entries (`-LE models|timing`) and
six affected Qwen3.8 model controls. Both sets passed once, with exact-name XML
audits reporting zero missing, unexpected, duplicate, failed or skipped entries.
The six controls each executed the installed ModelOpt and GGUF cases, covering
finite/repeatable steps, saved recurrence, zero/one/two accepted verification
rows and invalid arguments. These are functional controls, not quality or
whole-model performance qualification. CTest was the pinned SDK 4.4.3.

Both arms use the pinned `spark-native` RelWithDebInfo SDK
`aarch64-2c4eb5b2f68f3fa5`, GB10, driver `580.178.04` (driver API 13000), and
CUDA runtime 13040. The baseline checkout is `eadf8da8` plus the registration;
its GDN code is unchanged from `48bbb5ed`. The candidate starts at `48bbb5ed`
with the registration and this slice's six source/test changes. Their complete
source manifests differ only in those six files and eight unrelated Gemma
experiment files added by `eadf8da8`; no other compiled source differs.
[Provenance](provenance.json) records actual binary, source-manifest, compiler,
selection, collector and enrollment hashes. The stale Git version string in
the warm Spark copy is not used as a source identity. The earlier candidate
binary preceding the registry-description correction was superseded and was
not timed.

The fixture holds 1,152 device bytes and the launcher requires zero scratch.
Graph/event/runtime/module allocations are separate. NVIDIA's ordinary memory
counters report N/A on this unified-memory GB10, so no measured total-memory
claim is made. GPU temperatures before/after the job were 43/45°C; clocks were
not changed. Raw samples, logs and full machine receipts remain outside Git.

This changes the compiler-owned module identity to
`374364310fbe78618b0f332a189bb8d9db18fde6eea5ea646892aa088429c50e`.
Historical W5 rows retain their original identities and remain stale. This
spike closes only the GDN gate head/row whitelist; the other U7a kernels and
U7b–e remain open.
