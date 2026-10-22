<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# RMSNormMul versus its own decomposition: measured first class

**Historical first-class matrix; original identities are stale for current source.**
The reviewed trait-dispatch and compiler-owned identity changes require a new
source freeze and registration before remeasurement; see
[SOURCE-IDENTITY.md](SOURCE-IDENTITY.md). The twelve observations and their original
identity fields below are unchanged. They do not qualify current selection.

**Complete first-class matrix; proposed rank data, not installed or integration-qualified.**
All twelve prospectively listed F32 RMSNormMul cells have valid measurements:
eleven favor fusion by more than 10% lower latency; 4096×1 is neutral. The
4096×256 cell passed under the separately registered v12 capacity repair;
its original v11 refusal remains preserved.
The first 1024×1 screen measured 1,372 ns fused versus 1,807 ns decomposed,
24.073% lower latency. This is one operator on one GB10, not a model-speed,
model-quality or full-U6 result. Other fusions and shape/format domains remain open.

## Method and scope

[Registration](README.md) commit
`251475f9676b137692d95a9943c469b2af0c50d4` preceded the first eleven measurements.
The first successful screen was reviewed before the remaining, already
registered width/row matrix ran. The final 4096×256 measurement followed
independent review, full validation and capacity-amendment registration
`5d6fea5528e21b04209cdefc1540a4b9220511b4`; no earlier valid cell was rerun. All jobs used exclusive GPU ownership on
`spark-b` (`spark-56f5`, NVIDIA GB10, `sm_121`, 48 multiprocessors), CUDA driver
API 13000 and runtime API 13040. Each had a 600-second supervisor limit,
30-second termination grace and the reviewed 540-second child bound.

Both arms use production planning/lowering from the fused node's own expanded
IR, identical deterministic F32 inputs/scales, epsilon 1e-6 and complete outputs.
Only the primitive arm disables the tested rewrite. The logical decomposition
is ordered RMSNorm then Mul; its physical recipe is the reviewed two-launch
wrapper, not a different arithmetic comparator. Each event sample replays one
captured graph containing 128 complete chains. All twelve valid cells reported
128 fused versus 256 primitive graph nodes, matching output after capture/replay,
relative RMS difference 0 and maximum absolute difference 0. The unchanged
prospective bounds were finite output, relative RMS ≤1e-6 and max absolute ≤2e-5.
These are comparator checks on these deterministic fixtures, not general quality proof.

Four initial paired warmups were excluded, followed by a primitive bookend,
four alternating paired warmups, nine alternating fused-first/decomposition-first
pairs, and the final primitive bookend. The raw stream retains each warmup,
all nine pairs in execution order and both bookends. Independent recomputation
matched every summary pair, rounded integer median and classification. The
largest bookend drift was 0.897009%, below the unchanged 10% refusal bound.
Drift is `100 × (max(bookends)/min(bookends) − 1)`. No run was discarded for
an unfavorable timing, no endpoint was extrapolated and no bound changed.

## Measured cells

Times are median nanoseconds **per complete chain**, rounded as in the harness;
parentheses show the nine unrounded samples' minimum–maximum, displayed to
one decimal. Latency change is `100 × (fused/decomposition − 1)` using the
recorded integer medians. Fusion requires latency strictly below 90% of its
own decomposition; exactly 90% and 110% remain neutral. No measured cell
requires a losing-fusion veto.

| Width × rows | Fused ns (range) | Own decomposition ns (range) | Latency change | Bookend drift | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 256 × 1 | 973 (970.0–975.2) | 1,449 (1439.5–1457.0) | -32.850% | 0.138890% | Fusion |
| 256 × 4 | 992 (987.0–992.5) | 1,520 (1502.7–1523.0) | -34.737% | 0.897009% | Fusion |
| 256 × 32 | 1,032 (1022.5–1035.5) | 1,460 (1455.0–1470.5) | -29.315% | 0.600962% | Fusion |
| 256 × 256 | 1,465 (1456.5–1476.5) | 2,478 (2459.2–2503.7) | -40.880% | 0.100276% | Fusion |
| 1024 × 1 | 1,372 (1368.2–1376.7) | 1,807 (1803.8–1817.3) | -24.073% | 0.277318% | Fusion |
| 1024 × 4 | 1,374 (1372.0–1385.3) | 1,815 (1806.7–1821.7) | -24.298% | 0.802324% | Fusion |
| 1024 × 32 | 1,769 (1758.7–1775.5) | 2,330 (2320.3–2335.2) | -24.077% | 0.139504% | Fusion |
| 1024 × 256 | 7,420 (7401.5–8017.0) | 8,603 (8588.2–10012.0) | -13.751% | 0.320635% | Fusion |
| 4096 × 1 | 1,928 (1919.0–1934.3) | 2,123 (2121.0–2127.2) | -9.185% | 0.023516% | Neutral |
| 4096 × 4 | 1,981 (1976.8–1984.5) | 2,321 (2319.3–2332.2) | -14.649% | 0.559927% | Fusion |
| 4096 × 32 | 2,657 (2654.3–2670.0) | 3,406 (3404.5–3421.7) | -21.991% | 0.080774% | Fusion |
| 4096 × 256 | 10,778 (10688.2–11880.7) | 22,565 (22497.2–26243.5) | -52.236% | 0.033346% | Fusion (v12) |

The neutral 4096×1 cell is 9.185% lower latency, below the strict >10% benefit
threshold; it retains the incumbent. The proposed `data.json` contains exactly
these twelve independently audited observations with report/sample provenance.
The original raw measurement rows retain their `UNREVIEWED` label unchanged.
The final entry carries its v12 sample provenance. Records apply only when the exact query,
implementation identities and selection environment still match; installing
these data must not make stale evidence current by rewriting identity fields.

## Memory and paid work

Every valid case reports **33,554,432 bytes of library workspace**,
**21,015,040 bytes of selected persistent backing**, and a **16,777,216-byte
launch pool**. The harness also allocates a separate **33,554,432-byte activation
arena per arm**. These fixed allocated capacities differ from the lowerer's
used activation extents below. The 64 MiB construction-funding allowance is a
metadata admission bound, not a measured allocation. No peak unified-memory,
GPU-only memory, allocation accounting or production-model footprint is inferred
from these values; CUDA contexts, graphs and other runtime costs are not summed here.

| Width × rows | Fused used activation bytes | Decomposed used activation bytes |
| --- | ---: | ---: |
| 256 × 1 | 3,072 | 4,096 |
| 256 × 4 | 9,216 | 13,312 |
| 256 × 32 | 66,560 | 99,328 |
| 256 × 256 | 525,312 | 787,456 |
| 1024 × 1 | 12,288 | 16,384 |
| 1024 × 4 | 36,864 | 53,248 |
| 1024 × 32 | 266,240 | 397,312 |
| 1024 × 256 | 2,101,248 | 3,149,824 |
| 4096 × 1 | 49,152 | 65,536 |
| 4096 × 4 | 147,456 | 212,992 |
| 4096 × 32 | 1,064,960 | 1,589,248 |
| 4096 × 256 | 8,404,992 | 12,599,296 |

For the final v12 cell, production logical placement was 8,404,992 bytes in
both arms; lowered used activation was 8,404,992 fused and 12,599,296 primitive.
Both lowered programs required zero kernel-pool bytes. These observed
requirements fit the original arena/pool capacities. v12 adds actual placement
and lowered-resource checks, not more memory or different arithmetic.

## Source and measurement identity

The first eleven valid cells used v11:

| Identity | SHA-256 or value |
| --- | --- |
| Source base | `dfba7a78cd6f6a821148d3cc434ab182153ece7c` |
| Explicit wave-J dependency patch | `2cfe12da558abb5c84fe2b3b4d1a0641c5edcebe739e3bd847545c093054e775` |
| Reviewed v11 U6 patch | `ed7439d4e71f83698e3723a1508a5ec4f09186336c36f713b258348dc19e1756` |
| 2,778-file source manifest | `b01ba89bdd4689caccca6c5aa5d20e12efbc375bd9bbd567f537082148b0c978` |
| Benchmark | `6e9bcdebd527fdf2ea4dd63fe685d2254a064725704a6b2a58430e33bc4c3b46` |
| Reviewed first-screen launcher | `667bdf81939430fe247bb8f04a79775339b67b4846ab1eefd7e911573d952cda` |
| Selection environment | `9441b5493c88e6b931ef626d4a1c837110bbf3b397bb9a08827cbe94c3ce4cac` |
| Fused logical/physical `ggml.rms_norm_mul.fused` | `579261470f978640ca8aceb5d5f9ada1c0413a76c06026ab117853c56038d293` |
| Logical child 1 `ggml.rms_norm` | `040869e79c3bfec00f78acd195ea57379d801bf08464bbcdba1b803fa8656929` |
| Logical child 2 `ggml.mul` | `27267dea97f92ac56524daa25eb82aeb798bf5f92ffcc3caa52cbb99bff31871` |
| Physical decomposition `ggml.rms_norm_mul.unfused` | `de917689ddfec8b292373c3d095aa8380d8e38a83e74388469e9510a07ff599b` |

The last cell used the separately preregistered v12 harness:

| v12 identity | SHA-256 or value |
| --- | --- |
| Registration | `5d6fea5528e21b04209cdefc1540a4b9220511b4` |
| Full U6 patch | `43373c288f15490f21e32325212f9a593775e2c09d1531324a718d3dbd94db8a` |
| Narrow v11-to-v12 capacity repair | `c67f5451dfbf786b2bda35f6379aa1f3db1cdcf80dca206c248ee751b99e5856` |
| 2,778-file manifest | `acfb562360004d70ba11d80e09a0cde9dc594aee6925520587c7d89276456b98` |
| Benchmark | `ecb80b29f4df1271feaeba9ba0f950b61c5e6c6837207385370cabcca74b610b` |
| Last-case launcher | `38a2b379c9d9f99428f4c3b5b3e720120cb480b41e598699e38d9f98b54e570a` |

The base and explicit dependency remain identical. Independent manifest
comparison found only `benchmarks/fusion_ranks.cc` and the frozen protocol README
changed; all other 2,776 entries are identical. The actual recorded selection
environment, fused and primitive implementation IDs, and physical recipes match
v11 exactly. The new query is independently recorded for width 4096, rows 256;
no prior query identity was substituted. The v12 wrapper differs only in its
registered identities, paths and final-case arguments. Sampling and lifecycle
logic remain unchanged.

The inherited remote `61f8426f.dirty` build stamp is not the source identity;
the explicit base, dependency, patch and manifest above identify it. All thirteen
attempts (twelve valid plus the retained v11 refusal) authenticated all 2,778
source files **before** execution, with no mismatches, and authenticated the benchmark **before and after** retirement.
The launcher does not rehash all source files after execution. Each v11 matrix
wrapper was byte-compared against the reviewed first-screen wrapper: only the
predetermined width/row command literals and fresh output path differ. No
lifecycle, arithmetic, comparison or sampling logic changed.

All twelve successful jobs returned zero, their wrappers recorded retired
children and their final harness events followed teardown; the original failed
largest-cell attempt also retired. Exclusive acquired intervals do not overlap. Supervisor
`gpu.json` records lock acquisition, not a post-job process inventory; that
file alone is not evidence of an empty GPU process list. A separately retained
`post-matrix-busy.txt` records zero running/waiting jobs and zero GPU compute
processes after the matrix and subsequent v12 validation/negative controls;
this is a post-batch observation, not a per-case process inventory. The final
v12 timing job additionally has its own `post-job-busy.txt` recording no running
or waiting jobs and zero GPU compute processes.

Raw samples, launch receipts, stderr, supervisor records and wrapper sources
remain outside Git. First-screen job `m35-u6-first-1024x1-v11-r1` is retained at
`/tmp/llmp-m35-orchestration/u6-first-screen-v11-r1`. Other jobs are named
`m35-u6-matrix-WIDTHxROWS-v11-r1`, with results and supervisor records under
`/tmp/llmp-m35-orchestration/u6-matrix-v11`. Their original remote outputs are
`spark-b:~/.local/share/llmp/m35-u6-fusion-ranks/{first,matrix}-WIDTHxROWS-v11-r1`.
The final job `m35-u6-matrix-4096x256-v12-r1` is retained at
`/tmp/llmp-m35-orchestration/u6-last-screen-v12-r1`, originally
`spark-b:~/.local/share/llmp/m35-u6-fusion-ranks/matrix-4096x256-v12-r1`.
The independent audit retains all original pairs, bookends, helper and receipt
hashes at `/tmp/llmp-u6-measurement-review/audit.json`.

| Cell | Raw samples.jsonl SHA-256 |
| --- | --- |
| 256 × 1 | `9a7b3d698363c09b079d3b856cf8e491b16630c7d74007c69d9bf55f9d5539b7` |
| 256 × 4 | `e6e5db22d85faaf13c1678b2efb7880940a58c93abf4d0c96113c3229b8566c0` |
| 256 × 32 | `43fa1e1d7fde98a919dec36ccd4888347aaf85eae58731a8d79d33f1b990c532` |
| 256 × 256 | `436b2e966351ce5edad2cb7bc0267e97810ecd0ec4f8261dc9a0ecd34791c6c8` |
| 1024 × 1 | `86986af6409b80a3cdc77a2c28d47679d2a6d50e023d6f3aba369c64446a3368` |
| 1024 × 4 | `18f8d1814e0db132ea175c480b2d801ba383b834dde930c442644dbc48d87f85` |
| 1024 × 32 | `93422a5ce2d9e021046ad29897505285b356353f95e18be83f67a63c240b9d05` |
| 1024 × 256 | `17c64aa046c3a618c793d3aab91dff4483d4e6d7cf14d03f1b60185e7fd5ca5a` |
| 4096 × 1 | `e8c27ba89d3df625b4f8fe11bd3f77c0a0976d65fd71b2cc585b11e706e13861` |
| 4096 × 4 | `bc400b96bfc1454b15422e3fb2016084e74ae1ee182caa7cd78dd66283134f81` |
| 4096 × 32 | `7df3f5b86874aea3ab2b01c988891e49f546985e0ae3c194f220aa24b093c1f6` |
| 4096 × 256 (v11 refusal) | `70294182454a0a8193f8008feee1a732b6aa8c2f936bc2575a9b6d821bce30f4` |
| 4096 × 256 (v12 valid) | `395b9e62663ab27dbb5cb0fa8560daa457b6d9839dab199fc0c307992f9069bf` |

## Retained failures and remaining work

- Original A v8 launch: supervisor exit 127 before the helper, from unresolved
  relative `mise`; no model or benchmark work and no samples.
- Corrected A v8 launch: zero-byte backend workspace declaration refused during
  setup; no correctness or timing result. The wrapper and supervisor retained failure.
- B v9 validation-only: missing fused rewrite provenance refused during lowering.
- B v10 validation-only: joined input was not an exact owned graph port.
  Neither validation produced timing; v11 repaired the production startup and
  provenance path and passed full validation before its registration.
- B v11 4096×256: the conservative argument guard multiplied fixture demand by
  16 and refused before setup work. The benchmark returned 2, the wrapper and
  supervisor returned 1, and only the refusal event was emitted. This is a
  harness-capacity failure, not measured arithmetic, speed or device-memory
  insufficiency. It remains a failed attempt alongside the later separately registered successful cell.

The largest case subsequently passed under the reviewed v12 resource checks
and prospective amendment, with unchanged fixture capacities, inputs and
numerical/performance bounds. All nine final-case pairs are retained, including
the slower observations (fused maximum 11,880.7 ns, primitive maximum 26,243.5 ns);
none were dropped. Its primitive bookends differed by 0.033346%.

Final integration must verify the recorded implementation/query/selection
identities remain valid, validate generated data, and run the affected native
checks. The strict generator accepts the twelve-record proposal. No production
rank data have been installed by this report. This complete first-class matrix
does not complete all U6 fusion coverage or establish a model-speed or
model-quality result.
