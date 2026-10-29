<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fusion versus its own decomposition: coarse measured rows

A backend's measured fusion preferences are coarse rows
(`execution::FusionRank`, [rank_data.h](../../../src/execution/rank_data.h)),
consulted at selection (`execution::FusionPreferenceFor`, a 5% band):

- **Key.** The device class, the fused operation, the implementation and
  its kernel version (`Implementation::version`), operand 0's encoding and
  layout, and inclusive bands of the selection query's rows and width (its
  `k`: a row's elements). Nothing else: no build identity, source closure,
  graph, layer, weight name, address or policy digest, so a row survives
  unrelated code changes and stops deciding when its kernel changes.
- **Decision.** Within 5% of the decomposition keeps the incumbent; a fusion
  more than 5% slower is not a candidate, so the planner decomposes the node;
  one more than 5% faster qualifies a fusion that needs measurement. (The
  band was D-085's coarse 10% until 2026-10-28; 5% is beyond the repeat
  noise of the vecq column sweep, [vecq-columns](../vecq-columns/README.md),
  so a measured gap above it chooses the faster.) A row
  never grants numerical eligibility and is never extrapolated past its
  bands.
- **Staleness.** A row measured at another kernel version decides nothing.
  `FusionRanksTest.NoShippedRowNamesAMissingOrStaleImplementation` fails on
  any shipped row whose implementation is missing, of another operation or
  of another version. A kernel author who changes what an implementation
  launches in a way that could move its speed bumps its version (GGML:
  `KernelVersion` in `src/kernels/ggml/capabilities.cc`) and re-measures or
  retires its rows. Rows are refreshed at milestone boundaries, as
  TensorFold's pin is.
- **Data.** The GGML backend's rows are C++ data in
  [measured_ranks.h](../../../src/kernels/ggml/measured_ranks.h), beside its
  device ranks.

## Harness

`llmp_fusion_ranks` ([fusion_ranks.cc](../../../benchmarks/fusion_ranks.cc))
builds one case's fused node, expands it to its own decomposition and plans
that graph twice through the GGML backend's ordinary planner: as the policy
stands (the pattern rewrites it back into the fusion) and with
`disable <pattern>`. It lowers and binds both normally, runs each once and
compares the outputs (relative RMS at most 1e-5), captures each as a graph
of repeated runs and times them alternately with CUDA events: three warmup
pairs, then nine pairs, medians reported. It prints a JSON line and a
`FusionRank` initializer at the measured point; the reviewer chooses its
bands from the cases measured. Cases: `rms_norm_mul WIDTH ROWS`,
`spatial_channel_norm W H C [silu]`. Run it as an exclusive Spark job.

## Measurements

### Harness control: RMSNormMul (2026-10-26)

`ggml.rms_norm_mul.fused` against `ggml.rms_norm` then `ggml.mul`, outputs
bit-identical: width 1024, one row, 1,399 against 1,831 ns; width 4096, 256
rows, 11,652 against 20,724 ns (fusion both). The retired exact-query rows
measured 1,388/1,817 and 10,840/20,937 ns at the same shapes. The fusion
needs no row: it is already the incumbent.

### Spatial channel norm at Qwen-Image's VAE shapes (2026-10-26)

U6 found the decomposed `spatial_channel_norm` 1.3 to 4.7 times faster at
widths 144 to 1025 (rows 111). Its fixture was a [37, 3, C] tensor: 37
columns, 3 rows and C channels. `spark` (GB10, driver API 13000, runtime
13040), `llmp_fusion_ranks`, nine pairs each, medians, F32, `eps` 1e-12;
the fusion is `llmp.image.spatial_channel_norm` (kernel version 1), the
decomposition GGML's five or six launches. Outputs agree within relative
RMS 3e-7.

| Case [W, H, C] | Query rows / k | Fused ns | Decomposed ns | Decision |
| --- | ---: | ---: | ---: | --- |
| [37, 3, 144] (U6 fixture) | 432 / 37 | 4,551 | 4,836 | neutral |
| [37, 3, 1025] (U6 fixture) | 3,075 / 37 | 45,070 | 9,918 | decomposition, 4.5x |
| [128, 128, 384] | 49,152 / 128 | 241,702 | 1,092,164 | fusion, 4.5x |
| [128, 128, 384] + SiLU | 49,152 / 128 | 244,596 | 1,292,348 | fusion, 5.3x |
| [256, 256, 384] + SiLU | 98,304 / 256 | 971,968 | 5,447,424 | fusion, 5.6x |
| [512, 512, 192] + SiLU | 98,304 / 512 | 1,927,616 | 10,852,744 | fusion, 5.6x |
| [1024, 1024, 96] + SiLU | 98,304 / 1,024 | 3,815,768 | 21,623,159 | fusion, 5.7x |
| [1024, 1024, 96] | 98,304 / 1,024 | 3,824,784 | 18,129,305 | fusion, 4.7x |

The fused kernel loses only on a narrow row of many channels, which the
VAE never has; at the VAE decoder's shapes for a 1024x1024 image (the
default `image_size`) it is 4.5 to 5.7 times faster. The whole image
agrees: Qwen-Image-2.1 (composition `62497918`), 1024x1024, 40 steps, the
pinned latents and prompt of `tools/speed_gate.py`, one process each,
generation time after the model is ready:

| Arm | Generation s | Pixels SHA-256 |
| --- | --- | --- |
| Shipped policy (fusion) | 33.17, 33.40, 33.50 | `93356574…` (frozen) |
| `disable spatial_channel_norm` | 33.73, 33.97 | `2e5432ec…` |

Decomposing is 1.5% slower and changes the pixels, so no row is adopted:
the shipped table has no fusion rows. U6's two other alternative wins are
not re-expressed: its split-modulate cell was width 8193, which no served
model uses, and its two rope-store cells store into Q8_0 and Q4_0 caches,
a per-alias KV mode that is off by default, at a two-row fixture; measuring
them at a served cache geometry needs a state-binding case in the harness
(a follow-up).

## Retired: U6's exact-site recipe rows (2026-10-26)

U6 (commit `24849831`) installed 113 exact-site recipe rows and kept 12
legacy exact-query fusion rows, keyed on whole-graph and region digests,
fixture weight names, layers, offsets and the build's source-closure
identity, and ran an exact-site comparison (a second discovery plan, then a
replan and full relowering per fused site) on every GGML plan miss to look
them up. No production site could match a row: every subject was a synthetic
fixture. Commit `7c96e27b` then re-stamped all 99 new rows and the 12 legacy
rows onto a new build without re-measuring them (new build, recipe and
selection identities with byte-identical timings, including seven
`whole.image.block.bf16.width256.rows17` rows whose planned region had
changed), contradicting this report's former "never repinned" rule. Those
rows, the comparison path, the compiler-identity launcher
(`tools/compiler_identity.py`, `cmake/LlmpSourceClosure.cmake`) and
`data.json` are retired; production planning plans and lowers once per step
shape. The U6 harness found nine alternative wins among 97 latency
decisions (six spatial-norm cells, one split-modulate cell, two quantized
rope-store cells), all on synthetic fixture shapes; the measurements above
re-check the ones that could reach a served model.

The reports below are frozen history. Their identities, receipts, data
files and helper scripts describe the retired mechanism and are not
re-runnable against the current tree:
[U6 final results](U6-FINAL-RESULTS.md) and
[registration](U6-FINAL-MATRIX-REGISTRATION.md),
[shared-support pass](U6-SUPPORT-CLOSURE-RESULTS.md),
[production context](U6-PRODUCTION-CONTEXT-RESULTS.md),
[production regression registration](U6-PRODUCTION-REGRESSION-REGISTRATION.md),
[ready matrix](READY-MATRIX-RESULTS.md),
[first screens](READY-FIRST-SCREEN-RESULTS.md),
[NEOX](READY-NEOX-FIRST-SCREEN-RESULTS.md) and
[captured-event NEOX](READY-NEOX-CAPTURED-EVENTS-RESULTS.md),
[NormGate captured events](READY-NORMGATE-CAPTURED-EVENTS-RESULTS.md) and
[median bookends](READY-NORMGATE-MEDIAN-BOOKENDS-RESULTS.md),
[actual GDN first screen](ACTUAL-GDN-FIRST-SCREEN-RESULTS.md),
[W5 rerun](FRESH-W5-RESULTS.md), [original RMSNormMul results](RESULTS.md)
with [historical data](historical-v11-v12.json), and the
[source identity](SOURCE-IDENTITY.md) scheme.
