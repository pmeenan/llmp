<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 short-rung policy and phase controls — 2026-10-01

No single head/depth choice closes the strict 32K/64K speed gate. All four
32K combinations remain below the faster fresh Mia control. At 64K the
selected head with the existing adaptive policy is best. Keep production
defaults unchanged; this diagnostic supplies no new quality exception.

Sixteen fresh processes compare prefix 65,536 and externally supplied Mia
selected 47,172 heads at 32K and 64K, with adaptive depths 2–3 and fixed
depth 3. Every head/rung runs adaptive, fixed, fixed, adaptive. Each child
uses the unchanged qualified timing benchmark with one repeat and 512 output
IDs; its decode rate counts 511 outputs after the initial argmax. Stop IDs
are ordinary outputs for this fixed-budget control. These are unprofiled
benchmark rates, separate from HTTP runtime rates and completed-answer tests.

## Completed measurements

All sixteen children terminate successfully. Each within-policy pair has
byte-identical 512-token arrays and identical complete depth/verify-row/kept
traces. Whole Draft includes owed recurrent/MTP commit; Verify includes its
target graph and output read. The benchmark has no head-only timer.

| Input / head / policy | Two observations, tok/s | Median tok/s | Draft / Verify / whole loop, ms per step | Accepted / drafted | Verifies |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32K / prefix / adaptive | 39.802 / 39.432 | 39.617 | 7.267 / 50.290 / 57.584 | 287 / 526 | 224 |
| 32K / prefix / fixed 3 | 39.027 / 38.765 | 38.896 | 8.648 / 53.008 / 61.680 | 298 / 638 | 213 |
| 32K / selected / adaptive | 37.126 / 37.142 | 37.134 | 6.709 / 50.602 / 57.338 | 271 / 577 | 240 |
| 32K / selected / fixed 3 | 39.868 / 39.562 | 39.715 | 7.711 / 53.240 / 60.980 | 300 / 632 | 211 |
| 64K / prefix / adaptive | 38.123 / 38.096 | 38.110 | 7.724 / 52.104 / 59.861 | 287 / 537 | 224 |
| 64K / prefix / fixed 3 | 36.626 / 36.496 | 36.561 | 8.837 / 54.089 / 62.959 | 289 / 666 | 222 |
| 64K / selected / adaptive | 42.026 / 41.819 | 41.923 | 7.610 / 53.903 / 61.562 | 313 / 531 | 198 |
| 64K / selected / fixed 3 | 37.774 / 37.803 | 37.789 | 7.960 / 54.320 / 62.316 | 294 / 651 | 217 |

The strict descriptive comparison uses the faster fresh Mia observations:
42.234 tok/s at 32K and 39.702 at 64K. Its slower observations, 38.622 and
39.645, remain part of the [reference evidence](../qwen38-mtp-speed/README.md).
The best 32K native median is about 6% below the faster reference; only
64K selected/adaptive exceeds its strict rung in both observations. This
does not approve a context-dependent head selector or replace the final
runtime gate with a coarse ten-percent criterion.

Fixed 3 helps the selected 32K trajectory but loses at 64K. Verify consumes
approximately 86–88% of completed decode time, so the observed cost is chiefly
target verification. Different policies change proposal histories and step
boundaries; the phase table cannot attribute that cost to a head or product.

## Trajectory and reference limits

With fixed 3, prefix and selected outputs agree for all 512 IDs at both
depths, although acceptance and verify grouping differ. Adaptive versus
fixed first diverges at output index 13 at 32K and 19 at 64K. Adaptive
prefix versus selected first diverges at indices 110 and 137. The analysis
retains all twelve pairwise common-prefix/first-divergence records.

Native and Mia share exact canonical prompt IDs and authenticated weight
representations, but generated histories, state/cache precision and reduction
boundaries differ. Mia uses fixed 3, FP8 main KV, BF16 recurrent/conv state
and head input/output; native keeps F16 KV and F32 recurrent/conv state,
small-product inputs and outputs. Reference timing uses HTTP, DET0, its own
request loop and full-decode graph policy. No causal cache-quality loss,
same-history product speedup or new verifier-quality result is claimed.

The source audit also corrects an earlier HC attribution: the installed
`enable_qwen38next_low_latency_gemm` requires SM103. GB10 SM121 therefore
keeps standard BF16 vLLM linear projections, with fused Triton HC glue.
Its HC combine rounds the residual to BF16 before RMS; native preserves
F32 residual/mixed values. A Marlin backend name alone does not establish
activation bit depth: the installed consumer selects optional INT8/FP8
activation quantization through `VLLM_MARLIN_INPUT_DTYPE`; otherwise it
accepts BF16 input. These are source distinctions, not measured causes of
the speed gap.

## Configuration, provenance and retirement

The native ceiling is 262,144, chunk 4,096, graphs and runtime prefill on,
window 0, no profiler, depth 3 maximum and the unchanged adaptive policy
with relative cost 1.16. Every child loads and prefills independently.

| Frozen input | Prompt IDs | Little-endian I32 SHA-256 |
| --- | ---: | --- |
| 32K | 31,743 | `306136a7dc2a5f66c94cbf4ac874e9011c0af79b79072731407a144bfba1fbd5` |
| 64K | 64,110 | `143815331a501e67b4731cdcfe6c1a4b5c901e204e90ed67a38bede89949376e` |

The preserved Spark A executable is
`ac9988d0738ba5759ce6aa0e4f77c2fb84ebb0f3ba9f3f178bad6f9d0f7bc9c3`,
bound by supplement `f5398cfe…` to source map `f2047cb6…`, full
qualification `90d52cf7…`, actual compile database/native SDK receipt and
the archived cuBLAS payloads. No native source or arithmetic changes were
made for this batch. The selected artifact was provisioned before timing;
source-before/source-after/destination inventories and the complete shard
agree. The failed first preparation, which incorrectly expected a
prefix-owned head, remains excluded and preserved.

Job `qwen-policy-phase-timing-r2`, supervisor 1660136, runs on Spark A
**05:09:30–05:21:52 EDT** on 2026-10-01 (09:09:30–09:21:52 UTC), rc0.
The 12m22s actual wall time includes all sixteen loads/prefills, source
proofs, gates and retirement. Every child inherits the supervised job group,
has a bounded terminal wait and passes the strong 105 GiB/no-other-model
gate before load and after reap. The authenticated final retirement log reports
117.209 GiB available with GPU/container/native-model probes clear; the
supervisor records the job complete with rc0. Spark A was explicitly released.

Raw receipts, complete specs/traces and the excluded preparation remain
outside Git at `spark:~/scratch/m3-qwen-policy-phase-r2/` and the preserved
qualification/supplement roots. Local compact evidence is in
`/tmp/jitllm-qwen-policy-phase-timing-r2/` and
`/tmp/jitllm-qwen-policy-phase-analysis-r2.json`.

| Evidence | SHA-256 |
| --- | --- |
| Immutable controller | `bb4d403700344db2fea8a0c26cf489d72c68c8872c898043afbb03b5b63ebe13` |
| Prepared receipt | `515b2eb24664dc29cb867daf39e8708ab7de46a91905314397ccf925fd69ca5f` |
| Complete timing receipt | `e4ab63a9034c2a52f59d7d76d0b2f0ff679cd57284074446a75bd8e236ab7f09` |
| Offline analyzer | `686baff8654e470216517485c3c90465662606d92b7826dc533a624c4411b8a4` |
| Completed analysis | `b4b559aaf9ef5c647a890e728906c8db8b577c3e33f699421da4e6852089efa7` |

The completed analysis authenticates controller, prepared and timing receipts
and every raw spec, checks terminal records in the authenticated timing receipt,
reconstructs acceptance from actual traces, checks independent repeats and
reports descriptive divergences. The timing controller separately authenticates
the native source and static closure. Analysis runs only after all model children
retire. This slice retains diagnostic evidence and changes no production
code, runtime configuration or quality bound.
