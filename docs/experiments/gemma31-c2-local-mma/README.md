<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 C2 private local-MMA factor

Replacing the one-query D256 vector attention selection with the existing MMA/GQA2 implementation does **not** close the strict C2 quality failure. Nine of 66 complete heads have different winners with positive reference margins, versus eight in the [original screen](../gemma31-production-c2/README.md). The independent 64-target conditional loss improves by **8.19189%** relative to the retained reference. Production adoption remains withheld; the private selector is not a shipping policy.

| Gate or diagnostic | Actual result |
| --- | --- |
| Native complete finite own-repeat heads | 66/66 byte exact; 69,206,016 bytes |
| Native initialized states/layouts/input own repeat | Four 922,746,880-byte snapshots, both layouts and full carrier exact |
| Completed prefill states/layout/input versus original native | Exact |
| Reference complete finite own repeat | 66/66 byte exact, retained FIRST pair |
| Native/reference complete-head byte identity | 0/66 |
| Positive-margin differences / exact ties | 9 / 0; all differences in paid rows |
| Positive margin range | 0.0213127136 to 9.6777807474 |
| Maximum raw full-head difference | 13.1165294647 |
| Conditional mean NLL: native / reference | 10.4556528969 / 10.5411224465 |
| Relative conditional loss | −8.191890141%; PASS independently |
| Combined strict first-screen gate | FAIL |

The factor changes only the private executor's D256, one-query vector-selection gate. Existing graph dispatch then selects MMA/GQA2. The first decode plan witnesses **zero local vector, 100 local MMA and 20 global MMA steps** for the two owners. D512 selection, products, norms, cache stores, inputs and kernel bodies are unchanged. The dimension-scoped override is broader than Gemma and is unsuitable for adoption; any future family policy needs its own eligibility and controls.

The bounded recipe and zero allowance are unchanged: first two histories from the qualified 12×1024 carrier, context4096, prefill capacity256, head capacity2, ordinary joined norm policies, owner requested but zero owner-adapter steps, normal graphs/fusion, 32 forced steps and the unchanged 73,515,008-byte publication allowance. All 66 full heads are compared. Only retained rows 0..31 score the 64 supplied targets at positions 992..1023 using full-vocabulary FP64 NLL and the existing 3% relative-loss limit. Final row 32 predicts position 1024 and is unscored. This is not full-corpus PPL and borrows no C4/C8/C12 allowance.

The candidate native pair and full finite/state own freeze completed before comparison. Completed prefill state/layout/input identities match the original candidate. The oracle is exclusively the retained FIRST public C2 pair; no reference, scalar or calibration outputs were reacquired. Individual row/winner/margin records remain external in the authenticated comparison; [results.json](results.json) stores aggregates and provenance.

Paid candidate times were 3.30613/3.35547 seconds, mean 3.3308 and spread 49.34ms. The retained stock mean 3.294535 gives +1.10076%; the original candidate mean was 3.300335. These noncontemporaneous short pairs establish no isolated or sustained performance improvement.

The narrow build passed seven supervised steps and one no-launch planning control with four D256/D512, one/four-row witnesses and zero skips. Existing MMA scratch remains funded. Native acquisition passed six steps, own freeze passed three, and all four owned analysis containers have checked absence. The numerical comparison remains an official **FAILED rc1** job: its result completed, and stop-on-fail skipped the final guard. A separate one-step identity-only job passed that exact guard without rescoring or inference. Full suites, HTTP and further model screens were not run.

The private 18-file source is based on 774515e, preserves 13 original candidate files exactly and leaves the original 14-file source/helper/failure immutable. Helper b79567df… and SDK eb7a4ebc… are fully identified in the aggregate. The primary TensorFold entry remains cb2ebf0540f42604e2759b2ddef497861e928248/version0.6.6; its Gemma recipe is MLX/26, with no comparable CUDA/31 target. No TensorFold inference ran. This family-only factor leaves physical-stream partition geometry as an untested source lead, not a demonstrated cause.
