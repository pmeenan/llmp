<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 ordinary C2 first quality screen

The proposed ordinary serving recipe fails its first strict C2 comparison: eight of 66 complete heads have different winners with positive reference margins. The independent 64-target conditional loss gate passes at **+0.7910%**, below the preregistered 3% limit. **Production default adoption remains withheld.** Both engines repeat their own complete heads exactly; native initialized state copies and layouts are also exact. This comparison has no margin allowance and uses no C4/C8/C12 calibration.

| Gate or diagnostic | Actual result |
| --- | --- |
| Complete finite native own-repeat heads | 66/66 byte exact; 69,206,016 bytes |
| Native initialized state/layout/input own repeat | Four 922,746,880-byte snapshots, both layouts and full carrier exact |
| Complete finite reference own-repeat heads | 66/66 byte exact |
| Native/reference complete-head byte identity | 0/66 |
| Positive-margin winner differences | 8/66; all in paid rows, no frontier differences |
| Exact reference-tie differences | 0 |
| Positive reference margins | 0.0373859406 to 4.1159814596 |
| Maximum raw full-head difference | 12.8280391693 |
| Conditional mean NLL: native / reference | 10.5490013244 / 10.5411224465 |
| Relative conditional loss | +0.790999794%; PASS independently |
| Combined strict first-screen gate | FAIL |

[results.json](results.json) retains aggregate counts and the positive-margin range. All eight individual row/owner/winner/margin witnesses remain external in the authenticated raw comparison. Neither conditional loss nor own-repeat identity waives the strict failure. This screen does not identify the arithmetic cause.

## Fixed recipe and chronology

Both engines consume the first two independently qualified histories from the same 12×1024 carrier, 49,152 bytes, SHA `d584450079145f3d2c93f46ef24a0aabe2b3971279a1cbddbbb29f0a506ac5e3`. The checkpoint-specific carrier manifest is `f6cf32b29dd980c997915519339cdc1cd2c244e219c473a0331303bbc6e5708c`; native/public tokenizer IDs and native/Jinja rendered bytes were already qualified by the [input preparation](../gemma-input-preparation/README.md).

Native command: `ART OUT 31 2 joined norm CARRIER production`. It uses context4096, prefill capacity256 and head capacity2, plain norm/normRoPE/normADD, ordinary products, normal graphs/fusion and requested owner attention. Actual selected owner steps are **zero**, appropriate to C2; the recorded last-built plan has two rows/segments, 121 plain norms, 120 normRoPE and 120 normADD steps. Row-invariant, shared-Q8, RoPE-store and grouped cache writes are off. Counts describe selected plans, not launches per replay. Actual completion is 32 paid groups/64 token records/32 graph replays with a 73,515,008-byte helper publication allowance.

The original pinned public reference uses physical C2 in one `llama_decode`, total context8192, batch/ubatch256, F16 K/V, `swa_full=false`, `kv_unified=false` and normal graphs/fusion. Actual global/local caches are4096/1280 cells with two streams. The native own pair and full finite/state freeze completed before either reference output. The reference own pair then completed before numerical comparison. Each public call synchronized and retired, with checked owned-container absence.

Each history prefills IDs0..991, publishes frontier row0, then consumes supplied IDs992..1023 in32 forced steps. Paid token step `s` validates retained row `s+1`; all66 full heads are compared. Only rows0..31 supply the64 full-vocabulary FP64 NLL targets at positions992..1023. Final row32 predicts position1024 and has no supplied likelihood target. This conditional score is not a1023-transition corpus PPL result. Exact reference ties are reported separately; positive margins receive **zero** allowance.

## Timing and completed controls

Paid native times were3.30978/3.29089 seconds; reference times3.29867/3.29040. Means3.300335/3.294535 give native **+0.17605%**, with spreads18.89/8.27ms. These are short fixed-cohort observations and establish no sustained parity or performance adoption.

The source candidate passes34 targeted controls with zero skips: settings5, approved binding1, production row/ring/features4, scalar/default serving12, ordinary joined8 and invariant continuity4. Ordinary own-repeat/state/restore/cancellation checks cover C1/C2/C4/C8/C12; they do not establish cross-engine arithmetic quality. The initial focused compile attempt failed before tests on a fixture-local shadowed variable; the corrected retry passed all12 supervised steps. Full suites and HTTP acquisition were not run.

The strict comparison is preserved as an official **FAILED** job: its numerical step wrote the complete result and returned1; stop-on-fail skipped the final authentication step. A separate authorized one-step identity-only job completed that exact guard. It did not rerun inference or numerical scoring. All eight owned-container retirements and official statuses/counts are recorded in the aggregate; raw logs, vectors and initialized states remain outside Git.

## Source and limits

This is the isolated ordinary recipe candidate based on774515e, with exact14-file source receipt `524fe76c8868d20916c3e68aaa19e2e11ab958696be2601a31bff4eb23d188c2`. Native helper `5ac9f0ed…`, SDK receipt `eb7a4ebc…`, public client `12b36bed…` and native pre-oracle freeze `2dba40d5…` are fully identified in the aggregate. Native NVCC13.4.92/toolkit13.4.2 and original digest-pinned reference image837fc732 are retained. These candidate runtime defaults are not an adopted main-tree recipe.

The task-entry primary TensorFold pin is cb2ebf0540f42604e2759b2ddef497861e928248/version0.6.6. Its current Gemma recipe uses MLX and the26 profile; it supplies no comparable CUDA/31 target, so no TensorFold inference was run. The entry receipt and exact recipe fetch are bound in the aggregate.

C2/partial-cohort quality, full-corpus, sustained/depth/swap/shared-prefix and actual HTTP gates remain open. Existing whole-C8/C12 evidence and historical failures retain their own scopes and bounds. This first C2 failure changes none of them.
