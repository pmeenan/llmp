<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Joined Gemma transfer audit

This runtime diagnostic reuses existing runner contracts. It adds no kernels,
generic fusion enablement, scheduler or state representation. The
[optimization inventory](../../optimization-inventory.md) remains the source
for selected Qwen/DeepSeek mechanisms and their shape/format restrictions.

| Existing mechanism | Gemma eligibility and actual choice | Remaining evidence |
| --- | --- | --- |
| Independent completed-unit cohorts and joined products | Shared runtime prepares each private state, groups existing runner Wave units and publishes complete owned heads. C1 remains scalar. C12 is ordered8+4. | Matched fixed-prefix reference quality fails; full optimized serving selection remains open. |
| One-row invariant float/quant reductions | Explicit row-invariant option is bounded to8 tokens, at most64 top-eight pairs per shared group. Same-policy solo/joined complete heads/state repeat exactly at C1/2/4/8/12. Wider prefills retain ordinary products. | Synthetic ordinary math changes; natural reference math fails. Checked norms at new joined shapes need a separate candidate. |
| Private cache/state, stable captured plans and completion-aware lifetimes | Existing Gemma segmented graph, independent attention/cache, device masks, state footprints and stable addresses are unchanged. Actual checkpoint/spill/restart and departed/cancelled controls exercise them. | Full context/memory/swap qualification remains separate. |
| Full output versus lean greedy publication | Existing full-vocabulary output is retained for every owner; native128-row pinned capacity and per-owner float/sampler grants remain funded. Paid copies and decisions are inside the timer. | No head suppression, argmax-only kernel or reduced memory claim. |
| Bounded token histories and settings | Existing owner history grants, fixed2V candidate scratch, Close release, private continuations and uncalibrated128/12 caps are unchanged. Internal options are absent from public configuration. | No new calibrated limits or production default. |
| Norm/RoPE, norm/residual, RoPE/store and shared-Q8 policies | Existing checked policies are available; this rows-only joined screen selects none. Generic fusion is off. | Dense31's earlier128-row norm identity does not qualify C4/C12 or new prefill shape. |
| Original Gemma routing and scaled reduction | The integrated common base has default-off checked graph dispatch and primitive fallback. This diagnostic does not select either policy. | Representative compound26 selection already fails; no inference from joined equality. |
| Qwen NVFP4/CUTLASS grouped expert chain and retained prepared inputs | These consumers require their native quantization/operand contract. Approved Gemma artifacts are heterogeneous GGUF blocks; no such writer/consumer is selected. | Format-specific reuse requires independent checked operands and measured model gain. |
| Qwen GDN, DeepSeek hyperconnections/Markov drafts, MTP/assistants | These operations or state bindings are absent from this Gemma serving contract. Assistants/speculation are refused. | A separate bound feature contract is required, not a flag transfer. |
| Lean verify, cache-store elision, padded token-lane optimizations | No verify program or new lane is introduced. Full intermediates, raw K-as-V and stores remain owned by the existing graph. | Future consumers need their own keep/readability, capture, numerical and paid evidence. |

Joining speedups over scalar work establish a mechanism benefit at the measured
fixed inputs. They do not establish reference quality or competitive C12
performance, and do not select an optimization default.
