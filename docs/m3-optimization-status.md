<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3 optimization status

Snapshot: 2026-10-02. **Neither remaining engine comparison
gap is closed.** Pipeline reproduction, a useful individual replacement,
and a selected serving default are different outcomes. This page tracks
closure; individual reports retain conditions and evidence.

## Remaining gaps

| Question | Established result | Still needed |
| --- | --- | --- |
| Can jitLLM reproduce ds4's complete pipeline? | Yes. Full logits match byte for byte at 8K and 32K; throughput trails by 2.4% and 0.71%. | No repeat of this prerequisite. |
| Does the normal DeepSeek architecture match ds4 prefill? | No. Reduction and Q-head fusions are landed. Output-A gains 4.97% at 4096 rows; adding HCA takes 8K prefill to 9.8792 s (+21.19% incremental throughput). Genuine runner controls preserve the passing 4K candidate's complete heads, initialized state and ordinary continuation. IQ2 J64 with occupancy two adds a separate 4.15% whole-prefill gain, with all six heads byte-exact. The private 2048-row extension gains 3.95% but fails one fixed-history quality row, exceeding the unchanged bound by 1.669249 nats; it is stopped. | The literal inverse preparation control attributes only 166.9 ms / 2.21% to fused input preparation, with exact heads. Restore/attribute the remaining stages using the checked native integration. The reproduced literal pipeline is around 7.5 s at 8K with different cache/math contracts and output cadence, so these clocks are not a matched final speed ratio. |
| Is Qwen's solo MTP deficit explained? | Partly. Fixed-depth step times are close; acceptance and reference variation carry much of the reported rate difference. Curated vocabulary does not win at every depth. | Same-history paired acceptance, plus a decision on the actual remaining step-time slope. Existing generated-history runs do not establish acceptance parity or justify changing the gate. |
| Does concurrent serving match the other engines? | Fresh matched 8K C1/C2/C4 controls are complete. Against current same-checkpoint TensorFold NVFP4, native is +19.22% / +8.63% / −14.55% on completed-token rate including prefill and queueing. Native also trails the older affine TensorFold and original Mia at C4. HC/ragged-head sharing gains 2.76% on paid C2 wall; cap-four adaptive loses 13.16%. Fixed depth one alone loses 1.32%; four-head sharing adds 5.80% against that control with identical public replies. Conditional depth plus four-head sharing gains 7.05% against exact normal C4, but first completion is 83.23% later, median latency 15.70% higher and replies change. Capacity stays two. | Four-head controls pass 13 complete logits and 20 initialized-state/cursor comparisons, including discard/retry/continuation. Qualify the depth policy separately, including the extra 3.27 GiB fixed budget and latency tradeoff. Depth zero and DeepSeek independent-state batching remain open. Matching checkpoint/prompt IDs does not establish equal arithmetic or quality. |

Reports: [literal 8K](experiments/ds4-complete-plan/README.md),
[literal 32K](experiments/ds4-matched-32k/README.md),
[output-A](experiments/ds4-output-prefix/README.md),
[IQ2 occupancy two](experiments/ds4-iq2-occ2/README.md),
[attention preparation](experiments/ds4-attention-preparation/README.md),
[Qwen sharing](experiments/qwen38-combined-sharing/README.md),
[capacity rejection](experiments/qwen38-capacity/README.md),
[depth-one row budget](experiments/qwen38-row-budget/README.md),
[four-request head sharing](experiments/qwen38-four-head-sharing/README.md),
[conditional four-head serving](experiments/qwen38-conditional-heads/README.md),
[four-head recovery](experiments/qwen38-four-head-recovery/README.md),
[concurrent engine comparison](experiments/serving-concurrent/README.md).
The selected native output-A screen's actual receipt is retained externally
under `m3-ds4-qhead-short-records/outa-native-r2`. The selected guarded native
source passes its Spark check set; it is not a serving default.

## Independent review recommendations

The analysis-only review used main `7c700ba` on 2026-10-01. Its ranked list
is accounted for below; a related experiment is not marked as completion
of a different proposed experiment.

| Recommendation | Status and decision |
| --- | --- |
| 1. Same-history Qwen draft acceptance | **Open.** Existing fixed-depth and generated-history reports expose the noise, but the proposed paired anchor collection across both engines has not run. Stop further solo verify-product ports without a specific measured deficit. |
| 2. Attribute DeepSeek prefill by chain | **Done at the representative 4096-row geometry and refreshed after the selected factors.** The unchanged native graph selected six-slot reduction (+4.30%), Q-head fusion (+3.17%), output-A (+4.97% native screen), and HC-post/RMS (+1.94% private screen). These independent gains are not added together. The current output-A/HCA combination takes 9.763 s; its pair/down ordered intervals total 3.420 s (35.03% of wall), selecting that consumer group next. Native 2048-row chain attribution remains open. |
| 3. Measure the oracle's own noise at failing HCA rows | **Not run.** Bounds remain unchanged. A separate output-A plus HCA interaction now passes both 32K/128K fixed-history bounds, 128K held-out perplexity and one positive long-answer control. This resolves those candidate failures without a bound change, but does not measure oracle noise or qualify a serving default. |
| 4. Concurrency-aware row budget and both-model batching | **Partial.** Qwen's shared serving path and head/HC factors are landed. Cap-four adaptive is rejected. Fixed depth one alone loses 1.32%; sharing four full heads adds 5.80% with identical public replies. Conditional budgeting and four-head sharing then gain 7.05% against exact normal serving, but delay first completion by 83.23%, raise median latency 15.70%, fund 3.27 GiB more fixed memory and change replies. Four-head controls pass 13 complete heads and 20 initialized-state/cursor comparisons. Policy and sampling remain unqualified; no default changes. Depth zero and DeepSeek C2/C4 execution with independent request state remain open. |
| 5. J64 with occupancy two | **Done: measured and scoped native source checked.** Captured real gate/up products gain 28.05%; genuine native 8K prefill gains 4.15%, with six complete heads byte-exact. The final O3 object has 128 registers/16 stack bytes versus ordinary J128's 254/0; actual hardware occupancy was not measured. Production dispatch is restricted to the measured GB10 paired shape. |
| 6. Certified head screen/rescore | **Not run.** First count eligible candidate sets on captured real head inputs. No new kernel or model matrix is warranted before that count supports it. |

The review's additional 256K fixed-depth measurements, same-history controls,
all-active concurrent decode windows and wider-column GEMM comparison remain
separate open measurements. Its hypotheses and arithmetic estimates are not
recorded as local results.

## ds4 restoration coverage

| Stage or mechanism | Outcome |
| --- | --- |
| Complete original pipeline and deep selector | Reproduced; 8K/32K full-head equality and speed controls complete. |
| Output-B native consumer on original operands | Restored privately; full outputs exact and rate ratio 0.9983. It does not explain the gap. |
| Query-B native consumer on original D4 | One paid original/native/original screen is 3.11% slower, with 0.43% bookend movement. Original heads remain golden; native repeats exactly with changed logits and the same final argmax. Park this consumer factor; Q/KV producers remain open. |
| Routed FFN direct versus materialized chain | Paid materialization is 2.35% slower; keep direct in the reference. |
| Native IQ2 consumers, input preparation, J64 and shared worklist | Isolated. Preparation bytes match; the earlier unchanged J64 and shared-worklist controls did not justify adoption. The separate occupancy-two candidate gains 28.05% on captured pairs and 4.15% on whole native prefill with byte-exact complete outputs; scoped Spark unit/style/boundary, compiled golden/state and REUSE/header controls pass. |
| Route weighting and six-slot summation | Exact native ordered fusion landed; +4.30% whole-prefill screen. |
| Q-head normalization and rotation | Exact native fusion landed; +3.17% whole-prefill screen. |
| Output-A projection and inverse rotation | Positive private and native screens. Selected native code passes its Spark check set; normal output-B remains intact. It remains a default-off benchmark option at the qualified 4096-row shape. The private 2048-row extension gains 3.95% with HCA on but fails one 32K fixed-history row and is stopped. |
| Wide HC input and HC suffix | Screened and rejected: −2.18% and +0.48%; no expanded quality or context matrix. |
| HC-post followed by flat RMS | Positive exact private screen, +1.94%; native overlay prepared and reviewed, not yet run. |
| HCA attention with output-A | Focused 4096-row quality controls pass at unchanged bounds. A paid native 4096-row interaction gains 21.19% incremental throughput with 0.86% bookend movement. Genuine runner controls preserve full heads, initialized state and ordinary continuation, authenticate chunk positions before dispatch and fund all 32 128K plan shapes. The default-off integration passes Spark unit/style/boundary, final compiled full-head/state and REUSE/header controls. The 2048-row extension fails one unchanged quality bound and is stopped. |
| Remaining Q/KV, compression/indexer, shared-FFN and HC producer chains | Not fully restored or bisected. Their current native attribution is recorded; the piecewise replacement work is incomplete. |

Detailed evidence is in the [optimization inventory](optimization-inventory.md)
and its linked reports, including the
[native chain attribution](experiments/ds4-production-prefill-attribution/README.md).
Transfers retain each consumer's layout, precision, rounding and state
contracts; shared Qwen weights do not by themselves implement DeepSeek batching.

## Work order and M3 exit

Use both Sparks: isolate DeepSeek normalization output preparation/reuse
on one while testing Qwen conditional
row budgeting and broader head sharing on the
other. The fresh concurrent engine comparison is complete. Use one representative
bookended screen to settle each candidate. Measure
same-history acceptance before doing more solo Qwen kernel ports. After the
selected DeepSeek combination, refresh only its missing stage budget and
replace the largest remaining chain. Full ladders are reserved for an
unresolved context decision or qualification of a selected implementation.

Growing state, turn reuse, maximum-context execution and continuing-context
swap controls are complete. M3 remains open for the performance/quality and
concurrency items above, the final swap/client gate and frozen record, and
the deferred workstation/package checks. No package ships before the owed
checks. There is no claim here that all review suggestions, native pipeline
restoration, or the M3 gate are complete.
