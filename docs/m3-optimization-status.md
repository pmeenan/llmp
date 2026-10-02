<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3 optimization status

Snapshot: 2026-10-02, after `d20af4d`. **Neither remaining engine comparison
gap is closed.** Pipeline reproduction, a useful individual replacement,
and a selected serving default are different outcomes. This page tracks
closure; individual reports retain conditions and evidence.

## Remaining gaps

| Question | Established result | Still needed |
| --- | --- | --- |
| Can jitLLM reproduce ds4's complete pipeline? | Yes. Full logits match byte for byte at 8K and 32K; throughput trails by 2.4% and 0.71%. | No repeat of this prerequisite. |
| Does the normal DeepSeek architecture match ds4 prefill? | No. Reduction and Q-head fusions are landed. The checked native output-A screen takes 11.8877 s for 8192 tokens, versus 12.4620/12.4961 s ordinary bookends: +4.97% throughput. | Qualify production chunk geometry with attention; restore/attribute the remaining stages. The reproduced literal pipeline is around 7.5 s at 8K with different cache/math contracts and output cadence, so these clocks are not a matched final speed ratio. |
| Is Qwen's solo MTP deficit explained? | Partly. Fixed-depth step times are close; acceptance and reference variation carry much of the reported rate difference. Curated vocabulary does not win at every depth. | Same-history paired acceptance, plus a decision on the actual remaining step-time slope. Existing generated-history runs do not establish acceptance parity or justify changing the gate. |
| Does concurrent serving match the other engines? | Qwen's shared execution waves and HC/ragged-head sharing are landed. The latter gains 2.76% on paid C2 HTTP wall with exact replies. Four active slots lose 13.16%, so capacity stays two. | Fresh C1/C2/C4 comparison against Mia and TensorFold, now in flight. DeepSeek still needs independent-state batching. Solo parity does not establish concurrent parity. |

Reports: [literal 8K](experiments/ds4-complete-plan/README.md),
[literal 32K](experiments/ds4-matched-32k/README.md),
[output-A](experiments/ds4-output-prefix/README.md),
[Qwen sharing](experiments/qwen38-combined-sharing/README.md),
[capacity rejection](experiments/qwen38-capacity/README.md).
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
| 2. Attribute DeepSeek prefill by chain | **Done at the representative 4096-row geometry.** The unchanged native graph was measured on the literal study's input. It selected the exact six-slot reduction (+4.30%), Q-head fusion (+3.17%), output-A (+4.97% native screen), and the next HC-post/RMS fusion (+1.94% private screen). These independent gains are not added together. Native 2048-row chain attribution remains open. |
| 3. Measure the oracle's own noise at failing HCA rows | **Not run.** Bounds remain unchanged. A separate output-A plus HCA interaction now passes both 32K/128K fixed-history bounds, 128K held-out perplexity and one positive long-answer control. This resolves those candidate failures without a bound change, but does not measure oracle noise or qualify a serving default. |
| 4. Concurrency-aware row budget and both-model batching | **Partial.** Qwen's shared serving path and head/HC factors are landed. The cap-four adaptive screen regresses and is rejected. It did not test the suggested all-slot depth-zero/one strategy; that and DeepSeek C2/C4 shared execution with independent request state remain open. |
| 5. J64 with occupancy two | **Open.** J64 at the earlier occupancy was slower; that result does not reject the proposed two-block compiler specialization. It needs one captured-operand screen, not a model ladder. |
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
| Routed FFN direct versus materialized chain | Paid materialization is 2.35% slower; keep direct in the reference. |
| Native IQ2 consumers, input preparation, J64 and shared worklist | Isolated. Preparation bytes match; native consumer remains slower. J64 and worklist controls did not justify adoption. Occupancy-two remains untested. |
| Route weighting and six-slot summation | Exact native ordered fusion landed; +4.30% whole-prefill screen. |
| Q-head normalization and rotation | Exact native fusion landed; +3.17% whole-prefill screen. |
| Output-A projection and inverse rotation | Positive private and native screens. Selected native code passes its Spark check set; normal output-B remains intact. It remains a default-off benchmark option at the qualified 4096-row shape. |
| Wide HC input and HC suffix | Screened and rejected: −2.18% and +0.48%; no expanded quality or context matrix. |
| HC-post followed by flat RMS | Positive exact private screen, +1.94%; native overlay prepared and reviewed, not yet run. |
| HCA attention with output-A | Focused quality controls pass at unchanged bounds. Native production geometry and combined paid performance remain open. |
| Remaining Q/KV, compression/indexer, shared-FFN and HC producer chains | Not fully restored or bisected. Their current native attribution is recorded; the piecewise replacement work is incomplete. |

Detailed evidence is in the [optimization inventory](optimization-inventory.md)
and its linked reports, including the
[native chain attribution](experiments/ds4-production-prefill-attribution/README.md).
Transfers retain each consumer's layout, precision, rounding and state
contracts; shared Qwen weights do not by themselves implement DeepSeek batching.

## Work order and M3 exit

Use both Sparks: finish selected DeepSeek integration on one while obtaining
the fresh concurrent engine comparison on the other. Use one representative
bookended screen to settle occupancy and row-budget candidates. Measure
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
