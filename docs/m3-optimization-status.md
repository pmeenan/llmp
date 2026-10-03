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
| Does the normal DeepSeek architecture match ds4 prefill? | Nearly, at the measured 4096-row community geometry: the ds4 stage mechanisms bring native 8K prefill to 7.64–7.74 s against literal ds4's ~7.57 s, and they are now the fast plan's defaults. They now take every prefill chunk, a prompt's last, partial one included, and the 0731 GGUF's F32 HC, IQ2_XS and K-quant shared-expert types: runtime 7K prefill +4.2% (community) and +3.3% (original, byte-exact), and with output-A/HCA the community 7K prompt prefills in 6.78 s in-process (ds4's 6.53 s is an HTTP wall time from another session, not matched; native's served output-A/HCA time was 8.72 s there) ([partial chunks and other quant types](experiments/ds4-prefill-stages/README.md#partial-chunks-and-other-quant-types)). Before that, at production's then 2048-row chunks, they gained 7.21% on the community checkpoint but only 1.41% (runtime 1.26%) on the original checkpoint, whose types most guards did not yet admit. Reduction and Q-head fusions are landed. Output-A gains 4.97% at 4096 rows; adding HCA takes 8K prefill to 9.8792 s (+21.19% incremental throughput). Output-A/HCA is now serving's default on every prefill chunk of 64 rows or more (partial chunks under the owner's tie-aware rule): served 7K first token 8.26 s (community) and 9.37 s (original) on full chunks against main's 9.81 / 10.88, then 7.26 / 8.34 with partial chunks too ([default-on acceptance](experiments/ds4-output-prefix/README.md#default-on-acceptance)). Genuine runner controls preserve the passing 4K candidate's complete heads, initialized state and ordinary continuation. IQ2 J64 with occupancy two adds a separate 4.15% whole-prefill gain, with all six heads byte-exact. The private 2048-row extension gained 3.95% but failed one fixed-history quality row in its screen, exceeding the unchanged bound by 1.669249 nats, and was stopped; the 2026-10-03 acceptance's native output-A/HCA at 2,048-row chunks passes the 32K history (493/19/0), but serving keeps 4,096-row chunks. | The literal inverse preparation control attributes only 166.9 ms / 2.21% to fused input preparation, with exact heads. Restore/attribute the remaining stages using the checked native integration. The reproduced literal pipeline is around 7.5 s at 8K with different cache/math contracts and output cadence, so these clocks are not a matched final speed ratio. |
| Is Qwen's solo MTP deficit explained? | Partly. Fixed-depth step times are close; acceptance and reference variation carry much of the reported rate difference. Curated vocabulary does not win at every depth. | Same-history paired acceptance, plus a decision on the actual remaining step-time slope. Existing generated-history runs do not establish acceptance parity or justify changing the gate. |
| Does concurrent serving match the other engines? | [Four-request waves](experiments/qwen38-four-request-waves/README.md) replace fixed pairs: up to 16-row joined products, wide MXFP8 and expert-major routed kernels (bit-exact per request), 2048-cell wave alignment so graphs replay, and depth 2 in shared waves. Matched 8K HTTP cells: C4 32.0–32.7 (+20%), C2 29.6–29.7 (+9.8%), C1 unchanged with a byte-identical reply. Against current same-checkpoint TensorFold NVFP4 (31.87 / 24.79 / 21.46, an earlier session) native leads at C2/C1 and is level at C4 (+1.1%), and trails legacy Mia 18% at C4 and ~1% at C2. | C4 decode rate vs Mia (~58 vs ~100 tok/s); routed experts are near bandwidth at 12–16 rows. Replies under C2/C4 vary with arrival timing (HC on cuBLAS, head on GGML MMF are not column-count invariant). Same-checkpoint quality is not established by matching prompt IDs. DeepSeek batches up to four requests in exact waves ([report](experiments/deepseek-batching/README.md)). Against ds4 in the [same session](experiments/deepseek-batching/README.md#against-ds4-same-session), main's original-artifact plain waves give 0.84 / 0.89 / 0.87× ds4's 7K C1/C2/C4 rate. The gap is prefill: neither engine batches it, and ds4 prefills 1.71× faster. Decode-dominated, waves are level at C1/C2 and 0.80× at C4 (99 vs 82 ms four-row step). With output-A/HCA on full chunks and the routed products' pair scan fixed, community C4 is 1.08× ds4 at 7K and 0.93× at 124 tokens (85.5 vs ~80 ms step) in one session; partial decode waves between prompt units and plain waves from three requests were measured and not adopted ([wave step](experiments/deepseek-batching/README.md#four-request-wave-step-and-scheduling)). Prompts now prefill shortest remaining first, with aging and at most one prompt unit between a generating request's waves: 124-token C4 first tokens 0.9–3.4 s against 3.0–3.5 (ds4 0.8–3.5); mixed cells' short prompts no longer wait for a long one (DeepSeek 18.6–19.0 → 4.8–6.9 s); equal-length throughput 0.9–2.0% lower ([prompt order](experiments/deepseek-batching/README.md#prompt-order-adopted)). DSpark now runs with the community artifact; at HTTP C4 it trails plain waves (open). The community artifact ds4 runs now waves too (F16 HC mixes take the fused form): 7K C1/C4 11.17 / 15.50 tok/s, 0.84 / 0.93× ds4's earlier cells ([report](experiments/deepseek-batching/README.md#community-artifact-in-waves)). |

Reports: [literal 8K](experiments/ds4-complete-plan/README.md),
[literal 32K](experiments/ds4-matched-32k/README.md),
[output-A](experiments/ds4-output-prefix/README.md),
[IQ2 occupancy two](experiments/ds4-iq2-occ2/README.md),
[attention preparation](experiments/ds4-attention-preparation/README.md),
[native flat RMS](experiments/ds4-flat-rms/README.md),
[HC projection accumulation](experiments/ds4-hc-projection/README.md),
[native ds4 stage mechanisms](experiments/ds4-prefill-stages/README.md),
[Qwen sharing](experiments/qwen38-combined-sharing/README.md),
[capacity rejection](experiments/qwen38-capacity/README.md),
[depth-one row budget](experiments/qwen38-row-budget/README.md),
[four-request head sharing](experiments/qwen38-four-head-sharing/README.md),
[conditional four-head serving](experiments/qwen38-conditional-heads/README.md),
[four-head recovery](experiments/qwen38-four-head-recovery/README.md),
[captured GDN cohort](experiments/qwen38-gdn-cohort/README.md),
[concurrent engine comparison](experiments/serving-concurrent/README.md),
[four-request waves](experiments/qwen38-four-request-waves/README.md).
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
| 2. Attribute DeepSeek prefill by chain | **Done at the representative 4096-row geometry and refreshed after the selected factors.** The unchanged native graph selected six-slot reduction (+4.30%), Q-head fusion (+3.17%), output-A (+4.97% native screen), and HC-post/RMS (+1.94% private screen). These independent gains are not added together. Before occupancy-two tuning, the output-A/HCA profile takes 9.763 s; pair/down ordered intervals total 3.420 s (35.03% of wall). The selected occupancy-two factor then gains 4.15% whole-prefill. Remaining producer/product contracts and native 2048-row chain attribution stay open. |
| 3. Measure the oracle's own noise at failing HCA rows | **Not run.** Bounds remain unchanged. A separate output-A plus HCA interaction now passes both 32K/128K fixed-history bounds, 128K held-out perplexity and one positive long-answer control. This resolves those candidate failures without a bound change, but does not measure oracle noise. The 2026-10-03 acceptance shows the 32K history's outside rows flipping with arithmetic alone (the current default fails step 249; partial-chunk output-A/HCA fails 249 and 333; output-A alone 306), at level oracle-continuation likelihood. The owner then adopted a tie-aware greedy rule (D-085, 2026-10-03), judged against the model's reference run with a flip cap and a relative continuation bound. Under it the default's and partial chunks' flips pass, while output-A alone fails on its continuation. Its per-step tolerance is owner-accepted, not calibrated: step 249's NLL excess spreads 0.65–1.13 across jitLLM's own paths. Measuring the oracle's own noise remains open. |
| 4. Concurrency-aware row budget and both-model batching | **Partial.** Qwen's shared serving path and head/HC factors are landed. Cap-four adaptive is rejected. Fixed depth one alone loses 1.32%; sharing four full heads adds 5.80% with identical public replies. Conditional budgeting and four-head sharing then gain 7.05% against exact normal serving, but delay first completion by 83.23%, raise median latency 15.70%, fund 3.27 GiB more fixed memory and change replies. Four-head controls pass 13 complete heads and 20 initialized-state/cursor comparisons. Policy and sampling remain unqualified; no default changes. Depth zero and DeepSeek C2/C4 execution with independent request state remain open. |
| 5. J64 with occupancy two | **Done: measured and scoped native source checked.** Captured real gate/up products gain 28.05%; genuine native 8K prefill gains 4.15%, with six complete heads byte-exact. The final O3 object has 128 registers/16 stack bytes versus ordinary J128's 254/0; actual hardware occupancy was not measured. Production dispatch is restricted to the measured GB10 paired shape. |
| 6. Certified head screen/rescore | **Not run.** First count eligible candidate sets on captured real head inputs. No new kernel or model matrix is warranted before that count supports it. |

The review's additional 256K fixed-depth measurements, same-history controls,
all-active concurrent decode windows and wider-column GEMM comparison remain
separate open measurements. Its hypotheses and arithmetic estimates are not
recorded as local results.

The current TensorFold source inventory also selects independent-state GDN
launch sharing. One captured four-request native F32 operator screen reduces
paid replay wall by 26.51% (36.08% inverse-latency gain), with all outputs exact
and inputs/state/guards unchanged. Graphs are disabled and the serial control
uses the same private arithmetic-body refactor. Native wave integration,
graph/state controls and a paid serving gain remain untested; no default changes.

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
| Output-A projection and inverse rotation | Positive private and native screens. Selected native code passes its Spark check set; normal output-B remains intact. With HCA it is now serving's default on full 4,096-row chunks ([default-on acceptance](experiments/ds4-output-prefix/README.md#default-on-acceptance)). On every chunk (partial tails too) its two outside 32K rows are tie flips under the owner's tie-aware rule (2026-10-03), so partial chunks are serving's default too ([tie-aware re-scoring](experiments/ds4-output-prefix/README.md#tie-aware-re-scoring-and-partial-chunks)); at 2,048-row chunks it passes. |
| Wide HC input and HC suffix | Screened and rejected: −2.18% and +0.48%; no expanded quality or context matrix. |
| HC-post followed by flat RMS | Positive exact private screen, +1.94%; native overlay prepared and reviewed, not yet run. |
| Flat HC RMS launch size, 1024 versus 256 threads | One paid native 8K screen loses 1.30%, with −0.42% original duration movement. Ordinary complete heads remain exact; both candidate heads change all values with unchanged argmaxes. No isolated RMS latency or quality-bound verdict follows. Keep 1024 threads. |
| HC projection 32F accumulation/output | One paid native 8K screen gains 0.25%, inside −0.41% bookend movement; candidate heads change almost every value with unchanged argmaxes. Keep 16F. The remaining HC input cost is the F32 normalized tensor and its F32-to-F16 reconversion, which ds4 avoids by emitting F16 normalized rows from its HC expand. |
| ds4 stage mechanisms in the native graph | Eight exact changes (F16 HC mix rows, IQ2 pair SwiGLU + down-input Q8 write-back, expert sum in the HC post, shared F16 attention input, coalesced output-A repack, shared dense-pair quantization, F16 Q into jitLLM-owned CSA/HCA attention) gain 19.37% at 8K with byte-exact heads; with the qualified D2R down 23.70% (7.64–7.74 s vs literal ~7.57 s). D2R 32K PPL +0.04% vs ds4. **Now the fast plan's defaults**, each under its guard. Runner heads, state and continuation are byte-exact at 2048 rows on both checkpoints; the original-checkpoint 32K fixed-history control is byte-identical (491+21, none outside 0.947). Runtime 8K prefill on the served checkpoint gains 1.26%. |
| HCA attention with output-A | Focused 4096-row quality controls pass at unchanged bounds. A paid native 4096-row interaction gains 21.19% incremental throughput with 0.86% bookend movement. Genuine runner controls preserve full heads, initialized state and ordinary continuation, authenticate chunk positions before dispatch and fund all 32 128K plan shapes. **Serving's default on full 4,096-row chunks** (2026-10-03): every registered control passes on both GGUFs, byte-identical to the qualified runs; runtime 7K chat +7.5% at C1 and +11.5% at C4 against main; plans are no longer keyed by chunk position. Partial chunks failed the strict 32K history (two rows); under the tie-aware rule both are tie flips, and partial chunks are the default too: 7K chat +5% at C1 and +7–8% at C4 over full chunks alone ([tie-aware re-scoring](experiments/ds4-output-prefix/README.md#tie-aware-re-scoring-and-partial-chunks)). |
| Remaining Q/KV, compression/indexer, shared-FFN and HC producer chains | Not fully restored or bisected. Their current native attribution is recorded; the piecewise replacement work is incomplete. |

Detailed evidence is in the [optimization inventory](optimization-inventory.md)
and its linked reports, including the
[native chain attribution](experiments/ds4-production-prefill-attribution/README.md).
Transfers retain each consumer's layout, precision, rounding and state
contracts; shared Qwen weights do not by themselves implement DeepSeek batching.

## Work order and M3 exit

The literal preparation inverse factor and native flat-RMS launch-size screen
are complete: producer fusion/reuse contributes 2.21% to the literal pipeline,
while the smaller native RMS launch loses 1.30%. The conditional Qwen head
screen and its discard/retry/continuation controls are also complete; their
latency, memory and policy limitations retain the current serving default.
The fresh concurrent engine comparison is complete. The DeepSeek HC
projection's 32F accumulation/output screen is neutral (+0.25%) and
changes heads; keep 16F. Native ds4 stage mechanisms gain 19.37% exact and 23.70% with D2R at
8K, within about 1–2% of literal ds4, and are now the fast plan's
defaults. On the served original checkpoint at 2,048-row chunks only F16 Q
and the dense Q8_0 pairs apply (runtime 8K prefill +1.26%); the larger
native gap there is in its own types and chunk size. Serving now defaults
to 4,096-row DeepSeek chunks, 12.1% faster at 8K and 14.9% at 32K than
2,048 ([DeepSeek concurrent](experiments/deepseek-concurrent/README.md)).
On the other Spark, the positive captured GDN factor selects a bounded native
wave-composition screen with independent F32 state/output ranges, graph
dependencies and paid packing. That integration is unstarted. A separate
sparse-attention cohort factor would retain per-request selection and caches;
it is also unstarted and is distinct from the rejected shared-cell union.
Use one representative bookended screen to settle each candidate. Measure
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
