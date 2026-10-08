<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3 optimization status

Snapshot: 2026-10-04. **Remaining Qwen/DeepSeek speed gaps are accepted
for M3 and deferred to M9 by the owner.** The measurements still show those
gaps. Pipeline reproduction, a useful individual replacement,
and a selected serving default are different outcomes. M3 is complete
with that exception; the [frozen record](m3-record.md) maps its exit evidence. Individual reports retain conditions and evidence.

**Owner speed exception, 2026-10-04.** The owner explicitly accepts all
remaining Qwen/DeepSeek speed gaps for M3 and defers further optimization to
M9's full-engine pass. *Owner, 2026-10-07:* optimizations found on any
family, new kernels and fusions included, are ported to Qwen/DeepSeek as
they are found ([workflow.md](workflow.md)'s transfer rule); M9 keeps the
leads below. Experiments stopped at production main `3decb36`;
selected defaults stay in place. No queued candidate has a proven large
end-to-end gain. Known speed misses remain recorded; the exception does
not turn them into parity passes. The separate numerical, state, client,
swap and focused build/package controls now close M3 in their recorded
scopes; whole shipment tiers remain owed before publication. Broader
same-history acceptance is a retained diagnostic investigation.

The latest [paired shared-expert MXFP8 screen](experiments/qwen38-shared-mxfp8/README.md)
illustrates the distinction: operator latency falls 28.64%, but the paid C4
wave is neutral, so the candidate is rejected. A private DeepSeek greedy
ID-delivery prototype is frozen outside main after a narrow build and two
GPU argmax checks; its final benchmark evidence changes are unbuilt. It has
no model transcript/state qualification or measured speed result.

The [matched DeepSeek serving screen](experiments/deepseek-matched-serving/README.md)
now pays the same uncached 7,043 input IDs on both engines. Native's one-output
wall is 9.16% longer than ds4; at 256 outputs, C1 wall is 1.71% longer and
C2/C4 completed-token rates lead by 13.16%/18.36%. All 21 longer requests
complete, with no errors or reused prompt tokens, and native peak memory
drop is lower. These buffered literal/chat routes match model inputs;
they do not isolate decode, establish a cross-engine quality pass or close
the broader performance gate. Native bookend replies repeat exactly.

The [selected-head provenance audit](experiments/qwen38-head-provenance/README.md)
corrects requested-versus-effective row labels: the installed selected MTP
head has 47,172 rows even when 65,536 is requested. Existing timings and
acceptance counts remain valid; those nominal caps are not two layouts.

The [controlled Qwen slot-cap screen](experiments/qwen38-slot-knee/README.md)
retains four for its selected-head/4,096-row profile: six slots are neutral
in completed-token throughput and increase median request latency about
48% on two six-request short bursts. The runtime-generated key accepts a
controlled `max_slots: 4` in isolated owned state; overrides and a smaller
head invalidate it. Production state/defaults are unchanged.
The [controlled DeepSeek slot comparison](experiments/deepseek-slot-knee/README.md)
also retains four for its community+DSpark automatic profile. Six pays the
same six requests at every cap: short throughput gains 16.36% / 21.79%
with median latency 25.74% / 22.88% higher; one long cell gains 6.70% with
mean/median latency 9.30% / 10.26% higher. Six remains an explicit
throughput/tail option. A separate current-build prime produces an authentic
key for an owned slot-only calibrated-four record; explicit slot or
wave-cost-prefix overrides and a changed context invalidate it. Historical performance keys are
unchanged. Other profiles and broader slot calibration remain open.

<a id="remaining-gaps"></a>

## Retained performance gaps (M9)

The [controlled Qwen chunk screens](experiments/qwen38-prefill-chunks/README.md)
retain the 4,096-row fallback. At fresh 8K C4, 8,192 rows gain 2.69%
completed-token throughput with 4.58% more memory; 2,048 rows lose 9.28%.
The larger chunk preserves the difficult 32K anchor's complete logits and
four-slot waves' complete target rows/tokens, but a second prompt family
is 3.75% slower through its in-process check. Whole-state hashes span
different MTP history-buffer sizes and do not judge cross-chunk state
equivalence. No new default or calibrated value is selected.

| Question | Established result | Still needed |
| --- | --- | --- |
| Can jitLLM reproduce ds4's complete pipeline? | Yes. Full logits match byte for byte at 8K and 32K; throughput trails by 2.4% and 0.71%. | No repeat of this prerequisite. |
| Does the normal DeepSeek architecture match ds4 prefill? | Nearly, at the measured 4096-row community geometry: the ds4 stage mechanisms bring native 8K prefill to 7.64–7.74 s against literal ds4's ~7.57 s, and they are now the fast plan's defaults. They now take every prefill chunk, a prompt's last, partial one included, and the 0731 GGUF's F32 HC, IQ2_XS and K-quant shared-expert types: runtime 7K prefill +4.2% (community) and +3.3% (original, byte-exact), and with output-A/HCA the community 7K prompt prefills in 6.78 s in-process (ds4's 6.53 s is an HTTP wall time from another session, not matched; native's served output-A/HCA time was 8.72 s there) ([partial chunks and other quant types](experiments/ds4-prefill-stages/README.md#partial-chunks-and-other-quant-types)). Before that, at production's then 2048-row chunks, they gained 7.21% on the community checkpoint but only 1.41% (runtime 1.26%) on the original checkpoint, whose types most guards did not yet admit. Reduction and Q-head fusions are landed. Output-A gains 4.97% at 4096 rows; adding HCA takes 8K prefill to 9.8792 s (+21.19% incremental throughput). Output-A/HCA is now serving's default on every prefill chunk of 64 rows or more (partial chunks under the owner's tie-aware rule): served 7K first token 8.26 s (community) and 9.37 s (original) on full chunks against main's 9.81 / 10.88, then 7.26 / 8.34 with partial chunks too ([default-on acceptance](experiments/ds4-output-prefix/README.md#default-on-acceptance)). Genuine runner controls preserve the passing 4K candidate's complete heads, initialized state and ordinary continuation. IQ2 J64 with occupancy two adds a separate 4.15% whole-prefill gain, with all six heads byte-exact. The private 2048-row extension gained 3.95% but failed one fixed-history quality row in its screen, exceeding the unchanged bound by 1.669249 nats, and was stopped; the 2026-10-03 acceptance's native output-A/HCA at 2,048-row chunks passes the 32K history (493/19/0), but serving keeps 4,096-row chunks. | The literal inverse preparation control attributes only 166.9 ms / 2.21% to fused input preparation, with exact heads. Restore/attribute the remaining stages using the checked native integration. The reproduced literal pipeline is around 7.5 s at 8K with different cache/math contracts and output cadence, so these clocks are not a matched final speed ratio. |
| Is Qwen's solo MTP deficit explained? | Partly. Fixed-depth step times are close; acceptance and reference variation carry much of the reported rate difference. Curated vocabulary does not win at every depth. A [same-history first-verify screen](experiments/qwen38-same-history/README.md) now measures 9/12 native versus 11/12 Mia accepted drafts at four adjacent 32K anchors. Both nominal native caps execute the same selected 47,172-row head and give identical proposals; the fourth anchor loses two drafts where both targets agree on the shared input's next token. | The [cache-path controls](experiments/qwen38-same-history/README.md#cache-path-and-repeat-controls-2026-10-04) find reference draft and target variation, even a cold/cached difference with its determinism switches on; one repeated deterministic path matches native at 1/3. Smaller native prefill chunks retain the proposals. The [matched all-cold anchors](experiments/qwen38-same-history/README.md#matched-all-cold-anchors-2026-10-04) now accept 9/12 in both engines, with the separate cold p3 repeat also matching native. These four adjacent anchors do not establish broader acceptance parity; the remaining step-time slope stays open. |
| Does concurrent serving match the other engines? | [Four-request waves](experiments/qwen38-four-request-waves/README.md) replace fixed pairs: up to 16-row joined products, wide MXFP8 and expert-major routed kernels (bit-exact per request), 2048-cell wave alignment so graphs replay, and depth 2 in shared waves. Matched 8K HTTP cells: C4 32.0–32.7 (+20%), C2 29.6–29.7 (+9.8%), C1 unchanged with a byte-identical reply. Against current same-checkpoint TensorFold NVFP4 (31.87 / 24.79 / 21.46, an earlier session) native leads at C2/C1 and is level at C4 (+1.1%), and trails legacy Mia 18% at C4 and ~1% at C2. | C4 decode rate vs Mia (~58 vs ~100 tok/s); routed experts are near bandwidth at 12–16 rows. Replies under C2/C4 vary with arrival timing (HC on cuBLAS, head on GGML MMF are not column-count invariant). Same-checkpoint quality is not established by matching prompt IDs. DeepSeek batches up to four requests in waves whose steps equal the same steps alone ([report](experiments/deepseek-batching/README.md)). Against ds4 in the [same session](experiments/deepseek-batching/README.md#against-ds4-same-session), main's original-artifact plain waves give 0.84 / 0.89 / 0.87× ds4's 7K C1/C2/C4 rate. The gap is prefill: neither engine batches it, and ds4 prefills 1.71× faster. Decode-dominated, waves are level at C1/C2 and 0.80× at C4 (99 vs 82 ms four-row step). With output-A/HCA on full chunks and the routed products' pair scan fixed, community C4 is 1.08× ds4 at 7K and 0.93× at 124 tokens (85.5 vs ~80 ms step) in one session; partial decode waves between prompt units and plain waves from three requests were measured and not adopted ([wave step](experiments/deepseek-batching/README.md#four-request-wave-step-and-scheduling)). Prompts now prefill shortest remaining first, with aging and at most one prompt unit between a generating request's waves: 124-token C4 first tokens 0.9–3.4 s against 3.0–3.5 (ds4 0.8–3.5); mixed cells' short prompts no longer wait for a long one (DeepSeek 18.6–19.0 → 4.8–6.9 s); equal-length throughput 0.9–2.0% lower ([prompt order](experiments/deepseek-batching/README.md#prompt-order-adopted)). DSpark now runs with the community artifact. Its C4 waves trailed plain ones by 5–12% at HTTP. Each wave now chooses DSpark or plain decode from counted acceptance against a measured per-width cost, with no clock. That brings C4 to 1–4% under plain waves and up to 7% over DSpark alone, though where forms mix a C4 reply may differ from C1, and between runs whose requests arrive in another order; `wave_form = "speculative"` restores the cohort-equals-alone control ([DSpark and plain waves](experiments/deepseek-batching/README.md#adaptive-dspark-and-plain-waves)). [Wave lanes](experiments/deepseek-batching/README.md#wave-lanes) run each slot's attention and state work on a stream of its own inside the wave's graph, bit for bit: four-request step 86.5 → 82.8 ms; 124-token C4 +4.7–4.8% plain and +4.6–5.1% with DSpark (7K +1.1–1.9%, within noise); [Joined draft blocks](experiments/deepseek-batching/README.md#joined-draft-blocks) then draft every slot's block in one graph, also bit for bit: four-request DSpark wave 240.9 → 231.2 ms (community), end to end level within noise at C4 and C5; its verify's routed experts remain most of it. The community artifact ds4 runs now waves too (F16 HC mixes take the fused form): 7K C1/C4 11.17 / 15.50 tok/s, 0.84 / 0.93× ds4's earlier cells ([report](experiments/deepseek-batching/README.md#community-artifact-in-waves)). |

Qwen's GGUF plain waves now share one-row dense and routed quantized
products and the full head. Full heads and state match unjoined waves
byte for byte, eagerly and with capture/replay
([GGUF qualification](experiments/qwen38-gguf/README.md#one-row-gguf-waves-2026-10-03)).
Matched UD-IQ3_XXS HTTP gains 31.08% at C2 and 69.63% at C4 for
183-token prompts, and 29.42% at C4 for 8,258-token prompts (256 outputs
each, plain decode; solo within 0.45%).
Qwen now also runs private attention and recurrence on concurrent streams
from four requests, with exact full logits and final state on short and
long fixed histories. The short C4 HTTP screens gain 3.76% with NVFP4/MTP
and 7.65% with UD-IQ3_XXS plain decode
([wave lanes](experiments/qwen38-four-request-waves/README.md#wave-lanes)).
The NVFP4/Mia concurrency gap remains open. A [fresh fast-start Mia C4 screen](experiments/qwen38-four-request-waves/README.md#fast-start-mia-refresh--2026-10-04)
on 2026-10-04 measures 32.44 / 31.91 native bookends versus 37.86 completed
tokens/s at 8,256 input tokens: native is 15.01% below Mia, with 1.66% native
bookend wall movement and peak MemAvailable drop 0.801 times Mia's. Mia
reaches readiness in 172 seconds. This refresh includes wave lanes and the
tighter workspace; it does not qualify cross-engine quality.

Qwen's [selected MTP head sharing](experiments/qwen38-draft-head-waves/README.md)
now gains 1.77% completed tokens/s at C2, with short/8K target rows,
tokens and target/drafter states exact. C4's wider joined drafting remains
unadopted: it cancels the head gain and changes drafter-state arithmetic.
The production draft-wave limit stays two; this does not close C4 parity.

Qwen's [verify alpha/beta fusion](experiments/qwen38-gdn-gates/README.md)
retains both Linear products and original saved-row restoration while
replacing four pointwise launches with one. C4 verify latency falls 1.11%;
matched uncached 8K C4 HTTP gains 1.34%, with 0.36% bookend wall movement.
Fixed-history tokens, full target rows and initialized target/drafter
states remain exact, including a swap with rejected rows awaiting restore.
Prefill and the exact plan retain the primitives; broader parity stays open.

DeepSeek's [wide IQ2_XXS gate/up pass screens](experiments/deepseek-expert-passes/README.md)
share weight decoding across four or two tokens with the original warp-row
arithmetic. Both pass the existing exact solo/wave, discard/retry and
departed-slot controls, but C4 DSpark wave median latency rises 0.512% /
1.400% versus their bookend means. Keep the current loop; routed down and
worklist costs remain separate candidates.

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
source passes its Spark check set. That earlier standalone screen is not
the later default-on output-A/HCA qualification summarized below.

<a id="acceptance-evidence-still-owed"></a>

## Acceptance evidence at M3 exit

Existing short and long oracle/PPL controls, forced rejections, sampled
harness distributions and exact own-path restoration remain evidence for
their pinned profiles. Qwen's owner-accepted short-prompt divergence and
DeepSeek's tie-aware rule remain as documented; the owner's speed exception
changes no correctness, memory or swap bound.
The frozen record distinguishes inherited controls from an unqualified
profile; it does not relabel historical runs as executions of later commits.

| Item | Evidence at exit and later work |
| --- | --- |
| Final same-reference numerical qualification | **Qualified in the named scopes.** The [current Qwen C4 control](experiments/qwen38-concurrent-oracle/README.md) captures four slots twice on the frozen 32K history: all eight 512-row cells have 477 agreements, 35 near-ties and zero outside steps under the unchanged 1.765 bound. Complete rows and initialized target/MTP state match across slots/repeats. This teacher-forced geometry check supplies no natural acceptance or unrestricted cohort claim. DeepSeek retains its registered default-on oracle/PPL controls and exact forced-form cohort/state evidence; matched serving timings alone supply no quality pass. |
| Sampled route/cohort qualification | **Mapped.** Independent source review confirms serving uses the same full-target-row Sample/VerifyDraft algorithms and parameters, branch-local scratch, seed/stream/absolute-position keys and per-slot Accept/Discard. Resumed branch controls preserve those keys and anchors. This inherits the registered per-drafter histogram evidence in its named profiles; it does not claim a fresh C4 HTTP histogram measurement. |
| Qwen same-history acceptance (diagnostic investigation) | The matched all-cold collection accepts 9/12 in each engine at four adjacent 32K anchors, with a separate p3 repeat. Broader histories, contexts and steps remain unqualified. Broader parity is a retained investigation, not an additional exit criterion; this diagnostic does not replace target-quality or speed criteria. |
| Speed exception and M9 follow-up | Qwen's fresh C4 Mia comparison remains about 15% behind; no head clears every strict long-context rung. DeepSeek's matched 7K plain cells are useful but do not isolate all prefill/decode or speculative comparisons. The owner accepts all remaining Qwen/DeepSeek speed gaps for M3, including concurrent and long-context speed/scaling; further optimization moves to M9. Memory and correctness criteria remain unchanged. |
| Frozen record and shipment checks | **Recorded.** The [M3 record](m3-record.md) maps quality, sampler, state, swap and client evidence. The [final checks](experiments/m3-final-checks/README.md) include the 1,567-test Spark suite, 1,272-test ARM cross/qemu run (29 skips) and final ARM package inventory/install/purge fixture. Whole workstation/sanitizer/container/offline/confined-job tiers were not all run and remain owed before shipment. |

The current Qwen concurrent control records the physical selected head,
12-row verifies/four-row tail, literal conditioning, reference path and
complete repeats. It uses no new tolerance and leaves other histories,
widths and natural acceptance outside its numerical claim.
Additional stage ports, speed-focused head certification and calibration
of other profiles move to M9's full-engine optimization pass. Broader acceptance and oracle-noise
studies remain diagnostic investigations; none is converted into a
correctness pass by the speed exception.

## Independent review recommendations

The analysis-only review used main `7c700ba` on 2026-10-01. Its ranked list
is accounted for below; a related experiment is not marked as completion
of a different proposed experiment.

| Recommendation | Status and decision |
| --- | --- |
| 1. Same-history Qwen draft acceptance | **Representative collection complete; parity open.** Four adjacent 32K anchors use literal identical prompt IDs, actual three-draft sampler counts and complete first-verify heads. Native's two requested caps use the same selected 47,172-row head and agree exactly, accepting 9/12 versus the pinned fast Mia reference's 11/12. At the fourth anchor native proposes 1622 while both targets choose 1330 on the shared input ([report](experiments/qwen38-same-history/README.md)). Later [cache-path controls](experiments/qwen38-same-history/README.md#cache-path-and-repeat-controls-2026-10-04) show reference draft/target variation and a repeated deterministic path matching native. The original two-draft gap is not established as stable; smaller native prefill chunks retain its proposals. The [matched all-cold collection](experiments/qwen38-same-history/README.md#matched-all-cold-anchors-2026-10-04) now accepts 9/12 in both engines, with zero cached reference tokens and a separate p3 repeat also matching native. All twenty conditioned target-row argmaxes agree; reference logits still vary across that repeat. Broader acceptance parity and the remaining step-time slope stay open; no speed or quality-bound claim follows. |
| 2. Attribute DeepSeek prefill by chain | **Done at the representative 4096-row geometry and refreshed after the selected factors.** The unchanged native graph selected six-slot reduction (+4.30%), Q-head fusion (+3.17%), output-A (+4.97% native screen), and HC-post/RMS (+1.94% private screen). These independent gains are not added together. Before occupancy-two tuning, the output-A/HCA profile takes 9.763 s; pair/down ordered intervals total 3.420 s (35.03% of wall). The selected occupancy-two factor then gains 4.15% whole-prefill. Remaining producer/product contracts and native 2048-row chain attribution stay open. |
| 3. Measure the oracle's own noise at failing HCA rows | **Not run.** Bounds remain unchanged. A separate output-A plus HCA interaction now passes both 32K/128K fixed-history bounds, 128K held-out perplexity and one positive long-answer control. This resolves those candidate failures without a bound change, but does not measure oracle noise. The 2026-10-03 acceptance shows the 32K history's outside rows flipping with arithmetic alone (the current default fails step 249; partial-chunk output-A/HCA fails 249 and 333; output-A alone 306), at level oracle-continuation likelihood. The owner then adopted a tie-aware greedy rule (D-085, 2026-10-03), judged against the model's reference run with a flip cap and a relative continuation bound. Under it the default's and partial chunks' flips pass, while output-A alone fails on its continuation. Its per-step tolerance is owner-accepted, not calibrated: step 249's NLL excess spreads 0.65–1.13 across jitLLM's own paths. Measuring the oracle's own noise remains open. |
| 4. Concurrency-aware row budget and both-model batching | **Partial.** Qwen's shared serving path and head/HC factors are landed. Cap-four adaptive is rejected. Fixed depth one alone loses 1.32%; sharing four full heads adds 5.80% with identical public replies. Conditional budgeting and four-head sharing then gain 7.05% against exact normal serving, but delay first completion by 83.23%, raise median latency 15.70%, fund 3.27 GiB more fixed memory and change replies. Four-head controls pass 13 complete heads and 20 initialized-state/cursor comparisons. Policy and sampling remain unqualified; no default changes. Depth zero and DeepSeek C2/C4 execution with independent request state remain open. |
| 5. J64 with occupancy two | **Done: measured and scoped native source checked.** Captured real gate/up products gain 28.05%; genuine native 8K prefill gains 4.15%, with six complete heads byte-exact. The final O3 object has 128 registers/16 stack bytes versus ordinary J128's 254/0; actual hardware occupancy was not measured. Production dispatch is restricted to the measured GB10 paired shape. |
| 6. Certified head screen/rescore | **Not run.** First count eligible candidate sets on captured real head inputs. No new kernel or model matrix is warranted before that count supports it. |

The review's additional 256K fixed-depth measurements, broader same-history controls,
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
dependencies and paid packing. That [integration now completes](experiments/qwen38-gdn-cohort/README.md#native-wave-integration-screen--2026-10-04):
all full target rows and final state match the unmodified control, with
capture/replay and 1,404 recurrence cohort launches. Verify median is
107.399 ms against disabled bookends of 107.228 / 107.018 ms, so it is not
adopted. The isolated serial-launch gain does not establish a wave gain. A separate
sparse-attention cohort factor would retain per-request selection and caches;
it is also unstarted and is distinct from the rejected shared-cell union.
These are retained leads for M9's full-engine optimization pass, not queued
M3 work. Any resumed experiment should use one representative paid screen,
record reference variation before attributing a stable proposal deficit,
and expand only for an unresolved decision or selected implementation's
qualification. No isolated operator gain supplies an end-to-end gate pass.

Growing state, turn reuse, maximum-context execution and continuing-context
swap controls are complete. Pending model switches now pause chat cohorts
at completed units and resume their exact continuations after one substitute
cohort ([model turns](experiments/model-turns/README.md)); literal completions
now join those cohorts, with scores and response state preserved across
switches ([controls](experiments/literal-batching/README.md)). M3 is complete
with the named numerical/sampler mapping, frozen record and focused checks
above, plus the owner's speed exception. Remaining optimization and broader
acceptance studies belong to M9; unimplemented review suggestions and
pipeline-restoration ideas stay recorded. No package ships before its owed
whole shipment checks.


The [standard-client criterion passes](experiments/m3-standard-client/README.md)
on 2026-10-04: unmodified OpenAI SDK 3.3.1 completes DeepSeek → Qwen →
DeepSeek and streamed Qwen, matching the native controls' visible text,
reasoning and token accounting with natural stops. Production identity,
client isolation and strong retirement controls pass. This does not close
the separate performance/quality or image-inclusive swap criteria.

The [final image-inclusive swap table](experiments/m3-final-swap/README.md)
passes all 32 rows in one production process, with corrected fresh-token
endpoints: worst LLM and prepared swap 9.853 s. Restored 8K states and
16-token continuations, the pinned image control and four regenerated images
are exact; prepared 8K LLM graphs replay. This closes the current
implementation's final swap check. It does not establish numerical
qualification or turn the accepted speed gaps into measured parity.
