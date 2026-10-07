<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 bounded production recipe

The approved Gemma 4 26B-A4B artifact (`4ddb360c…`) now has a bounded
production recipe, adopted on the evidence below. It applies at resolved
context ≤8,192 and ≤4 slots:

- an uncalibrated 1024-row prefill fallback; smaller explicit overrides stay
  intact;
- both norm chains;
- MoE route/reduce fusion;
- joined waves with owner attention;
- stock's full final FFN before bounded frontier-head publication.

The earlier narrow-final recipe reproduced the recorded official llama.cpp
v0.6.0 greedy tokens for every owner and ran within 0.6% of that reference
method. Those adoption measurements ran on `spark-b`
(GB10), 2026-10-06, with the same 4×8,192-ID inputs as the
[dense31 bridge](../gemma31-serving-bridge/README.md), against the same
image. The reference is the bridge helper's diagnostic copy for 26B (layer
and width check, `n_batch`/`n_ubatch` 1024, F16 K/V, independent sequences,
normal ring), kept external.

## Before

Before adoption, Gemma26 kept the scalar route with a 128-row prefill cap:

- **C1:** paid cycles 7.91/7.94 s against stock's 4.97/5.04 s (+57%), with
  prefill 5.22 s against 2.33 s.
- **C4:** the scalar cohort decoded each owner separately, 20.24 s against
  14.29/14.46 s (+41%).
- **Tokens:** both diverged from stock's from the second token.

The cap alone doubled prefill time: 64 chunks against stock's 8.

## Each piece

These C1 screens were run in fresh processes with temporary toggles. The
toggles are not committed.

| Recipe | C1 second cycle | Tokens equal to stock's |
| --- | --- | --- |
| Default recipe, 128-row chunks | 7.91 s | no |
| Default recipe, 1024-row chunks | 5.71 s | no |
| Norm chains and MoE route/reduce, 1024-row chunks | 5.02 s | all 129 |
| Same with the fused quantized FFN | 5.01 s | all 129 |

For C4 at 1024-row chunks with the norm/MoE recipe:

- Joined waves without owner attention took 14.54 s, and every owner's
  tokens differed from stock's.
- With owner attention they took 14.41/14.42 s, and every owner's tokens
  matched.

The fused quantized FFN is neutral for Gemma26 (its shared-expert FFN is
small) and stays off. Exact agreement depends on the matched geometry, the
routing arithmetic and owner attention together.

## Recorded production adoption

These bookends ran reference/native/native/reference in fresh processes,
with the new defaults and no toggles. Native ran warm, second and third
cycles; the reference ran warm and paid cycles.

| Workload | Reference paid | Native second | Native third | Native − reference |
| --- | --- | --- | --- | --- |
| C1 | 5.013 / 5.023 s | 5.051 / 5.047 s | 5.002 / 4.993 s | +0.6% (third −0.3%) |
| C4 | 14.374 / 14.374 s | 14.462 / 14.451 s | 14.351 / 14.313 s | +0.6% (third −0.3%) |

All 129 emitted tokens match stock's in every owner and every cycle of both
native processes, for C1 and for C4. These recorded stock bookends publish
CPU heads; they do not establish parity against a later GPU-token reference
method. The correction's cost comparison below is native versus native.

**Corpus control.** Native `ScorePrompt` and the reference's scalar one-row
queries were compared over the first owner's first 1,024 authentic IDs:

- all 1,024 complete heads are byte-identical;
- zero strict choice differences;
- identical mean NLL (10.60303447) over 1,023 targets.

Each engine's first and repeat run is byte-exact, as is native's initialized
state. The high absolute perplexity on this corpus text is the same in both
engines. It shows parity, not semantic quality.

## Scope

This is bounded: approved 26B-A4B, context ≤8,192, ≤4 slots, ordinary
arithmetic. Larger contexts or cohorts and explicit diagnostics keep their
prior recipe and calibration identity. Default context 262,144 is not
reduced, so the recipe applies only where configured within bounds.
Thinking/tools, assistants and sustained performance remain owed. The HTTP
controls below qualify the corrected ordinary response lifecycle and preserve
the historical failed cross-geometry continuation assertion as evidence. A natural two-turn `jitllm-runtime chat`
through the default selection used 1024-row chunks and stopped naturally ("The
capital of France is Paris." / "The capital of Italy is Rome."). Among the GPU
suites, only the opposite-profile switch case in `gemma4_serving_gpu_test`
reaches this recipe (at 16-row chunks, 2 slots), and it passes; no unit test
covers the 1024-row, four-owner geometry, which this report's runs exercise.

## HTTP lifecycle screen

### Preserved initial failure and cause

The ordinary runtime on `945d5ab` was rebuilt on spark-b and exercised at
context 8,192 and four slots, with no prefill override. Resolved settings and
the actual startup log select the 1024-row fallback; the checked production
selection source supplies the recipe witness. No diagnostic policy was forced.
The runtime SHA-256 is `f1ed1bcd6ec25aa8d2c893a8543d9f03e22671e6929a4d5305b3ce7471105b26`.

| Control | Result |
| --- | --- |
| Chat nonstream/SSE text and finish reason | exact |
| Nonstream and SSE stop suppression | pass |
| Client disconnect followed by the same chat request | exact response |
| Four natural SSE requests, one coordinated departure | three peers complete; each observes 127 later content reads |
| Followup after the four-client control | healthy |
| Literal 16 tokens versus 4 + fresh-prefill 12 | **FAIL**, first difference at generated token five |

Client reads may consume buffered output. They establish client-observed peer
completion after departure, without proving precise backend cancellation or a
four-owner GPU batch. Token IDs are carried in logprobs; these lifecycle checks
do not establish numerical likelihood parity.

The literal comparison uses the first 256 authentic corpus IDs. Its extended
prompt is exactly those IDs plus the first four generated IDs. Literal requests
force fresh prefill: the comparison changes a 256-row prefill followed by four
scalar steps into a fresh 260-row prefill, rather than replaying saved state.
At the first difference, native's scalar route chooses 3496 over 2846 with a
0.38437271 logit gap; fresh prefill chooses 2820 over 3496 with a 0.06786156 gap.
The comparison remains failed, with no new quality allowance.

A bounded control using the original v0.6.0 public API and the identical
260-ID input confirms that stock's first four natural choices agree. Its
scalar head selects the same top two IDs and logprobs as native's scalar
response (maximum reported difference below `1e-13`). Stock's fresh prefill
also changes the next token, choosing 3527 over 2820 with a 0.01134968 gap.
Thus fresh-versus-scalar token invariance also fails in stock. Stock's two
complete heads have maximum absolute difference 5.15699220 and RMS difference
0.94970293.

A [native full-head control](native_continuation.cc) uses the exact HTTP
production options, including four slots and capacity for four frontier heads.
All three heads are finite and byte-exact across two native repeats before
comparison with stock. Their top two IDs and score gaps match the HTTP responses;
reported logprob differences are at most `2.65e-13` from softmax summation.

| Native versus stock head | Byte-exact | Maximum absolute difference | RMS difference |
| --- | --- | ---: | ---: |
| 256-row prefill | no | 0.33440667 | 0.05239455 |
| 256 rows + four scalar steps | yes | 0 | 0 |
| Fresh 260-row prefill | no | 0.33892155 | 0.04949747 |

The difference comes from narrowing the final attention/residual rows before the
last FFN. Stock's ordinary Gemma4 graph keeps all final FFN rows and gathers
after output normalization. Changing only native `frontier_head=false` restores
**all three complete heads byte for byte**, with finite, byte-exact own repeats.
One frontier head is still published; this does not request all output heads.

The route-fusion hypothesis was rejected. Disabling only native route fusion
increases the differences. A reversible, controller-only stock observer preserves
all three original stock heads byte for byte and records all 30 route and
reduction selections for rows 256, 260 and 1, with no memory-gate refusal.
Those records describe controller selection during graph construction/capture,
not kernel counts on graph replay. No floating device kernel was rebuilt.

A separate Dense31 control uses its ordinary 256-row cap, four slots and head
capacity four, with 128 authentic prompt IDs plus four supplied corpus IDs.
It compares 128-row prefill, four scalar steps and fresh 132-row prefill. Supplied
tokens are teacher forcing: native and stock each naturally match zero of the
four supplied corpus IDs, consistently across repeats. Native and stock each
freeze finite, byte-exact own repeats first. This is a forced-prefix control,
not a natural four-token Dense31 continuation.

| Dense31 native versus stock | Narrow-final maximum / RMS | Full-final maximum / RMS |
| --- | ---: | ---: |
| 128-row prefill | 0.43971109 / 0.04428264 | 0 / 0, byte-exact |
| 128 rows + four supplied scalar steps | 0 / 0, byte-exact | 0 / 0, byte-exact |
| Fresh 132-row prefill | 0.58243847 / 0.12342295 | 0 / 0, byte-exact |

This transfers to both approved Gemma4 profiles. It does not transfer to Gemma3:
stock's Gemma3 graph explicitly gathers final attention/residual rows before
post-normalization and the FFN.

The [native control](native_continuation.cc) and
[public-API stock control](llama_continuation.cc) accept explicit `gemma26` and
`gemma31` profiles. The [head comparator](head_compare.py) freezes same-shape
native repeats before diagnostic full-head comparison, without introducing a
quality allowance. The 26B external 260-ID input SHA-256 is
`598c3caf2cb265dbdf3c270b545fca0c3b805794d1a525d8c7b77dfd2668833f`;
its source is the failed HTTP screen's `literal-comparison.json`, field
`continuation_prompt_ids`. Dense31 uses its first 132 IDs, SHA-256
`a944840947055424c9666d84011f45803fafc8c8a2bb5b440588e49097d5e35d`.
The runtime, native nodes, stock contexts/models and owned containers retired.
Raw prompts, responses, heads, allocator addresses and logs stay outside Git.

## Full-final-FFN cost screen

A [benchmark-only constructor wrapper](frontier_control_wrap.cc) changes only
`frontier_head` on the exact bounded serving options. It reuses the unchanged
production caller's 8,063-ID prompt plus 128 decode steps, with warm, second and
third cycles. These cycles choose tokens on the device and publish no extra
heads. Four fresh processes run narrow/full/full/narrow; the
[aggregate checker](cycle_compare.py) requires every owner's 129 emitted IDs
and 8,191-token history to repeat exactly within and across same-policy
processes before averaging finite positive timings.

| Gemma26 C1 mean | Narrow final | Full final | Full − narrow |
| --- | ---: | ---: | ---: |
| Second cycle | 4.96094618 s | 4.96365170 s | +0.055% |
| Third cycle | 4.89918857 s | 4.91984721 s | +0.422% |

All tokens and histories also agree across policies. Narrow C1 tokens match
the retained production identity. This short paired screen measures native
cost; the stock timings above remain recorded evidence from their original
run. The additional affected-policy cost checks retain the same method:

| Mean cycle | Narrow final | Full final | Full − narrow |
| --- | ---: | ---: | ---: |
| Gemma26 C4 second | 14.18983055 s | 14.25786851 s | +0.479% |
| Gemma26 C4 third | 14.02155663 s | 14.13630140 s | +0.818% |
| Dense31 C1 second | 23.70689306 s | 23.79737919 s | +0.382% |
| Dense31 C1 third | 23.58890004 s | 23.65057434 s | +0.261% |
| Dense31 C4 second | 59.98956060 s | 60.00797026 s | +0.031% |
| Dense31 C4 third | 59.54649073 s | 59.46880915 s | −0.130% |

Every owner's emitted tokens and full history repeat exactly within and across
same-policy processes, and tokens also agree across policies. Gemma26 C4 uses
the authenticated current narrow baseline; no historical C4 token hash was
assumed. Dense31 C4 ran as two supervised halves, retaining the same source,
binary and receipt between N/F and F/N. These short screens support the
correction without a prompt-length exception or further kernel tuning.

## Corrected ordinary HTTP gate

The final ordinary runtime was rebuilt from `4bd1b64` plus this correction on
spark-b, after a full checksum source sync and an empty itemized dry run.
No diagnostic policy is forced. Its SHA-256 is
`78092be02877642427df5bcdf3948375167904bd378b16e4254573b89579c5d5`;
the unchanged locked build receipt is
`7ea0d2b65bb8ddeb35af2887c65e1b0281c1406edf58db411941aa57b5193d48`.
Both profiles use context 8,192/four slots and their resolved prefill fallback
(26B 1024, Dense31 256). Actual startup/settings and the checked selection
source witness the ordinary recipe, including full final FFN and bounded head
publication. Runtime/receipt hashes are consumed unchanged by each HTTP run.

[http_control.py](http_control.py) now requires the frozen reference input and
all three complete head hashes, lengths and finiteness. It keeps literal
16-token/four-token/fresh-12 requests, requires exact first-four prefixes and
same-geometry repeats of complete choices/logprobs, and compares first-row top2
IDs with the matched stock prefix and fixed fresh-prefix heads. The invalid
cross-geometry token-invariance assertion is replaced by these reference and
repeat gates; its original FAIL remains above. No numerical quality allowance
was added. Complete F32-logit parity is established by the separate three-head
controls, rather than inferred from HTTP token IDs carried in logprobs.

| Corrected HTTP control | Gemma26 | Dense31 |
| --- | --- | --- |
| Same-geometry literal token/score repeats and stock top2 IDs | exact | exact |
| Cross-geometry 16 versus four + fresh 12, descriptive only | first difference at index 4; 4/16 matching | 16/16 matching |
| Chat 16/4/16 checkpoint response and finish replay | exact; 20 cached prompt tokens on each replay | exact; 20 cached prompt tokens on each replay |
| Nonstream/SSE text, stops and disconnect recovery | pass | pass |
| Four natural requests, one coordinated client departure | three peers complete; 127 later reads each | three peers complete; 127 later reads each |
| Followup after departure and runtime shutdown | healthy; exit 0 | healthy; exit 0 |

The chat control starts with zero cached prompt tokens, then requires positive
reported cache reuse and exact full response replay after the shorter answer.
Chat exposes text here, so this is a checkpoint response-replay gate, distinct
from full-head or token-ID continuation proof. Client SSE reads can be buffered;
the departure check proves client-observed completion, without claiming precise
backend cancellation or a four-owner GPU shape.

The final focused Spark checks also pass: four Gemma settings tests, three
calibration identity/stale-record tests, and six GPU publication, complete
likelihood and paused opposite-profile continuation tests. The changed recipe
invalidates calibration through the existing build identity (commit version,
or executable stamp for modified/no-Git builds); no new setting or format is
needed. Manual/unbounded arithmetic defaults stay unchanged. The full suite,
broader quality/depth/peak and sustained qualification were not repeated for
this correction. Raw HTTP responses, token arrays, scores, heads and logs remain
external; all supervised jobs and runtime nodes retired.

## Provenance

The recorded adoption table used native main `171cadb` plus its recipe change.
The 2026-10-07 cause and cost controls used `945d5ab` and `e3a9ebb` plus the
checked-in harnesses on spark-b (GB10, driver 580.178.04, NVCC 13.4 SDK).
The intervening Gemma3 additions do not execute in these Gemma4 controls.
The reference is the official image
`ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f…0607db` (v0.6.0/d812350) with
`gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`. The adoption helper is the dense31 bridge's `llama_serving.cc`, changed only
to accept 26B's shape, take its batch from `JITLLM_DIAG_CHUNK` and allow profiler
injection. The newer three-head controls use `llama_continuation.cc` against
the retained original libraries. Dense31 uses approved artifact `32c92e07…`
and `gemma-4-31B-it-UD-Q4_K_XL.gguf`. Raw logs, heads and token
files stay external on spark-b.
