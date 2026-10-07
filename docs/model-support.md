<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model support matrix

Which models jitLLM runs, from which prepared artifacts, with what, and on
what evidence. Support is earned per checkpoint and configuration
([vision](vision.md#success-criteria), [features](features.md)): a model
not listed here is unsupported, and a row claims no more than its evidence
links show. Where evidence is missing the row says "not verified". Started
in M3 (moved from M5); every change that alters a row's capability or
evidence updates it in the same change ([workflow](workflow.md)).

## Status and level

**Status** is how a user can run the model today:

- **Served (M3):** registered by `jitllm-runtime` from `[models.<name>]`
  and run by its `chat` and `swap-table` commands
  ([runtime-serving](runtime-serving.md)). The LLMs also serve native Chat
  Completions and literal Completions on loopback and the tailnet. M3 acceptance is
  recorded per profile in the [M3 record](m3-record.md).
- **Harness-only:** runs only in a benchmark harness under `benchmarks/`.
- **Fixture:** a small M2 test model, harness-only, never served.

**Level** is the highest rung of the ladder in
[features.md](features.md) (unsupported, import-only, resident-correct,
paged-correct, distributed-correct, performance-validated) that the linked
evidence reaches. Performance-validated means a milestone exit judged it;
an accepted milestone speed exception does not establish measured parity.
The [M3 record](m3-record.md) keeps the remaining Qwen/DeepSeek gaps and
the owner's M9 deferral beside the qualification evidence.

The native GGML dependency now uses llama.cpp v0.6.0 (`d8123504`, GGML
0.26.0); its [integration checks](experiments/ggml-release-refresh/README.md)
cover the source update. Model-quality and performance results below retain
the source/image identities of their linked experiments. Those historical
results do not qualify the new pin; matched new-release model comparisons
are recorded separately as they complete.

## Summary

| Model | Role | Status | Level |
| --- | --- | --- | --- |
| [DeepSeek V4 Flash 0731](#deepseek-v4-flash-0731) UD-Q2_K_XL | target | Served (M3) | paged-correct |
| [DeepSeek V4 community IQ2_XXS](#deepseek-v4-community-iq2_xxs) | target | Served (M3), four-request waves; matched HTTP cells | resident-correct, short oracle trajectories |
| [DSpark](#dspark) for DeepSeek V4 Flash 0731 (Q8_0) | drafter | Served (M3), with its target | paged-correct |
| [Qwen3.8 Flash Next](#qwen38-flash-next) NVFP4 | target | Served (M3) | paged-correct, one accepted greedy divergence (below) |
| [Qwen3.8 MTP](#qwen38-mtp) | drafter | Served (M3), with its target | paged-correct |
| [Qwen3.8 Flash Next GGUF](#qwen38-flash-next-gguf) UD-IQ3_XXS | target | Served (runtime registers it; not in the swap table) | resident-correct against llama.cpp on the same GGUF; paged through the runtime |
| [Qwen-Image-2.1](#qwen-image-21) BF16 | composition of 3 components | Served (M3), one prompt a process | paged-correct |
| [Qwen2.5-0.5B-Instruct FP16](#m2-fixtures) | fixture | Fixture | paged-correct |
| [Qwen2.5-0.5B-Instruct EXL3](#m2-fixtures) 4.0 and 4.5 bpw | fixtures | Fixture | paged-correct |

Nothing is distributed-correct: two-node execution is M4's.

## Research candidates

**EmbeddingGemma 2** (`google/embeddinggemma-2`) was added to the
[model-family research list](m35-families.md#owner-requested-research-addition-embeddinggemma-2)
at the owner's request on 2026-10-06. Google's release describes multimodal
embeddings; no jitLLM checkpoint/format, importer, execution, media, batching,
quality/performance or embedding API is qualified for this candidate. This
listing is separate from the supported summary above.

## Architecture foundations

Gemma 3 4B QAT Q4_0 has a separate checked profile and strict
GGML tensor binding in `model/gemma3.h`; the [foundation contract](gemma3.md)
records the approved checkpoint's complete actual descriptor table and
three passing focused CPU controls. Its separate bounded state/input and
descriptor graph/plan foundation passes 13 focused CPU/no-launch controls; importer,
runner, model execution, batching and serving qualification remain open. It is
not an execution support entry.

Gemma 4 26B-A4B and 31B have checked profiles, strict GGML tensor bindings,
bounded independent-slot state and segmented host-input descriptions in
`model/gemma4.h`. The [foundation contract](gemma4.md) records actual pinned
metadata, CPU/fake controls and required optimization/batching qualification.
Both approved artifacts have checked import, segmented graphs, a shared native
runner and a [bounded scalar serving route](experiments/gemma31-serving/README.md)
for chat/literal completions, likelihoods and up to twelve independent owners.
The route disables thinking and refuses generated tools and assistants/speculation.
The [bounded dense31 production bridge](experiments/gemma31-serving-bridge/README.md)
selects ordinary joined serving, both norm chains and eligible owner attention for
approved 31B artifacts with resolved context at most 8,192 and at most four slots.
Its uncalibrated prefill fallback is 256; smaller explicit overrides remain intact.
The [bounded Gemma26 recipe](experiments/gemma26-production/README.md) selects
joined serving, both norm chains, MoE route/reduce and owner attention for the
approved 26B-A4B artifact under the same bounds, with a 1024-row prefill fallback;
its C1/C4 paid cycles are within 0.6% of stock with stock's exact tokens, and its
1,024-row corpus heads are byte-identical.
Larger configurations retain the prior scalar recipe and 128-row cap. Current-pin C1/C4 8K continuations have zero predicted-ID differences and
128/129 and 512/516 byte-exact complete heads; the 1,024-row corpus has
complete head parity. Whole serving cycles were 3.15%/3.51% slower than the
reference at adoption and 0.73%/1.36% after [gap closing](experiments/gemma-gap-closing/README.md). Natural HTTP continuation, stop and departed-client peer
progress pass. These are bounded controls, not sustained performance, broad semantic
quality, assistant admission or long-context qualification.
The following earlier scalar/diagnostic screens retain their historical pins and failures.
The approved dense 31B artifact executes through the
same native runner with ordinary 1/2/4-request replay and exact checkpoint/spill
controls. Its [bounded representative screen](experiments/gemma31-runner/README.md)
fails against full-fusion llama.cpp (14.61% higher PPL, 330 strict argmax
differences); its unfused diagnostic matches all 1,024 complete heads exactly.
Its explicit default-off [checked norm chains](experiments/gemma-native-norm/README.md)
match all 1,024 stock heads at the same 128-row shape. Paid 8K heads against
the screened ubatch-256 reference differ, and the short speed result is
inconclusive. These scalar route controls do not establish assistant, optimized
batching or full reference qualification.
The separate default-off [joined serving diagnostic](experiments/gemma-joined-serving/README.md)
preserves same-policy solo heads/state at C1/2/4/8/12, but natural fixed-prefix
quality fails against actual multi-sequence stock batches. HTTP controls and
native speedups do not select optimized batching or change supported status.
The 26B
[representative likelihood screen](experiments/gemma-quality/README.md) fails
against fusion-enabled llama.cpp (10.03% higher PPL); its unfused diagnostic
control matches exactly. The [short resident screen](experiments/gemma-performance/README.md)
is 4.33% below the bookended reference rate. The [paid 8K screen](experiments/gemma-prefill/README.md)
records slower prefill (6.931 s versus 2.612 / 2.605 s), with faster native
fixed-prefix decode; known numerical differences keep qualification open.
The bounded dense31 norm result does not close the required full reference,
long-context, selected optimization or optimized-batching support gates.

The [current representative teacher-forcing screen](experiments/gemma-current-quality/README.md)
checks 31B both256 and 26B all1024 with plain norm fusion. All 1,024 complete
31B heads and 1,023 target likelihoods match fresh normal ring-cache stock
exactly. The 26B transfer fails strict quality with nine positive-margin choices,
15 exact heads and +0.0528% PPL. Each policy's own repeat is independently frozen;
these samples add no full-model, frontier-state or optimized-batching support.
A [single routing keep diagnostic](experiments/gemma-keep28-routing/README.md)
then resolves the fixed 26B corpus: keeping layer 28's routing probabilities
matches all 1,024 stock heads exactly. This explains the earlier nine misses
for that recipe; the general selection rule and production qualification remain open.
The [actual refusal observation](experiments/gemma26-routing-gate/README.md)
confirms a weights/logits allocation overlap at 1,024 rows; it supplies no
universal layer or token-count rule.

The [fresh fixed-capacity Gemma26 control](experiments/gemma-fresh-quality-validation/README.md)
freezes native scheduling variation on untouched history 13 and transfers its
unchanged p99 bound to independent history 14. All 17 heldout disagreements fit
the bound and PPL increases 0.1970%, passing the predeclared operational gates.
Strict zero-difference still fails. The measured runtime candidate remains
uncommitted; this bounded C1 result changes no default, admission, previous
failure or broader model-support gate.

The [opt-in owner-cohort engine](experiments/gemma-owner-cohorts/README.md)
adds funded complete C4/C8 attention quads and safe short-read eligibility with
full configured backing. Its 31B C8 factor matches every paid decode head;
Gemma26 passes its unchanged prior margin and conditional-score gates. These
bounded results leave serving defaults, C12, depth and full model support open. A fresh 31B C8 bookend on the same measured candidate is 3.57% slower than stock (155.385 ms), with complete outputs unchanged; this is a short recipe-qualified elapsed comparison.

The [ordinary whole-C12 factor](experiments/gemma-c12-single-wave/README.md)
shares twelve product columns and uses three funded four-root attention quads;
row-invariant waves keep eight. All 384 paid 31B heads match retained stock;
26B retains three positive-margin choice differences, all inside its unchanged
pre-oracle bound. Both 384-target conditional-loss gates pass. The original 8+4
quality failures remain recorded, and serving owner/joined defaults stay off.
These short controls do not close corpus, depth, live-serving or full model support.

The [small real-owner adapter](experiments/gemma-small-owner-attention/README.md)
adds opt-in wholeC2/C3 attention with checked active cache roots and original
whole-stream geometry, preserving C1 and larger quad/tail fallback behavior.
Heads16/32 primitive controls pass, and both approved C2 model screens pass strict
zero-margin choices and independent 64-target conditional-loss bounds. All 64
paid 31B heads match FIRST stock; 26B has 34/66 exact full heads. The 31B C3 short
screen passes strict choices, while C3 transfer and SOURCE14 serving defaults
remain unqualified. Short matched timings
are +1.47%31B with stock spread larger than the mean gap and −1.064%26B; no sustained
parity or full model-support claim follows.

The [equal-width partial adapter](experiments/gemma-partial-owner-attention/README.md)
adds opt-in whole5/6/7/9/10/11 geometry with bounded active roots and original
whole-grid fixups. N5/N6 primitive controls pass heads16/32 across all three
fixup cases. Gemma31 C5 recovers its original strict/loss failure: zero strict
choices, 160 paid byte-exact heads and −0.20935% 160-target conditional loss against
retained FIRST stock. Serving defaults remain off; unequal widths, other model
partial counts, Gemma26 transfer, full corpus, depth and sustained qualification
remain open.


The [fresh current solo comparison](experiments/gemma-current-reference/README.md)
records stable 26B prefill/decode latency gaps of 2.43%/0.92%. Native 31B prefill
is stable, but reference bookends vary from 12.3124 to 10.9769 s; native is 2.08%
slower than the closing reference and decode is 2.13% slower. These explicit
all1024/both256 research recipes preserve prior native heads/state; known
cross-engine head differences and the separate quality/batching gates remain.

The [independent-cache C4 diagnostic](experiments/gemma-owner-root-c4/README.md)
removes packed K/V copies for the closed 31B context-256 recipe. Paid latency
falls 8.93%; with plain norm fusion the fresh-reference gap is 2.10%, and all
128 complete heads match exactly. The [26B transfer](experiments/gemma26-owner-root-c4/README.md)
reduces latency 7.13% with exact native heads/state, leaving 0.265% latency
excess against fresh stock and the unchanged two positive-margin disagreements.
Wider contexts, quality qualification and production batching remain open.

The shared [planning read index](experiments/gemma-plan-index/README.md) removes
repeated graph scans without changing selected operations or kernel arithmetic.
With the existing default-off 26B `all` and 31B `both` policies, initial matched
ring-cache 8K screens recorded native prefill 15.10% / 13.83% slower and decode
0.61% / 2.00% slower, respectively. Native retained heads, initialized state and
all 32 choices remain unchanged. This planning improvement preserves the existing
policy defaults and does not establish full model or optimized-batching qualification.
Its [exact source-use extension](experiments/gemma-use-index/README.md) further
reduces fresh unchanged-control prefill by 2.80% / 2.31% for those same 26/31
policies. That screen recorded prefill gaps of 12.38% / 11.40% and decode gaps
of 0.55% / 2.32%, with prior native heads, initialized state and choices unchanged. Full qualification
and policy defaults remain unchanged.

Gemma's [state-only prefill](experiments/gemma-state-only-prefill/README.md)
now omits unused final-layer work on non-final, non-scoring prompt chunks in
both scalar serving profiles. Final heads and retained features remain full.
Ordinary and optional-policy controls preserve complete initialized state and
continuation. Measured all1024/both256 prefill improves 3.04%/2.42%; the 26B
reference movement remains unresolved, and quality/batching qualification is
unchanged.

Gemma's [bounded prefill lookahead](experiments/gemma-prefill-lookahead/README.md)
now overlaps next-chunk CPU graph/placement construction with current execution,
with funding, completion and abandoned-hint controls. Exact native heads/state
are preserved; same-binary all1024/both256 prefill improves 1.55%/2.36%. This
changes neither the optional arithmetic policies nor model qualification.

The [checked plain RMSNorm/Mul selector](experiments/gemma-state-only-norm-policy/README.md)
is now a serving default for both approved profiles. Ordinary off/on controls
preserve complete retained heads, initialized state and 32 choices; focused
default continuation and captured replay controls pass. Other experimental
arithmetic policies remain off, and model quality/batching qualification is
unchanged.

## Chat templates

A chat template renders natively when a native renderer is registered for
the SHA-256 of its exact UTF-8 bytes, or when a native family renderer
reproduces it on the probe corpus; any other template renders through the
bounded, sandboxed Jinja-subset interpreter (D-067 as amended 2026-10-02;
[tokenizer.md](tokenizer.md#chat-templates)). An LLM is refused when it
registers only if neither accepts its template, with the template's hash
in the error (`chat::ChatTemplate::ForText`). The registry is `kTemplates`
in [src/chat/chat.cc](../src/chat/chat.cc); its hashes agree with
[tokenizer.md](tokenizer.md#chat-templates) and with the unit tests
(`chat_test`, `chat_template_test`, and `tokenizer_models_test`, which
hashes the model files on a Spark).

| Model | SHA-256 the renderer is keyed on | The bytes hashed | Renderer, stop tokens |
| --- | --- | --- | --- |
| DeepSeek V4 Flash 0731 | `e643c31fcec17f342f72296e02c46d35846bf4c70f6a0271f23bad73fd4eb645` | The 0731 GGUF's `tokenizer.chat_template` (Unsloth's port of DeepSeek's `encoding_dsv4.py`), kept in the artifact's GGUF metadata | `deepseek-v4-flash-0731`; `<｜end▁of▁sentence｜>` |
| DeepSeek V4 community IQ2_XXS | `872492071c22c8d2025238120309ffbddddb666b49f4433f55c19b69bf51af27` | The community GGUF's `tokenizer.chat_template` ("chat-v2", 5,016 bytes), kept in artifact `cd39d504…`'s GGUF metadata | `deepseek-v4-flash-chat-v2`; `<｜end▁of▁sentence｜>` |
| Qwen3.8 Flash Next | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` | `chat_template.jinja` of `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6` (pinned in [pins.json](experiments/fast-swap/pins.json); the MLX baseline ships the same bytes) | `qwen3.8-flash-next`; `<\|im_end\|>`, `<\|endoftext\|>` |
| Qwen3.8 Unsloth GGUFs (Flash Next, 27B) | none pinned; `12827f24…` today | The GGUFs' `tokenizer.chat_template`, Unsloth's variant | `qwen3.8-flash-next-unsloth` by probe equivalence; `<\|im_end\|>`, `<\|endoftext\|>` |
| Qwen-Image-2.1 | none: the prompt is diffusers `8b3c707e`'s fixed text-to-image string, not a chat template | — | `RenderQwenImagePrompt`; drops the 14 system-turn tokens |

Templates with no native renderer, which render through the interpreter
where it accepts them: the older DeepSeek `e3aa0d6a` GGUF's (`d05566eb…`,
not checked), the FP16 fixture GGUF's (`d5495a1e…`, not checked) and the
EXL3 fixtures' (`cd8e9439…`, Qwen2.5's, which the template corpus checks
against transformers). Of the 29 corpus templates, 22 interpret
([chat-template-corpus](experiments/chat-template-corpus/README.md)).
Serving them still needs a runner for their architecture. The
checkpoint's `processor/chat_template.jinja` for Qwen-Image (`3636d0f0…`)
is pinned but not used. The options each renderer supports and refuses
are in [tokenizer.md](tokenizer.md#chat-templates).

## DeepSeek V4 Flash 0731

- **Architecture:** `deepseek4` (`model/dsv4.h`): hyper-connections,
  window, CSA and HCA attention with the lightning indexer, 256 routed
  experts, three hash-routed layers.
- **Checkpoint:** `unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93`,
  UD-Q2_K_XL, three shards
  ([pins.json](experiments/fast-swap/pins.json)).
- **Artifact:** v0 `8a355bfb…`, from the GGUF by `import_m3.py`
  ([dsv4-native](experiments/dsv4-native/README.md#what-runs)).
- **Components:** the target; optionally [DSpark](#dspark) as its drafter
  (the configuration's `drafter` key).
- **Template:** `e643c31f…` (above), read from the artifact.
- **Tokenizer:** byte-level BPE from the artifact's kept GGUF metadata
  (129,280 tokens, pre-tokenizer `joyai-llm`).
- **Decoding:**

  | Mode | Where |
  | --- | --- |
  | Greedy, plain | runtime (`--plain`) and harnesses |
  | Greedy, speculative with DSpark | runtime (the default with a drafter) and `jitllm_spec_runner` |
  | Seeded sampling, plain and speculative | `jitllm_spec_runner` and the runtime's chat/literal routes; harness histograms qualify the shared sampler, with serving branch/key/resume mapping reviewed ([mapping](experiments/qwen38-concurrent-oracle/README.md#sampling-evidence-applicability)); no new HTTP histogram measurement |
  | Exact (reference) mode, `--exact on` | harness only: llama.cpp's graph node for node, unfused, and D-092's row-invariant verify |

- **Prefill arithmetic:** the runtime serves the output-A/HCA prefill on
  every prefill chunk of 64 to 4,096 rows (both DeepSeek GGUFs), qualified
  against the 32K/128K oracle histories under the tie-aware greedy rule
  (D-085, 2026-10-03), 128K perplexity and the 127K answer task
  ([default-on acceptance](experiments/ds4-output-prefix/README.md#default-on-acceptance)).
- **Context:** exercised at 4,096 (against the oracle) and at 8,704, the
  runtime's original default (8,192 tokens of conversation in the swap
  table); since
  the long-context baseline, through the runtime at `context = 262144`
  with 8K to 128K prompts, and against llama.cpp b11254 at 32K
  ([long-context](experiments/long-context/README.md#phase-2-deepseek-flat-with-depth-2026-09-29):
  greedy within the near-tie bound, perplexity +0.1%, retrieval at 8K to
  128K, a long run repeating bit for bit). Its per-token cost is flat with
  depth but for the indexer (decode 21.9 / 21.2 / 20.5 tok/s at 8K / 64K /
  128K, 1.18–1.22× llama.cpp's; the fast plan's window cache a ring).
  The default is now 262,144. The configuration accepts 512 to the
  checkpoint's trained 1,048,576-token ceiling, checked before model
  allocation; growing state admits physical backing only as it is used,
  within the runtime's memory guard. The final HTTP path completes its
  32K–256K ladder and a 1,038,047-token prompt at 1M capacity with all
  512 outputs, plain and with DSpark
  ([final context study](experiments/m3-final-context/README.md)). Neutral
  completed-answer retrieval also passes 8K–256K in both modes and at
  1M capacity with DSpark; the saved maximum-context continuation remains
  pending. The chat route uses a progress watchdog
  ([deadlines](runtime-serving.md#progress-and-deadlines)). The
  minimum, 512, starts and serves (checked on `spark`, speculative), its
  prefill chunk capped at 384 rows by the 128-position window; the
  default chunk is 4,096 rows to a 262,144-token context, 2,048 above
  ([prefill chunks](runtime-serving.md#prefill-chunks-and-cancellation)).
- **Verified** (on `spark-b`):
  - Exact mode, resident: bit-identical to llama.cpp `b29c606e` unfused
    on the same GGUF, logits and perplexity
    ([dsv4-native](experiments/dsv4-native/README.md#results-spark-b-2026-09-28)).
  - The default fast plan against llama.cpp: greedy equal except
    near-ties, perplexity within 0.5%
    ([dsv4-decode](experiments/dsv4-decode/README.md#results)).
  - Paged, swapped and restored, bit-identical to the unswapped run, with
    decode graphs replayed across swaps
    ([swap](experiments/fast-swap/swap.md), [graphs](experiments/fast-swap/graphs.md)).
  - Native tokenizer and renderer: token for token with llama.cpp on the
    corpus and the chat fixtures ([tokenizer.md](tokenizer.md#agreement-with-the-references)).
  - Through the runtime: greedy tokens equal the harnesses', speculative
    and plain ([swap](experiments/fast-swap/swap.md#through-jitllm-runtime-d-096)).
  - Through the loopback chat route, first and after a swap back: the
    greedy reply equals `jitllm-runtime chat`'s on the same prompt, and a
    seeded sampled request repeats exactly (D-097,
    [runtime-serving.md](runtime-serving.md#the-chat-route)).
  - Speed headline: plain decode 1.07–1.09× llama.cpp's
    ([dsv4-decode](experiments/dsv4-decode/README.md#results)).
- **Known divergences:**
  - The fast plan is not bit-identical to llama.cpp: 12 of 256 greedy
    steps against the unfused arm and 6 against the fused arm are
    near-ties; the recorded near-tie bound (6.11) is too loose to be a
    test ([the bound, going forward](experiments/dsv4-decode/README.md#the-bound-going-forward)).
  - Step 93 of `capital`'s forced-rejection run: a 3.62-nat disagreement
    that passes only under 6.11, diagnosed as kernel noise amplified by
    near-tied routing, not a defect
    ([step 93](experiments/dsv4-decode/README.md#step-93-diagnosed)).
  - Step 249 of the 32K long-context prompt: phase 1's fast plan chose
    another token where the oracle prefers its own by 2.62 nats; every
    jitLLM path has the two within 0.9 nats there, and the phase 2 fast
    plan agrees with the oracle ([step 249](experiments/long-context/README.md#step-249)).
    The [frontier follow-up](experiments/dsv4-frontier-head/README.md#head-arithmetic-and-quality)
    reproduces the disagreement in a current wide-path control matching
    production's compact scheduling floor, with the unchanged 0.947
    bound. Compact-off repeats every logit bit exactly there. Selecting
    ordinary MMA for count-based HCA while retaining CSA/window sharing
    passes fresh 32K/128K controls at that bound (491+21 and 500+12
    equal/near-tie rows), with exact own repeats and common later rows.
    Matched 128K PPL is 1.926517 versus 1.9298. Fresh sampled checks pass
    the fixed TV bound; the final 32K–256K runtime ladder exceeds matched
    llama.cpp speed with memory within 1.025×. The final 1M runtime
    also exceeds both modes' reference speed, with memory within 1.012×.
    Neutral retrieval at 1M capacity passes; maximum-length saved-state
    continuation remains pending;
    historical phase-2 agreement is not a
    current-path blanket pass.
  - The reference mode (`--exact on`) keeps GGML's top-k and does not
    repeat past 4,096 positions (RE-031).
  - DeepSeek's own `tokenizer.json` differs from the GGUF's tokenizer on
    2 of 184 corpus items (Unicode 16.0 emoji); jitLLM follows llama.cpp
    and serves the GGUF's.

## DeepSeek V4 community IQ2_XXS

The [ds4 study](experiments/ds4-study/README.md) imports
`antirez/deepseek-v4-gguf@f71f23d552d664e523b422157b2befbf74040380`'s
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
as artifact `cd39d504…`. IQ2_XXS gate/up, Q2_K down and F16 compressor
APE tables retain their original bytes and precision. Plain greedy runs
in the resident native harness through 128,817 prompt tokens, with F16
caches. On the recorded 8K trajectory all 32 argmaxes agree with ds4,
including its F32-cache control, and native forced repeats are bit
identical. A 32K own-plan PPL control is within the existing 3% gate.
On prefill chunks of 64 rows or more the fast plan's default D2R down
product changes its logits (heads move up to 3.87 on partial chunks); its
32K perplexity is within 0.04% of ds4's in 4,096-row chunks, and +0.53%
over the mechanisms-off control with unaligned partial chunks
([stage mechanisms](experiments/ds4-prefill-stages/README.md)).
ds4's default caches use FP8/FP4, so same weights alone do not establish
equivalent cache policies. This variant has no swap/restore or
seeded-sampling evidence in this study.

- **Template:** `87249207…` ("chat-v2", above), read from the artifact;
  its renderer differs from the 0731 template's where the templates do
  ([tokenizer.md](tokenizer.md#chat-templates)). The tokenizer the
  artifact keeps equals the 0731 GGUF's (`tokenizer_models_test`).
- **Through the runtime** (on `spark`, 2026-10-02): `[models.deepseek]`
  with this artifact, `speculation = false`, context 16,384, 2,048-row
  chunks, registers, and one greedy `/v1/chat/completions` turn on
  loopback (system "Be brief.", a question asking for a one-word answer)
  returned reasoning and the answer "Paris", ending at the stop token, 19
  prompt tokens.
  One turn only: no quality, continuation or swap claim.
- **Waves and the fused decode form:** its F16 HC mixing weights now take
  the fast plan's fused form (`jitllm.dsv4.hc_mix` widens them to F32
  exactly), so it starts with its request slots (four by default) and decodes in waves
  (between `64faee6` and this fix the runtime refused to start it). Wave
  controls (`--check wave`, 2 and 4 slots, plain): 142/142 and 332/332
  rows byte-identical to each slot alone. Its single-request decode now
  takes the fused form as well; on the forced 8K ds4 trajectory all 32
  argmaxes still agree with ds4, the prefill row is unchanged, the decode
  rows move by at most 5.45 (RMS 0.29) from the unfused form's, and a
  forced repeat is byte-identical. Against ds4's own decode logits the
  fused form's mean RMS is 0.450 (unfused 0.444), its largest difference
  5.86 (unfused 7.35). Timing and HTTP cells:
  [deepseek-batching](experiments/deepseek-batching/README.md#community-artifact-in-waves).
- **DSpark** (2026-10-03): the 0731 drafter runs with this artifact. The
  verify's device lookup of the drafts' rows now takes its F16 token table
  and equals the host's for every token. Greedy speculation has no
  near-tie violations on the eight `--check greedy` prompts (acceptance
  0.48–0.82). Forced rejections leave 0 stale bytes, and DSpark waves at
  2 and 4 slots equal each slot alone. Through the runtime it leads plain
  decode alone (7K C1 +31%, 124-token +54%). At C4 DSpark waves trailed
  plain ones (−6% / −12%). Each wave now chooses DSpark or plain decode
  from counted acceptance against a measured per-width cost, which puts
  C4 1–4% under plain on both artifacts (2–4% once wave lanes sped
  plain waves up more than DSpark's)
  ([deepseek-batching](experiments/deepseek-batching/README.md#adaptive-dspark-and-plain-waves)).

## DSpark

- **Architecture:** `dflash` (`model/dspark.h`): 3 window-only DeepSeek
  V4 blocks, MXFP4 experts, a Markov head; drafts 3 tokens a step.
- **Checkpoint:** `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` in the same
  pinned repository and revision.
- **Artifact:** v0 `dd2d3f9c…`, its own artifact; it binds the target's
  token table and head at load (no composition document, D-089's note).
- **Components:** drafter only; runs with DeepSeek V4 Flash 0731.
- **Template and tokenizer:** its target's.
- **Decoding:** as its target's speculative rows above.
- **Verified:** greedy speculation equal to plain greedy (bit for bit in
  exact mode; near-ties on the fast plan); forced rejections leave no
  stale state; rollback across a swap; sampled speculation within its
  total-variation bound ([dspark](experiments/dspark/README.md#correctness),
  [dsv4-decode](experiments/dsv4-decode/README.md#results)). Speed
  headline: 1.03× / 1.07–1.08× llama.cpp's DSpark decode on `prose` /
  `code`.
- **Known divergences:** step 93 (above). A new weight type needs its row
  kernel before exact-mode speculation runs on it (D-092).

## Qwen3.8 Flash Next

- **Architecture:** `qwen4exp` (`model/qwen38.h`): hyper-connections, the
  n-gram (PLE) layer, Gated DeltaNet, QSA attention with its indexer, 512
  routed experts top-10 plus a shared expert.
- **Checkpoint:** `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6` (ModelOpt
  NVFP4 experts, MXFP8 linears; [pins.json](experiments/fast-swap/pins.json)).
- **Artifact:** v0 `c4fb47a9…`, routed experts in the CUTLASS layout
  (the earlier GGML-layout `67617f87…` remains for harness options that
  read that layout)
  ([qwen38-native](experiments/qwen38-native/README.md#what-runs)). The
  28.8 GB n-gram table is read by rows from the SSD (D-035).
- **Components:** the target; optionally [its MTP drafter](#qwen38-mtp).
- **Template:** `c3cf9e34…` (above). The artifact keeps neither template
  nor tokenizer, so the configuration names the checkpoint's
  `chat_template.jinja` and `tokenizer.json`.
- **Tokenizer:** byte-level BPE from `tokenizer.json` (NFC, the `qwen35`
  pre-tokenizer).
- **Decoding:**

  | Mode | Where |
  | --- | --- |
  | Greedy, plain | runtime (`--plain`) and harnesses |
  | Greedy, speculative with MTP (adaptive depth 2–3; fallback cap 65,536, selected artifacts capped at their physical rows) | runtime (the default with a drafter) and `jitllm_qwen38_spec --draft 3 --adaptive-depth on` |
  | Seeded sampling, plain and speculative | `jitllm_qwen38_spec` and the runtime's chat/literal routes; shared sampler and branch/key/resume mapping reviewed ([mapping](experiments/qwen38-concurrent-oracle/README.md#sampling-evidence-applicability)); no new HTTP histogram measurement |
  | Exact (reference) form, `--exact` | harness only (`jitllm_qwen38_exec`); speculation has no exact mode |

- **Context:** exercised to 8,704 (8,192-token prefill and the swap
  table's 8K conversation); since the long-context baseline, through the
  runtime to its configured maximum, 262,144 (a 258,633-token prompt,
  prefilled in 2,040-row chunks: RE-037), and against Mia's vLLM at 32K
  and 128K ([long-context](experiments/long-context/README.md): greedy
  within the near-tie bound, perplexity −2.1% / −1.4%, retrieval at every
  rung to 256K). Since long context's phase 2 its per-token cost is flat
  with depth to within the indexer's scoring and selection (block keys
  cached, the selection on the device at any depth, attention over the
  2,051 kept cells alone): through the runtime prefill 2,314 / 2,359 /
  2,178 tok/s and plain decode 26.8 / 26.0 / 24.4 tok/s at 8K / 64K / 256K,
  1.2–1.4× and 1.03–1.08× Mia's vLLM from 32K on; bit-for-bit repeatable at
  64K and 128K; speculation with its MTP drafter to the configured
  262,144 ([long-context](experiments/long-context/README.md#phase-2-qwen38-flash-next-flat-with-depth)).
  The default and trained ceiling are both 262,144; registration refuses a
  higher value before model allocation. The minimum, 512, starts and serves
  (checked on `spark`, with MTP: 510 tokens usable), its prefill chunk
  504 rows; the default chunk is 4,096 rows at every context
  ([prefill chunks](runtime-serving.md#prefill-chunks-and-cancellation)).
- **Verified:**
  - Current four-slot NVFP4/MXFP8 waves on one frozen 32K oracle history:
    all eight 512-row cells pass the unchanged near-tie rule (477 agreements,
    35 near-ties, zero outside). Complete rows and initialized target/MTP
    state match across slots and repeats; actual selected head 47,172 rows,
    shared draft depth two and 12-row verifies plus a four-row tail
    ([concurrent control](experiments/qwen38-concurrent-oracle/README.md)).
    This forced-conditioning check supplies no natural acceptance trajectory
    or unrestricted quality claim for other histories/widths.
  - Against Mia's vLLM (the same checkpoint, deterministic mode, MTP
    off): greedy equal except near-ties on 191 of 192 steps, perplexity
    −0.8 to −1.2%; state spill and restore bit-identical
    ([qwen38-native](experiments/qwen38-native/README.md#results-second-pass)).
  - Paged (n-gram rows from the SSD), swapped and restored bit-identical;
    decode graphs replayed across swaps
    ([swap](experiments/fast-swap/swap.md#qwen38-flash-next-on-the-paged-node),
    [qwen38-mtp](experiments/qwen38-mtp/README.md#correctness)).
  - Native tokenizer and renderer: token for token with Hugging Face
    tokenizers ([tokenizer.md](tokenizer.md#agreement-with-the-references)).
  - Through the runtime: greedy tokens equal the harnesses', speculative
    and plain ([swap](experiments/fast-swap/swap.md#through-jitllm-runtime-d-096)).
  - Through the loopback chat route, streamed after a swap: the greedy
    reply equals `jitllm-runtime chat`'s on the same prompt (D-097,
    [runtime-serving.md](runtime-serving.md#the-chat-route)).
  - Speed headline: prefill 1.38–1.41× Mia's vLLM at 8K; plain decode
    1.01–1.03× with speculation off on both sides, then 5.4–8.1% faster
    with TensorFold's techniques (against jitLLM's previous build, same
    session; [tensorfold-techniques](experiments/tensorfold-techniques/README.md#adopted)).
- **Known divergences:**
  - `french` step 3: one greedy step of 192 outside the near-tie bound
    (oracle margin 2.0, bound 1.0) on the default fast form; accepted by
    the owner (2026-09-28) as a known divergence
    ([qwen38-native](experiments/qwen38-native/README.md#results-second-pass)).
  - The near-tie bound was set after the first comparison with the
    oracle, not pre-registered.
  - The reference and unfused graphs' QSA top-k is not repeatable past
    2,051 attended cells (RE-031), and their chunks stay bounded by
    RE-037; the default fast form breaks ties by cell and repeats at any
    depth.
  - The NVFP4 `tokenizer.json` normalizes to NFC, llama.cpp's Qwen3.8 GGUF
    does not (6 corpus items differ); jitLLM follows the NVFP4 file, as
    vLLM does.
  - The CUTLASS grouped GEMM and MXFP8 GEMM are built for `sm_121a` only:
    elsewhere this artifact's prefill is refused.

## Qwen3.8 Flash Next GGUF

- **Architecture:** `qwen4exp` (`model/qwen38.h` `Qwen38Format::kGguf`),
  the same model as [above](#qwen38-flash-next) from a GGUF checkpoint.
- **Checkpoint:** `unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee`, UD-IQ3_XXS
  (pins.json id `qwen3.8-flash-next-gguf-ud-iq3xxs`): IQ2_S gate/up
  experts (IQ3_S on one layer), IQ4_NL down experts and n-gram table,
  Q6_K and Q8_0 matrices, BF16 indexer projections, F32 norms and routers.
- **Artifact:** v0 `5356b5b0…`, imported verbatim by `import_m3.py build`
  ([artifact-format](artifact-format.md#qwen38-flash-next-gguf)); the
  n-gram hash and the hyperparameters come from its kept GGUF metadata,
  checked against the profile.
- **Other quantizations:** the binding takes any GGML type; the graph
  refuses one this build's products do not take. Matrix products cover
  Q8_0, Q4_0, Q2_0, the K-quants Q2_K–Q6_K, IQ1_S, IQ2_XXS/XS/S,
  IQ3_XXS/S, IQ4_NL/XS, MXFP4 and NVFP4 (MMVQ and MMQ); the n-gram table
  Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 or IQ4_NL. Only UD-IQ3_XXS has been run.
  IQ1_M (in unsloth's UD-IQ1 builds) has no tile kernel upstream, so a
  checkpoint with IQ1_M matrices or experts is refused when it is set up;
  ISTA-DASLab's GSQ-RCO Q2_0 builds are untested.
- **Components:** the target alone; speculation (its MTP GGUF) is not
  built.
- **Template and tokenizer:** the configuration names the NVFP4
  checkpoint's `tokenizer.json` and `chat_template.jinja` (the same
  vocabulary; the GGUF's own template is not registered).
- **Decoding:** plain only; greedy checked (the runtime's sampler is the
  NVFP4 model's, not checked on this artifact).
- **Verified** (on `spark-b`,
  [qwen38-gguf](experiments/qwen38-gguf/README.md)): against llama.cpp
  b11254 on the same GGUF, teacher-forced greedy 188 of 192 steps equal
  and the rest near-ties, perplexity −0.23% from the unfused arm's, the
  resident expert layout bit for bit; served through `jitllm-runtime`
  (n-gram rows from the SSD).
- **Speed headline** (C1, through the runtime and `llama-server`, the same
  requests): 8K prefill 1,118 tok/s against 676 (1.65×; `llama-bench`
  768), decode 33.8 against 31.2 tok/s (1.08×); first token after start
  6.3 s against 68.7 s; peak memory 56.9 against 80.1 GiB.

## Qwen3.8 MTP

- **Architecture:** `qwen4exp-mtp`: the checkpoint's one hybrid
  full-attention MTP layer.
- **Checkpoint:** the same pinned checkpoint (its last shard and
  `config.json`).
- **Artifact:** v0 `056a750e…`, its own artifact; binds the target's token
  table and head at load. An optional imported selected head stores BF16
  `draft_output.weight` and strictly ascending original token IDs in
  `draft_output.ids`; neither changes the target artifact. No vocabulary
  list ships with jitLLM. See the [draft-head study](experiments/qwen38-draft-head/README.md).
- **Depth:** greedy chooses between two and three passes from observed
  acceptance and a calibrated step-cost ratio, with bounded exploration.
  The policy is conversation state: restore preserves its schedule, and
  wall time never affects it. Seeded sampling retains depth two. A
  three-row prefill chunk retains the earlier depth-two path.
- **Components:** drafter only; runs with Qwen3.8 Flash Next.
- **Template and tokenizer:** its target's.
- **Verified:** greedy speculation equal to plain greedy except near-ties
  (0 violations); forced rejections at depths 2 and 3 equal to their
  control state for state; rollback across a swap; sampled speculation
  within its bound ([qwen38-mtp](experiments/qwen38-mtp/README.md#correctness)).
  Speed headline: 1.12× / 1.03× Mia's MTP-3 decode on `prose` / `code`;
  with TensorFold's techniques the rate moved with acceptance (`prose`
  −2.5%, `code` +6%, [tensorfold-techniques](experiments/tensorfold-techniques/README.md#speculation-the-adaptive-window)).
- **Context:** to its target's configured maximum, 262,144 (registered and
  run with 128K and 256K prompts; forced rejections and rollback across a
  swap checked at 64K). Until long context's phase 2 it was refused above
  32,768 (its selection)
  ([long-context](experiments/long-context/README.md#phase-2-qwen38-flash-next-flat-with-depth)).
- **Known divergences:** the verify is batched, not row-invariant, so
  speculation has no bit-exact mode; its own noise (p99 up to 2.37 on the
  forced run, 4.28 after TensorFold's techniques, whose plain decode and
  verify take their hyper-connection products from different kernels) was
  measured after the comparison, and the rows it moves by more than 2 are
  not diagnosed.

## Qwen-Image-2.1

- **Architecture:** composition `QwenImage21Pipeline` (`model/qwen_image.h`)
  of a `qwen3_vl` text encoder (text path only), the
  `QwenImage21Transformer2DModel` DiT and the `AutoencoderKLQwenImage21`
  VAE decoder; a 40-step flow-matching Euler loop.
- **Checkpoint:** `Qwen/Qwen-Image-2.1@790c9263`, BF16
  ([pins.json](experiments/fast-swap/pins.json)).
- **Artifacts:** composition `eca21baa…` (D-089) naming text encoder
  `ed89ed27…`, denoiser `d1184efd…` and VAE `44c1a20a…`
  ([qwen-image-native](experiments/qwen-image-native/README.md#what-runs)).
- **Template:** none; the fixed prompt string (above).
- **Tokenizer:** byte-level BPE from the composition's `tokenizer.json`
  (the processor's, Qwen3-VL-8B; NFC, the `qwen2` pre-tokenizer).
- **Generation:** text to image only; no guidance, no condition images.
  The initial latents come from a file (diffusers' for its seed); there is
  no native seeded generator yet. The runtime serves one prompt a
  process, fixed at setup.
- **Verified:** one prompt, 1024², 40 steps, seed 42, against diffusers
  `8b3c707e` BF16: tokens exact, every component and the image within its
  pre-registered bounds; the paged node's and the runtime's pixels equal
  the harness's, before and after swaps
  ([qwen-image-native](experiments/qwen-image-native/README.md#results-spark-2026-09-28),
  [swap](experiments/fast-swap/swap.md#qwen-image-21-on-the-paged-node));
  since the speed slice (same BF16 numerics; image 41.77 dB PSNR, SSIM
  0.996, pixels `3b7770ca…`) also its pixels, repeatable run to run
  ([speed](experiments/qwen-image-native/README.md#speed)).
  Speed headline: full generation 0.64× diffusers' time, weights
  resident. Other sizes, step counts, prompts and seeds: not verified; the
  products' pinned algorithms are the GB10's at this prompt's shapes.
- **Known divergences:** jitLLM rounds differently from diffusers'
  BF16 (bounded, above). The operations run through a plan bound against
  the implementation registry (D-053, `kernels/image/pipeline.h`).

## M2 fixtures

Small dense models from the backend proof, kept as fixtures. They run
only in the backend-proof harnesses (`jitllm_fp16_exec`,
`jitllm_fp16_paged`, `jitllm_exl3_paged` and others), on recorded token
IDs, and are never served. M5 serves them end to end and completes their
rows ([plan](plan.md)).

| | Qwen2.5-0.5B-Instruct FP16 | Qwen2.5-0.5B-Instruct EXL3 4.0 / 4.5 bpw |
| --- | --- | --- |
| Architecture | `qwen2` (`model/qwen2.h`), GGML FP16 | `qwen2` with EXL3 linears (`model/qwen2_exl3.h`) |
| Checkpoint | `Qwen/Qwen2.5-0.5B-Instruct-GGUF@9217f5db` `qwen2.5-0.5b-instruct-fp16.gguf` ([pins](experiments/first-slice/pins.json)) | `blockblockblock/Qwen2.5-0.5B-Instruct-exl3-4.0bpw@7009334d` and `-4.5bpw@030d3a41` ([pins](experiments/exl3-reference/pins.json)) |
| Artifact | v0 `b93cdc32…` | v0 `6e96e499…` / `00d77caf…` ([artifact-layout](experiments/artifact-layout/README.md)) |
| Template hash (no renderer) | `d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4`, the GGUF's `tokenizer.chat_template` ([first-slice](first-slice.md)) | `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`, `tokenizer_config.json`'s `chat_template`, both rates ([runtime.json](experiments/exl3-reference/runtime.json)) |
| Tokenizer | the GGUF's Qwen2 BPE; the native tokenizer not verified on it | `tokenizer.json`; the native tokenizer not verified on it |
| Decoding | teacher-forced trajectories only | teacher-forced trajectories only |
| Context | short trajectories; the GGUF declares 8,192 | short trajectories; the config declares 32,768 |
| Verified | bit-identical to the FP16 bridge (llama.cpp's GGML rebuilt with jitLLM's SDK) on all four arms, resident and paged, evicted, restored and relocated ([P2](experiments/backend-proof-p2/README.md), [aggregate](experiments/backend-proof/README.md)) | every linear byte-equal to ExLlamaV3 `6b84a21b` at a forced plan; end to end within Tier C's bounds; paged and restored bit-identical ([P3](experiments/backend-proof-p3/README.md)) |
| Known divergences | the GGUF's template differs from the base checkpoint's, and its context from the base's 32,768 | ExLlamaV3's autotuner is part of its numerical plan; the oracle runs a frozen tuning cache ([P0](experiments/backend-proof-p0/README.md)) |

Resident speed against each fixture's reference (BP-F3) is M5's, under
D-085; not measured.

## Pinned, not run by jitLLM

Comparators only, for speed and swap time
([baselines](experiments/fast-swap/baselines.md)): TensorFold's
`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa80` and llama.cpp's
Qwen3.8 UD-IQ3_XXS GGUF (both cross-quantization).
