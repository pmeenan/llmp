<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded dense Q8_0 cache opt-in

The shared dense-attention component now consumes existing quantized cache stores
and vector attention at D64/D256. F16 remains the default. The explicit per-alias
setting changes cache state, not weights, and does not widen serving admission.
This first screen measures incremental loss against the same checkpoint with
F16 cache. It establishes functional/state support, not an acceptable-loss
threshold, family quality approval, long-context behavior or a speed gain.
The sections through "Provenance" describe that first bounded slice;
[K1](#long-context-q8_0-vs-f16-quality-across-migrated-llms-k1) below extends
the opt-in to every migrated LLM and measures it at long context.

## Paired short teacher-forced quality

Spark A; fixed natural text/IDs selected before losses, plain teacher histories,
full finite vocabulary rows, all next-token targets scored. Each pair uses the
same artifact, histories and source policy; only the cache-encoding selection
changes. Its Q8 attention path also quantizes
queries to Q8_1 inside the existing kernel. The measured delta therefore covers
the selected Q8 path, not an isolated storage-only error.

| Model / cache head | Targets | F16 mean NLL / PPL | Q8_0 mean NLL / PPL | PPL change |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B FP16 artifact / D64 | 255 | 4.306976343547 / 74.21574642 | 4.351930424557 / 77.62817371 | +4.597983% |
| Gemma2 2B Q8_0 artifact / D256 | 91 | 5.974900297411 / 393.42887326 | 5.989320294526 / 399.14321787 | +1.452447% |

Both samples show higher loss with Q8 cache. They are short single-passage
observations, not a general quality estimate or a predeclared acceptance gate.
The public option remains explicit; automatic calibration does not choose it.
Original unquantized evidence is not required for this same-weight cache
comparison and no closer-to-original claim is made. The Qwen FP16 artifact's
conversion lineage remains separate from this paired comparison.

## State and memory scope

Logical Q8 cache bytes are 34/64 of F16. Qwen reports layout sizes
6,291,456→3,342,336 bytes. These are state-layout extents, not measured total
process or device occupancy. Gemma's logical state is
54,525,952→28,966,912 bytes, but both actual reserved layouts are 109,051,904
bytes because 52 banks round to 2 MiB each. No physical memory reduction is
claimed for that short Gemma context 512 setup.

The Qwen 76-position same-Q8 control compares 252 complete replay rows across
four evaluations, with checkpoint/spill copies of 3,342,336 bytes, two full-weight
evictions and 980 reloaded extents. The actual selected corpus has 24 quantized
attention operations, and the two replay shapes have 24 each. No cross-encoding
head-bit equality is required.

Gemma uses the normal Server alias and actual native tokenizer. The Q8 control
compares complete initialized 36-position state after checkpoint reconstruction,
spill and fresh-server kept restoration, plus the following 37th-position full
head after spill/kept restoration. There is no independently saved 37-position
state oracle. Its context 512 test does not wrap the 4096-window ring; odd-cell,
blocked geometry, ring/truncate and unsupported snapshot alignment have host
controls. At this slice D512, specialized QSA/MQA, sinks, borrowed state and
drafters remained outside the opt-in domain; K1 below covers them.

## Provenance and bounded execution

Main native build plus 519 host cases across 27 targets pass. The earlier
140-host proof remains applicable; the fixture-only compile and main build
used identical model binaries.
The four actual cases complete with no failures/skips, complete process-group
retirement, zero remaining GPU processes and zero memory-limit/OOM events.
The process/cache peak 480,501,760 bytes is not GPU or whole-model occupancy.
No timing comparison was run. Source/metadata/binary guards pass before and
after execution. Source quality checks are recorded by the final main carry.

Raw receipts stay in coordination scratch: `q8-state-main01/result.json`
SHA `93c46d6b1d3c589260b7e39e948d70428dee2f4e7928a72dd7ff76a20e3947df`
and `q8-state-models02/actual/result.json`
SHA `dd1ed6f9837c68e413a9ee38cb740523f46616ab02df83bfddc93dafaa146d93`.
The Qwen binary SHA is `acc3da660d89c60d70dc48a1fdbc3fef54b3262eafe5ef8f1abeeea2691b80dc`;
Gemma is `4d2d8d2a3fe84f66b54f35ec4ab7ae5ce5284f95242532cd940692cd1e7e3d61`.
Source base 6e939ff1 plus the reviewed 19-path slice; no other arithmetic changes.

Qwen artifact `b93cdc326ba4f4c1c71da613503ecd848cd2a0caf122214f26e04588a12a9073`;
256-ID input SHA `1ca6f2d78042d95978dd4ab34bc95b85dae14ab2c8ee454c8facdf51cac7d423`
reuses the authenticated natural corpus document 0 from the attention qualification.
Gemma artifact `eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870`;
literal UTF8 text SHA `010a99529ef187e6c90769840869608e57c33ccc9737e324faf7d87e299d2f29`,
actual 92 native IDs SHA `dc0714ab2ad1ef7af6d6ceba7a648744337ce0ac85a4b30ede6507fbe68ddfaf`.
The fixed literal remains in the model test; neither losses nor observed outputs
selected the text or cutoff. Full runtime/payment proof is retained in the
reviewed source, D64 applicability, D256 fixture and models02 packages.

## Long-context Q8_0 vs F16 quality across migrated LLMs (K1)

Rerun 2026-10-15 on Spark A (`spark-c4e2`, driver 580.178.04) with
`tools/k1_quality.py` and a runtime built natively there from `505df41a`
plus the prefill fix below (`llmp-runtime` SHA-256
`c9233e8cb5e62e5b653b512462831e091e2c51521dbee0062d2b9bae4fe50da2`). The
corpus is `~/.local/share/llmp/corpus/m3-reference-corpus.txt` (SHA-256
`55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb`), the
project's own markdown. These rows replace the first K1 run's
(`78cd2f8a`), whose Gemma 4 rows were meaningless
([below](#why-the-first-gemma-4-rows-were-withdrawn)) and whose contexts were
shorter.

Each arm is a fresh runtime process with the same alias settings except
`kv_cache_encoding` (and the first run's diagnostic switches
`LLMP_DIAGNOSTIC_COMPONENT_ENGINE=1`, `..._NODE_CAP_BYTES=120259084288`,
`..._STARTUP_HOST_BYTES=auto`). It scores one passage teacher-forced
through `/v1/completions` (`prompt_logprobs = 0`, full-vocabulary rows, every
target), then streams one 64-token greedy chat reply. The passage starts at a
fixed corpus line (index 100003; the start no longer depends on its length,
so a shorter passage is a prefix of a longer one) and is cut to the family's
token budget by the server's own token counts; both arms score the same
passage and the same IDs (checked). Gemma 4 scores it as the model's own
turn of its chat format, counting only the passage's rows; every other
family scores it as bare text after BOS. Both arms of a family share the
artifact, settings and histories; only the cache encoding changes, and the
Q8 path also quantizes queries to Q8_1 inside the existing kernels, so a
delta covers the whole selected Q8 path. Fixed thresholds were not set:
this reports the measured change, it is not an acceptance gate.

| Family (alias) | Context (slots) | Scored tokens | F16 PPL | Q8_0 PPL | PPL change | First 1K | Last 2K | Mean \|Δ logprob\| | Greedy chat F16 vs Q8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Gemma 2 2B Q8_0 | 8,192 (2) | 7,901 | 29.3847 | 29.3921 | +0.025% | −0.113% | +0.083% | 0.032 | same 64 tokens |
| Gemma 3 4B QAT Q4_0 | 8,448 (1) | 7,903 | 28.6014 | 28.6186 | +0.060% | +1.145% | −0.105% | 0.084 | differs |
| Gemma 4 26B-A4B UD-Q4_K_M (turn format) | 8,192 (4) | 7,903 | 30.8048 | 30.6350 | −0.551% | −0.796% | −0.181% | 0.110 | differs |
| Gemma 4 31B UD-Q4_K_XL (turn format) | 8,192 (4) | 5,977 | 38.4885 | 38.6885 | +0.520% | +1.034% | +0.759% | 0.068 | same 64 tokens |
| Qwen3.8 GGUF | 33,792 (4) | 15,834 | 9.9186 | 9.9164 | −0.022% | +0.254% | −0.422% | 0.171 | differs |
| Qwen3.8 NVFP4/MXFP8 + MTP | 33,792 (4) | 13,832 | 9.6584 | 9.6696 | +0.116% | +0.162% | +0.876% | 0.122 | differs (MTP: 22 and 25 chunks) |
| DeepSeek V4 Flash + DSpark | 262,144 (4) | 9,891 | 13.0489 | 13.0485 | −0.003% | −0.202% | −0.393% | 0.085 | differs (DSpark: 23 chunks each) |

Whole-passage perplexity changes by −0.55% to +0.52%, the first 1K and last
2K tokens by at most 1.15%, and no 2,048-token depth bucket by more than
0.017 nats (1.7%, Qwen3.8 GGUF at 12K–14K; it is −0.017, an improvement). The
changes go both ways and do not grow with depth, so on these single passages
Q8_0 cache costs no measurable long-context quality, while individual
tokens' log-probabilities move by 0.03–0.17 nats on average. These are one
passage per family, not a general quality estimate.

**Assistant and drafter configurations.** Scoring never consults a drafter:
with the Gemma 4 assistant attached, the first 1,007 passage rows score
bit-identically to the target-only rows above in both arms (26B and 31B).
The assistants therefore have no separate perplexity; what Q8 changes for
them is the verify path. With Q8 KV under the target and borrowed by the
assistant, the 64-token greedy chat replies are the same as with F16 for
both sizes (26B: 24 and 23 streamed chunks; 31B: 24 and 24), so speculative
decoding runs and accepts on the quantized cache. Qwen3.8's MTP drafter and
DeepSeek's DSpark run likewise in the chat check of their rows (streamed chunks
for 64 tokens: Qwen3.8 22 with F16 and 25 with Q8_0, DeepSeek 23 and 23).

### Why the first Gemma 4 rows were withdrawn

The first K1 run gave Gemma 4 F16 perplexities of 17,494 (26B) and 18,521
(31B) against 31 and 60 for Gemma 2 and 3. That was not a scoring bug; the
instruct checkpoints are not raw-text language models
([RE-056](../../rough-edges.md#re-056-gemma-4-instruct-checkpoints-are-not-raw-text-language-models--2026-10-15-status-worked-around)):

- llama.cpp v0.6.0 (`d812350`, `llama-perplexity -c 3072`, its own
  tokenizer and BOS) gives 18,990 (26B) and 25,131 (31B) on the same 26,000
  characters, and 29.79 for Gemma 3 4B QAT.
- Llmpalooza's tokenization of that text equals llama.cpp's for all 6,956
  IDs. Scoring llama-perplexity's own two 3,072-token windows by those IDs,
  llmpalooza gives Gemma 3 28.08 and 29.75 (llama.cpp 28.19 and 29.79) and
  Gemma 4 26B 22,987 and 21,232 (llama.cpp 20,626 and 18,990). Llmpalooza
  scores one row at a time and llama.cpp in batches; in Gemma 4's degenerate
  raw-text regime that arithmetic difference moves NLL by about 0.1 nats.
- The bare-text 26B predicts badly (top-1 accuracy 18%, and `-` is its
  first choice on a third of the rows). Scored as its own turn
  (`<|turn>user\nWrite the next part of the project documentation.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`
  before the same passage, only the passage's rows counted), it gives
  perplexity 31.12 and top-1 accuracy 45.7%; the frozen M3 runtime
  (`1facc957-fixed`, the legacy Gemma 4 runner) gives the same NLL to the
  last printed digit (3.43794148119958).

The withdrawn rows' small Q8 deltas (−2.9% and +4.6%) measured noise in that
regime. The assistant rows were the target rows again (scoring does not use
the drafter), not separate measurements.

### Context lengths

Each family's alias context is its configured serving context; the scored
passage is what one-row teacher forcing completes within about ten minutes
an arm (D-085), which is the binding cost: the route scores one row per step
(Gemma 2/3 about 15 ms a row, Gemma 4 26B 21 ms, 31B 100 ms, Qwen3.8 about
34–39 ms, DeepSeek 49–58 ms).

| Family | Model / approved context | Scored here | First K1 run |
| --- | --- | ---: | ---: |
| Gemma 2 2B | 8,192 (model maximum) | 7,901 | 6,923 |
| Gemma 3 4B QAT | 8,448 with one slot (approved; 128K trained, unqualified) | 7,903 | 2,407 |
| Gemma 4 26B-A4B | 8,192 optimized recipe, scalar route beyond | 7,903 | 6,955 (raw text) |
| Gemma 4 31B | 8,192 optimized recipe, scalar route beyond | 5,977 | 6,955 (raw text) |
| Qwen3.8 GGUF | 262,144 | 15,834 | 7,447 |
| Qwen3.8 NVFP4/MXFP8 | 262,144 | 13,832 | 7,447 |
| DeepSeek V4 Flash | 262,144 (1M measured) | 9,891 | 7,136 |

Gemma 2, Gemma 3 and the 26B fill their approved contexts to within 4%;
the 31B's 5,977 (73%) keeps an arm within ten minutes. Every row is far past the Gemma
sliding windows (4,096 for Gemma 2, 1,024 for Gemma 3 and 4), so the global
layers read thousands of quantized cells. For Qwen3.8 and DeepSeek the
scored lengths are well short of their 262K contexts. Scoring 32K or more
there costs 20–30 minutes an arm at one row per step; a long-context
claim at their full context needs a scoring path that takes the rows'
heads from prefill chunks (the one-row route was chosen for exact
agreement with decode) rather than more hours of one-row runs. It remains open.

### Q8_0 coverage by layer and stage

`kv_cache_encoding = "q8_0"` sets `kv_encoding` on every owned attention
component of every stage (`components/state_encoding.cc`:
`attention.dense`, `.borrowed`, `.qsa`, `.mqa`; target and drafter alike).
Graph planning sends every flash attention whose K or V is Q8_0 to the
quantized vector kernel (`ggml.flash_attn_ext.vec_q8_0`, compiled at D64,
D256 and D512 for Q8_0), at any row count; stores quantize through the
Q8_0 `set_rows`. The registered state extents of the runs above confirm it.

| Family | Layers and state on Q8_0 | Kept in F16/F32 | State extent F16 → Q8_0 |
| --- | --- | --- | --- |
| Gemma 2 2B | all 26 layers' K/V (D256, alternating 4,096 window and global; attention softcap 50) | — | 1,337,982,976 → 713,031,680 (0.533) |
| Gemma 3 4B QAT | all 34 layers' K/V (D256, 1,024-window local and global) | — | 325,058,560 → 174,063,616 (0.535) |
| Gemma 4 26B-A4B | all 30 layers: 25 sliding D256 and 5 global D512 | — | 2,348,810,240 → 1,249,902,592 (0.532) |
| Gemma 4 31B | all 60 layers: 50 sliding D256 and 10 global D512 | — | 6,878,658,560 → 3,657,433,088 (0.532) |
| Gemma 4 assistants | no K/V of their own: `attention.borrowed` reads the target's Q8_0 cells through the same kernel | — | (target's) |
| Qwen3.8 (both) | every full-attention (QSA, D256) layer's K/V; the MTP drafter's attention | QSA indexer keys (F32); GDN linear-attention state | GGUF 4,731,174,912 → 3,179,282,432 (0.672); NVFP4 + MTP 5,811,208,192 → 4,125,097,984 (0.710) |
| DeepSeek V4 + DSpark | every MQA layer's window ring and compressed cache (D512); DSpark's injected ring | indexer cache and compressor partial sums | 8,044,675,072 → 4,957,667,328 (0.616) |

Q8_0 cells are 34/64 of F16, so a state whose every tensor is K/V lands at
0.531 plus 2 MiB rounding: both Gemma 4 sizes do, so their D512 global layers
are quantized, not left in F16. Qwen3.8 and DeepSeek keep their non-K/V
state at full width, so their ratios are higher.

### Opt-in domain boundaries

- **Per alias, off by default.** `kv_cache_encoding` defaults to `"f16"`;
  `"q8_0"` is the only lossy value (D-117). It is refused for compositions
  (`model_settings.cc`: "KV cache encoding is not a composition setting"),
  for a model whose target or drafter architecture is not on the component
  engine (`ArtifactFactsOf`), and for an unauthenticated spec.
  `OnComponentEngine` (`component_admission.cc`) covers every served LLM
  architecture: `gemma2`, `gemma3`, `gemma4`, `gemma4-assistant`,
  `qwen4exp`, `qwen4exp-mtp`, `deepseek4` and `dflash`. The legacy runners
  never ran Q8_0 state (DeepSeek's is being deleted in G4d).
- **Spec bounds** (`state_encoding.cc`, the components' `Shapes`): one
  independent LLM target with at most one drafter stage; dense attention at
  D64/D256/D512 with equal K/V widths, no explicit scores, and an attention
  softcap only at D256; QSA at D64/D256/D512; MQA at D256/D512. Any other
  stateful or injected component is refused rather than silently kept in F16.
- **Not covered:** Qwen2's FP16 and EXL3 fixtures (not served aliases) have
  only the short D64 sample in the first section; no other lossy encoding
  (Q4_0 exists in the kernels but is not a setting).

### Prefill fix found on the way

Long prompts failed on `505df41a` for every component-engine model except
DeepSeek: a Gemma 2 prompt of 883 tokens (128-row chunks) or a Gemma 4 26B
prompt past one 1,024-row chunk returned HTTP 500, "the step's scratch
(28396800 bytes) exceeds the pool (26738688 bytes)". `a0e874cf` began
configuring four GGML lanes on every component launch, carving
min(4 MiB, pool/16) each from the end of the shared pool, but only
DeepSeek's serving plans have lane regions and are sized for them; every
other model's pool covers its stream scratch alone. `component_build.cc` now
configures lanes for DeepSeek serving only, which restores every other
model's launch to its gated form. Afterwards prompts of 7,473–14,235 tokens
prefill on Gemma 2, Gemma 3, both Gemma 4 sizes, both Qwen3.8 formats and
DeepSeek. The first K1 run did not see this (its binary predates
`a0e874cf`, and scoring does not prefill).

### Receipts

Raw receipts (per-arm files with every scored log-probability and the
passage, combined `*-k1.json`, runtime logs) stay in
`spark:/home/pmeenan/scratch/m36-k1fix-quality02/`; the reference probes
(llama.cpp tokenization and perplexity, window scoring, the frozen-runtime
turn-format score) in `spark:/home/pmeenan/scratch/m36-k1fix-probe/`.
llama.cpp ran from the pinned image
`ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db`
(v0.6.0, build 11429, `d81235049`) over the GGUFs in
`~/.local/share/llmp/reference-models/` and
`~/.local/share/llmp/models/ggml-org/gemma-3-4b-it-qat-GGUF@bbcac0d0/`.

| Family | Passage SHA-256 (prefix) | Characters |
| --- | --- | ---: |
| Gemma 2 | `f9c34b6e6e86b525` | 31,952 |
| Gemma 3, Gemma 4 26B | `97b7f200ef681f10` | 31,887 |
| Gemma 4 31B | `fb5c3065443fc1d8` | 23,607 |
| Gemma 4 assistants | `c4ae59334ef519df` | 4,422 |
| Qwen3.8 GGUF | `e63e63f04a593fcc` | 66,476 |
| Qwen3.8 | `d84098e54e342b07` | 58,193 |
| DeepSeek | `d9d5e376783be4c6` | 42,574 |
