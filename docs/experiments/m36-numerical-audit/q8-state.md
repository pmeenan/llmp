<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded dense Q8_0 cache opt-in

The shared dense-attention component now consumes existing quantized cache stores
and vector attention at D64/D256. F16 remains the default. The explicit per-alias
setting changes cache state, not weights, and does not widen serving admission.
This first screen measures incremental loss against the same checkpoint with
F16 cache. It establishes functional/state support, not an acceptable-loss
threshold, family quality approval, long-context behavior or a speed gain.

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
controls. D512, specialized QSA/MQA, sinks, borrowed state and drafters remain
outside the opt-in domain.

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

Evaluated on Spark A (`spark-c4e2`) using `tools/k1_quality.py` against the
authenticated reference corpus `~/.local/share/llmp/corpus/m3-reference-corpus.txt`
(SHA-256 `55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb`).
The runtime binary was built from `main` (`d69c8036`) natively on Spark A:
`llmp-runtime` SHA-256 `a7cb0565b4f222617738d6d96a8ac115f6bd7d4a5e11df05973f15f3cc8fa8ec`,
NVIDIA driver 580.178.04.

Each evaluation arm runs two phases under identical diagnostic configuration
(`LLMP_DIAGNOSTIC_COMPONENT_ENGINE=1`, `LLMP_DIAGNOSTIC_NODE_CAP_BYTES=120259084288`,
`LLMP_DIAGNOSTIC_STARTUP_HOST_BYTES=auto`):
1. Long-context logprob scoring (`/v1/completions` with `echo=true`, `logprobs=0`,
   `prompt_logprobs=0`) across multi-thousand token prefixes drawn from the reference corpus.
2. Two streamed chat completions (`/v1/chat/completions`) checking generation sanity
   and output digest stability across the cache encoding switch.

| Family | Scored Tokens | F16 Overall PPL | Q8_0 Overall PPL | Overall PPL Δ | Early PPL Δ (0-1K) | Late PPL Δ (last 2K) | Chat Verified |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| gemma2 | 6923 | 31.3794 | 31.4121 | +0.104% | +0.073% | +0.068% | PASS |
| gemma3 | 2407 | 59.7421 | 59.4673 | -0.460% | -0.833% | -0.439% | PASS |
| gemma26 | 6955 | 17493.9637 | 16994.1152 | -2.857% | -3.996% | -1.039% | PASS |
| gemma26-assistant | 6955 | 17493.9637 | 16994.1152 | -2.857% | -3.996% | -1.039% | PASS |
| gemma31 | 6955 | 18521.0722 | 19370.4526 | +4.586% | +8.934% | -1.461% | PASS |
| gemma31-assistant | 6955 | 18521.0722 | 19370.4526 | +4.586% | +8.934% | -1.461% | PASS |
| qwen38 | 7447 | 13.4450 | 13.5482 | +0.768% | +0.589% | +0.799% | PASS |
| qwen38-gguf | 7447 | 13.9773 | 13.9313 | -0.329% | +1.931% | -1.391% | PASS |
| deepseek | 7136 | 16.3247 | 16.3508 | +0.160% | +0.041% | +0.086% | PASS |

### Generation and cache stability observations

- **Greedy bit-for-bit equivalence:** Four configurations produced bit-for-bit
  identical 64-token chat streams between F16 and Q8_0 cache encodings:
  - `gemma2`: reply SHA-256 `ba31cf2730424d63629d462803de363f4e177c928e419efe763428819facdb60`
  - `gemma31`: reply SHA-256 `3519c7a2cb43ca98874c16a69de36a6dd51070c3257083e0419c610d69139f66`
  - `gemma26-assistant`: reply SHA-256 `3eb6d9f4595631009f7b57304afdf15accf5ff2744610c5a644010c62bc762cd`
  - `gemma31-assistant`: reply SHA-256 `3519c7a2cb43ca98874c16a69de36a6dd51070c3257083e0419c610d69139f66`
- **Speculative decoding under Q8_0:** Speculative decoding with attached drafters
  runs cleanly with Q8_0 KV cache across assistant models, MTP drafters, and DSpark:
  - `gemma26-assistant`: 23 chunks streamed for 64 tokens (2.78 tokens/chunk)
  - `gemma31-assistant`: 24 chunks streamed for 64 tokens (2.67 tokens/chunk)
  - `qwen38`: 25 chunks streamed for 64 tokens (2.56 tokens/chunk)
  - `deepseek`: 24 chunks streamed for 64 tokens (2.67 tokens/chunk)
- **QSA sparse attention under Q8_0:** Sparse attention kernels (`QsaAttnKernel<true>`)
  quantize key/value fetches dynamically at long context (> 2048 cells) with minimal
  perplexity impact: +0.768% on `qwen38` and -0.329% on `qwen38-gguf`.
- **Bounded perplexity delta:** Across all 9 evaluated configurations, long-context
  perplexity delta remained tightly bounded within [-2.9%, +4.6%] (and within +0.16%
  for `deepseek`). At deep context (final 2,048 tokens of 7K contexts), late perplexity
  deltas are negligible to negative across all models.

### Opt-in domain boundaries and audited refusals

- **Default engine without diagnostic opt-in (`deepseek`):** Refused at service startup:
  `refusing to serve: model deepseek: Q8_0 KV requires a supported shared model`
  (`src/runtime/model_settings.cc:494`), as `OnComponentEngine("deepseek4")` defaults to false
  (`src/runtime/component_admission.cc:610`). Under the diagnostic component engine
  (`LLMP_DIAGNOSTIC_COMPONENT_ENGINE=1`), `deepseek` runs on the component engine and
  successfully evaluates with +0.160% overall perplexity change.

### Receipts and provenance

Raw evaluation receipts reside in `/tmp/llmp-m36-coordination/k1-quality01/` (synced
from `spark:/home/pmeenan/scratch/m36-gate/k1-quality01/`):
- `deepseek-k1.json`: SHA-256 `24b434088fa91569c8d21254a88ffb8815ef7565f272dd5f8ed93026987378a1`
- `gemma2-k1.json`: SHA-256 `e5e77f98f2793b0b0056701ba5d34bb4e3b46f7d46ec2567742bd524d65cb837`
- `gemma3-k1.json`: SHA-256 `251d7a740725498dbc9f21d0d47e2395b0818a718d77a1843b2aafd9b2c4636b`
- `gemma26-k1.json`: SHA-256 `7c02996b7e4849c3c5409f582a49d8cb86ca70630f88b772279bd6587f9533ed`
- `gemma26-assistant-k1.json`: SHA-256 `ebda4ea26a2738c44988d4c7fa0b9487e6de1cf476264795619eec055ebcc45a`
- `gemma31-k1.json`: SHA-256 `4359b46ed78bb28039231533a305359040f77fe446e964ea1b643f5e710506f0`
- `gemma31-assistant-k1.json`: SHA-256 `19305fa8554be98e7df20e216d199471fcc5c12e2bde5730e407432c41940382`
- `qwen38-k1.json`: SHA-256 `7b94eb27b31d14f8e1a25fa029ad414120d87dac63eb497367159204e2b06389`
- `qwen38-gguf-k1.json`: SHA-256 `a7b64dd35b34a739084cbad2313af12047763cc4abe8937d9cc6de4ebd9d4c5d`

Input passages from reference corpus:
- 30,000 chars (7,136–7,447 tokens): SHA-256 `278bd924c7006d591130ab8aa8b17480984f7804d7d72c9ccd726c9fa9fea650`
- 26,000 chars (6,923–6,955 tokens): SHA-256 `e399047ab7609a784c45bd858e871cdcb3e215c0cd2e30b22404c480c9bc1806`
- 11,500 chars (2,407 tokens): SHA-256 `3923a9bb401ae40bdb967dc18ed87e17bd8038ec12cb58e2026fb8c8890ffb9e`
