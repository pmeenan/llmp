<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 Flash Next from a GGUF checkpoint

Qwen3.8 Flash Next's GGUF quantizations are its most-downloaded format
(unsloth's 1.49M a month, ISTA-DASLab's GSQ-RCO 1.14M); until this study
jitLLM ran only Mia's ModelOpt NVFP4 checkpoint. This study runs unsloth's
UD-IQ3_XXS natively from a v0 artifact imported verbatim, judges it against
llama.cpp on the same GGUF (the same-format oracle), and measures its speed
against llama.cpp's.

## Pre-registration (fixed before jitLLM's first run on the GGUF)

- **Checkpoint:** `unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee`, UD-IQ3_XXS,
  three shards pinned in [fast-swap/pins.json](../fast-swap/pins.json)
  (id `qwen3.8-flash-next-gguf-ud-iq3xxs`; M0's download).
- **Artifact:** `import_m3.py build` over the three shards (layout.py's
  verbatim GGUF path, [artifact-format.md](../../artifact-format.md#qwen38-flash-next-gguf)).
- **Oracle:** llama.cpp b11254 (`8019dc563`, the local image
  `jitllm-llamacpp:b11254-cuda13`, pins.json), through
  [oracle.cc](oracle.cc) ([run_oracle.sh](run_oracle.sh)): flash attention
  on, F16 caches, one sequence, context 8,192, ubatch 512, every layer on
  the GPU, no speculation. Two arms: upstream's CUDA fusion on (the
  default) and off (`GGML_CUDA_DISABLE_FUSION=1`).
- **Tokens:** the qwen38-native set, the same vocabulary:
  [prompts.tsv](../qwen38-native/prompts.tsv) (the six chat prompts, 60–72
  tokens) and [ppl.tsv](../qwen38-native/ppl.tsv) (3,558 tokens: QSA's
  indexer selects past 2,048). 32 greedy tokens a prompt.
- **jitLLM:** `jitllm_qwen38_exec` resident, its default fast form (the
  GGUF form, `kernels/ggml/qwen38_graph.h`), teacher-forced on the unfused
  oracle's tokens; prompts as one chunk; perplexity in 512-row chunks, as
  the oracle.

Bounds ([compare.py](compare.py)), the DeepSeek GGUF slice's
([dsv4-native](../dsv4-native/README.md)):

1. **Greedy tokens, teacher-forced:** at each of the 192 steps jitLLM's
   argmax is the unfused oracle's token, except at a near-tie: a step where
   the oracle's top-1 to top-2 logit margin is below twice the largest
   absolute logit difference between the engines at that step; every
   exception listed.
2. **Logits:** reported per prompt, max and RMS of the difference from
   each arm, beside the arms' own difference.
3. **Perplexity** within 2% (relative) of the unfused oracle's.
4. **Resident expert layout:** every layer's routed products over the slab
   equal those over the reference layout bit for bit
   (`--layout-proof`).

Speed is reported, not gated: 8K prefill and decode (C1), jitLLM's
runtime and llama.cpp's server through
[baseline.py](../fast-swap/baseline.py) (its `session`, the same requests
for both), and `llama-bench` beside them.

## Results (`spark-b`, 2026-10-02)

Artifact `5356b5b0…` (24,627 groups, 14 shards, 82.0 GB; the n-gram
table's group 28.8 GB). jitLLM at this branch's build; llama.cpp b11254 in
`jitllm-llamacpp:b11254-cuda13`. Raw outputs stay on `spark-b`
(`~/scratch/qgff/`).

**Correctness: every bound passes**, at the first run and again at the
final code (decode's tuned jitllm.vecq launches, below).

| Measure | Result |
| --- | --- |
| Greedy, teacher-forced (192 steps), final code | 188 equal; the 4 others near-ties (oracle margin against the step's largest difference): `haiku` 18 (0.196 / 1.10), `sky` 2 (0.591 / 2.90), `sky` 27 (0.060 / 0.96), `primes` 22 (0.259 / 1.14) |
| The same, first run | 187 equal; 5 near-ties (`haiku` 18, `sky` 26, 27, 30, `primes` 22; margins 0.011–0.259) |
| Logits vs the unfused arm, final code | max \|Δ\| 2.51–3.69, RMS 0.16–0.37 per prompt |
| The two arms (fusion on vs off), same prompts | max \|Δ\| 1.7–28.4, RMS 0.14–1.96: their own greedy tokens part at near-ties, after which they score different contexts |
| Perplexity (3,557 scored tokens) | jitLLM 15.015; unfused 15.049 (−0.23%), fused 14.999; mean \|ΔNLL\| 0.138 |
| Free-running greedy (32 tokens) | `capital`, `fibonacci`, `french` identical; `haiku`, `sky`, `primes` part at steps 18, 26, 22 (the near-ties above) |
| Resident expert layout | 129,024,000 routed outputs over the slab (stride 1,977,840 B) equal the packed layout's, 0 differing |
| Load (resident harness) | 82.0 GB in 6.0 s (13.6 GB/s); llama.cpp 72.5 s |

The GGUF form's arithmetic is not llama.cpp's node for node (QSA's sparse
selection and attention, Gated DeltaNet's fused convolution and step, the
fused router, jitllm.vecq at decode), so near-ties are expected, as in
the DeepSeek GGUF slice. The unfused graph (`--unfused`: GGML's nodes,
the port of b29c606e2's `qwen4exp.cpp`) is not llama.cpp b11254's bit
for bit either: 187 of 192 equal, 5 near-ties (margins 0.011–0.316),
max |Δ| 1.94–3.04 and RMS 0.16–0.36 against the unfused arm.

**Speed (C1).** [baseline.py](../fast-swap/baseline.py) `session` for
both engines, the same requests (its prefill filler fitted to each
engine's token counts, the `prose` and `code` decode prompts, 256 tokens,
three repeats, medians; no `ignore_eos`, and every reply ran its 256
tokens): `jitllm-runtime` with this artifact (context 32,768, plain, the
n-gram rows from the SSD) and `llama-server` b11254 (`-ngl all -fa on -c
16384 -np 1 --fit off -cram 0 -lm none`, `LLAMA_IMAGE`). `llama-bench`
(upstream's defaults, `-lm none`, three repetitions) beside them.

| Measure | jitLLM | llama.cpp server | ratio | llama-bench |
| --- | ---: | ---: | ---: | ---: |
| Prefill 512 (tok/s) | 637 | 630 | 1.01 | |
| Prefill 2,048 | 978 | 686 | 1.43 | |
| Prefill 8,192 | 1,118 | 676 | 1.65 | 768 (pp8192) |
| Decode `prose` (tok/s) | 33.8 | 31.2 | 1.08 | 32.1 (tg128, empty context) |
| Decode `code` | 33.8 | 31.2 | 1.08 | |
| Start to first token (s) | 6.3 | 68.7 | | |
| Peak `MemAvailable` drop (GiB) | 56.9 | 80.1 | | |

jitLLM's first measurement (before the launch tuning below) gave the same
prefill and 31.5 tok/s decode. Decode is bandwidth-bound: an nsys kernel
summary of the resident harness's decode put about 80% of a 33.8 ms step
in the quantized vector products (`jitllm.vecq`), about 4.8 GB of weights
a token (the Q6_K head alone 0.52 GB). `jitllm_vecq_bench`'s new `qwen`
cases (one launch each, cold L2) showed the dense Q6_K products at
210–220 GB/s already, but DeepSeek's default launches leaving 15–30% on
the routed IQ2_S gate/up (154 → 180 GB/s with one row a block), IQ4_NL
down (168 → 216 GB/s, two rows a warp) and the mixers' 320-value Q8_0
rows (155 → 238 GB/s, four rows a block); those defaults, scoped to
these types and shapes, took decode from 31.5 to 33.8 tok/s (+7%).

## Not done here

- Speculation: unsloth's MTP GGUFs are not imported; the GGUF form
  refuses a verify and the drafter's streams.
- Other quantizations: only UD-IQ3_XXS ran. The build's products take
  the types the UD and GSQ-RCO builds mix but IQ1_M (no tile kernel
  upstream: a build with IQ1_M matrices or experts is refused at setup),
  and the n-gram row lookup Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and IQ4_NL
  tables (unit-tested bit for bit against GGML's own dequantization); the
  other builds themselves, Q2_0 (GSQ) among them, are untested.
- The template: the GGUF's own chat template is not registered, so the
  runtime takes the NVFP4 checkpoint's `tokenizer.json` and template (the
  same vocabulary; the GGUF's tokenizer does not normalize to NFC).
- Prefill at depth (32K and beyond) and the swap table were not measured
  for this artifact.
