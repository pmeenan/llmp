<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tokenizer and chat-template references (M3), 2026-09-28

`generate.py` produced the expected token IDs and rendered texts that the
native tokenizer and renderers are tested against
([tokenizer.md](../../tokenizer.md)): `tests/unit/data/tokenizer/*.json`
(except the hand-written `corpus.json`) and `tests/unit/data/chat/*.json`.
Only those outputs are in the repository; the scratch directory's inputs
and container outputs are not.

## What ran where

On `spark-b` (spark-56f5), in a virtual environment holding
[requirements.txt](requirements.txt) (Python 3.12.3), with the model files
under `~/.local/share/llmp/`:

| Reference | Identity | Role |
| --- | --- | --- |
| llama.cpp `llama-tokenize` | `ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7` (b29c606e, the pinned reference, [reference-setup](../reference-setup/README.md)), `--no-escape --ids --no-bos`, with and without `--no-parse-special`, `--network none` | GGUF vocabularies: DeepSeek V4 0731 (its oracle) and the Qwen3.8 GGUF |
| Hugging Face tokenizers | 0.22.2; `encode(text, add_special_tokens=False)`, and with `encode_special_tokens` for the other mode | `tokenizer.json` vocabularies: Qwen3.8 NVFP4 (vLLM's), Qwen-Image 2.1's encoder, DeepSeek's own |
| transformers | 5.12.1 `apply_chat_template(..., chat_template=<pinned bytes>, tokenize=False)` (Jinja2 3.1.6) | Rendered texts of both LLM templates |
| DeepSeek's encoder | `encoding/encoding_dsv4.py` at `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b` (SHA-256 `abc0d261…`, MIT) | Second reference for DeepSeek's cases it defines |

The DeepSeek repository's `tokenizer.json` (`8f9f37ca…`),
`tokenizer_config.json` (`6ac8c8dc…`), `LICENSE` (MIT, `f2c6c602…`) and
`encoding/` were fetched at `7872f01b` into
`~/.local/share/llmp/tokenizer-reference/deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b/`
on `spark` and `spark-b`, beside the model downloads rather than among them.
The encoder is run, never incorporated; the models test reads that
`tokenizer.json`. Each fixture records its source file's SHA-256, which the
tests check.

The community GGUF's "chat-v2" fixture (`deepseek-v4-chat-v2.json`) was
added on `spark-b` on 2026-10-02 with the same environment; that run
regenerated the other chat fixtures byte for byte. Its template is read
from
`models/antirez/deepseek-v4-gguf@f71f23d5/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
(SHA-256 `87249207…`) and its `gguf_ids` come from llama-tokenize on that
file. transformers has no `from_json` filter, which this template applies
to string arguments, so `generate.py` adds Python's `json.loads` as it.
The script also checks that the template's `thinking` and
`enable_thinking` give the same text and that `reasoning_effort` changes
nothing.

## Rerunning

```sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt
HF_HUB_OFFLINE=1 venv/bin/python docs/experiments/tokenizer-reference/generate.py \
    --repo . --models ~/.local/share/llmp --work <scratch>
```

It needs `sudo -n docker` for the llama.cpp image and takes about two
minutes (each of the 184 llama-tokenize calls per GGUF loads its
vocabulary). A
changed corpus or case list regenerates every fixture; the tests then say
whether the native side still agrees.

## Conversations

The chat cases are in `generate.py` (`COMMON_CASES`, `DEEPSEEK_CASES`,
`DEEPSEEK_V2_CASES`, `QWEN_CASES`) and are copied into each fixture with
their outputs. A tool call's `arguments` may be a string, the client's
text, which the template sees as is and the test parses. A case
the template refuses records the template's exception; a case llmpalooza
refuses although the template renders it (an unknown `reasoning_effort`)
records the template's text beside the expected error. For DeepSeek,
`upstream_encoder` is `same` when `encoding_dsv4.py` gives the template's
text, `n/a` when the encoder cannot express the case (tools without a
system message, a null content, system messages after the first), and
`differs` never.

## Findings

- All five configurations agree with their reference on every corpus item
  but the one recorded divergence (tokenizer.md).
- llama.cpp and Hugging Face agree on the Qwen3.8 vocabulary except where
  NFC changes the text, and on DeepSeek's except the Unicode 16.0 emoji and
  the special-token kinds the two files declare differently.
- Neither reference matches `'ſ` (long s) as the `'s` contraction.
- The DeepSeek GGUF template and `encoding_dsv4.py` agree on all 17 cases
  both define.

## Gemma 4 raw UTF-8 BPE (M3.5), 2026-10-04

`generate.py --only tokens --config gemma-4-26b-gguf` uses the same pinned
llama.cpp image on `spark` and the approved
`unsloth/gemma-4-26B-A4B-it-GGUF@c099eb48e663fd284577b04978a94ffccb261841`,
`reference-models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`. Its first 15,784,342
bytes are the complete GGUF header and metadata, SHA-256
`9f1f00bb292382747a7035a8a3e5d5f41f31e7d7be80b2aefa044ff03434f990`.
The generator checks those bytes and keeps them outside Git at
`tokenizer-reference/gemma-4-26b/header.gguf`, which the models test reads;
the oracle tokenizes from the full checkpoint. The fixture contains 92
texts in both special-token modes, 184 encodings and 10,262 token IDs.
The checkpoint's 262,144-token vocabulary has 256 byte fallback tokens
and 514,906 ranked merges. No inference or Gemma runner is covered here.

## Classic SentencePiece (M3.5), 2026-10-04

On `spark` (spark-c4e2), supervised `m35-spm-oracle-corpus` generated the
three legacy fixtures using the same pinned llama.cpp image and corpus:
552 encodings, 38,284 token IDs, with no automatic BOS. Native token IDs
agree on every encoding (Phi 3.5: 17,836 IDs; Gemma 2: 10,194; Gemma 3:
10,254). These establish tokenizer behavior, without weight execution or
rendered-chat comparisons. Command R 7B was inspected but uses `gpt2` /
`command-r`, rather than SentencePiece, and is outside this slice.

Only complete GGUF header/key-value prefixes were fetched with bounded
HTTP ranges. Each pinned repository tree API confirmed the full file size
and LFS SHA-256; no multi-GB weights were downloaded or hashed. The fetched
prefix bytes are kept under `tokenizer-reference/legacy/` and are the
native models tests' inputs. `generate.py` verifies their exact lengths
and SHA-256 before making a vocabulary-only oracle copy: it sets the GGUF
tensor count to zero and pads to 32-byte alignment, preserving every
original key/value byte. `llama-tokenize` uses `vocab_only=true`, so tensors
are unnecessary. Fixtures record both prefix identity and the upstream
checkpoint/LFS provenance.

| Fixture | Approved repository/revision and file | Original prefix bytes | Prefix SHA-256 |
| --- | --- | ---: | --- |
| `phi-3.5-gguf` | `bartowski/Phi-3.5-mini-instruct-GGUF@6d70da17e749a471ccb62ade694486011a75cda3`, `Phi-3.5-mini-instruct-Q8_0.gguf` | 727,029 | `876536e399233294995425441ee13b2b5ae01a6ad88281696c8cb38dc03752fa` |
| `gemma-2-2b-gguf` | `bartowski/gemma-2-2b-it-GGUF@855f67caed130e1befc571b52bd181be2e858883`, `gemma-2-2b-it-Q8_0.gguf` | 6,029,343 | `22684623bd76697a54af7f4ee54b02ba50955106a724eaa763ca8d917c136a88` |
| `gemma-3-4b-gguf` | `ggml-org/gemma-3-4b-it-qat-GGUF@bbcac0d065076c47042838c0675c602411b0dd4c`, `gemma-3-4b-it-qat-Q4_0.gguf` | 6,514,895 | `3073f37db9c1977ed7d68cd64707ce32043eecd250000a929a2ceb6723b1ad4c` |

All three declare `tokenizer.ggml.model = "llama"`, `pre = "default"`,
256 byte tokens, finite F32 scores and no merge list. Phi 3.5 defaults to
a dummy space prefix; both Gemma files explicitly disable it. Their score
vectors have respectively 32,064, 256,000 and 262,208 entries. Gemma's
user-defined whitespace runs match even without parsing controls. The
approved Phi checkpoint's spaced name (`Phi 3.5 Mini Instruct`) does not
activate the pinned `phi-3` / `phi3` special whitespace-stripping override;
synthetic tests cover that override independently.

Rerun each configuration with the original prefixes present:

```sh
python3 docs/experiments/tokenizer-reference/generate.py --repo . \
    --models ~/.local/share/llmp --work <scratch> --only tokens \
    --config phi-3.5-gguf
```

Use `gemma-2-2b-gguf` or `gemma-3-4b-gguf` for the other fixtures. GGUF-only
runs do not need the Python transformer/tokenizer dependencies. Builds and
oracle jobs on a Spark run through the installed `spark-job --gpu`
supervisor; native builds use the locked SDK and prepared source receipts.

## Native Gemma 2/3 HF and ModelProto assets (M3.6 follow-up)

The native HF path reads the original tokenizer assets, without converting the
checkpoint to GGUF. The metadata-only archived official inputs are:

| Repository | Revision | Native readers |
| --- | --- | --- |
| `google/gemma-2-27b-it` | `aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0` | `tokenizer.json`, `tokenizer.model`, kept `tokenizer_config.json` |
| `google/gemma-3-27b-it` | `005ad3404e59d6023443cb575daa05336842228a` | the same |

Both model files declare scored BPE, byte fallback and identity normalization,
with dummy prefix and whitespace collapse disabled. Their HF pipelines declare
space-to-marker replacement, the Split on spaces after replacement, and the
inverse Replace/ByteFallback/Fuse decoder. HF's TemplateProcessing inserts BOS;
the kept tokenizer configuration gives the same insertion rule for ModelProto.
Unigram is a different algorithm and is not claimed by this BPE support.

Stage the metadata from the archive into the standard local `models/google/`
paths before the agreement checks; no weight shard is needed. Generate each
of `gemma-2-27b-hf`, `gemma-2-27b-spm`, `gemma-3-27b-hf` and
`gemma-3-27b-spm` using `--only tokens --config NAME`. The HF `tokenizer.json`
loaded by the independent Hugging Face tokenizers reference supplies expected
IDs for both native readers of the same official vocabulary. Each fixture
records the exact native file, configuration and reference JSON hashes. These
fixtures also record encoding with automatic special tokens, in both parsing
modes, so BOS behavior is checked rather than inferred from admission. The
existing corpus gives 184 encodings without insertion and 184 with insertion
per configuration. Existing served-model fixtures remain the regression oracle;
the new generator configurations do not regenerate them.

## Llama 3.2 GGUF and compiled split plans (R8), 2026-10-18

The new `llama-3.2-1b-gguf` configuration uses the current locked reference
from [llama-reference-refresh/pins.json](../llama-reference-refresh/pins.json):
llama.cpp v0.6.0, source `d81235049384534c167caea52b85a694f6103d14`,
ARM64 image
`ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db`
(config `c45165320931c3c957baef9edf518d731870f9c4f9eb98f749170d5009afe2bd`).
Historical configurations retain their original image and output identities.

The original archived source is
`bartowski/Llama-3.2-1B-Instruct-GGUF@067b946cf014b7c697f3654f621d577a3e3afd1c/Llama-3.2-1B-Instruct-Q8_0.gguf`,
1,321,083,008 bytes, archived manifest SHA-256
`432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3`.
Only its complete 7,822,851-byte metadata prefix is staged for the native
reader, at `~/.local/share/llmp/tokenizer-reference/llama32/header.gguf`, SHA-256
`bd1db0aa84ea2d4e51eaa0a2202888886ded4a67c28abb726b2d6db4686fd4ff`.
This is metadata staging, not a complete staged checkpoint or a new full-weight
hash verification. The oracle copy changes tensor count to zero and pads to
32-byte alignment: 7,822,880 bytes, SHA-256
`85bc054b2a2aa699a5e3c48e2386adf4fe8eb7e04372ed70f1e4c254b6c9c7de`.
It retains every original tokenizer metadata byte and remains outside Git.

On `spark-b`, supervised `r8-llama32-oracle-generate` ran the current
`llama-tokenize` with `--no-escape --ids --log-disable`, parsing specials and
with `--no-parse-special`, both with `--no-bos` and with that flag omitted.
The current binary's help confirms BOS is enabled by default. The resulting
fixture has 92 corpus inputs (83 valid, nine replaced after strict UTF-8
refusal), four encodings each, plus four independent special-token probes:
empty input, literal `<|begin_of_text|>`, `Hello!`, and BOS followed by that
text. Automatic insertion produces `[128000]` for empty input and two BOS IDs
for a parsed preexisting BOS; no deduplication is inferred. The sealed fixture
SHA-256 is `2099969b0945b6f1aa3e28eb3fb4824d9a9c8152f3cff84cdbe6c3155fb82e17`.

Rerun this configuration alone in an external fixture workspace, with the
unchanged checked-in corpus copied to its normal relative path:

```sh
python3 docs/experiments/tokenizer-reference/generate.py --repo <fixture-workspace> \
    --models ~/.local/share/llmp --work <scratch> --only tokens \
    --config llama-3.2-1b-gguf
```

Inventory the current image and helpers before acquiring anything. Model
metadata comes from the archive under [model-archive.md](../../model-archive.md).
The generator verifies the prefix before deriving the vocabulary-only oracle
copy. This establishes GGUF oracle evidence; it does not claim an official HF
Llama `tokenizer.json` was acquired or tested. Synthetic HF tests cover generic
regexes and shared postprocessors. Native agreement is reported in tokenizer.md
once its checks complete.

The independent engine is Apache-2.0. The GGUF expression/alias table in
`src/tokenizer/presets.cc` is transcribed MIT data from the same locked
`src/llama-vocab.cpp` (189,021 bytes, SHA-256
`18f624c7f5a64f6bd4458248d2cde0ba989cbd46c47b8007be1765e31486c938`),
with the ggml authors' copyright and `MIT AND Apache-2.0` tags. The table uses
actual runtime expression strings, including ASCII contraction classes and
GPT4O/Tekken's category/class lookahead workaround. HF expressions retain
Unicode case semantics directly. The pin's explicit `default` BPE preset has
four ordered stages; absent/empty BPE names select that same declared default,
independently of the SentencePiece algorithm's separate default.

Independent regex controls used the already installed Python 3.12.3,
tokenizers 0.22.2 and transformers 5.12.1 on `spark-b`. The exact two HF Qwen
expressions were read from source files with tokenizer.json SHA-256
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`
(Qwen2.5 EXL3) and
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`
(Qwen3.8 FP8). The corrected group-boundary probe JSON is SHA-256
`06c11ad9c6a028297e2657a241730c1efc50bd67c583d6040ab6325141f94d15`;
the bare start/space/in-word controls are SHA-256
`5f1c76c044512c286c7afcc943e9b94bfbdbad6192b955143b06c7b1e755a4b3`.
These controls establish long-s and Kelvin-sign literal equivalence, default
full folding inside literal runs, and its boundary behavior: `ss` can match
sharp-s, while `s{2}`, `[s][s]`, separate scoped groups and captures cannot
join its expansion. Sentinels `!`/`~` avoid confounding ASCII class matches.
The native tests retain original code-point spans; they do not fold the whole
input or normalize it as a substitute for matching.

The generated folding data comes directly from the official versioned
[UCD 15.1.0 CaseFolding.txt](https://www.unicode.org/Public/15.1.0/ucd/CaseFolding.txt),
84,870 bytes, SHA-256
`4e55acfdc32825a22e87670e9056a3bf94ad7c5400065778e9e10f8314372bcf`.
The generator checks this sixth input alongside the existing UCD files and
uses C/F records for default full folds, excluding simple-only S and Turkic T
records. Existing Unicode License V3 attribution applies; Python's runtime
case mappings are not its source.


The independent isolated-Split zero-width probe used the same pinned HF
environment, raw log SHA-256
`b5fe2135b53ca401a76c47b6d7309883283d3b925f0a1ff38a20f09e9d90420d`.
On `baac`, `(?=[a])` yields `b`, `a`, `ac`, with offsets [0,1], [1,2], [2,4].
On `aaa` it yields three characters; an empty regex similarly splits `abc`
into its characters. Anchors `^`/`$` retain the whole `abc`; `a*` yields
`b`, `aa`, `c` on `baac`. Successful zero-width matches therefore establish
boundaries without emitted empty pieces; they differ from unsuccessful matches.
Native controls also cover lazy nullable progress and retained input.
