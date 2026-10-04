<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
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
under `~/.local/share/jitllm/`:

| Reference | Identity | Role |
| --- | --- | --- |
| llama.cpp `llama-tokenize` | `ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7` (b29c606e, the pinned reference, [reference-setup](../reference-setup/README.md)), `--no-escape --ids --no-bos`, with and without `--no-parse-special`, `--network none` | GGUF vocabularies: DeepSeek V4 0731 (its oracle) and the Qwen3.8 GGUF |
| Hugging Face tokenizers | 0.22.2; `encode(text, add_special_tokens=False)`, and with `encode_special_tokens` for the other mode | `tokenizer.json` vocabularies: Qwen3.8 NVFP4 (vLLM's), Qwen-Image 2.1's encoder, DeepSeek's own |
| transformers | 5.12.1 `apply_chat_template(..., chat_template=<pinned bytes>, tokenize=False)` (Jinja2 3.1.6) | Rendered texts of both LLM templates |
| DeepSeek's encoder | `encoding/encoding_dsv4.py` at `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b` (SHA-256 `abc0d261…`, MIT) | Second reference for DeepSeek's cases it defines |

The DeepSeek repository's `tokenizer.json` (`8f9f37ca…`),
`tokenizer_config.json` (`6ac8c8dc…`), `LICENSE` (MIT, `f2c6c602…`) and
`encoding/` were fetched at `7872f01b` into
`~/.local/share/jitllm/tokenizer-reference/deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b/`
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
    --repo . --models ~/.local/share/jitllm --work <scratch>
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
the template refuses records the template's exception; a case jitLLM
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
    --models ~/.local/share/jitllm --work <scratch> --only tokens \
    --config phi-3.5-gguf
```

Use `gemma-2-2b-gguf` or `gemma-3-4b-gguf` for the other fixtures. GGUF-only
runs do not need the Python transformer/tokenizer dependencies. Builds and
oracle jobs on a Spark run through the installed `spark-job --gpu`
supervisor; native builds use the locked SDK and prepared source receipts.
