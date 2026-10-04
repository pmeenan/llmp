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
