<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Chat-template corpus and interpreter references (M3), 2026-10-02

The evidence for D-067's 2026-10-02 amendment: the Jinja-subset
interpreter (`src/chat/jinja*`) and the choice of renderer
(`chat::ChatTemplate`) against Hugging Face transformers on real chat
templates ([tokenizer.md](../../tokenizer.md#the-template-interpreter)).

- `snippets.py` writes `tests/unit/data/chat/jinja-snippets.json`: 133
  small templates written for jitLLM, each rendered by transformers'
  compiled chat-template environment with fixed JSON variables, text or
  exception recorded (`jinja_test`).
- `collect.py fetch` downloads the corpus's templates read-only at pinned
  revisions (a GGUF's metadata by an HTTP range read; no weights) and
  writes `manifest.json`; `collect.py render` renders each on
  `tests/unit/data/chat/corpus-conversations.json` with transformers'
  `render_jinja_template` and the manifest's BOS and EOS texts, passing
  messages as jitLLM does, and writes `references.json`
  (`chat_corpus_test`).
- `random_conversations.py SEED COUNT` writes random conversations in the
  same form, for `collect.py render` and `chat_corpus_test` pointed at
  them (its docstring says how).

Template texts and their renderings stay outside the repository, in
`~/.local/share/jitllm/chat-templates/` on `spark` and `spark-b`.

## What ran where

On `spark-b` (spark-56f5), in the tokenizer reference's environment
([requirements](../tokenizer-reference/requirements.txt): Python 3.12.3,
transformers 5.12.1, Jinja2 3.1.6), on 2026-10-02:

```sh
venv/bin/python collect.py fetch --out ~/.local/share/jitllm/chat-templates --models ~/.local/share/jitllm
venv/bin/python collect.py render --out ~/.local/share/jitllm/chat-templates \
    --conversations tests/unit/data/chat/corpus-conversations.json
venv/bin/python snippets.py --out tests/unit/data/chat/jinja-snippets.json
# the review's random conversations, in a copy of the corpus directory
venv/bin/python random_conversations.py 7 200 > REVIEW/data/chat/corpus-conversations.json
venv/bin/python collect.py render --out REVIEW/models/chat-templates \
    --conversations REVIEW/data/chat/corpus-conversations.json
```

The tests ran on `spark` (spark-c4e2), `spark-native` RelWithDebInfo, SDK
`aarch64-e0a0c85c42806fb1`; the review's sanitizer runs in a build of the
same configure with `JITLLM_SANITIZE="address;undefined"` and
`JITLLM_CUDA=OFF`.

## Corpus

| Template | Source (revision) | SHA-256 | Renders by |
| --- | --- | --- | --- |
| deepseek-v4-0731-gguf | unsloth/DeepSeek-V4-Flash-0731-GGUF (fbbb5b93), GGUF metadata | `e643c31f…` | native, hash |
| deepseek-v4-chat-v2-gguf | antirez/deepseek-v4-gguf (f71f23d5), GGUF metadata | `87249207…` | native, hash |
| qwen3.8-flash-next-nvfp4 | Mia-AiLab/Qwen3.8-Flash-Next-NVFP4 (925d7be6) | `c3cf9e34…` | native, hash |
| qwen3.8-flash-next | Qwen/Qwen3.8-Flash-Next (de4b8e4d) | `c3cf9e34…` | native, hash |
| qwen3.8-27b | Qwen/Qwen3.8-27B (1d4bf0f2) | `c3cf9e34…` | native, hash |
| qwen3.8-flash-next-unsloth-gguf | unsloth/Qwen3.8-Flash-Next-GGUF (38bb39ee), GGUF metadata | `12827f24…` | native, probe (Unsloth variant) |
| qwen3.8-27b-unsloth-gguf | unsloth/Qwen3.8-27B-GGUF (4ca72078), Q4_K_M metadata | `12827f24…` | native, probe (Unsloth variant) |
| qwen3-8b | Qwen/Qwen3-8B (b968826d) | `a55ee1b1…` | interpreter |
| qwen2.5-7b-instruct | Qwen/Qwen2.5-7B-Instruct (a09a3545) | `cd8e9439…` | interpreter |
| qwen3-coder-30b-a3b | Qwen/Qwen3-Coder-30B-A3B-Instruct (b2cff646) | `5a38bfa0…` | interpreter |
| gemma-4-26b-a4b-it, gemma-4-31b-it | google/gemma-4-26B-A4B-it (4d7ae498), google/gemma-4-31B-it (842da379) | `ae53464b…` | interpreter |
| gemma-3-4b-it | unsloth/gemma-3-4b-it (bf46152c) | `7de1c58e…` | interpreter |
| gpt-oss-120b | openai/gpt-oss-120b (b5c939de) | `a4c9919c…` | interpreter |
| kimi-linear-48b-a3b | moonshotai/Kimi-Linear-48B-A3B-Instruct (e1df551a) | `75e3cd76…` | interpreter |
| kimi-k2-instruct | moonshotai/Kimi-K2-Instruct (fd1984e2) | `39e8c195…` | interpreter |
| nemotron-3-super-120b | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 (ff433f54) | `575fb74f…` | interpreter |
| mistral-small-4-119b | mistralai/Mistral-Small-4-119B-2603 (a11f36be) | `de56319d…` | interpreter |
| mistral-small-3.2-24b | unsloth/Mistral-Small-3.2-24B-Instruct-2506 (3ed8d341) | `a126eea7…` | interpreter |
| llama-4-scout | unsloth/Llama-4-Scout-17B-16E-Instruct (afd8e498) | `93c6a232…` | interpreter |
| llama-3.3-70b | unsloth/Llama-3.3-70B-Instruct (99cd0d2c) | `e10ca381…` | interpreter |
| llama-3.2-3b | unsloth/Llama-3.2-3B-Instruct (006f5dcd) | `5816fce1…` | interpreter |
| glm-4.7-flash | zai-org/GLM-4.7-Flash (7dd20894) | `d63ad536…` | interpreter |
| mimo-v2.6-flash | XiaomiMiMo/MiMo-V2.6-Flash-RL (5711b268) | `853650be…` | interpreter |
| phi-4-mini | microsoft/Phi-4-mini-instruct (cfbefacb) | `febf5892…` | interpreter |
| phi-4 | microsoft/phi-4 (2db69c1c) | `fe412041…` | interpreter |
| deepseek-v3.1 | deepseek-ai/DeepSeek-V3.1 (c0781d03) | `45690185…` | interpreter |
| smollm3-3b | HuggingFaceTB/SmolLM3-3B (a07cc9a0) | `3e0e275a…` | interpreter |
| granite-4.1-8b | ibm-granite/granite-4.1-8b (1504002f) | `fed2756d…` | interpreter |

The manifest records each file's full hash, its BOS and EOS texts and the
repository's declared license.

## Results

- **Snippets:** 133 of 133 agree with transformers: 103 texts byte for
  byte, 30 refusals refused (two by design where transformers fails
  differently: `include` and `call` blocks). The fifteen `review-`
  snippets were added by the independent reviews (rounding, extreme slice
  steps, searches, strips, reversal, indents, `sum` of strings; case tests
  of any value, underscores in numbers, `%` with a mapping, a repeated
  keyword).
- **Corpus:** 29 templates × 19 conversations; all 537 renderings equal
  transformers' byte for byte, and every refusal (14 cases, the templates'
  own raises in Gemma 3, Mistral, Llama 3.x, DeepSeek V3.1 and Qwen3.8) is
  refused. That holds for the interpreter alone and for what
  `ChatTemplate` chooses (native or interpreted). Every template parses.
- **Random conversations (review):** 200 conversations drawn at random
  (`random_conversations.py 7 200`: 1–9 messages of every role, content
  from a pool of tags, control-token texts, Jinja syntax, Unicode and
  whitespace, reasoning, tool calls, tools and options), rendered by
  `collect.py render` and checked by `chat_corpus_test` pointed at them:
  all 4,055 interpreted renderings equal transformers' and every refusal
  is refused. The test's served-path check fails only where DeepSeek's
  native renderers refuse a reasoning effort other than low, high and max
  that its templates render (unchanged native behaviour, after
  encoding_dsv4.py). In a scratch harness, the five native-family
  templates (three pinned, both Unsloth GGUFs) through their native
  renderers and through the interpreter on 20,000 further random
  conversations each (efforts and `preserve_thinking` included) never
  render different texts.
- **Choice:** the five pinned-hash templates render natively by hash, both
  Unsloth Qwen3.8 GGUF templates natively by probe, and the other 22
  through the interpreter.
- **Pinned fixtures:** the interpreter, given each of the three pinned
  templates, reproduces every case of its existing chat fixture, the
  jitLLM-refused cases' `reference_text` included.
- **Long conversation:** 3,001 messages, about 1 MiB of text: every
  template renders within the bounds; Gemma 4's (quadratic in messages)
  took 1.2 s, using 23 million of the 50 million steps and 0.48 GB of the
  2 GiB of work; DeepSeek 0731's 0.6 s (13.8 million steps); the rest
  0.01–0.12 s.
- **Hostile templates:** nested loops of 10^10 iterations, doubling
  strings, deep recursion and nesting, huge repetitions and ranges, one
  string held millions of times over and printed, compared or serialized,
  quadratic searches and strips, extreme slice steps and million-link
  chains of namespaces or bound methods are all refused with a bound (or
  rendered, for the chains, without recursion); the slowest, a 2 GiB
  substring-search loop and a 50-million-step loop, after about 4.4 s and
  4.1 s; one spending both bounds after about 6 s. A third review charged
  what still ran uncharged (searches by name, prefix and case tests,
  parses, formats, key sorts, the control-token scan), held loop copies
  as live and pooled one registration's probes; its attacks (7 s to
  over 30 s, or up to 6.6 GB, before) are refused within 2.2 s. Under
  ASan and UBSan (a review build on `spark`), the unit tests and 60,000
  grammar-generated random templates report nothing.

## Limitations

The corpus is 29 templates, chosen from the M3 and M3.5 families and
popular repacks; any template outside it is checked only by the snippet
and fixture tests. Probe equivalence is evidence on the probe corpus, not a
proof for every conversation. Renderings use jitLLM's message shape (no
tool-call IDs, names or content parts), so templates that require those
refuse such requests, as transformers does.

Gemma 4's template is quadratic in messages: it used 23 million steps on
the 3,001-message conversation, so (extrapolated, not measured) one of
about 4,400 such messages reaches the 50,000,000-step bound and is
refused.
