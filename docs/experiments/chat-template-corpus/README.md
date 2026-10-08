<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Chat-template corpus and interpreter references (M3), 2026-10-02

The evidence for D-067's 2026-10-02 amendment: the Jinja-subset
interpreter (`src/chat/jinja*`) and the choice of renderer
(`chat::ChatTemplate`) against Hugging Face transformers on real chat
templates ([tokenizer.md](../../tokenizer.md#the-template-interpreter)).

- `snippets.py` writes `tests/unit/data/chat/jinja-snippets.json`: 133
  small templates written for llmpalooza, each rendered by transformers'
  compiled chat-template environment with fixed JSON variables, text or
  exception recorded (`jinja_test`).
- `collect.py fetch` downloads the corpus's templates read-only at pinned
  revisions (a GGUF's metadata by an HTTP range read; no weights) and
  writes `manifest.json`; `collect.py render` renders each on
  `tests/unit/data/chat/corpus-conversations.json` with transformers'
  `render_jinja_template` and the manifest's BOS and EOS texts, passing
  messages as llmpalooza does, and writes `references.json`
  (`chat_corpus_test`).
- `random_conversations.py SEED COUNT` writes random conversations in the
  same form, for `collect.py render` and `chat_corpus_test` pointed at
  them (its docstring says how).
- `gemma_fixtures.py` writes the Gemma renderers' fixtures
  (`tests/unit/data/chat/gemma-4.json`, `gemma-4-e.json`,
  `gemma-4-2604.json`, `gemma-3.json`; `chat_test`): its cases, the
  independent review's among them, rendered by transformers on three
  Google Gemma 4 templates (in `tests/unit/data/chat/templates`,
  Apache-2.0, with Google's four earlier 31B ones that
  `chat_template_test` checks the choice of renderer on) and Gemma 3's
  (Gemma terms; the corpus copy).
- `pycase_reference.py` writes `tests/unit/data/chat/pycase-reference.json`
  (`pycase_test`): Python 3.12.3's upper, lower, title and capitalize of
  every code point, islower and isupper as ranges, and 1,020 strings
  through each operation and Jinja2 3.1.6's `title` filter.

Template texts and their renderings stay outside the repository, in
`~/.local/share/llmp/chat-templates/` on `spark` and `spark-b`.

## What ran where

On `spark-b` (spark-56f5), in the tokenizer reference's environment
([requirements](../tokenizer-reference/requirements.txt): Python 3.12.3,
transformers 5.12.1, Jinja2 3.1.6), on 2026-10-02:

```sh
venv/bin/python collect.py fetch --out ~/.local/share/llmp/chat-templates --models ~/.local/share/llmp
venv/bin/python collect.py render --out ~/.local/share/llmp/chat-templates \
    --conversations tests/unit/data/chat/corpus-conversations.json
venv/bin/python snippets.py --out tests/unit/data/chat/jinja-snippets.json
# the review's random conversations, in a copy of the corpus directory
venv/bin/python random_conversations.py 7 200 > REVIEW/data/chat/corpus-conversations.json
venv/bin/python collect.py render --out REVIEW/models/chat-templates \
    --conversations REVIEW/data/chat/corpus-conversations.json
```

The Gemma entries (all but the three first Gemma rows below) were added
the same evening, merged into the manifest without refetching the rest:

```sh
venv/bin/python collect.py fetch --out ~/.local/share/llmp/chat-templates --only gemma-4-e4b-it \
    gemma-4-26b-a4b-it-fp8 gemma-4-31b-it-nvfp4 gemma-4-31b-it-unsloth-gguf gemma-4-e4b-it-unsloth-gguf \
    gemma-4-e4b-it-lmstudio-gguf gemma-3-1b-it-gguf gemma-3n-e4b-it gemma-3-270m-it
venv/bin/python collect.py render ...   # both conversation sets, as above
venv/bin/python gemma_fixtures.py --gemma4 tests/unit/data/chat/templates/gemma-4.jinja \
    --gemma4-e tests/unit/data/chat/templates/gemma-4-e.jinja \
    --gemma4-2604 tests/unit/data/chat/templates/gemma-4-20260428.jinja \
    --gemma3 ~/.local/share/llmp/chat-templates/templates/gemma-3-4b-it.jinja --out tests/unit/data/chat
```

Google's own Gemma 3 repositories are gated (no anonymous download); the
GGUFs converted from them (`ggml-org`, `bartowski`, `lmstudio-community`,
`MaziyarPanahi`) all embed Unsloth's copy's template, `7de1c58e…`.

The tests ran on `spark` (spark-c4e2), `spark-native` RelWithDebInfo, SDK
`aarch64-e0a0c85c42806fb1`; the review's sanitizer runs in a build of the
same configure with `LLMP_SANITIZE="address;undefined"` and
`LLMP_CUDA=OFF`. The Gemma additions' tests ran on `spark-b`, the same
build configuration.

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
| gemma-4-26b-a4b-it, gemma-4-31b-it | google/gemma-4-26B-A4B-it (4d7ae498), google/gemma-4-31B-it (842da379) | `ae53464b…` | native, hash |
| gemma-3-4b-it | unsloth/gemma-3-4b-it (bf46152c) | `7de1c58e…` | native, hash |
| gemma-4-e4b-it | google/gemma-4-E4B-it (ee0ef602; E2B's is the same) | `0a2c8073…` | native, hash (E2B/E4B) |
| gemma-4-26b-a4b-it-fp8 | RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic (ed35d7ab): Google's without the final newline | `6a1015c4…` | native, probe |
| gemma-4-e4b-it-lmstudio-gguf | lmstudio-community/gemma-4-E4B-it-GGUF (99210a71), Q4_K_M metadata: E4B's without the final newline | `603a42db…` | native, probe (E2B/E4B) |
| gemma-4-31b-it-unsloth-gguf | unsloth/gemma-4-31B-it-GGUF (c1ac76e9), Q4_K_M metadata; also its 12B, 26B-A4B, QAT GGUFs and safetensors copies | `845f1ee4…` | native, probe (Unsloth variant) |
| gemma-4-e4b-it-unsloth-gguf | unsloth/gemma-4-E4B-it-GGUF (bfc15c38), Q4_K_M metadata | `241c50d8…` | native, probe (E2B/E4B, Unsloth variant) |
| gemma-4-31b-it-nvfp4 | nvidia/Gemma-4-31B-IT-NVFP4 (4135a98a): Google's template of 2026-04-28 (google/gemma-4-31B-it@145dc250); also NVIDIA's 26B-A4B | `94899c0f…` | native, probe (2026-04 variant) |
| gemma-3-1b-it-gguf | ggml-org/gemma-3-1b-it-GGUF (f9c28bcd), Q4_K_M metadata | `7de1c58e…` | native, hash |
| gemma-3n-e4b-it | unsloth/gemma-3n-E4B-it (45e9fb1d): Gemma 3's with audio and image parts | `ac03dcb3…` | native, probe |
| gemma-3-270m-it | unsloth/gemma-3-270m-it (23cf460f) | `af95fbef…` | native, probe |
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

A survey of 50 popular Gemma repositories on 2026-10-02 found no other
template in current use: Google's 12B, the QAT GGUFs and AWQ, MLX and
other repacks carry `ae53464b…`, the E2B's `0a2c8073…`, Unsloth's
Gemma 4 repositories its two variants. Google's earlier Gemma 4 templates
(the release of 2026-04-02 and the revision of 2026-04-10, `2dfbfc7d…` and
`85a08664…` for 31B) differ further and stay interpreted.

## Results

- **Snippets:** 133 of 133 agree with transformers: 103 texts byte for
  byte, 30 refusals refused (two by design where transformers fails
  differently: `include` and `call` blocks). The fifteen `review-`
  snippets were added by the independent reviews (rounding, extreme slice
  steps, searches, strips, reversal, indents, `sum` of strings; case tests
  of any value, underscores in numbers, `%` with a mapping, a repeated
  keyword).
- **Corpus:** 38 templates × 19 conversations; all 696 renderings equal
  transformers' byte for byte, and every refusal (26 cases, the templates'
  own raises in Gemma 3 and its variants, Mistral, Llama 3.x, DeepSeek V3.1 and Qwen3.8) is
  refused. That holds for the interpreter alone and for what
  `ChatTemplate` chooses (native or interpreted). Every template parses.
- **Random conversations (review):** 200 conversations drawn at random
  (`random_conversations.py 7 200`: 1–9 messages of every role, content
  from a pool of tags, control-token texts, Jinja syntax, Unicode and
  whitespace, reasoning, tool calls, tools and options), rendered by
  `collect.py render` and checked by `chat_corpus_test` pointed at them:
  all 5,318 interpreted renderings (38 templates) equal transformers' and
  every refusal is refused; so do all the Gemma templates' native
  renderings. The test's served-path check fails only where DeepSeek's
  native renderers refuse a reasoning effort other than low, high and max
  that its templates render (unchanged native behaviour, after
  encoding_dsv4.py). In a scratch harness, the five native-family
  templates (three pinned, both Unsloth GGUFs) through their native
  renderers and through the interpreter on 20,000 further random
  conversations each (efforts and `preserve_thinking` included) never
  render different texts.
- **Choice:** the ten pinned-hash templates render natively by hash; both
  Unsloth Qwen3.8 GGUF templates and the seven Gemma variants (Unsloth's,
  NVIDIA's 2026-04 one, the copies without a final newline, Gemma 3n's and
  270m's) natively by probe; the other 19 through the interpreter.
- **Gemma renderers against their templates:** `chat_corpus_test` renders
  3,000 random conversations of its own (every role order, Gemma's
  channel and turn markers in content, reasoning, tool calls with object,
  null, string and list arguments and keys out of case order or
  non-ASCII, and tool declarations through every branch of Gemma 4's
  notation, malformed ones included) through each of the twelve Gemma
  templates natively and interpreted: the same text (1,984 to 2,563 per
  Gemma 4 template, 290 per Gemma 3 one) or both refuse, every one; none
  is refused as unsupported (keys such as `Ünïcode` and `città` sort by
  Python's full lower case in both).
- **Pinned fixtures:** the interpreter, given each of the five pinned
  templates in the tree (DeepSeek's two, Qwen3.8's, Gemma 4's two), and
  Google's 2026-04-28 Gemma 4 template, reproduces every case of its chat
  fixture, the llmpalooza-refused cases' `reference_text` included.
- **Review (2026-10-02):** the independent review's 28 conversations on
  the twelve Gemma corpus templates and Google's 31B templates of
  2026-04-02, 2026-04-10 and 2026-05-18, rendered by transformers: all
  420 served renderings equal or both refuse. Its finding that the
  2026-04-10 template passed for the 2026-04 variant (a system message
  without content, schema-key properties, blank content beside tool
  results) led to the variant probes; 2026-04-02 and 2026-04-10 are now
  interpreted, 2026-05-18 rendered as the 2026-04 variant.
  Its second round found ASCII-only case mapping served where a native
  renderer deferred to the interpreter (`строка`, `ß`, keys sorting
  differently once lower-cased in full): case mapping is now Python's in
  full in both (`pycase_test` checks every code point against Python
  3.12.3), and its five `fb-` cases are in the Gemma fixtures. On its
  harness (now with the DeepSeek and Qwen corpus templates too) every
  served rendering equals transformers' or both refuse, but for one
  unchanged DeepSeek 0731 case: an empty conversation with tools, which
  the native renderer renders and the template refuses (the chat route
  never sends one: its last message must be the user's).
- **Long conversation:** 3,001 messages, about 1 MiB of text: every
  template renders within the bounds; Gemma 4's (quadratic in messages)
  took 1.2 s, using 23 million of the 50 million steps and 0.48 GB of the
  2 GiB of work (each Gemma 4 variant 1.16–1.20 s); DeepSeek 0731's 0.6 s
  (13.8 million steps); the rest 0.01–0.12 s. The native Gemma renderers
  are linear: 20,000 messages with tool rounds (over 1 MiB) render through
  Gemma 4's and its 2026-04 variant, and 20,000 through Gemma 3's, in 8 ms
  together (`chat_test`); on 1,000 messages the two Google Gemma 4
  renderers equal their templates' interpreted text and control tokens
  (`chat_template_test`).
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

The corpus is 38 templates, chosen from the M3 and M3.5 families and
popular repacks; any template outside it is checked only by the snippet
and fixture tests. Probe equivalence is evidence on the probe corpus, not a
proof for every conversation. Renderings use llmpalooza's message shape (no
tool-call IDs, names or content parts), so templates that require those
refuse such requests, as transformers does.

Gemma 4's templates are quadratic in messages: interpreted, Google's used
23 million steps on the 3,001-message conversation, so (extrapolated, not
measured) one of about 4,400 such messages reaches the 50,000,000-step
bound and is refused (beyond the chat route's intake of 1,024 messages).
It applies only to Gemma 4 templates no native renderer serves (Google's
two earliest); the native renderers are linear.
