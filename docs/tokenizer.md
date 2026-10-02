<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tokenizer, chat templates and sampling

M3's native text front end (plan.md, "Tokenizer and chat templates";
D-067, D-088): a byte-level BPE tokenizer for the M3 models, native
family renderers for their chat templates and a bounded, sandboxed
Jinja-subset interpreter for any other template, stop tokens, and greedy and
seeded sampling. All of it is CPU code with no vendor types and builds in
every profile. D-088 (accepted 2026-09-28) clears its Unicode tables for
shipped binaries, and `jitllm-runtime` links them: its serving commands
and the chat route (runtime-serving.md) tokenize, render and sample with
these modules. A package built from the tables lists Unicode-3.0 and
carries its notice ([licensing.md](licensing.md#tokenizer-unicode-tables-m3)).

## Modules

| Module | Holds |
| --- | --- |
| `base/json.h` (`jitllm_json`) | General RFC 8259 JSON from untrusted bytes: strict UTF-8, unescaped strings, numbers kept as text, duplicate keys refused, caps on size, depth, values and string bytes |
| `tokenizer/` (`jitllm_tokenizer`) | `unicode.h` (UCD 15.1.0 properties, strict UTF-8, U+FFFD replacement, NFC), `pretokenize.h` (the three pre-tokenizers), `tokenizer.h` (vocabulary validation, encode, decode, streaming decode), `gguf.h` and `hf.h` (the readers of a GGUF file's tokenizer metadata and of `tokenizer.json`) |
| `chat/` (`jitllm_chat`) | `chat.h` (the conversation, the native renderers and their registry, `ChatTemplate`: the choice by hash, probe or interpreter; stop tokens, Python's `str.strip`), `jinja.h` (the Jinja-subset interpreter: `jinja_parse.cc`, `jinja_eval.cc`, `jinja_value.cc`), `pyjson.h` (JSON as Python's `json.dumps` prints it) |
| `execution/sampling.h` (`jitllm_sampling`) | Greedy and seeded sampling |

Both new modules sit in the model layer (architecture.md), `tokenizer`
first: it needs only `base`; `chat` needs `tokenizer` and `jitllm_json`.

## The tokenizer

Encoding follows what llama.cpp (the pinned `b29c606e`) and Hugging Face
tokenizers do for these vocabularies:

1. **Special tokens** are matched in the raw text, leftmost-longest.
   User-defined tokens (Hugging Face's non-special added tokens, such as
   Qwen's `<think>` and `<tool_call>`) always match; control tokens match
   only with `SpecialTokens::kParse`, or where a renderer marked them
   (`EncodeMarked`), so message content never becomes a control token.
   llama.cpp instead extracts special tokens longest first; the two agree
   unless a token ends with the start of another at least as long, which
   no M3 vocabulary has (a models test checks it).
2. **Normalization:** NFC for the tokenizer.json vocabularies that declare
   it (both Qwen files), none otherwise.
3. **Pre-tokenization** splits each fragment natively; no regex engine
   runs. Each pre-tokenizer is identified by name (GGUF) or by the exact
   expressions of its Split stages (`tokenizer.json`); anything else is
   refused, never approximated:

   | Pre-tokenizer | Names | Expression |
   | --- | --- | --- |
   | `kQwen2` | GGUF `qwen2`; Qwen2Tokenizer's Split | `(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\r\n\p{L}\p{N}]?\p{L}+\|\p{N}\| ?[^\s\p{L}\p{N}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+` |
   | `kQwen35` | GGUF `qwen35` (Qwen3.5, Qwen3.8) | as `kQwen2`, with `[\p{L}\p{M}]+` words and marks out of the symbol run |
   | `kDeepSeekV3` | GGUF `deepseek-v3`, `joyai-llm` (DeepSeek V3, V4) | three stages: `\p{N}{1,3}`, `[一-龥぀-ゟ゠-ヿ]+`, then ``[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{\|}~][A-Za-z]+\|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+\| ?[\p{P}\p{S}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+`` |

   Semantics are a backtracking regex's (the first alternative matching
   at the leftmost position), which both references use; text a stage
   does not match stays one piece. `\s` is White_Space and `\p{..}` the
   general category, from UCD 15.1.0. The contractions match ASCII letters
   of either case only: both references leave `'ſ` unmatched.
4. **BPE** on each piece's bytes in the GPT-2 byte alphabet: the lowest
   merge rank first, the leftmost on a tie, until no ranked pair remains.

**Byte fallback.** Creation requires a normal token for every byte and a
token for every merge's result, so every well-formed text encodes and
nothing is dropped (llama.cpp silently drops bytes a vocabulary lacks).

**Bounds and errors.** Input is untrusted: text must be well-formed UTF-8
(Unicode table 3-7) and is otherwise refused with `kInvalidUtf8` and the
offset of the first ill-formed sequence; `ReplaceInvalidUtf8` (maximal
subparts to U+FFFD, as Python's `replace` and WHATWG do) is the caller's
explicit lossy choice. `EncodeOptions` bounds the input (4 MiB by default)
and the output (2^22 tokens). Memory is linear in the input and work is
O(n log n) (NFC sorts combining runs, BPE uses a heap), except that
special-token matching may take up to 256 trie steps per byte: on spark-b
4 MiB takes 0.1–1.1 s with the real vocabularies and about 7 s with a
hostile one built for the worst case. Vocabulary caps:
2^22 tokens of at most 1,024 bytes, special tokens of at most 256 bytes.
Every refusal is a named `Rule` with a static reason and an item.

**Decoding** gives each token's bytes: normal tokens through the byte
alphabet, user-defined tokens as their text, control tokens only when asked,
unused padding tokens as nothing. `StreamDecoder` holds back a trailing
partial UTF-8 sequence and replaces what can never complete.

**Readers.** `ReadGgufTokenizer` parses a GGUF header's key-value section
from a byte prefix of the file: counts and lengths are checked against the
bytes left before anything is allocated, and the keys it uses against caps
(2^16 keys, 16 MiB strings, 2^24 array elements); other keys are skipped
unread; keys may not repeat, types must match. It accepts
`gpt2` vocabularies with the four pre-tokenizer names above and token types
normal, control, user-defined and unused. `ReadHfTokenizer` accepts the
`tokenizer.json` shape hf.h lists (BPE, NFC or no normalizer, the Split and
ByteLevel sequences above, no stripping flags) and refuses the rest. The
prepared artifact's tokenizer section is still the importer's to define;
carrying the checkpoint's own tokenizer data and reading it with these
readers is the suggested route.

## Unicode tables

`src/tokenizer/unicode_data.cc` is generated by `tools/gen-unicode-tables`
from UCD 15.1.0's UnicodeData.txt, PropList.txt and
DerivedNormalizationProps.txt, each pinned by SHA-256 in the tool (D-088):
general category, White_Space and NFC_QC per code point as two-stage
tables, canonical combining classes, full canonical decompositions and the
primary composites. 15.1.0 is the version llama.cpp's own tables reproduce
exactly (licensing.md), so the pre-tokenizers classify every code point as
the oracle does. On `spark` and `spark-b` the files are under
`~/.local/share/jitllm/ucd/15.1.0/`, with NormalizationTest.txt, which a
models test runs in full; a tools test regenerates the tables there and
compares them with the checked-in file.

## Agreement with the references

The corpus (`tests/unit/data/tokenizer/corpus.json`, 92 items) covers
English, contractions, numbers in several scripts, whitespace runs of every
kind, control and zero-width characters, code (Python, C++, JSON, HTML,
Markdown), eleven scripts with combining marks and decomposed forms, CJK
extensions, emoji sequences (ZWJ, flags, keycaps, skin tones, and Unicode
16.0 emoji), private use, unassigned code points and noncharacters, special
tokens whole, partial, adjacent and overlapping, long runs up to 3,000
characters, and 9 ill-formed UTF-8 inputs. Each item is encoded with and
without parsing special tokens (184 encodings per configuration):

| Configuration | Source (under `~/.local/share/jitllm/` on the Sparks) | Reference | Result |
| --- | --- | --- | --- |
| `deepseek-v4-0731-gguf` | `models/unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93/UD-Q2_K_XL/…-00001-of-00003.gguf`; 129,280 tokens, `joyai-llm` | llama.cpp `llama-tokenize` (the pinned image) | 184 of 184 equal |
| `qwen3.8-gguf` | `reference-models/Q/UD-IQ3_XXS/…-00001-of-00003.gguf`; 248,320 tokens, `qwen35` | llama.cpp | 184 of 184 |
| `qwen3.8-nvfp4` | `models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6/tokenizer.json`; NFC, qwen35 | Hugging Face tokenizers 0.22.2 | 184 of 184 |
| `qwen-image-2.1` | `models/Qwen/Qwen-Image-2.1@790c9263/processor/tokenizer.json` (Qwen3-VL-8B); NFC, qwen2 | tokenizers 0.22.2 | 184 of 184 |
| `deepseek-v4-0731-hf` | `tokenizer-reference/deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b/tokenizer.json` (DeepSeek's own) | tokenizers 0.22.2 | 182 of 184; the 2 others are a known divergence |

Ill-formed items are refused at the offset of their first ill-formed byte;
their replaced text then agrees with the reference's tokens of Python's
`replace` decoding. Decoding returns every non-normalizing encoding's text
exactly. Where the references disagree with each other, jitLLM follows each
model's oracle and records why:

- **Unicode version.** Hugging Face's Oniguruma knows Unicode 16.0's emoji
  (U+1FAE9, U+1FAC6) as symbols; UCD 15.1 and llama.cpp leave them
  unassigned, which the DeepSeek expression treats differently. jitLLM
  follows llama.cpp, DeepSeek's oracle; the `deepseek-v4-0731-hf` test
  asserts both the divergence and llama.cpp's result.
- **NFC.** The Qwen tokenizer.json files normalize to NFC and llama.cpp
  does not; 6 corpus items differ between `qwen3.8-gguf` and
  `qwen3.8-nvfp4` for that reason. M3's Qwen3.8 is the NVFP4 checkpoint, so
  its oracle, vLLM, tokenizes with NFC.
- **Special-token kinds.** DeepSeek's `tokenizer.json` marks `<｜User｜>`,
  `<｜Assistant｜>` and many others non-special, where the GGUF has them as
  control tokens; Qwen's fim and repo tokens likewise. Each configuration
  follows its own file.

The DeepSeek 0731 GGUF's tokens, types, merges and special IDs equal those
of the `e3aa0d6a` GGUF it replaced on the Sparks; only the chat template
changed. The Qwen3.8 GGUF and NVFP4 `tokenizer.json` share tokens and merges
(the GGUF adds 243 unused padding tokens and marks 6 more as control).

## Chat templates

A model's template renders one of three ways (D-067 as amended
2026-10-02; `chat::ChatTemplate::ForText`, when the model registers):

1. **Native by hash:** a native renderer whose fixtures pin the template's
   SHA-256 (the table below).
2. **Native by probe:** a native renderer whose output equals the
   template's own, rendered by the interpreter, on every conversation of
   the probe corpus (`ProbeEquivalent` in `chat/template.cc`: 23
   conversations, each with thinking unset, on and off, each accepted
   reasoning effort and `preserve_thinking` off; both refusing counts as
   equal, a case the renderer does not implement is skipped, and at least
   30 renderings must be equal). Each probe renders under 500,000 steps
   and 16 MiB of work, so a template that matches a family's text but
   works hard besides (or burns its budget on purpose) differs at its
   first probe instead of costing registration hundreds of full
   renderings; and all the probes of one registration (every family's,
   and the end-of-turn probe) draw on one pool of a single rendering's
   bounds, so a copy that stays just under each probe's budget, probe
   after probe, spends the pool and is interpreted. A repackaged copy of a supported template
   (other spacing, comments or line ends) renders natively; any change to
   what it renders falls through to the interpreter.
3. **Interpreted:** the template itself, through the interpreter below.

The runtime logs which, with the template's hash, unless the hash pins a
renderer. A template the interpreter cannot parse (a construct, filter or
test outside the subset, a bound) is refused at registration, naming its
hash, as before.

The native renderers reproduce their templates byte for byte:

| Template | SHA-256 | Where | Reference renderer |
| --- | --- | --- | --- |
| DeepSeek V4 Flash 0731 | `e643c31fcec17f342f72296e02c46d35846bf4c70f6a0271f23bad73fd4eb645` | The 0731 GGUF's `tokenizer.chat_template` (Unsloth's port of DeepSeek's encoder) | transformers 5.12.1 `apply_chat_template` (Jinja2 3.1.6), and DeepSeek's `encoding_dsv4.py` at `7872f01b` where it defines the case |
| DeepSeek V4 Flash community "chat-v2" | `872492071c22c8d2025238120309ffbddddb666b49f4433f55c19b69bf51af27` | The `tokenizer.chat_template` (5,016 bytes) of `antirez/deepseek-v4-gguf@f71f23d5`'s `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` | transformers 5.12.1 `apply_chat_template` with Python's `json.loads` as the `from_json` filter it lacks |
| Qwen3.8 Flash Next | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` | `chat_template.jinja` of `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6` | transformers 5.12.1 `apply_chat_template` |
| Qwen-Image 2.1 text-to-image prompt | none: a fixed string | diffusers `8b3c707e`'s `QwenImage21Pipeline` | the pipeline's string, through the processor's tokenizer |

**Unsloth's Qwen3.8 GGUF variant.** `unsloth/Qwen3.8-Flash-Next-GGUF` and
`unsloth/Qwen3.8-27B-GGUF` embed a variant (`12827f24…`) of the
checkpoint's template: leading system messages (any number) merged, each
trimmed, joined by a newline, empty ones skipped; `reasoning_effort: high`
taken as `xhigh`; no error without a user query (the last message stands
in); blank string tool-call arguments render no parameters. Its renderer
(`RenderQwen38Unsloth`, the same family code with these options) has no
fixtures of its own, so no hash pins it: it is chosen by probe
equivalence, and the corpus test checks it against transformers on both
GGUFs' templates.

**DeepSeek's authority.** DeepSeek publishes no template, only
`encoding_dsv4.py`, which defines the format; the GGUF's embedded template
is Unsloth's port of it, and its hash is what the runtime sees and selects
the renderer by. The renderer follows the template, and the fixtures hold
both: on every case the encoder defines (17 of 20), its text equals the
template's. The template also defines what the encoder cannot express, such
as request-level tools without a system message, and there it rules. Two
refinements: an unknown `reasoning_effort` is refused (the template ignores
it; the encoder asserts), and the older `e3aa0d6a` GGUF's template
(`d05566eb…`) has no renderer.

**DeepSeek "chat-v2".** The community IQ2_XXS GGUF embeds a shorter,
independent template with the same tokens, tool header, schemas and DSML
calls. Its own renderer (`RenderDeepSeekV4ChatV2`, the same file) shares
those parts and differs where the template does:

| | 0731 (`e643c31f…`) | chat-v2 (`87249207…`) |
| --- | --- | --- |
| Reasoning effort | `high`/`max` prefix while thinking; others refused | none: any value renders as none |
| Earlier reasoning while thinking | kept only after the last user message, or with tools | every assistant turn's kept |
| Thinking, empty reasoning (where kept) | `<think></think>` | `</think>` |
| Consecutive user and tool messages | one `<｜User｜>`, joined by blank lines | a `<｜User｜>` per user message and per run of tool results, which join with no separator |
| `<｜Assistant｜>` | after a user or tool message | after one since the last assistant message (a system message between does not reset it) |
| Generation prompt | always, when asked | only after a user or tool message |
| Tools with empty system text | a blank line before the header | none |
| Tool header and footer | "the user's question"; footer `\nYou MUST …calls.\n` | "the user question"; footer `\n\nYou MUST …calls.` |

Both read `thinking` (jitLLM's `enable_thinking`), off by default, end
turns with `<｜end▁of▁sentence｜>` and render a tool call's arguments as
DSML parameters; chat-v2 parses string arguments with `from_json`, which
equals rendering the client's string parsed (fixtures `string-arguments`,
and `string-arguments-not-object`, which both refuse).

**Supported.** DeepSeek V4: roles system, user, assistant and tool;
request-level tools, DSML tool calls, reasoning, `enable_thinking` (the
template's `thinking`) and `reasoning_effort` (0731: low, high, max;
chat-v2: ignored, as its template ignores it). Its
developer and latest_reminder roles, tasks and response formats are refused
as unsupported. Qwen3.8: system (first only), user, assistant and tool; tools
and XML tool calls, `<think>` blocks, `enable_thinking`, `reasoning_effort`
(xhigh, medium, low) and `preserve_thinking`. Tool-call arguments are JSON
objects (the client's string, parsed, as vLLM passes them), printed with
Python's `json.dumps` spacing and float formatting. Text content blocks are
joined before rendering; images are unsupported. What a template itself
raises (no user query, a late system message) is `kInvalid`.

**Content never becomes a control token.** A renderer marks the control
tokens it places, and `EncodeMarked` encodes only those as control tokens;
the references tokenize the whole rendered text, so a message containing
`<|im_end|>` becomes that token there but stays text here. User-defined
tokens (`<think>`, `<tool_call>`, DeepSeek's `｜DSML｜`) match in content in
both, as the references do. So the protection is only as good as the
file's kinds: DeepSeek's own `tokenizer.json` makes `<｜User｜>` and
`<｜Assistant｜>` user-defined, so with it message content can still forge
role tokens (as it can under Hugging Face and vLLM); the GGUF, M3's DeepSeek
source, makes them control tokens.

**Fixtures:** 21 cases per LLM template (conversations with tools, tool
calls and results, reasoning, options, Unicode, whitespace trimming, tags in
content, and the template's refusals), 41 for chat-v2 (the DeepSeek cases
and its differences above, string arguments and interleaved tool results
among them), and 4 image prompts. Text compares
byte for byte in every profile (`chat_test`); token IDs compare with the
references on the model files (`tokenizer_models_test`): all equal, with
DeepSeek's against llama.cpp (for chat-v2, llama.cpp on the community GGUF
and the native tokenizer from its artifact's kept metadata, which equals
the 0731 GGUF's) and against Hugging Face.

**Stop tokens:** DeepSeek V4 (both templates) `<｜end▁of▁sentence｜>`;
Qwen3.8 `<|im_end|>` and `<|endoftext|>` (its generation_config.json's
`eos_token_id`).

For an interpreted template, the stop tokens are the control token it
places after an assistant message's content (rendering a two-message
probe), and the tokenizer's EOS. That is a heuristic for one-channel
formats: a format whose turns end in more than one way, or that also
closes channels inside a turn (harmony's `<|call|>`, `<|return|>` and
`<|end|>`), needs its family's own stop rules before it serves.

### The template interpreter

`chat/jinja.h` renders what Hugging Face transformers' `apply_chat_template`
renders: Jinja2 3.1 under its `ImmutableSandboxedEnvironment` with
`trim_blocks`, `lstrip_blocks` and the loop controls, one trailing newline
of the template dropped, transformers' `tojson` (no HTML escaping;
`ensure_ascii`, `indent`, `separators`, `sort_keys`), `raise_exception`,
`strftime_now` (the runtime's local time) and the `{% generation %}` tag,
plus llama.cpp's `from_json` filter. The template sees what transformers
passes: `messages` (role, content or none, `reasoning_content` when given,
OpenAI-shaped `tool_calls` with object arguments, as vLLM passes them),
`tools` and `documents` (none when absent), `add_generation_prompt`,
`bos_token` and `eos_token` (the tokenizer's), and the options a
conversation sets (`enable_thinking`, also as `thinking`;
`reasoning_effort`; `preserve_thinking`).

| Part | The subset |
| --- | --- |
| Statements | `if`/`elif`/`else`; `for` with tuple targets, an `if` filter, `else`, `loop.*` (index, index0, revindex, revindex0, first, last, length, previtem, nextitem, depth, depth0, cycle), `break`, `continue`; `set` (names, tuples, a namespace's attribute, blocks with filters); `macro` with defaults, `varargs` and `kwargs`; `raw`; `generation`; comments and `-`/`+` whitespace control |
| Expressions | literals (Python string escapes, adjacent strings joined), lists, tuples, mappings with string keys; attributes and items (Python attributes first for `.`, items first for `[]`), slices, calls; `+ - * / // % **` and `~` with Jinja2's precedence; comparisons, `in`, `not in`, `and`, `or`, `not`, `x if c else y` |
| Filters | abs, capitalize, count, default/d, dictsort, escape/e, first, float, format (`%s %d %i %%`), from_json, indent, int, items, join, last, length, list, lower, map, max, min, reject, rejectattr, replace, reverse, round, safe, select, selectattr, sort, string, sum, title, tojson, trim, unique, upper |
| Tests | boolean, callable, defined, divisibleby, eq/equalto/==, escaped, even, false, float, ge/>=, gt/greaterthan/>, in, integer, iterable, le/<=, lessthan/lt/<, lower, mapping, ne/!=, none, number, odd, sameas, sequence, string, true, undefined, upper |
| Methods | strings: strip, lstrip, rstrip, split, rsplit, splitlines, startswith, endswith, replace, upper, lower, title, capitalize, find, rfind, count, join, removeprefix, removesuffix; mappings: get, keys, values, items, copy; lists: count, index, copy |
| Globals | range (at most 100,000 items, as the sandbox), namespace, dict, raise_exception, strftime_now |

Refused when parsed: `include`, `import`, `extends`, `block`, `call`,
`filter` blocks, `with`, recursive loops, `*args`, unknown filters and
tests, a call naming a keyword twice (as Python's compiler refuses it).
Refused when run (the request fails): a mutating method (as the
sandbox raises), another Python method, an integer beyond 64 bits, a
strftime directive other than `%Y %y %m %d %e %H %I %M %S %j %p %B %b %A
%a %%` (with `-` for no padding). Where Jinja2 raises (an undefined value's
attribute, a type error, `raise_exception`), rendering returns `kInvalid`;
a bound or an unimplemented feature returns `kUnsupported`. The bounds are
D-067's, in `chat::jinja::Limits`. A value's text or JSON stops at the
string bound as it grows, so one string held many times over costs no
more than that; comparisons charge every byte they compare; substring
searches (`in`, `find`, `rfind`, `count`, `split`, `rsplit`, `replace`)
are linear (Knuth-Morris-Pratt), with needles up to a sixteenth of the
live-bytes bound; searches by name (a scope's variables, a macro's
parameters, a namespace's or mapping's members, a call's keyword
arguments, a mapping literal's earlier keys) charge each entry they pass
and every byte they compare, and a lookup in an indexed mapping the key's
bytes; scans that build nothing (prefix and suffix tests, case tests,
`from_json`, `int`, `float`, `%` and `strftime_now` formats, `tojson`'s
key sorts) are charged as scanned; a loop's own copy of what it iterates
is held as live bytes while it runs; the scan for the control tokens a
rendering places is charged to its work. Containers release what they
hold without recursing, so no chain of them can exhaust the stack.

Known narrowings, none met by the corpus: case mapping (`upper`, `lower`,
`title`, `capitalize`) is ASCII-only; filters Jinja2 returns as generators
(`map`, `select`, `items`, `reverse`) are lists here, so printing one
directly differs; attributes of numbers are undefined; a loop object
prints as an error.

**Provenance.** Every rendered byte is the template's or the input's. A
template literal, the BOS and EOS texts and numbers the template computes
are the template's; message content, roles, tool schemas, arguments and
mapping keys from the client are not, through concatenation, slicing,
`strip`, `replace`, `join`, `split`, `tojson` and case mapping, nor as a
filter's formatting (`tojson`'s indent and separators, `indent`'s width):
a value whose text the client chose never becomes the template's. Numbers
and booleans the template computes are its own even when computed from
client values (a message count, a length, a comparison), so a control
token spelled with digits the template prints could be steered; no corpus
template builds one. The rendering
marks the vocabulary's control tokens that lie wholly in the template's
bytes (leftmost, longest), so `EncodeMarked` makes only those control
tokens, as with native renderers. An interpreted rendering reports one
boundary: where the generation prompt starts, when the rendering without
it is a prefix of the rendering with it; serving uses it for its turn
checkpoint only when a control token starts there.

**Evidence** ([chat-template-corpus](experiments/chat-template-corpus/README.md)):
`jinja_test` compares 133 snippet templates (`jinja-snippets.json`) with
transformers 5.12.1 (Jinja2 3.1.6), text and errors, and drives every
bound with hostile templates and a 20,000-mutation fuzz;
`chat_template_test` checks the choice of renderer and that the
interpreter reproduces every case of the three pinned templates' fixtures;
`chat_corpus_test` (label `models`, the corpus on the Sparks) renders 29
real templates on 19 conversations each, 537 renderings and every
refusal equal to transformers', through the interpreter and through the
chosen renderer.

## Sampling

`Greedy` takes the highest logit, the lowest ID among equals. `Sample`
applies temperature (0 is greedy), top-k, softmax, min-p and top-p in that
order and draws with one uniform number from Philox4x32-10 keyed by the
seed over (stream, position), so a draw depends only on its key: a
speculative verifier can redraw any position, and a restored request
continues exactly. NaN or +infinity logits and out-of-range parameters are
refused. The Philox implementation passes Random123's known-answer vectors.

With `top_k = 1`, both `Sample` and `VerifyDraft` take the greedy path
after parameter validation: the highest logit is the only kept token,
whatever the valid temperature, min-p or top-p. They still validate every
logit and break ties by the lower ID, but build no candidates and draw no
random numbers. Verification accepts exactly that token and otherwise
returns it as the replacement.

Measured on `spark-b`'s CPU, `spark-native` RelWithDebInfo with SDK
`aarch64-e0a0c85c42806fb1`, before (`a9c3e93`) and after the shortcut
(2026-09-29): `jitllm_sampling_bench --draws 200`, row seed 42, three
alternating before/after passes. These are medians of the pass medians,
including parameter and logit validation, at temperature 1 and top-k 1:

| Vocabulary | Synthetic row | Sample, before → after (ms) | VerifyDraft, before → after (ms) |
| --- | --- | --- | --- |
| DeepSeek, 129,280 | peaked | 0.192 → 0.154 | 0.192 → 0.154 |
| DeepSeek, 129,280 | flat | 0.193 → 0.168 | 0.193 → 0.169 |
| Qwen3.8, 248,320 | peaked | 0.365 → 0.296 | 0.365 → 0.296 |
| Qwen3.8, 248,320 | flat | 0.367 → 0.311 | 0.367 → 0.311 |

`Greedy` reuses the maximum from validation and finds its first
occurrence, instead of computing the argmax again: temperature-0 draws
fell from 0.369 to 0.154–0.168 ms at DeepSeek's vocabulary and from
0.710 to 0.296–0.310 ms at Qwen3.8's. These are sampler timings on
synthetic rows, not end-to-end decode measurements.

A draw never sorts the vocabulary. The weights are float exponentials
relative to the most likely token's; the uniform number picks a token by
inverse CDF over the kept tokens: in ID order without truncation; top-k
keeps the k highest logits (ties to the lower ID) in one pass that cuts
its buffer back as it fills; min-p is a threshold; top-p buckets the
weights by their bits (eight a binary octave) and sorts only the bucket
its boundary falls in (ties to the lower ID). Speculative verification
(`VerifyDraft`) builds the same distribution, so plain and speculative
decoding draw identically distributed tokens. On `spark-b`
(`jitllm_sampling_bench`, synthetic peaked and flat rows, median of 200
draws), a temperature-1 draw took 5.9 ms at DeepSeek's 129,280 logits and
11.9 ms at Qwen3.8's 248,320 with the full sort it replaced; now 0.45 ms
and 0.87 ms (a speculative verdict the same; top-p 0.95 0.68–1.0 ms and
1.5–2.0 ms from a peaked row to a flat one; top-k 0.20 and 0.37 ms).
Through the chat route on `spark-b` (384 tokens from a 27- or 77-token
prompt, two seeds), temperature 1 went from 19.1 to 21.4 tok/s for
DeepSeek's plain decode (greedy 21.7), from 18.8 to 26.1 for Qwen3.8's
(greedy 27.3), and speculative from 23.9 to 27.1 (DeepSeek, greedy 29.8)
and from 24.5 to 32.2 (Qwen3.8, greedy 43.5; sampled drafts are accepted
less often); the rates include the short prefill. With plain sampling
within 1–5% of greedy, the step's logits row copied to the host and
sampled there, a device-side softmax was not pursued.

**The seed-to-token mapping changed** when the sort went (M3, 2026-09-29;
D-085's performance over bit-exactness at the same quality). It is a
change of bits only: the same distribution, the full softmax with the
requested top-k, top-p and min-p exactly (no candidate pool, no
approximate cutoff; the weights differ from the old double-precision
ones by float rounding), but a given seed maps to different tokens than
before. The tokens a seed gives are fixed within a build, not promised
across builds. The evidence is `sampling_test`'s distribution tests
(every filter and their mix, plain and speculative, against a full-sort
double-precision reference) and the harnesses' sampled-speculation
checks, rerun on `spark-b` after the change (256 seeds, 8 tokens, 4
prompts; total variation, bound 0.1): DeepSeek 0.0034, 0.0186, 0.0381,
0.0142; Qwen3.8 0.0312, 0.0146, 0.0220, 0.0337.

## Reference generation

`docs/experiments/tokenizer-reference/` holds the generator and its pins;
its README says how to rerun it on a Spark. The generated fixtures are under
`tests/unit/data/`. Tests needing model files carry the `models` label and
skip where the files are absent; a present file with another hash fails.

## Not yet done

- Tool calls: parsing them out of generated text per family (DeepSeek's
  DSML, Qwen's XML), and rendering the tool definitions a client sends
  (the chat route refuses `tools` for now; runtime-serving.md). The
  reasoning split and client stop strings are the chat route's
  (`runtime/api.h`), as is the text subset of an OpenAI request.
- The prepared artifact's tokenizer and template section (the importer):
  a GGUF-sourced artifact carries them in its source metadata; Qwen3.8's
  are configured as files beside its artifact (runtime-serving.md).
- Segment boundaries for cache breakpoints inside content blocks (D-067):
  native renderers report prefix, message and generation-prompt
  boundaries only, an interpreted template the generation prompt's only.
- Tool-call output parsing for interpreted templates: they render tool
  definitions and calls, but the families they belong to have no output
  parser yet (with the chat route's `tools` refusal, unused for now).
