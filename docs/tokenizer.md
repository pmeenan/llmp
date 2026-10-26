<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tokenizer, chat templates and sampling

The native text front end (M3 and M3.5; plan.md, "Tokenizer and chat templates";
D-067, D-088): byte-level BPE for the M3 models, raw UTF-8 BPE
for GGUF and Hugging Face vocabularies, scored SentencePiece BPE from GGUF
or standalone ModelProto files, native
family renderers for their chat templates and a bounded, sandboxed
Jinja-subset interpreter for any other template, stop tokens, and greedy and
seeded sampling. All of it is CPU code with no vendor types and builds in
every profile. D-088 (accepted 2026-09-28) clears its Unicode tables for
shipped binaries, and `llmp-runtime` links them: its serving commands
and the chat route (runtime-serving.md) tokenize, render and sample with
these modules. A package built from the tables lists Unicode-3.0 and
carries its notice ([licensing.md](licensing.md#tokenizer-unicode-tables-m3)).

## Modules

| Module | Holds |
| --- | --- |
| `base/json.h` (`llmp_json`) | General RFC 8259 JSON from untrusted bytes: strict UTF-8, unescaped strings, numbers kept as text, duplicate keys refused, caps on size, depth, values and string bytes |
| `tokenizer/` (`llmp_tokenizer`) | `unicode.h` (UCD 15.1.0 properties, strict UTF-8, U+FFFD replacement, NFC), `regex.h` (bounded compiled expressions), `pretokenize.h` (ordered split plans), `presets.cc` (GGUF preset data), `tokenizer.h` (vocabulary validation, encode, decode, streaming decode), `gguf.h`, `hf.h` and `sentencepiece.h` (readers of GGUF tokenizer metadata, `tokenizer.json` and standalone `tokenizer.model`) |
| `chat/` (`llmp_chat`) | `chat.h` (the conversation, the native renderers and their registry, `ChatTemplate`: the choice by hash, probe or interpreter; stop tokens, Python's `str.strip`), `jinja.h` (the Jinja-subset interpreter: `jinja_parse.cc`, `jinja_eval.cc`, `jinja_value.cc`), `pyjson.h` (JSON as Python's `json.dumps` prints it, and values as `str()` prints them); the family renderers `deepseek_v4.cc`, `qwen.cc` and `gemma.cc` |
| `execution/sampling.h` (`llmp_sampling`) | Greedy and seeded sampling |

Both new modules sit in the model layer (architecture.md), `tokenizer`
first: it needs only `base`; `chat` needs `tokenizer` and `llmp_json`.

## The tokenizer

Byte-level BPE encoding follows each source vocabulary. Historical agreement
fixtures retain llama.cpp `b29c606e`; new Llama 3.2 evidence uses the locked
`d8123504` reference (the reference-generation README records both pins):

1. **Special tokens** are matched in the raw text, leftmost-longest.
   User-defined tokens (Hugging Face's non-special added tokens, such as
   Qwen's `<think>` and `<tool_call>`) always match; control tokens match
   only with `SpecialTokens::kParse`, or where a renderer marked them
   (`EncodeMarked`), so message content never becomes a control token.
   llama.cpp instead extracts special tokens longest first; the two agree
   unless a token ends with the start of another at least as long, which
   no M3 vocabulary has (a models test checks it). The approved Gemma 2
   and 3 vocabularies also contain overlapping homogeneous whitespace
   runs; both matching orders consume the longest run from the left.
2. **Normalization:** NFC for the tokenizer.json vocabularies that declare
   it (both Qwen files), none otherwise.
3. **Pre-tokenization** compiles source expressions once into immutable
   bytecode. HF `Split(Regex, Isolated)` stages and nested `Sequence` stages
   flatten into an ordered split plan; every later stage runs within each
   preceding piece. Unmatched gaps remain pieces. `ByteLevel` must be last:
   `use_regex=true` (the default) adds GPT-2 splitting, and false adds none.
   `add_prefix_space=true` (also the default if absent) is explicitly refused;
   false is supported. Offset-only flags are type checked. Unsupported fields,
   split behaviors, inversion and byte conversion in the middle are refused.

   GGUF names select expression lists and semantic defaults from a data table
   transcribed from llama.cpp `d81235049384534c167caea52b85a694f6103d14`,
   `src/llama-vocab.cpp`. This includes `llama-bpe`/`llama3`/`llama-v3`,
   `gpt-2`/`phi-2`, `gpt-4o`, `tekken`, `starcoder`, `falcon`, `command-r`,
   `qwen2`/`qwen35`, `deepseek-llm`/`deepseek-coder`/`deepseek-v3` and
   `joyai-llm`, with the pinned aliases recorded in `presets.cc`. Unknown
   names refuse. Explicit `default`, an empty name and absent byte-BPE `pre`
   select the pin's four-stage BPE default; this is distinct from GPT-2 and
   SentencePiece, and no architecture inference is used. Llama 3 and Tekken default to whole-piece direct vocabulary
   lookup before merges and automatic BOS; explicit GGUF BOS/EOS booleans
   override defaults, including false. HF supplies expressions directly:
   a new expression within the supported grammar needs no family code.

   Matching chooses the first successful alternative at the leftmost position.
   The grammar includes UTF-8 literals, escaped metacharacters and whitespace,
   dot, positive/negative classes and ranges, all 30 UCD general categories
   and their L/M/N/P/S/Z/C aggregates with `\p`/`\P`, `\s`/`\S`, ordered
   alternation, capturing/noncapturing grouping (captures are not returned),
   scoped `(?i:)`, greedy/lazy/possessive `? * + {m} {m,n} {m,}`, `^`,
   `$`, `(?!\S)`, and positive lookahead for one character class,
   `(?=[...])`. HF anchors are Oniguruma line anchors: `^` also matches
   after LF and `$` before LF; `\A` and `\z` are absolute. GGUF presets
   retain the locked llama.cpp fallback's non-multiline ECMAScript anchors
   over each stage fragment. HF classes support nested union and `&&`
   intersection; escaped ampersands remain literals. GGUF classes retain
   literal ampersands. Class algebra uses explicit source-sized compiler and
   matcher stacks and pays the shared transition allowance. `(?m:)` remains
   unsupported: Oniguruma uses that flag for dotall, not line-anchor selection.
   `\s` is UCD 15.1.0 White_Space.
   Case-insensitive literal runs use UCD 15.1.0 default full folding: `s`
   matches long-s, `k` matches Kelvin sign, and the literal run `ss` matches
   sharp-s. Folding consumes complete original code points and does not join
   across classes, repetition atoms, captures or scoped groups: `s{2}` and
   `[s][s]` do not match sharp-s. ASCII case-insensitive classes use single
   code-point fold equivalence; non-ASCII caseless classes, caseless category
   properties, backreferences, other assertions, scripts and unknown escapes
   refuse explicitly. Independent pinned tokenizers 0.22.2/Oniguruma probes
   establish these literal-run and group-boundary controls; Python case
   conversion is not the source of the folding data or the regex oracle.
   A successful empty match establishes a boundary without emitting an empty
   piece, then advances search by one code point. Unmatched input is retained;
   every emitted piece is nonempty and their lengths sum to the input. There are no separate handwritten family matchers.

   GGUF presets use the actual runtime expressions at the locked source,
   including ASCII contraction classes and GPT4O/Tekken's category/class
   lookahead workaround. HF expressions retain their own Unicode semantics;
   neither comment expressions nor a family label replace them. For example,
   HF Qwen's contraction alternative splits `a'ſz` as `a`, `'ſ`, `z`, while
   the GGUF preset's explicit ASCII classes retain `'ſz` as one word piece.
   This corrects the previous HF ASCII-only case matching on such inputs.
4. **BPE** on each piece's bytes in the GPT-2 byte alphabet: the lowest
   merge rank first, the leftmost on a tie, until no ranked pair remains.

**Gemma 4 GGUF BPE.** `tokenizer.ggml.model = "gemma4"` selects raw
UTF-8 BPE; its optional `tokenizer.ggml.pre` must also be `gemma4`.
Spaces become U+2581 (`▁`), without NFC, a dummy prefix or whitespace
removal. Pieces are newline runs and runs without newlines; their initial
symbols are Unicode code points, with `<0xNN>` byte tokens for absent
code points. Ranked merges use the same heap and leftmost tie rule as
byte BPE. A whole newline run present in the vocabulary becomes that
token directly, matching the pinned llama.cpp Gemma 4 shortcut. Decoding
normal tokens replaces `▁` with spaces and byte tokens with their bytes;
a literal `▁` in input therefore decodes to a space, as in the reference.
Byte coverage and the normal `▁` token are required; merges must
name normal constituents and a normal result. BOS/EOS and marked control
tokens retain their existing rules. The raw BPE engine also reads the official Gemma 2 and 3 Hugging Face
`tokenizer.json` pipeline. A normal model token for an unchanged ASCII
character can supply byte coverage when its `<0xNN>` fallback is absent:
the official Gemma 2 JSON stores tab this way. That token keeps its normal
kind and participates in merges. Literal spaces cannot substitute because
normalization transforms them; non-ASCII characters cannot substitute for
individual UTF-8 fallback bytes. It validates the declared space-to-`▁` Replace,
then Split on literal spaces with MergedWithPrevious behavior (there are
no literal spaces after replacement), and the decoder's inverse Replace,
ByteFallback and Fuse stages. This pipeline runs on whole fragments,
including across newlines, while GGUF raw BPE retains its newline split
and whole-newline-token shortcut. Model membership is independent of added
token matching: a non-special added whitespace token can still participate
in a declared BPE merge. Control tokens remain excluded from model merging.
The declared HF decoder replaces markers in added tokens as well. Raw-BPE
added tokens retain their normalization flag: raw matching runs before space
replacement, and normalized matching runs afterward, freezing each longest
match before model merges. Literal spaces in normalized added-token text are
normalized by that same replacement.
The shared postprocessor described below applies to raw and byte-alphabet
BPE alike.

Raw BPE can merge across spaces, so it encodes a whole fragment and
`Tokenizer::WorkingBytes` charges 96 times (input bytes + 1) for raw BPE.
Byte-alphabet BPE also funds the complete normalized fragment and VM scratch. Replacing
one space with three UTF-8 bytes still creates only one symbol because
`▁` is required. Every other code point contributes at most its original
byte count of symbols. Code points and piece lengths, expanded text,
symbols and links, and at most three heap candidates per symbol remain
within that charge. Merge queues and emission ask the thread pulse every
65,536 iterations, so a large fragment has progress and cancellation
checkpoints. Input and output caps still apply.

**Classic SentencePiece GGUF.** `tokenizer.ggml.model = "llama"`, with
an absent or `default` pre-tokenizer, selects score-ordered merging over
raw UTF-8 code points. ASCII spaces become `▁`, with no NFC or whitespace
collapse. `add_space_prefix` prepends one dummy marker to each nonempty
fragment, initially and after a recognized special token. It defaults to
true; the approved Gemma 2 and 3 GGUFs explicitly disable it, while Phi
3.5 keeps it. Adjacent symbols whose concatenation is a normal token merge
by greatest finite F32 score, leftmost on a tie. Missing code points stay
mergeable until final emission; a missing final symbol emits its original
UTF-8 bytes through the required 256 byte tokens. Literal control and byte
token spellings cannot become special tokens through merging. This
preserves the native content-is-text contract even for a hostile vocabulary;
llama.cpp's unrestricted concatenation lookup does not provide that guarantee.

The reader follows the pinned special-token overrides (including `</s>`
and `<end_of_turn>`) and the named Phi `RSTRIP` rule: a `general.name`
containing `phi-3` or `phi3`, case-insensitively, strips ASCII whitespace
after recognized specials, except `<unk>`, `<s>` and `<|endoftext|>`.
The approved checkpoint's name, `Phi 3.5 Mini Instruct`, does not trigger
that override. Strip scans have cancellation checkpoints and cannot cross
a marked control span. Unsupported whitespace-collapse flags are refused.

SentencePiece encodes each complete fragment. `WorkingBytes` charges 96 times (input bytes + 1): expanded
text needs at most 3 bytes per input byte or dummy symbol, symbols and
links at most 24, and the score heap at most 60 (three candidates per
symbol). The heap uses bounded token lookups and stale-entry checks rather
than rescanning the fragment after each merge. Preparation, merges,
emission and special stripping pulse periodically. The cheap output-size
lower-bound refusal is disabled when stripping could shorten the input;
the output cap remains enforced during emission. This slice establishes
tokenizer agreement, without claiming model execution or rendered-chat
token agreement. The separate `gpt2` / `command-r` preset is now supported;
actual Command R model agreement is not claimed.

**Byte fallback.** Byte BPE creation requires a normal token for every
byte and a token for every merge's result, so every well-formed text encodes and
nothing is dropped (llama.cpp silently drops bytes a vocabulary lacks).

**Bounds and errors.** Input is untrusted: text must be well-formed UTF-8
(Unicode table 3-7) and is otherwise refused with `kInvalidUtf8` and the
offset of the first ill-formed sequence; `ReplaceInvalidUtf8` (maximal
subparts to U+FFFD, as Python's `replace` and WHATWG do) is the caller's
explicit lossy choice. `EncodeOptions` bounds the input (4 MiB by default)
and the output (2^22 tokens); serving passes its own bounds (D-102): a
chat rendering's own bytes and the model's usable context in tokens, a
literal prompt's bytes and the context, a command's text its bytes, so no
prompt is cut below what the body and the context admit (a 4 MiB+ chat
renders and tokenizes, `llm_scores_test`). A text longer than `max_tokens`
times the vocabulary's longest token (four times that under NFC, which
composes text at most threefold) cannot fit and is refused
(`kOutputTooLarge`) before it is decoded. All modes encode whole fragments:
an arbitrary source regex can join across the former 1 MiB space/newline cuts.
`WorkingBytes(input_bytes)` is allocation-free and saturates on overflow. It
funds normalization (fourfold under NFC), code points, merging, split vectors,
and VM counters/choices; `OutputTokensBound` separately funds output and all
source wrappers. `ReserveOutput` checks the append bound and reserves it
before either `Encode` or `EncodeMarked` emits anything, avoiding geometric
growth while a compact source recipe expands. `max_tokens` still limits this
append, independent of existing output IDs. Callers fund existing capacity
plus replacement capacity when reserving a nonempty vector, and can reserve
an explicit extra endpoint BOS slot. Marked span-index storage is reserved
separately from token storage. Serving pays both before encoding; capacity and
temporary availability failures report actual required memory bytes. In particular a
line break does not make a generic regex's working set smaller.
`Llm::EncodeText` returns a move-only owner that retains the output-capacity
charge after temporary work is released. Swap-table context caches keep that
owner after shrinking the visible token count and borrow its immutable tokens
for pair operations; destruction releases token storage before its payment.

**Canonical forced spans.** `PlanForcedSpan` plans an owned canonical prefix
from a still-unpublished selected token and known bytes. It uses the encoder's
actual special-token matching, raw newline cuts and EOF-sensitive pre-token
blocks, checks exact decode reproduction, and holds the unstable final block.
Normalization or decoder transformations that change the requested bytes are
explicit capability refusals. The plan never rewrites published history.
`ForcedSpanBytes` bounds peak work and owned results before allocation; token
and context allowances cut only at canonical token boundaries. A terminal
boundary is an actual grammar boundary, not an output-token allowance.

Forced bytes continue a stream: their encoding (`EncodeOptions::continuation`)
adds no SentencePiece dummy `▁` to any fragment, so mid-stream text reproduces
exactly; only a span declared to start the stream (`remove_space_prefix`) keeps
the dummy prefix. With `marked`, the planner encodes as `EncodeMarked` does:
exactly the given spans are control tokens, in the planned text (the selected
token's bytes, then the forced bytes, so a span may begin inside the selected
token and heal it; a selected control token is a span of its own), and control
text elsewhere, such as a client's tool name, stays text; user-defined tokens
still match. The canonical encoding, block trace, following-byte check and
held tail's cover (its spans shifted) all follow that one policy.
`ControlSpans` finds the control tokens a `kParse` encoding of a marker would,
to build spans; `ContainsControlText` reports control-token text anywhere in a
string, for refusing names at intake. Unmarked planning keeps `kParse`.

Boundary cover is proved only for a closed real block represented by one
canonical token. An open block has zero cover: a future higher-priority merge
can split a token encoded in isolation. A text ending in a possible longer
special token (an open trie prefix the policy allows) commits only blocks whose
partition the closed trace up to that prefix reproduces, since the atom would
close the fragment there, which is not an append. Bytes that finish a character
whose lead byte is already published have no canonical encoding from there:
the planner holds them and the cover is zero, so ordinary sampling along the
forced bytes completes the character. Following-byte promotion requires
EOF-independent stability, identical prefix partitions and canonical IDs for
every allowed continuation. A finite known-suffix first-ID match is only a
consistency check; it never permits injecting an unstable token. The caller
may keep an unchanged ordinary sampled token through a held tail as a distinct
choice. Feeding proven spans in one causal unit, the sampling constraint and
speculative acceptance are the runtime's ([tool calls](tool-calls.md)).

Parser storage derives from the source and paid construction memory. There
are no separate small expression, group, stage or wrapper-count policy limits.
Compilation uses explicit source-sized frames, resolves deep group chains once,
and checks that worst-case node/program/literal offsets fit their uint32
representation and that byte arithmetic does not overflow. Decimal counts are
compact checked uint32 counters, never expanded into instructions. The existing
JSON parser's source/representation limits apply to HF input. `ConstructionBytes`
funds source capacities, program/classes, compiler peak stacks and the actual
compact postprocessor definition/reference capacities. Matching has one transition allowance
for the entire encoding, across search starts, stage pieces, stages and special
fragments: `16 * (4 * (input_bytes + 1) + 1) * plan_program_size`, saturating
on arithmetic overflow. Program size includes instructions, classes, class
terms and literal storage; every transition, scan, class term, counter snapshot and restoration
pays that allowance. Choice storage has its own cap derived from input and
program size; single-class repetitions use compact backtracking ranges rather
than a snapshot per character. Nullable loops must make progress once their
minimum is met. Exhausted work or choice storage returns `kBounds`, with a
static reason; it never returns approximate pieces or uses elapsed time.
This is bounded backtracking, not a claim of linear completion for every
expression. Compiler, matching, merging, extraction and emission pulse for
cancellation; grammar/syntax refusals remain distinct from resource exhaustion.
BPE itself is O(n log n), and special-token extraction takes at most 256 trie
steps per byte. Historical window-based performance measurements do not
qualify the generic regex path.

Vocabulary caps:
2^22 tokens of at most 1,024 bytes, special tokens of at most 256 bytes.
Every refusal is a named `Rule` with a static reason and an item.

**Source postprocessing.** `EncodeOptions::add_bos_eos` applies the complete
source insertion recipe when true. `TemplateProcessing` accepts string and
object templates, exactly one A in a single-input template, any bounded number
of prefix/suffix wrappers and multi-ID named special definitions. The reader
first validates/counts definitions and references, reserves those exact capacities,
then fills a compact recipe. Each definition's IDs are stored once, with checked
slice references for repeated uses; construction never materializes reference-count
× definition-length IDs. Prefix composition reverses references in place instead
of repeatedly copying earlier wrappers. Traversal and temporary recipe storage
remain proportional to JSON source under the caller's parser loan. Construction
and emission pulse for cancellation; emitted IDs stream under `max_tokens`, and
the checked expanded insertion count enters `OutputTokensBound`. Definitions'
IDs and text must agree with the vocabulary; type IDs are checked uint32 values.
Pair metadata is validated for one A and one B, but this is a single-input
encoding API. Repeated/missing A or B in a single layout refuses. `Sequence`
composes in source order: later prefixes wrap earlier prefixes and earlier
suffixes precede later suffixes. ByteLevel postprocessing changes offsets only,
so has no ID effect. Wrapper IDs count toward the append's output cap and marked
span indexes include preceding wrappers. Named BOS/EOS roles remain independent
of arbitrary wrappers; kept tokenizer configuration supplies those roles. An
explicit HF processor, including an empty one, is authoritative for the source
recipe. GGUF and ModelProto use their insertion flags when no processor exists.
Literal completions retain their documented BOS-only policy (runtime-serving.md):
source processing is disabled, then at most one enabled named BOS is inserted;
no EOS or arbitrary template wrapper is added, and explicit IDs remain exact.

Qwen-Image's renderer returns its fixed framing and the marked-span index at
the end of its dropped system segment. The runtime encodes the whole rendering
once and obtains the dropped token count from that marked span's actual output
index. It funds the exact fixed-framing bound, complete tokenizer work, output
and span indexes before rendering; user-spelled control markers remain ordinary
content. There is no hidden system-prefix tokenization in the renderer.

**Decoding** gives each token's bytes: normal tokens through the byte
alphabet (raw UTF-8 normal tokens replace `▁` with spaces and byte fallback
tokens decode to one byte), user-defined tokens as their text, control tokens only when asked,
unused padding tokens as nothing. `StreamDecoder` holds back a trailing
partial UTF-8 sequence and replaces what can never complete. SentencePiece
uses these raw pieces by default, preserving generated leading spaces.
`DecodeOptions::remove_space_prefix` optionally removes one leading space
from the first token when the vocabulary uses a dummy prefix; a leading
BOS consumes that first position, as in the pinned reference. Streaming
decode applies this option only to its first token.

**Readers.** `ReadGgufTokenizer` parses a GGUF header's key-value section
from a byte prefix of the file with the shared GGUF parser (`base/gguf.h`,
which the importer and the artifact's kept metadata use too): counts and
lengths are checked against caps (2^16 keys, 2^24 elements in any array,
16 MiB keys and wanted strings) and the bytes left before anything is
allocated; other keys' values are not converted; keys are non-empty
UTF-8 and may not repeat, bools are 0 or 1, no array holds arrays (as
pinned gguf.cpp refuses), types must match. It accepts
`gpt2` vocabularies with the preset split plans above and token types
normal, control, user-defined and unused, and the Gemma 4 raw UTF-8 BPE
metadata and byte fallback described above. Classic SentencePiece also
requires one finite F32 score per token, accepts unknown tokens as controls,
and rejects merge lists and other pre-tokenizers; the standalone reader
below supplies explicit identity-map whitespace flags.
`ReadHfTokenizer` accepts the
`tokenizer.json` shapes hf.h lists (the byte-BPE pipelines above or the
validated raw-BPE pipeline, with no stripping flags) and refuses the rest. The
prepared artifact's tokenizer section is still the importer's to define;
carrying the checkpoint's own tokenizer data and reading it with these
readers is the suggested route.

**Standalone SentencePiece.** `ReadSentencePieceTokenizer` reads ModelProto
wire bytes directly, without a protobuf runtime or checkpoint code. It accepts
byte-fallback BPE models with an identity normalization map and implements
dummy prefix and whitespace trim/collapse flags. The official Gemma 2 and 3
models disable both; a name alone never invents a normalization transform.
Nonempty compiled maps, other algorithms, whitespace suffixes and
denormalizers are refused explicitly. The public Gemma 3 model's unassigned
normalizer field 7 is accepted only as its disabled value, zero. Varints,
wire types, lengths, repeated singular fields, finite scores and vocabulary
kinds are checked before publication; the input is bounded at 64 MiB and
parsing has cancellation checkpoints. The shared scored-BPE engine freezes
longest native user-defined symbols in normalized text before merging. Native
ModelProto symbols belong to the model: they do not split normalization into
separate fragments, restart its dummy prefix, or trim neighboring internal
spaces. Their own spelling is protected while the identity normalizer escapes
spaces; the model then matches its exact declared spelling, without inventing
normalized vocabulary aliases. Externally configured added tokens still extract in their declared
phase; GGUF's special-token fragment rules follow its reference contract.
Native UNUSED pieces participate in scored merges and are resegmented only
when they survive to emission. The reverse split walk is iterative. For models
containing these pieces, `WorkingBytes` charges 128 bytes per input byte plus
the dummy-prefix byte, and four bytes per vocabulary entry for the split table;
models without them retain the 96-byte bound. Construction accounting covers
model-membership/symbol flags and each raw or normalized trie spelling.

For HF and ModelProto assets, `ApplyHfTokenizerConfig` reads the kept
`tokenizer_config.json` special-token identities (including lists and named maps
of `extra_special_tokens`, resolved to existing vocabulary IDs), added-token
metadata and BOS/EOS insertion flags. ID/text changes, gaps, unsupported
stripping/single-word options
and enabled SentencePiece sampling are refused. The runtime ordinarily loads
kept `tokenizer.json`, or `tokenizer.model` when JSON is absent, followed by
that configuration and the kept chat template. A malformed present JSON
file is an error; it never silently selects another tokenizer.

## Unicode tables

`src/tokenizer/unicode_data.cc` is generated by `tools/gen-unicode-tables`
from UCD 15.1.0's UnicodeData.txt, PropList.txt,
DerivedNormalizationProps.txt, SpecialCasing.txt,
DerivedCoreProperties.txt and CaseFolding.txt, each pinned by SHA-256 in the tool (D-088):
CaseFolding.txt is the official 84,870-byte versioned file, SHA-256
`4e55acfdc32825a22e87670e9056a3bf94ad7c5400065778e9e10f8314372bcf`.
Its C/F records provide default full folds (up to three code points); S/T
records do not substitute simple or Turkic folds. The generated data provides
general category, White_Space and NFC_QC per code point as two-stage
tables, canonical combining classes, full canonical decompositions and the
primary composites; and, for the chat templates' case operations, the full
case mappings as CPython derives them (SpecialCasing.txt's unconditional
entries, else the simple mappings) and the Lowercase, Uppercase, Cased and
Case_Ignorable properties as ranges (4,410 mappings of 8 bytes, 4,593
mapped code points and 1,916 ranges: 68,980 bytes). 15.1.0 is the version llama.cpp's own tables
reproduce exactly (licensing.md), so the pre-tokenizers classify every
code point as the oracle does. Its case data equals Python 3.12's Unicode
15.0.0: SpecialCasing.txt is the same file, and UnicodeData.txt and
DerivedCoreProperties.txt differ only in 627 new uncased characters (CJK
Extension I, five ideographic description characters) and in identifier
and grapheme properties (diffed on `spark-b`, 2026-10-02; `pycase_test`
checks every code point against Python 3.12.3). On `spark` and `spark-b` the files are under
`~/.local/share/llmp/ucd/15.1.0/`, with NormalizationTest.txt, which a
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

| Configuration | Source (under `~/.local/share/llmp/` on the Sparks) | Reference | Result |
| --- | --- | --- | --- |
| `llama-3.2-1b-gguf` | `tokenizer-reference/llama32/header.gguf` | llama.cpp `d8123504`, current locked ARM64 image | 92 inputs in four modes plus four independent BOS probes; native verification pending |
| `phi-3.5-gguf` | `tokenizer-reference/legacy/phi35.gguf`; 32,064 tokens, `llama` | pinned llama.cpp | 184 of 184 equal |
| `gemma-2-2b-gguf` | `tokenizer-reference/legacy/gemma2.gguf`; 256,000 tokens, `llama` | pinned llama.cpp | 184 of 184 equal |
| `gemma-3-4b-gguf` | `tokenizer-reference/legacy/gemma3.gguf`; 262,208 tokens, `llama` | pinned llama.cpp | 184 of 184 equal |
| `gemma-4-26b-gguf` | `tokenizer-reference/gemma-4-26b/header.gguf`, the approved 26B-A4B checkpoint's complete metadata; 262,144 tokens, `gemma4` | pinned llama.cpp | 184 of 184 equal |
| `deepseek-v4-0731-gguf` | `models/unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93/UD-Q2_K_XL/…-00001-of-00003.gguf`; 129,280 tokens, `joyai-llm` | llama.cpp `llama-tokenize` (the pinned image) | 184 of 184 equal |
| `qwen3.8-gguf` | `reference-models/Q/UD-IQ3_XXS/…-00001-of-00003.gguf`; 248,320 tokens, `qwen35` | llama.cpp | 184 of 184 |
| `qwen3.8-nvfp4` | `models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6/tokenizer.json`; NFC, qwen35 | Hugging Face tokenizers 0.22.2 | 184 of 184 |
| `qwen-image-2.1` | `models/Qwen/Qwen-Image-2.1@790c9263/processor/tokenizer.json` (Qwen3-VL-8B); NFC, qwen2 | tokenizers 0.22.2 | 184 of 184 |
| `deepseek-v4-0731-hf` | `tokenizer-reference/deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b/tokenizer.json` (DeepSeek's own) | tokenizers 0.22.2 | 182 of 184; the 2 others are a known divergence |

Ill-formed items are refused at the offset of their first ill-formed byte;
their replaced text then agrees with the reference's tokens of Python's
`replace` decoding. Byte-BPE decoding returns non-normalizing encodings'
text exactly; Gemma 4 additionally replaces a literal `▁` with a space.
SentencePiece agreement checks token IDs rather than round trips: its
dummy prefix, literal `▁` and special stripping can change decoded text.
Where the references disagree with each other, llmpalooza follows each
model's oracle and records why:

- **Unicode version.** Hugging Face's Oniguruma knows Unicode 16.0's emoji
  (U+1FAE9, U+1FAC6) as symbols; UCD 15.1 and llama.cpp leave them
  unassigned, which the DeepSeek expression treats differently. Llmpalooza
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
   conversations, and first, for a family whose template variants part
   where those do not reach (Gemma 4's), six more: tool-call arguments
   that are a string, a string holding an object and a list, a system
   message without content, properties named `type` and `description`,
   and blank content beside tool results; each with thinking unset, on
   and off, each accepted reasoning effort and `preserve_thinking` off;
   both refusing counts as equal, a case the renderer does not implement is skipped, and at least
   30 renderings must be equal). Each probe renders under 500,000 steps
   and 16 MiB of work, so a template that matches a family's text but
   works hard besides (or burns its budget on purpose) differs at its
   first probe instead of costing registration hundreds of full
   renderings; and all the probes of one registration (every family's,
   and the end-of-turn probe) draw on one pool with its own fixed basis
   (`chat::kProbePool`: 50,000,000 steps and 2 GiB of work, what a
   single rendering's bounds were before D-102 lifted them), so a copy
   that stays just under each probe's budget, probe after probe, spends
   the pool and is interpreted. A repackaged copy of a supported template
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
| Gemma 3 | `7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4` | `chat_template.jinja` of `unsloth/gemma-3-4b-it@bf46152c`, the template Unsloth's Gemma 3 copies and the converted GGUFs surveyed carry (Google's own repositories are gated; `ggml-org`'s, `bartowski`'s, `lmstudio-community`'s and `MaziyarPanahi`'s GGUFs converted from them embed it) | transformers 5.12.1 `render_jinja_template` |
| Gemma 4 | `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4` | `chat_template.jinja` of `google/gemma-4-31B-it@842da379` (Google's template of 2026-07-15; also 26B-A4B's and 12B's) | transformers 5.12.1 `render_jinja_template` |
| Gemma 4 E2B and E4B | `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5` | `chat_template.jinja` of `google/gemma-4-E4B-it@ee0ef602` (and E2B's) | transformers 5.12.1 `render_jinja_template` |
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

Both read `thinking` (llmpalooza's `enable_thinking`), off by default, end
turns with `<｜end▁of▁sentence｜>` and render a tool call's arguments as
DSML parameters; chat-v2 parses string arguments with `from_json`, which
equals rendering the client's string parsed (fixtures `string-arguments`,
and `string-arguments-not-object`, which both refuse).

**Generated tool output.** The protocol-neutral [tool-call foundation](tool-calls.md)
attaches a declarative output grammar to verified native templates and consumes
only generated text. It serves the Chat Completions route, with required and
named triggers forced through the canonical span planner above; real-model
qualification remains outstanding.

**Structured content.** `chat::Content` owns either nullable scalar text or
ordered text, tool-use, tool-result and image parts. Intake keeps text-part
boundaries; one shared checked projection joins text at the template boundary
without changing control-token marking or turn boundaries. Tool input JSON is
owned and parsed into stable documents whose lifetime follows projected calls;
interpreted templates receive call IDs and result IDs as well as the existing
function fields. Text after tool use, mixed/multiple or out-of-order results, error results and
images are refused until a faithful rendering path exists. Gemma 3 refuses
structured tools instead of silently ignoring them. The OpenAI route still
accepts text only, and `/v1/messages` is disabled. Reuse remains keyed by rendered
tokens and model/template state, so text partitioning alone changes no KV identity.

**Gemma.** `chat/gemma.cc` holds both generations, written from the formats
and linear in the
conversation (the templates rescan it for every message: Gemma 4's,
interpreted, would stop near 4,400 messages, beyond the chat route's
intake of 1,024):

- **Gemma 3:** `<bos>`, then `<start_of_turn>ROLE\n … <end_of_turn>\n`
  per message, the assistant as `model`, content stripped; a leading
  system message, unstripped plus a blank line, opens the first user turn.
  Roles must alternate user/assistant after it (a tool or later system
  message takes an assistant's place as its own role), and content must be
  given; otherwise the template raises and the renderer returns `kInvalid`.
  Tools, tool calls, reasoning and options are ignored, as the template
  ignores them. Gemma 3n's and Unsloth's 270m templates differ only in
  content parts and render alike, by probe.
- **Gemma 4:** `<bos>`, a system turn (`<|turn>system\n … <turn|>\n`) when
  thinking is on, tools are given or the first message is a system one,
  holding `<|think|>\n`, the stripped system text and each tool as
  `<|tool>declaration:NAME{…}<tool|>` in Gemma's notation (strings between
  `<|"|>`, keys sorted case-insensitively, types upper-cased, nested
  properties, items, enums, `nullable` and `required`, and a response);
  then turns `<|turn>ROLE\n`. An assistant's reasoning renders as
  `<|channel>thought\n…\n<channel|>` after the last user message (or, with
  `preserve_thinking`, on a message with tool calls); its tool calls as
  `<|tool_call>call:NAME{key:value,…}<tool_call|>`; the tool messages that
  follow it as `<|tool_response>response:NAME{value:<|"|>…<|"|>}<tool_response|>`
  inside the same turn (NAME is the message's last call's: the template
  matches call IDs, which llmpalooza's messages lack, so every call matches);
  its content with thought channels removed. A run of assistant messages
  shares one turn; a call without results ends the turn with
  `<|tool_response>`; the generation prompt is `<|turn>model\n` plus an
  empty thought channel unless thinking, or after tool results with
  thinking on, `<|channel>thought\n`. Arguments that are not an object
  (or null) raise.
- **Variants, chosen by probe:** E2B and E4B's template (pinned) has no
  empty thought channel in the prompt; Unsloth's (`845f1ee4…`, its 12B,
  26B-A4B and 31B GGUFs and copies; `241c50d8…` for E4B) renders string
  arguments: what is inside the braces when the stripped string is
  wrapped in `{…}`, otherwise the string as given; it ignores lists and
  numbers; Google's template of 2026-04-28 (`94899c0f…`, NVIDIA's NVFP4
  checkpoints) has no `preserve_thinking`, shows reasoning only after the
  last user message on a message with tool calls, prints a null as
  `None`, closes every assistant turn, renders string arguments as they
  are and raises on an empty conversation. Its 2026-05-18 revision
  (`36e3a42e…`) differs only in tool results given as content parts,
  which llmpalooza's messages cannot express, so it is the same variant by
  probe. Google's two earlier Gemma 4 templates (2026-04-02 `2dfbfc7d…`,
  2026-04-10 `85a08664…`) are interpreted: the 2026-04-10 one prints a
  system message without content as `None`, always filters properties
  named like schema keys and closes a turn after tool results beside
  blank content, which the variant probes find.

Case mapping (`upper` of types, `dictsort`'s key order) is Python's in
full, as in the interpreter (`chat/pycase.h`, below): `строка` upper-cases
to `СТРОКА`, `ß` to `SS`, and keys sort by their full lower case.

**Native refusals fall back to the template; native output is bounded.**
What a native renderer does not implement (`kUnsupported`, such as
DeepSeek's `preserve_thinking`) renders through the interpreter, from the
template's own text, which `ChatTemplate` keeps beside every native
renderer; the stop tokens stay the native renderer's. The fallback is the
template's own rendering, with the interpreter's semantics and evidence
(below), not a native approximation. A native rendering is bounded as
the interpreter's output is (`Conversation::max_render_bytes`, which
serving sets to the template's `jinja::Limits::max_output_bytes`; the
library's default 32 MiB): past it the renderer drops what it is given and
refuses (no fallback, the interpreter would refuse too), so text repeated
per message (Gemma 4 prints the last call's name for every tool result)
costs no more than the bound. Serving derives the bound from the model
(D-102): four times its context times its vocabulary's longest token, what
a prompt that fits the context could occupy, at least 32 MiB and at most
a sixteenth of memory (`runtime/intake_limits.h` `RenderBytes`); a value's
string bound is at least that and the values held at once at least four
times it.

**Supported.** DeepSeek V4: roles system, user, assistant and tool;
request-level tools, DSML tool calls, reasoning, `enable_thinking` (the
template's `thinking`) and `reasoning_effort` (0731: low, high, max;
chat-v2: ignored, as its template ignores it). Its
developer and latest_reminder roles, tasks and response formats are refused
as unsupported. Qwen3.8: system (first only), user, assistant and tool; tools
and XML tool calls, `<think>` blocks, `enable_thinking`, `reasoning_effort`
(xhigh, medium, low) and `preserve_thinking`. Gemma 3: system (first),
then user and assistant alternating. Gemma 4: system, user, assistant and
tool; tools, tool calls and results, reasoning, `enable_thinking` and
`preserve_thinking`; `reasoning_effort` is ignored, as its templates
ignore it. Tool-call arguments are JSON
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
among them), 59 for each Gemma 4 template (Google's, E2B/E4B's and the
2026-04-28 one) and 44 for Gemma 3's (`gemma_fixtures.py` in the
[corpus](experiments/chat-template-corpus/README.md): declarations through
every branch of the notation, tool rounds and chains, thinking,
`preserve_thinking`, channels in content, unparsed arguments, number and
string formats, key order, the independent review's cases, the templates'
refusals), and 4 image prompts. Text compares
byte for byte in every profile (`chat_test`); token IDs compare with the
references on the model files (`tokenizer_models_test`; Gemma rendered-chat
token agreement remains outside the tokenizer-only slice): all equal, with
DeepSeek's against llama.cpp (for chat-v2, llama.cpp on the community GGUF
and the native tokenizer from its artifact's kept metadata, which equals
the 0731 GGUF's) and against Hugging Face.

**Stop tokens:** DeepSeek V4 (both templates) `<｜end▁of▁sentence｜>`;
Qwen3.8 `<|im_end|>` and `<|endoftext|>` (its generation_config.json's
`eos_token_id`); Gemma 3 `<end_of_turn>`; Gemma 4 (every variant) `<turn|>`
and `<|tool_response>`. The runtime reads no generation_config.json, so a
native renderer's stop set carries what that file's `eos_token_id` lists
beside the tokenizer's EOS (added where the model registers), as Qwen3.8's
`<|endoftext|>` does. Gemma 4 lists `<|tool_response>`: the model ends a
turn that calls tools with it (the template writes it after a call that
has no results yet), and without it a model given tools would go on to
write the tool's response itself. The interpreted path's heuristic finds
only `<turn|>`, so a Gemma 4 template no native renderer serves (Google's
two earliest) stops at `<turn|>` and EOS alone.

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
D-067's, in `chat::jinja::Limits`, as D-102 amended them: memory (values
held, one string, the output), nesting (the stack), the template's own size
and syntax tree, and Jinja2's `MAX_RANGE`. A rendering may lower the memory
bounds (`jinja::Budget::max_live_bytes` and `max_output_bytes`, from
`Conversation::max_live_bytes` and `max_render_bytes`): serving sets them
from what the request is charged, four times its messages' bytes and 16
MiB of values, 1 MiB of output, within the model's bounds
(runtime-serving.md#the-chat-route). Time is not bounded: the steps
and bytes built or scanned are still counted, as below, but they are no
longer caps (50,000,000 steps and 2 GiB before); instead the rendering asks
its caller's cancellation every 65,536 steps and every 4 MiB of work
(`jinja::Budget::cancelled`), and serving answers it from the request (a
client gone, the runtime stopping, a configured deadline, which is set
before rendering), so a template
that would run for hours stops within milliseconds of its request ending
(`kCancelled`; `jinja_test` checks a 50,100,000-iteration loop renders to
its end, about 4 s on spark, and hostile templates end by a bound or their
cancellation). Only a `Budget` caps steps and work, as probes use it. A
value's text or JSON stops at the
string bound as it grows, so one string held many times over costs no
more than that; a string's trusted ranges (16 bytes each, eight times its
text where the template's text and a client's alternate byte by byte)
count toward the string bound with its text as it grows, and the output's
toward the output bound, which so covers the whole rendering (its ranges
pass to the caller without a copy, and the control tokens found in them,
16 bytes a span, count toward the same bound until the ranges are
released); a set block's or a macro's output is held as live bytes as it
grows (they nest, each building while the next runs); a container's repr
and `tojson`'s text are held as live bytes while they are copied into a
string; comparisons charge every byte they compare; substring
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
rendering places is charged to its work (and asks the cancellation as the
rendering does). Containers release what they
hold without recursing, so no chain of them can exhaust the stack.

Case mapping is Python's, in full (`chat/pycase.h`): `upper`, `lower`,
`capitalize` and the `title` method as str does them (SpecialCasing's
unconditional mappings, Final_Sigma), the `title` filter by Jinja2's own
rule, `dictsort`'s, `sort`'s, `unique`'s, `min`'s and `max`'s case
folding by str.lower, and the `lower` and `upper` tests by str.islower and
str.isupper; `pycase_test` checks every code point and 1,020 strings
against Python 3.12.3 and Jinja2 3.1.6. ASCII runs map byte for byte (about
330 MB/s on spark-b's GB10), other text a code point at a time through the
tables (40 to 80 MB/s), so the work count charges case mapping and the
case tests 2 per ASCII byte and 16 per other byte: spending the former 2
GiB bound on them took at most 3.3 s (Greek with combining marks
title-cased, the slowest measured; 2.5 s upper-casing Cyrillic, 1.1 s
testing CJK with `is lower`, 1.3 s mapping ASCII; before the charges,
Cyrillic took 15 s and CJK 18 s), so the cancellation, asked every 4 MiB
of that count, is asked every few milliseconds of such work.

Known narrowings, none met by the corpus: filters Jinja2 returns as generators
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
bound and the cancellation with hostile templates and a 20,000-mutation
fuzz;
`chat_template_test` checks the choice of renderer and that the
interpreter reproduces every case of the five in-tree pinned templates'
fixtures; `chat_corpus_test` (label `models`, the corpus on the Sparks)
renders 38 real templates on 19 conversations each, 696 renderings and
every refusal equal to transformers', through the interpreter and through
the chosen renderer, and the Gemma renderers against their templates on
3,000 random conversations.

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
(2026-09-29): `llmp_sampling_bench --draws 200`, row seed 42, three
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
(`llmp_sampling_bench`, synthetic peaked and flat rows, median of 200
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
