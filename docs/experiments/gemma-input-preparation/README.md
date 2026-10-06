<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fresh Gemma text and tokenizer inputs

Both checkpoints pass this input-only qualification, with no model execution
or quality result. Each supplies twelve 1,024-token literal histories and four
natural chat prompts. All supplied native/public IDs match byte for byte;
all four native/Jinja rendered prompts match completely, and every history has
exactly one leading BOS. The actual renderer is `gemma-4-unsloth`, selected
`native-by-probe`. Both supervised input batches retired successfully in
28 seconds, with all 32 owned containers proved absent.

The native CPU exporter reads the actual checkpoint's GGUF metadata, creates its
native tokenizer, and exports either literal-text IDs or one-user chat IDs.
Chat uses the production `ChatTemplate::ForText` probe/interpreter path for
the embedded template, thinking disabled and a generation prompt;
marked template controls are encoded without automatically adding another BOS.
String/render bounds are 4 MiB, and the receipt records renderer name and
selection method. The actual template is selected through production probe
equivalence rather than hash-only registration.
The public client's new `prepare-rendered` mode loads only the vocabulary,
uses `add_special=false/parse_special=true`, and never creates a context.
Its existing `prepare` mode still supplies the first 1,024 literal IDs with
`add_special=true/parse_special=false`.

Both actual prepared checkpoints carry template SHA-256
`845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b`
(18,924 bytes), BOS 2, EOS 106, automatic BOS enabled and no automatic EOS.
The 26B metadata is 15,784,342 bytes, SHA-256
`5d15e0c84e29af2ed9b4525dacdd31ccd7bae061eb28dde2dad984a9b1e45a59`;
the 31B metadata is 15,784,340 bytes, SHA-256
`79c54f13f654d98e93ba1496062c017baea53ddaed59de086fe3fa7cf09d8461`.
These are checkpoint-specific prepared metadata, not interchangeable fixture
coverage. Existing tokenizer corpus controls cover 26B; their result does not
establish 31B agreement. The generic Gemma chat fixture's template is different.

`prepare.py extract PPL NATURAL NEW_TEXT_DIR` freezes twelve literal slices of
the existing 3,274,124-byte normalized War and Peace corpus, SHA-256
`c7156148ecaa12b6416cf816540d8dede2014982554a835e61076f0dd8bf0c2d`.
For history i=1..12 it starts at the first paragraph boundary strictly after
byte 65,536*i and ends at the first paragraph boundary after another 32 KiB.
These text selections precede tokenization. A slice with fewer than 1,024
IDs in either checkpoint is a refusal, not an invitation to choose another
paragraph after seeing model outputs. The old corpus prefix starts with the
book title and table of contents; this is a concrete workload caveat, not an
explanation for a model's high absolute perplexity.

The same extraction copies the four previously fixed natural prompts from
`natural-prompts.json`, SHA-256
`116f5d709cfb18fc5681fabc6045d5b4d47d774ab9deba45c6a9d50dd5721b1c`.
They test explanation, arithmetic, exact instruction and short retrieval.
The retrieval prompt is not a retrieval-at-depth test.

For each checkpoint, run the native exporter on all twelve literal texts and
four natural user texts. Its chat output includes `template.jinja`. Independently
render that exact pinned template with `prepare.py render TEMPLATE NATURAL
NEW_RENDER_DIR` (the existing Jinja2 3.1.6 immutable-sandbox convention).
Run public `prepare` on the literal texts and `prepare-rendered` on these
independent renders. Put the public render directory at `PUBLIC/rendered`.
`prepare.py pack TEXT_DIR TEXT_MANIFEST_SHA NATIVE PUBLIC 26|31 NEW_INPUT_DIR`
requires byte-equal supplied IDs, matching native/Jinja rendered bytes, the
actual metadata/template pins and exactly one leading BOS in each history.
It emits a 49,152-byte owner-major `cohort12.i32`; C4 and C8 consume its first
four or eight histories. Natural IDs and per-history input manifests are separate.
No cross-checkpoint token identity is assumed.

The text manifest was frozen before either tokenization batch at SHA-256
`c30dfd97a19615fcc4eb1831a7cc053f39336e1466fa0744f9c78a1b218153d8`.
Both independently checked carriers are actually byte-identical: 49,152 bytes,
SHA-256 `d584450079145f3d2c93f46ef24a0aabe2b3971279a1cbddbbb29f0a506ac5e3`.
Their checkpoint-specific manifest hashes differ: 31B
`f6cf32b29dd980c997915519339cdc1cd2c244e219c473a0331303bbc6e5708c`,
26B `b91eecf9d54fc55392d15ea04ba18c5a4cc5ab8e8ac2294501094d5959b578e1`.
Each literal slice has 7,613–8,434 native tokens; only its first 1,024 supplied
IDs form the corpus carrier. The natural prompts contain 29, 30, 28 and 73
tokens respectively, with complete IDs checked against the public tokenizer.

Corpus scoring retains all 1,024 heads and scores exactly 1,023 within-history
transitions. The quality analyzer's optional trailing `INPUT_MANIFEST
MANIFEST_SHA` parameters authenticate fresh inputs before reading outputs;
the old invocation still requires the historical `b2d7...` ID hash. Both the
native own freeze and subsequent oracle comparison bind the same externally
fixed manifest SHA. No noise allowance or score computation changes.

Cohort frontier row 0 (992 completed positions) targets ID 992, and rows 1..31
target IDs 993..1023. Row 32 (1,024 completed positions) has no supplied target.
Retain its full-head and argmax comparison, but do not score it against the
next owner's BOS or pretend that the corpus carrier supplies position 1,024.
No additional target tail is introduced here.

Build only `jitllm_gemma_tokenize` and the existing public quality client after
source/configuration review. The tiny `test_controls.py [NATIVE_EXPORTER]`
checks manifest replacement, legacy-pin refusal, one-BOS enforcement, final-row
bounds, and exporter refusal paths using invalid synthetic metadata. It does
not tokenize a checkpoint or read quality outputs. All texts, renders, IDs,
per-input manifests and logs stay in owner-private external scratch. Runtime
quality/model acquisition remains a separate release.

The narrow exporter build and focused manifest/exporter refusal controls passed on Spark A
(`spark-c4e2`). The public vocabulary-only client was compiled against the
existing pinned reference image; Jinja2 3.1.6 and MarkupSafe 3.0.3 independently
rendered the actual template. [Aggregate identities](results.json) bind the
measured source, binaries, text freeze and checkpoint-specific inputs. No model
context, calibration, logits, generation, perplexity or quality comparison was
created by these batches. Actual model qualification remains open.
