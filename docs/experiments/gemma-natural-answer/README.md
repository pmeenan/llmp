<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma candidate natural answers

Both approved Gemma checkpoints produce coherent complete answers on the four fixed short prompts. Every answer stops at the actual EOS106, with no generation limit or truncation. The 31B native and llama.cpp answers match in complete text and generated IDs for all four cases. The 26B answers match exactly for arithmetic, instruction and short retrieval; its explanation changes only “needed” to “necessary” between engines and remains semantically equivalent.

The predeclared exact arithmetic string remains a **31B failure for both engines**: each returns `45.0` rather than `45`. Its numeric value and decimal-number response are correct; that separate semantic judgment does not normalize the stored answer or change the exact-string result. The other exact targets are `apple pear plum` and `ORBIT-COPPER-7392`; both engines match them for both checkpoints. The 26B arithmetic answer is exactly `45`.

Root and the experiment author read all complete explanation answers. Each has exactly two sentences connecting darkness with absent photosynthesis, glucose/energy and depletion of stored nutrients, followed by weakening or death. That agent review passes the predeclared semantic requirement; it is recorded separately from the automated exact checks. Full original answers, generated IDs, terminal IDs, reports and logs remain outside Git in the sibling `gemma-natural-answer-raw` directory. [Results](results.json) keep their identities and bounded per-case judgments.

## Actual recipe

The private `jitllm-runtime` was built on Spark A (`spark-c4e2`) from base86d7c55 with the candidate15/c40 production-recipe overlay. Both approved prepared artifacts use their actual tokenizer/template, context4096, one slot and one funded output row. Prefill caps are256 for31B and1024 for26B; candidate plain norm/norm-ROPE/norm-ADD and26B route/reduce policies are retained. Owner attention and joined serving remain default-off. No source or runtime change was made for this screen.

Each checkpoint runs one existing runtime process with `chat --fresh --plain --max-tokens128 --report` and four `--turn` arguments: explanation, arithmetic, instruction and short retrieval. New owner-only data/spill/state roots and enrollment/report paths refuse overwrite. Complete native prompt IDs match the independently qualified actual chat carriers of29,30,28 and73 tokens. Four fresh turns have no history reuse; runtime return0 includes successful server teardown. The pinned SDK and exclusive private binary are checked before and after acquisition.

The retained [caller](natural.py) uses the existing owned-server seam with image837fc732 and the original pinned llama.cpp engine, physicalC1 context4096, normal ring KV, F16 K/V, the same256/1024 batch/ubatch caps and ordinary graphs/fusion. Actual global/local caches are4096/1280 for31B and4096/2048 for26B. Each of four `/completion` calls receives the already qualified complete prompt-ID array, temperature0, no repeat/presence/frequency penalty, return_tokens and a128-token limit. `--special` exposes the control-token strings used by the stop matcher; `<turn|>`, `<|tool_response>`, `<|channel>` and `<|tool_call>` correspond to actual IDs106,50,100 and48. Full responses precede assessment, and checked owned-container retirement proves absence.

The runtime target-only build completed all five supervised steps. Each of the four native/public acquisitions completed its one supervised step, and each metadata-only assessment completed all three steps. Root and an independent reviewer closed the source/config and actual-output-bound assessment packets before execution. Aggregate identities are in results; raw supervisor/process records stay outside Git. No acquisition failure or rerun occurred.

## Qualification limits

This is a four-prompt C1 natural-answer check against the reference. It establishes neither general model capability nor batching, long context, retrieval at depth, corpus perplexity or performance. The short recovery-passphrase prompt is not a depth test. Independent corpus/cohort gates and historical strict-choice failures retain their own outcomes; these answers do not waive them. Candidate defaults are not widened by this report. The owner-deferred full regression suite was not run for this diagnostic.

## Reproduction

The caller is the exact reviewed acquisition/assessment source, SHA `058942f6…`. Copy it to the external `gemma-production-natural` directory with the separately supplied, authenticated prompt carriers, facts and checksum lists. The [input preparation](../gemma-input-preparation/README.md) records their generation and independent tokenizer qualification; results record the model, input, executable, configuration and output identities. Invoke `natural.py native|public|assess 26|31` only under the installed Spark supervisor. Supply the pinned private candidate runtime and its dependency/checksum lists, use new owner-only output roots, and retain official successful retirement before assessment. Raw prompts, IDs, responses, logs and machine-specific guards remain external.
