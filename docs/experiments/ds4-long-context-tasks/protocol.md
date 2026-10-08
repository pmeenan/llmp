<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fixed long-context answer checks

This protocol was frozen before collecting three independently authored
synthetic tasks and one separately judged code-review sample. The completed
[results](README.md) retain the original acceptance rules and failed R1
condition; the long pairs were scheduled independently under explicit root
clearance. The comparison uses the prior native
HCA-quality artifact `8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234`;
the community artifact used for the complete ds4 performance reference is a
separate experiment.

`llmp_long_context_task_prepare` reads the artifact's authenticated GGUF
metadata, native tokenizer and registered chat template without opening a
device. It renders a nonthinking user turn. Fixed source constants use seed
42, unique numeric target records and distractors, and chronological variable
bindings. Neutral filler alone changes during bounded full-render encoding;
facts, questions and chain links are never truncated or revised. Preparation
refuses an unreachable token target or an encoding/byte limit, preserving a
failure receipt. Successful output requires exactly the following final
rendered lengths and fact starts, with byte reconstruction and complete
token-span boundaries proved by the native tokenizer.

| Case | Prompt / capacity | Fact starts in final token IDs | Output cap | Selected HCA calls |
| --- | --- | --- | ---: | ---: |
| R1 single retrieval | 30,720 / 32,768 | 16,384 | 128 | 300 |
| R2 four-key retrieval | 126,976 / 131,072 | 8,192; 24,576; 106,496; 122,880 | 128 | 600 |
| R3 four-hop bindings | 126,976 / 131,072 | 8,192; 24,576; 57,344; 102,400; 120,832 | 128 | 600 |
| C1 code review, qualitative | 30,720 / 32,768 | 8,192; 24,576 | 768 | 300 |

The counters describe the actual mixed configuration: T=2,048 and used
padded compressed extent 256 or 1,024 selects twenty HCA layers per full
chunk. Extents 512 and 768 remain ordinary. R2/R3 end at chunk 62; this is
not an all-HCA 128K experiment. Runtime counters must match the table.

Freeze the complete four-case preparation receipt, rendered/user bytes,
token files, actual fact spans, answers, rubric, metadata/template/stop
hashes and checked source/binary identities before reading either arm's
answer. Outputs are fresh external directories; they are never overwritten.
Use the same checked binary, artifact, tokenizer, stop IDs, greedy decoding,
context, 2,048-row chunks, compact experts, frontier head and wide sparse
attention in both arms. Exact mode, Q2 D2R and MTP remain off. Only
`--ds4-hca` differs. Each arm starts a new process and fresh state, with no
prefix or turn reuse.

The benchmark's private `--stop-ids` is explicit and optional. It preserves
the first authenticated stop in raw IDs/logits and stops before feeding it
into another decode. An immediate stop produces zero answer tokens. The
default benchmark retains its existing fixed-length generation. Forced
trajectories refuse this option. Preserve every answer and distinguish EOS
from output-budget exhaustion.

R1 accepts only a one-integer JSON array; R2 accepts only the exact four-integer set;
R3 accepts only the exact five-variable-name set. Responses must be a single
JSON array and end at authenticated EOS. Duplicates, extra values, prose,
wrong types and incomplete or budget-exhausted responses fail. The variable
answer is checked against execution of the complete chronological facts.
A baseline failure is retained and uninformative for candidate acceptance;
it is not replaced after outputs. Repeat both arms with fresh state after a
candidate-only failure before attributing a regression.

C1 supplies a fictional two-file C++ length-prefixed parser and queued
dispatch over a reusable receive buffer. Ask for a concrete bounded patch
and its essential tests. The fixed qualitative rubric requires header/offset
validation before pointer arithmetic, overflow-safe body bounds and the
65,536-byte cap, owned or completion-fenced payload lifetime, tests for
truncation/zero/max/oversize/overflow/buffer reuse, and preservation of valid
semantics with bounded buffering. The helper preserves the complete answer
and rubric but never machine-accepts this case. Root judges the off/on pair
separately from the three exact synthetic checks.

Start with the paired R1 smoke; if both pass, run the two long pairs. The
prior estimate is 22–25 minutes before conditional failure repeats; this is
not a measured task duration. The result must be needed for the practical
quality decision before scheduling under D-085. C1 is an optional separate
pair with an announced estimate. Passing three fixed synthetic cases and
one qualitative sample establishes no broad task-quality gate, cannot
reverse the existing greedy/PPL bounds, and does not enable a default.

The generation and prompts are self-authored; no RULER implementation,
external corpus or unknown-license data is imported. Task concepts follow
the previously audited retrieval/variable-tracking proposal, with changed
native positions and strict scoring rather than a leaderboard claim.
