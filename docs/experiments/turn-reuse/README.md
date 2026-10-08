<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native turn-boundary reuse (2026-09-29)

The runtime retains two checkpoints on each model's current branch,
before the renderer's assistant opening. Exact token matching can restore
the nearest retained boundary when a client removes earlier reasoning,
then prefill only the supplied suffix. Mutable whole used pages go into
private unnamed direct-I/O files; immutable earlier pages remain in live
state or ordinary spill. Rollback restores the full footprint, removes
newer tail pages and restores the speculation cursor and adaptive depth.
Eligibility expires 24 hours after capture; file cleanup is lazy on the
model's next request or destruction. These are process-lifetime caches,
not restart recovery or independent retained branches under D-031.

## Control

`benchmarks/turn_reuse.cc` runs in the runtime's native `Server`, with
DSpark or MTP enabled. It renders a long first user message, then a second
turn containing a fixed supplied assistant answer without reasoning and a short
user follow-up. Its fresh control uses exactly the checkpointed prefix's
prefill chunks, then the second turn's chunks and assistant opening,
followed by 32 greedy tokens. The tested path runs the original turn and
32 tokens, restores its stable boundary, and processes the second turn.
Every last-row logit and every continuation logit/token must equal the
fresh control bit for bit. This separates rollback correctness from
floating-point differences caused by different prefill chunk boundaries.

Repeating the second request must select its newer checkpoint and repeat
all output bits, including after an optional full swap through the other
LLM. A zero checkpoint count or cache allocation refusal fails this
control's reuse assertion rather than presenting fresh recomputation as a
hit. Timing includes restore, checkpoint capture and suffix prefill; it
excludes model activation and generation. Raw reports stay outside Git.

Usage on a free Spark, after the memory/model-process gate:

```sh
build/spark-native/benchmarks/llmp_turn_reuse \
  CONFIG.toml ENROLLMENT_ANCHOR MODEL PROMPT.txt TOKENS SWAP_MODEL
```

## Verification

Final Spark check set: all 1,018 tests passed, including 194 GPU tests;
the 292-unit portability boundary check passed. Seven new GPU controls
cover overwritten page tails after eviction, direct-I/O padding, failed
capture and restore allocations, and retained staging after an uncertain
device copy, and cancellation before or during a transfer. Three CPU
controls cover nearest-boundary matching, expired
or unusable boundaries, and both renderers' stable prefix when reasoning
is absent. Changed units passed the Spark SDK's format and tidy checks;
the 981-file REUSE/header check passed. Native 8K runs on `spark-b`
(20:08:18–20:10:54 EDT) passed both models' fresh and post-swap controls:

| Model | Second prompt tokens | Reused prefix | Fresh prefill seconds | Restore/capture/suffix seconds | Newer-checkpoint repeat after swap seconds | First checkpoint MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek + DSpark | 8,212 | 8,193 | 18.414 | 0.411 | 0.142 | 289.141 |
| Qwen3.8 + adaptive MTP | 8,271 | 8,239 | 3.982 | 0.354 | 0.196 | 468.039 |

The 32-token continuations and every saved logit agreed bit for bit with
the fresh replay; the newer checkpoint also repeated exactly after a
full LLM swap. Two entries remained per model. Peak startup-relative
MemAvailable drops were 109.08 / 109.22 GiB, including both models' loads.
One run per model; initial prefill including capture took 17.919 / 3.762 s.
The control's second prompt is slightly longer than the initial prompt,
so these initial/fresh times do not isolate checkpoint overhead.

The 64K controls (20:11:32–20:18:22 EDT) also passed every logit/token
comparison, including the newer-checkpoint repeat after a full LLM swap:

| Model | Second prompt tokens | Reused prefix | Fresh prefill seconds | Restore/capture/suffix seconds | Newer-checkpoint repeat after swap seconds | First checkpoint MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek + DSpark | 65,556 | 65,537 | 141.607 | 0.507 | 0.190 | 323.141 |
| Qwen3.8 + adaptive MTP | 65,615 | 65,583 | 29.058 | 0.423 | 0.199 | 484.039 |

Initial prefill including capture took 141.374 / 28.679 s; peak
startup-relative MemAvailable drops were 109.91 / 109.22 GiB. One run
per model. Only 19 / 32 supplied suffix tokens were recomputed. The
64K tests preceded the final per-page cancellation/progress addition;
their empty continuation callback uses the unchanged transfer path.
Final HTTP integration checks use the actual watchdog callback.

An HTTP control (20:24:11–20:25:13 EDT) sent each model's first response's
content field back, omitting its reasoning field. DeepSeek reused
5,984 of the second prompt's 5,997 tokens; Qwen reused 6,092 of 6,118.
The second requests took 1.371 / 1.139 s including 32 generated tokens.
After full LLM swaps, repeated requests reused 5,995 / 6,113 tokens and
returned identical complete assistant messages. These HTTP prompts were
about 6K. First generations were capped at 32 tokens; visible content may
be empty while a model is still reasoning, so this checks the integration
without proving a completed-answer round trip.

The completed-answer coding control (20:42:39–20:46:34 EDT) used 290,000
characters of the pinned llama.cpp source context. Both first turns
finished normally with the visible answer `Python`, after 34 / 57 output
tokens. The client then omitted 127 / 244 characters of reasoning and
asked for a concrete implementation check:

| Model | Second prompt tokens | Reused prefix | New prompt tokens | Second request seconds | Reused on repeat after full swap | Repeat seconds including swap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek + DSpark | 71,833 | 71,819 | 14 | 1.242 | 71,831 | 10.628 |
| Qwen3.8 + adaptive MTP | 72,038 | 72,011 | 27 | 1.058 | 72,033 | 9.078 |

Both second generations produced 32 tokens. Repeated complete assistant
messages agreed exactly after full LLM swaps. Request times include
generation and any activation; unlike the native table, they do not
isolate suffix preparation. One run per model. An earlier coding control
capped at 32 tokens passed reuse but did not establish a completed first
answer. A broader code-risk query completed for DeepSeek but exhausted
Qwen's 2,048-token limit in reasoning; it was excluded from this table.
The final simpler source-code query explicitly required nonempty visible
content and `finish_reason = stop`.

An unmodified OpenAI Python SDK 2.6.1 client (20:26:31–20:27:13 EDT)
completed alternating DeepSeek/Qwen/DeepSeek requests and a Qwen streaming
request through the same loopback route. The client ran in an existing
CPU-only container, image
`sha256:9e1fb4c395b9c406136e10aa445b8784d06bca3839623b52cbe4a3b231a157a8`;
the native runtime alone executed models on the Spark.

Workstation checks are deferred until all optimization implementations
settle, at the owner's request. No package ships before the owed checks.

Provenance: `spark-b`, GB10, driver 580.178.04, SDK
`aarch64-e0a0c85c42806fb1`, CUDA 13.4.92; turn-reuse changes over `c0150de`.
Both models configured at 262,144, DeepSeek chunks 2,048 and Qwen 4,096.
Prompt text `ppl.txt`, SHA-256
`c7156148ecaa12b6416cf816540d8dede2014982554a835e61076f0dd8bf0c2d`;
DeepSeek target `8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234`,
DSpark `dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5`;
Qwen target `c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`,
prefix drafter `056a750e3a90be3ae6a4b12bb963ce45290aaa6f52b5ba9799e777d491f80aea`.
The HTTP coding input is the first 290,000 characters of the externally
prepared 256K source prompt (content SHA-256
`ab1912e3acfb4c95ec96a6fe7be980ad2243a80cffeab64a98c0f67b2d702378`);
the prefix SHA-256 is
`0303031edd637fb4fb6ddcb0ac883749f2f82a7fe8f862fc6aeca914ddbbbbcd`.
Raw reports and HTTP controls stay in `~/scratch/m3-turn-reuse/` and
the supervised job logs on the Spark.
