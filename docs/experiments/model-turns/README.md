<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Pending model turns

2026-10-03. A different model at the queue's head can now pause a chat
cohort at completed units, run one substitute cohort, then resume the
original members ahead of later arrivals. The configurable resident-work
turn defaults to 30 seconds; this is a policy fallback, not a measured
throughput knee or a response deadline. Same-model slot pressure does not
preempt running responses. The substitute may batch peers ready at its
first admission pass, with no later refill or nested model pause (D-069).

Continuations preserve their exact branch identities, response buffers,
sampling state and original cached-token accounting. They protect logical
state and spill files from expiry/deletion while allowing GPU backing to
spill. Partial prompts resume their completed prefixes. A final prefill
checkpoint chooses its anchor before pausing; runnable generating peers
restore before their resumed decode wave. A substitute that would exceed
the spill budget refuses without deleting the original requests' state.

## Native HTTP controls

Spark A (`spark-c4e2`, NVIDIA GB10 sm_121, driver 580.178.04, SDK CUDA
13.4.92), base `5989eac` plus this change. Final runtime executable SHA-256:
`1bf2bdbabe405399243a13ee06b6cebe0783c7410474b9a6f0dc6133f4719f2b`.
All cells use greedy sampling, `model_turn_seconds = 1`, fresh service
state, a one-token Qwen warmup, and 256-row prompt chunks. Original Qwen
uses NVFP4 with MTP, context 16,384 and four available slots; the DeepSeek
substitute uses plain decode, context 512 and one slot. Spill budget is
128 GiB except the explicit refusal control.

| Control | Original prompt / output tokens | Substitute | Result |
| --- | --- | --- | --- |
| C1 generation pause | 70 / 128 | 200, one token | Full reasoning, content, finish and usage equal the uninterrupted run. |
| C1 partial-prompt pause | 5,408 / 32 | 200, one token | Same equality; the substitute completes before the original's first output. |
| C1 zero spill budget | 70 / 128 | 503, retained request state exceeds spill budget | The original completes unchanged. |

Every original has zero cached tokens, including the partial-prompt resume.
The pause log directly witnesses 1,024 of 5,408 prompt tokens processed and
zero generated, before the substitute completes; the harness requires a
completed prefix larger than the warmup and smaller than the full prompt.
Successful cells record both Qwen → DeepSeek and DeepSeek → Qwen swaps;
the substitute completes before the original request retires. The final
five services exit zero, with 116.743 GiB free and GPU/container/native
model probes clear after retirement. The supervised final batch takes
114 seconds; all steps are waited successfully.

Canonical decoded-signature hashes (JSON of reasoning, content, finish and
usage, sorted keys, ordered request list): generation and refusal
`9eddd86780ecdfb4e15c3061cf2f3384abfef055e5bfcd963dc3d85ba61b10fd`;
partial prompt
`7a3992145cfd5326fcde1b3eeb9229831e54c05182b1b4fb47871cf1df70855e`.
These are reply controls, not a native full-logit comparison.

The final kept state was independently checked against the actual spill
payload, rather than accepting its checksum metadata alone. Every
initialized 2 MiB extent is byte-identical and its fresh SHA-256 matches
both records: generation has 137 extents (287,309,824 bytes per arm;
concatenated SHA-256
`66bb14234f7d28026b70445922fe94bdea247337e3d7e5f86695469c55bb6e5d`),
partial prompt has 202 extents (423,624,704 bytes;
`10ea1037a8bc390b8469f5e06ef762fa6d3671901e4b89426276ee22bd4085b7`).
Their 197 / 5,439 processed token IDs, cursors and adaptive decoding
records also match. These include target and MTP initialized state;
uninitialized extents and full logits are outside this comparison.

An earlier four-request plain Qwen screen, before the final anchor guard
and restore barrier, gives two exact replies and two differing replies
against its first uninterrupted baseline. A second uninterrupted baseline
matches all four switched replies; the same two differences occur without
a switch. This reproduces Qwen's existing arrival/width sensitivity and
does not establish a switch regression. It also does not qualify universal
C4 HTTP exactness. The first comparison's nonzero exit is retained in the
external records rather than presented as a passing control. Existing
wave tests qualify heads/state under fixed wave shapes.

## Reproduction and provenance

Artifacts (prepared IDs): Qwen target
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`,
MTP `8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`,
DeepSeek `cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac`.
Qwen tokenizer/template come from the installed
`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6` model directory. Source lock
SHA-256 is `440f03cdb52921c6c55843819e6ac950a5b2c4aafdc01055e52a0af3117ce32e`.

The reusable [HTTP harness](native_turns.py) retains the measured request
and comparison logic, with its admission probe path explicit, baseline-only
selection for both cases and a strict partial-progress witness. Its measured SHA-256 is
`aa3f1e90ad3940481e42ab62e492b27f7bcac4cb9e2d06fcd471ba70412c8935`.
Supply a Spark probe that accepts minimum free GiB and refuses if another
model process, GPU compute process or container is present. The measured
probe is `~/scratch/m3-final-launch/llmp-spark-preflight.py` on Spark A;
the host must have at least 105 GiB free before and after each service.
Run each case through the installed `~/.local/bin/spark-job start --gpu`
with a 600-second timeout, then wait it:

```sh
python3 -B docs/experiments/model-turns/native_turns.py \
  --binary build/spark-native/src/runtime/llmp-runtime \
  --preflight "$HOME/scratch/m3-final-launch/llmp-spark-preflight.py" \
  --out "$HOME/scratch/model-turns/new-generation" --case generation --slots 1
```

Use `--case prefill --slots 1` for the partial-prompt pair. Output paths
must be new, outside the repository. Raw responses, stream events,
configs, logs, spill inventories and metadata remain under `~/scratch/model-turns/` on
Spark A and `~/scratch/model-turn-review/` on the workstation; aggregate
results above stand on their own.

## Focused failure and state controls

The fake-backend API tests cover a full cohort switch, exact carried text
and usage, original-before-later-arrival ordering, no initial queue timeout
on admitted continuations, substitute refusal, client departure, same-model
slot pressure, substitute batching with no later refill/nested pause, and
a substitute's slow reader yielding even when the public queue is empty.

`llm_scores_test` covers greedy and sampled generation spill/restore,
zero-retention held state, partial-prompt continuation without duplicate
processing, and spilling a held unclaimed conversation when a restored peer
needs more memory. Initialized fake state and histories compare exactly;
protected logical state cannot be released. Configuration controls check
the default, an override and refusal of zero seconds. Independent review
and adversarial challenge are clean. Spark B passes the full locked native
build/test suite (1,531 tests); SDK format/tidy and REUSE/header/boundary
checks pass. Workstation/package checks remain
deferred under the owner's M3 instruction; no package ships on this evidence.
