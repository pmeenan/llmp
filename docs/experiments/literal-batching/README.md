<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Literal requests in native cohorts

2026-10-03–04. Qwen and DeepSeek literal `/v1/completions` requests now
join each model's native cohorts alongside chat requests. Admission,
capacity waits, cancellation, completed-unit retirement and pending model
switches use the same scheduling machinery. Each literal request starts
from a cleared branch, without a chat template or another conversation's
prefix reuse. Prompt scoring remains completed one-row teacher forcing,
interleaved with peers; generated rows join existing plain/speculative waves.
This does not add streamed literal output, a multi-prompt request shape or
jointly batched prompt-score products.

A carried literal response owns its score rows, text/row decoders and host
memory charge independently of the retired exchange. Partial prompt and
generation continuations preserve rows, Unicode offsets and sampling keys.
An untouched scoring session never settles an unselected or unrestored
slot. If the pending other-model request disappears during retirement,
original held branches resume before fresh same-model requests. A backend
that cannot continue carried literal work fails explicitly rather than
restarting its prompt.

## Native correctness controls

Spark A (`spark-c4e2`, GB10 sm_121, NVIDIA driver 580.178.04, SDK CUDA
13.4.92), base `5cea77c` plus this change. The pre-batching runtime was
built from `d4cdc30` on Spark B and copied to Spark A; all comparisons run
on A. Qwen uses the pinned NVFP4 target with MTP, context 16,384; DeepSeek
uses the community IQ2_XXS/Q2_K GGUF and the 0731 DSpark drafter, context 512, with
`wave_form = "speculative"`. Both expose four slots and 256-row prompt
chunks. Each runtime arm uses a fresh service, a one-token literal warmup where
specified by the harness, and a 128 GiB spill budget. Sampling is greedy
except the explicit seeded control.

| Control | Result |
| --- | --- |
| C1 echoed prompt scores and generated top scores | All serialized choices, probabilities, offsets and usage exactly match the pre-batching serial implementation, for both models. |
| C1 zero-output exact-ID scoring | Same equality, including the first null supplied-token score. |
| C1 seeded sampling, `logprobs: 0` | Same equality with temperature 0.7 and seed 7654. |
| C1 unscored generation | Reply, finish and usage match exactly. |
| Three scored literals beside one chat | All four complete with 200. Each literal's full choices/scores/usage equal its C1 control; row counts and generated greedy maxima are checked. |
| Qwen natural EOS | The reply is `Hi`, two generated tokens including `<\|im_end\|>`; both token scores and usage match serial, with EOS text omitted. |
| Full-context zero-output prompt | At context 512, scored rows exactly match the old serial scorer; unscored output is empty with 512 prompt tokens and zero generated. A 513-token prompt is refused with 400 `context_length_exceeded`. |
| Scored partial-prompt model switch | The 144-token prompt pauses after 24 completed tokens; all prompt probabilities and usage equal uninterrupted execution after Qwen→DeepSeek→Qwen. |
| Generated-score model switch | The 30-token prompt generates 128 tokens, pauses after 39 generated and resumes; full echoed scores, text, offsets, finish and usage equal uninterrupted execution. |

The boundary control also found an existing scalar unscored-prefill defect:
its `>= context` guard rejected a zero-output full-context request that
admission accepted, returning 500 and stopping the runtime. Both scalar
prefill and prompt-session admission now accept exactly the context, and
still reject an overrun. The old serial *scored* path already accepts it
and supplies the full-row numerical reference. The plain boundary control
qualifies response/usage, not equality of its tiled intermediate state.

Short mixed-cohort equality is scoped to these inputs. The longer throughput
screen below reproduces Qwen's existing arrival/width sensitivity; it does
not establish universal C1/C4 equality or same-history MTP parity.

## Literal throughput screen

Same-session serial-before → candidate → serial-after, on Spark A, Qwen
NVFP4 with MTP. The raw 30-token prompt asks for a long numbered ocean-fact
list, with a 128-token output limit. It is unscored, greedy, and identical
across requests. All requests return 200, finish by length, generate the
full 128 tokens and report zero cached tokens. Throughput counts completed
usage tokens over first request start to last response completion,
including prompt work and queueing. It does not count stream chunks.

| Runtime | Concurrent requests | Completed tokens | Wall seconds | Completed tok/s |
| --- | ---: | ---: | ---: | ---: |
| Serial before | 1 | 128 | 3.728 | 34.33 |
| Serial before | 4 | 512 | 13.702 | 37.37 |
| Cohorts | 1 | 128 | 3.813 | 33.57 |
| Cohorts | 2 | 256 | 5.191 | 49.32 |
| Cohorts | 4 | 512 | 9.095 | 56.30 |
| Serial after | 4 | 512 | 14.085 | 36.35 |

C4 gains 50.7–54.9% versus the two serial bookends; C1 is 2.2% slower
in this single short screen. The serial controls and cohort C1 share one
exact decoded signature. C2 has one reply with that signature and one
with another; C4 has one and three respectively. Both variants have the
same length/usage. The arithmetic and adaptive-depth differences between
solo and joined waves are existing behavior, not changed by this route
integration. This is a throughput screen, not a quality exception or an
M3 cross-engine gate. Peak memory was not sampled in these cells.

## Verification and reproduction

Native runtime identities (SHA-256):

- Pre-batching serial: `3ce17a89b688c7c6c25a66478d6a45ac3549fa1dff89ad7e1c2ab926ecc384a1`.
- Final candidate screens/throughput: `f1ee73b98a64e20bee69e0e689156b3b43c4b9b67b861ae45d805cd96e5a21ad`.
- Boundary candidate: `29471bd269f7b5122066a6c551f5ed45240cc58620317418cc690896bb8fb58f`; it precedes only the initial-row invariant's explicit fatal guard, with the same boundary/scoring behavior.

The measured harness SHA-256 for both final model screens and continuation
controls is `87709c5ac4d165d478621a8342ef1b4ea8987c23f1016b8af7e9b88d82dcdbd9`.
The throughput receipt records
`28f94fc36e9cef7015afbfa3f176128a3626d252dccd2fab4651d3c7da8c6cc9`,
before the additional short mixed-score/EOS/family controls, with the same
throughput requests and completed-work checks. Boundary uses
`3e6fdc0b51f98da752fdd274f909e4bb64a114a05200809a9567e045057a2a60`.

Prepared artifacts: Qwen target
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`,
MTP `8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`,
DeepSeek target
`cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac`,
DSpark `dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5`.
Qwen tokenizer/template come from installed
`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6`; DeepSeek uses its artifact's
GGUF tokenizer/template. Source-lock SHA-256:
`440f03cdb52921c6c55843819e6ac950a5b2c4aafdc01055e52a0af3117ce32e`.

The reusable [native harness](native_controls.py) requires an explicit
Spark admission/retirement probe accepting minimum free GiB and refusing
other model processes, GPU compute processes or containers. The measured
probe is `~/scratch/m3-final-launch/llmp-spark-preflight.py` on A, with
at least 105 GiB free before and after each service. Every qualified
service retires with exit 0; jobs use the installed `spark-job`,
`--gpu`, a 600-second timeout and an explicitly waited successful result.
Raw receipts, responses, configs, logs and kept state stay outside Git in
`~/scratch/literal-cohorts/` on A; summaries also live in the workstation's
`~/scratch/llmp-m3-literal-cohorts-2026-10-03/results/`.

```sh
python3 -B docs/experiments/literal-batching/native_controls.py \
  --binary build/spark-native/src/runtime/llmp-runtime \
  --baseline "$HOME/scratch/literal-cohorts/baseline-runtime" \
  --library-dir "$PWD/build/spark-native/lib/llmp" \
  --preflight "$HOME/scratch/m3-final-launch/llmp-spark-preflight.py" \
  --out "$HOME/scratch/literal-cohorts/new-screen" --case screen
```

Use `--family deepseek --context 512` for the DSpark screen; `--case
boundary`, `throughput`, `continuation-prefill` or `continuation-generation`
for the other controls. Continuation cases need no baseline executable.
Output directories must be fresh. Run each case under supervised Spark
GPU ownership and wait it before handing the host over.

Spark B passes the final full locked native build/test suite: 1,543 tests,
98.18 seconds of test wall time. SDK format and tidy on the six changed
translation units and their headers pass. REUSE, embedded-header and
portability-boundary checks pass. Focused controls cover exact
teacher-forced rows and MTP injection, prefix spill/restore, refusal/retry,
callback/settlement failures, untouched spilled-scorer cancellation,
full-context scalar/scored/plain sessions, carried Unicode offsets and
memory charges, mixed routes, model-switch ordering and unsupported
continuation refusal. Workstation/package checks remain deferred under
the owner's M3 instruction; no package ships on this evidence.
