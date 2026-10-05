<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native Gemma checked norm fusion screen

Implementation entered at `a9b5a92`; measured source uses `4f9be31`; final integration rebases additively to
`ed274f3`. Original compiled norm/RoPE launchers use separate
default-off controls for norm/MUL/RoPE and norm/MUL/residual ADD, primitive
fallback, preserved raw K-as-V and every diagnostic keep/view consumer.
The final-frontier residual may be produced by an intervening GET_ROWS. Its
paid primitive step remains materialized; only norm and MUL are deferred until
the checked ADD, retaining all descriptors and actual input lifetimes.
The first contract has no direct cache-store or intermediate allocation
elision and never enables broad upstream fusion. Existing runners, funding,
graphs, completions and state lifecycle remain shared.

At new task entry **2026-10-05 03:40:04 UTC**, fresh TensorFold HEAD is
`609ca419abecebdc5a059498a613680bd3aa847f`, package version **0.6.5**. Freshly
read [README](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md)
and [Gemma recipe](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/docs/recipes/gemma-4.md)
still declare MLX 26B-A4B, group-32/64 affine weights and an 8-bit router;
no dense31/CUDA Gemma comparator is available at this task pin.

Use the [approved dense31 baseline](../gemma31-runner/README.md) raw/prepared
artifact, original llama b29 digest image, complete corpus and freshly
prepared IDs. Keep context 4,096, F16 KV, eight teacher-forced 128-row calls,
all 1,024 complete 262,144-entry F32 heads and 1,023 next-ID transitions.
The original ordinary31 source/test manifest and failed reference quality
screen remain historical and immutable; the original noise freeze
`6ab1fd0bc4a39c303c112ff476b96faa0527a9fa8242cc67b3abd5873350f038`
is never replaced or widened.

First pass operand/metadata/planner/capture/source-preservation and complete
layer controls, existing26 regression and candidate31 checkpoint/spill/replay
controls. Then acquire candidate first and independent repeat, authenticate
complete bytes and freeze **candidate self-repeat** metadata before reading
the matched stock oracle. This freeze describes repeat stability only:
ordinary-to-candidate numerical movement is not benign noise and does not
enlarge the original ordinary allowance.

Compare with the authenticated retained stock31 full output. Complete-file
byte identity proves the existing stock NLL/PPL; no redundant NLL scan is
needed in that case. Otherwise report actual complete-head byte/argmax/raw
changes and needed target likelihoods without changing a gate. No automatic
adoption follows from the external reference attribution.

After correctness, use a short paid common-prefix bookend of ordinary,
candidate and the independently screened reference. Full vocabulary
publication and completion waits remain paid. Forced IDs, warm/reset/capture
work, requested outputs and retention funding remain explicit. Do not hide
native intermediate heads or reinterpret forced-prefix calls as generated
history throughput. Expand controls only where measured quality/performance
supports adoption. Default selection, optimized batching, context ladders,
physical peak-memory and full supported-model status remain separate.

Spark-b jobs use installed `spark-job --gpu`, timeout at most 600 seconds,
stop-on-failure and official waits. Local heavy work uses `hostlock shared`.
Raw vectors/logs remain external; aggregate identities and measurements
accompany the checked implementation. No source-lock patch or new upstream
CUDA compilation unit is needed for these already-built launchers.
