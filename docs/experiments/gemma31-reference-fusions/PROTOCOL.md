<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dense Gemma 31B reference norm controls

Start from native baseline `5b30341`; retain its actual measured ancestry
`c8edc06`, source/test manifest `01f2d058b4e177b2de73c3b9e91a0add432b178f357e54cd4e16e55a5fc8aab7`
and immutable 31B noise freeze
`6ab1fd0bc4a39c303c112ff476b96faa0527a9fa8242cc67b3abd5873350f038`.
No new native acquisition, production policy, quality allowance or model
qualification is introduced. Dense 31B has no routing; the 26B routed-input
attribution and combined-family result do not predict this result.

At task entry **2026-10-05 03:24:09 UTC**, a fresh TensorFold HEAD query resolves
to `609ca419abecebdc5a059498a613680bd3aa847f`, version **0.6.5**. Its freshly
read [README](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md)
and [Gemma recipe](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/docs/recipes/gemma-4.md)
declare 26B-A4B on MLX, affine group-32/64 weights and an 8-bit router;
no dense 31B or CUDA Gemma comparator is declared at this pin.

Reuse the [committed reference controller](../gemma-reference-fusions/README.md),
with its exact pinned-source patch and original CUDA math in image
`ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`.
The source pin is `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`; original controller
SHA `523470d6604755b82d0208414ce40f1378941b10bc1349763bbdf02edaab9634`,
generated controller SHA `d74c89feeaed4320ea8e1351b435a23618f66fad65eccbdc17a258254b92dcd6`.
Only external scratch roots and the approved raw 31B filename change in the
wrapper. No upstream checkout, native build inventory or source-lock patch is
modified. Host selection logging must not retain intermediate graph tensors.

Before attribution, require `all` to match the actual 31B stock ON complete
1 GiB file, SHA `c4b9b73ac7d9c62c62ff1daabc09b36bc716df5a5b124076d0b888bfc42fb0c1`,
and `none` to match the actual 31B OFF complete file, SHA
`87d2274ad1420412885cd118a33dc79e614cc743989f96eb0c953e5f8ffc3720`.
Then acquire `both` twice and require full-output byte identity. This policy
permits only norm/MUL/RoPE (including its eligible cache-store form) and
norm/MUL/residual ADD; all other generic fusions are excluded. Record actual
host selections and shapes, not CUDA launch/replay counts.

Use the existing explicit-ID quality C API harness, approved same-format raw
31B GGUF, and its freshly prepared first 1,024 IDs including BOS2, SHA
`b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610`.
Keep eight teacher-forced 128-row calls, context4096, F16 KV, FlashAttention,
no template/drafts/EOG. Changed predictions never change later prefixes.

The first screen authenticates the original freeze and baseline files, every
finite F32 cell, all-head bytes/strict argmax/raw deltas and candidate repeats.
Compute FP64 full-softmax TV and target NLL only for labelled fixed rows
0/127/128/512/1023 and first changed stock bytes/argmax. Score row j against
ID j+1; the final retained head is unscored. Selected rows are not corpus PPL,
outside-noise counts or a quality gate. A full NLL scan is owed only if needed
to decide native adoption; no adoption is part of this initial screen.

All work uses installed Spark-b `spark-job --gpu`, per-job timeout at most
600 seconds, stop-on-failure and official wait/retirement. Local heavy scans
require `hostlock shared`. Raw vectors/logs stay external in
`gemma31-reference-fusions`; aggregate results, source/library identities and
reusable reproduction code accompany the report. Native wrapper/planner work
remains a proposed default-off design; selection requires model evidence.
