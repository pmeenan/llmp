<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 26 representative teacher-forced screen

This batch scores **1,023 targets given 1,024 fixed input IDs**, including
unscored BOS 2, from the start of the existing War and Peace corpus. It is a
bounded prose likelihood screen, not general model, speed or serving
qualification. Native base **d5539c8** uses ordinary products and device masks;
shared-Q8, row-invariant and store policies remain off. A separately
norm-fused native arm supplies independent native-own noise data. The scalar
warm diagnosis at bd8e54f is not relabeled as this batch.

At **2026-10-05 01:29:38 UTC (October 4 EDT)**, upstream default-branch HEAD
was refreshed with `git ls-remote` to
[609ca419abecebdc5a059498a613680bd3aa847f](https://github.com/ashhart/TensorFold/tree/609ca419abecebdc5a059498a613680bd3aa847f).
Its freshly read [package version](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/src/tensorfold/__init__.py)
is **0.6.5**; the [pinned coverage table](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md)
still declares Gemma 26 on MLX only. This observation is frozen for this
batch and supplies no GB10 CUDA comparator.

The same-format reference remains llama.cpp
**b29c606e28a01b1bc8c1351026a0fa6e616bf6c4**, b10964, image
`ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`.
Reference fusion and graphs are enabled. Both engines use one sequence,
context 4,096, F16 KV, FlashAttention, no drafts/templates/EOG stopping, and
completed full-vocabulary F32 likelihood publication.

The input is canonical long-context `ppl.txt`, 3,274,124 bytes, SHA-256
**c7156148ecaa12b6416cf816540d8dede2014982554a835e61076f0dd8bf0c2d**,
from the corpus documented in [corpus.json](../long-context/corpus.json).
The pinned raw approved GGUF tokenizer processes the full text with
`add_special=true`, `parse_special=false`. Only the first 1,024 IDs are
supplied to inference, as a shared little-endian I32 file. Record its exact
hash before model runs; both arms validate its size, BOS and vocabulary
bounds. Each input row j publishes head j, predicting input ID j+1 for
j=0..1,022; the last head is retained but not scored. BOS has no target NLL.

The first matched screen uses **eight teacher-forced chunks of 128 rows**
in each engine, with every row requesting a head. It is batched
teacher-forcing arithmetic, not 1,024 scalar decode units. Reference batch
and ubatch are 128 for this numerical shape control. These settings do not
constrain subsequent competitive prefill timing: that must measure the
fastest supported reference chunk sizes and qualify a native envelope
accordingly. Scalar arithmetic and optimized multi-request joining require
separate controls.

Native ordinary and norm-fused arms each dump 1,024 completed vocabulary
rows. Freeze native-own top-two margin movement p99 and row hashes before
cross-engine analysis. Reference own-repeat must be checked. Both engines
are teacher-forced on the same fixed IDs, so a head argmax mismatch does not
change later input prefixes. Report strict argmax differences, raw deltas,
chosen/target NLL differences and full-softmax differences; do not widen a
frozen noise allowance after observing an oracle discrepancy. Report mean
NLL and PPL over exactly the 1,023 target transitions using FP64 sums.
The original literal calibration remains immutable and separate.

Raw vectors and logs remain external; source/binary hashes and aggregate
results accompany the final report. Work executes through installed
`spark-job --gpu`, with a 600-second limit and stop-on-fail. Full-logit
analysis on the workstation uses `hostlock shared`. The prepared native
artifact and source checkpoint pins are unchanged from the
[native runner](../gemma-runner/README.md).
