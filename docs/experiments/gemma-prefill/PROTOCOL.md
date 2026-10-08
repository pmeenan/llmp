<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# First Gemma 26 paid 8K prefill screen

This diagnostic measures paid prefill and subsequent one-row work on the
same physical Spark, with no quality, memory-peak or support pass implied.
The representative 1,024-row quality comparison failed against the
fusion-enabled same-format reference; the native heads matched the unfused reference
exactly. Neither result is relabeled as an 8K quality pass.

The native base is `d5539c8`. The same-format llama.cpp reference remains
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` (b10964), pinned container
`ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`.
Reference fusion and graphs are enabled. Native optional norm, shared-Q8,
row-invariant and RoPE/store policies are off; device masks are on. Both
use F16 KV, FlashAttention, one sequence, context 16,384, no drafts, chat
template or EOG stopping. The source checkpoint and native prepared
artifact identities are unchanged from the native-runner report.

At 2026-10-05 01:47:55 UTC (October 4 EDT), upstream TensorFold HEAD was
refreshed to `609ca419abecebdc5a059498a613680bd3aa847f` and its pinned
README and package version were read again: version 0.6.5, Gemma 26 MLX
only. This batch has no applicable TensorFold GB10 comparator.

The pinned GGUF tokenizer processes the canonical War and Peace corpus
`ppl.txt` with explicit BOS and `parse_special=false`. The complete corpus
SHA-256 is `c7156148ecaa12b6416cf816540d8dede2014982554a835e61076f0dd8bf0c2d`;
its source and trim markers are in the existing long-context corpus map.
The shared LE I32 fixture contains exactly its first 8,227 IDs: 8,192
prefill inputs, three post-prefill untimed one-row inputs, then 32 paid
one-row inputs. Its LE I32 SHA-256 is
`6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b`. Every later input is
teacher-forced from this fixture, keeping both engines on identical
prefixes despite the known policy-dependent argmax differences.

Each process initializes its engine and warms weights on the first six
IDs, then clears that discarded state. The prefill timer covers a fresh
8,192-row request through completed final full-vocabulary publication,
including host descriptors/staging, plan building, kernels, state growth
and copies occurring in that request. Reference batch is 8,192; the first
screen varies ubatch 512/1,024/2,048/4,096/8,192 and requests only the final
head. Native currently runs 64 chunks of 128 and publishes a frontier head
at each, so its extra 63 heads are paid and disclosed. No hidden
intermediate-head suppression is used. Reference eager context setup and
native setup remain outside the timer; this is not a total cold-load or
physical-peak measurement.

After prefill, each engine executes three identical untimed one-row
inputs. This covers the comparator's build/capture transition before the
32-unit timer; record their IDs and the common timed-start position 8,195.
Each timed unit scans the preceding complete vocabulary for CPU argmax,
then runs the next fixed input through completion and complete vocabulary
publication. The argmax is recorded, not used to change the shared input.
This is fixed-prefix decode timing, not generated-text equivalence. File
writes and printing are outside both timers. Reference getters prove
completion and copy a full vocabulary into a reusable vector, matching
native caller publication. GGML graph-reuse counts do not establish CUDA
replay counts; reference CUDA graphs are allowed, not asserted per unit.

First screen: compare the five reference ubatches and native 128; then
bookend the selected reference setting around native if the screen decides
a material prefill or decode gap. Later production optimization requires
checked output/state semantics and separate full-model quality controls.
No depth ladder or generic fusion policy is selected from this screen.
All builds, preparation and inference use installed `spark-job --gpu`,
600-second stop-on-fail batches. Raw heads and logs stay external.
