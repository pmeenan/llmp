<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 current corpus dispatch

Stock selects routing fusion in layers 0–27 and29 on the current 1024-row
[quality recipe](../gemma-current-quality/README.md). No routing selection is
recorded for layer 28. Native `all` selects all 30. All norm-family and ordered
reduction counts match. This establishes a concrete dispatch difference;
it does not attribute the nine positive-margin quality disagreements to it.

| Selected original host chain | Count |
| --- | ---: |
| RMS_NORM → MUL | 121 |
| RMS_NORM → MUL → ROPE | 60 |
| RMS_NORM → MUL → ADD | 90 |
| Ten-node MoE routing | 29 |
| Seventeen-node scaled ordered reduction | 30 |
| Softcap SCALE → UNARY → SCALE | 1 |

Both Q and K norm/RoPE chains occur in every layer. Named attention and
shared/MoE residual norm/add chains occur in every layer; another 30 norm/add
outputs have generated node names. Every reduction layer is selected.
No norm/RoPE/VIEW/SET_ROWS continuation is recorded. These are actual host
selection records, not CUDA launch or graph replay counts. This selected-chain
logger does not report rejection reasons or validate an unselected candidate;
the historical memory-gate observations at other row widths are not reused.

The complete 1 GiB observed output is byte-identical to the frozen stock output
from Task57 (SHA `ce5fd2ec…`), including all 1024 full-vocabulary heads. The
normal recipe remains context 4096, batch=ubatch 1024, one sequence, all outputs,
F16 KV, local 2048/global 4096 cells, SWA-full false and unified false. FlashAttention,
normal fusion and graph support remain enabled. No graph readers, keep flags,
allocator hints or floating math are changed; no numerical allowance changes.
The unchanged native zero-repeat calibration still describes its own policy.
The nine-row quality failure remains open. No timings, policy adoption or
whole-model causal conclusion follow from this observation.

## Reproduction and provenance

Base `5e9ed59`; Task57's exact supplied IDs, approved raw26 GGUF, current
thin scorer 225c40af… and frozen reference are retained. Stage the existing A
`~/.local/share/llmp/gemma-reference-fusions/controller-both.so` privately as
B `~/.local/share/llmp/gemma-current-quality26/controller-all.so`, verifying
library SHA `aef05aaf…`. The existing controller source is `d74c89fe…`, compiled
with NVCC 13.4.92; all floating operators remain in the original b29 CUDA 13.3
image library `5a13585e…`. This is a reused rebuilt controller, not an
unmodified-image build. Policy `all` (banner 0) keeps original selectors and
allocator dependency hints. See [original controller source](../gemma-reference-fusions/patch_controller.py).

Run `reference.sh` once under installed `spark-job --gpu --timeout 600
--stop-on-fail`, with the existing Task57 private scorer/input/retirement helper.
Before interpreting the log, require exact 1073741824-byte output and streaming
SHA `ce5fd2ecd80d64f2e53252a8fe2be39f814881ea6e4e2d20a1f39aeed5b7be9f`.
Validate actual banner 0, creation capacities, 1024-row shapes and each logged
chain length. Checked scorer return follows explicit batch/context/model/backend
release; the wrapper checks its owned Docker CID absent. Official staging and
observation jobs both completed DONE0. Raw heads/logs/addresses remain external;
[results](results.json) record aggregates, source identities and official links.
No additional model arm, PPL scan, compiler or regression suite was run.

Fresh TensorFold primary main at 2026-10-06 00:25:07 UTC remains 609ca419…,
version 0.6.5. Its [primary Gemma recipe](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/docs/recipes/gemma-4.md)
still supplies MLX 26 and no applicable CUDA Gemma comparator. No TF run was made.
A next common-input diagnosis should find the first divergent block and isolate
its actual selected operation; matching a layer list alone is insufficient.
