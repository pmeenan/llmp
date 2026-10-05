<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Checked standalone Gemma MoE primitives

The native kernel module can call the original pinned GGML routing and scaled
expert-reduction operators through checked, allocation-free wrappers. All 15
complete literal output files agree byte for byte with the original reference
image at rows 1/2/4/8/128. This is primitive availability and fidelity evidence;
Gemma graph dispatch, whole-model quality, performance and optimized batching
qualification remain separate. No production policy selects these wrappers.

## Contracts and fallback

`gemma_moe.h` declares two independent operations, each with zero scratch:

| Operation | Checked operands | Meaning |
| --- | --- | --- |
| Routing | F32 logits `[128,rows,1,1]`, F32 weights `[1,8,rows,1]`, I32 IDs `[8,rows,1,1]` | Original softmax/top-eight/renormalization, with the exact `2^-14` denominator clamp |
| Scaled reduction | F32 experts `[2816,8,rows,1]`, scales and routing weights `[1,8,rows,1]`, output `[2816,rows,1,1]` | Ascending selected-slot accumulation of `(expert * scale) * weight`, retaining original fast-math contraction |

Rows are bounded to 1–8,192 and operands use canonical contiguous layout,
except the IDs' required first-eight view. The actual Gemma graph already
uses `ggml_argsort_top_k`: a full 128-entry ARGSORT root and a zero-offset
first-eight view carrying its 512-byte row pitch. Routing requires the entire
root's caller-funded backing, including the tail; it writes only the first
eight IDs per row. It does not produce the full sort, router probabilities or
normalization intermediates. A caller needing those kept/debug values must
retain the original primitive producer chain. The standalone API checks the
declared top-eight consumer contract; it cannot inspect an external graph.
There is no repack or additional scratch allocation.

Both checks reject short storage, address-end overflow, invalid strides,
unsupported types/shapes, stale or cyclic views and writable operand aliases
before queueing work. Routing alias checks cover the unwritten root tail too.
Fixed shape bounds cover the original launchers' narrowed signed element and
grid counts. Caller-declared spans describe mapped bytes, rather than proving
catalog residency: the caller must fund and protect all operands through
completion and every captured replay. Successful submission is not completion;
unknown completion retains ownership until recovery establishes quiescence.
Finite payloads are a caller obligation, not a host metadata scan.

The reducer requires all three original inputs, including per-expert scales.
It is a different arithmetic contract from DeepSeek's width-4,096/top-six,
explicitly rounded two-input reducer. Neither that reducer nor its rounding
policy substitutes here. Existing Gemma primitive graph execution remains
available and unchanged.

## Controls and original-library comparison

Four CPU metadata controls cover the actual full-root/first-eight view,
maximum rows, compact-ID/full-sort refusal, exact clamp, full-tail aliasing,
short spans, stale/cyclic views, end arithmetic, types and strides. They also
refuse absent scales, wrong expert counts/widths and output aliases.

Two GPU controls cover literal ties and ranked routing, normalized weights
against independent FP64 arithmetic, complete scaled reductions, independent
row submissions versus joined calls, sentinel preservation in IDs 8–127,
fresh operands through a two-node captured replay, and clean refusal followed
by valid reuse. Cancellation inputs (`+1e20`, `-1e20`, then six small selected
contributions) make the reduction order observable. The fresh replay changes
experts with `nextafter(-value,1)`, exercising non-dyadic operands. Fixture
cleanup requires a completed fence before releasing graphs, contexts or device
storage; unproven retirement retains those owners until process teardown.

The external [client](oracle.cu) uses the exact reference headers and exported
`ggml_cuda_op_topk_moe` and `ggml_cuda_op_moe_weighted_reduction` symbols from
the pinned image. Only this thin API client is compiled; no reference floating
kernel source is rebuilt. The foreign GGML backend context exists solely in
this external diagnostic, never in native jitLLM. Literal native operands are
copied into independent client storage, both original operations complete,
and entire routing-weight, full-root ID and reduction-output files are
compared with `cmp` after retirement.

| Rows | Routing weights | Full 128-pitch IDs, including sentinel tail | Scaled values |
| --- | --- | --- | --- |
| 1 | Byte exact | Byte exact | Byte exact |
| 2 | Byte exact | Byte exact | Byte exact |
| 4 | Byte exact | Byte exact | Byte exact |
| 8 | Byte exact | Byte exact | Byte exact |
| 128 | Byte exact | Byte exact | Byte exact |

These finite literals establish neither universal compiler equivalence nor a
whole-model reference pass. No timing or memory-peak comparison was made.

## Sources and reproduction

Measured on NVIDIA GB10, physical host `spark-c4e2`, 2026-10-05 UTC. The
isolated source was based on `a9b5a92`, with this standalone slice; its original
implementation ancestry was `38ca137`. Native setup used SDK
`aarch64-c09daba6ac31edee`, NVCC 13.4.92, Clang host compilation and the pinned
GCC 16 runtime. GGML's compiler/upstream/license pins and fast-math flags are
unchanged. Build patch 0002 adds only the original `topk-moe.cu` and
`moe-weighted-reduction.cu` compilation units. Source contracts authenticate
their original implementation/header bytes and the actual compiled inventory.

| Identity | SHA-256 or pin |
| --- | --- |
| llama.cpp/GGML | `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, tag `b10964` |
| Newly prepared GGML tree | `026f1ac94af98011933e71cee240d0a405ac4220cb208356acf61aa2bcea6207` |
| Build patch 0002 | `67c5b094a83ae0871a21b350bce0b4228d0622aca191f9417ed4af5057054fcb` |
| Measured source lock | `7889fe441722fd9c95d39dd05bf268c86ef0d09abaf5de7efe72e83046a3fd9b` |
| Native GPU test binary | `2a43d5fc38e4edc50bff5107b27e7d809d67936baa6ca059272185e70723c3b5` |
| Build receipt | `af910cb38aeeddf819af640e8ea6dcc1b97f594ec0100bd98a073c291ad135ea` |
| Native validator `gemma_moe.cc` | `0a218c3b1f42b1eca56bfd342f74e37b87928efaed951e4a181928e1e5f1c267` |
| Native launcher `gemma_moe.cu` | `60957b0cbd458ba970fc1d121529cb64468fe6c9767e251749dbb4a2c06fd30e` |
| GPU fixture source | `f9a6826af66ed5e0bed980ffa23320238082a9e05f0fe82c9875c5e815d179cb` |
| Thin client source | `d3b33f5d16c1ec2ff5d6d3ae0530aef672920dabc6acf0eca69ff7b111e917d3` |
| Thin client binary | `3e710b3819c5dc7c30f7521b7e8601a877725c3686c2396ff63952cbe9a6df15` |
| Original image CUDA library | `5a13585ed1dc0263639e47f5154b0ea51c542df555e391a7b753daa73a56934e` |
| 20 literal input-file hash manifest | `decf4e69c7b2aec0114014fb17edaf08544a922c4aa1cbc461428dcdcd2f36d0` |
| 15 complete output-file hash manifest | `a8f1b7166af5d2531be696627304f3697bb131840c59fb2d8bc743979dfa0cd5` |

The last two manifests hash sorted UTF-8 `basename TAB file_SHA256 LF` lines.
Inputs are `rowsN-{logits,experts,scales,weights}.bin`; outputs are
`rowsN-{route,ids,values}.bin`, for the five row counts above. Raw vectors and
logs remain outside Git under Spark's
`~/.local/share/jitllm/gemma-moe-oracle-final/` and workstation
`/tmp/jitllm-m35-coordination/gemma-routing-raw/`.

The reference image is
`ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`.
Its math library was built with CUDA 13.3; the thin client uses native SDK
NVCC 13.4 and image GCC 13.3, linking those unchanged image math symbols.
This compiler distinction is retained despite the observed literal identity.
The client header tree is the pinned checkout's original `ggml` directory,
exported read-only with `git archive b29c606e28a01b1bc8c1351026a0fa6e616bf6c4 ggml`.
All four original source/header hashes are permanent assertions in
[the source-inventory test](../../../tests/sources/gemma_moe_test.py).

To repeat, prepare/build through the declared `mise run prepare` and
`mise run build -- spark-native --locked` workflow. Set
`JITLLM_GEMMA_MOE_ORACLE_ROOT` to a fresh `native` directory when running the
focused native tests. Supply the original header tree at
`~/.local/share/jitllm/gemma-reference-stock/ggml`; then use
[reference.sh](reference.sh) `build` and `run NEW_OUTPUT`, setting its
`JITLLM_GEMMA_MOE_ORACLE_ROOT` to the parent export directory and
`JITLLM_GEMMA_MOE_SOURCE_ROOT` to the source checkout. Compare every full
output with `cmp`. All builds/tests/container execution use the installed
Spark GPU supervisor.

TensorFold was refreshed at this task's entry, **2026-10-05 03:25:41 UTC**:
HEAD `609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5. Its official
README identifies Gemma 26B-A4B under MLX, without CUDA Gemma coverage, so
the pinned same-format llama.cpp image remains the applicable oracle.

Final focused Spark controls passed 10/10 with no skips: four CPU, two GPU,
and source lock/receipt/closure/original-unit inventory; all 15 original-image
output comparisons passed.
