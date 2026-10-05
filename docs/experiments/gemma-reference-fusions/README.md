<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma reference norm-fusion diagnosis

Both pinned reference norm-fusion families independently change the
representative Gemma 26 outputs. Neither family alone nor their combined
policy reproduces the full production reference. No native policy is selected and the
[representative quality failure](../gemma-quality/README.md) remains open.
This is a numerical diagnosis, with no competitive timing claim.

The [protocol](PROTOCOL.md) fixes the same 1,024 War and Peace IDs, eight
128-row chunks, context 4,096, F16 KV and full-vocabulary outputs. Every
later prefix is teacher-forced, including after a changed argmax. There are
1,023 scored targets and 1,024 retained heads. The unchanged native-only
calibration has zero p99 top-two margin movement; no differences below are
waived or used to widen that allowance.

| Reference policy | Mean target NLL | PPL | PPL above full production | Strict head argmax differences from full production |
| --- | ---: | ---: | ---: | ---: |
| Original full production fusion | 6.979519368234117 | 1074.4018512197504 | — | — |
| Only RMS_NORM → MUL → ROPE | 7.018928092803102 | 1117.5880248240849 | 4.019555% | 456 / 1,024 |
| Only RMS_NORM → MUL → ADD | 7.033855147758917 | 1134.3954533124192 | 5.583907% | 470 / 1,024 |
| Both norm families, no other fusion | 7.080982285121138 | 1189.1360156228518 | 10.678887% | 475 / 1,024 |
| Original unfused, byte exact to native ordinary | 7.0751330644224675 | 1182.200799205535 | 10.033392% | 476 / 1,024 |

Each allowlist repeats byte for byte on all 1,024 complete heads. Against
unfused/native, norm/RoPE has 475 strict argmax differences and 5.465465%
lower PPL; norm/residual has 478 differences and 4.043759% lower PPL.
Norm/RoPE preserves only row 0 byte for byte with unfused/native;
norm/residual preserves none. Neither preserves any full-production row
byte for byte. These effects do not identify an exclusive single-family
cause: the families, routing decisions and other production fusions may
interact. The combined policy repeats all 1,024 heads byte for byte but is
0.586636% worse in PPL than unfused/native, with 495 strict argmax
differences from it and no byte-exact baseline row. Its effect is not the
sum of the two individual changes. The full production policy remains the
quality reference; none of these restrictions replaces it. No same-input isolated-kernel result or instruction-level cause
is inferred from these full-model controls.

Large distribution changes remain visible alongside aggregate PPL:

| Allowlist against baseline | Maximum absolute logit delta | Maximum full-softmax total variation |
| --- | ---: | ---: |
| Norm/RoPE against unfused/native | 41.26120948791504 | 0.999997183428797 |
| Norm/RoPE against full production | 39.99172592163086 | 0.9999998011424712 |
| Norm/residual against unfused/native | 34.91945123672485 | 0.9999898339725548 |
| Norm/residual against full production | 36.85745620727539 | 0.9999998986622871 |
| Both against unfused/native | 35.66283130645752 | 0.9999968131637206 |
| Both against full production | 38.43495559692383 | 0.9999990802623664 |

## Actual selected operations

A controller-only diagnostic logs the pinned `try_fuse` decisions without
requesting graph intermediates. The stock policy's complete 1 GiB output
matches the original fusion-enabled acquisition byte for byte; the `none`
policy matches the original unfused/native acquisition byte for byte.
Policy banners confirm that the instrumented controller actually executes.
These checks authenticate this diagnostic setup before attribution.

| Original stock-policy host selection | Decisions across eight chunks |
| --- | ---: |
| RMS_NORM → MUL | 968 |
| RMS_NORM → MUL → ROPE | 480 |
| RMS_NORM → MUL → ADD | 720 |
| SOFT_MAX → RESHAPE → ARGSORT → VIEW → GET_ROWS → RESHAPE → SUM_ROWS → CLAMP → DIV → RESHAPE | 232 |
| Expert scale MUL → routing-weight MUL → eight VIEWs → seven ADDs | 240 |
| SCALE → UNARY → SCALE output softcap | 8 |
| Total | 2,648 |

Each norm/RoPE allowlist process selects exactly 480 of its own family;
each norm/residual process selects exactly 720. Each combined process
selects exactly 1,200, preserving both families' individual counts. The optional norm/RoPE
VIEW → SET_ROWS continuation is admitted by the pinned gate but is not
selected in this measured graph. These are host-side selection counts,
not CUDA launch counts or graph replay counts. They identify additional
applicable optimization transfers; they do not select a production native
policy or establish the performance of these paths.

## Source and reproduction

Measured on physical Spark `spark-c4e2`. The same-format oracle, artifact,
input and context pins are in the protocol. At task entry, **2026-10-05
02:14:23 UTC**, upstream TensorFold HEAD remained
[`609ca419abecebdc5a059498a613680bd3aa847f`](https://github.com/ashhart/TensorFold/tree/609ca419abecebdc5a059498a613680bd3aa847f),
version **0.6.5**. Its pinned
[README](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md)
still lists Gemma 26B-A4B on MLX, so it supplies no applicable CUDA Gemma
comparison for this GB10 batch.

The diagnostic's repository base is `c8edc06`; no native source is changed
or rebuilt for these reference-only runs. The retained native comparison
is the original `d5539c8` acquisition, with its existing immutable
calibration. Export exact upstream `ggml` sources using
`git -c gc.auto=0 archive b29c606e28a01b1bc8c1351026a0fa6e616bf6c4 ggml`.
The original source is MIT; `patch_controller.py` preserves its original
file and ports only the selected upstream predicates/launcher calls under
MIT and Apache-2.0. No prepared native source lock or production launcher
is changed.

Only `ggml-cuda.cu` is rebuilt: SDK NVCC **13.4.92**, pinned-image GCC
**13.3.0**, explicit `sm_121a`, upstream fast-math flags and graph support.
The image retains its CUDA **13.3** runtime, cuBLAS **13.5.1.27**, and all
floating-point CUDA math launchers. The controller translation unit's
compiled batched-pointer kernel performs integer pointer arithmetic.
The compiler change is explicit, and the stock/none full-head fidelity
checks are required; this is not represented as an unmodified-image build.

| Measured artifact | SHA-256 |
| --- | --- |
| Exact upstream controller source | `523470d6604755b82d0208414ce40f1378941b10bc1349763bbdf02edaab9634` |
| Initial diagnostic controller source | `d8418ebdca43650c9953693a546a77f19cc43681ee16fd3c8881077773f6a23d` |
| Initial diagnostic controller library | `7ddbd89e764b6eccbe2900d3cbde177f9f10896b02e7332b01a482577278bb97` |
| Combined-capable controller source | `d74c89feeaed4320ea8e1351b435a23618f66fad65eccbdc17a258254b92dcd6` |
| Combined-capable controller library | `aef05aaf605e2cfdcab7227b05903d62a22c505a1767c75f785a4c594d1d5d29` |
| Original image CUDA library | `5a13585ed1dc0263639e47f5154b0ea51c542df555e391a7b753daa73a56934e` |
| Existing explicit diagnostic C API source | `eac301a6130cf0e7543e1fa2b90c70b7005f46a9fc45121c9add6dbba7695c2a` |
| Existing diagnostic C API executable | `e291d8864695c1e4c80a1324fb0758ce127700584384aa3069239ee28cf593f0` |
| Original/full-stock complete heads | `c07c711bfc732ce498182917c0cc3cfd74f21374975e9fa69c90ec88461cc256` |
| Original/none-policy complete heads | `1c66029f1f088e04a7a4001fbdd0e24afbb97ad3b9e85f20fb73c58292e4b340` |
| Norm/RoPE first and repeat complete heads | `d53884358905393845f7a101a40210b4e2abd61694ba099e3017d72d5b87ae3b` |
| Norm/residual first and repeat complete heads | `8522bb9a466ee93c7308c2dd94051f27f2d9e9df31a1988eefdce315c098518d` |
| Both first and repeat complete heads | `879abf3088e5ff995fb97e39fc5aa1d86b3926779da099c531a07cf7807ec7cb` |
| Combined measured full-row analysis source | `edb818486c37dc3517f942dd49c4402b5a87ff4f9b819cdace650c6c02a52b70` |
| Initial measured full-row analysis source | `c1d553f5b608d7375675e7c36d0fb310ef94c429968c1e4ec4bf4b1cc5f9cc38` |

The initial controller produced the two separate norm-family controls;
the later controller adds `both`. Both controller versions passed the
complete stock/none fidelity checks independently. Original sources,
executables and acquisitions are retained, rather than retroactively
assigning them the later controller identity.

Use fresh external scratch for the source export and controller output.
`JITLLM_REFERENCE_ROOT` overrides the scripts' default
`~/.local/share/jitllm/gemma-reference-fusions` directory. Preserve the
original measured controller when rebuilding the combined-capable version.
Copy `patch_controller.py` there and run
`build_controller.sh` through the installed supervisor. The source digest
is checked before compilation. `run_reference.sh POLICY NEW_OUTPUT_NAME`
uses the existing `gemma-quality/llama_quality` executable and explicit IDs;
compile that C API harness as described in the earlier representative
report if reproducing from scratch. The policies are `all`, `none`,
`norm_rope`, `norm_add` and `both`.
The wrapper accepts only `controller.so` or `controller-both.so` through
`JITLLM_REFERENCE_CONTROLLER`; a fresh final build uses `controller.so`.
Its Docker invocation preloads the selected controller
and leaves both fusion/graph disable variables absent. Each model output
must be new. Compare stock/none heads against the original acquisition
before the allowlist runs.

`analyze.py FUSION_ROOT QUALITY_ROOT` compares the individual norm families;
append `both` for the separately acquired combined control and separate
output. Both modes authenticate the old calibration and
baseline file hashes, reject changed repeats, and use the existing
representative analyzer's finite-F32/FP64-normalizer helpers. Outputs are
exclusive; the calibration is never replaced. `selection_counts.py LOG...`
aggregates the policy banners and host selections from official logs.
`analyze_test.py` exercises the real analyzer with a four-entry vocabulary,
including signed zeros, changed repeats and altered frozen evidence. Raw
vectors, row tables, binary copies and logs remain external.

Installed supervised Spark jobs completed the controller build, two full-head
fidelity controls for each controller version, six allowlist acquisitions
and both full-row comparisons, retaining the discrepancies without changing
native policy or bounds.
