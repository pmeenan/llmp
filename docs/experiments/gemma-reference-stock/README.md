<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma stock routing and scaled-reduction diagnosis

Removing either stock routing fusion or the specialized scaled expert
reduction changes every retained representative head. Both omissions
repeat byte for byte, and neither reproduces native ordinary. This
supports checked reuse of the complete applicable stock contracts;
it does not identify a reference defect or select a native policy.
The [representative quality failure](../gemma-quality/README.md) remains
open, as do optimized batching, context, performance and memory gates.

[Protocol](PROTOCOL.md) fixes the same 1,024 War and Peace IDs and eight
128-row chunks as the earlier comparison. Every prefix is teacher-forced,
so changed predictions do not misalign later numerical comparisons.
The new controller's unrestricted `all` output matches all original stock
heads byte for byte before these omissions are interpreted.

| Stock-minus policy | Own repeat byte-exact heads | Byte-exact heads against stock/native | Strict argmax differences against stock | Against native | Maximum raw delta against stock | Against native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Remove routing/top-k fusion | 1,024 / 1,024 | 0 / 0 | 463 / 1,024 | 491 / 1,024 | 43.952613830566406 | 37.03509330749512 |
| Remove specialized scaled reduction | 1,024 / 1,024 | 0 / 0 | 469 / 1,024 | 491 / 1,024 | 37.625356674194336 | 36.64202308654785 |

For both omissions the first stock byte change is row 0 and the first
stock argmax change is row 2. These are complete-head strict counts,
not outside-noise counts; winner-margin allowances were not evaluated.
The original native calibration remains immutable. Neither omission
resolves the full production-policy discrepancy, and no full NLL/PPL scan
or new quality threshold is inferred from this first screen.

The following are **selected full-distribution rows**, not maxima across
the corpus or a quality pass:

| Row | Why retained | Routing removed: TV against stock | Scaled reduction removed: TV against stock |
| --- | --- | ---: | ---: |
| 0 | First changed bytes / fixed BOS head | 0.142034540 | 0.063876084 |
| 2 | First changed argmax | 0.653838475 | 0.729553662 |
| 127 | Last head in first chunk | 0.000028952 | 0.000104307 |
| 128 | First head in second chunk | 0.007196761 | 0.004844610 |
| 512 | Fixed middle boundary | 0.000207370 | 0.001425441 |
| 1023 | Final retained, unscored head | 0.507592361 | 0.321790194 |

Selected target NLL at row 2 (target ID 208088) is **10.2454269074** for
stock, **9.7247389996** without routing fusion and **11.1365108482** without
specialized reduction. The script retains corresponding native/stock/
omission NLLs for every selected scored row, without averaging those
samples into corpus PPL. The final retained row is not scored.

## Actual gates and fallback

The stock controller logs 2,648 host `try_fuse` decisions across eight
chunks. Removing routing eliminates exactly its 232 ten-node
SOFT_MAX/top-k/weight-normalization decisions. All other counts remain
unchanged: 968 RMS_NORM/MUL, 480 norm/RoPE, 720 norm/residual,
240 specialized scaled reductions and eight output softcaps.

Removing specialized reduction eliminates its 240 seventeen-node matches
and replaces them with **240 MUL → MUL and 240 seven-ADD fusions**.
Routing, norms and softcap retain their original counts. This is a
specialized-kernel omission with smaller stock fusions retained, not a
wholly ordinary expert reduction. Only this omission removes the upstream
allocator dependencies for the disabled specialized reduction. The
routing omission preserves them. Counts describe host selections, not
CUDA launches or graph replays; traced runs are not competitive timings.

Pinned `moe-weighted-reduction.cu` takes canonical contiguous F32 expert
outputs, per-selected-expert scale, routing weights and destination. It
computes `(expert * scale) * weight` and accumulates selected contributions
in slot order 0 through 7 under upstream fast-math flags. This is a different arithmetic
contract from native DeepSeek's explicitly rounded two-input reducer,
which also has different width and selected-expert count. No scale can be
dropped or that existing reducer silently substituted. Native Gemma keeps
separate scale/routing MULs and an ordered eight-expert sum; its planner's
existing ARGSORT/TopK does not implement the stock ten-node routing fusion.
The new upstream CUDA units are not currently compiled into native jitLLM;
reuse needs checked wrappers and a reviewed source-build inventory update.

## Provenance and reproduction

Measured on physical Spark `spark-c4e2`, repository base `24e9cbe`.
This is an external reference diagnosis, with no native/CMake changes.
The original native data remains the `d5539c8` acquisition. Latest
TensorFold was refreshed at **2026-10-05 02:51:37 UTC**:
[`609ca419abecebdc5a059498a613680bd3aa847f`](https://github.com/ashhart/TensorFold/tree/609ca419abecebdc5a059498a613680bd3aa847f),
version **0.6.5**, whose
[README](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md)
still lists Gemma on MLX. There is no applicable GB10 CUDA Gemma comparator
at that task pin. The same-format b29 llama.cpp image remains the oracle.

| Artifact | SHA-256 |
| --- | --- |
| Exact upstream controller source | `523470d6604755b82d0208414ce40f1378941b10bc1349763bbdf02edaab9634` |
| Generated stock-minus controller source | `fce9dd32edde4891ed180a4081d155ef921a9d1b50e619375caac58e2f6ba291` |
| Stock-minus controller shared library | `e55676ec9583ca7a954cfedbfa1deffa29f77d4f6a30bba5af766f6d2dfffd03` |
| Original image CUDA math library | `5a13585ed1dc0263639e47f5154b0ea51c542df555e391a7b753daa73a56934e` |
| Existing C API harness source | `eac301a6130cf0e7543e1fa2b90c70b7005f46a9fc45121c9add6dbba7695c2a` |
| Existing C API executable | `e291d8864695c1e4c80a1324fb0758ce127700584384aa3069239ee28cf593f0` |
| Original/new-stock complete heads | `c07c711bfc732ce498182917c0cc3cfd74f21374975e9fa69c90ec88461cc256` |
| Original native complete heads | `1c66029f1f088e04a7a4001fbdd0e24afbb97ad3b9e85f20fb73c58292e4b340` |
| Routing removed: first and repeat | `65be622067bf90bdbf68c51ae95e829f801ae13b6056af76649545a5134dcc73` |
| Specialized reduction removed: first and repeat | `5d2a54fcfe04c8815c01ecaece6ac65da47cf1479e3f8ee49c3eb57dac7a5b4b` |

Export `ggml` at exact pin
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` to fresh external
`~/.local/share/jitllm/gemma-reference-stock`, copy `patch_controller.py`
there, and run `build_controller.sh` through the installed supervisor.
`JITLLM_REFERENCE_ROOT` can name other fresh scratch. The script checks the
source identity and builds only the controller with SDK NVCC 13.4.92,
image GCC 13.3.0, `sm_121a`, upstream fast-math and graph-support flags.
Original image CUDA 13.3/cuBLAS 13.5.1.27 math kernels remain linked.
The original GGML source/predicate ports retain MIT attribution; no native
source lock, compiler pin or runtime dispatcher changes here.

`run_reference.sh POLICY NEW_OUTPUT_NAME` accepts only declared policies
and `controller.so`, uses the existing `gemma-quality/llama_quality` C API
harness and pinned GGUF, and leaves fusion/graph disable variables absent.
Compile that harness using the earlier representative report if starting
from scratch. Use `all` and verify its complete output against the original
stock file, then run `no_routing` and `no_reduction` twice each. Existing
output directories are refused. `no_norm_mul` and `no_softcap` are defined
but unmeasured here.

`screen.py ROOT QUALITY_ROOT POLICY` verifies the original calibration,
IDs, baseline complete-file hashes and own repeats before publishing an
exclusive result. It checks finite F32 rows, all-head bytes/argmax/raw
changes, and FP64 shifted normalizers only for the labelled selected rows.
`selection_counts.py LOG...` checks each process's policy banner and
summarizes host fusion decisions. `screen_test.py` exercises signed-zero
byte rejection, first-changed-row score alignment, repeat/calibration
refusals and undeclared policies with a four-entry vocabulary. Raw heads,
logs, row tables and binary copies remain outside Git.

Installed supervised Spark jobs completed the controller build, stock
full-head fidelity, four primary omission acquisitions and both complete
byte/argmax screens; no native policy or acceptance bound changed.
