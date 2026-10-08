<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dense Gemma 31B reference norm attribution

Allowing only the pinned reference's norm/MUL/RoPE and norm/MUL/residual ADD
fusions reproduces **all 1,024 stock 31B full-vocabulary heads byte for byte**,
including an independent repeat. These families suffice to reproduce the
observed stock arithmetic at this 128-row teacher-forced shape. This is
distinct from the [negative combined 26B result](../gemma-reference-fusions/README.md).
It does not imply a shared routed-input cause, a reference defect or native
policy adoption. The [native31 baseline](../gemma31-runner/README.md) still
fails its reference quality gate; no native source changes in this packet.

The [protocol](PROTOCOL.md) fixes the 31B checkpoint, common IDs, immutable
noise calibration, original reference math and attribution order. The
[proposed native contracts](native-design.md) describe separate default-off
checked wrappers with primitive fallback, preserving raw K-as-V and kept
view consumers. No initial cache-store elision is proposed.

## Complete-head identity

The external controller first proves `all` matches the actual stock31 ON
1 GiB acquisition and `none` matches the actual OFF acquisition. Only after
both complete-file comparisons pass are `both` and its repeat acquired.
Every process completes eight 128-row calls and publishes all 262,144 F32
logits per row. Every later prefix uses the same forced IDs regardless of
changed argmax choices.

| Policy | Complete-head byte identity against stock | Against native/OFF | Strict argmax differences against stock | Against native | Maximum raw delta against stock | Against native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `all` fidelity | 1,024 / 1,024 | — | — | — | — | — |
| `none` fidelity | — | 1,024 / 1,024 | — | — | — | — |
| `both`, repeated independently | 1,024 / 1,024 | 0 / 1,024 | 0 | 330 | 0 | 27.125676155090332 |

The `both` result and repeat each have stock's complete-file SHA
`c4b9b73ac7d9c62c62ff1daabc09b36bc716df5a5b124076d0b888bfc42fb0c1`.
The original native/OFF file remains
`87d2274ad1420412885cd118a33dc79e614cc743989f96eb0c953e5f8ffc3720`.
The original 31B calibration SHA
`6ab1fd0bc4a39c303c112ff476b96faa0527a9fa8242cc67b3abd5873350f038`
is authenticated before acquisition and after analysis; its zero allowance
is never replaced or widened.

This screen computes strict argmax counts, not new outside-noise counts.
Its complete byte identity establishes that `both` has the existing stock
scores (NLL 3.5773256064214842, PPL 35.77772905610885 across the original
1,023 target transitions). Those numbers are inherited from the authenticated
baseline scan, not a new NLL scan. Native remains NLL 3.713697329892001,
PPL 41.00513611237161, 14.610785% above stock. No redundant full NLL scan or
new quality gate is introduced.

The following **selected** full-distribution rows are diagnostic samples,
not maxima over the corpus or a quality qualification. No first stock byte
or argmax change exists, so only the predefined boundary rows are sampled.
Every sampled `both` distribution matches stock exactly.

| Row | TV against native | Native target NLL | Stock/`both` target NLL |
| --- | ---: | ---: | ---: |
| 0 | 0.008854438978 | 17.291443426653 | 17.199490681359 |
| 127 | 0.344056600720 | 0.835435830323 | 2.411967419079 |
| 128 | 0.190963415459 | 1.435059524506 | 2.991416602920 |
| 512 | 0.285751112369 | 0.898230392090 | 0.367612460852 |
| 1023 | 0.000011950620 | unscored final retained head | unscored |

## Actual selections and provenance

The stock controller records 2,896 host `try_fuse` decisions: 968
RMSNorm/MUL, 960 norm/RoPE, 960 norm/residual ADD and eight output softcaps.
`none` records zero. Each `both` process records exactly 1,920 decisions:
960 norm/RoPE and 960 norm/residual ADD; no simple norm/MUL or softcap fusion.
The actual matched forms have three nodes; no norm/RoPE/cache-store match
is selected. These are host decisions, not CUDA launches or graph replays.
Logging does not retain graph intermediates or change model tensors.
This batch contains no competitive timing or memory-peak measurement.

Measured on physical Spark-b `spark-56f5`, diagnostic tree base `38ca137`.
The original native acquisition retains measured ancestry `c8edc06` and
its eight-file source/test manifest. No new native model output or native
build is substituted. At this task entry, TensorFold HEAD was freshly
refreshed at 2026-10-05 03:24:09 UTC to `609ca419...`, version 0.6.5; the
[pinned Gemma recipe](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/docs/recipes/gemma-4.md)
still declares MLX 26B, with no applicable CUDA 31B comparator.

| Artifact | SHA-256 |
| --- | --- |
| Exact upstream controller | `523470d6604755b82d0208414ce40f1378941b10bc1349763bbdf02edaab9634` |
| Generated V2 norm controller | `d74c89feeaed4320ea8e1351b435a23618f66fad65eccbdc17a258254b92dcd6` |
| Measured controller shared library | `58a55e02f0e1a70f82b1bd633a6fa0e2586033ebdabfc0a28aa5ed321dfa0b98` |
| Original image CUDA math library | `5a13585ed1dc0263639e47f5154b0ea51c542df555e391a7b753daa73a56934e` |
| Existing explicit-ID quality harness source | `eac301a6130cf0e7543e1fa2b90c70b7005f46a9fc45121c9add6dbba7695c2a` |
| Existing quality C API executable | `e291d8864695c1e4c80a1324fb0758ce127700584384aa3069239ee28cf593f0` |

The exact original math remains linked from the digest-pinned b29 image.
Only the external controller is compiled, with declared SDK NVCC 13.4.92,
image GCC 13.3.0, explicit `sm_121a`, upstream fast-math and graph support.
Integer pointer helpers compiled into the controller do not replace the
floating-point launchers. The exact source patch and allocation dependencies
are the previously committed V2 implementation, unchanged for dense31.

## Reproduction

Export `ggml` from exact llama pin
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` to new external
`~/.local/share/llmp/gemma31-reference-fusions`, copy the checked
`patch_controller.py`, and run `build_controller.sh` through installed
`spark-job --gpu`. Use the existing authenticated 31B quality files under
`gemma31-quality` and its preserved C API executable. `run_reference.sh`
changes only the external roots and raw31 filename from the committed26
wrapper. Run `all`/`none`, compare their full outputs to the actual31 ON/OFF
files, then acquire `both` twice in new output directories and compare all
bytes. `LLMP_REFERENCE_ROOT` may select another fresh external scratch.

`screen.py ROOT QUALITY_ROOT both` authenticates the fixed31 calibration,
IDs and baseline complete files, checks every finite F32 cell and candidate
repeat, counts complete-row byte/argmax/raw changes, and computes FP64 shifted
normalizers only for labelled selected rows. `screen_test.py` controls signed
zero, next-ID target alignment, repeat/calibration refusal and unknown policy.
`selection_counts.py` validates banners and summarizes actual host decisions.
Aggregate [results](results.json) retain the identities and selected metrics;
raw vectors and per-row tables remain external.

Official supervised jobs `m35-gemma31-norm-fidelity1` and
`m35-gemma31-norm-screen1` complete 10/10 and 3/3 steps respectively, rc0;
the latter includes five bounded analyzer controls. Both jobs have 600-second
limits and successful official waits. Raw logs/aggregates are external under
`/tmp/llmp-m35-coordination/gemma31-reference-fusions-raw` and Spark-b job
directories. No native default, source-lock, model support or serving change
lands with this diagnosis.
