<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen twelve-column MXFP8 CTA screen — 2026-10-04

Keep the existing 16-warps-per-block schedule. The representative C4
control/candidate/control screen lowers verify median latency only
**0.116%**, while the controls move **−2.417%**. This does not establish
a useful gain. All measured tokens, full target logit rows and initialized
target/drafter states match exactly. The private candidate is rejected;
no production kernel, selector, default or calibration changes follow.

## One scheduling factor

The candidate changes only GB10's actual twelve-column MXFP8 product at
K=2,560 and N=10,240: the existing staged wide kernel takes two output
rows per warp and **eight** warps per block instead of sixteen. This is
the target QKV shape in 36 linear-attention layers. The profile gives it
928.125 MiB of codes plus scales across those layers; this static read
volume is not a current per-kernel timing or bottleneck attribution.

The existing [wide kernels](../qwen38-four-request-waves/README.md)
were selected from a sixteen-column microbenchmark. Its retained source
hardcodes sixteen columns, comparing two-row schedules with eight and
sixteen warps. The older reusable
[scheduling sweep](../qwen38-mxfp8-scheduling/README.md) accepts only one
through eight columns. This screen examines the unmeasured twelve-column
case; it does not repeat the rejected sixteen-column instantiation with
register spills, the narrow loading-order factor or the tensor-core
crossover at three/four/five columns.

Grid size changes from 320 to 640 blocks and block size from 512 to 256
threads. Total warps stay 5,120, with the same 30-KiB static shared input
plane per block. Each output retains its original lane's vector order,
sixteen ordered FMAs per decoded vector, scale FMA and XOR reduction.
Input staging, weight decoding, prefetch, PDL, strides, output layout and
precision remain unchanged. More blocks also repeat staging more often;
no occupancy or sole-cause explanation is inferred. Other devices,
shapes and actual column counts keep the original dispatch.

The existing `Qwen38OpsTest.WideMxfp8ProductsMatchLoneColumnsBitForBit`
passes on both freshly built binaries (804/801 ms). It invokes the actual
production launcher for the selected shape and compares complete uint32
outputs with actual single-column launches. It includes a twelve-element
padded input stride, underfilled column templates and nearby/tail shapes.
The candidate's exit witness confirms the selected twelve-column
2-row/8-warp path executed once in this operand control.

## Paid C4 wave screen

Spark A (`spark-c4e2`), GB10, driver 580.178.04, CUDA 13.4.92, SDK
`aarch64-e0a0c85c42806fb1`. Each fresh process uses four short frozen
fast-swap prompts, 96 generated outputs per slot, context 16,384 and
4,096-row prefill chunks. Target artifact
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`
and MTP artifact
`8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`
are unchanged. Requested draft cap 65,536 executes the selected 47,172-row
head. The harness uses fixed depth two for C4, normal wave lanes and the
unchanged two-request joined-drafting limit. Both binaries include the
adopted verify alpha/beta fusion.

| Arm | Verify median ms | Draft median ms | Process wall s |
| --- | ---: | ---: | ---: |
| control-before | 108.972 | 22.621 | 32.297124 |
| candidate | 107.530 | 21.591 | 31.422877 |
| control-after | 106.338 | 21.740 | 31.110894 |

Candidate verify latency changes by
`(candidate / mean control medians − 1) × 100 = −0.11611%`.
Control movement is `(after / before − 1) × 100 = −2.41713%`.
The separate phase medians are not a median of their sum. The phase clocks
pay runner input preparation, concatenation, planning, graph capture/replay,
execution and output copying. Caller-side work-vector construction and
post-return equality checks remain outside these phase clocks. Process wall
also includes loading, prefill and host equality checks; it is not
served-token throughput.

Every arm completes 43 verify waves, two verify captures and 38 replays.
All four per-slot token-history SHA arrays (prompt plus generated tokens),
complete target-row SHA
arrays and hashes read from actual initialized target/drafter state
agree across all three binaries/runs. The candidate-only witness records
72 host dispatches at cc1210/K2560/N10240/C12, grid640/block256; controls
record none. Its atomic host counter is **inside the measured prototype
phases**, while its sole print is deferred to process exit. The count
covers eager/capture dispatches, not graph-replayed kernel launches.

All three children exit zero and are reaped, source/binary identities stay
unchanged before and after each arm, and all six strong 105-GiB admission
and retirement gates pass with clear GPU/container/native-model probes.
The smallest recorded availability is 115.866 GiB. No HTTP ladder,
maximum-context, extra quality matrix or full production suite is run
for this rejected arithmetic-preserving factor. These exact controls do
not establish cross-engine parity or sampled quality bounds.

## Provenance and replay

Both executables are freshly built from checked `580aab0`, with only the
candidate's one-file private dispatch/instrumentation patch. Each 487-file
source inventory covers tracked source/build inputs, and differs only in
`src/kernels/ggml/jitllm_ops.cu`. The controller separately checks the
unchanged benchmark and operand-test sources. The ordinary 16-column
microbenchmark's retained source SHA-256 is
`8deb785975e92adb2ad6481e2e6fc6a2edabf75ea1d7d3e8ffb5b73f8dd22055`.

| Item | SHA-256 |
| --- | --- |
| Control compiled-source inventory | `34556cfa35be0168d903d2123dd489e37f18ba8533283dc35f3747770347d0ca` |
| Candidate compiled-source inventory | `b52b8c47b3b86aa7587f2b5990f91628e5b40a89af870984556d3e9055b94b8e` |
| Control benchmark | `b238b5be4b30318ae7ce5f8656fd1f10262e10bbaee934f641c1be6c9bef56ac` |
| Candidate benchmark | `4058f06226c93920c2aee731c650503e275ae34d89666833d4ec46315355f8e1` |
| Private patch | `a1674f13fa57a1156849cd06e6ffbb0fe92c117b0fb864301f8f6fa8afc041c9` |
| Wave controller | `2df542bfc4a9fffc44cd2b8d7859fee4187668429384854d8f96bde4d8f326a2` |
| Wave aggregate | `534b22da5d9b0e0a8397d4739ca425052437a3defc38dc7d2562f39fe0fe86bb` |
| Complete short prompt JSON | `d212009dadf1ddbf945c8dc7ad0214ba444236baf57c8ed9019c3ebe6b0805b4` |
| Benchmark source | `e7c95af532772e89c24d5315b36751e8ff8fab82bc10c2156a184aad3c700e72` |
| Operand-test source | `870170cb52a9bcba712b159a711e6f7105c625db1cfc617fed7199e997883dba` |

Use the unchanged [wave harness](../../../benchmarks/qwen38_spec.cc)
with the supplied target/drafter/tokenizer and frozen prompts:
`--check wave --tokens 96 --context 16384 --prefill-chunk 4096 --draft 3
--draft-vocab 65536 --slots 4 --wave-lanes on`. The wave check deliberately
uses depth two; the nominal `--draft 3` does not widen it.

Raw controllers, source snapshots, patch, frozen binaries, build/test logs,
per-arm specs and gate/retirement receipts stay outside Git at
`spark:~/scratch/mxfp8-twelve/` and
`/home/pmeenan/scratch/jitllm-m3-qwen-mxfp8-twelve-2026-10-04/` locally.
Installed GPU-supervised `mx12-control-build`, `mx12-candidate-build` and
`mx12-screen1` complete zero and are waited on; the model batch takes
96 seconds. One clone setup initially fails on an unstaged source directory
before compilation or model work, is corrected, and is excluded. The
production source is restored; the private candidate remains external.
