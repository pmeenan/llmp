<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma segmented graph/plan controls

These controls validate a graph/plan foundation, using the approved 26B-A4B
geometry and quantized tensor types with structured synthetic weights. They
load neither the checkpoint nor a serving runner. The 31B contract has CPU
descriptor/shape coverage only. Whole-model likelihoods, reference-engine
quality/performance, production batching, device masks, paging and exact
continuations remain separate exit gates.

## Inputs and paid work

`gemma4_exec_test` builds complete layer 0 local and layer 5 global arithmetic
through the same builder used by full chunks. Explicit diagnostic metadata
selects hidden input, one complete layer and an optional tied head; default
production descriptors still require token input, all layers and a head.
K2816, shared N2112, expert N704, 128 experts/top8, Q8_0 dense products,
Q4_K fused expert gate/up and Q5_1 expert down match the approved 26B fixture.
Each matrix row is constant, with coefficients varying by role, output row
and expert, so an independent FP64 oracle evaluates actual-width products
analytically. Learned norms, nonconstant hidden inputs, routes, cache values,
RoPE, windows and positions remain nontrivial.

The scalar reference has no graph/planner/node inspection. Separate controls
use the exact decoded uploaded matrix-row coefficients. Cache cells have
physical ring identities and F16 storage. Segments use distinct input values
and past positions 1279+7×slot; a two-row chunk crosses local ring wrap. Solo,
two- and four-segment executions join row-local products but retain separate
attention/cache/mask inputs. Returned output and complete active-layer K/V
buffers are compared across ordinary replay and CUDA capture/replay.

The short paid screen uses four warm runs and 32 measured complete bound
runs, with a primitive→candidate→primitive bookend. CUDA events bracket the
same provider stream used for HtoD staging and execution. Every iteration
restages the pageable diagnostic host sources on that stream, then executes
all normalization, quantization, products, routing, attention and cache writes.
The interval includes these copies and enqueue gaps. It excludes tokenization,
weight preparation/loading, plan construction and graph capture; it is not a
serving rate. Both padded local/global reference masks are staged even in a
one-layer diagnostic. A future qualified device-mask path can change this
balance and needs its own measurement.

## Correctness

The post-review focused Spark run passes 72 tests: 11 graph metadata,
5 plan/input/region, 30 base operand/fusion validators, 20 extended validators
and 6 GPU tests. The final post-rebase run passes 23 tests (11 graph, 5 plan
and 7 GPU), including explicit norm policies across solo/two/four segments.
The final descriptor-copy admission fix passes the 16 graph/plan tests.
Controls cover actual 26/31 descriptors, tensor identities,
quantized expert strides, independent slots, bounds/aliases, inactive retained
storage, causal/window mask authentication and fresh position shape reuse.

Conditioned complete local/global layers have worst ideal-oracle NMSE about
2.68e-7 and maximum absolute error 0.001255. The tied Q8_0 head with final norm
and softcap has maximum error 0.006706. The optional shared-Q8 mode gives
bit-exact joined-vs-solo outputs in the checked two/four-segment cases; ordinary
products meet the checked 0.002 envelope. K/V are bit-exact joined-vs-solo in
both modes. Restaged ordinary and captured repeats preserve output and the
entire active K/V buffers exactly. These are synthetic controls, not model
quality or a general numerical-equality guarantee.

Forward full 512 NEOX frequency-factor RoPE ordinary/extended outputs agree
byte for byte; fused cache stores equal their F16 conversion. Factors include
nonuniform finite prefix values and a 1e30 suffix. Ideal FP64 rotation error
at positions 1279/1280 is 0.000322; at 262143 it is 0.068322. An independently
computed host F32 phase plus FP64 trig differs by up to 0.053919 at that long
position: host pow is not NVCC fast-math device pow. Long-position differences
remain an explicit pinned phase-rounding diagnostic, not a widened accepted
model-quality bound. All positions preserve pair energy within 2e-6.

The original zero-centered V-cache fixture is retained. It produces a
near-cancellation global attention-output projection and an ideal-oracle
NMSE 0.0087147 / max 0.262465 after the sandwich norm. Decoding actual uploaded
weights, modeling F16 caches and independently rounding RoPE phase do not
remove it; using observed attention alone gives NMSE 0.0087816 / max 0.26293.
Additionally modeling the locked MMVQ projection's Q8_1 input preparation
(32-cell blocks, roundf integers, F16 stored scale) reduces the downstream
independent oracle to NMSE 1.99e-10 / max 4.10e-5. This isolates quantized-input
rounding amplified by the following norm. The same original inputs produce
identical ordinary/fused-norm outputs and exact ordinary/captured repeats.
It does not establish the effect on real model likelihoods, nor permit a
whole-model quality exception.

## Bounded screen

Times are microseconds per paid run; one short screen, not a confidence
interval or a general shape policy.

| Layer / slots | Primitive before | Shared Q8 | Fused norms | Primitive after |
| --- | ---: | ---: | ---: | ---: |
| Local 0 / 1 | 555.803 | 566.528 | 559.739 | 573.382 |
| Local 0 / 4 | 823.445 | 924.985 | 830.481 | 812.373 |
| Global 5 / 1 | 609.218 | 625.888 | 630.048 | 611.456 |
| Global 5 / 4 | 833.589 | 929.296 | 843.152 | 838.806 |

Shared preparation loses on both four-slot cases and the global solo case;
the local solo result lies inside bookend movement. Fused norms supply no
clear broad win here. Shared-Q8 and generic GeGLU fusion remain default off;
no universal model policy is adopted. A later actual-model writer/dispatch
slice must compare its complete paid chain and matched reference engine.

The primitive baseline binds nine unfused learned RMSNorm/weight products
per complete local/global layer. With explicit diagnostic `fuse_norms`, all
nine bind fused across one, two and four segments. The head adds an unfused
final norm in its primitive control. These counts describe bound policies,
not selected production defaults. No RoPE/cache-store fusion binds in this
graph even for a solo segment: projection reshapes and per-segment views
miss the existing fusion pattern, and generic graph fusion is disabled.
Standalone factor-aware fused-store correctness does not establish its
graph selection. This optimization transfer and qualified batching remain
owed before supported model execution.

## Provenance and reproduction

Host spark-b (`spark-56f5`), NVIDIA GB10, driver 580.178.04,
Linux 7.0.0-1019-nvidia, SDK `aarch64-c09daba6ac31edee`; native locked source
build. Primary arithmetic is llama.cpp
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` Gemma graph and FFN/MoE builders.
Prepared GGML source identity and archive pins remain the source lock's;
the graph port adds retained MIT/Apache scope/notice obligations.

Builder controls started at baseline 39d4c2235f7d6c17f799aa3a32ea28d9fcf3d98f plus the reviewed
attention production snapshot SHA256
`266f9414c7fa0a4dc13cec121a18231c974045531161635e51e3f1d4dfc99847`.
The final graph worktree is based on the committed attention prerequisite
`5198d2f`. Its dependency executable sources match this snapshot exactly;
the base refresh changes no graph/plan/GPU control arithmetic. Final bounded
pre-copy public descriptor rank/type and expert-stride-domain refusals are
checked separately. The later GPU harness adds policy logging and a seventh
correctness test; the paid screen retains its original executable identity.
Post-review code identities:

- Final `gemma4_graph.cc`: e198aca94b0cfb569b401682319ed53260e9a6b858684c9cfdd3f4d3d59bc2c6
- `gemma4_plan.cc`: 973c14a36b2d39fe7b7c200eb0cc8a37a204085396163dde5953be65c9dbe325
- Final `gemma4_exec_test.cc`: 97a87b45488fb58b30013112bc9d997536742cd5ce11b6190b71cb4ffac614a6
- Final built `gemma4_exec_test`: 875608603969eaba1a3be1bbe971930788924d617f825f256cdec1e57171a757
- Screen/check16 `gemma4_graph.cc`: 53fcbc678b7f64dbb63d2bec09bb9b3ee7c103fdc8dc8a9a8c969f0dfef43676
- Screen/check16 `gemma4_exec_test.cc`: 7c4d1f29d43f9c1b5b390f4ae64d3a9f7d9a667e4fe89eabc49607bbc5d06b04
- Screen/check16 built `gemma4_exec_test`: 553c5d85b2a136a6da1bd2249345eee213b2a96fab2c5d1f7b0c70479531f2a0

The installed supervised jobs are `m35-gemma-graph-check16` (locked build,
all focused tests, rc0), `m35-gemma-graph-screen1` (screen, rc0),
`m35-gemma-graph-check17` (post-rebase 23 controls, rc0), and
`m35-gemma-graph-check18` (final descriptor admission 16 controls, rc0). Raw logs
remain under the host's private `~/.local/share/llmp/jobs/` and the session
scratchpad, outside Git. Full source sync used checksum `rsync -rlpc`, excluding
`.git` and `/build`. Reproduction, after locked source preparation/build:

```sh
LLMP_TEST_DATA=build/spark-native/tests/unit/data \
  build/spark-native/tests/unit/gemma4_exec_test
LLMP_GEMMA_LAYER_SCREEN=1 LLMP_TEST_DATA=build/spark-native/tests/unit/data \
  build/spark-native/tests/unit/gemma4_exec_test \
  --gtest_filter=Gemma4ExecTest.OptionalPaidWholeLayerScreen
```

Every Spark command runs through installed `spark-job --gpu`; workstation
compiles/tests run under `hostlock shared`. Final integrated source/full-suite verification belongs to the commit gate;
independent general/adversarial reviews cover this graph slice; no model-support status is inferred from these diagnostics.
