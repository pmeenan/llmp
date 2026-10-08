<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared Q8 preparation with original Gemma MMVQ consumers

Gemma2 and Gemma3 now share one exact Q8_1 preparation across selected products
of the same F32 input while retaining each original GGML MMVQ consumer. The
bounded C2 screens preserve all full heads, generated IDs and state bytes.
Decode takes 0.358%/0.514% less time in the short Gemma2/Gemma3 samples; whole
paid-cycle differences stay within the off-bookend movement. This supports
narrow adoption, not a whole-model speed or stock-parity claim.

The final extension covers the original device-selected MMVQ routes through
eight columns in nine ordinary Q8_1-input formats: Q4_0, Q4_1, Q5_0, Q5_1,
Q8_0, Q4_K, Q5_K, Q6_K and IQ4_NL. Gemma2/Gemma3 runner and serving defaults
select this preparation sharing. Missing/false device choices, MMQ routes and
other shapes retain ordinary products. The graph selector is transient and
uses stored device metadata; it neither asks CUDA during CPU graph construction
nor retains a launch context. Public contexts, slots and chunks are unchanged.

## Arithmetic, ownership and funding

The shared cache keys the exact input-tensor pointer, with one planned Q8
producer whose activation storage and lifetime are funded by normal planning.
The new consumer authenticates immutable packed weights, original F32 input,
packed Q8 producer, exact Q8 draw with 512-element row padding, source
parameters, readable weight tails and mutual disjointness. It builds the ordinary descriptor by
value and invokes the original device plan and kernel. It does not replace
MMVQ with the separate row-invariant VecQOneToken arithmetic. Single-column
quantized gate/up stays ordinary to preserve its GeGLU fusion; single-consumer
products need no shared preparation. Owner attention still carries the full
cohort through the product lambda.

The shared QuantizeQ8 checker now refuses malformed nonpacked strides/exact
physical extents; forged one-byte storage cannot authorize a full Q8 write.
G4's existing diagnostic uses the extracted pointer cache but keeps its
original VecQOneToken arithmetic and default policy.

Startup measurement preserves the existing full-envelope endpoints and also
measures every positive ordered owner-row composition with total at most eight
inside the configured per-owner and wave budgets. This covers intermediate
MMVQ activations that a maximum-row MMQ plan cannot prove by monotonicity.
Duplicate vectors are removed. Setup elapsed and budget changes are recorded
separately below; the composition sweep is not assumed free.

## Matched native factors

All times below are seconds, n=2 per policy, from one warm O/A/A/O cycle for
each family. Full prompt, three independent seed rows and eight joined warm
steps precede Clear off-clock;
physical state, plans and graphs remain retained. Paid prefill includes state,
inputs, masks and actual capture/replay. Three independent seed rows per owner
then run off-clock before 32 paid joined decode steps per owner, with GPU token
selection until the final full-head step. The reported paid sum joins these
two intervals; it is not a continuous serving request. Setup, model load, warm
traversal, Clear, the three seed rows and off-clock observation are excluded.
Own diagnostic runs write full heads during their diagnostic intervals and provide correctness
only; they are not timing samples.

| Family / arm | Prefill | Decode | Paid sum |
| --- | ---: | ---: | ---: |
| Gemma2 off first | 1.203830 | 0.572801 | 1.776631 |
| Gemma2 on first | 1.205080 | 0.571547 | 1.776627 |
| Gemma2 on repeat | 1.201000 | 0.571548 | 1.772548 |
| Gemma2 off bookend | 1.207400 | 0.574405 | 1.781805 |
| Gemma3 off first | 0.459802 | 0.465716 | 0.925518 |
| Gemma3 on first | 0.459853 | 0.463179 | 0.923032 |
| Gemma3 on repeat | 0.458972 | 0.463634 | 0.922606 |
| Gemma3 off bookend | 0.455744 | 0.465888 | 0.921632 |

Gemma2 mean prefill/decode/paid changes are −0.214%/−0.358%/−0.260%.
Off-bookend movements are +0.297%/+0.280%/+0.291%. Gemma3 changes are
+0.358%/−0.514%/−0.082%, with off movements −0.883%/+0.037%/−0.420%.
Decode ranges are disjoint in both short samples. Whole paid movement remains
within bookend drift; there is no sustained or HTTP performance claim.

All factor arms have exact final two full finite heads, 64 valid chosen IDs,
owner state hashes and work/capture/selected-policy contracts. Gemma2 executes
39 paid prefill units (3 captures/36 replays); Gemma3 executes 13 (3/10).
Neither performs paid lookahead planning. Bound-plan counters select
156/416 shared preparations/consumers for Gemma2 and 204/544 for Gemma3;
these count planned selections, not GPU launches. The two policies retain the
same capture/fusion/owner/mask schedule.

## Focused correctness and integration

The final prerequisite executes nine named positives, including every format
at columns 1..8 against complete ordinary MMVQ outputs. It proves eager and
poisoned changed-input capture replay, protects unchanged peer columns and
trailing input/output/weight guards, and atomically refuses malformed source,
stride and alias cases. Host controls prove device-specific caps/fallback,
empty/false callbacks with zero Q8 nodes, existing C1 fusion and G4 policy.
Metadata controls independently enumerate all configured small compositions.
All pass with no skips/errors. Full suites, context ladders and repeated
unchanged factors were not run under the focused optimization policy.

The final current-main integration executes exactly one own-on application per
family against authenticated source6 own-off correctness oracles. Both pass
with exact finite full heads, 72 IDs, final state hashes and work, including
retained Clear, GPU-token continuation, refusal atomicity, partial departure,
spill/restore and checkpoint replay. The shapes include independent scalar
three-row warm fragments as well as C1/C2 decode. Startup observations (n=1,
not a startup A/B) are 0.370013s for Gemma2 and 0.587234s for Gemma3.

| Current startup component | Gemma2 | Gemma3 |
| --- | ---: | ---: |
| Activation bytes | 48,234,496 | 46,137,344 |
| Scratch bytes | 10,485,760 | 10,485,760 |
| Host input bytes | 2,097,152 | 2,097,152 |
| Plan-floor bytes | 1,197,936 | 1,663,072 |
| Plan-floor increase vs original off | 45,760 | 59,840 |
| Total derived budget bytes | 8,604,565,504 | 5,309,989,376 |

Activation, scratch, host input and total derived budgets match the original
off oracles; host plan floors grow as shown. Setup time includes the measured
shape sweep and its existing setup work; no isolated sweep cost or startup
speed claim follows. The first integration controller refused a missing
copied completion record before any application. That failed job remains
failed; after verifying all 27 pinned baseline files, a namespace-only retry
completed both applications and retired successfully.

The first factors use source6 on 002ed48. Final source7 carries the extension
onto d8ba59b; the combined correctness gate binds its current ELFs and source
to the original source6 own-off oracles, without pooling old timings. Earlier
build failures were pre-test CMake/compile contract errors, retained separately.
The first G3 controller used the wrong vocabulary/file contract and refused
after its completed own-off; the next overwrote its retained-proof variable
after completed own-on and one cycle. Both batches remain failed and their
cycle timings are excluded. Only reauthenticated own proofs carry as
correctness; all four reported G3 factor cycles are fresh.

## Family dispositions and remaining work

Gemma2/Gemma3 adopt original-consumer sharing only for authenticated selected
chains. Compatible additional quant formats and broader adjacent-product
routes beyond the original eight-column MMVQ selector remain explicit work,
not mathematical exclusions. G26/G31 ordinary-consumer sharing remains OPEN:
the prior G4 VecQOneToken diagnostic rejection does not reject this consumer.
Dsv4/DSpark and Qwen GGUF already cache compatible Q8_1 inputs; their distinct
routed/shared pairing policies are retained. Native Qwen per-16 NVFP4 Q8 and
MXFP8, EXL3 Hadamard/trellis and image BF16 input representations cannot consume
this Q8_1 preparation contract. No format conversion is introduced to manufacture
a consumer. Assistant-specific and other-cohort adoption need their own checks.

## Replay and identities

Run on Spark B (NVIDIA GB10, driver 580.178.04), official
`aarch64-c09daba6ac31edee` SDK and locked llama.cpp
`d81235049384534c167caea52b85a694f6103d14`/patched GGML. The final build
compiles both probes, runtime and affected controls. The nine-format operands
use current pinned GGML quantizers; no new quantizer or arithmetic kernel is
introduced. First-resolved private cuBLAS13/Lt payload hashes are
`ee7c1657a03695c0de790aa79e34cef9c9649756b1846b11dd44caca20ba656b` and
`ba3b942f4ea43433b65e8c492a7b73de887534dc20146506ddaa4a78c79c5d30`.
[Aggregate identities](results.json) retain source/build, positive XML,
failed/positive installed jobs and complete output hashes, independent of raw
logs. Raw payloads/logs stay external and may be deleted at milestone close.

The task-entry upstream check at 2026-10-08T11:06:27Z fixes TensorFold main
`f8fe17d24629aedabf90bbf78279dd776e6d62e7` and Python
`ed78d6fc204d89d90b045bf033d6551e7714f3a1`. Those pinned recipes supply no
applicable same-format Gemma2/Gemma3 CUDA workload. No TensorFold/stock
reference or parity result is claimed; this is a native factor.

The checked-in [Gemma2 probe](../../../benchmarks/gemma2_joint_prefill_probe.cc),
[Gemma3 probe](../../../benchmarks/gemma3_joint_prefill_probe.cc) and
[GPU operand control](../../../tests/unit/ggml_ext_ops_test.cc) are the reusable
harnesses. Build the two `jitllm_gemma{2,3}_joint_prefill_probe` targets using
the official Spark preset. Authenticate the build receipt, actual ELF and
first-resolved libraries before and after use. Supervise every run with the
installed `spark-job` GPU lock/600s limit, check foreign processes and
MemAvailable, preserve return codes, source retirement markers and kernel
boot/cursor guards. Keep correctness observations outside any cycle timers.

Prepared artifacts are supplied externally from the standing approved stores:

- Gemma2: `~/.local/share/jitllm/gemma2-import-20261007/artifacts/eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870`.
- Gemma3: `~/.local/share/jitllm/gemma3-import-20261007/artifacts/8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb`.

Standing replay inputs are retained independently of raw experiment output
under `~/.local/share/jitllm/references/gemma-prefill-copies/{gemma2,gemma3}`.
Supply each `input{0,1}/ids.i32` as little-endian I32, verify complete SHA/size
before use, and retain them through milestone cleanup. Missing stores must be
supplied with these exact authenticated bytes, not retokenized substitutes:

| Family / input | IDs / prompt prefix | SHA-256 |
| --- | ---: | --- |
| Gemma2 /0 | 4391 /4352 | `102c7b555b1caed5aed3d9880a173aae153f8f8dc1534e6a4c58685c243b6c0c` |
| Gemma2 /1 | 4903 /4864 | `a930726bd964ae88ef0448f50a51d2e376ce2487313f26063ab84c1b0b41d77f` |
| Gemma3 /0 | 1319 /1280 | `4499653ec0de5def2cd171e9e1ca797741ee372332fef9fb6d59645ba08cfe9f` |
| Gemma3 /1 | 1575 /1536 | `0d77cf38d5a4d3e620edb5df11ef293a07fb5e9f72665a3c28df7ab4e4c48bb9` |

Each contains prefix + 3 seed + 32 transition + 4 departure IDs. The own proof uses
teacher input IDs; cycle decode uses generated greedy IDs. For either family,
set `probe`, `artifact`, `ids0`, `ids1` to the corresponding paths, and provide
a new private output directory:

```sh
"$probe" "$artifact" "$ids0" "$ids1" "$output" own \
  bounded-roots device-masks prefill-ahead lookahead-capacity=2 \
  owner-prefill flexible-owner-prefill shared-q8
```

Omit `shared-q8` for the ordinary probe control (the runner's ordinary default
is enabled; this explicit diagnostic CLI preserves flag-absent historical
replay). For warm factors use `cycle` instead of `own`, four fresh output
directories in off/on/on/off order and identical remaining flags. Validate
complete finite final heads/64 IDs and state/work/selection equality before
interpreting intervals. Own proofs additionally require G2 prefill 2 rows,
heads 74 rows/final 2 rows, G3 heads 74 rows/final 2 rows with no separate prefill file,
and 72 IDs. Vocabularies are 256000/262208. State JSON must affirm GPU continuation,
restore and refused-work atomicity; source teardown must precede return 0.
The new default/funding correctness gate deliberately repeats no timing factor.
