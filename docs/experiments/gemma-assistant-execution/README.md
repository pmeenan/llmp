<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native Gemma assistant component: bounded first controls

The native Q8_0 assistant matches the original pinned image byte for byte on
one frozen Gemma26 C1 fixture: a one-step query and three recurrent queries
at constant position 64. Every complete 262,144-value head and 2,816-value
post-projection agrees in both native own repeats. Borrowed initialized cache
bytes remain unchanged. This establishes a bounded arithmetic seam, without
qualifying native-target chain quality, optimized batching, performance,
serving speculation or target verification.

The component is target-owned and uses the existing PagedNode, catalog,
request cohort, RunnerResources, PlanCache and GraphRuns. Its graph uses
ordinary native primitives: Q8_0 products, GELU-tanh/GeGLU, learned RMS norms,
factor-aware RoPE and per-owner D256/D512 attention. It writes no target KV.
The initial target feature stays protected; later recurrence writes separate
component-owned storage. Full canonical heads remain the baseline.

## Two input policies

The actual native-target component control sets up both artifacts and checks
paired vocabulary admission before registration. It completes distinct owner
histories at positions 6 and 5, then runs one-owner and two-owner three-step
chains twice. Complete heads/features repeat exactly; funded witnesses of all
initialized target KV and retained final-normalized features remain unchanged.
This tests native ownership and independent slots. Those target inputs are
not the original reference fixture, and there is no cross-engine target-chain
quality claim.

The separate manual helper maps and funds the original **stage-zero inputs**:
query position 64, initialized endpoint 64, feature position 63, anchor 108,
local layer 28 and global layer 29, local capacity 1,280, global capacity 4,096
and read width 256. The original cache arrays are immutable operands, not a
native checkpoint. The helper admits exactly eleven hashed input files and
uses the same production assistant graph/plan. All padded cells are initialized;
visibility excludes the unwritten current position.

Native source and binary were frozen before acquisition. Independent one-step
and three-step native endogenous repeats retired successfully and were frozen
before the original recurrent inputs or outputs were exposed. A later replay
uses the original incoming feature/anchor at each step through an optional
branch already present in that frozen helper. Both the endogenous and posthoc
runs match all original heads/projections completely. Full byte identity makes
raw, chosen-score and distribution differences zero; no sampled distribution
scan or tolerance adjustment is used. Native own winner movement was zero.

| Evidence | Result |
| --- | --- |
| C1 original-input one-step and three-step, each repeated | All complete heads and post-projections byte exact |
| Posthoc original incoming tensors, unchanged binary | Same complete byte identity at all three steps |
| Readonly cache witnesses, before/after every repeat | SHA-256 `ae4cd1ef1872b506f75df1d0f5f9f542bae58b08aaff863914368bad05c8fc35` unchanged |
| Native-target C1/C2, unequal histories | Own heads/features repeat exactly; initialized KV and target feature witnesses unchanged |
| Captured staged fixture | One capture and six replays per helper invocation |
| Competitive latency, memory peak, optimized C2/reference | Not measured or qualified |

## Provenance and limits

Measurements ran on Spark using the installed GPU supervisor. The 23 measured
source files were based on `4973381`; additive adoption of `c2717bf` changes
none of their bytes. The frozen native helper binary SHA-256 is
`be9ec4b750ff560423660e6b4289544e2f2425daf026eae3f4e0aff30a233b0c`.
The native endogenous receipt is
`fd742b1662a3ea71a04076c7acbe586cefaefe0f1670624cca4cb2af7be79d60`.
[Provenance](provenance.json) records source, input, binary and official
retirement hashes; [aggregates](results.json) contain no raw logit samples.
The [protocol](PROTOCOL.md) gives reproduction boundaries. The
[original-image companion](../gemma-assistant-reference/README.md) defines
its independently frozen C1, serial C2 and physical batch-two evidence.
Only its C1 result is compared here.

The original floating kernels come from image
`sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`,
llama.cpp `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`. Native GGML uses prepared
tree `026f1ac94af98011` and the existing pinned CUDA compilation flags.
At implementation entry, 2026-10-05 08:24:50 UTC, TensorFold HEAD was
`609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5; its Gemma26 recipe
remained MLX-only with no same-format CUDA assistant comparator.

The first native staged run passed. A later closure-check revision refused
before any assistant kernel because coverage was checked before request materialization;
its freeze correctly refused too. The final revision checks full activation,
pool, weight and resource coverage inside the held request, before uploads
or launches. Those failed receipts remain external. No failed arithmetic case
was replaced or quality allowance widened.

Focused graph/plan, retained-feature and actual native component controls,
the final locked helper build and official endogenous/posthoc acquisitions
passed. Local REUSE/header checks covered 1,426 files, portability 392 sources,
and changed-source format/diff plus Python/JSON syntax passed. Raw arrays, input
packages, logs and calibration receipts remain external.
