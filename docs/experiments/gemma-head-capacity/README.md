<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma bounded head publication capacity

Both approved Gemma runners now separate input-row capacity from pinned head
publication capacity. `Gemma4Options::max_head_rows` is immutable after runner
construction: zero retains `max_rows`, preserving manual teacher-forcing and
assistant callers. The serving adapter sets it to configured slots. Setup
requires `slots <= effective max_head_rows <= max_rows` before artifact access,
provider initialization or state reservations. Input chunk defaults and all
arithmetic policies are unchanged.

A wave checks its effective output mode before input allocation, placement
checks, host grants, state growth, planning or staging: state-only requests
zero heads, frontier requests one per owner, and all-head requests every input
row. Over-cap requests preserve completed vectors, positions, initialized
state, cached plans, graph counters and charges. The planned contiguous F32
logits shape and copy length are checked before staging. The pinned destination
remains fixed for the runner's lifetime and captured replay.

Startup still measures maximum-row frontier, state-only and full-feature
inputs. All-head sizing instead probes the cap-sized equal/ragged row budgets
at every owner count and both context endpoints. Retained features preserve
their full input-row envelope and force the actual head mode when needed.
Default-zero caps retain the original startup sweep and 128-row all-head path.

**18 focused Spark B checks pass, with zero failures or skips:** three CPU
bounds controls, seven runner GPU controls and eight two-profile serving
controls. They cover cap4 legal-four/refused-five ragged heads, untouched peers,
valid continuation and checkpoint restore, captured replay, cap1 maximum-row
frontier input, state-only KV equivalence, full feature rows, legacy 128-row
all-head replay and serving literal likelihood/scoring stop/resume. Both
profiles have reduced-cap boundary/frontier/replay/restore and serving controls;
the new full-feature and exact state-only KV-equivalence cap controls are 26B.

The catalog controls validate actual backing bounds. With 262,144 F32 logits
per row, cap4 reduces the runner's total pinned staging by **at least 124 MiB**
relative to the legacy 128-row setup. In the production 12-slot/chunk16 fixture,
the output extent is 12 MiB, and all runner-owned pinned staging together is
below the old 16 MiB output buffer alone, for both profiles. Precise total
staging values were not printed. These are catalog bounds, not GPU activation
peaks or a speed comparison.

Reproduce with the pinned SDK's locked Spark configuration, building only
`gemma4_runner_test`, `gemma4_runner_gpu_test` and `gemma4_serving_gpu_test`.
Run the `Gemma4Runner` head-capacity/invalid-envelope tests; all `Gemma4HeadCapGpu`,
`Gemma31HeadCapGpu`, `Gemma4SoloHeadCapGpu`, `Gemma4FeatureHeadCapGpu` and
`Gemma4HeadCapacityGpu` tests; the legacy `MaximumAllHeadRowsReplayWithSeparatelyFundedCallerVectors`;
and both serving parameters for pinned publication capacity, literal teacher
forcing, maximum literal scoring and complete likelihood rows. Serving controls
need `JITLLM_TEST_DATA=tests/unit/data`; run under installed GPU supervision.

Source base is `f5d946e`, with the six source/test identities and actual official
records in [results.json](results.json). Build math uses native NVCC 13.4.92
(toolkit 13.4.2), SDK `aarch64-c09daba6ac31edee`, and locked sources. The warm
checkout receipt reports an unknown Git version; the synchronized source bytes
are authenticated separately. Latest primary TensorFold at entry remains
`609ca419` (MLX recipe, no comparable CUDA run). The initial wrong-CMake refusal
and subsequent one-parenthesis compilation failure are retained; neither ran
model tests. Corrected `m35-gemma-head-capacity-check3` retired DONE0, all eight
steps, on `spark-56f5`. Raw official logs stay external. Full regression is
owner-deferred. This slice supplies no new corpus, reference-quality, optimized
batching or Qwen/DeepSeek capacity qualification.
