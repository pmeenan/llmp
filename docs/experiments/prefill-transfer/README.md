<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared funded prefill lookahead

## Gemma3 and Gemma4 lifecycle extraction (2026-10-07)

`engine/prefill_lookahead.h` centralizes the optional host allowance and
CPU-built future plan previously managed separately by Gemma3 and Gemma4.
Their prediction, output and capture policies remain in the adapters. This
slice makes no performance claim and changes no production defaults.

The helper takes an optional allowance before future plan allocation.
It builds only CPU descriptors during the current device job; implementation
binding, coverage checks and cache insertion remain after the driver's existing
successful-completion checks. A binding or coverage refusal destroys the future
plan while its temporary allowance remains charged. Successful insertion gives
back that allowance immediately before the cache's normal required charge.
Abandoning the prediction destroys its plan before returning the allowance.
Existing `PlanStep` scopes protect the current plan and capture throughout.
No future state backing or cursor is prepared by the helper.

Five host-only tests exercise optional refusal, failed and oversized builders,
abandoned predictions and failed current units, installation failure, and
single transfer to the normal cache while the current plan/capture remain
protected. Model controls compare exact state and heads for unhinted, hinted,
wrongly hinted and warm Gemma3 runs; they also check Gemma31 pressure refusal
and abandoned predictions, and Gemma26/Gemma31 state-only continuation/replay.

On Spark B, installed GPU-exclusive job `prefill-shared-check1` incrementally
built the three affected test targets against source base `0ea0a18` plus this
slice, using the pinned `aarch64-c09daba6ac31edee` SDK and CUDA 13.4 cuBLAS
130800. All **17 focused tests pass**, with positive XML counts and no skips:
13 host-only plan/cache tests (including the five new lifecycle tests), one
Gemma3 hinted-prefill test, and three Gemma26/Gemma31 controls listed above.
The job's final state is `done`, exit 0, after 95.04 seconds including the build.

The Gemma3 artifact was copied to B in supervised job `prefill-shared-fixture1`.
Its manifest hashes to approved identity `8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb`;
every declared payload's length and SHA-256 were independently checked after
the copy. Gemma26/Gemma31 used their existing approved fixture artifacts.
A separate source review and adversarial lifecycle challenge found no defects.
No new timing comparison was needed for this exact policy-neutral extraction;
exact state, heads, capture counters and memory accounting are the controls.
The full regression suite remains deferred under the owner's optimization
check override. Further transfers, including Gemma2 lookahead/capture and joined
prefill hints, remain separate work; this extraction does not close those ports.
