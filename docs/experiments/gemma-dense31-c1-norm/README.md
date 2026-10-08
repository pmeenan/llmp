<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dense31 scalar checked-norm first screen

The checked normRoPE + normADD candidate matches both fresh pinned original
C1/u128 runs byte for byte across all 32 complete 262,144-value heads.
Its paid block is 1.920155729% slower than the reference bookend mean.
This establishes math agreement for this supplied short prefix and policy;
it selects no production default and qualifies no longer context or batching.

| Arm | Complete byte-exact rows vs reference | Strict greedy differences | Maximum raw delta |
| --- | ---: | ---: | ---: |
| Ordinary control | 0/32 | 9/32 | 22.627453566 |
| Checked norms, ordinary products | 32/32 | 0/32 | 0 |

Both candidate own runs, its paid bookend, and the two fresh original runs
have whole-file SHA-256
`3b0df1867a30e6135b586933660b4c67c11e714285d37b8ff387c7c3666633b5`.
All heads are finite. Candidate full-byte own movement was zero before stock
exposure; its strict, positive reference-margin and exact-reference-tie counts
are all zero. The ordinary control was acquired once: its own noise is
unmeasured, so the candidate's zero bound does not calibrate that control.
Selected full-distribution TV and target NLL differences are zero for the
candidate at steps 0/7/15/30 (target positions 68/75/83/98).
The [aggregate](results.json) records those likelihoods and the control metrics;
there is no corpus PPL claim.

| Paid arm, in acquisition order | Seconds | Published heads / completed units |
| --- | ---: | ---: |
| Original C1/u128 | 3.00427 | 32/32 |
| Native checked norms | 3.06293 | 32/32 |
| Original C1/u128 | 3.00618 | 32/32 |

Reference mean is 3.005225 s; bookend span is 0.063556% of that mean.
Each arm pays for 32 complete head publications and lowest-index argmax scans.
Load, prefill, eight warm units, reset and three untimed anchors, finite
validation and disk output are outside the timer. The native paid block
records zero new captures and 32 replays. This short screen is not a
competitive performance pass.

## Recipe and boundary

The dedicated [helper](../../../benchmarks/gemma_dense31_c1_norm.cc) fixes
scalar C1/dense31, context256/max_rows128, the canonical supplied IDs,
64-row prefill and one-row decode. Checked norms apply to both prefill and
decode, so this is a compound policy comparison. Fresh selected plans show
120 normRoPE and 120 normADD choices in the candidate and zero in the
ordinary control. Row-invariant products, generic norm fusion, RoPE/store,
sharedQ8 and MoE choices remain zero. These are selected-plan counters,
not observed CUDA kernel counts. The historical joined helpers and all
production sources remain unchanged.

The unchanged original client uses pinned llama.cpp b29c606e, original image
and math libraries, F16 KV, context256 per sequence, batch128/ubatch128,
one sequence, fusion and graphs enabled. Its C-API defaults are
`swa_full=true` and `kv_unified=false`; the native local state uses its
bounded ring. This short prefix does not wrap the local window. The result
does not establish long-context ring/full-cache equivalence, representative
memory ratios, or speed parity for a production CLI/server cache recipe.

The node funds 35,651,584 bytes of retained/working host heads, in addition
to catalog state/output and plan/scratch capacity. The helper uses the existing
held request and stable heap lifetime and retains that complete owner on
unproven teardown. Funded capacity is not measured physical peak.
No state continuation, spill, cancellation, full-profile quality or serving
qualification is claimed. Gemma26 and Qwen/DeepSeek transfer requires their
own format, shape and representative controls; this dense31 result supplies
no automatic adoption for another model.

## Provenance and reproduction

Measured source base is `0a1a0a6`; eight exact new/historical source paths,
headers, four original libraries, canonical IDs, manifest/index, locked build
receipt and both binaries were frozen at 13:24:20 UTC. Native candidate own
freeze `1d89d3a2…` completed at 13:27:46 UTC, before the fresh original
bookend began at 13:30:43 UTC. All source/environment identities were checked
again after paid work, including libllama. Full production-source checksum
sync used `rsync -rlpc` without timestamp preservation; the pre-bookend dry
run was empty. [Provenance](provenance.json) contains the exact identities and
official successful build, native, freeze, bookend and analysis retirements.
Raw heads, per-step IDs, logs and receipts remain external.

Measurements ran on Spark-b (`spark-56f5`), NVIDIA GB10, driver580.178.04,
queried before the fresh reference arm. Native SDK is
`aarch64-c09daba6ac31edee`, with pinned CUDA13.4.92; the immutable original
image's previously authenticated CUDA environment is13.3.0/CUDART13.3.29-1.
These are separate compiler/runtime provenances. An optional version-file
probe failed because the container had no `/usr/local/cuda/version.json`;
it acquired no model data and its failed record is retained.

Task-entry primary [TensorFold](https://github.com/ashhart/TensorFold) refresh
at 2026-10-05T13:11:15Z resolved609ca419/version0.6.5. Its freshly fetched
Gemma recipe is26B-A4B MLX-only, with no dense31 CUDA comparator.

Follow [PROTOCOL](PROTOCOL.md) with the approved artifact/raw source and
canonical1,024 little-endian IDs identified there. Supply the eight pinned
headers in external `headers/`, use the wrapper's external scratch paths,
and checksum-sync sources to the owned Spark tree. Build only the manual
`llmp_gemma_dense31_c1_norm` target through the locked SDK, then run
`bash reference.sh build` before any source/output freeze. Every command
runs through the installed supervised GPU admission.

For the freezer, create external `source-ancestry.json` with `base` and
`head` equal to the measured base, `tracked_production_diff: []`, the exact
six dirty benchmark/experiment paths, and `source_files` mapping all eight
`SOURCES` paths to their byte lengths and SHA-256. Preserve the real locked
build receipt; do not manufacture one. Run `own_freeze.py source ROOT SCRATCH
BUILD_JOB` only after official successful build retirement. Acquire fresh
`ordinary-control`, `candidate-first` and `candidate-repeat` directories,
then `own_freeze.py own ROOT SCRATCH NATIVE_JOB SOURCE_FREEZE_SHA` after
successful native retirement. The receipts are exclusive-create, so use new
scratch directories for a new acquisition and preserve old evidence.

Only after authenticating that own freeze, run the fresh reference / native
candidate / reference bookend into `reference-first`, `candidate-bookend`
and `reference-repeat`. Reauthenticate the environment and all four pinned
libraries. `compare.py SCRATCH OWN_FREEZE_SHA` validates actual input and
whole-file own/reference identities before comparing complete heads;
`compare.py --self-test` checks signed-zero, reference-tie and nonfinite
metric handling. Source counters, finite/full-output controls and official
retirement passed; diagnostic-only workflow scope omits a routine unit suite.
