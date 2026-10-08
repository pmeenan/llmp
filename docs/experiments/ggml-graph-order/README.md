<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded graph-order membership

Native graph construction replaces repeated linear membership scans with an
arena-owned pointer table. It preserves source-array DFS order, insertion
before descent, duplicate/cyclic-edge handling and PARAM leaves. Standalone
callers retain the old overload. All nine Qwen2, Qwen3.8, DeepSeek, DSpark and
Gemma target/draft factories propagate a traversal-capacity refusal before
planning or execution. Gemma's state-only final KV cut stays unchanged.

The table bounds every reached descriptor, including omitted NONE leaves and
external weight descriptors. Capacity derives from the greater of actual
metadata and the sizing estimate; checked power-of-two storage keeps at most
half the slots occupied. Overflow, null outputs and foreign descriptors beyond
the bound refuse cleanly. Arena `bytes()` charges metadata plus traversal
storage, while `used()` and `Reserve()` remain metadata-only. Startup scratch
and retained plan envelopes include the table. Seal keeps its charge; reset
clears it; moves transfer ownership and zero the moved-from accounting.

| 8K state-only recipe | Before, s | Candidate repeats, s | Change versus before |
| --- | ---: | ---: | ---: |
| Gemma 26 all1024, normmul off | 2.66480 | 2.60794 / 2.61897 | −1.93% |
| Gemma 31 both256, normmul off | 11.7776 | 11.6492 / 11.6724 | −0.99% |

Both candidates preserve the before control's complete prefill/final heads,
592,445,440/1,740,636,160 initialized state bytes for 26B/31B and all
32 choices. Ten retained heads per profile are finite. Gemma 26 fresh reference
prefill is 2.41728/2.41409 s, leaving an 8.19% candidate latency gap and a 0.40%
decode gap. Fresh ring-reference prefill is 10.8453/10.8420 s, leaving a 7.54%
candidate latency gap; decode remains 2.23% above the reference mean. This is a
small end-to-end improvement, without a measured planning-only attribution.
The helper did not enable phase counters, and no extra inference was run to
obtain them.

Each arm uses the same 8,227 canonical IDs, six discarded warm rows, Clear,
8K prefill, three anchors and 32 forced units. State-only intermediate chunks
publish no unused heads. Final prefill/forced units pay complete publication;
forced units pay lower-index incoming-head argmax. Native masks, arithmetic
policies, stable graph capture and cache capacities are unchanged. Reference
SWA/unified remain false, F16 KV and context 16,384; physical ubatch is 256 for
31B and 1,024 for 26B. The selected final-plan implementation counters remain
those of the prior policy, rather than CUDA launch counts.

The narrow Spark A build passes 145 controls in 12 binaries, including the new
DFS/cycle/leaf-bound tests, arena funding/move/reset/overflow tests and existing
Qwen/DeepSeek/Gemma target/draft graph and placed-plan controls. Exact source
order controls cover primitive and selected plans. No prior-family whole-model
speed result is claimed. The owner stopped the integrated Spark suite after 1,679/1,714 targets had
completed: 1,678 passes and one old arena-accounting test failure. Its official step ended by SIGTERM;
the suite is incomplete and is not a pass. The test is corrected to check
trimmed metadata plus the separately retained traversal table; production
source and the measured binary remain unchanged. The corrected PlanCache
binary passes all eight focused controls without skips. Full regression is deferred until
the performance changes settle. The focused 145 controls and exact model
screens remain the completed checks.

Reproduce with the existing `llmp_gemma_prefill ARTIFACT IDS NEW_OUT PROFILE
POLICY ROWS normmul-off state-only`, selecting 31/both/256 or 26/all/1024. Use
original → preserved before binary → candidate first/repeat → original, with
the existing owned-container ring launcher from the preceding
[state-only screen](../gemma-state-only-prefill/README.md), preserving its
pinned original client/image and input recipe. [Results](results.json) retain the
source, binary, input and aggregate identities. Raw vectors, snapshots, token
telemetry and official logs remain external. Source base is 27bbc24; native
NVCC 13.4.92/toolkit 13.4.2 and the unchanged original CUDA 13.3 image are pinned.
Task-entry TensorFold 609ca419/0.6.5 supplies a Gemma 26 MLX recipe, without a
matching CUDA graph-order comparison. Model quality, optimized batching and
competitive parity remain open.
