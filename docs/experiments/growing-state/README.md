<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Conversation state grows with use (2026-09-29)

DeepSeek and Qwen3.8 reserve stable virtual addresses at their configured
ceilings, but materialize only the 2 MiB extents their actual state accesses
need. Unused extents consume neither device backing nor spill traffic.
The existing graph address pins remain valid. Sparse unnamed spill files
preserve initialized extents; clearing or trimming a discarded tail also
removes its saved bytes, so it cannot return as stale state.

Footprints include attention's padded cache reads, DeepSeek's final dummy
cells and fixed window/compressor rings, and Qwen's recurrent, convolution
and cached indexer-block state. A clean capacity refusal preserves the
completed prefix. A job whose effects are uncertain quarantines the state.
Diagnostic snapshots pack used extents and catalog their pinned staging;
the shared snapshot records its owner, history, cursor and adaptive depth.
These snapshots are controls, not the turn-reuse cache, which follows next.

## Native checks

On `spark`, GB10, driver 580.178.04, SDK `aarch64-e0a0c85c42806fb1`,
CUDA 13.4.92, growing-state changes over `5b451b3`. Both models have a
262,144-token ceiling, speculation enabled, and 8,192 used prompt tokens;
DeepSeek uses 2,048-row chunks and Qwen 4,096. The runtime's swap table
compares 32-token continuations and their logits against each saved state's
uninterrupted control, and hashes the packed state before and after return.

The initial run (19:19:04–19:21:11 EDT) passed every state/output comparison
and reported no problems. First-use swaps, including B's first output:

| Direction | Saved model's used state spilled | Swap seconds |
| --- | ---: | ---: |
| DeepSeek → Qwen | 286 MiB | 8.026 |
| Qwen → DeepSeek, restoring DeepSeek | 0 | 9.623 |
| Qwen → DeepSeek | 640 MiB | 9.788 |
| DeepSeek → Qwen, restoring Qwen | 0 | 7.849 |

Four prepared zero-context controls also passed (7.910–9.663 s).
Registration's fixed occupancy was 3.93 GiB, including a 3.21 GiB shared
workspace; available memory after setup was 112.53 GiB. Peak memory drop
from runtime startup was 110.0 GiB, with 7.11 GiB available at its low point.
Spill counts are physical extents; packed snapshots clamp each region's
last extent to its logical length. The virtual ceilings are not memory
allocations and do not imply that both models can execute concurrently.

The final prepared-context run (19:25:07–19:28:18 EDT) passed all 12 swap
rows, with no reported problems. At 8K, prepared returns took 9.506 s for
DeepSeek and 7.791 s for Qwen; they replayed 32 and 24 steps with four and
nine graphs retained, respectively, with no eager launches or captures.
State, tokens and logits remained exact. Peak startup-relative
MemAvailable drop was 110.57 GiB; the low point retained 6.47 GiB available.

A standalone DeepSeek draft at an empty prefix returns three valid token
IDs. The embedding check agrees with host lookup for all 129,280 rows;
the 32-token capital prompt matches plain decoding at every teacher-forced
and free-running position. This regression ran 19:22:14–19:23:19 EDT.
An earlier Qwen 8K rejection/swap control also preserved exact state,
logits and tokens, replaying eight retained graphs.

## Verification and limits

The Spark check set passed all 1,008 tests, including 187 GPU tests.
Eight new GPU live-state tests exercise growth, sparse spill/restore,
resident and spilled clearing, discarded-tail invalidation, packed copies,
dynamic staging retirement, low-budget prefix preservation and uncertain
copy ownership. Model tests cover padded reads and dummy cells; a catalog
test covers generation invalidation after dropping preserved contents.
Changed units passed the Spark SDK's format/tidy checks, the 289-unit
boundary check and the 975-file REUSE/header check. Final diagnostic-budget
fixes were rebuilt, linted and exercised by the native DeepSeek check.

Workstation checks are deferred by the owner until the entire optimization
run is settled. Maximum-context throughput, turn-boundary reuse and the
full M3 gate are separate remaining measurements.

Provenance: runtime prompt source `ppl.txt`, SHA-256
`c7156148ecaa12b6416cf816540d8dede2014982554a835e61076f0dd8bf0c2d`;
DeepSeek target `8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234`,
DSpark `dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5`;
Qwen target `c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`,
prefix drafter `056a750e3a90be3ae6a4b12bb963ce45290aaa6f52b5ba9799e777d491f80aea`.
Raw reports remain outside Git in `~/scratch/m3-growing/` on the Spark.
