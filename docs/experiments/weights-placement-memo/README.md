<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Successful immutable-weight placement checks

A scheduler placement stamp lets `PagedWeights` reuse a successful check
without rescanning every weight extent. In the current Gemma31 C4 recipe,
placement-check elapsed time falls from **23.50 to 2.92 ms** and the complete
paid interval falls by **25.615 ms (0.77%)**. All 128 complete heads, four
initialized states and 128 choices remain byte-exact across old/new and repeats.

| Arm, in acquisition order | Paid interval (s) | Checks (ms) | Execution (s) |
| --- | ---: | ---: | ---: |
| Before first | 3.31166 | 23.4564 | 3.23416 |
| Candidate first | 3.29267 | 2.78283 | 3.23455 |
| Candidate repeat | 3.28794 | 3.05061 | 3.23046 |
| Before repeat | 3.32018 | 23.5371 | 3.24376 |

This is a two-binary host-only factor on Spark A, using the unchanged
[phase-accounting C4 helper](../gemma-owner-c4-phases/README.md). Both binaries
use the same SDK, locked math sources, resolved cuBLAS libraries, artifact and
fixed IDs. Mean time outside runner execution decreases from 76.96 to 57.80 ms.
These are enclosing elapsed intervals; stream spans include submission gaps
and are not active-kernel time. The experiment supplies no fresh-reference,
sustained-speed, other-family-speed or broader quality qualification.

The scheduler's construction-only lifetime token prevents a false hit when a
new scheduler reuses an old address. Successful `SetSource` replacements and
actual pin/unpin changes advance its epoch. Exhausted tokens/epochs permanently
refuse caching. `PagedWeights` resets its memo before Open, Reserve, Register
and Release, including refused or partial operations, and caches only its own
successful scan. Existing errors in the caller's aggregate remain untouched.
The memo holds a fixed optional two-word stamp; it retains no extent list.

`node.Call` still runs on every wave. Mutable `LiveState` checks remain fresh:
its expected spill metadata can change before a failed scheduler registration.
No arithmetic, graph, fusion policy, resource placement or paging work changes.
The shared weight optimization applies to runner callers; measured speed here
is limited to the explicit Gemma31 C4 owner-root/plain-norm-on recipe.

Reproduce with the locked Spark SDK on base `017c25e`, preserving an unchanged
before helper and SDK receipt, then build `jitllm_gemma_owner_c4` with the six
candidate paths recorded in [results.json](results.json). Use the same
[prior C4 inputs and baseline](../gemma-owner-root-c4/README.md), with
`JITLLM_GEMMA_OWNER_C4=owners`, `JITLLM_GEMMA_C4_NORMMUL=1` and
`JITLLM_GEMMA_C4_PHASES=1`. The helper arguments are
`ARTIFACT NEW_OUTPUT_DIR 31 4 joined norm IDS`; the closed recipe uses F16
context 256, max rows 128, owner prefixes 64–67, eight warm waves, three reset
waves and 32 paid waves. Every full head, argmax and archive copy is paid;
finite scans and complete state hashes follow the timer.

The source receipt, binary/SDK identities, outputs, fixed input and official
records are recorded in the aggregate. Native NVCC is 13.4.92 (toolkit 13.4.2).
TensorFold's primary `609ca419` recipe remains MLX without a comparable CUDA
Gemma run. Raw logs, outputs and job records stay external.

The eight focused host descriptor/release controls pass. Nine existing
lifetime/paging/spill/restore/clear controls also pass, including both pinned-
address swap parameters. All jobs officially retired; no focused control skipped.
The first build failed on two test-fixture constructor calls; their correction
changed no production bytes. The full regression suite is deferred by owner
instruction while the performance gap is being closed.
