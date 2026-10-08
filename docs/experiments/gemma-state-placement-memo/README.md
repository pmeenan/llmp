<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Successful live-state placement checks

`LiveState` now reuses its successful source/pin validation under the scheduler's
lifetime and placement stamp. The focused controls prove the successful-hit
branch and fresh checks after mutations. In the short Gemma26 C8 comparison,
paid elapsed decreases **29.195 ms (1.7661%)**; the two baseline/candidate
spreads are 4.21/3.64 ms. Gemma31's mean decreases 8.660 ms, within its
22.01/15.91 ms spreads: its latency gain remains unestablished.

| Acquisition arm | Gemma31 paid (s) | Gemma26 paid (s) |
| --- | ---: | ---: |
| Baseline first | 4.43866 | 1.65518 |
| Candidate first | 4.41104 | 1.62570 |
| Candidate repeat | 4.42695 | 1.62206 |
| Baseline repeat | 4.41665 | 1.65097 |
| Baseline mean | 4.427655 | 1.653075 |
| Candidate mean | 4.418995 | 1.623880 |

All four arms per profile retain byte-exact complete 264-row heads, sixteen
initialized state files, layouts, supplied IDs and 256 choices; every helper
scans its complete retained heads for finiteness. The states total
14,763,950,080 bytes for 31B and 3,690,987,520 bytes for 26B. This is a
same-host two-binary host-work factor on Spark A (`spark-c4e2`), not a fresh
reference comparison. It supplies no phase attribution, sustained-speed claim
or explanation of the separately measured Spark B reference gap.

The existing [weight memo](../weights-placement-memo/README.md) supplies the
construction-only unique scheduler token and source/pin epoch. Saturated tokens
or epochs remain uncacheable. The state memo retains two inline words, caches
only its own successful scan and preserves prior aggregate errors. Failed scans
are always repeated. It resets before registration, growing-state changes,
source/restore changes, trim, clear and release, including refused operations.
`node.Call` still executes every wave. Residency, leases, contents, quarantine,
completion and coverage remain independently checked; eviction without moving a
source does not invalidate a successful source/pin check.

Current Gemma, Qwen3.8 and DeepSeek state consumers use this shared path. Their
state objects are constructed before startup `MemAvailable` sampling, so the
additional sixteen inline bytes are already included in observed occupancy.
The manual joined helper's maximum twelve state objects add 192 bytes within
its existing 64 KiB metadata allowance. The uncounted memory margin supplies no
funding justification. A future post-probe consumer must fund its enclosing
objects. Qwen/DeepSeek speed benefits have not been measured here.

Reproduce with the unchanged [production joined helper](../gemma-joined-serving/README.md)
`llmp_gemma_joined`, preserving the baseline built on `267c5fe` plus the
reviewed opt-in core and its receipt, then building the four candidate files on
`bc8fca6` with the same locked SDK. [results.json](results.json) records the exact
source, binary, SDK, input and output identities. Both private binaries use the
same `e57a1624` SDK receipt and unchanged arithmetic/helper sources; the remote
Git version is not used to authenticate workstation source provenance.

Run `ARTIFACT NEW_OUTPUT_DIR 31 8 joined norm COHORT12_I32 production`, or
`26 8 joined all` with the approved 26B artifact. Each checkpoint's qualified
carrier contains twelve distinct 1,024-token histories; these arms use its first
eight. Context is 4,096; input caps are 256/1,024, with head cap eight. Each arm
prefills 992 rows, runs eight warm joined waves, clears and repeats prefill, then
pays for 32 waves including full head retention and argmax. Finite scans and
frontier/final initialized-state copies follow the timer. All paid arms record
32 graph replays and no new captures. Compare every complete output and all
choices across baseline/candidate/repeats before interpreting elapsed time.

Nineteen focused controls pass without skips: six descriptor-only state checks,
six existing weight checks and seven GPU state controls for source pinning,
growth, eviction, clear, trim, restore and cancellation. Five official jobs
retired successfully. Two initial fixture failures are retained separately:
a missing `SpillPlace.directory` initializer stopped compilation, then the new
BareState fixture lacked the source-place pinning real runners perform. Its
strict placement checks correctly refused those unpinned extents. The final
fixture explicitly pins source places, preserves all state and memo-hit assertions and
unpins at completion. Neither correction changes production behavior.

TensorFold was refreshed at this task's entry: primary HEAD remains `609ca419`,
version 0.6.5; its Gemma26 recipe is MLX and it supplies no comparable CUDA
Gemma31 recipe. No TensorFold inference ran. Raw outputs and supervisor records
remain external. Existing quality/default/cohort gates are unchanged; the full
regression suite remains owner-deferred.
