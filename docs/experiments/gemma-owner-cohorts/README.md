<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded Gemma owner cohorts

The opt-in engine supports complete attention quads within one through eight
actual owners. Eight-row products stay shared; each attention call reads four
real independent K/V roots. Complete C8 waves with matching padded widths derive
the original whole-eight stream-K grid before rounding and divide an even grid
between the two calls. Odd grids preserve the four-owner path. C4, incomplete
quads and unequal widths preserve their existing geometry or independent fallback.
Every eligible writer is authenticated before graph mutation, and both planning
passes fund the additional descriptors and selected scratch.

The [Gemma31 C8 factor](../gemma-c8-cohort-geometry/README.md) resolves the paid
correctness gap: all 256 completed decode heads match the retained reference
byte for byte. The [Gemma26 transfer](../gemma26-c8-cohort-geometry/README.md)
passes its unchanged prior bound and conditional likelihood gate. Those bounded
model measurements use the separately identified source22/helper4514, including
pending runtime settings changes. This commit lands the opt-in engine portion
and its harnesses. Runtime settings, serving options and supported status remain
unchanged; production solo and optimized-batching qualification remain open.

## Full backing, bounded reads

A configured 262,144-token cache can serve a short owner read. The address checker
keeps every executed operand within 64 MiB and actual read widths within 16,384
cells, while checking backing parents up to 1 GiB. It verifies each parent span,
alignment, pointer addition, view offset and containment and rejects cycles.
The larger parent bound changes address validation only; shader indexing, grid
arithmetic, mapped-state acquisition and completion/graph lifetimes do not change.
Unsupported reads fall back before transformation. Packed attention retains its
16K configured-context and 64 MiB parent limits.

The pure graph controls bind symbolic full-capacity parents for both approved
profiles, C4/C8 and actual widths 1,024/16,384; every constructed owner node passes
its operand checker. A 16,640-cell read stays independent. Malformed oversized,
overflowed, mispositioned and cyclic parents are refused. These controls establish
eligibility and validation, without claiming a full-context model execution or
competitive performance at depth.

## Verification

The final opt-in16 source builds on Spark A and passes 15 focused controls with
zero skips: 11 graph, three placement/planning and one GPU operator case. The
operator compares original whole-eight MMA with two four-root calls for D256
and D512 at 256/1,024 cells, including eager execution and capture/replay; all
outputs are finite and byte-exact. The earlier six focused parent checks also
pass. An initial launcher omitted the fixture-directory environment and is
retained as a failed official attempt; the corrected retry uses identical source.
No full regression suite was run under the owner's optimization override.

[Results](results.json) record the exact core source, excluded pending defaults,
SDK and built binaries, bounds and official completion identities. These final
parent-validation binaries are distinct from the source22 C8 model-measurement
binaries. Raw logs and model vectors remain external.
