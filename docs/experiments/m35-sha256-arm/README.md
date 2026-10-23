<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ARM SHA-256 acceleration — bounded prerequisite screen

On Spark A, the shared SHA-256 implementation passed its fixed synthetic
screen with unchanged digests and about 4.28 times the scalar throughput.
This supports accelerating verification before attempting large native imports;
it does not measure import, disk or model performance. The machine-readable
[aggregate](results.json) retains actual source/compiler/binary/dependency pins.

## Implementation and correctness

The public hash API, buffering, padding, length bookkeeping, copy behavior and
finished-state checks are unchanged. A private complete-block compressor selects
ARM SHA2 through the pinned compiler's runtime CPU feature query, or the original
scalar arithmetic. Only the private ARM function has an explicit SHA2 target;
its translation unit retains baseline `armv8-a`. Other builds use the scalar
fallback. There are no new dependencies, global architecture flags, verification
shortcuts or caller-visible dispatch controls.

The actual pinned Clang 22.1.8 feature probe compiled and ran on Spark A with
`sha2=1`; disassembly confirms the SHA instructions in the accelerated object and
none in the generic hash object. Native builds of `base_test`, `artifact_test`,
`artifact_v1_test`, `import_test` and `import_writer_test` passed. Fresh CTest
selection and retained inner GoogleTest output authenticate exactly 141 cases,
all passing with no skips. Tests compare complete block states across 32 input
alignments, empty/multiple-block calls, padding boundaries, every fragmentation
split and copied incremental states; existing standard vectors and complete
artifact/source verification controls also pass.

The explicit x86-64-v3 portable mini-CMake build passed all 12 actual base tests
under the workstation shared lock, exercising the scalar fallback. It uses the
actual base CMake target and actual `base_test`, with the pinned compiler and
GoogleTest; it is not a full CPU-preset claim. The first fixture configure failed
because its compiler probe lacked static runtime link flags; that failed log is
retained. Supplying the existing toolchain flags fixed the scratch fixture, with
no product-code change.

## Fixed measurement

Before the candidate run, the criterion was fixed: both digests must match and
each elapsed time must be lower than its frozen exclusive scalar baseline.
One candidate run reused the exact driver source: stream the same deterministic
1 MiB buffer 512 times, then 1,024 times through the public incremental API.
Buffer byte `i` is `(29 * i + 7) % 256`, repeated without resetting the hash.
Timing includes the final digest. Baseline and candidate run under exclusive
supervision, without overlapping builds. An earlier shared scalar observation
is retained as preliminary only and is not used for the comparison.

| Stream | Scalar seconds | ARM SHA2 seconds | Candidate / scalar |
|---|---:|---:|---:|
| 512 MiB | 1.507837609 | 0.352398934 | 0.233711 |
| 1 GiB | 2.996729239 | 0.699465563 | 0.233410 |

Both results pass. Expected SHA-256 values were independently checked with
Python hashlib before enrollment and match in both runs. The fixed baseline
receipt is `c01829b8695f460e376b1a3f83645c79de483fce654ad871008aea48271f3991`.
The checked implementation sits on source-fold R6 as an explicit prerequisite;
the SHA patch contains only its own five source/test paths. A later comment
spacing correction and these documents do not relabel the checked binaries.

## Limits and handoff

No artifact is re-imported or promoted by this change. All existing source,
manifest, chunk and deep-read verification passes remain. Large-publication
feasibility must be checked with the actual rebuilt writer; synthetic hash
throughput does not establish an end-to-end duration. CPU fallback correctness
and native focused checks supplement, rather than replace, root integration
checks. Existing kernel performance evidence is not re-pinned: any compiler
identity change makes its historical evidence stale under normal validation.

Raw logs, supervisor outcomes, source guards, XML, disassembly and the scratch
driver remain outside Git under `/tmp/llmp-sha256-arm-evidence` and
`/tmp/llmp-source-fold-hash-feasibility` through milestone closeout.
