<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Owned joined Qwen target/MTP recovery screen

2026-10-12, Spark B, main candidate based on `28d99ebd`. This is actual
shared-engine execution through ComponentStepSource, ModelRunner, WaveComposer
and DecodingOrchestrator. Ordinary production serving remains unopted.

The prepared target is `c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`;
its MTP companion is `8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`.
The installed selected head has 47,172 rows, below the requested 65,536-row
ceiling. The immutable I32 map carries every selected row into the target
vocabulary; its SHA-256 is
`cfbfa2466c5c2b0246cdf62ca0ca2b6e7b94f72091fe650f62004909b1ac14b3`.
Ordinary truncated heads use their actual width with identity token mapping.

The physical v0 PLE table is U8 data plus a separate F32 tensor scale. The
source gate validates their complete inventory with the existing artifact
assembler against the logical NVFP4 row format. It retains exact v1 checks,
part/geometry/extent checks and the table's nonresident group requirement.
No forced intermediate rounding or new arithmetic policy is introduced.

## Observed controls

The full native build and all 306 selected host controls passed without skips
in `qwen-joined-mtp-main07`. The actual model registration
`Qwen38JoinedMtpGpu.OwnedUnequalPendingRestoresAndChangedInputCapture` passed
in `qwen-joined-mtp-gpu03`:

- Two owners with unequal pending ranges; 13 completed units and 132 finite
  full-head rows. Thirty reference rows and 60 restored rows are checked.
- Four checkpoint-restored owners, two spilled/restored owners, complete
  initialized-state checks and 30 changed-input head rows.
- Three captures and 33 replays. Changed inputs reuse the recorded program
  and produce different heads without another eager run.
- Thirty-five independently measured classes, complete physical admission
  91,628,901,123 bytes below the unchanged 104 GiB node cap, 12 GiB prepaid
  host allowance and 341,835,776 bytes per paid state-copy capacity.

The case took 20.499 seconds; this includes setup and storage and is not a
serving-performance measurement. Source, binary, artifact-metadata and recipe
guards passed before and after. The child process group retired, no GPU process
remained, and all memory limit/OOM event counters stayed zero. The cgroup's
2,156,793,856-byte process/page-cache peak is not total device residency.

Native recipe SHA-256: `d2ffd6efc15d7eaa589c608802f0606167898504c1e18ef997558f41ca4bd0dc`.
Actual recipe manifest SHA-256: `8158f99d116ad7a666179a6aef57ea782dd084f0e3b01bbcd3bd87f3b06b1434`.
Raw logs/XML remain outside Git under each host's session scratch.

This establishes same-implementation state restoration and owned-domain
execution. Broader quality and behavioral coverage against a reference of the
same quantized checkpoint, durable old/new restart compatibility, production
serving and matched family speed/swap remain open. The original Qwen
source-precision model remains absent from inspected local and NAS catalog
evidence, limiting original-agreement claims without blocking that quantized
baseline. Cross-library bit equality is diagnostic, not a quality requirement.

## Isolated commit check

The Qwen-only candidate excludes the in-flight DeepSeek bridge and reader-budget
changes. Its full native build and all 305 affected host controls passed in
`qwen-joined-mtp-main09`; the unchanged actual registration passed again in
`qwen-joined-mtp-gpu04`, with complete guards, retirement and zero memory-limit
or OOM events. The single removed host registration belongs to the unfinished
DeepSeek fixture. A final prose correction updates the stale pending status;
production and test code are identical to this checked candidate.

Isolated native recipe SHA-256: `94a970acb410e7e22b14d14eed280d7c0bdadb247708a861163bd5f8365b021f`.
Isolated actual recipe manifest SHA-256: `d6dca445587b2752f7d8d4d6be77489e7cc06cafe1457e0d0511a566b665b48e`.
