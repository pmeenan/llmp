<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3 final build and package checks

Focused M3 build and package qualification completed on 2026-10-04.
These checks cover specific build, source, tooling and license targets;
they do not establish that an entire `check`, `check:full` or `check:spark`
tier passed. The [frozen M3 record](../../m3-record.md) maps the separate
numerical, state, swap and client evidence.

## Results

CTest counts include registered tests that were skipped. A zero return
code means all executed tests passed; it does not turn a skip into a pass.

| Check | Host and scope | Result |
| --- | --- | --- |
| Original CPU qualification | x86-64 workstation, CUDA off | 1,138 registered, 22 model-data skips; sole failure `sources.closure`, return code 8. |
| Repaired CPU qualification | x86-64 workstation, CUDA off | 1,138 registered, the same 22 model-data skips; return code 0, CTest 32.37 s. Includes the source mechanism regression. |
| Native qualification before metadata repair | x86-64 workstation, CUDA compiled; ordinary `native` tests | 1,270 registered, 22 model-data skips; return code 0, CTest 42.70 s. This is not the workstation `native-gpu` inference suite. |
| Native source checks after repair | x86-64 workstation, CUDA compiled | All three `sources.*` checks passed, 5.93 s. The complete native suite was not repeated for the metadata-only change. |
| Final source-repair Spark suite | `spark-b`, native AArch64 and actual GPU tests | All 1,562 registered tests passed, CTest 98.20 s; supervised job completed and was waited on successfully. Source receipt and actual compile/link inventory pass with DS4 selected. |
| Combined tooling after both repairs | `spark` (`spark-c4e2`), `tools/tests` | 386 tests, return code 0, 46.911 s; one skip: `test_package.Provenance.test_every_notice_extracts`, because the Spark SDK has no sysroot. |
| Tracked license and boundary checks | `spark` (`spark-c4e2`), both repair overlays | 1,307 REUSE-covered files and embedded-header checks, 405 portability boundaries; no findings. Inventory contains 1,449 tracked paths. |
| Original ARM cross discovery | x86-64 workstation, clean `e540ddd` | All 652 build targets completed. CTest discovery failed for eight runtime-linked programs because the ARM SDK sysroot lacks cuBLAS's transitive `libgcc_s.so.1` dependency. No ARM CPU execution pass in this attempt. |
| SDK shared-runtime repair | x86-64 workstation, SDK repair overlay | Provisioning, deep doctor and 389 tooling tests passed. Four tooling skips: three unprepared EXL3 source checks and missing UCD data; all eight EXL3 checks passed separately against the authenticated prepared tree, leaving the UCD skip. Both static GCC runtime trees and cache inputs remain byte-identical. |
| SDK-repair Spark suite | `spark-b`, native AArch64 and actual GPU tests | All 1,562 registered tests passed, CTest 99.33 s. Doctor also passed; the unchanged native SDK tree remained byte-identical. |
| Repaired ARM cross discovery and CPU execution | x86-64 workstation, `bf64ba4` plus SDK overlay | All 681 build targets completed and discovery succeeded. Of 1,267 registered tests, 29 skipped and 1,237 executed tests passed; `JinjaBounds.HostileTemplatesAreRefusedOrCancelled` failed its 30 s bound with a 645.13 s self-list comparison under qemu. The test took 692.44 s; CTest 748.98 s. This attempt failed. |
| ARM CPU execution after comparison cancellation repair | x86-64 workstation, `5bd2938` plus exact reviewed Jinja overlay | All 1,243 executed tests passed, 1,272 registered and 29 skipped; return code 0, CTest 221.60 s. The complete hostile-template case passed in 42.29 s and all five new cancellation regressions passed. |
| Final Jinja-repair Spark suite and tracked lint | `spark-b` suite/format/tidy; `spark` tracked lint | All 1,567 registered tests passed, CTest 99.24 s. Changed C++ format/tidy passed; final 1,449-path inventory has 1,307 REUSE/header checks and 405 boundary checks, no findings. |
| Package inventory and network-denied install/reinstall/remove/purge | x86-64 workstation, ARM package under qemu-user, clean `e540ddd` | Passed, return code 0, 67.897 s. Package and install checks complete independently of the later hostile-template failure. |
| Final package inventory and network-denied install/reinstall/remove/purge | x86-64 workstation, ARM package under qemu-user, clean `7ed5c1e` and repaired SDK | Passed, return code 0, 68.841 s. Purge preserves the intended user and model storage while removing conversations. |

The Spark suite includes actual GPU/source proof. The source mechanism's
synthetic CUDA-on selection fixture uses a real C++ consumer without
pretending to execute CUDA; it complements the production GPU build.
Workstation heavy checks use the shared host lock. Spark builds, tooling
and lint use the installed GPU-lock supervisor and retained exit records.

## Repairs and coverage

The CPU failure was a provenance mismatch: the receipt selected DS4 even
though a CUDA-free build had no compile or link consumer for it.
`b834378` records DS4's
`requires_cuda: true` build condition. Preparation retains all six core
components; CPU configure selects five and CUDA configure selects six.
The full source-lock digest stays authenticated, and the strict inventory
check is unchanged. Tests exercise actual consumption, selected-but-unused
components, forged extra/missing receipt entries, excluded headers,
CUDA-on to CUDA-off output cleanup, invalid condition types and dependencies
on disabled components before component CMake runs. Kernel source, payload
pins and runtime arithmetic are unchanged.

The first tooling run had 382 tests, two errors and one skip. Both errors
came from the package license reader treating legitimate JSON license
sidecars as missing embedded headers.
`e540ddd` reads adjacent
sidecars for noncommentable files using the existing header classification;
code still requires embedded headers. Missing, unreadable or conflicting
declarations fail, and notice allowlists and in-tree provenance units remain
unchanged. The final 386-test run includes valid JSON sidecars, invalid or
unrecorded declarations, late conflicts and refusal to rescue code with a
sidecar. SPDX grammar and copyright checks remain REUSE/header duties.

The separate SDK repair `5bd2938` supplies the missing ARM shared GCC
support runtime through signed Ubuntu noble `libgcc-s1` and its exact
`gcc-14-base` copyright dependency. This is a runtime-only sysroot merge,
after GCC's existing static outputs; no compiler pin or setup code changed.
Tests cover locked component selection/order, preserved compiler cache
inputs, no-overwrite merging and the actual relative copyright link.
Authenticated package bytes and symbol closure cover both pinned cuBLAS
libraries' GCC-versioned dependencies. The repaired cross run gets past
discovery and exposes the independent hostile-template comparison bound.
The separate Jinja repair polls cancellation throughout nested comparisons,
string/BigInt work and dictionary lookup, and refuses to publish a completed
render after cancellation. Its exact reviewed overlay passed the complete
ARM CPU suite; the earlier failed run remains retained.
`7ed5c1e` records that reviewed repair after its Spark, ARM and lint checks.

## Identities and reproduction

The report worktree starts at `e540ddd`. Checks of the repairs ran on
pre-commit overlays; their receipts and source inventories are retained,
not relabeled as clean-commit executions. Workstation repaired receipts
identify `03dcec0` plus dirty changes. The copied Spark tree has no Git
version identity; its source inventory and reviewed patch supply identity.
The combined tooling overlay records source and package patch hashes and
the package reader/test SHA-256 values checked before and after the run.

The SDK identities are `x86_64-e0a0c85c42806fb1` and
`aarch64-e0a0c85c42806fb1`; Clang is 22.1.8 and NVCC is 13.4.92.
The SDK repair uses `x86_64-c09daba6ac31edee` and
`aarch64-c09daba6ac31edee` with the same compiler pins. Its Spark and
workstation builder checks ran on the retained `e540ddd` repair overlay;
the cross rerun records `bf64ba4` plus dirty SDK changes. They are not
relabeled as clean `5bd2938` executions.
The subsequent ARM CPU pass ran on `5bd2938` plus Jinja patch SHA-256
`3349d33f7b863115277f3c43234d0818ac5355b623cf6b7305b9eef6de8e6f52`,
using the repaired `c09daba6ac31edee` SDK. This is an overlay execution,
not a claim that the future commit was already tested by name.
The complete repaired source-lock SHA-256 is
`8383a58781d10669febed91ea6bb422a8ff5d4f278dd109cbcd3da13e7fb96f0`.
The combined tooling source inventory SHA-256 is
`920030dde85b604ab0e9708d5f2140abc94e94ae7abcc03d7a2035841682214a`.

The retained recipes use the pinned mise environment:

```sh
~/.local/bin/mise run prepare
~/.local/bin/mise run test -- cpu --locked --fresh
~/.local/bin/mise run test -- native --locked --fresh
~/.local/bin/mise run test -- native --locked --fresh -- -R '^sources\.'
~/.local/bin/mise run test -- spark-native --locked
~/.local/bin/mise exec -- python3 -B -m unittest discover -s tools/tests
```

Run workstation builds/tests under `hostlock shared`; run each Spark recipe
through installed `~/.local/bin/spark-job start --gpu --name N --timeout 600`,
then `~/.local/bin/spark-job wait N`. The source and package repairs were independently
reviewed and challenged before integration.

Raw evidence stays outside Git. The workstation bundle is
`~/scratch/llmp-m3-final-checks-2026-10-04/`: `source-checks.json`,
`cpu.log`, `cpu-fixed.log`, `native.log`, `native-source-fixed.log`,
`spark-full.log`, their retained build receipts, and `combined-lint.json`.
The builder bundle is
`~/scratch/llmp-m3-cpu-source-closure-2026-10-04/results/`:
`tools-combined.log`, `tools-combined-receipt.json`, `overlay.json` and
`combined-source-pins.json`. Remote supervised receipts and logs remain
in their owned scratch/job directories.

The earlier fixture produced `llmp_0.1.0~dev.335+ge540ddd83070-1_arm64.deb`,
577,285,132 bytes, SHA-256
`c3376f9e6454f4fa55039ce2525180af883fbb6f446655c23393a4454769194a`.
The actual Docker install-test image ID is
`71e452db8f15d86eb5c2675df07f6aeb69bd8211ce5d57b7597814a55913fea5`.
`package-run/receipt.json` preserves the clean commit, before/after source
pins, cross build receipt, package identity, image identity and successful
helper result. `cross.log` retains the separate failed discovery attempt.
`cross-runtime-fixed.log` and `cross-runtime-fixed-result.json` retain the
complete repaired-discovery run and its hostile-template failure.
`cross-comparison-fixed.log`, `cross-comparison-fixed-result.json`,
`cross-comparison-fixed-llmp-receipt.json`,
`cross-comparison-fixed-LastTest.log` and
`cross-comparison-fixed-LastTestsDisabled.log` retain the subsequent successful ARM CPU run,
its authenticated build receipt and skip list.
The final clean `7ed5c1e` package is
`llmp_0.1.0~dev.338+g7ed5c1e904d0-1_arm64.deb`, 577,284,354 bytes,
SHA-256 `99ece3950e74e9e9b6c24a99945a0fed0e2fc1924157d67857d03334e5e9ee34`.
Its actual install-test image ID is
`1febb907725e3f68081121fe5979617426b45638ac6754563b09f557a00ba9ab`.
`package-final-run/receipt.json` (SHA-256
`faa95d2f6a809acf6a3d6fc1b93df669daf4558bba8d1580a8ba1331e61ea5d6`),
log, extracted documents and cross receipt preserve that final run.
The rebuild replaces package outputs; the earlier package's receipt,
documents and log remain evidence, not a claim that its deb still exists.
The Jinja builder's `~/scratch/llmp-m3-jinja-cancel-2026-10-04/results/`
retains `full.log`, changed-C++ check receipts and `final-lint/result.json`.
The SDK builder bundle is `~/scratch/llmp-m3-sdk-libgcc-2026-10-04/`:
`authenticated-pins.json`, `runtime-symbol-closure.json`,
`gcc-inputs-equality.json`, `local-receipt.json`, `results/full-native-receipt.json`,
`results/full.log`, `results/prepared-source-supplement.json` and
`lint/result.json`. The first Spark doctor attempt failed because its
supervised environment lacked mise on PATH; the retained failed
receipt is not counted as a pass. The corrected doctor/full run and
tracked SDK-overlay license/header/boundary checks passed.
