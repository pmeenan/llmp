<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Test-suite audit (M3.5 F14, 2026-10-28)

Slice F14 of the M3.5 review fixes audited the suite of more than 4,000
tests. The goal (owner, 2026-10-26) was to remove what is no longer
relevant and merge real duplication without reducing coverage. When in
doubt, a test was kept. F12 had already handled the environment-gated
skips ([m35-f12-frozen](../m35-f12-frozen/README.md#env-gated-tests)).

All runs were on `spark` (GB10, Ubuntu 24.04.5, driver 580.178.04), with
SDK `aarch64-8486b6bf58a61039` (Clang 22.1.8), at main 1564dcd6 before
the change and the F14 tree after it. Raw profiles stayed outside Git and
were deleted; this page holds the aggregates.

## Result

- **Coverage:** no `src/` file lost coverage because of the change. Every
  file's line and branch counts that moved between runs are explained in
  [Run-to-run variance](#run-to-run-variance).
- **Tests:** 4,203 after (`ctest -N`), the same as before: one test was
  deleted and one added.
  The shared set (`-LE 'models|timing'`) has 3,964 tests, all passing in
  both runs, with no skips.
- **Shared set wall time:** 569.5 s before and 578.8 s after.
  - The 477 GPU tests run one at a time and take 567.3 and 576.5
    s·proc, so they are the whole wall time. F14 changed none of their
    work; the difference is run-to-run.
  - The 3,455 CPU tests take 381.4 and 397.4 s·proc, over 20 cores.

| Totals (`src/`, CPU and fake-backend tests) | Before | After | Second after run |
| --- | ---: | ---: | ---: |
| Lines covered (of 142,615) | 123,763 (86.78%) | 123,767 | 123,768 |
| Branches covered (of 105,357) | 74,792 (70.99%) | 74,794 | 74,797 |
| Functions covered (of 9,514) | 8,643 (90.85%) | 8,643 | 8,643 |
| Regions covered (of 130,458) | 111,872 (85.75%) | 111,874 | 111,876 |

## What changed

| Change | Tests | Why | Covered by |
| --- | --- | --- | --- |
| Deleted | `U4DiagnosticExporterTest.StreamsFramedPlanRecordsAndPreservesMalformedPartialFiles` and `tests/support/u4_diagnostic_export.h` | It tested only the test-support export of the U4 diagnostics, which F12 removed. Its only `src/` reach was the plan-record event enum. | Nothing is lost: the per-file comparison shows no drop. |
| Deleted (dead support) | `OpLine` (`tests/support/plan_record`), `benchmarks/paging_cases.h` | Their consumers, the EXL3 plan comparator and BP-P2's paged harness, were retired in M3.6 (0d127ac0). Nothing included or called them. | — |
| Added | `AcquireTest.ALaunchOverAClosureNotAllResidentIsRefusedBeforeItRuns` | `tests/support/paged_programs.h`'s `LaunchOnlyProgram` had lost its only user, the retired BP-P2 harness. No CPU or fake-backend test reached the refusal of a launch over a closure that is not all resident (`scheduler.cc` lines 470–472 had a count of 0). The new test covers it with none of the closure resident, all but one extent resident, and all of it resident. | — |
| Merged into one helper | `GgmlExtOpsTest.Gemma{2,3}DeviceMasksMatchEveryHostByte…` | The bodies were identical, about 90 lines each, apart from the model's window, cells and trained maximum. | `GgmlExtOpsTest::SlidingMaskControl`. Both test names stay. |
| Merged into one helper | `GgmlExtOpsTest.Gemma{2,3,4}UnequalOwnerPadding…` | About 200 lines each, differing only in head shape, the Q and K scales and Gemma 2's softcap. | `GgmlExtOpsTest::UnequalOwnerPaddingControl`. The three names stay. The FP64 NMSE values and plan lines they print are identical before and after. |
| Merged into one helper | `ImportWriterTest.{QwenSideTable…,Dsv4HashTables…,Qwen2Placement…,Qwen2Exl3Placement…}FundedBeforeEffects` | One flow (normalize, inspect the placement, refuse one byte less of metadata before any effect), differing only in the checkpoint and the expected placement. | `ImportWriterTest::ExpectPlacementFundedBeforeEffects`. The Qwen3.8 case keeps its extra receipt check. |
| Merged into one helper | `Qwen38{Gguf,Mtp}KeptGpu.ConversationKeptAcrossARestartContinuesExactly` | The bodies were identical apart from `expect_drafts`. | `Qwen38GgufKeptGpu::CheckKeptRestartContinuesExactly`. Both names stay. |
| Renamed | `ImportPipelineModelsTest.QwenImageComponentsAreTheV0Imports` → `…AreTheV1Imports` | It compares against the published v1 composition. v0 is gone. | — |
| Comments fixed | 11 `models` test files | The comments promised a skip where the code now fails (F12). | — |

## Candidates reviewed and kept

**Copy-paste.** A scan of literal-normalized token streams in
`tests/unit` found 14 same-file groups that are identical apart from
literals, and 94 pairs with similarity of at least 0.85. Most pairs test
different operations or formats through tests of the same shape. Examples:

- the biased and hash routes in `ggml_lower_test`;
- ModelOpt against MLX admission;
- dense against QSA Q8 state;
- the Qwen-Image block against its flow graph.

Merging those would interleave behaviours behind conditionals, so they
stay. The small groups are already one-line calls to a shared helper (the
template fixtures, the image transfer widths) or are short enough that a
table would not read better (the tool-call, Jinja and admission-deadline
cases).

**Coverage subsets.** Coverage was collected per test, which gives the
lines and taken branch arms of `src/` that each test reaches:

- 380 of the 3,406 profiled tests reach a subset of another test's lines
  and branches in the same executable;
- 422 pairs reach exactly the same lines and branches. Of those, 405 pairs
  fall in 30 groups of distinct tests, excluding instances of one
  parameterized test.

Every group read differs in what it asserts, so none was removed on
coverage grounds. Examples:

- `ReclaimOrder`'s staleness and cost-per-byte orders;
- `AdaptiveDepth`'s steady and changing acceptance;
- `Sample`'s distribution and reproducibility;
- the server's two "no limit by default" tests.

Three facts limit what per-test coverage says:

- A death test's child dies before it writes its profile, so death tests
  show as subsets of whatever shares their setup.
- Tests of `tests/support` or `benchmarks/` code reach almost nothing in
  `src/`.
- Two tests reaching the same lines can check different values.

**Slow GPU tests.** These cannot be measured with this build, which has
no CUDA. `GgmlExtOpsTest` (82 tests) and `Dsv4FastTest` (24 tests) take
most of the GPU time. Their slowest tests are deliberate type and shape
ladders: `WaveLaunchesKeepEachTokensSumsBitForBit` takes about 50 s and
`PairedExpertPreparationMatchesSeparateProductsExactly` about 34 s. No two
of them are duplicates, so they stay. Shrinking them is a judgment for the
kernels' owners, not for this audit.

## Run-to-run variance

Thirteen files changed their counts between the three runs. The
before/after/second-run counts for each:

| File | Lines covered | Branches covered | Reached by |
| --- | --- | --- | --- |
| `src/base/bounded_queue.h` | 54 / 54 / 54 | 17 / 18 / 17 | `KernelMemoryTest.IsMappedBeforeTheMeasurementsAndRetiredAfterTheLaunch` |
| `src/catalog/catalog.cc` | 657 / 657 / 657 | 304 / 305 / 304 | `EngineLlmTest.CheckpointCapacityIsHeldWhileFallbackStagingReclaims` |
| `src/chat/jinja_eval.cc` | 3,339 / 3,343 / 3,343 | 2,099 / 2,101 / 2,101 | `JinjaBounds.HostileTemplatesAreRefusedOrCancelled` (where a cancel lands) |
| `src/engine/execution_contract.cc` | 315 / 316 / 315 | 240 / 242 / 240 | `ExecutionMetadataAllowanceTest.ConcurrentRefundAndDriverChargeConserveTheOriginalReceipt` |
| `src/platform/direct_io.cc` | 116 / 116 / 116 | 71 / 71 / 73 | (only the second run moved) |
| `src/platform/kept_files.cc` | 298 / 297 / 297 | 169 / 168 / 168 | `KeptFiles.ANameChangedDuringTheOpenIsLookedAtAgain`, a rename race (line 115) |
| `src/runtime/api_connection.cc` | 980 / 980 / 980 | 475 / 475 / 479 | (only the second run moved) |
| `src/runtime/api_dispatch.cc` | 760 / 757 / 757 | 248 / 246 / 246 | `ServerTest.QueuedRequestsLeaveTheDriverItsHeadroom`, a wake that lands mid-loop (lines 784–786) |
| `src/runtime/state_keeper.cc` | 441 / 441 / 441 | 181 / 180 / 180 | `StateKeeper.ARecordDescribesItsFilesExactly`, a worker that sees the stop (line 64) |
| `src/scheduler/completions.cc` | 254 / 252 / 254 | 116 / 115 / 116 | `CompletionBoard.ThreadedPublicationReachesTheOwner` |
| `src/scheduler/pagein.cc` | 947 / 947 / 947 | 643 / 642 / 643 | `PageInTest.ThreadedLoadsAndEvictions…` |
| `src/scheduler/scheduler.cc` | 870 / 872 / 872 | 515 / 517 / 516 | the new acquire test (the refusal at lines 470–472), and `SchedulerTest.ThreadedLanesKeepEveryHandoffOrdered` (line 146) |
| `src/scheduler/services.cc` | 1,187 / 1,190 / 1,190 | 546 / 546 / 546 | `SchedulerTest.ThreadedLanesKeepEveryHandoffOrdered`, `PageInTest.ThreadedLoadsAndEvictions…` |

Apart from the new acquire test's refusal, the tests that reach the
moving lines are unchanged and live in unchanged executables. The lines depend on thread interleaving, so they come and go
with load:

- The completions and page-in branches came back in the second run.
- The three that stayed lower in both later runs were checked directly.
  Each test was repeated 40 times on its own, and every one of those lines
  was reached (`api_dispatch.cc` 784: 40 times; `kept_files.cc` 115: 24
  times; `state_keeper.cc` 64's stop arm: 34 times).

### Coverage by module (before, after)

Lines and branches of `src/` that the CPU-only build compiles (468
files). Files only CUDA builds compile are not in this table.

| Module | Files | Lines | Covered before | Covered after | Branches | Covered before | Covered after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `artifact` | 13 | 3,276 | 2,848 (86.9%) | 2,848 | 1,934 | 1,561 (80.7%) | 1,561 |
| `base` | 16 | 1,617 | 1,523 (94.2%) | 1,523 | 908 | 799 (88.0%) | 800 |
| `catalog` | 2 | 751 | 667 (88.8%) | 667 | 400 | 309 (77.2%) | 310 |
| `chat` | 19 | 11,628 | 10,439 (89.8%) | 10,443 | 7,995 | 6,716 (84.0%) | 6,718 |
| `cli` | 4 | 752 | 507 (67.4%) | 507 | 438 | 233 (53.2%) | 233 |
| `components` | 29 | 10,994 | 10,307 (93.8%) | 10,307 | 6,836 | 5,310 (77.7%) | 5,310 |
| `config` | 3 | 1,999 | 1,809 (90.5%) | 1,809 | 1,162 | 977 (84.1%) | 977 |
| `engine` | 67 | 19,083 | 16,503 (86.5%) | 16,504 | 12,243 | 8,376 (68.4%) | 8,378 |
| `execution` | 9 | 1,626 | 1,545 (95.0%) | 1,545 | 1,216 | 1,067 (87.7%) | 1,067 |
| `format` | 6 | 754 | 677 (89.8%) | 677 | 522 | 457 (87.5%) | 457 |
| `import` | 31 | 11,379 | 10,269 (90.2%) | 10,269 | 9,626 | 6,976 (72.5%) | 6,976 |
| `ir` | 13 | 5,372 | 4,814 (89.6%) | 4,814 | 3,736 | 2,655 (71.1%) | 2,655 |
| `kernels/common` | 14 | 770 | 557 (72.3%) | 557 | 406 | 258 (63.5%) | 258 |
| `kernels/decode` | 1 | 4 | 4 (100.0%) | 4 | 0 | 0 | 0 |
| `kernels/exl3` | 4 | 659 | 530 (80.4%) | 530 | 532 | 373 (70.1%) | 373 |
| `kernels/ggml` | 35 | 16,418 | 12,271 (74.7%) | 12,271 | 20,437 | 10,839 (53.0%) | 10,839 |
| `kernels/paging` | 2 | 97 | 91 (93.8%) | 91 | 44 | 36 (81.8%) | 36 |
| `kernels/reference` | 4 | 2,946 | 2,565 (87.1%) | 2,565 | 1,820 | 1,462 (80.3%) | 1,462 |
| `memory` | 4 | 704 | 628 (89.2%) | 628 | 382 | 297 (77.7%) | 297 |
| `model` | 9 | 1,557 | 1,470 (94.4%) | 1,470 | 1,036 | 854 (82.4%) | 854 |
| `planner` | 18 | 11,842 | 10,930 (92.3%) | 10,930 | 8,746 | 6,629 (75.8%) | 6,629 |
| `platform` | 19 | 1,968 | 1,310 (66.6%) | 1,309 | 934 | 549 (58.8%) | 548 |
| `providers` | 10 | 915 | 783 (85.6%) | 783 | 488 | 350 (71.7%) | 350 |
| `providers/fake` | 7 | 973 | 851 (87.5%) | 851 | 386 | 298 (77.2%) | 298 |
| `runtime` | 75 | 18,041 | 15,456 (85.7%) | 15,453 | 11,258 | 8,155 (72.4%) | 8,152 |
| `scheduler` | 20 | 5,269 | 4,754 (90.2%) | 4,757 | 3,180 | 2,522 (79.3%) | 2,522 |
| `spec` | 23 | 6,399 | 5,179 (80.9%) | 5,179 | 4,913 | 3,720 (75.7%) | 3,720 |
| `tokenizer` | 11 | 4,822 | 4,476 (92.8%) | 4,476 | 3,779 | 3,014 (79.8%) | 3,014 |

## The coverage build

The pinned SDK already has what source-based coverage needs: `clang`,
`bin/llvm-profdata`, `bin/llvm-cov`, and the profile runtime
`llvm/lib/clang/22/lib/linux/libclang_rt.profile-aarch64.a`. The build
is a separate binary directory on a Spark: CPU-only, the native Spark
toolchain, the `cpu` profile's settings, and coverage flags given through
the environment. It needs no preset.

```
S=$(tools/setup-toolchain --print-root)
env LLMP_SDK=$S CFLAGS="-fprofile-instr-generate -fcoverage-mapping" \
    CXXFLAGS="-fprofile-instr-generate -fcoverage-mapping" LDFLAGS="-fprofile-instr-generate" \
  $S/bin/cmake -S . -B build/coverage -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/toolchains/aarch64-linux-gnu-native.cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo "-DCMAKE_C_FLAGS_RELWITHDEBINFO=-O2 -DNDEBUG" \
    "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -DNDEBUG" -DLLMP_SDK=$S -DLLMP_CUDA=OFF \
    -DLLMP_LIBSTDCXX_ASSERTIONS=ON -DLLMP_REQUIRE_LOCKED_SOURCES=ON -DLLMP_MODULES= \
    -DCMAKE_TEST_LAUNCHER=$HOME/coverage/launch.sh
$S/bin/cmake --build build/coverage      # about 2 minutes on a Spark
F14_PHASE=before LLVM_PROFILE_FILE=$HOME/coverage/stray/other-%p.profraw \
  $S/bin/ctest --test-dir build/coverage -LE 'models|timing' -j 20   # about 40 s
```

`CMAKE_TEST_LAUNCHER` reaches every GoogleTest case, discovery included.
The launcher names each case's profile by its executable and filter, so
every profile belongs to one test:

```
#!/bin/sh
exe="$1"; name=$(basename "$exe"); filt=run
for a in "$@"; do
  case "$a" in
    --gtest_list_tests) export LLVM_PROFILE_FILE="$HOME/coverage/stray/list-%p.profraw"; exec "$@" ;;
    --gtest_filter=*) filt=${a#--gtest_filter=} ;;
  esac
done
safe=$(printf "%s" "$filt" | tr -c "A-Za-z0-9_.-" "_" | cut -c1-200)
export LLVM_PROFILE_FILE="$HOME/coverage/raw/${F14_PHASE:-x}/${name}__${safe}__%p.profraw"
exec "$@"
```

Profiles of processes the launcher does not start are counted in the
totals. These are the smoke tests' `llmp` runs and the toolchain
controls, written to `stray/other-*`. Discovery's `list-*` profiles are
dropped.

- **The aggregate:** `llvm-profdata merge -sparse` of every profile, then
  `llvm-cov export -summary-only` over every executable that ran plus
  `llmp` and `llmp-runtime`. Files outside `src/` are excluded with
  `-ignore-filename-regex='/src/llmp/(build|tests|benchmarks|third_party|cmake)/|^/usr/|/\.local/'`.
- **One test:** its profiles are merged, then
  `llvm-cov export -format=lcov -skip-functions` is run on its executable,
  keeping the lines with a nonzero count and the taken branch arms.

The 3,406 tests take about 2 minutes on 18 threads.
