<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Development tool pins: Ninja, GoogleTest, LLVM tools and libstdc++ 14

D-059 pins the rest of the development tool set:

- Ninja 1.13.2;
- GoogleTest 1.18.0;
- clang-format, clang-tidy, clangd and llvm-symbolizer, from the same
  LLVM 22.1.8 build as D-032;
- GCC 14.2 libstdc++ development files, which replace D-032's GCC 13 headers;
- the candidate [`.clang-format`](.clang-format), [`.clang-tidy`](.clang-tidy)
  and warning set in [CMakeLists.txt](CMakeLists.txt).

**Superseded in part (D-060):** Llmpalooza now uses a source-built GCC 16.2
runtime linked statically ([report](../gcc16-static/README.md)). The GCC 14.2
results below stay as the record for D-059. `build.sh` accepts
`STATIC_RUNTIME=1` and any `GCC_INSTALL_DIR`.

Exact URLs, hashes and sizes are in [pins.json](pins.json). All of it ran on
2026-09-23 on the workstation and `spark` (`spark-c4e2`, GB10, driver
580.178.04). The SDK was extracted into external scratch; no host packages or
defaults changed. This is an M0 tool smoke. M1's persistent SDK, presets,
CI, lock file and application build are still owed.

## Versions checked

Latest releases as listed upstream on 2026-09-23:

| Tool | Pin | Status |
| --- | --- | --- |
| Ninja | 1.13.2 | Latest release (2025-11-20) |
| GoogleTest | 1.18.0 | Latest release (2026-08-10); requires C++17 |
| LLVM | 22.1.8 | Latest 22.x. LLVM 23.1.2 exists, but D-032 found 23 outside CUDA 13.4's host-compiler range. NVIDIA's CUDA 13.4 Update 1 guide lists Clang 7–22 |
| CUDA | 13.4.92 | Newest `cuda-nvcc-13-4` in NVIDIA's Ubuntu 24.04 repository |
| libstdc++ | 14.2.0 | Newest official Ubuntu 24.04 package; GCC 16.2 is the latest upstream release (2026-08-07) |

CUDA 13.4 Update 1 supports GCC 6–16 as the host compiler, and libstdc++
as the only standard library
([installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html),
read 2026-09-23). The limit on newer libstdc++ comes from the platform, not
CUDA. Ubuntu 24.04's archive stops at GCC 14.2, and both hosts already run
the `libstdc++6` 14.2.0 runtime. The `ubuntu-toolchain-r/test` PPA offers
GCC 15.2 and a GCC 16 pre-release snapshot (16-20260315). Installing either
replaces the system runtime. D-059 records the static-runtime and
GCC 16.2-from-source evaluation as the next toolchain item.

## libstdc++ 14.2

D-032 selected GCC 13.3 headers, which have no `<print>`. GCC 14.2's
`libstdc++-14-dev` and `libgcc-14-dev` were downloaded through apt on each
host, so each was checked against the signed `noble-updates` index. They were
then extracted to scratch: a workstation overlay, and an ARM copy of D-032's
Spark sysroot snapshot. Every compile passes
`--gcc-install-dir=<root>/usr/lib/gcc/<triple>/14` explicitly. Clang would
otherwise pick the highest version it finds. The overlay's
`libstdc++.so.6` link points at the installed 14.2 runtime.

| Check | Result |
| --- | --- |
| C++23 library probe: `<print>`/`std::println`, `<generator>`, monadic `std::expected`, `__cpp_lib_print=202211` | Passed natively on the workstation and cross-built on `spark` |
| `<flat_map>`, `<mdspan>` | Not present in GCC 14.2 (GCC 15 or later) |
| Probe binary's highest required symbol version | `GLIBCXX_3.4.31`, provided by the installed 14.2 runtime |
| D-032/D-058 CPU and CUDA smoke, cross-built to `spark` | Both passed, `libstdc++=20240908`; CUDA checked 257 values |
| Same smoke built natively on `spark` (NVCC with Clang as host compiler) | Both passed, with the same checks |
| NVCC `.cu` whose host code uses `<print>`, `<format>`, `<expected>`, `<ranges>`, with `-Werror all-warnings` | Compiled and ran on the GB10 |

CUDA runs used `CUDA_DISABLE_PTX_JIT=1`.

## Build, test and sanitizer smoke

[prepare.cmake](prepare.cmake) populates GoogleTest from the local archive in
script mode with `URL_HASH` (D-057/D-058). The prepared tree was identical to
the archive. An archive with one byte appended failed SHA-256 verification.
Population stopped, leaving an empty source directory with no extracted
bytes. A later identity check must treat an existing directory as
unprepared. By default, script-mode population writes its stamp
and build directories into the caller's working directory, so a run from the
repository root would write into the checkout. The script therefore sets
`FETCHCONTENT_BASE_DIR` and `BINARY_DIR` beside `SOURCE_DIR`, and a rerun
left the working directory empty. [build.sh](build.sh) configures with CMake 4.4.3 and
the pinned Ninja, using `RelWithDebInfo` and fresh build directories.
GoogleTest is added `EXCLUDE_FROM_ALL SYSTEM`, with `INSTALL_GTEST=OFF` and
`GTEST_HAS_ABSL=OFF`. The candidate warnings apply only to llmpalooza targets:

    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
    -Wnon-virtual-dtor -Wold-style-cast -Wimplicit-fallthrough -Werror

[devtools_test.cc](devtools_test.cc) covers the GoogleTest features M1
expects to use:

- plain tests;
- `std::expected` results;
- value-parameterized tests with a name generator;
- typed tests;
- ordered gMock expectations;
- a thread-safe death test.

Every profile uses `gtest_discover_tests(... DISCOVERY_MODE PRE_TEST)`.

| Profile | Result |
| --- | --- |
| Workstation native x86-64 | 10/10 passed |
| Workstation, negative control (`FAILING=ON`) | CTest exit 8, `NegativeControl.Fails` reported |
| Workstation ASan+UBSan (GoogleTest instrumented too) | 10/10 passed |
| Cross AArch64 build; CTest on the workstation runs discovery and tests on `spark` through `CMAKE_CROSSCOMPILING_EMULATOR` (SSH, same absolute path) | 10/10 passed; negative control exit 8 |
| Native `spark` build | 10/10 passed |
| Native `spark` ASan+UBSan | 10/10 passed |

Cross test binaries need only `libstdc++.so.6`, `libm`, `libgcc_s` and
`libc`, and have no RPATH/RUNPATH.

[asan_probe.cc](asan_probe.cc) overflows a heap buffer on purpose. With
`ASAN_SYMBOLIZER_PATH` set to the pinned `llvm-symbolizer-22`, the report's
top frame resolves to `overflow_probe` at `asan_probe.cc:12` on both
architectures. Without the symbolizer, frames are addresses only.

**Finding: C++ module scanning.** With Ninja and C++20 or later, CMake 4.4
scans every source for module dependencies by default using
`clang-scan-deps`. Through the cross-compiler wrapper, CMake could not find
the scanner (`CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS-NOTFOUND`), and the build
failed. Llmpalooza uses no C++ modules, so the harness sets
`CMAKE_CXX_SCAN_FOR_MODULES OFF`. That also removes a scan and dyndep step
per source.

## Formatter, linter and language server

- **Style.** Google style with 100 columns. Across the tracked experiment
  C++, it needed the fewest changes (diff lines) of the built-in styles tried:
  2,568, against 2,992 for LLVM at 100 columns and 3,022 for Google at 80.
  On the most carefully written file, `async-model/main.cc`, it needed 62
  diff lines, against 206 for LLVM at 100 and 159 for Google at 80. The
  harness sources are clean under `--dry-run -Werror`. Formatting
  `async-model/main.cc` gave byte-identical output on x86-64 and AArch64.
  Existing experiment sources are frozen evidence and are not reformatted.
- **clang-tidy.** The check set is bugprone, clang-analyzer, concurrency,
  misc, modernize, performance, portability and readability, plus selected
  cert and cppcoreguidelines checks. The excluded checks are listed in the
  file. Findings are errors.
  - The harness sources are clean on both architectures.
  - A negative control fired five distinct checks, including
    `cppcoreguidelines-init-variables` and `bugprone-sizeof-expression`,
    with exit 1.
  - On existing experiment code it reports 6 findings in
    `async-model/main.cc`. These are `modernize-use-ranges` (3),
    `bugprone-unchecked-optional-access`, `concurrency-mt-unsafe` and
    `readability-avoid-nested-conditional-operator`. It reports 2 in
    `toolchain-smoke/main.cc`.
  - `HeaderFilterRegex` limits header findings to project paths. CUDA
    headers passed with `-I` instead of `-isystem` otherwise produce
    hundreds of findings.
  - `modernize-use-std-print` stays enabled because libstdc++ 14 has
    `<print>`.
- **clangd.** `clangd-22 --check` loaded the compile database, then indexed
  and built the test file's AST with no compile diagnostics. It also
  reported 22 failures from its exhaustive code-action ("tweak") pass. All
  of them are inside gMock/GoogleTest macro-generated classes. These are
  not diagnostics and don't affect editing. An earlier `PrintTo` overload
  drew an unused-function warning that the compiler didn't emit; a name
  generator replaced it.

## Reproduction

Use scratch outside the repository, as in the D-032 and D-058 reports.

1. Fetch every entry in [pins.json](pins.json) and check its SHA-256 and
   size. The LLVM entries use D-032's gpgv-verified index. For libstdc++, run
   `apt-get download` on a host of the matching architecture, or fetch the
   recorded URL and compare the hash. Extract D-032's LLVM packages plus the
   LLVM tool packages with `dpkg-deb -x` into `sdk-amd64` and `sdk-arm64`.
   Extract the libstdc++ 14 packages into an overlay on each host (amd64 on
   the workstation, arm64 on `spark`) and into a copy of D-032's Spark
   sysroot snapshot. Link each overlay's `usr/lib/<triple>/libstdc++.so.6`
   to that host's installed 14.2 runtime.
2. Prepare GoogleTest with the pinned CMake:
   `cmake -DARCHIVE=/abs/googletest-1.18.0.tar.gz -DSOURCE_DIR=/abs/new-dir -P prepare.cmake`.
3. Set `CMAKE`, `NINJA`, `CXX` (Clang 22), `LD_LIBRARY_PATH` (the SDK's
   library directory), `GOOGLETEST_SOURCE_DIR` and `GCC_INSTALL_DIR`. Then
   run `build.sh native|spark DIR` and `ctest` in `DIR`. `FAILING=ON` adds
   the negative control. `SANITIZE=1` instruments everything and adds the
   probe.
4. Cross: also set `SYSROOT`, `LLD` and `REMOTE`, and point `GCC_INSTALL_DIR`
   at the sysroot's GCC 14 directory. Run `build.sh cross DIR`, copy `DIR` to
   the same absolute path on the remote host, and run `ctest` in `DIR` on the
   workstation.
5. Tools: run `clang-format-22 --dry-run -Werror *.cc` and
   `clang-tidy-22 -p DIR <file>` from this directory, and
   `clangd-22 --compile-commands-dir=DIR --check=devtools_test.cc`.

## Provenance and use

- **Ninja** (Apache-2.0) and **the LLVM tools**
  (`APACHE-2-LLVM-EXCEPTIONS`) are D-017 build tools and are not shipped.
  The Ninja zip contains only the binary, so M1's SDK audit takes the notice
  from upstream.
- **GoogleTest** (BSD-3-Clause) is linked into test executables only.
  - It is the dependency on which M1 demonstrates D-057's lock and offline
    build.
  - `gmock-pp.h` and `gmock-generated-actions.h` have no per-file header and
    are covered by the root LICENSE. No other licenses were found in the
    compiled or included files.
  - Its `find_package(Threads)` resolves against the platform. The
    Abseil/RE2 and Python lookups are off or unused with these options.
- **libstdc++ 14** is a D-017 declared platform dependency under the GCC
  Runtime Library Exception.

No package, archive or build output is committed.
