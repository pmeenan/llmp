<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CMake 4.4.3 source preparation and build smoke

D-058 pins **CMake 4.4.3**, the stable release listed by
[Kitware](https://cmake.org/download/) on 2026-09-23. Both Linux binary
archives were downloaded over HTTPS and checked against the upstream
SHA-256 manifest; [pins.json](pins.json) records exact URLs, hashes and
sizes. The ARM archive was checked again after transfer to `spark`.
The source-preparation checks and a CMake-driven repeat of D-032's small
native/cross/CUDA smoke validate the pin under D-012. They do not validate
M1's future dependency manager, application build or packaging.

## Inputs and results

| Build host | Binary archive | SHA-256 | Result, 2026-09-23 |
| --- | --- | --- | --- |
| Workstation `linux`, x86-64 | `cmake-4.4.3-linux-x86_64.tar.gz` | `d6c83076c575bc00b823522ac974bda66d0af05d6ddc30e739c12385cf32c6cc` | 7/7 checks passed |
| `spark` (`spark-c4e2`), AArch64 | `cmake-4.4.3-linux-aarch64.tar.gz` | `2efc974dbd63b4444c0e8494b92f2e80c2d7e635b4b80eac2916985ddd8f72a6` | 7/7 checks passed |

The [retained harness](smoke.py) creates a synthetic local archive and uses
`cmake_minimum_required(VERSION 4.4.3)` in each script/project. It checks:

| Case | Observed on both hosts |
| --- | --- |
| Long-form script-mode `FetchContent_Populate(name ...options...)`, with empty `PATH` | Exact payload extracted without a generator/build tool; upstream `CMakeLists.txt` was not executed |
| Same call with both disconnected flags enabled, fresh source directory | Still extracts: this form ignores the flags |
| Incorrect archive SHA-256 | Population fails |
| Deprecated single-argument `FetchContent_Populate(name)` in a configured project | Rejected under `CMP0169=NEW` |
| Declared dependency with `FetchContent_MakeAvailable` and fully disconnected mode, missing source directory | Rejected under `CMP0170=NEW` before population returns |
| Same declared dependency with prepopulated source | Accepted |
| Same dependency with modified prepopulated payload | Modified bytes accepted: disconnected mode does not verify source integrity |

`CMP0168=NEW` removes FetchContent's internal sub-build; the long-form script
case therefore needs no Ninja/Make. The configured cases use system Make
as a generator with `LANGUAGES NONE`; no compiler or build executes.
The fixture's root `CMakeLists.txt` deliberately fails if run; the declared
cases select an absent `SOURCE_SUBDIR` so they only expose source bytes.

These are local-input semantic checks; the compiler integration check is
below. Neither tests HTTPS acquisition inside CMake, network isolation,
CTest/CPack, full SDK cleanliness or real library compatibility. M1 still
owes D-057's broader gates. No old-policy compatibility is selected or
promised, and no host-default tools were changed.

## Native, cross and CUDA integration

[CMakeLists.txt](CMakeLists.txt) and [build-smoke.sh](build-smoke.sh) drive
D-032's existing [CPU](../toolchain-smoke/main.cc) and
[CUDA](../toolchain-smoke/kernel.cu) fixtures through CMake's CXX/CUDA
language support. The compiler/SDK inputs are D-032's
[recorded artifacts](../toolchain-smoke/artifacts.json): Clang 22.1.8,
NVCC 13.4.92, the Spark sysroot and ARM cudart 13.4.92. The cross host
wrapper supplies AArch64, sysroot, GCC search root and LLD explicitly;
linker selection is applied only to linking steps. Profiles use separate,
fresh build directories, Release mode and system Make as the experiment
generator. This does not choose M1's Ninja pin.

| Build and execution | Result, 2026-09-23 |
| --- | --- |
| Workstation native x86-64 CPU | CMake configure/build and CPU C++23 execution passed |
| Workstation cross-build to AArch64; execute on `spark` | CMake configure/build and both CPU/CUDA executions passed |
| Native AArch64 build and execution on `spark` | CMake configure/build and both CPU/CUDA executions passed |

Compile commands selected C++23 for both languages, explicit `x86-64` or
`armv8-a` CPU targets and `121-real` CUDA architecture (`sm_121` cubin).
Both GPU runs used GB10 / driver **580.178.04**, with
`CUDA_DISABLE_PTX_JIT=1`, and verified all **257** values. Every CPU run
passed its C++23 checks and reported `__GLIBCXX__=20240904`. The ARM
executables' ELF architecture and dynamic dependencies were inspected;
CUDA executables depend on `libcudart.so.13` and have no SDK RPATH/RUNPATH.
The SDK was used from external scratch, without package installation or
system-default changes. This is a bounded compiler integration smoke,
not full M1 provisioning or application/backend evidence.

## Reproduction

Download the appropriate archive and checksum manifest from [pins.json](pins.json),
verify their SHA-256 values, and extract outside the checkout. With Python 3
and system Make available, run from the repository root:

```sh
python3 docs/experiments/cmake-fetchcontent/smoke.py \
  --cmake /absolute/path/to/cmake-4.4.3-linux-x86_64/bin/cmake
```

For Spark, copy the harness, `pins.json` and verified ARM archive to external
scratch, extract there, and supply the ARM binary's absolute path. The
harness refuses another CMake version, uses temporary directories outside
the repository, removes its fixtures and prints only the result summary.
It expects the caller to verify the downloaded binary archive against the
pins before running it. Re-running uses new source/build directories.

For the integration smoke, first reproduce the verified extracted SDK inputs
and environment from the [D-032 instructions](../toolchain-smoke/README.md).
Set `CMAKE` to the appropriate extracted 4.4.3 binary; `CXX` and
`LD_LIBRARY_PATH` select Clang as in that report. Cross builds additionally
set `SYSROOT`, `LLD` and `CUDA_ROOT`; native Spark sets `CUDA_ROOT` to its
ARM toolkit. Then use fresh directories outside the repository:

```sh
bash docs/experiments/cmake-fetchcontent/build-smoke.sh native /tmp/cmake-native
/tmp/cmake-native/cpu-smoke
bash docs/experiments/cmake-fetchcontent/build-smoke.sh cross /tmp/cmake-cross
```

Copy the cross `cpu-smoke` and `cuda-smoke` binaries plus the pinned ARM
`libcudart.so*` to Spark scratch. Execute both, supplying the cudart directory
through `LD_LIBRARY_PATH` and `CUDA_DISABLE_PTX_JIT=1` for CUDA. For native
Spark, copy this experiment and its sibling `toolchain-smoke` directory,
select ARM CMake/Clang/CUDA inputs, run `build-smoke.sh spark` into another
fresh directory, and execute both resulting binaries the same way. Inspect
`compile_commands.json`, ELF architecture and dynamic dependencies, including
absence of RPATH/RUNPATH. Generated files and logs stay outside Git.

## Provenance and use

CMake is a D-017 **build tool**. The archives contain Kitware's BSD-3-Clause
text at `doc/cmake/LICENSE.rst` plus bundled-component notices below
`doc/cmake/`; keep that directory intact. The smoke uses these binaries
externally and incorporates no CMake source or model data into llmpalooza.
This is not a full binary-bundle redistribution audit. M1's SDK audit must
record the bundled components and any shipped notices/terms.

The checked version's documentation describes
[script-mode population and disconnected flags](https://cmake.org/cmake/help/v4.4/module/FetchContent.html),
[CMP0168](https://cmake.org/cmake/help/v4.4/policy/CMP0168.html),
[CMP0169](https://cmake.org/cmake/help/v4.4/policy/CMP0169.html), and
[CMP0170](https://cmake.org/cmake/help/v4.4/policy/CMP0170.html).
