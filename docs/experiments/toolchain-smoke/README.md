<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M0 toolchain smoke — 2026-09-21

This report keeps aggregate outcomes and reproduction/dependency provenance.
Raw run output stays in external scratch and is not versioned.

**Passed with C++23 throughout:** workstation Clang native CPU build/run;
workstation AArch64 CPU and NVCC/Clang CUDA cross builds deployed and run on
`spark`; native Spark CPU/CUDA fallback build/run. D-032 selects CUDA Toolkit
13.4.2's compiler/runtime components with Clang 22.1.8 for M1.
The [compiler comparison](compiler-choice.md) explains the version ceiling
and why Clang remains the primary compiler.

This is a retained experiment, not application scaffolding or the M1 SDK
provisioner. The three `build.sh` profiles are the experiment's presets;
CMake presets, mise, clean-container provisioning, and CI remain M1 work.

## Validated versions

| Component | Exact version / selection |
| --- | --- |
| Clang and cross LLD | 22.1.8, apt.llvm.org Noble packages `1:22.1.8~++20260714014902+ca7933e47d3a-1~exp1~20260714135019.80` on x86-64 and ARM |
| Compiler-rt sanitizer support (2026-09-22 follow-up) | `libclang-rt-22-dev` at the same exact LLVM package version, amd64 and arm64; CPU ASan/UBSan execution validated in the [async experiment](../async-model/README.md) |
| LLVM compiler dependency | Ubuntu `libz3-4` `4.8.12-3.1build1`, both architectures |
| Ordinary C++ / CUDA dialect | C++23 / C++23 |
| libstdc++ and GCC support headers | `libstdc++-13-dev`, `libgcc-13-dev` `13.3.0-6ubuntu2~24.04.1`, both hosts |
| libstdc++ / libgcc shared runtime | `libstdc++6`, `libgcc-s1` `14.2.0-4ubuntu2~24.04.1`, both hosts |
| glibc runtime and development files | `2.39-0ubuntu8.9`, both hosts |
| Sysroot Linux headers | `linux-libc-dev:arm64` `6.8.0-139.139` |
| Selected CUDA release | Toolkit 13.4.2 (`cuda-toolkit-13-4` metapackage `13.4.2-1`, inspected, not installed) |
| NVCC, CRT, libNVVM, libnvptxcompiler, cudart/runtime development files | `13.4.92-1` for both x86-64 and ARM; NVCC V13.4.92, build `compiler.38855100_0` |
| CCCL | `cccl-13-4` package `13.3.4.3.1-1` on both architectures |
| Native Spark linker | GNU binutils `2.42-4ubuntu2.10` (cross path uses LLD) |
| CPU targets | `x86_64-linux-gnu -march=x86-64`; `aarch64-linux-gnu -march=armv8-a` |
| GPU target | NVCC `-arch=sm_121`, target directory `sbsa-linux` |
| Execution target | `spark` / `spark-c4e2`, NVIDIA GB10 12.1, driver 580.178.04 |
| Target OS snapshot | DGX OS base 7.5.0, OTA 7.6.0; Ubuntu 24.04.5; kernel 7.0.0-1019-nvidia |

The sysroot is a 2026-09-21 snapshot of Spark's include directories, ARM
libraries, and GCC 13 support files, with the merged-/usr library alias
restored. The selected CUDA headers/runtime come from the separate 13.4.2
SDK, not Spark's installed 13.0. `artifacts.json` records downloaded package
hashes and normalized tar hashes of the sysroot and CUDA SBSA trees. These
identify the bytes tested; re-copying a later Spark is **not** the same pin.
SDK packages were extracted without installation or running maintainer
scripts. Drivers, system compiler defaults, and security settings are unchanged.

CUDA 13.x supports R580 through minor-version compatibility, with restrictions
on newer driver features and PTX. Both selected GPU runs passed with PTX JIT
disabled. This establishes native `sm_121` execution for this kernel on
580.178.04, not universal CUDA 13.4 feature support. New driver-dependent
APIs or PTX/JIT backends must separately validate their driver requirements.
Sources: [NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
[13.4 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html),
[toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive).

## Reproduction

Use Ubuntu 24.04 x86-64 for workstation steps. Set `smoke_root` to scratch
outside the repository (the recorded run used `/tmp/llmp-toolchain-smoke`).
Also set `llvm_root=/tmp/llmp-clang22` for the new compiler extraction
and outputs. Remote commands below assume these displayed fixed scratch paths. Build products
and SDK bytes are local artifacts, not committed.

1. Fetch the selected `packages` in `artifacts.json`. LLVM and NVIDIA
   entries have exact URLs; the `libz3-4` entries specify the Ubuntu package
   and version to download on each architecture. Check each SHA-256 before
   extracting with `dpkg-deb -x`. Extract LLVM (including `libclang-rt-22-dev`)
   plus libz3 into
   `$llvm_root/sdk-amd64` and `$llvm_root/sdk-arm64`, x86 CUDA into
   `$smoke_root/sdk-134`, and ARM CUDA into `$smoke_root/sdk-134-arm`.
   LLVM's InRelease signature, both package-index hashes, and package hashes
   were verified; the manifest records the signer and index identities.
   NVIDIA hashes were also checked against each repository's `Packages.gz`.
   This extraction uses existing system dependencies; it is not clean SDK
   provisioning. `baseline_llvm_packages` and `baseline_cuda_packages`
   record earlier comparisons, not selected dependencies.
2. Snapshot matching target files and add the ARM CUDA target to the x86 SDK:

   ```bash
   mkdir -p "$smoke_root/sysroot"
   rsync -aR spark:/usr/include spark:/usr/lib/aarch64-linux-gnu \
     spark:/usr/lib/gcc/aarch64-linux-gnu/13 spark:/lib/ld-linux-aarch64.so.1 \
     "$smoke_root/sysroot/"
   ln -s ../usr/lib/aarch64-linux-gnu "$smoke_root/sysroot/lib/aarch64-linux-gnu"
   cp -a "$smoke_root/sdk-134-arm/usr/local/cuda-13.4/targets/sbsa-linux" \
     "$smoke_root/sdk-134/usr/local/cuda-13.4/targets/"
   ```

   The full ARM library directory was copied for this spike; this is not a
   redistributable sysroot or a dependency whitelist. Selected CUDA packages
   provide compiler support and runtime libraries, not the full math SDK.
   Compare `sysroot` and the selected `targets/sbsa-linux` with their
   `artifacts.json` snapshot hashes using, for each directory:

   ```bash
   set -o pipefail
   tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
     -cf - -C "$snapshot_directory" . | sha256sum
   ```

3. From the repository root, build and run:

   ```bash
   export LD_LIBRARY_PATH="$llvm_root/sdk-amd64/usr/lib/x86_64-linux-gnu"
   export CXX="$llvm_root/sdk-amd64/usr/bin/clang++-22"
   export LLD="$llvm_root/sdk-amd64/usr/bin/ld.lld-22"
   export SYSROOT="$smoke_root/sysroot"
   export CUDA_ROOT="$smoke_root/sdk-134/usr/local/cuda-13.4"
   bash docs/experiments/toolchain-smoke/build.sh native "$llvm_root/native"
   "$llvm_root/native/cpu-smoke"
   bash docs/experiments/toolchain-smoke/build.sh cross "$llvm_root/cross"
   ssh spark 'mkdir -p /tmp/llmp-clang22/cross'
   rsync -a "$smoke_root/sdk-134-arm/" spark:/tmp/llmp-toolchain-smoke/sdk-134/
   rsync -a "$llvm_root/cross/" spark:/tmp/llmp-clang22/cross/
   ssh spark '/tmp/llmp-clang22/cross/cpu-smoke'
   ssh spark 'LD_LIBRARY_PATH=/tmp/llmp-toolchain-smoke/sdk-134/usr/local/cuda-13.4/targets/sbsa-linux/lib CUDA_DISABLE_PTX_JIT=1 /tmp/llmp-clang22/cross/cuda-smoke'
   ```

   The cross profile supplies Clang's target/sysroot/GCC search path through
   an explicit host-compiler wrapper, including NVCC's preprocessing phases.
   NVCC uses `--target-directory sbsa-linux`; LLD links against ARM libcudart.
   No host-compiler compatibility override is used. Inspect with `file` and
   `readelf -d`: the CUDA object/executable must be AArch64; the executable
   uses `libcudart.so.13` and has no baked-in SDK rpath.

4. Native Spark fallback, using the extracted ARM LLVM and CUDA SDKs:

   ```bash
   rsync -a "$llvm_root/sdk-arm64/" spark:/tmp/llmp-clang22/sdk-arm64/
   rsync -a docs/experiments/toolchain-smoke/ spark:/tmp/llmp-clang22/source/
   ssh spark 'LD_LIBRARY_PATH=/tmp/llmp-clang22/sdk-arm64/usr/lib/aarch64-linux-gnu CXX=/tmp/llmp-clang22/sdk-arm64/usr/bin/clang++-22 CUDA_ROOT=/tmp/llmp-toolchain-smoke/sdk-134/usr/local/cuda-13.4 bash /tmp/llmp-clang22/source/build.sh spark /tmp/llmp-clang22/native'
   ssh spark '/tmp/llmp-clang22/native/cpu-smoke'
   ssh spark 'LD_LIBRARY_PATH=/tmp/llmp-toolchain-smoke/sdk-134/usr/local/cuda-13.4/targets/sbsa-linux/lib CUDA_DISABLE_PTX_JIT=1 /tmp/llmp-clang22/native/cuda-smoke'
   ```

## Results and limits

Every CPU invocation returned 0 and printed:

```text
CPU PASS: C++23, pointer_bits=64, libstdc++=20240904
```

Both selected GPU executables returned 0 and additionally printed:

```text
CUDA PASS: NVIDIA GB10, capability=12.1, checked=257
```

The CPU test checks C++23 `std::byteswap` and `std::to_underlying` at compile
time and a vector/span sum at runtime. The CUDA C++23 profile asserts
`__cplusplus >= 202302L` and exercises `if consteval` in a host/device
function, with different compile-time and runtime results. The kernel launches
two blocks with an out-of-range tail, synchronizes, copies back, and checks
all 257 integers. Both GPU runs disabled PTX JIT. Runtime API allocation here
tests compiler integration, not the future runtime's memory provider or VMM.
This subset does not establish support for every C++23 library feature.

### Earlier Clang 18 / CUDA 13.0 comparisons

The initial Clang/LLD 18.1.3 (`1:18.1.3-1ubuntu1`) combination also
passed with CUDA 13.4.2 and C++23 throughout. Its packages are retained
as `baseline_llvm_packages`. The selected Clang 22.1.8 follow-up reran
all three paths using the same CUDA and target-library inputs.

Before selecting 13.4.2, NVCC V13.0.88 / CRT / libNVVM / libnvptxcompiler
`13.0.88-1`, CCCL `13.0.85-1`, Spark cudart/runtime headers `13.0.96-1`
passed the original smoke on both the cross and native Spark paths using
C++20 CUDA with C++23 host files. The x86 compiler extraction had cudart
`13.0.88-1`; ARM runtime files came from Spark. The NVCC C++23 probe failed
with `Value 'c++23' is not defined for option 'std'` (RE-001).
This comparison is **not the selected pin**. To run it with the retained
harness, set `CUDA_STD=c++20` and point `CUDA_ROOT` at that older SDK;
the C++23-specific CUDA assertion/feature probe is then omitted explicitly.
Its downloaded packages and target subset hash remain in `artifacts.json`.

Not run in this original CPU/CUDA smoke: `spark-b`, workstation GPU execution,
VMM/staging/concurrency tests, model kernels, full backend builds, sanitizers,
clean-host/container setup, or license-profile CI. The later CPU-only sanitizer
check is documented below. These are outside this smoke's scope. No performance
numbers or redistribution permissions are inferred. LLVM and NVCC are build
tools; glibc/libstdc++/libgcc and CUDA execution components are platform
runtimes under D-017. Their full license/provenance audit and SDK provisioning
belong to M1; no third-party implementation is vendored here.

### Compiler-rt follow-up (2026-09-22)

The minimal compiler extraction omitted `libclang-rt-22-dev`; Clang could
compile ordinary code but could not link an ASan executable. Added the amd64
and arm64 packages at the **same D-032 LLVM version**, authenticated by the
existing signed repository indexes and verified package hashes. `packages`
and `sanitizer_followup` in `artifacts.json` record these inputs. There was no
compiler incompatibility or permission restriction. For this SDK-based setup,
extract the package into the matching SDK root so its files populate the
compiler's resource directory. Installing it only under the system compiler
path would not complete that extraction.

Both target SDK directories must be present on the workstation for sanitized
cross builds. Select the ARM SDK's `usr/lib/llvm-22/lib/clang/22` with
`-resource-dir` when cross-linking; retain the x86 compiler and pinned ARM
sysroot. The [async-model suite](../async-model/README.md#reproduction) passed
with Clang ASan+UBSan on the workstation and as a cross-built CPU executable
on `spark-c4e2`. This validates these CPU tests, not CUDA instrumentation,
threading, native Spark compilation or other sanitizer modes.

Under D-017 the linked sanitizer archives are compiler support **platform
dependencies for instrumented tests**, not optional implementation modules or
a production dependency. The package copyright file declares
`APACHE-2-LLVM-EXCEPTIONS` (Apache-2.0 with the LLVM exception), with separate
file-group declarations for bundled components. Its path and SHA-256 are
recorded for both packages; it is not a complete SDK redistribution audit.
Downloaded packages and instrumented executables remain outside Git. M1
provisioning must include the matching runtime packages and sanitizer test
profile.

## Handoff

Builder: the checked-in harness, including the C++23 CUDA feature probe,
passed on the workstation and `spark`. The exact versions and limits are
above. Shell syntax, artifact JSON parsing, ELF inspection, and whitespace
checks accompany the execution checks. No application code or M1 build
conventions beyond the measured pins were introduced. Changes remain
uncommitted for human review.

Initial LLVM 18 reviewer: reviewed the whole uncommitted diff and retained sources; no
actionable defects found. Independently rebuilt and ran native workstation,
cross-to-Spark, and native Spark CPU/CUDA profiles using the existing scratch
SDKs; both GPU runs passed with `CUDA_DISABLE_PTX_JIT=1`. Invalid profile
arguments returned 2, and hiding all CUDA devices made the GPU smoke return
1 without reporting CUDA success. Shell syntax, JSON structure, ELF target
and dynamic-library inspection (including absence of SDK rpath), and
whitespace checks passed. Challenged the pin/provenance and compatibility
claims against the manifest, source, execution, and NVIDIA's linked
minor-version guidance. This review did not independently provision a clean
SDK or extend the test scope beyond the limits above.

LLVM 22 follow-up reviewer: reviewed the entire uncommitted diff, including
the compiler comparison and manifest; no actionable defects found.
Independently rebuilt and ran native workstation, cross-to-Spark, and
native Spark profiles with Clang 22.1.8 and the existing SDK inputs. Both
GPU runs passed with PTX JIT disabled; hiding CUDA devices correctly
returned failure. Reverified the LLVM InRelease signature, signed package
index hashes, and manifest package identities. Checked the compiler ceiling
and release claims against NVIDIA, LLVM, and GCC's linked official sources,
and the extracted CUDA host-compiler guard. Shell syntax, JSON parsing, and
whitespace checks passed. Clean provisioning, GCC execution/performance
comparison, and the other scope exclusions above remain untested.
