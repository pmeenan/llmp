<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Compiler selection follow-up — 2026-09-21

Retain D-010's Clang-first direction; update the uncommitted D-032 compiler
pin to **Clang/LLD 22.1.8**, with NVCC 13.4.92 from Toolkit 13.4.2. The
same C++23 smoke passed native workstation, cross-to-Spark, and native Spark
paths. This replaces the initial LLVM 18 selection within the M0 smoke.

## Why this version

[LLVM's release list](https://github.com/llvm/llvm-project/releases) shows
23.1.1 as current upstream and
[22.1.8](https://github.com/llvm/llvm-project/releases/tag/llvmorg-22.1.8)
as the newest 22.x release. CUDA 13.4's
[host compiler table](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#host-compiler-support-policy)
supports Clang 7–22 and GCC 6–16 on both x86-64 and ARM SBSA. Its local
`crt/host_config.h` rejects `__clang_major__ >= 23`. We use 22.1.8 rather
than bypassing that check or maintaining two primary compiler versions.
This is the newest compatible release as checked on the date above, not a
floating dependency policy.

The tested binaries came from apt.llvm.org's Noble 22 branch, package version
`1:22.1.8~++20260714014902+ca7933e47d3a-1~exp1~20260714135019.80` on
both architectures. Its source revision `ca7933e47d3a` matches the upstream
22.1.8 release commit. The exact package hashes, URLs, and signed-index
identity are retained in `artifacts.json`. apt.llvm.org calls its service
nightly packages; the selected bytes are pinned to this release revision,
not an unversioned nightly. A newly needed compiler dependency, Ubuntu
`libz3-4` `4.8.12-3.1build1`, was extracted alongside LLVM on both hosts.
It is a compiler dependency, not linked into the smoke executables.

## Clang versus GCC

Both are current, mature C++ compilers. GCC is not an obsolete alternative:
[GCC 16.2](https://gcc.gnu.org/) is the current release and its major version
is inside CUDA's supported range. Their published
[Clang](https://clang.llvm.org/cxx_status.html) and
[GCC](https://gcc.gnu.org/projects/cxx-status.html) C++ feature tables must be
checked for specific language features; neither project's age nor the flag
`-std=c++23` implies complete language/library coverage.

| Project need | Assessment |
| --- | --- |
| x86 workstation building AArch64 binaries | Clang's single compiler can select targets using `--target`, with explicit sysroot/library paths. This is already verified in our harness. GCC supports cross compilation too, using a compiler built for the target; it is a viable alternative, not a capability gap. |
| Editor and static-analysis tooling | LLVM supplies clangd and clang-tidy. Keeping the primary compiler in that family simplifies matching language parsing and diagnostics. These tools can also be used with a GCC-built project. Tool pins and CI still belong to M1. |
| CUDA | Both Clang 22 and GCC 16 are supported NVCC host compilers. Clang 22.1.8 has now passed our actual cross/native GPU tests. NVCC still compiles device code; choosing GCC for the host would not replace NVCC's GPU compiler. |
| C++ standard library and target ABI | We use GCC's libstdc++ with Clang. Compiler, C++ library headers, and shared runtime have separate identities. Updating Clang does not update the GCC 13 library headers or prove complete C++23 library support. Keep the measured target ABI; revisit library pins for a concrete missing feature. |
| Generated code speed and build time | This smoke measures neither. No claim that Clang is universally faster, more correct, or produces better inference performance than GCC is justified here. Compare real runtime/backend workloads if performance becomes a reason to switch. |

Sources for the workflow assessments:
[Clang cross compilation](https://clang.llvm.org/docs/CrossCompilation.html),
[Clang toolchain/library selection](https://clang.llvm.org/docs/Toolchain.html),
[clangd](https://clangd.llvm.org/),
[clang-tidy](https://clang.llvm.org/extra/clang-tidy/), and
[NVCC's compilation model](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).

## Other choices and recommendation

There is no measured need to introduce a different host compiler family,
such as NVIDIA HPC SDK's NVC++, or replace NVCC with Clang's CUDA frontend.
Those would be separate integration paths with their own validation burden;
they were not tested here. Existing backend-specific GCC exceptions remain
allowed by D-010 when an actual integration requires them.

Use **Clang/LLD 22.1.8 + libstdc++ + NVCC 13.4.92** as the initial
combination. Consider a GCC CPU-only CI job in M1 as an independent compiler
check, without making GCC a second primary toolchain today. We have not
benchmarked or smoke-tested GCC 16.2, so this recommendation rests on the
verified workflow and tooling fit, not a head-to-head compiler ranking.

## Verification

The retained `main.cc`, `kernel.cu`, and `build.sh` passed using the new
compiler on the workstation and `spark`. CPU output and the 257-value GPU
checks match the main report; both GPU executables passed with
`CUDA_DISABLE_PTX_JIT=1`. No `--allow-unsupported-compiler` was used.
System compiler defaults, target system libraries, and drivers are unchanged.
The original CUDA 13.0 and Clang 18 runs remain historical comparisons.

Build directories: workstation `/tmp/llmp-clang22/{native,cross}`;
Spark `/tmp/llmp-clang22/{native,cross}`. The compiler SDKs are in
`/tmp/llmp-clang22/sdk-amd64` (workstation) and
`/tmp/llmp-clang22/sdk-arm64` (Spark); CUDA and sysroot inputs remain
under `/tmp/llmp-toolchain-smoke`. Clean provisioning, full backend
compilation, compiler-performance comparisons, and additional C++23 library
features remain untested. This follow-up changes no runtime code.
