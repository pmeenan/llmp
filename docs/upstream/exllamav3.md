<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ExLlamaV3

- **Repository:** [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3),
  MIT.
- **Llmpalooza's pin:** `6b84a21b6f1e5da3f291b9e1019061f0de788279` (one commit
  after v1.5.1, 2026-09-20; [sources.lock.json](../../third_party/sources.lock.json)).
  Llmpalooza compiles only the kernels its native EXL3 linear launches, from its
  own launchers (`src/kernels/exl3/`, D-053, D-080), and links them into tests
  and benchmarks only. Its changes are in
  [third_party/patches/exllamav3/](../../third_party/patches/exllamav3/).
- **Last upstream check:** 2026-09-29, master `d3739fd393337b1ff4d6c2a342b12f0c87a9592f`
  ("Bump to v1.5.3", 2026-09-27), by reading the files named below; nothing
  was built.

## The `register` storage class breaks NVCC with a Clang host compiler

- **Status:** open.
- **Found:** 2026-09 (M2), pin `6b84a21`, NVCC 13.4.92 with Clang 22 as host
  compiler.
- **Problem:** `register` is ill-formed since C++17. NVCC's front end rejects
  it with a Clang host compiler; upstream builds with GCC, which only warns.
  It has no effect on code generation. Four local arrays in
  `exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh` (`frag_a`, `frag_b`,
  `frag_c`, `frag_c_h`, about line 217) and two in `quant/reconstruct.cu`
  (`FragB frag[2]`) carry it.
- **Llmpalooza's workaround:** `0001-llmp-adaptations.patch` and
  `0003-llmp-kernel-split.patch` drop the keyword. The kernels' SASS is
  unchanged (checked against the reference build's hashes,
  [licensing.md](../licensing.md)). Cost: two small hunks to rebase.
- **Upstream master:** all four in `exl3_gemm_inner.cuh` are still there at
  `d3739fd`; `reconstruct.cu` not checked.
- **Upstream refs:** none found.
- **Proposed action:** a tiny PR removing `register` from those arrays
  (about 6 lines), noting the Clang-host failure and the unchanged SASS.
  Check the project's contribution guidance on AI-written code first.
- **Links:** the patches above.

## Process-exiting error helpers and ATen wrappers share files with the kernels

- **Status:** carry.
- **Found:** 2026-09 (M2), pin `6b84a21`.
- **Problem (for llmpalooza):** `util.cuh`'s `cuda_check`, `gpu_assert`,
  `cublas_check` and `cublas_assert` print and call `exit()`. And
  `quant/reconstruct.cu`, `quant/hadamard.cu` and `add.cu` hold both the
  kernels and their ATen host wrappers, which need PyTorch headers.
  Llmpalooza owns dispatch and returns errors as values (D-066), so it removes
  the helpers (0001) and the wrappers and their includes (0003), and
  instantiates only the kernels it launches (0002 adds its CMake and
  `llmp/llmp_exl3_kernels.cu`).
- **Upstream master:** not checked.
- **Proposed action:** none by default: this is upstream's design, not a
  bug. If the owner wants a smaller patch 0003, a PR moving each file's
  kernels into a header of their own, without ATen includes, would do it.
  Medium effort; upstream may not want it.

## x86-only CPU helpers fail to compile on AArch64 (RE-009)

- **Status:** fixed upstream at `d3739fd` (read, not built).
- **Found:** 2026-09-22, pin `6b84a21`, a Spark, GCC 13.3, CUDA 13.0.88,
  PyTorch 2.14.0+cu130.
- **Problem:** importing the extension builds every unit, including
  `avx512_target.cpp` (`__builtin_cpu_supports`) and host spin waits using
  `__builtin_ia32_pause`, so the unmodified build fails on ARM.
- **Llmpalooza's workaround:** only the external reference build carries
  [arm-reference.patch](../experiments/exl3-reference/arm-reference.patch).
  Llmpalooza's own build never compiles those files.
- **Upstream master:** at `d3739fd`, `avx512_target.cpp` guards the probe
  with an x86 check and `cpu/moe_handoff.cu` uses `yield` on AArch64. The
  other files in the reference patch (`avx2_target.cpp`, the CPU MoE and the
  CPU all-reduce) were not read.
- **Proposed action:** none upstream. When the EXL3 reference is next
  rebuilt, try it at a pin past `d3739fd` without the patch.
- **Links:** RE-009 in [rough-edges.md](../rough-edges.md);
  [exl3-reference](../experiments/exl3-reference/README.md).
