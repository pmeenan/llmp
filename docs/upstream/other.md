<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Other projects

Projects with a small number of findings each. Reporting and version-check
status are recorded per entry below.

## Linux CIFS / Synology SMB: archive lock exclusion is inconsistent (RE-054)

- **Status:** unlocalized; no upstream report until the responsible
  client/server behavior has been isolated.
- **Environment:** owner workstation and Spark, the same Synology
  CIFS 3.1.1 share with `cache=strict`, no `nobrl`, 2026-10-10.
- **Problem:** an initial cross-host `flock` test excluded the second
  client, but a repeat against a running writer allowed acquisition on
  both clients. Other contenders stalled in `open`. Kernel lock records
  alone do not establish server exclusion. This is an observation, not
  an attribution to Linux or Synology.
- **Workaround:** server-atomic directory ownership, a host-local recovery
  mutex, and recovery only after proof of local owner death (D-115).
- **Next investigation:** isolate lease/oplock handling on a separate
  mount; preserve an exact client/server reproduction before reporting.
- **Links:** [RE-054](../rough-edges.md#re-054-nas-smb-file-locks-do-not-reliably-exclude-a-second-archive-host--2026-10-10-status-worked-around),
  [archive writer protocol](../model-archive.md#writing-a-revision).

## MLX: mx.quantize's documented affine formula is not what it computes

- **Status:** open (a documentation fix; nothing reported).
- **Project:** [ml-explore/mlx](https://github.com/ml-explore/mlx), `main`
  as fetched 2026-10-09 (commit not recorded); found reading Qwen3.8 Flash
  Next's MLX 4-bit checkpoint (`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`
  at `dadefa80`) on `spark-b` with llmpalooza's importer.
- **Problem:** the `mlx.core.quantize` documentation gives affine mode as
  α = max wᵢ, β = min wᵢ, s = (α − β) / (2^b − 1), ŵᵢ = round((wᵢ − β) / s),
  which puts every dequantized value within s / 2 of the original. The CPU
  kernel (`mlx/backend/cpu/quantized.cpp`, `quantize<T, U>`) keeps zero
  exactly representable instead: it takes `edge` = the group's min (or
  max, by sign), `q0 = rint(edge / scale)`, and when q0 ≠ 0 sets
  `scale = edge / q0` and `bias = edge`; it then clamps each code to
  [0, 2^b − 1]. The rescaled step can leave the group's far end up to one
  step short, clamped there, so a value can be up to one step s (not
  s / 2) from the original. Against Qwen3.8's native BF16 head and token
  table, the MLX checkpoint's values were within one step everywhere
  checked, at most 0.918 of that bound (`import_models_test`,
  `ImportMlxModelsTest.DecodesLikeTheNativeCheckpoint`), which s / 2
  would not hold.
- **Llmpalooza's workaround:** none needed: the importer reads the stored
  scales and biases, and its reference decodes s · q + b, MLX's
  dequantization, whatever quantization produced them. Its tests bound
  MLX's values against other quantizations by one step.
- **Upstream current code:** as above (the kernel and the documentation
  both checked 2026-10-09 on `main`); the Metal and CUDA kernels were not
  checked.
- **Proposed action:** a documentation change to `quantize`'s docstring
  (`python/src/ops.cpp`): state the zero-preserving adjustment and that
  the error is at most one step. Small.
- **Links:** [engine-components.md](../engine-components.md) (the affine
  encoding; the MLX reader as built).

## LLVM: CUDA lint parsing needs an unused cuRAND header (RE-042)

- **Status:** open (SDK/parser compatibility to track).
- **Project:** [llvm/llvm-project](https://github.com/llvm/llvm-project),
  Clang22.1.8 in llmpalooza SDK `aarch64-e0a0c85c42806fb1`, Spark A, 2026-10-02.
- **Problem:** `clang/lib/Headers/__clang_cuda_runtime_wrapper.h:505`
  unconditionally includes `curand_mtgp32_kernel.h`. A CUDA host parsing
  check fails against the trimmed CUDA13.4.92 SDK because that header is
  absent, even for a kernel with no cuRAND calls. The earlier NVCC-only
  compile-entry driver flags must also be adapted before Clang can parse.
- **Llmpalooza's workaround:** retain the actual compile entry's definitions
  and SDK headers, adapt only parser driver flags, and add the installed
  `/usr/local/cuda-13.0/include` after SDK paths for lint only. The successful
  check records both original and adapted entries. Production NVCC flags,
  SDK libraries and numerical behavior remain unchanged.
- **Upstream current code:** not checked; no defect or fix is established.
- **Proposed action:** before changing the SDK, check the current wrapper
  and whether the required cuRAND headers should be provisioned for Clang
  CUDA parsing. This observation does not justify a production compiler pin.
- **Links:** [RE-042](../rough-edges.md#re-042-clangs-cuda-parser-requires-a-curand-header-absent-from-the-trimmed-sdk--2026-10-02-status-worked-around),
  [qualification report](../experiments/dsv4-qhead/README.md).

## Docker: a multi-arch index digest can run the wrong architecture from the local store (RE-015)

- **Status:** open.
- **Project:** Docker Engine ([moby/moby](https://github.com/moby/moby)) with
  the containerd image store.
- **Found:** 2026-09-23, the workstation (x86-64), Docker 29.8.1, qemu binfmt
  registered.
- **Problem:** `docker run ubuntu:24.04@sha256:008173c2…` (the multi-platform
  index digest) ran the **arm64** image under qemu-user, with only a
  platform-mismatch warning, because an arm64 `ubuntu:24.04` pulled earlier
  was the local content for that digest. With binfmt registered nothing
  fails, so builds silently run emulated. Repro: pull the arm64 variant by
  index digest with `--platform linux/arm64`, then `docker run` the same
  digest without `--platform` on an amd64 host with binfmt registered.
- **Llmpalooza's workaround:** every `FROM`, `docker run` and `docker build` of
  the reference container names `--platform linux/amd64`, and `doctor`
  reports the architecture inside the container. No cost.
- **Proposed action:** check a current Docker; if it still happens, an issue
  asking that an index digest resolve to the host's platform, or fail,
  rather than to whatever the store holds. Small.
- **Links:** RE-015 in [rough-edges.md](../rough-edges.md).

## SGLang: a failed processor import reports "No processor registered" (RE-012)

- **Status:** open.
- **Project:** [sgl-project/sglang](https://github.com/sgl-project/sglang).
- **Found:** 2026-09-22, `lmsysorg/sglang` nightly `0f6761b5` (arm64 digest
  `9e1fb4c3…`), both Sparks, MiMo-V2.6-Flash-RL.
- **Problem:** startup aborted with `No processor registered for
  architecture: ['MiMoV2ForCausalLM']`, even for text-only use. The real
  cause was that `multimodal/processors/mimo_v2.py` imports `torchcodec`,
  which the image lacks; processor discovery logs and skips modules that
  fail to import, so the root cause is lost.
- **Llmpalooza's workaround:** a derived image that adds hash-pinned `torchcodec`
  0.16.0 ([mimo-reference](../experiments/mimo-reference/README.md)).
- **Proposed action:** an issue or small PR: carry the import error into the
  "not registered" message, or add `torchcodec` to the image. Small.
- **Links:** RE-012 in [rough-edges.md](../rough-edges.md).

## Emulated ASan death tests spin during fatal-signal collection (RE-062)

- **Status:** unlocalized; no upstream report until QEMU guest-core handling
  and host Apport collection have been separated.
- **Projects:** [QEMU](https://gitlab.com/qemu-project/qemu), LLVM
  compiler-rt and Ubuntu Apport. Workstation Ubuntu 26.04.1,
  kernel 7.0.0-42-generic, qemu-user 1:10.2.1+ds-1ubuntu3.3,
  LLVM 22.1.8 / GCC 16.2 static runtime, GoogleTest 1.18.0.
- **Reproduction:** build the `cross-asan` preset's `gtest_contract_test`
  with the pinned SDK; run its `DeathTest.*` cases through qemu-aarch64
  with `ASAN_OPTIONS=detect_leaks=0`, without the non-dumpable test main.
  Ordinary cases and sanitizer controls pass; death-test children reach
  `tgkill(SIGABRT)` and QEMU's fatal-signal message, then spin for over
  90 s. The host has Apport's piped `core_pattern`. Neither forcing GTest
  to fork nor toggling ASan's `disable_coredump` option fixed it.
- **Source checks:** [QEMU 10.2.1 fatal-signal handling](https://raw.githubusercontent.com/qemu/qemu/v10.2.1/linux-user/signal.c)
  distinguishes guest dumping and raising a host fatal signal. Its
  diagnostic does not prove a guest core was written. Linux documents
  [piped core-handler behavior](https://docs.kernel.org/admin-guide/sysctl/kernel.html#core-pattern),
  including the different handling of core-size limits.
- **Llmpalooza's fix:** its test main marks every test process
  non-dumpable after initialization, including threadsafe re-executed
  children. All three death cases then pass in about 0.35 s each; assertions
  and restarts are retained. Startup-failure controls cover death and
  successful-exit predicates. No upstream source patch or version change.
- **Next investigation:** reduce to a standalone ASan AArch64 abort under
  the same QEMU/kernel/Apport setup, inspect host fatal-signal collection,
  then check upstream trackers before assigning blame or reporting.
- **Links:** [RE-062](../rough-edges.md#re-062-qemuasan-death-tests-spin-during-fatal-signal-collection-after-the-ubuntu-26-upgrade--2026-10-28-status-worked-around).

## LeakSanitizer aborts AArch64 programs under qemu-user (RE-014)

- **Status:** open (a limitation to track).
- **Project:** LLVM compiler-rt's LeakSanitizer, or qemu-user
  ([qemu/qemu](https://gitlab.com/qemu-project/qemu)); which side is at fault
  was not established.
- **Found:** 2026-09-23, the workstation, Ubuntu `qemu-user-static`
  1:8.2.2+ds-0ubuntu1.18, Clang 22.1.8 with `-fsanitize=address,undefined`,
  arm64 compiler-rt from the SDK.
- **Problem:** every test passes, then at exit LeakSanitizer reports "has
  encountered a fatal error" and the process exits 1.
  `ASAN_OPTIONS=detect_leaks=0` makes it exit 0, and ASan still works.
- **Llmpalooza's workaround:** leak detection is off for emulated AArch64 runs
  (D-061); LSan runs natively on x86-64 and on the Sparks.
- **Proposed action:** none unless it starts to matter; then search both
  trackers for an existing report before filing one.
- **Links:** RE-014 in [rough-edges.md](../rough-edges.md).

## Ubuntu's snapshot service has no ports archive (RE-016)

- **Status:** open (a limitation to track).
- **Project:** Canonical's `snapshot.ubuntu.com`.
- **Found:** 2026-09-24.
- **Problem:** `https://snapshot.ubuntu.com/ubuntu-ports/<timestamp>/` answers
  HTTP 401, so arm64 packages cannot be pinned by date; with
  `APT::Snapshot` set, `apt-get update` in an arm64 container still fetches
  the live ports indexes.
- **Llmpalooza's workaround:** the arm64 install-test image takes systemd from
  the live ports archive and prints the version. The SDK pins its arm64
  `.deb`s by URL and SHA-256 (D-070), so it is unaffected.
- **Proposed action:** check whether ports snapshots exist now; if not,
  optionally ask Canonical (Launchpad or Discourse). Low priority.
- **Links:** RE-016 in [rough-edges.md](../rough-edges.md).
