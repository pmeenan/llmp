<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Other projects

Projects with a single finding each. None has been reported, and none of
their current versions was checked.

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
