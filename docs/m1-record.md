<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M1 record — Bootstrap

*Milestones renumbered (D-087, 2026-09-27): numbers here are those of
their date; the 2026-09-23 ladder's M3–M8 are now M5–M10, and M4a is M6a.*

M1 ran from 2026-09-23 to 2026-09-24 and exited on the owner's word on
2026-09-24. This is its task-by-task record, moved out of [plan.md](plan.md)
at exit so the plan stays lean: what each item built, how it was verified
and on which hosts, and what it handed to later milestones.

**This is frozen history.** The living documents supersede it where they
differ: [plan.md](plan.md) owns what M1 handed on,
[decisions.md](decisions.md) the settled choices (D-070 to D-074 are M1's),
and [README.md](../README.md) and [architecture.md](architecture.md) how the
repository works now. Later corrections go in those documents, not here.

Goal: a reproducible, pinned developer setup and repository skeleton that
every later milestone builds, tests and packages on, with D-061's local check
gate in force and the first `llmp` binary running on a Spark.

**Entry:** M0 exited on 2026-09-23.

**Scope:**

- [x] **SDK provisioning** (D-012, D-049): `mise.toml`/`mise.lock`,
      `toolchains/manifest.toml`, `toolchains/artifacts.lock.json`,
      `tools/setup-toolchain`, `tools/check-toolchain`, and the
      digest-pinned reference container (`.devcontainer/`) built from the
      same logic. A persistent, versioned SDK outside the checkout and `/tmp`
      holds every profile's tools at their pins: Clang/LLD and the LLVM
      formatter, linter, language server and symbolizer 22.1.8 with matching
      compiler-rt sanitizer runtimes (D-032, D-059); CUDA Toolkit 13.4.2
      (NVCC 13.4.92); CMake 4.4.3 (D-058); Ninja 1.13.2; and the source-built
      GCC 16.2 runtime for each architecture, the AArch64 copy inside the
      target sysroot (D-060). Host prerequisites stay system-managed and are
      declared and checked, `qemu-user-static` with binfmt among them. The
      mise tasks are `setup`, `doctor`, `build`, `test` and `deploy`.
      *Landed (D-070):* the manifest, lock, prerequisite lists, `setup` and
      `doctor` tasks and reference container, with the sysroot built from
      pinned Ubuntu packages and the AArch64 GCC runtime cross-built. Setup
      passed on the workstation, on `spark` and in a clean reference
      container. `build`, `test` and `deploy` landed with the Build item.
- [x] **Build** (D-010, D-011): `CMakePresets.json` and `cmake/toolchains/`
      for native x86-64, the AArch64 cross build (CPU, and CUDA for
      `sm_121`), the native-Spark fallback, and a CPU-only configuration
      with no CUDA toolkit visible (D-026). C++23 for `.cc` and `.cu`;
      static libstdc++, libgcc and cudart (D-060); `compile_commands.json`.
      At the repository root: `.clang-format`, `.clang-tidy` (findings are
      errors), D-059's warning set with `-Werror`,
      `CMAKE_CXX_SCAN_FOR_MODULES OFF` and `-fno-exceptions`, tests
      included (D-066).
      *Landed:* presets `native`, `cpu`, `cross` and `spark-native`, whose
      toolchain files use the SDK (plus the host GNU linker on Spark) and
      check its receipt against the checkout; `tools/build` behind
      `mise run build`, `test` and `deploy`; and `tools/run-target`, which
      runs cross-built tests under qemu-user or, with `--host`, on a Spark over SSH. The
      `tests/toolchain/` contract tests (C++23 with the GCC 16.2 library, no
      exceptions and an aborting throw path, the explicit CPU baseline,
      glibc-only dynamic dependencies, no symbol version above 2.39 for Spark
      binaries, no RPATH, and a CPU-only build that never sees CUDA) passed
      in every preset, cross under qemu-user and on `spark`; there the sm_121
      CUDA test ran cross-built and natively built. NVCC's host pass uses
      D-059's warnings less `-Wold-style-cast`, which CUDA's own headers trip. GoogleTest under
      `-fno-exceptions` is proven with the Source dependencies item.
- [x] **Source dependencies** (D-057): the source lock, a preparation step
      separate from SDK setup, lock validation and the build receipt, proven
      on GoogleTest 1.18.0 with gMock through every
      [M1 gate](source-dependencies.md#upgrades-packaging-and-m1-gates).
      Kernel closures (the GGML subset and the ExLlamaV3 files) enter
      through this mechanism at M2's P0 stage, not here.
      *Landed:* [third_party/sources.lock.json](../third_party/README.md) with
      GoogleTest 1.18.0 (BSD-3-Clause, core, test-only, every option locked);
      `mise run prepare` (tools/prepare-sources, also run by `setup`), which
      validates the lock, selects the profile's closure before fetching,
      refuses archives with links or special files, and unpacks through
      FetchContent's script mode with exact patches and a tree digest; and
      `cmake/LlmpSources.cmake`, which validates the lock again, checks each
      tree before any third-party CMake runs and on every build, rejects
      unrecorded overrides, dependency providers, declared FetchContent
      population, reserved option names and undeclared `find_package()`
      lookups, refuses a profile without an optional module in a directory
      that built it, and writes `llmp-receipt.json`. GoogleTest and gMock
      build and pass with `-fno-exceptions` (D-066); `sources.closure` checks
      the compile and link inputs Ninja recorded against the receipt, accepts
      build-tree files only from current build rules and selected components,
      and checks every C++ compile's exception flags and every compiled or
      linked object outside the SDK for exception support, and
      `sources.mechanism` runs the gates on a synthetic lock with an optional
      module. *Verified 2026-09-24:* `native`, `cpu` and `cross` (qemu-user)
      on the workstation and offline in the reference container (`--network
      none`) after preparation from an empty cache; `cross` on `spark` over
      SSH and `spark-native` on `spark`. Owned elsewhere: the package half of
      the last gate (package inventory, NOTICE and SBOM) with the Package
      item; running the network-denied build, the only
      stop for downloads a component's own scripts attempt, as part of
      `check:full` with the Local check gate; vendored units with M2's first
      adapted kernel; build-host tools, which the lock refuses until the first
      generator needs them.
- [x] **Local check gate** (D-061): the `check`, `check:full` and
      `check:spark` tasks with D-061's contents. Benchmarks join the gate as
      regressions when their harnesses and baselines exist (the EXL3 kernel
      and parity benchmarks from M2/M3, trace replay from M4/M5); thresholds
      follow measurement.
      *Landed:* `tools/check` behind `mise run check`, `check:full` and
      `check:spark -- --host <spark>`; coverage and usage are in
      [README.md](../README.md#checks). Every tier reports each step's result
      with the host, commit and SDK. Check builds use the core profile from
      locked sources, configure afresh and ignore the caller's compiler,
      CMake, test and sanitizer environment. Formatting covers tracked and
      new owned sources; clang-tidy checks C++ units once per architecture,
      while NVCC's `.cu` units are formatted only. Sanitizer builds include
      GoogleTest and require deliberate defects to produce sanitizer reports;
      leak detection is off only under qemu-user (RE-014). The reference
      build checks the core closure from an empty source download cache and
      runs with no network; its provisioned SDK persists separately.
      *Verified 2026-09-24:* `check:full` (including `check`) on the workstation; `check:spark` on `spark` (GB10, driver 580.178.04).
      REUSE lint and the embedded-header check joined `check` with License
      and provenance. *Owned elsewhere:* the `.deb`, its install test and
      the package inventory against the receipt, NOTICE and SBOM join
      `check:full` with the Package item. `llmp doctor` joined
      `check:spark` with the Smoke binary item. The GPU, VMM, I/O and ARM
      stress suites join it in M2.
- [x] **License and provenance** (D-017, D-029): `LICENSES/`, REUSE
      metadata and lint, the embedded-header check, a root `NOTICE`, and an
      SBOM generated with the package. Audit the notices of what ships and
      what builds it: CMake, Ninja, LLVM, glibc, the CUDA runtime, and the
      static GCC runtime with its embedded components.
      *Landed (D-071):* `LICENSES/` (Apache-2.0, MIT), SPDX metadata on
      every file (headers where the format allows, `.license` sidecars
      otherwise, no `REUSE.toml`) and the root `NOTICE`. `check` gains
      `reuse`, REUSE lint 6.2.0 from hash-pinned wheels in the x86-64 SDK,
      and `headers` (`tools/llmp_headers.py`). `headers` checks REUSE's
      JSON report against each file's own header or sidecar. It fails on a
      commentable file without its own header, on metadata REUSE takes
      from anywhere else, on a misplaced or orphaned sidecar, on a file REUSE
      skips, and on an unclassified file type. The audit is in
      [licensing.md](licensing.md#what-builds-llmpalooza), with every locked SDK
      artifact, host prerequisite and mise tool recorded in
      `toolchains/provenance.toml` (category, license, what enters a binary,
      the notices that follow), which a test keeps complete. It corrects
      D-060's list of embedded runtime code. *Verified 2026-09-24:*
      `check:full` on the workstation, including the offline reference build
      with the new SDK. *Moved to the Package item:* the SBOM and the
      package's third-party notices, since no binary ships before it.
- [x] **Versioning** (D-062): `project(VERSION)`, the dev-version
      derivation and its Debian `~` mapping, `llmp --version` and the
      receipt (version, commit, license profile, SDK identity), a root
      `CHANGELOG.md`, and the version of D-047's opaque reasoning-signature
      representation.
      *Landed:* `project(VERSION 0.1.0)` and `cmake/LlmpVersion.cmake`,
      which derives the version at configure and again on every build
      (`cmake/version/`), so a commit needs no fresh configure; the rules
      and their refusals are in [README.md](../README.md#versions). Release
      tags are annotated `vX.Y.Z` tags HEAD contains; a tree without Git
      metadata, such as the reference build's copy, reports
      `X.Y.Z-dev+unknown`, and a shallow clone is refused. The receipt
      gains a `version` object (product and Debian versions, commit,
      modified state); the first `llmp` binary (`src/cli/`) has
      `--version` and `--help`; `CHANGELOG.md` is seeded; the reasoning
      signature's representation is version 1
      (`src/base/surface_versions.h`), whose contents M3 fixes. `git`
      joins the AArch64 host prerequisites. Tests: `version.derive` on
      synthetic repositories, `version.llmp` (`--version` against the
      receipt and the checkout now) and the binary's link contract, and
      the CLI unit tests.
- [x] **Smoke binary and capability probe** (D-026): a `llmp` binary,
      from the cross build and from the native Spark fallback, that runs on
      `spark` over SSH, with a first-cut `doctor` reporting VMM granularity,
      GDS mode, RDMA availability, driver and toolkit versions, glibc and
      ABI, the selected tools and host prerequisites, and any data role
      relocated onto a read-only filesystem (D-063).
      *Landed (D-072):* `llmp doctor` reports the build (version, SDK,
      target, compiler, C++ runtime), the host (kernel, glibc, page size,
      memory, `fs.protected_hardlinks`), RDMA ports with this user's access
      to their device nodes, the NVIDIA driver (kernel module, library,
      CUDA API against the build's toolkit and GPU code, `nvidia_fs`), and
      each GPU's compute capability, compute mode, VMM support and
      device-local and host-NUMA granularity. It exits 1 when this host
      cannot run this build: anything but a GB10 with VMM and host-backed
      VMM fails. New modules: `platform` (reads of `/proc` and `/sys`, the
      host probe) and `providers` (the device probe). CUDA builds link the
      driver's `libcuda.so.1`, a documented hard requirement, through
      NVIDIA's stub, which the SDK gains (`cuda-driver-dev-13-4`); tests
      that need no GPU load the stub on hosts without the driver.
      Tests: unit tests on fake `/proc`, `/sys` and `/dev` trees and on
      made-up driver facts; `smoke.doctor` on every host; and
      `smoke.doctor.gpu`, which requires a clean report on a GB10 and so
      runs in `check:spark` and `spark-native`.
      *Owned elsewhere:* the read-only data-role report, with Node
      configuration and the unit's sandboxing (Package); the direct-I/O
      probe of the storage roles, with Node configuration; the
      `libcuda.so.1` version floor, with the Package item. `mise run doctor`
      keeps reporting the SDK and the development host's prerequisites.
- [x] **Package and installed layout** (D-027, D-063): an arm64 `.deb`
      (M1 chooses CPack or debhelper) in the
      [installed layout](architecture.md#installed-layout): the `llmp`
      user; `llmp.service` with sandboxing that keeps GPU and RDMA device
      access, and its restart policy; the per-node process lock; core dumps
      off and a non-dumpable process, checked on each host by an abort that
      leaves no core file or apport report; `/etc/llmp` and the `/var/lib/llmp` roles
      with their modes. It depends on `libc6` (`GLIBC_2.38`) and a versioned
      `libcuda.so.1` floored at NVIDIA's minimum driver for the pinned
      toolkit. The install test runs in an arm64 container; M1 decides
      whether it also starts the unit. The package ships `LICENSE`,
      `NOTICE`, a third-party notices file and an SBOM, generated from the
      build receipt, the SDK receipt and `toolchains/provenance.toml`, and
      is built from the `cross` profile (D-071). The seven owner decisions in
      [licensing.md](licensing.md#what-a-packaged-binary-carries) are
      settled first.
      *Landed (D-074):* CPack writes
      the `.deb` from the `cross` preset `--locked` (`mise run package`);
      `llmp-runtime` (`src/runtime/`) runs the startup order to readiness
      with the per-node lock at `<anchor>.lock`; crashes exit through a
      signal handler instead of dumping; `llmp.service` with its restart
      policy, delegation and sandboxing; the sysusers and tmpfiles files;
      the dependencies from the binaries (`libcuda.so.1 (>= 580)`); and the
      notices, Debian copyright file and SPDX SBOM generated from the
      receipt, the source lock and `toolchains/provenance.toml`, which a
      check compares with the package. The install test runs in an arm64
      container without starting the unit. `check:full` runs both. The
      owner settled the seven licensing decisions, the maintainer address
      and enabling the service on install (2026-09-24).
      *Verified 2026-09-24:* the package, its inventory and the install test
      in `check:full` on the workstation; on `spark`, installed with the
      owner's approval and purged afterwards, the service reached readiness
      under its sandbox with the GPU and every RDMA node, an abort left no
      core file or apport report, and a reinstall and a stop behaved; then
      `check:spark`.
- [x] **Node configuration** (D-063): a strict TOML 1.0 parser (toml++ is
      the candidate under D-017, D-057 and D-066) and D-063's node-local
      keys, `[storage]` included, validated fail-closed with exhaustive
      diagnostics. Front-door and TLS keys are fixed in M3 and
      switching-policy keys in M4 (D-069).
      *Landed (D-073):* toml++ in the source lock (MIT, core, product) at
      commit `1e8829b`, past v3.4.0 for its input-reachable fixes;
      `src/config/` reads the main file and its drop-ins, refusing files and
      directories other users could change (`platform/path_trust`), merges
      them with one owning file per key, and validates cluster-design.md's
      node-local keys plus `[storage]` and `[limits]`, reporting every
      problem with its file, line and column. `PrepareRuntimeRoles` creates
      and checks the runtime's roles (modes, owners, aliasing by device and
      inode, ext4/XFS/Btrfs only, D-034's direct-I/O probe on `installed`
      and `spill`, the spill marker) for the runtime to call at startup.
      `llmp doctor [--config FILE]` reports the configuration and the
      roles, including a role outside the packaged unit's writable
      `/var/lib/llmp`. The configuration's `schema_version` joins the
      surface versions. *Owned elsewhere:* checks of the credential and
      cluster files on disk, and of enrollment records in `state`, with
      M4a; deleting runtime-named spill files, with M3's spill names.
- [x] **Confined job proof** ([job rules](architecture.md#import-install-and-archive-jobs)):
      choose the mechanism (a delegated cgroup or a subreaper) together with
      the unit's sandboxing; unprivileged `unshare` and `bwrap` are blocked
      on these hosts (RE-013). Prove, under the `llmp` account, containment
      of a child that outlives its job, the job lock inherited across exec,
      and correct handling across a runtime restart, before M3 builds the
      importer on it. Any system change the proof needs on a host is the
      owner's to approve.
      *Landed (D-074):* delegated cgroups (`Delegate=`,
      `DelegateSubgroup=runtime`), a job record lock inherited across exec
      and by nothing else, `cgroup.kill`, the runtime as child subreaper,
      and "ended" meaning an empty cgroup and a free lock
      (`src/platform/job.*`); stages that parse untrusted input confine
      themselves with Landlock and a seccomp filter against sockets
      (`src/platform/confine.*`). `tools/job-proof` runs the proof: a child
      that outlives its job, the lock across exec, no inheritance between
      jobs, a restarted runtime, a unit restart, and a confined stage.
      *Verified 2026-09-24* in the workstation user's systemd manager
      (`check:full` runs it there) and on `spark` as `llmp` with the
      service's sandbox settings (`tools/job-proof --host spark`).
- [x] Update AGENTS.md's repository-layout table as the scaffolding lands.

**Exit criteria:**

- Setup succeeds on a clean workstation host and in the reference container
  from the declared prerequisites alone, with no global compiler or
  environment changes and no dependency on temporary SDK directories.
  `doctor` reports the exact toolchain, driver and SDK identities.
- `check` passes on the workstation, including the CPU-only configuration
  and the AArch64 CPU tests under qemu-user. `check:full` passes from a fresh
  clone: the sanitizer builds, the copyleft-disabled build from empty caches,
  the network-denied container build, D-057's gates, the `.deb` and its
  install test, and the package inventory against the receipt, NOTICE and
  SBOM. `check:spark` runs the CUDA smoke, `doctor`, LeakSanitizer and
  ThreadSanitizer on `spark`. The confined-job proof passes on its chosen
  host, and an abort leaves no core file or apport report on each host.
- Every new dependency or pin records its D-017 category and tier, with a
  decision entry where AGENTS.md rule 1 calls for one.
