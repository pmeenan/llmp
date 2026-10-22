<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# XGrammar source and license audit

This is a source and dependency audit only. No XGrammar code was adopted,
integrated, or added to the source lock. The M3.5 dependency audit covers the
candidate C++ core for M7 constrained output and tool-call triggers.

## Upstream identity and license

The official [mlc-ai/xgrammar repository](https://github.com/mlc-ai/xgrammar)
was checked out at immutable commit
[`614ab01d2af3fdfdde73ae3e4171a7f36db45c23`](https://github.com/mlc-ai/xgrammar/commit/614ab01d2af3fdfdde73ae3e4171a7f36db45c23),
dated 2026-10-18. Its root `LICENSE` is Apache-2.0 and `NOTICE` attributes
XGrammar to its contributors. The pinned C++ target includes vendored
`3rdparty/picojson/picojson.h`; the header's two redistribution conditions
match BSD-2-Clause, including retaining its notice and reproducing it in
binary documentation. The target also includes DLPack submodule commit
[`bbd2f4d32427e548797929af08cfe2a9cbb3cf12`](https://github.com/dmlc/dlpack/tree/bbd2f4d32427e548797929af08cfe2a9cbb3cf12),
whose license is Apache-2.0. These obligations would need to be carried if
this closure is later adopted.

The selected static target uses the root `cpp/*.cc` glob, excluding
`cpp/tvm_ffi/*.cc`, and includes `include/`, PicoJSON and DLPack. Its CMake
language level is C++17. The Python binding path, GoogleTest, and cpptrace
were outside the audited target: Python bindings, C++ tests, and cpptrace
were configured off. Their licenses and other optional paths were not
audited here. This is not a complete dependency or package audit for the
whole XGrammar repository.

Revision-pinned source links: [root build](https://github.com/mlc-ai/xgrammar/blob/614ab01d2af3fdfdde73ae3e4171a7f36db45c23/CMakeLists.txt),
[default build options](https://github.com/mlc-ai/xgrammar/blob/614ab01d2af3fdfdde73ae3e4171a7f36db45c23/cmake/config.cmake),
[submodules](https://github.com/mlc-ai/xgrammar/blob/614ab01d2af3fdfdde73ae3e4171a7f36db45c23/.gitmodules),
[PicoJSON terms](https://github.com/mlc-ai/xgrammar/blob/614ab01d2af3fdfdde73ae3e4171a7f36db45c23/3rdparty/picojson/picojson.h),
and [DLPack license](https://github.com/dmlc/dlpack/blob/bbd2f4d32427e548797929af08cfe2a9cbb3cf12/LICENSE).

## Bounded no-exceptions compile probe

On Spark A (`spark-c4e2`), the exact upstream commit and only its DLPack
submodule were fetched into `/tmp/llmp-xgrammar-noexceptions-probe-r1/`.
All 16 audited upstream file hashes and both DLPack file hashes matched the
source inventory before and after the probe; the source worktree remained
clean. The dedicated build used the pinned AArch64 SDK `aarch64-2c4eb5b2f68f3fa5` and repository
`aarch64-linux-gnu-native.cmake` toolchain, Clang 22.1.8, Debug mode,
`-fno-exceptions`, and explicit CUDA architecture 121. The observed compile
commands used `--target=aarch64-linux-gnu`, `-march=armv8-a`, the SDK GCC 16
installation via `--gcc-install-dir`, `-std=gnu++17`, and `-fno-exceptions`. Python bindings, C++ tests, cpptrace, coverage and internal
checks were all off; only target `xgrammar` was requested.

CMake configuration succeeded. The target build failed with compiler errors
because exception syntax remains in the selected core, including `throw`
statements in [`logging.h`](https://github.com/mlc-ai/xgrammar/blob/614ab01d2af3fdfdde73ae3e4171a7f36db45c23/cpp/support/logging.h#L127-L160)
and `try`/`catch` in [`json_parse.h`](https://github.com/mlc-ai/xgrammar/blob/614ab01d2af3fdfdde73ae3e4171a7f36db45c23/cpp/support/json_parse.h#L97-L100).
These diagnostics confirm a source compatibility failure under the requested
no-exceptions mode, not a setup failure. No tests, benchmarks, install step,
or patches were run. The first probe reached the same exception errors but
its command-line C++ flags displaced the toolchain CPU/GCC initialization.
A second fresh build explicitly supplied those settings; it is the authoritative
compile result. Both runs are retained. The probe does not establish whether an adapted
core can be built or whether XGrammar should be adopted. Any future M7 use
needs a reviewed error boundary or source adaptation compatible with D-066.

The complete logs, CMake cache, compile database, Git tree listings and
pre/post hash manifests are retained outside the repository at
`/tmp/llmp-m35-orchestration/xgrammar-probe-spark-a-r1/` and
`/tmp/llmp-m35-orchestration/xgrammar-probe-spark-a-r2/`. The source-only
inventory and fetched-file hash list are in
`/tmp/llmp-m35-orchestration/xgrammar-audit-inventory.md`.
