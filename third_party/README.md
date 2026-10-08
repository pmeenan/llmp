<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Third-party sources

Every third-party source component a llmpalooza build may use is recorded in
[sources.lock.json](sources.lock.json) (D-017, D-057): its exact bytes, where
they came from, how its license was classified and how it builds. Nothing
else enters the build. [source-dependencies.md](../docs/source-dependencies.md)
explains the mechanism and why it was chosen.

It holds six core components: GoogleTest, toml++, GGML (D-077),
ExLlamaV3's kernels for the native EXL3 linear, CUTLASS's headers for
Qwen3.8's NVFP4 grouped GEMM, and ds4's MIT D2R and token-tile HCA units. toml++, GGML,
CUTLASS and ds4 ship (`use: product`: since D-096 the runtime links the
engine's kernels); GoogleTest
and ExLlamaV3 link only into tests and benchmarks (`use: test`). The last
four are adapted sources: each is its upstream archive narrowed by
`archive.keep`, with the reviewed
patches in [patches/](patches/)`<id>/` that add llmpalooza's build of the files
it compiles (and, for GGML, link only the quantized kernels of the types it
compiles and build without CUB; for ExLlamaV3, reduce three of them to
their kernels; for CUTLASS, name its header trees as one interface
target, compiling nothing; for ds4, add a raw-Q2 loader to the D2R unit and
extract only the original token-tile HCA numerical helpers/core with narrow
launchers). GGML's build compiles the operations of the
backend proof's models and of M3's DeepSeek V4 Flash and Qwen3.8 Flash,
plus checked standalone Gemma routing and scaled expert reduction from the
unchanged pinned `topk-moe.cu` and `moe-weighted-reduction.cu` units. These
units retain the original fast-math flags; the bounded Gemma26 serving recipe
selects them through llmpalooza's graph plan. It does not compile all of GGML (the lock's `license.scope` lists the files).
Patch 0005 adds an IQ2 compact-pair kernel specialization for the measured
GB10 shape; ordinary MMQ configurations remain unchanged.
[licensing.md](../docs/licensing.md) records their audits.

| Step | Where | Does |
| --- | --- | --- |
| Prepare | `mise run prepare` ([tools/prepare-sources](../tools/prepare-sources)); `mise run setup` runs it after the SDK | Validates the lock, selects the profile's closure, then fetches each archive into the persistent cache (`~/.cache/llmp/downloads/<sha256>/`, shared with the SDK and re-verified on every use). One parser, Python's `tarfile`, both checks and unpacks it (D-078): it refuses an archive with anything but plain files and directories at safe paths, two members at one path or a member inside a file member, then writes the checked members through tarfile's `data` filter, stripping a single top-level directory and keeping only the `archive.keep` paths if the lock names any. The unpacked tree must hold only directories and singly linked regular files. It then applies the recorded patches and checks the tree digest before the tree appears as `build/sources/<id>-<tree>` |
| Configure | [cmake/LlmpSources.cmake](../cmake/LlmpSources.cmake) | Validates the whole lock with the same Python code ([tools/inspect-sources](../tools/inspect-sources)), selects the build's CPU/CUDA closure, checks each selected prepared tree's digest, and only then adds the components as `SYSTEM`, `EXCLUDE_FROM_ALL` subprojects with their locked options. It never downloads. It rejects `FETCHCONTENT_SOURCE_DIR_*`, dependency providers and project-include hooks, makes FetchContent population of a declared dependency fail inside components (whatever their policy level), and fails on any `find_package()` lookup the lock does not declare |
| Build | [cmake/sources/verify.cmake](../cmake/sources/verify.cmake) | Checks each tree on every build before anything that uses it compiles, so an edit after configure (an added file, a mode change, an edit that keeps the timestamp) fails the build |
| Receipt | `build/<preset>/llmp-receipt.json` | Records what configure used: the lock's digest, the SDK identity, the license profile and modules, and each component's version, license, archive digest, patches, tree, options and source directory. It is official only when no component came from an override |

`--dry-run` shows what `prepare` would do, and `--check` validates the whole
lock, every patch file included; `prepare` and configure read only the patch
files of the selected closure. A missing tree stops configure with the
command that prepares it. A modified tree stops configure and `prepare`
alike: remove the tree and prepare again, or use an override (below). The
`sources.*` tests check the lock, the receipt, and the compile and link
inventory recorded by Ninja against the receipt. The inventory accepts a
build-tree file only if a current build rule or a selected component's
build directory produced it. No compiled or linked object or archive outside
the SDK may contain exception support (D-066). Nothing the package ships
(`llmp`, `llmp-runtime`, and every object, archive and header they are
built from) may come from a `use: test` component, whose notices the
package does not carry. See [tests/sources/](../tests/sources/).

## Profiles

The **core** profile, the default, is the copyleft-disabled profile of D-002:
the lock's `core` components only. The lock has no optional module yet;
when a confirmed-copyleft one lands, D-080 puts it in the default, with the
core profile as the opt-out. An optional module adds its components
when named at both steps: `mise run prepare -- --modules <m>`, then configure
with `-DLLMP_MODULES=<m>`. Neither step fetches, reads or builds another
module's sources. A build directory that has built a module never builds a
profile without it: configure refuses, because the module's payloads could
be anywhere in that tree. It keeps that history in
`llmp-modules-built.txt` and also refuses a directory that has component
outputs but no such record. Build the other profile in a new directory. The
record guards against mistakes, not a deliberate edit; a module's leftover
that a later build uses still fails the inventory check. The lock has no
optional modules yet. `tests/sources/` checks all of this on a
synthetic one.

Preparation keeps the CPU/CUDA superset of the selected license profile.
Configure excludes a component marked `requires_cuda: true` when
`LLMP_CUDA` is off, before executing that component's CMake. A missing
condition or `false` selects it in either build. The receipt records the
actual CUDA setting, selected components and the full source-lock digest;
the receipt and compile/link inventory checks must both agree with that
selection. Dependencies on excluded components fail. DS4 is CUDA-only:
all six core components are prepared, while CPU builds select five and
CUDA builds select six. The component's pins and license tier are unchanged.

## What the checks cannot see

Component build scripts run with the user's rights; CMake has no sandbox.
Configure blocks the ways a component could substitute a dependency that
CMake exposes: declared FetchContent population, source overrides,
dependency providers and package lookups. It cannot stop a script that
downloads with `file(DOWNLOAD)` or a tool, or copies host files into its own
build directory, which the inventory check accepts as that component's
output. So:

- auditing a component covers everything its build scripts read, write and
  run, and `license.scope` records that;
- the network-denied build (D-061's `check:full`, in the reference
  container with `--network none`) stops any download a build attempts.

Preparation unpacks with the same parser that checked the archive
(D-078), so what the check saw is what is written; an extractor that
parsed the archive differently could have written members the check never
saw.

## The lock (schema 1)

`modules` names each optional module with a `description`. Each entry of
`components` is keyed by a lowercase id and has:

| Field | Meaning |
| --- | --- |
| `version`, `upstream` | The release, its repository, tag and full commit |
| `kind` | `archive`, a hash-pinned upstream archive. Adapted sources come as an archive with patches (D-077); vendored units (`third_party/<id>/` in Git) are not supported |
| `category`, `tier`, `module` | D-017's classification. `implementation` is incorporated code. `core` needs a license in `CORE_LICENSES` (`tools/llmp_sources.py`: any recognized permissive license, and MPL-2.0; D-017, D-091); `optional` needs a `module`. A core component never depends on an optional one |
| `use` | `test` if only test executables link it (never shipped; `sources.closure` refuses a shipped executable built from it), else `product` |
| `machine` | `target`: built with the profile's target toolchain. Build-host tools and generators are not supported until the first one needs a host build |
| `requires_cuda` | Optional boolean, default `false`. `true` excludes the component from a CUDA-free build's configure and receipt; preparation retains it. It is a build condition, independent of the license tier |
| `archive` | `file`, `urls` (https), `sha256` and `size`. Bytes that change under the same URL are an error, not a lock update. A tar archive, plain or gzip-, xz- or bzip2-compressed, judged by its leading bytes. Members must be plain files and directories, one per path, none inside a file member. Optional `keep`: the paths (files or directories, relative to the unpacked tree, sorted, none inside another) that preparation keeps; the rest is never written and never reaches the tree digest, configure or the build. Every kept path must hold a file; a discarded member's name need only be printable ASCII without `\`. `keep` narrows the tree, not the license review: an archive holding implementation outside the component's tier still cannot be fetched and filtered ([source-dependencies.md](../docs/source-dependencies.md)) |
| `patches` | Ordered `path` (relative to this directory, conventionally `patches/<id>/`) and `sha256`, applied exactly: git-style unified diffs of text files with no fuzz, renames, mode changes or binary hunks. Text before the first file and git's signature are skipped; any other line between files is an error |
| `tree_sha256` | The prepared tree's digest: SHA-256 over one `<sha256> <x or -> <path>` line per file, sorted by path, where `x` marks an owner-executable file. Symbolic links, special files and empty trees are refused |
| `depends` | Other components that must be added first |
| `cmake` | `subdirectory` holding the project, `options` set for it alone, the `platform_packages` it may look up with `find_package()` (D-017's declared platform), and the `targets` llmpalooza links. Option values are plain words, never paths. Names may not start with `_`, `CMAKE_`, `LLMP_` or `FETCHCONTENT_`, or be `BUILD_SHARED_LIBS` |
| `license` | `expression` (SPDX identifiers joined by ` AND `), the license `files` in the tree, shipped `notices` (paths in the tree, whole or `path:FIRST-LAST` lines, which the package's third-party notices reproduce), the audited `scope` (what is compiled and executed), the `evidence`, and the `obligations` |
| `verification` | How the pin was checked |

## Adding or changing a component

1. Choose the release and confirm the archive is immutable. Download it and
   record its SHA-256 and size. Compare it with the tagged commit where you can.
2. Audit what the build compiles, includes, generates and executes, and
   what its scripts read, write and download: license headers, bundled
   code, generators and `find_package()` calls. Record the result under
   `license` and `verification`. A new dependency's handoff note names its
   D-017 category and tier. A decision entry is needed where AGENTS.md
   rule 1 calls for one.
3. Add the entry and lock every option the project declares, turning off
   tests, examples, installers and downloads. Where the build uses only part
   of the archive, `archive.keep` limits the prepared tree to that part (and
   the license files). `mise run prepare` then reports the tree digest to
   record. Review that tree before recording it.
4. Build and test every workstation preset.

## Local development overrides

To build a component from edited source, point
`LLMP_SOURCE_OVERRIDE_<ID>` (the id in upper case, `-` as `_`) at a copy
of its tree. Configure warns, and the receipt records the override and
whether it differs from the lock, and is marked unofficial.
`LLMP_REQUIRE_LOCKED_SOURCES=ON`, for check and release builds, rejects
any override.
