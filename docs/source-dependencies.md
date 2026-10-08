<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# C++ source dependencies

D-057 resolves M0 question 7: use **CMake FetchContent with a checked-in
source lock and curated vendoring for adapted source units**. D-012/D-049
continue to own SDK provisioning; this decision owns implementation source
and its build inputs. Implementation, concrete dependency pins and clean
build evidence land in M1, with kernel adoption in M2. No library or kernel
is admitted by choosing this mechanism.

## Why this fits

CMake is already selected. Most ordinary libraries can become local CMake
targets, while D-053 requires selected GGML/EXL3 kernels with llmpalooza-owned
dispatch and reviewed modifications. A small, explicit source closure fits
that work without a second package resolver. This is a maintenance judgment,
not a measured performance advantage or a claim of bit-reproducible binaries.

Official documentation checked 2026-09-23:

| Option | Capability and decision |
| --- | --- |
| CMake FetchContent | Built-in source population, including script mode; archive digests use `URL_HASH`. Selected for ordinary source acquisition, with project-owned selection and provenance. [FetchContent](https://cmake.org/cmake/help/v4.4/module/FetchContent.html), [download options](https://cmake.org/cmake/help/v4.4/module/ExternalProject.html#download-step-options) |
| CPM.cmake | Wraps FetchContent and adds package/cache conveniences. Those conveniences do not remove our source-selection and license work; initially omit the extra bootstrap dependency. [Upstream](https://github.com/cpm-cmake/CPM.cmake) |
| vcpkg manifest mode | Declares dependencies, baselines, overrides and features, with binary caching available separately. Useful if the ordinary library graph grows; initially avoid maintaining ports alongside the adapted kernel sources and the project SDK. [Manifest mode](https://learn.microsoft.com/en-us/vcpkg/concepts/manifest-mode), [binary caching](https://learn.microsoft.com/en-us/vcpkg/users/binarycaching) |
| Conan 2 | Lockfiles and distinct build/host profiles support pinned cross-build graphs. Also viable; recipe/profile maintenance is not justified by the currently selected source slices. [Lockfiles](https://docs.conan.io/2/tutorial/versioning/lockfiles.html), [cross-building](https://docs.conan.io/2/tutorial/consuming_packages/cross_building_with_conan.html) |
| Git submodules | A gitlink identifies an upstream commit, but patch, license and transitive-closure records remain ours. Do not make recursive submodule checkout the contributor bootstrap. [Git documentation](https://git-scm.com/docs/gitsubmodules) |
| Curated vendoring | Check in only audited source units that need adaptation, with upstream identity, original hashes, modifications and notices. Selected for such units; avoid copying complete framework trees merely to obtain kernels. |

## CMake version and policies

D-058 pins **CMake 4.4.3** for both build hosts, with
[verified binary identities, seven semantic checks on both hosts and a native/cross/CUDA smoke](experiments/cmake-fetchcontent/README.md).
M1 selects this exact version through D-049's SDK; distro CMake is not the
project pin. Use `cmake_minimum_required(VERSION 4.4.3)` in project entry
points and preparation scripts, plus the SDK's exact-version check.

This selects `NEW` behavior for `CMP0168` (no FetchContent sub-build),
`CMP0169` (reject deprecated single-argument population) and `CMP0170`
(reject missing source directories in declared fully disconnected use).
Preparation unpacked archives with long-form `FetchContent_Populate(name
...options...)` in script mode until D-078 moved unpacking to the Python
parser that checks the archive; configure and build still never acquire
source. Under
`CMP0168=NEW`, `DOWNLOAD_DIR` is ignored too: the setup layer owns the
persistent archive cache and supplies verified local archives for reuse.
See the pinned series' [FetchContent documentation](https://cmake.org/cmake/help/v4.4/module/FetchContent.html)
and [CMP0168](https://cmake.org/cmake/help/v4.4/policy/CMP0168.html).

## One source lock, two acquisition forms

M1 adds one authoritative, machine-readable lock for the selected source
graph: [third_party/sources.lock.json](../third_party/sources.lock.json), whose
schema and workflow are in [third_party/README.md](../third_party/README.md).
It records:

- A stable component ID, upstream location, exact release identity and full
  commit where available; immutable archive SHA-256 or vendored file hashes.
  Branch names, moving tags and version ranges are not sufficient identities.
- Explicit transitive source and generator inputs, build options, enabled
  profiles, and whether each executable runs on the build host or each
  library targets the runtime machine. SDK references use D-049's manifest,
  not a second independently maintained set of compiler pins.
- D-017 category and implementation tier, actual license expressions and
  evidence, selected-file scope, notices and source/redistribution duties.
  Generated headers, embedded tables, templates and bundled source count.
- For adaptations, the upstream file identity, ordered patch hashes or a
  reviewable upstream-to-vendored diff, and the resulting source identity.
  Keep required upstream attribution alongside the code; D-029 applies to
  patches containing upstream source too.

For an **ordinary library**, retain a hash-pinned upstream source archive
outside the checkout. Mirrors may supply the same bytes; an archive changing
under the same URL is an error, not an automatic lock update. Prepare fresh
source from the verified archive and apply recorded patches exactly once.
An upstream archive's bundled dependencies and selected build scripts must
be included in the audit, not hidden behind the root project's license.

For **adapted kernel/source units**, vendor the audited closure in the core
tree with its own CMake targets. Track unchanged origin bytes and local
changes separately enough to review an upgrade; no requirement to retain a
second full upstream tree in Git. Optional implementation payloads, including
patches carrying their source, belong in separately selected bundles rather
than the core checkout. This keeps D-017's no-fetch rule achievable.

D-077 amends this for GGML, the first adapted source: its pinned archive,
narrowed by `archive.keep` to the paths the build uses, with reviewed
patches and a llmpalooza build file added by patch. The archive's hash proves
the origin bytes and the patches are the local changes, so nothing is
vendored into Git.
ExLlamaV3's kernels enter the same way, as the `exllamav3` component:
`keep` holds exactly the closure of the compilation units the build
compiles (upstream's GEMM units and llmpalooza's instance unit, added by
patch, over the GEMV kernel and three sources a patch reduces to their
kernels), which excludes every ATen host wrapper
([licensing.md](licensing.md#exllamav3-gemm-kernels-in-the-core-m2)).

GGML and EXL3 reference revisions in D-051/D-052 are candidate origins, not
an instruction to import their whole repositories or runtimes. The
[licensing inventory](licensing.md) still blocks the untraced tokenizer
tables from core incorporation; D-080 made the dense EXL3 GEMV files
core-eligible. Other
selected units also need their own closure audit. This decision neither
clears those gates nor changes D-017's allowlist.

## Preparation and build

1. **Select before acquiring.** Resolve the finite closure for the requested
   CPU/CUDA, native/cross and license profile from the lock. Default
   preparation retains the CPU/CUDA superset of the license profile;
   configure excludes `requires_cuda: true` components from a CPU build
   before executing their CMake and records that selected closure. Optional
   implementation modules are in the default selection (D-080; the tooling
   still selects none, since the lock has none yet), and the
   copyleft-disabled profile is the build-time opt-out. Reject a dependency on a disabled or
   unclassified component before downloading or configuring any of it. A
   full archive containing excluded optional implementation cannot be fetched
   and then filtered for the copyleft-disabled build; use an audited curated
   source bundle or permitted vendored units instead.
2. **Prepare explicitly.** M1's mise setup task provisions the SDK and
   prepares the selected sources as separate steps. Fetch local archives
   or HTTPS URLs with SHA-256 and TLS verification, and unpack with the
   parser that checked the archive (D-078). Do not execute an
   upstream build during acquisition. Keep downloaded archives in a
   persistent cache outside the checkout; create patched working sources in
   an isolated build area, never by modifying a shared cache in place.
3. **Configure and build from local inputs.** Verify the prepared source
   identity against the lock before executing its build scripts. Use local
   `add_subdirectory` for admitted CMake projects or llmpalooza-owned targets for
   selected files. Turn off unselected upstream examples, tests, installers,
   auto-downloads, architecture detection and framework runtimes. Every
   transitive dependency must already be in the selected lock closure.
   Missing inputs fail with a preparation instruction; configure and build
   do not fetch or fall back to an installed implementation library.
4. **Keep the profiles separate.** Source bytes may be shared, but build
   directories and generated outputs are separate for native x86-64,
   cross AArch64 and native Spark, CUDA/CPU, sanitizers and license options.
   Libraries use the selected D-032 target toolchain; generators execute
   with the build-host toolchain. Target libraries/headers are found only in
   the selected target roots; host tools never come from an ARM target build.
   Declared SDK/platform libraries remain separately inventoried.
5. **Record what was used.** Emit a build receipt with the source-lock digest,
   actual selected closure, modifications/options, SDK identity and profile.
   The receipt feeds D-029's notices and SBOM, together with the compiled,
   linked and shipped-file inventory; the lock alone is not that inventory.

CMake's disconnected flags are not an offline bootstrap or a security
boundary. For declared population, `FETCHCONTENT_FULLY_DISCONNECTED` assumes
already populated sources, and `FETCHCONTENT_SOURCE_DIR_<NAME>` overrides skip fetching;
neither establishes source integrity. Our preparation/identity check owns
that guarantee. [CMake's documented behavior](https://cmake.org/cmake/help/v4.4/module/FetchContent.html#variables).
The check gate must additionally deny network access during configure/build
and inspect the resulting dependency closure. Until hosted CI exists, that is
D-061's `check:full` tier in the reference container with `--network none`
(RE-013), since upstream build scripts can execute
arbitrary commands. No CMake variable makes them a sandbox.

Supported check/release profiles reject unrecorded source overrides, dependency
providers and package-search fallbacks. A deliberate local development
override may use edited source, but records that identity as modified and
cannot produce an official release receipt. Source selection never bypasses
license admission. Platform discovery stays permitted only for D-017's
declared components, with version/path checks against the selected SDK and
sysroot. CMake documents the relevant build-host versus target search-root
controls in its [toolchain guide](https://cmake.org/cmake/help/v4.4/manual/cmake-toolchains.7.html#cross-compiling-for-linux).

No shared binary-package cache is required initially. If one becomes useful,
its identity must cover sources, patches, compiler and support runtimes,
sysroot, CPU/GPU targets, options and license profile. A source-cache hit is
not permission to reuse a binary from another build.

## Upgrades, packaging and M1 gates

An upgrade changes the lock, source/patch audit and relevant compatibility
tests together. Incompatible pins for the same component fail explicitly;
there is no automatic version negotiation. Review any numerical kernel
change against D-051/D-052's references. Tags and URLs are retrieval hints;
the reviewed bytes and their provenance are the build identity.

Package builds consume prepared sources with the network disabled. Preserve
the selected source archives or vendored files, patches, build instructions
and applicable notices with release provenance so upstream disappearance
does not make a shipped binary untraceable. Provide corresponding source
when its actual license requires it; source availability and redistribution
must be established before shipping. This does not authorize redistribution
of the D-032 sysroot, CUDA SDK, model weights or any other external input.
Optional packages and source bundles follow their selected closure; default
packages must not accidentally include disabled modules or their notices
as a substitute for excluding their code.

M1 must demonstrate the mechanism on its chosen, audited test dependency;
it need not import the M2 kernels to prove source acquisition:

- Clean preparation and a fresh offline configure/build with the same lock,
  both natively and cross-built, using D-049's shared host/container setup.
  Missing inputs fail before configure runs third-party code; a changed
  archive, patch or prepared source is rejected. Changing a locked input
  invalidates the old prepared tree and outputs.
- CPU-only builds with no CUDA SDK, and copyleft-disabled preparation/builds
  starting from empty caches. Verify excluded sources, headers, generators
  and generated outputs are never fetched or built; ordinary tests and the
  core remain usable. A synthetic optional fixture may test exclusion without
  adopting a copyleft dependency.
- Reconfiguration from an optional-enabled build to the core profile cannot
  reuse its generated or linked payloads. A transitive dependency attempting
  an undeclared download, a system-library substitution or an unrecorded
  source override fails the supported build checks.
- Build-host tools and target objects have the correct architecture; target
  discovery does not find host implementation libraries. Match the actual
  compile/link/package inventory to the receipt, notices and SBOM.

D-058 settled CMake, D-059 settled Ninja, the developer tools and
GoogleTest (M1's test dependency for these gates), D-061 the local check
gate that runs them and D-063 the installed layout the package follows. None
of these claims the M1 implementation done.
