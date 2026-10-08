<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llmpalooza

All models, all containers, all the time: an LLM inference engine with
just-in-time SSD paging. `llmp` for short; formerly jitLLM.

Llmpalooza is an independent, open-source inference runtime for setups where
more models should be available than fit in memory. Instead of loading and
unloading whole models, it keeps a node-wide catalog of every managed memory
extent, evicts the least valuable extents across all models when capacity is
needed, and pages missing weights or state back in on demand from prepared
on-disk artifacts. Routed-expert (MoE) models get exactly the experts the
router selected, loaded just in time.

The workload it is built for first is one person switching among a library of
models, or running an agent whose subagents use different models, with
conversations that last hours. Useful conversation state survives switches
within configured retention limits; expired caches can be rebuilt from the
history clients provide.
Standard web-API clients such as Cursor, OpenCode, and Codex work unmodified,
and with more than one node a single conductor places models across the
cluster and routes requests. NVIDIA and DGX Spark come first; the memory,
paging, and transport boundaries are kept portable so Apple silicon or AMD
single-machine ports stay possible later.

Initial target: one or two NVIDIA DGX Sparks, developed from an x86-64 Linux
workstation. Model support is earned per checkpoint and tracked in the
[support matrix](docs/model-support.md); see
[docs/features.md](docs/features.md) for what is confirmed scope versus
still being triaged.

Key properties (confirmed scope):

- **Partial, cross-model eviction** at extent granularity, with capacity
  reservations kept separate from residency leases so admission never
  eagerly evicts useful cache.
- **On-demand expert acquisition** at a routing boundary: no expert
  substitution, no dropped contributions, and execution is suspended while
  I/O is in flight so other work can run.
- **Explicit CUDA virtual memory management** with a semantic resource
  catalog, prepared and hashed model artifacts, and explainable
  eviction and admission decisions.
- **Native C++23 runtime**, Clang-first, no interpreter in the serving path;
  cross-built for Spark and tested over SSH.

Almost all code is written by AI agents working from the project
documentation. Every change gets a separate review pass, and a human directs
the work and reviews it. During M3, the main agent has standing authority
to commit reviewed, checked tasks; agents never push.

## Status

**Pre-release. M0 through M3 are complete; M3.6 is in progress and M3.5 is parked.** The native runtime serves Chat Completions and literal
Completions with target likelihoods, batches compatible requests, and
preserves conversation state across model switches. Its final swap table
covers DeepSeek V4 Flash, Qwen3.8 Flash Next and Qwen-Image-2.1, with a worst
prepared LLM swap of 9.853 s against the 20 s bound.

The owner accepts remaining Qwen/DeepSeek speed gaps for M3 and defers tuning
to M9's full-engine optimization pass. The [M3 record](docs/m3-record.md)
retains measured gaps, quality evidence and checks; the [living plan](docs/plan.md)
owns later work. The arm64 package includes the runtime, systemd service
and native diagnostic command. Planned distribution is a signed apt
repository for DGX Spark; changes are recorded in [CHANGELOG.md](CHANGELOG.md).

M3.6 replaces the per-family runners with one engine of shared components
over llmpalooza's own graph IR, with a format layer that keeps weights
compressed and a native importer ([design](docs/engine-components.md)).
M3.5 then adds the approved model families, formats, decision models and
media routes on that engine. M4 follows with two Sparks; M5 completes the
standard front door. M6 adds partial retention under memory pressure,
M6a configured placement across nodes, and later milestones add demand-paged
MoE, sharding and full-engine optimization. See [the plan](docs/plan.md).

## Development setup

Development happens on an x86-64 Ubuntu 24.04 workstation. Spark binaries
are cross-built there and tested over SSH. The compilers, CUDA, CMake and
Ninja come from a pinned SDK that the project provisions under your home
directory. Setup changes no system compilers, packages or shell files.

1. Install [mise](https://mise.jdx.dev) 2026.9.12 or newer and the packages
   in [toolchains/prerequisites/](toolchains/prerequisites/). Setup prints
   the `apt-get` command for any that are missing.
2. In the checkout, run `mise trust` and then `mise run setup`. Setup
   downloads the artifacts pinned in
   [toolchains/artifacts.lock.json](toolchains/artifacts.lock.json),
   checking each SHA-256, builds the GCC 16.2 runtimes and assembles the
   SDK. It then prepares the third-party sources pinned in
   [third_party/sources.lock.json](third_party/sources.lock.json) into
   `build/sources/` (`mise run prepare` does that step alone). Later runs
   return at once.
3. `mise run doctor` checks the SDK and reports the host, driver and
   toolchain identities.

[toolchains/README.md](toolchains/README.md) describes the SDK. The
reference container in [.devcontainer/](.devcontainer/) runs the same setup
on a clean Ubuntu image.

## Building and testing

`mise run build` configures and builds with the SDK's CMake, Clang and
Ninja; `mise run test` builds and runs the tests. Both take CMake preset
names after `--` (default: `native` on x86-64, `spark-native` on a Spark),
and write to `build/<preset>/`:

| Preset | Builds | Tests run |
| --- | --- | --- |
| `native` | x86-64, CUDA for `sm_121` and the workstation's discrete `sm_86` (D-082) | On the workstation, GPU tests skipped; its discrete-GPU tests only with `--gpu` |
| `cpu` | x86-64 with no CUDA toolkit | On the workstation |
| `cross` | AArch64 for DGX Spark, CUDA for `sm_121` | Under qemu-user, GPU tests skipped; or on a Spark with `--host` |
| `spark-native` | AArch64, built on a Spark (each slice's check, D-084) | On that Spark |
| `cpu-asan` | `cpu` with ASan, UBSan and LeakSanitizer | On the workstation |
| `cross-asan` | `cross` with ASan and UBSan | Under qemu-user without leak detection; or on a Spark with it, with `--host` |
| `cross-tsan` | `cross` with ThreadSanitizer | Only on a Spark, with `--host` |

For example, `mise run test -- native cpu cross` runs every workstation
profile, and arguments after a second `--` go to CTest
(`mise run test -- cpu -- -R contract`). `mise run test -- cross --host
<spark>` copies the cross build to the named SSH host, under
`~/.cache/llmp/deploy/`, and runs its CPU and GPU executables there; build
and binary inspections stay on the workstation. `mise run deploy --
--host <spark>` builds and copies without running tests. Neither ever picks
a host for you. Deploying needs `ssh` and `rsync` on the workstation and
`rsync` on the Spark. `mise run test -- native --gpu` runs only the `native`
build's `gpu-discrete` tests (test preset `native-gpu`), one at a time, on
the workstation's discrete GPU with its driver (D-082): the device-memory
and device-execution providers, the lanes, the page-in path through the
landing zone into device VMM (D-081), a GGML kernel smoke, the EXL3
kernels' loading, llmpalooza's EXL3 launchers and a phase of the native EXL3
plan, and `llmp doctor`. Tests that compare with GB10 records stay
GB10-only. Nothing runs them by default; on a shared machine, run them
under its lock for GPU work.

A CUDA build's binaries need the NVIDIA driver (`libcuda.so.1`) to start: a
hard requirement, which every Spark meets (D-072). The build links NVIDIA's
stub from the SDK, so building needs no driver. The workstation presets'
tests skip GPU tests and always load that stub from `build/<preset>/cuda-stub/`,
so they also run where there is no driver; only Spark runs and `--gpu` runs
use the driver. On a host without the driver, run the `cpu` preset's
`llmp` by hand instead.

A built `llmp doctor` (in `build/<preset>/src/cli/`) reports what a host
offers that build: the driver, each GPU's compute capability, VMM support and
backing granularity, and RDMA ports, and whether each GPU is unified (the
GB10) or discrete. Llmpalooza uses one GPU, device 0 (`CUDA_VISIBLE_DEVICES`
selects it), and doctor judges only that one. It exits 1 when the host cannot
run the build: a GPU 0 the build has no code for, or one without VMM and
host-backed VMM or of the other class than the build's code for it assumes
(D-072, D-082).
Spark builds have code only for the GB10; the `native` build also has it for
the workstation's discrete RTX 3080 Ti.

## The package and the runtime

`mise run package` builds the `cross` preset from locked sources and makes
the arm64 Debian package, `build/cross/package/llmp_<version>_arm64.deb`
(D-063, D-074). It installs `/usr/bin/llmp`, the node runtime
`/usr/libexec/llmp/llmp-runtime` and `llmp.service`, creates the
`llmp` user and `/var/lib/llmp`, and enables and starts the service.
The package depends on `libc6` and on the NVIDIA driver's `libcuda.so.1`
580 or newer. Its documentation in `/usr/share/doc/llmp/` holds the
license, the notices of everything the binaries carry
(`THIRD-PARTY-NOTICES`), a Debian `copyright` file, an SPDX SBOM and an
annotated example configuration. The package ships no configuration: with
none, the runtime is a standalone node with the defaults. Removing or
purging the package keeps `/var/lib/llmp`, `/etc/llmp` and the user.
Purge removes llmpalooza-owned spill and kept-conversation files, preserving
model storage.

The runtime reads `/etc/llmp/llmp.toml` and the fragments in
`/etc/llmp/llmp.d/` (D-073), prepares its storage roles, checks the host
as `llmp doctor` does, registers configured models and serves the minimal
chat/completions route on loopback and the tailnet. The full front door
arrives in M5. A refusal at startup
exits 78, which the unit does not restart after; `journalctl -u llmp`
and `llmp doctor` say why. For a development run,
name the configuration and the enrollment anchor (the process lock is
`<anchor>.lock`):

```bash
build/cpu/src/runtime/llmp-runtime --config ~/llmp-dev/llmp.toml --anchor ~/llmp-dev/enrollment
```

`tools/job-proof` runs the confined-job proof (D-074) in delegated cgroups of
your own systemd manager, and `tools/job-proof --host <spark>` runs it there
as `llmp` under the service's sandbox, which needs the package installed.

To run CMake by hand, point `LLMP_SDK` at the SDK and use its `cmake`:

```bash
export LLMP_SDK=$(tools/setup-toolchain --print-root)
"$LLMP_SDK/bin/cmake" --preset native && "$LLMP_SDK/bin/cmake" --build --preset native
```

## Versions

Llmpalooza uses [SemVer 2.0.0](https://semver.org/spec/v2.0.0.html) and stays at
0.x until 1.0 (D-062). `project(VERSION)` in [CMakeLists.txt](CMakeLists.txt)
holds the next release, and every build derives the version it reports from
it and Git:

| Checkout | Version |
| --- | --- |
| A clean checkout of the release tag `vX.Y.Z` | `X.Y.Z` |
| Anything else | `X.Y.Z-dev.N+g<commit>`: N commits since the last release tag (since the root before the first release), and `.dirty` after the commit when tracked files differ or untracked files Git does not ignore exist |
| A tree without Git metadata, such as the reference build's copy | `X.Y.Z-dev+unknown` |

A release tag is an annotated tag `vX.Y.Z` on a commit HEAD contains; the
owner creates and signs them, never an agent. Lightweight and other tags are
ignored. After the release commit, configure and build stop when
`project(VERSION)` is not above the last release tag (the first commit after
a release raises it). A clean release checkout must match its tag; an
uncommitted edit may raise the version so it can be checked before commit,
but cannot lower it. Shallow clones and Git errors also stop the build.
A clone fetched without its tags cannot be told apart
from one with no release yet, so it counts N from the root: fetch the tags
before building. Every build derives the version again, so a commit or an
edit needs no fresh configure.

`llmp --version` prints the version, commit, license profile, SDK and
target. The build receipt, `build/<preset>/llmp-receipt.json`, records the
same information: its `version` object holds the product and Debian
versions, commit, modified state and origin, while the license profile, SDK
and target are top-level fields. The Debian package version maps
`-` to `~` so that a dev build sorts before its release
(`X.Y.Z~dev.N+g<commit>-1`). A change with user-visible effect adds its line
to [CHANGELOG.md](CHANGELOG.md) in the same change (D-062).

## Checks

Each slice of work is checked once on a Spark before handoff (D-084):
`mise run test -- spark-native --locked` there, plus the light local steps
the change touches. The local check gate's tiers (D-061; there is no hosted
CI yet) run at milestone gates, for releases and for changes that need the
workstation. Each tier ends with a summary of its steps and the host,
commit and SDK they ran with:

| Task | When | Runs |
| --- | --- | --- |
| `mise run check` | Milestone gates; changes that need the workstation | clang-format, REUSE lint, the embedded-header check, the tooling tests, the `native`, `cpu` and `cross` builds and tests (cross under qemu-user), and clang-tidy |
| `mise run check:full` | Milestone gates; toolchain, dependency and packaging changes, and releases | `check`, then `cpu-asan` and `cross-asan`; the reference build: the checkout copied into the [reference container](.devcontainer/), its sources prepared from an empty cache, then `native`, `cpu` and `cross` built and tested with no network; the arm64 package, its inventory against the build receipt, `NOTICE` and the SBOM, and its install test in an arm64 container with no network; and the confined-job proof in your systemd user manager. Needs Docker |
| `mise run check:spark -- --host <spark>` | Milestone gates; changes that need the Spark sanitizer builds | The `cross`, `cross-asan` and `cross-tsan` tests on that Spark, GPU tests (`llmp doctor` on the GB10 among them) and leak detection included |

Check builds configure afresh with the build tool's `--locked`: the core
profile, from locked sources only, with nothing kept from a build
directory's cache. The steps ignore the caller's compiler, CMake, test and
sanitizer environment (`CXXFLAGS`, `GTEST_FILTER`, `ASAN_OPTIONS` and the
like; `tools/check` lists them). The reference build provisions its SDK inside the
container, into the `llmp-sdk` and `llmp-cache` volumes the dev
container also uses. The first run takes as long as `mise run setup`.

Every file carries its copyright and license as SPDX tags
([REUSE](https://reuse.software/spec-3.3/)): in a header comment wherever the
format allows one, otherwise in a `.license` sidecar beside it (`data.json`
and `data.json.license`). A new file copies the header of its neighbours:

```text
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
```

with the file's own comment syntax (`//` in C++ and CUDA, `<!-- -->` in
Markdown). The header check fails on a file type it does not know yet; add
the type to [tools/llmp_headers.py](tools/llmp_headers.py).

## License

Llmpalooza's own code is Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE));
[LICENSES/](LICENSES/) holds the text of every license a file in this
repository declares. Incorporated core
implementation dependencies use Apache-2.0, BSD, MIT, or MPL-2.0. Build tools
and declared platform dependencies, including system libraries and CUDA,
retain their separate terms and are included in the dependency audit.
AGPL-licensed kernels or importers live only in optional modules you choose
to enable at build time; those builds must report and satisfy the applicable
license, notice, and source obligations. See D-003 and D-017 in
[docs/decisions.md](docs/decisions.md), and
[docs/licensing.md](docs/licensing.md) for what each dependency and tool
contributes to a build and what its notices require.

## Start here

- [AGENTS.md](AGENTS.md) — constraints, doc map, agent rules
- [docs/vision.md](docs/vision.md) — why, who for, success criteria, non-goals
- [docs/features.md](docs/features.md) — confirmed / proposed / open questions
- [docs/plan.md](docs/plan.md) — the M1–M10 milestone ladder with exit criteria
- [docs/m0-record.md](docs/m0-record.md) — what M0's planning, spikes and reference runs did, with evidence links
- [docs/m1-record.md](docs/m1-record.md) — what M1's bootstrap items built and where each was verified
- [docs/m2-record.md](docs/m2-record.md) — what M2's resource core and backend proof built, with evidence, caveats and the gate
- [docs/workflow.md](docs/workflow.md) — how agents and the human collaborate
- [docs/rough-edges.md](docs/rough-edges.md) — findings log
