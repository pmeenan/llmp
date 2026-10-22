<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Comparison implementation identity

The v11/v12 measurements remain historical evidence with their original IDs.
Trait dispatch and the corrected implementation closure invalidate them. No
identity fields are rewritten to make observations current; requalification
requires a reviewed source freeze and a new prospective registration.

`cmake/LlmpSourceClosure.cmake` follows owned transitive build targets from the
GGML implementation and concrete device runtime, plus the actual
`engine/graph_plan.cc` object. CMake selects translation units and applies
consumer, interface and per-source requirements. The compiler selects headers.
The CMake front door automatically freezes owned-source membership before
project generation and refuses configure-time changes to that source tree.
Ordinary incremental authored edits still compile with current-byte receipts;
new files refresh through normal configure inventory dependencies. Ninja output
metadata and declared GENERATED properties identify generated inputs regardless
of directory. Unknown external compiler inputs need an explicit source/platform
or generation contract. Unrelated snapshot bytes never enter implementation
identity. There is no include-text scanner or second implementation of include lookup.
Unrelated serving/importer targets are outside this closure; linked
model/spec/artifact modules remain conservatively included.

GGML implementation objects compile before the generated identity translation
unit. The downstream public archive contains those implementation objects and
the header-free `const char*` identity function. Before generation, its resolved
compile command and fixed environment enter primary identity. Its compile accepts
only ordinary target/optimization/diagnostic flags and inert uppercase scalar or
string definitions; forced headers, forwarded preprocessor flags, PCH/modules,
plugins and semantic token redefinitions are refused. The compiler depfile must
contain only the exact literal source. Its fresh object receipt is checked after
compilation, without feeding the constant object or its value back into primary
identity. The split retains the original public/private
usage requirements. The decode/EXL3 selected adapters link the tensor/common
provider they actually use, with EXL3 explicitly linking the decode composite
provider; they do not link back to the downstream identity archive. An interface source that would add an extra compilation in
the downstream archive is refused rather than silently changing its context.

The compiler launcher in `tools/compiler_identity.py` records the actual command,
working directory, effective environment, compiler-produced `-MD` dependency
file, source hashes and object hash immediately after successful compilation.
Known inputs are checked before and after compilation. Newly discovered inputs
whose metadata changed during compilation are refused. Objects and inputs are
checked again over receipt creation and over final identity generation. Builds
require quiescent source/SDK/generated inputs; concurrent edits are not a
supported way to build a measurement binary. Missing, stale, ambiguous or
observably changing receipts fail the build. Ninja's historical `VALID` status
is not used as current-source attestation.

The complete authored-file membership and explicit generated declarations live
in `compiler_ownership.json`, separate from the compile semantics in
`compiler_identity.json`. Each compile and identity operation snapshots the
whole catalog; receipts reference a retained, content-addressed copy under
`compiler-ownership/`. Catalog changes during an operation refuse. At identity
validation, every actual receipt input and generator input is checked against
the current catalog and current Ninja output classification, including inputs
whose data values are deliberately excluded from implementation identity.
Removing or reclassifying a consumed input cannot reuse its old authority.

An unrelated catalog addition/removal schedules identity validation without
changing the object command stamp or compiler dependencies. Catalog membership
alone is audit provenance, not implementation identity. The catalog has no
exemption if a compiler actually consumes it: ordinary generated-input recipe
checks apply. Configuration, compiler-consumed bytes, generator recipes and
build declarations retain their existing invalidation rules. This build-recipe
change itself changes implementation identity; historical measurements retain
their original identities and are not re-pinned.

Configure/generate-time compiler configuration, ownership catalog, object-list
and CUDA host-wrapper files are
explicit `GENERATED`, noncompiled private sources of the identity target.
The generic source inventory reads their ownership and declaration backtraces
from CMake's current File API codemodel. It checks the source/build roots and
current Ninja target, and refuses a reply older than the Ninja graph. This
admits declared configure products without inventing a Ninja producer or
allowing arbitrary build-tree inputs; declaration origins still pass the
existing source/component checks.

Compiler dependency paths with literal POSIX backslash aliases are unsupported
because the pinned compiler can lose that distinction in MD output. The launcher
checks actual ancestor directory entries before lexical normalization and at
receipt validation; ambiguous existing aliases refuse. Unrelated directory
timestamps are not receipt inputs. This format bound retains the existing
quiescent-input requirement.

After validating the compiler's dependency rule, the launcher gives Ninja that
complete dependency set plus its authenticated wrapper, configuration, response
and generator inputs. Thus changes to inputs the compiler itself does not list
schedule the affected objects, rather than merely failing a downstream stale
receipt check. This also applies to the header-free constant: its original
compiler rule must still contain only its literal source before launcher inputs
are added. The emitted paths follow the pinned Ninja depfile grammar, verified
with actual dependency consumption, no-op and changed-input builds. Control
characters, a trailing backslash and backslash immediately before a dollar sign
are refused as unsupported dependency-path spellings. Unchanged toolchain reconfiguration preserves the wrapper's bytes and
metadata; permissions are set only when its content changes or execution access
needs repair. Automatic rebuilds cover ordinary content/mtime changes. A manual
permission-only change advances ctime without mtime; even if configure restores
execution access, the old receipt is strictly refused rather than silently
reauthenticated. All concurrent-change and stale-receipt checks remain in force.

The portable identity includes compiler dependency bytes, normalized effective
arguments, source/build declarations, generator code and source/SDK receipts.
Raw object hashes and filesystem metadata remain compilation provenance. In the
native proof, an unchanged NVCC compilation produced different object bytes;
those bytes therefore cannot define the portable implementation identity.
The SDK receipt's installation timestamp is likewise provenance, not semantics.

Only recognized filesystem argument operands are normalized, using complete
path-component boundaries for configured source/build/SDK/prepared roots.
Macro values remain literal; unsupported compiler options refuse. Actual compiler
file-prefix maps make `__FILE__` and `__BASE_FILE__` independent of those roots;
the native proof checks Clang plus NVCC host and device values. Wall-clock date/time/file-timestamp macros have explicit fixed compile definitions
(1970-01-01 UTC); conflicting command-line definitions are refused. This remains
deterministic even when source pragmas alter warnings. Implicit compiler
configuration is disabled.

This is a narrow pinned-build contract. Compiler processes receive an explicit
fixed environment. Unsupported ambient compiler and loader overrides fail before
execution; they are not silently stripped and accepted. Arbitrary launchers,
compiler plugins/configuration files, optimization profiles, PCH and C++ modules need an explicit
provenance contract before use. Ordinary translation units accept only the pinned
object-compilation option contract; NVCC host forwarding validates each supported
option, and unsupported forwarding aliases refuse. Source-owned response files are captured;
generated response files, sources and headers require a declared recipe or are
refused. The current supported generated compile input is the SDK's CUDA host
wrapper: its exact configured output is checked and its generator inputs and
semantic parameters are included. Other generated implementation code receives
no filename-based exemption.

The only excluded implementation objects are strictly derived module-identity
constants and the separate declarative fusion-rank data target. Their generators
remain covered. Before compilation, excluded outputs are regenerated into scratch
and compared in full against the source to compile. Only their authenticated generated value bytes and explicitly declared rank JSON
values are excluded. Their command, environment, generator recipe and other
compiler dependencies enter the implementation digest, so forced headers cannot
hide extra executable code. Object/input receipts also verify their integrity. Every canonical rank field participates in `RankDataDigest`,
registry identity and plan-cache identity. Rank data changes those identities
without feeding back into implementation identity. No broad version/report
exception is assumed: the generated object inventory must demonstrate that
report-only version code is outside the selected roots.

The build emits `compiler_objects.txt`, `compiler_identity.json`,
`compiler_ownership.json` and its retained catalog snapshots, per-object
`.llmp-compile.json` receipts and `compiler_identity_receipt.json`. The benchmark,
measurement launcher, full frozen source manifest and binary hash remain
separate measurement provenance. Python controls cover operand normalization,
depfile parsing and receipt refusals. The native compiler fixture exercises the
production launcher and CMake module across source/build roots, clean/no-op and
incremental builds, interface-header selection, and unsupported-input controls.
The generated identity checker authenticates the actual production build before
measurement. These are functional controls, not performance qualification.
