<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Licensing and provenance

jitLLM-authored code is Apache-2.0. [D-002, D-003, D-017, D-080 and
D-091](decisions.md) govern incorporation, optional modules, tools and
platform dependencies.

**What the core admits** (the owner, 2026-09-28; D-091). Any permissive
license (no copyleft or share-alike obligation, no field-of-use or
non-commercial restriction) may enter the core without a decision of its
own, beside MPL-2.0 with its file-level obligations. Copyleft code lives in
optional modules, which jitLLM's own builds ship by default, with the
copyleft-disabled profile as the build-time opt-out (D-080). Unknown,
non-permissive or proprietary terms still need a decision. Every component
still gets its provenance record, and its notices and attribution still
ship. `CORE_LICENSES` in [tools/jitllm_sources.py](../tools/jitllm_sources.py)
lists the permissive SPDX identifiers recognized so far; adding one (or a
`LicenseRef-` for custom terms read and found permissive) needs no
decision, only the heavy-path review of the change. The source lock
refuses a core component outside the list, and packaging refuses a `src/`
file outside it, or under a license no source-lock component or shipped
provenance unit declares, that has no record (below).

The first three sections record how the repository declares licenses and
what the pinned toolchain puts into a jitLLM binary (M1, checked
**2026-09-24**; D-071). The rest is the M0 reference inventory, checked on
**2026-09-22**: evidence for the first dense slice, early EXL3 proof and
later optional-backend design, not approval to import these repositories or
a license audit of their built containers. M2's incorporated sources are
audited in the [source lock](../third_party/sources.lock.json); the
[ExLlamaV3 GEMM kernels](#exllamav3-gemm-kernels-in-the-core-m2) have a
per-file record below.

## Repository license metadata

Every file follows the [REUSE specification 3.3](https://reuse.software/spec-3.3/)
(D-029, D-071):

- A file whose format has comments starts with `SPDX-FileCopyrightText` and
  `SPDX-License-Identifier` tags in a header comment. jitLLM's own files say
  `2026 jitLLM contributors` and `Apache-2.0`; third-party material names its
  holders and actual license, as the patches with MIT upstream context do
  (GGML's and ExLlamaV3's in `third_party/patches/`, and two experiments').
- A file that cannot hold a comment (JSON, a patch, plain text, `NOTICE`, or
  `mise.lock`, which mise rewrites) has a `.license` sidecar with the same
  tags. There is no `REUSE.toml`: it could override a file's own header.
- [LICENSES/](../LICENSES/) holds the text of every license a file declares:
  Apache-2.0 (the same bytes as the root `LICENSE`), MIT (SPDX License
  List 3.29.0) and Unicode-3.0 (the tokenizer's tables, below).
- [NOTICE](../NOTICE) carries jitLLM's attribution and names the
  third-party material in the repository.

`mise run check` enforces this with two steps. `reuse` runs REUSE lint 6.2.0
from the SDK. `headers` ([tools/jitllm_headers.py](../tools/jitllm_headers.py))
checks what REUSE lint accepts but D-029 does not, against REUSE's own JSON
report. Each commentable file must carry its own header rather than a
sidecar, and REUSE must read exactly that header's license from the file
itself. A license or copyright tag anywhere in a file or its sidecar, read
as a person would see it, must repeat an expression and a holder REUSE
reads for that file, so no tag can show a reader a license REUSE skipped.
Every sidecar must have a file, REUSE must cover every file the check does,
and no file type may go unclassified.

## What builds jitLLM

The M3 direct-product experiment also incorporates the MIT D2R unit from
Entrpi/ds4 at `76d51ef82a81b70b78e51a3a6ea11946286de976`, through its
locked archive and reviewed raw-Q2_K/build patch. The kept unit includes
the original IQ2/Q8 routines; native dispatch calls only raw Q2_K down,
with jitLLM's preparation and completion. The whole archive audit and
exact source scope are in the source lock. Packages carry the ds4 authors,
Entrpi, GGML and Marco Palaferri MIT notices; no ds4 runtime or cache
implementation is incorporated.

[toolchains/provenance.toml](../toolchains/provenance.toml) records every
locked SDK artifact, host prerequisite and mise tool with its D-017
category, license, what of it reaches a binary and the notices that follow;
a test fails when any of them lacks a record. Third-party source code is
recorded in the [source lock](../third_party/README.md) instead: GoogleTest
links into tests alone, and toml++ (MIT, D-073) into the shipped binaries,
whose notices then carry its MIT text and Bjoern Hoehrmann's copyright line
from its UTF-8 decoder. GGML (MIT, D-077) links so far into the backend
proof's tests only; a shipped binary that links it carries its MIT text
and the YaRN attribution from its RoPE kernel, and jitLLM's
`src/kernels/ggml/ggml_support.cu`, adapted from GGML, and
`src/kernels/ggml/fattn.cu`, which instantiates GGML's vector
flash-attention case and copies `launch_fattn`'s host arithmetic, keep
GGML's notice in their headers, as do the M3 units that instantiate or
copy from GGML (`fattn_mma.cu` with `fattn.cu`'s sparse-mask kernel,
`fattn_mma_d256.cu`, `fattn_mma_d512.cu`, `fattn_mma_shape.cuh`,
`mul_mat_q.cu`, `ops_ext.cu`), and the ports of llama.cpp's DeepSeek V4
graph and compressor plan (`src/kernels/ggml/dsv4_graph.cc`,
`src/model/dsv4.cc`; llama.cpp is MIT like GGML, under the same
copyright) and of its Qwen3.8 graph (`qwen4exp.cpp`), QSA block tables and
n-gram hash (`src/kernels/ggml/qwen38_graph.cc`, `src/model/qwen38.cc`).
jitLLM's own kernels for Qwen3.8's formats (`src/kernels/ggml/jitllm_ops.*`)
and its fusions of Qwen3.8's GGML nodes (`jitllm_fused.cu`, which repeats
their arithmetic without their code) and its importer (`docs/experiments/artifact-layout/modelopt_qwen38.py`,
which applies llama.cpp's converter's value-head order and norm rules
without its code) are Apache-2.0 only. The M3 widening compiles more of the kept
tree (the lock's `license.scope` lists it); every added file is under the
root MIT license with no header of its own (checked 2026-09-28; the NVFP4
MMQ instance unit Qwen3.8 added, `template-instances/mmq-instance-nvfp4.cu`,
likewise, checked the same day). Upstream's
`argsort.cu` and `top-k.cu` would include CUB directly, and jitLLM's patch
builds them without it (bitonic argsort; top-k's radix select, upstream's
HIP path). D-080 cleared libcu++ only. CUB's own headers in the SDK's
CCCL are BSD-3-Clause, Apache-2.0 or Apache-2.0 WITH LLVM-exception, but
what `argsort.cu` and `top-k.cu` would include also reaches two BSL-1.0
Thrust headers, `thrust/detail/preprocessor.h` and `type_deduction.h`
(checked 2026-09-28). The owner approved CUB with those headers on
2026-09-28 (D-091), adopting it only if a measurement shows it faster; it
is not faster where it matters, so the build still leaves it out. On `spark-b` (CUDA 13.4.92,
CCCL 3.4.3, `sm_121a`, idle GPU, the two builds of the same pinned files
in one scratch binary, median of 7), upstream's CUB top-k
(`DeviceTopK::MaxPairs` once per row) took 8.4–13.2 µs a call in a
captured graph for one row of 1,024 to 262,144 columns, against 18–41 µs
now (bitonic at 1,024, the radix select beyond). Across a decode step that is 0.4% of DeepSeek V4
Flash's 48 ms at 4,096 positions (21 CSA layers, top 512 of 1,024) and
0.8% at 256K, reaching 1.1% only at 1M; Qwen3.8's QSA top-k runs only past
2,051 cells and would save 0.3–0.8% of its 40 ms. At two rows it is even,
at four and more slower, and a 512-row prefill chunk's top-k is 3.6–71×
slower (DeepSeek, 4,096 positions: 5.0 ms against 0.19 ms a layer, +7% a
chunk; Qwen3.8, 8,192 cells: +10%). Argsort is the same bitonic kernel
either way for rows of at most 1,024 (MoE routing: 256 and 512 experts;
bitwise-identical output). The selections were equal as index sets on
distinct values and as value multisets under ties; which tied index is
chosen differs. ExLlamaV3's GEMM kernels (MIT) also link only into tests so
far; a shipped binary that links them carries ExLlamaV3's MIT text
([below](#exllamav3-gemm-kernels-in-the-core-m2)). CUTLASS 4.7.1
(BSD-3-Clause, lock component `cutlass`, headers only; M3's Qwen3.8
prefill) is included by two jitLLM units: `src/kernels/ggml/moe_cutlass.cu`
(Apache-2.0), which instantiates its SM120 block-scaled NVFP4 grouped GEMM
for the routed experts, and `jitllm_ops.cc`, which reads only
`cutlass/version.h` to pin the version the expert layout follows. Since
D-096 the runtime links the engine and so GGML's and CUTLASS's kernels:
both are `use: product` in the lock, and the package carries GGML's MIT
text (with the YaRN line from `rope.cu`) and CUTLASS's BSD-3-Clause text in
its third-party notices, and ships cuBLAS (D-076, the `cublas` unit
below). Every
kept header was checked for its BSD-3-Clause SPDX line and NVIDIA's
copyright (the lock's `license.evidence`); the CuTe DSL, under NVIDIA's
EULA, is not kept. The routed experts' activation quantization
(`src/kernels/ggml/jitllm_moe.cu`) copies GGML's
`nvfp4_native_scale_error` and repeats `quantize_mmq_nvfp4`'s scale
search, so that file keeps GGML's MIT notice.

| Unit (version) | Category | License | In a packaged binary |
| --- | --- | --- | --- |
| CMake 4.4.3, Ninja 1.13.2 | tool | BSD-3-Clause; Apache-2.0 | Nothing |
| Clang, LLD and the LLVM tools 22.1.8 | tool | Apache-2.0 WITH LLVM-exception (libz3: MIT) | Generated code |
| Clang's resource headers | platform (compiler support) | Apache-2.0 WITH LLVM-exception; `arm_neon.h` and `arm_fp16.h` carry an MIT text | Inline code and macros |
| compiler-rt | platform | Apache-2.0 WITH LLVM-exception | Nothing: sanitizer builds only; executables link libgcc, not its builtins |
| GCC 16.2.0 runtime (D-060) | platform | GPL-3.0-or-later WITH GCC-exception-3.1, with embedded code below | Every executable: libstdc++, libgcc, libgcc_eh, crtbeginS/crtendS. Not libatomic, which is not on the link line |
| glibc 2.39-0ubuntu8.9 (sysroot) | platform | LGPL-2.1-or-later | Start files and `libc_nonshared.a` members, under a linking exception; libc, libm and the loader stay dynamic |
| Linux UAPI headers 6.8.0-142.142 | platform | GPL-2.0-only WITH Linux-syscall-note | Constants and macros |
| CUDA 13.4.92 runtime and headers | platform | NVIDIA CUDA EULA | CUDA builds: `libcudart_static.a`, NVCC's host stubs and registration code, device code |
| CUDA driver link stub 13.4.92 (`cuda-driver-dev-13-4`) | platform | NVIDIA CUDA EULA | Nothing: CUDA builds need `libcuda.so.1`, which the driver supplies (D-072) |
| cuBLAS and cuBLASLt 13.8.0.4 (D-076) | platform | NVIDIA CUDA EULA | `jitllm-runtime` (CUDA builds, since D-096) links `libcublas.so.13` and `libcublasLt.so.13` dynamically; the package ships both unmodified and unstripped (EULA Attachment A, section 2.3) in `/usr/lib/jitllm`, which the runtime's run path names, carries the EULA notice and depends on `libgcc-s1`, which `libcublas.so.13` needs |
| CCCL 13.3.4.3.1 (libcu++, `nv/`) | platform | Apache-2.0 WITH LLVM-exception | Through CUDA headers such as `cuda_fp16.h` |
| NVCC, libNVVM, ptxas and the other CUDA tools | tool | NVIDIA CUDA EULA (internal use) | Generated code |
| REUSE lint 6.2.0 and nine wheels | tool | GPL-3.0-or-later and others (provenance.toml) | Nothing |
| GMP, MPFR, MPC, ISL and gettext (GCC's build inputs) | tool | LGPL-3.0-or-later (GMP also GPL-2.0-or-later), MIT, GPL-3.0-or-later | Nothing: none of their symbols are in the runtime |
| Host prerequisites, mise, Python | tool; host glibc is platform | Ubuntu package terms; MIT; PSF-2.0 | Nothing in a cross-built binary |

Embedded code in the static GCC runtime, as link maps show it (this
corrects and extends the [GCC 16.2 report](experiments/gcc16-static/README.md#provenance-and-use)):

- **Linked routinely.** HP and SGI STL code (headers, and `tree.o`,
  `list.o`); Ryu (Apache-2.0 OR BSL-1.0, taken under BSL-1.0) with *any*
  `<format>` or `<print>` use, not only floating point; libiberty's
  `cp-demangle.o` (GPL-2.0-or-later WITH GCC-exception-2.0) through the
  default terminate handler; glibc soft-fp comparison objects (LGPL-2.1 with
  a linking exception) on AArch64; and `<format>`'s tables generated from
  Unicode data.
- **Linked only when used.** fast_float (MIT) with floating-point
  `std::from_chars`; the tz database (public domain) with chrono time zones;
  IBM-HRL code with `ext/pb_ds`; Jeremy Siek's concept checks; `<barrier>`
  (Apache-2.0 WITH LLVM-exception). PSTL headers arrive through
  `<algorithm>`, `<memory>` and `<numeric>` but contribute code only with
  `<execution>`. libbacktrace never: the SDK has no `libstdc++exp.a`.

## What a packaged binary carries

For a CUDA-enabled arm64 `jitllm`, cross-built by the SDK:

- **Always:** jitLLM's `LICENSE` and `NOTICE`; the HP and SGI permission
  notices, which ask to appear in supporting documentation; a statement that
  the CUDA runtime and NVCC-generated code are under the NVIDIA CUDA EULA,
  not Apache-2.0; and the CUDA headers' Disclaimer and U.S. Government End
  Users Notice, which their text requires in user documentation. The SBOM
  must also identify the GCC 16.2.0 runtime and glibc.
- **When the code is used:** the Unicode notice (any `<format>` or `<print>`,
  or the tokenizer's tables;
  shipped by default unless the owner decides otherwise), fast_float's MIT
  notice, Norbert Juffa's and SoftFloat's notices for CUDA device math, the
  MIT text of Clang's NEON headers, the IBM-HRL and Siek notices, and
  glibc's 4.4BSD notice for BSD header macros.
- **Not needed:** Ryu, cp-demangle, soft-fp, the glibc start files and
  `libc_nonshared.a`, the tz database, PSTL, `<barrier>`, libcu++, Clang's
  other headers and the kernel headers. Their exceptions or terms ask
  nothing of object code.

No source offer is owed for such a binary. The EULA lists
`libcudart_static.a` as redistributable (Attachment A) inside an
application with material additional functionality. `cudart_static.o`'s
`.comment` names GCC 8.3.0, and it references no C++ runtime, consistent with
the GCC exception's eligible compilation process. NVIDIA's build process
itself is unverified.

Three constraints follow:

- **Never publish the SDK**, its caches or a reference image populated with
  it. The CUDA tools are for internal use only under the EULA, the runtime
  archives would owe GPL and LGPL source, and `cmake-gui` statically links
  LGPL-3.0 Qt.
- **Release packages come from the `cross` profile.** `spark-native` links
  the Spark's own, unpinned `libc6-dev` and GNU linker.
- **CUB or Thrust included directly is incorporated implementation**, not
  platform. Their licenses are permissive, so D-091 admits them (CUB by
  the owner's name), but `cccl`'s record must first name what is included
  and carry the notices that follow.

**How the package carries them** (D-074): `tools/jitllm_package.py` writes
`/usr/share/doc/jitllm/THIRD-PARTY-NOTICES` from the source lock's notices
and the text that each shipped unit's notices locate in
[provenance.toml](../toolchains/provenance.toml) (`extract`), including every
"when the code is used" notice and the whole CUDA EULA, whether or not the
build reaches the code; a Debian `copyright` file; and an SPDX 2.3 SBOM
naming the GCC 16.2.0 runtime, glibc and every other shipped unit. The
statement that NVIDIA code is under the EULA opens the notices.

**Decisions** (the owner's answers on 2026-09-24, for the Package item;
D-074):

1. jitLLM's CUDA sources do not carry the CUDA header notice; it ships in
   the documentation (`THIRD-PARTY-NOTICES`), which the headers' text also
   asks for. The headers ask for it "in the user documentation and internal
   comments to the code", and jitLLM's `.cu` files are its own code.
2. The package ships the CUDA EULA's full text, and its `copyright` file
   names the NVIDIA code under `LicenseRef-NVIDIA-CUDA-EULA`. The reading
   that static linking and stripping do not "modify" the runtime object
   code (EULA §2.3) is an interpretation, not confirmed with NVIDIA.
3. The LGPL-2.1 §5 reading for glibc stands. The start files and
   `libc_nonshared.a` carry the linking exception. `csu/init.c` has none but
   contributes one constant, `_IO_stdin_used`. A §6 reading would clash with
   the EULA's ban on reverse engineering the embedded runtime.
4. The Unicode notice ships. The [first dense slice](#first-dense-slice-d-051)
   still treats Unicode-derived tables it would incorporate as needing their
   own provenance.
5. Clang's resource headers and CCCL reached through CUDA headers belong to
   D-017's platform family, as recorded, although D-017 does not name
   compiler headers or `Apache-2.0 WITH LLVM-exception`. This also covers
   libcu++ included directly by incorporated code (D-080).
6. The evidence above that `libcudart_static.a` meets the GCC exception's
   eligibility test is accepted, as D-060 asked.
7. Code compiled from CUDA headers that Attachment A does not list ships as
   object code under EULA §1.1.1's "as incorporated in object code format".
   Their own notice prohibits reproducing or disclosing them to third
   parties "notwithstanding" the EULA, and Attachment A names only some
   headers (the fp16, bf16 and fp8 family, `cuda_occupancy.h` and the
   runtime-compilation set). The inline code of `cuda_runtime.h`, the
   host-stub headers NVCC uses and the device math headers reach the binary
   only as object code. The reading is not confirmed with NVIDIA.

**Method.** Link probes cross-built with the SDK's `cross` flags (`base`
using `<vector>`, `<algorithm>`, `<memory>`, `<string>`, `<format>`,
`<print>`, `<expected>`, `<span>`, `<atomic>`, `<thread>`, `<mutex>` and
`<chrono>`; `double` printing; floating-point `from_chars`; time zones), plus
a relink of the cross-built CUDA contract test with the build's own link
line, all with lld link maps and `--why-extract`. License texts come from
the SDK's packages and files, from glibc's sources at `glibc-2.39` (Ubuntu's
`libc6` copyright file is stale at 2.23 and omits the start files'
exception), from the GCC 16.2.0 tarball, from the CUDA EULA (updated
2026-01-26; the packaged copy matches NVIDIA's page) and from SPDX License
List 3.29.0. The HP and SGI notices match `HPND-sell-variant` by manual
comparison; the IBM-HRL and Siek notices match no SPDX identifier. The
probes and maps stay outside the repository.

## First dense slice (D-051)

The [selection](first-slice.md) pins the official Qwen2.5-0.5B-Instruct FP16
GGUF, its base-metadata cross-check and the existing llama.cpp reference.
[Artifact/tool pins](experiments/first-slice/pins.json) and the
[bounded source-unit inventory](experiments/first-slice/source-audit.json)
record immutable revisions, file hashes, actual notices and adoption status.
Seventeen inspected upstream file hashes matched the immutable GitHub source.
The inventory is not clearance of the future compiled dependency closure.

| Unit and use | Category / observed terms / disposition |
| --- | --- |
| Official Qwen GGUF and base metadata/tokenizer files | External model/test data; both repositories supply the same Apache-2.0 license file. No weights redistributed; retain source terms and required notices with future derived artifacts. Exact source-to-GGUF conversion lineage is unverified |
| Pinned llama.cpp executable/libraries and GGUF Python inspector | External reference/inspection tools, root MIT and gguf-py MIT; actual container/runtime terms remain in the reference setup record. Used only outside jitLLM serving |
| Selected GGML core, CPU/CUDA kernels; Qwen2 graph/tensor semantics; native GGUF reader | Future core implementation candidates, root MIT plus local MIT notices (including Mozilla llamafile SGEMM and YaRN authors). Preserve notices and audit the selected compiled closure before adoption. GGML allocator/workspace behavior still needs the M2 proof |
| Tokenizer implementation and generated Unicode tables | Root MIT implementation is not a blanket grant for derived data. `src/unicode-data.cpp`'s provenance was established on 2026-09-28: UCD 15.1.0 ([below](#tokenizer-unicode-tables-m3)). Not incorporated: jitLLM's tokenizer generates its own tables from the pinned UCD files, admitted to the core by D-088 (accepted 2026-09-28) |
| Chat-template rendering | The pinned template is model data under its source terms. Future owned native rendering must pass exact byte/token fixtures for enabled branches. Full `common/jinja`, chat/parser and vendor closure is not adopted or cleared |
| HF converter and Python package closure | Inspected only; neither executed nor incorporated. The split converter has remote-code/legacy-checkpoint paths outside the selected model path. Any future use needs independently pinned/audited tools, allowlisted data formats and remote code disabled |
| CUDA, driver and standard runtimes; NumPy/GGUF Python packages | D-017 platform dependencies and external experiment tools, respectively; retain exact image/component identities and their own terms. No new platform exception or source dependency is approved |

The current [Unicode license](https://www.unicode.org/license.txt) is Unicode
License V3 with notice requirements. D-017 did not list that license;
D-088, which the owner accepted on 2026-09-28, amends its core allowlist to
admit Unicode-3.0 for data derived from Unicode's data files, not code
([below](#tokenizer-unicode-tables-m3)); D-091 has since admitted every
permissive license, which subsumes it.
No whole-vendor-tree, complete-image or redistribution clearance follows from
root MIT.

## Tokenizer Unicode tables (M3)

The provenance record for the native tokenizer's Unicode data (D-017's
fields; checked 2026-09-28). D-088, which the owner accepted on 2026-09-28
("Unicode license accepted"), admits the Unicode License V3 (SPDX
`Unicode-3.0`) to D-017's core allowlist for data: the generated tables
below, not code. D-091 (2026-09-28), which admits every permissive
license, subsumes that admission; the record below stands.

**llama.cpp's tables, traced.** At the pinned `b29c606e`,
`src/unicode-data.cpp` says only "generated with
scripts/gen-unicode-data.py", which fetches
`https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt` and calls
Python's `unicodedata` for the NFD table. Its tables last changed in
llama.cpp commits `b43272af` (2024-05-17) and `37bef894` (2024-06-18); later
commits changed container types and a comment only. Rerunning the
generator's logic on UCD 15.0.0, 15.1.0, 16.0.0 and 17.0.0 (fetched from
unicode.org on spark-b; 15.1.0's files are hashed below) reproduces its category-flag
ranges (2,273), lowercase (1,433) and uppercase (1,450) maps from 15.1.0
only; 15.0.0 differs in 59 flag ranges, 16.0.0 and 17.0.0 in hundreds. Its
whitespace set equals PropList.txt's White_Space, and its NFD table equals
what 15.0.0 and 15.1.0 give alike, so it does not say which Python made it.
The tables are therefore UCD 15.1.0 data.

**What jitLLM uses instead.** `tools/gen-unicode-tables` (jitLLM's,
Apache-2.0; a developer-run build tool) generates
`src/tokenizer/unicode_data.cc` from three UCD 15.1.0 files and checks each:

| File | SHA-256 |
| --- | --- |
| UnicodeData.txt | `2fc713e6a31a87c4850a37fe2caffa4218180fadb5de86b43a143ddb4581fb86` |
| PropList.txt | `05672956317b6296bc2ec3d6cef1f6452b57ff4f2efc6dc55b0a19277d5fcfd1` |
| DerivedNormalizationProps.txt | `8875dccee2bc1a7c1fe568a3b502a9e78c9e0495afd96b6568b4294d0ed1f7e1` |
| NormalizationTest.txt (tests only) | `871238e37e3be0696ec2bd0891119a041b052da1a84485eda05a5438724b223e` |

| Field | Record |
| --- | --- |
| Role | Incorporated data: general categories, White_Space, NFC quick-check values, combining classes, canonical decompositions and compositions, compiled into `jitllm_tokenizer` |
| Version | Unicode Character Database 15.1.0 (release 2023-09) |
| License | Unicode License V3 (SPDX `Unicode-3.0`; [LICENSES/Unicode-3.0.txt](../LICENSES/Unicode-3.0.txt)). The files point to unicode.org's terms of use, which (read 2026-09-28) place every data file under `/Public/` under that license |
| In a binary | None shipped yet: `jitllm` and `jitllm-runtime` do not link `jitllm_tokenizer` (the swap runner and the chat route will). Tests and harnesses link it |
| Obligations | The copyright and permission notice, with the data or in associated documentation; no source offer. The repository carries it: [NOTICE](../NOTICE) names the file, its UCD version and Unicode's copyright, and [LICENSES/Unicode-3.0.txt](../LICENSES/Unicode-3.0.txt) holds the text. Every package carries it already: `THIRD-PARTY-NOTICES` includes the notice's full text ([provenance.toml](../toolchains/provenance.toml)'s `unicode` notice, `packaging/notices/Unicode-3.0.txt`, the same bytes as the LICENSES copy) whenever the GCC runtime ships, which is always (decision 4 above), and the package installs NOTICE unchanged. When a packaged executable is built from `unicode_data.cc`, `tools/jitllm_package.py` also lists the data: `Unicode-3.0` in the copyright file's license and a Unicode Character Database 15.1.0 entry in the SBOM, its notice under its own heading in `THIRD-PARTY-NOTICES`; the package check requires them. Packaging stops if a file under `src/` declares a license outside the core's (`CORE_LICENSES`, D-091), one no source-lock component or shipped provenance unit declares, or Unicode-3.0, which only this record lists, without a record in the tool's `IN_TREE_UNITS`, if a recorded file is gone, or if Ninja cannot list the executables' inputs |
| Source file | `src/tokenizer/unicode_data.cc` declares `Apache-2.0 AND Unicode-3.0` and "1991-2023 Unicode, Inc."; [NOTICE](../NOTICE) names it |

**Chat templates' text.** The renderers write the format text their
templates print: DeepSeek's tool, reasoning-effort and DSML strings (from
`encoding_dsv4.py`, MIT, which Unsloth's template, declared Apache-2.0,
repeats) and Qwen3.8's tool and reasoning-effort instructions (from the
NVFP4 checkpoint's `chat_template.jinja`; the card declares Apache-2.0, the
base model the Qwen Community License 1.0). These are format constants a
compatible prompt must contain, recorded here as model-format data, like
the special-token strings. The owner confirmed that reading on 2026-09-28:
chat-template text is model data, so the renderers' embedded template
wording follows D-087's rule that model licenses are informational and
gate nothing.

**Reference tools** (run, never incorporated;
[tokenizer-reference](experiments/tokenizer-reference/README.md)): the
pinned llama.cpp image (MIT, as above), Hugging Face tokenizers 0.22.2 and
transformers 5.12.1 (Apache-2.0), Jinja2 3.1.6 (BSD-3-Clause), and DeepSeek's
`encoding_dsv4.py` at `7872f01b` (MIT). The sampling tests use Random123's
published Philox known-answer vectors (values, cited in the test).

## Early EXL3 companion (D-052)

The [bring-up contract](exl3-bringup.md) and
[artifact pins](experiments/exl3-reference/pins.json) select two small
third-party Qwen EXL3 conversions for M2. Their downloaded Apache-2.0 license
files match the base Qwen license. The [Spark baseline](experiments/exl3-reference/README.md)
verifies full weight digests, execution and tokenizer/template identity; exact
conversion/calibration lineage remains unknown. These are external model/test
data; no weights are redistributed. Any new quantization job needs cleared
calibration inputs and pinned converter/input provenance.

The external reference uses the ExLlamaV3 revision below, the pinned container
recipe, three additional hash-pinned wheels and recorded compiler/runtime
identities. Its [runtime inventory](experiments/exl3-reference/runtime.json)
records compiled translation units, loaded library hashes and package notice
identities. An ARM host-helper patch includes MIT source context and retains
Turboderp's notice in [UPSTREAM-NOTICE.txt](experiments/exl3-reference/UPSTREAM-NOTICE.txt).
It disables x86 CPU MoE/collective helpers; no device kernels are changed.
No MiaAI patches or calibration corpus are used by this reference.

Selected MIT device kernels are core-eligible implementation candidates
once their native selected-file and compiled-closure audit is recorded and
the owner accepts it; the GEMM kernels' audit is
[below](#exllamav3-gemm-kernels-in-the-core-m2), accepted on 2026-09-27
(D-080). The external
PyTorch/driver/toolkit stack is a declared reference tool/platform dependency,
not incorporated native implementation or clearance to redistribute its
whole container. Native wrappers must remove upstream allocator, stream and
scheduling ownership; their actual closure and notices must be audited anew.

**Resolved provenance question: the GEMV kernel.** On 2026-09-27 the owner
judged it not copyleft ("Sounds like GEMV isn't actually viral", D-080): it
is MIT-declared, and the scan below found no run of 20 or more tokens
shared with QTIP beyond identical PTX `mma` operand strings. The dense
GEMV kernel, its host wrapper and `exl3_gemv_half_inst.cu` are treated as
MIT and core-eligible, and the gate below closed on that judgment without
the structural comparison. It reopens if evidence of copied GPL code
appears. The record that led there follows.

At the pinned revision,
[`quant/exl3_gemv_kernel.cuh`](https://github.com/turboderp-org/exllamav3/blob/6b84a21b6f1e5da3f291b9e1019061f0de788279/exllamav3/exllamav3_ext/quant/exl3_gemv_kernel.cuh#L3-L4)
describes its small-m path as a "QTIP-style structure". It cites
Cornell-RelaxML/qtip `qtip-kernels/src/inference.cu`.

- The file carries no notice of its own. The same path, with the same QTIP
  reference, is among the nine headers GLM TP3 vendors from upstream
  `02aef45c` (below). That earlier version differs from the pinned file.
- QTIP's repository `LICENSE` at
  [`e90c6688c8dfae326a3a81b5eb032db7c6680ec0`](https://github.com/Cornell-RelaxML/qtip/tree/e90c6688c8dfae326a3a81b5eb032db7c6680ec0)
  (head on 2026-09-22) is the GPL-3.0 text. The cited file exists there.
- ExLlamaV3's README says EXL3 is based on QTIP and calls it a streamlined
  variant of QTIP.

**What the review scan found.** On 2026-09-22 a review compared both GEMV
versions with QTIP's kernel sources at that revision, by token sequence.
Apart from identical PTX `mma` operand strings, it found no shared run of
20 or more tokens. That is an observation, not the structural comparison
the gate asked for, and not clearance. This inventory draws no conclusion about
derivation.

**The same gate covers related files.** It applies to any file that
includes this header, directly or through another header, or follows its
body:

- upstream `quant/exl3_gemv.cu`, the host wrapper, which includes the header
  and creates its variants;
- upstream `quant/exl3_moe_coop_kernel.cuh`, and the files that include it
  (`quant/exl3_moe_coop.cu`, `comp_units/exl3_moe_coop_inst_*.cu`);
- upstream `comp_units/exl3_gemv_half_inst.cu`;
- the GLM, GLM TP3 and DeepSeek `cooperative_moe_kernel.cuh` derivatives.

The gate asked, before any of these entered a core-eligible module, for a
structural comparison with the cited file, its record, and the owner's
D-017 disposition. The owner's judgment (D-080) settles the dense GEMV
files; D-079's optional GEMV module is not created. The upstream MoE
cooperative files lose the QTIP question they inherited but still need
their own audit, and the MiaAI derivatives keep their mixed provenance
([below](#file-specific-declarations-and-mixed-provenance)).

**Each selected kernel still needs its per-file audit.** None of them
cites QTIP or another third-party source in its own text. The remaining
QTIP comments sit in host files that dispatch to this path (`exl3_gemm.cu`,
`exl3_gemv.cuh`). The kernels' audit is recorded
([below](#exllamav3-gemm-kernels-in-the-core-m2)): the GEMM kernels' in P1,
accepted, and the GEMV, reconstruction, Hadamard and bias-add kernels' in
P3, when they were ported.

## ExLlamaV3 GEMM kernels in the core (M2)

The source lock's `exllamav3` component brings ExLlamaV3's kernels for the
native EXL3 linear into the core build (D-017 incorporated implementation,
core tier, MIT), as GGML entered (D-077): GitHub's archive of the reference
revision [`6b84a21b`][exl], hash-pinned, narrowed by `archive.keep`, with
three reviewed patches. Checked **2026-09-27**: the GEMM kernels in
backend-proof P1, the GEMV, reconstruction, Hadamard and bias-add kernels
in P3.

**Status: cleared for the core** by this audit (owner, 2026-09-27, D-080;
the answers close this section). The files P3 added are classified by the
same rule: none states a license of its own, so the root MIT `LICENSE`
governs; the GEMV header's QTIP citation is the implication D-080 judged
not copyleft. Only tests and a benchmark link them so far: the lock marks
the component `use: test`, and `sources.closure` refuses a packaged
executable built from one.

- **What is compiled.** In CUDA profiles only:
  - upstream's compilation units for the mcg codebook at K = 4, 5, 6 and
    8, the M2 fixtures' rates, unchanged. They instantiate
    `exl3_gemm_kernel` and `exl3_mgemm_kernel` for FP16 and FP32 outputs
    at tile shapes 1 to 4;
  - jitLLM's instance unit (`jitllm/jitllm_exl3_kernels.cu`, patch 0002,
    Apache-2.0), which includes the GEMV kernel's header and the three
    sources patch 0003 reduces to their kernels, and instantiates what the
    native linear launches: the K = 4 mcg GEMV (as upstream's
    `exl3_gemv_select_kernel` does), both reconstruction kernels at the
    four rates, three Hadamard variants, and `add.cu`'s kernels as that
    file defines them.

  So far only test and benchmark executables link them. In every profile
  jitLLM includes `exl3_devctx.cuh` for the device context's sizes.
- **What is kept.** Exactly the files those units include (23 files,
  5,087 lines at the pin), and the root `LICENSE`. A test follows the
  includes of the prepared tree and fails if `keep` holds more or less
  (`tools/tests/test_sources.py`).
- **What is not kept.** Everything else:
  - the Python package, `setup.py` and `ext.py`;
  - the ATen host wrappers that remain whole (`exl3_gemm.cu`, which
    carries a QTIP comment, `exl3_kernel_map.cu`, `exl3_devctx.cu`,
    `exl3_gemv.cu`, `hgemm.cu`) and the headers declaring them, and
    `bits_k.cuh` with its c10 include. jitLLM's launchers
    (`src/kernels/exl3/`) replace them; the GEMV choice is a recorded
    copy (`upstream_gemv.cc`, MIT AND Apache-2.0);
  - other codebooks and rates, the half-integer GEMV instances and the
    int8 GEMV;
  - the MoE kernels and their cooperative derivatives;
  - the quantizer and the calibration corpora.

**Per-file record.** Paths are under `exllamav3/exllamav3_ext/` except
`LICENSE`; the SHA-256 prefix is of the upstream bytes, identical to the
commit's Git blobs. No file carries a license or copyright line of its own:
all fall under the root MIT `LICENSE` (Copyright (c) 2025 Turboderp). None
cites QTIP or any other source; the only links are to NVIDIA's PTX
documentation.

| File | SHA-256 | Holds |
| --- | --- | --- |
| `LICENSE` | `27a32b6263fcd96c` | MIT, Copyright (c) 2025 Turboderp |
| `quant/comp_units/exl3_comp_unit_{4,5,6,8}_cb1.cu` | `d7b2bd80d427c6f0`, `07c53ef89d7c12a4`, `564e1a8847568bdf`, `44497b3d7f670d94` | One macro each, instantiating the tables of kernels for its rate and cb 1 |
| `quant/comp_units/exl3_comp_unit_{4,5,6,8}.cuh` | `313d479451fe5ecb`, `f377ef4dd47edca5`, `11497c8618830b9e`, `3354be2f6db645bf` | The tables' `extern` declarations |
| `quant/exl3_gemm_kernel.cuh` | `3e94f9e1f3acb0dd` | The GEMM and multi-GEMM kernels: input Hadamard, tile loop, grid syncs |
| `quant/exl3_gemm_inner.cuh` | `c50147d165a72627` | The tiled main loop: pipelined loads, dequantization, MMA, split-K reduction. **Patched** (0001) |
| `quant/exl3_dq.cuh` | `4e48a37af4811e8e` | Trellis bit extraction and dequantization to MMA fragments |
| `quant/codebook.cuh` | `0e3c63b323f8d3cc` | The procedural codebooks (3INST, mcg, mul1) |
| `quant/hadamard_inner.cuh` | `8d8e437aced88735` | 128-point Hadamard transforms by warp shuffle, and activations |
| `quant/exl3_kernel_map.cuh` | `91fa4be2b7b23cc3` | Kernel signatures, tile shapes and table macros; host selector declarations |
| `quant/exl3_devctx.cuh` | `effb1827e9b6c61b` | Lock-area and workspace sizes; the device-context class, declared only |
| `ptx.cuh` | `8ceacb1b321af587` | Inline PTX for MMA, `ldmatrix`, `cp.async`, barriers; a group barrier on libcu++'s `cuda::atomic_ref` |
| `util.cuh` | `1907cea115260db7` | Half vector types and helpers; exiting CUDA and cuBLAS error checks. **Patched** (0001) |
| `util.h` | `ba89ac6793bf31cf` | Min/max and debug macros; TORCH_CHECK wrappers, unused |
| `compat.cuh` | `dd6038fa6eb8b28d` | `tanh` approximation |
| `quant/exl3_gemv_kernel.cuh` | `cf03005caa04e5ad` | The small-m GEMV kernel (P3). Its header comment calls the structure "QTIP-style" and names QTIP's `inference.cu`: an implication, not a license statement (D-080) |
| `quant/reconstruct.cu` | `3541a9e4bfa2fec9` | The reconstruction kernels, rotated and with both Hadamard transforms (P3). **Patched** (0003): ATen wrappers and instance tables removed, two `register`s dropped |
| `quant/hadamard.cu` | `7df512a2d5c3c2ce` | The 128-point Hadamard kernels (P3). **Patched** (0003): ATen wrappers removed |
| `add.cu` | `d89730c6154be633` | Element-wise adds, `add_kernel_hhh` among them, and two MoE bias adds (P3). **Patched** (0003): ATen wrappers removed |

**Patches** (`third_party/patches/exllamav3/`, MIT AND Apache-2.0 for
0001, which carries upstream context; Apache-2.0 for 0002):
- `0001` removes `util.cuh`'s `cuda_check`, `gpu_assert`,
  `cublas_check` and `cublas_assert`, which print and call `exit()`
  (D-066). They are inline code in a header, so a guard macro could not
  keep them out of a unit that includes it without the macro; a tooling
  test refuses any kept file that can end the process (host exits, aborts
  and asserts; device traps and asserts). It also drops a no-op `register` from four
  arrays in `exl3_gemm_inner.cuh`: NVCC rejects it with a Clang host compiler.
- `0002` adds `jitllm/CMakeLists.txt`, jitLLM's build of the units, and
  `jitllm/jitllm_exl3_kernels.cu` and `.h`, jitLLM's instance unit and its
  lookups.
- `0003` (MIT AND Apache-2.0) reduces `quant/reconstruct.cu`,
  `quant/hadamard.cu` and `add.cu` to their kernels: it removes their ATen,
  c10 and PyTorch-API includes, their host wrappers (which use ATen tensors
  and PyTorch's stream and end the process on a CUDA error) and
  `reconstruct.cu`'s tables instantiating every rate and codebook, and drops
  two no-op `register`s. The kernels' text is unchanged.

Every ExLlamaV3 function the build holds has SASS identical to its
namesake in the P0 reference's NVCC 13.4.92 extension, the 27 the P3
per-linear sweep launches among them
([report](experiments/backend-proof-p3/README.md#sass)).

**Comparison with QTIP and the GEMV kernel.** A token-run comparison (C
tokens, comments removed) set every kept file against QTIP's kernel sources
at `e90c6688` (`qtip-kernels/src/`: `inference.cu`, `inference.h`,
`qtip_torch.cu`, `test.cu`, `wrapper.cpp`) and against the pinned
`exl3_gemv_kernel.cuh` and `exl3_moe_coop_kernel.cuh`:
- **QTIP.** No shared run is longer than 11 tokens (include lists, loop
  headers, `asm volatile` operand lists), except one: `util.cuh`'s
  `gpu_assert` shares 32 tokens with QTIP's `gpuAssert` in `inference.h`.
  Both copy the common CUDA error-check idiom posted on Stack Overflow.
  Patch 0001 removes it. The four files P3 added, patched, share at most
  10 tokens with any QTIP source (an include list or a loop header; the
  GEMV kernel at most 7, a `reinterpret_cast`).
- **The GEMV kernel.** `exl3_gemm_kernel.cuh` shares a 103-token input
  Hadamard prologue with it, `exl3_dq.cuh` 91 tokens of bit extraction and
  `codebook.cuh` 62 tokens of mcg decoding. Upstream's history gives the
  direction:
  - each run is already in the kept file's version before the GEMV
    kernel's: the prologue at `df1a9690` (2026-05-01), the others at
    `0f2da5d6` (2025-10-12);
  - the GEMV kernel arrived at `377c8423` (2025-10-22) and was rewritten
    at `485fa6c0` (2026-07-08), and neither commit changed a file of the
    GEMM units;
  - two later commits changed both: `405028b7` added a missing `break;` to
    `hadamard_inner.cuh`, and `07b8a2e0` added half-integer rates to every
    kernel. Neither carries GEMV code into a GEMM unit's file.

  So the GEMV kernel reuses the GEMM kernels' code, not the reverse. None
  of the GEMM units' files follows the GEMV kernel's body. (The GEMV
  kernel itself is kept since P3; this comparison was made for the P1
  audit, before it was.)
- **The gated MoE kernel.** `exl3_moe_coop_kernel.cuh`, which includes
  the GEMV kernel, shares a 66-token Hadamard butterfly with
  `hadamard_inner.cuh`. The run is already in `hadamard_inner.cuh` before
  the MoE kernel arrived (`58d4d73`, 2026-09-13). The only commit that
  changed the MoE kernel and a kept file is `07b8a2e0` above, which leaves
  `hadamard_inner.cuh` alone.

That is an observation, not the structural comparison the GEMV gate
asked for, and not a finding that EXL3's format or GEMM owe nothing to
QTIP's ideas. Ideas are not what D-017 audits; incorporated code is.

**Platform code the kernels reach.** Through NVCC: CUDA's `cuda_fp16.h`,
`cuda_bf16.h`, `cooperative_groups.h` and `cublas_v2.h` (declarations only;
nothing links cuBLAS). `ptx.cuh` also includes libcu++'s `<cuda/atomic>`
directly, and the multi-GEMM kernel's group barrier compiles its
`cuda::atomic_ref` into device code. libcu++ is CCCL, recorded as platform
code under Apache-2.0 WITH LLVM-exception; decision 5 above covers it,
reached through CUDA headers or included directly (D-080).

**Obligations.** None while only tests link the kernels. A shipped binary
that links them carries ExLlamaV3's MIT
text in its third-party notices (the lock's `notices`, with `use` set to
`product`). No source offer is owed.

**The owner's answers** (2026-09-27, D-080: "It all sounds fine and
permissive with no viral code issues"):
1. This audit (the per-file record and the comparisons above) clears the
   GEMM kernels for the core; no structural comparison with QTIP is needed.
2. Decision 5's CCCL classification also covers libcu++ included directly
   by incorporated code, as `ptx.cuh` does: permissive either way.

## Reference instrumentation

The [fused-routes experiment](experiments/fused-routes/README.md) reads MoE
routes from the unmodified pinned llama.cpp image with jitLLM-authored
Apache-2.0 harness code. It retains `route-outputs.patch`, a rejected
libllama change with MIT upstream context and The ggml authors' notice in
`UPSTREAM-NOTICE.txt`, only to reproduce a negative result. The runner does
not build it, and nothing from it enters jitLLM.

## Additional reference candidates

The [Qwen-Image GGUF study](experiments/image-gguf/README.md) runs
stable-diffusion.cpp and its patched GGML fork (both MIT) as external
reference tools built from pinned sources; the GGUFs and Comfy-Org files
remain under the Qwen Research License, like the BF16 study. The
[MiMo reference](experiments/mimo-reference/README.md) uses the MIT-declared
checkpoint, the pinned SGLang (Apache-2.0) image and hash-pinned `torchcodec`
0.16.0 (BSD-3-Clause, Meta), plus one audited checkpoint configuration file
executed for config parsing only. The AGPL-3.0 MiaAI MiMo recipe was read as
documentation; none of its code or patches ran or entered this repository.
No weights are redistributed, and neither study clears a container's full
component closure or approves any of these for jitLLM's core.

## Fast-swap models and baselines (M3, M4)

D-087's models and baselines. M3's are pinned and audited below (checked
**2026-09-28**); M4's rows [at the end](#still-owed) are still owed.
"Cleared" means confirmed from the license texts themselves; a declared
license is recorded as declared, and D-080's rule separates a project's
own code from included files that state a different license for
themselves. Nothing here is incorporated into jitLLM: running a recipe
or engine as a baseline incorporates nothing, and any reuse of code is a
later decision with its own per-file record.

**Model weights are informational (owner, 2026-09-28; D-087).** jitLLM
never distributes model weights: users supply checkpoints, and artifacts
are made locally from them. A weight license therefore blocks no import,
execution, baseline or support; the rows below record it for information,
and the support matrix may note it for users. Code licenses (kernels,
runtimes, and scripts or recipes we incorporate or run) still follow
D-002, D-017, D-080 and D-091.

### M3 pins

Every file's size and SHA-256 (the LFS digest, or the hash of the bytes
for small files, with their Git blob) and the engines' digests are in
[experiments/fast-swap/pins.json](experiments/fast-swap/pins.json),
resolved from the Hugging Face, GitHub and registry APIs on 2026-09-28.

| Item | Pin | License (weights: informational) |
| --- | --- | --- |
| DeepSeek V4 Flash 0731 GGUF, UD-Q2_K_XL, 3 shards (96,832,508,352 B) and the DSpark drafter `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` (10,896,057,440 B) | `unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93fb787c21338159b0af3318bb3f4d9768`; base `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b` | Card MIT; the base's `LICENSE` is the MIT text, SHA-256 `f2c6c602…`, the same bytes as V4 Flash's in `full-pins.json`. Replaces `e3aa0d6a`, which stays on the Sparks |
| Qwen3.8 Flash Next NVFP4, 34 shards plus index, scales and tokenizer (105,935,742,983 B) | `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6c14c6c9442ef83e8f05b5a3c39304f69` | Card `apache-2.0`. Its README says it is a mirror of `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (card `other`; head `7c4f1bc1` on 2026-09-28), not Mia's quantization. The base `Qwen/Qwen3.8-Flash-Next@de4b8e4d` is under the Qwen Community License 1.0 (`a0dc4225…`); the base's terms are expected to govern (not a legal finding). The repo has no license file |
| Qwen-Image-2.1, BF16 diffusers pipeline (33,131,614,782 B) | `Qwen/Qwen-Image-2.1@790c92633540aa0cb11d9abf19eb46d861714758`, the revision the [BF16 study](experiments/image-reference/README.md) pinned and still the head | Qwen Research License (`LICENSE`, `8dc973f0…`), non-commercial research and evaluation |
| TensorFold's checkpoint, MLX affine 4-bit, group size 32 (`config.json`), with its MTP layer (113,233,046,214 B) | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa8066e3be900a0d148d0f5a2f4eb1cf6534` | `LICENSE` is the Qwen Community License 1.0, byte-identical to the base's (`a0dc4225…`) |
| MiaAI single-Spark recipe | `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark@b8439110eec0230facbe4ddf0dffe01b8f769be0` (main on 2026-09-28) | AGPL-3.0-or-later ([below](#miaai-single-spark-recipe)) |
| Its default engine | `vllm/vllm-openai:qwen38-flash-next`, index `sha256:fc120ece…`, arm64 `sha256:3b0e188f…`; built 2026-08-26 as a local build (its labels give no vLLM commit); FlashInfer 0.6.17, CUDA 13.0.1, NCCL 2.30.7 | vLLM Apache-2.0; the image's full closure is not audited |
| Its opt-in lane (`start-v030.sh`) | `vllm/vllm-openai:v0.30.0`, index `sha256:8a69ffad…`, arm64 `sha256:4864d466…`; vLLM `ced6857a` (tag v0.30.0); FlashInfer 0.6.18.post1, CUTLASS v4.7.1, CUDA 13.0.2. It serves `nvidia/Qwen3.8-Flash-Next-NVFP4`, not pinned: only if this lane becomes a baseline | As above |
| TensorFold | `ashhart/TensorFold@71377a5373ed7b394f1b480ba2a6a3986b03af1c` (0.3.6.2, main on 2026-09-28; moved from `beddbb7b`, 0.3.5.1, the same day); runs in `nvcr.io/nvidia/pytorch:26.07-py3`, index `sha256:2140e699…` | MIT ([below](#tensorfold)) |
| llama.cpp, the GGUF oracle | `b29c606e` (build 10964), the source lock's pin and M0's image `ghcr.io/ggml-org/llama.cpp@sha256:837fc732…`. It has `deepseek4`, `dflash` (DSpark) and `qwen4exp`; the drafter's card needs b10269 or newer | MIT (recorded under the first dense slice) |

vLLM or SGLang as DeepSeek's cross-quantization comparator is not pinned:
Mia's recipe uses only vLLM, and the SGLang image on the Sparks
(`lmsysorg/sglang@sha256:9e1fb4c3…`) is the MiMo reference's. That choice
waits for the baselines item.

### MiaAI single-Spark recipe

Pinned at `b8439110`: **70 tracked files** (79 tree entries with the 9
directories; the 2026-09-27 survey's "93" is not reproduced at this
commit). The root `LICENSE` is the AGPL-3.0 text, byte-identical to
gnu.org's `agpl-3.0.txt` (SHA-256 `0d96a4ff…`); the README grants
"AGPL-3.0-or-later" and says the license covers the repository's files but
not vLLM, the container image or the checkpoints they act on. Every path:

| Paths | Declaration |
| --- | --- |
| `LICENSE` | The AGPL-3.0 text |
| `download.sh`, `start.sh`, `stop.sh`; `bench/mixed.py`, `bench/structured.py`, `bench/sweep.py`; `files/build_draft_vocab.py`, `files/build_ple_packed_table.py`, `files/memwatch.sh`; `scripts/alert.sh`, `health-probe.sh`, `heartbeat.sh`, `launch-lane.sh`, `maintenance-relaunch.sh`, `memwatch-rotate.sh`, `smoke-test.sh`, `start-memwatch.sh`, `supervise.sh`; `systemd/qwen38-flash-heartbeat.timer`, `qwen38-flash-maintenance.service`, `qwen38-flash-maintenance.timer`, `qwen38-flash-supervisor-failure@.service`, `qwen38-flash-supervisor.service` (23) | an SPDX tag for `AGPL-3.0-or-later` in the file, most with "Copyright (C) 2026 MiaAI Lab" |
| The patch generators `files/patch_block_drop.py`, `patch_determinism.py`, `patch_modelopt_mxfp8.py`, `patch_mtp_draft_vocab.py`, `patch_mtp_draft_vocab_v030.py`, `patch_ple_layer.py`, `patch_ple_mmap_v030.py`, `patch_ple_offload.py`, `patch_qsa_fp8_kv.py`, `patch_qsa_fp8_kv_v030.py` (10) | The same AGPL-3.0-or-later header. They hold vLLM source fragments as the old side of old-to-new replacement pairs, and at run time write modified copies of files `start.sh` extracts from the image; per the README those copies keep vLLM's Apache-2.0 headers. `patch_qsa_fp8_kv.py` credits its FP8-KV approach to `lancelind/qwen3.8-Flash-DGX` (Apache-2.0) as "reimplemented"; that project was not compared. What they generate is a combined work, not audited |
| `start-v030.sh`, `files/mtp_block.py`, `files/build_draft_vocab_extend.py`, `bench/audit-spanish.py`, `bench/structured-protocol.py`, `bench/verify-smoke.py`, and the 10 files in `tests/` (16) | No notice in the file; the repository default, AGPL-3.0-or-later |
| `README.md`, `CHANGELOG.md`, the 5 files in `docs/`, the 6 in `.github/`, `.env.sample`, `.gitignore`, `files/sysctl-spark3.conf`, `files/draft_vocab_en_code_47k.txt`, `files/draft_vocab_es_en_code_65k.txt` (18) | No notice; the repository default. The draft vocabularies are token-ID lists |
| `files/chat-template/chat_template.jinja` | No notice in the file. `.env.sample` and `.gitignore` name it froggeric's v22.5 template (`hf.co/froggeric/Qwen-Fixed-Chat-Templates`), Apache-2.0: a declaration by reference, not a statement in the file. Used only when `CHAT_TEMPLATE` is set; the default is the checkpoint's own template |
| `files/smoke-vision-fixture.jpeg` | No license; a screenshot of the model's release-countdown page, used by `smoke-test.sh` |

**Disposition** (owner, 2026-09-28): the scripts may be **run unmodified
as a baseline recipe**. Running them locally conveys nothing and
incorporates nothing into jitLLM; the AGPL's section 13 applies to a
modified version offered over a network, which a local benchmark of the
unmodified recipe is not. No file here is a candidate for jitLLM's code;
any later reuse starts from vLLM's or another upstream's own sources.

### TensorFold

Pinned at `71377a53` (0.3.6.2): 540 tracked files, 58 commits after the
first pin `beddbb7b` (0.3.5.1, 371 files), with commits by ten outside
contributors besides the author and no separate terms stated for them. The root `LICENSE`
is the MIT text, "Copyright (c) 2026 TensorFold contributors", unchanged
(SHA-256 `be6a9ee4…`), and `pyproject.toml` declares MIT. No file carries
an SPDX tag; five EXL3 files name ExLlamaV3 in their first line (below).
`THIRD_PARTY_NOTICES.md` names what is adapted or vendored and, since
0.3.6, carries the MIT permission text; `LICENSES/Apache-2.0.txt` ships the
Apache text (unchanged):

| Paths | Declaration |
| --- | --- |
| `src/tensorfold/drafters/vendor/z_lab_dflash/model_mlx.py` | Vendored unmodified from `z-lab/dflash`, MIT, Copyright (c) 2026 Z Lab: stated in its directory's `README.md` and the notices, not in the file. Unchanged since `beddbb7b` |
| The n-gram ID helpers in `src/tensorfold/families/qwen4_exp/model.py` and `cuda/ngram.py` | Translated from transformers' `modeling_qwen4_exp.py`, Apache-2.0, Copyright 2026 The Qwen Team and The HuggingFace Inc. team (per the notices) |
| `src/tensorfold/kernels/qwen/dense/v1/lane_gdn.py`, `lane_tree.py` | Adapt mlx-lm's `qwen3_5` and `gated_delta` math, MIT, Apple (per the notices) |
| EXL3 (new since `beddbb7b`): the shared module `src/tensorfold/cuda/exl3/` (15 files: trellis decoders, the row-invariant linear, grouped experts), GLM's `families/glm5_next/cuda/exl3.{py,cu}`, `exl3_mm.py`, and the Qwen loaders `families/qwen3_5/cuda/exl3_load.py`, `families/qwen4_exp/cuda/exl3.py` | Read ExLlamaV3's EXL3 format (trellis layout and bitstream, the 3inst, mcg and mul1 codebooks, fragment order, the n-gram row codec), MIT, Copyright (c) 2025 Turboderp. The notices call them "separate implementations, checked bit for bit against ExLlamaV3's dequantization"; `exl3/decode.cuh`, `exl3/experts_grouped.cuh`, `exl3/format.py`, `glm5_next/cuda/exl3.cu` and `exl3.py` say "after ExLlamaV3" with that notice in their first lines |
| `src/tensorfold/families/qwen4_exp/cuda/kvcache.py`, `kvquant.py` (new) | The int8 and int4 KV caches follow ExLlamaV3's `-cq 8` / `-cq 4` scheme (MIT, Turboderp); per the notices the quantizer and dequant are TensorFold's own code |
| GLM on Apple Silicon (new; Metal, not the CUDA path): `families/glm5_next/` MLX engine, `kernels/glm/flash/v1/` | Per the notices: follows mlx-vlm's `glm5_next` (PR #2030) as vendored by oMLX (Apache-2.0), "nothing imported"; `kda.py` and `sparse_attention.py` ported from mlx-vlm PRs #2105 and #2245 (MIT, Copyright (c) 2025 Prince Canuma); the hyper-connection kernel repeats mlx-vlm's `deepseek_v4` one (MIT, Copyright (c) 2026 Apple Inc.); row kernels repeat MLX 0.32's arithmetic (MIT, Apple) |
| Everything else | TensorFold's MIT. The notices say the CUDA DeltaNet kernel follows flash-linear-attention's numerics and the NCCL wrapper vLLM's stream convention, "without copying"; the GLM engine follows Mia's recipe "without including recipe code". The Mac-only `ssd` extra builds a small MLX extension with nanobind 2.15.0 (not on the CUDA path) |

**Status: cleared** to run as a baseline (at `beddbb7b`; the terms at
`71377a53` are the same MIT, so the owner's clearance is carried, not
re-granted), and its own code is core-eligible (MIT) if ever ported, with
the Apache-2.0 and MIT notices above kept for the files they cover. A port
of the EXL3 or KV-cache code would keep ExLlamaV3's MIT notice (Turboderp)
beside TensorFold's; a port of the Metal GLM kernels, mlx-vlm's and
Apple's. Its runtime (PyTorch, Triton) comes
from NVIDIA's container and is not audited. Its Qwen3.8 Flash Next family
reads groups of 32 (`families/qwen4_exp/cuda/qmm.py`), matching the
checkpoint; the 64 in [tensorfold-assessment.md](tensorfold-assessment.md)
is the dense Qwen3.8-27B family's.

### Mia's NVFP4 and MXFP8 path: vLLM, FlashInfer and CUTLASS

Identified at the v0.30 lane's pins (vLLM `ced6857a`, FlashInfer v0.6.17
`a0a6b019` and v0.6.18.post1 `8bc3b578`, CUTLASS v4.7.1 `cb424739`). The
default image, pulled on 2026-09-28, reports vLLM `0.1.dev20073+g8e685d198`
(commit `8e685d198`, abbreviated; not audited at that commit). Its engine
log on the GB10 selects vLLM's Marlin weight-only kernel for the NVFP4
linear layers (`MarlinNvFp4LinearKernel`; the drafter's MoE also uses
Marlin), FlashInfer's CUTLASS kernels for the MXFP8 linear layers and the
routed-expert NVFP4 MoE, and Triton/FLA and CUDA Gated DeltaNet kernels
([baselines](experiments/fast-swap/baselines.md#qwen38-flash-next-mias-vllm-nvfp4)).
Marlin's license is not recorded here yet. No reuse decision is made here.

| Part | What it does | License, as found |
| --- | --- | --- |
| vLLM `model_executor/layers/quantization/modelopt.py` | ModelOpt NVFP4, MXFP8 and mixed-precision configs; Mia's `patch_modelopt_mxfp8.py` patches it | Apache-2.0 (SPDX header, "Copyright contributors to the vLLM project"; root `LICENSE` the Apache text) |
| vLLM `model_executor/kernels/linear/{nvfp4,mxfp8}/` | The linear-kernel choice, in priority order on CUDA. MXFP8: FlashInfer CuTe-DSL, FlashInfer CUTLASS, Marlin, B12X, emulation, Humming, FlashInfer TRT-LLM. NVFP4: FlashInfer CuTe-DSL, FlashInfer CUTLASS, FlashInfer B12X, vLLM CUTLASS, then weight-only and fallbacks | Apache-2.0, as above |
| vLLM `model_executor/layers/fused_moe/oracle/nvfp4.py`, `experts/flashinfer_cutlass_moe.py` | NVFP4 routed experts through FlashInfer's CUTLASS fused MoE. The default lane always mounts Mia's copy of this file (its determinism patch), so it is very likely that lane's MoE backend; the engine log confirms it | Apache-2.0, as above |
| vLLM `csrc/libtorch_stable/quantization/fp4/` | vLLM's NVFP4 activation quantization and CUTLASS GEMMs, `nvfp4_scaled_mm_sm120_kernels.cu` among them | Apache-2.0: 9 files "Copyright (c) 2025, NVIDIA CORPORATION" with the Apache notice, 2 MXFP4 files vLLM's SPDX header |
| FlashInfer `csrc/fused_moe/cutlass_backend/`, `csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/`, `csrc/cute_sm120_mxfp8_groupwise/` (97 files) | The CUTLASS fused MoE and FP4 GEMMs, from TensorRT-LLM, and an sm_120 MXFP8 groupwise GEMM | Apache-2.0 notices, NVIDIA or FlashInfer team copyright. Root `LICENSE` Apache-2.0; `NOTICE` names NVIDIA and the FlashInfer community; `licenses/` holds CUTLASS's BSD-3-Clause, FlashAttention-3's, fmt's and spdlog's texts |
| FlashInfer `flashinfer/gemm/kernels/` CuTe-DSL kernels, `dense_blockscaled_gemm_sm120_b12x.py` among them | Block-scaled FP4 and FP8 GEMMs written in CUTLASS's Python DSL | BSD-3-Clause SPDX headers (NVIDIA); three cuTile files MIT. They compile through `nvidia-cutlass-dsl` 4.7.1, whose wheel is under NVIDIA's proprietary CUTLASS Python DSL license (PyPI "Other/Proprietary License"; [terms](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/license.html)). Those terms are proprietary, not permissive, so reusing a CuTe-DSL kernel would need a decision (D-091) |
| CUTLASS C++ (vLLM fetches v4.7.1; FlashInfer's submodule `b46b16d0`) | Templates under both | BSD-3-Clause. jitLLM's source lock pins v4.7.1's headers for its own grouped GEMM ([above](#what-builds-jitllm)) |
| `humming-kernels` 0.1.12, the `b12x` extra | Other vLLM candidates | PyPI declares no license for `humming-kernels`; `b12x` not checked. Unknown until needed |

### Still owed

| Item | Found | To do |
| --- | --- | --- |
| Qwen3.8 Dual repository | Our pin `d2f54b78` (below); HEAD `2c86a1d0` | Re-audit only if its configuration becomes a baseline |
| `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` (M4) | Pinned below at `c1b7d4c9`; HEAD `943912cd` is 65 commits ahead. The checkpoint mirror's license is not verified. Its DFlash2 drafter's weights are CC BY-NC-ND 4.0 | Re-audit the recipe if the baseline moves to HEAD; record the mirror's and the drafter's licenses for information. The drafter is allowed in artifacts and benchmarks (owner, 2026-09-28) |
| DeepSeek v4.1 Flash EXL3 checkpoint (M4; weights, informational) | `Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw@64ba41b6`, MIT; the recipe is pinned below at its HEAD | Record the checkpoint pin |
| Kernels from Mia's recipes (M4) | E3 fat-expert and cooperative MoE: AGPL or mixed provenance (below) | Under D-080 once copyleft is confirmed; mixed provenance blocks until clarified |
| Wider GGML closure (M3) | llama.cpp MIT at the locked pin; widened for DeepSeek V4 Flash and Qwen3.8 Flash on 2026-09-28: the added ggml-cuda units carry no header of their own, and cite only upstream PRs, CCCL issues and the stream-k paper (arXiv 2301.03598) by reference | Done for these models' operations (lock `license.scope`), the NVFP4 MMQ instance unit Qwen3.8's A/B chose included; audit again for the image pipeline's operations. CUTLASS 4.7.1's headers became a lock component of their own for Qwen3.8's prefill ([above](#what-builds-jitllm)) |
| stable-diffusion.cpp graph code (M3) | MIT at `c92d73c4`; its GGML fork's patches are separate | Audit the ported code; audit the fork's patches before any is used |

## Pinned reference inventory

The links below identify the exact trees inspected; moving branch names are
not audit identities. All tracked paths were enumerated (including hidden
files), license files and declarations were inspected, and source notices
were checked separately from license strings embedded in test/patch payloads.
Counts include docs, data and configuration, not just implementation files.

| Key | Repository and immutable revision | Tracked paths |
| --- | --- | ---: |
| GLM | [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks, c1b7d4c9f98c16af65640fef05fa493bbde52ccf][glm] | 252 |
| DeepSeek | [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks, 6f7d1590ad49a2b8995188e45d7b9db31e677452][ds] | 96 |
| Qwen | [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks, d2f54b78c0d2f9d74ac61aa56200e3c40fac3f22][qwen] | 226 |
| ExLlamaV3 | [turboderp-org/exllamav3, 6b84a21b6f1e5da3f291b9e1019061f0de788279][exl] | 743 |

The following is a **declared-license inventory and adoption disposition**.
A retained upstream notice establishes upstream terms; it does not alone
establish the terms of every downstream modification. Unresolved cases are
explicitly blocked from permissive-core incorporation under D-017. The
inventory is complete for this planning task; clearing a selected component's
history and full dependency closure remains an adoption-time requirement.

### Repository defaults and historical notices

| Scope | Observed declaration | Disposition |
| --- | --- | --- |
| GLM scripts, overlays and docs without a more specific declaration | README License section and root `LICENSE`: AGPL version 3 | Optional implementation tier if copied/adapted; no blanket MIT permission |
| DeepSeek launcher/overlay and other files without a more specific declaration | README and root `LICENSE`: AGPL version 3; README also mentions MIT files | Optional implementation tier; resolve file-specific exceptions below |
| Qwen files without a more specific declaration, including `tp1/` | README: `AGPL-3.0-or-later` | Optional implementation tier |
| GLM and DeepSeek root `LICENSE.MIT`; GLM `extensions/cooperative_moe/tp3/LICENSE.MIT` | Preface limits retained MIT notice to contributions before 2026-09-07 | Historical evidence only; not a dual-license offer for the current tree |
| Direct ExLlamaV3 upstream source | Root `LICENSE`: MIT, copyright Turboderp; `exllamav3/vendor/fla/LICENSE` separately retains MIT and its authors | Candidate for core-allowed implementation tier after selected-file/dependency audit; preserve actual notices |

GLM/DeepSeek say AGPL-3.0 without an explicit project-wide “or later” grant.
Do not infer that grant from the sample application notice at the end of the
standard license text. Record the literal declaration here; resolve the exact
SPDX expression for any adopted unit. Qwen explicitly provides “or later”.
For old MIT contributions, obtain a pinned historical version and verify the
actual code and notices before reuse; this audit did not clear old revisions.

### File-specific declarations and mixed provenance

Paths in a row are relative to its pinned repository. Unlisted MiaAI-Lab paths
retain the repository default above **for screening**, not a claim that
third-party material loses its original license. Full builds may combine
these categories.

| Repository and paths | Evidence and classification |
| --- | --- |
| GLM `overlay/ablit_runtime.py` | File starts with an MIT SPDX license identifier. Explicit MIT candidate; this does not cover its checkpoint inputs or the surrounding importer/launcher |
| GLM `overlay/exl3.py`, `overlay/dflash2_speculator.py`, `overlay/qwen3_dflash2.py` | File headers declare Apache-2.0; do not label these MIT merely because they integrate EXL3 |
| GLM `overlay/tp3/vllm/model_executor/parameter.py`, `overlay/tp3/vllm/model_executor/layers/vocab_parallel_embedding.py`, `overlay/tp3/vllm/model_executor/model_loader/weight_utils.py`, `overlay/tp3/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` | Apache-2.0 headers with vLLM attribution; modified overlay provenance still needs review before adoption |
| GLM `tests/fixtures/flashinfer_mla_sparse_sm120-487ecf187.py.txt`, `tests/fixtures/kda-487ecf187.py.txt` | Apache-2.0 file headers; fixtures are also incorporated source if copied |
| GLM `tests/test_gen_defaults.py` | Describes an embedded vLLM fixture as Apache-2.0; that notice does not independently license the entire test harness |
| DeepSeek `overlay/exl3.py` | Apache-2.0 file header, distinct from the AGPL-default patch scripts and native overlays |
| Qwen `files/ple_layer_patched.py`; `files/ple_offload/{connector,ple_offload_layer,protocol,worker}.py`; `files/ple_offload/orig/{connector,ple_offload_layer,protocol,worker}.py`; `tp1/files/ple_offload/orig/{connector,ple_offload_layer,protocol,worker}.py` | Apache-2.0/vLLM headers; README expressly preserves file-specific SPDX terms under `files/`. Record headers, but audit modifications and the `tp1/` copies before adoption |
| Qwen `bench/sweep.py`, `files/build_draft_vocab.py`, `files/evict_page_cache.py`, `files/patch_qsa_fp8_kv.py` | Explicit AGPL-3.0-or-later headers; `files/patch_mtp_draft_vocab.py` also attributes an AGPL-3.0-or-later origin |
| GLM/DeepSeek `overlay/exl3_fat_gemm.{cu,cuh}`, `overlay/exl3_fat_moe.{cu,cuh}`, `overlay/build_exl3_fat_moe_ext.py`; DeepSeek `overlay/e3v2/exl3_fat_moe.{cu,cuh}`, `overlay/row_store.cpp`, `overlay/engram_{file_backend,layout}.py` | No separate permissive grant found for these current files; use AGPL-default optional classification. Including MIT ExLlamaV3 headers does not make these files MIT |
| GLM/DeepSeek `extensions/cooperative_moe/native/{cooperative_moe.cu,cooperative_moe_kernel.cuh,exl3_moe_coop.cuh}` and GLM TP3 counterparts plus `tp3/native/dispatch.cu` | Modified native implementations with retained `LICENSE.exllamav3` and documented MIT upstream origin. Downstream specialization is not proven MIT by that notice. Treat as mixed provenance, optional/blocked for permissive-core reuse pending clarification; preserve MIT notices in any permitted combined work. `cooperative_moe_kernel.cuh` (native and TP3) includes and follows the GEMV kernel, whose QTIP question D-080 closed |
| GLM `extensions/cooperative_moe/tp3/vendor/exllamav3_ext/` headers listed below | Nine byte-identical upstream MIT headers verified against the pinned upstream tree; separable MIT candidates, not evidence that the surrounding TP3 module is MIT. `quant/exl3_gemv_kernel.cuh`'s QTIP provenance question, in [Early EXL3 companion](#early-exl3-companion-d-052), is closed by D-080 |
| GLM `overlay/patch_sparse_mla_slice.py` | Attributes a patch to punkjazz-labs under MIT, but does not provide a separate whole-file license declaration. Resolve source revision, copied portion and later changes; do not promote the entire script to MIT |
| GLM `overlay/patch_flashkda_tp3.py` and `docs/licenses/Apache-2.0-FlashKDA.txt` | Script names external adaptation and vLLM origins; accompanying Apache license text does not prove every adaptation is Apache. External origin and modifications remain adoption blockers for a permissive classification |
| ExLlamaV3 `exllamav3/conversion/standard_cal_data/*.utf8` | Calibration corpora contain third-party text, not just upstream-authored implementation. In `code.utf8`, embedded source carries GPL-2.0 SPDX notices and a separate All Rights Reserved notice. These do not license the entire corpus under either term, and the root MIT grant does not clear the embedded material. Corpus use or redistribution needs its own provenance review; no corpus is cleared here |

The Apache/MIT headers above are observed declarations, not a finding that all
modifications have independently verified grants. In particular GLM and
DeepSeek combine blanket AGPL descriptions with inherited Apache headers;
selecting a file requires resolving that scope against its history. Prefer
retrieving the needed permissive implementation directly from a verified
upstream revision when downstream changes are not required.

`overlay/patch_dflash2.py`, tests such as `test_suppress_stops.py`, and other
patch/fixture builders contain SPDX strings **inside generated content**.
Those strings do not license the enclosing script. Copied patch payloads,
generated source, headers and native binaries follow their actual provenance;
a generator is not an escape from D-017's implementation audit.

The ExLlamaV3 calibration-data exception also matters when packaging: its
`pyproject.toml` includes `conversion/standard_cal_data/*.utf8` as package
data, and `exllamav3/conversion/calibration_data.py` loads that corpus. An
audit of selected MIT kernels does not clear the whole upstream package or
its calibration inputs.

### Verified ExLlamaV3 boundary

The cooperative-kernel READMEs identify upstream kernel origin
[`58d4d7322a1b3bd70aae8412487b21cc5e205cf4`][kernel-origin]. The GLM build
script and DeepSeek archive helper instead pin support headers to
[`02aef45cd681b960a00afcd0749a4ab99e6c1bfe`][header-origin]. These are different
identities. Both revisions' root licenses were fetched and checked as MIT.
The source kernel family is
`exllamav3/exllamav3_ext/quant/exl3_moe_coop.{cu,cuh}` and
`exl3_moe_coop_kernel.cuh`; downstream fixed-shape specializations must not
be represented as unchanged upstream files.

GLM TP3 `PROVENANCE.json` names the header pin and nine SHA-256 hashes.
All nine vendored files matched both those recorded hashes and the actual
upstream Git blobs under `exllamav3/exllamav3_ext/`:

- `compat.cuh`, `util.cuh`, `ptx.cuh`, `util.h`;
- `quant/codebook.cuh`, `quant/hadamard_inner.cuh`, `quant/exl3_dq.cuh`,
  `quant/exl3_gemv_kernel.cuh`, `quant/exl3_kernel_map.cuh`.

The vendored `LICENSE.exllamav3` retains the MIT notice. This comparison proves
identity of those files, not the validity of every archive hash or the whole
TP3 provenance chain. TP3 also ships `LICENSE.upstream-AGPL-3.0` and a
historical `LICENSE.MIT`; those must not be dropped when considering the
combined module. GLM/DeepSeek's Dockerfiles pin other ExLlamaV3 revisions
(`c5d9c657966ffeeaa9353f0cc899f18629da4a13` and
`e648f1a131365aae15920073e761a3fa5a527654`, respectively); their built images
and transitive dependencies were not audited here.

## AGPL in a served process

[AGPLv3 section 13][agpl13] requires a modified covered program that supports
remote network interaction to prominently offer interacting users its
Corresponding Source, with free access through a customary network copying
mechanism. A served modified runtime must account for that obligation even
when no binary is distributed. [Section 1][agpl1] defines the source scope,
including relevant build/install/run scripts and covered dependencies;
[sections 4–6][agpl4] separately govern conveying source and object code.

**Shipping (D-080).** jitLLM's own builds and packages include its
optional copyleft modules by default; the copyleft-disabled profile is the
build-time opt-out. A component counts as copyleft only once that is
confirmed; a provenance suspicion alone does not. A project's declared
license covers its own code; an included library or file that directly
states a different license for itself (its own license file, an SPDX tag
or license text) is classified by that statement, and an implied origin
(a citation, "based on", "QTIP-style") is not a statement. A package with
a confirmed-GPL module is conveyed under its terms as a whole; an AGPL one
also owes the network source offer below.

For a future enabled AGPL backend, review the **actual combined program** and
provide source for the covered version and build, preserving MIT/Apache and
other required notices too. A C ABI, shared library, optional build flag,
sidecar or reverse proxy does not by itself settle whether works are separate
or limit the required source to one kernel. Do not interpret “network” as
only public Internet access. Choose and verify the user-facing source-offer
mechanism before serving that configuration; this inventory does not add an
API or approve an optional-module boundary.

This does not relicense independently authored jitLLM source files. It means
the combined configuration must meet its applicable terms. The copyleft-disabled
profile must exclude the optional implementation's entire source/header/
generator/generated-code/binary closure from fetching and building, while
retaining an independently useful scheduler and allocator (D-002/D-017).
Tools and platform runtimes are recorded separately under D-017, and model
weights remain user-supplied and outside project licensing scope per D-002;
since 2026-09-28 (D-087) a weight license gates nothing in jitLLM and is
recorded for information only.

## Adoption gate and verification handoff

Before selecting any of these units, record its exact upstream path/revision,
license and copyright notices, modifications, implementation/tool/platform
role, dependency closure, and shipped notice/source obligations. Resolve the
mixed-provenance cases above or use a verified alternative; neither the EXL3
format name nor a repository's top-level license clears all implementations.
M1 builds the provenance tooling; D-052 advances selected upstream EXL3
integration to M2. Other optional recipes remain future work under D-028.
The early proof changes scheduling, not D-017's license policy.

Builder verification on the x86-64 workstation: clean initial working tree;
read-only Git snapshots of all four listed revisions; tracked-path and notice
inspection; fetched both cooperative upstream revisions; nine header hash
and byte-identity comparisons passed. Documentation links and whitespace were
checked. No runtime behavior changed, so no application tests, builds, Spark
runs or upstream recipe execution were needed. Upstream raw snapshots remain
outside the repository. All changes remain uncommitted for human review.

Independent review on the x86-64 workstation covered the complete uncommitted
documentation change against D-017 and the M0 inventory task. The reviewer
rechecked all four checkout identities and tracked-path counts, scanned
tracked files for SPDX/copyright/license notices, inspected the repository
defaults and historical MIT prefaces, and compared the network-source wording
with the pinned AGPL section 13. This found and fixed an omitted ExLlamaV3
calibration-corpus exception; package-data inclusion and its loader were
checked too. No remaining findings for this planning scope. The separate
adversarial pass challenged historical MIT, modified native files, embedded
SPDX strings, the AGPL service boundary and nine-header upstream identity.
Neither pass clears complete dependency closures, corpus rights, old MIT
revisions or built containers. No runtime tests or Spark runs were warranted
for these documentation-only changes.

[glm]: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/c1b7d4c9f98c16af65640fef05fa493bbde52ccf
[ds]: https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/tree/6f7d1590ad49a2b8995188e45d7b9db31e677452
[qwen]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/tree/d2f54b78c0d2f9d74ac61aa56200e3c40fac3f22
[exl]: https://github.com/turboderp-org/exllamav3/tree/6b84a21b6f1e5da3f291b9e1019061f0de788279
[kernel-origin]: https://github.com/turboderp-org/exllamav3/tree/58d4d7322a1b3bd70aae8412487b21cc5e205cf4
[header-origin]: https://github.com/turboderp-org/exllamav3/tree/02aef45cd681b960a00afcd0749a4ab99e6c1bfe
[agpl13]: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/c1b7d4c9f98c16af65640fef05fa493bbde52ccf/LICENSE#L540-L559
[agpl1]: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/c1b7d4c9f98c16af65640fef05fa493bbde52ccf/LICENSE#L101-L140
[agpl4]: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/c1b7d4c9f98c16af65640fef05fa493bbde52ccf/LICENSE#L185-L328
