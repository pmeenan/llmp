<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# U6 standalone EXL3 dependency closure — prospective refresh

Run only after this addendum is reviewed and committed. The
[production-context registration](U6-PRODUCTION-CONTEXT-REGISTRATION.md),
committed as `caadac6e67aea97e5ce77abc184de0171c8b02eb`, and its linked
[complete command inventory](U6-FINAL-MATRIX-REGISTRATION.md) govern the exact
63 A commands and 26 B cases, source artifacts, numerical contracts, timing
methods and refusal rules. This refresh changes only the build identities below.
No samples under these identities have been collected.

The final test-artifact build exposed an incomplete EXL3 F32 converter link
closure: its GGML converter also requires llmpalooza's existing GGML CUDA support
symbols. Production binaries happened to supply them through the GGML module;
standalone EXL3 consumers did not. The repair places the unchanged support
translation unit in one shared support library and declares that dependency for
both modules, avoiding a GGML/EXL3 dependency cycle. Native arithmetic and timing
bodies are unchanged. Build identities change, so earlier samples and rows remain
under their original identities and are never repinned.

## Frozen build

| Identity | SHA-256 |
| --- | --- |
| Complete source manifest, 1,416 paths | `649212eec6f693688c0270483a8618b3cadf81f19d57d81ec2f67dc1986abaea` |
| Compiler-owned module, both hosts | `37cccb3c26d539d10e1ec4fe581e911577bf62b3b4fc469923dbf1ab19a2b007` |
| Qualified backend declaration, both hosts | `3f3b319d70b4313f13b18ccca0464041af3703eba64ec70e5231f2ee9cdf789d` |
| A ranks binary | `e28ed9f85ae4140b9b70919c07a12a84feb7e729d3933d4db2ddb75ed823c4c8` |
| A GDN binary | `efda9943f04a398b444f238df724dbd27af0bb0b3dbdd96efd77c537331350bd` |
| A GPU comparison binary | `2f3e93389f7a2dfc78766b19d0ac31ea163b52cd27e064821e1414cec3f14a53` |
| B ranks binary | `f45723612ee6ec18472e8459ad9b632b5c05b6c05e3f9ab19edc857a4ac8fb4b` |
| B GDN binary | `88efcec595cb81bdaae470b3ed80337b1ce53c7f62b70e841ca9d7c6466809c9` |
| B GPU comparison binary | `7e7d06d0ee8d649c748a474769598e87071a60582c92198c1fc3fbc4f2fad093` |

The source manifest is retained at
`/tmp/llmp-m35-orchestration/source-cuts/u6-support-closure-sampled/manifest.json`.
It excludes prose Markdown. Both native builds pass, including standalone
`exl3_linear_test` linkage and the complete assigned shared-test artifacts.
Untimed existing Q8 and NormGate controls authenticate actual native Bound
keys and the backend declaration without collecting samples.

The complete assigned shared-test artifacts and affected-model artifacts build
on both hosts before sampling. Every source path, baseline data and native
binary is verified before and after each supervised batch. The rank-data
baseline is the unchanged 12 legacy and 14 historical recipe rows, SHA-256
`e767895d6e49d20937e71132e4af3c4bb809f4071a251710b6941b01b9b103d0`.
The preceding 94 reviewed rows and their 21 passing installed controls remain
prior-build evidence; they are not input to this refresh.

## Execution

Execute the complete unchanged inventory once: A63 on `spark`, and B's 20
whole-source, four state CTest and two untimed quality cases on `spark-b`.
Independent cases run serially in exclusive supervised batches of at most 600 s,
with their own output and exit status and continued execution after a refused
case. Batching changes no case's method or observation order. No concurrent
build or other GPU work runs during measurement. There is no outcome-selected
retry, new cell, numerical limit adjustment, capture promotion or fitted bound.

Keep all ordered observations, nine-sample medians, fixed bookend drift and
strict 10% classification, full original outputs/states/immutable checks and
two changed frames. OrdinaryOnly remains pairwise; quality cases remain untimed.
The conditional state R4 cell remains omitted if its native selected schedule
matches R1. The corrected normal-context diagnostic is parsed as strict JSON
with separate `phase` and `execution_phase` fields; the preceding pass's narrow
duplicate-field analysis exception does not apply to this refresh.

Install only independently reviewed new-build rows, preserving historical rows
and numerical refusals. Verify actual installed selections, unmeasured fallback
and any measured losing-fusion veto, then run the final shared and affected-model
regressions. Commit U6 and pause for owner review; do not begin U7.
