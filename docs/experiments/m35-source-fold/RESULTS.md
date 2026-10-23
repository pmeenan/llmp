<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Explicit source-fold publication proof

All three affected targets have fresh private native publications with explicit
`source-fold` receipts. Their full index files and every encoded tensor byte are
unchanged. Gemma2/Gemma3 specs and all listed files remain identical; their
manifest changes only importer provenance and preprocessing receipts. QwenGGUF
combines the three receipts with its independently documented sampling refresh;
only `spec.json` changes among its listed files. At proof capture, canonical promotion and
current-ID adoption had not yet occurred.

The [aggregate](results.json) binds the actual source, native compiler/CLI,
archive revision and each old/new identity. Raw manifest SHA is recorded
separately from the semantic artifact ID; importer build provenance is excluded
from the latter by the existing identity rule.

| Proof | Native publication seconds | Encoded parts compared | Payload bytes compared |
|---|---:|---:|---:|
| Gemma2 Q8_0 | 12.464502138 | 288 | 2,778,448,896 |
| Gemma3 Q4_0 | 11.318196009 | 444 | 2,519,539,712 |
| Qwen3.8 GGUF IQ3_XXS | 436.749880959 | 74,808 | 81,950,799,360 |

These are actual supervised publication durations, including complete source
hashing and the native writer's structural open, StoredSpec, component
validation and full-file/chunk Deep checks before publication. They are not
model performance measurements. Independent old/new part hashing additionally
compares every byte, format and part kind. Full index byte equality covers roles,
access, readable bounds, all expert/group semantics and physical offsets without
any descriptor exclusions. Every non-spec listed-file identity also agrees. Qwen preserves all 144
existing expert-slice records (512 experts each) and adds only the three
source-fold records. Its exact added sampling object is
`{"temperature":"1","top_k":20,"top_p":"0.949999988079071"}`. Removing only that
field reconstructs the complete old spec; the new full digest is the independent
W7 value `74116647d445819311c58dc16336099c95d675d827a5cd4125cfd88fadbd47cb`.

Each target's exact native import-model control passed once (one outer CTest
case and one positive inner GoogleTest case, zero skips). The controls used the
full unchanged fixture tree with only the private proof ID overlaid. Native
normalization, the stored spec and complete tensor inventory agree. New artifact
directories/files are actually 0700/0600; old stores and input files remain
unchanged.

This migration does not change numerical expectations, performance exceptions,
license grants or source identity. Converter-applied source folds remain distinct
from importer-applied folds for qualification matching; only execution fold
membership takes their union. The four legacy drafters still need their separate
StoredSpec/provenance migration, and broader U3 obligations remain open.

Raw proof metadata, logs, exact test XML and supervisor results are retained
outside Git in `/tmp/llmp-source-fold-migration-r1`; the Qwen recipe/evidence is
under `/tmp/llmp-source-fold-qwen-r1`. Both retain the exact immutable native
source snapshot. All nine bounded publication/parity/readback supervisor jobs passed. Independent source and actual-results reviews are clean. Canonical promotion
on Spark A passed in 139.300582 seconds: all 28 listed files (87,365,828,027
bytes) reuse the private proof publications through verified hard links. Full
file hashes and 0700/0600 modes were checked before reuse and again before
atomic publication; old artifacts remain intact. The paired exclusive peer copy
and complete verification on Spark B passed in 153.021371 seconds. It reused 27
whole files (87,365,702,265 bytes) after checking their full hashes and matching
private modes. Changed-file size was 163,675 bytes; rsync sent 25,383 literal
bytes and reused 138,292 matched bytes within those changed files. Actual
inode/link counts confirm reuse. Original bytes and modes remain intact; reuse
increases link counts. Both canonical stores now contain the three new IDs.

Adoption changes exactly three current ID mappings and the Qwen GGUF spec pin.
None of these targets appears in a current drafter composition, so no composition
reference changes. Historical protocol/result IDs and all numerical oracles remain
unchanged. Root validates the integrated source separately; these publication and
copy proofs do not claim new model performance or numerical qualification.
