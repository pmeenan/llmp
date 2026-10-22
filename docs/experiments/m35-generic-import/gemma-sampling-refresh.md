<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma target sampling metadata refresh

The existing Gemma26 UD-Q4_K_M and Gemma31 UD-Q4_K_XL targets now retain
sampling defaults from their exact approved GGUF sources: temperature 1,
top_k 64 and top_p 0.949999988079071. Their current IDs and paired target
references change together. Four target/pair spec pins record the resulting
canonical metadata. Source receipts, admission limits, numerical qualification
records and legacy assistant artifacts are unchanged.

The native writer published the two selected files from the retained archive
revisions, including source authentication, StoredSpec/component validation and
Deep artifact verification. Publication took 253.346 s and 281.777 s on Spark A;
these observations are not model performance measurements. The actual W8 writer
snapshot is `616545478ad909e8b9854fa1fbe0ab0aa82d9c808dfb58eaed97a44be1d2303f`.
The [aggregate](gemma-sampling-refresh.json) retains source revisions, full IDs,
actual compiler/binary identities and content digests. Raw manifest hashes and
semantic artifact IDs are recorded separately; build provenance is retained as
emitted by the writer.

| Target | Named encoded parts | Compared payload bytes | New ID prefix |
| --- | ---: | ---: | --- |
| Gemma26 | 8,278 | 16,931,716,216 | `a1f1c523` |
| Gemma31 | 833 | 18,807,134,448 | `09080998` |

Every encoded part, including all Gemma26 expert slices, has identical format,
length and payload SHA-256 before and after publication. Complete source,
model, layout and transformation metadata match. Complete canonical target JSON
is identical except for `/stages/0/sampling`. The two native GGUF import controls
passed with exact normalized/stored spec and inventory equality, checking
658/658 and 833/833 tensors respectively.

Actual native ComposeDrafter JSON with each unchanged assistant reproduces its
historical full pair digest after removing only the new target-stage sampling
field. The authenticated W8 old-map approval PASS supplies the old control.
A message-only diagnostic captured precisely four expected old-pin failures,
two targets and two pairs; it is retained as a negative diagnostic, not a
passing adoption test. No field is excluded from product digests or equality.
The existing assistant components still lack authoritative StoredSpec; that
separate migration remains open.

The first Gemma26 evidence wrapper assumed an EXL3-only `bytes_compared`
property after the native test had passed. Its failed job and output remain
retained. The corrected audit reused the same positive XML with the GGUF
658-tensor count contract; the separate 8,278-part proof authenticates full
payload bytes. No successful test rerun or expectation change was needed.

Canonical promotion on Spark A authenticated every manifest-listed file and
retained both original artifacts. On the adopted IDs and four new pins, the
native controls passed four CTest entries containing 11 actual cases, zero
skips: both GGUF imports, published-artifact/pair approval and eight existing
Gemma target/assistant/verify/envelope repeatability cases. Complete fixture data
was unchanged except the two current IDs and their two pair target references.
The executed approval/runner/import binaries and complete counts are pinned in
the aggregate.

Both canonical stores now contain the verified new artifacts, with the originals
retained. The peer operation used the old artifact as an rsync checksum basis:
35,782,224,606 bytes of shard/index/retained-metadata files are identical, while
the two specs and two manifests total 166,924 differing file bytes. Actual rsync
literal data was 2,807 bytes, with 5,007 bytes sent and 1,601 received; copy plus
complete file verification took 65.808 s. The inode audit found no hard links:
at copy time new native files/directories were 0644/0755, versus retained old
files 0600, so
permission preservation prevented hard-link reuse. These are copy observations,
not model-speed claims; no source license or qualification record changed. All
new proof/canonical artifact directories were subsequently normalized to 0700
and files to 0600 on A, and both new canonical artifacts likewise on B. The
before/after stat receipts preserve inode, size and mtime; old artifacts and
checkpoint inputs were untouched. The native writer's original public modes
remain a separate generic defect, not silently described as private publication.
License/header checks passed; this final outcome-only paragraph leaves all SPDX
declarations unchanged.
Existing frozen Gemma numerical-oracle failures and scalar qualification work
remain separate. Repeatability controls do not create numerical qualification;
this metadata refresh neither changes frozen oracles nor claims a numerical fix.
Historical experiment IDs/results remain intact, and U3 remains open for the
remaining target metadata refreshes and drafter StoredSpec migration.
