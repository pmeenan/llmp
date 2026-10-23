<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek V4 target sampling metadata refresh

The approved DeepSeek V4 Flash 0731 UD-Q2_K_XL target now preserves the
source GGUF sampling defaults `temperature=1` and `top_p=1`. The canonical
target spec changes only at `/stages/0/sampling`; the DSpark artifact itself,
source identity, model layout, transformations, index and encoded payload do
not change. The target's semantic artifact ID changes, and so does the
canonical target/DSpark composition digest. The old artifact and community
DeepSeek approval remain intact. This metadata refresh adds no model-quality,
serving or speed qualification.

## Identities and parity

| | Previous target | Refreshed target |
| --- | --- | --- |
| Semantic artifact ID | `d2046349e81da7019c9d787c2eb92521b7a3f4ca9d09c398448da1a99f8eef3f` | `01a5e5ba5a911528fd58268406ac6cff032e767fd09a20fd4c9cbf84f2043f13` |
| Stored spec SHA-256 | `ddaaa1d38bd6a501b9ef891bc0f639a70f41b19ff85a85674be5d77470e7a8e8` | `44c7eb37336be510acca6fe22ad676036bd8a40504593d1145e560879f7493de` |
| Raw manifest SHA-256 | same as old ID | `115009913ef53b64002af7c1bf2a0d70888211f9d0fa3351951203d754c461b6` |
| Index SHA-256 | `4f84772b3cfa5912e6a5ece00f438b9b8ac330679dd848a1896810f7d48a8e80` | unchanged |

The semantic ID is not the SHA-256 of the raw manifest: the refreshed
manifest's raw digest is `11500991…`, while its semantic artifact ID is
`01a5e5ba…`. The identity rule excludes build provenance. Both specs were
parsed and validated natively. Native `ComposeStoredDrafter` with the unchanged
DSpark artifact `e14bf29f28c408a742ba6ab19879e415b1ec13019c0ed33b87b7802232a0bd90`
reproduces the old pair digest `362a0d7ba26d65a9ca564b97786708dae036e5b337abc361c675ab93859f640a`;
the refreshed pair digest is
`ac3d1a4d67f8d77b288ba70c227c92a00b38c639dcaac3cad9e6a600b5857ccd`.
Canonical pair JSON comparison finds only the target stage's sampling values
changed. Both outputs pass `Validate` and a canonical JSON parse/serialize
round-trip.

Independent full payload comparison found 34,223 named indexed parts and
96,827,162,972 payload bytes, with no added, missing or changed part. The
payload digest is
`576effe569bce93510da64009a28169b0473f741a90c7c42a2db321afb050df1` on both
artifacts. The index digest, source names and hashes, architecture/mapping
(`deepseek4`, version 1), resource listing, layout and transformations match.
The source-bound import took 513 seconds on Spark A and completed native deep
publication into the private proof store.

## Source and build provenance

The source is `unsloth/DeepSeek-V4-Flash-0731-GGUF` at archive revision
`fbbb5b93fb787c21338159b0af3318bb3f4d9768` (archive manifest SHA-256
`3a2146f9adffe5c8b9b0646ab888ff571f4a76fd8697e3f13982b99b9145f980`):

| GGUF member | Bytes | SHA-256 |
| --- | ---: | --- |
| `DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf` | 5,257,664 | `de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b` |
| `DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf` | 49,437,013,568 | `c8728a298862cd8736782e2e4056e5e3e03c415ba552a38173264494e41ef327` |
| `DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf` | 47,390,237,120 | `a4bae0788c025c1b2388754592c98fefa1372981748ac9875039bf6a30e6eeb4` |

The import ran with the pinned `aarch64-2c4eb5b2f68f3fa5` SDK and Ubuntu Clang
22.1.8. The CLI SHA-256 was
`12c049c60e932373a4cd8d4cf53c8652d07156209be14fbbed012631639a9a47`; the
build receipt SHA-256 was
`5390ed59a7019a6e3b14f355b60b6c2fe3bb884adf4153f6f0665f2ca7d89578`.
Its guarded source snapshot has 2,917 tracked files, zero pre/post mismatches,
and manifest SHA-256
`77b4f3b93fa09ce7abeac0e9c9a9d661c092b0acf95920ed9e27289e7684cab9` (root
HEAD `ee88938d83c74b75f6463efa17b4bcc2ccc8e7df`). The Spark checkout reported
Git ID `61f8426f7e2f45931d91645b016069e13f6f2236` with a dirty tree; that ID
alone does not identify the synchronized source snapshot.

The retained compiler-identity receipt was emitted after the historical W21
CLI/runtime identities. Its file SHA-256 is
`c28ab235b2173c2e427a0b76c0435742c2da5c83a15b1b3b4f2e3bb6af09132d`; the
historical compiler-receipt identity recorded with those binaries is
`99e19623ecb0c2cd3f8725c258aea77a70ad24cb66025a2d910910b4dd4d01e3`. These
are different records from different times and are not interchangeable.

## Publication, retained failures and controls

The import wrapper initially reported failure after native publication because
it asserted that the raw manifest SHA must equal the semantic artifact ID.
Those values are intentionally distinct; the wrapper output and job log are
retained unchanged. An earlier preflight attempt stopped on a missing remote
scratch directory before invoking the importer. Neither failure changes the
successful 513-second native publication or its authenticated proof.

The existing canonical artifacts on Spark A and Spark B were retained. Each
host published the new ID by native-open validating the old artifact, staging
artifact and published artifact (including `StoredSpec` digest), then using
no-replace publication. For each host, all 27 unchanged listed members (the
payload, index and retained metadata; 96,841,227,236 bytes) reuse the exact old
inode and size. The new 144,897-byte `spec.json` and raw manifest are copied
from authenticated proof metadata. This is inode reuse based on the prior
artifact's authenticated manifest, not a new full payload hash. Receipts:
Spark A promotion `beac5293fdff47c0d18eca6fa1b5f1f5debb4c6791879c981ff0ee3b1b7bad05`;
Spark B promotion `424cc0f5a621ece8101b63e9a6fe33fa1fa89237250da87122229b4fd158cfbd`.

The source pin patch updates the approved target and DSpark pair references,
target IDs used by existing controls, and adds a source-backed sampling test
while retaining the old metadata-free fixture as a no-default control. The
community artifact approval and frozen numerical expectations remain
unchanged. On Spark A, the two whole-source GGUF import cases and the
source/artifact equivalence case passed. Current approval/pair pins and the
complete target/drafter structural fixture then passed: five positive model
controls, zero skips. The batched shared suite passed all 3,674 registrations
with zero CTest failures/skips; see the [FP8 check limitations](block-fp8.md#verification)
for inner-counter retention.

Original failed checks remain retained: the old raw-manifest expectation, an
ineffective fixture repair that supplied sampling keys to structural synthesis
(which does not read them), and an approval test executable that had not been
relinked after the pin update. The final fixture explicitly supplies the
independently verified defaults after synthesis, and the actual approval
consumer was rebuilt before its passing run. No production behavior or frozen
numerical expectation was changed to repair these checks.

The refreshed target's frozen execution/spill screen still has nine differences;
an additional run of the exact same executable with the retained old artifact
(`LLMP_TEST_DSV4_V1_ID=d2046349…eef3f`) reproduces every actual/expected mismatch
exactly. The DSpark screen retains three differences, identical to the earlier
legacy-drafter control. These twelve observations are formal test failures,
not numerical passes or new qualification. Both target runs and the DSpark
run completed their native execution and state/spill/restore controls without
skips. The comparison establishes that this metadata refresh introduced no
additional frozen discrepancy; U4's numerical obligations remain open.
