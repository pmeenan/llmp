<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Owned Qwen3.8 target GGUF normalization

This slice adds the supported 48-layer `qwen4exp` target GGUF domain to the
existing owned normalizer. It also makes that domain eligible for the ordinary
`llmp import` normalization and writer path. The full native build, 103 affected host controls and the actual-header
normalization check pass on `spark-b`, with zero failures or skips. This historical
normalization slice did not publish or execute the full artifact. N4b8 later
completed ordinary publication and stored-target execution; see the
[2026-10-13 evidence](../m36-numerical-audit/qwen-joined-mtp.md#native-gguf-publication-and-target-serving--2026-10-13).
Family quality/performance and public migration remain separate.

## Source evidence and retained meaning

The catalog source is `unsloth/Qwen3.8-Flash-Next-GGUF` revision
`38bb39ee97821de2c9009abb7e93950eec396e66`, `UD-IQ3_XXS`, already archived
on the NAS. Revision manifest SHA-256:
`506dfcaff491132eadb74e9c5f13f42dbe5f9668289aa130e0790c7fc4a549dc`.
The metadata-only inventory SHA-256 is
`bc35f7f6cbffb09b5ffa5bac61abfeddd07d2c9ba9437ac22f360c58e6e4164e`.
It reads three bounded GGUF headers and checks their file identities before
and after. Archived payload hashes are provenance from the revision manifest;
this inspection did not rehash payloads or authenticate an upstream converter
invocation. Model-weight license metadata is informational.

The headers describe 1,224 tensors: 557 F32, 24 BF16, 250 Q6_K, 248 Q8_0,
49 IQ4_NL, 94 IQ2_S and two IQ3_S. The mapping preserves their formats and
whole encoded copy routes. Its supported dimensions, GDN/QSA layer schedule,
PLE layer selector and rotary sections are explicit. Multipliers and per-head
n-gram ranges are copied from the source, including different valid values;
only lengths, catalog parameter ranges and valid I32 table access constrain
them. No n-gram weights or numeric constants are synthesized. MTP, vision,
other profiles and unrecognized source tensors/keys refuse.

The table projects the existing Qwen mapping's GGUF semantics, including its
planning policy and already-applied `norm.plus_one`, `ssm.a_neg_exp` and
`gdn.value_heads_tiled` assertions. Those operations are not run again. This
is representation normalization, not a new numerical policy or cross-library
bit-identity requirement.

## Construction and placement checks

The metadata fixture independently represents all 1,224 header descriptors,
using synthetic nonoverlapping extents without allocating tensor payloads.
Controls check complete roles/shapes and seven encodings, source-list ownership
after the caller disappears, different valid hash constants, all 121 state
contract tensors, insufficient construction/work allowances and malformed sources.
Existing Gemma normalizer controls remain part of the affected check set.

The shared writer's exact metadata planner is exercised without file opens.
The target produces 74,808 destinations: 1,224 whole sources, with each of the
144 expert arrays split into 512 byte-preserving destinations. There are
24,576 expert groups, 48 layer groups, one head group and, in this historical
planner check, one row-table group containing both token embedding and side
n-gram weights. N4b8 separates those tables into two exclusive groups for the
actual stored-target row-owner contract. The mixing component
has no weights. Underfunded placement and a source lacking parsed identity
receipts refuse before creating a publication directory.

The incremental normalizer ledger is below 512 MiB for these headers; the
source reader/checkpoint remains separately owned and funded. The writer's
existing metadata allowance independently covers placement and streaming.
The metadata tests do not traverse or materialize the approximately 82 GB
tensor payload. Actual publication and stored-target execution were subsequent
N4b8 work; N5 all-artifact qualification remains open.

## Validation — 2026-10-12

The five complete host inventories contain 47 reader, 20 normalization,
18 writer, four ModelOpt normalization and 14 spec cases. One separately
confined real-header case preserves all 1,224 routes across the three source
files and seven encodings. It authenticates 11,024,519 parsed header bytes, retains 37 selected
metadata keys and reports the pre-derived 334,238,696-byte normalization bound.
The complete source payload size is 81,950,799,360 bytes; these checks do not
traverse or hash that payload or publish the target artifact. The existing
reader fetches bounded prefixes and may read some payload bytes beyond the
parsed header; this is not a zero-payload-I/O claim.

The actual-header run used a 2 GiB process/cache cap, zero swap and an
additional 2 GiB physical margin. Its observed cgroup peak was 58,945,536 bytes;
all memory-limit/OOM counters were zero and the child retired. This is a
metadata-only result, with no GPU, quality or speed measurement.

Provenance: main native/host receipt
`64c82f4409a7f0cadaf0627500f07ba2fabfde6440945d37cf91b6869138142d`;
actual-header receipt
`2bc565c169ef812a2005fe81384e9f9c84904e748ec47250a5a08620ef824e62`.
Raw logs and recipes remain outside Git under the coordination root's
`qwen-gguf-normalize-main05`.
