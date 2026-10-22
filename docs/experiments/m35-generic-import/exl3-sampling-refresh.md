<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# EXL3 sampling metadata refresh

The two Qwen2.5-0.5B EXL3 artifacts now carry the source's sampling defaults:
`temperature=0.7`, `top_k=20`, `top_p=0.8`. The prior artifacts omitted
`/stages/0/sampling`; a native e033 baseline reproduced the 4.0 fixture's same
stored-spec mismatch. Complete old/new canonical JSON comparison finds exactly
that added field. The full spec digest changes from
`b2cf6f15206be2712e20fe080df11f7edcbf8aaeda23f830755f767f11433d6a` to
`7d52f0cb668ef4a6e29ddad3d11c8cb482ee98e8646e095d9906a3397a941477`.
Both source-matched approval records name the new complete spec; source
identity, admission bounds and numerical qualifications are unchanged.

| Fixture | Previous artifact | Current artifact | Encoded parts / bytes |
|---|---|---|---|
| qwen2-exl3-4.0 | `4b995657242c6e74323bc7010b4d2adb3072096eb1688928bbc5d1dbdbbca77e` | `e931b03cbb89e98cb913ee86d05a72cb32df638e46c5de17cdb485ab7e69cb2e` | 798 / 588,866,468 |
| qwen2-exl3-4.5 | `a27ae586b7e2aa49dc1e01f82129733791cbe9c6acb4d68c2d0478811c65555e` | `d772b542199ce99a73dd2162fa68de0103da329ff15e830136e00197dbb68e48` | 798 / 611,173,284 |

Existing archived revisions `7009334dc77ad71f6a98d942d6df3ec64017f565` (4.0)
and `030d3a4173c5a3e27e3c27dd7f6ab0fc00636d00` (4.5) were published with the
current native writer into an isolated proof store. Each keeps the exact five
source names, lengths and SHA-256s: configuration, generation configuration,
weights, tokenizer and tokenizer configuration. No source, license, transform
or encoding changed. Every named encoded part has the same format, part role,
length and payload SHA-256; there are no missing, added or changed parts.

Native publication completes structural open, stored-spec equality, component
binding validation and deep file/chunk validation before publication. Exact
existing import controls then compared each new artifact against its historical
source fixture: both passed, including full spec/inventory equality and
588,866,468 / 611,173,284 source bytes respectively. Each filtered GoogleTest
process positively completed its one named case; outer CTest also passed both
with zero skips. Full raw and inner logs remain outside Git.

The reviewed recipe and positive publication/parity/native-test receipts are
summarized in [the aggregate](exl3-sampling-refresh.json). The actual importer
binary SHA-256 is `f77aede27a4b58f3f7b595662a8806a95bb46a9d494055dae2e588789451f8f4`;
source manifest is `e6227b0cc7fbb6cbe36350d60369f02a1f61b2bbdff4c47bbec82ee7b96470d9`.
Spark's stale Git metadata supplied the retained `61f8426f…dirty` build label;
the actual source/compiler/binary receipts establish what ran. Raw manifest
hashes differ from artifact IDs because the identity excludes build provenance.
Historical protocols and their original IDs remain unchanged.

The artifacts are installed on both Sparks. Promotion preserved each exact raw
manifest and rehashed every listed file before making the new ID visible;
old artifacts remain intact. Two initial copy-wrapper failures (missing legacy
staging directory and reversed direct-link direction) stopped before copying
payloads and are retained with their corrected successful receipts.

Spark A rebuilt the EXL3 execution and approval consumers, then passed exactly
the two `SameImplementation40G/45GCheckpointSpillContinuation` controls and the
published-store approval check, with three positive inner completions and no
skips. The complete fixture overlay retains every frozen oracle and changes
only the two current IDs. Each lifecycle control compared 169 rows over 42
executed steps, reached position 161, restored 6,291,456 bytes through both
checkpoint and spill paths, and preserved 292 / 302 immutable weight extents.
The tests record the actual new artifact IDs. Full frozen state/logit assertions
remain unchanged; no digest assertion was relaxed.

This is no new performance or numerical qualification. Four other legacy target
sampling refreshes and the legacy drafter stored-spec migrations remain open.
