<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen selected-head provenance — 2026-10-04

The installed selected MTP artifact executes **47,172 rows**, including
when `draft_vocab = 65536` is requested. Several recent reports incorrectly
called that requested cap a 65,536-row prefix head. Their paid counts,
timings and controls remain valid, but they do not establish a comparison
between selected and prefix layouts. The affected reports are corrected;
recorded commands, hashes and earlier genuine prefix experiments are retained.

## Artifact and execution

Drafter artifact
`8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`
has these resources in its validated `index.json`:

| Role | Representation | Bytes |
| --- | --- | ---: |
| `draft_output.weight` | GGML BF16 `[2560, 47172]` | 241,520,640 |
| `draft_output.ids` | GGML I32 `[1, 47172]` | 188,688 |

The index SHA-256 is
`5cd45fc5354ab224d281c2416027f224c61e32e2ae0acf8e7580d063fc274d99`.
`Qwen38Runner::MtpInputs` caps a positive requested row count against this
map's rows. The graph uses the selected BF16 matrix and remaps its argmax
through these IDs. Zero requests all available selected rows. The unselected
prefix artifact
`056a750e3a90be3ae6a4b12bb963ce45290aaa6f52b5ba9799e777d491f80aea`
has neither selected resource; it is required for an actual prefix comparison.
A requested count alone does not identify the executed layout.

The original same-history collections `native-curated-final` and
`native-prefix-four` both use the selected artifact above, with nominal caps
47,172 and 65,536. The matched cold collection also uses it. Their identical
native proposals therefore compare the same head, not two layouts. Fresh
native/reference acceptance remains 9/12 on the four recorded cold anchors.
The [head-sharing screen](../qwen38-draft-head-waves/README.md),
[prefill chunks](../qwen38-prefill-chunks/README.md),
[expert tiles](../qwen38-expert-tiles/README.md), later
[four-request waves](../qwen38-four-request-waves/README.md),
[standard-client gate](../m3-standard-client/README.md) and
[final swap table](../m3-final-swap/README.md) also execute the selected head.
GGUF/plain cells have no live drafter. Earlier prefix measurements using the
unselected artifact remain prefix measurements.

## Nominal-cap screen: no execution factor

A fresh Spark A HTTP triplet requested 65,536 → 47,172 → 65,536 using that
same selected artifact. Every arm executes the same head; this is not a
layout or truncation A/B and selects no new default or calibration.

| Requested cap | Effective selected rows | Completed tokens/s | Peak MemAvailable drop GiB |
| ---: | ---: | ---: | ---: |
| 65,536 before | 47,172 | 32.165009 | 81.373 |
| 47,172 | 47,172 | 32.170448 | 81.281 |
| 65,536 after | 47,172 | 31.716302 | 81.207 |

Each arm serves four uncached 8,256-token prompts and four 256-token
outputs, with HTTP 200 and length finishes. All three services exit zero
and are reaped; all six strong 105-GiB admission/retirement gates pass.
Arrival-dependent wave arithmetic remains. These clocks qualify paid work,
not a performance benefit from a head change. No reference engine runs.

The fixed runtime is checked `ab83d05`, SHA-256
`64fccb395bbeabd2bb7569ec7597f7d9cc6672afb8abb236d4d5deaa7b99b9ab`;
its independently retained compiled-source inventory is
`ecdc71d00d4fea3c56c4aaa540ac49b183df852409a62e26330c3843ba54c0e9`.
Spark A is `spark-c4e2`, GB10, driver 580.178.04, SDK CUDA 13.4.92.
Context is 33,792, chunks 4,096, slots four, normal speculation and the
existing two-request draft-wave limit. The measured controller SHA-256 is
`bf0f2059d8901071b9c5641b87a4e51c61db7b26ba58cb3039866a132dba8789`;
its original docstring incorrectly names two head layouts and is preserved
as evidence rather than edited after the run. Raw output is external at
`spark:~/scratch/qwen-shared-vocab/screen47172/` and
`~/scratch/llmp-m3-qwen-shared-vocab-2026-10-04/`. Installed supervised job
`qwen-shared-vocab47172` finishes zero and is waited on.

## Reporting correction

Model settings now derive the selected head's physical cap, retain smaller
owner limits, preserve an override's source and explain the nominal request
when capped. An oversized nominal limit still fails before normalization,
as the runner already required. Plain serving does not adopt a live head.
This changes diagnostics, not the existing head-selection arithmetic.

`llmp_qwen38_spec` retains requested `draft_vocab` and adds effective
`draft_head_kind` and `draft_head_rows`. The same-history analyzer records
`requested_head_rows` separately; older captures lacking effective metadata
report null head kind/rows, not an inferred prefix layout. Use the artifact
and recorded arguments to audit those historical captures.

The corrected production source passes all 1,547 tests on Spark B (98.63 s),
changed-file SDK format/tidy, and tracked REUSE/header/boundary checks.
Nine direct settings controls exercise a tiny physical BF16 head, smaller
and zero/all overrides, oversized-request refusal, missing token maps,
wrong head/map types and mismatched map rows. These extract physical facts;
model binding still owns executable geometry and token-ID validation. All
five temporary fixture artifacts are removed. The control script SHA-256 is
`d8b4aa898c9a4bf18a2f85db336b600fcd2c22502dda47b75ca75ccf6a8b3e4b`;
raw records are external at `spark-b:~/scratch/qwen-shared-vocab-review/controls1/`.

One final production benchmark capture repeats the literal 32K `p3` anchor,
nine outputs and two repeats. It reports requested 65,536, kind `selected`
and actual 47,172 rows; its complete first-verify hash remains
`5a2d9ad6882028869e92b7427d99dc0065a49c2fa2c5c0493800170b8615f556`.
The child exits zero, is reaped and both strong gates pass. External evidence
is `spark-b:~/scratch/qwen-head-provenance/native-metadata/`. The revised
analyzer also passes the original five-request all-cold collection on Spark A,
retaining 65,536 as the request and null effective fields for the historical
capture. No new acceptance ladder or workstation/package checks are claimed.
