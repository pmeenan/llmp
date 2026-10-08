<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded Gemma target verification

Both approved Gemma profiles pass eleven focused controls for a default-off,
engine-only C1 verification transaction. `Verify` completes one to four target
rows and retains every head and post-finalnorm feature. `AcceptVerify` commits
a checked prefix and its actual final feature; `DiscardVerify` restores all
saved KV writes. Neither call admits an assistant or implements serving
speculation. Scalar arithmetic parity and cross-engine verify quality remain
unqualified.

## Transaction and funding

`max_verify_rows=0` preserves the ordinary runner. Enabling 1..4 requires
`retain_features=true` and an explicit `max_head_rows` covering both slots and
verify rows, within `max_rows`; setup refuses an undersized envelope before
artifact admission or providers. The caller funds complete head/feature vectors.
No serving head capacity is silently expanded. Graph options and primitive
fallback remain the caller's policy.

The caller holds the request through synchronous accept or discard. Fresh
row-tagged copies save every actual local/global K/V write before target work,
outside its captured graph. Private feature storage preserves the previously
published feature and cursor while verification is pending. Same-slot state
mutation, checkpoint copying and frozen borrowing refuse; an independent held
peer may progress. Accepted cursor and feature publish only after rollback and
feature-copy retirement. Unknown-effect writes quarantine the affected slot;
unproven shared completion faults/quarantines the cohort and retains uncertain
owners. Secondary rollback failure repeats the shared job
health check before local quarantine. Fenced release can abandon a pending
transaction, without exposing a reusable prefix.

At four rows, each slot's raw snapshot is 901,120 bytes for 26B or 3,866,624
for 31B, before normal mapping/extent rounding. Snapshot/private-feature maps and
pinned copy/commit/validation buffers are cataloged. A separate host grant funds
`slots*(R*tensor_count*sizeof(Saved)+256)+8192`, including reserved Save metadata,
the commit callback and bounded per-row write descriptors. Actual reserved
capacity is checked. `AllocateSnapshot(..., reserve_saves=true)` is opt-in;
existing Qwen/DeepSeek callers retain the default allocation chronology. The
64 MiB caller fixture grant precedes vector growth; state payload witnesses use
separately cataloged pinned memory. This adds storage and makes no unchanged
peak-memory claim.

## Exact focused proof

Spark A (`spark-c4e2`) completed all nine supervised steps: seven new controls,
two existing Gemma feature/checkpoint lifetime controls and two existing
LiveState snapshot/unproven-copy controls. All eleven pass with zero skips.
The new controls took 72.392 s; existing controls took 5.315 s and 0.268 s. These
are check durations, not inference performance measurements.

Both profiles use context 1536, maximum input rows 16, two slots, explicit
head/verify capacity 4 and local ring 1280. Ordinary repeated verification
compares complete heads/features across eager execution, capture and replay;
whole discard restores the entire initialized checkpoint and old feature.
Acceptance lengths 1..4 compare against an independent normal four-row wave
with identical prefix and options: every head, exact semantic KV prefix and
selected feature agree; rejected write ranges restore their original bytes.
A projected peer checkpoint proves the subsequent ordinary continuation.
Actual writes across local ring wrap 1279..1282 restore exactly; invalid tails,
head capacity and pending-owner mutations refuse cleanly.

The first two attempts failed compilation before tests (a shadowed local and a
range-loop fixture copy). Two subsequent official runs each recorded 3 PASS/4 FAIL:
host pressure did not force refusal; the original scalar-prefix expectation
failed; a revised state oracle included padded future cells. All records and
source frames remain external. Product row invariance does not make scalar and
four-query attention identical. The final fixture instead checks a five-row
capacity refusal and hashes exact semantic prefix ranges, retaining independent
four-query, rejected-byte and continuation assertions. It claims neither
memory-pressure admission nor scalar parity. No tolerance or kernel changed.

## Provenance and remaining work

The six checked source files are frozen against 8cda4be in [results.json](results.json),
with source/queue identities, binaries, the actual SDK/build receipt and compact
identities for all five official jobs. The final job is
`m35-gemma-target-verify-check5` (9/9 DONE0); the four earlier attempts remain
failed records. Raw logs and checkpoints stay outside Git. The build receipt
has Git version `unknown` in the checksum-staged warm tree; the source frame is
the provenance authority. Only three targets and eleven focused controls ran;
the full suite remains deferred.

The task-entry [TensorFold primary recipe](https://github.com/ashhart/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/docs/recipes/gemma-4.md)
was observed at 0.6.6; its Gemma26 MLX recipe provides no comparable approved
Q8 CUDA assistant or 31B reference here. No new reference, quality, performance
or peak-memory acquisition ran. New post-write CUDA fault/cancellation cases,
assistant admission/recurrence/acceptance, independent batch chains and actual
whole-chain quality/performance remain separate work. Existing
[assistant component limits](../../gemma4-assistant.md#execution-work-still-owed)
remain in force.
