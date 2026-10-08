<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 target-plus-assistant greedy transaction

One C1/P64/depth-three transaction passes the pinned original-image comparison
when the native target uses its existing norm/RoPE and norm/add chains. The
complete initial head, four verify heads, three assistant heads and all initial,
recurrent, verify and selected features are byte-exact. Both engines accept two
rows and carry the same pending next anchor; strict differences and relative
four-target conditional loss increase are zero. This is one matched transaction,
with no serving, scalar-width, 26B or performance qualification.

The first native screen used plain fused norms with both target norm chains off.
Its complete proposal agreed with the same original, but it failed three positive-
margin target choices, acceptance (one versus two rows), and conditional loss
(+133.96285%). That result remains a failure. A separately frozen native factor
changed only the benchmark's two existing target flags for profile 31 and added
selected-plan witnesses. Its actual prefill64 and normal4 plans each select 120
norm/RoPE and 120 norm/add steps. The assistant planner is unchanged. This
bounded factor restores exact outputs against the **same first original**;
no oracle reset, calibration or allowance was introduced. Profile 26 retains
its prior benchmark flags and has not run this transaction comparison.

Both targets prefill the canonical first 64 IDs in one actual query64 chunk,
with max rows/ubatch128, C1/context4096 and ordinary F16 split caches. The
assistant starts from the actual post-finalnorm feature at position63, drafts
three rows at constant query position64, and releases its frozen cache borrow
before verification of the authoritative anchor plus drafts in one query4 wave.
Only the retired accepted prefix is committed. Head and feature at keep−1 carry
an uncommitted next anchor; the last proposed row is not silently selected.

The native helper separately runs a normal Wave4 over the same proposal, then
clears and restores the exact full initial head/feature/initialized state using
query64 before the real engine unit. It checks full verify rows, actual final
draft head, accepted semantic KV and exact rejected-row restoration against
that independent control. Public `seq_rm` proves logical cell rejection and
accepted semantic rows; rejected physical bytes need not become zero. There is
no cross-engine KV-byte claim or scalar-versus-four-query oracle. Each engine's
complete finite first/repeat outputs are frozen independently before cross read.

The predeclared gates are proposal equality, zero positive-reference-margin
target choice differences (ties separate), accepted count/pending-anchor equality,
and at most 3% relative conditional loss increase. Four targets use the initial
head and verify rows0..2 under the generated proposal. Verify row3 remains
unscored. Conditional loss is not corpus PPL. The [frozen protocol](PROTOCOL.md)
records the baseline recipe; the final native benchmark adds the explicit31
norm chains described above. [Aggregate results](results.json) retain both
recipes and all completion/failure identities.

Native caller vectors are charged before allocation: a temporary 512 MiB bounded
vocabulary admission and retained 64 MiB caller envelope, with actual capacities
limited to 63 MiB plus 1 MiB metadata. Full state witnesses use catalog-funded pinned
buffers rather than uncharged host cache copies. Setup measures/funds target,
assistant, retained features and verifier workspace before startup. Original
known vectors use a separate 384 MiB bound and per-snapshot 64 MiB bound; library
and context occupancy is separate. No unchanged peak claim follows from the norm
factor. Successful retirement precedes release; unknown-effect owners are retained.
These paths inherit completion protection; this screen injects no new failures.

Both narrow builds pass three synthetic analyzer controls and pre-context invalid
profile refusal. The initial native acquisition failed before inference because
the reviewed source manifest was not copied; the identical retry succeeds after
that missing-file staging correction. The real baseline comparison fails and
its separately authorized post-authentication succeeds. All other producer,
own-freeze and candidate comparison jobs succeed; raw logs, IDs, caches and row
payloads remain external. No full suite or paid timing was run.

Reproduce with the manual `llmp_gemma_greedy_reference` target and
[`reference.sh`](reference.sh), which builds the pinned original
[`llama_greedy.cc`](llama_greedy.cc) under image 837fc732 and source b29c606e.
Native arguments are target artifact, assistant artifact, new output directory,
profile 31, canonical input IDs and `unit`; the public wrapper uses the retained
raw checkpoint pair and unique owned container. Use [`analyze.py`](analyze.py)
`own` separately after each official successful producer retirement, then
`compare` with both fixed own-proof hashes. Do not expose original recurrence
before native own-freeze, replace an existing output, or treat official rc0 as
numerical qualification. Input and retired identities are reauthenticated after
freeze. The [bounded fixed32 all-cost follow-up](../gemma-assistant-throughput/README.md)
now passes its continuation gates; a separately released teacher control and
wider performance/serving qualification remain future work.

TensorFold was re-pinned at entry to cb2ebf0540f42604e2759b2ddef497861e928248,
version 0.6.6. Its authenticated MLX recipe supplies no comparable CUDA Q8
assistant measurement. Current engine/serving defaults and support status are
unchanged.
