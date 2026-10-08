<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hang recovery and conversations kept across a restart (2026-10-03)

D-102's part B (the owner, 2026-10-03: "A genuine hang should be
recovered. Canceling, evicting or even restarting the whole process if
needed. Minimizing data loss but recovery is the main goal") and D-105's
conversations kept across a restart. This report checks each rung of the
recovery with a forced hang, the restart's adoption of kept conversations
against an uninterrupted control, with the refusals of a tampered and a
foreign record, and what the records' background hashing costs a swap.
Everything ran on `spark-b` (`spark-56f5`, GB10, driver 580.178.04,
systemd 255) with the `spark-native` build of this change on main
`09eddbe`, except where a row says it ran on the first build (on main
`c1dde25`, before the review's fixes). The harness is
[restart_check.py](restart_check.py); its raw records stayed outside the
repository.

## Setup

The service (`llmp-runtime` with no command) serving DeepSeek V4 Flash
(artifact `8a355bfb…`, DSpark drafter `dd2d3f9c…`, context 32,768) and
Qwen3.8 Flash Next (artifact `c4fb47a9…`, MTP drafter `8600a998…`,
context 33,792, prefill chunk 4,096) on loopback, greedy (`temperature`
0), non-streaming. The conversation: R1 and R2 are two turns to Qwen3.8
(R2 resends R1's reply, reasoning included); R3 is one turn to DeepSeek,
whose swap writes Qwen3.8's state back to its spill file; R4 is Qwen3.8's
third turn; R5 is DeepSeek's second (R4's swap writes DeepSeek's state
back, or a graceful stop spilled it).

The test hook `LLMP_TEST_HOLD_READS` (a file; while it exists the
node's reads are held: `engine/paged_node.h` CountingStorage) has two
forms. By default a held read behaves as one a drive or a hung mount
holds: the lane's cancellation does not end it (io_uring's cancellation
is best effort; `providers/direct_reader.cc` cancels and still waits for
the completion), and it completes only once the hold lifts. With
`LLMP_TEST_HOLD_READS_CANCELLABLE=1` a held read completes as cancelled
when the lane cancels it, as a read still queued would. The first build's
hook was the cancellable form only, so its rung-2 evidence did not cover
a real stuck read; the independent review reproduced that case under a
systemd user unit (three requests failed after 63/234/234 s, never
restarted: rung 2 reset the model with the read still in flight). The
ladder now judges a cancellation drained only once no storage operation
submitted before it is still in flight (`PagedNode::oldest_io`), and the
stuck form below reaches rung 3.

## Rungs 1 and 2 through the service (reads still queued)

`hang_seconds = 60`, `stall_seconds = 1`, the cancellable hook set after
R1, then a new conversation sent to Qwen3.8: its state's growth reads its
fresh extents from the slot's spill file, which never landed.

| Event | Time from the request |
| --- | ---: |
| Watchdog: no progress (report only) | 2.6 s |
| Rung 1: confirmed hang, the node's wait cancels its request | 60 s |
| The cancellation drained (the held reads completed as cancelled; nothing older in flight) | +0.0 s |
| Rung 2: the cohort's request fails, Qwen3.8 fenced, reset in place and evicted | — |
| The client's 503 `backend_hung` | 62.6 s |

Then the hold was lifted and R2 sent: Qwen3.8 loaded whole again and R2
reused 172 of its 192 prompt tokens (R1's conversation, idle and
untouched by the failure, was kept and written back by the reset's
eviction). The process never exited.

## Rung 3 through the service: a stuck read, under systemd

The service as a systemd user unit (`systemd-run --user`, Type=notify,
Restart=on-failure, RestartSec=5s), `hang_seconds = 60`, the hook in its
default form. R1, R2 and R3 ran (Qwen3.8's conversation written back and
recorded), then the hold was set and a new conversation sent to DeepSeek.

| Event | Time from the request |
| --- | ---: |
| Rung 1: confirmed hang, the node's wait cancels its request; the wait returns, the read stays in flight | 60 s |
| Rung 2 waits for the cancellation to drain; it does not | — |
| Rung 3: "a storage operation it submitted 120 s ago is still in flight 60 s after the cancellation"; the queued records written; exit 1 | 120 s |
| The client's connection closed (no reply: requests under way end with their connections) | 120.1 s |
| The process gone (MainPID changed; the hold lifted here) | 123.8 s |
| systemd restarted it (NRestarts 1); it adopted Qwen3.8's kept conversation (265 tokens, 2 turn checkpoints, 446 MiB) and was ready | 131.8 s |

R4 then reused all 265 tokens and replied as the uninterrupted control
did, reasoning and answer. R5 replied as the control did but prefilled
(0 of 23 cached): DeepSeek's conversation was resident at the exit, so
it was lost, as D-105 says.

## Rungs 1 and 3 on the device

`cuda_paged_node_test` (GPU), the node's patience set short:

| Case | Rung | Result |
| --- | --- | --- |
| A page-in whose reads are held, cancellable | 1 | cancelled after 0.5 s without progress; drained; the load fails flagged as a hang's; once reads flow it loads and reads back whole |
| A request's lease whose materialization is held, cancellable | 1 | likewise: BeginRequest fails flagged; the request opens later |
| A page-in whose reads a drive holds (not cancellable) | 1, then 3's condition | the wait returns failed and flagged as a hang's while the read is still in flight (`oldest_io` set, from before the cancellation); once the hold lifts the read completes, `oldest_io` clears, and the weights load and read back whole |
| A job whose stream waits on a value that never comes (`cuStreamWaitValue32`), as a program and as a request's step | 3 | cancelled after 0.3 s; nothing drains; the patience reaches rung 3 after another 0.3 s (recorded in place of the exit; then the gate opens, the wait drains flagged, and the node reads its weights back whole) |

The ladder itself (`hang_ladder_test`, a synthetic clock) covers the
rest: a read still in flight keeps the cancellation undrained however
much else moves (rung 3 after the grace) and one that completes lets it
drain; CPU work that beats its pulse is never a hang, one that stops
beating is asked to stop (rung 1) and drains when it does, and one that
never reaches a checkpoint restarts; a model reset twice without serving
a request between escalates. The chat route's watch (`api_test`): a
unit hung in a node wait is cancelled and the route goes on; one hung
where nothing beats or cancels is cancelled, then restarts once.

## Restart adoption

| Scenario | Build | How A stopped | B's start | R4 cached tokens | R4 against the control | R5 cached tokens | R5 against the control |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| Control (one service, no restart) | both | — | — | 265 of 285 | — | 8 of 23 | — |
| Kept | this | SIGTERM (5.0 s: Qwen3.8's and DeepSeek's records, 1.8 GiB hashed in 2.1 s) | 3.5 s, 2 adopted (736 MiB) | 265 of 285 | equal: answer and reasoning | 8 of 23 | equal |
| Stuck read, rung 3 (above) | this | exit 1, systemd's restart | 1 adopted (446 MiB) | 265 of 285 | equal: answer and reasoning | 0 (resident: lost) | equal |
| Crash | first | SIGKILL once Qwen3.8's record existed | 3.5 s, 1 adopted (446 MiB) | 265 of 285 | equal: answer and reasoning | 0 (resident: lost) | equal |
| Tampered | first | SIGTERM; one byte of `slot-0.state` flipped | 3.0 s, Qwen3.8 refused: `its digests differ`; DeepSeek adopted | 0 | prefilled whole (a near-tie reply, as chunking allows) | 8 of 23 | equal |
| Foreign | first | SIGTERM; B with Qwen3.8's context 32,768 | 3.0 s, Qwen3.8 refused: `its state layout is not this runner's`; DeepSeek adopted | 0 | likewise | 8 of 23 | equal |

In every scenario R1–R3 replied identically. A refused record's files
were removed at the start, the rest of the directory kept; the adopted
DeepSeek conversation in the refusal scenarios shows a refusal is the
record's alone. R5's cached count is the same as the control's because
DeepSeek's reasoning is dropped from the resent history: the turn
restores its turn checkpoint before the reasoning, from the adopted file
after a restart. On this build B's start runs with the calibration A
recorded (D-103), so Qwen3.8's draft-depth cost differs from the one the
record saved: the record is adopted at the machine's cost, no longer
refused.

## What the records' hashing costs a swap

One service, R1 to Qwen3.8, then three alternations of DeepSeek ("Say
hello in one word.", 24 tokens) and Qwen3.8 (R1's prompt again), each
request sent as the last returned, with conversations kept (each swap's
write-back hashed in the background) or not (`[memory]
keep_across_restart = false`):

| | Kept | Not kept |
| --- | ---: | ---: |
| Swap Qwen3.8 → DeepSeek, ready (three) | 9.415 / 9.450 / 9.471 s | 9.469 / 9.474 / 9.510 s |
| Swap DeepSeek → Qwen3.8, ready (three) | 7.636 / 7.689 / 7.703 s | 7.725 / 7.761 / 7.759 s |
| DeepSeek request, mean of three | 10.36 s | 10.40 s |
| Qwen3.8 request, mean of three | 10.37 s | 10.42 s |

No cost shows within the run-to-run spread (the kept run was the faster
one); every Qwen3.8 reply equalled R1's. The kept run's stop hashed
3.5 GiB of records in all (4 written). The hashing threads run at nice 19
and the idle I/O class and wait between extents while a swap or a
prefill runs.

## Caveats

- A device that hangs cannot be recovered in-process; rung 3 restarts
  the process, and conversations then resident (not wholly on disk) are
  lost. The device case was checked with a gated stream in the unit test,
  not through the service.
- A request under way when rung 3 exits gets no reply: its connection
  closes.
- Kept conversations are adopted only by the same build (the executable's
  identity is part of a record): an upgrade starts cold.
- A conversation whose last verify's restore is owed while a cohort holds
  the model's stream is spilled but not recorded (the record must hold the
  state whole).
- The crash, tamper and foreign rows ran on the first build; the code
  they exercise (the record, its checks and adoption) changed since only
  in the draft depth's cost, the time checks and the checkpoint cleanup.
