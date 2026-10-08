<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded Gemma26/31 scalar serving

The two approved text artifacts share one native serving adapter. The factory
opens the trusted artifact, uses architecture and expert count only to choose
a candidate, and requires complete approved tensor binding before construction.
Setup repeats that binding. The selected existing engine variant is immutable;
there is no filename override, new configuration schema or independent lifecycle.

Both profiles accept plain chat and literal completions, target likelihoods and
up to twelve independent scalar request owners. Prefill is capped at 128 rows,
context at 262,144, and the uncalibrated fallback remains one owner. Device masks
remain native. Optional math policies remain off. Thinking, generated tool
calls and assistants/speculation remain unavailable. Literal completions retain
the generic API's `stream=false` restriction; chat supports SSE.

These controls establish bounded admission and own-engine continuity. They do
not qualify model support, optimized joined batching, reference quality,
competitive performance or long-context depth. The original ordinary dense31
[failed quality screen](../gemma31-runner/README.md) remains unchanged. The
optional [norm policy's 1,024-head identity](../gemma-native-norm/README.md) is
neither selected here nor a qualification of its separate paid 8K shape.

## Identities and memory

Source base is `622dfe5`. The approved artifacts are:

| Profile | Prepared artifact | Source GGUF revision |
| --- | --- | --- |
| 26B-A4B | `4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3` | `c099eb48e663fd284577b04978a94ffccb261841` |
| Dense31 | `32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08` | `c1ac76e99d5513b141e8adde7288b85c3f9c32ec` |

Dense31's complete source is `gemma-4-31B-it-UD-Q4_K_XL.gguf`,
18,822,970,304 bytes, SHA-256
`9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575`.
Its prepared artifact was deeply verified before this task. This slice changes
neither import nor binding/state formats. Both actual templates have SHA-256
`845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b`.
The existing 65 normalized Conversation fixtures compare native rendering,
checked interpreter bytes and target token IDs with the retained pinned Jinja
oracle for each actual artifact. This does not establish HTTP tool-history
normalization: the generic parser refuses those messages.

At task entry, 2026-10-05 04:54:13 UTC, TensorFold HEAD was freshly queried as
`609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5. Primary coverage
still provided Gemma26 on MLX, with no dense31/CUDA comparator. No new reference
inference or speed comparison was performed in this serving task.

The selected profile determines the runner's weight/state/plan envelopes.
Before sampling, the startup guard includes each owner's two complete F32
vocabulary rows and twice-vocabulary 16-byte sampling-candidate capacity:
10,485,760 bytes per owner, 125,829,120 bytes for twelve owners. Candidates
allocate their fixed capacity from empty storage and release on every session
Close, including cancellation, pause and failed initial sampling. They cannot
remain allocated behind a switch to another model's workspace floor.

Each configured runner separately catalogs pinned output storage for its
maximum row count: 134,217,728 bytes at 128 rows and vocabulary 262,144.
Native source staging, host-built input bounds, charged plan/graph retention,
request memory and lazily materialized per-slot KV remain separate charges.
The native fixture controls additionally fund retained own head copies before allocation;
row comparisons use spans without temporary full-vocabulary copies. No claim
about measured physical peak or context-wide allocation is made.

## State and output controls

The parameterized serving fixture uses both approved profiles and checks:

- Complete own frontier rows and seeded token histories agree across solo
  and scalar owner counts 1, 2, 4, 8 and 12; stable captured plans replay.
- Snapshot rollback, valid turn reuse, initialized-state spill, kept restart
  and corrupted cursor/boundary records preserve exact own continuations.
- A redigested kept record carrying the opposite profile's layout is refused
  before adoption. The untouched peer restores from its actual retained tag.
- Teacher-forced rows match independent one-token own frontiers, score the
  next supplied ID, and stop at exactly the callback's completed prefix.
  A 128-ID scorer interleaves and resumes without repeating reported scores.
- A callback can stop one owner while its peer finishes; branch history
  publishes at Finish, with exactly the completed tokens before the anchor.
- Clean state-capacity refusal preserves the nonzero completed prefix and
  state occupancy while an active peer progresses. Each profile's actual KV
  growth boundary is checked before creating pressure; after release, the
  refused owner produces its own uninterrupted token history.
- Completed paused generation retains owned output and callbacks through
  both 26→31 and 31→26 switches, restoring and resuming each token once.
- Large `top_k`, failed first sampling, cancellation and normal completion
  release all fixed sampler capacity. Plain chat refuses normalized thinking
  and generated tools; settings refuse assistants and speculation.

The fixture owns its server and all borrowed configuration in one stable heap
bundle. Unproven teardown retains that entire bundle and its state files;
a failed retirement is sticky across every mid-test and final teardown call,
so a repeated no-op cannot substitute for proof. Startup failures stop
each actual caller before branch access. Every model fixture uses the existing
CTest `MODELS` resource lock.

The actual HTTP harness checks both profiles' solo/cohort token output, chat
SSE, stop-string suppression, tool refusals, disconnected-reader followup and
127 supplied-token scores aligned after a null first row. Pending switches
start the other profile after the first chat SSE publication and compare both
owned results with their own solo output. The existing `model_turn_seconds=1` setting and 128-token chat generations
trigger bounded time slicing. A positive source-specific runtime pause-message
delta is required independently for each direction, authenticating that its
source request was active. Aggregate results
and successful retirement are retained in [results.json](results.json).

## Validation and limits

Validation uses the locked native Spark-b SDK `aarch64-c09daba6ac31edee` and
declared GGML prepared tree
`026f1ac94af98011933e71cee240d0a405ac4220cb208356acf61aa2bcea6207`.
The first seven parameterized controls passed for both profiles. A later
historical run failed three CTest targets because the new fixtures incorrectly
expected history publication before Finish, ordinary same-length prompt reuse,
and the same KV allocation boundary for both profiles. A follow-up compile
rejected an implicit size narrowing in the corrected boundary assertion.
The corrected affected controls passed; those failures were not discarded.

The first HTTP run passed six profile-specific controls but failed its two
switch controls because the harness requested unsupported literal SSE. The
corrected harness uses chat SSE and surfaces refused streams immediately.
A second HTTP run reproduced every own-output case but failed its pause
evidence check: the default 30-second scheduling turn let 32-token responses
finish. Those results alone do not establish active switching. The final narrow
run uses the existing one-second turn and checks each direction separately.
The focused Spark check passed 122/122 CTest targets without skips (74.43 s).
After the sticky teardown correction, all three affected restart/switch targets
(six model cases) passed again without skips (95.98 s). The six other HTTP controls passed for both profiles in the retained
second run; its final result remains failed because the pause proof was absent.
The final supervised switch-only run passed both directions: 128 source tokens
for 26→31 and 110 for 31→26, one active-source pause each, exact own SSE and
peer likelihood output, and runtime exit zero. Official job identities
and limits are recorded in the aggregate results. The first integrated full
suite was stopped through the supervisor after the retirement finding; it is
not a passed suite. The corrected final integrated Spark-b suite passed
1,655/1,655 targets without skips (424.47 s). Raw logs remain external.

Optimized serving requires a separate joined runner path and a real
multi-sequence pinned reference: common token IDs, explicit sequence ownership,
matched KV/masks, complete head/state comparisons, departed/cancelled owners,
captured replay, likelihood/publication alignment and paid batch bookends.
Scalar cohorts and individual-request reference calls cannot substitute for
that comparison. Fresh reference selection, context/depth and full support
gates remain separate.

See [PROTOCOL.md](PROTOCOL.md) for the bounded reproduction commands.
