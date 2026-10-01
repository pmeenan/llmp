<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fixed long-context answer checks

The four frozen OFF/ON pairs completed on Spark A on 2026-10-01. Both arms
passed the 126,976-token variable-binding case. Both retrieved the correct
numeric values in the other two exact tasks but failed their required JSON
number types. Both answers to the separate practical code-review sample
failed its frozen rubric; the ON answer also exhausted its output budget.
These results leave candidate acceptance inconclusive and do not qualify
HCA as a default or change the existing greedy/PPL bounds.

The [pre-output protocol](protocol.md) fixed three self-authored synthetic
tasks and one separately judged code sample. It imported no external corpus
or evaluation implementation. Native CPU preparation authenticated the
artifact metadata, tokenizer, nonthinking template and stop IDs, rendered
the full prompts, and proved exact final token counts and complete fact
spans. Facts, questions, answers and rubric were frozen before model output;
only neutral filler was adjusted during bounded preparation.

Both arms used the native quality artifact
`8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234`,
the same checked executable, compact experts, frontier head, wide sparse
attention, greedy decoding and 2,048-row chunks. Exact mode, Q2 D2R and MTP
were off. Only the private `--ds4-hca` flag differed. Each arm loaded a new
process with fresh state, with no prefix or turn reuse. This artifact and
configuration are separate from the community complete-pipeline performance
reference.

## Exact answers and stopping

The scorer requires a single exact JSON array and authenticated EOS. R1/R2
require integers; R3 requires the five variable names derived from the
chronological bindings. Semantic retrieval below is a descriptive reading
of the preserved answers, not a replacement score.

| Case | Prompt / context capacity | Frozen score OFF / ON | Complete answer in both arms | Answer tokens OFF / ON | Stop OFF / ON |
| --- | --- | --- | --- | ---: | --- |
| R1 single retrieval | 30,720 / 32,768 | fail / fail | `["731942"]` | 4 / 4 | EOS / EOS |
| R2 four-key retrieval | 126,976 / 131,072 | fail / fail | `["620143","815207","390461","574829"]` | 13 / 13 | EOS / EOS |
| R3 four-hop bindings | 126,976 / 131,072 | pass / pass | `["v_seed42", "v_hop42a", "v_hop42b", "v_hop42c", "v_hop42d"]` | 34 / 34 | EOS / EOS |

R1 and R2 contain the correct values as strings. They fail the unchanged
type requirement in both arms; these baseline failures are uninformative
for candidate acceptance. The raw generated IDs match across each exact
task's pair. Their full logit values differ across arms. R3 supplies one
positive fixed long-context case, rather than a broad task-quality gate.

C1's complete saved answers were judged against the pre-output five-part
rubric. Both identify unread-header, arithmetic and deferred-buffer-lifetime
problems and propose owning queued payloads with a 65,536-byte cap. OFF
stops at EOS after 727 answer tokens but uses an overflow-prone initial
`offset + 4` check and changes the supplied `Consume(View)` API. ON keeps
that API and fixes the initial offset/header bound, but leaves body-end
addition unchecked and omits explicit maximum-valid and truncated-body
tests. It reaches the fixed 768-token budget without EOS. Both fail the
practical rubric. Their descriptive differences do not establish a
candidate regression because the OFF baseline also fails. The machine
scoring receipts retain `accepted: null` for this qualitative case; the
manual judgment is preserved separately.

## Measured scope

| Case | Prefill seconds OFF / ON | Decode seconds OFF / ON | Selected HCA calls OFF / ON |
| --- | ---: | ---: | ---: |
| R1 | 53.1851 / 48.2714 | 0.2370 / 0.2375 | 0 / 300 |
| R2 | 244.5986 / 231.6411 | 0.6997 / 0.6986 | 0 / 600 |
| R3 | 245.5448 / 231.2634 | 1.7578 / 1.7570 | 0 / 600 |
| C1 | 53.6721 / 48.9812 | 34.6005 / 36.6274 | 0 / 300 |

These are one fresh pair per task, with different C1 stopping lengths; they
are not a repeated performance qualification. The ON configuration is
mixed: the literal HCA consumer selects full 2,048-row chunks only when the
used padded compressed extent is 256 or 1,024. The middle extents 512 and
768 use ordinary attention. R2/R3 contain 62 chunks, with 30 eligible
chunks times twenty HCA layers giving 600 calls. No all-HCA 128K claim is
made.

All eight native processes and scoring processes exited zero and were
reaped under detached supervision. The controller authenticated every
saved 129,280-element F32 logit row, checked finiteness and exact lower-ID
argmax against the raw ID sequence, and retained EOS in the raw output
while excluding it from scored answer tokens. The eight outputs contain
1,604 saved rows including EOS where present. Full answers, IDs, logits,
summaries and score receipts remain outside Git.

## Scheduling deviation

The original conditional protocol required both R1 arms to pass before
scheduling the long pairs. That condition failed. Root separately cleared
R2/R3 to collect independent 128K evidence under the owner's request for
real-answer and long-context testing, announcing a 22–25-minute estimate
as needed now. The external deviation record was written after R2 launch
but before any R2/R3 answer, summary or model receipt was received or
inspected. It does not claim to precede generation or launch. The tasks,
type requirements, caps and quality bounds were unchanged, and no failed
answer was replaced.

## Provenance

Spark qualification passed the locked full suite of 1,204 tests, including
256 GPU controls, SDK format/tidy, boundaries, REUSE and 1,110 source
headers. Native CPU preparation took 1.377579 seconds. Historical compile,
token-boundary and warm-object failures are retained; the final check
invalidated every changed source/header/build input before rebuilding.

| Record | SHA-256 |
| --- | --- |
| Completed source qualification | `fbf92103e09ca9ae338d4803a44b5583754d4d63223e8fe98bafb14d9015f052` |
| Exact 1,230-file source map | `caebc8f6687e55877cc5dca9ec66943b1817bccc91a1cd9dcc62072e170a3051` |
| Measured `jitllm_dsv4_exec` | `1ff74abfc648a62182d507ab06eae54bf042b1b694048aefa365f40c1601fe3f` |
| Native CPU preparation/scoring helper | `86dcb79051a44ab5aab5f07f164e3acb346eb1831ef708ea8049f1bbb7e3361a` |
| Complete four-task preparation | `5776c2c3969d10a6c40c0e11e3255552d79e3f9384081ebce27977b0255aa1b3` |
| Frozen pre-output protocol | `b917ff2e8ecf9d57f03755f85b267819f04bf01dc3ca3dbf9e7bb509282adc9a` |
| R1 pair | `03668d5c96b9e2083661fedac2b8f1b8a00a383b7f0c004a332b2dc2ac049f65` |
| R2 pair | `4242dd9b58dc6b57a10ece8a560c179fecaeaabb820f4f55db20b9dde7d8606c` |
| R3 pair | `5457f8da099f88735dd4b3f3e21f27a519f8dcc0084aec2169c049b3384b4abe` |
| C1 pair | `9aaf3f7c487bf031c73d49bb282d33f4c4fb8d02edcdeae293e84774a547bb4d` |
| Final source/build/input/raw archive | `34a96be8b4df1fc4b6af8e6885ec8c925775dae801460b178aeb44112321e9c8` |

The SDK was `aarch64-e0a0c85c42806fb1`. The qualification and pair receipts
bind the actual compile database, native build receipt, loaded cuBLAS
libraries, tokenizer/template metadata, all thirteen prepared files,
controllers and strong preflight helper. The final immutable archive is
Spark A `~/scratch/m3-long-context-qa-final-handover-r1/`, with source,
qualified binaries and libraries, preparation, all four complete raw trees
and supervisor terminal records independently hash-audited. Its retirement
probe found 117.309 GiB free with GPU/container/native-model probes clear.
The original check and pair roots remain under
`~/scratch/m3-long-context-qa/`; the scheduling deviation and C1 manual
judgment are also preserved externally.
