<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DS/DSpark and Qwen prefill prediction — 2026-10-08

Scalar DeepSeek/DSpark and native/MTP/GGUF Qwen now select the shared two-stage
CPU plan lookahead and eligible fresh-zero backing preparation. Complete heads,
histories and initialized target/drafter states match the ordinary control.
The two short repeated injected recipes take 0.562% and 0.936% less prefill time;
the three plain pairs are correctness and timing diagnostics, not individual
speed claims. [Aggregate, identities and all payload hashes](results.json).

| Actual recipe | Context/chunk, two slots | OFF/ON prefill seconds | Evidence | Additional temporary plan floor |
| --- | --- | --- | --- | ---: |
| DeepSeek plain | 16384/4096 | 12.192344 / 11.275946 | n=1 diagnostic | 9,914,912 B |
| DSpark injected | 16384/4096 | 11.367555 / 11.303662 | n=2, −0.562%; OFF movement −0.329% | 9,994,784 B |
| Qwen native plain | 2048/512 | 2.289619 / 2.271748 | n=1 diagnostic | 7,489,920 B |
| Qwen native MTP | 2048/512 | 2.312797 / 2.291160 | n=2, −0.936%; OFF movement +0.482% | 7,945,984 B |
| Qwen GGUF plain | 2048/512 | 4.140569 / 4.104058 | n=1 diagnostic | 6,701,632 B |

DSpark OFF ranges 11.348802–11.386307s and ON 11.295690–11.311634s;
MTP OFF 2.307241–2.318353s and ON 2.285463–2.296856s do not overlap.
These small n=2 results concern the combined planning/backing policy, not either
component in isolation or complete generation throughput. DSpark explicitly
selects its supported speculative wave form for the off-paid companion proof;
this is not a default adaptive-wave end-to-end comparison. Other adaptive/depth,
selected-head and arithmetic policies remain the actual serving recipes.

Both arms provision lookahead/preparation identically. The internal OFF control
withholds only hints. The table's real host-plan allowance covers two independent
DS plans or two target plus two MTP plans; the aggregate also records the derived
ordinary false-option floor. Activation, scratch, staging, common plan floors,
paid graph paths and model work are identical per recipe. Do not interpret the
matched funding as zero cost relative to the older unprovisioned policy.

## Bounds and ownership

`PredictPrefill` checks current/near/far bounds without allocating or publishing
future state. Actual input builders authenticate pure compressor/QSA geometry.
All current and cached future keys are protected before reclamation-capable
charges; DS future lookup never repatches an in-flight HCA plan. Independent
grants precede CPU descriptor construction; driver choices are captured before
Job. Each plan transfers its charge only after successful completion/binding.
Qwen predicts MTP from post-success pending1 and funds its companion independently.

Fresh backing starts before the current Job and drains before outputs, cursors,
pending rows or future cache installation. Postwrite collection failure
quarantines affected state. The existing shared refusal/cancellation/retirement
proofs carry unchanged. DS's caller-supplied Status meanwhile retains its legacy
Submit/Await contract, including page-in and failure behavior; optional CPU work
is disabled in that diagnostic branch. No CUDA or acquisition occurs in the new
CPU-only callback.

Actual UsedState controls prove target extent 143 becomes fresh from 4096 to 4352
at DS context 16384/ring4352; the same short recipe at context 8704 has no eligible
fresh extent. Qwen scalar context 2048/read256 adds target extent 87 from 512 to 1024;
prebacking with the decode read2048 would hide this boundary. Each ON process
submits/adopts two fresh target extents. The MTP fixed hidden carrier leaves no
fresh drafter extent in this recipe; only its independently built companion
plans are claimed. Host geometry tests also cross the real QSA boundary at 2048;
the short actual Qwen prompts keep QSA off.

## What ran

Five unique focused host cases cover shared hint bounds/disappearing suffix,
DS actual compressor geometry, both real fresh-extent boundaries, and Qwen
QSA/tails/pending0/1. The final selected-default build reruns only the two changed
policy/geometry cases. The unchanged shared group lifecycle tests were reused.

One supervised 600s acquisition completed 14 fresh processes in 411.6s:
three OFF/ON plain pairs and two OFF/ON/ON/OFF injected factors. Each activates
weights, warms three scalar tokens, clears and drops plans. The timer starts
after initial pure reuse and encloses the first actual interleaved scalar prompt
traversals: four DS chunks or six Qwen chunks, including input staging, state
growth, planning/overlap, preparation drain and current frontier publication.
Setup, observation/copy/hash/persistence and subsequent generation are excluded.

Off-paid controls compare 224 complete finite heads across 14 processes, all
choices/histories and complete initialized target/drafter state ranges/digests.
They include actual joined generation/draft/verify, nine/three outputs, actual
C2→C1 cohort departure with protected peer state, owner0 spill/restore and stable
checkpoint continuation consuming the final pending anchor. Observation owns an
explicit 64 MiB host charge and bounded 1 MiB pinned chunks; an unproven teardown
retains the whole heap lifetime. Every child and installed supervisor returned 0,
boots stayed stable, new kernel messages were empty and GPU compute was absent
after retirement. Actual SDK/receipt/private cuBLAS identities are in the aggregate.

The factor source is the additive 9c5be88 tree, manifest 16d60fd0. Final defaults
compose with main f5f09ca's bit-exact host interval fills/validators; that commit
already contains the four pure geometry files. The final incremental build
passes with both selected-default assertions. No factor was repeated for this
math-preserving source bridge. The aggregate distinguishes executed and selected
source inventories and lists the five approved default/assertion deltas.

## Replay after raw-result cleanup

Build `llmp_prefill_prediction_probe` with the pinned Spark SDK. Use the five
prepared artifact manifest/index pins and actual library hashes in results.json;
no raw checkpoint, interpreter or new artifact is needed. The checked-in probe
contains the paid endpoint and all model/state/retirement controls. Run it under
installed GPU-exclusive `spark-job`, timeout600/grace30, with 120s per process,
checking each return and retirement marker before the next arm. Keep per-arm boot,
kernel cursor/new-error and empty-GPU guards as in the acquisition. Local builds
or payload analysis use `hostlock shared`.

Required tiny inputs are retained on Spark B under
`~/.local/share/llmp/references/prefill-ds-qwen/{ids0,ids1,qwen0,qwen1}.i32`.
DS inputs 4352/4608 are verified copies of Spark A's standing `ds4-mask4` inputs;
the [original text/tokenization recipe](../deepseek-device-masks/README.md#checks-and-provenance)
recreates them if needed. Qwen 1536/1280 are little-endian I32 arrays from the
matching-length `decode[].prompt_token_ids` in the retained
`references/qwen-device-masks/inputs/wave-prompts.json`; parent/slice hashes and
counts are in the aggregate. Preserve these standing inputs through milestone
raw cleanup. Tokenizer/template come from the approved Mia925d7be6 directory.

Invocation (private fresh output directory; target/drafter are artifact IDs):

```text
llmp_prefill_prediction_probe ds|dspark|qn|qmtp|qg STORE TARGET DRAFTER|- TOKENIZER_DIR|- IDS0 IDS1 NEW_OUT off|on
```

Use three plain pairs and the two repeated factors in the table. Require 16
full-vocabulary finite F32 files per arm, exact byte/hash equality of every
head/choice/history, all three state-boundary digests and ranges, acceptance/work
and common budgets. Require positive ON plan caching and submitted/adopted
backing; native MTP additionally installs companion plans. OFF submits none.
Plain pair timing remains n=1; repeated factors report means/ranges/bookend drift.

Capture ahead and state-only output cuts remain T23/T54 work. Production wide
joined prefill remains T55 OPEN: DS decode waves are one-row, Qwen target waves
cap at four; widening needs explicit admission, matcher/output and funding work.
T67's broader context/cohort/output/kept-source qualification remains OPEN.
This scalar transfer changes neither reference arithmetic nor the original
Qwen fast-FP8 row 29 quality lead, and makes no public HTTP/parity/PPL claim.
