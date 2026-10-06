<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 real-serving bridge preregistration

This registration governs a candidate, internal default-false serving bridge.
It changes no production default, public setting, model support or arithmetic
implementation. The native serving caller uses `Server` and `Llm::Branch`;
quality and timing therefore include the actual generation frontend. The
source base is 73b89f4, with exact current v0.6.0 native GGML inputs. Final
integration must preserve newer Gemma3 and benchmark changes from main.

## Closed work and lineage

Only the approved Gemma 4 31B UD-Q4_K_XL checkpoint, C1 and C4, are admitted.
Native context is 8192, explicit maximum prefill rows 256, headcap/slots equal
the cohort, ordinary joined generation, existing norm-ROPE and norm-ADD ON,
owner attention requested. C1 selects primitive attention; C4 must select
60 owner-attention layers in its actual last-built decode plan. Existing
unopted31 and26 recipes and calibration identities stay unchanged.

Four independent literal paragraphs come from the retained authenticated
3,274,124-byte corpus (SHA c7156148ecaa12b6416cf816540d8dede2014982554a835e61076f0dd8bf0c2d).
The text manifest froze before tokenization. Both native and original vocab-only
input preparation must agree on all first 8192 IDs per owner, with one leading
BOS, before the model stage. The original input remains unchanged and is used
for natural warm/paid cycles and the corpus control.

Each representative trajectory has 8063 authentic prefix IDs and 129 emitted
argmax IDs. Exactly 128 generated anchors are consumed; the initialized cursor
is 8191, and the final prediction is pending at position 8191. Stops/EOG are
ignored in both engines. This is a bounded fixed-length greedy continuation,
not a completed natural response. Prefill uses independent 256-column chunks
per owner, including the actual final 127-column chunk. All 129 heads,
including the prefill frontier, qualify. Row 128 is unscored.

Native first/repeat full heads, IDs, history, initialized state/layout and
successful retirement freeze independently before FIRST new d812 output.
Only after its complete finite/exact own proof does the controller derive
stock anchors from authenticated native emitted IDs 0..127 and the unchanged
prefix. Stock quality consumes those exact histories. Stock first/repeat
full heads/IDs and its actual initialized driver/cache/CID evidence then
freeze independently before cross comparison.

The quality gates are all129 argmax IDs exactly equal (a differing tied ID
also fails), zero positive reference-margin differences, and relative
whole-vocabulary FP64 conditional NLL increase <=3%. Exact ties and full-head
byte identities are reported separately. The 128 scored targets are the
**frozen native emitted IDs 0..127**, authenticated from its own proof; these
are generated-history conditional likelihoods, not an independent corpus
PPL measurement. Exact-ID lineage across all129 rows is required before
interpreting the stock forced trace as its own deterministic greedy trajectory.
No inherited empirical bound, noise calibration, C5 exception or new allowance
is available. Finite gross-loss overflow records null plus an explicit FAIL.

## Mandatory current-source corpus control

After the representative quality-gated bookends complete, use the first owner’s first1024
unchanged authentic input IDs in native `Branch::ScorePrompt` and public scalar
query1. Own first/repeat full1024 heads and native initialized state/layout
freeze before FIRST original corpus output. Score the1023 authentic next-token
transitions; the final head is unscored. Gates are strict zero positive margin
and independent <=3% relative PPL increase. Report tied choices, absolute NLL
and absolute PPL separately; no old source-vintage1K parity or semantic quality
claim carries forward. This control is required for adoption, not a broader
context/depth suite.

## Public and completion admission

The original helper uses unchanged release v0.6.0, exact d812 source/image
ARM manifest c604, the approved raw checkpoint, physical C1/C4, context
8192 per sequence, batch/ubatch256, F16 local1280/global8192 and normal ring,
default graphs/fusion. Actual loaded libcuda is observed without loading an
alternate driver; cuInit/current-context/device identity precede paid work.
Runtime dependency/header/source/binary/SDK identities are authenticated.
Raw-model approval SHA is inherited; current regular-file name/length is
checked, avoiding a repeated18GB payload scan. Vocab-only preparation validates
bounded scalar metadata because d812 skips typed hparams in that mode; full
model execution retains typed60-layer/5376-width/vocab262144 checks.

Every producer is a separate installed Spark B GPU-held job, timeout600,
step timeout300 and stop-on-failure, with official wait. Exact successful
job/final/steps/log hashes and the output-log/CID hashes admit the own proof.
Proof and file lengths are rechecked before later use; full retained SHA work
is done once in the supervised cross/exact stage. Native unknown completion
retains config/node/callback/session and pinned owners until process exit;
no unproven pointer is freed. Public containers have unique names/CIDs and
owned retirement checks. Completed request retirement and teardown markers
are necessary, not substitutes for official success.

## Whole-serving bookends

For C1 then C4, run its quality pair and separate postauth; only that quality
PASS releases its fresh stock/native/native/stock bookends. The mandatory
current-source corpus follows both screens before any adoption decision.
An earlier completed screen remains evidence if a later gate fails.
Each process has one untimed warm and one paid cycle. Admission/model load and
activation are outside paid work. Paid work includes Clear/branch selection,
all prefix rows, all128 generation waves, full-head publication, one argmax
per output, token publication, Finish and successful request retirement.
Public uses an already emitted anchor, avoiding a duplicate vocabulary scan.
The final full-head diagnostic copy is included in both engines. Archives,
full native state exports, source guards and final process teardown are outside
the clock. Component walls are reported without subtraction or causal attribution.

Minimal paid loops archive no intermediate full heads. The separate exact
checker requires every emitted ID, final full head and native complete
initialized state/layout/history to match that engine’s frozen quality own
proof. It admits actual warm/paid logs, physical cache geometry and unique
CID retirement for all four bookends. Report means and both within-pair
spreads as a short whole-serving screen, with no sustained claim.

## Funding, controller and remaining adoption boundary

The native caller charges64MiB before bounded vectors grow; a separate
catalog-funded16MiB pinned slice streams complete state. Existing runtime
weights/state/history/sampling/plan budgets remain authoritative. At cursor8191
initialized proof bytes are1719664640 per owner (6878658560 forC4); corpus1024
proof bytes are922746880. Public known vectors are checked against1100MiB
before allocation, including its largest1GiB corpus trace. Require100GiB
free external disk before every producer. Zero extra scanner work is claimed
inside paid timing; full identity/finite scans are supervised outside it.

The workstation controller advances only reviewed commands and literal
actual hashes. It never holds an outer remote GPU job or starts nested holds.
Each new stage uses exclusive queue/receipt/output names. FIRST failure stops
all dependent work and preserves its official records/aggregate; separate
source-only postauth runs even after strict failure. No reset, rescore,
new stock oracle, source retry, tolerance change or automatic adoption follows.
Actual HTTP stop/cancel/continuation and existing strict lifecycle controls,
followed by an explicit reviewed production adoption decision, remain owed.
The current production executable keeps the candidate flag false.
