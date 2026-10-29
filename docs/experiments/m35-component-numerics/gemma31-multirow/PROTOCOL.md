<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 ordinary multirow quality registration

**Prospective registration; collect only after its reviewed commit.** pins.json
binds the Spark B source snapshot, runtime, dependency closure, build receipts,
artifact metadata, helper bytes and executors. The external registration starts
with `preparation_only=true`, `reviewed_registration=false`, `commit=null`.
After committing this protocol, root records that commit and enables the external
registration; the executor refuses collection before those checks pass.
The immutable source manifest (2,920 files, SHA
`425ca8f4df184d19ce8c5a467cfc7d03b4c3665d70370e65577970bed48bdef3`) is the
source authority; the build receipt's remote Git commit `61f8426f...` is retained
as lineage and is not substituted for source identity. Runtime SHA is
`84510e35229fe96255737700abde85472706df51ed6f9d71ebc5f6298e3c3a37`.
The source-based C2 owner inference is recorded and hash-pinned in
[source-inference.json](source-inference.json).

The question is whether the current ordinary multirow VecQ path preserves
incumbent quality. No production repair is presumed necessary: the two-cell
operand discriminator found approximately1e-6 default differences, below the
existing NMSE bound, while incumbent geometry removed those differences exactly.
Completed scalar quality/speed evidence remains separate. No frozen expectation,
source grant, family policy, quality bound or artifact is changed. This first
screen covers the current Gemma31 **target only**; assistants, Gemma26, DSpark,
context/depth qualification and speed remain outside this collection.

## Fixed inputs and two distinct execution domains

Use the same immutable runtime and resolved libraries for both arms on Spark B,
current target090809981d2909085798273e665a0db58351a8e6029b13bc04ced857c9130943,
[the config](node.toml), F16 KV, context8192, two slots and **prefill_chunk2**.
No drafter. Authenticate published metadata and payload identities on B; the
old A device/inode/mtime observations are historical and are not B pins.

Reuse the scalar registration's exact fixed corpus/passage, full1071 IDs and
both chat inputs from pins.json. Corpus SHA55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb;
4000-byte passage SHA d94ee6ec0c784a57a6f363883b91d089525b1bffa8e141a8968035734986d03b;
IDs int32-LE SHA e3c3bcd233290b3ec0219b64ada504eb90363d995af9a59b5acd89541cce40af.
The metadata-only tokenizer used parse_special=false; runtime literals use true.
Require complete exact echoed-ID equality, one BOS2, leading null score row and
all1070 actual teacher keys/finite scores. No special-token substitution.

1. **Teacher score: one owner, ordinary two-column chunks.** The unchanged
   completions request uses echo, prompt_logprobs0, max_tokens0 and literal token
   labels. Require phase4/kScore, stage0, **535 completed uses with rows2 plus
   one use with rows1**, one segment/owner each:536 uses,1071 executed input rows,
   1070 teacher targets. Count completions multiplied by each plan's row count;
   do not confuse536 uses with1071 rows. Audit every completed plan, including
   the final single-row tail. Config alone is not shape proof.
2. **Chat behavior: two owners, one decode row each.** Barrier-launch exactly
   the two fixed QUALITY_PROMPTS in pins.json, T0/max_tokens128, using the
   existing stream_chat HTTP/SSE implementation. Bind results by prompt index,
   not completion order. Retain each exact request body reconstructed from the audited pinned collector,
   complete SSE, parsed reply,
   usage, finish/error and actual completed decode plans. Require genuinely
   joined two-row/two-owner decode, inferred from the complete pinned ordinary
   caller chain and both requests completed; two simultaneous HTTP requests alone are insufficient.
   Installed headers must explicitly report segments2. Transient completed
   headers omit segments: retain a separate source-inferred evidence record,
   without fabricating a header field. `Single` emits one decode row per owner,
   no-tool generation supplies no forced spans, continuation decode peers and
   prefill continuations use one row, and `CheckShape` enforces distinct slots.
   Manual review must also confirm actual two-row dense non-ID products.
   Preserve any scalar tail after one request finishes. The one-owner score
   proof cannot substitute for this two-owner proof.

One service per arm: barrier-paired chat first, then the fixed teacher passage.
Both requests remain live to retirement if one fails; no replacement request.
Retain normal stop or length finish and positive completion counts up to128,
exactly as the previous behavior control; the cap is not a demand to ignore EOS.
Fully retain and independently parse the raw SSE, including DONE and usage.
Any transport, truncation, parse, execution/retirement or actual-domain failure
invalidates that arm and stops enrollment of the next arm. No retry or adaptive
prompt/scheduling change.

## Incumbent arithmetic and manual plan gates

Both arms retain the existing paid diagnostic interface: startup auto,
node cap120259084288, max_decode_rows16 and disable owner_padded_reads.
A additionally disables grouped_vecq, vecq_glu and vecq_inputs; D uses normal
policy. No owner_row_invariant, candidate numerical grant or rank override.

The policy names do not prove a genuine pre-U4 arithmetic baseline. Before D,
root/reviewer must trace every affected A product to its own operands and the
retained pinned upstream MMVQ/formula source, establishing incumbent quantization,
reduction, GLU decomposition and head behavior for the actual row count. Record
this provenance explicitly against the retained historical source/plan evidence.
If A silently selects another changed arithmetic path, mark the contrast invalid.
Do not inherit an old scalar manual receipt.

For D, require actual dense VecQ Q4_K/Q5_K/Q6_K and GLU Q4_K/Q5_K on their own
operands, with actual K/shape/row geometry, in both execution domains. Gemma31
has zero Q8_0 weights; Q6_K is not a GLU gate/up encoding. Expected GLU weights
are Q4_K54layers/Q5_K6layers. Preserve physical dtype names (q4_K etc) and
semantic encoding names, compare normalized names only after checking consistency.
The earlier scalar auditor's capitalization bug must not be copied.

Manual A review precedes D; manual D-versus-A review precedes pairing. Reuse
byte-bound receipts over registration, runtime, source, full plans/logs, config,
requests/SSE/score responses, launch, audit and pre/postflight. Review each domain
separately. Require unaffected attention/Rope/norm arithmetic and full operand
and output descriptor correspondence after normalizing graph-local IDs by their
relationships. Explicitly account for allowed input-preparation and fused/separate
GLU differences; aggregate implementation counts are insufficient. Missing or
inconclusive bindings stop this screen.

## Existing numerical and behavior criteria

Use unchanged tools/quality_pair.py and the original Gemma31 family bound file:
1070 aligned targets; absolute PPL delta≤0.5%; mean absolute NLL delta≤0.05;
maximum absolute NLL delta≤1.0; at least95% within0.1. Full response and per-target
scores precede validation. No comparison to the completed scalar score as a
replacement for this actual multirow A arm.

Report the two actual replies alongside their A counterparts for independent
coherence/task-completion and material-regression review. Do not infer semantic
PASS from HTTP success, a digest, numerical PPL PASS or matching lengths. Preserve
inherited failures in both arms. Exact reply bytes are observed, not a new
acceptance requirement. Numerical and behavior outcomes remain distinct; this
small screen is not general family quality qualification or permission to refresh
frozen oracles. A valid numerical failure ends this screen without revised bounds.
Debug timings are observational only; any matched C2 speed adoption is separate.

## Minimal executor reuse and prelaunch closure

Reuse the scalar preflight/run/audit/CLI-stub structure and existing capture,
greedy_capture, plan_reports, speed_gate and quality_pair modules. Parent helper
paths/hashes in pins.json identify historical source to adapt, not executable
pins for this draft. Keep the original scalar evidence/helpers unchanged.

Only the necessary changes are planned: private BASE/SUB/output paths; concrete
B runtime/source/artifact pins; preparation-state refusal; score completion shape
histogram535×2+1×1; physical/semantic dtype handling from the reviewed recovery;
C2 full-plan domain audit; and barrier-pairing the existing two quality prompts.
The reviewed collector now has an opt-in quality-chat concurrency option, with
serial default unchanged, calling stream_chat twice with its existing barrier
and distinct raw files. Its exact checked bytes are bound in pins.json. The
run_cell corpus prompts do not substitute for these fixed quality prompts. No new inference
framework or separate scoring implementation is needed.

Parser-only prelaunch validation stubs the real service/collection methods and exercises
both complete real CLI forms (old/new labels, explicit config, exact policy argv,
concurrency2 and all paths), invalid-arm stop, manual-gate refusal and full
preparation-state rejection. No service or model construction occurs in that validation.
The retained receipt records actual helper hashes, parsed arguments and all review-gate controls. Materialize
only the committed registration/helper snapshot into fresh private scratch.

Concrete enrollment after those fields exist: installed Spark B exclusive jobs,
600s/grace30 each, execute the adapted existing run_arm.py a; inspect all raw
and source guards and issue independent A receipt; execute run_arm.py d once;
inspect both complete plan domains and issue D receipt; execute the unchanged
quality_pair invocation through the existing gated pair step under short shared
supervision. All private names/absolute argv must be frozen in pins before the
registration commit. No source sync/rebuild merely for documentation. Same pinned
runtime/dependencies before/after, ASLR-normalized ldd identity, TERM0 retirement,
no remaining GPU processes. Root owns final review/registration; executor does
not self-issue manual validity gates.
