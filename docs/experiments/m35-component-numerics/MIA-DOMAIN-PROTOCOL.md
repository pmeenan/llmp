<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prospective U4 target and MTP qualification scope

This clarification is fixed before the additional Mia captures below. It
changes no numerical acceptance bound, source reference, input or runtime.
The D registration's statement that changed native arithmetic requires the
independent 1016-target Mia comparison also encompasses its changed drafter.
Teacher-forced target likelihood cannot measure a drafter. Treating that
target-only comparison as MTP qualification, or silently waiving the stated
comparison, would both give the evidence the wrong scope.

The completed ordinary captures establish a narrower fact: D and A have
identical target arithmetic and concrete operands at the captured native target
shapes, including score rows 997, and identical 996 target losses. The MTP
stage's HC and NVFP4 expert operations still differ. That 996-target result
remains informational; it does not replace the independent Mia comparison.

## Additional fixed target control

Use the exact frozen A and D executables, dependencies, current configuration
and source pins from [llm-capture-pins.json](llm-capture-pins.json). Keep MTP
configured identically in both arms. Use the retained Mia inputs from
`spark-b:/home/pmeenan/scratch/m36-qwen-quant-reference03`, with no reference
rerun or new reference tokenization:

| Retained input | SHA-256 |
| --- | --- |
| `inputs.json` | `50110476b05de2b2a87e012b46cff967d179c48917406797368309eb3f27e25d` |
| `losses.jsonl` | `42da92d6471da854c910b02f821b29433392a1c5ec1b806a1fb0f2d4256f65cf` |
| `manifest.json` | `b3371be28d315ac9d919444305a9c80bf2519e027e2d9e3af34f69d37bc2287b` |

Authenticate all four retained document texts too. Each document has 255 fixed
input IDs. Score actual targets at positions 1 through 254, yielding exactly
254 targets per document and 1016 total. Preserve the exact first ID in the
echo while leaving it unscored. Reject missing, extra, nonfinite, positive or
wrong-target probabilities, mismatched IDs/positions/usage or changed input
hashes. Use the production full-vocabulary scored reducer; no top-one
substitution, generated labels or restricted-vocabulary normalization.

Run both additional arms on Spark B (`spark-56f5`, NVIDIA GB10, driver
580.178.04). Its ten resolved dependency hashes must match the qualified
frozen arms; do not substitute its current build executable. This leaves
Spark A available for the separately registered timing jobs.
Run one complete four-document capture for A and one for D, each in its own
exclusive supervised job of at most 600 seconds, with fresh private output and
service state. Keep readiness 420 seconds and retirement 20 seconds, under the
outer supervisor's lifetime. Record exact requests/responses and successful-use
physical plan inventories, including reused installed plans. Associate completed
target uses with each document; inventory is not a per-kernel execution trace.
Associated completions must be target stage 0, score phase 4, with positive row
counts totaling exactly 255 per document. Capacity-driven chunks may split that
total. One synchronous request at a time in the isolated service supplies the
request owner; the diagnostic schema does not invent physical owner IDs.
Authenticate source/helper/configuration/runtime identities before capture and
the runtime again after successful retirement. Preserve every failed attempt.

Require exact A/D target-ID alignment and equality of all 1016 target losses.
Also require equivalence of the actual target operations, product arithmetic,
concrete operand/output formats, dimensions, strides, views, recipes, routing
geometry and roots. Report identity-only metadata changes separately. Missing
or ambiguous plan evidence, or any actual target difference, fails this reuse
control; do not infer equivalence from aggregate NLL alone.

Separately pair **each** arm with the retained Mia losses using the unchanged
[qwen38-mia.json](../m36-final/prereg/qwen38-mia.json): absolute aggregate PPL
delta at most 1%, each document at most 4%, mean absolute target-loss difference
at most 0.25 nats and maximum at most 5.0. Report every strict result, including
failures. The historical M3.6 target path already had an explicitly accepted
FP8-KV/BF16-SSM limitation; see [the final record](../m36-final/README.md).
Only proven-identical target arithmetic and its complete target-loss control
can retain that existing target limitation. It supplies no exception for
changed target arithmetic or for the changed MTP stage. This protocol creates
no new quality exception and widens no threshold.

## Changed drafter controls

Qualify the changed MTP numerical implementations with the existing independent
HC FP64 formula and NVFP4 selected-expert/whole-path oracles, preserving every
assertion and requiring actual candidate selection with no skip. The final
integrated source must also pass the real-model MTP ladder comparison, full
logit/default-schedule controls and owned joined-MTP state/replay control.
Those tests establish arithmetic and state properties, not source-model corpus
quality. Retain the registered greedy prompts and semantic reviews, including
truncated or reasoning-only replies, and the separately registered
[C1/C2/C4 matched speed screen](LLM-SPEED-PROTOCOL.md). Changed MTP receives
no exemption from those controls through the target's historical limitation.

Promotion requires all applicable controls and an honest classification of the
strict Mia results. An incomplete capture is not a completed strict comparison;
a completed strict comparison is not automatically a quality pass. Full U4
remains open until its integrated checks, affected controls, evidence review
and matched speed obligations are complete.

## Frozen collector and invocation

mia_capture.py, SHA-256
`736fb71535c6b33969e1830ddba04d8d137c93d84c6d5fe8facc0d7d7bb89461`,
enforces the collection contract above. Its additional
input pins have SHA-256
`bc246e63a64a9aa84235e539c57daa3a4a43a7f0f2312b1992e17d2b4b587e2b`.
The existing capture helper, physical-plan parser, speed helper, quality-pair
helper, arm pins and strict Mia preregistration are authenticated too.
Independent whole-change and adversarial review is clean after enforcing
score-phase and complete paid-row coverage. All 36 CPU helper controls passed;
mutating away either coverage check fails its regression. These checks invoke
no model and imply no numerical result.

After this registration is committed, stage the collectors with their unchanged
repository-relative helper/preregistration layout on Spark B. For each arm use
an unused private `OUT` directory and an installed exclusive 600-second job:

```sh
/home/pmeenan/.local/bin/mise exec -- python3 -B /ABS/HELPERS/mia_capture.py \
  --repo /ABS/STAGED_REPO --arm A --runtime /ABS/FROZEN_A \
  --reference /home/pmeenan/scratch/m36-qwen-quant-reference03 --out /ABS/OUT
```

Then use D, its frozen executable and a different output directory. Inspect
`capture.json`, exact target losses, per-document paid plans and waited
retirement. `capture_complete: true` is separate from `mia_strict.status`;
strict FAIL remains a reported failure. Run no alternative inputs or bounds
to replace an adverse result.
