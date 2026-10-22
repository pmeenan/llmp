<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Proposed Gemma26 short likelihood first screen

**Inputs frozen; not GPU preregistration.** See the [preparation result](PREPARATION.md),
including its retained first timeout. Keep the completed Paris
result and its frozen helpers unchanged. Next ask only whether the native
runtime's likelihood on one short, identical teacher history stays within the
existing U3 screening allowance relative to this checkpoint's source engine.
One pair can find a mismatch; it cannot establish repeat stability or close U3.

## Input and first comparison

Reuse the existing U3 corpus, 817,275 bytes, SHA-256
`55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb`,
at `~/.local/share/llmp/corpus/m3-reference-corpus.txt` on Spark A. Its
[recorded provenance](../../../m36-baselines/README.md#provenance) is
`docs/decisions.md`, `docs/architecture.md` and `docs/plan.md` at `1facc957`,
concatenated. These are repository documents, not new private prompts or a new
download. Reauthenticate the existing bytes; do not regenerate from current docs.

The pinned source tokenizer will encode that corpus without special insertion.
Freeze exactly `[2] + body_ids[:1023]`, 1,024 integers, before registration.
Require source tokenizer BOS and generation-config BOS both equal 2 and no BOS
already in the selected body; otherwise preserve and stop, never strip/replace
IDs. There is no chat template on this corpus. The existing tokenizer's
postprocessor inserts no special tokens; its official chat template and
generation config establish BOS 2. Separate `encode` and direct tokenizer calls
must agree before freezing. Neither Llama nor Qwen's model-specific frozen IDs
can be reused. Their short-screen geometry is reused: positions 1 through 1,023,
1,023 actual teacher targets, document key 0; position 0 is unscored BOS.

Propose **one native/source pair**, fixed native-then-source order, fresh state
per arm. Reuse Paris's checkpoint, artifact `1d20ce879b2fe72c947699357da5e20d3f6388e9cc7858b9eb6fcc2516cd3875`,
runtime `d18c7b7a9c8c6cf383a6f607a8606d2ce2026c4b2a8cb47632ec424601704dff`,
image `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`,
loader and dispatch checks. Reauthenticate the existing pins and availability
before enrollment. Keep context 4,096, prefill 512, one slot/TP1, no drafter,
no prefix cache, T0 and the disclosed native F16/source BF16 KV difference.
No weight, scale, arithmetic, dispatch, template or source replacement is proposed.

Prospectively adopt the existing [Llama U3](../../PROTOCOL.md#geometry-and-acceptance)
and [Qwen U3](../../qwen27/PROTOCOL.md#corpus-scoring-and-acceptance) allowance:
native PPL increase **at most +1.0%**. This is an explicitly borrowed conservative
screening judgment, not a measured Gemma tolerance. Freeze it in the reviewed
registration before either score collection. A pass is preliminary: this one
pair does not test the existing 0.1% own-repeat bound. Report NLL/PPL, PPL change,
mean/max absolute per-target NLL differences and their positions. No favorable
repeat selection, threshold widening or automatic retry follows a failure.

## Existing API support and completed source review

Reuse `qwen27/contract.py`'s `native_records` and `ActualTargets` for actual-target
validation, and the existing `tools/quality_pair.py` JSONL contract and pairing.
Native accepts literal IDs with `max_tokens=0`, `echo=true`, `prompt_logprobs=1`,
`logprobs=1`, `add_special_tokens=false`. Preserve request and raw response before
validation; require the complete echoed `logprobs.tokens` to equal the frozen
`token_id:<integer>` list (a missing optional `prompt_token_ids` field is not
identity evidence), first prompt row `None`, exact IDs/count/positions and each
actual teacher key, including a teacher that is not top-1. Do not use a generated
token as a teacher target or infer native identity solely from token count.

The actual pinned vLLM `SamplingParams` exposes `prompt_logprobs`, rejects
`max_tokens<1`, and skips prefix-cache reads when prompt scores are requested.
`RequestOutput` exposes `prompt_token_ids` and `prompt_logprobs`. Therefore the
source candidate uses `prompt_logprobs=1`, `max_tokens=1`, normal literal input
and `raw_logprobs`; retain that one generated token but exclude it from all
1,023 likelihood targets. Independent review of the captured installed path
confirmed actual teacher inclusion, full-vocabulary normalization, chunk
assembly, the model head and runner selection. No source forward/head adapter
is needed. The exact observed module hashes are frozen in
[source-hashes.json](source-hashes.json) and must match the later registration.

## CPU-only preparation for review

`prepare_launch.py` reuses the prior bounded creation/inspection/retirement
mechanics, under one assigned installed shared job with 180-second supervision.
It uses the same immutable image, `runc`, no network or GPU devices/visibility,
two CPUs and 3 GiB; create/inspect/start deadlines are 20/5/120 seconds with
4+1-second exact-name retirement. It copies only the six already pinned metadata
files, the existing authenticated corpus and helpers into private scratch.
Only these copies are mounted, never checkpoint/weights. It records Docker
inspection and pre/post hashes of helpers, copied metadata and corpus.

`prepare.py` authenticates unchanged parent helpers before importing their
constants, authenticates the already observed vLLM source hashes and captures
installed source files/excerpts for prompt-score review without importing vLLM,
constructing a model or loading weights. It records the complete tokenization
diagnostics and exact 1,024 IDs. Newly captured source hashes are observations
from the immutable image. The completed preparation and independent semantic
source review are recorded in [PREPARATION.md](PREPARATION.md).

CPU preparation is complete. Freeze input/collector/runtime/protocol identities
and the +1.0% criterion in a separate reviewed, committed registration before GPU work. Then
two separate installed exclusive jobs, at most 600 seconds each, collect the
single pair. Missing/nonfinite/positive scores, identity/dispatch mismatch,
unknown teacher normalization, or unconfirmed retirement invalidate collection.
All raw output precedes assertions and stays outside Git. No depth, repeat or
behavior ladder is enrolled here; further controls need their own justified
prospective scope after this first screen.
