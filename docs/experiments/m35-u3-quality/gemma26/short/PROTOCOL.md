<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prospective Gemma26 short likelihood first pair

**No likelihood arm has run. Review and commit this registration before either
GPU collection.** This is one short native/source comparison following the
successful Paris pair. It does not enroll repeats, depth, a behavior ladder,
batching or speed, and cannot close U3. The earlier raw-ID failures, Paris
result and first CPU-preparation timeout remain unchanged.

## Immutable input, runtime and source

The [CPU preparation](PREPARATION.md) froze [ids.json](ids.json), SHA-256
`389216b176c694cd5f43e3068b20d6fad97e09cbd956bcd6c2837d6b400858b2`:
exactly BOS 2 plus the first 1,023 body IDs from the authenticated U3 corpus.
Both official tokenizer calls agreed on all 213,814 body IDs. No template,
additional BOS, truncation or retokenization is allowed at collection.

The [Paris identities and integrated-runtime qualification](../paris/PROTOCOL.md)
carry forward unchanged: source `nvidia/Gemma-4-26B-A4B-NVFP4` at
`a19cfe00be84568a6867111c9a68c9c44fdcffe6`, corrected artifact
`1d20ce879b2fe72c947699357da5e20d3f6388e9cc7858b9eb6fcc2516cd3875`,
native runtime SHA-256
`d18c7b7a9c8c6cf383a6f607a8606d2ce2026c4b2a8cb47632ec424601704dff`
and actual source manifest
`d0c8f0023776b0994de9e97beff04cb7bcd3e77ddad78bf87390bb312794cff4`.
The runtime's stale Git-derived version is not its source identity. Use the
existing A binary and verified artifact/source staging; no rebuild, import or
download is part of this control.
Keep collectors in a separate snapshot and pass A's unchanged
`/home/pmeenan/src/llmp` as native `--repo`. Its authenticated historical Service
bytes differ from current main's `speed_gate.py` after the U8 image-ID update.

Reference image:
`sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`.
The existing instanttensor 0.2.0 payload, parent API checks, callback transport,
source metadata and live backend inspection remain mandatory. The 37 installed
source files in [source-hashes.json](source-hashes.json), SHA-256
`80ad88e3b7f01de67e86f72c2ec59e3fdac6ec108ee0961bf434b1b6efe5a9b4`,
must match before and after source collection. The actual GPU container must
retain `VLLM_USE_V2_MODEL_RUNNER=1`, as the Paris launch did. No kernel, head,
weight-scale or arithmetic adapter is introduced.

Independent source review was CLEAN:
`/tmp/llmp-gemma26-short-source-review-r2.json`, SHA-256
`3277197af8d189adec0e42550d9ddacdebcaeb4b21acb60331498dd0058b5e99`.
The selected V2 path projects the model's head, trims vocabulary padding,
applies Gemma's normal final softcap 30, then normalizes all 262,144 vocabulary
logits. It includes the actual teacher independently of the reported top-1
alternative. `raw_logprobs` omits sampling transformations; it preserves the
model's head transformation. With prefill chunks of 512, target positions
1–512 and 513–1,023 are assembled once, preceded by an unscored `None` row.
The generated-token row remains separate. This source review establishes the
interface semantics, not numerical quality.

## One paired score request

Freeze execution order **native then source**, one fresh state per arm, on A
under two separate installed exclusive jobs, each at most 600 seconds with
30-second termination grace. Reauthenticate available runtime, metadata,
helpers, input, protocol and source pins before enrollment; record the full
reviewed registration commit in both receipts. Keep context 4,096, prefill
512, one slot, TP/PP1, no drafter or prefix caching, greedy T0/top-p1 and no
optional lossy modes. The source retains its registered NVFP4 W4A4 path,
BF16 arithmetic/BF16 KV and unquantized dense projections; native retains F16
KV. This precision difference is disclosed, not silently equalized.

Native submits literal IDs through `/v1/completions`, with `max_tokens=0`,
`echo=true`, `prompt_logprobs=1`, `logprobs=1`,
`return_tokens_as_token_ids=true`, `add_special_tokens=false`, no streaming.
Require the **complete** returned `logprobs.tokens` to equal all 1,024 frozen
`token_id:<integer>` labels. Its token count or an absent optional
`prompt_token_ids` field cannot substitute for this identity proof. Require
zero generated tokens and exactly 1,024 consumed input tokens.

Source uses ordinary `LLM.generate(TokensPrompt(prompt_token_ids=ids), ...)`
with the unchanged parent engine arguments and `raw_logprobs`. Only the
request sampling adds `prompt_logprobs=1` and changes `max_tokens` from 64 to
1: the pinned API refuses zero. Retain that one generated token but **exclude
it from every likelihood score and target count**. All original greedy options,
penalties, stop IDs `[1,106,50]`, seed and `ignore_eos=false` remain unchanged.
Require the actual returned prompt IDs to equal the frozen list, one completed
request, and exactly one unused generated token. No generated output is a
behavior control or a teacher target.

Both collectors require 1,024 prompt-score rows, first row `None`, and one
finite, nonpositive log probability at each actual teacher key for zero-based
positions 1–1,023. A non-top-1 teacher must still be present. Reuse the existing
`qwen27/contract.py` actual-target validation and `tools/quality_pair.py`
JSONL format: document 0, position, actual target ID, nonnegative NLL. Require
exactly 1,023 records in each arm. Save complete requests and returned values
before output assertions; retain raw failures, source warnings and all logs.

## Acceptance and stops

[pair-prereg.json](pair-prereg.json) fixes **native PPL increase at most +1.0%**,
with exactly 1,023 paired targets. This explicitly borrows the conservative
prospective U3 Llama/Qwen allowance; it is not a measured Gemma tolerance.
Compare the sole native loss file to the sole source loss file with the
authenticated `tools/quality_pair.py` (SHA-256
`9bf1d1597a8d6e97d65a648a9db68d8a51c694cc3772a6f01ba158fc420137bd`).
Report aggregate NLL/PPL, PPL change and mean/max absolute per-target NLL
differences, including positions. Do not substitute a favorable replicate,
widen the bound or claim repeat stability: this pair does not test U3's
separate 0.1% own-repeat bound. A pass is a preliminary short-screen result.

Identity mismatch, wrong/missing/reordered/extra targets, nonfinite or positive
scores, unknown dispatch, invalid source return, missing runtime dependency,
failed startup or unconfirmed retirement stops collection for diagnosis.
Preserve all raw evidence; no automatic retry or fallback follows. If native
collection is invalid, do not start source. If both collections are valid,
compute the fixed comparison and retain its PASS or FAIL without recollection.
Native uses the authenticated existing `Service` owner; source uses the
authenticated existing bounded Docker lifecycle. Both record helper/protocol
pre/post checks and exact retirement; source also records unchanged loader and
installed-source hashes. Verify the host has no surviving work before handoff.

## Reviewed implementation boundary

New files `control.py`, `native.py`, `source.py`, `source_launch.py` and
`test_control.py` bind this input and score-only contract. Parent Paris control,
raw Gemma reference/lifecycle, Qwen row validator and native service remain
unchanged and authenticate before import/use. Source reuses the model's
ordinary prompt-logprob interface; it does not patch the parent request globals
or implement its own forward pass. The GPU registration must include these
helpers, this protocol, the frozen input/source manifest and the numeric pairing
preregistration. Record actual file hashes in enrollment and launch receipts.
