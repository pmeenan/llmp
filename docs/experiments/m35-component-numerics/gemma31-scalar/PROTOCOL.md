<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 scalar schedule transfer: prospective first screen

One current-artifact native incumbent-MMVQ/VecQ pair tests the shared scalar
schedule repair. Collection requires this registration's review and commit.
This is neither raw-source checkpoint qualification nor permission to replace
frozen expectations. Historical Gemma31 frozen logits/state failures remain
unexplained beyond their identified arithmetic domains; token histories agree
in those retained controls. Gemma31's current weight inventory has no Q8_0:
304 Q4_K, 37 Q5_K, 70 Q6_K and 422 F32 tensors. The earlier Gemma2 Q8 repair,
its quality PASS and its speed result do not qualify this domain.

The reviewed generic schedule retains the incumbent's per-encoding small-K,
iteration-halving and IQ3 exclusions for actual GB10 dense one-token/no-ID
calls. The 174-cell operand matrix and eight fixed controls passed against
actual incumbent arithmetic (182 positive inner checks, five unchanged fixed
controls retained). Only nine formats support prepared MMVQ; the other nine
were compared with ordinary MMVQ and their exact prepared refusal checked.
This does not establish model quality. Ordinary multirow and routed-ID schedules
remain unchanged, and no owner-row-invariance policy is enabled here.

## Same binary, artifact and input

[pins.json](pins.json) freezes the actual runtime `5965e83c…`, its complete
2903-file source manifest, reviewed two-file delta, compiled module/compiler
receipts and runnable dependencies retained outside build directories. The
runtime-only build/retention job passed in20s; it neither changed source nor ran a model.
Use **Spark A for both arms**, one binary and the same resolved libraries.
Remote Git version strings are not the source authority.

Both arms use [node.toml](node.toml), current Gemma31 target
`090809981d2909085798273e665a0db58351a8e6029b13bc04ced857c9130943`,
context8192, two slots, **prefill_chunk1**, explicit F16 KV and no drafter.
The manifest/source provenance and all metadata hashes are pinned; five payload
shards retain their published full hashes and current stat identities. This
preparation read/hash checked metadata only, relying on the existing publication
and tensor-parity proof for payload bytes. No re-import or weight substitution.

Use unchanged `speed_gate.passage(corpus,100003,4000)`. Corpus SHA is
`55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb`;
4000-byte passage SHA is
`d94ee6ec0c784a57a6f363883b91d089525b1bffa8e141a8968035734986d03b`.
Existing CPU-only `llmp_gemma_tokenize` against this artifact's kept GGUF
metadata froze **1071 IDs, 1070 teacher targets**, one leading BOS2, vocabulary
262144. Complete IDs are in pins.json; their signed-int32 little-endian hash is
`e3c3bcd233290b3ec0219b64ada504eb90363d995af9a59b5acd89541cce40af`.
The helper's literal mode uses `parse_special=false` with the declared BOS;
the runtime completions route uses `parse_special=true` and inserts BOS when
needed. The fixed passage contains no angle brackets, but that observation is
not an identity proof: **every actual echoed ID must match the frozen sequence**.
No substitutions, trimming or special-token adjustments are allowed.
The helper's chat-only receipt fields have no effect in literal mode.

CPU preparation also freezes both greedy renders:26 tokens each, thinking off,
`gemma-4-unsloth` native-by-probe renderer, actual template hash
`845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b`.
The first metadata wrapper failed before tokenizer invocation by passing a Path
instead of text to the existing passage API; its supervisor remains retained.
The corrected metadata-only run passed in1s. No inference was collected.

## Actual arithmetic selection

Use the existing paid diagnostic policy interface, identically funded in both
arms: startup `auto`, node ceiling120259084288 bytes (112GiB, the existing
matched-control ceiling). Automatic construction admission remains active;
this ceiling is not a grant or bypass. Each arm sets the whole canonical target
policy, preserving normal `max_decode_rows 16` and `disable owner_padded_reads`.

- **A / old:** additionally disable `grouped_vecq`, `vecq_glu`, `vecq_inputs`.
  This restricts the candidate rewrites/selector and exposes ordinary/prepared
  incumbent MMVQ; source admission and all other component policies remain.
- **D / new:** the normal policy only. Existing authenticated component
  evidence supplies whatever qualification the runtime admits. No
  `candidate_numerics`, rank override, fake approval or source-code change.

This contrast asks whether the repaired selected scalar VecQ products preserve
current incumbent quality; it does not compare the previously failing default
schedule or establish any speed improvement. Require actual completed score
plans with stage0, phase4 (`kScore`), rows1, one owner/segment, **exactly1071
completed uses in total**, and all descriptor/selection records.
`PromptSession::NextUnit`/`NextPhase` execute through `tokens_.size()`
(`src/runtime/llm_prompt.cc:135–176`); score publication omits only the final
head, which has no following teacher (`:298–320`). Thus all1071 input rows
execute while only1070 teacher scores are published.
 A must select incumbent MMVQ and no
`llmp.vecq.*`; D must actually select dense VecQ for Q4_K/Q5_K/Q6_K operands and fused
VecQ GLU for Q4_K/Q5_K operands. The actual
GLU gate/up inventory is Q4_K in54 layers and Q5_K in6; Q6_K occurs in
attention V (44) and FFN down (26), never GLU. Audit actual operand encodings/K, including Q4_K/Q5_K
K5376, against the retained index and complete descriptors. Preserve preparation,
product and GLU decomposition differences; require attention/Rope/norm selections
to agree. If the intended paths are not exercised, or unrelated arithmetic
changes, report an invalid discriminator and stop; do not add a numerical grant.

The automated inventory/shape/count audit is necessary but **does not declare
an arm valid**. Before D, root or its independent reviewer must inspect A's
full completed plans and write a distinct `arm-a/manual-plan-review.json`.
Before pairing, the same requirement applies to D and its comparison with A,
in `arm-d/manual-plan-review.json`. The builder does not issue these receipts.
The runner refuses D without A's receipt and refuses pairing without both.

Each receipt has schema1, arm `a`/`d`, `arm_valid:true`, `approved_by:"/root"`,
a named independent `reviewer`, `review_type:"independent-full-plan"`, and
`files` exactly equal to the external `review_bindings(arm)` result. These
bind audit, complete plans, launch, pre/postflight, quality/full score response,
raw runtime log, rendered config, runtime and registration bytes; D additionally
binds A's manual review, plans, audit, quality and raw log. It records1071
`score_completed_uses`,1070 `teacher_targets`, true review fields
`per_encoding_recipe_operands_reviewed`,
`allowed_preparation_glu_differences_reviewed` and
`unaffected_arithmetic_descriptors_reviewed`, plus substantive `encoding_paths`,
`normalization_and_unaffected_operands`, `allowed_differences` and `conclusion`.

Review follows each selected recipe to its **own** weight operands, encoding,
K and shape: dense Q4_K/Q5_K/Q6_K and GLU Q4_K/Q5_K, including the affected
K5376 products. VecQ elsewhere cannot stand in for Q5_K still on MMVQ.
Normalize graph-local tensor/node/storage IDs by their operand relationships;
compare unaffected arithmetic and full operand/output descriptors, not just
aggregate norm/Rope/attention counts. Explicitly account for permitted input
preparation and fused-versus-separate GLU differences. A's review establishes
its complete incumbent baseline; D's review establishes the actual contrast
and unaffected equivalence. Missing/inconclusive proof makes the discriminator
invalid and stops this screen; it does not trigger another experiment ladder.

## Fixed collection and validation

Use the existing `capture.load_speed_gate` wrapper with debug plans, its
420-second readiness and20-second retirement, and `speed_gate quality` with
`--family gemma31 --label old|new --retain-streams`, the pinned explicit config,
corpus and runtime. The explicit config replaces speed_gate's historical default
Gemma31 artifact. Pass the three exact per-arm `--runtime-env KEY=VALUE` values
from pins.json as argv elements (policy newlines remain actual newlines).
The concrete executor freezes complete argv and helper hashes before launch.
The pinned external executor is `/tmp/llmp-u4-gemma31-quality/run_arm.py`;
its preflight and audit are direct adaptations of the retained Gemma2 wrappers.
With `REG=/tmp/llmp-u4-gemma31-quality/reg`, materialize the full committed
registration and its helper snapshot before execution. The registration receipt
must contain that full commit and explicit reviewed status; its preparation-only
form cannot launch collection. The actual commands inside separate installed
exclusive600/grace30 jobs are:

```sh
~/.local/bin/mise exec -- python3 -B /tmp/llmp-u4-gemma31-quality/run_arm.py a
# Only after root/reviewer supplies the byte-bound manual A receipt:
~/.local/bin/mise exec -- python3 -B /tmp/llmp-u4-gemma31-quality/run_arm.py d
```

After both automated audits and independent manual arm-valid receipts pass,
the gated pairing command calls the unchanged registered `quality_pair.py pair`
with the exact old/new quality paths and existing Gemma31 bound file:

```sh
~/.local/bin/mise exec -- python3 -B /tmp/llmp-u4-gemma31-quality/run_arm.py pair
```

Authenticate every registration/helper/config/input/runtime/resolved-library,
retained-source and artifact-metadata pin before enrollment and after retirement;
check payload stat identities against this preparation. Preserve raw ldd
transcripts, but compare resolved path/hash maps and load-address-normalized
transcripts, not ASLR addresses. First exercise full preflight and the real
collector CLI parser with collection stubbed; no service may start in that check.

Then use **two separate installed exclusive jobs, A then D**, timeout600/grace30,
fresh private outputs named in pins.json, idle GPU and no other host jobs. One
service launch consumes each arm's attempt. No retries, fallback, extra prompts
or adaptive inputs. Stop before D if A has any infrastructure, response, identity,
alignment, retirement or selected-path failure. Preserve semantic negatives;
complete D once after a valid A. A valid numerical FAIL is retained unchanged.

The unchanged collector runs two sequential C1 greedy requests, temperature0,
max_tokens128, followed by the one score request. The prompts remain:

- Explain, step by step, how a hash table handles collisions.
- Write a short story about a lighthouse keeper who finds a map.

Retain full SSE/client records before validation. Require two nonempty complete
responses, normal stop/length, valid usage,26 input tokens each and at most128
generated tokens; audit actual prose for coherence and task fulfillment, retaining
negative content. Greedy token differences alone are not the enrolled NLL rule.

Score request: literal fixed passage, max_tokens0, prompt_logprobs0, echo true,
logprobs0, return_tokens_as_token_ids true. Require all1071 echoed `token_id:<id>`
values exactly equal the frozen list in both arms; leading null score; exactly
1070 singleton teacher keys at positions1–1070 matching the echoed IDs, finite
nonpositive log probabilities, prompt usage1071 and completion usage0. Retain
raw responses before assertions. Generated greedy tokens never enter this score.
Use `quality_pair.from_speed_gate` for the established teacher alignment checks
and independently compare the complete echoed sequence; do not rely on row count.

Pair once with unchanged `tools/quality_pair.py pair` and existing
`docs/experiments/m36-final/prereg/pair-gemma31.json`:1070 targets, absolute PPL
delta≤0.5%, mean absolute NLL difference≤0.05 nats, max≤1.0, and at least95%
within0.1 nats. Its historical v0/v1 description is historical; this registration
fixes current arms and the independently prepared count. Report each bound,
actual texts, complete score/plan/raw hashes and successful TERM/returncode0
retirement. Require both independently reviewed, byte-bound arm-valid receipts before pairing.

Existing stream clocks yield only observational TTFT, end-to-end time and decode
rate `(completion_tokens-1)/(t_last-t_first)` and A/D ratios in prompt order;
disclose unequal counts. Debug logging, one sample and this quality workload do
not constitute a D-085 speed gate. No timing job is enrolled. A PASS supports
choosing the next bounded affected-domain control, not Gemma26 joined rows,
Gemma31 assistants, frozen-state parity, full family quality or U3/U4 closure.
