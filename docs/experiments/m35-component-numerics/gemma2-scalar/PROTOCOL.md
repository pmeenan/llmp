<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma2 scalar Q8 transfer: prospective first screen

This amendment applies U4's existing affected-family quality requirement to
Gemma2. No collection is authorized until this amendment is reviewed and
committed. It is one representative A/D pair, not a Gemma family qualification,
assistant test, speed gate, or permission to refresh frozen expectations.

The unchanged frozen Gemma2 control passes immediately before U4 (`a666deb0`)
and fails at `3cf3118c`, with the same ten differences as W7/e033. Actual scalar
plans change from 105 prepared MMVQ, 26 GGML MMVQ and 26 fused GEGLU selections
to 130 `llmp.vecq.dense`, 26 `llmp.vecq.glu` and one prepared MMVQ selection.
Attention and Rope selections remain unchanged. Both initial 18-row prefill
heads still match. This supports intended Q8 arithmetic transfer; it does not
establish an accuracy defect. U4's earlier A/C functional PASS and C-to-D
unchanged-path statement do not establish A-to-D scalar likelihood quality.
Raw failures and historical controls remain retained.

## Fixed arms and input

[pins.json](pins.json) binds both original U4 diagnostic runtimes, all ten
resolved dependencies, their original source manifests/patches, and the unchanged
Gemma2 Q8_0 artifact. The runtimes and cuBLAS libraries were copied, individually
verified and resolved outside build directories at
`~/.local/share/llmp/baselines/m35-u4-gemma-screen`. No current build receipt or
remote Git version is used as an old runtime's identity. Retained source trees
match all 2,688 A / 2,694 D manifest entries. Existing SDK/driver qualification
and the original U4 diagnostic patch apply; no rebuild or runtime switch occurs.

Both arms use [node.toml](node.toml): the same canonical artifact,
context 8192, two slots, **prefill_chunk=1**, default F16 KV, no drafter or lossy
alias override. Requests are sequential C1. Only fresh data directories, arm
labels and runtime paths differ. The embedded GGUF tokenizer/template metadata
is SHA-256 `03bfd1bac861b68641f726cfb6adde758bffe830d434b8d772f2756d813a50af`;
there is no tokenizer override, source-template edit, or external tokenizer.

Use `speed_gate.passage(corpus,100003,4000)` without alteration. The original
817275-byte public corpus hash is
`55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb`.
The 4000-byte passage hash is
`d94ee6ec0c784a57a6f363883b91d089525b1bffa8e141a8968035734986d03b`.
The original Gemma2 quality receipt and existing registration establish 1070
teacher targets. No new tokenization or inference preparation is needed.

The native runtime tokenizes the fixed passage through the existing literal
completions route. Audit the **actual** full 1071 echoed `token_id:<id>` values
in each raw score response after collection: identical A/D sequences, exactly
one leading BOS 2, leading null score, 1070 subsequent singleton teacher keys
matching those echoed IDs at positions 1–1070, and finite nonpositive scores.
No generated token is requested or counted. Do not substitute IDs or trim rows
if these checks fail. The full ID lists and their explicitly encoded hashes
belong in the result receipt; no pre-run ID list is claimed here.

`ScoreChunkRowsFor` in both frozen sources takes the minimum of configured
prefill rows and paid row capacity; chunk selection preserves that upper bound.
With one row configured, all teacher scoring traverses scalar products while
retaining complete history. Require successfully completed target score plans
with **one row and one owner/segment**, and actual A/D product selections that
show the intended transfer. A wide score plan would miss this question and
invalidates the screen. Preserve all plans, not just selected examples.

## One collection per arm

After chat's Spark B jobs retire, authenticate runtime/library, helper, config,
corpus and artifact metadata hashes before enrollment and after retirement.
Authenticate payload identity against the retained same-inode/size/mtime receipt
(the earlier artifact publication already verified its listed full hash).
Record installed supervisor, host/GPU vacancy, registration commit, exact argv,
raw outputs and actual dependency resolution. Use two separate installed
exclusive jobs, **A then D**, each timeout600/grace30, fresh private output/data.
Stop after infrastructure/identity/invalid-response failure; retain semantic
negative evidence and complete D once if A's collection is valid. No retries,
input selection, fallback, extra prompts or widened bounds.

Use the existing `capture.load_speed_gate` service wrapper (420-second readiness,
20-second retirement, `--debug-plans`) and the existing `speed_gate` quality
command. Only the opt-in `--retain-streams` addition changes retention: raw SSE
is written before parsing, and full existing client timing records are retained.
No request, model math or sampling behavior is changed. The fixed greedy
requests remain the two existing `QUALITY_PROMPTS`, temperature0, max_tokens128:

- Explain, step by step, how a hash table handles collisions.
- Write a short story about a lighthouse keeper who finds a map.

With `REG` the private snapshot of the committed registration and `ARM` either
`baseline` (A, label `old`) or `candidate-d` (D, label `new`), the supervised
command is the following, with a distinct fresh `OUT` for each arm:

```sh
PYTHONPATH="$REG/docs/experiments/m35-component-numerics" \
~/.local/bin/mise exec -- python3 -B -c \
'import capture,pathlib,sys; repo=pathlib.Path(sys.argv.pop(1)); raise SystemExit(capture.load_speed_gate(repo).main())' \
"$REG" quality --family gemma2 --label "$LABEL" \
--runtime "$HOME/.local/share/llmp/baselines/m35-u4-gemma-screen/src/runtime/m35-u4-$ARM-diagnostic-runtime" \
--config-template "$REG/docs/experiments/m35-component-numerics/gemma2-scalar/node.toml" \
--corpus "$HOME/.local/share/llmp/m36-baselines/inputs/corpus-1facc957.md" \
--retain-streams --port 18114 --out "$OUT"
```

Retain raw score responses and SSE, error records, runtime logs, pre/post pins
and TERM/returncode0 retirement. Parse retained SSE with existing
`greedy_capture.parse_sse`; require two nonempty complete replies with normal
stop/length finish and valid usage, at most128 generated tokens. Compare the
actual texts for task fulfillment/coherence; different greedy tokens alone are
not a quality failure. Preserve any anomalous text for review without changing
the enrolled numerical rule.

## Unchanged decision rule

Pair the existing quality receipts once using `quality_pair.py pair` and
[the existing Gemma2 bound file](../../m36-final/prereg/pair-gemma2.json):
1070 targets, absolute PPL delta at most0.5%, mean absolute per-target NLL
difference at most0.05 nats, maximum at most1.0, and at least95% within0.1 nats.
The old bound file's historical v0/v1 description is not this comparison's
identity; the A/D arms and same artifact/config are fixed above.

```sh
~/.local/bin/mise exec -- python3 -B "$REG/tools/quality_pair.py" pair \
--candidate "$OUT_D/quality/gemma2-quality-new.json" \
--reference "$OUT_A/quality/gemma2-quality-old.json" \
--prereg "$REG/docs/experiments/m36-final/prereg/pair-gemma2.json" \
--report "$RESULT/pair.json" --json
```

The retained existing stream clocks permit a coarse observation for each prompt:
completion count, first-token latency and `(completion_tokens-1)/(t_last-t_first)`.
Report actual counts, end-to-end time, and A/D ratios; do not compare unequal
counts as equal work. These are two fixed-prompt single samples with debug-plan
logging, not a completed10% speed gate. No timing ladder or extra generation is
enrolled. A numerical PASS permits choosing the next bounded affected-consumer
control; it does not by itself validate Gemma26/31, assistants, frozen states,
all-row domains, or U3 closure. A FAIL retains the bounds and initiates diagnosis.
