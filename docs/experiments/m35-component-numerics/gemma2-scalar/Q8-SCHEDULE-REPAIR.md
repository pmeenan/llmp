<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prospective dense Q8 scalar schedule repair arm

This supplements [the registered scalar protocol](PROTOCOL.md) with one repaired
candidate, **R**, against the exact retained A collection. Review and commit this
supplement before inference. A is not rerun. The original A/D **FAIL** remains:
1006/1070 targets (94.0187%) within 0.1 nats, below the unchanged 95% requirement.
The two original hash-table openings' misleading claims remain negative semantic
evidence. This is one representative Gemma2 screen, not broader Gemma/Qwen
qualification, a speed gate or permission to refresh frozen expectations.

## Candidate and prerequisite evidence

[q8-schedule-repair-pins.json](q8-schedule-repair-pins.json) freezes the new runtime,
all 2883 source files, compiled module/receipt, actual resolved dependencies,
retained dependency bytes, the exact two-file repair patch, passed controls and
original A raw files. The source is root's W10 union plus the two repair files;
the remote Git label `61f8426f` is stale and is not its source authority. The
runtime is SHA-256
`2188a8bb2fd0e126a0da02f6d5af1c809ff0e4e1682021d37869e22dc01d0d05`.
Its executable, source snapshot and CUDA libraries are retained outside build
directories under `~/.local/share/llmp/baselines/m35-q8-schedule-r5`.

The generic GB10 dense Q8_0 scalar schedule now uses the incumbent MMVQ's
small-K and four/eight-warp iteration rule. Quantization, integer dot products,
product eligibility and evidence authority are unchanged. The scope includes
scalar dense/grouped and joined rows marked `SetVecQOneToken`; ordinary multirow,
routed, other encodings and other platforms are unchanged. There is no family,
checkpoint, dimension whitelist or approval gate. Changed consumers do not
inherit unchanged-arithmetic qualification.

Four focused GPU tests pass with no skips. The expanded control covers 17 K
widths, including Gemma K2304/9216, retained Qwen GGUF PLE/mixer K2560/320/640/10240,
and small-K and adjacent scheduling decisions. Each uses 64 output rows and
three distinct input rows; complete joined/solo Q8_1 bytes match, dense results match prepared and
ordinary incumbent MMVQ bitwise, and fused GEGLU matches actual incumbent fused
MMVQ bitwise. These retained Qwen Q8 schedules are unchanged; their actual head
and attention-output roles are Q6_K and outside this Q8-only change. Width
coverage does not establish another encoding's qualification. Existing
grouped/multirow/routed controls remain unchanged and pass. Output-row tails
were not covered by this N64 matrix; include the existing
`EveryConfigurationCoversEveryTypeTailRowsAndPaddedRows` control in final
adoption checks.

This is operand evidence, not model-quality evidence. The original unexplained
CTest abort remains retained alongside unchanged-binary GDB/memcheck passes;
fixture expected-error reporting now propagates completion failures. The later
compile/discovery harness failures and their precise repairs are also retained.

## One fixed collection, then one pairing

All parent input, request, parser, quality and retirement requirements remain.
Use the same artifact/config, native tokenizer metadata, 4000-byte passage,
F16 KV, context 8192, two slots, prefill_chunk1, sequential C1, and the same two
T0, max_tokens128 greedy prompts. Authenticate the parent helper/config/corpus
and artifact pins plus R's source/runtime/dependencies before enrollment and
after retirement. Preserve both raw `ldd` transcripts; dependency equality uses
resolved paths/hashes, ignoring only hexadecimal load addresses. Retain exact
argv, environment, full raw score response/SSE and successful-use plans before
validation. No inference retries, fallback, additional prompts or changed bounds.

### Pre-launch administrative corrections (2026-10-22)

The first enrollment, `m35-u3-q8-scalar-repair-r1`, stopped in preflight because
the registered A `audit.json` and `plans.json` existed only on the orchestrator.
Their exact pinned bytes were copied and authenticated. The second enrollment,
`m35-u3-q8-scalar-repair-r2`, rejected `--label q8-repair` during CLI parsing.
Neither launched the runtime service or collected a sample. Preserve both jobs.

Before the first runtime service launch, validation-only failures may be fixed
and enrollment repeated after authenticating every unchanged pin. Validate the
complete collector argv with its quality handler replaced by a no-service
capture. Use the existing supported `--label new`; the explicit pinned config
remains authoritative, and R remains this arm's identity. This label changes
only output names and report labels. Any runtime service launch consumes the
one collection attempt; no subsequent inference retry is authorized. Inputs,
bounds, executable, dependencies and retirement requirements remain unchanged.

Run once on Spark B in a fresh private output/data directory under installed
exclusive `spark-job --gpu --timeout 600 --grace 30`. Use the parent's exact
`capture.load_speed_gate` command with its committed private `REG` snapshot,
`--label new`, and runtime:

```text
/home/pmeenan/.local/share/llmp/baselines/m35-q8-schedule-r5/src/runtime/llmp-runtime
```

Keep readiness 420 seconds, TERM retirement 20 seconds, `--debug-plans`,
`--retain-streams`, the existing config/corpus and port 18114. Do not synchronize
this registration into the authored runtime source tree or rebuild to collect.
Stop on infrastructure/identity/response failure and retain raw evidence.

Require complete identical 1071 echoed IDs against retained A, exactly one
leading BOS 2, leading null and 1070 finite nonpositive teacher scores with exact
position/ID keys, and zero generated scoring tokens. The frozen ID-list hash is
`ee85f4ffc9be5825936ff379d873322a78c048018375805ffbb4a5cb5b1946c4`
(`json.dumps(list)` default spacing, UTF-8). Require successfully completed
one-row/one-owner target score plans, inspect actual Q8 dense/fused selections,
and retain attention/Rope and operand-domain comparisons; count alone is not
proof of the executed recipe. A wide score plan invalidates this scalar screen.

After valid collection, run the unchanged `quality_pair.py pair` once with
R's `quality/gemma2-quality-new.json`, retained
`/tmp/llmp-u3-gemma2-quality-78747648/arm-a/quality/gemma2-quality-old.json`, and
`docs/experiments/m36-final/prereg/pair-gemma2.json`. The unchanged bounds are
absolute PPL delta≤0.5%, mean absolute NLL difference≤0.05 nats, maximum≤1.0,
and at least95% of 1070 targets within 0.1 nats. Preserve every result.

Review both retained greedy texts against the same tasks and disclose anomalies;
differing greedy tokens alone do not fail the numerical rule. Report original A
and new R counts/clocks plus A/R decode-rate and elapsed-time ratios in prompt
order. A is an earlier authenticated sample, not a contemporaneous timing pair;
these single debug-logged observations establish no speed gate. A numerical PASS
only supports choosing the next affected-consumer control. A FAIL preserves the
bounds and returns to diagnosis, with no broader qualification or golden update.
