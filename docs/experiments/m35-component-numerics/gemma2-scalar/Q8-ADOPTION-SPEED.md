<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prospective Q8 schedule adoption speed screen

**Prospective: not enrolled.** Actual same-host baseline transfer and candidate
pins are frozen in [q8-adoption-speed-pins.json](q8-adoption-speed-pins.json).
Review and commit before timing. This is
one representative Gemma2 D-085 regression screen; it does not qualify other
consumers or replace numerical/state controls. The original A/D quality FAIL
and later exact R/A PASS remain separate records.

Run both arms on **Spark B**. Copy the preserved W13 A baseline runtime and
its actual runnable CUDA libraries byte-identically into B's private baseline
directory; authenticate the copied files and actual B dependency resolution.
No cross-host timing comparison is permitted. Preserve its complete source,
module and receipts. The candidate is the checked W15 main build with the Q8 repair and 18-width
exact operand control: runtime c940d8c2…, source manifest84cd7472… (2,889 files),
module88f44c7c…. Its actual runtime and immutable resolved libraries are retained
under `baselines/m35-w15-q8-adoption`; the source/receipt pins, not a Git label,
identify this build. Never change either build during collection.

Use the pinned existing `tools/speed_gate.py run`, without debug plans,
`capture.py`, extra priming or runtime environment overrides. Materialize the
exact JSON `config_template` into a private file, verify its hash and pass it
through `--config-template` for both labels. The same Gemma2 artifact, F16 KV,
context8192, chunk128 and two slots apply; `old`/`new` select report names only.
The corpus and helper hashes are fixed in the pins.

Keep the existing family workload: `--family gemma2 --cells 1,2`,
26,000-character corpus prompts, T0, 128 generated tokens, one warmup pass and
six measured passes per fresh service. C1 and joined C2 exercise scalar
schedules. Give every arm a fresh private output/data namespace. Run exactly
three pairs, in baseline/candidate, candidate/baseline, baseline/candidate
order, with `--block 1|2|3 --position 0|1`. Each arm is one installed exclusive
`spark-job --gpu --timeout 600 --grace 30` job, waited to retirement before the
next. Authenticate runtime/dependencies/helper/config/artifact/corpus before
and after each arm and require empty runtime overrides, complete work and
successful retirement. Preserve raw `ldd`; compare paths/hashes, excluding
only hexadecimal ASLR addresses from transcript equality.

Pair actual prompt indices, widths, prompt/generated-token usage and cache
counts for every warmup and measured request. Require 128 generated tokens,
zero cached prompt tokens, no helper problem/error, and every requested metric.
A completed arm with an infrastructure failure, recorded helper problem or
invalid response stops collection before the next arm or pair. The unchanged
helper may finish that arm's prescribed passes before reporting a problem;
exit status zero alone is insufficient. Retain the failure; do not replace a
measured sample by retrying.

Recompute each run's C1/C2 metrics using the pinned helper: `prefill_s` is the
client interval from cohort release to the last first token, with decode rate
and wall throughput reported separately. Each run metric is the helper's mean
over its six measured passes; the final aggregation is the median of three
paired ratios for each metric. Slower candidate is candidate/baseline for times and
baseline/candidate for rates. **Stop after pair1 if any ratio exceeds1.10**;
retain and diagnose rather than collecting the remaining pairs. Otherwise,
collect pairs2/3: each of six metrics must have median paired ratio≤1.10.
This reuses the existing U4 D-085 10% bound and three-pair aggregation, not the
1% statistical protocol. Report all absolute values, all pair ratios, token
counts, peak occupancy and successful retirement; do not average metrics or
select a better pair.

The measured historical Gemma2 arm duration is57s in
[the baseline table](../../m36-baselines/README.md#same-binary-spread), with
this same C1/C2/six-pass workload. Thus estimate114s for pair1 and342s for all
six arms, plus changed startup/retirement overhead. These are planning estimates,
not current measurements. The result is needed now to decide kernel adoption.
No model, timing or speed-gate result is claimed by this registration.
