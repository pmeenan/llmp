<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 scalar repair: prospective C1 adoption speed check

Review and commit before timing. This completes the current scalar schedule
spike after the registered single-passage numerical transfer passed exactly;
it opens no new quality domain. Compare the quality-matched incumbent MMVQ
policy A with repaired normal VecQ policy D, using the same retained runtime
`5965e83c…`, current artifact `09080998…`, F16 KV, input and host. This is
competitiveness against the incumbent, not an isolated before/after measurement
of the old inaccurate VecQ schedule. No frozen expectation changes.

speed-pins.json binds the actual runnable binary, source2903,
compiled identities, retained/resolved libraries, artifact metadata and payload
stat identities, input, method, executors and prior quality/manual-review
receipts. It reuses the authenticated quality preflight directly. Run only on
Spark A. Do not synchronize or rebuild its source for these documents.
[speed-node.toml](speed-node.toml) sets context8192, chunk256, two slots and no
drafter. Both arms have this identical config; only the already exercised paid
A/D target policy differs. Automatic startup accounting and the112GiB diagnostic
node ceiling remain unchanged. No numerical grant, rank override or debug plans.

The changed selector is GB10 dense/no-ID/token1 (including existing one-token
callers). Normal owner-row-invariance remains disabled. Ordinary C4 multirow
schedules are unchanged; this bounded adoption check therefore uses **C1 only**.
It does not qualify an assistant, a new batching domain or frozen-state parity.

Use the existing pinned `speed_gate.py run --family gemma31 --cells 1`, directly,
without capture.py, debug-plan logging, extra priming or new collector. Preserve
its26,000-character corpus prompts, temperature0 and128 generated tokens, one
warmup pass and one measured pass per fresh service. Each arm starts prompt
indices0 and1, respectively; the ordinary helper loads the registered complete
corpus. Neither prompt depends on a measured outcome. The exact complete argv
is produced by the pinned external `/tmp/llmp-u4-gemma31-speed/run.py`.

Collect exactly three matched pairs, **A/D, D/A, A/D**. Each of six arms gets a
fresh private output directory, its own installed exclusive600/grace30 job,
and successful retirement before the next. The command within that job is:

```sh
~/.local/bin/mise exec -- python3 -B /tmp/llmp-u4-gemma31-speed/run.py PAIR ARM
# PAIR=1,2,3; ARM=a,d in the order fixed above.
```

The executor binds the reviewed full registration commit and files before
admission; the preparation-only receipt cannot collect. It verifies runtime,
actual resolved dependency bytes, source/compiled identity, config, input,
artifact and helper pins before and after each arm, retaining raw ldd and
normalizing only ASLR addresses for transcript equality. Existing payload full
hashes are publication provenance; this check revalidates metadata and payload
stat identities without rehashing all weights. Verify idle GPU/no other jobs
before enrolling; installed exclusive locking remains the admission authority.

Before timing, exercise all six complete argv through the **real** run parser
with collection replaced by a capture stub, prohibit service/process/network
launch in the stub, render/parse each config and exercise the complete actual
read-only preflight. A/D policy spelling is unchanged from the completed quality
services. No service is allowed in this validation. Retain its receipt and hashes.

An arm is valid only with no helper error/problem, positive complete warmup and
measured C1 requests,128 tokens/chunks per request, normal `length` finish,
positive prompt counts, zero cached prompt tokens, finite ordered times, all
three metrics, actual memory occupancy, and successful TERM/returncode0
retirement. Check raw records even when the helper returns0. Preserve actual
prompt indices, prompt/generated/total/cache usage, full raw records and logs.
Across both arms and all pairs, require identical workload identities; do not
substitute count agreement for a tokenizer change (artifact/tokenizer is pinned).
No invalid completed arm may be followed by the next arm. Preserve failures;
no measured sample retries, replacements, retuning or extra passes.

Recompute metrics with the unchanged helper. Its `C1.prefill_s` is client TTFT
(cohort release to first token), **not isolated prefill time**. Decode and wall
rates remain separate. For each pair, slower-candidate ratios are D/A for TTFT
and A/D for rates. **Stop after pair1 if any exceeds1.10.** Otherwise complete
the two remaining pairs; require the median of the three paired ratios≤1.10
for **each** of TTFT, decode and wall throughput. This is the existing coarse
D-08510% rule, not the1% M3.6 statistical gate. Report every absolute value and
pair ratio, all three medians, actual occupancy and lifecycle; no favorable
pair selection or combined metric. Occupancy is informational, but every
complete workload must fit. No timing result is claimed by this registration.

Planning estimate:60–90s/arm,2–3min for pair1,6–9min for all six. The historical
[Gemma31 baseline table](../../m36-baselines/README.md#same-binary-spread)
reports178s for C1/C4 with one warmup and one measured pass, C1 wall5.854tok/s
and C4 wall8.535tok/s. Removing the two C4 cohorts estimates approximately
`178 − 2×512/8.535 = 58s` before changed startup/workload overhead. This is a
planning estimate, not a new measurement. The result is needed now to decide
adoption of the already reviewed arithmetic repair; no other measurement is
scheduled here.
