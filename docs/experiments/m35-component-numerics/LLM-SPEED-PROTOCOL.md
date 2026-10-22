<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prospective U4 LLM speed screen

This recipe fixes the affected A/D speed screen before timing. It uses the
frozen arms and current artifact/configuration/source pins in
[llm-capture-pins.json](llm-capture-pins.json), registered by `81af1dc8`.
Native Qwen has new MTP arithmetic; Qwen GGUF has transferred five-row products.
Time both. DeepSeek needs this screen only if its completed-plan comparison
finds changed arithmetic. Quality, completion and affected model controls remain
separate requirements. No timing result is claimed here.

Use the existing pinned `tools/speed_gate.py`, SHA-256
`0f1135a2283fe63797e15c8dffe48c9df39b0baae91a57f925b5ba21c3e3252e`.
Its `run` command supplies the production service, concurrent clients, monotonic
timing and memory sampler. This is D-085's coarse 10% screen, not a rerun of
M3.6's sequential 1% migration protocol or its statistical stopping rule.

## Work and pairing

Run on Spark A, the original capture host, with no other job or GPU process.
Authenticate the frozen runtime and all ten resolved dependency hashes before
each arm; check the runtime hash again after retirement. Disable debug plans
and export. Each arm is a fresh process with a fresh private data directory,
and each attempt has a new external output namespace. A failed attempt is kept.

Supply the exact family's `configs` string from the capture pins through
`--config-template`; both arms receive that same file. Explicitly pass the
original corpus path and `--cells 1,2,4`. Never use the helper's historical
old/new default configuration. `--label old` names A, `--label new` names D;
these labels select record names only because configuration is overridden.
Pass no `--runtime-env` override and require the recorded `runtime_env == {}`;
the production arms use their default arithmetic. Do not pass `--prime`.

Keep the pinned helper's workload unchanged: 30,000-character corpus prompts,
128 output tokens, greedy streaming, and C1/C2/C4 concurrent requests. Every
process performs one warm-up pass of all cells. Native Qwen then performs two
measured passes; GGUF and DeepSeek perform one. Prompt indices advance exactly
as in the helper, restarting at zero for each fresh process. Paired arms must
pay the same prompt indices, widths, prompt-token counts and generated-token
counts for every warm-up and measured cell; no cached prompt tokens are allowed.
Reject any helper error/problem, incomplete request, missing metric, changed
input/configuration or failed retirement before interpreting speed.

| Pair | Process order | `--block` | `--position` |
| --- | --- | ---: | --- |
| 1 | A, D | 1 | 0, 1 |
| 2 | D, A | 2 | 0, 1 |
| 3 | A, D | 3 | 0, 1 |

Each arm runs in a separate installed `spark-job --gpu --timeout 600` job and
is waited to completion before the next arm. Preserve all six records per
family. Each arm's first pass supplies its own warm-up; there is no unequal
extra priming run. Do not use the helper's `plan` or `verdict` commands to
silently substitute their different 1% protocol. Stop and diagnose a failed
control or a clear first-pair regression before expanding that family.

## Metrics and acceptance

Recompute each run's C1/C2/C4 prefill seconds, decode tokens/second and wall
tokens/second from its actual recorded requests using the pinned helper. Keep
every pair's ratio. Express slower D as a ratio above one: D/A for prefill
seconds, A/D for decode and wall throughput. Each of the nine metrics must have
a three-pair median ratio at most 1.10. No averaging across metrics, selecting
the better pair, or post-result threshold change is permitted. Report any
individual regression too. This adopts the existing 10% bound and freezes the
pairing and aggregation before timing; it claims no finer precision.

Report absolute values and all ratios, complete-work/token checks, runtime and
source identities, actual host/driver/toolkit, load/readiness and retirement
separately. Report each run's whole-node peak MemAvailable drop sampled every
20 ms as an occupancy estimate, not exact GPU allocation. Keep actual memory
admission active. Raw run records and logs remain outside Git; the aggregate
table and its conditions belong with the U4 results.
