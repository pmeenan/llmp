<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Checked plain RMSNorm/Mul serving default

Plain RMSNorm/Mul is now enabled by default for both approved Gemma profiles.
The existing checked selector preserves complete retained heads, initialized
state and 32 choices in ordinary off/on controls, while other experimental
policies remain off. The research-policy off/on/on/off diagnostics below also
show a modest prefill benefit. The 31B research screen selects 121 additional plain
RMSNorm/Mul implementations in prefill and decode, while the existing 120
norm/RoPE and 120 norm/residual selections remain intact. These counts describe
plan selections, not directly measured CUDA launches. Only `Gemma4Options::fuse_norms` changes its default; all other arithmetic
flags remain explicit opt-ins.

| Normmul mode | 8K prefill, s | 32 completed decode units, s |
| --- | ---: | ---: |
| Off first | 12.0398 | 3.18928 |
| On first | 11.3124 | 3.19155 |
| On repeat | 11.3283 | 3.19636 |
| Off repeat | 11.6728 | 3.19015 |

Means are 11.8563 s off and 11.32035 s on: a 4.52% prefill reduction. Decode
changes by +0.13%. Off varies by 367 ms (3.14%), so the effect's magnitude is
uncertain; even the slower on arm is 2.95% faster than the faster off arm. The
preceding Task52 norm-off/lookahead-on mean was 11.38815 s, only 0.60% above
this diagnostic's on mean. All samples remain recorded. There is no fresh
reference bookend or parity claim.

Both complete vocabulary heads, all 1,740,636,160 initialized state bytes and
32 greedy choices are exact across all four arms and the retained native
baseline. All eight retained heads are finite. Each arm builds/caches 31
predicted plans with no refusal and retains one captured graph/33 replays.
This checks two full vectors and 32 choices, not corpus quality or the open
optimized C4 gate.

The same-binary Gemma26 all1024 transfer on Spark B (`spark-56f5`) also shows
a modest benefit: off 2.55883 / 2.57430 s versus on 2.53058 / 2.51346 s gives
means 2.566565 / 2.522020 s, a 44.545 ms (1.74%) prefill reduction. Decode changes
by +0.014%. Both on arms select 121 plain norms; 60 norm/RoPE, 90 norm/residual
and 30 route/reduce selections remain unchanged. All four arms retain exact
complete heads, 592,445,440 initialized state bytes and 32 choices; all eight
heads are finite. Each builds/caches all seven forecasts with no refusal.
The 26B recipe uses local/global 2,048/16,384 cells. The 31B diagnostic above
runs on Spark A (`spark-c4e2`). These results support adopting the checked
selector in both approved profiles, without an ordinary-policy speed claim.

Ordinary-policy controls use the same pre-adoption a5ea helper with explicit
normmul-off/on and all other arithmetic experiments disabled. Gemma31 ordinary256
prefill is 11.8435 / 11.5605 s; Gemma26 ordinary1024 is 3.27601 / 3.15367 s.
These are single pairs, not sustained speed or reference parity measurements.
Both complete heads, every initialized state byte and all 32 choices are exact
to the corresponding ordinary-off baseline; all eight retained heads are finite.
Plain norms select 361/271 implementations for 31B/26B. The established ordinary
reference-quality differences and optimized C4 gates remain open; this change
does not inherit research-policy quality or speed claims into ordinary serving.

The production adoption uses source base 756ab6e and changes only the options
header, plus focused fixture coverage. Historical fixtures explicitly retain
norms off. New default fixtures omit the override and reuse scalar/unequal-wave
state-only, complete continuation and captured replay controls; a flag guard
checks that other experimental defaults stay off.

The research and ordinary comparisons use the unchanged pre-adoption helper from [Task52](../gemma-prefill-lookahead/README.md):
a5ea8065 helper, receipt 995ee819, base 1ceb7d2 plus the reviewed production
lookahead. Only the existing optional `normmul-off|normmul-on` argument changes.
Each 31B research arm uses state-only intermediate chunks and lookahead-on, the same
Gemma31 artifact, 8,227 fixed IDs, six discarded warm rows, Clear, paid 8,192
rows at both256, three untimed anchors and 32 forced units. Context is 16,384,
F16 KV, local/global cells 1,280/16,384, SWA/unified false. Finite scans,
snapshots and hashes stay outside paid timers.

Reproduce with `llmp_gemma_prefill ARTIFACT IDS NEW_OUT 31 both 256
normmul-off|normmul-on state-only lookahead-on`; for the 26B transfer use
`26 all 1024` with the same trailing modes. The task-entry primary
TensorFold git query at 2026-10-05 23:01:41 UTC remained
609ca419abecebdc5a059498a613680bd3aa847f, without a matching Gemma31 CUDA recipe. The adoption-entry check at
2026-10-05 23:10:38 UTC returns the same HEAD.
[Aggregate results](results.json) preserve counters, hashes and timing means.
All eight official supervised arms and both existing exact checkers pass; raw
logs, full heads and state snapshots remain external. No trace, full suite or new benchmark was added for these screens. The
subsequent default adoption changes no kernel arithmetic or source lock.

The canonical adoption build and 13 focused controls pass: default flags,
four ordinary/default actual-model state/continuation/replay fixtures, two
norm registry/kernel controls, three norm descriptor bounds controls and three
selected Gemma placement/binding controls. The first plan-control driver omitted
its test-data environment and aborted before finishing; the narrow retry with
that environment passes without a source change. Independent source review is
clean. The full suite remains deferred by the owner until selected performance
work settles.
