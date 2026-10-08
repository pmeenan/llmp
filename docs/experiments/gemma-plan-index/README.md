<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Call-local graph reader indexing

Graph planning now indexes logical view roots, producers, source edges and keeps
once per invocation. Reader queries inspect affected roots instead of rescanning
the entire graph. Duplicate exact edges and producer skips remain valid;
coincident activation addresses do not identify a dependency. MoE retains its
strict per-descriptor view/span validation, depth 64 bound and exact selected-ID
exception. Standalone matchers keep their allocation-free scans. Primitive
policies without reader queries allocate no index.

Both planning passes construct fresh indices, including validation after
activation placement, and still require SamePlan. Nothing persists between
passes; kernel arithmetic and model policies are unchanged. Startup measurement
sets aside the maximum conservative index scratch allowance once alongside
SizedArena. Runtime growth is refused before rebinding or allocation. For the
26B screen this combined allowance is 9,409,024 B, not a measured physical peak.

The unchanged short 8K recipe uses context 16,384, F16 KV, explicit ring SWA
(`swa_full=false`, `kv_unified=false`), local 2,048/1,280 cells (26/31), global 16,384,
six discarded warm rows, reset, three untimed anchors and 32 forced decode units.
Native uses chunks 1,024/256 with 60/90/30/30 norm-RoPE/norm-add/route/reduce at 26,
and 120/120/0/0 at 31; all other optional math flags stay off. The
[26 recipe](../gemma26-swa-ring-transfer/README.md) and
[31 recipe](../gemma-swa-ring-h1/README.md) retain exact inputs and commands.
The screen retains two complete heads, initialized native
state and 32 incoming-head choices. Both candidate repeats match retained native
controls exactly for both profiles. Reference bookends repeat their own heads
and choices; all 32 strict choices also agree across engines. Full quality remains
open: both retained 26 heads differ from reference, while 31
prefill differs and its final head is byte-exact.

| Profile / host | Native prefill (s), first/repeat | Reference bookends (s) | Prefill gap | Decode gap |
| --- | --- | --- | --- | --- |
| 26 / all /1024 / Spark-b | 2.77167 / 2.77446 | 2.42169 / 2.39683 | +15.1003% | +0.6096% |
| 31 / both /256 / Spark | 12.4190 / 12.3674 | 10.8711 / 10.9044 | +13.8270% | +1.9957% |

Gemma26 prefill takes 27.3% less time than the retained native baseline
(3.82166 / 3.80773 s); this is a historical comparison, not a fresh old/new A/B.
No same-host old/new gain is claimed for 31. Native still publishes eight/32
prefill heads versus one in the reference. Those costs and all existing warm,
reset, fixed-token, mask, ring-cache and publication policies remain unchanged.
No new quality, optimized-batching or competitive-performance gate is claimed.

The bookends use native binary ba02e166… before the final scratch-accounting and
primitive-bypass changes. The final funded 26 control again matches both heads,
all initialized state and all 32 choices exactly (2.77254 s prefill). The helper's
only final change accounts for planning scratch in its budget and host floor.
79 focused matcher, norm, generic graph, placed-plan, plan-cache, DeepSeek and
Qwen wave controls pass. Raw logs, vectors, token telemetry and job receipts
remain external; [results](results.json) and [provenance](provenance.json) retain
aggregates and identities. The final Spark-b suite passed 1,698/1,698 tests without
skips (598.87 s), including 311 GPU and 57 model tests.
