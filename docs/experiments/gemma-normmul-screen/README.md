<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Plain RMSNorm/Mul first screen

Enabling the existing checked two-node RMSNorm/Mul selector did not establish
an end-to-end gain for Gemma31. It selected 121 additional implementations in
both the final prefill plan and decode plan; the existing 120 norm/RoPE and
120 norm/residual selections remained intact. Counts describe selected plan
implementations, not CUDA launches. The option remains unselected in production.

The benchmark accepts an optional eighth argument, `normmul-off|normmul-on`,
after the row cap. Omitting it preserves every existing recipe with the flag
off. Only `Gemma4Options.fuse_norms` changes; checked norm chains, ordinary
products, weights, cache/mask policies, graph behavior and funded output/state
publication are unchanged.

| Arm on Spark A | 8K prefill (s) | 32 completed decode units (s) |
| --- | ---: | ---: |
| Original before | 10.8794 | 3.11261 |
| Native off | 12.3456 | 3.17728 |
| Native on first | 12.5842 | 3.17911 |
| Native on repeat | 12.3092 | 3.18922 |
| Original after | 10.8991 | 3.11560 |

The candidate mean was 0.8189% slower in prefill and 0.2167% slower in decode
than the one fresh off control, and 14.3026%/2.2498% slower than the reference
bookend means. Candidate prefill varied by 2.2% between its two runs; this short
screen does not resolve a small effect. No wider shape ladder or Gemma26
transfer run was performed.

All ten retained complete full-vocabulary heads are finite. Candidate first
and repeat match both complete heads, all 1,740,636,160 initialized native
state bytes and all 32 greedy choices exactly. They also match the off control
and retained native baseline exactly. Both original bookends match their prior
ring heads and each other; all 32 cross-engine choices agree. The final native
head is byte-exact with reference; prefill retains the known maximum raw-logit
delta 0.41361475. This checks two retained vectors, not all 32 decode vectors,
corpus likelihood or whole-model quality.

The unchanged [ring recipe](../gemma-swa-ring-h1/README.md) uses Gemma31
UD-Q4_K_XL, context 16,384, F16 KV, `swa_full=false`, `kv_unified=false`, local
1,280/global 16,384 cells and row cap/ubatch 256. Both engines discard six warm
rows, clear, pay 8,192 prefill rows, append the same three untimed anchors and
complete 32 forced decode rows. Native pays 31 additional intermediate heads;
both pay full-vocabulary publication and CPU argmax. Head files, state copies,
finite scans and hashes occur outside both timers. Native reports one captured
graph and 33 replays per arm; reference graphs are allowed with the original
fusion policy. Reference GGML graph reuse is not a CUDA replay count.

Reproduce the native arm with `llmp_gemma_prefill ARTIFACT IDS NEW_OUTPUT_DIR
31 both 256 normmul-off` or `normmul-on`, using the artifact and 8,227-ID identity
in [results](results.json). Use the existing untraced original ring client and
owned-container wrapper from the planner-index comparison with unique output
names. The wrapper changes only its accepted name prefix from task45 to task46;
its pinned executable, input checks and checked container retirement remain
intact. Both contexts use the same physical Spark A. Raw logs, vectors, states,
commands and official completion records remain external.

The canonical warm Spark A source sync, narrow benchmark build, nine CLI
refusal controls and all five model invocations completed successfully under
the installed GPU supervisor. The post-run checks completed successfully too;
both original containers were confirmed absent. This benchmark-only experiment
made no kernel, runtime, source-lock or production-default change and ran no
routine full unit suite. TensorFold was refreshed at task entry on
2026-10-05 19:12:43 UTC: HEAD 609ca419…/0.6.5 still documents Gemma26 on MLX,
with no applicable Gemma31 CUDA recipe. The same-format pinned llama client
therefore remains the comparison.
