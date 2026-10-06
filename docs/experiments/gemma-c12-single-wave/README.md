<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ordinary whole-C12 Gemma factor

Ordinary Gemma waves can share twelve one-query product columns and use three
four-root attention quads. D-092 row-invariant waves keep their eight-row limit.
Both checkpoint factors pass their original frozen margin and 384-target
conditional-loss gates. The [original 8+4 failures](../gemma-c12-joined-screen/README.md)
remain failures; no recalibration or new reference outputs were used.
Serving owner attention and joining remain off by default.

| Fixed 396-head control | 31B | 26B |
| --- | ---: | ---: |
| Unchanged pre-oracle p99 margin | 2.2024709702 | 7.5425544739 |
| Strict positive-margin choice differences | 0 | 3 |
| Differences outside the original bound | 0 | 0 |
| Complete stock-byte-exact paid heads /384 | 384 | 0 |
| Complete stock-byte-exact prefill frontier heads /12 | 0 | 0 |
| Maximum raw delta across all 396 heads | 0.78764248 | 2.75038028 |
| Relative conditional loss,384 targets | −0.091920% | +0.083892% |

The 26B differences are rows 109/349/395, with reference winner margins
0.02856523/0.10283422/0.00739002, all inside the unchanged bound. Strict
zero-difference still fails for 26B. Row 395 is the final owner 11 head and has
no supplied-token target. The conditional score uses all vocabulary values in
FP64 over rows 0..31 per owner, predicting supplied positions 992..1023;
row 32 predicts 1024 and remains unscored. The 3% conditional-loss limit is
independent of the margin gate; this is not a 1,023-transition corpus PPL claim.

Both candidate pairs retain byte-exact finite complete heads and all 24
frontier/final initialized-state snapshots and layouts. All twelve frontier
state/layout identities match the retained 8+4 candidate. This does not establish
final-state equality with stock. Full result/proof/source identities and exact
three 26B differences are in [results.json](results.json); raw vectors/logs stay external.

| Short paid 32-step elapsed,seconds | 31B | 26B |
| --- | ---: | ---: |
| Prior 8+4 candidate | 8.27676 /8.27600 | 2.75774 /2.76887 |
| Whole 12 candidate | 5.73670 /5.68788 | 1.93163 /2.04589 |
| Before/after mean change | −30.9808% | −28.0297% |
| Retained first stock 12 pair | 6.08929 /5.54673 | 1.99187 /1.97779 |

These are short screens on B, not sustained parity measurements. Stock 31 varies
9.33% of its mean and the new 26B pair varies 5.75%. The source vintages also
include the landed live-state placement memo and backing-parent validation;
the timing difference is not isolated to one code change. The new graphs use
one twelve-column product group per step, 32 groups/replays instead of 64, plus
whole-twelve stream-K geometry split into three real four-root calls. Shader,
fixup and Queue bodies are unchanged. A non-divisible whole grid falls back to
four-owner geometry; unsupported quads remain independent without reordering
owners or changing the product grouping.

Twenty-one focused controls pass with zero skips: 11 graph, 3 planning, 5 grouping
and 2 operator tests. They cover third-quad writer refusal before mutation,
quad/tail and unequal-width fallback, invariant8 limits, and real SizedArena
funding through both placement passes. Original full-twelve MMA matches the
three independently allocated four-root calls byte-for-byte for D256/D512,
heads 16/32, cells 256/1024, eager and poisoned capture/replay. On the actual
48-SM GB10 those operator plans use whole/quad grids 48/16 and96/32, effective
cohort 12. This is a synthetic descriptor proof, not an observed stock-model
Q descriptor. The initial build's obsolete negative 12 fixture failed; changing
that invalid-case token to 16 preserves every production byte. The retry passes.

The measured helper is `a130902c…`, SDK receipt `eb7a4ebc…`, based on
`3de8b67` plus the fourteen recorded source paths. Native NVCC 13.4.92/toolkit 13.4.2,
B GB10/driver 580.178.04, and pinned reference image 837fc732 are unchanged.
Each checkpoint uses its independently qualified twelve-history carrier
`d5844500…`, context 4096, rows 256/1024, headcap 12, plain norm ON and existing
norm/all policies. Prefix 992 precedes 32 supplied forced positions, with no
anchors. The helper retains the unchanged 440,516,608-byte publication
allowance. Actual first-decode metadata reports owner/request12 steps 180/90,
request 8 zero, 32 paid groups and 32 replays. These are selected-plan counts,
not per-replay launch records. Reuse the input/artifact/client pins and normal
physical12 F16 reference recipe in the [original screen](../gemma-c12-joined-screen/README.md):
`jitllm_gemma_joined ARTIFACT OUT PROFILE 12 joined norm|all CARRIER production`.
Two own native arms precede a separately supervised complete own-repeat proof
and retained-FIRST-stock assessment using the unchanged analyzer/judge numeric
functions. All six acquisition/proof/comparison jobs retire DONE0; all eight
owned analysis containers are absent after checked retirement. Full regression
is deferred by owner instruction. Latest task-entry TensorFold remains primary
`609ca419`/0.6.5 with a Gemma26 MLX table, without a comparable GB10 CUDA recipe.
No production default, corpus, depth, swap or broader model-support gate is closed.
