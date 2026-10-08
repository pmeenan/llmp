<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma4 bounded owner roots

Both approved Gemma4 profiles preserve the stock common attention geometry
without copying short KV prefixes into temporary padded caches. On the same
binary and inputs, bounded roots reduce paid latency by 6.01% for Gemma26 and
5.24% for dense31. Both short screens retain all 68 stock heads and both
initialized states byte for byte; current stock latency differs by less than
0.1%. These are two-observation runner screens, rather than sustained or
end-to-end serving parity claims.

The preceding [common-read repair](../gemma4-common-width/README.md) established
why unequal owners need one logical read width. Its physical zero padding kept
that arithmetic but introduced per-layer KV copies. The bounded implementation
retains the original logical stream-K grid, mask prepass, partition/fixup
arithmetic and common mask width, while stopping each owner's physical reads
at its own initialized, aligned prefix. Empty physical partitions publish the
original neutral partial result without preloading an absent first K tile.
Actual SET_ROWS-derived views remain independently bounded; no larger cache
view or cursor is fabricated. K and V remain distinct, including tied-projection
layers. Equal-width waves retain the previous node, operands and kernel.

Admission is two actual owners, logical cohort two, offset zero and one query,
with no attention softcap: H16/H32, D256/GQA2 or D512/GQA8. The inherited Gemma2
H8/cap50 and Gemma3 H8/no-cap paths are preserved separately. The exact bounded specialization's
resident capacity is queried, but its grid and scratch remain those of the
original reference specialization. Internal options default false. Ordinary
serving selects this recipe at context<=4096 with exactly two configured slots
after both-profile HTTP controls; wider/partial cohorts and compatible multirow
Gemma4 prefill remain separate qualification work.

## Same-binary causal screen

Each owner uses a different authenticated real history, independent prefixes
256/768, three supplied scalar warm rows and 32 joined teacher transitions.
Context is 4096 per owner with F16 KV, two slots and two funded head rows.
Gemma26 uses ordinary 1024-row prefill chunks and dense31 uses 256. Both retain
full final FFN and their checked norm/RoPE/residual and routed/dense recipes.
The sole padded/bounded arithmetic-path choice is the physical-root option.

One current padded own run and two bounded own runs agree in all eight exported
head/choice/history/state payloads. GPU choices equal full-head argmax, both
initialized states survive Clear/backing reuse, spill/restore and checkpoint
advance/restore, and malformed work leaves state unchanged. Both profiles match
all two initial and 66 later stock heads; all 64 scored transitions have zero
strict/positive-margin differences, target-loss delta, total variation and raw
logit difference. Actual bound-plan selection is 30 / 60 bounded owner nodes,
rather than an observed kernel-launch count.

The six paid arms are stock/padded/bounded/bounded/padded/stock. All six retain
identical 64 generated IDs, histories and both final full verification heads.
Each pays independent prefill and 32 generated steps, including final heads;
three supplied warm rows are outside the paid interval. Stock uses the unchanged
official backend greedy sampler, including full sampled-logit host transfers.

| Profile/policy | Prefill (ms) | Decode (ms) | Paid total (ms) |
| --- | ---: | ---: | ---: |
| 26 stock | 336.7400 | 736.7390 | 1073.4790 |
| 26 padded | 348.6455 | 792.3870 | 1141.0325 |
| 26 bounded | 346.9855 | 725.4285 | 1072.4140 |
| 31 stock | 1280.4950 | 3267.1100 | 4547.6050 |
| 31 padded | 1306.9500 | 3490.1150 | 4797.0650 |
| 31 bounded | 1306.4350 | 3239.2850 | 4545.7200 |

Each mean has two paid observations per policy. Bounded total latency changes
by −6.013720% / −5.239558% versus padded and −0.099210% / −0.041450% versus stock
(26B / 31B). The same-binary decode reductions are 66.9585 / 250.8300 ms over
32 steps. This isolates the cost of temporary KV padding on the tested shape;
it does not establish broader context, cohort or sustained performance.

## Operator and adoption controls

The 47 focused host cases pass. The new operator control covers sixteen
configurations: H16/H32, both head dimensions, 512/1024 and 256/1536 physical
prefixes, and both short-owner directions. Initial and independently refreshed
owners match the original physically padded oracle byte for byte across eager,
capture and replay. FP64 normalized error is at most 1.19208e−6. Poisoned absent
mask/scratch/output data and NaN guard tails retain meaningful real-root checks;
source half bytes remain unchanged. Completion-empty partitions occur in every
configuration, and fixup-empty coverage is checked separately for each compiled
head-dimension specialization. Bounded occupancy is 1 / 2 for D256 / D512;
original columns 4 / 1, logical grids, mask prepass and scratch are unchanged.
The inherited Gemma2 cap50 operand control also passes unchanged; final
composition repeats the Gemma2, Gemma3 and Gemma4 bounded controls together.

The adoption case uses prefixes 2560/3072, crossing both physical local rings:
2048 cells for Gemma26 and 1280 for dense31. Actual Gemma4Chunk write indices and
read widths are checked against the runner layout. Both profiles preserve all
eight padded-versus-bounded payload/state files and all 68 stock heads, with
zero differences over 64 scored transitions. At frontiers 2563/3075, local write
cells are 515/1027 for Gemma26 and 3/515 for dense31; global reads are 2816/3328.
Local reads share the physical capacity, so only global layers select bounded
roots (5 / 10); the 25 / 50 local layers keep the equal-width legacy path.
No ring timing ladder is claimed.

The final true-parent composition preserves the Gemma2/Gemma3 bounded operators
and serving recipes, reruns all three families' operator controls, and preserves
all eight current own payload/state files for both Gemma4 profiles. Ordinary HTTP
then passes five cases per profile: parallel unequal-prefix literal likelihoods;
physical-ring continuation, bounds refusal and healthy followup; cached chat,
SSE and stops; queued third request and client departure; and two-slot clean
restart with exact matched cached scalar replay. Kept record bytes remain
unchanged during adoption and all four runtime epochs retire with exit zero.
The first epochs complete 67 / 65 joined generation groups (134 / 130 owner
units) for Gemma26 / dense31, with maximum selected bounded-plan counts 30 / 60.
The restored scalar epochs complete without joined groups. These counters
combine successful waves with bound-plan evidence; they are not observed kernel
launch counts. Cold/cached and scalar/joined equality are not imposed across
different geometry, and this HTTP control makes no serving latency claim.

## Reproduction and provenance

The [native probe](../../../benchmarks/gemma4_common_width_probe.cc) and
[original public-API caller](../gemma4-common-width/llama_probe.cc) accept
`ARTIFACT IDS0 IDS1 NEW_OUTPUT 26|31 POLICY MODE` and
`GGUF IDS0 IDS1 NEW_OUTPUT 26|31 MODE`, respectively. Native policies are `baseline`, `candidate` (physical
padding) and `bounded`; modes are `own` or `cycle`. Stock modes are `teacher` or
`cycle`. Inputs remain external: retained source
`90bf82fd26b7040c513e7370eca2cd283ac8fd15ce89a2e8d05d6f7e670b5ba2`
contains four 8192-token histories; owner0 starts at zero, owner1 at 8192.
Each supplied file has its prefix plus three warm and 32 teacher IDs. The
[common analyzer](../gemma4-common-width/analyze.py) bounds the short and ring
recipes explicitly; raw results and supervised queues stay outside Git.

Measured on Spark A / GB10, driver 580.178.04 and pinned SDK c09 / CUDA 13.4.
Approved artifact/source pins remain in [Gemma4](../../gemma4.md). Stock is
llama.cpp v0.6.0 `d81235049384534c167caea52b85a694f6103d14` in its original
`c604ea4f…0607db` image. Source, actual binaries, receipts and retired containers
are bound to the supervised results under
`~/.local/share/llmp/gemma4-owner-bounded-20261007` and local
`/tmp/llmp-m35-coordination/gemma4-owner-bounded-*` evidence.

Task-entry TensorFold HEAD was freshly checked as
[`041d14a94e951834470fd514ed33e65b8be1059a`](https://github.com/ashhart/TensorFold/blob/041d14a94e951834470fd514ed33e65b8be1059a/README.md).
The native Zig 1.0.0 README lists Gemma4 under qualification and no matching
qualified approved Gemma4 CUDA/GGUF recipe. No TensorFold timing is claimed.
