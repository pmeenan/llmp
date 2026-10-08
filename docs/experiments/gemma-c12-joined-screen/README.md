<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# First Gemma C12 joined screen

Both profiles fail their prospective C12 quality gates. Native execution uses
ordinary eight-column shared products followed by four-column products, with
whole-eight attention geometry for the first group. The normal reference runs
all twelve sequences in one physical batch. This screen qualifies neither
production batching nor a default policy.

| Profile | Native joined paid seconds | Reference paid seconds | Strict differences / outside frozen p99 | Conditional score change |
| --- | --- | --- | --- | --- |
| 31B | 8.27676 / 8.27600 | 6.08929 / 5.54673 | 34 / 8 | +3.304079%: FAIL |
| 26B | 2.75774 / 2.76887 | 1.99187 / 1.97779 | 74 / 5 | +10.454855%: FAIL |

The elapsed mean gaps are +42.25% and +39.22%, respectively. The 31B reference
pair moved substantially; these are short-screen measurements, with all four
walls retained, not sustained performance claims. Each native paid loop executes
64 groups and the reference executes 32 whole-twelve groups.

Each profile first completed scalar first/repeat and joined first/repeat on the
same immutable binary. All 396 complete head rows are finite and byte-exact
within each own pair; all twelve frontier and final initialized-state snapshots
and layouts also match within each pair. Across scalar versus joined, all twelve
frontier state identities match and all twelve final state identities differ.
This does not identify the arithmetic cause. Scalar paid seconds were
37.3483/38.2185 for 31B and 7.45295/7.50061 for 26B.

Independent native-only C12 calibration was frozen before the first C12 reference
outputs: p99 2.202470970153809 for 31B and 7.542554473876955 for 26B. Neither C4
nor C8 allowances apply. Against the first reference baseline, 0/396 rows are
byte-exact in both profiles; maximum raw differences are 21.58823 and 24.41163.
There are no reference-winner ties among the strict differences. All outside
rows and positive margins remain in [results.json](results.json); both primary
comparison jobs remain FAILED. The p99 gate is empirical, not a maximum-error
guarantee; it was not widened after observing failures.

The separate conditional check computes complete-vocabulary FP64 NLL for the
384 supplied within-history targets, using rows 0..31 to predict positions
992..1023. Row 32 predicts position 1024 and has no supplied target. Native and
reference mean NLL are 11.62398094/11.59147427 for 31B and
11.07918039/10.97974369 for 26B. Both relative exp(mean NLL delta)-1 values exceed
the independent 3% limit. This is a 384-target conditional score, not the
separate 1023-transition corpus PPL gate.

The unchanged production helper recipe retains twelve checkpoint-qualified
1024-ID histories, prefills 992 positions, then forces 32 more positions without
anchors. It uses context 4096 per sequence, head capacity twelve, plain norm
fusion on, 31B norm-RoPE/Add or 26B all routing/reduction policies, state-only
intermediate prefill, lookahead, and normal graphs. Reference context is 49152,
physical sequences twelve, F16 cache and normal graph/fusion behavior; batch and
ubatch are 256/1024 for 31B/26B. Local cache capacities are 1280/2048 and global
capacity 4096. Publication allowance remains 440516608 bytes, with no grant or
capacity enlargement. The carrier has 49152 bytes and SHA d584450079145f3d2c93f46ef24a0aabe2b3971279a1cbddbbb29f0a506ac5e3.

Measured source is the immutable 22-file candidate derived from c3300ab,
helper 4514f149 and SDK receipt eb7a4ebc. The later opt-in core and placement memo
commits were not rebuilt into this measured binary. Full identities, actual
native freezes, reference own-repeat proof, source bindings and official job
retirement records are in [results.json](results.json). The pinned client/image
and source recipe are reused from the [C8 factor](../gemma-c8-cohort-geometry/README.md)
and [production input preparation](../gemma-input-preparation/README.md).
TensorFold primary 609ca419/0.6.5 was checked at this cohort task entry; its
Gemma26 recipe is MLX-only, so no comparable GB10 CUDA run was available.

Both completed comparison and conditional jobs return quality failure. Their
skipped post-source checks were run in separately supervised authentication-only
jobs, all DONE0. Native acquisition, native freeze, reference acquisition and
all owned-container retirement checks completed successfully. Raw complete heads,
states and logs remain external. No full suite or additional model arm ran.

The next bounded factor is one ordinary twelve-column model pass with three
attention quads and whole-twelve grid geometry. It remains source-only here;
these failed results and both frozen bounds must remain unchanged for that test.
