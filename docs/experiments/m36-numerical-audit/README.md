<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3.6 numerical-policy audit

Owner direction, 2026-10-12: optimize quality and performance against the
original unquantized model; do not preserve another library's quantized
rounding merely to obtain identical bits. This clarifies D-107's existing
"speed and quality, not bit-exactness" rule. It does not change the matched
1% migration speed gate or the memory, state and source-integrity contracts.

## Scope and conclusion

Read-only audit at `50a2b72047b44886d2ccdfb712a7281727bf8d78`: inventory of
72 M3.6 commits and source inspection of numerical decisions in the Qwen,
DeepSeek, Gemma, FP16/EXL3, importer and lifecycle work, including pending
Qwen45 and DS53 experiments. This was not a line-by-line review of all 72
commits. No new performance or original-model quality runs were performed.

Concrete opportunities exist: landed code can split joined products and
force legacy vector kernels to recover matching bits. Pending experiments
extend that machinery. Extra graph construction and selection restrictions
are established; their net latency, memory and quality costs are unmeasured.
Diagnostic exactness had become a blocking criterion despite D-107. Stop
that drift before adding another arithmetic correction.

## Candidates and evidence

| Work | What the audit established | Next decision |
| --- | --- | --- |
| Qwen G3b, `3c74d425` | [Mapping policies](../../../src/spec/mappings/qwen38.cc) enable `owner_row_invariant` and retain the old fast form's fusion selection. [Lowering](../../../src/kernels/ggml/lower.cc) can force singleton/vector products. Available norm/product fusions are disabled; individual eligibility and quality benefits are unproved. | Compare ordinary eligible joined products and selected fusions independently. Preserve optional exact-mode claims, but do not impose them on the fastest qualified default. |
| Owner products, `ceee5d69` | [Planner rewrite](../../../src/planner/owner_products.cc) expands joined products into owner views/products/concatenations; lowering insists on original kernel recipes. Landed MTP protection is extended by pending Qwen router/PLE experiments. | Separate source, row coverage, funding and completion proof from legacy arithmetic selection; retire unused numerical machinery after callsite review. |
| Selected HC, `0490a98a` | [Recorded experiment](../m36-selected-owner-hc/README.md) has identical consumed inputs/weights but changed bits with vector versus cuBLAS. It restores the vector recipe without matched quality/performance qualification. | Evaluate both lawful implementations by quality and speed; changed bits alone select neither. |
| Pending Qwen41/45 | Router/PLE dense owner splitting restores numerical agreement. Authenticated Qwen45 captures show all 34 late roles, 53 checkpoints and all eight consumed head-input rows exact. Remaining owner0 head difference is singleton MMVF versus joined MMF. | Freeze45 as a diagnostic arm. First recovery candidate removes router/PLE recipe requests; do not add a head split just for parity. |
| Pending DS50/53/53b | DS50 router IDs already matched; six weights differed by 1–2 ULP before sequential-sum preservation. DS52 full GLU inputs matched, but down-product accumulation differed by less than `1.2e-7`; DS53 proposes another default-off vector recipe. | No adoption merely to match the quantized legacy reduction. Keep actual selected experts, coefficient semantics, immutable weights and replay safety. |
| Other pending DeepSeek HC/MoE paths | Several experiments add FMA fusions rather than disabling them. Projection-A differences can disappear at projection-B output. No original-model quality or matched speed preference is established. | Screen a small set of fused/ordinary alternatives; do not confuse a restored fusion with mandatory legacy rounding. |
| Gemma migrations | Gemma2/3 keep useful norm/GLU/attention fusions; Gemma4 keeps qualified bounded recipes. The existing migration gates passed, including the 31B assistant's approximately 4.4% gain. Owner isolation and disabled patterns may offer further batching/fusion gains, but also encode qualified capacity/attention domains. | Requalify those domains before widening them. No measured regression solely for bit parity was found. |
| Qwen2 FP16/EXL3, `ab21dd76`, `04470706` | Float product/add and GLU fusions were added. Score scaling, explicit attention, F16 intermediates and paired EXL3 products preserve fixture recipes and merit reevaluation. EXL3 kernels also have real F16 input/output requirements. | Compare avoidable attention materialization/rounding separately from the selected kernel's physical precision contract. No serving performance qualification follows from fixture exactness. |

Current ordinary Qwen serving remains on the legacy runner. The shared
Qwen serving path admits only its diagnostic sole-owner envelope
([component build](../../../src/runtime/component_build.cc)); joined owner
machinery is not proof of a released joined-serving speed regression.
Pending Qwen and DeepSeek edits are not landed on main. Diagnostic screen
elapsed times include captures and expensive branch certification; they
are not matched serving latency measurements.

Follow-up source inspection also found inherited multiply/add rounding in
`src/kernels/ggml/llmp_fused.cu`, whose header explicitly prevents contraction
with `__fmul_rn`, and in `dsv4_hc_norm.cu` / `dsv4_weighted_reduce.cu`.
These arithmetic expressions predate M3.6; its Qwen lowering reuses them.
They are additional candidates for a fused alternative, rather than evidence
that M3.6 newly disabled FMA. Their quality and speed tradeoffs are unmeasured;
changing the approved operation recipe and comparing it comes before adoption.

## Safeguards that remain

- Exact source bytes, layouts, hashes, imported permutations and artifact
  publication protect representation integrity. BF16 transient widening,
  zero-centered norm `+1` and state transforms have semantic purposes.
  Resident weights still cannot exceed source encoding.
- Declared precision conversions are IR semantics. The fusion matcher's
  `ElidesF32` check must not silently drop a declared non-F32 intermediate.
  If the conversion is an avoidable legacy recipe, change the component or
  spec policy deliberately before optimizing it away. RoundedIndexScore
  similarly declares F16 rounding; revisit intended indexer quality before
  changing that operation, retaining visibility and selected-mask fidelity.
- Owner histories, recurrence, rollback/snapshots, owed drafter progress,
  admission, source binding, capture lifetimes and retirement remain real
  correctness contracts. Every expert selected by the candidate must run
  with its own weights and contribution. Different qualified arithmetic
  need not select the legacy reference's router IDs.
- Within one selected implementation, uninterrupted versus checkpoint,
  spill/restore and rollback continuation remains exact. A new numerical
  recipe needs a new plan/retained-state identity or a compatibility proof.
  G5 trajectory/eviction tests add no new hot-path bit-parity policy.

Historical strict failures and exact passes remain recorded as observed.
They are not retrospectively relabeled quality failures or quality passes.

## Prospective quality and recovery protocol

1. Pin original source-precision weights, conversion provenance, tokenizer,
   model/config/cache semantics and reference implementation. Compare the
   candidate and migrated baseline to that same reference; equal-or-better
   quality does not require a quantized model to reproduce original logits.
   Quantized
   legacy engines remain secondary controls; dequantizing their checkpoint
   does not reconstruct the original model. Existing accepted results stay
   historical, without a silent reference rebase.
2. Use identical fixed teacher histories and document/target coordinates.
   Report full-vocabulary next-token NLL, token-weighted perplexity and
   paired losses, alongside existing behavioral and long-context controls.
   Freeze coverage and acceptance bounds before new qualification runs;
   old 3% and near-tie budgets do not automatically transfer to a new oracle.
3. Reuse one shared target-only `kScored` execution path and bounded reducer.
   Pay full-head outputs, copied histories, workspace, scratch and retained
   report bounds before execution; stream per-token NLL rather than retaining
   whole-corpus vocabulary heads. Scored-phase evidence does not by itself
   qualify production decode, verification or speculation recipes.
4. First screen the smallest reversible candidate with a short quality and
   matched speed comparison. Expand only when a win or unresolved risk
   warrants it. Measure cold construction separately from steady execution.
   Choose the fastest qualified implementation; remove dominated machinery
   after its safety responsibilities have been retained or retired.

The [original Qwen2 BF16 smoke](../m36-original-qwen2/README.md) now supplies
a complete authenticated original checkpoint and a tiny natural-language
likelihood observation. The existing official F16 fixture remains a separate
publication with unproved original conversion lineage. Inspected Qwen3.8/DeepSeek
stores currently provide quantized references, not authenticated original BF16
execution. Qwen's catalog lists a possible approximately 354 GB
BF16 variant as absent; this is not an exhaustive filesystem search or proof
of a usable reference. Missing primary evidence stays explicit. Secondary
comparisons and bounded operand/timing experiments can proceed without
claiming original-model quality qualification or blocking unrelated work.

## Audit provenance

Frozen external packages, all under `/tmp/llmp-m36-coordination/`, retain the
source inventories and independent reports. Raw model data stays outside Git.

| Package/file | SHA-256 |
| --- | --- |
| `m36-numerical-audit-root01/milestone-history.txt` | `42a2287acec0451ea87fbf8ca6b8e40df962174f37d4de5e75265fe65b8685db` |
| `m36-landed-numerical-audit01/review.txt` | `6bdeab3026f10050748f0835d909530ecc32cf1b695493ecba73f22b2a02432c` |
| `qwen-numerical-retrospective01/audit.md` | `4e44ba65cb528ac9c790c0c0da99dc409b4c72a3a881932a284af28fbefc1cc1` |
| `ds53-owner-steering-report.json` | `93ad5c95cb92564d7bea4b6fde708fe984caf09906deb64e62fc65f3d8c55a9d` |
| `shared-corpus-scoring-design01/design.md` | `f0e5f06164e649469521c3d1682c0ad7045864d414e1010f48b45170533fa8aa` |

The inventories bind the inspected source and observed experiments. They
establish neither a recovered speed figure nor a new quality pass.
