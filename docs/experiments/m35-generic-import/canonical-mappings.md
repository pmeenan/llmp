<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Canonical LLM mappings and source dialects

U3’s explicit import, mapping and Llama3 RoPE exits are complete (2026-10-23).
Independent source review and adversarial challenge are clean. Numerical
qualification of the separate U4 arithmetic changes remains open.

Every registered LLM architecture now has one canonical mapping. The final
consolidation covers Gemma4, Qwen3.8 target and Qwen3.8 MTP; earlier work covers
Gemma2/3, Qwen2, Llama and Mistral. Canonical component roles, slot order, serving
recipes and drafting contracts are shared. A selected source dialect carries
container tensor spellings/parts, full ordered parameter chains, folds, layer
schedules and selectors, metadata requirements and the existing planning policy.
Native MTP retains native wire semantics for both canonical and checkpoint
architecture spellings; synthetic fixtures use those actual wire names.

The resolver owns changed slot and draft arrays in move-stable storage. Full
source parameter chains may use only canonical parameter identities; source
tensor tables may use only canonical roles. Source policy selection uses the
actual source architecture/container and stable static rule spans. Missing or
ambiguous context refuses; the deliberate structural-fixture default remains
separate. Qwen GGUF retains common-read and owner-row policy flags, while native
forms retain their existing source gates and flags. Both GGUF explicit key prefixes retain their original empty value; the effective prefix still derives from canonical architecture. Direct unprefixed source aliases therefore remain accepted. No numerical grant changes.

Selected-head imports authorize geometry against the actual target and classify
head tensors in the checkpoint supplying their bytes. The validated drafter
contract supplies the target role for canonical drafter source names. Native,
cross-container and canonical-source regressions check exact gathered row bytes.
Target source interpretation is resolved before import-time drafter composition;
canonical mapping identity remains the ownership and provenance authority. This
prevents a canonical repeating layer schedule from replacing an explicit native
target schedule. Serving continues to consume authenticated stored specs.

Source checkpoint identity, execution architecture and mapping provenance are
kept distinct. Native Gemma4/Qwen source aliases retain their original checkpoint
identity and canonical execution name. Existing Gemma3_text execution identity
remains Gemma3_text, while mapping provenance now correctly names Gemma3. A
read-only inventory of canonical artifact and U3 proof stores found no published
Gemma3_text model/provenance entry requiring migration:52 manifests on Spark A,
50 on Spark B, including retained originals and refreshed publications. Inventory
SHA-256 values are 7099fe4f2023367a28b00ee290eeb135a73816047565ebabfea35bcc50cb12c4
and ea7304e72a22df1befb385930b7b4364d0a9b280582a6166f779dd9ed5022aa9.
This is a scoped inventory, not a claim about arbitrary external stores.

The exact final 29-file code/test change is
SHA-256 `a231c47fa56c932530aa4231df1fa415766f5eb60436179cdfaaa6675cc2b6af`.
Independent whole-source review and adversarial challenge are clean, including
selected-head ownership, move lifetime, bounds and preservation of source fields.
The first integrated checks exposed three obsolete fixture assumptions and one
real regression: making the GGUF prefix explicit had rejected existing source
aliases. The repair restores the original field and retains the failing test;
no output oracle, numerical bound or frozen result changed.

Spark A built all native consumers from the final formatted source. The final
349 targeted registrations and inner cases pass without skips. Together with
3,340 unchanged cases from the earlier shared wave, this covers all 3,689 shared
registrations. The earlier wave's obsolete converted-fixture failure remains
recorded; its corrected five-case group and complete targeted group pass. All
31 combined changed C++ files pass pinned Clang 22.1.8 formatting with the repository's
explicit style. The final formatted A source manifest is
`412e7daba7e9d365b201656b5d387ac6c69d84d697cf133f4d4cad8d2a680044`.
This combined verification includes the separately reviewed U4 schedule candidate.
Final REUSE and header checks pass (2,590 header-checked files); all 29 U3 C++
files pass pinned formatting. They ran under the workstation’s shared host lock.

The final CLI source-context correction passes 52 focused CLI, image-reuse and
writer checks. The final native build, explicit 8-GiB confined ModelOpt ownership
control, joined-MTP replay/state control and installed-spec inventory all pass.
The final model-test source manifest is
`4e2e9192a20855adb48f586e871341bcf622650751a88d480865e2b017d0be92`;
its 2,933 paths match before and after the final check. The earlier missing-source
skip and compile failure from a shadowed test variable remain recorded; their
explicit configured reruns pass.

The affected-model union has twelve registrations: eleven pass and the Qwen GGUF
frozen numerical case remains FAIL. Both Gemma structural cases, Qwen native
ladder/depth-two/default-output controls, plain GGUF repeat, joined-MTP replay and
all three Qwen component inventories completed. No unresolved skip is counted as
passing. The remaining numerical failure is independently reproduced before the
U3 consolidation, as detailed below.

## Publication fixtures and numerical attribution

The GGUF comparison now derives sampling expectations from its authenticated raw
GGUF metadata, preserving complete spec equality and metadata-free structural
controls. The joined-MTP preflight uses its independently authenticated 4,092-byte
raw manifest hash; its semantic artifact ID remains distinct. Neither fix changes
an artifact or numerical oracle.

The native fixture now resolves its actual source dialect. Those native rules
already used `projection_scales=false` and three individual scale flags before
this consolidation; the older published artifact used `projection_scales=true`.
The model comparison explicitly checks all 48 native MoE sites, then adapts only
its expected copy to the fixed published representation before full JSON equality.
`ffn.moe` consumes the effective flags by OR, and the actual native policy does
not consult the differing informational gate. Shared fixtures and production
rules retain the native form. Complete tensor/format/part comparisons remain.

The final current Qwen GGUF frozen test differs in two heads, two sets of complete
logit rows and three state images. Tokens and histories still match; spill and
restoration preserve initialized state exactly within the run. A native build
of the pre-consolidation R source on Spark B, guarded by the full 2,933-file
manifest `0a53dea6a1df1b79002b4e3f1af689ed724ee376c9d8ded94c9d8e4f7b0eb99a`,
reproduces all seven actual/frozen strings exactly against the same current
`b31c469e…` artifact. Its binary is
`a852e28c3704ce068bba0317fa1cd2329e72d64c15aead63dbeb9b55468d9ded`;
the single control took 10.58 seconds. This attributes the bounded difference
outside the U3 consolidation; it does not qualify U4 or establish its arithmetic
cause. Both frozen test results remain FAIL, with every expectation unchanged.
The [aggregate](canonical-mappings.json) retains each actual/frozen digest and
source/result identities.

An older retained executable passes against its original `f60d801a…` artifact;
it cannot read the current artifact’s newer source-fold receipt and refuses
before execution when given `b31c469e…`. That original-artifact pass is separate
from the direct R-source attribution above. Its early setup-wrapper failures,
post-test ASLR-comparison failure and schema refusal remain retained outside Git;
none supplies current numerical qualification.

## Explicit U3 exit evidence

Original Qwen ModelOpt/MLX parity covers all 323,607 named parts, including expert
slices ([proof](original-parity.md)). Stored target/drafter publications and
source folds retain their separately documented payload/spec/provenance proofs.
The final source tables preserve those chains and require no new payload import
in the inventoried stores.

The owner-approved Llama3.3 plain ModelOpt NVFP4 checkpoint and the distinct
qwen3_5 MLX checkpoint import and serve. Llama's short/depth likelihood controls
and five original behavior controls pass their fixed bounds. Its original
256-token Fibonacci observation remains native FAIL/reference FAIL/comparative
UNRESOLVED; the separately registered 512-token supplement passes all four fresh
completed observations. Those are distinct protocols, not a relabeled original
six-case pass. Qwen27's short/depth repeat pairs and all six completed raw-ID
behavior cases pass; the earlier 64-token chat smoke still has no final content.
See the [Llama controls](../m35-u3-quality/RESULTS.md), [supplement](../m35-u3-quality/FIBONACCI512-RESULTS.md)
and [MLX controls](../m35-u3-quality/qwen27/RESULTS.md).

A source-contract audit establishes applicability without claiming identical
current binaries or newly captured full plans: Llama already used its qualified
NVFP4/MXFP8 prepared products before U4, outside the changed block-Q8 VecQ domain.
The qualified MLX source already includes the component evidence join and SiLU
gate repair; affine products retain their arithmetic. The later shared GDN gate
shape expansion has independent exact primitive-chain/output-view/save-consumer
checks ([bridge](../gdn-gates/RESULTS.md)); it is explicitly a routing expansion,
not identical selection. The final canonical consolidation does not change either
proof's source tables, stored spec, payload, F16 state or greedy request settings.

These bounded controls satisfy the explicit plain-NVFP4, second-architecture MLX
and Llama3 RoPE quality requirements with the final integration checks above. They do
not qualify general Gemma NVFP4 quality, assistants, new formats/devices, full
context, batching or competitive model speed. Those remain in their own plan
items. Future AWQ/GPTQ formats will use this generic route when added; no absent
encoding is declared supported by the routing contract.
