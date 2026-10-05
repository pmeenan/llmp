<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Closed dense31 packed-attention C4 screen

Benchmark-only comparison, base 7842403; no production change. Latest primary TensorFold
observed 2026-10-05 12:05:02 UTC at 609ca419abecebdc5a059498a613680bd3aa847f,
0.6.5. Gemma26 MLX recipe; no dense31 same-format CUDA comparator.

Compile a separate helper derived from gemma_joined.cc, adding funded completed
state witnesses and outside-timer finite checks, with a benchmark-only
BuildGemma4Graph link wrapper. The prior helper remains unchanged. Control returns the original builder unchanged. Candidate accepts
only the full dense31/context256/four-owner/one-row/equal read256/head4 graph.
Other shapes retain the original graph and report scope. C1 prefill is unchanged.
Both arms use existing norm/RoPE and norm/residual policies, ordinary products;
row-invariant/sharedQ8/RoPE-store/MoE/generic fusion stay off.

Authenticate four original attention descriptors and eight exact cache writers
per layer. Balanced CONCAT raw Q/K/V on dimension3, then permute0,2,1,3;
CONCAT the separately padded [256,32] masks on dimension3. K/V packing views
use the respective SET_ROWS result as their source, enforcing all current-owner
writes before packing through existing DAG dependencies. Use one existing
checked MMA attention per layer. Its packed [D,heads,1,4] output supplies separate
owner views to unchanged downstream consumers. Preserve readable names/keeps,
all intermediate funding and source generations. Regenerate bounded DAG order;
check selected norm counts and no old attention execution. Only benchmark-owned
wrappers/additive manual CMake are permitted; no new VMM/scratch/runner policy.
Extra descriptor capacity must enter existing arena/Setup funding. Actual
PlaceAndPlan activation/scratch envelope and shared closure remain authoritative.

Use exact1024-ID SHA b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610.
Owner i prefills [0,64+i); warm8, reset identically, then three untimed anchors.
Existing helper pays32 waves/128 full vocabulary heads at positions67+i+s,
including input staging/publication/argmax. Full-array finite validation and
initialized-state witnesses run outside the timer, as does disk output.

Freeze source/input/binaries/build receipt before acquisition. Acquire unchanged
control and candidate first/repeat, verify finite complete heads and same-policy
whole-byte equality; retain initialized-state witnesses outside timer. Freeze
independently/exclusively BEFORE fresh stock comparison. Inspect first1/3-unit
rows before attributing a broader effect; no stock-led native selection.

Fresh original b29/digest837fc... reference uses physical C4/ubatch128/F16KV and
fusion+graphs ON. Same forced IDs/prefixes/timers/output publication. Short
reference/candidate/reference bookend; all128 rows get bytes/raw/strict and
reference-winner-over-native-choice margin checks. TV/NLL is only explicitly
selected rows, never corpus PPL. Preserve all repeat failures and shape variation.

This packs BOTH local/global layers: a joint dispatch/stream geometry factor,
not precision alone. All CONCAT copies are paid and funded. Analytic byte traffic
is not measured latency/peak. No default, support, verification, C12/context or
ubatch ladder follows a negative result. Source review precedes model work;
final independent GENERAL/root ADV/light, no agent commit. Each initial native
or bookend job expected15-30s under installed GPU supervisor timeout600 on
Spark A m3fixb. Original comparator and current C1 paths remain unchanged.
