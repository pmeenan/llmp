<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native v1 Gemma target admission

Registration and component construction share a closed admission check over
the authenticated stored spec and native tensor formats. The approved Gemma 2
2B Q8_0 source is the initial declaration. Complete mapping equality preserves
its norm.plus_one assertion, tied output, dimensions and policy; the admitted
stored spec executes unchanged. v0 support remains. N4b2 below extends the
closed declarations to exactly three more approved Gemma targets; other native
identities and drafter compositions remain refused. Neither slice closes N4 or N5.

On Spark A, 31 focused host cases and the actual native-v1 comparison passed
with no failures or skips. The comparison runs v0 and published v1 on two
owners with unequal prompts, checks every full 256,000-entry head and generated
logit row, four tokens, history, step counts and complete initialized-state
hashes, then spills state and checks restored continuation and reused positions.
All observed rows must be finite. The fixture pays its retained result allowance
before execution and retains the complete owner on unproven retirement. Extra
external metadata parsing was removed during review; runtime admission remains
on the ordinary accounted path. The actual test took 7 seconds; this is a
functional diagnostic, not a performance gate.

The source checkpoint is the normalized Gemma 2 GGUF already qualified by N4a.
Native artifact f176f912bc7761beee21cda0c22aff2080b8b84ed99faa6b962846c7e1379076
contains 288 tensors and 1,340 chunks. No re-import or payload rewrite occurs
in this test. Source and recipe hashes, exact raw case inventories, zero skips
and unchanged test binary are checked before and after the jobs. See
[results.json](results.json) for aggregate evidence. Final regression passed 2,544 native, 1,999 CPU and all 131 model/timing
tests, with no failures or skips. Format, boundary, header and REUSE checks
passed. Main built successfully and passed 31 integration checks and the quality gates.

## N4b2: the other approved Gemma targets

The extension adds only Gemma 3 4B QAT Q4_0 and the approved
Gemma 4 26B-A4B UD-Q4_K_M and 31B UD-Q4_K_XL target files. Their exact source
names, complete-file byte counts and SHA-256 values come from
`tests/unit/data/gemma3/gemma3_4b_qat.json` and
`tests/unit/data/gemma4/gemma4_{26,31}b.json`; their upstream revisions are
`bbcac0d065076c47042838c0675c602411b0dd4c`,
`c099eb48e663fd284577b04978a94ffccb261841` and
`c1ac76e99d5513b141e8adde7288b85c3f9c32ec`, respectively.
No other native identity or drafter composition is admitted by this declaration.
The existing v0 path remains.

Each complete native tensor directory must match a pinned inventory digest.
For tensors sorted by bytewise name, SHA-256 consumes the name and LF, the
native lower-case encoding name and LF, each original dimension as unsigned
decimal followed by a comma, then LF. Format validation separately requires
default parameters, one stack, the data part, dense F32 or GGML block layout,
and expert status exactly for rank-three tensors. Thus the digest fixes every
role, dimension and per-layer encoding, while the explicit checks fix all
remaining format fields. These digests are derived from the existing captured
GGUF directories, not measured kernel support or invented checkpoint names:

| Target | Tensors / expert arrays | Inventory SHA-256 |
| --- | ---: | --- |
| Gemma 2 2B | 288 / 0 | `91678220743a65166375bad2f712380aed8e7cfec56203bfdc87321a5e5ed0f5` |
| Gemma 3 4B QAT | 444 / 0 | `8aedee1cffc43f331cb38dcd9dc60e98a2aa67cd98df16248f1c78e36bd4a8e7` |
| Gemma 4 26B-A4B | 658 / 60 | `97551b8fcfead217ce2feb8c0abe7228211b15bca6867158a955f3d5a7790e02` |
| Gemma 4 31B | 833 / 0 | `a5ba88997fe0b72a54148526573c38e6a619f5a91980c59dcb5cbc5123760f52` |

The complete stored spec must equal the existing approved profile's mapping,
including the Gemma 3 `norm.plus_one` assertion, Gemma 4's empty fold set,
tied head, all component parameters, local/global layers, and routed/dense
structure. Importer-applied folds and relayouts remain refused. The actual
stored spec and `SourceOf` formats execute unchanged. Policy selection first
rechecks this declaration, then uses the same compiled profile facts that were
proved equal to the stored spec. Gemma 4's expert-presence gates choose its
existing routed or dense recipe; no legacy artifact descriptor is fabricated.
Registration retains the admitted Gemma 4 profile for the existing default
prompt-chunk selection (1,024 or 256). The execution envelope remains bounded
by Gemma 3's existing two owners through 4,096 tokens and one owner above
that (the trained maximum is available only through the existing diagnostic
settings hook), and Gemma 4's existing
8,192-token/four-owner bounded policy domain. These are admission
bounds; the actual native-v1 control below covers only two owners at context
4,096 with short seven-token prompts, not every admitted shape.

Native expert slab construction reads block geometry from the actual
`TensorFormat`. The existing v0 element/block tables retain their semantics;
per-layer group/count agreement, member ranges, alignment and overflow checks
still govern the resulting pitch. Host controls use the actual Gemma 26
Q4_K/Q5_1 and final-layer Q4_K/Q8_0 block combinations in small v1 artifacts.

The extension compiled natively and passed 44 focused host controls with zero
failures or skips. All three sources were then published into separate fresh
local stores and independently checked against the complete source payloads,
file/spec identities, tensor/expert inventories, chunks and zero padding.
No archive write or source modification was required.

| Target | Native artifact ID | Tensors / expert arrays | Chunks | Payload bytes compared |
| --- | --- | ---: | ---: | ---: |
| Gemma 3 4B QAT | `634bc193468f5968ac142c2e5547f72777b182f3f694873075d7a50660f46626` | 444 / 0 | 1,226 | 2,519,539,712 |
| Gemma 4 26B-A4B | `f0f72ab3d6c733d15472ceeecd05fdf200c266101c537bd3253548fc38559ab7` | 658 / 60 | 9,060 | 16,931,716,216 |
| Gemma 4 31B | `4d9a255ab47519b8061fcbc38bb5e72b9008d8e3538aad5edf0e7530fa82adc8` | 833 / 0 | 9,011 | 18,807,134,448 |

The Gemma 26 publication includes 598 ordinary resources and 60 expert arrays
with 7,680 slices. The three publications check 1,935 tensors and 19,297 chunks
in total. Full source plus output page-cache occupancy is included before
publication, with no reclaim assumed: twice the pinned source file size plus
512 MiB. The per-profile bounds and actual inventories are in
[results.json](results.json); the physical admission bound also includes
6 GiB host/import workspace and a separate 6 GiB free-memory margin.

All three actual fixture06 comparisons passed on Spark A with zero failures or
skips and matching pre/post source and binary guards. Each uses the existing
shared EngineLlm path for both v0 and v1, two owners with distinct seven-token
prompts, context 4,096 and four generated tokens per owner. Every arm checks
three joined decode waves/six owner units and exactly three actual runner
executions, ruling out scalar splitting. Gemma 3 additionally joins its prompt
step; Gemma 4 deliberately retains prompts-alone and its approved 1,024/256
prefill defaults. The exact 24 full vocabulary rows per profile (262,208 entries
for Gemma 3; 262,144 for Gemma 4) must be finite and bit-identical between arms.
Tokens, steps, histories, complete initialized state coordinates/bytes/hashes,
spill restoration without recomputation and reused-history continuation agree.

Before either artifact opens, each fixture pays 32 MiB for both arms' retained
rows/metadata and 4 GiB for spill page cache within its explicit 32 GiB node
execution cap; ordinary runtime admission pays compiler, device and pinned
copies. Complete options, sessions, outputs and node owners survive uncertain
retirement. No extra external artifact parse is retained. Profile-specific
store paths select the three authentic publications; no artifact ID is invented.

These short functional controls do not qualify assistants, all context/owner
shapes, performance, other native domains or N5. Final combined checks passed
2,567 native and 2,012 CPU tests, plus all 135
model/timing cases (182 raw GoogleTest cases), with zero failures or skips;
format, boundary, header and REUSE gates passed. Main rebuilt and passed 74
integration controls (44 admission/settings/slab/spec and 30 BF16-component
controls), all three actual native-artifact comparisons and the quality gates.
The main comparisons executed three raw GoogleTest cases under one parameterized
CTest entry, with zero failures or skips and matching source, recipe and binary
guards. N4b2 is complete; broader N4 and N5 remain open. Historical Gemma 2
results above are unchanged.
