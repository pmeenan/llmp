<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma2 bounded C1 execution

This unit adds a native Gemma2 runner over the shared paged engine,
closed to the approved 2B Q8_0 profile and prepared source identity. No serving
adapter, owner attention or supported execution-matrix admission is added.
The representative C1 screen reproduces all 33 stock full heads byte for byte;
a short matched GPU-token timing screen is level with stock. Broader model
qualification remains open.

The graph follows llama.cpp d81235049384534c167caea52b85a694f6103d14
`src/models/gemma2.cpp`: scaled embeddings; direct GGUF norm weights; distinct
raw Q/K/V projections without Q/K norms; NEOX RoPE base 10000/scale 1; Q multiplied
by 0.0625 after RoPE; F16 separate K/V; flash attention scale 1/softcap 50;
post-attention norm/residual; parallel GeGLU; post-FFN norm/residual; final-row
gather before the last post-attention norm/FFN; output norm/tied head; then
`/30 → tanh → *30`. The GGUF converter already applied the norm +1; import and
execution keep its payload unchanged. Every quantized width 2304 leaf marks the
existing whole-tensor 272-byte readable tail, funded by prepared import.

State descriptors retain alternating local/global layers, aligned separate
F16 K/V and bounded local rings with `pad(min(context,4096+max_rows),256)`
cells. Independent descriptor segments remain checked, while this first
runner deliberately admits one slot. Public mutable bindings, cache roots,
consumer correspondence, placement overlap, capacity and staged sources are
rechecked. Held requests, completion-aware output/capture ownership, retained
Clear and initialized-state spill/restore reuse the shared runner skeleton.

The default runner retains primitive fusion settings. Explicit `optimized`
probe policy requests the already available norm/Mul and quantized GeGLU,
plus the newly operand-qualified width 2304 norm/residual chain. Norm/RoPE is
ineligible because Gemma2 has no Q/K norms. Primitive fallback remains.

## Source and own controls

`pins.json` names the approved exact-revision source and whole-file identity.
The shared importer preflight checks every typed metadata field and all 288
actual descriptors before planning; the unchanged writer hashes the whole
source and reparses it before publication. No format/schema/writer identity
changes. The complete 2,784,495,456-byte source locally matches SHA-256
`2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe`.
Deep verification accepts prepared artifact
`eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870`
(1,340 chunks, three files). The converter remains
`m3-1+layout-a0d1980a9eddd1ad`. The earlier fixture's range-only provenance
remains historical, distinct from this authenticated full-source import.

The first native screen uses context 4096/C1/F16 KV, max_rows 128 and one head
publication. One actual text is tokenized to 291 IDs including BOS 2. Two 128-row
chunks prefill 256 (first state-only, second one head), then three supplied
scalar rows precede 32 teacher-forced rows. The 33 finite full-vocabulary heads
include 32 labeled transitions and one unscored final row. Eager and two fresh
captured processes must be byte-exact before stock is read. Lifetime controls
check refusal leaves state unchanged, state-only/head equality, retained Clear
and exact initialized-state spill/restore. A separate C1 device-greedy control
compares 32 chosen IDs, final full head and initialized-state hash against the
full-head path, with mixed/duplicate/output-mode refusal and unchanged metadata.

CPU graph/plan/state tests include an actual ring wrap beyond 4096, fresh
causal/window masks, wrong shapes/types/readability, detached mutable roots,
short/overlapping storage and one-byte-short source/activation funding. The
width 2304 norm/residual operand control uses the existing FP64 bound, both
ADD orders, rows 1/128 and captured fresh inputs; neighboring widths stay refused.

## Reference and limits

The tiny public caller uses the retained original v0.6.0 CUDA image and public
API, with the same context/chunk/KV geometry and independently checked 291-ID
lineage. Two stock own repeats precede comparisons. Teacher heads are complete
finite F32 rows. Timing compares native device-token publication against the
original backend greedy sampler, with a paid final verification head. Stock
also exports full sampled-logit rows internally, as in the Gemma3 method. Original stock
fusion/graphs remain enabled; all containers must retire before interpretation.
No new numerical allowance is introduced: strict greedy differences and
likelihood deltas remain separate, and any failure is retained before timing.

## Actual focused results

Checks ran on spark-b (NVIDIA GB10, driver 580.178.04, locked ARM SDK
`aarch64-c09daba6ac31edee`, CUDA Toolkit 13.4.2) on 2026-10-07. The native
probe SHA-256 is
`8f052d481dc0158c5995df8ff4f5c8233b21cd24b8029cef59be3e724652e481`;
its build receipt is
`874aaf7a5967cfbe91054e0d8fc1a0630f952e0e8eb54b831d09719f1e08ce89`.
The retained stock image is
`ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db`
(v0.6.0/d812); its public-API helper SHA-256 is
`4849f6d683c4e498e0c1856a3a4891ed3eda04478a08314f20c80e39010e6847`.
Whole-source checksum sync and an empty itemized dry run preceded the native
build. Every run retired successfully under installed supervision; full suites
and shipment tiers were not repeated for this focused slice.

| Control | Result |
| --- | --- |
| Gemma2 model/state, graph and plan tests | 8 + 4 + 5 PASS |
| Norm CPU and width2304 GPU operand controls | 15 + 1 PASS |
| Shared importer controls | 26 PASS, including seven Gemma2 and seven Gemma3 |
| Native primitive own repeats | Exact 33 heads and 32 choices |
| Optimized eager, two fresh graph processes and device-greedy own control | Exact 33 heads and 32 choices |
| Native lifetime | Unchanged refusal state, state-only/head, retained Clear and initialized-state spill/restore PASS |
| Stock teacher own repeats and native comparison | All 33 heads byte-exact; zero strict/tie differences and likelihood deltas |

The actual 291-ID input SHA-256 is
`c9d26c7b701cc2e1b06f8f272eaa2d5745c54877e816008a9fce0a641af9768f`;
its private UTF-8 prompt SHA-256 is
`3016ddb3960e77f9a9dc649bca88dad0a7578d8afb07dd8ecf00b98144dc3756`.
Supply that text externally, use the native probe's `prepare METADATA TEXT NEW_OUT`
mode, and require these identities plus independent stock retokenization before
replaying this exact screen. The 33 optimized native and stock heads share
SHA-256 `ebdc660b8e79703d56af9672b7a30d39a1f8e119f8e4b605ead06a1c802ee442`.
The 32 scored-target mean NLL is 3.72364433815 in both engines; minimum stock
top-two margin is 0.124127388. Maximum raw-logit, target-NLL and total-variation
deltas are zero. No historical noise allowance was used. Selected-plan counters
for an optimized teacher process are 157 norm/Mul, 27 quantized GeGLU and
154 norm/ADD, with zero norm/RoPE; these count bound-plan selections, not replays.

## Short matched GPU-token timing

A single R/N/N/R uses two fresh processes per engine and the same input,
context4096/C1/F16KV/128-row recipe. Each warms 256 prompt + three supplied rows
+ eight greedy transitions, clears state, pays the two-chunk 256-row prefill,
runs three supplied rows untimed, then pays 32 greedy transitions. Native publishes tokens during the paid greedy path, except the paid final
full head. Stock uses its original backend greedy sampler and sampled-token API,
with original fusion and graphs enabled; d812 also transfers full sampled
logits at each output. Skipping the separate raw-logit copy does not suppress
that sampled-logit transfer. Startup, import, warm-up, Clear and the
three supplied rows are excluded from paid time. The native warm-up retains
three off-clock full-head publications from the established caller recipe;
the paid boundaries match. The recorded stock caller clears memory with
`data=true`, unlike the logical-only reset in the subsequent Gemma4 screen;
these measurements retain that configuration. `performance.py` requires finite positive timings,
exact same-policy repeats, matching natural histories/final heads and retired
containers before accepting the screen.

| Mean of two processes | Stock | Native | Native latency change |
| --- | ---: | ---: | ---: |
| Prefill | 50.4891 ms | 50.54445 ms | +0.1096% |
| Decode | 431.7115 ms | 431.4155 ms | −0.0686% |
| Paid prefill + decode | 482.2006 ms | 481.95995 ms | −0.0499% |

All four natural 32-token histories share SHA-256
`6eacddf7bf55b1b395f65f6664ae23085f33aefdad2d5b609f4cc53f6b61b641`;
all final heads share
`867b75a72cc7c9bd433613304ea34e5b30866f7242e6f643edfb36755dda925d`.
This n=2 short screen establishes no sustained speed advantage. No extra kernel
retuning or production serving policy is selected.

The per-task TensorFold check on 2026-10-07 resolves HEAD
`ed78d6fc204d89d90b045bf033d6551e7714f3a1`; its current documented model/backend
list has no Gemma2 CUDA/GGUF path. This is a current applicability check, not a
performance claim. Same-format stock remains the reference for this profile.

This representative C1 screen does not establish broader context, batching,
softcapped owner attention, chat-template/HTTP behavior, sustained performance
or model-switch qualification. Those gates remain separate before support closes.
