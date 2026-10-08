<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Engine components

The design M3.6 builds: one engine assembled from shared components, which
runs any model the importer can describe, in any weight format the format
layer can decode, on any backend that implements the operations (D-107 to
D-110). It replaces the per-family runners, graph builders, bindings and
serving adapters that M3 and M3.5 grew one model at a time. The owner set
its direction on 2026-10-08; this document is the contract the M3.6 tasks
in [plan.md](plan.md) build against, and [engine.md](engine.md) describes
the code as it lands.

## Why

M3's cleanup ([portability.md](portability.md#the-runners-shared-skeleton))
shared the *mechanisms*: paging, live-state regions and spill files, plan
caches, graph capture, runner resources. Each family still writes its own
*orchestration*, graph and binding. Measured on 2026-10-08 (`main` at
`cc70d69`):

- About 35.5k lines are family-specific (`src/engine/*_runner`, `*_plan`,
  `src/model/<family>`, `src/kernels/ggml/*_graph`), against about 8.3k of
  shared skeleton. `runtime/serving.cc` adds one serving class per family
  (about 2.8k lines), chosen by an architecture-string `if` chain.
- Gemma2's and Gemma3's runner, plan and serving class differ by about 100,
  19 and 32 lines after their names are normalized. The copies have drifted: only
  Gemma2 refuses a selection that drops a peer with a pending restore and
  refreshes closures inside `Restore`; only Gemma3's graph bounds the packed
  mask at 131,072 cells, and only Gemma2's marks row padding readable.
- Slot-lifecycle methods are 80–99% similar across families (`Restore`
  0.99 and `PrepareRestore` 0.96 between Gemma3 and Gemma4; `SplitForSpill`
  0.95, `CheckPlaces` 0.94 and `RecoverInPlace` 0.88 between Qwen3.8 and
  DeepSeek), with no common interface: every runner is its own
  `final : PagedModel` exposing the same twenty-odd verbs.
- One concept is implemented several times: hyper-connections twice
  (`llmp.hc.mix` for Qwen3.8, `llmp.dsv4.hc_mix` for DeepSeek), MoE
  three ways.
- Formats are woven into families: Gemma2's graph carries its Q8_0 paths,
  Qwen3.8's binding its own `Qwen38Mxfp8` (codes, scales, BF16, matrix).
- Each runner admits one checkpoint by profile equality (`Gemma3_4BQat()`),
  so another size of a known architecture is a code change.
- The importers are per-checkpoint Python scripts under `docs/experiments/`
  that apply family-specific folds.

The 2026-10-07 transfer rule made every optimization a per-family port; the
open transfer item held 128 open cells across 47 techniques, and 114 of
those cells belonged to 33 techniques each open for two to eight recipients. The rule existed because duplicate
implementations kept missing work already done elsewhere. The fix is to
have no duplicates: an optimization is written once, in the component,
pattern or lifecycle it belongs to, and every model that uses that piece
has it.

## Goals

The owner's requirements (2026-10-08):

- **Optimize once, benefit everywhere.** Every mechanism, kernel, fusion
  and format exists once. A model is a description, not code.
- **Every family, every quant, every container format, within reason.**
  Adding a model of a known shape is a mapping entry; a new shape is a new
  component; a new format is a decoder and a layout; a new container is a
  reader.
- **All pipelines share the building blocks**: decoder LLMs, drafters,
  vision and audio encoders, image diffusion and VAEs, embeddings and
  rerankers, decision models and, later, speech synthesis.
- **Speed and quality, not bit-exactness with today's code.** During the
  migration no family may get more than 1% slower than its current
  implementation, and should be at least as fast; quality must be equal or
  better ([migration](#migration-and-its-gate)). New models later must be
  within 1–2% of the best competing engine, and faster where possible.
- **Weights stay compressed.** Resident weights never grow beyond their
  source encoding; when hardware lacks a format, execution gets slower,
  not bigger ([formats](#weight-and-state-formats)).
- **Sharding of any shape**: tensor, pipeline and expert parallelism, any
  split count, mixed, chosen to minimize cross-device traffic.
- **Every platform eventually**: NVIDIA (GB10, then discrete), Apple, Intel,
  AMD; Linux, macOS, Windows. The core names no vendor and no kernel
  library.

## Layers

| Layer | Owns | Replaces |
| --- | --- | --- |
| Importer (native C++) | Container readers, format recognition, the architecture mappings, folds, the prepared artifact with its model spec | Per-checkpoint Python importers |
| Artifact | Encodings and layouts of every tensor, dependency groups, the model spec, per-target layout variants | v0's library-named representation families |
| Component catalog | Each component's graph fragment, weight roles, state contract, batching capability, partitioning options and participating patterns | Per-family profiles, bindings and graph builders |
| Graph IR and planner | One model graph from the spec; fusion patterns; implementation selection; activation placement; partitioning; capture keys | GGML-graph planning, per-family plan files |
| Execution | One runner over the skeleton: slot and state lifecycle, wave composition, prefill orchestration, decoding-mode orchestration, graph runs | Per-family runners and wave planners |
| Serving | One adapter per pipeline kind (LLM, image, embedding, speech, decision) | One serving class per family |

The resource core (catalog, memory, scheduler, artifact reader, tokenizer,
chat) is unchanged and still names no architecture (D-068).

## The graph IR

Llmpalooza's own IR (D-107), modelled on GGML's graph and taking the best of
the other engines; GGML becomes one source of kernel implementations, not
the graph runtime.

- **Nodes** are operations from one versioned op set: products (dense,
  vector, grouped/indexed), attention (with variant parameters), norms,
  rotations, elementwise and reductions, routing, recurrences, convolutions,
  gathers and scatters, cache stores, sampling, collectives. An op set
  change is a reviewed change, like an artifact schema.
- **Tensors** carry shape, strides and views, a role (weight, state,
  activation, input, output), and for weights and state an
  [encoding and layout](#weight-and-state-formats). No vendor or library
  type appears in the IR.
- **Construction** comes from the model spec: the planner walks the spec's
  layer list and asks each component to emit its fragment for the step's
  shape (rows per request, past length, phase kind). Families have no graph
  code.
- **Fusion is a rewrite**: patterns match subgraphs (norm then rotation,
  residual add then next norm, gate/up/GLU, attention with its masks and
  cache stores, routed experts with their reduction) and replace them with
  a fused node when an implementation exists for the matched encodings,
  shapes and device. Component boundaries never block a pattern. A
  pattern that does not match leaves the primitive graph, which is always
  correct (the registry's primitive-fallback rule, portability.md).
- **Selection** stays D-053's: per op, encoding and layout, shape range,
  device capability and build profile, by correctness then measured speed,
  bound into the admitted plan and visible in diagnostics.
- **Lowering** binds each node to its implementation and the planner's
  activation placement, then records it as a launch sequence the device
  runtime can capture and replay (`RecordedWork`; CUDA graphs today, the
  launch-by-launch fallback elsewhere).
- **Not a tensor compiler** (vision.md's non-goal): the IR selects among
  explicit, compiled implementations and generates no kernels.
- **Plan keys** are the spec's digest, the step shape and the selected
  patterns, so capture-ahead, plan caches and `PlanAccount` work unchanged
  for every model.

Which GGML pieces to adopt as the IR's starting point, and which ideas to
take from MLX, vLLM/FlashInfer, SGLang, ExLlamaV3 and TensorRT-LLM, is
[surveyed below](#survey-of-other-engines).

## Components

A component is everything one building block needs, declared once:

1. **Graph fragment**: the ops it emits for a step shape, with its
   parameters (head counts, dimensions, window, softcap, rotation variant).
2. **Weight roles**: the tensors it binds, each with its accepted encodings;
   the importer's mapping fills them.
3. **State contract**: what state it keeps per request (none, KV cache,
   ring, compressed cache, indexer cache, recurrent or convolution state,
   drafter ring), its size rule against context and rows, its encoding,
   which ranges a step reads and writes, and whether the state supports
   truncation, snapshot or neither for speculative rollback. The live-state
   skeleton derives regions, backing, spill, restore, snapshot and adoption
   from this contract; no runner writes them.
4. **Batching capability**: which ops join rows across requests (products,
   routed experts, heads) and which stay per owner (attention over one
   request's cache, recurrences), so the wave composer can join any model.
5. **Partitioning options**: how it splits across devices (heads, columns,
   rows, experts, whole layer) and what each split communicates per step.
6. **Patterns** it takes part in, and its **CPU reference**: a slow scalar
   implementation of its fragment, the oracle for tests and ports.

### Initial catalog

What today's families use, from the source inventory of 2026-10-08 (G2,
G3, G4 = Gemma2, Gemma3, Gemma4 26B-A4B and 31B; Asst = Gemma4 assistant;
Q2 = the Qwen2 FP16 and EXL3 fixtures; Q38 = Qwen3.8 with MTP; DS =
DeepSeek V4 Flash; DSp = DSpark; Img = Qwen-Image-2.1). The last column is
what exists more than once today and becomes one component.

| Component | Used by, with variants | Implemented more than once today |
| --- | --- | --- |
| Token embedding | all; scaled by √width (G2–G4), host lookup (Q2, DS), into hyper-connection streams (Q38, DS) | host and device lookups in `model/qwen2.cc`, `engine/dsv4_plan.cc`, `kernels/image/ops.cu` |
| RMS norm | all; weighted, unweighted (DS heads), L2 (Q38 GDN), gated output norm (Q38), BF16-rounded (Img), true `(1 + w)` only in Img (GGUF folds Gemma's offset into the weight) | a `Norm` helper in every graph builder and in `kernels/exl3/qwen2.cc` |
| Norm placement | pre (Q2), sandwich pre/post with a layer output scale (G2–G4, Asst), inside the residual mix (Q38, DS), adaLN (Img) | inline per graph builder |
| Rotation | NEOX (Gemma, Q2, Img text encoder), NORM pairs with YaRN (DS), interleaved multi-section, partial (Q38), proportional frequencies (G4 global), 3-axis complex (Img DiT); base and scaling per layer type | wrappers in `qwen2_graph`, `exl3/qwen2.cc`, `dsv4_graph`; literal bases in `gemma3_graph` |
| QK norm and rotation, fused | G3, G4, Asst, Q38, DS, Img | four kernels: `ggml.rms_norm_mul_rope.fused`, `llmp.qsa.prep`, `llmp.dsv4.qhead`, `image.head_norm_rope.*` |
| Dense attention | GQA with softcap (G2), sliding ring and global layers (G2–G4), tied K=V and V-norm (G4 global), MQA with sinks and window (DS), query-only reading the target's cache (Asst), small causal (Img text encoder), bidirectional FA2 (Img) | Q2's GGML and EXL3 builders differ; image attention separate |
| Owner-root attention (joined requests reading their own caches) | G2 (two owners), G3 (2–12), G4 (rewrite pass) | inline in two builders plus `gemma4_attention.cc` |
| Masks | causal/ring producer (G2–G4, Q38, DS), noncausal block (DSp), host-built (Asst, Q2) | adapter per `*_plan.cc`; hand-written ring in the assistant plan |
| Sparse attention with an indexer | QSA top-k over pooled blocks (Q38), lightning indexer top-512 (DS) | top-k masks in both builders |
| Compressed KV | QSA block-key pooling (Q38), CSA/HCA gated pooling (DS) | `llmp.qsa.pool`, `llmp.dsv4.compress` |
| Linear attention | gated delta net with convolution state (Q38) | one, with fixed head count |
| Attention output gate | Q38 | one |
| Hyper-connection residual | four streams: sigmoid mix (Q38), Sinkhorn mix (DS, DSp); head mix | three kernel sets; stream init and head in both builders |
| Dense GLU | GeGLU-tanh (Gemma), SwiGLU (Q2, Q38 shared expert, Img), clamped SwiGLU (DS), GELU-tanh (Img) | image and GGML versions |
| MoE router | softmax top-8 of 128 with expert scales (G4), softmax top-10 of 512 (Q38), √softplus with selection bias, scaled (DS), token-hash layers (DS) | three routers and three plain-graph copies |
| Shared or parallel experts | parallel dense FFN (G4), sigmoid-gated shared expert (Q38), ungated (DS) | Q38 and DS copies |
| Routed products and combine | `mul_mat_id`, CUTLASS NVFP4, `llmp.vecq`, compact MMQ, D2R | four combine/reduce kernels; vecq MoE in two builders |
| Per-layer side input | n-gram embedding table with its own convolution state (Q38, layer 1), token-hash routing table (DS) | one each |
| Heads | tied or untied, final softcap (G2, G4), hyper-connection head mix (Q38, DS), selected draft heads (Q38) | per builder |
| Drafters | in-checkpoint MTP over the target's streams (Q38), companion assistant reading target KV and features (Asst), block drafter with feature injection and a Markov head (DSp) | `Inject` and `InjectWave` copies in `dsv4_graph` |
| State | F16 KV and rings (Gemma, Q2, DSp), raw plus compressed caches and compressor state (DS), QSA KV, index and block keys, GDN convolution and recurrent state (Q38), prefix KV (Img) | layout and chunk code nearly identical across `model/gemma{2,3,4}.cc` |
| Image blocks | adaLN, gated residual, 3×3 and 1×1 convolution, channel RMS norm, upsampling, flow-match step (Img) | BF16 rounding points follow diffusers; a component carries a rounding policy |

Constants that become parameters on the way: Gemma3's literal local
rotation base and query scale, Gemma2's literal softcap 50 in its owner
attention, the width whitelist of `ggml.rms_norm_mul_add.fused`, Gemma's
route/reduce shape (128 experts, top-8, width 2816), the gated delta net's
48 heads, DeepSeek's 4096×4 hyper-connection norm and the QSA dimensions.

### The model spec

A model is a spec in its artifact's manifest, written by the importer and
validated by the runtime against the compiled catalog (D-109):

- the **pipeline kind** and its stages (a vision tower feeding a decoder, a
  text encoder feeding a diffusion transformer and a VAE, a target with its
  drafters);
- the **residual stream**: a single stream or hyper-connection streams with
  their count and mixing;
- **side inputs**: per-layer tables (Qwen3.8's n-gram embeddings,
  DeepSeek's token-hash routing), image or audio features;
- the **layer list**: each layer's components and parameters, so per-layer
  variation (local and global attention, linear and full attention, two
  compressed attention kinds alternating) is data;
- **feature taps** that drafters and assistants read;
- **heads**: final norm, output projection (tied or not), logit softcap,
  MTP heads.

Checkpoint approval is a list in data, not a profile in code
([admission](#admitting-a-checkpoint)).

## Weight and state formats

Decided with the owner on 2026-10-08 (D-108).

**Encoding and layout are separate.** The *encoding* is the numeric
content: codes, scales and their formats, block geometry (Q4_K, NVFP4,
MXFP8, EXL3 trellis K=4, BF16). It is preserved, because changing it
changes quality. The *layout* is the byte arrangement: block interleave,
where scales sit, tile swizzle for a matrix unit, alignment. A layout can
be repacked without loss. The artifact names encodings independently of
any kernel library (v0's `ggml`/`exl3`/`plain` families give way), so a
GGUF Q4_K tensor and a repacked Q4_K tensor are one encoding in two
layouts. A model may mix encodings tensor by tensor.

**Resident weights never grow.** Weights are paged in and stay in their
source encoding; nothing keeps a dequantized copy. When hardware lacks a
format, the ladder is:

1. a native kernel for the encoding on that device;
2. on-chip decode (registers and shared memory, tile by tile) to a
   lower-precision path the device supports (INT8, FP8, FP16 matrix units);
3. a lossless relayout at import, only if it is no larger than the source
   (page alignment and the backends' declared over-read excepted);
4. a lossy transcode, only as an opt-in per-alias quality flag;
5. at worst, on-chip decode to BF16, accepting slower execution.

**Decode never expands into device memory.** Decode is bound by memory
bandwidth, so a decode-shaped product reads each weight once at its
compressed size and expands it on-chip. For prefill the rule is whatever is
fastest measured on the Spark: on-chip tile decode is preferred where it is
competitive; a transient expansion into funded workspace (D-052's bounded
reconstruction, GGML's cuBLAS path today) is allowed where it measures
faster. Neither is ever needed for correctness or fit.

**Decoders compose.** Each backend has one small block decoder per encoding
under shared tiling templates (vector, matrix, grouped/indexed), so a new
encoding gets every product shape at a reasonable speed from its decoder
alone, and tuned kernels come where measurements justify them. Fused
kernels take a decoder per input, so mixed per-tensor encodings (Unsloth
dynamic GGUF and similar) fuse instead of falling back; a pattern keyed on
one encoding, such as shared Q8 activation preparation, applies wherever
its operands match.

**Transforms run at import, not at page-in.** Page-in stays a raw direct
read and copy (D-081), so swaps pay nothing for layout. A target-specific
layout is a prepared artifact of its own (same checkpoint, one artifact per
target layout), which D-026 and D-052 anticipated as "alternative layouts".
A relayout during the device copy is only adopted if measured free.

**Quantized state.** KV caches and other state take encodings like weights,
per component. Lossy state encodings are configured per alias, through
presets or the owner's own settings, and are off by default.

**First proof.** Qwen3.8's native NVFP4 and MXFP8 tensors on the discrete
`sm_86` target, which has neither FP4 nor FP8 tensor cores, exercise the
decode path on the workstation long before an Apple or Intel port. They are
checked product by product against CPU references: the whole model (peak
memory about 100 GiB on the Spark) does not fit the 3080 Ti's 12 GiB, and the
discrete target has no on-demand expert paging (D-082).

## The importer

Native C++, part of the distribution; no Python (D-109).

- **Containers by capability**: GGUF, safetensors with `config.json`
  (Hugging Face, including AWQ, GPTQ, ModelOpt NVFP4/MXFP8 and compressed
  tensors), diffusers pipelines (each component an artifact in a D-089
  composition), MLX, EXL3. Each reader normalizes tensors to encodings,
  layouts and shapes and exposes the source metadata.
- **Architectures as data.** Tensor names and some semantics are
  architecture-specific: names differ between families, and some facts
  (Gemma's `(1 + w)` norm, which GGUF conversion folds into the weights and
  Hugging Face checkpoints leave implicit; Qwen3.8's `A = −exp(A_log)`)
  appear in no metadata. Each architecture is a declarative mapping: tensor-name
  patterns to component roles, configuration keys to component parameters,
  and folds chosen from a fixed, compiled catalog. Mappings are data
  compiled into the importer and validated as untrusted input like
  everything else it reads (D-009); a mapping cannot run code.
- **Output** is a prepared artifact whose manifest carries the spec, every
  tensor's encoding and layout, the folds applied, and the importer's
  identity, under a new format version whose identifiers are `llmp-`
  (v0's `jitllm-` names retire with v0's reader, D-111). Import stays
  deterministic and published by one rename (D-056).
- **Admission.** <a id="admitting-a-checkpoint"></a>A checkpoint of a known
  architecture is supported once it passes a standard automated gate:
  logits and perplexity against a reference engine on a fixed corpus, and
  the state, spill and swap controls every model passes. The approved list
  is data. The owner still approves which checkpoints are claimed in
  [model-support.md](model-support.md) (AGENTS.md: agents never invent
  supported combinations).

## Execution

One runner, holding the skeleton as today (weights, live state, planned
shapes, graph runs, prefill lookahead, resources, cohort) plus what every
family re-implemented:

- **Slot and state lifecycle**: spill, restore (prepare and complete),
  clear, idle clear, adopt, place checks, closure refresh, reclaim
  candidates, written-back checks, recovery in place, all derived from the
  components' state contracts. Where today's copies disagree (the Gemma2
  restore guards above), the reconciled behaviour is chosen once and tested.
- **Wave composition**: joins compatible requests by the components'
  batching capabilities, with independent outputs and per-owner state; one
  composer for every model.
- **Prefill orchestration**: chunking, lookahead, capture-ahead, fresh state
  preparation and dependency cuts, once.
- **Decoding modes** (D-068): plain, sampled, scored, and speculative with
  any number of drafters (in-checkpoint MTP, companion assistant, block
  drafters such as DSpark), verify, accept and rollback through the state
  contracts; block diffusion when it arrives. This closes plan.md's "one
  target, at most one drafter" gap.
- **One interface to serving**, per pipeline kind, with a virtual call per
  unit of work at most; nothing virtual per kernel.

Kept state written by today's runners (D-105) is keyed by layout IDs such
as `gemma3-4b-f16-kv-device-v1`. Those IDs are bumped and old kept state
discarded rather than migrated; per-model settings and calibration
(D-103) are re-keyed from family to spec.

## Other pipelines

The same IR, components and runner serve encoders and generators. A
pipeline is a composition of models (D-089) with typed hand-offs: a vision
or audio tower's features into a decoder's side input, a text encoder into a
diffusion transformer, latents into a VAE, codebooks into a speech codec.
Shared blocks (attention variants, norms, GLUs, convolutions, rotations)
are the same components the LLMs use. Qwen-Image moves onto them after the
LLM families; decision models, embeddings and speech arrive as specs.

## Partitioning

Tensor, pipeline and expert parallelism with any number of splits, mixed
(TP=2 × PP=2 and so on), over a configured or discovered topology (D-019,
D-107). Each component declares its split options and their per-step
traffic; the planner chooses the assignment that minimizes cross-device
bytes for the topology, which usually means pipeline cuts between layers,
experts placed whole on a device (expert parallelism) rather than split,
and attention split by heads. Collectives are IR ops with their own
implementations (NCCL over RDMA on the Sparks today). Only two Sparks exist
for testing; the planner and its tests assume any count. M4's two-Spark
swap is the first consumer.

## Platforms

The order after M3.6 and the families: Apple silicon (`mac`), then Intel
Arc (`plex`), AMD when hardware exists; Windows through the NVIDIA host for
the OS port (D-110). A backend supplies the providers, the device runtime,
decoders for every encoding and implementations of the op set; the
primitive fallback plus the decoders run every model before any tuning.
GGML's Metal, SYCL and HIP backends are kernel sources there; llmpalooza's own
CUDA kernels (CUTLASS NVFP4/MXFP8, its attention, its fusions) are not, so
the format ladder is what keeps every model running on a new platform.

## Testing

- **CPU references** per component, run on the workstation or a Spark's CPU
  (the workstation is often busy with measurements), compared with the
  reference implementations' outputs; every backend port checks against
  them.
- **Per-component tests** on the fake backend: state contracts (spill,
  restore, snapshot, rollback), batching joins, partition equivalence.
- **Per-model controls** keep running on the Sparks: quality against the
  same-format oracle, state and swap controls, matched speed.

## Migration and its gate

Each family moves to the new engine with its current implementation as the
baseline:

- **Speed**: prefill and decode, solo and at its qualified cohorts, no more
  than 1% slower than the current implementation at worst, preferably at
  least as fast. This is stricter than D-085's coarse 10% check and needs
  matched bookends that resolve 1%: alternating old and new binaries on the
  same host, repeated until the spread is well under the gap judged, with
  the protocol fixed before the first run.
- **Quality**: equal or better against the same-format oracle and the
  family's existing quality controls. Bit-identity with the old code is not
  required; where op order is unchanged it is expected and its loss is
  investigated.
- **Memory, swap and state**: the existing controls pass unchanged.

The old runner stays beside the new one until the family passes, then is
deleted in the same task, with its serving class and graph builder.

### Order

1. **Format layer** (encodings and layouts in the artifact, decoders,
   the product templates, the ladder) and **shared slot lifecycle, one
   runner interface and one serving adapter**, in parallel: they touch
   disjoint files.
2. **IR and planner**, with the patterns today's fused paths need.
3. **Families, one at a time**: Gemma2 and Gemma3 (simplest; collapses the
   twins and proves the spec), Gemma4 with its assistant, Qwen3.8 (native,
   GGUF, MTP), DeepSeek V4 with DSpark.
4. **The native importer** and the architecture mappings for every migrated
   family, replacing the Python importers.
5. **Qwen-Image** onto the shared blocks.
6. **Structural optimizations** retired from the transfer item, each once
   ([mapping](#retired-transfer-items)).

M3.5's resumed families, formats and pipelines are the first test of the design;
each one's needs are checked against it as M3.6 proceeds.

## Retired transfer items

D-107 retires M3.5's "complete open optimization transfers" item and the
per-family transfer rule. At retirement (2026-10-08, after `cc70d69`) the
item had 128 open recipient cells, matching the
[inventory's matrix](optimization-inventory.md#complete-cross-family-dispositions--2026-10-07)
cell for cell, across 47 techniques. Each technique now lands once, in the
piece of the new engine it belongs to, and reaches every model that uses
that piece:

| Lands in | Techniques |
| --- | --- |
| Format layer: shared input preparation per tensor and encoding, quantized heads | T04, T11, T58, T95 |
| Wave composer: per-owner lanes, joinable drafters, column-invariant groups, joins by view, whole-wave products, image batching, shared HC products | T15, T16, T20, T29, T59, T64, T88 |
| Fusion patterns: HC post, quantized GLU, checked norm/add, grouped KV stores, multi-bank attention roots, routed GLU epilogue, residual add with next norm, rotation into cache stores | T10, T31, T56, T86, T92, and the adoption half of T09, T41, T57 |
| Prefill orchestration and capture: capture ahead, image capture beside eager, joined prefill admission, wide tile admission | T23, T26, T55, and the admission half of T93 |
| State contracts: fresh backing prepared ahead | T67 |
| Decoding-mode orchestration: device verify verdicts, learned heads and adaptive depth, incremental stopping | T39, T45, T81 |
| Components: mask descriptors on the shared producer, liveness pruning of unneeded outputs, drafters through the same planner, shared MoE route worklists, wider owner-root attention variants | T22, T69, T54, T71, T72, and the adoption half of T61 |
| Planner Setup: one placement policy | T96 |
| Node pager (already shared): partial-weight adoption, one cold-switch gate instead of eight cells | T68 |

**Kernel-level work may continue** independently of the refactor, because it
is an implementation behind the existing per-operation contract and survives
the move unchanged: T03, T07, T12, T14, T19, T21, T37, T78, T80, T87, T89,
and the kernel halves of T09, T22, T41, T57, T61 and T93 (their adoption
waits for the pattern or composer above). T53 was a measurement of a
mechanism already shared, and is dropped.

Several *completed* transfers are also per-family copies that collapse in
the refactor: the mask-producer adapters in each `*_plan.cc` (T22/T69), the
lookahead wiring in each runner (T23/T70), per-runner greedy publication
hooks (T27), state-only cuts written into each graph builder (T54),
per-runner opt-ins for grouped KV stores (T86), actual-root K/V in the
Gemma2 and Gemma3 builders (T92), and per-runner placement thresholds
(T96).

## Survey of other engines

Read from current sources on 2026-10-08; facts first, then what llmpalooza
takes.

- **GGML/llama.cpp** (`master` `71ad059`). `ggml_tensor` carries type,
  `ne`/`nb` with block padding, an op, 64 bytes of op parameters, up to ten
  sources and views through `view_src`
  ([ggml.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h));
  the op set has about 100 ops plus unary and GLU sub-ops, and has started
  adding model-specific ones (gated delta net, the lightning indexer,
  DeepSeek V4's hyper-connections). Fusion is a pattern match over
  consecutive nodes, and CUDA's gate/up fusion requires both weights to
  share one type, so differently quantized tensors do not fuse
  ([ggml-cuda.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu)).
  Matrix products pick cuBLAS, MMVF, MMF, MMVQ or MMQ by type, compute
  capability and batch. MXFP4 and NVFP4 decode in software on the int8
  path unless activations may be quantized to 4 bits on Blackwell. Every
  architecture is still a C++ class with its own graph builder
  (`src/models/`, about 160 files) plus a Python tensor-name mapping and
  converter; no declarative effort exists. Quantized KV covers q8_0, q4_0,
  q4_1, iq4_nl, q5_0 and q5_1 and needs flash attention. Tensor split mode
  is experimental (NCCL, allow-listed architectures); there is no expert
  parallel mode. *Take:* the tensor and view model, the op-set discipline,
  MMVQ/MMQ as decoder-plus-template kernels. *Avoid:* model-specific ops in
  the core op set, fusion keyed on identical operand types, per-architecture
  graph code.
- **MLX and mlx-lm** (v0.32.3). Primitives implement CPU and GPU
  evaluation; `mx.compile` fuses only elementwise chains, never products or
  attention ([compile.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/compile.cpp)).
  Quantization modes: affine (2–8 bits), MXFP4, MXFP8 and NVFP4; on Metal
  FP4 and FP8 decode in the shader to half or BF16 in threadgroup memory
  ([fp_quantized.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/fp_quantized.h)),
  which is D-108's ladder step 2 on Apple. Quantized KV caches are read
  directly by quantized products. About 125 architectures are Python
  files; sharding is per model (`shard_linear`, a pipeline mixin) over
  ring, MPI, NCCL or Thunderbolt RDMA. Apple documents M5's neural
  accelerators as FP16/FP32 and INT8/INT4 today, FP8/FP4 in macOS 27,
  without saying which the hardware accelerates
  ([WWDC26](https://developer.apple.com/videos/play/wwdc2026/330/)).
  *Take:* software FP4/FP8 decode on Metal as the Apple baseline.
- **vLLM and SGLang.** vLLM splits a traced graph at attention, compiles the
  pieces per token count, and fuses norm, activation and quantization in
  compiler passes ([fusions](https://docs.vllm.ai/en/latest/design/fusions.html));
  CUDA graphs come in full and piecewise modes chosen per attention backend.
  Each format has an ordered list of kernels (Marlin, Machete, CUTLASS,
  FlashInfer…) with an override per scheme
  ([linear kernels](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/__init__.py));
  Marlin runs FP4 weight-only with 16-bit activations on GPUs without FP4
  hardware. compressed-tensors matches quantization schemes per module by
  path, class or regex. About 280 model files plus a Transformers fallback;
  GGUF moved to an out-of-tree plugin. Expert parallelism is EP = TP × DP
  with load balancing; for several nodes vLLM advises TP inside a node and
  PP across nodes without NVLink. SGLang's breakable CUDA graphs run marked
  functions eagerly between captured segments, the prefill default.
  *Take:* the per-encoding ordered implementation list (D-053 already has
  it), per-module scheme matching for mixed formats, segmented capture,
  PP-across-nodes as the partitioner's starting assumption.
- **ExLlamaV3** (v1.6.0). Procedural trellis codebooks decode in registers;
  architectures are modular `Module` subclasses with load, forward and
  tensor-parallel export; cache layers cover FP16, quantized, MLA, DSA and
  recurrent state. *Take:* module-level TP export, cache types as state
  contracts.
- **TensorRT-LLM** (1.3.0rc29) removed its TensorRT engine backend and,
  on 2026-09-24, its AutoDeploy pattern-matcher path; PyTorch model files
  remain. Its mapping covers TP, PP, context parallelism, MoE TP/EP and
  attention data parallelism. *Take:* the mapping's vocabulary; a
  graph-rewrite deployment path is not established practice to copy.
- **Hardware for low-precision encodings.** The GB10 (sm_121) has
  block-scaled MMA for MXFP8/6/4 and NVFP4; Ampere (sm_86) multiplies
  F16, BF16, INT8 and INT4 only, and FP8 needs sm_89
  ([PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)).
  Metal 4.1's `matmul2d` adds FP4 E2M1 and FP8 operands and MXFP4 tensors
  with E8M0 scales only, so NVFP4's E4M3 scales are not native there; the
  spec does not say whether M5 runs these natively
  ([MSL 4.1](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)).
  Intel's Xe2 (the B50) is native for BF16, F16 and INT8, and converts FP8
  and FP4 to higher precision
  ([oneDNN](https://uxlfoundation.github.io/oneDNN/dev_guide_data_types.html)).
  AMD RDNA4 has FP8 but no FP4 matrix support; CDNA4 has scaled F8F6F4.
  Engines decode without hardware support: vLLM's Marlin turns FP4 and FP8
  into FP16/BF16 for weight-only products; llama.cpp maps MXFP4 and NVFP4
  through a 16-entry int8 table against Q8 activations (W4A8) before
  Blackwell. So every port beyond the GB10 depends on ladder step 2, and
  NVFP4 is the encoding most often decoded in software.
- **Others.** MLC-LLM/TVM keep a two-level IR (Relax, TensorIR) driven from
  Python; candle and mistral.rs are Rust engines with per-model code.

None of the surveyed engines builds models from declarative specs; each
writes per-architecture code, and llama.cpp and vLLM have each grown to 150–280
architecture files. Llmpalooza's spec-plus-components design is the point of
difference, and its risk: the catalog must stay expressive enough that a
new shape is a component, not an escape hatch.

## Needs measurement

Open questions this design does not settle by argument:

- The 1% protocol's repeat count and spread on each family's controls.
- Where on-chip tile decode matches transient expansion for prefill, per
  encoding and row count, on the GB10.
- Whether a relayout during the device copy is free enough to avoid
  per-target artifacts.
- Communication cost of each split option over the Sparks' link, for the
  partitioner's model.
