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

**As built** (`src/ir`, `kernels/reference/exec.h`). Op set v1's
semantics are written in `ir/op.h`: GGML's at the pin wherever GGML has
the op (shapes, broadcasting, `nb` strides and views), and the cited model
definitions for the composites. A node has up to ten sources and four
outputs (routing's ids and weights, the gated delta rule's attention and
state, grouped stores); a store's output is a view of its cache, and node
order is execution order. Shape rules are one function the builder makes
outputs from and the verifier checks recorded outputs against; the
verifier also holds GGML's int32 extents and strides (RE-037). The digest
names leaves by what they bind and excludes names, so declaration order and
names do not change a plan key; node order does. Every fused kind and
composite op has a primitive decomposition (`ir/decompose.h`, `Expand`),
and the reference executor runs every op over host buffers in their
formats, fused kinds through their decompositions, collectives over
simulated devices. Where GGML's kernels differ from the definition (a
fully masked row is 0, not NaN; top-k and argsort order among ties),
`op.h` says so, and a port's comparison allows for it.

Op-set changes after the first review (2026-10-09; still version 1, as
nothing has been released): products (`kMatMul`, `kMatMulId`) carry
`ProductParams`, the accumulation (`kDefault`, or `kF32`: GGML's
`ggml_prec_set_acc`) and a hint (`kHadamard`: GGML's
`GGML_HINT_SRC0_IS_HADAMARD`, a square matrix only); attention's
parameters add the same accumulation, `n_kv_max` (a promise on the finite
cells of every mask row, which needs the mask) and `sparse_any` (which
needs `n_kv_max`). These are hashed, so a lowering reproduces production's
graphs node for node, and the reference executor ignores them. A RoPE
angle is defined as GGML's CUDA kernel forms it, each step rounded to F32
(`theta_scale`, its power, the position times it, the frequency factor,
the YaRN mix and magnitude); only the rotation by that angle is exact, and
a port allows for the device's fast-math `powf`, `sinf`, `cosf` and FMA
contraction (a test pins the F32 angles near position 2^20, where they are
0.004 to 0.1 radians from the exact ones). `fused.mul_mat_glu` takes
(gate weight, up weight, input), as GGML's `ggml_glu_split(gate, up)` over
its two products and the decode declarations do, and
`fused.prepared_products` takes its weights first too (w0, …, input). `fused.moe_combine`'s
shared-expert gate is optional (DeepSeek V4's shared expert has none).
A routing scale is the factor itself and above zero: a checkpoint's unset
0 is written 1. Every composite op now decomposes for every shape the
verifier admits: pooling over several matrices along dim 3, and the gated
delta rule where each q and k sequence serves several value sequences;
the latter stores each token's output at its own row of one buffer and
gathers them back (copying linear in the tokens), so any permutation from
its argsort serves as the rows.

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

**As built** (`src/components`, 2026-10-09). `component.h` holds the
compiled catalog (`Catalog`, `Find`): a `ComponentDef` per kind and
version with its parameters (typed, ranged, defaulted; `Params`), weight
roles with the encodings each accepts and, per instance, the roles in use
and their shapes; a per-instance `model::StateContract`; its ops' batch
classes and its own `model::Batching` limits; split options with the
bytes a split sends per step; the pattern names it takes part in; the
masks it reads; `emit`, its primitive IR fragment for a step; and
`reference`, a scalar CPU implementation written from the op set's
definitions, independent of `emit`. `step.h` makes a spec usable:
`ValidateSpec` (the spec's own rules, then every component a known kind
and version with parameters in range, every role it uses bound once to a
tensor of the bound artifact in an accepted encoding and the implied
shape, no binding of an unused role, the catalog's width and layer
bounds, one stream, reads of state naming a state tensor);
`StageLayout` (a layer's contract is its components' contracts in order),
`StageBatching`, and `BuildStep`, which asks each component for its
fragment for a `planner::StepShape` and returns the verified graph with
its interface: the step's inputs (tokens, positions, the head's rows, each
owner's store cells and its read cells' positions; `InputValues` computes
them), weight leaves (one per bound tensor, so a tied head reads the
embedding's leaf) and state leaves (stage, the layer's contract tensor,
layer, owner). Masks are the framework's: per owner, before the
embedding, causal first, `kMaskBuild` over a view of the positions and the
positions the read cells hold once the step's rows are stored, with
Pad(rows, 32) rows. A state-only step (no outputs) emits only the last
layer's state writes. The catalog holds what Gemma 2 and Gemma 3 use:
`embedding.token` (rows of a table, scaled), `norm.rms`,
`attention.dense` (pre or sandwich norms, Q/K/V products, optional
per-head Q and K norms, NEOX or normal rotation at the instance's base and
frequency scale, queries scaled after the rotation and attention at scale
1 with F32 accumulation, an optional logit softcap, a ring for a sliding
window and a cell per position otherwise, F16 K and V), `ffn.glu` (pre or
sandwich, GELU-tanh or SiLU) and `head.linear` (the output rows, the
product, the stage's final softcap as scale, tanh, scale, and the
host-greedy argmax on a greedy step).

Each fragment records today's builders' nodes in their execution order
(GGML's `GraphOrder` over their outputs: V's product before K's, the gate
before the up product, each owner's K and V stores before its attention,
the owners' outputs joined after the last one's attention), with the same
types, extents, view strides and arithmetic parameters; a CPU test
compares the Gemma 2 and Gemma 3 steps with `BuildGemma2Graph` and
`BuildGemma3Graph` node for node, with host and device masks: every
node's sources, the leaves among them by what they are (the weight
resource, the layer's, owner's and tensor's cache, the step input), and
each mask's window (and the device producer's rows, capacity and
context). `BuildStep` refuses a ring that wraps with fewer than rows +
window − 1 cells (its stores precede attention, so a smaller ring would
overwrite positions its first rows still see). What a
fragment does not express, for the lowering and the patterns:

- **Masks.** `kMaskBuild` reads the cells' positions as an input; GGML's
  producer (`llmp.gemma4.mask`) derives them from the ring's capacity,
  the window and the context limit, which the lowering takes from the
  step's input table (`StepInput::cells`) and the layout. A host-mask
  build evaluates the producer and its positions view on the host. The
  32-row mask padding is GGML's flash-attention tile (`GGML_KQ_MASK_PAD`).
- **Attention reads its cache leaf**, not the store's output: the stores
  precede it in node order, as the builders' expanded list orders them.
- **Weight over-read** (`MarkRowPaddingReadable`) is the format layer's
  readable rule, not a graph property.
- **The builders' options** are patterns and selection (slice G1b), not
  fragments: owner decode and owner, packed and flexible prefill with
  bounded roots, shared Q8 preparation, grouped KV stores, the fused norm
  forms and device masks; the diagnostic scopes (`first_layer`,
  `layer_count`, `hidden_input`, `narrow_final`) are not ported. The
  owner-attention kernel's literal softcap 50 is an implementation's
  limit the pattern checks against the instance's `softcap`.
- **Batching.** The primitive graph joins any rows of up to 16 owners
  (attention per owner), so the components declare no limits of their
  own; the owner-root carriers' group sizes and equal-row prefill come
  with their patterns.

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

**As built** (`src/spec`, 2026-10-09). `spec.h` holds `ModelSpec`: spec
version 1, the pipeline kind, the folds applied and the stages; a stage
has its role (target, drafter, encoder, text encoder, denoiser, VAE), its
artifact index in the bound list, its trained context, an embedding, the
residual stream (stream count, width, and a mixing component exactly when
there are several streams), side inputs with the layers that read them,
layers of components in execution order, feature taps (a layer's output,
its streams, or the final norm's output), heads (final norm, output,
softcap, MTP heads) and reads of another stage's tap or borrowed state.
A component reference is a kind and version, which only the catalog
interprets, with typed parameters (integers, reals, flags, names and
lists of integers or reals) and weight bindings (role, artifact tensor
name, part, whether an expert array). Its JSON is the canonical subset of
the v0 manifest's JSON, so parsing reuses the artifact reader and refuses
anything non-canonical, unknown or duplicate keys, out-of-range numbers,
unsorted lists, unknown folds and dangling taps and reads; reals are
strings in their shortest round-trip form, finite and never -0 (which
would be a second encoding of 0). Its digest is the SHA-256 of
those bytes. Kept-state layout IDs and plan keys are to derive from
`BoundDigest`: the spec digest with the length-prefixed IDs of the
artifacts its stages bind (of a spec `Validate` accepts), since another
checkpoint of the same spec shares the layout but not the state. The spec nests eight levels, the v0 JSON's limit: a v1
manifest that holds it as a member raises that limit or keeps it as a
document of its own.

`mapping.h` makes specs from architecture mappings: compiled tables of
slots (component kinds at the embedding, the layers of some types, the
final norm and the output), tensor-name patterns with `{layer}` (and
fallbacks: Gemma's tied output binds the embedding), parameter rules from
source keys with a fixed set of transforms (reciprocal, square root,
reciprocal square root) and constants, tried in order per layer type,
requirements on source values, and folds from a fixed catalog
(`norm.plus_one`, `ssm.a_neg_exp`, `gdn.value_heads_tiled`) asserted per
source container. Mappings are validated like untrusted input.
Synthesis binds every source tensor to exactly one rule and every
required role to a tensor, and resolves every parameter. For a v0
artifact the tensors are its resources' roles and expert arrays and the
values come from today's compiled profiles under the GGUF keys a mapping
reads (`mappings/gemma2.h`, `gemma3.h`), so no new file is parsed; the
importer will apply the same mappings to checkpoints. Gemma 2 and
Gemma 3 have mappings: `attention.dense` and `ffn.glu` (sandwich norms,
GQA, softcap, sliding windows, NEOX rotation, GELU-tanh GLU),
`embedding.token` scaled by sqrt(width), `norm.rms` and `head.linear`;
their specs from the installed artifacts equal those from the tensor
tables in `tests/unit/data`. They also read the optional keys llama.cpp
reads (a sliding-window rotation base, Gemma 3's final softcap) and
refuse a layer pattern other than theirs; neither can yet express the
27B query scale (1 / sqrt(width / heads), which llama.cpp picks by layer
count), so the importer must not apply them to a 27B checkpoint. The catalog that gives these kinds meaning
is the component framework's.

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

**Where it stands (2026-10-08).** `src/format` names the encodings and
layouts; `kernels/reference` decodes every format on the CPU;
`kernels/decode` holds CUDA's block decoders (BF16, F16, Q8_0; NVFP4
split, in CUTLASS's scale atoms, row-interleaved and in GGML's blocks;
MXFP8 split and in scale atoms) under the vector (1 to 16 rows), GLU
(a decoder per input, mixed pairs instantiated where used), matrix
(mma.sync BF16/F16, or F32 FMA), grouped (an expert worklist) and rows
templates, each declared to the registry at rung 2 (rung 1 for BF16 and
F16) with no expansion. The first proof passes on the 3080 Ti: every
product shape of Qwen3.8's layers 1 and 3, 30 of layer 1's routed experts
in both imports' layouts (bit-identical to each other), the n-gram rows
and the head, within bounds derived from an F64 product of the reference
values, with no run keeping device memory beyond the compressed weights
and its operands (`qwen38_decode_proof_test`); no product allocates or
takes a workspace, each capturing into a graph as one kernel node
(`decode_gpu_test`). The templates are not wired into a
model; their speed against the tuned kernels is in
[m36-decode-templates](experiments/m36-decode-templates/README.md).
v0's CUTLASS import keeps gate's and up's per-expert tensor scales as two
resources, while a stacked format has one tensor-scale part ordered
[matrix][stacked]: a launch gathers them until the importer writes that
part, which the v1 manifest should hold as such.

**The EXL3 trellis (2026-10-09).** `kernels/reference` decodes the
trellis as ExLlamaV3's pinned kernels define it: `exl3_dq.cuh`'s 16-bit
states over each tile's circular bit stream, `codebook.cuh`'s 3inst, mcg
and mul1 values, `reconstruct.cu`'s placement of a tile's positions, and
the 128-point Hadamard transforms with the `suh`/`svh` vectors, at every
integer rate from 1 to 8 (half-integer rates have no format yet). It gives
the rotated weights exactly as `reconstruct_kernel` writes them, and the
original-basis weights computed in F64 and rounded once to F32 (exact for
3inst and mcg); upstream's own transforms round to F16 along the way, so
those agree within that arithmetic. The EXL3 module declares its paths'
capabilities from their host checks: the GEMV (K = 4, up to 8 rows), the
packed GEMM and the multi-GEMM at rung 1, and the two reconstruction paths
as transient expansions, selectable only on the GB10 (where their cuBLASLt
algorithms were pinned) and never for a decode-shaped product. For those
two a candidate is necessary, not sufficient: each slice's GEMM runs only
with an algorithm pinned for exactly its shape, and `exl3-recon-pin.json`
pins only the fixtures' shapes at 145, 1,023 and 1,024 rows, so a planner
drops a reconstruction candidate where its tuning data holds no pin. A
capability query's operand formats now carry the encoding's parameters,
which the EXL3 predicates read (mcg at K = 4, 5, 6 or 8 is what is
compiled). On spark-b (`exl3_trellis_test`), `reconstruct_kernel` equals
the reference bit for bit on every linear of both Qwen2.5 EXL3 fixtures
(169 each, the head's first and last 1,024 columns) and on random
trellises at K = 4, 5, 6 and 8; `reconstruct_had_kernel` is within 1.0e-3
relative L2 of the reference's F32 weights; and layer 0's q, gate and down
products at 1, 8, 145 and 1,024 rows through every candidate, and gate
with up through the multi-GEMM, are within 1.2e-3 relative RMS of an F64
product of the reference's weights (6.1e-4 on the paths that keep the
weights exact). The trellis has no
block decoder in `kernels/decode`: on CUDA its rung-1 kernels are
ExLlamaV3's, and another backend needs a trellis decoder of its own.

**State encodings (the format half).** `format::StateEncodings()` lists
the encodings the format layer admits for a state tensor, each of which
the CPU reference encodes and decodes: F32, F16, BF16 and the blocks
GGML's attention reads at the pin, Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0.
Membership is not a device claim: a state may use an encoding only where
its plan finds an attention that reads it and a cache store that writes
it, which today means F16 or F32 everywhere and, on CUDA, Q8_0 or Q4_0 at
head sizes 64 and 256 (below). Whether one is
exact for a state is `format::Lossless(the contract's encoding, the stored
one)`: an F16 KV cache stores exactly as F16 or F32 and lossily as BF16 or
any block. The CPU reference encodes and decodes each (`EncodeRows`
through GGML's reference quantizers, `DecodeRows`), so the reference
executor's cache stores can write them.

**Lossy state encodings, the kernel half (2026-10-09).**
`src/kernels/ggml/kv_quant.h` reads and writes block-quantized KV caches,
declared with capabilities but selected by no model yet:
- *Attention.* GGML's vector kernel at the pin, compiled for K and V both
  Q8_0 or both Q4_0 at D 64 and 256 only (`fattn_vec_quant.cu`, four
  `ggml_cuda_flash_attn_ext_vec_case` instances with GGML's device flags),
  as `ggml.flash_attn_ext.vec_q8_0` and `_q4_0`: Q F32, K and V in GGML's
  blocks, an F16 mask; GQA of any whole ratio, softcap at D 256, no sinks
  or ALiBi. The kernel reads the blocks directly at every number of query
  rows (one, or two to a CUDA block), quantizing Q to Q8_1 in shared
  memory for the K product; so the capability takes any rows at rung 1
  with no expansion. The earlier note here had the conversion wrong: it
  belongs to GGML's tensor-core and tile kernels, which upstream's
  selector takes beyond one or two rows on Ada and later, and at this pin
  it converts K and V to F16 beside the output (the node's extra
  allocation, `ggml_cuda_flash_attn_ext_get_alloc_size`), not in pool
  scratch. That convert-then-tensor-core path is a transient expansion
  (D-108: prefill only, where measured faster) and is not built; how much
  slower the vector kernel prefills than the F16 tensor-core kernels is
  not measured. D 64 serves Qwen2 and D 256 Gemma 2 and 3, Gemma 4's
  local layers and Qwen3.8's head size; no current model's K or V has
  D 128, and upstream has no quantized path above 256, so D 512 (Gemma
  4's global layers, DeepSeek V4, DSpark) still needs a decoder under a
  shared attention kernel. On spark-b the unit compiles in 19.0 s alone
  (`fattn.cu`'s two F16 cases: 7.5 s) into a 1.77 MB object, and with the
  stores below `llmp-runtime` grows from 226.3 to 228.3 MB (sm_121a SASS
  only; the workstation's profile adds sm_86's).
- *Quantizing stores.* `llmp.set_rows.quant` (ggml_set_rows's shapes, I32
  or I64 indices), `llmp.set_rows.grouped_quant` (2 to 16 stores in one
  launch) and `llmp.rope_set_rows.quant` (NEOX RoPE's rotated rows stored
  so, `ggml.rope_set_rows.fused`'s contract) quantize each block of 32
  with the reference quantizers' arithmetic written as IEEE intrinsics
  (`kv_quant.cuh`), so the plain and grouped stores equal `EncodeRows`
  byte for byte by construction and in the tests; the RoPE store's
  rotation copies `rope_neox` under GGML's device flags and equals GGML's
  RoPE followed by the plain store byte for byte. The exceptions are
  blocks below F32's normal range: the reference's reciprocal scale can
  overflow there (its integer conversion of the product is undefined in
  C), and under `-ftz` the RoPE store flushes subnormal values and
  scales; every block that differs so stores an F16 zero scale on both
  sides and decodes to zeros.
  Upstream's own quantizing store builds with `-use_fast_math`, so its
  agreement is not guaranteed; on the test's corpus it agreed on all
  132,864 blocks. The F16 and F32 stores are unchanged.
- *Contracts.* `model::CellBytes` gives a cell's bytes by the size rule
  (a Q8_0 cell of four heads of 256 is 1,088 bytes, Q4_0 576), and
  `Layout` refuses a cell of partial blocks, so every used, chunk and
  checkpoint range starts at a block; `KeptLayoutId` already hashes each
  tensor's encoding, so kept state of another encoding is never adopted.
- *Planning.* `PlanGraph` gives a quantized destination the quantizing
  store (grouped, and fused with its RoPE, under the F16 forms' options)
  and quantized K the vector attention named by its type.
- *Evidence.* `kv_quant_gpu_test` on spark-b: the stores byte for byte
  against `EncodeRows` (decode, prefill and broadcast shapes, both index
  types, blocks at the quantizers' rounding boundaries), the RoPE store
  with frequency factors and YaRN, blocks below F32's normal range (on
  that corpus 14 of 56 Q8_0 and 13 of 56 Q4_0 RoPE-store blocks differ,
  each with zero scales; no plain-store block), and the attention at 1, 2,
  3, 17, 33, 1,024 and 1,025 rows (the mask pre-pass), without GQA and at
  GQA 2, 7 and 8, with and without softcap, against the reference
  executor over the same bytes: NMSE 2.3e-5 to 4.4e-5 for Q8_0 and
  8.7e-5 to 2.4e-4 for Q4_0, within upstream's 5e-4 for flash attention
  (the kernel's Q8_1 Q and F16 V accumulation). The builder's six tests
  (before the below-normal-range test and the GQA-1 and 1,025-row cases
  were added) pass on the workstation's RTX 3080 Ti (sm_86).

What each model's opt-in still needs, besides the per-alias setting below
and its long-context quality report: Qwen2 (D 64) its runner's cache
stores and forced vector attention renamed to the quantized forms; Gemma
2 and 3 (D 256) the same for decode, and for prefill and joined cohorts
either the vector kernel (the owner and tensor-core kernels take F16 only)
or the measured conversion path; Gemma 4 and its assistant (which reads
the target's cache) the same for the local layers, and the global layers'
D 512 kept F16 (a per-component encoding) until a quantized D 512 path
exists; Qwen3.8 its QSA kernels (`llmp.qsa.attn`) and tensor-core
attention reading blocks; DeepSeek V4 and DSpark a D 512 path and their
compressed and indexer caches' readers.

**The per-alias setting (not added yet).** A lossy state encoding is a
model setting under `[models.NAME]` (D-096), one key per kind of state a
component's contract names (the KV cache first), resolved through D-103's
layers: the derived default is each contract's own exact encoding;
calibration never picks a lossy one, since it trades quality rather than
speed; an override or a preset may name any encoding the contract admits
from `StateEncodings()` for which the backend declares an attention and a
cache store, the schema refusing others, and the `settings`
command reports it with its source and whether it is lossy. Off by
default. It changes the kept-state layout ID, so kept state of another
encoding is discarded rather than adopted. Adding the key is a public
configuration change and takes a decisions entry with the item.

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

The fixed protocol, the frozen baseline binary and every family's
baseline speed, quality, memory and swap rows are in
[m36-baselines](experiments/m36-baselines/README.md); its harness is
`tools/speed_gate.py`.

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

- The 1% protocol's repeat count and spread on each family's controls:
  measured ([m36-baselines](experiments/m36-baselines/README.md#same-binary-spread)).
- Where on-chip tile decode matches transient expansion for prefill, per
  encoding and row count, on the GB10.
- Whether a relayout during the device copy is free enough to avoid
  per-target artifacts.
- Communication cost of each split option over the Sparks' link, for the
  partitioner's model.
