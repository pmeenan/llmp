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
state); a store's output is a view of its cache, and node
order is execution order. Shape rules are one function the builder makes
outputs from and the verifier checks recorded outputs against; the
verifier also holds GGML's int32 extents and strides (RE-037). The digest
names leaves by what they bind and excludes names, so declaration order and
names do not change a plan key; node order does. Every fused kind (but
`grouped_stores`, a batch key, below) and composite op has a primitive
decomposition (`ir/decompose.h`, `Expand`),
and the reference executor runs every op over host buffers in their
formats, fused kinds through their decompositions, collectives over
simulated devices. Where GGML's kernels differ from the definition (a
fully masked row is 0, not NaN; top-k and argsort order among ties),
`op.h` says so, and a port's comparison allows for it.

Exact selection has separate primitives (G4b2b2a): `VisibleTopK` reads
F32 scores and an I32 visible-prefix count per row, clamps the count, ranks
numeric values above NaNs with lower-cell ties, and returns ascending
selected IDs followed by -1 padding. `SelectedMask` copies raw F16 mask
bits, fills padding and compressed cells with literal -Inf, and sets each
valid selected compressed cell to +0. Duplicate IDs are idempotent; every
other integer is ignored. These definitions cover all input bit patterns
without changing ordinary `IndexScore` or `TopK`. Their scalar executors
are independent sorting and mask construction; the native selection shares
the existing radix routine, and the native mask uses a duplicate-safe
membership bitset with one writer per output half. The standalone forms
do not establish a score recipe, a Lid fusion or a new component version.

`RoundedIndexScore` is a separate mathematical composite: explicit half-query
rounding, head-weighted ReLU dots (a NaN dot becomes zero), and literal -Inf
outside an I32 visible prefix. Its independent scalar uses wider sums. The
portable graph funds both F32→F16→F32 query conversions, ordinary products,
head weighting/reduction and `PrefixOverwrite`. That exact primitive copies
visible F32 representations and writes literal fill bits outside the prefix,
including for NaN/Inf inputs. Per-node F32 intermediate rounding can differ
from the direct wider formula; exact selection consumes whichever scores its
numerical implementation produced. The optional native score implementation
uses the existing Lid score kernel and its original group, row-block and key-tile
choices. A paid full-output literal -Inf fill precedes its visible score writes;
all declared key rows remain in the read closure even when the visible count is
zero. The native recipe retains half query conversion, the device MMA reduction,
and the compiled head reduction/FMA order. Its D-053 numerical relation to the
wider scalar and ordinary products does not promise identical near-cutoff CPU
selection; native standalone/fused equivalence is checked against a retained
original producer. No whole-indexer fusion is registered. `IndexScore`, `TopK` and default
mapping policies retain their definitions; the opt-in component consumer below
uses the separately declared rounded recipe.

Native score admission uses optional operand strides, root-relative offsets and
explicit root-allocation alignment guarantees. `GraphRootAlignments` owns those
facts, the canonical graph digest and an exact numeric-root-to-semantic-binding
seal: canonical digest alone deliberately ignores leaf declaration order.
That exact-graph helper remains conservative across rewrites. The planner now
forwards external leaf facts through authenticated per-pass tensor maps and
semantic bindings, including pattern rounds and repeated decomposition. Equal
shapes or canonical digests alone cannot lend an old root ID another allocation's
promise. Hypothetical descriptors and absent facts remain unknown.

The provider separately declares physical reservation-base alignment (unknown by
default), as distinct from backing/offset granularity. Component execution passes
the activation reservation's guarantee and exact state-root guarantees derived
from the owned or borrowed region and item offset. Independently placed roots
and actual staged inputs get the arena/placement guarantee; contained outputs
stay unknown without a checked inner-offset contract. Commit-row inputs bind to
saved buffers and do not inherit the arena promise, even while their legacy
placement reservation remains. Final placement and actual binding recheck these
promises; residency and full operand coverage remain independent requirements.
The default-off `rounded_indexer` policy now lets the bounded target component
emit `RoundedIndexScore` → `VisibleTopK` → `SelectedMask`. Its one full mask
preserves the raw window bits and adds only the compressed IDs actually selected
from this candidate's scores, in the same raw/compressed order as the key bank.
`ComponentStepSource` requires the real owned backend, prepared readiness and a
complete single-target numerical domain before emission; actual mode/shape and
cached-use checks enforce one owner, ordinary phases, at most eight rows and
context 4–4096. The policy participates in existing source/plan/state identity.
Default component emission remains unchanged.

The [bounded consumer evidence](experiments/m36-dsv4-rounded-consumer/README.md)
separates real component/source admission host controls from a small declared
backend GPU corridor with changed-input capture replay. A bounded synthetic
mapped attention prefix also passes actual `ComponentStepSource` construction,
its literal owned lower callback, eager execution and retained replay: 320 finite
output values and 376832 initialized state bytes match same-implementation
results. Full model admission, full-mask fusion and family quality, state and
speed qualification remain open.

`StepContext::VisibleCells` declares an ordinary staged I32 input for each
owner's completed positions-basis blocks. Counts use the owner's past and
local row, block ratio, scored width and explicit real-cell capacity;
the latter excludes padding and dummy storage from `CellKey.capacity`.
Only zero-based, non-ring, non-frozen positions blocks are accepted.
The runner refreshes these bytes on every execution, including graph
replay; visibility does not reduce the state backing required by readers.

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
`kExpertReduce`'s scale may be one already gathered per slot [1, U, T]
(slice G2b: llama.cpp gathers the experts' scales with get_rows before
its products, and GGML's fused Gemma reduce reads them so), the ids then
only naming each slot's expert for expert parallelism (which masks such a
scale with its device's experts gathered by the ids); a scale per expert
[E] still needs the ids. The decompositions of `kRoute` and
`kExpertReduce` are llama.cpp's `build_moe_ffn` sequences in its order
(the probabilities' rows before the sort; every slot's view before the
ordered adds onto the first, without a copy), so a plan that runs them as
primitives launches today's GGML nodes.
A routing scale is the factor itself and above zero: a checkpoint's unset
0 is written 1. Every composite op now decomposes for every shape the
verifier admits: pooling over several matrices along dim 3, and the gated
delta rule where each q and k sequence serves several value sequences;
the latter stores each token's output at its own row of one buffer and
gathers them back (copying linear in the tokens), so any permutation from
its argsort serves as the rows. `fused.owner_attention` (amended with the
planner, 2026-10-09) lays its owners on dim 3: queries [D, R, H, owners],
a mask matrix an owner [C, >= R, 1, owners] read through its first C_o
cells and R rows, and the result [Dv, H, R, owners]. Its first form put
the owners' rows one after another on dim 1, with mask rows at o · R;
but each owner's mask (the framework's `kMaskBuild`, or GGML's producer)
pads its rows to 32 on its own, which that layout cannot hold without a
copy of every mask a layer, and GGML's owner carriers (`fattn_owner.h`:
Q [D, 1, heads, N], masks [cells, 32, 1, N]) and the builders' joined
masks (padded to a common width and concatenated on dim 3) are this
layout already, so the lowering binds views and copies nothing. Its
parameters gained the node's cohort and offset (slice G1b, 2026-10-09; a
reviewed op-set change, still version 1): a step joining more than four
owners runs as several nodes of up to four (GGML's four-root carriers),
each naming every joined owner (the cohort, up to 64) and its first
owner's place among them, which an implementation partitions its grid by
and the reference executor ignores; 0 and 0 is a lone node.
DeepSeek V4 (slice G4a, 2026-10-10; still version 1) adds a mask kind
and lets a hash route take its ids already looked up: `MaskKind::kAhead`
(cell c seen from position p where c > p or p − c < window, a window of
0 every earlier cell: DSpark's block drafter, whose rows see their whole
block and the window before it; GGML's `MaskPolicy::kBlock` producer),
beside `kBlock`, which keeps its meaning; and `kRoute` `kHash` without
tokens, its table then the rows' ids I32 [used, rows] (a token-hash side
input's), which decomposes to no lookup. The builder's mask validation
now requires a window for `kWindow`, a block for `kBlock`, and no
window for `kCausal` or `kBlock`.

**The planner, as built** (`src/planner`, 2026-10-09). `Plan(primitive
graph, registry, device facts, policy, options)` gives a `StepPlan`: the
rewritten graph, its steps (`Selected`: an implementation over one node
or a batch of consecutive ones, with the capability's group, which the
lowering resolves inside its module), the lanes' regions, the activation
placement, the rewrites applied and the maps from the primitive graph's
nodes and tensors (the step's inputs, weights and state keep their
primitive ids). It is deterministic. The steps:

- **Patterns** (`pattern.h`): one fixed table whose order is priority,
  each pattern applied over the whole graph before the next. A rewrite
  matches by dataflow (any component's nodes; every intermediate read
  once, by the match, and neither returned nor kept) and only where the
  registry has a candidate for its result on the device under the policy,
  and emits the replacement at the match's last node. The table:
  `rms_norm_mul_rope`, `rms_norm_mul_add`, `mul_mat_glu`, `rope_store`,
  `rms_norm_mul`, `owner_attention` (below), `prepared_inputs` (two or
  more products of one F32 input share one Q8_1 `kPrepare`, each product
  reading it: the builders' shared Q8, kept as primitives in the
  builders' order rather than `fused.prepared_products`, which would
  move the products to one place;
  placement keeps the prepared input's source live with it, as GGML's
  prepared product reads both) and the batch `grouped_stores`
  (consecutive independent cache stores under one implementation of the
  `fused.grouped_stores` key whose capability declares a `batch`, up to
  16 as `set_rows_group.h` and `llmp.set_rows.grouped_quant` take; the
  stores stay nodes). That key is now a batch key only (amended
  2026-10-09): its first form, a fused node of two or three stores, held
  none of GGML's groups and had no user, so it was dropped; the builder
  and verifier refuse such a node, it has no decomposition, and a
  capability of it describes one member store (GGML's quantizing group
  declares `batch` 16). A rewrite is refused where emitting it at its
  last node's place would move a read past a store into the same bytes
  (a cache, or an activation a store targets). Slice G2b added a
  composite kind and option entries, which a policy disables by name as
  any pattern: `route_reduce` (the selection may take an implementation
  of `kRoute` and `kExpertReduce` themselves, GGML's fused Gemma routing
  and scaled reduce; disabled, they decompose into llama.cpp's
  primitives); `owner_bounded_roots`, `owner_padded_reads` and
  `owner_chained_masks` (owner_attention's forms, below); and
  `narrow_final` (the step builder's narrowed final layer, below).
  Slice G3b added opt-in rewrites (`Pattern::opt_in`: a policy runs one
  only where it enables it, `enable <name>`, since each changes rounding
  as a fast form does) and a second round over the first round's fused
  nodes (`Pattern::round`), in `fusions.cc`. A rewrite may emit at several
  of its nodes' places (`Emissions`) and pull leaves made after its
  first place. Round 0: `hc_prep` (a mix as `llmp.hc.prep`, the BF16
  products, `llmp.hc.lo` and `llmp.hc.mix_bf16`, with the combine before
  it and the next combine's logits), `hc_mix` and `hc_combine` (the
  reference form's), `gdn_conv` (with the next history,
  `llmp.gdn.history`, and a verify's save of its result), `gdn_gates` (a
  verify's), `gdn_step` (up to 16 rows, the state written in place),
  `gdn_norm_gate` (folding the output product's MXFP8 preparation),
  `qsa_prep`, `qsa_pool` (each block's pooled key stored at its cell),
  `qsa_select` (the selection and the attention over the kept cells,
  reading the cached block keys), `qsa_gate` (the output gate with the
  output product's preparation), `moe_router`, `moe_experts` (the
  CUTLASS layout's whole routed block), `moe_glu` and `moe_combine`
  (GGML's NVFP4 blocks), `vecq_glu` (a GGUF's decode GLUs, the shared
  expert's only beside routed ones that take it), and `mxfp8_inputs`
  (the MXFP8 tensor-core products' inputs prepared once). Round 1:
  `hc_fold_inputs` (a fold's prepared and rounded rows stand for the
  preparations and BF16 conversions of its result), `bf16_inputs`,
  `bf16_rows` (a product whose rows only a convolution or a gated norm
  reads records them in BF16; its owners' views made again over them)
  and `vecq_inputs`. An opt-in option, `owner_row_invariant`, makes a
  step of several owners compute each owner's rows as its lone step does
  (`StepPlan::owner_row_invariant`: today's Qwen3.8 waves; the lowering's
  forms below). Each new fused kind (`ir/op.h`: `kHcPrep` to
  `kMatMulIdGlu`) is defined by its decomposition; a fused output may be
  recorded prepared (`ir::PreparedRows`: MXFP8 rows in CUTLASS's scale
  atoms), and capability queries see a node's outputs as operands from
  `execution::kOutputOperand`.
- **Policy** (`policy.h`): disabled patterns, the decode-row bound,
  row-invariant plans (D-092), the lossy opt-in and measured ranks per
  device class, as canonical text whose digest is a plan key's input. The
  first migrations' selections are policies: Gemma 2 serves with
  `disable grouped_stores`, `disable rms_norm_mul_rope` and `disable
  rope_store`, Gemma 3 with `disable rope_store`. Each family's default
  policy text lives in its architecture mapping (`spec::Mapping::policies`,
  slice G1a; rules gated by a source key as slots are, G2a: Gemma 2 and
  Gemma 3 one ungated rule each, Gemma 4's 26B-A4B, 31B and assistant
  their own), each rule's text printable lines; the component step
  source parses the rule `spec::DefaultPolicy` picks for the source's
  values and the serving envelope (the layout's context and the runner's
  owners: a rule may hold only within a context and owner bound, slice
  G2b) unless it is given another policy (an alias's settings, later).
  Gemma 4's rules are the recipes its M3 class served (deleted in
  G2d, the rules now the only record): within 4,096 cells
  and two owners, every row to the head (`narrow_final` off), the fused
  norm-mul, norm-rope and norm-add, `route_reduce` on 26B-A4B, the
  quantized GeGLU on 31B, owner attention with bounded roots and masks
  joined in pairs; within 8,192 and four owners the same without bounded
  roots (owners of equal reads only); outside, the narrowed final layer
  and the fused norm-mul alone. The assistant's is plain primitives.
  Qwen3.8's (slice G3b) are today's fast form in every envelope, gated
  by the import's form (`experts_stacked`: the ModelOpt import with its
  experts in the CUTLASS layout; `scaled`: in GGML's NVFP4 blocks; else
  a GGUF quantization's), each enabling its form's fusions and disabling
  the norms' and owners' patterns today's Qwen3.8 graphs do not take; the
  MTP drafter's its BF16 inputs, its mixes', QSA's and the CUTLASS
  experts' forms. A gate holds for a flag value too (a v0 binding's
  fact).
  Owner attention's Gemma 4 forms (16 or 32 heads, D 512) are selectable
  on the GB10 alone, as the M3 runner kept independent attention on
  other devices (`llmp.flash_attn.owner_roots`' capability).
- **Selection** (`select.h`): each node's first candidate from
  `Candidates()`, asked with the device's architecture, features, tuning
  data (an implementation that `needs_tuning` is offered only for a shape
  the device has data for: EXL3's pinned reconstruction) and measured
  ranks for its class; a node without a candidate is replaced by its
  decomposition where it has one (the primitive-fallback rule) and the
  plan refused otherwise.
- **Lanes** (`lanes.h`) and **placement** (`placement.h`): ports of
  `AssignLanes` and `PlaceActivations` onto the IR's steps (largest first,
  lowest free offset, step lifetimes, regions live for their spans). The
  implementations that stay on the stream (a borrowed library handle) are
  the caller's `on_stream`: unlike `AssignLanes`, which keeps cuBLAS and
  the BF16 GEMM there by default, the planner knows no module's names, so
  a lowering with such implementations must pass it.
- **Keys**: `KeyOf` makes `PlanKey` from the spec's bound digest, the
  shape, the enabled patterns' digest, the policy's and the partition.

On the Gemma 2 and Gemma 3 specs (`planner_gemma_test`, CPU), at every
shape the runners measure at setup with the serving defaults (Gemma 2 at
8,192 with two owners, 322 shapes; Gemma 3 at 4,096 with two and 8,448
with one, 378), with a fake registry declaring what the recipe selects and
the same model of the GB10's product routing on both sides (upstream's
`ggml_cuda_should_use_mmvq` at the DGX Spark's compute capability: MMVQ
through 8 columns for Gemma's Q4_0 and Q8_0, and `MulMatVecQGluFusible`'s
fused GeGLU at one), the planned steps' fusions (by kind and node set,
each node paired with the builders' by operation, and each product's and
weight multiply's by weight resource) and shared Q8 groups (by weight)
equal `PlanGraph`'s for the recipe, and the activation extent is no more
than `PlaceActivations`' over the builders' graphs (54% to 100% of it), with
the owner-attention forms or without, with no two live tensors sharing
bytes. (Those comparisons were made before the M3 builders and runners were
deleted, slice G1d; `planner_gemma_test` now checks that the recipe's
fusions and shared Q8 groups apply at every shape and that no two live
tensors share bytes.) A capability query carries the node's parameters
(`CapabilityQuery::params`), so a predicate can refuse what its kernel does
not take (a rotation's mode and dimensions; the owner kernel's softcap,
G1b), and a pattern asks with its match's parameters; an operand's
extents ride along (`OperandFormat::ne`) for a kernel whose conditions
are on them (the owner kernel's heads, cells and each owner's cache).
A pattern also sees the step's shape where the planner is given it
(`PlanOptions::shape`).

**Owner attention** (`owner_attention`, slice G1b). One match covers a
whole step: every layer's per-owner attention of all the step's owners
(each owner's query view, cache views, attention and result rows, and
their joins), in one of the forms GGML's owner kernel takes: one row an
owner with every owner returning its row (the builders' owner decode, 2
to 12 owners; a state-only step keeps per-owner attention, as they do), or
two owners of equal rows and reads (owner prefill, 2 to 512 rows, the
flexible tiles included). Every layer joins or none does, as the
builders chose a step's form. The rewrite keeps the owners' masks and,
after the last, joins each cell rule's masks once on dim 3 (each padded
with -inf cells to the widest: the builders' `join_mask`), then at each
layer's place emits the queries as [D, R, H, owners], a
`fused.owner_attention` node per group of up to four owners (views of the
queries and the joined mask on dim 3 for more than four, with the cohort
and offset), each owner's cache views, and the groups' rows joined. Each
owner's cache is read at its own width where an implementation takes
that (bounded roots: GGML's for two owners alone), else padded to the
mask's with zeros made once per owner and rule (the builders' zero tails,
Gemma 3's three to twelve owners). The builders' copying packed prefill
(two owners' caches concatenated for flash attention past 512 rows or
131,072 cells) is not ported: no serving envelope plans it (chunks are at
most 128 rows), and the IR keeps per-owner attention there; nor is the
whole-twelve bounded carrier (`bounded_whole12`, a default-off factor),
which the capability does not declare.

Slice G2b extended it for Gemma 4: each layer its own head sizes and KV
heads (the local layers' D 256, the global layers' D 512), and three
options a policy may leave off: `owner_bounded_roots` (each owner's
cache at its own width where an implementation takes that),
`owner_padded_reads` (narrower caches padded with zeros to the widest);
with both off owner attention joins only owners whose reads are equal in
every kind (Gemma 4's recipe within 8,192 cells and four owners); and
`owner_chained_masks` (the masks joined one after another; off, in
pairs, the pairs then joined, at most four owners: Gemma 4's owner
transform, once `gemma4_attention.cc` over the builder's graph, now the
planner's pattern, `planner/pattern.cc`). Gemma 4's transform post-processes
the per-owner graph and leaves the views it replaced in it, and orders
the later owners' stores and the joined masks where its traversal
reaches them; the rewrite emits the same launches over the same operands
in the IR's order (stores, then the joins after the masks, then each
layer's owner node), which only the structural comparison sees.

**The narrowed final layer** is the components' (below): with a head,
the stage's last component that mixes rows gathers its output rows
(`narrow_final`, the runners' `frontier_head`, which serving keeps on),
and the existing `rms_norm_mul_add` matches the norm-add over the
gathered residual (`GemmaNormAddGatherFusionAt`'s nodes), the gather
between its mul and its add. A policy that disables `narrow_final`
(Gemma 4 within its envelope, whose serving turns `frontier_head` off)
has `BuildStep` keep every row to the final norm (`StepOptions::narrow`)
and the head gather its rows there.

**The GGML lowering, as built** (`kernels/ggml/lower.h`,
`capabilities.h`, `step_program.h`; 2026-10-09). The GGML module declares
a capability for every implementation the Gemma 2 and Gemma 3 recipes
select and the other one-node operations PlanGraph names
(`capabilities.h`, a table every profile builds, so a CPU build plans for
the module; `implementations.cc` attaches it to the declarations). Each
carries its kernel's conditions: operand encodings and layouts, the build's
SASS architectures, the fused norms' widths and full NEOX D 256/512
rotation, the quantized GLU's one column past Pascal over gate and up
weights of one type, a GLU's kind, and upstream's routing where the planner's choice depends on it (`MmvqRouted`,
`ggml_cuda_should_use_mmvq` at the pin, for the shared Q8_1 preparation and
the fused quantized GLU). The product and flash-attention families are
groups (`ggml.mul_mat`, `ggml.flash_attn_ext`): the plan records the
group, and the lowering resolves it by upstream's routing. The grouped KV
store declares its batch (16). `LowerStep` makes the planned graph's
tensors in a `TensorArena` in node order, expanding a fused node into the
primitive nodes its rewrite replaced (GGML's fused implementations run
over those, as PlanGraph groups them), makes each mask on the device
(`llmp.gemma4.mask`) or, by option, as a host-built input, binds the
activations at the IR placement and the intermediates a fused kernel
never writes at its result's place (so it refuses a fused RoPE store,
whose result is a view of the cache and leaves its rotation no place: the
Gemma 2 and 3 recipes disable `rope_store`, and Gemma 4's migration owes
it one), derives `DeviceChoices`' fusion flags
from the selection, runs `PlanGraph` without upstream's fusion and refuses
the step unless its steps are the selection's, in order (a group's member
for a group). One placement input is the module's: MMVQ's prepared Q8_1
input is written with each row padded to 512 values (`Q8Bytes`), more than
the tensor's own bytes, so a kernel source gives the planner what its
implementations write (`PlanOptions::written`, `WrittenBytes`) and the
lowering refuses a tensor its place does not hold. `GgmlBackend` makes
programs of lowered steps (binding as `BindPlanned` does, running the bound
graph) and a `GgmlStepLaunch` over the launch context, whose captures are
the context's CUDA graphs; the engine reaches it as a `StepBackend`
(`kernels/common/lower.h`) through `engine/component_source.h`.

Slice G1b added: `fused.owner_attention` lowers to GGML's owner-root node
(`FlashAttnOwnersNode`, `llmp.flash_attn.owner_roots`, whose capability
carries `CheckFlashAttnOwners`' contract over the node's extents), with its
cohort, its offset for a partial cohort's groups only (the builders give
whole eight- and twelve-owner carriers offset 0), its softcap as the
kernel's integer and bounded roots where an owner's cache is narrower
than the mask. A fused node GGML runs over its primitive nodes makes each
of them at its own place in the primitive graph where its sources are
made, else at the fused node's: the builders' order, with the narrowed
final layer's residual gather between the norm-add's mul and its add,
where PlanGraph defers the fused norm-add. A view's offset, which the IR
counts from the root, is passed to GGML from the viewed tensor.

Checked: at every Gemma 2 and Gemma 3 setup shape above, and Gemma 3's
joined one-row steps of 2 to 12 owners with every count of long owners
(its runner's four-root carriers, beyond the serving envelope), with
device and with host masks (644 and 1,218 lowered steps, 66 and 374 of
them with owner attention), the step planned under its mapping's policy
and lowered is `BuildGemma2Graph`'s or `BuildGemma3Graph`'s under the
serving recipe (owner decode with bounded roots, owner and flexible owner
prefill, `narrow_final`, and the fusions above), node for node
(operation, type, extents, strides, view offset, operation parameters,
flags, every source), and PlanGraph's steps over it are the builders'
(`ggml_lower_test`, CPU; the comparison went with the builders in slice
G1d, and the test now checks that every one of these shapes lowers to
well-formed graphs with the plan covering each node that launches). On
spark-b, the model runner over the component
source runs two owners of the real Gemma 2 2B and Gemma 3 4B QAT artifacts
through one script of waves (joined prefill chunks of 128, 37 and 2 equal
rows, ragged and single-owner chunks, state-only chunks, one owner past
its ring until it wraps, joined one-row decodes returning logits and
device-chosen tokens over unequal reads), each owner's outputs bit for bit
those of the M3 runners under the full serving recipe, the repeated
decode shapes captured and replayed (`component_runner_gpu_test`, which
since the runners' deletion checks the script's outputs finite and exact
from one runner to the next).

Slice G2b added Gemma 4 (2026-10-09). The lowering makes `kMatMulId`
GGML's `mul_mat_id` (capabilities `ggml.mul_mat_id.mmvq` and `.mmq`, the
group `ggml.mul_mat_id` resolved by upstream's routing), an expert
array's leaf (its `WeightRef::expert_array`, which `BuildStep` now sets
from the binding) at the runner's slab pitch
(`LowerOptions::expert_stride`),
and the composites where the plan selected their own implementation
(`route_reduce`): `kRoute` as the ten nodes GGML's fused Gemma routing
takes (`ggml.gemma.route.fused`: a normalized softmax top 8 of 128,
scale 1, at most 8,192 rows; its ids a view of the whole sort, which
their place holds, `WrittenBytes`) and `kExpertReduce` with each slot's
gathered scale as the seventeen the fused scaled reduce takes
(`ggml.gemma.scaled_reduce.fused`: 8 slots of 2,816; its products and
partial sums take its place); otherwise the planner decomposed them into
llama.cpp's primitives. `fused.rms_norm_mul_rope`'s capability takes the
rotation's frequency factors (upstream's fused launcher reads them:
Gemma 4's global layers), and `rms_norm_mul_add` matches only a norm, its
mul and its add one after another (leaves aside) or with the residual's
gather between, as PlanGraph fuses them (where Gemma 4's routed and dense
sums meet, the later one). Unfused, PlanGraph runs an RMS norm and the
mul after it as one step of both launchers, which the lowering takes
as the IR's two steps (the assistant's plain plans). A drafter's features
are a features input (`InputKind::kFeatures`, never staged: the engine
copies them in on the device from its stage's tap or its carry, slice
G2c); a target's tap cells a store-cells slot of the segment's last
`outputs` rows (`InputSlot::outputs`), which the engine stages as those
rows' cells; a frozen read's mask the producer's frozen policy
(`MaskPolicy::kFrozen`: the cells hold the positions before the segment's
first row) or a host mask whose slot says so (`InputCells::frozen`, which
the engine's staging honours; serving's backend sets
`LowerOptions::frozen_host_masks`). A drafter head's next features are
each owner's output slot after its logits, marked `features` (the
engine carries them to the slot's next pass).

Slice G3b added Qwen3.8's fast form (2026-10-10). The fused kinds
lower to `qwen38_graph.cc`'s kernels over the same operands
(`MakeDirect`): `llmp.hc.prep` (its blob's later outputs within the
first's place, `WithinRoot`, which placement honours), `llmp.hc.lo`,
`llmp.hc.mix_bf16`, `llmp.gdn.conv`, `.history`, `.step` (written in
place), `.gates` and `.norm_gate` (its I8 rows the output product's
MXFP8 input), `llmp.qsa.prep`, `.pool` (its marker past the positions'
bytes), `.topk` and `.attn` (the selection's two kernels for one
`kQsaAttention`), `llmp.qsa.gate_quantize`, `llmp.moe.router`, the
CUTLASS experts' gemv or grouped-GEMM chains for one `kMoeExperts`,
`llmp.moe.glu` and `.combine`, and `llmp.vecq`. A weight of several
parts (MXFP8 codes and scales, the NVFP4 table's rows and scale, the
CUTLASS slab) is a leaf a part (`LoweredLeaf::part`); MXFP8 products
are `llmp.mxfp8.mul_mat_vec` up to 8 columns, else the swizzle and the
tensor-core GEMM over prepared rows (`RowsLayout`'s bytes,
`WrittenBytes`); a capability whose node launches several or other
steps says which (`Capability::launches`). Intermediates of a kernel
chain sit in its result's place after it (`KernelBytes`). Nodes nothing
needs are not made (a fused selection's masks and block-cell inputs).
The new inputs lower: side rows (`kSideRows`, the caller's), block
positions, fixed, per-block and per-row cells, positions in four
sections, a verify's saves stored at `Addresses::save` by their rows and
a commit's saved rows read from there (`kCommitRows`); masks with
`MaskParams::exact_rows` the producer's exact rows over their proved
position source: the replicated first section or original plain positions
(`CausalMaskRows::kExact`, as today's). GGML's own convolution
(`ggml.ssm_conv`, a policy without `gdn_conv`) pads its input past 32
tokens to whole windows (RE-032), as the unfused builder does. The
primitive form keeps each owner's block keys every step (its complete
blocks' and, unread, its incomplete ones'); `llmp.qsa.pool` writes only
the blocks a chunk completes, which is all a selection reads. Under
`owner_row_invariant`, a step of several owners' rows (their products
joined) keeps each owner's results its lone step's: `llmp.vecq` one token
a pass (`SetVecQOneToken`, as today's waves), the hyper-connections' BF16
products on their vector kernel at every width it takes
(`SetGemvBf16Wide`), and GGML's float products of up to 8 columns on the
row-invariant vector kernel (`DeviceChoices::vector_floats`).

The specialized QSA lowerings require the component step's typed staging
contract. Before creating a program or eliding operands, the lowerer
checks `StepInput` declarations and graph producers: positions are four
copies of each row's position, block positions and cells follow
`InputValues`, stores target the owner's non-ring caches, and causal,
block-seen and attention masks derive from those same positions and cells.
Views must describe the full cache prefixes the kernels read, and inferred
inputs and masks must not be overwritten. Input declarations must name
unique leaves. Tensor names provide no proof. Generic IR with arbitrary
external position tables, masks or indices remains defined and reference
executable, but these GGML specializations refuse unsupported staging
semantics during lowering; this refusal does not select a primitive fallback.
Exact device masks that remain primitive also prove their plain owner
positions. They reuse replicated positions only when one unchanged typed
input proves the same first section; otherwise they stage the original
plain input and produce the same exact row shape.

Checked (`ggml_lower_qwen38_test`, CPU): for each import (CUTLASS
experts, GGML's NVFP4 blocks, a GGUF quantization's) at the shapes
today's runner measures at setup and the fast kernels' row limits
(prompt chunks of 512 at 0 and at the context's end, the widest chunk
that does not select, decodes at 0 and at the end, their token shapes,
8, 9, 16, 17, 33 and 40 rows, each state-only), with the MTP drafter
(each step's streams for it, verifies of three rows at 0, 40 and the
end, the drafter's prefill and draft passes), the step planned under the
mapping's policy and lowered launches `BuildQwen38Graph`'s and
`BuildQwen38MtpGraph`'s fast-form nodes (96 steps), compared as
multisets (the n-gram layer's and the routed experts' independent nodes
are ordered otherwise), every operand by its root's description (so
reshapes compare as what they describe), nodes nothing kept or written
depends on aside (today's drafter prefill pass computes its queries),
and a state-only step's last mix compared through its readers (today's
also computes the next combine's unread logits); commits of 1 to 3
kept rows of one and two owners lower (llmp.gdn.step written); several
owners' decodes, chunks, verifies and drafter passes lower, each owner
its own pool, convolution and recurrence. On spark-b
(`component_runner_qwen38_gpu_test`), the model runner over the component
source runs two owners of each installed import through one script
(prompt chunks, short chunks at the row limits, a state-only chunk, one
owner past the indexer's budget, joined decodes of logits and
device-chosen tokens, captured and replayed) against today's runner's
fast form: all 31 output rows or tokens and all 242 state tensors are
bit-identical in each of the three imports, with two captured replays.
G3b covers verify, commit and drafter lowering structurally. G3c1 adds
native target verify parity and generated commits from actual verify saves
in all three forms; G3c2a adds the durable drafter execution contract and
scalar native MTP comparisons ([the model runner](engine.md#the-model-runner)).

Checked (`ggml_lower_gemma4_test`, CPU; the node-for-node comparison with
`BuildGemma4Graph` and `BuildGemma4AssistantGraph` was made before slice
G2d deleted the builders, and the test now checks structure: every
leaf binds a weight, a cache or a step input, the plan covers the nodes,
and the policy's patterns apply):
at every shape the M3 runner
measured at setup under each Gemma 4 serving recipe (within 4,096 and two
owners, within 8,192 and four, and outside the envelope, at the recipes'
chunk bounds), with device and host masks (324 lowered steps of each of
the 26B-A4B and 31B, 60 with owner attention), the step planned under the
gemma4 mapping's policy for that envelope and lowered launches
`BuildGemma4Graph`'s GGML nodes over the same operands (each view's
extents, strides and offset of the same storage) with the same operation
parameters and flags, and PlanGraph's steps over them are the builder's,
in order (as multisets where the builder's owner transform reorders
independent nodes); the assistant's draft steps lower to
`BuildGemma4AssistantGraph`'s with its host masks (frozen), and with
device masks to the frozen producer. On spark-b, the model runner over
the component sources runs the real 26B-A4B and 31B through one script
of waves per recipe (joined prefill, ragged, single-owner and state-only
chunks, one owner past its ring, joined one-row decodes of two or four
owners returning logits and device-chosen tokens, captured and
replayed), each owner's outputs bit for bit the M3 runner's; and, as
stage 1 of the same runner with the target's state borrowed, two joined
draft passes of each assistant over its target's cache from the M3
assistant's first feature and anchors, each pass's logits and next
features bit for bit its (`component_runner_gemma4_gpu_test`, which since
the deletion checks the outputs finite and exact from one run to the
next, including the verify and draft cases; the engine's served
assistant has `Gemma4AssistantServingGpu`).

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
| Dense attention | GQA with softcap (G2), sliding ring and global layers (G2–G4), tied K=V and V-norm (G4 global), MQA with sinks and window (DS), query-only reading the target's cache (Asst), small causal (Img text encoder), bidirectional FA2 (Img) | Q2 uses shared component emission and per-operation GGML/EXL3 lowering; image attention separate |
| Owner-root attention (joined requests reading their own caches) | G2 (two owners), G3 (2–12), G4 (rewrite pass) | inline in two builders plus Gemma 4's rewrite pass (now `planner/pattern.cc`) |
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
classes and its own `model::Batching` limits; split options as data (the
roles and state tensors each divides, its collective and the bytes it
joins per step, [Partitioning](#partitioning)); the pattern names it takes part in; the
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

The head reads only the step's output rows, so `BuildStep` narrows (slice
G1b): with a head, the stage's last component that mixes rows (a
definition's `row_local` false; `attention.dense`, whose `narrows` says it
can) gathers its output rows after the last of its own ops that mixes
rows (the projection's and the residual's rows, after the output
product), and every later component, row-local, runs at the outputs'
width, the head gathering none (`Fragment::narrow` and `narrowed`): the
builders' `narrow_final`, which the runners' `frontier_head` builds and
serving keeps on. Gemma 4's `ffn.moe` and `scale.learned` are row-local
(the 26B-A4B and 31B narrow after the last layer's attention, as the
builder's `narrow_final` does); `attention.borrowed` narrows as
`attention.dense` does (its emit is the same), but a head of every row
(`head.linear` with `gather` false, `ComponentDef::gathers`) has nothing
to narrow to, so a drafter's stage never narrows. A target's tap write
(below) stores the narrowed stream's rows directly: after the narrowing
they are the output rows, in segment order. The M3 Gemma 4 runner turned
`narrow_final` off when it retained features, because it keeps every
row's; the components' tap keeps only the output rows' and narrows. A
state tensor's cell is the format layer's
(`model::CellBytes`), its leaf a block encoding's whole blocks where an
alias opts into a lossy cache (D-108; F16 today).

Each fragment records today's builders' nodes in their execution order
(GGML's `GraphOrder` over their outputs: V's product before K's, the gate
before the up product, each owner's K and V stores before its attention,
the owners' outputs joined after the last one's attention), with the same
types, extents, view strides and arithmetic parameters; a CPU test
compared the Gemma 2 and Gemma 3 steps with `BuildGemma2Graph` and
`BuildGemma3Graph` node for node (`components_order_test`, deleted with
them in slice G1d), with host and device masks: every
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
- **The builders' options** are patterns and selection, not fragments:
  shared Q8 preparation, grouped KV stores, the fused norm forms and owner
  attention (owner decode and owner and flexible owner prefill, bounded
  roots, Gemma 3's four-root carriers) are the planner's (above); the
  narrowed final layer is the framework's (above); device masks are the
  lowering's (above); the diagnostic scopes (`first_layer`,
  `layer_count`, `hidden_input`), the matched controls (`frontier_head`,
  `owner_prefill` or `flexible_owner_prefill` off, `bounded_whole12`) and
  the copying packed prefill are not ported. The owner-attention kernel's
  literal softcap 50 is an implementation's limit its capability checks
  against the node's `softcap`.
- **Batching.** The primitive graph joins any rows of up to 16 owners
  (attention per owner), so the components declare no limits of their
  own; the owner-root forms (one-row decode of up to twelve owners in
  groups of four, equal-row prefill of two) are the pattern's, which
  leaves any other step's attention per owner.

**Gemma 4** (slice G2a, 2026-10-09). `attention.dense` gains optional
forms, each off by default: `kv_tie` (V is K's product before K's norm
and rotation; the layer has no V weight: Gemma 4's global layers),
`v_norm` (an unweighted RMS norm of V's heads) and `rope_freqs` (role
`rope_freqs`: per-pair frequency factors dividing the angles, op.h's
`t / freq[i]`: the global layers' proportional rotation, one table the
layers share); a query scale of 1 emits no node. Head dims, KV heads,
windows, rotation dims and bases are per-instance parameters, so the
local (D 256) and global (D 512) layers differ only in data.
`attention.borrowed` is the same attention for queries only over another
stage's K and V: its contract names the target's tensors
(`borrowed_from_target`, `target_layer`), it stores nothing, and its
masks read the cells frozen (`CellKey::frozen`: the positions before the
segment's, so a draft row sees only its target's committed positions).
`scale.learned` is the layer output scale. `ffn.moe` is the routed block
with a dense GLU beside it (placement `parallel`: Gemma 4 26B-A4B's), its
route and reduce shapes parameters (`experts`, `used`, `expert_hidden`,
`normalize`, `expert_scales`, `stacked` gate-up arrays); its fragment is
built of three pieces with their own scalar references (`moe.h`: the
router, emitting `ir::kRoute` after Gemma's scaled-norm router input; the
routed experts' `kMatMulId` products, GLU, each slot's expert scale
gathered as the builders gather it (the scales' rows repeated a row each,
then each row's selected ones: slice G2b) and the ordered
`kExpertReduce`; the gated feed-forward product `ffn.glu` shares), and
it splits by
experts (EP; the partitioner does not yet pass a stacked array's halves,
`kView` of an expert-local value). `embedding.projected` is the
assistant's input: its target's token table's rows (a weight of another
stage's artifact, `WeightBinding::stage`, with the target's `WeightRef`)
beside a feature (a step input, `InputKind::kFeatures`, which the engine
binds: the target's tap for a unit's first pass, the drafter's last
feature after), projected to its width; its state is the tap ring the
target writes (`target_writes`, kPositionRing of `tap_cells` cells by
position, at least a verify's rows: the runner refuses a narrower one).
A sliding-window cache truncates a verify's rejected rows (slice G2c;
before, no verify could write it): its ring holds a chunk's rows beside
its window, so a later query's mask puts a rejected row's cell, and the
older position it overwrote, outside its window. `head.linear` gains `gather` (false: every row an output, a
drafter's) and `features` (its rows projected to its target's width:
`StepGraph::features`, the next pass's feature).

The spec's reads link to the contracts (`ValidateSpec`): a borrowed
tensor needs a state read of its name and the target's tensor of that
name in its target layer with the same kind, cells, window and cell
bytes; a tensor the target writes needs a tap read of its name, a final
norm's tap of the target's width in F32; every read must be used; only a
drafter reads, and only its target (the spec's first stage). A target's
step with outputs writes each tap a drafter reads: the final norm's
output rows (the narrowed stream itself where the step narrowed),
stored per owner with outputs at their positions' cells of the
drafter's tensor (`InputKind::kOutputCells`), which needs the drafter's
layout (`Bindings::stages`). `CheckStateWrites`, run on every built step,
refuses a store into a borrowed tensor or into another stage's tensor
that is not a tap its target writes. The GGML lowering takes the new
inputs and frozen masks (slice G2b, above).

The structural comparison (`components_gemma4_order_test`, deleted with
the builders in slice G2d) held the
26B-A4B and 31B steps to `BuildGemma4Graph` with its default options
but `narrow_final` (the components narrow) and
the assistant's draft steps to `BuildGemma4AssistantGraph`, node for
node, with host and device masks. What differed: the router's selection
and the experts' reduce are the composite ops their GGML runs decompose
to (soft_max through the weights' reshape; the two products by each
slot's gathered scale and weight, the slot views and the ordered adds),
compared by position, outside sources (the reduce's ids aside) and
parameters. Not compared: the builders' options (the Gemma route and
reduce fusions, owner-root attention and common-width reads,
rope-store, shared Q8 inputs), which are patterns and selection, held
to the builders by the lowering's comparison (slice G2b, above).

**Qwen3.8** (slice G3a, 2026-10-10). The framework gains what the
hyper-connected, partly recurrent stage needs. A stage of several
streams carries its residual F32 [width, streams, rows] beside the
stream a block reads: its mixing component (`ComponentDef::mixing`,
`residual.hc`) repeats the embedding's rows into every stream (an
embedding may return the streams itself: `embedding.mtp`), a fragment
reads and returns the residual and a carry (`Fragment::residual`,
`carry`: what a mix hands the combine after its block), and the final
norm may gather the output rows itself (`FragmentOut::narrowed`).
`hc.mix` (each stream RMS-normed, the streams as one row times the norm
weight, the low-rank sigmoid gate, the gated streams' mean) and
`hc.combine` (the block's output added to every stream, weighted by
2 · sigmoid(inject · xn / streams)) wrap each block; `hc.mix` with
`inject` false is the head's mix. Side inputs are components of their
own (`ComponentDef::side_source`, numbered after the embedding):
`side.ngram` gathers each row's `heads` rows of its table (a role read by
rows on demand, `WeightRole::rows_on_demand`) at rows the engine computes
from the token history (`InputKind::kSideRows`, `ComponentDef::side_rows`:
llama.cpp's n-gram hash, equal to `Qwen38PleRows`), and exactly one
component of each layer it lists reads it (`side_input`: `residual.ple`,
the n-gram block before the layer's first mix, which adds to every
stream). Three cell rules join the contracts' (`CellKey`): fixed state
(one cell a request rewritten whole, stored at `StateRow`), a cell per
block of positions (whose held position is its last once complete), and a
cell per chunk row (a drafter's streams, the rows before the chunk
first); positions may come in several sections (a multi-section
rotation's) and blocks' first positions are an input
(`kBlockPositions`). `attention.gdn` is Gated DeltaNet: the qkv, gate,
beta and alpha products, the causal depthwise convolution over its fixed
history and SiLU, q and k L2-normed (rms_norm at eps / d, scaled by
1 / sqrt(d)), `kGatedDeltaNet` over the fixed recurrent state, the gated
output norm and the output product. It always requires the
`gdn.value_heads_tiled` fold. The default requires `ssm.a_neg_exp`; explicit
`a_log=true` instead requires that fold to be absent and computes the negative
exponential in the graph. `attention.qsa` is QSA: the
indexer's raw keys stored F32 a cell per position, the gated query, K and
V products, per-head norms and the interleaved multi-section rotation,
F16 K and V; past the budget the primitive fragment pools every block of
the read cells each step (mean, normed, rotated at the block's first
position, BF16), scores them against the indexer's BF16 query
(`kIndexScore`, the token's own block kept by a 1e9 bias over the blocks'
held positions), spreads the scores over the cells under the causal mask
and keeps the best min(read, budget + ratio − 1) cells (`kTopK`) as a
-inf mask whose `n_kv_max` promises that width; its contract holds the
BF16 block keys (`indexer_blocks`, a cell per block) for implementations
that keep them (`llmp.qsa.pool` writes each once), which the primitive
fragment never reads or writes; `rollback` is `snapshot` for a target and
`truncate` for an MTP drafter, whose catch-up rewrites its caches.
The HC mixing norm, QSA/indexer norms, PLE norms and convolution, MTP input
norms, and GDN's dt bias, A, convolution and output norm admit F32 or BF16.
Their immutable weight leaves keep the source encoding. A fragment lazily
converts each BF16 role once into joined F32 graph storage and reuses that
transient value across owners; the planner charges its ordinary activation
placement: E BF16 source elements remain 2E resident bytes, while their
conversion produces 4E logical transient bytes before placement alignment.
Logarithmic A uses ordinary F32 exp/neg operations there too.
No converted value becomes a resident weight or a persistent decoded cache.
The optional `norm_zero_centered` parameter adds one in F32 to the sigmoid
HC mixing norm (including the final head), all four QSA/indexer norms,
PLE's key/query/convolution norms, and the two MTP input norms. It defaults
false; true refuses an already-recorded `norm.plus_one` fold and is not
supported by Sinkhorn or GDN. Each affected role lazily shares its widening
and one joined F32 scale-bias result; raw and transformed cache entries are
distinct. The offset adds 4E logical transient bytes before placement
alignment; BF16 additionally needs its 4E widening, with no resident growth
or BF16 roundtrip. This is a component operand contract, not ModelOpt
normalization, publication, runtime admission or matched GPU evidence.
Existing F32/default graphs retain their lazy weight lookup order. This
component contract does not authenticate any MLX converter convention or
admit an MLX artifact; source normalization, norm/head-order authentication
and its gate remain separate. Initial BF16/logarithmic-A native checks passed
24 component and six structural lowering controls, with no skips; ten
historical F32 graph digests match the unchanged parent. These are scalar-reference and structural
lowering results, not new GPU arithmetic or MLX execution evidence.
N3b2b's zero-centered norm checks passed independent review, 36 focused
native controls, and the unchanged ten historical F32 graph digests. Its
isolated final checks passed 2,567 native and 2,016 CPU cases with zero
failures or skips, plus format, boundary, header and REUSE gates. Final main
checks passed 2,582 native and 2,027 CPU cases, 36 integration controls and
135 model/timing cases (182 raw GoogleTest runs), with zero failures or skips.
Source, recipe and model-binary guards passed, as did the quality gates.
These regression checks do not extend source or GPU arithmetic qualification.

`ffn.moe` gains placement `none` with the router over the stream itself
(`router_input` `stream`), per-expert projection scales (gate, up and down
roles; down's applied in the reduce) and the shared expert behind its
sigmoid gate (`shared_gate`: SumRows of the stream times the gate's row).
`embedding.mtp` is the MTP drafter's input: its target's token row,
normed and projected, added to every stream of the feature it reads (its
target's last layer's streams, or its own last pass's), the feature
normed over its streams together and each stream projected; its state is
that tap, a cell a chunk row its target writes every row of.
`head.linear` gains `selected` (role `ids`: its rows' token ids, the
greedy pick mapped through them) and `embedding.token` emits no scale of
1.

A verify cannot rewrite fixed state (`Rollback::kCommit`): in a verify
step (`StepContext::phase`) `attention.gdn` and `residual.ple` read their
state, write none, and save each row's convolution input, convolution
output, gate and beta (`StepContext::Save`, `StepGraph::saves`); the
commit of each owner's kept rows (`BuildCommit`, from `ComponentDef::
commit` fragments over `kCommitRows` inputs the caller binds to the
saves' first rows) leaves the state those rows alone would have, bit for
bit on the reference executor; `BuildStep` refuses a verify that writes
its stage's commit-rolled state. A step's state leaves say whether a node
reads them and a store writes them (`StepState::read`, `written`), and
`CheckStateWrites` refuses a store into a weight, a borrowed tensor or
another stage's tensor that is not a tap the step writes (a store into an
activation, a mask built by stores, is the step's own). Taps may be a
layer's output or its streams (`TapPoint::kStreams`, F32 streams times
width, written every row after their layer, which the MTP drafter reads),
and reads reach any earlier stage (its target, or the drafter before it
in a cascade), each name of one stage.

The structural comparison (`components_qwen38_order_test`) holds the
target's steps, from a GGUF import's tensors and the ModelOpt import's
with GGML-layout experts, to `BuildQwen38Graph`'s unfused form node for
node over five one-owner prefill and decode shapes below the indexer's
budget (the unfused form builds one sequence; its selection rounds in
F32 where the components round as the fast form's kernels do, in BF16).
The selection, the MTP drafter, verifies with their commits and several
owners are held to the components' references here, and to today's
runner by G3b's lowering. What differs: GGML's graph order visits the
n-gram layer's history, value and key products before layer 0's shared
expert and combine (the comparison allows reorders only within that
window: the lowering's order, G3b); the builder's MXFP8 dequantization
nodes are the products' operands; the GDN's per-segment views and the
shared gate's row trick (a reshape and `get_rows`, where the components
multiply and sum) compare by alias; the builders' host masks and the
convolution's padding past 32 rows are the lowering's. The fast form
(`llmp.hc.*`, `llmp.gdn.*`, `llmp.qsa.pool`, `.topk`, `.attn`, the MXFP8
tensor-core products), the default planning policy and the lowering of
the supported new inputs and cell rules are
slice G3b's (as built: [the graph IR](#the-graph-ir)); the runtime's
on-demand n-gram binding, commit execution and chunk-feature staging landed
in G3c1. G3c2a adds an opt-in durable pending interval and prompt companions;
serving recipes and the migration gates remain G3c2's.

**DeepSeek V4 and DSpark** (slice G4a, 2026-10-10). Four new kinds and
extensions of three, each with a CPU reference independent of its
fragment (`components_dsv4_test`):

- the Sinkhorn hyper-connections are `residual.hc`, `hc.mix` and
  `hc.combine` with `mix` `sinkhorn` (G3a's sigmoid mix unchanged):
  the streams normed as one row, their product with `fn` split into pre
  weights (sigmoid plus `hc_eps`, summed by `kHcPre`, then the block's
  own input norm, `post_norm`), post weights and the streams² Sinkhorn
  combination (`kHcMix`, `iterations`); `hc.combine` is `kHcPost`. A mix
  now returns two carries (`Fragment::carry2`: post weights and
  combination), and a fragment learns whether a later component reads
  its carry (`Fragment::carry_needed`): a state-only step's last layer
  emits no combination;
- `attention.mqa` is DeepSeek's attention in one kind, chosen by
  parameters: the low-rank query with per-head norms, one K=V head, YaRN
  NORM rotation of the tail (`defs::YarnRotation`, the angles as GGML's
  kernel forms them) and the inverse rotation of the output, learned
  sinks, grouped low-rank output; the window cache a ring, a cell per
  position (`cache` `full`, today's `Dsv4Window::kFull`) or a drafter's
  injected ring; with a `ratio`, the gated compressor (CSA, with
  `overlap`, and HCA) pooling into the compressed cache through
  `kPool`, and with the `indexer` the lightning indexer (its own
  compressor, the Hadamard rotation, `kIndexScore` and the top 512
  through `kTopK`, a −inf fill with zeros stored at the selected cells);
- `ffn.moe` gains the `sqrt_softplus` router (with `router_bias`, F32
  accumulation, `route_scale`), the `hash` router reading a per-layer
  side input, and SwiGLU clamped at `limit` (`silu_clamp`; the shared
  expert's `shared_limit`); `side.hash` is that side input, each row's
  token's row of the host table (`ffn_gate_tid2eid`, rows on demand);
- `embedding.inject` is DSpark's input: its target's token rows, scaled,
  in its own steps, and in its target's its injection
  (`ComponentDef::inject`, `inject_reads`, `inject_rows`): the taps it
  reads (`TapPoint::kStreamMean`, the mean of the target's streams
  entering each named layer) joined per row, each segment's last rows
  projected by `fc` and normed; each `attention.mqa` with an injected
  cache then writes its ring from that input. `BuildStep` runs every
  injection after the target's head, as llama.cpp's graph does, and a
  drafter's injected tensors are taps its target writes;
- `head.linear` gains the Markov head (`markov`: each row's logits
  biased by `markov_w2 · markov_w1[previous token]`, the chain run per
  segment) for a drafter's head of every row.

The compressors need inputs and cell rules of their own: rings of recent
positions (`CellKey::recent`, read whole and kept as their last rows),
compressed caches a cell per block of positions (`CellKey::positions`:
their read width follows the segment's end, not its read cells), and the
host-built inputs `kPositionPhases`, `kKeptRows`, `kKeptCells`,
`kPoolReads`, `kBlockCells`, `kBlockStarts` (`PlanCompressor`, a port of
llama.cpp's `dsv4_build_comp_plan`, equal to `Dsv4CompressorPlan`) and
the constant `kHadamard`. A component's weight formats need cover only
the roles it uses. The mapped layouts are today's `Dsv4State` (both
window modes) and `DsparkState` byte for byte.

The structural comparison (`components_dsv4_order_test`) holds the
target's steps to `BuildDsv4Graph`'s reference form (`fused` false:
llama.cpp's graph) node for node over one-owner prefill (7 rows; 64 after
300 with two outputs; 130, an HCA block), decode, greedy decode,
state-only and an injecting chunk, and DSpark's draft block to
`BuildDsparkGraph`. The routing, the experts' reduce and the pooling are
composite ops compared as the run of nodes each decomposes to. Out of
order by design: each attention mix's carries (post weights and
combination), made right after the mixing product where the reference
form makes them just before their combine (its FFN mix already records
them there), eight entries a layer, the hash layers' lookups, made as
side inputs after the embedding, and views of leaves (weights', a
compressor's ring), which compute nothing: 388 entries of a target step (380
state-only), 24 of the draft block, each checked to be one of those. The
reference form builds one owner only (`BuildDsv4WaveGraph` and
`BuildDsparkWaveGraph` are the fast plan alone) and its verify is the
row-invariant plan's (D-092), so several owners' waves, joined draft
blocks and the verify are held to the components' references here and
compared numerically by G4b's lowering against today's runner.

G4b1 adds the position-derived dimensions to the shared plan key through
component callbacks: each owner's compressor block count and compressed
read width. Positions remain input data; advancing them with unchanged
dimensions reuses the plan, including lookahead and capture decisions.
Fragments, scalar references and keys derive compressed reads from the
state contract's cell padding (256 in production), independently of the
outer raw read alignment. Contract-backed ranges cover those views even
when joined raw reads align to 2,048 cells.

The GGML primitive lowering now stages the typed compressor inputs and
Hadamard constants, builds compressed masks on the host and ahead masks
on either the host or device, and lowers offset/inverse rotations. Its
registry declares the existing top-k, same-type half mask add, indexer
and F16/F32 conversion operations. The indexer's decomposition explicitly
widens a half mask before its F32 add; it also lowers with the native
indexer disabled. Structural checks cover target and draft steps and
validate the newly selected operations' launch constraints.

G4b2a adds four opt-in local patterns, disabled by default:
`dsv4_qhead` (F32 head norm and tail RoPE), `dsv4_compress` (source assembly
and pooling), `dsv4_hc_pre` (the full Sinkhorn carry), and `dsv4_hc_post`
(the post's streams and flat F32 norm). They authenticate the primitive
formula, complete parameters, ownership and intervening writes before
selecting existing helpers. HC pre exposes packed post and combination
carries through explicit copies of the helper's pitched tail; placement
funds its entire blob, mixing partials and both copies. HC post requires
packed boundaries, so strided inputs retain the primitive path. The mix
weight stays in F32, F16 or BF16 through the format layer.

Compressor dummy rows are literal zero/-infinity fills, independent of
masked state scores. Pooling's primitive decomposition clamps softmax
probabilities to [0,1]: GGML CUDA's fminf/fmaxf clamp converts its
all-masked NaNs to the IR's zero probabilities. The existing compressor
helper preserves zero-probability multiplication of nonfinite values,
including NaNs, while retaining its finite-value arithmetic. Its computed
read indices must come from the typed owner/ratio/ring contract; launch
checks bound both the channel and block grids.

These local forms have structural checks over target and draft fragments,
scalar definitions, and GPU operand comparisons against their existing
helpers, including capture replay and unequal owner carries.

G4b2b1 adds `state_bank_attention`, also disabled by default. It removes
only a whole raw F16 cache plus compressed-prefix concatenation and its
K/V permutations. The original query view, IndexScore, TopK, masks, sinks
and every Attention parameter remain unchanged. A retained copy or view,
an intervening write through either state root, a partial raw prefix or an
alignment gap keeps the primitive copy. Query views of joined projections
retain their exact offsets and strides; both banks belong to one owner,
stage and layer.

`model::ContiguousStateSpan` derives adjacency from the actual state layout
and rejects overlapping allocations, borrowed items, gaps and overflow.
`PlanOptions::state_spans` and the lowering receive the same immutable
facts. With this opt-in enabled, the component source adds its exact
layout identity to the plan key and checks supplied item addresses share
the declared region base. Lowering independently checks the bound virtual
addresses and reports the entire bank interval as state use. It changes
neither layout alignment nor backing: those reads must fit the existing
`UsedThrough` closure. A native recipe declares `needs_state_span` in its
capability and identity; queries without proof cannot select it. The typed
formula still has its ordinary Concat/Permute/Attention decomposition,
including requested F16/BF16 output conversion; the native recipe returns
F32 and retains the existing flash-attention recipe and parameters.
Structural checks lower the target's CSA/HCA banks with unequal owners and
2,048-cell raw alignment. GPU operand tests at D256/D512 compare the bound
span bit for bit with the primitive copy, including nonzero layout offsets
and persistent capture replay after query, mask and state changes.

The Lid indexer transfer remains G4b2b2. Hiding TopK IDs behind the final
mask does not establish equivalence of the selected cells. For example,
with one visible cell, top-1 and masked scores `[NaN, -Inf]`, the scalar
TopK definition selects the unseen `-Inf` cell and the causal mask leaves
none selected; a helper searching only visible cells selects cell zero.
The visible NaN can arise from two heads weighted +1 and -1 whose positive
dot products are both infinite. Finite rounding also needs an explicit
oracle: a scalar query `[1.0001f, 1.0f]` with keys `[0,1]` and `[1,0]`
selects cell one, while rounding the query to F16 ties both at one and a
lower-index tie rule selects cell zero. This latter witness compares
scalar policies, not measured CUDA recipes: the existing native IndexScore
also rounds queries to F16. No Lid capability or changed TopK semantics
is introduced by the bank form.

Default policy, original-reference quality and performance remain open.
G4b2c ports output-A and product policies after G4b2b2, executing all selected
contributions and preserving same-implementation state, rollback and replay.
Legacy bit parity is historical diagnostic evidence, not the prospective
acceptance rule; a blanket row-invariant verify recipe is not required.
Owner-safe verification protects each owner's inputs, selected contributions
and state. Unfused/fused/legacy score comparisons describe numerical relations;
they do not force one recipe solely to reproduce another's bits. Exact source
representations, declared precision conversions, masks for the actually selected
IDs and same-implementation restore controls remain required. The engine's
binding of the hash rows and injection's features, and its complete setup and state-lifecycle
integration, are G4c's. Expert
parallelism with as many experts as devices (`partition.cc`) is not
reachable for DeepSeek on two Sparks (256 experts) and is left as it is.

### Owned execution configuration

A required component source retains the native backend's immutable device and
explicit tuning facts. Tuning names are owned, and entries include physical
rows, K, N and output encoding. They never replace operand, stride, alignment,
selected-expert or capture checks. Ordinary sources keep their existing path;
no tuning record or family policy is installed automatically.

Component planning receives conservative alignment facts for single-part weights
from their actual nonzero virtual reservation addresses, keyed by the complete
weight binding and part role. The source checks power-of-two alignment and
address divisibility in both measurement and bound lowering, and preserves
semantic root identity through rewrites. Multipart logical roots remain unknown.
These facts do not select a numerical policy: the joined BF16 vector capability
still requires an explicit matching physical-shape and F32-output tuning record;
ordinary untuned products retain their existing selection.

The declaration uses the qualified installation's compiled source/module/SDK
and effective build recipes, actual device architecture, CUDA driver API and
runtime versions, cuBLAS/Lt versions, and queried math, pointer, atomics and
workspace settings. It assumes the qualified pinned installation and no hostile
mutation of running code. Runtime versions alone do not authenticate an arbitrary
installation. There is no runtime library hashing, loader-object attestation,
selected-SASS extraction or recursive compiler-invocation prerequisite.

Runner setup maps the paid workspace and prepares the real handle before
publishing the source's identity or measuring plans. All stages, effective
policies, domain, bound spec, layouts and capture requirements participate.
Failed preparation publishes no partial staged identity. Plan cache lookup,
opaque snapshots and kept restart compatibility retain that identity; changed
required configuration refuses instead of becoming an absent legacy identity.
Recorded and direct execution validate the same live backend, and a retired
launch cannot be revived by recreating an equal configuration. Physical
addresses remain live binding checks, outside durable identities.

Configuration metadata uses an explicitly admitted, pinned node-catalog allowance
before construction, independent of the pre-start host counter. Its bounded local
usage is refunded after objects are destroyed; the full catalog payment stays
through node retirement and is reused on setup retries. Bind and recordings also
retain the launch generation, so recycled addresses cannot revive old work.
The existing paid runtime/workspace and completion rules still apply,
including retention after unknown retirement. The initial declaration covers
ordinary GGML/cuBLAS; selected EXL3 and image providers remain outside it.
This source/plan/state integration does not grant joined repeat-pending
capability or prove the complete future/companion admission closure. Those
controls and family quality, state, swap and matched 1% gates remain necessary.

Internal diagnostics may validate the real lowerer's unchanged program through
an optional synchronous read-only callback. It runs after both measurement and
bound lowering, can refuse, and cannot replace the backend, selected program or
required owner declaration. The installing harness prepays and retains its
metadata through complete retirement. Test scoring also supports a synchronous
finite-row witness, including the last unscored row; sink refusal or cancellation
is terminal before further scoring or successful completion.

The internal Qwen target-only scoring envelope requires the actual sole target
stage, four slots, context 512, chunk two, read alignment 256, no drafter or
speculation, and its exact owned numerical domain. Setup additionally measures
full-head score shapes for the joined cohort and each fallback owner. This
does not enable ordinary serving, MTP, automatic tuning or a new numerical
default. Its bounded [secondary quality/state screen](experiments/m36-numerical-audit/qwen-joined-score.md)
remains separate from original-reference and migration qualification.

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
version 1, the pipeline kind and the stages; a stage has its role
(target, drafter, encoder, text encoder, denoiser, VAE), its artifact
index in the bound list, the folds its artifact's tensors carry (a
stage's, not the model's: a composition's stages may come from different
containers, so a drafter never inherits its target's folds; stages of
one artifact list the same; a binding to another stage's tensor carries
its holding stage's, `HoldingStage`), its trained context, an embedding, the
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
checkpoint of the same spec shares the layout but not the state. The spec nests eight levels, the v0 JSON's limit, so the v1
format keeps it as a document of its own, `spec.json`, listed and hashed
in the manifest, whose SHA-256 is then the spec's digest; `stored.h`
reads it from a v1 artifact or composition (D-113).

`mapping.h` makes specs from architecture mappings: compiled tables of
slots (component kinds at the embedding, the layers of some types, the
final norm and the output), tensor-name patterns with `{layer}` (and
fallbacks: Gemma's tied output binds the embedding), parameter rules from
source keys with a fixed set of transforms (reciprocal, square root,
reciprocal square root) and constants, tried in order per layer type,
requirements on source values, and folds from a fixed catalog
(`norm.plus_one`, `ssm.a_neg_exp`, `gdn.value_heads_tiled`) asserted per
source container; a stage's folds are its source container's with the
folds its importer recorded (`Source::folds`). Mappings are validated like untrusted input.
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

Gemma 4 (slice G2a) extends the mapping language, all as data: slots
gated by a source key (`ffn.moe` layers when `gemma4.expert_count` is
above zero, `ffn.glu` otherwise; a pattern may repeat only across two
such exclusive slots); parameters read per layer from a key's list
(`per_layer`: the KV head counts, llama.cpp's get_key_or_arr); a
layer-flag key that must agree with the layer types (the sliding-window
pattern); tensor rules limited to layer types (`attn_v` on local layers
only) or naming a tensor every layer of their types shares (no
`{layer}`: `rope_freqs.weight`); architectures with '-'; default
planner policies per gated variant (`PolicyRule`, `DefaultPolicy`; text
the planner parses), each rule with how serving composes the recipe's
steps (`prompts_alone`, `owners_alone`, the prompt futures planned and
captured ahead: `DefaultRule`, slice G2c; Gemma 4's prefill alone with one
future planned and none captured, and outside its envelope step one owner
at a time, as its M3 serving did; the prompt chunk the recipe was
qualified at, `prefill_chunk`: 1,024 rows for 26B-A4B and 256 for 31B); and drafter mappings (role, stage name, reads,
rules binding the target's tensors) that `ComposeDrafter` applies with
a target's spec, adding the taps the drafter reads and the target's
facts its parameters read (`llmp.target.last_layer.<type>`). The spec's
weight bindings may name another stage's artifact (`stage`, written in
the JSON only when set). The `gemma4` mapping asserts no fold: llama.cpp's
converter keeps Gemma 4's norm weights as they are (its
`Gemma4Model.norm_shift` is 0, read at 8019dc56).

Qwen3.8 (slice G3a) extends it again, all as data: side-input and
mixing slots (`Site::kSide`, `kMixing`); slots whose layers a source
key lists (`SlotRule::layers_key`: `qwen4exp.ple.layers`); tensor rules
binding any part of a tensor (`any_part`: the ModelOpt import's MXFP8
codes and scales) or accounting for tensors the spec holds as parameters
(`unbound`: the hash's constants); folds asserted for a v0 artifact's
converter identity (`FoldRule::converter`, the ModelOpt importer's, which
records no transformation: the start of one of the identity's `+`-joined
parts; a v1 artifact records its importer's folds, so no converter rule
applies to it) as well as per container; v0 resources paired into
one tensor's parts (`V0Assembly`: MXFP8 codes and scales, the n-gram
table's NVFP4 rows and scale, the CUTLASS expert arrays); integer-list
values (`ValueType::kInts`: the rotation's sections, the hash's
constants); and drafter taps of a layer's output or streams (the target's
last layer). `SourceOfV0` takes the mapping to assemble, and the caps
rise to 2^20 source tensors (an artifact's are 301,730) and 2^16 values.
The `qwen4exp` mapping makes one spec structure from the ModelOpt
import (experts in the CUTLASS layout or GGML's blocks) and a GGUF's:
the streams' mixing, the n-gram side input read by `residual.ple` at
layer 1, three `attention.gdn` layers to a `attention.qsa` one (the
per-layer compression ratios must agree), every layer's MoE, each block
between `hc.mix` and `hc.combine`, the head's mix and the untied output.
The forms differ in data only (the indexer's projection fused or split,
the experts stacked or not, projection scales or none). `qwen4exp-mtp`
composes the MTP drafter stage: it reads its target's streams after the
last layer, its token rows from the target's table, one QSA layer
(rollback `truncate`) and MoE, the head's mix returning its streams as
the next feature, and its own selected head where the artifact has one.
Both mappings' values come from today's profile, binding and hash
(`Qwen38V0Values`, `Qwen38MtpV0Values`); their default policies (G3b)
are today's fast form, by the import's form. The drafter's head reads
its weight's first rows (`head.linear`'s `rows`: the draft vocabulary,
65,536 or a selected head's), and the verify's and the drafter's tokens
are GGML's argmax (`native_argmax`). On the installed artifacts
(`components_qwen38_models_test`), the ModelOpt specs equal the unit
fixture's, every import's layers are the same components, the layouts are
today's state, and every step kind builds; the reference executor does
not run them (it binds whole weights: 97 GB a ModelOpt artifact), so
their numerical gate is the lowering against today's runner (G3b).

DeepSeek V4 (slice G4a) extends the mappings again: slots whose count a
key gives (`SlotRule::count_key`, `count_above`: the hash layers, then
the bias-routed ones), one slot instance a layer (`per_layer`: each hash
layer's own `side.hash`, named `hash_<layer>`), string parameters naming
their layer (`{layer}`), a layer type per layer from an integer-list key
(`Mapping::layer_type_key`, `layer_type_codes`: compress ratios 0, 4,
128 to `w`, `c`, `h`), and drafter reads over the layers a key lists
(`ReadRule::layers_key`, `index`, `entering`: the stream means entering
`dflash.target_layers`' layers). The `deepseek4` mapping makes the
target (Sinkhorn mixing, `attention.mqa` per layer type, `ffn.moe` with
hash or biased routing, the head's mix, the untied output) and `dflash`
composes the DSpark stage `dspark` (three window layers over a 256-cell
injected ring, `embedding.inject`, the head's mix of every row, its
target's output and the Markov head; `conf_proj` unbound). Values come
from today's profiles (`Dsv4V0Values`, `DsparkV0Values`). On the
installed artifacts (`components_dsv4_models_test`: UD-Q2_K_XL
`8a355bfb`, the community IQ2_XXS `cd39d504`, DSpark `dd2d3f9c`), the
specs equal the unit fixture's, the layouts are today's, and every step
kind builds, the injection writing every drafter layer's ring.

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
part, which the v1 index holds as one resource of the stacked format with
part `tensor_scale` (D-113).

The native split-row adapter connects only NVFP4 split `GetRows` to that
same ordered step program. `ModelRunner` validates every row-addressable
part against its artifact descriptor before table setup, and requires the
same lookup-order slot map across parts before I/O or state admission.
Codes and block scales remain compressed in bounded device row slots;
the optional matrix scale follows the original format and stays resident.
The adapter accepts only direct checked side-row inputs belonging to that
paged component. Selected sources explicitly opt into capture; this decoder
queues one existing kernel and retains the runner's stable allocations through
completed replay. The bounded synthetic source, changed-ID eager/capture/replay
and malformed-table controls pass on Spark. This is an operation foundation;
native ModelOpt model admission and product/expert support remain open
([evidence](experiments/m36-decode-templates/README.md#native-split-row-foundation--2026-10-12)).

The grouped-operation foundation validates a completed route into caller-owned
worklist storage, preserving original routed-slot order and repeated expert
contributions. Its descriptor helper checks every format part and per-expert
pitch, retains stacked scales, and initializes descriptors even for empty expert
intervals. Both helpers validate before writing their retained outputs; their
bounds describe storage the caller must fund before allocating it. They acquire
no leases and provide no source or residency authority. Main passes the full
native build and all eight affected host controls. A bounded test-only owned
operation now executes the existing grouped decoder under actual selected-expert
leases: both input bases, repeated contributions and changed-routing capture
replay pass on the isolated source and main. The full native main build and
eight helper controls also pass. The production
runner's mid-step routing/admission continuation remains separate work
([evidence](experiments/m36-native-grouped-foundation/README.md)).

Supplemental selected closures now use the ordinary scheduler while retaining
an existing request's base lease. Admission validates the whole closure before
file opens or page-in; materialization and device submission recheck current
catalog facts. A funded owner retains the closure, complete caller payload and
external request counter until its own task and all selected page-ins positively
retire. Failed work cancels the request; unknown completion retains its resources
and prevents continuation or teardown. Existing ordinary steps remain subset-only.
This is the lease prerequisite, not the routed ModelRunner continuation or a new
reclaim policy. Full native main build and 145 focused host controls pass.

The common step program binds approved EXL3 selected operations with authenticated
packed and reconstruction tuning facts. Both 4.0- and 4.5-bpw fixtures execute
in G and O profiles; O uses actual occupancy facts checked again at launch.
All four profiles have original-model corpus observations and same-implementation
checkpoint, spill, kept-file adoption into a fresh owner, and weight backing
reload controls ([evidence](experiments/m36-exl3-selected/README.md)). The legacy
family phase planner and executor are removed; immutable source bindings,
generic kernels and the linear sweep remain. These unserved fixtures have no
speed qualification, OS-process restart or changed-virtual-address claim.

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
algorithms were pinned) and never for a decode-shaped product. Those two
also `needs_tuning`: each slice's GEMM runs only with an algorithm pinned
for exactly its shape, and `exl3-recon-pin.json` pins only the fixtures'
shapes at 145, 1,023 and 1,024 rows, so `Candidates()` offers a
reconstruction only for a shape the query's tuning data
(`TunedShape`, the device's, which the planner passes) lists for it. A
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
A 3inst trellis has no codebook-marker part: upstream names mcg and mul1
by a marker tensor beside the trellis and 3inst by its absence, so the
size rule gives the marker only to those two (2026-10-09, with the EXL3
reader below).

**The affine encoding (2026-10-09).** `format::Encoding::kAffine` is
integer quantization with an offset: unsigned codes q of 1 to 8 bits,
a scale s per group of values along a row, and per group either a bias b
(MLX's affine mode: s · q + b) or an integer zero point z (AWQ's and
GPTQ's arithmetic: s · (q − z)). Its parameters are the bits, the group
(a multiple of 8, so every group's codes are whole bytes; ne[0] a
multiple of it), the scales' element type (F16, BF16 or F32; biases
share it) and whether it has zero points. It is held only in `kSplit`:
a row's codes as one little-endian bit stream, code i at bit i · bits
(MLX's U32 words and its 3-, 5- and 6-bit byte packs are this stream),
K · bits / 8 bytes; then K / group scales; then K / group biases, or the
zero points as a bit stream of `bits`-bit codes padded to a byte a row.
Quantized, it is `Lossless` only into itself, and it is no state
encoding. The CPU reference decodes it as the definition: s · q + b
rounded once to F32 (a fused multiply-add), or s · (q − z); MLX's own
kernels compute in the activations' type, rounding there, so they agree
within that type's rounding. `reference_decode_test` checks hand-computed
values, MLX's CPU unpacking expressions for 3, 5 and 6 bits, every width
from 1 to 8 through the packing, and MLX's documented quantization
(mx.quantize: s = (max − min) / (2^b − 1), b = min) at 2 to 8 bits and
groups of 32, 64 and 128, each value within s / 2. MLX itself was not run.
There is no CUDA decoder yet. What one needs, under the block-decoder
contract: an `AffineSplit` decoder whose Load reads `bits`-wide codes
from the row's bit stream (whole bytes at 2, 4 and 8 bits; a 64-bit
window for 3, 5 and 6) and the group's scale and bias or zero point once
a group, instantiated per bits and scale type where used (MLX's BF16 at
4 bits first) to bound compile time; and a `DotQ8` rung-2 path, since a
group's Σ (s · q + b) · x is s · Σ q · x + b · Σ x, the INT8 product plus
the activation block's sum, which Q8_1's 32-value blocks carry (as GGML's
Q4_1 dot product), for groups that are multiples of 32. AWQ's and GPTQ's
checkpoints pack codes across output columns (AWQ in its own nibble
order) and need a layout of their own or a relayout at import before
their reader can describe them.

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
  deterministic and published by one rename (D-056). *As built
  (2026-10-09, D-113):* [format version 1](artifact-format.md#version-1)
  and its reader: each index entry's tensor format and part in place of
  v0's descriptors, the spec as `spec.json`, `fold` and `relayout`
  records and the importer's identity in the manifest, compositions of
  version 1 that may carry a model's spec; the groups, chunks and page-in
  are v0's, and the reader takes both versions.
- **Admission.** <a id="admitting-a-checkpoint"></a>A checkpoint of a known
  architecture is supported once it passes a standard automated gate:
  logits and perplexity against a reference engine on a fixed corpus, and
  the state, spill and swap controls every model passes. The approved list
  is data. The owner still approves which checkpoints are claimed in
  [model-support.md](model-support.md) (AGENTS.md: agents never invent
  supported combinations).

**As built: the container readers** (`src/import`, 2026-10-09). A reader
turns a checkpoint into a `Checkpoint`: its files, opened as any file is
(links followed; only regular files, a device or FIFO refused before it
is opened and a link to nothing refused rather than taken for an absent
file; who could change them is the filesystem's business, D-073 as
amended on 2026-10-09), each open only while its reader reads it, with
its path, device and inode kept so that a later read opens it again and
refuses (`kIo`) one that is no longer the file read; every tensor
as a `format::TensorFormat`, the part it holds and its byte extent; and
the source metadata as typed key/values. `SourceOf` and `ValuesOf` give
exactly the `spec::Source` and values `spec::Synthesize` reads. The
output is deterministic (files in split or shard-name order, tensors by
offset, metadata by key), errors name a rule, a static reason and an
index, never the checkpoint's own text, and nothing is relaid out. A
reader refuses only what it cannot describe correctly (ambiguous or
inconsistent sizes and extents, overlaps, escapes from the checkpoint,
unknown encodings) and what its security bounds exclude (sizes, counts,
paths); what is merely unusual for its container is read as it is
(owner, 2026-10-09). D-009's protection is this content hardening. One
narrow path rule guards what is copied rather than parsed: a file kept
verbatim (the diffusers reader's kept files now, the writer's tokenizer
and configuration copies when it lands) must resolve inside the
checkpoint (`ResolvesInside`; for a Hugging Face cache snapshot, the
repository's `blobs/` counts), and one resolving elsewhere, such as a
link a cloned repository planted to one of the user's files, is left out
and named in `Checkpoint::skipped`, never refused.

- *GGUF* (`readers/gguf.h`) parses the header with `base/gguf.h`, which
  the artifact's kept-metadata reader and the tokenizer now share: every
  count and length checked against its cap and then the bytes left before
  it drives a loop or an allocation, keys and tensor names UTF-8 and
  unique, no array of arrays, at most four non-negative dimensions. It
  reads each header in growing prefixes (1 MiB, then four times as much;
  all of a checkpoint's headers 256 MiB, the metadata kept from them 512
  MiB and their tensors 2^20 in all). general.alignment is a power of two
  (of any integer type); every GGML type of the pin is named by the
  format layer (removed and unknown types refused), rows are whole
  blocks, no tensor is empty (the format layer has no empty shape); every
  tensor's bytes lie inside the file and overlap no other's. Where
  gguf.cpp is stricter without the description needing it (offsets packed
  in info order at the alignment, names up to 63 bytes, u16 split keys)
  the reader accepts the file, and it ignores bytes after the last
  tensor as gguf.cpp does. The architecture and tensor names are UTF-8
  without control characters (C1's included). A
  split GGUF is read from its first file, `<prefix>-00001-of-<n>.gguf`,
  each sibling's `split.no` its place, the counts agreeing, every name
  once, the tensors `split.tensors.count` when the first file gives it
  (llama.cpp does not need it), and a key a later file repeats equal to
  the first's. A three-dimensional tensor named `*_exps` is an
  expert array.
- *safetensors with config.json* (`readers/safetensors.h`) reads
  config.json (strict RFC 8259, 16 MiB, nesting 32, `model_type` the
  architecture, or a diffusers component's `_class_name` without one;
  objects flattened into dotted keys, typed) and either
  `model.safetensors` or the shards `model.safetensors.index.json` names
  (plain file names, every tensor in exactly the shard the index gives);
  a diffusers component's `diffusion_pytorch_model` files likewise
  (weights under both names: refused, ambiguous). With neither default
  name, the one variant present (`<base>.fp16.safetensors`,
  `<base>.safetensors.index.fp16.json`, as transformers and diffusers
  name them) is read; several variants and no default are refused as
  ambiguous. Shards whose `__metadata__.format` is "mlx" make the
  checkpoint MLX's (below), unquantized ones included; shards of MLX's
  format beside another writer's are refused.
  A shard is safetensors' own format, read as permissively as still
  describes it: the 8-byte length at most 100,000,000, the header an
  object of entries with dtype, shape and offsets (other members, the
  index's other members and `__metadata__` not read), the dtypes of the
  table (the 4- and 6-bit ones refused), and every tensor inside the data
  section and overlapping no other (gaps and trailing bytes, which the
  reference reader refuses, describe every tensor as well). Quantization
  (`readers/hf_quant.h`) is read from `quantization_config` and ModelOpt's
  hf_quant_config.json, which must agree: ModelOpt NVFP4 (`U8` codes,
  `F8_E4M3` scales per 16, an F32 `weight_scale_2`) and MXFP8 (`F8_E4M3`
  codes, E8M0 scales per 32) are paired into `kSplit` formats with their
  parts, alone or per module (`MIXED_PRECISION`'s `quantized_layers`, the
  longest covering module); a scaled weight no module names is described
  when its dtypes and shapes match one of the two (Qwen3.8's n-gram
  table), and an NVFP4 weight without its own `weight_scale_2` takes the
  nearest ancestor module's lone one (the n-gram table's 128 row shards
  share one, which is then the tensor-scale part of each); exclusion
  patterns are matched within a fixed budget of steps, so hostile ones
  cannot stall a read. A scaled weight of an excluded module is described
  by its dtypes; a weight without block scales, even in a configured
  module, and block scales without a weight are dense tensors. Refused,
  naming why: a pair whose dtypes match neither scheme, or not its
  configured scheme, or whose part sizes disagree; and the schemes no
  encoding describes: compressed-tensors
  other than `dense` (its NVFP4 global scale is the reciprocal of the
  tensor scale, which needs a fold), AWQ and GPTQ (the affine encoding
  with zero points, but packed across output columns: no layout yet),
  block FP8, other ModelOpt algorithms, MLX's `quantization` and EXL3's
  method (their readers') and unknown methods.
- *MLX* (`readers/mlx.h`, container `mlx`, 2026-10-09) reads mlx-lm's
  directories as the safetensors reader does, then pairs each module M
  with both M.weight and M.scales (mlx-lm's own test for a quantized
  module) under config.json's `quantization`: the default `group_size`,
  `bits` and `mode`, and per module path an override object (bits and
  group required: MLX's defaults for them differ between its versions),
  `false` or `true`. Affine weights become {affine, bits, group, the
  scales' type, `kSplit`} with M.weight the codes (any unsigned integer
  dtype; MLX writes U32), M.scales and M.biases; mxfp4, nvfp4 and mxfp8
  become the existing encodings in `kSplit` (their bits and group fixed,
  E8M0 or E4M3 scale bytes, no biases; NVFP4 without a tensor scale). K is
  the scales' columns times the group, which every part's bytes must
  match; three-dimensional weights (mlx-lm's `switch_mlp`) are expert
  arrays. Refused: a module with scales whose override says false, a
  missing or incomplete scheme, a mode MLX does not have, a group the
  encoding does not take, and parts of other dtypes, ranks or sizes.
  `quantization_config` (mlx-lm writes a copy) is not read. An
  unquantized MLX checkpoint has no `quantization` and is known by its
  shards' format ("mlx"). MLX keeps its own axis orders, described as
  stored: a convolution is channels-last (mlx.nn.Conv1d's [out, kernel,
  in], a depthwise conv1d [channels, kernel, 1] where PyTorch's is
  [channels, 1, kernel]), a mixture's experts stacked under `switch_mlp`;
  the container tells a mapping which fold applies.
- *EXL3* (`readers/exl3.h`, container `exl3`) reads ExLlamaV3's
  directories (config.json's `quantization_config.quant_method` "exl3")
  and pairs each linear M with M.trellis, as upstream's `LinearEXL3`
  does: I16 [K / 16, N / 16, 16 · k] at a whole 1 to 8 bits (half-integer
  rates refused: no format), F16 M.suh and M.svh of K and N values (the
  packed M.su and M.sv alone are refused: they need an unpacking fold),
  and the codebook by marker presence (M.mcg, M.mul1, neither for 3inst;
  both refused), each part with the trellis's format and its role.
  quantization_config.json is not read; biases, norms and embeddings are
  dense. A config.json with MLX's `quantization` too is refused (which
  describes the tensors is ambiguous).
- *diffusers* (`readers/diffusers.h`, container `diffusers`) reads a
  pipeline as a `Checkpoint` of components (D-089): model_index.json
  (1 MiB, strict; `_class_name` the architecture, every member the
  metadata) names a component by a [library, class] pair of strings,
  whose name, a path component, must be a composition role
  (`[a-z][a-z0-9_]{0,63}`, at most 64). A component directory with a
  config.json is read by the reader that claims it (never as a pipeline
  itself), one to 16 of them, each a checkpoint with its own files and
  metadata; another's regular files (a scheduler's, a processor's; at
  most 4,096 directory entries, subdirectories left out) are the
  pipeline's own files, `<role>/<name>`, at most 256 and within a
  composition's 64 MiB a file and 128 MiB in all, after model_index.json;
  a kept file that resolves outside the pipeline is skipped and named
  (above). A component's directory, its links resolved, lies beneath the
  pipeline's (a link out of it is refused as an escape, `kEscape`; a
  weight component's files may link anywhere, as a Hugging Face cache's
  link to its blobs), and an error from inside a component names it by
  its role's index.

`ReadCheckpoint` asks the readers in order: GGUF for a `*.gguf` file,
diffusers for a directory with model_index.json, EXL3 and MLX for a
directory whose config.json (opened as the reader opens it and parsed
within its caps) names their quantization, and safetensors for any other
directory. A reader holds one file open at a time beyond the few it is
choosing between, so a checkpoint of any number of files (16 components
of 4,096 shards at most, beside 256 kept files) reads within the usual
soft open-file limit.

On spark-b (`import_models_test`), the Gemma 2 and Gemma 3 source GGUFs
give, through their mappings and their own metadata, the spec the
installed v0 artifacts give with today's profiles, and every tensor of
the eight GGUF-sourced M3 artifacts (Gemma 4 26B, 31B and both
assistants, DeepSeek V4 UD-Q2_K_XL in three splits and antirez's, the
DSpark drafter, Qwen3.8 UD-IQ3_XXS in three splits) has the format the
reader gives its source tensor; Qwen3.8's ModelOpt checkpoint (301,730
tensors in 35 shards) reads with 75,419 NVFP4 and 467 MXFP8 weights
paired, 128 of the NVFP4 ones (the n-gram shards) by their dtypes, and
agrees with the v0 import the Python importer made of the same files:
layer 1's MXFP8 linears have the format v0's resources assemble to and
the same bytes, every routed expert's tensor scale is v0's, and the
n-gram shards' rows are v0's row-interleaved table under its one tensor
scale (`Qwen38ModelOptMatchesTheV0Import`).
Qwen-Image-2.1's pipeline reads as the composition the Python importer
made of it (`eca21baa…`): the same architecture, components and their
architectures, the same eleven kept files by name and size, and in each
component every tensor of its v0 artifact in v0's format, all 33.1 GB
byte for byte. Both Qwen2.5 EXL3 fixtures' source checkpoints read as
their v0 imports: 169 trellises each, every part in the format v0's
resources assemble to, every other tensor v0's, and every resource's
bytes (589 and 611 MB). Qwen3.8's MLX checkpoint (3,747 tensors in 22
shards) pairs all 1,020 quantized modules (4 bits, group 32, BF16), the
147 `switch_mlp` ones as expert arrays; decoded by the reference against
the native checkpoint, the head and token table (BF16 there) are within
one MLX step of the native values everywhere checked (at most 0.918 of
the bound) at 0.090 and 0.084 relative RMS, and the shared expert's and a
full-attention q_proj's MXFP8 and routed expert 7's NVFP4 agree at 0.095,
0.089 and 0.124 relative RMS (`DecodesLikeTheNativeCheckpoint`; test
bounds: one step, and 0.15).
**Owned source normalization** (`import/normalize.h`, N3b1). Compiled
`(container, architecture)` tables select the existing Gemma 2, Gemma 3 and
Gemma 4 target GGUF mappings, plus the supported Qwen3.8 Flash-Next
target GGUF profile. A move-owned plan classifies every metadata
key, owns every canonical tensor and copy route, and validates the common
component contracts. Gemma 2/3's `norm.plus_one` is asserted already applied
by GGUF, never executed again; Gemma 4 declares no fold. An absent output
weight binds the embedding through the existing mapping without inventing
or copying a second payload.

Unknown tensors or model keys, duplicate aliases, invalid scalar or
per-layer list types, unsupported container/family pairs, inconsistent
formats or file extents, and overlaps through physical file aliases refuse.
Only declared `general.`, `tokenizer.` and `quantize.imatrix.` metadata
namespaces are retained as auxiliary data. Gemma 4 uses the same mapping's
mutually exclusive dense/MoE slots, local/global layer rules, tied global
K/V and shared rotary table. Expert gate/up and down arrays retain their
whole source encodings and rank-three geometry; this path neither repacks
nor dequantizes them. Gemma 2's 46-layer and Gemma 3's 62-layer variants
remain refused because their alternate query scale is not yet mapped.

`NormalizationBound` checks the closed input domain without new heap
allocations; the caller must cover its checked incremental storage and work
bounds before construction. The source checkpoint remains separately owned
and funded. Copy cursors describe source byte ranges; they neither read
payloads nor publish artifacts.

Focused native checks passed on Spark on 2026-10-11: 70 host controls,
including 16 normalization cases and complete 288/444/658/833-tensor
inventories, plus the real GGUF Gemma 2 2B, Gemma 3 4B QAT, Gemma 4 26B-A4B
and Gemma 4 31B source/artifact comparisons. Each real case compares the
normalized spec with its existing v0 profile and checks complete tensor
formats and source routes. Raw results had no failures or skips; the
independent source review and adversarial challenge were clean. Final Spark
checks passed: 2,516 native, 1,971 CPU and all 114 model tests, with no skips.
Main build and integration checks passed. This covers normalization, not
artifact publication or serving. Assistants, other Qwen/DeepSeek domains,
other containers and remaining transforms keep N3b open; writer and admission work remain
N4/N5.

The additional Qwen GGUF target table preserves all 1,224 encoded tensors
of the supported 48-layer profile, including its side n-gram table, mixing,
GDN/QSA state contracts and 512-expert arrays. Structural scalar/list checks
and storage bounds precede synthesis. Source hash multipliers, offsets and
vocabulary sizes are owned values, not generated constants or exact-value
pins; row ranges must fit the encoded table and I32 lookup domain. The
existing GGUF assertions for norm, SSM and GDN conversion are retained, with
no new payload conversion or claim about a converter invocation. Registering
this table makes the existing CLI normalization and common writer reachable.
The common writer places `side.ngram` with row tables and still checks full
source/destination coverage and its own funding before publication effects.
[The bounded normalization and placement checks](experiments/m36-qwen-gguf-normalization/README.md)
pass with a full native build, 103 affected host controls and one real-header
normalization case. Actual full publication, admission and model qualification
are separate.

**Stored-spec runtime admission** (`runtime/component_admission.h`).
Registration and component construction use the same closed check over
`StoredSpec` and `SourceOf`. N4b1 qualified the approved Gemma 2 2B Q8_0
source: copy-only native tensors, F32 norms, tied embedding/output, complete
component parameters and the existing `norm.plus_one` assertion agree with
its mapping. Execution consumes the stored spec unchanged; no legacy
representation is fabricated. Bound plan and kept-state identities include
the native artifact ID through the bound-spec digest. N4b1 passed 31 focused
host controls, the actual v0/v1 comparison, final 2,544 native, 1,999 CPU and
131 model/timing tests, quality gates and main build/integration checks.

N4b2 adds only the three approved Gemma 3/4 target GGUFs,
using exact source and tensor inventories, stored-spec equality and
authenticated profile facts for Gemma 4's gated policies. Native expert slabs
use `TensorFormat` geometry directly. Its 44 focused host controls, three
complete copy-only publications and all three actual v0/v1 comparisons passed
with zero failures or skips. Each comparison checks 24 full vocabulary rows,
tokens/history, complete initialized state and spill/restore continuation.
Gemma 4 retains prompts-alone and its default prefill sizes; all three targets
execute three actual joined two-owner decode waves per arm. Other native
identities and drafter compositions remain refused; v0 synthesis remains
until N5. Final combined regression passed, as did the main native build,
74 integration controls, all three actual comparisons and quality gates.
N4 and N5 remain open. See the
[admission record](experiments/m36-native-v1-admission/README.md).

**Import transform primitives** (`import/transforms.h`). The historical
ModelOpt import's `norm.plus_one` and `ssm.a_neg_exp` convert explicitly
selected BF16 scalars to F32 using binary64 arithmetic and nearest-even
rounding; integer F32 encoding preserves subnormals when hardware float
conversion would flush them to zero. This is their import-time semantic
conversion, not a runtime weight promotion. A refused call changes no output
bytes; finite binary64 exponential overflow is refused, while F32 overflow is stored
as infinity, as the historical converter does. The source-normalization
tables that select these operations remain to be built.

`gdn.value_heads_tiled` means a permutation from grouped value heads to
tiled order, so output head `j*K+g` holds source head `g*(V/K)+j` and
reads key head `g`. The transform's validated plan enumerates source
bytes monotonically for each part with constant-size metadata: dense
F32, or split MXFP8 codes and their block scales, whose corresponding
head blocks move together without changing the encoding or any byte.
Each part has its own cursor so either can arrive first. Cursor
exhaustion describes complete copy ranges, not successful writes;
the writer must track that separately. No new fold identity is added.
N3b2c extends only this plan to dense BF16 rank-one/two
parameters, preserving their two-byte source encoding. The same per-part
cursor describes complete byte movement; it neither computes values nor
allocates a promoted weight copy. Convolution singleton removal is a separate
caller-owned descriptor reshape. Synthetic complete payload and refusal
controls and independent review passed. Final main checks passed 2,576 native,
2,021 CPU and 135 model/timing cases (182 raw GoogleTest runs), with zero
failures or skips. Format, boundary, header and REUSE gates passed, as did
source, recipe and selected model-binary guards (2026-10-11). This foundation
does not establish a checkpoint's source head order, apply a source transform,
normalize a model, publish an artifact or admit a runtime profile.

The bounded part-route executor (`import/route.h`) applies checked copies,
head-permutation cursors and historical scalar folds with constant metadata,
caller-funded workspace and sink-acknowledged coverage. Cancellation, failure,
reentry and moved-from cursors refuse further writes. The existing writer
uses this executor only for its copy routes. Independent review, 44 focused
controls and isolated checks (2,581 native and 2,026 CPU tests) passed with
zero failures or skips and quality gates. Combined checks passed 2,587 native
and 2,032 CPU cases; main build, 44 integration controls and all 135 model/timing
cases (182 raw GoogleTest runs) passed with zero failures or skips. Source,
recipe and model-binary guards and quality gates passed (2026-10-12).
ModelOpt/MLX mappings, multipart/gather normalization,
source head-order authority and new publication/admission remain open. This
helper does not authorize resident BF16 expansion.

The owned indexed part plan (`import/indexed_parts.h`) checks complete source
dispositions and multipart coverage before assembly. Numeric expert and row
gathers retain physical source order separately from destination order;
shared tensor scales, descriptor reshapes and encoding-preserving head
permutations use the same checked routes. Allocation-free preflight charges
new planner storage before taking checkpoint ownership. Eight full-payload
and refusal controls, 2,595 native and 2,040 CPU cases and quality checks
passed with zero failures or skips. Main build, eight integration controls
and all 135 model/timing cases (182 raw GoogleTest runs) passed with zero
failures or skips. Source, recipe and model-binary guards and quality gates
passed (2026-10-12). Geometry validation does not authorize a source mapping,
authenticate checkpoint bytes, publish an artifact or admit a model.

A separate closed ModelOpt target normalizer (`import/modelopt_qwen38.h`)
uses an owned indexed part plan. It validates the complete source set with
finite role/index grammars, including explicitly classified MTP, vision and
activation-scale entries, and reads exactly 280 bytes of pinned PLE semantic
parameters. Canonical parts bind through a native declarative Qwen3.8 mapping
and the shared component validator. Numeric gathers preserve split NVFP4
expert matrices and the ngram table; one inherited table scale remains one
part, and expert tensor scales bind to the weight role exactly once. Dense
BF16 parameters remain BF16 with explicit transient norm/log-A computation;
the plan describes byte-preserving BF16 and MXFP8 head permutations. Only the planned
head-order transform contributes a fold; historical converter assertions and
execution policies are absent. Unknown roles, indices, part coverage, source
formats or configurations refuse normalization. This API produces an owned
plan, not a published artifact or runtime admission. Independent challenge, 26 main focused controls, 2,599 native and
2,044 CPU checks, all 136 model/timing cases (183 raw GoogleTest runs) and
quality gates passed with zero failures or skips (2026-10-12). The confined
actual-source metadata control produced 1,886 canonical parts over 1,212
logical outputs; it read only the 280 semantic bytes. The writer qualification
below separately establishes complete source authentication and execution of
the planned transforms. The bounded
[source inventory](experiments/m36-modelopt-normalization/source-inventory.json)
records all 139 observed descriptor patterns and the separate semantic receipts;
its diagnostic categories alone do not qualify the mapping.

The native writer (`import/writer.h`) uses one placement and publication
path for owned copy-only plans and closed ModelOpt indexed-part plans. It
checks complete canonical bindings and destination coverage, streams each
whole source member through its checked route, and scatters expert slices
without decoding resident weights. Physical source order is independent of
numeric destination order. Metadata funding covers member/cursor/placement
storage before allocation or filesystem mutation. Semantic parameters are
reauthenticated from the same source stream, alongside parsed-prefix and
supplied whole-file digest checks. Planned fold indices become canonical
applied-fold receipts only after all routes and source checks finish.
Stored-spec validation, complete file/chunk readback, durable publication
and verified reuse remain common to both inputs. All 21 focused host controls
passed with zero failures or skips, including independent full-payload
multipart/permutation/expert-scatter comparisons and malformed-owner,
semantic-mutation and unpaid-work refusals. Final Spark checks passed
2,601 native, 2,046 CPU and all 137 model/timing cases (184 raw GoogleTest
runs), with zero failures or skips; main built and passed the 21 integration
controls and quality gates (2026-10-12). Actual publication authenticated
all 38 source files (105,872,821,791 bytes), preserving 103,797,070,596 weight
bytes in 25 verified files and 66,261 chunks. An independent destination-order
oracle compared every byte of all 468 permuted parts (2,153,140,992 bytes);
complete source hashes and writer readback cover the remaining payload.
The 8 GiB, zero-swap run completed in 31 minutes, reached its memory cap
and incurred no OOM kills. This API extension
does not admit a runtime model or import MTP/vision components.

The `llmp import` CLI uses the registered Gemma and supported Qwen target GGUF paths and dispatches
safetensors sources declared as `qwen3_8_flash_next` with
`modelopt MIXED_PRECISION` to the closed native ModelOpt target normalizer.
That dispatch is not admission authority: the normalizer still validates the
complete source contract and refuses unsupported inputs without falling back
to another mapping. Existing flags, ordinary/archive source verification,
parsed-prefix receipts, publication/reuse results and post-publication archive
events use the same writer flow. The requested metadata ceiling covers both
the owned plan and the writer, with reader storage bounded separately.

Independent challenge and main checks passed 22 import integration controls,
2,602 native, 2,047 CPU and all 137 model/timing cases (184 raw GoogleTest
runs), with zero failures or skips; quality and source/recipe/binary guards
passed (2026-10-12). Confined actual CLI checks refused normalization at
1 MiB and writer funding at 512 MiB without creating a store. Funded ordinary
CLI publication then used its default 4 GiB metadata ceiling: every one of the
38 complete source hashes in the published receipts matched the fixed pins,
all canonical bindings and expert geometry passed, and the independent oracle
matched all 468 full permuted parts. Common writer readback verified every
artifact file and chunk before publication. The same 8 GiB, zero-swap hard
domain paid the process and file cache. This ordinary CLI proof checks the
fixed source hashes after publication; archived input additionally enforces
its supplied hashes during the source stream. No new ModelOpt archive event,
runtime, MTP, vision or migration admission is established by this control.
See [the writer contract](artifact-format.md#native-writer).

Still to come: AWQ's and GPTQ's layouts, the complete fold catalog (MLX's and
EXL3's sources need theirs: MLX's stacked experts and moved convolution
axes, EXL3's packed sign vectors), the remaining writer normalization
domains and the admission gate.

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

**As built: repeatable drafter progress** (G3c2a). Components may declare
cache/token/feature coordinate relations and a repeat-pending policy.
Its fixed 64-byte I64 control is ordinary cataloged state, with layout-bound
validation and completion-owned publication and hydration. Accept, whole
discard, prompt companions, checkpoint restore and fresh-runner adoption
preserve the declared pending schedule. Private draft writes require exact
owner/unit permits and checkpoints cover their bounded backward replay
tail. Version 1 contracts coexist with the opt-in version 2 MTP embedding
and replayable QSA contract; default mappings and serving remain unchanged.
The shared runner derives side-row IDs from authenticated token history
before admission, with explicit rows authoritative. Native component sources
remain unqualified for multi-owner repeat-pending execution; the shared
entry points refuse it before state mutation. Joined arithmetic and routing
qualification is a required follow-on before serving recipes or migration gates: original-model
quality is primary, legacy comparisons diagnose differences, and state
restoration must reproduce the same selected implementation. The bounded
selected-HC recipe screen (G3c2b1b1) is evidence for a default-off native
recipe correction only: its 09c report found equal K1/K3 head inputs and logits
across all 47,172 vocabulary logits per owner, while its four K3 products had equal
weights, consumed inputs and outputs across the compared columns. The earlier
08 report found unequal K3 product outputs despite equal weights and consumed
inputs. These are bounded operation observations; branch names are
source-resolved from bound fields and immutable compiled source, not new
dispatch instrumentation. The cleaned correction also passed a model-free
`hc.mix` regression through the canonical output-row producer and normal
`StepProgram`: complete mixed outputs, ragged owners and changed-input replays
of the same recorded program. This component-only test is separate from the
full-vocabulary diagnostic and does not cover target/drafter state. Joined
state, full continuation replay,
persistence and admission remain open under G3c2b1b2. See
[the selected-HC evidence](experiments/m36-selected-owner-hc/README.md) and
[the model runner](engine.md#the-model-runner) for the state, completion and
checked numerical scope.

Kept state written by today's runners (D-105) is keyed by layout IDs such
as `gemma3-4b-f16-kv-device-v1`. Those IDs are bumped and old kept state
discarded rather than migrated; per-model settings and calibration
(D-103) are re-keyed from family to spec.

**As built: the serving adapters** (2026-10-09). `runtime/engine_llm.h`
and `runtime/engine_image.h` are the one adapter of the LLM and image
pipeline kinds over the shared runner ([engine.md](engine.md#the-serving-adapter),
[runtime-serving.md](runtime-serving.md#engine-backed-models)), every
`Llm` unit of work mapped to the runner, the wave composer, prefill steps
and the decoding orchestrators, every state verb to the slot lifecycle,
settings from D-103's layers. `Server::Make` dispatches a family to them by
its mapping's pipeline kind once a diagnostic set of architectures names
it (Gemma 2's, Gemma 3's and Gemma 4's by default since their wiring); an LLM's parts
are its synthesized spec, its components' layout and batching and a
`ComponentStepSource` under the mapping's policy over a `GgmlBackend` on
its own stream, with cuBLAS over a workspace the runner maps
(`RunnerConfig::kernel_memory`) and device greedy tokens, built for
Gemma 2, Gemma 3 and Gemma 4 (with its assistant) so far. The kept
layout is `model::KeptLayoutId` of `spec::BoundDigest` and every stage's
layout, and the kept record gains format version 2, a spill file per
stage (D-112); a model refuses the other version, so the M3 runners' kept
state is discarded at a family's switch. The image adapter is a shell
until Qwen-Image moves: it registers and pages its runner, and its work
refuses. `engine_llm_test` runs `Llm` sessions over it on the fake
backend: prompts, joined prompt and generation waves, greedy device
choice, speculation with two cascaded drafters, spill and restore, a turn
checkpoint's restore, recovery, and a restart adopting a version 2 record
while refusing a version 1 one.

**Cold kept-state hydration (G3c2b2b).** An adopted prefix may be eligible
for reuse before its repeat-pending drafter progress has been read from disk.
That eligibility requires the current owner's replacement ticket and no
active restore, borrow, verification or owed work. The adapter hydrates the
saved state before execution; execution readiness remains separate from
reuse eligibility. A partial restore capacity refusal preserves the prefix
and ticket for retry. Native build and 77 focused host/fake controls pass on Spark B,
including complete state/hash preservation without extra model steps and
independent restoration of cold peers. These controls do not qualify GPU
arithmetic, served performance or the joined-family gates.

**Qwen3.8 sole-owner serving (G3c2b2a).** An internal, default-off diagnostic
route composes native Cutlass and GGML experts, GGUF and the native MTP replay
stage through this adapter. It requires one actual slot and keeps joined
repeat-pending execution disabled. Four focused Spark comparisons pass:
depth-two MTP full logits, the default depth-three token schedule, native
GGML experts and GGUF plain decoding, each including a continuation turn.
Two startup ownership/refusal controls, 2,500 native non-model tests and
1,955 CPU-only tests pass. All 124 model/timing registrations pass, including
two authenticated selected-head fixture reruns after the full run skipped them.
The full-pass drafter's configured depth is deducted from usable context;
ordinary and disabled drafters keep their existing context. Broader state,
quality, swap, cohort and matched
1% gates remain open; the legacy runner stays in place.

The vendor-free `engine/execution_contract` value types separately seal an
ordered numerical-domain declaration and execution inputs (effective policy,
pattern and registry identities, device/tuning facts, declared backend recipe,
build and selector identities, and allocation guarantees). They own their
inputs and reject incomplete declarations. These values do not discover
companion work or admit an operation: authoritative production backend
declarations, source/plan/kept integration and complete operation admission
remain required before a component can require this contract. No production
adapter declares that identity yet, and the model spec identity is unchanged.

Diagnostic snapshots now use `runtime/SnapshotStore`, one bounded,
catalog-charged pinned allocation for initialized state plus aligned captured
ranges, token history, cursor, layout, optional execution identity and decoding
metadata. A move-only capability names the completed capture's store,
generation and live branch issuer; it exposes no writable state pointer.
Foreign, stale, incomplete and incompatible restores refuse before history
funding, settlement or state copies. The server reuses this allocation and its
existing reclaim retry; its diagnostic hash authenticates the capability and
covers only initialized payload bytes. A failed copy retains the entire
allocation and charge through the node's completion-aware pinned retention and
prevents reuse. Any failed release also permanently disables the store: the
node retains the allocation only if it still owns it, since a bookkeeping
error can follow the physical free. Explicit node-lifetime invalidation also
protects stores that outlive teardown. Legacy direct raw-buffer APIs keep their existing behavior
when no execution identity is declared; a declared or unavailable identity
refuses those APIs. This snapshot protocol is not numerical-domain admission
or production backend qualification.

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

**As built** (`planner/partition.h`, 2026-10-09; CPU only, two-Spark
execution is M4's). A component's split options are data
(`components::SplitOption`): the weight roles and state tensors a split
divides and along which dim, the collective that joins the parts, the
units the ways must divide (attention's KV heads, so a query head's KV
head is on its device under GQA; null where only the split roles' shapes
and encodings bound it, which keeps every part whole blocks of its
encoding), and the bytes of the tensor that collective joins per step.
The catalog declares attention by heads (Q, K, V by their rows, the output
product by its columns, both caches by their heads, an all-reduce),
`ffn.glu` by its hidden columns (an all-reduce) and `head.linear` by the
vocabulary (the logits gathered). A future component declares only that
table; the test suite's routed-expert block (`moe.test`, from the op set's
routing ops) declares experts placed whole, then columns inside each
expert, and nothing else. The embedding has no split (a vocabulary split
would need masked lookups and a reduction; nothing needs it while the
table fits each stage's devices whole).

- **Topology and workload.** A `Topology` is devices with the bytes they
  give weights and state, and links with bandwidth and latency, any count
  up to 16; a `Workload` is the prefill and decode rows, the share of
  decode steps, and the context and slots the state is sized for. The
  measured two-Spark link (environment.md: 22.35 GB/s NCCL SendRecv, about
  1.4 µs one way) is the test suite's description of the Sparks; the
  application has none built in (D-019).
- **The traffic model.** A collective over a group runs as a ring in the
  group's order: an all-reduce is 2(g − 1) steps and a gather g − 1, each
  moving 1/g of the tensor over every ring link and as slow as the slowest
  (bytes over bandwidth plus latency). With a head, what follows the
  narrowing (`components::NarrowPoint`, the point `BuildStep` narrows at:
  the last layer's feed-forward block after a narrowing attention, the final
  norm and the head) joins its output rows, not the step's rows; the
  narrowing component's own all-reduce stays at the step's rows, since the
  gather follows the output product and a partial sum is completed before a
  row gather. A tensor leaving its group goes to
  each device that needs it from the holder with the cheapest link,
  receivers holding it too, so a stream crosses a slow link once and then
  spreads over the fast one. The objective is the decode-share-weighted
  modelled time of a prefill and a decode step; bytes are reported too.
- **Memory.** Each device holds its components' weights (a tensor bound by
  several components once, a split part at its own format's bytes) and
  their state for the workload's slots at its context. Activations,
  workspace and graphs are not counted: a device's described bytes must
  leave room for them.
- **The choice** (`Assign`). A pipeline cuts the layers into contiguous
  ranges, each on a group of devices; the embedding runs with the first
  range, the final norm and the head with the last. Within a stage of
  several devices every layer component splits by its first option of
  least modelled traffic that fits the group (a stage of several devices
  exists only because its layers need splitting), and one without a fitting
  option runs whole on each device; the embedding, final norm and head run
  whole where that fits and split otherwise. The search is a dynamic
  program over (next layer, previous group, devices used) with the stage's
  group and end as choices, exact over every pipeline and grouping under
  those per-component rules; devices that are twins in the topology (equal
  memory, equal links to every other device) are taken lowest first, which
  keeps two nodes of identical devices small, and a group's ring runs in
  twin-class order so that pruning loses nothing (in plain device order a
  twin swap changes the ring, and the pruning would miss cheaper groups).
  Bounds on its candidate groups (4,096: thirteen devices without twins
  are past it), states and work refuse a topology too irregular to search
  quickly; `AssignPipeline` still evaluates a configured placement on it.
  Ties go to fewer devices, then fewer stages. Compute is not modelled:
  under a traffic objective one device wins whenever the model fits it,
  pipeline cuts beat tensor splits (one stream transfer a boundary against
  two collectives a layer), and tensor parallelism appears where a layer
  does not fit a device, inside the fast-linked groups, with the pipeline
  cut across the slow link (the survey's TP-inside, PP-across). A layer's
  memory in the search is summed without cross-layer sharing (conservative);
  `AssignPipeline` evaluates the chosen pipeline exactly, and evaluates a
  configured one the same way. An `Assignment` names each component's
  devices (rank order) and split option, each layer's stage, the weight
  aliases (a tied head reads the embedding's leaf), each device's bytes and
  the modelled traffic.
- **The rewrite** (`Partition`). From the primitive graph (before
  patterns) to one verified graph per device: each node runs on its
  component's devices, and a split propagates from the leaves its option
  divides through the ops that keep it (views and reshapes by their byte
  layout, permutes, elementwise ops with broadcasting operands sliced
  alike, norms over unsplit rows, rotations over heads, products by their
  operands' splits, stores into split caches, attention by heads, the
  softmax and row reductions, routed products). A row-parallel product's
  partial sums stay partial through sums, scalings and the expert
  reduction, and an all-reduce completes them where something else reads
  them (before the post norm or the residual add); a split tensor read
  whole is gathered; a tensor read on devices outside its group is sent
  and received (a pipeline cut). Expert parallelism keeps routing whole on
  every device: each device's routed products run its own experts for
  every slot, ids mapped to its own by a constant table (other experts'
  slots computing with its first expert), and the weighted sum masks the
  other experts' slots by a constant 0/1 scale, so each sum is a partial
  sum the all-reduce completes. Each device still runs every routed slot's
  products (others' with its first expert, then masked): the split divides
  the experts' memory and weight reads, not the routed compute, and a
  masked slot's product must stay finite. Dispatch needs no all-to-all
  while the routed tokens are whole on every device; an all-to-all
  dispatch pays only once attention splits by rows (data-parallel
  attention), which no component declares yet, and the op set's
  fixed-size all-to-all would need worst-case capacity. An op a split
  cannot pass gathers its sources and runs whole on each device of the
  group (attention or a softmax under ALiBi, whose slopes follow the
  global head, and a mask broadcast over heads that a part's heads would
  read differently, among them); a store into a split cache or an
  expert-local value that cannot pass is refused, as are fused nodes
  and collectives in the input. Inputs, constants and what is made from
  them alone (masks, positions' views) are made on every device that reads
  them. Each device's leaves name the primitive graph's leaf and the slice
  they bind (part r of n along a dim); a part of a weight the device holds
  whole is a view of it.

The tests (`planner_partition_test`, CPU) run a six-layer Gemma 2-like spec
(GQA, sliding and global layers, sandwich and Q/K norms, softcaps, tied
and untied heads) over a two-owner prefill and a greedy decode with its
state carried, and the routed block, partitioned over the reference
executor's simulated devices against the unpartitioned graph: four ways by
heads and columns (with the head split by vocabulary), a four-stage
pipeline (bit-identical), TP2×PP2, and on six devices TP6, TP3×PP2,
TP2×PP3 and a six-stage pipeline; experts placed whole and columns inside
experts over four and six. Logits, greedy tokens and every cache agree
within the F16 caches' rounding where partial sums reorder, and each
partition's collectives carry exactly the bytes and modelled time the
model counts. Assign's choices match hand-derived optima: one device when
the model fits; the fewest pipeline stages on uniform links; TP2 inside
each of two fast-linked pairs with the pipeline cut across the slow link;
experts placed whole; a split refused where its parts would break a Q8_0
block; Gemma 2 2B on the measured two-Spark link on one Spark, or a
two-stage pipeline when each holds 70% of it; refusals for memory no
placement fits. A brute-force enumeration of every pipeline and grouping
on random topologies with twins (a routed block on six devices, the tiny
Gemma on four) agrees with the search's optimum; sixteen devices without
twins are refused at once; attention under ALiBi or with a mask whose
broadcast over heads a part cannot follow runs whole and still equals the
unpartitioned graph; TP3 and TP8 (where twelve KV heads keep attention
whole beside split feed-forward blocks) equal it too.

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
  original source-precision model, secondary same-format comparisons, state
  and swap controls, matched speed. Reference availability is recorded.

The common target-only corpus scoring foundation is
`tests/support/corpus_score.{h,cc}`. It reuses the shared scored orchestrator,
validates full F32 vocabulary heads and packed owner output ranges before
execution, and counts each next-token target across chunk boundaries. Its
finite host charge pays its IDs, copied histories, outputs and scratch;
model/compiler/state/device funding remains the harness's responsibility.
Ten host controls cover analytical NLL, ragged joined owners, invalid heads,
admission refusal, sink refusal/cancellation and unknown retirement. A synchronous
per-target sink supplies bounded document reports; the caller funds retained
records. Qwen2 FP16 and EXL3 controls use the real scored backends and a common
selected-plan recorder. FP16 uses eight-row chunks; the approved EXL3 tuning uses
seven-row chunks with explicit refusal for absent tuning. They are test support,
not production linking or a completed model quality gate. The
[original Qwen2 BF16 smoke](experiments/m36-original-qwen2/README.md) records
an authenticated 43-token smoke and four independent 256-token passages with
1,020 aligned original/baseline/fused-attention targets and bounded per-target
reports. The FP16 fixture adopts the observed small attention tradeoff with
exact same-plan 76/577-row state controls. Final main integration passes six host
controls and twelve actual historical/selected-state cases without failures or
skips. General quality and serving qualification remain open; random-token
trajectory fixtures are numerical stress controls,
not language-model perplexity evidence.

## Migration and its gate

Each family moves to the new engine with its current implementation as the
baseline:

- **Speed**: prefill and decode, solo and at its qualified cohorts, no more
  than 1% slower than the current implementation at worst, preferably at
  least as fast. This is stricter than D-085's coarse 10% check and needs
  matched bookends that resolve 1%: alternating old and new binaries on the
  same host, repeated until the spread is well under the gap judged, with
  the protocol fixed before the first run.
- **Quality**: compare the candidate and migrated baseline to the same
  original unquantized model for prospective numerical improvements (owner,
  2026-10-12); candidate quality must be equal or better, with the family's
  existing behavioral controls and secondary same-format diagnostics.
  Freeze reference provenance, coverage and acceptance bounds before new
  qualification; disclose missing primary evidence. Bit-identity with the
  old code is not required. Investigate unexpected differences without
  imposing its reduction order, rounding or kernel choice. Within one
  selected implementation, continuation/restore remains exact. See the
  [retrospective audit](experiments/m36-numerical-audit/README.md); historical
  results and the speed protocol remain unchanged.
- **Memory, swap and state**: the existing controls pass unchanged.

Owner answers that shape the migration's work (2026-10-08, 2026-10-09):

- First migrations reproduce each family's current pattern and fusion
  selection (bit-identity is expected where op order is unchanged);
  default-off diagnostic options and benchmark-only probes tied to deleted
  runners are retired, not ported, unless the production recipe, a gate or the
  swap table uses them. This establishes an initial comparison arm, not a
  permanent arithmetic constraint: the owner's 2026-10-12 clarification
  permits better qualified fusion/kernel choices without reproducing old bits.
- "N drafters" means both a cascade (drafter *i* extends drafter *i*−1's
  proposal, one verify) and a per-unit choice among alternatives, picked by
  the alias's policy or adaptive logic (D-068); fake-backend tests cover both
  with two drafters.
- The per-family `ServingOptions` flags are diagnostics: they may be removed
  or replaced freely, with a CHANGELOG line and no decision entry.
- `ggml_cgraph` exists only as a private lowering target inside
  `src/kernels/ggml`; the IR is the source of truth. The engine core builds
  CPU-only with a fake device runtime (amends D-096).
- Work lands as reviewed sub-task commits, main always building and passing
  ([workflow](workflow.md#ground-rules)). The importer's admission gate uses
  the MLX checkpoint `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa80`.

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
