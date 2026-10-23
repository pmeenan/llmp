<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to llmpalooza are recorded here, in the format of
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Llmpalooza
follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html) and
stays at 0.x, where a minor release may break compatibility, until 1.0
(D-062). A change to a public surface names its version bump here.

## [Unreleased]

**M3.6, engine components (2026-10-17).** One engine of shared components
over llmpalooza's own graph IR now serves every model M3 served (Gemma 2, 3
and 4 with their assistants, Qwen3.8 native with MTP and GGUF, DeepSeek V4
with DSpark, Qwen-Image-2.1), within the matched 1% speed gate of M3's
speed, bar the accepted exceptions listed in the record; a native importer (`llmp import`)
publishes the format-v1 artifacts they serve from, and the per-family
runners, plans, graph builders and serving classes are gone (the Python
importers of the experiments are no longer in the production path).
The entries below record each change; the record of the milestone is
[docs/m36-record.md](docs/m36-record.md). Re-import any v0 artifact (D-118).

### Added

- HF block-scaled FP8 imports retain E4M3FN codes and F32 scales with explicit
  block geometry, CPU reference decoding and validated stored specs. Device
  execution support remains separate (M3.5 U3b).

- DeepSeek V4 Flash 0731's native target spec retains its GGUF sampling
  defaults (`temperature=1`, `top_p=1`) with unchanged indexed payload and
  refreshed target/DSpark composition identity
  ([proof and remaining numerical gaps](docs/experiments/m35-generic-import/deepseek-sampling.md)).
- ModelOpt MXFP4 weight imports preserve packed codes and block scales through
  the generic importer. Qwen2, Llama and Mistral share one canonical mapping
  per architecture across GGUF, HF and EXL3 source dialects
  ([checks](docs/experiments/m35-generic-import/dense-dialects-mxfp4.md)).

- GB10 scalar quantized products use incumbent reduction schedules across all
  18 VecQ encodings. Operand controls and the bounded Gemma31 scalar quality
  transfer pass, with a matched C1 speed check within the 10% adoption bound;
  broader assistant, joined-row and frozen-state qualification remains open
  ([results](docs/experiments/m35-component-numerics/gemma31-scalar/RESULTS.md)).

- Stored drafter execution contracts describe anchored, block and MTP protocols;
  serving derives structure and selected-head ownership from the stored graph,
  with optional performance defaults and actual row/state bounds. Four native
  drafter publications installed on both Sparks preserve complete encoded
  payloads and carry their own execution specs; borrowed target roles rebind generically across names and
  split parts. Five integrated model controls pass; three frozen numerical cases
  reproduce their legacy observations without changing goldens or qualification
  ([proof](docs/experiments/m35-stored-drafter/RESULTS.md)).

- Gemma 2, Gemma 3 and Qwen3.8 GGUF publications carry explicit source-fold
  provenance for spec-based serving admission. Complete index and tensor bytes
  are unchanged; Qwen also stores its existing source sampling defaults
  ([proof](docs/experiments/m35-source-fold/RESULTS.md)).

- SHA-256 uses runtime-selected ARM SHA2 instructions where available, with a
  portable scalar fallback and unchanged digests, artifact IDs and verification
  passes ([bounded hash screen](docs/experiments/m35-sha256-arm/README.md)).

- Opt-in quality runs retain raw streamed replies and timing records, support
  concurrent fixed prompts, and join all started requests before reporting a
  failure. A failed reply stops likelihood collection.

- LLM startup shares admitted artifact metadata and composed structure across
  settings and runner construction; retries retain successful preparation while
  each build keeps its own funded state and execution checks (M3.5 U3a).

- Shared GDN gate fusion accepts positive head and row counts within checked
  storage bounds, preserving its original pointwise arithmetic (M3.5 U7a).

- Image imports accept explicit `--reuse-component ROLE=ARTIFACT_ID` selections
  after matching their stored spec, tensor inventory and complete archive source
  receipts, avoiding another payload pass for unchanged prepared components.

- Image flow and VAE requests execute stored stage topology with checked
  geometry bindings, preserving alternate graph wiring from the artifact
  (M3.5 U8, D-122; affected image artifacts require re-import).

- Image components own their spatial construction and operand contracts, so
  adding a spatial definition no longer requires compiler kind-name branches
  (M3.5 U8; stage wiring and quantized image support remain in progress).

- Fusion/decomposition comparisons now bind exact queries, numerical contracts
  and compiler-owned implementation provenance; stale or absent evidence retains
  incumbent selection (M3.5 U6, coverage still in progress). The first twelve
  RMSNormMul observations remain in an unchanged historical archive; twelve fresh
  GB10 exact-query comparisons are independently audited (eleven fusion wins, one
  neutral), with native live-registry/planner consumption verified.
- IR operation traits select typed parameter, inference, decomposition and
  partition handlers, with exhaustive dispatch controls and indexed fusion-rank
  lookup (M3.5 U6).

- Chat Completions tool intake, generated calls and result continuation through
  the shared format-driven parser and renderer, with canonical stable-span
  forcing in the common engine (M3.5; real-model/client qualification pending).
  Gemma missing-ID histories now resolve unambiguous call names instead of
  matching every result to the final empty-ID call; ambiguous histories fail.
  Call-only assistant history accepts omitted content. Tool names that cannot
  round-trip through the selected template fail before generation.
  Tagged tool formats can declare structural aliases without changing raw
  argument values. Open calls exclude premature stop IDs, and exact tool
  markers continue past legacy route-only stops.

- Owned structured conversation parts with bounded text/template projection
  (M3.5 U5b); images and the Messages route remain disabled.

- Route-owned streaming encoders for OpenAI-compatible and Messages events
  (M3.5 U5a); the Messages request route remains disabled.
- Generic architecture-owned normalization for native quantized tensor parts,
  expert gathers and shards (M3.5 U3, D-125), with import-time role-based
  transforms, mapping provenance and stored drafter specs. Llama 3 frequency
  scaling is represented through attention and rotation components.
- Physical memory modes (D-124): `[memory] mode = "dedicated" | "shared"` and
  `budget_gib` configuration support. In dedicated mode, physical memory handles
  are pre-allocated synchronously up to the unmapped budget headroom into a
  physical backing pool (`BackingService`) at node start, eliminating hot-path
  driver allocation latency and recycling handles on eviction, with unmapped
  handles separated from virtual catalog extent charges. In shared-machine mode,
  unmapped handles are not pre-allocated, physical backing is allocated on demand
  and released on eviction, workspace memory is pageable by default, two automatic
  reclamation levers (idle timeout and external `PressureTrim`) drop resident
  models and trim node budget/heap, and loopback clients can proactively trigger
  immediate working set release via `POST /llmp/v1/memory/release`.

### Fixed

- Dense scalar Q8 products on GB10 use the incumbent four/eight-warp reduction
  schedule, including joined one-token rows. The registered Gemma2 screen
  exactly matches all 1,070 baseline teacher scores; other qualification remains
  open (M3.5 U4).

- Native artifact and composition publication creates private files (0600) and
  directories (0700), preserving existing store and input permissions.

- Adding or removing unrelated reports no longer rebuilds unchanged compiler
  inputs; current source ownership and generated-input checks remain enforced.

- Refresh Gemma26 and Gemma31 target and paired spec metadata with their source
  sampling defaults, preserving all encoded payloads and assistant artifacts.

- Record the failed Gemma2 scalar numerical transfer screen and reopen its
  qualification, preserving the registered bounds and frozen expectations.

- Chat startup retries reuse paid construction capacity, preserve previous assets
  and reasoning settings on failure, and fund reader workspace before loading
  tokenizer or template data. Replacement admits both old and new live assets.

- Refresh the two EXL3 fixture artifacts with source sampling defaults while
  preserving all encoded tensor parts and source identities (M3.5 U3).

- Align the Qwen ModelOpt MTP approval record with its equivalent individual
  expert-scale declarations, preserving the selected head and source receipts.

- Match drafter recipes against the resolved canonical target architecture, so
  source aliases remain compatible during import, admission, construction and
  calibration. Fresh drafter component artifacts record their canonical execution
  architecture while retaining authenticated checkpoint provenance; re-import only
  drafter artifacts previously published under their parent target architecture.

- Scalar generation keeps its completed conversation prefix after a clean
  dispatch refusal, so a retry can resume without clearing proven state.

- Substitute model cohorts time-slice and preserve request continuations,
  allowing the original and other waiting models to progress (M3.5 U5c).

- Preserve Gemma 4 native RMS weights as direct scales instead of applying
  the Gemma 2/3 plus-one convention. Affected native artifacts need re-import.

- Preserve Qwen3.5-family GDN SiLU output gates through shared component graphs
  and fused kernels; keep Qwen3.8's sigmoid contract and numerical grants distinct.

- Accept both lists and named maps of Hugging Face extra special tokens, with
  the same vocabulary lookup and matching validation.

- Qwen3.5-family MLX imports preserve converter-folded RMS scales instead of
  adding one twice; Hugging Face zero-centered norms retain their import fold.
- Swap reports follow each activation's asynchronous releases across
  same-model requests, cancellation and overlapping model switches (M3.5 U10).
- Kept-state and process-lock files pin existing regular objects before I/O
  opens, including replacement identity checks; exclusive lock creation keeps
  dangling-link behavior without opening a substituted device or FIFO (M3.5 U10).
- Idle conversation and turn-checkpoint retention defaults to seven days,
  preserving state across an idle weekend within the configured spill budget
  (M3.5, D-055).
- Source-only native drafters receive startup metadata funding while target
  bindings continue to require a stored spec (M3.5).
- Direct artifact reads and buffered retries open the same pinned regular
  file, preventing pathname replacement during selection (M3.5 U10).
- Drafter continuation features survive ordinary state spill and restore
  while logical state replacement still clears them (M3.5 U10).
- IR byte addressing uses the format layer's physical block geometry, with
  vendor and device facts checked at neutral module boundaries (M3.5 U6c).
- Checkpoint and ordinary artifact readers pin regular files before opening
  them for I/O, closing pathname replacement races (M3.5 U10).
- Kernel selection scopes measured ranking data to its providing backend and
  includes that data in plan-cache identity (M3.5 U6b).
- The GGML backend validates original operands as well as fused plans, so a
  fusion cannot hide dimensions outside the backend's supported range; the
  shared IR now gets operation traits from one table (M3.5 U6a).
- Test-only paths and v0 artifact reader retired (M3.5 U2): `PlanGraph`
  (`kernels/ggml/graph_plan.{h,cc}`) and `Qwen38Commit` removed with their
  tests and `benchmarks/decode_templates.cc` moved onto the production IR
  lowering and component runners; the v0 artifact/composition reader
  (`artifact/representation.{h,cc}`, `artifact/format_map.{h,cc}`,
  `V0ArtifactSource`, `tests/unit/artifact_corpus.py`) removed and all model
  fixtures, tests and gate scripts (`speed_gate.py`, `swap_table.py`,
  `assistant_gate.py`, `k1_quality.py`) moved onto v1 artifacts; the layout
  profile renamed from `"spark-v0"` (version 0) to `"spark"` (version 1, D-121);
  `LLMP_TEST_HOLD_READS` retired from the production binary (`NodeSettings::hold_reads`
  remains for unit tests); the repeat-pending drafter ledger removed; and
  `llmp_import_fuzz` extended
  through `spec::Synthesize`, `import::Normalize`, `planner::Plan`, and v1
  artifact/composition opening with a seed corpus in
  `tests/unit/data/import_fuzz/` (`llmp_import_fuzz
  --corpus=tests/unit/data/import_fuzz --iterations=10000`, seed `20261021`,
  0 failures on `spark-b`).
- Speculative multi-slot prefill continuations (`src/engine/prefill.cc`):
  `PrefillOrchestrator::Predict` now keeps ready decode peers in `cohort` for
  speculative multi-slot continuations when `max_segments > 1`, preserves
  `min_verify` row width when `min_verify > 1`, and scopes continuation `builds`
  expansion to `!has_prompt_future`, avoiding late solo step re-captures during
  multi-turn waves.
- Engine seams for physical memory modes (M3.5 U1): `Scheduler` owns the live
  node budget (`SetBudget`/`budget`) read by `AcquireProgram`, `LiveState` and
  `PagedNode`; `[memory]` accepts `mode`, `idle_release_seconds` and
  `pressure_*` trim thresholds wired to `Server::pressure_`; `RunnerResources`
  owns one engine-side backend and stream-library factory (`DeviceBackend`,
  `StreamLibrary`), `expert_slab.cc` sizes from the format layer, and
  `tools/llmp_boundaries.py` enforces that `src/engine/` and `src/runtime/`
  name no kernel libraries (`ggml`, `exl3`, `cublas`, `cutlass`); one
  `UnprovenWork` type cleared only after a positive stream fence replaces the
  separate unproven-launch flags across `ModelRunner`, `PromptBatch`,
  `StagedExecution`, `LiveState` and `PagedNode`; `PagedNode` splits along its
  driver-thread, region, accounting and request contracts; `SupplementalJob`
  reclaims evictable extents before materializing selected experts; and
  `ChargeRuntime` and `StepTransaction::Begin` close two accounting/validation
  edge cases from U10.
- Qwen3.8 MTP drafters take their expert layout (stacked gate/up, tensor
  scales) from their own tensors rather than the target's, so the relaid
  ModelOpt drafter composes with the GGUF and MLX targets again; the MLX
  drafter import no longer refuses its pre-stacked affine experts as an
  ungathered relayout table, and composes without ModelOpt's per-expert
  tensor scales (R18's MLX MTP import had never completed).
- Hang recovery's in-place reset (D-102 rung 2) now clears a step or routed
  continuation whose end was unproven (a lost fence; a faulted launch still
  restarts), so the model serves again instead of refusing every step until
  a restart; a cooperative cohort no longer stops
  refilling or pauses its members for another model's request that has
  already left the queue; the spill budget now counts the files restored
  conversations keep for their next spill (released first when the budget
  is passed), the reclaim order prices idle state as the spill or drop it
  will actually take, a restore refused for room keeps the conversation's
  record for a restart, and a request body waiting for request memory is
  freed when its client leaves (R20).
- `llmp import` refuses ambiguous or inconsistent checkpoints it used to
  import with wrong results: an empty (zero-element) EXL3, ModelOpt or MLX
  quantization part; `tie_word_embeddings: false` without a head (Qwen2
  safetensors and every EXL3 table); an image component's tensor its mapping
  neither reads nor declares as left out; a config member named `""` (and
  dotted member names are no longer flattened as if nested); an EXL3 marker
  that contradicts `quantization_config.codebook`; an MLX
  `quantization_config` that disagrees with `quantization`; an
  hf_quant_config.json beside an EXL3 or MLX checkpoint; a pipeline file that
  cannot be inspected; and unconsumed keys in a GGUF's own architecture
  namespace, now derived from the mapping rather than a hand-kept list.
- Planner and kernel correctness from the M3.6 review: a fusion whose
  emission fails falls back to primitives instead of failing the step;
  QSA's four-section positions are a typed, digested input fact rather
  than a name; every write-set check derives from one definition (the
  mean-pool store's was mis-indexed); pitched routing ids come from a
  declared capability fact; a catalog test proves components declare
  position-dependent emission in their plan-key words; MXFP8 and MXFP4 row
  tables are no longer relaid out to a layout no kernel reads. NVFP4 block
  scales are read as UE4M3 and MX scales as E8M0 by every llmpalooza
  decoder, and `llmp import` refuses a checkpoint whose block scales are
  NaN or outside their code (bytes GGML's readers and tensor cores decode
  differently).
- Unpin runtime numerical qualifications, planner policies, kernel capability
  predicates, on-chip decode formats and importer normalizers from specific
  checkpoint receipts and model dimensions (D-123, R14–R19): derive fusion and
  kernel eligibility from component roles, operand formats/encodings and
  hardware limits; support `placement = "pre"`/`"sandwich"` across `gdn`, `mqa`
  and `qsa`, `F16`/`F32` scalar parameter folds, role-driven expert relayout,
  MLX MTP drafter normalization and single-pass interpreted chat boundaries;
  and share `BackingService` 2 MiB handle reserves across `CarryRegion` and
  `PageInRegion` while skipping redundant routed-expert catalog closures when
  all selected pages are covered by the base lease.
- Add a generic pre-norm dense-decoder mapping (`llama`, `mistral`, `qwen2`)
  with a default `ServingRecipe`, normalizing and admitting GGUF, safetensors
  and EXL3 sources. Served end to end on the shared component engine
  (`EngineLlm`, `dense_decoder_serving_gpu_test`): Llama 3.2 1B Instruct
  (GGUF `Q8_0` and `Q4_K_M`) and Qwen2.5 0.5B Instruct (GGUF `FP16` and
  `EXL3`). Mistral, and Llama from safetensors or EXL3, are backed only by
  synthetic-source normalization and admission tests, with no served
  checkpoint. Accept per-dimension `freq_factors` on normal-pairing RoPE, and pass
  each row's published index during multi-row prompt scoring.
- Prepared ordinary quantized vector products may overlap immutable-weight
  prefetch and dependent launches while preserving their quantization,
  accumulation and reduction. Dependent inputs still wait for predecessor
  completion; unsupported CUDA images retain the ordinary path.
- Compile bounded native regex split plans from tokenizer sources, admitting
  Llama 3 and additional GGUF presets. Apply shared multi-ID HF template
  postprocessors, and fund complete normalized tokenizer work and outputs.
  Image prompts derive their dropped prefix from one complete encoding.
- Prepared expert vector products preserve broadcast versus per-slot input rows,
  and fused vector GLUs preserve SwiGLU, clamped SwiGLU and GELU-tanh formulas.
  Unsupported shapes or GLUs keep their ordinary fallback.
- The measurement client retains server errors sent inside an SSE response,
  instead of replacing their cause with a missing-usage message.

- Unequal-read joined token steps admit their exact activation storage before
  state or model work, preserving cached addresses and the old request on
  clean capacity refusal instead of silently losing the joined batch.
- Numerical qualifications retain their paid construction token on every
  serving path, including sources without a declared execution domain.
  Row-invariant quantized products execute every advertised block encoding.
- Owner attention accepts equivalent singleton view strides while checking
  every live row, head and owner pitch and current backing address.
- Report selected implementations, fusions and fallbacks at model registration
  and on demand; `--debug-plans` logs detailed paid plans on cache installation
  for the service, `chat` and `swap-table` (unreleased 0.2.0; no separate
  public-surface version changes).
- Debug plan reports include every physical operand view's dtype and geometry,
  with semantic formats for mapped tensors. Successful LLM uses identify their
  installed or private plan, distinguishing execution from unused lookahead.

- Scratch planning and funding use the backend's actual lane configuration,
  so serial multiowner programs do not reserve four unused lane workspaces.
  Launch, binding and cache identity share that immutable configuration.
- Automatic fusion queries and final implementation selection require measured
  qualification for newly available decode GLUs and EXL3 paired products.
  Rewrite provenance survives graph transformations; explicit typed operations
  retain their own selection contract.
- Decode selection declares actual device requirements and expansion, honors
  vector-row policy and per-matrix batch shapes, and keys cached plans by the
  implementation set, backend identity and capture requirements.
- NVFP4 row gathers consistently produce zeros for invalid IDs; the decode
  primitive also sets its optional error flag. Shared CUDA scalar conversions
  retain an independent CPU decoder as the numerical oracle.
- Row-interleaved NVFP4 products and gathers retain their optional tensor
  scales and every stacked scale; only single-scale unbatched gathers use
  the specialized native implementation.
- Batched MXFP8 products and short-block row gathers select a decoder that
  supports their batches; NVFP4 expert products preserve separate tensor scales.
- Import native Gemma 2/3 text checkpoints using their explicit query scale
  and source norm convention, and accept explicit GGUF attention scales.
  Native SentencePiece BPE assets and equivalent Hugging Face tokenizers load
  through the ordinary artifact path.
- Resolve architecture mappings by source container, derive admission policy
  gates and vocabulary from stored specs and tensors, and bound prompt masks
  from actual state geometry. Container provenance no longer acts as an
  encoding whitelist; executable support is proved by planning. Mapping
  recipes contain no callbacks or compiled target-size profiles.
- Refuse invalid fallback parts, unused shared mapping tensors, oversized
  mapping lists and forward or cyclic stage reads; canonicalize signed zero
  and honor converter folds during mapping evaluation.

- Native imports validate structural model variants, preserve tokenizer and
  chat metadata, and retain source encodings while reporting unsupported
  runtime encodings at admission.
- New artifact identities are independent of build provenance; composition
  format v2 keeps metadata filenames under their roles, including duplicate
  tokenizer filenames. Existing artifact IDs and v1 compositions remain
  readable (D-120).
- The chat route serves a request whose whole body, with pipelined bytes
  behind it, arrived with its head and then waited for request memory
  (it was left waiting for bytes it did not owe until its inactivity
  timeout, a 408). A request's head is checked (Host, Origin, route, method,
  media type) before reading more body data, charging it or sending `100 Continue`; an HTTP/1.0 request's `Expect: 100-continue` is ignored.
- Hang recovery's last resort exits at its deadline whatever blocks (a
  drive holding a kept record's write, a log that cannot be written): the
  state keeper touches no file and logs nothing under its lock, and its
  drain has a deadline for the lock too. The ladder and the chat route log
  outside their locks. A CPU-work cancellation is no longer cleared before
  the work saw it.
- A state keeper stopped while it hashes finishes the record it is making
  instead of writing one without its turn checkpoints (whose files the next
  start then removed).
- Calibration records of two aliases of one artifact (another drafter,
  other settings) no longer overwrite each other: the record's file is
  named by its configuration too (D-103 as amended).
- Turn checkpoint files are admitted against the spill budget before they
  are written, including checkpoints held by active requests. A full or
  zero budget skips this optional cache and preserves the conversation.
  State spills count the checkpoint files too, reclaiming eligible caches
  first and refusing a spill when held files leave too little room. Admission
  counts state spilled synchronously while updating a trimmed checkpoint's
  kept record too. Captures hold disk capacity through staging and copies,
  so synchronous reclaim cannot spend it on a state spill before publication;
  failure or cancellation returns that capacity.
- A request slot's admission by memory refuses a request whose state could
  not be measured instead of counting it as none.
- A pinned drafter pairs with an unpinned target of its architecture (a
  re-quant): its own inventory is pinned, its composition checked
  structurally (D-119 as amended).
- A drafter step wholly behind its slot's committed positions is refused;
  the prefill lookahead predicts joined futures only where their heads
  match, as the steps join; a failed step quarantines only what its own work
  may have written, not what its owners owed.
- A conversation's spill record keeps each written range once, so a long
  speculative session that is never spilled no longer grows it (and the
  reclaim order's scan of it) without bound.
- Element-aligned F32 attention views and overlapping-row copies use safe
  kernel paths. Validators reject oversized EXL3 slice bounds and overlapping
  pointer tables, and check DeepSeek routing and vector-product bounds.
  Failures after work was queued fault the device context (CUTLASS grouped
  and MXFP8 products, multi-launch EXL3 linears). Flash attention accepts
  long-context masks whose unused head stride exceeds 32 bits, and argmax
  accepts more than 65,535 rows.
- A smaller `draft_vocab` override now uses the same composed model identity
  during settings resolution and runner construction.
- On-demand row gathers use 64-bit byte offsets from planning through CPU and
  GPU execution, removing the 4 GiB landing-buffer limit.
- Shared partition planning places and budgets side sources and residual
  mixing components alongside the model's layers.
- Parallel routed feed-forward blocks apply the shared branch's own clamp
  limit, including when it differs from the routed experts' limit.
- Image decoder graph bounds follow the configured number of levels instead
  of assuming five.
- Quantized product optimizations are independent planner choices, so a
  model can use them without selecting unrelated DeepSeek operations.
- Component planning can enable indexed compressed-attention masks alongside
  dense, borrowed and QSA attention without reading another component's
  parameter schema. Other attention masks keep their ordinary behavior.
- The fused output-A projection keeps unsupported group geometries, rotation
  offsets and frequency inputs on the primitive path instead of aborting or
  changing the requested arithmetic.
- Native checkpoint admission takes expert counts and affine quantization
  parameters from the approved inventory instead of hardcoding Qwen3.8's
  values. Source, spec and complete tensor-inventory approval checks still
  apply; this does not qualify additional checkpoints for serving.
- A node whose shared workspace fits beside its largest model's weights
  maps it resident at the start again, as one region, and pages it with the
  active model only when it does not fit (the start logs which and why).
  Backed during each activation's page-in, the workspace could run the
  image's denoiser 1.7% slower on average, and up to 5%, on a host whose
  page cache was full.
- Gemma 2 prefills at the M3 speed again (it was 1% slower for one
  request, 1.3% for two; docs/experiments/m36-final/README.md, "State
  layout"): state is laid out by a per-tensor rule, each growing tensor of
  2 MiB or more on its own extents (as M3's layouts were) so that a prompt
  grows its state once every extent's worth of cells instead of every
  chunk or two, and smaller or whole-used tensors packed at 256 bytes, so
  that a slot's state stays near the packed layout's (DeepSeek V4's C2
  capacity); a compressed cache stays contiguous with its raw window.
  Kept conversations from before are not adopted (their layout changed).
- Planning a new step shape takes about half as long (Gemma 4
  26B-A4B 23 to 12 ms a plan, 31B 28 to 16 ms), so a decode that meets a
  new shape (a joined wave whose owners cross a read boundary) stalls less:
  Gemma 4 26B-A4B's decode was 1.2% (one request) and 2.2% (four) slower
  than M3, now 0.5%. The planner authenticated the physical alignment
  facts it carries through each pass by hashing the whole graph four times
  a plan; facts carried to a pass's graph are now sealed to that graph's
  identity, which no other graph shares. A plan's GGML metadata arena is
  no longer zeroed (GGML writes what it makes there in full).
- DeepSeek V4 with DSpark decodes joined speculative waves at the M3
  speed again on v1 imports: its hash-routing tables (3 MB) were read by
  rows from disk before every step that routed new tokens, a synchronous
  read of about 10 ms before each joined verify. A row table is now read
  by rows only when it is larger than the landing and slots that reading
  needs; smaller ones are resident like any weight.
- A node serving several models no longer sizes its start for the largest
  model's weights beside another model's larger workspace: the shared
  workspace (activations and GGML pool) now pages with the active model,
  each model naming only its own part of it, and the start admits the
  largest of each model's weights plus its own workspace. M3's swap table
  (DeepSeek V4 with DSpark, Qwen3.8 with MTP at four slots, Qwen-Image)
  starts again from an ordinary configuration; it was refused by 2.4 GB.
- `llmp-runtime swap-table` runs its prepared rows with the image model as
  B again: B's generation ends at its first output (the row's endpoint, as
  an LLM B's conversation does) instead of staying unfinished, which
  refused the next cycle's first output ("image first output order") and
  the next pair's image control.
- The native MLX importer publishes Qwen3.8 as `qwen4exp` (its mapping's
  architecture), not `config.json`'s `qwen4_exp`.
- Importing from an archived or staged revision no longer refuses a
  safetensors variant whose directory also holds files its index does not
  name (ModelOpt's `amax*.safetensors`): those are not the checkpoint's
  weights.

- Serve Qwen3.8 (native and GGUF), DeepSeek V4 with DSpark and a
  Qwen-Image-2.1 composition from an ordinary configuration again: since
  their legacy classes' removal they were refused without the internal
  `LLMP_DIAGNOSTIC_STARTUP_HOST_BYTES` / `LLMP_DIAGNOSTIC_NODE_CAP_BYTES`
  pair. Every model now registers under the automatic startup payment by
  default, its ceiling the node's measured available memory less the
  uncounted margin; the pair remains an optional diagnostic override. This
  changes Gemma's admission too: an ordinary Gemma start now pays its
  construction (a compiler allowance of about 501 MB shared by the aliases,
  each alias's retained spec, bindings and chat assets) in the catalog, and
  a start whose available memory is unknown is refused. Qwen3.8 at its default 262,144-token context bounds
  its prefill chunk by its sparse-attention selection tensors' 32-bit plane
  strides (2,040 rows) instead of failing to build. A model whose plans run concurrent
  lanes (DeepSeek) sizes its scratch pool with the lanes' share, so a
  `prefill_chunk` below the default no longer fails its first chunk ("the
  step's scratch exceeds the pool").
- The driver's wait for a request step keeps each stream's step history
  across requests (a refreshed lease or a new turn keeps it), and a
  stream's first step, if long, no longer keeps a core polling its fence
  (every model). Image stages keep their lease between steps; a refusal
  between stages releases it and returns the generation to its start.
- Normalize and publish the approved DeepSeek V4 GGUF target through the native
  importer, preserving source encodings and exclusive embedding/hash-table
  ownership. Validate stored-target full-head, state and spill-continuation
  equivalence through the existing paid shared route; public defaults and
  stored DSpark admission remain unchanged.
- Treat the internal diagnostic node cap as an upper bound at both initial
  physical admission and final dynamic-budget selection, retaining all existing
  startup footprint and handle-reserve checks.

- Admit DeepSeek target/DSpark through the internal shared Server route with
  component-derived decoding bounds and explicit physical admission. Share one
  paid transient compiler allowance across component aliases while retaining
  separate source and live-program ownership. Serial speculative HTTP completes;
  the bounded two-client screen remains blocked by conversation-state capacity.

- Publish the approved Qwen GGUF target through the ordinary native importer
  and admit its stored spec through the internal paid component route. Keep
  embedding and PLE tables independently owned; validate full-head, state and
  spill-continuation equivalence against the existing v0 target. Public defaults
  and stored-drafter admission remain unchanged.

- Prepare installed image compositions through the existing shared Server route,
  retaining paid source ownership across changed prompts, schedules and geometry.
  Grow construction accounting on demand and write private image output only
  when requested; public serving defaults remain unchanged.
- Grow shared image request resources at completed boundaries while retaining
  stable workspace addresses and other models' borrowers. Retire geometry-owned
  ports and staging after completion; an internal prepared-source Server bridge
  reuses the same model/backend ownership without changing public defaults.
- Extend the shared DeepSeek target scoring diagnostic beyond 2,048 history
  tokens using the existing scorer and same-checkpoint fused reference. Derive
  actual wave shapes, deduplicate complete plan keys and pay measured retained
  owners plus one replacement before execution; preserve the short diagnostic.
- Apply the same validated internal shared-engine options to HTTP and local
  serving commands. Retain exact quality token IDs/losses from one score call
  for same-checkpoint reference comparison, and validate existing 8K saved-state
  continuations across native+MTP and GGUF swaps without changing public defaults.

- Allow an internal, default-off Verify policy to capture beside its first eager
  execution through the existing paid graph lifecycle. A bounded shared-engine
  screen observes higher C4 decode throughput with extra graph memory and
  slightly lower overall throughput; public defaults remain unchanged.

- Replace completed shared image request content and schedules while retaining
  model weights, typed ports and captured programs. Active or unknown requests
  and incompatible layouts refuse before mutation; serial generations release
  completed request content instead of accumulating outputs.

- Build shared image requests from paired output dimensions, the actual prompt
  and configured Euler step count, with checked source geometry and paid
  admission instead of fixed diagnostic dimensions or schedules.

- Preserve unknown completion when a lane join fails after an earlier clean
  kernel refusal, so the enclosing owner cannot mistake unfinished work for a
  safely rejected step.

- Return zero for fully masked SoftMax and GGML MMA attention rows. Shared
  spatial attention can explicitly select paid transient F16 packing, with
  dense F32 fallback and unchanged default precision policy.

- Derive shared-service startup accounting from enforced parser limits and actual
  source, settings and tokenizer inventories. The internal selector accepts
  `auto`, repays its temporary parsing extent after retained grants exist,
  and preserves explicit grants, public defaults and ordinary runtime admission.

- Validate image diagnostic repeats against the current candidate output
  instead of a stale pixel hash from an earlier implementation.

- Report migration peak-memory differences without rejecting a worthwhile
  speed or quality benefit solely for using more than its reference. Speed
  and run-validity checks remain mandatory; hardware admission is unchanged.

- Reuse bounded F32 im2col and shared cuBLAS for qualified spatial convolutions
  on Spark, preserving source formats and explicit copies. Exact measured
  shapes use the shared selector; other shapes retain direct convolution.

- Extend the explicitly funded shared Qwen diagnostic to native target-only
  serving, deriving ordinary prompt/decode authority from the actual target and
  configured envelope. Two- and four-request HTTP cohorts and retained-prefix
  reuse pass; public defaults and composed/scoring routes are unchanged.

- Qualify the existing shared Qwen GGUF target-only HTTP route with two- and
  four-request cohorts and retained-prefix reuse, using native preflight token
  counts from its kept chat assets; public defaults remain unchanged.

- Connect the internal, explicitly funded shared Qwen diagnostic to the existing
  HTTP service. Two concurrent streamed MTP requests complete through joined
  execution; public model selection remains unchanged. Failed native Server
  teardown now exits before releasing potentially borrowed owners.

- Reuse shared matrix products for supported F32 pointwise spatial convolutions,
  preserving explicit layout copies and source formats. A bounded image sample
  retains native migration agreement while other convolution shapes keep their
  existing implementation.

- Let the internal shared Qwen Server route derive a paid execution domain from
  its actual target/drafter composition. A bounded two-conversation control
  exercises default prompt futures and restores turn checkpoints under fresh
  Server owners; public model selection and numerical policies remain unchanged.

- Honor explicit legacy image output directories with an atomic private RGBA
  export of the completed image. Ordinary serving and swap measurements keep
  exports disabled; the shared pipeline passes a bounded same-source native
  migration agreement screen without requiring identical pixels.

- Execute a complete 1024×1024 image generation through shared text, flow and
  decoder programs, with completed device-port handoffs and retained-plan reuse.
  The same paid program lifecycle now serves language and image adapters. Valid
  cuBLAS odd-pitch and wide batched products retain their own indexing limits;
  paired kernels keep their existing restrictions. Legacy image serving remains
  unchanged while reference quality and migration performance are qualified.

- Compose all 32 image denoiser blocks with source-owned text/image projections,
  final adaptive normalization and an explicit Euler update through shared
  components. BF16 weights, existing fusions and owner-local input/output ports
  are retained; whole-pipeline quality and serving remain separate.

- Compose an image transformer block with its timestep conditioner through shared
  adaptive normalization, three-axis rotations, masked attention and gated
  residuals. BF16 source weights and existing fusions are retained; bounded
  execution passes, while full flow, reference quality and serving remain open.

- Add explicit per-alias `kv_cache_encoding = "q8_0"` for supported shared
  dense attention at D64/D256, with checked blocked state and kept identity.
  F16 remains the default; short paired quality costs are reported and unsupported
  state consumers refuse. Configuration schema 2 remains compatible (D-117).

- Compose the complete image timestep conditioner from shared projections and
  explicit output ports, preserving BF16 source weights. Both conditioning
  branches pass bounded finite-output and changed-input capture/replay checks;
  full flow, image quality, comparative speed and serving remain separate.

- Bind durable kept-state layouts to a present prepared execution identity,
  refusing state from a different owned backend, tuning or declared domain
  before adoption or restore. Legacy sources without that identity keep their
  existing layouts; no serving policy or on-disk fields change.

- Compose the stored first-frame VAE decoder from shared spatial fragments and
  owner-local residual values, preserving F32 source weights and reusing the
  existing program path. GPU graph construction can omit unused full scalar
  reference buffers. The bounded complete first-frame decoder and capture/replay
  screen passes; whole-image quality, speed and serving remain separate.

- Invalidate shared calibration when the actual bound model spec changes,
  preserving default settings, artifact storage paths and legacy records.

- Compose the prepared DeepSeek target and DSpark companion through shared
  block drafting and injected features, checking complete proposals and combined
  snapshot continuation within existing memory limits. Ordinary serving remains
  unopted; draft-head quality and family qualification remain separate.

- Run typed spatial convolution and nearest upsampling through the shared
  StepProgram and completion/capture path using existing F32 kernels and source
  VAE parts. The bounded main screen passes.
  Whole-image quality, comparative speed and pipeline serving remain separate.

- Execute the complete prepared DeepSeek target through the shared component
  engine in a bounded diagnostic, with real hash/compressor inputs and snapshot
  continuation. Reuse already-funded graph-reader space and retain state-only
  prefill's omitted final expert products; ordinary serving remains unopted.

- Add source-preserving spatial component graphs for nearest upsampling and
  channel-changing convolution through the shared planner, with checked owner
  geometry and host reference controls; whole-pipeline execution remains separate.

- Prepay stored-component binding construction and reuse validated role/part
  locations in admitted native Gemma builds, removing discovery-only setup
  graphs while retaining real setup measurement and existing admission limits.

- Reduce selected-expert page lookup CPU work for complete device reservations
  without a persistent index. Keep the original fallback; the bounded screen
  records slower small dense and host-fallback cases alongside large-inventory
  gains. This does not establish model latency improvement.

- Carry checked reserved-weight alignment into component planning, so explicitly
  tuned supported products can be selected without inventing physical facts.
  Unknown facts retain ordinary selection; invalid promises refuse before
  lowering. No tuning policy is installed.

- Admit shared teacher-forced scoring and ordinary decode by their actual
  numerical mode, preserving the separate companion-execution permission.

- Restore adopted component prefixes before reuse when drafter progress is
  still on disk, avoiding needless prompt recomputation. Execution waits for
  successful hydration.

- Native artifact copies check complete source coverage and acknowledged
  output writes before publication.

- Archive downloads now enforce one writer through atomic NAS ownership (D-115,
  checkpoint writer protocol 2, tools version 2). Remove periodic partial
  pathname checks that stalled transfers on SMB; keep inline hash
  verification and resume abandoned downloads immediately. Stop older
  checkpoint writers before upgrading. Stored archive schemas remain 1.

### Added

- Every GGML backend exposes the complete CUDA decode ladder: 74 operations,
  including MXFP4 split/scale-atoms decoding, dense NVFP4, MXFP8 routed experts,
  resident row gathers, batched products and both compiled mixed GLUs.
  Production EXL3 uses the audited kept MIT kernels with actual occupancy facts
  and the existing exact reconstruction pins.
  Selected kernel state is separately funded and retained per backend, so
  model switches and scratch remapping preserve its ownership and initialization.
- Native drafter imports take `--target TARGET_ID` in the selected store and
  validate the stored target spec; published drafter artifacts remain usable
  without reimport. This CLI semantic change raises the next product version
  to 0.2.0; stored schema versions are unchanged.

- The DeepSeek V4 community IQ2_XXS GGUF
  (`antirez/deepseek-v4-gguf@f71f23d5`, chat-v2 imatrix) serves again, as
  v1 `e7c0d0e4…` from `llmp import` under its own approval row (contexts to
  262,144, four owners). The importer now accepts its IQ2_XXS, Q2_K and F16
  tensors, its `deepseek4.nextn_predict_layers` and `deepseek4.vocab_size`
  keys and its 44-entry layer-type list; without
  `deepseek4.swiglu_clamp_shexp` the shared expert clamps as the routed
  experts. No other source's import changes.
- `llmp import SOURCE --component mtp [--draft-head IDS]` publishes Qwen3.8
  Flash Next's MTP drafter from its ModelOpt NVFP4/MXFP8 checkpoint as a v1
  artifact in its source encodings, its routed experts relaid out as the v0
  drafter's (no spec of its own; composed with its target at serving), optionally with a
  selected draft head whose BF16 rows are gathered from `lm_head.weight` for
  the strictly ascending token IDs IDS lists; the list joins the artifact's
  sources and its importer options, so the artifact ID changes with it.
  The approval table admits the drafter with Mia's 47,172-ID list, composed
  with the native ModelOpt target through its mapping's drafting recipe
  (its block's values compiled constants, its head's rows from its
  tensors, under a pinned composition). Additive CLI options; the
  artifact format stays version 1.
- Qwen3.8's native ModelOpt target, its MTP drafter and its MLX target have
  pinned approval rows (no row is transitional): their stored specs'
  digests, inventories and the applied folds' and relayouts' records, and
  the mapping each spec was synthesized by. Native ModelOpt and MLX
  artifacts imported before 2026-10-15 are refused; re-import them.
- The importer relays NVFP4 routed experts out losslessly into the
  scale-atom layout the block-scaled grouped GEMM reads (a MoE's gate and up
  stacked, tensor scales gathered per projection), by canonical name and
  encoding, recorded as an applied relayout; the CPU references decode it.
  NVFP4 row tables are row-interleaved the same way. The native ModelOpt
  Qwen3.8 target with its v1 drafter then runs v0's expert and n-gram
  kernels. The importer also folds small per-channel parameters a
  component would derive every step into F32, chosen by the role each
  tensor binds and recorded as applied folds: the ModelOpt and MLX
  targets' zero-centered norms (1 + w) and A (-exp(A_log), in double), the
  drafter's norms. On a device with
  FP4, the W4A4 grouped GEMM runs a target's experts where the checkpoint
  quantized their activations too (their `input_scale` tensors present) and
  a speculative stage's always (verification bounds its quality); other
  experts decode on the format ladder's grouped product. Previously
  imported native ModelOpt targets and drafters must be re-imported. The
  NVFP4 scale-atom layout is named `scale_atoms` (no artifact recorded its
  old name); the rename changes the decode implementations' identities, so
  v0 Qwen calibrations and kept conversations recorded before it are not
  reused.
- Split NVFP4 and affine routed experts run in ordinary component programs:
  part-by-part slab bindings and a capturable device-routed grouped product
  over resident experts. Fused patterns read source-encoded norms and A as
  derived parameters. The native ModelOpt Qwen3.8 target publishes as
  `qwen4exp` with its multipart n-gram table a group a part.

- An internal joined target/MTP diagnostic requires a prepared, complete source
  composition and explicit numerical-domain permission before owner state can
  change. Ordinary Qwen serving admission remains unchanged.

- Shared `norm.adaptive` and `residual.gated` component fragments express
  scale-only adaptive LayerNorm and tanh-gated residuals using explicit dynamic
  conditioning and the saved original stream. This CPU/IR foundation does not
  activate image GPU execution or pipeline conditioning.

- Add a shared channel-normalization component with learned scaling and optional
  SiLU, using F32 arithmetic. This CPU/IR foundation does not activate image
  serving or GPU lowering.

- Enable native GGUF normalization and the existing `llmp import` writer path
  for the supported Qwen3.8 Flash-Next target profile, preserving encoded
  tensors and source n-gram constants. Place side n-gram weights in row-table
  groups. Actual full publication and runtime admission remain separate.

- Run the active swap and speculation benchmarks' FP16 comparison through
  the shared component engine, with explicit host memory limits and one
  setup compilation per shape class. Historical standalone paging modes
  are refused by this adapter. Retire the obsolete standalone FP16/EXL3
  executors and family graph/phase builders; generic kernels and shared
  execution controls remain.

- Route the closed native ModelOpt target through `llmp import` using the
  existing authenticated writer and archive flow. Actual CLI checks cover
  metadata-budget refusal and complete pinned-target publication; runtime
  admission remains separate.

- Extend the native writer API to owned multipart normalization plans, with
  expert scatter, byte-preserving permutations and applied-fold receipts.
  The pinned native ModelOpt target passed complete source authentication
  and publication; runtime admission remains separate.

- Prepare owned native ModelOpt target normalization plans, preserving source
  weight encodings through multipart gathers and head permutations. Publication
  and runtime admission remain separate.

- Keep declared small component parameters in BF16 and widen them in paid
  execution workspace. Explicit logarithmic GDN A uses transient arithmetic.
  Optional zero-centered norm parameters add one in F32 workspace and reject
  an already-applied offset fold, preserving resident encodings and default
  graphs.

- Bind approved EXL3 operations through the shared step program. The
  4.0-bpw reconstruction fixture matches a native 145-row reference exactly;
  remaining execution profiles and lifecycle gates stay open.

- Add closed native-v1 admission declarations for the approved Gemma 3 QAT
  and Gemma 4 target profiles, with exact tensor inventories and native expert
  block geometry. All three published artifacts match v0 full logits, tokens,
  initialized state and spill continuation, including joined decoding.
- Admit the approved native v1 Gemma 2 Q8_0 artifact through its stored
  component spec and native tensor formats. Full logits, tokens, initialized
  state and spill continuation match v0. Native identities beyond the four
  declared Gemma targets and drafter compositions remain refused.

- Shared-component specs for the Qwen2.5-0.5B FP16 and approved EXL3
  fixtures. The shared runner reproduces the FP16 fixture’s frozen fused
  and unfused control trajectories; complete migration controls remain open.

- Add `llmp import SOURCE --store DIRECTORY` for native v1 publication
  from normalized Gemma 2/3/4 target GGUFs, including archived and staged
  sources, source-header and full-file verification, embedded specs,
  expert slicing and archive import events. Artifact/archive schemas
  remain version 1; family re-import and admission qualification remain owed.

- Add an internal sole-owner Qwen3.8 component serving route for native
  MTP and GGUF migration checks, with funded startup ownership and draft
  context admission. The family migration remains in progress.

- Archive download telemetry can report known byte counts and transfer phases
  to a local controller without querying model storage. The initial archive
  batch pushes status once per second to connected progress viewers over a
  bounded, nonblocking Unix socket.

- The model archive (D-114, docs/model-archive.md): source checkpoints
  on a long-term store as one directory per pinned commit with a verified
  manifest, and a catalog relating quantizations and versions, written by
  the new `tools/archive-model`, `tools/archive-catalog` and
  `tools/stage-model` (`mise run archive`, `archive:catalog`, `stage`).
  New on-disk formats, each at version 1; nothing in the runtime reads
  them yet.

- Prepared-artifact format version 1 (D-113), read beside version 0:
  each tensor's format and part in the index in place of v0's
  library-named descriptors, the model spec as a hashed `spec.json`, the
  folds and relayouts the importer applied and its identity in the
  manifest, and compositions of version 1 that may carry a model's spec.
  Groups, chunks and page-in are unchanged. An artifact or composition of
  any other version is refused as `unsupported-version`, naming the
  version; nothing is migrated. No importer writes version 1 yet; a minor
  change of the 0.x line.

- `spark-job start --shared` runs builds and non-timing functional tests
  up to `$LLMP_JOBS_SHARED` (default 2) at a time on a Spark, never beside
  a `--gpu` job, which stays exclusive and, while it waits, admits no new
  shared job. `busy` reports both classes and exits 0 free, 1 busy, 3
  when only shared jobs run. The CTest label `timing` marks tests that
  bound wall-clock or CPU time, which run only in exclusive jobs.

- Kept conversations' record format version 2 (D-112): a conversation
  kept across a restart by a model on the shared engine names one spill
  file per pipeline stage (`slot-N.state`, then `slot-N.stage-S.state`),
  each checked and hashed on its own; a minor change of the 0.x line. The
  M3 serving classes keep writing version 1; a record of the version a
  model does not write is refused and its conversation prefilled again,
  never migrated. One serving adapter per pipeline kind (LLM, image) over
  the shared runner backs a model once its family's migration enables it
  (Gemma 2 and Gemma 3 first, below).

- Serve the approved Gemma2 2B Q8_0 artifact through bounded chat/literal
  completions at context<=8192 and up to two owners. Compatible prefill and
  common-width decode preserve cap50 and independent state; actual system-role
  refusal, scoring, stops, peer progress and kept restart replay are checked.

- Serve the approved Gemma3 4B QAT artifact through ordinary chat and literal
  completions at context<=4096 and up to two owners, with checked optimized
  decode, initialized checkpoint/restore and kept-conversation adoption.
  Unequal-width/ring/departure and HTTP lifecycle controls pass; joint prefill,
  wider cohorts and sustained qualification remain open.

- Execute the approved Gemma3 4B QAT Q4_0 artifact through the shared native
  runner, with bounded C1 eager/graph, Clear and spill/restore own controls.
  Explicit optimized C1 and independently-prefilled C2 decode recipes pass
  strict bounded reference screens; C2 preserves exact own states/choices.
  The later bounded serving route adds lifecycle controls; broader quality/
  performance, joint prefill and wider batching remain unqualified.

- Prepare the approved Gemma 3 4B QAT Q4_0 checkpoint through the generic
  artifact importer, with strict source, metadata and tensor preflight. The
  real artifact passes deep verification.

- Gemma3 QAT text state/input and descriptor planning foundations with checked
  independent slots and primitive operation plans.

- Opt-in Gemma equal-width partial owner cohorts retain whole-stream attention
  partitions with bounded independent cache roots. The 31B C5 recovery passes
  unchanged strict choices and conditional loss against retained stock; defaults
  stay off and wider model/unequal-width qualification remains open.

- Opt-in Gemma attention carries two or three actual cache owners with whole-stream
  geometry, retaining existing C1 and quad/tail fallbacks. Both approved C2 screens
  pass strict choices and bounded conditional loss; serving defaults remain off.

- Opt-in ordinary Gemma waves share twelve owners' products with three funded
  independent-cache attention quads; row-invariant waves retain eight. Both
  C12 fixed-bound/conditional-score screens pass, with three strict 26B choice
  differences retained. Serving owner/joined defaults remain off.

- Live-state source/pin checks reuse successful scans until relevant mutations,
  retaining every scheduler call and independent residency/lease checks. Both
  Gemmas retain exact C8 heads/state/choices; the short 26B factor improves
  1.77%, while 31B and Qwen/DeepSeek latency benefits remain unestablished.

- Gemma serving pins head publication storage for configured owner slots rather
  than every input row, with early capacity refusal and preserved manual all-head
  defaults. Both approved profiles retain exact continuation/scoring controls.

- Immutable weight placement checks reuse successful results until the scheduler's
  sources or pins change. Gemma31 C4 saves 25.615 ms / 0.77% with exact heads,
  state and choices; shared Qwen/DeepSeek speed benefits remain unmeasured.

- Shared GGML execution reuses validated kernel wrappers within each bind,
  retaining fresh descriptor, arity and lane checks. Gemma31/26 binding falls
  73.98%/69.29% with exact prior heads/state; reference parity remains open.

- Checked plain RMSNorm/Mul is enabled by default for both approved Gemmas.
  Ordinary off/on controls retain exact complete heads, initialized state and
  continuations; other experimental arithmetic flags remain opt-ins.

- Gemma prefill builds a bounded next-chunk CPU plan during current execution.
  Exact head/state controls pass; same-binary Gemma26/31 prefill improves
  1.55%/2.36% under the existing optional policies. Reference parity remains open.

- Bounded arena-owned graph traversal preserves DFS order and explicit host
  accounting across Qwen, DeepSeek and Gemma graph factories. Fresh Gemma26/31
  state-only prefill screens improve 1.93%/0.99%; reference parity remains open.

- Gemma prompt prefill omits unused final-layer work on non-final chunks while
  preserving all KV writes. Final heads, scoring and retained features keep the
  full path. Existing all1024/both256 screens improve 3.04%/2.42%; reference and
  full model qualification remain open.

- Exact source-use counts reuse the graph index across fusion gates without
  changing math or policies. Fresh Gemma26/31 prefill screens improve
  2.80%/2.31%; reference gaps remain 12.38%/11.40%, with exact prior heads/state.

- Call-local graph reader indexing preserves checked planning and exact Gemma
  head/state controls while reducing Gemma26 8K prefill time by about 27%.
  Initial Gemma26/31 prefill gaps were 15.10%/13.83%; quality remains open.

- Gemma26 coarse prefill diagnosis preserves prior heads/state and measures
  about 1.093 s of caller CPU in the two graph-planning passes.
  Production algorithms and selected policies remain unchanged.

- Gemma26 late-prefill routing and reduction controls on captured common inputs:
  native primitive/fused outputs match each original policy byte-for-byte.
  Recipe rounding differs; the remaining model disagreements stay unexplained.

- Gemma26 matched prefill tracing with exact prior head/state controls. Most
  of the observed gap lies outside recorded GPU activity; CPU and wait
  attribution remains open, with no production optimization selected.

- Gemma26 packed C4 routing/reduction comparison reduces positive-margin
  disagreements from 68 to 2 of 128 heads, with 92 heads byte-exact to llama.cpp.
  The candidate remains unselected and takes 7.15% more time.

- Gemma26 8K ring-cache reference transfer: all 32 decode choices agree,
  but both retained full heads differ. Native prefill/decode take 57.64%/0.49%
  more time; full quality and competitive performance remain open.

- Gemma26 stock dispatch observations with byte-exact full-head controls:
  routing is fused throughout four-owner decode and skips two layers during
  prefill. Scaled reductions stay fused; model qualification remains open.

- Dense31 8K comparisons with a matched ring-cache reference: all 32 decode
  choices agree and the final head is byte-exact. Prefill quality still differs;
  native prefill/decode take 16.20%/2.05% more time, leaving qualification open.

- Dense31 solo norm screening with all 32 complete heads byte-exact to fresh
  llama.cpp. Short decode latency is 1.92% higher; production selection and
  broader quality and performance qualification remain open.

- Gemma26 packed-attention transfer controls with exact native head/state repeats
  and matched reference comparisons. Quality fails and latency rises 9.19%;
  the candidate remains unselected.

- Dense31 scalar FFN replay with actual captured inputs and observed reference
  fusion. Native, fused and unfused paths produce identical complete activation
  and down-projection bytes for the tested input; broader qualification remains open.

- Dense31 packed C4 attention diagnosis with byte-exact complete heads against
  fresh llama.cpp. Packing costs 12.2774% more latency; production selection
  and Gemma26 transfer remain open.

- Gemma26 assistant C2 reference comparisons with exact native serial/joined
  recurrence and short matched timing controls. Original batch-two distribution
  drift remains explicit; serving speculation and qualification remain open.

- Dense31 C4 attention replay that isolates vector/MMA and stream-packing
  arithmetic. Four-stream native MMA matches the original backend exactly;
  model qualification and production selection remain open.

- Native Gemma Q8 assistant component with protected post-finalnorm features,
  readonly frozen target caches and independent-slot recurrence. Bounded C1
  original-image arithmetic passes; serving speculation and qualification remain open.

- Reproducible original-image Gemma26 assistant reference controls at C1,
  serial C2 and batch two, with exact repeats and unchanged borrowed target
  state. Native assistant quality and performance qualification remain open.

- Dense31 joined norm screening with ordinary product selection and matched
  reference bookends. Quality and performance failures keep selection open.

- Reproducible dense31 joined norm/row-policy screening, with exact native
  repeats and matched stock comparisons. Quality failures keep the candidate
  unselected.

- Explicit default-off joined Gemma decode controls, with ordered bounded
  subwaves, independent publication/refusal and owned spill/restart. Native
  solo/joined agreement and real multi-sequence reference comparisons retain
  quality/performance gaps; optimized batching is not selected.

- Checked Q8_0 Gemma assistant profiles, tensor/target bindings and paired
  canonical-vocabulary validation. Serving speculation remains unqualified.

- Reproducible larger-row Gemma prefill diagnostics, with same-shape state
  repeats and matched reference speed/quality comparisons. Numerical failures
  keep production row caps and optimization defaults unchanged.

- Explicit default-off Gemma routing and scaled ordered-reduction dispatch,
  with full-sort backing, kept-value primitive fallback and bounded state/
  numerical controls. Failed representative quality keeps selection off.

- Bounded dense Gemma31 chat and literal completion admission through the shared
  approved-profile adapter, with scalar cohorts, target likelihoods and owned
  cross-profile continuation/restart controls. Optimized joining and model
  qualification remain separate.

- Explicit default-off checked Gemma norm/rotation and norm/residual fusion
  controls, preserving kept intermediates, raw K-as-V and paid residual gathers.

- Checked standalone Gemma top-eight routing and scaled ordered expert
  reduction, retaining original pinned GGML arithmetic and primitive fallback.
  Graph dispatch and whole-model qualification remain separate.

- Bounded Gemma26 chat and literal completion route on the native driver,
  with target likelihoods and up to twelve independent scalar request owners.
  Thinking, generated tools and assistants remain unavailable.

- Explicit default-off Gemma per-segment RoPE/cache-store fusion, with
  checked primitive fallback and readable diagnostic keeps. Whole-model
  optimization selection remains separate.

- Bounded native Gemma 26B-A4B engine runner with independent request slots,
  initialized KV, device masks and exact checkpoint/spill continuations.
  Serving and full model reference qualification remain separate.

- Dense Gemma 31B executes through the shared native runner, with ordinary
  replay and checkpoint/spill controls and distinct restore layout identity.
  Its representative reference quality gate fails; serving remains unavailable.

- Checked D256/F16 local attention primitives with GQA2 vector/MMA shape
  selection, independent ring masks and charged capture workspace. Gemma
  whole-model execution and batching qualification remain separate.

- Gemma 4 segmented text graph and checked plan foundations, with independent
  slot caches, global frequency-factor RoPE and complete-layer diagnostics.
  Serving and whole-model reference qualification remain separate.

- Native GGML Q4_1/Q5_0/Q5_1 product primitives and legacy Q4_0/Q4_1/Q5_0/
  Q5_1/IQ4_NL row-preserving and joined vector products, with readable
  padding and independent column controls. Model qualification remains separate.

- Native classic SentencePiece GGUF tokenization for the approved Phi 3.5,
  Gemma 2 and Gemma 3 fixtures, with score-ordered merging, byte fallback,
  bounded working memory and pinned reference agreement.

- Native prompt, generation, retained and snapshot token histories are
  charged by allocated capacity before growth. Idle histories across the
  model library can release that memory with their state; active and
  paused exact continuations retain their funded tokens.

- Native Gemma 4 GGUF tokenization: raw UTF-8 BPE, space markers, byte
  fallback, pinned reference agreement and mode-aware request memory bounds.

- Qwen speculative verifies fuse the recurrent alpha/beta pointwise chains
  with original F32 arithmetic and saved-row restoration. A matched four-
  request 8K HTTP screen improves completed-token throughput by 1.34%.

- Calibration records track explicit chunk/slot overrides, including an
  override equal to the fallback. Qwen also tracks its effective draft head,
  shared draft cap, joined-draft limit, read alignment and depth-cost ratio,
  so reused measurements follow the execution policy.

- Qwen settings list the effective selected MTP head size and explain a
  capped request. Benchmark diagnostics record actual head kind and rows;
  the selected artifact's execution remains unchanged.

- Qwen MTP waves share selected BF16 heads while retaining each column's
  ordinary vector arithmetic and independent state. A matched two-request
  HTTP screen improves completed-token throughput by 1.77%; the existing
  two-request joined-drafting limit remains in force.

- Literal `/v1/completions` requests join Qwen and DeepSeek's native
  cohorts alongside chat requests. Prompt and generated likelihoods,
  Unicode offsets, response state and sampling survive model switches;
  zero-token scoring is supported at the full context. Unsupported
  continuations fail explicitly rather than restarting. Compatible with
  inference API version 1, a minor change in the 0.x line (D-100).

- A pending model switch pauses a running chat cohort at completed units
  after `[client] model_turn_seconds` of resident work (default 30), serves
  one substitute cohort, then resumes the original requests ahead of later
  arrivals. Paused state spills with its exact branch identity protected;
  response text, sampling and usage continue across the switch. Filled
  same-model slots still wait without preemption. Compatible with schema 2,
  a minor change in the 0.x line (D-069).

- Qwen3.8 waves run each request's attention and recurrence on its own
  stream from four requests, with shared products and dependencies ordered
  inside the same graph. NVFP4 and GGUF retain exact logits and state.
  `wave_lanes` in the model table defaults to true and can disable it;
  compatible with configuration schema 2, a minor bump of the 0.x line.

- Every model setting resolves in three layers (D-103): a default derived
  from the artifact (the trained context, the checkpoint's sampling
  defaults, the drafter's block, the vocabulary's reasoning markers; never
  a checkpoint's name), a calibration on the machine, or an override in
  `[models.<name>]`, else the constant measured on a GB10. Calibration is
  measured from a model's first uses (the progress floors, the recompute
  cost, DeepSeek's wave costs per width, Qwen3.8's draft depth cost) and
  recorded under the state directory per artifact, device, driver, build
  and the settings the measurements depend on (a new on-disk record,
  `llmp-model-calibration-v1`; delete it to measure again), in force
  from the next start; a stale record is measured again, a corrupt one
  refused and replaced. The reclaim order's cost of recomputing dropped
  state is now a cost a token (`recompute_ms_per_token`), scaled by each
  conversation's tokens. The model table's schema is one table of keys, types, ranges and
  the models that take them, with new keys for every setting the runtime
  had fixed (sampling defaults, reasoning markers, draft rows, DeepSeek's
  wave costs and output-A/HCA prefill switches, Qwen3.8's draft
  vocabulary, depth cost, shared-wave depth, joined-draft width and wave
  read alignment, the reclaim order's recompute cost, the image's size and
  steps): compatible with schema version 2, a minor bump of the 0.x line.
  A request that sends no temperature, top_p, top_k or min_p now samples
  with the model's default, which is the checkpoint's own where its
  artifact keeps one (`do_sample: false` is greedy): unchanged for
  DeepSeek V4 Flash's GGUFs (temperature 1, top_p 1) and the Qwen3.8
  NVFP4 artifact (none kept), but Qwen3.8's GGUF quantizations now sample
  with their checkpoint's top_k 20 and top_p 0.95 when a request sends
  neither. The start logs each value and its source, and the new
  `llmp-runtime settings [--json]` lists them with what each came from,
  beside a running service. A key a composition does not take is now
  refused naming that key; the throughput floors are resolved once per
  model instead of looked up per request.
- A genuine hang is recovered (D-102): the engine's fixed ten-minute step
  patience is gone, and a hang is work under way with no progress of any
  kind (no unit ending, no fence completing, no read or write landing)
  for `[client] hang_seconds` past its unit's allowance, so slow healthy
  work (a swap from a slow disk, a long prefill chunk, a long rendering
  or tokenization, which now beat as they go) never counts. The runtime
  then cancels the stuck work, and once the cancellation drained (nothing
  submitted before it still in flight: a read a drive holds is not ended
  by its cancellation), fails only the requests that needed it (503
  `backend_hung`) and resets the model in place (its stream fenced, the
  touched conversation state discarded, plans and graphs dropped, the
  model reloaded whole on its next request) while the service goes on;
  only when nothing less frees it (a read the drive never returns, a hung
  device, a stuck driver, a model that hangs again before serving a
  request, a failed reset) does it exit for systemd to restart it. Each
  step is logged with why.
- Conversations survive a restart of the service (D-105, a new versioned
  on-disk format, version 1, a minor bump of the 0.x line): a
  conversation whose state is wholly on disk (spilled,
  written back by a swap, or spilled at a graceful stop) is kept in named
  owner-only files beneath the spill role with a hashed record, and the
  next start adopts it after checking the build, model, layout, files and
  their SHA-256 digests, so its next turn restores the state exactly
  instead of prefilling again (measured: the same reply as an
  uninterrupted run, all 265 prior tokens reused, after a SIGTERM or a
  SIGKILL). A record that does not validate is refused and its files
  removed; retention and the spill budget apply while the service runs
  and across the restart, and a spill budget of 0 keeps nothing. The
  records hold the conversations' tokens and their files the state, on
  disk while the service is stopped: the new `[memory]
  keep_across_restart = false` (compatible with schema version 2) keeps
  nothing past the process, and purging the package removes the default
  spill directory's kept conversations. The records are hashed in the
  background at the lowest priority, waiting while a swap or a prefill
  runs.
- `llmp.service` retries a failing start forever, backing off from 5 s
  to 30 s (it gave up after ten starts in five minutes), and its
  start and stop timeouts are extended while the start registers and
  adopts conversations and the stop spills and records them.
- A model's concurrent chat requests are no longer fixed at four (D-104):
  each LLM has request slots up to a cap, `[models.<name>] max_slots` (1
  to 16; a new key compatible with schema version 2, a minor bump of the
  0.x line), by default the measured knee (4 for DeepSeek V4 Flash and
  Qwen3.8 Flash Next with today's 16-row joined products; DeepSeek with
  DSpark takes at most 8, its waves of six to eight plain by the wave-form
  choice's measured costs), and the start logs the value and its source.
  Below the cap memory decides: a request joins the running ones only
  when the budget holds its prompt's state beside theirs, free or freed
  through the reclaim order, and otherwise waits first in the queue
  instead of being admitted to wait or be set aside. Qwen3.8's wave
  workspace is now sized from its widest measured wave instead of four
  prefill chunks: 7.45 → 2.17 GiB fixed with four slots, 5.45 GiB more
  room for conversation state, and a slot costs about 16 MiB of fixed
  buffers (DeepSeek with DSpark about 4 MiB). The reclaim order now covers
  a need at the least total expected cost to restore, so a small need
  takes a few stale plans and graphs instead of spilling a whole idle
  conversation, and admission never spills the conversation its request
  continues.

- Memory is used fully and given back gracefully under pressure (D-055
  and D-090 as amended 2026-10-02): plans and graphs no longer have fixed
  counts and survive a swap, growing into free memory; idle conversations
  are spilled to disk (writing only what changed since their last spill)
  and restored exactly at their next turn instead of being cleared and
  prefilled again (a request set aside for its peers' capacity likewise
  resumes from its spilled state, choosing no token again); everything
  reclaimable goes through one order, GreedyDual-Size over measured
  restore costs with stale entries aging out, taking all that is needed
  or nothing (caches never spill conversations); turn checkpoints are
  never refused for a budget full of caches; a swap that cannot make room,
  or fails (a read error) and is undone, or a first load that fails,
  fails only its requests (503) and the service goes on; and memory
  pressure from other processes (MemAvailable, the kernel's pressure-stall
  information) has the runtime give back what it can, in one trim to a
  target headroom and at a growing back-off while the pressure persists,
  never what the running model's next step uses. New configuration
  keys, compatible with schema version 2 (a minor bump of the 0.x line):
  `[memory] retention_hours` (default 24, the former fixed constant) and
  `spill_budget_gib` (default 128). A DeepSeek plan now holds 2.4 MiB
  instead of 9.9 (its arena holds what its graph uses), and the start's
  guard sets apart only one step's plans, so the DeepSeek DSpark beside
  Qwen3.8 configuration gains conversation-state room and can take a
  diagnostic state snapshot (`swap-table`), which the guard's margin was
  counted twice against before.

- Any chat template now gets chat routes when it can be rendered exactly
  (D-067 as amended): a native family renderer serves a template with a
  pinned hash or one it reproduces on a probe corpus (so repackaged copies
  and Unsloth's Qwen3.8 GGUF variant render natively), and any other
  template renders through a bounded, sandboxed Jinja-subset interpreter
  that matches Hugging Face transformers byte for byte on 29 real templates
  (Gemma, Llama, Mistral, gpt-oss, GLM, Kimi, Phi and others). Only control
  tokens in the template's own text become control tokens. Models whose
  templates were refused at registration now register; the request and
  response formats and `llmp-inference-version` are unchanged.
- Native renderers for the Gemma 3 and Gemma 4 chat templates (template
  support only: llmpalooza has no Gemma runner yet). By hash: Google's Gemma 4
  template (`ae53464b…`, 26B-A4B, 31B, 12B and its QAT GGUFs), the E2B and
  E4B one (`0a2c8073…`) and Gemma 3's (`7de1c58e…`, Unsloth's copies and
  the converted GGUFs surveyed); by probe: Unsloth's Gemma 4 templates,
  Google's of 2026-04-28 and 2026-05-18 (NVIDIA's NVFP4 checkpoints carry
  the first), Gemma 3n's and 270m's. They equal transformers byte for
  byte, in time linear in the conversation (the templates themselves are
  quadratic). Gemma 4 stops at `<turn|>` and at `<|tool_response>` (as its
  generation_config.json lists), so a model given tools ends its turn at a
  call instead of writing the tool's response itself.
- A case a native chat renderer does not implement now renders through
  the interpreter from the template's own text instead of failing the
  request, and native renderings are bounded as interpreted ones are
  (32 MiB of text).

- The community DeepSeek V4 Flash IQ2_XXS GGUF's "chat-v2" template
  (`antirez/deepseek-v4-gguf@f71f23d5`, SHA-256 `87249207…`) has a native
  renderer, so that model registers with chat routes instead of being
  refused for its template.
- Qwen3.8 Flash Next runs from its GGUF quantizations, imported verbatim
  (unsloth's UD-IQ3_XXS checked against llama.cpp b11254 on the same
  file: greedy equal but for near-ties, perplexity within 0.3%); served
  with the NVFP4 checkpoint's tokenizer and template, plain only. Through
  the runtime: 8K prefill 1.65× and decode 1.08× llama-server's. The build
  compiles GGML's tile kernels for Q4_0, Q2_0, Q3_K, IQ1_S, IQ2_S, IQ3_S,
  IQ4_NL and IQ4_XS too.
- DeepSeek chat serves up to four requests at once: their decode steps (and
  DSpark draft-and-verify steps) run in waves that read each weight once for
  all of them, each request's steps bit-identical to the same steps alone (a
  request preempted for state capacity is rebuilt by prefill, and a turn
  reuses whichever branch's cached prefix is longest, so those can still
  change a reply; so can concurrency with a drafter, since DSpark waves may
  take plain decode steps, below). At four matched 7K-token requests the
  completed-token rate rises 29% plain and 4.5% with DSpark. A request whose state cannot grow beside
  its peers' waits (an idle conversation's cached state is cleared first)
  instead of failing; only one that cannot fit alone fails for capacity.
- DeepSeek prefill uses ds4's stage mechanisms by default on GB10: F16 HC
  and attention rows, fused expert sums and pair activations, shared
  quantizations and the D2R Q2_K down product, each where its measured
  shape applies. Served 8K prefill gains 1.26% with identical logits; the
  community IQ2 checkpoint gains 7.21% at 2,048-row chunks, also exact.
- DeepSeek's qualified 4K GB10 IQ2 expert pairs use a scoped J64 kernel,
  improving the measured native 8K prefill rate by 4.15% with exact logits.
- An opt-in native DeepSeek output-A operation fuses inverse rotation,
  Q8 weight preparation and the grouped projection on the qualified
  4,096-row GB10 shape, preserving the diagnostic's complete logits.

- Qwen3.8 chat requests share decode work across up to four independent
  conversations (previously two), drafting two tokens each in a shared wave;
  matched 8K HTTP screens complete 20% more tokens a second at four requests
  and 9.8% more at two, with single-request output unchanged. A request whose
  conversation state does not fit beside its peers' waits, or is set aside and
  rebuilt once they finish, instead of failing; only a request that cannot fit
  alone is refused.
- Qwen3.8 chat requests share decode work across two independent conversations,
  retaining separate state, sampling and cancellation while reusing model weights.
  Compatible multirow HC products and unequal verification heads also share weights.
- `/v1/completions` accepts raw text or exact token IDs, legacy OpenAI
  echo/logprobs and vLLM prompt_logprobs with zero-token scoring. The
  bounded inference subset establishes `llmp-inference-version: 1`.
- A bounded Qwen draft-head capture diagnostic supplies real operands for
  isolated product comparisons without changing serving arithmetic.
- A bounded Qwen routed-down capture diagnostic preserves common-input
  operands for isolated expert-product comparisons without changing serving.
- DeepSeek V4 Flash's community IQ2_XXS/Q2_K GGUFs run natively, including
  F16 attention-compressor APE tables, without requantizing their weights.
- LLM turns reuse stable checkpoints when clients remove earlier reasoning,
  with two private disk checkpoints per model and a 24-hour reuse limit.
- LLM conversation state allocates and spills only used cache extents;
  long context ceilings no longer allocate the whole cache at registration.
- The `llmp` command, with `--version`: the product version (`X.Y.Z` for a
  release, `X.Y.Z-dev.N+g<commit>` otherwise), the commit, the license
  profile, the SDK identity and the target. The build receipt records the
  same version, with its Debian form (`X.Y.Z~dev.N+g<commit>-1`).
- `llmp doctor`, a capability probe: the build, the host (kernel, glibc,
  memory), RDMA ports, the NVIDIA driver and each GPU's compute capability,
  compute mode, VMM support and backing granularity, and whether it is
  unified (the GB10) or discrete. It exits 1 when the host cannot run the
  build: a GPU the build has code for is needed, with VMM and host-backed
  VMM.
- CUDA builds of `llmp` need the NVIDIA driver (`libcuda.so.1`) to start.
- Discrete NVIDIA GPUs are a secondary target (D-082): the x86-64 build
  also has code for compute capability 8.6 (`sm_86`), and `llmp doctor`
  and the runtime accept such a GPU with VMM and host-backed VMM. Arm64
  builds stay GB10-only. Llmpalooza uses one GPU, device 0: `llmp doctor`
  judges only it and warns on a host with more.
- The node's configuration, `schema_version = 2`: `/etc/llmp/llmp.toml`
  and the fragments in `/etc/llmp/llmp.d/`, strict TOML 1.0 in which
  every key has one owning file, with the `[storage]` roles, `[limits]` and
  a cluster member's keys. Every problem is reported with its file, line
  and column, and files or directories that other users could change are
  refused.
- An arm64 Debian package, `llmp`: the `llmp` command, the node runtime
  `/usr/libexec/llmp/llmp-runtime` and `llmp.service`, which runs it
  as the new `llmp` system user with `/var/lib/llmp` as its data
  directory. It depends on the NVIDIA driver 580 or newer. The runtime reads
  its configuration, prepares its storage, checks the host and waits; it
  serves nothing yet. It exits on a fatal signal instead of dumping core.
- `llmp doctor --config FILE` reads a configuration other than the
  default, and doctor now reports the configuration and the storage roles:
  their owners, modes and filesystems.
- The configuration names the models a node serves, `[models.<name>]`
  (still `schema_version = 2`: the keys are new): an installed artifact or
  composition by ID, a speculative drafter, whether to speculate, the
  conversation's context, its prefill chunk, and a tokenizer and chat
  template where the artifact keeps none (D-096).
- `llmp-runtime` serves models by hand, in its own process (D-096):
  `llmp-runtime chat --turn MODEL TEXT...` sends each turn to its model,
  swapping models as needed, and `llmp-runtime swap-table` measures M3's
  swap table between the configured models. Replies are greedy, and
  speculative where a model has a drafter (`--plain` turns it off). CUDA
  builds only.
- With models configured, the runtime service serves a minimal
  OpenAI-compatible chat route (D-097): `POST /v1/chat/completions` (text
  messages; `max_tokens` or `max_completion_tokens`, `temperature`,
  `top_p`, `top_k`, `min_p`, `seed`, `stop`, `stream` with
  `stream_options.include_usage`; reasoning returned as `reasoning`),
  `GET /v1/models` and `GET /v1/models/{id}`, one request at a time,
  first come first served, behind a queue of 64, swapping models as
  requested. It listens on loopback and the node's Tailscale addresses by
  default (`[client] bind = ["loopback", "tailscale"]`, `port = 8114`), or
  on any configured address; none needs authentication, as with other
  engines, and the start log names each listener beyond loopback and the
  tailnet as served without it. The Host and Origin must name the node as
  it listens (its tailnet MagicDNS name included), and cross-site browser
  requests are refused. Connections persist (HTTP/1.1 keep-alive, up to
  `[client] max_connections`, 1,024 by default) on an event loop where no
  slow client holds up another; streams get `: keepalive` comments while
  they wait, swap or prefill. Known fields the route does not implement
  are refused when they would change the answer; unknown fields are
  ignored, their names (never values) counted at
  `GET /llmp/v1/ignored-fields` for loopback clients. Every intake
  bound, timeout and refusal is listed in docs/runtime-serving.md. The
  configuration's schema version stays 2. The runtime can sample (seeded,
  with speculative sampling where a model has a drafter) as well as decode
  greedily. A CPU-only build refuses to start with models configured.
- A DeepSeek artifact whose chat template has no native renderer is now
  refused when it registers, as Qwen3.8's already was, naming the
  template's SHA-256.
- The arm64 package ships NVIDIA's cuBLAS (`libcublas.so.13`,
  `libcublasLt.so.13`, unmodified) in `/usr/lib/llmp` for the runtime, and
  depends on `libgcc-s1`; its third-party notices now include GGML's and
  CUTLASS's, whose kernels the runtime links (D-076, D-096).
- The model support matrix, `docs/model-support.md`: the models llmpalooza
  runs and serves (DeepSeek V4 Flash 0731 with DSpark, Qwen3.8 Flash Next
  NVFP4 with its MTP drafter, Qwen-Image-2.1, and the M2 fixtures), each
  with its checkpoint, artifact, chat template hash, decoding modes,
  evidence and known divergences.

### Changed

- Standalone component targets validate stored preprocessing receipts and tensor roles
  without requiring an import mapping. Converter folds have explicit `source-fold`
  provenance; affected older artifacts require republication. Known performance
  policies and importer-specific approval evidence retain their existing scope.

- Bare JSON tool-call terminal and candidate checks retain incremental envelope
  metadata instead of rescanning accepted argument prefixes.

- Import normalization consumes the authoritative architecture mapping directly,
  with selected-container validation and unchanged mapping provenance.

- Model activation and memory reclamation use named phases with per-attempt
  ownership of snapshots, rollback and accounting (M3.5 U5d).

- Numerical optimizations join component semantics and stored encodings to
  explicit device evidence (M3.5 U4, D-126), consistently for pinned and new
  checkpoints. Missing evidence preserves exact arithmetic and is reported;
  image attention no longer infers qualification from the architecture number.

- Unify dense, borrowed, MQA and QSA attention around one shared IR and scalar
  reference core; merge duplicate fused kinds and name IR and planner operations
  by mechanism; make Qwen2 specs format-independent across GGUF and EXL3 with
  planner-level EXL3 precision matching; and store spatial stage links and
  exports in `spec::Stage` while mapping image text and denoiser blocks onto
  `attention.dense` and `ffn.glu` (D-122; re-import Qwen2 and Qwen-Image-2.1 v1
  artifacts).
- Select common F16 GQA attention kernels at head widths 64, 80, 96,
  112, 128 and 192; gate norm and MoE fusions by their launch limits;
  and expose an alias-specific compact-expert crossover while retaining
  existing default selections.
- An approved checkpoint's envelope (its context, owners and wide context)
  is its settings' default, not a refusal: a configuration naming more is
  served with a warning in the log and in `llmp-runtime settings` (its JSON
  gains a `warnings` array), while the stored spec's context and the
  recipes' own bounds still refuse (D-119 as amended).
- Turn checkpoints are kept by real resources, not two a conversation:
  within `retention_hours`, the 64 a kept record lists, and the spill
  budget's disk beside the spilled conversations, the oldest deleted first.
- A target's planning policy is exactly the rule admission chose, or the
  engine's default where it chose none, never one chosen again from its
  mapping without its gates; admission runs once a registration's settings
  and once its build.
- Plans are found through an index of their keys' hashes, with no
  allocation a lookup; a drafter's feature rows are copied in at most two
  copies a segment.
- A swap makes room round after round while each makes progress, not at
  most four rounds.
- The component build option for the shared route is `shared_route` (was
  `diagnostic_solo`).
- Qwen-Image-2.1 serves from a native v1 import: `llmp import` of the
  diffusers pipeline publishes its three components (each the tensors text
  to image runs: not the text encoder's vision tower and output head, nor
  the VAE's encoder) and a composition that stores the family spec its
  image mapping synthesizes from the pipeline's configs (`spec.json`). The
  runtime admits the composition by its approval (kept files, family spec
  digest, each component's sources, inventory and spec) and builds every
  graph from the stored family; no Qwen-Image number is compiled in. Name
  the new composition ID (`2170db12…`) in `[models]`; the v0
  composition `eca21baa…` is no longer served.
- Serving becomes v1-only (D-118, amending D-113's "v0 stays readable"):
  re-import a v0 artifact with `llmp import` from its archived source (its
  v1 ID differs; nothing is migrated) and name the v1 ID in `[models]`.
  This build still serves v0 artifacts; their serving path is removed in
  the next slice. Qwen3.8's ModelOpt and MLX imports keep their v0 IDs
  until their own v1 artifacts are approved.
- Checkpoint approval is data (D-109, D-118): `runtime/approvals.cc` lists
  each approved source (its files' receipts, its native tensor inventory)
  with the SHA-256 of the spec it executes (a target's stored `spec.json`,
  a drafter's composition with its approved target), its envelope and its
  policy gates. A v1 artifact is admitted when its stored spec's digest is
  its approval's; the expected spec is no longer synthesized from a compiled
  profile, and a stored drafter's composition values are read from the GGUF
  metadata it keeps. Serving reads each architecture's serving recipe from
  its mapping (`spec::ServingRecipe`: default prompt chunks, owner bounds,
  shared route, lanes, the forced DeepSeek policy, the chat route's served
  slice) and dispatches by the stored spec's pipeline kind.
- A model's default context stays within its approval (and its configured
  drafter's): Gemma 4 26B-A4B and 31B, approved to 8,192 tokens, now
  default to 8,192 (with the qualified 1,024- and 256-row prompt chunks)
  instead of 262,144, which their admission refused, so a Gemma 4 entry
  without a `context` starts. No other family's default changes.
- Qwen-Image-2.1 generations change slightly (PSNR 41.26 dB against the
  diffusers reference, from 40.69 dB): the image's fusions are now chosen
  by its planning policy, keeping the shipping path's BF16 activations,
  instead of matched at run time.
- An image model records its stages' graphs when it loads, so its first
  generation after a load replays them (the load takes about 0.2 s longer
  the first time in a process; the generation as much less), and its
  denoising steps upload the inputs that do not change within a generation
  (the text rows, positions and mask) once. The planner runs a node it
  finds twice once and drops the nodes nothing reads, which also removes
  repeated mask builds and unread conversions from the LLM plans that had
  them. Qwen-Image-2.1's pixels change slightly (a faster VAE channel norm
  adds its channels in another order; PSNR 41.26 dB unchanged).
- Gemma 4 26B-A4B and 31B are served by the shared engine (D-107): their
  steps come from the `gemma4` mapping's components and recipes, lowered
  onto the same GGML kernels in the same order, so their replies and
  scores are unchanged. A model entry with a `drafter` (the profile's
  assistant, which M3 refused) now speculates: three drafts a unit, each
  pass reading the target's cache and its final-norm features, then one
  verify. Its greedy replies keep the tokens the target's verify rows
  choose, whose several-row arithmetic is not a one-row decode's, so a
  reply can differ from the same model's plain one (the 26B-A4B's do); a
  Gemma 4 entry's `speculation = true` without a `drafter`, refused
  before, is now ignored and reported, as other families' is. Their
  conversations are kept across a restart in record version 2 (D-112):
  one the previous build kept (version 1) is refused once at the first
  start and its next turn prefilled again; their turn checkpoints are now
  saved past about 1K tokens, as Gemma 2's and Gemma 3's are below.
  Sliding-window caches now truncate a rejected verify's rows instead of
  refusing a verify, which changes the kept layout identity of Gemma 2
  and Gemma 3 conversations the previous build kept: refused once, then
  prefilled again.
- A stored calibration of a Gemma 4 model is not found after this build:
  its key no longer carries `gemma31_production=true` or
  `gemma26_production=true` (the bounded recipes are now the mapping's
  rules, not an option), so the runtime measures and records it again.
- The runtime keeps a model's weight shards open only while the model is
  active: from its activation until another model replaces it, rather
  than every registered model's from startup. Each shard is reopened at
  activation and checked against the size its installed manifest lists:
  a shard gone or resized since startup refuses that activation, naming
  the shard, and the other models still serve. The reopen no
  longer compares the file's identity with the one validated at
  registration, so a shard whose mode or hard links changed meanwhile
  still opens, and one replaced by another file of the same size is read
  as found (page-in does not hash; D-056 as amended on 2026-10-10). The
  open-file limit is
  still raised for the chat route's connections (D-102).
- Gemma 2 and Gemma 3 are served by the shared engine (D-107): one model
  runner over their components' steps, planned under each architecture
  mapping's policy and lowered onto the same GGML kernels in the same
  order, so their replies and scores are unchanged. Their conversations
  are kept across a restart in record version 2 (D-112): a conversation
  the previous build kept for either model (version 1) is refused once at
  the first start, its files discarded, and its next turn prefilled again.
  Their turn checkpoints are now saved past about 1K tokens (the M3
  classes' saves failed there), so a later prompt that diverges after a
  turn's boundary restores it instead of prefilling the whole prompt.
- The directories and files the runtime keeps conversations and calibration
  records in are no longer refused for their owner, modes or links (D-105
  and D-103 as amended on 2026-10-09). What the runtime creates is still
  0600 in 0700 directories (a 0700 directory exactly, not through the
  umask); one found with another owner, looser modes, a second hard link
  or as a symbolic link is used, links followed, and said once per
  directory: a `warning:` line in the start log and in `llmp doctor`
  (`storage.spill: kept conversations ...`, `storage.state: calibration
  ...`); a model's directory the runtime cannot write keeps no
  conversations that start and is left alone (one closed to writes after
  the start looked keeps nothing either, and the model still starts). A
  spill or checkpoint file
  the runtime opens empty is a new file, so a link or another name never
  loses its bytes; a file put under a kept name after the start checked it
  is never adopted or written; a record is
  replaced whole by rename as before. Still refused: what is not a
  regular file (not used, and not opened unless swapped in meanwhile), a
  link to nothing (by name, never created through), and every check of a
  record's identity, digests, sizes and versions.

- The runtime, `llmp doctor` and the importer no longer refuse files and
  directories for who owns them or who else could change them (D-073 as
  amended on 2026-10-09): configuration files, the drop-in directory,
  storage roles, installed artifacts, the process lock, configured
  tokenizer files, calibration records and checkpoints are opened as any
  file is, links followed, whatever their owner, mode, group, ACL, hard
  links or filesystem (FUSE included); an installed artifact's directory
  may be a link, though links inside an artifact are still refused. An
  existing storage role owned by another user, a `spill` or `state`
  directory others may read, an `installed` one anyone can write, a role
  with a default ACL and a configuration file anyone can write are
  warnings in `llmp doctor` and in the runtime's start log. What still
  refuses: a path that cannot be resolved or is not a regular file or
  directory, roles that are one directory or nest, the enrollment anchor,
  a role the runtime's user cannot write, a role on a filesystem other
  than ext4, XFS or Btrfs or failing the direct-I/O probe, a role through
  a link to nothing, and a non-empty `spill` without its marker (whose
  owner is no longer checked). Missing roles are still created 0755
  (`installed`) and 0700 (`spill`, `state`), and the files the runtime
  writes are still owner-only. The configuration's `schema_version` stays
  2: everything accepted before still is.

- Rename the project from jitLLM to llmpalooza (`llmp`, D-111): the `llmp`
  and `llmp-runtime` commands, the `llmp` package, service, user and group,
  `/etc/llmp/llmp.toml` and `/etc/llmp/llmp.d/`, `/var/lib/llmp`, the
  `llmp-inference-version` header, the `/llmp/v1/` routes, `owned_by`
  `"llmp"` and the `.llmp-spill` marker. Existing installations, kept
  conversations and calibration records are not migrated, and a spill
  directory holding the old marker is refused until it is emptied; v0
  artifacts keep their `jitllm-*` format identifiers and load unchanged.

- Omit unused nonfinal prompt dependencies for ordinary plain DeepSeek and
  native/GGUF Qwen while preserving exact initialized state and scoring. Bounded
  repeated prefill takes 1.928%/2.282% less time for DS/native Qwen; speculative
  required-stream consumers remain explicit opt-ins.

- Group independent Gemma3 KV stores under one checked launch, preserving
  exact rounding, heads and initialized state. A bounded C2 screen reduces
  combined paid prefill/decode time 0.45%; other measured recipients remain off.

- Plan upcoming DeepSeek/DSpark and native/GGUF Qwen scalar prompt chunks and
  prepare eligible fresh backing beside current work, preserving exact heads,
  histories and target/drafter state. Short DSpark/MTP prefill takes 0.562%/0.936%
  less time; broader capture and joined-prefill qualification remain open.

- Share exact interval fills for Gemma and DeepSeek host masks and Qwen QSA
  bias, with faster existing Gemma mask validation. GPU mask paths and model
  arithmetic stay unchanged.

- Skip redundant DeepSeek/DSpark and native/GGUF Qwen startup activation
  placement under a checked exact maximum, preserving funding and runtime
  plans. Native Qwen with MTP takes 2.441% less Setup time in the short screen.

- Add optional Gemma4 original-MMVQ input sharing, preserving exact target
  heads/state and existing fused/routed arithmetic. It stays off by default
  after short native factors and startup-cost observations. Skip redundant
  Gemma startup activation placement only under a checked existing maximum;
  exact funding and row-invariant consumer policies remain unchanged.

- Prepare eligible fresh state backing for Gemma2, Gemma3 and both Gemma4
  profiles beside current prefill work, with completion-aware ownership, scoped page-in retirement and exact
  continuation state. Kept restart sources retain their original read path.

- Share eligible Gemma2/Gemma3 Q8 input preparation while preserving original
  MMVQ arithmetic, exact outputs/state and fused one-column FFN behavior.
  Device-specific selection and startup funding cover nine formats through
  eight columns; other product routes retain ordinary execution.

- Choose plain target greedy decode tokens on the GPU for DeepSeek and native/GGUF
  Qwen, preserving complete states and continuation heads. Share scalar
  dispatch/output validation and match host NaN/tie behavior across Gemma too;
  sampling/scoring and draft/verify retain their existing contracts. DeepSeek
  adaptive joined plain waves also publish device IDs while preserving target
  features, drafter-ring writes and exact verify continuations.

- Add a checked opt-in DSpark noncausal GPU block-mask producer through shared
  F16/F32 source contracts. A short screen is neutral, so host masks stay default.

- Accept an optional prepared Q4_1 Qwen3.8 selected draft head, with exact
  scalar/joined prefix arithmetic and restored state. Its payload is 68.75%
  smaller; bounded C1 improves while C2 regresses, so BF16 remains the default.
  Authenticate joined confidence as the actual I32-packed probability view.

- Extend explicitly configured Gemma2/Gemma3 two-owner prefill through 512
  rows, preserving exact wrapped state, restart continuations and llama.cpp
  outputs. Public chunks remain 128; measured reference gaps remain recorded.

- Allow explicit experimental Gemma26 prefill capture with retained frontier
  features, preserving exact feature/state and checked assistant proposals.
  Ordinary and assistant-setup capture defaults remain off.

- Build two distinct future prefill shapes with independent shared funding
  for Gemma2/Gemma3, enabling cold changing-width graph capture without
  advancing future state. Configured 256-row and ordinary 128-row controls remain exact.

- Reduce shared completion wake/queue-lock work and qualify an internal
  partial-weight swap foundation with missing-only reads and exact recovery.
  Ordinary serving keeps full swaps because the cold-switch floor is unmet.

- Build Qwen3.8 native/GGUF causal masks and native headed MTP masks on the GPU
  through the shared F16/F32 producer. Exact outputs and restored state are
  preserved; the tested native C2 and GGUF recipes each charge 8 MiB less.

- Extend Gemma2/Gemma3 actual-root prefill to configured equal two-owner
  chunks through 256 rows with checked mask, cache and query-tile bounds.
  Native 256-row prefill improves about 5.1%/0.8%; matched-cache llama.cpp
  outputs are exact, with measured speed residuals retained. Public chunks
  remain 128 rows.

- Extend Gemma2/Gemma3 actual-root prefill to flexible 2–128-row query tiles
  and partial tails, preserving original MMA arithmetic. Short 64-row native
  comparisons improve prefill 10.5%/2.8%; ordinary chunks and public contexts
  remain unchanged. Serving tails/state/restart and fresh reference outputs
  remain exact.

- Read actual K/V roots for compatible two-owner 128-row Gemma2/Gemma3
  prefill, preserving MMA arithmetic and removing cache-packing copies.
  Short native prefill screens improve 8.7%/2.9%; fresh Gemma2 llama.cpp
  bookends are level with exact outputs. Gemma3's primitive is checked
  through trained 131K cache width; public context limits are unchanged.

- Build DeepSeek target causal/ring masks on the GPU through shared source
  checks, preserving exact sparse-query dimensions and unchanged outputs/state.
  A bounded two-owner prefill screen improves about 0.7%; DSpark block masks
  retain their separate visibility contract.

- Plan and capture upcoming Gemma2 prefill shapes during current work, and
  forward checkpoint-aware joined prompt hints for Gemma2 and Gemma3. A short
  Gemma2 first-traversal screen improves 5.3%; the short warm comparison is
  0.6% slower, with no warm benefit claimed.

- Build Gemma2 causal/ring masks on the GPU using shared mask-source checks
  across Gemma2, Gemma3 and Gemma 4. Wrapped two-owner prefill improves about
  6.6%, with exact heads/state and checked restart replay.

- Keep a small reserve of device memory handles created ahead of need and
  zero new conversation state on the GPU instead of reading it from disk:
  growing state during long prefills costs about 38% less (Gemma3 at 131K
  positions now at or slightly ahead of stock llama.cpp).

- Keep a swapped-out model's backing mapped until the incoming model's
  loads take it, moving each unmap beside a page-in read: LLM swaps no
  longer wait 1.3–1.9 s for eviction (DeepSeek ↔ Qwen3.8 6.1–8.4 s from
  7.9–9.7 s).

- Plan Gemma3's upcoming prefill shapes beside the current chunk's device run
  and capture their graphs ahead of their first run, so long prompts replay
  nearly every chunk; trained-maximum first-traversal prefill falls about 9%.

- Build Gemma3 causal and ring masks on the GPU from checked positions,
  avoiding host mask construction, scanning and staging in ordinary serving.

- Reduce Gemma3 causal-mask construction and validation work on the host,
  retaining exact checks of every supplied mask value.

- Admit explicit Gemma3 context up to 8448 with one request slot, preserving
  the 4096 default and two-slot bound at 4K. Checked 8K literal continuation
  and model-switch state restore use the ordinary native runtime.

- Avoid temporary KV prefix copies for unequal-length Gemma4 two-slot decode
  at context<=4096, preserving checked common attention geometry, physical-ring
  continuations and cached conversation restart on both approved profiles.

- Avoid temporary full-prefix cache copies for unequal-length Gemma3 two-slot
  decode, preserving checked attention geometry and checkpoint replay.

- Avoid temporary full-prefix cache copies for unequal-length Gemma2 two-slot
  decode, preserving checked attention geometry and checkpoint replay.

- Add an experimental Gemma4 two-owner common-read option for unequal cache
  prefixes. Real temporary K/V and mask padding preserve original cache bounds
  and restore exact bounded stock heads; ordinary serving selection is unchanged.

- Join compatible Gemma3 plain prefill chunks across two owners, with a funded
  256-row wave and unchanged 128-row per-owner state. Packed prefill attention
  and real temporary padding for unequal decode reads preserve bounded stock
  quality, initialized-state restore and independent cache cursors.

- Extend the experimental Gemma2 Q8_0 runner to opt-in two-slot softcap50
  owner decode, with funded attention, wrapped-ring restore/replay and bounded
  stock quality controls. Explicit diagnostic defaults retain ordinary attention.

- Gemma4 serving reuses an unchanged cohort's execution closure within a held
  request, avoiding a redundant scheduler call between completed units. Changed
  selections and state growth, Clear and restore still refresh their closures.

- Add an experimental bounded Gemma2 2B Q8_0 native runner with authenticated
  import, checked state and graph replay. Its representative one-slot control
  matches stock full heads exactly; broader model qualification remains open.

- Less host time between decode steps, for every model. A request's steps
  run on the driver thread, which waits on each step's fence itself, so no
  other thread is on a step's path (D-106). The LLM runners no longer call
  the scheduler before each step. Greedy Gemma steps without logprobs also
  choose the token on the GPU and copy back only the token. A decode step's
  host round trip falls from 85–188 µs to 6–11 µs on the Gemma controls;
  Gemma26/Gemma31 C1 and
  Gemma31 C4 decode are 2.8%/0.6%/1.1% faster, with identical tokens.

- Select a bounded Gemma26 serving recipe at context<=8192 and at most four
  slots: 1024-row prefill fallback, norm chains, MoE route/reduce and joined
  owner attention. C1/C4 go from 57%/41% to within 0.6% of llama.cpp v0.6.0,
  with its exact greedy tokens. Larger configurations keep the prior recipe.

- Faster new conversations and repeated prompts on every LLM: clearing a
  conversation zeroes its state in place and reuses the memory for the next
  one, and a CUDA graph is now captured beside its shape's second run instead
  of stalling the device. The bounded Gemma31 recipe also uses llama.cpp's
  fused quantized gate/up/GeGLU product for decode. Gemma31 C1/C4 paid cycles
  go from 3.15%/3.51% to 0.73%/1.36% behind llama.cpp v0.6.0.

- Select the checked dense Gemma31 serving recipe at context<=8192 and at most
  four slots: prefill fallback 256, existing norm chains and eligible owner
  attention. Gemma26, larger configurations and explicit diagnostics retain
  their prior policies and calibration identities.

- Update the native GGML kernel dependency to llama.cpp v0.6.0 / GGML
  0.26.0, preserving bounded native dispatch and carried compact/paired
  optimizations. Historical comparisons retain their original pins; new
  quality and performance qualification uses the new release separately.

- Qwen3.8 GGUF concurrent decode now shares dense and routed quantized
  products and the full target head across requests, with byte-identical
  full logits and state against the original waves, including graph replay.
- DeepSeek's concurrent requests now run their own attention and state
  work at the same time, one CUDA stream each, inside a wave, with each
  request's results unchanged bit for bit. At four 124-token requests
  plain waves complete 4.7–4.8% more tokens a second, and with DSpark
  4.6–5.1%, whose wave-form costs were measured again (at 7K, where
  prefill dominates, the gains are within noise).
- A DeepSeek DSpark wave now drafts every request's block in one graph,
  reading the drafter's and the head's weights once for all of them, with
  each request's drafts unchanged bit for bit: a DSpark wave of four or
  five requests takes 4–5% less time (same session). End to end at four
  and five requests the rate is unchanged within noise, since the chosen
  form mostly runs plain waves there.
- With its DSpark drafter, DeepSeek now chooses per wave of concurrent
  requests between draft-verify waves and plain decode waves: the tokens
  draft-verify waves accept against a cost measured for each wave width.
  No clock enters the choice, so the same requests in the same waves choose
  the same forms. At four requests this is up to 7% faster than DSpark
  waves alone and 1–4% under plain waves without the drafter. A lone request
  keeps speculating (31–54% faster), and a wave with a sampling request
  always speculates, so seeded replies repeat. A greedy reply at several
  requests may now differ from the same request alone, and between runs
  whose requests arrive in a different order (the forms' arithmetic
  differs, and arrival order decides which waves a request joins). A
  model's new `wave_form` key (`"auto"`, the default, `"speculative"` or
  `"plain"`) fixes the form; with `"speculative"` a reply at several
  requests equals the reply alone.
- Concurrent chat prompts now prefill shortest remaining first, so a short
  prompt no longer waits behind a long one that arrived first, and equal
  prompts each stream their first token in turn instead of all at once at
  the end. A long prompt still runs after at most 12 others' prefill units,
  and a generating request waits for at most one prefill unit between its
  tokens. With one ~32K-token DeepSeek prompt followed by three short
  ones, the short prompts' first tokens come at 4.8–6.9 s instead of
  18.6–19.0 s; four 124-token chats' at 0.9 / 1.7 / 2.5 / 3.4 s instead
  of 3.0–3.5 s. Throughput is 1–2% lower for four equal prompts (their
  first finishers decode between the others' prefill units). DeepSeek's
  replies are unchanged.
- DeepSeek's output-A/HCA prefill now also takes a prompt's last, partial
  chunk (64 rows or more), which passes every quality control under the
  owner's tie-aware greedy rule: 7K-token chat completes 5% faster alone
  and 7.3% (community GGUF) and 8.5% (0731) faster four at a time (first
  token 8.3 → 7.3 s on the
  community GGUF, 9.4 → 8.3 s on the 0731). Replies change with the
  arithmetic.
- The chat route's limits follow real resources and default to permissive
  (D-102): no message, content-part or stop-string counts, no 8 MiB
  message or 128-byte stop caps, and no queue count, queue wait or
  non-streaming deadline unless `[client]` sets one (`max_queued`,
  `queue_wait_seconds`, `deadline_cap_seconds`). Requests' host memory is
  one pool charged by what each request really holds, bodies as they
  arrive, parses, queued requests, renderings, tokenizations and a
  stream's unread output included: 256 MiB is set apart at the start, and
  past it the pool grows within the memory budget, displacing caches and
  idle conversations, and shrinks as requests end (`request_memory_bytes`
  caps it); what cannot fit is refused before the work (413, or 503 to
  retry); a prompt that cannot fit the model's context is refused before
  it is queued. A body follows the pool and the largest configured context (on
  a Spark serving DeepSeek V4 Flash at 300K tokens, 221 MiB instead of 16
  MiB; `max_body_bytes`), connections the open-file hard limit. A stream
  whose client stops reading pauses, and if it keeps others waiting
  yields its place, continuing once its client reads. Large bodies parse off the I/O
  thread; stop strings of any number are matched together in one shared
  automaton; a chat text longer than any prompt that fits is refused
  before rendering, and tokenization works in bounded windows. JSON
  values follow the body (a 1M-token ID prompt parses) and nest to 64.
  `max_tokens` and literal top scores (`logprobs`, `prompt_logprobs`, now
  up to the vocabulary) are bounded by the model, not fixed caps. A stall
  is reported (log, health, service status) without failing requests
  (`stall_action = "fail"` restores that); head and body timeouts count
  inactivity (`request_inactivity_seconds`); a stream whose client stops
  reading pauses at a completed step instead of being dropped after 30 s
  (`write_inactivity_seconds` restores a drop). Chat templates have no
  step or work caps: a long rendering ends when its request does, and
  rendering bounds follow the model's context and memory. The
  configuration no longer caps models at 16 or contexts at 1,048,576. All
  earlier configurations and requests stay valid: `schema_version` 2 and
  `llmp-inference-version` 1 are unchanged.
- DeepSeek serves its output-A/HCA prefill by default on full 4,096-row
  chunks, where it passes every registered quality bound on both GGUFs:
  7K-token chat completes 7% faster alone and 12% faster four at a time
  (first token 9.8 → 8.3 s on the community GGUF, 10.9 → 9.4 s on the
  0731). Replies change with the arithmetic. Those plans are no longer keyed by chunk position, so chunks of
  one shape share a plan. Four-request decode waves read each dense
  weight once for all requests and find their routed experts' rows with
  one warp's ballots instead of one thread's scan (bit-identical to
  before, waves 7% faster). With output-A/HCA, four concurrent 7K-token
  chats complete 15–16% faster on both GGUFs, and four 124-token chats
  12% faster.
- DeepSeek prefill is 3–4% faster on both GGUFs through the runtime
  (matched 7K-token prompts, one and four at a time). A prompt's last,
  partial prefill chunk of 64 rows or more now takes the stage mechanisms
  a full chunk takes (D2R; the IQ2 pair's activation write-back on
  compact chunks; with the opt-in output-A/HCA prefill, those too: the
  community GGUF's 7K prompt then prefills 21% faster, in 6.8 s). On the
  community GGUF partial chunks' logits move with D2R (32K perplexity
  +0.53%, inside the 3% bound). The 0731 UD-Q2_K_XL GGUF's F32
  hyper-connection mixes, IQ2_XS gate/up experts and Q5_K/Q6_K shared
  experts now take the fused HC norm, expert sum, write-back and shared
  quantization; its replies are unchanged byte for byte.
- Chat templates' case mapping (`upper`, `lower`, `capitalize`, `title`,
  case-insensitive `dictsort`, `sort` and `unique`, the `lower` and `upper`
  tests) is now Python's in full, as transformers renders it, instead of
  ASCII-only: `ß` upper-cases to `SS`, Cyrillic and Greek map (with the
  final sigma), and the `title` filter follows Jinja2's own rule. The
  Unicode tables gain the case data of UCD 15.1.0's SpecialCasing.txt and
  DerivedCoreProperties.txt.
- DeepSeek prefills in 4,096-row chunks by default (was 2,048): 12-15%
  faster at 8K-32K tokens on the 0731 GGUF for 1.28 GiB more fixed memory.
  Above a 262,144-token context the default stays 2,048 rows, whose
  smaller attention mask leaves a 1M conversation its state (there 4,096,
  capped to 4,024 rows, fixed 2.14 GiB more). Any DeepSeek chunk is now also capped
  where that mask fits the kernel's 32-bit strides (13,528 rows at a
  262,144-token context, 4,024 at 1M); wider chunks refused a deep
  prompt's last chunks.

- DeepSeek fast prefill fuses eligible Q-head normalization and rotation,
  preserving native F32 arithmetic without a normalization intermediate.

- DeepSeek's fast wide prefill combines its six expert outputs in one F32
  kernel with the original multiplication and addition order; unsupported
  shapes and reference plans keep their ordinary operations.
- DeepSeek fast production prefill with the measured Q4_K head computes
  only its frontier head row, while retaining every DSpark feature row;
  reference, verify and scoring paths keep their requested heads.
- DeepSeek count-based HCA attention retains the ordinary MMA path;
  selected-list CSA and window attention keep shared sparse gathers.
- Qwen3.8's small-output MXFP8 verify products use measured GB10 warp
  schedules, preserving their arithmetic and adding no workspace.
- DeepSeek fast prefill can share sparse KV gathers across eight queries
  and expert input preparation across gate/up products, and use compact
  expert tile lists from 2,048 rows, retaining F16 caches and the original
  quantized weights. Primitive/reference plans keep their original
  dispatch choices.
- Models default to 262,144 tokens of context (was 8,704). DeepSeek V4
  Flash accepts up to its trained 1,048,576-token ceiling; Qwen3.8 Flash
  Next remains capped at 262,144, checked before model allocation. Growing
  state remains subject to the physical memory guard. The chat route accepts
  bodies up to 16 MiB and message text up to 8 MiB, with `max_tokens` parsed
  up to 1,048,576; its aggregate intake bounds and prefill chunk policy stay
  unchanged. Schema version remains 2.
- Qwen3.8 greedy MTP decoding chooses depth two or three from acceptance;
  state restore preserves the schedule. Draft confidence is computed only
  when requested. The experimental importer can prepare a selected BF16
  draft head and original token-ID map from an external vocabulary list.
- Sampling with `top_k = 1` uses the greedy path, including speculative
  verification, avoiding candidate selection and random draws while keeping
  parameter and logit validation and the same token choices.
- DeepSeek V4 Flash's per-token cost no longer grows with the
  conversation: its window cache is a ring of the window and a prefill
  chunk, its attention reads only each token's window and the indexer's
  selected (or HCA's visible) compressed rows, and its indexer scores on
  tensor cores and selects deterministically. Through the runtime on a
  GB10, prefill runs at 471 / 466 / 445 tokens a second and plain decode
  at 21.6 / 21.2 / 20.5 at 32K / 64K / 128K tokens of context (before:
  333 / 230 / 154 and 14.7 / 10.8 / 7.1); a long prompt now repeats bit
  for bit, and its greedy tokens may differ from the previous build's at
  near-ties. At `context = 262144` it maps 3.5 GiB beside its weights
  (was 20.4), and with its DSpark drafter it now starts there (it was
  refused above 143,360).
- The runtime's start keeps 6 GiB (was 4) beside the largest model's
  weights for memory the node does not count, and also counts the
  largest chunk inputs a model builds on the host; a configuration that
  started within 2 GiB of the old limit may now be refused, with a
  message naming both.
- Sampling (temperature above 0, the chat route's default) no longer sorts
  the vocabulary for each token: a draw takes 0.45 ms instead of 5.9 ms at
  DeepSeek V4 Flash's vocabulary and 0.87 ms instead of 11.9 ms at
  Qwen3.8's, on a GB10, speculative verification included. A seed still
  draws the same tokens every time within a build, from the same
  distribution, but may draw other tokens than the previous build did.
- On the chat route, a client that shuts its sending side after a whole
  request now gets its response (a stream starts at once; a non-streaming
  response on HTTP/1.1 is preceded by an interim `102 Processing`), where
  before the request was cancelled; a client that closes
  its connection still cancels. A connection kept alive between requests
  no longer keeps its last request's body or response allocated.
- The runtime prefills in wider chunks by default, chosen per model
  (DeepSeek V4 2,048 rows, Qwen3.8 4,096): an 8K-token prompt prefills
  1.48× (DeepSeek) and 1.78× (Qwen3.8) as fast as in the 512-row chunks
  before, for about 1.4 GiB more workspace. `[models.<name>]
  prefill_chunk` sets a model's chunk; it is capped at what the model
  allows at its context, so every context the configuration accepts
  starts (at 512, DeepSeek was refused and Qwen3.8 with its MTP drafter
  failed at startup).
- The chat route no longer ends a request after a fixed 600 s, which
  failed healthy long prefills: a request fails only when the model
  backend makes no progress (no swap, prefill chunk or decode step
  ending) for `[client] stall_seconds` (default 120), with a 504 before
  the headers or an in-stream error after; the backend is then reported
  unhealthy (in the log and `systemctl status`) and requests get a 503
  until it makes progress again. A stream has no deadline; a
  non-streaming request's is scaled to its work at the model's
  `prefill_floor_tok_s` and `decode_floor_tok_s` (100 and 5 by default),
  three times over, at most `[client] deadline_cap_seconds` (4 hours). A
  stream waiting in the queue no longer runs out of time there. The
  configuration's schema version stays 2 (new keys).
- The chat route now ends a request whose client left, whose deadline
  passed or whose runtime is stopping between prefill chunks (and after a
  swap), not only once generation starts; the service keeps serving, and
  the conversation keeps the chunks that ran, so a retried request
  continues from them.
- Qwen3.8 Flash Next no longer slows with context: it caches each
  block's indexer key once, selects its attended cells on the GPU at any
  depth, and attends those 2,051 cells alone instead of every cell. Through
  the runtime on a GB10, plain decode is 26.8 / 26.0 / 24.4 tok/s at 8K /
  64K / 256K (before: 27.4 / 17.1 / 6.8) and prefill 2,314 / 2,359 / 2,178
  tok/s (before: 2,320 / 1,367 / 487). Its selection now repeats bit for
  bit at any depth, speculation with its MTP drafter works to its
  configured 262,144 (it was refused above 32,768), the prefill chunk
  stays 4,096 rows at every context (it was capped at 2,040 at 262,144),
  and at 262,144 the memory it reserves falls from 24.7 to 11.1 GiB (its
  peak from 101.2 to 84.7 GiB). Greedy tokens may differ from the
  previous build's at near-ties.
- Qwen3.8 Flash Next's plain decode is 5–8% faster (its recurrent state
  updated in place, and fused and clustered one-row kernels); its greedy
  tokens may differ from the previous build's at near-ties.
- Qwen-Image-2.1 generates about 8% faster (33.4 s against 36.2 s at
  1,024², 40 steps, weights resident, on a GB10) with the same BF16
  numerics (pinned cuBLASLt products, fused norms and attention,
  implicit-GEMM VAE convolutions, a CUDA graph per denoising step), its
  operations run through a plan bound against the implementation
  registry; its pixels differ from the previous build's in the last bits
  of the VAE's convolutions (still within the bounds of diffusers' image).

### Removed

- Kept conversation records of version 1 (the M3 runners' one spill file a
  slot) are no longer read: such a record is refused and its files
  discarded at start; version 2's records read as before (D-112 as
  amended).
- `Llm`'s per-request state interface beside its per-branch one: a model
  implements the per-branch interface alone.
- The legacy GGML plan caches and graph runs (`engine/planned.h`,
  `engine/graph_runs.h`), unused since the per-family runners went.
- Serving prepared-artifact format version 0 (D-118): a configured target or
  drafter of v0 is refused at registration with "re-import its source
  checkpoint with `llmp import SOURCE --store DIRECTORY` and configure the new
  artifact ID". Re-import each v0 artifact from its source (the published v1
  IDs are in [model-support.md](docs/model-support.md)); nothing is migrated in
  place. The Qwen3.8 GGML-experts v0 artifact has no v1 import and is no
  longer served. The per-family
  profiles and tensor binders, and the v0 spec synthesis behind them, are
  deleted; `llmp import` produces the same v1 artifacts as before. The
  Qwen-Image composition is unchanged until its own migration.
- The per-architecture diagnostic selectors: `LLMP_DIAGNOSTIC_COMPONENT_QWEN`
  and `ServingOptions`' `diagnostic_component_solo` and
  `diagnostic_component_all` (with their "diagnostic shared alias" log
  lines) are gone; `LLMP_DIAGNOSTIC_STARTUP_HOST_BYTES` and
  `LLMP_DIAGNOSTIC_NODE_CAP_BYTES` remain the funded diagnostic pair on
  their own. The DeepSeek-named prefill overrides become the shared route's
  `prefill_lookahead`, `prepare_state` and `state_only_prefill`, which now
  apply to every shared-route model (Qwen3.8 too, not only DeepSeek), and
  `gemma3_trained_max` the generic `trained_max_context`, all internal. The
  service logs a "serving final" line for every LLM in place of the
  per-family ones.
- The internal Gemma 4 diagnostics of `ServingOptions` (`gemma_joined`,
  `gemma_row_invariant`, `gemma31_production`, `gemma26_production` and
  the prepared-state switch, with `RunService`'s matching parameters and
  its "Gemma diagnostic" log lines) go with the M3 Gemma 4 class, runner,
  assistant and graph builders they controlled, deleted now that Gemma 4
  26B-A4B and 31B and their assistants serve on the shared engine (D-107);
  absent from the configuration and the command line, they affect no
  setting. The recipes they selected are the `gemma4` mapping's policy
  rules, whose prompt chunk (1,024 rows for the 26B-A4B and 256 for the
  31B within a context of at most 8,192 and four slots) serving takes
  where no setting overrides it. The internal Gemma 4 benchmarks
  (`llmp_gemma26_*`, `llmp_gemma31_production`, `llmp_gemma_owner_c4`,
  `llmp_gemma_joined`, `llmp_gemma_prefill`, the attention, dense-FFN and
  assistant-fixture captures and the rest built on those classes) are gone
  with them.

- The internal Gemma 2 and Gemma 3 diagnostics of `ServingOptions`
  (`gemma2_*`/`gemma3_*`: owner and flexible owner prefill, shared Q8,
  prefill lookahead, capture ahead, prepared state and host masks, and the
  `component_engine` set; the settings hook `gemma3_trained_max` stays) go
  with the M3 classes they controlled, deleted now that both models serve
  on the shared engine (D-107); absent from the configuration and the
  command line, they affect no setting. The internal
  `llmp_gemma2_probe`, `llmp_gemma3_probe` and the other Gemma 2 and Gemma 3
  runner probes built on those classes are gone with them.

- The temporary complete ds4 prefill reference (the `llmp_ds4_complete`
  benchmark and its original cache, HC/norm, compressor, repack, product,
  indexer, sparse-attention and router/FFN stages, weight preparation and
  state layout) no longer builds into llmpalooza's libraries; production keeps
  the original output-A and F16-Q token-tile HCA cores it selects. Commit
  `ec8af04` holds the rest
  ([report](docs/experiments/ds4-complete-plan/README.md)).

- The DeepSeek V4 model settings `prefill_outa_hca` and
  `prefill_outa_hca_partial` are removed from the configuration, the settings
  listing and the example configuration. Nothing consumed them once DeepSeek
  moved to the shared engine (an override already refused DeepSeek at start); a
  configuration that names them is now refused as an unknown setting. The
  output-A/HCA prefill is the fixed production form, and the calibration
  fingerprint keeps its `prefill_outa_hca=true/true` token so existing records
  stay in force. Pre-1.0 minor-release break (D-062).

- The legacy DeepSeek V4/DSpark runner (`Dsv4Runner`, its plans and graph
  builders, the `Dsv4` serving class and the M3-era DeepSeek harnesses) is
  deleted with G4d. Its internal `ServingOptions` flags
  (`diagnostic_plain_device_tokens`, `predicted_prefill_hints`,
  `prefill_output_intent`) and the `LLMP_DIAGNOSTIC_COMPONENT_ENGINE` selector go
  with it: the funded diagnostic tuple
  (`LLMP_DIAGNOSTIC_STARTUP_HOST_BYTES`, `LLMP_DIAGNOSTIC_NODE_CAP_BYTES`) now
  selects every component pipeline, and `LLMP_DIAGNOSTIC_COMPONENT_QWEN` keeps
  its narrower scope. None is in the configuration or the command line.

### Fixed

- Qwen3.8 and DeepSeek V4 no longer bring back a speculative verify's
  rejected rows after a restore. A spill or swap write-back taken while a
  verify still owed its restore or commit (or awaited its Accept) marked
  the spill file current; the owed work then ran as a job of its own, and
  the next incremental spill kept the file's older bytes. Such a state's
  next spill now writes it whole.

- DeepSeek V4 with its DSpark drafter no longer stops on four-request
  speculative waves ("graph reader index exceeds the startup scratch
  allowance"): a wave of 9 to 15 rows plans more graph nodes than the
  widest waves the start measures, so the graph-reader allowance now keeps
  a quarter's margin over the measurement, as the activations do.

- `swap-table` continues Qwen3.8 after a saved-state swap: its
  continuation check compared the 248,320-wide logit rows with the
  tokenizer's 248,077 tokens instead of the model's logit width.

- Ending a residency lease while external work finishes no longer risks
  accessing a released lease record or leaving its holder waiting forever.

- DeepSeek V4 with its DSpark drafter funds each step's history for its
  whole verify (the anchor and every draft). A long generation whose
  verify kept more tokens than the funded room could abort the runtime.

- A start whose memory cannot fund a model's kept conversation histories
  (or whose available memory is unknown) no longer deletes them: unknown
  availability bounds nothing, and a model with a valid record that does
  not fit leaves its records for a later start (that run keeps none of the
  model's own conversations across a restart).

- A Gemma4 slot whose verify rollback fails is quarantined without a
  pending verify, so Clear recovers it instead of every later call
  refusing it.

- A request step whose end was never proven no longer stalls the next step
  on its stream until the hang check: the idle task is cancelled, as the
  request's end already did.

- GPU and model-file tests resolve artifacts under `LLMP_TEST_MODELS`
  (else `~/.local/share/llmp`) and scratch under `LLMP_TEST_SCRATCH`,
  and skip when the files are absent instead of failing.

- Bounded Gemma26 and Gemma31 serving keeps the full final FFN before
  publishing a frontier head, matching llama.cpp v0.6.0's arithmetic.
  Gathering the final row earlier changed same-geometry prefill likelihoods
  and could change the next token. Both small three-head controls now match
  stock byte for byte; head publication remains bounded by request slots.

- Package purge deletes kept conversations as the service user, so a
  service-owned parent path swapped for a symlink cannot redirect a root
  deletion.

- Spark jobs keep their GPU lock in the child as well as the supervisor;
  losing the supervisor no longer admits another job alongside orphaned
  work, and orphaned or waiting jobs report the host busy.

- Paused streams no longer accumulate keepalive comments behind pending
  output. Prepared prompt tokens keep their request-memory charge through
  shared continuation ownership; continuation copies and growing request
  token histories are charged until they retire.

- Incremental stop matching compacts retained text only after enough
  bytes are emitted, avoiding repeated copies of a long stop prefix.

- Kept-conversation work queues retain only the latest record for each
  model slot and release invalidated queued copies immediately, bounding
  stale history storage while hashing yields to inference.

- Jinja scope variables count toward live memory. Mixed integer/float
  comparisons preserve integer precision and unordered NaN comparisons;
  float floor division and remainder match Python, while complex powers
  and overflowing float powers are refused.

- A failed diagnostic DeepSeek tensor dump keeps its pinned destination
  alive when submitted DMA retirement is unproven.

- The engine's paging kernels (range fills, n-gram row gathers, draft-id
  clamps) clear the CUDA error a failed launch leaves, so it no longer
  surfaces in a later, unrelated check, and the DeepSeek prefill HCA
  kernel refuses graph capture, whose replay would keep a stale first
  position.

- A model whose post-load checks fail is evicted instead of staying
  resident unchecked; a first load after a failed swap makes room through
  the reclaim order; a graceful stop's spills and an oversized idle spill
  respect the spill budget; and a refused swap keeps the incoming model's
  conversations kept on disk.

- The chat route answers `Expect: 100-continue` once even when the body
  waits for request memory, and a half-closed client whose body waits no
  longer spins the I/O loop.

- Kept-conversation records with a speculation cursor past their tokens are
  refused, a failed record removal is retried, and a record whose write
  failed after its rename is still removed when its state changes.

- Jinja `map` of `map` counts against the evaluation depth bound (a hostile
  template could overflow the stack); `%d` and `int` of non-finite or
  out-of-range floats, `float` of large integers, `'1e500'|float` and
  `split(None, n)` remainders now match Python or refuse.

- Jinja strings and renderings count their trusted ranges (16 bytes each)
  toward the string and output bounds as they grow, so a template that
  alternates its own text and a client's byte by byte can no longer hold
  several times those bounds uncharged; the rendering's ranges pass to the
  caller without a copy. The control tokens an interpreted template places
  count toward the same output bound, so a template repeating a short
  control token is refused rather than holding several times its output in
  spans, and a container's repr and `tojson`'s text are charged while they
  are copied.

- Qwen drafts record every catch-up row for incremental spill, so kept
  drafter cells are never stale on disk; DeepSeek's `pair_glu_q8` accepts
  partial prefill chunks; and several kernel host checks are tightened.

- A Qwen prefill chunk with the drafter's injection after a verify (a new
  turn after speculative decoding) catches the MTP drafter up on every row
  the verify kept, each from its own streams, instead of running the
  previous position from a stale stream row and skipping the others; turns
  no longer lose acceptance at their start (target output was unaffected).
  A scalar Qwen verify saves the stream rows it overwrites, as a wave's
  does, and a one-request speculative step (Qwen or DeepSeek) whose host
  judgement fails undoes its verify and keeps the conversation's prefix,
  as a wave already did, instead of clearing it.

- Package purge refuses a symlinked spill directory before its root `rm`.

- Adopting a kept conversation empties every extent of its spill file that
  its record does not list, so a record whose removal had not reached the
  disk before a power loss can no longer hand a later conversation's state
  to the adopted one. The keeper writes and syncs a record and syncs its
  directory outside its lock (only the rename is under it), so a turn
  restoring a spilled conversation never waits on a disk sync, no
  stale temporary record is left behind, and a temporary record is renamed
  into place only while it is still the file the keeper wrote.

- The spill budget counts what a spill or swap will hold on disk: an idle
  spill and a graceful stop's count the conversation's whole state, and a
  swap's write-back the outgoing model's whole state, never deleting the
  incoming model's conversations for it, nor any until the swap's room is
  made; a spill that cannot fit deletes nothing, and a graceful stop keeps
  the most recently used conversations. A reclaim whose victim gives back
  nothing selects again without it, and idle state that could only be
  dropped while a continuation holds it is no longer offered.

- Jinja container comparisons and member lookups poll request cancellation
  while scanning their values, including repeated references to large strings.

- The cross SDK includes a pinned ARM shared GCC support runtime for
  cuBLAS, allowing CPU test discovery under qemu without changing the
  compiler or llmpalooza's static C++ runtime.

- Package provenance checks read license sidecars for JSON and other
  noncommentable source files, while code still requires embedded headers.

- CUDA-free builds exclude the CUDA-only DS4 source component from their
  receipts, so the strict source inventory agrees with the CPU build.

- DeepSeek automatic calibration exploration respects explicitly supplied
  wave-cost prefixes, including zero, which always speculates. Widths
  outside the override prefix retain bounded first-use exploration.

- Calibration records become stale when effective context or DeepSeek
  automatic wave-cost overrides change. Equal fallback-valued overrides
  and different override-prefix lengths have distinct dependencies. The
  existing record format and fresh serving defaults are unchanged.

- Fresh LLM rows in `swap-table` now generate one greedy token before
  ending their clocks. Their complete prefill-head hashes still check
  repeatability; saved-context returns retain their decode-step endpoint.
  Previously fresh rows stopped after prefill, before token selection.

- Concurrent chat prompts again prefill in order of what is really left:
  a conversation's stale history no longer counts as reused, a unit that
  prefills nothing ages no other prompt, and a started prompt about to
  finish goes before an aged one. At four 7K-token requests the second
  first token comes at 14.7 s again instead of 18.8 s.
- DeepSeek's DSpark drafter failed with the community IQ2_XXS GGUF ("get_rows
  of quantized rows into F32"): the verify's device lookup of the drafts'
  rows now takes its F16 token table. With the drafter, one 7K-token chat
  completes in 24% less time than without it (16.98 against 12.93 tok/s),
  a short one 35% less. Draft ids are now bounded to the vocabulary before
  any lookup, so a drafter's non-finite logits can no longer index past a
  table.
- A chat prompt over 4 MiB of rendered text, or over 4,194,304 tokens, was
  refused by the tokenizer's library defaults below the body limit and
  DeepSeek's 1M context; the chat and literal routes now bound
  tokenization by the rendering and the model's context.
- `llmp-runtime` refused to start the community DeepSeek V4 Flash
  IQ2_XXS GGUF ("a DeepSeek V4 wave needs every layer in the fast plan's
  fused form"): its F16 hyper-connection mixing weights kept every layer
  off the fused form that four-request waves need. The fused form now
  takes F16 and BF16 mixing weights (and quantized ones through GGML's
  product), so that artifact starts, serves four requests in waves
  (+41% at four 7K-token requests over one at a time), and its
  single-request decode takes the fused form too (+4%). An artifact whose
  weights cannot form a wave is now served one request at a time, with a
  log line at start, instead of being refused.
- Stopping `llmp-runtime` with two models configured, after a swap,
  logged "a compute stream could not be fenced" and exited with a failure
  status: the stop's fence of the swapped-out model's stream tried to page
  its spilled state back in, beyond the execution budget. The fence now
  leases nothing, so it neither fails nor reads anything back, and that
  stream's work is proven complete before its model is released; if a
  fence still fails, the stop leaves the models' memory as it is rather
  than release it under work that may still be running.
- A `/v1/completions` request (or a serial chat turn) refused for
  conversation-state capacity stopped the runtime, even when idle chat
  conversations' retained state held the capacity it needed. It now clears
  that idle state, the largest first, and goes on; one that cannot fit
  alone fails with a 500 and the service keeps serving.
- Cached plans and graphs, which live outside the catalog, are now
  bounded: each runner caps them (chunk and drafter plans shared by its
  request slots, wave plans, and every graph, its drafter's and waves'
  included, by count and by bytes; least recently used dropped) and reports
  the most they can hold, and a swap drops the outgoing model's. The
  runtime's memory guard sets apart the largest model's bound (DeepSeek
  with DSpark: 0.63 GiB), so many distinct request shapes can no longer eat
  its uncounted margin and push the host into swap. A model returning from
  a swap plans and captures its graphs again.
- Routed quantized products allocate scratch for partial expert tiles'
  dummy columns, including raw FP4, so bounded workspace covers every read.
- Qwen3.8's sparse attention rejects unequal key/value cache row strides
  before launch, preventing incorrect reads of padded caches.
- The long-context correctness judge refuses incomplete or non-finite
  captures instead of passing shortened comparisons; a repeatability
  mismatch now also returns a failing exit status.
- Qwen3.8 Flash Next at its configured maximum context (262,144 tokens)
  was refused at startup, and past about 147K tokens its prefill failed:
  its prefill chunk is now capped where a chunk's [context, rows] tensors
  would pass 2^31 bytes (2,040 rows at 262,144; the startup log says so),
  and a 258,633-token prompt prefills and answers.
- DeepSeek V4 Flash's prefill failed past about 52K tokens of context (an
  operation check stricter than the kernel it guards); it no longer does
  (checked with a 64K prompt, and 128K in the resident harness).
