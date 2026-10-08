<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ds4

## Original sparse attention and indexer comparison stages

- **Status:** retired with the complete-plan reference on 2026-10-04; commit
  `ec8af04` is the last that holds them. They were qualified temporary
  benchmark primitives; no production selection, complete-model parity or
  model quality/speed result followed.
- **Source:** the same authenticated `ds4_cuda.cu` pin, with 34 complete
  indexer and 68 complete attention definitions and retained-region
  provenance beside the native MIT cores. Original numerical bodies and
  default dispatch ordering remain intact. Packed readers receive explicit
  decode-table/diagnostic operands; MXF4 is guarded to the qualified target.
- **Native ownership:** borrowed checked operands and paid producer
  chains: query preparation, causal selection, union/bitmap or deep sort,
  chronological raw/packed mirrors, split combines and sink normalization.
  Llmpalooza supplies its stream, workspace and completion; no original model
  runtime, allocator, handles or global sidecar registry is imported.
- **Evidence:** Spark A's final locked check passes 1,168 tests, including
  248 GPU tests, ten SDK format checks, four host-unit tidies, boundaries
  and REUSE/header checks (1,079 headers). Full original source/region and
  102-function correspondence, original CUDA flags and `sm_121a` SASS are
  verified; independent whole reviews are clean. The complete configured
  model comparison remains pending. Details and receipt pins are in the
  [complete-plan report](../experiments/ds4-complete-plan/README.md).
- **Proposed upstream action:** expose standalone score/select and complete
  attention producer chains with explicit visibility, packed layout,
  workspace and completion contracts. No submission is claimed.

## Isolated original dense/vector products and headers

- **Status:** retired with the complete-plan reference on 2026-10-04; commit
  `ec8af04` is the last that holds them. The DeepSeek output-A stage
  keeps one: the fused own output-A core (`dsv4_ds4_product_original.cuh`,
  verbatim, behind `dsv4_ds4_product_raw.cu`). The thirteen original headers
  stay in the prepared tree (patch 0003), but no llmpalooza target includes them.
- **Source:** the same original source pin. Four private numerical
  derivatives retain complete function spans (including embedding, conversion
  and fused wide Q/KV normalization); the companion source record lists each
  span and the thirteen unchanged original headers. The kept/patched tree
  changes while the authenticated archive does not. A private namespace
  isolates this original closure; production D2R continues to include the
  locked GGML numerical headers explicitly.
- **Native ownership:** borrowed checked operands, explicit F16/F32 input
  forms, D4/Q8_1 producer identity, original launch tiers, charged scratch
  and ragged physical padding. Llmpalooza supplies the stream, cuBLAS handle,
  workspace and completion. No original allocator, pool, global sidecar
  registry or model runtime is linked.
- **Evidence:** Spark A's final locked check passes 1,135 tests, including
  234 GPU tests, SDK format/tidy, boundaries and REUSE/header checks. Complete
  numerical spans and all thirteen original headers match the authenticated
  sources. Independent source review is clean; full model assembly remains
  pending.
- **Proposed upstream action:** expose independently callable products with
  explicit producer and physical layout contracts. No submission is claimed.

## Literal compressor and aligned weight-repack primitives

- **Status:** retired with the complete-plan reference on 2026-10-04; commit
  `ec8af04` is the last that holds them. No upstream patch or
  production selection change followed.
- **What:** nine complete compressor/RoPE functions and three original
  IQ2_XXS/Q2_K/Q8 repack kernels, with authenticated source pins and
  verbatim-function provenance beside their native cores.
- **Contract:** zero-prefix, aligned and ragged compressor paths retain
  original finite boot sentinel, later infinity resets, mandatory QAT and
  ratio-four last-four-token refresh. The caller supplies the original small
  projection products for that refresh. Repack uses bounded raw chunks and
  whole Q2 row pairs, preserving every scale/code bit; the caller owns files,
  allocations and completion. This does not supply a full prepared model.
- **Evidence:** independent whole reviews and Spark B's final locked check
  pass 1,110 tests, including 223 GPU tests, SDK format/tidy, boundaries and
  REUSE/header checks. All twelve original functions match the pinned source;
  analytical pooling, frontier, byte-permutation/inverse, padding and graph
  controls pass. No end-to-end equality or performance result is claimed.
- **Proposed upstream action:** none; the adapters expose llmpalooza-owned
  buffers and streams. Prepared archive/tree identities are unchanged.

## Literal wide-prefill normalization and hyper-connections

- **Status:** retired with the complete-plan reference on 2026-10-04; commit
  `ec8af04` is the last that holds them. No upstream patch.
- **What:** twelve full numerical functions from the study pin are copied
  under MIT into `src/kernels/ggml/dsv4_ds4_hc_core.cuh`, with source ranges
  in its companion provenance. They cover plain and weighted RMS with F16
  and original compact Q8 sidecars, the twenty-iteration HC4 Sinkhorn split,
  weighted sums, plain expansion, fused expansion/next RMS and head weights.
- **Native contract:** checked borrowed views, explicit row counts, streams
  and output sidecars; no upstream allocation, pointer registry or dispatch.
  The original six-slot guarded ascending MoE sum can feed fused expansion.
  Original wide-row eligibility and fused contraction are preserved;
  fused expansion is not claimed equal to a separate expansion and RMS.
- **Evidence:** independent whole-source/provenance review completed;
  Spark A's final combined locked cache/HC build passes 1,059 tests, including
  211 GPU tests, SDK format/tidy, boundaries and REUSE/header checks.
  This supplies operators, not a complete model run or quality/speed result.
- **Proposed upstream action:** none; the borrowed-view adapters serve
  llmpalooza's ownership contract. Prepared source/archive identities are unchanged.

- **Repository:** [Entrpi/ds4](https://github.com/Entrpi/ds4), fork of
  [antirez/ds4](https://github.com/antirez/ds4), MIT.
- **Study pin:** `76d51ef82a81b70b78e51a3a6ea11946286de976` (M3,
  2026-09-29), CUDA 13.4.92, `sm_121`, on `spark`.
- **Evidence:** [same-GGUF study](../experiments/ds4-study/README.md).
  The locked MIT D2R product unit is linked in CUDA builds; the raw Q2_K
  product is available through an explicit benchmark-only opt-in.

## Cold-context measurements require a fresh session

- **Status:** no upstream action; measurement method corrected.
- **What:** `ds4_bench.c` reuses one session as `ctx-start` advances to
  `ctx-max`. Its reported prefill rate is the added frontier tokens,
  rather than a fresh prefill of each reported context. Llmpalooza's first
  ladder used that result as a cold rung and excluded it after inspection.
- **Reproduction:** a rung uses a fresh `ds4-bench` process with
  `--ctx-start N --ctx-max N --ctx-alloc N_PLUS_1024`; study commands,
  model hash, prompt hash and chunk policy are in the report.
- **Proposed action:** none. Distinguish incremental and cold measurements
  in future comparisons.

## DSpark and MTP use different interfaces

- **Status:** no upstream action; invocation corrected.
- **What:** `ds4-bench --mtp FILE` expects MTP head tensors.
  `DeepSeek-V4-Flash-DSpark-support-0731.gguf` is a DSpark block drafter;
  the benchmark refuses it with missing `mtp.0.hc_head_base.weight`.
  `ds4-server --dspark` supplies the block-drafter interface.
- **Proposed action:** use plain decode for the serial same-format
  benchmark. Do not treat the refusal as a DSpark speed result.

## Transferable prefill techniques

- **Status:** study recorded; no upstream patch.
- **What:** isolated profiles show fused IQ2 gate/up products and Q2 down
  products, expert-major tiles, and token-tiled attention. Llmpalooza shares
  sparse attention gathers across up to eight queries using GGML's
  query-tile union, an explicit fast-plan choice that defaults off for
  unknown sparse patterns after disjoint D512 lists regressed. Its paired-MMQ path shares the expert maps and
  quantized input preparation while retaining the two ordinary products.
- **Original layout mismatch:** `cuda/mmq/ds4_mmq_d2r.cu` requires dense
  whole-array SoA `[half scales, pad64, uint2 codes]`; llmpalooza's resident
  experts use raw GGUF blocks at padded per-expert strides. ds4's fused
  epilogue weights the activation before Q8 quantization and the down
  product; llmpalooza weights after down. It also replaces nonfinite values
  with zero. A direct kernel copy would change layout and arithmetic.
  The raw Q2 loader below resolves the weight-layout mismatch while
  retaining post-down weighting and llmpalooza's cache precision. The IQ2
  fused epilogue and its nonfinite-value sanitization are not selected.
- **Earlier Q2 prototype assessment:** GGML's D2S6 activation layout retains
  original F32 subgroup sums for its affine minimum correction. A simple
  fully dequantized F16 product would discard that correction. Matching
  it through per-16-value integer dots and epilogues adds synchronization
  to an existing integer-MMA path; no speed advantage is established, so
  that exact-arithmetic source proposal was not implemented. The later
  port tests ds4's original reconstruction approximation instead.
- **License:** the fork's root LICENSE is MIT. Its D2R product source
  also attributes Marco Palaferri's MIT code from
  `xangel82/DS4-GB10-GX10-DSpark-CUDA` at `910501e`. The source lock keeps
  the complete root notice and this original product attribution.
- **Precision:** default compressed KV and indexer caches use FP8 and
  FP4 respectively. `DS4_CUDA_FP8_KV=0` and `DS4_CUDA_FP4_INDEX=0`
  restore F32 primary storage, retaining FP8/FP4-rounded values.
  Llmpalooza keeps its F16 caches; default ds4 timings are a same-weight
  comparator with the different transforms and storage disclosed.

## Raw Q2_K weight loader under llmpalooza's launch contract

- **Status:** local adaptation in
  `third_party/patches/ds4/0001-llmp-raw-q2-d2r.patch`; the fast
  DeepSeek plan's default down product on GB10 prefill chunks of 64 to
  4,096 rows ([stage mechanisms](../experiments/ds4-prefill-stages/README.md));
  exact plans keep it off.
- **What:** the original Q2_K D2R MMA and scatter read raw 84-byte GGUF
  blocks through an added loader, preserving its paired-row integer MMA
  and F32 accumulation from the stored half coefficients. Weight row and
  expert strides are explicit;
  no persistent SoA or dequantized weight copy is created. Llmpalooza owns
  expert maps, unchanged GGML Q8 input preparation, bounded worklist
  scratch and completion. It does not invoke ds4's runtime or dispatch.
- **Bounds:** the original worklist stores an expert in the high 16 bits
  of a signed integer, then decodes it with `packed >> 16`. The raw
  interface therefore admits at most 32,768 experts; a 32,769-expert
  shape refuses before preparation. The worklist's column tiles and
  total CUDA Y-grid bound are checked separately. Selection is limited to
  the measured GB10 Q2_K 2,048-input, 4,096-output, 256-expert, six-slot
  product on chunks of 64 to 4,096 tokens.
- **Arithmetic:** raw and original SoA D2R outputs are bit-identical on
  the captured real layer-zero input. Compared with native compact MMQ,
  the raw product's NMSE is `2.1331352209781956e-7`; the original D2R
  reconstruction and tensor-core reduction remain a different
  approximation from GGML's half-rounded affine coefficients and
  original-input minimum correction. Quality uses the unchanged model
  bounds; isolated error is not a model-quality result.
- **Compiled scope:** patch
  `third_party/patches/ds4/0004-llmp-raw-q2-only.patch` makes the native
  target define `LLMP_DS4_RAW_Q2_ONLY`. It excludes unselected original
  IQ2 paired/fused, dense Q8 and SoA Q2 entry points and their global
  instantiations. The raw-Q2 launcher, down numerical template, worklist
  and availability/scratch helpers retain their complete original bodies.
  Original standalone source may still enable the guarded entries.
  The literal comparison derivatives that used the original header
  closure were removed on 2026-10-04.
- **License review:** the entire pinned archive was audited before
  narrowing `archive.keep`. The retained root MIT license names the
  ds4, Entrpi and GGML authors; the D2R source retains Marco Palaferri's
  attribution. The complete product source remains retained; the native
  target compiles its raw Q2 entry point and helpers. No upstream model runtime, cache kernel,
  server, shim or original build script is included in this component.
- **Proposed upstream action:** consider a raw-GGUF/explicit-stride
  product entry point, and document the signed expert-index limit and
  unique-experts-per-token routing precondition. The new loader is
  llmpalooza-specific; no upstream submission is claimed.

## Literal token-tile HCA core over native F16 state

- **Status:** native benchmark-only integration checked, default off;
  `third_party/patches/ds4/0002-llmp-hca-tokentile.patch`.
- **Source:** `ds4_cuda.cu` at the same study pin, numerical sections
  12,313–12,577 and 12,844–13,476 copied verbatim. The original four-token,
  G8, M32/R32 core retains RN-F16 Q loading, F32 QK and PV tensor-core
  accumulation, original F32 online softmax and F16 probabilities. Only
  these helpers/core and narrow launch wrappers compile, under the root
  MIT notice; no ds4 runtime or cache transform is incorporated.
- **Native adaptation:** bit-copy the F16 raw ring to a chronological
  mirror, prepare the original dense causal records, and run the core in
  one planned scratch scope. Negative prefix mirror rows are excluded by
  the original `raw_row_min`. The chunk's first position is a launch
  parameter, so the adapter refuses graph capture. The default-off benchmark caches first
  position explicitly; native runtime builders and defaults are unchanged.
  Joined compressed state retains its original F16 bytes. The trusted
  binder tag vouches for canonical zero finite mask entries and causal
  counts; it is not a content check of arbitrary device masks. The launcher
  also refuses nonstandard scale, ALiBi, softcap and incompatible layouts.
- **Evidence:** identical captured community-GGUF layer-three first-4K
  operands on `spark`, 2026-09-30: native 89.030/89.012 ms before/after,
  literal path 7.056 ms including mirror and record preparation. Candidate
  scratch was 6,425,600 bytes, native attention scratch 7,892,992 bytes.
  Candidate versus native NMSE was `9.484399536172616e-8`; on 22 selected
  token/head rows its full-F32-Q FP64-reference NMSE was
  `1.65870549908486e-8` versus native `9.65217482737688e-8`. Four fresh
  same-binary community 8K processes give 17.88% mean prefill throughput
  gain, identical 128 IDs and exact own full-logit repeats. The separate
  actual 2,048-row charged replay takes 3.557 ms versus 46.560/46.553 ms;
  its matched 8K ABBA gains 15.76% with the same ID/repeat controls.
  Late actual 2K chunks take 4.418 ms versus 11.014/11.011 ms at 32K
  and 8.597 ms versus 33.498/33.479 ms with 126,976 IDs at 128K capacity.
  Selected-row strict references and own/captured repeats pass; the larger
  shape has 1,024 compressed cells. These limited results disclose
  numerical differences. Two fresh original-GGUF 32K model repeats fail
  the unchanged near-tie bound at step 249: 2.616249 versus 0.947 nats.
  The mixed 128K eligible selection passes fixed-window PPL at 1.927031
  versus 1.9298 (−0.1435%), without reversing the greedy failure.
  The original core does not eliminate the prior failure; no default
  change or ds4 reference-quality conclusion follows. Full evidence is in the
  [literal HCA report](../experiments/ds4-hca-tokentile/README.md).
- **Proposed upstream action:** expose a narrow standalone token-tile
  entry point with explicit causal first position, raw prefix availability,
  record extents, and workspace ownership. The native ring adaptation is
  llmpalooza-specific; no upstream submission is claimed.

## Explicit cache/QAT numerical stages for the complete comparison

- **Status:** retired with the complete-plan reference on 2026-10-04; commit
  `ec8af04` is the last that holds them. They were never selected by
  production. A cache stage alone does not establish complete-model parity
  or quality.
- **Source:** the same pinned `ds4_cuda.cu`, with original E4M3/power-of-two
  rounding, normalized Hadamard/E2M1 indexer rounding, packed layouts and
  batch raw-ring store. The reviewed MIT derivative is
  `src/kernels/ggml/dsv4_ds4_cache_core.cuh`; its companion provenance lists
  the original numerical functions and the two pointer-signature adaptations.
  The only FP8 read adaptation replaces a device global with an explicit
  caller-owned, accounted decode table. Original numerical compile options
  remain `-O3 --use_fast_math -lineinfo`.
- **Native ownership:** checked logical spans, no allocation or stream/cache
  counter ownership; llmpalooza's launch scope queues the stages. Raw stores
  retain the original F16 round trip into physically F32 storage. Their host
  position is eager-only to prevent stale graph replay. Packed KV preserves
  the F32 rotary tail; indexer query preparation supports scale-only output.
  The complete host plan, alternate state identity, counters and other
  original stages remain separate work.
- **Proposed upstream action:** expose numerical cache producers/consumers
  with explicit table, ranges and execution-time position contracts. No
  upstream submission is claimed.

## Literal router and routed/shared FFN for the temporary reference

- **Status:** retired with the complete-plan reference on 2026-10-04; commit
  `ec8af04` is the last that holds them. They were checked primitives
  for the temporary complete-prefill benchmark, which matched the whole
  configured pipeline and restored native stages individually before the
  literal reference was retired; no upstream patch.
- **Source:** `dsv4_ds4_moe_core.cuh` and `dsv4_ds4_moe_original.cuh`,
  under the study pin, plus the unchanged prepared original MMQ header
  closure. Companion provenance identifies five original CUDA units:
  `ds4_cuda.cu`, `ds4_mmq_d2r.cu`, `ds4_mmq.cu`, `mmid.cu` and
  `quantize.cu`. The authenticated correspondence proof verifies 115 complete
  numerical functions/templates and ten prepared headers unchanged against
  those pinned sources. Original CUDA numerical options remain
  `-O3 --use_fast_math -lineinfo`.
- **Evidence:** Spark A's final combined locked check passes 1,197 tests,
  including 256 GPU tests and the corrected FFN quantization controls, with
  actual SDK format/tidy, boundaries and REUSE/header checks. The complete
  source proof is SHA-256
  `691bcddc9d74f09eebaee273df05c6265784cde4e9d701d4034a5ae6f49f8e86`.
- **Arithmetic:** authoritative selected-six routing, original expert maps,
  IQ2 G1, weighted/clamped SwiGLU, original D4/canonical Q8_1 and D2S6
  preparation, aligned Q2 down and the guarded six-slot sum. Independent
  controls distinguish direct-F16 Q8_1 scales, reciprocal-F32 D4 scales and
  reciprocal-F16 D2S6 scales; the existing physical bound is unchanged.
- **Native ownership:** checked borrowed extents and explicit six-of256
  source strides, llmpalooza streams/workspace/fixup and current producer
  identities. No original context, allocator, weight server, global sidecar
  registry or model dispatcher is linked. Complete source and transformed
  scratch are charged. D2R's CTA-barrier contract permits complete 128-row
  output tiles and even tails below 16; larger partial tails are refused
  before packing. Original numerical bodies remain untouched.
- **Scope:** the initial full-model scope is two 4,096-row chunks through all
  43 layers and one final full head at 8K. Original serial live-scalar
  routing and the default small materialized/shared-MMQ tier below 1,024
  assignments remain follow-ons; Classic has a distinct original activation
  and is diagnostic. Operator controls do not qualify decode, full-model
  equality, quality, speed or defaults. The measured external original
  and native 8K pipeline match all 129,280 logits byte for byte, with
  native throughput 2.4% lower on a separate Spark. Those fixed-input
  results and the remaining restoration work are in the
  [complete-plan report](../experiments/ds4-complete-plan/README.md).
- **License:** the numerical derivative and prepared original headers retain
  the MIT notices of The ggml authors, The ds4.c authors, Entrpi and Marco
  Palaferri, including the original fused-G1 attribution. Authored adapters,
  tests and provenance are Apache-2.0. The source lock, NOTICE and shipped
  third-party notices retain those obligations; no source offer is owed.
- **Proposed upstream action:** document the low-level D2R tail/barrier
  eligibility and explicit producer-buffer contracts. No submission is
  claimed; native ownership adapters are llmpalooza-specific.

## Cache-off is a storage control, not an unrounded quality oracle

- **Status:** source clarification; no upstream patch proposed.
- **What:** at the study pin, `fp8_kv_quantize_kernel` and
  `indexer_hadamard_fp4_kernel` in `ds4_cuda.cu` always write rounded,
  dequantized values into F32 tensors. Null packed-mirror pointers only
  disable the additional packed writes. The model graph still calls both
  transforms when `DS4_CUDA_FP8_KV=0 DS4_CUDA_FP4_INDEX=0`.
  `--quality` changes several product/attention kernels, not every fused
  path, and does not disable these transforms.
- **Interpretation:** the official DeepSeek inference code also simulates
  these quantization-aware-training transforms. A packed/F32-storage
  comparison alone neither measures their quality effect nor proves a
  long-window gate. The study's earlier higher/full-precision labels are
  corrected; same-GGUF oracle/perplexity qualification remains pending.
- **Proposed action:** document storage and value precision independently
  in future benchmark comparisons; retain the pinned original device
  objects in the external full-window scoring adapter.
