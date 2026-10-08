<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Experimental prepared-artifact format v0

This is the bring-up encoding and layout ABI chosen for open question 5
(D-056). It implements D-009's prepared artifacts and D-035's import-time
repacking. It is **experimental** under D-018: any change may require
re-import, and no compatibility guarantee exists until a dense model and a
small MoE have passed import, execution, eviction and restoration. The
executable reference is the M0 prototype in
[experiments/artifact-layout](experiments/artifact-layout/README.md). It
built, verified and page-checked real artifacts from the D-051/D-052 fixtures
and Gemma 4, and planned three more MoE models. It is not the native C++
importer (M3.6, D-109) or the verifier (M5); until the importer lands it
writes the artifacts (D-087).

**Next version (M3.6, D-108, D-109).** The native C++ importer replaces the
prototype and the per-checkpoint scripts. The next format version names
each tensor's encoding independently of any kernel library, records its
layout separately, and carries the model spec, the folds applied and the
importer's identity in the manifest; target-specific layouts are prepared
artifacts of their own. Every v0 artifact is re-imported; nothing migrates
in place ([engine-components.md](engine-components.md#weight-and-state-formats)).

## Terms

- **Artifact.** One immutable, content-addressed directory holding one
  model representation.
- **Group.** A dependency group: bytes that page in together. Examples are
  one dense layer's tensors, one expert's private weights in one layer, a
  row table or the output head. A group is **one contiguous file range**
  whose start and stored length are multiples of the profile's file
  alignment (4 KiB). A group never spans shards.
- **Chunk.** A group-relative 2 MiB slice of a group: `[k·2 MiB, min((k+1)·2 MiB,
  stored))`. Chunks are the unit for integrity records, closures and
  independent backing handles. They are **not** a read size: reads are
  contiguous file runs that may cover many chunks and groups.
- **Resource.** A named logical byte range inside a group, with a
  representation descriptor. Expert slices are resources that an expert
  array expands.
- **Profile.** The fixed alignment and container parameters. v0 has one,
  `spark-v0`. The manifest's `layout` object must equal it exactly.

D-035's "2 MiB aligned payload extent" now lives only in memory: when backing
uses independent 2 MiB handles, each chunk gets its own handle. On disk,
groups start at 4 KiB, so no file space pays for 2 MiB padding (D-056).

## Directory, identity and publication

```text
<installed store>/<artifact-id>/        artifact-id = SHA-256 of manifest.json bytes
  manifest.json                         root: versions, profile, sources, file list
  index.json                            resource index (groups, resources, chunk hashes)
  data/00000.safetensors ...            immutable payload shards
  meta/...                              source metadata: tokenizer, config, template
```

- Import writes into a private `.staging/job-<digest>` directory under the
  store, holding an exclusive `flock` on `job-<digest>.lock` throughout.
  Source identity covers every input file: checkpoint files, the
  `config.json` that decides a safetensors architecture, and every metadata
  file kept in the artifact. A kept file is a verbatim source file published
  under its own name, and the verifier checks that coverage: each
  `meta/<name>` must match the size and digest of the source entry `<name>`,
  and each `meta/<stem>.kv.gguf` needs a source `<stem>.gguf`. The
  digest covers source and metadata content and the converter, so different
  imports never share a directory. A restarted import of the same inputs
  replaces its own leftovers. A startup sweep removes only leftovers whose
  lock is free, and refuses a symlinked `.staging`. Released lock files
  stay behind; they are harmless. The job checks the sources against the
  identities recorded at download (D-054), re-confirms planned ties on the
  bytes being imported, and re-parses the hashed sources, requiring the
  plan's tensor table and metadata. In the same reads that hash each file,
  the identity pass also digests:
  - each exact header range the planner parsed (which must match the digest
    recorded at parse time);
  - each kept key/value range;
  - every byte range to be copied, and every tie candidate.

  Every copy is checked against its digest, and the sources are hashed again
  after writing. So the
  published bytes are exactly those the recorded source identity covers.
  Source parts must agree on every metadata key other than `split.*`.
  It then verifies the complete staged artifact exactly as an installer
  would, flushes every file and the `data/`, `meta/` and job
  directories, publishes with one `rename` to the artifact ID, and flushes
  `.staging` and the store. If the ID already exists, that copy is verified
  and reported as damaged if it fails; it is never silently reused.
  A published name is exactly 64 lowercase hex characters. Staging
  directories are never listed, loaded or archived, so an interrupted
  multi-shard import never appears installed. The unit tests simulate a
  crash after the first shard. In the runtime, a job renames only after
  the runtime grants it, and conflicting publication and removal of one ID
  never overlap
  ([job rules](architecture.md#import-install-and-archive-jobs)).
- Removal renames `<id>` to `.staging/remove-<id>-<generation>` while
  holding `remove-<id>-<generation>.lock`, flushes `.staging` and the
  store, then deletes that directory. A crash therefore leaves either the
  published artifact or a leftover that the startup sweep removes once its
  lock is free.
- The artifact ID is the digest of the exact manifest bytes. Import is
  deterministic: canonical JSON (sorted keys, no insignificant whitespace,
  one trailing newline, lists in a defined order), no timestamps and no host
  paths. Every string is ASCII from a fixed pattern that excludes `"` and
  `\`, so the canonical bytes don't depend on a JSON library's escaping. Lists are ordered: `source` by
  name, `files` by path, `transformations` by their canonical JSON,
  `resources` by (group, offset), `expert_arrays` by (first group, offset),
  and alias roles after the canonical name, sorted.
  Each shard header is fully determined by the index. The same sources,
  converter and options give the same bytes and ID; readers reject anything
  else, so one content has one ID. A Qwen2.5 rebuild reproduced its ID. D-054's archive and
  peer replication key on this ID; the ID travels over the authenticated
  session and the bytes are verified against it.
- The directory holds exactly the files that `manifest.json` lists, and only
  the `data/` and `meta/` subdirectories. Files are regular and singly
  linked (a hard link could be modified through another path); there are no
  symlinks (including the artifact root) and no other directories. Paths match
  `index.json`, `data/NNNNN.safetensors` or `meta/<safe name>`.

## Container: safetensors shards

The data shards are valid safetensors files, so standard tools can list,
hash and read every stored tensor. The pinned upstream reader (0.8.0) reads
every entry of the built artifacts, pads included, byte-for-byte. Shard
rules:

- The 8-byte header length plus the JSON header are padded with trailing
  spaces so that the data section starts at a multiple of 4 KiB. Upstream
  explicitly allows whitespace padding for aligning the data section. The
  header must stay under the upstream 100,000,000-byte limit. The largest
  planned shard header is 0.97 MB (Qwen3.8), with a 4 GiB shard target.
- The header is exactly the one the index implies: compact ASCII JSON with
  `__metadata__` first (`{"format": "jitllm-shard", "format_version": "0"}`),
  then entries in offset order, with a single pad per gap numbered in order,
  followed by spaces up to the smallest 4 KiB-aligned data offset. The
  verifier rebuilds these bytes from the index and compares them.
- Entries are contiguous and cover the data section, as upstream requires.
  Each resource is one entry named by its resource name. Expert slices are
  named `<source tensor>#<expert>`. Gaps, including backend over-read
  ranges, are explicit zero entries named `~pad.<n>` with dtype `U8` and
  shape `[length]`. Resource names match `[A-Za-z0-9_][A-Za-z0-9_.-]*`, so
  they cannot collide with `~pad.` or `__metadata__`.
- Types that safetensors expresses natively (F32, F16, BF16, I8, I16, I32,
  I64, F64) keep their dtype and shape; GGML shapes are reversed into
  row-major order. GGML block-quantized payloads are `U8[bytes]`. **The
  jitLLM index, not the container dtype, is the authority** for
  representation (D-052).

Why safetensors over GGUF (D-056): both are reusable, and both require
contiguous entries. The pinned GGML reader (`gguf.cpp:780`) checks that each
tensor starts at the running, alignment-padded sum. safetensors wins on four
counts:

- Its header is the smallest parser surface: a length plus JSON, and jitLLM
  needs a JSON parser for its APIs anyway.
- Its reference validation is linear. The pinned GGUF reader checks for
  duplicate names in O(n²) (`gguf.cpp:655`), and Qwen3.8 would have 124,227
  entries.
- It explicitly supports aligning the data section.
- EXL3 sources and the Hugging Face ecosystem already use it.

GGUF's typed GGML blocks would give tools nicer listings. They would also
be a second, GGML-only type authority, which D-052 rejects. GGUF remains the
**source-metadata** carrier (below).

## manifest.json

Keys are strict at every level: unknown or missing keys, wrong JSON types
(`false` is not `0`), floats, non-finite numbers, duplicate keys, a BOM or
non-UTF-8 text, and non-canonical formatting are rejected. No valid string
needs a JSON escape, so the native reader refuses any escape outright
(`canonical`). The prototype refuses it too, under the first rule it
reaches: `json` for an invalid escape or a later syntax error, a check
made before canonical form, `canonical`, or the field's rule.

| Key | Content |
| --- | --- |
| `format`, `format_version` | `"jitllm-artifact"`, `0`. Readers accept exactly the versions they implement; anything else is `unsupported-version` → re-import |
| `experimental` | Must be `true` until D-018's gate records a compatibility policy |
| `layout` | `{profile: "spark-v0", profile_version: 0, file_alignment: 4096, chunk_bytes: 2097152, member_alignment: 256, container: "safetensors", byte_order: "little"}`; any difference is `unsupported-profile` → re-import. A node checks provider compatibility with the profile before accepting an artifact (D-054). `member_alignment` 256 is load-bearing: the fast Qwen-Image plan's cuBLASLt products take weights in place and rely on 256-byte operands (`kernels/image/gemm.h`), so the importer must keep guaranteeing it (owner, 2026-09-29) |
| `model` | Architecture, expert count (equals every expert array's count), sorted representation families present (`ggml`, `exl3`, `plain`), which must equal the families the index uses |
| `source` | Every source file's name, size and SHA-256. They must match the identity recorded at download (D-054) |
| `transformations` | Lossless import transformations: `dedupe-identical` (resource, role, and the digest that proved identity) and `expert-slice` (tensor, count). They must describe the index exactly: one per tied role, one per expert array |
| `converter` | Importer name and version; M5 adds its build identity and an options digest |
| `files` | Path, role (`index`, `shard`, `source-metadata`), size and SHA-256 of every other file; exactly one `index.json` |

## index.json

| Key | Content |
| --- | --- |
| `format`, `format_version`, `file_alignment`, `chunk_bytes`, `member_alignment` | Must match the manifest profile |
| `shards[]` | `path`, `data_offset` (4 KiB multiple), `data_bytes`, `header_sha256` over the bytes before the data section |
| `groups[]` | `kind` (`table`, `layer`, `expert`, `global`, `head`), `layer` (integer for `layer`/`expert`, otherwise null), `expert` (integer for `expert`, otherwise null), `shard`, `offset` within the shard data section, `used_bytes` (end of the last readable range), `stored_bytes` = `used_bytes` rounded up to 4 KiB, `first_chunk`. Groups tile each shard's data in order, and chunks are numbered in group order |
| `resources[]` | `name`, `group` (never an expert group), `offset` (256-byte multiple within the group), `bytes` (must equal the size `repr` implies), `readable_bytes` (the backend over-read rule below, exactly), `roles` (canonical name first, then tied aliases; every role bound once), `repr`, optional `access: "rows"` |
| `expert_arrays[]` | One record per sliced 3D expert tensor per layer: `name`, `layer`, `count` (= `model.expert_count`), `first_group`, `group_offset`, `slice_bytes`, `readable_bytes`, `repr` of one 2D slice. All arrays of a layer share the same `count` consecutive expert groups, in expert order and of equal size, at the same offset in each; every expert group is covered, and each (layer, expert) appears once |
| `chunk_sha256[]` | One digest per chunk, over its stored bytes (last chunk ≤ 2 MiB) |

Readable ranges of resources never overlap, and bytes between `bytes` and
`readable_bytes` are zero pad. Representation descriptors:

- **GGML:** `{family: "ggml", type, ne}` with a known type, 1–4 positive
  `ne`, and `ne[0]` a multiple of the block. For an expert slice, `ne` is the
  2D per-expert shape. **Over-read:** pinned GGML CUDA allocates, zeroes and
  may read `row_size(512 − ne[0] mod 512)` bytes past a quantized tensor whose
  `ne[0]` is not a multiple of 512 (`common.cuh:186`, `ggml-cuda.cu:763-771`,
  `ggml-cuda.cu:917-921`). v0 therefore sets `readable_bytes` to include it
  and reserves those bytes as zero padding inside the same group. Each
  expert slice gets its own, so an over-read never reaches another expert's
  chunk. The planned library carries 1.54 MB of this padding in Gemma 4 and
  5.35 MB in Qwen3.8; the other five models need none.
- **EXL3:** `{family: "exl3", role, dtype, shape}`. A trellis (`I16`,
  `[k/16, n/16, 16·K]`) adds `k_bits` (K), `in_features`, `out_features` and
  `codebook`, taken from marker presence, never from the marker's value.
  Rates are per tensor; there is no global bitrate. v0 accepts only what the
  fixtures cover: codebook `mcg` with K ∈ {4, 5, 6, 8}, full-length FP16
  `suh`/`svh` side vectors, and a 4-byte `I32` `mcg` marker, kept so the
  source maps losslessly. Every trellis must have its `suh` (length
  `in_features`), `svh` (length `out_features`) and marker, and no side
  vector or marker may appear without its trellis. Packed-sign `su`/`sv`,
  other codebooks and other rates are rejected by the importer and the
  verifier until their own fixtures exist (exl3-bringup.md). **Over-read:**
  none, so `readable_bytes` = `bytes`. Backend-proof P3 ran every
  projection of both fixtures, on every path upstream takes (packed GEMM
  and GEMV, fused gate/up, reconstruction and its fused form) at its
  forced plans, with each trellis, side vector, bias and activation flush
  against an unmapped VMM granule on either side: no kernel faulted, and
  every output was upstream's
  ([report](experiments/backend-proof-p3/README.md#placements-alignment-and-over-read)).
  A GPU unit test extends the probe, with random weights at every linear
  shape and rate of both fixtures, to the packed variants those plans
  never launched: the GEMV's wide configuration and its two-to-eight-row
  mode, every GEMM and multi-GEMM tile shape each shape takes (tile
  shape 4 takes none of them, so it ran on a synthetic 896 × 1,024) and
  grids below the co-resident limit. Other rates, codebooks, kernels or
  shapes need their own probe.
- **Plain:** `{family: "plain", dtype, shape}`, for non-quantized EXL3-side
  tensors (BF16 embedding, norms). A name ending in an EXL3 part suffix
  (`.trellis`, `.suh`, `.svh`, `.su`, `.sv`, `.mcg`, `.mul1`) must have the
  `exl3` family.
- **Row tables:** a resource with `access: "rows"` must be 2D. Its row
  count and row size come from `repr`, never from the caller, and lookups
  reject out-of-range row IDs.

## Import layout rules (v0 importer policy)

These rules produced the measured layouts. The verifier enforces the
structural rules above: alignment, tiling, bounds, no overlap, sizes
implied by representation, over-read, EXL3 closure and expert-array
structure. Grouping choices are importer policy and can change without an
ABI change. A group's `kind` and `layer` are placement labels: the runtime
binds tensors by role and representation, never by these labels.

1. **Dense layers:** all of a layer's non-expert tensors form one group, in
   source order, 256-byte aligned. Norms, biases, routers, shared experts
   and hash tables pack between the large matrices. No tensor is padded to
   2 MiB.
2. **Routed experts:** a 3D GGML expert tensor (`ffn_{gate,up,gate_up,down}_exps.weight`
   with `ne[2]` = expert count) is sliced along its outer axis. Each slice
   is contiguous in the source. All projections of expert *e* in layer *L*
   form one group, so an expert's closure is **one contiguous 4 KiB-aligned
   file range**. Small expert-axis vectors that kernels index by expert,
   such as Gemma's per-expert `ffn_down_exps.scale`, stay whole in the
   layer group.
3. **Row tables:** token and per-layer embeddings are their own groups with
   `access: "rows"`. A tied embedding that is also the output head goes in
   the head group.
4. **Ties:** candidate head/embedding pairs with equal type, shape and size
   are hashed. Byte-identical ones are stored once, with both roles and a
   `dedupe-identical` transformation. Equal names or shapes alone never
   imply a tie.
5. **Globals:** the final norm, RoPE factors and similar tensors pack into
   the head group, or into a small `global` group when there is no stored
   head.
6. **Shards:** groups are placed in execution order into shards of at most
   4 GiB of data (a larger group gets its own shard). Shards are a transfer
   and filesystem unit, not a paging boundary.
7. **Source metadata:** for a GGUF source, the original header through its
   key/value section is kept with `n_tensors` set to 0. That is a valid,
   zero-tensor GGUF carrying the tokenizer, template and hyperparameters
   byte-for-byte; upstream `gguf-py` read every field of the Qwen2.5 and
   Gemma sources identically (only `tensor_count` differs, by design). The
   verifier checks that its `general.architecture` and expert count agree
   with the manifest. For safetensors sources, the importer keeps the
   metadata files it is given (the prototype's `--meta`; the EXL3 builds
   passed `config.json`, tokenizer files and the quantization config)
   verbatim. `config.json` is also part of the hashed source identity. Native tokenizer/template extraction remains
   M3 (D-051).

### Qwen3.8 Flash Next (ModelOpt NVFP4 and MXFP8)

M3's importer for this checkpoint (`artifact-layout/modelopt_qwen38.py`)
writes the same container, index and groups, but not the source's bytes
verbatim: it repacks them, losslessly, into what the kernels read, and its
converter version names the module's SHA-256. No transformation is
recorded for a repack (the manifest's kinds are the two above); the
converter identity is the record. Resources are named as llama.cpp's
`qwen4exp` tensors.

- **Routed experts:** ModelOpt NVFP4 (codes with element 2i in a byte's low
  nibble, E4M3 scales per 16 elements in a separate tensor, one F32 global
  scale), one expert group per expert, in the CUTLASS layout
  ([below](#executable-views)): four expert arrays of GGML `I8` bytes
  packed from the group's start, `blk.L.ffn_gate_up_exps.codes`
  (`[w/2, 2f]`: gate's f rows then up's, ModelOpt's code bytes verbatim),
  `blk.L.ffn_gate_up_exps.scales` (`[512, atoms]`: their E4M3 scales in
  CUTLASS's swizzled 512-byte atoms), then `blk.L.ffn_down_exps.codes`
  (`[f/2, w]`) and `.scales`. The global scales are F32 `[experts]` vectors
  in the layer group (`…_exps.weight_scale_2`), applied after each product.
  The first Qwen3.8 import (`67617f87…`, still readable) wrote GGML
  `NVFP4` slices instead (`block_nvfp4`: 64 elements, 4 scale bytes, then
  32 code bytes, byte j of a 16-element sub-block holding elements j and
  j + 8), as expert arrays `blk.L.ffn_{gate,up,down}_exps.weight`; the
  runtime binds either (`model/qwen38.h Qwen38Experts`).
- **MXFP8 matrices** (attention, linear attention, the shared expert, the
  indexer's fused q/k projection) stay in the checkpoint's layout as plain
  resources: `X.weight` (F8_E4M3 `[out, in]`) and `X.weight_scale` (U8, E8M0,
  `[out, in/32]`).
- **The n-gram table:** its 128 shards concatenated into one row table,
  `per_layer_token_embd.weight` (plain U8 `[320,001,536, 90]`, `access:
  "rows"`), each row its 80 code bytes then its 10 E4M3 scales, so one row
  is one contiguous read; its global scale is a plain F32 `[1]`. The hash's
  constants are plain I64 resources of the n-gram layer.
- **Linear attention** reorders its value heads from grouped to tiled order
  (llama.cpp's converter's rule), rows or column blocks moving whole with
  their MXFP8 scales.
- **Everything else:** BF16 matrices keep their bytes as GGML `BF16`; norms
  become F32 with the model's (1 + w) folded in (all but linear attention's
  gated norm); A becomes −exp(A_log), and dt_bias and the convolution
  kernels F32.

### Qwen3.8 Flash Next (GGUF)

A GGUF quantization of the same model (unsloth's UD builds,
ISTA-DASLab's GSQ-RCO and others; llama.cpp's `qwen4exp` tensors) needs no
importer of its own: `import_m3.py build` takes it through layout.py's
GGUF path verbatim, as DeepSeek's GGUF. Every tensor keeps its GGML type
and bytes; each layer's `ffn_{gate,up,down}_exps.weight` is sliced into
one expert group per expert (rule 2; the down slices' 640-value rows
reserve GGML's over-read); `per_layer_token_embd.weight` (the 28.8 GB
n-gram table, IQ4_NL `[160, 320,001,536]` in UD-IQ3_XXS, 90 bytes a row)
is a row table (rule 3); and the first shard's header is kept as
`meta/<stem>.kv.gguf` (rule 7). The runtime binds it as Qwen3.8's GGUF
form (`model/qwen38.h` `Qwen38Format::kGguf`), detected by that table
being a GGML representation:

| Role | GGUF tensors (UD-IQ3_XXS's types) | What the binding takes |
| --- | --- | --- |
| Attention, linear attention, shared expert, PLE projections, mixers | `attn_{q,k,v,output}`, `attn_qkv`, `attn_gate`, `ssm_out`, `ffn_{gate,up,down}_shexp`, `ple_{key,value}`, `hc_*_{down,up}` (Q6_K, Q8_0) | Any GGML type, its shape exact; the graph refuses a type this build's products do not take |
| Indexer projection | `indexer.q_proj` `[2560, 512]`, `indexer.k_proj` `[2560, 128]` (BF16): llama.cpp's converter splits the checkpoint's fused one | Two linears (the ModelOpt form's fused `indexer.qk_proj` is one) |
| Router, β/α, HC inject, shared gate | `ffn_gate_inp`, `ssm_{beta,alpha}`, `hc_*_inject`, `ffn_gate_inp_shexp` (F32) | Any type; the shared gate F32 or BF16 |
| Routed experts | `ffn_{gate,up,down}_exps` expert arrays (IQ2_S gate/up, IQ3_S on one layer, IQ4_NL down) | GGML's layout at the slab's uniform stride, no global scales |
| Norms, convolutions, recurrence | `*_norm`, `ssm_conv1d`, `ple_conv1d`, `ssm_dt.bias`, `ssm_a` (F32, (1 + w) folded, A = −exp(A_log), value heads tiled) | F32, exactly as the ModelOpt importer writes them (it follows the same converter) |
| Token table, head | `token_embd`, `output` (Q6_K) | Any type |
| n-gram table | `per_layer_token_embd` (IQ4_NL) | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 or IQ4_NL (`jitllm.qrows.get_rows`), F32, F16 or BF16, or a 256-value block type GGML's `get_rows` takes |
| n-gram hash | **metadata**, not tensors: `qwen4exp.ple.layer_multipliers`, `.head_offsets`, `.head_vocab_sizes` | Read from the kept `.kv.gguf` (`artifact/gguf_metadata.h`, the prototype verifier's `_check_kv_gguf` rules), with every hyperparameter the shapes do not fix checked against the compiled profile |

The kept metadata is the first native reader of a `.kv.gguf`'s values
beyond the tokenizer, so it applies the rules above (magic, version 3,
zero tensors, unique non-empty keys, no arrays of arrays, the
`general.*` and `*.expert_count` types, no trailing bytes) and checks
every length against the bytes left.

## Page-in contract

- **Closure.** A resource needs every chunk its `[offset, offset+readable_bytes)`
  touches. A row lookup needs the chunks containing its row, deduplicated.
  An incomplete closure blocks launch (architecture invariants 1–2).
- **Reads.** Missing chunks of the same shard that are adjacent in the file
  coalesce into one direct read. The read is **vectored**: one iovec per
  chunk, each 4 KiB-aligned, into that chunk's own admitted, protected
  destination (a separate 2 MiB handle or a slab slot). A run breaks at a
  resident chunk (never overwritten to bridge a gap), at a shard boundary,
  at a bounded request size, and at the kernel's iovec limit (`IOV_MAX`,
  1,024 on both hosts). This replaces D-035's rule that disk adjacency alone
  never permits one read into scattered destinations: it may, when each
  destination is separately admitted and protected. The run size is a
  tuning value (64 MiB in the prototype), not ABI. In the Python harness,
  fewer, longer requests loaded Gemma faster. D-034's native io_uring
  already reached about 15 GB/s with four 2 MiB reads in flight, and
  natively coalescing brought no gain: the runtime's reader (BP-P1)
  loaded the FP16 fixture 2–7% slower with 64 MiB spans than with one
  read per chunk, at depth 4 through the zone, in ~30% fewer requests
  ([backend proof](experiments/backend-proof/README.md#bp-p1-coalesced-reads)).
  So the reader reads one chunk per request by default, and coalescing
  is an option; the run size stays a tuning value for M3's swap path.
- **No hashing at page-in.** Integrity is established when a file enters
  the installed store: at import/publish, at install from an archive or a
  peer, and by explicit `verify` (D-054). After that it rests on the local
  store's permissions. A chunk is usable when its read has completed with
  the full length and its destination generation is still current.
  Measured SHA-256 on Spark runs at 2.49 GB/s per Cortex-X925 core and
  2.12 GB/s per Cortex-A725 core, so hashing every read would take about 6
  cores at D-034's ~15 GB/s. Chunk hashes exist so a verify can locate a
  corrupt chunk and so replication can resume.
- **Row reads below a chunk (a runner's path, not the format's).** A plain
  row table may be read by rows instead of whole chunks: M3's Qwen3.8
  runner reads the n-gram table's rows on demand
  ([swap](experiments/fast-swap/swap.md#qwen38-flash-next-on-the-paged-node)).
  The format needs nothing new for it: a row's bytes are its resource's
  offset plus the row index times the row size, the group's chunks are
  consecutive in one shard (the runner refuses a table otherwise), and each
  read is 4 KiB-aligned inside the group's stored range, so it reads only
  bytes the chunks store. What changes is the reader's contract, which
  D-035 asks to be explicit: rows are fetched per chunk of tokens into
  fixed, charged buffers, used by that chunk's job alone and never
  resident as extents, so there is no row residency to validate or evict;
  there is no hashing, as for chunks. The runtime's own reader stays
  chunk-granular (D-056).

## Executable views

Addresses are rebuilt at load and never serialized.

- **Dense GGML / EXL3 / plain:** a resource's view is its backing base plus
  its group offset. That base is contiguous VA over the group's chunks,
  whether they are separate handles mapped side by side or one slab slot.
  A tensor crossing a chunk boundary is contiguous in VA; paging only needs
  both chunks.
- **Routed GGML experts.** Upstream `mul_mat_id` takes one 3D tensor and
  computes expert *i*'s address as `base + i·nb[2]`, with `nb[2]/block_bytes`
  as an integer stride (`mmq.cu:124`, `mmvq.cu:1496`, `mmf.cu:39`,
  `mmvf.cu:689` at the pin). There are two options:
  - **Pointer table (recommended).** jitLLM passes a per-expert base-pointer
    table through a build-time kernel patch (D-053). Each expert group can
    then live in any slot or handle. VA equals the backing actually used.
  - **Uniform stride, stock kernels.** Each expert group is remapped at
    `base + e·S`, where `S` is a multiple of 2 MiB and of every projection's
    block size. With Q4_K (144 B) and Q6_K (210 B) blocks in one expert,
    `S` = 630 MiB. That needs 93.5 GiB–37.0 TiB of VA per model
    (table below). One Spark process could reserve 128 TiB in one range and
    about 255 TiB in total, so this is possible but wasteful, and it only
    works with independently mapped handles.

  The artifact supports both: slices sit at the same group offset in every
  expert group, and expert groups are uniform per layer.

  **M3's resident GGML layout** (every expert resident, copied into device
  memory through the landing zone, D-081): uniform stride over a slab,
  stock kernels. Each layer's expert groups are placed in one device
  allocation at `slab + e·S`, `S` the group's stored bytes rounded up to a
  multiple of every expert projection's block size and 16 bytes; the three
  `mul_mat_id` weights are views at `slab + group_offset` with `nb[2] = S`.
  Because the groups are copied, not mapped, `S` need not be a multiple of
  2 MiB: DeepSeek V4 Flash's `S` is 8,064,224 bytes (IQ2_XS and IQ3_XXS
  blocks, 3,296 bytes over the group) on 41 layers, 10,888,976 and
  9,309,200 on the two MXFP4 layers, 37.9 MB (0.04%) over the groups in
  all. The pointer table would need a kernel patch and gives a resident
  model nothing, so it waits for demand-paged dispatch (M7); every layer's
  routed products over the slab equal the reference layout's bit for bit
  ([dsv4-native](experiments/dsv4-native/README.md)). Qwen3.8 Flash Next's
  NVFP4 experts take the same layout: `S` is 2,768,976 bytes on every
  layer (80 bytes over the group), 1.97 MB in all. Its down projection's
  640-element rows are not whole 512-element steps; GGML's products read
  past a row up to the next step, into the next row or, after the last
  expert's last row, into the readable bytes the slice reserves inside its
  group, which the slab holds; the graph marks those weights as padded
  (`kernels/ggml/validate_ext.h`), and their products over the slab equal
  the packed reference's bit for bit ([qwen38-native](experiments/qwen38-native/README.md)).
- **Routed NVFP4 experts, CUTLASS layout.** CUTLASS's block-scaled grouped
  GEMM reads E2M1 codes with element 2i in the low nibble and its E4M3
  scales in a swizzled layout (128-row by 4-scale atoms of 512 bytes), not
  GGML's `block_nvfp4` (four scales, then codes interleaved by 8). The
  Qwen3.8 importer writes each expert group in that layout
  (`kernels/ggml/moe_layout.h`): gate and up as one block of 2f rows, then
  down, codes then scales, 2,764,800 bytes (675 × 4 KiB, so the group is
  stored with no padding and `S` is 2,764,800), which the runtime copies
  or pages in as they are, with no rewrite at load. ModelOpt's codes are
  already CUTLASS's; the scales are a permutation, so the import is
  lossless, and what it writes equals the harness's former load-time
  conversion of the GGML-layout artifact byte for byte (unit-tested on
  synthetic experts; every logit of the six prompts' 32 steps equal
  between the two, [qwen38-native](experiments/qwen38-native/README.md)).
  The slab serves CUTLASS's grouped GEMM (prefill) and jitLLM's vector
  products over the same layout (decode), not GGML's `mul_mat_id`. The
  arrays are GGML `I8` bytes, so the container and both readers are
  unchanged; the binder checks their shapes and that they are packed
  from each group's start.
- **Row tables:** rows keep their source stride, so a `get_rows` view is
  unchanged. Some rows straddle two chunks (below); their lookups need both.

## Worked examples (measured plans of real files)

`Group bytes` is the sum of `used_bytes`: payload, 256-byte alignment gaps
and reserved over-read. `Disk padding` is the share of stored bytes above
that, from 4 KiB group alignment. `Handle padding` is the share of backing
bytes above it if every chunk gets its own 2 MiB handle, which is also what
2 MiB-aligned groups on disk would have stored. Byte counts are exact.
Sources are the pinned files named in the experiment report.

| Model | Groups | Chunks | Group bytes | Disk padding | Handle padding | Shards | Container entries (pads) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B FP16 (tied, deduped) | 25 | 490 | 988,208,640 | 0.001% | 3.834% | 1 | 315 (25) |
| Qwen2.5-0.5B EXL3 4.0 bpw | 26 | 292 | 588,909,056 | 0.016% | 3.831% | 1 | 992 (194) |
| Qwen2.5-0.5B EXL3 4.5 bpw | 26 | 302 | 611,215,872 | 0.016% | 3.493% | 1 | 992 (194) |
| Gemma 4 26B-A4B UD-Q4_K_M | 3,872 | 9,059 | 16,933,382,416 | 0.037% | 10.868% | 4 | 16,194 (7,916) |
| Ornith 1.5 35B Q4_K_M | 10,539 | 11,316 | 21,702,479,872 | 0.001% | 8.549% | 6 | 32,219 (101) |
| Qwen3.8 Flash Next UD-IQ3_XXS | 24,627 | 41,000 | 81,957,145,030 | 0.083% | 4.682% | 14 | 124,227 (49,419) |
| DeepSeek V4 Flash UD-Q2_K_XL | 11,053 | 48,171 | 96,827,192,840 | 0.000% | 4.152% | 23 | 34,419 (196) |

**Dense layer (Qwen2.5 FP16, layer 0).** One group of 29,830,656 used bytes,
stored in 29,831,168 bytes and 15 chunks. The 3,584-byte `attn_norm` shares
chunk 0 with the start of `ffn_down` (8,716,288 bytes, chunks 0–4). The
biases and `attn_k` share chunk 12 with the ends of `ffn_up` and `ffn_norm`.
Consequences:

- **Leased tensor sharing a chunk.** Leasing `attn_norm` protects all of
  chunk 0, so that chunk cannot be reclaimed even if `ffn_down` would
  otherwise be evictable.
- **Chunk-boundary read.** A tensor crossing a boundary, such as
  `attn_output` over chunks 12–13, needs both chunks.
- **Resident holes between misses.** If chunks 0–4 and 8–12 are missing
  but 5–7 are resident, the reads are two runs. The resident chunks are
  never re-read into live backing.

**Tied weights (Qwen2.5 FP16).** `output.weight` and `token_embd.weight`
are byte-identical (SHA-256 `d74257dc…`). They are stored once as
`token_embd.weight` with roles `[token_embd.weight, output.weight]`, in the
head group after `output_norm`: 272,272,896 bytes in 130 chunks. That saves
272,269,312 bytes, 21.6% of the source's tensor payload. The EXL3 fixtures
are not tied: the embedding is BF16 and the head is a K=8 trellis, so both
are stored. Tying is content identity, never a name rule.

**Routed expert (Ornith, layer 0, expert 7).** One group holds `down#7`
(Q6_K, 860,160 B), `gate#7` (Q4_K, 589,824 B) and `up#7` (Q4_K,
589,824 B). That is 2,039,808 bytes stored exactly (a 4 KiB multiple) in
one chunk, and it is one contiguous direct read. Every layer-0 expert group
has the same offsets.

Expert closure sizes across the library (closure bytes include the
reserved over-read of Gemma's and Qwen3.8's slices):

| Model | Closure bytes × layers | Chunks each | Handle padding |
| --- | --- | ---: | ---: |
| Gemma 4 | 3,717,520 × 29; 4,337,296 × 1 | 2; 3 | 11.4%; 31.1% |
| Ornith | 2,039,808 × 21; 1,769,472 × 20 | 1; 1 | 2.7%; 15.6% |
| Qwen3.8 | 1,971,456 × 47; 2,329,856 × 1 | 1; 2 | 6.0%; 44.5% |
| DeepSeek V4 | 8,060,928 × 41; 9,306,112 × 1; 10,878,976 × 1 | 4; 5; 6 | 3.9%; 11.3%; 13.5% |

Expert-view VA per model:

| Model | Uniform stride, one array | Uniform stride, per projection | Pointer table, 2 MiB handles | Pointer table, slots |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 | 103.5 GiB | 93.5 GiB | 15.2 GiB | 13.4 GiB |
| Ornith | 3,397.5 GiB | 1,561.5 GiB | 20.5 GiB | 18.7 GiB |
| Qwen3.8 | 17,838 GiB | 4,396 GiB | 49.0 GiB | 45.4 GiB |
| DeepSeek V4 | 37,897.5 GiB | 2,624.5 GiB | 87.5 GiB | 83.6 GiB |

**Sparse rows (Qwen3.8 `per_layer_token_embd`).** IQ4_NL, 320,001,536 rows
of 90 bytes (28.8 GB), in 13,733 chunks, 23,301.69 rows per chunk. Rows 0
and 1 need chunk 0, row 23,301 straddles chunks 0 and 1, and row 5,000,000
needs chunk 214. 13,427 rows straddle a boundary. An isolated lookup reads
2 MiB for 90 useful bytes (23,302×). This remains D-035's default until
evidence supports smaller reads. The token embeddings amplify
700.9–1,820.4× per isolated row.

**Final tail.** In these plans no group ends exactly on a chunk boundary,
though the format allows it. Mean last-chunk fill is 1.57–1.94 MB for the
MoE models. The smallest is Gemma's 4,096-byte `global` group. Gemma stores no separate head tensor: its output reuses
`token_embd`, which stays a row-table group, so the final norms get their
own group. On disk the tail is padded only to 4 KiB. With 2 MiB handles,
memory holds the rest of the handle. It is charged but never read.

## Compositions

A model of several components, such as an image pipeline's text encoder,
denoiser and VAE, is one ordinary artifact per component and a
**composition** that names them (D-089). Components keep this format
unchanged; nothing is copied between artifacts, so a component two
compositions share is one artifact, stored and verified once.

```text
<installed store>/<composition-id>/     composition-id = SHA-256 of manifest.json bytes
  manifest.json                         format "jitllm-composition", version 0
  meta/model_index.json ...             the pipeline's metadata, verbatim
```

| Key | Content |
| --- | --- |
| `format`, `format_version`, `experimental` | `"jitllm-composition"`, `0`, `true` |
| `model` | `{architecture}`: the pipeline's, model_index.json's `_class_name` |
| `components` | 1–16 of `{role, artifact, architecture}`, sorted by role: the model_index.json entry (`[a-z][a-z0-9_]{0,63}`), the component artifact's ID and its manifest's `model.architecture` |
| `source` | Every kept file's name, size and SHA-256, as recorded at download |
| `converter` | As an artifact's |
| `files` | Every kept file, `meta/<name>` with role `source-metadata`; model_index.json is required |

The rules are an artifact manifest's where they apply: strict canonical
JSON, the directory named by the manifest digest, regular singly linked
files, lists in canonical order, each kept file a verbatim recorded source
(and each source kept), the 64 MiB and 128 MiB metadata caps. A
composition has no `index.json` and no `data/`. Its kept files are small,
so opening one always checks their digests. Each component is opened as an
artifact by its ID and its architecture compared with the recorded one.
D-054's installer (M5) is to refuse removing a component a composition
names, and installing or replicating a composition without its
components; nothing enforces that yet.

**Qwen-Image-2.1** (imported on `spark`, 2026-09-28,
[qwen-image-native](experiments/qwen-image-native/README.md)): composition
`eca21baa…` binds `text_encoder` (`qwen3_vl`, 38 groups, 17.53 GB in 5
shards: the token table, 36 layers, and a head group holding the vision
tower and `lm_head`, which the text-to-image path never reads),
`transformer` (`QwenImage21Transformer2DModel`, 33 groups: 32 blocks and a
global group, 14.23 GB in 4 shards) and `vae` (`AutoencoderKLQwenImage21`,
F32 as the checkpoint stores it, 6 groups: the decoder's up blocks and a
global group with the encoder, 1.35 GB), and keeps model_index.json, the
scheduler's config and the processor's nine files. The importer's grouping
policy per component (the prototype's layer pattern and row tables) is
`import_m3.py`'s, and the layout rules above are otherwise unchanged.

## Verification and rejection

`verify` treats the artifact as untrusted input. It parses every document
from the same bytes it hashes, and it fails closed with a named rule on any
of:

- a version or profile mismatch (`unsupported-version`, `unsupported-profile`:
  re-import);
- JSON that isn't strict, typed and canonical (`json`, `canonical`,
  `schema`): unknown or missing keys, wrong types, floats, duplicate keys,
  a BOM or non-UTF-8 text, non-ASCII or unpatterned strings, lists out of
  canonical order or duplicated;
- an artifact name that isn't the manifest digest (`identity`);
- a symlink (including the root), a hard-linked or non-regular file, an
  unexpected directory, or a file that is listed but missing or present but
  unlisted (`file-type`, `file-set`, `path`);
- a size that differs from the manifest, checked before any read in both
  deep and shallow mode (`file-size`); a file, header, chunk or
  dedupe-digest mismatch (`hash`, `chunk-hash`); a kept `.kv.gguf` that is
  not a complete zero-tensor GGUF (`meta`);
- groups that don't tile shards, chunk numbering errors, or a resource
  that is misaligned, overlapping or out of its group by its readable
  range (`bounds`, `alignment`, `overlap`);
- a representation that is unknown, unsupported (EXL3 variants above),
  incomplete (EXL3 closure), or implies a different size or over-read than
  recorded (`repr`); any bound name (resource, alias role or expert array)
  with an EXL3 part suffix but not the `exl3` family, or reserved; roles
  bound twice, bound to an expert array's name, or `roles[0]` not the
  name;
- expert groups that are orphaned, duplicated or not uniform, or arrays
  that disagree with each other or with `model.expert_count`
  (`expert-array`);
- a shard header that differs from the one the index implies (`container`),
  or a non-zero pad (`pad`).

Deep verification streams the data in chunk-sized pieces, so its memory
does not grow with group size. Unexpected exceptions become `malformed`,
and I/O errors become `io`, rather than crashing; this is a safety net, and
the typed checks are meant to make that path unreachable. Before anything
is expanded per expert, the verifier caps resources plus expert slices at
2^18 per artifact and each shard header at the upstream 100,000,000 bytes.
Manifests and indexes must be pure ASCII (canonical `dumps` output always
is). Before any JSON is parsed, a linear, copy-free scan bounds nesting depth
(8), bracket balance, arrays/objects (2^19), values (2^21 separators) and
strings (their sum plus one), failing as soon as a cap is exceeded. Its
string matcher cannot backtrack or retry within a string, so escape-heavy
or unterminated strings stay linear; `manifest.json` may not exceed 1 MiB,
`index.json` 64 MiB and kept metadata 64 MiB. The caps are independent and an
artifact must satisfy all of them. Qwen3.8, the largest planned model, uses
about 75,000 entries, 29,000 containers and 460,000 values in a 6.33 MB
index. Kept metadata is checked before the index is parsed; for kept GGUF
keys the type is checked before any value is read, and
`general.architecture` may not exceed 200 bytes. All kept metadata together
may not exceed 128 MiB. Files are opened
non-blocking, so a FIFO swapped in mid-walk cannot stall verification. Raising a cap is a format-version change. Together these keep a small
document from demanding unbounded memory or time. Kept GGUF metadata follows
the pinned `gguf.cpp` rules: no arrays of arrays, non-empty unique keys, and
`general.alignment` a u32 power of two, `general.architecture` a string,
every `*.expert_count` a u32, and every array element type valid even when
the array is empty. In every kept `.kv.gguf` file, an architecture must equal
the manifest's, and any `*.expert_count` must be the manifest architecture's
key with exactly the manifest's count. The native artifact reader (`src/artifact/artifact.h`)
does not parse kept `.kv.gguf` files when it opens an artifact: it checks
only their caps and source coverage, so it accepts one these rules
refuse. Their readers apply the rules when they read the contents:
`artifact/gguf_metadata.h` (a GGUF Qwen3.8 artifact's hash and
hyperparameters) applies them all but the manifest comparisons, which
its caller makes for the keys it reads (the architecture `qwen4exp` and
`qwen4exp.expert_count` the profile's; another architecture's
`*.expert_count` is only type-checked); the tokenizer's reader
(`tokenizer/gguf.h`) applies its own subset. Each resource may carry at most 8
alias roles, and aliases count toward the entry cap. Unkept arrays are skipped without being built.
Single links are checked for every file in both modes. The prototype's 104 unit
tests cover each rule with a negative case: the classes found by ten
adversarial rounds (the last clean) and two reviews, plus determinism, the interrupted
import, staging locks and sweep, source and tie re-checks, over-read
reservation, EXL3 variants and closure, closure and row bounds, and
coalescing limits. The oracle is exact because Python integers do not
overflow; the C++ verifier must check every product and sum for overflow
(for example, a shape of `[2^32, 2^32+1]` wraps in unchecked 64-bit
arithmetic). The M5 C++ verifier (the standalone verification tool in
features.md) must match these accept/reject decisions. The importer (native
from M3.6, D-109) must also apply D-009's input validation to the source. Page-in helpers reject
out-of-range groups, chunks, byte ranges and row IDs: token IDs are
untrusted input.

**Re-import.** An experimental artifact is re-imported, never migrated in
place, when:

- its format or profile is unsupported;
- a provider rejects its profile;
- the importer declares its converter version incompatible;
- verification fails.

Re-import creates a new ID. D-054's installer quiesces users of the old
one and removes it. The source must remain available (D-018).

## Deliberately open

- **Backing strategy (M2, D-035 comparison).** Independent 2 MiB handles
  avoid external fragmentation but pay the handle padding above in memory.
  Slab slots pay none, but a swap must find a hole for the incoming group,
  for example 3.72 MB for a Gemma expert or 8.06 MB for DeepSeek. Options:
  - evict a contiguous run of cold residents regardless of their individual
    value;
  - size classes, or a hybrid;
  - **activity-sorted compaction (owner proposal):** track expert activity
    and relocate hot groups together so cold ones can be shed as contiguous
    runs.

  Compaction is a catalog move: copy into admitted backing, wait for the
  copy to complete, publish the new location and generation at a boundary
  where no lease holds the group, and free the old slot after its last
  consumer retires. With pointer-table dispatch no VA remap is needed. The
  copy uses memory bandwidth instead of storage, but on unified memory it
  competes with decode, which is itself bandwidth-bound, so it is scheduled
  and measured, not assumed free. The file format serves all of these. The
  comparison should use the closure size distribution above and a
  cross-model swap trace.
- **Expert dispatch:** pointer-table patch versus uniform-stride remapping.
  The initial binding is chosen per format in M3, where resident repacked
  expert groups need views that GGML's `mul_mat_id` and the NVFP4 grouped
  GEMM accept, proven bit-identical to the reference layout on one layer.
  Compaction and demand-paged dispatch stay with the M7 GGML proof. The
  dense EXL3 proof is M2.
- **Companion artifacts** (D-068). A pipeline of text encoder, denoiser
  and decoder is settled as a composition ([above](#compositions), D-089).
  A speculative drafter that uses its target's embedding table is to be a
  composition of its own artifact and its target's, binding the shared
  table from the target's artifact. DeepSeek V4's DSpark import settled
  the binding: the drafter's own artifact holds no copy of the tables, and
  at load it binds the target artifact's `token_embd.weight` and
  `output.weight` by name (`model/dspark.h`); the composition document for
  such a pair waits for a version without model_index.json (D-089).
  Stored MTP layers are ordinary tensors in their checkpoints. Qwen3.8's is
  imported the same way as DSpark: as its own drafter artifact
  (architecture `qwen4exp-mtp`, `modelopt_qwen38.py` `plan_mtp`), binding
  the target's `token_embd.weight` and `output.weight`, so the target's
  artifact is neither re-imported nor changed
  ([qwen38-mtp](experiments/qwen38-mtp/README.md#import)).
  The experimental importer accepts a selected draft vocabulary with
  `--draft-vocab-ids FILE`: ascending original decimal token IDs, one per
  line. It packs the corresponding BF16 target-head rows as
  `draft_output.weight` (dimensions `[hidden, selected]`) and an I32
  `draft_output.ids` map (dimensions `[1, selected]`) in the drafter.
  Both tensors must be present together; binding checks their shapes,
  and load checks map bounds and strict ordering before unchecked device
  indexing. The list is a hashed source with a metadata copy; the command
  requires its size and SHA-256 in the supplied pins file. Without them the drafter uses the
  target head's first 65,536 rows. These are ordinary v0 tensors, not a
  container-schema change; list selection and its license are the
  importer's responsibility.
- **Model-parallel sharding** (TP/EP partitioning, one artifact per rank or
  sliced at load) is deferred with a deadline of M4 entry, where each node
  holds its own shard on disk. It depends on M4's sharding design, and v0
  artifacts are whole-model. File shards are not model shards.
- **Backing reuse beyond a chunk.** A 2 MiB handle holding a chunk
  shorter than 2 MiB has bytes that reloads never write. They may host
  other data only under the general suballocation rules (architecture.md):
  union protection, content generation and lifetime, never as an
  unprotected pool. Small state blocks never inherit the 2 MiB chunk size.
- **Page-in policy tuning:** max run size, in-flight depth and 128 KiB
  device splits (`max_sectors_kb`) are measured with native asynchronous I/O
  in M2 and M3.
- **Out of scope:** mutable spill (D-055/M6), alternative layouts per resource
  (deferred to M9), row-granular reads in the runtime's reader (a runner
  reads Qwen3.8's n-gram rows so, [above](#page-in-contract), with no
  format change), packed-sign EXL3 derivation, a
  binary index. The largest planned index is Qwen3.8's at 6.33 MB, which
  the prototype's strict Python parser loads in 0.048–0.050 s (3 runs). The
  C++ parser cost is not measured.
