<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GGML and llama.cpp

- **Repository:** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
  GGML is developed there and synced with
  [ggml-org/ggml](https://github.com/ggml-org/ggml). MIT.
- **jitLLM's pin:** tag `b10964`, commit
  `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` (2026-09-14)
  ([sources.lock.json](../../third_party/sources.lock.json)). jitLLM compiles
  only GGML's tensor code and the CUDA kernels it launches, with its own
  dispatch (D-053, D-077). Its changes are in
  [third_party/patches/ggml/](../../third_party/patches/ggml/).
- **Last upstream check:** 2026-09-29, master `8019dc563` (tag `b11254`),
  290 commits and 15 days ahead of the pin.
- **Before sending anything, read the contribution rules** in
  [README.md](README.md#contributing-to-llamacpp). In short: the owner
  writes issue and PR text himself (no AI-written descriptions), AI-written
  code is disclosed, and a new contributor may have one PR open at a time.
  So the entries below give facts, a repro and a diff pointer, not prose.

Suggested order for PRs, one at a time: the mask pre-pass bound (RE-036),
then ssm_conv's load bound (RE-032), then the null-buffer guard, then the
sinks bound (RE-030).

## Native Gemma assistant graph port

- **Status:** native component foundation; no upstream kernel change or
  assistant serving/speculation qualification.
- **jitLLM port:** `src/kernels/ggml/gemma4_assistant_graph.cc` ports the pinned
  `src/models/gemma4-assistant.cpp` pre/post-projection, Q-only blocks,
  sandwich normalization, GeGLU-tanh and readonly shared-cache attention into
  native independent-slot descriptors. Its MIT AND Apache-2.0 header retains
  the GGML authors' copyright; the source-lock scope and existing packaged MIT
  attribution cover the port. No backend runtime, allocator or dispatcher is
  copied. Prepared archive bytes, source pin and CUDA flags are unchanged.
- **Evidence:** [bounded component controls](../experiments/gemma-assistant-execution/README.md)
  compare original-image C1 complete heads/projections at identical frozen
  stage-zero operands, with source/output own repeats frozen before oracle
  exposure. Native-target chain and optimized-batch qualification remain open.

## Standalone Gemma routing and scaled reduction

- **Status:** native availability only; no upstream numerical change or new
  Gemma graph selection.
- **jitLLM change:** build patch 0002 adds the pinned original `topk-moe.cu`
  and `moe-weighted-reduction.cu` to the existing CUDA target, preserving
  its fast-math flags and MIT notices. The source lock records their compiled
  scope and the newly prepared tree; targeted source contracts authenticate
  the original four implementation/header files and actual compile inventory.
- **Native contract:** checked caller-funded F32 routing for 128 experts/top
  eight retains the existing full ARGSORT-root pitch and the exact 2^-14
  normalization clamp. Only the first eight IDs are initialized; full-sort
  consumers must use the primitive chain. The three-input scaled ordered
  reducer is bounded to Gemma's width 2,816 and eight selected experts.
- **Evidence:** the [primitive controls](../experiments/gemma-moe-primitives/README.md)
  compare literal full outputs with the exported original pinned-image CUDA
  operators, without rebuilding reference floating kernels. This does not
  establish whole-model quality, speed, or production policy.

## MMVQ reads rounded output rows (RE-045)

- **Status:** pinned limitation, guarded by jitLLM's launch plan; current
  upstream master has not been checked for this finding.
- **Found:** 2026-10-04 during independent review of legacy quant primitives.
- **Problem:** pinned `mmvq.cu` dots every selected rows-per-block row even
  in the final partial block. Q5_1 K704/N129 on GB10 selects four rows:
  the final block reads three additional 528-byte rows, while the canonical
  512-value tail funds only 384 bytes. No faulting launch was needed to
  establish the source bounds violation.
- **jitLLM's workaround:** ordinary MMVQ derives the pinned host-table/type/
  small-K geometry and refuses N not divisible by the selected row block,
  for all compiled types, before submission. Its own row-invariant kernel
  receives actual N and guards reads, prefetch and writes; partial rows and
  output-stride gaps are tested with canonical padding only.
- **Proposed action:** before upstreaming, verify current master, then pass
  actual N to the dot loop and bound its row reads. No upstream numerical
  source patch is carried in this slice. See the
  [bounded controls](../experiments/m35-legacy-quants/README.md).

## 32-bit strides: flash attention's mask and `ggml_permute` (RE-037)

The pinned quantized products have a related limit: MMQ's `offset_x`
(`mmq.cuh:1061,1155,1239`) and MMVQ's `kbx_offset`
(`mmvq.cu:697,895`) are signed 32-bit native quant-block indices, even
though their host arguments use wider integers. jitLLM now validates the
combined row/channel/sample block span and the final padded 512-value step,
not just each stride. For example, Q5_1 [512,128,3] with a whole-block expert
pitch of 2^30 blocks passes individual stride checks but expert 2 starts at
2^31, which cannot be indexed. Host-only boundary and planning controls
prove refusal without submitting an unsafe operation. This is a native
operand-check correction; no additional numerical source patch is carried,
and current upstream master has not been checked.

- **Status:** `ggml_permute`: fixed upstream after the pin, in #29227
  (`c21284cdf`, in master `8019dc563`, b11254); it comes with the pin bump.
  Flash attention's `int32_t` `nb31` and `nb32`: unchanged at `8019dc563`
  (`fattn-mma-f16.cuh:1828`); carried, no action (unread while `ne32` is 1).
- **Found:** 2026-09-29, pin `b29c606e2`, `spark-b`, Qwen3.8 at 147K–262K
  tokens.
- **Problem:** `ggml_permute` computes a view's `nb` in `int`, so a
  permuted F32 tensor of 2^31 bytes or more gets a wrapped plane stride
  (seen: 18446744071572553728 for Qwen3.8's expanded QSA scores, [3,584,
  150,528]). `flash_attn_ext_f16`'s launch takes `nb31` and `nb32` as
  `int32_t`; the MMA kernel reads the mask only through `nb33` (int64)
  with `ne32` = 1, so nothing reads a truncated value there today.
  llama.cpp's 512-row micro-batches stay far under both.
- **jitLLM's workaround:** its operation checks refuse such strides, and
  Qwen3.8's chunk bound keeps every F32 [n_kv, rows] tensor under 2^31
  bytes (RE-037); DeepSeek V4's keeps its F16 attention mask under it
  (4,029 rows at 1,048,576 positions).
- **Proposed action:** take #29227 with the pin bump; then the chunk bound
  can follow the flash-attention mask (F16) alone.

## Concat's two kernels have different grid limits (RE-038)

- **Status:** no upstream action; a jitLLM-side check, fixed.
- **Found:** 2026-09-29, pin `b29c606e2`, `spark-b`, DeepSeek V4 past ~52K
  positions.
- **What:** `concat_cuda` (`concat.cu:142-196`) runs `concat_cont`, a
  one-dimensional grid, for operands contiguous in their first three
  dimensions, and the per-row `concat_non_cont` (a `dim3(ne1, ne2, ne3)`
  grid, so at most 65,535 channels) otherwise. Upstream is correct;
  jitLLM's `CheckConcat` applied the per-row kernel's limits to both and
  refused DeepSeek's CSA concatenation past 65,535 cells. The check now
  follows the dispatch.
- **Proposed action:** none.

## Flash attention's mask pre-pass reads past the mask's last row (RE-036)

- **Status:** open.
- **Found:** 2026-09-29, pin `b29c606e2`, `spark` (GB10):
  `jitllm-runtime chat` with 4,096-row prefill chunks on 2,164- and
  2,362-token prompts (one chunk of that many rows each); the RE entry has
  the observation.
- **Problem:** from 1,024 query rows, `launch_fattn` runs
  `flash_attn_mask_to_KV_max<ncols1>` (`ggml/src/ggml-cuda/fattn-common.cuh`,
  about lines 666–706 at the pin). Each block reads whole tiles of `ncols1`
  mask rows (up to 8) with no bound on the query row count. When the row
  count is not a multiple of `ncols1`, the last tile reads up to
  `ncols1 − 1` rows past the mask. llama.cpp never shows it: its default
  micro-batch is 512 rows, below the pre-pass threshold, and a larger batch
  over-reads into its compute buffer. It does not pad the mask.
  Repro: attention with, for example, 1,025 query rows and a mask of exactly
  1,025 rows, run under `compute-sanitizer --tool memcheck`, or with the mask
  ending at an unmapped page.
- **jitLLM's workaround:** the kernel checks refuse such shapes
  (`src/kernels/ggml/fattn_mma.cu`, "the mask pre-pass reads whole column
  tiles past the mask's rows"; `src/kernels/ggml/validate.cc`, "the mask row
  the pre-pass reads"), and the runtime (`src/runtime/prefill.h`) runs a
  chunk of 1,024 rows or more in whole 8-row tiles and its few leftover rows
  as a chunk of their own. The harnesses' `--max-rows` runs are not split
  and are refused instead. Cost: at most one extra chunk of under 8 rows
  per prompt.
- **Upstream master:** unchanged on CUDA at `8019dc563` (2026-09-29).
- **Upstream refs:** Metal fixed the same bug in
  [#29220](https://github.com/ggml-org/llama.cpp/pull/29220) (merged
  2026-09-21). Nothing open for CUDA.
- **Proposed action:** a small PR that passes the query row count to the
  pre-pass and bounds its row index. Cite #29220 as precedent. About an
  hour, plus a test in `test-backend-ops` at 1,025 rows.
- **Links:** RE-036 in [rough-edges.md](../rough-edges.md).

## ssm_conv reads up to 31 floats past its window (RE-032)

- **Status:** open.
- **Found:** 2026-09-28, pin `b29c606e2`, `spark-b`.
- **Problem:** in `ggml/src/ggml-cuda/ssm-conv.cu`, past 32 tokens the
  launcher splits the tokens into 32-token blocks, and
  `ssm_conv_long_token_f32` loads `d_conv − 1 + 32` columns of each channel
  into shared memory whatever the block's real token count (`local_n_t`).
  The last block of the last channel reads up to `(32 − n_t % 32) % 32`
  floats past the window. Only the loads are unbounded: the outputs use
  loaded columns below `local_n_t + d_conv − 1`, so results are right.
  Repro: `ssm_conv` with 40 tokens (d_conv 4, 128 channels) under
  `compute-sanitizer --tool memcheck`: 25 invalid global reads, 13–17 bytes
  past the allocation.
- **jitLLM's workaround:** `CheckSsmConv`
  (`src/kernels/ggml/validate_ext.cc`) refuses more than 32 tokens that are
  not whole 32-token blocks. The unfused Qwen3.8 graph pads such windows to
  whole blocks (`src/kernels/ggml/qwen38_graph.cc`). The default fused graph
  uses `jitllm.gdn.conv`, which does not over-read, so only `--unfused`
  pays for the padding.
- **Upstream master:** unchanged at `8019dc563`.
- **Upstream refs:** none found.
- **Proposed action:** a PR bounding the shared-memory load by the window's
  columns, about a one-line condition. Under an hour.
- **Links:** RE-032 in [rough-edges.md](../rough-edges.md);
  [qwen38-native](../experiments/qwen38-native/README.md).

## Weights without a backend buffer assert in the MMQ and MMVQ padding clear

- **Status:** open (jitLLM carries the guard in its patch).
- **Found:** 2026-09 (M2), pin `b29c606e2`.
- **Problem:** `ggml_cuda_mul_mat_q` (`ggml/src/ggml-cuda/mmq.cu`) and
  `ggml_cuda_mul_mat_vec_q` (`mmvq.cu`, about line 1471 at the pin) call
  `ggml_backend_buffer_get_usage(src0->buffer)` to decide whether to clear
  padding, with no null check. A tensor whose `buffer` is null (valid for
  tensors that point at memory GGML does not own) trips the assert inside
  that call.
- **jitLLM's workaround:** `third_party/patches/ggml/0001-jitllm-adaptations.patch`
  adds `src0->buffer != nullptr &&` under `GGML_JITLLM`. No cost.
- **Upstream master:** still no null check at `8019dc563`.
- **Upstream refs:** none found.
- **Proposed action:** a tiny PR adding `src0->buffer &&` to both
  conditions. It lets jitLLM drop two hunks. Minutes of work, but it spends
  the one open PR slot, so send it after the two above.
- **Links:** the patch above.

## Tensor-core flash attention reads sinks past the last head (RE-030)

- **Status:** open.
- **Found:** 2026-09-28, pin `b29c606e2`, `spark-b`.
- **Problem:** `ggml/src/ggml-cuda/fattn-mma-f16.cuh` groups 8 query heads
  per tile (`ncols2`) and reads `sinks_f[jc % ncols2]` from each group's
  first head (lines 1402 and 1889 at the pin) with no bound. When query heads
  per KV head are not a multiple of 8, the last group reads past the sinks
  tensor. Repro: sinks, 24 query heads over 2 KV heads, D = 256, and a
  24-float sinks tensor ending flush against an unmapped page: an illegal
  address (jitLLM's probe in `tests/unit/ggml_ext_ops_test.cc`). In
  llama.cpp the sinks sit inside a larger weight buffer and feed only
  discarded heads, so nothing shows.
- **jitLLM's workaround:** `CheckFlashAttnMma`
  (`src/kernels/ggml/validate_ext.cc`) refuses sinks unless the ratio is a
  multiple of 8. DeepSeek V4 (64) passes, and Qwen3.8 has no sinks, so there
  is no cost for current models.
- **Upstream master:** unchanged at `8019dc563` (`sinks_f[jc % ncols2]`).
- **Upstream refs:** none found.
- **Proposed action:** a small issue and PR bounding the read by the head
  count. Low priority.
- **Links:** RE-030 in [rough-edges.md](../rough-edges.md).

## Radix top-k breaks ties nondeterministically (RE-031)

- **Status:** carry: both models' default fast plans select with
  jitLLM's own kernels, ties to the lower index (Qwen3.8's at any depth
  since long context's phase 2); their reference (`--exact`) and unfused
  forms still use GGML's top-k.
- **Found:** 2026-09-28, pin `b29c606e2`, `spark-b`; DeepSeek V4 on
  2026-09-29, `spark`.
- **Problem:** for rows over 1,024 columns, the radix select in
  `ggml/src/ggml-cuda/top-k.cu` (lines 170–173 at the pin; jitLLM builds it
  without CUB, as HIP does) compacts elements equal to the threshold with
  `atomicAdd`, so which of several tied values land in the top k depends on
  timing. Qwen3.8's QSA indexer scores blocks of 4 cells, so every score
  appears four times, and many are zero after ReLU. The selection, and so
  the attention, changes run to run past 2,051 cells: in 7 of 12 reruns of
  one 8,192-token prefill, logits differed from position 5,120 on.
  DeepSeek V4's lightning indexer (`ggml_top_k` of 512 over its compressed
  cells, the radix path past 1,024 of them, so past 4,096 positions) hits
  it too: one 8,088-token prefill gave one of two last rows (top logit
  22.639 or 22.914) across 7 runs, and with the radix gather replaced by an
  index-ordered one (one thread a row, scanning the columns in order) 6 of
  6 runs repeated exactly.
- **jitLLM's workaround:** DeepSeek V4's fast plan (the default) scores and
  selects with jitLLM's `jitllm.dsv4.lid_topk`
  (`src/kernels/ggml/dsv4_sparse.cu`), ties to the lower row: its 32K run
  repeats bit for bit (long-context phase 2); `--exact on` does not. For
  Qwen3.8, the default fast graph selects with jitLLM's `jitllm.qsa.topk`
  (`src/kernels/ggml/jitllm_ops.h`, `qsa_sparse.cu`), which keeps the lower cell among equals at any depth
  to its configured 262,144 (a radix select over tiles of 8,192 blocks,
  then over the tiles' candidates), and attends the kept cells alone
  (`jitllm.qsa.attn`): its 64K and 128K runs repeat bit for bit
  ([long-context](../experiments/long-context/README.md#phase-2-qwen38-flash-next-flat-with-depth)).
  Until phase 2 its selection held the scores in 32 KiB of shared memory
  and fell back to GGML's top-k past 8,192 blocks. The `--exact` and
  `--unfused` graphs still select with GGML's top-k and are not
  repeatable past 2,051 cells.
- **Upstream master:** unchanged at `8019dc563`. The CUDA path with CUB uses
  CUB's top-k with `determinism::not_guaranteed`.
- **Upstream refs:** issue
  [#28497](https://github.com/ggml-org/llama.cpp/issues/28497) (Qwen3.8 QSA
  over 2,051 cells). A maintainer says it is within `ggml_top_k`'s
  contract. The reporter proposes top-k over the block scores, then
  expanding. A deterministic radix PR,
  [#28871](https://github.com/ggml-org/llama.cpp/pull/28871), was closed
  unmerged under the AI-text policy.
- **Proposed action:** in jitLLM, a long-context slice extends its own
  selection past 32K cells, selecting over the block scores with ties
  broken by index. TensorFold's tiled select is a model
  ([tensorfold.md](tensorfold.md#upstream-techniques-to-adopt), PR #93).
  DeepSeek V4's is done (above). Upstream, optionally a short comment on
  #28497 by the owner.
- **Links:** RE-031 in [rough-edges.md](../rough-edges.md);
  [qwen38-native](../experiments/qwen38-native/README.md#results-second-pass).

## The graph object trips UBSan on creation (RE-021)

- **Status:** fixed upstream at
  [ggml-org/ggml#1644](https://github.com/ggml-org/ggml/pull/1644), synced
  to llama.cpp in [#29396](https://github.com/ggml-org/llama.cpp/pull/29396)
  (`bced4595b`, 2026-09-24).
- **Found:** 2026-09-27, pin `b29c606e2`, `spark-b`, the `cross-asan` build.
- **Problem:** `ggml_graph_nbytes` advanced a null pointer (`ggml.c:7424`),
  so `ggml_new_graph_custom` ended the process under UBSan.
- **jitLLM's workaround:** jitLLM builds no `ggml_cgraph`. Its fusion gates
  take a node list (`src/kernels/ggml/fusion.h`). No cost now.
- **Proposed action:** none. At the next pin bump, mark RE-021 fixed.
- **Links:** RE-021 in [rough-edges.md](../rough-edges.md).

## Broadcast ops abort on strides above 2^32 elements (RE-011)

- **Status:** won't fix (the 32-bit limit is deliberate).
- **Found:** 2026-09-22, stable-diffusion.cpp `c92d73c` on GGML `8846b79`,
  `spark-b`, a 2048² Qwen-Image VAE decode.
- **Problem:** `GGML_ASSERT(s02 <= UINT32_MAX)` in
  `ggml/src/ggml-cuda/binbcast.cu` (line 270 then).
- **jitLLM's workaround:** none needed: jitLLM's image path uses its own
  kernels (`src/kernels/image/`).
- **Upstream master:** unchanged at `8019dc563`; the asserts are intended
  ([#24706](https://github.com/ggml-org/llama.cpp/issues/24706)).
- **Proposed action:** none.
- **Links:** RE-011 in [rough-edges.md](../rough-edges.md).

## MMVQ's launch depends on the column count, so batched rows differ from single rows (RE-033)

- **Status:** won't fix (by design upstream).
- **Found:** 2026-09-28, pin `b29c606e2`, `spark-b`.
- **Problem:** `calc_nwarps` and `calc_rows_per_block` in `mmvq.cu` depend on
  `ncols_dst`, so a column of a multi-column product is reduced in a
  different order than the same column alone. A speculative verify of k + 1
  rows is not bit-identical to k + 1 decode steps.
- **jitLLM's workaround:** only under `--exact on`, the row-invariant plan
  (`src/kernels/ggml/mmvq_rows.cu`, D-092). Its verify ran at 28.75 and
  29.70 tok/s against 29.85 and 34.05 batched. The default plan batches.
- **Upstream master:** unchanged at `8019dc563`.
- **Upstream refs:** open issues
  [#25618](https://github.com/ggml-org/llama.cpp/issues/25618) and
  [#28111](https://github.com/ggml-org/llama.cpp/issues/28111).
- **Proposed action:** none.
- **Links:** RE-033 in [rough-edges.md](../rough-edges.md);
  [dspark](../experiments/dspark/README.md).

## Extra graph outputs or eval callbacks change the CUDA plan (RE-010, RE-006)

- **Status:** won't fix (fusion and CUDA-graph behaviour, not a kernel bug).
- **Found:** 2026-09-21/22, pin `b29c606e2`, GB10, Gemma 4 26B A4B.
- **Problem:** adding graph outputs, or an eval callback at a fused node,
  changes which fusions qualify and the compute-buffer layout, and with them
  the logits.
- **jitLLM's workaround:** none needed in jitLLM, which owns its plans.
  Reference captures read routes at a fusion boundary
  ([fused-routes](../experiments/fused-routes/README.md)).
- **Proposed action:** none.
- **Links:** RE-006 and RE-010 in [rough-edges.md](../rough-edges.md).

## SWA sequence state saves "successfully" but cannot be reused (RE-004, RE-007)

- **Status:** open (upstream master not checked).
- **Found:** 2026-09-21, pin `b29c606e2`, GB10, Gemma 4 26B A4B, llama.cpp
  server.
- **Problem:** with default (windowed) SWA, a slot save and a fresh-process
  restore both report 627 tokens, yet the next request re-processes all
  639 prompt tokens (RE-004). Serializing one sequence drops cells outside
  the final SWA window (`src/llama-kv-cache.cpp:2080` at the pin), so a
  restored snapshot lacks positions a rolled-back prompt needs: 770 of them
  in the four-turn probe, which changed 58 of 3,072 predictions (RE-007).
  `--swa-full` avoids both, at 1,760 MiB of KV against 460 at 8K context.
- **jitLLM's workaround:** the reference harnesses run full-SWA and check
  the retained window before reuse. jitLLM's own spill does not use this code.
- **Upstream master:** not checked.
- **Proposed action:** check master; if unchanged, possibly an issue by the
  owner asking the restore API to report reusable tokens, not saved ones.
  Low priority: it affects only llama.cpp reference runs.
- **Links:** RE-004 and RE-007 in [rough-edges.md](../rough-edges.md);
  [reference-aba](../experiments/reference-aba/README.md).

## jitLLM's carried adaptations (patches 0001 and 0002)

- **Status:** carry.
- **What:** `third_party/patches/ggml/0001-jitllm-adaptations.patch`, all
  under `GGML_JITLLM`:
  - `ggml.c`: `ggml_print_backtrace` does nothing, so a served process never
    forks a debugger;
  - `common.cuh`: `ggml_cuda_error` is not `[[noreturn]]`, so jitLLM's
    definition returns the error instead of aborting;
  - `common.cuh`: PDL is not read from `GGML_CUDA_PDL`;
  - `mmq.cu`: the type switch names only the compiled MMQ instances;
  - the null-buffer guard (its own entry above) and building without CUB
    (below).
  `0002-jitllm-build.patch` adds jitLLM's CMake for the files it compiles.
  The 2026-10-04 legacy primitive slice adds the unchanged generated
  Q4_1/Q5_0/Q5_1 MMQ instance units beside the existing Q4_0/IQ4_NL units.
  Native ordinary/paired dispatch and validators name that same closure.
  Row-invariant and joined vector products use the pinned original dot
  helpers; no upstream numerical source or format is changed.
- **Upstream master:** all still needed at `8019dc563`. At master, the
  `common.cuh` and `mmq.cu` switch hunks no longer apply; 0002 applies.
- **Proposed action:** none upstream, apart from the null-buffer guard.
  Rebase the hunks at the pin bump.

## CUB stays off

- **Status:** carry.
- **Why:** upstream's CUB top-k is nondeterministic (it would bring RE-031
  back) and runs once per row. CCCL's segmented top-k request
  [NVIDIA/cccl#6391](https://github.com/NVIDIA/cccl/issues/6391) is still
  open. The pin's CUB argsort also has an in-place-keys corruption bug,
  fixed upstream in [#28389](https://github.com/ggml-org/llama.cpp/pull/28389)
  (`b23701f77`).
- **Proposed action:** none; revisit if CCCL's segmented top-k lands with a
  determinism option.

## Shared sparse gathers across query tiles

- **Status:** narrow backport carried in
  `0003-jitllm-wide-sparse-attention.patch`; no new upstream fix.
- **Source:** [#28770](https://github.com/ggml-org/llama.cpp/pull/28770),
  merge `3cf03257f219afbe7334045ff7c6a06ac68c627d`, and
  [#29298](https://github.com/ggml-org/llama.cpp/pull/29298), merge
  `dc9879cf66aeb5c2f7c38e9578e6a5f38c497865`, MIT. The compact-mask
  implementation is in `src/kernels/ggml/fattn_mma.cu` with GGML credit.
- **Adaptation:** one bounded ascending union and live count per query
  tile, with each query's original mask still applied to gathered cells.
  The pinned MMA configuration, swizzle and KV precision remain. Enable
  its existing D512 eight-query case as well as upstream's D256 case;
  jitLLM's explicit sparse marker permits smaller caches. Partial query
  tiles are bounded. The header backport is used by jitLLM's own
  launchers; GGML's original `fattn.cu` is not compiled.
- **Evidence:** [ds4 study](../experiments/ds4-study/README.md): native
  8K prefill 16.375 to 13.204 seconds, own forced repeats exact, 32K
  perplexity -0.215%. Random overlapping and disjoint lists, partial
  tiles, empty sink rows, finite mask biases and scratch/domain bounds
  match the FP64 reference. The reduction change moves some logits;
  the existing oracle near-tie bound remains fixed.
- **Model limitation and selection:** the [frontier follow-up](../experiments/dsv4-frontier-head/README.md#head-arithmetic-and-quality)
  reproduces a 32K original-target disagreement at forced row 249 with
  production's 2,048-row compact scheduling floor. The oracle's margin
  is 2.616249, above the unchanged 0.947 bound; native's own lead is
  −0.585857. An earlier narrow control changes that lead to +0.096939,
  but a narrow 128K experiment has another outside-bound case. Those
  initial narrow runs used compact scheduling on partial chunks, so
  those are retained as experiments, not a qualified global rollback.
  Historical short/PPL evidence and FP64 operation tests do not close
  this checkpoint gate.
  A matched compact-off 32K control repeats every logit bit exactly,
  ruling out compact scheduling for that fixture. An external control
  keeps CSA/window sharing but returns count-based HCA to the original
  path: 491 equal rows plus 21 within-bound near ties, zero violations,
  native step-249 lead +0.184032. Its completed implementation counts
  confirm the split. Prefill takes 9.26% longer at 32K and 12.23% longer
  at 128K than matched all-wide controls. Its
  corrected-floor 128K arm also passes (500 equal rows, 12 within-bound
  near ties). Fresh all-head/frontier repeats at both depths are exact;
  matched 128K PPL is 1.926517 versus 1.9298. The planner therefore
  selects registered ordinary MMA for the existing count-based HCA
  mask semantic, while opted-in CSA/window sharing remains. Production
  state/injection, forced rejection and swap continuations are exact.
  Final sampled and maximum-context runtime gates remain pending.
- **Scope:** DeepSeek's fast sparse attention and the D256 reference
  capability. Selection defaults off: the old D512 one-query and D256
  dense choices remain the primitive/reference defaults. Only measured
  DeepSeek fast CSA/window planners enable wide unions; HCA retains
  ordinary shape selection. Fully disjoint D512 lists
  regress in the cross-shape control. Qwen3.8's default `jitllm.qsa.attn`
  is independent.
- **Unmeasured arithmetic candidate:** the pinned NVIDIA D512 MMA
  variants use F32 score accumulators and F16 weighted-value
  accumulators (`T_C_VKQ = half2`). ds4's four-query token-tile kernel
  uses F32 MMA accumulators for both products and rescales in F32.
  That source difference does not prove the cause of the model failure
  above. A real-input replay and the unchanged long-window model gates
  must precede a change to the qualified HCA selection; no new patch
  or upstream defect is claimed.
- **Proposed action:** none upstream; take the full backport with the
  planned pin bump after re-auditing the remaining launch arithmetic.

## Compact routed MMQ tiles

- **Status:** carried in `0004-jitllm-compact-expert-tiles.patch`;
  enabled only by DeepSeek's fast prefill planner from 2,048 rows.
- **Technique:** ds4's expert-major scheduling motivated a bounded
  device-built list of `(expert, token tile)` entries over GGML's existing
  routed-row bounds. The same full-K MMQ inner product reads the original
  raw block layout at each tensor's row/expert stride. Maps and
  type-specific Q8 quantization remain unchanged. No SoA replica or
  early route weighting is introduced.
- **Dispatch:** explicit `compact_experts`, default off, with separate
  single/paired registry identities. The compact launch is NVIDIA-only;
  other devices and small shapes retain the ordinary
  launch; FP4 remains on its separate preparation contract. The list
  capacity is `ceil(assignments / J) + experts`, so capture needs no
  host count readback and workspace is bounded before launch.
- **Evidence:** Spark operation controls cover all eight non-FP4 formats,
  padded expert strides, broadcast/per-slot inputs, partial tiles,
  split-K controls, changing captured routes and exact VMM/workspace
  bounds. Four fresh 8K trials per checkpoint give 1.111× community and
  1.098× original-checkpoint prefill throughput, with all 32 full logit
  rows bit identical to ordinary scheduling and fresh repeats, and stable
  physical memory. Synthetic uniform products improve 1.151–1.428×;
  concentrated routes give larger gains. See [ds4-study](../experiments/ds4-study/README.md).
  At 256/512 rows Q5_K and Q8_0 regress materially, so the production
  planner retains ordinary scheduling there; explicit experimental
  calls remain available. At 2,048 rows all eight gate formats are
  neutral or faster in the same synthetic protocol.
- **Proposed action:** offer the bounded launch as an opt-in for routed
  quantized prefill; retain stream-K/fallback for other shapes. The
  earlier study attempt had an inactive launch hook and does not count
  as evidence.

## Scoped IQ2 compact-pair occupancy two

- **Patch:** `0005-jitllm-iq2-compact-pair-occ2.patch` adds a separate
  default-false MMQ template specialization, leaving every configuration
  row unchanged. The specialized IQ2_XXS/J64/nonfallback/compact kernel
  requests two CTAs per SM. Its single bridge is defined in the existing
  O3 IQ2 instance unit; no duplicate defining translation unit is added.
  (The same specialization for IQ2_XS measured slower than its ordinary
  J128 compact product on GB10, 29.8 vs 23.6 ms for a 4,096-token pair,
  so it is not offered.)
- **Dispatch:** only the native paired-compact path on GB10, with weights
  `[4096,2048,256,1]`, broadcast input `[4096,1,T,1]`, six routes per
  token and T of 256 to 4,096 tokens (a full prefill chunk, or a prompt's
  last, partial one; 256 is the compact expert list's floor). Input and
  both outputs must be packed;
  IDs keep their original validated token stride (including top-k views).
  Existing weight-stride and disjointness validation remains mandatory.
  The original map/scatter preparation and J128 dummy guards remain;
  sequential J64 worklists require 640 rather than 448 entries at 4,096
  tokens (+1,536 bytes; fewer at fewer tokens). Single products, ordinary
  pairs, other formats/geometries and other devices retain their existing
  launches.
- **Evidence:** one captured real-input J128/J64-occ2/J128 comparison
  improves resident paired-product rate 28.05%, with 0.51% bookend movement
  and all six full gate/up outputs byte-exact. One paid native 8K
  comparison improves whole-prefill rate 4.15%, with 0.27% bookend movement
  and all six full vocabulary heads byte-exact. These are separate
  measurements, not additive gains. Actual PTX requests two CTAs; measured
  scheduling occupancy is not claimed. See
  [the occupancy-two screen](../experiments/ds4-iq2-occ2/README.md).
- **Proposed action:** offer this as a shape-scoped routed-prefill variant,
  independently of changing the generic IQ2 configuration. The isolated
  production specialization still needs its selected-code checks; the
  screens do not certify other shapes or models.

## Routed MMQ dummy-column scratch padding

- **Status:** correction carried in patch 0004 and mirrored in jitLLM's
  scratch planner and paired preparation; target controls pass.
- **Finding:** the routed host path asks `get_J_max` about the activation
  slot axis, commonly 1 or 6. That rounds to zero, while the selected
  token tile can have 128 columns. The full Q8/FP4 tile load and the
  output-ID shared load include dummy columns after the final sorted
  assignment. An allocator's overprovisioning or a later stream-K fixup
  allocation can hide the read; jitLLM's workspace has a declared bound.
- **Correction:** size routed Q8/FP4 and output-ID tails for the maximum
  supported `J <= 128`. NVFP4 scale storage also needs dummy columns
  when stream-K stages a full tile into its fixup buffer. Dense NVFP4
  scale storage receives matching padding. No kernel arithmetic or
  primitive dispatch identity changes; compact eligibility stays non-FP4.
- **Control:** ordinary and compact products, single and paired where
  supported, with a partial final expert tile and no ordinary fixup
  allocation, at both exact VMM boundaries. Raw MXFP4/NVFP4 ordinary
  launches are included. All pass on Spark, and the ordinary model's
  saved full-logit capture remains bit identical. The original read was
  identified statically; no deliberate faulting old-binary run is claimed.
- **Proposed action:** upstream the bounded-allocation correction
  independently of compact scheduling.

## Selected expert IDs are unique within a routed token

- **Status:** documented precondition; no upstream patch proposed.
- **Pin and source:** llama.cpp/GGML `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`,
  `ggml/src/ggml-cuda/mmid.cuh`, `mm_ids_helper`.
- **What:** the helper counts every selected slot, but its per-token
  expert lookup has one slot per expert. Duplicate IDs for the same
  expert within one token can therefore disagree with the assigned-row
  count and leave an output index uninitialized. Repetition across
  different tokens is supported. Model routing selects top-k distinct
  experts; a synthetic routed fixture must obey the same contract.
- **Evidence:** an early external Q2 D2R control inserted duplicate
  experts within one token and the ordinary MMQ control failed with an
  illegal device address before the candidate comparison. The fixture
  was corrected to unique IDs within each token; ordinary, compact and
  D2R products then passed repeats, guards and padded-tail controls.
- **Proposed action:** document this precondition in the helper interface
  and test fixtures. No runtime ID validation or new duplicate-routing
  semantics is claimed by the jitLLM port.

## Flash-attention signed iteration and efficiency bounds

- **Source:** pinned `fattn-common.cuh` launcher and `fattn-mma-f16.cuh`
  stream-k iteration indexing at llama.cpp b10964.
- **Issue:** `ntiles_KV * ntiles_dst` and kernel iteration products are signed
  int before widening. The launcher's efficiency helpers also multiply by 100
  before division. Bounded tensor extents/output alone do not bound these
  products when operand rows or sequences share storage. Combined Q byte
  spans and K/V head byte offsets also need bounds before their int products;
  K/V sequence products already use int64. Vector mask row starts multiply
  row stride by query index in int too; their byte offsets need a bound.
- **jitLLM workaround:** checked D64/D256 vector planning bounds the worst
  candidate tile count at `INT32_MAX/100`, conservatively including trials
  that might never be visited. D128 and shared D256/D512 MMA planning
  bounds output tiles at that efficiency limit and total KV/output iterations
  plus the last unaligned tile advance at `INT32_MAX`, rejecting oversized metadata before submission. Validators separately bound combined Q bytes and
  K/V head offsets and vector mask row starts, preserving int64 K/V
  sequence addressing. KV ceil division must also fund its signed addition
  before dividing, including unpadded D128 extents. Ordinary
  eligible shapes still call unchanged pinned kernels. [RE-046](../rough-edges.md#re-046-ggml-flash-attention-total-iterations-use-signed-integers--2026-10-04-status-worked-around).
- **Additional compiled closure:** Gemma local attention instantiates the
  unchanged F16 D256 vector case and D256/group2 MMA query tiles 4/8/16/32 in
  jitLLM wrapper units. No upstream source patch or pin change; the lock's
  license inventory records the additional wrapper.

## Upstream changes to adopt

Checked 2026-09-29 at master `8019dc563`.

- **Sparse flash attention for DeepSeek V4 prefill**
  ([#29298](https://github.com/ggml-org/llama.cpp/pull/29298)): pp2048 on a
  DGX Spark 1.23× at 64K depth and 1.42× at 128K. jitLLM's DeepSeek fast plan now
  gathers every layer's cells with the pinned kernel's one-row sparse case
  (its own condition in `fattn_mma.cu`, long-context phase 2), flat with
  depth; the ds4-study slice now carries the wide union backport above.
- **Sparse flash attention for Qwen3.8**
  ([#28770](https://github.com/ggml-org/llama.cpp/pull/28770)): 1.08–1.26×
  prefill and 1.03–1.18× decode at 10K–100K. Not needed from upstream
  since long context's phase 2: jitLLM's fast graph gathers the kept cells
  with its own kernel (`jitllm.qsa.attn`, a warp a token's KV head, where
  #28770 takes the union of 8 query rows' cells in GGML's MMA kernel), so
  only the reference form would use it.
- **The D 256/512 MMA retune**
  ([#29152](https://github.com/ggml-org/llama.cpp/pull/29152)): 1.00–1.02×
  on the Spark.
- **Optional 8-bit activations for FP4 weights on Blackwell**
  ([#24364](https://github.com/ggml-org/llama.cpp/pull/24364)): KLD 0.022
  against 0.045. A quality mode, not a default.
- **The `ggml_permute` 32-bit truncation fix**
  ([#29227](https://github.com/ggml-org/llama.cpp/pull/29227)): hit by
  Qwen3.8 past ~147K tokens (RE-037, worked around by a chunk bound).

The speedups in this list are reported upstream; the separate backport
entry above has jitLLM's measurements.

**Cost of bumping the whole pin:** two hunks of 0001 no longer apply
(`common.cuh` and the `mmq.cu` switch); `launch_fattn`'s host arithmetic,
which jitLLM's plans record, needs a re-audit; and the D 512 retune changes
attention numerics, so the DeepSeek llama.cpp baselines must be re-run.

**Recommendation:** keep the narrow sparse prefill backport. Bump the
whole pin at the M3→M4 boundary.
