<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# vLLM, as Mia's Qwen3.8 recipe runs it

jitLLM uses vLLM only as a baseline: Qwen3.8 Flash Next NVFP4 in MiaAI's
single-Spark recipe
([baselines](../experiments/fast-swap/baselines.md)). None of jitLLM's code
comes from vLLM.

- **Recipe:** [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
  at `b8439110eec0230facbe4ddf0dffe01b8f769be0`, AGPL-3.0-or-later. At run
  time it mounts its own patched copies of several vLLM files.
- **Engine:** the Docker image `vllm/vllm-openai:qwen38-flash-next`, vLLM
  `0.1.dev20073+g8e685d198`, PyTorch 2.13.0+cu130, CUDA 13.0.1. The image is
  a local build: `8e685d198` is not in upstream vLLM's history, and the image
  has files that upstream `main` lacks (`vllm/models/qwen3_8_flash_next/`,
  `layers/ple_offload_layer.py`).
- **Upstream vLLM:** [vllm-project/vllm](https://github.com/vllm-project/vllm),
  Apache-2.0; the study compared against `main` at `f6877f56` (2026-09-28).
- **The study** is outside this repository, on the workstation at
  `/home/pmeenan/src/mia-load-study/` (2026-09-28). Its `README.md` has the
  method and phase tables. Two self-contained handoffs hold the diffs and
  details, and are what a new effort should start from:
  - `vllm/PATCHES.md`: four vLLM patches and a bug report, diffs against
    `8e685d198` in `vllm/patches/` and patches 1 and 2 rebased onto `main` in
    `vllm/patches/main/`;
  - `mia-recipe/PATCHES.md` and `mia-recipe/ISSUE.md`: the recipe's changes
    and the issue text that was filed.

## Start in ~2.5 min instead of ~11: load with `instanttensor` and persist Triton's cache

- **Status:** reported, as
  [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark#82](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/issues/82)
  by the owner (2026-09-28); open, no maintainer reply on 2026-09-29.
- **Problem:** `./start.sh` takes 11–13 minutes to `/health`, about 9 of
  them copying weights (below).
- **Proposed change (in the issue):** `--load-format instanttensor`, which
  the image's vLLM already supports once the `instanttensor` package is
  installed; keep the PLE-offload worker on the safetensors loader, or
  memory runs out; persist `TRITON_CACHE_DIR`. Measured start to ready: 672 s
  to 151–160 s (211 s on the very first launch); main weights 458 → 40 s.
- **Proposed action:** follow the issue. Nothing more to send unless the
  maintainers ask.
- **jitLLM testing, owner instruction 2026-10-02:** future legacy Mia
  launches use this measured startup patch, preserving the selected model
  runner and inference settings. The tested package is `instanttensor`
  0.2.0; the PLE worker's safetensors override is mandatory. The original
  package and patched recipe are retained at
  `spark:~/.local/share/mia-load-study/{pylib,mia-perf}/`, with the method in
  `/home/pmeenan/src/mia-load-study/issue-runs.sh`. The owner reaffirmed
  this on 2026-10-03: all new runs use the repository
  [pinned launcher](../experiments/fast-swap/mia-launch.py) and its
  [complete payload pin](../experiments/fast-swap/mia-launch.json); the
  [replay instructions](../experiments/fast-swap/baselines.md#default-mia-launcher-for-new-runs-2026-10-03)
  include the mandatory PLE override. Both Sparks retain the package and
  patched recipe. Historical unpatched runs retain their original provenance.

## Lazy safetensors views copy to the GPU at 0.13–0.2 GB/s on the GB10

- **Status:** open (a patch is ready, not sent).
- **Found:** 2026-09-28, the image above, `spark`.
- **Problem:** with `--safetensors-load-strategy lazy`, each tensor is a view
  of a private mmap of the checkpoint, and each weight loader calls
  `param.copy_(loaded_weight)`: a synchronous pageable copy per tensor. On
  the GB10 it runs at 0.13–0.18 GB/s even from a warm page cache. The copy
  engine appears to take a GPU page fault per untouched page (the
  `UVM GPU1 BH` kernel thread was busy; inferred, not traced).
  `h2d_bench.py` in the study reproduces it on 1 GB of expert tensors.
- **Patch 1 (measured):** pin each MoE host tensor before its copy, 4 lines
  in `fused_moe/routed_experts.py`. Main weights 471.1 → 180.3 s, start to
  ready 791.5 → 408.6 s, same greedy output. The version rebased onto `main`
  is untested.
- **Upstream refs:** none for the copy itself. Upstream's `instanttensor`
  loader sidesteps it for users who opt in; the default `lazy` path is
  still slow.
- **Proposed action:** a PR with patch 1 (main version), after a
  discrete-GPU measurement; then, optionally, patch 1b (an opt-in `pinned`
  strategy: large sequential reads into pinned windows) and patch 2 (a
  checkpoint-name filter so the MTP drafter stops reading 27 GB it drops).
  1b and 2 are UNTESTED drafts. A day or two with tests.
- **Links:** `vllm/PATCHES.md` in the study, patches 1, 1b and 2.

## Scanning the expert mapping per tensor (fixed upstream)

- **Status:** fixed upstream at
  [vllm#58720](https://github.com/vllm-project/vllm/pull/58720)
  (`ad6817b68`, 2026-09-26).
- **Problem:** `RoutedExperts.load_weights` scans a 2,563-entry mapping for
  every per-expert tensor, never stopping early: 75.8 µs a tensor, about
  22 s for this checkpoint (estimate).
- **Proposed action:** none upstream. Mia's image predates the fix; a
  backport diff against `8e685d198` is in the study (patch 3).

## `VLLM_LOGGING_LEVEL=DEBUG` breaks CUDA graph capture

- **Status:** open.
- **Found:** 2026-09-28, the image above; the code is unchanged on `main`
  `f6877f56`.
- **Problem:** with DEBUG logging, startup fails during the drafter's
  prefill graph capture. `IrOp.dispatch` (`vllm/ir/op.py`, main lines
  346–354) logs skipped providers through `tensors_str_no_data`
  (`logging_utils/torch_tensor.py`), which calls `str(tensor)`. PyTorch's
  formatter runs `masked_select` on the device, which is not permitted
  while a stream is capturing, and the capture is broken. Observed once.
  Minimal repro (untested): call `tensors_str_no_data((x,))` on a CUDA
  tensor inside `torch.cuda.graph(...)`.
- **Proposed action:** a bug report with the traceback, and the fix
  `patches/bugfix-debug-log-graph-capture.UNTESTED.diff` from the study
  (applies to the image and `main`, byte-compiles; not run). Small.
- **Links:** `vllm/PATCHES.md` in the study, "Bug report draft".

## Mia's PLE-offload patch misreads the GB10's stream memory operations (RE-029)

- **Status:** open.
- **Found:** 2026-09-28, recipe `b8439110`, `spark`, driver 580.178.04.
- **Problem:** the recipe's `patch_ple_offload.py` states that the GB10
  reports `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 0` and that after a
  `cuStreamWaitValue32` "the *next* kernel launch on that stream blocks the
  host thread". Attribute 92 is the deprecated `_V1`; the current attributes
  (122, 123) read 1, and the operations work. With the driver API, the next
  launch did not block: a stream takes about 1,020 operations first
  ([cuda.md](cuda.md)).
- **Proposed action:** optional: a short comment to the recipe's authors.
  Low priority; their workaround still works.
- **Links:** RE-029 in [rough-edges.md](../rough-edges.md).

## Same-history acceptance varies across cache paths and repeats

2026-10-04, pinned image/recipe above, Spark B. The [native/reference
controls](../experiments/qwen38-same-history/README.md#cache-path-and-repeat-controls-2026-10-04)
supply identical 31,746 prompt IDs and record actual three-draft first
verifies. Default-mode p3 acceptance varies from 1/3 to 3/3; a warm repeat
changes a target argmax on the same conditioning as well as draft proposals.
With both recipe determinism switches enabled, one p0-warmed p3 repeat
has byte-identical full target heads and matches native at 1/3, but cold
and cached p3 still differ. This does not isolate a faulty operation or
establish general repeatability; the image is a private build, and upstream
main was not tested.

Reference acceptance diagnostics must record request order, cache counts,
actual draft IDs and target heads, and repeat the reference. The original
four-anchor 9/12 versus 11/12 observation is not a stable-deficit proof. No
engine patch or upstream report is ready; preserve the linked reproduction
and isolate the cache/math path before attributing a defect.
