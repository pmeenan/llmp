<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model reference comparisons

Every new model or quantization needs a matched quality and performance
comparison after integration, including its production solo and optimized
batched paths. The milestone's numerical, speed, memory, context and state
criteria still apply; primitive tests do not qualify a model.

## Current llama.cpp reference

For new comparisons, use the [official release reference](experiments/llama-reference-refresh/README.md)
and its frozen source/image pair. This refresh preserves every historical
experiment pin and result; existing comparisons are not requalified by a
dependency update. Runtime/API checks and model qualification have separate
completion records.

## TensorFold refreshes at each task

Use the latest [TensorFold](https://github.com/ashhart/TensorFold) as a
competitive performance target wherever it supports the model, execution
backend and topology (owner, 2026-10-04). **Refresh at the start of each
applicable model, quantization or optimization task**, not just at milestone
entry. Resolve upstream's default-branch HEAD and inspect that revision's
recipes, formats and batching support. Record the full commit, package
version and observation date in the task's comparison report. An older
observation in this document is never the next task's default pin.

Freeze that revision for the task's matched runs. An upstream update during
a comparison requires a separate comparison with its own pin; do not mix
versions in bookends or replace a historical result's provenance. If work
resumes after a substantial delay, check upstream again before starting a
new comparison batch and name which pin that batch evaluates.

Keep the same-format correctness oracle. TensorFold can fill that role
when the checkpoint representation and execution semantics match. A
different quantization is a speed/memory comparator, with its quality
measured separately; it cannot establish same-format token agreement.
For an applicable TensorFold configuration, meeting the older oracle's
speed alone does not establish competitiveness: also report the matched
TensorFold performance and any remaining gap. A gap requires further work
or an explicit owner-approved exception before claiming the performance
gate passed.

Record checkpoint and drafter revisions, container digest and runtime
dependencies, device/topology, effective precision and KV format, prompt
tokens and reasoning mode, context, memory budget, warmup and cache state.
Record effective cache topology and capacity, sliding-window size, physical
chunk size and attention read policy, including `swa_full` and `kv_unified`
where applicable. Use the reference's normal production CLI/server settings
for competitive comparisons; explicitly override differing C-API defaults or
label the alternate recipe. Cache layout and read order can change arithmetic,
so agreement must be measured for the actual recipe. Historical full-cache
runs retain their measured results but do not establish representative memory
ratios or speed parity for a ring-cache production recipe.

Compare prefill and completed decode at 8K and depth, peak memory, and
concurrency 1/2/4/8/12 where both engines support it. Report unsupported
reference configurations explicitly. Speculation and lower-precision modes
get their own quality controls and timings. Run the reference ourselves on
the target; upstream headline rates are not our measurements.

## Coverage observed on 2026-10-04

Default-branch HEAD resolved with `git ls-remote` to
[`609ca419abecebdc5a059498a613680bd3aa847f`](https://github.com/ashhart/TensorFold/tree/609ca419abecebdc5a059498a613680bd3aa847f),
whose [package version](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/src/tensorfold/__init__.py)
is **0.6.5**. Its [README](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md)
declares the following coverage. These are reference candidates, not
jitLLM qualification results or proof that our approved checkpoints match.
Format-specific topology limits come from the pinned
[27B recipe](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/docs/recipes/qwen3.8-27b.md)
and [Flash Next recipe](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/docs/recipes/qwen3.8-flash-next.md).

| Model | Declared CUDA formats/topology | Comparison limits at this observation |
| --- | --- | --- |
| Qwen3.8-27B | MLX affine on one/two ranks; NVFP4 and experimental EXL3 on one GPU | Continuous batching available; verify the exact export and EXL3 layout |
| Qwen3.8 Flash Next | MLX affine on one/two ranks; NVFP4/FP8 exports and experimental EXL3 on one GPU | Continuous batching available; match dense-layer precision and KV dtype |
| Nemotron 3.5 Lightning 30B-A3B | MLX affine; one/two ranks | CUDA requests serialize; this does not imply support for Nemotron 3 Super |
| GLM-5.3 Flash | MLX affine, experimental EXL3/TR3; two ranks | CUDA requests serialize; verify M4's exact format and checkpoint |

Gemma 4 26B-A4B, DeepSeek V4 Flash and Ternary Bonsai2 appear as
**MLX-backend** models at this pin. That is not a CUDA GB10 comparator;
retain their applicable GPU references and check again at their tasks.
DeepSeek v4.1 Flash's M4 reference is likewise resolved at its own task.
The historical [TensorFold assessment](tensorfold-assessment.md) and
[M4 references](m4-references.md) retain their original pins and results;
they do not override this refresh policy.
