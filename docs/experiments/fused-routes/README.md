<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fusion-preserving MoE route capture, 2026-09-22

[RE-006](../../rough-edges.md#re-006-reading-moe-routes-through-the-llamacpp-callback-changes-the-cuda-path--2026-09-21-status-worked-around)
forced the paging study to disable CUDA fusion and graphs, because reading
routes through llama.cpp's eval callback split the graph inside the fused
top-k kernel's node sequence. This experiment fixes that for the external
reference. **Reading each layer's selected-expert IDs at the end of its
gated-activation fusion group leaves logits bit-identical to an untraced run
with upstream fusion and CUDA graphs**, on Gemma 4 26B A4B and Ornith 1.5 35B.
It uses the unmodified, digest-pinned llama.cpp image. This concerns only
external route observation; it is not llmpalooza runtime code or paging evidence.

## Method

[`route_capture.cc`](route_capture.cc) is the paging study's session capture
with the same inputs, validation and route record format, plus a hash of
every output's full logit row (FNV-1a over the raw floats). Mode 4 registers
an eval callback that records each `ffn_moe_topk-N` pointer without
requesting data. It requests data only at the `GLU` node that consumes that
layer's expert products (`MUL_MAT_ID` with those IDs as `src[2]`). The
scheduler therefore splits after the gated activation, which ends a fusion
group, and before the down projection, which has not yet consumed the IDs,
so their buffer is still live. The top-k, gate/up/GLU and down-projection
fusion groups all stay intact. Reads use the exact tensor the kernels
consume, validate strides, range and uniqueness, and fail closed.

The workload is the study's four-turn teacher-forced sessions (`normal-spec`,
batch 512, prefix reuse, default SWA, F16 KV, Flash Attention, all layers on
`spark-c4e2`'s GB10, driver 580.178.04). Each model ran five arms in fresh
processes: untraced, the legacy split callback, the fusion boundary, and the
untraced and boundary arms with fusion and CUDA graphs disabled.
[`run_routes.py`](run_routes.py) verifies the model, input, header and
reference-capture identities, runs every arm without network access and
compares outputs with [`compare_routes.py`](compare_routes.py). It requires
exact predictions and logit hashes for boundary vs untraced in both
configurations, and exact predictions and ordered routes when reproducing
the study's disabled-optimization reference. A mismatch saves the comparison
receipt and exits unsuccessfully before reporting `complete`. The legacy
split and cross-configuration comparisons remain diagnostics: their
differences are expected and do not fail those equality controls.

## Results

| Comparison (3,072 outputs per model) | Gemma predictions / logit rows differing | Ornith predictions / logit rows differing |
| --- | ---: | ---: |
| Fusion boundary vs untraced, fusion and graphs on | **0 / 0** | **0 / 0** |
| Legacy split callback vs untraced, fusion and graphs on | 55 / 3,072 | 99 / 3,071 |
| Fusion boundary vs untraced, both off | 0 / 0 | 0 / 0 |
| Untraced, fusion and graphs both off vs both on | 51 / 3,072 | 110 / 3,072 |
| Untraced, both off vs the study's recorded capture | 0 predictions | 0 predictions |

With fusion and graphs off, the boundary reads reproduce the study's recorded
legacy routes exactly: all 94,110 Gemma and 123,000 Ornith route events, row for
row and in order. This confirms the new read point observes the same tensor
the old one did. The legacy callback still perturbs the optimized engine in
both models.

Routes themselves depend on the numerical plan. Between the optimized plan
and the plan with fusion and graphs both disabled, the selected expert *set*
differs in 452,483 of 1,117,920
Gemma token-layer rows and 87,689 of 202,600 Ornith rows. The difference is
near zero at layer 0 (4 of 37,264 Gemma rows; none for Ornith) and grows with
depth, reaching 63–78% of rows in each model's last three layers, mostly
in prefill. Decode rows differ 11.5% (Gemma) and 26.4% (Ornith).
Accumulated numerical drift crossing near-tied router scores is a plausible
explanation, but router scores and selection margins were not measured.
The controls change fusion and CUDA graphs together, so their individual
contributions are not isolated. The exact boundary/untraced output controls
establish transparency within each tested configuration.
Per-layer counts are in [`results.json`](results.json). The paging study's
recorded routes therefore describe the plan with fusion and graphs disabled;
locality derived from them is conditional on that plan.

## Rejected design: routes as graph outputs

The first attempt, [`route-outputs.patch`](route-outputs.patch), rebuilt only
libllama so each layer's IDs were copied (`ggml_cont`) into an output tensor
appended after every other node. It avoided callback splits, but on Gemma:

- the patched library with the feature off matched the official library
  bit-for-bit on all 3,072 outputs;
- with it on, 44 predictions and every logit row differed with fusion and
  graphs on, but nothing differed with both disabled;
- in a 17-output diagnostic, a callback that never requests data changed
  nothing, while the appended outputs changed every logit row with or
  without a callback.

Extending the IDs' lifetime and adding outputs changes the compute-buffer
layout. One hypothesis is that this changes which fusions qualify, since
the CUDA backend checks fused operands' memory ranges. These controls
disabled fusion and CUDA graphs together; neither fusion's individual role
nor the specific affected operation was isolated. The patch is retained,
with its upstream MIT notice, only to reproduce this negative result; see
[RE-010](../../rough-edges.md#re-010-adding-graph-outputs-changes-logits-with-cuda-optimizations--2026-09-22-status-open).

## Limits

Two models, one teacher-forced workload and one host. The method requires a
gated activation (`GLU`) after each MoE layer's expert products; other
activation layouts fail closed rather than silently reading late. Any future
model or llama.cpp revision needs the same untraced/boundary equality check.
The split still adds a synchronization per MoE layer, so capture step times
are not performance evidence. The study's existing captures were not
re-recorded.

## Provenance and reproduction

[`pins.json`](pins.json) records the image, source revision, header and patch
hashes and workload identity; [`results.json`](results.json) holds all
comparisons and SHA-256 receipts for every raw events, predictions,
logit-hash and log file, which stay under
`/home/pmeenan/.local/share/llmp/fused-routes-20260922` on `spark`
(`run-2` final, `run-1` and `diag-1` for the rejected design). The runner
builds the harness with the image's GCC 13.3.0 exactly as `pins.json` lists.
The runner hash in `results.json` identifies the original capture version.
The current runner adds required-comparison enforcement after writing the
receipt; the capture harness, comparer, measured values and historical
hashes are unchanged. Its new acceptance checks pass the recorded comparisons
and CPU regression tests; no new GPU measurement is implied.
With the paging study's external inputs:

```bash
P=/home/pmeenan/.local/share/llmp/paging
DOCKER="sudo -n docker" python3 run_routes.py "$P/normal-spec.json" \
  /home/pmeenan/.local/share/llmp/reference-models \
  "$P/llama.cpp-b29c606e28a01b1bc8c1351026a0fa6e616bf6c4" "$P/sessions-1" \
  "$P/normal-capture-2" /path/to/new-output
```

`test_compare_routes.py` and `test_run_routes.py` cover comparison and
acceptance rules, including a runner failure with unchanged predictions but
different logit hashes
(`python3 -m unittest discover -s docs/experiments/fused-routes`).
