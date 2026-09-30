<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 MXFP8 vector scheduling — 2026-09-30

Small-output MXFP8 products benefit from more independent warps and fewer
rows per warp. A bounded GB10 schedule transfer improves the original
128K, 512-output trial by 0.8% with the prefix head and 1.3% with the
curated head. It preserves the format, arithmetic, F16 caches and one-row
decode path. This is an incremental improvement; the 128K MTP speed gate
remains open.

## Scope and conditions

The prior [MTP profile](../qwen38-mtp-speed/README.md) attributes 29.1%
of target verification kernel time to MXFP8 vector products. Their existing
kernel already shares weights across input columns, but its fixed grouping
leaves very few CTAs for small output counts.

The benchmark copies jitLLM's owned MXFP8 arithmetic and sweeps rows per
warp 1/2/4, warps per block 4/8, and two input/weight loading orders.
Every lane retains its original `v = lane + 32j` order, sixteen ordered
FMAs per vector, scale FMA per block, and XOR reduction offsets 16 through
1. PDL still waits before reading inputs and triggers only after weight
reads; inputs and outputs remain unrestricted. Both the production and
copied units compile without `-use_fast_math`.

Runs use `spark-b` / `spark-56f5`, GB10 with 48 SMs and measured 24 MiB
L2, SDK `aarch64-e0a0c85c42806fb1`, and base `e4c2fd1` plus this slice.
Generated finite E4M3 weights include signs, zero and extreme finite
codes; E8M0 scales include zero's subnormal scale. F32 inputs and outputs
retain their existing precision. These fixtures establish arithmetic and
scheduling behavior, not model quality on their own.

The timing graph touches every allocated weight bank, with a target
working set of 256 MiB and a cap of 1,024 banks. Every measured model
shape exceeds L2; tiny correctness shapes report their smaller set and
are not timed. Each timing is the median of three CUDA-event graph
batches, normalized by the actual number of calls, with production
controls before and after the candidates.

## Isolated decision

`m3-qwen-mxfp8-sweep2`, 00:11:41–00:12:06, checks 288 candidate
combinations over short/tail shapes: columns 1–8, outputs 5/47/257,
inputs 32/640/6,144, and twelve F32 elements of stride padding. All
repeat exactly and match the actual production launcher bit for bit.
The same job times eight real shapes at three/four/five columns; all
288 of those candidate comparisons are exact too.

Large products offer little gain, with some production controls drifting
2–5%. A second small-output sweep, `m3-qwen-mxfp8-small`,
00:14:54–00:15:11, covers all multi-row column counts. Its selected
three/four/five-column times, in microseconds:

| Columns | Outputs | Original | Selected | Rows/warp | Warps/block |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 48 | 6.387 | 3.745 | 1 | 4 |
| 3 | 512 | 8.035 | 7.507 | 2 | 4 |
| 3 | 640 | 9.320 | 8.787 | 2 | 4 |
| 4 | 48 | 9.978 | 4.146 | 1 | 4 |
| 4 | 512 | 11.332 | 8.051 | 2 | 4 |
| 4 | 640 | 12.124 | 9.466 | 2 | 4 |
| 5 | 48 | 7.814 | 4.556 | 1 | 4 |
| 5 | 512 | 11.593 | 8.896 | 4 | 4 |
| 5 | 640 | 12.568 | 9.931 | 4 | 4 |

The retained dispatch is restricted to GB10, 2,560 inputs and these
measured shapes:

- 48 outputs: one row per warp, four warps per block at columns 2–8;
- 512/640 outputs: two rows per warp, four warps per block at columns
  3/4, and four rows per warp, four warps per block at columns 5/7.

At 512/640 outputs the attempted rule regresses two-column products by
about 7–8%, six-column products by about 51–56%, and offers no gain at
eight columns. Those column counts keep their original schedule. Other
input/output shapes, one-row decode and other devices also keep theirs.
No scratch allocation or extra launch is added.

Loading one input column at a time reduces registers at several shapes
but is never the best measured schedule in this confirmation. For
example, columns 4 / rows 1 / warps 8 uses 56 registers instead of 91,
with no resulting selected speed win. That loading variant remains a
benchmark comparison, not a production choice.

## Whole-model comparison

`m3-qwen-mxfp8-model`, 00:24:00–00:29:18, runs the canonical phase-1
128K prompt, fixed depth 3 and 512 generated tokens. Runtime-style
prefill uses 4,096-row chunks and the stable generation boundary:
128,799 rendered tokens, boundary 128,794. User-content SHA-256 is
`b7154f4bdfea17ea92e20ddaa95b2ece37ea306972af7039894fd0e81854da1d`.
Candidate then baseline runs separately for each head; each load gates
available memory and other GPU processes.

| Head | Baseline tok/s | Candidate tok/s | Gain | Baseline verify ms | Candidate verify ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefix 65,536 | 45.242 | 45.616 | 0.83% | 55.565 | 54.913 |
| Curated 47,172 | 45.652 | 46.261 | 1.33% | 55.822 | 55.019 |

Each pair has identical prompt IDs and all 512 generated tokens.
Acceptance and the verify-row trajectory match too. All four runs replay
167 target graphs, with no graph refusals or drops and no reported
problems. Peak MemAvailable drops are 82.88–83.14 GiB. The modest result
does not replace the final adaptive runtime ladder or its open speed
gate; Mia's 128K comparator remains 48.652 tok/s.

The unchanged target artifact is `c4fb47a9…`; prefix head `056a750e…`
and final curated head `8600a998…`. The curated artifact imports the
externally supplied Mia 47,172-ID list, input SHA-256 `ee819d25…`; it is
not an independently generated vocabulary. Its source, license and
complete artifact pins are in the [draft-head study](../qwen38-draft-head/README.md#provenance-and-reproduction).

The baseline and candidate binaries are built from the same base and
recorded in external scratch. SHA-256:

- baseline `fec47be09c80c86a0a6772bc678d3ee4a1421482a8faffa3cb007517ef7218e9`;
- candidate `f78a795453258aefc0be23ae02c2bd475b96c35cd44d4b1af98e815f04155cf0`.

## Repeat, rejection and swap controls

`m3-qwen-mxfp8-controls`, 00:30:58–00:32:59, completes all six checks
at an 8K context ceiling, with the original fixed `prose`/`capital`
prompts and both heads:

- Two 64-token timing generations repeat their tokens exactly, replay
  target and draft graphs, and report no problems. Timing mode keeps
  argmax verdicts; this is not a full-logit repeat comparison.
- Forced rejection over 32 generated tokens covers all-reject, one and
  two accepted cases. Each head has 14 steps, 10 with rejected rows, and
  zero state fingerprints different from the control. All 32 tokens equal
  the plain run.
- Swap after a rejected step preserves all 32 tokens and logits, with
  zero state or logit differences. After return, seven verifies and nine
  drafts replay graphs kept across the swap.

The swap fixture and expected logits hash are the unchanged pins from
the draft-head study. Together with exact scheduling controls, the fresh
512-token paired trajectories and the unchanged sampling/draft policy,
these support the arithmetic-only transfer. The prior fresh curated
distribution protocol is recorded in the [MTP calibration](../qwen38-mtp-speed/README.md);
it was not unnecessarily repeated for this bit-preserving schedule.

## Reproduction and remaining work

The reusable diagnostic is `jitllm_qwen38_mxfp8_tune`, built only in CUDA
profiles with GB10 device code. For example:

```sh
jitllm_qwen38_mxfp8_tune --check-only on --padding 12 \
  --shape 3,47,640 --shape 8,257,6144
jitllm_qwen38_mxfp8_tune --shape 3,48,2560 --shape 4,512,2560 \
  --shape 5,640,2560 --reps 1024
```

Raw JSON lines, logs, source/binary hashes and copied control binaries
remain at `spark-b:~/scratch/m3-qwen-mxfp8-tune/`; no raw traces or
generated operands are shipped. The benchmark's temporary allocations
remain protected by completion, and unknown CUDA/provider failure exits
without resource destructors or later GPU work.

The next source lead is a benchmark-only crossover comparison with the
existing MXFP8 tensor-core path at three/four/five rows, charging input
quantization, scale swizzle, launches and workspace. That path changes
F32 activation precision and is not authorized by this arithmetic-only
result; quality, distribution, rejection, repeat and swap evidence would
be required before a production choice. The global graph threshold is
unchanged.
