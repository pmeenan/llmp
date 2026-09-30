<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 MTP at the final runtime's 128K prompt (M3)

The final 512-token runtime ladder leaves Qwen3.8's MTP speed gate open,
especially at 128K: the default prefix head gives 45.034 tok/s against
Mia's 48.652 on that prompt. This study calibrates draft depth against the
same state, profiles the verify, and tests two possible transfers from the
[ds4 study](../ds4-study/README.md). Neither expert-kernel experiment improved
end-to-end speed, so neither implementation is retained. The existing
default head and depth policy remain in place.

## Protocol and harness

Host `spark-b` (GB10), 2026-09-29, driver 580.178.04, CUDA 13.4.92, SDK
`aarch64-e0a0c85c42806fb1`. Source is `26bae74` plus the benchmark changes
in this unit. The target and its F16 caches, weight quantization, sampling
implementation and runtime wake policy are unchanged. The head artifacts
are:

| Role | Artifact |
| --- | --- |
| Target | `c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93` |
| Prefix, 65,536 rows | `056a750e3a90be3ae6a4b12bb963ce45290aaa6f52b5ba9799e777d491f80aea` |
| Optional selected, 47,172 rows | `8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40` |

The selected head was imported in jitLLM from Mia's externally supplied
47,172-ID curated list, SHA-256
`ee819d2560b52ba1351acdd1c4a0244b77bb60694a7660c6363097a482b0afb5`,
from recipe `b8439110eec0230facbe4ddf0dffe01b8f769be0`.
The generic selected-head importer is jitLLM's Apache-2.0 code; the external
list and Mia's AGPL code are not shipped in this repository. See the
[draft-head study](../qwen38-draft-head/README.md) and
[license record](../../licensing.md#miaai-single-spark-recipe).
The tokenizer is from
`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6`.

The canonical input is
`~/.local/share/jitllm/m3lc/prompts/qwen3.8/128k.json` on either Spark;
its user-content SHA-256 is
`b7154f4bdfea17ea92e20ddaa95b2ece37ea306972af7039894fd0e81854da1d`.
It renders to 128,799 tokens, with the runtime's stable boundary at
128,794, immediately before the assistant opening.

`jitllm_qwen38_spec --runtime-prefill on` now follows serving's split at
that boundary and its `RunPrefillChunks` policy: 4,096-row chunks, tiled
tails down to eight rows, then the assistant opening. This matters because
a multi-row prefill and repeated short rows can differ in arithmetic.
The earlier native 128K study also used a different prompt and cannot
replace this comparator. `--check timing` performs the lean argmax-only
speculative path without the plain and teacher-forced quality controls.
It reports acceptance, requested draft depths, actual verify rows and
draft/verify costs. Timing results alone establish neither near-tie
agreement nor sampled distribution.

Long-context timing trials generated 512 tokens, with a 262,144-token ceiling,
graphs on, the runtime wake, one run per case and no concurrent GPU job.
Every load required at least 105 GiB MemAvailable and no other GPU/model
process. Rates exclude prefill and the first token, as in the final
runtime ladder. These are coarse single-run comparisons, not confidence
intervals.

## Depth and confidence

Fixed-depth calibration, job `m3-qwen-mtp-fixed`, 21:36:02–21:44:00:

| Head | Draft depth | tok/s | Draft ms/step | Verify ms/step | Whole ms/step | Acceptance by draft position |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prefix | 3 | 45.062 | 8.988 | 55.763 | 64.800 | 0.7874, 0.6416, 0.5087 |
| Prefix | 4 | 45.613 | 11.632 | 61.534 | 73.222 | 0.7843, 0.6447, 0.5066, 0.4172 |
| Prefix | 5 | 42.500 | 14.261 | 66.942 | 81.240 | 0.7619, 0.6370, 0.4795, 0.3379, 0.2690 |
| Selected | 3 | 45.657 | 8.084 | 55.823 | 63.955 | 0.7931, 0.6416, 0.5029 |
| Selected | 4 | 45.756 | 10.382 | 62.086 | 72.519 | 0.7727, 0.6340, 0.5033, 0.4211 |
| Selected | 5 | 43.108 | 12.514 | 66.459 | 79.025 | 0.7651, 0.6284, 0.4595, 0.3265, 0.2585 |

Prefix depth 3 reproduces the final runtime's adaptive result within
0.1%, resolving the previous harness/runtime prefill difference. Depth 4
adds 1.2% for prefix and 0.2% for selected; depth 5 loses speed. No depth
here reaches Mia's 48.652. Peak MemAvailable drops were 82.58–83.12 GiB.
Depths 3 and 4 dropped no graphs; depth 5 dropped three.

The existing benchmark confidence window, 0.3, drafts four tokens and
truncates verification after the first low-confidence later token. It
keeps the first draft and the target remains authoritative. Job
`m3-qwen-mtp-window`, 22:15:00–22:17:39:

| Head | tok/s | Draft ms/step | Verify ms/step | Whole ms/step |
| --- | ---: | ---: | ---: | ---: |
| Prefix | 45.481 | 11.824 | 61.094 | 72.958 |
| Selected | 44.669 | 10.316 | 61.573 | 71.947 |

This cutoff does not improve on fixed depth 4. It still computes all four
drafts before truncation; it saves verify work only. It supplies no reason
to change the runtime's policy.

## Decode profile and expert transfers

`m3-qwen-mtp-profile`, 21:44:31–21:45:47, profiles only 128 generated
tokens after the same prefill, prefix head, fixed depth 3. Capture uses
`--profile-decode on`, nsys `--capture-range=cudaProfilerApi`, graph-node
tracing and `--capture-range-end=stop`. With repeats, that capture ends
after the first speculative call. The graph summary covers 42 four-row
verifies:

| Part | Mean accumulated kernel time per verify, ms | Share |
| --- | ---: | ---: |
| NVFP4 experts | 20.040 | 35.4% |
| MXFP8 dense products | 16.477 | 29.1% |
| GGML float products | 6.551 | 11.6% |
| cuBLAS BF16 products | 6.330 | 11.2% |
| Hyper-connection preparation | 1.179 | 2.1% |
| QSA scoring | 0.887 | 1.6% |
| QSA attention | 0.734 | 1.3% |

Total accumulated kernel time is 56.648 ms; the graph span is 54.812 ms.
PDL allows overlap, so accumulated kernel time is not elapsed time. The
draft-head float product occupies 4.964–5.611 ms in this **prefix-only**
capture, depending on the catch-up shape. The selected head's lower whole
draft cost in the calibration is measured separately; those prefix kernel
times are not a selected-head measurement.

QSA attention is already a small part of decode. The ds4 wide sparse
query-union optimization operates on a different mask representation and
does not directly replace Qwen's per-row selected-cell lists. Dense MXFP8
products already share weights across up to eight columns.

Two bit-exact expert prototypes were tested against the preserved original
kernel binary, on both heads and the original 128K prompt:

| Prototype | Prefix candidate / control tok/s | Selected candidate / control tok/s |
| --- | ---: | ---: |
| Group matching slots in pairs, load each expert weight once | 44.830 / 44.834 | 45.284 / 45.469 |
| Prepare each logical input's Q8 codes/scale once, reuse over output rows | 44.678 / 44.734 | 45.106 / 45.527 |

The paired-slot prototype repeats the
[earlier grouped-expert experiment](../qwen38-mtp/README.md#judgement-calls)
at the final prompt. That earlier experiment attributed its neutral/slower
result to L2 already serving duplicate expert reads. The new profile is
consistent with it: experts take 20.395 ms with sharing versus 20.040 ms
original. Register counts rise from 98/106 (down/gate) to 128/134, with
an 8-byte down-kernel stack; neither reports local spill allocation.

Q8 preparation retains each original activation code, per-16-value scale,
integer dot product, FMA/block order and reduction. Prepared products use
78/74 registers and no stack/local allocation, plus a 28-register
quantization kernel. Their extra scratch is half the logical F32 input
size, allocated through the existing launch pool and included in planning.
Despite less repeated quantization, end-to-end speed is neutral.

Both prototypes pass six expert GPU tests, including maximum 512 routed
slots, duplicate and invalid expert IDs, partial output rows and original
per-token arithmetic. Q8 preparation also checks actual scratch against
the plan. Each candidate and its paired control produces exactly the same
512 tokens for each head. These checks validate the experiments; their
neutral performance does not justify retaining their code. The earlier
[BF16 vector verify experiment](../tensorfold-techniques/README.md)
already lost to cuBLAS, so it was not repeated.

Reusable pieces remain possible experiments: Q8 preparation reduces
repeated quantization and register use, but adds a launch and global
scratch traffic. Producing those same codes/scales inside an existing
activation producer could remove that added launch. The expert grouping
loop reuses decoding as well as weight loads; it may be useful when a
different shape or working set fails to get the same reuse from L2.
Neither possibility is a measured speed gain, and lower register counts
alone do not establish one.

## Incremental confidence stop

A third prototype avoids later draft work itself. It runs the initial
two-pass catch-up graph, exports the last MTP streams into a 40 KiB
working slice after the bounded verify snapshot, and adds one pass at a
time while the preceding candidate's confidence is at least 0.5, up to
four. It leaves the preserved target-stream rows and state layout intact.
No sampled policy was changed. Job `m3-qwen-mtp-incremental-ab`,
22:39:07–22:41:46, same 128K/512-token protocol:

| Head | tok/s | Draft ms/step | Verify ms/step | Whole ms/step | Steps at depth 2 / 3 / 4 |
| --- | ---: | ---: | ---: | ---: | --- |
| Prefix | 45.001 | 11.033 | 61.257 | 72.326 | 17 / 22 / 118 |
| Selected | 45.564 | 9.775 | 60.283 | 70.094 | 23 / 22 / 115 |

Neither improves on fixed depth 4. Its graph-planning test passes, and a
short prefix-head forced control passes: 32 tokens equal plain greedy,
11 steps, seven with rejected rows, zero state differences, all three
rejection positions covered. The forced control uses the same
incremental producer on both sides; a direct comparison to the old
whole-draft producer at cache/block boundaries was not completed. These
results justify dropping a neutral experiment, not a claim that its
draft arithmetic is identical to the old producer. Its source and
binary are retained externally and its code is removed from this unit.

## Selected-head sampled distribution

The old plain-sampling capture predates the current sparse QSA policy,
so it was not reused. Job `m3-qwen-mtp-sampled-curated`,
22:18:19–22:33:45, reran both modes with the unchanged original kernels:
`capital`, `haiku`, `sky`, `fibonacci` from
[the fixed prompt set](../fast-swap/prompts.json), seeds 0–255, the first
eight tokens, temperature 1, top-k 0, top-p 1, min-p 0, fixed draft depth
2, 8K ceiling and normal harness prefill. The speculative mode used the
final selected-head artifact above; the plain mode used the same target
without MTP injection. Each mode produced 8,192 sampled tokens.

| Prompt | Total variation, top 16 pooled tokens plus other | Bound |
| --- | ---: | ---: |
| capital | 0.0312 | 0.1 |
| haiku | 0.0146 | 0.1 |
| sky | 0.0220 | 0.1 |
| fibonacci | 0.0337 | 0.1 |

All four pass. Plain sampling took 497.4 s and selected-head speculation
409.0 s; these are protocol durations, not a long-context speed
comparison. Both modes reported no problems or refused graphs. This
adds sampled-distribution evidence for the optional selected head under
the existing finite protocol, alongside its earlier greedy, forced and
swap controls. It does not establish distribution equality for every
prompt or seed.

## Raw provenance and repeat

External results are on `spark-b:~/scratch/m3-qwen-mtp-speed/`:

- `fixed-{prefix,selected}-{3,4,5}/spec.json` and logs;
- `window-{prefix,selected}-4-03/spec.json` and logs;
- `profile-prefix-3.{nsys-rep,sqlite}`, graph summary and kernel table;
- `shared-{prefix,selected}-3/`, `baseline-{prefix,selected}-3/`;
- `prepared-{prefix,selected}-3/`, `control-{prefix,selected}-3/`;
- compiler resource tables `shared-resources.txt`, `prepared-resources.txt`;
- immutable binaries `qwen38-spec-baseline`, `qwen38-spec-sharing-only`,
  `qwen38-spec-prepared`; the prepared prototype's source is external in
  `prepared-source/`;
- `sharing-patches.json` preserves the original sharing prototype's two
  source patches recovered from its tool records;
- `incremental-{prefix,selected}-4-05/`, `incremental-forced/`,
  `incremental-full.diff` and `qwen38-spec-incremental`;
- `sampled-current-{plain,spec}/`, including counts and the comparison.

Baseline binary SHA-256:
`22e3bea83050d1b10910383b2512eff4b0e85c1adb9331056a480bd98b4749f8`;
sharing prototype:
`1295cde3492edbcb02249dbc297fa96164ec58a13880e1ca2780536d57024ad7`;
prepared prototype:
`b2bf474a0648a25feb5b900a5448b1b9e796644f07f912e279f2f01120404cc6`.
Sharing patch archive SHA-256:
`e3796ca4e6600579884161728a58f5fffab8c4c87fa39d44abc207ce0f5eff7f`;
prepared `jitllm_moe.cu`:
`dd86ba7d9e60beba281e4baafd1725fabd5c41ac933281cb4205def7239e91f4`.

To repeat a fixed-depth case, wrap the canonical JSON prompt in the
harness's `decode` array without changing its messages, and use:

```sh
jitllm_qwen38_spec --qwen38-artifact "$target" --drafter "$drafter" \
  --tokenizer "$tokenizer" --prompts "$wrapped_prompt" --only qwen3.8-128k \
  --context 262144 --prefill-chunk 4096 --check timing --tokens 512 \
  --repeats 1 --draft 3 --adaptive-depth off --runtime-prefill on --out "$out"
```

Change only `--draft` for the depth sweep. Add `--window 0.3` at depth 4
for the cutoff experiment. The target and head paths above are under the
Spark's `~/.local/share/jitllm/m3-artifacts/`.
