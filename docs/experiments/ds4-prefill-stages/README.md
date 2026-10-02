<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek native prefill: ds4 stage mechanisms

Native 8K prefill now runs within about 1–2% of the literal ds4 pipeline.
Eight exact changes make it **19.37% faster** with every complete head
byte-identical to the native goldens. Adding the qualified Q2 D2R down
product makes it **23.70% faster**: 7.64–7.74 s against literal ds4's
about 7.57 s. D2R changes logits, but its 32K perplexity is within 0.04% of
ds4's own. All of them are now the fast plan's defaults, each under its own
shape guard ([production defaults](#production-defaults)). On the served
original checkpoint at 2,048-row chunks only two of them apply, and the
measured runtime 8K prefill gain is 1.26%.

Each row below is one bookended 8K screen on Spark A (`spark-c4e2`,
2026-10-02): base, candidate, base, all in one process. The gain is
`(mean(base times) / candidate − 1) × 100`. Times are paid seconds for two
4096-row chunks, including both complete frontier-head copies.

| Candidate (base) | Base / candidate / base, s | Gain | Heads |
| --- | --- | ---: | --- |
| 1. Existing `--q2-d2r` down (original) | 9.524911 / 8.988674 / 9.454088 | +5.57% | drift |
| 2. HC mix input as F16 rows (original) | 9.486461 / 8.826181 / 9.445169 | +7.25% | exact |
| 3a. SwiGLU in the pair's up write-back (original) | 9.511771 / 9.306682 / 9.471696 | +1.99% | exact |
| 3b. 3a plus down-input Q8 in that write-back (original) | 9.480565 / 9.183841 / 9.443878 | +3.03% | exact |
| 4. Expert sum inside the HC post (2 + 3b) | 8.646382 / 8.495655 / 8.588455 | +1.43% | exact |
| 5. Shared F16 attention input (2 + 3b + 4) | 8.565569 / 8.412689 / 8.516739 | +1.53% | exact |
| 6. Coalesced output-A weight repack (D2R stack 1+2+3a+4+5) | 8.134863 / 7.987255 / 8.079647 | +1.50% | exact |
| 7. Shared quantization for dense Q8_0 pairs (D2R stack + 6) | 8.033900 / 7.922774 / 7.978717 | +1.05% | exact |
| 8. F16 Q into CSA/raw/HCA attention (D2R stack + 6 + 7) | 8.008440 / 7.637240 / 7.961685 | +4.55% | exact |
| **Exact 2+3b+4+5+6+7+8** (original) | 9.485116 / 7.921833 / 9.427379 | **+19.37%** | exact |
| **D2R 1+2+3a+4+5+6+7+8** (original) | 9.583747 / 7.737932 / 9.559654 | **+23.70%** | drift |

Bookends moved −0.25% to −0.74% in every screen. The D2R stack's candidate
time differs between two processes (7.637 vs 7.738 s), so treat the
remaining gap as roughly 1–2%.

**What each change does**
- **Item 2.** The HC post writes the F32 streams and the F16 normalized
  mix-input rows in one kernel, and the mix's cuBLAS product reads the F16
  rows directly. Layer 0 uses a standalone norm-to-F16 kernel. The
  arithmetic is native rms_norm's 1024-thread reduction, then
  round-to-nearest F16.
- **Item 3a.** A jitLLM copy of the occupancy-two J64 compact IQ2 kernel
  runs the up product after gate. Its write-back stores
  `swiglu_clamp(gate, up)`.
- **Item 3b.** That write-back also quantizes the activation straight into
  the Q2_K down product's D2S6 Q8_1 blocks, using quantize.cu's exact lane
  and shuffle arithmetic. The down product skips its own quantization. 3b
  and D2R both replace the down product's input path, so the D2R stack
  uses 3a.
- **Item 4.** The ordered six-slot expert reduction and the shared-expert
  add are formed inside the FFN HC post kernel.
- **Item 5.** Each attention input gets one RN F16 copy, which all of its
  F16-weight products read: the compressor KV and gate products, the
  indexer compressor products and the indexer projection.
- **Item 6.** The output-A operation repacked its raw Q8_0 weights with an
  uncoalesced byte copy on every call (1.44 ms × 86). It now moves 16-bit
  words instead, producing identical bytes.
- **Item 7.** `q_a` with `kv`, and shared up with gate, each read one
  activation. Each pair now shares one Q8_1 quantization of it, and the
  second product runs at the first one's place in the graph.
- **Item 8.** The Q-head writes RN F16 Q, which is exactly what the
  attention kernels would round F32 Q to.
  - CSA and raw attention run a jitLLM-owned copy of GGML's D512 MMA flash
    attention (`fattn_mma_q16.cuh`). It is generated from the locked
    header, changing only the Q pointer type, its load and its strides,
    and it lives in namespace `jitllm_fattn_q16`.
  - HCA runs a jitLLM copy of the ds4 token-tile core with an F16 Q loader.
  - The F32 Q write and read are gone, and the attention kernels' Q-tile
    loads are cheaper.

**D2R quality.** This control used the community artifact, the registered
War-and-Peace 32K text (`755b00b9…`), 4096-row chunks and output-A/HCA, one
process per arm.

| Arm, 32K perplexity | Second half (16,383 targets) | All 32,767 targets |
| --- | ---: | ---: |
| Native without D2R | 2.973515 | 2.689859 |
| Native with D2R | 2.981815 | 2.690088 |
| Original ds4, recorded on the same checkpoint, text and scope | 2.980670 | — |

D2R is +0.28% against native and +0.04% against ds4, well inside the
registered 3% bound. The 0.947-nat greedy oracle cannot test D2R: that
oracle's checkpoint (`8a355bfb…`) has IQ3_XXS down experts, which D2R does
not handle.

**Remaining stage gaps.** The native column is the final D2R stack's marked
pass, classified by graph dependencies. The literal column is the
[ds4 restoration profile](../ds4-restoration-profile/README.md). Native
intervals include host gaps; one nsys pass showed the GPU busy for 99% of
the paid span. The fused post steps span several categories, so the HC and
output rows are compared as one group.

| Stage group | Native now, ms | Literal ds4, ms | Gap |
| --- | ---: | ---: | ---: |
| Routed FFN (pair GLU 1601, D2R 1137) | 2742 | 2642 | +100 |
| Compression and indexer | 446 | 349 | +97 |
| Attention (CSA/raw flash 657, HCA 199) | 869 | 822 | +47 |
| HC input, attention output, FFN input, shared FFN, fused posts | 2725 | 2685 | +40 |
| Q/KV | 819 | 929 | −110 |
| Total ordered intervals | ~7630 | 7433 | |

**Not pursued** (the gap is now under 3%):
- Fuse the indexer Q RoPE with its Hadamard transform (about 95 ms
  combined).
- Have the norm write the shared F16 copy itself (about 20 ms).
- Shorten the compressor F16 GEMMs (about 190 ms in cuBLAS).

**Conditions**
- Inputs: community artifact `cd39d504…`, fixed 8192-ID TSV `0cbafc4b…`.
- Settings: context 8192, native fast plan, F16 KV, frontier heads,
  compact experts, output-A and HCA.
- Source: main `27f610b` plus this worktree's uncommitted change, built
  incrementally in a same-length copy of the warm tree on Spark A.
- Harness: the private O/C/O collector, with a plan-time variant selector.
  Plans are bound before each paid pass. The model load is outside paid
  time.
- Runs: all completed rc0; the final preflight showed about 117 GiB free
  with clear probes.
- Records: raw heads, perplexity losses, the nsys trace and
  classifications stay under
  `spark:~/scratch/m3-ds4-prefill-stages-records/`. The kit is in
  `~/scratch/m3-ds4-prefill-stages/`.
- No unit suites were run (owner override, 2026-10-01).

## Production defaults

The fast plan now enables all nine mechanisms by default
(`SetDsv4PrefillStages` in `graph_plan.h` and `dsv4_graph.h`, set by
`engine/dsv4_plan.cc`). This covers the runtime, the runner harnesses and
`jitllm_dsv4_exec`. Exact and reference plans keep them off, and so do
named-tensor diagnostics. `jitllm_dsv4_exec --ds4-stages off --q2-d2r off`
gives the previous fast plan for A/B runs. Where D2R takes the Q2_K down
product, the pair writes the F32 activation (3a). Item 3b is therefore
selected only with `--q2-d2r off`.

Each mechanism keeps its guard, so what selects depends on the checkpoint
and the chunk size. Decode and verify chunks (under 64 rows) are unchanged.

| Item | Guard | Original 0731 (served) | Community IQ2_XXS |
| --- | --- | --- | --- |
| 1. D2R down | GB10, Q2_K down, 4,096-row chunks | No: IQ3_XXS/MXFP4 down | 4,096-row chunks |
| 2. F16 HC rows | 64+ rows, F16 HC mix weights | No: F32 mixes | Yes |
| 3a/3b. Pair write-back | GB10 IQ2_XXS occupancy-two pair, 4,096 rows | No: IQ2_XS | 4,096-row chunks (3a) |
| 4. Expert sum in the HC post | With 2 | No | Yes |
| 5. Shared F16 input | F16-weight products of one input, 64+ rows | No: Q8_0/F32 | Yes |
| 6. Output-A repack | With opt-in output-A (4,096 rows) | Not by default | Not by default |
| 7. Dense Q8_0 pairs | Two Q8_0 products of one input, 64+ rows | Q8_0 products sharing an input (41+ pairs) | Q-A/KV, shared up/gate (86) |
| 8. F16 Q | 64+ rows, fused Q-head | Yes | Yes |

Production DeepSeek prefill uses 2,048-row chunks. On the served original
checkpoint, only items 7 and 8 apply.

**2,048-row screen.** `jitllm_dsv4_exec` ran three processes, OFF/ON/OFF
(`--ds4-stages`/`--q2-d2r` off, then the defaults). Settings: the same
8,192 IDs, context 9,216, frontier heads and compact experts. Every head
was byte-identical across arms.

| Artifact | OFF / ON / OFF, s | Gain | Bookends |
| --- | --- | ---: | ---: |
| Community `cd39d504…` | 14.1973 / 13.2551 / 14.2241 | +7.21% | +0.19% |
| Original `8a355bfb…` | 14.3615 / 14.1461 / 14.3291 | +1.41% | −0.23% |

**Runtime, end to end.** The setup was `jitllm-runtime`'s chat route on the
original checkpoint, plain, with `context = 262144` and 2,048-row chunks. It
used the final-context harness (`longctx.py jitllm`) with the 8K retrieval
prompt (7,594 prompt tokens, none cached) and 64 outputs. Each run was a
fresh process after its warm-up request. Prefill is the first streamed
piece minus the send time.

| Run | Base | New | Base | New | Base |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill, s | 14.164 | 14.019 | 14.232 | 14.052 | 14.243 |

The new defaults are **+1.26%** prefill throughput (14.2129 s vs 14.0353 s
means), with +0.56% movement across the base runs. Decode is 21.5–21.6
tok/s in both. All five replies are identical, and each one finds all
three needles.

**Controls.**
- Runner, `jitllm_spec_runner --check frontier`: 8,192 IDs, 3 forced
  continuation rows, graphs off.
  - Original checkpoint with DSpark at 2,048 rows, plain and with
    injection: the target state, DSpark ring, first head and continuation
    logits of all 8 rows are byte-identical to the previous binary.
  - Community at 2,048 rows, using a private no-drafter build of the same
    harness: all 4 rows and 20 chunk heads are identical.
- Community at 4,096 rows with output-A/HCA: own repeats are exact. State
  and continuation are exact across all-head and frontier arms. The heads
  differ from the previous goldens because D2R applies: every value
  changes, with maximum 1.01 and RMS 0.10–0.19, but the argmaxes stay the
  same. D2R's quality evidence is the 32K perplexity above.
- 32K fixed history on the original checkpoint, with 2,048-row chunks,
  31,705 prompt IDs, 512 forced rows and bound 0.947: all 512 complete
  logit rows are byte-identical to the previous binary. 491 rows are equal
  and 21 are near ties (largest margin 0.725); none is outside the bound.
- Unit tests: `dsv4_stages_test` has 11 GPU tests, each checking a kernel
  byte for byte against the launches it replaces, plus selection, refusals
  and registry. A `dsv4_test` case checks model-level selection on both
  checkpoints' types.

**Provenance.** All runs were on Spark A, 2026-10-02. The previous binaries
are main `27f610b`; the previous runtime is `39bf4ff5…` and the measured new
one is `62377051…`. The shipped source differs from the measured one only
by later clang-format and clang-tidy fixes: whitespace, include order and
test-only changes. Raw records are under
`spark:~/scratch/m3-ds4-prod-defaults-records/` (`controls-r1`,
`runtime-r1`), and the kit is under `~/scratch/m3-ds4-prod-defaults/`.
