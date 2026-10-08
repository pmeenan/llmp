<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 26 matched scalar screen

On one GB10 Spark, the native ordinary runner completes the short warm
comparison about **4.33% below** the fastest pinned same-format reference's
throughput. All 32 greedy IDs agree, but the full probability distributions
differ substantially from the fusion-enabled reference. Neither speed nor
quality qualification passes from this screen. Optional norm, shared-Q8 and
RoPE/store policies remain unselected for production; representative perplexity, 8K/depth,
optimized batching and comparable physical peak measurements remain required.

## Inputs and comparator

The measured native base is **bd8e54f**, with the benchmark changes described
below. Later store/serving commits are not measured by these results. The
prepared artifact is
`4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3`, imported
from Unsloth revision `c099eb48e663fd284577b04978a94ffccb261841`:
`gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`, 16,947,541,728 bytes, SHA-256
`f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f`.
The same raw checkpoint supplies the reference weights.

The reference is llama.cpp **b29c606e28a01b1bc8c1351026a0fa6e616bf6c4**, b10964,
image
`ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`.
CUDA fusion and graphs are enabled for competitive timing. Fusion-disabled
runs below are numerical diagnostics. The task-entry TensorFold observation
was HEAD **609ca419abecebdc5a059498a613680bd3aa847f**, version **0.6.5**;
its pinned README/package still lists Gemma 26 under MLX only, supplying no
CUDA GB10 comparator for this batch. A subsequent task refreshes that
observation rather than inheriting it.

All runs use the same physical host `spark`/`spark-c4e2`, driver 580.178.04,
context 4,096, F16 K/V, FlashAttention, sequence 0, maximum prefill chunk 128,
no drafts, and the literal UTF-8 `The capital of France is` without a newline
or template. Explicit input IDs are **[2, 818, 5279, 529, 7001, 563]**, including
BOS once. There is no EOG stopping. Prefill publishes only its frontier head;
every published head has the full 262,144-entry F32 vocabulary.

Both warm arms execute the literal prefill and eight untimed scalar chunks,
clear state, repeat the literal prefill, and execute three untimed scalar
chunks before timing. Those three chunks append **45518, 107, 101**, so timing
starts at the same nine-token prefix. This pays the reference's reset-induced
GGML graph replacement and its CUDA build/capture/replay transition before
the timer. The timed loop pays 32 completed chunks, CPU argmax, fresh input
staging, execution/completion and full-vocabulary host-vector publication.
File output and state snapshots are outside the warm timer. The reference
public getter synchronizes; an extra preceding synchronize is omitted.

## Paid scalar result

| Arm | Seconds for 32 completed chunks | Chunks/s |
| --- | ---: | ---: |
| Reference A | 0.602562 | 53.1066 |
| Native ordinary B | 0.630152 | 50.7814 |
| Reference A after | 0.603176 | 53.0525 |

All timed IDs and the three untimed seeds agree exactly. Reference
`ggml_graph_reused_delta` is 32 in each arm; this is a GGML reuse counter,
not proof of CUDA replay count. Native totals across setup/timing are two
captures and 41 replays. No optional norm, shared-Q8, store or lane step is
selected. GPU clocks end at 2,411 MHz/P0 in all arms. Clocks were not locked;
the first arm started idle, before the common warm sequence. The short
bookends do not resolve a long-context or production throughput gate.

The 50 ms observer's whole-process sampled MemAvailable drops are
18.3770/17.5921/18.3517 GiB for A/B/A. They include reclaimable page cache and
unrelated system activity and may miss a peak. These are **preliminary system
observations**, not comparable physical device peaks or a 1.1x memory pass.
Native reported envelope bytes are 17,047,748,608 weights, 337,641,472
activations, 10,485,760 scratch, 2,097,152 host inputs and 2,187,792 plans;
slab padding is 7,929,856 bytes, including 7,667,712 pitch padding. Envelope
and catalog figures are distinct from physical peak occupancy.

## Numerical controls and rejected axes

The quality fixture records its frontier head plus 31 scalar heads, each
on the same prefix. Native ordinary versus independently norm-fused paths
are byte exact on all 32 full heads. Their top-two margin movement p99 is
zero, frozen before cross-engine analysis in calibration SHA-256
`40a0ff4c005df88e0ca6ec90929ccd639a869c47bc7feaed52c7de6818aae8d5`.
References were acquired before these native runs; the analysis reads and
freezes only native data before reading the oracle. This tiny fixture is not
a representative numerical allowance. The fusion-enabled reference's own
repeat is also byte exact on all 32 heads.

| Comparison | Maximum raw logit delta | Maximum chosen-token NLL delta | Maximum full softmax TV |
| --- | ---: | ---: | ---: |
| Native ordinary vs reference fusion on | 5.737098 | 0.950814 | 0.490946 |
| Native ordinary vs reference fusion off, first head | 0.040565 | 0.007264 | 0.003253 |
| Native ordinary vs reference fusion off, later 31 heads | 0, byte exact | 0 | 0 |
| Reference on vs off, first head | 5.746083 | 0.958078 | 0.492266 |
| Reference on vs off, later 31 heads | 3.937213 | 0.367878 | 0.225012 |
| Reference on frontier vs all-prefill heads, first head | 0.00002480 | 0.0000000372 | 0.0000001181 |

The all-prefill-head difference is zero on later scalar heads. Every strict
greedy ID agrees, so each compared head still has the same input prefix.
Large distribution differences remain failures to establish the intended
quality equivalence, despite this greedy agreement.

Separate native ordinary/norm/ordinary/Q8/ordinary warm times are
0.632245/0.628584/0.627302/0.612714/0.631675 seconds. Norm lies within bookend
movement and remains off. Its 32 full heads and initialized state are byte
exact to ordinary. Shared Q8 reduces this short latency by about 2.67%
relative to adjacent ordinary bookends, but changes all 32 heads:
maximum raw/NLL/TV deltas are 6.399552/0.625702/0.496582. All greedy IDs still
agree. At completed past 37, 5,367,357 of 57,671,680 initialized state bytes
change. Actual Q8 scalar dispatch counts are 266 VecQ and 30 row-product
fallbacks; no norm/store/lane policy is selected. This is not an accepted
quality or speed policy.

A funded ordinary prefix-six checkpoint restored into an empty Q8 runner
isolates decode from prefill. After feeding verified gold ID 45518, none of
31 aligned decode heads is byte exact to ordinary; maximum raw/NLL/TV deltas
are 3.958571/0.194219/0.129339. Greedy IDs agree and 4,142,309 initialized state
bytes change. Thus prefill is not the sole source of the Q8 difference.
Repeated ordinary heads still match the original frozen hashes. Q8 also does
not match the fast reference: their first-head raw/NLL/TV differences are
3.378479/0.325112/0.282851; later maxima are 5.933873/0.088797/0.258824.

## Bounded reference attribution

Broad and detailed callback traces change the fusion-enabled final heads and
are retained as instrumentation limitations. They cannot attribute the
original full-model drift. End-layer-only taps, separate routed boundaries,
and the combined layer-zero expert-input/GeGLU/top-eight taps preserve all
32 final heads byte for byte in **both** reference policies.

The first changed endpoint is layer zero. In its six prompt rows, all eight
selected expert IDs are identical between policies. Its learned expert input
changes by at most 4.768372e-7 (4,587 of 16,896 floats); routed GeGLU changes
by 0.001756132 (5,561 of 33,792), and scaled expert down output by 0.008260742
(22,527 of 135,168). The top-eight view has a 512-byte row stride; comparison
uses that stride rather than treating its raw storage as 48 packed integers.

[llama_routed.cc](llama_routed.cc) uploads the identical captured input and
expert IDs with the actual layer-zero Q4_K merged gate/up weights, Q5_1 down
weights and F32 expert scales. Four isolated arms cross the two input
variants with fusion on/off. For either fixed input, GeGLU and scaled down
outputs are **byte exact between fusion policies**. Each input variant also
reproduces its corresponding valid full-model tap byte for byte. Changing
only the input exactly reproduces the observed GeGLU and down differences.
This attributes those first-layer differences to the incoming precision
change; it does not attribute all later full-model drift or prove an
upstream defect.

Pinned source predicates agree with this control: the actual merged gate/up
path is one 1,408-row `MUL_MAT_ID`, two views and split GeGLU. The two-product
MMVQ gate/up-writer fusion is ineligible for that graph. Q4_K six-row routed
products choose MMVQ before MMQ, as do single-row products. GEGLU enum
switches use the intended GELU-tanh arithmetic. There is no evidence here
of a wrong SwiGLU enum, dropped Q5_1 scale or forced MMQ-to-MMVQ switch.

## Reproduction and identities

[llama_performance.cc](llama_performance.cc), [llama_trace.cc](llama_trace.cc)
and [llama_routed.cc](llama_routed.cc) are external pinned-reference C API
harnesses, not linked into llmpalooza. Compile them in the pinned image using
headers from the exact commit, `g++ -std=c++23 -O2 -march=armv8-a
-Wall -Wextra -Werror -I<headers>`, `/app` library/rpath, and
`-lllama -lggml -lggml-base` for full-model harnesses. The isolated harness
uses `-lggml-cuda -lggml -lggml-base`; NVIDIA CDI driver libraries are present
during its compile and run. Runs are network-disabled, with a read-only
container root, read-only source/model mounts, a writable external result
mount and `CUDA_DISABLE_PTX_JIT=1`.

Run `llama_performance MODEL OUTDIR warm|quality|quality-all|quality-unfused`;
only `quality-unfused` sets `GGML_CUDA_DISABLE_FUSION=1`. Native syntax is
`llmp_gemma_runner ARTIFACT OUTDIR 2,818,5279,529,7001,563 32 1
ordinary|norm|q8 warm|control device`. For restored-prefix isolation, use
31 steps and append the ordinary control's `initialized-prefix.bin`; the
first saved head then aligns with ordinary row 1. Checkpoint buffers are
funded pinned memory and released only after proven completion.

Trace syntax is `llama_trace MODEL OUTDIR quality|quality-unfused PROFILE`.
Use `layers`, `router`, `geglu`, `down` or `routed` for the valid controls;
`broad`/`first` are the retained intrusive diagnostics. `taps.tsv` records
type, shape, byte strides and raw storage length. The isolated command is
`llama_routed MODEL TAPDIR OUTDIR fused|unfused`, where TAPDIR is the valid
`routed` trace. It validates the exact approved GGUF length and selected
weight contracts, reads the actual weights, and requires a new OUTDIR.
The output shapes are `[704,8,6]` GeGLU and `[2816,8,6]` scaled down, F32.

Wrap each arm in [measure.py](measure.py) for the preliminary memory observer.
[compare.py](compare.py) freezes native calibration (`ROOT calibrate`), checks
the oracle (`ROOT oracle`), and compares further aligned pairs (`ROOT pair
LEFT RIGHT ROWS [LEFT_OFFSET RIGHT_OFFSET]`). Its pair mode requires callers
to establish identical initial prefixes; it stops numerical comparison after
a greedy divergence. Run full-logit scans under `hostlock shared` locally.

Measured source/binary SHA-256 pairs:

| Measurement | Source SHA-256 | Binary SHA-256 |
| --- | --- | --- |
| Initial native quality/warm | e6afcc6e765cd8df09a3a9222b8bb30b53e74e2720434f66deadd115157f8edb | e8815b7c32093c649d77502fb6f44a30a42a118238c5a9a1245ee0b30ba8b0c8 |
| Initial reference quality | 29f7cdc423e1d49aa862dae8a7a6e464fe8e0de16c54bfdc84f1f9f0f539664c | e6240f4a12fea98e62107afd45a28bd586222703c4488a7c21f56a1f54555961 |
| Reference competitive warm/factor controls | 23beb6315ebcadef631383981b4e6b3ccd4354fc27e01074436723d7df40676f | 7ef824b71dbef8ee120c59a3a5366a49c536abeb8efb7797c1955a821abe5045 |
| Native optional axes/state capture | 1b60dd5bb89f343cf1a479c04d320eaf5c9bbe5a00cb63556c3c0512b1190d64 | f3e9754585f77f582d967db94d1319c8c1102cc6314702fd5458b230c98dfba2 |
| Native restored-prefix control | e96eb890f9d24194545f01e2e3dd36b72293ae9801a2fc2b96c737fc1595e848 | e6a830d694f5630454a4164addbb7a4f3b3e5e1ef0643633e96cc09e6443188b |
| Final valid routed traces | d9c2d4f0990c179d8a31ff56ce68a664a9a3ac582445beead472d5a90c42868d | 3cdd6485f2fbc501917388ecb3eeccc5385c5d4b4a5095a6c2fe4fbccb3728a4 |
| Identical-input routed chain | 40e0fde08699807560c95388bb22dd7f8a2dce1dc3e16acddd9b2917b6bd0bf0 | aa5c40566e559d7fc714528148ff225c1a14380254280728fa38b0ee0de868b3 |

The checked-in reference harnesses received formatting after these runs.
A supervised rebuild produces the same three reference binary hashes byte
for byte; formatting therefore changes no measured executable. The table
still identifies the measured source hashes explicitly. Raw vectors, state bytes, samples and logs remain external
under `~/.local/share/llmp/gemma-performance/` on Spark. Supervised
`m35-gemma-performance-{quality1,warm1,native-axes1,native-prefix1,
reference-diag1,reference-layers1,reference-isolate1,reference-routed1,
reference-identical2}` completed successfully; each had a 600-second limit.
