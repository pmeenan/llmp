<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek V4 on ds4 and jitLLM

M3's same-format study uses Entrpi's MIT-licensed ds4 fork at
`76d51ef82a81b70b78e51a3a6ea11946286de976` and
`antirez/deepseek-v4-gguf` at `f71f23d552d664e523b422157b2befbf74040380`.
The target is
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`,
86,720,111,488 bytes, SHA-256
`ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
jitLLM imports the same tensor bytes, including IQ2_XXS gate/up experts,
Q2_K down experts, and F16 APE tables, without changing their precision.

## Comparison fixed before the first native run

Both engines run on `spark`, a GB10 with CUDA 13.4.92 from SDK
`aarch64-e0a0c85c42806fb1`. Each model load requires at least 105 GiB
available and no existing GPU compute process. Builds and other model
runs do not overlap timing. Context rungs use fresh processes, plain
greedy decode, and the same raw input token IDs. The input is the user
text of the phase-1 long-context coding prompt, 527,518 bytes, SHA-256
`e94961ec63563e0e8b4b5eecdaad05f12608e7110d2156d92df453415299748d`.
ds4 tokenizes it to 128,817 tokens. Rungs take its first 8,192, 32,768,
65,536, and 128,817 IDs, with context capacity of the rung plus 1,024.
Prefill is the entire prompt from empty state; decode is reported
separately at the frontier. The earlier ds4 ladder measured incremental
frontier deltas and is excluded from the cold comparison.

The existing long-context correctness rules apply: force jitLLM on
ds4's generated token IDs and report every differing argmax with ds4's
margin between its token and jitLLM's choice. The existing DeepSeek
near-tie bound is 0.947 nats; a difference above it requires investigation.
Report full-logit maximum and RMS differences, and repeatability of
jitLLM's own forced run. Performance decisions use D-085's coarse 10%
threshold; memory is the sampled drop in `MemAvailable`, with about
1.1 times the reference as the bound. Perplexity evidence is required
before adopting a change in quantization or activation precision.

Profiling isolates a prefill chunk after startup repacking and boot
warm-up. Whole-run kernel totals are not prefill timings. ds4's
single-stream plain decode is the comparator here; its batched serving
throughput and DSpark creator reports describe different workloads.
The driver is 580.178.04. ds4 uses eight CPU threads and its pinned
`sm_121` build; jitLLM uses the locked `spark-native` build.

## Reproduction inputs

External raw results and instrumentation live in `spark:~/scratch/m3-ds4/`.
The pinned source is `ds4/`; the source model is under
`~/.local/share/jitllm/models/antirez/deepseek-v4-gguf@f71f23d5/`.
jitLLM's generic imported artifact is
`cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac`
in `~/.local/share/jitllm/m3-artifacts/`.
Its disk payload is 86,714,925,056 bytes and its extent handles total
101,160,321,024 bytes; those accounting values are distinct from a
measured peak working-set drop.
The extent-handle total is 14.28% above disk payload. The resident
measurement below uses the harness's separately packed expert slabs;
it does not establish that the runtime catalog materializes this
community artifact with the same peak.

The reproduction import uses the existing generic `import_m3.py build STORE PINS ID
MODEL` at SHA-256
`5886d444da46a8e50bd1aad19fdc267a9169646e6ce213a5fdfdc773454c3f0b`.
The external pins record the full repository revision, filename,
86,720,111,488-byte length and SHA-256 above. The ds4 baseline is built
with `make -j8 ds4-bench CUDA_HOME=SDK/cuda CUDA_ARCH=sm_121
NATIVE_CPU_FLAG=-march=armv8-a`; its executable SHA-256 is
`994d0e550426108cf2cb4738313a63ff5c0863753b823b079b4397ed644132ec`.

The unmodified ds4 baseline command is `ds4-bench --model MODEL --cuda
--threads 8 --prompt-file prompt.txt --ctx-start N --ctx-max N
--ctx-alloc N_PLUS_1024 --gen-tokens 128 --dump-frontier-logits-dir OUT
--csv OUT/result.csv`. Each rung starts a new process. ds4's own
512-token boot warm-up precedes timing; no prompt prefix is reused.

The DSpark support file is a block-drafter input for ds4's `--dspark`
server interface, not an MTP head for `ds4-bench --mtp`. The latter
refuses it with `required tensor is missing: mtp.0.hc_head_base.weight`.
This study does not treat that failure as a decode measurement.
The support file is 5,989,114,272 bytes, SHA-256
`7e319924541db3f7a163ed7e11d7532a70d48228ab59d36cb81e1d4511885360`,
at the same HF revision.

The native command is `jitllm_dsv4_exec --artifact ARTIFACT --prompts
prompt-N.tsv --generate 128 --context N_PLUS_1024 --max-rows 4096 --out
OUT`. The TSV contains `ds4`, a tab, and the space-separated raw IDs.
Its SHA-256 at 8,192 / 32,768 / 65,536 / 128,817 IDs is, respectively:

- `c1d7137841d7d594254e5540806b645e8fb7e6b741f8bcee77776bff67256e29`
- `45e76a03b053d067dd1d9ca691e401d62978649eb9321f08bb1516420f21a7d3`
- `458f2673107cf745766d481f9aaefdf90e21b5f2830b06256ac0da1673ed83ab`
- `f73c7066de4f5c53452d44cffd1c0bbcb3039f50e37a218bfe652e44ab405492`

The raw text is the phase-1 long-context `deepseek/128k` coding fixture;
its construction is pinned by [corpus.json](../long-context/corpus.json)
and [build_prompts.py](../long-context/build_prompts.py), using llama.cpp
`8019dc563b1ecbae6b161a70c3a1359f1b206c1e` and the pinned DeepSeek
counting tokenizer, `--model deepseek --rung 128k=131072`. The external
`prompt.txt` and exact TSVs above are retained with the study. For an
arbitrary supplied text, ds4's `--dump-frontier-logits-dir` captures its
tokenized input for constructing the same native TSV. A different text
does not reproduce these numbers. No chat template is applied.

## Cold-context results

Both arms request 128 outputs. Native decode times cover the 127 steps
after the prefill frontier; ds4's steady rate excludes its first output.
Each row is one fresh process per engine, rather than a statistical
distribution. The native arm includes both retained transfers below.

| Prompt tokens | ds4 prefill tok/s | Native prefill tok/s | ds4 steady decode tok/s | Native decode tok/s | Native peak GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 1,064.32 | 625.5 | 18.83 | 18.46 | 87.91 |
| 32,768 | 1,026.31 | 617.4 | 18.07 | 18.19 | 87.99 |
| 65,536 | 1,020.14 | 602.1 | 17.18 | 18.01 | 88.30 |
| 128,817 | 965.66 | 572.2 | 15.65 | 17.69 | 88.78 |

The ds4 128,817-token process's sampled peak is 97.19 GiB; native is
0.914 times that drop. The artifact's extent padding is not the runtime's
slab padding: the harness's slabs add only 40,157,184 bytes.

Cache policies differ. ds4's defaults use FP8 compressed KV and FP4
indexer caches; native uses F16. An additional 8K cold ds4 run with
`DS4_CUDA_FP8_KV=0 DS4_CUDA_FP4_INDEX=0` uses F32 caches and gives
1,054.22 prefill and 17.55 steady decode tok/s, with a 95.26 GiB peak.
Native's F16 prefill is 0.593 times that F32 control, leaving a 40.7%
throughput deficit; the prefill gap remains without ds4's compressed
caches. This is a higher-precision reference control, not a matching
F16-cache comparison. ds4 exposes no F16 mode at this pin.

An interleaved default-cache 8K repeat measured 952.82 prefill and 18.77
steady decode tok/s (94.68 GiB peak), versus 1,064.32 in the ladder.
The 10.5% prefill difference is material under D-085; the cold ladder is
not precise enough to rank changes near that scale. Isolated paired
microbenchmarks and native controls decide the retained transfers.

## Isolated profile and transfers

The first 4,096 prompt tokens, from empty state, give the matched prefill
profile below. ds4's capture starts after startup repacking and boot warm-up.
Native totals stop at the first chunk's logit readback. The native harness
computes the vocabulary head and copies logits for every prefill row;
ds4 computes only the frontier's head. These are harness timings, not
HTTP serving throughput.
The instrumented ds4 capture adds profiler start/stop around prefill and
exports its exact input IDs and 32 frontier logits; its executable
SHA-256 is
`9db6abcfd17a0fa67e515fd8f60c3dab8a151fd6af9d2ba2f800341b208ba117`.
Its scratch source `capture_bench.c` has SHA-256
`c9f553686be5e7c459ba673212bfbd70500918d352dc755ff663042e7cd01de9`.
The ds4 capture command uses `nsys profile --trace=cuda
--capture-range=cudaProfilerApi --capture-range-end=stop --sample=none
--cpuctxsw=none --export=sqlite`, `--ctx-start 4096 --ctx-max 4096
--ctx-alloc 5120 --gen-tokens 0`. Native's full 8K trace uses the same
nsys options without a capture range and `--generate 3`; the first
whole-row vocabulary copy after the first kernel delimits the first 4K
chunk. Repack, boot and decode kernels are excluded from the table.

| First 4K chunk | ds4 | jitLLM before transfer |
| --- | ---: | ---: |
| GPU kernel busy | 4.295 s | 8.140 s |
| Kernel span | 4.532 s | 8.221 s |
| IQ2 gate/up | 1.085 s, fused with SwiGLU and input Q8 | 1.279 s, two products |
| Q2 down | 0.452 s | 0.961 s |
| Attention | 0.431 s, token tile | 2.320 s, one query at a time |
| Dense Q8 D2R | 0.509 s | included in 0.932 s Q8 MMQ |

After both transfers, the matched native chunk has 5,087 kernels,
6.209 s GPU busy and a 6.290 s span. Attention is 0.433 s; MMQ is
3.174 s (51.1% of GPU time), including IQ2_XXS 1.287 s, Q2_K 0.956 s
and Q8_0 0.932 s. Input Q8 preparation is 0.333 s (431 launches,
previously 474) and expert-map preparation 0.029 s (86 launches,
previously 129). The Q2 down-product gap alone is 0.505 s per chunk.
ds4's fused IQ2 product also includes SwiGLU and Q8 preparation, work
still separate in native. Q8 totals are not perfectly matched buckets:
ds4's 0.463 s attention-output fusion includes additional products.
These observations motivate the next bounded expert-product experiment.

The retained sparse-attention change shares the union of up to eight
queries' gathered KV cells. It applies each original per-query mask,
including its finite bias, to that union; KV remains F16. It backports
GGML PRs 28770/29298's union/live-count machinery, extending the pinned
D512 eight-query case while retaining its MMA configuration and swizzle.
It also supports D256 sparse reference graphs. Qwen3.8's default fast QSA
operation is independent.
The generic planner and launch default retain D512's one-query sparse
and D256's dense choices. `wide_sparse_attention` explicitly selects
the distinct wide implementation; only measured DeepSeek fast planners
enable it. The reference mode retains its original choice.

On the 8K prompt, native prefill changed from 16.375 s (500.27 tok/s)
to 13.204 s (620.42 tok/s); plain decode stayed 18.46/18.44 tok/s. The
first attempted compact expert-tile experiment changed 16.375 to
16.330 s, but a source audit found its launch hook behind an undefined
macro. That optimization was inactive: the result does not assess
compact scheduling. Its forced logits were exact to the original plan;
the saved binary is used only as the original one-query attention control
below. Corrected expert scheduling is a separate product experiment.

The paired expert path shares the routed-row maps and quantized input
between adjacent gate/up products, retaining both ordinary MMQ products
and their original weight/output strides. It is an explicit fast-plan
choice; the reference and default primitive planner remain unpaired.
Native 8K prefill changes from 13.204 to 13.0959 s (about 0.8%). Its
32 forced frontier outputs are bit identical to separate preparation,
and its own fresh repeat is bit identical. This small end-to-end effect
is consistent with input preparation being a fraction of the product
work; it does not close the product-kernel gap.

ds4's full fused D2R product is not a direct transfer: its dense SoA
weights differ from raw GGUF blocks at jitLLM's expert strides, and its
fused epilogue weights before quantizing the activation and before down.
jitLLM retains post-down weighting. Keeping an extra SoA replica would
also invalidate the memory comparison. No cache precision change,
early weighting or nonfinite-value sanitization is adopted.

## Numerical controls for the attention change

Own comparison precedes the oracle assessment: original one-query to
wide attention changes full logits by maximum 7.074 and RMS 0.5077.
Its top-one/top-two margin movement is p99 2.141 nats, maximum 2.710;
this is not a common row offset. The wide path's fresh forced repeats
are bit identical. All 32 native greedy choices on ds4's 8K trajectory
agree with ds4; there are no exceptions to judge against the existing
0.947-nat oracle bound. That bound is unchanged. This short trajectory
does not establish universal greedy agreement.

Fresh tokenization of the existing War and Peace perplexity text by the
community GGUF supplies 32,768 IDs. Both native variants score the same
32,767 transitions with 4,096-row chunks and capacity 33,792:

| Own variant | PPL, all transitions | PPL, second half |
| --- | ---: | ---: |
| Original one-query attention | 2.704342 | 2.987167 |
| Wide sparse attention | 2.698530 | 2.984785 |

The relative changes are -0.215% and -0.0797%, inside the registered
3% tolerance. This is an own-plan quality control, not ds4 perplexity.
ds4 defaults to FP8 compressed KV and FP4 indexer cache; jitLLM uses
F16 caches. ds4 exposes F32 controls with `DS4_CUDA_FP8_KV=0` and
`DS4_CUDA_FP4_INDEX=0`, but no matching F16 mode. Same weights alone
do not make these precision policies equivalent.

The ds4 F32-cache forced 8K trajectory also gives all 32 native argmaxes
equal to ds4. The community checkpoint's PPL TSV is freshly tokenized
from the existing War and Peace text (text SHA-256
`c7156148ecaa12b6416cf816540d8dede2014982554a835e61076f0dd8bf0c2d`,
TSV SHA-256
`755b00b9ac2eeab723d2ce3b80d703149df1e085960c18c21ea7068e9e645ad7`).
No ds4 PPL result is inferred from the native own-plan control.

The original UD-Q2_K_XL checkpoint is also affected by the shared
attention and paired IQ2_XS preparation. On the existing 31,705-token
coding fixture, old/new prefill is 59.118 / 50.990 s; both use 4,096-row
chunks, capacity 33,280 and the first 32 recorded llama.cpp outputs as
forced inputs. The new path's fresh repeat is bit identical. Against
llama.cpp's recorded trajectory, 31 of 32 argmaxes agree; step 12 is a
near tie with oracle margin 0.2694, below the unchanged 0.947 bound.
On its existing 32K PPL IDs, old/new second-half PPL is 1.852794 / 1.857934;
the new value is +0.277% against the old path and +0.277% against
llama.cpp's 1.8528, within the existing 3% gate. These controls retain
F16 caches in both native arms.

Both PPL tables score indices 16,384 through 32,766 in the second half
(16,383 transitions), as the long-context judge does. Earlier scratch
aggregates included index 16,383 and are superseded by these values.

## Cross-model and format applicability

`jitllm_prefill_transfer_bench` measures nine alternating samples per
arm after two warmups, one invocation per sample. A separate 512 MiB
write before each timing displaces operands from L2 and is excluded
from the CUDA event interval. These are medians on finite synthetic
operands, not model throughput. Operation tests provide the FP64 and
exact separate-versus-paired correctness controls.

Attention uses 256 query rows, 8,192 F16 KV cells and 256 allowed cells
per row. Overlapping lists share neighboring cells; disjoint lists in
each eight-query tile share none. D256 uses 24 heads / 2 KV heads;
D512 uses 64 / 1. The primitive arm is the prior D256 dense or D512
one-query sparse kernel; the shared arm explicitly enables wide sparse.

| Shape | Primitive ms | Shared ms | Speedup |
| --- | ---: | ---: | ---: |
| D256, overlapping | 0.870144 | 0.190176 | 4.58× |
| D256, disjoint | 0.862752 | 0.392000 | 2.20× |
| D512, overlapping | 0.961152 | 0.641632 | 1.50× |
| D512, disjoint | 0.972544 | 1.747648 | 0.556× |

The disjoint D512 union causes a material regression. This is why the
choice defaults off and is enabled only for measured DeepSeek fast
sparse shapes, whose windows share cells. D256 capability is reusable
but is not enabled for another model on the strength of a synthetic
test alone. Randomized masks, overlapping/disjoint lists, finite mask
biases, empty tail rows, partial query tiles and masked union/dummy cells
are tested against FP64; both head sizes have VMM over-read probes.

Expert controls use K=4,096, M=2,048, 64 experts, 512 tokens and four
uniformly routed experts per token, with broadcast F32 input. Two
ordinary products are compared with one shared preparation step; each
weight tensor retains its own bytes and stride.

| Type | Separate ms | Paired ms | Speedup |
| --- | ---: | ---: | ---: |
| IQ2_XXS | 3.690208 | 3.812032 | 0.968× |
| IQ2_XS | 5.100224 | 5.033920 | 1.013× |
| Q2_K | 5.977824 | 6.121408 | 0.977× |
| Q4_K | 4.940512 | 4.940544 | 1.000× |
| Q8_0 | 7.700192 | 7.612128 | 1.012× |

No synthetic product gain is decisive under D-085. Preparation sharing
does preserve both products exactly for all eight admitted non-FP4
types, including padded and different expert strides, broadcast and
per-slot activations, skewed routes, empty experts and tail tiles. Its
small measured model gain is retained; unknown planners default off.

Qwen3.8's fast NVFP4 expert path already fuses gate/up, SwiGLU and input
quantization under a different per-16-value scale contract; ordinary
GGML MMQ sharing cannot replace it. Reusing a selected expert's packed
weights across verify rows is the relevant separate transfer. Its fast
QSA attention already packs 12 query heads per KV head into a 16-row
tile; merging tokens needs a new device-selected cell union and
per-token masks. The wide dense-mask implementation is not a direct
replacement. The Qwen reference D256 capability stays available but
keeps its prior default choice.

EXL3's current dense fixture has no expert-row mapping or GGML Q8 input
preparation to share; its packed weights and reconstruction path have
their own contract. Qwen-Image uses dense D128 BF16 FlashAttention-2,
already tiled, without sparse cell lists. Neither is silently switched
to these GGML sparse or quantized products. The transferable rule is
to share operands and preparation when the original layout/arithmetic
contract permits it, then measure the affected shape.

DeepSeek production returns only the frontier logit row, while its graph
computes every row's head. A future explicit frontier-head choice can
gather the last hidden row before output normalization while retaining
all hidden rows for DSpark injection; PPL and verify must keep all heads.
The current head costs about 65 ms per 4K chunk, about 1% of GPU time;
eliminating full-row harness readback is a distinct measurement change.
Qwen's fast production path already gathers its requested frontier rows
before its head and preserves all injection streams.

The final fast path's 8K forced 32-output capture is bit identical to the
paired capture above after adding the explicit dispatch gate and keeping
the one-column kernel's original fixed loop bound. This verifies that
the preservation guards do not change the measured fast trajectory.
The checked native harness binary has SHA-256
`6a7b22e833a2437dc8071404f4f88d26c026adfb491dbc4d5d8858154dfc6863`;
the synthetic transfer harness has SHA-256
`ed5a5de7da7993b0c3abe2a86ecf7c74055b552b1e3dcacc979cc5506f549646`.
The archived pre-wide native control has SHA-256
`d2dbb1ccabe5525ef72829109b95a4adce3e72ad8e2eef9cbb96fc7a1090e1d9`.
The full result directories and captures are retained under the external
scratch location above; the report's aggregate tables do not require
loading those raw bundles.
