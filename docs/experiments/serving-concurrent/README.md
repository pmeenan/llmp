<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Common-prompt Qwen concurrent serving screen

One short condition on Spark B (`spark-56f5`), 2026-10-02. Each measured
request has 8,256 actual rendered prompt tokens, zero cached prompt tokens,
and 256 actual generated tokens ending in `length`. C1 uses u0, C2 u2/u3,
and C4 u0/u1/u2/u3 from the same frozen four-request set. An unrelated
one-token weight prime precedes each cell and is excluded. Clock is buffered
HTTP submission through final complete response, including prompt processing,
generation and queueing; these are not pure-decode rates or TTFT measurements.

| Engine | C1 completed tok/s | C2 completed tok/s | C4 completed tok/s |
| --- | ---: | ---: | ---: |
| Llmpalooza selected normal runtime | 25.58882 | 26.92553 | 27.23548 |
| TensorFold affine 4-bit | 28.43261 | 34.81075 | 41.58761 |
| Mia original unpatched recipe (legacy) | 20.31579 | 30.04613 | 39.50226 |
| Current TensorFold 0.6.2 NVFP4 | 21.46301 | 24.78645 | 31.87218 |

| Engine/cell | First completion, s | All completed, s | Per-request latency, s, input order |
| --- | ---: | ---: | --- |
| Llmpalooza C1 | 10.00437 | 10.00437 | 10.00437 |
| Llmpalooza C2 | 18.56155 | 19.01541 | 19.01513 / 18.56155 |
| Llmpalooza C4 | 18.12259 | 37.59801 | 18.12197 / 37.36712 / 22.64620 / 37.59801 |
| TensorFold C1 | 9.00375 | 9.00375 | 9.00375 |
| TensorFold C2 | 14.47641 | 14.70810 | 14.70810 / 14.47626 |
| TensorFold C4 | 24.35684 | 24.62272 | 24.47179 / 24.41873 / 24.62251 / 24.35684 |
| Mia C1 | 12.60104 | 12.60104 | 12.60104 |
| Mia C2 | 16.90768 | 17.04047 | 17.04013 / 16.90768 |
| Mia C4 | 24.63979 | 25.92256 | 24.93997 / 24.63964 / 25.92256 / 25.42472 |
| Current TensorFold C1 | 11.92750 | 11.92750 | 11.92750 |
| Current TensorFold C2 | 20.22146 | 20.65644 | 20.22131 / 20.65644 |
| Current TensorFold C4 | 30.14570 | 32.12833 | 30.14524 / 30.75360 / 31.04212 / 32.12833 |

Every engine/cell retains complete raw/parsed responses and actual usage.
Native repeated u0/u2/u3 across solo/two-request and four-request cells have
equal API text, reasoning and usage. Native generation IDs are not exposed
by this API; no generated-ID equality claim is inferred from text. TensorFold
and Mia return full actual generated IDs. Cross-engine generation trajectories are
retained descriptively and do not establish quality or equivalent precision.
Mia's own u3 generated IDs differ between C2 and C4 from token index2;
this API equivalence is not asserted, and the complete differing replies remain.
Current TensorFold's own u0/u2/u3 repeats preserve the complete generated-ID
arrays, text, reasoning, usage and finish across C1/C2 and C4. All seven current
TensorFold outputs contain 256 actual IDs; this is independent of the native
API's unavailable generated IDs.

## Input agreement and residual engine differences

The earlier native v1 fixture was 8,266 rendered tokens. Five literal role
markers embedded inside its user content were encoded as ordinary content
by native `EncodeMarked` but recognized as control tokens by the HF tokenizer
(8,243 rendered tokens). It is preserved as a diagnostic, excluded from this
matched-input table. V2 changes only those three start/two end markers into
ordinary text delimiters: 8,205 user IDs and 8,256 complete rendered IDs.
All four native actual renderer/tokenizer vectors match the original HF and
affine-model vectors byte for byte, including the complete rendered text.
No checkpoint or chat-template change was made.

Native uses NVFP4 weights/MXFP8 products, F16 KV/F32 recurrent state,
47,172 curated draft entries, adaptive depth 2/3, chunk 4,096, context
33,792, four retained branches but only two funded simultaneous requests.
TensorFold uses its pinned MLX affine 4-bit/group-32 checkpoint, BF16 KV,
MTP cap 6/confidence 0.30, context 33,792 and parallel 4. Each native and
TensorFold cell runs in a fresh service. TensorFold receives the exact pinned
template's xhigh thinking option. These format/policy differences prevent
calling this an identical-precision/kernel comparison. Original Mia uses the
same NVFP4 checkpoint as native, FP8 KV/BF16 recurrent and draft IO,
curated47,172/fixed3, context33,792/chunk2,048/max-sequences4. Its single
service handles C1/C2/C4 with distinct cache salts; every actual measured
request reports zero cached prompt tokens. The initial startup is804.513 s
with the original safetensors-lazy loader. Its exact result is retained as a
legacy reference; future Mia launches use the owner's measured instanttensor
startup fix. The current same-checkpoint TensorFold release below is the new
reference; the legacy engines and their provenance remain unchanged.

Current TensorFold is the unmodified v0.6.2 release at `56e2e3ec`, using the
same `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6` weight checkpoint as native.
Its installed renderer/tokenizer reproduces all four complete frozen rendered
ID vectors exactly. It uses BF16 KV, MTP cap 6/confidence 0.30, context 33,792,
parallel 4 and the source-resolved default 79,591-entry draft vocabulary;
the service log does not print the head size. Its draft-only BF16 experts and
head are requantized to 4-bit by its loader. The startup log records eager
execution, zero captured decode graphs and 2,048-row idle prompt pieces.

Precision correction: the frozen controller/protocol's nominal checkpoint
activation label is incorrect for this Flash Next family. Although the launch
passes the released global `--precision checkpoint`, its family loader does
not consume that precision mode, and its NVFP4/MXFP8 products take BF16
activations. The actual service log says `prompts: bf16 activations`; the
unmodified source confirms this path. The raw receipt is preserved with its
original nominal label. This is a same-weight-checkpoint reference, with
different activation/state arithmetic, draft weights/head and policy; no
equal-math or cross-engine quality conclusion follows.

TensorFold fresh startup is 137.024 / 137.098 / 134.981 seconds. Native
sampled minimum node MemAvailable is 40.160 / 39.093 / 38.954 GB;
TensorFold is 34.947 / 34.729 / 34.743 GB. Native sampled maximum process
RSS is 1.828 / 2.168 / 2.311 GB and TensorFold 34.749 / 35.849 / 36.861 GB.
MemAvailable includes unified allocation, system and file-cache occupancy;
RSS is not total device/unified allocation. Each cell is fully retired before
the next load for native/affine; Mia is retired after all three cells and the
short warm probe below. No cold-page-cache equivalence is asserted.

Current TensorFold startup is 198.870 / 73.367 / 73.431 seconds. The first
service includes 121.8 seconds of prompt-kernel warming; the persistent new
extension cache is retained across its fresh services. Current TensorFold's
sampled node MemAvailable (minimum / mean) is 34.730 / 52.920 GB in C1,
38.600 / 68.828 GB in C2 and 38.781 / 66.087 GB in C4. Sampled process RSS
(maximum / mean) is 4.877 / 4.001 GB, 5.594 / 4.210 GB and 6.277 / 4.338 GB.
These are 844 / 376 / 422 samples across loading and requests; RSS exists
only after the service PID is known. They are not allocated-memory totals or
steady-state averages. Each current service and its launcher retires before
the next load; all three retire successfully.

## Warm legacy Mia anchor probe: collection only

One frozen31743-token literal prefix and its saved512-token native fixed3
continuation are retained. Five serial9-token probes use true offsets0–3,
plus offset0 repeated; the first generated token is the native uncached
anchor and the next eight form the window. Actual literal prompt IDs match,
all five first anchors match, and all five produce nine actual completed tokens.
Cold offset0 reports0cached and matches only the first5 IDs; its own warm
repeat reports31616cached and matches all9. The other offsets also report
31616cached and match all9. Retain the cold/warm trajectory difference.
Actual accepted/proposed counts are6/6(cold0),5/9(repeat0),6/9(offset1),
5/9(offset2),5/12(offset3), spanning2/3/3/3/4 verifies respectively.
These are whole public-window deltas, not one3-draft native verifier's
acceptance statistic. No matched native8-token collector,200-anchor expansion,
draft-quality inference or policy adoption has occurred. The five requests
take20.219 s, including17.583 s cold0; warm windows take0.437–0.961 s.

## Exact local evidence

- V2 input receipt: `d5a35e6341e53de0286cfd777e4fadd707c12cf9d18f95010a4a38ffac0ab59d`.
- Actual native model-free full renderer/ID proof: `2f67d5c2e85db441f5f21e36f7b54b7d6272713c10cac325674d56c5a9a2dae2`.
- Native outer: `9ae11e11a4a15e4277136998164329b18f45e059abc692cefcf0b3d16bc8533c`;
  normal runtime `ec0914c16259fac3dc97a6cb4c8a46fc12648d40a96fc9955d5b6a207c159fbe`.
- TensorFold outer: `6a415ed73a2d10f0026982a476c2d35d2b43b2708bb5f1e40fb3b76210fc6e78`;
  image `sha256:1a2afff2bd001cdea746d96bdb7614a4f937caa28e1646babd9acb951ea3aa48`,
  checkpoint `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa80`.
- Legacy Mia outer: `948bbbcd050b5b9e4fba4efb0b0f0520cd33884d0730d130d12fb89803c8cc9f`;
  image `fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`,
  recipe `b8439110`, exact checkpoint `925d7be6`; all cells and warm probe
  finish successfully, owned launcher/container retire cleanly, terminal
 117.239 GiB strong-clear gate. Cold-load refusal r1 is retained separately;
  the repair only removes a newline from the offline HF revision pointer.
- Warm anchor receipt: `31a28b300f3279e035bbcbb9a252ff06780574408974669ef54b51c468f501d3`;
  frozen prompt IDs `306136a7dc2a5f66c94cbf4ac874e9011c0af79b79072731407a144bfba1fbd5`,
  continuation file `c0f9706146aca18113f1cd91cdc1e1107c18df34c72fdd6a56e3192dc6a7aa3c`.
- Current TensorFold outer: `21fc73b1f832bc6b2afedb3ec08c1553a86061b4abc043af106a47f5622045bc`;
  package/calibration build `5532ffc386645b13955f2e8ffedcb360d84af10a48821b6d5524266e0eb4d99f`,
  official source `56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21`,
  image `sha256:c8dc97d6dab8995704b6c151715f11f84419775a193c96a5ed3399407ad41c20`.
  Actual TF 0.6.2 / torch 2.13.0a0+9186a08b2c.nv26.07 / CUDA 13.3 /
  Triton 3.7.1 are retained with the installed package and declared-dependency
  maps. All 11 commands finish rc0 and are reaped under PGID 3851298;
  supervisor 3851295 finishes rc0. Terminal receipt gate is 117.214 GiB clear;
  the subsequent fresh gate is 117.243 GiB clear. Failed model-free image-build
  r1 is retained; the repair only binds the verified parent image to a private
  Docker alias. No toolchain/dependency or numerical source replacement.
- Raw local roots: `/home/pmeenan/scratch/m3-serving-concurrent-records/native-v2-r1/`,
  `tf-v2-r1/`, `mia-v2-r2/`, `mia-anchor-four-r1/`, `native-ids-r1/`,
  `tf-current-build-r2/` and `tf-current-r1/`.
  All complete services/model subprocesses
  terminate and are reaped; terminal strong GPU/container/process gate passes.

This is one condition/one sample per cell. Native two-request throughput is
nearly flat through C4; TensorFold scales further. More source attribution
is needed before assigning the gap to any one kernel or scheduling stage.
The original Mia cells also scale; native leads its solo rate but trails C2/C4.
Against current same-checkpoint TensorFold, native's completed-token rate is
19.22% higher at C1 and 8.63% higher at C2, but 14.55% lower at C4. Native
completes its first C4 request sooner (18.12 versus 30.15 seconds), while current
TensorFold completes all four sooner (32.13 versus 37.60 seconds). This identifies
a remaining concurrency gap without assigning it to one stage or draft policy.
