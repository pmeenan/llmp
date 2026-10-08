<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen causal masks through the shared GPU producer

Qwen3.8 native and GGUF runners now build dense causal masks on the GPU;
each native headed MTP pass uses its own checked positions. The shared
producer supports exact-row F16/F32 output and retains the existing padded
F16 Gemma contract. Fast device QSA selection keeps its existing sparse
producer. Host block tables remain independent of mask materialization;
RE-037's dense-consumer limits, state envelopes and public admission do not
change. The host route remains an explicit comparison control.

## Same-native factor

Spark B, GB10, driver580.178.04, CUDA13.4, pinned SDK
`aarch64-c09daba6ac31edee`; actual cuBLAS/Lt SHA below. Four fresh
processes in off/on/on/off order, each warming plans/graphs and then paying
for a fresh 1536-ID prompt and 32 completed greedy outputs: context 2048,
chunk 512, depth 3, adaptive depth off, confidence 0, graphs enabled. Plain and
native MTP modes each warm independently. Complete finite vocabulary heads,
choices, step trajectories, initialized target/draft state and graph modes
are exact across all arms. Timers include state reset/growth, input work,
planning/capture, execution and final owed settlement. Observation/state
copy/hash and payload persistence occur afterward. Full-head publication
inside generation is paid; this screen is not a lean-verdict serving benchmark.

| Arm | Plain generation seconds | Speculative generation seconds |
| --- | ---: | ---: |
| off1 | 2.329084520 | 1.863570907 |
| on1 | 2.317180283 | 1.863459539 |
| on2 | 2.322242871 | 1.857521817 |
| off2 | 2.326261768 | 1.860507058 |
| mean off → on | 2.327673144 → 2.319711577 | 1.862038983 → 1.860490678 |
| descriptive change, n=2 | −0.342% | −0.083% |

Plain ranges do not overlap, with −0.121% off-bookend drift; speculative
ranges overlap and off-bookend drift is −0.164%, so there is no speculative
speed claim. Decode means move −0.051%/−0.160%, within noise. Total minus
decode is not an isolated prefill measurement. Plain staged mask bytes fall
3,256,832→0 with 34 GPU producers; MTP staged bytes fall6,627,328→0 with
14 target and 33 draft producers. The selected graph schedules are unchanged.

## Focused qualification and memory

Four unique tests pass on Spark B: host-mask omission preserves selection
metadata; headed/headless MTP graphs retain only actual consumers; F16/F32
GPU operands match eager and changed-position poisoned capture/replay;
and native/GGUF fast/reference graphs authenticate actual output dtype,
producer/source/input membership, placement/funding and malformed-source
refusals. The GPU fixture executes eight dtype/layout/window cases;
these are cases within one test, not eight additional tests.

Native C2 qualification executes actual DraftWave/VerifyWave, per-owner
17/25 complete finite heads,16 choices each, capture/replay, initialized
state spill/restore and a protected peer. All outputs/state/work are exact
off/on; staged mask bytes 11,081,728→0, target/draft producers 21/30. GGUF
C1 target-only qualification preserves16 complete finite heads/choices,
initialized state and15 replays; staged bytes 3,199,488→0 and 18 target
producers. These n=1 qualification pairs establish correctness, not speed.
Native F32 fallback/GGUF F32 contracts are qualified by checked planned
graphs plus the executed shared F32 primitive, without a forced public
reference mode or fallback-model performance claim. Existing HTTP adapters
inherit the runner default; this is not a fresh HTTP throughput measurement.

Both measured setup recipes reduce activation charges2 MiB, pinned charges
4 MiB and host-input charges2 MiB: **8 MiB less total distinct charge**. Input
staging falls4 MiB and is already included in the pinned reduction; do not
count it twice. Scratch, PLE, output, snapshot, runner-map and state charges
are unchanged. Native activations 180,355,072→178,257,920, pinned 93,589,292→
89,394,988 and host 4,194,304→2,097,152 bytes. GGUF activations 197,132,288→
195,035,136, pinned 74,821,484→70,627,180 and the same host reduction.

Earlier failed records remain separate: initial shared-GPU fixture signed
conversions prevented compilation; a first device-MTP setup failed on an
unused headless producer and its sole valid off arm is not pooled; a later
host negative-control fixture mutated the custom-op header rather than
parameters 6/7. Narrow corrections and successful retries supersede these
attempts. No model ran after the host-fixture failure. Full regression and
maximum-context ladders remain deferred under the owner's optimization override.

## Fresh competitive references

On the same GB10, fresh R/N/N/R processes each warm the same 1536-ID prompt,
clear prompt state without dropping warmed plans, then generate 32 greedy
IDs with logical context 2048 and actual 512-row prefill chunks. Reference
bookends and both native arms positively retire; complete vocabulary outputs
are finite and the native arms retain exact choices, state and schedules.
The following are measured generation endpoints, not HTTP throughput. No
profiled timings or cold-checkpoint loading enter the paid measurements.

TensorFold task-entry native HEAD is
`f8fe17d24629aedabf90bbf78279dd776e6d62e7`; its applicable CUDA Python branch is
`ed78d6fc204d89d90b045bf033d6551e7714f3a1` (0.6.6), checked again unchanged
before final comparison. The native branch's Flash-Next recipe is MLX-only.
The Python package is installed from that exact source over dependency image
`sha256:c8dc97d6dab8995704b6c151715f11f84419775a193c96a5ed3399407ad41c20`;
the image's older TensorFold is bypassed. All 458 installed runtime payloads
match pinned source. Only three package-omitted documentation files are
excluded: families/README.md, families/qwen3_5/cuda/README.md and
kernels/README.md. Eight prepared extensions and actual loaded cuBLAS/Lt/
cudart payloads are authenticated after real forward/teacher execution.

Both engines use the approved Mia NVFP4/MXFP8 checkpoint. TensorFold runs its
fast public FP8 prompt policy (`--prefill-fp8`, actual cuts 512/512/512), BF16
KV and convolution state, and F32 GDN recurrent matrices; native uses F16 KV
and F32 convolution/recurrent state. TensorFold plain follows its public
serial eager policy; speculative mode uses graphs, fixed cap 3, confidence 0
and the unmodified public 79,591-token draft list with requantized four-bit
draft-head/experts. Native speculative mode uses the approved 47,172-row
prepared head and ordinary lean verdicts. The pinned copy-match chain requires
at least eight nodes, so repeated-tail substitution does not execute at fixed
cap 3 even though the public copy-index policy/overhead remains enabled. These precision/head/cache policies
are disclosed differences, not byte-identical cross-engine arithmetic.
TensorFold's logical context is 2048; cache capacity is 2052 including draft
slack. Native plain publishes full target heads inside generation while
TensorFold plain publishes tokens. Native speculative full-head Teacher32
observation runs after the paid, settled lean generation endpoint. Reference
full-head observation likewise stays outside paid generation.

| TensorFold comparison arm | Plain generation seconds | Lean/public speculative generation seconds |
| --- | ---: | ---: |
| reference1 | 2.491353649 | 1.780775360 |
| native1 | 2.312032537 | 1.833047196 |
| native2 | 2.309071054 | 1.836765171 |
| reference2 | 2.481176308 | 1.773984807 |
| native versus reference mean, n=2 | −7.067% | +3.237% |

Reference bookend drift is −0.409% plain / −0.381% speculative. Native plain
is faster at this target-only endpoint; TensorFold's public speculative policy
is faster than native lean generation here. These short n=2 results do not
establish sustained or universal competitiveness. Native total minus decode
is not an isolated prefill interval.
The selected-head layout is a concrete remaining transfer lead: native's
47,172 BF16 rows occupy 241,520,640 bytes, while the same IDs in affine Q4
(group 32, BF16 scale/bias) would occupy 75,475,200 logical bytes before
padding. TensorFold's public 79,591-row Q4 head occupies 127,345,600 logical
bytes. This is a source-derived layout comparison, not a measured speed
cause. Native MTP experts already use NVFP4 CUTLASS. A same-ID quant-only
probe must preserve acceptance, output quality and funding before adoption.

The same-format GGUF comparator is llama.cpp v0.6.0,
`d81235049384534c167caea52b85a694f6103d14`, immutable image
`ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db`.
Its exact runtime CUDA backend, all 49/49 offloaded layers and enabled flash
attention are checked, alongside actual backend/library payloads. Both engines
use the same UD-IQ3_XXS shards, F16 KV and target-only greedy decoding; there is
no GGUF drafter. Stock requests a prompt head only at the final chunk while
native retains its intermediate heads. Nonfinal headless stock chunks have
no extra harness synchronization. Both publish all 32 generation heads.

| GGUF arm | Generation seconds | Decode seconds |
| --- | ---: | ---: |
| reference1 | 3.283775400 | 1.089505601 |
| native1 | 3.091262868 | 0.971649667 |
| native2 | 3.077612229 | 0.968491512 |
| reference2 | 3.259321311 | 1.061719985 |
| native versus reference mean, n=2 | −5.719% | −9.812% |

Reference drift is −0.745% generation / −2.550% decode. This supports a
bounded target-only speed advantage, not an HTTP or all-context claim.

### Common-history quality remains separate

All compared heads contain all 248,320 finite vocabulary values. The retained
32 generated-history targets use identical conditioning within each quality
comparison. They are selected from native generation, not a held-out corpus;
NLL and its exponential below are descriptive, with native-history selection
bias. Neither is a perplexity qualification or evidence of better quality.
The existing reference top-1 minus top-2 near-tie bound remains **1.0**.

| Comparator | Same-conditioned argmax agreement | Outside original 1.0 bound | Mean NLL native−reference | exp(mean NLL delta) |
| --- | ---: | ---: | ---: | ---: |
| TensorFold FP8 prompts, both modes and both reference arms | 31/32 | row 29 | −0.099247814 | 0.905518280 |
| llama.cpp GGUF, reference2 on native1 history | 30/32 | none | −0.063772518 | 0.938218403 |

TensorFold row 29 chooses 47149 while native chooses 17723. Its reference
margin is 1.5625; native's margin is 0.828717232. Choosing the other engine's
argmax costs 0.828717232 native logits and 2.6875 TensorFold logits. This is
outside the existing bound and remains an **open M3.5 quality lead**, without
extending the margin or applying the earlier French-fixture exception.
Natural trajectories agree only 29/32 and first diverge at index 29; the
teacher result is a separate same-history check. Native full-head bytes are
identical to both earlier host-mask and device-mask controls, so the transfer
preserves native arithmetic; that does not pass the TensorFold quality gate.
The controlled BF16-prompt follow-up below isolates prompt policy under the
same history; it does not attribute the remaining difference to a kernel.

The maximum absolute log-probability delta over the union of each engine's
top five is 3.651273648 for TensorFold and 2.347130313 for GGUF; means of the
per-row union-average absolute deltas are 0.442955186 / 0.303769515.
GGUF disagreements at rows 13 and 28 have reference margins 0.424640656 /
0.643239975, both inside the original bound. GGUF common-history quality is
one reference2 observation; reference1 uses its own natural history and is
not pooled into that NLL claim. Both stock natural runs match native on only
13/32 choices, first diverging at index 13. Natural branching and
same-conditioning agreement are different quantities.

### Controlled BF16-prompt follow-up

On 2026-10-08 at 06:55:10 UTC, task-entry TensorFold native/Python refs
remained `f8fe17d24629aedabf90bbf78279dd776e6d62e7` /
`ed78d6fc204d89d90b045bf033d6551e7714f3a1`. One fresh Python engine used the
same checkpoint, literal 1536-ID prompt and retained 32-ID native history as
the preceding reference. Only its upstream prompt-precision policy changed:
`prompt_precision.set_fp8(False)` before engine construction. This is an
untimed Teacher32 diagnostic, without natural generation, warmup or native
execution. The original fast-FP8 competitive comparison remains unchanged.

The [BF16 teacher harness](tensorfold_bf16_teacher.py) uses the public eager
serial twin, logical context 2048/cache 2052, actual 512/512/512 cuts, BF16
KV/convolution state and F32 GDN recurrence. Actual prompt consumers report
576 folded BF16 calls, no lane fallback and no FP8 calls. All 32 full
248,320-value heads are finite. Package, all eight actual loaded extensions,
cuBLAS/Lt/cudart, immutable checkpoint authentication, conditioning and
positive shutdown/container retirement were checked. Torch allocated bytes
were 77,701,179,904 before / 77,701,200,384 after the teacher, with process peak
78,757,937,152; these are allocator observations, not physical occupancy or a
memory-saving result.

| Comparison on the same history | Argmax agreement | Outside unchanged reference-margin 1.0 | Mean NLL first−second | exp(mean NLL delta) |
| --- | ---: | ---: | ---: | ---: |
| Native / TensorFold BF16 prompt | 30/32 | none | −0.153334639 | 0.857842606 |
| TensorFold FP8 / BF16 prompt | 31/32 | none | −0.054086825 | 0.947349849 |

At row 29, both TensorFold policies choose 47149, while native chooses 17723.
TensorFold's top-two margin moves **1.5625 → 0.6875** and its logit loss for
native's choice moves **2.6875 → 1.5**. Native's margin/loss remains
0.828717232. Prompt policy therefore changes this row's classification under
the existing reference-top-two near-tie bound; it does **not** recover the
native argmax or establish which arithmetic is correct. The original fast-FP8
quality finding remains open.

BF16 also introduces a new disagreement at row 5: native/FP8 choose 364,
BF16 chooses 835. Their top-two margins are 5.618597984 / 5.625 / 0.125;
BF16's loss for native's choice is 0.125, while native's loss for BF16's
choice is 8.158902168. The supplied target's NLL is 0.010121244 native,
0.034020265 FP8 and 2.387459279 BF16. This substantial posterior movement
precludes a universal closer/correct claim. Maximum union-top-five absolute
log-probability deltas are 6.991114038 native/BF16 and 4.459060986 FP8/BF16;
means of per-row union-average deltas are 0.465648910 / 0.403931900.
These generated-native-history observations are descriptive, not held-out
PPL or evidence of better quality. Further frontier/arithmetic qualification
remains open; only the prompt-policy isolation step is complete.

The first diagnostic failed after checkpoint loading, before any teacher
forward or payload: the fresh public engine initializes its serial twin
lazily. The corrected harness follows the pinned public `engine.e.twin()`
sequence and records `serial_initialized_here=true`. The failed application
retired positively and is not pooled. No native rebuild or inference was
needed for the correction.

Successful diagnostic source inventory SHA
`21dab3fe688938133f1c8bbb6268256e1b249d7a0b0e1884ffcb8668c7dfee07`,
harness SHA `c6aa1503cf8682cb86483d4f25d1dfd5c30a4204e16fee4519b02287984578b2`,
passed receipt SHA
`0d13948fa14b5011a7f5f5276c4d084d6d0828600d89c8e5b4999fc04e05e7e2`.
Compared payload SHAs are native
`aa5b3e0876e7ac8d2618b08aadfc261ffdb4e3dafc8e4f68cb40c02c28c2ee74`,
FP8 `b1aaccb17c4f862a0ed137c606fc06214feaf98fb86235ee9793b77dcbae48aa`,
and BF16 `791ec7a68621444cce358a524b898fa9ed2945a8d673759cf69e27208ba2c53c`.
Raw evidence remains outside Git at
`/tmp/llmp-m35-coordination/qwen-bf16-prompt-raw/diagnostic2` and
Spark B `~/scratch/m35-qwen-bf16-prompt/diagnostic2`; it may be deleted when
M3.5 closes. The standing inputs and replay recipe below recreate the heads.

### Controlled prompt/serial frontier follow-up

A second untimed diagnostic puts the same row-29 conditioning entirely into
both engines' prompt path: the original 1536 IDs followed by the first 29
native history IDs, totaling 1565. Task-entry TensorFold native/Python refs
remained the pins above. Native reuses the qualified target-only benchmark
without a rebuild; TensorFold keeps BF16 prompt policy before construction.
Both use logical context 2048 and chunk 512. TensorFold actually executes
512/512/512/29 cuts, 768 folded BF16 calls, zero FP8/lane-fallback calls and
**zero serial suffix forwards**. Its public serial twin is initialized for
prefill, but the observed head comes only from the prompt frontier. Native
publishes five complete finite heads; only its first is this frontier. The
other four are excluded from this comparison. TensorFold publishes one
complete finite head, also 248,320 values.

| Same conditioning at row 29 | Native argmax / margin | TensorFold argmax / margin | Maximum absolute logit difference |
| --- | ---: | ---: | ---: |
| Both prompt paths, TF BF16 | 17723 / 2.602531433 | 47149 / 1.3125 | 4.430672884 |
| Both serial paths, TF BF16 (retained) | 17723 / 0.828717232 | 47149 / 0.6875 | 2.314872384 |
| Both serial paths, TF FP8 (original) | 17723 / 0.828717232 | 47149 / 1.5625 | 2.825488567 |

Within native, prompt versus serial keeps argmax 17723 but changes all head
values, with maximum absolute delta 2.757405281. Within TensorFold BF16,
prompt versus serial keeps argmax 47149, with maximum delta 1.703125. Matching
this prompt/serial geometry therefore **does not recover the cross-engine
argmax**. In the all-prompt comparison, TensorFold's loss for native's choice
is 3.3125, and native's loss for TensorFold's choice is 2.602531433. TensorFold's
reference top-two margin 1.3125 exceeds the unchanged 1.0 bound. The
[aggregate](frontier-results.json) retains full-vector identities, directional
losses, union-top-five log-probability deltas and the supplied target's NLL.
These are one-position observations, not PPL, correctness or speed claims.
The original competitive FP8 finding remains open; this completes only the
prompt/serial isolation step. No particular kernel, cache or state-precision
cause is established.

The acquisition authenticated actual package/JIT/runtime library consumers,
immutable checkpoint receipt and input identities before/after. Both
applications returned 0 with completion markers; native teardown, independent
container removal and the empty GPU process list were proved. No failed
attempt or retry occurred. Default `--fixed-prefix-rows 0` retains the
existing Teacher32 route; new `--fixed-prefix-rows 29` suppresses its suffix
loop and requires exactly one head. No production source changed, and no
native build, regression suite or performance run was needed.

Source inventory SHA
`05504b89eff58a454d78f8dd59625f63dd7ad24b0ea5eae0cdfda6c2f712dfa2`,
executed harness SHA
`cc5f8448216970cd57352f7e2256dbb17d31d26966f9edec2c684b01e4d2dc5b`,
positive receipt SHA
`e0268e11e6dfd6a1606b34b866ef99d0d19dc0716403b2d724be9d6e978f508a`.
Derived prompt int32 SHA
`b274b0359d6d37db002c0ae3f8f70e1bc1a9d65f4a5cd523411184c9aaf0ae10`.
Native first head SHA
`06b771d11c8af1aa2df5be2617cd760c5bde7faa4b120083ff50b84219b0d59a`;
TensorFold prompt head SHA
`9e0dbeacaccd6cc4f0d74e1a91956cad65bf2f469265eed31f928007ea8d21d1`.
Raw evidence is external at
`/tmp/llmp-m35-coordination/qwen-frontier-quality-raw/frontier1` and
Spark B `~/scratch/m35-qwen-frontier-quality/frontier1`; it may be deleted at
M3.5 close. Standing authenticated inputs and the replay recipe below remain.

### Reference provenance and retained failures

Native acquisition source inventory is
`54f502cffce084c8f69194b20235b457037e7e64f0a67bbc90c8e8eed02964c7`,
build receipt aggregate
`29f130bd936fa3e1b08a144facfd1d889b96e5a9ef465dc56193f089bf5b8002`;
spec ELF SHA 209fc1d668dcadc9d2159bb5d29537c526e800f98285a423c34e758f45039b8e,
runtime ELF SHA 7b2e0258d065e02ed4e91d35147186c97dd79c5b82ac1e8898c351491c74fbee.
The final reference source inventory is
`44b5baeca06f5dfa47c6a00b86543969dd0b8723fc7c377be3ca26f3a6aa1b3d`;
its only change from the compiled source is the documentation-owned
TensorFold harness's logical-context/cache-capacity check. Native production
and benchmark bytes are unchanged. TensorFold aggregate
SHA `ccdebf19d7a298f040f05b61aac40502d417743f255fc42693148b9c44c9833c`;
GGUF aggregate
SHA `60ff44fd429d8b2dfe0d16adfefbfa31753b2a156d19a5659dc925f0fd67492d`.
The stock helper source/ELF SHAs are
`94f2b5c91182897b9ee444179ef1d66bb4972e41de13b56ea863802f83b8daa2` /
`db4e48aac16c523d2ae6639b97962c89de3703f66f4d74f5c7e8582cf264b971`.

The first TensorFold attempt completed full checkpoint authentication and
warmups, then refused the incorrect logical 2048/cache 2052 assertion before
any paid arm. Its failed record is unpooled. The harness-only correction
checks logical context and cache capacity separately, and the fresh four-arm
retry supplies the results above. An authentication-only stage retained the
50 complete pinned checkpoint hashes (105.936 GB) with exact dev/inode/size/
mtime/ctime, reused only for the private immutable standing checkpoint mounted
read-only. Preparation failures (absent remote script parent, then three
package-omitted README files) are retained separately; no inference/JIT
success is inferred from them. Corrected package authentication and all
extension-preparation groups positively completed before the references.

### Integration with the shared pager foundation

After the references completed, the candidate was reconciled with main
`46cfd5c` (partial-weight foundation and shared Wake/completion improvements).
All 26 Qwen graph/model/runner/benchmark/shared-mask/test paths in the measured
source are byte-identical; the reference timings above keep their original
pre-foundation identity. Integrated source inventory
`221219ab18b8e1f00ff4cede0d044501c5c157861e00b7826474192a6d6eb5b6`
then rebuilt the current runtime/spec and two focused test binaries.
Four named controls pass again, followed by one current native C2 off/on
pair with exact complete finite heads/choices/state/work, actual joined
DraftWave/VerifyWave, capture/replay and protected-peer spill/restore.
Staged masks remain11,081,728→0 and producers 21 target / 30 draft. Build/model
receipt SHAs are
`ee29fdc5be29954a1aeb2eedfc07654614bee46d2d4704c55b553facec29ead7` /
`67b817cbaba05b1f790407cd3aef1bb781171fe57d623aee22ebd17f417a2b02`.
This qualifies current composition correctness without rerunning or relabeling
the earlier reference times as measurements of the integrated pager source.
Ordinary partial eviction remains off; its cold-switch gate is separate.

## Replay and provenance

Prepared target `c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`
and MTP `8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`
are under `~/.local/share/llmp/m3-artifacts` on Spark B, imported from
`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6`. GGUF target
`5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78`
is under `~/.local/share/llmp/qgguf-artifacts`, from
`unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee` UD-IQ3_XXS; no GGUF drafter.
Manifest SHA equals each artifact ID. Tokenizer is the native checkpoint's
`tokenizer.json`, SHA 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3.

Replay inputs are retained independently of raw results at
`~/.local/share/llmp/references/qwen-device-masks/inputs` on Spark B.
`standing-literal-prompts.json`, SHA 6d3b3aded76477a6cb1c40a64c941c0cab73411b059b3cf548f6a9293ff15809,
is the previously captured literal set: use `p3`'s first 1536 IDs without
retokenizing. `prompt.json`, SHA 204185510a2c6f05f7b9672a96e66de5d816d188832cde728b7d64952423c066,
is that single named decode input; packed little-endian I32 IDs SHA
35ece15522faff9e2895774eb9146b83a7ad623de08b484ac7e6a8feeb29c35d.
`wave-prompts.json`, SHA 52dd48eb2b96be3ebc6de1a9c8e3f5155016599a20c85c1dc1dd3778d2123952,
uses first 1536 and first 1280 IDs of the same prefix. Obtain these verified
external captures from that standing store, or supply byte-identical files;
raw logs/traces are unnecessary and may be deleted when M3.5 closes.

Build [qwen38_spec.cc](../../../benchmarks/qwen38_spec.cc), then use this
invocation inside installed `spark-job --gpu` supervision. Replace the
uppercase paths with the verified stores named above and a new output path.

```sh
llmp_qwen38_spec --check masks --qwen38-artifact TARGET --drafter MTP \
  --draft-vocab 65536 --tokenizer TOKENIZER --prompts INPUT/prompt.json \
  --only mask-short-1536 --prompt-token-ids on --tokens 32 --context 2048 \
  --prefill-chunk 512 --draft 3 --adaptive-depth off --window 0 --graphs on \
  --device-masks on --out NEW_DIRECTORY
```

Use `--device-masks off` for the old control. Target-only GGUF uses
`--check masks-target` and omits `--drafter`/`--draft-vocab`. Joined native
qualification uses `--check masks-wave --slots 2 --tokens 16 --prompts
INPUT/wave-prompts.json` and omits `--only`. The benchmark-only lean
speculative comparison uses `--check masks-lean`: no full verifier-head
publication in its paid endpoint, followed by an untimed teacher-forced
full-head observation on its own emitted IDs. Each successful app must
explicitly tear down its node, print `DONE OUT`, return 0, and leave no GPU
process before the next arm.

Actual native cuBLAS13 SHA ee7c1657a03695c0de790aa79e34cef9c9649756b1846b11dd44caca20ba656b;
Lt SHA ba3b942f4ea43433b65e8c492a7b73de887534dc20146506ddaa4a78c79c5d30.
Raw results are external under `/tmp/llmp-m35-coordination/qwen-gpu-mask-raw`
on the workstation and its synced Spark B scratch copy. Qualification
`passed.json` SHA db3e8bc3ee824f324266aa394c4f67d0d2d5fba6517f6a6ca3f71726ff516221;
first-factor aggregate SHA 5b27e0b4eda935065939e1930ef05098424ca83651cee6fd97cdbfbb740bbcfd.

The transfer closes inventory T22/T69 for Qwen native/MTP and GGUF targets.
DSpark noncausal block and compressed visible-count policies remain open;
Gemma families already use the shared F16 producer; image causality is
implicit and has no host matrix to replace. Host-reference fill T63 remains
separate useful work even though the ordinary runner now bypasses it.

## Durable reference replay

The checked-in [native build controller](build_native.py),
[stock build controller](build_llama.py), [llama helper](llama_mask_reference.cc),
[reference cycle](reference_cycle.py), [TensorFold harness](tensorfold_reference.py),
[TensorFold maps wrapper](tf_reference_wrapper.py),
[stock wrapper](stock_reference_wrapper.sh) and [quality analyzer](quality.py)
retain the reusable method. They accept freshly built source/binary receipts;
no deleted raw-job receipt is required. The aggregate identities above describe
the original acquisition rather than authorizing new binaries implicitly.
Each controller refuses mismatched source, input, model, library or retirement
proofs. Run each heavy stage separately through installed Spark supervision,
with a new output directory, 600-second timeout, 30-second grace and stop-on-fail.
Do not continue after a refusal or pool failed arms.

Keep the verified standing replay inputs, TensorFold source/package/JIT-cache
receipts, stock headers/image closure and checkpoint authentication independently
of milestone raw cleanup. Copy those authenticated standing stores from Spark B
when replaying elsewhere, or recreate them from the pinned upstream/checkpoint
sources and verify the same payloads before measurement. The reference cycle
checks the actual eight loaded prepared extensions against the preparation
receipts and validates all runtime package files; receipt names alone are not
proof. For freshly prepared JIT files, supply `--tf-preparation-receipts` as a
JSON object mapping authenticate/core/gdn/nvfp4 to the four new receipt SHAs;
each receipt still binds the exact upstream commit/source inventory and actual
extension bytes. This avoids requiring an earlier raw bundle.

The upstream source archive SHA is
`e8f8bb73a45aa8e9dd2700201981ba6021ffb12add58331e825c51ebb53fdb1d`;
standing source inventory `references/tensorfold/ed78d6fc/tf-source.json`
SHA `f12e61f93dda091508445182bf06a36eb745028f13bebe2c2d834953e556bad3`.
Source, installed package and prepared receipts live beside that inventory;
JIT payloads live under `tensorfold-0.6.2-nvfp4-home`. The name identifies a
retained dependency cache, not the selected runtime version. The source archive
retains the upstream licenses. Stock headers/source identity and image closure
live under `llama-reference-v060`; closure receipt SHA
`63ccae40a696fb574bedf1145a75f9576a0f7914efbd8081fc6b085babe3cf64`.

The native checkpoint's 50 file identities and sizes come from
[fast-swap/pins.json](../fast-swap/pins.json), revision
`925d7be6c14c6c9442ef83e8f05b5a3c39304f69`. A new copy needs a separate full
[authentication stage](authenticate_checkpoint.py) before inference, using
that pins file. The current private standing copy has authentication receipt
`references/qwen-device-masks/tf-checkpoint-auth.json`, SHA
`e2ff99b2ea1405ca2c7162e86855eb885e68edd7fd9f4fe92abe8511214b4b1c`.
The cycle requires all recorded dev/inode/size/mtime/ctime values to remain
exact before/after, and mounts the checkpoint read-only. This reuse assumes
trusted private immutable storage, not protection from adversarial host mutation.
For a newly authenticated copy, pass its receipt and exact SHA explicitly.

GGUF raw shards are retained under `reference-models/Q/UD-IQ3_XXS/`, filenames
`Qwen3.8-Flash-Next-UD-IQ3_XXS-0000N-of-00003.gguf`:

| N | Bytes | SHA256 |
| --- | ---: | --- |
| 1 | 10946624 | `268f81fdedf3149a538f252308927a4d5d1f6e062c178568a51e3b519744f8a8` |
| 2 | 49567921344 | `cfe600b236b88c7fad1613a5ca5e83b9f2beb63cbd44c32b2be50a44747c695f` |
| 3 | 32382955968 | `f1912ba34c79427d2295a58dcb2b732b5931af5bef7a373c60557a57d9ee7250` |

`native-histories.json` in the standing input directory has SHA
`11250b3551f0492dd0a6b4f59f09ab6f438a6eb7474e9e6b31befb8c250c437b`;
it retains the 32-ID plain/speculative conditioning histories separately.
The helper consumes literal IDs directly, avoiding retokenization. Reference2
GGUF Teacher32 takes native1's actual history; reference1 follows its own
history. TensorFold Teacher32 takes the authenticated native histories in
both arms. The cycle checks those distinctions explicitly.

Create a fresh complete source inventory on the workstation from the checkout
used for the build; place the JSON outside the repository. Each `files` entry
is a repository-relative path mapped to SHA256. Include tracked and intended
untracked files and a `base` field containing HEAD; reject any merge conflicts.
Checksum-sync that exact checkout to the Spark and require an empty checksum
dryrun. `build_native.py` checks the inventory before/after its locked SDK build
and produces `passed.json` binding actual ELFs, receipt and private cuBLAS/Lt.
Do not edit the inventoried source during a build or acquisition. Example
commands on the Spark, with absolute `TREE`, `MANIFEST` and fresh `OUT` paths:

```sh
~/.local/bin/spark-job start --gpu --name qwen-replay-native-build \
  --timeout 600 --grace 30 --stop-on-fail -- \
  python3 TREE/docs/experiments/qwen-device-masks/build_native.py \
  --tree TREE --source-manifest MANIFEST --out OUT/native-build
~/.local/bin/spark-job wait qwen-replay-native-build

~/.local/bin/spark-job start --gpu --name qwen-replay-stock-build \
  --timeout 600 --grace 30 --stop-on-fail -- \
  python3 TREE/docs/experiments/qwen-device-masks/build_llama.py \
  --out OUT/stock-build
~/.local/bin/spark-job wait qwen-replay-stock-build
```

Only after both builds positively complete, run each fixed R/N/N/R separately:

```sh
~/.local/bin/spark-job start --gpu --name qwen-replay-tensorfold \
  --timeout 600 --grace 30 --stop-on-fail -- \
  python3 TREE/docs/experiments/qwen-device-masks/reference_cycle.py tensorfold \
  --tree TREE --source-manifest MANIFEST \
  --native-build OUT/native-build/passed.json --out OUT/tensorfold
~/.local/bin/spark-job wait qwen-replay-tensorfold

~/.local/bin/spark-job start --gpu --name qwen-replay-gguf \
  --timeout 600 --grace 30 --stop-on-fail -- \
  python3 TREE/docs/experiments/qwen-device-masks/reference_cycle.py gguf \
  --tree TREE --source-manifest MANIFEST \
  --native-build OUT/native-build/passed.json \
  --stock-build OUT/stock-build/passed.json --out OUT/gguf
~/.local/bin/spark-job wait qwen-replay-gguf
```

The controllers bind exact immutable images, source/header/link closures,
actual loaded backend/libraries, models, literal inputs and pre/post GPU/memory
guards. Reference containers are independently retired in `finally`, even
when the application fails. Accept only completed return0 applications with
explicit teardown markers, absent containers and empty GPU process lists.
The quality analyzer reads already persisted heads; run it off the paid
endpoint under the workstation shared lock (or a supervised Spark job):

```sh
hostlock shared --label 'llmpalooza: Qwen common-history analysis' -- \
  python3 TREE/docs/experiments/qwen-device-masks/quality.py \
  OUT/tensorfold/native1-plain/plain-heads.f32 \
  OUT/tensorfold/reference2/plain-heads.f32 \
  INPUT/native-histories.json --history-key plain --out OUT/tf-quality.json
```

Use the actual payload filenames emitted by the harness (`plain-heads.f32`
versus `spec-lean-heads.f32`) for each mode. For GGUF use native1's
`plain-heads.f32`, reference2's `teacher-heads.f32` and a JSON array copied
from native1's validated `tokens` field. Original aggregate quality is
reproducible from these source/inputs; retaining old raw outputs is optional.

### Replaying the BF16 policy isolation

First recreate native/FP8 heads with the TensorFold `reference_cycle.py`
recipe above, or authenticate retained complete finite payloads with the
recorded conditioning. The BF16 teacher consumes the same standing
`prompt.json` / `native-histories.json`; no raw result bundle is required.
Use the same prepared Python source/package, eight extensions, dependency
image and full checkpoint authentication. Verify the checkpoint receipt's
exact file stats before/after, as `reference_cycle.py` does; a changed copy
needs full authentication again. Freeze/hash the source inventory, harness,
wrapper, input files and preparation receipts before/after acquisition.

Create a short external shell runner with the following body, setting `TREE`
and a fresh absolute `OUT` directory at its start. Run that file through
`~/.local/bin/spark-job start --gpu --name qwen-bf16-replay --timeout 600
--grace 30 --stop-on-fail -- bash RUNNER`, then wait for the installed
supervisor. Preflight sufficient memory and absence of foreign GPU processes;
do not overlap reference measurements. The owned container is independently
removed by the shell trap even on application failure or timeout.

```sh
set -eu
H="$HOME/.local/share/llmp"
F="$H/references/tensorfold/ed78d6fc"
INPUT="$H/references/qwen-device-masks/inputs"
MODEL="$H/models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6"
NAME=qwen-bf16-replay
IMAGE=sha256:c8dc97d6dab8995704b6c151715f11f84419775a193c96a5ed3399407ad41c20
mkdir "$OUT"
test -z "$(docker ps -aq --filter "name=^/$NAME$")"
retire() { docker rm -f "$NAME" > "$OUT/container-retirement.log" 2>&1 || :; }
trap retire EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
timeout --kill-after=10s 180s docker run --name "$NAME" --network none \
  --device nvidia.com/gpu=all --ipc host \
  --mount "type=bind,src=$OUT,dst=/out" \
  --mount "type=bind,src=$F,dst=/reference,readonly" \
  --mount "type=bind,src=$INPUT,dst=/inputs,readonly" \
  --mount "type=bind,src=$MODEL,dst=/model,readonly" \
  --mount "type=bind,src=$H/tensorfold-0.6.2-nvfp4-home,dst=/tfhome" \
  --mount "type=bind,src=$TREE/docs/experiments/qwen-device-masks/tensorfold_bf16_teacher.py,dst=/harness.py,readonly" \
  --mount "type=bind,src=$TREE/docs/experiments/qwen-device-masks/tf_reference_wrapper.py,dst=/wrapper.py,readonly" \
  --env HOME=/tfhome --env PYTHONPATH=/reference/package \
  --env PYTHONDONTWRITEBYTECODE=1 --env TENSORFOLD_PREFILL_ROWS=512 \
  --env MAX_JOBS=4 --env TORCH_CUDA_ARCH_LIST=12.1 \
  --entrypoint python3 "$IMAGE" -B /wrapper.py /harness.py \
  --model /model --inputs /inputs/prompt.json \
  --histories /inputs/native-histories.json --out /out/teacher \
  > "$OUT/application.log" 2>&1
grep -q TENSORFOLD_BF16_DIAGNOSTIC_COMPLETE "$OUT/application.log"
grep -q TENSORFOLD_REFERENCE_WRAPPER_COMPLETE "$OUT/application.log"
retire
test -z "$(docker ps -aq --filter "name=^/$NAME$")"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
```

Accept only return 0 plus both completion markers, absent container, empty GPU
process list, unchanged bindings and all 32 complete finite heads. Inspect
`teacher/diagnostic.json`: actual BF16 calls must be positive, FP8 calls zero,
cuts exactly 512/512/512, context/cache 2048/2052, fresh serial initialized and
teacher history equal to the supplied plain history. The wrapper authenticates
actual loaded preparation extensions and records runtime libraries. Capacity
and allocation observations establish this diagnostic's admission, not speed.

Analyze the persisted heads under the workstation shared lock using the same
`quality.py` command above, once native/BF16 and once FP8/BF16, with the plain
history key. In the second analysis, fields named `native_*` identify the first
operand (FP8), not llmpalooza. Keep the fixed 1.0 margin and both original reference
and controlled-policy results; no new performance or PPL claim follows from
this replay.

### Replaying the prompt/serial frontier isolation

Use the same standing `prompt.json` and `native-histories.json` as above.
Append `histories["plain"][:29]` to the 1536-ID prompt in a private input copy;
verify 1565 IDs and the aggregate's little-endian int32 SHA. No tokenization or
new prompt is needed. The [BF16 replay runner](#replaying-the-bf16-policy-isolation)
above accepts `--fixed-prefix-rows 29` after its other harness arguments; keep
its original 1536-ID input because the harness appends the history itself.
Require actual cuts 512/512/512/29, positive BF16 consumers, zero FP8 consumers,
`fixed_prefix_rows=29`, `prompt_rows=1565`, `suffix_forward_calls=0`, one finite
head, unchanged bindings and the same positive retirement proofs. Omit the
flag to recreate the original BF16 Teacher32 payload.

For native, use a freshly bound build of the existing `llmp_qwen38_spec`
benchmark with the authenticated target artifact/tokenizer, derived 1565-ID
input, `--check masks-target --tokens 5 --context 2048 --prefill-chunk 512`
and `--graphs on --device-masks on`, preserving the existing reference's other options.
Use the [native invocation and supplied-output recipe](#replay-and-provenance) above for
artifact, input and output flags. The first full head is the observed prompt
frontier; require five finite complete heads, no draft state, actual input SHA,
return 0 plus `DONE`, completed teardown and empty GPU process list. No timing
from this diagnostic is interpreted. Bind current binary/source/SDK/library
identities rather than depending on the old executable remaining available.

To recreate the serial controls, use `reference_cycle.py` and the default
BF16 Teacher32 route above on the original 1536-ID prompt and same 32-ID
history, then extract row 29 (zero-based, 993,280 bytes per row) from each
full-head file. Use [frontier_compare.py](frontier_compare.py) for each pair:

```sh
hostlock shared --label "llmpalooza: compare one Qwen frontier" -- \
  python3 docs/experiments/qwen-device-masks/frontier_compare.py \
  --left native-frontier.f32 --right bf16-frontier.f32 --target 17723 \
  --out comparison.json
```

Compare both engines prompt versus serial and the cross-engine prompt pair;
keep the original fast-FP8 result separately. The comparator requires all
248,320 finite F32 values and retains the original 1.0 reference-margin rule.
Standing inputs, pinned checkpoint/package preparation and their authentication
receipt survive raw-result cleanup; these recipes recreate the observed heads
without requiring a deleted job directory.
