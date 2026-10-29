<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# U3 Llama 3.3 NVFP4 bounded quality results

Retention (F10, 2026-10-28): this directory keeps the collectors and the
reference and native runners that the reproduction recipe below names. The
per-model run-once drivers under `gemma26/` and `qwen27/` (freeze, launch,
lifecycle and source-adapter scripts, their tests) and the per-file source-hash
inventories were deleted under AGENTS rule 8; the protocols, pre-registrations,
frozen inputs and results stay, and the deleted files are in Git at `1564dcd6`.

Measured on 2026-10-22 on `spark` (`spark-c4e2`, NVIDIA GB10).
The [registered protocol](PROTOCOL.md) passed both likelihood phases and
five paired behavioral controls. Both Fibonacci replies truncated at the
registered 256-token limit: each reply fails, and the comparative cell remains
unresolved. The ordinary native chat smoke passed. This evidence does not close
all of U3 or qualify speed, batching, full context or sustained serving. The
separate [512-token completion supplement](FIBONACCI512-RESULTS.md) passes its
new bound without changing the original failure.

## Provenance and executed configuration

The original protocol was committed as `e73c968b`. The independently reviewed
FP16 reference amendment was committed as
`483a5de9a56bb2cae6c2bd94cd7890c389373ec8` before any quality requests. Earlier
reference attempts failed during startup and supplied no quality scores.

| Identity | Value |
| --- | --- |
| Checkpoint | `nvidia/Llama-3.3-70B-Instruct-NVFP4@f792b707002f2c613e0729c6b37bbcedc4efa94e` |
| Native artifact | `9cca62f1e6582324de2ebce34f38b89a8aa8890cdced957a50782e407ff4d7ee` |
| Artifact manifest SHA-256 | `6d4b2ebc689516dc8da7a5586dafd7090cfd720452faeeca5f95ec003ec89c8c` |
| Artifact index SHA-256 | `f7009e3c654ca4664b2bf4413c00348f00c8c106f9c034f5424c73103c4fa46b` |
| Artifact spec SHA-256 | `9647472757577a29b988466202b27cecad070627d25f627a2210336a1694d912` |
| Native source base | `3965d69d439d1545b13bb77022a81ac45f8255df` |
| Native source patch SHA-256 | `da6310b3138153ac8f22049b0c0bfcb4b6a376363d71e91717a60d016e071405` |
| Native executable SHA-256 | `32e247723ebe2c00bcdae1cb1aff12884ce7fb7fb6b421e05e47eea9e1c124af` |
| Reference image | `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8` |
| Reference version | `0.1.dev20073+g8e685d198` |

The native source is the final wave-E snapshot: full copy accounting, seven-day
retention and private file pinning, without U4 or the newer swap implementation.
The remote checkout's `61f8426f...dirty` version banner does not identify those
synced sources; the base, patch and executable hashes above do. The tested
304,183,144-byte executable was preserved on `spark` beside `llmp-runtime` as
`build/spark-native/src/runtime/m35-u3-native-wave-e-runtime`; its SHA-256
matched and `ldd` reported no missing libraries.

Both arms used context 12,288, one request slot, explicit F16 KV, 512-row prefill
chunks, greedy sampling, and no drafter or optional lossy mode. Reference logs
confirmed `LlamaForCausalLM`, `dtype=torch.float16`, `kv_cache_dtype=float16`,
`quantization=modelopt_fp4`, `FlashInferCutlassNvFp4LinearKernel` and
FlashAttention 2 in all three phases. The pinned ModelOpt implementation maps
the checkpoint's `NVFP4` to W4A4. Its complete instanttensor 0.2.0 payload was
authenticated against [mia-launch.json](../fast-swap/mia-launch.json).
Reference prefix caching was disabled. Native likelihood requests each logged
zero cached prompt tokens.

Native used the component engine's ordinary production plan. Both likelihood
phases reported three measured plan shapes, with zero primitive or available
fast fallbacks, decode-expansion rules, missing numerical qualifications,
unqualified numerical candidates or omitted roots. Runtime coverage recorded
8,732 tensor uses for short and 78,588 for depth, with zero violations. No
per-operation kernel trace was collected; these are runtime plan/coverage
observations, not a claim that native used the reference's kernels.

## Fixed inputs and likelihood results

The 817,275-byte corpus is the existing `tools/k1_quality.py` default,
SHA-256 `55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb`.
The frozen 9,216-token JSON has BOS 128000 and SHA-256
`cb67d66726c438813f2f4378b9943e83e7db10b0fe80326cbf14e991ed558f9d`.
Both engines received these same integers without template rendering or
special-token insertion. Source tokenizer/config/template hashes and the
input reproduction helper are retained in [the protocol](PROTOCOL.md).

Each arm made two complete-history requests per phase. Both arms' own repeats
were byte-identical loss files: absolute PPL delta 0%, within the registered
0.1% bound. Comparisons used exactly native/reference pairs (1,1) and (2,2).
Both pairs have the values below; no replicate was selected or discarded.

| Phase | Input tokens | Scored positions (zero-based) | Targets per repeat | Native PPL | Reference PPL | Native PPL change | Both fixed pairs |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Short | 1,024 | 1–1,023 | 1,023 | 15.675666571184209 | 15.83861422148795 | −1.0287999191411101% | PASS |
| Depth | 9,216 | 8,192–9,215 | 1,024 | 18.361913645557717 | 19.34223550087691 | −5.068296553801909% | PASS |

The preregistered bound is **one-sided**: native PPL change,
`100 * (native PPL / reference PPL - 1)`, must be at most **+1.0%**.
These negative deltas pass that bound on the fixed samples; they do
not establish general quality superiority or numerical equivalence.
Actual-target probabilities use the full-vocabulary denominator, even where
the supplied target is not the reported top-one alternative.

| Phase | Native mean NLL | Reference mean NLL | Mean absolute target NLL difference | Maximum absolute difference | Maximum at (position, target ID) |
| --- | ---: | ---: | ---: | ---: | --- |
| Short | 2.752109610092306 | 2.76245089654198 | 0.1741651908419348 | 2.152723766989638 | (797, 2727) |
| Depth | 2.910278608821751 | 2.962291072842452 | 0.1857670299364484 | 3.7303835722940466 | (8558, 1715) |

The corresponding losses from different engines are not bit-identical.
Per-phase loss-file SHA-256 values below apply to both repeats in that arm.

| Phase | Native losses SHA-256 | Reference losses SHA-256 |
| --- | --- | --- |
| Short | `4d54af007254a2a3898d91b2f7a87e3aec1f719f37adf65104e4c2be9b144d78` | `4e7ccc06c9c9d0aae131eb30471c89f00eba5de82ccd323721e060c1d584d6a6` |
| Depth | `6c333fb836f298c0ebf93cbccd7d3545df911d2b86c0c1121fb0f5e7ecca1449` | `ff127a0fe58f98b1c9abfe570fe527c2f7a556019940ad6514dea00f888ae248` |

## Behavior and native chat

The six prompts are [fast-swap/prompts.json](../fast-swap/prompts.json),
SHA-256 `c697236c56a09b0a3f2550f7514b3e4d826e1d14a96b4d1c79e3bd33a3a6f859`.
The shared literal IDs were rendered once with the authenticated source
template; frozen input SHA-256
`ef8b44d0f97e7813edbc506d3105ec0dfd112de05efc4e4bbbe3de9158e9c960`.
Both arms used the registered 256-token allowance, temperature 0, top-p 1
and effective EOS. Every reply was reviewed against the original criteria.

| Control | Native | Reference | Evidence / comparative verdict |
| --- | --- | --- | --- |
| Capital | PASS | PASS | Paris in one complete sentence, without contradiction. |
| Haiku | PASS | PASS | Coherent three-line autumn-leaves poems, without garbling or repetition. |
| Sky | PASS | PASS | Two sentences explaining preferential scattering of shorter wavelengths. |
| Fibonacci | FAIL | FAIL | Both returned `finish_reason=length` at 256 tokens and cut off explanatory prose; comparative cell **UNRESOLVED**. |
| Primes | PASS | PASS | Exactly 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, comma-separated. |
| French | PASS | PASS | Faithful library translations retaining 09:00 opening and 18:00 closing. |

Both Fibonacci replies contained the same complete iterative function. The
reviewed function initializes `(a,b)=(0,1)`, handles 0 and 1 directly, then
updates `(a,b)=(b,a+b)` for `range(2,n+1)` and returns `b`. Executing that
function from each reply produced `F(0)=0`, `F(1)=1`, `F(10)=55`. The extracted
function SHA-256 is
`01d9b9bdfbf909602a994c85dc9d336315045f04398237b90dc115d625cb16b1`.
Those function checks do not override the registered nontruncation requirement.
Neither the token allowance nor the prompt was changed, and neither reply was
replaced by a later attempt.

A separate ordinary native `/v1/chat/completions` request asked for France's
capital in one sentence, with greedy sampling and at most 64 output tokens.
It returned Paris in one complete sentence with `finish_reason=stop`.
The runtime confirmed the sandboxed renderer used source template SHA-256
`e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65`.
This checks the renderer/route separately from the literal-ID behavior suite.

## Execution and reproduction

Each model run held the installed supervisor's exclusive GPU lock with a
600-second limit. All seven completed with supervisor exit code 0; every
reference container was removed and native retirement returned 0 after TERM.
The behavioral job's exit code confirms collection, not a semantic pass.

| Arm | Short job | Depth job | Behavior job |
| --- | --- | --- | --- |
| Reference | `m35-u3-llama-short-reference-r4` | `m35-u3-llama-depth-reference-r1` | `m35-u3-llama-behavior-reference-r1` |
| Native | `m35-u3-llama-short-native-r1` | `m35-u3-llama-depth-native-r1` | `m35-u3-llama-behavior-native-r1` |

The seventh was `m35-u3-llama-native-chat-smoke-r1`. Executable preservation
ran separately as the shared, 60-second `m35-u3-preserve-native-runtime` job
and completed with exit code 0. Spark A was released with no running/waiting
jobs or GPU compute processes.

Reproduce using the pinned checkpoint and native source snapshot above, the
committed [reference](reference.py), [native](native.py), input-preparation and
collection helpers, and the [protocol's launch instructions](PROTOCOL.md#running-the-registered-arms).
Stage the full checkpoint through `tools/stage-model` and verify its archive
manifest; the measured reference used the verified local copy. Apply only
file-scoped clean-cache advice after buffered preparation as recorded in the
protocol. Use fresh output directories. Compare repeats with
`tools/quality_pair.py pair`, first using the phase's `*-repeat-prereg.json`,
then each fixed corresponding pair using its `*-prereg.json`.

Raw responses, losses, logs, source/binary receipts and semantic reviews remain
outside Git in the experiment's `m35-u3-quality` scratch directory on `spark`.
This report retains the aggregates, identities, criteria and limitations needed
to assess the result without that raw bundle. No startup timing here is an
inference-speed qualification, and the unresolved Fibonacci cell is retained.
