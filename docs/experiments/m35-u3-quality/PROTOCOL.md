<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# U3 Llama 3.3 NVFP4 prospective quality protocol

**No governed quality requests have run or quality scores been collected.**
The original pre-registration is `e73c968b`; its reference startup attempts
loaded weights and reached warmup but never became ready. Independent review
and the main agent's amendment pre-registration commit must precede inference
with the FP16 reference recipe below. This is a bounded
same-checkpoint quality screen. Successful publication or a 64-token smoke is
not its quality verdict. Speed, batching, full-context and sustained serving
qualification remain separate.

## Immutable inputs and reference

- Checkpoint: `nvidia/Llama-3.3-70B-Instruct-NVFP4` revision
  `f792b707002f2c613e0729c6b37bbcedc4efa94e`.
- Native artifact: `9cca62f1e6582324de2ebce34f38b89a8aa8890cdced957a50782e407ff4d7ee`.
- Config SHA-256: `e7cbee89103cc488ef55c142c2a2a376ea00ce67b49c3ae9282d944107d97c33`.
- Tokenizer SHA-256: `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b`.
- Tokenizer-config SHA-256: `5c756ad6048c884974a331963ae3c3e1624b548a11174dadda02a69041c87352`.
- Source chat-template SHA-256, extracted from tokenizer config:
  `e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65`.
- Generation-config SHA-256:
  `b52d7103195cf7520887804dc703c715515dca31c7a2db20afdb2bf8f78203f0`.
  Its defaults are temperature 0.6, top-p 0.9 and sampling enabled; this protocol
  explicitly overrides them with greedy temperature 0 and top-p 1.
- Llama 3 rotary parameters: factor 8, low/high frequency factors 1/4,
  original context 8,192, base 500,000. These come from the pinned config.
- Reference image: `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`.
  Use its `LlamaForCausalLM` and plain `ModelOptNvFp4Config`, with the complete
  installed instanttensor 0.2.0 payload authenticated by
  `docs/experiments/fast-swap/mia-launch.json`.

`reference.py` uses `sudo -n docker` and CDI `nvidia.com/gpu=all`. It invokes the
pinned generic API server directly. The Qwen-specific Mia recipe's PLE table,
fixed KV byte estimate and memory subtraction do not apply to this checkpoint.
The inspected image has Llama registration, plain NVFP4 configuration and an
SM120+ NVFP4 linear implementation; source inspection is not an execution pass.
`validate_reference_cli.py` must pass against this exact image before inference,
retaining the full CLI help, parsed options, version and parser source hash.

Record the source archive manifest, source/config/tokenizer identities, native
source/binary/artifact identities, actual loaded quantization and kernel choices,
precision and cache dtype, Docker inspection and launch logs. A dispatch or
encoding fallback is a diagnostic, not permission to silently change the oracle.

## Geometry and acceptance

Both arms use one request slot, context 12,288, explicit F16 KV and 512-row
prefill chunks. The reference uses FP16 model arithmetic and the checkpoint's
NVFP4 W4A4 path. Native uses its ordinary production plan and records the actual
qualified choices; lossy optional modes and drafters are off. Prefix caching is
off in the reference; both collectors submit independent complete histories.
These chunk settings control arithmetic for this screen, not competitive tuning.

### Prospective reference precision amendment (2026-10-22)

The original BF16-arithmetic/F16-KV recipe failed during startup warmup in the
pinned image's FlashAttention 2 implementation: `query and key must have the
same dtype`. Its `flash_attn.py:1176` passes the model query and stored keys
directly to that kernel. No likelihood request or behavioral collection ran.
The revised recipe sets `--dtype float16`, retaining explicit F16 KV. This is
a supported standard model precision in the pinned `ModelOptNvFp4Config`
(`modelopt.py:1038` accepts `torch.half`); the checkpoint's `NVFP4` still selects
the W4A4 `ModelOptNvFp4LinearMethod`, not W4A16. This changes reference model
arithmetic prospectively. Native execution, the checkpoint and loader/image
pins, remaining CLI settings, inputs and acceptance bounds are unchanged.
Parser validation and source support do not yet qualify the revised execution.

Two environment issues preceded the dtype failure on `spark`. Instanttensor's
default staging budget is half of CUDA-reported free memory: the first attempt
allowed 991,371,264 bytes, below its required 2,101,346,304-byte embedding/head
buffer. File-scoped `POSIX_FADV_DONTNEED` on the four verified inactive artifacts
from the import parity proof released clean cache, following RE-041; the
unchanged launch then loaded all weights. NAS weight loading took 364.50 s and
the 420 s readiness deadline expired during subsequent startup. Official
`stage-model` staging verified the complete pinned revision locally; weight
loading from that copy took 4.76 s before the dtype failure. These are startup
diagnostics, not inference-speed or quality results. All attempts retired.

Use the verified local checkpoint at
`~/.local/share/llmp/models/nvidia/Llama-3.3-70B-Instruct-NVFP4@f792b707` on
`spark`; its archive manifest is byte-identical to the NAS revision's. After
buffered preparation, release only the verified inactive source/artifact
files' clean cache before the exclusive launch, recording paths and memory
before/after; do not change memory guards or flush global caches. Startup logs,
cache-preparation and staging receipts, and pinned-source inspection hashes
remain outside Git under `spark:~/.local/share/llmp/m35-u3-quality/` and the
supervised jobs `m35-u3-llama-short-reference{,-r2,-r3}` and
`m35-u3-stage-llama33-reference`.

The retained source corpus is the existing `tools/k1_quality.py` default,
`~/.local/share/llmp/corpus/m3-reference-corpus.txt` on spark: 817,275 bytes,
SHA-256 `55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb`.
The pinned-image inputs-only preparation froze 9,216 IDs, with BOS 128000:
SHA-256 `cb67d66726c438813f2f4378b9943e83e7db10b0fe80326cbf14e991ed558f9d`.
`prepare_quality_inputs.py` authenticates the corpus and tokenizer and reproduces
that JSON input. Both engines receive the same integers with special-token
insertion disabled. No template is applied to the corpus.

| Phase | Input IDs | Scored positions, zero-based | Targets | Cross-engine acceptance |
| --- | ---: | --- | ---: | --- |
| Short | first 1,024 | 1 through 1,023 | 1,023 | native PPL increase <= 1.0% |
| Depth | first 9,216 | 8,192 through 9,215 | 1,024 | native PPL increase <= 1.0% |

Separate JSON preregistrations freeze those counts and bounds. Run two independent
requests in each engine. Each engine's own-repeat absolute PPL delta must be
<= 0.1%, as fixed in its repeat preregistration. A repeat failure diagnoses noise;
it does not widen the cross-engine allowance. After own repeats pass, compare
native repeat 1 with reference repeat 1, and repeat 2 with repeat 2: both pairs
must pass. Do not select the better replicate. Report aggregate and per-target
NLL/PPL deltas, including maximum and mean absolute NLL differences.

The 1% bound is a new prospective, conservative screening choice, stricter than
the historical 3% screen. No historical Qwen near-tie or PPL allowance is silently
transferred to this new oracle. The separate depth phase exercises positions
past the source's original 8,192-token context. A short pass does not waive it.

## Actual-target likelihood contract

`collect_quality.py` sends literal `/v1/completions` requests with fixed IDs,
`max_tokens=0`, `echo=true`, `logprobs=1`, `prompt_logprobs=1`, temperature 0 and
`add_special_tokens=false`. It takes the supplied target's key from each
`choices[0].prompt_logprobs` row, even when the target is not the highest-ranked
alternative. The reference's `--logprobs-mode raw_logprobs` and native's existing
full-head scored reducer supply full-vocabulary probabilities; reporting one
alternative does not renormalize the softmax. The collector rejects missing
actual targets, non-finite/positive scores, count/ID mismatch and missing BOS
sentinels. It saves raw responses and receipts outside Git and emits the
`tools/quality_pair.py` target-loss format. Contract fixtures cover non-top-one
targets, depth coordinates and invalid responses.

Use `tools/quality_pair.py pair` with the phase's own-repeat preregistration for
each engine's two loss files, then with the phase's cross-engine preregistration
for each fixed corresponding pair. No metrics are inferred from generated text.

## Behavioral and serving controls

The six prompts are `docs/experiments/fast-swap/prompts.json`, SHA-256
`c697236c56a09b0a3f2550f7514b3e4d826e1d14a96b4d1c79e3bd33a3a6f859`.
`prepare_behavior.py` uses the authenticated source tokenizer and template,
`add_generation_prompt=true`, to freeze shared literal prompt IDs. The frozen
input file SHA-256 is `ef8b44d0f97e7813edbc506d3105ec0dfd112de05efc4e4bbbe3de9158e9c960` (six cases). `collect_behavior.py` submits
those IDs to either serving endpoint, greedily, allowing 256 output tokens so
the code and translation requests can finish. EOS remains effective. This
qualifies model behavior with the source template, not the runtime renderer;
a separate ordinary chat smoke checks that serving integration.

Review every reply from both engines against these fixed semantics:

- Capital: Paris, in one sentence, with no contradictory claim.
- Haiku: a coherent three-line autumn-leaves haiku, without garbled/repeated text.
- Sky: two sentences correctly explaining preferential scattering of shorter
  visible wavelengths, with no false claim that the ocean causes the blue sky.
- Fibonacci: an iterative Python function computing F(0)=0, F(1)=1 and F(10)=55;
  record the evaluated function and actual checks if claiming functional success.
- Primes: exactly 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, separated by commas.
- French: a faithful French translation preserving library, 09:00 opening and
  18:00 closing. Wording need not match bit for bit.

A truncated, missing or invalid reply fails its control. Record each semantic
verdict and reason; the collector intentionally leaves verdicts pending review.
A reference failure makes that behavioral cell unresolved, not an automatic
native pass. No token-equality gate or post-hoc prompt substitution is used.

## Bounded execution and remaining scope

Each model launch/collection runs alone through installed `spark-job --gpu`,
timeout at most 600 seconds. `reference.py` owns a unique container and retires it
on completion, error and SIGTERM. Its readiness deadline is 420 seconds, with the
outer job governing total lifetime. If the actual profile needs longer, state the
measured need before a new longer supervised job. No inference runs during
imports, digest proof, builds or another exclusive job.

The second-architecture MLX native artifact is
`ca25eee473b02f4a13fe1d8680dea5f2aa244b556d26ee50eaa0b62b8719d3b5`, from
`TensorFold/Qwen3.8-27B-MLX-4bit@70ae7fac63274ff2eac54152031433374cb80f2f`.
`prepare_serving.py` authenticates its actual tokenizer/template/generation
metadata too. Its serving smoke and independent reference qualification remain
separate. Historical clean TensorFold `71377a5373ed7b394f1b480ba2a6a3986b03af1c`
has a CUDA MLX `qwen3_5` implementation; the milestone's recorded
`cb2ebf0540f42604e2759b2ddef497861e928248` pin must be staged/inspected before new
reference qualification. Do not silently substitute the historical reference.

## Running the registered arms

After review and the pre-registration commit, run `prepare_serving.py llama33
CHECKPOINT CONFIG_DIR --store ARTIFACT_STORE` on the execution host. It checks
all pinned tokenizer/config/generation files, extracts the source template and
writes a native config with a private data directory and loopback port 18974.
Choose an unused output directory for every arm/phase. Run through the installed
supervisor, for example:

```sh
~/.local/bin/spark-job start --gpu --name u3-llama-short-native --timeout 600 -- \
  python3 /ABS/HELPERS/native.py --repo /ABS/REPO \
  --runtime /ABS/REPO/build/spark-native/src/runtime/llmp-runtime \
  --config /ABS/CONFIG_DIR/native.toml --out /ABS/RESULTS/native-short \
  --ids /ABS/INPUTS/ids.json --phase short
~/.local/bin/spark-job wait u3-llama-short-native
```

`native.py` uses `tools/speed_gate.py`'s existing `Service` lifecycle and retains
the runtime/config identities and waited retirement. Both helpers collect the two
own repeats within the selected phase. Replace `--phase short` with `depth` only
after the short phase passes. The reference equivalent is `reference.py
--checkpoint CHECKPOINT --repo REPO --out RESULT_DIR --ids IDS --phase short`,
under a separate exclusive 600-second job. Its port 18973 must be unused.

For behavior, select `--phase behavior --behavior-inputs FILE
--behavior-sha256 FROZEN_SHA` on either launcher. The input file is prepared in
the pinned image using `prepare_behavior.py`; its hash is frozen before either
arm runs. All paths in the receipts are actual host paths, not invented staged
locations. Generated inputs, raw responses, logs and receipts stay outside Git.

## Preparatory validation receipt

The FP16 amendment's exact pinned-image parser and complete CLI help passed on
`spark` in `m35-u3-fp16-cli-amendment` (21 s), with parsed model dtype `float16`
and KV dtype `float16`. No model loaded. The image/version and parser source
hash below are unchanged; receipts and full help are at
`spark:~/.local/share/llmp/m35-u3-quality/cli-fp16-amendment/`. Static parsing
of the staged helpers and all five collector contract controls also passed.
This validates the amended CLI, not model execution or quality.

The exact pinned-image parser and complete CLI help passed on spark in `u3-quality-cli-final`, step 1 (15 seconds); no model loaded. The supported request-log flag is `--no-enable-log-requests`. Parser default construction requires visible device metadata on this image, so the parser/help-only container used CDI visibility under an exclusive lock, without inference. vLLM version `0.1.dev20073+g8e685d198`; CLI parser source SHA-256 `5b6c1c61bc9d25c703086b92a3e0040d21cc054305d9a0fbe69767dfd26af646`. Full help and parsed option receipts remain outside Git in `/tmp/llmp-u3-original-proof/cli-validation`.

Behavior fixture preparation passed in `u3-quality-behavior-final` (7 seconds), using the image without a GPU. Explicit `return_dict=False` pins the Transformers chat-template return shape. Frozen file is `/tmp/llmp-u3-original-proof/behavior-inputs.json`; quality IDs are `/tmp/llmp-u3-original-proof/llama33/inputs/ids.json`. Both raw input files remain outside Git. Collector contract tests passed (five cases). None of these preparations loads model weights or qualifies execution/quality.

The reviewed cleanup repair passed three local, mocked lifecycle controls: log-open
failure and log-command timeout each still attempt container removal; a reused
output directory is refused before launch. All five collector contract controls
passed again after review. These checks perform no model inference.
