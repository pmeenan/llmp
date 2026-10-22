<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 4 26B NVFP4 exact raw-ID source diagnostic

**Diagnostic collected; no general quality qualification.** The original
registration `c696e743` reached startup but failed inspection before `generate`.
After reviewed transport amendment `b970961d`, the separately recorded second
launch returned the same 64-token repetition as native on the unchanged raw IDs.
See the [collected result](../GEMMA26-NORM.md#exact-source-raw-id-diagnostic).
The registrations below retain their prospective execution requirements.
This diagnostic asks whether the native five-token repetition also occurs in
the source engine. It is not a likelihood, speed or general quality gate and
cannot close U3. The [native norm report](../GEMMA26-NORM.md) preserves both
negative raw cells and the corrected artifact's positive Paris chat.

## Frozen comparison

| Item | Identity or setting |
| --- | --- |
| Source | `nvidia/Gemma-4-26B-A4B-NVFP4@a19cfe00be84568a6867111c9a68c9c44fdcffe6` |
| Archive | `/mnt/llm/checkpoints/hf/nvidia/Gemma-4-26B-A4B-NVFP4@a19cfe00be84568a6867111c9a68c9c44fdcffe6` |
| Archive manifest SHA-256 | `4f36e82e02c29b3e99530efa38495459667a4c461a4d03bc9f610664db162ca1` |
| Corrected native artifact | `1d20ce879b2fe72c947699357da5e20d3f6388e9cc7858b9eb6fcc2516cd3875` |
| Frozen native runtime SHA-256 | `894ce8998791951fd9ae71afbb38cb263a384b726699369ce3afa9f770ca6e9b` |
| Raw input IDs, in order | `[818,5279,529,7001,563]` |
| Input processing | Literal IDs; no BOS, special-token insertion, template, truncation or padding |
| Request | One sequence, greedy temperature 0, top-p 1, no top-k restriction, maximum 64 output tokens |
| Geometry | Context 4096; prefill chunk 512; one slot; no drafter or prefix caching |
| Native observation | ` is` repeated 64 times, 64 finite log probabilities, length finish |
| Reference arithmetic | BF16 model arithmetic, BF16 KV, checkpoint NVFP4 W4A4 experts |
| Native precision difference | Native screen used F16 KV; this is not a bit-identical arithmetic pair |

The image is the existing immutable
`sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`,
version `0.1.dev20073+g8e685d198`. Use its built-in
`Gemma4ForCausalLM` architecture override with the original nested config and
checkpoint. That wrapper reads `text_config`, remaps `model.language_model`
and individual expert/scales, and omits multimodal towers. No checkpoint code,
source edits, remote code trust or alternative checkpoint is permitted by this
recipe. Source and metadata SHA-256 pins are executable constants in
[reference.py](reference.py). In particular:

- Gemma implementation: `6074e4a2f0c40a5bb28d101060fb4fe64f11ac6ee0cb6f524a9bb3deaceab797`.
- ModelOpt implementation: `3f3ca743fd3c66d72be92b7544591a8632f1aa422b73bb50c83e8a3281196e7d`.
- Source config: `4e379cc809c617a49179a49140f553a2d6a5ec538ed480832b0c54f6ace43d98`.
- Tokenizer: `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.

`ModelOptNvFp4Config` explicitly supports BF16 and F16 activation dtypes;
checkpoint `quant_algo=NVFP4` selects its W4A4 path. The
[Llama precision amendment](../PROTOCOL.md#prospective-reference-precision-amendment-2026-10-22)
addressed BF16 queries paired with F16 keys in FlashAttention 2, not a general
NVFP4 BF16 restriction. This recipe pairs BF16 queries with BF16 keys. The first registered launch
completed initialization with those dtypes; its inspection RPC then failed
before generation. Per-layer dispatch validation and the raw request remain
uncollected.
Any unsupported startup is a retained failure and requires a reviewed,
prospective amendment before another attempt. Never silently choose FP16,
FP8 KV, W4A16, a new engine image or another attention/MoE backend.

Sampling ignores the source generation configuration's temperature/top-p/top-k
via `generation_config='vllm'` and explicit `SamplingParams`. Stops remain
explicitly `[1,106,50]`, from the authenticated source generation configuration:
`<eos>`, `<turn|>`, `<|tool_response>`. Source BOS is 2 and is never prepended.
The native code applies template base stops plus tokenizer EOS to literal
requests, excluding its additional chat markers. Its Gemma native template
variants declare `<turn|>` and `<|tool_response>`, compatible with this set;
the old native receipt does not record the actual selected template's complete
stop set. Therefore this static agreement is not a claimed measured stop-set
identity. If the reference stops on a token whose native effective handling is
uncertain, preserve that result and resolve the stop contract separately.

## Validated CPU contract and remaining startup checks

On `spark-b`, supervised shared job
`m35-u3-gemma26-token-source-inspect` extracted source from the immutable image;
`m35-u3-gemma26-token-api-check` then passed the helper's `validate` mode.
Both ran with `--runtime=runc`, `NVIDIA_VISIBLE_DEVICES=void`, network disabled,
no mounts or device access, no checkpoint and no model construction. Both
returned zero and successful exact-name container listings confirmed retirement.
External receipts are under
`/tmp/llmp-u3-gemma26-diagnosis-evidence/reference-inspection-r2` and
`reference-api-r1` on the workstation; Spark originals are
`/tmp/llmp-u3-gemma26-reference-inspection-r2` and
`/tmp/llmp-u3-gemma26-reference-api-r1`.

The actual official `TokensPrompt`, `parse_dec_only_prompt` and
`BaseRenderer._tokenize_singleton_prompt` APIs preserve exactly the five IDs.
A tokenizer tripwire raises on any encode/decode or tokenizer-attribute access;
none occurred. `TokenizeParams` disables special-token insertion and imposes
only the declared context/output lengths. This validates the raw-token branch,
not an invented mock implementation. Actual `EngineArgs` construction accepts
all frozen options, and actual `SamplingParams` accepts the request. That initial CPU validation did not test callback transport. The first
registered model launch subsequently passed device configuration and loading;
the amendment below records the additional CPU transport controls.

Review-only CPU source extractions, supervised shared jobs
`m35-u3-gemma26-review-inspect` and `m35-u3-gemma26-review-inspect-r2`,
also returned zero and proved exact-name container retirement under the same
no-device/no-mount/network-disabled restrictions. Their external receipts are
in `reference-review-inspection-r1` and `reference-review-inspection-r2` beside
the workstation receipts above. They confirm the actual linear selector's
Marlin/Humming W4A16 fallbacks and the scheduler's EOS/stop-token output contract;
the helper checks these before accepting a result.

`launch.py` authenticates and copies only the complete instanttensor 0.2.0
payload listed by `docs/experiments/fast-swap/mia-launch.json` (SHA-256
`61732d66793a85ca140cb090a2c8aecad23da141eb4980f3cb8767953f3127a1`).
The loader's license is Apache-2.0; the existing reference image and model retain
their recorded terms. This helper calls standard APIs and does not copy the
AGPL Mia recipe, apply its Qwen PLE patch, or use its Qwen KV-byte estimate.
`VLLM_USE_V2_MODEL_RUNNER=1` selects the existing fast-start runner. No pin refresh.

Before the request, `LLM.apply_model` captures the actual loaded model,
quantization methods, expert kernels/backends and attention backends/cache dtype.
The helper requires the Gemma causal wrapper, `ModelOptNvFp4Config`,
`quant_method=NVFP4`, the W4A4 linear method, all 30 NVFP4 expert blocks with
`use_a16=false` and no Marlin fallback, and all 30 attention layers with BF16 KV.
Every NVFP4 linear module must report its actual kernel and may not use the
Marlin or Humming W4A16 kernels, even under the nominal W4A4 method class.
Declared unquantized checkpoint modules are legitimate; this is not a demand
that every dense projection be NVFP4. Missing or unexpected dispatch evidence
stops before collection. The official loader resolves the exact staged source;
startup refusal is preserved, not converted into a result.

After generation the helper saves the complete returned objects, then asserts
`RequestOutput.prompt_token_ids == [818,5279,529,7001,563]` exactly. It records
all generated IDs, actual selected-token and returned alternative log
probabilities, text, finish/stop reason and counts. Missing actual-token scores,
nonfinite/positive scores, incomplete requests, changed prompt IDs or invalid
length/finish/stop-token combinations fail the diagnostic contract. Repeated nonsense is
valid captured evidence, never a passing quality verdict.

## Prospective callback-transport amendment (2026-10-22)

The registered `m35-u3-gemma26-source-raw-r1` launch on `spark` completed
BF16/BF16-KV initialization from the exact staged source. Startup logs report
17.07 GiB for model loading and select `FLASHINFER_CUTLASS` for NVFP4 MoE and
`TRITON_ATTN` for Gemma's heterogeneous attention head dimensions. These are
startup observations, not a speed result or completed per-layer dispatch audit.
The source also warned that `w1_weight_scale_2` and `w3_weight_scale_2` differ
and accuracy may be affected. Preserve that observation; this amendment does
not modify scale handling, weights, kernels or arithmetic.

The next operation, `LLM.apply_model(inspect_model)`, failed in the pinned
`v1/serial_utils.py` encoder with `TypeError: Object of type <class 'function'>
is not serializable`, identifying `VLLM_ALLOW_INSECURE_SERIALIZATION=1` as the
supported callable transport. The exception occurred before `LLM.generate`.
The result is `failed`, collection is false, and successful exact-name listing
confirmed container retirement. It supplies no reference raw output, likelihood
or quality evidence. Retain the entire first directory unchanged:
`spark:~/.local/share/llmp/m35-u3-quality/gemma26-source-raw-r1` (workstation
copy `/tmp/llmp-m35-orchestration/gemma26-source-raw-r1`).

Source inspection confirms `LLMEngine.apply_model` passes our callable as the
argument to `collective_rpc("apply_model", ...)`. The encoder's supported
function path uses cloudpickle only when the named environment flag is enabled;
the decoder's corresponding extension handler has the same gate. The
amendment sets `VLLM_ALLOW_INSECURE_SERIALIZATION=1` **only in this owned
reference container**, whose network remains disabled and whose helper is an
authenticated read-only snapshot. It serializes llmpalooza's own registered
inspection function; `trust_remote_code=false` remains in force and no
checkpoint code is executed. No host-wide environment setting or source patch
is made. Prompt IDs, stops, sampling, 64-token limit, BF16/BF16 precision,
image, loader, kernel selection and the actual W4A4 dispatch gates are unchanged.

The helper now tests this callback route before constructing `LLM`. Its CPU
validation uses the actual `MsgpackEncoder` and `MsgpackDecoder` to transport
the registered `inspect_model` callable and RPC arguments, compares the restored
code object, invokes it through the official `WorkerBase.apply_model` method
on a tensor-free metadata fixture, and round-trips the returned inspection via
the official `UtilityResult` codec. The fixture exercises expert/kernel and
attention provenance fields. A separate process with serialization disabled
must reproduce the original function refusal. These checks validate the owned
codec and callback contract without constructing a model or making a device
call; they do not claim a live model-worker RPC pass.

CPU transport validation passed on `spark` in shared supervised job
`m35-u3-gemma26-rpc-api-r1`: the flag-off process reproduced the function
refusal; the flag-on process transported 2,385 bytes, preserved the callback
code, passed official worker invocation and returned equal utility metadata.
The existing actual token/parser/renderer and argument checks passed again.
Both immutable-image containers used `runc`, GPU visibility `void`, no network,
no mounts and no model construction; both returned zero and were confirmed
absent. Tested helper SHA-256:
`847191af80ee1cead7979e06d825fc6460e1e73a408e6a415c48c9bc2c28d7ef`.
Raw receipts are outside Git at `spark:/tmp/llmp-u3-gemma26-rpc-api-r1` and
`/tmp/llmp-gemma-rpc-evidence/api-r1` on the workstation. Source inspection
job `m35-u3-gemma26-rpc-source-r1` likewise passed and retired. All eight
result/dispatch/retirement fixtures also pass under the workstation's shared
host lock. No second model launch or reference request ran during this repair.

Additional executable source pins are:

| Source | SHA-256 |
| --- | --- |
| `v1/serial_utils.py` | `d75c17a098231b2398cbdb7da6f9983e5b9229c0ba138b2874934d45fafd1f28` |
| `envs.py` | `fb701e932ff263612d520c1447dfdbd882da3286c889c4e01735c8456c3db508` |
| `v1/engine/llm_engine.py` | `415b7d02460984e8678f78420be22dbc336d8bb73734d7a753b2badb62232744` |
| `v1/worker/worker_base.py` | `b0684c84f2f2c4bff2d7f0574e13f48c01b4d76dbf555e7c52b32e9a702a9061` |

The official stage is already complete: job `m35-u3-gemma26-stage-r1` verified
12 files / 18,825,679,760 bytes, with log SHA-256
`9fb846ebf68691cd63465c999d0a2e4ca9dd64037b09c2998bae230370a52593`.
Reuse the existing source at
`spark:~/.local/share/llmp/models/nvidia/Gemma-4-26B-A4B-NVFP4@a19cfe00`;
its manifest retains the identity above. No second stage or peer transfer is
needed for the next launch on `spark`.

After review and the amendment commit, coordinate the host and run the same
bounded launcher with a **new** job `m35-u3-gemma26-source-raw-r2` and a fresh
`~/.local/share/llmp/m35-u3-quality/gemma26-source-raw-r2` output directory.
Supply the verified stage job log and its SHA-256 above. Use the same 600-second
exclusive supervisor and 510-second inner execution deadline shown below.
Do not delete or overwrite the first failure. If the live callback or dispatch
gate still fails, preserve that failure and stop before collecting requests.

## Bounded execution after review and pre-registration

Coordinate host ownership with the orchestrator. No helper source sync, stage
job, model launch or timing may overlap another owner's host assignment. Source
sync uses `rsync -rlpc --exclude=.git --exclude=/build`; never copy build trees.
There is no C++ change or native build needed for this source-only diagnostic.

First check for an existing verified local stage of this exact revision. If none
exists, stage once on `spark`, then peer-copy over the direct link if execution
will use `spark-b`; retain the successful supervised job log and archive stage
event. The source is already archived; do not download it again. For example,
on the assigned staging host, from the synchronized source tree:

```sh
~/.local/bin/spark-job start --shared --name u3-gemma26-stage --timeout 600 -- \
  ~/.local/bin/mise exec -- tools/stage-model \
  nvidia/Gemma-4-26B-A4B-NVFP4@a19cfe00be84568a6867111c9a68c9c44fdcffe6
~/.local/bin/spark-job wait u3-gemma26-stage
```

If a peer is needed, use the documented `--peer`/`--peer-address` options in the
[archive workflow](../../../model-archive.md#staging-to-the-sparks), on that same
verified stage. A new 600-second staging attempt is preparation, not inference;
a timeout is reported and handled under the existing resumable stage procedure.
The expected default local directory is
`~/.local/share/llmp/models/nvidia/Gemma-4-26B-A4B-NVFP4@a19cfe00`;
verify the actual stage result instead of assuming that path exists.

The orchestrator reviews the successful staging evidence's model, revision,
path and full-payload verification, then supplies its actual path and SHA-256 to
`launch.py`. The launcher copies and authenticates that receipt/log but does not
infer its success from an arbitrary file's hash. It also authenticates the exact
archive and metadata and checks both shard sizes. It does not rehash all model
bytes immediately before inference or claim that metadata hashing does so.

Launch exactly one raw-ID request, no chat request, in one installed exclusive
job. Substitute actual absolute paths and the SHA-256 of reviewed successful
staging evidence below; these are arguments, not new model or input choices.

```sh
~/.local/bin/spark-job start --gpu --name m35-u3-gemma26-source-raw-r2 --timeout 600 -- \
  python3 -B /ABS/REPO/docs/experiments/m35-u3-quality/gemma26/launch.py \
  --repo /ABS/REPO --checkpoint /ABS/VERIFIED_LOCAL_STAGE \
  --out /ABS/PRIVATE_RESULTS/gemma26-source-raw-r2 \
  --staging-evidence /ABS/STAGE_JOB_EVIDENCE --staging-sha256 ACTUAL_SHA256
~/.local/bin/spark-job wait m35-u3-gemma26-source-raw-r2
~/.local/bin/spark-job busy
```

`launch.py` refuses a reused output directory, snapshots authenticated helpers
and loader files, records Docker inspection and before/after helper hashes,
and streams startup/output logs to private external storage. Creation/inspection
failure streams remain in the launch receipt. Docker runs with network disabled
and a read-only checkpoint. Attached execution has a 510-second
inner deadline within the 600-second supervisor; container creation/inspection
and retirement are separately bounded. On success, exception, timeout or TERM,
it force-removes its unique owned container and requires a successful exact-name
listing with no remaining container. A cleanup failure is reported before any
new job. Keep full logs, raw results and receipts outside Git; record compact
findings and provenance in the existing norm report after review.

## Interpretation fixed before collection

- **Same raw repetition:** the source engine shares the observed symptom on
  this input. This supports an input/source-behavior explanation but does not
  prove the prompt is malformed or that native arithmetic is correct generally.
- **Different source behavior:** a remaining cross-engine disagreement exists
  under the disclosed BF16/F16-KV difference. Investigate input/stop handling and
  local arithmetic/graph stages; do not attribute the cause from text alone.
- **Startup or contract failure:** no valid reference comparison exists. Keep
  failure evidence; resolve capability or helper defects prospectively before a
  new reviewed registration. No blind retry or adaptive prompt/precision change.

The native negative cell remains negative in all cases. Do not add BOS, lengthen
64, swap prompt text or substitute chat to erase it. A future paired Paris
control requires freezing the **actual native-renderer-derived IDs** and
comparing source rendering before either arm; the old usage count of 35 is not
an input fixture. General source-quality, batching, context and speed gates are
separate work.
