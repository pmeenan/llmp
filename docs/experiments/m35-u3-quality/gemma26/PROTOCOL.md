<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 4 26B NVFP4 exact raw-ID source diagnostic

**Prospective; no model startup or reference request has run.** Independent
review and the pre-registration commit must precede the one model launch below.
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
NVFP4 BF16 restriction. This recipe pairs BF16 queries with BF16 keys. It has
passed API construction only; actual Gemma device support is unvalidated.
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
all frozen options, and actual `SamplingParams` accepts the request. Device
configuration resolution, loader startup and weights remain untested.

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
~/.local/bin/spark-job start --gpu --name u3-gemma26-source-raw-r1 --timeout 600 -- \
  python3 -B /ABS/REPO/docs/experiments/m35-u3-quality/gemma26/launch.py \
  --repo /ABS/REPO --checkpoint /ABS/VERIFIED_LOCAL_STAGE \
  --out /ABS/PRIVATE_RESULTS/gemma26-source-raw-r1 \
  --staging-evidence /ABS/STAGE_JOB_EVIDENCE --staging-sha256 ACTUAL_SHA256
~/.local/bin/spark-job wait u3-gemma26-source-raw-r1
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
