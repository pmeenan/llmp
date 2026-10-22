<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prospective Gemma26 source-template Paris control

**GPU collection preregistration; no arm has run.** Input-freeze registration
`03655eaf` records the frozen input and intended comparison. The prospective
collection amendment below adds the reviewed helpers and an explicit replacement
runtime pin. It must be reviewed and committed before either arm launches.
CPU-only metadata preparation froze the input before registration, preserving
its first failed attempt below. This is one paired completion
control, not representative quality qualification or closure of U3. The
[original raw-ID diagnostic](../PROTOCOL.md) and all its negative observations
remain unchanged. Its executed reference was vLLM; Transformers supplied source
inspection and the tokenizer, not the model execution oracle.

## Question and immutable identities

The corrected native artifact previously answered the Paris chat, while native
and source vLLM both repeated on the five untemplated IDs without BOS. No further
native arithmetic defect or bad source checkpoint has been established. This
control asks whether both engines complete the existing Paris task when given
identical IDs rendered by the source's official chat template.

This prospectively replaces the earlier raw-ID protocol's requirement to freeze
native-renderer-derived IDs for a future Paris pair: both new arms instead
receive the source-template IDs frozen here as literal input. It does not claim
that these are the earlier native chat's IDs; that receipt recorded only their
count. The original raw-ID controls and results remain unchanged.

| Item | Identity |
| --- | --- |
| Source | `nvidia/Gemma-4-26B-A4B-NVFP4@a19cfe00be84568a6867111c9a68c9c44fdcffe6` |
| Archive manifest | `4f36e82e02c29b3e99530efa38495459667a4c461a4d03bc9f610664db162ca1` |
| Corrected artifact | `1d20ce879b2fe72c947699357da5e20d3f6388e9cc7858b9eb6fcc2516cd3875` |
| Earlier native screen runtime | `894ce8998791951fd9ae71afbb38cb263a384b726699369ce3afa9f770ca6e9b` |
| Reference image | `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8` |
| Source template | `94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413` |
| Tokenizer | `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` |
| Tokenizer config | `90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df` |
| Model config | `4e379cc809c617a49179a49140f553a2d6a5ec538ed480832b0c54f6ace43d98` |
| Generation config | `d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de` |

The collection amendment explicitly replaces the earlier native screen binary;
the new executable must be reauthenticated before launch. No artifact repair,
download, re-import, checkpoint transformation or backend substitution is part
of this control. Reuse the verified source already staged on Spark A at
`~/.local/share/llmp/models/nvidia/Gemma-4-26B-A4B-NVFP4@a19cfe00`.

## Official input preparation

`freeze_launch.py` runs once through installed `spark-job --shared --timeout
180`, with a 120-second container attachment bound. The immutable image uses
`runc`, `NVIDIA_VISIBLE_DEVICES=void`, no GPU devices, no network, two CPU cores
and at most 3 GiB RAM. It copies and authenticates only the six small files in
`freeze.py` into the mounted metadata directory. No weight file is mounted,
read or hashed. The launcher preserves failures and proves exact-name container
absence after removal; unsupported tokenizer APIs stop preparation without an
install, fallback, checkpoint-code execution or blind retry.

`AutoTokenizer.from_pretrained(local_files_only=True, trust_remote_code=False)`
uses the checkpoint metadata. Its actual `apply_chat_template` receives:

- System: `Answer only the requested capital city, without explanation.`
- User: `What is the capital of France?`
- `tools=None`, `enable_thinking=False`, `add_generation_prompt=True`.

The 178-byte rendering must have SHA-256
`d01c83b27d5600c694366037eeccd563440a46beac7555e70c186bbbc0b0651e`:

```text
<bos><|turn>system
Answer only the requested capital city, without explanation.<turn|>
<|turn>user
What is the capital of France?<turn|>
<|turn>model
<|channel>thought
<channel|>
```

The final line has no newline. `encode(add_special_tokens=False)`, the default
direct template result's documented `input_ids` projection, and explicit
`apply_chat_template(tokenize=True, return_dict=False)` must supply identical
complete integer lists, containing BOS 2 exactly once at the beginning. Freeze
the returned IDs and exact `input.json` SHA-256 before review/registration. Do not infer ID
identity from the earlier native chat's 35-token count. The receipt records the
actual tokenizer class, source module hash, package versions, helper hashes,
image and CPU visibility. Preparation provides no model-quality observation.

### First CPU preparation, retained failure (2026-10-22)

After the root agent independently reviewed both complete CPU helpers, Spark A
shared job `m35-u3-gemma26-paris-freeze-r1` logged start/end at 14:23:39/14:23:49
(host local time, America/New_York)
and returned 1. The official tokenizer loaded and its string rendering passed
the exact 178-byte/hash assertion. The subsequent separate `encode` completed,
but the additional equality assertion against direct
`apply_chat_template(tokenize=True)` failed. No input JSON, successful freeze
receipt, model construction or inference resulted. The first helper did not
save the two return values before asserting; the mismatch's type/content cannot
be diagnosed from that log, and no return-shape explanation is established.

All six copied metadata hashes and both helper hashes matched before/after;
container removal and successful exact-name absence verification passed. U8's
host owner was informed that preparation had retired. No retry was performed.
The preserved helper hashes are `freeze.py`
`dc4e3b20b2f224669d80260a1cb0dff852d054f9de48304938d6529235ca88fc`
and `freeze_launch.py`
`e54114bc612a750e8ef315d942d5632fc71565e765c8003ff8e2703d93d30e97`.

| Receipt | SHA-256 |
| --- | --- |
| Launch | `87cc7e658c92a3eef14e704d823d217275c4ae8ecf8229b21ceaed96f36c0e9c` |
| Container inspection | `80f779dd2e5ef11d0bfbf975de51c58af604f20d2bbd956d0e5c4997ea778134` |
| Container log | `989c24b82930336c89a8d77410a66e9143b5e74f74765666bf610d176d5cfad3` |
| Supervisor final | `5a6b685eeea2b7ff98a2ad156fb963507dd838012d3958cc0cb4e242a6673887` |

Raw receipts and the exact executed helper snapshot remain at
`spark:~/.local/share/llmp/m35-u3-quality/gemma26-paris-freeze-r1/` and
`/tmp/llmp-gemma26-paris-freeze-evidence-r1/` on the workstation; the latter
includes the supervisor receipt directory. The subsequent local diagnostic
repair saved both return types and complete representations before comparison.
The root agent then explicitly authorized the separate CPU-only evidence job
below on Spark B, coordinated with its shared-build owner. No automatic retry,
package installation or checkpoint-code execution occurred.

### Pinned API diagnosis and successful input freeze (2026-10-22)

Spark B shared job `m35-u3-gemma26-paris-freeze-r2` returned zero. Only the six
already authenticated r1 metadata copies were transferred from the workstation;
no checkpoint weights or source-tree changes were involved. The immutable image
contains Transformers **5.15.1** and tokenizers **0.22.2**, using built-in
`GemmaTokenizer`. These are the measured image versions, distinct from the
checkpoint's declared Transformers version used for the earlier RMS source audit.

The actual `apply_chat_template` signature and docstring declare
`return_dict: bool = True`. Its default result is
`transformers.tokenization_utils_base.BatchEncoding`, containing `input_ids`
and `attention_mask`. The recorded method source returns the mapping for
`return_dict=True` and `out["input_ids"]` otherwise. Thus the first helper's
comparison of a list with the default mapping was invalid; no reproducible
token-ID mismatch remains. The new helper recorded the signature, full method
source, complete default mapping, separate encoded list and explicit flat-list
result before asserting their contents agree. It selected `return_dict=False`
only after confirming that parameter in the actual signature.

All three paths produced these same 35 IDs:

```text
[2,105,9731,107,7925,1186,506,15633,5279,3207,236764,2180,15569,
236761,106,107,105,2364,107,3689,563,506,5279,529,7001,236881,
106,107,105,4368,107,100,45518,107,101]
```

The frozen [input.json](input.json) SHA-256 is
`3d0e98600de96f8a96f7bab3655d85b9d2e974c6e39b768d092e6fa1961069ec`.
It records the messages, unchanged template options, complete rendering and IDs.
Its only BOS is the template's initial 2. The full default attention mask is 35
ones. No tokens were substituted, removed or inserted to make the paths agree.
This freezes the prospective paired input; the old native chat receipt did not
record its complete IDs, so its equal count alone is not retrospective ID proof.

The [compact preparation receipt](freeze-receipt.json) records all metadata pins,
API options, package versions, helper and source hashes. Its SHA-256 is
`4e48911689fd7d0a584406d2242e47c811c8cebdf7e980b6c1ceb1e0f6b29997`.
The actual method's source file SHA-256 is
`520a914d0dcc873f8c8788e60ad3007f7db032c82c0b2b2c44ee7f44f2cdad8e`;
the tokenizer class source SHA-256 is
`ff2481e06983015188575dc62bd2fa0e6f33ed2d01165d940efe77b80ee8d419`.
Executed helper `freeze.py` SHA-256 is
`f8c21049c1e783ba865ac7bcc9bb9764e134b6b7bbfeeda3de4504dc52f7d8f2`;
the launcher remains `e54114bc612a750e8ef315d942d5632fc71565e765c8003ff8e2703d93d30e97`.

Container inspection confirmed runc/no network/no devices, and the helper saw
GPU visibility `void` with no NVIDIA device nodes. Both helper hashes and all
six copied metadata hashes matched before/after. Removal and a successful
exact-name absence listing confirmed retirement, and the B owner was notified
before its subsequent exclusive work. No model was constructed or inference run.

| External r2 receipt | SHA-256 |
| --- | --- |
| Launch | `a555144f66ada295d1e28a22da2dd63651068e0cebd76995ef0ffb7ee659b229` |
| Container inspection | `8a09bef7dde37c3502560111b6c52295899606253242c95652e28c504863067b` |
| Container log | `cc23026bc0dd642e4b49d69969edc4dc339c2d8f9b9679d6697d53c29eb9bd9f` |
| Complete API diagnostic | `e6e46e7bd2087470ff8acd985b57bc2022087a6676517f8b9749d38cc9b31b35` |
| Supervisor final | `f30663465f1972a9475467a1363d96e117c4d7aca5c5f8b4785f1591faeebe3f` |

Raw receipts remain at
`spark-b:~/.local/share/llmp/m35-u3-quality/gemma26-paris-freeze-r2/`,
`/tmp/llmp-gemma26-paris-freeze-evidence-r2/` on the workstation, and
`/tmp/llmp-gemma26-paris-freeze-evidence-r2-supervisor/` for its supervisor copy.
The first failure stays unchanged. GPU collection helpers must bind to the
frozen input and receive independent review as part of the separate prospective
GPU collection amendment, committed before either arm runs. The existing parent
directory's raw-ID helpers still name the five-token diagnostic and cannot run
this 35-token control unchanged. This successful metadata preparation is not a
model-quality result.

## Paired collection and acceptance

Both arms receive precisely the frozen integer list, without another template,
padding, truncation or special-token insertion. Each process handles one request.
Context 4096, prefill 512, one slot, no drafter, no optional lossy mode, no reused
conversation, temperature zero, top-p one, no top-k restriction, maximum 64
generated tokens. Reference prefix caching is disabled. Native F16 KV differs
from reference BF16 model arithmetic/BF16 KV; preserve and disclose that difference.

Reference execution reuses the authenticated existing raw-diagnostic source and
its complete actual dispatch checks: built-in `Gemma4ForCausalLM`, plain ModelOpt
NVFP4 W4A4, all 30 expert blocks without A16 fallback, and BF16 attention KV.
Record actual backend choices and retain the known W1/W3 scale warning; change
no scale or kernel. Use the previously authenticated instanttensor 0.2.0 loader,
V2 runner and container-local callback serialization setting. Reference stops
remain `[1,106,50]`; EOS remains active. Native uses the authenticated source
template's base stops plus tokenizer EOS, and records its actual output IDs
where the completion API supports them. An unexpected stop is unresolved
evidence, not permission to change stop handling.

Save requests and raw outputs before validation. Require exact reference prompt
IDs, matching native prompt count, complete output accounting, finite nonpositive
selected-token log probabilities and supported finish reasons. Source output
IDs include its terminating stop token. A native API refusal or score-accounting
mismatch remains a failed collection, not a reason to silently relax validation.

The task's answer criterion is **whitespace-trimmed text exactly `Paris`, normal
stop, and valid finite selected-token scores**. The answer comes from the existing
system/user request; the 64-token allowance is unchanged from its original
screen. No numerical likelihood tolerance is introduced. A length finish fails
even if the text contains Paris. Capture failures retain all raw evidence.

| Outcome | Interpretation |
| --- | --- |
| Both complete correctly | Reference usable for this bounded chat; broader source-quality qualification remains open |
| Only native fails | Concrete paired mismatch needing diagnosis |
| Only reference fails | Reference usability unresolved; no checkpoint-failure or native-superiority claim |
| Both fail | Shared input/reference assumptions remain unresolved |

The absence of BOS/chat in the old prompt is an observed difference, not a proven
sole cause. This control changes the entire source-template context, so success
cannot isolate BOS from role/turn/channel formatting. Preserve the original
five-ID failures and both source-launch records.

## Prospective collection and runtime amendment (2026-10-22)

The earlier `894ce899…` runtime has no known retained executable. This pair
prospectively uses Spark A's integrated runtime at
`/home/pmeenan/src/llmp/build/spark-native/src/runtime/llmp-runtime`, SHA-256
`d18c7b7a9c8c6cf383a6f607a8606d2ce2026c4b2a8cb47632ec424601704dff`.
`control.py` enforces that exact binary before and after collection; the older
screen and its binary identity remain historical evidence, not a paired result.

The replacement's frozen source manifest contains 2,828 files and has SHA-256
`d0c8f0023776b0994de9e97beff04cb7bcd3e77ddad78bf87390bb312794cff4`.
Its base is `a21f9c1edb2894cba031ce669a451c9b427b6545`, with integrated patch
SHA-256 `847571d9aee1c51263051508ea6fc3eb4718a48011e87c2671192446ef3f380e`.
The remote Git-derived build version still names stale `61f8426`; it is not an
authenticated source identity for this executable. The frozen source manifest
and patch above, together with A's measured binary hash, are the provenance
authority; the manifest's base does not mean an unmodified Git checkout was built.
The manifest/patch/freeze receipt are retained outside Git at
`/tmp/llmp-m35-orchestration/integration-wave-w6-r3/`. The launch records this
source identity alongside the binary hash. A's build completed all 203 native
test consumers and the CLI. Installed job `m35-w6-r3-shared-a` completed
3,599 ordinary tests on A with no failures or skips. Spark B's later source
differs only in the explicit 64-token fixture setting in
`model_settings_test.cc`; that case was excluded from A's set and passed after
correction on B, alongside 16 other focused passes. Independent reconciliation
against the native CTest exclusion inventory confirms 3,616 unique ordinary
passes, no missing cases, duplicate coverage or unresolved failures/skips. The
original fixture failure remains recorded. A's XML SHA-256 is
`fedb0f1460c0ef3fd1a2f0fe086bac481804379b054f82e9c4edb522cde1c76d`;
the combined audit is `root-combined-final-audit.json` beside the source freeze.
The production change subsequently landed in `b96e2ca6`.
The product sources are identical across those two snapshots, but this control
uses A's own binary hash, never B's executable as an interchangeable substitute.

`source.py` authenticates the unchanged parent `reference.py` before import,
then binds only its request-data `IDS` to the committed 35 integers. Its existing
engine setup, official parser/callback validation, live dispatch checks,
generation, complete returned-object capture and exact returned prompt-ID checks
run unchanged. `source_launch.py` reuses the authenticated parent launcher
`execute`/`retire` lifecycle and snapshots the new adapter plus the old reference
module; the explicit helper PYTHONPATH also makes that module available to the
worker callback. The loader and image pins remain those of the raw diagnostic.

`native.py` uses the authenticated `tools/speed_gate.py` `Service` owner with
a fresh state directory and loopback port 18976. It creates the registered
configuration, authenticates source/artifact metadata, then saves the literal
35-ID request before sending it to `/v1/completions`. Native source review
confirmed that this API preserves integer prompts with special-token insertion
disabled; the returned token count alone is not independent prompt-ID proof.
`return_tokens_as_token_ids=true` and `logprobs=1` expose each generated ID and
its selected-token score, including the final stop. Both collectors preserve raw
responses before validation, check stops and scores, and retain well-formed
negative answers with semantic `fail`. Both launch receipts bind the input,
protocol, registration commit and helper hashes; post-checks detect changed
protocol/helpers, native binary/configuration or reference loader snapshots.

Independent whole-helper review was clean. The reviewer reran all 11 contract
fixtures under the workstation's shared host lock and passed four additional
mocked native cases: semantic failure, malformed JSON, HTTP 500 and interrupted
startup, including raw capture and retirement. These are collector checks, not
GPU execution evidence. Review receipt SHA-256:
`6da9c6360facd7289264ef09870291aa9339e0c1924271c79dab70cefacb490a`,
at `/tmp/llmp-gemma26-paris-collection-review-r1.json`. The subsequent amendment
changes only the runtime/source pin and protocol-hash accounting; root review
verified that exact delta before registration.

| Collection helper | SHA-256 |
| --- | --- |
| `control.py` | `e01815d98d1395d631847211275e771ac208be7593689edba790564f00294ce9` |
| `native.py` | `d346e56948736820e77603d8978dc4cc9ba6b407cd7f70e08716b58cb67ca8d2` |
| `source.py` | `ad9bc4afe653ef1cb7a402853b58dd5fe952f2bc8a7a6c28dafb314445ab8cd4` |
| `source_launch.py` | `b8d22551b8053b783edf4c5d496a5221b2f6a867a46ef885cdf0fcf451b064ee` |
| `test_control.py` | `3a33edd7d8ce1db7604bde2db646bafa13fa401ff03738b9e9cd279a34a1948c` |

Run `native.py` and `source_launch.py` in separate exclusive 600-second jobs,
with fresh output directories and the full reviewed amendment commit supplied
as `--registration`. The orchestrator authenticates that commit's contents;
the helper's commit-syntax check is not proof of review. Source staging evidence
is the existing successful `m35-u3-gemma26-stage-r1/log`, SHA-256
`9fb846ebf68691cd63465c999d0a2e4ca9dd64037b09c2998bae230370a52593`.
No restage, re-import, source/template change, new allowance or numerical bound
is introduced. All original negative results remain intact.

## Execution ownership and retained evidence

After the separate GPU collection registration, the orchestrator assigns a free
Spark. Each model arm uses
installed `spark-job --gpu`, at most 600 seconds, with exclusive GPU admission and
waited retirement. No build, model test or measurement overlaps it. Fresh output
directories are mandatory. Reference container retirement uses removal plus a
successful exact-name absence listing; native uses the established `Service`
lifecycle and records TERM/exit status. A failure is retained and reviewed before
any new attempt. This file does not authorize new GPU work during preparation.

Record preregistration commit; protocol, input and helper hashes; image inspection;
source/archive, artifact and runtime identities; precision/dispatch; complete
request/response/score/finish data; and retirement receipts. Raw logs and container
outputs remain outside Git; only compact aggregate results and input provenance
belong here. Speed, batching, long context, likelihood and representative behavior
remain separate U3 work.
