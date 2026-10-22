<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama 3.3 Fibonacci completion supplement: 512 tokens

**Prospective; no supplement results yet.** This asks whether the original
Fibonacci request finishes coherently with a larger output allowance. The
512-token allowance was chosen **after observing** both engines' truncated
256-token replies. It is a new completion question, not an independent
confirmation of the original bound or a revision to that experiment.
The original [results](RESULTS.md#behavior-and-native-chat) remain **native
FAIL, reference FAIL, comparative cell UNRESOLVED**. Even a successful
supplement cannot relabel that cell or close all of U3.

Both original replies already contained a complete iterative function, whose
0/1/10 checks passed; explanatory prose was truncated. Doubling the allowance
is a bounded attempt to observe a natural finish. There is no escalation to
another allowance, prompt edit, model substitution, selective repeat or
quality exception in this registration. If 512 also truncates, retain it as
failure of this supplement too.

## Unchanged inputs and engines

Reuse the original six-case input file, SHA-256
`ef8b44d0f97e7813edbc506d3105ec0dfd112de05efc4e4bbbe3de9158e9c960`.
Its retained location is `/tmp/llmp-u3-original-proof/behavior-inputs.json`
on the preparation workstation; authenticate any execution-host copy by that
hash. Select only `cases[3]`, ID `fibonacci`, with its original **49 literal
IDs**, including BOS and source-template markers. `fibonacci512.py` checks
both the full file hash and the exact ordered ID list. Do not re-tokenize or
re-render. The user text remains:

> Write a Python function that returns the n-th Fibonacci number iteratively.

The request is the original `/v1/completions` body with only `max_tokens`
changed from 256 to 512: model `llama33`, temperature 0, top-p 1,
`add_special_tokens=false`, `stream=false`. EOS remains enabled through the
original engine/tokenizer configuration. No custom stop strings, ignored EOS,
new stop IDs, sampling penalties or token filters are introduced.

All checkpoint, tokenizer, source-template, archive, artifact and precision
pins from [the original protocol](PROTOCOL.md#immutable-inputs-and-reference)
and [result provenance](RESULTS.md#provenance-and-executed-configuration) remain
binding. In particular:

| Item | Frozen identity / setting |
| --- | --- |
| Checkpoint | `nvidia/Llama-3.3-70B-Instruct-NVFP4@f792b707002f2c613e0729c6b37bbcedc4efa94e` |
| Native artifact | `9cca62f1e6582324de2ebce34f38b89a8aa8890cdced957a50782e407ff4d7ee` |
| Native runtime SHA-256 | `32e247723ebe2c00bcdae1cb1aff12884ce7fb7fb6b421e05e47eea9e1c124af` |
| Native source | base `3965d69d439d1545b13bb77022a81ac45f8255df`, patch `da6310b3138153ac8f22049b0c0bfcb4b6a376363d71e91717a60d016e071405` |
| Reference image | `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8` |
| Reference model / KV dtype | F16 / F16, original ModelOpt NVFP4 W4A4 path |
| Native KV / plan | F16 / original ordinary production component plan; no drafter or optional lossy mode |
| Context / slots / prefill chunk | 12,288 / 1 / 512 |
| Reference loader pin file SHA-256 | `61732d66793a85ca140cb090a2c8aecad23da141eb4980f3cb8767953f3127a1` |
| Native Service helper SHA-256 | `0f1135a2283fe63797e15c8dffe48c9df39b0baae91a57f925b5ba21c3e3252e` |

The frozen executable is preserved on Spark A at
`/home/pmeenan/src/llmp/build/spark-native/src/runtime/m35-u3-native-wave-e-runtime`.
Copy that executable if required; never rebuild it from current source or
substitute the current `llmp-runtime`. Authenticate it before and after each
native launch. Record actual shared-library resolution and refuse missing
libraries. This is reuse of the pinned executable, not a claim that a current
source checkout/version banner identifies it. The native preflight verifies
artifact manifest/index/spec hashes, all four source metadata files, the
actual configured tokenizer file and source-template bytes and the exact single-model configuration; no drafter,
extra model setting or preexisting native state directory is accepted.

The reference reuses the original owned container launcher and exact server
arguments, including `--generation-config vllm`, disabled prefix caching and
F16/F16. The image and complete installed instanttensor payload remain
verified. Retain Docker inspection, startup log, source archive and helper
identities. Before sending the request, the collector checks the saved startup log for
actual loaded `LlamaForCausalLM`, ModelOpt FP4, F16/F16, disabled speculation
and prefix caching, `FlashInferCutlassNvFp4LinearKernel`, and FlashAttention 2. All actual architecture,
linear and attention selection records must agree: appended conflicting
selections or numerical/dispatch fallback records are refused;
a changed/fallback path is a diagnostic, not a qualified supplement.

## Four fixed independent observations

Run two fresh process launches per arm, exactly one complete-history request
per launch. The default is sequential native/reference jobs on `spark`, where the original
source and frozen executable already exist. If the orchestrator instead stages
and authenticates the native inputs on `spark-b` before the first request,
record that host assignment before collection and keep native on `spark-b`
for both repeats; reference stays on `spark`. The arms may then run concurrently
after each host is released. Repeat 2 always uses the same host as repeat 1. There is no timing comparison or AB/BA timing
protocol. Record host/driver/dependencies; no speed inference follows from
these different hosts or startup durations.

Use fresh output and native state paths for every run, with private 0700/0600
creation. Prefix caching remains disabled in the reference. Native starts
from a new state directory; any reported nonzero cached-prompt usage refuses
the collection. Fixed comparisons are native/reference repeats (1,1) and
(2,2), with both own repeats reported. Do not choose a favorable replicate.
Text equality and digest equality are reported, not acceptance requirements.

Each reply needs all of the following, independently assessed:

1. A structurally valid response to the 49-token prompt, one nonempty choice,
   consistent full token usage, and `finish_reason=stop`. `length` at 512 is
   recorded as truncation, never a completion pass. Missing/invalid replies
   fail their observation.
2. A coherent, complete answer to the iterative Fibonacci request, without
   truncated prose, repetition, contradictory explanation or broken code.
   A stop reason alone cannot establish this.
3. A complete iterative Python function with `F(0)=0`, `F(1)=1`, `F(7)=13`,
   and the original `F(10)=55` behavior. Before executing generated code, an
   independent reviewer inspects the complete reply and isolated function.
   Run only the reviewed pure function, with no imports, external access or
   unrelated response code, under a bounded local process (5 seconds) and
   `hostlock shared`. Retain the extracted function bytes/hash, inspection,
   actual inputs/outputs and execution receipt. Unsafe or non-iterative code
   fails; do not repair the model's code before checking it.

All four observations must pass for the supplement's paired completion result
to pass. An arm's inconsistent own-repeat verdict prevents a paired pass.
Any reference failure makes the corresponding comparison unresolved; it does
not pass native automatically. The collector deliberately emits
`semantic_verdict=pending_independent_review` even when finish is `stop`.
Collection success, function correctness and whole-answer completion are
separate reported facts. No PPL, model-wide quality, numerical identity,
speed, batching or maximum-context qualification is claimed.

## Owned, bounded execution

Independent review and a root pre-registration commit precede any request.
Keep every attempt, including startup/HTTP/parse/retirement failures. No output
path is reused. Infrastructure failure is retained and diagnosed before any
fresh attempt is authorized; it is never silently overwritten or dropped.
Use installed `spark-job --gpu --timeout 600 --grace 30`, one job per process.
Existing readiness is bounded to 420 seconds, the single request to 120 seconds,
and native retirement to 20 seconds before kill; the supervisor owns the total
lifetime. The reference's unique owned container has one bounded cleanup phase,
including signal/error paths. The first cancellation starts protected cleanup;
repeated TERM/INT/HUP cannot interrupt it. Logs have a two-second bound, removal
five seconds, and an exact-owned-name `docker ps --all` check one second. Log
open/command or removal errors do not bypass that check. Only a successful
empty listing proves absence; daemon errors/timeouts leave it unconfirmed.
The retirement receipt retains command outputs and failures. Any cleanup
failure fails the job even if the later check confirms absence. Confirm supervisor retirement
and no residual GPU work with `spark-job busy` before releasing either host.
Raw private outputs remain outside Git.

Prepare a new native configuration with the existing `prepare_serving.py`
using the verified local source and original artifact store. Use a different
configuration directory for each repeat so `storage.data_dir` is absent before
launch. Authenticate the generated recipe; this preparation does not run a
model. Example paths below are assigned at execution, not claims of existing
staged files:

```sh
# Assigned native host, repeat 1 (repeat 2 uses fresh config/output/job names).
~/.local/bin/spark-job start --gpu --timeout 600 --grace 30 \
  --name m35-u3-llama-fibonacci512-native-r1 -- \
  /home/pmeenan/.local/bin/mise exec -- python3 /ABS/HELPERS/native.py \
  --repo /ABS/PINNED_HELPER_REPO \
  --runtime /ABS/m35-u3-native-wave-e-runtime \
  --config /ABS/FRESH_NATIVE_CONFIG_R1/native.toml \
  --out /ABS/PRIVATE_RESULTS/fibonacci512-native-r1 --phase fibonacci512 \
  --behavior-inputs /ABS/behavior-inputs.json \
  --behavior-sha256 ef8b44d0f97e7813edbc506d3105ec0dfd112de05efc4e4bbbe3de9158e9c960
# Spark A, repeat 1; repeat 2 likewise uses a new job/output path.
~/.local/bin/spark-job start --gpu --timeout 600 --grace 30 \
  --name m35-u3-llama-fibonacci512-reference-r1 -- \
  /home/pmeenan/.local/bin/mise exec -- python3 /ABS/HELPERS/reference.py \
  --repo /ABS/PINNED_HELPER_REPO --checkpoint /ABS/VERIFIED_LOCAL_CHECKPOINT \
  --out /ABS/PRIVATE_RESULTS/fibonacci512-reference-r1 --phase fibonacci512 \
  --behavior-inputs /ABS/behavior-inputs.json \
  --behavior-sha256 ef8b44d0f97e7813edbc506d3105ec0dfd112de05efc4e4bbbe3de9158e9c960
```

Wait each job through the installed supervisor. The original `behavior` phase
still sends all six original prompts with 256 tokens; `fibonacci512` alone
selects the new single-request collector. Raw request/response bytes and their
hashes, finish reason, full usage, launch/provenance and retirement receipts
support independent review. Native records runtime/config pre/post hashes;
reference records its pinned image/loader and actual container. Do not infer
output token identity where the API returns only text and usage.

## Frozen supplement helper identities

Authenticate these files against this registration before launching either arm.
The launch receipts also retain the actual executed helper hashes. The original
collector and behavior input are unchanged; the helpers below add only the
separate supplement phase and its preflight/receipt checks.

| File, relative to this directory | SHA-256 |
| --- | --- |
| `native.py` | `e1a4f6fb9e75e689ef3e4b5b1bf2e2c038365a8efff4aeb15a0407623e754c98` |
| `reference.py` | `d528abd0a390c0f8293be279da0edb2710fb191791ba739e0e42e1b71b6ec6fe` |
| `fibonacci512.py` | `ea381a489a0b101ec02a5932f9a5588445741d5618be51910c7f610cdf45fa42` |
| `test_fibonacci512.py` | `a4330f59b5a2dc012580b6547175500e0650e99cdf273cd72e6c7d1c2144862e` |

Preparation validation: all 20 local contract cases passed under `hostlock`
(15 supplement, 5 original likelihood contracts), with no model or GPU work.
All three supplement-related CLI help commands and the whitespace check passed.
The original retained input authenticated as 49 exact Fibonacci IDs, and the
actual original reference startup log passed the new precision/dispatch check.
These checks qualify the collector contract, not a 512-token model response.
