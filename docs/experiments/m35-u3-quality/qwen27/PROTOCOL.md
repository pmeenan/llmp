<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prospective Qwen27 MLX source-quality screen

**Inputs are frozen. Run only after this protocol has received independent
review and been committed. No Qwen source-quality results exist yet.**
`prereg.json`'s `frozen` status means its input identities are complete, not that
a preregistration commit or model validation has happened. The actual reviewed
commit must be recorded in each later receipt. The source adapter validation
below precedes governing collections. A mismatch stops qualification; it does
not authorize changing the head implementation, precision, prompt, or bounds.

This is a separate Qwen protocol. The earlier [Llama results](../RESULTS.md),
including its unresolved 256-token Fibonacci control, remain unchanged. The
latest Qwen gate-repair smoke used the original 64-token limits: literal text
became coherent, while ordinary chat exhausted the limit in coherent reasoning
with empty answer content. That diagnostic is not a quality qualification.

## Immutable identities

| Item | Identity |
| --- | --- |
| Source checkpoint | `TensorFold/Qwen3.8-27B-MLX-4bit@70ae7fac63274ff2eac54152031433374cb80f2f` |
| Archive manifest SHA-256 | `e64f621c146c3b14ad7023eab05a476acaa663f1d9ba97a57a7768986d4d6a66` |
| TensorFold source | `cb2ebf0540f42604e2759b2ddef497861e928248` |
| Source image, linux/arm64 | `sha256:1010df9519efb5fb2ca8a09c1504e95829206e4fff8c8e8f3b671b4d71a6737a` |
| Parent image | `nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c` |
| Native artifact | `bbf67ee688746792a1fbb06552b59bc0132e74bc79ced2333302b431549d0fa5` |
| Native runtime SHA-256 | `078a8da5a78242477891757b88675ed685b4ebe6d687db634c8c116cef1f3181` |
| Native manifest SHA-256 | `0e0e325360f02b4a678eab55c4cb9847538a849695f5c1c92c9902369f56b1bd` |
| Native ModelSpec SHA-256 | `3930c5bc8c74c5ddb2a3b043e1011c6dc15d82bb04e87e3e359d235dc8a3ec2a` |

The source image was built with the existing
[Dockerfile.tensorfold](../../fast-swap/Dockerfile.tensorfold) and exact source
revision. It contains TensorFold 0.6.6, torch `2.13.0a0+9186a08b2c.nv26.7` and
Triton `3.7.1+gitf797708c.nv26.7`. The 449 installed package files are authenticated
by [tensorfold-source-hashes.txt](tensorfold-source-hashes.txt), whose SHA-256 is
`8605fa47ae126abb91a79688ab72cb440032679b741ce3bcd7aeb75ae65a7cd3`.
Image inspection, full dependency inventory and build logs remain external raw
receipts. Neither a moving tag nor a different quantization is an oracle substitute.

The approved checkpoint was officially staged on Spark A at
`~/.local/share/llmp/models/TensorFold/Qwen3.8-27B-MLX-4bit@70ae7fac`:
17 files, 16,081,520,742 bytes, complete archive verification. Metadata digests
are in [inputs-receipt.json](inputs-receipt.json) and enforced by `contract.py`.
No host RAM offload or second memory tier is introduced.

## Corpus, scoring and acceptance

The retained corpus is `m3-reference-corpus.txt`, 817,275 bytes, SHA-256
`55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb`.
`prepare_inputs.py` uses the pinned source image's `tokenizers.Tokenizer` on
`tokenizer.json`, `add_special_tokens=False`, and keeps the first 9,216 IDs.
There is no corpus chat template and no inserted BOS. The first ID is 6164;
its position is unscored because it has no preceding token, irrespective of its
value. Vocabulary size is 248,320.

- Frozen `ids.json` SHA-256:
  `477546a2035ee466e127f307305d9158ee848650952d8cd8f8aaa7e7086ab2ef`.
- Frozen `behavior.json` SHA-256:
  `2333d7d57a73e9d12b45c809083ec22367ffe84ed63b57df0f1400ab614341f6`.

The preparation ran as the CPU-only shared job
`m35-u3-qwen27-freeze-inputs-r1` on `spark-c4e2`, return code zero. Its immutable
image used the ordinary CPU runtime, no GPU visibility or network, and only
metadata mounts: zero weight mounts or loads. All 449 source-file hashes passed.
Frozen files and complete launch/inspection receipts remain at
`~/.local/share/llmp/m35-u3-quality/qwen27-frozen-inputs-r1/` on that host.
The helpers reproduce their exact bytes from authenticated source inputs.

| Phase | Input IDs | Scored absolute positions | Targets | Document key |
| --- | ---: | --- | ---: | ---: |
| Short | 1,024 | 1–1,023 | 1,023 | 0 |
| Depth | 9,216 | 8,192–9,215 | 1,024 | 1 |

Both arms receive the same complete integer histories. Depth retains all prefix
state; it does not score a detached suffix. Positions are zero-based. This depth
screen is a bounded long-prefix check, not a claim about the model's maximum or
original training context.

Run two fresh repeats per engine and phase. Each engine's own absolute PPL delta
must be at most 0.1%. After those pass, compare native repeat 1 to reference repeat
1, and repeat 2 to repeat 2; **both native PPL increases must be at most +1.0%**.
The bound is one-sided: a decrease is not evidence of general model superiority.
Do not choose a favorable replicate or widen a bound after a failure. The four
`*-prereg.json` files encode counts and bounds for `tools/quality_pair.py`.
Report aggregate NLL/PPL, relative PPL change, and mean and maximum absolute
per-target NLL differences. Refusals, missing/extra/reordered targets, identity
mismatches and nonfinite scores invalidate collection rather than pass a bound.

## Source adapter and precision

`Qwen27Engine` has no public prompt-likelihood or score-labels interface at this
pin. `source_adapter.py` uses its actual loaded weights, `forward.State`,
`prefill.prefill_chunk(every=True)`, `forward._mm(hidden, w.head)` and
`tensorfold.cuda.logprobs.capture`. It implements no separate eager model.

The engine performs its ordinary capacity admission at explicit context 12,288,
one stream, one GPU, no drafter and no vision. `max_rows=16` reserves the source
engine's verify-row scratch domain at least as large as the adapter's 16-row
head batch. The adapter retains the admitted context and attaches the engine's
`KVRoom` to every fresh state. It refuses an admitted prompt-row budget below
512. It then explicitly sets the source prefill limit to 512 rows; its own
`prefill.chunks` partitions the full teacher prefix into balanced chunks of at
most that size. Native uses fixed prefill chunk 512, one slot and context 12,288.
Their partition boundaries need not coincide; the same complete histories and
actual targets must coincide.

The source explicitly calls `prompt_precision.set_fp8(False)` before loading:
its standard BF16 prompt arithmetic, packed MLX affine weights and BF16 attention
KV remain intact. The native arm retains F16 KV and its production component
plan, with optional lossy modes and drafters off. **Source BF16 KV and native F16
KV differ and are disclosed; neither is silently converted to the other.**
Actual backend, encoding, precision, dispatch and source capacity receipts are
retained. A missing implementation is a diagnostic, not permission to replace
this source reference.

For an input of length N, the adapter processes `ids[:-1]`. Hidden row at absolute
position t predicts the actual ID at t+1. Every chunk boundary is covered once.
The head reads at most 16 selected hidden rows at a time through the exact head
function used by ordinary `decode.prefill`; it does not use `prefill._mm`.
Depth warmup chunks still update all model state but need no output-head scores.
Capture uses `top=0`, no label restriction, and the pinned source's full-vocabulary
FP32 log-sum-exp plus actual-target gather. Generated tokens and top-k restricted
normalizers are never teacher targets. The collector rejects nonfinite logits
and malformed, positive or nonfinite log probabilities.

### Adapter validation before governing scores

After the reviewed preregistration commit, run `reference.py --phase validate`
first. On the first 513 and first 1,024 frozen corpus IDs, use independent fresh
states for (a) the unmodified normal `decode.prefill` first-token path and (b) the
all-row adapter path. Require identical greedy first tokens. On the adapter's
last 16 hidden rows, require the final row's **full vocabulary logits to be
bit-identical** between the 16-row and one-row calls to `forward._mm`. The
controls cross chunk boundaries and use the same balanced 512-row recipe.

Any failure stops the protocol for diagnosis and prospective amendment. No
alternate head, relaxed check, pin or precision is allowed by this document.
Later reference and native launchers require the successful validation receipt,
bound to the protocol hash, preregistration commit, image and all helper hashes.
Validation is adapter verification, not a quality pass.

## Paired behavior controls

This is a new concise Qwen pack, designed before source behavior collection.
`prepare_inputs.py` contains the exact six prompts and expected results. It uses
pinned `ChatTemplate.render(..., tools=None, enable_thinking=False)`, then the
same tokenizer without special-token insertion. The actual frozen rendering ends
in the completed empty `<think>\n\n</think>` block before answer generation.
Both engines receive those exact rendered IDs through their ordinary generation
paths, with greedy sampling, EOS stopping and a 512-token maximum. This allows
room for a concise function without spending the reply budget on requested
reasoning; it does not repair or replace the earlier Llama control.

| Control | Prompt IDs | Required completed answer |
| --- | ---: | --- |
| Capital | 26 | Only `Paris` |
| Arithmetic | 27 | Only `391` |
| Sorting | 37 | JSON array `[1,2,2,9]` |
| Structured data | 35 | JSON object with exactly `planet: "Mars"` and `moons: 2` |
| Local context | 47 | Only `AZ-417` |
| Fibonacci | 59 | Only concise Python `fib(n)`, returning the first n values starting 0,1 |

Trim surrounding whitespace for scalar answers. JSON must parse and match the
stated value with no surrounding prose. For Fibonacci, require one complete
function, no markdown/explanation, and correct results at n=0 (`[]`), n=1 (`[0]`)
and n=7 (`[0,1,1,2,3,5,8]`), with a sensible nonnegative-n implementation.
A reviewer inspects code before any bounded functional execution; the collectors
do not execute generated code. Record semantic review separately from collection.

Run each case as its own supervised job and fresh source/native process. A
length stop is a failed completion control even if an extractable function or
substring passes checks. All six completed controls must pass in both engines
for the bounded behavior screen. If both fail a case, comparative status remains
unresolved; if only native fails it is a native regression; if only source fails,
the control is unresolved rather than proof of general native superiority.
Preserve all negative cells and do not increase bounds or rewrite prompts after
results. An ordinary native chat-renderer smoke remains a separate observation;
this raw-ID pairing isolates model behavior from renderer differences.

## Execution and provenance

All model work uses separately supervised `spark-job --gpu --timeout 600` jobs;
inputs-only preparation and CPU helper tests use `--shared`. No concurrent build,
copy or other model work accompanies a quality job. Source and native launchers
have `finally` cleanup, and supervisors must confirm retirement and no residual
GPU process after every job. Use fresh, unused output directories.
Container names contain a fresh UUID. The shared lifecycle helper allows four
seconds for forced removal and one second for a successful exact-name absence
check, within the supervisor's ten-second termination grace. It suppresses repeated
termination signals during cleanup, retains the outcome even when collection
fails, and refuses to mark a collection complete if retirement is unconfirmed.
Native retirement waits at most five seconds before the existing service helper
forces exit. A nonzero native exit or changed executable invalidates collection.

- `launch_inputs.py`: immutable CPU-only image, metadata/corpus authentication,
  source-template freezing; never invoke a model.
- `launch_reference.py`: inspect the immutable image; run `reference.py` with
  read-only checkpoint/inputs/helpers and no network. For phases after validation,
  pass the resulting `collected/validation.json` using `--validation`; its identity
  and the selected case are checked before any container invocation.
- `native_runner.py`: authenticate the frozen executable and artifact metadata,
  create a fresh configuration/state directory, use existing `speed_gate.Service`,
  collect one phase/repeat or one behavior case, and wait for retirement. Record
  the service helper digest, validation receipt digest and executable digests
  before startup and after retirement.
- `native.py`: transport and actual-target validation only; its expected identities
  are backed by the runner's actual binary/config/artifact launch receipt.

Launchers take `--checkpoint`, `--protocol prereg.json`, `--inputs` (the frozen
folder), `--out`, `--prereg-commit` (the actual 40-character commit), and `--phase`.
Use `--repeat 1` or `2` for likelihood and `--case` for one named behavior case.
The native runner additionally takes `--repo`, `--runtime`, `--store` and the
successful source `--validation` receipt. Pair corresponding `losses.jsonl`
files with `tools/quality_pair.py pair --candidate ... --reference ... --prereg`
and the matching short/depth JSON. Pair each engine's own repeats with the
matching repeat JSON first. Do not interpret a collector's return code as a
semantic verdict.

Raw request IDs, generated text and logs are retained only in these explicit
experiment outputs. Launchers and directly invoked collectors use a private
umask (0700 directories, 0600 files) and refuse an existing output directory.
The source validation prerequisite is published only after adapter completion
and cleanup succeed. Raw records remain external and follow milestone cleanup;
no default runtime prompt logging is added.

The CPU contract suite passed 12/12 cases with no skips in shared job
`m35-u3-qwen27-reference-contract-r1` on Spark A. It covers chunk boundaries,
full-prefix depth geometry, actual-target identity, missing/duplicate/extra rows,
invalid/nonfinite scores, native refusals, frozen-input gating, stale validation
receipts and exact pairing bounds. Static parsing passed. Source model loading,
CUDA adapter validation and quality collection have **not** run under this
protocol; the reviewed preregistration commit is required before those runs. These bounded
screens do not qualify competitive speed, batching, full context or all of U3.

Independent review added mocked cancellation, removal timeout, daemon failure,
existing-output preservation, early validation rejection, failed adapter close,
private-file and native executable-change controls. The complete 23-test suite
passed locally in 0.010 seconds without container, executable, model, GPU or
network invocation. The original frozen IDs, prompts, pins and acceptance bounds
were unchanged by these lifecycle repairs.
