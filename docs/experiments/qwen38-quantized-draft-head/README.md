<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 selected Q4_1 draft head

The optional prepared Q4_1 head retains all 47,172 selected token IDs and
reuses existing QuantizeQ8/VecQ kernels, including scalar-equivalent OneToken
arithmetic for joined requests. Its logical payload is **75,475,200 bytes**,
compared with 241,520,640 BF16 bytes (68.75% less). BF16 remains the ordinary
configured artifact: the short same-native C1 factor takes 4.064% less total
generation time, while C2 takes 1.482% more. Quantization changes proposals
and acceptance, so these are whole-format results.

A separate own-Q4 C2 sharing factor reduces its measured drafting interval
5.049%, but total generation changes only −0.143%, below the 0.229% movement
between unshared bookends at n=2. The existing TensorFold target-quality
question remains separate from exact same-native quantization/batching proofs.

## Prepared format and execution

The offline [preparer](prepare.py) reuses the existing artifact writer and
validator, and pinned GGML v0.6.0 quantizer
`d81235049384534c167caea52b85a694f6103d14`. The immutable BF16 parent is
`8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`;
the prepared Q4 child is
`95e8d51aa17f7f9e9199d034dccc51978b820194a2dc3bd1bf8e42f8c59765ff`.
All 2,081 other resource/expert slices remain exact, including the selected
ID map. Aggregate results retain their identities and preparation evidence.

Every copied range is authenticated in the same read pass as its containing
manifest shard. The helper reports the digest of the actual BF16 bytes it
quantizes, and publication requires the authenticated expected digest.
It streams 128 rows using 2,170,880 combined bytes for BF16/F32/Q4 buffers.
Before calling the reference quantizer, it checks each 32-value block's
finite inputs, span, scale, reciprocal and representable F16 coefficients.
A valid 65,536 spread passes; tiny reciprocal overflow is refused. Exclusive
creation prevents clobbering an existing output. The existing writer fsyncs
payload/index/manifest files; directory fsync and full verification precede
atomic artifact publication. Source artifacts remain immutable.

Binding/settings accept explicit BF16 or Q4_1 selected heads. The Q4 path
uses 2,880 Q8 activation bytes per column and existing VecQ. Joined products
preserve ordinary scalar variant arithmetic through OneToken. Prefix sharing
authenticates the immutable parent, direct zero-offset VIEW, producer/source
and data identity, encoded offset, 1,600-byte quantized row stride, containment
and span. Mutable overlap anywhere in the parent, including unused suffix
rows, prevents a join. Other unqualified views keep separate products.
No new CUDA arithmetic is introduced.

The focused operand proof executes the complete 47,172-row head and prefixes
1/31/32/33/16,385/47,171 with two and three actual columns. Complete finite
outputs match ordinary scalar bytes through eager execution and poisoned
changed-input capture/replay; guards, remapped argmax and protected input
peers survive. Actual two-request full/prefix paths preserve target heads,
proposals, confidence, initialized state, capture/replay, spill/restore and
protected peers.

The new proof exposed a pre-existing joined confidence check: Argmax stores
probability bits in an I32 view, while the runner expected F32. The corrected
boundary authenticates I32/4 bytes and copies those bits into the float
destination without conversion. A real BF16 C2 confidence control also passes.

## Same-native format and sharing factors

Standing prompts contain 1,536 tokens for C1 and 1,536/1,280 for C2. Both use
context 2,048, chunk 512, graphs/GPU masks and confidence window zero. Configured
draft capacity is three; C1 executes three passes, C2 two. Each paid request
returns 32 outputs and includes final owed settlement. Full same-conditioned
Teacher32 heads, state observations, finite checks, hashes and persistence run
after the paid endpoint, with observation capacity charged beforehand.

| Factor | BF16/off 1 | Q4/on 1 | Q4/on 2 | BF16/off 2 |
| --- | ---: | ---: | ---: | ---: |
| C1 generation seconds | 1.831711619 | 1.757921481 | 1.759660095 | 1.834884397 |
| C1 decode seconds | 0.657787168 | 0.575233494 | 0.577526810 | 0.651473504 |
| C2 generation seconds | 3.461826440 | 3.500321435 | 3.526186160 | 3.462061287 |
| C2 decode seconds | 1.203350743 | 1.241414095 | 1.256337959 | 1.209092969 |
| Own-Q4 C2 generation seconds | 3.498808274 | 3.498378613 | 3.497209758 | 3.506821667 |
| Own-Q4 C2 draft seconds | 0.117488220 | 0.111944324 | 0.111488775 | 0.117824840 |

C1 generation/decode means change −4.064%/−11.953%; prefill +0.318%. BF16
generation bookends move +0.173%. Its considered proposals/accepted/verifies/
physical passes change 32/20/11/33 → 29/21/10/30. Considered proposals clip to
requested verification rows; physical passes count all executed drafting.

C2 generation/decode means change +1.482%/+3.536%; prefill +0.384%. BF16
generation bookends move +0.0068%. Owner zero stays 21/20/11/22; owner one
changes 32/14/17/34 → 34/13/18/36. The extra verify is observed work; this does
not prove it alone causes the regression. Complete committed IDs and target
Teacher32 heads are exact across formats. Same-format repeats also preserve
both states and work; draft state/proposals may differ across quant formats.
These n=2 screens establish a configuration tradeoff, not a global default.

The own-Q4 factor fixes verifier pairing and all other policies, switching
only DraftWave sharing. Every arm executes 11 cohort draft calls, with exact
complete heads/history, target+draft states, acceptance, verifier geometry,
target work and static setup budgets. Shared arms select 11 joins/22 actual
Q4 head pairs; separate arms select zero. Generation/decode means change
−0.143%/−0.276%; the drafting interval changes −5.049%. This measures the whole
draft-sharing policy. Every arm retains ten 3/3 shared-head verify calls and
one legitimate 2/3 head fallback. It does not erase the BF16-to-Q4 C2 loss.

## Fresh TensorFold comparison

Fresh task-entry checks retain latest applicable Python CUDA
`ed78d6fc204d89d90b045bf033d6551e7714f3a1` (0.6.6); native HEAD is
`f8fe17d24629aedabf90bbf78279dd776e6d62e7`, whose Flash-Next recipe is Metal.
The fast public FP8-prompt MTP path uses depth three/confidence zero, graphs,
BF16 KV/convolution, F32 recurrence and its unchanged 79,591-row MLX Q4 head.
Native uses its NVFP4/MXFP8 prompt path, F16 KV, F32 convolution/recurrence and
the prepared 47,172-row GGML Q4_1 head with Q8 inputs. Coefficients and head
lists differ. Public CopyIndex requires eight tokens, so chain substitution
cannot run at depth three; index bookkeeping still runs.

Full same-conditioned Teacher32 quality uses the unchanged margin 1.0.
These are descriptive generated-history targets, not a held-out PPL sample.
Both complete finite Teacher32 payloads repeat exactly per engine. Agreement
is **31/32**; row29 remains outside the unchanged bound: native/TF top-two
margins are 0.828717/1.5625, and directional argmax losses are
0.828717/2.6875. Mean generated-history NLL delta is −0.099248
(exp ratio 0.905518), descriptive only. Native's teacher heads also remain
byte-exact to its earlier BF16-head control. The target-quality lead remains
open; the draft quant does not repair it.

| Speculative C1 arm | Generation seconds | Decode seconds |
| --- | ---: | ---: |
| TensorFold R1 | 1.789887567 | 0.6077 |
| Native Q4 N1 | 1.755203094 | 0.574180146 |
| Native Q4 N2 | 1.755633087 | 0.575793549 |
| TensorFold R2 | 1.792575369 | 0.6091 |

Native mean generation is 1.755418091 s versus 1.791231468 s, **1.999% less
time**; decode means are 0.574986848 versus 0.6084 s, **5.492% less time**.
TF's reported decode has four decimal places. Reference generation/decode
bookends move +0.150%/+0.230%; n=2 is a bounded screen, without a full HTTP,
long-context or batch-reference claim. Both engines pay their documented
warm/clear policy and 32-output generation endpoint; the public TF harness's
plain mode is auxiliary and has no fresh native plain counterpart here.
Native considered drafts/accepted/verifies are 29/21/10; TF 30/21/11.
Natural continuations agree 29/32 and first differ at row29; quality uses
identical supplied history throughout instead of comparing divergent rows.

## Qualification and replay

Ten unique focused tests, exact model controls and eight direct quantizer
refusal controls are covered across source-qualified stages. The full suite
is deferred by the owner. GPU operand tests require `JITLLM_Q4_HEAD_FIXTURE`;
an ordinary suite without it skips them. The focused controller sets it and
requires exact named positive XML, without skips/errors/disabled cases.

Failed records remain separate: the first model proof exposed the confidence
descriptor mismatch; the first C2 factor used an invalid universal head-sharing
assertion, later narrowed to actual eligible 3/3 calls with explicit fallback
accounting. That failed call's exact geometry was not observed. Compiler name/
initializer corrections are mechanical provenance. Completed earlier positive
controls are reused only with explicit source/binary/result bridges; no failed
arm is pooled. The import challenge's input-authentication finding was fixed
before artifact publication.

The final source composition carries all 14 qualified implementation/test paths
byte-exact onto `fa7cf93`; its 22 intervening Gemma source/documentation paths
are disjoint. No native rebuild or reference rerun is implied by that source
bridge and the final documentation/replay-only additions.

Required replay inputs live in the standing store and survive milestone raw
cleanup. The eight authenticated actual-model operand files under
`~/.local/share/jitllm/references/qwen-q4-head/operand-fixture-v1` are **KEEP**;
aggregate results list all SHA-256 values. Raw logs and head samples may be
deleted without removing these inputs. The target is prepared artifact
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`,
with the existing approved `925d7be6` tokenizer. Literal prompts/history and
TensorFold authentication/package/JIT inputs are documented in the
[shared reference recipe](../qwen-device-masks/README.md#durable-reference-replay).

Recreate the BF16 parent through its existing selected-head import recipe.
Build `jitllm_qwen38_quantize_head` with the pinned SDK, then run `prepare.py`
with explicit source ID, helper SHA, a new output store and receipt. Select the
result as the native runner's drafter; keep the BF16 source for its control.
Input identities, source/receipt/library bindings, XML identities and
retained-stage relationships are in [aggregate results](results.json). TF preparation reuses the completed
105.936 GB checkpoint identity receipt plus exact file stats; no repeated raw
checkpoint import is needed.

Compatible DSpark and Gemma-assistant compression remains open. DSpark borrows
the target head; Gemma assistants tie Q8_0/F32 head to embedding. Independent
binding/closure or tied-weight policy, funding and quality need their own
qualification. Wider contexts/cohorts and public HTTP performance are not
established by these short direct-runner factors. BF16 stays the ordinary
configuration, and the original TensorFold target-quality lead stays open.


### Repeating the bounded checks

Use the pinned SDK and prepared sources (`mise run build -- spark-native`),
then build `jitllm_qwen38_spec`, `jitllm_qwen38_quantize_head`,
`qwen38_wave_plan_test` and `qwen38_quant_head_test`. Heavy work runs through
the installed Spark supervisor with `--gpu --timeout 600 --stop-on-fail`.
Record the new source inventory, SDK receipt, actual binary/loaded library
hashes, artifact identities and successful retirement; historical raw job
paths are not replay prerequisites.

With `store` naming the standing `~/.local/share/jitllm/m3-artifacts` store and
`quantizer_sha` the freshly built helper's SHA-256, prepare a separate artifact:

```sh
python3 docs/experiments/qwen38-quantized-draft-head/prepare.py \
  --source "$store/8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40" \
  --source-id 8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40 \
  --quantizer build/spark-native/benchmarks/jitllm_qwen38_quantize_head \
  --quantizer-sha256 "$quantizer_sha" --out-store "$new_store" --receipt "$new_receipt"
```

The approved parent and child are in that standing store. Authenticate their
manifest/index and full import identity before use; the importer refuses to
replace an existing child/receipt. Obtain approved literal inputs from
`~/.local/share/jitllm/references/qwen-device-masks/inputs/`; aggregate results
record the hashes for `prompt.json` (1536 IDs), `wave-prompts.json`
(1536/1280 IDs) and `native-histories.json`. If absent, supply the authenticated
files externally or reproduce the original literal selection using the
[shared recipe](../qwen-device-masks/README.md#durable-reference-replay), then
require those hashes. They remain external to Git.

The following invocation demonstrates C1 with observation outside timing.
Set `target`, `tokenizer`, `drafter`, `inputs` and a fresh `out` explicitly:

```sh
build/spark-native/benchmarks/jitllm_qwen38_spec \
  --qwen38-artifact "$target" --drafter "$drafter" --tokenizer "$tokenizer" \
  --prompts "$inputs/prompt.json" --only mask-short-1536 --prompt-token-ids on \
  --context 2048 --prefill-chunk 512 --draft 3 --draft-vocab 47172 \
  --adaptive-depth off --window 0 --graphs on --device-masks on \
  --check masks-lean --tokens 32 --out "$out"
```

For C2, replace the prompt file with `wave-prompts.json`, use
`--check masks-wave-lean --slots 2 --wave-lanes on --paired-draft on`.
Switch only the BF16/Q4 drafter for the format factor. For the own-Q4 sharing
factor keep Q4 fixed and use `--paired-draft off/on/on/off`; target verifier
pairing remains enabled. Compare complete teacher heads/committed IDs and
same-quant initialized states/work, not draft-state bytes across formats.
For actual 47171-prefix lifetime proof use `--check masks-wave --tokens 16
--draft-head-proof on --draft-vocab 47171`, keeping the same other settings,
paired and unpaired lifetimes. This path also checks spill/restore and peers.

Run the two GPU operand tests with `JITLLM_Q4_HEAD_FIXTURE` set to the standing
`operand-fixture-v1` directory. Its eight file hashes are in results.json;
refuse missing/mismatched files before launch. To recreate that fixture,
run the same artifact with `--check draft-head --tokens 2` and capture its
first/changed actual inputs and complete logits; copy their IDs/argmax rows
and extract the authenticated prepared Q4 resource. Keep captured inputs in
the standing test-input store, not a disposable raw-result directory.

For a new public TensorFold comparison follow the
[durable reference recipe](../qwen-device-masks/README.md#durable-reference-replay)
and re-pin both upstream branches at task entry. The reusable
[reference controller](../qwen-device-masks/reference_cycle.py) accepts
`--native-drafter "$drafter" --native-spec-only` for this Q4 speculative
comparison, fresh source/build receipts and an output directory. It keeps
R/N/N/R, actual runtime source/extensions/libraries, immutable checkpoint
identity, logical context/cache capacity, reset/work and independent container
retirement checks. Use [quality.py](../qwen-device-masks/quality.py) on both
TF teacher payloads with the same native history and fixed margin 1.0.
Do not substitute a natural divergent history or infer held-out PPL.
