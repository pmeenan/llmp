<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen27 MLX normalization diagnosis

The first bounded Qwen27 native serving smoke produced garbled text. A subsequent
small-tensor audit proved that the importer added one to RMS scales that the MLX
checkpoint already stores with that addition. This establishes an import defect;
it does not establish that this is the only cause of the failed smoke. No Qwen
reference quality comparison has been collected or preregistered.

## Original smoke and source identity

On Spark A (`spark-c4e2`), supervised job
`m35-u3-qwen27-native-smoke-r1` used the preserved native runtime
`32e247723ebe2c00bcdae1cb1aff12884ce7fb7fb6b421e05e47eea9e1c124af` from
[source E](RESULTS.md). The artifact was
`ca25eee473b02f4a13fe1d8680dea5f2aa244b556d26ee50eaa0b62b8719d3b5`, imported
from `TensorFold/Qwen3.8-27B-MLX-4bit` revision
`70ae7fac63274ff2eac54152031433374cb80f2f`. That artifact and the negative
receipt remain preserved. Source metadata and tokenizer hashes were authenticated
by [prepare_serving.py](prepare_serving.py); no checkpoint download or replacement
was involved.

Configuration was one slot, context 12,288, F16 KV, prefill chunk 512, temperature
zero, top-p one and top-k zero. Readiness took 1.607 seconds. The component engine
reported architecture `qwen3_5`; its three measured plan shapes had no primitive
fallbacks or missing numerical qualifications. Final tensor coverage reported
8,176 tensors and zero violations. Those structural checks did not establish
semantic correctness.

The literal source-vocabulary IDs `[760, 6511, 314, 9338, 369]` represent
`The capital of France is`. The completion returned HTTP 200 with 64 finite token
log probabilities, garbled text and a length stop. An ordinary chat asking the
capital of France used 70 prompt tokens and also exhausted its 64-token bound:
its reasoning field was garbled and its content field empty. The collector's job
return code was zero, indicating collection and cleanup, **not a semantic pass**.
The runtime retired through SIGTERM with return code zero; the host then had no
running jobs or GPU compute processes. This smoke is not the governed Llama
256-token behavior pack and does not change that pack's unresolved result.

## Tensor proof and convention

The source implementation is TensorFold commit
`cb2ebf0540f42604e2759b2ddef497861e928248`. Its
[`qwen3_5/cuda/weights.py`](https://github.com/pmeenan/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/src/tensorfold/families/qwen3_5/cuda/weights.py)
loads MLX norm vectors unchanged. Its
[`glue.py`](https://github.com/pmeenan/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/src/tensorfold/families/qwen3_5/cuda/glue.py)
multiplies normalized activations by those scales directly. In contrast,
`qwen3_5/cuda/nvfp4_load.py` explicitly adds one to the zero-centered Hugging Face
norm vectors; it leaves GDN's gated norm unchanged.

Small direct reads of source and artifact tensors compared every element of all
209 relevant vectors. This audit did not rehash whole source or artifact shards.

| Role | Vectors | Elements | Artifact minus source, every element |
| --- | ---: | ---: | ---: |
| Block input norm | 64 | 327,680 | +1 exactly |
| Block post-attention norm | 64 | 327,680 | +1 exactly |
| Attention query norm | 16 | 4,096 | +1 exactly |
| Attention key norm | 16 | 4,096 | +1 exactly |
| Final norm | 1 | 5,120 | +1 exactly |
| GDN gated norm | 48 | 6,144 | 0 exactly |

For example, layer-zero input scale `1.046875` became `2.046875`, and final
scale `1.9609375` became `2.9609375`. The full layer-zero input source vector
SHA-256 is `647b28cdc1313387bc49ca87560581951478d31f3de735676c1f4ab2963bf8fa`;
its incorrect artifact vector SHA-256 is
`4a8765dafa954503fa316a2927c5648fc64b3a98fc937889b381d9fc859778b7`.

The repair changes the declarative `qwen35` MLX `norm.plus_one` fold from
importer-applied to converter-applied. The existing mapping interpreter then
suppresses zero-centered runtime parameters and copies the converted norm bytes.
Hugging Face's importer fold, GDN's ordinary gated norm, `ssm.a_neg_exp`, and the
GDN head reorder retain their contracts. No weight-value heuristic, checkpoint
name condition, runtime workaround or precision change is introduced. A corrected
artifact must be reimported with a new identity; editing the old artifact is not
a repair.

The paired regression fixture exercises the real MLX and Hugging Face readers,
normalization and output routes. Explicit BF16 input vectors describe the same
scales in converted and zero-centered conventions; all eight norm vectors, covering every role, produce
the expected runtime values. It also checks converter versus importer provenance
and preservation of the other GDN transforms.

## Corrected import and bounded recheck

The checked source is base `7e193ff8a5a69df6f6c47df684a2fe44e3ce55f7` plus the
tracked source/test/changelog patch SHA-256
`abc97e2e82d1b088616f9f1d060b6110a9ed1783287fcda450352e96c7f3aa69`.
The report itself is additional documentation. The Spark checkout's dirty Git
version label is not the source identity. The resulting runtime SHA-256 is
`3a994d33a9643ff3ee8db046722e17f6d42ce9efbfeeaf6d5fe3808005789a38` and importer
SHA-256 is `90138fa21171f73af64f0c0552cc35b2057931b3af47334ba4a288012da7654c`.
The preserved source-E runtime remains unchanged.

On Spark A, `m35-u3-qwen27-norm-build-r2` built the CLI, runtime and four touched
import test executables. `m35-u3-qwen27-norm-tests-r2` passed all 71 functional
cases (55 normalization, three parameter folds, eight MLX and five ModelOpt),
with no skips. The initial build exposed fixture API errors, corrected before
these checks; an initial direct test invocation lacked the declared test-data
environment, also corrected before the complete passing run. Independent static
review and adversarial challenge were clean. Final integrated checks remain the
main agent's commit gate.

Exclusive job `m35-u3-qwen27-norm-reimport-r2` imported only the same approved
checkpoint in 311.430611 seconds. The writer checked the stored spec against its
plan, validated that spec, and performed deep artifact verification before
publication. The new artifact is
`65834c669c64ce29da12260ca5317c9c130707b19eca9c6a956ad5a9c45acfd1`.
Its manifest, index and spec SHA-256 digests are respectively
`0cb9756e1e679a4ddb386e3d0ed9247d2393a193b8c9a295860b4c945e0a677e`,
`0170880976d804a3d19f543bbca78d7ddb4edc3e47100926f76a4e231bcba219`, and
`d73d3cca23bd8f5ff415aeeb93c4554860da73d32c5e33417ac311ca3b904725`.
A direct audit of all 209 vectors and every element in the table above found
**exact equality with the source**, including all 48 unchanged GDN vectors.
The earlier reimport attempt mistakenly used a shared hold; it was explicitly
terminated after 47 seconds and contributes no publication or qualification
evidence. Its temporary writer state was left untouched. The original artifact
was preserved.

Exclusive job `m35-u3-qwen27-native-smoke-r2` repeated the original two requests
with the same settings, literal IDs, messages and 64-token limits. Readiness took
1.607579 seconds. Both requests returned HTTP 200, but **both still failed the
semantic smoke**:

| Request | Prompt / output tokens | Actual output | Finish |
| --- | ---: | --- | --- |
| Literal capital prompt | 5 / 64 | ` 0888888888888888888888888888888888888888888888888888888888881\n1` | length |
| Ordinary capital chat | 70 / 64 | Empty content; reasoning begins `1313333333`, followed by another `3` and repeated whitespace | length |

All 64 literal token log probabilities were finite. The component engine reported
`qwen3_5`, four retained plans, zero refused runs, and final coverage of 8,188
tensors with zero violations. These are structural observations, not semantic
success. The collector returned zero after collection; the runtime retired with
SIGTERM and return code zero, and the supervisor confirmed no running/waiting
jobs or GPU compute processes. No wider retry or quality comparison followed.
The normalization defect is repaired; a further cause of incoherent output
remains unresolved. This bounded failure does not qualify Qwen27 quality,
context length, batching or speed.

Raw request/response, runtime, retirement, import and vector-audit receipts stay
outside Git under the Spark A experiment directory, in
`qwen27-native-smoke-r2` and `qwen27-norm-reimport-r2`. The identities, prompts,
settings, counts and negative result above are sufficient to interpret this
report without those raw bundles.

## Adjacent mapping audit

The separate `qwen4exp` native mapping also currently declares MLX norms as
importer-folded. Its pinned TensorFold implementation distinguishes conventions:
`qwen4_exp/cuda/reader.py::norms_around_one` selects how `weights.py::cscale`
reads them. On the approved Flash Next MLX revision
`dadefa8066e3be900a0d148d0f5a2f4eb1cf6534`, all 48 block attention hyper-connection
norm means exceed 0.5 and their median is 1.0409034729003905, selecting unchanged
scales in that source implementation. The ModelOpt revision
`925d7be6c14c6c9442ef83e8f05b5a3c39304f69` selects the centered convention
(median 0.04417793972534127 across its 49 corresponding source vectors).
This is an additional MLX convention discrepancy to resolve explicitly; the
Qwen27 repair does not change Qwen38 or ModelOpt mappings. No Gemma MLX convention
is inferred from these Qwen findings.
