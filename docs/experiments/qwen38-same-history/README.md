<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen first-verify acceptance on the same history

2026-10-03: four adjacent anchors from one frozen 32K history expose a
draft-proposal deficit. Native accepts 9 of 12 offered drafts; the pinned
Mia reference accepts 11 of 12. Native's curated 47,172-row and prefix
65,536-row heads give identical proposals and complete first-verify heads
on these anchors. This is a representative diagnostic, not acceptance
parity across contexts, a performance comparison, or a new quality bound.

## Result

Each request supplies the identical literal prompt IDs to both engines.
The anchor is the target's next token; only its first verify, containing
that anchor and three actually offered drafts, qualifies the comparison.
Later generated histories can diverge and do not qualify a paired count.
Native independently clears, prefills and repeats each request twice; its
complete first-verify F32 rows repeat bit for bit. All nine-output native
greedy controls pass the unchanged bound, with no reported problems.

| Anchor offset | Prompt tokens | Anchor ID | Native drafts (both heads) | Mia drafts | Native accepted / offered | Mia accepted / offered | Mia cached tokens |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 0, cold | 31,743 | 1596 | 1144, 4087, 1156 | 1144, 4087, 1156 | 3 / 3 | 3 / 3 | 0 |
| 0, reference repeat | 31,743 | 1596 | 1144, 4087, 1156 | 1144, 4087, 1156 | 3 / 3 | 3 / 3 | 31,616 |
| 1 | 31,744 | 1144 | 4087, 1156, 579 | 4087, 1156, 579 | 3 / 3 | 3 / 3 | 31,616 |
| 2 | 31,745 | 4087 | 1156, 579, 1622 | 1156, 579, 1622 | 2 / 3 | 2 / 3 | 31,616 |
| 3 | 31,746 | 1156 | 579, 1622, 13 | 579, 1330, 8806 | 1 / 3 | 3 / 3 | 31,616 |

The four unique anchors total 9/12 versus 11/12; the cold/warm reference
repeat is reported separately. At offset 3, both targets choose 1330
after the shared anchor and first draft, but native proposes 1622. Thus
the measured loss at this anchor occurs in the proposal, rather than a
target disagreement on the shared input. This identifies an operand for
further investigation; it does not establish which arithmetic operation
caused the different proposal.

[compare.py](compare.py) independently checks all four raw rows against
each engine's own recorded argmax, then compares cross-engine rows only
while their conditioning tokens agree. All 18 such rows per native head
(including the reference repeat) have matching argmaxes. Their maximum
absolute logit differences range from 1.984 to 3.964; mean absolute
differences range from 0.299 to 0.607. Those differences are descriptive,
not a calibrated tolerance or a quality verdict. Rows after the first
different draft have different inputs and are excluded from that comparison.

Native ran on `spark` and Mia on `spark-b` (GB10, driver 580.178.04;
native SDK CUDA 13.4.92). Both use the pinned Qwen checkpoint, but native
uses F16 KV, F32 recurrence/draft input and 4,096-row prefill chunks;
Mia uses FP8 KV, BF16 recurrence/draft input and 2,048-row chunks. Mia
keeps one service and its prefix cache across the five requests. These
cache, precision and node differences remain part of this diagnostic.
Observation copies and hashes logits and writes files inside the decode
clock; no decode rate, step time or end-to-end speed from these runs is
qualified. The observed warm-cache launcher readiness was 168.141 s.

## Reproduce the collection

Use [the pinned fast Mia launcher](../fast-swap/baselines.md#default-mia-launcher-for-new-runs-2026-10-03)
from commit `64acbb6`, with its complete payload and loader-patch pin;
do not fall back to the historical slow launcher. Runtime code is
`58ac425`; `64acbb6` changes reference tooling and documentation only.
The native target artifact is
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`,
drafter artifact
`8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`.
Checkpoint revision is `925d7be6c14c6c9442ef83e8f05b5a3c39304f69`;
tokenizer JSON SHA-256 is
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`,
and curated vocabulary SHA-256 is
`ee819d2560b52ba1351acdd1c4a0244b77bb60694a7660c6363097a482b0afb5`.

Captured input remains external. Retrieve
`spark-b:~/scratch/m3-serving-concurrent-r1/frozen-v2/anchor-history.json`
(SHA-256 `c0f9706146aca18113f1cd91cdc1e1107c18df34c72fdd6a56e3192dc6a7aa3c`),
or supply that exact file from its owner. Build four literal entries
`p0` through `p3`: the original prompt plus the first zero through three
frozen continuation IDs; each has `stable_boundary = 31738`. No chat
template or tokenizer round trip enters their input. The literal schema is:

```json
{"decode":[{"id":"p0","prompt_token_ids":[1,2,3],"stable_boundary":3}],"chat":[]}
```

The example IDs illustrate the schema; they are not the measured input.
The prompt digests below hash little-endian signed I32 IDs:

| Entry | SHA-256 |
| --- | --- |
| p0 | `306136a7dc2a5f66c94cbf4ac874e9011c0af79b79072731407a144bfba1fbd5` |
| p1 | `53bda5259e6751681f80455d531d6dd89d4e255f8779b35446d59e2c245f963b` |
| p2 | `a3261b4f298733484a629fe27f0e0109b4945e0d8d750d92ef8d85e855341e28` |
| p3 | `184d7c02214ef662c0e927ca63f4e0091eb42a918a1f79423c1be34b7303ae1e` |

Run `jitllm_qwen38_spec` with the target/drafter/tokenizer paths,
`--prompts INPUT --prompt-token-ids on --runtime-prefill on --context 33792
--prefill-chunk 4096 --draft 3 --draft-vocab 47172 --check greedy
--tokens 9 --repeats 2 --out OUTPUT`; repeat with `--draft-vocab 65536`.
Literal mode requires greedy, two repeats, fixed depth three, no confidence
window and at least five outputs. It rejects malformed IDs, duplicate
entry names, empty selections and insufficient context before model admission.
Use installed `spark-job start --gpu` and `wait` for all model work.

For Mia, preserve the matched inference settings: context 33,792, four
sequences, FP8 KV, BF16 recurrence, fixed MTP depth three, curated 47,172
vocabulary, chunk 2,048, GPU utilization 0.786, reserve 26 GiB, block
drop disabled, deterministic mode off, compilation mode zero and
`FULL_DECODE_ONLY` graphs. The qualified controller and its launch
environment are retained at
`spark-b:~/scratch/acceptance/mia-first-verify-run.py` and
`~/scratch/acceptance/launch-environment.json`; controller SHA-256 is
`151e257fd1d8ac43dcfe8fbf57a98103bee6ba0836f40208506cf713caf85b87`.
It uses the frozen lifetime/client helpers beside the original history,
records actual paid-work and cache counts, and stops/removes only its owned
container and reaps its owned launcher on success or failure.

The pinned V2 rejection sampler is
`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`, SHA-256
`308d4171f892f27231655289ceb03cd4566533c7d26b8f98bd62ef0a34f986f0`.
Copy its unchanged bytes to an external overlay and append:

```python
# jitLLM diagnostic observer; the original sampler runs unchanged.
from jitllm_mia_observer import enable as _jitllm_enable
_jitllm_enable(RejectionSampler)
```

Mount that overlay and [mia_observer.py](mia_observer.py), named
`jitllm_mia_observer.py`, read-only into their Python package paths.
Set `JITLLM_MIA_TRACE_DIR` to an owned mounted directory with mode 0700.
The observer calls the original sampler first, reads inputs through
`batch.logits_indices`, and asserts its independently derived acceptance
against the actual returned sampled prefix, kept and rejected counts.
Python assertions must be enabled. Its source SHA-256 is
`fc34ef7a55ed95204632006ebac1ec31a5b6f573f74d37b97b2cc9710ce8cab9`;
the complete sampler overlay SHA-256 is
`f87e3bcec9453f41b289f02ed3cf82bc899ff105a0fd6a24d114d9bdc928b539`.
Join each worker request to its API response identity using the pinned
input processor's exact eight-hex-digit random suffix, not a loose prefix
match. Its source SHA-256 is
`f9a7946a16acc2374ff2bdfc22f212cb43461d9ef4d99c5e19a536339f11212f`.

## Evidence retained outside Git

Native captures are on `spark:~/scratch/acceptance/native-curated-final`
and `native-prefix-four`; qualified Mia captures are on
`spark-b:~/scratch/acceptance/mia-first-verify-fast2` (also copied to Spark
A for analysis). Keep `spec.json`, each complete four-row F32 file,
Mia's receipt and traces, and the supervisor records together. The native
JSON alone does not prove retirement. Jobs `qaccept-native-final`,
`qaccept-native-prefix`, `qaccept-mia-fast2` and `qaccept-analysis` all
finished successfully and were waited on. Mia's container is absent and
launcher reaped, with no cleanup errors.

| Evidence | SHA-256 |
| --- | --- |
| Curated native spec.json | `370d4c5f634316df0d5c2e3833d496cdc80221175a19bd51c78d42be10a23d7d` |
| Prefix native spec.json | `3f4d5b8095c39bab75e229d07c39a389a29182b200ff093815995288b0ab4dae` |
| Qualified Mia receipt.json | `3cb6198db3d5a029fc964aa64d4d558d42cfca24e1778aaf9542583eeb5f1903` |
| Archived native executable, `spark:~/scratch/acceptance/bench-executable` | `0d491ac953bbd3bba90a2ae640dc6466e2cb58ed2f34d02311092ec675aa5cde` |

The measured executable precedes the final empty-selection parser guard;
valid model inputs and arithmetic are unchanged. Malformed-input checks
and a 33-output chat-list literal control validate the final guard and
both required repeats separately. An earlier fast-launch run completed
its requests but failed the collector's identity join; it is excluded
from this paired result. Neither its captures nor historical generated
windows establish acceptance parity or justify relaxing a quality bound.

On a Spark, run `python3 compare.py --native CURATED --native PREFIX
--mia MIA --out NEW_REPORT`. The analyzer requires the exact five-request
schedule, complete paid-work and cleanup receipt, matching prompt hashes
and anchors, finite complete F32 rows and their hashes, and consistent
actual acceptance. Raw inputs, rows, logs and receipts stay outside Git.
