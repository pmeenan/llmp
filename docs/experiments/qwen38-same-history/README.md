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

## Cache-path and repeat controls (2026-10-04)

The fourth anchor's reference acceptance varies with the recorded cache path
and between default-mode repeats. The original 9/12 versus 11/12 screen is
valid for its recorded requests, but does not establish a stable two-draft
native deficit. Native's 2,048-row prefill does not change this anchor's
proposals or acceptance. No arithmetic change, acceptance-parity claim,
performance gate or quality exception follows from these diagnostics.

Mia runs on Spark B with the same pinned fast launcher, image, checkpoint,
observer and inference settings above. Each collection starts a fresh service.
Every listed p3 request supplies the same 31,746 prompt IDs and anchor 1156;
all requests complete nine output tokens. The two-path collections use a
separate cache salt for each path: p3 cold → p3 repeat, then p0 cold → p3 →
p3 repeat. Actual cache counts distinguish cold from reuse; both paths' warm
requests reuse 31,616 tokens. Observed readiness is 159.14 / 159.09 / 193.14 s
for the cold-p3, two-path default and two-path deterministic collections.
Observation costs remain inside the decode clock; no inference timings qualify.

| Collection / request | Cached tokens | Drafts | Accepted / offered |
| --- | ---: | --- | ---: |
| Default cold-p3, p3 cold | 0 | 579, 1330, 8806 | 3 / 3 |
| Default cold-p3, p3 repeat | 31,616 | 579, 1622, 13 | 1 / 3 |
| Default two-path, p3 cold | 0 | 579, 1330, 8806 | 3 / 3 |
| Default two-path, p3 repeat | 31,616 | 579, 1330, 8806 | 3 / 3 |
| Default two-path, p3 after p0 | 31,616 | 579, 1622, 13 | 1 / 3 |
| Default two-path, p3 repeat after p0 | 31,616 | 579, 1330, 8806 | 1 / 3 |
| Deterministic two-path, p3 cold | 0 | 579, 1622, 13 | 1 / 3 |
| Deterministic two-path, p3 repeat | 31,616 | 579, 1330, 8806 | 3 / 3 |
| Deterministic two-path, p3 after p0 | 31,616 | 579, 1622, 13 | 1 / 3 |
| Deterministic two-path, p3 repeat after p0 | 31,616 | 579, 1622, 13 | 1 / 3 |

The deterministic diagnostic turns on both recipe switches,
`VLLM_QSA_DET_TOPK=1` and `VLLM_MOE_DET_FINALIZE=1`; actual container
settings are verified. It is a separate diagnostic configuration, not a
replacement for the pinned performance comparator. Its last two complete
four-row target heads repeat byte-exactly (SHA-256
`f2f0d63200a4d9423f9ac0175cada5f2bc93b7d5253a5fa7ea86cf204d27dabc`),
with the same drafts and acceptance as native. Its cold and first cached
p3 requests still differ. These few observations neither establish general
reference repeatability nor isolate a particular cache or arithmetic operation.

The default two-path p3 repeat after p0 rejects draft 1330 because its target
chooses 1622 after the shared anchor and first draft; the preceding request's
target chose 1330 on that same conditioning. Thus target variation, as well
as draft variation, affects these counts. The first cold-p3 collection also
changes p2's acceptance from the historical 2/3 to 3/3. Comparing only prompt
IDs or adding those counts to the historical total hides these differences.
Future same-history acceptance controls must record request order, cache
counts, actual drafts and complete target heads, and include reference repeats.
Broader acceptance qualification remains open.

Native on Spark A uses the same target/drafter and existing unmodified
benchmark, context 33,792, fixed depth three, head 47,172, runtime prefill and
two independent greedy repeats, with p3 selected alone. Three supervised
invocations use chunk rows 4,096 → 2,048 → 4,096. Each passes all nine-output
greedy controls with zero violations. All three propose 579, 1622, 13 and
accept 1/3. The 4,096-row first-verify heads are byte-identical (SHA-256
`5a2d9ad6882028869e92b7427d99dc0065a49c2fa2c5c0493800170b8615f556`);
2,048 rows changes the full head bytes
(`2ff0f6c4670548e3088a948892dbdccd25ced4e48284d8ed42b67c1a2f4718c3`),
while retaining all four argmaxes. The executable SHA-256 is
`4d3228cc89520db7c7866d78a4c5db6ac5bdbbcd79e4363d192063e9ab52ae93`.
Chunk size, executable identity and retirement are established by the
archived supervisor commands/receipts, not by `spec.json` alone.

### Reproduce and validate

[cache_control.py](cache_control.py) materializes the exact measured diagnostic
controllers/clients from the pinned originals, checking input and generated
source hashes. On B, pass `--controller ~/scratch/acceptance/mia-first-verify-run.py`,
`--client ~/scratch/m3-serving-concurrent-r1/frozen-v2/mia-anchor-client.py`,
`--helper ~/scratch/m3-serving-concurrent-r1/frozen-v2/jitllm-mia-concurrent-run.py`
and `--out ~/scratch/acceptance`. Run each emitted `mia-cold-p3-run.py`,
`mia-cache-paths-run.py` and `mia-cache-det1-run.py` under installed
`spark-job start --gpu --timeout 600`, then wait. Each requires its output
directory to be absent; archive a prior output before re-running. The
existing pinned observer, loader payload, launcher and frozen history above
are prerequisites. Controllers verify actual launch settings and owned
retirement; generated source pins are embedded in the materializer.

For native, use the earlier benchmark command with `--only p3`, head 47,172
and each of the three chunk-row settings, with separate output directories.
The measured collection instead filtered the input to its identical p3 entry;
its I32 digest remains the p3 digest above. Keep the supervisor commands and
successful completion records alongside the output.

Run [cache_compare.py](cache_compare.py) on a Spark with `--cold COLD_P3
--paths DEFAULT_PATHS --det DETERMINISTIC_PATHS --native-before BEFORE
--native-small SMALL --native-after AFTER --out NEW_JSON`. It validates
complete finite F32 payloads/hashes and actual argmaxes, prompt digests,
exact API/worker identity joins, actual sampled prefixes, cache salts/counts,
launch determinism settings, cleanup and native head bookends. It reports
only the collected controls; it does not qualify speed or general parity.

Raw reference files remain on B at `~/scratch/acceptance/{mia-cold-p3,
mia-cache-paths,mia-cache-det1}`. Native controls remain on A at
`~/scratch/acceptance/native-p3-{4096-before,2048,4096-after}`; reference
copies and `cache-analysis-qualified.json` are on A beside them. Jobs
`qaccept-cold-p3`, `qaccept-cache-paths`, `qaccept-cache-det1`,
`qaccept-p3-chunks` and `qaccept-cache-analysis-qualified` all completed with
exit zero and were waited on. Every reference container was removed and
its launcher reaped with no cleanup errors. No production code changed;
no new unit suite or workstation tier was needed for these diagnostic drivers.

| Evidence | SHA-256 |
| --- | --- |
| Cold-p3 reference receipt | `6e7e678751b844dbc1820508221002128b446a0d7e9c6b3b2250727465be6fd9` |
| Default two-path reference receipt | `cacec5edd2d4bdd709afb3125db4194ea492f2513fa48ab3895b6da8c6b36528` |
| Deterministic two-path reference receipt | `3c6ab11912c56fdd5ec33001a42bdf1bef98d8779d9061484c54ce3f079545df` |
| Native 4,096 before spec.json | `a1d6edb319fd72ba5e9e5fd55afd27e57e707f8673b916bb4c15edcb4ccfac40` |
| Native 2,048 spec.json | `36316ab42202dd8dcf6c6bc40944eafb92da5fbecf2db5c8ddaca821f0df4526` |
| Native 4,096 after spec.json | `45dd16715baab445fbf1a56d1df7161eee1ffbce4a102027177ae07f4bf44c3c` |
