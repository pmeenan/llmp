<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 preliminary short likelihood result

**Registered preliminary aggregate PASS; general U3 qualification remains open.**
The single 1,023-teacher pair met the prospective native PPL increase bound of
+1.0%. Native PPL was 0.9598% lower. However, mean absolute per-target NLL
difference was **1.3657 nats**, with a **19.3307-nat maximum** at zero-based
position **884**, target **232021**. No per-target bound was enrolled. Opposing
score differences offset in the aggregate; this is not tokenwise/logit parity
or evidence of native superiority.

| Measure | Native | Source reference |
| --- | ---: | ---: |
| Input IDs / scored teacher targets | 1,024 / 1,023 | 1,024 / 1,023 |
| Mean NLL (nats) | 5.9302727887 | 5.9399170515 |
| Perplexity | 376.2571385313 | 379.9034158216 |
| Generated tokens used as teachers | 0 | 0 |

Native minus source mean NLL was -0.0096442628 nats. Native NLL was lower on
508 targets, higher on 515, and identical on none. The comparison was computed
once with the authenticated existing `quality_pair.py`; no retries, replacement
samples or changed bounds were used. The +1.0% allowance was prospectively
borrowed from the U3 Llama/Qwen screens, not measured as a Gemma tolerance.

## Registered input and execution

The [protocol](PROTOCOL.md) and [numeric bounds](pair-prereg.json) were committed
as `eadf8da897a6968ddeed4696408d7759d56a5ab1` before either arm. Executed
protocol SHA-256:
`bd294f02be8b52a72a27046d82d247f11e8644a5c68e212594f13e6b7b3644e4`.
The [CPU freeze](PREPARATION.md) supplies BOS 2 exactly once plus the first
1,023 body IDs of the existing public U3 corpus. Both arms received the same
literal [1,024 IDs](ids.json), SHA-256
`389216b176c694cd5f43e3068b20d6fad97e09cbd956bcd6c2837d6b400858b2`.

Source checkpoint `nvidia/Gemma-4-26B-A4B-NVFP4` revision
`a19cfe00be84568a6867111c9a68c9c44fdcffe6`, corrected artifact
`1d20ce879b2fe72c947699357da5e20d3f6388e9cc7858b9eb6fcc2516cd3875`
and immutable reference image
`sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`
were unchanged. Native runtime SHA-256 was
`d18c7b7a9c8c6cf383a6f607a8606d2ce2026c4b2a8cb47632ec424601704dff`;
its actual 2,828-file source-manifest hash was
`d0c8f0023776b0994de9e97beff04cb7bcd3e77ddad78bf87390bb312794cff4`.
The stale Git-derived version is not the runtime's source identity. Native used
A's unchanged historical Service tree; collectors came from a private committed
snapshot. No rebuild, reimport, restaging or weight transformation occurred.

Native ran first, then source, on Spark A in separate installed exclusive jobs:
`m35-u3-gemma26-short-native-r1` and
`m35-u3-gemma26-short-source-r1`, each limited to 600 seconds with 30-second
termination grace. Context 4,096, prefill 512, one slot, T0, no drafter or prefix
caching and the prior precision conditions were retained: native F16 KV versus
source BF16 arithmetic/BF16 KV. Native requested zero generated tokens. The
source API required one unused generated token: ID 236772 (`-`), finish
`length`; it was excluded from all scores and counts and is not a behavior test.

Complete requests and raw returns were saved before assertions. Native's
complete echoed `token_id:<integer>` labels and the source's actual prompt IDs
matched all 1,024 frozen IDs. Each arm had an unscored leading `None` and
exactly 1,023 finite, nonpositive actual-teacher log probabilities at positions
1–1,023. Both loss files were reconciled with their complete raw returns. Independent
raw/numerical/provenance review was CLEAN, including all teacher positions and
the prefill boundary; its receipt is recorded in `results.json`. The substantial
per-target disagreement remains unexplained.
Source live inspection retained all 30 NVFP4 expert blocks on
`FLASHINFER_CUTLASS`, `use_a16=false`, Triton attention and
`VLLM_USE_V2_MODEL_RUNNER=1`. The previously reviewed ordinary scoring path
normalizes the complete 262,144-token vocabulary after Gemma's final softcap 30.
All 37 installed source-file hashes matched before and after collection.
The original source warning remains unresolved and verbatim in the log:
`w1_weight_scale_2 must match w3_weight_scale_2. Accuracy may be affected.`

## Provenance, retirement and limits

Both supervisors returned zero. Native retired with TERM/return code 0; the
exact named source container was removed and its absence confirmed. Runtime,
historical Service, registered helpers, metadata and loader checks matched;
postflight verified the unchanged source tree and snapshot, with no running or
waiting jobs or GPU compute processes. Native dependency verification compared
normalized library resolution; this does not claim fresh full-library payload
hashing. Prior staging and full artifact-transfer verification remained the
weight provenance; this pair did not rehash weight payloads.

[results.json](results.json) records full metrics, identities, registered file
hashes, preflight/enrollment/postflight, raw response/loss/launch/log hashes and
both supervisor receipts. Raw samples remain outside Git at
`/tmp/llmp-gemma26-short-{native,source}-r1`, with supervisors in the corresponding
`-supervisor` directories. Spark A copies are under
`~/.local/share/llmp/m35-u3-quality/gemma26-short-{native,source}-r1`.
The frozen snapshot and pairing receipts are at
`/tmp/llmp-gemma26-short-registered-eadf8da8`.

This one short pair does not establish own-repeat stability, depth, broader
behavior, batching, speed or general source-quality qualification. The
[source-template Paris pass](../paris/RESULTS.md), earlier
[raw-ID negatives and normalization diagnosis](../../GEMMA26-NORM.md), failed
source launch, missing-artifact preflight and [first CPU-preparation timeout](PREPARATION.md)
remain unchanged. No new diagnostic or collection protocol is enrolled here.
