<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 same-quantized-checkpoint reference

The pinned Mia implementation successfully scored the existing four-document
Qwen corpus: 1,016 next-token targets, mean NLL 3.559477762612 and perplexity
35.144838403. This supplies the quantized reference required by the revised
quality policy; it does not require an original BF16 checkpoint or matching
head bits. It is a short-corpus observation, not a family acceptance gate.

## Observed comparison

The reference and current-main native A/C observations use identical four
255-token histories, without BOS/EOS or a chat template. All positions 1–254
are scored independently per document using the actual target token. The
reference tokenizer independently agrees with all four native token prefixes.

| Implementation | Targets | Mean NLL | Perplexity | PPL relative to reference |
| --- | ---: | ---: | ---: | ---: |
| Pinned Mia, target only | 1016 | 3.559477762612 | 35.144838403 | baseline |
| Current native A, ordinary joined products | 1016 | 3.563363450261 | 35.281665930 | +0.389325% |
| Current native C, three explicit product tuning keys | 1016 | 3.556988278363 | 35.057454697 | −0.248639% |

The current main diagnostic completes both arms with all 1,016 finite losses
and 1,020 finite full-head rows per arm, including terminal rows. Both
experimental policies disable `owner_row_invariant`; C additionally selects
existing physical-eight vector products for 48 routers and two PLE products.
Each recorded plan has 2,482 implementation entries. At eight physical rows,
exactly those 50 entries change from `ggml.mul_mat.mmf` to
`llmp.mul_mat.mmvf_rows`; the four-row remainder retains the same implementation
names. The complete recorded product geometry otherwise agrees.

C reduces aggregate PPL by 0.635489% relative to A, with 518 target losses
lower and 498 higher. Two documents improve and two regress. Per-document
PPL changes against Mia are:

| Document | A versus reference | C versus reference |
| --- | ---: | ---: |
| 0 | +0.157242% | +1.339807% |
| 1 | +3.475109% | −1.147987% |
| 2 | −0.185127% | −1.259724% |
| 3 | −1.816925% | +0.095694% |

Every current A/C loss also equals its corresponding historical
[broader native screen](qwen-joined-score.md) observation. This is a post-run
stability observation across different source/binary snapshots, not an
acceptance requirement. That earlier screen found a gain dominated by its
initial prefixes. Neither experimental policy is adopted here. Broader
default-policy qualification, long-context behavior, MTP and serving migration
remain open; no claim about closeness to the unquantized model follows.

## Matched one-row mapping-policy comparison

A separate current-source pair retains identical histories and uses one row
per owner for all 255 waves. Neither arm supplies explicit tuning entries.
The first retains the mapping policy; the second disables only
`owner_row_invariant`. Factory prefill settings remain two rows and all 16
factory classes are measured under each arm's actual policy. Both execute
the existing four-owner Score4/read256 class with 1,020 finite full-head rows
and all 1,016 targets. The plan's owner-row policy flag differs; the
`StepShape::row_invariant` flag remains false in both.

| One-row policy | Mean NLL | Perplexity | PPL relative to Mia |
| --- | ---: | ---: | ---: |
| Unchanged mapping | 3.566793653199 | 35.402897009 | +0.734272% |
| Only owner-row restriction disabled | 3.563363450261 | 35.281665930 | +0.389325% |

Disabling the restriction reduces mean NLL by 0.003430202938 and aggregate
PPL by 0.342433% relative to the unchanged mapping: 525 target losses improve
and 491 worsen. Documents 0 and 1 improve; 2 and 3 regress. Every disabled-arm
loss equals the earlier two-row A observation, descriptively, with no such
identity required by the run.

Each actual program records 2,482 implementation entries. The default arm
has 51 `llmp.mul_mat.mmvf_rows` entries; the disabled arm has 51 corresponding
`ggml.mul_mat.mmf` entries, with all remaining names unchanged. Default
policy can select these vector implementations without explicit tuning.
Names do not expose every internal per-owner operation parameter, so this
is a whole-policy comparison, not isolated kernel-error attribution. The
short aggregate result favors disabling the restriction relative to this
quantized reference; it does not establish agreement with an unquantized
model, a general quality improvement or a serving-default decision.

A prospective policy change still needs ordinary Prefill/Decode evidence at
one and four owners, including same-policy state continuation across those
cohorts. This bounded next check can reuse the existing factory and state
controls; no larger corpus or context ladder follows automatically. Ordinary
serving admission, MTP, long-context and migration-speed gates remain open.

Mia uses FP8 KV and BF16 SSM; these native observations use F16 KV. The
arithmetic and cache choices differ across implementations. We compare their
complete losses without forcing a common rounding sequence.

## Checkpoint, policy and result provenance

Spark B (GB10), 2026-10-12; `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` revision
`925d7be6c14c6c9442ef83e8f05b5a3c39304f69`, using the standing source checkpoint.
The [pinned fast-start launcher](../fast-swap/baselines.md#default-mia-launcher-for-new-runs-2026-10-03)
selects recipe `b8439110eec0230facbe4ddf0dffe01b8f769be0`, image
`sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`,
instanttensor 0.2.0 and the existing CPU PLE loader override. The effective
loader, source metadata and indexed payload inventory were checked before
and after execution; this run did not rehash every large weight payload.

Target only, MTP 0, DET 1, `QSA_DET_TOPK=1`, `MOE_DET_FINALIZE=1`, maximum
context 33,792, four slots, batch 2,048, GPU memory utilization 0.786,
26 GiB host reserve, graph mode 0 with full-decode-only captures at sizes
1/2/3/4. A fresh credential-free private recipe uses read-only checkpoint
mounts and the existing packed PLE. Full-context capability is not tested
by the four short documents.

| Identity | SHA-256 |
| --- | --- |
| Safetensors index | `435eef76fc10fc6e932a208a1f85a86bc9c7ffc389b27b34ba83bb6a5d0371e9` |
| Target tokenizer | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| 1,020 little-endian int32 input IDs | `3cd946c3876501eccbe079b9d2700e542e66242be241a8549546df8d9b23e468` |
| Collector source | `2808e63be9c08fe1378bf88e5dcf2f09835681d9f50894ea198bc6f61514621f` |
| Reference actual receipt | `1dd33b3265f61cc993973c5651cde03db6b90660b3fa3c9f1cb73c476070f8e9` |
| Historical native actual receipt | `92ad695820d1eb909d5d4336a57a7525d34b2fba5ca622c459b7743f904c535b` |
| Current native full-build/32-host receipt | `d85243126855c65d387425da3432f2519fc0b03ca62f2dfb5e31a597d346cc3b` |
| Current native model binary | `d70b3a0a64f473b74aa2d1084fe4d67f15f959c59e9c6260782300f8fa400987` |
| Current native actual receipt | `7a83de2515d5d28fae3244abda23335878f982f97fe6f7c45e70cb1542f6c304` |
| Independent current actual review | `7d7bb8b8cdf6fb27a66007e084f16101f8cc354c354e0f338db2fd97f8d37986` |
| One-row full-build/32-host receipt | `18c0246b84b53d429e18b2e4cb87afd5e0e468d3c640907e8adc7755befbf489` |
| One-row model binary | `ec131650499b2815f907518f8a052e3a654ad3ff5045834cb539747a1680fdfb` |
| One-row actual receipt | `dcb203a6b4be744d0e0db234a765ea47f6c25c682dfff8bf78de73b6bd6e5246` |

Each document has 254 targets; reference summed NLL is respectively
747.6428986233683, 1006.8320033252003, 985.9432543228786 and
876.011250541982. All 1,016 unrounded losses have complete, unique target
coordinates and finite nonnegative values. Independent review recomputed
document sums and the token-weighted aggregate.

The supervised reference run completed in 213.255 seconds, including
206.469 seconds to readiness. These are startup observations, not inference
speed measurements. The owned launch group retired, its container was
removed, memory-limit/OOM events were zero and no GPU process remained.
A preceding attempt failed before scoring because a cached absolute model
link was unavailable inside the container; mounting its unchanged resolved
checkpoint read-only fixed the environment. Its failure remains recorded.

## Current native verification and replay

Spark A, 2026-10-12: main native build and all 32 selected host controls pass;
the manually invoked model target has no automatic CTest registrations.
Both fresh native cases exit successfully without skips. Source and binary
guards pass, owned process groups retire, no GPU process remains and all
memory-limit/OOM counters are zero. Peak process/cache charge is
1,111,531,520 bytes; this is not device occupancy. The 46-second supervised
job includes construction, storage and scoring and is not a latency gate.
Pinned formatting, boundary/header checks and REUSE 6.2.0 pass.

The subsequent one-row pair also passes the main native build, 32 host
controls and source quality checks, with four manual cases listed and zero
automatic registrations. Both actual cases exit zero; guards, whole process
group retirement, GPU-process and memory-event checks pass. Its process/cache
peak is 1,038,852,096 bytes. The unbookended 55-second job and individual
elapsed counters are not a comparative speed measurement. An earlier wrapper
incorrectly treated any vector selection as explicit tuning; its failed
receipt and successful default native case remain preserved. The corrected
wrapper retains the complete selection inventory and all numerical/lifetime
gates, and both fresh cases pass.

Build `qwen38_joined_score_gpu_test`, then replay the two cases
`Qwen38JoinedScoreGpu.OrdinaryBroaderNaturalTeacher` and
`Qwen38JoinedScoreGpu.TunedBroaderNaturalTeacher` through the supervised
captured-input recipe `qwen-corpus-current-models02` in the milestone scratch
coordination directory. Its `run.py`, `run.steps`, input receipt and literal
source/build/binary guards remain together. This diagnostic is deliberately
not an ordinary all-model CTest case: it requires the captured inputs and
explicit resource admission.

For the matched one-row comparison, use the separately sealed
`qwen-corpus-default-policy-models02` recipe and its current literal binary
and source receipts. It invokes only
`Qwen38JoinedScoreGpu.DefaultSingleRowNaturalTeacher` and
`Qwen38JoinedScoreGpu.JoinedSingleRowNaturalTeacher`; the earlier two cases
remain available in the same manual target. Both recipes use the same captured
input identity and the resource/retirement requirements below.

The wrapper supplies `LLMP_TEST_MODELS`, `LLMP_QWEN50_IDS` pointing to the
4,080-byte fixed ID file authenticated above, `LLMP_QWEN50_IDS_SHA256`, and a
fresh per-case `LLMP_QWEN50_SCRATCH`. It uses installed `spark-job --gpu` with
an explicit wait, a 112 GiB zero-swap process/cache ceiling, the 104 GiB node
budget including prepaid construction/observations, and a separate 6 GiB
available-memory margin. Preserve the 512 MiB per-file allowance for the
314 MiB mapped spill files, separate 8 MiB log/XML bounds, source guards and
whole-group retirement on every exit. These limits apply even without an
explicit spill test because slot registration creates the mapped files.
Raw results and the preceding file-limit refusal remain outside Git. A new
source build requires refreshed literal guards, not reuse of an old binary
identity.

## Reusing the collector

Copy [collect-qwen-reference.py](collect-qwen-reference.py) to a fresh scratch
directory as `collect.py`; outputs are created exclusively beside it. Supply
the frozen `inputs.json` and `doc0.txt` through `doc3.txt` from
`/home/pmeenan/scratch/m36-qwen-quant-reference03` on Spark B, or the local
coordination copy `qwen-same-quant-reference03`. These are retained captured
replay inputs, not checked-in raw results. The metadata SHA is
`50110476b05de2b2a87e012b46cff967d179c48917406797368309eb3f27e25d`.
Document hashes are authenticated by that metadata and the collector; do
not substitute new text under the same measurement identity.

Admit and supervise the pinned reference server separately through installed
`spark-job --gpu`; use the declared policy above, model name
`qwen3.8-flash-next` and owned loopback port 18253. Run
`python collect.py http://127.0.0.1:18253 qwen3.8-flash-next` within that job.
The collector bounds responses, rejects missing/nonfinite targets and emits
`result.json` only after complete scoring. It does not start, admit or retire
the model; the owning wrapper must verify source/runtime identity and retire
all work even on failure. Six offline positive/negative controls and the
actual run checked this unchanged collector. Raw losses, logs and actual
receipts remain outside Git.
