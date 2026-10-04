<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen prefill chunk trades — 2026-10-04

Keep the **4,096-row fallback**. At four simultaneous 8K requests,
8,192 rows gain 2.69% completed-token throughput with 4.58% more peak
memory; 2,048 rows lose 9.28% and save 2.13% memory. A separate four-slot
prompt family is 3.75% slower through the entire in-process check at
8,192 rows. These mixed results do not select a new default or a
machine calibration value.

The difficult 32K fixed-history anchor gives identical drafts and all
plain/speculative logit rows at 4,096 and 8,192 rows. Four-slot waves also
preserve every generated token and target logit row. Their whole-state
hashes compare different layouts and cannot establish state equivalence:
`Qwen38MtpStateOf` sizes its history buffer to `max_rows + 1`.

Both hosts are DGX Sparks (GB10, driver 580.178.04, SDK
`aarch64-e0a0c85c42806fb1`, CUDA 13.4.92). Target NVFP4 artifact is
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`,
drafter `8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`.
Every comparison is control/candidate/control with fresh processes,
changing only the prefill chunk setting within its triplet.

## Matched four-request HTTP screens

Spark A uses the frozen [concurrent fixtures](../qwen38-four-request-waves/README.md),
context 33,792, selected head 47,172 (requested cap 65,536), four slots,
normal speculative serving,
wave lanes, greedy sampling and the unchanged two-slot draft-wave limit.
Each request pays 8,256 uncached prompt tokens and returns 256 tokens,
HTTP 200 and `length`. Counts come from complete buffered responses;
there is no first-token latency measurement. A one-token weight-prime
request before each burst is excluded from its clock and counts.

| Triplet | Rows | Completed tok/s | Burst wall s | Each request latency s | Peak MemAvailable drop GiB |
| --- | ---: | ---: | ---: | --- | ---: |
| Larger, before | 4,096 | 32.036705 | 31.963337 | 31.963068 / 30.679747 / 31.428708 / 31.429091 | 81.289 |
| Larger, candidate | 8,192 | 33.026630 | 31.005283 | 29.599175 / 30.133715 / 31.005283 / 30.908908 | 84.970 |
| Larger, after | 4,096 | 32.284657 | 31.717852 | 31.717500 / 29.533292 / 30.258667 / 31.183916 | 81.206 |
| Smaller, before | 4,096 | 31.826258 | 32.174691 | 31.766063 / 30.821474 / 31.594143 / 32.173891 | 81.271 |
| Smaller, candidate | 2,048 | 29.210826 | 35.055496 | 32.613516 / 35.054941 / 33.500872 / 34.626898 | 79.445 |
| Smaller, after | 4,096 | 32.573067 | 31.437015 | 30.712767 / 30.256993 / 30.005460 / 31.437015 | 81.069 |

Ratios use the mean of the two controls. Control rate movement is
+0.774% / +2.347% for the larger/smaller triplet. All 24 measured replies
complete, totaling 6,144 output tokens, with zero cached tokens and no
errors. Services exit zero and are reaped; samplers finish without errors
or required KILL. No all-arrival-order reply identity or cross-engine
quality claim follows.

All six arms use the same runtime SHA-256
`64fccb395bbeabd2bb7569ec7597f7d9cc6672afb8abb236d4d5deaa7b99b9ab`,
the checked `ab83d05` implementation. Its independently qualified source
inventory is `ecdc71d00d4fea3c56c4aaa540ac49b183df852409a62e26330c3843ba54c0e9`.
This predates selected-head sharing in `5f37654`; these are chunk-factor
results on that fixed binary, not a new final serving comparison.

## Fixed-history and four-slot controls

Spark B uses the checked `5f37654` production source and
`jitllm_qwen38_spec` SHA-256
`444659b963f5777000a48f32b1812c26dba8c385bbdc6d0203ecce3862b7d533`.
No numerical or library environment overrides are set.

The [same-history study's](../qwen38-same-history/README.md) `p3` anchor
contains 31,746 literal IDs, SHA-256
`184d7c02214ef662c0e927ca63f4e0091eb42a918a1f79423c1be34b7303ae1e`.
The check runs context 33,792, fixed depth three, selected head 47,172
(requested cap 65,536), nine
outputs and two speculative repeats, including plain and teacher-forced
controls. Each arm passes the existing near-tie and own-repeat checks.

| Rows | Entire process wall s | Peak MemAvailable drop GiB |
| ---: | ---: | ---: |
| 4,096 before | 66.716122 | 78.304 |
| 8,192 | 61.751360 | 80.604 |
| 4,096 after | 66.304633 | 78.368 |

Candidate process wall falls 7.16%; this includes model loading, planning,
multiple prefills and generations, rather than isolating prefill. All
nine output IDs, full plain/speculative logit captures and the first
verify agree across arms. The first drafts remain `579, 1622, 13`, with
one of three accepted; increasing chunks does not repair this acceptance
loss. First-verify SHA-256 is
`5a2d9ad6882028869e92b7427d99dc0065a49c2fa2c5c0493800170b8615f556`;
plain/speculative logit digests are
`8babec050c74e12fc8e8cf98bc1b63c0ad680bd298326fc088e7653df7f7386a` /
`2169f3ee8409a094c62c619c6d78c9d55b5b50fb0b229ae5db0f8539668e01df`.
This representative native check is not a broader fixed-oracle quality pass.

The four-slot control uses the [wave-lane long fixture](../qwen38-four-request-waves/README.md#wave-lanes),
context 16,384, depth two, 96 outputs per slot and lanes on.

| Rows | Entire process wall s | Draft / verify median ms | Peak MemAvailable drop GiB |
| ---: | ---: | ---: | ---: |
| 4,096 before | 42.618481 | 23.004 / 106.372 | 83.011 |
| 8,192 | 44.105934 | 23.070 / 105.156 | 86.420 |
| 4,096 after | 42.402553 | 23.244 / 106.264 | 82.825 |

Each arm completes 384 outputs, 42 waves, one capture and 37 replays.
Every slot's full target-row and token digests agree across arms; the
4,096-row controls' final-state digests agree with each other. The
8,192-row whole-state digests differ. The MTP history buffers have 4,097
versus 8,193 rows, so those hashes are not a comparison of equal state
geometry. No claim about equal common-state bytes or the cause of every
byte difference follows. Normalize component ranges before using this
check to judge cross-chunk state arithmetic.

## Provenance and remaining work

All 24 strong 105-GiB admission/retirement gates pass. All six HTTP
services and six benchmark children exit zero and are reaped. Each of
the four model jobs uses the installed GPU-locking supervisor with a
600-second limit and is waited on. Source and binary identities are
checked before and after the in-process arms; input identities are
recorded at controller startup.

| Item | SHA-256 |
| --- | --- |
| Larger HTTP controller | `c9d446e6957f1a6d9e62564501205026f350032150dae63b4644eaa85e97acfa` |
| Smaller HTTP controller | `5cf01eacb6f93d5bb4076bb1a636278212fb0fe8a03a4230effffd0ebd383a73` |
| Frozen HTTP client | `4f76adb8e36bb97e04c30f22d28d8d2ffc985d67aa4621f083ac92fc5e85d8f3` |
| Frozen HTTP input receipt | `d5a35e6341e53de0286cfd777e4fadd707c12cf9d18f95010a4a38ffac0ab59d` |
| Fixed-history controller | `9d7d2798da49750b6ac198d0adf33bd0491147fdb85cdfda7f4ad2af4f09da36` |
| Wave controller | `940a995082938da92d43f861deba2c7f9546818d5101b60dd42df336874d6e10` |
| Native benchmark source inventory | `4a05996429a4d9f1988dda047782ab61897c50fe83fff61096d5266149b77235` |
| Literal anchor JSON | `cf5d310b278d56aac057d0fba445248bebe7bb5764036893a8eceae16c3fd46d` |
| Long wave JSON | `5613e76b48e0d31b899099ae0d127f306afe36341b5bb523c177c80cbc5c6022` |

The HTTP controller additionally checks its Spark A library/source tree
inventory `153f5b2cdb456b42ff73a62cb19a4c082178d88bda7d1eba8cb194f8eb8346d5`;
that tree is the earlier head prototype, not the separately compiled
control runtime. The control's independent source identity above supplies
its provenance; no prototype candidate binary runs in these HTTP cells.

Raw controllers/results remain under `~/scratch/qwen-prefill-chunks/`
on both Sparks and
`/home/pmeenan/scratch/jitllm-m3-qwen-prefill-chunks-2026-10-04/` locally;
the local copy excludes serving state/spill. Larger chunks remain an
optional configured trade, with no new default or calibrated value.
DeepSeek's controlled chunk selection, both models' controlled slot-knee
calibration, normalized state comparison and wider prompt/context
qualification remain open.
