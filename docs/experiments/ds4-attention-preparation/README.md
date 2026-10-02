<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Attention producer preparation — one literal 8K inverse factor

Completed on Spark A (`spark-c4e2`) on 2026-10-02. Keeping normalized
F32/F16/D4 production fused and shared is **2.2072% faster** than preparing
the same consumer inputs separately. The unshared candidate adds
**166.865 ms** to this literal pipeline. This is a small causal contribution;
it cannot explain the descriptive native ~9.4 s versus literal ~7.5 s gap
alone. It is not a measured native integration gain or an additive gain.

| Paid 8K prefill, including full-head copy | Seconds |
| --- | ---: |
| Original fused, before | 7.544671416 |
| Candidate, unshared preparation | 7.726922902 |
| Original fused, after | 7.575444969 |

Candidate rate gain versus the original mean: **−2.1595%**. Original
bookends moved **0.4079%**. All six complete 129,280-F32 heads, including
each arm's warmup, matched the retained literal golden byte for byte:
`499a05df44162d26dd151f44003388a68a494c86265ea756fecdabed85ae13b8`.
Own-repeat and finite checks passed; changed values and maximum drift were
zero. Completed original passes counted 86 fused producers. Candidate
passes counted 86 F32-only RMS producers, 172 standalone D4 preparations
and 290 F16 conversions, as predicted.

The candidate retains the original 256-thread weighted RMS arithmetic,
F16 conversion and D4 format. Every product, query-B/output-B consumer,
cache, precision and configuration stayed original. Both arms funded the
ordinary sidecars and guards, 512 MiB native pool and 32 MiB cuBLAS
workspace; neither allocated diagnostic capture storage. All preparation
launches and traffic were paid. This isolates producer fusion/reuse from
the separate native-versus-literal product accumulation/output rounding
question.

Actual private compile/link took **12.889 s**. The one model command took
**348.414 s**, including one **291.428 s** weight preparation and **6.402 s**
load. The supervised job took **364.365 s**, completed rc0 and reaped its
children, with no run repairs or warm source/archive mutations. The final
strong terminal gate was **117.161 GiB**, with native/GPU/container jobs
clear; Spark A was explicitly returned.

Closure: normal warm products tree corresponding to root `9dd6efd`, locked
GGML `cc7b7f0b962e3f536ccdd051161b6fb2117feac5ac8e2c0d6c1a417cfb394520`.
All four private host enum consumers compiled against its actual current
headers and static archives. Actual symbol proof admitted the five-parameter
MMQ definitions and scoped occupancy-two bridge, rejected old four-parameter
definitions; the link map rejected old benchmark runner/executor/binding
members. Binary SHA:
`68b4a16c69486bf628e8e66ed52ff770b6c3b37d58f346473e70a35b83c9c599`.

Recommendation: record this measured contribution and keep broader gap
isolation ahead of a native producer port. Any later port must retain native
normalization order and receive its own paid integration measurement. The
reuse mechanism can transfer where several consumers need identical input
formats; these numbers establish no gain for Qwen, image or other kernels.

Evidence retained locally in `screen-r1/receipt.json`, its command receipts,
all six raw heads and `supervisor-r1/{job,final,child}.json`. Private source
kit: `/home/pmeenan/scratch/m3-ds4-attention-preparation-factor/`.
No production adoption, wider quality claim, suite or context ladder.

Provenance: actual screen receipt Git blob `4b73c2129b2541e8e203f3287011164e6a389bef`, controller SHA256
`59bad4dc7b3e9f210450bc7ee75da0b242d8cd049efb32f8958c14c41db25489`.
Raw records, including six complete heads, are retained at
`/home/pmeenan/scratch/m3-ds4-attention-preparation-records/`.
