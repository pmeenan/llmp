<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Complete ds4 stage profile

The matched native reference spends 35.55% of its measured GPU chain time
in routed FFN and 17.94% in attention output. These are the first restoration
targets. The profile preserves the original numerical pipeline and all eight
saved full heads are byte-identical to the original ds4 control. It selects
the next experiments; it does not adopt a new kernel or change production.

## Scope and controls

Spark A runs the [complete native reference](../ds4-complete-plan/README.md)
on the authenticated community IQ2_XXS/Q2_K artifact: exact 8,192 IDs,
43 layers, two 4,096-row chunks, original FP8 KV/FP4 indexer settings and one
final 129,280-entry head. One preparation takes 291.430 s. An ordinary
warmup precedes three unmarked passes, one diagnostic profiled pass and
three unmarked passes. Each pass initializes fresh state.

The private `--profile` benchmark allocates 16 provider timing marks before
warmup and reuses them after existing completed jobs. It records the eight
sequential layer chains plus embeddings and the final head. There are no
additional fences, model jobs or operand readbacks. Mark owners remain alive
through fenced retirement, including partial creation and failed submission.
With profiling disabled, the added code makes no timing-mark calls.
Original numerical cores, stage dispatch and paid producers are unchanged.

All eight full heads, including warmup, are finite and byte-identical to
the saved original head:
`499a05df44162d26dd151f44003388a68a494c86265ea756fecdabed85ae13b8`.
All seven measured passes have the same resolved dispatch, 93 native jobs
and the existing 86 layer fences. This fixed-input 8K equality control does
not replace real-answer, long-context or decode quality gates.

## Measured chains

One diagnostic pass records 691 ranges: 687 active and four explicitly
absent compression ranges. Active ranges total 7,433.175 ms. Each range
includes its configured producers and consumers; elapsed stream time is
not an isolated kernel measurement. There is one profiled repetition, so
these totals establish no per-chain variance estimate.

| Paid chain | Total ms | Share | Active / recorded ranges |
| --- | ---: | ---: | ---: |
| HC attention input | 279.520 | 3.760% | 86 / 86 |
| Q/KV production | 929.079 | 12.499% | 86 / 86 |
| Compression and indexer | 349.347 | 4.700% | 82 / 86 |
| Attention | 821.824 | 11.056% | 86 / 86 |
| Attention output | 1,333.221 | 17.936% | 86 / 86 |
| FFN input and route | 288.224 | 3.878% | 86 / 86 |
| Routed FFN | 2,642.226 | 35.546% | 86 / 86 |
| Shared FFN and expand | 783.775 | 10.544% | 86 / 86 |
| Embeddings | 3.276 | 0.044% | 2 / 2 |
| Final full head | 2.682 | 0.036% | 1 / 1 |

Compression consists of 42 active CSA ratio-4 ranges totaling 319.288 ms
and 40 active HCA ratio-128 ranges totaling 30.059 ms. Layers 0 and 1
have no compression: their four chunk/layer records are inactive and zero.
The attention-output range includes output-A, output-B and HC expansion;
the full 17.936% cannot be attributed to output-B alone. Routed FFN includes
guard initialization, route maps, fused gate/up, weighted SwiGLU, D2S6
activation preparation and the Q2 down product.

## Ordinary timing and memory

Comparable timing includes the complete prefill and final result copy,
excluding preparation, as in the original reference. Use the six unmarked
passes for performance comparisons; the profiled pass is diagnostic.

| Role | Fresh pass seconds |
| --- | --- |
| Original before | 7.539535222 / 7.556195682 / 7.572107647 |
| Original after | 7.578999017 / 7.574452314 / 7.585876878 |

The six-pass mean is 7.567861127 s, or 1,082.47 tokens/s. The maximum/minimum
ratio is 1.006146, and the after/before mean ratio is 1.003154. Both are
inside the fixed 1.1 stability limit. The diagnostic wall time is
7.569953428 s and its existing device-job sum is 7.434948118 s.

The execution budget is 98,922,016,256 bytes, with 656,408,576 bytes of
physical state. Provider free-memory readings before and after creation
of the 16 marks are both 17,556,877,312 bytes; this counter does not prove
that event allocations cost zero bytes. At 250 ms sampling, peak process
RSS is 902,381,568 bytes and the node MemAvailable drop is
102,725,160,960 bytes, including paging and cache. The model child exits 0
and is reaped; the strong retirement gate passes with 117.083 GiB available
and no active GPU job. Source and binaries remain unchanged after execution.

## First restoration controls

First isolate output-B's existing native GGML MMQ consumer using the
original paid D4 producer, raw Q8 weights, guard initialization and output
sanitizer. Keep stream, job fences, geometry and head cadence fixed. The
current native public product entry prepares its input itself, so borrowing
the existing D4 operand requires an explicit bounded native entry and its
own workspace contract. The completed [consumer singleton](../ds4-native-outputb/README.md)
is comparable at a 0.9983 rate ratio, with twelve full heads and three
complete original-input operator outputs byte-identical. It preserves the
native consumer seam; production dispatch remains unchanged.

The larger routed-FFN restoration separates aligned versus raw weight
layout, original versus native consumers, and fused gate/up/SwiGLU versus
separate products. Retain precision, routing and every paid adapter in each
control; avoid simultaneously resident full raw and aligned expert replicas.
Measure singleton changes before interactions. All model speed comparisons
use unmarked same-node repeats with original bookends and saved full heads.
The completed [first FFN factor](../ds4-routed-ffn-first-axis/README.md) finds
the fully paid Materialized path 2.35% slower than Direct, with all twelve
heads and the complete captured D2S6/down outputs byte-identical. This
producer/gather + fusion/storage factor does not explain the earlier large
engine gap; native compact consumer/layout factors remain distinct.

## Qualification and provenance

Measured on Spark A, 2026-10-01, 00:38:43–00:44:48 America/New_York;
supervised job `ds4-profile-model-r1`, exit 0. The five source changes passed
the full locked Spark-native suite: 1,197/1,197 tests, including 256 GPU
tests, in 80.01 s. SDK format and all four actual tidy compile arms pass,
along with boundaries and 1,104 SDK REUSE/header checks. The original
115 numerical definitions and ten prepared headers retain exact source
correspondence. Independent whole-source and controller reviews pass.

Raw samples, complete dispatch records, source maps, all saved heads and
receipts stay outside Git under
`spark:~/scratch/m3-ds4-profile-r1/{check-r1,model-r1}/`, mirrored locally
under `/home/pmeenan/scratch/m3-ds4-profile-{check-r1,model-r1}/`.
The compact `profile-extracted.json` retains every recorded range.

| Identity | SHA-256 |
| --- | --- |
| Community artifact | `cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac` |
| Exact prompt ID bytes | `60329b1e4ff5d19d40082666e8c08b86655d4b1173486aecbc4a752bb6aa3ac7` |
| Qualified source map, 1,224 files | `2f8e18baedd9f7cb1a20d5cebf95c5a98fda4f201b2201a1d074bdac778cae59` |
| Locked qualification receipt | `a2ff43ebde349932d82f4e62679a7f36f86b347d8552276649fe6f398e825e53` |
| Native benchmark binary | `4118554c16de6e77ff56454ee35634feea34a53b755c91e265bef4551d1ae130` |
| Model controller | `a9f24ac34379ad2a819529968ede8cdf47bdfeba87911c39d17673f0e5d24ed2` |
| Model receipt | `e804debde563d3e0d313486a6206756b3be47d2d577ca0fc7e4a72273c98ec54` |
| Native result receipt | `43e3b4717c528962cc5c2f7dab078b20bb7162b0a6da37df0b4aae680741ce7d` |
| Complete extracted profile | `62dbba8fc199f3ce08fe17d6b753332d01f68e46737e55536787cbfc9fe840ba` |
| SDK metadata, `aarch64-e0a0c85c42806fb1` | `f38891fcc394ddf956aaa196c6895d7309148b8b5159b0c3a0459721946992bf` |
