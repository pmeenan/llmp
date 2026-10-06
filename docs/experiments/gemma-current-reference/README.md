<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Current Gemma solo performance against fresh ring-cache references

The current 26B research recipe retains a measured 2.43% prefill and 0.92%
decode latency gap against fresh llama.cpp bookends. The 31B comparison has
stable native prefill but a large reference spread, so it establishes neither
parity nor a speed win. Native 31B mean is 2.08% slower than the closing reference;
its decode mean is 2.13% slower than the reference mean. These are short solo
measurements, with the existing quality and batching gates still open.

| Seconds | Reference first | Native first | Native repeat | Reference repeat |
| --- | ---: | ---: | ---: | ---: |
| 31B prefill | 12.3124 | 11.1940 | 11.2168 | 10.9769 |
| 31B decode | 3.12017 | 3.18827 | 3.18866 | 3.12374 |
| 26B prefill | 2.41652 | 2.47975 | 2.47556 | 2.42106 |
| 26B decode | 0.680048 | 0.689799 | 0.688509 | 0.685692 |

31B native prefill spread is 22.8 ms, versus 1,335.5 ms for the reference.
A comparison to the reference mean would favor native by 3.77%, but the slow
first reference makes that unsuitable evidence of a gain. The closing 10.9769 s
reference is nearer the earlier 10.85–10.87 s bookends. No timing is excluded,
and no cause is assigned to the reference movement. Decode has much smaller
spread. 26B native prefill spread is 4.19 ms and reference spread 4.54 ms;
its means are 2.477655 s native versus 2.418790 s reference, a 58.865 ms gap.
This experiment measures the accumulated current recipe rather than attributing
a gain to an individual optimization.

Both reference arms on each host use the unchanged original ring helper,
launcher, model, input and image pins. Actual logs confirm context 16,384,
batch 8,192, ubatch 256 for 31B or 1,024 for 26B, F16 keys and values, SWA full
false and unified KV false. Physical local/global cells are 1,280/16,384 for
31B and 2,048/16,384 for 26B. Normal stock fusion and CUDA graphs remain enabled;
32 decode graph reuses are observed. No new CPU thread override or affinity
policy is introduced: the original helpers retain llama context defaults.
All heads match the earlier reference hashes, and both containers per host
are proven absent after checked retirement. A light post-run 31B host snapshot
shows no queued GPU job or compute process and an idle GPU at 41 degrees C /
208 MHz; it does not establish clocks or temperature during the timed arms.

Each native arm preserves the two complete finite heads, all initialized state
bytes, layout and 32 continuation choices from its previous native baseline,
and its own repeat. The 31B state is 1,740,636,160 bytes; 26B is 592,445,440.
All 32 reference/native choices agree in this fixed input. Cross-engine head
identity is still limited: 31B prefill maximum raw difference is 0.41361475,
while its final head is byte-exact; 26B prefill/final differences are
0.85720921/0.22439957. No tolerance or quality gate is waived. The separate
current representative corpus comparison, especially the 26B positive-margin
choice failures, remains authoritative for quality qualification; two retained
heads and 32 fixed choices do not replace it.

Native phase accounting remains available without another trace. 31B requires
32 plan calls with 31 hits / one miss and 31 built/cached forecasts without
refusal; 26B requires eight calls with seven hits / one miss and seven
built/cached forecasts. All native paired budgets are identical. The chunked
state API remains selected, with allocation and initialization inside paid
prefill. Phase spans and exact identities are retained in [results](results.json);
CUDA event spans include CPU submission gaps and do not isolate GPU arithmetic.

Reproduce in order reference, native, native, reference using the unchanged
owned-container launcher from the [state-only comparison](../gemma-state-only-prefill/README.md)
and its [31B](../gemma-swa-ring-h1/README.md) /
[26B](../gemma26-swa-ring-transfer/README.md) ring recipes. Native uses
`jitllm_gemma_prefill ARTIFACT IDS NEW_OUT PROFILE POLICY ROWS normmul-on
state-only lookahead-on phases-on state-chunked`, with 31/both/256 or
26/all/1024. Both engines discard six warm rows, reset state, pay for 8,192 rows,
append three untimed anchors and execute 32 forced decode units. One paid
prefill and one final full head are retained; publication, snapshots and scans
follow the existing timer boundaries. These explicit research norm-chain
policies remain distinct from ordinary serving defaults.

31B runs on Spark A (`spark-c4e2`) and 26B on Spark B (`spark-56f5`). The retained
native helper SHA is `68368ed6`, SDK receipt `b2174122`: source `845617f` plus
the [state-preparation helper argument](../gemma-state-preparation/README.md).
Its selected arithmetic and executor match `ab88b9d`; that later commit adds
an unselected experimental attention module. This is not a binary built from
the latest main commit. Native NVCC is 13.4.92, toolkit 13.4.2 and driver
580.178.04; original-image math uses CUDA 13.3. Full input, source, binary,
launcher and image pins are in results. Task-entry TensorFold HEAD was checked
at 2026-10-06 00:07:53 UTC and remained `609ca419` / 0.6.5; its
[pinned primary model table](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md#models)
provides Gemma26 on MLX without a matching CUDA Gemma26/31 target.

Both four-arm screens and both existing exact checkers retire DONE0 under the
installed GPU supervisor. Raw logs, heads and state remain outside Git under
`gemma-current-reference-raw`. No source change, extra model run, trace or full
suite is part of this task. The stable 26B residual and variable 31B reference
remain measurements; corpus quality, C4 batching and the accumulated regression
gate remain separate work.
