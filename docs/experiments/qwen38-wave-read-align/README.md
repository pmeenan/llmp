<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen C4 wave read-alignment screen — 2026-10-04

Keep `wave_read_align = 2048`. A fixed-history C4 wave follow-up finds a
small **0.851% verify-latency saving** with complete token-history and verify-row
hashes equal. The serving screen remains **inconclusive**: candidate rate is
6.845% above the inverse mean control duration, while control duration moves
−14.693%; candidate is 1.653% slower than the final control. This does not
support a default or calibration change. There is no production change and
no broader matrix.

## Factor and paid work

The existing setting is forwarded unchanged from resolved model settings
into the Qwen runner. `WaveReadAlign` uses it for waves with more than one
request; lone-request read geometry remains 256 cells. `Qwen38Chunk`
rounds a wave's end into `n_kv` and the indexer block count. `StateRanges`
and `Qwen38UsedState` back target and drafter K/V/indexer ranges through
the same padding, so speculative or masked reads never reach absent
backing. The QSA path suppresses positions beyond the actual history.

At 8,256 prompt tokens plus 256 outputs, the multi-request target geometry
stays in one bucket for each setting: **10,240 cells / 2,560 indexer blocks**
at 2048 versus **9,216 / 2,304** at 1024. The QSA selected-width budget stays
2,051 cells. The screen therefore tests a smaller padded read geometry;
it does **not** cross an alignment boundary or test the hypothesized
increase in plan churn at finer boundaries. Plans include `n_kv` and QSA
geometry in their keys. End-of-service plan/graph totals below are
descriptive observations, not isolated planning or kernel timings.

The prior [GGUF qualification](../qwen38-gguf/README.md) compares joined
and unjoined waves at the same 256 or 2048 alignment. It explicitly does
not establish cross-alignment scalar equality. The HTTP screen
neither compares full heads nor reads state; the follow-up wave cell retains
every verify row and token history, and compares state only between the two
2048 controls. `ReadState` would first
settle owed restore, then copy actual initialized extents; different
initialized ranges and byte lengths across alignments would not be a
meaningful whole-state equality control.

## One C4 HTTP bookend

Spark A (`spark-c4e2`), GB10, driver 580.178.04, CUDA toolkit/compiler
13.4.92, runtime/device API CUDA 13.0, SDK
`aarch64-e0a0c85c42806fb1`. The one immutable runtime is built from
`068d544`, with the adopted GDN gates and calibration implementation from `995cced`. Three
fresh services have separate data directories and anchors. Every service
uses the same target artifact `c4fb47a9…5a93`, drafter `8600a998…ee40`,
checkpoint `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6`, context 33,792,
prefill chunk 4,096, four slots, speculation on, draft rows three,
depth-cost ratio 1.16, shared depth two, joined-draft limit two and wave
lanes on. Requested draft cap 65,536 executes the physical selected
47,172-row head. Only `wave_read_align` changes.

Before each service starts, the settings CLI must resolve every listed
policy to the expected override, including effective head rows and read
alignment. The actual registration log separately proves alignment
2048/1024/2048 and the fixed head/depth/lanes policies. Both native code
and artifact/checkpoint metadata are checked before and after each arm.

The frozen common-v2 client sends one excluded **one-output C1 prime request
on the normal four-slot service**, then four concurrent buffered Chat
Completions. Each measured response is HTTP 200, 8,256 prompt tokens,
known zero cached tokens, 256 outputs and `length`. All twelve measured
requests complete, totaling 3,072 outputs. All three prime requests also
complete uncached with one output. The clock runs from the earliest
measured submission to the final complete response, paying prefill,
generation, queueing and HTTP through completion; prime, startup and
retirement are excluded. Buffered responses provide no TTFT measurement.

| Chronological arm | Alignment | Completed tok/s | Burst wall s | Peak MemAvailable drop GiB | Plans / graphs held at retirement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control before | 2048 | 26.924156 | 38.032762 | 81.316944 | 90 / 41 |
| Candidate | 1024 | 31.048179 | 32.981000 | 80.664181 | 87 / 37 |
| Control after | 2048 | 31.561501 | 32.444591 | 81.232212 | 90 / 40 |

Candidate rate change is
`(mean(control wall seconds) / candidate wall seconds − 1) × 100 = +6.845383%`;
all arms complete the same 1,024 measured outputs. Control duration moves
`(after / before − 1) × 100 = −14.693044%`. The controls' larger movement
prevents attribution of the mean-based apparent gain to read alignment.
The memory column is sampled against each cell's own pre-service baseline,
with no sampler error; it is a whole-process physical-budget observation,
not an allocation maximum or isolated KV-memory saving.

Complete reasoning/content strings match the final control in all four
candidate responses. Both candidate and final control match the first
control only for `u1` and `u2` (two of four). The retained raw aggregate's
`text_candidate_equal = false` compares candidate with the first control.
Signatures hash `reasoning + NUL + content`; none of these actual fields
contains NUL. This is a text observation, with no cross-alignment full-logit,
state, perplexity, cross-engine parity or quality-bound claim.

All three services exit zero and are reaped. Their samplers join with no
error, and all six strong 105-GiB admission/retirement gates pass with clear
GPU, container and native-model probes; the smallest availability is
116.236 GiB. The installed GPU-supervised `qwen-read-align-build1` and
`qwen-read-align-screen1` complete zero and are waited on. The owned
minimal runtime build takes 154.766 s; the three service/retirement cells
take approximately 170 s in all. Frozen earlier control trees and binaries
are preserved. There is no production change or full unit rerun for this
rejected diagnostic under the M3 experimental override.

## One fixed-history C4 wave follow-up

The HTTP control movement leaves geometry cost unresolved, so one bounded
in-process 2048/1024/2048 cell uses the **same immutable benchmark binary**
with fresh state per arm. Its original current `qwen38_spec.cc` is adapted
outside Git only to accept a bounded power-of-two `--wave-read-align`, allow
literal IDs for the existing wave driver, and emit the setting plus initial
prompt/read/block counts. The strict token and context bounds remain. This
is a private diagnostic adaptation, not a new supported CLI or production op.

The four common-v2 rendered I32 histories are copied directly into the
existing literal-ID schema and checked against their original receipt:
**8,256 IDs per slot**. Initial three-row verify ends at position 8,259,
so the emitted geometry is 10,240 cells / 2,560 blocks for both controls
and 9,216 / 2,304 for the candidate. These metadata calculations and prints
occur before prefill and outside phase clocks. All 256-output trajectories
fit context 16,384 and cross no alignment boundary.

Artifacts, selected head, chunk 4,096, four slots, graphs and lanes stay fixed.
The wave driver actually drafts at **depth two**, as serving does when shared;
`--draft 3` supplies runner capacity. Its required nominal `--repeats 2` does
not repeat `--check wave`: each arm is one complete four-slot generation.

| Chronological arm | Alignment | C4 verify median ms | C4 draft median ms | Total waves | Verify captures / replays |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control before | 2048 | 97.000 | 22.079 | 121 | 2 / 116 |
| Candidate | 1024 | 96.436 | 21.984 | 121 | 2 / 116 |
| Control after | 2048 | 97.528 | 22.083 | 121 | 2 / 116 |

Candidate verify latency relative to mean controls is
`(96.436 / mean(97.000, 97.528) − 1) × 100 = −0.851291%`;
control verify latency moves `+0.544330%`. The medians use waves with all
four requests active. The verify clock pays `VerifyWave` preparation,
input concatenation, planning/capture/replay, kernels and logit copies;
caller work-vector construction and row hashing, prefill, draft and final
state reads are outside this clock. This is a verify-phase result, not an
HTTP throughput, isolated attention-kernel or complete-generation rate.

All three arms complete exactly 256 generated tokens per slot. All four
**prompt-plus-generated token-history** SHA arrays and full F32 verify-row
SHA arrays match across all arms, with the same 121 waves. This satisfies
the cell's predeclared output/work qualification. The target-plus-drafter
initialized-state SHA arrays also match between the two 2048 controls.
The raw cross-alignment state SHA arrays also match here, descriptively.
`ReadState` copies initialized 2-MiB extents: different padded requested
ranges can occupy the same extent set. These hashes are not used to claim
general logical-state equality across geometries. There is no cross-alignment
swap, common-state, perplexity or cross-engine claim.

All three benchmark processes exit zero and are reaped, and all six strong
105-GiB gates pass with clear GPU/container/native-model probes (minimum
115.788 GiB). The installed GPU-supervised `qwen-read-align-wave-build3`
and `qwen-read-align-wave-screen1` complete zero and are waited on; the final
host-TU build takes 39.119 s and all three gated cells approximately 171 s.
The preliminary flags-only build is unmeasured. A later build's metadata
separator fails targeted clang-tidy, is corrected, and is also excluded
without inference. Final compile, targeted SDK tidy and format checks pass.
The linked checked runtime archives are unchanged and authenticated before
and after every arm. No full production suite is repeated for this private
diagnostic under the experimental override.

The smaller geometry has a qualified small verify benefit here. It does
not resolve the noisy serving result enough to adopt 1024; retain 2048
without expanding the state/swap or workload matrix.

## Provenance and replay

| Item | SHA-256 |
| --- | --- |
| Runtime executable | `7869ecfdce90c340631daafa9116074331480a3fab455f283a2debcef3c5fa3b` |
| 487-file compiled-source inventory | `c4332305fc048aa9b340c09751e460c2a1651e1cae188e61b2f57d534a368fe2` |
| HTTP controller | `5e67e9c12f2804e7898ee974275dac99c48e7a1977c8036060eeefba74ecb80c` |
| Frozen buffered common-v2 client | `4f76adb8e36bb97e04c30f22d28d8d2ffc985d67aa4621f083ac92fc5e85d8f3` |
| Complete common-v2 input receipt | `d5a35e6341e53de0286cfd777e4fadd707c12cf9d18f95010a4a38ffac0ab59d` |
| Validated screen aggregate | `62cc1d6f16275641d156f3107c4533f416913e5f5ba33b2e38c6b19b5943da62` |
| Original current benchmark source | `e7c95af532772e89c24d5315b36751e8ff8fab82bc10c2156a184aad3c700e72` |
| Final private benchmark source | `e9bced9a5db0cea0b6e738e26db692ec3cfa04251558a00374527aeecc75aa36` |
| Private benchmark executable | `4293d427d2a58eb1d0f24063d4fc4596409649500abcfad588eb470018069877` |
| Literal four-history fixture | `1e6b10cafc5a2c1779ab6975446d09b5071df080e68961731c369ecf54b1d0da` |
| Fixed-wave controller | `8832875fcddd06c7f31a8cc61fed59cbffa154152e9489bdb7cb22bad561e58e` |
| Fixed-wave source/archive/binary pins | `6984eb71684973da5c13f0187f467dbfbe92100feae532222d34473b24d7d937` |
| Qualified fixed-wave aggregate | `739425c6bb5acd4ad9f0dad1c5170a1c37f3bf0d0f39d8d9abbd5f1d235db154` |

Reproduce with the same runtime/artifacts, frozen common-v2 inputs and
client, fresh state per arm, and the fixed settings above. Set only
`wave_read_align` to 2048, 1024 and 2048 in chronological order. Each arm
must retain the resolved/registered settings, complete public responses,
zero-cache usage, sampler and proven retirement before counting a ratio.
The fixed-wave follow-up uses the same four literal histories, final private
benchmark, fixed policies above and chronological alignment order; require
initial geometry, complete token/history rows and wave-count qualification
before attributing its phase timing. Full artifact IDs, metadata pins,
request/config bodies, private benchmark patch/compile commands, source/archive
inventory, raw logs and installed-supervisor receipts remain outside Git at
`spark:~/scratch/qwen-read-align/` and
`/home/pmeenan/scratch/jitllm-m3-qwen-wave-read-align-2026-10-04/` locally.
