<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen MTP head sharing — 2026-10-04

Selected BF16 MTP heads now join across requests using GGML's vector
float arithmetic. A matched C2 HTTP screen gains **1.77% completed tokens/s**
at two uncached 8K prompts, with 0.04% baseline movement. Short and 8K
in-process controls preserve every generated token, full target logit row
and initialized target/drafter state byte. The production draft-wave limit
remains two; wider four-request drafting is not adopted.

Spark A (`spark-c4e2`), driver 580.178.04, CUDA 13.4.92, SDK
`aarch64-e0a0c85c42806fb1`. Target NVFP4 artifact
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`, MTP
`8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`,
65,536-entry prefix head, depth two in shared waves and
4,096-row prefill chunks. In-process context capacity is 16,384, with
96 outputs a request; serving context is 33,792, with 256 outputs.

## Implementation and arithmetic

The qualified head shape is BF16 weights `[2560, N]`, `16384 ≤ N ≤ 65536`,
F32 inputs and outputs. Only single-column products join, up to eight
columns. Inputs concatenate with their packing charged to the wave; each
slot receives its own output-column view. Prefix views must have equal
geometry and offsets over common immutable backing; selected heads require
the same immutable weight leaf. Different weights, diagnostic head capture
or disabled pairing retain separate products. Original arenas remain alive.

Per-product vector selection makes original and joined heads use
`jitllm.mul_mat.mmvf_rows`; the wave authenticates both plans against the
same implementation. This invokes GGML's existing MMVF arithmetic, also
used by the ordinary single-column head. Other float products keep their
device selector, including the full 248,320-entry target head. Existing
per-product overrides are preserved.

CPU planning tests cover prefix and selected heads across all three passes,
sparse slot IDs, unpaired/captured heads and different backing. The GPU
operand test compares all columns 1–8 byte for byte with ordinary
single-column MMVF at K=2,560 and N=16,384. Real-model controls below use
the 65,536-entry prefix; no new curated-head performance claim follows.

## Representative in-process screens

Each screen is control/candidate/control with fresh processes. Times are
separate phase medians, not a median or measurement of their sum. Packing,
attention, state work and capture/replay are paid.

| Screen | Control before: draft / verify ms | Candidate: draft / verify ms | Control after: draft / verify ms | Result |
| --- | ---: | ---: | ---: | --- |
| C4, identical joined drafting | 28.358 / 107.253 | 21.999 / 106.565 | 28.188 / 107.836 | Head sharing reduces draft time 22.19%; exact tokens, target rows and final states |
| C4, current production drafting | 21.782 / 107.795 | 21.920 / 106.767 | 21.834 / 107.261 | Draft time +0.51%; target rows/tokens exact, drafter-state bytes differ; wider drafting rejected |
| C2, short prompts | 11.179 / 67.827 | 8.944 / 68.410 | 11.095 / 68.203 | Draft time −19.69%; all tokens, target rows and final states exact |
| C2, two 8K prompts | 11.452 / 69.120 | 9.299 / 68.043 | 11.262 / 68.983 | Draft time −18.12%; all tokens, target rows and final states exact |

The isolated C4 control gives both arms four-slot joined drafting and the
same per-product vector selector, disabling only head joining in control.
The following C4 screen compares that bundled candidate with production's
per-slot drafting beyond two requests. Joining other MTP products cancels
the head gain and changes drafter-state arithmetic. It is not selected.

C2 already uses joined drafting in production, so that comparison isolates
the new head path. All arms complete: C4 has 43 verify waves, two captures
and 38 replays; short C2 has 39 waves, one capture and 37 replays; long C2
has 42 waves, one capture and 38 replays. The receipts record the actual
counts; no GPU execution claim rests only on the prototype's planning print.

## Matched C2 HTTP screen

Two simultaneous buffered requests each pay 8,256 uncached prompt tokens
and return 256 tokens with HTTP 200 and `length`. These are the frozen
[concurrent fixtures](../qwen38-four-request-waves/README.md), with greedy
sampling and four available slots. The service uses its unchanged two-slot
draft-wave limit. A one-token weight-prime request is excluded from the
burst. Counts come from complete responses; no first-token latency is
inferred from buffering.

| Arm | Completed tok/s | Burst wall s | Each request latency s | Peak MemAvailable drop GiB |
| --- | ---: | ---: | --- | ---: |
| native-before | 29.533053 | 17.336508 | 17.071167 / 17.336508 | 79.687 |
| candidate | 30.051415 | 17.037467 | 16.752736 / 17.037328 | 79.933 |
| native-after | 29.522702 | 17.342586 | 17.065443 / 17.342489 | 79.896 |

Candidate rate is 1.773% above the mean control rate; control movement
is −0.035%. Peak memory drop is within 0.2% of the control mean. All six
measured requests complete, with no errors or prompt reuse. Every service
exits zero and is reaped; samplers terminate cleanly, and all six strong
105-GiB admission/retirement gates pass. Arrival-dependent arithmetic in
the existing cohort remains; HTTP responses are retained without claiming
that every arrival order has the same reply or that Mia quality parity
is established.

## Provenance and limits

The control runtime is the checked `ab83d05` implementation, copied from
Spark B with SHA-256 `64fccb395bbeabd2bb7569ec7597f7d9cc6672afb8abb236d4d5deaa7b99b9ab`
and source inventory `ecdc71d00d4fea3c56c4aaa540ac49b183df852409a62e26330c3843ba54c0e9`.
The candidate runtime SHA-256 is
`36b3c1ae11a329c539da5a2be26343c9787a91a086ce77d7eb3dcc775dc15236`.
The HTTP controller checks the candidate tree throughout all arms; its
`source_pins_sha256` fields identify that tree, not the separately compiled
control binary. The independent control identity above supplies that
provenance. Both binaries use the same Spark A library tree and configuration.

The prototype benchmark allowed four-slot joined drafting for its C4
screens; the production implementation retains the original benchmark and
serving threshold. Final changes replace a planning-only print with a head
pair counter and preserve inherited vector-selection callbacks. The tested
default callback is empty, so those changes preserve the measured arithmetic.

| Item | SHA-256 |
| --- | --- |
| HTTP controller | `9f0188b85244053244032f0aa87a7e990ab863999f05cd7a2081af261d2dcf6c` |
| Frozen HTTP client | `4f76adb8e36bb97e04c30f22d28d8d2ffc985d67aa4621f083ac92fc5e85d8f3` |
| Candidate source inventory | `153f5b2cdb456b42ff73a62cb19a4c082178d88bda7d1eba8cb194f8eb8346d5` |
| C4 isolated controller | `1e33546e3fa9dda76b6ee4d397b619e899693af8258a240458ce99736359528b` |
| C4 production controller | `644a6c8f726fdef80c4e80bee881ce27a883bb0bd4c00613074a702895414097` |
| C2 short controller | `ecd3f679b524e684e1a05a0602079324bdd71da246fe327eb4cd1dca900c37c2` |
| C2 long controller | `4bfb713594f9632e92e577715b530110c1510ae2fd664908736afc89fa133c0d` |

Long prompts come from frozen members `spec-c4-u2` and `spec-c4-u3`;
their prompt JSON SHA-256 is
`24866481815846e35c827b316dc274b9d725ab837dff811e655fded5226fee9d`.
The frozen HTTP input receipt SHA-256 is
`d5a35e6341e53de0286cfd777e4fadd707c12cf9d18f95010a4a38ffac0ab59d`.

Raw results/controllers stay external under `~/scratch/qwen-head-wave/`
on Spark A and `/home/pmeenan/scratch/jitllm-m3-qwen-head-wave-2026-10-04/`
locally. The local copy excludes serving spill/state. All GPU model runs
use the installed supervisor with a 600-second limit and are waited on.
These screens qualify this sharing mechanism; the broader M3 concurrency,
long-context speed and same-reference quality gates remain open.
