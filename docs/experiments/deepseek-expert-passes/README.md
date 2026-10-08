<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek joined-wave expert passes — 2026-10-04

**Neither candidate wins; retain the current loop.** Two private model screens
share IQ2_XXS gate/up weight decoding across four or two selected tokens per
pass, instead of one. The R2/W4 warp-row traversal and reduction stay the
same. Each token keeps GGML's sign expansion, odd integer scale, signed
integer division and floating-point order. The P1 body stays unchanged.

The selector applies only to routed IQ2_XXS with gate weights, K4096/N2048
and more than eight joined rows. Scalar chunks and verifies stay on their
current paths. The private trace records actual selection at 16 rows in both
candidates. Only gate/up changes; routed down, drafts, attention, state and
lane scheduling stay as before. Two tokens per pass tests whether the larger
four-token loop's live register set hides a useful sharing gain.

Spark B (`spark-56f5`), driver 580.178.04, CUDA 13.4.92, SDK
`aarch64-e0a0c85c42806fb1`. Each fresh process runs the existing native
`--check wave`: first four short `fast-swap/prompts.json` prompts, context
16,384, chunk 4,096, DSpark depth three, four slots, output-A/HCA, joined
drafts, lanes and graphs on. This is an in-process forced speculative-wave
control, not an HTTP rate or an adaptive serving policy comparison.

| Screen | Original before: C4 median ms | Candidate: C4 median ms | Original after: C4 median ms | Candidate latency versus bookend mean |
| --- | ---: | ---: | ---: | ---: |
| Four tokens per pass | 230.122 | 231.996 | 231.508 | +0.512% |
| Two tokens per pass | 230.644 | 233.391 | 229.695 | +1.400% |

Each arm measures 20 width-four waves, seven width-three and five width-two.
Bookend C4 median movement is +0.602% / -0.411%. Neither single screen
establishes a confidence interval or a gain. Whole-cohort times include
first-use planning/capture and vary more than these medians; they are not
used to select a candidate. No operator-only timing is added to a model gain.

All six arms finish with no problems and zero exact mismatches. Every
compared draft vector and complete target verify block equals its scalar
control, including the discarded verify's retry. Discard leaves zero stale
state bytes outside the allowed draft-ring writes; surviving slots' final
state fingerprints equal their solo controls, and the departed slot's
fingerprint stays unchanged. These are the existing benchmark's checks:
its `rows_compared`/`rows_identical` fields count plain rows and are zero
in this speculative scope. They do not count the speculative comparisons.
Each arm captures eight graphs and replays 154, records 380 solo generated
tokens (the initial anchors excluded), and 336 wave tokens with the final
slot intentionally leaving early. No cross-engine quality or swap gate follows.

All three build jobs and both screen jobs finish successfully under the
installed GPU supervisor and are waited on. Every model child returns zero
and is reaped (34.91–36.16 s); strong admission/retirement probes pass around
all six arms. Final retirement finds 116.816 GiB available, no GPU/container/
native-model work and no busy or waiting GPU job. The prototypes remain
outside main. No HTTP ladder or production suite ran for these rejected
experiments; the private P2 variant test-array update and trace removal
would have been required before adoption.

Reproduction: apply `p4.patch` or `p2.patch` against `16d55a5` in an isolated
warm tree, build `llmp_spec_runner`, then use the corresponding retained
controller with the installed GPU supervisor. Workstation patches, controllers
and raw receipts live in
`/home/pmeenan/scratch/llmp-m3-dsv4-expert-passes-2026-10-04/`.
Spark B retains both source/binary snapshots and results in
`~/scratch/dsv4-expert-passes/`; its private build is `~/src/llmp-wt/dsxp0001/`.
The target artifact is
`cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac`,
DSpark `dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5`.
The compiled GGML `vecdotq.cuh` is byte-identical to the local source used
for arithmetic review (SHA-256
`a7fd4281ad9f0a6d9373c776325411d2522e0b62ffae2c16e91e667a1af251fe`).

| Provenance | SHA-256 |
| --- | --- |
| Original benchmark | 539b401e4459a0948e03ead71fb4b196da80bbde6ecbbb64216dcac317659b98 |
| P4 benchmark | f2d2325fc0adb7df7c127de6162e531966b8fecc4b1cc17857f0ac1cd09ebc4d |
| P2 benchmark | 40be5a5073c85e4db34b51bec05e7ad066aecf6d2257e8c8ae2b71d9498b7680 |
| P4 source patch | 21df1c3170c65ac4802a6d189469308643d4af100a8cd4b1d73d09574ebbbce8 |
| P2 source patch | 22c239d0ef13ca0704ace05737ec9c4e96bac8704a8838cecd73f210b2c02790 |
| P4 controller | 96f641aa03a3ba5ee3cc8b1a3dd4abc900aa1ceb4c8d7666ffb03ed8c0dce0fa |
| P2 controller | 9045e283233cd35f79db13f4b0343f741ddadf963f96862ba7e1dd3508d26349 |
| P4 receipt | 0e095401ee5ecd6edc32e42619a3d673b3159226bc3a2edd7e1ecb57c22b5581 |
| P2 receipt | 2c73ff4a698196c8085fcb62787dfe5cc470151f724172623e7e4ada06e29acf |
