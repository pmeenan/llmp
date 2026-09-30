<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 own draft-vocabulary study

The fixed candidate and held-out comparison are pending. Their objective,
collection balance, same-history controls and acceptance rules are frozen
in the [protocol](protocol.md). The portable string seed and installation
adaptation are separate later optimization passes.

## Full-head control on existing native inputs

On 2026-09-30, Spark `spark-56f5` (GB10, driver 580.178.04, SDK
`aarch64-e0a0c85c42806fb1`, CUDA 13.4.92) projected twelve previously
captured F32 MTP vectors through the original full BF16 target head.
The captures are four native draft steps at positions 128,799, 128,800,
128,802 and 128,805, with three draft passes per step. Both reduced-head
captures have identical complete histories and all twelve vectors, not
only the first anchor. They are an existing diagnostic prompt, not
calibration or held-out examples.

The target's unchanged `output.weight` has 248,320 rows of width 2,560:
1,271,398,400 bytes. The standalone native vector operation used F32
inputs directly, without BF16 input conversion, confidence calculation,
or a library-product substitute. It loaded the head only, not the model.
Every retained original-ID row had to reproduce the corresponding native
captured reduced-head logit byte for byte before interpreting coverage.

| Control | Prefix 65,536 | Selected 47,172 |
| --- | ---: | ---: |
| Captured retained logits reproduced exactly | 12 / 12 | 12 / 12 |
| Full-vocabulary greedy winner present in shortlist | 12 / 12 | 12 / 12 |
| Shortlist winner's full-vocabulary rank | 1 at every input | 1 at every input |
| Full row equal across duplicated capture arms | 12 / 12 | 12 / 12 |

The full winners are 1,144; 4,087; 1,156; 4,087; 1,156; 579; 579;
1,622; 13; 14,235; 22,903; and 303. Eager repeats, graph repeats,
eager-versus-graph full bytes and CPU/device lower-ID argmax controls
also pass. Each input retained three finite positive event samples,
each covering 64 graph replays. Their per-input median full-head cost
ranges from 4.885 to 5.101 ms across the two serial duplicate arms.
Both arms execute the same full head; this timing is not a comparison of
the reduced heads or an end-to-end speed measurement.

For this small same-state case, missing full-head winners do not explain
the difference between shortlists. It does not identify the cause of the
free-running context ladder, establish corpus coverage, justify a
context-dependent selector, or qualify a new candidate. The next study
must retain a common continuation and complete restored target/MTP state
while measuring natural proposals and target verification, as registered.
The default prefix is unchanged.

## Provenance

Target manifest/artifact
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`;
index `5b65dcce8169374e638264d7ebdcb5cca517234b3dec26f1272a5f4f9c0e4c89`;
extracted full head
`40bddd25d0d94a128ab08280faad39cfc3ee3064252761269115f544722607c9`.
Every original indexed chunk was authenticated before extraction.
The concatenated twelve-vector SHA-256 is
`a1f76278fef27af4315f3ca72c8ff458bdd4b89e5955a3f9ab798820b6763884`.
The receipt retains all four full-history hashes, original-ID maps,
captured logits and native capture-spec identities.

Supervised job `qwen-full-head-r2`, 18:16:22–18:17:00 America/New_York,
native/controller exit 0; both serial arms passed the strong 105 GiB
free-node probes before and after execution. The first launcher attempt
was refused before weight loading because the copied library path was
escaped; it supplies no operator result.

Formatted source
`31a5e22130f727464d791402c404ed01e1624fc616f09a1c85825100c84ea07f`;
executable `b52f7a04f9d7b236cc96ffd98e49e1988afab5fd6a0c84419aef798b183dfa9f`;
completed receipt
`405a569872844ef26a98e1c1cadc8c2a6f5e901932f9d858b10c575f3d37227b`.
The build reused the actual native compile/link inventory, recorded every
linked archive, passed SDK formatting/tidy, and verified the resolved
cuBLAS library paths and hashes. It does not claim a new full-tree build.
Sources, extraction/controller/build receipts, 72 raw event samples and
full output rows remain outside Git on `spark-b` under
`~/scratch/m3-qwen-full-head-r2/`; supplied helpers remain in
`~/scratch/m3-final-launch/`, and qualified reduced-head captures in
`~/scratch/m3-qwen-draft-input/`. No workstation execution occurred.
