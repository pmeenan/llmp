<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 own draft-vocabulary study

The first independently constructed candidate is complete and remains
experimental. It has only **22 IDs with nonzero calibration scores** from
32 observations; the other rows follow the frozen lower-ID fallback.
On all 32 held-out anchors, the prefix, externally curated and own heads
contain the full-head winner, propose the same first token and naturally
accept the same 48 of 96 draft positions. This bounded study found no
membership or acceptance gain from the own list. It does not justify
adoption or a representative corpus claim. The default prefix is unchanged;
the additional 25–30-minute free-generation comparison was stopped before
launch because this candidate supplied no gain to qualify.

Collection, construction and held-out ordering follow the frozen
[protocol](protocol.md). Both independent, self-authored Apache-2.0 input
manifests were prepared before outputs: four equally weighted domains at
8K/32K, one example and four common-history anchors per cell. The
target-frequency signal is the unchanged control's natural greedy choice;
the authored continuation supplies conditioning only. No held-out output,
answer label or external selected-map membership enters construction.
The portable string seed, larger training calibration and installation
adaptation remain separate later optimization passes.

## Completed calibration and held-out controls

The candidate uses the preregistered integer ranking
`3 * full-head proposal count + natural-target count`, reserves metadata
stops 248,044 and 248,046, breaks ties/fills zero scores by lower original
ID, and contains exactly 47,172 ascending IDs. Its SHA-256 is
`6c0b7d9e9f09d5acd82ebe26c484eb70ebf79694d3b57c1c6158094c672702a9`.
It overlaps 47,169 prefix IDs and 29,910 externally curated IDs; neither
overlap was an input to ranking. The generic importer authenticated all
selected original BF16 rows and the ID map, with unchanged MTP resources
and routed-expert payloads. The imported experimental artifact is
`39e67397a3ebad6249cf80205932d4cfcb3d9e52fd2dca5437141b00aad3a3a0`.

Calibration first collected 32 anchors for each incumbent head, then
projected every first-pass F32 input through the unchanged full BF16 head.
Both complete physical subsets reproduced the captured logits byte for
byte, and full rows/own repeats agreed. Both incumbents contained the
full winner at 32/32 anchors and agreed on all first proposals. Each
matched the independent natural target at 18/32 anchors and accepted
36/96 positions. These calibration observations constructed the candidate;
they are not held-out results or an own-head inference measurement.

Held-out collection ran all 32 frozen anchors for each of the three
heads before full-head projection. The initial operand, complete target
and MTP state, pending/catch-up cursor, anchor position, history and full
natural-target row agree across heads. Every head executes four arms
(capture off/on/on/off) per anchor. Each completed arm settles its natural
draft/verify/commit before exact restoration; one unchanged control then
advances the common history. Each head records 128 canonical SHAs,
96 reference captures and 320 complete byte comparisons. Missing, extra,
short or reordered used ranges, cursor differences and file truncation
refuse; successful node completion precedes all host reads and retirement.

| Held-out result | Prefix 65,536 | Curated 47,172 | Own 47,172 |
| --- | ---: | ---: | ---: |
| First-pass physical subset exactly reproduces capture | 32/32 | 32/32 | 32/32 |
| Full-head greedy winner included | 32/32 | 32/32 | 32/32 |
| Proposal's full-head rank / maximum logit regret | 1 / 0 | 1 / 0 | 1 / 0 |
| Natural target ID included | 32/32 | 30/32 | 32/32 |
| First proposal equals independent natural target | 22/32 | 22/32 | 22/32 |
| Naturally accepted three-position draft prefixes | 48/96 | 48/96 | 48/96 |

Each head accepts 23, 15 and 10 positions at depths one, two and three.
The independent one-row target matches above differ from the batched
verify's first-position acceptance count; neither is silently substituted
for the other. The existing agreement bound is unchanged. Accepted
positions per cell are identical for all three heads:

| Domain | 8K: accepted / 12 | 32K: accepted / 12 |
| --- | ---: | ---: |
| Code | 9 | 6 |
| Prose | 6 | 3 |
| Instruction | 7 | 4 |
| Arithmetic | 4 | 9 |

All first proposals equal the full-vocabulary winner. Thus first-position
shortlist exclusion explains none of the ten disagreements with the
independent natural target in this sample. The two curated-map target
exclusions occur on arithmetic anchors but do not change the winning MTP
proposal or accepted prefix. Later draft inputs depend on earlier
proposals; these depth counts do not establish a causal exclusion rate
for every later position. No completed-answer accuracy, free-running
throughput, sampled-distribution gate or default policy was measured here.

## Diagnostic cost and source qualification

The original one-8K-anchor pilot measured 831,102,976 used-state bytes
in 48.407527 seconds. Fourteen complete SHA traversals took 33.702485
seconds; its conservative 64-anchor/full-capacity state-control estimate
was 71.2 minutes before model work. The revised collector retains four
independent canonical SHAs and replaces ten redundant hashes with complete
byte comparisons against three bounded direct-file references. It keeps
every off/on/on/off, draft, commit, restore, range and cursor proof.
The revised pilot exactly reproduced the original four canonical hashes,
full captures and trajectory. Its native command took 27.111228 seconds:
four digests 9.770979 seconds, reference capture 7.712738 seconds,
comparison 2.264677 seconds, checkpoint capture 0.192545 seconds and
four restores 0.702474 seconds. Direct I/O, staging, completed GPU copies,
SHA and comparison time remain charged diagnostic costs, not decode rates.

Calibration collection completed in 23.43 minutes, with native commands
723.482/682.382 seconds for prefix/curated. Held-out collection completed
in 35.15 minutes, with native commands 711.684/692.698/696.451 seconds
for prefix/curated/own. Every held-out head hashed 156,247,785,472 state
bytes, captured 117,196,324,864 reference bytes and compared
390,640,435,200 bytes. These are cumulative traversals, not simultaneous
host replicas or physical state occupancy. The held-out projection used
six serial 24+8 operand invocations and finished in 122 seconds; the
read-only aggregate analysis then completed successfully.

Final checked source passed 1,204 locked Spark tests (256 GPU), native
prepare, seven SDK format checks, five actual compile-database tidies,
portability boundaries, REUSE and 1,111 headers. Meaningful controls
cover full-file byte mismatch, geometry reorder, short/extra files,
checked ranges and frozen manifest refusal. The private exact-ID
`vocab-greedy` route was compiled/checked but not run; it retains the
existing fixed-depth-three, 256-output, two-repeat and margin-1.0 rules.
Its fixed budget continues through stop IDs and would be verifier evidence,
not completed-answer accuracy.

Final qualification `f5599326…`, full 1,231-file source map `8adf7ce1…`,
native executable `a0b3722d…`, compile database `cd99bd63…` and native
build receipt `7b89a7aa…` are authenticated by every held-out phase.
The source snapshot includes baseline `5d0eaa6` plus this private unit;
unrelated later output-B work was not part of these measurements.
All loads passed the strong 105 GiB free-node/container/GPU/native-model
gates; all six projection children returned zero and were reaped.
Excluded setup attempts remain preserved: a check refused undeclared
new baseline source files before base reconciliation, and import r1
misread the generic importer's retained `.staging` directory. The latter
failed before deep qualification/model load; fresh r2 passed the corrected
guard and complete byte proofs. No failed attempt supplies an adoption gate.

Raw roots on `spark-b` remain under `~/scratch/m3-qwen-own-vocab/`:
`study-reference-r1/{prepare,collect,projection-r1,candidate-r1}` and
`heldout-r2/{import,anchors,projection}` plus `analysis-r1.json`.
Compact authenticated receipts: calibration `b7a6ac86…`, projection
`db277a06…`, candidate `16de7609…`, import `1fe69a0d…`, held-out anchors
`8a370d66…`, held-out projection `b970adaf…`, analysis `8413c6aa…`.
The no-model preservation job archived all 1,231 checked source files,
both executables, actual cuBLAS libraries, build inputs/controllers and
compact receipts under `final-preservation-r1/`, without duplicating raw
payloads. Its complete 1,273-file receipt is `e6c14700…`; busy/probe gates
confirmed retirement. No workstation execution occurred.

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
context-dependent selector, or qualify a new candidate. The independent
study above subsequently retained a common continuation and complete
restored target/MTP state while measuring natural proposals and target
verification, as registered.
The later independent calibration/held-out result above preserves that
default and separates shortlist exclusion from the full drafter's error.

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
