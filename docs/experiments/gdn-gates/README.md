<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GDN gates shape-generalization screen

Prospective U7a first screen, 2026-10-22. This screens the changed shared
pointwise kernel at the existing 48-head, one-row decode workload. It is
not a model throughput gate, a fusion-rank update, or evidence for the
remaining U7 kernels. Synthetic non-reference shapes are correctness tests.

Result: [the registered first screen](RESULTS.md) passed; the registration below
retains its prospective wording and fixed criteria.

## Arms and execution

A is the production implementation at `48bbb5ede4a3333c80c4dd37df877e358baaa997`.
B is the candidate shape-generalization diff from that base. Build the identical
`llmp_gdn_gates_bench` collector against each arm, with the pinned `spark-native`
toolchain and the same source/dependency locks. Retain source file manifests,
binary hashes, build receipts and the collector's actual selection digest for
each arm. Copying an executable is permitted; copying a build directory is not.
The Spark copy's Git version string is not source evidence.
The baseline checkout also contains the unrelated Gemma documentation and
collectors through `eadf8da897a6968ddeed4696408d7759d56a5ab1`, plus this
registration's three files. Its GDN implementation is unchanged from A above;
record the complete actual source manifest for each built arm.

Run on idle Spark B under the installed `spark-job --gpu` exclusive supervisor,
with the regular 600-second timeout and 30-second retirement grace. Run A, B, A
bookends three times, in one exclusive job, with no build between the runs.
No clocks are changed. Record device, driver, toolkit and ordinary memory facts.
Each process initializes the same deterministic finite F32 inputs, verifies
both output planes against host math with an absolute error bound of `2e-5`,
then captures 128 calls to the real checked `RunGdnGates`. It verifies the
capture has 128 nodes, poisons the output, replays and requires every output bit
to equal the direct launch. Every timed sample replays that graph 128 times,
waits for CUDA events and the provider completion fence, and verifies its output.
There are four untallied warmup samples and seven tallied samples per process.
Raw samples and complete receipts remain outside Git; aggregates and provenance
are recorded here. Failure, missing completion, changed fixtures or unequal
old/new output hashes invalidate the comparison.

## Decision fixed before timing

For each A/B/A triplet, compare B's median ns/kernel with the mean of its two A
medians. Report all three ratios and their median. The coarse D-085 first screen
passes if the median ratio is at most 1.10. If either A bookend differs from the
other by over 10% relative to their mean, the screen is inconclusive; investigate
interference or unstable timing before another prospective run, rather than
selectively dropping samples. The threshold follows D-085's existing 10%
criterion, not a claimed baseline measurement. No measurements have been run
when this protocol is registered.

The timed quantity is captured device execution per kernel, including graph
replay scheduling amortized over 128 nodes. Host construction and checking run
before timing. The fixture
allocates 1,152 device bytes and no kernel scratch; timing/event, graph, runtime
and module allocations are separate. Existing bitwise GGML-chain tests cover
rounding boundaries, FTZ, saved views, tail heads and rows beyond sixteen. The
slice also requires its focused Qwen3.8/component correctness tests and the
shared suite. Historical W5 rank records remain stale after this module changes.
