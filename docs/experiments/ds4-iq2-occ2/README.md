<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Scoped IQ2 compact-pair occupancy two

One real-input paired-product screen and one native 8K screen select J64
with a compiler minimum of two CTAs per SM. The numerical inner product,
raw IQ2 weights, native maps and Q8 preparation remain unchanged.

| Captured pair arm | Median paid CUDA time, ms |
| --- | ---: |
| Original J128 | 20.345505 |
| J64, occupancy two | 15.928576 |
| Original J128 | 20.448832 |

Resident paired-product rate improves **28.05393%**, with **0.507861%**
bookend movement. Each arm pays two tile-list builders and two products,
with one warmup and nine paid samples. All six complete gate/up outputs
match the original native byte goldens, all own repeats match, and borrowed
input/initialized guard checks pass. Fixture: K4096/M2048/T4096, 256
experts, six routes per token; 249 experts active, 358/533 live J128/J64
tiles. D4/maps/raw banks, both F32 outputs and 512 MiB workspace are funded
equally (2,227,490,308 CUDA allocation bytes). Producers, preparation,
allocation, transfer and readback are outside this resident event timing.

| Observed kernel resource | J128 | J64 candidate |
| --- | ---: | ---: |
| PTX minimum CTAs per SM | 1 | 2 |
| Threads/block | 256 | 256 |
| Registers/thread | 254 | 128 |
| Runtime local bytes | 0 | 16 |
| Dynamic shared bytes | 57,856 | 48,384 |
| ptxas stack / spill stores / spill loads, bytes | 0 / 0 / 0 | 16 / 16 / 20 |
| Worklist bytes | 3,584 | 5,120 |

These are compiler/resource observations; actual scheduling occupancy was
not measured. The private screen changes only the J64 IQ2 configuration
row's occupancy value, with O3/fast-math/SASS121a unchanged. This global
experimental override is replaced by a separate production specialization.

| Native 8K arm | Paid whole-prefill wall time, s |
| --- | ---: |
| Original | 9.798561853 |
| Eligible J64 occupancy-two pairs | 9.396063131 |
| Original | 9.772578109 |

Whole-prefill rate improves **4.145426%**, with **−0.265179%** bookend
movement. One process uses two 4K chunks, fresh state per arm, and the same
Q-head/output-A/HCA native bound executor. Exactly 0/86/0 pairs select the
candidate. All six complete 129,280-F32 frontier heads match the existing
combined native goldens byte for byte. Original maps/scatter quantization,
output strides and single-down products remain paid. Both arms fund the
additional 1,536-byte pair worklist edge; the shared pool maximum remains
169,869,312 bytes. Common activation/state/staging allocations remain
1,828,716,544 / 264,126,464 / 207,618,048 bytes. These allocations are not
a measured process peak-memory ratio.

The captured gain and whole-prefill gain are separate measurements, not
additive. The model screen establishes this native 8K factor, without a
maximum-context/task-quality claim or a new quality/performance mode.

The production source variant leaves all generic IQ2 configuration rows
unchanged. It adds a default-false numerical template parameter and a unique
IQ2_XXS/J64/nonfallback/compact specialization, instantiated only by the
existing O3 IQ2 defining unit. Native paired compact dispatch requires
GB10 and the measured shapes: weights `[4096,2048,256,1]`, packed broadcast
F32 input `[4096,1,4096,1]`, I32 IDs `[6,4096,1,1]` at their original validated
token stride, and both packed
F32 outputs. Original stride/alias validation and J128 dummy-column guards
stay mandatory. The shared predicate drives both workspace funding and
execution. Single/down products, ordinary pairs, other shapes/formats and
other devices retain their existing launches. No private selector or extra
runtime option is carried forward.

Selected source validation passes 1,258 locked Spark tests, including 259
GPU tests. After host/style repairs, ten affected controls, including three
paired/compact GPU controls, also pass; the full suite is retained rather
than repeated. All seven changed units pass clang-tidy, touched units and
headers pass clang-format, and the boundary check reports 356 files with
no problems. Workstation builds remain deferred to the final optimization gate.

The final locked O3 IQ2 defining object contains the unique bridge and both
ordinary specializations. Compiled-object resource inspection reports
128 registers/16 stack bytes for the new J64 specialization, versus
244/0 for ordinary J64 and 254/0 for ordinary J128. These object fields
are distinct from the earlier runtime local/dynamic-shared observations;
they do not measure hardware occupancy. The same final build then runs
the genuine engine's four all/frontier/frontier/all controls and three
ordinary continuation rows. All eight complete 129,280-F32 chunk heads
match their prior goldens exactly, with exact initialized state, own
repeats and common continuation. It uses the normal runner header and
archives, with no private audit getter, runner or sizing code.

Final qualification receipt SHA256 is
`1bd4c2dbcde52f2c8222901a3c30063152fa310ccd49af02314a4e23c9cd3d9f`;
the defining object SHA256 is
`07914f31b138856c4f40e7f6e18d391f9d7fa4fd2ca7c3ea8ba5151392bd49cb`.
The final engine binary SHA256 is
`da7a85f6b3c02838f61c7c4ea17ba904c5e93e190c9505c132f80761f79fedc6`.
Raw local validation roots are
`/home/pmeenan/scratch/m3-ds4-qhead-short-records/integration-qualify-r5/`,
`iq2-resources-r1/` and `runner-selected-r1/`. All selected commands retire
successfully; the final model takes 53.689 seconds and leaves a fresh
116.926-GiB clear memory/GPU/container/native-process gate. Full source-only REUSE/header validation passes: 1,168 headers across
1,290 source paths, REUSE rc0 and no problems. Its receipt is in sibling
`integration-headers-r1/` (Git blob `c37a42d7ae5a34ad3d54b73edba920ab58a863e5`);
the terminal clear gate is 117.184 GiB.

Measured on `spark-c4e2`, 2026-10-02, using SDK
`aarch64-e0a0c85c42806fb1` and its original NVCC/native compiler flags.
The captured supervised job completes in 50.727 s (CUDA compile 16.361 s);
the native job completes in 54.950 s (wrapper/host compiles 5.161/5.789 s,
model child 41.072 s). Both finish rc0 with all commands reaped, followed
by fresh clear native/GPU/container gates. Actual command/source/library
closure, PTX, raw inputs and full outputs remain outside Git under
`~/scratch/m3-ds4-iq2-occ2-short-r1/model-r2/` and
`~/scratch/m3-ds4-iq2-occ2-native-r1/model-r1/` on that Spark. The native
screen compiles against the coherent runner-control-r3 graph header and
matching rebuilt archives; it does not mix old and new source ABIs.
