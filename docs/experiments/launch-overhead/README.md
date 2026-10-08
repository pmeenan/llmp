<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Host time per launch through the K-C launch context — 2026-09-27

Backend-proof P1 asks for the host cost per launch
([BP-F4](../../backend-proof.md#case-matrix)),
which is reported and not gated. This experiment measures that cost for the
one operation a plan can now select between two implementations
([registry](../../../src/execution/registry.h)), RMSNorm scaled by a weight:

- `ggml.rms_norm_mul.fused`: GGML's fused launcher, one kernel;
- `ggml.rms_norm_mul.unfused`: rms_norm's launcher, then mul's, two kernels.

It covers per-launch host time only. BP-F4's per-token figures and its
comparison with both upstreams' decode need a whole native decode step, which
P2 builds.

The harness is [`benchmarks/launch_bench.cc`](../../../benchmarks/launch_bench.cc).

## Method

The operation runs on decode's shape: one row of Qwen2.5-0.5B's width, 896
F32 values, with its weight, intermediate and output in device VMM. Each
implementation is timed through four layers:

| Layer | What each operation goes through |
| --- | --- |
| direct | GGML's launchers alone, called in a loop inside one launch-context run: upstream's own host cost |
| run | One launch-context run per operation, with the launchers and no operand checks: what the K-C context adds (the provider's `Submission`, the error reads, the pool limit) |
| checked | The `ops.h` entry point: the operand checks, then a run |
| plan | The kernel a resolved plan selects, `RmsNormMulKernel::Run`: the checks and a run behind one indirect call |

A batch submits 64 operations through one arm, timing only the submission
with `steady_clock`. The stream then drains, untimed, so the device queue
never fills. Arms alternate batch by batch. There are 20 warm-up batches,
then 500 measured batches per arm. The tables give per-operation host time
across the measured batches.

- **Host:** `spark` (`spark-c4e2`), a GB10 (Cortex-X925 and Cortex-A725, 20
  threads) on DGX OS 7.6.0 with kernel 7.0.0-1019-nvidia and driver
  580.178.04 (CUDA driver API 13000). The cpufreq governor is `performance`.
- **Build:** the `cross` preset from the pinned SDK `x86_64-e0a0c85c42806fb1`
  (NVCC 13.4.92), deployed by `mise run deploy -- --host spark`, from
  commit `7a4b7688` with the P1 plan-selection changes uncommitted. GGML is
  the locked tree `8a4d1968…`; the module digest the identities carry was
  `65a650ca…`.
- **Conditions:** no other compute process was on the GPU before or after
  the runs, the load average was 0.17 before them and 0.24 after, and no
  process on the host used more than 4% of a CPU (sshd). Clocks were not
  locked, threads were not pinned, and no system settings were changed.
  Three consecutive processes ran; the table gives the range across them.
  Two earlier sets of three, before the identity gained the module digest
  (a change outside the timed path, which binds once before timing), gave
  medians within 0.05 µs of these. Raw output stayed in session scratch.

Source identities (SHA-256) of the files these runs built from. For
`implementations.cc` the hash is of the file after a later formatting fix,
which moved one line-continuation backslash in a preprocessor check; the
measured build had `23ecc224…`. The module digest the build records
therefore differs from `65a650ca…`; nothing timed changed.

| File | SHA-256 |
| --- | --- |
| `benchmarks/launch_bench.cc` | `df9884124c498b9610655a8306cd6aed81561582a80a67977203814a2c14541a` |
| `src/execution/registry.cc` | `fe71401d84fd324f61b49fb30e9b1d251af541fbb938fdd10680e827ad473f2b` |
| `src/kernels/ggml/implementations.cc` | `c983810e9abfd398fdf0a3cc335d74e5ddfb3b4e788c2f699febd46408a69409` |
| `src/kernels/ggml/launch.cu` | `4e88df20603cce53705f63da29c73d45163303aa032304d838c527ad6a62b225` |
| `src/kernels/ggml/ops.cu` | `8d5596a412ca4d762266eb5e74ae0377475ea49daabea1d683e0f409ea091d51` |
| `src/kernels/ggml/validate.cc` | `9a9749284e40ef0d2978f7cabafdec8a2e47f89450aadc98f156f3f22acfbbf4` |

## Results

Host time per operation in µs; each cell is the range of that statistic over
the three processes.

| Implementation | Kernels | Layer | p10 | p50 | p90 |
| --- | ---: | --- | ---: | ---: | ---: |
| fused | 1 | direct | 1.75 | 1.75–1.77 | 1.77–1.78 |
| fused | 1 | run | 1.81–1.82 | 1.82–1.83 | 1.83–1.85 |
| fused | 1 | checked | 1.86–1.87 | 1.87–1.88 | 1.89–1.90 |
| fused | 1 | plan | 1.86–1.87 | 1.87–1.88 | 1.89–1.90 |
| unfused | 2 | direct | 3.55–3.61 | 3.58–3.63 | 3.60–3.65 |
| unfused | 2 | run | 3.63–3.73 | 3.68–3.77 | 3.73–3.82 |
| unfused | 2 | checked | 3.89–3.97 | 3.92–3.99 | 3.93–4.00 |
| unfused | 2 | plan | 3.89–3.96 | 3.91–3.99 | 3.93–4.00 |

## Conclusions

- GGML's launchers cost about 1.8 µs of host time per kernel at this shape:
  upstream's own host code and the kernel launch.
- The launch context adds about 0.06 µs per run for the fused
  implementation and 0.10–0.14 µs for the unfused one (p50, run against
  direct). The operand checks add about 0.05 µs for the fused
  implementation and 0.22–0.24 µs for the unfused one, which checks the
  norm and the mul separately, then their overlap. A bound plan adds
  nothing measurable over calling the implementation directly: selection
  happens once, when the plan is bound.
- Fusion halves host time as well as launches: 1.9 µs against 4.0 µs per
  operation through a plan.
- These are per-launch numbers only. Whether decode needs graph capture
  (BP-F4's rule) depends on the per-token host time of a whole native
  decode step against upstream's measured time per token, which P2 can
  measure.
