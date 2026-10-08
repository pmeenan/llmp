<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# D-048 task lanes: wakeup, worker counts and queue sizes — 2026-09-24

M2's task-lane item asks for measured worker counts, queue sizes, and wakeup
and polling behavior before those settings are chosen
([async model](../../async-model.md)). This experiment measures llmpalooza's own
lane primitives on the workstation and on `spark`:

- `base::WakeFlag`, the owner's coalesced wake flag;
- `scheduler::Lane`, worker threads over one mutex-guarded
  `base::BoundedQueue`;
- `scheduler::CompletionBoard`, the per-operation mailboxes.

The harness is [`benchmarks/lanes_bench.cc`](../../../benchmarks/lanes_bench.cc).
Its commands are synthetic: no GPU, disk or network work.

## Method

The harness runs three measurements:

- **Wakeup.** A producer idles 100–400 µs, stamps the time, publishes and
  signals. The owner waits for the signal in one of four ways, then records
  the time from signal to running. 5,000 samples per mode. "Owner CPU" is
  the owner thread's CPU time divided by wall time.
- **Lane throughput.** One producer submits commands that spin for either
  0 or 2 µs, retrying with a yield whenever the lane refuses one because it
  is full. It covers 1, 2, 4 and 8 workers, with queue capacities 16 and
  256.
- **Board round trip.** The owner keeps up to *N* operations open. A
  provider lane accepts and completes each one at once, and the owner
  harvests and closes it. It measures operations per second and the time
  from opening a mailbox to closing it.

Hosts and builds:

- **Workstation:** an 11th Gen Intel Core i9-11900KF with 16 hardware
  threads, on Ubuntu 24.04.5 with kernel 7.0.0-31-generic. The `native`
  preset, built with the pinned SDK `x86_64-f469d317c88c3044` at `-O2`.
  cpuidle is `intel_idle` (POLL, C1, C2 and C3; C3 exit latency 1048 µs).
- **`spark` (`spark-c4e2`):** a GB10 (Cortex-X925 and Cortex-A725, 20
  threads) on DGX OS 7.6.0 with kernel 7.0.0-1019-nvidia. The `cross`
  preset from the same SDK, deployed with `tools/build deploy`. cpuidle is
  `acpi_idle` with the `menu` governor, LPI-0 to LPI-3, with exit latencies
  of 0, 42, 231 and 433 µs. The cpufreq governor is `performance`, and the
  load average was 0.08 before the runs.
- **Run conditions:** no system settings were changed. Clocks were not
  locked, threads were not pinned, and idle states were left as
  configured. Each host ran three consecutive processes; the tables give
  the range across those runs. Raw output stayed in session scratch.

Source identities (SHA-256) of the files these runs built from:

| File | SHA-256 |
| --- | --- |
| `benchmarks/lanes_bench.cc` | `239f5c5be19f7cf939f8c4bb57ae0a90796e388285517425de951f0a905df691` |
| `src/base/bounded_queue.h` | `2ea9614dde2021c2e98173eb3033e89974c91067bc8a9a8b794f75d385bf5f96` |
| `src/base/wake.h` | `a58ffa305f7f1ee7287fa66fffafcd7f75a537765ea9df6b0e8f52f4b64797c3` |
| `src/scheduler/completions.cc` | `8880bafb4036af70d44b16348ddaed2b2458ea5fc9583b20dac424bfc8ac155d` |
| `src/scheduler/completions.h` | `2f654d9d5de978af8f6148891a647b30440f219a23844bad30a85bfa1f6cfc58` |
| `src/scheduler/lane.h` | `30ca50af8a09555a0a907bb6ec517f5674c14cf22a3881bd966d815017aa0d77` |

These are the sources measured. A later review fix to `completions.cc`
changed only the path where a result races `Close` on an operation that
never started, which the benchmark never takes.

## Results

### Wakeup: signal to owner running

All times are in µs.

| Owner waits by | Workstation p50 | p99 | max | CPU | Spark p50 | p99 | max | CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sleeping on the wake flag | 37–72 | 90–93 | 173–202 | 5–6% | 207–283 | 451–485 | 667–822 | 4–5% |
| Polling, yielding | 0.1–0.2 | 0.2–0.6 | 3–12 | 100% | 0.6 | 0.9–1.0 | 9–260 | 100% |
| Polling every 50 µs | 54 | 102 | 130–134 | 1% | 50 | 101 | 164–352 | 2% |
| Polling for 200 µs, then sleeping | 2.5 | 73–74 | 93–148 | 62% | 3.1–5.0 | 269–480 | 577–641 | 53–59% |

### Lane throughput: thousands of commands per second, one producer

| Workers | Capacity | Work | Workstation | Spark | Spark refusals per command |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 0 | 3344–3627 | 1380–1506 | 0.35–0.49 |
| 1 | 256 | 0 | 3866–4153 | 2362–2872 | 0.00–0.07 |
| 2 | 256 | 0 | 4751–4975 | 1103–2082 | 0.00–0.01 |
| 4 | 256 | 0 | 1175–1465 | 181–404 | 0.00 |
| 8 | 256 | 0 | 312–325 | 159–218 | 0.33–0.74 |
| 1 | 256 | 2 µs | 447–451 | 306–366 | 6.29–7.51 |
| 2 | 256 | 2 µs | 886–901 | 611–637 | 1.40–1.79 |
| 4 | 256 | 2 µs | 1727–1740 | 938–1130 | 0.23–0.46 |
| 8 | 16 | 2 µs | 929–1534 | 141–172 | 0.02–0.72 |
| 8 | 256 | 2 µs | 2700–2740 | 140–166 | 0.02–0.06 |

Capacity 16 behaves like capacity 256 except with one worker and no work,
where the producer is refused more often; the harness prints every row.

### Completion board round trip, one provider worker

| Mailboxes | Workstation k ops/s | p50 µs | p99 µs | Spark k ops/s | p50 µs | p99 µs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1704–1749 | 3.8–3.9 | 5 | 331–493 | 12 | 15–37 |
| 64 | 2544–2626 | 20–21 | 28–29 | 874–940 | 53–56 | 97–133 |
| 512 | 2191–2231 | 145–148 | 225–232 | 695–871 | 436–518 | 1078–1358 |

Four provider workers were slower at every size. On the Spark they managed
174–465k ops/s, with a p50 between 23 µs (8 mailboxes) and 1140 µs (512).

## Conclusions

1. **A sleeping scheduler wakes slowly on the Spark.** From sleep, the
   owner takes 207–283 µs at p50 and up to 485 µs at p99 to run after a
   signal, against 37–72 µs on the workstation.
   - **Likely cause:** the `menu` governor picking LPI-2 or LPI-3 (231 and
     433 µs exit latency) during the idle gap. This is not isolated: no
     idle state was disabled to confirm it, since that is a system change.
     See [RE-017](../../rough-edges.md).
   - **Polling:** with yields it wakes in under 1 µs, at the cost of a
     whole core.
   - **Polling 200 µs, then sleeping:** this brings the Spark's p50 to
     3–5 µs. Its p99 still shows the deep-idle cost, because in this
     harness many gaps outlast the 200 µs window.
   - **Design consequence:** the scheduler thread should poll while it
     waits on operations whose completion is imminent and on the critical
     path, such as a decode step's kernels or a page-in the running phase
     needs, and sleep on the wake flag when idle.
   - **What is not settled:** the polling window and its bound are settings
     for M3 onward to tune against measured per-token latency. No default is
     chosen here.
2. **Keep each lane queue to four workers or fewer.** One mutex-guarded
   queue stops scaling beyond two workers for trivial commands, and beyond
   four for 2 µs commands. On the Spark, eight workers fell to 140–172k
   commands/s, from 938–1130k with four. More parallelism means more lanes
   (sharded queues), not more workers on one queue. Storage and device lanes
   carry a few large commands (2 MiB chunk reads, launches), so the
   queue's rate is not their limit. A lock-free ring waits for a lane whose
   measured command rate needs it.
3. **The completion board is not a bottleneck at M2's scale.** With one
   provider worker, a round trip costs microseconds, and the open-to-close
   time grows with the number of operations in flight: queueing, not
   per-operation overhead. More provider workers on one lane made it slower
   (contention). Provider lanes should start with one worker unless their
   provider's ordering contract and a measurement say otherwise.

## Limitations

- **Workload:** synthetic commands and a single producer, with no GPU, I/O
  or network work.
- **Host state:** no CPU pinning, no clock locking, and idle and frequency
  settings as found.
- **Variance:** earlier development runs of the same wake flag and lanes
  put the Spark's sleeping p50 anywhere from 88 to 370 µs, and the
  workstation's from 2.7 to 72 µs. The ranges above are three runs, not a
  distribution.
- **Scope:** the benchmark measures the primitives, not a scheduler loop.
  Real completion traffic, as in the M2 backend proof, may change the
  picture, and the conclusions above should be revisited with it.

## Reproduce

From the repository root, on the workstation:

```bash
tools/build build native
```

```bash
build/native/benchmarks/llmp_lanes_bench
```

On `spark`, through a cross build (the deploy directory name is the one
`tools/build deploy` prints):

```bash
tools/build deploy --host spark cross
```

```bash
ssh spark '~/.cache/llmp/deploy/cross-<id>/benchmarks/llmp_lanes_bench'
```
