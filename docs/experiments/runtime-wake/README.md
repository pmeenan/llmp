<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The runtime wake: from a step's fence to the next step's launch — 2026-09-28

The owner, on 2026-09-28: "We should benchmark with the actual runtime
wake mechanism in place so we have representative measurements." Until
then the paged harness polled for 100 ms (`NodeSettings::poll_window`),
so the scheduler thread and the device submission lane spun through
every step, the completion lane spun on its fence and the harness's
driver on the step's result: four cores, and a round trip of 0.01 ms a
step. The runtime's own defaults (200 µs windows) slept between steps
and paid 0.26–0.65 ms a step in wakeups (RE-017). This measures the
candidate ways to wait on the GB10, picks one, makes it the scheduler's
and lanes' default (D-094), and re-measures the models with it.

## The path

A decode step crosses four threads, each of which may be asleep:

1. the **device completion lane** (C) sees the step's fence complete
   (a query) and publishes it on the completion board;
2. the **scheduler thread** (K) harvests it and reports the step to the
   request's client;
3. the **client** (D; the harness's driver, which samples the next token
   on the host) does its host work and hands K the next step;
4. K posts it to the **device submission lane** (S), which launches it.

A sleeping thread on the Spark takes 0.1–0.8 ms to run once woken
(RE-017), and each hop can pay it.

## Method

`benchmarks/wake_bench.cc` (`llmp_wake_bench`), two modes:

- **chain:** the four threads over raw CUDA, K and S waiting on llmpalooza's
  `WakeFlag` as the scheduler and lanes do, the step a one-thread kernel
  that spins on the GPU's global timer and stamps its start and end. Round
  trip = the GPU's idle gap between a step's end and the next step's
  start, less the client's host work. Hops are host times; the GPU's
  timer is related to the host clock by a calibration (the smallest
  host-minus-GPU difference over 30 ms of a kernel publishing its timer,
  per configuration: the two drift apart by ~2 µs a second, so a hop can
  be off by some microseconds, and "detect" can read a few microseconds
  negative). CPU = the process's CPU time over wall, in cores, while
  stepping and while idle (the second after the last step, from 0.3 s
  on). Configurations: how C learns of the fence, and how K, S and D
  wait (below).
- **node:** the runtime's own path on `tests/support`'s paged node: a
  request holding its lease (D-093), 100 steps of the same kernel as
  device jobs under it, the harness's driver waiting for each; round trip
  = a step's wall less its device span (CUDA events around the job, as
  `StepTimes`).

Scenarios (chain): 45 ms steps with 0.2 ms of client host work (DeepSeek's
decode), 40 ms with 1.3 ms (Qwen3.8's), 5 ms with 0.2 ms (a short step),
and 45 ms mixed pseudo-randomly with 5 ms (a third of steps), so a
prediction is often wrong. 100 steps (200 for 5 ms), three repeats each.

Host: `spark-b` (GB10, Cortex-X925 and A725, 20 cores, kernel
7.0.0-1019-nvidia, driver 580.178.04, cpufreq `performance`, cpuidle
`menu`, LPI-0–3), the `spark-native` build of this slice's tree
(SDK `aarch64-e0a0c85c42806fb1`). No system settings changed. Each run
waited until no other GPU process ran and the 1-minute load was under 6
(twice, 20–40 s apart); other agents used the host between runs (load
0.6–2.5 at the runs' starts). Raw outputs: `~/scratch/m3wake/before`
(09:45–10:04, the tree before the runtime changed) and `after`
(10:09–10:26) on `spark-b`.

## The mechanisms (chain, before the change)

How C detected the fence: (a) `cuEventSynchronize` on a
`CU_EVENT_BLOCKING_SYNC` event; (b) a host function after the step
(`cuLaunchHostFunc`) bumping a futex word C sleeps on; (c) a stream
memory operation (`cuStreamWriteValue32`) writing a mapped host flag,
polled with yield, with a 50 µs timed futex wait (a GPU write wakes no
futex, so this is a timed poll), or with WFE; the query, yielding between
(the runtime until now); (d) an adaptive spin: sleep until 1 ms before
the expected end (in these runs the shortest of the stream's last eight
lengths; after the change each of them, below), querying at least every
1 ms, then spin to 1 ms past it, then back off; (e) (d) plus the relay:
as C starts to spin it anticipates K and S (`WakeFlag::Anticipate`), which
poll until then, and K, after a step, polls for about as long as the
client took lately and has S do the same; optionally with (b)'s host
function as the backstop in C's sleep. Whatever woke C, a query proves
the fence (D-048). Except in the first two rows, D waits by the same
adaptive spin as C (asleep through most of the step, woken early by K);
in the harness rows it spins. K and S otherwise poll for their 200 µs
window after their last progress and then sleep.

45 ms steps, 0.2 ms host work (µs; ranges over three runs of 99 hops;
p99 includes the run's first step, which has no history):

| Configuration | Round trip p50 | p99 | C detects, p50 | Cores stepping | Cores idle |
| --- | ---: | ---: | ---: | ---: | ---: |
| Harness-polled (100 ms windows; C, D spin) | 8 | 9 | ~0 | 3.99 | 0 |
| Runtime until now (200 µs windows; C, D spin) | 234–676 | 523–809 | ~0 | 2.00 | 0 |
| C queries, spinning; D adaptive | 277–462 | 677–774 | ~0 | 1.03 | 0 |
| (a) blocking-sync event | 1,589–1,781 | 2,163–2,396 | 991–1,415 | 0.05 | 0 |
| (b) host function + futex | 1,860–2,121 | 2,471–3,294 | 1,401–1,719 | 0.05–0.06 | 0 |
| (c) memop flag, yield | 286–711 | 719–1,538 | 0–1 | 1.03 | 0 |
| (c) memop flag, 50 µs futex poll | 304–346 | 392–821 | 17–68 | 0.06–0.07 | 0 |
| (c) memop flag, WFE | 289–732 | 765–1,089 | ~0 | 1.03 | 0 |
| (d) adaptive spin, C only | 271–477 | 497–878 | ~0 | 0.08 | 0 |
| **(e) adaptive spin + relay** | **29–30** | **38–212** | ~0 | **0.12–0.13** | **0** |
| (e) with a host function as backstop | 108–394 | 270–418 | ~0 | 0.13–0.14 | 0 |

40 ms steps, 1.3 ms host work: harness-polled 7–8 µs (3.97 cores); until
now 690–705 (1.97); (a) 698–1,959; (b) 1,589–2,203; (c) 599–1,744;
(d) 625–946 (0.11–0.12); (e) 29 µs, p99 100–315 (0.20–0.21 cores); (e) with the
host function 29–30 (0.20–0.22). 5 ms steps: harness-polled 8 (3.95);
until now 238–481 (2.00); (a) 1,634–1,712; (b) 902–1,900; (c) 224–715;
(d) 230–274 (0.43–0.47); (e) 24–29, p99 32–53, at 0.98–1.10 cores (C, K,
S and D each spin about 1.3 ms of every 5.2 ms); (e) with the host
function 38–95.

What they show:

1. **The GPU's own signal is slow on the GB10.** A blocking-sync event's
   wait returned 1.0–1.4 ms after the step's end at the median, a host
   function's futex 1.4–1.7 ms. Queries and a mapped flag see the end
   within microseconds, so detection wants polling.
2. **A host function holds its stream.** The next step, launched while
   the driver's callback thread has not yet run the previous step's host
   function, waited for it: S to GPU start was 114–398 µs with 0.2 ms of
   host work, and ~35 µs with 1.3 ms (the callback thread had run by
   then). It also adds a queue entry per fence (RE-029).
3. **Detection is not the cost; sleeping hops are.** Every configuration
   that detects in microseconds but lets K, S or D sleep pays 100–600 µs
   at each sleeping hop: (c) and (d) cost what the old defaults did.
4. **Only keeping the next thread awake ahead of time removes it.** With
   the relay, every hop is under 2 µs at the median; the remaining ~20 µs
   over the harness's is S's launch (S to GPU start 31–38 µs against
   12–15 µs when every thread spins; not isolated: possibly the core S
   runs on after sleeping, or its caches).
5. **A single expected length fails mixed steps.** With 45 ms and 5 ms
   steps mixed, (e) predicting the shortest recent length gave 920–1,159 µs
   at the median (C spun around 5 ms, then backed off through the long
   steps). Taking each recent length as a likely end fixed it (below).

## The choice (D-094)

(e), without host functions, is the default of the scheduler and the
device lanes (`src/scheduler/services.h` `DeviceSettings`,
`scheduler.h` `SchedulerSettings`, `base/wake.h`):

- The completion lane takes each of its stream's last eight fence
  lengths as a likely end (`base::Expectation`), sleeps until 1 ms before
  the next one a fence has not outlasted (querying at least every 1 ms),
  spins to 1 ms past it, and past the longest backs off (50 µs doubling to
  1 ms). A stream with no history spins its first 1 ms. With no fence it
  sleeps.
- As it starts to spin around a likely end it anticipates the scheduler
  (`CompletionBoard::Anticipate`) and its submission lane until 1 ms past
  it.
- The scheduler polls while anticipated, and after a step (device work,
  under a request's lease or its own) for 1.5 times the recent gap to the
  next step's publication plus its 200 µs window, at most 10 ms
  (`follow_limit`), and has the device lane poll as long.
- The submission lane sleeps on its own wake flag (signalled by `Submit`
  and `Close`) outside its window and anticipations.

Why not the others: (a) and (b) wake 1–2 ms late; (b) also stalls the
next launch and adds stream entries; (c) needs polling anyway (a GPU
write wakes nothing) and adds an entry per fence, for no faster
detection than a query; (d) alone leaves K, S and D asleep. The margins
(1 ms ahead, 1 ms past, 1 ms backstop) cover a timed sleep's wake on this
host (up to ~0.8 ms, RE-017); they are measured choices, not tuned per
model. Completion semantics are unchanged: only a query that sees a fence
complete proves it; an anticipation only says when to poll.

The harness's 100 ms window stays as a labelled diagnostic
(`NodeSettings::poll_window`, `--poll-us` in `llmp_swap_pairs` and
`llmp_spec_runner`); by default the harness runs the runtime's wake,
and its driver waits the same way (asleep through most of a step,
spinning around its likely ends, woken early by the step's report).

## After the change

Chain, (e) as built (each recent length a likely end), same scenarios,
three runs:

| Scenario | Harness-polled p50 (cores) | Until now p50 (cores) | (e) p50 | (e) p99 | (e) cores stepping |
| --- | ---: | ---: | ---: | ---: | ---: |
| 45 ms, 0.2 ms host | 8 (3.99) | 218–487 (2.00) | 26–31 | 50–1,122 | 0.12 |
| 40 ms, 1.3 ms host | 8–9 (3.97) | 700–869 (1.96–1.97) | 30–32 | 240–353 | 0.20–0.21 |
| 5 ms, 0.2 ms host | 8 (3.95) | 228–238 (2.01–2.02) | 28 | 48 | 0.95–0.98 |
| 45 ms and 5 ms mixed | 8 (3.99) | 239–676 (1.99–2.00) | 25–34 | 656–1,655 | 0.33–0.35 |

Every configuration was idle at 0.00 cores. The p99s over 99 steps
include the first step, which no history predicts, and in the mixed
scenario the short steps that follow eight long ones (C asleep: its
backstop sees them within ~0.3 ms, p99).

**The runtime path** (node, 45 ms steps, 0.2 ms host work, three runs of
99 steps each):

| Node | Round trip p50 | p99 | Cores stepping | Cores idle |
| --- | ---: | ---: | ---: | ---: |
| Before, harness-polled (100 ms) | 9.1–9.3 µs | 13.3–18.4 µs | 3.99 | 0 |
| Before, runtime defaults (200 µs) | 561–667 µs | 772–776 µs | 2.00 | 0 |
| **After, runtime wake (defaults)** | **25.6–27.8 µs** | **46–132 µs** | **0.11–0.12** | **0** |
| After, 100 ms windows (diagnostic) | 10.6–10.9 µs | 120–151 µs | 2.06 | 0 |

## Re-benchmark: the models with the runtime wake

`spark-b`, 10:31–11:39, the artifacts of the M3 runs (DeepSeek
`8a355bfb…`, Qwen3.8 `c4fb47a9…`, the DSpark drafter `dd2d3f9c…`), the
memory gate before each process (no GPU process, more than 110 GB
`MemAvailable`, load under 6, twice), coarse (D-085): two or three runs of
the decode benches, one of each other; this change on the
lease-per-request tree (`a84146c`), then again after main's Qwen3.8 fast
prefill (`f9a4e0f`), from which Qwen3.8's figures come. "100 ms windows" is the same binary
with `--poll-us 100000`, the old harness's polling (its completion lane
and driver now wait the new way; on a synthetic step it took 10.6–10.9 µs
at the median against the old harness's 9.1–9.3), as a same-session
reference. Details and raw outputs: [swap](../fast-swap/swap.md#with-the-runtime-wake),
[graphs](../fast-swap/graphs.md), [dspark](../dspark/README.md#performance-and-memory).

| Measure | Runtime wake | 100 ms windows | Before (harness-polled unless noted) | Reference |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek decode, graphs, a request, tok/s | **20.42; 20.50; 20.48** | 20.32 | 20.34–20.46 | llama.cpp tg64 20.01 (fusion off) / 20.43 (on), same session: **1.020–1.024× / 0.999–1.003×** |
| DeepSeek's round trip a step | 0.033–0.046 ms | 0.017 ms | 0.009–0.013 ms | |
| Qwen3.8 decode, a request, tok/s (`f9a4e0f`) | **24.09; 23.85** | 24.03 | 23.71–23.81 | Mia's vLLM, speculation off, 25.12–25.33: **0.94–0.96×** |
| Qwen3.8's round trip a step | 0.029–0.033 ms | 0.015 ms | 0.009–0.012 ms | |
| DeepSeek with DSpark, `prose` / `code`, tok/s (median of three) | **29.67 / 30.85** | 29.63 / 30.30 | 27.94 / 29.33 (lanes at 200 µs, a lease per step) | llama.cpp with the same drafter 30.80 / 31.94: **0.96× / 0.97×** |
| DeepSeek prefill, 8,192 tokens | 27.37–27.45 s | 27.56 s | 27.34 s | |
| Qwen3.8 prefill, 8,192 tokens (`f9a4e0f`) | 6.84 s | | 8.10 s (before the fast prefill; lanes at 200 µs, a lease per step) | |
| DeepSeek ↔ Qwen3.8 swaps | 7.06–8.81 s | 7.40–8.80 s | 6.86–8.89 s (lanes at 200 µs) | |

A step with a lease of its own (the per-step arms, M7's path for routed
experts) first paid more with the wake than polled (Qwen3.8's round trip
1.65–2.04 ms against 1.05): its closure is walked and leased before the
submission, by which time the device lane's anticipation had lapsed. The
follow window was then extended from a request's steps to every step
(device work through a task), measuring the gap to the next step's
publication, walk included; a confirming run (11:19–11:24) gave 1.00 ms
(Qwen3.8) and 1.32 ms (DeepSeek, graphs), against 1.05 and 1.51 with the
100 ms windows.

So with the runtime's own wake the models decode as fast as they did
harness-polled: the 0.01–0.03 ms a step it adds is less than the device's
run-to-run spread (DeepSeek's step 48.5–48.9 ms, Qwen3.8's 40.2–40.9).
DSpark gained 5–6% over its first figures (lanes at 200 µs, a lease per
step), mostly from running each generation as a request (D-093), which
the spec runner now does: per step the lease alone cost DeepSeek
1.3–2.6 ms, against the old windows' 0.26–0.65 ms of wakeups that the
wake removes, and the rest of the 5.4 ms a step is not isolated; against
the 100 ms windows in the same session the wake changed nothing
measurable. Every check stayed
exact: every bench pass equal to its warm-up, every DSpark greedy check,
every swap's state digest and continued steps.

## A latency hold

The owner, on 2026-09-28, approved testing what D-094 left untried: a PM
QoS CPU latency request, which keeps idle CPUs out of the idle states
whose exit latency exceeds its value while its descriptor is open
(`/dev/cpu_dma_latency`: write an int32, keep the descriptor; root-only,
0600).

**The platform** (`spark-b`, read-only): cpuidle driver `acpi_idle`,
governor `menu` (`ladder` and `teo` available), the same four states on
all 20 CPUs, none disabled:

| State | Description | Exit latency | Target residency |
| --- | --- | ---: | ---: |
| LPI-0 | core idle, WFI | 0 µs | 0 µs |
| LPI-1 | core off | 42 µs | 1,930 µs |
| LPI-2 | core off, DSU off | 231 µs | 2,542 µs |
| LPI-3 | core off, DSU off, platform | 433 µs | 2,542 µs |

So a request of 0 to 41 µs allows only LPI-0, and 42 to 230 µs adds
LPI-1. The request's own cost (a root Python loop, 200 samples, 5 ms
apart): engaging an open descriptor (writing 0) 32 µs at the median,
49 µs p99 (the kernel wakes every idle CPU to apply it); releasing
(writing -1, the kernel's default, no constraint) 8 µs, 15 µs p99;
opening, writing and closing 71 µs, 88 µs p99. Opening alone constrains
nothing.

**Method.** The chain and node modes above, with a root holder process
(`sudo -n`, a few lines of Python that open the device, write the value
and sleep; killed after each run, and the effective value read back from
the device before and after). Configurations interleaved
(no hold, hold, no hold, ...), each run once no GPU process ran and the
1-minute load was under 6 (other agents used the host between runs; load
0.1–3.6 at the starts). `--spin-ahead-us` (new, a diagnostic) sets how
long before a likely end the completion lane and the client start to
spin, D-094's 1 ms by default. The tree of this slice (main `00aaa97`
plus the harness's new options), `spark-native`, 12:12–12:39 on
2026-09-28; raw outputs in `~/scratch/dmalat/` on `spark-b`. One run
(45 ms steps, 200 µs margin, no hold, the second) is left out: a 2 s
hold of this slice's own ran during it by mistake. The device read back
2,000,000,000 (the default: no constraint) and no holder remained after
every one of the 68 runs with a holder or without.

**Every mechanism, 45 ms steps, 0.2 ms host work** (µs, two runs each of
99 hops, p50 / p99; cores while stepping; idle was 0.00 in every run):

| Configuration | No hold | Hold at 0 µs | Hold at 50 µs | Cores (no hold / 0 µs) |
| --- | ---: | ---: | ---: | ---: |
| Harness-polled (100 ms windows) | 8 / 9 | 8 / 8–9 | 8 / 9 | 3.99 / 4.00 |
| Runtime before D-094 (200 µs windows) | 229–232 / 560–693 | **20–21 / 23–42** | 234–267 / 489–522 | 2.00 / 2.01 |
| C queries, spinning; D adaptive | 293–723 / 834–974 | 22–23 / 25–53 | 51–503 / 617–1,396 | 1.03 / 1.04 |
| (a) blocking-sync event | 1,461–1,748 / 2,345–2,833 | **59–60** / 86–140 | 245–1,207 / 559–1,520 | 0.06 / 0.05 |
| (b) host function + futex | 1,521–1,952 / 2,313–2,779 | 58–66 / 70–129 | 905–1,061 / 1,022–1,501 | 0.05 / 0.05 |
| (c) memop flag, yield | 707–736 / 993–1,013 | 27–29 / 34–36 | 516–585 / 640–924 | 1.03 / 1.04 |
| (c) memop flag, 50 µs futex poll | 281–798 / 374–914 | 44–81 / 123–133 | 251–279 / 575–591 | 0.06 / 0.07 |
| (c) memop flag, WFE | 483–724 / 826–1,004 | 28–29 / 32–36 | 389–506 / 622–656 | 1.03 / 1.04 |
| (d) adaptive spin, C only | 223–709 / 719–1,324 | **22** / 27–84 | 220–227 / 254–394 | 0.07 / 0.07 |
| **(e) the runtime wake (D-094)** | **29–31** / 42–672 | **9** / 65–75 | 27 / 122–389 | 0.12 / 0.14 |
| (e) with a host function as backstop | 34–96 / 187–1,393 | 9 / 17–18 | 34–162 / 378–465 | 0.11–0.13 / 0.14 |

With the hold at 0 µs, each sleeping hop (the scheduler, the client, the
submission lane) took ~5 µs to run instead of 100–600, and the step's
launch after S took the command (S to GPU start) 6–7 µs instead of
26–28: the "~20 µs left over" above was S's core leaving a deep state.
The GPU's own signal became usable (a blocking-sync event's wait returned
5–35 µs after the step's end at the median, against 1.0–1.5 ms), but not
faster than D-094's wake. At 50 µs, which still allows LPI-1, nothing
improved: the cost is the cores' power-down (LPI-1 on), not only the
platform's states. 20 µs behaved as 0 (one run each at 1 ms and 0 µs
margins: 8 and 267 µs), as the table of states predicts.

**The margin under the hold** (D-094's wake, µs, p50 / p99, cores):

| Scenario (runs) | Spin ahead | No hold | Hold at 0 µs |
| --- | --- | ---: | ---: |
| 45 ms, 0.2 ms host (3) | 1 ms (default) | 28–29 / 44–1,395, 0.11–0.12 | **8 / 14–83, 0.14** |
| | 200 µs | 375–578 / 402–1,279, 0.06 (2) | 8 / 18–328, 0.07 |
| | 50 µs | 625–941 / 698–1,367, 0.04–0.05 | 59–261 / 81–268, 0.04 |
| | 0 | 997–1,050 / 1,120–1,378, 0.05 | 269–322 / 304–371, 0.04 |
| 40 ms, 1.3 ms host (2) | 1 ms | 29–30 / 122–347, 0.20 | **8 / 10–25, 0.22** |
| | 200 µs | 288–291 / 637–799, 0.15 | 8 / 616–620, 0.14–0.15 |
| 5 ms, 0.2 ms host (2) | 1 ms | 28 / 48–68, 0.92–0.95 | **8–9 / 9–20, 1.15** |
| | 200 µs | 382–384 / 551–1,149, 0.41 | 8 / 138–144, 0.54 |
| 45 and 5 ms mixed (2) | 1 ms | 30 / 1,131–1,906, 0.29–0.31 | **8 / 292–294, 0.37** |
| | 200 µs | 358–566 / 881–1,147, 0.15–0.18 | 8 / 294–296, 0.20 |
| Node, the runtime's path (2) | 1 ms | 27.6–28.9 / 62–161, 0.11–0.12 | **9.7–10.0 / 14–15, 0.14** |
| | 200 µs | 308–559 / 610–968, 0.05–0.06 | 9.8–10.1 / 18–273, 0.07 |

(The p99s over 99 steps include the first, which no history predicts.)

1. **The hold removes ~20 µs a step at the median in every scenario**,
   and on the runtime's own path (node) 28 µs becomes 10: what polling
   every thread bought at four cores, at 0.14 of one. The tail improved
   too where the margin was kept.
2. **The margin cannot go.** At 0 or 50 µs the completion lane wakes
   late: a timed sleep overshoots by the thread's timer slack (50 µs by
   default), and a late wake records a longer length, which the next
   step's prediction starts from, so the prediction ratchets late until
   the 1 ms backstop bounds it (detection 210–310 µs late at the median).
3. **200 µs keeps the median and halves the CPU, but not the tail:**
   p99 138–620 µs, where 1 ms gave 9–83. On a 45 ms step that is about
   0.07 of a core saved against a few steps in a hundred paying a few
   hundred microseconds; D-094's 1 ms would stay even with a hold.
4. **CPU while stepping rose slightly with the hold** (0.12 → 0.14 cores
   at 45 ms, 0.92–0.95 → 1.15 at 5 ms): not isolated; the threads that
   spin now run on cores that wake at once. Idle stayed at 0.00.

**DeepSeek decode** (`llmp_swap_pairs --a dsv4 --b qwen38 --cycles 0
--bench 64`, the artifacts and driver of the re-benchmark above, the
memory gate before each process, 12:31–12:38, alternating; the hold from
outside, at 0 µs, for the whole process):

| Run | Request lease, graphs: tok/s (mean of 3) | Round trip a step | Job's host time | Device a step |
| --- | ---: | ---: | ---: | ---: |
| No hold | 20.42; 20.42 | 0.041; 0.042 ms | 0.156; 0.161 ms | 48.64; 48.71 ms |
| Hold at 0 µs | 20.48; 20.33 | **0.010; 0.013 ms** | 0.056; 0.080 ms | 48.55; 48.92 ms |

The hold took ~0.03 ms a step off the round trip (and ~0.08 ms off the
job's host time, a graph's replay), now within ~0.002 ms of every thread
polling; decode is not measurably faster, because the device's own
run-to-run spread (48.5–48.9 ms a step) is ten times the gain. The
per-step-lease arms (M7's path) stayed at 1.3–2.0 ms of round trip:
their cost is the lease's walk, not a wakeup. Every pass of all four runs
equalled its warm-up (0 steps differ, 16 of 16 each).

**The scheduler's own hold** (a prototype, since removed: the node with
the scheduler engaging the hold itself, run as root so that it could
open the device; two runs each, alternating, 12:38–12:39): 9.8–10.5 µs
at the median (8.5 and 10.5; p99 15–17), 0.14 cores, against 25.5–26.5
µs (p99 55) without. Sampled every 0.5 s, the host's effective value was
0 from the first step to about 1 s after the last (its linger), and the
default again before the process exited.

**Not kept (D-095).** The owner, on 2026-09-28: "D-095 is accepted -
host-wide effect is fine but if there's no meaningful impact it's not
worth keeping and our own scheduler is much more efficient." About 17 µs
a step against 40–48 ms steps changed no decode throughput beyond the
device's spread, so the prototype was removed; it is worth rebuilding
for short-step workloads (M7's per-step routed experts), where ~17 µs a
step matters. What it would take:

- **Only while active.** The prototype's scheduler engaged the hold once
  an operation was in flight (not counting quarantined ones) and
  released it once none had been for a linger of 1 s: longer than a
  client's gap between steps or a conversation's quick turns, so they do
  not re-engage it (32 µs each time); and when it stopped. A hold the
  host refused was asked once, and the scheduler carried on without it.
- **One descriptor, kept:** engaging writes the limit (0 µs), releasing
  writes -1. Closing also ends the request, but opening and closing
  costs more, and an unprivileged service cannot reopen a root-only
  device.
- **Permissions, least privilege:** systemd's `OpenFile=` in
  `llmp.service` (`OpenFile=/dev/cpu_dma_latency:cpu-latency:graceful`,
  systemd 253+; `spark-b` runs 255) has the service manager open the
  device and pass the descriptor. The runtime finds it by name
  (`LISTEN_FDS`, `LISTEN_FDNAMES`), marks it close-on-exec so its jobs do
  not inherit it, and needs no capability, no device access
  (`DevicePolicy=closed` stays) and no change to the device node. The
  alternatives give more: a udev rule granting the `llmp` group `rw`
  (plus `DeviceAllow=`) lets every process of that group, and every job
  in the unit's cgroup, open a host-wide knob, and changes the node for
  the host's life; a root helper service is a second process and an IPC
  channel. `graceful` starts the service on a host without the device,
  and an older systemd ignores the line; either way the runtime would run
  without the hold, as would a development run whose user may not open
  the device.
- **Checked on `spark-b`** with a transient unit (`systemd-run`, as
  `nobody`, `DevicePolicy=closed`, no capabilities,
  `ProtectSystem=strict`): the service received `LISTEN_FDS=1`,
  `LISTEN_FDNAMES=cpu-latency`; its own open of the device was refused
  (EACCES); a write of 0 through the descriptor set the host's effective
  value to 0, a write of -1 returned it to the default, and nothing
  remained after the unit exited.

## Tests

- `unit.WakeFlag.*`, `unit.Expectation.*` (`lanes_test`): an
  anticipation only raises the deadline and wakes the owner; `WaitUntil`
  returns for a signal or at its deadline; four publishers that
  anticipate between publications lose no wakeup of an owner that polls
  while anticipated and sleeps otherwise; the likely ends.
- `unit.DeviceWakeTest.*`, `unit.DeviceWakeShutdown.*` (`wake_test`,
  fakes, both device lanes on their own threads): fences that end on
  time, early, late and at once are all seen with their proof; one ending
  at once when 100 ms is expected is seen within 50 ms (the backstop, not
  the expectation), and ones far longer than every recent length within
  20 ms of completing (median); a query whose outcome is unknown, or that
  keeps being refused, while the lane sleeps is published unproven long
  before the expected end; the owner and the submission lane are told to poll
  before an expected end, and not at the launch; a sleeping submission
  lane wakes for each command (median under 50 ms, the tick being 100);
  four submitters lose no command or fence; closing wakes sleeping lanes;
  a close with a fence out waits for it, however late.
- `unit.HeldLeaseTest.AStepsEndHasTheDeviceLanePollForTheNext`: a
  request's step ending has the device lane poll for the next.
- `unit.CudaPagedNodeTest.RequestStepsSleepBetweenAndLoseNoCompletion`
  (GB10): a request's steps gated on a host flag the test opens on time,
  early, late or at once all complete; before a step's expected end the
  scheduler and the device lane are told to poll, after the completion
  lane slept through most of it; stepping costs under a core, and nothing
  spins once the request has ended.
- The death tests of the lanes' and the scheduler's settings cover the
  new bounds.
- Under ThreadSanitizer (`spark-native` with `LLMP_SANITIZE=thread` on
  `spark-b`): `wake_test` and `held_lease_test` three times each, and the
  scheduler, page-in, lanes and acquisition tests, with no report.

## Limits

- One host, a shared one; three runs per configuration; D-085's coarse
  comparison. Synthetic steps in the chain and node modes.
- The relay needs history: a stream's first fences, and a step shorter
  than any of the last eight, pay a sleeping hop or two (up to ~1 ms).
- Short steps cost more CPU: the spins around each likely end are fixed
  at 1 ms, so 5 ms steps keep about a core busy in all (idle is still
  zero).
- The ~20 µs left over the harness's polling was S's core leaving a deep
  idle state: a latency hold removes it ([above](#a-latency-hold)), but
  it is not kept (D-095).
- The latency hold's measurements: two or three runs per configuration,
  interleaved. The idle power it costs was not measured: the GB10 has no
  CPU power reading that is cheap to take, and `nvidia-smi`'s power draw
  (the GPU's) read 4.46–4.47 W idle with and without it. What it changes
  is where idle time goes: over 10 s of an idle host, the 20 CPUs spent
  201.6 s in LPI-3 and 1.5 s in LPI-0 without it, and 196.8 s in LPI-0
  with it. A per-CPU request (`power/pm_qos_resume_latency_us`, only the
  runtime's CPUs) and a shorter timer slack (`PR_SET_TIMERSLACK`, which
  might let the margin shrink) were not tried.

## Reproduce

On `spark-b`, from the tree's `spark-native` build:

```bash
build/spark-native/benchmarks/llmp_wake_bench chain --steps 100 --step-us 45000 --host-us 200 --repeats 3
```

```bash
build/spark-native/benchmarks/llmp_wake_bench node --steps 100 --repeats 3 [--poll-us 100000]
```

`--mix` mixes in 1/9-length steps; `--only PREFIX,...` picks
configurations by name; `--spin-ahead-us US` sets the margin (both
modes). A latency hold held from outside for one run (root; always kill
it after, and check the device reads 2000000000 again):

```bash
sudo -n python3 -c 'import os,struct,time; fd=os.open("/dev/cpu_dma_latency",os.O_WRONLY); os.write(fd,struct.pack("i",0)); time.sleep(600)' &
build/spark-native/benchmarks/llmp_wake_bench node --steps 100 --repeats 1
sudo -n pkill -f 'struct.pack\("i",0\)'; sudo -n od -An -td4 /dev/cpu_dma_latency
```
