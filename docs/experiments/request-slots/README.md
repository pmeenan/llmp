<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Request slots sized by memory (2026-10-03)

The owner's plan item of 2026-10-03: a model's concurrent request slots are
no longer fixed at four; they follow the memory free for them, and each
model's default cap is the knee of its throughput against each request's
own rate (D-104). This report measures that knee for both LLMs, the
memory a slot costs, and the admission by memory built on it. Everything
ran on `spark-b` (`spark-56f5`, GB10, driver 580.178.04) with the
`spark-native` build of this change on main `fcf78c8` and of main itself
in the same session; the wave-form section ran the change rebased on main
`318638a` against that main.

## Results

- **The knee is four slots for both LLMs** with today's 16-row joined
  products. Past four, DeepSeek DSpark gains 7.6% at six and 1% more at
  eight while every request decodes 26% and then 24% slower; with long
  prompts nothing is gained. With main's chosen wave form, extended here
  to widths 5 to 8 (plain waves past five, where a cut DSpark verify
  cannot pay), it gains 8.3% and 6.5% while each request decodes 25% and
  20% slower: still the knee. Qwen3.8 gains nothing. The first cells
  planned every new wave composition cold inside the burst, so a second,
  warm burst in the same service decides it ([warm
  plans](#warm-plans)): there Qwen3.8 completes 63.7 tok/s at four, 60.6
  at six (two groups, five and one, each reading the weights) and 63.9 at
  eight (five and three) while each request decodes 36–48% slower, and
  DeepSeek gains 9.3% at six and 2.6% more at eight while each request
  decodes 24% then 12% slower (long prompts: level, 29% then 18% slower).
  Both defaults stay 4 and are now the fallback of a configurable cap
  (`max_slots`, 1 to 16).
- **A large cap costs plan building and plan memory, for as long as the
  service runs.** Every wave composition (which slots, at which shapes) is
  planned when first met, and more slots meet far more of them, so
  planning does not stop after a first burst and grows steeply with the
  cap: across two bursts Qwen3.8 planned for 1.8 s at four slots, 10.7 s
  at eight and 68 s at sixteen (the second burst alone, against a single
  cold burst's, still about 3.6 s at eight and 28 s at sixteen), and held
  210 MiB, 794 MiB and 2.7 GiB of plans beside 0.7, 2.9 and 3.6 GiB of
  graphs (DeepSeek: 1.3 and 3.9 s, 117 and 278 MiB of plans, 0.6 and 1.5
  GiB of graphs at four and eight). They are charged inside the budget and
  given back through the reclaim order, but they take state room and
  planning time a smaller cap does not: the real cost of a large
  `max_slots`.
- **A slot's fixed buffers are small**: 15.8 MiB for Qwen3.8 (its verify
  snapshot and output slices) and 4.2 MiB for DeepSeek with DSpark,
  measured from the fixed allocations at four, six and eight slots. A
  slot's state, the part that grows (hundreds of MiB to GiB), is already
  lazy and given back through the reclaim order, so the buffers are set up
  at start for the cap rather than funded lazily.
- **Qwen3.8's workspace no longer scales with slots.** It was four times
  the largest activation, a 4,096-row prefill chunk's, though waves run
  only few-row verifies. Sized from its widest wave as measured at setup,
  the workspace falls from 6.80 to 1.53 GiB at four slots (fixed 7.45 →
  2.17 GiB), its conversation-state room grows from 31.33 to 36.78 GiB,
  and the peak `MemAvailable` drop of every cell falls by 5.5 GiB. Eight
  slots cost 0.06 GiB more than four.
- **No regression at C1/C2/C4** against main in the same session; replies
  at C1 and C2 are byte-identical, and DeepSeek's at C4 too.
- **More slots are exact.** DeepSeek's wave checks at six and eight slots
  give every row bit-identical to each slot alone (plain: 522/522 and
  712/712 rows; DSpark: drafts and verify rows exact, discarded verifies
  restored with 0 stale bytes).

## How each model scales past four

### DeepSeek V4 Flash, pure decode (harness)

`jitllm_spec_runner --check wave --slots N --tokens 96`, original artifact
(`8a355bfb…`), the fast-swap prompts (two decode, six chat). Plain decode
waves (one row a request) and DSpark waves (each request a share of the
wave's sixteen rows: four up to four requests, two at six and eight).

| Slots | Plain waves, tok/s | A request's rate | DSpark waves, tok/s | Rows identical / checks |
| ---: | ---: | ---: | ---: | --- |
| 1 (solo) | 21.9 | 21.9 | 32.3–33.6 | — |
| 2 | 27.5 | 13.8 | 36.1 | 142/142, DSpark exact |
| 4 | 41.5 | 10.4 | 40.3 | 332/332, DSpark exact |
| 6 | 47.4 | 7.9 | 44.0 | 522/522, DSpark exact |
| 8 | 50.9 | 6.4 | 44.5 | 712/712, DSpark exact |

The step grows with the rows: each request's routed experts are mostly
its own (a layer's 256 experts, 6 a token), so a wave reads about as much
per row whatever its width, and the shared dense weights are a small part
of the step. Two to four slots gain 51%; four to six 14% (DSpark 9%), six
to eight 7% (1%). DSpark's verify shrinks to two rows a request past four,
so it accepts fewer drafts a step and trails plain waves there.

### Served (HTTP cells)

A fresh service per cell, C concurrent streamed greedy chat requests,
production configuration (DeepSeek: artifact `8a355bfb…` with DSpark
`dd2d3f9c…`, context 262,144; Qwen3.8: `c4fb47a9…` with MTP `8600a998…`,
context 33,792, 4,096-row chunks). *Short*: ~120-token prompts, 512
outputs (decode-dominated). *Long*: the matched ~7K-token prompts (u0–u3;
past four, the same with a distinct first line), 256 outputs. One sample
a cell. *Rate*: completed tokens over the cell's wall. *Decode*: the median
request's tokens a second after its first. *Done*: the median request's
completion. *Main* has four slots: its C8 queues four requests behind
four.

| Model, workload | Cell | Main: rate, decode, done s | This change: rate, decode, done s |
| --- | ---: | --- | --- |
| DeepSeek DSpark, short | C1 | 32.51, 35.1, 15.7 | 32.47, 34.9, 15.8 |
| | C2 | 35.87, 19.8, 27.8 | 35.99, 19.8, 27.7 |
| | C4 | 37.91, 10.5, 53.2 | 38.46, 10.4, 52.5 |
| | C6 | — | 41.40, 7.7, 72.8 |
| | C8 | 38.60, 10.5, 78.0 (4 slots) | 41.77, 5.9, 95.3 |
| DeepSeek DSpark, long | C1 | 16.21, 34.1, 15.8 | 16.08, 33.7, 15.9 |
| | C2 | 17.32, 16.3, 29.4 | 17.26, 16.2, 29.5 |
| | C4 | 17.73, 7.8, 56.8 | 17.72, 7.7, 57.0 |
| | C6 | — | 17.88, 4.8, 84.3 |
| | C8 | 17.98, 6.1, 93.6 (4 slots) | 17.68, 4.6, 113.1 |
| Qwen3.8 MTP, short | C1 | 42.33, 44.1, 12.1 | 41.95, 43.7, 12.2 |
| | C2 | 53.32, 28.7, 18.7 | 53.22, 28.7, 18.7 |
| | C4 | 61.65, 16.6, 32.0 | 60.75, 16.6, 32.4 |
| | C6 | — | 57.83, 10.6, 51.8 |
| | C8 | 62.92, 16.9, 49.5 (4 slots) | 60.05, 8.7, 64.6 |
| Qwen3.8 MTP, long | C1 | 24.36, 37.5, 10.5 | 24.42, 37.6, 10.5 |
| | C2 | 28.86, 22.2, 17.5 | 28.95, 22.2, 17.4 |
| | C4 | 31.25, 12.3, 32.0 | 31.40, 12.4, 31.4 |
| | C6 | — | 29.70, 7.3, 50.6 |
| | C8 | 31.90, 11.5, 51.0 (4 slots) | 30.82, 6.3, 64.8 |

- **C1/C2/C4 against main**: within −1.5% to +1.5% in every cell (the
  largest, Qwen3.8 short C4 −1.5%, is one sample of a cell whose replies
  vary with arrival timing on both binaries). C1 and C2 replies are
  byte-identical on both models, and DeepSeek's C4 too.
- **The knee.** In a burst of eight the median request completes later
  with eight slots than with four on every workload (DeepSeek 95 against
  78 s short, 113 against 94 long; Qwen3.8 65 against 50 and 65 against
  51): with four slots the first four finish early and the rest follow,
  with eight every request shares every wave. Only DeepSeek short ends
  the whole burst sooner (98 against 106 s). Throughput past four rises at most 8%
  (DeepSeek short, six slots) and falls for Qwen3.8 and for long prompts,
  whose serial prefill dominates. These cells plan every wave cold inside
  the burst, more of them the more slots run; the [warm](#warm-plans)
  bursts below gain 2–9% everywhere and lead to the same knee: four is
  where returns diminish and individual requests are delayed, the default
  for both.
- **Replies past four.** In these cells (forced DSpark, before the
  rebase) DeepSeek's verify shrinks to two rows a request past four
  slots, so its steps, and its replies, differ from C1's (plain waves
  stay byte-identical). With the chosen form, waves of six to eight are
  plain, whose rows equal a request's plain steps alone; a reply still
  differs from its DSpark reply alone. Qwen3.8's vary with arrival timing
  at any concurrency, as before.

### Warm plans

The cells above start a fresh service and run one burst, so every new
wave composition (each set of slots and their shapes) is planned and
captured inside it: more slots meet more compositions, and Qwen3.8's
planning grew from 1.3 s at four slots to 7.1 s at eight and 40 s at
sixteen of a 162 s run (the review's cells). Each cell here runs two
bursts in one service, the second's prompts the first's with one letter
of their tag changed (the same shapes; no conversation reused, 0 cached
tokens), and the second is reported. *Warm* means its compositions met
before are planned: the second burst still meets new ones (waves form
from other arrival orders and positions) and plans them, more the more
slots run (about 3.6 s at eight and 28 s at sixteen for Qwen3.8, two
bursts' planning less a single cold burst's). The rebased build (main `318638a`),
`spark-b`, one sample a cell; rate, median request's decode rate after
its first token, median completion:

| Model, workload | C4 | C6 | C8 | C16 |
| --- | --- | --- | --- | --- |
| Qwen3.8 MTP, short | 63.72, 17.4, 30.7 | 60.55, 11.1, 48.8 | 63.85, 9.0, 61.6 | 56.49, 4.1, 140.9 |
| Qwen3.8 MTP, long | 32.22, 12.9, 30.9 | 31.95, 8.8, 47.7 | 32.23, 8.0, 61.8 | 29.24, 3.6, 136.5 |
| DeepSeek DSpark, short | 41.15, 11.0, 49.5 | 44.98, 8.4, 66.6 | 46.17, 7.4, 78.9 | — |
| DeepSeek DSpark, long | 18.33, 8.5, 55.0 | 18.23, 6.1, 82.2 | 18.45, 5.0, 108.2 | — |

The first (cold) burst of the same services: Qwen3.8 short 61.32, 58.00,
62.02, 53.03; long 31.08, 29.29, 30.32, 26.96; DeepSeek short 39.75,
43.84, 46.62; long 17.57, 17.97, 18.23. The second burst runs 2–9% faster
at every width, C4 included, and changes no conclusion: past four Qwen3.8 gains
nothing at any width (its groups each read the weights again, and a
row's routed experts are mostly its own), DeepSeek's short prompts gain
less than its requests slow, and its long prompts gain nothing. The
knee stays four.

Plans and graphs over both bursts, as each service counted them at its
end:

| Cell | Plans | Planning, s | Graphs counted |
| --- | --- | ---: | ---: |
| Qwen3.8 short C4 / C8 / C16 | 109 / 178 / 328, 210 / 794 / 2,708 MiB | 1.8 / 10.7 / 68.2 | 691 / 2,945 / 3,551 MiB |
| Qwen3.8 long C4 / C8 / C16 | 111 / 189 / 347, 276 / 842 / 2,556 MiB | 2.5 / 11.3 / 60.1 | 731 / 2,418 / 3,129 MiB |
| DeepSeek short C4 / C8 | 32 / 62, 117 / 278 MiB | 1.3 / 3.9 | 557 / 1,468 MiB |
| DeepSeek long C4 / C8 | 46 / 103, 184 / 562 MiB | 2.2 / 9.4 | 878 / 3,913 MiB |

### DeepSeek's wave form past four requests

With DSpark, main chooses each wave's form (`execution/adaptive_wave_mode.h`,
[deepseek-batching](../deepseek-batching/README.md#adaptive-dspark-and-plain-waves)):
a draft-verify wave while the tokens it commits a request exceed its
width's cost in plain waves. Its costs covered widths 2 to 4. Each form's
wave timed in the wave check (`--check wave --slots N`, community GGUF
`cd39d504…` with DSpark, the same prompts; `width_ms` in its record), ms a
wave:

| Width | DSpark wave | Plain wave | Ratio | Recorded cost | A DSpark verify's rows a request |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 135.4 | 62.3 | 2.17 | 1.94 (runtime) | 4 |
| 3 | 200.4 | 75.0 | 2.67 | 2.21 (runtime) | 4 |
| 4 | 256.5 | 87.1 | 2.94 | 2.90 (runtime) | 4 |
| 5 | 261.9 | 101.6 | 2.58 | 2.58 | 3 |
| 6 | 239.1 | 113.7 | 2.10 | 2.10 | 2 |
| 7 | 278.5 | 126.7 | 2.20 | 2.20 | 2 |
| 8 | 315.9 | 140.5 | 2.25 | 2.25 | 2 |

The wave check's plain waves run 62 against the runtime's 78 ms at width
2, so its ratios there sit above the runtime's recorded costs; at width 4
they agree (2.94 and 2.90). Widths 5 to 8 take the wave check's ratios,
as fallbacks for D-103's calibration (`kWaveCost`, `serving.cc`). Past
four requests a request's verify is cut to its share of sixteen rows, so
a draft-verify wave commits at most three tokens a request at width 5 and
two past it. The chooser now knows each width's cut (`Rows`): it counts
the average against that bound, the cut waves do not feed the average
(they would pull it below the full verifies' and turn widths 2 to 4
plain), and where the bound cannot pay the cost (widths 6 to 8: two
tokens against 2.10–2.25 plain waves) the wave is plain from the start,
without probes. Width 5 speculates while the full verifies of narrower
waves keep more than 2.66 tokens a request; its own waves cannot teach
the average, so it never probes and runs plain until narrower waves have
shown enough (plain was measured faster there, 46.0 against 41.1 tok/s in
the wave check). Unit-tested (`adaptive_depth_test`).

*With wave lanes* ([deepseek-batching](../deepseek-batching/README.md#wave-lanes),
2026-10-03), every width's cost is now the wave check's median ratio,
lanes on (the means above include each width's first, planning wave):
2.18, 2.68 and 2.99 at widths 2 to 4 (`ln2`), and at widths 5 to 8
246.9 / 95.6, 224.7 / 106.1, 258.0 / 116.6 and 287.3 / 129.1 ms: 2.58,
2.12, 2.21 and 2.23 (`spark:~/scratch/dss5/s9c`, `s9w`). Widths 5 to 8
barely move; past five the bound still cannot pay the cost.

Through the runtime, same session, production configuration
(`8a355bfb…` with DSpark), the rebased build (on main `318638a`) with the
chosen form (*auto*) against each form forced (`wave_form`) and main:

| Cell | Main `318638a` | Auto | DSpark forced | Plain forced |
| --- | ---: | ---: | ---: | ---: |
| Short C1: tok/s | 32.12 | 32.46 | — | — |
| Short C2: tok/s | 35.91 | 35.59 | — | — |
| Short C4: tok/s (decode) | 40.37 (11.0) | 40.69 (11.1) | — | — |
| Short C6: tok/s (decode) | — | 44.05 (8.3) | 41.55 (7.8) | 44.92 (8.1) |
| Short C8: tok/s (decode) | — | 46.92 (6.6) | 41.88 (5.8) | 47.56 (6.5) |
| Long C4: tok/s | 17.64 | 17.77 | — | — |
| Long C8: tok/s (decode) | — | 18.28 (4.9) | — | — |

At six and eight requests the chosen form runs within 1–2% of plain waves
and 6–12% above DSpark alone; C1 to C4 match main (C1 and C2 replies
byte-identical; at C4 main's chosen forms follow arrival order, as its
report says). The knee stays at four: past it, short prompts gain 8.3% and
then 6.5% while each request decodes 25% and then 20% slower, and long
prompts gain 2.9% at eight while each request decodes 37% slower and the
median request completes 109 against 57 s.

### Would wider joined products move the knee?

Not measurably, on this evidence, so they were not built. Qwen3.8's six
and eight requests split into two groups because three rows a request
fill sixteen at five. Drafting one token a request past five (two rows
each, so eight requests join one sixteen-row group) was tried and
measured slower: short C6/C8 56.2 / 57.9 tok/s against 57.8 / 60.1 at
depth 2, long 30.1 / 29.8 against 29.7 / 30.8; four requests at depth 2
reach 60.8 / 31.4. A wave's cost follows its rows (the routed experts are
near bandwidth and mostly each row's own: [four-request
waves](../qwen38-four-request-waves/README.md#open)), and fewer accepted
drafts a row lose what the joined group saves. DeepSeek already fills
sixteen one-row steps per wave and gains 7% from six to eight slots.
Joined products past sixteen rows would let Qwen3.8 keep depth 2 past
five requests, but the per-row cost they would still pay is what limits
both models; a large-slot configuration is for an owner who values a
burst's total rate over each request's.

## Memory a slot costs

The start's allocation guard (`fixed_catalog_bytes`) at each cap, same
configuration otherwise:

| Model | 4 slots | 6 slots | 8 slots | A slot |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek DSpark | 2,903,936,000 | 2,912,660,992 | 2,921,385,984 | 4.16 MiB |
| Qwen3.8 MTP | 2,325,188,396 | 2,358,350,124 | 2,391,511,852 | 15.8 MiB |
| Qwen3.8 MTP, main | 8,004,276,012 | — | — | (workspace ×4) |

Qwen3.8's setup plans its widest waves to size the workspace: setup
takes 1.01 s at four slots and 1.53 s at eight, against 0.79 s on main.
That is once, at registration; a swap does not set up again.

## Admission by memory

A request joins the running requests only when the budget holds its
prompt's state and what its peers' prompts have yet to take (each
`Llm::StateBytesThrough` its tokens, less what its branch holds), free or
freed through the reclaim order; otherwise it waits first in the queue
until a peer retires.

Eight long Qwen3.8 prompts at eight slots, with memory held outside the
runtime before it starts so that both arms have the same
conversation-state room (1.44–1.48 GiB, about two 8K conversations). The
review's matched A/B (`spark-b:~/scratch/slots-review/R3/admab`), two
services an arm, interleaved, the same build but for admission (*none*:
every request admitted, the capacity policy then making members wait at
their chunks):

| Arm | Waits for memory (queue) | Waits for capacity (cohort) | Set aside | Rate, tok/s | Median done, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Admission, 1 | 7 | 1 | 0 | 26.08 | 42.95 |
| Admission, 2 | 7 | 1 | 0 | 26.01 | 43.87 |
| None, 1 | 0 | 25 | 0 | 26.25 | 46.88 |
| None, 2 | 0 | 21 | 0 | 26.19 | 45.22 |

Every request completed in every service, and the service went on. With
admission almost no member waits inside the cohort: a request that does
not fit waits in the queue, holding no slot. The rate is 0.6% lower and
the median request completes 5.8% sooner (43.4 against 46.1 s). An
earlier pair of cells (Qwen3.8 25.27 against 28.89 tok/s, DeepSeek 17.69
against 17.62) compared services with different state room (1.62 against
1.00 GiB, 2.35 against 2.05) and is withdrawn.

Admission's reclaim spares the branch the request is about to continue
or clear: its state already counts as the request's own (`Llm::
ReusablePrefix` continued, or cleared for a new conversation), and the
review reproduced the reclaim spilling it (704.6 MiB for 55.3 MiB
needed), which then waited in the cohort and, for a new conversation,
wrote a spill the prompt discarded (`AddIdleStateCandidates`, unit-tested).
That reclaim also ranked a whole idle conversation (0.17 s a GiB) before
a few stale graphs (0.3–0.4 s a GiB) for a small need; the order now
takes, among its selections over each set of kinds, the one that covers
the need at the least total expected cost to restore, keeping least
recent use within a kind except that an entry larger than what is still
needed gives way to a later one of its kind that fits (100 MiB takes a
newer 30 MiB conversation and nine graphs, not an older 704 MiB one;
`memory::SelectReclaim`, unit-tested).

Admission looks at memory only while its model is resident: a request
that arrives while its model is being made resident joins as before and
the cohort's capacity policy governs it.

## Provenance

- **Host**: `spark-b` (`spark-56f5`), GB10, driver 580.178.04; the
  admission gate (`jitllm-spark-preflight.py 105`) passed before every
  cell.
- **Builds**: `spark-native` of this change and of main `340e368` (the
  source of `fcf78c8` but for two documents), same session. The served
  cells ran this change before admission counted its peers' prompts (the
  first estimate, which never acted in them; the final rule needs no
  wait there either, with 7–37 GiB of state room).
- **Harnesses**: `jitllm_spec_runner --check wave --slots N` (in the
  tree); the HTTP cells' controller (scratch, `cells.py`, a fresh service
  a cell, streamed greedy requests, a one-token prime excluded).
- **Raw records**: `spark-b:~/scratch/slots/` (`m1/` waves, `h1/` cells,
  `h2/`–`h4/` memory and depth runs, `m2/` each width's wave times, `h5/`
  the wave-form cells, `h6/` the warm bursts); the review's in
  `spark-b:~/scratch/slots-review/` (`R3/admab` the matched admission
  A/B, `R*/victim*` the spared-branch reproduction). The *before* admission cells (`h2`)
  ran a build that also carried the depth-1 experiment (`h3`), which acts
  only from six requests decoding at once, never reached under that
  memory.
