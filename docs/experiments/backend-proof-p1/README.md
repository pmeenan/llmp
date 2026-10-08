<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Backend proof P1: BP-F1, host and device VMM against cudaMalloc — 2026-09-27

BP-F1 asks whether llmpalooza's GGML kernels run slower when their memory is
llmpalooza's VMM than when it is `cudaMalloc` memory: host VMM under rule v1
(D-034's reopen condition), device VMM under rule v2 (D-081's;
[backend-proof.md](../../backend-proof.md#performance-protocol-rule-approved-2026-09-26-bp-f2s-reference-pre-registered-at-p3-entry)).
The approved kernel-timing rule needs BP-F1's own noise calibration and
holdout before any comparison. This report records that calibration, which
settles BP-F1's rule under D-079, and then the gated comparison, run once
the pre-registration was reviewed and committed; then the same for rule v2.

**Results in brief.**

- **BP-F1 passes on device VMM (rule v2).** Under the committed
  pre-registration (`72c7c62`), no case fails the stage and the aggregate
  passes in both sessions (`t` 0.58 and −0.36, limit 3.143). In the
  primary session two cases went over `z`, which triggered the mirrored
  confirmation: KQV at 17 rows (d = 4.45) and the k/v projection at 17
  rows (d = 3.65). Neither did in the confirmation (2.2 and 2.0). Every
  case's ratio of arm medians is 0.957–1.037 across both sessions; the
  output head at one row takes 1,048.8 µs on `cudaMalloc` and 0.999× that on
  device VMM. Launches and outputs are identical in both memory kinds.
  D-081 stands ([comparison](#comparison-device-vmm-against-cudamalloc-bp-f1-rule-v2-gated)).
- **Rule v2 (device VMM, D-081) was pre-registered, and its holdout
  passed.** The harness gained a device-VMM memory kind, so the rerun D-081
  asks for needed a new calibration. Four A/A `cudaMalloc` sessions with
  the new binary give a median `σ` of 1.17% (0.15–3.15%). Both holdout
  sessions pass alone, with no case over `z` and aggregate `t` 0.03 and
  −0.59. backend-proof.md registers the harness (`0c191da7…`), the
  unchanged case file and the calibration (`567cb849…`) as "BP-F1 v2"
  ([rule v2](#rule-v2-device-vmm-d-081)).

- **BP-F1 fails: host VMM is slower.** Under the pre-registered rule, 41
  of 53 cases fail both the primary and the mirrored confirmation session,
  and the aggregate fails in both (`t` 54.2 and 57.0 against 3.143).
  Every matrix product is slower, by 1.10× to 4.9×; the output head at one
  row runs 2.14× slower. The launches and outputs are identical in both
  memory kinds. Per the protocol this reopens D-034 for the owner and
  blocks nothing else ([comparison](#comparison-host-vmm-against-cudamalloc-bp-f1-gated)).
  Why, and the options: the [host-VMM diagnosis](../host-vmm-diagnosis/README.md);
  the owner's answer is D-081 (device VMM behind a landing-zone copy).

- **53 cases**, derived from the FP16 bridge's recorded plan for the
  held-out trajectory at 1, 16, 17 and 512 rows. Each is a kernel llmpalooza
  has: RMSNorm, fused RMSNorm-multiply, multiply, three kinds of add, five
  projections, and attention's KQ and KQV products. Every block verified
  every captured launch against the recorded plan.
- **Calibration** from four A/A `cudaMalloc` sessions on `spark` (`c1` to
  `c4`, two in each order): the median `σ` is 1.23%, from 0.10% (the output
  head at one row) to 2.75% (the k/v projection at one row). With
  `z = 3.555` for 53 cases, the per-case thresholds `z · σ · √½` are below
  2% for 13 cases and below 5% for 45; the largest is 6.9%.
- **The holdout passes.** It was declared in advance to reject the rule if
  either holdout session, taken as the primary with the other as its
  confirmation, failed the stage. `h1` (primary order) passes alone:
  no case over `z`, aggregate `t` 1.87. `h2` (mirrored) as the primary has
  one case over `z` (fused RMSNorm-multiply at 16 rows, d = 3.73), which
  `h1` clears as its confirmation; aggregate `t` −2.48 and 1.87. The stage
  passes both ways.
- **Every in-sample pairing** of the calibration sessions passes; those
  with `c3` as the primary need a confirmation (16-row 128-wide bias add,
  d = 3.68).
- **Power** for a small slowdown of one case is lower than BP-F2's,
  because these kernels are small: slowed by 2% in both sessions, a case
  is detected 21–28% of the time (BP-F2: 48–54%); at 5%, 74–83%; at 10%,
  98–100%. A slowdown of every case, or of every matrix
  product, fails the stage from 0.5%.
- **The calibration file** [`bpf1-calibration.json`](bpf1-calibration.json)
  has SHA-256
  `aa1581271357e8c1bfeed2b8da98ae98ee35031b4160df6500cb8b6e991b1369`,
  pre-registered in backend-proof.md with the harness's and the case
  file's. The session driver refuses a host-VMM arm under any other
  calibration, harness or case file.

## Conditions and provenance

- **Host.** `spark` (`spark-c4e2`): GB10, kernel 7.0.0-1019-nvidia, driver
  580.178.04, persistence mode on, application clock 2,418 MHz, CPU
  governor `performance`. The clock policy and idle states were left
  unchanged. Before each session the driver checked that no compute
  process was on the GPU and the load average was below 1.0; from `c3` on,
  a wrapper also waited for it to fall below 0.5 (our own previous
  session's decay). The
  driver checked for other compute processes again before every block; none
  appeared. Between block boundaries the SM clock stayed at 2,405–2,437 MHz
  and the GPU at 49–63 °C, with no active throttle reason.
- **Sessions.** 2026-09-27, 05:27–05:51 UTC: `c1` (primary order), `c2`
  (mirrored), `c3` (primary), `c4` (mirrored), then `h1` (primary) and `h2`
  (mirrored). Each took about two minutes: three discarded warm-up
  processes and eight timed blocks. A first `c3` attempt stopped before
  running anything because the load average (1.04) was above the limit; its
  empty directory was removed.
- **Harness.** `llmp_ggml_vmm_bench` from the `cross` preset, SHA-256
  `05348df868e83768a441302bc2831df8ebbe4fa609a13c5b5cd874d9048f3b92`,
  copied read-only beside the pinned cuBLAS 13.8.0.4 (`libcublas.so.13`
  `ee7c1657…`, `libcublasLt.so.13` `ba3b942f…`, the libraries the FP16 plan
  was recorded with). That copy, kept with the raw sessions, is the binary
  the comparison runs: a rebuild of the same source reproduces it only at
  the same path, since it embeds its source paths. The case file
  [`bpf1-cases.txt`](bpf1-cases.txt) is `fe78d033…`; the session driver
  `bpf1_session.py` was `846f41b3…` (the current one adds only the
  host-VMM check of the harness and case file). Sources: commit `7a4b7688` plus the
  uncommitted BP-F1 work, SHA-256 of `git diff --binary HEAD` followed by
  `sha256sum` of each untracked file `f89a3b70…`. Each manifest records
  all of these.
- **Rule.** `timing_protocol.py` at `c05fd2dd…` (the approved rule, the
  copy on `spark` from P0) computed the calibration, the outcomes and the
  power. The current script gives the same calibration byte for byte and
  the same outcomes.
- **Aggregates.** [`bpf1-timing.json`](bpf1-timing.json) holds every
  session's manifest (host, GPU settings, clocks and temperatures at block
  boundaries, identities) and each case's block medians for both arms,
  graph and stream-launched; the in-sample pairings, the holdout, and the
  power estimates. Raw samples and logs stay outside Git, on `spark` in
  `~/.local/share/llmp/bpf1-20260927/`.

## Cases

[`bpf1_cases.py`](bpf1_cases.py) derives the cases from
[`fp16-plan.json`](../backend-proof-p0/fp16-plan.json), the bridge's
recorded executed plan for the held-out trajectory. It flattens each chunk
size's sequence, cuts it into operations (a cuBLAS product is its
conversions, pointer setup, memset and GEMM, and its output conversion when
it computes in F16), and labels them by position in the layer. Operations
of the same shape (q and o, k and v, gate and up, the three norms, the
two residual adds) must have launched identically in every layer; each
becomes one case, whose launches are the recorded ones. At one row a
residual add has the bias add's operands exactly, so they are one case.
The unfused arm gives the plain operations; the fused arm gives the fused
RMSNorm-multiply.

| Operation | Shape | 1 row | 16 rows | 17 rows | 512 rows |
| --- | --- | --- | --- | --- | --- |
| `rms_norm`, `mul`, `rms_norm_mul` | 896 wide | GGML | GGML | GGML | GGML |
| `add.bias_896`, `add.bias_128` | bias broadcast over rows | GGML | GGML | GGML | GGML |
| `add.residual` | 896 × rows, both operands | (= bias add) | GGML | GGML | GGML |
| `linear.q_o` | 896 → 896 | MMVF | MMF | cuBLAS | cuBLAS |
| `linear.k_v` | 896 → 128 | MMVF | MMF | cuBLAS (split-K) | cuBLAS |
| `linear.gate_up` | 896 → 4,864 | MMVF | MMF | cuBLAS | cuBLAS |
| `linear.down` | 4,864 → 896 | MMVF | MMF | cuBLAS | cuBLAS |
| `linear.lm_head` | 896 → 151,936 | MMVF | MMF | cuBLAS | cuBLAS |
| `attn.kq` | 64 × 14 heads over 2 KV heads, cache 1,024 | MMVF (256 cells), MMF (768) | MMF (256) | cuBLAS TF32, pointer array (256) | cuBLAS (768) |
| `attn.kqv` | the same, V transposed | MMVF (256 and 768) | MMF (256) | cuBLAS, pointer array (256) | cuBLAS (768) |

The KV cells in use are the trajectory's (`n_past + rows`, padded to 256)
at each sequence's first occurrence: 256 up to the 512-row chunk and 768
after it, so one row has both.

Not cases, because llmpalooza has no implementation of them yet: RoPE, softmax,
`set_rows`, `get_rows`, the copy after attention, the SiLU gate, and the
fused MMVF variants (bias, gate and residual fused into the product) of the
fused arm's single-row steps.

## Method

[`../../../benchmarks/ggml_vmm_bench.cc`](../../../benchmarks/ggml_vmm_bench.cc)
runs one block: every case in one process, all its memory of one kind. The
session driver [`bpf1_session.py`](bpf1_session.py) runs the blocks.

- **Operand placement.** In a block, every buffer the case's kernels are
  given is in the block's memory kind: weights, activations and outputs,
  GGML's scratch pool (sized by the plan) and the cuBLAS workspace (32 MiB,
  upstream's). For `cudaMalloc` each is a `cudaMalloc` allocation; for host
  VMM, a host-backed VMM reservation from llmpalooza's provider, mapped
  read-write. Only a staging buffer for setup (host VMM in both kinds) is
  outside; no timed kernel touches it.
- **Rotation.** Each case has a ring of operand sets, each holding all its
  tensors, with more sets than fit four times into the queried L2 (24 MiB
  on GB10), and at least two. Invocation `k` uses set `k mod N`, so no
  invocation reads what the one before it used, and a set is read again
  only after the ring's other sets, which with it total more than four
  times L2. The graph arm uses sets 0 to 359 and the stream arm the next
  360, so the rings of more than 720 sets (the small operations) are only
  partly read. The sets
  hold identical data. N ranges from 2 (the output head) to 65,537
  (the 128-wide bias add at one row).
- **Launch verification.** Each sample's graph is captured separately (36
  graphs, ten invocations each, on consecutive sets) and read back through
  the driver's graph API. Every graph's launches must equal the recorded
  plan's, ten times over, in order: kernel name (with NVCC's per-file
  internal-namespace tag normalized), grid, block, static plus dynamic
  shared memory and registers; or a memset's size and value. A mismatch
  stops the block before anything is timed. All 53 cases matched in every
  block.
- **Samples.** Five warm replays, then 31 samples, each one graph replay
  bracketed by two captured events, the interval divided by ten. The
  stream-launched arm follows: the same through the launch context on the
  stream, on the next 360 sets of the ring, reported beside and not gated.
- **Outputs.** Set 0's output is hashed after an eager invocation and again
  after both arms; the two must match, and `bpf1_stats.py` requires one
  hash per case across all eight blocks of a session.
- **Session.** A discarded warm-up process per arm, one more immediately
  before the first timed block, then eight blocks in the primary order
  A1 B1 B2 A2 B3 A3 A4 B4 or the mirrored B1 A1 A2 B2 A3 B3 B4 A4. Here A
  and B are both `cudaMalloc`.
- **Statistics.** [`bpf1_stats.py`](bpf1_stats.py) summarizes a session in
  the schema `timing_protocol.py` reads: per case the ratio of the arms'
  medians over their 124 samples and each block's median.
  [`bpf1_record.py`](bpf1_record.py) condenses the sessions and computes the
  outcomes.

## Calibration

`σ` per case is the relative standard deviation of the eight block medians
within a session, pooled over `c1`–`c4` (32 process medians per case).

| Kernels | Cases | Median `σ` | Largest `σ` |
| --- | --- | --- | --- |
| Elementwise and norms | 23 | 1.50% | 2.51% (`mul`, 1 row) |
| Projections | 20 | 1.11% | 2.75% (`linear.k_v`, 1 row) |
| Attention products | 10 | 0.75% | 2.15% |
| All | 53 | 1.23% | 2.75% |

The quietest cases are the long ones: the output head (0.10–0.43%) and the
single-row KQV at 256 cells (0.14%). The noisiest are the one- to
two-microsecond kernels at 1, 16 and 17 rows. Across the six sessions the
A/A ratios of arm medians spanned 0.962–1.046.

Whole processes run slightly fast or slow, as P0 found for EXL3: a sign
test over the cases' ratios rejects at 1% in two of the six A/A sessions
(`c1`, 34 slower and 14 faster; `h2`, 13 slower and 38 faster). That is what
the rule's block-level aggregate test is for, and its `t` stayed within
−2.48 to 2.39.

## Holdout

Declared before `h1` and `h2` ran, and recorded in their manifests: the rule
is rejected if either holdout session, taken as the primary with the other
as its confirmation, fails the stage.

| Primary | Confirmation | Over `z` in the primary | Aggregate `t` | Stage |
| --- | --- | --- | --- | --- |
| `h1` | (not needed) | none | 1.87 | passes |
| `h2` | `h1` | `rms_norm_mul` at 16 rows, d = 3.73 | −2.48, 1.87 | passes |

In sample, every ordered pairing of `c1`–`c4` passes; the three with `c3`
as the primary need a confirmation (`add.bias_128` at 16 rows, d = 3.68).

## Power

One case slowed in both sessions of a pair (`timing_protocol.py --power`):

| Slowdown | `h1` + `h2` | `c1` + `c2` | `c3` + `c4` |
| --- | --- | --- | --- |
| 2% | 21% | 28% | 25% |
| 3% | 38% | 43% | 38% |
| 5% | 74% | 83% | 75% |
| 10% | 100% | 98% | 100% |

A subset slowed together (the stage fails at or above):

| Subset | Cases | `h1` + `h2` | `c1` + `c2` | `c3` + `c4` |
| --- | --- | --- | --- | --- |
| All cases | 53 | 0.5% | 0.5% | 0.5% |
| Matrix products | 30 | 0.5% | 0.5% | 0.5% |
| Elementwise and norms | 23 | 2% | 1% | 2% |
| 512 rows | 13 | 0.5% | 1% | 0.5% |
| Noisiest quarter | 14 | not at 3% | 3% | 3% |

A host-VMM penalty, if there is one, would be expected across many cases
at once, where the aggregate and the per-case tests together are
sensitive.

## Comparison: host VMM against `cudaMalloc` (BP-F1, gated)

Run 2026-09-27 on `spark` after the pre-registration was reviewed and
committed (`d3b4f2a`), exactly under it: the registered harness
(`05348df8…`), case file (`fe78d033…`) and calibration (`aa158127…`), which
the session driver checked against backend-proof.md before running and
each session's manifest records; every host-VMM block also records the
calibration's. A is `cudaMalloc` (the reference), B host VMM (the
candidate).

**Outcome: BP-F1 fails.** 41 of the 53 cases fail both sessions, and the
aggregate fails in both (`t` 54.16 and 56.98, limit 3.143). Under the
protocol this reopens D-034 for the owner; it blocks no other stage.

| Session | Order | When (UTC) | Cases over `z` | Aggregate `t` | Candidate slower / faster |
| --- | --- | --- | --- | --- | --- |
| `p1` (primary) | A1 B1 B2 A2 B3 A3 A4 B4 | 06:30–06:33 | 41 | 54.16 | 45 / 8 |
| `m1` (confirmation) | B1 A1 A2 B2 A3 B3 B4 A4 | 06:34–06:37 | 41 | 56.98 | 45 / 8 |

The primary failed (41 cases and the aggregate), so the mirrored
confirmation ran, as the procedure requires; the same 41 cases exceeded
`z` in both. The block-level shifts put every host-VMM block 13.5–16.7%
above the per-case mean and every `cudaMalloc` block 14.5–16.2% below it,
in both sessions. `timing_protocol.py` at `c05fd2dd…` applied the rule; the current
script gives the same outcome. Nothing in the rule was ambiguous in
applying it.

- **Conditions.** Both sessions started with no compute process on the
  GPU and the load average below 0.5 (0.14 and 0.43); no other compute
  process appeared before any block. SM clock 2,405–2,457 MHz and
  49–60 °C at block boundaries, no active throttle reason; driver
  580.178.04, kernel 7.0.0-1019-nvidia, CPU governor `performance`.
  Source: commit `d3b4f2a` with no uncommitted changes; the session driver
  was the committed `bpf1_session.py` (`40d5daa5…`).
- **The validity checks held in every block of both arms.** Every
  captured launch matched the recorded plan on host VMM as on `cudaMalloc`,
  and each case's output hash was the same in all 16 blocks: host VMM
  changed the time, not the kernels or the results.
- **Where it is slower.** Every matrix product fails, from 1.10× (the
  output head at 17 rows) to 4.9× (single-row KQV at 768 cells). The
  output head at one row, which reads its 272 MB of weights once per
  invocation, takes 2,184 µs against 1,019 µs: about 125 GB/s of weights
  read against 267 GB/s. The multiply, the fused RMSNorm-multiply and the
  896-wide bias add fail at 16 rows and above, and every elementwise
  kernel but the residual add fails at 512 rows.
- **Where it is not.** The 12 passing cases are elementwise and norm
  kernels at 1, 16 or 17 rows, and the residual add at 512 rows; several
  ran faster on host VMM (fused RMSNorm-multiply at one row, 0.86×). The
  rule is one-sided, so they pass.
- **The stream-launched arm** (reported, not gated) also has host VMM
  slower for every matrix product (1.04× to 6.0×); for the adds and the
  multiply at 16 and 17 rows its ratios are closer to 1 than the graph
  arm's, and for RMSNorm there they are further from it (1.10–1.12×).

Per case: the reference's median (µs, the median of its four block
medians in `p1`), each session's ratio of arm medians and its `d`
(threshold `z` = 3.555), the stream-launched ratios, and the stage verdict.

| Case | Rows | `cudaMalloc` µs | `p1` ratio | `p1` d | `m1` ratio | `m1` d | Stream `p1` / `m1` | Stage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `add.bias_128` | 1 | 1.89 | 0.971 | -3.0 | 0.988 | -1.3 | 1.03 / 1.03 | passes |
| `add.bias_896` | 1 | 2.00 | 0.905 | -5.4 | 0.895 | -6.0 | 1.03 / 1.03 | passes |
| `attn.kq.kv256` | 1 | 5.52 | 4.450 | 964.3 | 4.396 | 949.1 | 5.00 / 4.97 | **fails** |
| `attn.kq.kv768` | 1 | 3.56 | 1.436 | 50.1 | 1.421 | 48.4 | 1.48 / 1.48 | **fails** |
| `attn.kqv.kv256` | 1 | 3.47 | 4.772 | 3739.3 | 4.740 | 3707.6 | 6.03 / 6.03 | **fails** |
| `attn.kqv.kv768` | 1 | 4.90 | 4.900 | 702.2 | 4.834 | 690.3 | 5.36 / 5.24 | **fails** |
| `linear.down` | 1 | 35.69 | 1.268 | 34.7 | 1.294 | 38.0 | 1.28 / 1.29 | **fails** |
| `linear.gate_up` | 1 | 34.40 | 2.265 | 176.1 | 2.286 | 178.9 | 2.30 / 2.34 | **fails** |
| `linear.k_v` | 1 | 2.71 | 2.362 | 69.9 | 2.344 | 69.0 | 2.97 / 2.97 | **fails** |
| `linear.lm_head` | 1 | 1,018.80 | 2.143 | 1598.1 | 2.132 | 1582.6 | 2.15 / 2.13 | **fails** |
| `linear.q_o` | 1 | 8.16 | 2.589 | 163.5 | 2.530 | 157.4 | 3.05 / 3.03 | **fails** |
| `mul` | 1 | 1.92 | 0.946 | -3.0 | 0.942 | -3.3 | 1.00 / 1.00 | passes |
| `rms_norm` | 1 | 2.69 | 0.919 | -6.0 | 0.916 | -6.3 | 0.96 / 0.92 | passes |
| `rms_norm_mul` | 1 | 4.29 | 0.862 | -28.6 | 0.859 | -29.4 | 0.90 / 0.90 | passes |
| `add.bias_128` | 16 | 2.03 | 1.011 | 0.9 | 1.009 | 0.8 | 1.00 / 1.00 | passes |
| `add.bias_896` | 16 | 2.23 | 1.227 | 14.4 | 1.225 | 14.2 | 1.05 / 1.07 | **fails** |
| `add.residual` | 16 | 2.38 | 0.963 | -2.6 | 0.958 | -3.0 | 1.01 / 1.00 | passes |
| `attn.kq.kv256` | 16 | 3.83 | 1.667 | 57.0 | 1.639 | 54.6 | 1.44 / 1.47 | **fails** |
| `attn.kqv.kv256` | 16 | 4.91 | 1.379 | 133.5 | 1.379 | 133.4 | 1.31 / 1.31 | **fails** |
| `linear.down` | 16 | 40.96 | 2.109 | 138.1 | 2.070 | 133.2 | 2.08 / 2.04 | **fails** |
| `linear.gate_up` | 16 | 39.51 | 1.814 | 83.0 | 1.850 | 86.6 | 1.76 / 1.81 | **fails** |
| `linear.k_v` | 16 | 5.40 | 1.137 | 13.7 | 1.136 | 13.6 | 1.04 / 1.05 | **fails** |
| `linear.lm_head` | 16 | 1,128.05 | 1.289 | 95.5 | 1.297 | 98.2 | 1.29 / 1.30 | **fails** |
| `linear.q_o` | 16 | 9.47 | 1.913 | 123.3 | 1.910 | 122.8 | 1.87 / 1.86 | **fails** |
| `mul` | 16 | 2.24 | 1.272 | 25.6 | 1.272 | 25.6 | 1.08 / 1.08 | **fails** |
| `rms_norm` | 16 | 2.94 | 1.013 | 1.5 | 1.030 | 3.4 | 1.10 / 1.11 | passes |
| `rms_norm_mul` | 16 | 4.17 | 1.174 | 29.8 | 1.174 | 29.8 | 1.17 / 1.18 | **fails** |
| `add.bias_128` | 17 | 1.97 | 1.043 | 2.8 | 1.018 | 1.2 | 0.99 / 1.03 | passes |
| `add.bias_896` | 17 | 2.24 | 1.280 | 24.8 | 1.279 | 24.7 | 1.08 / 1.11 | **fails** |
| `add.residual` | 17 | 2.44 | 0.944 | -13.6 | 0.955 | -10.9 | 1.00 / 1.00 | passes |
| `attn.kq.kv256` | 17 | 7.50 | 1.710 | 46.6 | 1.718 | 47.1 | 1.53 / 1.53 | **fails** |
| `attn.kqv.kv256` | 17 | 9.10 | 1.606 | 60.5 | 1.595 | 59.4 | 1.49 / 1.48 | **fails** |
| `linear.down` | 17 | 43.25 | 1.369 | 44.5 | 1.393 | 47.3 | 1.39 / 1.41 | **fails** |
| `linear.gate_up` | 17 | 45.26 | 2.289 | 133.1 | 2.326 | 136.8 | 2.18 / 2.23 | **fails** |
| `linear.k_v` | 17 | 8.14 | 1.396 | 64.8 | 1.415 | 67.9 | 1.23 / 1.24 | **fails** |
| `linear.lm_head` | 17 | 1,289.05 | 1.109 | 19.5 | 1.103 | 18.5 | 1.11 / 1.10 | **fails** |
| `linear.q_o` | 17 | 12.85 | 1.354 | 42.2 | 1.379 | 45.2 | 1.41 / 1.44 | **fails** |
| `mul` | 17 | 2.25 | 1.278 | 26.7 | 1.273 | 26.2 | 1.21 / 1.12 | **fails** |
| `rms_norm` | 17 | 3.01 | 1.024 | 2.3 | 1.002 | 0.2 | 1.12 / 1.11 | passes |
| `rms_norm_mul` | 17 | 4.16 | 1.183 | 29.4 | 1.180 | 29.0 | 1.20 / 1.20 | **fails** |
| `add.bias_128` | 512 | 3.75 | 1.528 | 34.0 | 1.536 | 34.5 | 1.49 / 1.52 | **fails** |
| `add.bias_896` | 512 | 16.41 | 1.158 | 17.1 | 1.156 | 16.9 | 1.15 / 1.16 | **fails** |
| `add.residual` | 512 | 23.13 | 0.953 | -5.0 | 0.965 | -3.8 | 0.94 / 0.95 | passes |
| `attn.kq.kv768` | 512 | 124.34 | 1.680 | 186.9 | 1.694 | 190.6 | 1.65 / 1.66 | **fails** |
| `attn.kqv.kv768` | 512 | 186.25 | 1.624 | 123.8 | 1.651 | 129.3 | 1.60 / 1.60 | **fails** |
| `linear.down` | 512 | 122.16 | 3.715 | 212.5 | 3.687 | 210.3 | 3.64 / 3.61 | **fails** |
| `linear.gate_up` | 512 | 139.95 | 2.679 | 342.4 | 2.752 | 357.4 | 2.62 / 2.65 | **fails** |
| `linear.k_v` | 512 | 17.27 | 2.318 | 207.0 | 2.336 | 209.8 | 2.12 / 2.09 | **fails** |
| `linear.lm_head` | 512 | 4,068.76 | 2.884 | 1538.6 | 2.895 | 1548.0 | 2.88 / 2.89 | **fails** |
| `linear.q_o` | 512 | 34.05 | 2.962 | 235.7 | 2.966 | 236.2 | 2.86 / 2.87 | **fails** |
| `mul` | 512 | 16.37 | 1.187 | 19.8 | 1.143 | 15.2 | 1.18 / 1.14 | **fails** |
| `rms_norm` | 512 | 16.77 | 1.087 | 7.3 | 1.080 | 6.7 | 1.07 / 1.07 | **fails** |
| `rms_norm_mul` | 512 | 17.54 | 1.290 | 41.6 | 1.317 | 45.5 | 1.28 / 1.29 | **fails** |

[`bpf1-comparison.json`](bpf1-comparison.json) holds both sessions'
manifests, block medians and `d`, and the rule's outcome, written by
[`bpf1_compare_record.py`](bpf1_compare_record.py). The raw samples and
logs stay on `spark` in `~/.local/share/llmp/bpf1-20260927/p1` and `m1`.

What the comparison does not show: why host VMM is slower here when
D-034's scan kernels read host and device VMM at the same bandwidth
(242 GB/s), and whether placing only the weights, or only the activations
and workspaces, on host VMM changes the result. The rule placed every
buffer in the candidate's memory; separating them is for the owner's
D-034 review, not part of this gate.

## Rule v2: device VMM (D-081)

D-081 moved weights and state to device-located VMM and asked for BP-F1 to
be rerun against it. Rule v1 stays as history: its harness and calibration
are for host VMM. Rule v2 is v1 with two changes: the candidate's memory
kind, and the harness binary. So v2 needed a new calibration and holdout,
pre-registered in backend-proof.md before any device-VMM session ran on
`spark`. The cases (`fe78d033…`), the statistic, `z = 3.555`, the
aggregate limit, the declared holdout and the confirmation procedure are
v1's, unchanged.

- **Placement.** The harness gains a `device-vmm` memory kind: every
  buffer the kernels are given (weights, activations, outputs, GGML's
  scratch and the cuBLAS workspace) is a device-located reservation from
  llmpalooza's CUDA provider, mapped read-write for the device only, where
  D-081 places weights and state. The setup staging buffer stays host VMM,
  as in v1, and no timed kernel touches it. The `cudaMalloc` arm is
  unchanged. A device-VMM block, like a host-VMM one, refuses to run
  without a calibration hash.
- **Harness.** `llmp_ggml_vmm_bench` built by the `spark-native` preset
  on `spark-b`, SHA-256
  `0c191da79557793ee779e2cac3de241072e83f052224d1f037c7adef596d5d8c`,
  copied read-only to `spark` beside the same cuBLAS 13.8.0.4 as v1
  (`ee7c1657…`, `ba3b942f…`). Like v1's, the binary embeds its source
  paths, so the comparison runs this copy. Sources: commit `961cc09b`
  plus the uncommitted v2 changes, identity `06409b28…` by the formula
  under [Reproduction](#reproduction); session driver `bpf1_session.py`
  `3156c2bf…`.
- **Registration.** backend-proof.md records the harness, the case file
  and [`bpf1-v2-calibration.json`](bpf1-v2-calibration.json) (SHA-256
  `567cb8494dbb36022be6ba64fb185be272561bf7e7f89680c3413931f93bb3bc`) as
  "BP-F1 v2 …". The session driver runs a device-VMM arm only if all three
  match the v2 registration, and a host-VMM arm only if they match v1's.
  `bpf1_stats.py` checks both again. [`bpf1-v2-timing.json`](bpf1-v2-timing.json)
  records every v2 session as `bpf1-timing.json` does for v1.
- **Before registration.** A development run on `spark-b` put all 53
  cases through one device-VMM block and one `cudaMalloc` block. It
  confirmed that device VMM launches the recorded kernels with identical
  outputs. It also timed them, on a GPU another agent was using, so its
  times are not evidence. The host-VMM diagnosis had already measured
  device VMM at 0.98–1.02× of `cudaMalloc`
  ([report](../host-vmm-diagnosis/README.md)). Neither result could shape
  the rule: everything except `σ` is v1's, and `σ` comes mechanically from
  the A/A `cudaMalloc` sessions.

### Conditions

`spark` (`spark-c4e2`), 2026-09-27, 23:00–23:17 UTC: `c1` (primary
order), `c2` (mirrored), `c3` (primary), `c4` (mirrored), then `h1`
(primary) and `h2` (mirrored). Each took about two minutes. The
declaration (the rule is rejected if either holdout session, taken as the
primary with the other as its confirmation, fails the stage) is in all six
manifests, so it was written before `c1` ran.

- **Host and GPU.** GB10, kernel 7.0.0-1019-nvidia, driver 580.178.04,
  application clock 2,418 MHz, CPU governor `performance`. At block
  boundaries the SM clock was 2,405–2,431 MHz and the GPU 51–63 °C, with
  no active throttle reason.
- **Idle checks.** Before each session a wrapper waited until three things
  held: no other measurement script or benchmark was running, no compute
  process was on the GPU, and the load average was below 0.5. The load
  average at start was 0.24–0.40. The driver checked again for compute
  processes before every block, and found none.
- **The other agent's run.** When the sessions were first queued, another
  agent's page-in timing session was running on `spark`. The first
  wrapper, which checked only the GPU and the load, was stopped before any
  session began; the wrapper above replaced it. The page-in run's last
  result is at 23:00:00, and its processes were gone when `c1` started at
  23:00:44.

### Calibration

`σ` per case is the relative standard deviation of the eight block medians
within a session, pooled over `c1`–`c4` (as for v1).

| Kernels | Cases | Median `σ` | Largest `σ` |
| --- | --- | --- | --- |
| Elementwise and norms | 23 | 1.21% | 2.91% (`mul`, 1 row) |
| Projections | 20 | 1.31% | 3.15% (`linear.k_v`, 1 row) |
| Attention products | 10 | 0.74% | 1.60% (`attn.kq`, 17 rows) |
| All | 53 | 1.17% (v1: 1.23%) | 3.15% |

The quietest cases are the output head at 16 and 1 rows (0.15%, 0.18%)
and the single-row KQ at 256 cells (0.20%). The per-case thresholds
`z · σ · √½` are 0.37–7.9%: below 2% for 18 cases and below 5% for 47.
Across the six sessions the A/A ratios of arm medians spanned 0.957–1.039.
No session's sign test rejects at 1%: the smallest p is 0.011 (`c2`, 16
cases slower, 35 faster). The aggregate `t` stayed within −0.97 to 0.41.
Every ordered pairing of `c1`–`c4` passes without a confirmation; the
largest in-sample `d` is 2.83 (`linear.q_o` at one row, `c3`).

### Holdout

| Primary | Confirmation | Over `z` in the primary (largest `d`) | Aggregate `t` | Stage |
| --- | --- | --- | --- | --- |
| `h1` | (not needed) | none (`mul` at 1 row, 1.92) | 0.03 | passes |
| `h2` | (not needed) | none (`attn.kqv` at 16 rows, 0.94) | −0.59 | passes |

Rule v2 stands. `timing_protocol.py` at `c05fd2dd…`, the approved rule,
computed the calibration and outcomes. The current script (`b17a2e84…`)
gives the same calibration byte for byte and the same holdout outcomes.

### Power

One case slowed in both sessions of a pair:

| Slowdown | `h1` + `h2` | `c1` + `c2` | `c3` + `c4` |
| --- | --- | --- | --- |
| 2% | 25% | 25% | 28% |
| 3% | 45% | 47% | 47% |
| 5% | 85% | 81% | 81% |
| 10% | 94% | 98% | 98% |

A subset slowed together (the stage fails at or above):

| Subset | Cases | `h1` + `h2` | `c1` + `c2` | `c3` + `c4` |
| --- | --- | --- | --- | --- |
| All cases | 53 | 0.5% | 0.5% | 0.5% |
| Matrix products | 30 | 0.5% | 0.5% | 0.5% |
| Elementwise and norms | 23 | 1% | 2% | 2% |
| 512 rows | 13 | 1% | 1% | 1% |
| Noisiest quarter | 14 | not at 3% | not at 3% | not at 3% |

The raw sessions and the harness copy stay on `spark` in
`~/.local/share/llmp/bpf1v2-20260927/`.

## Comparison: device VMM against `cudaMalloc` (BP-F1 rule v2, gated)

Run 2026-09-27 on `spark` after the pre-registration was reviewed and
committed (`72c7c62`), exactly under it. The session driver checked the
registered harness (`0c191da7…`), case file (`fe78d033…`) and calibration
(`567cb849…`) against the committed backend-proof.md before running; each
session's manifest and every device-VMM block record them. A is
`cudaMalloc` (the reference), B device VMM (the candidate).

**Outcome: BP-F1 passes.** No case fails both sessions, and the aggregate
passes in both (`t` 0.58 and −0.36, limit 3.143). D-081 stands.

| Session | Order | When (UTC) | Cases over `z` | Aggregate `t` | Candidate slower / faster |
| --- | --- | --- | --- | --- | --- |
| `p1` (primary) | A1 B1 B2 A2 B3 A3 A4 B4 | 23:47–23:49 | 2 | 0.58 | 28 / 22 |
| `m1` (confirmation) | B1 A1 A2 B2 A3 B3 B4 A4 | 23:51–23:53 | 0 | −0.36 | 17 / 34 |

The primary had two cases over `z`, so the mirrored confirmation ran, as
the procedure requires: `attn.kqv.kv256` at 17 rows (ratio 1.026, d =
4.45) and `linear.k_v` at 17 rows (1.027, d = 3.65). In `m1` they were
1.013 (d = 2.2) and 1.015 (d = 2.0), under `z`, so neither fails the
stage. Both are short kernels (about 9 and 8 µs) with calibrated `σ` of
0.81% and 1.03%, so their thresholds are 2.0% and 2.6%. In the six A/A
sessions of the calibration and holdout no case reached `z` (largest d
2.83); here two did in one session and none in the other.
`timing_protocol.py`
at `c05fd2dd…` applied the rule; the current script gives the same
outcome. Nothing in the rule was ambiguous in applying it, and no session
was set aside.

- **Conditions.** The sessions were queued behind another agent's SSD
  sweep on `spark`, which the owner cancelled at 23:46:41 UTC. Each
  session started only after the sweep's results file had ended with its
  `done` line, no `fio`, `dd`, sweep or llmpalooza benchmark process was
  running, no compute process was on the GPU, and the load average was
  below 0.5, at two checks 30 s apart. The load average was 0.27 and
  0.20 at the starts, and 0.46–0.90 at block boundaries (the sessions'
  own processes). No other compute process appeared before any block.
  SM clock 2,411–2,437 MHz and 50–60 °C at block boundaries, with no
  active throttle reason; application clock 2,418 MHz, driver 580.178.04,
  kernel 7.0.0-1019-nvidia, CPU governor `performance`. Source: commit
  `72c7c62` with no uncommitted changes; the session driver was the
  committed `bpf1_session.py` (`3156c2bf…`), which is also the copy the
  calibration ran.
- **The validity checks held in every block of both arms.** Every
  captured launch matched the recorded plan on device VMM as on
  `cudaMalloc`, and each case's output hash was the same in all 16 blocks.
- **The ratios.** Every case's ratio of arm medians is 0.957–1.037
  across both sessions, as in the six A/A sessions of the calibration
  and holdout (0.957–1.039). The
  cases host VMM slowed most are at 0.999–1.000 here: KQV at one row and
  768 cells (4.90× on host VMM), the output head at one row (2.14×) and
  at 512 rows (2.88×). The stream-launched arm (reported, not gated) is
  0.92–1.09, its extremes the 17-row multiply (0.92 in `p1`, 1.09 in
  `m1`).

Per case: the reference's median (µs, the median of its four block
medians in `p1`), each session's ratio of arm medians and its `d`
(threshold `z` = 3.555), the stream-launched ratios, and the stage verdict.

| Case | Rows | `cudaMalloc` µs | `p1` ratio | `p1` d | `m1` ratio | `m1` d | Stream `p1` / `m1` | Stage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `add.bias_128` | 1 | 1.84 | 1.002 | 0.2 | 0.997 | -0.4 | 1.00 / 1.00 | passes |
| `add.bias_896` | 1 | 2.03 | 0.957 | -2.8 | 1.016 | 1.0 | 1.00 / 1.00 | passes |
| `attn.kq.kv256` | 1 | 5.52 | 1.001 | 0.8 | 0.999 | -0.4 | 0.99 / 1.00 | passes |
| `attn.kq.kv768` | 1 | 3.68 | 0.999 | -0.4 | 1.001 | 0.2 | 1.00 / 1.01 | passes |
| `attn.kqv.kv256` | 1 | 3.48 | 0.999 | -0.3 | 0.998 | -0.5 | 1.00 / 1.00 | passes |
| `attn.kqv.kv768` | 1 | 4.91 | 0.999 | -0.2 | 0.999 | -0.2 | 1.00 / 1.00 | passes |
| `linear.down` | 1 | 36.49 | 1.003 | 0.3 | 0.996 | -0.4 | 1.00 / 1.00 | passes |
| `linear.gate_up` | 1 | 34.92 | 1.005 | 0.5 | 0.988 | -1.1 | 1.00 / 1.00 | passes |
| `linear.k_v` | 1 | 2.80 | 0.993 | -0.3 | 0.971 | -1.3 | 1.01 / 1.00 | passes |
| `linear.lm_head` | 1 | 1,048.80 | 0.999 | -0.5 | 1.000 | 0.2 | 1.00 / 1.00 | passes |
| `linear.q_o` | 1 | 8.25 | 1.004 | 0.5 | 0.998 | -0.2 | 1.01 / 1.00 | passes |
| `mul` | 1 | 1.94 | 0.995 | -0.2 | 1.022 | 1.1 | 0.99 / 1.00 | passes |
| `rms_norm` | 1 | 2.68 | 1.000 | 0.0 | 0.998 | -0.6 | 0.97 / 1.00 | passes |
| `rms_norm_mul` | 1 | 4.29 | 0.999 | -0.3 | 1.001 | 0.3 | 1.00 / 1.00 | passes |
| `add.bias_128` | 16 | 2.03 | 1.002 | 0.1 | 1.037 | 2.3 | 1.00 / 1.03 | passes |
| `add.bias_896` | 16 | 2.25 | 0.996 | -0.6 | 0.998 | -0.3 | 0.97 / 1.00 | passes |
| `add.residual` | 16 | 2.44 | 1.005 | 0.4 | 0.996 | -0.3 | 0.98 / 0.98 | passes |
| `attn.kq.kv256` | 16 | 3.89 | 0.995 | -0.5 | 0.988 | -1.1 | 1.00 / 0.99 | passes |
| `attn.kqv.kv256` | 16 | 4.97 | 1.005 | 0.4 | 0.987 | -1.2 | 1.00 / 1.00 | passes |
| `linear.down` | 16 | 41.11 | 1.004 | 0.4 | 1.001 | 0.1 | 1.00 / 0.99 | passes |
| `linear.gate_up` | 16 | 39.79 | 1.000 | 0.0 | 0.982 | -1.8 | 1.00 / 0.99 | passes |
| `linear.k_v` | 16 | 5.51 | 1.000 | 0.0 | 0.998 | -0.4 | 1.00 / 1.00 | passes |
| `linear.lm_head` | 16 | 1,184.75 | 0.999 | -1.0 | 0.999 | -1.2 | 1.00 / 1.00 | passes |
| `linear.q_o` | 16 | 9.97 | 1.000 | 0.0 | 0.976 | -2.1 | 1.01 / 0.98 | passes |
| `mul` | 16 | 2.25 | 0.997 | -0.6 | 1.000 | 0.0 | 1.00 / 1.00 | passes |
| `rms_norm` | 16 | 3.00 | 1.004 | 0.5 | 1.017 | 1.8 | 1.00 / 1.00 | passes |
| `rms_norm_mul` | 16 | 4.18 | 1.000 | -0.1 | 0.999 | -0.3 | 1.00 / 1.00 | passes |
| `add.bias_128` | 17 | 1.97 | 1.018 | 1.2 | 1.003 | 0.2 | 1.02 / 1.00 | passes |
| `add.bias_896` | 17 | 2.25 | 1.002 | 0.3 | 0.997 | -0.4 | 0.99 / 0.98 | passes |
| `add.residual` | 17 | 2.46 | 1.000 | 0.0 | 1.000 | 0.0 | 1.00 / 1.00 | passes |
| `attn.kq.kv256` | 17 | 7.47 | 1.004 | 0.3 | 0.999 | -0.1 | 1.00 / 1.00 | passes |
| `attn.kqv.kv256` | 17 | 8.92 | 1.026 | 4.5 | 1.013 | 2.2 | 1.01 / 1.00 | passes (over `z` in `p1` only) |
| `linear.down` | 17 | 42.87 | 1.009 | 0.9 | 0.989 | -1.1 | 1.01 / 0.99 | passes |
| `linear.gate_up` | 17 | 44.81 | 1.016 | 1.8 | 0.990 | -1.1 | 1.01 / 1.00 | passes |
| `linear.k_v` | 17 | 8.08 | 1.027 | 3.6 | 1.015 | 2.0 | 1.03 / 0.95 | passes (over `z` in `p1` only) |
| `linear.lm_head` | 17 | 1,320.44 | 0.999 | -0.4 | 1.001 | 0.7 | 1.00 / 1.00 | passes |
| `linear.q_o` | 17 | 13.12 | 1.011 | 1.2 | 0.994 | -0.6 | 1.00 / 0.99 | passes |
| `mul` | 17 | 2.25 | 0.997 | -0.5 | 1.009 | 1.6 | 0.92 / 1.09 | passes |
| `rms_norm` | 17 | 3.05 | 1.002 | 0.3 | 0.998 | -0.3 | 1.03 / 1.00 | passes |
| `rms_norm_mul` | 17 | 4.15 | 1.001 | 0.2 | 0.996 | -0.6 | 1.00 / 1.00 | passes |
| `add.bias_128` | 512 | 4.08 | 1.006 | 0.6 | 0.995 | -0.4 | 1.00 / 1.00 | passes |
| `add.bias_896` | 512 | 17.06 | 0.998 | -0.2 | 0.983 | -1.9 | 0.99 / 0.98 | passes |
| `add.residual` | 512 | 23.88 | 0.989 | -0.8 | 0.995 | -0.4 | 1.01 / 0.98 | passes |
| `attn.kq.kv768` | 512 | 128.38 | 0.997 | -0.7 | 1.000 | 0.0 | 1.00 / 1.00 | passes |
| `attn.kqv.kv768` | 512 | 188.38 | 1.005 | 1.2 | 0.994 | -1.2 | 1.00 / 1.00 | passes |
| `linear.down` | 512 | 123.68 | 0.999 | -0.1 | 0.994 | -0.6 | 0.99 / 0.99 | passes |
| `linear.gate_up` | 512 | 138.50 | 1.012 | 2.8 | 1.002 | 0.4 | 1.01 / 1.00 | passes |
| `linear.k_v` | 512 | 17.62 | 0.991 | -1.8 | 1.003 | 0.5 | 1.00 / 1.00 | passes |
| `linear.lm_head` | 512 | 4,135.06 | 0.999 | -0.4 | 1.000 | 0.0 | 1.00 / 1.00 | passes |
| `linear.q_o` | 512 | 34.41 | 1.003 | 0.3 | 1.016 | 1.2 | 1.00 / 1.00 | passes |
| `mul` | 512 | 17.29 | 1.002 | 0.2 | 1.021 | 2.0 | 1.01 / 1.00 | passes |
| `rms_norm` | 512 | 17.76 | 0.996 | -0.4 | 0.984 | -1.7 | 1.00 / 0.99 | passes |
| `rms_norm_mul` | 512 | 18.10 | 1.006 | 0.8 | 0.983 | -2.1 | 1.00 / 0.99 | passes |

[`bpf1-v2-comparison.json`](bpf1-v2-comparison.json) holds both sessions'
manifests, block medians and `d`, and the rule's outcome, written by
[`bpf1_compare_record.py`](bpf1_compare_record.py). The raw samples and
logs stay on `spark` in `~/.local/share/llmp/bpf1v2-20260927/p1` and
`m1`.

## Limitations

- **One host, one day.** All sessions ran on `spark` on 2026-09-27:
  rule v1's calibration and holdout in 05:27–05:51 UTC and its comparison
  in 06:30–06:37; rule v2's calibration and holdout in 23:00–23:17 and
  its comparison in 23:47–23:53.
  `spark-b` served only development runs, which are not evidence.
- **Not the whole model.** The cases are the kernels llmpalooza has; RoPE,
  softmax, the KV writes, the SiLU gate and the fused single-row MMVF
  variants are not covered until they exist. Attention runs at the
  trajectory's two KV lengths only.
- **Synthetic operand values.** Deterministic pseudo-random values stand in
  for the fixture's weights; the same values fill both memory kinds.
- **Everything in one kind.** The comparison places scratch and the cuBLAS
  workspace with the operands; it does not separate weights on host VMM
  from activations elsewhere.
- **Small kernels are noisy.** Thresholds reach 5–7% (v2: up to 7.9%) for
  some one-row elementwise kernels and the one-row k/v projection; a regression smaller
  than that in one such kernel alone would pass.
- **The rule is BP-F2's.** Its structure (four processes per arm, the
  aggregate test, the confirmation) was tuned on EXL3 kernels; this
  calibration and holdout validate it for these cases, as D-079 delegates.

## Reproduction

On the workstation, build and deploy the `cross` preset
(`mise run deploy -- --host spark cross`), regenerate or check the case
file (`bpf1_cases.py ../backend-proof-p0/fp16-plan.json bpf1-cases.txt`;
`tools/tests/test_bpf1.py` checks it), and compute the source identity:

```bash
git rev-parse HEAD
{ git diff --binary HEAD; git ls-files -o --exclude-standard -z | sort -z | xargs -0 sha256sum; } | sha256sum
```

On `spark`, with the harness binary beside a `cublas/` directory holding
the SDK's `libcublas.so.13` and `libcublasLt.so.13` (the binary's RUNPATH is
`$ORIGIN/../cublas`), run each session:

```bash
bpf1_session.py OUT/c1 --harness benchmarks/llmp_ggml_vmm_bench --cases bpf1-cases.txt \
  --arm-a cuda-malloc --arm-b cuda-malloc --order primary \
  --source-commit COMMIT --source-diff-sha256 DIFF --note '…'
```

Then, on the workstation:

```bash
bpf1_stats.py OUT/c1 > c1.json          # likewise for each session
timing_protocol.py --calibrate c1.json c2.json c3.json c4.json > bpf1-calibration.json
timing_protocol.py bpf1-calibration.json h1.json h2.json
bpf1_record.py --protocol timing_protocol.py --calibration bpf1-calibration.json \
  --sessions c1.json c2.json c3.json c4.json --holdout h1.json h2.json > bpf1-timing.json
```

A comparison session passes `--arm-b host-vmm --registry
docs/backend-proof.md --calibration bpf1-calibration.json` with the
registered harness and case file, and `bpf1_stats.py` needs `--registry`
for it. The comparison was then:

```bash
bpf1_stats.py OUT/p1 --registry docs/backend-proof.md > p1.json     # and m1
timing_protocol.py bpf1-calibration.json p1.json m1.json
bpf1_compare_record.py --protocol timing_protocol.py --calibration bpf1-calibration.json \
  --primary p1.json --confirmation m1.json > bpf1-comparison.json
```

Rule v2 differs in two places. The harness is built on a Spark
(`mise run test -- spark-native --locked` on `spark-b` builds
`build/spark-native/benchmarks/llmp_ggml_vmm_bench` beside
`build/spark-native/cublas/`) and copied to `spark`. The records are
`bpf1-v2-calibration.json` and `bpf1-v2-timing.json`, computed on a Spark
with the mise-pinned Python, from sessions named `c1`–`c4`, `h1` and `h2`
as for v1. A device-VMM comparison session passes `--arm-b device-vmm
--registry docs/backend-proof.md --calibration bpf1-v2-calibration.json`
with the v2 harness and case file; `bpf1_stats.py --registry` summarizes
it, and `bpf1_compare_record.py --calibration bpf1-v2-calibration.json`
wrote `bpf1-v2-comparison.json`.
