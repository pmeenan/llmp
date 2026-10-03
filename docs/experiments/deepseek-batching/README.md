<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek V4 Flash concurrent request batching

DeepSeek chat now serves up to four requests at once in decode **waves**,
instead of one request at a time. On the matched 7K-token HTTP protocol of
[deepseek-concurrent](../deepseek-concurrent/README.md), the completed-token
rate at C4 rises **+29.0% plain** and **+4.5% with DSpark**. In these cells every reply is
byte-identical across C1, C2 and C4. Spark A (`spark-c4e2`), 2026-10-02.
These cells predate per-wave forms: DSpark waves were then always
draft-verify. With the [chosen form](#adaptive-dspark-and-plain-waves) a
DSpark reply at C4 may differ from C1 unless `wave_form = "speculative"`.

## Design

- **Request slots.** `Dsv4Runner` keeps four request states. Each has its
  own live state (target caches and DSpark ring), verify snapshot, output
  staging and plans bound to its state. The weights, workspace, staging
  and launch context are shared. The cohort rules (active set, closures,
  fault handling) are the engine's new `request_cohort.h`. Serving maps
  four `Llm` branches onto the slots, as Qwen3.8 does, and the cooperative
  chat backend admits DeepSeek.
- **Wave graph.** `BuildDsv4WaveGraph` builds one fast-plan graph over
  every slot's rows. Row-local work runs once over all rows: every
  quantized product (`jitllm.vecq`, routed experts deduplicated across
  slots), the HC mixes, routing, combine, norms, rotations and the head.
  Each slot's compressors, indexer, attention, cache writes and DSpark
  injection run on its own state. The attention outputs are then joined
  again. A wave holds at most 16 rows.
- **Exactness.** A request's rows in a wave equal its rows alone, bit for
  bit:
  - **Verify rows.** `jitllm.vecq` gives each token the same arithmetic at
    any count from 2 to 16, so it now takes 16 tokens. GGML's float vector
    kernel is count-invariant only to 8 columns, so past 8 rows a wave
    runs the router, indexer-weight and head-mix products per slot.
  - **One-row steps.** A wave of these sets `SetVecQOneToken`, which gives
    each token the one-token launch. Weights are read again from cache for
    each token.
  - **Injection.** The drafter's injection products are GGML MMVQ, so they
    run per slot.
  - **One-row verifies.** These occur at a mask-width boundary or on the
    last token, and run alone.
- **Policy.** Plain models run decode waves. With DSpark, a draft-verify
  wave has each slot draft its own block, then one joined verify covers
  every slot's natural rows (up to 4). Since the
  [per-wave forms](#adaptive-dspark-and-plain-waves), a DSpark wave may
  instead be a plain decode wave, chosen from counted acceptance or fixed
  by the model's `wave_form`. A lone request keeps its ordinary step.
  Prefill chunks alternate with peer waves (the cooperative backend's
  existing policy).
- **Capacity.** A slot's state growth that the execution budget refuses
  beside its peers' leased state is typed (`Slot::state_refused`), so the
  cohort's capacity policy
  ([runtime-serving](../../runtime-serving.md#state-capacity-in-a-cohort))
  clears an idle conversation's state first, then makes the request wait,
  and preempts the youngest when nobody can go on. A decode step grows its
  state in its preparation, through the most rows its verify may take, so a
  wave never grows a member's state.
- **Plan memory.** The slots share one cap of 24 chunk plans; waves keep 6
  plans; every graph (chunks', draft blocks', waves') counts toward 16
  graphs and 256 MiB, and a capture alone past that is not made. The runner
  reports the bound those caps allow, a swap drops the outgoing model's
  plans and graphs, and the memory guard sets apart the largest model's
  bound ([below](#plan-memory)).

## Controls

Harness: `jitllm_spec_runner --check wave --slots N`. The first N
decode/chat prompts of `fast-swap/prompts.json`, 96 tokens each.
Results are against each slot running alone in the same process.

| Check | 2 slots | 3 slots | 4 slots |
| --- | --- | --- | --- |
| Plain waves: rows byte-identical (teacher-forced) | 142/142 | — | 332/332 |
| Injected decode waves (DSpark loaded): rows byte-identical | 142/142 | — | 332/332 |
| DSpark waves: drafts and verify rows bit-identical, same tokens | all | all | all |
| DSpark: final state fingerprints equal | yes | yes | yes |
| DSpark: discarded wave verify, state restored (stale bytes) | 0 | 0 | 0 |
| DSpark: discarded step re-run equals solo | yes | yes | yes |
| A slot leaving the cohort: state untouched afterwards | yes | yes | yes |

The DSpark discard comparison excludes only the draft block's own ring
cells, which hold uncommitted positions.

On the final build (the plan caps and the cohort capacity policy, base
main `5a5a0b9`) every control above was run again with the same results.

Before these exactness fixes, the controls measured the following against
solo:

- Plain waves: argmax agreement 217/220, margin-move p99 1.82.
- DSpark: drafts differed at two steps because of the injection.

Unit tests:

- `dsv4_test` (wave structure): joined products, per-slot state, writes and
  refusals.
- `dsv4_fast_test` (vecq): 16-token and one-token launches bit for bit,
  dense and routed.
- `request_cohort_test`.

## HTTP cells

These use the [deepseek-concurrent](../deepseek-concurrent/README.md)
protocol unchanged:

- buffered non-streaming chat, greedy, 256 outputs;
- 7,043-token prompts u0–u3;
- a fresh service per cell, with an excluded prime;
- production config: artifact `8a355bfb…`, drafter `dd2d3f9c…`, context
  262,144, 4,096-row chunks.

Each cell is one sample. Every reply of the final build's cells is
byte-identical to the earlier run's, at every concurrency.

| Native | C1 tok/s | C2 tok/s | C4 tok/s | C4 vs before |
| --- | ---: | ---: | ---: | ---: |
| DSpark, waves | 13.312 | 14.069 | 14.268 | +4.5% |
| DSpark, waves, final build (plan caps, capacity policy) | 13.392 | 14.217 | 14.387 | +5.4% |
| DSpark, before (one at a time) | 13.442 | 13.515 | 13.654 | — |
| Plain, waves | 11.076 | 13.255 | 14.441 | +29.0% |
| Plain, waves, final build | 11.092 | 13.249 | 14.466 | +29.3% |
| Plain, before | 11.091 | 11.152 | 11.192 | — |
| llama.cpp b11254 DSpark / plain (same GGUF) | 8.839 / 7.635 | 9.503 / 8.949 | 10.078 / 9.793 | — |
| ds4 0.6.5, plain, community IQ2_XXS artifact | 13.177 | 14.590 | 16.395 | — |

- **llama.cpp.** Native now leads at every cell:
  - DSpark: +50.6% / +48.0% / +41.6%.
  - Plain: +45.1% / +48.1% / +47.5%.
- **ds4.** The comparison is not matched. ds4 ran a different artifact,
  through literal completions, which still run one at a time. On those
  numbers native plain trails by 15.9% / 9.2% / 11.9% (before: 16.1% /
  23.8% / 32.2%). Native plain C1 measured the same on both artifacts
  (11.06 / 11.09). The [same-session comparison](#against-ds4-same-session)
  below splits prefill from decode.
- **Latency.** All four requests now finish together (C4: 70.6–71.8 s).
  Before, the first finished at about 19–23 s and the last at 75–92 s.
- **Memory.** Peak `MemAvailable` drop at C4 is 109.85 GiB with DSpark
  (before: 107.83) and 98.44 plain (before: 97.04).
- **Remaining gap.** The cells are now prefill-bound: four 7K prefills take
  about 45 s of C4's 71 s. ds4 prefills about 1.7× faster.

Pure decode, from the harness (4 slots, waves of 3–4, short prompts):

| Mode | Solo | Waves |
| --- | ---: | ---: |
| Plain | 21.8 tok/s | 37.6 tok/s |
| DSpark | 32.8 tok/s | 37.4 tok/s |

DSpark gains less because a wave of four 4-row verifies reads about 78
distinct routed experts, plus four separate draft blocks.

## Against ds4, same session

This section compares native main `64faee6` with ds4 0.6.5, measured
interleaved on Spark A on 2026-10-02 (22:00–22:45 EDT). It splits each
engine's time into prefill and decode. Every native cell's output is
greedy, and so is ds4's. Each cell is one sample.

**Matching is partial.** Main `64faee6` cannot serve the community IQ2_XXS
artifact (`cd39d504…`) that ds4 runs (since fixed:
[community artifact in waves](#community-artifact-in-waves)); it refuses
at start:

> measuring a wave of 4 slots: a DeepSeek V4 wave needs every layer in
> the fast plan's fused form

A wave needs every layer in the fused form, and that form requires F32 HC
mix weights. The community GGUF stores them as F16 (`Builder::Fused`,
`dsv4_graph.cc`), so the artifact cannot start with four slots. The same
check also keeps its single-request decode on the unfused path. Two native
arms therefore stand in for main on that artifact:

- **Community, one slot (serial).** A private shim, kept outside the
  repository, recompiles only `serving.cc` with the build's flags. It reads
  the DeepSeek model's request slots from the environment, set to one. The
  community artifact then runs exactly as on the base, one request at a
  time.
- **Community, one slot, output-A/HCA.** The same shim also turns on the
  runner's default-off output-A/HCA experiment.

Main itself runs unmodified on the original artifact `8a355bfb…`, in
waves, both plain and with DSpark. That arm does batch, but on a different
artifact from ds4's.

**Protocol.** The protocol is [deepseek-concurrent](../deepseek-concurrent/README.md)'s,
with these settings:
- prompts u0–u3, 7,043 tokens;
- a fresh service per cell, with an excluded one-token prime;
- context 262,144 and 4,096-row chunks;
- requests streamed with `include_usage`, so each request's first token is
  timed.

ds4 gets the non-thinking chat request (`think: false`, greedy, defaults,
`--no-spec`). Native has no request field for thinking, so it renders
thinking on. Its prompt is ds4's recorded IDs with `<think>` in place of
`</think>` as the last ID. Both report 7,043 prompt tokens. Native's
outputs are therefore reasoning text and ds4's are answers; their lengths
are fixed (finish `length` in every cell). No quality claim is made.

Prefill and decode are timed in three ways:
- ds4's own per-request `timings`: TTFT, prefill tok/s and decode tok/s;
- each stream's first token;
- separate cells that measure one part alone: prefill-only cells
  (`max_tokens` 1) and decode-dominated cells (124-token prompts, 512
  outputs).

All 40 cells completed, each with exit code 0 and with the admission
check passed before and after.

**7,043-token prompts, 256 outputs (completed tok/s):**

| Arm | C1 | C2 | C4 | C4 first / last completion, s | Native / ds4 at C1 / C2 / C4 |
| --- | ---: | ---: | ---: | --- | --- |
| ds4, community | 13.234 | 14.864 | 16.687 | 58.50 / 61.37 | — |
| Native, community, serial | 10.478 | 10.962 | 11.061 | 23.39 / 92.58 | 0.79 / 0.74 / 0.66 |
| Native, community, serial, output-A/HCA | 11.639 | 11.758 | 11.767 | 21.99 / 87.02 | 0.88 / 0.79 / 0.71 |
| Native main, original, plain waves | 11.135 | 13.281 | 14.457 | 70.50 / 70.83 | 0.84 / 0.89 / 0.87 |
| Native main, original, DSpark waves | 13.168 | 14.259 | 14.387 | 70.42 / 71.18 | 1.00 / 0.96 / 0.86 |

**Prefill alone** (`max_tokens` 1, same prompts, wall from submission to
the last reply):

| Arm | C1 s | C2 s | C4 s | Prompt tok/s (C1 / C4) |
| --- | ---: | ---: | ---: | --- |
| ds4, community | 6.53 | 13.05 | 26.19 | 1079 / 1076 |
| Native, community | 10.31 | 20.34 | 40.31 | 683 / 699 |
| Native, community, output-A/HCA | 8.72 | 17.19 | 34.21 | 808 / 824 |
| Native, original | 11.28 | 22.40 | 44.73 | 625 / 630 |

Neither engine batches prefill: the wall grows linearly with the number of
prompts. At C4 ds4 prefills 1.54× native on the same artifact (1.58× at
C1), and 1.31× with output-A/HCA. Against main's original artifact it is
1.71×.

**Decode-dominated** (124-token prompts, 512 outputs):

| Arm | C1 tok/s | C2 tok/s | C4 tok/s | Per-request decode tok/s, C1 / C2 / C4 |
| --- | ---: | ---: | ---: | --- |
| ds4, community | 20.85 | 30.93 | 46.55 | 21.5 / 15.9–16.3 / 11.9–12.6 |
| Native, community, serial | 18.84 | 18.89 | 18.89 | 19.6 / 19.6 / 19.5–19.6 |
| Native main, original, plain waves | 20.94 | 29.98 | 37.36 | 21.9 / 16.0 / 10.1 |
| Native main, original, DSpark waves | 32.18 | 34.20 | 34.55 | 34.6 / 18.3–20.2 / 9.3–10.3 |

Native's ratio to ds4 is:
- plain waves: 1.00 / 0.97 / 0.80;
- DSpark: 1.54 / 1.11 / 0.74.

**Where the time goes**

- **C1 (same artifact).** The gap is prefill.
  - ds4: 6.5 s prefill (1,088 tok/s), then decode at 19.8 tok/s.
  - Native: 10.3 s prefill (8.7 s with output-A/HCA), then decode at
    19.25 tok/s.
  - So prefill accounts for 3.8 s of the gap and decode for about 0.4 s.
    The 7K C1 cell's own first token came at 11.18 s, against 10.28–10.32 s
    for the first request of every other native community cell; that
    0.9 s is unexplained.
    Native decode on this artifact is 10% below its rate on the original
    artifact (19.6 vs 21.9 tok/s at 124 tokens): F16 HC mixes keep every
    layer off the fused decode form.
- **C4, 7K (main's waves vs ds4).** All of the gap is prefill.
  - Native: four prefills take 44.6 s; then 256 four-row waves take 26.0 s,
    about 39 tok/s.
  - ds4: four prefills take 26.2 s. Its other 35.2 s are 297 decode steps,
    many narrower than four rows because ds4 decodes while later prompts
    still prefill.
  - Native spends 18.4 s more on prefill and about 9 s less on decode.
- **C4, decode-dominated.** The gap is the wave step.
  - Native's four-row wave takes 99 ms (10.1 tok/s each); its solo step
    takes 46 ms. ds4's four-row step takes about 82 ms (about 12.2 tok/s
    each), with a solo step of 46.5 ms.
  - At C2 both engines are about 62 ms. The difference appears at four
    rows.
  - In these waves every token takes the one-token vecq launch, so that
    each request's rows stay byte-identical to solo. Each weight is
    therefore read again for each token. ds4 reads it once per step, and
    its replies change with batch width.
- **Scheduling.** Native's cooperative backend picks prompt chunks round
  robin across slots and starts no decode until every prompt is prefilled.
  At C4 every request's first token arrives at 44.5–45.0 s, against ds4's
  6.8 / 17.1 / 27.9 / 38.9 s. This costs native no throughput: its waves
  stay four rows wide. It does delay first completion: 70.5 against 58.5 s.
- **DSpark at C4.** DSpark waves trail plain waves when decode dominates
  (34.55 vs 37.36 tok/s). DSpark wins at C1 (+54%) and at C2 (+14%).

**Prefill by prompt length.** Native, community artifact, literal IDs,
one request at a time. Each length was run twice; the table gives seconds,
and in brackets each arm's tok/s:

| Prompt tokens (chunks) | 4,096 (1) | 4,097 (4,096 + 1) | 6,144 (4,096 + 2,048) | 7,043 (4,096 + 2,947) | 8,192 (2 × 4,096) |
| --- | --- | --- | --- | --- | --- |
| Native | 5.47–5.75 | 5.55–5.56 | 8.65–8.68 | 9.88–9.94 | 10.88–10.89 |
| Native, output-A/HCA | 3.94–4.01 (1,021–1,039) | 4.03–4.05 | 7.12–7.16 (858–863) | 8.38–8.43 (836–841) | 8.00–8.01 (1,023–1,024) |

With output-A/HCA, a full 4,096-row chunk runs at about 1,025 tok/s, about
5% under ds4's 1,079. The 7K prompt is slower than 8,192 tokens, though: its
2,947-row last chunk takes 4.4 s against 4.0 s for a full chunk. D2R,
the IQ2 pair write-back, output-A and HCA are each guarded to 4,096-row
chunks. Below that size the chunk falls back to the slower forms. At the
full-chunk rate, the 7K prompt would take about 6.9 s against ds4's 6.5 s.
That figure is inferred, not measured. The chat route's turn boundary
(an extra 2-row unit and the turn checkpoint) adds about 0.4 s more
(chat 10.31 vs literal 9.91 s).

**Next steps, by expected effect**

1. **Partial-chunk prefill.** Generalize the 4,096-row-only mechanisms
   (output-A, HCA, D2R down, pair write-back) to any chunk of up to 4,096
   rows. This is most of the remaining 7K prefill gap. Done: the stage
   mechanisms take every chunk of 64 rows or more, and so does
   output-A/HCA under the tie-aware rule
   ([tie-aware re-scoring](../ds4-output-prefix/README.md#tie-aware-re-scoring-and-partial-chunks)).
2. **Output-A/HCA by default** (or as a per-alias flag). It gains 22% per
   full chunk. Done
   ([default-on acceptance](../ds4-output-prefix/README.md#default-on-acceptance)).
3. **F16 HC mixes in the fused form.** Done: see
   [below](#community-artifact-in-waves).
4. **A multi-token vecq launch with the one-token reduction order.** Done
   for dense products (+2–3% a wave, [below](#not-adopted-and-open)). The
   wave step's profile then found the routed products' pair scan
   ([wave step](#four-request-wave-step-and-scheduling)).
5. **Scheduling.** Both measured and not adopted
   ([below](#four-request-wave-step-and-scheduling)):
   - Finish the oldest prompt first, then alternate its decode with later
     prompts, for earlier first tokens.
   - Prefer plain waves over DSpark from three active requests.
6. **The original artifact's prefill** (11.2 s for 7K tokens). Its
   IQ3_XXS/IQ2_XS experts and F32 HC mixes bypass the stage mechanisms.
   ds4 cannot load that GGUF, so there is no reference for it.

**Conditions.** Spark A (`spark-c4e2`, GB10, driver 580.178.04).
- **Native.** `jitllm-runtime` `60d1eb20…`, the `spark-native` build of a
  tree equal to main `64faee6`.
- **Shim.** `cbc1acc5…`. It recompiles only `serving.cc`, reading
  `DS4M_WAVE_SLOTS` and `DS4M_OUTA_HCA`, with the build's exact flags and
  link line. Records are in `shim/`.
- **ds4.** `ds4-server` `b8ea076d…` (`76d51ef8`, 0.6.5), arguments as in
  [deepseek-concurrent](../deepseek-concurrent/README.md#against-ds4).
- **Harness.** Controller `run.py` `31e1eff5…`, length screen `plen.py`
  `94cf0f67…`.
- **Jobs.** Supervised jobs `ds4m-long`, `ds4m-ps` and `ds4m-plen`. The
  admission check found at least 114.8 GiB available, with no GPU,
  container or model process.
- **Peak `MemAvailable` drop.** Native community 87–88 GiB, original 97–98
  (DSpark 108–109), ds4 105–106.
- **Raw records.** `spark:~/scratch/ds4m/` (`records/`, `tools/`, `shim/`).

## Community artifact in waves

The fused form now takes any HC mixing-weight type. `jitllm.dsv4.hc_mix`
reads F32, F16 and BF16 weights, widening each to F32 exactly, so the sums
keep one order for every type. A quantized mix takes GGML's product inside
an otherwise fused layer, per slot in a wave. The community artifact
(F16 mixes) therefore starts with four slots, decodes in waves, and its
single-request decode takes the fused form too. A wave still needs each
layer's expert products to be `jitllm.vecq` types; an artifact whose
experts are not is served one request at a time, with a log line, instead
of refused (`Dsv4WaveSupport`).

**Exactness and quality.**
- Wave controls (`--check wave`, plain, community): 2 slots 142/142 and
  4 slots 332/332 rows byte-identical to each slot alone, every argmax
  the same, a leaving slot's state unchanged. The original artifact's 4
  slots: 332/332, as before.
- Forced 8K ds4 trajectory (`jitllm_dsv4_exec`, community): all 32
  argmaxes agree with ds4 (the unfused base: also 32), and all 32 equal
  the base's. The prefill row is byte-identical to the base's; the 31
  decode rows move by at most 5.45 (mean RMS 0.29). Against ds4's own
  logits for those 31 rows, the fused form's mean RMS is 0.450 (the
  unfused base's 0.444) and its largest difference 5.86 (base 7.35). A
  forced repeat is byte-identical.
- Original artifact through the runtime: every 7K reply at C1 and C4 is
  byte-identical to the earlier session's (`ds4m`).

**HTTP cells.** The [same-session](#against-ds4-same-session) protocol,
one sample a cell, measured back to back on Spark A. *Serial* is main's
runtime limited to one slot by the shim, the community artifact's serving
before this change (unfused decode, one request at a time).

| Arm | 7K C1 | 7K C4 | 124-token C1 | 124-token C4 |
| --- | ---: | ---: | ---: | ---: |
| Community, serial | 10.913 | 10.990 | 18.781 | 18.873 |
| Community, waves (this change) | 11.170 | 15.502 | 19.493 | 38.648 |
| Original, plain waves (this change) | 11.127 | 14.410 | — | — |
| ds4, community (earlier session, above) | 13.234 | 16.687 | 20.85 | 46.55 |

- Single-request decode on the community artifact rises from 19.3–19.5
  to 20.1–20.3 tok/s at HTTP (the original's: 21.6); on the forced 8K
  trajectory, from 17.25 to 18.64–18.79 tok/s.
- C4 rises +41% (7K) and +105% (124 tokens). Against ds4's earlier cells
  that is 0.93× and 0.83×. The C4 decode wave is the 99 ms step above
  (10.1–10.5 tok/s a request).
- Replies are byte-identical across C1 and C4. They differ from the serial
  arm's, whose decode took the unfused form.
- Peak `MemAvailable` drop at C4: 89.3 GiB (serial 88.0).

Records: `spark:~/scratch/dss0/` (`wave-*`, `forced-*`, `records/`),
controller `tools/run.py` (the `ds4m` one with this build's arms),
jobs `dss0-checks` and `dss0-http`, 2026-10-02.

## Four-request wave step and scheduling

**Profile.** `nsys` with graph nodes traced, community artifact,
`--check wave --slots 4`; kernel time per graph replay, ms:

| Kernels | One request | Wave of four, before | Wave of four, after |
| --- | ---: | ---: | ---: |
| Graph span | 49.5 | 93.4 | 86.0 |
| `jitllm.vecq`, all | 39.3 | 73.5 | 65.8 |
| — routed gate+up | 5.6 | 22.1 | 19.5 |
| — routed down | 3.4 | 18.1 | 12.7 |
| Attention | 1.3 | 5.3 | 5.3 |
| Compressor, cache writes, row gathers | 1.4 | 5.4 | 5.5 |
| Float products (MMVF) | 5.2 | 6.2 | 6.3 |

The routed down ran at about 145 GB/s in waves, against 205 alone. One
thread of each routed block scanned every (token, slot) pair of the
product for the block's expert, serially: 24 pairs in a wave of four, 6
alone. The down's rows are short (eight Q2_K blocks of 2,048 values), so
the scan cost more than the product. Now the block's first warp scans 32
pairs a ballot, keeping the pairs in order; no sum changes.

- `vecq_bench`, four tokens of distinct experts, in the wave's one-token
  configuration (`block r2 w4 p1`): Q2_K down 411 → 286 µs, IQ3_XXS down
  513 → 345, IQ2_XS gate+up 517 → 495. The bench's four-token default row
  (`warp r2 w4 p1`) barely moves (about 1.01× in the review's runs).
- Wave controls stay 332/332 rows byte-identical on both artifacts. The
  four-slot waves take 7.85 s against 8.44 (community; main 8.60) and
  8.00 against 8.57 (original; main 8.79).
- What remains of the step is mostly reads. The routed products run near
  bandwidth: about 210 GB/s if a layer's four requests name 22 distinct
  experts, as in `vecq_bench` (an estimate; the wave's own expert count
  was not recorded). The dense ones read each weight once (33.5 ms
  against 30.3 alone). Per-slot attention, compressor and
  state operations add about 12 ms and 2,444 kernels over one request,
  the next lever (a multi-slot launch).
- Two variant changes were tried. The Q2_K down four rows a block is
  bit-identical but, after the fix, no faster; it was dropped. The
  IQ3_XXS down a warp's row changes the sums and fails the 32K history
  at step 249 (2.616 nats), so it was dropped too.

**HTTP cells, same session.** The [same-session](#against-ds4-same-session)
protocol, one sample a cell. *This tree* adds output-A/HCA on full
chunks ([default-on acceptance](../ds4-output-prefix/README.md#default-on-acceptance)),
dense products read once a wave, and the pair scan, to main `6052286`.

| Arm | 7K C4 tok/s | 7K C4 decode, tok/s a request | 124-token C4 tok/s |
| --- | ---: | ---: | ---: |
| Main, community | 16.01 | 10.2 | 38.87 |
| This tree, community | 18.57 | 11.3–11.4 | 43.52 |
| Main, original | 14.76 | 9.8–9.9 | — |
| This tree, original | 17.03 | 11.2–11.3 | — |
| ds4, community | 17.13 | — | 46.68 |

- 7K C4 gains 16.0% (community) and 15.4% (original) on main and leads
  ds4 1.08× (it trailed 0.93×). 124-token C4 gains 12.0%: 0.93× ds4,
  from 0.83×. Its decode step is 85.5 ms against ds4's ~80.
- 124-token replies are byte-identical to main's. The 7K replies change
  with output-A/HCA and are the same at every concurrency.
- A 124-token prompt's first token took 3.1 s against ds4's 0.8. Prompt
  units are now taken shortest remaining first
  ([below](#prompt-order-adopted)), and the first comes at 0.9 s.

**Oldest prompt first.** Main's backend alternates prompt units across
requests (round robin), with one decode wave after each unit, so four
concurrent prompts finish prefill together. Taking the oldest prompt
first was measured with one decode wave a unit, and with decode waves for
a share of each unit's time. Community 7K C4, one sample a cell:

| Policy | tok/s | First tokens, s | Completions, s |
| --- | ---: | --- | --- |
| Round robin (main) | 18.59 | 32.3–32.8 | 54.7–55.1 |
| Oldest first, one wave a unit | 18.27 | 8.2 / 16.7 / 25.3 / 33.9 | 54.7–56.0 |
| Oldest first, decode 0.25 × unit time | 17.10 | 8.3 / 18.6 / 29.2 / 39.8 | 52.4–59.9 |
| Oldest first, decode 0.5 × unit time | 16.14 | 8.3 / 20.6 / 33.1 / 45.7 | 49.8–63.4 |
| Oldest first, decode 1.0 × unit time | 14.23 | 8.3 / 24.7 / 41.2 / 57.7 | 38.2–71.9 |
| ds4 | 17.17 | 6.5 / 16.8 / 27.3 / 37.9 | 56.9–59.7 |

The first three rows and ds4 share a session. The original artifact
loses 1.6% with one wave a unit (17.09 → 16.82). Qwen at 8K C4, two runs
each, ABBA, loses 2.3% (32.30 / 32.08 → 31.25 / 31.63), its first
completion a little later. Replies are byte-identical across policies.

A request decoding alone takes about 50 ms a token, against 21.5 in a
full wave. Decode time spent before every prompt has prefilled therefore
costs throughput, about twice what it gains in first completion. These
variants were not adopted; see [below](#prompt-order-adopted) for the
order adopted later.

**Plain waves against DSpark from three requests.** Original artifact with
the drafter, `--check wave`, 96 tokens a request: three slots, DSpark
39.63 tok/s against injected decode waves' 35.21; four slots, 40.34
against 40.85. DSpark stays.

### Prompt order, adopted

The short prompts' first-token gap was not waves. A 124-token prompt
runs a 122-row chunk, its turn checkpoint and a 2-row chunk. The
122-row chunk reads nearly every routed expert once (0.55 s of GPU time
in `nsys`, mostly `mul_mat_q` and the D2R down). Round robin ran all four
prompts' first chunks before any prompt's last unit. So every first
token waited about 2.3 s behind the other prompts, against ds4's
sequential 0.8 s for the first.

**First try: oldest prompt first.** That fixed the equal-length cells:
124-token C4 first tokens went from 3.0–3.5 s to 0.9 / 1.7 / 2.5 / 3.4 s
at unchanged throughput. Generating requests waited up to 4 s between
waves (`kDecodeStall`). Review found two faults:
- **Head-of-line blocking.** A long prompt that arrives first holds every
  short one behind it. In a mixed C4 cell (one ~32K prompt, then three
  126-token prompts 0.3 s later), the short prompts' first tokens went
  from 18.6–19.0 s (round robin) to 38.8–40.5 s for DeepSeek, and from
  7.9–8.4 to 16.0–17.0 s for Qwen3.8.
- **The stall bound.** The clock was checked only before a unit was
  chosen, so the measured gaps reached 5.7 s (DeepSeek) and 5.1 s
  (Qwen3.8). And 4 s is a DeepSeek chunk, while Qwen3.8's take under 2 s,
  so its streaming stall grew from 1.9 to 5.1 s. Partial waves did run
  between prompt units, too.

**Adopted: shortest remaining prompt first, bounded by units**
(`runtime/cohort_schedule.h`, unit-tested; `serve_api.cc`
`NodeBackend::NextUnit`, shared with Qwen3.8):
- **Order.** The prompt with the fewest tokens left to prefill goes next,
  the oldest of equals. A prompt not yet started counts what its
  conversation can reuse. Prompts of equal length still finish one after
  another.
- **Aging.** A prompt passed over for 12 other prompt units goes next
  regardless, so a long prompt cannot be starved by short ones cycling
  through the other slots.
- **Stall bound.** A generating request waits for at most one prompt
  unit: once one has run since its last wave, a wave runs before the
  next. That is round robin's cadence, bounded by a unit rather than a
  clock.

Same session, fresh service per cell, against round robin (the review's
build of this tree with main's schedule). Mixed C4: one ~32K prompt
(32 outputs) and three 126-token prompts (200 outputs), arriving 0.3 s
apart, long first (LF) or short first (SF):

| Cell | Round robin: short first tokens, s | Adopted | Round robin: tok/s | Adopted | Largest gap, s |
| --- | --- | --- | ---: | ---: | --- |
| DeepSeek, LF | 18.6 / 18.8 / 19.0 | 4.8 / 5.8 / 6.9 | 12.45 | 12.31 | 4.50 → 4.53 |
| DeepSeek, SF | 6.1 / 10.7 / 10.9 | 0.9 / 1.9 / 3.0 | 12.34 | 12.13 | 4.52 → 4.60 |
| Qwen3.8, LF (two runs) | 7.9–8.4 | 2.2 / 3.0 / 3.9 | 24.43, 24.37 | 23.56, 23.56 | 1.84 → 1.86 |
| Qwen3.8, SF (two runs) | 2.8 / 5.0 / 5.3 | 0.5 / 1.3 / 2.2 | 23.69, 23.70 | 23.68, 23.76 | 1.85 → 1.84 |

The long prompt's own first token comes about 0.7–0.9 s later. The
DeepSeek SF row is from the first build, with 8 units of aging; the rest
use 12. Equal-length cells (7K C4, four prompts at once, 256 outputs):

| Cell | Round robin | Adopted |
| --- | --- | --- |
| DeepSeek community: tok/s | 19.90 | 19.60 |
| DeepSeek community: first tokens, s | 28.5–29.0 | 7.3 / 14.7 / 22.4 / 30.0 |
| DeepSeek original: tok/s | 18.18 | 18.01 |
| Qwen3.8 8K C4 (ABBA, two runs each): tok/s | 32.27 / 31.96 | 31.29 / 31.70 |

Equal-length throughput is 0.9–2.0% below round robin. The mixed Qwen3.8
long-first cell is 3.3–3.6% slower in two runs. The cost is the bound's:
a request that finishes prefill early decodes in partial waves, one after
each later prompt unit, where round robin finished every prompt together
before decoding.
DeepSeek's replies were byte-identical across schedules (measured before
per-wave forms; a DSpark reply with the chosen form now follows its waves,
[below](#adaptive-dspark-and-plain-waves)). Qwen3.8's vary
with timing under either schedule (its round robin runs differ from each
other).

### DSpark with the community artifact

The drafter's verify embeds its drafts on the device. That lookup took
only quantized token tables, but the community GGUF's is F16, so it
failed ("get_rows of quantized rows into F32"). The runner's lookup
(`LookUpRows`, `engine/dsv4_runner.cc`) now gives F32, F16 and BF16 tables
to GGML's float gather. That widens exactly, as the host's lookup does.
Checks on the community artifact with the drafter (`jitllm_spec_runner`):
- the device lookup equals the host's for all 129,280 tokens' rows;
- greedy speculation: every speculative token the plain engine's argmax,
  or a near-tie within the verify's noise (2.61); no violations on the
  eight prompts;
- forced rejections: 0 stale bytes over 7.8 GB compared;
- DSpark waves at 2 and 4 slots: identical to each slot alone.

Acceptance per prompt is 0.48–0.82 (original artifact: 0.56–0.78).
Through the runtime, same session, community, DSpark against plain:

| Cell | Plain | DSpark |
| --- | ---: | ---: |
| 7K C1: tok/s | 12.93 | 16.98 (+31%) |
| 7K C4: tok/s | 19.94 | 18.77 (−5.9%) |
| 124-token C1: tok/s | 19.72 | 30.40 (+54%) |
| 124-token C4: tok/s | 43.88 | 38.64 (−11.9%) |

As on the original artifact, DSpark leads alone but trails plain waves at
four requests through the runtime, though the harness's wave check finds
them level (42.54 against 42.29 tok/s). Choosing the form per wave
(below) narrows that gap.

### Adaptive DSpark and plain waves

Per-wave timing through the runtime (community, 124-token C4, each mode
forced) shows where DSpark loses. Its waves complete more tokens a second
at two and three requests, and fewer at four:

| Wave width | DSpark: ms a wave, tok/s | Plain decode: ms a wave, tok/s |
| --- | --- | --- |
| 2 | 151, 31 | 78, 26 |
| 3 | 197, 34 | 89, 34 |
| 4 | 249, 42 | 86, 47 |

The host's gap between waves is under 1 ms at four requests, so the
difference is in the waves themselves: the four draft blocks run one
after another, and the joined verify reads every slot's 4 rows.

With a drafter, the runtime now chooses each wave's form per width
(`execution/adaptive_wave_mode.h`; `serving.cc` `RunPreparedGenerationWave`)
from counted tokens and a recorded cost, never from wall time:
- **Cost.** For each width, a DSpark wave's time in plain decode waves:
  1.94, 2.21 and 2.90 at widths 2, 3 and 4 (151/78, 197/89, 249/86 ms
  above). They are recorded per model (`kWaveCost` in `serving.cc`) and
  must be measured again when the wave step changes, as per-slot streams
  would change it.
- **Acceptance.** One moving average (weight 1/8) over the service's
  draft-verify waves of any width, of the tokens each commits per
  request. As Qwen's draft depth does, it counts only complete verifies:
  a verify cut short by a mask width or the reply's end, one run alone,
  and a wave with a sampling member are left out. A plain wave commits
  one per request, so DSpark wins at a width while that average exceeds
  the width's cost.
- **Choice.** The first three waves of two or more speculate. After that,
  each width switches form only when the average crosses its cost by 3%.
  After every 64 plain waves, two waves speculate to update the average.
- **Overrides.** A lone request and an uncalibrated width speculate. A
  wave with any sampling request speculates, since the two forms turn
  one seed into different tokens; seeded replies therefore repeat. The
  model's `wave_form` key (`"auto"`, `"speculative"`, `"plain"`;
  `AdaptiveWaveMode::Force`) fixes the form for exactness controls: with
  `"speculative"` a reply at C4 equals the reply alone, on the shipped
  build.
- **Ring.** A plain wave of a speculative model still feeds the drafter's
  ring (`Dsv4Runner::DecodeWave`, the injected decode wave), so its
  requests can speculate again later.

The same requests in the same waves, from the service's start, therefore
choose the same forms, and the harness's runs repeat exactly. Which waves
a request joins still follows when its peers arrive: equal prompts
prefill in arrival order, and concurrent HTTP clients arrive in no fixed
order. So with DSpark a greedy reply at four requests may differ from the
same request alone, and between two runs of the same cell, because the
forms' arithmetic differs. Two runs of the 124-token C4 cell (`h20s`,
`h21s`) prefilled in different orders and gave different replies on both
artifacts at the same rate (41.58 and 41.84 tok/s community, 40.66 and
40.98 original; that build still averaged every verify). Plain waves
without the drafter, and forced DSpark waves, gave the same replies in
every run. A failed plain wave now keeps the conversation's prefix as the
ordinary plain step does.

`jitllm_spec_runner --check wave --wave-mode alternate` checks the
drafter's ring across forms: plain and DSpark waves alternate on the same
slots, and each slot is compared with a solo run that alternates the same
way; a plain row that differs is a reported problem. Community: 82 of 82
plain rows identical and 0 mismatched tokens; original: 84 of 84 and 0.

Same session (`h22s`, `h22`), fresh service per cell, one build (the
final source): plain without the drafter, DSpark with
`wave_form = "speculative"`, and the chosen form (`"auto"`):

| Cell | Plain | DSpark only | Chosen | Over DSpark | Under plain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Community 124-token C4: tok/s | 43.00 | 38.58 | 41.23 | +6.9% | −4.1% |
| Original 124-token C4: tok/s | 40.89 | 37.66 | 40.32 | +7.1% | −1.4% |
| Community 7K C4: tok/s | 19.55 | 18.63 | 19.16 | +2.8% | −2.0% |
| Original 7K C4: tok/s | 17.97 | 17.76 | 17.72 | −0.2% | −1.4% |

At four requests the chosen form gains up to 7% over DSpark alone and
lands 1–4% under plain waves; the 7K cells' differences are within or
near this study's noise. The remaining gap to plain is not attributed.
A run of the build before acceptance counted only complete verifies
(`h20s`, `h20`, `bin/det7`) gave 1–9% over DSpark alone and 1–5% under
plain. A lone request keeps DSpark's +31% (7K) to +54% (124 tokens). The
`"speculative"` replies equal the forced build's (`bin/spec7`) and the
build's before per-wave forms (`dss6`) byte for byte, so the configured
form reproduces the cohort-equals-alone control on the shipped build.
The plain replies equal them too. The earlier adaptive build chose by
measured wall time (`h16`–`h19`) and gained 2–11%. It was replaced
because its schedules, and so its replies, could change between runs.

Records: `spark:~/scratch/dss5/` (`v2`, `v3`, `h1`–`h6`, `q1`, `bin/`,
profile `v3/wave4.sqlite`; the adopted scheduling `h10`, `h10s`, `q2` and
profile `short4.sqlite`; the community drafter `dr`, `h9`, `h9s`; the
wall-time adaptive waves `h16`–`h19`, `h17s` with the per-wave timing
behind the costs; the chosen form `h20s`, `h21s`, `h20`, binaries
`bin/det7` and `bin/spec7`; the final source `h22s`, `h22`, `w8`, binary
`bin/det8`), controller
`runx.py` over `dss0/tools/run.py`, 2026-10-03.

## Plan memory

*Superseded 2026-10-02:* the caps and the swap's drop below are gone.
Plans and graphs are charged inside the execution budget past one step's
floor and given back through the node's reclaim order, and a plan's arena
holds what its graph uses
([memory-pressure](../memory-pressure/README.md); D-055 and D-090 as
amended). This section records the interim design.

Plans and graphs are host and driver memory outside the catalog. Before
the caps, four slots' 32 chunk plans each, 32 wave plans and 16 graphs
(draft blocks' uncounted) could grow the service by about 2 GiB. The
adversarial churn-then-fill probe (`cf`, ten rounds of four distinct
prompt lengths, then four conversations filling the state budget) took
the service to 3.74 GiB RSS, `MemAvailable` to 0.40 GiB and 1.34 GiB into
swap: the 6 GiB margin meant for the driver and cuBLAS had been consumed.

Measured one new shape at a time (`--check plan-memory`, process heap by
`mallinfo2`; a kind's first shape also loads its kernels and is not
counted), against what the runner now counts (`PlannedHostBytes`: the
arena plus 512 bytes a launched node; a graph then 12 KiB a node):

| Kind (4 slots, fast plan) | Heap, MiB | `MemAvailable` drop, MiB | Counted, MiB |
| --- | ---: | ---: | ---: |
| One-row chunk plan | 9.9 | — | 10.6 |
| 2,048-row prefill plan | 10.1 | — | 11.2 |
| Decode wave plan | 18.4–18.6 | — | 19.7 |
| DSpark wave plan | 18.5–18.6 | — | 19.8 |
| One-row chunk graph | 15.6–17.0 | 13–25 | 29.5 |
| Decode wave graph | 33.0–38.3 | 43–56 | 56.6–57.6 |
| DSpark wave graph | 36.9–41.0 | ~50 | 61.6 |

Replays add nothing. A graph is counted at 16 KiB a launched node, 37% over
the largest drop measured (a decode wave graph's 56 MiB at 4,912 nodes;
its count is 77 MiB). The caps are sized from the working sets the
challenge measured in the C4 HTTP cells (up to 26 plans, 330 MiB, and 3
graphs, 143 MiB at 12 KiB a node): 24 chunk plans shared by the slots, 6
wave plans, one draft plan a slot, and 16 graphs within 256 MiB, least
recently used dropped. They bound the total at 647 MiB with DSpark.
Qwen3.8's runner gets the same treatment from its measured C4 working set
(24 plans, 250 MiB; 16 graphs, 105 MiB at 12 KiB a node): 24 chunk and 24
drafter plans shared by its slots, 4 target and 16 draft waves, and 16
graphs within 192 MiB, a bound of 628 MiB.

A swap drops the outgoing model's plans and graphs (D-090 as amended), so
the start's memory guard sets apart only the largest model's bound
(`plan_host_bytes` in its log line), not their sum. With every model's
kept, DeepSeek at 262K with DSpark beside Qwen3.8 at 33,792 with MTP (a
configuration that serves on the base) was refused at start (3.75 GiB of
plans set apart); now it starts with 0.63 GiB set apart and serves. Its
DeepSeek state room is the base's less 0.63 GiB: 1.35 GiB in these runs
(108.4–108.5 GiB available beside the fixed memory), and about 0.4 GiB on
the same host an hour earlier, when 1 GiB less was available.

| Two-model service (DeepSeek 262K DSpark, Qwen3.8 33,792 MTP) | Result |
| --- | --- |
| A DeepSeek cohort in flight, a Qwen3.8 request between its turns, swaps back and forth, against each conversation run alone | 24 replies identical, 0 different |
| DeepSeek → Qwen3.8 → DeepSeek, one idle DeepSeek conversation | 7.67 / 9.46–9.48 s (two runs) |
| The same, four idle DeepSeek conversations (every slot's state spilled and restored) | 7.72–7.75 / 9.51–9.52 s |
| `swap-table --pairs deepseek:qwen3.8` (DeepSeek plain at 32K), first use and prepared, 8K and 0 context | all six rows exact, 7.8–8.9 s; a prepared return plans again (0.04 s) |

Each return plans and captures again; the swaps stay far inside the 20 s
bound. (The same table with DSpark stops at its diagnostic state snapshot,
which the guard's margin refuses in this two-model configuration; the base
build refuses it the same way.) Matched HTTP cells after the tighter caps
(single model): DeepSeek
DSpark 13.41 / 14.28 / 14.39 and plain 11.11 / 13.23 / 14.50 tok/s at
C1/C2/C4; Qwen3.8 8K C4 32.70 and C2 29.89 against the base build's 32.56
and 30.02 in the same session.

`cf` again, after (with the first caps, 32 chunk and 8 wave plans and 768
MiB of graphs; two runs, the same probe and a hog sized for a similar
room): the service's RSS peaks at 2.44 GiB, and `MemAvailable` stays at
3.5–4.3 GiB through the churn (both runs) and the fill (the first; its four
conversations now all complete, [below](#capacity-under-pressure)). The only swap is the
hog's own idle pages, which the kernel pages out: in the second run its
RSS falls by exactly what swap gains, 0.59 GiB. One ~10-second dip to
0.58 GiB in the first run coincided with another session's 2m24s of CPU
on the host (the journal), as did similar dips during `react2` and
`fill3`; the service's RSS fell, not rose, through each.

## Capacity under pressure

The challenge's state-capacity probes, re-run against the cohort policy
(with the first caps).
A memory hog sizes the state room (1.6–1.9 GiB beside the 1.3 GiB plan
bound, about what the challenge had):

| Probe | Before | After |
| --- | --- | --- |
| `fill3`: four 27–29K-token conversations, 2,500 outputs each | three failed at the 16,384 chunk; the fourth completed, or failed mid-reply | all four complete: three wait, the youngest is preempted and rebuilt, a finished conversation's idle state is cleared |
| `react2`: an idle 40K conversation, three 24–33K growers, then its continuation | the three growers failed while the idle state stayed resident | all four complete: two growers and then the continuation wait, the continuation is preempted and rebuilt, a finished grower's idle state is cleared; the continuation prefilled its 40K tokens again (0 cached; the challenge's run reused 40,007) |
| `fill4`: two idle 26–27K ballast conversations, two decoders of 2,000+ tokens | both decoders failed mid-reply | both complete; a ballast's idle state is cleared |

Every probe ends with the service healthy and its short follow-up requests
answered. A request that waits runs nothing while it waits; a preempted
one rebuilds its state by prefill, so its later tokens may differ from an
uninterrupted run's, while tokens already streamed never change
([runtime-serving](../../runtime-serving.md#state-capacity-in-a-cohort)).

The runner's typed refusal itself (`--check capacity`, two slots and a
1 GiB state room): beside slot 0's 612 MiB, slot 1's growth to the same
is refused for capacity, its state usable and unchanged; a growth past
the context is refused, but not for capacity; a selected slot cannot be
cleared as idle; once idle slot 0 is cleared, slot 1 grows and runs.

## Not adopted, and open

- **Verify waves of 2 rows a slot at 3–4 slots.** These measured +8%
  pure-decode wave throughput (40.4 tok/s) and +4% at HTTP C4. They change
  each request's step form, so replies differ from C1. Not adopted, to
  keep byte-identical replies.
- **Injected decode waves instead of DSpark waves past 2 slots.** 14.27 at
  HTTP C4, no better than the adopted waves.
- **A multi-token vecq launch with the one-token reduction order.** Done,
  for dense products, with a small gain. Each one-token configuration
  (rows, warps, reduction) now has a four-tokens-a-pass sibling whose
  tokens' sums equal the one-token launch's bit for bit (unit-tested on
  every DeepSeek type, dense, GLU and routed, at 3, 4 and 6 tokens).
  `vecq_bench` at four tokens, GB10:
  - Dense products read once gain: Q8_0 1024×32768 179 → 152 µs,
    8192×4096 172 → 162, the Q4_K head 2,059 → 1,845.
  - Routed and GLU products lose: routed IQ2_XS gate+up 517 → 677 µs,
    IQ3_XXS down 517 → 709, the Q5_K shared gate+up 62 → 84. A wave's
    four requests rarely share an expert, so a pass has nothing to share
    and only pays the registers.
  So a wave of one-row steps takes the sibling for dense products without
  a GLU and keeps one token a pass elsewhere. The 4-slot plain wave
  control stays 332/332 byte-identical; its waves take 8.44 s against
  main's 8.60 (community) and 8.57 against 8.79 (original), and HTTP C4
  decode rises from 10.1 to 10.4 tok/s a request
  ([output-A default](../ds4-output-prefix/README.md#default-on-acceptance)
  cells). The remaining gap to ds4's four-row step (99 vs 82 ms) was
  not the dense products' reads; the
  [wave step's profile](#four-request-wave-step-and-scheduling) found the
  routed products' pair scan.
- **Partial decode waves between prompt units.** Measured, not adopted:
  no earlier completions for 1.6–2.3% throughput, or earlier first
  completions for much more
  ([above](#four-request-wave-step-and-scheduling)). The adopted order is
  shortest remaining prompt first, with one prompt unit between waves
  ([above](#prompt-order-adopted)).
- **Plain waves instead of DSpark from three requests.** DSpark is 12.6%
  faster at three slots and level at four; not adopted.
- **One launch for every slot's attention and state operations.** Open.
  About 12 ms of the 86 ms four-request step.
- **DSpark at four requests through the runtime.** Mostly closed by
  [choosing the form per wave](#adaptive-dspark-and-plain-waves): 1–4%
  under plain waves at C4, against 5–12% before. A joined draft block (one launch for
  every slot's draft) would cut the DSpark wave's cost at its source.
- **Joined DSpark draft blocks.** Open. A wave runs one draft per slot.
- **Literal completions (`/v1/completions`).** Open. They remain serial.

## Provenance

- **Host.** Spark A (`spark-c4e2`, GB10, driver 580.178.04).
- **Build.** The `spark-native` build of worktree `cc/dsv4-batching` (base
  main `bae7a0d`); after the plan caps and the cohort capacity policy, of
  worktree `cc/dsv4-batching-3` (base main `5a5a0b9`); after the swap-out
  drop and the measured caps, of worktree `cc/dsv4-batching-4` (base main
  `418887b`).
- **Harnesses.**
  - The HTTP cells used `deepseek-concurrent`'s `run.py`, pointed at this
    build.
  - The wave controls used `jitllm_spec_runner --check wave`; plan memory
    `--check plan-memory`, the runner's capacity refusal `--check capacity`.
  - The probes (`cf`, `fill3`, `react2`, `fill4`) are the challenge's
    `dsprobe.py`, with the state room read from the guard line (its plan
    bound included) and a memory hog sized for a target `MemAvailable`.
  - Supervised jobs `dsb-wave1`–`5`, `dsb-http1`–`3`; then `dsb2-pm4`,
    `dsb2-atk2`–`5`, `dsb2-final1`, then `dsb4-q1`–`q3` and
    `dsb4-swaptable4` (the base's refusal: `dsb4-swaptable-base`).
- **Admission.** Every model load passed the admission check: at least 105
  GiB available, no GPU process.
- **Raw records.** Kept outside Git in `spark:~/scratch/dsbatch/`
  (`wave*-*/spec.json`, `http/native-*-r3/`), `spark:~/scratch/dsb2/` and
  `spark:~/scratch/dsb4/`.
