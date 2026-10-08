<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen conditional four-head C4 greedy screen

One approved NORMAL/CANDIDATE/NORMAL screen completed on Spark B, 2026-10-02. The candidate improves aggregate API throughput by **7.04657%** against the arithmetic mean of normal bookend rates. This is a descriptive performance result; candidate replies differ from normal, and no production selection or quality claim is made.

| Cell | Paid wall seconds | Aggregate completion tok/s | Four request latencies seconds (u0/u1/u2/u3) |
| --- | ---: | ---: | --- |
| Exact normal before | 37.626115491 | 27.215139980 | 37.367577790 / 23.133264046 / 37.626115491 / 18.829575382 |
| Conditional four-head candidate | 34.915004533 | 29.328365088 | 34.573533744 / 34.914759079 / 34.233381236 / 34.232849078 |
| Exact normal after | 37.127836344 | 27.580384446 | 36.165941995 / 37.127836344 / 22.277587101 / 18.536927604 |

The first completed reply is 83.23% later than the mean normal bookend, and median request latency is 15.70% higher. Fixed funding increases by 3.265625 GiB; these are allocation budgets, not measured peaks. Keep the current cap-two default while the four-head mechanism and depth policy are qualified separately.

Mean normal rate 27.397762213 tok/s; after/before rate ratio 1.013420635 (+1.34206%). All cells paid exactly four actual 8256-token rendered prompts, zero cached tokens and 256 completions per request, finish reason length. The unrelated one-token prime completed before the paid burst and is excluded. Timing includes native chat prefill, generation, queue and HTTP through full buffered response; TTFT is unavailable.

The original profile is the exact selected normal cap2/adaptive main binary, not a privately recompiled baseline. Candidate cap4 uses depth one when the prepared wave has three or four units, and the existing adaptive depth2/3 policy at width one or two. This frozen input screen is greedy. The private wide-wave override also applies to sampling in source; this screen establishes no sampling-policy preservation or sampling qualification.

Candidate completed wave widths 1/2/3/4: **5 / 3 / 1 / 146** (155 total). Width-specific maximum passes: **3 / 3 / 1 / 1**; maximum per-unit verify rows **4 / 4 / 2 / 2**; maximum wave verify rows **4 / 8 / 6 / 8**. Authentic four-original ordinary-MMF groups: **145**, combined columns **1160** (eight per group), paid F32 packed bytes **26,726,400** (184,320 per group). General, HC and MTP products remain pairwise; MXFP8 remains at most eight columns. Normal is uninstrumented, so no private baseline wave counts are claimed.

Normal and candidate allocation envelopes differ. Actual startup budgets:

| Bytes | Exact normal | Private candidate |
| --- | ---: | ---: |
| Fixed catalog | 4,176,973,612 | 7,683,411,756 |
| Shared activation | 3,049,259,008 | 6,096,420,864 |
| Shared scratch | 442,499,072 | 884,998,144 |
| Host input | 18,874,368 | 35,651,584 |
| Registered state virtual extent | 5,762,973,696 | 5,762,973,696 |

Candidate funding conservatively supports up to sixteen full-head input columns (2–4 rows per slot), with three prefix concat inputs and output sum funded once. Actual sharing in this screen has eight columns. Both profiles passed the actual allocation guard; no memory-equivalence claim is made.

Complete normal public replies (text, reasoning, usage, finish) repeat exactly across bookends. Candidate usage and finish match normal in all four replies. Candidate reasoning differs in all four; u0 text also differs. First differing reasoning character indexes u0/u1/u2/u3: 311 / 859 / 1000 / 302. Normal/candidate reasoning character counts: 1216/911, 1244/1268, 1232/1238, 1223/1162. u0 text character counts 0/317; other text fields remain empty. All complete replies are retained. Native raw generated IDs are unavailable; text does not establish ID equality.

The separate [four-head recovery control](../qwen38-four-head-recovery/README.md)
now passes 13 complete full-logit/argmax comparisons and 20 initialized-state
and cursor comparisons, including discard, retry and continuation. It uses
the same candidate header/archive with unchanged scalar numerical controls.
It qualifies that bounded mechanism; conditional versus adaptive history,
sampling, long context and production selection remain open.

Context33792, prefill4096, target head47172, unchanged prepared target/drafter and tokenizer/template, F16 KV/F32 recurrent state. Common-v2 inputs receipt SHA d5a35e6341e53de0286cfd777e4fadd707c12cf9d18f95010a4a38ffac0ab59d. Selected qualification CHECK b07fcb4538859708c4b9caf90fbc7618023d8fe50f04e572953a5498e63c73e1 / TARGET5802ceb0908018a0d42dc204d25affaf65fd5463e6ec3ca718222e4f42030c8c; selected source base e591e1cd190f1d0650f329f4d62c8823ece14eee. Exact normal binary SHA ec0914c16259fac3dc97a6cb4c8a46fc12648d40a96fc9955d5b6a207c159fbe. Candidate SHA94d588ff1f670514a8114b6fecb14a84ab3bc7f3bd587e6fb7ff72c9c196d88f.

Private build21s, one candidate compile/link only. Actual Ninja dependency evidence identified and rebuilt exactly three linked source consumers: qwen38_wave_plan.cc, qwen38_runner.cc, serving.cc. Each emitted private dependency file authenticates the changed private header/layout. Original37-input link closure and source pins authenticated before/after; copied archive actual paths/hashes recorded. Candidate compile/link receipt SHA c756d50f44e6c4fb199615314815f58e9b9288b0e67178fde834b50fb0dc8ede.

Frozen kit: B /home/pmeenan/scratch/m3-qwen-conditional-heads/frozen/; local source/build/controller kit /home/pmeenan/scratch/m3-qwen-conditional-heads/. Builder SHA7ba4e1439e7e0bfff5f89ee451e7b4968b3ded685865fdc68593526afdfa8975; controllerSHA0ec5cd7335b437f180b9604564948279b8ff82e0b3e83738d8ccef3dccb5bdd3. Four source SHAs: serving8bb504885b27a55253a6a6654864ebfc8789de59dee3e4126c8b493438d358ae, wave-header414c2928624f6fc268cef9882ddb4856e90fa7078b50c3fccb8f8877a8f02d2e, wave-source6b8ba44575da48c3a8ce8a603ba19f33e3f15a22de3e3f70de71904efa660e1c, runnercdcdf38777198a5775c01ea88265dc6498bd97ade06933938d2688af0d5c31b9. Full source delta retained beside kit. Earlier row-budget and positive head-only frozen kits remain unchanged.

Raw receipts/logs/replies: B /home/pmeenan/scratch/m3-qwen-conditional-heads/{build-r1,http-r1}/; local /home/pmeenan/scratch/m3-qwen-conditional-heads-records/{build-r1,http-r1}/. Build receipt SHA ca52de715a1737631212cdfab7f0a9e91e47d3a2939a99b9f64d00ec0744e91f; completed HTTP receipt SHA a770e96a126ed73dc0909536f7d0dc9c91decbe56341441ca4f6a152584f0efc. Supervised jobs qwen-conditional-build-r1 pid3934513 and qwen-conditional-http-r1 pid3935500 completed rc0. All15owned HTTP requests terminated uniquely200; all three client/service statuses0, services reaped. Fresh >=105GiB GPU/container/native gates passed before every model load; final strong retirement **117.167GiB** clear, spark-job busy reports0GPU jobs/processes. No suite, matrix, rescue or production-default change; no workstation scripts/build/model work and no commit.
