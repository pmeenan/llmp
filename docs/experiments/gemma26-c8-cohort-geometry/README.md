<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 C8 geometry transfer

The first Gemma26 C8 oracle screen passes the unchanged pre-oracle cohort bound and independent conditional-loss gate. It retains **three positive-margin greedy differences**, all inside the old bound 8.085711284875883. This is a bounded operational pass; strict-zero does not pass.

| Completed heads | Reference byte-exact | Strict differences | Outside old bound |
| --- | ---: | ---: | ---: |
| Eight prefill frontiers | 0/8 | 0 | 0 |
| Paid decode rows | 118/256 | 3 | 0 |

The three rows are 53 (owner 5, completed prefix 998), 207 (owner 7, completed prefix 1017) and 208 (owner 0, completed prefix 1018), with reference winner margins 0.1417822838, 0.0191278458 and 0.0268664360. There are no ties. Maximum raw logit difference over all 264 finite rows is 2.042461157. Both engines repeat complete heads byte for byte; the native initialized frontier/final states and layouts also repeat exactly, and its frontier state matches the old native freeze.

For the separate 256 supplied within-history targets, complete-vocabulary FP64 mean NLL is 10.953604858686354 native versus 10.949650248640475 reference: relative conditional loss **+0.396244%**, below the predeclared 3% degradation gate. The target chronology is row r*8+owner predicting ID 992+r for r=0..31. The final eight heads predict position 1024 and have no supplied target. This is not the separate 1023-transition corpus gate. No allowance is recalculated; the old 5e8 native-only freeze was authenticated before the first stock output, and new complete finite native own-repeat proof 9484 was closed before either stock arm.

This uses exactly the compiled 4514/eb7 source22 [geometry factor](../gemma-c8-cohort-geometry/README.md). There is no new kernel build, math change, layer whitelist or grant enlargement. The 26B profile selects 60 owner/requested-cohort8 steps, 121 plain norms, 60 normRoPE, 90 normADD and 30 routing/reduction steps each, with 32 graph replays. The synthetic D256/D512 operator proof and 13 focused controls are recorded in the preceding report. Actual stock whole-model Q descriptors are not claimed observed.

The first eight checkpoint-qualified histories from the fixed twelve-history carrier are used at prefix 992, max_rows=1024, max_head_rows=8, context=4096, F16 cache with local/global widths 2,048/4,096, plain norm plus normRoPE/normADD/MoE routing and reduction. Each native eight-column wave runs two attention quads; stock uses a normal physical eight-sequence batch. Graphs and fusion stay enabled. The helper command is `ARTIFACT NEW_DIR 26 8 joined all CARRIER production`; it retains 264 complete heads and two initialized state snapshots per owner under the unchanged 293,715,968 B publication allowance. All inputs, artifacts, source/method, binary/SDK and official identities are in [results.json](results.json), with the full 22-source inventory linked to the 31B report.

The short timing screen records native 1.63509/1.63915 s versus fresh stock 1.67582/1.67658 s: native mean 1.63712 s versus 1.67620 s, or 2.33146% less elapsed paid time. This does not establish sustained performance, default serving parity or other depths/cohorts. No earlier 26B C8 stock oracle existed, so this is not an old-versus-new numerical comparison for 26B.

All four official jobs complete DONE0, with 6+3+6+3 successful steps; application retirement, source checks and all owned container absence checks pass. TensorFold primary was checked at this transfer entry: 609ca419, version 0.6.5, with Gemma26 MLX only and no comparable GB10 CUDA recipe. The helper measured from the 22-source frame remains distinct from the subsequently integrated opt-in parent-validation helper. Full regression is deferred by the owner. Serving owner/joined defaults and default-admission qualification remain unchanged/open; historical scalar or other-cohort failures are not waived.
