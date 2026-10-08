<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma C2 real-owner attention factor

The opt-in real two-owner adapter passes the existing strict zero-margin choice and independent 64-target conditional-loss gates on both approved profiles. Gemma31 matches all 64 paid full heads to the retained FIRST stock physical-two-stream outputs, with two nonexact prefill frontier heads. Gemma26 matches 34/66 complete heads and all 66 choices. Neither profile has a strict difference or reference tie. This closes the measured C2 screen; C3 model arithmetic, partial cohorts and serving defaults remain unqualified.

| Profile | Strict differences / complete heads | Byte-exact heads | Maximum raw delta | Relative conditional loss |
| --- | --- | --- | --- | --- |
| 31B | 0 / 66 | 64 paid; 2 frontier nonexact | 0.787642 | −0.148562% |
| 26B | 0 / 66 | 34 | 0.558619 | +0.018077% |

The separate conditional gate uses whole-vocabulary FP64 normalization on 64 within-history targets and retains its 3% limit. Row 0 predicts position 992; likelihood rows 0..31 target positions 992..1023. Final row 32 predicts 1024 and is unscored. It is not a 1,023-transition corpus result.

The original [independent attention screen](../gemma31-production-c2/README.md) failed eight strict choices; its [private one-query local-MMA factor](../gemma31-c2-local-mma/README.md) failed nine. Both failures remain preserved. Actual two-owner operands and whole physical-stream attention geometry match the paid 31B outputs for this fixed carrier. This does not establish a general defect in independent attention or qualify a global executor override.

The adapter accepts two or three active real cache owners in a fixed four-pointer carrier, leaving unused roots null. All active writer dependencies preflight before mutation. The original shader/fixup bodies remain unchanged, and direct whole-two/three geometry preserves existing four/eight/twelve partition behavior. C1, retained features, unequal widths and larger partial-cohort tails retain their existing fallback behavior. Active counts and scratch/descriptor envelopes are checked and funded through sizing and both placement passes.

The model recipes use the first two independently authenticated 1,024-ID histories, context 4096, prefill 992 at caps 31B=256 / 26B=1024, head cap 2, normal graphs/fusion and the unadopted ordinary norm/ROPE/ADD recipe, with routing/reduction on for 26B. Actual first decode plans select 60/30 owner-adapter steps, one per layer carrying both owners. Both complete 32 paid waves / 64 records with the unchanged 73,515,008-byte publication allowance for 66 full heads. Each native own pair freezes before its stock comparison and verifies all finite complete heads, four full initialized-state snapshots, layouts and carrier bytes. Each 31B state is 922,746,880 bytes; each 26B state is 230,686,720 bytes. Stock own complete heads repeat exactly. Native/reference state equality is not claimed.

Ten focused graph, placement and GPU controls pass with zero skips. A separate one-control operator expansion passes all eight heads 16/32 × active roots 2/3 × D256/D512 cases at read 1024, comparing contiguous whole physical-stream MMA with independent real roots through eager execution, capture and two poisoned replays. The test-only heads16 expansion does not rebuild or change the measured source4 helper. Compile and fixture-environment failures remain recorded: both failed supervisors return 1; the environment failure's step returns 250.

The initial 31B native 3.46276/3.48563s pair is 5.453273% slower than the retained stock 3.29867/3.29040s pair. These noncontemporaneous walls are not the matched speed decision. Fresh stock/native/native/stock bookends record:

| Profile | Native walls / mean (s) | Stock walls / mean (s) | Native / stock elapsed | Native / stock spread |
| --- | --- | --- | --- | --- |
| 31B | 3.37414, 3.37341 / 3.373775 | 3.35801, 3.29200 / 3.325005 | +1.466765% | 0.73 / 66.01 ms |
| 26B | 0.775437, 0.776954 / 0.7761955 | 0.782866, 0.786217 / 0.7845415 | −1.063806% | 1.517 / 3.351 ms |

The 31B stock spread exceeds its 48.77 ms mean gap, so the short bookend does not establish a stable remaining gap or sustained parity. All new timing native outputs match the corresponding frozen own proof. The fresh 31B stock heads/input match its immutable FIRST stock pair, supplying timing/output confirmation only. The 26B stock bookends are its first own-repeat oracle and normal physical-two-stream workload, with F16 local/global capacities 2048/4096. These short timings do not qualify sustained service or default batching.

The measured isolated candidate is based on aeddf3f plus exact unadopted SOURCE14 recipe and the reviewed real-owner factor. Its private helper is d13c461d… with SDK receipt eb7a4ebc…; source4/source5 and complete source/input/artifact/reference identities are recorded in [results.json](results.json). Core integration excludes SOURCE14 runtime/settings/harness changes. Fourteen official attempts and all fourteen owned container retirements are recorded, including the two initial failed builds. No scalar calibration, margin allowance, old failure, production default or full model-support gate changes. Raw heads, states, individual scoring records and official logs stay external.
