<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen four-slot depth-one C4 screen

Spark B (`spark-56f5`), 2026-10-02. One paid O/C/O screen over selected
HC/ragged sharing and the same 47,172-entry draft head, prepared checkpoint,
template, context 33,792 and prefill chunk 4,096. Frozen common-v2 requests
each report 8,256 prompt tokens, zero cached tokens and 256 completion tokens
with `length` finish. One unrelated one-token prime precedes each fresh service
and is excluded. Wall covers HTTP submission through complete buffered replies,
including prefill, queueing and generation.

| Arm | C4 wall, s | Aggregate tok/s | Latencies u0/u1/u2/u3, s |
| --- | ---: | ---: | --- |
| Two slots, adaptive 2/3, before | 37.050722194 | 27.637787859 | 36.746552 / 20.928608 / 19.114017 / 37.050722 |
| Four slots, fixed depth one | 37.591850462 | 27.239946622 | 36.952907 / 37.591850 / 37.106703 / 36.153953 |
| Two slots, adaptive 2/3, after | 37.138738988 | 27.572287803 | 18.310541 / 36.817476 / 37.138306 / 22.474649 |

Mean original duration divided by candidate duration gives **−1.322414%**
throughput. Bookend duration movement is **+0.237558%**. The candidate does
not improve this condition; this one short screen does not justify adoption.

| Arm | Completed waves | Width 1/2/3/4 counts | Maximum passes/unit | Maximum verify rows/unit | Maximum verify rows/wave |
| --- | ---: | --- | ---: | ---: | ---: |
| Original before | 227 | 8 / 219 / 0 / 0 | 3 | 4 | 8 |
| Fixed one | 161 | 10 / 2 / 5 / 144 | 1 | 2 | 8 |
| Original after | 231 | 16 / 215 / 0 / 0 | 3 | 4 | 8 |

Counters aggregate the actual completed DraftWave/VerifyWave path and include
the untimed prime. Width four is observed 144 times in the candidate; all
candidate passes are one and its per-unit and whole-wave verify budgets stay
within two/eight rows. Independent slots, judgement, acceptance and rollback
retain the existing implementation. Production defaults remain unchanged;
only the private macro-enabled serving object differs.

Original bookend requests preserve complete public text, reasoning, usage
and finish exactly. Candidate reasoning differs for all four requests; u0
also differs in public text. First reasoning character differences are
311 / 47 / 576 / 302 for u0/u1/u2/u3. All complete original and candidate
raw/parsed replies are retained. No completed-answer quality verdict or
generated-token-ID equality follows: the native API does not expose those IDs.

The serving-only private build replaces `serving.cc.o` in a copied serving
archive over the qualified 37-input link closure, with selected SDK cuBLAS
copies and actual resolved dependencies recorded. Both original and candidate
private profiles carry the same aggregate counters. The selected warm tree
and its ordinary binary are unchanged. Build finishes rc0 in 21 seconds;
the three-service screen finishes rc0, all services return rc0 and are reaped,
and all 15 HTTP terminals match their complete replies. Final strong admission
probe is clear at 117.189 GiB; Spark B has zero GPU jobs/compute processes.
No build or hash/copy work occurs inside paid bursts. No suites or workstation
scripts run.

Provenance:

- Source SHA256 `a7ac608005768b7f0bef506fd1f297228b1e746253dbbabdd10744783f9ac3dc`;
  private macros `LLMP_QWEN_ROW_BUDGET_SCREEN` in both arms and
  `LLMP_QWEN_ROW_BUDGET_FIXED_ONE` in the candidate.
- Builder SHA256 `39d430f69a5fb2e9a4a841cc313249b6c57f8e13d4dd5100cdbfea0835da6b01`;
  build receipt `5d86181ad598d95d98f40ff6d2998a336cb7249843d917c02aa8b1e6a6d708cb`.
- Controller SHA256 `9053bc2b6359ee81759eaf710aab4a9af50f2e687088b2fdbb89b8a8135ef488`;
  screen receipt `0b43ec00ff14ed87368900aaba2705d305942c146c148dd53c15b59dcdcb2650`.
- Same selected qualification `b07fcb45`, target `5802ceb0`, normal runtime
  `ec0914c1`, startup artifact receipt `02fc48d8`, client `4f76adb8` and
  full matched-v2 input receipt `d5a35e63` as the fresh concurrent-reference run.
- Supervisors `3893088` (build), `3893932` (screen), bounded at 600 seconds.
- Raw retained local records:
  `/home/pmeenan/scratch/m3-qwen-row-budget-records/{build-r1,http-r1}/`;
  frozen kit and full private binaries remain on B under
  `/home/pmeenan/scratch/m3-qwen-row-budget/`.
