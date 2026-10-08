<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma2 attention-softcap primitive

This bounded operator slice admits finite positive logit softcaps on the
existing D256/F16-KV/GQA2 vector and dense MMA primitives. Gemma2 needs 50;
zero and 25 are controls. No model, source import, runner, owner-attention
adapter or serving path is admitted by this slice.

The upstream cases already choose `use_logit_softcap=true` for a nonzero
parameter. The vector planner previously queried only the false kernel;
the shared MMA shape query also hard-coded false. Their validators refused
nonzero parameters, so this was an admission gap rather than an existing
misplanned production launch. Planning now queries the exact specialization
that the upstream case launches, including its actual occupancy. MMA's new
internal softcap argument defaults false; D64, D128, D256/D512 group8 and
owner paths keep their previous zero-only contracts and specialization.
No floating device-kernel arithmetic or source is changed.

Nonzero caps require a positive finite scale and cap, with finite
`scale / cap`, matching the upstream launcher's normalization before tanh.
Negative, NaN, infinite and quotient-overflowing parameters are refused before
submission. Existing zero-softcap admissibility is unchanged. ALiBi, sinks,
sparse gather, grouping, padding, alignment and current-descriptor checks
continue to apply to the existing primitives.

## Focused operator controls

The two new `GgmlExtOpsTest.Gemma2Softcap*` controls cover:

- D256/H8/KV4, scale 1/sqrt(256), 512 padded KV cells and finite masked rows;
- caps 0, 50 and 25, vector rows 1/3 (one/two columns), MMA rows 1/5/9/17
  (tiles 4/8/16/32, including partial query tiles), plus two-sequence MMA
  row5 with the paid mask pre-pass and its mask-padding/broadcast refusals;
- an independent FP64 `softmax(cap*tanh(scale*QK/cap)+mask)*V` reference,
  retaining the existing flash-attention NMSE bound 5e-4; no new allowance;
- complete finite outputs, exact eager repeats, two fresh graph captures with
  two exact replays each, and byte equality with the separately launched
  original MMA case;
- exactly funded pool scratch and one-byte-short refusal without a fault,
  actual occupancy/KV-batch/grid/scratch reporting, and registry selection;
- invalid caps/scales/ALiBi, quotient overflow, other head/group contracts,
  and unchanged omitted-versus-explicit false MMA shape queries.

Softcapped results must also differ from the same operands' uncapped outputs,
so accidentally ignoring the parameter cannot pass only through own repeats.
The focused native ARM run passed on spark-b on 2026-10-07 under installed
600-second GPU supervision. Both new controls passed, followed by nine focused
legacy GGML attention/owner/boundary controls and two D64 EXL3 attention controls.
A test-only follow-up strengthened repeat comparisons to `memcmp` (including
signed zero), rebuilt the affected target and passed both new controls again.
No legacy production source changed between these runs.

| Cap | Largest vector FP64 NMSE | Largest MMA FP64 NMSE |
| ---: | ---: | ---: |
| 0 | 1.54e-12 | 2.09e-6 |
| 50 | 5.80e-9 | 8.18e-7 |
| 25 | 1.17e-9 | 5.69e-7 |

All 21 cap/shape cells satisfy the unchanged 5e-4 bound. Complete finite outputs,
byte-exact own repeats/fresh capture replays and original MMA equality pass.
On this GB10 the measured occupancy and funding happen to agree across caps
0/25/50; the planner still queries the launched specialization rather than
assuming that agreement. Vector rows 1/3 use parallel blocks 2 and scratch
16,640/49,664 bytes. MMA rows 1/5/9/17 use tiles 4/8/16/32, blocks per SM
1/1/2/2, KV batches 64/64/32/32 and scratch
266,240/532,480/2,129,920/4,259,840 bytes. Two-sequence MMA row5 funds
798,976 bytes including mask preprocessing. Exactly paid launches and
one-byte-short refusals pass.

Whole-source checksum synchronization, an empty itemized dry run, checked
source inventories and actual binary/build-receipt hashes bind the runs.
The final test binary SHA-256 is
`9369ccb6674dbce653cb6c7a5028fa9ec2315921f5822d17c7657fa7374f7d05`;
the receipt is
`7ea0d2b65bb8ddeb35af2887c65e1b0281c1406edf58db411941aa57b5193d48`.
Private logs and receipts remain outside Git in the session's
`gemma2-softcap-raw` packet. Local format, license/header and portability
checks pass. This is not a full suite or whole-model numerical/performance
qualification.

## Transfer limits

Gemma2 H8 owner batching remains separate: current owner kernels instantiate
`use_logit_softcap=false`. An H8 zero-softcap Gemma3 transfer does not qualify
Gemma2's softcap50 adapter. Bounded state/graph work must also preserve the
actual final-row gather boundary, final logit softcap30, unchanged GGUF norm
weights and width2304 quantized readable tails before inference is qualified.
The pinned d812 kernels and an independent operand reference are used here;
no model reference selection or TensorFold performance claim changes.
