<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ready NEOX first screen: refused evidence

The [registered single-body screen](READY-NEOX-FIRST-SCREEN-REGISTRATION.md),
commit `00cd82a444e92cd1f86578fd94dc40dd771d8895`, ran once on `spark`.
Its alternative-arm bookends were 6,368 ns and 7,296 ns, a 14.573%
increase. This exceeds the prospective 10% bound. The run was refused;
no candidate row, accepted median, ranking or data installation resulted.
The original bound was retained and the run was not retried.

All 36 ordered warmup, sample and bookend records were emitted. Original-IR
reference, gamma, positions and capture-inventory checks passed before the
drift refusal. The process then aborted through the benchmark failure path
(supervisor exit 134), rather than completing normal teardown. The supervisor
completed its failure record and source/binary post-checks passed. A separate
post-run supervisor check reported no remaining GPU processes. This is not
successful timing retirement.

The run used the exact source, binary, protocol and key recorded in the
registration: 115-path r46 manifest
`0feb99c513af3ebda2e6d6457a609f6b39974e85e60c8ab7caa1585ebd7740d8`,
binary `f9ba576ffc3daee0b659f0536f0e4c9412feddc77209f1a800238cf7ed5e083c`,
and protocol `e61f221d8f4677d15ed8a868b4d693bd7468d63012493e33cb394e52de080fb4`.
Both source and binary identities were checked before and after execution.
Raw observations remain outside Git at
`/tmp/llmp-m35-orchestration/u6-resume/ready-neox-first-screen/refused-run.log`
(SHA-256 `78424ce4c062ad090798632a530fea3875445b77ef4085d4004e5f3efb062a9e`).

Source review found that the external start event precedes host-side projection,
launch and resource validation before graph submission. GPU idle time during that
host gap can enter the interval for this tiny body. This is a possible timing
contamination, not proof of the cause of the observed drift. A default-off option
that records timing events inside the original body capture was subsequently
qualified, including its two extra event nodes and completion-aware event
lifetime. A separately registered
[captured-event comparison](READY-NEOX-CAPTURED-EVENTS-RESULTS.md) is accepted;
it does not reuse observations from this refused run.
No model-speed, adoption or U6 completion claim follows from this refused run.
