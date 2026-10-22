<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 source-template Paris result

**PASS for this single paired completion; general Gemma U3 quality remains
open.** Both engines returned `Paris` and stopped normally on the same frozen
source-template input. This establishes a usable source reference for this
control; it does not qualify representative likelihood or behavior, batching,
context, or speed.

The [protocol](PROTOCOL.md) was committed before collection in
`3202f46fb1f9343d528b88afb66ba6819940da04`. Its executed SHA-256 is
`9cdd6a2cac1dc68abed9ecae9c1ebcd12d9926491fd15310f06d2eba46dc56a0`;
the frozen 35-ID [input](input.json) SHA-256 is
`3d0e98600de96f8a96f7bab3655d85b9d2e974c6e39b768d092e6fa1961069ec`.
The registered helpers and protocol remain unchanged. Each arm ran once on
Spark A, in the enrolled native-then-source order, using separate installed
exclusive jobs with 600-second supervision and 30-second termination grace.

| Arm | Job | Text | Output IDs | Finish | Selected-token log probabilities |
| --- | --- | --- | --- | --- | --- |
| Native | `m35-u3-gemma26-paris-native-r1` | `Paris` | `[50429,106]` | Stop | `[-6.114664809613695e-08,-6.925708366781607e-08]` |
| Source vLLM | `m35-u3-gemma26-paris-source-r1` | `Paris` | `[50429,106]` | Stop, token `106` | `[0.0,0.0]` |

Both emitted two tokens within the registered 64-token bound, and all selected
scores are finite. The native literal request and the source's actual returned
prompt IDs match all 35 frozen integers. The native ID interpretation relies
on the reviewed literal-input API contract, not its reported token count
alone. Complete requests and returned values were retained before validation.
Native F16 KV and source BF16 KV differ as disclosed in registration; these
selected scores are not a bitwise logit comparison.

The native arm used corrected artifact
`1d20ce879b2fe72c947699357da5e20d3f6388e9cc7858b9eb6fcc2516cd3875`
and the prospectively pinned integrated runtime SHA-256
`d18c7b7a9c8c6cf383a6f607a8606d2ce2026c4b2a8cb47632ec424601704dff`.
Its actual source manifest is
`d0c8f0023776b0994de9e97beff04cb7bcd3e77ddad78bf87390bb312794cff4`;
the runtime's stale Git-derived version is not the source identity. The
registered amendment records the source patch and integrated qualification.

The source arm used the registered original checkpoint and immutable image.
Live inspection confirmed `Gemma4ForCausalLM`, `ModelOptNvFp4Config`, NVFP4
linears, all 30 expert blocks on `FLASHINFER_CUTLASS` with `use_a16=false`, and
30 Triton attention layers with BF16 KV. Its existing
`w1_weight_scale_2 must match w3_weight_scale_2. Accuracy may be affected.`
warning remains in the raw log. No scales, kernels, weights or dispatch policy
were changed for this control.

Both supervisors returned zero. Native retirement recorded TERM and exit zero;
source retirement removed its exact named container and confirmed absence.
Helper, protocol, source snapshot and loader checks remained unchanged as
applicable. Post-job receipts showed no running or waiting jobs and no GPU
compute processes; Spark A was released after collection.

The first read-only preflight failed because the corrected artifact was absent
on A; no model arm had launched. That failure remains preserved. The existing
B artifact was then copied directly to A under coordinated exclusive jobs,
with all 14 manifest-listed files verified by size and full SHA-256 before
atomic installation. This was transport of the existing artifact, not another
import. The successful second preflight preceded both collections.

The original wrong-fold artifact, corrected native raw-ID failure, matching
source raw-ID repetition, failed source launch and first CPU tokenizer-freeze
failure remain recorded. The successful template control does not isolate BOS
as the explanation: it changes the whole prompt/template relative to the raw
five-ID diagnostic. It establishes neither a bad checkpoint nor a native
arithmetic defect.

The compact aggregate [results.json](results.json) has SHA-256
`7b5f6ab8654c13eb8b188ff67f22aef124277767e14630ba8bd5bb19bd629f34`.
It records raw evidence hashes, registration, actual dispatch, output, precision
and retirement. Raw requests, responses, logs and supervisor receipts remain
outside Git at `/tmp/llmp-gemma26-paris-native-r1` and
`/tmp/llmp-gemma26-paris-source-r1`, with sibling `-supervisor` directories;
Spark A copies are under `~/.local/share/llmp/m35-u3-quality/` with the same
`gemma26-paris-*-r1` names. Enrollment, both preflights and the artifact-copy
receipt are in `/tmp/llmp-gemma26-paris-registered-3202f46f`.
