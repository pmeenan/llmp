<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 HC and ragged target-head sharing

Selected on spark-b, 2026-10-02. Compatible two-to-four-row HC BF16
products share immutable weights through the original cuBLAS path.
Three-plus-four target verification heads also share the ordinary
seven-column MMF product. Input concatenations are paid; preparation,
nonlinear mixing, attention, recurrence and each request's commits stay
independent. One-row and unsupported products retain their existing path.
Conservative workspace provisioning funds both inputs and outputs.

## Paid two-request screen

Adaptive depth and the 47,172-token draft vocabulary are unchanged.
Context is 262,144, prefill chunks 4,096. Each fresh chat renders 8,266
tokens with no cached tokens and completes 256 length-capped tokens.
An unrelated one-token weight prime precedes each burst and is excluded.

| Arm | Pair wall, s | Aggregate tokens/s | First finish, s |
| --- | ---: | ---: | ---: |
| Original before | 19.265277106 | 26.576311214 | 19.087102711 |
| Combined | 18.797415897 | 27.237786449 | 18.625110693 |
| Original after | 19.367993505 | 26.435366155 | 19.190823124 |

Mean original duration divided by candidate duration gives **+2.762185%**
throughput; original bookend movement is **0.533169%**. All nine capped
replies preserve text, reasoning, usage and finish fields exactly.
These are paid HTTP bursts including prefill, queueing and host work,
not pure decode or a completed-answer quality suite. This single screen
does not establish solo or cross-engine parity. Separate HC and head
screen gains are not additive.

## Native controls and qualification

Same-history controls cover 3+3, 3+4, 4+3 and 4+4 verification rows.
Each original/candidate arm retains 44 finite full target vectors,
24 initialized-state captures, 40 page/cursor comparisons, 28 full-output
controls and 20 complete HC comparisons. All 71,702,400 HC values per
arm match exactly, as do full target vectors and initialized state.
Each completed two-slot plan executes 194 grouped HC products and one
full-head pair. Forced rejection, own discard/resubmit, accepted peers
and shared continuation are exact. Capture-only retention hooks and
256 MiB extra activation funding are absent from the HTTP measurement.

The selected change keeps two funded slots and adaptive depth. The
[four-slot screen](../qwen38-capacity/README.md) loses 13.16%, so no larger
capacity matrix or default change follows.

Final qualification uses all 1,276 files at main `e591e1c` plus the
selected four-file change, including DeepSeek Q-head `07d61bc` and the
Qwen serving bridge `2611466`. Changed source timestamps were invalidated;
the target log records the Q-head, graph-plan, registry, runner, wave and
serving rebuilds. All **1,253 locked Spark tests pass**, including 259 GPU
tests, in 82.40 s. SDK format and all three changed-unit tidies pass;
boundaries check 353 files with zero problems, headers 1,155 with zero
problems, and SDK REUSE 6.2.0 reports 1,155 compliant files. Workstation
checks remain deferred to the end of optimization.

Native and HTTP subprocesses, all three services, and final target/check
jobs finish rc0 and are reaped. Final available memory is 113.193 GiB,
with GPU/container/native process probes clear. Before/after source,
compiler metadata and runtime identities agree.

Provenance: native receipt `0821dba7`, HTTP `3e30d1ac`, target `5802ceb0`,
final check `b07fcb45`; full source map
`64a9c44a70e37a3fc52d17a255cb9fb4fe4a6226a15a779e05d652c85f397063`,
runtime
`ec0914c16259fac3dc97a6cb4c8a46fc12648d40a96fc9955d5b6a207c159fbe`.
Raw records remain outside Git under
`/home/pmeenan/scratch/m3-qwen-combined-sharing-records/{build-r3,native-r1,http-r1}`
and `/home/pmeenan/scratch/m3-qwen-combined-production-records/{target-r3,check-r1}`.
Two prototype compiler failures and routine target metadata refusals are
retained; none required a numerical rerun.
