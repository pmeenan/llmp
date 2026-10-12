<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# EXL3 selected operations and first reconstruction execution

The shared GGML lowering can bind approved EXL3 linear operations into its
ordered step program. Single products and paired gate/up products preserve
all physical weight parts, existing F16 rounding and selected implementation
identity. Packed G/O variants use 98 authenticated P3 tuning records; all
69 earlier records are unchanged. The 18 output-specific reconstruction
facts remain authoritative. Unknown tuning keys and unsupported shapes refuse.
Resident norm scales remain BF16; the existing GGML copy kernel widens them
to transient F32 operands. Primitive expansion waits for all fusion inputs,
so a computed scale cannot split its selected RMSNorm/multiply operation.

Construction submits no work. The selected source owns stable pointer tables,
disjoint persistent locks and its first-use initialization on the owned stream.
The diagnostic explicitly selects uncaptured execution through a keyed option;
ordinary component-source capture requirements remain the default. Each run
retains complete ownership through completion, including on an uncertain result.

On Spark B, 26 focused host checks, six complete selected-operation controls
and one actual GPU conversion control passed with zero skips. The descriptor
controls cover all four profile identities and the approved reconstruction
boundaries; they do not qualify actual execution on all four profiles.

A newly sealed native reference completed the fixed 145-token prefill and
16 suffix steps of the approved 4.0-bpw G fixture. Its 145-row output then
matched the shared runner bit for bit across all 151,936 vocabulary entries:
22,030,720 F32 values, or 88,122,880 bytes. Both input/admission controls passed.
The same original 1,040-ID file is authenticated by both arms; the shared
screen consumes its first 145 IDs. The reference uses an externally admitted,
source-bounded workload under an exclusive job; it has no memory-cap option.
The candidate prepays its 4-GiB host inventory within its 8-GiB domain and
checks the separate physical margin before construction.

The actual screen took 964 ms; this is functional evidence, not a speed gate.
It does not qualify packed-product execution, O variants,
4.5-bpw fixture, historical Tier C, spill/restore/kept-state or deletion. Those
remain open in G5b/G5c/G5d.

On 2026-10-12, G5b2a extended the shared 4.0-bpw G control to the
complete 145-token prefill and 16 one-row suffix steps. A fresh native
reference completed all 17 phases using the same original 1,040 IDs and
approved typed plan. The shared runner matched all 161 full-vocabulary rows
bit for bit: 24,461,696 F32 values, or 97,846,784 bytes. The original
145-row control also passed. Both cases verify completed positions and
step/row counts; setup measures only the two distinct shape keys.

The full control retains the same 4-GiB host allowance and 8-GiB domain
with the separate 6-GiB physical margin. Its source-derived host bound is
4,109,194,928 bytes, including the second native program and the larger
oracle. Native build, two input/admission controls, six selected-operation
controls, 30 component/state/lifecycle checks and the quality gates passed.
Both actual model cases passed with zero failures or skips under a 16-GiB
zero-swap process limit. These are functional checks; other profiles, packed
execution, historical Tier C and model lifecycle controls remain open.
The unrelated full regression suites were deferred under the M3.6
incremental check policy.

Final regression on the frozen checkpoint17 passed: Spark A completed 2,554
native and 2,003 CPU CTest cases plus format, boundary, header and REUSE gates.
Spark B completed all 134 model/timing CTest cases, containing 179 raw GTest
arms, including the actual reconstruction control and all four FP16 bridge
arms. All suites and every raw GTest arm had zero failures or skips. These
regressions preserve existing coverage; they do not close the remaining EXL3
profile or lifecycle migration gates. Main built and passed all 91 focused
integration checks and the quality gates, with zero failures or skips.
Source, recipe, input and binary guards passed before and after each job.
[results.json](results.json) records the aggregate evidence; raw data remain
in external scratch until the milestone closes.
