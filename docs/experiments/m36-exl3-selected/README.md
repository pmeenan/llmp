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
It does not qualify the shared suffix, packed-product execution, O variants,
4.5-bpw fixture, historical Tier C, spill/restore/kept-state or deletion. Those
remain open in G5b/G5c/G5d.

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
