<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma multirow prefill without cache-packing copies

Gemma2's remaining warm prefill gap came from copying the independent owners'
K/V cache prefixes into temporary packed inputs on every joined chunk. A
matched current timeline showed lower native arithmetic time and lower fully
inactive GPU gaps, but 3432 extra device copies. The new closed multirow
actual-root path keeps the original selected MMA body, geometry, softcap,
mask and reduction arithmetic while reading the two real cache roots.
Gemma3 uses the same copy-producing graph and receives the cap0 transfer.

## Attribution before the change

One unprofiled control pair and one profiled pair ran on Spark B. Both used
Gemma2's approved Q8_0 artifact, F16 KV, independent 4352/4864 prefixes,
8192 context per owner and 128 rows per owner (256 total). Each fresh process
warmed the whole prefix, three supplied rows and eight natural joined decode
units, then logically cleared. Native asserted that all 520 physical state
extents, 279433328 plan bytes and 22 captured graphs survived Clear. Paid
work was the full second prefill, followed by three supplied rows off clock
and 32 natural C2 decode steps with the final two complete heads. No optional
planning or ahead capture occurred in paid native prefill.

| Measurement | Native | llama.cpp |
| --- | ---: | ---: |
| Unprofiled paid prefill | 1.30859 s | 1.20730 s |
| Unprofiled paid decode | 0.581349 s | 0.581936 s |
| Profiled paid prefill | 1.33606 s | 1.24683 s |
| Profiled paid decode | 0.601480 s | 0.593801 s |
| Prefill first-to-last-kernel span | 1333.391232 ms | 1245.616640 ms |
| Kernel active union | 1156.227370 ms | 1181.288234 ms |
| Device-to-device copies | 3432 | 0 |
| Device-to-device summed duration | 124.908704 ms | 0 |
| All kernel/copy active union | 1281.442730 ms | 1187.492394 ms |
| Fully inactive GPU gaps inside span | 51.948502 ms | 58.124246 ms |

These are single-pair descriptive measurements, not sustained parity results.
The 87.774592 ms profiled span excess is bounded by the actual kernel edges;
it is not the end-to-end wall-time difference. Kernel arithmetic is lower by
25.060864 ms of active union. All GPU activity is higher by 93.950336 ms while
fully inactive gaps are lower by 6.175744 ms. Further host scheduler work is
not the first intervention supported by this trace.

Trace/source checks establish exactly 39 paid prefill waves: 33 joined and
six scalar, with 37 state-only waves and two output frontiers. The whole
process contains 40 two-row vocabulary projections: eight warm and 32 paid
decode projections. Native paid prefill uses 35 replays and four eager runs
beside capture; stock uses 17 replays and 22 eager runs. Native prunes the
37 state-only final attention operations (977 versus 1014 attention calls).
Capture API time overlaps GPU work and is not additive to the GPU span.

The 3432 copies split into 3300 K/V copies (33 waves times 25 non-final
attention layers times four owner K/V copies) and 132 global/local mask
concatenation copies. The source-derived logical bytes are 15,227,682,816:
padded widths sum to 73984; K/V contribute 25*8192*73984 bytes and masks
128*2*4*73984 bytes. Nsight's graph-replay byte metadata reports a larger
inconsistent total; it is preserved in raw evidence but is not actual-traffic
evidence. No 27 GB traffic claim is made.

## Representative factors

The native off/on/on/off factors use fresh processes and the exact corrected
warm-cycle output work above. Every arm requires complete finite final heads,
valid natural choices, actual selected-path counters, retained warm backing,
zero paid optional planning, successful process/device retirement and matched
source/input/binary/SDK/cuBLAS identities. Values are n=2 means.

| Family/workload | Packed prefill | Actual-root prefill | Change | Paid prefill+decode change |
| --- | ---: | ---: | ---: | ---: |
| Gemma2, 4352/4864 | 1.313470 s | 1.199695 s | −8.66217% | −6.00538% |
| Gemma3, 1280/1536 | 0.479084 s | 0.465039 s | −2.93164% | −1.24215% |

Gemma2 decode changes +0.00215%; Gemma3 changes +0.47383%. These are short
comparisons with unchanged decode implementation and do not establish a
decode speed change. Packed/actual-root prefill bookend drift is −0.1126% /
−0.3370% on Gemma2 and +0.6693% / −0.6076% on Gemma3. The measured factor
includes the observed capture behavior: Gemma2 paid capture/replay changes
4/35 to 3/36; Gemma3 remains 5/8. It does not assume all 124.9 ms of baseline
copy duration becomes end-to-end gain. Gemma2's 64 choices and two complete
256000-value heads also match the frozen stock-control payload byte for byte;
Gemma3's 64 choices and two complete 262208-value heads match across all arms.

## Closed implementation and bounds

The additional primitive accepts D256/H8/GQA2, two equal-width real F16 cache
roots, 128 query rows, owner offset zero and cap50 (Gemma2) or cap0 (Gemma3).
It preserves original Columns32 geometry, four query tiles, the mask prepass,
stream-K partition and both reduction fixups. The existing MMA body is
unchanged. The graph retains mask concatenation but removes K/V packing;
64-row and other ineligible multirow chunks retain the prior packed path.

Only cap0 multirow accepts aligned widths through Gemma3's trained 131072
maximum. Its actual-root span is at most 256 MiB, Q/output 2 MiB and joined
mask 64 MiB. Validation and every overlap check use the same explicit root
bound; the address-validation parent bound remains 1 GiB. At KV batch32,
the maximum is 4096 KV tiles and 32 output tiles, with flattened index 131072,
well inside the original signed index bounds. The widened partition limit
requires this exact two-owner/four-query-tile grid. One-query and cap50 limits
remain unchanged.

## Evidence and limitations

The initial compile attempt caught two signed-conversion warnings; bounded
uint64 casts corrected them. The next launcher omitted JITLLM_TEST_DATA and
aborted on the first plan test before GPU execution. Both failure records
remain preserved. Corrected qualification passed 18 focused controls, then
23 controls for the cap0 transfer. Independent review found that the initial
G3 selector exceeded the primitive's 16K width bound; the temporary fallback
fix passed 17 plans with seven unchanged GPU controls reused. The owner then
requested trained-maximum support; the widened final kernel passed all
25 boundary controls (17 plans and eight GPU/partition tests), including
exact eager and changed-input captured replay at 16K, 32K and 131072 with
an executed last cache cell. Independent source challenge was clean.

The final ordinary defaults pass 42 additional focused controls: 26 host
plan/graph cases, 12 Gemma checkpoint/lookahead/serving GPU cases and four
actual prompt-adapter controls. Together with the 25 wide-boundary controls,
these are 67 unique positive tests with no skips. Fresh-server packed/on/on/
packed adapter controls hold lookahead enabled and compare complete finite
heads, initialized state hashes and histories. Both owners spill, survive
server teardown, restore through named kept-state metadata in a new server,
and append a continuation with exact state and heads. The paid joined portion
excludes the initial scalar advance: Gemma2 means 0.413352312 to 0.398963155 s
(−3.48109%); Gemma3 0.507809363 to 0.494589904 s (−2.60323%). These n=2
adapter measurements are separate from the full warm-prefix factors above.

Wrapped own-probe off/on pairs additionally preserve all 72 choices, all
74 complete finite teacher-logit rows and both final heads byte for byte,
plus both initialized state hashes. Gemma2 also compares both prefill-frontier
heads. GPU token publication, spill/reload, ring checkpoint restoration,
malformed-publication refusal and departed-owner isolation pass. Actual
multirow-owner counters are 425 (Gemma2) and 165 (Gemma3), versus zero with
the factor off. These diagnostics include teacher publication and state
operations and supply no performance claim.

## Fresh final Gemma2 reference

The final adopted source ran llama.cpp/native/native/llama.cpp, with the same
corrected warm-cycle paid boundary used above. Each stock arm independently
checks tokenizer inputs and hashes its actual loaded CUDA backend and resolved
cuBLAS libraries. Native proves retained warm state/plans/graphs, zero paid
optional planning, three captures plus 36 replays and the selected actual-root
path. Source, binary, SDK receipt, input and artifact hashes are checked before
and after acquisition. Every target positively retires; container cleanup is
independent of client success. All four arms have the same 64 valid natural
choices and both complete finite 256000-value final heads byte for byte.

| Arm, acquisition order | Prefill | Decode | Paid sum |
| --- | ---: | ---: | ---: |
| llama.cpp 1 | 1.209760 s | 0.581728 s | 1.791488 s |
| Native 1 | 1.206460 s | 0.582583 s | 1.789043 s |
| Native 2 | 1.201770 s | 0.582381 s | 1.784151 s |
| llama.cpp 2 | 1.205540 s | 0.583411 s | 1.788951 s |
| llama.cpp mean | 1.207650 s | 0.5825695 s | 1.7902195 s |
| Native mean | 1.204115 s | 0.5824820 s | 1.7865970 s |

Native is −0.29272% prefill, −0.01502% decode and −0.20235% paid sum.
Prefill stock/native bookend drift is −0.34883% / −0.38874%, so this short
n=2 workload is level within its observed movement. It closes the measured
warm gap for this bounded workload; it establishes neither sustained HTTP
parity nor all-context/all-shape parity. Gemma3 receives its own native factor,
exact own-state/teacher controls and trained-maximum operand qualification;
this slice adds no fresh Gemma3 stock cycle.

Ordinary Gemma2/Gemma3 select the 128-row actual-root path while retaining an
explicit packed control. Their public contexts and slot capacities are
unchanged: Gemma2 8192/two, Gemma3 4096/two or explicit scalar 8448. The
131072 qualification is a primitive/graph bound, not public context admission.
Remaining compatible row counts, query tiles and partial tails are current
M3.5 transfers, not exclusions caused by today's closed selector.

The attribution uses baseline 763bcbd; candidate work starts from 6beb7ca,
which adds DeepSeek masks without changing Gemma2 behavior. The baseline
probe SHA is 1447a1836ec856bdc842807c139b7d00189ea1a266187a842b4d0b6553fbf573;
strict attribution summary SHA is
2411e4ef6defd133639a421c1b680a006b6bddffe6acbf8fee50337a9b0a2de5.
Raw Nsight/SQLite traces, XML, payloads, process retirement receipts and failed
records remain external under the M3.5 session coordination scratch and
Spark B's scratch/m35-gemma2-warm-timeline and m35-gemma2-owner-prefill.
The final ordinary source manifest is
`76683a69b54c9a50113d5c80bca193c3c98555637a67f8cf4177da2a018ac43e`;
the wide-boundary source manifest is
`9e47d1e16eefea440aa4903ba2aa3b30e7eacff1aa2a02d3d6ab1192f5945c66`.
The installed final qualification, own-diagnostic and fresh-reference jobs
finish in 112, 80 and 25 seconds respectively, under separate 600-second
limits. The full regression suite and workstation tiers remain deferred
under the owner's current override.

## Replay inputs and entry points

Authenticated replay inputs are retained separately from disposable raw logs
in Spark B's standing reference store
`~/.local/share/jitllm/references/gemma-prefill-copies/`, under `gemma2/` and
`gemma3/`. `inputs.json` authenticates the four ID files and matching original
texts. Another host may supply externally retained files with these identities;
raw traces/logs and this session's scratch directories are not required to
rerun model controls. IDs are little-endian signed int32. Each full input has
its prefix plus three supplied rows, 32 decode rows and four departure rows.
Cycle mode uses the departure suffix only to determine the same prefix length.

| Family / owner | Prefix / full ID count | ID-file SHA-256 |
| --- | ---: | --- |
| Gemma2 / 0 | 4352 / 4391 | `102c7b555b1caed5aed3d9880a173aae153f8f8dc1534e6a4c58685c243b6c0c` |
| Gemma2 / 1 | 4864 / 4903 | `a930726bd964ae88ef0448f50a51d2e376ce2487313f26063ab84c1b0b41d77f` |
| Gemma3 / 0 | 1280 / 1319 | `4499653ec0de5def2cd171e9e1ca797741ee372332fef9fb6d59645ba08cfe9f` |
| Gemma3 / 1 | 1536 / 1575 | `0d77cf38d5a4d3e620edb5df11ef293a07fb5e9f72665a3c28df7ab4e4c48bb9` |

The matching Gemma2 original texts have SHA-256
`84a72b3bab88f084898edb6b202d7cae17cad20849c3cf556f4800c3f70159a2` /
`ac2cc3d0953da8ef932203721f91fc9f5f5ba3473208d5598ebaaeeb030f0a89`;
Gemma3 texts have
`fd5ea2494dbc3c7fb27f3bdc5c962fbb83c84d3bf153843bdd5bef29790b73f7` /
`70f5e21ab695eac68ee7b5737d42ad5a120eabb1db208ff1d43fd4894180c50c`.

Checked-in [Gemma2 probe](../../../benchmarks/gemma2_joint_prefill_probe.cc)
and [Gemma3 probe](../../../benchmarks/gemma3_joint_prefill_probe.cc) own the
paid boundaries, retained-warm checks and retirement markers. Inside an
installed GPU-exclusive Spark job, supply the approved prepared artifact,
matching ID paths and a fresh private output directory:

```text
jitllm_gemma2_joint_prefill_probe ARTIFACT IDS0 IDS1 NEW_OUT own bounded-roots device-masks prefill-ahead owner-prefill
jitllm_gemma3_joint_prefill_probe ARTIFACT IDS0 IDS1 NEW_OUT own bounded-roots owner-prefill
jitllm_gemma2_joint_prefill_probe ARTIFACT IDS0 IDS1 NEW_OUT cycle bounded-roots device-masks prefill-ahead owner-prefill
jitllm_gemma3_joint_prefill_probe ARTIFACT IDS0 IDS1 NEW_OUT cycle bounded-roots owner-prefill
```

Omit only `owner-prefill` for the packed control; use separate fresh processes
in off/on/on/off order. Own mode checks full teacher rows and state operations;
cycle mode warms, logically clears and measures the second prefix plus 32
natural decode units. The [stock helper source](../gemma2-serving/llama_joint_prefill_probe.cc)
compiled against the pinned llama.cpp commit runs inside the immutable CUDA
image with `MODEL IDS0 TEXT0 IDS1 TEXT1 NEW_OUT cycle`. Its independent
retokenization, full-logit length and completion markers must pass. Retire each
container in a finally path and verify absence independently of client exit.

For a new trace, preserve Nsight CUDA graph-node activity and export SQLite.
The checked-in [step analyzer](../../../tools/nsys_steps.py) includes device
copies and launch gaps as well as kernels. Its summed durations are not active
unions; the report's union figures merge overlapping kernel/copy intervals
between the independently confirmed paid kernel edges. Reject interpretation
if the expected 39 prefill waves, 40 whole-process vocabulary projections or
payload/retirement controls disagree. Aggregate results and authenticated
input identities remain durable even after raw telemetry is deleted at
milestone close.

The Spark build uses SDK `aarch64-c09daba6ac31edee`, CUDA13.4 and
cuBLAS130800. The linked runtime SHA-256 is
`58234cfd6352c2104856b066686c8c38c3e132e01a893079f297fea2f81f09b4`;
G2/G3 probe hashes are respectively
`0dec62dba8cb986f60ee3f6600ac9f432960369027f553777c29f4da83e4c11b` /
`7bf35a27ea5d351ef45006d51c95439e1aaf59207be16dd26e7541e1b709eda5`.
Final acquisition binds the actual first-resolved cuBLAS payloads, the
qualified SDK receipt and those binaries before/after; each arm requires no
foreign GPU process and at least 32 GiB MemAvailable.

| Approved artifact | Identity |
| --- | --- |
| Gemma2 Q8_0 prepared manifest | `eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870` |
| Gemma2 source GGUF SHA-256 | `2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe` |
| Gemma3 QAT Q4_0 prepared manifest | `8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb` |
| Gemma3 source GGUF SHA-256 | `ee91c3e7a4ab95d8c95672f9fcb58bf236b257e9f217966bcf53a5a6df4ab49a` |

TensorFold was re-pinned at both attribution and Gemma3-transfer entry:
[native f8fe17d2](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/README.md)
and [Python ed78d6fc](https://github.com/ashhart/TensorFold/blob/ed78d6fc204d89d90b045bf033d6551e7714f3a1/README.md).
Neither offers a Gemma2/Gemma3 CUDA recipe for the approved GGUF artifacts.
llama.cpp d81235049384534c167caea52b85a694f6103d14 remains the applicable
CUDA reference, through immutable image
sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db.
