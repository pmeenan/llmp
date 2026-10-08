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

## Initial 128-row implementation and bounds

The additional primitive accepts D256/H8/GQA2, two equal-width real F16 cache
roots, 128 query rows, owner offset zero and cap50 (Gemma2) or cap0 (Gemma3).
It preserves original Columns32 geometry, four query tiles, the mask prepass,
stream-K partition and both reduction fixups. The existing MMA body is
unchanged. The graph retains mask concatenation but removes K/V packing;
At this initial slice, 64-row and other multirow chunks retain the packed
path; the subsequent flexible transfer below expands that eligibility.

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
the initial 128-row slice adds no fresh Gemma3 stock cycle.

The initial ordinary Gemma2/Gemma3 slice selects the 128-row actual-root
path while retaining an
explicit packed control. Their public contexts and slot capacities are
unchanged: Gemma2 8192/two, Gemma3 4096/two or explicit scalar 8448. The
131072 qualification is a primitive/graph bound, not public context admission.
The flexible transfer below subsequently qualifies 2–128-row tiles/tails;
larger compatible shapes remain current M3.5 transfers.

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

The initial 128-row slice builds with SDK `aarch64-c09daba6ac31edee`, CUDA13.4 and
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

## Flexible query tiles and partial tails

The subsequent transfer extends the same actual-root body to equal C2 query
chunks of 2–128 rows. It mirrors the original packed Columns4/8/16/32
specializations (rows ≤4/8/16/otherwise), pads only mask rows to 32 and launches
ceil(real rows / Columns) query tiles. Original query-tail guards, stream-K
partition/fixups and compiled arithmetic remain; the additional 8/16-column
instantiations use one shared dispatcher. Cap0 roots remain qualified through
131072 cells, with old one-query/cap50 width limits unchanged. Ordinary chunks
remain 128 rows per owner, with unchanged public contexts and slot capacities.
Partial tails now use the same qualified path. Internal overrides retain both
all-packed and 128-only controls.

The new native off/on/on/off 64-row screen uses the standing prefixes above,
warm retained plans/state/graphs, zero paid planning/eager work and the same
cycle output contract. All 64 natural choices and both complete finite final
heads match byte for byte across every arm within each family's 64-row shape.
Values are n=2 means.

| Family | Packed prefill | Actual-root prefill | Change | Paid prefill+decode change |
| --- | ---: | ---: | ---: | ---: |
| Gemma2 | 2.008650 s | 1.798255 s | −10.47445% | −8.07035% |
| Gemma3 | 0.6030255 s | 0.586207 s | −2.78902% | −1.46882% |

Prefill packed/on bookend drift is −0.72126% / −0.23606% for Gemma2 and
−1.47603% / +0.83217% for Gemma3. Decode moves +0.25170% / +0.21064% with
unchanged implementation; these short controls establish no decode speed
change. Gemma2 executes 77 paid waves (67 joined / ten scalar), Gemma3 25;
capture/replay remains 3/74 and 3/22 respectively in every arm. Selected
owner-prefill counts are 425/165 with the factor on, zero off. This factor
extends eligibility without changing the ordinary chunk size.

Focused qualification passes 52 unique controls with no skips: 36 plan and
operand cases, 12 checkpoint/lookahead/serving cases and four actual production
adapter cases. The multirow operand control exercises 50 row/width/cap
combinations, including selector boundaries
2/3/4/5/7/8/9/15/16/17/31/32/33/63/64/65/127/128, retained full-row boundaries
and wide partial 5/33-row cases at 32768/131072 cells. Complete finite outputs
match original packed MMA byte for byte eagerly and in poisoned, changed-input
captured replay, including an executed last cache cell. One-query/128-row
controls, invalid cohorts/spans/scratch and >128-row packed fallback remain
positive. Independent adversarial source review is clean.

The first ordinary qualification found a fixture error before adoption: tails
at 1285/1541 positions have different padded KV widths (1536/1792), so the
adapter correctly runs them separately. Both new selection assertions failed;
heads, initialized state, histories and restart continuations still matched.
Those failure records remain preserved. The corrected tail control uses
1285/1413 positions, a 128-row skew with the same 1536-cell read width. Only
the two failed cases rerun; the 14 positive controls from the first run remain
evidence for unchanged binaries/source. Both families now prove selected
2–127-row owner nodes, exact complete finite heads and every used-state byte,
kept spill/restart adoption, history restoration and a scalar anchor
continuation. Aligned 128-row joined-hint controls remain unchanged.

The corrected fresh-server tail adapter's all-packed/on/on/all-packed means
are 0.389937619 / 0.3720590955 s (Gemma2, −4.58497%) and
0.492719524 / 0.4726112255 s (Gemma3, −4.08108%). These n=2 measurements
cover the combined 128-row-plus-tail owner policy and exclude the initial
scalar advance; they do not isolate the five-row kernel's benefit.

## Fresh 64-row references

Each family separately runs llama.cpp/native/native/llama.cpp with identical
64-row per-owner chunks, 128-row total batches, bounded C2/F16 KV, warmed
prefix/state/plans and the same paid final-head output work. Both engines execute 25 Gemma3 paid waves: 19 joined and six scalar. Their
shorter-remaining-owner-first, homogeneous-head schedule publishes the shorter
owner's frontier separately; the extra wave over 24 is shared, not evidence
for a native-only launch penalty. The public stock
helpers accept optional `chunk=N` (2–128, default 128) and preserve ordinary
stock graph/fusion/FlashAttention policies. Native proves zero paid optional
planning/eager work and the expected 77/25 waves. Current source, binaries,
SDK receipt, replay IDs/texts, source GGUF/artifact identities and actual
resolved libraries are bound before and after acquisition. Every target
positively retires; stock container absence is independently checked in a
finally path. All four arms per family match 64 valid natural choices and
both complete finite final vocabulary heads byte for byte. The 64-row runs
supply their own reference oracle, without inheriting a 128-row allowance.
The Gemma3 joint probe retains host masks and supplies no prefill hints;
ordinary serving already selects device masks and checkpoint-aware hints.
Its remaining reference gap therefore describes this diagnostic recipe,
not the complete ordinary-serving defaults. The matched owner-root/flexible
factors remain valid. The [bounded runner-policy alignment below](#bounded-gemma3-runner-policy-alignment) subsequently enables both established policies; full HTTP reference performance remains open.

| Family / arm, acquisition order | Prefill | Decode | Paid sum |
| --- | ---: | ---: | ---: |
| Gemma2 llama.cpp 1 | 1.797630 s | 0.584597 s | 2.382227 s |
| Gemma2 native 1 | 1.798790 s | 0.580849 s | 2.379639 s |
| Gemma2 native 2 | 1.798670 s | 0.581077 s | 2.379747 s |
| Gemma2 llama.cpp 2 | 1.797720 s | 0.581240 s | 2.378960 s |
| Gemma2 llama.cpp mean | 1.797675 s | 0.5829185 s | 2.3805935 s |
| Gemma2 native mean | 1.798730 s | 0.5809630 s | 2.3796930 s |
| Gemma3 llama.cpp 1 | 0.573535 s | 0.468713 s | 1.042248 s |
| Gemma3 native 1 | 0.587203 s | 0.474750 s | 1.061953 s |
| Gemma3 native 2 | 0.585101 s | 0.473940 s | 1.059041 s |
| Gemma3 llama.cpp 2 | 0.572521 s | 0.468464 s | 1.040985 s |
| Gemma3 llama.cpp mean | 0.573028 s | 0.4685885 s | 1.0416165 s |
| Gemma3 native mean | 0.586152 s | 0.4743450 s | 1.0604970 s |

Gemma2 is +0.05869% prefill and −0.03783% paid sum, level in this short n=2
comparison. Its decode change −0.33547% is smaller than stock's 0.57424%
bookend movement. Prefill stock/native drift is +0.00501% / −0.00667%.
Gemma3 remains +2.29029% prefill, +1.22848% decode and +1.81262% paid sum;
its prefill stock/native drift is −0.17680% / −0.35797%. That residual exceeds
this short comparison's movement and remains open. No Gemma3 parity,
sustained serving or all-context claim follows from the useful native factor.

At this task's entry, TensorFold native HEAD is
`f8fe17d24629aedabf90bbf78279dd776e6d62e7` (1.0.0), and Python 0.6 HEAD is
`ed78d6fc204d89d90b045bf033d6551e7714f3a1` (0.6.6). Their pinned README
recipes provide no qualified Gemma2/Gemma3 CUDA/GGUF comparator; llama.cpp
v0.6.0 remains applicable. This is an entry-time pin, not a claim that upstream
will retain it for later tasks.

The remaining T93 transfer includes >128 compatible rows, mixed-width roots
and other compatible cohorts/layouts. Source assessment finds no established
arithmetic barrier to larger equal C2 chunks: the foundation's 8192 total rows
could permit 4096 per owner, with 64 MiB Q/output, a 2 GiB mask at 131072 cells,
128 query tiles per owner and a roughly four-million flattened tile index.
These are derived bounds, not qualified support. Mask/parent-span validation,
funding, launch bounds, memory behavior and recipient performance still need
qualification. Mixed-width 1285/1541 tails currently use the correct scalar
fallback. Historical selector/span limits do not close either extension.

The experimental/final ordinary source manifests are
`cc8a50ce574e11cdc52958b92f3240e5e31778942ef08483decb830f8ef8a277` /
`5845d71f0654bb3a3b756b8ed3d1b4f77e09a8d4a8872b6bf8f78f4aabc7f07f`,
at base `b4ee2b9`. Aggregate qualification receipt SHA-256 is
`d96db2aefbf15fa81b6b6773ab1761e7ceeca2466bb8bd82345ca2b25ab56f0b`.
The flexible runtime/probe SHA-256 identities are:

| Payload | SHA-256 |
| --- | --- |
| Runtime | `decfa4c6b64b31f2b701d67a02bcc6840f4eb0774b8e9d286200f35c01936d01` |
| Gemma2 native probe | `552881e34618f423647e0b2b157e1a85299c5bd1e6935e5ff50c2f8ffdaa297e` |
| Gemma3 native probe | `ef94d0c68a25104b107781323ef8c2896e2babf03f0d9733a26a3a5cf8f5a1a7` |
| SDK build receipt | `874aaf7a5967cfbe91054e0d8fc1a0630f952e0e8eb54b831d09719f1e08ce89` |
| Gemma2 stock helper | `5782476b282c5e2e84d00f7830d894b902464b970ad06c241a314bb06b26c4ca` |
| Gemma3 stock helper | `498b33a980aa52da1fd75cd3556ab0be6db0302b7511783806adbb91e5540365` |

Raw XML, payloads, library/retirement bindings and failures remain external
under Spark B's `scratch/m35-gemma-flexible-prefill/` and the session
coordination scratch. Replay inputs remain in the standing reference store
above. Full regression/workstation tiers remain deferred.

To replay the new factor, append `chunk=64` to both native cycle commands
above and add `flexible-owner-prefill` only to the actual-root arm (together
with `owner-prefill`); omit both owner flags for the packed control. Compile
the checked-in [Gemma2 stock helper](../gemma2-serving/llama_joint_prefill_probe.cc)
or [Gemma3 stock helper](../gemma3-execution/llama_joint_prefill_probe.cc)
against the pinned original-image headers/libraries and append `chunk=64`
to `MODEL IDS0 TEXT0 IDS1 TEXT1 NEW_OUT cycle` inside that immutable image.


## One matched Gemma3 warm timeline

At source `698a077`, one CUDA-only Nsight 2025.3.2 pair on Spark B checks the
64-row diagnostic recipe above. It reuses the four completed unprofiled
controls and their unchanged qualified binaries, inputs, artifact, reference
helper/image and resolved cuBLAS payloads; the intervening source changes
are Markdown only. GPU: GB10, driver 580.178.04, SDK c09/CUDA 13.4. Both
profiled applications complete explicit teardown, return zero, publish the
same 64 valid choices and two complete finite vocabulary heads as all four
controls, and leave no GPU process; stock container absence is independently
checked. No new competitive speed claim comes from profiling.

Actual trace counts authenticate both paid phase boundaries: 40 two-row
vocabulary projections (eight warm, 32 paid decode), with two scalar prompt
frontier heads followed by six supplied scalar steps between them. Paid
prefill has 25 waves: 19 joined and six scalar. Native executes 827 attention
kernels (627 actual-root, 200 scalar); stock executes 850, retaining the
otherwise unused final-layer work in 23 state-only waves. Both use original
quantized product templates and query-tile geometries. Native has 22 prefill
replays and three captures beside eager execution; stock has 16 replays and
nine eager waves. Paid decode has 32 native replays versus stock's 31 replays
and one eager wave.

| Paid prefill GPU interval, milliseconds | Native | llama.cpp | Native minus stock |
| --- | ---: | ---: | ---: |
| First-to-last kernel span | 612.128896 | 603.528960 | +8.599936 |
| Kernel duration sum | 569.855352 | 569.078528 | +0.776824 |
| Kernel activity union | 563.321336 | 560.654592 | +2.666744 |
| Copy activity union within those bounds | 0.556352 | 1.862304 | −1.305952 |
| Combined activity union | 563.877688 | 562.516896 | +1.360792 |
| No kernel/copy activity | 48.251208 | 41.012064 | +7.239144 |

The native first input/mask copies and mask concatenations precede its first
kernel. Extending each interval from its first paid input copy through its
final prompt-frontier publication gives a +9.041088 ms native span difference:
+1.285816 ms combined GPU activity and +7.755272 ms inactivity. Previous warm
full-head publication and the following supplied scalar step's inputs are
explicitly excluded. No memset occurs within either paid phase. Kernel
duration sums can overlap under programmatic dependent launch; unions are
used for inactive time. Recorded replay-copy bytes are retained as raw
metadata, not presented as verified traffic.

| Native capture wave, zero-based | Owner / past / rows | Output | Global / local read cells |
| --- | --- | --- | --- |
| 19 | slot 0 / 1216 / 64 | Greedy frontier | 1280 / 1280 |
| 20 | slot 1 / 1216 / 64 | State only | 1280 / 1280 |
| 24 | slot 1 / 1472 / 64 | Greedy frontier | 1536 / 1280 |

These three shapes each occur only once during warm prefill. The second-use
`PlanRuns::CaptureDue` policy therefore captures them during the paid pass;
there is no evidence of graph loss. Slot 1's following recurring 1536-cell
state-only shape captures during warm wave 22 and replays in the paid pass.
The three paid captures contribute 5.161 ms of native intra-wave inactivity
versus 0.927 ms in their stock counterparts. Across all 25 waves, intra-wave
inactivity is 17.411/4.529 ms (native/stock), while inter-wave inactivity is
31.619/36.744 ms. Thus native already saves inter-wave time; the data do not
support a generic scheduler-wake explanation.

The remaining replay gaps are distributed: quantize-to-MMQ and MMQ-to-fixup
transitions contribute about 3.78 and 2.98 ms more kernel-to-kernel gap than
stock, mostly submicrosecond gaps; the largest individual intra-wave gap is
16.6 microseconds. These transition sums can include copy activity and are
not independent critical-path savings. No cause or safe optimization is
established. The profiled decode copy-inclusive difference is +28.303 ms,
substantially above the unprofiled difference of about 5.8 ms, so it is
instrument-sensitive evidence, not a revised performance result. CPU CUDA
API interval sums overlap work and do not measure critical-path overhead.

This one pair closes the attribution experiment without closing the
diagnostic residual. Complete ordinary-recipe comparison, compatible larger
query tiles and mixed-width roots remain explicit work. In particular, the
probe's host masks and absent future hints must not be mistaken for ordinary
serving's already-adopted GPU masks and shared lookahead. No extra inference,
full regression or trace ladder ran.

Replay uses the authenticated Gemma3 standing inputs, probes and immutable
reference image identified above, with `cycle bounded-roots owner-prefill
flexible-owner-prefill chunk=64` for native and `cycle chunk=64` for stock.
Profile each whole process with `--trace=cuda --cuda-graph-trace=node
--sample=none --cpuctxsw=none --duration=90 --stop-on-exit=true --wait=all
--kill=sigterm --export=sqlite`. Verify full application retirement and payload
equality before interpreting traces; use the projection counts and actual
post-head tails above to isolate the paid prompt and decode. Raw captures
and the one-off analyzer remain outside Git under Spark B
`~/scratch/m35-gemma3-warm-timeline/run1` and coordination scratch. The source
manifest is `c97f53510381286a7b514fce75a20df666b9322dc3e1ff2ff6b489e386f53356`;
native/stock SQLite SHA-256 identities are
`10a2ac94092737383abed1df492dd4f8305b3bf2a8723967e443f722f5f7a96d` /
`529b4f6baf0a3f733e34be35375e10a39a16ecc53318dc9f8fe20ef50317eb2e`.
TensorFold's two task-entry refs remain the exact native/Python pins above,
with no applicable Gemma3 CUDA/GGUF recipe.


## Bounded Gemma3 runner-policy alignment

The joint probe now accepts explicit `device-masks` and `prefill-ahead`
controls, retaining the flag-absent host-mask/no-hint diagnostic route. It
passes the same two-stage hint descriptors as the checked Gemma2 probe to the
shared runner lifecycle, with ended-owner filtering, independent next/after
head modes and actual selection/cache/capture counters. Future GPU-token
frontier stages remain suppressed: the existing hint contract describes
full-head or state-only work. This compares bounded runner policies; ordinary
HTTP serving publishes full prompt-frontier heads and is a separate gate.
No model/kernel/runner implementation or public default changes.

One acquisition on Spark B runs reference/old/aligned/aligned/old/reference
with the same 64-row/C2 inputs, warm-retained backing/plans/graphs and paid
work described above. Aligned adds both new words, exercising optimizations
already selected by serving. Every arm matches all 64 valid choices and
both complete finite final heads byte for byte, including the retained
oracle. Each application completes teardown and returns zero; stock
container absence is independently checked and all final guards are clean.

| Arm, acquisition order | Prefill | Decode | Paid sum |
| --- | ---: | ---: | ---: |
| llama.cpp 1 | 0.579273 s | 0.467327 s | 1.046600 s |
| Old diagnostic 1 | 0.584664 s | 0.473546 s | 1.058210 s |
| Aligned 1 | 0.578964 s | 0.473448 s | 1.052412 s |
| Aligned 2 | 0.575162 s | 0.471310 s | 1.046472 s |
| Old diagnostic 2 | 0.584530 s | 0.473188 s | 1.057718 s |
| llama.cpp 2 | 0.572693 s | 0.468148 s | 1.040841 s |
| llama.cpp mean | 0.575983 s | 0.4677375 s | 1.0437205 s |
| Old diagnostic mean | 0.584597 s | 0.473367 s | 1.057964 s |
| Aligned mean | 0.577063 s | 0.472379 s | 1.049442 s |

Aligned prefill is 1.289% faster than the same-binary diagnostic control and
0.188% above fresh stock, within stock's 1.136% bookend movement (aligned
movement 0.657%, old 0.023%). Aligned decode is 0.992% above stock; paid sum
is 0.548% above stock, comparable to stock/aligned paid movement
0.550%/0.564%. At n=2, these are descriptive results: bounded prefill is
level within the observed movement, with no full HTTP or universal parity
claim. The combined policy factor improves paid sum by 0.806%; its small
0.209% decode movement is not a demonstrated benefit. It does not isolate
the contribution of masks from hints.

Aligned selects 40 GPU-mask plans, caches five lookahead plans and captures
five graphs ahead; old selects none. Both have 25 paid prefill waves, three
captures beside eager execution and 22 replays, with zero paid optional
planning/cache/capture-ahead or eager-path counts. The three one-off
token-frontier/state-only shapes discussed above still capture on second
use. The earlier diagnostic comparison does not establish a
production-default kernel gap; the larger-row factor starts from the
explicitly aligned policy.

Only the probe target was rebuilt; no regression suite or additional trace
ran. Compiled source base is `c382965`, manifest
`ce19d9ca6b20d9756ea0c048769edca54f573f73d4a3281c8fdd1dec615e8b9e`;
probe source SHA-256 is
`fea690c51d5d3d69436e354ac7987a7c778060ff25ace383dbb15227eeeb514c`,
ELF `ca82b6739de3f031eec6aa2d7fad651111724c3293840476cc275a7f14c4fb8d`.
SDK receipt, artifact/GGUF, input, immutable helper/image and actual cuBLAS
identities remain those above and are checked before/after acquisition.
TensorFold's task-entry native/Python refs remain unchanged without an
applicable recipe. To replay, add `device-masks prefill-ahead` to the native
64-row cycle invocation above; omit both for the diagnostic control and
retain reference `cycle chunk=64`. Raw build, six-arm payload/retirement
records and aggregate calculation remain external under Spark B
`~/scratch/m35-gemma-prefill-wide/alignment-build1` and `alignment1`, plus
coordination scratch. Replay inputs stay in the standing reference store.
