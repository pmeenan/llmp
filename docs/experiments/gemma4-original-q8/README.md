<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma4 original-consumer Q8 sharing and bounded startup placement

This slice transfers the original-MMVQ Q8 input cache to the approved Gemma4
26B-A4B and dense 31B targets. It preserves every checked full head, generated
ID and initialized state. Short C2 samples show less decode time by 0.310%
for 26B and 0.087% for 31B; the 31B whole paid difference stays within ordinary
bookend movement. Startup remains materially slower with sharing. These are
native factors, with no stock parity, assistant or HTTP performance claim.

The option remains **off by default** for both profiles: the modest recurring
decode change does not justify blanket selection given the measured startup
cost and neutral 31B whole paid sample. No C4/8K default or assistant adoption
run is claimed.

Two mechanisms are separately tracked: T11 shares selected product inputs;
T96 avoids unnecessary greedy activation placement during repeated Setup
measurement. T96 preserves the exact activation maximum rather than replacing
it with an inflated bound.

## Consumer and policy

The new `dense_shared_q8` option retains original GGML MMVQ consumers and their
512-element Q8_1 padding. The per-graph cache keys the exact F32 input tensor.
The transient device selector admits only existing MMVQ routes through eight
columns, in the nine already-qualified formats. Empty/false callbacks, MMQ,
row-invariant policies and unsupported shapes retain ordinary products.
C1 fused GeGLU, single-consumer down/head products and routed expert math stay
ordinary. State-only tied-KV final layers keep their singleton K product
ordinary. The old `shared_q8` VecQOneToken diagnostic takes precedence if both
options are requested and keeps its prior arithmetic/vector-float policy.

The G2/G3 row-invariant contract receives the same suppression guard in both
sizing and final construction. A compatible prepared row-invariant consumer
remains open; current original-consumer sharing cannot silently replace its
reduction. Existing nine-format operand evidence is retained, without another
arithmetic ladder.

## Placement measurement and funding

Setup retains its ordinary endpoint measurements. For every supplemental
positive ordered owner-row composition through eight total rows, the same
selected graph, plan, lanes, scratch and sources are constructed. Checked
256-byte-rounded storage of unique input leaves and computed non-view roots
bounds greedy placement. When this disjoint sum is at most the already-proven
exact maximum, placement is unnecessary. Otherwise exact placement runs on
that same graph and updates the maximum. Endpoint ordering is per configured
row-budget/owner envelope; no global monotonic dispatch assumption is made.

All storage multiplication, addition, rounding and address ends are checked,
including noncanonical blocked strides. A typed measurement marker cannot
bind or execute: real activation addresses and `BindPlanned` reject it.
Scratch, host inputs, plan/index accounting and computed-node coverage remain
exact. Ten host controls exercise both threshold branches, malformed spans,
overflow, runtime refusal and all three family row-invariant guards.

The actual-device descriptor oracle checks 3,098 supplemental comparisons,
including head/frontier/state/greedy modes, context/cache endpoints and Gemma4
retained-feature rows. It loads no model weights and runs no model arithmetic.
Every comparison preserves plan/source/scratch/staging/host metadata and the
final exact activation maximum.

| Recipient | Comparisons | Placement skipped | Exact fallback | Activation maximum bytes |
| --- | ---: | ---: | ---: | ---: |
| Gemma2 | 285 | 193 | 92 | 59,854,336 |
| Gemma3 | 285 | 18 | 267 | 36,836,608 |
| Gemma26 | 1,264 | 1,264 | 0 | 382,953,216 |
| Gemma31 | 1,264 | 288 | 976 | 326,264,576 |

A single current 26 correctness run observes Setup 1.045500s, versus 1.396780s
in the earlier sharing run, with identical full outputs/state/work and every
budget. These are n=1 startup observations from separate runs, not a matched
startup factor. The ordinary earlier run took 0.332088s. Remaining graph
construction/selection work is still a compatible optimization lead. Source
shows that the current C2 envelope adds 204 planning calls and 408 graph
constructions (one sizing and one final build per call), even when placement
is skipped. This is a count inferred from the configured loops, not a timing
attribution or proof that either stage dominates. No
startup gain is inferred for the other recipients from descriptor counts.

## Matched native factors

Both profiles use context 4096/C2, prompt lengths 256/768, three independent
untimed seed rows per owner, then 32 joined decode steps. Chunks are 1024 for 26
and 256 for 31. A complete off/on own pair checks full finite vocabulary heads,
64 choices, conditioning histories, initialized state and GPU/host,
checkpoint/spill/restore/protected-peer/refusal behavior. Own intervals include
observation and are excluded from timing factors.

Warm O/A/A/O cycles retain backing/plans/graphs across Clear. Paid prefill
includes inputs, masks, state and actual capture/replay; paid decode includes
GPU greedy selection and final head publication. Startup, model loading,
warm traversal, Clear, seed rows and observation/persistence are excluded.
The paid sum combines two intervals, not a continuous HTTP request.

| Profile / arm | Prefill seconds | Decode seconds | Paid seconds |
| --- | ---: | ---: | ---: |
| 26 off first | .343500 | .752808 | 1.096308 |
| 26 on first | .340699 | .750847 | 1.091546 |
| 26 on repeat | .341640 | .750238 | 1.091878 |
| 26 off bookend | .341622 | .752950 | 1.094572 |
| 31 off first | 1.302280 | 3.330740 | 4.633020 |
| 31 on first | 1.304830 | 3.325150 | 4.629980 |
| 31 on repeat | 1.299210 | 3.328910 | 4.628120 |
| 31 off bookend | 1.300540 | 3.329120 | 4.629660 |

26 mean prefill/decode/paid changes are −0.406%/−0.310%/−0.340%, versus
ordinary bookend movements −0.547%/+0.019%/−0.158%. Decode and paid ranges
are disjoint. 31 changes are +0.047%/−0.087%/−0.049%, versus movements
−0.134%/−0.049%/−0.073%. Its decode ranges are narrowly disjoint; paid ranges
overlap. Each policy has only two samples. No sustained performance or
whole-model causal claim follows.

26 cycle Setup is .312433/.312000s off and 1.413660/1.397290s on (before the
placement shortcut); 31 current Setup is .370212/.369269s off and
1.655650/1.656700s on. Sharing raises plan floors by 52,800B/105,600B;
activation, scratch, host-input and total derived budgets remain identical.
Bound-plan selections are 60/145 Q8 producers/prepared consumers for 26 and
120/290 for 31. These are selections, not executed kernel counts. Paid cycles
have 34/36 units, captures 2/4 and replays 32, with 88 GPU token picks; work and
other selected policies are identical across each factor.

All acquired applications return 0 after explicit teardown, with positive
installed waits, no residual GPU processes and no new NVRM/Xid/UVM/OOM evidence.
31 own-off records two unrelated Ubuntu Pro AppArmor perfmon audit denials;
they do not match the predeclared driver-error refusal expressions. All other
31 arm kernel observations are empty. Failed prerequisite/compiler/oracle
attempts remain failed and unpooled; only positively completed controls carry
through exact source/ELF bridges.

## Remaining recipients and scope

T96 is selected for G2/G3 supplemental measurement and qualified for the
optional G4 dense-sharing measurement path. DeepSeek and Qwen native/GGUF
Setup also repeatedly account exact placement maxima and are compatible OPEN
recipients, even without Gemma supplemental compositions. Their latency and
actual maxima need qualification. Image Setup instead takes maxima of three
fixed text/DiT/VAE layout sizes and has no analogous GGML greedy placement to
skip. Larger-column/MMQ-adjacent and prepared row-invariant consumers remain
open separately from T96. Assistant adoption needs its feature/state math
controls; target exactness does not qualify assistant rounding.

## Source, replay and standing inputs

The initial 26 factor binds source1 `8f8ec1d63b5dbac2d5e899f103bf7b88f4c409dd586add215d5867830e3e2399`
and probe `f2815363ccc7236a327c5e0ae4d93212c457956973db3416bd6b09f08171dac1`.
[Aggregate identities](results.json) retain the complete proof bindings.
The current 31 factor binds source5
`c53707beea479d6bb59b2ea8369194980a36669744bb7471faf3ed1b3c9ae8c7`
and probe `68babcf5f16fd36cfe4ba5aad27993b7deecdbcd82722c319fc20bbb96d041c5`.
Final integration carries these changes additively onto 60d0c30, preserving
Gemma2/Gemma3 state-preparation hooks. Only their Setup measurement regions
change from that main; the other 22 qualified paths remain byte-exact. All
seven code/test/benchmark paths from the Gemma2 adoption remain present.
The current integration build compiles runtime, probe and shared serving
control, then executes exactly the existing default-preparation adapter case
for G2 and G3. Both compare explicit preparation off against unset ordinary
default while original-MMVQ Q8 sharing remains selected. Complete finite heads,
initialized-state hashes, histories and joined progress agree; G2 adopts 416
prepared extents (33 joined groups/8,448 rows), G3 adopts 136 (9/2,304).
Both explicit server teardowns succeed. Three children return 0 and installed
job 3247240 reports DONE0, with empty kernel-error observations/stable boots/no
GPU leftovers. This is a combined Setup/state-hook correctness check, not a
new G4 factor or ordinary G4 default-adoption claim. The full focused positive
union is 17 unique names: six initial graph/plan cases, ten funding/guard cases
with two overlaps, one descriptor oracle and these two integration cases.

Aggregate identities and output/state digests in results.json are independent
of external raw logs, which may be removed when the milestone closes.

Replay uses the checked-in [common-width probe](../../../benchmarks/gemma4_common_width_probe.cc),
[graph controls](../../../tests/unit/gemma4_graph_test.cc),
[host placement controls](../../../tests/unit/ggml_graph_test.cc),
[plan controls](../../../tests/unit/gemma4_plan_test.cc) and
[actual-device descriptor oracle](../../../tests/unit/gemma_measurement_gpu_test.cc).
Build current targets with the official Spark preset and authenticate the
source, ELF, build receipt and resolved libraries before/after use; historical
raw build receipts are not prerequisites to a fresh build. The locked SDK is
`aarch64-c09daba6ac31edee`, GGML comes from llama.cpp
`d81235049384534c167caea52b85a694f6103d14`; Spark B uses GB10/driver 580.178.04.
The fixed private cuBLAS13/Lt payloads are
`ee7c1657a03695c0de790aa79e34cef9c9649756b1846b11dd44caca20ba656b` and
`ba3b942f4ea43433b65e8c492a7b73de887534dc20146506ddaa4a78c79c5d30`.

Approved prepared artifacts are supplied externally from
`~/.local/share/jitllm/m3-artifacts/`:

| Profile | Manifest SHA / directory | Index SHA |
| --- | --- | --- |
| 26 | `4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3` | `e748198836025cc2d1bc8b0b61a1d39dc1eefdc2d34feb158a425f6f913f1171` |
| 31 | `32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08` | `9ae365a17ed73528262d7befdb505bf07e6695e03f0fa36f69e69099dbd5f34c` |

Standing replay inputs are KEEP data, separate from raw experiment output,
under `~/.local/share/jitllm/references/gemma4-original-q8/`. Supply these exact
little-endian I32 bytes externally if the store is absent; no retokenized
substitutes. `parent-histories.i32` is 12 owner-major histories of 1024 IDs,
49,152B, SHA `d584450079145f3d2c93f46ef24a0aabe2b3971279a1cbddbbb29f0a506ac5e3`.
`input0.i32` is its first 291 IDs at byte offset 0 (1,164B), SHA
`622251ed82cd99c4bdd6add8642309986c646da7f2b7c10472c0b8feb2ee59de`.
`input1.i32` is owner 1's first 803 IDs at byte offset 4096 (3,212B), SHA
`0e9eefbac4c2199196dc8934a5fcd7ae304bfda13d4f353591193edca43998dc`.
All IDs are in 0..262143 and begin with BOS 2; the two histories differ.

Invoke the probe with
`ARTIFACT INPUT0 INPUT1 OUTPUT_DIRECTORY {26|31} bounded {own|cycle}`;
append `dense-shared-q8` only for on arms. Its absent flag preserves ordinary
behavior. Run own-off/on first, comparing complete finite predecode 2,
teacher 66 and final 2 vocabulary rows, 64 choices and exact initialized state.
Then acquire four fresh cycle lifetimes off/on/on/off, comparing full final
heads/choices/histories/state/work and budgets. Keep own diagnostic intervals
out of performance samples. The descriptor oracle filter is
`GemmaMeasurementGpu.AllConfiguredCompositionsPreserveExactFunding`.

Every heavy acquisition uses the installed `spark-job` GPU lock, 600s deadline,
30s grace, bounded children and positive installed wait. Require explicit
successful node teardown plus return 0, no remaining compute processes,
MemAvailable>=32GiB, stable boot identity and a per-arm journal cursor refusing
new NVRM/Xid/UVM/out-of-memory/oom-kill evidence. Preserve failed batches;
never pool their timing samples into a successful factor. Host-heavy analysis
uses the workstation shared hostlock. No historical archives or deleted raw
folders are required by this replay recipe.

Task entry 2026-10-08T12:28:56Z authenticates TensorFold main
`f8fe17d24629aedabf90bbf78279dd776e6d62e7` and Python
`ed78d6fc204d89d90b045bf033d6551e7714f3a1`. This is a same-native factor,
not a new TensorFold or stock comparison. Broader serving/reference quality
requirements remain open; no previous accepted quality tolerance changes.
