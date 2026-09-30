<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek frontier head

Production prefill returns its frontier logit row. DeepSeek's previous
graph nevertheless gathered every row before its final hyperconnection
mix, output normalization and vocabulary product. Qwen3.8 already selects
the requested frontier before its head while exporting all MTP streams.
The [ds4 study](../ds4-study/README.md) identified the same separation:
its matched first-4K community Q8 profile spent about 65 ms in the vocabulary
head, roughly 1% of GPU work, and retained the full-row logits. That head
format is distinct from the original Q4_K model measured here.

`Dsv4ChunkShape::outputs` now names the requested trailing head rows;
zero preserves the default all-row graph. Fast production prefill opts
into one row through `Dsv4Options::frontier_head`, restricted in serving to
the measured Q4_K `[4096,129280]` head and F32 `[16384,4]`
hyperconnection product. Community Q8 heads remain all-row automatically.
Generic runners and
graphs default to all rows. Reference, verify, full-window diagnostics
and named diagnostic plans retain every requested head row. PPL stays
all-row. One-row decode and its captured graph keys are unchanged.
The final component selection keeps CSA/window sharing and returns
count-based HCA masks to the registered ordinary MMA implementation.
Fresh 32K/128K repeats and matched PPL qualify this body/head combination
at the existing bounds. Final maximum-context runtime gates
remain separate and do not close the whole M3 gate here.

Selection occurs after every target layer and after the full feature
streams are captured. DSpark still receives all three target streams'
means for every chunk row, and its requested last injection rows. The
same head weights, hyperconnection formula and normalization run; no
cache or weight precision changes. Selecting one row changes product
dispatch and reduction order, so agreement with the old all-row head
must be measured rather than assumed bit exact. Each shape's own repeats
and restored continuation must remain exact.

At vocabulary 129,280, the declared F32 logit tensor is 1,059,061,760
bytes for 2,048 rows and 2,118,123,520 bytes for 4,096 rows; one row is
517,120 bytes. This tensor reduction is not a measured workspace or
physical-peak reduction. Placement, full DSpark stream lifetimes and
late sparse-mask/indexer/attention work may set another highwater.
Intermediate prefill chunks still return one frontier row; dropping
their heads entirely would require another caller contract.

## Controls fixed before measurement

- Use the unchanged original-DeepSeek oracle near-tie bound of 0.947.
  Report same-input head mixed/norm/logit errors and margin movement
  before the oracle comparison; do not derive a wider bound from it.
- Run each isolated head shape twice over identical captured real
  pre-head stream bytes, requiring exact repeats. Record its selected
  implementations and placement/scratch separately from model timing.
- Force a common continuation in the original model controls.
  Require exact target state and DSpark feature/injection controls,
  followed by fresh-repeat and swap/rollback controls.
- Keep PPL, teacher-forced scoring and verify output counts unchanged.
  Include partial final chunks and production 2,048-row chunks.
- Measure actual 1M production workspace sizing without another full
  maximum-context model run; physical-fit and long-context quality
  remain separate gates owned by the final context study.

The community Q8 frontier path remains an explicit harness experiment.
It has not been measured in this follow-up and is not enabled in serving.

## Harness

`jitllm_dsv4_exec --frontier-head` selects the final head row for prompt
prefill and `--bench-prefill`; PPL and named dumps stay all-row. The
experimental flag is off by default. `jitllm_spec_runner
--frontier-head on|off` drives the same production runner choice for
DSpark state, rollback and swap controls.

`jitllm_spec_runner --check frontier` runs all-head, frontier, frontier,
all-head prefills with production last-row copies, both with and without
DSpark injection. Timing stops before diagnostic state readback; raw
target/ring state and the common forced continuation must match exactly,
and each head shape repeats its own full logits exactly. This diagnostic
retains host snapshots and is capped at 131,072 context, 4,096 chunk rows
and 1,024 outputs,
with prompt plus outputs within context. It is not a maximum-memory test.
`--check sizing` records planned production allocation sizes without
loading or running model weights. Its `weights_loaded` field is false;
`read_bytes` and `drafter_read_bytes` describe the registered payloads,
not disk reads in that sizing invocation.

`jitllm_dsv4_exec --probe-head --prompts TSV` captures the first prompt's
final chunk's complete `l_last` streams, then runs only the original
builder's head suffix at all rows and at one row over those same bytes.
It repeats both, writing their last mixed/norm/logit rows, the input
streams, and `head.json` with actual paths, placement and scratch. It
does not run target state nodes or duplicate the head formula. Extra
diagnostic allocations, retained streams, copies and probe work make
that invocation unsuitable for model performance/memory claims.

Qwen3.8 already implements the same separation: its selected head rows do
not narrow the exported MTP streams. The Qwen2 GGML fixture already gathers
requested output IDs, though its current envelope requests all rows.
EXL3's fixture keys casts and linear launches by the full phase row count,
so transferring frontier selection there needs a separate output-envelope
contract. Image text encoding retains every conditioning token rather than
an autoregressive vocabulary frontier; dropping those rows is ineligible.

## Head arithmetic and quality

On `spark-b`, 2026-09-30, the original Q4_K model uses the same weights
and F16 state/cache precision in both arms. The all-row suffix selects
cuBLAS for the hyperconnection product and MMQ for the vocabulary
product. The frontier suffix selects the F32 row-vector product and
Q8_1 preparation plus the existing quantized vector product. Both shapes
repeat their mixed, normalized and complete logit rows bit exactly.
These initial head/no-added-divergence measurements used wide attention
and an experimental native compact-tail policy. The corrected native
model harness now matches production's 2,048-row compact floor, using
the shared `kDsv4CompactMinRows` constant. The earlier native flag had
also selected compact products on partially filled tail chunks. The
31,705-ID fixture ends with 985 rows. For 128,821 IDs, both the native
harness and HTTP's `RunPrefillChunks` round chunks of at least 1,024
rows to an eight-row tile, so they end with 1,840 plus five rows. The
direct runner diagnostic does not round and separately exercises a
1,845-row tail. Smaller direct operation/micro controls remain available. The
initial captures are retained as experiments. Faithful planner controls
are recorded separately below; they reproduce the 32K wide-path case.

| Real input | Final chunk | Max absolute logit delta | Mean logit offset | Top-two margin movement | Softmax TV | KL, all to frontier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First 8,192 IDs of the coding prompt, same captured streams | 2,048 | 0.175148 | −0.016214 | −0.007881 | 0.00009736 | 0.000000698 |
| First 8,229 IDs, same captured streams | 37 | 0.364998 | +0.063355 | −0.005489 | 0.00000551 | 0.000000074 |
| Actual 31,705-ID prefill, first output row | 985 | 0.106270 | +0.011246 | −0.002010 | 0.00056475 | 0.000001341 |

Argmax and runner-up are unchanged on all three inputs. The first two
rows isolate the head over identical real pre-head streams; the third
compares complete target runs. The 32K fixture's first row differs from
the pinned llama.cpp argmax within its oracle margin of 0.725294, below
the unchanged bound of 0.947. The following 511 teacher-forced complete
logit rows are bit identical between all-head and frontier; a fresh
frontier run repeats all 512 rows exactly.

The experimental compact-tail 32K oracle comparison does **not** pass: all-head, frontier
and fresh frontier each agree at 493 of 512 rows, with 18 within-bound
near ties and one shared violation at step 249. Native token 10386
differs from oracle 82437, whose oracle margin is 2.616249, above 0.947.
Both prompt IDs and all 512 forced IDs exactly match that oracle.
The new head adds no disagreement. This initial comparison used a compact
tail choice different from serving, so it alone did not qualify the
production body.

A corrected-floor, wide-attention, all-head control uses the same 31,705
prompt IDs, 512 forced IDs, oracle and bound. It again has 493 equal rows,
18 within-bound near ties and the same step-249 violation. Its complete
row-249 values are unchanged: `logit[10386] = 38.968403` and
`logit[82437] = 38.382545`, a native lead of −0.585857 for the oracle's
token. Thus the harness's earlier compact-tail mismatch does not explain
this 32K disagreement. This is a successful execution with a failed
oracle gate; no complete production-quality pass is claimed.

Disabling compact scheduling in that same corrected-floor, wide,
all-head configuration produces all 512 complete F32 logit rows bit
identically (SHA-256 `5d81633784f55b2feb5fa75bd4d05519175b4506a0998d5c70080929b5f6acba`).
It preserves the same oracle counts and step-249 values. Prefill is
57.5741 seconds without compact scheduling versus 53.4019 with it;
decode is 24.4221 versus 24.4059 seconds. Compact scheduling is therefore
not the cause of this fixture's disagreement. This does not recreate
the phase-2 source, which also precedes the shared attention changes.

An external component control returns only count-based HCA sparse
masks to the original one-query path. CSA top-k and raw-window attention
retain wide sharing. It has 491 equal rows, 21 within-bound near ties
and zero outside-bound rows. At step 249 the native lead for oracle
token 82437 is +0.184032. Completed multirow implementation calls are
336 wide CSA, 320 ordinary HCA and 32 wide window; completed one-row
calls are 10,731, 10,220 and 1,022 respectively, with the same selected
implementation identities. The one-row launch remains the original
one-query kernel. Prefill is 58.3449 seconds, 9.26% more time than the
current wide control; decode is 24.3891 seconds. A corrected-floor
128K/512 all-head arm has 500 equal rows, 12 within-bound near ties
and zero outside-bound rows. At step 315, `logit[295] = 32.960655` and
`logit[1492] = 32.146660`, lead +0.813995 for the oracle's token.
Completed multirow calls are 1,344 wide CSA, 1,280 ordinary HCA and
128 wide window; one-row counts are unchanged. Prefill takes 260.1947
seconds and decode 25.7668. The selection uses existing mask semantics,
without a context, token or oracle-dependent dispatch rule. Its final
planner uses the distinct registered ordinary implementation for HCA,
so placement, scratch and launch identity agree. Target and DSpark use
the same planner; DSpark's pinned three blocks are window-only and keep
sharing. Completed-call counters belong only to the archived diagnostic
binary, not the serving path.

Fresh all-head and frontier runs each repeat all 512 complete logit rows
bit exactly at both depths. Every arm retains the same oracle counts
above, with zero outside-bound rows. The following 511 rows are bit
identical across head choices. At 32K, all-to-frontier first-head maximum
absolute delta is 0.108030, mean offset +0.023485 and centered maximum
0.084545; tokens 671/2581 remain the top two, with margin
0.134235 to 0.132559. Softmax TV is 0.00046499 and KL is 0.000000788.
The oracle prefers 2581 by 0.725294, within 0.947. At 128K, all 512
rows are bit identical across head choices: the five-row final chunk
already uses the same vector arithmetic. Its first argmax equals oracle
671, leading 2581 by 1.502068.

| Fixture | Configured context | All-head prefill / decode, s | Frontier prefill / decode, s | Oracle rows: equal + near ties |
| --- | ---: | ---: | ---: | ---: |
| 31,705 prompt IDs, 512 forced rows | 32,768 | 58.3449 / 24.3891; repeat 58.9588 / 24.4710 | 58.0528 / 24.4663; repeat 58.0352 / 24.5096 | 491 + 21 |
| 128,821 prompt IDs, 512 forced rows | 262,144 | 260.1947 / 25.7668; repeat 260.2921 / 25.7459 | 258.7562 / 25.7350; repeat 257.7364 / 25.7321 | 500 + 12 |

These resident native times include their respective logit copies and
are not the production head-speed comparison. Matched HCA-body PPL over
65,535 second-half tokens is 1.926517 versus 1.9298 (−0.1701%), within
the existing 3% bound. Its 131,072 input IDs use 64 full 2,048-row chunks;
all requested scoring heads remain present.

The component selects the registered ordinary MMA implementation for
HCA, retaining its original shape selection. At the measured 32K/128K
shapes that is one-query gathered sparse attention. When the visible
HCA prefix fills enough of the allocated KV rows, its existing sparse
eligibility test instead selects dense attention. No fixed one-column
claim is made for every maximum-context chunk.

The quality-valid component costs prefill time. A fresh same-config
128K wide all-head control takes 231.8348 seconds versus 260.1947 /
260.2921 for ordinary HCA, **12.23% more prefill time**. Both use the
same 128,821 IDs, context 262,144, 2,048-row chunks, corrected compact
floor and all-row copies. The wide timing run requests one output;
the full forced comparison requests 512, so this row compares only
prefill. The faster all-wide 32K path fails the unchanged oracle bound.
The qualified HCA resident prefill is about 495 tok/s at 128K, versus
the pinned historical llama.cpp 258 tok/s; final serving throughput
and maximum-context timing remain pending. The 32K time cost is 9.26%.

The archived phase-2 records `m3lc/raw/p2-f32k-fast{,2,3}` used capacity
33,280 and 2,048-row chunks, with the same original artifact, 31,705
prompt IDs and 512 forced rows. The current corrected-floor control
uses capacity 32,768. Production commonly uses capacity 262,144. Source
inspection shows all three use the same 2,304-cell window ring and the
same visible compressed-prefix shapes at these positions. Configured
capacity changes the allocated compressed-cache extents, dummy-row
indices, tensor metadata/cache identity and addresses, but does not
enter YaRN parameters or the visible attention/indexer reduction sizes.
This is source evidence, not a measured cross-capacity bit comparison.
Phase 2 predates the shared-wide/paired preparation at `66d5b91` and
compact scheduling/tail allocation correction at `e221aa8`; disabling
compact scheduling alone does not recreate the phase-2 binary.

A diagnostic-only `--wide-sparse off` retains all other fast choices,
compact expert scheduling, all-row heads and the same fixture/oracle.
It produces 493 equal rows and 19 within-bound near ties with zero
violations. Eighteen argmax rows differ from wide attention. At step
249, native `logit[82437]−logit[10386]` changes from −0.585857 with wide
attention to +0.096939 with the original sparse path. Thus the shared
violation changes with attention within that compact-tail experiment;
the bound and oracle remain
unchanged. Exact mode always uses the primitive regardless of the
diagnostic override. A proposed all-depth narrow replacement is not
adopted: the matched 128K compact-tail narrow experiment has one
outside-bound case at step 315, native 1492 versus oracle 295 with
oracle margin 0.970619. Both fresh frontier runs and the all-head
control share that row, and all 511 post-first logit rows match exactly.
Native `logit[295]−logit[1492]` is −0.631413 in narrow and +0.747059
in the earlier wide capture. The all-narrow proposal is not adopted from
those experimental-tail captures. The final qualified component instead
uses ordinary HCA with CSA/window sharing. `--wide-sparse on|off` records
the harness's explicit sharing permission; HCA remains ordinary under
either setting in the final planner. Generic/reference defaults remain
unchanged.

The initial wide-path actual 128,821-ID frontier prefill's first output equals the pinned
128K oracle argmax, token 671. Its lead over runner-up 2581 is 1.439474.
Prompt IDs match exactly. This is a changed-head first-row check, not a
new 512-row repeat or PPL result. The earlier compact-target 128K
512-row repeat and PPL 1.927387 versus 1.9298 precede this head change;
those are historical wide-attention results. A fresh narrow PPL control
scores 65,535 second-half tokens at 1.927517 versus 1.9298, within 3%.
Its input has 131,072 tokens (64 complete 2,048-row chunks), so it has
no partially filled compact tail. This narrow experiment is separate
from the final ordinary-HCA/shared-CSA-window component and from its
fresh head quality gate.

The captured softmax distances bound only these real first-head rows,
not every possible prompt's distribution. Target state and later
conditional logits are unchanged in the controls, both plain and
speculative decoding use the same target head, and sampler/verification
arithmetic is untouched. The initial head-only scope retained the earlier
finite sampled control. The final HCA change also changes attention-body
arithmetic, so fresh plain/speculation checks ran on `spark` on
2026-09-30, 04:54–05:11 EDT. The existing capital, haiku, sky and Fibonacci
prompts each contribute 256 seeds × eight outputs: 8,192 samples per mode.
All four pooled top-16-plus-other TV distances pass the unchanged 0.1 bound:

| Prompt | Plain/speculation TV |
| --- | ---: |
| Capital | 0.0034 |
| Haiku | 0.0098 |
| Sky | 0.0186 |
| Fibonacci | 0.0112 |

Both processes use the final counter-free HCA/frontier planner, F16 caches,
`context = 262144`, 2,048-row chunks, graphs on, the fast plan and the
original target/DSpark artifacts. Sampling is temperature 1 with no
top-k, top-p or min-p truncation. Counts are checked at 2,048 per prompt
per mode, and both runs report no problems. This finite marginal protocol
does not establish equality of every conditional or joint distribution.
Plain/speculation process times are 583.55 / 412.21 seconds; they include
the protocol workload and are not runtime throughput comparisons.
The spec-runner SHA-256 is
`81954b884ec317ddf0536851895aface44f896282167c1219c4f8340e6a91008`;
the final planner SHA-256 is `1355f003…` as recorded below.
Prompt-fixture SHA-256 is
`d212009dadf1ddbf945c8dc7ad0214ba444236baf57c8ed9019c3ebe6b0805b4`.
Raw captures and the binary/source/fixture identities are under
`~/scratch/m3-final-ds-sampled/frontier-hca-1355f003/` on `spark`.

Initial `head-8192/` and `head-8229/` diagnostic captures were invalid:
activation placement rebound external probe leaves to uninitialized
storage. They are excluded from every result above. Corrected
`head-fixed-8192/` and `head-fixed-8229/` preserve the external leaves,
check their final pointers, require finite nonzero streams/outputs and
prove completion before clean allocation unwind. No serving path used
that diagnostic placement.

## Workspace and timing scope

The isolated 2,048-row head suffix placement falls from 1,126,170,624
bytes to 554,496, and scratch from 9,455,616 bytes to zero. That local
reduction does not determine the full production highwater. At context
1,048,576 with 2,048-row chunks and DSpark, both initial wide production
arms need 2,373,976,064 activation bytes, 169,869,312 pool bytes and
46,137,344 host input bytes. Late trunk/state work retains the same
peak. Diagnostic narrow sizing has the same three byte counts in both
head arms too. The final counter-free HCA planner refresh confirms the
same 2,373,976,064 / 169,869,312 / 46,137,344 bytes in both head arms.
**No maximum-context workspace or
physical-fit gain is claimed**. Sizing loads no weights and runs no
model kernels.

Resident native 32K runs allocate 1,367,343,104 activation bytes for
all heads versus 905,969,664 for frontier. Their prefill times are
53.1752 versus 52.4332/52.4765 seconds, but the native all-head arm
copies every logit row while frontier copies one. Historical wide-body
production timing
uses last-row copies in both arms and ends before diagnostic state
readback. Two fresh arms of each head shape give the following mean
times; each prompt runs both plain and with full DSpark injection,
in all/one/one/all order.

| Prompt IDs | Injection | All-head prefill, s | Frontier prefill, s | Throughput gain |
| ---: | --- | ---: | ---: | ---: |
| 8,192 | off | 13.358730 | 13.170926 | 1.43% |
| 8,192 | on | 13.422774 | 13.286590 | 1.03% |
| 8,229 | off | 13.672935 | 13.534295 | 1.02% |
| 8,229 | on | 13.809391 | 13.746717 | 0.46% |

All 16 historical wide-body production arms completed successfully. Raw target/ring state is
bit identical between head choices; each choice repeats its first row
and complete trajectory exactly. The 31 subsequent complete logit rows
match exactly when fed the baseline's common forced inputs. These
production controls already used the 2,048-row compact floor, including
the 37-row tail.

The final HCA-body production control repeats those 16 arms and adds
eight direct-runner arms at 10,037 IDs (four 2,048-row chunks plus a
1,845-row stress tail). All 24 complete successfully: raw target/ring
state is bit identical across head choices, each shape repeats its first
row and trajectory exactly, and all 31 common forced continuation rows
match exactly, with and without DSpark injection.

| Prompt IDs | Injection | All-head mean prefill, s | Frontier mean prefill, s |
| ---: | --- | ---: | ---: |
| 8,192 | off | 15.404189 | 15.222278 |
| 8,192 | on | 15.491337 | 15.366870 |
| 8,229 | off | 15.748402 | 15.600324 |
| 8,229 | on | 15.802474 | 15.661688 |
| 10,037, direct 1,845-row tail | off | 19.381967 | 19.198080 |
| 10,037, direct 1,845-row tail | on | 19.428214 | 19.248953 |

These means give 0.8–1.2% prefill throughput gain for the head change
alone. The separate HCA quality choice costs the prefill time reported
above; it must not be hidden inside the head-only gain.

Final HCA/frontier forced rejection generates 160 outputs across 80
steps, 68 rejected; 14,473,556,224 compared state bytes have zero stale
bytes, covering zero/one/two accepted and compressed-block rejection
cases. The unchanged greedy plain/speculation check has 159 equal rows
and one near tie (plain margin 0.0993), zero outside 0.947.
The 160-output full-swap continuation has zero state differences,
including a rejection immediately before the swap, and replays captured
graphs after return. This is a bounded state/greedy control, not the
maximum-state swap gate. The fresh sampled check is recorded above.

## Provenance and remaining gates

Source base is `a5dff04` plus this uncommitted slice, including the
checked Qwen small-output schedules at `5735edb`. GB10 target `sm_121`,
SDK `aarch64-e0a0c85c42806fb1`, CUDA 13.4.92. The original artifact is
`8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234`
from `unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93`, UD-Q2_K_XL;
its three shard identities remain in
[the checkpoint pins](../fast-swap/pins.json). DSpark artifact is
`dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5`.

External raw records and exact command scripts are under
`~/scratch/m3-dsv4-frontier/` on `spark-b`: `head-fixed-{8192,8229}/`,
`quality-{all,frontier,repeat,narrow}/`, `quality-128k-frontier/`,
`quality-analysis.json`, `narrow-analysis.json`,
`quality-128k-analysis.json`, `sizing-{9216,131072,1048576}-{off,on}/`,
`production-clean/`, `final-forced-{a,b}/`, `final-all-head/`,
`final-ppl/`, `final-quality-analysis.json`, `all-head-128k-analysis.json`,
`sizing-narrow-{9216,131072,1048576}-{off,on}/`,
`floor-32k-wide/`, `floor-32k-analysis.json`,
`floor-32k-wide-compact-off-32768/`,
`floor-32k-wide-compact-off-32768-analysis.json`,
`floor-32k-hca-narrow/`, `floor-32k-hca-narrow-analysis.json`,
`floor-128k-hca-narrow-all/` and
`floor-128k-hca-narrow-all-analysis.json`,
`floor-{32k,128k}-hca-narrow-{all-repeat,frontier,repeat}/`,
`floor-128k-hca-narrow-ppl/`, `hca-qualification-analysis.json`,
`hca-production/`, `hca-direct-1845/`, `forced/` and `swap/`.
Final HCA sizing is `sizing-hca-final-{9216,131072,1048576}-{off,on}/`.
The matched wide 128K timing is
`floor-128k-wide-prefill-performance/`; exact commands are in
`frontier-wide-128k-performance.sh`, `frontier-hca-qualification.sh`
and `frontier-hca-production-controls.sh` beside these raw folders.
Each model invocation records its command and sampled MemAvailable in
`memory.json`; `run.log` records the required empty-GPU/free-memory
load gate. Diagnostic peaks include retained captures and are not a
production physical-memory comparison.

The 32K oracle SHA-256 is
`49de2d51973b4bef32cf174c861d44d74bf63b6ec8ee576641dcba8bcc042aa6`;
the 128K oracle is
`d38333599da4e9bae0f55980663e307fd1f0b386d22a0d2c2fb44cd254a35f92`.
The 31,705-ID prompt TSV SHA-256 is
`0961e248af647f3e9c838d43ba2cbad821fc48a7e9f066679ae7b2e8f118e6cb`;
the 128,821-ID TSV is
`c68362cce5e09e15e834eea8ecdddc8f5e9e00aaf10d8c423d120beb1a17a510`.
Corrected probe/32K/128K native binary SHA-256 is
`fec242612bd0fb0c31cc34b11c793b3964824b46ec58efd173dafb9b6547826b`.
The narrow 128K forced/all-head/PPL experiment uses native binary
`f0207d2b518e83c52e90428fc0c681d0ac061c2e7be9089d8cebde619955851b`.
The corrected-floor wide 32K control uses
`39b6df505cf0bec9f185e6883e50f17b5bceb338cab0126399f7117a59177a4b`.
The external HCA-only 32K control uses
`cea0b61bc51f73d0d6ebe8dda62f382f3ea15e77601b22c212a106f6ddfb9590`.
The matched wide 128K timing binary is
`c07a8542769325f36d494d4d1fb8ceaaceb4e500d9fbbf7f9a3fb295e19ab2d3`.
The HCA control's archived dispatch source SHA-256 is
`a2dc8acfa0701b5c5a34f549143db42f7a3c292186b92bf1f64a4947c0a88292`;
`hca-narrow-binary-source.sha256` records the diagnostic counter source
too. The final planner retains this same HCA predicate; the completed-call
instrumentation is removed without changing GPU arithmetic. The measured
counter source identity is separately retained in that record.
The final SDK style check rewrites the predicate by DeMorgan's theorem;
its mask classification and short-circuit behavior are identical.
Production binary identities are separately recorded in
`production-clean-binary.sha256` (historical wide),
`hca-production-binary-source.sha256` and `rollback-swap-binary.sha256`.
Final counter-free build/check identities are recorded in
`final-hca-binary-source.sha256`: native harness
`8303383715ea1f26ce6e711ee94f0c879b5abd4c8a2c362f8c760208d3834192`,
spec runner
`b5667e7a61a0483305fb62a1699cbcf64dc52280424e1c55f4f3b0aae1f69b37`,
runtime
`33d11c15750eeedefe0943c596a674c8fa997e1d9d5465f32529f7ee5a1567fe`.
The final planner SHA-256 is
`1355f00338f844ea60daa530d2d995bfd9b178b3e58680e1862321660efbff07`.
No completed-call diagnostic counters remain in production or the final
native harness. On `spark-b`, final prepare/locked checks pass 1,034
tests including 200 GPU tests, SDK tidy on 12 units, 294 boundary checks,
seven CLI refusal controls, REUSE and 1,001 header checks. Workstation
checks remain deferred by the owner until the optimization run ends.

Phase-2 command/config provenance survives on `spark` in
`~/.local/share/jitllm/jobs/lcds-f32k/job.json`,
`~/.local/share/jitllm/lcds/steps-c{1,2}.txt` and the raw summaries above.
Its adopted source is `1f33ab0`. The old `lcds` build was rebuilt later
on 2026-09-29 at 19:30 EDT with growing-state work, so its present
`21c6371e…` binary and dirty `cdcf66b…` receipt are not a trustworthy
binary pin for the earlier noon phase-2 captures. No archived per-run
binary hash was found in those records; this provenance limitation is
kept explicit.

Maximum-context runtime quality, physical-fit/admission and full long
swaps remain the [final context study's](../m3-final-context/README.md)
gates. Its earlier actual 1M request fit is a pre-frontier result and
does not validate this changed head.
