<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded Gemma joined-serving diagnostic

The first screen uses four independent owners on each approved 26B/31B
profile. `ServingOptions::gemma_joined` and `gemma_row_invariant` are internal
diagnostic switches, both false by default and absent from configuration and
CLI. The ordinary scalar route remains the production control. One-owner
controls use the same selected product policy as the joined candidate.

The runtime prepares each owner's state independently, validates distinct
current one-anchor units, then invokes bounded existing `Gemma4Runner::Wave` groups.
Full heads and native cursors publish only after completion. Existing session
logic samples and calls back per owner afterward; a shared execution failure
applies no callbacks. Per-owner preparation refusal does not exclude prepared
peers. Existing two full-vocabulary float rows and fixed two-vocabulary
sampling candidates per owner remain funded. Native pinned output, plan,
activation, state and staging bounds remain separate.

The initial model control uses sparse slots 1/4/7/11 with unequal six-to-nine
token histories. Every complete head, greedy history and initialized state is
compared with the same owner's solo run and fresh joined repeats. Captured
addresses stay owned by the existing runner. Row-invariant products reuse
checked one-row reductions for at most eight columns/tokens; four routed owners
require 32 selected pairs. The 128-pair routed-writer bound is a separate
contract and does not extend the eight-token product limit. Larger cohorts use
ordered subwaves, so twelve owners dispatch eight plus four. All tuples are
validated before the first write. A clean subwave refusal records errors only
for its owners and preserves completed peer results; an unusable cohort fails
the hook before callbacks. Generic fusion, norm chains, shared
Q8, cache-store and MoE policies stay off. Ordinary joined products remain a
separate arithmetic control, never an assumed equivalent path.
The row policy is selected for every eligible chunk, including synthetic
six/seven/eight-row prefills; the fourth nine-row prefill and natural64+owner
prefills use ordinary products. Ordinary-versus-rows is therefore not an
isolated synthetic decode arithmetic comparison.

The short paid helper supplies explicit IDs `[2,818,5279,529,7001,563]`,
followed by `owner` copies of ID 563. BOS appears exactly once. Warm eight
discarded units, clear/reset and prefill the same independent prompts, then
three untimed units. For owner `i`, unit `s` supplies
`[45518,107,101][(s+i)%3]`. Timed units use `s=3..34`: 32 completed waves,
each with one frontier per owner. Argmax and all complete head copies stay
inside the timer; disk output stays outside. Rates are fixed-prefix completed
units, not generated-token throughput. Context is 256 per owner, sufficient
for the supplied histories; this does not qualify context depth.

Native first/repeat full heads and unchanged ordinary controls are acquired
and frozen before oracle comparisons. A scalar/candidate/scalar bookend uses
identical supplied units. Preserve failures and original ordinary calibration;
ordinary-to-candidate changes are not own-repeat noise. Freeze source, binary,
input and full-head identities, ties and non-finite handling before inspecting
reference differences. Strict argmax changes are separate from oracle-winner
margin failures: an exact reference tie has zero margin. Byte-exact native
repeat implies zero native winner-margin movement, retained without widening
after seeing the reference. Sampled target likelihoods must name the selected
rows and next forced IDs; they are not a corpus PPL gate.

`llama_joined.cc` calls pinned llama.cpp's C API with a real batch of four or twelve
distinct sequence IDs. Each token names exactly one sequence and its absolute
position. Context `n_seq_max` equals the owner count; verify reported per-owner
context is at least 256. Every completed frontier is requested and obtained
by its batch index, with all owner heads copied and published inside the
timer. Retain stock fast fusion/graphs, F16 KV and the approved same-format
raw GGUF. Screen physical ubatches rather than serializing four reference
calls. Compare reference own repeats before cross-engine metrics. The stock
math library/image remains pinned at llama.cpp
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, image
`837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`.

Task-entry upstream refresh at 2026-10-05 05:54:42 UTC observed TensorFold
HEAD `609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5. Its pinned
[README](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md)
and [Gemma recipe](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/docs/recipes/gemma-4.md)
expose Gemma26 on MLX, without a CUDA Gemma26/31 comparator. Same-format
pinned llama.cpp is the oracle on GB10.

The subsequent controls cover C1/2/4/8/12, changing membership,
cancel/stop/refusal, exact spill/server restart and actual HTTP. Physical peak
requires measurement; budgeted
capacity is not peak evidence. Production selection, representative full
quality, optimized batching support, long context, and swap gates remain open.

A second short axis reads the exact 1,024 little-endian token IDs retained
from each profile's War and Peace tokenizer control. The two independently
fetched files have SHA-256
`b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610`.
Owner `i` receives IDs `[0,64+i)` with BOS once, then one successive supplied
ID at each unit. Eight warm units are discarded before reset/prefill and
three untimed units; paid unit `s` consumes ID `67+i+s`. Its next target is
ID `68+i+s`, which is the labelled sampled-likelihood position. All native
and reference arms retain and authenticate the exact input file outside
paid time. This bounded natural-prefix screen is still not corpus PPL.
C4 uses one native shared group; C12 uses 8+4, while the stock comparator
executes all twelve independent sequence rows in one batch.

The dedicated `jitllm_gemma_joined_http` executable uses the ordinary runtime
startup and HTTP serving implementation, selecting only these two internal
Gemma diagnostics in a local command copy. Its production counterpart keeps
them false. `http_control.py` compares same-policy own solo outputs with
concurrent cohorts, literal likelihoods, chat/SSE/stops, refusals, cancellation
and pending cross-profile turns; it does not compare a stock HTTP server.
Actual startup status must report both selected diagnostics, and final counters
must prove completed shared units. Offered twelve-request concurrency is not
proof of an actual fixed-width twelve-owner native group. Each pending switch
direction must have its own positive source pause delta under the recorded
one-second model turn; exact owned SSE output is compared independently.

## Approved artifacts and reproduction

26B uses raw `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` at revision
`c099eb48e663fd284577b04978a94ffccb261841`, 16,947,541,728 bytes,
SHA-256 `f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f`,
prepared artifact `4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3`.
31B uses raw `gemma-4-31B-it-UD-Q4_K_XL.gguf` at revision
`c1ac76e99d5513b141e8adde7288b85c3f9c32ec`, 18,822,970,304 bytes,
SHA-256 `9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575`,
prepared artifact `32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08`.
Fresh-location approved payload transfers were supervised and destination
verified before atomic publication. They were coordinated outside measurements.

Use the declared Spark-native locked build and installed GPU supervisor for
each bounded job. `jitllm_gemma_joined ARTIFACT OUTDIR 26|31 OWNERS
scalar|joined ordinary|rows [IDS_I32]` acquires native outputs. For each axis
acquire ordinary, row-policy scalar-a, joined-first, joined-repeat, scalar-b,
then freeze before any oracle with `analyze.py freeze AXIS OWNERS`.
`reference.sh build` compiles only the C API client against `/app` in the pinned
image, with headers extracted by `git show` from the exact llama.cpp pin.
`reference.sh PROFILE OUTDIR OWNERS UBATCH [IDS_FILENAME]` runs real batches;
place the optional supplied-ID file in the wrapper scratch directory. Acquire
first/repeat at each C/32/128 ubatch. `analyze.py compare AXIS REFERENCE_FIRST
REFERENCE_REPEAT FROZEN_SHA256` authenticates both own freezes and all heads.
All output directories are external, fresh and distinct. The optional ID input
is the exact tokenizer control from the complete 3,274,124-byte corpus SHA-256
`c7156148ecaa12b6416cf816540d8dede2014982554a835e61076f0dd8bf0c2d`.
Never substitute a corpus or omit BOS validation. The scripts emit raw per-row
data externally; checked-in results retain only aggregate metrics and identities.
