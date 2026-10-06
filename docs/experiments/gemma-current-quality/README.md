<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Current Gemma teacher-forcing recipes

Gemma31 at 256 rows with the checked norm chains and plain norm fusion matches
fresh original `score-ring` output **byte for byte across all 1,024 complete
262,144-element heads**. The 1,023 next-token target likelihoods also match:
mean NLL 3.4969662596564586 and PPL 33.01514051246665. Strict choices,
positive-reference-margin differences, raw deltas and full softmax TV are zero.
This is one representative supplied corpus screen, not full model qualification.

The Gemma26 transfer at 1,024 rows with its compound policy **fails strict
quality**: nine choices differ, all with positive reference-winner margins;
only 15 complete heads match. Native mean NLL/PPL are
7.1596140423725565 / 1286.414335654398, against
7.1590858381130875 / 1285.7350255459037 (+0.0528344% PPL). Maximum raw delta is
8.48530387878418; maximum/mean full softmax TV are
0.13918319964140577 / 0.008975268308795565. Neither small relative score movement
nor repeat equality waives the nine differences. Both engines also had high
absolute PPL in the historical 26B screen; no cause or broad quality claim is
inferred from this short supplied prefix.
All nine raw row/margin records remain external: their row-index range is
218–983 and positive raw reference-winner margins span
0.0029506683349609375–0.13225078582763672. Results authenticate that retained
record; no token vectors or per-row likelihood telemetry are added to Git.

The [historical quality harness](../gemma-quality/PROTOCOL.md) supplies exactly
1,024 War and Peace integer IDs, including BOS 2. IDs occupy 4,096 bytes, SHA
`b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610`.
Each engine publishes every head, totaling 1,073,741,824 bytes per arm. Gemma31
four 256-row chunks select 121 plain-norm, 120 norm/ROPE and 120 norm/add matches
per chunk, with other optional counts zero. The 26B single 1,024-row chunk selects
121 plain norm, 60 norm/ROPE, 90 norm/add and 30 routing/reduction matches each,
with other optional counts zero. Its charged caller publication is 1,073,741,824
bytes. These are selected-plan counts, not observed kernel launch counts or performance measurements.

Both clients use context 4,096 and F16 caches. The original uses physical batch
and ubatch 256, `swa_full=false`, `kv_unified=false`, enabled flash attention and
normal graph/fusion policy. Its actual local/global cache capacities are
1,280/4,096 cells. The 26B reference uses batch/ubatch 1,024 and actual
local/global 2,048/4,096 cells with the same normal ring settings. All teacher rows are exposed, so this does
not test the
frontier-only final-block path or the 8K state-only prefill recipe.

For each model separately, native first/repeat full heads were scanned for finite values and
complete byte
equality, then frozen before either new original run. All 1,024 original repeat
rows are also finite and byte exact. The existing analyzer authenticates the
native-only receipt and source recipe before comparing all vectors, using FP64
softmax/NLL arithmetic and lower-index argmax. It keeps strict mismatches,
positive reference-winner margin and exact reference ties distinct. Calibration
is the current same-policy zero full-byte movement; no older 128-row allowance
is inherited. [Results](results.json) contain aggregate identities and metrics;
raw heads, per-row hashes and choices stay external.

The native helper now accepts chunks up to 1,024, uses `max_rows=chunk`, and
accepts trailing `normmul-on|normmul-off`. Historical 128-row invocations retain
their old policy defaults. Historical smaller chunks now have a correspondingly
smaller capacity and may change placement/numerical identity. Existing `score`
and `score-unfused` modes remain available with their original cache settings;
`score-ring` explicitly chooses the production ring topology. Existing analyzer
modes and their old 128-row recipes remain unchanged.

Build only `jitllm_gemma_quality` and the existing pinned-image
`docs/experiments/gemma-quality/llama_quality.cc` client. A current run is:

```sh
jitllm_gemma_quality ARTIFACT IDS NEW_NATIVE_OUT 256 both 31 normmul-on
llama_quality MODEL IDS NEW_REFERENCE_OUT score-ring 256
python3 -B docs/experiments/gemma-quality/analyze.py current-selftest
python3 -B docs/experiments/gemma-quality/analyze.py current-freeze ROOT 256 31 both normmul-on SOURCE_JSON SOURCE_SHA
python3 -B docs/experiments/gemma-quality/analyze.py current-oracle ROOT FREEZE_SHA
```

Under `ROOT`, supply `input1/ids.i32`; run native into `native-first` and
`native-repeat`, then freeze successfully before collecting `reference-first`
and `reference-repeat`. For 26B transfer use a separate ROOT, chunk 1,024, variant 26
and policy `all` in the same commands; keep its freeze and references independent. `SOURCE_JSON` records the exact `native_recipe` keys
`context`, `teacher_chunk`, `max_rows`, `variant`, `policy`, `normmul` and
`all_outputs`, plus authenticated source/binary/SDK/artifact/input and reference
recipe identities. Authenticate actual logs, chunk counts and cache capacities
separately; logits alone do not establish invocation policy. Use the installed
Spark supervisor for builds, models and complete-vector analysis; require
successful official retirement and absence of each owned Docker container.

The measured base is `845617f`. The native private helper is `23ba5fa7…`, the
original client `225c40af…`, and SDK receipt `b2174122…`. Native NVCC 13.4.92
uses toolkit 13.4.2 on Spark B / GB10, driver 580.178.04. The original uses the
unchanged b10964/b29c606e image and libraries identified in results. Clients were
preserved privately before the warm checkout was reused for Task58; unchanged
reviewed source snapshots and the successful canonical build record identify
what compiled. No production policy or kernel implementation changes.

Task-entry TensorFold primary HEAD remained `609ca419…`, version 0.6.5. Its
[pinned Gemma recipe](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md#models)
provides MLX26 and no comparable CUDA Gemma31 run. No TensorFold inference or
full regression suite ran. Other corpora, longer contexts, generation,
frontier-state equivalence and competitive batching remain separate gates.
