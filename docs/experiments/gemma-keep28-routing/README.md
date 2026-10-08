<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 layer 28 routing keep diagnostic

Keeping only `blk.28.router_probabilities` makes this native 26B/C1 1024-row
corpus byte-identical to stock across all 1024 full-vocabulary heads (1 GiB,
SHA `ce5fd2ec…`). Two candidate runs are finite and byte-exact before comparison.
Routing selections become 29; plain norm 121, norm/RoPE 60, norm/add 90 and
reduction 30 remain unchanged. The [current dispatch observation](../gemma-current-dispatch/README.md)
records the same stock counts, with no routing selection at layer 28.

The unchanged native `all` arm previously had nine positive-margin choice
misses in [current quality](../gemma-current-quality/README.md). This single
keep factor proves the stock29/native30 routing-policy difference caused that
fixed-corpus discrepancy. The candidate inherits exact stock mean NLL
7.1590858381130875 and PPL1285.7350255459037 from complete vector equality;
no new PPL/softmax scan was needed. It does not establish a general layer whitelist, explain stock's
refusal reason, qualify other shapes or authorize a production default.

The dedicated manual target copies the quality helper with a closed
26B/C1/context4096/maxrows1024/all-output recipe. Plain norm, norm/RoPE,
norm/add, routing and reduction are enabled; ordinary products and every other
optimization remain unchanged. Normal graphs/fusion and F16 local2048/global4096
KV remain enabled. The required process environment
`LLMP_GEMMA_KEEP28_ROUTING=0|1` is read once before setup. Mode 0 forwards the
original planner keep span exactly; it was not acquired again. Mode 1 appends
only the named probabilities through the existing six-argument
`PlanGemma4Chunk` keep seam. That external reader refuses the fused routing
chain and retains its primitive steps. There is no production whitelist,
option, graph change or kernel change.

The keep applies during startup size measurement and actual planning, including
both placement/planning passes. Startup also measures one-row/frontier/state-only
shapes; applying it only during execution would understate workspace. Normal
activation maxima and plan/host-input floors account the live intermediate
before model execution. The caller's 1 GiB publication is charged before
allocation. Unsupported contracts forward the original planner; the closed
helper refuses different completed policy counts. Completion-aware teardown
and unknown-completion quarantine are copied unchanged.

## Reproduction and provenance

Base `fbac6dc`. Build only `llmp_gemma_quality_keep28` using the locked Spark
SDK. B's actual engine archive `nm` authenticated the six-argument ABI against
`gemma4_plan.h` SHA `756364ad…`. Measured private helper `846287ab…`, SDK receipt
`3474b122…`, source record `438df705…` and own freeze `2e9373f0…` are recorded
in [results](results.json). Production sources and historical helpers are
unchanged. Results distinguish the preregistered measured README identity from
this final report; the compiled helper/wrapper/header/CMake bytes are unchanged.

Use exact Task57 supplied IDs (4096 bytes, SHA `b2d7aaf6…`, BOS2) and approved
prepared26 artifact `4ddb360c…`/index `e7481988…`. Set mode1 and invoke
`ARTIFACT IDS_I32 NEW_OUTPUT_DIR` twice under installed `spark-job --gpu
--timeout600 --stop-on-fail`, umask077. Require successful retirement, actual
121/60/90/29/30 counts, shared/vector-row counts0 and complete 1073741824-byte
outputs. Reuse the existing current-quality native-only freeze controls with
exact source/binary/recipe identity and the diagnostic keep metadata; require
all1024 heads finite and exact across own repeats before reference comparison.
Then compare the complete frozen SHA against the authenticated retained stock
`ce5fd2ecd80d64f2e53252a8fe2be39f814881ea6e4e2d20a1f39aeed5b7be9f`.
[Task57](../gemma-current-quality/README.md) retains the original b29 image,
client, libraries and score-ring1024 recipe. [Task61](../gemma-current-dispatch/README.md)
independently confirmed that complete reference identity with observed dispatch.

All five official build/source/native/freeze/comparison jobs completed DONE0;
B returned free with no GPU jobs, waiters or compute processes. No new stock
run, performance timing, numerical allowance, additional model arm or full
regression suite was run. Raw vectors/logs remain external. Native NVCC13.4.92
(toolkit13.4.2); inherited Task57 host metadata records GB10/driver580.178.04; the original math libraries are CUDA13.3.
Fresh TensorFold primary main remains `609ca419…`, version0.6.5; its
[Gemma recipe](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/docs/recipes/gemma-4.md)
provides MLX26 and no applicable CUDA comparator, so no TF run was made.
