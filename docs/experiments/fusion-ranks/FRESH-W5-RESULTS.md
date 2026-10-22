<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fresh W5 RMSNormMul comparisons

**Independent measurement audit and native production data adoption PASS.**
All twelve registered exact F32 tuples completed on one GB10: eleven favor
fusion by more than 10% lower latency; 4096×1 is neutral. These are operator
comparisons against each fusion's own decomposition, not model-speed or general
quality claims. Full U6 and the other fusion families/format domains remain open.

The [prospective registration](FRESH-W5-REGISTRATION.md), committed as
`70aacc42e08596f7605b20b5b86cde9c79a902b1`, preceded every measurement. The
1024×1 first screen passed independent review before the remaining eleven ran
in registered order, each in a separate exclusive job, with no retry or omitted
cell. The [historical report](RESULTS.md) and original twelve rows remain unchanged
in [historical-v11-v12.json](historical-v11-v12.json). That archive's exact SHA-256
is `53e2e9dcb5957bdb5f21b6f16ba59bdb5d7e9a406b60887652fc4cec1717dedf`. Active `data.json` contains only the twelve fresh rows: its existing
format permits one comparison per fusion/query, so historical duplicate queries
cannot coexist in the active table. No identity was relabeled.

## Method and paid work

Both production planner/lowering arms start from the fused operation's own IR
expansion. Only the primitive arm disables the `rms_norm_mul` rewrite. Operands
are distinct dense F32 input[width,rows], scale[width], output[width,rows],
epsilon 1e-6 and declared 256-byte activation alignment. Exact query keys bind
all operands, parameters, numerical choices, device facts and requirements;
no width intervals, checkpoint gates or extrapolation are installed.

Each event replay pays 128 complete chains, with 128 fused or 256 decomposition
graph nodes. Four paired warmups, a primitive bookend, four further alternating
paired warmups, nine alternating AB/BA pairs and the final primitive bookend
are retained in every raw stream. Independent audit recomputed all 108 timed
pairs across 552 records, preserving raw order, checking rounded medians and
both symmetric latency bounds against decomposition D. F < 0.9D favors fusion;
F > 1.1D favors decomposition; inclusive endpoints remain neutral. Every bookend
ratio stayed below 1.1; maximum drift was 1.46406%. No bound changed.

Every run passed finite-output checks with relative RMS and max absolute error
both zero, against registered bounds 1e-6 and 2e-5. Captured replay matched the
complete eager outputs. These facts cover the deterministic measured fixtures,
not arbitrary-input quality. The ordered logical children are RMSNorm then Mul;
the physical decomposition remains the existing two-launch wrapper.

Each arm retains its original 32 MiB activation arena. Maximum lowered extent
was 12,599,296 bytes at 4096×256; primitive lowering pays the complete additional
norm intermediate. Every lowered pool demand was zero within the 16 MiB launch
pool. Library workspace is 33,554,432 bytes, selected persistent backing
21,015,040 bytes, construction-funding bound 64 MiB. These selected bytes and
capacities do not claim peak process or unified-memory usage.

## Measured cells

Median nanoseconds per complete chain, in registered execution order. Latency
change is `100 × (F/D − 1)`; drift is `100 × (max(bookends)/min(bookends) − 1)`.
The 4096×1 result is 9.53271% lower latency and correctly remains neutral.
No measured cell requires the losing-fusion veto.

| Width × rows | Fused ns | Own decomposition ns | Latency change | Bookend drift | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 1024 × 1 | 1,388 | 1,817 | -23.61035% | 0.08247% | Fusion |
| 256 × 1 | 957 | 1,465 | -34.67577% | 0.00000% | Fusion |
| 256 × 4 | 999 | 1,515 | -34.05941% | 0.16469% | Fusion |
| 256 × 32 | 1,024 | 1,469 | -30.29272% | 0.42633% | Fusion |
| 256 × 256 | 1,470 | 2,493 | -41.03490% | 1.46406% | Fusion |
| 1024 × 4 | 1,390 | 1,823 | -23.75206% | 0.21963% | Fusion |
| 1024 × 32 | 1,772 | 2,350 | -24.59574% | 0.08513% | Fusion |
| 1024 × 256 | 7,519 | 8,688 | -13.45534% | 0.73647% | Fusion |
| 4096 × 1 | 1,936 | 2,140 | -9.53271% | 0.05839% | Neutral |
| 4096 × 4 | 1,991 | 2,347 | -15.16830% | 0.28766% | Fusion |
| 4096 × 32 | 2,659 | 3,424 | -22.34229% | 0.29812% | Fusion |
| 4096 × 256 | 10,840 | 20,937 | -48.22563% | 0.36872% | Fusion |

## Source, device and evidence

The compiled snapshot is base `e9965820ceba09cedfd9395eb60f0ae6e6cc3599` plus
W5 patch `a4b5d01d67254631b708c055267eab048969f5106c685561bb51ed2bcd2c2eed`.
Its 2,814-file manifest is
`70812bd25843e0c06dcf75af729a6fcc36259e96c7c16fbdd640b70eee84ca34`;
committed source tree `5ca10c0b3af9bcd70572d10d607fc5afb260599c` matches it.
All twelve pre/post source guards and compiler-checker runs passed. Actual
benchmark SHA-256 was
`1394ae098f4159787bec6dc4c417407f36fc6ac4679e022792a83eca6e9f7637` pre/post;
compiler implementation identity was
`4c2b14cf0a66eebb8f433daefff9b58b4cfc0afdc082d6ba0e5a1ec3ac200cee`.
The registration retains complete configuration/object/compiler-receipt pins.

All runs used Spark A (`spark-c4e2`), NVIDIA GB10, sm_121, 48 SMs, driver
580.178.04, CUDA API/runtime 13000/13040, SDK `aarch64-2c4eb5b2f68f3fa5`.
GPU UUID was `GPU-6384bad1-3fd5-9c6b-b1c6-1047c098c559`. Each exclusive supervisor
returned zero within its registered 600-second bound, with the helper's shared
540-second budget and bounded retirement unchanged. All 48 child groups retired
with zero return codes, no live/unexpected descendants or postcheck errors.
The final installed busy receipt reports no job or GPU process.

| Identity | SHA-256 |
| --- | --- |
| Selection environment | `9f6dfdbe73202030a1ed946696cc239e0d2a4b12a54ee3c7d43742a1e1f9c92b` |
| Fused logical/physical recipe | `d58f028b52b3918356be39fe6d010471985c17443739902de203f81f8b30a8b3` |
| Logical RMSNorm child | `932c2ca4652dc9f178bfef2fe5f49ccea2f431eaa47f4491adf8d5396ac02918` |
| Logical Mul child | `9c2806345cee4448c571a625a9008aa53caa24ad645c7c4297b1914944c8b5a3` |
| Physical two-launch decomposition wrapper | `8af3084aae79647cac1c3146e4bab7dae9c894b8b9019d1393b54ad7d8ac2255` |
| Independent whole-matrix audit note | `348391b38bda93f8416e1dac7283c9dea4e55aae7bb22155908fc8f88806f6c4` |
| Independent whole-matrix audit detail | `bc1938502359cb5189507b45a94f42d548daef8cc9e5cb1e9e6a13dd09088855` |

Raw receipts stay outside Git under
`/tmp/llmp-m35-orchestration/u6-fresh-w5-matrix/<case>/`; the independent audit
note/detail are `u6-fresh-w5-matrix-review.md` and `.json` in that parent directory.
Each active row binds its exact query, implementation/selection identities,
measurements, this report and its raw sample hash. Raw summaries retain their
original `UNREVIEWED` provenance; the audited adoption does not rewrite them.

| Width × rows | Raw samples SHA-256 |
| --- | --- |
| 1024 × 1 | `f5f58dad1a9fe140b89c366a5267015799e304005c973927bed50cbd2fb46d14` |
| 256 × 1 | `3d071f1e2699ebd42bf84bd63bcf0258237e5b1b81fe7c2dba888918faae555e` |
| 256 × 4 | `db8025cb8e35712cb6c210d8cdd70ec8cf1572f730ddef6298da290a8e662180` |
| 256 × 32 | `d73ebe71b4a4eb95641679d0db0761c82632c75f013166483b531144b37cc004` |
| 256 × 256 | `3656ed276d12d811f93fccdb22ddc1934c480faef976fede283cbc98e3b6a03d` |
| 1024 × 4 | `456509331884b35c22475ed387a103691cd3b31c364e5121615e3308251575db` |
| 1024 × 32 | `631f4525363d58e9183d1710bd3fb503ec13847efe9a31df2004b87914758004` |
| 1024 × 256 | `1fddb4b924c0e2114d5d53e351794f86eb95a9ef9851bdd461d2a6dd999b9369` |
| 4096 × 1 | `06e152c5bb94b814df35931c298e7246e5a4ce013b8794c83527d7293e8424de` |
| 4096 × 4 | `ef11ee72d673496071cda893f921ed9c9432d69f0b8bd570450fea67b96e78d3` |
| 4096 × 32 | `20ff3241e46985bd681866b2189a8d5e97e46709d68917c55cf5c70fdba0d2cb` |
| 4096 × 256 | `d2a59408e7f6dbb57ce80f708d1c265ee57d0d4644d879ea66aa3f70519a6471` |

## Production consumption

The data-only native rebuild and production compiler-receipt check pass on
`spark`. The 195-object implementation identity remains
`4c2b14cf0a66eebb8f433daefff9b58b4cfc0afdc082d6ba0e5a1ec3ac200cee`.
The strict live GB10 registry/planner check passes with all twelve exact queries
current, the measured ordered child identities and selection environment intact,
and the neutral 4096×1 cell recognized. Width 512, row 1 remains unmeasured and
selects the incumbent. The test requires the expected module digest and twelve
current rows through explicit adoption-only environment settings; ordinary future
source changes may safely leave these observations stale.

The live test and 136 focused rank/planner controls pass without skips, including
the existing losing-fusion, neutral, stale and repeated-input refusal controls.
The benchmark binary remains byte-identical to the measured executable. These
functional checks introduce no new timing or model-performance claim. Broader
fusion coverage remains open.
