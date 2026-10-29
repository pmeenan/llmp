<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Remaining U4 C1 adoption speed

Gemma31, Gemma26 and DeepSeek pass; **Qwen GGUF fails its first-token
latency bound**. All 24 arms complete their fixed work with unchanged inputs,
zero prompt-cache hits and clean retirement. The failed Qwen result is retained
without retries, exclusions or altered thresholds. These are bounded C1 results
on the two GB10 Sparks;
they do not grant general batching, maximum-context or completed-answer quality.

The [prospective protocol](PROTOCOL.md), static pins and
executor were committed before collection at
`f987915079e179dc54fc3325356d9e5202107789`. The enabled external receipt has
SHA-256 `3f827c09b4eec7a86882916b521dc17bad437255dd5f82c1ee3a8f5e3bb20b51`;
independent preregistration review `f7dde5eb3e71627fea6e98dd0ede527e0ef62aa3614a5312e1b99df218c99c6f`.

A is the qualified current-source upstream-MMVQ reference and R is the repaired
default candidate. Their immutable executable hashes are `cadfa5956dce42f017560958400fe6e68a5731490bb489971f31030fb68e0289`
and `95a2238acec146b9f27003804da35f3881becf83d5597feb5c74fcffc0f8be47`.
Both arms use the same artifacts, corpus, configuration, resolved ten-library
closure and normal transport. Explicit arithmetic contracts, routed/grouped
products and wide products remain unchanged. Only eligible implementation-chosen
ordinary one-through-eight-column products use the upstream reference policy.

Each fresh process receives one warmup prompt and one distinct measured prompt,
temperature zero and 128 output tokens per request. All finish by the length
bound. Pair order is A/R, R/A, A/R. The pinned collector recomputes metrics from
client chunk timestamps. Slower-candidate ratios are R/A for TTFT and A/R for
decode and wall rates; each metric's median must be at most 1.10. Whole-node
peak GiB means largest observed loss of `MemAvailable`, not GPU memory.

| Family | TTFT R/A | Decode A/R | Wall A/R | Verdict |
|---|---:|---:|---:|---|
| gemma31 | 0.997194 | 1.058326 | 1.007296 | PASS |
| gemma26 | 1.001594 | 0.989085 | 0.998723 | PASS |
| qwen38-gguf | 1.121499 | 0.978360 | 1.008945 | FAIL |
| deepseek | 1.001294 | 1.004120 | 0.998270 | PASS |

The [aggregate](results.json) retains all 24 absolute arm observations, memory,
request counts, pair ratios, medians and hash bindings. Root independently
recomputed all metrics and checked work, immutable pre/post receipts and
retirement against retained records. Independent review confirms the same
three passing families and Qwen failure. Raw logs,
client timing records and supervisor receipts remain outside Git at
`/tmp/llmp-m35-orchestration/u4-c1-speed-results-{a,b}-r1/` and their original
Spark experiment directories. Jobs use installed exclusive supervision, bounded
600-second arms and no overlapping builds.

One supervisor invocation failed argument parsing before any collector or
runtime launch (`--arm g31-a-warm?`, return code 2). Its log SHA-256 is
`a08d87ac7bd12c25f6e3a92f2a26f5cc193df18e9ba2644a17c2d5d70ae5c6ec`.
The corrected invocation used a fresh supervisor name and the original exact
registered first arm. This setup error is separate from all 24 measured arms.

Qwen pair TTFT ratios are 1.143175, 1.110856 and 1.121499; the median
1.121499 exceeds 1.10. Decode and wall medians pass. Both arms use the same
104/111 warmup/measured prompt-token counts and matching reply hashes. The
[separately registered repaired candidate](../qwen-mmvq-routing-repair/RESULTS.md)
passes the same fixed screen. That later result does not relabel this original
failed comparison.

The separately registered [Gemma31 C2 result](../gemma31-multirow-repair-speed/RESULTS.md)
and unchanged native Qwen/MTP and image evidence retain their own scope. This
comparison preserves the existing frozen-result differences and strict Mia
failures; it changes no expected values or numerical bounds.
