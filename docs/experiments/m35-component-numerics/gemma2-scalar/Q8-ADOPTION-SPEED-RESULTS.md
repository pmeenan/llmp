<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Q8 schedule adoption speed result

**PASS for the registered representative Gemma2 C1/C2 speed screen.** All six
median paired slowdown ratios satisfy the unchanged D-085 limit of 1.10. This
does not close broader U4/U3 qualification or the retained Gemma26 frozen-state failure.

The [prospective registration](Q8-ADOPTION-SPEED.md), committed as
`586051a322fc1fbe7662cb1c16f416edbcbb8568`, fixed all three pairs before collection.
Both arms ran on Spark B, using immutable W13 baseline `a74178c8…` and W15
candidate `c940d8c2…`, with actual resolved dependency hashes checked before and
after every arm. The baseline was copied byte-identically from A before timing;
the comparison uses only measurements from B. Exact identities, absolute values,
all paired ratios, counts, peak occupancy and raw hashes are in
[the aggregate record](q8-adoption-speed-results.json).

| Metric | Pair 1 | Pair 2 | Pair 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| C1.prefill_s | 0.987202 | 0.978946 | 1.004322 | 0.987202 |
| C1.decode_tok_s | 1.002867 | 0.996814 | 0.999178 | 0.999178 |
| C1.wall_tok_s | 0.980997 | 0.963868 | 1.016019 | 0.980997 |
| C2.prefill_s | 0.983750 | 0.996884 | 0.998694 | 0.996884 |
| C2.decode_tok_s | 1.004841 | 0.998714 | 0.989254 | 0.998714 |
| C2.wall_tok_s | 1.000736 | 1.006781 | 0.999805 | 1.000736 |

Values above 1 mean slower candidate. Time ratios are candidate/baseline; rate
ratios are baseline/candidate. Each run uses the helper’s mean of six measured
passes; the final value is the median of its three paired ratios. `prefill_s`
is the client interval from cohort release through the last first token, not
isolated GPU prefill compute.

All six exclusive supervised jobs retired successfully with TERM/0. Each fresh
service completed one warmup and six measured C1/C2 passes: 21 requests, 128
generated tokens each, zero cached prompt tokens, no helper errors/problems,
and matched prompt indices and full token usage. There were no retries, replaced
samples, extra priming, runtime overrides or debug plans. First-pair ratios all
passed the early-stop rule. Actual arm durations were 103.655–104.102s, exceeding the
historical 57s planning estimate; peak occupancy was 5.997406–6.019485 GiB.

The separate unchanged-oracle Gemma26 native control retained ten logits/state
failures; the literal Gemma26 assistant control passed. Its completed plans
selected the repaired scalar branch for three stage 1 Q8 K4096 products, but the
K5632 preprojection used prepared MMVQ. Target verification with rows 2/4 had
one owner segment and ordinary multirow VecQ; those plans do not demonstrate
the changed scalar branch. No wider Gemma26 or Gemma31 repair is claimed.

The original A/D numerical FAIL, its ASLR-only wrapper recovery, both later
pre-service failures, and the repaired R/A exact score PASS remain separate
evidence. This speed result neither certifies the retained greedy semantic
negatives nor updates frozen expectations. Raw output remains external under
`/tmp/llmp-u3-q8-speed-586051a3/`; Spark B was verified idle after collection.
