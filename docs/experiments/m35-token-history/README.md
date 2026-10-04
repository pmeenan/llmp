<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native token-history correctness control — 2026-10-04

The production runtime restores exact 8,192-token states and reproduces the
unswapped 64-token greedy continuation on both M3 LLMs. All four swap rows
report `exact=true` and `state_exact=true`, with no reported problems. The
ordinary report exposes 98,312 bytes of native token capacity and 2,097,152
bytes of rounded token catalog occupancy at its end.

This is a narrow correctness control, not a refreshed performance table or a
new batching qualification. It uses scalar continuations, one first-use
A→B→A cycle in each ordered pair, and no zero-context or image rows. Paused
session ownership, pressure refusal and reclamation are covered by the
separate accounting regressions; this real control performs no history
reclamation. Its swap timings are retained below solely as run provenance.

| A / B | Direction | Endpoint seconds | Exact state/output |
| --- | --- | ---: | --- |
| deepseek / qwen3.8 | A→B | 13.165066 | pass / pass |
| deepseek / qwen3.8 | B→A | 9.589739 | pass / pass |
| qwen3.8 / deepseek | A→B | 9.803151 | pass / pass |
| qwen3.8 / deepseek | B→A | 7.843509 | pass / pass |

Spark B (`spark-56f5`), NVIDIA GB10, driver 580.178.04, official
`spark-native` SDK `aarch64-c09daba6ac31edee`, locked CUDA toolkit 13.4.92.
The source lock SHA-256 is
`da170114dba30451b2cc979c0b85bac1b8185bdc8fab5833fa36c126ded3141f`.
Sources were synchronized by checksum from the detached `2220324` worktree
with the native token-history change. The remote checkout has no Git
metadata, so its receipt reports `0.1.0-dev+unknown`; it is not an unchanged
`2220324` binary. This run used the implementation before the final
admission-credit arithmetic correction. That correction changes reclaim
credit only, leaves runner/page/state buffers unchanged, and is checked by
the final CPU and Spark regressions below rather than by this earlier binary.
The serving source SHA-256 before that correction is
`d738680c8e27c9a60b26306d8508f34962aded6e03606600292f865f4bca59ba`.

A fresh scratch data directory supplies no earlier calibration or kept state.
Both models use speculative decoding, 4,096-row prefill chunks and four
configured slots. DeepSeek uses context 262,144 and forced speculative waves;
Qwen uses context 33,792, the selected 47,172-entry MTP head (requested cap
65,536), and the fallback wave/depth policy. Artifact identities:

| Role | Prepared artifact |
| --- | --- |
| DeepSeek V4 Flash target | `8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234` |
| DeepSeek DSpark drafter | `dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5` |
| Qwen3.8 Flash Next target | `c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93` |
| Qwen3.8 MTP drafter | `8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40` |

The context is frozen `4655685:docs/decisions.md`, SHA-256
`6b159ff20d198a3ff825d78b5edc0c2bd8c9cf5af7222875fb88301635e14db5`.
Supply it outside the checkout with `git show 4655685:docs/decisions.md`;
the harness independently tokenizes it for each target and retains 8,192
tokens. The configuration uses the production artifact/settings entries
from the [M3 final swap control](../m3-final-swap/README.md), omits image,
and names a fresh data directory.

Run through the installed GPU supervisor, from the prepared checkout:

```sh
~/.local/bin/spark-job start --gpu --name m35-history-exact2 --timeout 600 -- \
  build/spark-native/src/runtime/jitllm-runtime \
  --anchor ~/scratch/m35-history-control/enrollment \
  --config ~/scratch/m35-history-control/models.toml \
  swap-table --pairs deepseek:qwen3.8,qwen3.8:deepseek \
  --context-text ~/scratch/m3-final-table-20261004/decisions-4655685.md \
  --context-tokens 8192 --continue 64 --cycles 1 --zero-context off \
  --report ~/scratch/m35-history-control/table.json
~/.local/bin/spark-job wait m35-history-exact2
```

Raw logs and JSON remain outside Git on Spark B under
`~/.local/share/jitllm/jobs/m35-history-exact2/` and
`~/scratch/m35-history-control/`. The run completed with exit status zero;
its report has an empty `problems` array, and runtime retirement reports
zero reclaim attempts that took nothing or ended short.

The final source passes the pinned CPU build and 12 selected
`TokenStorage.*:IntakeLimits.*` checks. On Spark B, supervised
`m35-history-check4` rebuilds with `mise run build -- spark-native --locked`
and passes all 67 `llm_scores_test`, 39 `runtime_test` and 11
`kept_state_test` cases, plus both GPU
`TokenCapacityStateTest.*:TokenBoundaryStateTest.*` checks: 119 total.
These include independent copy/transfer charges, worker retirement,
concurrent and aliased admission, protected snapshots/continuations,
growing-library pressure, clean refusal without native dispatch, bounded
geometric growth, rounded catalog release, prospective boundary admission,
and the final admission-credit correction. The GPU boundary checks use
one- and two-extent budgets rather than relying on abundant model capacity.
The full integrated Spark suite is recorded separately at the commit gate.
