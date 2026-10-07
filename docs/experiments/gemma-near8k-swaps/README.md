<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Near-8K Gemma initialized-state swaps (2026-10-07)

Gemma2 2B Q8_0 and both approved Gemma4 profiles now pass one scalar model-switch
pair each against the approved Qwen3.8 NVFP4 artifact. All three use context 8192,
8063 saved Gemma tokens and 64 emitted continuation tokens. This is near-8K,
not an exact 8192-token saved state. The earlier [public Gemma3 scalar gate](../gemma3-execution/README.md#public-scalar-8k-and-one-model-switch-pair-2026-10-07)
separately saves 8192 at context 8448. Public settings and arithmetic are unchanged
by these measurements.

The existing `swap-table` runs two cycles and zero-context controls, six rows per
pair. Both saved-context returns per model compare SHA-256 of the complete
initialized-state snapshot to the unswapped state, then compare all 64 emitted
tokens and complete continuation heads byte for byte. All six actual saved-state
returns and all 18 table rows pass. The other rows' default state flags are not
additional initialized-state proofs. Prepared saved returns retain graphs and
actually replay them.

| Scalar pair with Qwen3.8 | Worst first-use handoff | Worst prepared handoff | Prepared Gemma graphs kept / replays |
| --- | ---: | ---: | ---: |
| Gemma4 26B-A4B | 6.287826 s | 6.200046 s | 1 / 63 |
| Gemma4 31B | 6.407776 s | 6.294532 s | 1 / 63 |
| Gemma2 2B Q8_0 | 6.027503 s | 5.911946 s | 32 / 63 |

The bound is 20 s for prepared handoff and 40 s for first use. Times end at the first
output after activation and exclude diagnostic state hashing; all 64 continuation
rows are checked separately. These are completed observations of one pair per
profile, not repeated statistical timing arms, cold-SSD measurements, sustained
throughput or universal swap parity.

| Bytes | Execution budget | Fixed catalog charge | MemAvailable start | Lowest sample | Largest sampled decrease |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma4 26B-A4B | 117,459,595,804 | 2,118,332,956 | 124,894,359,552 | 43,083,857,920 | 81,810,501,632 |
| Gemma4 31B | 117,482,664,476 | 2,118,332,956 | 124,902,621,184 | 42,014,937,088 | 82,887,684,096 |
| Gemma2 2B Q8_0 | 115,882,511,900 | 2,122,501,660 | 123,235,897,344 | 41,425,895,424 | 81,810,001,920 |

The diagnostic whole-state snapshot is charged within each execution budget.
The last column is the largest row `peak_bytes`: the 20 ms-sampled whole-node
MemAvailable decrease from the process-start baseline, exactly start minus the
lowest row sample. It is not catalog occupancy, isolated process/inference
allocation peak, or an isolated measurement of snapshot cost. No maximum-context
memory claim follows from these near-8K observations.

Gemma26 uses ordinary 1024-row prefill, dense31 uses 256 and Gemma2 uses 128. Qwen
uses 4096, with one request slot per model and speculation disabled by `--plain`.
The approved source/artifact bindings and each family's existing optimized
recipe are preserved. The authenticated 2276-byte short text from the prior
Gemma3 depth screen is repeated 32 times (72832 bytes, SHA-256
`4e41c6aac01454db43f44522999d2dff87098891cd87f7737ab7a0a58d636a01`).
Table tokenizes it with each native tokenizer and constructs the 8063-token
saved prefix. This predictable repeated fixture establishes boundary/state
invariance, not real-corpus, retrieval, long-answer or new stock quality.

The Gemma4 artifacts already reside in the installed M3 store. Only the approved
Gemma2 artifact is transferred from Spark B to Spark A: four declared files,
2,784,782,856 bytes total. The reader authenticates the prior deep-import/source
receipts, holds no-follow source FDs, and checks declared length, full SHA-256
and stable source identity. A independently bounds/hashes each SSH stream and
verifies all destination bytes before private, fsynced, atomic no-replace
publication. Existing identical artifacts can be verified/reused; originals
remain intact. The reviewed private-group/no-ACL trust policy matches runtime
path handling. No raw checkpoint/Qwen duplicate or runtime discovery fallback
is introduced. The actual transfer step takes 15.229 s.

The existing runtime is SHA-256
`72ede5c8ab239c7b8b3378fb95fd92ffe41fc517ed94902b3ce79e84e7fbdd46`,
with source/library binding
`00cecae3912d55d7705e5caa01a58c9980ee4b595ea59333003b35db040c9f95`.
It is the 20e0990-based public Gemma3 build: all exercised Gemma2/Gemma4/Table
source paths remain unchanged in the f25ff81 report parent; independent internal
C3 changes do not execute here. No rebuild or old inference repeats were needed.
The native CUDA 13.4 cuBLAS/Lt closure is the authenticated SDK c09 installation.
Qwen's approved external tokenizer/template files are configured explicitly.

Reproduction uses the existing runtime command with private configuration and
state/spill directories. `MODEL` is one of `gemma26`, `gemma31`, `gemma2`, each
configured at context 8192/slots1 and its ordinary chunk size above:

```sh
jitllm-runtime --config CONFIG --anchor PRIVATE swap-table \
  --pairs MODEL:qwen38 --context-text TEXT --context-tokens 8063 \
  --continue 64 --cycles 2 --zero-context on --handoff on --plain \
  --report REPORT.json
```

Official supervised jobs `m35-gemma26-near8k-swap1` and
`m35-gemma31-near8k-swap1` finish three steps DONE0 each;
`m35-gemma2-near8k-swap1` finishes four steps DONE0 including transfer.
There are no failures in this slice and no full-suite run. Raw configurations,
inputs, logs, source/binary/library bindings, exact reports and aggregates remain
outside Git under Spark A `~/.local/share/jitllm/gemma-context-swap-{g26,g31,g2}-1`
and local `/tmp/jitllm-m35-coordination/gemma-context-swap-{g26,g31,g2}-result1`.
TensorFold HEAD was freshly checked at task entry as
[041d14a94e951834470fd514ed33e65b8be1059a](https://github.com/ashhart/TensorFold/blob/041d14a94e951834470fd514ed33e65b8be1059a/README.md);
that native 1.0.0 README supplies no matching qualified Gemma GGUF CUDA recipe
for this state-swap control.
Maximum contexts, broader pairs/cohorts, real-corpus/retrieval quality and
sustained memory/swap qualification remain open.
