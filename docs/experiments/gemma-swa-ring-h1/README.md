<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dense31 8K ring-cache reference screen

With the pinned original reference explicitly using `swa_full=false`, the
unchanged native BOTH256 policy has zero strict differences across 32
incoming-head greedy choices and a byte-exact complete final head. The
prefill head remains different. Native paid prefill is 16.1957% slower and
decode 2.0504% slower than this ring-reference bookend mean. No production
policy, full-quality, state-restore, memory or performance gate is selected.

| Complete retained head | Byte-exact | Maximum raw delta | Full-distribution TV | Absolute reference-argmax NLL delta |
| --- | --- | ---: | ---: | ---: |
| Prefill, after 8,192 rows | No | 0.413614750 | 0.027604194 | 0.009189653 |
| Final, after 8,227 rows | Yes | 0 | 0 | 0 |

The final head SHA-256 is
`33ed447e371ed91df91c5d0d9f6ce0bbc3bf78d4f13483bfa56bf8a5ad8a8e29`.
Both ring reference runs have identical prefill/final heads and all 32 choices.
Fresh native first/repeat and its bookend have identical two heads, choices
and complete initialized state. Those native outputs additionally match the
retained historical BOTH256 result. Only two full vectors are retained:
there is no all-32-vector agreement, calibrated margin-noise or corpus PPL
claim. The likelihood column uses the reference head's argmax, not a
missing gold token beyond the supplied input.

| Paid arm, in order | Prefill seconds | 32 decode units, seconds |
| --- | ---: | ---: |
| Ring reference | 10.7120 | 3.21634 |
| Native BOTH256 | 12.4696 | 3.28538 |
| Ring reference | 10.7511 | 3.22240 |

Native publishes and pays for all 32 prefill heads; reference requests one
final prefill head. Both publish each completed decode head and pay for its
incoming-head lowest-index argmax. Setup, six discarded warm rows, reset,
three common anchors, file writes and the native state snapshot are outside
the respective paid blocks. This is resident execution with the original
fusion/graph policy and the existing native timers intact.

## Actual cache recipe

Pinned llama.cpp `b29c606e` has public C-API defaults `swa_full=true`,
`kv_unified=false`. Its CLI/server common defaults instead set
`swa_full=false`; common context setup forwards that value. The dedicated
[copied client](llama_prefill_ring.cc) explicitly sets both flags false,
closes dense31/C1/ubatch256, and leaves historical clients and results intact.
Actual context-creation logs establish 1,280 local and 16,384 global cache
cells in both new reference runs. It keeps original math libraries, F16 KV,
context16,384, batch8,192/ubatch256 and one sequence.

Native has the same physical local/global capacities, with a 1,024-token
local window, position-modulo-local-capacity writes and padded bounded
attention reads. Its unchanged [benchmark](../../../benchmarks/gemma_prefill.cc)
uses checked normRoPE/normADD, 120/120 selected choices at prefill and decode,
ordinary products, device masks and zero other optional policy counts.
Those counts describe selected plans, not observed CUDA launches.
Capacity agreement alone did not predict arithmetic equality: the prefill
head still differs, and the remaining 30 decode vectors were not retained.
No complete cross-engine KV-state identity is claimed. Native final-block
narrowing and original final-block row selection differ at prefill; that is
an existing source-backed hypothesis for the remaining head movement, not
an attribution established by this cache-recipe screen.

The [historical full-cache screen](../gemma-prefill-large/README.md) remains
a record of its actual recipe and gaps, including three strict dense31
choices. Its speed/memory observations do not establish representative
ratios or parity for a production ring-cache recipe. The new result isolates
a reference configuration axis with unchanged native heads/state; it does
not prove every remaining arithmetic cause or a universal kernel equivalence.
No physical peak measurement was taken here. Catalog/host capacities are
reported separately in [results](results.json).

The independent [Gemma26 transfer](../gemma26-swa-ring-transfer/README.md)
uses actual ubatch1,024/local2,048 and its current native policy. All 32 decode
choices agree, but both retained full heads still differ and native prefill/decode
take 57.64%/0.49% more time. Neither screen qualifies Gemma26.
Qwen/DeepSeek transfer likewise depends on actual cache/window
and attention contracts. The remaining dense31 prefill speed gap is a
measured optimization lead, with no new kernel or narrowed-head selection.

## Frozen provenance and reproduction

Source base is `ccb2b1c`. No separately retained historical executable was
identified, so the unchanged native benchmark was rebuilt and acquired
fresh first/repeat before any new ring oracle. Source freeze `0b4c7d0c…`
contains eight source paths, binaries, actual locked build receipt, eight
pinned headers, all four original libraries, IDs, manifest/index and source
ancestry. Native own freeze `409bbe84…` completed at 13:58:31 UTC with
finite complete heads, all 32 incoming choices and exact 1,740,636,160-byte
initialized state, layout
`gemma31-f16-kv-scalar-device-v1:16384:256:16384:1280`.
No older noise bound was inherited. The full production-source manifest's
457-file source/lock/toolchain bundle hash is `afc9dc19…`; its manifest-file hash is `107e…`.
[Provenance](provenance.json) supplies exact hashes and official retirements.

Before oracle exposure, an analyzer type defect was found: parsed choices
were tuples, while JSON saved choices were lists. Original `compare.py` and
all eight pre/own source bytes remain preserved. [compare_v2.py](compare_v2.py)
normalizes that equality, tests actual equal/mismatched input and greedy
streams, and authenticates its own supplied SHA. The final derivative
`01a13267…` was frozen separately in `analysis-source2.json` before the
ring bookend. The first derivative receipt is retained externally. No
native/original binary, math, paid work or acquisition changed for this fix.
Use v2 for new analysis; the preserved initial analyzer has this known defect.

Measurements ran on Spark-b (`spark-56f5`), NVIDIA GB10, driver580.178.04
queried before reference acquisition. Native SDK
`aarch64-c09daba6ac31edee` pins CUDA13.4.92; the same immutable original
image's previously authenticated environment is CUDA13.3.0/CUDART13.3.29-1.
Task-entry primary [TensorFold](https://github.com/ashhart/TensorFold) refresh
at 2026-10-05T13:45:21Z resolved609ca419/version0.6.5; its fresh Gemma recipe
remains26B-A4B MLX-only and supplies no dense31 CUDA comparator.

Follow [PROTOCOL](PROTOCOL.md) with the approved GGUF/artifact and canonical
8,227 LE I32 IDs SHA6b6567ca…. Supply the exact pinned headers externally,
checksum-sync sources with `rsync -rlpc` without timestamps, and build only
native `jitllm_gemma_prefill` through the locked SDK and `reference.sh build`.
Record the real build receipt and source ancestry: base/head, empty tracked
production diff, the measured source paths that differ from that base, and
all eight `SOURCES` byte/SHA records. Supply the full production-source
manifest. No model payload rescan is needed for already authenticated assets.

Run `own_freeze.py source ROOT SCRATCH BUILD_JOB` after successful installed
build retirement, acquire fresh `native-first` and `native-repeat` with
`31 both 256`, then `own_freeze.py own ROOT SCRATCH NATIVE_JOB SOURCE_SHA`.
Exclusively freeze the exact v2 SHA together with these pre/own identities
before exposing the ring oracle. After authentication, acquire exactly
`reference.sh replay ring-first`, native `native-bookend`, and
`reference.sh replay ring-repeat`. Every job uses installed GPU supervision,
timeout600 and successful official retirement. Never overwrite a freeze.

`compare_v2.py SCRATCH ROOT OWN_SHA BOOKEND_JOB ANALYZER_SHA` checks the
actual source/environment, native head/state repeats, input/choice alignment,
reference capacities/repeats and full finite head metrics. Its `--self-test`
checks tuple/list equality and rejects changed greedy/input or incomplete
streams. Raw heads, states, IDs, per-step choices and logs stay external.
The native benchmark's existing funded pinned snapshot and whole-owner
quarantine are unchanged. Narrow builds and the actual source/own/bookend/
analysis controls passed; diagnostic workflow scope omits a routine unit suite.
