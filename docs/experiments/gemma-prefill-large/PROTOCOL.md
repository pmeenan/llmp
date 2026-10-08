<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Larger native Gemma prefill screen

This is a diagnostic on native base `c2d2147`, with no production row-cap,
execution-policy, model-support or quality allowance change. TensorFold was
refreshed at 2026-10-05 06:05:56 UTC to latest
`609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5. The pinned primary
README and Gemma recipe still expose Gemma26 through MLX only. Same-format
llama.cpp remains the comparator at
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, original image digest
`837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`.

One physical Spark runs the approved Gemma26 GGUF/prepared artifact, F16 KV,
context 16,384, device masks, native ordinary policies and enabled reference
fusion/graphs. The unchanged shared LE I32 fixture contains 8,227 canonical
War and Peace IDs including explicit BOS, SHA-256
`6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b`.
No prompt/template, sampling or cache policy changes. The prior screen's
[protocol](../gemma-prefill/PROTOCOL.md) supplies full corpus and artifact pins.

Each engine warms on six discarded rows then clears that state. Paid prefill
consumes 8,192 rows, including current-shape plan construction/capture, state
growth, CPU staging and completed full-vocabulary publication. Native128
publishes 64 heads; native1024 publishes eight. Every intermediate head is
paid and disclosed. Reference ubatch1024 requests one final head. This does
not suppress intermediate native work or constrain competitive reference
ubatch choices to a smaller native cap.

Three common untimed one-row inputs `[236761,108,236913]` warm the decode
shape, ending at position 8,195. The next 32 paid units use the exact fixed
fixture inputs through position 8,227. Full-vocabulary CPU argmax is paid
but does not alter those inputs. Disk writes, reporting and completed state
snapshots are outside both timers. Setup, eager context allocation and weight
loading are discarded consistently; this is resident execution timing.

First acquire ordinary128, ordinary1024 and an independent ordinary1024
repeat. Authenticate source/build identities before inference and freeze the
two complete heads and exact initialized state hashes before new reference
acquisition or analysis. This bounded repeat does not supply representative
quality or inherit prior128-row noise. Retained final/prefill heads are finite
checked; whole-state hashes compare only the same layout. Later reference
bookends or a separate norm-both axis require a measured question from this
screen. The separate compound `all` axis requests only existing checked norm/RoPE,
norm/residual, routing and scaled-reduction policies. It receives its own
first/repeat head/state freeze before reference analysis. Shared-Q8,
row-invariant and store options remain off. These flags are diagnostic and
do not select a production policy.

Caller publication retains one catalog-charged frontier head. The runner
separately funds its full max_rows×vocabulary pinned envelope and checked
activation/scratch/input envelopes. Plan/graph retention uses the actual
bounded call count, not a percentage margin. The diagnostic snapshot writes
from completion-proven, catalog-charged pinned storage directly to the file;
no full CPU payload vector is allocated. Unproven retirement retains the
complete node, runner, graph/context and entered-model ownership bundle.

Matched physical peak measurement is a later decision control using D-085's
idle-host median baseline and 20-ms MemAvailable sampling through setup,
paid work and retirement. Catalog capacity is not physical peak; RSS and
cgroup memory.current miss device VMM. No memory gate follows this first
speed screen. All heavy work uses installed GPU supervision, timeout600.

The independent dense31 axis uses the same supplied IDs and context with
ordinary128, ordinary256 first/repeat, then norm-both256 first/repeat.
Each cap-256 policy freezes its own two heads, full initialized state and
layout before the reference acquisition. Native256 publishes 32 heads;
reference ubatch256 requests one. This ubatch was the fastest of the prior
31B screen's tested settings; it is not a universal optimality claim. The
norm-both bookend and reference repeats then use exactly that 256-row shape.
MoE flags remain zero for dense31. The raw and prepared approved31 assets
were copied from the other Spark during an agreed quiet window; the newly
published destination raw file was size/SHA checked and the artifact deeply
verified before use. Historical source assets were not downloaded again.

The memory observation in the report applies only to ordinary26 cap1024.
The other larger policies fail the strict token screen, so no additional
peak measurement or expanded quality ladder is selected by this diagnostic.
