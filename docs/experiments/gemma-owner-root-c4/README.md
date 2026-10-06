<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 C4 attention without K/V packing

Removing balanced K/V CONCAT lowers the closed paid C4 interval **8.93%**, with
all 128 complete heads and four initialized states byte-exact. A subsequent
plain-norm-on bookend is **2.10% slower than the fresh original mean**; all 128
heads match both original runs byte for byte. This is an unselected manual
consumer, not production batching or wider-context qualification.

| Arm | Two paid intervals (s) | Mean (s) |
| --- | --- | --- |
| Packed K/V, plain norm off | 3.65871 / 3.74650 | 3.702605 |
| Independent K/V roots, plain norm off | 3.39031 / 3.35341 | 3.371860 |
| Independent roots, plain norm on | 3.32711 / 3.32084 | 3.323975 |
| Fresh original physical C4 / u128 bookends | 3.25790 / 3.25319 | 3.255545 |

The first factor changes only K/V packing. Both arms retain identical Q/mask
packing, ordinary products and normRoPE/normADD. The separate plain-norm transfer
selects the existing RMSnorm/MUL fusion: 121 selected implementations rather
than zero; normRoPE/normADD remain 120/120 and all other optional counts zero.
Its 1.42% lower mean relative to the earlier owner-root norm-off runs is a
sequential comparison, not an interleaved isolated speed result. Counts describe
selected plan implementations, not CUDA launches.

Each process uses four independent prefixes of 64–67 tokens, three forced
anchors and 32 paid one-query C4 waves, context 256 / max rows 128 and 256
readable local/global cells. All 128 canonical vocabulary heads are published
and scanned for argmax inside the timer; full finite admission and funded state
copies are outside it. Each native process replays 32 graphs and captures none
inside the paid interval. State witnesses have 120 ranges and 230,686,720 bytes
per owner, ending at completed positions 99–102. They are exact across own
repeats, packed/owner policies, plain-norm off/on and the retained packed control.
All complete heads have SHA `9de4f80b…`, also equal to both fresh original heads;
all 128 choices agree. Original/native state serialization is not compared.

The distinct custom operation takes ten actual source descriptors: Q, mask and
eight owner K/V views. Each cache view depends on its exact SET_ROWS writer and
owned physical root. Placement, ordering, coverage and lane access retain all
ten edges; no contiguous eight-root span or virtual-address pitch is invented.
Both scratch planners use the checked owner launcher. The original packed
compiled geometry supplies the grid/reduction partitions (48 local D256 and
96 global D512), independently of owner-kernel occupancy. Existing tile/query
conversion/fixup arithmetic is unchanged. The prior
[D256](../gemma-owner-root-attention/README.md) and
[D512](../gemma-owner-root-global/README.md) operator proofs remain separate.
Analytic balanced-CONCAT K/V read-plus-write traffic removed is 3.4375 GiB per
wave at this profile; no traffic counter or bandwidth attribution was measured.

Three no-device metadata modes cover 12 profile/owner/query-row cases each plus
D256/D512 custom-node tag, tenth-source, parameter, shape and alias refusals.
The CPU descriptor object retains its checkers and has no undefined CUDA or
owner-launcher dependency; execution stays in the CUDA registry. Unsupported
builder shapes use the original graph unchanged. Default graphs emit no owner
operation. All model arms and output checks officially retired successfully;
both reference containers have checked owned-CID absence. Full suites are
deferred under the owner’s optimization override.

Measured source is based on `fbac6dc`, after initial entry `5cf8f5b`. Norm-off
uses source frame `ef1dbb6a…` / helper `29128fff…`; norm-on uses `95dca329…` /
helper `e22a9695…`. The locked SDK receipt is `fbf84c74…` (NVCC 13.4.92, CUDA
13.4.2, Clang 22.1.8). Checksum sync excludes Git, so its commit is null/origin
none; explicit source identities qualify the build. Spark A is `spark-c4e2` /
GB10; driver 580.178.04 is inherited from the previously checked environment.
The original client `01889d8c…` and pinned image `837fc732…` are unchanged.
New core/diagnostic code is Apache-2.0; the existing owner CUDA launcher keeps
its MIT AND Apache-2.0 attribution. No shader or source-lock changes.

Build1 compiled successfully then hit the pinned custom factory’s fewer-than-ten
argument assertion. Its scoped correction installs the final real source slot
after constructing the first nine. Build2 compiled, then lacked the external
fixture-directory environment. The unchanged compiled retry passed metadata,
symbol/source and private-binary checks before any model run. Both failures are
retained. [Aggregate results and identities](results.json) contain completion
records; raw heads, state, logs and telemetry remain external.

Reproduce with `jitllm_gemma_owner_c4 ARTIFACT NEW_DIR 31 4 joined norm IDS_I32`,
fixing `JITLLM_GEMMA_OWNER_C4=packed|owners` before construction. The optional
`JITLLM_GEMMA_C4_NORMMUL=0|1` defaults off. Use the canonical 4096-byte IDs with
SHA `b2d7aaf6…`, installed Spark supervision, private outputs and owned reference
retirement as described in [the protocol](PROTOCOL.md). TensorFold at entry is
`609ca419…` / 0.6.5: its primary Gemma recipe is MLX26 only, without a matching
Gemma31 CUDA recipe. Global8K/local-window crossings, wider read widths and
production selection remain separate gates. The later [bounded Gemma26 transfer](../gemma26-owner-root-c4/README.md)
records a 7.13% gain and 0.265% fresh-reference latency excess with exact native
heads/state; its two strict quality failures remain unresolved.
