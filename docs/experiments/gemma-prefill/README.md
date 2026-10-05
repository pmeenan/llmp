<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 26 paid 8K prefill screen

Native prefill is materially slower in this first matched 8K screen:
**6.931 s versus 2.612 / 2.605 s** for the fastest screened reference
setting. The subsequent 32 fixed-prefix units are faster in native,
**0.691 s versus 0.866 / 0.869 s**. Neither observation closes a gate:
representative quality failed against fusion-enabled reference, optimized
request joining is unqualified, and comparable physical peak memory was
not measured. All optional native execution policies remain off.

[Protocol](PROTOCOL.md) fixes one physical Spark, exact approved same-format
weights, F16 KV, context 16,384, enabled reference fusion/graphs, paid
completion/staging/full-vocabulary publication and shared teacher-forced
input IDs. The native base is `d5539c8`; these measurements do not test the
later serving adapter. The freshly observed TensorFold pin/coverage is in
the protocol. Raw vectors and logs remain external.

The 8,227 supplied IDs are the first tokens of the canonical War and Peace
corpus, explicit BOS included. Their LE I32 SHA-256 is
`6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b`.
Both arms prefill 8,192 rows. The common three untimed inputs are
`[236761, 108, 236913]`, ending at position 8,195; then both pay exactly
32 one-row units, ending at position 8,227. Every case verifies the same
32 forced IDs. CPU argmax scans are paid, but their results do not change
the input fixture, avoiding prefix divergence from known policy differences.
This is fixed-prefix timing, not free-generation equivalence.

| First screen | Prefill seconds | 32 units seconds | Argmax differences from reference 1,024 |
| --- | ---: | ---: | ---: |
| Reference ubatch 512 | 2.93384 | 0.867339 | 4 |
| Reference ubatch 1,024 | 2.59917 | 0.865037 | 0 |
| Reference ubatch 2,048 | 2.76427 | 0.863642 | 6 |
| Reference ubatch 4,096 | 2.82678 | 0.867508 | 6 |
| Reference ubatch 8,192 | 3.16811 | 0.866077 | 5 |
| Native 128 | 6.91768 | 0.686445 | 5 |

| Fastest-reference bookend | Prefill seconds | 32 units seconds |
| --- | ---: | ---: |
| Reference 1,024 A | 2.61196 | 0.866499 |
| Native 128 B | 6.93119 | 0.691027 |
| Reference 1,024 A | 2.60496 | 0.869291 |

Reference is allowed to process the entire prompt in one batch, with the
screened ubatch sizes and only its final head requested. Native processes
64 chunks of 128, computing and publishing **63 additional intermediate
heads**. Plan building/capture, state growth, CPU staging and publication
inside these paths are paid. The prefill gap therefore includes the
smaller native product shapes, additional chunk/plan work and intermediate
head work; no isolated contribution is asserted. Checked intermediate-head
suppression and a qualified larger native envelope are optimization leads,
not a hidden timing bypass or an adopted improvement.

Native records 33 captures and 33 replays across setup and the measured
request, with zero norm/store/shared-VecQ/row-product selections in the
reported policy. Reference permits CUDA graphs and records a GGML graph
reuse delta of 32 in the unit timer; that counter alone does not prove
CUDA replay. A during-screen GPU sample was 2,509 MHz / P0; clocks were
not explicitly locked or sampled continuously through every timer. These
short bookends remain a preliminary performance screen.

The retained full prefill and final heads are byte exact between native
repeats, and across all three fastest-reference repeats. Other reference
ubatches have different head bytes despite identical input IDs. At the
prefill head, reference-relative total variation reaches 0.307528 among
those settings; at the final head it reaches 0.238392. Native versus the
fastest reference has TV 0.138244 / 0.237357 and maximum raw deltas
5.081410 / 7.865779 for the prefill / final heads. Both retained-head
argmax IDs agree (1,082 / 1,340), while five of the 32 intervening CPU
argmax choices differ. No noise allowance is inferred from these repeats,
no earlier calibration is widened or reused, and no 8K quality pass follows.
Only two complete heads were retained per run; full likelihood/depth and
initialized-state continuation controls remain separate requirements.

The first native launch cleanly refused state funding because its small
benchmark budget omitted accumulated per-shape plan/graph catalog charges.
The corrected benchmark bounds at most 100 calls, each with at most one
plan and captured graph. `PlannedHostBytes` includes 512 bytes per launched
node, and the graph catalog allowance is 16,384 bytes per node. Therefore
`100 * plan_floor * (1 + 16384/512)` bounds their capacity need without a
magic percentage. With plan floor 2,187,792 bytes, this is 7,219,713,600
bytes. The execution capacity is fixed occupancy 536,871,936 + weight
extents 17,047,748,608 + twice virtual state capacity 650,117,120 + caller
publication 1,048,576 + that plan/graph bound = **26,105,616,960 bytes**.
These are capacity/catalog terms, not measured physical peak payload.
No production runner or pager behavior changed.

| Measured source / executable | SHA-256 |
| --- | --- |
| Corrected native harness | `5507395cc99b4688e15b5c5f0895fd56eab9463d096a9ea7c9c39c4ea8286015` |
| Native executable | `7f197eacfebf758d163ed2bcbf2a92339a57f62cd37deeedd852baa4e9be2eac` |
| Reference harness | `0e7092d7163d98f109cd7a5fb99b922cedfefe28f3da9632bb95f8cef13403ba` |
| Reference executable | `bd2dd3aca6e1eeac227f18a3af6099c7a3a8631be7aae27206eb0bac50b28e56` |

Native invocation is `jitllm_gemma_prefill ARTIFACT IDS_I32 NEW_OUTPUT_DIR`.
Reference invocation is `llama_prefill RAW_GGUF IDS_I32 NEW_OUTPUT_DIR
screen UBATCH`, with fusion/graph disable variables absent; `prepare`
produces the shared prefix from the full corpus. Both output full F32
`prefill.f32` and `final.f32` outside their timers. `compare.py ROOT`
validates finite heads, actual byte hashes and fastest-path repeats, and
computes FP64 normalizers/TV without introducing a quality allowance.

Installed supervised Spark jobs completed the locked native build,
reference tile screen, corrected native run, A/B/A bookend and retained-head
comparison. The original clean capacity refusal is retained as a failed
benchmark invocation; no result from it is included in the timing table.
