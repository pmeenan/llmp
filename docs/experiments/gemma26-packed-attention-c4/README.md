<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 packed attention C4 transfer

The packed Gemma26 candidate still fails the original-reference arithmetic
screen: 68 of 128 greedy choices differ, and paid latency is 9.19% above the
reference bookend mean. It remains unselected.

This benchmark transfers the dense31 packing mechanism to the approved Gemma26
Q4_K_M artifact at context 256, four one-row owners, 256-cell reads and four
published heads. It changes local attention dispatch and local/global stream
geometry together. Production graph, kernel and serving choices are unchanged.

Both native policies use ordinary products plus the existing norm/RoPE and
norm/residual chains. The selected C4 plans have 60 rotation and 90 residual
fusions; routed topk/reduction, shared Q8 preparation, row-invariant products,
RoPE-store and broad fusion remain off. All actual expert bindings and routing
math are retained: Q4_K routed gate/up, Q5_1 down in layers 0–28 and Q8_0 down in
layer 29. The earlier stock layer-28 topk allocation-alias eligibility difference was
observed at 128-row prefill with optional native MoE policies (29 versus 30
selected routes). That observation is not a proven upstream bug or a cause of
this screen, whose optional routing/reduction policies are off. This experiment
neither reproduces that allocation nor introduces a layer-specific policy.

The [closed protocol](PROTOCOL.md) and separate manual targets reproduce the
screen. Build base is `802c33bb3a600f37fe183bdb889551e23bd416b1`. The measured dense31
helpers remain unchanged. TensorFold was refreshed at task entry on
2026-10-05 at 12:59:21 UTC, HEAD
`609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5; its Gemma26 recipe is
MLX-only. The original same-format CUDA comparator remains pinned b29 in the
unchanged image named in the protocol.

All paid CONCAT copies, execution, input staging, completed full-vocabulary
publication and CPU argmax are included. Complete finite scans, state witnesses,
file writes and hashing occur outside the timer in both engines. Native captured
replays and original GGML graph reuse are reported separately. The 32 waves
produce 128 full heads and final initialized states at positions 99/100/101/102;
each native owner witness is 57,671,680 bytes across 60 ranges. State copies are
funded and retired before buffer reuse. These retained initialized bytes establish
same-policy repeat evidence, not a comparable physical-memory peak.

Both policies produce byte-identical own repeats across every full head and all
four initialized-state files. The candidate paid heads/states also match its own
freeze. The fresh original bookends repeat byte-for-byte. Neither native policy
has a byte-identical head to the original in this screen:

| Native policy | Exact heads / 128 | Strict choices / 128 | Maximum raw-logit delta |
| --- | ---: | ---: | ---: |
| Unchanged segmented attention | 0 | 77 | 33.53945 |
| Packed local/global attention | 0 | 68 | 32.15202 |

All changed choices have positive original-winner-over-native-choice margins;
there are no reference ties. The first wave differs on 2/4 control and 3/4 candidate
choices, the first three waves on 6/12 and 8/12, and the later 29 waves on 71/116
and 60/116. For only 16 designated likelihood rows (waves 0/1/2/31, all owners),
maximum TV is 0.94861 control and 0.93919 candidate; maximum absolute target-NLL
delta is 7.88070 and 5.66798 nats. Mean signed native-minus-reference NLL delta
on those rows is +0.57513 and +0.26477 nats. These selected rows are not corpus
PPL, and fewer overall misses do not establish a quality pass: the early candidate
miss counts increase.

| Paid 32-wave arm | Seconds |
| --- | ---: |
| Original reference A | 0.996769 |
| Packed candidate | 1.090450 |
| Original reference B | 1.000610 |

Native own control first/repeat take 1.00144/1.00258 seconds and candidate
first/repeat 1.09469/1.09361 seconds. Packing is also slower in this native-only
screen. All native timed waves replay captured graphs (32 replays, zero new
captures); original `reused_delta=32` counts GGML graph reuse, with CUDA graphs
allowed. It does not authenticate a CUDA replay count.

The dense31 full-head byte identity does not transfer automatically to this routed
profile. This result changes attention dispatch and stream geometry together;
it does not isolate the remaining routed or ordinary-product math. The historical
layer-28 routing eligibility finding is not a proven cause of the current failure.
No additional omission arm, ladder or production policy follows this screen.

[Aggregate results](results.json) preserve counts, phase summaries and state/head
identities; [provenance](provenance.json) records the source/binary/build receipts,
fixed inputs, task-entry reference observation and official log hashes. The native
own receipt was frozen exclusively before the fresh original acquisition. To
repeat, supply the authenticated 1,024 little-endian IDs to the new manual target
with `JITLLM_GEMMA26_PACKED_C4=0` or `1`, using CLI
`ARTIFACT OUTPUT_DIR 26 4 joined norm IDS_I32`. The protocol supplies the owner
prefixes, warm/reset sequence, timed positions and original image contract;
`reference.sh build` compiles only the unchanged thin client against original
libraries, and `reference.sh run NAME` executes the same-format reference.
Raw vectors, state witnesses and logs remain external. No comparable peak-memory
or whole-model quality/support gate is claimed. Optimized serving, longer context
and assistant transfers still need their own controls.

Focused locked build, 24 descriptor controls, native own freeze, fresh bookend,
full-row comparisons and paid head/state verification passed without skips.
