<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded Gemma2 compatible prefill and serving

This slice transfers the checked shared prompt-wave driver to the approved
Gemma2 2B Q8_0 profile. Each owner retains a 128-row chunk bound and its
4,352-cell local ring at context 8,192. A separate 256-row wave envelope
permits two compatible chunks. Scalar and diagnostic defaults remain unchanged.
The serving adapter requests the qualified norm/multiply, quantized GeGLU,
norm/ADD and cap-50 owner attention paths; it requests no Q/K norm/RoPE fusion.

Two compatible multirow owners use actual F16 K/V concatenation and original
C2 MMA attention. Unequal one-row owners temporarily pad the shorter readable
cache with F16 zeros and every padded mask row with negative infinity. These
are funded activations: the original cache roots, initialized state, source
bounds, ring layout and spill identities remain unchanged. Attention softcap
50 and final-logit softcap 30 are preserved. Neither operation makes a view
span unrelated allocations.

## Focused qualification method

The first screen uses two different approved-tokenizer inputs: 256/768 prompt
rows, three supplied untimed rows, 32 supplied joined steps and four departure
steps per owner. `gemma2_joint_prefill_probe` exports two initial prefill heads
plus 74 continuation/departure heads and 72 pre-step choices. Native same-shape
repeat, device/full-head agreement, refused-work state identity and exact
checkpoint continuation precede the stock comparison. The stock caller uses
the original pinned d812 v0.6.0 CUDA image and public API, F16 separate KV,
`swa_full=false`, `kv_unified=false`, ordinary greedy backend samplers and
logical memory reset. Its sampled logits still copy complete vocabulary rows
to the host; calling the token API does not suppress those transfers.

Both full prefill rows and all continuation rows must be finite. Stock own
repeat precedes comparison; zero positive-margin greedy differences and the
existing 3% conditional-loss bound govern both the two initial prefix targets
and the 72 continuation targets. No new numerical allowance or cross-geometry
identity requirement is introduced. The two cap-50 operand controls first
compare real F16 concatenation against original physical C2 MMA at common
widths 512/1,024 and unequal owner padding against the physical C2 control.
They retain actual specialization occupancy/scratch checks, independent FP64
arithmetic, fresh sources, poisoned destinations and captured replay.

Only after this representative model gate passes does the ordinary runtime
HTTP harness exercise literal choice/likelihood repeats, cached chat 16/4/16
replay, SSE, stops, the actual template's system-role refusal, queued peer
progress after disconnect and two-slot clean-restart adoption. Successful
joined prefill/decode counters are checked after drain. Client reads may be
buffered and do not establish precise backend cancellation. Cold/cached and
fresh split-prefix geometry comparisons are descriptive; cached matched
replays remain exact. The 591-byte template is interpreted by the existing
native Jinja fallback (SHA-256
`ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6`).

An optional short stock/native/native/stock C2 timing screen follows quality.
It uses fresh processes, a warm prefix/three supplied/eight greedy sequence,
then paid compatible prefill and 32 greedy steps per owner. The three supplied
rows between paid prefill and decode are off clock. Both engines pay the last
full-head publication; native intermediate choices stay on device. This
n=2 screen is neither sustained performance nor an HTTP stock-parity claim.

The approved source is `bartowski/gemma-2-2b-it-GGUF@855f67ca`,
`gemma-2-2b-it-Q8_0.gguf`, 2,784,495,456 bytes, SHA-256
`2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe`.
Reuse the authenticated/deep import artifact
`eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870`;
no quantization, writer, schema or import conversion changes are made.
Inputs and raw responses/heads/logs stay external. The probes prepare actual
BOS-2 token IDs from the original prompt text and the stock caller independently
checks the same tokenization. Replay supplies those external inputs and a new
private output directory; the analyzer binds their hashes and lengths.

At this task's entry, a fresh TensorFold HEAD check returned
`041d14a94e951834470fd514ed33e65b8be1059a`. Its documented native Zig GB10
Nemotron scope provides no Gemma2 CUDA/GGUF baseline; the original same-format
llama.cpp path remains the applicable comparison. Historical reference pins
and measurements are retained.

## Results on Spark B, 2026-10-07

The five installed supervised jobs all completed with exit zero. The focused
checks pass 45 tests: two new cap-50 operands, nine Gemma2 profile/state,
six Gemma2 graph, eight Gemma2 plan, nine unchanged Gemma3 plan, two runner,
four checkpoint GPU, three actual-template and two settings controls. No
controls were skipped or disabled. Native own repeats preserve all 76 heads,
72 choices and both initialized state hashes; full-head/device choices,
refusal atomicity and restore-next comparisons pass.

| Operand control | Physical versus candidate | Maximum FP64 NMSE | Planned scratch |
| --- | --- | ---: | ---: |
| Packed C2, rows 128, cells512/1024, two fresh passes each | Byte exact, including captured repeats | 8.59749e-7 | 6,390,016 B |
| Unequal one-row owners 512/1024, three fresh passes | Byte exact, including poisoned replay | 8.82162e-7 | 399,616 B |

The packed true-softcap specialization reports 32 columns, two blocks per SM,
KV batch 32 and 96 blocks; the unequal owner control reports four columns,
48 blocks and the common-stream mask prepass. Actual true-specialization
occupancy is checked against the owner plan. Both use the unchanged existing
FP64 NMSE bound of 5e-4. Fresh sources, independent roots and original mask
bytes remain intact.

Both retired stock teacher runs repeat exactly. All 76 native/stock full
heads are byte-identical: two initial prefix heads plus 74 joined/departure
heads. All 74 scored targets (two initial prefix targets and 72 continuation
targets) have zero target-NLL delta. Greedy, positive-margin, tie, total-
variation and raw-logit differences are zero. The stock observer records
actual packed Q.ne3=2 and joined common-width attention with cap50.

All five ordinary HTTP cases pass and both runtime epochs exit zero. They
cover finite literal likelihoods and exact same-geometry repeats, exact
cached chat 16/4/16 replay, nonstream/SSE equality, suppressed stops, actual
system-role refusal/recovery, two active clients plus a queued third with
one departure and two completing peers, and two kept conversations adopted
and replayed exactly after restart. Cache reuse is positive (20 tokens in
the 16/4/16 control and 21 after restart). The first epoch records seven
successful joined prefill groups/362 rows and 180 joined decode groups/360
units; the second records 2/12 and 30/60. Both retain cap50, normRoPE=0 and
actual norm/Mul, quantized GeGLU and norm/ADD selections. These are actual
completed group counters, not inferences from client count.

| Short C2 RNNR arm | Paid prefill ms | Paid decode ms | Total ms |
| --- | ---: | ---: | ---: |
| Stock1 | 167.666 | 472.171 | 639.837 |
| Native1 | 173.573 | 501.642 | 675.215 |
| Native2 | 176.945 | 500.907 | 677.852 |
| Stock2 | 168.601 | 470.666 | 639.267 |

Native mean 676.5335 ms versus stock 639.552 ms is **5.7824% slower**.
The prefill/decode differences are 7.1255/29.856 ms. All four processes have
identical 64 natural choices, both final heads and completed positions 291/803.
This n=2 screen retains a measured gap; it establishes neither parity nor
sustained performance. No endpoint timing or HTTP stock-parity comparison
was made.

The host was Spark B, NVIDIA GB10, driver 580.178.04, with pinned SDK
`aarch64-c09daba6ac31edee`, locked d812/GGML0.26.0 sources and original CUDA
reference image
`ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db`.
Whole checksum source synchronization and empty dryrun preceded the narrow
build. Actual runtime SHA-256 is
`edb0f5d7b95282fafac2e06d7f7760a8166f21f8da9a84d07a31205385d5ae8d`,
receipt SHA-256
`874aaf7a5967cfbe91054e0d8fc1a0630f952e0e8eb54b831d09719f1e08ce89`;
the relocated runtime explicitly uses the hash-bound warm build cuBLAS
closure. All four reference CIDs are proven absent after checked retirement.
Raw heads, responses, prompt inputs and official logs remain in private
external storage; the source harnesses and aggregate results are retained here.
Local REUSE, header, portability, format, Python AST, shell and diff checks pass.

This first model geometry does not qualify every wrapped-ring/joined-prefill,
long-context, wider-batch, memory-pressure, model-switch or sustained-serving
workload. The earlier ring/checkpoint controls remain separate evidence for
their original independent-prefill geometry. The full regression suite was
not repeated for this focused slice.
