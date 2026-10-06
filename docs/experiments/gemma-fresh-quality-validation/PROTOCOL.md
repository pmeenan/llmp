<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fresh Gemma26 schedule calibration and heldout corpus control

This prospective control tests the current all-policy Gemma26 candidate without
changing routing eligibility, a production layer list, or the numerical method.
Earlier zero-bound failures remain failures. The method is fixed before any
new tokenizer, native likelihood or reference likelihood output.

Select untouched histories 13 and 14 from the pinned Wiki prose corpus. For each
index, start at the first complete paragraph strictly after `65536 * index`;
end at the first paragraph boundary after another 32 KiB, refusing a slice over
64KiB or invalid UTF-8. Freeze both text identities before tokenization. Use the
qualified native GGUF tokenizer and public vocabulary-only `prepare1024` on
exactly these bytes. Require one BOS and byte-identical supplied 1,024-token
prefixes for each independent history. Never score a transition into the other
history or beyond the final row.

`jitllm_gemma_quality_schedule ARTIFACT IDS NEW_OUTPUT 128|1024` holds context 4,096,
configured maximum rows 1,024, local F16 capacity 2,048, global F16 capacity 4,096,
slot count 1 and full-head capacity unchanged. Both schedules fund the same
1 GiB publication vector and select plain norms, norm/RoPE, norm/Add, routing
and reduction throughout all 30 layers. Actual plan counts must be 121/60/90/30/30;
shared products and row products remain zero. No owner attention, state-only
pruning or future-plan callback runs in this full-head scalar helper. The
lookahead option remains enabled but unused because `Chunk` supplies no forecast.

History 13 runs two fresh processes at schedule 1,024 and two at schedule 128.
Authenticate source, exact inputs, successful official completion and the full
finite 1,024-head byte-exact repeat for each schedule. Then calibrate on their
frozen first outputs. Production 1,024 is anchor A; supported 128 is B. At each
row select A's highest two token IDs, resolving ties toward the lower ID, and
compute `abs((A[top1]-A[top2])-(B[top1]-B[top2]))`. The single bound is the
existing nearest-rank p99: sorted movement at index `ceil(0.99 * 1024) - 1`.
Freeze its receipt identity before exposing any oracle output from history 13
or 14. The bound represents supported scheduling variation, while exact own
repeats remain a separate zero-difference requirement.

History 14 runs two fresh production 1,024 native processes and freezes their full
finite byte-exact heads before stock runs. Stock uses the same raw checkpoint,
1,024 IDs, context 4,096, physical C1, batch/ubatch 1,024, F16 local 2,048/global 4,096,
normal ring (`swa_full=false`), independent KV (`kv_unified=false`), ordinary
public dispatch and complete heads. Public backend teardown, owned-container
absence and official completion precede comparison.

Apply the unchanged history 13 bound to **every** strict native/reference choice
difference on history 14. The oracle margin is `reference[reference_argmax] -
reference[native_argmax]`; zero outside-bound differences are required, including
all 1,024 rows. Separately require PPL increase at most 3%, using all-vocabulary
FP64 normalized likelihood and exactly 1,023 within-history targets. Row 1,023
has a predicted choice but no next-token likelihood target. Retain every strict
mismatch and its margin in external raw data; report strict zero-margin status
separately. Official exit 0 means measurement completed, not qualification passed.
No history 14 recalibration, tolerance increase or outside-bound exception is
permitted. This control does not establish batching, performance or depth.

The source-only capacity test checks both host mask schedules against the same
causal history. In the device mask, visibility depends on each actual position,
physical capacity and window; a smaller read span excludes future cells that
are masked in the larger span. The schedules can select different tensor and
kernel shapes, which is the operational variation being calibrated. Both
schedules preserve all routing and reduction fusions.
