<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen ordinary MMVQ routing repair

Collection requires complete binary/source/dependency/identity pins, this
reviewed protocol committed before observation, and separate external enablement. Preserve original remaining-C1 Qwen failure: three valid pairs,
TTFT median 1.12149868 > 1.10. No input, bound or sample is replaced.

The sole production repair is GGML capability selection: on GB10, ordinary
unstacked Q6_K/Q8_0 physical columns 5..8 favor the already qualified upstream
MMVQ reduction instead of VecQ. The column class shares the same VecQ schedule;
only column 5 has this new measured performance comparison. Automatic GLU
rewrites stay separate; directly supplied prepared fused GLUs and explicit
product arithmetic/accumulation/hints retain their contracts. Routed/grouped,
other encodings/devices, columns 1..4 and >8 remain unchanged. There is no new
kernel arithmetic, numerical grant, model-name dispatch or shape whitelist.

Use the frozen upstream reference from the prior transfer/C1 registration and
one newly built repaired candidate on Spark B. Retain source maps and complete
compiler-object ledger, executable bytes and the unchanged ten-library closure.
Reconstruct exact observed implementation identities with production IdentityOf
and exact module-token replacement, as before. Because capabilities.cc changes
its source identity, the ledger must explicitly authenticate that source delta;
all actual declaration metadata except the exact module/source digest remains
identical. No wildcard hash removal. Missing pins or unexpected source changes
stop collection.

1. Reuse the existing `remaining-domain-transfer/run-arm.py` unchanged with a
   new disabled head-state pins file enrolling only Qwen candidate plain in a
   fresh output namespace. Same five literal Prompt IDs, context 512, slots1,
   chunk 16, 8+8 output tokens, full heads/kept rows/history/state/spill controls.
   Compare all semantic results exactly to the retained reference plain arm.
   Require complete exports and successful retirement. Independently inspect
   full completed-plan relationships. Five-column Q6_K/Q8_0 sites must now use
   upstream MMVQ (ordinary or declared prepared form), with separate GLU.
   Prepared sharing is allowed only with exact Q8 preparation/source/stride
   proof; preserve each unaffected logical/physical relationship and identity.
   Row1 continuation/decode retains its previous candidate route. Unknown
   identity, missing dose, incomplete rows or any unexpected semantic difference
   stops this registration without retry or changed tolerance.
2. Only after that positive prerequisite, reuse `remaining-c1-speed/run-c1.py`
   unchanged with new disabled speed pins/output namespace and the repaired
   runtime. Collect exactly six Qwen arms in AR/RA/AR order on Spark B. A remains
   frozen upstream runtime; R is repaired candidate. Same 128-character corpus,
   context 512/chunk 5/slots1, one warmup request 0 and one measured request 1,
   128 outputs, temperature0, cached 0, normal transport. No priming or debug
   observer. All arms must be valid; invalid stops without retry. All three
   paired slowdown medians must be <=1.10 (TTFT R/A, decode A/R, wall A/R).
   Do not stop on a valid individual pair's ratio. Retain every sample and
   independently recompute from raw timestamps; report system memory loss.

No teacher or reply recollection is enrolled. Existing Qwen 996 wide teachers
have 997-row phase4 products, outside5..8; the previous source/reuse proof and
artifact/input/config pins remain binding. The two current Qwen replies each
had 66 prompt tokens, zero cache, chunk 4096 and target-only scalar decoding:
whole 66-row prefill (last selected head 1) and one-row decode do not enter5..8.
No internal grouped/routed operation changes. Existing raw SSE/request/config
receipts and semantic review remain the quality evidence, with their original
128-token incomplete-answer limitation; this is source applicability, not a
claim that new replies were collected. If source review finds any actual 5..8
ordinary site in those executions, this reuse condition fails before enabling
collection, and a new fixed recipe must be registered.

The [head/state pins](head-state-pins.json) and [speed pins](speed-pins.json)
remain statically disabled. Executors are the unchanged
[head/state runner](../remaining-domain-transfer/run-arm.py) and
[speed runner](../remaining-c1-speed/run-c1.py), bound by their registered hashes.
The original [speed failure](../remaining-c1-speed/RESULTS.md) remains evidence.

The completed source/domain reuse review has SHA-256
`f520d243a9b5e607d3adccceb4860d72f0052108d7e7bff3d1025c71f37c7125`.
Actual warmup/measured prefill tails are Gemma31 54/234, Gemma26 822/234,
and DeepSeek 3398/3559; their full chunks exceed eight rows. Runtime and stored
contracts bound verification to four rows and assistant work to one/three rows.
The review joins ordinary projections, gathered heads, HC flattening, compressor
projections and grouped products to source and prior complete plans. This is
source-backed applicability evidence; the C1 runs emitted no per-operation
trace. It supports reuse of those three speed results and the unchanged
head/state, Gemma31 C2, Qwen reply and wide-teacher controls at their recorded scope.

Additional source changes require a new applicability decision. Frozen numerical
negatives, bounds and expected values remain unchanged. This is no new device,
encoding, maximum-context, general batching or full-answer qualification.
