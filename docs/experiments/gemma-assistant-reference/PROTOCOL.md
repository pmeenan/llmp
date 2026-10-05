<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Original-image Gemma assistant reference protocol

This diagnostic uses the unchanged original llama.cpp image at
`sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`,
source `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`. The small external client links
its exported libraries against an authenticated closure of the exact source
headers. It neither rebuilds math nor changes fusion, graphs, aliases or graph
outputs through evaluation callbacks. The scope is the approved 26B-A4B
Q4_K_M target and its approved Q8_0 assistant.

## Inputs and physical contract

Use the previously authenticated 1,024 canonical tokenizer IDs, including BOS
once. Owner zero consumes the first 64 IDs; owner one consumes the first 65.
Target prefill is one original physical chunk per owner, with ubatch 128,
context 4,096 per owner, separate streams (`kv_unified=false`), F16 K/V,
Flash Attention enabled and `swa_full=false`. This selects local physical
capacity 1,280 and global capacity 4,096. Graph/fusion defaults remain enabled.
Enable unmasked nextn output before prefill and immediately copy each owner's
last post-finalnorm feature and lowest-index full-head argmax. The feature is
at P−1; the pending anchor has position P and has not been inserted into KV.

The original split-cache getters expose local layer 28 and global layer 29.
Require the actual F16 views to be `[D,KVH,256,1]`, with D/KVH 256/8 local and
512/2 global, packed cell-major strides and the exact owner stream offset.
Validate each original storage root, its physical capacity, complete span and
stream count. Export raw global V separately; it is the raw K-as-V source,
not the rotated K cache. Inspect original cell membership and positions for
all physical cells: initialized cells are exactly 0 through P−1, with no
current-P cell. Observed occupied physical end independently yields read
width 256. Original source explicitly clears complete buffers at allocation;
also verify every unoccupied physical payload cell is zero. A mask does not
establish initialized storage.

Export both canonical read-width arrays for native staging and complete
physical K/V arrays as immutable witnesses. Capture opaque public sequence
state bytes as an additional equality witness; no parser or translation of
that serialization is introduced. All these bytes and the cell metadata must
remain unchanged after every assistant step and repeated chain.

## Assistant and calibration order

The assistant context uses the original `ctx_other=target` and MTP context
type. Construct both contexts before target prefill: the pinned shared-cache
constructor resizes shared cell metadata, and that resize resets occupancy.
Shared layer tensors bypass new allocation/clear; the constructor clears only
its newly allocated buffers. Frozen witnesses are taken after this setup and
prefill, then checked immediately and after each assistant step. Both token
and F32 feature arrays are supplied, following the pinned
speculative driver's allocation and graph input contract. Query position P
remains constant through one and three recurrent steps. Step zero uses the
target feature and pending anchor; later steps use the preceding assistant
postprojection and lowest-index selected token. Each output contains the
complete 262,144-value head and 2,816-value postprojection, all finite.

Acquire C1 first. An optional C2 serial arm calls one original row per owner
and is labelled a serial C2 oracle. A genuine two-sequence, physical batch-two
arm is separate. Neither C1=C2 numerical identity nor fastest batching is
asserted; native joined dense products can use different column arithmetic.
Each shape takes independent one/three-step chains and exact same-shape own
repeats from identical frozen target inputs, with full byte comparisons
(including signed zero). Complete source/input/output identities and
successful supervised retirement precede cross-analysis.

Only stage-zero target cache/feature/anchor inputs may be supplied to the
native author before its source and endogenous one/three-step own repeats
are frozen. Stock step-one/two incoming features and anchors are themselves
stock outputs and remain withheld, along with heads and postprojections.
After that native freeze, unchanged-source same-input replay may consume all
stock incoming tensors to isolate each step's arithmetic. Its results must
be labelled posthoc same-input replay; they do not replace endogenous
recurrence evidence or retrospectively calibrate a bound.

## Raw schema and completion

Files are little endian and owner-only, created exclusively in a new
directory. Owner directories hold stage-zero feature/anchor, four canonical
cache arrays, four complete physical witness arrays, full physical cell
positions/membership, descriptor metadata and opaque state. Shapes and byte
lengths are checked before extraction. No process address is serialized;
descriptor metadata contains logical dimensions, strides and relative
stream offset only. `steps-1` and `steps-3` contain arrays ordered
`[step,owner,value]`: incoming features/anchors, full heads, postprojections
and independently repeated heads/postprojections. A separate exclusive
receipt authenticates all source, header, library, input and output hashes.

The client bounds known retained host vectors to a small fixed shape (at
most two owners and three steps); original image model/backend allocations
remain reference-owned and are not a jitLLM memory qualification. Extraction
and state checks are untimed. No performance claim comes from this client.
On refusal it fails the process without unwinding borrowed contexts; successful
cleanup synchronizes both contexts, then destroys the assistant before target.
Every build, probe, model read and acquisition uses the installed Spark job
supervisor with GPU admission and timeout at most 600 seconds. Missing exports,
unexpected types/strides/occupancy, nonfinite payloads, repeat differences or
cache mutation fail this seam; no reinterpretation or tolerance widening.

Raw inputs, tokens, heads, projections, state and logs remain external.
Git retains the reusable harness, aggregate availability/numerical findings
and authenticated reproduction identities. This is no target-quality pass,
assistant verification/rollback, serving, speculation, or model qualification.
