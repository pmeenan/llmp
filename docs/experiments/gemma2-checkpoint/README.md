<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma2 checkpoint and kept-state adoption

This internal foundation transfers the checked Gemma3 completion-aware
checkpoint path to the approved Gemma2 Q8_0 runner. It adds no serving route,
borrowed-feature API, checkpoint file schema, importer or quantization change.
The existing 8K/two-slot options, independent prefill, cap50 owner decode,
primitive fallback and fusion defaults remain unchanged.

A layout identifier binds family, F16 storage, context, per-owner chunk envelope and
local/global capacities. The bounded context 8,192 / max_rows 128 per-owner recipe retains a
4,352-cell local ring. A future 256-row total wave envelope is separate and
does not change that per-owner layout. A restored footprint must be an ordered set of whole
2 MiB extents (with the layout's final short extent), region zero, covering
every initialized logical range at the declared completed position. Malformed,
duplicate, missing, unordered and out-of-layout extents are refused before
publication or footprint adjustment.

PrepareRestore releases the destination's state protection while preserving
all selected peers, funds and retains its exact footprint, then records the
logical copy ledger. A clean capacity refusal preserves the completed cursor;
an unproven failure quarantines state or faults the cohort. The pending state
cannot execute, grow, spill or silently disk-restore. It cannot be removed from
the selected cohort; unchanged held selection still takes the existing fast
path after this exclusion. Clear cancels the pending restore. CopyState
supports explicit restore writes only during that transaction, and completion
advances the cursor only after independently retired copies contiguously cover
every needed logical range. A later extent or failed copy cannot stand in for
a missing first range. Node-owned pinned buffers remain retained if retirement
cannot be proved.

Adopt accepts an identical-layout nonempty footprint only into an empty idle
healthy slot. It records kept disk state without acquiring backing until
Restore. Named owner-only spill files can be reopened with keep=true using the
existing LiveState protocol. No file or record format changes. Successful work
invalidates the earlier whole-checkpoint flag, and Clear, growth and restore
still refresh their execution closures.

The focused controls pass on Spark B on 2026-10-07. Metadata controls cover uninitialized
lifecycle refusal and extent validation at positions 1/4096/4224/4353/8192.
GPU controls cover proven/partial/invalid copies, capacity refusal, unchanged
peer state, pending cancellation and held-cohort selection. The wrapped-ring
control saves at 4355, checks exact restore-next heads against a live peer,
spills and proves source-node retirement, then reopens the actual named files
in a fresh node with keep=true. It adopts 4356 and compares the next full finite
head byte for byte with the independently continued source peer. A failed
teardown retains its node, open directory and files until process exit.

## Checked result

The installed three-step job completes with exit zero: locked narrow native
build, focused CPU/descriptor controls, then the four new GPU lifecycle
controls. All 19 tests pass without disabled cases or a full-suite run.

| Control | Passing tests |
| --- | --- |
| New Gemma2 lifecycle/extent metadata | 2 |
| Existing Gemma3 lifecycle/extent metadata | 2 |
| Existing Gemma2 graph / plan | 6 / 5 |
| New Gemma2 checkpoint GPU lifecycle | 4 |

The GPU controls check proven complete copy coverage and clean capacity
refusal with exact preserved bytes, pending-restore cancellation and forbidden
work, unchanged/changed held selections, and actual wrapped-ring restore plus
fresh-node kept adoption. In the last control the local capacity is 4,352, the
snapshot is at 4,355, named-file adoption is at 4,356 and the next finite full
head equals the independently continued source peer at 4,357 byte for byte.
The source node retires before the destination opens keep=true; both nodes
retire successfully and private files are removed. This is native own-state
proof, not a new stock-quality, performance or HTTP qualification.

The measured source is based on `ded9443`, with SDK
`aarch64-c09daba6ac31edee`, locked GGML v0.6.0/d812 and unchanged prepared
artifact `eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870`.
Actual checkpoint GPU test SHA-256 is
`8741810eaf6a5a47e6307ac90e03cfd4a3be9173f316d33746653c09bf90d39f`;
metadata test `3c1a87ed57f2abae0a9c62d4df63cd6c77163fa58ca3a9f0051a18363e8b56a5`;
receipt `874aaf7a5967cfbe91054e0d8fc1a0630f952e0e8eb54b831d09719f1e08ce89`.
Whole-source checksum sync and an empty dry run precede the build, and every
step rechecks the frozen source/method inventory and actual binary binding.
The existing fully authenticated/deep import is reused and its retained
source stat and metadata/index identities are checked before inference.

Source and actual binary/receipt bindings, private files, logs and installed
supervisor receipts stay outside Git under
`/tmp/jitllm-m35-coordination/gemma2-checkpoint-raw` and Spark B's
`~/scratch/m35-gemma2-checkpoint/run1`. The focused installed GPU job has a
600-second limit and stops on failure. No full suite, fresh stock inference or
performance comparison is needed for this internal state foundation; arithmetic
and reference evidence remain in the prior C1/C2 reports. Actual template
qualification, bounded settings/serving adapter, compatible joint prefill,
departures and broader serving gates remain next work.

Per-task TensorFold HEAD checked on 2026-10-07 remains
[`041d14a94e951834470fd514ed33e65b8be1059a`](https://github.com/ashhart/TensorFold/tree/041d14a94e951834470fd514ed33e65b8be1059a).
The qualified native 1.0.0 CUDA README still documents no Gemma2 GGUF comparator;
no new reference comparison is claimed by this checkpoint slice.
