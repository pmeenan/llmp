<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Current llama.cpp reference

New comparisons use official **v0.6.0**, source
[`d81235049384534c167caea52b85a694f6103d14`](https://github.com/ggml-org/llama.cpp/tree/d81235049384534c167caea52b85a694f6103d14),
with the matching ARM64 CUDA image recorded in [pins.json](pins.json).
The official latest-stable release API was rechecked on 2026-10-07 and still
reports v0.6.0; newer daily prereleases do not change this frozen reference.
The release's `full-cuda13-b11429` image has that exact source revision in its
registry configuration. The moving `full-cuda13` tag was already newer at
entry; the `full-cuda13-v0.6.0` alias was absent. Freeze the recorded digest,
not a moving tag. Spark B checks completed on 2026-10-06: exact executable
build/revision,
resolved dependencies and native `sm_121` ELF presence pass. Three unchanged
helpers compile/link and refuse invalid arguments before creating model
contexts. No model quality or performance checks ran at this new pin.

The [release](https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0) includes
Clef text and vision support and `/v1/systemone`. This establishes upstream
availability, not native jitLLM Clef support or a measured GB10 speed result.
The owner's requested dependency refresh covers new comparisons. All existing
b10964/b29c606/image837 experiment launchers, results and serialized-state
receipts retain their original pins. Sequence/session format changes require
fresh states; old serialized states are not migration inputs.

## Bounded compatibility preparation

[reference.sh](reference.sh) provides only `pull`, `inspect`, `compile` and
`usage` modes, with a new owned name such as `llama-refresh-inspect1`. It uses
`~/.local/share/jitllm/llama-reference-v060`, an authenticated staged header and
helper closure, the existing pinned SDK and the existing owned-container
retirement helper. Run each step through the installed Spark supervisor and
wait for official retirement. The recorded preparation ran on Spark B;
new comparison tasks use their own bounded supervisor queues and evidence.

Pull fixes the ARM64 manifest and authenticates its configuration/revision.
Inspection records actual executable versions, `/app` libraries and their
resolved runtime dependencies. Compilation uses the existing c09 SDK's Clang
and GCC runtime against those captured image libraries; the release image
itself does not declare a compiler. Three unchanged reference helper translation
units cover joined decode, corpus scoring, vocabulary-only preparation and
assistant transactions. Pre-context argument refusals check their loader and
entry points in the same image. No models are mounted and no inference or
model contexts are requested. These compile/link/entry controls are
prerequisites; they do not exercise
model-context ABI behavior or qualify numerical behavior, cache topology,
assistant rollback or performance. Preserve failures separately from later
successful checks.

The old `llama_batch`/`llama_decode` compatibility adapter explicitly preserves
mixed token and embedding MTP batches at this release. Assistant cache classes
and next-token feature getters remain present. These source observations
justify the narrow compile probes; private class compatibility still needs the
actual object execution and later operand/model evidence. New recipes must
set normal
ring-cache policies explicitly (`swa_full=false`, `kv_unified=false`, F16 K/V),
record effective capacities and query widths, and freeze independent native
outputs before exposing the first reference outputs.

TensorFold task entry resolves to
[`cb2ebf0540f42604e2759b2ddef497861e928248`](https://github.com/ashhart/TensorFold/tree/cb2ebf0540f42604e2759b2ddef497861e928248),
version **0.6.6**. Its Gemma recipe remains MLX-only and describes a different
DFlash drafter; it supplies no matching CUDA GB10/Q8-assistant comparator.
[pins.json](pins.json) retains the exact README, package and Gemma recipe hashes.
No TensorFold job is needed for this reference refresh.

## Recorded checks

The successful image inspection completed all five queued steps; helper
compilation/entry controls completed all four. Both officially retired with
exit zero, and both owned containers were checked absent. Image executables
report **0.6.0-dev, build 11429, commit d81235049**, built with GNU 14.2.0 for
AArch64. The three helpers use pinned SDK Clang 22.1.8 and the static GCC 16.2
runtime. The image's resolved CUDA runtime is 13.4.49 and cuBLAS is 13.7.0.27;
compiler, toolkit, library and disassembler versions are distinct identities.
[pins.json](pins.json) records actual executable/library/helper hashes, SDK
installation provenance and official outcomes. No compiler or host packages
were installed.

The first inspection failed after the successful exact image pull: this host's
Docker inspection reported the manifest digest as `Id`, whereas the first guard
expected the config digest. The correction accepts only either frozen digest
and also authenticates the exact repository digest, Linux/ARM64, complete
configuration and root filesystem layer identities. The failed job and its
metadata remain external; this was a guard correction, not a different image.

New numerical, cache/state, assistant and performance qualification remain
open. The native GGML migration and Clef integration have their own source and
checks; these reference probes establish neither.
