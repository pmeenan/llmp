<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dense source dialects and ModelOpt MXFP4

Qwen2, Llama and Mistral each have one canonical mapping across GGUF, HF
safetensors and EXL3. Source dialects select the existing wire names and part
rules while preserving canonical slots, component roles, parameter fallback
order and performance policies. In particular, EXL3 retains its trellis/sign
parts and float32 epsilon conversion; Llama/Mistral retain GGUF normal rotation
and HF/EXL3 neox rotation. The legacy mapping accessors return the canonical
entry. Selected-head extraction and unbound writer ownership select the source
dialect before traversing its tensor rules.

The reader also describes ModelOpt `MXFP4`, `W4A16_MXFP4` and
`W4A8_MXFP4_FP8` weight exports generically. Packed U8 E2M1 codes have shape
`[N,K/2]`, and U8/E8M0 scales have shape `[N,K/32]`. Both parts keep their
physical names, dimensions and bytes; the normalized descriptor uses the
existing `mxfp4` split layout. Activation scales remain separate dense tensors.
This addition grants no activation arithmetic or new GPU implementation.

The source convention was checked against NVIDIA ModelOpt commit
`c00ac438331a6308e6cb40cf961935b7fcd09ea5`, including its quantization-format,
MXFP4 tensor and HF-export implementations. NVFP4 remains distinguished by its
E4M3 block scales. Both NVFP4 and MXFP8 now validate the complete physical scale
grid rather than accepting an equal byte count with different dimensions.

## Validation

The integrated source is based on `4533033c`; the native-build source manifest
SHA-256 is `70073a27a7771a51bb98dfbefd6f936e10d64c0af638fd6b069190d566e8220f`.
The subsequent check manifest is
`425ca8f4df184d19ce8c5a467cfc7d03b4c3665d70370e65577970bed48bdef3`:
only the independently reviewed quality collector and its Python tests changed
between those manifests. Remote Git HEAD remains historical and does not
identify the synchronized source tree.

On Spark B, the seven affected native targets built, followed by the shared
test consumers. The focused import/mapping set passed 93 GoogleTest cases
with no skips. It includes reader classification and malformed-part controls,
one complete MXFP4 reader-to-normalization fixture, and replay of every indexed
part against the original source bytes. Canonical registry and dialect controls
cover unresolved aliases, unknown roles and unconsumed architecture metadata.

The shared complement passed 3,582 registrations, including 3,751 inner
GoogleTest cases and 21 other checks. Together with the focused set, the
non-overlapping union is 3,675 registrations and 3,844 GoogleTest cases, with
zero failures or skips. The quality collector's separate Python module passed
all 64 tests. All 2,920 source-file guards matched before and after the checks.

Six affected-model controls also passed on Spark B, with one completed inner
case each and no skips: Qwen2 GGUF source/spec equivalence; both Qwen2 EXL3
source/artifact rate fixtures; published artifact and composed-pair approvals;
and Gemma2/Gemma3 stored-spec/tensor-table controls. These eight seconds of
functional checks are not a serving speed or quality measurement. The 11
changed C++ files pass the pinned formatter. REUSE/header validation passed on
Spark B for 2,559 worktree paths with no problems; all 241 files of the reused
license-tool snapshot matched its pinned manifest. The four documentation files
were synchronized separately after the native checks; their 2,921-file source
guard also matched before and after license validation.

Whole-source and adversarial reviews found and closed an unresolved-alias
validation gap, synthetic fixtures that mutated the canonical rather than the
resolved source mapping, and selected-head lookup that traversed the canonical
wire table. Original failed attempts remain in the external evidence; subsequent
passes do not relabel them. Raw logs, test inventories, inner XML and compiler
receipts are retained outside Git under
`/tmp/llmp-m35-orchestration/resume-wave-r1` and the supervised Spark job store.

No artifact payload or stored spec changes for the existing dense dialects;
their previous source rules are selected through the canonical entry. Other
architectures' mapping consolidation and the remaining quantization ingress
are still U3 work. This slice does not close U3 or qualify model quality/speed.
