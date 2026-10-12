<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Original Qwen2 BF16 likelihood smoke

2026-10-12: the original checkpoint and shared FP16/EXL3 scorers completed
one authenticated natural-language passage. This establishes a usable primary
reference and aligned scoring, not a model-quality or migration-speed gate.
No numerical recovery has been adopted.

## Inputs and reference

Original `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`: complete safetensors
988,097,824 bytes, SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
All 290 tensors are BF16; original configuration and tokenizer are authenticated.
The existing approved Apache-2.0 source and audited reference image are reused.
This is original stored precision with a declared arithmetic recipe, not ideal
real arithmetic or recovered original weights from a quantized checkpoint.

The passage is 221 UTF-8 bytes, three complete lines about client timeouts,
starting at byte 403006 of the frozen M3.6 baseline document corpus
([corpus construction](../m36-baselines/README.md)). Text SHA-256:
`ae10676ed92021fb8589a5477e5ec042564d5428be2d2e783d40ccf689571ccf`.
Original offline tokenization produces 43 ordinary tokens, without BOS or chat
template. Their i32le SHA-256 is
`1b1c01818993d60cad3b4046c2ba22c24d81af32434fc65977a3faf12a4a9e40`.
All used IDs have identical unique token strings and ordinary-token types in
original tokenizer JSON and the official F16 artifact's GGUF metadata; their
byte alphabet reconstructs the entire passage. This proves these input/target
meanings, not complete vocabulary/merge equivalence or weight-conversion lineage.

The independent reference uses stock `Qwen2ForCausalLM` in immutable image
`sha256:5d405cce512e8bb8c9886501fbec2b4323565a691f6d838b76a076cd5d787537`
([existing image closure](../exl3-reference/runtime.json)): Torch 2.14.0,
Transformers 5.17.0, safetensors 0.8.0, tokenizers 0.23.2, NumPy 2.2.6.
BF16 parameters and emitted logits, eager attention, explicit positions, no KV
cache, autocast, TF32, reduced-precision reductions or `torch.compile`.
The image's stock eager Triton dispatch remains active. No remote model code,
dependency installation or image refresh occurs.

Native scoring uses official F16 artifact
`b93cdc326ba4f4c1c71da613503ecd848cd2a0caf122214f26e04588a12a9073`,
the shared target-only scored orchestrator and real GGML backend, six chunks
(8/8/8/8/8/3), F16 KV and explicit attention. Each complete F32 vocabulary row
is reduced by stable double logsumexp. Row r predicts input[r+1]; the last row
is finite-validated but unscored. All 42 global coordinates and target IDs are
checked. Both actual bound shape classes and implementation names are recorded.

EXL3 scoring reuses the same scorer and plan recorder with approved 4.0-bpw G
artifact `6e96e499ea9b859ea726327d4f82fb1739cb116ce4e9d3541d1ba67506efb4b4`.
Its complete tokenizer JSON is byte-identical to the original (SHA-256
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`).
The frozen EXL3 tuning supports seven-row and one-row steps, so this run uses
7/7/7/7/7/7/1 with the same teacher histories and 42 target coordinates.
An absent three-row tuning bucket explicitly refuses; no tuning, arithmetic or
kernel changes were introduced to force the FP16 schedule. Each selected shape
records 657 implementation entries, including existing EXL3 products and vector
attention. Entries and successful run/record calls are not physical launch counts.

## Observation

| Recipe | Targets | Mean NLL | Perplexity |
| --- | ---: | ---: | ---: |
| Original BF16, eager/no cache | 42 | 6.496176475891133 | 662.6033037486249 |
| Shared official F16, existing unfused | 42 | 6.542964050104692 | 694.3415942215227 |
| Shared official F16, existing fused | 42 | 6.542964050104692 | 694.3415942215227 |
| Shared EXL3 4.0-bpw G, existing recipe | 42 | 6.458775094111763 | 638.2787461445553 |

Each original case completed two forwards. The second repeated all full logits
exactly; repeat parity is diagnostic, not an adoption requirement. A separate
33-token uniform-random input also completed, but is alignment stress rather
than language-model perplexity evidence. Native fused/unfused arms have equal
per-target NLLs here despite different selected norm/state-store implementations;
this does not establish equal full logits.

Native minus original mean NLL is +0.0467875742135595 (perplexity +4.79%) on
this tiny passage, with 16 lower and 26 higher per-target losses. No acceptance
threshold was assigned. The official F16 publication has no proved conversion
lineage to these original BF16 weights; cache and arithmetic recipes also differ.
The observation cannot attribute a quality difference to any one optimization,
qualify broad language quality, or establish production-phase/context behavior.
EXL3 has lower empirical loss than original BF16 on this sample; that is not
a general quality improvement or a ranking of formats. Stored weights, native
phase/cache arithmetic and chunk schedules differ, and 42 targets are insufficient
for a broad quality conclusion.

## Execution and limits

Original runs on `spark`; native builder runs on `spark-b`, both NVIDIA GB10.
The original driver is 580.178.04, reference CUDA 13.0. Installed `spark-job`
supervises exclusive model work; source acquisition is a separate shared job.
Original source/helper/input hashes are checked before and after; both containers
exit without OOM and retire with PID 0, and no compute process remains.
Original host/cache cap is 32 GiB, allocator cap 4 GiB, physical startup payment
44 GiB including 6 GiB margin. Sampled CUDA counters are explicitly global
(RE-041), not process-external allocation accounting. Observed peak allocator
reservation is 1,096,810,496 bytes; minimum MemAvailable is 122,143,637,504 bytes.

Native's existing 8 GiB catalog cap, prepaid compiler/state/source inventory and
6 GiB margin remain. The complete wrapper/native process shares a 10 GiB host
and page-cache cap with zero swap; startup requires 24 GiB physical availability,
conservatively double-counting native host allocations. Used-token proof and
log/XML outputs are bounded. Two raw cases pass with no skips; all source,
artifact metadata/header, binary and input guards pass. Observed host/cache peak
is 289,316,864 bytes, not total GPU/unified occupancy. No timing gate is inferred
from these short job durations.

The first evaluator attempt failed during its first forward because the bounded
Docker temporary filesystem disallowed loading Triton's compiled CUDA helper.
The corrected attempt permits execution in the same 64 MiB GPU temporary cache;
no model/kernel policy changes or downloads are repeated. Both attempts retire
cleanly and remain recorded.

Raw outputs and frozen source/recipe/review packages remain external under
`/tmp/llmp-m36-coordination/` locally and the matching Spark scratch directories:
original `qwen2-original-reference02` acquisition, `qwen2-original-reference03`
evaluation; native `corpus-qwen2-models02`. Original result SHA-256
`f2280c7afcaa4b139d398aa9338d5a9b5d1c6278b23bc199427dc371d3ba6771`;
native result SHA-256
`bb2efca7f1743a16c9310fd85cac2c02fff590ded85613d47849c840914e822f`.
Raw model logits are not checked into Git. Main native integration passes all 114 affected registrations without skips;
its two actual natural-ID policies also pass, with identical per-target losses
to the builder and complete postguards/retirement. Main actual result SHA-256
`1907f99d3162463c50e1989a494816d11294efa5963a42f9feb3c9248b6e4206`,
external package `corpus-qwen2-models-main03`. The shared scorer additionally
passes ten host controls and two input-bound controls on the builder.

The EXL3 builder passes 20 affected host controls without skips and one actual
scoring case with all source/input/binary guards and retirement checks passing.
Observed host/cache peak is 276,508,672 bytes. Raw external package
`corpus-exl3-models01` result SHA-256 is
`ea90ebd0e0632945616e74ab7b7e5ae9430d2d50bff9b0cc0e0ecad35fd8f6a4`.
This adds a common-scorer observation, not EXL3 serving, state or speed qualification.

Main integrates the common recorder with a native build and 124 affected
registrations passing without skips. Both FP16 policies and the EXL3 scorer
then pass actual model replay with unchanged per-target losses, complete
postguards and zero surviving compute processes. Main package
`corpus-exl3-main-models02` result SHA-256 values are
`96f950f32cce70a5d1aba59cb95fbd3643940fbe1f63b46e664688959fb323c4` (FP16)
and `2f801e20c41d7811df2425fc86b38aa4c2742c940f1d588e27c7818f7336df4e` (EXL3).
