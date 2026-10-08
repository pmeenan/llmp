<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma stock-minus fusion controls

Follow the [norm-family protocol](../gemma-reference-fusions/PROTOCOL.md):
the same approved Gemma 26 GGUF, pinned llama.cpp b29 image, exact first
1,024 War and Peace IDs, context 4,096, eight 128-row chunks, F16 KV,
FlashAttention and complete 262,144-entry F32 heads. All later prefixes
remain teacher-forced. There is no template or speculative decoding.

The new external controller removes one stock fusion family, preserving
all other original gates and CUDA math launchers:

- `all`: the original stock selections; require complete head byte identity
  with the original production acquisition before attribution.
- `no_routing`: skip only the original top-k/MoE fusion block. Preserve
  the specialized weighted reduction and its allocation dependencies.
- `no_reduction`: skip only the specialized scaled MoE weighted-reduction
  match. Remove its allocation dependencies. Other original MUL/ADD fusion
  gates remain eligible; record those actual fallback selections.

The controller also defines `no_norm_mul` and `no_softcap` for individually
removing the corresponding stock gates. They are **not measured in this
batch** and supply no numerical or adoption evidence.

Check the exact upstream source SHA before constructing the controller;
reuse the pinned image's floating-point launchers and libraries. Build
only the controller translation unit, including its integer batched-pointer
helper, with the declared SDK NVCC 13.4.92 and image GCC 13.3.0. Preserve
upstream fast-math flags, explicit GB10 target and graph support. Require
policy banners and the new controller's complete stock-head fidelity.
Logging must not retain graph intermediates or change model tensors.

Acquire both primary omissions twice and require whole-output byte
identity between their independent repetitions. Authenticate the original
native-only calibration SHA
`2c97b2a4d65277f00b242180d7f54128a395284490d84db0db24e634adcfc901`
and the exact original native/stock head files. Never replace calibration
or widen its zero p99 allowance to accommodate the omission results.

The first screen compares **every head** for actual row byte identity,
strict argmax IDs and maximum raw delta. It computes full-softmax TV and
target NLL only on fixed rows 0/127/128/512/1023 and the first stock byte
and argmax changes. The final retained row has no next-token target.
These selected rows are explicitly labelled; no sampled PPL or whole
quality gate is inferred. Strict argmax counts are not outside-noise
counts: this screen does not evaluate winner margins for that purpose.
The unchanged calibration continues to describe its original native paths.

Retain actual host fusion selections, source/binary identities, repeat
hashes and the differences from both complete stock and native ordinary.
No omission becomes the competitive or quality reference. No native
operator, format, compiler pin, default or support status changes here.
