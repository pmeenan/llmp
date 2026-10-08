<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# First dense checkpoint and numerical reference

D-051 selects **Qwen2.5-0.5B-Instruct, the official FP16 GGUF**, for M2's
GGML backend proof and M5's end-to-end fixture parity. D-052 adds a required
[EXL3 companion](exl3-bringup.md) in M2, including upstream performance gates.
Use the already pinned
llama.cpp reference on Spark. This resolves M0 question 4; it does not
establish native llmpalooza support or close the separate backend-proof scope,
artifact schema, source-dependency or toolchain tasks.

## Exact input and rationale

| Item | Selection |
| --- | --- |
| Import/reference input | [`Qwen/Qwen2.5-0.5B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/9217f5db79a29953eb74d5343926648285ec7e67), revision `9217f5db79a29953eb74d5343926648285ec7e67` |
| File | `qwen2.5-0.5b-instruct-fp16.gguf`, 1,266,425,696 bytes |
| SHA-256 | `8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc` |
| Weight representation | Published F16/F32 mix, no low-bit quantization: 170 F16 and 121 F32 tensors |
| Reference engine | llama.cpp `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`; exact ARM64 image and library/tool identities in [pins](experiments/first-slice/pins.json) |
| Upstream base cross-check | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775), revision `7ae557604adf67be50417f59c2c2f167def9a775`; config, tokenizer files, model card and license only |
| Model terms | Both selected repositories supply matching Apache-2.0 license files; hashes in the pins. External test/model data, not incorporated implementation; no weights redistributed |

This is a small dense, text-only model with full-attention KV state, RoPE,
RMSNorm, SwiGLU and Q/K/V biases. Its published GGUF exercises small F32
tensors, bulk F16 tensors and duplicated tied weights without adding MoE,
sliding-window rollback, recurrent state or multimodal components to the
first proof. The file's 24 layers, hidden size 896, intermediate size 4,864,
14 query heads and 2 KV heads match the pinned base config. These are
artifact observations, not memory-admission envelopes or performance claims.

The inspected GGUF stores **630,167,424 elements** across 291 tensors. Its
input embedding and output matrix each occupy 272,269,312 bytes and are
byte-identical; counting that tied matrix once gives **494,032,768 elements**.
Do not infer logical identity from names or double-count physical backing.
Import may deduplicate only with verified content/representation identity
and correct alias/extent leases; both duplicate-storage and shared-storage
views must produce the same computation. The artifact-schema task decides
the physical layout and records padding/read amplification.

Qwen's official F16 publication and explicit license file make the source
identity straightforward. SmolLM2-360M-Instruct was also considered at
revision `a10cc1512eabd3dde888204e902eca88bddb4951`; the existing Gemma/Ornith
references are MoE and have more state semantics to bring up. This is a
selection for implementation/debugging coverage, not a quality ranking.
The FP16 control excludes quantization error from its import/paging
comparison. D-052's EXL3 companion uses its own identical-quant reference;
both representations inform the early artifact and operation contracts.

## Tokenizer, template and context identity

The **selected GGUF is authoritative**, including its embedded Qwen2 BPE
vocabulary, pre-tokenizer identifier, merges, special IDs and chat template.
It has 151,936 vocabulary entries and 151,387 merges; BOS/pad ID 151643,
EOS ID 151645, and automatic BOS insertion disabled. The chat template's
UTF-8 SHA-256 is
`d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4`.
Render/tokenize using that identity, preserving tool/message branches as
they are enabled by the M5 client contract. Arbitrary checkpoint code or
templates cannot execute unchecked in the native runtime.

The GGUF declares an **8,192-token context**, whereas the pinned base config
declares 32,768. Its template also differs byte-for-byte from the base
tokenizer config. Do not silently substitute the base template, advertise
32K for this artifact, or apply RoPE overrides. The initial reference smoke
uses a 512-token context and just 76 actual tokens; it does not validate the
8K ceiling. M2/M5 choose and validate finite request/context profiles within
that ceiling under D-050. Context increases need new numerical and envelope
evidence.

The published GGUF's exact converter invocation and source-weight revision
are not attested by this study. Base metadata was cross-checked, but its
safetensors weights were not downloaded or compared. Accordingly, the
numerical oracle is this exact GGUF, not an asserted equivalence to the base
BF16 checkpoint. Reproducing conversion or accepting another quantization
requires its own pins and comparisons. No converter or checkpoint code was
executed for this selection.

## Numerical comparison contract

The **primary reference is llama.cpp CUDA on Spark** using this exact GGUF;
its CPU path is a diagnostic reference. The initial profile uses F16 K/V,
one sequence, no Flash Attention or CUDA graphs, no speculative decoding,
and no RoPE override. CUDA operation fusion stays **enabled** (the upstream
default): unlike graphs, it changes these logits, so it is part of the
numerical plan, and the fusion-disabled arm is recorded as the matching
reference for a native plan that does not fuse. Explicit context, batch,
thread and environment values are in the [experiment](experiments/first-slice/README.md).
[RE-008](rough-edges.md#re-008-extended-reference-runs-do-not-always-preserve-exact-top-1-predictions--2026-09-21-status-open)
records non-repeatable predictions for larger MoE models at this llama.cpp
pin; this bounded dense run repeats exactly, which does not generalize. The external
reference retains its pinned upstream build; llmpalooza retains the D-032
Clang/CUDA toolchain. A rebuild or changed numerical plan needs a new
reference record.

Use fixed teacher-forced token IDs and positions, identical prefill chunks
and decode steps, and **raw logits before sampling or penalties**. M2 can
consume fixed IDs without implementing a tokenizer. M5 separately proves
this fixture's rendered bytes, token IDs, special-token handling, stop
rules and seeded sampling, including tool round trips, on M3's native
tokenizer. Matching a short generated string or
only top-1 predictions is insufficient to validate inference.

M2/M5 comparisons must cover:

- Exact tensor values, shapes, types and alias identity after import and
  restoration; exact rendered bytes/token IDs at the tokenizer boundary.
- Full-logit absolute/RMS errors, nonfinite values, per-position top-1
  agreement and top-1/top-2 margins on fixed inputs. Compare intermediate
  normalization, Q/K/V, attention and FFN outputs where a discrepancy first
  appears; any tracing callback needs its own uninstrumented control (RE-006).
- A resident llmpalooza control and the same implementation after weight/state
  eviction and restoration, using the same computation order and backend.
  Exact storage recovery is mandatory. Expect bit-identical logits when
  the numerical path is unchanged; any nondeterminism needs separately
  measured controls and predeclared bounds, never a relaxed threshold
  introduced after seeing a failure.
- Separate reference reruns and CPU/CUDA diagnostics before selecting any
  cross-implementation floating-point tolerance. Pin acceptance thresholds
  and held-out inputs before accepting the native backend. Observed
  CPU/CUDA differences below do **not** supply such a threshold. A tie or
  small margin explains a differing argmax only after logit errors meet
  the agreed bound; it does not excuse an incorrect computation.

The bounded reference check passed on `spark-c4e2` on 2026-09-22: both CPU
and CUDA repeated all 11,547,136 logits exactly, and restored a fresh context
after 32 tokens with exact continuation logits. CPU versus CUDA had all
76 top-1 IDs equal but maximum absolute logit difference 0.2375702858 and
RMS difference 0.0152053890. This illustrates why backend-specific controls
matter. It is one short synthetic trajectory, not acceptance evidence for
llmpalooza, long contexts, a spill format or pending-work cancellation.

## Selected reuse and outstanding gates

The [source inventory](experiments/first-slice/source-audit.json) records
paths, roles, notices and verified hashes at the pinned llama.cpp revision.
The [licensing index](licensing.md#first-dense-slice-d-051) records the
disposition. Nothing from upstream is copied or linked into llmpalooza by this
planning change; the experiment links an external reference tool only.

For the native proof, use GGML core and its CPU/CUDA operations behind
llmpalooza's operation contract. Adapt the selected Qwen2 graph/tensor semantics;
do not adopt libllama's scheduler, weight loader, KV allocator or residency
ownership. Backend workspace, hidden allocations, registration and captured
pointers remain the M2 proof's responsibility. Sources follow D-057's
[source-dependency mechanism](source-dependencies.md). The backend proof's P0
recorded the bridge's executed CUDA closure
([`fp16-plan.json`](experiments/backend-proof-p0/fp16-plan.json)). Admitting
the selected GGML subset through D-057 is still M2 work.

Native tokenizer incorporation had an explicit blocker: llama.cpp's Unicode
tables are generated from inputs whose exact revisions are not recorded in
the generated file. The generator reads a moving Unicode URL and Python's
Unicode data; root MIT alone does not resolve the derived-data terms.
The provenance is now established (2026-09-28): the tables are UCD 15.1.0
data ([licensing.md](licensing.md#tokenizer-unicode-tables-m3)). Llmpalooza's
tokenizer does not copy them; it generates its own from the pinned UCD
15.1.0 files, and D-088 admits Unicode-licensed data to the core. The
owner accepted D-088 on 2026-09-28, closing the gate: the native tokenizer
([tokenizer.md](tokenizer.md)) may link into shipped binaries. Owned native
renderers pass exact fixtures without the upstream Jinja/parser/vendor
closure; this fixture's Qwen2.5 template (`d5495a1e…`) has none yet (M5).

The early integration proof covers this control and the D-052 EXL3 companion:
prepared artifacts, llmpalooza-owned weight/state/workspace backing, chunk-closure
direct reads, completion-safe cancellation/reclaim and correctness after
restoration. The immutable artifact is D-056's experimental
[v0 format](artifact-format.md); the mutable spill format remains its own
decision. Selecting a GGUF **input** settles neither.
