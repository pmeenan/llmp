<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Early EXL3 execution and performance contract

D-052 makes a real EXL3 quant a **required M2 companion** to D-051's
[FP16 GGML control](first-slice.md), before settling the operation contract
and initial executable artifact layout. M4 serves sharded flagship EXL3
models; M5 includes resident EXL3 serving of the fixtures;
M6 includes EXL3 switching and restoration. Initial EXL3 performance work
starts with this proof, rather than waiting for M9 or a flagship model.
This is required implementation scope, not a claim of working support.

## Concrete small checkpoints

Use two published quants of **Qwen2.5-0.5B-Instruct**, keeping the model
architecture shared with the FP16 control. The immutable repositories,
published payload hashes, downloaded metadata hashes and bounded header
inspection are in [pins.json](experiments/exl3-reference/pins.json).

| Fixture | Immutable revision | Published file size | Observed trellis rates |
| --- | --- | --- | --- |
| [4.0 bpw](https://huggingface.co/blockblockblock/Qwen2.5-0.5B-Instruct-exl3-4.0bpw/tree/7009334dc77ad71f6a98d942d6df3ec64017f565) — primary bring-up | `7009334dc77ad71f6a98d942d6df3ec64017f565` | 588,951,098 bytes | 168 matrices at K=4; output head at K=8 |
| [4.5 bpw](https://huggingface.co/blockblockblock/Qwen2.5-0.5B-Instruct-exl3-4.5bpw/tree/030d3a4173c5a3e27e3c27dd7f6ab0fc00636d00) — required mixed-rate coverage | `030d3a4173c5a3e27e3c27dd7f6ab0fc00636d00` | 611,257,914 bytes | 42 matrices at K=4, 118 at K=5, 8 at K=6, output head at K=8 |

Both use the `mcg` codebook and contain 798 stored tensors: 169 trellises,
169 full-length input vectors, 169 full-length output vectors, 72 biases,
169 codebook markers and 50 BF16 tensors. The BF16 embedding alone occupies
272,269,312 bytes. An advertised average bitrate is neither the bitrate of
each matrix nor the complete resident-memory cost. In particular, the
mixed fixture's config says `bits: 4` and `bits_per_weight: 4.5`, while its
quantization sidecar says `bits: 4.5`; actual tensor shapes establish the
per-matrix rates above. There is no K=4.5 matrix in that file.

Each repository supplies an Apache-2.0 license matching the selected base
model's license. These are third-party conversions, not official Qwen EXL3
publications. Their exact source-weight revision and conversion/calibration
lineage are not established. No weights are redistributed. The inspection
verified metadata hashes, safetensors header shapes/dtypes/ranges and their
agreement with the quantization sidecar on 2026-09-22. The subsequent
[small EXL3 Spark baseline](experiments/exl3-reference/README.md) verified
both full payloads, tokenizer/template identities and an executable reference.
Both quants passed repeated-logit and synchronized in-place cache restoration
controls; resident Model API and 176 linear-kernel cases are recorded. Full Generator
serving controls remain required before M5 parity acceptance.

The reference uses ExLlamaV3
[`6b84a21b6f1e5da3f291b9e1019061f0de788279`](https://github.com/turboderp-org/exllamav3/tree/6b84a21b6f1e5da3f291b9e1019061f0de788279)
with a bounded ARM host-helper patch; CUDA device kernels are unchanged.
The patch rejects unsupported CPU MoE/collective helpers. Its external
Python/PyTorch environment, compiler, loaded libraries and notices are
recorded with the baseline. This reference does not establish native serving,
paging, cross-process restore or all EXL3 variants. Native cross-implementation
numerical bounds and complete physical-memory envelopes remain M2 gates;
measured cross-plan differences are not tolerances.

## Representation and native ownership

The source-backed requirements below use that exact upstream revision.
They describe the input and proof obligations, not a frozen artifact ABI.

- Preserve the packed trellis and its executable dependency closure:
  input `su`/`suh`, output `sv`/`svh`, optional bias, codebook identity and
  any derived runtime resources. The loader accepts packed sign vectors
  or full-length FP16 vectors; the executable `suh` and `svh` lengths are
  **k and n**, not k/16 and n/16. Small tensors sharing an extent are still
  distinct logical dependencies. [Loader](https://github.com/turboderp-org/exllamav3/blob/6b84a21b6f1e5da3f291b9e1019061f0de788279/exllamav3/modules/linear.py#L385-L425)
- Record format version, logical dimensions, physical shape/dtype, actual
  per-tensor rate, codebook identifier, side-vector representation and
  checksummed ranges. Validate all products, spans, tile constraints and
  rate/codebook combinations before dispatch. The codebooks are procedural:
  marker presence selects them; do not invent a required resident lookup
  table or infer a codebook from an unused marker scalar value.
  [Markers](https://github.com/turboderp-org/exllamav3/blob/6b84a21b6f1e5da3f291b9e1019061f0de788279/exllamav3/modules/quant/exl3_lib/quantize.py#L1663-L1686)
- Import into D-056's v0 dependency groups (D-035) without changing the
  quantized model. Prepare lossless derived representations at import with
  provenance and hashes; no CPU payload repacking on page-in. A 2 MiB chunk
  boundary is a paging boundary, not permission to execute part of a trellis
  matrix.
  No GGUF conversion, requantization or permanently expanded FP16 shadow
  satisfies this proof. Kernel-readable padding and all side tensors belong
  in the resource index and admission envelope. The v0 descriptors carry
  per-tensor `k_bits`, the codebook and 4-byte marker resources
  ([format](artifact-format.md#indexjson)).
- Port selected device kernels behind a narrow native C++/CUDA boundary;
  llmpalooza owns allocations, explicit streams, contexts, completion and errors.
  Shared Qwen operations may still use GGML. Upstream ATen wrappers, implicit
  PyTorch allocations and Python scheduling do not enter the serving path.
  Catalog the per-device locks and workspace, autotuning allocations,
  activation transforms, outputs, graph objects and any pointer tables.
  Autotuning waits run outside the global scheduling/catalog lock.
  [Device context](https://github.com/turboderp-org/exllamav3/blob/6b84a21b6f1e5da3f291b9e1019061f0de788279/exllamav3/exllamav3_ext/quant/exl3_devctx.cu#L48-L72)
- Large-prefill **bounded transient reconstruction is valid** when it is
  the declared, measured execution plan. Upstream normally uses packed
  compute through 144 rows, reconstruction plus GEMM above that, and may
  fuse Hadamards into reconstruction at 1,024 rows. Reconstructed weight
  scratch alone can require `2*k*min(n,32768)` bytes, in addition to
  activation/output scratch. Charge the full peak and hold its allowance
  through completion-proven retirement, including cancellation unwind under
  D-048/D-050; never borrow an
  unaccounted FP16 cache to pass performance. These upstream thresholds
  are not measured Spark optima. [Dispatch](https://github.com/turboderp-org/exllamav3/blob/6b84a21b6f1e5da3f291b9e1019061f0de788279/exllamav3/modules/quant/exl3.py#L131-L223)
- Captured arguments may contain trellis, side-vector and scratch pointers.
  Validate backing generations even with stable virtual addresses; rebuild
  or patch objects when addresses change. Submitted consumers protect all
  referenced backing until retirement; an idle graph/table with stale
  generations cannot be submitted before revalidation or rebuilding.
  Shared scratch needs exclusive scheduling or separate admitted instances.
  Test implemented capture paths; otherwise
  reject capture explicitly until its relocation proof exists.
  [Captured pointers](https://github.com/turboderp-org/exllamav3/blob/6b84a21b6f1e5da3f291b9e1019061f0de788279/exllamav3/exllamav3_ext/quant/exl3_gemm.cu#L199-L223)

The initial required checkpoint coverage is `mcg` with K=4/5/6/8 and their
actual full-length side vectors. Integer K=1–8 and fractional K=1.5/2.5/3.5
exist upstream; fractional rates require `mul1` at this source pin.
Additional codebooks, fractional rates, legacy packed signs and MoE need
their own valid fixtures and negative tests before being advertised. The
importer must explicitly reject unsupported variants. The schema must
represent their differences without assuming one global bitrate or
codebook. Dense success never means all EXL3 variants work.
[Rate validation](https://github.com/turboderp-org/exllamav3/blob/6b84a21b6f1e5da3f291b9e1019061f0de788279/exllamav3/exllamav3_ext/quant/bits_k.cuh#L4-L19)

## Correctness and paging gates

M2 runs both real fixtures from prepared artifacts on llmpalooza-owned
device VMM (D-081), with full dense prefill/decode execution. A loader,
one matrix multiply, or an external reference process is not the native
proof. The FP16 control remains useful for diagnosing the common model
graph, but the **EXL3 oracle is the identical EXL3 artifact in ExLlamaV3**.
Quantization loss against source weights is a separate quality question;
it cannot be used as tolerance for import, kernel or paging errors.

Before native acceptance, record held-out fixed token IDs/positions/chunks
(the report declares the first held-out trajectory; the backend proof's
[P0](experiments/backend-proof-p0/README.md) executed it),
finite context/output bounds, numerical settings and cross-implementation
tolerances from independent reference controls. Compare all raw logits and
intermediate outputs around the first discrepancy; also compare individual
packed linear outputs against upstream reconstruction with the same
Hadamard/basis convention. Include decode, short batches, medium/long
prefill and either side of dispatch thresholds. Exact storage recovery is
mandatory. Resident versus restored native runs keep the same numerical
plan and must match exactly unless separate controls establish bounded
nondeterminism before testing. Do not infer state compatibility between
the FP16 and EXL3 artifacts or use the GGUF's template/context as EXL3 metadata.

The M2 challenge matrix includes:

- Valid trellis paired with wrong codebook/rate, invalid shapes/offsets,
  truncation, unsupported version/variant and damaged payload checksums.
- Partial eviction of trellis and of side vectors/bias independently,
  shared chunks, padded tails and matrices crossing chunk boundaries;
  never execute a dependency until its entire required closure is resident.
- Restore at completed state boundaries; delayed reads into retired
  generations; relocated backing and stale executable pointers; cancellation
  during I/O and between reconstruction and GEMM; no early reclaim.
- Tight-budget admission that accounts for the largest reconstruction phase,
  failed allocation and complete unwind, plus shared-workspace conflicts.
  Unsupported concurrency must serialize or reject safely, not race.

## Performance gates from the first proof

The target is **match or beat upstream ExLlamaV3 on Spark**, with equivalent
quality and bounded memory, for the declared resident workloads. Establish
it as a gate now: a measured regression blocks the stage until fixed or an
explicit owner-approved tradeoff changes the acceptance contract. Merely
loading EXL3 or eventually optimizing it in M9 does not meet D-052.

The reference task predeclares input/shape sweeps, repetition counts,
warm-up, comparison statistic, uncertainty/noise allowance, numerical
settings and acceptance bounds before native results are evaluated. It
records the Spark, driver/toolkit/build identities, memory budget, clocks
and thermal state. Measure enough repeat controls to support those bounds;
an unstable reference is not a pass. Do not invent tolerances here.

| Stage | Required evidence |
| --- | --- |
| M0/early M1 reference task | Recorded on Spark: both full-hash fixtures, executable environment, tokenizer identities, repeat/restore controls, resident performance and larger synthetic kernel cases ([report](experiments/exl3-reference/README.md)). Unstable timing cases cannot pass native acceptance; repeat reference beside the candidate. 145-token prefill and one synthetic kernel case are flagged from earlier unstable runs |
| M2 native kernels and packing | GPU kernel-time parity and predeclared workspace/memory bounds for decode, small batches, prefill and dispatch boundaries; both fixtures' full-model correctness/restore proof and measured resident timings. Close per-kernel regressions before settling the operation contract |
| M5 resident serving | Full-model prefill latency/throughput, time to first token and decode inter-token p50/p95/p99 meet the declared upstream parity bounds; exact tokenizer/template and numerical profile checks, complete memory accounting |
| M6 switching | EXL3 participates in A→B→A under pressure with retained-state and forced-restore arms; compare resident, partial reload and full-swap controls at the same budget; correct continuations, read/write bytes and stall distributions |
| M4/M7 extensions | Add representative EXL3 routed-expert and sharded cases with their own closure, kernel, quality and performance baselines before claiming those capabilities; the small dense fixture cannot validate them |

For kernel comparisons, exclude Python launch overhead using GPU timings
and verify the actually selected kernel/plan. For end-to-end comparisons,
include native orchestration and separately report load, first-use tuning,
warm compute and reload stalls. Record all peak physical memory and
workspace, including non-evictable allocations, retained backing and any
reconstruction; separately expose chunk/backing padding and native runtime overhead
against upstream instead of requiring identical allocator footprints. Report
both matched settings and upstream's normal optimized
configuration under D-021; do not secure parity by disabling a reference
optimization without showing the resulting gap. Precision changes need
their own quality evidence.

The small model is an early correctness and integration gate, not proof of
top-tier flagship performance. Include representative larger projection
shapes in the M2 kernel sweep with declared synthetic provenance, then real
larger dense/MoE workloads as those models enter support. D-036's existing
measured workload thresholds do not automatically apply to this new model;
set EXL3-specific switching budgets from the new matched controls before
M6 acceptance. Keep these benchmarks as regressions through later changes.

## Adoption boundary

Selected upstream ExLlamaV3 kernels are MIT implementation candidates and
can be a core-eligible build-time CUDA backend after the compiled closure
audit in [licensing.md](licensing.md). The small-m GEMV kernel additionally
cites QTIP's GPL-3.0-licensed `qtip-kernels/src/inference.cu` as the model
for its structure. The owner judged it MIT and core-eligible on
2026-09-27 (D-080); the review scan had found no run of 20 or more tokens
shared with QTIP beyond identical PTX `mma` operand strings. The
format name does not imply an
AGPL module or clear third-party patches and calibration data. Preserve
per-file provenance and notices. The reference includes an attributed MIT-source
patch for external ARM host helpers. The selected kernels entered llmpalooza's
build through the source lock in backend-proof P1 (GEMM) and P3 (GEMV,
reconstruction, Hadamard, bias add), behind llmpalooza's own launchers
(`src/kernels/exl3/`); only tests and benchmarks link them so far.
Native builds retain the CPU-only/fake-backend guardrail and no runtime
plugin ABI. The experimental artifact schema is settled separately
(D-056), informed by both real representations.
