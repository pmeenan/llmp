<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Small EXL3 reference on Spark

The two D-052 Qwen2.5-0.5B-Instruct EXL3 quants execute on Spark, with
exact repeated logits and exact continuation after synchronized in-place
cache restoration. The baseline includes 4.0 bpw and mixed-rate 4.5 bpw,
optimized Model API execution, an attention-graph control, and 176
real/synthetic projection cases. This is external reference evidence for the
[early EXL3 contract](../../exl3-bringup.md); native llmpalooza execution, paging
and performance parity remain M2–M6 work.

[Aggregates](aggregates.json) contain statistics, numerical controls, memory
observations, selected kernel symbols and raw-result receipts.
[Prior-run controls](prior-run-controls.json) and the
[kernel repeat control](kernel-repeat-control.json) retain earlier complete
runs excluded from the aggregates.
[Artifact/build pins](pins.json), [runtime inventory](runtime.json),
[model protocol](model-protocol.json) and [kernel protocol](protocol.json)
make the comparison reproducible. Raw logits, timing arrays, profiler traces,
logs, weights and build caches remain outside Git.

## Conditions and provenance

Measured 2026-09-22 on `spark-c4e2`, NVIDIA GB10, compute 12.1, driver
580.178.04. The container uses PyTorch 2.14.0+cu130, Triton 3.8.0, NumPy 2.2.6,
Transformers 5.17.0 and tokenizers 0.23.2. It is built by the checked-in
[Dockerfile](Dockerfile), reusing the pinned reference stack and adding
three hash-pinned wheels: ninja 1.13.2, marisa-trie 1.4.1 and llguidance 1.8.0.
Package versions/notices and loaded shared-library hashes are in the runtime
inventory. The base image digest and resulting local image identity are in
`pins.json`; the local image is not a published artifact.

ExLlamaV3 source is pinned to
[`6b84a21b6f1e5da3f291b9e1019061f0de788279`](https://github.com/turboderp-org/exllamav3/tree/6b84a21b6f1e5da3f291b9e1019061f0de788279).
Its extension compiles 181 translation units with NVCC 13.0.88 and GNU
13.3.0, C++20, explicit host `-march=armv8-a` and device `sm_121`.
Upstream retains `-Ofast`/CUDA `-O3 --use_fast_math`. Driver PTX JIT is disabled;
Triton's own compilation remains enabled. This external build does not change
llmpalooza's native C++23/Clang toolchain pins.

Unmodified upstream fails to compile its unconditional x86 host helpers on
ARM ([RE-009](../../rough-edges.md#re-009-pinned-exllamav3-compiles-x86-only-cpu-helpers-on-spark--2026-09-22-status-worked-around)).
The [patch](arm-reference.patch) returns false from x86 capability probes,
rejects CPU MoE/reduction entry points explicitly, and uses ARM host spin
waits. **CUDA device kernels are unchanged.** Negative runtime controls
verify the capability flags and CPU MoE rejection. CPU offload, CPU
collectives and multi-GPU execution are outside this build's evidence.
The patch retains [Turboderp's MIT notice](UPSTREAM-NOTICE.txt).

Both full checkpoint SHA-256 values and every supplied metadata/tokenizer
file match the immutable pins. Their repositories supply Apache-2.0 licenses
matching the base Qwen model. Exact conversion/calibration lineage remains
unknown. No checkpoint code executes, no weights are redistributed and no
MiaAI patches or calibration corpus are used. External tooling/platform terms
remain separate from native implementation under D-017; package notice
inventory is not permission to redistribute the whole container or clearance
of the future native compiled closure.

The two fixtures have identical tokenizer/template identities. Transformers
and ExLlamaV3 produce identical IDs for the rendered template probe. The
checkpoint declares 32,768 context positions; this study uses a 4,096-token
F16 KV allocation and validates only the bounded trajectories below.

## Numerical controls

One sequence uses deterministic synthetic token IDs
`1000 + ((37*i + 20260922) mod 29000)`, with
prefixes 32, 144, 145, 1,023 and 1,024 and a 16-token teacher-forced suffix.
Every vocabulary logit is checked at every suffix position and at every
prefill position for prefixes through 145. Larger prefill runs check suffix
logits only. Each profile repeats each trajectory three times. All values
must be finite and byte-identical within the same numerical plan; top-1
agreement alone cannot pass.

At each prefix, synchronize and copy all 48 F16 cache tensors to CPU
(50,331,648 bytes), hash them, poison live tensors with NaNs, verify the
poison, restore in place and verify every byte before resuming. All four
model profiles pass every repeat and restored-suffix comparison. Weights
stay resident and addresses stay fixed, including captured pointers. This
proves a completed-boundary, same-process cache round trip, not serialized
state compatibility, relocation, cancellation safety or native spill.

All 176 projection cases repeat exactly and their captured outputs exactly
match warmed eager execution. Packed-versus-reconstructed outputs and
optimized-versus-attention-eager logits are diagnostic comparisons in the
aggregates. The two attention profiles happen to match exactly on these
trajectories; packed/reconstructed linear results can differ. Distinct plans
need not be byte-identical. At 145 rows and above the eager operation
already is upstream reconstruction, so `versus_reconstruction` compares
that plan with itself and is trivially exact; fused and unfused
reconstruction are never compared on the same input. Their observed errors
are **not native acceptance tolerances**. Exact storage and same-plan restore
remain mandatory. Independent controls and a held-out trajectory must set
cross-implementation bounds before evaluating native results. The harness's
offset formula is not a held-out input: changing 20260922 to 20260923 moves
every ID by one on the same stride-37 progression. The declared held-out
trajectory instead draws 1,040 regular-vocabulary IDs (0–151,642, excluding
special tokens) with NumPy 2.2.6
`default_rng(20260923).integers(0, 151643, size=1040, dtype=int64)`,
SHA-256 `6dd8da897821f5615e7796f4d882795faf306a941f050e2eb68b619096e3fb0c`
over little-endian bytes, at the same prefixes and suffix length. It has not
been executed in this study.

## Resident performance

The optimized Model API profile keeps upstream attention/MLP optimizations.
Both profiles enable pinned CPU embedding staging for single-token decode,
as upstream Generator does; prefill retains its unpinned default. Each decode
call synchronizes before that staging buffer can be reused. Both profiles
use the bundled Triton attention path; `attention_eager` sets only `EXL3_BC_ATTN=0`.
Gated-MLP graphs remain enabled in both. Each profile runs in a fresh process,
with four CPU threads, one interop thread and no other inference workload.
Untimed boundary observations span 41–66 °C and 2,398–2,509 MHz SM clocks;
they are not continuous in-kernel telemetry. The host clock policy is unchanged. The container has a 32 GiB host-memory
limit and 2 GiB shared-memory allowance; that cgroup limit is not a complete
CUDA/unified-memory bound.

There are two blocks of 31 trials per shape, after 15 prefill warmups, at
32, 144, 145, 512, 1,023, 1,024 and 2,048 prefix tokens. Decode runs 64 tokens
after a 512-token prefix: 62 requests and 3,968 token observations per
profile. Inputs are fixed, teacher-forced IDs; the per-token timed path
includes making CPU argmax output available. These are direct Model API resident compute
measurements, not quality evaluation or free-running sampled conversations.
Generator additionally stages token IDs and cache-position/block metadata
in reusable pinned buffers and owns scheduling/sampling. Our `batch_shape`
route uses its default metadata construction instead. Kernel cache-table
capacity matches Generator for this decode range, but host preparation and
metadata uploads differ. A normal Generator serving control remains required
before M5 claims end-to-end upstream parity.

Prefill excludes the output head. The TTFT proxy adds separately timed
prefix prefill and a final prompt-token forward including CPU token
availability. Its prompt has prefix length plus one; it includes an extra
synchronization boundary and excludes tokenizer, request queue and network.
Model CUDA-event elapsed intervals can include CPU submission gaps and are
not isolated active-kernel time. First observed shape use is reported
separately; preceding shapes may already have warmed shared kernels.

| Quant / profile | Decode tokens/s | Token p50 / p95 / p99 (ms) | 2,048-prefix tokens/s | 2,048-prefix TTFT proxy (ms) |
| --- | ---: | ---: | ---: | ---: |
| 4.0 bpw / optimized | 291.33 | 3.410 / 3.690 / 4.054 | 41,097 | 53.50 |
| 4.0 bpw / attention_eager | 281.62 | 3.516 / 3.845 / 4.241 | 40,846 | 54.10 |
| 4.5 bpw / optimized | 277.23 | 3.581 / 3.744 / 4.235 | 41,366 | 53.37 |
| 4.5 bpw / attention_eager | 269.26 | 3.681 / 3.928 / 4.421 | 41,998 | 52.79 |

Medians combine both blocks. Token p95/p99 are descriptive; uncertainty on
token medians resamples whole requests, preserving within-request dependence.
The load measurement follows full-file hashing, so the page cache is warm.
It excludes source compilation, payload verification,
tokenization and process startup; it is not cold disk-load or reload latency.
The JSON includes all shapes, median bootstrap 95% intervals and per-case
noise allowances. No samples are discarded.

## Packed-kernel and reconstruction boundaries

Each quant supplies first-layer Q, K, MLP up/down and the quantized output
head at 1, 8, 16, 32, 144, 145, 1,023 and 1,024 rows: 80 real cases. Another
96 cases use `(k,n)` of `(4096,4096)`, `(4096,14336)` and `(14336,4096)` at
K=4/5/6/8 and the same row sweep. Synthetic trellises contain seeded random
int16 words, full-length signed side vectors and the `mcg` codebook. They
are valid decoder/stress inputs, not quantized trained projections or quality
evidence. The real head's width 151,936 also exercises reconstruction slicing
above upstream's 32,768-column limit.

After five warm calls, one graph captures timing events around ten sequential
linear invocations. Each of 31 samples replays this graph once; reported
microseconds divide its device interval by ten. Events inside the graph
remove Python dispatch gaps between invocations. The summarizer derives
each case's plan and trellis rate from profiled kernel symbols and rejects
disagreement with the row-count plan; all 176 agree (110 packed,
44 reconstruct, 22 fused). Profiler durations are instrumented diagnostics, not
the parity metric. Its raw event totals cover ten invocations and the
`active_device_us` diagnostic is normalized per invocation.

| Projection | K | 1 row (µs) | 144 rows (µs) | 145 rows (µs) | 1,023 rows (µs) | 1,024 rows (µs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.0 bpw real output head, 896×151,936 | 8 | 590.66 | 5,576.31 | 3,599.07 | 8,328.18 | 5,309.38 |
| Synthetic 4,096×14,336 | 4 | 114.08 | 1,053.84 | 1,291.02 | 2,481.68 | 2,190.85 |

Upstream dispatches packed computation through 144 rows, reconstruction plus
GEMM from 145, and fused Hadamard/reconstruction from 1,024. These thresholds
produce substantial shape-dependent changes; they are not established Spark
optima. A permanently expanded FP16 shadow would not satisfy the EXL3 gate.
Bounded transient reconstruction is a legitimate plan when its full peak and
retirement lifetime are admitted.

## Memory and acceptance limits

| Quant / profile | GPU weight arena payload (bytes) | GPU weights incl. arena capacity (bytes) | Timed-phase peak Torch allocated / reserved (bytes) |
| --- | ---: | ---: | ---: |
| 4.0 bpw / optimized | 46,246,144 | 404,574,336 | 625,188,864 / 645,922,816 |
| 4.0 bpw / attention_eager | 46,246,144 | 404,574,336 | 624,184,320 / 643,825,664 |
| 4.5 bpw / optimized | 68,553,728 | 404,574,336 | 625,188,864 / 645,922,816 |
| 4.5 bpw / attention_eager | 68,553,728 | 404,574,336 | 624,184,320 / 643,825,664 |

The BF16 CPU embedding alone occupies 272,269,312 bytes. Although config says
`tie_word_embeddings`, the stored quantized output head
is a different representation; it cannot be deduplicated with the embedding
on that flag alone. The two rates have different arena payloads but the same
128 MiB rounded arena capacity. The F16 KV allocation is 50,331,648 bytes;
optimized attention statics add 1,003,522 bytes.

PyTorch allocated/reserved peaks are reset after each shape's warm-up, so
they cover the timed trials only: first-use tuning and warm-up transients
are excluded from the allocated peak, though the caching allocator's
reserved peak retains blocks they left behind. They are measured before
numerical probes copy full logits or cache snapshots to CPU. Kernel peaks record the captured batch
and can be compared to its pre-capture allocation; they do not automatically
isolate every reusable workspace. Upstream's direct CUDA device context also
allocates a 16 MiB workspace and 4,202,760 lock bytes outside Torch's allocator,
with additional graph/library/driver resources. Host RSS is separate evidence; the aggregate end-of-run maximum also includes
numerical snapshots and logit copies, while the boundary RSS ranges come
from the performance phase.

On this unified-memory Spark, upstream's `non_torch` subtraction reports
roughly 100 GB while the node has abundant available memory and a large file
cache. It includes node-wide use; it is not this process's non-Torch overhead.
Do not sum CUDA usage, RSS and page cache as independent physical budgets.
Complete native allocation envelopes, extent padding, graph resources and
retained backing remain mandatory M2 evidence; these tracked reference peaks
are not a claimed total process physical-memory bound.

The predeclared comparison rule sets each workload's allowance to the larger
of relative reference block-median drift and the sum of median-confidence
half-widths divided by their mean median. Kernel samples use first 15/last 16
subblocks. An allowance above 10% is **unstable and cannot pass native
acceptance**. Reference must be repeated beside the candidate, with no
arbitrary percentage floor and no averaging away individual regressions.

All model timing cases and 176/176 kernel cases satisfy this check in the
final run. Earlier complete runs did not, and remain on record:

- The first complete model set used 9 trials per block. **All four profiles
  exceeded 10% at 145-token prefill** (10.40–15.04%), the packed-to-
  reconstruction dispatch boundary; numerical controls passed. The model
  protocol was then re-declared with 31 trials and 15 warm-ups. In the
  accepted run 145-token prefill is still the largest allowance in every
  profile (up to 9.42% for 4.5 bpw / attention_eager), and it was the largest
  in 7 of 8 exploratory 31-trial profiles
  ([prior-run controls](prior-run-controls.json)).
- An [earlier kernel control](kernel-repeat-control.json) found a
  **10.56%** allowance for synthetic 4,096×4,096 K=5 at 1,024 rows, with
  receipt hashes and exact prior runner/protocol reconstruction retained.
- The earliest kernel runs exceeded 10% for 4.5 bpw down_proj at 16 rows and
  synthetic 4,096×4,096 K=8 at 1 row. Their records lack the ten-invocation
  graph field, so that timing method differs and is not directly comparable.

145-token prefill (all profiles) and synthetic 4,096×4,096 K=5 at 1,024 rows
remain flagged: native acceptance needs a stable reference repeated beside
the candidate under the frozen rule. A quieter repeat does not erase earlier
variability. No samples or failing numerical cases were dropped.

This baseline establishes executable packing and upstream measurements early.
Native M2 still must prove prepared-artifact execution on llmpalooza-owned VMM,
complete dependency closures, bounds, relocation/cancellation/restore and
per-kernel parity. M5 adds actual request TTFT and resident serving parity;
M6 adds switching under pressure. Dense K=4/5/6/8 `mcg` success does not
validate other codebooks, fractional rates, MoE, batching or sharding.

## Reproduction

Use an external directory on a Spark. Clone the pinned upstream revision into
`$EXL_RUN/source`, apply `arm-reference.patch` with `git apply --check` first,
and copy this harness directory to `$EXL_RUN/experiment`. Copy `cxx-target`
to `$EXL_RUN/cxx-target` and make it executable. `fetch.py --output
"$EXL_RUN/models"` downloads only the pinned files and verifies full hashes.
It refuses mismatched existing files, symlink targets, truncation, extra bytes
and replacement of a destination created concurrently.

Build from `docs/experiments` with `docker build -f exl3-reference/Dockerfile
-t llmp-exl3-reference:20260922 .`. The measured container mounts
`$EXL_RUN:/experiment` and `/usr/local/cuda-13.0:/usr/local/cuda-13.0:ro`,
uses `--gpus all --shm-size 2g --memory 32g --memory-swap 32g`, and sets:

```text
CUDA_HOME=/usr/local/cuda-13.0
TORCH_CUDA_ARCH_LIST=12.1
MAX_JOBS=8
TORCH_EXTENSIONS_DIR=/experiment/build-cache
PYTHONPATH=/experiment/source
CXX=/experiment/cxx-target
CUDAHOSTCXX=/experiment/cxx-target
CUDA_DISABLE_PTX_JIT=1
OMP_NUM_THREADS=4
```

Keep these settings when importing ExLlamaV3; its loader compiles/caches the
extension. The first compile and all tuning are outside the accepted timing
samples. The reference runs from source, not an unpinned installed ExLlama
package. Run each profile serially, with a new output directory (the harness
refuses to overwrite one). Inside that container:

```bash
set -euo pipefail
for fixture in 4.0bpw 4.5bpw; do
  for profile in optimized attention_eager; do
    python3 /experiment/experiment/measure.py \
      --model /experiment/models/$fixture --profile "$profile" --mode model \
      --protocol /experiment/experiment/model-protocol.json \
      --output /experiment/results/accepted-$fixture-$profile
  done
  python3 /experiment/experiment/measure.py \
    --model /experiment/models/$fixture --mode kernels \
    --output /experiment/results/final-kernels-$fixture
done
python3 /experiment/experiment/measure.py \
  --model /experiment/models/4.0bpw --mode synthetic \
  --output /experiment/results/final-kernels-synthetic
python3 /experiment/experiment/inspect_runtime.py \
  --model-root /experiment/models --source /experiment/source \
  --output /experiment/results/runtime.json
python3 /experiment/experiment/summarize.py \
  --results /experiment/results --output /experiment/results/aggregates.json
python3 -m unittest discover -s /experiment/experiment -p 'test_*.py' -v
```

Run the inventory and summary after GPU measurements. The summarizer fails on
missing/duplicate workloads, incorrect identities/shapes/rates, failed detail
checks, inconsistent sample counts or statistics, and absent profiler evidence;
a manifest's success boolean alone cannot pass. The checked-in aggregate
contains receipt hashes of the exact raw bundle. Reproduction uses fresh
results and regenerates receipts; prior raw files are not required.

The measured external directory is
`/home/pmeenan/.local/share/llmp/exl3-reference-20260922` on `spark`.
Only `accepted-*` and `final-kernels-*` enter the report; exploratory runs
remain external. Reuse or retrieve raw files only after matching their listed
receipt hashes. Host prerequisites and runtime/compiler identities must be
checked against `runtime.json`; a different stack creates a new baseline.
Stop/remove the experiment container after use. No driver setting is changed.
