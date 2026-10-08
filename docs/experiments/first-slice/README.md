<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# First dense reference: Qwen2.5-0.5B-Instruct FP16

This bounded M0 check supports [D-051's selection](../../first-slice.md).
It inspects a hash-verified official GGUF and tests the pinned external
llama.cpp reference. It does not implement a llmpalooza model adapter, importer,
pager or spill format. `pins.json` records identities/profile;
`results.json` contains aggregate observations, not raw logits or traces.

## Evidence

Verified 2026-09-22 on `spark-c4e2`, GB10, driver 580.178.04, kernel
7.0.0-1019-nvidia. The existing ARM64 reference image is
`ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`,
source `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`. The harness built with
the image's default `g++`, GNU 13.3.0 (the upstream libraries were built with
GNU 14.2.0), C++23, explicit `-march=armv8-a`, `-O2` and warnings as errors. This uses the existing external reference's ABI/build; it does
not change llmpalooza's D-032 compiler pins.

The synthetic input is fixed in `reference.cc`: plain text, numeric/code
fragments, accented Latin and Chinese text inside an explicitly rendered
chat. It tokenizes to 76 IDs. The first 32 are processed together, then the
remaining 44 one at a time, requesting every vocabulary logit at every
position. No model-generated tokens, sampling or evaluation dataset is used.
The harness uses a 512-token context, batch/microbatch 64, one sequence,
8 CPU threads, F16 K/V and Flash Attention disabled. CUDA offloads all
25/25 layers, disables PTX JIT and CUDA graphs, and keeps CUDA operation
fusion at the upstream default, **enabled** (`GGML_CUDA_DISABLE_FUSION`
unset). CPU runs without GPU device access and disables KQV/operation offload.

| Check | Observed result |
| --- | --- |
| Downloaded GGUF | 1,266,425,696 bytes; publisher SHA-256 matched |
| Layout | 291 tensors: 170 F16, 121 F32; 1,260,477,952 tensor payload bytes; all finite |
| Tied embedding/output | Separate stored tensors, byte-identical; each 272,269,312 bytes |
| Unique elements with verified tied matrix counted once | 494,032,768; the file stores 630,167,424 |
| Base config cross-check | Hash-verified base `config.json`: layers 24, hidden 896, intermediate 4,864, 14 query/2 KV heads, RoPE base 1,000,000, RMS epsilon (float32) and vocabulary 151,936 all match the GGUF |
| Context/template metadata | GGUF context 8,192 versus base config 32,768; embedded template differs byte-for-byte from base tokenizer config |
| CPU fresh-context repeat | All 11,547,136 logit values bit-identical |
| CUDA fresh-context repeat, graphs disabled | All 11,547,136 logit values bit-identical |
| CPU and CUDA context restore, compared within each backend | After 32 tokens, serialize 394,201 bytes, free context, create new context and restore; all 44 subsequent logit rows bit-identical to resident control |
| CPU versus CUDA, graphs disabled | Top-1 equal at 76/76 positions; maximum absolute logit difference 0.23757028579711914; RMS 0.015205389032228013 |
| CUDA fusion control (same binary) | Fusion on reproduces the recorded CUDA logits bit-for-bit with graphs on or off. Fusion off also repeats/restores exactly and is graph-invariant, but differs from fusion on: top-1 76/76, maximum absolute 0.047740936279296875, RMS 0.003304913375054328 |

One exploratory CUDA run with the image's graph-enabled default also repeated
and restored exactly. The selected profile disables graphs to keep the
initial backend proof's numerical path explicit; graphs do not change these
logits. Fusion does: it selects different fused CUDA kernels, so it is part
of the numerical plan. The original CUDA run had fusion enabled implicitly;
the profile now records it, and `results.json` keeps both arms' logit hashes
so a native plan that does not fuse is compared with the matching arm. This
is not a graph, fusion or performance comparison; no timings or peak
admission budgets are claimed. Earlier paging-study captures disabled fusion
for a different reason: a route-reading callback changed the fused MoE path
([RE-006](../../rough-edges.md#re-006-reading-moe-routes-through-the-llamacpp-callback-changes-the-cuda-path--2026-09-21-status-worked-around));
this dense model reads no routes.

The restored state never leaves host memory, the model weights remain loaded,
and the source context is synchronized before destruction. Thus this proves
only the reference's short same-process context round trip, not cross-process
state compatibility, native host-VMM execution, weight eviction/restoration,
pending-work cancellation, long-context correctness or llmpalooza's state format.
The observed CPU/CUDA error is not an acceptance tolerance. The fixed input,
token-ID and full-logit hashes allow reproduction without a retained raw bundle.

## Provenance and use boundaries

Qwen's GGUF and pinned base repositories both provide matching Apache-2.0
license files. The base files supply a config/tokenizer/license cross-check;
base safetensors were not downloaded and their equivalence to the published
GGUF was not established. The GGUF, including its own template/context
metadata, is the numerical input. No conversion or checkpoint code ran.

`source-audit.json` is a bounded selection inventory, not clearance of an
entire compiled dependency closure. llama.cpp/GGML are external MIT reference
tools here; NumPy 2.2.6 and installed GGUF 0.19.0 are inspection/comparison
tools inside the pinned image. CUDA, the driver and the container's standard
runtime libraries retain the separately recorded terms in the
[reference setup](../reference-setup/README.md). The full image contains
additional unused packages; its digest is not a blanket permissive-license
claim or redistribution approval. All new harness code is llmpalooza-authored
Apache-2.0; no upstream implementation was incorporated into the application.

The future tokenizer's generated Unicode data has an explicit provenance and
D-017 license gate. M2 uses fixed token IDs. Future graph/tokenizer/renderer
reuse and notice requirements are recorded in the
[selection contract](../../first-slice.md#selected-reuse-and-outstanding-gates).

## Reproduction on a Spark

Use a new output directory, the pinned source tree from the reference setup,
and a copy of this harness directory. Paths below are examples external to
the checkout; no host setting is changed. Use the existing Docker access
method (the owner's Spark uses `sudo -n docker`). Download/verify the model
and source before any container executes them. The image may be pulled by
digest if absent. No Hugging Face credentials are required.

```bash
set -euo pipefail
FIRST_MODEL=/home/pmeenan/.local/share/llmp/reference-models/qwen2.5-0.5b-instruct-fp16.gguf
FIRST_SOURCE=/home/pmeenan/.local/share/llmp/paging/llama.cpp-b29c606e28a01b1bc8c1351026a0fa6e616bf6c4
FIRST_HARNESS=/absolute/path/to/docs/experiments/first-slice
FIRST_RUN=/tmp/llmp-first-slice-run
FIRST_IMAGE=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
FIRST_DOCKER=(sudo -n docker)
mkdir -m 700 "$FIRST_RUN"
curl --fail --location --output "$FIRST_MODEL.part" \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-fp16.gguf
```

Verify before publishing the download or compiling against reference headers:

```bash
python3 - "$FIRST_HARNESS/pins.json" "$FIRST_MODEL.part" "$FIRST_SOURCE" "$FIRST_MODEL" <<'PY'
import hashlib, json, sys
from pathlib import Path
pins = json.loads(Path(sys.argv[1]).read_text())
model = Path(sys.argv[2])
expected = pins['downloaded_files'][0]
def check_model(path):
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if digest != expected['sha256'] or path.stat().st_size != expected['size_bytes']:
        raise ValueError('Model identity mismatch')
check_model(model)
if Path(sys.argv[4]).exists():
    check_model(Path(sys.argv[4]))
for name, digest in pins['reference']['headers'].items():
    if hashlib.sha256((Path(sys.argv[3]) / name).read_bytes()).hexdigest() != digest:
        raise ValueError('Header identity mismatch: ' + name)
PY
mv -n "$FIRST_MODEL.part" "$FIRST_MODEL"
FIRST_MOUNTS=(--mount "type=bind,src=$FIRST_HARNESS,dst=/harness,readonly"
              --mount "type=bind,src=$FIRST_SOURCE,dst=/source,readonly"
              --mount "type=bind,src=$FIRST_MODEL,dst=/model.gguf,readonly"
              --mount "type=bind,src=$FIRST_RUN,dst=/output")
FIRST_COMMON=(run --rm --network none --read-only --user "$(id -u):$(id -g)"
              --tmpfs /tmp:rw,size=1g "${FIRST_MOUNTS[@]}")
"${FIRST_DOCKER[@]}" "${FIRST_COMMON[@]}" --entrypoint g++ "$FIRST_IMAGE" \
  -std=c++23 -O2 -march=armv8-a -Wall -Wextra -Wpedantic -Werror \
  -I/source/include -I/source/ggml/include /harness/reference.cc \
  -L/app -Wl,-rpath,/app -lllama -lggml -lggml-base -o /output/reference
"${FIRST_DOCKER[@]}" "${FIRST_COMMON[@]}" --entrypoint /output/reference "$FIRST_IMAGE" \
  /model.gguf /output/cpu cpu > "$FIRST_RUN/cpu.log" 2>&1
"${FIRST_DOCKER[@]}" "${FIRST_COMMON[@]}" --device nvidia.com/gpu=all \
  --env CUDA_DISABLE_PTX_JIT=1 --env GGML_CUDA_DISABLE_GRAPHS=1 \
  --entrypoint /output/reference "$FIRST_IMAGE" \
  /model.gguf /output/cuda cuda > "$FIRST_RUN/cuda.log" 2>&1
"${FIRST_DOCKER[@]}" "${FIRST_COMMON[@]}" --entrypoint python3 "$FIRST_IMAGE" \
  /harness/compare.py /output/cpu /output/cuda > "$FIRST_RUN/comparison.json"
"${FIRST_DOCKER[@]}" "${FIRST_COMMON[@]}" --env PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 "$FIRST_IMAGE" /harness/test_compare.py
```

Both executable invocations must exit zero and report zero repeat/restore
bit differences. Verify the CUDA log reports all 25/25 layers offloaded;
the CPU invocation must have no device access. `compare.py` rejects failed
repeat/restore records, mismatched token streams, nonfinite or truncated
logits, and reports cross-backend differences without declaring acceptance.
Five checker tests exercise identity, changed predictions, mismatched inputs,
truncation, nonfinite data and a failed restore; three inspector tests cover
the base-config cross-check and pinned base-file identities.
They passed in the Spark reference image and on the x86-64 workstation using
an externally unpacked, hash-verified NumPy 2.2.6 wheel recorded in `pins.json`.
With that version available, run `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest
discover -s docs/experiments/first-slice -p 'test_*.py' -v` from the repo root.

For the metadata inspection, fetch `tokenizer_config.json` and `config.json`
from the pinned base revision using their `url` entries in `pins.json`, place
them in `FIRST_RUN`, then run through the same Python container invocation:

```bash
"${FIRST_DOCKER[@]}" "${FIRST_COMMON[@]}" --env PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 "$FIRST_IMAGE" /harness/inspect_model.py /model.gguf \
  /output/tokenizer_config.json /output/config.json > "$FIRST_RUN/inspection.json"
```

The inspector verifies both base files' pinned sizes and hashes and the
GGUF's full hash before parsing, then reports the base-config cross-check;
mismatches are reported, never repaired. It is scoped to this known artifact
and is not llmpalooza's future adversarial import validator.

The fusion control reruns the same binary with fusion disabled, keeping the
other CUDA settings, and compares it with the selected CUDA run:

```bash
"${FIRST_DOCKER[@]}" "${FIRST_COMMON[@]}" --device nvidia.com/gpu=all \
  --env CUDA_DISABLE_PTX_JIT=1 --env GGML_CUDA_DISABLE_GRAPHS=1 \
  --env GGML_CUDA_DISABLE_FUSION=1 --entrypoint /output/reference "$FIRST_IMAGE" \
  /model.gguf /output/cuda-fusion-off cuda > "$FIRST_RUN/cuda-fusion-off.log" 2>&1
"${FIRST_DOCKER[@]}" "${FIRST_COMMON[@]}" --entrypoint python3 "$FIRST_IMAGE" \
  /harness/compare.py /output/cuda /output/cuda-fusion-off > "$FIRST_RUN/fusion-comparison.json"
```

The recorded run kept weights, metadata and raw results under
`/home/pmeenan/.local/share/llmp/first-slice-20260922` on `spark`; the
fusion control and re-inspection are under `first-slice-fusion-20260922`.
No persistent containers or servers were left running. Keep subsequent raw
logits, token streams, logs and downloaded weights outside the repository.
