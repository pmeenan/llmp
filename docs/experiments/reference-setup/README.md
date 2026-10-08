<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M0 llama.cpp reference setup — 2026-09-21

This installs an external reference engine on `spark` for the later A→B→A
and expert-route experiments. It does not implement llmpalooza inference,
establish paging feasibility, or measure the full-swap floor.
The subsequent [A→B→A experiment](../reference-aba/README.md) now measures
Gemma/Ornith switching and long-context state reuse; the scope and historical
installation results below remain unchanged.

## Installed identities

`artifacts.json` pins the downloaded container, checkpoint, inspected upstream
files, and observed runtime components. Tags and model `main` branches are
discovery aids only; repeat runs use the digest and immutable model revision.

| Component | Exact selection |
| --- | --- |
| Host | `spark-c4e2`, AArch64 GB10, driver 580.178.04, kernel 7.0.0-1019-nvidia |
| Container | `ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`, `linux/arm64`, 3,051,638,872 image bytes |
| Source identified by image label and executable | `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` |
| Executable version | `0.4.1-dev (build 10964, commit b29c606e2)`, GNU 14.2.0, Linux AArch64 |
| CUDA in container | CUDA 13.3 image; cudart 13.3.29-1, cuBLAS 13.5.1.27-1 |
| Model | `unsloth/gemma-4-26B-A4B-it-GGUF`, revision `c099eb48e663fd284577b04978a94ffccb261841` |
| File | `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`, 16,947,541,728 bytes |
| File SHA-256 | `f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f` |

The discovery tag `full-cuda13-v0.4.1` resolves to the **development build
above**, not the GitHub release API's `391fac16460f15233a7740550d858ac96df3419d`.
Use the observed image/source identities when patching the router or comparing
results. The image is unpatched. Its full tool set includes the server,
completion/CLI tools, benchmarks, and GGUF inspection/conversion tooling;
conversion and checkpoint Python code were not executed.

The upstream build disables `GGML_NATIVE`, builds its explicit ARM CPU
variants, and includes native GB10 CUDA code. GPU enumeration passed with
`CUDA_DISABLE_PTX_JIT=1`; the inference smoke also uses this setting. This is
the reference's upstream GCC/CUDA build, not a change to llmpalooza's D-032
Clang/C++23 toolchain. A future source rebuild must explicitly select its
CPU and CUDA targets and record the resulting new image identity.

The existing NVIDIA CDI device `nvidia.com/gpu=all` provides GPU access.
Docker is invoked with the existing passwordless `sudo -n`; no group,
daemon, driver, SDK, security policy, power, or clock changes were needed.
The host CUDA installation remains 13.0. No persistent server is left running.

## Model choice and provenance

All three planned GGUF repositories were publicly readable without an API
key at the recorded revisions. Gemma was selected for this first smoke:
its Q4 file is small enough to leave ample memory for setup and state tests.
Only the text model is installed; vision, MTP, other quants, and the two
large-model comparison pair have not been validated.

- Gemma: the pinned quantizer card declares Apache-2.0 and links to
  [Google's Apache-2.0 terms](https://ai.google.dev/gemma/apache_2).
  The [base config](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/4d7ae4984b7db7de8f8457170b3f1a419ee76d52/config.json)
  supplies the architecture cross-check. No model files are redistributed.
- Ornith: the pinned card declares **MIT**, also confirmed by the owner in
  this task. Its linked `LICENSE` path returned 404 at the base revision
  `10fbf86fed7ecee4a061f8b499a618f46001cac1`; retain the declaration and this
  retrieval gap accurately. Ornith remains eligible for reference work; it
  was not downloaded or executed in this setup.
- Qwen3.8-Flash-Next: the current card declares **Qwen Community License 1.0**,
  rather than Apache-2.0. This setup does not evaluate those terms or install
  that checkpoint. Its model architecture and memory fit remain unverified
  here. The large-model selection is still part of the switching experiment.

Dependency categories under D-017:

| Unit used | Category and terms | Scope |
| --- | --- | --- |
| llama.cpp/GGML and bundled GGUF reader | External reference/inspection tools, MIT; exact source license hash in `artifacts.json` | No implementation copied or linked into llmpalooza |
| Gemma GGUF | External benchmark data, Apache-2.0 | Local test input only |
| NVIDIA driver, CUDA runtime, cuBLAS | Reference platform dependencies, NVIDIA/component terms | Existing host driver plus libraries inside the digest-pinned image |
| glibc; libstdc++, libgcc, libgomp | Reference platform dependencies; LGPL-2.1-or-later; GPL with applicable GCC runtime exceptions | Container's observed versions recorded; not vendored |
| OpenSSL | Reference dependency, Apache-2.0 | Upstream server linkage; container version recorded |
| GCC, Python, NumPy and GGUF dependencies | Reference build/inspection tools under their own package terms | Upstream GCC build; Python only in experiment tooling, outside inference |

The upstream **full** image also contains unused conversion and multimedia
dependencies under their own licenses. Its digest pins that external
environment; this is not a claim that the whole image is permissively
licensed, an audit for redistribution, or admission into llmpalooza's
copyleft-disabled implementation profile. Upstream/license links:
[llama.cpp MIT](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/LICENSE),
[container recipe](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/.devops/cuda.Dockerfile),
[CUDA terms](https://docs.nvidia.com/cuda/eula/index.html).

## Reproduce on a chosen Spark

Run these commands on the chosen target, with this directory copied there
and the working directory set to that copy. The harness contains no host
names. Keep weights and outputs outside the repository. The installed model
in this run lives in `/home/pmeenan/.local/share/llmp/reference-models`;
raw results live in `/home/pmeenan/.local/share/llmp/reference-results`.

```sh
set -eu
umask 077
export DOCKER='sudo -n docker'
REFERENCE_IMAGE=$(python3 -c 'import json; print(json.load(open("artifacts.json"))["engine"]["image"])')
sudo -n docker pull --platform linux/arm64 "$REFERENCE_IMAGE"
mkdir -p "$HOME/.local/share/llmp/reference-models"
REFERENCE_MODEL="$HOME/.local/share/llmp/reference-models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"

# Skip this transfer when the installed file already matches the checksum.
# A .partial file is resumable; only promote it after verification.
curl --fail --location --retry 3 --continue-at - \
  --output "$REFERENCE_MODEL.partial" \
  https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/c099eb48e663fd284577b04978a94ffccb261841/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
printf '%s  %s\n' f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f "$REFERENCE_MODEL.partial" | sha256sum --check -
# Run only after the preceding checksum passes, with no existing final file.
test ! -e "$REFERENCE_MODEL" && mv "$REFERENCE_MODEL.partial" "$REFERENCE_MODEL"

python3 smoke.py "$REFERENCE_MODEL" \
  "$HOME/.local/share/llmp/reference-results/smoke-1"
```

`smoke.py` verifies exact size and SHA-256 before loading the GGUF. The
output directory must be new. It starts its own uniquely named container,
publishes the API on **host loopback only**, sets a 48 GiB **cgroup** memory
limit with no additional container swap, runs as the invoking UID, uses read-only
weights/root filesystem, generates an ephemeral API key, and attempts container
cleanup on normal exit or error (Docker failures are reported).
The cgroup setting is not a validated bound on all CUDA allocations in
Spark's unified physical memory. Docker statistics alone cannot establish a
node-wide execution budget; record host available memory and actual buffers
separately, and validate budget enforcement in the pressure experiment.
The private output directory retains synthetic responses, server logs, and
a slot-state file for inspection. These are deliberately enabled benchmark
inputs/outputs, not user conversations. Remove that output directory when
no longer needed; no service or automatic retention job is installed.
SIGTERM and Ctrl-C unwind through cleanup. A forced kill, host failure, or
interruption during Docker creation can still leave a container; its unique
name is written to `results.json` before creation for explicit recovery.

The run uses all GPU layers, 8 CPU threads, an 8192-token context, one slot,
512-token batch/microbatch, f16 K/V, **`--swa-full`**, Flash Attention on, RAM prompt cache
disabled, no drafter/MTP, and greedy decoding. The server's normal warmup
remains enabled. The fixed synthetic prefix is shorter than the 1024-token
sliding window. Results do not establish long-context state coverage.

To repeat model inspection after checksum verification:

```sh
sudo -n docker run --rm --network none --read-only \
  --user "$(id -u):$(id -g)" --memory 2g --memory-swap 2g \
  --mount "type=bind,src=$REFERENCE_MODEL,dst=/model.gguf,readonly" \
  --mount "type=bind,src=$PWD/inspect_model.py,dst=/inspect_model.py,readonly" \
  --env PYTHONPATH=/app/gguf-py --entrypoint python3 \
  "$REFERENCE_IMAGE" /inspect_model.py /model.gguf
```

This uses the reader bundled with the pinned source, not an independently
updated PyPI reader. `inspect_model.py` fails on unexpected architecture,
expert axis, projection names, or missing expert layers. It summarizes tensor
storage bytes, not llmpalooza's future 2 MiB paging layout or actual read traffic.

## Verified model and memory accounting

The hash-verified GGUF contains 658 tensors with 25,233,142,046 elements,
30 MoE layers, **128 routed experts per layer, top-8, and one shared FFN**.
Its expert intermediate width is 704; the shared FFN width is 2112.
These values agree with the pinned base config. Stored tensor accounting:

| Component | Bytes |
| --- | ---: |
| Entire GGUF | 16,947,541,728 |
| All tensor payloads | 16,931,716,216 |
| Routed experts, including per-expert down scales | 14,353,054,720 |
| All other tensors, including routers and shared FFNs | 2,578,661,496 |
| Shared FFN projections, a subset of the preceding row | 568,719,360 |
| One selected expert's closure in each of layers 0–28 | 3,717,124 |
| One selected expert's closure in layer 29 | 4,336,644 |

Each expert closure includes fused `ffn_gate_up_exps.weight` (Q4_K,
2,230,272 bytes per expert), `ffn_down_exps.weight` (Q5_1,
1,486,848 bytes; Q8_0 and 2,106,368 bytes in the final layer), and a
4-byte f32 `ffn_down_exps.scale`. Shared projections are the ordinary
`ffn_gate`, `ffn_up`, and `ffn_down` tensors, not `_shexp` tensors. Routers
and their input scales are in the non-routed partition. These are logical
stored closures; no 2 MiB extent padding, shared-read amplification, or
observed disk traffic is implied.

The successful 8192-context run logged **31/31 layers offloaded to the GPU**:

| Reference allocation | Logged MiB |
| --- | ---: |
| CUDA model buffer | 16,147.43 |
| CUDA host model buffer | 748.00 |
| Global KV, 8192 cells across 5 layers, f16 K and V | 160.00 |
| Full SWA KV, 8192 cells across 25 layers, f16 K and V | 1,600.00 |
| CUDA compute buffer | 156.02 |
| CUDA host compute buffer | 27.02 |
| CUDA host output buffer | 1.00 |

From these observed buffers, global K/V costs **20 KiB per token**, and
SWA K/V costs **200 KiB per retained cell**, or 220 KiB per token with
full-context retention. This model has no recurrent state. With the default
SWA policy, the observed SWA allocation is 1536 cells (1024-token window
plus batch allowance), 300 MiB; global KV stays 160 MiB. Thus the workaround
raises KV allocation from **460 to 1760 MiB** at this configuration.

After the initial completion, host `MemAvailable` was 103,563,644 and
104,071,616 KiB in the two successful runs, about **98.8–99.3 GiB** of
121.69 GiB visible physical memory. These are snapshots with warm file
cache, not measured peak usage or an admission guarantee. Docker reported
only 1.372 GiB in the first successful run despite the roughly 16 GiB CUDA
model buffer: do not treat its cgroup statistics/limit as the node's CUDA
occupancy or as a validated pressure mechanism.

## Verification and handoff

Two complete GPU/restore smoke runs passed on `spark`; the second used the
final harness with the explicit GPU-log assertion and SIGTERM handler.
Both used the same immutable artifacts and full-SWA settings above:

| Check | Observed result in each successful run |
| --- | --- |
| GGUF integrity before load | Exact size and full SHA-256 matched |
| GPU execution | 31/31 layers offloaded, PTX JIT disabled |
| OpenAI chat endpoint | Synthetic request returned `READY` |
| Native completion endpoint | 626 input tokens evaluated, 2 output tokens |
| Slot save | 627 tokens, 141,268,896 bytes |
| Restore in a new process | Same token and byte counts |
| Resident continuation | 627 cached tokens, 12 prompt tokens processed |
| Restored continuation | 627 cached tokens, 12 prompt tokens processed |
| Continuation comparison | All 32 generated token IDs matched within each run |
| Cleanup | No test containers left running |

The 627 saved tokens include the initial prompt and the first generated
token; the generated end-of-turn token is processed as part of the 12-token
extension. Explicit token-array requests avoid text retokenization changing
the tested prefix. This is a short synthetic native-completion reuse test,
not a chat-quality assessment or a general numerical/logit equivalence proof.

An earlier run without `--swa-full` is a **negative control**, not a pass:
save/restore returned identical counts, but the restored request processed
all 639 prompt tokens while the resident path processed 18 (checkpoint
rollback). The server explicitly reported missing cache coverage and full
re-processing. Generated continuations then differed, which does not alone
demonstrate state corruption because their computation/batch paths differed.
The source's SWA checkpoint fallback explains the observed path. See
[RE-004](../../rough-edges.md#re-004-llamacpp-gemma-slot-restore-reports-success-but-default-swa-re-prefills--2026-09-21-status-worked-around).
Retain both reference configurations in the later comparison protocol;
the full-SWA memory cost cannot be silently omitted or treated as normal
windowed-cache cost.

The model had just been hashed/read before each smoke, and both loads were
warm-cache runs. No cold-storage, latency-distribution, sustained throughput,
or full-swap performance number is claimed. Raw responses, logs, inspection
output, and synthetic slot files stay outside Git. The recorded identities,
aggregate counts, allocation tables, and retained harness suffice to repeat
the checks without that raw bundle.

Workstation checks: Python compilation, JSON parsing, `git diff --check`,
plus the independent review's adversarial harness/inspector checks.
No llmpalooza application tests exist yet; no inference ran on the workstation
or `spark-b`. Host drivers/toolkits/security settings were unchanged, and
all changes remain uncommitted for the human.

No A→B→A or paging result is claimed by installation alone.
Router instrumentation, long contexts,
second-model admission/displacement, cold/warm cache controls, measured
switch/restore latencies, and generation-stall criteria remain separate plan
items. Checkpoint/engine compatibility is demonstrated only by the tests
actually reported here; numerical reference logits and broader model support
remain unvalidated.

## Independent review

A separate agent reviewed the full uncommitted change and challenged the
external-input, provenance, cleanup, and restore assertions. No unresolved
findings remain. The review verified recorded source hashes and model
identities against fetched provenance, reconciled the accounting and both
successful runs against their raw results, and inspected the negative-control
and final-run logs on `spark`. No reference containers remained there.

Workstation failure injection verified rejection of a bad hash, an existing
output directory, mismatched restore counts/bytes/token IDs, excessive
re-prefill, and missing GPU evidence; startup resets recovered, and stop/log
failures and SIGTERM still attempted container removal. Inspector challenges
rejected missing expert scales/shared projections and unexpected expert axes
or projection names. The builder fixed the initially weak reuse assertion,
cleanup paths, checksum-promotion instructions, and cgroup-memory wording;
the final harness passed the repeated challenge checks.

The review did not run additional inference or a pressure experiment.
Long-context reuse, A→B→A costs, physical-memory limit enforcement, and
forced-kill/host-failure cleanup remain outside this result as stated above.
