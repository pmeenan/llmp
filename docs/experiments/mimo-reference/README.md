<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MiMo-V2.6-Flash-RL two-Spark sharded reference

External reference on `spark` and `spark-b` (2026-09-22), not llmpalooza
execution, sharding support or a supported model combination. It answers
the [candidate matrix](../model-candidates.md)'s first MiMo step: a bounded
text-only sharded boot with per-node physical peaks, weight/state bytes,
loading, prefill/decode and transport behavior. Raw logs, generated text and
per-second traces stay outside Git.

**Result.** The full 309B-parameter checkpoint runs as TP=2/EP=2 across the
two Sparks from each node's own copy, with the routed experts kept in packed
MXFP4. Each rank holds 81.9–83.8 GiB after loading; with a deliberately
small KV pool the lowest sampled `MemAvailable` was 18.8/20.0 GiB and the
memory guard never fired. Greedy answers were correct and identical. The
server took about 611 s from container start to health, dominated by 455–507 s
of weight loading. Prefill reached about 2,360 tokens/s at 20K tokens, and a
resident prefix-cache hit cut the 20K-token first-token time from 8.68 s to
0.36 s. Decode without speculation or CUDA graphs ran at 18.5 tokens/s with
54 ms median and 61–62 ms p99 token gaps. NCCL used both RoCE HCAs of the DAC
port; the link carried about 0.84 MB per prefill token and about 4 MB per
decoded token in each direction, far below its capacity.

## Results — boot 3, 2026-09-22

Nodes `spark-c4e2` (rank 0, head) and `spark-56f5` (rank 1), GB10, driver
580.178.04, DAC `sparky` link. One boot; each workload row is one request.
SGLang reports memory in GiB, and on the GB10 its "available" figure is host
available memory, not a separate device pool.

| Stage | Rank 0 (`spark`) | Rank 1 (`spark-b`) |
| --- | ---: | ---: |
| Weight load | 507.11 s | 454.85 s |
| Available at load start → end, GiB | 111.90 → 29.96 | 113.54 → 29.72 |
| Weight memory reported, GiB | 81.94 | 83.82 |
| Full-attention KV (131,072 tokens, FP8), GiB | 0.70 | 0.70 |
| Sliding-window KV (3,904 tokens, FP8), GiB | 0.18 | 0.18 |
| Available after pools, GiB | 28.93 | 29.41 |
| Lowest sampled `MemAvailable` (at 572/576 s) | 18.83 GiB | 19.97 GiB |
| Swap-out during boot and workload | 3,225 pages | 5 pages |

The rank-0 log shows the packed path for all 47 MoE layers ("Preparing DSv4
MXFP4 experts for FlashInfer SM120 CUTLASS W4A8"); layer 0 is dense. Both
ranks logged `NET/IB : Using [0]rocep1s0f1:1/RoCE [1]roceP2p1s0f1:1/RoCE` with
out-of-band traffic on the DAC interface; the loaded NCCL is 2.30.7, not the
2.29.7 reported by the PyTorch wheel. Only `configuration_mimo_v2.py` appeared
in Transformers' remote-module cache on either node. Multimodal mode was
active (vision attention backend selected, 0.10 GiB of the KV budget reserved
for multimodal allocations; see [RE-012](../../rough-edges.md#re-012-pinned-sglang-mimo-v2-startup-fails-with-a-misleading-processor-error-without-torchcodec--2026-09-22-status-worked-around));
requests were text only. Boot took about 611 s from container start to health
(617.9 s from harness start); teardown removed both
containers 8 s after the workload, and `nvidia-smi` reported no compute
processes on either node within 0.8 s after removal, with `MemAvailable`
back to 118.0 GiB on both, at or above preflight. The worker logged an `AttributeError` in SGLang's HTTP
entrypoint 6 s after the head's graceful SIGTERM drain, when its scheduler
lost the NCCL peer; it is a shutdown-path error after all requests
completed, not a serving failure.

Workload (client-side, loopback, greedy, thinking disabled):

| Request | Prompt tokens | First content, s | Notes |
| --- | ---: | ---: | --- |
| Arithmetic smoke ×3 | 25 | 0.205–0.223 | All answered `42`, identical |
| ~1K documents ×2 | 1,309 / 1,291 | 0.686 / 0.658 | ≈1,910–1,960 tokens/s |
| ~4K documents ×2 | 5,157 / 5,133 | 2.367 / 2.161 | ≈2,180–2,380 tokens/s |
| ~16K-record documents ×2 | 20,416 / 20,473 | 8.645 / 8.685 | ≈2,360 tokens/s |
| Exact repeat of the last | 20,473 | 0.363 | 20,416 cached (319 pages of 64) |
| 256-token decode ×2 | 1,303 / 1,300 | 0.663 / 0.659 | 18.52 / 18.51 tokens/s; gaps median 54.0 ms, p95 59.6–60.1, p99 61.1–61.9, max 62.3 |

First-content time on a one-token request approximates prefill; the engine's
own decode throughput log (18.45–18.63 tokens/s after each request's first
interval) agrees with the client. The
prefix-cache repeat is reuse on a resident server, not state restoration.
Short prompts below one 64-token page reported no cached tokens on repeat.

Transport, from the head's RoCE port counters summed over both HCAs (each
carried about half; transmit and receive are nearly equal):

| Request | Bytes per direction | Per token |
| --- | ---: | ---: |
| 1,309-token prefill | 1.110 GB | 0.85 MB per prompt token |
| 20,416-token prefill | 17.195 GB | 0.84 MB per prompt token |
| 1,303-token prompt + 256 decoded | 2.127 GB | ≈4.0 MB per decoded token after subtracting prefill |

Two all-reduces of a 4,096-wide BF16 hidden state per layer would be about
0.79 MB per token, close to the prefill rate. Decode moves roughly five times
that per token; small-message protocol overhead and other per-step collectives
are plausible contributors but were not isolated. At these rates prefill uses
about 2 GB/s and decode about 75 MB/s of a link measured at 22 GB/s for large
NCCL transfers: this workload is not bandwidth-bound. Collective latency
versus compute per step was not isolated; per-step collective latency, not
link bandwidth, is the transport question for M4.

Limits: one boot and single requests (no distributions, concurrency or
long-context validation); no speculative decoding, CUDA graphs or multimodal
inputs; no switching, state save/restore, pressure or failure injection.
The 32,768-token context and 131,072-token pool are harness choices, not
capacity. No numerical reference exists for this model, so beyond the
arithmetic smoke and repeat identity no correctness claim is made. Recipe
throughput figures are not comparable: they use speculation, graphs and
different memory settings.

## Identity

[`pins.json`](pins.json) records all 90 files of
`XiaomiMiMo/MiMo-V2.6-Flash-RL` at `5711b268169967567844e1e560e8a3966da959b1`
(177,767,644,228 bytes) with SHA-256 for every file; non-LFS files were
checked against their Git blob IDs before their SHA-256 was recorded. The
MiaAI recipe pins `3b38d063180c3e4aed9691fdc735f3d10b266ee4`. Between the two
revisions only `README.md`, the technical-report PDF and `dflash/config.json`
(a JSON-syntax fix) differ; all weights, configs, tokenizer and chat template
are byte-identical, so the recipe's observations refer to the same weights.

The checkpoint was fetched once on `spark-b` with [`../hf_fetch.py`](../hf_fetch.py)
(size and SHA-256 checked before publication, exact tree enforced), copied to
`spark` over the DAC link (`10.100.208.0/24`, one rsync-over-ssh stream,
410.6 s) and re-verified there. Each node reads its own copy; there is no
shared filesystem.

The runner is `lmsysorg/sglang@sha256:9e1fb4c3…` (tag
`nightly-dev-cu13-20260921-0f6761b5`, the first nightly carrying the packed
MXFP4 MoE path per the recipe): SGLang `0f6761b54`, Transformers 5.12.1,
PyTorch 2.13.0+cu130, Triton 3.7.1; PyTorch reports NCCL 2.29.7 but the ranks
loaded NCCL 2.30.7. [`Dockerfile`](Dockerfile) derives
`llmp-mimo-reference:20260922` from it by adding hash-pinned `torchcodec`
0.16.0 ([`requirements-extra.txt`](requirements-extra.txt)); without it the
engine cannot start this architecture (RE-012). Both boots before the one
reported here failed at startup, before loading weights, on the missing
`torchcodec`; the first also omitted `--enable-multimodal`, which did not
change the failure. The harness tore both down and confirmed both GPUs
released.

The pinned Transformers has no native `mimo_v2` configuration, so config
parsing requires `--trust-remote-code`. The executed file,
`configuration_mimo_v2.py` (SHA-256 `773062ac…`, Apache-2.0 header), was
audited first: a `PretrainedConfig` subclass importing only `copy` and
Transformers, with no I/O, network or dynamic execution. Model execution uses
SGLang's own `mimo_v2` implementation, not the checkpoint's
`modeling_mimo_v2.py`. This reference exception does not relax llmpalooza's rule
that native import never executes checkpoint code.

## Method

[`run.py`](run.py) runs from the workstation with [`config.json`](config.json).
Node names, addresses and interfaces there are this deployment's measured
inputs (see the [interconnect baseline](../interconnect/README.md)), not
llmpalooza configuration.

- **Preflight:** both nodes idle (no CUDA compute processes) with at least
  100 GiB `MemAvailable`.
- **Guard:** [`guard.py`](guard.py) samples `MemAvailable` every 100 ms on
  each node and kills that node's rank container below 6 GiB. Spark's CUDA
  allocations share host memory and are not bounded by a container cgroup;
  the recipe reports node reboots from exhausted memory. The guard is a
  harness safety net, not a llmpalooza budget mechanism.
- **Ranks:** one container per node, worker first. Host networking is needed
  for NCCL and the rendezvous on the DAC address; the API binds to head
  loopback only. Containers run as the invoking user with `IPC_LOCK`,
  unlimited memlock and `/dev/infiniband`, not `--privileged`, and are
  offline (`HF_HUB_OFFLINE`). NCCL uses both RoCE HCAs of the shared DAC port
  (`=rocep1s0f1:1,roceP2p1s0f1:1`) with `NCCL_DEBUG=INFO` to record the
  transport actually selected.
- **Engine settings:** TP=2, EP=2, packed MXFP4 experts on
  `flashinfer_mxfp4`, no all-to-all backend, Triton attention, FP8 KV cache,
  SWA ratio 0.03, page size 64, chunked prefill 2,048. Deliberately bounded:
  32,768-token context, 131,072-token KV cap, memory fraction 0.85, CUDA
  graphs disabled, speculative decoding off. Multimodal mode is enabled
  (`triton_attn` vision attention); the engine initialized this
  architecture's multimodal processor even without the flag, and a boot
  with `torchcodec` but without it was not tried. Every request is text.
  These follow the recipe's validated engine choices where they concern
  correctness or memory safety; its throughput-oriented settings (0.93
  fraction, ~2.7M-token pool, EAGLE/DFlash, decode graphs) are not used.
- **Workload:** [`workload.py`](workload.py) on the head, loopback only,
  greedy (`temperature=0`) chat requests with thinking disabled and synthetic
  prompts: a three-times-repeated arithmetic smoke; a prefill ladder of
  distinct ~1K/4K/16K-token documents, two each, generating one token; an
  exact repeat of the last 16K prompt (prefix-cache reuse on the resident
  server); two 256-token decodes with `ignore_eos`. Client-side timings are
  first streamed content and inter-arrival gaps; RDMA port counters on the
  head bracket each request.
- **Teardown:** head then worker are stopped; the harness waits until
  `nvidia-smi` reports no compute processes on both nodes.

[`summarize.py`](summarize.py) reduces a run to [`aggregate.json`](aggregate.json):
engine load/pool/transport log lines, guard minima, and workload
measurements with generated text replaced by its length (smoke answers kept).

## License and provenance boundary

The checkpoint's Hugging Face license field is MIT; it is external model
data, not redistributed. SGLang (Apache-2.0) and its image are external
reference tools under D-017; the container's full component terms are not
audited here. The [MiaAI two-Spark recipe](https://github.com/MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks/tree/201be3e743d215291feee09fe470822ec453c63d)
is AGPL-3.0. Its README and default profile were read for validated flags and
failure modes; **none of its scripts, patches or image are used**, and this
harness is independently written. Its reported numbers are external
observations, never substitutes for ours. In particular its
`sitecustomize.py` loader/MTP patch is not applied, so our weight loading
uses SGLang's default mmap path and speculative decoding stays off.

## Reproduction

```sh
python3 docs/experiments/hf_fetch.py --pins docs/experiments/mimo-reference/pins.json \
  --output ~/.local/share/llmp/mimo-reference/model --jobs 6   # on each node
sudo -n docker pull lmsysorg/sglang@sha256:9e1fb4c395b9c406136e10aa445b8784d06bca3839623b52cbe4a3b231a157a8
sudo -n docker build --platform linux/arm64 -t llmp-mimo-reference:20260922 docs/experiments/mimo-reference   # on each node
python3 docs/experiments/mimo-reference/run.py --name RUN --results PRIVATE_DIR   # workstation
python3 docs/experiments/mimo-reference/summarize.py PRIVATE_DIR/RUN > aggregate.json
```

Run directories on the nodes and the copied results are private and
unrotated; remove them when no longer needed.

Teardown makes bounded attempts on each node independently and saves
`run.json` even when a node or result copy is unavailable. Launch, workload
or cleanup failures return a nonzero exit status. `cleanup_errors` identifies
failed steps; `not_released` includes nodes whose GPU release could not be
confirmed. An unreachable node may still have a container or guard running;
check that node before another run.

Local orchestration checks (mocked SSH and containers):
`python3 -m unittest discover -s docs/experiments/mimo-reference -p 'test_run.py'`.
