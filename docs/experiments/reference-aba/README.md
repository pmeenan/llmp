<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M0 A→B→A reference experiment — 2026-09-21

This measures the unpatched, pinned llama.cpp reference on `spark`, using
Gemma 4 as A and Ornith 1.5 as B. It supplies the first measured reference
cycle for D-025. It is not llmpalooza execution or evidence of expert-paging
feasibility. The large DeepSeek/Qwen pair remains a separate workload.

## Protocol fixed before repeated measurements

The external frozen trace contains three turns building A's synthetic service
notebook (14,619, 16,461 and 18,301 input tokens), a 31-token B request, and
an 18,339-token continuation on A. The saved A state covers 18,303 tokens;
18,297 form the reusable prefix and 42 continuation tokens require processing.
A produces up to 128 output tokens and B
up to 64. Greedy sampling uses seed 42. Identical token arrays, including
the previous generated history, are supplied in every arm. This tests state
coverage and performance, not answer quality. Complete accumulated messages
are formatted by the checkpoint's own `/apply-template` endpoint. Gemma's
template removes the generation-only empty thinking marker from completed
assistant turns; that rewrites six saved tail tokens in this trace. The
longest common prefix determines expected reuse. A's continuation finishes
in 118 output tokens; B reaches its 64-token cap.

`artifacts.json` pins both model revisions, sizes and SHA-256 hashes, the
container digest, source revision, pressure helper binary, and frozen trace.
The execution environment is `spark-c4e2`, NVIDIA GB10, driver 580.178.04,
Linux 7.0.0-1019-nvidia, with the reference's CUDA 13.3 ARM64 image and
source `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`. PTX JIT is disabled.
The run manifest also records the exact harness hash and filesystem/device.
The trace is kept outside Git; its SHA-256 is
`7625929fd143bafa31f5d3ebfe03e4dcf96452129b0d080a67434940c85e2383`.
No model files, captured token arrays, generated text, logs or state files
are redistributed in this repository.

| Arm | Resident models | Incoming file cache | Locked pressure | State on return |
| --- | --- | --- | ---: | --- |
| `resident` | A and B throughout | Warm | 0 GiB | Live A slot |
| `warm_restore` | Native LRU permits one | Warm | 0 GiB | Saved/restored |
| `warm_recompute` | Native LRU permits one | Warm | 0 GiB | Re-prefill |
| `cold_restore` | Native LRU permits one | Cold | 0 GiB | Saved/restored |
| `cold_recompute` | Native LRU permits one | Cold | 0 GiB | Re-prefill |
| `pressure_restore` | Native LRU permits one | Cold | 80 GiB | Saved/restored |
| `pressure_recompute` | Native LRU permits one | Cold | 80 GiB | Re-prefill |
| `normal_pressure_restore` | Native LRU permits one | Cold | 80 GiB | Attempt default-SWA restore |
| `normal_pressure_recompute` | Native LRU permits one | Cold | 80 GiB | Default-SWA re-prefill |

Warm and cold zero-pressure arms isolate file-cache effects. The 80 GiB
arms separately exercise physical pressure. They are not a warm-versus-cold
comparison with one factor changed. Cold means zero incoming GGUF pages
resident according to `mincore`, established with `POSIX_FADV_DONTNEED`
before timing. It does not mean a reset NVMe controller cache. Warm requires
at least 99.9% of incoming pages resident. A's saved state is conditioned
the same way on return. Conditioning and initial history creation are outside
the switch timer; every save, durable flush, eviction, load, restore and
continuation after the switch decision is inside it.

First-token time starts immediately before saving/unloading the outgoing
model and ends at the first nonempty token-ID event in the streamed response.
HTTP headers and empty events do not count. First nonempty text is recorded
separately. The reported wait includes native router work and polling overhead.
Both outgoing A and outgoing B are saved in restore arms; B is not silently
discarded on the return switch. Recompute arms save neither state.

`/models/load` triggers the reference's own LRU unload and child-process load;
the harness polls actual status and verifies the outgoing child is unloaded.
The reference's `--models-max` is a **count**, not a physical-memory limit.
One native router plus model child processes is llama.cpp's reference design,
not llmpalooza's future one-native-process-per-node design. `/completion` and
`/slots/0?action=save|restore` carry the model alias. Slot files are written
by the server, then the harness fsyncs both file and directory. This adds
durability that the pinned save API does not itself guarantee. Restore
counts must match the file size and save receipt; actual cached/prefilled
tokens and generated token IDs establish reuse.

Matched arms use full Gemma SWA allocation, no RAM prompt cache, f16 K/V,
Flash Attention, one slot, 512-token batch/microbatch, eight CPU threads,
all GPU layers and normal startup warmup. A's context is 32,768; B's is
8192. No speculative decoder or MTP is enabled. The normal-reference arms
retain default SWA and the default 8192 MiB RAM cache/checkpoint behavior;
all other workload/compute settings match. They include both the available
slot-restore attempt and recomputation, so a failing default reuse path is
not mistaken for the best available default result. Effective child
arguments and full reference logs are retained externally.

These reference behaviors were checked against the pinned
[server documentation](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/tools/server/README.md),
[native model router](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/tools/server/server-models.cpp),
and [slot handling](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/tools/server/server-context.cpp).
No expert-route logging patch is applied in this experiment. The native router
does not automatically persist the outgoing slot across child-process unload;
the explicit save/restore/durable-flush orchestration is part of this harness.

## Physical budget and acceptance

The host has 130,661,203,968 bytes (121.6877 GiB) of physical memory. A small
C++23 helper allocates, locks and touches 80 GiB; its exact executable hash
is checked before `sudo`. Its RSS and `VmLck` are verified, its liveness is
sampled during execution, and its lock is rechecked at phase boundaries.
This leaves at most 41.6877 GiB for models and everything else. The helper
requires 24 GiB of initial remaining headroom and releases on stdin EOF.
A 50 ms monitor releases pressure and rejects the trial if `MemAvailable`
drops below 2 GiB. No host swap, driver, clock or security setting is changed.

The matched reference's separate logged CUDA allocations are:

| Allocation | A, MiB | B, MiB |
| --- | ---: | ---: |
| Model buffer | 16,147.43 | 19,902.90 |
| K/V buffer(s) | 7040.00 | 160.00 |
| Recurrent state | — | 62.81 |
| Compute buffer | 204.02 | 108.05 |
| Sum | 23,391.45 | 20,233.76 |

Their combined minimum is 43,625.21 MiB (42.6027 GiB), exceeding the
remaining physical capacity by about 0.915 GiB before any OS, host buffers
or file cache. Each model alone fits. This proves the two measured matched
execution envelopes cannot be simultaneously resident with the holder.
It does not prove the smaller default-SWA envelopes cannot fit; normal arms
use the same pressure and one-model LRU policy but carry that qualification.
Those two rows measure **normal optimizations under a forced one-model
policy**. They do not establish the best normal placement or an unconstrained
normal-reference floor. Two-model placement with default SWA under this budget
remains unvalidated; enabling defaults is not shown to require displacement.
Zero-pressure unload arms are policy-forced, not physically forced.
CUDA buffer sizes are counted once; CPU staging and the page cache are not
treated as additional memory tiers.
The pinned [KV allocator](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/src/llama-kv-cache.cpp)
allocates and clears the complete logged KV buffers, including full SWA;
these are not untouched logical context reservations. Even omitting compute
scratch, the weights/state/holder sum exceeds physical memory by 624.906 MiB.

During pilots, strict zero-swap and then 1 MiB rules rejected completed
pressure cycles for 8 KiB and 93.9 MiB of whole-node swap traffic respectively.
Attribution is unknown. All pilots are excluded. Before main repeats,
acceptance was fixed to allow at most **256 MiB of observed combined swap-in
plus swap-out per measured trial**, including initial setup from before the
pressure allocation through A's completed continuation, before cleanup,
using the actual host page size and applying the same rule to every arm.
Exact counters and traffic are retained. Even subtracting that entire
allowance from the non-fit margin and omitting both compute buffers leaves
368.906 MiB of excess. This is not a claim of zero swap or zero swap overhead.
The allowance is not raised to admit main-run failures. Any OOM,
lost pressure allocation, low-memory guard, cache-condition violation,
incomplete stream, restore mismatch or cleanup error rejects a trial.

Storage counters are whole-node `nvme0n1` sectors read/written, including
reference startup, filesystem traffic and unrelated node activity. They are
observed physical block traffic, not per-model accounting or inferred bytes
from file length. Logical/allocated state-file bytes are reported separately.
Memory is sampled `MemTotal - MemAvailable`, including the pressure holder
and other host activity; the reported maximum covers the switching cycle,
not initial setup. It is not exact process memory or a continuous peak.
The 50 ms interval is requested, not a real-time guarantee: independent
validation of 19,649 cycle samples measured a 50.534 ms median gap and
108.341 ms maximum gap.

## Results

All **27 main trials passed**: three repetitions of each of the nine arms,
interleaved in the table order within each repetition. All initial A turns,
all 64-token B outputs and all 118-token A continuations matched the frozen
token sequences, including recompute/default arms. Every measured first
token also contained nonempty text. Cold files had zero resident pages;
warm weight and saved-state receipts met the 99.9% threshold.

The run started at 2026-09-21 21:44:19 UTC on `spark-c4e2`, using ext4 on
`/dev/nvme0n1p2`. The executed `experiment.py` SHA-256 is
`9639d085b682ff92c198b3033054109b8a420b8a2fad660f6bcee93b98d43f83`.
Raw results are in `/home/pmeenan/.local/share/llmp/aba/main-1` on the
target. All pilots are excluded; no main trial failed or was replaced.

Values below are **median [observed minimum–maximum]**, with n=3 per arm.
This sample count does not establish tail percentiles or confidence bounds.

| Arm | A→B first token, s | B→A first token, s | A cached / processed tokens |
| --- | ---: | ---: | ---: |
| `resident` | 0.155 [0.155–0.157] | 0.089 [0.089–0.092] | 18297 / 42 |
| `warm_restore` | 3.879 [3.479–3.964] | 4.062 [4.049–4.165] | 18297 / 42 |
| `warm_recompute` | 3.561 [3.206–3.565] | 11.215 [11.173–11.242] | 0 / 18339 |
| `cold_restore` | 22.080 [21.252–24.229] | 18.597 [18.365–21.056] | 18297 / 42 |
| `cold_recompute` | 20.973 [19.809–24.005] | 24.504 [24.502–25.469] | 0 / 18339 |
| `pressure_restore` | 21.232 [20.934–21.664] | 18.304 [17.647–18.783] | 18297 / 42 |
| `pressure_recompute` | 21.247 [21.011–25.034] | 25.236 [24.757–25.675] | 0 / 18339 |
| `normal_pressure_restore` | 21.182 [20.869–25.227] | 24.375 [24.360–24.492] | 0 / 18339 |
| `normal_pressure_recompute` | 21.042 [20.382–22.850] | 23.377 [23.038–24.249] | 0 / 18339 |

For this trace, cold-cache return under the matched constrained budget took
**18.304 s with saved state versus 25.236 s with recomputation**. The
warm-cache equivalents were 4.062 and 11.215 s; live residency was 0.089 s.
These are measured reference waits, not a prediction of llmpalooza performance.

B→A component medians, in seconds (separate medians need not sum exactly):

| Arm | Save B + fsync | Native eviction/load A | Restore A | Continuation to first token |
| --- | ---: | ---: | ---: | ---: |
| `warm_restore` | 0.059 | 3.791 | 0.067 | 0.143 |
| `warm_recompute` | 0.000 | 3.761 | 0.000 | 7.413 |
| `cold_restore` | 0.058 | 17.910 | 0.490 | 0.145 |
| `pressure_restore` | 0.058 | 17.625 | 0.484 | 0.151 |
| `pressure_recompute` | 0.000 | 17.776 | 0.000 | 7.459 |
| `normal_pressure_restore` | 0.060 | 17.543 | 0.484 | 6.310 |
| `normal_pressure_recompute` | 0.000 | 17.082 | 0.000 | 6.294 |

In the matched pressure restore arm, saving A before the outward switch
took 0.351 s median, including fsync. Cold model reload dominates the
wait. Its time includes reference process startup, model loading, buffer
allocation and warmup; it is not an isolated SSD bandwidth measurement
and must not be replaced with the earlier direct-I/O spike’s throughput.

Observed whole-node block traffic through each first token, GiB:

| Arm | A→B read | A→B written | B→A read | B→A written |
| --- | ---: | ---: | ---: | ---: |
| `resident` | 0.0000 [0.0000–0.0000] | 0.0000 [0.0000–0.0000] | 0.0000 [0.0000–0.0000] | 0.0000 [0.0000–0.0000] |
| `warm_restore` | 0.0000 [0.0000–0.0250] | 0.5448 [0.5448–0.5449] | 0.0000 [0.0000–0.0000] | 0.0632 [0.0632–0.0632] |
| `warm_recompute` | 0.0000 [0.0000–0.0001] | 0.0003 [0.0003–0.0056] | 0.0000 [0.0000–0.0069] | 0.0001 [0.0000–0.0065] |
| `cold_restore` | 19.7133 [19.7133–19.7303] | 0.5502 [0.5500–0.5520] | 16.3284 [16.3283–16.3285] | 0.0633 [0.0633–0.0634] |
| `cold_recompute` | 19.7135 [19.7133–19.7150] | 0.0073 [0.0061–0.0073] | 15.7836 [15.7836–15.7836] | 0.0002 [0.0002–0.0044] |
| `pressure_restore` | 19.7789 [19.7784–19.7792] | 0.5603 [0.5578–0.5678] | 16.3530 [16.3529–16.3531] | 0.0677 [0.0648–0.0947] |
| `pressure_recompute` | 19.7241 [19.7236–19.7264] | 0.0072 [0.0060–0.0135] | 15.8164 [15.8144–15.8190] | 0.0005 [0.0001–0.0006] |
| `normal_pressure_restore` | 19.7270 [19.7185–19.7451] | 0.5506 [0.5505–0.5519] | 16.3474 [16.3354–16.3742] | 0.0634 [0.0633–0.0681] |
| `normal_pressure_recompute` | 19.7502 [19.7456–19.7656] | 0.0085 [0.0063–0.0242] | 15.8355 [15.8070–15.8501] | 0.0063 [0.0031–0.0064] |

Rounded zeros can contain a small amount of node/filesystem traffic. Writes
in recompute arms are not serialized conversation state. The separately
reported logical spill is zero in resident/recompute arms and exactly
652,659,012 bytes (622.424 MiB) in every restore arm.

Node footprint and observed swap traffic:

| Arm | Sampled maximum used, GiB | Minimum available, GiB | Trial swap traffic, MiB |
| --- | ---: | ---: | ---: |
| `resident` | 49.141 [49.047–49.161] | 72.547 [72.527–72.641] | 0.738 [0.055–1.457] |
| `warm_restore` | 28.348 [27.938–28.422] | 93.340 [93.266–93.750] | 0.094 [0.012–0.820] |
| `warm_recompute` | 28.403 [27.943–28.480] | 93.285 [93.208–93.745] | 0.059 [0.020–0.242] |
| `cold_restore` | 28.057 [27.824–28.394] | 93.631 [93.293–93.864] | 0.184 [0.047–0.285] |
| `cold_recompute` | 27.861 [27.777–27.886] | 93.827 [93.802–93.910] | 0.074 [0.035–0.121] |
| `pressure_restore` | 107.881 [107.859–108.017] | 13.807 [13.671–13.829] | 21.801 [14.992–40.898] |
| `pressure_recompute` | 107.804 [107.784–107.916] | 13.884 [13.772–13.904] | 1.672 [0.234–11.539] |
| `normal_pressure_restore` | 104.272 [104.240–104.348] | 17.416 [17.340–17.448] | 0.156 [0.031–0.609] |
| `normal_pressure_recompute` | 104.186 [104.183–104.216] | 17.502 [17.471–17.504] | 8.496 [4.684–24.371] |

The footprint includes the 80 GiB holder where applicable. Across all 27
trials, observed VM counters totaled 1612 swap-in pages and 32,622 swap-out
pages (4 KiB pages), with **zero OOM kills**. Maximum combined traffic in
one trial was 40.898 MiB, below the fixed 256 MiB bound. Attribution and
latency cost of this swap activity were not isolated. All pressure locks
remained live through the measured work; no low-memory guard fired.

Generation throughput is also configuration-dependent. Reference-reported
decode tokens/s, excluding the first predicted token from its decode timer:

| Arm | B decode tokens/s | A continuation decode tokens/s |
| --- | ---: | ---: |
| `resident` | 75.33 [74.91–77.24] | 27.82 [27.12–27.88] |
| `warm_restore` | 75.45 [73.96–75.46] | 47.13 [46.51–48.07] |
| `warm_recompute` | 76.24 [74.66–78.08] | 27.64 [27.58–27.84] |
| `cold_restore` | 76.45 [76.06–76.59] | 47.07 [46.58–47.69] |
| `cold_recompute` | 76.79 [75.73–77.36] | 27.46 [27.19–27.56] |
| `pressure_restore` | 74.93 [74.04–75.82] | 46.84 [46.11–47.92] |
| `pressure_recompute` | 76.54 [76.23–77.50] | 27.21 [27.15–27.56] |
| `normal_pressure_restore` | 76.64 [74.34–77.03] | 46.10 [46.09–46.13] |
| `normal_pressure_recompute` | 75.57 [74.88–76.94] | 46.09 [45.28–46.18] |

The live/full-recompute full-SWA path is about 27–28 tokens/s here, while
the restored path and default-SWA paths are about 46–47. Their outputs
match, but their runtime state histories differ. The cause of the decode
difference was not isolated. Do not attribute it to llmpalooza paging or use
only the slower reference path when later setting generation-stall goals.
No inter-token tail-latency histogram was collected in this experiment.

The default-SWA restore attempt restored the reported 18,303 tokens but
then processed all 18,339 input tokens (RE-004). It saved/restored bytes
without recovering prefix reuse. The default recompute arm avoids that
unproductive work and returned in 23.377 s median under the same forced
one-model policy. Its smaller memory footprint is reported above;
unconstrained normal placement is not established by these rows.

A separate, fixed post-main recall probe requested the service, shard and
timeout for records 0063 and 0612. Both the resident and restored paths
correctly returned `svc-12 / 8 / 83 ms` and `svc-0 / 7 / 72 ms`.
They reused 18,297 tokens, processed 69 and produced **the same 45 token
IDs**. The prompt/records were fixed before execution; this was one
no-pressure recall check, excluded from every timing table. Raw evidence
is in `aba/recall-1` on the target. Together with the recurrent-state B
probe, this strengthens bounded continuation evidence; it is not an
exhaustive KV integrity or numerical-logit equivalence proof.

## Model and state accounting

A's verified expert layout is in the [setup report](../reference-setup/README.md):
30 layers, 128 routed experts per layer, top-8, one shared FFN; a selected
expert closure is 3,717,124 bytes in layers 0–28 and 4,336,644 in layer 29.
Its file is 16,947,541,728 bytes. At context 32,768, full-SWA f16 allocation
is 7040 MiB. This is an allocated capacity, not the serialized state size.
Default SWA at the same context allocates 940 MiB (640 global + 300 SWA)
and 173.52 MiB CUDA compute scratch, versus the matched 204.02 MiB. Its
RAM prompt cache limit remains 8192 MiB and its context checkpoints remain
enabled, with maximum 32 and minimum spacing 8192 tokens. Those are limits
and settings, not claims that every allowed byte is occupied.
The long conversation's save is **584,866,556 bytes** for 18,303 tokens.
Serialization retains the required SWA window plus global history; it does
not serialize the entire full-SWA allocation. Global KV scales at 20 KiB
per token, while SWA contributes a bounded retained window. The earlier
short-prompt bytes/token ratio must not be extrapolated to long conversations.

Ornith's hash-verified GGUF metadata and tensor accounting, inspected with
the same pinned reader:

| Property | Value |
| --- | ---: |
| File bytes | 21,713,463,040 |
| Tensors / elements | 753 / 35,505,251,456 |
| Tensor payload bytes | 21,702,472,192 |
| Routed expert payload, including stored MTP layer | 20,025,704,448 |
| Other tensor payload | 1,676,767,744 |
| Shared FFN projection bytes, subset of other tensors | 78,225,408 |
| Stored MTP layer tensor payload | 546,703,360 |
| Primary model tensor payload, excluding MTP | 21,155,768,832 |
| Primary routed expert bytes | 19,503,513,600 |

The artifact has 40 primary layers plus one stored next-token/MTP layer,
256 routed experts per layer, top-8, and a shared FFN of width 512. Routed
expert width is also 512. Each closure has separate gate/up/down projections:
Q4_K gate and up use 589,824 bytes each; down uses 589,824 bytes (Q4_K) or
860,160 (Q6_K). Across the 40 primary layers there are 20 of each closure
size: **1,769,472 or 2,039,808 bytes per expert**. Shared gating/router weights
remain in the non-routed partition. These are tensor bytes, not future
2 MiB prepared extents or measured paging reads. No MTP execution is enabled.

B combines full attention every fourth primary layer with recurrent state.
At context 8192 its f16 KV is 160 MiB, or 20 KiB per token, plus a fixed
62.8125 MiB recurrent allocation (2.8125 MiB convolution state and 60 MiB
recurrent matrices). Its saved 94-token state occupies **67,792,456 bytes**.
The separate save/unload/reload probe reused 94 tokens, processed eight new
tokens and matched all seven continuation token IDs. A's analogous probe
reused 18,297, processed 42 and matched all 118 output IDs.

The matched restore arms retain both outgoing state files: **652,659,012
logical bytes**, 652,664,832 allocated bytes. This is state retained after
switching, separate from whole-node physical writes and reference memory.

## Reproduce

Use the container/platform and license records in the
[reference setup report](../reference-setup/README.md), plus this directory's
`artifacts.json`. Ornith is MIT per its pinned card and owner confirmation;
both models are external benchmark inputs, not incorporated implementation.
The helper and harness are original Apache-2.0 experiment tooling. Python
orchestrates native reference processes and never executes in inference.

Build the pressure helper on the x86-64 workstation with D-032's verified
Clang 22.1.8, LLD and AArch64 sysroot from the
[toolchain smoke](../toolchain-smoke/README.md):

```sh
export LD_LIBRARY_PATH=/tmp/llmp-clang22/sdk-amd64/usr/lib/x86_64-linux-gnu:/tmp/llmp-clang22/sdk-amd64/usr/lib/llvm-22/lib
export CXX=/tmp/llmp-clang22/sdk-amd64/usr/lib/llvm-22/bin/clang++
export LLD=/tmp/llmp-clang22/sdk-amd64/usr/lib/llvm-22/bin/ld.lld
export SYSROOT=/tmp/llmp-toolchain-smoke/sysroot
bash docs/experiments/reference-aba/build-ballast.sh /tmp/llmp-aba/ballast
```

Copy this directory and the verified helper to an external tool directory
on the selected Linux target. Fetch each model from its pinned Hugging Face
`resolve/<revision>/<filename>` URL in `artifacts.json`; verify size and
SHA-256 before promoting any partial download. Existing verified files are
reused. Public downloads needed no Hugging Face API key.

The recorded target root is `/home/pmeenan/.local/share/llmp/aba`, with
models in `/home/pmeenan/.local/share/llmp/reference-models`. Frozen input
is available at `aba/final-trace.json` on that host. Copy and verify it
for exact replay. If unavailable, `prepare` deterministically constructs
synthetic inputs and checks A/B save/load against resident continuations,
then emits a candidate frozen trace; it must match the recorded hash before
these results can be called a repeat. A different hash requires a newly
identified experiment, not changing the pin to make validation pass.

From the copied tool directory on the target:

```sh
umask 077
export DOCKER='sudo -n docker'
# Optional: regenerate the trace in a NEW external directory.
python3 experiment.py prepare --models "$HOME/.local/share/llmp/reference-models" \
  --output "$HOME/.local/share/llmp/aba/new-prepare" --device nvme0n1
# Replay the verified frozen trace; output must not already exist.
python3 experiment.py run --models "$HOME/.local/share/llmp/reference-models" \
  --output "$HOME/.local/share/llmp/aba/new-repeats" --device nvme0n1 \
  --trace "$HOME/.local/share/llmp/aba/final-trace.json" \
  --ballast "$HOME/.local/share/llmp/aba/tools/ballast" --repeats 3
# Validate all cases/receipts/cache conditions and emit aggregate medians/ranges.
PYTHONDONTWRITEBYTECODE=1 python3 summarize.py \
  "$HOME/.local/share/llmp/aba/new-repeats" /tmp/aba-aggregate.json
# Run separately, after timing finishes: fixed early/late notebook recall.
python3 recall.py --models "$HOME/.local/share/llmp/reference-models" \
  --output "$HOME/.local/share/llmp/aba/new-recall" --device nvme0n1 \
  --trace "$HOME/.local/share/llmp/aba/final-trace.json"
```

For Ornith accounting, run `inspect_ornith.py` inside the pinned image with
`PYTHONPATH=/app/gguf-py`, read-only mounts for the verified GGUF and inspector,
no network and the invoking UID, following the setup report's inspection
command. Keep its JSON outside Git. Do this before or after the benchmark;
even a metadata reader can prevent complete cold-cache eviction while it
holds a mapping.

Verify the supplied block device actually backs the model/state filesystem.
Do not run other inference, model downloads or storage benchmarks concurrently.
Each trial has a new private output directory and uniquely named container;
the API is authenticated and published only on host loopback. The container
uses read-only models/root, drops capabilities, and runs as the invoking UID.
Normal exceptions, Ctrl-C and SIGTERM unwind cleanup; a forced process/host
kill may require removing the name in `container.json`. Output contains only
synthetic prompts, tokens and state and has explicit manual retention: remove
the external result directory once no longer needed. No serving daemon or
scheduled retention task is installed.

## Verification and handoff

Workstation checks cover streaming first-token detection, failed/truncated
streams, durable file/directory synchronization, restore identity and actual
reuse, canonical template forwarding and bounded tail rewrites, pressure
identity/liveness/lock loss, low-memory release, cleanup failure receipts,
swap limits, and aggregate/model-accounting failure cases. Run:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s docs/experiments/reference-aba -p 'test_*.py' -v
git diff --check
```

The pressure helper cross-build reproduced its pinned binary SHA-256. Its
1 GiB Spark smoke verified RSS, locked backing, clean release on stdin EOF,
and no persistent host setting changes. The selected D-032 compiler/sysroot
and their dependency categories remain those in the toolchain smoke report.
The Ornith inspector ran inside the pinned ARM64 reference container and
validated every primary and stored MTP expert closure. No llmpalooza application
tests exist yet; inference and memory-pressure measurements run on `spark`
only, not the workstation or `spark-b`.

All 32 workstation regression tests passed, as did Python/JSON/shell syntax
checks and `git diff --check`. The complete-run validator independently
checked all 27 receipts, their configuration/identity/budget, all incoming
weight-cache conditions, and warm/cold saved-state conditions before producing
the tables. The separate recall check passed. No experiment container or
pressure process remains on the target; host configuration is unchanged.
Reusable harnesses, pins and this aggregate report are in the working tree;
captured inputs, model files, state, telemetry and raw output remain external.
All changes remain uncommitted for human review. Expert-route capture and the
paging-feasibility experiment are the next plan work; this does not establish
larger-model combinations, two-node execution or final generation-stall limits.

Independent review (2026-09-21): a separate agent reviewed the entire
uncommitted setup/experiment unit and the affected status, architecture, plan
and RE-004 documentation. Earlier measurement, cleanup, template-reuse and
recall-scoring findings were fixed and covered by the 32 passing workstation
regression tests. Pinned upstream routing/state behavior, model provenance
and license categories, tensor accounting and the physical non-fit arithmetic
were checked. All 27 raw receipts and reference logs reconciled with the
complete aggregate and every reported table; 19,649 target memory samples
reproduced the reported summaries. Actual sampling gaps had a 50.534 ms median
and 108.341 ms maximum, reinforcing the sampled-maximum qualification.
Effective child/container settings, exact continuation IDs, cache conditions,
swap counters and the separate recall result were verified. Read-only target
checks found no remaining experiment containers or pressure helpers. No
additional GPU experiments were run by the reviewer. `git diff --check`
passed; no outstanding correctness finding remains within the stated scope.
Normal concurrent placement, the decode-speed cause, numerical KV/logit
equivalence and paging feasibility remain explicitly unproven. Changes remain
uncommitted for the human commit gate.
