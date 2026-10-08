<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3 baselines — 2026-09-28

The reference engines llmpalooza's M3 swap, speed, memory and correctness are
judged against ([plan.md M3](../../plan.md#m3--single-spark-fast-full-swap--in-progress)),
installed and run by us on `spark`. Coarse by design (D-085): a few
repeats, medians, and the only question asked of a difference is whether it
exceeds about 10%. Every number here is **measured** on `spark` in this
session unless marked **creator-reported** (from the recipe's or engine's
own documentation) or **computed**.

**Headline (measured, `spark`, cold page cache; TensorFold 0.3.6.2 on `spark-b`):**

| Model | Engine and format | Load to first token | Prefill at 8K, tok/s | Decode, tok/s: spec off / on | Peak memory (`MemAvailable` drop) |
| --- | --- | ---: | ---: | --- | ---: |
| DeepSeek V4 Flash 0731 | llama.cpp, UD-Q2_K_XL (oracle and comparator) | 104.0 s (92.3 s with the drafter) | 352 | 19.9 / 30.8–31.9 (DSpark) | 93.4 GiB (104.7 with the drafter) |
| Qwen3.8 Flash Next | Mia's vLLM, NVFP4 (oracle and comparator) | 13 min 13 s (MTP 3, one cold start) | 2,066 | 25.1–25.3 / 37.9 (MTP 3) | 103.4 GiB |
| Qwen3.8 Flash Next | TensorFold 0.3.6.2 (`71377a53`), MLX 4-bit (cross-quantization); measured on `spark-b` | 143.0 s | 2,323 | 37.5 / 55.4–56.1 (MTP) | 88.9 GiB |
| Qwen3.8 Flash Next | TensorFold 0.3.5.1 (`beddbb7b`, the first pin; history) | 140.6 s | 2,282 | 36.9–37.6 / 55.1–55.2 (MTP) | 85.7 GiB |
| Qwen3.8 Flash Next | llama.cpp, UD-IQ3_XXS (cross-quantization) | 68.6 s | 616 | 30.4 / not run | 79.9 GiB |
| Qwen-Image-2.1 | diffusers, BF16 | 212.2 s to the first step's output | 1.259 s per step at 1024², 40 steps | 52.5–52.7 s per full generation | 43.4 GiB |

The swap to beat in llama.cpp (one A→B→A cycle, DeepSeek 0731 holding an
8,225-token saved conversation, Qwen3.8's UD-IQ3_XXS GGUF as B): **76.6 s**
A→B and **104.4 s** B→A to the first token, the return restoring A's state
with 11 prompt tokens processed. M0's run of the older DeepSeek revision
measured 75–93 s per switch.

**Llmpalooza's swaps beside these** (measured on `spark-b`, 2026-09-28, one
process per ordered pair, from the swap request to the first token or
the image's first denoising step; [swap.md](swap.md#results-m3s-swap-pairs-spark-b-2026-09-28)):

| Swap | llmpalooza | Baseline |
| --- | ---: | ---: |
| DeepSeek 0731 (8K context) → Qwen3.8 | 7.7–8.8 s | 76.6 s (llama.cpp, Qwen3.8's UD-IQ3_XXS GGUF: cross-quantization, speed only) |
| Qwen3.8 → DeepSeek 0731, A's 8K state restored | 8.8–9.0 s | 104.4 s (llama.cpp, the same) |
| Qwen3.8 (NVFP4) to its first token | 6.2–8.8 s from another model | 13 min 13 s (Mia's vLLM, from start); 143 s (TensorFold 0.3.6.2 on `spark-b`, MLX 4-bit, cross-quantization; 141 s at 0.3.5.1) |
| Qwen-Image-2.1 to its first step's output | 5.0–6.3 s from an LLM | 212.2 s (diffusers BF16, from process start) |
| Worst LLM↔LLM swap | 9.38 s | — |

Not like for like: the baselines start processes and load from a cold
page cache; llmpalooza swaps in one running process with direct reads (no
page cache), the LLMs' artifacts at rest (3–5 h old) and the image's
recently written (2 h, RE-027's faster rate). Swaps into an LLM are
page-in bound, so the margin under 10 s is the SSD's at-rest rate for
75–97 GB (11.5–13.4 GB/s in these runs).

Correctness reference data, saved beside this file: greedy tokens and top-5
log-probabilities for the six chat prompts, 32 tokens each, from
[Mia's vLLM](reference-qwen3.8-nvfp4-vllm.json) (deterministic mode, MTP
off) and [llama.cpp on the 0731 GGUF](reference-deepseek-v4-flash-0731-llamacpp.json);
the diffusers reference image stays outside Git on both Sparks, identified
by its hashes ([below](#qwen-image-21-diffusers-bf16)). Mia's
default launch (MTP 3, its deterministic mode off) is **not repeatable**:
the same greedy request gave different tokens within 3–29 tokens on five of
the six prompts, so the Qwen3.8 oracle is its deterministic mode
([below](#greedy-reference-data)).

## Default Mia launcher for new runs (2026-10-03)

Use [mia-launch.py](mia-launch.py) with [mia-launch.json](mia-launch.json),
including for correctness and acceptance studies. The owner selected the
measured fast-start recipe on 2026-10-02 and reaffirmed it on 2026-10-03.
The tables above retain their original, unpatched startup measurements;
those launches are historical evidence, not the default for new work.

The launcher pins the original recipe and image, the complete installed
`instanttensor` 0.2.0 payload, and the PLE loader patch. It selects
`--load-format instanttensor`, V2 and the persistent Triton cache.
**The PLE CPU worker must still use safetensors:** its independent checkpoint
pass must not stage the full model through the GPU beside the main load.
The [pinned patch](mia-ple-loader.patch) supplies that override. The owner's
[startup study](../../upstream/vllm.md#start-in-25-min-instead-of-11-load-with-instanttensor-and-persist-tritons-cache)
measured 151–160 s to readiness after the first launch, 211 s on the first.
These are that study's numbers, not a new startup measurement.

Both Sparks now retain the recipe and payload at
`~/.local/share/mia-load-study/{mia-perf,pylib}/`. To prepare another
checkout, use recipe `b8439110eec0230facbe4ddf0dffe01b8f769be0` and apply
`mia-ple-loader.patch` with `git apply`; copy the original installed payload
from a Spark with `rsync -rlpc`. The complete file inventory in the JSON pin
must match; a different wheel or installation needs its own qualified pin.
The loader is an external reference dependency, not part of llmpalooza's runtime.
Keep each experiment's original `.env` and explicit inference settings.

Inside an installed `spark-job --gpu` job, launch with:

```sh
python3 -B /ABS/REPO/docs/experiments/fast-swap/mia-launch.py \
  --recipe /home/pmeenan/.local/share/mia-load-study/mia-perf \
  --pylib /home/pmeenan/.local/share/mia-load-study/pylib
```

Set the experiment's owned container name, port, model/cache/head/depth,
graph and sampling overrides before this command. Keep the existing
supervised readiness and owned-container retirement controller around it;
the launcher replaces only its `start.sh` invocation. Record the actual
container arguments/environment, mounted loader hashes and generated overlay
hashes, including its PLE worker, at readiness and after collection.
Do not infer unchanged logits or throughput merely from faster loading.

## Conditions

| Item | Value |
| --- | --- |
| Host | `spark` (`spark-c4e2`), NVIDIA GB10, driver 580.178.04, kernel 7.0.0-1019-nvidia, Ubuntu 24.04.5; 121.69 GiB visible memory; ext4 on the internal NVMe |
| Isolation | Nothing else on the GPU during a timed run (checked with `nvidia-smi` and `docker ps` before each); nothing ran on `spark-b` |
| Prompts | [prompts.json](prompts.json) (`fast-swap-prompts-v1`), fixed before the first run; the filler text is `baseline.py`'s `filler()` |
| Harness | [baseline.py](baseline.py) (sessions and the llama.cpp swap), [summarize.py](summarize.py) (the JSON beside this file); raw logs, server logs and per-request records on `spark` under `~/.local/share/llmp/baselines-20260928/` |
| Cold load | Before each timed start the model's files (and Mia's packed PLE table) were dropped from the page cache with `POSIX_FADV_DONTNEED`; the clock starts before `docker run` (llama.cpp) or `start.sh` (Mia) |

File age (RE-027: the SSD reads recently written data about 11% faster):

| Files | Written (birth time) | Age at the runs |
| --- | --- | --- |
| DeepSeek V4 Flash 0731 UD-Q2_K_XL and DSpark Q8_0 | 2026-09-28 00:31–01:08 EDT | about 1 hour: **recent** |
| Qwen3.8 NVFP4 (Mia) | 2026-09-28 00:34–01:14 EDT; PLE table built 02:11 | about 1–2 hours; PLE table minutes: **recent** |
| Qwen3.8 UD-IQ3_XXS GGUF (M0's, the swap's B) | 2026-09-21 22:26–22:37 EDT | 6 days: **at rest** |
| TensorFold's MLX 4-bit checkpoint | 2026-09-28 01:16–01:32 EDT on `spark`; 16:49–17:04 EDT on `spark-b` (downloaded there) | about 1.5 hours on `spark`, 7–37 minutes on `spark-b`: **recent** |
| Qwen-Image-2.1 BF16 | 2026-09-28 00:30 EDT (copied into the store) | about 2.5 hours: **recent** |

So every load here except the swap's B read recently written files, the
fast case of RE-027. None of these engines reads at SSD speed, though
(below), so the effect on their load times is small.

## Methods

- **Load:** process start to `/health` returning 200 (*ready*), and to the
  first streamed piece of the fixed short chat request sent right after
  (*first token*).
- **Prefill:** a chat request whose user message is the synthetic service
  log fitted to 512, 2,048 and 8,192 tokens with the engine's tokenizer,
  `max_tokens` 1, a unique first line per request (no prefix-cache hits;
  llama.cpp also `cache_prompt: false`). Rate = prompt tokens (the
  engine's `usage`) ÷ request wall time, so one decode step and the HTTP
  round trip are included. llama.cpp's own `prompt_per_second` agrees within
  1% and is in the JSON. Three repeats per length; the median is reported.
- **Decode:** the two fixed chat prompts (`prose`, `code`), greedy,
  256 tokens with `ignore_eos`. Rate = (completion tokens − 1) ÷ (last
  streamed piece − first streamed piece). Three repeats per prompt. Draft
  acceptance is accepted ÷ drafted tokens from llama.cpp's `timings` or
  vLLM's `/metrics` counters.
- **Peak memory:** `MemAvailable` sampled every 200 ms; peak = the value
  just before start minus the minimum over the whole session (load and all
  measurements). It includes the engine's host-side processes and anything
  the page cache could not give back; it is a sampled minimum, not an
  exact peak.
- **Speculation:** reported off and on separately, and every row says which
  (D-087).

Runs longer than 10 minutes (D-085), each stated before it started: Mia's
cold start (13 min 11 s, the only way to get the Qwen3.8 oracle and
comparator) and a second Mia launch with MTP off in its deterministic mode
(10 min 52 s, below). Every other command finished in under 10 minutes;
the whole session took about 80 minutes of `spark` time.

## DeepSeek V4 Flash 0731: llama.cpp

Same-format correctness oracle and performance comparator. llama.cpp
`b29c606e` (build 10964), image `ghcr.io/ggml-org/llama.cpp@sha256:837fc732…`
(pins.json), unpatched, `CUDA_DISABLE_PTX_JIT=1`, as M0 ran it. Server
arguments: `-ngl all -fa on -c 16384 -np 1 --fit off -cram 0` (RAM prompt
cache off so it holds no host memory and no prompt is reused); the
drafter run adds `-md dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf --spec-type
draft-dspark --spec-draft-n-max 3 -ngld all`, the DSpark card's
recommended settings. Weights `unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93`.

| Measure | Speculation off | DSpark drafter, n-max 3 |
| --- | ---: | ---: |
| Start to ready / first token, s | 103.7 / 104.0 | 91.9 / 92.3 |
| Prefill, 541 / 2,067 / 8,213 prompt tokens, tok/s | 327 / 362 / 352 | 322 / 360 / 351 |
| Decode, `prose` / `code`, tok/s | 19.90 / 19.95 | 30.80 / 31.94 |
| Draft acceptance (accepted ÷ drafted), `prose` / `code` | — | 0.536 / 0.569 |
| Peak `MemAvailable` drop, GiB | 93.4 | 104.7 |

Medians of three; every range is within 2% of its median except the
drafter's first 512-token prefill (284 tok/s). The drafter costs no
measurable prefill here (the card reports 21–24% on B200s, creator-reported)
and gives 1.55–1.60× decode (the card: 1.91× on one B200 at acceptance 0.764,
creator-reported). Decode without it matches M0's 19.9 tok/s on the older
revision. The two loads differ by 12%, more than the drafter could explain
(it adds 10.9 GB to read); one load each, so the load time is 92–104 s.
Loading 96.8 GB in about 100 s is about 1 GB/s (computed): llama.cpp's load
is far from the SSD's 13.3 GB/s at rest, so file age hardly matters to it.
Greedy decoding with the drafter is not bit-identical to decoding without
it in this llama.cpp (the card cites llama.cpp issue #25618); it was not
checked here.

### The llama.cpp swap: DeepSeek 0731 and Qwen3.8

One cycle, as M0 ran it: one llama.cpp server per model, a swap being save
the slot, stop the container, start the other, and (on return) restore the
slot. A is DeepSeek 0731 (arguments above, no drafter) holding an 8,210-token
prompt plus a 16-token reply; B is Qwen3.8 Flash Next's
`UD-IQ3_XXS` GGUF (M0's, `unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee`, same
arguments) answering the short first-token prompt. The incoming model's
files were dropped from the page cache before each swap, outside the clock.

| Part, seconds | A→B | B→A |
| --- | ---: | ---: |
| Save the outgoing slot | 0.04 (A: 8,225 tokens, 74.6 MB) | 0.06 (B: 65 tokens, 119.8 MB) |
| Stop the outgoing server | 1.46 | 2.25 |
| Start the incoming server to ready | 74.57 | 100.93 |
| Restore A's slot | — | 0.07 (8,225 tokens) |
| Request to first token (computed: the total minus the parts above) | 0.53 | 1.13 (11 tokens processed, 8,225 reused) |
| **Swap request to first token** | **76.60** | **104.44** |

The load dominates; saving and restoring 8K tokens of DeepSeek state is
under 0.1 s each. M0's six-request run of the older revision measured
75–93 s per switch; the return here is slower because it includes
DeepSeek's whole 101 s load.

## Qwen3.8 Flash Next: Mia's vLLM (NVFP4)

Same-format correctness oracle and performance comparator.
`MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark@b8439110`, run unmodified
(owner, 2026-09-28) with its default image `vllm/vllm-openai:qwen38-flash-next`
(index `sha256:fc120ece…`, the local image ID; vLLM
`0.1.dev20073+g8e685d198`, FlashInfer 0.6.17, PyTorch 2.13.0+cu130).
`.env` is `.env.sample` unchanged plus `BIND=127.0.0.1`: 262,144 context,
MTP 3 with the 47k draft vocabulary, FP8 KV, BF16 SSM state,
`MAX_NUM_SEQS=4`, `MAX_NUM_BATCHED_TOKENS=2048`, `HOST_RESERVE_GIB=26`, V2
model runner, full decode graphs. Three things around the unmodified
scripts: the checkpoint from our store appears in a Hugging Face cache
layout built from hard links (same files, no copy), `docker` on the `PATH`
is a two-line wrapper calling `sudo -n docker`, and `./start.sh
--no-launch` ran once first, which built the 27 GiB packed PLE table in
45 s (one-time, outside the timed start, as it is for every later start).

The engine log's kernel choices, which the licensing record left for this
item: NVFP4 linear layers on `MarlinNvFp4LinearKernel` (vLLM reports the
GB10 lacks native FP4 compute and uses weight-only FP4), MXFP8 linear layers
on `FlashInferCutlassMxfp8LinearKernel` with one shape (N 96, K 2,560) on BF16
emulation, routed experts on the `FLASHINFER_CUTLASS` NVFP4 MoE backend
(the MTP drafter's on `MARLIN`), Gated DeltaNet prefill on Triton/FLA and
decode on its CUDA kernel, attention on the recipe's
`QWEN38_FLASH_NEXT_EXP_QSA_STATE` backend.

| Measure | Default launch: MTP 3 | Second launch: MTP off, deterministic mode |
| --- | ---: | ---: |
| `start.sh` to `/health` / first token | **791.5 s (13 min 11.5 s)** / 793.0 s | 652.4 s (10 min 52 s) / 654.1 s |
| Of which, from the engine log: weights / engine init (profile, KV, autotune, graphs) | 511.9 s + 70.0 s MTP / 119.3 s | 468.9 s / 107.4 s |
| Prefill, ~590 / ~2,130 / 8,266 prompt tokens, tok/s | 1,198 / 1,625 / 2,066 | 1,220 / 1,398 / 2,101 |
| Decode, `prose` / `code`, tok/s | 37.85 / 37.85 | 25.12 / 25.33 |
| Draft acceptance, `prose` / `code` | 0.423 / 0.422 | — |
| KV pool | 16.17 GiB | 19.21 GiB |
| Peak `MemAvailable` drop, GiB | 103.4 | 102.7 |

The cold start is inside the creator-reported 11–14 minutes; the second
launch matches the README's 10 min 51 s to `/health` (creator-reported).
The weights load at about 0.13 GB/s (72.8 GiB in 590 s, computed): the
recipe's load is nowhere near disk speed.

The second launch exists because vLLM has no per-request switch for
speculation, and because the default launch turned out not to be
repeatable (next section). It sets `MTP_NUM_SPECULATIVE_TOKENS=0` and the
recipe's own deterministic-mode knobs, `VLLM_QSA_DET_TOPK=1` and
`VLLM_MOE_DET_FINALIZE=1`, from the environment; the scripts are unchanged.
So its decode is "speculation off in deterministic mode"; the recipe
reports that mode's decode within noise and prefill 3.4% slower at 47K
(creator-reported), and its prefill here agrees with MTP 3's within 2% at
8K. MTP 3 decodes 1.49–1.51× faster; the recipe's sweep reports K=0 losing
32–46% against K=3 (creator-reported), and 25.2 against 37.85 is −33%.

Against the creator's numbers: prefill at 8K is 2,066 tok/s against 2,200
reported (−6%, within 10%). Decode is 37.85 tok/s against 48.7 reported for
single-stream prose (−22%). The engine step is not slower: from vLLM's
counters, `prose` accepted 1.27 draft tokens per step (per position 0.67,
0.39, 0.21), so 2.27 tokens per step and about 60 ms per step (computed),
against the README's 61.5 ms per step at 3.00 tokens per step. The gap is
acceptance on this prompt set, whose first 256 tokens are the model's
reasoning prose, under the code-tuned 47k draft vocabulary; llmpalooza will
be compared on the same prompts. The first request of each prefill length
after the cold start was slow (420 tok/s at 512, 770 at 2K; the recipe
attributes this to the PLE table's page cache warming); medians of three
are reported, and the 8K rows varied by under 3%.

## Qwen3.8 Flash Next: TensorFold (MLX 4-bit; cross-quantization)

Speed and memory only, never correctness, and never gated. The M3
baselines pinned `ashhart/TensorFold@beddbb7b` (0.3.5.1) and measured it on
`spark`; the pin then moved to main's tip `71377a53` (0.3.6.2) the same day,
re-measured on `spark-b` under the same protocol, prompts and client. Both
are below; the tip's are current. Each is installed as its runbook says
into `nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699…`
([Dockerfile.tensorfold](Dockerfile.tensorfold); local images
`sha256:305b9263…` and `sha256:1a2afff2…`), serving
`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa80` from a read-only mount
with its defaults: one CUDA rank, MTP drafts on (1 to 6 a round, a chain
stopping before a later draft under 30% confidence), thinking on, a bf16
KV cache, and a context from its startup estimate (137,206 and 150,042
tokens allocated). Speculation off is the request field `"draft": false`,
on the same warm server. CUDA Qwen engines ignore `ignore_eos`; every
decode still produced 256 tokens. The tip's launch adds a 110 GiB
container memory cap and leaves the image's own `TORCH_CUDA_ARCH_LIST`
(below).

| Measure | 0.3.6.2, drafts | 0.3.6.2, `"draft": false` | 0.3.5.1, drafts | 0.3.5.1, `"draft": false` |
| --- | ---: | ---: | ---: | ---: |
| Start to ready / first token, s (kernels cached) | 142.8 / 143.0 | (same server) | 140.3 / 140.6 | (same server) |
| First start, empty kernel cache: ready / first token, s | 224.4 / 224.8 | | 199.8 / 225.4 | |
| Prefill, 532 / 2,075 / 8,212 prompt tokens, tok/s | 1,330 / 2,095 / 2,323 | — | 1,387 / 2,095 / 2,282 | — |
| Decode, `prose` / `code`, tok/s | 55.39 / 56.05 | 37.55 / 37.53 | 55.24 / 55.13 | 36.87 / 37.61 |
| Draft acceptance, `prose` / `code` (accepted ÷ drafted, the engine's per-reply counts) | 0.461 / 0.463 | — | not recorded | — |
| Tokens per draft round, `prose` / `code` | 2.21 / 2.35 (the engine's rounds) | 1 | 2.19 / 2.33 (streamed pieces, computed) | 1 |
| Peak `MemAvailable` drop, GiB | 88.9 | | 85.7 | |

- **Old against new:** every speed is within 4% (the load +1.8%, prefill
  at 8K +1.8%, decode −0.2 to +1.8%), under D-085's ~10% question, on a
  different Spark of the same hardware, driver and kernel. The 58 commits
  change this path's defaults little: their Flash Next work is EXL3
  checkpoints, opt-in int8 and int4 KV caches, memory accounting and the
  startup prefill below. Peak memory is 3.2 GiB (4%) higher; not
  investigated.
- **The first start.** At 0.3.5.1 it failed until
  `TORCH_CUDA_ARCH_LIST=12.1` was set: the container's list (`8.0 8.6 9.0
  10.0 11.0 12.0+PTX`) made the JIT build every kernel for sm_80 too, where
  the cluster API and FP8 MMA do not exist (about 10 minutes lost). 0.3.6.1
  (`34bae79`) passes `-gencode` for the GPU present, and the tip's first
  start built in the image's default. The tip's startup also prefills a
  synthetic prompt (`416106f`) so that no request compiles a prompt kernel:
  26.4 s on the first start and 0.9 s once cached. So the first start is
  ready 24.6 s later than 0.3.5.1's, and its first token arrives at the
  same time.
- **Exactness,** TensorFold's own claim: each drafted reply's `token_sha`
  equals its `"draft": false` reply's (`prose` `71e063b2b477`, `code`
  `a3bf76a03fa6`), in all three repeats.
- **The load** is 143 s against the about 90 s creator-reported. A side
  study outside the repository traced it to small buffered reads and an
  int64 nibble shuffle in the expert repack, and two small patches cut it
  to about 45 s at 0.3.5.1. Neither is upstream at 0.3.6.2, whose reader
  and shuffle are unchanged; one phase-marked start there (unpatched,
  148.6 s) spent 132.7 s in `weights.load`, 70.9 s of it reading at
  1.13 GB/s. The owner is taking the patches upstream; the tip's patched
  start was not measured.
- **Against the other Qwen3.8 numbers:** TensorFold's speculation-off
  decode (37.5) matches Mia's MTP 3 decode (37.85), and with drafts it is
  1.46–1.48× Mia's. Llmpalooza's NVFP4 Qwen3.8 decodes at 25.8–26.1 tok/s plain
  and 42.46 / 39.09 with MTP depth 2
  ([qwen38-mtp](../qwen38-mtp/README.md#performance-and-memory)): 0.69–0.70× and
  0.77 / 0.70× TensorFold's. A different quantization, so this is
  information, not a target.
- **Conditions on `spark-b`,** 17:11–17:27 EDT: before each start the GPU
  was idle, at least 115 GiB available and no container running, twice
  30 s apart; other agents' jobs ran on `spark-b` between the runs, and one
  was waited out before the measured session. The first drafted `prose`
  repeat (50.97 tok/s) is the one outlier. The kernel cache of the
  measured start is the first start's.

## Qwen3.8 Flash Next: llama.cpp (UD-IQ3_XXS; cross-quantization)

The plan's other cross-quantization comparator, on the GGUF M0 used and
the swap's B: same llama.cpp image and arguments as DeepSeek's, no drafter,
files at rest. Start to ready 68.1 s, first token 68.6 s; prefill 501 /
611 / 616 tok/s at 590 / 2,129 / 8,267 tokens; decode 30.35 / 30.39 tok/s
(`prose` / `code`); peak drop 79.9 GiB. Speed and memory only. M0 measured
30.9 tok/s decode on the same file.

## Qwen-Image-2.1: diffusers BF16

Speed, format and correctness reference. The earlier study's container
([image-reference](../image-reference/README.md): local image
`llmp-image-reference:20260922`, `sha256:700d6668…`, diffusers `8b3c707e`,
PyTorch 2.14.0+cu130), checkpoint `Qwen/Qwen-Image-2.1@790c9263`, run with
[image_baseline.py](image_baseline.py): the pipeline in BF16 on the GPU (no
CPU offload), prompts.json's `image` entry (the teapot prompt, 1024×1024, 40
steps, seed 42, `true_cfg_scale` 1.0, prefix KV cache on). The first
request records a timestamp at every step's callback (synchronized); three
plain requests follow.

| Measure | Value |
| --- | ---: |
| Process start to pipeline on the GPU | 208.9 s (container start and imports 7.1, `from_pretrained` 1.0, `.to("cuda")` 200.7) |
| Process start to the first denoising step's output (the swap endpoint) | 212.2 s |
| Per denoising step, median [range] | 1.259 s [1.255–1.263] |
| Full generation, plain, three runs | 52.55 / 52.71 / 52.73 s |
| Full generation, with per-step synchronization | 53.93 s |
| CUDA allocator peak | 36.84 GiB |
| Peak `MemAvailable` drop | 43.4 GiB |

All four images are pixel-identical: RGB SHA-256
`7d00b052878a03cc01baccbede3c3cafeca58b8160687ff93e915614387aac8f`. The
first, the reference for llmpalooza's similarity check, is on `spark` and
`spark-b` at
`~/.local/share/llmp/references/qwen-image-2.1/teapot-1024-40steps-seed42.png`
(PNG, 1,250,893 bytes, file SHA-256
`9de442f1e714160e0475c56fc343aa326b4d0253047f761d16218224e79777a1`); it
depicts the requested red teapot. It is kept out of Git as a generated
output (the repository holds no binary files, and its header check has no
rule for images); the pipeline regenerates it pixel for pixel. The generation time matches the earlier study's 52.586 s
at the same size and steps, and so does the 200 s placement on the GPU
(about 0.16 GB/s for its 32.4 GB of parameters, computed), which dominates
the load.

## Greedy reference data

For llmpalooza's correctness check (plan.md M3: greedy tokens match the
same-format oracle with small logit differences allowed): the six chat
prompts of prompts.json, each rendered with the model's own chat template
through the engine's API (default template arguments), then 32 greedy
tokens with the top five log-probabilities at each position. The files
carry the prompt token IDs, so llmpalooza can feed the same tokens without
reproducing the template.

- [reference-deepseek-v4-flash-0731-llamacpp.json](reference-deepseek-v4-flash-0731-llamacpp.json):
  llama.cpp, speculation off. Two repeats per prompt; all six identical.
- [reference-qwen3.8-nvfp4-vllm.json](reference-qwen3.8-nvfp4-vllm.json):
  Mia's vLLM in its deterministic mode with MTP off. Two repeats per
  prompt; all six identical. The file also lists, for information, the
  default launch's tokens.
- [reference-qwen3.8-nvfp4-vllm-ppl.json](reference-qwen3.8-nvfp4-vllm-ppl.json):
  the perplexity reference, from one more cold launch in the same
  deterministic, MTP-off configuration (start to ready 677 s). The text is
  `docs/async-model.md` at `e7973e5`, the same text the DeepSeek slice used,
  tokenized by `/tokenize` without a template or special tokens: 3,558
  tokens, under the 4,096-token cut, so all of it is used. One
  `/v1/completions` request with `prompt_logprobs` 1 (4.6 s) gives each
  position's log-probability of the actual next token and the top-1 token:
  **perplexity 14.658** over positions 1..3,557 (mean NLL 2.6850 nats); the
  top-1 token is the actual next token at 43.9% of positions. Made by
  [ppl.py](ppl.py).

**Mia's default launch is not a usable oracle.** With MTP 3 and the
deterministic knobs off, the same greedy request sent twice gave different
tokens on five of the six prompts, first at token 3, 7, 15, 21 and 29; on
five prompts at least one of its two repeats also differs from the
deterministic launch, first within 2–21 tokens. These are not near-ties: on `capital` the first run's
chosen token had log-probability −0.35 against −1.22 for the one the repeat
chose, so the logits themselves moved by about a nat. The recipe documents
two sources, the QSA top-k's order (vllm#55122) and the MoE finalize
(vllm#54948); MTP's verify batches may add more. The cause was not isolated.
So llmpalooza's Qwen3.8 correctness is checked against the deterministic mode,
and "greedy with speculation equals greedy without" cannot use Mia's MTP 3
run as its control.

## Not run, and why

- **vLLM or SGLang for DeepSeek V4 0731: not available.** The vLLM image on
  the Sparks has no DeepSeek V4 model and no GGUF loader. The SGLang image
  (MiMo's, `lmsysorg/sglang@sha256:9e1fb4c3…`, SGLang `0f6761b5`) has
  `deepseek_v4` and an SSD "expert-pack" loader that reads a DeepSeek V4
  GGUF's non-expert tensors (Q8_0) beside MXFP4 expert packs, which need a
  preparation step and a GGUF whose experts are MXFP4, not UD-Q2_K_XL's
  IQ2/IQ3 types. The native checkpoint is not on the Sparks and, at about
  160 GB (estimated from unsloth's 162 GB UD-Q8_K_XL), does not fit one
  Spark. A lead for M4's two-Spark comparators, not an M3 baseline.
- **stable-diffusion.cpp's GGUFs** (the plan's extra image quality and
  format comparison): not part of this run; still open.
- **A swap in Mia's vLLM or TensorFold:** neither swaps models in a running
  process; their swap is a stop plus the load above. **An LLM↔image swap
  in a reference:** diffusers and llama.cpp run in separate processes; the
  sum of their loads above is the comparison.
- **llama.cpp's Qwen3.8 GGUF with a drafter:** not run. TensorFold's draft
  acceptance was not recorded at 0.3.5.1; at 0.3.6.2 it is, from the
  engine's per-reply counts (above).

## Reproduce

On a Spark, with this directory copied to `tools/` in a new external
directory, the checkpoints in the M3 model store
([environment.md](../../environment.md#m3-model-store-2026-09-28)) and
`DOCKER='sudo -n docker'` (the default):

```sh
M=$HOME/.local/share/llmp/models
python3 tools/baseline.py session raw/ds-off \
  --llama-model $M/unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  --llama-args "-ngl all -fa on -c 16384 -np 1 --fit off -cram 0" \
  --evict $M/unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93/UD-Q2_K_XL \
  --extra '{"cache_prompt": false}' --reference llama
# Mia: HF_HOME holds hard links to the checkpoint in a Hugging Face cache
# layout, PATH starts with a directory whose `docker` runs `sudo -n docker`.
python3 tools/baseline.py session raw/mia --port 8888 --model qwen3.8-flash-next \
  --start "cd src/mia && exec ./start.sh" --stop "cd src/mia && ./stop.sh" \
  --evict $M/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6 ~/.cache/vllm/ple_cache \
  --tokenizer vllm --reference vllm --ready-timeout 2400
python3 tools/baseline.py swap raw/swap --a DEEPSEEK_SHARD_1 --b QWEN_GGUF_SHARD_1 \
  --a-args "-ngl all -fa on -c 16384 -np 1 --fit off -cram 0" --b-args "(same)"
python3 tools/summarize.py results.json LABEL=raw/DIR ...
```

TensorFold's and diffusers' exact `docker run` lines are in the raw logs
(`logs/tf-run-cmd.txt`; for 0.3.6.2, `tf-run-cmd.txt` on `spark-b` under
`~/.local/share/llmp/baselines-20260928-tensorfold-71377a53/`, run as
`baseline.py session --port 8090 --model tf --start "$(cat tf-run-cmd.txt)"
--stop "sudo -n docker rm -f tfbase" --evict CHECKPOINT --tokenizer usage
--variant mtp={} --variant 'spec_off={"draft": false}'`, after a first
start with `--no-measure` and an empty kernel cache; and the image run
mounts the checkpoint at `/model`, this directory at `/tools` and an
output directory at `/out`, then runs `/tools/image_baseline.py /model
/out/run1 /tools/prompts.json 3` with `PROCESS_START` set just before
`docker run`). Mia's packed PLE table stays in `~/.cache/vllm/ple_cache/`
on `spark` (27 GiB, root-owned, written by the recipe's container).
