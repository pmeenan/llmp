<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The swap path: the M3 models on the paged node, full swaps A→B→A (M3)

M3's swap path and swap runner ([plan](../../plan.md#m3--single-spark-fast-full-swap-in-progress);
since D-096 the runtime's own, [through llmp-runtime](#through-llmp-runtime-d-096)):
DeepSeek V4 Flash 0731, Qwen3.8 Flash Next and the Qwen-Image-2.1
pipeline run as device jobs over leased closures on the paged node
(D-086), paged into device VMM through the landing zone (D-081); a full
swap evicts the outgoing model, spilling its conversation state, and hands
its backing to the incoming one (D-033); the zone's copies have a lane of
their own (RE-029). The swap runners drive A→B→A in one process and time
each part: [M3's swap pairs](#m3s-swap-pairs) between the three models
(`llmp_swap_pairs`), and [DeepSeek with the FP16 stand-in](#what-was-built-deepseek-and-the-fp16-stand-in)
(`llmp_swap_runner`), where the path was first built.

## M3's swap pairs

### Qwen3.8 Flash Next on the paged node

`src/engine/qwen38_runner.h` (in `benchmarks/` until D-096, as the other
runners and helpers named here): the resident harness's graph, plan and
kernels (`qwen38_common.h`, one planning path that `qwen38_exec.cc` now
calls too; since the prefill slice, the fused graph and, from the
CUTLASS-layout artifact, CUTLASS's grouped GEMM and llmpalooza's vector
products over the slots as they land) over catalog extents, in DeepSeek's
layout, now one helper (`paged_weights.h`):
- **Dense groups:** every group but the n-gram table's, a 2 MiB-aligned
  region each, a chunk an extent, landed from its shard.
- **Expert slabs:** each layer's 512 experts at the resident layout's
  stride (2,768,976 bytes), an extent a 2 MiB page of the slab landed in
  pieces (`LayOutSlab`). The gap between groups is 80 bytes, too small to
  hold DeepSeek's 256-byte alignment of the slab where a layer's experts
  change shard, so the slab's offset in its first page is a multiple of
  16, the stride's own alignment (`LayOutSlab` takes it as a parameter
  now; the resident harness's odd experts are 16-aligned too).
  35,873 extents, 75,235,266,560 bytes read per load (the table's
  28,800,138,240 not among them). The CUTLASS-layout artifact's groups
  are 2,764,800 bytes with no padding, so its stride is that and a load
  reads 75,002,167,296 bytes; nothing is rewritten after page-in.
- **The state** (`Qwen38StateLayout`: the QSA layers' K, V and indexer
  caches, the linear-attention layers' recurrent and convolution state,
  the n-gram layer's convolution history): kPreserve live state with a
  write-back place in an unnamed direct-I/O spill file, first in the
  closure, as DeepSeek's. 184 extents at the runs' 8,704-token context.
- **Setup after each full load:** the n-gram hash's constants read back
  and checked (`CheckQwen38PleHash`), as DeepSeek's hash-routing tables
  are.
- **A chunk:** its host-built inputs over the whole history (the n-gram
  hash reads each token's predecessors), its n-gram rows read (below),
  the graph planned for its shape, and one job that copies the inputs,
  gathers the rows, runs the bound plan and copies the last row's logits
  out; BP-A1's check on the first chunk of each shape.

### The n-gram table by rows (D-035)

`src/engine/ple_rows.h`. The table (28.8 GB of 90-byte NVFP4 rows) is
never resident. Before each chunk's job the runner computes the chunk's
rows (16 a token, `Qwen38Chunk`'s hash), deduplicates them, reads them from
the artifact on a ring of its own (32 in flight) into a pinned landing,
and the job gathers each into a row slot on the device with a small kernel
(the GPU copies; no CPU payload copy). The graph is built over a copy of
the binding whose table has the slots' rows (512 × 16 = 8,192 slots,
737,280 bytes), and the chunk's row indices are the slots'.
- **Granularity: 4 KiB-aligned direct reads, smaller than a chunk.** A
  row's 90 bytes are covered by the one or two 4 KiB blocks around them;
  rows whose blocks touch or overlap share a read, up to 64 KiB. Every read
  lies within the table group's stored range, inside one chunk or across
  two consecutive chunks of the table's one shard (checked at setup). This
  is the only path that reads below a chunk; artifact-format.md keeps
  row-granular reads out of the runtime reader's scope and now records
  this runner's path and why the format needs no change
  ([page-in contract](../../artifact-format.md#page-in-contract)).
- **Why not whole chunks (D-035's default):** a token's 16 rows are
  hashed across the table, so almost every lookup lands in a different
  2 MiB chunk. Over the six chat prompts (192 chunks, 9,184 lookups, 8,880
  distinct rows a chunk summed), the row reads took 8,880 requests and
  37.4 MB; whole chunks would have read 18.2 GB (486×), and the useful
  bytes were 0.80 MB. An 8,192-token context touches nearly every chunk
  of the table.
- **Validity and accounting:** the rows belong to the chunk that read
  them. The landing (64 MiB, the bound for 8,192 lookups of two blocks)
  and the slots are fixed, cataloged and charged whole (staging and
  scratch); nothing carries from one chunk to the next, so no row has a
  residency to track, evict or restore across a swap. Every row index is
  checked against the table before a read is planned (whatever the hash
  gave), every read against the group's stored range, and the plan is
  refused, never split, past the landing's bytes or the slots. The rows
  are read to completion before the job that gathers them is posted: a
  short or failed read refuses the chunk after the rest drain, an unknown
  submission is in flight and waited for (the storage lane's rules), and
  reads that stall stop the rows for good, their ring and landing never
  reused or freed under them. The ring reads only the table's group, which
  is no extent, into the cataloged landing.

### Qwen-Image-2.1 on the paged node

`src/engine/qwen_image_runner.h`: the three component artifacts, joined by
their composition (D-089), each a set of extents (`paged_weights.h`,
only the groups its phase reads: the text encoder's table and language
layers, 15.14 GB; the denoiser, 14.23 GB; the VAE's decoder, 1.35 GB of
F32), with the resident harness's kernels in its call order (copied from
`qwen_image_exec.cc`, whose comparison with diffusers then needs no rerun;
since the image speed slice both run one pipeline through the same bound
plan, `kernels/image/pipeline.h`, and later steps replay a captured graph).
- **Phases run over only their component:** encode (one job over the text
  encoder's closure), denoise (a job per step over the denoiser's; the
  first also projects the text rows and fills the prefix K/V cache),
  decode (one job over the VAE's), each with the image's own memory, the
  shared workspace, the cuBLAS workspace and the staging. A generation is
  one request, which leases all three components once (below); outside a
  request each job leases its own component. The VAE's F32
  weights are paged as stored and rounded to BF16 by the decode job into
  its workspace (the resident harness rounds them on the host, as
  diffusers' `torch_dtype=bfloat16` load does; same rounding).
- **The image's own memory** (16.7 MB, mapped at setup): what lives from
  one job to the next within a generation (the prompt embeddings, the text
  rows, the prefix cache, the rotary tables, the latents and the noise
  prediction). Nothing outlives a generation, so a swap has no image state
  to spill; every per-job buffer (3.02 GB at most, the decoder's) is the
  node's shared workspace.
- **Endpoint:** the prompt encoded and the first denoising step's output
  produced (M3's image endpoint). The rest of the generation and the
  decoder follow when the image is A.
- **After the image speed slice** (`spark`, 2026-09-28, the host otherwise
  idle, Qwen3.8 A with 8,192 context tokens, the image B, the page cache
  dropped before each, one run each of the M3 slice's build and this one,
  back to back): the swap into the image's first output took 5.28 / 5.21 s
  before (first use / prepared) and 5.11 / 5.09 s after, the encode and
  first step 1.60 / 1.58 s before and 1.52 / 1.52 s after; eviction and
  the 30.72 GB page-in (13.2–13.3 GB/s) unchanged. The paged image's
  pixels equal the resident harness's (`3b7770ca…`), its whole generation
  33.0 s (`spark-b`)
  ([qwen-image-native](../qwen-image-native/README.md#speed)).

### The swap pairs runner

`benchmarks/swap_pairs.cc` (`llmp_swap_pairs`, a harness binary; the
runtime's `swap-table` runs the same protocol over every configured model
in one process, [below](#through-llmp-runtime-d-096)): two of the three models on one node, one process per
ordered pair, A→B→A as `llmp_swap_runner` does it (see its header for
the protocol): a control, a first-use cycle (B never ran in the process;
A's plans dropped before it returns), a prepared cycle, and for an LLM A a
0-context pair. An LLM A holds 8,192 tokens of `docs/decisions.md` at
`4655685` (SHA-256 `6b159ff2…`), each model's own tokenization, context
8,704; an LLM B answers the first prompt of its correctness set from a
cleared state (DeepSeek: `capital`, 6 tokens, BOS first, no template;
Qwen3.8: `capital`, 64 tokens, the chat template rendered); the image is
the teapot prompt at 1,024², 40 steps, from diffusers' seed-42 latents.
The budget is the fixed memory plus the larger model's weights: the two
never fit together.
- **Swap correctness for an LLM A** is checked against the same state,
  not a rerun: after each cycle's prefill the state is saved to the host
  and hashed, the unswapped continuation run from it (the reference) and
  the state put back; after the swap back the restored state must hash
  the same (outside the timed parts) and the continuation's every logit
  equal the reference's. Qwen3.8 past 2,051 attended cells was not
  repeatable in these runs (RE-031: GGML's radix top-k picks among tied
  indexer scores nondeterministically; since the second prefill pass the
  default graph selects with ties to the lower cell up to 32,768 cells,
  [qwen38-native](../qwen38-native/README.md#prefill-second-pass-speed-before-bit-exactness)),
  so a rerun of the prefill is no reference for it;
  each cycle's prefill is still compared with the control's, and noted.
  The substitution is sound one way only: the state digest is exact, and
  a continuation equal to the reference's shows the weights came back
  whole (wrong weights cannot give equal logits), but each continued step
  attends past 2,051 cells too, so RE-031 could make a continuation
  differ with nothing wrong in the swap. None did (below); a difference
  would need a rerun of the same state to tell the two apart.
- **An image A** has nothing to spill: its control is one full generation,
  and after the swap back the generation runs again from the prompt, its
  pixels equal to the control's.
- **Places (D-090):** every model pins its weights' and state's places
  when it registers them, graphs or not; after each swap the incoming
  model's are checked still pinned, DeepSeek's also against the sources
  its graphs name.
- **Requests** (since the lease per request, below): each turn leases its
  model's closure once, and its chunks run under that lease: the controls,
  each cycle's prefill with its reference continuation, B's first output
  (its lease taken inside the timed first output), A's return and its
  continuation or the rest of its generation, each prompt of `--prompts`.

## A lease per request

M3's follow-up to the decode graphs ([graphs](graphs.md)): a decode step
on the paged node paid 3–4 ms of round trip, most of it the scheduler
walking and leasing the ~46,500-extent closure and releasing it again, per
step. For a full-swap model the closure is the whole model, so a request
(a prompt or turn; for the image, one generation) now leases it once
(owner, 2026-09-28).

**Design** (`scheduler.h`, `paged_programs.h` `RequestProgram`,
`paged_node.h`):
- The request's task materializes the closure and leases it
  (`TaskContext::HoldLease`): an ordinary catalog lease, all or none at
  the recorded contents, held by the task instead of by one operation.
- Each step is device work under that lease (`SubmitLaunch(held, …)`):
  no closure walked, nothing leased or released. It still takes a mailbox
  and the task's lifetime hold, and still ends on its fence; the lease
  counts the operations under it.
- The lease is released by `EndLease` or by the task finishing (the
  request's end, a failure, a cancellation), but only once the last
  operation under it has concluded with proof of no further access; a
  quarantined one keeps it for good. Release records the use and changes
  eligibility, not residency (D-007).
- While it is held no eviction of its extents can begin (the catalog
  refuses a leased extent). A swap that needs them waits for the release
  (`AwaitRelease`: the evicting task is woken then, and does not spin);
  the harness's node, asked for a swap or an eviction between a request's
  steps, ends the requests holding the outgoing extents first
  (`SwapReport::requests_ended`).
- The request's driver (the harness's thread) hands each step to the task
  through a channel and a `SignalRequest` control; the task waits for it
  with `AwaitSignal`, and a signal that comes first is kept. The control
  queue orders what the driver wrote before the signal.
- The per-step lease stays for work whose closure changes from step to
  step (M7's routed experts); a job outside a request takes it as before.
- **Does a step still need its own fence round trip?** Yes, with host
  sampling: the greedy token is the next step's input, which the host
  builds (DeepSeek's embedding rows are dequantized on the host), so the
  logits must be on the host each step. Sampling on the device would not
  remove the trip, only shrink what crosses it; host sampling stays.
- **Polling through a step (harness-polled).** The scheduler (while a
  critical operation is in flight or a request holds its lease) and the
  device lane poll for a window after their last progress before they
  sleep (RE-017). At the lanes' 200 µs a step still paid 0.26–0.46 ms of
  wakeups; the paged node then polled for 100 ms, longer than a step, and
  the round trip was 0.01 ms, at the cost of four cores spinning while a
  request stepped (the scheduler, the device lane's two threads and the
  harness's driver). The figures below marked "100 ms poll" are those.
  **Since the runtime wake** (D-094,
  [runtime-wake](../runtime-wake/README.md)) the node runs the runtime's
  own defaults: the completion lane sleeps through most of a step and
  wakes the scheduler and the submission lane just ahead of its end, the
  harness's driver waits the same way, and the round trip is about 26 µs
  on a synthetic step at 0.11–0.12 of a core; the 100 ms window remains a
  diagnostic (`NodeSettings::poll_window`, `--poll-us`). Its figures are
  [below](#with-the-runtime-wake).
- **No hold and wait.** A task holding a request's lease, or whose
  ancestor does, is refused `AwaitRelease`, so no two holders can wait for
  each other. The wait itself has no bound or priority: only the holder
  ends a lease, which is why the node ends the requests in a swap's way
  before posting it (the harness has one driver thread, so a swap is
  never asked for with a step in flight). Ending a request only releases
  its lease; the swap's own evictions write the state back (kPreserve at
  a write-back place) before any backing goes.

**Round trip and decode** (`spark-b`, GB10, driver 580.178.04,
`spark-native`, `CUDA_DISABLE_PTX_JIT=1`, the artifacts above; the memory
gate before each run; `llmp_swap_pairs --bench N`, three passes per arm
after a warm-up, two runs per poll window; raw outputs in
`~/scratch/m3lease/r1`, `r2`). Round trip = a step's wall less the
device's span of its work (CUDA events around the job):

| Model, arm (100 ms poll) | tok/s | Round trip per step | Device per step |
| --- | ---: | ---: | ---: |
| DeepSeek, lease per step, graphs | 19.71–19.92 | 1.30–1.55 ms | 48.7–49.0 ms |
| DeepSeek, lease per request, graphs | **20.34–20.46** | 0.009–0.013 ms | 48.7–49.0 ms |
| DeepSeek, lease per request, launch by launch | 19.39–19.44 | 0.009–0.010 ms | 51.3–51.4 ms |
| Qwen3.8, lease per step (128 steps) | 23.10 | 1.16 ms | 41.1 ms |
| Qwen3.8, lease per request | **23.70** | 0.010 ms | 41.1 ms |
| Qwen3.8 (`c4fb47a9…`), lease per step | 23.24–23.31 | 1.10–1.15 ms | 40.7–40.8 ms |
| Qwen3.8 (`c4fb47a9…`), lease per request | **23.71–23.81** | 0.009–0.012 ms | 40.8 ms |

DeepSeek's rows are two runs (r1, r2), each the mean of three passes;
the review's re-run with its fixes (09:14, the same artifacts, one run)
gave 20.41 tok/s for a lease per request with graphs and 19.82 per step
(means of three passes), within them, and the 8 prompts again exact. The
first two Qwen3.8 rows are one run (r1, passes 23.67–23.73 per request)
from its first artifact (`67617f87…`) and runner, before the prefill
slice; the last two are the review's one run (per pass) from the
CUTLASS-layout artifact and the fused graph that slice landed, which
the Qwen3.8 statements below use.

At the lanes' 200 µs poll: DeepSeek's round trip 2.2–2.6 ms per step with
a lease per step, 0.26–0.46 ms per request; Qwen3.8's (first artifact)
1.4–1.8 and 0.40–0.65 ms. Against the references: DeepSeek at 1.016–1.022× llama.cpp's tg64 with
fusion off and graphs on (20.02 ± 0.06, the same session) and 0.990–0.996×
with fusion on (20.54 ± 0.06); graphs are still worth 1.046–1.055×, so they
stay ([graphs](graphs.md#re-measured-with-a-lease-per-request-spark-b-2026-09-28)).
Qwen3.8 (`c4fb47a9…`) at 0.94× Mia's vLLM with speculation off
(23.71–23.81 against 25.12–25.33 tok/s, [baselines](baselines.md)), where
its resident harness decodes at 0.99×. The lease is not the cause: a
token takes 42.1 ms, of which the job's span on the device is 40.8 ms
(itself 1.03× vLLM's whole 39.5–39.8 ms step) and about 1.3 ms is host
work outside the job (the n-gram rows read on the caller's thread, the
inputs built over the whole history); the round trip is 0.01 ms. Qwen3.8
has no decode graphs yet (graphs.md's limits).

**Bit-identity** (r2, graphs on, 100 ms poll; every check exact):

| Check | Result |
| --- | --- |
| Paged DeepSeek against the resident harness (so llama.cpp with fusion off; context 4,096, the 8 dsv4-native prompts, prefill + 31 steps, a request per prompt) | 0 logits differ; 246 steps replayed, 1 captured, 9 launch by launch |
| Paged Qwen3.8 against the resident harness (the six prompts, as above) | 0 logits differ |
| The image, one request per generation | pixels `95fbcbc5…`, the resident harness's |
| Every bench pass against the first warm-up (per step and per request, launch by launch and replayed; 32 passes) | 0 steps differ |
| DeepSeek ↔ Qwen3.8, A→B→A (the protocol above, one run) | A's state digest and every continued step exact in both returns; B's output identical; DeepSeek's cycle prefills equal to the control's |

**Swap times did not move** (DeepSeek ↔ Qwen3.8 from its first artifact,
r2, seconds; before: this file's table): A→B 7.79 first use, 7.74 prepared, 7.72 at 0 context (8.79,
7.67, 7.77 before); B→A 8.67, 8.65, 8.83 (9.04, 8.76, 8.85); page-in
13.3–13.4 GB/s; the prepared return's first token 0.060 s, replayed from a
graph captured before the swap. A request's lease is taken inside the
first output: Materialize and the lease over the closure, once.

**Tests.**
- `unit.HeldLeaseTest.*` (fakes, managed backing): a request's steps run
  under its one lease and its end leaves the closure resident and unleased
  with its last use advanced; an eviction is refused while it is held and
  an evicting program waits, without spinning, until it ends; a swap
  waits for the request's end, then hands the backing over; a request
  cancelled with a step in flight keeps its lease (no eviction can begin)
  until that step's fence; ending with a step in flight waits for its
  fence, a second end waits for the same release, and a step after the
  end is refused; a signal after the request's end (or for no request)
  wakes and keeps nothing; a task holding a lease, and its child, are
  refused a wait for another's, and may wait once their own has ended; a
  signal that comes before the wait is kept; only the holder submits
  under or ends a lease, and
  only a resident closure is leased; an unproven step keeps the lease for
  good and faults the stop; and a threaded run (every lane on its own
  thread) of 400 steps loses no signal, with a swap waiting throughout.
  With the scheduler, page-in, lanes and acquisition tests, it passes
  under ThreadSanitizer (`spark-native` with TSan, three runs).
- `unit.CudaPagedNodeTest.ARequestLeasesOnceAndItsStepsRunUnderIt` (GB10):
  steps under one lease per extent throughout, their bytes the file's; a
  step over the other model's closure refused before it runs; the
  model resident and unleased after the end; a swap between a request's
  steps ends it first, and the incoming model reads back whole under its
  own request.

### With the runtime wake

The figures above were harness-polled. From D-094 on the node runs the
runtime's own wake ([runtime-wake](../runtime-wake/README.md)): the
completion lane sleeps through most of a step and wakes the scheduler and
the submission lane just ahead of its end, and the harness's driver waits
the same way. Re-measured on `spark-b` with the same driver, build
conditions and artifacts as the Qwen3.8 review's above (DeepSeek
`8a355bfb…`, Qwen3.8 `c4fb47a9…`, the DSpark drafter `dd2d3f9c…`), the
memory gate before each process, one process per run, in two sessions:
on the lease-per-request tree (`a84146c` with this change, 10:31–11:24),
and after main's Qwen3.8 fast prefill (`f9a4e0f` with this change,
11:26–11:39), from which Qwen3.8's figures come. "100 ms windows" is the
same binary with `--poll-us 100000`, the old harness's polling, as a
same-session reference. Raw outputs: `~/scratch/m3wake/models`,
`models2` and `models3` on `spark-b`.

| Decode (tok/s, mean of three passes) | Runtime wake | 100 ms windows | Round trip a step (wake; windows) |
| --- | ---: | ---: | ---: |
| DeepSeek, lease per request, graphs (64 steps; `a84146c` twice, `f9a4e0f` once) | **20.42; 20.50; 20.48** | 20.32 | 0.033–0.046; 0.017 ms |
| DeepSeek, lease per step, graphs | 19.92; 19.94; 19.97 | 19.68 | 1.36–1.45; 1.51 ms |
| Qwen3.8, lease per request (128 steps, `f9a4e0f`) | **24.09; 23.85** | 24.03 | 0.029–0.033; 0.015 ms |
| Qwen3.8, lease per step (`f9a4e0f`) | 23.26; 23.56 | 23.67 | 1.01–1.67; 0.90 ms |
| DeepSeek with DSpark, `prose` / `code` (256 tokens, median of three, `a84146c`) | **29.67 / 30.85** | 29.63 / 30.30 | ([dspark](../dspark/README.md#performance-and-memory)) |

Every bench pass equalled its first warm-up (0 steps differ), and every
DSpark greedy check was bit-identical. The runtime wake's round trip is
0.01–0.03 ms a step above polling's, which the steps' device time (48.5–48.9
ms for DeepSeek, 40.2–40.9 for Qwen3.8) varies by more from run to run;
decode is not slower for it. The per-step arms (a lease of their own per
step, M7's path) paid more before the follow window covered them
([runtime-wake](../runtime-wake/README.md#re-benchmark-the-models-with-the-runtime-wake));
the `f9a4e0f` rows have it. Against the references: DeepSeek at
1.020–1.024× llama.cpp's tg64 with fusion off and graphs on (20.01 ± 0.07,
the same session) and 0.999–1.003× with fusion on (20.43 ± 0.10);
Qwen3.8 at 0.94–0.96× Mia's vLLM with speculation off (25.12–25.33);
DeepSeek with DSpark at 0.96× / 0.97× llama.cpp's with the same drafter
(30.80 / 31.94, baselines.md).

**Prefill** (the pairs' controls: 8,192 tokens of `docs/decisions.md` in
the model's chunks, under one request, runtime wake): DeepSeek 27.37–27.45
s (298–299 tok/s; 27.56 s with the 100 ms windows; 27.34 s harness-polled
in the lease-per-request slice's run); Qwen3.8 6.84 s at `f9a4e0f` (1,198
tok/s; 8.20 s on the `a84146c` tree, 8.10 s before D-093 with the lanes'
200 µs windows). Prefill chunks run for seconds, so the wake is noise
there.

**DeepSeek ↔ Qwen3.8** (seconds, runtime wake; the protocol above, every
check exact: A's state digest and every continued step in both returns,
B's output across cycles, each cycle's prefill equal to the control's for
DeepSeek, and for Qwen3.8 at `f9a4e0f`, whose fast graph breaks ties by
cell (RE-031); on the `a84146c` tree Qwen3.8's still differed, as RE-031
allows):

| A ↔ B | Swap | `f9a4e0f` | `a84146c` | `a84146c`, 100 ms windows | Page-in GB/s (`f9a4e0f`) |
| --- | --- | ---: | ---: | ---: | ---: |
| DeepSeek ↔ Qwen3.8 | A→B, first use | 7.462 | 7.421 | 7.788 | 13.89 |
| | B→A, first use | 8.692 | 8.704 | 8.786 | 13.37 |
| | A→B, prepared | 7.390 | 7.382 | 7.404 | 13.86 |
| | B→A, prepared | 8.666 | 8.698 | 8.719 | 13.37 |
| | A→B, 0 context | 7.373 | 7.358 | 7.483 | 13.87 |
| | B→A, 0 context | 8.806 | 8.774 | 8.804 | 13.35 |
| Qwen3.8 ↔ DeepSeek | A→B, first use | 8.801 | 8.771 | | 13.35 |
| | B→A, first use | 7.186 | 7.082 | | 13.89 |
| | A→B, prepared | 8.704 | 8.702 | | 13.34 |
| | B→A, prepared | 7.184 | 7.057 | | 13.88 |
| | A→B, 0 context | 8.702 | 8.651 | | 13.36 |
| | B→A, 0 context | 7.246 | 7.233 | | 13.88 |

The swaps did not move (7.1–8.8 s, against 6.9–8.9 s in the Qwen3.8
review's runs above): they are page-in bound, and the zone's copies,
whose completion lane now sleeps between likely ends as well, still land
at 13.3–14.1 GB/s. The prepared return's first token 0.058–0.062 s. Peak
memory 95.6–96.5 GiB (98.8 in the 100 ms run; one sample each, the host
shared).

## Through llmp-runtime (D-096)

### Final integrated table, 2026-09-30

On `spark-b`, 10:56:55–11:05 EDT, step one of supervised job
`m3-final-table-and-qwen-http-v3` completes rc0 on the checked native
path. One production process registers all three models, with DeepSeek
and Qwen ceilings 262,144, chunks 2,048 / 4,096, DSpark and adaptive
prefix MTP enabled, and the final growing-state/turn-reuse/HCA/frontier
implementations. Every ordered pair runs first-use and prepared cycles
with an 8,192-token saved context; each LLM A also runs a zero-context
cycle. First-use/prepared retain the definitions below.

All 32 swaps pass. Every 8K return preserves its state digest and all
16 continued token/logit rows; every prepared 8K return replays graphs
retained across the swap. B's output and the image controls are exact.
The zero-context LLM returns compare independently recomputed prefill
hashes across both partners, rather than a saved-state continuation.

Seconds from swap request to the same first-output endpoints as below.
Each cell lists A→B / B→A separately, covering all 32 observations.

| A ↔ B | First use, 8K | Prepared, 8K | Prepared, zero context |
| --- | ---: | ---: | ---: |
| DeepSeek ↔ image | 6.112 / 9.655 | 5.788 / 8.719 | 5.753 / 8.865 |
| DeepSeek ↔ Qwen3.8 | 8.015 / 9.549 | 7.981 / 9.565 | 8.014 / 9.701 |
| image ↔ DeepSeek | 8.911 / 5.716 | 8.870 / 5.789 | — |
| image ↔ Qwen3.8 | 6.596 / 5.098 | 6.557 / 5.127 | — |
| Qwen3.8 ↔ DeepSeek | **9.749** / 7.796 | 9.684 / 7.785 | 9.657 / 7.778 |
| Qwen3.8 ↔ image | 5.211 / 6.431 | 5.136 / 6.417 | 5.052 / 6.447 |

The worst LLM↔LLM and first-use swap is 9.748554 seconds; the worst
prepared swap is 9.700686. All satisfy the approximately 10-second goal,
20-second prepared bound and 40-second first-use bound. Peak sampled
memory drop is 119,427,960,832 bytes; available memory starts at
125,658,554,368 and falls no lower than 6,215,024,640 bytes. Fixed
allocations total 5,364,111,644 bytes. This is one final table, not a
distribution or a maximum-context restore measurement; the separate
[long swaps](../m3-final-context/README.md#continuing-context-swap-protocol)
cover the latter.

The fast image runs 40 steps at 1024², seed 42, CFG 1, on the original
red-teapot prompt and BF16 noise. Its final RGBA SHA-256 is the previously
qualified `3b7770ca…`, checked by `--image-expect`; the table's image
endpoint hash is the first denoising output, a different buffer. The final
RGBA check is exact continuation evidence, not a new diffusers comparison.

Provenance: GB10, driver 580.178.04, CUDA 13.4.92, `CUDA_DISABLE_PTX_JIT=1`;
runtime SHA-256
`757c29452d26bfbd0fcc07686d0a13a632eef0850621be68e02ad7144886eaf0`.
The production kernels match `cf63418`'s unchanged default path; subsequent
head/grouped trials are absent. Target/drafter artifacts are DeepSeek
`8a355bfb…` / `dd2d3f9c…` and Qwen `c4fb47a9…` / `056a750e…`;
image composition `eca21baa…` retains its three original components.
Input context is `4655685:docs/decisions.md`, SHA-256
`6b159ff20d198a3ff825d78b5edc0c2bd8c9cf5af7222875fb88301635e14db5`;
noise SHA-256
`eb7333893ca0409a100955e5305482bb9de8225ee3c21d33fd01fd6c16c042bd`.
All-query memory/process preflight passed before load. Raw config, input
pins, native table and validated summary are
`spark-b:~/scratch/m3-final-table/`; `table.json` SHA-256 is
`e4057841e5f596b9897a9941e85fc411fa72808d3d51ca493272ceaf955b93c7`.
The earlier invalid enrollment-anchor attempt refused before model load
and is excluded. No workstation checks were run during this table.

### Initial integrated table

Since D-096 the runtime runs the swap path itself
([runtime-serving.md](../../runtime-serving.md)): `llmp-runtime
swap-table` registers every configured model on one node and runs the
swap pairs' protocol over every ordered pair in one process, every fast
path on: the CUTLASS-layout Qwen3.8 artifact (`c4fb47a9…`), decode graphs,
a lease per request, the runtime wake, the handoff, and speculation, so
DeepSeek pages its DSpark drafter with it (108.36 GB with its state,
against 97.46 without) and Qwen3.8 its MTP block (77.03 GB against
75.39). The differences from `llmp_swap_pairs`: all three models are
registered at once (so the fixed memory holds all three's own memory, 4.39
GiB, and first use means the incoming model's plans and graphs are
dropped, since it may have run in an earlier pair: the CUDA modules and
library state an earlier pair loaded stay loaded, which the harness's
first-use B, new to its process, did not have); B's prompt is the
fixed set's `capital` rendered by B's chat template from a cleared state
(the harness fed DeepSeek 6 raw tokens); an LLM A's return continues with
speculative steps, its endpoint the first step's end (the first token
generated after the swap), and its 16 tokens and their logits (the verify
rows') must equal the same state's unswapped continuation; B's
conversation is dropped after its first output, so a swap never spills or
restores it.

**Results** (`spark-b`, GB10, driver 580.178.04, the `spark-native`
build, `CUDA_DISABLE_PTX_JIT=1`; one run, 16:00–16:11, after the memory
gate; the artifacts' shards 6–15 h old). Seconds, each part from the end
of the one before; total from the swap request to the endpoint (an LLM B:
its first token for the short prompt; an LLM A: the first token generated
after the swap, its state digest excluded; the image: the first denoising
step's output). Page-in counts the incoming weights and a returning LLM A's
state. Every row exact: an LLM A's restored state hashed as it left, its
16 continued tokens and logits equal; B's first output the same every
cycle; the image regenerated to the reference's pixels (`95fbcbc5…`);
every prepared return replayed decode graphs captured before the swap.

| A ↔ B | Swap | Total | Evict and spill | Restore | Page-in (GB at GB/s) | First output (planning) | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek ↔ Qwen3.8 | A→B, first use | 8.024 | 1.951 | — | 5.753 (76.60 at 13.3) | 0.318 (0.033) | 108.0 |
| | B→A, first use | 9.596 | 1.354 | 0.099 | 8.019 (108.36 at 13.3) | 0.120 (0.075) | 108.0 |
| | A→B, prepared | 7.948 | 1.925 | — | 5.778 (76.60 at 13.3) | 0.242 | 108.1 |
| | B→A, prepared | 9.535 | 1.328 | 0.100 | 8.007 (108.36 at 13.4) | 0.095 | 108.0 |
| | A→B, 0 context | 7.913 | 1.892 | — | 5.775 (76.60 at 13.3) | 0.243 | 108.0 |
| | B→A, 0 context | 9.706 | 1.358 | — | 8.091 (107.89 at 13.3) | 0.253 (0.080) | 108.0 |
| Qwen3.8 ↔ DeepSeek | A→B, first use | **9.719** | 1.375 | — | 8.080 (107.89 at 13.4) | 0.260 (0.073) | 108.0 |
| | B→A, first use | 7.892 | 2.011 | 0.065 | 5.720 (77.03 at 13.3) | 0.093 (0.036) | 108.0 |
| | A→B, prepared | **9.644** | 1.373 | — | 8.084 (107.89 at 13.3) | 0.185 | 108.0 |
| | B→A, prepared | 7.746 | 1.887 | 0.064 | 5.719 (77.03 at 13.3) | 0.073 | 108.0 |
| | A→B, 0 context | 9.627 | 1.343 | — | 8.095 (107.89 at 13.3) | 0.186 | 108.4 |
| | B→A, 0 context | 7.888 | 1.961 | — | 5.750 (76.60 at 13.3) | 0.173 (0.032) | 108.4 |
| DeepSeek ↔ image | A→B, first use | 6.029 | 1.879 | — | 2.376 (30.72 at 12.9) | 1.771 | 107.7 |
| | B→A, first use | 8.790 | 0.536 | 0.093 | 8.037 (108.36 at 13.3) | 0.121 (0.075) | 107.8 |
| | A→B, prepared | 5.827 | 1.856 | — | 2.309 (30.72 at 13.3) | 1.659 | 108.0 |
| | B→A, prepared | 8.738 | 0.538 | 0.093 | 8.011 (108.36 at 13.4) | 0.093 | 108.0 |
| | A→B, 0 context | 5.830 | 1.888 | — | 2.312 (30.72 at 13.3) | 1.627 | 108.0 |
| | B→A, 0 context | 8.879 | 0.553 | — | 8.073 (107.89 at 13.4) | 0.249 (0.074) | 108.0 |
| image ↔ DeepSeek | A→B, first use | 8.939 | 0.562 | 0.092 | 8.022 (108.36 at 13.4) | 0.259 (0.074) | 108.7 |
| | B→A, first use | 5.874 | 1.862 | — | 2.311 (30.72 at 13.3) | 1.700 | 108.7 |
| | A→B, prepared | 8.831 | 0.562 | — | 8.081 (107.89 at 13.4) | 0.185 | 108.4 |
| | B→A, prepared | 5.964 | 1.848 | — | 2.473 (30.72 at 12.4) | 1.641 | 108.4 |
| Qwen3.8 ↔ image | A→B, first use | 5.315 | 1.382 | — | 2.330 (30.72 at 13.2) | 1.602 | 78.6 |
| | B→A, first use | 6.433 | 0.560 | 0.058 | 5.719 (77.03 at 13.3) | 0.094 (0.035) | 78.5 |
| | A→B, prepared | 5.300 | 1.372 | — | 2.312 (30.72 at 13.3) | 1.614 | 78.5 |
| | B→A, prepared | 6.393 | 0.540 | 0.058 | 5.721 (77.03 at 13.3) | 0.073 | 78.5 |
| | A→B, 0 context | 5.255 | 1.321 | — | 2.319 (30.72 at 13.2) | 1.613 | 78.5 |
| | B→A, 0 context | 6.407 | 0.528 | — | 5.745 (76.60 at 13.3) | 0.132 (0.032) | 78.5 |
| image ↔ Qwen3.8 | A→B, first use | 6.520 | 0.551 | — | 5.757 (76.60 at 13.3) | 0.211 (0.033) | 78.5 |
| | B→A, first use | 5.225 | 1.311 | — | 2.311 (30.72 at 13.3) | 1.602 | 78.5 |
| | A→B, prepared | 6.482 | 0.551 | — | 5.754 (76.60 at 13.3) | 0.176 | 78.4 |
| | B→A, prepared | 5.266 | 1.323 | — | 2.310 (30.72 at 13.3) | 1.632 | 78.4 |

Setup (DeepSeek's hash-routing check, Qwen3.8's n-gram hash) took 1–5 ms
every time. Every swap handed off every extent of backing the incoming
model could take (36,554–36,759 between the LLMs, 14,719 with the image).
In this run the image ↔ DeepSeek first-use A→B also restored a DeepSeek
conversation an earlier pair had left spilled (0.092 s); the table now
drops the other models' conversations at each pair's start, and a rerun of
that pair and DeepSeek ↔ Qwen3.8 (16:22–16:27) gave 9.016 s with no restore
there, the rest within 0.2 s of the table's, every row exact.

**Against M3's targets:** all 32 swaps are under the ~10 s goal (so under
the ~20 s bound; first use has its own ~40 s). **The worst LLM↔LLM swap
is 9.72 s** (Qwen3.8 → DeepSeek, first use, 8K context), the worst
prepared at 8K 9.64 s; swaps into the image reach its first denoising step
in 5.2–6.0 s. Swaps into DeepSeek are page-in bound: 108 GB at the SSD's
13.3–13.4 GB/s is 8.0–8.1 s, 0.8 s of it the DSpark drafter, which the
first token does not need (not tried: paging the drafter after the first
token). Peak memory 107.7–108.7 GiB with DeepSeek resident; the lowest
`MemAvailable` in the run was 6.8 GiB of 115.5 at its start.

**Speculation off** (`--plain`, no drafters; 16:11–16:15), beside the
harness's table from the CUTLASS-layout artifact above (one process a
pair, fresh shards then):

| A ↔ B | Swap | Runtime | Harness |
| --- | --- | ---: | ---: |
| DeepSeek ↔ Qwen3.8 | A→B, first use | 7.722 | 7.917 |
| | B→A, first use | 8.667 | 8.886 |
| | A→B, prepared | 7.611 | 7.031 |
| | B→A, prepared | 8.680 | 8.638 |
| Qwen3.8 ↔ DeepSeek | A→B, first use | 8.903 (0.106 of it the stale restore above) | 8.767 |
| | B→A, first use | 7.476 | 6.863 |
| | A→B, prepared | 8.809 | 8.698 |
| | B→A, prepared | 7.487 | 6.879 |

The swaps into Qwen3.8 are 0.4–0.6 s slower than the harness's: page-in
at 13.3 GB/s against 14.7–14.8 then, when the new artifact's shards were
under an hour old (RE-027's recent-write rate).

## Results: M3's swap pairs (`spark-b`, 2026-09-28)

GB10, kernel 7.0.0-1019-nvidia, driver 580.178.04, the `spark-native`
build, `CUDA_DISABLE_PTX_JIT=1`, the node as above (8 landing slots of
2 MiB + 8 KiB, four reads in flight, copy lane, handoff on). DeepSeek
`8a355bfb…` and Qwen3.8 `67617f87…` as imported on `spark-b`; the image's
composition `eca21baa…` and its three components copied from `spark` over
the direct link (10.100.208.x, rsync, 05:02–05:04, 84 s for 33 GB). File
ages at the final runs, from their change times (RE-027): DeepSeek's
shards 4.7–5.2 h, Qwen3.8's 3.2–3.8 h (both at rest), the image's 1.7–2.0 h
(written by the copy, and still reading at 14.2–14.8 GB/s, RE-027's
recent-write rate). Each pair started once `spark-b` had no GPU process,
more than 110 GB `MemAvailable` and a 1-minute load average under 6;
other agents' work shared the host between and during runs (load
averages up to 10 at a run's end). One process per ordered pair, one run
each (`pairs-final`, 06:25–07:02); raw outputs in
`~/scratch/m3pairs/` on `spark-b` (`pairs-final`, the earlier `pairs-try1`
with the same swap path, `q38-*`, `img-paged-1`, the probes).

**Correctness** (every check of the final runs passed; `exact` in every
row below):

| Check | Result |
| --- | --- |
| Paged Qwen3.8 against the resident harness (context 4,096, the six qwen38-native prompts, prefill + 31 greedy steps, same build) | 0 of 6 × 32 × 248,320 logits differ |
| Paged image against the resident harness (teapot, 1,024², 40 steps, seed 42's latents) | pixels `95fbcbc5…`, the resident harness's, in every generation: 2 controls, 4 after swaps, 1 alone |
| An LLM A's state after the swap back against the state it left with (SHA-256 of the whole region) | identical in all 12 returns at 8K context (DeepSeek 462,635,008 bytes, Qwen3.8 385,425,408) |
| An LLM A's 16 continued steps against the same state's unswapped continuation | every logit and token identical, all 12 returns |
| B's first output across its cycles and the 0-context pair (logits, or the image's first noise prediction) | identical, every pair |
| BP-A1: bound tensors in cataloged extents of their class | 0 outside (DeepSeek and Qwen3.8, every shape planned) |
| Qwen3.8's cycle prefill against the process's control (a rerun, not a swap check) | differs in 3 of 4 cycles from the 11th–14th chunk on (RE-031); DeepSeek's always equal |

**Swap times** (seconds, each part from the end of the one before, adding
up to the total from the swap request to the first output; LLM B: its
first token for its short prompt from a cleared state; LLM A: the next
token after its 8,192-token context, or at 0 context a 16-token prompt's;
image: the prompt encoded and the first denoising step's output). Page-in
counts the incoming weights and, returning to an LLM A, its state. Peak:
in use at the swap's lowest `MemAvailable` against the process's start.

| A ↔ B | Swap | Total | Evict and spill | Restore | Page-in (GB at GB/s) | First output (planning) | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek ↔ Qwen3.8 | A→B, first use | 8.789 | 1.746 | — | 6.539 (75.24 at 11.5) | 0.501 (0.087) | 94.6 |
| | B→A, first use | **9.040** | 1.263 | 0.099 | 7.556 (97.46 at 12.7) | 0.119 (0.053) | 94.8 |
| | A→B, prepared | 7.673 | 1.701 | — | 5.599 (75.24 at 13.4) | 0.370 | 96.0 |
| | B→A, prepared | **8.763** | 1.287 | 0.096 | 7.312 (97.46 at 13.2) | 0.064 | 94.9 |
| | A→B, 0 context | 7.773 | 1.755 | — | 5.643 (75.24 at 13.3) | 0.373 | 94.9 |
| | B→A, 0 context | 8.849 | 1.329 | — | 7.296 (97.00 at 13.3) | 0.221 (0.054) | 94.4 |
| Qwen3.8 ↔ DeepSeek | A→B, first use | 8.729 | 1.275 | — | 7.261 (97.00 at 13.4) | 0.189 (0.079) | 96.3 |
| | B→A, first use | 7.372 | 1.593 | 0.059 | 5.602 (75.62 at 13.4) | 0.115 (0.057) | 96.2 |
| | A→B, prepared | 8.635 | 1.266 | — | 7.262 (97.00 at 13.4) | 0.104 | 95.6 |
| | B→A, prepared | 7.423 | 1.635 | 0.059 | 5.677 (75.62 at 13.2) | 0.049 | 95.6 |
| | A→B, 0 context | **9.378** | 1.295 | — | 7.975 (97.00 at 12.2) | 0.105 | 96.2 |
| | B→A, 0 context | 7.495 | 1.598 | — | 5.603 (75.24 at 13.4) | 0.292 (0.086) | 96.2 |
| DeepSeek ↔ image | A→B, first use | 5.609 | 1.747 | — | 2.083 (30.72 at 14.7) | 1.777 | 98.1 |
| | B→A, first use | 7.938 | 0.532 | 0.092 | 7.195 (97.46 at 13.4) | 0.116 (0.053) | 98.3 |
| | A→B, prepared | 5.512 | 1.709 | — | 2.083 (30.72 at 14.7) | 1.718 | 98.4 |
| | B→A, prepared | 8.278 | 0.514 | 0.086 | 7.610 (97.46 at 12.7) | 0.063 | 100.1 |
| | A→B, 0 context | 5.457 | 1.665 | — | 2.083 (30.72 at 14.7) | 1.708 | 100.0 |
| | B→A, 0 context | 8.148 | 0.529 | — | 7.364 (97.00 at 13.2) | 0.251 (0.084) | 99.2 |
| image ↔ DeepSeek | A→B, first use | 9.094 | 0.536 | — | 8.217 (97.00 at 11.8) | 0.336 (0.096) | 103.9 |
| | B→A, first use | 6.280 | 2.530 | — | 2.132 (30.72 at 14.4) | 1.615 | 104.0 |
| | A→B, prepared | 8.350 | 0.540 | — | 7.699 (97.00 at 12.6) | 0.106 | 100.0 |
| | B→A, prepared | 5.620 | 1.656 | — | 2.255 (30.72 at 13.6) | 1.708 | 100.0 |
| Qwen3.8 ↔ image | A→B, first use | 5.060 | 1.310 | — | 2.078 (30.72 at 14.8) | 1.671 | 77.1 |
| | B→A, first use | 6.277 | 0.524 | 0.044 | 5.593 (75.62 at 13.4) | 0.115 (0.056) | 77.3 |
| | A→B, prepared | 5.025 | 1.303 | — | 2.090 (30.72 at 14.7) | 1.631 | 77.3 |
| | B→A, prepared | 6.234 | 0.538 | 0.044 | 5.600 (75.62 at 13.4) | 0.050 | 77.3 |
| | A→B, 0 context | 5.047 | 1.308 | — | 2.086 (30.72 at 14.7) | 1.652 | 77.3 |
| | B→A, 0 context | 6.425 | 0.546 | — | 5.641 (75.24 at 13.3) | 0.235 (0.082) | 77.2 |
| image ↔ Qwen3.8 | A→B, first use | 6.503 | 0.516 | — | 5.623 (75.24 at 13.4) | 0.363 (0.081) | 76.6 |
| | B→A, first use | 5.002 | 1.261 | — | 2.099 (30.72 at 14.6) | 1.641 | 76.6 |
| | A→B, prepared | 6.591 | 0.535 | — | 5.787 (75.24 at 13.0) | 0.266 | 79.4 |
| | B→A, prepared | 5.045 | 1.265 | — | 2.168 (30.72 at 14.2) | 1.610 | 79.5 |

Setup (DeepSeek's hash-routing check, Qwen3.8's n-gram hash; the image
has none) took 1–5 ms every time and is not shown. Qwen3.8's row reads
within its first output took at most 10 ms. Every swap handed off the
outgoing backing the incoming model could take (35,873 extents between
the LLMs, 14,719 with the image), and released the rest after the swap.
Bytes read per swap are the page-in column's; the n-gram rows added at
most 0.2 MB.

**Against M3's targets** (plan.md's swap table): every swap is under the
~10 s goal and so under the ~20 s bound, first use included (its own
target is ~40 s). **The worst LLM↔LLM swap is 9.38 s** (Qwen3.8 →
DeepSeek, prepared, 0 context); at 8K saved context the worst prepared
one is 8.76 s and the worst first-use one 9.04 s (both DeepSeek's
return). An earlier run of the same swap path (`pairs-try1`, before the
state-digest check was added) measured 7.24–9.66 s for the LLM↔LLM swaps,
its worst 9.66 s a Qwen3.8 → DeepSeek at 12.3 GB/s: the margin to 10 s is
the SSD's rate for DeepSeek's 97 GB. The swaps into an LLM are page-in
bound (75–97 GB at 11.5–13.4 GB/s is 5.6–8.2 s; the eviction before it
0.5 s of the image's 14,719 extents, 1.3–1.8 s of an LLM's 36,057–46,453;
the first token 0.05–0.5 s); into the image they take 5.0–6.3 s, 2.1 s of
it the 30.7 GB page-in and 1.6–1.8 s the encode and first step, whose step
runs at about 1.5 s against 0.90 s alone while the backing no load took is
released beside it (judgement calls). These runs had no CUDA graphs:
they predate the decode-graphs slice (D-090).

**After the rebase onto the decode graphs** (one run, `pairs-review1`,
07:29–07:32, DeepSeek → Qwen3.8 only, the same protocol): DeepSeek's
decode steps ran as graphs (4 captured, 72 replayed, none refused), every
model's places pinned at registration and found pinned after every swap
(DeepSeek's also at their registered sources). Every check exact: A's
state digest in both returns, every continued step, B's output, DeepSeek's
cycle prefills equal to the control's. Totals A→B 7.80, 7.84 and 7.66 s (0
context), B→A 8.65, 8.66 and 8.96 s (0 context), page-in 13.1–13.4 GB/s,
the shards 5.7–5.8 h (DeepSeek) and 4.2 h (Qwen3.8) old; the prepared
return's first token 0.059 s (a replayed graph) against 0.064 s before.
Peak memory 96.1–97.0 GiB in five swaps and 107.6 GiB in the first
(94.4–96.0 GiB before); not investigated (one sample, `spark-b` shared).

**Qwen3.8 from the CUTLASS-layout artifact** (the prefill slice's
review: `c4fb47a9…`, the fused graph with CUTLASS's grouped GEMM on the
paged node, no rewrite at load; `spark-b`, 08:51–08:57, one process per
ordered pair, the same protocol, raw outputs in
`~/scratch/m3qpre-review/`). The paged runner's logits equal the resident
harness's for the six prompts' 32 steps (0 of 6 × 32 × 248,320 differ, the
resident run from the same build and artifact). Every swap row exact: A's
state digest after every return, every continued step, B's output across
cycles.

| A ↔ B | Swap | Total | Evict | Page-in (GB at GB/s) | First output | Before (above) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek ↔ Qwen3.8 | A→B, first use | 7.917 | 2.424 | 5.109 (75.00 at 14.7) | 0.380 | 8.789 |
| | B→A, first use | 8.886 | 1.211 | 7.458 (97.46 at 12.9) | 0.119 | 9.040 |
| | A→B, prepared | 7.031 | 1.654 | 5.083 (75.00 at 14.8) | 0.291 | 7.673 |
| | B→A, prepared | 8.638 | 1.262 | 7.216 (97.46 at 13.3) | 0.061 | 8.763 |
| Qwen3.8 ↔ DeepSeek | A→B, first use | 8.767 | 1.330 | 7.263 (97.00 at 13.4) | 0.171 | 8.729 |
| | B→A, first use | 6.863 | 1.679 | 5.038 (75.39 at 14.8) | 0.084 | 7.372 |
| | A→B, prepared | 8.698 | 1.335 | 7.254 (97.00 at 13.4) | 0.106 | 8.635 |
| | B→A, prepared | 6.879 | 1.704 | 5.055 (75.39 at 14.7) | 0.059 | 7.423 |

Swaps into Qwen3.8 took 6.9–7.9 s against 7.4–8.8 s before: nothing is
added at load (the prefill work's load-time rewrite took 3.9–4.0 s), and
its page-in ran at 14.7–14.8 GB/s against 11.5–13.4 (the new artifact's
shards were under an hour old, RE-027's recent-write rate, so part of
that is file age). The swaps out of it are unchanged.
Peak memory 94.2–100.3 GiB (100.3 in the first DeepSeek → Qwen3.8 swap,
95.1–95.5 in the other order), against 94.4–96.3 before; one sample.
Qwen3.8's cycle prefills against the process's control differ from the
11th chunk on, and so its continued steps against the control's (RE-031,
not a swap check).

Beside the baselines ([baselines.md](baselines.md), cold page cache, one
run each): llama.cpp's DeepSeek 0731 → Qwen3.8 (UD-IQ3_XXS) swap took
76.6 s and the return with 8K state restored 104.4 s; llmpalooza's
DeepSeek → Qwen3.8 took 7.7–8.8 s and the return 8.8–9.0 s. Mia's vLLM
reaches Qwen3.8's first token 13 min 13 s from start and TensorFold
141–143 s (0.3.5.1 and 0.3.6.2); diffusers reaches Qwen-Image's first
denoising step 212 s from process start, llmpalooza 5.0–6.3 s from the swap
request.

## What was built (DeepSeek and the FP16 stand-in)

**DeepSeek on the paged node** (`src/engine/dsv4_runner.h`). The resident
harness's graph, plan and kernels (`dsv4_common.h`, from `dsv4_exec.cc`),
over catalog extents:
- **Dense groups:** each group's 2 MiB chunks are extents at a 2 MiB-aligned
  region, landed from the artifact as FP16's are.
- **Expert slabs:** the resident expert layout keeps each layer's routed
  experts at a uniform stride S, the group's stored bytes rounded up to its
  blocks (8,064,224 bytes on 41 layers), so that GGML's stock `mul_mat_id`
  addresses expert e at slab + e·S. S is not a multiple of 2 MiB, so an
  artifact chunk cannot be an extent with its own 2 MiB backing. An extent
  is instead a 2 MiB **page** of the slab's address range. Its contents
  are the stored bytes of the at most two groups it overlaps, which are
  consecutive in the file: one 4 KiB-aligned read of up to 2 MiB + 8 KiB
  into a landing slot (slots are that size for this model), then up to two
  copies into the page (`PageSource::pieces`, new). The S − stored bytes
  between groups (3,296 on most layers) are never read by the kernels and
  are not written. Where a layer's groups change shard, the slab starts δ
  bytes into its first page (a multiple of 256) so that a page boundary
  falls in the gap between two groups, and no page needs two files
  (`LayOutSlab`, which refuses anything else). DeepSeek: 46,232 extents,
  97,001,283,584 bytes read per load (0.18% over the stored bytes, the
  pages' alignment).
- **The token table:** host VMM extents, read in place; embedding rows
  dequantized on the host, as llama.cpp's CPU backend does.
- **The state** (`Dsv4StateLayout`: the window cache, the compressed and
  indexer caches, the compressor rings; three D-068 representations):
  kPreserve live state, each 2 MiB extent with a write-back place in an
  unnamed direct-I/O spill file. Evicting it writes it back through the
  zone (D-081's reverse path); loading it restores it. Its extents come
  first in the closure, so a swap back restores the state before paging
  the weights in.
- **A chunk** is one device job on the model's stream, leasing the whole
  closure (weights, state, workspace, staging) until its fence. Plans are
  kept per chunk shape; the first chunk of each shape checks every bound
  tensor against the catalog (BP-A1). The hash-routing tables are checked
  after every full load.

**The handoff** (`scheduler.h`, D-033). An eviction asked for with a
handoff keeps the backing on the VMM lane (since 2026-10-07 still mapped
where it was, the lazy handoff, whose unmap runs in the load that takes it
or the release of what none took), and parks: the
extent stays EVICTING and charged, so the catalog counts the kept backing,
and the evictor is told it is done. A page-in whose managed backing has the
same class and size takes a parked eviction's backing: in one step the load
begins, allowed that extent's bytes over B, and the parked eviction
completes, so the charge moves and occupancy never exceeds B. The VMM lane
maps the kept backing at the new place and sets access: no `cuMemCreate`,
no release. A parked extent materialized again takes its own backing back.
Backing no load took is released when the evicting task finishes, a few at
a time (16) so thousands of releases never take every mailbox; never an
idle pool. A load that took kept backing and ends without mapping it
(cancelled, or its map refused) releases it before the extent is
nonresident again.

**Page-in submission and RE-029** (`paged_node.h`, `scheduler.h`,
`cuda_device_execution.h`). A stream holds about 1,020 pending operations
and a launch into a full one blocks the launching thread (RE-029); a
DeepSeek chunk is 4,972 launches. The zone's copies, in and out, now go to
a **copy lane**: a second `DeviceService` with its own submission and
completion threads over the zone's stream alone. And since `cuEventCreate`
turned out to block too while any thread is blocked in such a launch, the
CUDA provider takes fences' events from a **pool** made when it opens.
Together, a job that blocks the device lane's thread no longer holds up a
single copy.

**The full swap** (`src/scheduler/programs.h` `SwapProgram`): evicts
the outgoing extents, state first, 256 at a time, with or without the
handoff, then materializes the incoming closure, and notes when each ended.

**The swap runner** (`benchmarks/swap_runner.cc`, `llmp_swap_runner`): a
harness binary, built on the test harness's paged node (the native
tokenizer it links is cleared for production binaries by D-088). See its
header for the
protocol. In short:
- **A** is DeepSeek V4 Flash (artifact `8a355bfb…`), context 8,704. Its
  8,192-token context is the first 8,192 tokens, BOS first, of
  `docs/decisions.md` at `4655685` (SHA-256 `6b159ff2…`), tokenized by the
  native tokenizer from the artifact's GGUF metadata. **B** is the
  Qwen2.5-0.5B FP16 fixture (`b93cdc32…`) running the backend proof's
  `control` trajectory (32-token prompt, then 44 teacher-forced steps),
  whose logits must hash to rung 3's `bb8ae5e7…`.
- **Control:** A prefills the context (chunks of 512) and decodes 16 tokens
  greedily, never swapped. **Each cycle:** A prefills the context again
  (its logits must equal the control's), swaps to B (A's state written
  back and A's weights evicted; B's weights paged in), B runs its prompt,
  swaps back (B evicted; A's state restored, A's weights paged in), and A
  continues: each of the 16 continued steps' logits must equal the
  control's bit for bit. The first cycle is **first use** (B never ran in
  the process, so its cuBLAS handle is made in its first token; A's plans
  are dropped before it returns); the second is **previously prepared**. A
  last pair runs at **0 context**: A's state is not spilled (it stays
  resident) and A's first token is a 16-token prompt's.
- **Reloads:** DeepSeek → evict everything (state and weights) → DeepSeek
  again, twice: the heaviest page-in.
- **Overlap (RE-029):** B's weights paged in while one of A's 512-row
  prefill chunks (4,972 launches) runs on A's stream, against B paged in
  alone.
- **Parts of a swap,** from the swap request: *evict* until every outgoing
  extent is evicted or parked (A's state written back included);
  *restore* until A's state is resident again; *page-in* until the
  incoming weights are; *setup* (A: the hash-routing check); the *first
  token* (B: its cache cleared, its cuBLAS handle on first use, its
  32-token prompt; A: one decode step, planned on first use). Backing no
  load took is released after the swap, off the critical path: *released
  by* is when the last of it was (observed after the first token).

## Results with the FP16 stand-in (`spark-b`, 2026-09-28)

GB10, kernel 7.0.0-1019-nvidia, driver 580.178.04, the `spark-native`
build, `CUDA_DISABLE_PTX_JIT=1`, lanes on their own threads, 8 landing
slots of 2 MiB + 8 KiB, four reads in flight, no coalescing. The DeepSeek
artifact's shards were written at 01:43–01:47 and read from 04:14: about
2.5 hours old, at rest (RE-027; 13.3–13.4 GB/s here). No other GPU
process ran (checked before each run; `spark-b` is shared, see the
handoff table's note on CPU load). Raw outputs in
`~/.local/share/llmp/m3swap-20260928/` on `spark-b` (`swap-5`, handoff
on; `swap-6-nohandoff`; `swap-7-nocopylane`; `prompts-2`, and
`prompts-1` on an earlier build, the same). A re-run after review's
changes (the handoff kept within a domain, 1,042 events made ahead;
`review-1`, one run with one reload) was exact throughout and within 3%
of `swap-5` on every total: B→A 7.42 and 7.38 s, reload 9.11 s, the
overlap probe 0.135 s alone and 0.194 s beside the chunk.

**Correctness.**

| Check | Result |
| --- | --- |
| Paged DeepSeek against the resident harness (context 4,096, the 8 dsv4-native prompts, prefill + 31 greedy steps) | 0 of 8 × 32 × 129,280 logits differ; the same 256 tokens |
| A resumed after B against A never swapped (8,192 context tokens, 16 continued steps) | every step's logits bit-identical, and the tokens, in both returns (first use, prepared) of every run: 4 runs, handoff on and off |
| A's prefill in each cycle against the control's | bit-identical |
| B after each swap | logits SHA-256 `bb8ae5e7…`, rung 3's, every time |
| A's bound tensors in cataloged extents of their class (BP-A1, once per shape) | 775,732 checked, 0 outside |

**Swap times** (seconds; handoff on; each part from the end of the one
before, so they add up to the total, the swap request to the first
token). A's context is 8,192 tokens unless noted; its state is 462,635,008
bytes (221 extents, 463,470,592 bytes written back and read again).

| Swap | Total | Evict and spill | Restore | Page-in (GB at GB/s) | Setup | First token | Handed off | Released by |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A→B, first use | 1.857 | 1.689 | — | 0.107 (1.26 at 11.8) | — | 0.061 | 620 | 1.55 |
| B→A, first use | **7.439** | 0.027 | 0.087 | 7.179 (97.46 at 13.4) | 0.003 | 0.142 (0.079 planning) | 620 | 1.11 |
| A→B, prepared | 1.790 | 1.655 | — | 0.110 (1.26 at 11.4) | — | 0.025 | 620 | 1.54 |
| B→A, prepared | **7.362** | 0.027 | 0.086 | 7.182 (97.46 at 13.4) | 0.003 | 0.064 | 620 | 0.98 |
| A→B, 0 context | 1.828 | 1.689 | — | 0.114 (1.26 at 11.0) | — | 0.026 | 620 | 1.57 |
| B→A, 0 context | 7.522 | 0.030 | — | 7.244 (97.00 at 13.4) | 0.003 | 0.246 (0.080 planning) | 620 | 0.25 |
| DeepSeek reload 1 (evict all, reload) | 9.071 | 1.672 | 0.091 | 7.193 (97.46 at 13.4) | 0.003 | 0.112 | 46,453 | 0.12 |
| DeepSeek reload 2 | 9.074 | 1.725 | 0.091 | 7.199 (97.46 at 13.4) | 0.003 | 0.056 | 46,453 | 0.06 |

Beside M3's targets (D-087), from the swap request to the first token:
every swap here is under the ~10 s goal (worst 7.52 s among the A↔B swaps,
9.07 s for the DeepSeek reload), and first use is within 0.08 s of
prepared. This is not M3's exit measurement: B is a 0.5B stand-in, so
A→B pages in 1.26 GB where Qwen3.8 pages in 75 GB; M3's pairs are
[above](#results-m3s-swap-pairs-spark-b-2026-09-28). The baselines' numbers stand beside them:
the pinned llama.cpp switched between DeepSeek V4 and Qwen3.8 in 75–93 s
to the first token (measured, one run, M0), TensorFold loads Qwen3.8 in
about 90 s and Mia's vLLM in 11–14 min (both creator-reported). B→A is
page-in bound: 97.46 GB at the SSD's at-rest rate is 7.2 s, and everything
else in the swap took 0.18–0.28 s. A's first load in each run, cold, took
7.26 s (13.37 GB/s). A's prefill of 8,192 tokens took 27.3 s
(300 tokens/s) and 16 decode steps 1.10 s (14.5 tokens/s at 8K context),
on the paged node with no CUDA graphs (decode graphs since: [graphs](graphs.md)). Peak memory by `MemAvailable`:
96.4 GiB over the whole run (A and B both resident at its end).

**The handoff's effect** (the same runs with `--handoff off`: evictions
unmap and release, loads create; seconds):

| Swap | Handoff on (`swap-5`) | Off (`swap-3`; `swap-6`) | What changes |
| --- | ---: | ---: | --- |
| A→B, 0 context: evict | 1.689 | 2.975; 2.995 | 46,232 unmaps kept, not released: 36.5 against 64.8 µs each |
| DeepSeek reload: evict | 1.672, 1.725 | 3.094, 3.020; 3.076, 3.019 | the same, plus the state's write-back |
| DeepSeek reload: total | 9.071, 9.074 | 10.510, 10.361; 10.472, 10.361 | −1.3 to −1.4 s |
| A→B, 8K context: evict | 1.689, 1.655 | 3.052, 3.047; — | |
| B→A: page-in | 7.179–7.244 | 7.189–7.245; 7.200–7.256 | none: page-in is read-bound, and the VMM lane's creates (46k, about 110 µs each) run ahead of the reads |

So the handoff removes the release from every eviction and the create
from every load that takes kept backing; on this path the release is what
shows, 1.3–1.4 s of a 46k-extent eviction, because the creates already
hide behind the reads. Two runs with the handoff off: `swap-3`, before
the event pool below (with the handoff off, an eviction fences only its
221 write-back copies), and `swap-6`, on the measured build but not
clean: another workload shared the host during it (load average 9.2 over
those minutes, and its peak memory in use 110.7 GiB against 95–96 GiB in
every other run; `spark-b` is shared with the Qwen3.8 and image slices).
`swap-6`'s two 8K-context A→B evictions (12.2 and 21.7 s) and its control
prefill (31.9 s against 27.3 s) are therefore not used; its other rows
agree with `swap-3`'s within 0.04 s.

**Page-in beside a busy stream (RE-029).** B's 1.26 GB paged in while one
of A's 512-row prefill chunks (4,972 launches, 1.56–1.72 s) runs on A's
stream:

| | B alone | B beside A's chunk |
| --- | ---: | ---: |
| Copy lane and event pool (`swap-5`) | 0.118 s | 0.188 s |
| Zone copies on the device lane (`swap-7`) | 0.111 s | 1.661 s |
| Copy lane, events made per fence (`swap-2`, before the pool) | 0.119 s | 0.569 s |

On the device lane the page-in waits for nearly the whole chunk: its
copies queue behind the job on the one submission thread, which is blocked
launching into the full stream. The copy lane alone was not enough:
`cuEventCreate`, which fenced each copy, blocks while any thread is
blocked launching into a full stream (RE-029's update; a gated stream held
it for 30 s in the unit test). With fences from a pool made ahead, the
page-in beside the chunk takes 0.07 s more than alone.

## Tests

- `unit.VmmWork/PageInTest.*` (fake providers, VMM work on its own lane
  and on the device lane): a handoff moves evicted backing to the loads
  that follow with no backing created and occupancy never over B; parked
  backing stays charged while its evictor runs and is released when it
  finishes; a parked extent takes its own backing back; live state written
  back and parked comes back whole; a load that took kept backing and is
  cancelled, or whose map is refused, releases it; a landed read lands in
  pieces, which are refused for write-back and beyond the read.
  `unit.VmmWork/CopyLaneTest.*`: page-ins land through the copy lane while
  the device lane never turns, and a job still runs on the device lane.
- `unit.CudaPagedNodeTest.*` (GB10): a full swap with the handoff each way
  between two synthetic models on the real VMM provider, every byte read
  back, the counts exact, no backing left; RE-029's case, a page-in beside
  a job whose stream is full.
- `unit.CudaExecutionPoolTest.FencesReuseEventsMadeAhead` (scripted driver):
  fencing makes and destroys no event once the pool is made.
- `unit.SlabLayoutTest.*`: every stored byte of every group lands once, at
  its place, from its place in one file; a shard change falls on a page
  boundary in a gap; the layouts the runner cannot page are refused; at
  Qwen3.8's 80-byte gap, a shard change the default 256-byte alignment
  cannot place lays out at 16, and alignments that are not a power of two
  from 16 to 4,096 are refused.
- `unit.PleRowsTest.*` (host): each lookup's slot holds its row's bytes
  once the planned reads land, duplicates and rows crossing a block or a
  chunk included; reads are 4 KiB-aligned, packed in the landing,
  ascending, inside the table's stored range and at most 64 KiB, and rows
  whose blocks touch share one; the whole-chunk count is the chunks the
  rows touch; rows outside the table, a landing too small, too many
  distinct rows and a table past its range are refused; on the storage
  fake, an unknown submission is waited for and its row used, and a short
  or failed read refuses the chunk's rows only after every read has
  drained.
  `unit.CudaPleRowsTest.*` (GB10): the reads through io_uring from a real
  file and the gather kernel put every row in its slot.
- Not unit-tested, checked by the runs above instead: the paged Qwen3.8
  and image runners and the pairs runner (the bit-identity and pixel
  checks against the resident harnesses, the state digests and
  continuations), and `PagedWeights`, whose layout is DeepSeek's.

## Judgement calls

- **Pages, not chunks, for the expert slabs.** A chunk-sized extent would
  straddle two 2 MiB backing pages at the slab's uniform stride, which
  needs shared backing refcounted across extents; a stride that is a
  multiple of 2 MiB and of the IQ2_XS and IQ3_XXS blocks would be 74 MiB
  per expert. A page read of up to 2 MiB + 8 KiB into a slightly larger
  slot, copied in up to two pieces, keeps D-033's one handle per extent,
  and the handoff, with 0.18% more bytes read.
- **The copy lane, not bounded streams,** for RE-029, plus the provider's
  event pool once the probe showed `cuEventCreate` blocking. Bounding each
  stream's queue would split every DeepSeek phase (4,972 launches) into
  jobs of under 1,000. By default the pool makes 256 events when the
  provider opens and keeps up to 4,096; released fences return theirs to
  it. Past what it holds, a fence makes its event as it goes (and may
  block as above). The paged node makes 1,042, every fence its device and
  copy lanes can hold at once, so neither lane ever does; the timed runs
  above made 256.
- **Parked backing is released when its evictor finishes,** and only its
  releases are paced (16 at a time): the first run without pacing
  released 45,833 handles at once, took every mailbox, and B's first job
  was refused.
- **A full swap evicts the outgoing model whole,** though B (1.3 GB) would
  fit beside A: the plan's M3 swap is between models that do not fit
  together (DeepSeek and Qwen3.8), and the stand-in B must not hide that.
  A→B's times are therefore mostly A's eviction; B→A's are the
  representative page-in.
- **The resident harness stays as it was;** the paged runner's graph and
  input code is a copy of it (`dsv4_common.cc`), so `dsv4_exec.cc`, the
  validated comparison with llama.cpp, needs no rerun. The prompts check
  above shows the two equal bit for bit.
- **0 context keeps A's state resident** rather than spilling a cleared
  one: a real 0-context model has nothing to save.
- **The n-gram table by 4 KiB-aligned row reads, per chunk, no cache**
  (D-035 asks for evidence before a smaller-read path; above): whole
  chunks would read 486× the bytes on the correctness prompts and nearly
  the whole table at 8K context. A row cache across chunks would save the
  repeated rows of nearby tokens but needs a residency contract; the reads
  cost 0.3 ms a chunk on those prompts (0.056 s over 192 chunks). They run
  on the runner's own ring on the caller's thread, not the scheduler's
  storage lane, whose sources are whole extents.
- **Qwen3.8's slab offset aligned to 16, not 256** (the resident layout's
  odd experts are 16-aligned too); the logits equal the resident
  harness's.
- **The weights' unwritten bytes are left as the backing had them.** A
  page-in writes neither a slab page's bytes between groups nor a dense
  chunk's tail past its stored length, so after a handoff they hold the
  outgoing model's bytes (weights, or spilled state: an LLM's KV), where
  the resident harness has zeros. No kernel reads them, from the code:
  every tensor a plan binds is a resource or an expert slice, and a
  quantized product reads past a row only into the slice's readable
  bytes (`CheckMulMatQ` refuses short rows otherwise, and the Qwen3.8
  graph marks them readable only inside the stride); the reader refuses
  an artifact whose readable range leaves its group's stored bytes
  (`CheckPlacement`: offset + readable ≤ stored, the last ending at
  `used_bytes`), and the artifact writes that padding as zeros inside the
  group; each page-in writes a group's whole stored range (a dense chunk
  its stored length, a slab page its groups' stored pieces). So what a
  kernel reads, the page-in wrote from the file, whatever the backing
  held. Probed too, when a first rerun differed (`--scrub-probe`, 4,096
  tokens): filling them with 0xFF (NaN in every float format, and NVFP4's
  scales, which the padding's zero activations would turn into NaN)
  changed none of Qwen3.8's logits, nor did filling the shared workspace
  so before each chunk (`--poison-probe`, Qwen3.8 and DeepSeek). The
  difference was RE-031. Nothing zeroes them: one process serves one
  user (D-019), and nothing reads or exports those bytes; a multi-tenant
  runtime would zero handed-off backing's unwritten bytes (D-014).
- **BP-A1's check skips a fill's source:** QSA's selection mask fills a
  shape-only tensor that is never bound (and never read), which the
  check first counted as 312 tensors outside the catalog.
- **One process per ordered pair,** so that every pair has a genuine first
  use of B; the image A's return has no plans to drop (its "first use"
  labels B only).
- **Copies, not shared code,** of the resident harnesses' planning
  (Qwen3.8) and phases (the image), as DeepSeek's: the validated resident
  comparisons need no rerun, and equality is checked instead.
- **The swaps' check for an LLM A compares against the same state**
  (above), since Qwen3.8 is not repeatable past 2,051 cells (RE-031); the
  fix belongs to the Qwen3.8 work, not the swap path.
- **Backing no load took is released while B makes its first output,**
  not before: with the image as B, waiting for the release first
  (`--release-first on`, DeepSeek → image, `try3rf`) took 1.00–1.01 s and
  the first output then 0.98–1.07 s, 5.73–5.83 s in all, against
  1.67–1.79 s overlapped and 5.40–5.56 s in all (the release's ~31,000
  `cuMemRelease` calls slow the first denoising step from 0.90 s alone to
  about 1.5–1.6 s, but less than they take).

## Reproduction

(The image's expected pixels are `3b7770ca…` since the image speed slice's
fast plan, its default; the runs recorded above, before it, gave and
checked `95fbcbc5…`, which its `--plan legacy` still gives.)

Through the runtime (D-096): a configuration naming the three models (as
[runtime-serving.md](../../runtime-serving.md#configuration) shows, with
`storage.installed` the store holding the artifacts and the composition,
and Qwen3.8's `tokenizer` and `chat_template` the checkpoint's), then

    llmp-runtime --config FILE --anchor PATH swap-table \
      --context-text decisions.md --image-noise ref1/latents_init.bf16 \
      --image-expect 3b7770ca… --report table.json [--plain] [--pairs A:B,...]

It takes about 11 minutes for the whole table (4 for the two LLM pairs)
and needs about 110 GiB free; it exits 1 on any failed check.

The harnesses: on `spark-b`, with the `spark-native` build, the artifacts installed as
dsv4-native's and the backend proof's are, and P2's `control-tokens.txt`:

    llmp_swap_runner --dsv4-artifact DSV4 --fp16-artifact FP16 \
      --tokens control-tokens.txt --fp16-expect bb8ae5e7e3ac6da7… \
      --text decisions.md --out DIR --reload 2 --overlap \
      [--handoff off] [--copy-lane off --cycles 0 --overlap]
    llmp_swap_runner ... --cycles 0 --context 4096 \
      --prompts dsv4-native/oracle/unfused/prompts.tokens \
      --expect dsv4-native/jit-free --generate 32

Each run takes 1–4 minutes and needs about 97 GiB free; it exits 1 on any
failed check, and `swap.json` holds every number above. Every run used
above started only once `spark-b` had no GPU process and more than 110 GB
`MemAvailable` (a wrapper that waits for both); the runner itself does not
check, and a first attempt started beside another 99 GB process was
killed by the kernel's OOM killer (no number here comes from it).

M3's pairs, each ordered pair one process (3–4 minutes each), with the
image's component artifacts and the reference's initial latents
installed as qwen-image-native's are (copied to `spark-b` for these runs):

    llmp_swap_pairs --a dsv4|qwen38|image --b dsv4|qwen38|image --out DIR \
      --dsv4-artifact DSV4 --qwen38-artifact QWEN38 --image-store STORE \
      --image-composition eca21baa… --image-noise ref1/latents_init.bf16 \
      --text decisions.md --qwen38-tokenizer tokenizer.json \
      --dsv4-prompt dsv4-native/oracle/unfused/prompts.tokens \
      --qwen38-prompt qwen38-native/prompts.tsv [--image-expect 3b7770ca…]
    llmp_swap_pairs --a qwen38 --b dsv4 ... --cycles 0 --context 4096 \
      --prompts qwen38-native/prompts.tsv --expect RESIDENT --generate 32
    llmp_swap_pairs --a image --b qwen38 ... --cycles 0 --image-expect 3b7770ca…
    llmp_swap_pairs --a dsv4|qwen38 --b ... --cycles 0 --bench 64|128 [--poll-us 200]

where RESIDENT is `llmp_qwen38_exec --context 4096 --max-rows 512
--prompts qwen38-native/prompts.tsv --generate 32`'s output from the same
build. The runs used a wrapper that also waits for the 1-minute load
average to fall below 6, and checks twice 20–40 s apart: another agent's
run started beside one of these in the same second once (its logs are
not used).

## Limits

- One process, one run per configuration; timings are single samples on
  `spark-b` (D-085's coarse comparison), not distributions.
- In the FP16 stand-in's runs, B's own page-in is 1.26 GB; M3's pairs
  are measured with `llmp_swap_pairs` above.
- The pairs' table has no CUDA graphs: "prepared" means A's plans exist
  and B ran before in the process. Since the rebase DeepSeek's decode
  steps run as graphs (`--graphs on`, the default) and one pair was rerun
  with them (above); Qwen3.8 and the image run launch by launch, their
  places pinned all the same.
- The n-gram rows are read on the caller's thread before each chunk's
  job, synchronously: a decode step waits for its 16 reads (0.3 ms). No
  row cache, no overlap with the previous chunk's job.
- The image's phases run their kernels directly, not as registry-bound
  plans (D-053), and are not checked by BP-A1's coverage check; the image
  runner's memory is cataloged but its tensors are not.
- The pairs' pinned state snapshot (the harness's check, up to 0.46 GB)
  is outside the catalog and inside the peak memory figures.
- The spill file is unnamed (`O_TMPFILE`, mode 0600, gone when the
  process exits) and a restore must read every byte back, but nothing
  checks the bytes it reads: silent corruption on the SSD would come back
  as state (D-055's named spill format stays in M6).
- The FP16 fixture's file age at the runs was not recorded (RE-027); its
  page-in is too small for the at-rest rate to show.
- These runs had no CUDA graphs: "prepared" meant A's plans and B's
  cuBLAS handle exist. With decode graphs (D-090, [graphs](graphs.md)) a
  prepared return's first token replays a graph captured before the swap:
  0.060 s against 0.064 s here, within noise, so the table above stands.
- The overlap probe's page-in beside a busy chunk is still 0.07 s slower
  than alone (the GPU and memory are shared); not investigated. Other
  driver calls than those checked (record, query, copy, synchronize on an
  idle stream) may also block behind a full stream's launch.
- Unmapping the outgoing extents one at a time dominates A→B (about 35 µs
  each over 46,453 extents); one unmap per contiguous range is not yet
  measured.
