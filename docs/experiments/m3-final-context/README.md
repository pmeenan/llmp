<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3 final context checks (2026-09-30, in progress)

This follows the [long-context study](../long-context/README.md),
[growing state](../growing-state/README.md),
[turn reuse](../turn-reuse/README.md), and
[ds4 transfers](../ds4-study/README.md). It records final-path quality,
maximum fits, continuing-context swaps and the watchdog-floor check.
The M3 gate remains open until the pending rows below have evidence.

Runs use a DGX Spark GB10, driver 580.178.04 and the pinned Spark SDK
(CUDA 13.4.92). Each timed run owns its Spark, checks at least 105 GiB
available before loading a model, and checks for other model processes.
Raw captures stay outside Git. A binary hash identifies each measured
path; a result from an earlier path does not validate a later optimization.

## Compact DeepSeek target quality at 128K

On `spark`, 2026-09-29 23:55–2026-09-30 00:09 EDT, the original
UD-Q2_K_XL target artifact `8a355bfb…` ran with F16 caches, the fast plan,
compact expert scheduling, `context = 262144` and 2,048-row chunks.
These are `e221aa8`'s target kernels, before the subsequent frontier-head
optimization. Three fresh processes cover two forced runs and perplexity;
the forced prompt contains 128,821 raw token IDs.

| Check | Result |
| --- | --- |
| Oracle forced trajectory, 512 steps | 504 matching argmaxes; eight disagreements all in the oracle's captured top five and within the unchanged 0.947-nat bound. Largest oracle margin 0.323643; zero violations. |
| Fresh-process repeat | All 512 complete 129,280-element F32 logit rows identical; maximum difference zero. |
| Long-document perplexity | 1.927387 versus llama.cpp's 1.9298, −0.125%, within the existing 3% gate. Both score indices 65,536 through 131,070 of the same 131,072-token window: 65,535 next-token losses. |
| Process time | Forced runs 273.11 / 271.12 seconds; perplexity run 294.40 seconds. These are harness process times, not runtime prefill throughput. |
| Peak drop in `MemAvailable` | Forced runs 97.082 / 97.070 GiB; perplexity 96.821 GiB. |

The full-text native perplexity summary scores a different window; its
1.882248 value is not the matched comparison above. These checks cover
an earlier native all-shape compact experiment, including partial
prefill chunks, whereas serving selects compact scheduling only from
2,048 rows. They are historical evidence, not the final corrected-floor
gate; see the [frontier follow-up](../dsv4-frontier-head/README.md#head-arithmetic-and-quality).
They do not cover DSpark, the HTTP route or maximum retrieval. The final
HCA/frontier path's fresh 32K/128K oracle controls, matched PPL and sampled
plain/speculation protocol are recorded in the
[frontier follow-up](../dsv4-frontier-head/README.md).

Provenance:

- Native `jitllm_dsv4_exec` SHA-256:
  `8ab9bab817cdb43557a57f3c6dbb3663e0770925cae9bb9c13a14c26bb0e8867`.
- Original artifact:
  `8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234`.
- Pinned llama.cpp b11254 128K forced-oracle JSON SHA-256:
  `d38333599da4e9bae0f55980663e307fd1f0b386d22a0d2c2fb44cd254a35f92`.
- Perplexity token IDs SHA-256:
  `47f997b83854b52d0eb2c3ae605b505d87cd778f573fde9664d93a0e205bf2c6`.
- Each complete forced-logit capture SHA-256:
  `c88b400701cea3613f2b27a5b0f9a59a448935f5521ed0996070494be640446b`.
- Native per-token NLL capture SHA-256:
  `28e2d8ed6a0c72d5f98077655b52fdf2d6f16dbcb5892378646324517efaeed5`.
- Raw results: `~/scratch/m3-final-ds-quality/` on `spark`. The existing
  `long-context/judge.py` judges the fixed oracle bound, full repeat and
  matched perplexity window. Native arguments include `--compact-experts
  --context 262144 --max-rows 2048`.

## Final DeepSeek runtime ladder at 262K capacity

The checked HCA/frontier path runs through the production HTTP route on
`spark-b`, 2026-09-30 05:41:57–06:19:47 EDT, job
`m3-final-ds-runtime-ladder`. Both modes use the original UD-Q2_K_XL
artifact, F16 caches, 2,048-row prefill chunks and `context = 262144`.
Plain and DSpark start fresh processes and reuse no prompt tokens.
Each of the eight requests completes its entire 512-output budget,
with the terminal marker, finish reason `length` and no stream error.

Ratios compare the pinned llama.cpp b11254 long-context captures on
the same canonical prompts, with equal prompt/output counts. The plain
reference ran on `spark-b`; the DSpark reference ran on `spark`. These
are single-run measurements on the two GB10s, rather than confidence
intervals.

| Prompt tokens | Plain prefill tok/s (× reference) | Plain decode tok/s (×) | DSpark prefill tok/s (×) | DSpark decode tok/s (×) |
| ---: | ---: | ---: | ---: | ---: |
| 31,705 | 537.924 (1.880×) | 21.660 (1.151×) | 519.508 (1.880×) | 34.152 (1.116×) |
| 64,447 | 521.907 (1.896×) | 21.280 (1.181×) | 519.159 (1.929×) | 40.462 (1.397×) |
| 128,821 | 488.504 (1.891×) | 20.565 (1.227×) | 486.573 (1.928×) | 36.961 (1.208×) |
| 258,856 | 439.434 (1.916×) | 19.544 (1.327×) | 430.089 (1.911×) | 34.836 (1.226×) |

Sampled peak drops in `MemAvailable` are 98.175 GiB plain and
108.636 GiB with DSpark, 1.025× / 1.015× their reference's
95.812 / 107.061 GiB. Native minimum available memory is
20,376,899,584 / 9,091,158,016 bytes. All rates exceed the matched
reference; both memory ratios pass the approximately 1.1× bound.
The speculative rates also depend on each prompt's acceptance; this
table does not isolate that factor or prove context-independent cost.
The remaining prefill slope and maximum-context cost require the
final profile rather than an inference from these means.

The timing prompts quote the answer markers and the responses stop at
their output budget, so these rows are not neutral retrieval passes.
The fresh 32K/128K fixed-bound oracle controls, matched perplexity and
sampled distribution checks remain the quality evidence linked above.
Neutral retrieval and continuing-context swaps remain separate gates.

Provenance:

- Runtime SHA-256:
  `08e68c23920f93c1d74c20df516e1a189cd27b14e81225f1d81d77b9eedc7254`;
  kernel source is `d753aa0` plus the unchanged `a5dff04` diagnostic fields.
- Measurement harness SHA-256:
  `20b10388886c1abff54e96ca8ec12a5f00701fffa2d8955fd9a534cc8e63f437`.
- Target/drafter identities and the llama.cpp pin are the same as below.
- Raw native captures:
  `~/scratch/m3-extrapolation-ds/final-262k-{plain,spec}/run.json` on
  `spark-b`; references:
  `~/.local/share/jitllm/m3lc/raw/ds-llama-{plain,dspark}/run.json`
  on the respective hosts above.

## DeepSeek one-million-token fit before the final optimizations

The runtime on `spark` completed a 1,038,047-token chat prompt at
`context = 1048576`, with DSpark and 2,048-row chunks, on 2026-09-29
22:00–23:02 EDT. This is the `66d5b91` production path, before compact
experts and the frontier head. It proves physical fit for this request.

| Measurement | Result |
| --- | --- |
| Prefill | 3,637.66 seconds; 285.36 tok/s |
| Decode | 29.23 tok/s; 439 outputs, from a request for at most 512 |
| Fresh prompt | Zero cached tokens; all three timing-prompt codenames found |
| Peak drop in `MemAvailable` | 113.701 GiB |
| Minimum `MemAvailable` | 3,014,848,512 bytes, 2.808 GiB |
| Registered fixed bytes / workspace / host chunk inputs | 2.51 / 2.37 / 0.04 GiB |
| Runtime execution budget | 109.86 GiB |

The used prompt plus outputs leaves 10,090 tokens of context headroom.
The timing fixture quotes its answer markers and therefore does not
substitute for neutral retrieval. This is neither a literal fully used
ceiling nor the final-path speed/memory result. Its long prefill and decode
must be compared with the same prompt and DSpark settings; earlier 128K
throughput alone cannot establish a regression.

Runtime SHA-256:
`db065b6ac18bc6cb5e731243b01929742c7b9c8451ce22711d51cfd349e01e8d`.
The canonical `1m.json` prompt's content SHA-256 is
`aad4e576a3671dc5deeedb8b56081ffdb0c442000399cdbc53f380b530786561`;
its HF count is 1,037,954 before native chat rendering.
Raw results are `~/scratch/m3-extrapolation-ds/final-1m-spec/` on `spark`.

The pinned llama.cpp b11254 image also started at a configured 1M context
on `spark-b`, plain and with DSpark, in a separate seven-token startup
probe. That probe establishes startup/capacity only. The complete matched
plain/speculative long-prompt runs completed on `spark`, 00:11–04:39 EDT
on 2026-09-30. Both process 1,037,958 prompt tokens and produce all 512
requested outputs from the same canonical timing fixture:

| Mode | Prefill seconds | Prefill tok/s | Decode tok/s | Peak drop in MemAvailable, GiB |
| --- | ---: | ---: | ---: | ---: |
| Plain | 7,719.44 | 134.460 | 8.219 | 102.696 |
| DSpark, maximum draft depth 3 | 8,086.05 | 128.364 | 16.870 | 113.788 |

Minimum available memory is 15,737,749,504 bytes plain and 3,757,690,880
bytes with DSpark. The image is `jitllm-llamacpp:b11254-cuda13`, source
`8019dc563b1ecbae6b161a70c3a1359f1b206c1e`, all target/drafter layers
on GPU, flash attention on, context 1,048,576, one slot, automatic fitting
off and prompt cache disabled. Target and drafter are the same GGUFs as
the native comparison. Chat rendering accounts for the 89-token difference
from the earlier native prompt; token counts are retained separately.
These captures precede the harness's terminal-marker/finish-reason fields;
those fields are not retroactively inferred. Successful captures and
512-output counts are retained in
`~/scratch/m3-extrapolation-ds-b/llama-1m-{plain,spec}/run.json`.
Final native plain/speculative timing and neutral retrieval started on
`spark` at 05:23 EDT using the checked HCA/frontier runtime
`7dd83ca9ceb0b077268b8448f6124ca7760425afd8ebb68b177eb9935b917770`.

## Continuing-context swap protocol

`jitllm_long_swap` uses the production runtime and runners. It warms the
alternate model, clears the target, prefills exactly the requested number
of raw token IDs and generates 64 uninterrupted tokens. It then clears
and independently repeats that prefill, swaps A→B→A, and generates the
same 64-token continuation. Both the last prefill logits and every
continuation token/logit must be bit identical. Stop handling is disabled
so all 64 comparison tokens are present. No turn checkpoints are made.

The harness reports actual used/spilled state, fixed allocations, the
execution budget, peak/minimum memory, swap parts, prefill chunk counts
and the longest completed chunk. Swap-part totals describe activation;
they are not the first-token latency used by the separate 8K swap table.

| Model | Configured context | Required saved prompt | Continuation | Status |
| --- | --- | --- | --- | --- |
| DeepSeek + DSpark, long | 262,144 | 131,072 | 64 | Pending |
| DeepSeek + DSpark, maximum | 1,048,576 | 1,048,512 | 64 | Pending |
| Qwen + MTP, long | 262,144 | 131,072 | 64 | Pending |
| Qwen + MTP, maximum | 262,144 | 262,077 | 64, plus three reserved draft positions | Pending |

The maximum continuing prompt leaves exactly the output and draft
headroom required by the production runner. The saved prompt is therefore
below the configured ceiling. The last generated token is returned before
it is processed as the next input, so this does not claim every ceiling
cell has been initialized. It checks a maximum-length continuation and
exact restoration within the supported context.

The original harness passed an 8,192-token smoke on `spark-b` at
`26bae74` plus its initial source, 2026-09-29 21:27–21:29 EDT. Both
models used their drafters and 262,144 configured ceilings; all last
prefill logits and all 64 continuation tokens/logits were exact:

| Target | Away / back activation seconds | Return restore seconds | State spilled / used bytes |
| --- | --- | --- | --- |
| DeepSeek | 7.641 / 9.398 | 0.0874 | 299,892,736 / 296,894,464 |
| Qwen | 9.379 / 7.584 | 0.0786 | 671,088,640 / 669,032,448 |

Raw captures are `~/scratch/m3-context-limits-long-swap/` on `spark-b`.
That smoke checks the comparison protocol, not the pending optimized
128K/maximum paths. The subsequent chunk-count/longest-chunk fields add
observations to the same protocol.

DeepSeek's maximum fixture extends the canonical timing corpus with the
pinned perplexity book; the combined UTF-8 input SHA-256 is
`19c009943d27e7bcbf87271528193a241acf90b66d97708a24a9fa88fe579f83`.
It is an exact-state fixture, not a retrieval or oracle-quality prompt.
The book is the long-context corpus's `ppl.txt`, SHA-256
`c7156148ecaa12b6416cf816540d8dede2014982554a835e61076f0dd8bf0c2d`.

## Remaining final-path checks

- Qwen native runtime prefill, plain decode and speculation at 32K, 64K,
  128K and 256K, with 512 requested outputs and zero cached prompt tokens.
  DeepSeek's final ladder above is complete; its largest fitting prompt
  at 1M remains in flight. Preserve actual completed output counts when
  a model stops early.
- Neutral `-r` retrieval at each rung and each maximum, with at most
  1,024 outputs, checks all three codenames. The neutral 1M prompt content
  SHA-256 is
  `a4321cf9a22cc5a384905df8b394d88d5c1dafe35dfcbe51b61171c575a98015`.
- Final frontier-head and Qwen kernel controls retain their original
  oracle/PPL bounds, seeded sampling rules and exact own-path rollback,
  repeat and swap requirements. Qwen's selected vocabulary remains a
  separate measured option; the default prefix head must meet its gate.
- Final 128K and maximum continuing-context swaps use the protocol above.
- Re-check watchdog floors against actual longest chunks and route
  completion, then record the recommendation. Current defaults are
  prefill 100 tok/s, decode 5 tok/s and a 120-second stall interval.
  `Allowance = stall + 3 × expected` permits 181.44 seconds for a
  2,048-row chunk at the prefill floor. The floor also scales the whole
  non-streaming request deadline; streamed requests have no whole-request
  deadline. A mean prefill rate alone does not prove the worst chunk
  fits its allowance. The native swap harness does not itself exercise
  the HTTP watchdog.

Workstation CPU/native checks remain deferred by the owner until all
implementations settle. The frozen M3 record will point to completed
results here without converting pending checks into passes.
