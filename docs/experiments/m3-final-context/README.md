<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
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

- Native `llmp_dsv4_exec` SHA-256:
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
Neutral retrieval results follow; continuing-context swaps remain separate
gates.

Provenance:

- Runtime SHA-256:
  `08e68c23920f93c1d74c20df516e1a189cd27b14e81225f1d81d77b9eedc7254`;
  kernel source is `d753aa0` plus the unchanged `a5dff04` diagnostic fields.
- Measurement harness SHA-256:
  `340f7d2a3ca6397c807cf0b2502954fe0a22552c3e5aa44997b34c9ba24f5cae`.
- Target/drafter identities and the llama.cpp pin are the same as below.
- Raw native captures:
  `~/scratch/m3-extrapolation-ds/final-262k-{plain,spec}/run.json` on
  `spark-b`; references:
  `~/.local/share/llmp/m3lc/raw/ds-llama-{plain,dspark}/run.json`
  on the respective hosts above.

## Final DeepSeek neutral retrieval through 256K

On `spark-b`, job `m3-final-ds-neutral-retrieval` completes successfully
2026-09-30 06:51:02–07:27:12 EDT. The runtime binary, target/drafter,
capacity and chunks are the final ladder's checked configuration above.
Plain and DSpark use separate fresh processes and zero cached tokens.
The neutral fixtures place three release codenames in the source corpus
and ask for those facts without quoting the answers in the question.
All ten requests finish with `stop`, a terminal marker and a complete
stream; all three facts appear in the visible answer, with no error.

| Rung | Actual prompt tokens | Plain / DSpark output tokens | Visible retrieval |
| --- | ---: | ---: | --- |
| 8K | 7,594 | 82 / 82 | All three facts, both modes |
| 32K | 31,628 | 74 / 88 | All three facts, both modes |
| 64K | 64,492 | 163 / 163 | All three facts, both modes |
| 128K | 128,744 | 164 / 164 | All three facts, both modes |
| 256K | 258,779 | 74 / 74 | All three facts, both modes |

Each request permits at most 1,024 outputs; a completed shorter answer
passes this retrieval check and is not a 512-output timing measurement.
Peak sampled memory drops are 98.068 / 108.715 GiB. Raw results are
`~/scratch/m3-extrapolation-ds/final-retrieval-262k-{plain,spec}/run.json`
on `spark-b`, using the measurement harness pinned above. These checks
do not substitute for the fixed-bound oracle or full-window PPL. The
separate completed 1M retrieval request is recorded below.

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
bytes with DSpark. The image is `llmp-llamacpp:b11254-cuda13`, source
`8019dc563b1ecbae6b161a70c3a1359f1b206c1e`, all target/drafter layers
on GPU, flash attention on, context 1,048,576, one slot, automatic fitting
off and prompt cache disabled. Target and drafter are the same GGUFs as
the native comparison. Chat rendering accounts for the 89-token difference
from the earlier native prompt; token counts are retained separately.
These captures precede the harness's terminal-marker/finish-reason fields;
those fields are not retroactively inferred. Successful captures and
512-output counts are retained in
`~/scratch/m3-extrapolation-ds-b/llama-1m-{plain,spec}/run.json`.
## Final DeepSeek runtime at one-million-token capacity

The final HCA/frontier runtime on `spark`, job `m3-final-ds-1m` started
2026-09-30 05:23 EDT, completes both plain and DSpark timing requests.
Each processes 1,038,047 prompt tokens from the same canonical fixture
as the reference above and produces all 512 outputs. Both start fresh,
cache zero prompt tokens, finish with `length` and a terminal marker,
and complete their streams without errors. Native chat rendering adds
89 tokens relative to the reference; this is not an equal-ID comparison.
Both use `context = 1048576`, F16 caches and 2,048-row chunks.

| Mode | Prefill seconds | Prefill tok/s (× reference) | Decode tok/s (×) | Peak drop GiB (×) |
| --- | ---: | ---: | ---: | ---: |
| Plain | 3,542.776 | 293.004 (2.179×) | 16.545 (2.013×) | 103.911 (1.012×) |
| DSpark | 3,558.859 | 291.680 (2.272×) | 31.750 (1.882×) | 114.847 (1.009×) |

All rates exceed the matched same-GGUF reference; both memory ratios
pass the approximately 1.1× bound. Minimum available memory is
14,291,140,608 / 2,565,365,760 bytes. Registered fixed allocations are
2.35 / 2.51 GiB, including workspace 2.21 / 2.37 GiB. Host chunk
inputs need an additional 0.04 GiB; execution budgets are
110.57 / 110.58 GiB.
The prompt and output leave 10,017 tokens below the configured ceiling.
Timing fixtures are not neutral retrieval evidence. The separate 1M
completed-answer retrieval passes below; the maximum saved-state swap
also passes the direct validation recorded below.
The residual slope and watchdog floors still require completed-chunk
observations rather than these mean rates.

Runtime SHA-256:
`7dd83ca9ceb0b077268b8448f6124ca7760425afd8ebb68b177eb9935b917770`;
the checked kernel source is the same final HCA/frontier body as the
262K ladder. Harness SHA-256 is the same as above. Raw native results
are `~/scratch/m3-extrapolation-ds/final-new-1m-{plain,spec}-timing/run.json`
on `spark`; reference captures and input/source pins are recorded above.

### Final DeepSeek neutral retrieval at 1M capacity

The same supervised job completes successfully at 08:22:38 EDT on
2026-09-30 after its separate DSpark retrieval request. The neutral fixture
contains 1,037,970 native prompt tokens, with no cached tokens, and permits
1,024 outputs. All three release codenames appear in the visible answer;
the response naturally finishes after 44 outputs, with `stop`, a terminal
marker and a complete stream without errors. This is a completed retrieval
pass, separately from the two 512-output timing requests.

Prefill takes 3,546.938 seconds (292.638 tok/s); decode is 30.682 tok/s
on this short answer. Peak sampled memory drop is 114.568 GiB, with
2,871,328,768 bytes minimum available. Runtime, artifacts, chunks and
measurement harness are the final 1M configuration pinned above. This
request also completes through the production HTTP progress watchdog;
its mean rates do not establish a bound on individual completed chunks.

The neutral prompt content SHA-256 is
`a4321cf9a22cc5a384905df8b394d88d5c1dafe35dfcbe51b61171c575a98015`.
Raw results are
`spark:~/scratch/m3-extrapolation-ds/final-new-1m-spec-retrieval/run.json`,
SHA-256 `4d84c3c47c694041f4ac4d01d9da901734807d191536fbce3d4cc15e3facefe2`;
the separate response capture retains the visible answer and reasoning.
The separate maximum-length saved-state continuation passes below.

## Continuing-context swap protocol

`llmp_long_swap` uses the production runtime and runners. It warms the
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
| DeepSeek + DSpark, long | 262,144 | 131,072 | 64 | Pass: repeated prefill and every returned token/logit exact |
| DeepSeek + DSpark, maximum | 1,048,576 | 1,048,512 | 64 | Pass: repeated prefill and every returned token/logit exact |
| Qwen + MTP, long | 262,144 | 131,072 | 64 | Pass: both prefix and selected heads, repeated prefill and returned tokens/logits exact |
| Qwen + MTP, maximum | 262,144 | 262,077 | 64, plus three reserved draft positions | Pass: both heads, repeated prefill and returned tokens/logits exact |

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

### Final DeepSeek 128K saved-state result

On `spark-b`, 2026-09-30 07:37:42–07:47:20 EDT, supervised job
`m3-final-ds-128k-swap-v3`
completes successfully with the final HCA/frontier body. Both independent
131,072-token prefills and all 64 continued tokens/logits compare bit
exactly, including after the prepared Qwen round trip. The final prefill
logit row's SHA-256 is
`693c162cd81b493df0325f62206dd4e66b6550c5972136edf9af614220edcb22`.

| Measurement | Result |
| --- | --- |
| Control / repeated prefill | 266.822 / 268.468 seconds; 64 chunks each |
| Longest completed chunk | 4.641 / 4.653 seconds |
| Uninterrupted / restored decode | 1.637 / 1.622 seconds for 64 outputs |
| Away / back activation | 7.601 / 9.458 seconds |
| Return state restore | 0.143 seconds |
| Spilled / initialized state | 1,128,267,776 / 1,125,269,504 bytes |
| Peak sampled memory drop / minimum available | 116,553,003,008 / 9,155,997,696 bytes |

The partner uses the original Qwen artifact, plain decode, 512-token
capacity and 384-row chunks. No turn checkpoints are retained. These
activation totals are swap parts, not the separate first-token endpoint
measurement. The observed chunk durations are well inside the current
181.44-second prefill allowance, but this harness does not exercise the
HTTP watchdog or prove its maximum-context case.

Harness SHA-256:
`c9e7a3b63ccb8c32121aa2774fef47347829fba8b8b37a81383c29c22a8e7ffd`.
The raw UTF-8 fixture SHA-256 is
`d1e6ac95a35fa0e27b9fe66505d85555c90dbf0c044e5458124b68a6bb2b80c9`;
native encoding supplies the first 131,072 IDs without chat rendering.
Raw summary, logs and input pins remain in
`~/scratch/m3-extrapolation-ds/final-128k-swap/` on `spark-b`.

### Final DeepSeek maximum saved-state result

On `spark`, 2026-09-30 15:13:58–17:16:04 EDT, supervised job
`m3-final-ds-max-direct` completes with native exit zero and a successful
process/memory retirement check. This directly supervised run has no
profiler. Two independent 1,048,512-token prefills and every one of the
64 continuation tokens and complete logit rows compare bit exactly,
including after the prepared Qwen round trip. It uses DSpark, F16 caches,
2,048-row chunks and 1,048,576 capacity on the unchanged production path.

| Measurement | Result |
| --- | --- |
| Control / repeated prefill | 3,683.025 / 3,585.332 seconds; 512 chunks each |
| Longest completed chunk | 14.722 / 11.413 seconds |
| Uninterrupted / restored decode | 3.505 / 3.086 seconds for 64 outputs |
| Away / back activation | 14.442 / 9.911 seconds |
| Return state restore | 0.583 seconds |
| Spilled / initialized state | 7,331,643,392 / 7,328,645,120 bytes |
| Peak sampled memory drop / minimum available | 122,689,716,224 / 3,179,405,312 bytes |

The longest observed chunk is inside the current 181.44-second allowance.
Together with the completed maximum-capacity HTTP timing/retrieval and
Qwen chunk observations, this supports retaining the prefill/decode floors
of 100/5 tok/s and the 120-second stall interval. The swap harness itself
does not exercise the HTTP watchdog. Its activation totals are separate
from the first-token swap table; maximum saved state adds spill work.

Fixed allocations are 2,787,125,020 bytes, host chunk inputs 46,137,344
bytes and the execution budget 118,656,870,172 bytes. The prepared partner
is plain Qwen with 512-token capacity and 384-row chunks. No turn
checkpoints are retained. Both full-logit captures have SHA-256
`8229c9587f7ea672e67b51972c17e05332fee06428f03ea4bcd620c2ac36c329`.

Native harness SHA-256:
`ed2a9e146507a54ed39302d61f192057eb64b5715f7ae9af23d288c179d7c3d1`;
controller SHA-256:
`eae2bbb9ed9e5a67d2e7ec81c5fff126572ee0afcdc205d1b73e651effef4ba9`.
The raw UTF-8 input is the maximum fixture pinned above. Native summary
SHA-256 is
`b65e4067b814f771131e672190f3b15b1c74254cca54cea188e165cfa058a884`.
Results and completed receipt remain in
`spark:~/scratch/m3-extrapolation-ds/max-state-direct-final/`.
The earlier profiled attempt remains excluded under RE-039.

### Final Qwen 128K and maximum saved-state results

On `spark-b`, 2026-09-30 10:36:39–10:51:25 EDT, step two of job
`m3-final-qwen-and-table-v2` completes successfully. Four independent
processes cover prefix and selected heads at 131,072 and 262,077 saved
tokens, with adaptive MTP, 4,096-row chunks and 262,144 capacity. Both
prefills and every one of the 64 continuation tokens/logits are exact
in each case, including after the DeepSeek round trip. The partner uses
plain decode, 512-token capacity and 384-row chunks. No turn checkpoints
are retained.

| Head / saved tokens | Control / repeated prefill seconds | Longest completed chunk seconds | Uninterrupted / restored decode seconds | Away / back activation seconds | State restore seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefix / 131,072 | 58.381 / 58.350 | 1.922 / 1.893 | 1.375 / 1.436 | 8.945 / 7.744 | 0.364 |
| Prefix / 262,077 | 121.244 / 121.173 | 2.022 / 2.022 | 1.685 / 1.784 | 9.356 / 8.078 | 0.660 |
| Selected / 131,072 | 58.622 / 58.492 | 1.929 / 1.897 | 1.367 / 1.449 | 9.008 / 7.785 | 0.366 |
| Selected / 262,077 | 121.411 / 121.297 | 2.028 / 2.026 | 1.557 / 1.636 | 9.407 / 8.136 | 0.661 |

Each 128K prefill has 32 chunks; each maximum prefill has 65. Used/spilled
state is 4,857,044,992 / 4,859,101,184 bytes at 128K and
9,228,107,776 / 9,231,663,104 at the maximum, for either head.
Peak sampled memory drops range from 105,271,848,960 to 105,348,288,512
bytes, with at least 20,307,099,648 bytes available. These raw-ID
exact-state fixtures are not the canonical HTTP timing or neutral retrieval
prompts, so their prefill times do not establish those throughput rates.
Activation totals retain the separate endpoint limitation above.

The prefill logit SHA-256 is identical for both heads at each size:
`4db8d218546bca911d6509a011fbbaff5b35690b3aaf10f727f8cbf3d1b2f539`
at 128K and
`0abed21af08263834acaab6691673a6dd0c6e6c1f5a3eac617ec9636bce0f69a`
at the maximum. Harness SHA-256 is
`1cc15b612782efc1b52264724ab6cac7b227b4bfb2fb8fb80471bd53d318415c`;
the checked production kernels are `cf63418`'s unchanged default path.
Raw results are `spark-b:~/scratch/m3-final-qwen/swaps/`,
`{prefix,selected}-{131072,262077}.json` and their logs. Other steps of
the initial queue failed before loading a model and are excluded; this
step has its own successful exit and complete native records.

## Final Qwen HTTP timing and neutral retrieval

Job `m3-final-table-and-qwen-http-v3` completes rc0 on `spark-b`,
2026-09-30 10:56:55–11:33:48 EDT. Its second step takes 27 minutes and
starts six fresh runtime processes: plain, adaptive prefix MTP, and
adaptive selected MTP, separately for timing and neutral retrieval.
All use the same Qwen target, F16 caches/F32 recurrent state, 262,144
capacity and 4,096-row prefill chunks. The main kernels are unchanged
from `cf63418`; both negative expert/head prototypes are absent.

Each timing request has zero cached prompt tokens and completes all
512 outputs, with `length`, a terminal marker and no stream error.

| Actual prompt tokens | Plain prefill / decode tok/s | Prefix MTP prefill / decode tok/s | Selected MTP prefill / decode tok/s |
| ---: | ---: | ---: | ---: |
| 31,743 | 2,313.940 / 26.355 | 2,271.915 / 39.380 | 2,265.031 / 37.063 |
| 64,110 | 2,307.213 / 25.837 | 2,268.627 / 37.615 | 2,266.070 / 41.652 |
| 128,799 | 2,257.390 / 25.351 | 2,224.327 / 45.332 | 2,220.862 / 43.799 |
| 258,702 | 2,171.404 / 24.550 | 2,145.252 / 36.727 | 2,141.444 / 44.069 |

Peak sampled memory drops are 84.830 / 87.278 / 87.532 GiB. Minimum
available memory is 34,646,167,552 / 32,050,442,240 / 31,806,234,624
bytes. These are single-run native observations; the quoted-answer
timing fixture does not qualify retrieval. The
[fresh same-ID Mia ladder](../qwen38-mtp-speed/README.md#fresh-two-pass-mia-ladder-2026-09-30)
completes two full passes. Its decode observations are 42.234 / 38.622
at 32K, 39.702 / 39.645 at 64K, 39.447 / 41.347 at 128K, and
42.885 / 38.766 at 256K. The default prefix head beats both at 128K
but falls below both at 64K and 256K; the selected head beats both at
those three rungs but falls below both at 32K. Neither closes every
depth's speed gate, so matched-piece optimization continues.
That recipe's fixed depth three and FP8 KV/BF16 recurrent state differ from these native
cache settings; neither a favorable seed/history nor lower bit depth
alone classifies a quality tradeoff.

All fifteen neutral retrieval requests complete naturally with `stop`,
a terminal marker and no stream error. All three facts appear in the
visible answer, in every mode, with zero cached prompt tokens.

| Rung | Actual prompt tokens | Plain / prefix / selected output tokens |
| --- | ---: | ---: |
| 8K | 7,517 | 271 / 219 / 219 |
| 32K | 31,674 | 272 / 274 / 274 |
| 64K | 64,041 | 271 / 270 / 270 |
| 128K | 129,037 | 172 / 221 / 221 |
| 256K | 258,633 | 170 / 170 / 170 |

Each request permits 1,024 outputs. A complete shorter answer passes
retrieval; these rates are not 512-output timing measurements or the
fixed-bound oracle/PPL quality comparison. Retrieval peak drops are
84.840 / 87.245 / 87.497 GiB. All requests complete through the production
progress watchdog. The saved-state runs above separately measure completed
chunks up to 2.028 seconds, well inside its 242.88-second allowance for
4,096 rows; with DeepSeek's completed maximum-chunk check above, this
supports retaining the existing floors.

Runtime SHA-256 is
`757c29452d26bfbd0fcc07686d0a13a632eef0850621be68e02ad7144886eaf0`.
The HTTP harness is the pinned `340f7d2a…` helper above, with
`baseline.py` SHA-256
`41c693de64e2d63b32622cb03f29024d76455c0ad7efac4394ea3f5d45f4b1e2`
and its `prompts.json` fixture SHA-256
`c697236c56a09b0a3f2550f7514b3e4d826e1d14a96b4d1c79e3bd33a3a6f859`.
All three were pinned and their import checked before loading. Raw results
are `spark-b:~/scratch/m3-final-qwen/{plain,prefix,selected}-{timing,retrieval}/`.
Earlier rejected old-helper captures are preserved separately and excluded.
The full source audit covered the 449 measured source/build/dependency/tool
paths; header/license exports alone are not that audit.

## Remaining final-path checks

- Native ladders and neutral retrieval above are complete on their pinned
  paths, as is the fresh same-ID two-pass Mia reference. Finish the Qwen
  matched-piece optimization before closing its remaining speed gate.
- Final frontier-head and Qwen kernel controls retain their original
  oracle/PPL bounds, seeded sampling rules and exact own-path rollback,
  repeat and swap requirements. Qwen's selected vocabulary remains a
  separate measured option; the default prefix head must meet its gate.
- DeepSeek's 128K/maximum and both 128K/maximum Qwen heads pass the
  continuing-context protocol above.
  The attempted profiled maximum run produced a completed bounded trace
  but no complete native result or retained application exit status
  ([RE-039](../../rough-edges.md)); it is excluded and superseded
  by the successful directly supervised validation above.
- Retain watchdog floors based on the actual longest chunks and completed
  maximum-capacity routes above. Current defaults are
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
