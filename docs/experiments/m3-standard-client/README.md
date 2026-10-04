<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3 standard-client gate — 2026-10-04

The unchanged OpenAI Python SDK **3.3.1** completes DeepSeek → Qwen →
DeepSeek through one production runtime process, then a streamed Qwen reply.
All four responses stop naturally with visible answer `Python`. Their
visible text, reasoning and prompt/completion counts equal the independent
native command controls, including DeepSeek after its return. This passes
M3's standard-client chat-and-model-switch criterion, without qualifying
numerical quality, concurrent throughput or the image-inclusive swap table.

Spark A (`spark-c4e2`), driver 580.178.04, CUDA 13.4.92, SDK
`aarch64-e0a0c85c42806fb1`. The implementation is the final checked
literal-cohort runtime at `17d577e`; subsequent commits through `2df115d`
record experiments and documentation. The source inventory matches main's
487 production/build/source-lock files. The production source union passed
1,543 tests on Spark B; the refreshed Spark A runtime target builds cleanly.
No runtime implementation changes for this gate.

Both the native `chat --fresh` control and service use independent fresh
anchors/data, original DeepSeek 0731 plus DSpark and Qwen NVFP4 plus prefix
MTP, greedy sampling, normal stops and output budget 256. DeepSeek context
is 262,144, with `wave_form = "speculative"`; Qwen context is 33,792 with the
65,536-entry prefix head. Both use 4,096-row prefill chunks and max slots
four. There is no imported calibration or execution/numerical environment
override. Each client sends exactly this frozen user message:

> Which programming language uses 'def' to define a function? Reply with just
> the language name.

| Request | Prompt tokens | Completion tokens, including stop | Visible answer | SDK wall s | Cached prompt tokens |
| --- | ---: | ---: | --- | ---: | ---: |
| DeepSeek first | 23 | 121 | Python | 12.598 | 0 |
| Qwen | 71 | 44 | Python | 9.354 | 0 |
| DeepSeek returned | 23 | 121 | Python | 13.094 | 21 |
| Qwen streamed | 71 | 44 | Python | 9.088 | 66 |

Times include switching and delivery, and are single functional-gate samples,
not a performance comparison. The service becomes ready in 2.766 seconds;
the three native controls take 41.608 seconds together. Native first/returned
DeepSeek prompt IDs and generated IDs match exactly, with speculation and
actual drafted tokens confirmed. The API can reuse earlier checkpoints;
these are not all cold-state comparisons. Cached counts are retained but
excluded from answer/accounting equality. Total usage equals prompt plus
completion counts for every response.

The native control's generated IDs are decoded by the actual native tokenizer
and `StreamDecoder`, respecting the prompt's initial thinking state,
`<think>`/`</think>` transitions and the API's leading newline/space rule
following reasoning. The terminal stop is hidden from text but remains in
token usage. Comparing raw command text to visible API content would omit
this channel contract. This is token/text transport agreement, not complete
logit equality.

The SDK uses `max_retries=0`. All requested/returned model aliases and
single index-zero choices match. Qwen streaming explicitly requests usage:
19 actual SDK chunks have stable ID/creation/model, exactly one initial
assistant role, one natural finish and one final usage-only empty-choice
chunk, with no data after finish. Terminal SDK iteration is recorded.
Visible text, the retained `reasoning` extension and usage equal the
nonstream Qwen control. No raw SSE DONE marker is inferred from SDK objects.

The client runs in the cached pinned image
`sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`,
Python 3.12.3, under the ordinary CPU container runtime with one CPU,
512 MiB memory, read-only root, no capabilities or GPU assignment. The
container user matches the output directory owner. All 1,521 SDK Python
source hashes match the prior package-only probe; no torch, vLLM or
transformers module is imported. The image supplies a client package only.
The runtime binary, source map, linked-library hashes and actual mapped
paths, artifact manifests/indexes/kept metadata, tokenizer/template,
controller, client and decoder-helper identities are authenticated. Stable
identity dictionaries match before and after work.

Both native processes return zero and are reaped without forced kill.
The client returns zero and is reaped, its owned container is absent, and
strong 105-GiB admission/retirement probes pass around both phases. The
installed GPU-supervised `final-standard-client-gate3` job finishes
successfully and is waited on. Two excluded setup attempts remain recorded:
`screen1` could not inspect the non-dumpable runtime's process mappings;
`screen2` completed the native controls and first API answer but the client
could not save its output under the image's default user. Administrative
read access for mappings and an explicit client user/write probe correct
those harness failures. Neither supplies a gate-pass claim.

Raw receipts, complete responses/chunks, command prompt/generated IDs,
logs and reproduction sources are external at `~/scratch/m3-client-gate/`
on Spark A and `/home/pmeenan/scratch/jitllm-m3-client-gate-2026-10-04/`
locally. `screen3` is the qualified run. For reproduction, use its pinned
`run.py`, `client.py` and native `channels` helper with a fresh output path,
under the installed GPU supervisor, then wait. Saved runtime data/spill files
remain on the Spark; the local evidence copy excludes them.

| Identity | SHA-256 |
| --- | --- |
| Native runtime | `f1ee73b98a64e20bee69e0e689156b3b43c4b9b67b861ae45d805cd96e5a21ad` |
| Production source inventory | `62b29cbd9f576cd44d94d5451534d055caac184030e75d0e99fa5640dc958339` |
| Controller | `d35ec1ec22233e97dfbadf7f1cc023e59e788b0f5a17726e0706a5023066bfa9` |
| SDK client | `f1a8a626dc596c878f2d14bf01b8f5e2e23b0b0515721a0d029dda961ebaf8f7` |
| Native channel helper | `b318f4f16e2f5cb384b0090338d6de69e761b60cfe899045f2588269bb88aa64` |
| Authenticated file pins | `daabc22222058c58a452852029d5bf4d5b1688325537da070917e4b06a679c81` |
