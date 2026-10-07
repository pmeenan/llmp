<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 3 prepared import and bounded native execution

The [approved pin](pins.json) names the exact 4B QAT Q4_0 source from
[the Gemma3 foundation](../../gemma3.md). The existing generic
`artifact-layout/import_m3.py build` path prepares its F32/Q4_0/Q8_0
weights with the unchanged v0 writer. Gemma3 preflight is closed to this
one source identity: its typed profile metadata, vocabulary count and all
444 tensor descriptors must match the checked factual fixture. Its complete
header digest must also match before planning. The actual approved table has
separate V matrices and no independent output head; the native binding aliases
the embedding at execution time.

The importer still authenticates the whole source against the pin, reparses
its metadata and descriptors, rechecks source bytes during writing, deep
verifies the prepared artifact and publishes atomically. A header digest or
preflight success alone does not authenticate weight payloads. Other GGUF
architectures keep the existing generic path. No artifact schema or writer
identity changes. Import alone establishes no model execution support.

On a Spark, after downloading the exact-revision source into `SOURCE` and
checking its size and whole-file SHA-256 against `pins.json`:

```sh
python3 docs/experiments/artifact-layout/import_m3.py build \
  STORE docs/experiments/gemma3-execution/pins.json \
  gemma3-4b-qat-q4_0 SOURCE/gemma-3-4b-it-qat-Q4_0.gguf
python3 docs/experiments/artifact-layout/import_m3.py verify STORE/ARTIFACT_ID
```

Run downloads, import and verification through the installed `spark-job
start --gpu` supervisor, with bounded timeouts and its official `wait`.
The checkpoint repository is `ggml-org/gemma-3-4b-it-qat-GGUF`, revision
`bbcac0d065076c47042838c0675c602411b0dd4c`. Checkpoint terms are `gemma`
and remain informational under D-087. Checkpoints and prepared model artifacts
stay in the external model store; this change ships no weights in the core.

The seven `test_gemma3_import` controls require no model payload. They check
recipe identity, every typed metadata field, the actual parser's compact
vocabulary count, the complete
tensor table and tied-head contract, source length/pin/header/shard refusals,
refusal before output publication and the unchanged generic architecture path.
The existing importer controls retain whole-source hash/publication coverage.
On 2026-10-07, Spark A downloaded and locally hashed the full approved
2,526,080,992-byte source, prepared it and passed an explicit independent deep
verification. The artifact ID is
`8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb`;
its converter remains `m3-1+layout-a0d1980a9eddd1ad`. It contains 444 resources,
36 groups, 1,226 chunks and one 2,519,662,592-byte safetensors shard. The
19 focused importer controls passed on both the workstation and Spark A.
Official supervised jobs `m35-gemma3-download` and `m35-gemma3-import`
completed with exit 0; their logs and verification receipt remain external.

The source lives under Spark A's
`~/.local/share/jitllm/models/ggml-org/gemma-3-4b-it-qat-GGUF@bbcac0d0/`.
The prepared store is
`~/.local/share/jitllm/gemma3-import-20261007/artifacts/`.

## Bounded native own control

On 2026-10-07, `jitllm_gemma3_probe` executed that artifact on Spark A
(GB10, driver 580.178.04, SDK `aarch64-c09daba6ac31edee`, core profile,
locked GGML tree `d50cb7f97867b4c5…`). This is an internal C1 probe at
context 4096, F16 K/V, 128-row chunks and one published head row. It consumes
256 prompt rows, three supplied scalar warm rows and 32 teacher-forced rows.
The actual native tokenizer produced the first 291 IDs from an authored prose
input; their 1,164 bytes have SHA-256
`3f94d579e6749a32aee252d01562102fcd835bdb231e1620229b77d0d7308753`.
The source text SHA-256 is
`3016ddb3960e77f9a9dc649bca88dad0a7578d8afb07dd8ecf00b98144dc3756`.
The text and IDs stay in external scratch; no user prompt or model payload is
checked in.

All 33 full 262208-vocabulary heads are finite and byte-identical between eager
execution and two graph runs. Their complete-file SHA-256 is
`ff2cf41c82b19f6996f1c646a8cb3fcd96039bebc87058c6363ae6c9a81b5324`.
Each graph teacher run captures once and replays 33 times, with no graph refusal
or coverage violation. Clear preserves 68 resident state extent identities;
full versus state-only prefill produces byte-identical heads and initialized
state hashes. Malformed token, past and head-capacity requests preserve the
completed prefix and initialized state. Spill/restore preserves the logical
positions and exact initialized-state hash. All 32 pre-step choices and final
heads also match the lifetime run; every probe retires successfully.

The measured envelope is 23,068,672 activation bytes, 10,485,760 scratch bytes,
4,194,304 host-input bytes and 1,341,520 plan-floor bytes. The probe separately
funds an 8 MiB caller floor and, for the lifetime control, an 8 MiB pinned state-copy
buffer. This screen runs ordinary primitives; generic norm and quantized FFN
switches remain explicit diagnostics.

The 13 focused Gemma3 foundation/graph/plan tests pass, including the regression
that actual unmasked host inputs refuse before launch. Workstation changed-file
formatting, REUSE/header 1795 and portability boundary 411 checks pass. The initial
probe construction error and an overly broad CTest selector are retained in
external official job records; their narrow corrections preserve all controls.
The final focused tests and own controls are repeated in
`m35-gemma3-runner-final`. The initial own-control binary SHA-256 is
`6e52c84ea3eef1af2065ebece5bd76a2dba77c6036844253ffe8178f2b0b96ec`.
The final binary adds selection counters and has SHA-256
`bb7902b05b4a1859d66993db1c1350b6ba9a8538017fb1fb595e424a498f3fd4`,
with native receipt SHA-256
`cf6bce22c79008fc49245a74206f778b036a4efb6b69ae4f7aa1f572fb6469f5`.
Full source identities, raw heads and logs remain external.

These dimensions establish a representative own control, not new supported
context limits, batching, sustained performance, a serving route or media
support. Reference quality/performance qualification remains open.

## Current stock first screen: quality remains open

The same Spark runs the original llama.cpp v0.6.0 public API from commit
`d81235049384534c167caea52b85a694f6103d14`, image
`ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db`.
[`llama_probe.cc`](llama_probe.cc) keeps stock fusion and graphs, explicitly
matches context 4096/C1/F16 K/V/128-row batches, and independently reproduces
the native 291-ID prefix from the same text. Its first 128-row chunk is state-only,
the second publishes one full head; three supplied scalar rows precede 32
teacher-forced rows. Two stock runs have identical finite 33-row files, SHA-256
`01aaccb8c1b50538d9499489c385a7005cf2f68a5d3d941c768502fae0c527dd`.
All contexts, models and owned containers retire before the queue continues.
The native own freeze completes before the first stock evaluation.

[`analyze.py`](analyze.py) reuses the existing Gemma quality analysis arithmetic.
It scores targets 259..290 against the first 32 heads and leaves the final 33rd
head unscored; all 33 rows participate in finite, distribution and greedy checks.
The first-screen gate requires zero positive-margin greedy differences and
`expm1(mean_target_nll_delta)<=0.03`, with exact ties reported separately and
no inherited numerical allowance. This is a 32-target screen, not corpus PPL.

| Primitive native versus stock | Result |
| --- | ---: |
| Positive-margin greedy differences / full rows | 1 / 33 |
| Exact-tie differences | 0 |
| Miss row / native ID / stock ID | 23 / 15808 / 28251 |
| Stock margin at that miss | 0.2225341796875 |
| Native / stock mean target NLL | 3.8654058210 / 3.8594656475 |
| Mean target NLL delta | +0.0059401735 |
| `expm1` of mean target NLL delta | +0.595785% |
| Mean / maximum total variation | 0.03453917 / 0.08167455 |
| Maximum raw logit difference | 0.72173023 |

The strict greedy gate **fails** despite the likelihood screen remaining within
3%. The matched cycle timing queue therefore did not run; this unit makes no
performance or stock-parity claim. The separate norm-only and quantized-GLU-only
teacher repeats remain byte-exact to the primitive heads and preserve the same
quality miss. Runtime plan counters establish actual selections:

| Explicit native policy | Bound plans | Selected RMSNorm/Mul steps | Selected quantized GeGLU steps |
| --- | ---: | ---: | ---: |
| Primitive | 3 | 0 | 0 |
| Norm only | 3 | 610 | 0 |
| Quantized GLU only | 3 | 0 | 35 |

These cumulative counts describe successfully bound plans, excluding Setup's
envelope probes; they do not count kernel execution or graph replay. All three
counter-instrumented teacher runs preserve the frozen primitive head, choice
and final-head bytes. Both switches remain explicit diagnostics and do not fix
quality. The source-backed next lead is the eligible D256 norm/Mul/RoPE chain;
width 2560 norm/Mul/ADD would require a deliberate checked gate extension.
The supported execution matrix remains unchanged.

TensorFold's per-task current source was checked on 2026-10-07 at
[`ed78d6fc204d89d90b045bf033d6551e7714f3a1`](https://github.com/ashhart/TensorFold/blob/ed78d6fc204d89d90b045bf033d6551e7714f3a1/README.md)
(release 0.6.6). Its documented model table lists Gemma 4 through MLX; there is no
matching documented Gemma3 CUDA/GGUF recipe for this screen. This makes that
comparator ineligible here, without claiming that all Gemma3 execution is
unsupported by TensorFold.
