<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 3 prepared import

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
identity changes, and no model execution support is established here.

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
`~/.local/share/jitllm/gemma3-import-20261007/artifacts/`. Bounded C1 execution,
independent-slot state, restore, quality, performance and batching
qualification remain open.
