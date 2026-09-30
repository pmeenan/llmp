<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Upstream log

One file per upstream project. Each file lists the fixes, limitations and
patches jitLLM has found or made in that project, so the owner can start a
separate effort to upstream any one of them. **Every entry stands alone.**
A new conversation that has only the entry must be able to act on it. So
each entry names the repository, the pinned version, the files and lines,
and the repro, and does not rely on the rest of this file.

| File | Project |
| --- | --- |
| [ggml.md](ggml.md) | GGML as vendored in llama.cpp (D-077), plus llama.cpp's own reference findings |
| [cuda.md](cuda.md) | NVIDIA: CUDA driver and toolkit, cuBLAS, cuFile, the GB10 and DGX OS |
| [cutlass.md](cutlass.md) | CUTLASS (Qwen3.8's NVFP4 and MXFP8 GEMMs) |
| [exllamav3.md](exllamav3.md) | ExLlamaV3's kernels (the native EXL3 linear) |
| [vllm.md](vllm.md) | vLLM, as Mia's Qwen3.8 fork and recipe run it (an M3 baseline) |
| [flashinfer.md](flashinfer.md) | FlashInfer's pinned CUTLASS NVFP4 consumer, inspected for Qwen component transfers |
| [tensorfold.md](tensorfold.md) | TensorFold (an M3 baseline and a source of techniques) |
| [ds4.md](ds4.md) | Entrpi's ds4 fork (same-GGUF DeepSeek baseline and prefill techniques) |
| [other.md](other.md) | Projects with a single finding: Docker, SGLang, qemu-user with LeakSanitizer, Ubuntu's snapshot service |

## Upstream entries and rough edges

A [rough-edges.md](../rough-edges.md) entry (RE-NNN) is the jitLLM-side
finding: what bit us, and how jitLLM works around it. The upstream entry is
the handoff: what upstream's current code does, what exists upstream
already, and what to send. It links its RE entry and does not repeat the
investigation. Not every RE needs an upstream entry: one with nothing to
send and nothing to track upstream (a tool used wrongly, behaviour by
design) has none. An upstream entry may also have no RE, for example a
patch jitLLM carries or a change upstream worth adopting.

When an agent logs a rough edge in a third-party component, or adds or
changes a patch under `third_party/patches/`, it adds or updates that
project's entry here in the same unit of work
([workflow](../workflow.md)).

## Status

- **open**: not reported upstream.
- **reported**: an issue or discussion is open upstream.
- **PR open**: a pull request is open upstream.
- **fixed upstream at \<ref\>**: upstream's code no longer has the problem.
  jitLLM may still carry the workaround until its pin passes that ref.
- **won't fix**: upstream behaviour by design, or not worth sending. The
  entry says why.
- **carry**: a jitLLM-specific change that stays in jitLLM's patches.

## Entry template

```
## <title> (RE-NNN)

- **Status:** open | reported | PR open | fixed upstream at <ref> | won't fix | carry
- **Found:** <date>, <pinned version or commit>, <host>
- **Problem:** what goes wrong, with file:line at the pin, and a minimal
  repro if there is one.
- **jitLLM's workaround:** the patch or check (file path), and its cost in
  speed, correctness or maintenance.
- **Upstream master:** what the current code does (checked <date> at <commit>).
- **Upstream refs:** issues and pull requests that already exist.
- **Proposed action:** what to send or do, and roughly how much work.
- **Links:** RE entry, experiment reports, decisions.
```

## Contributing to llama.cpp

llama.cpp's
[CONTRIBUTING.md](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
forbids AI-written issue and pull-request descriptions and requires AI-written
code to be disclosed. A new contributor may have only one pull request open at
a time. So for llama.cpp and GGML:

- **The owner writes the report text.** Entries give facts, a repro and a
  pointer to the diff, not ready-to-paste prose.
- **Code changes stay small enough to review by hand.** Send them one at a
  time, in the order ggml.md suggests.

Other projects have no such rule on record. Check each project's
contributing guide before sending anything.
