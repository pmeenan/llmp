<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 C2 on llama.cpp v0.6.0

The first representative new-release C2 screen passes both strict quality gates: zero positive-reference-margin choices, zero ties and −0.148562% relative conditional loss across 64 targets. 64 of 66 complete heads are byte-exact; the analyzer did not count exactness by phase. Short matched bookends place native elapsed 1.71513% above stock. This completes a bounded screen, with broader quality, context and sustained performance still open.

[Results](results.json) contain exact operands, helpers, runtime receipts and eleven successful supervised jobs/17 successful steps on spark-b, 2026-10-06. The [protocol](PROTOCOL.md) is the immutable preregistration text; its unexecuted/held statements describe the pre-run registration. Completed status is recorded here and in results. Historical helpers, image 837/b29 comparisons and their failures retain their original pins.

## Work and gates

Both owners use supplied 992-token prefixes and 32 forced IDs each from the same complete 12×1024 I32 carrier. Both engines prefill in 256-column chunks with a 224-column tail, then execute one physical C2 group per wave. Native context 4096/max_rows 256/headcap 2 uses ordinary products and the checked norm/ROPE/ADD recipe. Stock has total context 8192/per-sequence 4096/batch=ubatch 256, F16 K/V, normal local 1280/global 4096 caches, separate streams, default graphs/fusion, and no full sliding-window cache override. Both publish 66 whole-vocabulary F32 heads, including the two frontier heads. No target features are retained.

Native first/repeat full heads, inputs, four initialized-state witnesses and two layouts match exactly, before FIRST d812 stock exposure. Stock first/repeat complete heads and inputs then match exactly. All heads are finite and every published argmax/forced-ID record is admitted. There is no cross-engine KV-byte claim. The unique selected-plan record describes the first warm decode, before reset; the paid summary separately admits 32 completed waves and 60 owner-attention/121 norm/120 norm-ROPE/120 norm-ADD steps.

The fixed strict gate permits no positive-margin choice differences; exact ties are reported separately. Whole-vocabulary FP64 conditional NLL covers 64 targets: retained row 0 predicts 992 and rows 0..31 predict 992..1023. Completed 1024 row 32 is unscored. Relative conditional loss must independently be ≤3%; no noise calibration or inherited bound applies. Means are 10.5293372315 native and 10.5308239553 stock. These conditional scores do not qualify full-corpus perplexity or natural generation.

## Short matched elapsed

| Engine | First (s) | Repeat (s) | Mean (s) | Spread (ms) |
| --- | ---: | ---: | ---: | ---: |
| Stock | 3.28255 | 3.27672 | 3.279635 | 5.83 |
| Native | 3.33447 | 3.33730 | 3.335885 | 2.83 |

Order is stock→native→native→stock after the quality pass. The 56.25 ms mean gap is larger than these individual spreads, but two observations per engine do not establish sustained performance or causation. Every complete timing head/input/native state/layout file matches its independently frozen engine proof. Paid time includes 32 completed physical C2 waves, full head publication and argmax; loading, prefill, warm/reset, frontier/final state export and post-run finite/SHA checks are outside elapsed. This is not an end-to-end prefill or memory-peak measurement, and native does not meet elapsed parity in this short screen.

## Source, runtime and reproduction

The external [release reference](../llama-reference-refresh/README.md) uses official v0.6.0/d812350/b11429 and immutable ARM64 CUDA image c604. The native [GGML refresh](../ggml-release-refresh/README.md) provides immutable helper 0d6d0952 with measured build receipt 7d8f1847. Canonical receipt 5ab70ad5 is metadata-only: all fields agree except source_lock; it did not relink the binary. The native build version is 0.1.0-dev+unknown, with exact source and binary identities separately retained. SDK installation 7b8167cd is distinct from either build receipt. The newly compiled public C2 helper a3122de5 preserves the historical [caller](../gemma-joined-serving/llama_joined.cc) 4786817b unchanged.

The model process proves successful CUDA initialization, a current context, ordinal 0, one GB10, sm_121/48 SM and driver API 13040. The actually loaded driver path is /usr/local/cuda-13.4/compat/libcuda.so.615.71.09, authenticated against the image closure; ldd and no-argument usage are not substitutes for this proof. All four stock containers retire with checked absence. NVCC 13.4.92, image CUDA runtime 13.4.49 and cuBLAS versions 13.8.0.4 (native)/13.7.0.27 (image) name different components.

[llama_c2.cc](llama_c2.cc)/[build.sh](build.sh) compile and refuse missing arguments before contexts; [run.sh](run.sh) supplies the fixed recipes and owned retirement, and [analyze.py](analyze.py) implements source/metadata/finite-own/strict/timing admission. Supply the externally retained approved checkpoint/artifact and full carrier with the results' identities; use fresh owned output roots and the installed supervisor. The raw release-c2 source.json, protected closure, pipeline templates and producer admissions are retained under gemma-release-c2-raw; future literal successful-producer/own bindings keep those templates' method unchanged. No raw heads, states, IDs or logs are in Git. Fixed DriverProof metadata adds no device scratch or pointer table; existing native publication grant 73,515,008 B and initialized-state export funding remain the helper contract. No unchanged peak claim follows.

TensorFold's task-entry primary receipt fixes cb2ebf0540f42604e2759b2ddef497861e928248/version 0.6.6 at 2026-10-06T17:07:44.779278Z. Its observed Gemma26 MLX/DFlash recipe is not a matching CUDA GB10/GGUF comparator for Gemma31. No TensorFold workload was run. This C2 result does not transfer to 26B, scalar/C1/C3/larger cohorts, other contexts, serving defaults, speculation, swaps or general supported status. No engine/default changes or full-suite qualification are part of this unit.
