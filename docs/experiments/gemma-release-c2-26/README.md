<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 C2 on llama.cpp v0.6.0

The representative 26B-A4B transfer passes the unchanged strict and conditional
quality gates: zero positive-reference-margin choices, zero ties and
+0.040572% relative conditional loss across 64 targets. 64 of 66 complete heads
are byte-exact; exact phases were not separately counted. Native elapsed is
2.153583% slower in the short matched bookends. This completes a bounded C2
screen, without adopting engine policies or serving defaults.

[Results](results.json) bind the exact sources, operands, binaries, receipts and
ten successful installed-supervisor jobs/13 successful steps on spark-b,
2026-10-06. The [protocol](PROTOCOL.md) is immutable preregistration text:
its unrun/held statements describe registration, while completed status lives
here and in results. The completed [31 screen](../gemma-release-c2/README.md)
and every historical b29/image837 helper/result retain their original pins.

## Matched work and independent quality

First two histories from the complete 12×1024 I32 carrier each supply a
992-column prefill under max_rows/ubatch 1024, then 32 forced owner-specific
steps (IDs 992..1023). Native uses context 4096/headcap 2, ordinary products,
requested real owner attention and the existing all-norm/MoE recipe. Actual
warm decode reports owner 30/norm 121/ROPE 60/ADD 90/route 30/reduce 30 with
2 rows/2 segments; both paid summaries admit 32 physical groups/replays.
The selected record is from warm construction before reset, not first paid.
Stock uses physical C2, total context 8192/per-sequence 4096, batch=ubatch 1024,
F16 K/V, two separate normal local 2048×25-layer/global 4096×5-layer streams,
default graphs/fusion and no full sliding-window override. No Keep28 or
new numerical allowance is supplied.

Native own first/repeat full heads/input/four initialized states/two layouts
are exact before FIRST d812 stock exposure. Stock complete heads/input then
pass their own finite exact repeat. Every published argmax and forced-ID
record is admitted. There is no cross-engine KV-byte equality claim.
Both publish 66 whole-vocabulary F32 heads; no features are retained.

The unchanged Python FP64 whole-vocabulary score includes 64 targets:
frontier row 0 predicts 992 and rows 0..31 target 992..1023; completed 1024 row 32
is unscored. Means are native 9.865990837966 and stock 9.865585200058.
Strict differences must be zero and relative conditional loss independently
≤3%; both pass. These conditional scores do not establish full-corpus
perplexity, natural-answer quality or scalar arithmetic qualification.

## Short bookends after quality PASS

| Engine | First (s) | Repeat (s) | Mean (s) | Spread (ms) |
| --- | ---: | ---: | ---: | ---: |
| Stock | 0.761823 | 0.762195 | 0.762009 | 0.372 |
| Native | 0.778753 | 0.778086 | 0.778420 | 0.667 |

Order is stock→native→native→stock. Every complete timing head/input/native
state/layout file matches its independently frozen engine proof before
interpretation. Paid walls include all 32 physical C2 waves, full head
publication, completed copies and argmax. Loading, prefill, warm/reset,
frontier/final state export and post-run finite/SHA scans are outside elapsed.
The 16.4105 ms mean gap exceeds these individual spreads, but two observations
per engine do not establish sustained parity or causation. Native does not
meet elapsed parity in this short screen.
This is not an end-to-end prefill/serving or memory-peak measurement.

## Provenance, funding and reproduction

Native 0d6d0952 and public a3122de5 are unchanged from the [new release reference](../llama-reference-refresh/README.md)/[native refresh](../ggml-release-refresh/README.md)
and 31 screen; no rebuild was performed. Native measured receipt 7d8f1847
and canonical metadata receipt 5ab70ad5 differ only in source_lock, without a
relink. The native build version is 0.1.0-dev+unknown; source/operands are
independently authenticated. SDK installation 7b8167cd is a different receipt.
NVCC 13.4.92, native cuBLAS 13.8.0.4, image CUDA runtime 13.4.49 and image
cuBLAS 13.7.0.27 name different components.

All four actual stock model processes prove CUDA initialization, a current
context, ordinal 0/one GB10, sm_121/48 SM and driver API 13040. Their loaded
`/usr/local/cuda-13.4/compat/libcuda.so.615.71.09` path is authenticated against
the frozen image closure. ldd and argument-refusal controls do not establish
this fact. All four owned model containers retire with checked absence.

The existing helper charges its 73,515,008-byte publication allowance before
allocation. Each of the four native initialized-state witnesses is
230,686,720 bytes, exported through the existing funded catalog/pinned-copy
contract outside paid walls. Analysis retains only two full head rows and a
streaming SHA block; full states stay external. No unchanged peak claim follows.

[run.sh](run.sh) fixes the closed 26 arguments/owned paths; [analyze.py](analyze.py)
reuses the 31 numerical and own/timing checks with actual 26 factual admissions.
Its postproducer admission validates source/metadata/forced roster/retirement
before a next arm. Supply the externally retained approved raw GGUF/artifact,
full carrier and source closure from results, use fresh owned output roots,
and execute the preregistered templates with the installed supervisor.
Only actual successful official/output/own hashes substitute into the reviewed
conditional templates. No raw heads, states, IDs or logs are in Git.

Fresh TensorFold task entry at 2026-10-06T18:22:59.558793Z fixes
cb2ebf0540f42604e2759b2ddef497861e928248/version 0.6.6 from primary HEAD
and five exact-commit files, including its actual CUDA recipe. Gemma26 is
listed on MLX; the CUDA recipe states no GGUF support. There is no matching
CUDA GB10/GGUF comparator and no TensorFold job was run.
Scalar/C1/C3/larger cohorts, longer context, features, serving/speculation,
swaps, broader quality and sustained performance remain separate gates.
