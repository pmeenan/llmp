<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared exact host mask intervals (2026-10-08)

T63 is transferred across the applicable LLM families. Checked prefix/ring
intervals replace per-cell host construction in Gemma2, Gemma3, both Gemma4
profiles and DeepSeek raw masks. DeepSeek compressed masks use checked prefix
fills. Gemma's existing content validators reduce all F16 bits over constant
intervals. Native/GGUF Qwen dense F16/F32 construction already used prefix
fills; QSA bias now uses three constant runs and its original finite dead
sentinel. No content-validation pass was added to DeepSeek or Qwen.

This removes host work without changing CUDA mask producers, model arithmetic,
public settings or admission. Ordinary device-mask paths bypass these host
matrices. The measurements below are CPU helper results, not model speedups.
Qwen-Image has no causal mask and is inapplicable.

## Exact contract and focused check

After a whole chunk writes through `end-1`, a ring cell holds its latest
congruent logical position below `end`. For a query at `position`, the visible
logical interval is
`[max(0, end-capacity, position+1-window), position+1)`. Its projection has at
most two physical intervals; the read width clips them, leaving padding
blocked. Checked subtraction and `position < end` avoid unsigned overflow.
Malformed descriptors refuse before any output mutation. The existing Gemma
validators still reject every differing F16 bit, including negative zero and
NaN payloads.

QSA retains its different formula: with `full=end/ratio`, cells before
`min(full,(position+1)/ratio)` are `+0.0f`, remaining full blocks are `1e9f`,
and unwritten blocks are `-inf`. If an incomplete/dead block exists, its bias
at `full` is overwritten with `1e9f` exactly. Dense masks and block tables are
unchanged.

One combined check passes nine unique cases:

- Five new CPU controls compare complete old-formula buffers from all four
  actual Gemma profile builders, DeepSeek full/ring and all compressed masks,
  and Qwen dense F16/F32/QSA. Shared interval controls cover wrap, overwritten
  early queries, clipped/empty reads, partial rows, windows, uint64 limits,
  malformed descriptors and single-bit poison at every physical cell.
- Existing G2/G3 ragged-source controls and G4 reference-source/ring-reuse
  controls authenticate actual source validation and reject edited masks.

The first build failed on signed alignment literals in the new generic G4
oracle adapter. It ran no checks or measurements and is unpooled. Changing
only `256,0` to `256U,0U` allowed the incremental build and original checks to
pass. All production and benchmark bytes were unchanged by that correction.

The additive model geometry helpers from the concurrent prefill batch are
included (patch `2ce18c03…`). This check covers the construction consumers;
it does not claim that batch's future-state or QSA boundary controls.

## CPU helper comparison

Spark A, ARM64 RelWithDebInfo (-O2), pinned SDK `aarch64-c09daba6ac31edee`. One process
runs old/new/new/old for each formula, 16 calls per arm, 512 rows by 4608
cells. The table shows milliseconds per call; ranges are the two arm means.
The old loops are retained in the reusable probe. Complete outputs and F32
bits agree before and after every arm.

| Formula | Old mean [range], ms | New mean [range], ms | Change | Old bookend movement |
| --- | --- | --- | --- | --- |
| Prefix fill | 1.2521 [1.2488, 1.2555] | 0.1094 [0.1092, 0.1096] | −91.26% | −0.53% |
| Compressed prefix fill | 0.1121 [0.1120, 0.1122] | 0.1090 [0.1086, 0.1094] | −2.81% | −0.11% |
| Ring fill | 1.4052 [1.4006, 1.4098] | 0.0938 [0.0933, 0.0942] | −93.33% | −0.65% |
| Prefix check | 1.3589 [1.3570, 1.3608] | 0.0848 [0.0846, 0.0850] | −93.76% | −0.28% |
| Ring check | 2.1728 [2.1701, 2.1755] | 0.0905 [0.0893, 0.0917] | −95.83% | +0.25% |
| QSA bias fill | 0.4325 [0.4323, 0.4326] | 0.1044 [0.1026, 0.1062] | −75.87% | −0.08% |

These are short, helper-only bookends, `n=2` per implementation. Buffers are
reused. Half-fill blocked initialization is paid; QSA's common allocation and
initialization are excluded. Prefix fill/check use positions 4095 through
4606; ring fill/check use positions 8191 through 8702, capacity 4608 and
window 4096, after the whole chunk ends at 8703. QSA uses ratio 4 at the same
late positions. These material formula shapes are not an end-to-end model
recipe. Compressed-prefix construction was already compiler-friendly; its
small 0.0031 ms absolute change earns no broad performance claim. Gemma3's
historical host-reference model result remains separate.

## Reproduce without raw results

The committed [samples/provenance](results.json) retain all 24 batch timings,
actual source hashes, SDK receipt and five executable identities, nine case
names and XML identities. [compare.py](compare.py) recomputes means, ranges,
deltas and bookend movement directly from that file. No disposable bundle,
checkpoint, tokenizer input or model acquisition is needed.

In a normally prepared `spark-native` tree, put the following commands in a
scratch script with `set -eu`, `cd` to that tree and set `JITLLM_TEST_DATA` to
its `tests/unit/data`. Run the script through the installed `spark-job start
--name UNIQUE --gpu --timeout 600 --grace 30 --stop-on-fail -- /usr/bin/bash
SCRATCH_SCRIPT`, then positively wait for it. The GPU lock excludes concurrent
heavy measurements; the test/probe itself does not acquire a device.

```sh
SDK="$HOME/.local/share/jitllm/sdk/aarch64-c09daba6ac31edee"
export JITLLM_TEST_DATA="$PWD/tests/unit/data"
timeout 480 "$SDK/bin/cmake" --build build/spark-native --parallel 4 --target \
  host_mask_test jitllm_host_mask_probe gemma2_plan_test gemma3_plan_test gemma4_plan_test -- -k 0
timeout 100 build/spark-native/tests/unit/host_mask_test --gtest_filter='HostMaskTest.*'
timeout 100 build/spark-native/tests/unit/gemma2_plan_test \
  --gtest_filter='Gemma2Plan.FreshRaggedSourcesAreFundedPaddedAndRevalidatedForReuse'
timeout 100 build/spark-native/tests/unit/gemma3_plan_test \
  --gtest_filter='Gemma3Plan.FreshRaggedSourcesAreFundedPaddedAndRevalidatedForReuse'
timeout 100 build/spark-native/tests/unit/gemma4_plan_test \
  --gtest_filter='Gemma4Plan.ReferenceSourcesAreFundedPaddedAndIsolatedBeforeStaging:Gemma4Plan.FreshPositionsReuseTheSameGraphAcrossARingWrap'
timeout 100 build/spark-native/benchmarks/jitllm_host_mask_probe
python3 docs/experiments/host-mask-intervals/compare.py
```

Check exactly nine completed positive cases, 24 ordered formula/arm rows and
`exact=1`, successful owned process retirement, stable source/ELF/receipt
identities and no kernel errors. The actual run finished with empty GPU
processes and unchanged pre/post first-resolved pinned cuBLAS libraries in the
plan test closure; CPU-only test/probe ELFs have no cuBLAS dependency. Full
regression, models and reference comparisons were deliberately not repeated.

Fresh task-entry TensorFold pins are main `f8fe17d24629aedabf90bbf78279dd776e6d62e7`
and python-0.6 `ed78d6fc204d89d90b045bf033d6551e7714f3a1`; full README hashes
are recorded in results. Its current qualified CUDA recipes do not supply a
same-format oracle for this CPU formula transfer; no upstream speed claim.

Final repository light checks pass: REUSE and embedded headers (1953 files),
portability boundaries (423 files), changed C++ formatting (13 files), diff
checks and replay of all six formula aggregates. All 25 executed source and
fixture bindings remain exact. Nine completed positive controls, six owned
children and installed supervisor 2241421 retire successfully; checked receipt
`a85e713d…` records the executed evidence. No full-suite or model result is
implied.
