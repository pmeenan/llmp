<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dense31 scalar layer0 real-weight FFN replay

The original fused Q4_K GeGLU, original separate products/GeGLU, and native
primitive FFN produce identical complete activation and down-output bytes on
this real dense31 scalar layer0 input. Actual launch observation confirms that
the original fused arm used its quantized fusion. This input does not support
the proposed fused-GeGLU arithmetic-gap explanation. Other layers, inputs,
product shapes and whole-model differences remain untested by this screen.

The input came from native dense31 query P67 after 64 supplied natural IDs and
three forced anchors, context256/max_rows128, ordinary products and checked
normRoPE/normADD. Row-invariant and other optional flags were off. Capture and
disabled controls both used graphs=false. Two captures and two disabled runs
had identical full 262,144-value heads; this establishes eager acquisition
fidelity, not graph-on target fidelity or original-model head agreement.
Final selected scalar plans recorded normRoPE 120/normADD 120, rows 1/segments 1;
row products, generic norm fusion, RoPE-store, shared VecQ and MoE policies 0.

## Complete outputs and actual selection

The actual layer0 gate/up are Q4_K [5376,21504]; down is Q6_K [21504,5376].
Each gate/up has65,028,096 logical bytes and65,028,240 readable bytes,
including its authenticated144-byte tail. Down has94,832,640 bytes. No host
weight conversion or synthetic replacement was used.

All 21,504 activation values and 5,376 down values are byte-exact between A,
A0 and B. Maximum raw delta and reference-energy NMSE are 0 for each output.
The combined 107,520-byte SHA is
`d1063198d3858f31ebb4271f14d9c3e73528a310f664ce2f3be65f4a5a6f7517`.
Native B also reconstructs both actual captured arrays exactly. Independent
own repetitions agree, changed-input eager/replay outputs agree and differ
from the original, restored outputs agree, and GPU input/quant-weight witnesses
pass before, during changed-input controls, after restore and after paid work.
Coordinate argmax agreement is an activation diagnostic, not vocabulary quality.

Nsight2025.3.2 observes A's
`mul_mat_vec_q<12,1,true,false,false>` at grid[21504,1,1], block[32,4,1].
A0 instead has two `mul_mat_vec_q<12,1,false,false,false>` events and a
separate `unary_gated_op_kernel<op_gelu,float>`. Both use Q6_K down
`mul_mat_vec_q<14,1,false,false,true>` at grid[5376,1,1], block[32,8,1].
The fused arm prepares its input once; the separate arm prepares it for both
products. These are observed replay launches on reconstructed public-backend
weight roots, not a capture of the original target graph. Default graph-level
tracing reports eager kernel events plus 76(A)/152(A0) CUDA graph-launch calls;
those event counts are not totals across replayed kernels. Complete traced
first/repeat/changed-input outputs equal their untraced counterparts. Application
completion receipts occur after backend/resource release, and both trace steps
also retired through the installed supervisor. Traced durations are excluded.

## Short component timing

| Untraced arm | 32 full chains (s) | API groups |
| --- | ---: | ---: |
| A original fused, first | 0.039472544 | 64 |
| B native primitive captured replay | 0.039180289 | 32 |
| A0 original separate | 0.039104753 | 128 |
| A original fused, bookend | 0.039060113 | 64 |

Each chain includes input Q8 preparation, all gate/up/GeGLU/down work and
complete 107,520-byte activation/down publication, finite validation and coordinate
scan. Setup, uploads, warm/capture, witnesses and disk writes are excluded.
Native executes 128 primitive operations across 32 chains in one captured
submission per chain; original submits two graphs per fused chain or four
separate graphs. Submission groups are not CUDA launch counts. Native is
0.219118% below the fused bookend mean; separate original is 0.411486% below it.
Both differences are smaller than the 1.050343% span between fused bookends.
This tiny single component screen establishes no performance winner or adoption.

Native's declared host allowance is384MiB, mapped device roots256MiB,
catalog-backed pinned staging94,940,160 bytes and planned scratch peak24,192
bytes. Original's declared host allowance is384MiB; public weight allocation
224,912,896 bytes and public compute buffers387,072(A)/645,120(A0).
Those occupancies exclude foreign library/pool allocations and are not physical
memory peaks. Complete constructed tensor roots, readable tails and extra-row
zeros have byte witnesses; public extra allocation padding is cleared/reported,
not claimed as a full-allocation witness. Native holds leases/requests through
completed jobs; unknown completion retains the whole lifetime. Original failures
fail-stop before any success marker.

## Provenance and reproduction

Measured ancestry is c67dd4dbb598c42a3207b73a260ea57a81dfbdd5 with only the
ten benchmark/protocol source paths in the pre-acquisition receipt changed.
No production source changed. The [protocol](PROTOCOL.md) is the exact
pre-acquisition document, including its initial planning label. Compiled sources,
packing header and build/replay/trace wrapper retain their frozen bytes;
these analysis/report files were added afterward. [Provenance](provenance.json)
records exact sources, binaries, inputs, headers, libraries, model metadata,
own freezes and official job identities. [Results](results.json) contain only
aggregate output/timing/launch evidence; raw payloads/logs/traces remain external.

Capture binary `e7a482ec…`, original thin client `7b2cdcf1…` and source freeze
`f36ba02f…` preceded acquisition. Native1 cleanly refused planning because its
standalone binder lacked the existing readable-padding marker. The correction
adds that marker only after imported tail/region validation, with no captured
input or original math change. The refused binary/source and original source
freeze remain preserved. Corrected native binary `405c14d9…` and separate
replay source freeze `1daede94…` preceded native2; native own freeze `c8fc23d4…`
then preceded all original outputs. Capture freeze `de28c663…` still points to
the original source freeze. A source1 missing-rg failure and analysis1 missing
log alias are preserved; successful successors do not rewrite those records.

Use the pinned original b29c606e28a01b1bc8c1351026a0fa6e616bf6c4 headers and
digest image in reference.sh. Build both manual native targets with the locked
Spark SDK, then run `bash reference.sh build` before creating a source freeze.
Supply the already verified prepared dense31 artifact and canonical 4096-byte
IDs SHA `b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610`.
No download or historical full-model rescan is part of reproduction.

The source freezer expects external `inputs.i32`, `headers/`, the original
client, `libraries.sha256` (three `/app/libggml*.so` SHA lines), the actual
native build receipt, and `source-ancestry.json`. The ancestry object records
base `c67dd4d`, actual Git HEAD, exact dirty paths, empty `tracked_production_diff`,
and `{path:{bytes,sha256}}` for the ten `SOURCE_FILES` named by own_freeze.py.
Record Git state honestly: use a clean committed packet or an acquisition-only
checkout whose dirty paths are those sources; exclude unrelated in-flight work.
All subsequent phases must reauthenticate that environment and source table.
If a replay-only source changes, preserve the original source/capture receipt
and capture build/ancestry as `capture-build-receipt.json` and
`capture-source-ancestry.json`, then create a separate replay source record.
These receipts are exclusive-create; retries use fresh directories/records.

Under installed `spark-job start --gpu --timeout 600`, build and officially wait;
run source freeze with the retired build job name. Run capture helper as
`ARTIFACT NEW_OUTPUT INPUT_IDS capture|control` in order control-first,
capture-first, capture-repeat, control-repeat, then officially wait and run
`own_freeze.py capture ROOT SCRATCH ACQUISITION_JOB`. Run native replay as
`capture-first NEW_OUTPUT` into two independent directories, officially wait,
and run `own_freeze.py native ROOT SCRATCH NATIVE_JOB --native-dirs FIRST REPEAT`
(with `--source-record` for a separately frozen replay correction).
Do not run originals until actual activation/down reconstruction and native
own freeze pass. Link `SCRATCH/operands` to capture-first, then run the untraced
sequence `reference.sh replay original-first A`, native replay to native-bookend,
`reference.sh replay original-unfused A0`, `reference.sh replay original-repeat A`.
Run `reference.sh trace A` and `trace A0` separately and export each report with
`nsys export --type sqlite`. Require completion markers and successful official
retirement, then analyze retained outputs with `analyze.py SCRATCH NEW_RESULT EXPECTED_NATIVE_FREEZE_SHA`
and `observe.py SCRATCH NEW_OBSERVATION`. Supply the independently recorded
pre-oracle native-freeze SHA; the analyzer authenticates its linked capture/source
receipts before trusting the paid outputs.
The analyzer expects `SCRATCH/bookend1.log` to name the official untraced log.

Task-entry [TensorFold](https://github.com/ashhart/TensorFold) was freshly pinned
at 609ca419abecebdc5a059498a613680bd3aa847f/version0.6.5. Its Gemma26 recipe
is MLX-only; it provides no dense31 CUDA comparator. No production default,
whole-model quality, batching or depth qualification follows from this result.
Gemma26 transfer remains an explicit future check of its actual shared/expert
weights, formats, columns and fusion eligibility. Qwen/DeepSeek likewise require
their own formats/activation/shape checks; this one dense31 GEGLU/Q4_K/Q6_K
result does not qualify a cross-family policy.

Locked native/original builds, four eager target head controls, two native
reconstructions, the untraced bookend, two launch traces and final authenticated
retained-array analysis all retired successfully on Spark-b. Local touched
format, REUSE/header checks (1,454 files), boundaries (392 files), diff, script
syntax and analyzer controls passed. The owner’s benchmark-only comparison
override applies; no routine full unit suite or model qualification ladder ran.
