<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 26 native runner: bounded first controls

These controls exercise a complete 30-layer 26B-A4B engine runner over the
native prepared artifact, independent request slots, initialized state and
stable-address graph replay. They do not qualify serving, assistant execution,
representative quality, long context, optimized batching or reference speed.
The 31B runner is not implemented. The numerical baseline acquires the full
weight closure; route-discovered expert acquisition remains a later execution
slice. No unsupported model or format is added to the support matrix.

## Inputs and provenance

The actual GGUF is `unsloth/gemma-4-26B-A4B-it-GGUF` revision
`c099eb48e663fd284577b04978a94ffccb261841`, file
`gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`, 16,947,541,728 bytes, SHA-256
`f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f`.
The deep-verified native import uses `m3-1+layout-a0d1980a9eddd1ad` and
artifact identity `4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3`.
Its immutable directory lives on both Sparks under
`~/.local/share/jitllm/m3-artifacts/`. The supervised checksum copy published
an absent destination atomically; it did not requalify the source hash.

Native controls run on Spark-b with the locked Spark-native SDK/sources.
The first host-mask arithmetic screen used the runner working diff over
`b657317`; final device-mask controls use the working diff over reviewed
`891ceea`. The norm alternate-screen executable SHA-256 is
`87d30d18631930e8ecb3cb943e78ddae077fb708c955c652d842df55720b012e`.
The warm A/B/A executable, before later formatting and test additions, is
`cc42d4d672eebbe3eae4382a0cf14f5ed852586b248413d240548f542c72edae`.
These identities describe measured binaries, not a later edited source tree.
Final focused-control executable hashes are runner
`c44071705dc544ab8191339989e49d7f9a539d6be30cb63fd4a3530dff5adfd8`
and GPU test
`8dc854ac8feccf7b78a01c96ae7b72d166982c5508185c74b1d5bf4ed45cadd5`.
The final five-file source bundle is
`68a0a09949af08ec705f47e5ac8fc5098528050dc6b83a55c95d95744fff6927`:
sorted paths of `src/engine/gemma4_runner.{h,cc}`,
`benchmarks/gemma_runner.cc`, `tests/unit/gemma4_runner_test.cc` and
`tests/unit/gemma4_runner_gpu_test.cc`, each path/NUL/content/NUL concatenated before SHA-256.

The same-format numerical oracle is llama.cpp
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` / b10964, image
`ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`.
Its C API harness used context 4,096, batch/ubatch 128, F16 KV, FlashAttention,
no drafts and `GGML_CUDA_DISABLE_FUSION=1`. Fusion is disabled for this
numerical oracle only; a competitive performance comparator must enable the
reference's fastest supported production settings, including fusion/graphs.
The task-entry TensorFold observation was HEAD
`609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5, Gemma 26 MLX only;
it supplies no GB10 comparator. Refresh it again at each later applicable task.

## Quality smoke and frozen calibration

The literal UTF-8 input is `The capital of France is`, with no template,
newline or trailing space. Token IDs are `[2,818,5279,529,7001,563]`, including
explicit BOS 2. Both engines execute exactly 32 raw argmax steps, deliberately
continuing through EOG. This is an arithmetic fixture, not natural chat.

Before comparing references, independently selected native ordinary and
norm-fused paths produced byte-identical full 32 logits vectors. The frozen
p99 top-two margin movement is 0, max logit delta 0. Its JSON SHA-256 is
`d72869187aed263c1da3c3de3ea5fd4abb0c0ff7e118177db57efa8d1f19c2ca`.
Both cleared reference repeats are also byte-identical. This small calibration
supplies no tolerance for later batch/depth policies.

All 32 native greedy IDs exactly match the gold reference prefix. Native/reference
max raw logit difference is 0.040576934814453125; max chosen-target NLL difference
is 0.0072642275310559334. No mismatch receives a near-tie allowance. Future
mismatches require teacher-forcing the reference prefix rather than continuing
the divergent native prefix. Full 32 host/device-mask logits are byte-identical.

A separate 27-ID toy text fixture supplies 26 teacher-forced targets, BOS
unscored and the final unused head row dropped. Native/reference full target
logits give identical FP64 `math.fsum` mean NLL 4.845988085373614 and
PPL 127.22893295432857. The oracle's original naive C++ sum differs by about
2.9e-13 relative PPL. This is not a representative PPL corpus or a quality bound.

The exact teacher text is:

> Paris is the capital of France. The Seine flows through the city. A triangle has three sides. Two plus two equals four.

Its full input IDs are
`[2,50429,563,506,5279,529,7001,236761,669,69598,17752,1343,506,3207,236761,562,17852,815,1806,9174,236761,8512,2915,1156,14339,2390,236761]`.
Row `j` predicts ID `j+1` from the prefix ending at ID `j`; score rows 0–25
and drop native row 26. Decode all 27 input rows in one batch with logits
enabled per input token. NLL is `logsumexp(full_F32_row) - target_logit`,
computed in FP64, never estimated from a truncated top-probability list.

## State, joining and selection

Actual controls check slots 1/2/4 with independent, unequal prefix lengths;
three fresh one-row decode units; complete initialized padded KV; and exact
same-policy eager/capture/replay. They check maximum ragged prefill 125+1+1+1,
checkpoint into an empty slot, spill/restore, peer-preserving clear and plan
reclaim followed by exact continuation. Refusals publish nothing. Separate
state-growth and paged-out-weight execution admission controls retain the
completed prefix and usable state; releasing capacity resumes exact logits.
Test pressure is a catalog host capacity reservation, not a physical payload.
Shared LiveState/cohort controls cover uncertain-copy retirement and shared
fault quarantine. Full ring-wrap, cancellation/time-slicing and long-depth
runner controls remain owed.

The retained ordinary solo/join byte-equality failure is significant: the
maximum observed full logit difference is 4.32151, with roughly 0.4–0.48 million
changed initialized KV bytes per joined slot. Strict greedy IDs still agree
in these small rows, and each joined policy replays exactly. This does not
qualify ordinary joining or widen the frozen quality calibration.

An explicit experimental combination of shared Q8 preparation, one-row sums
and norm fusion gives byte-exact solo versus joined logits and initialized KV
for the tested 1/2/4-request decode shapes. Its bound plan selects 271 norm
fusions, 266 shared VecQ products and 30 remaining row-preserving products,
with 0 RoPE/store fusions and 0 lane-tagged steps, at each of these shapes.
`kRowInvariantColumns` is 8: generic one-row products are enabled only through
8 joined rows. Shared VecQ eligibility extends separately through 16 rows;
9–16 rows can mix VecQ with ordinary float products. No wider-row scalar
agreement is claimed. All three optional policies remain off by default.
Qualified lane scheduling, route-discovered paging, RoPE/store selection,
paid whole-chain Q8/GeGLU writer choices and cohorts 8/12 remain open.

## Warm latency and memory boundaries

A short paid solo A/B/A runs 32 completed one-row chunks after 8 warm chunks,
then resets the identical literal prefix. Timing includes argmax, CPU input
preparation/staging, graph execution, completion and output publication;
it excludes setup, disk reads, raw logit writes and printed logging.
Ordinary/norm/ordinary wall times are 0.641216/0.638594/0.640489 seconds,
about 50 chunks/s. The ordinary bookends move 0.11%; this short sample does
not justify selecting norm fusion or establish reference parity. Each run
records 2 captures and 38 replays across warm/reset/measured work.

For context 4,096/max_rows 128/one slot, device inputs require a measured shared
activation grant 337,641,472 bytes, scratch 10,485,760 bytes, host input envelope
2,097,152 bytes and plan floor 2,187,792 bytes. These are setup envelopes, not
measured peak memory. Larger/cohort setup probes balanced and maximal ragged
rows at initial and full read depth, including every output-head row.
State ceilings are virtual reservations; only padded initialized prefixes
materialize. Failed capacity growth does not eagerly map all slot ceilings.

Actual native weight backing is 17,047,748,608 bytes, 8,129 extents. Runtime
slab padding is 7,929,856 bytes; the Gemma-only joint 256/GGML-block pitch adds
7,667,712 raw pitch bytes over the minimal block-aligned pitch. That raw delta
is not an equal physical-page delta. Actual layer 14, expert 67 crosses shards
with a 64-byte minimal gap that cannot accommodate the required 256-aligned
slab delta; the joint pitch preserves every member/tail and fixes this cut.
No artifact rewrite or shared Qwen/DeepSeek pager change is made. Physical
padding must enter the later measured reference peak-memory comparison;
older independent-group backing estimates are not comparable measurements.

## Reproduce and inspect

[oracle.cc](oracle.cc) retains the C API control in Git. Its measured scratch
source SHA-256 was
`3f0aa14bf8536db9a4e5612cb3af01b144da77a29bda84a703c8ff2d96069a3c`
and measured executable SHA-256
`e7d3992f1cbcc0310322049574b949cb80c63f4b6d4db5401f600aaf73980fbe`.
The retained reproducer only corrects the inherited introductory comments
and formatting; its SHA-256 is
`5e4846b2d2ef71456e3415264f8c800bfe297bcbe3ef71f145dace63cfd30218`.
It is external reference tooling, not linked into native inference.

Place this source and the locked revision's `include/llama.h` plus
`ggml/include/{ggml.h,ggml-alloc.h,ggml-backend.h,ggml-cpu.h,ggml-opt.h,gguf.h}`
in a scratch directory. Mount it at `/scratch` and the approved GGUF's
directory read-only at `/model` in the digest-pinned image above. The exact
in-container compile and run commands are:

```sh
g++ -std=c++23 -O2 -march=armv8-a -Wall -Wextra -Werror -I/scratch \
  /scratch/oracle.cc -L/app -Wl,-rpath,/app -lllama -lggml -lggml-base \
  -o /scratch/oracle
CUDA_DISABLE_PTX_JIT=1 GGML_CUDA_DISABLE_FUSION=1 /scratch/oracle \
  /model/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf /scratch/output
```

Use installed `spark-job start --gpu --timeout 600` around the container
compile/run, NVIDIA CDI `--device nvidia.com/gpu=all`, `--network none`,
`--read-only`, a writable bounded `/tmp`, and the calling UID/GID. The source
sets all layers on GPU, plain reads (`load_mode=NONE`, lazy off), one sequence,
F16 K/V, context 4,096, batch/ubatch 128 and FlashAttention. Default CUDA graphs
remain enabled; clear memory before each of the two repeats and the teacher
batch. Explicit positions start at zero, and argmax ties choose the lower ID.
The fixed literal and teacher inputs are embedded in this reproducer.

For native generation, use `2,818,5279,529,7001,563`, 32 steps, one slot,
`ordinary`, and `device` or explicit `host`. For native teacher scoring, use
the full comma-separated 27-ID list above, zero steps, one slot, `ordinary`
and `teacher`. Both commands use the same prepared artifact named above.

`benchmarks/jitllm_gemma_runner` takes
`ARTIFACT OUT TOKEN_CSV STEPS [SLOTS] [ordinary|norm] [teacher|warm] [host|device]`.
Its default mask path is device-produced; `host` explicitly funds reference
masks. Run every Spark build/test/inference via installed `spark-job --gpu`
and the locked build; source synchronization is `rsync -rlpc`, excluding
`.git` and `/build`. The focused actual-model tests are
`gemma4_runner_gpu_test`; host lifecycle/admission tests are
`gemma4_runner_test`, alongside the shared cohort/plan/state controls.

Raw fixtures stay outside Git. Local protocol/reference files:
`/tmp/jitllm-m35-coordination/gemma-oracle-prep/{PROTOCOL.md,reference.json,output/}`.
Native immutable vectors/calibration/comparison:
`/tmp/jitllm-m35-coordination/gemma-runner-raw/{ordinary,norm,teacher,device}/`,
`native-noise-frozen.json`, `first-comparison.json`, `host-device-exact.json`.
Reproduction analysis scripts are adjacent
`gemma-runner-noise.py` and `gemma-runner-compare.py`; root independently
verified the same fixtures with `root-gemma-first-verify.py`.
Remote native outputs are Spark-b `/tmp/m35-gemma-native-{first3,noise1,teacher1,device1}`;
reference output is Spark `~/.local/share/jitllm/gemma-oracle-prep/output`.
Official logs live at each host's `~/.local/share/jitllm/jobs/NAME/log`;
first/native jobs are `m35-gemma-runner-first3`, `noise1`, `teacher1`, `device1`,
`warm1`, `state2` (retained failed byte-equality diagnostic), `state3`,
`invariant1`, `admission1`, `check1` and `shared-state1`. The oracle job is
`m35-gemma26-oracle-prep1`. These are bounded controls, not the final model gate.

Focused `check1` runner checks pass 4 host and 5 actual-model GPU tests, with
4 shared cohort and 7 plan-cache controls. The 21 shared LiveState controls
pass in `shared-state1`. Local format, REUSE/header 1,272-file and 373-file
portability checks pass.
