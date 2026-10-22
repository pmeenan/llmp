<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fresh RMSNormMul matrix: frozen W5 compilation

**Prospective registration, 2026-10-22: actual A W5 build frozen.**
Independent review and a root-owned commit of this registration and helper are
required before execution. No fresh timing has run. The previous r6, W3 and W4 proposals
are superseded without measurement; their evidence retains its original identity.
W3 was held after the later compiler depfile alias-ambiguity defect and generated
input ownership gap were demonstrated. W4 was then held when its full source
audit found the authenticated CUDA host wrapper also needed a generated-input
declaration. W5 includes all reviewed repairs; none of these proposals ran timing.
The original v11/v12 twelve observations keep their original identities and
remain historical. No fresh rank data will be installed before independent raw
row audit. Full U6, other fusion families and model qualification remain open.

## Exact compilation and execution boundary

Measure the new W5 benchmark compiled on **Spark A**, at the existing source
path `/home/pmeenan/src/llmp` and build path `build/spark-native`. After its
qualified freeze, do not sync, build, reconfigure or edit them during the
experiment. Stage only helper/manifest/review artifacts under `/tmp/u6-fresh-w5`.
The full source manifest and whole patch identify the actual integration source,
including dependencies composed before the source commits. The remote
`61f...dirty` checkout
label does not identify this compilation. The registration's main base and
later documentation commits likewise do not claim to be the compiled source. Main source
commit `c5e7584e` now has tree `5ca10c0b3af9bcd70572d10d607fc5afb260599c`,
which matches the frozen W5 source tree. The original base-plus-patch and full
manifest below remain the actual compilation provenance; remote Git metadata
was not changed to obtain this equivalence.

| Identity | SHA-256 or value |
| --- | --- |
| Compiled source base | `e9965820ceba09cedfd9395eb60f0ae6e6cc3599` plus the whole W5 patch below |
| Whole W5 source patch | `a4b5d01d67254631b708c055267eab048969f5106c685561bb51ed2bcd2c2eed` |
| Full 2,814-file source manifest | `70812bd25843e0c06dcf75af729a6fcc36259e96c7c16fbdd640b70eee84ca34` |
| Benchmark fixture source `benchmarks/fusion_ranks.cc` | `d08cf6969c2d6206326327060187a6785f4dab06464ba1bba4d2a173dd761e2f` |
| Native benchmark binary | `1394ae098f4159787bec6dc4c417407f36fc6ac4679e022792a83eca6e9f7637` |
| Compiler-owned implementation identity | `4c2b14cf0a66eebb8f433daefff9b58b4cfc0afdc082d6ba0e5a1ec3ac200cee` |
| Compiler configuration | `aea97f6ac4defe73b6a2415a36b7ed260788006f295cce1fba0b35f76c263060` |
| Current compiler identity receipt | `d9750b7372dcacdf0ca699459bd2bcbe3374c03db6b271a1d41154e3566a50f1` |
| Actual object inventory | `6c11b744b3a20468ff9a0fc2f0cbe5667f7d7270e8531a0d33c3485ad2189047` |
| Selected 195-object dependency/command inventory | `10311c623abc8997f5ec20c9127194702c7345fddf6dd1866152c544fe47fb51` |
| Launch helper `run-fresh-w5.py` | `4bcf42be6b9c14eb41c219b91b5e7a018c6323b902ef57a28b16b87c92b23ee1` |
| Host / GPU | `spark` / `spark-c4e2`, NVIDIA GB10, architecture 121, 48 multiprocessors |
| CUDA driver API / runtime API | `13000` / `13040`; full driver version and GPU UUID also recorded at launch |
| SDK | `aarch64-2c4eb5b2f68f3fa5` |

The whole source patch and manifest are retained on the workstation at
`/tmp/llmp-m35-orchestration/wave-w5-source.patch` and
`/tmp/llmp-m35-orchestration/wave-w5-source.json`. W5 includes the reviewed U6
mechanism/traits/index/compiler identity, tool-route integration, actual-checkpoint
oracle and direct test-link repairs, and the reviewed incremental wrapper fix.
It also includes the declared generated-input ownership repair, the independently
reviewed compiler depfile alias-ambiguity repair, the explicit CUDA host-wrapper
producer declaration, and the integrated tool-result report, plan and D-097
documentation clarifications. These source changes deliberately change implementation
identity. W5 must not inherit r6, W3 or W4 identities or reuse historical measurements.

A's guarded six-step W5 job passed: native benchmark build, absolute-path
production identity check, actual Clang/NVCC and Ninja fixture controls, and
1024×1 setup/capture/output validation. Validation recorded zero output errors,
128/256 graph nodes and timing disabled; all 2,814 source files matched before
and after. Compiled metadata, selected 195-object inventory and supervisor
receipts are retained under `/tmp/llmp-m35-orchestration/wave-w5-benchmark/`.
B's separate guarded job also passed all 220 native targets, five targeted
CTest entries including `sources.closure`, and 44 source-inventory controls;
its 2,814-file pre/post source guards matched. Root retains its receipts under
`/tmp/llmp-m35-orchestration/wave-w5-b-native-supervisor/` and
`wave-w5-build-checks-b.xml`. Broader affected-model qualification remains
root-owned and open; these results do not claim full U6 or milestone completion.

## Unchanged prospective method and bounds

First run **1024 × 1** only. Its independent review checks source, compilation,
actual own-decomposition selection, arithmetic, paid work, every raw sample,
completion and retirement. A valid neutral result or fusion loss is evidence,
not grounds to rerun for a favorable result. Only after accepting that screen
run the remaining eleven cells in this order:

`256×1, 256×4, 256×32, 256×256, 1024×4, 1024×32, 1024×256, 4096×1, 4096×4, 4096×32, 4096×256`.

Each is a separate exclusive supervised job and fresh output directory. These
are exact operand tuples, not intervals, checkpoint geometry gates or capability
restrictions. No extrapolation or additional matrix is authorized here.

The frozen harness constructs F32 RMSNormMul with epsilon 1e-6 and obtains its
own expansion. Both arms use production planning/lowering of that expansion;
only the primitive arm disables the `rms_norm_mul` rewrite. The fusion selects
`ggml.rms_norm_mul.fused`; the logical primitive chain is ordered `ggml.rms_norm`
then `ggml.mul`, physically launched by the existing two-launch unfused wrapper.
Logical child identities and physical recipe identities remain separate. Every
operand, output, parameter, numerical contract and selection environment is
reported; distinct source roots and unchanged storage/stream dependencies are
required by the existing proof. No arithmetic mode changes.

Before timing, all outputs must be finite, relative RMS error ≤1e-6 and maximum
absolute error ≤2e-5. Capture/replay must preserve complete output and the exact
128 fused versus 256 decomposition graph-node inventory. Four untimed paired
warmups precede the first primitive bookend; four more alternating paired
warmups precede nine alternating AB/BA timing pairs, followed by the second
primitive bookend. Every event sample replays 128 complete chains. Preserve all
unsorted pairs and their actual order; compute medians from copies and round
nanoseconds using the frozen shared classifier's harness path. The harness
flushes each arm and bookend, including partial work before a failure.

Reject evidence if `max(bookends) > 1.1 × min(bookends)`. Keep the D-085 band:
with rounded median fused F and decomposition D, F < 0.9D favors fusion,
F > 1.1D favors its own decomposition, and both inclusive boundaries are neutral.
Both directions use D as denominator. No bound changes after observing results.
Correctness and eligibility remain prerequisites; timings grant no arithmetic
change. Missing or stale data retain incumbent selection and remain unmeasured.

Paid work is unchanged: 32 MiB library workspace, 16 MiB launch pool, separate
32 MiB activation arenas per arm, and the 64 MiB construction-funding allowance.
Record actual selected persistent bytes and logical/lowered activation extents
from each run rather than replacing them with capacity. These capacities and
selected backing do not claim peak process/unified-memory usage. The v12
capacity fix is already present: per-input overflow checks and actual planned
and lowered requirements must fit those original fixed capacities. Do not
increase arenas or padding to force a cell through.

## Staging and bounded execution recipe

Root first verifies the committed helper bytes against the table above, then
stages that exact file as `/tmp/u6-fresh-w5/run-fresh-w5.py` and the exact manifest
as `/tmp/u6-fresh-w5/source-manifest.json`. This is not a source-tree sync.
The executor verifies the real committed registration locally with `git show`;
`--prereg-commit` is its recorded commit identifier, not a remote Git-HEAD check.
Do not mutate the frozen checkout's Git metadata to simulate source provenance.

For the first screen create an external steps file containing this one command,
replacing `COMMITTED_REGISTRATION` with that real 40-hex commit:

```text
/home/pmeenan/.local/bin/mise exec -- python3 -B /tmp/u6-fresh-w5/run-fresh-w5.py --prereg-commit COMMITTED_REGISTRATION --case 1024x1
```

After host ownership is assigned and installed `spark-job busy` is clear, start
with the installed supervisor, cwd `/home/pmeenan/src/llmp`, job name
`m35-u6-fresh-w5-1024x1-r1`, `--gpu --timeout 600 --stop-on-fail --steps STEPS`.
Wait using the same installed supervisor. No build/shared job may overlap the
exclusive timing job. The supervisor's 30-second termination grace remains
unchanged. All helper subprocesses share one 540-second budget; cancellation
or failure retires the owned child group with one shared 25-second cleanup budget: up to
20 seconds after TERM and five after KILL. The exited leader remains unreaped
through `waitid(WNOWAIT)`, reserving its PID/PGID until no live or stopped group
member remains. Zombies are recorded separately; only then is the leader reaped
and the group permanently retired. TERM/INT are blocked only across that final
reap/marker transition, then pending cancellation is delivered. A successful leader with live descendants
is a failed run, even after cleanup succeeds. Cancellation during process
creation waits for ownership; repeated signals cannot restart cleanup. Core dumps are disabled.

Outputs are private and never overwritten:
`~/.local/share/llmp/m35-u6-fusion-ranks/fresh-w5-r1/1024x1/`.
`launch.json` records helper/source/binary/compiler identities and each process's
return/retirement status. `samples.jsonl` and `stderr.txt` retain harness output;
`hardware.txt` records full driver/GPU identity; `identity_before.*` and
`identity_after.*` retain the current compiler-receipt checker. The helper checks
all source files, frozen config/object/identity receipts and binary pre/post;
it checks actual current objects and dependencies with the production checker
before and after timing. Source, build-receipt and binary postchecks run
independently after failure too. The compiler postcheck runs after an ordinary
refusal if the original child budget remains, or records explicit unavailability
on cancellation, exhaustion or engaged cleanup; it never silently disappears
or extends the budget. A running compiler postcheck remains cancellable; only
its bounded retirement ignores repeated signals. Postcheck failure/unavailability prevents completion.
It flushes/fsyncs retained files before final status.
Only successful complete timing, successful identity checks, zero-return
retirement and unchanged source/binary set `completed=true`. Independent review
still recomputes all pairs, medians, drift and decision; helper completion alone
is not acceptance. After each job require supervisor rc 0 and a clear installed
`busy` report with no surviving GPU process. Preserve failed/refused raw evidence;
no same-directory overwrite or automatic retry is permitted.

After the first screen is independently accepted, root creates an external
`/tmp/u6-fresh-w5/first-screen-review.json` with these exact fields:

```json
{
  "verdict": "PASS",
  "prereg_commit": "ACTUAL_REGISTRATION_COMMIT",
  "first_launch_sha256": "ACTUAL_FIRST_LAUNCH_HASH",
  "first_samples_sha256": "ACTUAL_FIRST_RAW_SAMPLES_HASH",
  "independent_review_sha256": "ACTUAL_INDEPENDENT_REVIEW_NOTE_HASH"
}
```

This is a future audit receipt, not fabricated approval. Hash the actual file.
For each remaining cell, replace only `--case` and add
`--first-screen-review-sha256 ACTUAL_RECEIPT_HASH` to the registered command;
use job `m35-u6-fresh-w5-WIDTHxROWS-r1` and its matching unique output directory.
The helper refuses these cells without a matching review receipt that binds the
completed first screen, raw samples and same registration. Root retains the
independent note and supervisor receipts alongside all raw data.

Only independently audited fresh rows may later enter production data, with
actual measured identities and complete reviewed provenance. Integrating data,
qualifying W or another build, and measuring other fusion classes are separate
follow-up work. Nothing in this registration marks full U6 complete.
