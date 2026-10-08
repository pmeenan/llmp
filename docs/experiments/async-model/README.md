<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M0 async/task ownership experiment

This CPU-only, deterministic C++23 experiment supports
[D-048's design](../../async-model.md). It models a bounded task phase that
joins fake read/GPU/network operations, then resumes with a terminal outcome.
It is not an application scheduler, a CUDA test, or question 9's reservation
progress proof. All implementation here is original Apache-2.0 code with no
new third-party source dependency; the compiler/library provenance is the
existing [D-032 toolchain smoke](../toolchain-smoke/README.md).

## Evidence and limits

Verified 2026-09-22: Clang 22.1.8 native x86-64 build/run on the workstation,
and its AArch64 cross build executed on `spark` (`spark-c4e2`). Both used
C++23, explicit baseline CPU targets, `-O2`, and
`-Wall -Wextra -Wpedantic -Werror`. Neither build uses CUDA headers or libraries.
Supplementary AddressSanitizer/UndefinedBehaviorSanitizer runs at `-O1 -g`
passed with the pinned Clang 22.1.8 SDK natively on the workstation and
cross-built for `spark`, using the `libclang-rt-22-dev` runtime archives
recorded in the [toolchain-smoke artifacts](../toolchain-smoke/artifacts.json)
(`sanitizer_followup`), and with the installed GCC `13.3.0-6ubuntu2~24.04.1`
natively (2026-09-22).

| Check | Result on both hosts |
| --- | --- |
| Read cancellation, acceptance and physical completion in every order; registration delays reuse | 6 event orders passed |
| Three consumers sharing one backing, two registration lifetimes, cancellation at each boundary or absent | 210 schedules passed (30 legal completion/cleanup orders × 7 cancellation positions/absence cases) |
| All 4 task and 8 operation slots occupied; repeated cancellation, completion and cleanup with no free slot | Drained; bounded ready flags coalesced duplicate wakeups; round-robin progress passed |
| Partial submission rejection, completion before acceptance, unrelated ready phase while another waits | Passed |
| Rejected registered read/network submission, cleanup before or after the scheduler observes rejection | 4 cases passed; registration still prevented early reuse |
| Read error or invalid content, known terminal consumer error | Failed safely without publishing invalid content; accepted siblings drained |
| Unknown submission or unknown completion plus client cancellation | Backing stayed charged/protected; new admission stopped; client outcome did not retire task |
| Task/operation/backing slot reuse and stale notifications | Old generation could not change new backing or release its hold |
| Successful verified read followed by a dependent consumer phase | Passed; lease retirement retained resident data until explicit reclaim |

The bounds (4 tasks, 8 operations, 4 backing slots) are small exhaustive-test
fixtures, **not runtime defaults or a capacity-reservation policy**. Read
validation is an injected result, not an implemented hash/short-read parser.
Physical access writes an integer to detect early reuse; there is no device
DMA. Registrations and cancellation acknowledgements are synthetic. A
provider contradiction stops the test; production fault reporting is absent.
Unknown-completion cases deliberately retain their simulated records until
the fixture ends; no real resource is released by that destruction.

The simulator has one thread and explicitly controlled event order. It
does not prove atomic publication, sleep/wakeup correctness, ARM concurrency,
provider queue capacity, actual cancellation, timeouts or GPU/transport error
recovery. It models one phase per task, not task trees, shared page-in waiter
coalescing, output backpressure, full model dependencies, mutable spill or
admission envelopes. Those remain M2/M6/M6a/M8 gates in the design; no measured
latency, throughput, model correctness or memory-budget claim follows here.

## Reproduction

Build products stay outside the repository. With the D-032 SDK extractions
and pinned Spark sysroot still present, run from the repository root:

```sh
mkdir -p /tmp/llmp-async-model
export LD_LIBRARY_PATH=/tmp/llmp-clang22/sdk-amd64/usr/lib/x86_64-linux-gnu
/tmp/llmp-clang22/sdk-amd64/usr/bin/clang++-22 \
  --target=x86_64-linux-gnu -march=x86-64 -std=c++23 \
  -Wall -Wextra -Wpedantic -Werror -O2 \
  docs/experiments/async-model/main.cc -o /tmp/llmp-async-model/native
/tmp/llmp-async-model/native

/tmp/llmp-clang22/sdk-amd64/usr/bin/clang++-22 \
  --target=aarch64-linux-gnu --sysroot=/tmp/llmp-toolchain-smoke/sysroot \
  --gcc-toolchain=/tmp/llmp-toolchain-smoke/sysroot/usr \
  --ld-path=/tmp/llmp-clang22/sdk-amd64/usr/bin/ld.lld-22 \
  -march=armv8-a -std=c++23 -Wall -Wextra -Wpedantic -Werror -O2 \
  docs/experiments/async-model/main.cc -o /tmp/llmp-async-model/aarch64
scp /tmp/llmp-async-model/aarch64 spark:/tmp/llmp-async-model-20260922
ssh spark /tmp/llmp-async-model-20260922
ssh spark rm /tmp/llmp-async-model-20260922
```

Expected final line: `PASS: deterministic CPU-only async ownership experiment`.
The executable fails on the first violated condition even in optimized builds.

Supplementary sanitizer commands (the AArch64 build takes the arm64
package's resource directory for its runtime archives):

```sh
/tmp/llmp-clang22/sdk-amd64/usr/bin/clang++-22 \
  --target=x86_64-linux-gnu -march=x86-64 -std=c++23 \
  -Wall -Wextra -Wpedantic -Werror -O1 -g \
  -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer \
  docs/experiments/async-model/main.cc -o /tmp/llmp-async-model/clang-sanitized
/tmp/llmp-async-model/clang-sanitized

/tmp/llmp-clang22/sdk-amd64/usr/bin/clang++-22 \
  --target=aarch64-linux-gnu --sysroot=/tmp/llmp-toolchain-smoke/sysroot \
  --gcc-toolchain=/tmp/llmp-toolchain-smoke/sysroot/usr \
  --ld-path=/tmp/llmp-clang22/sdk-amd64/usr/bin/ld.lld-22 \
  -resource-dir=/tmp/llmp-clang22/sdk-arm64/usr/lib/llvm-22/lib/clang/22 \
  -march=armv8-a -std=c++23 -Wall -Wextra -Wpedantic -Werror -O1 -g \
  -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer \
  docs/experiments/async-model/main.cc -o /tmp/llmp-async-model/aarch64-sanitized
scp /tmp/llmp-async-model/aarch64-sanitized spark:/tmp/llmp-async-model-san-20260922
ssh spark /tmp/llmp-async-model-san-20260922
ssh spark rm /tmp/llmp-async-model-san-20260922

g++ -march=x86-64 -std=c++23 -Wall -Wextra -Wpedantic -Werror -O1 -g \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  docs/experiments/async-model/main.cc -o /tmp/llmp-async-model/gcc-sanitized
/tmp/llmp-async-model/gcc-sanitized
```
