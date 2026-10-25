<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ready NormGate first screen: refused evidence

The [registered captured-event screen](READY-NORMGATE-CAPTURED-EVENTS-REGISTRATION.md),
commit `e3c775e84aea0156a953db50bfa92e7252744227`, ran once on `spark-b`.
Alternative bookends are 4,800 and 5,760 ns, a **20%** difference, exceeding
the prospective 10% limit. The run is refused. No candidate row, accepted
median, classification or data installation follows from it. Its original
bound remains unchanged and the protocol was not retried.

All 36 ordered warmup, sample and bookend observations were emitted. Both
changed original-IR frames and every arm's CPU-reference, gamma and actual
3/5-node capture-inventory checks pass before the drift refusal. Each capture
contains one complete body with independently restaged original inputs. The
benchmark then aborts through its refusal path; supervisor exit is 134, without
normal teardown or a completion receipt. Source and binary checks match before
and after; a separate supervisor check reports no remaining GPU processes.
This is not successful timing retirement.

The run uses source manifest
`304f975ecd39a67720c592813e353e4842df80846c592519f45280d9a597bba8`,
binary `632dc6488da877436e327033ba302cb33f20ce2bc9b2d1d7cd794360c037d856`,
and protocol `9c2681420e189b79d9a5226741e6c516c7b92363da5d4fad31a13fa154736671`
from its registration. Raw evidence remains outside Git at
`/tmp/llmp-m35-orchestration/u6-resume/normgate-captured-r49/screen.log`
(SHA-256 `e72b971d79e2258d00e2c84e62a379ea3e33362ad4dd554c6a68bfc8d071c167`).

A separately [registered median-bookend screen](READY-NORMGATE-MEDIAN-BOOKENDS-REGISTRATION.md)
subsequently ran with nine alternative observations at each bookend, the same
10% bound and all 52 observations retained. Its independently reviewed
[result](READY-NORMGATE-MEDIAN-BOOKENDS-RESULTS.md) is accepted. This measures
median drift, can conceal occasional slow executions and does not establish
the cause of the refusal above. No observation from this refused run enters
that result.
