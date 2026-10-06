<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded assistant component protocol

Use the approved prepared target artifact
`4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3`
and assistant artifact
`1040a0299a459e00ad0a77efd77bd319ac593986ba2c9ef29eb03d07ce97db42`.
Actual native component setup parses no vocabulary itself: its caller funds,
authenticates and decodes both kept metadata files, validates each tokenizer
and passes raw strings, merge ranks, finite scores and kinds to
`SetupAssistant`. That function runs the paired compatibility check before
committing allocations or registrations. The successful immutable pairing is
retained; parser spans are not retained.

For native component controls, enable `retain_features` explicitly before
Setup. This disables final-layer narrowing while keeping selected head and
feature row IDs independent. Complete owner 0 tokens
`[2,818,5279,529,7001,563]` and owner 1 tokens `[2,818,5279,529,818]`.
Acquire same-slot frozen borrows only under the existing held request. Each
chain starts with that slot's latest final-normalized feature and pending
canonical anchor, then uses separate component recurrence. Fund caller full
heads/features and initialized-cache/retained-feature witness copies before
allocation. Compare all completed bytes twice, including peer bytes during
one-owner execution. Do not restore an original fixture as a native checkpoint.

## Original-stage-zero arithmetic control

The original-image companion supplies the following eleven **input-only**
files for C1 owner 0 after its official retirement:

- `metadata.json`, `feature.f32`, `anchor.i32`;
- `local-k.f16`, `local-v.f16`, `global-k.f16`, `global-v.f16`;
- `local-positions.i32`, `global-positions.i32`;
- `local-membership.u8`, `global-membership.u8`.

Admit their exact lengths and SHA-256s from the external stage-zero-only
manifest, SHA-256
`bf5c5c3dba1ab22a069475d7fc3ce3ec1746cefa8ceae4e11b7d07088b803923`. Never copy the
whole original owner directory: it contains other witnesses, and its sibling
recurrence folders are withheld until native endogenous calibration is frozen.
Require P64, query64, feature63, local layer28/global29, capacities1280/4096,
read256, canonical anchor108, finite features/cache, zero read padding and
literal physical position/membership maps. Target state initialization ends
before position64; the assistant query64 does not allocate or expose cell64.

Build with the declared toolchain and prepared sources:

```sh
mise run build -- spark-native --locked
```

Before model acquisition, freeze all changed source hashes, executable hash
and exact admitted input hashes. Run the manual target through the installed
supervisor, with a new output directory:

```sh
jitllm_gemma_assistant_fixture STAGE0_DIR STAGE0_MANIFEST NEW_OUTPUT
```

The helper owns separate cataloged mapped cache operands, pinned upload and
witness storage, weights, activations, scratch and output storage. Under its
held request it checks complete execution coverage, then uploads each cache
from a pinned span in one completed Job before reusing that span. It launches
independent one-step and three-step chains, each repeated from stage zero;
query64 remains constant. It writes owner-only, exclusively created raw F32
heads/projections after completion, with checked flushes. Every repeat retains
exact cache bytes, with initialized padding. Unknown completion retains the
entire node/model/context/graph/storage lifetime bundle.

After official successful retirement, check all full head/projection first
and repeat files for byte equality, unchanged source/binary/input identities
and all cache witnesses. Exclusively freeze the receipt **before** viewing
original recurrence features, anchors, heads or post-projections. The measured
native receipt hash is recorded in `provenance.json`; its own movement is zero.

## Posthoc comparison with unchanged native source

Only after that freeze, admit the original C1 `steps-3` files:
`incoming-feature.f32` (33,792 bytes), `incoming-anchor.i32` (12 bytes),
`heads.f32` (3,145,728 bytes), `postprojection.f32` (33,792 bytes).
The external release manifest has SHA-256
`8be990edd96ad56e4fbf5a8d480830896750cac454cbf0ad402222e043005375`
and ties the original receipt `856d622b…` to native receipt `fd742b16…`.
Verify exact complete file lengths/hashes before the paid replay; keep all
files owner-only and unchanged. The optional branch was compiled before the
native endogenous freeze:

```sh
jitllm_gemma_assistant_fixture STAGE0_DIR STAGE0_MANIFEST NEW_OUTPUT ORIGINAL_STEPS3
```

This supplies each original incoming feature/anchor independently, while
using the same native graph, binary, query64 and caches. First incoming bytes
must equal stage zero; all values must be finite/canonical. Compare complete
joined native three-step heads/projections against complete original files,
including both native repeats and the one-step first row. `compare.py` records
that exact comparison for the external measured directory layout. Complete
byte equality supports zero distribution/score difference directly; otherwise
retain the failure and perform aligned full-row analysis without widening the
frozen native allowance. No native-target feature or checkpoint is modified.

This protocol contains no competitive latency measurement, accept/reject
verification, sampler, speculative serving, optimized joined assistant policy
or original C2 cross-engine comparison. Those are separate qualifications.

## Closed 31B stage-zero replay

The explicit leading `--profile31` option selects prepared target
`32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08`
and assistant
`447a5c20a0a25632bf35e118d5dde1867a182cd209b9ccd3afe93866c3696120`.
Legacy invocations keep the original 26B input sizes and behavior. Admit only
C1/P64 stage-zero inputs after the original 31B own-freeze: feature width 5376,
local58/global59, D/head pairs256/16 and512/4, canonical read256, physical
capacities1280/4096, absolute positions0..63, query64 and feature63. The
externally fixed metadata length is positive and at most 2 KiB; all eleven
file lengths and hashes plus profile/descriptor semantics are checked before
prepared-model allocations. The first anchor is the actual original full-head
argmax, not the historical 26B anchor108.

The existing 128 MiB caller host grant precedes manifest parsing and vector
construction. Input payloads total less than 7 MiB, retained incoming/output
metadata is bounded by the fixed C1/one-or-three-step shape, and the single
16 MiB pinned upload/witness buffer is separately charged by RunnerResources.
The four borrowed physical tensors total 52 MiB (two 10 MiB local tensors and
two 16 MiB global tensors). Upload each entire tensor from the pinned span:
canonical read inputs occupy the first 256 cells and all remaining cells are
explicitly zero. Each upload retires before span reuse. After every assistant
step and chain, copy all four complete tensors back through that same buffer,
compare canonical input bytes and zero tails, and hash the complete witness.
Only these four immutable operands are initialized or claimed as protected;
they are not a serialized native target checkpoint. Global V uses its separately
authenticated raw source. Complete heads/features are published only after
Job retirement; feature recurrence is separate from constant target P64.

Run the unchanged one-step and three-step repeat controls using:

```sh
jitllm_gemma_assistant_fixture --profile31 STAGE0_DIR STAGE0_MANIFEST NEW_OUTPUT
```

Freeze complete native heads/projections, cache witnesses and official successful
teardown before viewing the original recurrence. All full-row differences are
reported without borrowing a quality bound or widening tolerance. Any later
same-input posthoc replay uses the already compiled optional input branch and
requires a separate frozen-input release. No runner/serving defaults or math
policy change accompanies this manual component extension.
