<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen captured four-request GDN recurrence

One captured-operand operator O/C/O reduced four-request recurrence latency
from 25.515377 to 18.750844 microseconds, a 26.5116% reduction or 36.0759%
inverse-latency gain. All four complete F32 outputs were finite and byte-exact
against their captured scalar goldens; captured inputs, recurrent states and
guards remained unchanged. This is a positive, graph-disabled operator result,
not a production default, full-model speedup or quality qualification.

TensorFold's current CUDA path batches independent request windows through
DeltaNet recurrence, whereas native waves share selected products and retain
per-request recurrence launches. This screen keeps native F32 arithmetic and
state ownership; it does not port TensorFold's BF16 values/output or deferred
accepted-state folding.

The Spark B diagnostic used four distinct common-v2 rendered prompts of 8256
tokens, context 33792, chunk 4096 and drafter head 47172. Each slot used the
existing forced prefix and one-pass draft control before ordinary scalar
verification. Private hooks captured the first target GDN node in each slot:
head width 128, 48 value heads, 16 key heads, two rows and update=false.
Six operands/state were copied before the ordinary recurrence launch, and
its complete output immediately afterward, on the same owned stream.
The original launch still ran; no later consumer read a skipped recurrence.
Each capture owned independent allocations rather than borrowing transient
graph storage. Scalar verification then settled through existing
Accept/Rollback.

The recurrence body was extracted unchanged into a force-inline device
helper. Both the original launch wrapper and the private cohort wrapper use
that body and the selected CUDA numerical flags. The baseline therefore
measures this private normal-wrapper refactor, not an unmodified production
binary. The ordinary checker remains unchanged. The coherent candidate
header/archive and selected normal numerical link closure were authenticated;
only the private CUDA archive member and diagnostic entrypoint differed.

| Chronological arm | Seconds for 4096 four-request replays | Microseconds per replay |
| --- | ---: | ---: |
| Four serial launches before | 0.105611884 | 25.784151 |
| One cohort launch | 0.076803456 | 18.750844 |
| Four serial launches after | 0.103410087 | 25.246603 |

The mean serial time was 0.104510986 seconds. After/before serial latency
changed by -2.0848%. Both arms paid host packing of four records and CUDA
by-value transfer of 512 parameter bytes. There was no separate device-record
copy or adapter kernel. Each arm had 64 warmup replays; the single paid case
used 4096 repetitions per cell and included host submission and the final
stream fence. Capture, validation and file writes were outside this clock.
These observations do not establish a confidence interval or serving gain.

The original grid was 48×8×1, the cohort grid 48×8×4, and the block 32×4×1.
All slots had element strides
[q-head, q-token, v-head, v-token, beta-head, beta-token]
[128, 10240, 128, 10240, 1, 48]. Operand sizes for
q/k/v/g/beta/state/output were
[49152, 49152, 65536, 384, 384, 3145728, 49152] bytes.
The four recurrent state pointers were distinct. All 49,152 combined output
F32 values matched their goldens after warmup and every paid cell. Every byte
of all six captured inputs/state and every 256-byte prefix/suffix guard also
matched. These controls retained 28 raw operand/golden files and four complete
target-head files outside Git. The latter are completed diagnostic outputs,
not a cross-policy quality comparison.

Raw diagnostic CUDA allocations outside the catalog totaled 13,452,288 bytes
under a 32 MiB bound. They are declared private diagnostic allocations and
do not qualify a production or sampled-memory envelope. The native private
fixed catalog was 7,683,411,756 bytes, activation 6,096,420,864 bytes, scratch
884,998,144 bytes and execution budget 90,481,069,868 bytes. Replay completed
before EndRequest while the CUDA context/stream remained owned; its fence
preceded diagnostic frees and native TearDown. This graph-disabled scope is
separate from the earlier graph-enabled four-head/full-state recovery control.

Three excluded setup attempts remain preserved. build-r1 failed compilation
on signed iterator arithmetic in host golden copying and loaded no model;
resize plus memcpy corrected it. The first build-r2 supervisor stopped on an
incorrect retained-helper command path before preflight or compilation.
The corrected build-r2a supervisor completed the private CUDA compile/link in
14 seconds. run-r1 stopped after one completed scalar slot because the driver
used wave-only DiscardVerify after ordinary Verify; no operator timing ran
and the native owners retired. Existing scalar Accept/Rollback corrected
settlement without changing captured pre-verify buffers or CUDA arithmetic.
The driver-only replacement build-r3 reused the unchanged reviewed CUDA
object/archive and completed in 7 seconds. Successful run-r2 completed in
29 seconds; its owned diagnostic child took 25.297837 seconds, returned zero,
was reaped and reported no cleanup error. No suite, ladder, successor factor
or production rewrite ran.

The source-only kit is retained at
`/home/pmeenan/scratch/m3-qwen-gdn-cohort/frozen-r3` on B.
Its private files are `control.cc`, `control.py`, `jitllm_fused.cu` and
`gdn-screen.inc`; earlier frozen kits remain unchanged.
Local raw receipts, outputs and supervisor records are under
`/home/pmeenan/scratch/m3-qwen-gdn-cohort-records/`.
The aggregate external result is
`/home/pmeenan/scratch/m3-qwen-gdn-cohort-report.md`.

| Provenance | SHA-256 |
| --- | --- |
| Driver | adff549a6dfbacc818137558849cc29aad1011a0a71172717de5aef1a93a31c4 |
| Controller | 98e35bd826af175d065235f2c87efd7b68399072dc604437a84123f7c8033598 |
| Private CUDA source | 82713c67faaf986c992d17fc623ce3977dd25fb7aea0650deed418b3b31968ca |
| Private include | 74a10268eec1ab6060789bd83c96fb73fb1acdc3bc8f11e7559863525064316d |
| CUDA build-r2 receipt | c5fdfd51fcc04c95c50eb7d7b94ca4fa7437db2ac4081f0c1b873e0fe65a0c88 |
| Driver build-r3 receipt | b0e24c2db421e87f1f4f5e673c96d75277dc7581f9f07b37f939d8bc6338efbd |
| Successful run receipt | c4e5d85740d7a010ed116448b6a880d2209d385a24d5a14557fd76f45efaeef4 |
| Operator proof | 97cf6e0ef55023103bdd2cd5250a3e363c5da97d85bcfb290e53ab60e9e898a8 |
| Native completion/retirement proof | ab7831a9d0b7f88cd670850fe590bd82d1b762c45c60aa743ef91c661d20b848 |
| Diagnostic executable | 9bd7db199fa419a521f00f306d6962bbaa8e02341efa509afb9acb0b0ab30bcc |
| Actual link receipt | a5243d1757eec4fa443bbffae0af2195b4d4c9e75ffdd81898202b31f519a2c0 |

The successful receipts record all 28 operand hashes, all four complete
rendered-input hashes and the actual numerical/runtime closure. Golden-output
SHA-256 values for slots 0–3 are:

- `e86d13efa54b647524cffbbe50325fa111a75454e1ffa933097c331abc8baddb`
- `ea6c2b78286474b00a1437b29f90c9e98e535f0ea6e08626cf4e7faee452e801`
- `0e410b4328b99eff0fa13339fd4581b1219b6f01bf034ff43a85b6f4e7876b7a`
- `e051d3db8e4da776b939c7558bbc8a2e882b960f827028b1aec801bf2ec8f5d2`

B was explicitly returned after a strong retirement probe found 117.230 GiB
free, no GPU/container/native-model work, and zero busy GPU jobs. Successful
build/run supervisors 3981842/3982645 both completed with zero status.

The next candidate remains unstarted: a real wave composition of four native
GDN recurrence nodes using bounded per-request records, preserving F32
arithmetic, update=false and independent state/output ranges. Likely source
files are `src/engine/qwen38_wave_plan.{h,cc}`,
`src/kernels/ggml/jitllm_fused.cu` and a private launch descriptor.
It still needs graph/dependency/funding authentication and a paid model
comparison; convolution, normalization and Accept/Discard behavior must
remain unchanged.

