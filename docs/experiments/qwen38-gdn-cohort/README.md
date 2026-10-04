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

## Native wave integration screen — 2026-10-04

**Exact, but no measured wave gain; not adopted.** The private integration
composes four matched, non-writing GDN verifies into one launch, using the
unchanged F32 arithmetic body and independent input, state and output ranges.
It places all four outputs together before activation allocation. Existing
lane dependencies fence each slot's preceding work before the combined launch.
By-value records need no new device allocation or packing kernel; host packing
is inside the verify clock. Ragged, one-row and smaller groups retain the
original launch. Convolution, normalization and Accept/Discard stay unchanged.

Spark A (`spark-c4e2`), driver 580.178.04, CUDA 13.4.92, SDK
`aarch64-e0a0c85c42806fb1`. Each fresh process prefills the first four prompts
in `fast-swap/prompts.json`, then generates 96 outputs per slot. The context
is 16,384, chunk 4,096, vocabulary head 47,172, lanes on and graphs on.
The existing `--check wave` uses depth two, irrespective of the driver's
`--draft 3` setting. These are short in-process controls with the benchmark's
drafter-injected prefill, not 8K HTTP requests or runtime-prefill timings.

| Chronological arm | Draft median ms | Verify median ms | GDN cohort launches |
| --- | ---: | ---: | ---: |
| Unmodified copied benchmark | 21.434 | 107.109 | 0 |
| Refactored, disabled before | 21.511 | 107.228 | 0 |
| Combined recurrence enabled | 21.594 | 107.399 | 1,404 |
| Refactored, disabled after | 21.451 | 107.018 | 0 |

The enabled verify median is 0.258% higher than the disabled bookends'
mean (107.123 ms); their after/before movement is -0.196%. This single screen
does not establish a confidence interval. The earlier 26.51% operator gain
against four serial launches did not transfer to a native wave already using
concurrent lanes. This does not exclude other shapes or recurrence schedules.

All four arms complete 43 waves and 384 generated tokens. Every slot's
complete target-row hash, generated-token hash and final initialized-state
hash equals the copied unmodified benchmark. Each arm captures two target
wave graphs and replays 38; the enabled arm executes 1,404 cohort launches,
while the disabled arms execute zero. Full-model output/state agreement
therefore covers actual capture/replay and ordinary accepted/rejected drafts,
not merely an operator golden. There is no swap/recovery, sampled quality,
HTTP throughput or memory-envelope qualification of this candidate.

`gdn-wave-build3` and `gdn-wave-screen1` both finish successfully under the
installed GPU supervisor. All four model children return zero and are reaped
(31.28–31.47 s each); admission and retirement probes pass around every arm.
Final retirement finds 116.296 GiB available, no GPU/container/native-model
processes and no busy or waiting GPU job. The diagnostic prototype remains
outside main. No production suite or HTTP ladder ran for this rejected screen.

The patch against `ce8173b`, per-file source hashes, controller and raw outputs
are retained in workstation scratch
`/home/pmeenan/scratch/jitllm-m3-qwen-gdn-waves-2026-10-04/`.
Spark A retains the controller and outputs in `~/scratch/gdn-waves/`, and the
private build in `~/src/jitLLM-wt/gdnw4/`. Reproduction uses that source patch,
the pinned target artifact `c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`,
drafter `8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`
and `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6/tokenizer.json`.

| Integration provenance | SHA-256 |
| --- | --- |
| Source patch | 3ce45bae14634fcb0610c5a6258285c0c9d0e249a9b8b527633c5bc19936f47f |
| Per-file source hashes | 00542a34ee3a8e255749010544756dd28b336542bae4299b37e031272913915d |
| Controller | 2fa13166684f52844f0d10c63bfeec8250698b3b49b250d4028d2bff7f49237e |
| Prompt source | d212009dadf1ddbf945c8dc7ad0214ba444236baf57c8ed9019c3ebe6b0805b4 |
| Unmodified copied benchmark | 4d3228cc89520db7c7866d78a4c5db6ac5bdbbcd79e4363d192063e9ab52ae93 |
| Refactored benchmark (both settings) | 8a98e510ee702042eaa256f0a7760cfac010d04884c188c5be20a9bca2f2d2d4 |
| Successful controller receipt | 69435eba8a3758f8294eae1de8b0a2852ef805222350a5431d8cc0bef6351096 |

