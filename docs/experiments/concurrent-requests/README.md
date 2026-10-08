<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Concurrent generation measurements

## TensorFold: first 8K screen

TensorFold's explicit four-request deployment increased fresh
prompt-plus-generation throughput from 23.06 to 41.05 tokens/s in plain
mode (1.78×), and from 31.62 to 40.53 with speculation (1.28×). All fourteen
concurrent outputs matched their matching solo controls exactly. These are
one observation per cell, with serialized fresh prefills; they do not
establish pure decode scaling or parity with llmpalooza's different weight format.

Spark B, 2026-10-01, one CUDA rank, image `llmp-tensorfold:71377a53`
(`sha256:1a2afff2bd001cdea746d96bdb7614a4f937caa28e1646babd9acb951ea3aa48`),
TensorFold 0.3.6.2. Model: Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP at
`dadefa80`, affine 4-bit, group 32. This is a cross-quantization baseline
beside native/Mia NVFP4. The image and eight runtime source files were
authenticated, as were the model revision, configuration and read-only
file inventory; this run did not freshly hash every weight file.

Settings: `--parallel 4 --context 33792 --kv-dtype bf16 --mtp-drafts 6
--mtp-confidence 0.30`, Docker memory cap 110 GiB. Literal prompts contain
exactly 8,192 IDs in both the original and affine tokenizers. Every request
is greedy, has a 256-token cap, and has unique beginning/end markers.
Plain requests use the engine's per-request `draft: false` switch. All
timed requests reported `cached == 0`. The six bursts and their fourteen
matching solo controls completed at the length cap, without errors or EOS.
Solo controls can reuse their repeated prompt and are excluded from timing.
See the [protocol](protocol.md) for the common-barrier and cache conditions.

| Mode | Requests | Actual output tokens | Burst time (s) | Aggregate tokens/s | Median / worst request latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain | 1 | 256 | 11.101 | 23.060 | 11.101 / 11.101 |
| Plain | 2 | 512 | 16.016 | 31.969 | 16.014 / 16.015 |
| Plain | 4 | 1,024 | 24.948 | 41.046 | 24.944 / 24.948 |
| Speculative | 1 | 256 | 8.097 | 31.617 | 8.097 / 8.097 |
| Speculative | 2 | 512 | 13.485 | 37.968 | 13.349 / 13.485 |
| Speculative | 4 | 1,024 | 25.264 | 40.532 | 25.108 / 25.264 |

Aggregate throughput is actual completion tokens divided by first
submission to last terminal event, including fresh prefill and scheduling.
It is not the sum of per-request rates. First visible text arrived at the
following offsets from each request's own submission, in input member order:

| Mode | Requests | First visible text (s) |
| --- | ---: | --- |
| Plain | 1 | 3.595 |
| Plain | 2 | 6.744, 3.390 |
| Plain | 4 | 10.151, 6.787, 13.531, 3.421 |
| Speculative | 1 | 3.488 |
| Speculative | 2 | 3.494, 6.967 |
| Speculative | 4 | 13.964, 3.530, 10.484, 7.010 |

The roughly 3.4-second staircase shows fresh prefills running serially.
At four requests the whole-burst speculative rate was slightly below plain.
Warm conversation branches and arrivals during an existing decode are
needed to separate shared decode gains from prefill interference. SSE
chunks can contain multiple tokens, so these offsets and retained chunk
gaps are delivery measurements. This small screen supplies no stable p99.
The default single-request graph deployment, repeats and 32K remain open.

All 28 responses retained full output IDs, text, usage, finish reason and
wire events. All fourteen solo/concurrent comparisons matched IDs, text,
usage and finish exactly. The analyzer independently reconstructed all
six burst rates from per-request records. This is a same-engine isolation
check; it is not a model-quality comparison across weight formats.

Across 1,489 memory samples, node available memory ranged from
125,766,742,016 to 34,877,444,096 bytes, an observed drop of
90,889,297,920 bytes (84.65 GiB). Peak sampled server RSS was
30,775,250,944 bytes (28.66 GiB), separate from unified GPU backing.
These samples include setup and file cache; they are not interchangeable
allocation totals or an incremental concurrency-memory estimate.

Startup was 143.446 seconds; supervised total time was 373.484 seconds.
The same live server PID/start/image and source inventory were confirmed
before and after. Client and launcher retired, the container was absent,
and the strong post-run gate reported 117.146 GiB available.

## Provenance and exclusions

Raw records are external on Spark B at
`~/scratch/m3-concurrent-r1/tf8k-model-r2/` and on the workstation at
`/home/pmeenan/scratch/m3-concurrent-records/tf8k-model-r2/`. The input
receipt and analysis are beside them as `inputs8k-r2/receipt.json` and
`tf8k-analysis-r1.json`. The external frozen prepare/client/run/analyze
scripts are Spark-only; the model controller authenticates their identities
and the prepared input receipt. Repeating this screen requires those inputs,
the pinned image/model, and the same settings above.

| Input or result | SHA-256 |
| --- | --- |
| Prepared input receipt | `a947338b1d0fb09822bb37d19747711926e1e4d37cbfacc9e5cd75cc689bb791` |
| Prepare script | `47c4b02d6194a94040ebcc4d0c22c0b8ddf595e08390f44a1efd7b5ae2fa53a7` |
| Client script | `71e5497f1c24e31fd28d201ed95663d758513584bdafea337d9f2a6a6daadea0` |
| Model controller | `41033d5901fa3c605927e99c856068a44bf38e0bfc4d48606b22217d07420504` |
| Analyzer | `fc67308c4d49cb4dff6df761bca0dab77e240b54c91840721e7a897eaaed4dba` |
| Outer complete receipt | `8236f3f6ac6d00c4c93148c584a0e8ad7a4105047b856596fa11d8d351af0c02` |
| Client complete receipt | `b8aad1975031c2acb189f9e18823a3385e28c709502689f990394f038ecce5c5` |

Input preparation r1 failed before model work because its container UID
could not write the owned output directory. Preparation r2 ran as UID/GID
1000 and passed. Model job r1 was refused before model load: the strong
process check interpreted `tensorfold` in the supervisor's job name as a
model process. The neutral r2 name passed the unchanged check. Both failed
records are preserved and excluded from timings; neither ran model work.

## Mia: first 8K speculative screen

Mia's fresh prompt-plus-generation throughput increased from 23.44 to
40.26 tokens/s at four requests (1.72×). All fourteen requests completed
with 256 output IDs, finite returned log-probabilities, the original 8,192
prompt IDs and zero cached prompt tokens. Every matching solo output
differed from its burst member, including the one-request repeat. This
completed performance screen supplies no output-equality or quality pass.

Spark A, 2026-10-01. Recipe `b8439110eec0230facbe4ddf0dffe01b8f769be0`,
image `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`,
resolved vLLM `0.1.dev20073+g8e685d198`. Checkpoint:
Mia-AiLab/Qwen3.8-Flash-Next-NVFP4 at
`925d7be6c14c6c9442ef83e8f05b5a3c39304f69`. The selected draft vocabulary
has 47,172 IDs. One rank, fixed MTP depth three, FP8 KV, BF16 recurrent
state/head I/O, `DET=0`, context 33,792, four request slots, chunked
prefill 2,048, GPU utilization 0.786 and 26 GiB host reserve. Compilation
mode zero, `FULL_DECODE_ONLY`, graph capture sizes 4/8/12/16; asynchronous
scheduling and profiling are off. Launch/configuration and source pins were
checked before and after. The model identity scope is revision, index and
unchanged file stats, without a new full-shard hash.

Use the same seven speculative input members as the TensorFold screen,
with literal string prompts, no inserted special tokens, temperature zero,
256-token caps and one observation per cell. Unique cache salts isolate
requests. Returned prompt IDs were independently compared with the exact
prepared packed IDs. Actual KV capacity was 601,014 tokens, above the
configured length; this is capacity telemetry, not concurrent allocation
usage. Plain mode requires a separate deployment and remains unmeasured.

| Requests | Actual output tokens | Burst time (s) | Aggregate tokens/s | Median / worst request latency (s) | First visible text, input order (s) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 256 | 10.923 | 23.437 | 10.923 / 10.923 | 4.853 |
| 2 | 512 | 15.528 | 32.972 | 15.440 / 15.528 | 3.827, 8.596 |
| 4 | 1,024 | 25.435 | 40.260 | 25.037 / 25.434 | 12.427, 16.375, 8.580, 3.811 |

All seven controls matched usage and finish reason, but their full IDs
and text differed after common prefixes of 35–44 tokens. Variation already
exists in the C1 repeat, so these records do not isolate a concurrency cause
or measure a quality loss. Timings are retained descriptively under the
declared `DET=0` condition. This is separate from the historical deterministic
quality oracle. Aggregate speculative counters cover whole bursts; they
are not attributed to individual concurrent requests. SSE token batches
and their gaps supply delivery timing, without a per-token kernel claim.

At four requests, Mia and TensorFold's speculative fresh-burst rates are
close (40.26 and 40.53 tokens/s). Their weight formats, state precisions,
draft policies and Sparks differ. Fresh prefill and scheduling remain
inside both endpoint timers, and there is only one observation per cell.
This comparison cannot establish same-format decode or quality parity.

Readiness took 792.508 seconds; the matrix and controls took 120.505
seconds, and the controller took 917.040 seconds including retirement.
Minimum sampled node available memory was 15,428,767,744 bytes; observed
drop was 110,410,014,720 bytes (102.83 GiB), including setup/cache. This is
node-wide memory, separate from any process RSS or allocation total.
The same live server identity and static configuration were confirmed
before and after. Load/client/retirement commands completed and were reaped,
the container was absent, and the retained retirement log reported
117.341 GiB available.

Raw receipts and the descriptive analysis are external at
`/home/pmeenan/scratch/m3-mia-concurrent-r1-records/`; the supervised job
on Spark A is `mia-concurrent-spec-r1`. The prepared receipt binds the
input receipt listed above, actual recipe source, launch assets, environment
and model inventories. The analyzer reconstructs all three rates and
seven comparisons from completed per-request records.

| Input or result | SHA-256 |
| --- | --- |
| Prepared receipt | `dcdc6985a16096d45da435ac8470084e391ea365190884a9b892e32d99a8ee19` |
| Client script | `bfe5aa8a1aa318d24e28a0352463bc2aac44693c4baae8884aada0d61e4cf61c` |
| Model controller | `53b0cc57c2ae77f2d56c6984d0257c3c9c2ceafc86b3acf455d9f02573e342a1` |
| Analyzer | `8ece3c0e4130c82e4fc444f6bd86449d35647585758efc7332fd850c0423ec6c` |
| Complete launch receipt | `1e4585acbde5312cd620b8ce35c3f94d0b624b11c367455f586a328595beed48` |
| Complete client receipt | `90562f54abf4aae571d5fcd9c3258218e03f995081addac6a6b9cdcc0939cbf8` |
| Descriptive analysis | `f0732207dd39601ca8347af63e16c085537d63176f351946a8b6443474764e8c` |

## Llmpalooza: first 8K queue screen

The current native service runs one request to completion while other
clients wait. Its aggregate fresh-prompt throughput stayed near 20.2
tokens/s in plain mode and 27.3–28.5 tokens/s with speculation as the
number of clients increased from one to four. All 28 native responses
completed their 256-token budgets, and all fourteen matching solo/burst
comparisons had identical text, usage and finish. Native output token IDs
were not returned. The concurrent speed gate remains open.

Spark A, 2026-10-01. Native and Mia used this same Spark; the TensorFold
screen used Spark B. The native source was the authenticated 1,244-file
`4e9dd06` export and a locked, target-only runtime build with the actual
Spark A SDK and SDK-identical cuBLAS payloads. This diagnostic did not run
a fresh full unit suite or style pass, following the owner's experimental
workflow override. No production source or scheduling policy changed.

Both native deployments used target `c4fb47a9`, selected 47,172-row MTP
artifact `8600a998`, context 33,792, prefill chunk 4,096, F16 KV and F32
recurrent state. Speculation used the existing adaptive depth two/three;
plain mode disabled speculation. The two deployment configurations
differed only in speculation and private storage paths. All fourteen
frozen input members were authenticated against the common `a947338b`
receipt. Literal text requests used the original pinned tokenizer,
`add_special_tokens: false`, temperature zero and a 256-token cap. Every
burst and solo response reported 8,192 prompt tokens and zero cached
tokens. This checks authenticated text/tokenizer plus usage; it does not
independently prove native HTTP prompt IDs.

| Mode | Requests | Actual output tokens | Burst time (s) | Aggregate tokens/s | Median / worst request latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain† | 1 | 256 | 12.684 | 20.183 | 12.684 / 12.684 |
| Plain† | 2 | 512 | 25.274 | 20.258 | 18.954 / 25.273 |
| Plain† | 4 | 1,024 | 50.572 | 20.248 | 31.598 / 50.571 |
| Speculative | 1 | 256 | 9.383 | 27.285 | 9.383 / 9.383 |
| Speculative | 2 | 512 | 17.951 | 28.522 | 13.447 / 17.951 |
| Speculative | 4 | 1,024 | 36.639 | 27.949 | 22.832 / 36.638 |

† The first controller completed all plain requests and comparisons, then
failed because its memory sampler read a transient process status without
`VmRSS` after the service had exited. The outer receipt remains incomplete
and supplies no memory qualification. The same live service identity was
proven before and after the matrix; requested shutdown returned zero, the
service was reaped, and the independent strong retirement check passed.
The table retains these completed timings descriptively under root's
explicit instrumentation-only disposition. No successful plain requests
were repeated or replaced. The next controller stopped and joined the
sampler before deliberate service exit, froze a speculative-only mode,
and completed that separate deployment cleanly.

These are one observation per cell. Aggregate throughput divides actual
completion tokens by first submission to last complete response, including
fresh prefill, admission, queue waits and HTTP work. Submission spreads
were below 0.9 ms. Nonstreaming native replies expose neither first-token
timing nor an independent decode rate. Service logs explicitly recorded
successive queue waits; this is a queue baseline, not horizontal batched
execution. Same-engine text equality is not a model-quality pass or a
token-ID equality claim.

The speculative C1 result exceeds Mia's 23.437 tokens/s but is below
TensorFold's 31.617. At C2 and C4 it is below both fresh speculative
references, whose C4 rates are 40.260 and 40.532. Native C4 median latency
is 22.832 seconds, while its last request takes 36.638 seconds; Mia's and
TensorFold's worst requests take 25.434 and 25.264 seconds. Thus aggregate
throughput and per-request latency cannot be declared matched. The
comparators have different draft policies, state precisions and batching
rules; TensorFold also uses different affine weight quantization. Mia's
seven solo/burst outputs differed under `DET=0`. None of these endpoint
timings isolates a numerical kernel or establishes cross-engine quality.

Native listener/model-list readiness took 1.506 seconds in plain mode and
1.505 seconds with speculation. Those endpoints precede first model
page-in; they are not comparable to a fully loaded reference startup.
An excluded nine-prompt-token, sixteen-output-token warmup took 6.456
and 6.530 seconds and included initial paging. The plain matrix and its
controls took 177.136 seconds; the speculative matrix took 128.008.
Controller wall times were 190.284 and 141.365 seconds. The speculative
run's observed node-available-memory drop was 83,623,931,904 bytes and
peak sampled process RSS was 1,825,263,616 bytes. These are different
unified-node/process scopes and include setup/cache, not isolated GPU
allocation totals or incremental concurrency memory. Plain memory samples
remain unqualified. The receipt-bound retirement gates reported 117.286
and 117.284 GiB available.

Raw records are retained on Spark A at
`~/scratch/m3-native-concurrent-a-r1/run-r1/` and `run-r2/`; compact local
copies are `/home/pmeenan/scratch/m3-native-concurrent-a-model-r1-records/`
and `/home/pmeenan/scratch/m3-native-concurrent-a-model-r2-records/`.
The original plain job was `native-concurrent-a-model-r1`, 08:58:37–09:01:48
EDT, outer rc 1 solely for the sampler failure. The speculative-only job
was `native-concurrent-a-model-r2`, 09:10:23–09:12:45 EDT, rc 0. Both
services retired with rc 0. The descriptive analyzer authenticates the
prepared, launch, client and terminal receipts and reconstructs the
per-request metrics; its r1 exception is restricted to the exact retained
instrumentation failure and does not repair the original launch status.
Both descriptive reconstruction jobs completed with rc 0. The measured
runtime, actual SDK cuBLAS payloads and build receipts were copied without
rewriting the binary into the external `preserve-r1/` archive and their
hashes checked before and after; the source export and raw roots remain
in place.

| Input or result | SHA-256 |
| --- | --- |
| Target-build receipt | `0a016387befe4fb76476fe16a4ae3234d0e0d5b3e91246f5533b77cb5ae5ed0b` |
| Native runtime | `bec33fe429780623af7d577f6450dae32f6db9aff6253d0d722f55ec24abc27c` |
| Authenticated source map | `c71b616d848839b7526a78fd68d28c14185cce9777b95a2fe7617339c497232f` |
| Client source | `61de4ae8411a82cdaf46f12afad4bdb4ef51455f52913419800ad714f2dba85e` |
| Plain prepared receipt | `2fec63be31214218658c8514019bfe0f8e07a7b9460dc5ef3bf52fbf71f26ae5` |
| Plain outer incomplete receipt | `2b686aef2e7f4bc969b851ca7cea9b87451c9faae8492e629e22dfef570808aa` |
| Plain complete client receipt | `440fe27e6f7ca381015d2d9deff22b4ffb590b5133c9b24e3d35accf379af527` |
| Speculative prepared receipt | `3bd59444c06c63498e8d8921825115cc313670b28c4392471a135892c39e09cd` |
| Speculative controller source | `9e9be1e31db687d1971cade5a20ff7d63eaedfb540238790a3d80001233778ca` |
| Speculative complete launch receipt | `15ab4a32beb9a6ea1a78eb54568f927da56d8f08757ad84e7faac61cfd4fd618` |
| Speculative complete client receipt | `8f684ad80258d85e5028ed67b06aedf1e3983a49e2635a28b5435b1c93a2f76a` |
| Descriptive analyzer source | `fcd10073979882f3b43c2442694c1cea323dbdb460ae6198ed374831d23822ab` |
| Plain descriptive analysis | `3abad2b1cc99d1df70b9dee7fb03c89b6ac288d4dcf264fd771d3f8d8f3fc59b` |
| Speculative descriptive analysis | `ae198ae21aadc5abf687d88c1c3360ebb19b03c880fc65f16ce79e97e8498b86` |
| Native preservation receipt | `72278232c7b4a846ab54429900277c06c03af5aee75c0360a0420cb088c78630` |

Continuous batched native execution, warm branches, longer contexts and
repeated decision-relevant cells remain open. This diagnostic changes no
defaults and closes no concurrent parity or quality gate.

## TensorFold: repeated strict-prefix warm screen

On the existing four-stream TensorFold deployment, warm aggregate throughput
increased from a median 57.53 tokens/s for one request to 82.17 for two and
109.81 for four. All nine bursts and all 21 matching fresh solo controls
completed their 256-token budgets, and every warm/solo output ID, text,
usage and finish comparison was exact. Four-request observations varied
by 14.01% from minimum to maximum, exceeding the preregistered 1.1 range
flag. These are retained descriptive results, with no native concurrency
parity or deployment-default conclusion.

The client matrix completed, but the outer controller remains failed.
After intentional server retirement, its sampler encountered a process
status with no `VmRSS`. This is the sole recorded cleanup error; actual
launch return code is zero, its client is reaped, the owned container is
absent and the independent strong retirement check passed. Memory
qualification is unavailable. The successful paid request work is retained
without a model rerun or replacement of the failed outer receipt.

Spark B, 2026-10-01, 09:21:39–09:28:58 EDT. This uses the same immutable
TensorFold 713 image/runtime 0.3.6.2, affine four-bit checkpoint, context
33,792, BF16 KV, four streams, maximum six MTP drafts and confidence 0.30
as the initial fresh screen. C1 also runs on this four-stream deployment;
it is not the default single-stream configuration. Requests use temperature
zero, literal text, speculation enabled and output cap 256.

CPU preparation froze 21 distinct 8,192-token prompts, balanced across
orders C1/C2/C4, C4/C2/C1 and C2/C1/C4. Both authenticated tokenizers
produce identical IDs. Each prepared 8,191-token seed round-trips exactly
and is the strict proper prefix of its target, with a one-token tail.
All 21 first-64-ID prefixes are distinct. The 105 exact prepared files and
ten installed runtime source identities were authenticated before and
after the matrix. Prompt IDs are not returned by this endpoint; prompt
identity rests on the frozen text/tokenizer proof and reported usage.

Before each burst, its independent seeds run serially with a one-token
cap and cached count zero. All 21 timed target responses report cached
8,191, while all 21 fresh same-target solo responses report cached zero.
The original runtime retains the seed prompt snapshot before its first
output, so that output does not become part of the reused prefix. The
timed warm request includes prefix-state restoration, one-token tail
prefill, adaptive drafting/verification and HTTP completion. This is not
an isolated decode-kernel timer.

| Concurrent requests | Three aggregate rates (tokens/s) | Median | Max/min | Median of burst median latencies (s) |
| --- | --- | ---: | ---: | ---: |
| 1 | 56.258 / 58.260 / 57.528 | 57.528 | 1.0356 | 4.450 |
| 2 | 80.507 / 82.169 / 83.858 | 82.169 | 1.0416 | 6.194 |
| 4 | 103.719 / 109.809 / 118.247 | 109.809 | 1.1401 | 8.824 |

The rate denominator is first barrier submission to final `DONE`, using
actual complete output tokens. Submission spreads stay below 0.8 ms.
First visible generated-text event ranges are 0.0818–0.1092 seconds for
C1, 0.0851–0.1615 for C2 and 0.1103–0.3682 for C4. These are visible SSE
event times, not individual token-kernel times. Worst completed request
latencies across the three observations are 4.550 / 6.360 / 9.873 seconds.

Loaded endpoint startup took 139.360 seconds; the request matrix took
296.210 and the complete controller wall was 438.649. The 21 seed request
latencies sum to 73.267 seconds, the 21 fresh solo latencies to 161.850,
and the nine warm burst spans to 59.949. These sums are not substituted
for measured setup or matrix wall and are outside the warm rate numerator
and denominator. The receipt-bound retirement reports 116.792 GiB
available, with GPU/container/native-model probes clear.

Raw 63 JSON responses, 63 bounded SSE files, source/config/live-process
proofs, the failed outer receipt, command terminal records and supervisor
logs remain outside Git. Local records are
`/home/pmeenan/scratch/m3-tf-warm-model-r2-records/`; Spark B retains
`~/scratch/m3-tf-warm-r1/model-r2/`. The earlier r1 invocation stopped
before model loading because its inputs argument named the receipt rather
than its directory; it is preserved separately and excluded.

| Input or result | SHA-256 |
| --- | --- |
| Immutable image ID | `1a2afff2bd001cdea746d96bdb7614a4f937caa28e1646babd9acb951ea3aa48` |
| CPU-prepared inputs | `cae29a57817cb3479d743517f4bc87b68e05f50abe22ed250d89a7452cdd3e00` |
| Preparation source | `939cf40123d50c351f72cd7458578139ad26cb662b2843919a3dbd5e223d97be` |
| Client source | `352cde12e4a8fdb6d76d252bbe776a3c9b67c1b5e0724d8f8ce96d8fb2121420` |
| Controller source | `7f4ccce5829f5b2726731d1720c00dead7a6395ea0ccbffba99094148faab646` |
| Protocol | `78d38b6d7714a24bc486d622a0d9372701216f00019a960f10e1fe0d9d05503d` |
| Failed outer receipt | `4a05feaaccdd32bf7f17704e1a6ac476439112125164424588bc7a15875c7e8e` |
| Complete client receipt | `c3bf0f07ec39e3b1fa08733ec2473e9ba1b788e4c1c115e874136fe1041ed6b9` |
| Independent strong retirement log | `d22f6ba0f0553901e4acd61081557ffabdf415753dbeaa0c1b8dd7506198da98` |
| Supervisor terminal record | `c20b8b7f69de82b8f62d54f12b2059c43fceb1def3b3d6175f9fdc41a842432c` |

Same-engine warm/fresh equality is not a model-quality comparison. The
fresh and warm screens have different paid work scopes and independent
inputs, so their rates do not isolate a cache algorithm's causal gain.
Native warm branches, longer contexts, default parallelism and
decision-relevant stable concurrent controls remain open.

## Mia warm concurrency: completed requests, failed outer instrumentation

Mia reused **6,656 of 8,192 prompt tokens** in every warm request, leaving a
**1,536-token tail**. The three C1/C2/C4 observations returned all 256 generated
tokens. Median complete-burst rates were **37.43 / 56.59 / 82.04 tok/s**.
These are descriptive results from a completed matrix whose outer controller
failed two bookkeeping checks. The original failure is retained; no model
work was repeated and no memory or model-quality gate is claimed.

| Observation | Concurrency | Complete-burst tok/s | Median request latency (s) | Worst request latency (s) |
| --- | --- | --- | --- | --- |
| 0 | 1 | 36.4551 | 7.0223 | 7.0223 |
| 0 | 2 | 51.8956 | 9.8380 | 9.8660 |
| 0 | 4 | 83.8897 | 12.1659 | 12.2059 |
| 1 | 4 | 81.6081 | 12.2271 | 12.5477 |
| 1 | 2 | 56.6116 | 8.8298 | 9.0440 |
| 1 | 1 | 39.0555 | 6.5548 | 6.5548 |
| 2 | 2 | 56.5922 | 9.0180 | 9.0472 |
| 2 | 1 | 37.4304 | 6.8394 | 6.8394 |
| 2 | 4 | 82.0377 | 12.1523 | 12.4818 |

C1/C2/C4 observed rate ranges were 36.4551–39.0555,
51.8956–56.6116 and 81.6081–83.8897 tok/s; max/min ratios were
1.0713, 1.0909 and 1.0280. The three observations use independently marked
prompts in balanced order. They are a bounded screen, not a confidence interval.
First generated SSE events arrived in 0.769–0.784 s at C1, 0.761–2.461 s at
C2 and 0.773–3.418 s at C4. These are visible streaming batch boundaries,
not per-token GPU timings.

All **63 requests** completed: 21 independent 8,191-token seeds with a
one-token output cap, 21 warm 8,192-token requests and 21 fresh same-target
solo controls. Every seed and fresh control reported zero cached tokens;
each warm request reported 6,656. Returned prompt IDs, raw SSE token IDs,
finite actual-token scores, text, usage and terminal markers were preserved
and independently reconstructed. All 21 warm/fresh output-ID and text
comparisons differed; all 21 usage and finish comparisons matched. DET0
differences remain descriptive and establish no quality pass.

TensorFold's completed warm screen reused 8,191 tokens and processed one
tail token. Mia's 1,536-token tail therefore prevents a comparison that assumes
equal cached work. The engines also retain their stated KV precision,
speculative depth, graph and scheduling configurations. Both screens use the
same frozen target and strict-prefix IDs, parallel capacity four and three
C1/C2/C4 observations; neither establishes cross-engine output parity.

The unchanged Mia recipe used context 33,792, four sequence slots, FP8 KV,
BF16 SSM, fixed MTP depth three, selected 47,172-token head and DET0.
Readiness took 781.528 s; the matrix took 381.780 s. Summed seed and fresh
control latencies were 81.757 and 201.987 s, recorded separately from warm
cell rates. Whole launch wall was 1,167.777 s. Startup, seed setup and fresh
controls are not included in the warm cell numerator or wall.

The outer controller accidentally returned the last protected mount's SHA
in its `arguments` field. Docker mount ordering changed that string even
though the live PID, start time, image, speculation configuration,
environment and mounted-source dictionaries remained identical. Its second
failure capped every retained file at 16 MiB, including the valid
31,262,224-byte aggregate containing all 63 response rows. Individual request
files stayed bounded. Both original errors and outer exit 1 remain immutable.
The sampler itself completed without error and stopped before deliberate
retirement. Its 4,645 Linux MemAvailable samples observed a minimum
17,131,659,264 bytes and a drop of 108,768,202,752 bytes; these remain
descriptive shared-node observations, with no repaired formal memory gate.
Maximum sampling interval was 0.262 s and the final sample preceded matrix
completion by 0.104 s.

The model client exited zero and was reaped, the owned container was removed,
and the original source/input/asset terminal checks passed. The original
retirement log recorded 117.374 GiB free and clear probes. CPU-only
reconstruction and preservation later passed in about one second, copied
255 hash-bound evidence files, and left the failed outer qualification intact.

| Evidence | SHA-256 |
| --- | --- |
| Frozen target/seed inputs | `cae29a57817cb3479d743517f4bc87b68e05f50abe22ed250d89a7452cdd3e00` |
| Actual prepared receipt | `37453b172044896ae600604bc658a901c065f7eef3136cbf852b2ddab5e8fdaa` |
| Failed outer run | `4a538d6a65d17f91cfb74a3837244c6656ef276ce947d09ad2b7ec640c3d8e8f` |
| Complete 63-request receipt | `12b6cb90d667b598cc1f74e618b5af089f9577c4dbb5e73702a879cc2f33cd10` |
| Original failed supervisor terminal | `f26c1a7cacfbcbac903ee77dcd85295968607ae928ea75863cfbc1f293e61850` |
| Reconstruction/preservation receipt | `c11f3ffd0e14ed13dff3ebec8d65bd8bffd55b0806e506ff633342fedbc789e3` |
| Descriptive reconstruction source | `026dc0aabc279bd95c09d168eb75b2d65bc9067755c923e33b7a6d87f149aae7` |

The original client/controller/protocol source SHAs are respectively
`6645eb5d2716afe08d09ae93b5f732d762a17f53002d42de3edffb067e4de29d`,
`637af3509c82e18e92f43b7419ec8d5b3dcec2300217f0ecc34885e200067092`
and `030c50f2bdcdecc373386c31ad2ae3f0b4131d7b657ab3a2e88ae8efbe2b4c8b`.
Raw records and all 127 request files remain outside Git at A's
`~/scratch/m3-mia-warm-r1/study-r1/`; the independently authenticated copy is
`~/scratch/m3-mia-warm-r1/analysis-preservation-r1/preserved/`. Local mirrors
are `/home/pmeenan/scratch/m3-mia-warm-model-r1-records/` and
`/home/pmeenan/scratch/m3-mia-warm-analysis-r1-records/`. The original model job
was `warm-cache-matrix-r1` / supervisor 1917829; the separate CPU-only job was
`warm-data-retain-r1` / supervisor 1975031. Existing immutable image, recipe,
source/asset/stat pins and prior qualification scope are retained in the
prepared and run receipts; no new full checkpoint-payload SHA is claimed.

## DeepSeek short solo/C2 screen

One buffered service screen on Spark A compares the community IQ2_XXS model
in original ds4 and a diagnostic native service. Independently rendered
input IDs agree completely: solo has 1918 tokens and the two fresh concurrent
prompts have 2355/2634. Every request produces 64 tokens with a length finish
and zero cached prompt tokens. These inputs differ from the Qwen 8K cells.

| Engine | Solo request, s | Solo output tokens/s | C2 pair, s | C2 aggregate output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Original ds4 | 4.771588 | 13.412726 | 10.283758 | 12.446812 |
| Diagnostic native | 13.830224 | 4.627546 | 16.844065 | 7.599116 |

Each cell has one observation. The native solo pays its first 6.532-second
weight activation; original weight loading occurs during startup, outside
the request clock. Both C2 arms have resident weights and pay both fresh
prefills. Native queues the requests: one finishes at 8.756791 s, the other
at 16.844065 s. Original continuous execution finishes them at
10.027234/10.283758 s, with no serial fallback or batch failure.

Both use context 16384 without speculation. Original funds four batch slots,
uses prefill chunks of 4096 and default FP8 KV/FP4 indexer; native retains
the selected ordered reduction, F16 caches and serving chunks of 2048.
This exposes a current service gap and selects native batching work, but
does not isolate its cause, compare matched cache/chunk precision, measure
pure decode, or establish generated-ID equality, quality or streaming TTFT.

The normal native service refuses this artifact's unsupported chat template.
A private host-only shim admits that exact artifact/template for literal
completions with its validated original EOS; chat returns 400 before
activation. Numerical objects and options remain unchanged. This diagnostic
does not qualify production registration without a supported renderer.

Provenance: Spark A SDK f38891fc, community artifact cd39d504, calibrated
inputs b3ba219a, targets 2a302753, original receipt 9c61f649, literal build
6cd6443c/binary 3335e766 and combined receipt 43612961. The completed original
arm is reused after native startup refusal. All six raw responses agree with
nested records and parsed bodies; services and helper commands exit zero
and are reaped. Final probes are clear at 117.207 GiB. Raw records remain
outside Git under `/home/pmeenan/scratch/m3-deepseek-concurrent-short-records/`.
