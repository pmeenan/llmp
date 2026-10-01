<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Concurrent generation measurements

## TensorFold: first 8K screen

TensorFold's explicit four-request deployment increased fresh
prompt-plus-generation throughput from 23.06 to 41.05 tokens/s in plain
mode (1.78×), and from 31.62 to 40.53 with speculation (1.28×). All fourteen
concurrent outputs matched their matching solo controls exactly. These are
one observation per cell, with serialized fresh prefills; they do not
establish pure decode scaling or parity with jitLLM's different weight format.

Spark B, 2026-10-01, one CUDA rank, image `jitllm-tensorfold:71377a53`
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

## jitLLM: first 8K queue screen

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
