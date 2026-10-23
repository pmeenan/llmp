<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Generic importer proof — M3.5 U3

The owner-approved plain-NVFP4 and second-architecture MLX checkpoints use
one mapping-driven normalization/publication route. Successful publication
includes the writer's deep payload verification, `Artifact::Open`, `StoredSpec`
and `ValidateSpec` in the CLI;
it does not qualify serving, output quality or speed.

## Published NVFP4 proofs (2026-10-21)

Both ran on `spark-b` from base `58956527` plus frozen source patch SHA-256
`69abcba05f43ca7a27e2e64bb0e504735e64e95477a40461a1e2a85a1eeb3015`.
The CLI binary SHA-256 was
`6442340eb9e4a35df3826e41c84e08fe0e526d7e0f332d4879754c806c7e1dfc`.
The remote Git HEAD alone does not describe this synchronized source tree.
Later importer-core repairs address Qwen aliases, optional PLE dimensions and
schedule validation; these source configurations do not exercise those deltas.

| Checkpoint and revision | Published artifact ID | Result |
| --- | --- | --- |
| `nvidia/Gemma-4-26B-A4B-NVFP4@a19cfe00be84568a6867111c9a68c9c44fdcffe6` | `0ce111cc287b80c9fb1ff248c03115f5e5e3ec87bc0400f257260f0c7569ccba` | Deep publication passed; 362 s |
| `nvidia/Llama-3.3-70B-Instruct-NVFP4@f792b707002f2c613e0729c6b37bbcedc4efa94e` | `9cca62f1e6582324de2ebce34f38b89a8aa8890cdced957a50782e407ff4d7ee` | Deep publication passed; 857 s |

Gemma: mapping `gemma4/1`, 687 indexed resources, 3,872 groups, five shards,
17,644,769,280 payload-shard bytes; spec SHA-256
`ecd096c13c3fb89a53f052b20d3206f119b17e6fb14c9f35137bc69a95432e3e`.
Llama: mapping `llama/1`, 1,843 resources, 82 groups, eleven shards,
42,709,614,592 payload-shard bytes; spec SHA-256
`9647472757577a29b988466202b27cecad070627d25f627a2210336a1694d912`.
Artifacts are under `~/.local/share/llmp/models/m35-u3-proof/` on `spark-b`.

## Published second-architecture MLX proof (2026-10-22)

On `spark`, `TensorFold/Qwen3.8-27B-MLX-4bit` at revision
`70ae7fac63274ff2eac54152031433374cb80f2f` published artifact
`ca25eee473b02f4a13fe1d8680dea5f2aa244b556d26ee50eaa0b62b8719d3b5`.
Its declared architecture is `qwen3_5`, container `mlx`, with no experts;
this is distinct from the original Flash-Next MLX `qwen4exp` target.
Deep publication passed in 322 s. The artifact has four shards, 1,847
resources and 68 groups; its spec SHA-256 is
`d73d3cca23bd8f5ff415aeeb93c4554860da73d32c5e33417ac311ca3b904725`.
It is in the same proof-store path on `spark`.

Source base `58956527` plus frozen patch SHA-256
`f888a50630454b5a4f969812ecf20a6265920d8c930e4718fb29dc4fa40603c7`;
CLI SHA-256
`b7f9bb99ad0f26a3f1d879fae3815a42e34e8ed49f7deb77ec69a9c194b0d14f`.
Later repairs affect rejected-input diagnostics, MoE shared-gate singleton
assembly, test fixtures and spatial metadata funding; this dense proof's
published payload and stored spec do not depend on those deltas.

## U3 closeout (2026-10-23)

The explicit U3 exits are complete. The [canonical mapping closeout](canonical-mappings.md)
records final source/native/shared/affected checks and the bounded numerical
applicability assessment. Original ModelOpt/MLX parity covers all 323,607 parts;
Llama NVFP4 and the distinct qwen3_5 MLX checkpoint import, serve and pass their
separately documented bounded quality controls. Llama’s original 256-token
Fibonacci result remains failed/unresolved; its separate 512-token supplement
passes. General Gemma quality, assistant arithmetic and other U4 numerical
obligations remain open, as do separate speed, batching and maximum-context work.

Publication migrations retain complete payload proofs: [source folds](../m35-source-fold/RESULTS.md),
[Gemma sampling](gemma-sampling-refresh.md), [EXL3 sampling](exl3-sampling-refresh.md),
[DeepSeek sampling](deepseek-sampling.md) and [stored drafters](../m35-stored-drafter/RESULTS.md).

Raw job logs remain outside Git under the supervised job stores. Timing here
is publication elapsed time, not an inference benchmark.

## Llama 3 quantized-store selection

The production capability and direct launcher reject Llama 3 quantized-store
fusion. Its primitive rotation plus quantized store passed the fixed quality
bound and beat the accurate fused candidate in every screened shape. Ordinary
RoPE and YaRN still fuse; their four existing controls retain byte equality.
This selects by rotation semantics, without checkpoint or position pins.

The screen ran on `spark-b` (GB10, driver 580.178.04, pinned SDK
`aarch64-2c4eb5b2f68f3fa5`, CUDA 13.4). CUDA events measured three repeats of
256 launches or rotation/store pairs on the same stream in an exclusive job;
the table gives medians. It establishes neither model throughput nor migration
qualification. Inputs were random F32 values scaled by three, a 1,024-cell
cache, and positions `base + (t * 977) % 30000`. Rotation used base 1e6,
scale 0.125, original context 4,096, high-frequency factor 32 and low factor 1.

The quality bound was fixed before the screen: decoded-store NMSE must be at
most `(sqrt(Eq) + sqrt(1e-7))²`, where `Eq` is the independent IR reference's
CPU encode/decode error in that quantization. The reference rounds angles to
F32 before CPU trigonometry. Unwritten cache rows remain byte-exact. Both
accurate fusion and decomposition passed all fourteen cases and produced the
same stored blocks in this screen.

|encoding|head|heads|tokens|position base|accurate fused us|decomposed us|
|---|---:|---:|---:|---:|---:|---:|
|q8_0|64|2|7|100000|20.493|8.196|
|q8_0|128|8|1|131071|20.505|8.196|
|q8_0|128|8|37|131071|24.635|14.555|
|q8_0|128|8|128|131071|24.570|18.428|
|q8_0|128|8|1|1000000|22.531|8.195|
|q8_0|128|8|37|1000000|26.642|14.368|
|q8_0|128|8|128|1000000|26.596|18.429|
|q4_0|64|2|7|100000|20.494|8.196|
|q4_0|128|8|1|131071|21.391|8.196|
|q4_0|128|8|37|131071|22.531|10.259|
|q4_0|128|8|128|131071|21.638|14.800|
|q4_0|128|8|1|1000000|22.536|8.196|
|q4_0|128|8|37|1000000|24.572|10.248|
|q4_0|128|8|128|1000000|22.538|14.777|

The U3 fast-fusion candidate's Q8 NMSE was 3.635087875e-5 against a bound of
3.207489036e-5 at position base 100,000. Precise trigonometry alone passed that
small case but failed the larger-head cases at bases 131,071 and 1,000,000.
Precise power, division and trigonometry passed, at the cost shown above.
The inaccurate and slower experimental branches are absent from production.

The measured binary SHA-256 was
`f436f299ea9b6449090a162cfafef09b61d713cd1974a6152dade7e8fae13d92`;
job `m35-u3-kv-precise-shapes` passed. Source was the frozen U3 build20
integration plus the spatial schema repair and these experimental files:

- Kernel SHA-256: `2523d5f816042db55899d93a933ccb4767bfa156527cdb3ed1bdb0754a173c4a`.
- Measurement test SHA-256: `6f8ef03a0a0a0b465436e963ec35d40406bd716f0494607e52a54a1cdbf78d33`.
