<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen NVFP4 expert output-tile screens

2026-10-04: doubling or halving the output tiles in each routed expert
block does not improve the representative four-request verify wave.
Retain production's four gate/up tiles and eight down tiles. Both private
candidates preserve every captured token, full target row and initialized
final target/drafter state; neither is selected for production.

## Factor and result

`GemvExperts` repeats its expert-pair discovery for each output block.
Changing output tiles trades that repeated work against the number of
independent blocks. Only the launch grid and matching template tile counts
change in the private `jitllm_moe.cu` variants: gate/up and down use 8/16
instead of 4/8 in the doubled candidate, and 2/4 in the halved candidate.
The routing list, `kMatched = 8` cap, per-output arithmetic, input
quantization and wave policy stay fixed. This is separate from the earlier
[smaller expert-sharing-group screen](../qwen38-four-request-waves/README.md).

Each candidate runs between two production controls, in fresh processes
on Spark A. Lower latency is better; relative changes use the mean of the
two control medians, without adding draft and verify medians.

| Candidate triplet | Arm | Draft median ms | Verify median ms |
| --- | --- | ---: | ---: |
| Doubled output tiles | Control before | 21.858 | 107.100 |
| Doubled output tiles | Candidate | 22.039 | 108.801 |
| Doubled output tiles | Control after | 22.096 | 107.285 |
| Halved output tiles | Control before | 22.227 | 107.337 |
| Halved output tiles | Candidate | 21.804 | 107.789 |
| Halved output tiles | Control after | 21.735 | 107.540 |

Doubled verify latency rises 1.501%, while its controls move +0.173%.
Halved verify latency rises 0.326%, while its controls move +0.189%.
Neither result justifies a larger context, HTTP or swap ladder. This screen
does not measure bandwidth, registers or hardware occupancy, establish a
new kernel-schedule calibration, or settle other shapes and formats.

All six arms execute 43 waves, two verify captures and 38 verify replays.
Within each triplet the four slots' generated-token hashes, every full
verify-row hash and initialized final target/drafter state hashes match
exactly. Thus the comparison pays for actual decode and graph work, rather
than launch-only timing. Six children exit zero and are reaped; all twelve
strong admission/retirement gates require at least 105 GiB available and
clear GPU, container and native model-process probes.

## Reproduce and provenance

The compiled production baseline is `5f37654`. The two private prototypes
remain outside Git; apply the tile substitutions above to that baseline
and build only `jitllm_qwen38_spec`. Use installed
`~/.local/bin/spark-job start --gpu --name NAME --timeout 600`, then
`wait NAME`, for builds and every measurement. Sources are synchronized
with `rsync -rlpc --exclude=.git --exclude=/build`, so changed bytes trigger
rebuilds. No production default or runtime implementation changed here.

Each arm uses the same four entries from
`docs/experiments/fast-swap/prompts.json`, with:

```text
--check wave --tokens 96 --context 16384 --prefill-chunk 4096
--draft 3 --draft-vocab 65536 --slots 4 --wave-lanes on
```

Shared waves retain production's effective depth two. The target artifact
is `c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`,
drafter `8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`,
checkpoint `925d7be6c14c6c9442ef83e8f05b5a3c39304f69`, tokenizer SHA-256
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
Host is `spark-c4e2`, GB10, driver 580.178.04, SDK CUDA 13.4.92. All arms
use the same SDK cuBLAS directory through `LD_LIBRARY_PATH`; no numeric
policy overrides differ between arms.

The controller authenticates each selected executable and its separately
retained compiled-source inventory. The live source tree is the candidate
in all three arms; its authentication is additional protection, not proof
that the control executable was compiled from candidate sources. Each
source inventory contains 487 implementation/build files.

| Evidence | SHA-256 |
| --- | --- |
| Production benchmark | `6ef0eb4a0426139262cb26b80d11428c775a3d3ebe6384ad9305c78384c712c6` |
| Doubled benchmark | `7c18c8be45af32e8c919d9860a0d436a3b0eb12acc23e8d56d6a3d6626707e8f` |
| Halved benchmark | `2257f1a76bf5d1deed404c889c89363f4b68e3b18da56cdf6dd3f0d6e6913195` |
| Production source inventory | `4a05996429a4d9f1988dda047782ab61897c50fe83fff61096d5266149b77235` |
| Doubled source inventory | `a282b7fd72a14ce70fec4882bc2c5e11742adbbc2d7b98fee3b7f5e578ec096c` |
| Halved source inventory | `232c4d88dabe07bf5c29f9758714d7752aebe0fc3f8c3849a78518cc136b03ed` |
| Doubled controller | `7995477d1eb0096da9eaaa6be883a3e06c85e62ff1b5102f052e9d6c9c516554` |
| Halved controller | `f01532dccbea84b6a77c527be7def3ee6b1d1656dfd4f52d469f933dbd8633ed` |
| Doubled receipt | `23cd4a4628accc8205e0db84ceb3bdaf2c1403dbcc0c19e3f0d8c417e5ffb57d` |
| Halved receipt | `1ca4e2eb27907f42fac7cb8b3de5a08dfb4dc7f0a65422222d35b150fd10d33f` |

Raw samples, complete row/state digests, preflight logs, candidate sources
and frozen binaries remain in `spark:~/scratch/qwen-expert-tiles/`, under
`screen1/` and `screen2/`. Local evidence is retained in
`~/scratch/jitllm-m3-qwen-expert-tiles-2026-10-04/`.
The baseline/candidate builds and jobs `qwen-expert-tiles-screen8-16` and
`qwen-expert-tiles-screen2-4` finish zero and are waited on. Production
source is unchanged, so the rejected diagnostics do not rerun the full
unit suite or deferred workstation/package checks.
