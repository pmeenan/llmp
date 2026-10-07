<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma4 common attention reads

This first bounded two-owner screen gives unequal cache prefixes stock's common
read width with real temporary K/V padding. Both approved profiles change from
eight positive-margin differences to all 66 teacher heads byte-exact, while its
short candidate cycles remain 6.36% / 5.43% slower than stock (26B / 31B).
The results below retain that historical correctness repair and padding cost.
The later [bounded-root transfer](../gemma4-bounded-owner-roots/README.md) removes
KV copies, preserves exact short and physically wrapped heads/state, and selects
the checked ordinary context<=4096/exactly-two-slot recipe after HTTP controls.
Internal flags still default false; wider serving policies remain separate.

The original separate attention calls use each owner's own padded read width.
Exact upstream `llama-kv-cache.cpp` instead selects one `n_kv` across active
streams. Native independent prefill heads and state are unchanged by the option.
The first two predecode heads match stock in both policies; divergence begins
only after joined unequal-width decode.

The graph validates every original owner width and exact SET_ROWS writer before
padding. A short owner's actual writer-derived K and V views each concatenate a
real graph-owned F16 zero tail. Every one of the 32 padded mask rows concatenates
an invisible tail. This gives the existing owner MMA implementation the common
width without exposing unbacked cache, moving cursors or conflating K and V.
Equal-width inputs retain the previous no-copy graph. Only two actual owners,
one row each, are admitted by the new option; wider/partial cohorts and multirow
prefill retain their existing policies. Mixed endpoint probes fund both short
owner directions before admission.

## Representative model screen

Both owners use different authenticated real token histories, independent
prefixes 256/768, three scalar warm tokens and 32 joined teacher transitions.
Context is 4096 per owner, F16 KV, two slots and two funded head rows. The ordinary
recipe uses full final FFN, checked norm/RoPE/residual chains, owner attention,
Gemma26 routed reductions and dense31 quantized GeGLU. The common-read flag is
the only baseline/candidate difference. Gemma26 prefill chunks are 1024; dense31
chunks are 256. Separate initial heads prevent independent-prefill effects from
being attributed to joined decode. The final two teacher heads are finite but
unscored; 64 targets determine the loss screen.

| Profile/policy | Initial exact heads | Later exact heads | Positive-margin differences | Relative conditional loss |
| --- | ---: | ---: | ---: | ---: |
| 26 baseline | 2/2 | 2/66 | 8 | +1.3576593% |
| 26 common reads | 2/2 | 66/66 | 0 | 0% |
| 31 baseline | 2/2 | 2/66 | 8 | −5.0824806% |
| 31 common reads | 2/2 | 66/66 | 0 | 0% |

Baseline mean TV is 0.1710428 / 0.0584530 and maximum raw logit difference
13.31708 / 13.52567 (26B / 31B). The negative 31B conditional-loss delta does
not pass its eight positive-margin differences. Common-read TV and raw
difference are zero. Both policies retain exact own
repeats. Full-head versus device choices, both initialized states, immutable
refusals, Clear/backing reuse, spill/restore and checkpoint advance/restore
controls pass. Initial heads and initialized-state hashes are byte-identical
between policies. Actual bound-plan selection is 30 / 60 owner nodes (26B / 31B).
These counts describe selected plans, rather than observed kernel launches.

## Short paid cycle

An original-stock/native/native/original-stock bookend has two paid observations
per implementation. Each pays independent prefill and 32 generated steps, plus
both final full verification heads. Three supplied warm rows are outside the
paid interval. All four arms' 64 choices, generated histories and final heads
match byte for byte. Stock retains its ordinary full sampled-logit host copies;
its backend greedy token API does not imply token-only traffic.

| Profile | Native prefill | Stock prefill | Native decode | Stock decode | Native total | Stock total | Native latency over stock |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 26 | 349.127 ms | 335.6365 ms | 791.462 ms | 736.699 ms | 1140.589 ms | 1072.3355 ms | +6.364939% |
| 31 | 1306.565 ms | 1282.830 ms | 3488.785 ms | 3265.400 ms | 4795.350 ms | 4548.230 ms | +5.433322% |

This short runner screen establishes neither end-to-end serving parity nor
sustained/context-wide performance. The costly temporary padding remains an
explicit optimization lead; removing copies must preserve the proven common
geometry and actual root bounds.

## Operator and funding controls

The focused host tests pass 47 cases. The composed GPU control covers H16/H32,
D256/GQA2 and D512/GQA8, actual 512/1024 roots padded to 1024, both original packed
MMA and independent-root owners. Results are byte-exact across implementations,
eager/capture/replay and fresh short/long inputs; FP64 normalized error remains
below 2.3e-7. Poisoned temporary tails are overwritten, and source half bytes
remain unchanged. Actual columns/grid/mask-prepass/scratch match:

| D / heads | Columns | Blocks | Mask prepass | Scratch bytes |
| --- | ---: | ---: | ---: | ---: |
| 256 / 16 or 32 | 4 | 48 | yes | 399616 |
| 512 / 16 | 1 | 64 | yes | 1057024 |
| 512 / 32 | 1 | 96 | yes | 1585408 |

The original failed host lookup, incorrect D512 expected-column assertion and
stock observer's incorrect 32-row mask expectation remain in external receipts.
The latter is factual harness metadata: exact stock supplies one query-mask row,
while native sources pad to 32. No stock arithmetic, numerical gate or tolerance
changed to recover those controls.

## Provenance and limits

Measured on Spark A / GB10, driver 580.178.04, pinned aarch64 c09 SDK, with the
approved prepared artifacts in [model support](../../model-support.md).
The stock image is the unmodified official v0.6.0 / d81235049384534c167caea52b85a694f6103d14,
`ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db`.
The retained four-history input is 131072 bytes, SHA-256
`90bf82fd26b7040c513e7370eca2cd283ac8fd15ce89a2e8d05d6f7e670b5ba2`;
owner inputs are slices of histories 0 and 1 with their supplied teacher suffixes.
Actual source/binary/receipt identities and raw job outputs remain outside Git
under `gemma4-common-width-20261007` on A. The final model26 retry has 12 successful
steps after the preserved observer-only failure; native own controls were reused
from that authenticated first job, without a redundant native rebuild. The 31B
follow-on completes 17 steps with the same authenticated binaries. Final native
probe SHA-256 is
`e216ea143f533f4e9c5611a8c85db2999b4db83ef75b3748a3626f5612c5ac13`;
model binding receipts are
`92c09d6dc72889795e49498deb3509cf8f54e93a204c2a76e403fe16c2ba069f`
and `f280aa1d7c4fca299a8aee12905bee64aa3558db147ff516f1c8c67d86fa29f1`.

At task entry, TensorFold HEAD
[041d14a94e951834470fd514ed33e65b8be1059a](https://github.com/ashhart/TensorFold/blob/041d14a94e951834470fd514ed33e65b8be1059a/README.md)
has no matching documented qualified CUDA/Gemma4/GGUF recipe. Its native 1.0.0
README qualifies GB10 Nemotron3.5 MLX affine4 and lists Gemma4 under further
qualification. No inapplicable reference build was run.

Broader context/ring, partial/wider cohorts, compatible Gemma4 multirow prefill,
assistant/speculation, sustained memory/swap and corpus qualification remain
open. Public support limits and serving defaults are not enlarged by this
internal first screen.
