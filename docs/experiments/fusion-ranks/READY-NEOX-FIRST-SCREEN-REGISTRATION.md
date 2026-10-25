<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ready NEOX norm/RoPE first-screen registration

Prospective registration, 2026-10-25. No NEOX body timing has run. Screen one
F32 width-128, two-row NEOX RMSNorm/Mul/RoPE recipe against its own normally
planned decomposition on `spark` (GB10). The complete accepted three-member
`interleaved_norm_rope` group retains its competing-rewrite exclusions; the
alternative retains the RMSNorm/Mul child fusion. Frequency factors are absent
in this timing cell. Optional-frequency correctness is qualified separately.

## Frozen implementation and correctness

| Fact | Value |
| --- | --- |
| Development Git base | `f0ba0a0fc0d48cdb73cb484c1b3c2900f6c8064e` |
| Source checkpoint | `u6-ready-neox-first-screen-r46-a`, 115 paths |
| Source manifest SHA-256 | `0feb99c513af3ebda2e6d6457a609f6b39974e85e60c8ab7caa1585ebd7740d8` |
| Native binary SHA-256 | `f9ba576ffc3daee0b659f0536f0e4c9412feddc77209f1a800238cf7ed5e083c` |
| Host / GPU / driver | `spark` (`spark-c4e2`) / NVIDIA GB10 / `580.178.04` |
| Pinned SDK | `aarch64-2c4eb5b2f68f3fa5` |
| Backend implementation identity | `b8d52eba05343dd01adc3c59bf1504920d94e1739d363f66f1a511365827180e` |
| Complete recipe-key digest | `4f50d66c5799aa6f25ec0f5810e1cbfc5de553ab74b4faecc24a5c0bc2c3d751` |
| Ready preparation domain | `4924c9434424f6bdf10171e29f90fb2b5b27ff9fb34a123368f6ff0142adc710` |
| Protocol SHA-256 | `e61f221d8f4677d15ed8a868b4d693bd7468d63012493e33cb394e52de080fb4` |

The r45 core checkpoint passes all 3,959 shared cases across both Sparks and
the two real ModelOpt/GGUF saved-recurrence cases without failures or skips.
The later benchmark-only timing change passes independent source/challenge
review and eight native qualification steps. It validates two changed original
input/position frames against the original-IR CPU reference, including a
position-only effect control. Both arms preserve gamma and I32 positions.
Maximum absolute errors are `4.287809133529663e-6` and
`4.302710294723511e-6`, within the fixed `1e-5 * (1 + abs(expected))` bound.
The actual recordings contain one fused node versus two decomposition nodes,
with exactly one complete body per capture. No timing invocation, CUDA timing
events or samples were used for qualification.

## Screen and decision

After this registration is committed, run
`build/spark-native/benchmarks/llmp_fusion_ranks --ready-norm-rope-time 128 2`
in one installed-supervisor exclusive job, bounded to 600 seconds. Expected
runtime is seconds. Verify the frozen source and binary before and after;
retain actual successful completion after teardown and check the host is free.

Repeat two-frame reference validation before timing. Before every warmup,
sample and bookend arm, poison the activation arena, restore the original F32
Features and I32 positions, then finish `PrepareFrame` outside the events.
One event pair encloses exactly one complete captured local-body replay.
Check every CPU output, immutable gamma, positions and capture inventory after
each arm, outside the events. There is no frequency operand in this cell.

Use four fixed incumbent/alternative warmup pairs, one alternative bookend,
four alternating warmup pairs, nine alternating AB/BA sample pairs beginning
AB, then one alternative bookend. Preserve the 36 event records in order;
compute medians from copies of the nine measured samples per arm. Refuse the
run if the larger bookend exceeds 1.1 times the smaller, with no automatic
retry or threshold change. Classify medians with the existing overflow-safe
10% `ClassifyRecipeLatency` band.

Export both complete arm identities and the complete typed key, protocol
text/digest, all events and their exact compact-JSON SHA-256 receipt, medians
and classification. Keep candidate provenance `UNREVIEWED` until independent
source, numerical, event-order, drift, schema and retirement review. Review
this first result before expanding measurements or installing data. This is
local-body evidence, with no model applicability, adoption or U6 completion
claim. Historical RMS rows keep their original identities.

The exact protocol text is:

```text
llmp-ready-neox-norm-rope-single-body-timing-v1;prep=4924c9434424f6bdf10171e29f90fb2b5b27ff9fb34a123368f6ff0142adc710;frames=2-original-ir-validated-before-events;frequency=none;input=poison-arena-then-restage-original-f32-features-and-i32-positions-before-each-replay;prefix=prepare-once-finish-before-events;body=captured-complete-local-region;body-replays=1;fixed-warmup-pairs=4:incumbent-then-alternative;bookends=alternative-before-and-after;alternating-warmup-pairs=4:start-incumbent-then-alternate;alternating-sample-pairs=9:start-incumbent-then-alternate;events=one-start-stop-per-single-body;gamma-positions-and-cpu-output-check=after-every-arm;bookend-drift=10pct;retry=none
```
