<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TileKernels Engram source audit

This is a source and license inventory only, with no integration. The M3.5 dependency-audit item asks for a revision, license, and sourced hash and gate formulas. DeepInfer's assessment identifies these as the Engram hash and gate kernels V4.1 needs and points to TileKernels or DeepSeek reference code for the formulas ([plan](plan.md), [assessment](deepinfer-assessment.md#v41-flash-as-deepinfer-implements-it), [row read-ahead](deepinfer-assessment.md#4-token-keyed-row-read-ahead)).

## Upstream identity and license

The official repository is [deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels). `main` resolved to immutable commit `c1cf914a82e6901c679445f872051ce44ac2cc63` (2026-10-14, merge PR #38, v2.1.0 release merge) during this audit. The links below name that exact revision. This is a source pin for the audited formulas, not a request to import or build the Python package.

The upstream [`LICENSE`](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/LICENSE) is MIT, copyright 2026 DeepSeek. Its notice and permission text must accompany any copied or adapted source. No upstream source was copied into Llmpalooza in this audit. [`pyproject.toml`](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/pyproject.toml) declares MIT at line 13; package runtime dependencies are `torch>=2.13` and `tilelang>=0.1.15` at lines 32–35, Python `>=3.12` at line 36, build requirements at lines 1–3, and development extras at lines 46–47. Those dependency licenses were not audited.

## Hash formula

The reference accepts an n-gram ID row `t[0..N-1]`, per-layer/per-position multipliers `m[l,i]`, table sizes, and flattened table offsets. It computes products in int64: `p[i] = int64(t[i]) * m[l,i]`. It initializes the running hash with `p[0]`; for each `i >= 1`, it XORs in `p[i]` and emits one index per table head `j`:

`offset[l, col] + (h % vocab_size[l, i-1, j])`, with `col = (i-1)*heads + j`.

Image-masked tokens get index `-1`. The reference defines this in [`engram_hash_ref`, lines 118–157](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/torch/engram.py#L118-L157); the CUDA implementation uses the same running XOR, int64 product and `%` expression at [`engram_hash_cuda.py`, lines 54–65](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/engram/engram_hash_cuda.py#L54-L65). The `make_offsets` helper constructs exclusive flattened prefix-sum offsets from table sizes ([lines 99–115](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/torch/engram.py#L99-L115)). This records the source expressions; it does not assert cross-language signed modulo or overflow equivalence.

## Gate formula and source-level difference

For hidden state `x`, key `k`, value `v`, channel weights `wh` and `we`, hidden size `D`, and `eps`, the PyTorch reference computes in float:

- `rx = rsqrt(mean(x^2) + eps)` and `rk = rsqrt(mean(k^2) + eps)`;
- `raw = sum((x * wh) * (k * we))`; `z = raw * rx * rk / sqrt(D)`;
- `q = sign(z) * sqrt(max(abs(z), clamp_value))`; `g = sigmoid(q)`;
- result `bf16(x + g * v)`, with masked positions copied from `x`.

See [`engram_gate_ref`, lines 188–213](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/torch/engram.py#L188-L213). The CUDA kernel's forward epilogue at [`engram_gate_fwd_cuda.py`, lines 149–166](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/engram/engram_gate_fwd_cuda.py#L149-L166) reduces the dot and RMS factors and uses `copysign(sqrt(clamp(abs(z), clamp_value, +infinity)), z)` before sigmoid. This is a source-level difference from the reference's `sign(z) * sqrt(max(abs(z), clamp_value))` at zero: for a positive clamp floor the expressions differ at `z = +0` and `z = -0`. No numerical execution was done, so this is not a measured or tested result and no exact bitwise equality is claimed.

The reference returns `x + g*v`. Upstream [issue #19](https://github.com/deepseek-ai/TileKernels/issues/19) reports that this fused result does not match the full Engram module's contribution/residual placement: the module adds its contribution, including short convolution, outside the gate. The issue was open when checked. This report records the upstream claim; it does not independently establish the full model behavior. The formulas here identify TileKernels source behavior, not a complete V4.1 model contract. M4's use must still be checked against the pinned model reference.

## Fetched-source hashes

The source bytes were fetched from `https://raw.githubusercontent.com/deepseek-ai/TileKernels/c1cf914a82e6901c679445f872051ce44ac2cc63/` into session scratch only. SHA-256 values:

| Upstream path | SHA-256 |
| --- | --- |
| `LICENSE` | `ebb4f09972aee8608be255debaf78451a68e95c290f55c240dec2ecfa16ea6be` |
| `pyproject.toml` | `71731a93b777e820ccd8fc4857dabba84241f40dd59f1fb895773ef6ca502936` |
| `tile_kernels/torch/engram.py` | `db7b779e95885f7bd965ff3cb34d20818d013bd5a50be69994ef598da4fcb48c` |
| `tile_kernels/engram/engram_hash_cuda.py` | `ee93da263405247d472a376139dbacd3726d7beca028aed0f1b5b8bbe6c82ad5` |
| `tile_kernels/engram/engram_gate_fwd_cuda.py` | `2fc32801525abba7ba0e1713cf8d0c3bb770ac1de3107b2b119820f5b8b33e26` |
| `tile_kernels/modeling/engram/engram_gate.py` | `0f895b97fc90a5823b4fe5b8d6d9ae8c05ab03025a1554ca397c02735bb59755` |

The audit did not execute upstream code, install dependencies, alter the source lock or `NOTICE`, or integrate any code.
