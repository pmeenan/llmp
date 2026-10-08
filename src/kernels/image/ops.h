// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Llmpalooza's own CUDA kernels for the Qwen-Image-2.1 pipeline's BF16
// operations (M3; D-053's "our own, on measured need": GGML computes these
// in F32 between its operations, where the reference, diffusers in BF16,
// rounds every intermediate to BF16). Each kernel computes in F32 and
// rounds to BF16 exactly where the pinned diffusers 8b3c707e and
// transformers 5.17 code materializes a BF16 tensor, so that a fused kernel
// still produces the unfused reference's values: the comment of each names
// the PyTorch expression it reproduces. Tensors are row-major and packed
// unless a stride is given; BF16 values are raw 16-bit patterns.
//
// Launches only queue work on `stream` (a cudaStream_t); completion is the
// caller's. Every function refuses (with nothing queued) sizes its kernels
// cannot index, and reports a launch error. No kernel reads or writes
// outside the extents it is given.

#ifndef LLMP_KERNELS_IMAGE_OPS_H_
#define LLMP_KERNELS_IMAGE_OPS_H_

#include <cstdint>
#include <expected>
#include <string>

namespace llmp::kernels::image {

using Bf16 = std::uint16_t;
using F16 = std::uint16_t;
using Status = std::expected<void, std::string>;
using Stream = void*;

// ---- shared

// out = a + b (BF16 + BF16 -> BF16), n elements.
Status Add(const Bf16* a, const Bf16* b, Bf16* out, std::int64_t n, Stream stream);
// out[r, :] = table[ids[r], :] for `rows` rows of `width`; ids are checked
// against `table_rows` on the device (an out-of-range id writes zeros and
// sets *bad, a device flag the caller reads after completion).
Status EmbedRows(const Bf16* table, std::int64_t table_rows, const std::int32_t* ids,
                 std::int64_t rows, std::int64_t width, Bf16* out, std::int32_t* bad,
                 Stream stream);
// out = silu(x) in BF16 (F.silu on a BF16 tensor), n elements.
Status Silu(const Bf16* x, Bf16* out, std::int64_t n, Stream stream);
// out = gelu_tanh(x) in BF16 (nn.GELU(approximate="tanh")), n elements.
Status GeluTanh(const Bf16* x, Bf16* out, std::int64_t n, Stream stream);
// out = silu(gate) * up: bf16(bf16(silu(gate)) * up) (act_fn(gate) * up).
// gate and up are rows of `width` at row strides gate_stride and up_stride.
Status SwiGlu(const Bf16* gate, std::int64_t gate_stride, const Bf16* up, std::int64_t up_stride,
              Bf16* out, std::int64_t rows, std::int64_t width, Stream stream);
// Conversions: BF16 to F16 (exact inside F16's range), F32 to BF16
// (round to nearest even), BF16 to F32.
Status Bf16ToF16(const Bf16* x, F16* out, std::int64_t n, Stream stream);
Status F32ToBf16(const float* x, Bf16* out, std::int64_t n, Stream stream);
Status Bf16ToF32(const Bf16* x, float* out, std::int64_t n, Stream stream);

// ---- the text encoder (Qwen3-VL's text path, transformers 5.17)

// Qwen3VLTextRMSNorm: out = w * bf16(x * rsqrt(mean(x^2) + eps)), in F32
// then BF16, over rows of `width`.
Status RmsNorm(const Bf16* x, const Bf16* w, Bf16* out, std::int64_t rows, std::int64_t width,
               float eps, Stream stream);
// Per head: q_norm or k_norm (as RmsNorm, over head_dim 128), then
// apply_rotary_pos_emb (rotate_half, "NEOX"): out = bf16(bf16(n * cos) +
// bf16(rotate_half(n) * sin)), cos and sin BF16 rows of 128 per position.
// In place over rows of heads x 128 at row stride `stride` (elements).
Status HeadNormRopeNeox(Bf16* x, std::int64_t stride, std::int64_t rows, std::int64_t heads,
                        const Bf16* w, const Bf16* cos, const Bf16* sin, float eps, Stream stream);
// Scaled dot-product attention over a short sequence, F32 scores and
// softmax, BF16 out: q [rows, q_heads, 128], k and v [rows, kv_heads, 128]
// (query head h reads KV head h / (q_heads / kv_heads)), each at its row
// stride; causal: query i sees keys 0..i. Rows at most 1,024.
Status SmallAttention(const Bf16* q, std::int64_t q_stride, const Bf16* k, std::int64_t k_stride,
                      const Bf16* v, std::int64_t v_stride, Bf16* out, std::int64_t out_stride,
                      std::int64_t rows, std::int64_t q_heads, std::int64_t kv_heads, bool causal,
                      float scale, Stream stream);
// The same over F32 q and F16 k and v (the denoiser's text rows at its first
// step, whose q and k the rotary kernels write in those types).
Status SmallAttentionF32F16(const float* q, std::int64_t q_stride, const F16* k,
                            std::int64_t k_stride, const F16* v, std::int64_t v_stride, Bf16* out,
                            std::int64_t out_stride, std::int64_t rows, std::int64_t q_heads,
                            std::int64_t kv_heads, bool causal, float scale, Stream stream);

// ---- the denoiser (QwenImage21Transformer2DModel)

// QwenImage21ZeroCenterRMSNorm: bf16(x * rsqrt(mean(x^2) + eps) * (w + 1)),
// all in F32.
Status ZeroCenterRmsNorm(const Bf16* x, const Bf16* w, Bf16* out, std::int64_t rows,
                         std::int64_t width, float eps, Stream stream);
// LayerNorm (no affine) then the causal-condition modulation:
// out = bf16(bf16(layer_norm(x)) * bf16(1 + scale)), where row r takes
// scale from `mod` row 0 (the sampled timestep) if r >= first_target, else
// row 1 (t = 0): _select_modulation_rows. `mod` rows are `mod_stride`
// apart; scale is the `width` elements at mod + offset.
Status LayerNormModulate(const Bf16* x, Bf16* out, std::int64_t rows, std::int64_t width, float eps,
                         const Bf16* mod, std::int64_t mod_stride, std::int64_t first_target,
                         Stream stream);
// Attention's q or k: per head, diffusers' RMSNorm over 128
// (bf16(bf16(x * rsqrt(mean(x^2) + eps)) * w)), then apply_rotary_emb_qwen
// with complex frequencies (pairs (2i, 2i+1) times (cos_i + j sin_i) in F32,
// rounded to BF16). freqs: F32 [rows, 64, 2] (cos, sin). Reads x
// [rows, heads, 128] at row stride x_stride; writes F32 (q) or F16 (k)
// values of the BF16 result, packed [rows, heads, 128].
Status HeadNormRopeComplexF32(const Bf16* x, std::int64_t x_stride, std::int64_t rows,
                              std::int64_t heads, const Bf16* w, const float* freqs, float eps,
                              float* out, Stream stream);
Status HeadNormRopeComplexF16(const Bf16* x, std::int64_t x_stride, std::int64_t rows,
                              std::int64_t heads, const Bf16* w, const float* freqs, float eps,
                              F16* out, Stream stream);
// The same, writing the BF16 result in place (for FlashAttention below).
Status HeadNormRopeComplex(Bf16* x, std::int64_t x_stride, std::int64_t rows, std::int64_t heads,
                           const Bf16* w, const float* freqs, float eps, Stream stream);
// Scaled dot-product attention without a mask, head dimension 128, BF16 in
// and out, as PyTorch's flash kernel computes it (FlashAttention-2's
// forward: F32 scores and online softmax, probabilities rounded to BF16 for
// the value product, F32 accumulation, the output rounded once): q
// [q_rows, heads, 128], k and v [kv_rows, kv_heads, 128] (query head h
// reads KV head h / (heads / kv_heads)), each at its row stride (elements,
// a multiple of 8, as are the base addresses in 16-byte units), out
// [q_rows, heads, 128] at out_stride. Tensor cores (mma.sync m16n8k16).
Status FlashAttention(const Bf16* q, std::int64_t q_stride, const Bf16* k, std::int64_t k_stride,
                      const Bf16* v, std::int64_t v_stride, Bf16* out, std::int64_t out_stride,
                      std::int64_t q_rows, std::int64_t kv_rows, std::int64_t heads,
                      std::int64_t kv_heads, float scale, Stream stream);
// The same with the first `prefix` keys and values read from k_prefix and
// v_prefix (the prefix K/V cache, at k's and v's strides) and the rest,
// kv_rows - prefix of them, from k and v: the same keys in the same order
// as one range holding both, so the same values bit for bit. With q_norm,
// q is the raw projection, and each query row is first normed and rotated
// as HeadNormRopeComplex does it (its weight, the row's F32 frequencies,
// rows 128 floats apart from q's first row, eps), bit for bit, in
// registers: the rotated queries are never written.
struct QueryNorm {
  const Bf16* weight = nullptr;  // [128]
  const float* freqs = nullptr;  // [q_rows, 64, 2]
  float eps = 0.0f;
};
Status FlashAttentionPrefixed(const Bf16* q, std::int64_t q_stride, const Bf16* k_prefix,
                              const Bf16* v_prefix, std::int64_t prefix, const Bf16* k,
                              std::int64_t k_stride, const Bf16* v, std::int64_t v_stride,
                              Bf16* out, std::int64_t out_stride, std::int64_t q_rows,
                              std::int64_t kv_rows, std::int64_t heads, std::int64_t kv_heads,
                              float scale, Stream stream, const QueryNorm* q_norm = nullptr);
// Strided BF16 rows to packed F16 rows (v for the attention kernel).
Status Bf16RowsToF16(const Bf16* x, std::int64_t x_stride, F16* out, std::int64_t rows,
                     std::int64_t width, Stream stream);
// The gated residual: x = bf16(x + bf16(bf16(tanh(gate)) * y)), gate from
// `mod` rows as in LayerNormModulate, y at row stride y_stride.
Status GatedResidual(Bf16* x, const Bf16* y, std::int64_t y_stride, std::int64_t rows,
                     std::int64_t width, const Bf16* mod, std::int64_t mod_stride,
                     std::int64_t first_target, Stream stream);
// GatedResidual, then LayerNormModulate of the new x into `out` (with
// `scale_mod` rows `scale_stride` apart), in one pass: the same values, bit
// for bit, as the two calls (both select their modulation row by
// first_target).
Status GatedResidualNorm(Bf16* x, const Bf16* y, std::int64_t y_stride, std::int64_t rows,
                         std::int64_t width, const Bf16* gate_mod, std::int64_t gate_stride,
                         std::int64_t first_target, Bf16* out, float eps, const Bf16* scale_mod,
                         std::int64_t scale_stride, Stream stream);
// FlowMatchEulerDiscreteScheduler.step: out = bf16(f32(sample) +
// bf16(dt * noise)) when dt_bf16 is false, else with dt rounded to BF16
// first (bf16(f32(sample) + bf16(bf16(dt) * noise))); n elements. `out`
// may be `sample`.
Status EulerStep(const Bf16* sample, const Bf16* noise, Bf16* out, float dt, bool dt_bf16,
                 std::int64_t n, Stream stream);
// The same with dt read from device memory when the kernel runs (a step
// that a graph replays): *dt is used as it is (the caller rounds it to
// BF16 first, EulerStepDt).
Status EulerStepAt(const Bf16* sample, const Bf16* noise, Bf16* out, const float* dt,
                   std::int64_t n, Stream stream);
// dt as EulerStep uses it: rounded to BF16 when dt_bf16.
float EulerStepDt(float dt, bool dt_bf16);

// ---- the VAE decoder (AutoencoderKLQwenImage21), channels-first [C, H, W]

// out[c, r] = x[r, c]: a [rows, cols] BF16 matrix transposed (the packed
// latents [tokens, channels] to channels-first).
Status Transpose(const Bf16* x, Bf16* out, std::int64_t rows, std::int64_t cols, Stream stream);

// out[c, p] = bf16(bf16(z[c, p] * std[c]) + mean[c]), z [channels, pixels].
Status ScaleShiftChannels(const Bf16* z, const Bf16* std, const Bf16* mean, Bf16* out,
                          std::int64_t channels, std::int64_t pixels, Stream stream);
// QwenImage21RMS_norm, then optionally SiLU: n = bf16(x / max(||x||_c,
// 1e-12)) over channels (F.normalize in F32), y = bf16(bf16(n * sqrt(C)) *
// gamma[c]), and silu(y) in BF16 if `silu`.
Status ChannelRmsNorm(const Bf16* x, const Bf16* gamma, Bf16* out, std::int64_t channels,
                      std::int64_t pixels, bool silu, Stream stream);
// im2col for a 3x3, stride 1, zero-padded convolution: col[(c*9 + ky*3 +
// kx), j] = x[c, y + ky - 1, x + kx - 1] for output pixels j in
// [pixel0, pixel0 + count) of an [channels, height, width] input; col is
// [channels*9, count], packed.
Status Im2Col3x3(const Bf16* x, std::int64_t channels, std::int64_t height, std::int64_t width,
                 std::int64_t pixel0, std::int64_t count, Bf16* col, Stream stream);
// A 3x3, stride 1, zero-padded convolution plus bias as one implicit GEMM
// (conv.cu): out[co, y, x] = bf16(bias[co] + sum over ky, kx, ci of
// w[co, ky, kx, ci] x[ci, y + ky - 1, x + kx - 1]) in F32, channels-first
// x [in_channels, height, width] and out [out_channels, height, width];
// the weights in KRSC order (Conv3x3WeightsKrsc); out does not overlap x.
// Input channels a multiple of 16, an even width (Qwen-Image's VAE widths
// are at every size it takes, a multiple of 32), tensors 16-byte aligned.
Status Conv3x3Implicit(const Bf16* x, std::int64_t in_channels, std::int64_t height,
                       std::int64_t width, const Bf16* w_krsc, const Bf16* bias, Bf16* out,
                       std::int64_t out_channels, Stream stream);
// PyTorch's F32 [co, ci, 3, 3] convolution weights, rounded to BF16, in
// KRSC order [co, 3, 3, ci].
Status Conv3x3WeightsKrsc(const float* w, Bf16* out, std::int64_t out_channels,
                          std::int64_t in_channels, Stream stream);
// out[c, p] = bias[c] for every pixel (the convolution's C before beta = 1).
Status FillBias(const Bf16* bias, Bf16* out, std::int64_t channels, std::int64_t pixels,
                Stream stream);
// Nearest 2x upsampling (nearest-exact at scale 2): [C, H, W] -> [C, 2H, 2W].
Status Upsample2x(const Bf16* x, Bf16* out, std::int64_t channels, std::int64_t height,
                  std::int64_t width, Stream stream);
// x += DupUp3D(shortcut) at the last kept frame (first_chunk):
// shortcut [in_channels, H, W] -> [out_channels, 2H, 2W], each output
// channel o, sub-pixel (b, d) reading input channel
// (((o * ft + ft - 1) * 2 + b) * 2 + d) / repeats, repeats =
// out_channels * ft * 4 / in_channels; x = bf16(x + value).
Status AddDupUp(Bf16* x, const Bf16* shortcut, std::int64_t in_channels, std::int64_t out_channels,
                std::int64_t factor_t, std::int64_t height, std::int64_t width, Stream stream);
// Row softmax of F32 scores times `scale`, into BF16 probabilities
// (rows x cols), for the mid block's single-head attention.
Status SoftmaxRowsToBf16(const float* scores, Bf16* probs, std::int64_t rows, std::int64_t cols,
                         float scale, Stream stream);

}  // namespace llmp::kernels::image

#endif  // LLMP_KERNELS_IMAGE_OPS_H_
