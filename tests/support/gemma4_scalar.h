// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Independent analytic scalar oracle for structured, constant-per-matrix-row
// weights at the ACTUAL 26B widths. No graph/node/planner inspection. Encoded
// weight quantization and CUDA input rounding are judged by a quality bound;
// solo/batch and own-repeat controls separately compare actual output bytes.
#ifndef LLMP_TESTS_SUPPORT_GEMMA4_SCALAR_H_
#define LLMP_TESTS_SUPPORT_GEMMA4_SCALAR_H_
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ggml.h"
#include "model/gemma4.h"

namespace llmp::test_support::gemma4_scalar {
inline double Coefficient(std::string_view name, std::uint32_t row, std::uint32_t expert = 0) {
  const auto centered = [](std::uint32_t value, std::uint32_t modulus) {
    return static_cast<double>(value % modulus) - static_cast<double>(modulus / 2);
  };
  if (name == "token_embd.weight") return centered(row, 13) / 4096;
  if (name.contains("ffn_gate_inp.weight")) return (static_cast<double>(row) + 1) / 8192;
  if (name.contains("attn_q.weight")) return centered(row, 7) / 1024;
  if (name.contains("attn_k.weight")) return centered(row + 1, 11) / 1024;
  if (name.contains("attn_v.weight")) return centered(row + 2, 5) / 1024;
  if (name.contains("attn_output.weight")) return centered(row, 9) / 2048;
  if (name.contains("ffn_gate_up_exps.weight")) {
    const bool up = row >= 704;
    return centered((row % 704) + expert + (up ? 2U : 0U), 7) / 512;
  }
  if (name.contains("ffn_down_exps.weight")) return centered(row + expert, 11) / 2048;
  if (name.contains("ffn_gate.weight")) return centered(row, 7) / 512;
  if (name.contains("ffn_up.weight")) return centered(row + 2, 7) / 512;
  if (name.contains("ffn_down.weight")) return centered(row, 9) / 2048;
  return 0;
}
inline double Vector(std::string_view name, std::uint32_t i) {
  if (name.contains("layer_output_scale")) return 1.125;
  if (name.contains("ffn_down_exps.scale")) return 0.5 + static_cast<double>(i) / 128;
  std::uint32_t tag = 0;
  for (const char ch : name) tag += static_cast<unsigned char>(ch);
  return 1.0 + (static_cast<double>((i + tag) % 7) - 3.0) / 32;
}
inline std::vector<double> Norm(const std::vector<double>& in, std::string_view weight = {}) {
  double sum = 0;
  for (const auto x : in) sum += x * x;
  const double scale = 1.0 / std::sqrt(sum / static_cast<double>(in.size()) + 1e-6);
  std::vector<double> out(in.size());
  for (std::size_t i = 0; i < in.size(); ++i)
    out[i] = in[i] * scale * (weight.empty() ? 1.0 : Vector(weight, static_cast<std::uint32_t>(i)));
  return out;
}
using CoefficientOracle = std::function<double(std::string_view, std::uint32_t, std::uint32_t)>;
inline std::vector<double> Product(const std::vector<double>& in, std::string_view role,
                                   std::uint32_t outputs, std::uint32_t expert = 0,
                                   const CoefficientOracle& coefficient = Coefficient) {
  const double sum = std::accumulate(in.begin(), in.end(), 0.0);
  std::vector<double> out(outputs);
  for (std::uint32_t i = 0; i < outputs; ++i) out[i] = sum * coefficient(role, i, expert);
  return out;
}
inline double Gelu(double x) {
  return 0.5 * x * (1 + std::tanh(std::sqrt(2.0 / std::acos(-1.0)) * (x + 0.044715 * x * x * x)));
}
inline std::vector<double> Glu(const std::vector<double>& gate, const std::vector<double>& up) {
  std::vector<double> out(gate.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = Gelu(gate[i]) * up[i];
  return out;
}
inline float Initial(std::int64_t position, std::uint32_t dim, std::uint32_t head, bool value,
                     bool positive_values = true) {
  if (position < 0) return 0;
  const auto code =
      (static_cast<std::uint64_t>(position) * 3 + dim * 5 + head * 7 + (value ? 17 : 3)) % 41;
  return static_cast<float>(static_cast<double>(code) - 20.0) / 64 +
         (value && positive_values ? 0.25f : 0.0f);
}
inline std::int64_t AbsoluteCell(std::uint32_t past, std::uint32_t cell, std::uint32_t capacity,
                                 bool local) {
  if (!local) return cell < past ? static_cast<std::int64_t>(cell) : -1;
  auto absolute = static_cast<std::int64_t>(past) - 1;
  absolute -= (absolute - cell + capacity) % capacity;
  return absolute;
}
inline std::vector<double> Rotate(std::vector<double> x, std::uint32_t position, bool local,
                                  bool independently_rounded_phase = false) {
  const auto d = x.size();
  const double base = local ? 10000.0 : 1000000.0;
  for (std::size_t i = 0; i < d / 2; ++i) {
    const double factor = !local && i >= 64 ? 1e30 : 1;
    double angle = static_cast<double>(position) /
                   (std::pow(base, 2.0 * static_cast<double>(i) / static_cast<double>(d)) * factor);
    if (independently_rounded_phase) {
      const float theta_scale = std::pow(static_cast<float>(base), -2.0f / static_cast<float>(d));
      angle = static_cast<float>(position) * std::pow(theta_scale, static_cast<float>(i)) /
              static_cast<float>(factor);
    }
    const double a = x[i], b = x[i + d / 2];
    x[i] = a * std::cos(angle) - b * std::sin(angle);
    x[i + d / 2] = a * std::sin(angle) + b * std::cos(angle);
  }
  return x;
}
inline double Half(double x) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(static_cast<float>(x))); }
struct LayerResult {
  std::vector<double> output, keys, values, attention;
};
inline LayerResult Layer(const model::Gemma4Profile& p, std::uint32_t il, std::uint32_t past,
                         const std::vector<double>& input, std::uint32_t rows,
                         std::uint32_t capacity, bool positive_values = true,
                         const CoefficientOracle& coefficient = Coefficient,
                         bool independently_rounded_phase = false,
                         std::span<const double> observed_attention = {},
                         bool rounded_projection_input = false) {
  const auto prefix = std::string("blk.") + std::to_string(il) + ".";
  const bool local = p.local(il);
  const auto d = p.head_dim(il), kvh = p.kv_heads(il), ratio = p.heads / kvh;
  LayerResult result;
  std::vector<std::vector<double>> queries;
  for (std::uint32_t r = 0; r < rows; ++r) {
    const std::vector<double> x(
        input.begin() + static_cast<std::ptrdiff_t>(std::size_t{r} * p.width),
        input.begin() + static_cast<std::ptrdiff_t>(std::size_t{r + 1} * p.width));
    const auto norm = Norm(x, prefix + "attn_norm.weight");
    const auto qraw = Product(norm, prefix + "attn_q.weight", d * p.heads, 0, coefficient);
    const auto kraw = Product(norm, prefix + "attn_k.weight", d * kvh, 0, coefficient);
    const auto vraw =
        local ? Product(norm, prefix + "attn_v.weight", d * kvh, 0, coefficient) : kraw;
    for (std::uint32_t h = 0; h < p.heads; ++h) {
      auto q = Norm(std::vector<double>(qraw.begin() + h * d, qraw.begin() + (h + 1) * d),
                    prefix + "attn_q_norm.weight");
      q = Rotate(std::move(q), past + r, local, independently_rounded_phase);
      // Pinned group2/group8 MMA converts Q to half; local solo vector does not.
      if (!local || rows > 1)
        for (auto& value : q) value = Half(value);
      queries.push_back(std::move(q));
    }
    for (std::uint32_t h = 0; h < kvh; ++h) {
      auto k = Norm(std::vector<double>(kraw.begin() + h * d, kraw.begin() + (h + 1) * d),
                    prefix + "attn_k_norm.weight");
      k = Rotate(std::move(k), past + r, local, independently_rounded_phase);
      auto v = Norm(std::vector<double>(vraw.begin() + h * d, vraw.begin() + (h + 1) * d));
      for (auto& value : k) value = Half(value);
      for (auto& value : v) value = Half(value);
      result.keys.insert(result.keys.end(), k.begin(), k.end());
      result.values.insert(result.values.end(), v.begin(), v.end());
    }
  }
  for (std::uint32_t r = 0; r < rows; ++r) {
    std::vector<double> attention(std::size_t{d} * p.heads);
    for (std::uint32_t h = 0; h < p.heads; ++h) {
      const auto kh = h / ratio;
      const auto begin = local && past + r + 1 > p.window ? past + r + 1 - p.window : 0;
      std::vector<double> scores(past + r + 1 - begin);
      const auto key = [&](std::uint32_t pos, std::uint32_t i, bool value) {
        if (pos >= past)
          return (value ? result.values
                        : result.keys)[std::size_t{pos - past} * d * kvh + std::size_t{kh} * d + i];
        return static_cast<double>(Initial(pos, i, kh, value, positive_values));
      };
      for (std::uint32_t pos = begin; pos <= past + r; ++pos) {
        double score = 0;
        for (std::uint32_t i = 0; i < d; ++i)
          score += queries[std::size_t{r} * p.heads + h][i] * key(pos, i, false);
        scores[pos - begin] = score;
      }
      const auto max = *std::max_element(scores.begin(), scores.end());
      double denominator = 0;
      for (auto& score : scores) {
        score = std::exp(score - max);
        denominator += score;
      }
      for (std::uint32_t pos = begin; pos <= past + r; ++pos) {
        for (std::uint32_t i = 0; i < d; ++i)
          attention[h * d + i] += scores[pos - begin] / denominator * key(pos, i, true);
      }
    }
    result.attention.insert(result.attention.end(), attention.begin(), attention.end());
    if (!observed_attention.empty()) {
      std::copy_n(
          observed_attention.begin() + static_cast<std::ptrdiff_t>(std::size_t{r} * d * p.heads),
          attention.size(), attention.begin());
    }
    if (rounded_projection_input) {
      // Independent numerical diagnostic for pinned MMVQ Q8_1 preparation:
      // quantize.cu rounds each32-cell block with an F16 stored scale.
      for (std::size_t block = 0; block < attention.size(); block += 32) {
        float maximum = 0;
        for (std::size_t j = 0; j < 32; ++j)
          maximum = std::max(maximum, std::abs(static_cast<float>(attention[block + j])));
        const float scale = maximum / 127;
        const double stored_scale = Half(scale);
        for (std::size_t j = 0; j < 32; ++j)
          attention[block + j] =
              maximum == 0
                  ? 0
                  : std::round(static_cast<float>(attention[block + j]) / scale) * stored_scale;
      }
    }
    auto projected =
        Norm(Product(attention, prefix + "attn_output.weight", p.width, 0, coefficient),
             prefix + "post_attention_norm.weight");
    std::vector<double> residual(p.width);
    for (std::uint32_t i = 0; i < p.width; ++i)
      residual[i] = projected[i] + input[std::size_t{r} * p.width + i];
    const auto shared_in = Norm(residual, prefix + "ffn_norm.weight");
    auto shared = Product(Glu(Product(shared_in, prefix + "ffn_gate.weight", p.ffn, 0, coefficient),
                              Product(shared_in, prefix + "ffn_up.weight", p.ffn, 0, coefficient)),
                          prefix + "ffn_down.weight", p.width, 0, coefficient);
    if (p.experts != 0) {
      shared = Norm(shared, prefix + "post_ffw_norm_1.weight");
      auto router_in = Norm(residual);
      for (std::uint32_t i = 0; i < p.width; ++i)
        router_in[i] *=
            Vector(prefix + "ffn_gate_inp.scale", i) / std::sqrt(static_cast<double>(p.width));
      auto probabilities =
          Product(router_in, prefix + "ffn_gate_inp.weight", p.experts, 0, coefficient);
      const auto max = *std::max_element(probabilities.begin(), probabilities.end());
      double denom = 0;
      for (auto& x : probabilities) {
        x = std::exp(x - max);
        denom += x;
      }
      for (auto& x : probabilities) x /= denom;
      std::vector<std::uint32_t> ids(p.experts);
      std::iota(ids.begin(), ids.end(), 0);
      std::stable_sort(ids.begin(), ids.end(),
                       [&](auto a, auto b) { return probabilities[a] > probabilities[b]; });
      double selected_sum = 0;
      for (std::uint32_t i = 0; i < p.experts_used; ++i) selected_sum += probabilities[ids[i]];
      selected_sum = std::max(selected_sum, 6.103515625e-5);
      const auto expert_in = Norm(residual, prefix + "pre_ffw_norm_2.weight");
      std::vector<double> routed(p.width);
      for (std::uint32_t i = 0; i < p.experts_used; ++i) {
        const auto e = ids[i];
        auto pair = Product(expert_in, prefix + "ffn_gate_up_exps.weight", p.expert_ffn * 2, e,
                            coefficient);
        auto activation = Glu(std::vector<double>(pair.begin(), pair.begin() + p.expert_ffn),
                              std::vector<double>(pair.begin() + p.expert_ffn, pair.end()));
        const auto down =
            Product(activation, prefix + "ffn_down_exps.weight", p.width, e, coefficient);
        for (std::uint32_t j = 0; j < p.width; ++j)
          routed[j] +=
              down[j] * Vector(prefix + "ffn_down_exps.scale", e) * probabilities[e] / selected_sum;
      }
      routed = Norm(routed, prefix + "post_ffw_norm_2.weight");
      for (std::uint32_t j = 0; j < p.width; ++j) shared[j] += routed[j];
    }
    shared = Norm(shared, prefix + "post_ffw_norm.weight");
    for (std::uint32_t j = 0; j < p.width; ++j)
      result.output.push_back((shared[j] + residual[j]) * 1.125);
  }
  std::ignore = capacity;  // virtual ring changes addressing, not the unrolled oracle
  return result;
}
}  // namespace llmp::test_support::gemma4_scalar
#endif  // LLMP_TESTS_SUPPORT_GEMMA4_SCALAR_H_
