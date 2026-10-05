// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// External reference-only diagnostic. Taps preserve the shared FFN's
// gate/up/GeGLU interior; verify that final fused/unfused drift survives
// instrumentation before attributing a difference to these intermediates.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::array<llama_token, 6> kPrompt = {2, 818, 5279, 529, 7001, 563};
constexpr int kSteps = 32;
void Require(bool good, const char* text) {
  if (!good) throw std::runtime_error(text);
}
double Seconds(Clock::duration time) { return std::chrono::duration<double>(time).count(); }
struct Trace {
  std::filesystem::path out;
  bool enabled = true;
  std::uint32_t count = 0;
  std::ofstream manifest;
  bool layers_only = false;
  bool first_only = false;
  std::string one_tag;
  static bool Callback(ggml_tensor* tensor, bool ask, void* opaque) {
    auto& trace = *static_cast<Trace*>(opaque);
    if (!trace.enabled) return false;
    std::string name(tensor->name);
    bool selected = false;
    // Explicitly retain this normalized/scaled router input. This can
    // suppress a larger fusion, so final untraced/traced agreement matters.
    if (tensor->op == GGML_OP_MUL && tensor->src[1]) {
      const std::string weight(tensor->src[1]->name);
      if (weight.starts_with("blk.") && weight.ends_with(".ffn_gate_inp.scale")) {
        const auto end = weight.find('.', 4);
        name = "router_input-" + weight.substr(4, end - 4);
        selected = true;
      }
    }
    const auto dash = name.rfind('-');
    if (dash != std::string::npos) {
      const auto tag = name.substr(0, dash);
      for (const auto* wanted : {"attn_out", "ffn_norm_1", "ffn_geglu", "ffn_mlp", "ffn_norm_2",
                                 "ffn_moe_logits", "ffn_moe_topk", "ffn_moe", "ffn_moe_out",
                                 "ffn_moe_weights_norm", "ffn_moe_combined", "l_out"})
        selected |= tag == wanted;
      if (trace.layers_only) selected = tag == "l_out";
      if (trace.first_only && tag != "l_out")
        selected &= name == tag + "-0" || name == tag + "-0 (reshaped)";
      if (!trace.one_tag.empty()) {
        const bool wanted =
            tag == trace.one_tag ||
            (trace.one_tag == "routed" &&
             (tag == "ffn_norm_2" || tag == "ffn_moe_geglu" || tag == "ffn_moe_topk"));
        selected =
            tag == "l_out" || (wanted && (name == tag + "-0" || name == tag + "-0 (reshaped)"));
      }
    }
    if (ask) return selected;
    if (!selected) return true;
    const auto bytes = ggml_nbytes(tensor);
    Require(bytes <= 1048576, "tap exceeds diagnostic envelope");
    Require(tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32, "unexpected tap type");
    std::vector<char> raw(bytes);
    ggml_backend_tensor_get(tensor, raw.data(), 0, bytes);
    std::ofstream file(trace.out / (name + ".bin"), std::ios::binary);
    file.write(raw.data(), static_cast<std::streamsize>(raw.size()));
    file.flush();
    Require(bool(file), "writing tap failed");
    trace.manifest << name << '\t' << static_cast<int>(tensor->type);
    for (int i = 0; i < 4; ++i) trace.manifest << '\t' << tensor->ne[i];
    for (int i = 0; i < 4; ++i) trace.manifest << '\t' << tensor->nb[i];
    trace.manifest << '\t' << bytes << '\n';
    ++trace.count;
    return true;
  }
};
}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc == 4 || argc == 5,
            "usage: MODEL OUTDIR quality|quality-unfused "
            "[broad|layers|first|geglu|down|router|routed]");
    const std::string mode(argv[3]);
    const bool warm = mode == "warm", all_heads = mode == "quality-all";
    const bool unfused_diagnostic = mode == "quality-unfused";
    Require(warm || mode == "quality" || all_heads || unfused_diagnostic, "unknown mode");
    Require(!warm, "trace is a numerical diagnostic, not a performance harness");
    // Presence/values of these switches would change the comparator policy.
    const char* disabled = std::getenv("GGML_CUDA_DISABLE_FUSION");
    Require(unfused_diagnostic ? disabled && std::string(disabled) == "1" : !disabled,
            "fusion-disable environment disagrees with mode");
    Require(!std::getenv("GGML_CUDA_DISABLE_GRAPHS"), "graph-disable environment present");
    const std::filesystem::path out(argv[2]);
    std::filesystem::create_directories(out);
    Trace trace{out, true, 0, std::ofstream(out / "taps.tsv"), false, false, {}};
    if (argc == 5) {
      const std::string profile(argv[4]);
      Require(profile == "layers" || profile == "broad" || profile == "first" ||
                  profile == "geglu" || profile == "down" || profile == "router" ||
                  profile == "routed",
              "unknown trace profile");
      trace.layers_only = profile == "layers";
      trace.first_only = profile == "first";
      if (profile == "geglu") trace.one_tag = "ffn_moe_geglu";
      if (profile == "down") trace.one_tag = "ffn_moe_down_scaled";
      if (profile == "router") trace.one_tag = "ffn_moe_topk";
      if (profile == "routed") trace.one_tag = "routed";
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    Require(bool(model), "model load failed");
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    Require(vocab == 262144, "unexpected vocabulary");
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 128;
    cp.n_ubatch = 128;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.no_perf = false;
    cp.cb_eval = Trace::Callback;
    cp.cb_eval_user_data = &trace;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    Require(bool(ctx), "context creation failed");
    // Reuse the batch container. Only the last input row requests a head,
    // matching native frontier-head prefill and full-vocabulary publication.
    llama_batch batch = llama_batch_init(128, 0, 1);
    std::vector<float> published(static_cast<std::size_t>(vocab));
    const auto decode = [&](const llama_token* ids, int rows, int past) {
      batch.n_tokens = rows;
      for (int i = 0; i < rows; ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = all_heads || i == rows - 1;
      }
      Require(llama_decode(ctx.get(), batch) == 0, "decode failed");
      // The public getter synchronizes before returning its host output.
      const float* logits = llama_get_logits_ith(ctx.get(), -1);
      Require(logits != nullptr, "missing full logits");
      std::copy_n(logits, vocab, published.begin());
    };
    const auto best = [&]() -> llama_token {
      return static_cast<llama_token>(std::max_element(published.begin(), published.end()) -
                                      published.begin());
    };
    auto memory = llama_get_memory(ctx.get());
    llama_memory_clear(memory, true);
    decode(kPrompt.data(), static_cast<int>(kPrompt.size()), 0);
    trace.enabled = false;
    std::cout << "LLAMA_TAPS count=" << trace.count << '\n';
    int past = static_cast<int>(kPrompt.size());
    if (warm) {
      for (int i = 0; i < 8; ++i) {
        const auto token = best();
        decode(&token, 1, past++);
      }
      llama_memory_clear(memory, true);
      decode(kPrompt.data(), static_cast<int>(kPrompt.size()), 0);
      past = static_cast<int>(kPrompt.size());
      // The first stable call builds, the second captures, the third replays.
      for (int i = 0; i < 3; ++i) {
        const auto seed = best();
        decode(&seed, 1, past++);
        std::cout << "LLAMA_TIMED_PREFIX appended=" << seed << " past=" << past << '\n';
      }
    }
    const auto before = llama_perf_context(ctx.get());
    std::array<llama_token, kSteps> chosen{};
    const auto started = Clock::now();
    for (int i = 0; i < kSteps; ++i) {
      const auto token = best();
      chosen[static_cast<std::size_t>(i)] = token;
      if (warm) {
        decode(&token, 1, past++);
      } else {
        std::ofstream file(out / ("logits-" + std::to_string(i) + ".f32"), std::ios::binary);
        file.write(reinterpret_cast<const char*>(published.data()),
                   static_cast<std::streamsize>(published.size() * sizeof(float)));
        Require(bool(file), "writing full logits failed");
        if (i + 1 < kSteps) decode(&token, 1, past++);
      }
    }
    const double elapsed = Seconds(Clock::now() - started);
    const auto after = llama_perf_context(ctx.get());
    std::cout << (warm ? "LLAMA_WARM" : "LLAMA_QUALITY") << " seconds=" << elapsed
              << " completed_chunks=" << (warm ? kSteps : kSteps - 1)
              << " ggml_graph_reused_delta=" << after.n_reused - before.n_reused
              << " context=" << llama_n_ctx(ctx.get())
              << " fusion=" << (unfused_diagnostic ? "disabled-diagnostic" : "enabled")
              << " graphs=allowed head=" << (all_heads ? "all" : "frontier") << '\n';
    for (int i = 0; i < kSteps; ++i)
      std::cout << "LLAMA_TOKEN step=" << i << " id=" << chosen[static_cast<std::size_t>(i)]
                << '\n';
    llama_batch_free(batch);
    ctx.reset();
    model.reset();
    llama_backend_free();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
