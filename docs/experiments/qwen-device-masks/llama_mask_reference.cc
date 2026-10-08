// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// External llama.cpp v0.6.0 C1 reference, literal IDs and paid full heads.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"

namespace {
using Clock = std::chrono::steady_clock;
void Require(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
void Decode(llama_context* ctx, const std::vector<llama_token>& ids, int first, bool head) {
  auto batch = llama_batch_init(static_cast<std::int32_t>(ids.size()), 0, 1);
  batch.n_tokens = static_cast<std::int32_t>(ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i) {
    batch.token[i] = ids[i];
    batch.pos[i] = first + static_cast<int>(i);
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = head && i + 1 == ids.size();
  }
  const int result = llama_decode(ctx, batch);
  if (head) llama_synchronize(ctx);
  llama_batch_free(batch);
  Require(result == 0, "reference decode failed");
}
struct Result {
  double seconds = 0, decode_seconds = 0;
  std::vector<llama_token> tokens;
  std::vector<float> heads;
};
Result Generate(llama_context* ctx, const std::vector<llama_token>& prompt,
                const std::vector<llama_token>* history = nullptr) {
  Result result;
  result.heads.reserve(32U * 248320U);
  result.tokens.reserve(32);
  const auto start = Clock::now();
  llama_memory_clear(llama_get_memory(ctx), true);
  for (std::size_t at = 0; at < prompt.size(); at += 512) {
    const auto n = std::min<std::size_t>(512, prompt.size() - at);
    Decode(ctx,
           {prompt.begin() + static_cast<std::ptrdiff_t>(at),
            prompt.begin() + static_cast<std::ptrdiff_t>(at + n)},
           static_cast<int>(at), at + n == prompt.size());
  }
  const auto read = [&] {
    const float* row = llama_get_logits_ith(ctx, -1);
    Require(row != nullptr, "reference missing vocabulary head");
    result.tokens.push_back(static_cast<llama_token>(std::max_element(row, row + 248320) - row));
    result.heads.insert(result.heads.end(), row, row + 248320);
  };
  read();
  const auto decode_start = Clock::now();
  for (int i = 1; i < 32; ++i) {
    Decode(ctx, {history ? (*history)[static_cast<std::size_t>(i - 1)] : result.tokens.back()},
           static_cast<int>(prompt.size()) + i - 1, true);
    read();
  }
  result.decode_seconds = Seconds(decode_start);
  result.seconds = Seconds(start);
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc == 5, "usage: llama_mask_reference MODEL IDS_TXT OUT NATIVE_HISTORY_OR_DASH");
    std::ifstream input(argv[2]);
    Require(bool(input), "reference literal input absent");
    std::vector<llama_token> prompt;
    long long token = 0;
    while (input >> token) {
      Require(token >= 0 && token < 248320, "reference literal ID out of range");
      prompt.push_back(static_cast<llama_token>(token));
    }
    Require(input.eof() && prompt.size() == 1536, "reference needs 1536 literal IDs");
    const auto out = std::filesystem::path(argv[3]);
    Require(std::filesystem::create_directory(out), "reference output already exists");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    Require(bool(model), "reference model load failed");
    Require(llama_vocab_n_tokens(llama_model_get_vocab(model.get())) == 248320,
            "reference vocabulary differs");
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    Require(bool(ctx), "reference context allocation failed");
    const auto warm = Generate(ctx.get(), prompt);
    const auto paid = Generate(ctx.get(), prompt);
    Require(std::all_of(paid.heads.begin(), paid.heads.end(),
                        [](float value) { return std::isfinite(value); }),
            "reference head is nonfinite");
    Require(warm.tokens == paid.tokens, "reference choices not repeatable");
    Require(warm.heads.size() == paid.heads.size() &&
                std::memcmp(warm.heads.data(), paid.heads.data(),
                            paid.heads.size() * sizeof(float)) == 0,
            "reference full heads not repeatable");
    auto history = paid.tokens;
    if (std::string(argv[4]) != "-") {
      std::ifstream supplied(argv[4]);
      Require(bool(supplied), "native quality history absent");
      history.clear();
      while (supplied >> token) {
        Require(token >= 0 && token < 248320, "native quality token invalid");
        history.push_back(static_cast<llama_token>(token));
      }
      Require(supplied.eof() && history.size() == 32, "native quality history needs32 IDs");
    }
    // Separate untimed common-history quality; every input comes from the
    // supplied native history, regardless of the reference's own choices.
    const auto teacher = Generate(ctx.get(), prompt, &history);
    Require(std::all_of(teacher.heads.begin(), teacher.heads.end(),
                        [](float value) { return std::isfinite(value); }),
            "reference teacher head is nonfinite");
    std::ofstream teacher_payload(out / "teacher-heads.f32", std::ios::binary);
    teacher_payload.write(reinterpret_cast<const char*>(teacher.heads.data()),
                          static_cast<std::streamsize>(teacher.heads.size() * sizeof(float)));
    teacher_payload.close();
    Require(bool(teacher_payload), "reference teacher-head write failed");
    std::ofstream payload(out / "plain-heads.f32", std::ios::binary);
    payload.write(reinterpret_cast<const char*>(paid.heads.data()),
                  static_cast<std::streamsize>(paid.heads.size() * sizeof(float)));
    payload.close();
    Require(bool(payload), "reference full-head write failed");
    std::ofstream record(out / "reference.json");
    record.precision(12);
    record << "{\"context\":2048,\"chunk\":512,\"prompt_tokens\":1536,"
              "\"generated\":32,\"full_head_rows\":32,\"seconds\":"
           << paid.seconds << ",\"decode_seconds\":" << paid.decode_seconds << ",\"tokens\":[";
    for (std::size_t i = 0; i < paid.tokens.size(); ++i) record << (i ? "," : "") << paid.tokens[i];
    record << "],\"teacher_history\":[";
    for (std::size_t i = 0; i < history.size(); ++i) record << (i ? "," : "") << history[i];
    record << "]}\n";
    record.close();
    Require(bool(record), "reference result write failed");
    ctx.reset();
    model.reset();
    llama_backend_free();
    std::puts("LLAMA_MASK_REFERENCE_COMPLETE");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "reference error: %s\n", error.what());
    return 1;
  }
}
