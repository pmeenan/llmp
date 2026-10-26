// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// External llama.cpp reference only; this does not implement llmpalooza inference.
//
// Teacher-forced rows of one fixed token sequence: the prompt
// (tests/unit/dsv4_engine_serving_gpu_test.cc's Tokens(0, 1285), rebuilt
// here) decoded in ubatches of 512 from a cleared memory, then for
// each forced token the last row's logits of the WATCH IDs and its argmax,
// before decoding that forced token alone. Flash attention on, F16 caches,
// one sequence, context 4,096.
//
//   forced_rows MODEL FORCED(comma) WATCH(id,id;...)  > rows.jsonl
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"

namespace {
void require(bool ok, const std::string& m) {
  if (!ok) throw std::runtime_error(m);
}

std::vector<int> ints(const std::string& s, char sep) {
  std::vector<int> out;
  std::stringstream in(s);
  std::string t;
  while (std::getline(in, t, sep))
    if (!t.empty()) out.push_back(std::atoi(t.c_str()));
  return out;
}

void decode(llama_context* ctx, const std::vector<llama_token>& ids, int pos0) {
  for (size_t at = 0; at < ids.size(); at += 512) {
    const size_t n = std::min<size_t>(512, ids.size() - at);
    llama_batch batch = llama_batch_init((int32_t)n, 0, 1);
    batch.n_tokens = (int32_t)n;
    for (size_t i = 0; i < n; ++i) {
      batch.token[i] = ids[at + i];
      batch.pos[i] = pos0 + (int)(at + i);
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = at + i + 1 == ids.size();
    }
    const int status = llama_decode(ctx, batch);
    llama_synchronize(ctx);
    llama_batch_free(batch);
    require(status == 0, "decode failed");
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 4, "usage: forced_rows MODEL FORCED WATCH");
    constexpr llama_token seed[7] = {2, 818, 5279, 529, 7001, 563, 42};
    std::vector<llama_token> prompt(1285);
    for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = seed[(i + i / 7) % 7];
    const std::vector<int> forced = ints(argv[2], ',');
    std::vector<std::vector<int>> watch;
    {
      std::stringstream rows(argv[3]);
      std::string r;
      while (std::getline(rows, r, ';')) watch.push_back(ints(r, ','));
    }
    require(watch.size() == forced.size(), "one watch list a forced row");
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    require(bool(model), "model load failed");
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    require(bool(ctx), "context creation failed");
    decode(ctx.get(), prompt, 0);
    for (size_t row = 0; row < forced.size(); ++row) {
      const float* l = llama_get_logits_ith(ctx.get(), -1);
      require(l != nullptr, "missing logits");
      const int top = (int)(std::max_element(l, l + n_vocab) - l);
      std::printf("{\"row\":%zu,\"argmax\":%d,\"top\":%.6f,\"watch\":{", row, top, (double)l[top]);
      for (size_t i = 0; i < watch[row].size(); ++i)
        std::printf("%s\"%d\":%.6f", i ? "," : "", watch[row][i], (double)l[watch[row][i]]);
      std::printf("}}\n");
      if (row + 1 < forced.size()) decode(ctx.get(), {forced[row]}, (int)(prompt.size() + row));
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "forced_rows: %s\n", e.what());
    return 1;
  }
}
