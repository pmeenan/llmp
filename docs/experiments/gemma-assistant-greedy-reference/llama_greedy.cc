// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Standalone diagnostic linked to the unchanged original-image libraries.
// TARGET ASSISTANT IDS_I32 NEW_OUTPUT PROFILE(26|31) MODE(unit|teacher).
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama-ext.h"
#include "llama-kv-cache-iswa.h"
#include "llama.h"

namespace {
constexpr int kVocab = 262144;
void Require(bool value, const char* message) {
  if (!value) {
    std::cerr << "REFUSAL " << message << '\n';
    // Fail-stop, rather than destroy borrowed contexts on an unproved error path.
    std::exit(1);
  }
}
template <class T>
void Write(const std::filesystem::path& path, std::span<const T> values) {
  std::ofstream file(path, std::ios::binary | std::ios::noreplace);
  Require(bool(file), "exclusive file creation failed");
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size_bytes()));
  file.flush();
  Require(bool(file), "complete file write failed");
}
void Text(const std::filesystem::path& path, const std::string& value) { Write<char>(path, value); }
void Directory(const std::filesystem::path& path) {
  Require(std::filesystem::create_directory(path), "output directory must be new");
}
std::vector<float> Row(const float* row, int width) {
  Require(row != nullptr, "missing completed output row");
  std::vector<float> result(row, row + width);
  Require(std::ranges::all_of(result, [](float f) { return std::isfinite(f); }),
          "nonfinite completed output");
  return result;
}
llama_token Winner(std::span<const float> row) {
  return static_cast<llama_token>(std::max_element(row.begin(), row.end()) - row.begin());
}
void Decode(llama_context* ctx, llama_batch batch) {
  const int result = llama_decode(ctx, batch);
  llama_synchronize(ctx);
  Require(result == 0, "decode did not complete successfully");
}
void Architecture(llama_model* model, const char* expected, int layers, int width) {
  std::array<char, 64> arch{};
  const int length =
      llama_model_meta_val_str(model, "general.architecture", arch.data(), arch.size());
  Require(length > 0 && length < static_cast<int>(arch.size()) &&
              std::string(arch.data()) == expected && llama_model_n_layer(model) == layers &&
              llama_model_n_embd_out(model) == width &&
              llama_vocab_n_tokens(llama_model_get_vocab(model)) == kVocab,
          "model does not match the closed target/assistant shape");
}
std::vector<std::uint8_t> SequenceState(llama_context* ctx, int owner, bool profile31,
                                        bool bounded32 = false) {
  llama_synchronize(ctx);
  const auto size = llama_state_seq_get_size(ctx, owner);
  Require(!bounded32 || profile31, "bounded32 state is profile31 only");
  Require(size > 0 && size <= (bounded32 ? 128U : (profile31 ? 64U : 32U)) * 1024 * 1024,
          "unbounded opaque sequence state");
  std::vector<std::uint8_t> data(size);
  if (bounded32)
    Require(data.capacity() <= 128ULL * 1024 * 1024, "bounded32 opaque capacity exceeds funding");
  Require(llama_state_seq_get_data(ctx, data.data(), data.size(), owner) == data.size(),
          "opaque sequence state capture failed");
  return data;
}
struct Cache {
  int layer{}, dimension{}, heads{}, capacity{}, occupied{}, read{};
  std::vector<std::int32_t> positions;
  std::vector<std::uint8_t> membership;
  std::vector<std::uint16_t> k, v, physical_k, physical_v;
  std::string descriptors;
  bool operator==(const Cache&) const = default;
};
std::string TensorMetadata(const ggml_tensor* tensor) {
  std::ostringstream text;
  text << "{\"type\":" << static_cast<int>(tensor->type) << ",\"ne\":[";
  for (int i = 0; i < 4; ++i) text << (i ? "," : "") << tensor->ne[i];
  text << "],\"nb\":[";
  for (int i = 0; i < 4; ++i) text << (i ? "," : "") << tensor->nb[i];
  text << "]}";
  return text.str();
}
std::vector<std::uint16_t> CacheTensor(ggml_tensor* tensor, int dimension, int heads, int read,
                                       int capacity, int owners, int owner, std::string& metadata) {
  const auto width = static_cast<std::size_t>(dimension * heads);
  const auto bytes = width * static_cast<std::size_t>(read) * 2;
  const auto stream_bytes = width * static_cast<std::size_t>(capacity) * 2;
  Require(tensor && tensor->type == GGML_TYPE_F16 && tensor->ne[0] == dimension &&
              tensor->ne[1] == heads && tensor->ne[2] == read && tensor->ne[3] == 1 &&
              tensor->nb[0] == 2 && tensor->nb[1] == static_cast<std::size_t>(dimension) * 2 &&
              tensor->nb[2] == width * 2 && tensor->nb[3] == stream_bytes &&
              tensor->view_offs == stream_bytes * static_cast<std::size_t>(owner),
          "unexpected F16 cache view shape, pitch or stream offset");
  const auto* root = tensor->view_src;
  Require(root && !root->view_src && root->type == GGML_TYPE_F16 && root->buffer && root->data &&
              root->ne[0] == static_cast<std::int64_t>(width) && root->ne[1] == capacity &&
              root->ne[2] == owners && root->ne[3] == 1 && root->nb[0] == 2 &&
              root->nb[1] == width * 2 && root->nb[2] == stream_bytes &&
              root->nb[3] == stream_bytes * static_cast<std::size_t>(owners) &&
              tensor->view_offs <= ggml_nbytes(root) &&
              bytes <= ggml_nbytes(root) - tensor->view_offs && ggml_nbytes(tensor) == bytes,
          "unexpected cache storage root or bounds");
  metadata = "{\"view\":" + TensorMetadata(tensor) + ",\"root\":" + TensorMetadata(root) +
             ",\"relative_offset\":" + std::to_string(tensor->view_offs) + "}";
  std::vector<std::uint16_t> data(width * static_cast<std::size_t>(read));
  ggml_backend_tensor_get(tensor, data.data(), 0, bytes);
  // Original allocation zero-clear plus observed unused cells establishes padding.
  // Cold prefixes below 256 are the only accepted physical occupancy here.
  for (const auto bits : data) Require((bits & 0x7c00u) != 0x7c00u, "nonfinite F16 cache payload");
  return data;
}
Cache Capture(llama_kv_cache* source, ggml_context* descriptors, int owner, int owners, int past,
              int layer, int dimension, int heads, int capacity, bool zero_tail = true) {
  Require(source && source->get_size() == static_cast<std::uint32_t>(capacity) &&
              source->get_n_stream() == static_cast<std::uint32_t>(owners) &&
              source->type_k() == GGML_TYPE_F16 && source->type_v() == GGML_TYPE_F16,
          "unexpected split cache capacity, streams or types");
  const auto layers = source->get_layer_ids();
  Require(std::ranges::find(layers, static_cast<std::uint32_t>(layer)) != layers.end(),
          "borrowed source layer is absent");
  const auto& cells = source->get_cells(owner);
  Require(cells.size() == static_cast<std::uint32_t>(capacity) &&
              cells.get_used() == static_cast<std::uint32_t>(past) &&
              cells.used_max_p1() == static_cast<std::uint32_t>(past),
          "cold prefix physical occupancy differs");
  Cache result{.layer = layer,
               .dimension = dimension,
               .heads = heads,
               .capacity = capacity,
               .occupied = past,
               .read = 256,
               .positions = {},
               .membership = {},
               .k = {},
               .v = {},
               .physical_k = {},
               .physical_v = {},
               .descriptors = {}};
  result.positions.resize(capacity, -1);
  result.membership.resize(capacity);
  for (int i = 0; i < capacity; ++i) {
    const bool empty = cells.is_empty(i);
    if (!empty) {
      result.positions[i] = cells.pos_get(i);
      Require(cells.seq_count(i) == 1, "unexpected shared physical cell");
    }
    result.membership[i] = static_cast<std::uint8_t>(cells.seq_has(i, owner));
    Require(i < past ? (!empty && result.positions[i] == i && result.membership[i] == 1)
                     : (empty && result.membership[i] == 0),
            "positions/membership do not authenticate the frozen cold prefix");
  }
  llama_kv_cache::slot_info slot{};
  slot.resize(1);
  slot.s0 = slot.s1 = static_cast<std::uint32_t>(owner);
  slot.strm[0] = owner;
  // Getter-only descriptor, never submitted for cache allocation or writes.
  slot.idxs[0].push_back(0);
  Require(source->get_n_kv(slot) == 256, "unexpected padded cache read width");
  std::string km, vm;
  auto* kt = source->get_k(descriptors, layer, 256, slot);
  Require(kt->view_src == source->get_k_storage(layer), "K view does not use original storage");
  result.k = CacheTensor(kt, dimension, heads, 256, capacity, owners, owner, km);
  result.v = CacheTensor(source->get_v(descriptors, layer, 256, slot), dimension, heads, 256,
                         capacity, owners, owner, vm);
  std::string unused;
  result.physical_k = CacheTensor(source->get_k(descriptors, layer, capacity, slot), dimension,
                                  heads, capacity, capacity, owners, owner, unused);
  result.physical_v = CacheTensor(source->get_v(descriptors, layer, capacity, slot), dimension,
                                  heads, capacity, capacity, owners, owner, unused);
  const auto first_pad = static_cast<std::size_t>(past) * dimension * heads;
  Require(!zero_tail || (std::ranges::all_of(std::span(result.physical_k).subspan(first_pad),
                                             [](auto x) { return x == 0; }) &&
                         std::ranges::all_of(std::span(result.physical_v).subspan(first_pad),
                                             [](auto x) { return x == 0; })),
          "unused read padding is not the original initialized zero bytes");
  result.descriptors = "{\"k\":" + km + ",\"v\":" + vm + "}";
  return result;
}
struct Decision {
  int keep;
  llama_token next;
};
Decision Judge(std::span<const llama_token> proposal, std::span<const float> heads) {
  Require(proposal.size() == 4 && heads.size() == 4U * kVocab &&
              std::ranges::all_of(heads, [](float x) { return std::isfinite(x); }),
          "invalid complete greedy judge rows");
  int matched = 0;
  while (matched < 3 && proposal[matched + 1] == Winner(heads.subspan(matched * kVocab, kVocab)))
    ++matched;
  return {matched + 1, Winner(heads.subspan(matched * kVocab, kVocab))};
}
void Append(std::vector<float>& out, std::span<const float> values) {
  out.insert(out.end(), values.begin(), values.end());
}
void Run32(llama_context* target, llama_context* draft, llama_batch batch, llama_batch target_batch,
           const std::array<llama_token, 1024>& ids, const std::filesystem::path& out, int width,
           bool speculate, llama_token eos) {
  constexpr int emitted = 32;
  struct Unit {
    int past{}, depth{}, rows{}, keep{};
  };
  std::vector<float> frontier, incoming, initial_feature, verify, features, draft_head, predictions,
      archived_verify, archived_features, archived_last;
  frontier.reserve(kVocab);
  predictions.reserve(emitted * kVocab);
  if (speculate) {
    incoming.reserve(width);
    initial_feature.reserve(width);
    verify.reserve(4U * kVocab);
    features.reserve(4U * width);
    draft_head.reserve(kVocab);
    archived_verify.reserve(emitted * 4U * kVocab);
    archived_features.reserve(emitted * 4U * width);
    archived_last.reserve(emitted * kVocab);
  }
  std::uint64_t capacity = 0;
  for (const auto* v : {&frontier, &incoming, &initial_feature, &verify, &features, &draft_head,
                        &predictions, &archived_verify, &archived_features, &archived_last})
    capacity += v->capacity() * sizeof(float);
  // Add128MiB bounded endpoint96 snapshot and1MiB descriptors/tokens/metadata.
  Require(capacity + (129ULL << 20U) <= (speculate ? (384ULL << 20U) : (192ULL << 20U)),
          "original bounded32 known vector capacity exceeds envelope");
  auto copy = [](std::vector<float>& dest, const float* src, int count) {
    Require(src != nullptr, "missing completed bounded32 output");
    dest.assign(src, src + count);
  };
  for (int round = 0; round < 5; ++round) {
    const bool quality = round == 1 || round == 2, timed = round >= 3;
    const char* name = round == 0   ? "warm"
                       : round == 1 ? "quality-first"
                       : round == 2 ? "quality-repeat"
                       : round == 3 ? "timed-first"
                                    : "timed-repeat";
    const auto dir = out / name;
    Directory(dir);
    predictions.clear();
    archived_verify.clear();
    archived_features.clear();
    archived_last.clear();
    if (draft) {
      llama_synchronize(draft);
      llama_memory_clear(llama_get_memory(draft), false);
    }
    llama_synchronize(target);
    llama_memory_clear(llama_get_memory(target), true);
    target_batch.n_tokens = 64;
    for (int row = 0; row < 64; ++row) {
      target_batch.token[row] = ids[row];
      target_batch.pos[row] = row;
      target_batch.n_seq_id[row] = 1;
      target_batch.seq_id[row][0] = 0;
      target_batch.logits[row] = row == 63;
    }
    Decode(target, target_batch);
    copy(frontier, llama_get_logits_ith(target, 63), kVocab);
    if (speculate) {
      copy(incoming, llama_get_embeddings_nextn_ith(target, 63), width);
      initial_feature.assign(incoming.begin(), incoming.end());
    }
    std::array<llama_token, emitted> tokens{};
    std::array<Unit, emitted> units{};
    int count = 0, unit_count = 0, verified_rows = 0, drafted_rows = 0, eos_count = 0;
    const auto started = std::chrono::steady_clock::now();
    while (count < emitted) {
      const int remaining = emitted - count, past = 64 + count;
      const int depth = speculate && remaining >= 2 ? std::min(3, remaining - 1) : 0;
      Require(std::ranges::all_of(frontier, [](float x) { return std::isfinite(x); }),
              "original target frontier refused nonfinite head");
      const auto anchor = Winner(frontier);
      if (quality) Append(predictions, frontier);
      std::array<llama_token, 4> proposal{};
      proposal[0] = anchor;
      int keep = 1;
      if (depth != 0) {
        for (int step = 0; step < depth; ++step) {
          batch.n_tokens = 1;
          batch.token[0] = proposal[step];
          std::copy(incoming.begin(), incoming.end(), batch.embd);
          batch.pos[0] = past;
          batch.n_seq_id[0] = 1;
          batch.seq_id[0][0] = 0;
          batch.logits[0] = true;
          Decode(draft, batch);
          copy(draft_head, llama_get_logits_ith(draft, 0), kVocab);
          copy(incoming, llama_get_embeddings_nextn_ith(draft, 0), width);
          Require(std::ranges::all_of(incoming, [](float x) { return std::isfinite(x); }),
                  "original recurrent feature nonfinite");
          Require(std::ranges::all_of(draft_head, [](float x) { return std::isfinite(x); }),
                  "original draft judge refused nonfinite head");
          proposal[step + 1] = Winner(draft_head);
        }
        // All draft jobs retire and its cache borrow ends before target writers.
        llama_synchronize(draft);
        target_batch.n_tokens = depth + 1;
        for (int row = 0; row <= depth; ++row) {
          target_batch.token[row] = proposal[row];
          target_batch.pos[row] = past + row;
          target_batch.n_seq_id[row] = 1;
          target_batch.seq_id[row][0] = 0;
          target_batch.logits[row] = true;
        }
        Decode(target, target_batch);
        verify.clear();
        features.clear();
        for (int row = 0; row <= depth; ++row) {
          const auto* h = llama_get_logits_ith(target, row);
          const auto* f = llama_get_embeddings_nextn_ith(target, row);
          Require(h && f, "missing original complete verify row");
          verify.insert(verify.end(), h, h + kVocab);
          features.insert(features.end(), f, f + width);
        }
        Require(std::ranges::all_of(verify, [](float x) { return std::isfinite(x); }),
                "original target judge refused nonfinite head");
        int matched = 0;
        while (matched < depth &&
               proposal[matched + 1] == Winner(std::span(verify).subspan(matched * kVocab, kVocab)))
          ++matched;
        keep = matched + 1;
        const auto selected = std::span(verify).subspan((keep - 1) * kVocab, kVocab);
        const auto selected_feature = std::span(features).subspan((keep - 1) * width, width);
        // Keep selected raw postnorm row before any next decode can overwrite it.
        frontier.assign(selected.begin(), selected.end());
        incoming.assign(selected_feature.begin(), selected_feature.end());
        Require(llama_memory_seq_rm(llama_get_memory(target), 0, past + keep, -1),
                "original logical rejection refused");
        llama_synchronize(target);
        if (quality) {
          Append(archived_verify, verify);
          Append(archived_features, features);
          Append(archived_last, draft_head);
          for (int row = 1; row < keep; ++row)
            Append(predictions, std::span(verify).subspan((row - 1) * kVocab, kVocab));
        }
        std::copy_n(proposal.begin(), keep, tokens.begin() + count);
      } else {
        target_batch.n_tokens = 1;
        target_batch.token[0] = anchor;
        target_batch.pos[0] = past;
        target_batch.n_seq_id[0] = 1;
        target_batch.seq_id[0][0] = 0;
        target_batch.logits[0] = true;
        Decode(target, target_batch);
        copy(frontier, llama_get_logits_ith(target, 0), kVocab);
        if (speculate) {
          copy(incoming, llama_get_embeddings_nextn_ith(target, 0), width);
          if (quality) {
            Append(archived_verify, frontier);
            Append(archived_features, incoming);
          }
        }
        tokens[count] = anchor;
      }
      for (int row = 0; row < keep; ++row) eos_count += tokens[count + row] == eos;
      units[unit_count++] = {.past = past, .depth = depth, .rows = depth + 1, .keep = keep};
      count += keep;
      verified_rows += depth + 1;
      drafted_rows += depth;
    }
    const auto finished = std::chrono::steady_clock::now();
    const double seconds = timed ? std::chrono::duration<double>(finished - started).count() : 0;
    Require(
        std::ranges::all_of(frontier, [](float x) { return std::isfinite(x); }) &&
            (!speculate || std::ranges::all_of(incoming, [](float x) { return std::isfinite(x); })),
        "original final bounded32 head/features nonfinite");
    // Full opaque state and cell checks are post-timer witnesses, not paid scanner cost.
    auto* split = dynamic_cast<llama_kv_cache_iswa*>(llama_get_memory(target));
    Require(split, "original split cache interface missing");
    Require(split->get_swa()->get_size() == 1280 && split->get_base()->get_size() == 4096,
            "original normal-ring capacity differs");
    for (auto* cache : {split->get_swa(), split->get_base()}) {
      Require(cache->get_n_stream() == 1 && cache->type_k() == GGML_TYPE_F16 &&
                  cache->type_v() == GGML_TYPE_F16,
              "original cache streams/types differ");
      const auto& cells = cache->get_cells(0);
      Require(cells.get_used() == 96 && cells.used_max_p1() == 96,
              "original completed endpoint differs");
      for (int i = 0; i < static_cast<int>(cells.size()); ++i)
        Require(i < 96 ? (!cells.is_empty(i) && cells.pos_get(i) == i && cells.seq_has(i, 0) &&
                          cells.seq_count(i) == 1)
                       : (cells.is_empty(i) && !cells.seq_has(i, 0)),
                "original endpoint membership differs");
    }
    if (round == 0) continue;
    auto state = SequenceState(target, 0, true, true);
    Write<llama_token>(dir / "tokens.i32", tokens);
    Write<float>(dir / "final-head.f32", frontier);
    Write<std::uint8_t>(dir / "end-state.bin", state);
    if (speculate) Write<float>(dir / "final-feature.f32", incoming);
    if (quality) {
      Write<float>(dir / "prediction-heads.f32", predictions);
      if (speculate) {
        Write<float>(dir / "initial-feature.f32", initial_feature);
        Write<float>(dir / "verify-heads.f32", archived_verify);
        Write<float>(dir / "verify-features.f32", archived_features);
        Write<float>(dir / "last-draft-heads.f32", archived_last);
      }
    }
    std::string encoded = "[";
    for (int i = 0; i < unit_count; ++i) {
      const auto& u = units[i];
      if (i) encoded += ',';
      encoded += "{\"past\":" + std::to_string(u.past) + ",\"depth\":" + std::to_string(u.depth) +
                 ",\"verified_rows\":" + std::to_string(u.rows) +
                 ",\"keep\":" + std::to_string(u.keep) + "}";
    }
    encoded += ']';
    Text(dir / "completion.json",
         "{\"mode\":\"" + std::string(speculate ? "unit32" : "plain32") + "\",\"phase\":\"" +
             (quality ? "quality" : "timed") +
             "\",\"emitted\":32,\"endpoint\":96,\"prefill_queries\":64,\"max_rows\":128,"
             "\"context\":4096,\"eog_ignored\":true,\"eos_id\":" +
             std::to_string(eos) + ",\"eos_occurrences\":" + std::to_string(eos_count) +
             ",\"verified_rows\":" + std::to_string(verified_rows) + ",\"drafted_rows\":" +
             std::to_string(drafted_rows) + ",\"seconds\":" + std::to_string(seconds) +
             ",\"actual_vector_capacity_bytes\":" + std::to_string(capacity) +
             ",\"semantic_cell_endpoint_checked\":true,\"physical_rejected_zero_claim\":false,"
             "\"units\":" +
             encoded + "}\n");
    std::cout << "GREEDY32_COMPLETED mode=" << (speculate ? "unit32" : "plain32")
              << " phase=" << name << " emitted=32 endpoint=96 verified=" << verified_rows
              << " drafted=" << drafted_rows << " seconds=" << seconds << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  static_assert(std::endian::native == std::endian::little);
  static_assert(sizeof(float) == 4 && sizeof(llama_token) == 4);
  umask(0077);
  Require(argc == 7 && (std::string(argv[5]) == "26" || std::string(argv[5]) == "31") &&
              (std::string(argv[6]) == "unit" || std::string(argv[6]) == "teacher" ||
               std::string(argv[6]) == "plain32" || std::string(argv[6]) == "unit32"),
          "TARGET ASSISTANT IDS_I32 NEW_OUTPUT PROFILE(26|31) MODE(unit|teacher)");
  const bool profile31 = std::string(argv[5]) == "31", teacher = std::string(argv[6]) == "teacher";
  const bool repeated = std::string_view(argv[6]).ends_with("32");
  const bool plain = std::string_view(argv[6]) == "plain32";
  Require(!repeated || profile31, "bounded32 first profile is31 only");
  const int width = profile31 ? 5376 : 2816, layers = profile31 ? 60 : 30;
  const int local_layer = layers - 2, global_layer = layers - 1;
  const int local_heads = profile31 ? 16 : 8, global_heads = profile31 ? 4 : 2;
  Require(!std::getenv("GGML_CUDA_DISABLE_GRAPHS") && !std::getenv("GGML_CUDA_DISABLE_FUSION"),
          "original graph/fusion defaults required");
  std::array<llama_token, 1024> ids{};
  std::ifstream input(argv[3], std::ios::binary);
  input.read(reinterpret_cast<char*>(ids.data()), sizeof(ids));
  Require(bool(input) && input.peek() == std::char_traits<char>::eof() && ids[0] == 2 &&
              std::ranges::all_of(ids, [](auto x) { return x >= 0 && x < kVocab; }),
          "invalid canonical 1024-token input");
  const std::filesystem::path out(argv[4]);
  Directory(out);
  Write<llama_token>(out / "inputs.i32", ids);
  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = 999;
  mp.load_mode = LLAMA_LOAD_MODE_NONE;
  mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
  auto* target_model = llama_model_load_from_file(argv[1], mp);
  Require(target_model != nullptr, "target load failed");
  Architecture(target_model, "gemma4", layers, width);
  auto cp = llama_context_default_params();
  cp.n_ctx = 4096;
  cp.n_batch = cp.n_ubatch = 128;
  cp.n_seq_max = 1;
  cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
  cp.type_k = cp.type_v = GGML_TYPE_F16;
  cp.kv_unified = false;
  cp.swa_full = false;
  auto* target = llama_init_from_model(target_model, cp);
  Require(target && llama_n_ctx_seq(target) == 4096 && llama_n_seq_max(target) == 1,
          "target context bounds differ");
  llama_set_embeddings_nextn(target, !plain, false);
  auto* split = dynamic_cast<llama_kv_cache_iswa*>(llama_get_memory(target));
  Require(split != nullptr, "original split cache interface absent");
  ggml_init_params gp{.mem_size = 1024 * 1024, .mem_buffer = nullptr, .no_alloc = true};
  auto* descriptors = ggml_init(gp);
  Require(descriptors != nullptr, "descriptor-only allocation failed");
  // Shared-cache context construction resets cell metadata: both contexts must
  // exist before every target prefill, not be created after the prefix.
  llama_model* draft_model = nullptr;
  llama_context* draft = nullptr;
  if (!plain) {
    draft_model = llama_model_load_from_file(argv[2], mp);
    Require(draft_model != nullptr, "assistant load failed");
    Architecture(draft_model, "gemma4-assistant", 0, width);
    std::array<char, 16> blocks{};
    const int length = llama_model_meta_val_str(draft_model, "gemma4-assistant.block_count",
                                                blocks.data(), blocks.size());
    Require(length == 1 && std::string(blocks.data()) == "4" &&
                llama_model_n_layer_nextn(draft_model) == 4 &&
                llama_model_n_embd(draft_model) == 1024,
            "assistant layer/internal width differs");
    cp.ctx_other = target;
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    draft = llama_init_from_model(draft_model, cp);
    Require(draft && llama_get_ctx_other(draft) == target && llama_n_ctx_seq(draft) == 4096 &&
                llama_n_seq_max(draft) == 1,
            "borrowed assistant context differs");
    llama_set_embeddings_nextn(draft, true, false);
  }
  auto batch = llama_batch_init(128, plain ? 0 : width, 1);
  if (!plain) batch.token = static_cast<llama_token*>(std::malloc(128 * sizeof(llama_token)));
  Require(batch.token && (plain || batch.embd), "assistant token/feature batch failed");
  auto target_batch = llama_batch_init(128, 0, 1);
  Require(target_batch.token, "target token batch failed");
  const auto capture = [&](int past, bool local, bool zero_tail) {
    return Capture(local ? split->get_swa() : split->get_base(), descriptors, 0, 1, past,
                   local ? local_layer : global_layer, local ? 256 : 512,
                   local ? local_heads : global_heads, local ? 1280 : 4096, zero_tail);
  };
  if (repeated) {
    const auto eos = llama_vocab_eos(llama_model_get_vocab(target_model));
    Require(eos == 106, "approved original target EOS differs");
    Run32(target, draft, batch, target_batch, ids, out, width, !plain, eos);
  } else
    for (int repeat = 0; repeat < 2; ++repeat) {
      const auto dir = out / (repeat == 0 ? "first" : "repeat");
      Directory(dir);
      llama_synchronize(draft);
      llama_synchronize(target);
      // Explicitly clear target payload as well as shared cells before an own
      // repeat. Rejected rows from the previous round need not have been zeroed.
      llama_memory_clear(llama_get_memory(draft), false);
      llama_memory_clear(llama_get_memory(target), true);
      target_batch.n_tokens = 64;
      for (int row = 0; row < 64; ++row) {
        target_batch.token[row] = ids[row];
        target_batch.pos[row] = row;
        target_batch.n_seq_id[row] = 1;
        target_batch.seq_id[row][0] = 0;
        target_batch.logits[row] = row == 63;
      }
      Decode(target, target_batch);
      auto initial = Row(llama_get_logits_ith(target, 63), kVocab);
      auto feature = Row(llama_get_embeddings_nextn_ith(target, 63), width);
      const auto protected_state = SequenceState(target, 0, profile31);
      const auto local = capture(64, true, true), global = capture(64, false, true);
      std::array<llama_token, 4> proposal{};
      proposal[0] = Winner(initial);
      std::vector<float> draft_heads, draft_features, incoming(feature);
      draft_heads.reserve(3U * kVocab);
      draft_features.reserve(3U * width);
      if (!teacher) {
        for (int step = 0; step < 3; ++step) {
          batch.n_tokens = 1;
          batch.token[0] = proposal[step];
          std::copy(incoming.begin(), incoming.end(), batch.embd);
          batch.pos[0] = 64;
          batch.n_seq_id[0] = 1;
          batch.seq_id[0][0] = 0;
          batch.logits[0] = true;
          Decode(draft, batch);
          auto head = Row(llama_get_logits_ith(draft, 0), kVocab);
          incoming = Row(llama_get_embeddings_nextn_ith(draft, 0), width);
          Append(draft_heads, head);
          Append(draft_features, incoming);
          proposal[step + 1] = Winner(head);
          llama_synchronize(target);
          Require(protected_state == SequenceState(target, 0, profile31) &&
                      local == capture(64, true, true) && global == capture(64, false, true),
                  "assistant changed protected target state/cache/cells");
        }
      } else {
        std::copy_n(ids.begin() + 64, 4, proposal.begin());
      }
      // Borrow ends: the assistant has completed and synchronized before target
      // writers run. Every one of the four real target queries is published.
      llama_synchronize(draft);
      target_batch.n_tokens = 4;
      for (int row = 0; row < 4; ++row) {
        target_batch.token[row] = proposal[row];
        target_batch.pos[row] = 64 + row;
        target_batch.n_seq_id[row] = 1;
        target_batch.seq_id[row][0] = 0;
        target_batch.logits[row] = true;
      }
      Decode(target, target_batch);
      std::vector<float> verify_heads, verify_features;
      verify_heads.reserve(4U * kVocab);
      verify_features.reserve(4U * width);
      for (int row = 0; row < 4; ++row) {
        Append(verify_heads, Row(llama_get_logits_ith(target, row), kVocab));
        Append(verify_features, Row(llama_get_embeddings_nextn_ith(target, row), width));
      }
      const auto judged = Judge(proposal, verify_heads);
      const int keep = teacher ? 4 : judged.keep;
      const auto selected_head = std::span(verify_heads).subspan((keep - 1) * kVocab, kVocab);
      const auto selected_feature = std::span(verify_features).subspan((keep - 1) * width, width);
      const auto next = Winner(selected_head);
      const auto verified_local = capture(68, true, false);
      const auto verified_global = capture(68, false, false);
      // Logical removal deliberately does not claim physical rejected-tail zero.
      Require(llama_memory_seq_rm(llama_get_memory(target), 0, 64 + keep, -1),
              "logical rejected tail removal refused");
      llama_synchronize(target);
      const auto accepted = SequenceState(target, 0, profile31);
      const auto accepted_local = capture(64 + keep, true, false);
      const auto accepted_global = capture(64 + keep, false, false);
      const auto semantic_equal = [keep](const Cache& a, const Cache& b) {
        const auto values = static_cast<std::size_t>(64 + keep) * a.dimension * a.heads;
        return a.dimension == b.dimension && a.heads == b.heads && a.k.size() >= values &&
               b.k.size() >= values && a.v.size() >= values && b.v.size() >= values &&
               std::memcmp(a.k.data(), b.k.data(), values * sizeof(a.k[0])) == 0 &&
               std::memcmp(a.v.data(), b.v.data(), values * sizeof(a.v[0])) == 0;
      };
      Require(semantic_equal(accepted_local, verified_local) &&
                  semantic_equal(accepted_global, verified_global),
              "logical rejection changed accepted semantic cache rows");
      // Retained POST-finalnorm selected carry is copied before any further
      // decode can replace public output pointers. Pending next is uncommitted.
      std::array<llama_token, 4> committed{};
      std::copy_n(proposal.begin(), keep, committed.begin());
      Write<float>(dir / "initial-head.f32", initial);
      Write<float>(dir / "initial-feature.f32", feature);
      Write<float>(dir / "draft-heads.f32", draft_heads);
      Write<float>(dir / "draft-features.f32", draft_features);
      Write<float>(dir / "verify-heads.f32", verify_heads);
      Write<float>(dir / "verify-features.f32", verify_features);
      Write<float>(dir / "selected-head.f32", selected_head);
      Write<float>(dir / "selected-feature.f32", selected_feature);
      Write<llama_token>(dir / "proposal.i32", proposal);
      Write<llama_token>(dir / "committed.i32", committed);
      Write<std::uint8_t>(dir / "protected-state.bin", protected_state);
      Write<std::uint8_t>(dir / "accepted-state.bin", accepted);
      Write<std::uint16_t>(dir / "accepted-local-k.f16",
                           std::span(accepted_local.k).first((64U + keep) * 256 * local_heads));
      Write<std::uint16_t>(dir / "accepted-local-v.f16",
                           std::span(accepted_local.v).first((64U + keep) * 256 * local_heads));
      Write<std::uint16_t>(dir / "accepted-global-k.f16",
                           std::span(accepted_global.k).first((64U + keep) * 512 * global_heads));
      Write<std::uint16_t>(dir / "accepted-global-v.f16",
                           std::span(accepted_global.v).first((64U + keep) * 512 * global_heads));
      Text(dir / "completion.json",
           "{\"profile\":\"" + std::string(profile31 ? "31" : "26") + "\",\"feature_width\":" +
               std::to_string(width) + ",\"vocab\":262144,\"mode\":\"" + std::string(argv[6]) +
               "\",\"past\":64,\"prefill_query_rows\":64,"
               "\"verify_rows\":4,\"keep\":" +
               std::to_string(keep) + ",\"completed_endpoint\":" + std::to_string(64 + keep) +
               ",\"next_anchor\":" + std::to_string(next) +
               ",\"pending_anchor_committed\":false,\"draft_protected_target\":true,"
               "\"semantic_cell_endpoint_checked\":true,\"physical_rejected_zero_claim\":false}\n");
    }
  if (draft) llama_synchronize(draft);
  llama_synchronize(target);
  llama_batch_free(batch);
  llama_batch_free(target_batch);
  if (draft) llama_free(draft);
  if (draft_model) llama_model_free(draft_model);
  ggml_free(descriptors);
  llama_free(target);
  llama_model_free(target_model);
  llama_backend_free();
  Text(out / "retired.json",
       repeated
           ? "{\"successful_teardown\":true,\"warm\":1,\"quality_rounds\":2,\"timed_rounds\":2}\n"
           : "{\"successful_teardown\":true,\"rounds\":2}\n");
  if (repeated)
    std::cout << "GREEDY_REFERENCE_RETIRED profile=" << argv[5] << " mode=" << argv[6]
              << " C=1 P=64 prefill_query=64 emitted=32 endpoint=96 warm=1 quality=2 timed=2\n";
  else
    std::cout << "GREEDY_REFERENCE_RETIRED profile=" << argv[5] << " mode=" << argv[6]
              << " C=1 P=64 prefill_query=64 depth=3 verify_rows=4 rounds=2\n";
}
