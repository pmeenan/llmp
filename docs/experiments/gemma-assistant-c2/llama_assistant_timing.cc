// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Standalone diagnostic linked to the unchanged original-image libraries.
// TARGET ASSISTANT IDS_I32 NEW_OUTPUT 2 MODE(serial|batch) STAGE0_ROOT INCOMING.
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
constexpr int kWidth = 2816, kVocab = 262144;
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
void Architecture(llama_model* model, const char* expected, int layers) {
  std::array<char, 64> arch{};
  const int length =
      llama_model_meta_val_str(model, "general.architecture", arch.data(), arch.size());
  Require(length > 0 && length < static_cast<int>(arch.size()) &&
              std::string(arch.data()) == expected && llama_model_n_layer(model) == layers &&
              llama_model_n_embd_out(model) == kWidth &&
              llama_vocab_n_tokens(llama_model_get_vocab(model)) == kVocab,
          "model does not match the closed 26B target/assistant shape");
}
std::vector<std::uint8_t> SequenceState(llama_context* ctx, int owner) {
  llama_synchronize(ctx);
  const auto size = llama_state_seq_get_size(ctx, owner);
  Require(size > 0 && size <= 32 * 1024 * 1024, "unbounded opaque sequence state");
  std::vector<std::uint8_t> data(size);
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
              int layer, int dimension, int heads, int capacity) {
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
  Require(std::ranges::all_of(std::span(result.physical_k).subspan(first_pad),
                              [](auto x) { return x == 0; }) &&
              std::ranges::all_of(std::span(result.physical_v).subspan(first_pad),
                                  [](auto x) { return x == 0; }),
          "unused read padding is not the original initialized zero bytes");
  result.descriptors = "{\"k\":" + km + ",\"v\":" + vm + "}";
  return result;
}
struct Owner {
  int past{};
  llama_token anchor{};
  std::vector<float> feature;
  Cache local, global;
  std::vector<std::uint8_t> state;
};
void SaveCache(const std::filesystem::path& path, const Cache& cache, const char* prefix) {
  const std::string name(prefix);
  Write<std::uint16_t>(path / (name + "-k.f16"), cache.k);
  Write<std::uint16_t>(path / (name + "-v.f16"), cache.v);
  Write<std::uint16_t>(path / (name + "-physical-k.f16"), cache.physical_k);
  Write<std::uint16_t>(path / (name + "-physical-v.f16"), cache.physical_v);
  Write<std::int32_t>(path / (name + "-positions.i32"), cache.positions);
  Write<std::uint8_t>(path / (name + "-membership.u8"), cache.membership);
}
template <class T>
std::vector<T> Read(const std::filesystem::path& path, std::size_t count) {
  Require(std::filesystem::file_size(path) == count * sizeof(T), "fixed input byte count differs");
  std::vector<T> data(count);
  std::ifstream f(path, std::ios::binary);
  Require(bool(f.read(reinterpret_cast<char*>(data.data()),
                      static_cast<std::streamsize>(count * sizeof(T)))),
          "fixed input read failed");
  return data;
}
void SameFile(const std::filesystem::path& a, const std::filesystem::path& b) {
  const auto bytes = std::filesystem::file_size(a);
  Require(bytes <= 32 * 1024 * 1024 && std::filesystem::file_size(b) == bytes,
          "reconstructed stage0 bytes differ");
  Require(Read<std::uint8_t>(a, bytes) == Read<std::uint8_t>(b, bytes),
          "reconstructed stage0 payload differs");
}
}  // namespace

int main(int argc, char** argv) {
  static_assert(std::endian::native == std::endian::little);
  static_assert(sizeof(float) == 4 && sizeof(llama_token) == 4);
  umask(0077);
  Require(argc == 9,
          "TARGET ASSISTANT IDS_I32 NEW_OUTPUT 2 MODE(serial|batch) STAGE0_ROOT INCOMING");
  const int count = std::string(argv[5]) == "1" ? 1 : (std::string(argv[5]) == "2" ? 2 : 0);
  const bool joined = std::string(argv[6]) == "batch";
  Require(count == 2 && (joined || std::string(argv[6]) == "serial") && (!joined || count == 2),
          "closed owner/shape mode refused");
  Require(!std::getenv("GGML_CUDA_DISABLE_GRAPHS") && !std::getenv("GGML_CUDA_DISABLE_FUSION"),
          "original graph/fusion defaults required");
  std::vector<std::int32_t> ids(1024);
  std::ifstream input(argv[3], std::ios::binary);
  input.read(reinterpret_cast<char*>(ids.data()), 4096);
  Require(bool(input) && input.peek() == std::char_traits<char>::eof() && ids[0] == 2 &&
              std::ranges::all_of(ids, [](auto x) { return x >= 0 && x < kVocab; }),
          "invalid canonical 1024-token input");
  const std::filesystem::path out(argv[4]);
  Directory(out);
  Write<std::int32_t>(out / "inputs.i32", ids);
  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = 999;
  mp.load_mode = LLAMA_LOAD_MODE_NONE;
  mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
  auto* target_model = llama_model_load_from_file(argv[1], mp);
  Require(target_model != nullptr, "target load failed");
  Architecture(target_model, "gemma4", 30);
  auto cp = llama_context_default_params();
  cp.n_ctx = static_cast<std::uint32_t>(count * 4096);
  cp.n_batch = cp.n_ubatch = 128;
  cp.n_seq_max = static_cast<std::uint32_t>(count);
  cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
  cp.type_k = cp.type_v = GGML_TYPE_F16;
  cp.kv_unified = false;
  cp.swa_full = false;
  cp.no_perf = false;
  auto* target = llama_init_from_model(target_model, cp);
  Require(target && llama_n_ctx_seq(target) == 4096 && llama_n_seq_max(target) == unsigned(count),
          "target context bounds differ");
  llama_set_embeddings_nextn(target, true, false);
  auto* split = dynamic_cast<llama_kv_cache_iswa*>(llama_get_memory(target));
  Require(split != nullptr, "original split cache interface absent");
  ggml_init_params gp{.mem_size = 1024 * 1024, .mem_buffer = nullptr, .no_alloc = true};
  auto* descriptors = ggml_init(gp);
  Require(descriptors != nullptr, "descriptor-only context allocation failed");
  // Construct shared-cell contexts while target is empty. Original cache
  // construction resizes/resets the shared cell metadata. Prefill follows setup.
  auto* draft_model = llama_model_load_from_file(argv[2], mp);
  Require(draft_model != nullptr, "Q8 assistant load failed");
  // Original public n_layer is the trunk count, excluding nextn blocks.
  Architecture(draft_model, "gemma4-assistant", 0);
  std::array<char, 16> blocks{};
  const int block_length = llama_model_meta_val_str(draft_model, "gemma4-assistant.block_count",
                                                    blocks.data(), blocks.size());
  Require(block_length == 1 && std::string(blocks.data()) == "4" &&
              llama_model_n_layer_nextn(draft_model) == 4 &&
              llama_model_n_embd(draft_model) == 1024,
          "unexpected assistant total/trunk/MTP layer count or internal width");
  cp.ctx_other = target;
  cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
  auto* draft = llama_init_from_model(draft_model, cp);
  Require(draft && llama_get_ctx_other(draft) == target && llama_n_ctx_seq(draft) == 4096 &&
              llama_n_seq_max(draft) == unsigned(count),
          "borrowed assistant context differs");
  llama_set_embeddings_nextn(draft, true, false);
  auto batch = llama_batch_init(128, kWidth, 1);
  batch.token = static_cast<llama_token*>(std::malloc(128 * sizeof(llama_token)));
  Require(batch.token && batch.embd, "assistant token/feature batch allocation failed");
  auto target_batch = llama_batch_init(128, 0, 1);
  std::vector<Owner> owners(count);
  for (int owner = 0; owner < count; ++owner) {
    const int past = 64 + owner;
    target_batch.n_tokens = past;
    for (int i = 0; i < past; ++i) {
      target_batch.token[i] = ids[i];
      target_batch.pos[i] = i;
      target_batch.n_seq_id[i] = 1;
      target_batch.seq_id[i][0] = owner;
      target_batch.logits[i] = i == past - 1;
    }
    Decode(target, target_batch);
    owners[owner].past = past;
    owners[owner].feature = Row(llama_get_embeddings_nextn_ith(target, past - 1), kWidth);
    owners[owner].anchor = Winner(Row(llama_get_logits_ith(target, past - 1), kVocab));
  }
  const auto capture = [&](int owner, bool local) {
    return Capture(local ? split->get_swa() : split->get_base(), descriptors, owner, count,
                   owners[owner].past, local ? 28 : 29, local ? 256 : 512, local ? 8 : 2,
                   local ? 1280 : 4096);
  };
  for (int owner = 0; owner < count; ++owner) {
    auto& state = owners[owner];
    state.local = capture(owner, true);
    state.global = capture(owner, false);
    state.state = SequenceState(target, owner);
    const auto dir = out / ("owner-" + std::to_string(owner));
    Directory(dir);
    Write<float>(dir / "feature.f32", state.feature);
    Write<llama_token>(dir / "anchor.i32", std::span(&state.anchor, 1));
    Write<std::uint8_t>(dir / "target-state.bin", state.state);
    SaveCache(dir, state.local, "local");
    SaveCache(dir, state.global, "global");
    std::ostringstream metadata;
    metadata << "{\"version\":1,\"profile\":\"26B-A4B\",\"owner\":" << owner
             << ",\"stream\":" << owner << ",\"sequence\":" << owner
             << ",\"completed_endpoint\":" << state.past << ",\"query_position\":" << state.past
             << ",\"feature_position\":" << state.past - 1
             << ",\"feature_width\":2816,\"vocabulary\":262144,\"local_window\":1024,"
                "\"local_capacity\":1280,\"global_capacity\":4096,\"read_cells\":256,"
                "\"local_layer\":28,\"global_layer\":29,\"local_descriptors\":"
             << state.local.descriptors << ",\"global_descriptors\":" << state.global.descriptors
             << "}\n";
    Text(dir / "metadata.json", metadata.str());
  }
  const auto unchanged = [&]() {
    llama_synchronize(draft);
    llama_synchronize(target);
    for (int owner = 0; owner < count; ++owner)
      Require(owners[owner].state == SequenceState(target, owner) &&
                  owners[owner].local == capture(owner, true) &&
                  owners[owner].global == capture(owner, false),
              "assistant mutated original target state, caches or cell metadata");
  };
  // Stage zero must be reconstructed byte-for-byte, including physical
  // descriptor offsets/strides and every cell position/membership, before timing.
  const std::array<const char*, 11> files{
      "metadata.json",       "feature.f32",          "anchor.i32",          "local-k.f16",
      "local-v.f16",         "global-k.f16",         "global-v.f16",        "local-positions.i32",
      "local-membership.u8", "global-positions.i32", "global-membership.u8"};
  for (int owner = 0; owner < 2; ++owner)
    for (auto name : files)
      SameFile(out / ("owner-" + std::to_string(owner)) / name,
               std::filesystem::path(argv[7]) / ("owner-" + std::to_string(owner)) / name);
  auto incoming =
      Read<float>(std::filesystem::path(argv[8]) / "incoming-feature.f32", 3 * 2 * kWidth);
  auto anchors = Read<llama_token>(std::filesystem::path(argv[8]) / "incoming-anchor.i32", 3 * 2);
  Require(std::ranges::all_of(incoming, [](float f) { return std::isfinite(f); }) &&
              std::ranges::all_of(anchors, [](llama_token t) { return t >= 0 && t < kVocab; }),
          "nonfinite/noncanonical fixed input");
  for (int owner = 0; owner < 2; ++owner)
    Require(std::memcmp(incoming.data() + owner * kWidth, owners[owner].feature.data(),
                        kWidth * sizeof(float)) == 0 &&
                anchors[owner] == owners[owner].anchor,
            "fixed initial input differs from authenticated stage zero");
  std::vector<float> published_heads(2 * kVocab), published_features(2 * kWidth);
  std::array<llama_token, 2> winners{};
  const auto wave = [&](int step) {
    for (int group = 0; group < (joined ? 1 : 2); ++group) {
      batch.n_tokens = joined ? 2 : 1;
      for (int row = 0; row < batch.n_tokens; ++row) {
        const int owner = joined ? row : group;
        batch.token[row] = anchors[step * 2 + owner];
        std::copy_n(incoming.data() + (step * 2 + owner) * kWidth, kWidth,
                    batch.embd + row * kWidth);
        batch.pos[row] = owners[owner].past;
        batch.n_seq_id[row] = 1;
        batch.seq_id[row][0] = owner;
        batch.logits[row] = true;
      }
      Require(llama_decode(draft, batch) == 0, "timed decode refused");
      // The original getter synchronizes the queued full-vocabulary D2H copy;
      // no redundant explicit synchronize is charged ahead of this getter.
      for (int row = 0; row < batch.n_tokens; ++row) {
        const int owner = joined ? row : group;
        const auto* head = llama_get_logits_ith(draft, row);
        const auto* feature = llama_get_embeddings_nextn_ith(draft, row);
        Require(head && feature, "completed output absent");
        std::copy_n(head, kVocab, published_heads.data() + owner * kVocab);
        std::copy_n(feature, kWidth, published_features.data() + owner * kWidth);
      }
    }
    Require(std::ranges::all_of(published_heads, [](float f) { return std::isfinite(f); }) &&
                std::ranges::all_of(published_features, [](float f) { return std::isfinite(f); }),
            "nonfinite completed wave");
    for (int owner = 0; owner < 2; ++owner)
      winners[owner] = Winner(std::span(published_heads).subspan(owner * kVocab, kVocab));
  };
  for (int step = 0; step < 3; ++step) wave(step);
  unchanged();
  const auto reuse = llama_perf_context(draft).n_reused;
  std::array<std::array<llama_token, 2>, 32> chosen{};
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 32; ++i) {
    wave(i % 3);
    chosen[i] = winners;
  }
  const auto seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  const auto reuse_after = llama_perf_context(draft).n_reused;
  unchanged();
  Write<float>(out / "last-heads.f32", published_heads);
  Write<float>(out / "last-postprojection.f32", published_features);
  Write<std::byte>(out / "paid-winners.i32", std::as_bytes(std::span(chosen)));
  std::cout.precision(12);
  std::cout
      << "REFERENCE_PAID mode=" << (joined ? "batch" : "serial")
      << " waves=32 rows=64 seconds=" << seconds
      << " ggml_graph_reused_delta=" << reuse_after - reuse
      << " stage0_byte_exact=1 physical_state_unchanged=1 graphs_allowed=1 fusion_allowed=1\n";
  llama_synchronize(draft);
  llama_synchronize(target);
  llama_batch_free(batch);
  llama_batch_free(target_batch);
  llama_free(draft);
  llama_model_free(draft_model);
  ggml_free(descriptors);
  llama_free(target);
  llama_model_free(target_model);
  llama_backend_free();
  std::cout << "ASSISTANT_REFERENCE_COMPLETE owners=" << count << " mode=" << argv[6]
            << " context_per_owner=4096 ubatch=128 local=1280 global=4096 waves=32"
               " target_state_unchanged=1\n";
}
