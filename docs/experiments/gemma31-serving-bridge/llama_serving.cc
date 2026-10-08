// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Dedicated Gemma31 C1/C4 real-sequence serving comparison.
// MODEL OUT 1|4 INPUT_4X8192 quality|cycle|corpus; prepare MODEL TEXT NEW_OUT.
#include <cuda.h>
#include <dlfcn.h>
#include <limits.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "llama.h"

namespace {
void Require(bool good, const char* detail) {
  if (!good) throw std::runtime_error(detail);
}
// Observe the driver already used by this model process; never load another driver.
void DriverProof() {
  Require(!std::getenv("LD_PRELOAD"), "unqualified stock preload");
  struct LoadedDriver {
    void* handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    ~LoadedDriver() {
      if (handle) dlclose(handle);
    }
  } driver;
  Require(driver.handle != nullptr, "CUDA driver not already loaded");
  std::array<char, PATH_MAX> defining{};
  const auto symbol = [&](const char* name) {
    void* address = dlsym(driver.handle, name);
    Dl_info info{};
    std::array<char, PATH_MAX> path{};
    Require(address && dladdr(address, &info) && info.dli_fname &&
                realpath(info.dli_fname, path.data()),
            "unresolved defining CUDA driver library");
    if (defining[0] == 0) defining = path;
    Require(std::strcmp(defining.data(), path.data()) == 0,
            "CUDA functions resolve to different libraries");
    return address;
  };
  const auto init = reinterpret_cast<decltype(&cuInit)>(symbol("cuInit"));
  const auto version =
      reinterpret_cast<decltype(&cuDriverGetVersion)>(symbol("cuDriverGetVersion"));
  const auto count = reinterpret_cast<decltype(&cuDeviceGetCount)>(symbol("cuDeviceGetCount"));
  const auto context = reinterpret_cast<decltype(&cuCtxGetCurrent)>(symbol("cuCtxGetCurrent"));
  const auto device = reinterpret_cast<decltype(&cuCtxGetDevice)>(symbol("cuCtxGetDevice"));
  const auto name = reinterpret_cast<decltype(&cuDeviceGetName)>(symbol("cuDeviceGetName"));
  const auto uuid = reinterpret_cast<decltype(&cuDeviceGetUuid)>(symbol("cuDeviceGetUuid_v2"));
  const auto attribute =
      reinterpret_cast<decltype(&cuDeviceGetAttribute)>(symbol("cuDeviceGetAttribute"));
  int driver_version = 0, devices = 0, major = 0, minor = 0, sms = 0;
  CUcontext current = nullptr;
  CUdevice ordinal = -1;
  CUuuid id{};
  std::array<char, 256> description{};
  Require(
      init(0) == CUDA_SUCCESS && version(&driver_version) == CUDA_SUCCESS &&
          count(&devices) == CUDA_SUCCESS && devices == 1 && context(&current) == CUDA_SUCCESS &&
          current != nullptr && device(&ordinal) == CUDA_SUCCESS && ordinal == 0 &&
          name(description.data(), static_cast<int>(description.size()), ordinal) == CUDA_SUCCESS &&
          uuid(&id, ordinal) == CUDA_SUCCESS &&
          attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, ordinal) ==
              CUDA_SUCCESS &&
          attribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, ordinal) ==
              CUDA_SUCCESS &&
          attribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, ordinal) == CUDA_SUCCESS &&
          major == 12 && minor == 1 && sms > 0,
      "CUDA initialization/current device proof failed");
  std::array<char, 33> hex{};
  constexpr char digits[] = "0123456789abcdef";
  for (std::size_t i = 0; i < sizeof(id.bytes); ++i) {
    const auto byte = static_cast<unsigned char>(id.bytes[i]);
    hex[i * 2] = digits[byte >> 4];
    hex[i * 2 + 1] = digits[byte & 15];
  }
  std::cout << "SERVING_DRIVER initialized=1 current_context=1 ordinal=" << ordinal
            << " devices=" << devices << " major=" << major << " minor=" << minor << " sms=" << sms
            << " driver_api=" << driver_version << " uuid=" << hex.data()
            << " libcuda=" << defining.data() << " name=" << description.data() << '\n';
  std::cout.flush();
  Require(bool(std::cout), "publishing initialized CUDA driver proof failed");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    const bool prepare = argc == 5 && std::string_view(argv[1]) == "prepare";
    Require(prepare || argc == 6,
            "MODEL OUT 1|4 INPUT_4X8192 quality|cycle|corpus; prepare MODEL TEXT NEW_OUT");
    const std::string_view mode = prepare ? "prepare" : argv[5];
    int count = 1;
    if (!prepare) {
      const std::string_view argument(argv[3]);
      const auto parsed =
          std::from_chars(argument.data(), argument.data() + argument.size(), count);
      Require(parsed.ec == std::errc{} && parsed.ptr == argument.data() + argument.size(),
              "invalid exact cohort argument");
    }
    Require((count == 1 || count == 4) && (prepare || mode == "quality" || mode == "cycle" ||
                                           (mode == "corpus" && count == 1)),
            "unbounded profile/cohort/mode");
    constexpr int vocab = 262144, history_rows = 8192, prefix_rows = 8063, steps = 128;
    const std::filesystem::path out(prepare ? argv[4] : argv[2]);
    Require(!std::filesystem::exists(out), "output already exists");
    std::vector<std::int32_t> supplied;
    std::string text;
    if (prepare) {
      const auto length = std::filesystem::file_size(argv[3]);
      Require(length > 0 && length <= 262144 && std::filesystem::is_regular_file(argv[3]),
              "unbounded text input");
      text.resize(static_cast<std::size_t>(length));
      std::ifstream file(argv[3], std::ios::binary);
      file.read(text.data(), static_cast<std::streamsize>(text.size()));
      Require(bool(file) && file.peek() == std::char_traits<char>::eof(), "text input changed");
    } else {
      supplied.resize(4 * history_rows);
      std::ifstream file(argv[4], std::ios::binary);
      file.read(reinterpret_cast<char*>(supplied.data()), supplied.size() * sizeof(std::int32_t));
      Require(
          bool(file) && file.peek() == std::char_traits<char>::eof() &&
              std::ranges::all_of(supplied, [](auto token) { return token >= 0 && token < vocab; }),
          "invalid bounded authenticated carrier");
      for (int owner = 0; owner < 4; ++owner) {
        auto sequence = std::span(supplied).subspan(owner * history_rows, history_rows);
        Require(sequence[0] == 2 &&
                    std::ranges::count(mode == "quality" ? sequence.first(prefix_rows) : sequence,
                                       2) == 1,
                "each source history needs one leading BOS");
      }
    }
    Require(std::filesystem::create_directory(out), "exclusive output directory refused");
    std::filesystem::permissions(out, std::filesystem::perms::owner_all);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = prepare ? 0 : 999;
    mp.vocab_only = prepare;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(prepare ? argv[2] : argv[1], mp), llama_model_free);
    Require(bool(model) && llama_vocab_n_tokens(llama_model_get_vocab(model.get())) == vocab,
            "checkpoint is not the approved Gemma31 vocabulary");
    if (prepare) {
      // d812 vocab_only returns before typed hparams, but retains scalar GGUF metadata.
      const auto metadata_is = [&](const char* key, std::string_view expected) {
        std::array<char, 16> value{};
        const auto length = llama_model_meta_val_str(model.get(), key, value.data(), value.size());
        return length >= 0 && static_cast<std::size_t>(length) < value.size() &&
               std::string_view(value.data(), static_cast<std::size_t>(length)) == expected;
      };
      Require(metadata_is("general.architecture", "gemma4") &&
                  metadata_is("gemma4.block_count", "60") &&
                  metadata_is("gemma4.embedding_length", "5376"),
              "vocabulary-only metadata is not the approved Gemma31 shape");
    } else {
      Require(llama_model_n_layer(model.get()) == 60 && llama_model_n_embd(model.get()) == 5376,
              "checkpoint is not the approved Gemma31 shape");
    }
    const auto* vocabulary = llama_model_get_vocab(model.get());
    if (prepare) {
      supplied.resize(text.size() + 2);
      const auto tokens =
          llama_tokenize(vocabulary, text.data(), static_cast<int>(text.size()), supplied.data(),
                         static_cast<int>(supplied.size()), true, true);
      Require(tokens >= history_rows && supplied[0] == 2, "fixed excerpt has insufficient IDs");
      supplied.resize(history_rows);
      Require(std::ranges::count(supplied, 2) == 1, "prepared prefix contains another BOS");
      std::ofstream file(out / "ids.i32", std::ios::binary);
      file.write(reinterpret_cast<const char*>(supplied.data()),
                 history_rows * sizeof(std::int32_t));
      file.flush();
      Require(bool(file), "complete vocabulary-only publication failed");
      model.reset();
      llama_backend_free();
      std::cout << "SERVING_INPUT_RETIRED vocab_only=1 model_contexts=0 supplied_rows=8192\n";
      return 0;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = count * history_rows;
    cp.n_batch = 256;
    cp.n_ubatch = 256;
    cp.n_seq_max = count;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.swa_full = false;
    cp.kv_unified = false;
    cp.no_perf = false;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    Require(bool(ctx) && llama_n_ctx_seq(ctx.get()) >= history_rows &&
                llama_n_seq_max(ctx.get()) == static_cast<std::uint32_t>(count),
            "context shape differs from the admitted independent-owner recipe");
    DriverProof();
    // Caller vectors are bounded separately from llama.cpp's logged context/cache allocations.
    constexpr std::uint64_t host_bound = 1100ULL << 20;
    const std::uint64_t publication_bytes =
        mode == "cycle" ? 0
                        : static_cast<std::uint64_t>(mode == "corpus" ? 1024 : 129 * count) *
                              vocab * sizeof(float);
    Require(publication_bytes + 2ULL * count * vocab * sizeof(float) +
                    supplied.capacity() * sizeof(std::int32_t) +
                    129ULL * count * sizeof(std::int32_t) <=
                host_bound,
            "caller vector envelope refused");
    std::cout << "SERVING_REFERENCE_HOST bound=" << host_bound
              << " publication_bytes=" << publication_bytes << '\n';
    llama_batch batch = llama_batch_init(256, 0, 1);
    std::vector<std::vector<float>> heads(count, std::vector<float>(vocab));
    std::vector<float> publication;
    if (mode != "cycle")
      publication.resize(static_cast<std::size_t>(mode == "corpus" ? 1024 : 129 * count) * vocab);
    std::vector<std::int32_t> emitted(129 * count);
    std::vector<float> final_capture(static_cast<std::size_t>(count) * vocab);
    const auto copy_row = [&](int index, int owner) {
      const auto* row = llama_get_logits_ith(ctx.get(), index);
      Require(row != nullptr, "completed full vocabulary head unavailable");
      std::copy_n(row, vocab, heads[owner].begin());
    };
    const auto argmax = [&](int owner) {
      return static_cast<std::int32_t>(std::max_element(heads[owner].begin(), heads[owner].end()) -
                                       heads[owner].begin());
    };
    const auto prefill = [&] {
      llama_memory_clear(llama_get_memory(ctx.get()), true);
      for (int owner = 0; owner < count; ++owner) {
        for (int first = 0; first < prefix_rows; first += 256) {
          batch.n_tokens = std::min(256, prefix_rows - first);
          const bool last = first + batch.n_tokens == prefix_rows;
          for (int index = 0; index < batch.n_tokens; ++index) {
            batch.token[index] = supplied[owner * history_rows + first + index];
            batch.pos[index] = first + index;
            batch.n_seq_id[index] = 1;
            batch.seq_id[index][0] = owner;
            batch.logits[index] = last && index == batch.n_tokens - 1;
          }
          Require(llama_decode(ctx.get(), batch) == 0, "independent 256-column prefill failed");
          if (last) copy_row(batch.n_tokens - 1, owner);
        }
      }
    };
    const auto cycle = [&](std::string_view phase, bool timed, bool trace) {
      const auto started = std::chrono::steady_clock::now();
      prefill();
      const auto prefilling = std::chrono::steady_clock::now();
      for (int owner = 0; owner < count; ++owner) {
        emitted[owner] = argmax(owner);
        if (trace)
          std::copy(heads[owner].begin(), heads[owner].end(),
                    publication.begin() + static_cast<std::size_t>(owner) * vocab);
      }
      for (int step = 0; step < steps; ++step) {
        batch.n_tokens = count;
        for (int owner = 0; owner < count; ++owner) {
          batch.token[owner] = timed || !trace
                                   ? emitted[step * count + owner]
                                   : supplied[owner * history_rows + prefix_rows + step];
          batch.pos[owner] = prefix_rows + step;
          batch.n_seq_id[owner] = 1;
          batch.seq_id[owner][0] = owner;
          batch.logits[owner] = true;
        }
        Require(llama_decode(ctx.get(), batch) == 0, "physical C1/C4 generation batch failed");
        for (int owner = 0; owner < count; ++owner) {
          copy_row(owner, owner);
          emitted[(step + 1) * count + owner] = argmax(owner);
          if (trace)
            std::copy(
                heads[owner].begin(), heads[owner].end(),
                publication.begin() + (static_cast<std::size_t>(step + 1) * count + owner) * vocab);
        }
      }
      // One final-row diagnostic copy is matched by the native serving callback.
      for (int owner = 0; owner < count; ++owner)
        std::copy(heads[owner].begin(), heads[owner].end(),
                  final_capture.begin() + static_cast<std::size_t>(owner) * vocab);
      llama_synchronize(ctx.get());
      for (int owner = 0; owner < count; ++owner)
        Require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), owner) == 8190,
                "completed public cache endpoint differs");
      const auto ended = std::chrono::steady_clock::now();
      const auto directory = out / phase;
      std::filesystem::create_directory(directory);
      std::filesystem::permissions(directory, std::filesystem::perms::owner_all);
      const auto write = [&](const char* name, const auto& values) {
        std::ofstream file(directory / name, std::ios::binary);
        file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(values[0]));
        file.flush();
        Require(bool(file), "complete public publication failed");
      };
      if (trace) write("heads.f32", publication);
      write("tokens.i32", emitted);
      write("final-head.f32", final_capture);
      std::cout << "SERVING_REFERENCE phase=" << phase << " owners=" << count
                << " emitted=" << 129 * count
                << " waves=128 cursor=8191 cache_max=8190 pending_position=8191 timed=" << timed
                << " cycle_seconds=" << std::chrono::duration<double>(ended - started).count()
                << " prefill_seconds="
                << std::chrono::duration<double>(prefilling - started).count()
                << " decode_seconds=" << std::chrono::duration<double>(ended - prefilling).count()
                << " context=" << llama_n_ctx(ctx.get())
                << " per_sequence_context=" << llama_n_ctx_seq(ctx.get())
                << " batch=256 ubatch=256 F16=1 normal_ring=1 seq_max="
                << llama_n_seq_max(ctx.get()) << '\n';
    };
    const auto corpus = [&](std::string_view phase) {
      llama_memory_clear(llama_get_memory(ctx.get()), true);
      for (int row = 0; row < 1024; ++row) {
        batch.n_tokens = 1;
        batch.token[0] = supplied[row];
        batch.pos[0] = row;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        Require(llama_decode(ctx.get(), batch) == 0, "scalar corpus query failed");
        copy_row(0, 0);
        std::copy(heads[0].begin(), heads[0].end(),
                  publication.begin() + static_cast<std::size_t>(row) * vocab);
      }
      const auto directory = out / phase;
      std::filesystem::create_directory(directory);
      std::filesystem::permissions(directory, std::filesystem::perms::owner_all);
      std::ofstream file(directory / "heads.f32", std::ios::binary);
      file.write(reinterpret_cast<const char*>(publication.data()),
                 publication.size() * sizeof(float));
      file.flush();
      Require(bool(file), "complete corpus publication failed");
      std::cout << "SERVING_REFERENCE_CORPUS phase=" << phase
                << " rows=1024 targets=1023 query_rows=1 cursor=1024\n";
    };
    if (mode == "quality") {
      cycle("first", false, true);
      cycle("repeat", false, true);
    } else if (mode == "cycle") {
      cycle("warm", false, false);
      cycle("paid", true, false);
    } else {
      corpus("first");
      corpus("repeat");
    }
    llama_synchronize(ctx.get());
    llama_batch_free(batch);
    ctx.reset();
    model.reset();
    llama_backend_free();
    std::cout << "SERVING_REFERENCE_RETIRED\n";
    std::cout.flush();
    Require(bool(std::cout), "public successful retirement publication failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "serving reference failed: " << error.what() << '\n';
    return 1;
  }
}
