// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// jitllm.vecq's launch configurations on DeepSeek V4 Flash's product
// shapes (kernels/ggml/dsv4_fast.h), for choosing its defaults: each
// configuration's time per launch and the weight bytes it streams per
// second. Weights and activations are random bytes (the time does not
// depend on the values); each launch reads other experts (routed) or
// another copy of the matrix (dense), so the L2 cache holds none of it.
// Needs a CUDA device and about 6 GB.
//
//   jitllm_vecq_bench [--launches N] [--overlap F] [--vmm on|off] [--only TEXT]
//
// --overlap: in a multi-token routed case, the fraction of each later
// token's experts taken from the token before it (a verify's neighbouring
// tokens share some experts). --vmm on: the weights in device VMM as the
// paged node maps them (a 2 MiB backing each 2 MiB) instead of cudaMalloc.
// --only: the cases whose name contains TEXT.

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <format>
#include <print>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/dsv4_fast.h"

namespace {

namespace kg = jitllm::kernels::ggml;

bool Ok(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::println(stderr, "{}: {}", what, cudaGetErrorString(e));
    return false;
  }
  return true;
}

bool Ok(CUresult e, const char* what) {
  if (e != CUDA_SUCCESS) {
    const char* text = nullptr;
    (void)cuGetErrorString(e, &text);
    std::println(stderr, "{}: {}", what, text != nullptr ? text : "?");
    return false;
  }
  return true;
}

// Device memory as the paged node maps it (D-081): a reserved range with a
// separate 2 MiB backing mapped at each 2 MiB.
constexpr std::size_t kExtent = std::size_t{2} << 20U;
bool VmmAlloc(void** out, std::size_t bytes) {
  const std::size_t size = (bytes + kExtent - 1) / kExtent * kExtent;
  CUdeviceptr base = 0;
  if (!Ok(cuMemAddressReserve(&base, size, kExtent, 0, 0), "reserve")) {
    return false;
  }
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  for (std::size_t at = 0; at < size; at += kExtent) {
    CUmemGenericAllocationHandle h = 0;
    if (!Ok(cuMemCreate(&h, kExtent, &prop, 0), "create") ||
        !Ok(cuMemMap(base + at, kExtent, 0, h, 0), "map")) {
      return false;
    }
    (void)cuMemRelease(h);
  }
  CUmemAccessDesc access{};
  access.location = prop.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (!Ok(cuMemSetAccess(base, size, &access, 1), "access")) {
    return false;
  }
  *out = reinterpret_cast<void*>(base);  // NOLINT(performance-no-int-to-ptr)
  return true;
}

struct Case {
  std::string name;
  ggml_type type;
  int k;
  int n;
  bool routed;
  int tokens;
  bool glu;
  bool per_slot;
  double upstream_us;  // GGML's MMVQ on the same product (the model's profile), or 0
};

constexpr int kExperts = 256;
constexpr int kUsed = 6;
constexpr int kDenseCopies = 16;
constexpr int kQ8Block = 36;

std::size_t U(std::int64_t v) { return static_cast<std::size_t>(v); }

std::int64_t RowBytes(ggml_type type, int k) {
  return static_cast<std::int64_t>(ggml_row_size(type, k));
}

}  // namespace

int main(int argc, char** argv) {
  int launches = 64;
  double overlap = 0.25;
  bool vmm = false;
  std::string only;
  int spin = 0;
  const std::span args(argv, static_cast<std::size_t>(argc));
  for (std::size_t i = 1; i + 1 < args.size(); i += 2) {
    const std::string_view a = args[i];
    const std::string_view v = args[i + 1];
    if (a == "--launches") {
      std::from_chars(v.data(), v.data() + v.size(), launches);
    } else if (a == "--overlap") {
      overlap = std::stod(std::string(v));
    } else if (a == "--vmm") {
      vmm = v == "on";
    } else if (a == "--only") {
      only = v;
    } else if (a == "--spin") {
      std::from_chars(v.data(), v.data() + v.size(), spin);
    }
  }
  // CPU threads spinning meanwhile, as the paged node's lanes poll through
  // a request's steps.
  class Spinners {
   public:
    explicit Spinners(int n) {
      for (int s = 0; s < n; ++s) {
        threads_.emplace_back([this] {
          std::uint64_t count = 0;
          while (!stop_.load(std::memory_order_relaxed)) {
            ++count;
          }
          (void)count;
        });
      }
    }
    Spinners(const Spinners&) = delete;
    Spinners& operator=(const Spinners&) = delete;
    Spinners(Spinners&&) = delete;
    Spinners& operator=(Spinners&&) = delete;
    ~Spinners() {
      stop_ = true;
      for (std::thread& t : threads_) {
        t.join();
      }
    }

   private:
    std::atomic<bool> stop_{false};
    std::vector<std::thread> threads_;
  };
  const Spinners spinners(spin);
  const std::vector<Case> cases = {
      {"routed gate+up IQ2_XS 4096x2048, 1 token", GGML_TYPE_IQ2_XS, 4096, 2048, true, 1, true,
       false, 151.8},
      {"routed down IQ3_XXS 2048x4096, 1 token", GGML_TYPE_IQ3_XXS, 2048, 4096, true, 1, false,
       true, 101.7},
      {"routed gate+up IQ2_XS 4096x2048, 4 tokens", GGML_TYPE_IQ2_XS, 4096, 2048, true, 4, true,
       false, 496.5},
      {"routed down IQ3_XXS 2048x4096, 4 tokens", GGML_TYPE_IQ3_XXS, 2048, 4096, true, 4, false,
       true, 356.0},
      {"routed gate+up MXFP4 4096x2048, 3 tokens", GGML_TYPE_MXFP4, 4096, 2048, true, 3, true,
       false, 0},
      {"shared gate+up Q5_K 4096x2048, 1 token", GGML_TYPE_Q5_K, 4096, 2048, false, 1, true, false,
       59.7},
      {"shared down Q6_K 2048x4096, 1 token", GGML_TYPE_Q6_K, 2048, 4096, false, 1, false, false,
       35.9},
      {"shared gate+up Q5_K 4096x2048, 4 tokens", GGML_TYPE_Q5_K, 4096, 2048, false, 4, true, false,
       67.4},
      {"dense Q8_0 1024x32768, 1 token", GGML_TYPE_Q8_0, 1024, 32768, false, 1, false, false,
       172.9},
      {"dense Q8_0 8192x4096, 1 token", GGML_TYPE_Q8_0, 8192, 4096, false, 1, false, false, 159.8},
      {"dense Q8_0 4096x1024, 1 token", GGML_TYPE_Q8_0, 4096, 1024, false, 1, false, false, 26.9},
      {"dense Q8_0 4096x512, 1 token", GGML_TYPE_Q8_0, 4096, 512, false, 1, false, false, 13.6},
      {"dense Q8_0 4096x256, 1 token", GGML_TYPE_Q8_0, 4096, 256, false, 1, false, false, 8.7},
      {"dense Q8_0 1024x8192, 1 token", GGML_TYPE_Q8_0, 1024, 8192, false, 1, false, false, 43.9},
      {"dense Q5_K 4096x1024, 1 token", GGML_TYPE_Q5_K, 4096, 1024, false, 1, false, false, 19.1},
      {"head Q4_K 4096x129280, 1 token", GGML_TYPE_Q4_K, 4096, 129280, false, 1, false, false,
       1540.4},
      {"head Q4_K 4096x129280, 4 tokens", GGML_TYPE_Q4_K, 4096, 129280, false, 4, false, false,
       1580.8},
      {"dense Q8_0 1024x32768, 4 tokens", GGML_TYPE_Q8_0, 1024, 32768, false, 4, false, false,
       177.2},
      {"dense Q8_0 8192x4096, 4 tokens", GGML_TYPE_Q8_0, 8192, 4096, false, 4, false, false, 167.0},
      // Qwen3.8 Flash Next's UD-IQ3_XXS GGUF at decode (docs/experiments/qwen38-gguf/).
      {"qwen dense Q6_K 2560x10240, 1 token", GGML_TYPE_Q6_K, 2560, 10240, false, 1, false, false,
       0},
      {"qwen dense Q6_K 2560x6144, 1 token", GGML_TYPE_Q6_K, 2560, 6144, false, 1, false, false, 0},
      {"qwen dense Q6_K 6144x2560, 1 token", GGML_TYPE_Q6_K, 6144, 2560, false, 1, false, false, 0},
      {"qwen dense Q6_K 2560x12288, 1 token", GGML_TYPE_Q6_K, 2560, 12288, false, 1, false, false,
       0},
      {"qwen shared gate+up Q6_K 2560x640, 1 token", GGML_TYPE_Q6_K, 2560, 640, false, 1, true,
       false, 0},
      {"qwen head Q6_K 2560x248320, 1 token", GGML_TYPE_Q6_K, 2560, 248320, false, 1, false, false,
       0},
      {"qwen dense Q8_0 10240x320, 1 token", GGML_TYPE_Q8_0, 10240, 320, false, 1, false, false, 0},
      {"qwen dense Q8_0 320x10240, 1 token", GGML_TYPE_Q8_0, 320, 10240, false, 1, false, false, 0},
      {"qwen dense Q8_0 640x2560, 1 token", GGML_TYPE_Q8_0, 640, 2560, false, 1, false, false, 0},
      {"qwen routed gate+up IQ2_S 2560x640, 1 token", GGML_TYPE_IQ2_S, 2560, 640, true, 1, true,
       false, 0},
      {"qwen routed down IQ4_NL 640x2560, 1 token", GGML_TYPE_IQ4_NL, 640, 2560, true, 1, false,
       true, 0},
  };
  std::mt19937 rng(42);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::uniform_int_distribution<int> byte(0, 255);
  cudaStream_t stream = nullptr;
  if (!Ok(cudaStreamCreate(&stream), "stream")) {
    return 1;
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (!Ok(cudaEventCreate(&start), "event") || !Ok(cudaEventCreate(&stop), "event")) {
    return 1;
  }
  std::println("launches {}, routed overlap {:.2f}", launches, overlap);
  if (vmm && (!Ok(cudaFree(nullptr), "context") || !Ok(cuInit(0), "driver"))) {
    return 1;
  }
  std::println("weights in {}", vmm ? "device VMM, a 2 MiB backing each 2 MiB" : "cudaMalloc");
  for (const Case& c : cases) {
    if (!only.empty() && !c.name.contains(only)) {
      continue;
    }
    const std::int64_t row = RowBytes(c.type, c.k);
    const std::int64_t matrix = row * c.n;
    const int copies = c.routed ? kExperts : kDenseCopies;
    const std::size_t weight_bytes = U(matrix) * U(copies);
    void* w = nullptr;
    void* g = nullptr;
    const auto alloc = [&](void** p) {
      return vmm ? VmmAlloc(p, weight_bytes) : Ok(cudaMalloc(p, weight_bytes), "weights");
    };
    if (!alloc(&w) || (c.glu && !alloc(&g))) {
      return 1;
    }
    // Random weights: every block's scale is some value; the kernels read
    // whatever bytes are there.
    {
      std::vector<unsigned char> host(1U << 24U);
      for (unsigned char& b : host) {
        b = static_cast<unsigned char>(byte(rng));
      }
      for (std::size_t at = 0; at < weight_bytes; at += host.size()) {
        const std::size_t n = std::min(host.size(), weight_bytes - at);
        if (!Ok(cudaMemcpy(static_cast<char*>(w) + at, host.data(), n, cudaMemcpyHostToDevice),
                "fill") ||
            (g != nullptr &&
             !Ok(cudaMemcpy(static_cast<char*>(g) + at, host.data(), n, cudaMemcpyHostToDevice),
                 "fill"))) {
          return 1;
        }
      }
    }
    const int k_padded = (c.k + 511) / 512 * 512;
    const int y_row = k_padded / 32;  // Q8_1 blocks a row
    const int rows = c.per_slot ? kUsed * c.tokens : c.tokens;
    void* y = nullptr;
    float* dst = nullptr;
    std::int32_t* ids = nullptr;
    const std::size_t y_bytes = U(y_row) * U(kQ8Block) * U(rows);
    const int slots = c.routed ? kUsed : 1;
    if (!Ok(cudaMalloc(&y, y_bytes), "y") ||
        !Ok(cudaMalloc(reinterpret_cast<void**>(&dst),
                       sizeof(float) * U(c.n) * U(slots) * U(c.tokens)),
            "dst") ||
        !Ok(cudaMalloc(reinterpret_cast<void**>(&ids),
                       sizeof(std::int32_t) * U(kUsed) * U(c.tokens) * U(launches)),
            "ids")) {
      return 1;
    }
    {
      std::vector<unsigned char> host(y_bytes);
      for (std::size_t i = 0; i < host.size(); ++i) {
        // Small scales (half 0x2000-ish), small quants: no infinities.
        int value = byte(rng) % 16;
        if (i % kQ8Block < 4) {
          value = i % 2 == 0 ? 0 : 0x20;
        }
        host[i] = static_cast<unsigned char>(value);
      }
      if (!Ok(cudaMemcpy(y, host.data(), host.size(), cudaMemcpyHostToDevice), "y")) {
        return 1;
      }
      // Each launch's routing: distinct experts per token, a later token
      // taking `overlap` of its experts from the token before.
      std::vector<std::int32_t> all;
      std::uniform_int_distribution<int> expert(0, kExperts - 1);
      std::uniform_real_distribution<double> coin(0.0, 1.0);
      std::int64_t distinct = 0;
      for (int l = 0; l < launches; ++l) {
        std::vector<std::int32_t> prev;
        std::vector<std::int32_t> seen;
        for (int t = 0; t < c.tokens; ++t) {
          std::vector<std::int32_t> mine;
          while (static_cast<int>(mine.size()) < kUsed) {
            std::int32_t e = expert(rng);
            if (!prev.empty() && coin(rng) < overlap) {
              e = prev[static_cast<std::size_t>(mine.size())];
            }
            if (std::ranges::find(mine, e) == mine.end()) {
              mine.push_back(e);
            }
          }
          for (const std::int32_t e : mine) {
            if (std::ranges::find(seen, e) == seen.end()) {
              seen.push_back(e);
            }
          }
          all.insert(all.end(), mine.begin(), mine.end());
          prev = mine;
        }
        distinct += static_cast<std::int64_t>(seen.size());
      }
      if (!Ok(cudaMemcpy(ids, all.data(), all.size() * sizeof(std::int32_t),
                         cudaMemcpyHostToDevice),
              "ids")) {
        return 1;
      }
      const double per_launch_matrices = c.routed ? static_cast<double>(distinct) / launches : 1.0;
      const double bytes = per_launch_matrices * static_cast<double>(matrix) * (c.glu ? 2.0 : 1.0);
      std::println("{} (upstream {:.1f} us; {:.1f} distinct matrices a launch, {:.1f} MB)", c.name,
                   c.upstream_us, per_launch_matrices, bytes / 1e6);
      for (int v = -1; v < kg::VecQVariants(); ++v) {
        const auto describe = [&](int l) {
          kg::VecQDesc d;
          const std::int64_t blocks_row = row / static_cast<std::int64_t>(ggml_type_size(c.type));
          d.w = static_cast<const char*>(w) + (c.routed ? 0 : (l % copies) * matrix);
          d.g = g == nullptr ? nullptr
                             : static_cast<const char*>(g) + (c.routed ? 0 : (l % copies) * matrix);
          d.y = y;
          d.ids = c.routed ? ids + (static_cast<std::ptrdiff_t>(l) * kUsed * c.tokens) : nullptr;
          d.dst = dst;
          d.ncols_x = c.k;
          d.nrows = c.n;
          d.stride_row = static_cast<int>(blocks_row);
          d.stride_expert = static_cast<int>(blocks_row * c.n);
          d.y_token = c.per_slot ? y_row * kUsed : y_row;
          d.y_slot = c.per_slot ? y_row : 0;
          d.ids_stride = kUsed;
          d.used = c.routed ? kUsed : 1;
          d.tokens = c.tokens;
          d.dst_token = c.routed ? c.n * kUsed : c.n;
          d.dst_slot = c.routed ? c.n : 0;
          d.glu = c.glu ? 2 : 0;
          d.limit = 10.0f;
          return d;
        };
        if (!kg::LaunchVecQ(c.type, describe(0), v, stream)) {
          continue;
        }
        if (!Ok(cudaStreamSynchronize(stream), "warm-up")) {
          return 1;
        }
        if (!Ok(cudaEventRecord(start, stream), "record")) {
          return 1;
        }
        for (int l = 0; l < launches; ++l) {
          kg::LaunchVecQ(c.type, describe(l), v, stream);
        }
        if (!Ok(cudaEventRecord(stop, stream), "record") ||
            !Ok(cudaEventSynchronize(stop), "sync")) {
          return 1;
        }
        float ms = 0;
        if (!Ok(cudaEventElapsedTime(&ms, start, stop), "elapsed")) {
          return 1;
        }
        const double us = 1000.0 * ms / launches;
        std::println(
            "  {:<16} {:8.1f} us  {:6.1f} GB/s{}", kg::VecQVariantName(v), us, bytes / (us * 1e3),
            c.upstream_us > 0 ? std::format("  {:.2f}x upstream", c.upstream_us / us) : "");
      }
    }
    if (!vmm) {  // the VMM ranges live until the process exits
      cudaFree(w);
      cudaFree(g);
    }
    cudaFree(y);
    cudaFree(dst);
    cudaFree(ids);
  }
  return 0;
}
