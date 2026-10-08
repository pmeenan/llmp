// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_BENCHMARK_GEMMA_ATTENTION_OWNER_METADATA_H_
#define LLMP_BENCHMARK_GEMMA_ATTENTION_OWNER_METADATA_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

#include "kernels/ggml/fattn_owner.h"

namespace attention_owner_metadata {
namespace kg = llmp::kernels::ggml;

// These addresses are never dereferenced. Descriptor admission is a CPU-only
// operation; the manual GPU proof separately maps and holds real operands.
struct Fixture {
  ggml_tensor q{}, mask{}, output{};
  std::array<ggml_tensor, 4> k{}, v{};
  kg::FlashAttnOwners inputs;
  static void Tensor(ggml_tensor& t, ggml_type type, std::array<std::int64_t, 4> ne,
                     std::array<std::size_t, 4> nb, std::uintptr_t address) {
    t = {};
    t.type = type;
    std::copy(ne.begin(), ne.end(), t.ne);
    std::copy(nb.begin(), nb.end(), t.nb);
    t.data = reinterpret_cast<void*>(address);
  }
  Fixture(int d = 256, int heads = 32) {
    const auto dimension = static_cast<std::size_t>(d);
    const auto head_count = static_cast<std::size_t>(heads);
    const std::size_t kvheads = head_count / (d == 256 ? 2U : 8U);
    const std::size_t qrow = dimension * head_count * 4, kvrow = dimension * kvheads * 2;
    Tensor(q, GGML_TYPE_F32, {d, 1, heads, 4}, {4, qrow, dimension * 4, qrow}, 0x10000000);
    Tensor(mask, GGML_TYPE_F16, {256, 32, 1, 4}, {2, 512, 16384, 16384}, 0x11000000);
    Tensor(output, GGML_TYPE_F32, {d, heads, 1, 4}, {4, dimension * 4, qrow, qrow}, 0x12000000);
    inputs.q = &q;
    inputs.mask = &mask;
    inputs.output = &output;
    for (std::size_t owner = 0; owner < 4; ++owner) {
      const auto base = 0x20000000 + owner * 0x10000000;
      Tensor(k[owner], GGML_TYPE_F16, {d, 256, std::int64_t(kvheads), 1},
             {2, kvrow, dimension * 2, kvrow * 256}, base);
      Tensor(v[owner], GGML_TYPE_F16, {d, 256, std::int64_t(kvheads), 1},
             {2, kvrow, dimension * 2, kvrow * 256}, base + 0x01000000 + owner * 256);
      inputs.k[owner] = &k[owner];
      inputs.v[owner] = &v[owner];
    }
  }
};

inline bool Controls() {
  int count = 0;
  bool ok = true;
  const auto check = [&](Fixture& f, bool expected, const char* name) {
    const bool actual = kg::CheckFlashAttnOwners(f.inputs).has_value();
    ++count;
    if (actual != expected) {
      std::cerr << "OWNER_METADATA_FAILED " << name << '\n';
      ok = false;
    }
  };
  const auto reject = [&](const char* name, auto change) {
    Fixture f;
    change(f);
    check(f, false, name);
  };
  for (int d : {256, 512})
    for (int heads : {16, 32}) {
      Fixture f(d, heads);
      check(f, true, "supported profile/head");
    }
  reject("missing owner", [](auto& f) { f.inputs.k[3] = nullptr; });
  reject("wrong cache type", [](auto& f) { f.v[2].type = GGML_TYPE_F32; });
  reject("query rows", [](auto& f) { f.q.ne[1] = 2; });
  reject("query streams", [](auto& f) { f.q.ne[3] = 3; });
  reject("cache streams", [](auto& f) { f.k[1].ne[3] = 4; });
  reject("cache width", [](auto& f) { f.k[0].ne[1] = 255; });
  reject("head grouping", [](auto& f) { f.k[0].ne[2] = 8; });
  reject("query stride", [](auto& f) { f.q.nb[2] += 16; });
  reject("cache stride", [](auto& f) { f.v[1].nb[1] += 16; });
  reject("mask padded rows", [](auto& f) { f.mask.ne[1] = 4; });
  reject("mask broadcast", [](auto& f) { f.mask.ne[3] = 1; });
  reject("output stride", [](auto& f) { f.output.nb[3] += 16; });
  reject("output aliases cache", [](auto& f) { f.output.data = f.k[0].data; });
  reject("crossed owners", [](auto& f) { f.v[1].data = f.k[0].data; });
  reject("unaligned address", [](auto& f) { f.k[0].data = reinterpret_cast<void*>(0x20000001); });
  reject("address overflow", [](auto& f) {
    f.output.data = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 15);
  });
  reject("self view", [](auto& f) { f.k[0].view_src = &f.k[0]; });
  {
    Fixture f;
    ggml_tensor parent = f.k[0];
    f.k[0].view_src = &parent;
    check(f, true, "bounded current parent");
    parent.data = reinterpret_cast<void*>(0x20000100);
    check(f, false, "stale parent address");
    parent.data = f.k[0].data;
    parent.ne[0] = std::numeric_limits<std::int64_t>::max();
    check(f, false, "hostile parent dimension");
  }
  {
    Fixture f;
    ggml_tensor parent = f.k[0];
    f.k[0].view_src = &parent;
    f.k[0].view_offs = std::numeric_limits<std::size_t>::max();
    check(f, false, "view offset overflow");
  }
  {
    Fixture f;
    f.inputs.v[0] = f.inputs.k[0];
    check(f, true, "read-only same-owner K-as-V");
  }
  reject("unsupported head", [](auto& f) { f.q.ne[0] = 128; });
  std::cout << "OWNER_METADATA cases=" << count << " device_open=0 launches=0 pass=" << ok << '\n';
  return ok;
}
}  // namespace attention_owner_metadata
#endif  // LLMP_BENCHMARK_GEMMA_ATTENTION_OWNER_METADATA_H_
