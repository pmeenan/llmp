// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "exl3_common.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "model/qwen2_exl3.h"

namespace llmp::benchmarks {
namespace {

using Status = std::expected<void, std::string>;
namespace model = llmp::model;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

std::uint64_t Round(std::uint64_t bytes, std::uint64_t to) { return (bytes + to - 1) / to * to; }

constexpr std::string_view kHeldOut =
    "6dd8da897821f5615e7796f4d882795faf306a941f050e2eb68b619096e3fb0c";

}  // namespace

std::string Hex(std::span<const std::byte> bytes) {
  llmp::base::Sha256 hash;
  hash.Update(bytes);
  return llmp::base::ToHex(hash.Finish());
}

std::string HexFile(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  return Hex(std::as_bytes(std::span(bytes)));
}

// A .npy file (format 1.0) of little-endian values.
std::expected<void, std::string> WriteNpy(const std::filesystem::path& path, std::string_view descr,
                                          const std::vector<std::int64_t>& shape,
                                          std::span<const std::byte> data) {
  std::string dims;
  for (const std::int64_t d : shape) {
    dims += std::format("{}, ", d);
  }
  if (shape.size() > 1) {
    dims.resize(dims.size() - 2);
  } else if (shape.size() == 1) {
    dims.resize(dims.size() - 1);
  }
  std::string header =
      std::format("{{'descr': '{}', 'fortran_order': False, 'shape': ({}), }}", descr, dims);
  const std::size_t total = 10 + header.size() + 1;
  header.append(Round(total, 64) - total, ' ');
  header += '\n';
  std::ofstream file(path, std::ios::binary);
  const std::array<char, 8> magic = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
  file.write(magic.data(), magic.size());
  const auto length = static_cast<std::uint16_t>(header.size());
  file.put(static_cast<char>(length & 0xFFU));
  file.put(static_cast<char>(length >> 8U));
  file << header;
  file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  return file.good() ? Status() : Error("cannot write " + path.string());
}

std::expected<std::vector<std::int32_t>, std::string> LoadIds(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (bytes.size() != std::size_t{1040} * 8 || Hex(std::as_bytes(std::span(bytes))) != kHeldOut) {
    return Error("not the declared held-out IDs");
  }
  std::vector<std::int32_t> ids;
  for (std::size_t i = 0; i < 1040; ++i) {
    std::uint64_t id = 0;
    for (int b = 7; b >= 0; --b) {
      id = (id << 8U) | static_cast<unsigned char>(bytes[(i * 8) + static_cast<std::size_t>(b)]);
    }
    ids.push_back(static_cast<std::int32_t>(id));
  }
  return ids;
}

// The launch table from PLAN.txt, checked against the artifact's linears.
std::expected<model::Exl3LaunchTable, std::string> LoadTable(const std::filesystem::path& path,
                                                             const model::Exl3Binding& binding) {
  std::ifstream in(path);
  if (!in) {
    return Error(std::format("cannot read {}", path.string()));
  }
  std::map<std::string, std::array<int, 3>> linears;  // name -> k, n, bits
  const auto note = [&](const model::Exl3LinearBinding& l) {
    linears[l.name] = {l.k, l.n, l.bits};
  };
  note(binding.lm_head);
  for (const auto& layer : binding.layers) {
    for (const auto* l :
         {&layer.q, &layer.k, &layer.v, &layer.o, &layer.gate, &layer.up, &layer.down}) {
      note(*l);
    }
  }
  std::string cache;
  std::vector<std::tuple<std::string, int, model::Exl3LinearPlan, std::string>> entries;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream words(line);
    std::string kind;
    words >> kind;
    if (kind == "#") {
      std::string key;
      words >> key;
      if (key == "cache") {
        words >> cache;
      }
      continue;
    }
    if (kind == "linear") {
      std::string name;
      int k = 0;
      int n = 0;
      int bits = 0;
      words >> name >> k >> n >> bits;
      const auto found = linears.find(name);
      if (found == linears.end() || found->second != std::array<int, 3>{k, n, bits}) {
        return Error(std::format("the plan's linear {} is not the artifact's", name));
      }
      continue;
    }
    if (kind != "case") {
      if (!kind.empty()) {
        return Error(std::format("unknown line: {}", line));
      }
      continue;
    }
    std::string id;
    int rows = 0;
    std::string path_name;
    words >> id >> rows >> path_name;
    model::Exl3LinearPlan plan;
    std::string first;
    std::string second;
    if (path_name == "gemm") {
      plan.path = model::Exl3Path::kGemm;
      words >> first >> plan.shape >> plan.blocks;
    } else if (path_name == "gemv") {
      plan.path = model::Exl3Path::kGemv;
      words >> first >> plan.config >> plan.blocks;
    } else if (path_name == "multi") {
      plan.path = model::Exl3Path::kMulti;
      words >> first >> second >> plan.shape >> plan.blocks >> plan.concurrency;
    } else if (path_name == "recon" || path_name == "fused") {
      plan.path =
          path_name == "recon" ? model::Exl3Path::kReconstruct : model::Exl3Path::kReconstructFused;
      int slices = 0;
      words >> first >> slices;
      for (int s = 0; s < slices; ++s) {
        model::Exl3Pin pin;
        std::string gemm;
        words >> gemm >> pin.m >> pin.k >> pin.n >> pin.ldc;
        if (gemm != "HSH" && gemm != "HSS") {
          return Error(std::format("a pin of kind {} in: {}", gemm, line));
        }
        pin.f32 = gemm == "HSS";
        for (std::uint64_t& value : pin.config) {
          words >> value;
        }
        plan.pins.push_back(pin);
      }
    } else {
      return Error(std::format("unknown path in: {}", line));
    }
    if (words.fail()) {
      return Error(std::format("malformed: {}", line));
    }
    entries.emplace_back(first, rows, std::move(plan), second);
  }
  if (cache.size() != 64) {
    return Error("the plan names no frozen tuning cache (# cache SHA256)");
  }
  model::Exl3LaunchTable table(std::format("tuning cache {}; plan {}", cache, HexFile(path)));
  for (auto& [first, rows, plan, second] : entries) {
    if (auto added = table.Add(first, rows, std::move(plan), second); !added) {
      return std::unexpected(added.error());
    }
  }
  return table;
}

// BF16 to F32, exactly.
float Bf16ToFloat(std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

}  // namespace llmp::benchmarks
