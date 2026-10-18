// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "kernels/exl3/packed_tuning.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <limits>
#include <span>
namespace llmp::kernels::exl3 {
namespace {
struct Record {
  std::uint64_t key;
  int shape, threads, blocks, concurrency;
};
// Exact authenticated records from backend-proof-p0/results.json and
// backend-proof-p3/results.json (frozen linear sweep). Key
// arithmetic is the locked coop_autotune v4 contract, reproduced by that
// experiment's decode_tuning.py; unknown keys are refused, never tuned.
// tune-40-gemvoff: SHA-256 c6c01843042cde6ee97624ca83d32e694be2b3839fda8eb53b7d686c98d5fdd0
// P3 frozen linear sweep adds 6 records; every P0 record remains unchanged.
// P3 cache SHA-256 41bcd3b1937442965105ff6abb99184725468a3f85f7a8e354579dbe05af05d6
constexpr std::array<Record, 19> k40G{{
    {0xd0fc81a31552c98dULL, 2, 512, 14, 1}, {0xd63578982608a88dULL, 2, 512, 4, 1},
    {0x3454a980d3c6b624ULL, 2, 512, 14, 1}, {0x57386e01d1a9a7bdULL, 3, 512, 24, 2},
    {0x53d463e0a4022ca4ULL, 2, 512, 40, 1}, {0x4acbb86e36d304a9ULL, 2, 512, 48, 1},
    {0x4d0ebe1d454f4b20ULL, 3, 512, 24, 2}, {0x7b5285ddeeeb43bbULL, 1, 256, 48, 1},
    {0x0a80944c677f95a4ULL, 2, 512, 48, 1}, {0x0adb88759eae926fULL, 2, 512, 20, 1},
    {0xfb30a3966c8cf56fULL, 2, 512, 4, 1},  {0x6e33a8535d22716eULL, 2, 512, 22, 1},
    {0x04083b602e249aeeULL, 2, 512, 48, 1}, {0x86bd27f7a9a72c95ULL, 2, 512, 22, 1},
    {0x8bf61eecba5d0b95ULL, 2, 512, 4, 1},  {0xff44afd6051aee8cULL, 2, 512, 28, 1},
    {0x96c3d290271d3165ULL, 3, 512, 24, 2}, {0xdfc4f57634df780cULL, 2, 512, 34, 1},
    {0x64e1ba5f4b668f41ULL, 2, 512, 48, 1},
}};
// tune-45-gemvoff: SHA-256 e5de5226aa03bf1238b4fbe589c2b76e230c13a5fc77c0d888df535eb13748db
// P3 frozen linear sweep adds 11 records; every P0 record remains unchanged.
// P3 cache SHA-256 5c2d5f6238b417993ffe04b879a8780f4835c6914a25e32df05732cce627fc62
constexpr std::array<Record, 35> k45G{{
    {0x70b64a8fabaace97ULL, 2, 512, 14, 1}, {0x75ef4184bc60ad97ULL, 2, 512, 4, 1},
    {0xd40e6a6d6a1ead96ULL, 2, 512, 14, 1}, {0xf1d65f48332c4012ULL, 3, 512, 24, 2},
    {0x1cc4dceed059dfbbULL, 2, 512, 48, 1}, {0x609d526cbe92636aULL, 2, 512, 28, 1},
    {0x65d64961cf48426aULL, 2, 512, 4, 1},  {0xfd45228f001e693bULL, 2, 512, 14, 1},
    {0x57386e01d1a9a7bdULL, 3, 512, 24, 2}, {0x53d463e0a4022ca4ULL, 2, 512, 48, 1},
    {0x4acbb86e36d304a9ULL, 2, 512, 48, 1}, {0xaaafdf8adf560d85ULL, 2, 512, 28, 1},
    {0x9b04faabad347085ULL, 2, 512, 4, 1},  {0xce79e766c6ca79fcULL, 2, 512, 28, 1},
    {0x58d44e5d5139e2fbULL, 3, 512, 24, 2}, {0x8d954429d1cf048dULL, 2, 512, 40, 1},
    {0x5b18d8fabf40c7a4ULL, 2, 512, 20, 1}, {0x4b6df41b8d1f2aa4ULL, 2, 512, 4, 1},
    {0xf7c0b11d00ccdb0dULL, 2, 512, 28, 1}, {0x4d0ebe1d454f4b20ULL, 3, 512, 22, 2},
    {0x7b5285ddeeeb43bbULL, 2, 512, 48, 1}, {0xd3710d5a93d748bbULL, 2, 512, 48, 1},
    {0x0a80944c677f95a4ULL, 2, 512, 48, 1}, {0x04083b602e249aeeULL, 2, 512, 48, 1},
    {0xe6e8d0e268ffb17fULL, 2, 512, 28, 1}, {0xec21c7d779b5907fULL, 2, 512, 4, 1},
    {0x4a40f0c02773907eULL, 2, 512, 28, 1}, {0x9b855e86deb27d5aULL, 3, 512, 24, 2},
    {0x93860e83c43755c3ULL, 2, 512, 40, 1}, {0xd6cfd8bf7be74652ULL, 2, 512, 20, 1},
    {0xdc08cfb48c9d2552ULL, 2, 512, 4, 1},  {0xb305c8e39472cc43ULL, 2, 512, 28, 1},
    {0x96c3d290271d3165ULL, 3, 512, 24, 2}, {0xdfc4f57634df780cULL, 2, 512, 38, 1},
    {0x64e1ba5f4b668f41ULL, 2, 512, 48, 1},
}};
// tune-40: SHA-256 4760513062b614b29d254a8917d7fb0e7aa539e7b86223cee231cd8b351ff565
// P3 frozen linear sweep adds 2 records; every P0 record remains unchanged.
// P3 cache SHA-256 6977506009d453fd715bc2080dcd54278bc79d49f913ab6c3ba39333a295cd61
constexpr std::array<Record, 11> k40O{{
    {0xd0fc81a31552c98dULL, 2, 512, 14, 1},
    {0xd63578982608a88dULL, 2, 512, 4, 1},
    {0x3454a980d3c6b624ULL, 2, 512, 14, 1},
    {0x57386e01d1a9a7bdULL, 3, 512, 24, 2},
    {0x53d463e0a4022ca4ULL, 2, 512, 40, 1},
    {0x4acbb86e36d304a9ULL, 2, 512, 48, 1},
    {0x4d0ebe1d454f4b20ULL, 3, 512, 24, 2},
    {0x7b5285ddeeeb43bbULL, 1, 256, 48, 1},
    {0x0a80944c677f95a4ULL, 2, 512, 48, 1},
    {0x96c3d290271d3165ULL, 2, 512, 24, 2},
    {0x64e1ba5f4b668f41ULL, 1, 256, 48, 1},
}};
// tune-45: SHA-256 d85a85ae1029cf1d6e4b87a444aa505ca4c2cee39f3a0aa551f7823fa4ce6ae1
// P3 frozen linear sweep adds 10 records; every P0 record remains unchanged.
// P3 cache SHA-256 64cf8ecd8da2fde214d64d0163da9ca75dac004fe10ffd80a98705cac0d7ccb5
constexpr std::array<Record, 33> k45O{{
    {0x70b64a8fabaace97ULL, 2, 512, 14, 1}, {0x75ef4184bc60ad97ULL, 2, 512, 4, 1},
    {0xd40e6a6d6a1ead96ULL, 2, 512, 14, 1}, {0xf1d65f48332c4012ULL, 3, 512, 24, 2},
    {0x1cc4dceed059dfbbULL, 2, 512, 48, 1}, {0x609d526cbe92636aULL, 2, 512, 28, 1},
    {0x65d64961cf48426aULL, 2, 512, 4, 1},  {0xfd45228f001e693bULL, 2, 512, 14, 1},
    {0x57386e01d1a9a7bdULL, 3, 512, 24, 2}, {0x53d463e0a4022ca4ULL, 2, 512, 48, 1},
    {0x4acbb86e36d304a9ULL, 2, 512, 48, 1}, {0xaaafdf8adf560d85ULL, 2, 512, 28, 1},
    {0x9b04faabad347085ULL, 2, 512, 4, 1},  {0xce79e766c6ca79fcULL, 2, 512, 28, 1},
    {0x58d44e5d5139e2fbULL, 3, 512, 24, 2}, {0x8d954429d1cf048dULL, 2, 512, 40, 1},
    {0x5b18d8fabf40c7a4ULL, 2, 512, 20, 1}, {0x4b6df41b8d1f2aa4ULL, 2, 512, 4, 1},
    {0xf7c0b11d00ccdb0dULL, 2, 512, 28, 1}, {0x4d0ebe1d454f4b20ULL, 3, 512, 22, 2},
    {0x7b5285ddeeeb43bbULL, 2, 512, 48, 1}, {0xd3710d5a93d748bbULL, 2, 512, 48, 1},
    {0x0a80944c677f95a4ULL, 2, 512, 48, 1}, {0xe6e8d0e268ffb17fULL, 2, 512, 28, 1},
    {0xec21c7d779b5907fULL, 2, 512, 4, 1},  {0x4a40f0c02773907eULL, 2, 512, 28, 1},
    {0x9b855e86deb27d5aULL, 3, 512, 24, 2}, {0x93860e83c43755c3ULL, 2, 512, 48, 1},
    {0xd6cfd8bf7be74652ULL, 2, 512, 20, 1}, {0xdc08cfb48c9d2552ULL, 2, 512, 4, 1},
    {0xb305c8e39472cc43ULL, 2, 512, 28, 1}, {0x96c3d290271d3165ULL, 3, 512, 24, 2},
    {0x64e1ba5f4b668f41ULL, 2, 512, 48, 1},
}};
std::span<const Record> Records(FrozenCache cache) {
  switch (cache) {
    case FrozenCache::k40G:
      return k40G;
    case FrozenCache::k45G:
      return k45G;
    case FrozenCache::k40O:
      return k40O;
    case FrozenCache::k45O:
      return k45O;
  }
  return {};
}
std::uint64_t Mix(std::uint64_t h, std::uint64_t v) { return (h ^ v) * 1099511628211ULL; }
std::expected<Record, std::string> Lookup(FrozenCache cache, std::uint32_t arch, int sms, int rows,
                                          int k, int n, int bits, Output output, bool multi) {
  if (arch != 121 || sms != 48 || rows < 1 || rows > 144 || k < 128 || n < 128 ||
      k > kMaxDimension || n > kMaxDimension || k % 128 || n % 128 || !RateCompiled(bits))
    return std::unexpected("packed tuning does not cover this device or shape");
  const auto bucket =
      std::min(std::bit_ceil(static_cast<unsigned>(std::max(rows, multi ? 1 : 2))), 16U);
  std::uint64_t key = 1469598103934665603ULL;
  const std::array<std::uint64_t, 10> values{0,
                                             bucket,
                                             static_cast<unsigned>(k),
                                             static_cast<unsigned>(n),
                                             static_cast<unsigned>(bits),
                                             output == Output::kF32 ? 1U : 0U,
                                             0,
                                             5,
                                             48,
                                             1};
  for (const auto value : values) key = Mix(key, value);
  if (multi) key = Mix(Mix(key, 1), 2);
  key = Mix(key, 4);  // tuner version
  const auto records = Records(cache);
  const auto found = std::ranges::find(records, key, &Record::key);
  if (found == records.end()) return std::unexpected("no authenticated packed tuning record");
  if (found->shape < 1 || found->shape > kShapes ||
      found->threads != kBlockDim[static_cast<std::size_t>(found->shape)] || found->blocks < 1 ||
      found->concurrency != (multi ? 2 : 1))
    return std::unexpected("authenticated packed tuning record violates the kernel contract");
  return *found;
}
}  // namespace
std::expected<GemmPlan, std::string> DefaultGemmPlan(const PackedCapacity& capacity, int rows,
                                                     int k, int n, int bits, Output output,
                                                     bool multi) {
  constexpr std::array rates{4, 5, 6, 8};
  const auto rate = std::ranges::find(rates, bits);
  if (capacity.arch < 80 || capacity.sms < 1 ||
      (output != Output::kF16 && output != Output::kF32) || rate == rates.end() || rows < 1 ||
      rows > kMaxRows || k < 128 || n < 128 || k > kMaxDimension || n > kMaxDimension || k % 128 ||
      n % 128 ||
      std::uint64_t(rows) * static_cast<std::uint64_t>(std::max(k, n)) * (multi ? 2U : 1U) >
          INT32_MAX ||
      std::uint64_t(k / 16) * static_cast<std::uint64_t>(n / 16) * 16 *
              static_cast<unsigned>(bits) >
          INT32_MAX)
    return std::unexpected("no default packed EXL3 plan within kernel indexing bounds");
  const auto& counts =
      (multi ? capacity.multi : capacity.gemm)[output == Output::kF32 ? 1 : 0]
                                              [static_cast<std::size_t>(rate - rates.begin())];
  GemmPlan best{};
  std::uint64_t work = std::numeric_limits<std::uint64_t>::max();
  for (int shape = 1; shape <= kShapes; ++shape) {
    const auto at = static_cast<std::size_t>(shape);
    if (k % kTileK[at] || n % kTileN[at] || counts[at - 1] <= 0) continue;
    const int slices = (k / kTileK[at]) * (n / kTileN[at]);
    // Multi-GEMM concurrency1 permits the complete device occupancy.
    const int blocks = std::min({counts[at - 1], slices, INT32_MAX / slices});
    if (blocks <= 0) continue;
    const auto serial = std::uint64_t((slices + blocks - 1) / blocks) *
                        static_cast<unsigned>(kTileK[at]) * static_cast<unsigned>(kTileN[at]);
    if (serial < work) {
      work = serial;
      best = {shape, blocks};
    }
  }
  if (best.shape == 0) return std::unexpected("no resident packed EXL3 kernel divides this shape");
  return best;
}

std::expected<GemmPlan, std::string> FrozenGemmPlan(FrozenCache cache, std::uint32_t arch, int sms,
                                                    int rows, int k, int n, int bits,
                                                    Output output) {
  const auto r = Lookup(cache, arch, sms, rows, k, n, bits, output, false);
  if (!r) return std::unexpected(r.error());
  return GemmPlan{r->shape, r->blocks};
}
std::expected<MultiGemmPlan, std::string> FrozenMultiGemmPlan(FrozenCache cache, std::uint32_t arch,
                                                              int sms, int rows, int k, int n,
                                                              int bits, Output output) {
  const auto r = Lookup(cache, arch, sms, rows, k, n, bits, output, true);
  if (!r) return std::unexpected(r.error());
  return MultiGemmPlan{r->shape, r->blocks, r->concurrency};
}
}  // namespace llmp::kernels::exl3
