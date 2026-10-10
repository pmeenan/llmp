// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include "kernels/reference/decode_exl3.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "format/encoding.h"
#include "format/tensor_format.h"
#include "kernels/reference/decode.h"

namespace llmp::kernels::reference {
namespace {

using format::Exl3Codebook;

// codebook.cuh's lop3.b32 with immediate 0x6A, (x & b) ^ c: each half
// keeps its sign and low 12 bits under exponent bits 0b011xx, an F16 in
// ±[2^-3, 2).
std::uint32_t Lop3(std::uint32_t x) { return (x & 0x8FFF8FFFU) ^ 0x3B603B60U; }

// The halves of x added as __hadd adds them: rounded once to F16. Both are
// in ±[2^-3, 2) with 11-bit significands, so their sum is a multiple of
// 2^-13 below 4 in magnitude, exact in F32.
std::uint16_t HalfSum(std::uint32_t x) {
  const float sum = F16ToF32(static_cast<std::uint16_t>(x & 0xFFFFU)) +
                    F16ToF32(static_cast<std::uint16_t>(x >> 16U));
  return F32ToF16(sum);
}

}  // namespace

std::uint16_t Exl3CodebookValue(format::Exl3Codebook codebook, std::uint16_t state) {
  std::uint32_t x = state;
  switch (codebook) {
    case Exl3Codebook::k3Inst:  // decode_3inst<0>
      x = (x * 89226354U) + 64248484U;
      return HalfSum(Lop3(x));
    case Exl3Codebook::kMcg:  // decode_3inst<1>
      x *= 0xCBAC1FEDU;
      return HalfSum(Lop3(x));
    case Exl3Codebook::kMul1: {  // decode_3inst<2>
      x *= 0x83DCD12DU;
      // __dp4a(x, 0x01010101, 0x6400): the unsigned byte sum plus 0x6400,
      // whose low 16 bits are an F16 in [1024, 2047].
      const std::uint32_t sum =
          (x & 0xFFU) + ((x >> 8U) & 0xFFU) + ((x >> 16U) & 0xFFU) + (x >> 24U) + 0x6400U;
      // __hfma(h, 0x1EEE, 0xC931), rounded once: the integer h times
      // 0x1EEE (1774 · 2^-18) is exact in 22 bits, and adding 0xC931
      // (-10.3828125) keeps it a multiple of 2^-18 below 16, exact in F32.
      const float h = F16ToF32(static_cast<std::uint16_t>(sum & 0xFFFFU));
      return F32ToF16(std::fma(h, F16ToF32(0x1EEEU), F16ToF32(0xC931U)));
    }
    case Exl3Codebook::kNone:
      break;
  }
  return 0x7E00U;  // NaN: no codebook (format::Validate refuses it)
}

namespace exl3 {
namespace {

constexpr std::uint64_t kTile = 16;
constexpr std::uint64_t kPositions = kTile * kTile;

// A uint32 of the tile as the kernels load it (little-endian on both
// targets): halves 2w and 2w + 1 of the I16 trellis.
std::uint32_t Word(std::span<const std::byte> tile, std::uint32_t w) {
  std::uint32_t v = 0;
  std::memcpy(&v, tile.data() + (std::size_t{4} * w), sizeof(v));
  return v;
}

// Position t's 16-bit trellis state, as dq and dq4 extract it
// (exl3_dq.cuh): the tile's 256 · K bits form a circular stream, most
// significant bit of each uint32 first, and state t is the 16 bits that
// end at bit (t + 1) · K, wrapping to the stream's end for the first.
std::uint16_t State(std::span<const std::byte> tile, std::uint32_t bits, std::uint32_t t) {
  const std::uint32_t words = bits * 256 / 32;
  const std::uint32_t b0 = (t * bits) + bits - 16 + (256 * bits);  // the state's first bit
  const std::uint32_t b1 = b0 + 16;                                // past its last
  const std::uint32_t i0 = b0 / 32;
  const std::uint32_t i1 = (b1 - 1) / 32;  // may be i0
  const std::uint32_t s0 = ((i1 + 1) * 32) - b1;
  const std::uint64_t merged =
      (std::uint64_t{Word(tile, i0 % words)} << 32U) | Word(tile, i1 % words);
  return static_cast<std::uint16_t>((merged >> s0) & 0xFFFFU);
}

// Where reconstruct_tile (reconstruct.cu) stores each position of a 16 × 16
// tile, as row · 16 + column of the K × N weights: lane L of the warp
// decodes positions 8L to 8L + 7 (dq_dispatch at t_offset 8L) into four
// half2 registers, frag[0] = {(8L, 8L + 1), (8L + 2, 8L + 3)} and
// frag[1] = {(8L + 4, 8L + 5), (8L + 6, 8L + 7)}; each lane without bit 2
// pairs every register half with the same half of lane L + 4's
// (__shfl_down_sync by 4) and stores the pair at row (L % 4) · 2, + 1 for
// a high half, + 8 for the second register of a fragment, and columns
// 2c and 2c + 1 for c = L / 8, + 4 for frag[1].
constexpr std::array<std::uint8_t, kPositions> Placement() {
  std::array<std::uint8_t, kPositions> at{};
  for (int lane = 0; lane < 32; ++lane) {
    if ((lane & 4) != 0) {
      continue;  // lane - 4 stores its values
    }
    for (int reg = 0; reg < 4; ++reg) {  // frag[reg / 2][reg % 2]
      for (int high = 0; high < 2; ++high) {
        const int row = ((lane % 4) * 2) + ((reg % 2) * 8) + high;
        const int pair = (lane / 8) + ((reg / 2) * 4);
        const int position = (8 * lane) + (2 * reg) + high;
        at.at(static_cast<std::size_t>(position)) =
            static_cast<std::uint8_t>((row * 16) + (2 * pair));
        at.at(static_cast<std::size_t>(position + 32)) =
            static_cast<std::uint8_t>((row * 16) + (2 * pair) + 1);
      }
    }
  }
  return at;
}
constexpr std::array<std::uint8_t, kPositions> kPlacement = Placement();

// Every state's value in a codebook, computed once (Exl3CodebookValue is
// the definition; a table only saves recomputing it per weight).
using Table = std::array<std::uint16_t, 65536>;
Table Tabulate(Exl3Codebook codebook) {
  Table t{};
  for (std::size_t s = 0; s < t.size(); ++s) {
    t[s] = Exl3CodebookValue(codebook, static_cast<std::uint16_t>(s));
  }
  return t;
}
const Table& TableOf(Exl3Codebook codebook) {
  static const Table three_inst = Tabulate(Exl3Codebook::k3Inst);
  static const Table mcg = Tabulate(Exl3Codebook::kMcg);
  static const Table mul1 = Tabulate(Exl3Codebook::kMul1);
  static const Table none = Tabulate(Exl3Codebook::kNone);
  switch (codebook) {
    case Exl3Codebook::k3Inst:
      return three_inst;
    case Exl3Codebook::kMcg:
      return mcg;
    case Exl3Codebook::kMul1:
      return mul1;
    case Exl3Codebook::kNone:
      break;
  }
  return none;
}

// One tile's 256 values, F16 bits, row-major.
void Tile(std::span<const std::byte> tile, std::uint32_t bits, Exl3Codebook codebook,
          std::array<std::uint16_t, kPositions>& out) {
  const Table& values = TableOf(codebook);
  for (std::uint32_t t = 0; t < kPositions; ++t) {
    out[kPlacement[t]] = values[State(tile, bits, t)];
  }
}

// The tile at (kt, nt): [K / 16][N / 16][16 · k_bits] I16.
std::span<const std::byte> TileAt(const format::TensorFormat& f, const Matrix& m, std::uint64_t kt,
                                  std::uint64_t nt) {
  const std::uint64_t bytes = std::uint64_t{32} * f.params.k_bits;
  return m.codes.subspan(((kt * (f.shape.ne[1] / kTile)) + nt) * bytes, bytes);
}

// The natural-order 128-point Walsh-Hadamard transform, unscaled, as the
// kernels' butterflies compute it (hadamard_inner.cuh: a value's index is
// 4 · lane + i, lanes exchanged across the warp): H[i][j] =
// (-1)^popcount(i & j).
void Hadamard(double* v) {
  for (std::uint64_t h = 1; h < kHadamardBlock; h *= 2) {
    for (std::uint64_t i = 0; i < kHadamardBlock; i += 2 * h) {
      for (std::uint64_t j = i; j < i + h; ++j) {
        const double a = v[j];
        const double b = v[j + h];
        v[j] = a + b;
        v[j + h] = a - b;
      }
    }
  }
}

double Half(std::span<const std::byte> part, std::uint64_t i) {
  std::uint16_t bits = 0;
  std::memcpy(&bits, part.data() + (2 * i), sizeof(bits));
  return F16ToF32(bits);
}

}  // namespace

bool Transformable(const format::TensorFormat& format) {
  return format.shape.ne[0] % kHadamardBlock == 0 && format.shape.ne[1] % kHadamardBlock == 0;
}

void Rotated(const format::TensorFormat& f, const Matrix& m, std::uint64_t first,
             std::uint64_t count, std::span<std::uint16_t> out) {
  const std::uint64_t k = f.shape.ne[0];
  std::array<std::uint16_t, kPositions> tile{};
  for (std::uint64_t kt = 0; kt < k / kTile; ++kt) {
    for (std::uint64_t nt = first / kTile; nt < (first + count) / kTile; ++nt) {
      Tile(TileAt(f, m, kt, nt), f.params.k_bits, f.params.codebook, tile);
      for (std::uint64_t r = 0; r < kTile; ++r) {
        std::memcpy(&out[((kt * kTile + r) * count) + (nt * kTile) - first], &tile.at(r * kTile),
                    kTile * sizeof(std::uint16_t));
      }
    }
  }
}

// W = diag(suh) · H · Ŵ · H · diag(svh) / 128, H block-diagonal over 128
// (reconstruct_had_kernel's comment, and the packed and reconstruction
// paths' input and output transforms): each 128-column block of Ŵ is
// decoded, each of its rows transformed, then each requested column along
// K, in F64. For the 3inst and mcg codebooks every value of Ŵ is a multiple
// of 2^-13 below 4 in magnitude, so the transformed sums (29 bits) and their
// products with suh and svh (11 bits each) are exact, and W is rounded
// once, to F32.
void Rows(const format::TensorFormat& f, const Matrix& m, std::uint64_t first, std::uint64_t count,
          std::span<float> out) {
  const std::uint64_t k = f.shape.ne[0];
  const std::uint64_t end = first + count;
  std::vector<double> block(k * kHadamardBlock);
  std::vector<double> column(k);
  std::array<std::uint16_t, kPositions> tile{};
  constexpr std::uint64_t kTilesPerBlock = kHadamardBlock / kTile;
  for (std::uint64_t b = first / kHadamardBlock; b * kHadamardBlock < end; ++b) {
    for (std::uint64_t kt = 0; kt < k / kTile; ++kt) {
      for (std::uint64_t j = 0; j < kTilesPerBlock; ++j) {
        Tile(TileAt(f, m, kt, (b * kTilesPerBlock) + j), f.params.k_bits, f.params.codebook, tile);
        for (std::uint64_t r = 0; r < kTile; ++r) {
          for (std::uint64_t c = 0; c < kTile; ++c) {
            block[((kt * kTile + r) * kHadamardBlock) + (j * kTile) + c] =
                F16ToF32(tile.at((r * kTile) + c));
          }
        }
      }
    }
    for (std::uint64_t row = 0; row < k; ++row) {
      Hadamard(&block[row * kHadamardBlock]);
    }
    const std::uint64_t lo = std::max(first, b * kHadamardBlock);
    const std::uint64_t hi = std::min(end, (b + 1) * kHadamardBlock);
    for (std::uint64_t n = lo; n < hi; ++n) {
      const std::uint64_t j = n - (b * kHadamardBlock);
      for (std::uint64_t row = 0; row < k; ++row) {
        column[row] = block[(row * kHadamardBlock) + j];
      }
      for (std::uint64_t kb = 0; kb < k; kb += kHadamardBlock) {
        Hadamard(&column[kb]);
      }
      const double sv = Half(m.svh, n);
      float* y = out.data() + ((n - first) * k);
      for (std::uint64_t row = 0; row < k; ++row) {
        y[row] = static_cast<float>(column[row] * Half(m.suh, row) * sv / 128.0);
      }
    }
  }
}

}  // namespace exl3
}  // namespace llmp::kernels::reference
