// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/exl3/validate.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace llmp::kernels::exl3 {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

// A device range [base, base + bytes).
struct Range {
  std::string_view name;
  std::uint64_t base = 0;
  std::uint64_t bytes = 0;
};

bool Overlap(const Range& a, const Range& b) {
  return a.bytes > 0 && b.bytes > 0 && a.base < b.base + b.bytes && b.base < a.base + a.bytes;
}

// The range is non-null, aligned and does not wrap. Sizes come from bounded
// dimensions, so the sum cannot overflow below 2^64 unless the base is
// near its top.
std::expected<void, KernelFailure> CheckRange(const Range& range, std::uint64_t alignment) {
  if (range.base == 0) {
    return Rejected(std::format("{} is null", range.name));
  }
  if (range.base % alignment != 0) {
    return Rejected(
        std::format("{} at {:#x} is not {}-byte aligned", range.name, range.base, alignment));
  }
  if (range.base + range.bytes < range.base) {
    return Rejected(std::format("{} wraps the address space", range.name));
  }
  return {};
}

// `written` overlaps none of `others`.
std::expected<void, KernelFailure> Disjoint(const Range& written,
                                            std::initializer_list<Range> others) {
  for (const Range& other : others) {
    if (Overlap(written, other)) {
      return Rejected(std::format("{} overlaps {}", written.name, other.name));
    }
  }
  return {};
}

std::expected<void, KernelFailure> CheckRows(int m) {
  if (m < 1 || m > kMaxRows) {
    return Rejected(std::format("{} rows; a launch takes 1 to {}", m, kMaxRows));
  }
  return {};
}

// m × width stays within int for the kernels' index arithmetic, for
// `matrices` matrices side by side.
std::expected<void, KernelFailure> CheckExtent(int m, int width, int matrices) {
  const std::uint64_t elements = static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(width) *
                                 static_cast<std::uint64_t>(matrices);
  if (elements > INT32_MAX) {
    return Rejected(std::format("{} rows of {} columns overflow the kernels' indices", m, width));
  }
  return {};
}

std::uint64_t Half(std::uint64_t count) { return count * 2; }

Range TrellisRange(const Weights& w) { return {"the trellis", w.trellis, TrellisBytes(w)}; }
Range SuhRange(const Weights& w) { return {"suh", w.suh, Half(static_cast<std::uint64_t>(w.k))}; }
Range SvhRange(const Weights& w) { return {"svh", w.svh, Half(static_cast<std::uint64_t>(w.n))}; }

// The checks every packed launch of one linear shares.
std::expected<void, KernelFailure> CheckLinear(const LinearOperands& o, std::uint64_t locks) {
  if (auto weights = CheckWeights(o.weights); !weights) {
    return weights;
  }
  if (auto rows = CheckRows(o.m); !rows) {
    return rows;
  }
  if (auto extent = CheckExtent(o.m, o.weights.k, 1); !extent) {
    return extent;
  }
  if (auto extent = CheckExtent(o.m, o.weights.n, 1); !extent) {
    return extent;
  }
  const auto m = static_cast<std::uint64_t>(o.m);
  const Range x{"x", o.x, Half(m * static_cast<std::uint64_t>(o.weights.k))};
  const Range a_had{"a_had", o.a_had, ScratchBytes(o)};
  const Range y{"y", o.y,
                m * static_cast<std::uint64_t>(o.weights.n) *
                    static_cast<std::uint64_t>(OutputBytes(o.output))};
  const Range lock_area{"the lock area", locks, kLockBytes};
  // x is read as 8-byte vectors, a_had as 16-byte ones (cp.async), and y is
  // written, then transformed in place, in 8-byte (F16) or 16-byte (F32)
  // vectors.
  for (const auto& [range, alignment] :
       {std::pair{x, std::uint64_t{8}}, std::pair{a_had, std::uint64_t{16}},
        std::pair{y, o.output == Output::kF32 ? std::uint64_t{16} : std::uint64_t{8}},
        std::pair{lock_area, std::uint64_t{16}}}) {
    if (auto checked = CheckRange(range, alignment); !checked) {
      return checked;
    }
  }
  if (auto disjoint = Disjoint(a_had, {x, y, TrellisRange(o.weights), SuhRange(o.weights),
                                       SvhRange(o.weights), lock_area});
      !disjoint) {
    return disjoint;
  }
  return Disjoint(
      y, {x, TrellisRange(o.weights), SuhRange(o.weights), SvhRange(o.weights), lock_area});
}

std::expected<void, KernelFailure> CheckCoresident(int blocks, int coresident) {
  if (blocks < 1) {
    return Rejected(std::format("a grid of {} blocks", blocks));
  }
  if (blocks > coresident) {
    return Rejected(std::format(
        "a grid of {} blocks, of which only {} fit on the device at once: a cooperative kernel's "
        "grid must be co-resident (launch_contract.h)",
        blocks, coresident));
  }
  return {};
}

std::expected<void, KernelFailure> CheckShape(int shape, const Weights& w) {
  if (shape < 1 || shape > kShapes) {
    return Rejected(std::format("tile shape {} (1 to {})", shape, kShapes));
  }
  const auto index = static_cast<std::size_t>(shape);
  if (w.k % kTileK.at(index) != 0 || w.n % kTileN.at(index) != 0) {
    return Rejected(std::format("tile shape {} does not divide {} × {}", shape, w.k, w.n));
  }
  return {};
}

// Split-K slices of one matrix under the shape: upstream never launches more
// blocks than slices (exl3_kernel_map.cu, "Avoid empty blocks").
int Slices(int shape, const Weights& w) {
  const auto index = static_cast<std::size_t>(shape);
  return (w.k / kTileK.at(index)) * (w.n / kTileN.at(index));
}

}  // namespace

bool RateCompiled(int bits) { return bits == 4 || bits == 5 || bits == 6 || bits == 8; }

int OutputBytes(Output output) { return output == Output::kF32 ? 4 : 2; }

std::uint64_t TrellisBytes(const Weights& w) {
  if (w.k < 0 || w.n < 0 || w.bits < 0) {
    return 0;
  }
  return static_cast<std::uint64_t>(w.k / 16) * static_cast<std::uint64_t>(w.n / 16) * 16 *
         static_cast<std::uint64_t>(w.bits) * 2;
}

std::expected<void, KernelFailure> CheckWeights(const Weights& w) {
  if (!RateCompiled(w.bits)) {
    return Rejected(std::format("K = {}: this build has kernels for K = 4, 5, 6 and 8", w.bits));
  }
  for (const auto& [name, value] : {std::pair{"k", w.k}, std::pair{"n", w.n}}) {
    if (value < kHadamardBlock || value > kMaxDimension || value % kHadamardBlock != 0) {
      return Rejected(std::format("{} = {}: a multiple of {} from {} to {}", name, value,
                                  kHadamardBlock, kHadamardBlock, kMaxDimension));
    }
  }
  if (TrellisBytes(w) / 2 > INT32_MAX) {
    return Rejected("the trellis has more elements than the kernels index");
  }
  for (const auto& [range, alignment] :
       {std::pair{TrellisRange(w), std::uint64_t{16}}, std::pair{SuhRange(w), std::uint64_t{8}},
        std::pair{SvhRange(w), std::uint64_t{8}}}) {
    if (auto checked = CheckRange(range, alignment); !checked) {
      return checked;
    }
  }
  return {};
}

int GemvThreads(int config) { return config == 0 ? 512 : 256; }
int GemvColumns(int config) { return config == 0 ? 32 : 64; }

std::uint64_t ScratchBytes(const LinearOperands& o) {
  return Half(static_cast<std::uint64_t>(o.m < 0 ? 0 : o.m) *
              static_cast<std::uint64_t>(o.weights.k < 0 ? 0 : o.weights.k));
}

std::expected<void, KernelFailure> CheckGemm(const LinearOperands& o, const GemmPlan& plan,
                                             int coresident, std::uint64_t locks) {
  if (auto linear = CheckLinear(o, locks); !linear) {
    return linear;
  }
  if (auto shape = CheckShape(plan.shape, o.weights); !shape) {
    return shape;
  }
  if (plan.blocks > Slices(plan.shape, o.weights)) {
    return Rejected(
        std::format("{} blocks for {} split-K slices", plan.blocks, Slices(plan.shape, o.weights)));
  }
  return CheckCoresident(plan.blocks, coresident);
}

std::expected<void, KernelFailure> CheckGemv(const LinearOperands& o, const GemvPlan& plan,
                                             int coresident, std::uint64_t locks) {
  if (auto linear = CheckLinear(o, locks); !linear) {
    return linear;
  }
  if (o.weights.bits != 4) {
    return Rejected(std::format("the GEMV is compiled for K = 4, not {}", o.weights.bits));
  }
  if (o.m > kGemvMaxRows) {
    return Rejected(std::format("{} rows; the GEMV takes at most {}", o.m, kGemvMaxRows));
  }
  if (plan.config != 0 && plan.config != 1) {
    return Rejected(std::format("GEMV configuration {} (0 or 1)", plan.config));
  }
  // Upstream's grid is at most one block per column group
  // (exl3_gemv_try_launch): more would leave blocks without work.
  const int groups = o.weights.n / GemvColumns(plan.config);
  if (plan.blocks > groups) {
    return Rejected(std::format("{} blocks for {} column groups", plan.blocks, groups));
  }
  return CheckCoresident(plan.blocks, coresident);
}

std::uint64_t ScratchBytes(const MultiLinearOperands& o) {
  return 2 * Half(static_cast<std::uint64_t>(o.m < 0 ? 0 : o.m) *
                  static_cast<std::uint64_t>(o.first.k < 0 ? 0 : o.first.k));
}

std::array<std::uint64_t, 6> MultiGemmTables(const Weights& first, const Weights& second) {
  return {first.trellis, second.trellis, first.suh, second.suh, first.svh, second.svh};
}

std::expected<void, KernelFailure> CheckMultiGemm(const MultiLinearOperands& o,
                                                  const MultiGemmPlan& plan, int coresident,
                                                  std::uint64_t locks) {
  for (const Weights* w : {&o.first, &o.second}) {
    if (auto weights = CheckWeights(*w); !weights) {
      return weights;
    }
  }
  if (o.written != MultiGemmTables(o.first, o.second)) {
    return Rejected(
        "the tables were written for other tensors than the operands' weights: rebuild them "
        "after a move (BP-P5)");
  }
  if (o.first.k != o.second.k || o.first.n != o.second.n || o.first.bits != o.second.bits) {
    return Rejected("the two linears of a multi-GEMM share k, n and K");
  }
  if (auto rows = CheckRows(o.m); !rows) {
    return rows;
  }
  if (auto extent = CheckExtent(o.m, o.first.k, 2); !extent) {
    return extent;
  }
  if (auto extent = CheckExtent(o.m, o.first.n, 2); !extent) {
    return extent;
  }
  const auto m = static_cast<std::uint64_t>(o.m);
  const Range x{"x", o.x, Half(m * static_cast<std::uint64_t>(o.first.k))};
  const Range a_had{"a_had", o.a_had, ScratchBytes(o)};
  const Range y{"y", o.y,
                2 * m * static_cast<std::uint64_t>(o.first.n) *
                    static_cast<std::uint64_t>(OutputBytes(o.output))};
  const Range lock_area{"the lock area", locks, kLockBytes};
  const std::array<Range, 3> tables{{{"the trellis table", o.trellis_table, 16},
                                     {"the suh table", o.suh_table, 16},
                                     {"the svh table", o.svh_table, 16}}};
  for (const auto& [range, alignment] :
       {std::pair{x, std::uint64_t{8}}, std::pair{a_had, std::uint64_t{16}},
        std::pair{y, o.output == Output::kF32 ? std::uint64_t{16} : std::uint64_t{8}},
        std::pair{lock_area, std::uint64_t{16}}, std::pair{tables[0], std::uint64_t{8}},
        std::pair{tables[1], std::uint64_t{8}}, std::pair{tables[2], std::uint64_t{8}}}) {
    if (auto checked = CheckRange(range, alignment); !checked) {
      return checked;
    }
  }
  for (const Weights* w : {&o.first, &o.second}) {
    for (const Range& written : {a_had, y}) {
      if (auto disjoint = Disjoint(written, {TrellisRange(*w), SuhRange(*w), SvhRange(*w)});
          !disjoint) {
        return disjoint;
      }
    }
  }
  if (auto disjoint = Disjoint(a_had, {x, y, lock_area, tables[0], tables[1], tables[2]});
      !disjoint) {
    return disjoint;
  }
  if (auto disjoint = Disjoint(y, {x, lock_area, tables[0], tables[1], tables[2]}); !disjoint) {
    return disjoint;
  }
  if (auto shape = CheckShape(plan.shape, o.first); !shape) {
    return shape;
  }
  if (plan.concurrency < 1 || plan.concurrency > 2) {
    return Rejected(std::format("concurrency {}: one or two matrices at a time", plan.concurrency));
  }
  if (plan.blocks > Slices(plan.shape, o.first)) {
    return Rejected(
        std::format("{} blocks for {} split-K slices", plan.blocks, Slices(plan.shape, o.first)));
  }
  if (plan.blocks < 1) {
    return Rejected(std::format("a grid of {} blocks", plan.blocks));
  }
  return CheckCoresident(plan.blocks * plan.concurrency, coresident);
}

std::expected<void, KernelFailure> CheckReconstruct(const ReconstructOperands& o) {
  if (auto weights = CheckWeights(o.weights); !weights) {
    return weights;
  }
  if (o.columns < kHadamardBlock || o.columns % kHadamardBlock != 0 || o.column < 0 ||
      o.column % kHadamardBlock != 0 || o.column > o.weights.n - o.columns) {
    return Rejected(std::format("columns [{}, {}) of {}: 128-column blocks inside the matrix",
                                o.column, static_cast<std::int64_t>(o.column) + o.columns,
                                o.weights.n));
  }
  const Range w{
      "w", o.w,
      Half(static_cast<std::uint64_t>(o.weights.k) * static_cast<std::uint64_t>(o.columns))};
  // reconstruct_kernel stores 16-byte vectors, reconstruct_had_kernel
  // 8-byte ones; the reconstruction GEMM reads w with 16-byte alignment
  // (exl3-recon-pin.json), so both need 16.
  if (auto checked = CheckRange(w, 16); !checked) {
    return checked;
  }
  return Disjoint(w, {TrellisRange(o.weights), SuhRange(o.weights), SvhRange(o.weights)});
}

std::expected<void, KernelFailure> CheckHadamard(const HadamardOperands& o) {
  if (auto rows = CheckRows(o.rows); !rows) {
    return rows;
  }
  if (o.columns < kHadamardBlock || o.columns > kMaxDimension || o.columns % kHadamardBlock != 0) {
    return Rejected(std::format("{} columns: a multiple of {}", o.columns, kHadamardBlock));
  }
  if (o.input_scale && o.type != Output::kF16) {
    return Rejected("the input transform is compiled for F16 only");
  }
  if (auto extent = CheckExtent(o.rows, o.columns, 1); !extent) {
    return extent;
  }
  const std::uint64_t bytes = static_cast<std::uint64_t>(o.rows) *
                              static_cast<std::uint64_t>(o.columns) *
                              static_cast<std::uint64_t>(OutputBytes(o.type));
  const std::uint64_t vector = o.type == Output::kF32 ? 16 : 8;
  const Range x{"x", o.x, bytes};
  const Range y{"y", o.y, bytes};
  const Range scale{"the scale", o.scale, Half(static_cast<std::uint64_t>(o.columns))};
  for (const auto& [range, alignment] :
       {std::pair{x, vector}, std::pair{y, vector}, std::pair{scale, std::uint64_t{8}}}) {
    if (auto checked = CheckRange(range, alignment); !checked) {
      return checked;
    }
  }
  if (o.x != o.y) {
    if (auto disjoint = Disjoint(y, {x}); !disjoint) {
      return disjoint;
    }
  }
  return Disjoint(y, {scale});
}

std::expected<void, KernelFailure> CheckBias(const BiasOperands& o) {
  if (auto rows = CheckRows(o.rows); !rows) {
    return rows;
  }
  if (o.columns < 1 || o.columns > kMaxDimension) {
    return Rejected(std::format("{} columns", o.columns));
  }
  if (auto extent = CheckExtent(o.rows, o.columns, 1); !extent) {
    return extent;
  }
  const std::uint64_t bytes =
      Half(static_cast<std::uint64_t>(o.rows) * static_cast<std::uint64_t>(o.columns));
  const Range x{"x", o.x, bytes};
  const Range y{"y", o.y, bytes};
  const Range bias{"the bias", o.bias, Half(static_cast<std::uint64_t>(o.columns))};
  for (const Range& range : {x, y, bias}) {
    if (auto checked = CheckRange(range, 2); !checked) {
      return checked;
    }
  }
  if (o.x != o.y) {
    if (auto disjoint = Disjoint(y, {x}); !disjoint) {
      return disjoint;
    }
  }
  return Disjoint(y, {bias});
}

std::expected<void, KernelFailure> CheckReconGemm(const ReconGemmOperands& o) {
  if (auto rows = CheckRows(o.m); !rows) {
    return rows;
  }
  for (const auto& [name, value] : {std::pair{"k", o.k}, std::pair{"n", o.n}}) {
    if (value < 1 || value > kMaxDimension) {
      return Rejected(std::format("{} = {}", name, value));
    }
  }
  if (o.ldc < o.n || o.ldc > kMaxDimension) {
    return Rejected(std::format("an output row stride of {} for {} columns", o.ldc, o.n));
  }
  if (auto extent = CheckExtent(o.m, o.ldc, 1); !extent) {
    return extent;
  }
  const auto m = static_cast<std::uint64_t>(o.m);
  const Range w{"w", o.w, Half(static_cast<std::uint64_t>(o.k) * static_cast<std::uint64_t>(o.n))};
  const Range x{"x", o.x, Half(m * static_cast<std::uint64_t>(o.k))};
  const Range y{"y", o.y,
                (((m - 1) * static_cast<std::uint64_t>(o.ldc)) + static_cast<std::uint64_t>(o.n)) *
                    static_cast<std::uint64_t>(OutputBytes(o.output))};
  // The pinned algorithms were chosen for 16-byte aligned operands
  // (exl3-recon-pin.json), which holds for every row only if the leading
  // dimensions keep it (cublasLtMatmulAlgoCheck sees no addresses).
  const std::uint64_t row_bytes =
      static_cast<std::uint64_t>(o.ldc) * static_cast<std::uint64_t>(OutputBytes(o.output));
  if (o.k % 8 != 0 || o.n % 8 != 0 || row_bytes % 16 != 0) {
    return Rejected(
        std::format("k = {}, n = {} and a row stride of {} keep no 16-byte rows", o.k, o.n, o.ldc));
  }
  for (const Range& range : {w, x, y}) {
    if (auto checked = CheckRange(range, 16); !checked) {
      return checked;
    }
  }
  return Disjoint(y, {w, x});
}

LinearPath UpstreamPath(int rows) {
  if (rows >= 1024) {
    return LinearPath::kReconstructFused;
  }
  return rows > 144 ? LinearPath::kReconstruct : LinearPath::kPacked;
}

std::vector<int> ReconstructSlices(int n) {
  std::vector<int> slices;
  for (int start = 0; start < n; start += kReconstructSliceColumns) {
    slices.push_back(std::min(kReconstructSliceColumns, n - start));
  }
  return slices;
}

std::uint64_t ReconstructScratchBytes(const Weights& weights) {
  if (weights.k < 0 || weights.n < 0) {
    return 0;
  }
  return Half(static_cast<std::uint64_t>(weights.k) *
              static_cast<std::uint64_t>(std::min(weights.n, kReconstructSliceColumns)));
}

std::expected<void, KernelFailure> CheckReconstructedScratch(const ReconstructedOperands& o,
                                                             bool fused) {
  if (o.m < 0 || o.weights.k < 0 || o.weights.n < 0) {
    return Rejected("negative dimensions");
  }
  const auto m = static_cast<std::uint64_t>(o.m);
  const Range x{"x", o.x, Half(m * static_cast<std::uint64_t>(o.weights.k))};
  const Range xh{"xh", fused ? 0 : o.xh,
                 fused ? 0 : Half(m * static_cast<std::uint64_t>(o.weights.k))};
  const Range w{"w", o.w, ReconstructScratchBytes(o.weights)};
  const Range y{"y", o.y,
                m * static_cast<std::uint64_t>(o.weights.n) *
                    static_cast<std::uint64_t>(OutputBytes(o.output))};
  if (auto disjoint = Disjoint(w, {x, xh, y}); !disjoint) {
    return disjoint;
  }
  if (auto disjoint = Disjoint(xh, {x, y}); !disjoint) {
    return disjoint;
  }
  // Nothing the path writes may hold a tensor it reads later: the weights
  // (read by every slice), and the bias (read last).
  const Range bias{"the bias", o.bias,
                   o.bias == 0 ? 0 : Half(static_cast<std::uint64_t>(o.weights.n))};
  for (const Range& written : {xh, w, y}) {
    if (auto disjoint = Disjoint(
            written, {TrellisRange(o.weights), SuhRange(o.weights), SvhRange(o.weights), bias});
        !disjoint) {
      return disjoint;
    }
  }
  return {};
}

std::expected<void, KernelFailure> CheckBiasApart(const LinearOperands& o, std::uint64_t bias,
                                                  std::uint64_t locks) {
  const Range range{"the bias", bias,
                    Half(static_cast<std::uint64_t>(o.weights.n < 0 ? 0 : o.weights.n))};
  return Disjoint(range,
                  {{"a_had", o.a_had, ScratchBytes(o)}, {"the lock area", locks, kLockBytes}});
}

}  // namespace llmp::kernels::exl3
