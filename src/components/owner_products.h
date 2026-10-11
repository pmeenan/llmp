// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Checked ownership of physical columns in an ordinary dense product.
// A component names a product; the framework proves its input lineage from
// typed step inputs. No caller supplies a columns-per-row multiplier.
#ifndef LLMP_COMPONENTS_OWNER_PRODUCTS_H_
#define LLMP_COMPONENTS_OWNER_PRODUCTS_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "base/sha256.h"
#include "ir/graph.h"

namespace llmp::components {
struct StepInput;
struct SegmentRows;

enum class RowBasis : std::uint8_t { kInput, kOutput };
struct OwnerColumns {
  std::uint8_t slot = 0;
  std::uint8_t segment = 0;
  std::uint64_t first_row = 0;
  std::uint64_t rows = 0;
  std::uint64_t first_column = 0;
  std::uint64_t columns = 0;
  std::uint64_t byte_offset = 0;
  std::uint64_t bytes = 0;
};
struct OwnerProduct {
  ir::NodeId node{};
  ir::TensorId root{};
  ir::TensorId weight{};
  ir::TensorId input{};
  ir::TensorId output{};
  RowBasis basis = RowBasis::kInput;
  std::uint8_t axis = 1;
  std::uint64_t columns_per_row = 0;
  std::vector<OwnerColumns> owners;
};

// Immutable proof tied to the graph semantics and registered dependencies'
// numbered edges. The private constructor prevents unchecked factors/intervals.
class OwnerProducts {
 public:
  OwnerProducts() = default;
  std::span<const OwnerProduct> products() const { return products_; }
  const base::Sha256Digest& graph_digest() const { return graph_; }
  const base::Sha256Digest& digest() const { return digest_; }

 private:
  friend std::expected<OwnerProducts, std::string> ProveOwnerProducts(
      const ir::Graph&, std::span<const StepInput>, std::span<const SegmentRows>,
      std::span<const ir::TensorId>);
  base::Sha256Digest graph_{};
  base::Sha256Digest digest_{};
  std::vector<OwnerProduct> products_;
};

// The canonical graph digest plus ordered node keys and numbered edges.
// This seals registered product dependencies against used-leaf reordering;
// it does not distinguish binding swaps of unused same-kind leaves.
base::Sha256Digest OwnerGraphIdentity(const ir::Graph& graph);

// Requests name MatMul outputs. Accepted lineage: canonical token GetRows
// or features, row-local RMS/Mul, and whole contiguous reshapes/views. All
// inputs/owners and alias writes are checked; unknown lineage refuses.
std::expected<OwnerProducts, std::string> ProveOwnerProducts(
    const ir::Graph& graph, std::span<const StepInput> inputs,
    std::span<const SegmentRows> segments, std::span<const ir::TensorId> requests);
}  // namespace llmp::components
#endif  // LLMP_COMPONENTS_OWNER_PRODUCTS_H_
