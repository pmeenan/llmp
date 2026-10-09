// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The reference executor (D-107, D-108): every op of op set v1 (ir/op.h)
// on the host, the oracle every component, pattern and backend port is
// checked against. Built in every profile; no device code.
//
// It runs a verified graph's nodes in order over host buffers it owns: one
// per root tensor, in the tensor's format, so a view reads and a cache
// store writes its root's bytes as a device would. Weights and state are
// read through the CPU decoders (decode.h) and written through their
// encoders; arithmetic is in double over the decoded values, and each
// result is rounded once to its tensor's encoding. A fused node runs as
// its primitive decomposition (ir/decompose.h); composite ops run their
// own definitions, which the tests hold against their decompositions.
//
// Where an op's definition leaves an order to the implementation (top-k,
// argsort among equals) the reference takes the lower index first; where
// it leaves a value undefined (a fully masked attention row) it gives 0
// (op.h). Correctness before speed: products are plain loops, and a
// small layer runs in a unit test's time.
//
// Collectives run over simulated devices: RunDevices steps several
// executors, one a device, until each reaches a collective, and completes
// a collective once every member of its group has reached one. A send
// queues its bytes for its peer; a receive waits for them.

#ifndef LLMP_KERNELS_REFERENCE_EXEC_H_
#define LLMP_KERNELS_REFERENCE_EXEC_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "format/encoding.h"
#include "ir/graph.h"
#include "kernels/reference/decode.h"

namespace llmp::kernels::reference {

enum class ExecError : std::uint8_t {
  kGraph,        // the graph does not verify, or its decomposition does not
  kUnbound,      // a leaf without its bytes at Run
  kBinding,      // not a leaf, or bytes of the wrong parts or size
  kUnsupported,  // a format the reference does not decode or encode
  kIndex,        // an index outside its table (ids, rows, cells, tokens)
  kNumeric,      // sampling refused its logits
  kCollective,   // collectives that do not pair up, or deadlock
};
std::string_view ExecErrorName(ExecError error);

struct ExecFailure {
  ExecError error = ExecError::kGraph;
  std::uint32_t node = 0;  // the graph's node it failed at, where one did
  std::string detail;
};

// A device group of RunDevices: its members, as indices into the
// executors, in member order (a member's rank is its position).
struct DeviceGroup {
  std::uint32_t group = 0;
  std::vector<std::uint32_t> devices;
};

class Executor {
 public:
  // `graph` must outlive the executor; it is verified and its fused nodes
  // expanded here.
  static std::expected<Executor, ExecFailure> Create(const ir::Graph& graph);

  Executor(Executor&&) noexcept = default;
  Executor& operator=(Executor&&) noexcept = default;
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;
  ~Executor() = default;

  // A leaf's bytes: a view per part, as format::PartsOf lists them (copied).
  std::expected<void, ExecFailure> Bind(ir::TensorId leaf, std::span<const PartView> parts);
  // A leaf of a single-part format from its values in logical order
  // (encoded, as a quantized activation or state would be).
  std::expected<void, ExecFailure> BindValues(ir::TensorId leaf, std::span<const float> values);
  std::expected<void, ExecFailure> BindInts(ir::TensorId leaf,
                                            std::span<const std::int64_t> values);

  // Runs every node; leaves keep their bytes (state its stores) for the
  // next run, results are recomputed.
  std::expected<void, ExecFailure> Run();

  // Any tensor of the graph, in logical order: values decoded to F32, or
  // an integer tensor's values.
  std::expected<std::vector<float>, ExecFailure> Values(ir::TensorId id) const;
  std::expected<std::vector<std::int64_t>, ExecFailure> Ints(ir::TensorId id) const;
  // A root tensor's bytes of one part (a state tensor after its stores).
  std::span<const std::byte> Bytes(ir::TensorId id,
                                   format::PartRole role = format::PartRole::kData) const;

 private:
  friend std::expected<void, ExecFailure> RunDevices(std::span<Executor* const> devices,
                                                     std::span<const DeviceGroup> groups);
  struct Root {
    std::vector<std::vector<std::byte>> parts;
    bool bound = false;
  };
  Executor() = default;

  const ir::Graph& graph() const { return *expanded_; }
  std::expected<void, ExecFailure> Prepare();
  // Runs nodes from next_ until a collective (returns its index) or the end.
  std::expected<std::size_t, ExecFailure> Advance();
  std::expected<void, ExecFailure> Exec(const ir::Node& node);

  std::expected<std::vector<double>, ExecFailure> Load(ir::TensorId id) const;
  // A dense tensor's values as integers, exactly (64-bit keys included).
  std::expected<std::vector<std::int64_t>, ExecFailure> LoadInts(ir::TensorId id) const;
  std::expected<void, ExecFailure> Store(ir::TensorId id, std::span<const double> values);
  // Row (i1, i2, i3) of a single-part tensor, in its encoding.
  std::expected<void, ExecFailure> StoreRow(ir::TensorId id, std::uint64_t i1, std::uint64_t i2,
                                            std::uint64_t i3, std::span<const double> row);
  std::expected<ir::TensorId, ExecFailure> Mapped(ir::TensorId id) const;
  Root& RootOf(ir::TensorId id);
  const Root& RootOf(ir::TensorId id) const;

  const ir::Graph* source_ = nullptr;
  std::optional<ir::Graph> expanded_;
  std::vector<ir::TensorId> map_;  // source tensor -> expanded tensor
  std::vector<Root> roots_;        // by expanded tensor (roots only)
  std::size_t next_ = 0;
};

// Runs the executors as devices, each from its first node to its last,
// completing collectives across `groups`. A group not listed, with one
// executor, is that executor alone.
std::expected<void, ExecFailure> RunDevices(std::span<Executor* const> devices,
                                            std::span<const DeviceGroup> groups);

}  // namespace llmp::kernels::reference

#endif  // LLMP_KERNELS_REFERENCE_EXEC_H_
