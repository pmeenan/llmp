// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The graph IR's verifier (D-107): whether a graph is one op set v1 gives
// a meaning to. Backend-specific indexing limits are checked by each lowering. It checks, node by
// node in order:
//   - structure: every source made by an earlier node (so the graph is
//     acyclic and in a runnable order), every tensor made by exactly one
//     node, the outputs and kept tensors existing, the bounds on nodes and
//     tensors;
//   - arity and parameters: each op's sources present or absent as op.h
//     lists them, parameters in range, reserved parameter bytes zero;
//   - shapes and formats: each output's shape, encoding, layout and strides
//     what the op makes of its sources (infer.h), integer operands integer,
//     formats valid (format::Validate), shapes in canonical rank;
//   - views: a view's root a root, its encoding the root's, every byte it
//     covers inside the root's bytes, no view of a format stored in parts;
//   - bindings: a weight leaf bound to a WeightRef, a state leaf to a
//     StateRef (its segment the tensor's), no weight or state bound twice;
//     a per-owner node's outputs its owner's segment, and its sources of
//     that segment or joined.

#ifndef LLMP_IR_VERIFY_H_
#define LLMP_IR_VERIFY_H_

#include <array>
#include <concepts>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

#include "format/tensor_format.h"
#include "ir/op.h"
#include "ir/storage_admission.h"

namespace llmp::ir {

class Graph;
enum class GraphError : std::uint8_t;

// Optional synchronous funding hook for diagnostic strings constructed on
// comparison-owned IR paths. The implementation must be allocation-free.
class DiagnosticAdmission {
 public:
  virtual ~DiagnosticAdmission() = default;
  virtual bool TryChargeDiagnostic(std::uint64_t bytes) = 0;
};

struct DiagnosticValue {
  enum class Kind : std::uint8_t { kText, kUnsigned, kSigned, kShape, kOpKey, kOutputName } kind;
  std::string_view text{};
  std::uint64_t number = 0;
  std::int64_t signed_number = 0;
  format::Shape shape{};
  OpKey op_key{};
  std::size_t output_index = 0;
};

struct DiagnosticOutputName {
  OpKey key;
  std::size_t output;
};

template <std::size_t N>
struct DiagnosticMessage {
  std::string_view format;
  std::array<DiagnosticValue, N> values;
};

inline DiagnosticValue DiagnosticArg(std::string_view text) {
  return {.kind = DiagnosticValue::Kind::kText, .text = text};
}
inline DiagnosticValue DiagnosticArg(const std::string& text) {
  return DiagnosticArg(std::string_view(text));
}
inline DiagnosticValue DiagnosticArg(const char* text) {
  return DiagnosticArg(std::string_view(text));
}
inline DiagnosticValue DiagnosticArg(const format::Shape& shape) {
  return {.kind = DiagnosticValue::Kind::kShape, .shape = shape};
}
inline DiagnosticValue DiagnosticArg(OpKey key) {
  return {.kind = DiagnosticValue::Kind::kOpKey, .op_key = key};
}
inline DiagnosticValue DiagnosticArg(DiagnosticOutputName output) {
  return {.kind = DiagnosticValue::Kind::kOutputName,
          .op_key = output.key,
          .output_index = output.output};
}
template <std::unsigned_integral T>
inline DiagnosticValue DiagnosticArg(T value) {
  return {.kind = DiagnosticValue::Kind::kUnsigned, .number = static_cast<std::uint64_t>(value)};
}
template <std::signed_integral T>
inline DiagnosticValue DiagnosticArg(T value) {
  return {.kind = DiagnosticValue::Kind::kSigned,
          .signed_number = static_cast<std::int64_t>(value)};
}

std::expected<std::string, GraphError> MakeDiagnostic(DiagnosticAdmission* admission,
                                                      std::string_view format,
                                                      std::span<const DiagnosticValue> values);

template <typename... Args>
std::expected<std::string, GraphError> FormatDiagnostic(DiagnosticAdmission* admission,
                                                        std::string_view format,
                                                        const Args&... args) {
  const std::array<DiagnosticValue, sizeof...(Args)> values{DiagnosticArg(args)...};
  return MakeDiagnostic(admission, format, values);
}

template <typename... Args>
DiagnosticMessage<sizeof...(Args)> DiagFormat(std::string_view format, const Args&... args) {
  return {.format = format, .values = {DiagnosticArg(args)...}};
}

template <std::size_t N>
std::expected<std::string, GraphError> MakeDiagnostic(DiagnosticAdmission* admission,
                                                      const DiagnosticMessage<N>& message) {
  return MakeDiagnostic(admission, message.format, message.values);
}

enum class GraphError : std::uint8_t {
  kShape,      // an output's or operand's shape is not what the op takes or makes
  kFormat,     // an encoding, layout or format the op does not take
  kArity,      // a source missing, extra or absent where required
  kBounds,     // a view past its root, a node or tensor count past the bounds
  kCycle,      // a source not made by an earlier node; a tensor made twice
  kParams,     // parameters out of range, or reserved bytes set
  kBinding,    // a leaf's role, binding or segment wrong, or bound twice
  kLimit,      // past the GGML lowering's int32 extents and strides (RE-037)
  kAdmission,  // comparison-owned construction payment refused before growth
};
std::string_view GraphErrorName(GraphError error);

struct VerifyError {
  GraphError error = GraphError::kShape;
  std::uint32_t node = 0;  // the node it is found at (its index)
  std::string detail;      // what, in words: the operand, shapes and limits
};

std::expected<void, VerifyError> Verify(const Graph& graph,
                                        std::shared_ptr<const StorageFunder> funder = {},
                                        DiagnosticAdmission* diagnostic = nullptr);
// Conservative temporary-allocation bound for Verify's exact graph counts.
// This does not include Graph::Digest's independent canonicalization scratch.
std::expected<std::uint64_t, VerifyError> VerifyScratchBound(const Graph& graph);

}  // namespace llmp::ir

#endif  // LLMP_IR_VERIFY_H_
