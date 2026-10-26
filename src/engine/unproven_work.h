// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// One fence-proof contract for work whose end could not be proven (D-102's
// hang recovery; docs/engine.md, "Unproven work"):
//
// - The node owns a per-stream quarantine of the pinned buffers and storage
//   rings such work may still touch (PagedNode::KeepPinned, RetireRing).
//   Only a fence it saw complete on that stream frees them (FenceStream,
//   TearDown).
// - Every other owner keeps what that work may use in a typed retention of
//   its own: a flight (RetainedFlight: a runner's step transaction or routed
//   flight) or a plain mark (UnprovenMark: a private batch, a staged image's
//   state).
// - Every clear of either takes a FenceProof, which only the node mints, for
//   the stream it fenced. A flight whose own completion proved its end goes
//   with ReleaseRetired instead: that is its retirement, not a clear.
//
// Part of the engine core.

#ifndef LLMP_ENGINE_UNPROVEN_WORK_H_
#define LLMP_ENGINE_UNPROVEN_WORK_H_

#include <cstdint>
#include <memory>
#include <utility>

namespace llmp::engine {

class PagedNode;

// What only the node can say: every piece of work queued on `stream()`
// before the proof was minted has ended (a fence the node recorded after it
// was seen complete, in stream order).
class FenceProof {
 public:
  // Never copied or kept: a proof outliving its fence would claim work
  // queued after it.
  FenceProof(const FenceProof&) = delete;
  FenceProof& operator=(const FenceProof&) = delete;
  std::uint32_t stream() const noexcept { return stream_; }

 private:
  friend class PagedNode;
  explicit FenceProof(std::uint32_t stream) noexcept : stream_(stream) {}
  std::uint32_t stream_ = 0;
};

// An owner's mark: work it queued may still run. Set on the failure that
// left it unproven; cleared only by a fence of that work's stream.
class UnprovenMark {
 public:
  bool active() const noexcept { return marked_; }
  void Mark() noexcept { marked_ = true; }
  void Clear(const FenceProof& /*proof*/) noexcept { marked_ = false; }

 private:
  bool marked_ = false;
};

// An owner of type T kept alive while work that uses it may still run.
template <typename T>
class RetainedFlight {
 public:
  void Retain(std::shared_ptr<T> flight) noexcept { flight_ = std::move(flight); }
  bool held() const noexcept { return flight_ != nullptr; }
  const std::shared_ptr<T>& get() const noexcept { return flight_; }
  T* operator->() const noexcept { return flight_.get(); }
  // Its own completion proved it ended (or it never started).
  void ReleaseRetired() noexcept { flight_.reset(); }
  // A fence of its stream proved it ended.
  void Clear(const FenceProof& /*proof*/) noexcept { flight_.reset(); }

 private:
  std::shared_ptr<T> flight_;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_UNPROVEN_WORK_H_
