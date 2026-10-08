// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Phase kinds, decoding modes and request programs (D-050, D-068;
// docs/architecture.md#adapters-and-request-programs), as small as M2's
// shape-expressibility scenarios need them. Internal: the
// operation-contract decision (backend-proof P6) settles these types, and
// a phase kind's operation invocations join it through the plan registry
// (registry.h).
//
// A ProgramContract is what a plan binds for one model context and request
// profile: the context's limit, its live-state representations, the
// transient working state its programs keep across their own completed
// boundaries (a canvas between denoising steps, drafted positions awaiting
// verification), its phase kinds and its decoding mode. A phase kind
// declares the components its closure reads, its validated widths (the
// positions one phase processes: a k+1-token verify or a 256-position
// canvas, never assumed to be one) with the working set at each, how its
// width follows the request, how often it may run, and whether it appends
// state or emits output.
//
// PlanProgram turns a contract and a finite request into the request's
// D-050 envelope, or a rejection that names the phase kind, width,
// required bytes and shortfall:
//   - R_i: each live state's allowance through the admitted bound, every
//     transient working state and the output buffer. Everything the
//     contract says a program keeps across a completed boundary is here,
//     never in E_i, as D-069 requires of a pausable plan (the runtime's
//     own task and result records are charged by the runtime).
//   - E_i: the largest phase envelope: the unique bytes of the phase's
//     closure over the context's components (shared extents once) plus
//     its working set, at every width it may run at.
// Every bound is fixed before admission: the output bound (lowered for
// block output to end on a block the context holds), the draft depth, the
// denoising steps and hence the phase count. Run-time outcomes, such as a
// rejected draft or an early stop, only shorten the program.
//
// A phase that fails, or is cancelled, ends its request: D-050's explicit
// unwind, never a retry within the admitted program. Its tentative
// positions are never committed, and its output never published; they go
// with the request's state at retirement, so a failed block commit needs
// no truncation. Only speculation rolls positions back on its ordinary
// path, so only a speculative plan requires every live state to truncate
// (to any position, or through a snapshot at the prefix and after every
// drafted position). Whatever publishes a request's state for reuse
// (D-055) publishes its committed prefix, and only a state with no
// tentative positions.
//
// A ProgramCursor tracks the phase in flight and refuses a phase beyond the
// admitted program or at an unvalidated width, and an OutputBuffer holds a
// request's output credits: a phase that can emit n positions starts only
// once the buffer has room for their worst-case bytes and n positions
// remain before the output bound, and its output becomes visible all at
// once at its completed boundary, or not at all. Nothing ties a cursor to
// admission: whoever reports a completed boundary to the switching policy
// (D-069) checks ProgramCursor::AtBoundary first.
//
// None of this names a model architecture; the resource core (catalog,
// ledgers, admission) never includes it.

#ifndef LLMP_EXECUTION_PROGRAM_H_
#define LLMP_EXECUTION_PROGRAM_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "base/bytes.h"
#include "catalog/catalog.h"
#include "memory/commitment.h"
#include "model/context.h"
#include "model/state.h"

namespace llmp::execution {

using base::Bytes;

// How a phase kind's width follows the request.
enum class WidthRule : std::uint8_t {
  kChunk,   // the widest validated width that fits the budget (prefill chunks)
  kOne,     // one position (an autoregressive step)
  kDraft,   // the request's draft depth
  kVerify,  // the draft depth plus one
  kBlock,   // the decoding mode's block (a canvas)
};

// How many times a phase kind may run in one program, at most.
enum class Repeat : std::uint8_t {
  kPromptChunks,  // enough chunks to cover the prompt at its narrowest width
  kOutputSteps,   // once per output position
  kBlockSteps,    // the denoising steps of every block
  kBlocks,        // once per block
};

struct PhaseWidth {
  std::uint64_t positions = 0;
  // Activations and workspace at this width, beyond the closure: released
  // at the phase's completed boundary.
  Bytes working;
};

struct PhaseKind {
  std::string name;
  std::vector<model::ComponentRole> components;  // whose resources its closure holds
  std::vector<PhaseWidth> widths;                // validated, strictly ascending
  WidthRule rule = WidthRule::kOne;
  Repeat repeat = Repeat::kOutputSteps;
  // Writes up to `width` positions, tentatively, to the live states it
  // updates: which ones is the adapter's (a drafter's own KV, say), since
  // every live state is charged its allowance at the bound either way.
  bool appends = false;
  bool emits = false;  // may emit up to `width` positions at its boundary
};

// How output arrives (D-068): single tokens, accepted runs or whole blocks.
enum class OutputUnit : std::uint8_t { kToken, kAcceptedRun, kBlock };

struct DecodingMode {
  std::string name;
  OutputUnit unit = OutputUnit::kToken;
  std::uint32_t max_draft = 0;  // speculative when nonzero: the deepest draft
  std::uint64_t block = 0;      // block output when nonzero: the canvas positions
  std::uint32_t max_steps = 0;  // block output: the most denoising steps per block
  Bytes wire_per_position;      // worst-case output wire bytes per position
};

// Transient working state (docs/architecture.md#adapters-and-request-programs):
// kept across the program's own completed boundaries, never published.
struct WorkingState {
  std::string name;
  Bytes bytes;
};

struct ProgramContract {
  std::uint64_t context_limit = 0;  // positions the context can hold
  std::vector<model::StateRepresentation> states;
  std::vector<WorkingState> working;
  std::vector<PhaseKind> phases;
  DecodingMode mode;
};

// A finite request (D-050): every bound is resolved before planning.
struct ProgramRequest {
  std::uint64_t prompt = 0;
  std::uint64_t max_output = 0;
  std::uint32_t draft_depth = 0;  // speculative modes
  std::uint32_t steps = 0;        // block modes: denoising steps per block
};

enum class ProgramError : std::uint8_t {
  kInvalid,           // a malformed contract or request
  kUnsupported,       // a width, capability or feature the contract has not validated
  kContextExhausted,  // the prompt leaves no room for the output (D-045's 400)
  kDoesNotFit,        // a phase's envelope does not fit above the retained state
  kOverflow,
  kCatalog,        // the context's resources are not in the catalog
  kBeyondProgram,  // a phase past the admitted program
  kWrongState,     // a phase begun inside another, or ended outside one
  kOutputFull,     // the output buffer lacks room for the phase's worst case
};

std::string ToString(ProgramError error);

struct ProgramRejection {
  ProgramError error = ProgramError::kInvalid;
  std::string phase;  // the phase kind, where one is at fault
  std::uint64_t width = 0;
  Bytes required;   // R_i plus the phase's envelope, where it does not fit
  Bytes shortfall;  // what `required` exceeds the available bytes by
  std::string detail;
};

std::string ToString(const ProgramRejection& rejection);

// One phase kind as planned: the explainable part of a plan.
struct PlannedPhase {
  std::string kind;
  std::uint64_t width = 0;            // the widest it runs at
  std::vector<std::uint64_t> widths;  // every validated width it may run at, ascending
  Bytes closure;                      // its closure's unique bytes in the domain
  Bytes working;                      // the largest working set at those widths
  Bytes envelope;                     // closure + working
  std::uint64_t max_count = 0;        // the most times it runs
};

struct RetainedItem {
  std::string name;
  Bytes bytes;
};

// The output buffer's name among a plan's retained items; no state or
// working state may take it.
inline constexpr std::string_view kOutputName = "output";

struct ProgramPlan {
  memory::Envelope envelope;
  std::uint64_t prompt = 0;
  std::uint64_t output = 0;       // the admitted output bound
  std::uint64_t state_bound = 0;  // positions each live state may reach
  std::uint64_t blocks = 0;       // block modes
  std::vector<PlannedPhase> phases;
  // R_i, itemized: each live state, then each working state, then the
  // output buffer.
  std::vector<RetainedItem> retained;
  Bytes output_capacity;  // the output buffer's wire bytes
  std::uint64_t max_phases = 0;

  const PlannedPhase* Phase(std::string_view kind) const;
  const RetainedItem* Retained(std::string_view name) const;
};

// Plans `request` under `contract` for `context`, whose resources are in
// `catalog`, in `domain`. `available` is what the request may have alone:
// the budget less fixed overhead (B - F). `granularity` is the provider's
// backing granularity: working sets, working states and the output buffer
// are rounded up to it, and state blocks and snapshots must be whole
// multiples of it. A closure's bytes are the catalog's physical extent
// sizes, taken as recorded; a closure that reaches another domain is
// refused, since this plan checks no envelope there. A contract that could
// only fail once running is refused too: one with no phase that emits, or
// output of no wire bytes, or retained items (states, working states and
// the output buffer) that share a name.
std::expected<ProgramPlan, ProgramRejection> PlanProgram(const ProgramContract& contract,
                                                         const model::ModelContext& context,
                                                         const catalog::Catalog& catalog,
                                                         catalog::DomainId domain,
                                                         const ProgramRequest& request,
                                                         Bytes available, Bytes granularity);

class ProgramCursor {
 public:
  explicit ProgramCursor(const ProgramPlan& plan);

  // Starts a phase of `kind` at `width`, a validated width no wider than
  // planned, if the program has not already run all it may of that kind.
  std::expected<void, ProgramError> Begin(std::string_view kind, std::uint64_t width);
  // The phase reached its completed boundary.
  std::expected<void, ProgramError> End();

  bool AtBoundary() const { return current_.empty(); }
  const std::string& current() const { return current_; }
  std::uint64_t CountOf(std::string_view kind) const;
  std::uint64_t phases_run() const { return phases_run_; }

 private:
  struct Kind {
    std::vector<std::uint64_t> widths;
    std::uint64_t max_count = 0;
    std::uint64_t count = 0;
  };
  std::map<std::string, Kind, std::less<>> kinds_;
  std::string current_;
  std::uint64_t phases_run_ = 0;
};

// A request's output buffer, in wire bytes: its capacity is charged to
// R_i. One reservation at a time, made before the phase that may emit.
// It also holds the program to its admitted output bound: no phase may
// reserve, and so none may publish, a position past it.
class OutputBuffer {
 public:
  // `capacity` wire bytes (ProgramPlan::output_capacity) for a program
  // admitted through `bound` output positions (ProgramPlan::output).
  OutputBuffer(Bytes capacity, std::uint64_t bound) : capacity_(capacity), bound_(bound) {}

  // Room for a phase's worst case: `positions` positions in `bytes` wire
  // bytes. kOutputFull if the buffer lacks room: the phase does not start,
  // and production resumes when the client drains. kBeyondProgram if the
  // positions would pass the output bound: the phase must narrow.
  std::expected<void, ProgramError> Reserve(Bytes bytes, std::uint64_t positions);
  // The phase completed: `positions` positions of `bytes` wire bytes
  // become visible together. At most the reservation; the rest returns.
  std::expected<void, ProgramError> Publish(Bytes bytes, std::uint64_t positions);
  // The phase failed or was cancelled: nothing becomes visible.
  void Cancel() {
    reserved_ = Bytes();
    reserved_positions_ = 0;
  }
  // The client consumed published bytes.
  std::expected<void, ProgramError> Drain(Bytes bytes);

  Bytes capacity() const { return capacity_; }
  std::uint64_t bound() const { return bound_; }
  Bytes used() const { return used_; }
  Bytes reserved() const { return reserved_; }
  std::uint64_t published_positions() const { return published_positions_; }

 private:
  Bytes capacity_;
  std::uint64_t bound_ = 0;
  Bytes used_;      // published, not yet drained
  Bytes reserved_;  // the open reservation
  std::uint64_t reserved_positions_ = 0;
  std::uint64_t published_positions_ = 0;  // at most bound_
};

}  // namespace llmp::execution

#endif  // LLMP_EXECUTION_PROGRAM_H_
