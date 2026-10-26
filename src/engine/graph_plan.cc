// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/graph_plan.h"

#include <algorithm>
#include <format>
#include <limits>
#include <memory>
#include <new>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace llmp::engine {

namespace {
// Two retained allocations: the wrapper and the inventory's single exact
// array/string block. Each pays an allocator header/alignment quantum. The
// payment is embedded, so it creates no shared control block or temporary heap.
constexpr std::uint64_t kReportAllocationMetadataBytes = 64;
constexpr std::uint64_t kRetainedReportAllocations = 2;
struct ReportPayment {
  std::shared_ptr<const kernels::BackendOwner> owner;
  std::shared_ptr<const kernels::ConstructionFunder> funder;
  std::uint64_t bytes = 0;
  ReportPayment(std::shared_ptr<const kernels::BackendOwner> paid_owner,
                std::shared_ptr<const kernels::ConstructionFunder> paid_funder,
                std::uint64_t paid_bytes)
      : owner(std::move(paid_owner)), funder(std::move(paid_funder)), bytes(paid_bytes) {}
  ReportPayment(ReportPayment&& other) noexcept
      : owner(std::move(other.owner)),
        funder(std::move(other.funder)),
        bytes(std::exchange(other.bytes, 0)) {}
  ReportPayment(const ReportPayment&) = delete;
  ~ReportPayment() { Release(); }
  void Release() {
    if (owner)
      owner->Release(bytes);
    else if (funder)
      funder->release(bytes);
    funder.reset();
    owner.reset();
    bytes = 0;
  }
};
// The report follows exactly the returned program's lifetime and host charge.
// It forwards source-specific diagnostics without retaining the compiler graph.
class ReportedProgram final : public kernels::ForwardingStepProgram {
 public:
  ReportedProgram(std::unique_ptr<kernels::StepProgram> program,
                  planner::PlanReportInventory inventory, kernels::PlanCoverage coverage,
                  ReportPayment payment)
      : kernels::ForwardingStepProgram(std::move(program)),
        payment_(std::move(payment)),
        inventory_(std::move(inventory)),
        coverage_(coverage) {}
  ~ReportedProgram() override { ResetProgram(); }
  void ConstructionPaid() override {
    if (construction_transferred_) return;
    construction_transferred_ = true;
    payment_.Release();
    kernels::ForwardingStepProgram::ConstructionPaid();
  }
  std::uint64_t host_bytes() const override {
    return kernels::ForwardingStepProgram::host_bytes() + sizeof(*this) +
           inventory_.storage_bytes() + kRetainedReportAllocations * kReportAllocationMetadataBytes;
  }
  void StreamPlanReport(const kernels::PlanReportSink& sink) const override {
    if (!sink) return;
    inventory_.Stream(sink);
    kernels::ForwardingStepProgram::StreamPlanReport(sink);
  }
  // Convenience for callers that explicitly own assembly storage. Serving uses
  // ReportPlanRecords, which frames fragments without allocating a JSON string.
  void ReportPlan(const kernels::PlanReportSink& sink) const override {
    if (!sink) return;
    std::string record;
    ReportPlanRecords([&](kernels::PlanReportEvent event, std::string_view part) {
      if (event == kernels::PlanReportEvent::kBegin) record.clear();
      if (event == kernels::PlanReportEvent::kFragment) record.append(part);
      if (event == kernels::PlanReportEvent::kEnd) sink(record);
    });
  }
  kernels::PlanCoverage plan_coverage() const override {
    auto out = kernels::ForwardingStepProgram::plan_coverage();
    out.primitive_fallbacks += coverage_.primitive_fallbacks;
    out.available_fast_fallbacks += coverage_.available_fast_fallbacks;
    out.decode_expansion_rules += coverage_.decode_expansion_rules;
    out.missing_numerical_qualifications += coverage_.missing_numerical_qualifications;
    out.unqualified_numerical_candidates += coverage_.unqualified_numerical_candidates;
    return out;
  }

 private:
  // Payment dies last on a failed/uninstalled program, after all report bytes.
  ReportPayment payment_;
  planner::PlanReportInventory inventory_;
  kernels::PlanCoverage coverage_;
  bool construction_transferred_ = false;
};
}  // namespace

std::expected<planner::PlanOptions, std::string> PrepareComponentPlanOptions(
    const GraphBuildIO& io, const kernels::StepBackend& backend,
    const kernels::Addresses& addresses, planner::PlanOptions options) {
  auto prepared = PrepareComponentPlanOptionsInPlace(io, backend, addresses, options);
  if (!prepared) return std::unexpected(std::move(prepared.error()));
  return options;
}

std::expected<void, std::string> PrepareComponentPlanOptionsInPlace(
    const GraphBuildIO& io, const kernels::StepBackend& backend,
    const kernels::Addresses& addresses, planner::PlanOptions& options) {
  if (backend.registry == nullptr || !std::ranges::equal(io.inputs, io.graph.inputs()) ||
      !std::ranges::equal(io.outputs, io.graph.outputs()))
    return std::unexpected("a component graph needs its exact typed interface and registry");
  for (const auto& weight : io.weights) {
    if (!weight.tensor.valid() || weight.tensor.v >= io.graph.tensor_count())
      return std::unexpected("a component weight names no graph tensor");
    const auto& desc = io.graph.tensor(weight.tensor);
    const auto* ref = std::get_if<ir::WeightRef>(&desc.binding);
    if (desc.is_view() ||
        io.graph.node(io.graph.producer(weight.tensor)).key.op != ir::Op::kWeight ||
        ref == nullptr || *ref != weight.ref)
      return std::unexpected("a component weight differs from its graph root binding");
  }
  const auto power_of_two = [](std::uint64_t n) { return n != 0 && (n & (n - 1)) == 0; };
  if (addresses.activation_alignment != 0 && !power_of_two(addresses.activation_alignment))
    return std::unexpected("the activation reservation alignment is not a power of two");
  if (!addresses.measuring() && addresses.activation_alignment != 0 &&
      addresses.activations % addresses.activation_alignment != 0)
    return std::unexpected("the activation base breaks its reservation alignment promise");
  std::vector<std::pair<ir::TensorDesc, std::uint64_t>> physical_facts;
  if (addresses.state_alignment) {
    for (const auto& node : io.graph.nodes()) {
      if (node.key.op != ir::Op::kState) continue;
      const auto& desc = io.graph.tensor(node.out[0]);
      const auto* ref = std::get_if<ir::StateRef>(&desc.binding);
      if (ref == nullptr) continue;
      const auto alignment = addresses.state_alignment(*ref);
      if (alignment == 0) continue;
      if (!power_of_two(alignment))
        return std::unexpected("a state reservation alignment is not a power of two");
      const auto address = addresses.state ? addresses.state(*ref) : 0;
      if (address != 0 && address % alignment != 0)
        return std::unexpected("a state base breaks its reservation alignment promise");
      physical_facts.emplace_back(desc, alignment);
    }
  }
  if (addresses.weight_alignment) {
    for (const auto& weight : io.weights) {
      // A multipart logical root has no single physical address. Its parts
      // remain unknown here; no first-part alignment is promoted to the root.
      if (weight.parts.size() != 1) continue;
      const auto& desc = io.graph.tensor(weight.tensor);
      const auto parts = format::PartsOf(desc.format);
      if (!parts || parts->size() != 1 || (*parts)[0].role != weight.parts[0].second)
        return std::unexpected("a weight alignment does not name its single format part");
      const auto part = weight.parts[0].second;
      const auto alignment = addresses.weight_alignment(weight.ref, part);
      if (alignment == 0) continue;
      if (!power_of_two(alignment))
        return std::unexpected("a weight reservation alignment is not a power of two");
      const auto address = addresses.weight ? addresses.weight(weight.ref, part) : 0;
      if (address == 0 || address % alignment != 0)
        return std::unexpected("a weight base breaks its reservation alignment promise");
      physical_facts.emplace_back(desc, alignment);
    }
  }
  planner::RootAlignment root_alignment;
  if (!physical_facts.empty()) {
    const auto key_less = [](const ir::TensorDesc& a, const ir::TensorDesc& b) {
      return std::tie(a.role, a.segment, a.binding) < std::tie(b.role, b.segment, b.binding);
    };
    std::ranges::sort(physical_facts,
                      [&](const auto& a, const auto& b) { return key_less(a.first, b.first); });
    root_alignment = [facts = std::move(physical_facts), key_less](
                         const ir::Graph& graph, ir::TensorId root) -> std::uint64_t {
      if (!root.valid() || root.v >= graph.tensor_count()) return 0;
      const auto& desc = graph.tensor(root);
      const auto op = graph.node(graph.producer(root)).key.op;
      if (desc.is_view() || (op != ir::Op::kState && op != ir::Op::kWeight)) return 0;
      // Exact semantic leaf identity survives authorized rewrites. The format
      // also fixes the single part checked above; no unrelated binding gains it.
      auto it = std::ranges::lower_bound(
          facts, desc, key_less,
          [](const auto& entry) -> const ir::TensorDesc& { return entry.first; });
      for (; it != facts.end() && !key_less(desc, it->first); ++it) {
        const auto& [saved, alignment] = *it;
        if (desc.format == saved.format && desc.nb == saved.nb) return alignment;
      }
      return 0;
    };
  }
  options.root_alignment = std::move(root_alignment);
  options.activation_alignment = addresses.activation_alignment;
  return {};
}

std::expected<planner::StepPlan, std::string> PlanComponentGraph(
    const GraphBuildIO& io, const kernels::StepBackend& backend, const planner::Policy& policy,
    const kernels::Addresses& addresses, planner::PlanOptions options) {
  auto prepared = PrepareComponentPlanOptions(io, backend, addresses, std::move(options));
  if (!prepared) return std::unexpected(std::move(prepared.error()));
  auto plan = planner::Plan(io.graph.Clone(), *backend.registry, backend.device, policy, *prepared);
  if (!plan)
    return std::unexpected(std::format("planning a step ({}): {}",
                                       planner::PlanErrorName(plan.error().error),
                                       plan.error().detail));
  return std::move(*plan);
}

std::expected<std::unique_ptr<kernels::StepProgram>, std::string> LowerComponentPlan(
    const kernels::LowerInput& input, const kernels::StepBackend& backend,
    const planner::Policy& policy, const kernels::ProgramValidator& validate,
    std::shared_ptr<const kernels::BackendOwner> report_owner,
    const kernels::ExecutionResources* resources) {
  if (backend.registry == nullptr || !backend.lower)
    return std::unexpected("a component plan needs a backend registry and lowerer");
  if (resources &&
      (!*resources || !resources->parent() || &backend != &resources->parent()->backend()))
    return std::unexpected("private lowering resources differ from the canonical planning backend");
  if (!std::ranges::equal(input.io.inputs, input.io.graph.inputs()) ||
      !std::ranges::equal(input.io.outputs, input.io.graph.outputs()))
    return std::unexpected("a component plan needs its exact typed graph interface");
  for (const auto& weight : input.io.weights) {
    if (!weight.tensor.valid() || weight.tensor.v >= input.io.graph.tensor_count())
      return std::unexpected("a component plan weight names no graph tensor");
    const auto& desc = input.io.graph.tensor(weight.tensor);
    const auto* ref = std::get_if<ir::WeightRef>(&desc.binding);
    if (desc.is_view() ||
        input.io.graph.node(input.io.graph.producer(weight.tensor)).key.op != ir::Op::kWeight ||
        ref == nullptr || *ref != weight.ref)
      return std::unexpected("a component plan weight differs from its graph root binding");
  }
  if (input.plan.primitive_digest != input.io.graph.Digest() ||
      input.plan.graph_digest != input.plan.graph.Digest() ||
      input.plan.policy_digest != planner::Digest(policy))
    return std::unexpected("the supplied plan differs from its source, graph, or policy");
  if (input.token) {
    if (&input.token->step.graph != &input.io.graph ||
        input.token->step.inputs.size() != input.io.inputs.size() ||
        input.token->step.weights.size() != input.io.weights.size())
      return std::unexpected("token lowering metadata differs from the supplied graph interface");
    for (const auto tensor : input.io.inputs) {
      const auto matches =
          std::ranges::count(input.token->step.inputs, tensor, &components::StepInput::tensor);
      if (matches != 1)
        return std::unexpected("token input ports differ from the supplied typed graph interface");
    }
    for (const auto& graph_weight : input.io.weights) {
      const auto graph_id_count = std::ranges::count(input.io.weights, graph_weight.tensor,
                                                     &components::StepWeight::tensor);
      const auto token_id_count = std::ranges::count(input.token->step.weights, graph_weight.tensor,
                                                     &components::StepWeight::tensor);
      const auto matches = std::ranges::count_if(input.token->step.weights, [&](const auto& item) {
        return item.tensor == graph_weight.tensor && item.ref == graph_weight.ref &&
               item.parts == graph_weight.parts;
      });
      if (graph_id_count != 1 || token_id_count != 1 || matches != 1)
        return std::unexpected("token weight ports differ from the supplied typed graph interface");
    }
  }

  auto lowered = resources ? resources->Lower(input) : backend.lower(input);
  if (!lowered) return lowered;
  if (!*lowered) return std::unexpected("the backend returned no step program");
  if (validate) {
    auto valid = validate(input, **lowered);
    if (!valid) return std::unexpected(valid.error());
  }
  const auto& plan = input.plan;
  auto inventory_bytes = planner::PlanReportInventory::StorageBytes(plan, *backend.registry);
  if (!inventory_bytes) return std::unexpected(inventory_bytes.error());
  std::uint64_t payment_bytes = 0;
  if (__builtin_add_overflow(
          *inventory_bytes,
          static_cast<std::uint64_t>(sizeof(ReportedProgram) +
                                     kRetainedReportAllocations * kReportAllocationMetadataBytes),
          &payment_bytes))
    return std::unexpected("the plan report inventory payment overflows");
  if ((*lowered)->host_bytes() > std::numeric_limits<std::uint64_t>::max() - payment_bytes)
    return std::unexpected("the retained program and report host bytes overflow");
  if (!report_owner) report_owner = backend.owner;
  auto report_funder = report_owner ? nullptr : backend.construction_funder;
  if (!report_owner && (!report_funder || !report_funder->charge || !report_funder->release))
    return std::unexpected("the plan report inventory needs a paid construction funder");
  if (report_owner ? !report_owner->Charge(payment_bytes) : !report_funder->charge(payment_bytes))
    return std::unexpected("the plan report inventory could not be funded");
  // Declared before inventory: failure destroys retained bytes before release.
  ReportPayment payment(std::move(report_owner), std::move(report_funder), payment_bytes);
  auto inventory = planner::PlanReportInventory::Capture(plan, *backend.registry, *inventory_bytes);
  if (!inventory) return std::unexpected(inventory.error());
  kernels::PlanCoverage coverage{
      .primitive_fallbacks = plan.decomposed.size(),
      .missing_numerical_qualifications = plan.missing_numerical.size(),
      .unqualified_numerical_candidates = static_cast<std::uint64_t>(
          std::ranges::count(plan.applied, true, &planner::Applied::numerical_candidate))};
  for (const auto& key : plan.decomposed) {
    for (std::size_t i = 0; i < backend.registry->size(); ++i) {
      const auto& impl = backend.registry->at(i);
      if (impl.capability && impl.capability->rung != execution::Rung::kPrimitive &&
          ir::OpKeyName(impl.capability->op) == key) {
        ++coverage.available_fast_fallbacks;
        break;
      }
    }
  }
  if (input.token && input.token->shape.rows() <= policy.max_decode_rows) {
    const planner::SelectionView view(*backend.registry, backend.device, policy, false, false);
    for (const auto& selected : plan.steps) {
      const auto& impl = backend.registry->at(selected.implementation);
      if (!impl.capability || impl.capability->expansion == execution::Expansion::kNone) continue;
      for (const auto node : selected.nodes)
        view.Inspect(plan.graph, plan.graph.node(node), [&](const execution::CapabilityQuery& q) {
          coverage.decode_expansion_rules += execution::Expands(*impl.capability, q);
        });
    }
  }
  auto* wrapper = new (std::nothrow)
      ReportedProgram(std::move(*lowered), std::move(*inventory), coverage, std::move(payment));
  if (!wrapper) return std::unexpected("allocating the paid plan report wrapper failed");
  std::unique_ptr<kernels::StepProgram> result(wrapper);
  if (resources && !result->HoldResources(*resources))
    return std::unexpected("the paid report wrapper lost private resource identity");
  return result;
}

std::expected<std::unique_ptr<kernels::StepProgram>, std::string>
detail::BuildOrdinaryComponentProgram(const GraphBuildIO& io, const kernels::StepBackend& backend,
                                      const planner::Policy& policy,
                                      const kernels::Addresses& addresses,
                                      planner::PlanOptions options,
                                      std::optional<kernels::TokenLowerInput> token,
                                      const kernels::ProgramValidator& validate,
                                      std::shared_ptr<const kernels::BackendOwner> report_owner,
                                      const kernels::ExecutionResources* resources) {
  if (!backend.lower) return std::unexpected("a component graph needs a backend lowerer");
  if (resources &&
      (!*resources || !resources->parent() || &backend != &resources->parent()->backend()))
    return std::unexpected("private lowering resources differ from the canonical planning backend");
  if (token && (&token->step.graph != &io.graph || options.shape != &token->shape))
    return std::unexpected("token lowering metadata differs from the planned graph or shape");
  if (!token && (options.shape != nullptr || options.state_spans))
    return std::unexpected("a spatial graph cannot borrow token shape or state facts");
  std::optional<planner::OwnerProducts> proved_owners;
  if (token && !token->step.owner_products.empty() && options.owner_products == nullptr) {
    if (token->shape.segments.empty() || token->shape.segments.size() > components::kMaxSegments)
      return std::unexpected("owner products: missing or excessive segments");
    std::vector<components::SegmentRows> segments;
    std::uint64_t first = 0;
    for (const auto& s : token->shape.segments) {
      if (first > std::numeric_limits<std::uint32_t>::max())
        return std::unexpected("owner products: row partition overflow");
      segments.push_back({.slot = s.slot,
                          .first_row = static_cast<std::uint32_t>(first),
                          .rows = s.rows,
                          .past = s.past,
                          .read_cells = s.read_cells,
                          .outputs = s.outputs});
      first += s.rows;
    }
    auto proved = planner::ProveOwnerProducts(io.graph, token->step.inputs, segments,
                                              token->step.owner_products);
    if (!proved) return std::unexpected(std::move(proved.error()));
    proved_owners = std::move(*proved);
    options.owner_products = &*proved_owners;
  }
  auto plan = PlanComponentGraph(io, backend, policy, addresses, std::move(options));
  if (!plan) return std::unexpected(std::move(plan.error()));
  const kernels::LowerInput input{*plan, io, addresses, token};
  return LowerComponentPlan(input, backend, policy, validate, std::move(report_owner), resources);
}

std::expected<std::unique_ptr<kernels::StepProgram>, std::string> BuildComponentProgram(
    const GraphBuildIO& io, const kernels::StepBackend& backend, const planner::Policy& policy,
    const kernels::Addresses& addresses, planner::PlanOptions options,
    std::optional<kernels::TokenLowerInput> token, const kernels::ProgramValidator& validate,
    std::shared_ptr<const kernels::BackendOwner> report_owner,
    const kernels::ExecutionResources* resources) {
  if (!token && !options.shape && !options.state_spans && backend.registry &&
      !backend.registry->ranks().recipe_comparisons.empty() && !addresses.measuring())
    return BuildCumulativeRecipeProgram(io, backend, policy, addresses, std::move(options),
                                        validate, std::move(report_owner), resources);
  return detail::BuildOrdinaryComponentProgram(io, backend, policy, addresses, std::move(options),
                                               token, validate, std::move(report_owner), resources);
}

}  // namespace llmp::engine
