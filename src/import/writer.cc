// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/writer.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <format>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>

#include "artifact/composition.h"
#include "artifact/format_json.h"
#include "artifact/schema.h"
#include "base/gguf.h"
#include "base/json.h"
#include "components/component.h"
#include "components/step.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "import/indexed_parts.h"
#include "import/route.h"
#include "import/stream.h"
#include "platform/files.h"
#include "platform/lock_file.h"
#include "platform/sockets.h"
#include "spec/mapping.h"
#include "spec/stored.h"

namespace llmp::import {
namespace {
namespace fs = std::filesystem;
using FD = artifact::FileDescriptor;
constexpr std::uint64_t kShardTarget = std::uint64_t{4} << 30U;
constexpr std::uint64_t kMaxOutput = std::uint64_t{1} << 50U;
std::unexpected<Error> Bad(std::string_view reason, Rule rule = Rule::kIo) {
  return std::unexpected(Error{.rule = rule, .reason = reason});
}
std::string Q(std::string_view value) {
  std::string result;
  base::json::AppendQuoted(value, result);
  return result;
}
std::string N(std::uint64_t value) { return std::to_string(value); }
std::string A(const std::vector<std::string>& values) {
  std::string text = "[";
  for (const auto& value : values) {
    if (text.size() > 1) text += ',';
    text += value;
  }
  return text + ']';
}
std::string O(std::map<std::string, std::string> values) {
  std::string text = "{";
  for (const auto& [key, value] : values) {
    if (text.size() > 1) text += ',';
    text += Q(key) + ':' + value;
  }
  return text + '}';
}
std::string Hex(std::string_view bytes) {
  return base::ToHex(base::Sha256().Update(bytes).Finish());
}
std::uint64_t Align(std::uint64_t n, std::uint64_t alignment) {
  return (n + alignment - 1) / alignment * alignment;
}
bool SafeName(std::string_view name) { return artifact::schema::IsFileName(name); }
std::expected<void, Error> Write(int fd, std::uint64_t at, std::span<const std::byte> bytes) {
  while (!bytes.empty()) {
    const auto n = ::pwrite(fd, bytes.data(), bytes.size(), static_cast<off_t>(at));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return Bad("cannot write an artifact file");
    at += static_cast<std::uint64_t>(n);
    bytes = bytes.subspan(static_cast<std::size_t>(n));
  }
  return {};
}
void SweepOrphanedStagingJobs(const fs::path& staging) {
  std::error_code error;
  for (const auto& entry : fs::directory_iterator(staging, error)) {
    if (error) break;
    if (entry.is_directory(error) && entry.path().filename().string().starts_with("job-")) {
      std::error_code remove_err;
      fs::remove_all(entry.path(), remove_err);
    }
  }
}
std::expected<FD, Error> Create(const fs::path& path) {
  FD file(::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0644));
  if (!file.valid()) return Bad("cannot create an artifact staging file");
  if (::fchmod(file.get(), 0644) != 0) return Bad("cannot set artifact file permissions");
  return file;
}
std::expected<void, Error> Save(const fs::path& path, std::string_view text) {
  auto file = Create(path);
  if (!file) return std::unexpected(file.error());
  if (auto ok = Write(file->get(), 0, std::as_bytes(std::span(text))); !ok) return ok;
  if (::fsync(file->get()) != 0) return Bad("cannot sync an artifact file");
  return {};
}
std::expected<void, Error> Sync(const fs::path& path) {
  FD file(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (!file.valid() || ::fsync(file.get()) != 0) return Bad("cannot sync an artifact directory");
  return {};
}
struct Cleanup {
  fs::path path{};
  ~Cleanup() {
    std::error_code ignored;
    if (!path.empty()) fs::remove_all(path, ignored);
  }
};
struct Member {
  std::uint32_t tensor = 0;
  std::optional<std::uint32_t> expert{};
  std::uint64_t offset = 0, bytes = 0, readable = 0;
  format::TensorFormat format{};
};
struct Group {
  std::string kind{};
  std::optional<std::uint32_t> layer{}, expert{};
  std::uint32_t shard = 0;
  std::uint64_t offset = 0, used = 0, stored = 0, first_chunk = 0;
  std::vector<Member> members{};
};
struct Shard {
  std::string path{}, header{};
  std::uint64_t bytes = 0;
};
// Borrowed only from private owned normalization products. This type and its
// construction stay inside the common writer; it is not normalization authority.
struct PublicationView {
  const Checkpoint& checkpoint;
  const spec::Source& canonical_source;
  const spec::ModelSpec& model_spec;
  std::uint64_t payload = 0;
  const NormalizedPlan* copy = nullptr;
  const IndexedPartPlan* parts = nullptr;
  std::span<const CopyRoute> copy_routes{};
  std::span<const NormalizedPartBinding> bindings{};
  std::span<const SemanticReceipt> semantic{};
  std::span<const PlannedPartFold> planned_folds{};
  // The trailing canonical tensors, each one whole part the importer derived
  // from one source file whose whole content is its parsed receipt.
  std::span<const GeneratedPart> generated{};
  // Each canonical tensor's group where the spec's bindings do not say
  // (ComponentPlan::groups); empty: by the spec.
  std::span<const ComponentGroup> groups{};
  std::string_view mapping_architecture{};
  std::uint16_t mapping_version = 0;
  const Checkpoint& source() const { return checkpoint; }
  const spec::Source& canonical() const { return canonical_source; }
  const spec::ModelSpec& spec() const { return model_spec; }
  std::size_t members() const {
    if (copy) return copy->routes().size();
    if (!copy_routes.empty()) return copy_routes.size();
    return parts ? parts->members().size() : 0;
  }
};
bool IsDrafter(const spec::ModelSpec& model) {
  return model.stages.size() == 2 && model.stages[1].role == spec::StageRole::kDrafter &&
         model.stages[1].artifact == 1;
}
struct Copy {
  // `source_tensor`: the checkpoint's tensor it streams, whose block scales
  // are checked as they stream (format::FirstUndefinedScale).
  std::uint32_t file = 0, tensor = 0, source_tensor = 0;
  // `span`: the output range its writes lie in (an interleaved lane's rows,
  // beside its sibling lanes; otherwise its output bytes).
  std::uint64_t source = 0, destination = 0, bytes = 0, output_bytes = 0, span = 0;
  std::optional<PartRouteExecution> route{};
};
struct Destination {
  std::uint32_t tensor = 0, shard = 0;
  std::uint64_t begin = 0, end = 0, output = 0;
};
struct SemanticCheck {
  const SemanticReceipt* receipt = nullptr;
  std::uint64_t received = 0;
  base::Sha256 hash{};
};
static_assert(format::PartList::kMax <= 4);
static_assert(sizeof(Copy) <= 256 && sizeof(Destination) <= 64 && sizeof(SemanticCheck) <= 256);
struct Layout {
  std::uint64_t metadata_bytes = 0;
  std::vector<Group> groups{};
  std::vector<Shard> shards{};
  // All route cursors borrow this stable allocation; destruction retires the
  // cursors first. Serial source callbacks share it without payload retention.
  std::vector<std::byte> workspace{};
  std::vector<Copy> copies{};
  std::vector<Destination> destinations{};
  std::vector<std::size_t> first_destination{};
  std::vector<SemanticCheck> semantic{};
  std::vector<artifact::AppliedFold> folds{};
  std::vector<artifact::Relayout> relayouts{};
  std::uint32_t experts = 0;
};
struct GroupKey {
  // Table, global, each layer followed by its experts, head.
  std::uint32_t order = 0, layer = 0;
  auto operator<=>(const GroupKey&) const = default;
};
std::expected<void, Error> Transfers(const PublicationView& plan, Layout& layout) {
  const auto& source = plan.source();
  const auto& tensors = plan.canonical().tensors;
  const auto valid_extent = [&](const Extent& e) {
    return e.file < source.files.size() && e.bytes != 0 && e.offset <= source.files[e.file].bytes &&
           e.bytes <= source.files[e.file].bytes - e.offset;
  };
  std::map<std::pair<std::uint32_t, format::PartRole>, std::uint32_t> binding;
  std::vector<std::uint8_t> used(source.tensors.size(), 0);
  const std::size_t bound = tensors.size() - std::min(tensors.size(), plan.generated.size());
  if (!plan.generated.empty() && (!plan.parts || bound + plan.generated.size() != tensors.size()))
    return Bad("generated parts need a part publication", Rule::kConfig);
  for (std::size_t g = 0; g < plan.generated.size(); ++g) {
    const auto& gen = plan.generated[g];
    const auto parts = format::PartsOf(tensors[bound + g].format);
    const auto part = parts ? parts->Find(tensors[bound + g].part) : std::nullopt;
    if (gen.tensor != bound + g || tensors[gen.tensor].experts || !part ||
        part->bytes != gen.bytes.size() || gen.file >= source.files.size() ||
        source.files[gen.file].parsed.bytes != source.files[gen.file].bytes ||
        std::ranges::any_of(source.tensors,
                            [&](const Tensor& t) { return t.extent.file == gen.file; }))
      return Bad("generated part identity, size or source file", Rule::kConfig);
  }
  if (plan.parts) {
    if (plan.bindings.size() != bound || plan.parts->dispositions().size() != source.tensors.size())
      return Bad("part publication binding or disposition count", Rule::kConfig);
    for (std::size_t i = 0; i < bound; ++i) {
      const auto& b = plan.bindings[i];
      if (b.output >= plan.parts->outputs().size())
        return Bad("part publication binding index", Rule::kConfig);
      const auto& output = plan.parts->outputs()[b.output];
      if (tensors[i].format != output.format || tensors[i].part != b.part ||
          !output.parts.Find(b.part) ||
          !binding.emplace(std::pair(b.output, b.part), static_cast<std::uint32_t>(i)).second)
        return Bad("part publication binding geometry or duplicate", Rule::kConfig);
    }
    std::set<std::string_view> fold_names;
    std::size_t folded_parts = 0;
    for (const auto& receipt : plan.planned_folds) {
      if (!spec::FindFold(receipt.fold) || !fold_names.insert(receipt.fold).second ||
          receipt.outputs.empty())
        return Bad("publication fold name or duplicate", Rule::kConfig);
      // The stages that run from this artifact: a drafter's alone (stage 1,
      // artifact 1), beside its target's carrier.
      const bool drafter = IsDrafter(plan.spec());
      for (const auto& stage : plan.spec().stages)
        if (stage.artifact == (drafter ? 1U : 0U) &&
            !std::ranges::contains(stage.folds, receipt.fold))
          return Bad("publication fold absent from stage", Rule::kConfig);
      artifact::AppliedFold fold{.name = receipt.fold, .tensors = {}};
      std::set<std::uint32_t> folded_outputs;
      for (const auto output : receipt.outputs) {
        if (output >= plan.parts->outputs().size() || !folded_outputs.insert(output).second)
          return Bad("publication fold output index or duplicate", Rule::kConfig);
        for (const auto& part : plan.parts->outputs()[output].parts.parts()) {
          const auto at = binding.find({output, part.role});
          if (at == binding.end() || ++folded_parts > artifact::kMaxEntries)
            return Bad("publication fold part coverage", Rule::kBounds);
          fold.tensors.push_back(tensors[at->second].name);
        }
      }
      std::ranges::sort(fold.tensors);
      layout.folds.push_back(std::move(fold));
    }
    // Relayouts (D-113): every part of each relaid-out output, by the
    // layout its sources held.
    for (std::uint32_t o = 0; o < plan.parts->outputs().size(); ++o) {
      const auto& output = plan.parts->outputs()[o];
      if (!output.relayout_from) continue;
      auto found =
          std::ranges::find(layout.relayouts, *output.relayout_from, &artifact::Relayout::from);
      if (found == layout.relayouts.end()) {
        layout.relayouts.push_back({.from = *output.relayout_from, .tensors = {}});
        found = std::prev(layout.relayouts.end());
      }
      for (const auto& part : output.parts.parts()) {
        const auto at = binding.find({o, part.role});
        if (at == binding.end()) return Bad("publication relayout part coverage", Rule::kConfig);
        found->tensors.push_back(tensors[at->second].name);
      }
    }
    for (auto& relayout : layout.relayouts) std::ranges::sort(relayout.tensors);
    std::size_t required = 0;
    for (const auto& output : plan.parts->outputs()) required += output.parts.size();
    if (binding.size() != required)
      return Bad("part publication omits an output part", Rule::kConfig);
  }
  layout.workspace.resize(65536);
  layout.copies.reserve(plan.members());
  const auto add = [&](std::uint32_t si, std::uint32_t ti, Extent extent, std::uint64_t destination,
                       const PartRoutePlan& route) -> std::expected<void, Error> {
    if (si >= source.tensors.size() || ti >= tensors.size() || used[si]++ != 0 ||
        source.tensors[si].extent != extent || !valid_extent(extent) ||
        route.input_bytes() != extent.bytes ||
        (plan.parts && plan.parts->dispositions()[si].kind != SourceDispositionKind::kRouted))
      return Bad("part publication source identity or duplicate", Rule::kConfig);
    const auto parts = format::PartsOf(tensors[ti].format);
    const auto part = parts ? parts->Find(tensors[ti].part) : std::nullopt;
    if (!part || destination > part->bytes || route.span() > part->bytes - destination)
      return Bad("part publication route destination bounds", Rule::kConfig);
    auto cursor = route.Start(layout.workspace);
    if (!cursor) return std::unexpected(cursor.error());
    layout.copies.push_back({extent.file, ti, si, extent.offset, destination, extent.bytes,
                             route.output_bytes(), route.span(), std::move(*cursor)});
    return {};
  };
  const auto routes = plan.copy ? plan.copy->routes() : plan.copy_routes;
  if (!routes.empty()) {
    if (routes.size() != tensors.size()) return Bad("copy publication route count", Rule::kConfig);
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      const auto& route = routes[i];
      auto checked = PartRoutePlan::Copy(route.extent.bytes);
      if (!checked) return std::unexpected(checked.error());
      if (auto ok =
              add(route.source_tensor, static_cast<std::uint32_t>(i), route.extent, 0, *checked);
          !ok)
        return ok;
    }
  } else {
    for (const auto& member : plan.parts->members()) {
      const auto at = binding.find({member.output, member.part});
      if (at == binding.end()) return Bad("part publication member lacks binding", Rule::kConfig);
      if (auto ok = add(member.source, at->second, member.extent, member.destination, member.route);
          !ok)
        return ok;
    }
    layout.semantic.reserve(plan.semantic.size());
    for (const auto& receipt : plan.semantic) {
      if (receipt.source >= source.tensors.size() || used[receipt.source]++ != 0 ||
          source.tensors[receipt.source].extent != receipt.extent ||
          !valid_extent(receipt.extent) ||
          plan.parts->dispositions()[receipt.source].kind != SourceDispositionKind::kSemantic)
        return Bad("semantic publication source receipt", Rule::kConfig);
      layout.semantic.push_back({.receipt = &receipt});
    }
    for (std::size_t i = 0; i < used.size(); ++i) {
      const auto kind = plan.parts->dispositions()[i].kind;
      if (used[i] != (kind == SourceDispositionKind::kAuxiliary ? 0 : 1))
        return Bad("part publication source disposition coverage", Rule::kConfig);
    }
  }
  // Independent complete destination coverage. Whole route executions retain
  // their own permutation/scalar state through every expert destination slice.
  std::vector<std::size_t> order(layout.copies.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::sort(order, [&](auto a, auto b) {
    const auto& x = layout.copies[a];
    const auto& y = layout.copies[b];
    return std::tie(x.tensor, x.destination) < std::tie(y.tensor, y.destination);
  });
  std::size_t next = 0;
  for (std::size_t ti = 0; ti < bound; ++ti) {
    const auto parts = format::PartsOf(tensors[ti].format);
    const auto part = parts ? parts->Find(tensors[ti].part) : std::nullopt;
    if (!part) return Bad("part publication output format", Rule::kConfig);
    std::uint64_t at = 0;
    while (next < order.size() && layout.copies[order[next]].tensor == ti) {
      // Copies at one destination are lanes of one span (an interleaved
      // table's codes and scales), their writes covering it exactly.
      const auto& c = layout.copies[order[next++]];
      if (c.destination != at || c.span > part->bytes - at)
        return Bad("part publication output gap or overlap", Rule::kConfig);
      std::uint64_t written = c.output_bytes;
      while (next < order.size() && layout.copies[order[next]].tensor == ti &&
             layout.copies[order[next]].destination == at) {
        const auto& lane = layout.copies[order[next++]];
        if (lane.span != c.span || lane.output_bytes > c.span - written)
          return Bad("part publication lanes disagree on their span", Rule::kConfig);
        written += lane.output_bytes;
      }
      if (written != c.span) return Bad("part publication output gap or overlap", Rule::kConfig);
      at += c.span;
    }
    if (at != part->bytes) return Bad("part publication incomplete output", Rule::kConfig);
  }
  if (next != order.size()) return Bad("part publication extra output", Rule::kConfig);
  std::ranges::sort(layout.copies, {}, [](const Copy& c) { return std::tie(c.file, c.source); });
  std::ranges::sort(layout.semantic, {}, [](const SemanticCheck& c) {
    return std::tie(c.receipt->extent.file, c.receipt->extent.offset);
  });
  return {};
}
std::expected<Layout, Error> Place(const PublicationView& plan, std::uint64_t funding) {
  const auto& tensors = plan.canonical().tensors;
  if (tensors.empty() || tensors.size() > artifact::kMaxEntries ||
      plan.source().tensors.size() > kMaxTensors || plan.source().files.empty() ||
      plan.source().files.size() > kMaxSourceFiles || plan.members() > kMaxTensors ||
      plan.semantic.size() > plan.source().tensors.size() ||
      plan.planned_folds.size() > spec::Folds().size())
    return Bad("invalid normalized output count", Rule::kBounds);
  std::uint64_t fold_outputs = 0;
  for (const auto& fold : plan.planned_folds) {
    if (fold.outputs.size() > artifact::kMaxEntries - fold_outputs)
      return Bad("publication fold metadata exceeds cap", Rule::kBounds);
    fold_outputs += fold.outputs.size();
  }
  std::uint64_t expanded = 0, output_bytes = 0;
  for (const auto& t : tensors) {
    const auto count = t.experts ? t.format.shape.ne[2] : 1;
    if (count > artifact::kMaxEntries - expanded)
      return Bad("expert expansion exceeds the artifact entry cap", Rule::kBounds);
    expanded += count;
    const auto parts = format::PartsOf(t.format);
    const auto part = parts ? parts->Find(t.part) : std::nullopt;
    if (!part || part->bytes > kMaxOutput - output_bytes)
      return Bad("normalized output byte bound", Rule::kBounds);
    output_bytes += part->bytes;
  }
  // Explicit conservative storage ceiling includes JSON serialization, layout,
  // maps, parsed-prefix buffers, stream buffers and the artifact reader.
  const std::uint64_t payload = std::max(plan.payload, output_bytes);
  if (payload > kMaxOutput) return Bad("source payload exceeds output cap", Rule::kBounds);
  const std::uint64_t chunk_bound = payload / artifact::kChunkBytes + expanded * 2;
  if (chunk_bound > artifact::kMaxIndexBytes / 128)
    return Bad("output chunk metadata exceeds index cap", Rule::kBounds);
  const std::uint64_t bound = (std::uint64_t{256} << 20U) + expanded * 8192 + chunk_bound * 512 +
                              plan.source().files.size() * 1024 + plan.members() * 512 +
                              plan.source().tensors.size() * 32 + plan.semantic.size() * 256 +
                              fold_outputs * 8192 + 65536;
  if (funding < bound) return Bad("artifact metadata has not been funded", Rule::kBounds);
  const bool is_drafter = IsDrafter(plan.spec());
  if (!is_drafter) {
    if (!components::ValidateSpec(plan.spec(), tensors, components::ValidationPurpose::kImport))
      return Bad("publication spec does not bind canonical tensors", Rule::kConfig);
  }
  std::map<std::string, GroupKey, std::less<>> owners;
  const auto bind = [&](const spec::ComponentRef& component, GroupKey key) {
    for (const auto& w : component.weights) {
      if (is_drafter && !w.stage.empty()) continue;
      auto [it, inserted] = owners.try_emplace(w.tensor, key);
      if (!inserted && it->second != key) {
        // A tied embedding remains a row table; other shared weights are global.
        if (key.order == 0 || it->second.order == 0)
          it->second = {0, 0};
        else
          it->second = {1, 0};
      }
    }
  };
  const std::span<const spec::Stage> stages_to_bind =
      is_drafter ? std::span<const spec::Stage>(&plan.spec().stages[1], 1)
                 : std::span<const spec::Stage>(plan.spec().stages);
  if (!plan.groups.empty()) {
    if (plan.groups.size() != tensors.size())
      return Bad("component groups do not name every tensor", Rule::kConfig);
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      const auto& g = plan.groups[i];
      owners.try_emplace(tensors[i].name,
                         GroupKey{static_cast<std::uint32_t>(g.kind),
                                  g.kind == ComponentGroup::Kind::kLayer ? g.layer : 0U});
    }
  }
  for (const auto& stage : stages_to_bind) {
    if (!plan.groups.empty()) break;
    if (stage.embedding) bind(*stage.embedding, {0, 0});
    if (stage.stream.mixing) bind(*stage.stream.mixing, {1, 0});
    for (std::size_t i = 0; i < stage.side.size(); ++i) {
      const auto& side = stage.side[i];
      // On-demand side rows must not share an unbacked group with the ordinary
      // token embedding (or another table). Keep each side's ownership intact.
      const auto* def = components::Find(side.source.kind, side.source.version);
      const bool rows =
          def && std::ranges::any_of(def->roles, &components::WeightRole::rows_on_demand);
      bind(side.source, {rows ? 0U : 1U, rows ? static_cast<std::uint32_t>(i + 1) : 0U});
    }
    for (std::size_t i = 0; i < stage.layers.size(); ++i)
      for (const auto& c : stage.layers[i].components) bind(c, {2, static_cast<std::uint32_t>(i)});
    if (stage.heads.final_norm) bind(*stage.heads.final_norm, {3, 0});
    if (stage.heads.output) bind(*stage.heads.output, {3, 0});
    for (const auto& c : stage.heads.mtp) bind(c, {3, 0});
  }
  const spec::Mapping* mapping_ptr =
      !plan.mapping_architecture.empty()
          ? spec::FindMapping(plan.mapping_architecture, plan.source().container,
                              plan.mapping_version ? plan.mapping_version : 1)
          : nullptr;
  if (!mapping_ptr && !plan.mapping_architecture.empty())
    mapping_ptr = spec::FindMapping(plan.mapping_architecture);
  if (!mapping_ptr)
    mapping_ptr = spec::FindMapping(plan.source().architecture, plan.source().container);
  if (!mapping_ptr) mapping_ptr = spec::FindMapping(plan.source().architecture);
  if (mapping_ptr) {
    const auto& mapping = *mapping_ptr;
    for (const auto& r : mapping.tensors) {
      if (!r.unbound) continue;
      GroupKey key{3, 0};
      for (const auto& slot : mapping.slots) {
        if (slot.slot == r.slot) {
          switch (slot.site) {
            case spec::Site::kEmbedding:
              key = {0, 0};
              break;
            case spec::Site::kMixing:
            case spec::Site::kSide:
              key = {1, 0};
              break;
            case spec::Site::kFinalNorm:
            case spec::Site::kOutput:
              key = {3, 0};
              break;
            case spec::Site::kLayer:
              key = {2, 0};
              break;
          }
          break;
        }
      }
      if (r.pattern.contains("{layer}")) {
        const std::size_t layers =
            stages_to_bind.empty() ? 0 : stages_to_bind.front().layers.size();
        for (std::uint32_t l = 0; l < layers; ++l) {
          std::string name = std::string(r.pattern);
          const auto pos = name.find("{layer}");
          if (pos != std::string::npos) name.replace(pos, 7, std::to_string(l));
          owners.try_emplace(std::move(name), GroupKey{key.order == 2 ? 2U : key.order,
                                                       key.order == 2 ? l : key.layer});
        }
      } else {
        owners.try_emplace(std::string(r.pattern), key);
      }
    }
  }
  std::map<GroupKey, std::vector<std::uint32_t>> batches;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    if (!SafeName(tensors[i].name))
      return Bad("a tensor name is outside the artifact vocabulary", Rule::kUnsupported);
    const auto found = owners.find(tensors[i].name);
    if (found == owners.end()) return Bad("a normalized tensor is unbound", Rule::kConfig);
    batches[found->second].push_back(static_cast<std::uint32_t>(i));
  }
  Layout out{.metadata_bytes = bound};
  std::uint64_t total = 0, next_chunk = 0;
  for (auto& [key, indices] : batches) {
    std::ranges::sort(indices, [&](auto a, auto b) { return tensors[a].name < tensors[b].name; });
    std::vector<std::uint32_t> dense, experts;
    for (auto i : indices) (tensors[i].experts ? experts : dense).push_back(i);
    // In an expert group a stacked table (a MoE's gate and up) precedes its
    // other tables, each table's parts in name order: the slot the block-
    // scaled grouped GEMM reads at fixed offsets (kernels/ggml/moe_layout.h
    // ExpertLayout; the runtime's admission checks them).
    std::ranges::stable_sort(experts, [&](auto a, auto b) {
      return tensors[a].format.stacked > tensors[b].format.stacked;
    });
    if (!experts.empty()) {
      if (key.order != 2) return Bad("an expert array has no unique layer", Rule::kUnsupported);
      const auto count = tensors[experts.front()].format.shape.ne[2];
      if (count == 0 || count > artifact::kMaxExperts || (out.experts && out.experts != count))
        return Bad("expert counts disagree", Rule::kConfig);
      out.experts = static_cast<std::uint32_t>(count);
    }
    const auto add = [&](const std::vector<std::uint32_t>& members,
                         std::optional<std::uint32_t> expert) -> std::expected<void, Error> {
      if (members.empty()) return {};
      Group g{.kind = expert           ? "expert"
                      : key.order == 0 ? "table"
                      : key.order == 1 ? "global"
                      : key.order == 2 ? "layer"
                                       : "head",
              .layer = key.order == 2 ? std::optional(key.layer) : std::nullopt,
              .expert = expert};
      for (auto i : members) {
        const auto& t = tensors[i];
        auto format = t.format;
        if (expert) {
          if (format.shape.rank != 3 || format.shape.ne[2] != out.experts)
            return Bad("an expert array has inconsistent geometry", Rule::kConfig);
          format.shape.rank = 2;
          format.shape.ne[2] = 1;
        }
        const auto parts = format::PartsOf(format);
        if (!parts) return Bad("an output format is invalid", Rule::kConfig);
        const auto part = parts->Find(t.part);
        if (!part) return Bad("an output format lacks its part", Rule::kConfig);
        if (g.used > kMaxOutput - 256 || part->readable > kMaxOutput - g.used - 256)
          return Bad("artifact output exceeds its byte cap", Rule::kBounds);
        const auto offset = Align(g.used, artifact::kMemberAlignment);
        g.members.push_back({i, expert, offset, part->bytes, part->readable, format});
        g.used = offset + part->readable;
      }
      g.stored = Align(g.used, artifact::kFileAlignment);
      if (g.stored > kMaxOutput - total)
        return Bad("artifact output exceeds its byte cap", Rule::kBounds);
      total += g.stored;
      if (out.shards.empty() ||
          (out.shards.back().bytes && out.shards.back().bytes + g.stored > kShardTarget)) {
        if (out.shards.size() >= 99999) return Bad("too many artifact shards", Rule::kBounds);
        out.shards.push_back({.path = std::format("data/{:05}.safetensors", out.shards.size())});
      }
      g.shard = static_cast<std::uint32_t>(out.shards.size() - 1);
      g.offset = out.shards.back().bytes;
      out.shards.back().bytes += g.stored;
      g.first_chunk = next_chunk;
      next_chunk += (g.stored + artifact::kChunkBytes - 1) / artifact::kChunkBytes;
      out.groups.push_back(std::move(g));
      return {};
    };
    // A row table of several parts (a split encoding's codes, scales and
    // tensor scale): each part its own group, since a row-paged part owns
    // its group alone (engine/model_runner.h RowTable). One-part tables and
    // every other kind keep one group.
    if (key.order == 0 && dense.size() > 1 && std::ranges::any_of(dense, [&](auto i) {
          return tensors[i].part != format::PartRole::kData;
        })) {
      for (const auto i : dense)
        if (auto ok = add({i}, std::nullopt); !ok) return std::unexpected(ok.error());
    } else if (auto ok = add(dense, std::nullopt); !ok) {
      return std::unexpected(ok.error());
    }
    if (!experts.empty())
      for (std::uint32_t e = 0; e < out.experts; ++e)
        if (auto ok = add(experts, e); !ok) return std::unexpected(ok.error());
  }
  if (out.groups.size() > artifact::kMaxEntries)
    return Bad("too many artifact groups", Rule::kBounds);
  // Shard headers have the exact byte order reconstructed by Artifact::Open.
  std::vector<std::uint64_t> pads(out.shards.size(), 0);
  for (auto& shard : out.shards)
    shard.header =
        std::string(8, '\0') + R"({"__metadata__":{"format":"llmp-shard","format_version":"1"})";
  const auto entry = [&](std::uint32_t s, std::string_view name, std::string_view dtype,
                         const std::vector<std::uint64_t>& shape, std::uint64_t begin,
                         std::uint64_t end) {
    std::vector<std::string> dims;
    for (auto n : shape) dims.push_back(N(n));
    out.shards[s].header += ',' + Q(name) + ":{\"dtype\":" + Q(dtype) + ",\"shape\":" + A(dims) +
                            ",\"data_offsets\":[" + N(begin) + ',' + N(end) + "]}";
  };
  for (const auto& g : out.groups) {
    auto at = g.offset;
    const auto pad = [&](std::uint64_t end) {
      if (at < end) entry(g.shard, "~pad." + N(pads[g.shard]++), "U8", {end - at}, at, end);
    };
    for (const auto& m : g.members) {
      const auto& t = tensors[m.tensor];
      const auto target = g.offset + m.offset;
      pad(target);
      const auto view = artifact::ContainerViewOf(m.format, t.part, m.bytes);
      entry(g.shard, t.name + (m.expert ? "#" + N(*m.expert) : ""), view.dtype, view.shape, target,
            target + m.bytes);
      at = target + m.bytes;
      const auto begin = m.expert ? *m.expert * m.bytes : 0;
      out.destinations.push_back({m.tensor, g.shard, begin, begin + m.bytes, target});
    }
    pad(g.offset + g.stored);
  }
  for (auto& shard : out.shards) {
    shard.header += '}';
    if (shard.header.size() > artifact::kMaxShardHeader - 4096)
      return Bad("shard header exceeds cap", Rule::kBounds);
    shard.header.resize(Align(shard.header.size(), 4096), ' ');
    const auto size = shard.header.size() - 8;
    for (std::size_t i = 0; i < 8; ++i)
      shard.header[i] = static_cast<char>((size >> (8U * i)) & 255U);
  }
  std::ranges::sort(out.destinations, {},
                    [](const Destination& d) { return std::tie(d.tensor, d.begin); });
  out.first_destination.resize(tensors.size() + 1);
  std::size_t next = 0;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    out.first_destination[i] = next;
    std::uint64_t end = 0;
    while (next < out.destinations.size() && out.destinations[next].tensor == i) {
      const auto& d = out.destinations[next++];
      if (d.begin != end || d.end <= d.begin)
        return Bad("publication destination overlap or gap", Rule::kConfig);
      end = d.end;
    }
    const auto parts = format::PartsOf(tensors[i].format);
    const auto part = parts ? parts->Find(tensors[i].part) : std::nullopt;
    if (!part || end != part->bytes)
      return Bad("publication expert slices do not cover the logical part", Rule::kConfig);
  }
  out.first_destination.back() = next;
  if (next != out.destinations.size()) return Bad("publication extra destination", Rule::kConfig);
  if (auto ok = Transfers(plan, out); !ok) return std::unexpected(ok.error());
  return out;
}
std::string FileJson(std::string_view path, std::string_view role, std::uint64_t bytes,
                     std::string_view hash) {
  return O({{"bytes", N(bytes)}, {"path", Q(path)}, {"role", Q(role)}, {"sha256", Q(hash)}});
}
// Deep verification is deliberately payload bounded. Structural verification
// already reconstructs headers and validates the zero-pad geometry.
std::expected<void, Error> Deep(const fs::path& root, const artifact::Artifact& artifact) {
  std::vector<std::byte> buffer(artifact::kChunkBytes);
  for (const auto& listed : artifact.files()) {
    auto input = OpenCheckpointFile(root / listed.path, kNoItem);
    if (!input) return std::unexpected(input.error());
    base::Sha256 hash;
    for (std::uint64_t at = 0; at < listed.bytes.value();) {
      auto bytes = std::span(buffer).first(static_cast<std::size_t>(
          std::min<std::uint64_t>(buffer.size(), listed.bytes.value() - at)));
      if (auto ok = ReadExactly(input->fd.get(), at, bytes, kNoItem); !ok) return ok;
      hash.Update(bytes);
      at += bytes.size();
    }
    if (input->bytes != listed.bytes.value() || hash.Finish() != listed.sha256)
      return Bad("a staged or existing artifact file failed full verification");
  }
  for (std::size_t gi = 0; gi < artifact.groups().size(); ++gi) {
    const auto& g = artifact.groups()[gi];
    auto input = OpenCheckpointFile(root / artifact.shards()[g.shard].path, kNoItem);
    if (!input) return std::unexpected(input.error());
    for (std::uint32_t k = 0; k < g.chunks; ++k) {
      auto range = artifact::ChunkRangeOf(artifact.layout(), {static_cast<std::uint32_t>(gi), k});
      if (!range) return Bad("an artifact chunk has invalid geometry");
      auto bytes = std::span(buffer).first(static_cast<std::size_t>(range->length.value()));
      if (auto ok = ReadExactly(input->fd.get(), range->file_offset.value(), bytes, kNoItem); !ok)
        return ok;
      if (base::Sha256().Update(bytes).Finish() != artifact.chunk_sha256()[g.first_chunk + k])
        return Bad("an artifact chunk failed verification");
    }
  }
  return {};
}

std::expected<WriteResult, Error> Publish(const PublicationView& plan,
                                          const WriteOptions& options) {
  const auto& source = plan.source();
  if (options.store.empty() || !SafeName(options.build) ||
      (!options.expected.empty() && options.expected.size() != source.files.size()))
    return Bad("invalid writer options", Rule::kConfig);
  auto layout = Place(plan, options.metadata_bytes);
  if (!layout) return std::unexpected(layout.error());
  std::set<std::string_view> source_names;
  for (const auto& file : source.files) {
    if (!SafeName(file.name) || !source_names.insert(file.name).second || file.path.empty() ||
        (file.parsed.bytes == 0 && file.bytes != 0) || file.parsed.bytes > file.bytes)
      return Bad("a parsed source lacks a valid identity receipt", Rule::kConfig);
  }
  std::error_code error;
  if (auto ok = CreateImportDirectory(options.store, true); !ok) return std::unexpected(ok.error());
  const fs::path staging = options.store / ".staging";
  auto status = fs::symlink_status(staging, error);
  if (error && error != std::errc::no_such_file_or_directory)
    return Bad("cannot examine staging directory");
  if (fs::is_symlink(status)) return Bad("artifact staging must not be a symlink");
  error.clear();
  if (auto ok = CreateImportDirectory(staging, false); !ok) return std::unexpected(ok.error());
  if (::chmod(staging.c_str(), 0700) != 0) return Bad("cannot protect staging directory");
  auto lock = platform::LockFile::Acquire(staging / "native-import.lock");
  if (!lock) return Bad("another import owns this store or its lock cannot be opened");
  SweepOrphanedStagingJobs(staging);
  std::array<std::byte, 32> random{};
  if (!platform::FillRandom(random)) return Bad("cannot generate a private staging name");
  const fs::path job = staging / ("job-" + base::ToHex(base::Sha256().Update(random).Finish()));
  if (!fs::create_directory(job, error) || error)
    return Bad("cannot create private staging directory");
  Cleanup cleanup{job};
  if (::chmod(job.c_str(), 0700) != 0) return Bad("cannot protect staging directory");
  fs::create_directory(job / "data", error);
  if (error) return Bad("cannot create shard directory");
  fs::create_directory(job / "meta", error);
  if (error) return Bad("cannot create metadata directory");
  for (const auto& shard : layout->shards) {
    auto file = Create(job / shard.path);
    if (!file) return std::unexpected(file.error());
    if (auto ok = Write(file->get(), 0, std::as_bytes(std::span(shard.header))); !ok)
      return std::unexpected(ok.error());
    if (::ftruncate(file->get(), static_cast<off_t>(shard.header.size() + shard.bytes)) != 0)
      return Bad("cannot size a staging shard");
  }
  std::map<std::string, std::string> files;
  std::map<std::string, std::string> sources;
  std::uint64_t kept_bytes = 0;
  std::size_t next_copy = 0, next_semantic = 0;
  std::optional<std::size_t> active_cursor;
  FD output;
  std::optional<std::uint32_t> output_shard;
  for (std::size_t f = 0; f < source.files.size(); ++f) {
    const auto& input = source.files[f];
    const bool gguf = source.container == "gguf";
    const bool keep = gguf || ((input.name.ends_with(".json") || input.name.ends_with(".jinja") ||
                                input.name == "tokenizer.model") &&
                               ResolvesInside(input.path.parent_path(), input.path));
    // Source metadata copies must stay within the checkpoint (cache blobs
    // count); ordinary parsed weight files retain their usual symlink rules.
    const auto prefix_limit =
        gguf ? std::min(input.parsed.bytes, artifact::kMaxMetadataBytes) : input.bytes;
    if (keep && prefix_limit > artifact::kMaxMetadataBytes)
      return Bad("kept metadata exceeds its artifact cap", Rule::kBounds);
    std::string metadata;
    if (keep) metadata.reserve(static_cast<std::size_t>(prefix_limit));
    const auto streamed = StreamSource(
        input, f,
        [&](std::uint64_t at, std::span<const std::byte> bytes) -> std::expected<void, Error> {
          if (keep && at < prefix_limit) {
            const auto count =
                static_cast<std::size_t>(std::min<std::uint64_t>(bytes.size(), prefix_limit - at));
            metadata.append(reinterpret_cast<const char*>(bytes.data()), count);
          }
          const auto end = at + bytes.size();
          for (std::size_t si = next_semantic; si < layout->semantic.size(); ++si) {
            auto& semantic = layout->semantic[si];
            const auto& extent = semantic.receipt->extent;
            if (extent.file != f || extent.offset >= end) break;
            const auto first = std::max(at, extent.offset);
            const auto last = std::min(end, extent.offset + extent.bytes);
            if (first >= last) continue;
            if (first - extent.offset != semantic.received)
              return Bad("semantic source stream gap or duplicate", Rule::kConfig);
            semantic.hash.Update(bytes.subspan(static_cast<std::size_t>(first - at),
                                               static_cast<std::size_t>(last - first)));
            semantic.received += last - first;
            if (semantic.received == extent.bytes) next_semantic = si + 1;
          }
          for (std::size_t i = next_copy; i < layout->copies.size(); ++i) {
            auto& c = layout->copies[i];
            if (c.file != f || c.source >= end) break;
            const auto first = std::max(c.source, at);
            const auto last = std::min(c.source + c.bytes, end);
            if (first < last) {
              const auto piece = bytes.subspan(static_cast<std::size_t>(first - at),
                                               static_cast<std::size_t>(last - first));
              // A scale only some readers agree on (format::ScaleCode):
              // NVFP4's above 0x7E, an E8M0 0xFF.
              const Tensor& from = source.tensors[c.source_tensor];
              if (const auto bad = format::FirstUndefinedScale(
                      format::ScaleBytesOf(from.format, from.part), first - c.source, piece))
                return std::unexpected(
                    Error{.rule = Rule::kQuantization,
                          .reason = "a block scale is NaN or outside its code (UE4M3 "
                                    "0x00-0x7E, E8M0 0x00-0xFE), which readers decode apart",
                          .file = f,
                          .kind = ItemKind::kByte,
                          .item = first + *bad});
              if (!c.route) return Bad("copy route lacks its checked cursor");
              base::Check(!active_cursor || *active_cursor == i,
                          "only one publication cursor may borrow the shared workspace");
              active_cursor = i;
              if (auto ok = c.route->Consume(
                      first - c.source, piece,
                      [&](std::uint64_t offset,
                          std::span<const std::byte> payload) -> std::expected<void, Error> {
                        if (offset > UINT64_MAX - c.destination)
                          return Bad("publication scatter offset overflow", Rule::kBounds);
                        auto destination = c.destination + offset;
                        const auto first_target = layout->first_destination[c.tensor];
                        const auto last_target = layout->first_destination[c.tensor + 1];
                        auto targets = std::span(layout->destinations)
                                           .subspan(first_target, last_target - first_target);
                        auto found =
                            std::ranges::upper_bound(targets, destination, {}, &Destination::end);
                        while (!payload.empty()) {
                          if (found == targets.end() || destination < found->begin ||
                              destination >= found->end)
                            return Bad("publication scatter outside destination", Rule::kBounds);
                          const auto count = static_cast<std::size_t>(
                              std::min<std::uint64_t>(payload.size(), found->end - destination));
                          if (!output_shard || *output_shard != found->shard) {
                            output = FD(::open((job / layout->shards[found->shard].path).c_str(),
                                               O_RDWR | O_CLOEXEC));
                            if (!output.valid()) return Bad("cannot reopen a staging shard");
                            output_shard = found->shard;
                          }
                          if (auto ok = Write(output.get(),
                                              layout->shards[found->shard].header.size() +
                                                  found->output + destination - found->begin,
                                              payload.first(count));
                              !ok)
                            return ok;
                          destination += count;
                          payload = payload.subspan(count);
                          ++found;
                        }
                        return std::expected<void, Error>{};
                      });
                  !ok)
                return ok;
            }
            if (c.route && c.route->Finish()) {
              active_cursor.reset();
              next_copy = i + 1;
            }
          }
          return {};
        },
        options.expected.empty() ? std::nullopt : options.expected[f]);
    if (!streamed) return std::unexpected(streamed.error());
    sources.emplace(input.name, O({{"bytes", N(input.bytes)},
                                   {"name", Q(input.name)},
                                   {"sha256", Q(base::ToHex(*streamed))}}));
    if (keep) {
      std::string name = input.name;
      if (gguf) {
        auto header = base::gguf::ParseMetadata(std::as_bytes(std::span(metadata)));
        if (!header || header->metadata_end > artifact::kMaxMetadataBytes)
          return Bad("GGUF metadata cannot be kept within artifact bounds", Rule::kBounds);
        metadata.resize(static_cast<std::size_t>(header->metadata_end));
        std::fill(metadata.begin() + 8, metadata.begin() + 16, '\0');
        metadata[4] = '\3';  // Kept metadata uses GGUF v3; v2 has the same parsed fields.
        name = input.name.substr(0, input.name.size() - 5) + ".kv.gguf";
      }
      if (metadata.size() > artifact::kMaxMetadataTotal - kept_bytes)
        return Bad("kept metadata total exceeds artifact cap", Rule::kBounds);
      kept_bytes += metadata.size();
      const std::string path = "meta/" + name;
      if (auto ok = Save(job / path, metadata); !ok) return std::unexpected(ok.error());
      files.emplace(path, FileJson(path, "source-metadata", metadata.size(), Hex(metadata)));
    }
  }
  output = FD();
  for (const auto& copy : layout->copies)
    if (!copy.route || !copy.route->Finish())
      return Bad("a source stream left an output range unwritten");

  for (auto& semantic : layout->semantic)
    if (semantic.received != semantic.receipt->extent.bytes ||
        semantic.hash.Finish() != semantic.receipt->sha256)
      return Bad("semantic source bytes changed before publication", Rule::kConfig);
  // Every source, each generated part's whole file included, has streamed
  // against its parsed receipt: the derived bytes are the authenticated
  // content's.
  for (const auto& gen : plan.generated) {
    std::uint64_t written = 0;
    for (auto d = layout->first_destination[gen.tensor];
         d < layout->first_destination[gen.tensor + 1]; ++d) {
      const auto& target = layout->destinations[d];
      if (target.begin != written || target.end > gen.bytes.size())
        return Bad("generated part destination gap", Rule::kConfig);
      FD file(::open((job / layout->shards[target.shard].path).c_str(), O_RDWR | O_CLOEXEC));
      if (!file.valid()) return Bad("cannot reopen a staging shard");
      const auto bytes = std::span<const std::byte>(gen.bytes).subspan(
          static_cast<std::size_t>(target.begin),
          static_cast<std::size_t>(target.end - target.begin));
      if (auto ok =
              Write(file.get(), layout->shards[target.shard].header.size() + target.output, bytes);
          !ok)
        return std::unexpected(ok.error());
      written = target.end;
    }
    if (written != gen.bytes.size()) return Bad("generated part left unwritten", Rule::kConfig);
  }

  std::vector<std::string> chunks, groups, resources, arrays, shards, transformations;
  std::vector<std::byte> buffer(artifact::kChunkBytes);
  for (const auto& shard : layout->shards) {
    FD file(::open((job / shard.path).c_str(), O_RDWR | O_CLOEXEC));
    if (!file.valid()) return Bad("cannot reopen a written shard");
    base::Sha256 hash;
    const auto size = shard.header.size() + shard.bytes;
    for (std::uint64_t at = 0; at < size;) {
      auto bytes = std::span(buffer).first(
          static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), size - at)));
      if (auto ok = ReadExactly(file.get(), at, bytes, kNoItem); !ok)
        return std::unexpected(ok.error());
      hash.Update(bytes);
      at += bytes.size();
    }
    if (::fsync(file.get()) != 0) return Bad("cannot sync a written shard");
    files.emplace(shard.path, FileJson(shard.path, "shard", size, base::ToHex(hash.Finish())));
    shards.push_back(O({{"data_bytes", N(shard.bytes)},
                        {"data_offset", N(shard.header.size())},
                        {"header_sha256", Q(Hex(shard.header))},
                        {"path", Q(shard.path)}}));
  }
  for (std::size_t gi = 0; gi < layout->groups.size(); ++gi) {
    const auto& g = layout->groups[gi];
    const auto& shard = layout->shards[g.shard];
    auto file = OpenCheckpointFile(job / shard.path, kNoItem);
    if (!file) return std::unexpected(file.error());
    for (std::uint64_t at = 0; at < g.stored; at += artifact::kChunkBytes) {
      auto bytes = std::span(buffer).first(
          static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), g.stored - at)));
      if (auto ok =
              ReadExactly(file->fd.get(), shard.header.size() + g.offset + at, bytes, kNoItem);
          !ok)
        return std::unexpected(ok.error());
      chunks.push_back(Q(base::ToHex(base::Sha256().Update(bytes).Finish())));
    }
    groups.push_back(O({{"expert", g.expert ? N(*g.expert) : "null"},
                        {"first_chunk", N(g.first_chunk)},
                        {"kind", Q(g.kind)},
                        {"layer", g.layer ? N(*g.layer) : "null"},
                        {"offset", N(g.offset)},
                        {"shard", N(g.shard)},
                        {"stored_bytes", N(g.stored)},
                        {"used_bytes", N(g.used)}}));
    for (const auto& m : g.members) {
      const auto& tensor = plan.canonical().tensors[m.tensor];
      std::string format;
      artifact::AppendFormatJson(m.format, format);
      if (!m.expert) {
        std::map<std::string, std::string> r{{"bytes", N(m.bytes)},
                                             {"format", format},
                                             {"group", N(gi)},
                                             {"name", Q(tensor.name)},
                                             {"offset", N(m.offset)},
                                             {"part", Q(format::PartRoleName(tensor.part))},
                                             {"readable_bytes", N(m.readable)},
                                             {"roles", A({Q(tensor.name)})}};
        if (g.kind == "table" && format::RowBytes(m.format, tensor.part))
          r.emplace("access", Q("rows"));
        resources.push_back(O(std::move(r)));
      } else if (*m.expert == 0) {
        arrays.push_back(O({{"count", N(layout->experts)},
                            {"first_group", N(gi)},
                            {"format", format},
                            {"group_offset", N(m.offset)},
                            {"layer", N(*g.layer)},
                            {"name", Q(tensor.name)},
                            {"part", Q(format::PartRoleName(tensor.part))},
                            {"readable_bytes", N(m.readable)},
                            {"slice_bytes", N(m.bytes)}}));
        transformations.push_back(O({{"count", N(layout->experts)},
                                     {"kind", Q("expert-slice")},
                                     {"tensor", Q(tensor.name)}}));
      }
    }
  }
  const std::string index = O({{"chunk_bytes", N(artifact::kChunkBytes)},
                               {"chunk_sha256", A(chunks)},
                               {"expert_arrays", A(arrays)},
                               {"file_alignment", N(artifact::kFileAlignment)},
                               {"format", Q("llmp-index")},
                               {"format_version", "1"},
                               {"groups", A(groups)},
                               {"member_alignment", N(artifact::kMemberAlignment)},
                               {"resources", A(resources)},
                               {"shards", A(shards)}}) +
                            '\n';
  const bool is_drafter = IsDrafter(plan.spec());
  const auto spec = spec::ToJson(plan.spec());
  if (spec.size() > artifact::kMaxSpecBytes)
    return Bad("an artifact document exceeds its cap", Rule::kBounds);
  if (auto ok = Save(job / "spec.json", spec); !ok) return std::unexpected(ok.error());
  files.emplace("spec.json", FileJson("spec.json", "spec", spec.size(), Hex(spec)));
  if (index.size() > artifact::kMaxIndexBytes)
    return Bad("an artifact document exceeds its cap", Rule::kBounds);
  if (auto ok = Save(job / "index.json", index); !ok) return std::unexpected(ok.error());
  files.emplace("index.json", FileJson("index.json", "index", index.size(), Hex(index)));
  std::vector<std::string> file_list, source_list;
  for (const auto& [path, json] : files) file_list.push_back(json);
  for (const auto& [path, json] : sources) source_list.push_back(json);
  // Only reached after all source streams, semantic receipts and whole route
  // cursors completed. Historical source-converter folds emit no new record.
  for (const auto& fold : layout->folds) {
    std::vector<std::string> names;
    for (const auto& name : fold.tensors) names.push_back(Q(name));
    transformations.push_back(
        O({{"kind", Q("fold")}, {"fold", Q(fold.name)}, {"tensors", A(names)}}));
  }
  for (const auto& relayout : layout->relayouts) {
    std::vector<std::string> names;
    for (const auto& name : relayout.tensors) names.push_back(Q(name));
    transformations.push_back(O({{"kind", Q("relayout")},
                                 {"from", Q(format::Info(relayout.from).name)},
                                 {"tensors", A(names)}}));
  }
  std::ranges::sort(transformations);
  std::map<std::string, std::string> importer{{"build", Q(options.build)},
                                              {"name", Q("llmp-import")},
                                              {"options_sha256", Q(Hex(options.options))},
                                              {"version", Q("1")}};
  if (!plan.mapping_architecture.empty() && plan.mapping_version != 0) {
    importer.emplace("mapping_architecture", Q(plan.mapping_architecture));
    importer.emplace("mapping_version", N(plan.mapping_version));
  }
  const auto manifest = O({{"experimental", "true"},
                           {"files", A(file_list)},
                           {"format", Q("llmp-artifact")},
                           {"format_version", "1"},
                           {"importer", O(std::move(importer))},
                           {"layout", O({{"byte_order", Q("little")},
                                         {"chunk_bytes", N(artifact::kChunkBytes)},
                                         {"container", Q("safetensors")},
                                         {"file_alignment", N(artifact::kFileAlignment)},
                                         {"member_alignment", N(artifact::kMemberAlignment)},
                                         {"profile", Q("spark")},
                                         {"profile_version", "1"}})},
                           {"model", O({{"architecture", Q(source.architecture)},
                                        {"container", Q(source.container)},
                                        {"expert_count", N(layout->experts)}})},
                           {"source", A(source_list)},
                           {"transformations", A(transformations)}}) +
                        '\n';
  if (manifest.size() > artifact::kMaxManifestBytes)
    return Bad("artifact manifest exceeds its cap", Rule::kBounds);
  if (auto ok = Save(job / "manifest.json", manifest); !ok) return std::unexpected(ok.error());
  const auto identity = artifact::json::Parse(manifest, true);
  if (!identity || !identity->canonical()) return Bad("invalid canonical manifest", Rule::kConfig);
  const std::string id = base::ToHex(artifact::schema::ManifestIdentity(identity->root()));
  auto opened = artifact::Artifact::Open(job, {.expected_id = id});
  if (!opened) return Bad("the produced artifact failed structural verification", Rule::kConfig);
  auto stored = spec::StoredSpec(*opened);
  if (!stored || *stored != plan.spec())
    return Bad("the produced artifact has an invalid embedded spec", Rule::kConfig);
  auto stored_source = spec::SourceOf(*opened);
  if (!stored_source ||
      (!is_drafter && !components::ValidateSpec(*stored, stored_source->tensors,
                                                components::ValidationPurpose::kImport)))
    return Bad("the produced artifact does not bind its embedded spec", Rule::kConfig);
  if (auto ok = Deep(job, *opened); !ok) return std::unexpected(ok.error());
  for (const auto& directory : {job / "data", job / "meta", job}) {
    if (::chmod(directory.c_str(), 0755) != 0)
      return Bad("cannot set artifact directory permissions");
    if (auto ok = Sync(directory); !ok) return std::unexpected(ok.error());
  }
  const auto destination = options.store / id;
  const auto verify_existing = [&]() -> std::expected<WriteResult, Error> {
    auto previous = artifact::Artifact::Open(destination, {.expected_id = id});
    if (!previous) return Bad("the existing artifact is damaged");
    if (auto ok = Deep(destination, *previous); !ok) return Bad("the existing artifact is damaged");
    auto prev_stored = spec::StoredSpec(*previous);
    if (!prev_stored || *prev_stored != plan.spec()) return Bad("the existing artifact is damaged");
    auto prev_source = spec::SourceOf(*previous);
    if (!prev_source ||
        (!is_drafter && !components::ValidateSpec(*prev_stored, prev_source->tensors,
                                                  components::ValidationPurpose::kImport)))
      return Bad("the existing artifact is damaged");
    return WriteResult{id, destination, true};
  };
  const bool exists = fs::exists(destination, error);
  if (error) return Bad("cannot inspect an existing artifact");
  if (exists) {
    return verify_existing();
  }
  if (auto renamed = platform::RenameNoReplace(job, destination); !renamed) {
    if (renamed.error() == std::errc::file_exists) {
      std::error_code remove_err;
      fs::remove_all(job, remove_err);
      cleanup.path.clear();
      return verify_existing();
    }
    return Bad("cannot publish artifact");
  }
  cleanup.path.clear();
  if (auto ok = Sync(staging); !ok)
    return Bad("artifact published; staging directory durability sync failed");
  if (auto ok = Sync(options.store); !ok)
    return Bad("artifact published; store directory durability sync failed");
  return WriteResult{id, destination, false};
}
}  // namespace

std::expected<publication_detail::PlacementSummary, Error> publication_detail::InspectPlacement(
    const NormalizedPlan& plan, std::uint64_t metadata_bytes) {
  auto layout = Place({.checkpoint = plan.source(),
                       .canonical_source = plan.canonical(),
                       .model_spec = plan.spec(),
                       .payload = plan.cost().payload_bytes,
                       .copy = plan.is_indexed() ? nullptr : &plan,
                       .parts = plan.is_indexed() ? &plan.parts() : nullptr,
                       .bindings = plan.bindings(),
                       .semantic = plan.semantic(),
                       .planned_folds = plan.planned_folds(),
                       .generated = plan.generated(),
                       .mapping_architecture = plan.mapping_architecture(),
                       .mapping_version = plan.architecture_version()},
                      metadata_bytes);
  if (!layout) return std::unexpected(layout.error());
  PlacementSummary out{.metadata_bytes = layout->metadata_bytes,
                       .copies = layout->copies.size(),
                       .destinations = layout->destinations.size()};
  for (const auto& group : layout->groups) {
    out.tables += group.kind == "table";
    out.globals += group.kind == "global";
    out.layers += group.kind == "layer";
    out.heads += group.kind == "head";
    out.expert_groups += group.kind == "expert";
    if (group.kind == "table")
      for (const auto& member : group.members)
        out.row_tables +=
            format::RowBytes(member.format, plan.canonical().tensors[member.tensor].part)
                .has_value();
  }
  return out;
}

std::expected<WriteResult, Error> WriteArtifact(const NormalizedPlan& plan,
                                                const WriteOptions& options) {
  if (plan.has_options() && options.options != plan.options())
    return Bad("drafter publication options differ from its plan's", Rule::kConfig);
  return Publish({.checkpoint = plan.source(),
                  .canonical_source = plan.canonical(),
                  .model_spec = plan.spec(),
                  .payload = plan.cost().payload_bytes,
                  .copy = plan.is_indexed() ? nullptr : &plan,
                  .parts = plan.is_indexed() ? &plan.parts() : nullptr,
                  .bindings = plan.bindings(),
                  .semantic = plan.semantic(),
                  .planned_folds = plan.planned_folds(),
                  .generated = plan.generated(),
                  .mapping_architecture = plan.mapping_architecture(),
                  .mapping_version = plan.architecture_version()},
                 options);
}
std::expected<WriteResult, Error> WriteArtifact(const ComponentPlan& plan,
                                                const WriteOptions& options) {
  return Publish({.checkpoint = plan.checkpoint,
                  .canonical_source = plan.canonical,
                  .model_spec = plan.spec,
                  .payload = plan.payload_bytes,
                  .copy_routes = plan.routes,
                  .groups = plan.groups},
                 options);
}

std::expected<WriteResult, Error> publication_detail::WriteIndexed(
    const IndexedPartPlan& parts, const spec::Source& source, const spec::ModelSpec& model,
    std::uint64_t payload, std::span<const NormalizedPartBinding> bindings,
    std::span<const SemanticReceipt> semantic, std::span<const PlannedPartFold> folds,
    std::span<const GeneratedPart> generated, const WriteOptions& options) {
  return Publish({.checkpoint = parts.source(),
                  .canonical_source = source,
                  .model_spec = model,
                  .payload = payload,
                  .parts = &parts,
                  .bindings = bindings,
                  .semantic = semantic,
                  .planned_folds = folds,
                  .generated = generated},
                 options);
}

std::expected<void, Error> ValidateCompositionFiles(std::span<const SourceFile> files) {
  if (files.size() > 256) return Bad("too many kept composition files", Rule::kBounds);
  std::set<std::string_view> names;
  std::uint64_t total = 0;
  for (const auto& file : files) {
    if (!artifact::schema::IsKeptName(file.name) || !names.insert(file.name).second)
      return Bad("invalid or duplicate metadata path in composition", Rule::kConfig);
    if (file.bytes > artifact::kMaxMetadataBytes ||
        file.bytes > artifact::kMaxMetadataTotal - total)
      return Bad("composition metadata exceeds its cap", Rule::kBounds);
    total += file.bytes;
  }
  return {};
}

std::expected<WriteResult, Error> WriteComposition(const CompositionWriteOptions& options) {
  if (options.store.empty() || !SafeName(options.build) || options.architecture.empty() ||
      options.components.empty() || options.components.size() > artifact::kMaxComponents) {
    return Bad("invalid composition write options", Rule::kConfig);
  }
  if (auto valid = ValidateCompositionFiles(options.files); !valid)
    return std::unexpected(valid.error());
  const bool scoped =
      std::ranges::any_of(options.files, [](const auto& f) { return f.name.contains('/'); });
  if (!options.expected.empty() && options.expected.size() != options.files.size()) {
    return Bad("expected identities count disagrees with composition files", Rule::kConfig);
  }
  for (std::size_t i = 0; i < options.components.size(); ++i) {
    const auto& c = options.components[i];
    if (c.role.empty() || c.role.size() > 64 || c.artifact.size() != 64 || c.architecture.empty()) {
      return Bad("invalid component descriptor in composition", Rule::kConfig);
    }
    if (i > 0 && options.components[i - 1].role >= c.role) {
      return Bad("components must be sorted uniquely by role in composition", Rule::kConfig);
    }
  }

  std::error_code error;
  if (auto ok = CreateImportDirectory(options.store, true); !ok) return std::unexpected(ok.error());
  const fs::path staging = options.store / ".staging";
  auto status = fs::symlink_status(staging, error);
  if (error && error != std::errc::no_such_file_or_directory)
    return Bad("cannot examine staging directory");
  if (fs::is_symlink(status)) return Bad("composition staging must not be a symlink");
  error.clear();
  if (auto ok = CreateImportDirectory(staging, false); !ok) return std::unexpected(ok.error());
  if (::chmod(staging.c_str(), 0700) != 0) return Bad("cannot protect staging directory");
  auto lock = platform::LockFile::Acquire(staging / "native-import.lock");
  if (!lock) return Bad("another import owns this store or its lock cannot be opened");
  SweepOrphanedStagingJobs(staging);
  std::array<std::byte, 32> random{};
  if (!platform::FillRandom(random)) return Bad("cannot generate a private staging name");
  const fs::path job = staging / ("job-" + base::ToHex(base::Sha256().Update(random).Finish()));
  if (!fs::create_directory(job, error) || error)
    return Bad("cannot create private staging directory");
  Cleanup cleanup{job};
  if (::chmod(job.c_str(), 0700) != 0) return Bad("cannot protect staging directory");
  if (!options.files.empty()) {
    fs::create_directory(job / "meta", error);
    if (error) return Bad("cannot create metadata directory");
  }

  struct KeptFileEntry {
    std::string name;
    std::uint64_t bytes = 0;
    std::string sha256;
  };
  std::vector<KeptFileEntry> kept_entries;
  std::set<fs::path> metadata_dirs;
  for (std::size_t f = 0; f < options.files.size(); ++f) {
    const auto& file = options.files[f];
    const std::string& name = file.name;
    const fs::path out_path = job / "meta" / name;
    if (auto ok = CreateImportDirectory(out_path.parent_path(), false); !ok)
      return std::unexpected(ok.error());
    metadata_dirs.insert(out_path.parent_path());
    auto out = Create(out_path);
    if (!out) return std::unexpected(out.error());
    const auto digest = StreamSource(
        file, f,
        [&](std::uint64_t at, std::span<const std::byte> chunk) {
          return Write(out->get(), at, chunk);
        },
        options.expected.empty() ? std::nullopt : options.expected[f]);
    if (!digest) return std::unexpected(digest.error());
    if (::fsync(out->get()) != 0) return Bad("cannot sync metadata file");
    kept_entries.push_back({.name = name, .bytes = file.bytes, .sha256 = base::ToHex(*digest)});
  }
  for (const auto& directory : metadata_dirs) {
    if (::chmod(directory.c_str(), 0755) != 0)
      return Bad("cannot set metadata directory permissions");
    if (auto ok = Sync(directory); !ok) return std::unexpected(ok.error());
  }
  std::ranges::sort(kept_entries, {}, &KeptFileEntry::name);

  // spec.json, listed last (its path sorts after every meta/ one).
  if (options.spec) {
    if (auto valid = spec::Validate(*options.spec); !valid)
      return Bad("composition spec is not valid", Rule::kConfig);
    for (const auto& stage : options.spec->stages)
      if (stage.artifact >= options.components.size())
        return Bad("composition spec names a component it does not have", Rule::kConfig);
    const auto json = spec::ToJson(*options.spec);
    if (json.size() > artifact::kMaxSpecBytes)
      return Bad("composition spec exceeds its cap", Rule::kBounds);
    if (auto ok = Save(job / "spec.json", json); !ok) return std::unexpected(ok.error());
  }
  std::vector<std::string> comp_jsons;
  for (const auto& c : options.components) {
    comp_jsons.push_back(
        O({{"architecture", Q(c.architecture)}, {"artifact", Q(c.artifact)}, {"role", Q(c.role)}}));
  }
  std::vector<std::string> source_jsons;
  std::vector<std::string> file_jsons;
  for (const auto& k : kept_entries) {
    source_jsons.push_back(
        O({{"bytes", N(k.bytes)}, {"name", Q(k.name)}, {"sha256", Q(k.sha256)}}));
    file_jsons.push_back(O({{"bytes", N(k.bytes)},
                            {"path", Q("meta/" + k.name)},
                            {"role", Q("source-metadata")},
                            {"sha256", Q(k.sha256)}}));
  }
  if (options.spec) {
    const auto json = spec::ToJson(*options.spec);
    file_jsons.push_back(FileJson("spec.json", "spec", json.size(), Hex(json)));
  }
  const auto manifest = O({{"components", A(comp_jsons)},
                           {"experimental", "true"},
                           {"files", A(file_jsons)},
                           {"format", Q("llmp-composition")},
                           {"format_version", scoped ? "2" : "1"},
                           {"importer", O({{"build", Q(options.build)},
                                           {"name", Q("llmp-import")},
                                           {"options_sha256", Q(Hex(""))},
                                           {"version", Q("1")}})},
                           {"model", O({{"architecture", Q(options.architecture)}})},
                           {"source", A(source_jsons)}}) +
                        '\n';
  if (manifest.size() > artifact::kMaxManifestBytes) {
    return Bad("composition manifest exceeds its cap", Rule::kBounds);
  }
  if (auto ok = Save(job / "manifest.json", manifest); !ok) return std::unexpected(ok.error());
  const auto identity = artifact::json::Parse(manifest, true);
  if (!identity || !identity->canonical()) return Bad("invalid canonical manifest", Rule::kConfig);
  const std::string id = base::ToHex(artifact::schema::ManifestIdentity(identity->root()));

  auto opened = artifact::OpenComposition(job, id);
  if (!opened) return Bad("the produced composition failed verification", Rule::kConfig);
  if (options.spec) {
    const auto stored = opened->spec_json();
    auto parsed = stored ? spec::FromJson(*stored)
                         : std::expected<spec::ModelSpec, std::string>(std::unexpected("none"));
    if (!parsed || *parsed != *options.spec)
      return Bad("the produced composition has an invalid spec", Rule::kConfig);
  }

  if (!options.files.empty()) {
    if (::chmod((job / "meta").c_str(), 0755) != 0)
      return Bad("cannot set metadata directory permissions");
    if (auto ok = Sync(job / "meta"); !ok) return std::unexpected(ok.error());
  }
  if (::chmod(job.c_str(), 0755) != 0) return Bad("cannot set composition directory permissions");
  if (auto ok = Sync(job); !ok) return std::unexpected(ok.error());

  const auto destination = options.store / id;
  const auto verify_existing = [&]() -> std::expected<WriteResult, Error> {
    auto previous = artifact::OpenComposition(destination);
    if (!previous) return Bad("the existing composition is damaged");
    return WriteResult{id, destination, true};
  };
  const bool exists = fs::exists(destination, error);
  if (error) return Bad("cannot inspect an existing composition");
  if (exists) {
    return verify_existing();
  }
  if (auto renamed = platform::RenameNoReplace(job, destination); !renamed) {
    if (renamed.error() == std::errc::file_exists) {
      std::error_code remove_err;
      fs::remove_all(job, remove_err);
      cleanup.path.clear();
      return verify_existing();
    }
    return Bad("cannot publish composition");
  }
  cleanup.path.clear();
  if (auto ok = Sync(staging); !ok)
    return Bad("composition published; staging directory durability sync failed");
  if (auto ok = Sync(options.store); !ok)
    return Bad("composition published; store directory durability sync failed");
  return WriteResult{id, destination, false};
}
}  // namespace llmp::import
