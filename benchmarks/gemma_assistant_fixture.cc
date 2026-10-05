// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Manual, stage-zero-only original-input replay through the production
// assistant graph/plan. These immutable arrays are not a target checkpoint.
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "base/json.h"
#include "base/sha256.h"
#include "engine/gemma4_assistant.h"
#include "engine/support.h"
#include "providers/device_runtime.h"

namespace {
namespace en = jitllm::engine;
namespace kg = jitllm::kernels::ggml;
namespace md = jitllm::model;
namespace sc = jitllm::scheduler;
namespace js = jitllm::base::json;
using en::support::Address;
using en::support::Error;
using en::support::Pointer;
using en::support::Round;
constexpr auto kHostBytes = std::uint64_t{128} << 20U;
constexpr std::array<std::string_view, 11> kFiles{
    "metadata.json",       "feature.f32",          "anchor.i32",          "local-k.f16",
    "local-v.f16",         "global-k.f16",         "global-v.f16",        "local-positions.i32",
    "local-membership.u8", "global-positions.i32", "global-membership.u8"};
constexpr std::array<std::uint64_t, 11> kSizes{971,    11264, 4,    1048576, 1048576, 524288,
                                               524288, 5120,  1280, 16384,   4096};
std::expected<std::vector<std::byte>, std::string> Read(const std::filesystem::path& path,
                                                        std::uint64_t size) {
  std::error_code ec;
  if (std::filesystem::file_size(path, ec) != size || ec) return Error("fixture file size differs");
  std::vector<std::byte> data(size);
  std::ifstream file(path, std::ios::binary);
  if (!file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size)))
    return Error("fixture read failed");
  return data;
}
std::string Hash(std::span<const std::byte> bytes) {
  return jitllm::base::ToHex(jitllm::base::Sha256().Update(bytes).Finish());
}
bool Integer(js::Value object, std::string_view key, std::int64_t n) {
  const auto field = object.find(key);
  return field && field->int64() == n;
}
bool Numbers(js::Value object, std::string_view key, std::span<const std::int64_t> values) {
  const auto field = object.find(key);
  if (!field || !field->is_array() || field->size() != values.size()) return false;
  for (std::size_t i = 0; i < values.size(); ++i)
    if (field->at(i).int64() != values[i]) return false;
  return true;
}
en::Status Descriptor(js::Value metadata, std::string_view key, std::uint32_t d,
                      std::uint32_t heads, std::uint32_t capacity) {
  auto domain = metadata.find(key);
  if (!domain) return Error("fixture cache descriptor absent");
  const auto pitch = std::int64_t{d} * heads * 2;
  const std::array<std::int64_t, 4> view_ne{d, heads, 256, 1}, root_ne{d * heads, capacity, 1, 1},
      view_nb{2, d * 2, pitch, pitch * capacity},
      root_nb{2, pitch, pitch * capacity, pitch * capacity};
  for (const auto name : {"k", "v"}) {
    auto tensor = domain->find(name);
    if (!tensor || !Integer(*tensor, "relative_offset", 0))
      return Error("fixture cache offset differs");
    auto view = tensor->find("view"), root = tensor->find("root");
    if (!view || !root || !Integer(*view, "type", 1) || !Integer(*root, "type", 1) ||
        !Numbers(*view, "ne", view_ne) || !Numbers(*root, "ne", root_ne) ||
        !Numbers(*view, "nb", view_nb) || !Numbers(*root, "nb", root_nb))
      return Error("fixture original cache descriptor differs");
  }
  return {};
}
class Replay final : public en::PagedModel {
 public:
  explicit Replay(en::PagedNode& node) : node_(node), resources_(node, 0, 0), runs_(true) {}
  en::Status Setup(const std::filesystem::path& input, const std::filesystem::path& manifest) {
    if (!node_.ChargeHost(kHostBytes, false)) return Error("fixture caller host budget");
    charged_ = true;
    std::error_code manifest_error;
    const auto manifest_size = std::filesystem::file_size(manifest, manifest_error);
    if (manifest_error || manifest_size > 16384) return Error("fixture manifest size");
    auto manifest_data = Read(manifest, manifest_size);
    if (!manifest_data || manifest_data->size() > 16384) return Error("fixture manifest size");
    auto doc = js::Parse(
        std::string_view(reinterpret_cast<const char*>(manifest_data->data()),
                         manifest_data->size()),
        {.max_bytes = 16384, .max_depth = 8, .max_values = 256, .max_string_bytes = 8192});
    if (!doc) return Error("fixture manifest JSON");
    auto files = doc->root().find("files");
    if (!files || !files->is_object() || files->size() != kFiles.size())
      return Error("fixture input allowlist differs");
    for (std::size_t i = 0; i < kFiles.size(); ++i) {
      auto declared = files->find(kFiles[i]);
      if (!declared || !Integer(*declared, "bytes", static_cast<std::int64_t>(kSizes[i])))
        return Error("fixture input manifest bytes differ");
      auto hash = declared->find("sha256");
      auto bytes = Read(input / kFiles[i], kSizes[i]);
      if (!hash || !hash->is_string() || !bytes || Hash(*bytes) != hash->string())
        return Error("fixture input hash differs");
      data_[i] = std::move(*bytes);
    }
    auto metadata =
        js::Parse(std::string_view(reinterpret_cast<const char*>(data_[0].data()), data_[0].size()),
                  {.max_bytes = 2048, .max_depth = 8, .max_values = 256, .max_string_bytes = 1024});
    if (!metadata) return Error("fixture metadata JSON");
    const auto m = metadata->root();
    for (const auto [key, value] : std::initializer_list<std::pair<std::string_view, std::int64_t>>{
             {"version", 1},
             {"owner", 0},
             {"stream", 0},
             {"sequence", 0},
             {"completed_endpoint", 64},
             {"query_position", 64},
             {"feature_position", 63},
             {"feature_width", 2816},
             {"vocabulary", 262144},
             {"local_window", 1024},
             {"local_capacity", 1280},
             {"global_capacity", 4096},
             {"read_cells", 256},
             {"local_layer", 28},
             {"global_layer", 29}})
      if (!Integer(m, key, value)) return Error("fixture stage-zero semantic differs");
    auto name = m.find("profile");
    if (!name || name->string() != "26B-A4B") return Error("fixture profile differs");
    if (auto r = Descriptor(m, "local_descriptors", 256, 8, 1280); !r) return r;
    if (auto r = Descriptor(m, "global_descriptors", 512, 2, 4096); !r) return r;
    std::memcpy(&anchor_, data_[2].data(), sizeof(anchor_));
    if (anchor_ < 0 || anchor_ >= 262144) return Error("fixture anchor is not canonical");
    initial_.resize(2816);
    std::memcpy(initial_.data(), data_[1].data(), data_[1].size());
    if (!std::ranges::all_of(initial_, [](float f) { return std::isfinite(f); }))
      return Error("fixture feature nonfinite");
    for (std::size_t i = 3; i <= 6; ++i)
      for (std::size_t at = 0; at < data_[i].size(); at += 2) {
        ggml_fp16_t value;
        std::memcpy(&value, data_[i].data() + at, 2);
        if (!std::isfinite(ggml_fp16_to_fp32(value))) return Error("fixture padded KV nonfinite");
        const auto cell = at / ((i < 5 ? 2048U : 1024U) * sizeof(ggml_fp16_t));
        if (cell >= 64 && value != 0)
          return Error("fixture unused padded KV is not initialized zero");
      }
    for (const auto [positions, membership, capacity] :
         {std::tuple{7U, 8U, 1280U}, std::tuple{9U, 10U, 4096U}})
      for (std::uint32_t cell = 0; cell < capacity; ++cell) {
        std::int32_t position;
        std::memcpy(&position, data_[positions].data() + cell * 4, 4);
        if (position != (cell < 64 ? static_cast<std::int32_t>(cell) : -1) ||
            data_[membership][cell] != (cell < 64 ? std::byte{1} : std::byte{0}))
          return Error("fixture physical occupancy differs from frozen prefix");
      }
    const auto store = std::filesystem::path("/home/pmeenan/.local/share/jitllm/m3-artifacts");
    if (auto r = target_.Open(store /
                              "4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3");
        !r)
      return r;
    if (auto r = assistant_.Open(
            store / "1040a0299a459e00ad0a77efd77bd319ac593986ba2c9ef29eb03d07ce97db42");
        !r)
      return r;
    auto tb = md::BindGemma4(profile_, target_.artifact());
    auto ab = md::BindGemma4Assistant(assistant_profile_, assistant_.artifact());
    auto state = md::Gemma4State(profile_, 4096, 128);
    if (!tb || !ab || !state) return Error("fixture prepared model contract");
    target_binding_ = std::move(*tb);
    assistant_binding_ = std::move(*ab);
    state_ = std::move(*state);
    if (auto r = md::CheckGemma4AssistantTarget(assistant_profile_, assistant_binding_, profile_,
                                                target_binding_);
        !r)
      return Error(r.error());
    std::vector<en::GroupPlace> tp(target_.artifact().groups().size(), en::GroupPlace::kNone),
        ap(assistant_.artifact().groups().size(), en::GroupPlace::kDevice);
    auto embedding = target_.artifact().ResourcePlacement(target_binding_.token_embd.index);
    if (!embedding) return Error("fixture embedding placement");
    tp[embedding->group] = en::GroupPlace::kDevice;
    if (auto r = target_.Reserve(node_, tp, {}); !r) return r;
    if (auto r = assistant_.Reserve(node_, ap, {}); !r) return r;
    if (auto r = resources_.OpenCublas("assistant fixture cuBLAS"); !r) return r;
    if (auto r = resources_.Map(cache_, "immutable assistant fixture KV", state_.bytes,
                                jitllm::catalog::MemoryClass::kLiveState);
        !r)
      return r;
    target_model_.profile = &profile_;
    target_model_.binding = &target_binding_;
    target_model_.state = &state_;
    target_model_.resources.resize(target_.artifact().resources().size());
    target_model_.resources[target_binding_.token_embd.index] = {
        target_.resource_address(target_binding_.token_embd.index),
        target_binding_.token_embd.readable};
    target_model_.slots.push_back({cache_.base, state_.bytes});
    model_.profile = &assistant_profile_;
    model_.binding = &assistant_binding_;
    model_.target = &target_model_;
    for (std::uint32_t i = 0; i < assistant_.artifact().resources().size(); ++i)
      model_.resources.push_back(
          {assistant_.resource_address(i), assistant_.artifact().resources()[i].readable.value()});
    shape_.segments.push_back({0, 64, 256, 256});
    auto measuring = resources_.MeasuringContext();
    if (!measuring) return Error(measuring.error());
    auto measured = en::PlanGemma4Assistant(model_, shape_, kg::DeviceChoicesOf(**measuring), 0, 0);
    if (!measured) return Error(measured.error());
    auto scratch = kg::PlanScratch(**measuring, (*measured)->plan);
    if (!scratch) return Error(scratch.error().detail);
    scratch_ = Round(*scratch + (1U << 20U), en::kPagedExtent);
    if (auto r =
            node_.MapWorkspace(Round((*measured)->placement.extent, en::kPagedExtent), scratch_);
        !r)
      return r;
    auto staging = resources_.Pinned(Round((*measured)->inputs_bytes, en::kPagedExtent));
    auto heads = resources_.Pinned(262144 * sizeof(float)),
         features = resources_.Pinned(2816 * sizeof(float)), witness = resources_.Pinned(1048576);
    if (!staging || !heads || !features || !witness) return Error("fixture pinned sources/outputs");
    runs_.SetStaging(*staging, Round((*measured)->inputs_bytes, en::kPagedExtent));
    head_ = *heads;
    feature_ = *features;
    witness_ = *witness;
    node_.SetHostFloor(kHostBytes);
    const auto fixed = node_.catalog().OccupancyOf(node_.domain()).Total().value();
    if (auto r = node_.Start(jitllm::base::Bytes(
            fixed + (target_.extents().size() + assistant_.extents().size()) * en::kPagedExtent +
            kHostBytes));
        !r)
      return r;
    if (auto r = target_.Register(node_, 0); !r) return r;
    if (auto r = assistant_.Register(node_, 0); !r) return r;
    auto extents = managed_extents();
    if (auto r = node_.scheduler().PinPlaces(extents); !r) return Error(sc::ToString(r.error()));
    const auto own = resources_.extents();
    extents.insert(extents.end(), own.begin(), own.end());
    for (const auto* workspace : {&node_.activations(), &node_.pool()})
      extents.insert(extents.end(), workspace->extents.begin(), workspace->extents.end());
    auto closure = node_.catalog().ClosureOfExtents(extents);
    if (!closure) return Error(jitllm::catalog::ToString(closure.error()));
    closure_ = std::move(*closure);
    if (auto r = resources_.BindLaunch(scratch_); !r) return r;
    runs_.SetLaunch(&resources_.launch());
    auto p = en::PlanGemma4Assistant(model_, shape_, kg::DeviceChoicesOf(resources_.launch()),
                                     node_.activations().base, node_.activations().bytes);
    if (!p) return Error(p.error());
    planned_ = std::move(*p);
    if (auto r = en::BindPlanned(*planned_, resources_.launch(), resources_.registry(),
                                 "assistant fixture");
        !r)
      return Error(r.error());
    node_.Run();
    return {};
  }
  en::Status Execute(const std::filesystem::path& output,
                     const std::filesystem::path& stock_incoming = {}) {
    std::vector<float> stock_feature;
    std::array<std::int32_t, 3> stock_anchor{};
    if (!stock_incoming.empty()) {
      auto feature = Read(stock_incoming / "incoming-feature.f32", 3 * 2816 * sizeof(float));
      auto anchors = Read(stock_incoming / "incoming-anchor.i32", 3 * sizeof(std::int32_t));
      if (!feature || !anchors) return Error("posthoc input shape differs");
      stock_feature.resize(3 * 2816);
      std::memcpy(stock_feature.data(), feature->data(), feature->size());
      std::memcpy(stock_anchor.data(), anchors->data(), anchors->size());
      if (!std::ranges::all_of(stock_feature, [](float v) { return std::isfinite(v); }) ||
          !std::ranges::all_of(stock_anchor, [](std::int32_t t) { return t >= 0 && t < 262144; }) ||
          std::memcmp(stock_feature.data(), initial_.data(), 2816 * sizeof(float)) != 0 ||
          stock_anchor[0] != anchor_)
        return Error("posthoc first input differs from authenticated stage zero");
    }
    std::error_code ec;
    if (std::filesystem::exists(output, ec) || !std::filesystem::create_directory(output, ec) || ec)
      return Error("fixture output must be new");
    return node_.WithRequest(
        0, closure_, "assistant original stage-zero endogenous replay", [&]() -> en::Status {
          std::vector<const ggml_tensor*> caches;
          for (const auto& segment : planned_->graph.segments)
            for (const auto& [k, v] : segment.caches) {
              caches.push_back(k);
              caches.push_back(v);
            }
          en::Coverage coverage;
          en::CheckCoverage(node_, 0, planned_->graph.nodes,
                            {.state = caches, .inputs = planned_->graph.inputs}, coverage);
          if (coverage.violations != 0)
            return Error("fixture catalog coverage: " + coverage.first_violation);

          // Provider async-copy sources are node-owned pinned buffers. Retire
          // each upload before reusing this single, fully funded staging span.
          for (std::size_t i = 0; i < 4; ++i) {
            const auto layer = i < 2 ? 28U : 29U;
            std::memcpy(witness_, data_[i + 3].data(), data_[i + 3].size());
            auto uploaded = node_.Job(
                closure_,
                [&](jitllm::providers::NativeStream stream) {
                  return jitllm::providers::CopyAsync(
                             stream,
                             Pointer(cache_.base + state_.tensors[layer * 2 + i % 2].offset),
                             witness_, data_[i + 3].size(),
                             jitllm::providers::CopyKind::kHostToDevice)
                                 .ok()
                             ? sc::JobResult::kQueued
                             : sc::JobResult::kUnknown;
                },
                "fixture immutable KV pinned upload", 0);
            if (!uploaded) return uploaded;
          }
          auto factors = node_.Job(
              closure_,
              [&](jitllm::providers::NativeStream stream) {
                return jitllm::providers::CopyAsync(
                           stream, witness_,
                           Pointer(
                               assistant_.resource_address(assistant_binding_.rope_freqs.index)),
                           256 * sizeof(float), jitllm::providers::CopyKind::kDeviceToHost)
                               .ok()
                           ? sc::JobResult::kQueued
                           : sc::JobResult::kUnknown;
              },
              "fixture frequency factors", 0);
          if (!factors) return factors;
          if (auto r = md::CheckGemma4RopeFactors(
                  profile_, std::span(static_cast<const float*>(witness_), 256));
              !r)
            return Error(r.error());
          auto before = Witness();
          if (!before) return Error(before.error());
          std::cout << "immutable_cache_before=" << *before << '\n';
          for (const unsigned steps : {1U, 3U}) {
            std::array<std::array<std::string, 2>, 3> expected;
            for (unsigned repeat = 0; repeat < 2; ++repeat) {
              auto incoming = initial_;
              auto anchor = anchor_;
              for (unsigned step = 0; step < steps; ++step) {
                if (!stock_incoming.empty()) {
                  incoming.assign(stock_feature.begin() + step * 2816,
                                  stock_feature.begin() + (step + 1) * 2816);
                  anchor = stock_anchor[step];
                }
                const std::array<en::Gemma4AssistantInput, 1> input{{{0, 64, anchor}}};
                auto source =
                    en::Gemma4AssistantSources(planned_->graph, input, incoming, false, kHostBytes);
                if (!source) return Error(source.error());
                auto copies = runs_.Stage(source->sources, 0);
                if (!copies) return Error(copies.error());
                const std::array<en::RunCopy, 2> outputs{
                    {{Address(head_), Address(planned_->graph.logits->data),
                      262144 * sizeof(float)},
                     {Address(feature_), Address(planned_->graph.next_features->data),
                      2816 * sizeof(float)}}};
                auto posted = node_.Job(
                    closure_,
                    [&](jitllm::providers::NativeStream stream) {
                      auto queued = runs_.Queue(plan_runs_, *copies, {}, *planned_->bound, outputs,
                                                plan_runs_.CaptureDue(true), stats_, stream);
                      if (!queued.result) return sc::JobResult::kUnknown;
                      en::Count(stats_, queued.path);
                      return sc::JobResult::kQueued;
                    },
                    "assistant stage-zero endogenous query", 0);
                if (!posted) return posted;
                const auto head = std::span(static_cast<const float*>(head_), 262144),
                           feature = std::span(static_cast<const float*>(feature_), 2816);
                if (auto r = en::CheckGemma4AssistantOutputs(1, 2816, head, feature); !r) return r;
                const std::array<std::string, 2> hashes{Hash(std::as_bytes(head)),
                                                        Hash(std::as_bytes(feature))};
                if (repeat == 0)
                  expected[step] = hashes;
                else if (hashes != expected[step])
                  return Error("fixture endogenous own repeat differs");
                for (const auto [name, bytes] : {std::pair{"head", std::as_bytes(head)},
                                                 std::pair{"projection", std::as_bytes(feature)}}) {
                  std::ofstream file(output / ("chain" + std::to_string(steps) + "-repeat" +
                                               std::to_string(repeat) + "-step" +
                                               std::to_string(step) + "-" + name + ".f32"),
                                     std::ios::binary | std::ios::noreplace);
                  if (!file.write(reinterpret_cast<const char*>(bytes.data()),
                                  static_cast<std::streamsize>(bytes.size())))
                    return Error("fixture output write failed");
                  file.flush();
                  if (!file) return Error("fixture output flush failed");
                }
                const auto next = static_cast<std::int32_t>(
                    std::max_element(head.begin(), head.end()) - head.begin());
                std::cout << "posthoc=" << !stock_incoming.empty() << " chain=" << steps
                          << " repeat=" << repeat << " step=" << step
                          << " P=64 input_anchor=" << anchor << " next_anchor=" << next
                          << " head_sha256=" << hashes[0] << " projection_sha256=" << hashes[1]
                          << '\n';
                incoming.assign(feature.begin(), feature.end());
                anchor = next;
              }
              auto after = Witness();
              if (!after) return Error(after.error());
              if (*after != *before) return Error("fixture frozen KV bytes changed");
              std::cout << "repeat=" << repeat << " immutable_cache_after=" << *after << '\n';
            }
          }
          if (stats_.replayed == 0) return Error("fixture captured replay absent");
          std::cout << "captured=" << stats_.captured << " replayed=" << stats_.replayed << '\n';
          return {};
        });
  }
  std::uint32_t stream() const override { return 0; }
  const jitllm::catalog::Closure& fence_closure() const override { return closure_; }
  std::vector<jitllm::catalog::ExtentId> managed_extents() const override {
    auto extents = target_.extents();
    const auto extra = assistant_.extents();
    extents.insert(extents.end(), extra.begin(), extra.end());
    return extents;
  }
  en::Status Release() override {
    plan_runs_.DropGraph();
    planned_.reset();
    std::vector<std::string> errors;
    resources_.Release(errors);
    for (auto* weights : {&target_, &assistant_})
      if (weights->opened())
        if (auto r = weights->Release(node_.memory()); !r) errors.push_back(r.error());
    if (charged_) {
      node_.UnchargeHost(kHostBytes);
      charged_ = false;
    }
    return en::support::Joined(errors);
  }

 private:
  std::expected<std::string, std::string> Witness() {
    jitllm::base::Sha256 hash;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto layer = i < 2 ? 28U : 29U;
      const auto bytes = data_[i + 3].size();
      auto copied = node_.Job(
          closure_,
          [&](jitllm::providers::NativeStream stream) {
            return jitllm::providers::CopyAsync(
                       stream, witness_,
                       Pointer(cache_.base + state_.tensors[layer * 2 + i % 2].offset), bytes,
                       jitllm::providers::CopyKind::kDeviceToHost)
                           .ok()
                       ? sc::JobResult::kQueued
                       : sc::JobResult::kUnknown;
          },
          "fixture frozen KV witness", 0);
      if (!copied) return Error(copied.error());
      if (Hash(std::span(static_cast<const std::byte*>(witness_), bytes)) != Hash(data_[i + 3]))
        return Error("fixture uploaded KV differs");
      hash.Update(std::span(static_cast<const std::byte*>(witness_), bytes));
    }
    return jitllm::base::ToHex(hash.Finish());
  }
  en::PagedNode& node_;
  en::RunnerResources resources_;
  en::PagedWeights target_, assistant_;
  md::Gemma4Profile profile_ = md::Gemma4_26BA4B();
  md::Gemma4AssistantProfile assistant_profile_ = md::Gemma4Assistant26();
  md::Gemma4Binding target_binding_;
  md::Gemma4AssistantBinding assistant_binding_;
  md::Gemma4StateLayout state_;
  en::Gemma4Model target_model_;
  en::Gemma4AssistantModel model_;
  en::Mapped cache_;
  kg::Gemma4AssistantShape shape_;
  std::unique_ptr<en::Gemma4AssistantPlanned> planned_;
  en::GraphRuns runs_;
  en::PlanRuns plan_runs_;
  en::GraphStats stats_;
  jitllm::catalog::Closure closure_;
  std::array<std::vector<std::byte>, 11> data_;
  std::vector<float> initial_;
  std::int32_t anchor_ = 0;
  std::uint64_t scratch_ = 0;
  void *head_ = nullptr, *feature_ = nullptr, *witness_ = nullptr;
  bool charged_ = false;
};
struct Lifetime {
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  Replay replay{node};
  std::array<en::PagedModel*, 1> entered{&replay};
};
}  // namespace
int main(int argc, char** argv) {
  umask(0077);
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: jitllm_gemma_assistant_fixture STAGE0 MANIFEST NEW_OUTPUT "
                 "[POSTHOC_INCOMING_DIR]\n";
    return 2;
  }
  auto owner = std::make_unique<Lifetime>();
  auto result = owner->node.Open();
  if (result) result = owner->replay.Setup(argv[1], argv[2]);
  if (result) result = owner->replay.Execute(argv[3], argc == 5 ? argv[4] : "");
  if (!result) std::cerr << result.error() << '\n';
  auto retired = owner->node.TearDown(owner->entered);
  if (!retired) {
    std::cerr << retired.error() << '\n';
    std::ignore = owner.release();
  }
  return result && retired ? 0 : 1;
}
