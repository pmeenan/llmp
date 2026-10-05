// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Manual, stage-zero-only original-input replay through the production
// assistant graph/plan. These immutable arrays are not a target checkpoint.
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
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
constexpr auto kHostBytes = std::uint64_t{256} << 20U;
constexpr std::array<std::string_view, 11> kFiles{
    "metadata.json",       "feature.f32",          "anchor.i32",          "local-k.f16",
    "local-v.f16",         "global-k.f16",         "global-v.f16",        "local-positions.i32",
    "local-membership.u8", "global-positions.i32", "global-membership.u8"};
constexpr std::array<std::uint64_t, 11> kSizes{0,      11264, 4,    1048576, 1048576, 524288,
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
                      std::uint32_t heads, std::uint32_t capacity, std::uint32_t owner) {
  auto domain = metadata.find(key);
  if (!domain) return Error("fixture cache descriptor absent");
  const auto pitch = std::int64_t{d} * heads * 2;
  const std::array<std::int64_t, 4> view_ne{d, heads, 256, 1}, root_ne{d * heads, capacity, 2, 1},
      view_nb{2, d * 2, pitch, pitch * capacity},
      root_nb{2, pitch, pitch * capacity, pitch * capacity * 2};
  for (const auto name : {"k", "v"}) {
    auto tensor = domain->find(name);
    if (!tensor || !Integer(*tensor, "relative_offset", pitch * capacity * owner))
      return Error("fixture cache offset differs");
    auto view = tensor->find("view"), root = tensor->find("root");
    if (!view || !root || !Integer(*view, "type", 1) || !Integer(*root, "type", 1) ||
        !Numbers(*view, "ne", view_ne) || !Numbers(*root, "ne", root_ne) ||
        !Numbers(*view, "nb", view_nb) || !Numbers(*root, "nb", root_nb))
      return Error("fixture original cache descriptor differs");
  }
  return {};
}
struct OwnerInputs {
  std::array<std::vector<std::byte>, 11> data;
  std::vector<float> initial;
  std::int32_t anchor = 0;
};
en::Status ReadOwner(const std::filesystem::path& input, const std::filesystem::path& manifest,
                     std::uint32_t owner, OwnerInputs& out) {
  std::error_code manifest_error;
  const auto manifest_size = std::filesystem::file_size(manifest, manifest_error);
  if (manifest_error || manifest_size > 16384) return Error("fixture manifest size");
  auto manifest_data = Read(manifest, manifest_size);
  if (!manifest_data || manifest_data->size() > 16384) return Error("fixture manifest size");
  auto doc = js::Parse(
      std::string_view(reinterpret_cast<const char*>(manifest_data->data()), manifest_data->size()),
      {.max_bytes = 16384, .max_depth = 8, .max_values = 256, .max_string_bytes = 8192});
  if (!doc) return Error("fixture manifest JSON");
  auto role = doc->root().find("role"), pin = doc->root().find("original_pin");
  if (!role || role->string() != "stage0 input only; no stock later input/output identities" ||
      !pin || pin->string() != "b29c606e28a01b1bc8c1351026a0fa6e616bf6c4")
    return Error("fixture stage0 manifest identity differs");
  auto files = doc->root().find("files");
  if (!files || !files->is_object() || files->size() != kFiles.size())
    return Error("fixture input allowlist differs");
  for (std::size_t i = 0; i < kFiles.size(); ++i) {
    auto declared = files->find(kFiles[i]);
    auto size = declared ? declared->find("bytes") : std::nullopt;
    auto count = size ? size->int64() : std::nullopt;
    if (!count || *count <= 0 ||
        (i == 0 ? *count > 2048 : *count != static_cast<std::int64_t>(kSizes[i])))
      return Error("fixture input manifest bytes differ");
    auto hash = declared->find("sha256");
    auto bytes = Read(input / kFiles[i], static_cast<std::uint64_t>(*count));
    if (!hash || !hash->is_string() || !bytes || Hash(*bytes) != hash->string())
      return Error("fixture input hash differs");
    out.data[i] = std::move(*bytes);
  }
  auto metadata = js::Parse(
      std::string_view(reinterpret_cast<const char*>(out.data[0].data()), out.data[0].size()),
      {.max_bytes = 2048, .max_depth = 8, .max_values = 256, .max_string_bytes = 1024});
  if (!metadata) return Error("fixture metadata JSON");
  const auto m = metadata->root();
  for (const auto [key, value] : std::initializer_list<std::pair<std::string_view, std::int64_t>>{
           {"version", 1},
           {"owner", owner},
           {"stream", owner},
           {"sequence", owner},
           {"completed_endpoint", 64 + owner},
           {"query_position", 64 + owner},
           {"feature_position", 63 + owner},
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
  if (auto r = Descriptor(m, "local_descriptors", 256, 8, 1280, owner); !r) return r;
  if (auto r = Descriptor(m, "global_descriptors", 512, 2, 4096, owner); !r) return r;
  std::memcpy(&out.anchor, out.data[2].data(), sizeof(out.anchor));
  if (out.anchor < 0 || out.anchor >= 262144) return Error("fixture anchor is not canonical");
  out.initial.resize(2816);
  std::memcpy(out.initial.data(), out.data[1].data(), out.data[1].size());
  if (!std::ranges::all_of(out.initial, [](float f) { return std::isfinite(f); }))
    return Error("fixture feature nonfinite");
  for (std::size_t i = 3; i <= 6; ++i)
    for (std::size_t at = 0; at < out.data[i].size(); at += 2) {
      ggml_fp16_t value;
      std::memcpy(&value, out.data[i].data() + at, 2);
      if (!std::isfinite(ggml_fp16_to_fp32(value))) return Error("fixture padded KV nonfinite");
      const auto cell = at / ((i < 5 ? 2048U : 1024U) * sizeof(ggml_fp16_t));
      if (cell >= 64 + owner && value != 0)
        return Error("fixture unused padded KV is not initialized zero");
    }
  for (const auto [positions, membership, capacity] :
       {std::tuple{7U, 8U, 1280U}, std::tuple{9U, 10U, 4096U}})
    for (std::uint32_t cell = 0; cell < capacity; ++cell) {
      std::int32_t position;
      std::memcpy(&position, out.data[positions].data() + cell * 4, 4);
      if (position != (cell < 64 + owner ? static_cast<std::int32_t>(cell) : -1) ||
          out.data[membership][cell] != (cell < 64 + owner ? std::byte{1} : std::byte{0}))
        return Error("fixture physical occupancy differs from frozen prefix");
    }
  return {};
}
class Replay final : public en::PagedModel {
 public:
  explicit Replay(en::PagedNode& node) : node_(node), resources_(node, 0, 0), runs_(true) {}
  en::Status Setup(const std::filesystem::path& input) {
    if (!node_.ChargeHost(kHostBytes, false)) return Error("C2 caller host budget");
    charged_ = true;
    for (unsigned owner = 0; owner < 2; ++owner)
      if (auto s = ReadOwner(input / ("owner-" + std::to_string(owner)),
                             input / ("owner-" + std::to_string(owner) + "-manifest.json"), owner,
                             owners_[owner]);
          !s)
        return s;
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
    if (auto r = resources_.Map(cache_, "immutable assistant fixture KV", state_.bytes * 2,
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
    for (unsigned owner = 0; owner < 2; ++owner)
      target_model_.slots.push_back({cache_.base + owner * state_.bytes, state_.bytes});
    model_.profile = &assistant_profile_;
    model_.binding = &assistant_binding_;
    model_.target = &target_model_;
    for (std::uint32_t i = 0; i < assistant_.artifact().resources().size(); ++i)
      model_.resources.push_back(
          {assistant_.resource_address(i), assistant_.artifact().resources()[i].readable.value()});

    shapes_[0].segments.push_back({0, 64, 256, 256});
    shapes_[1].segments.push_back({1, 65, 256, 256});
    shapes_[2].segments = {{0, 64, 256, 256}, {1, 65, 256, 256}};
    auto measuring = resources_.MeasuringContext();
    if (!measuring) return Error(measuring.error());
    std::uint64_t activation = 0, staging_bytes = 0;
    for (const auto& shape : shapes_) {
      auto p = en::PlanGemma4Assistant(model_, shape, kg::DeviceChoicesOf(**measuring), 0, 0);
      if (!p) return Error(p.error());
      auto scratch = kg::PlanScratch(**measuring, (*p)->plan);
      if (!scratch) return Error(scratch.error().detail);
      scratch_ = std::max(scratch_, *scratch);
      activation = std::max(activation, (*p)->placement.extent);
      staging_bytes = std::max(staging_bytes, (*p)->inputs_bytes);
    }
    scratch_ = Round(scratch_ + (1U << 20U), en::kPagedExtent);
    if (auto s = node_.MapWorkspace(Round(activation, en::kPagedExtent), scratch_); !s) return s;
    auto staging = resources_.Pinned(Round(staging_bytes, en::kPagedExtent));
    auto heads = resources_.Pinned(2 * 262144 * sizeof(float)),
         features = resources_.Pinned(2 * 2816 * sizeof(float)),
         witness = resources_.Pinned(1048576);
    if (!staging || !heads || !features || !witness) return Error("C2 pinned allocation");
    runs_.SetStaging(*staging, Round(staging_bytes, en::kPagedExtent));
    head_ = *heads;
    feature_ = *features;
    witness_ = *witness;
    node_.SetHostFloor(kHostBytes);
    auto fixed = node_.catalog().OccupancyOf(node_.domain()).Total().value();
    if (auto s = node_.Start(jitllm::base::Bytes(
            fixed + (target_.extents().size() + assistant_.extents().size()) * en::kPagedExtent +
            kHostBytes));
        !s)
      return s;
    if (auto s = target_.Register(node_, 0); !s) return s;
    if (auto s = assistant_.Register(node_, 0); !s) return s;
    auto extents = managed_extents();
    if (auto s = node_.scheduler().PinPlaces(extents); !s) return Error(sc::ToString(s.error()));
    auto own = resources_.extents();
    extents.insert(extents.end(), own.begin(), own.end());
    for (auto* workspace : {&node_.activations(), &node_.pool()})
      extents.insert(extents.end(), workspace->extents.begin(), workspace->extents.end());
    auto closure = node_.catalog().ClosureOfExtents(extents);
    if (!closure) return Error(jitllm::catalog::ToString(closure.error()));
    closure_ = std::move(*closure);
    if (auto s = resources_.BindLaunch(scratch_); !s) return s;
    runs_.SetLaunch(&resources_.launch());
    for (unsigned i = 0; i < 3; ++i) {
      auto p = en::PlanGemma4Assistant(model_, shapes_[i], kg::DeviceChoicesOf(resources_.launch()),
                                       node_.activations().base, node_.activations().bytes);
      if (!p) return Error(p.error());
      plans_[i] = std::move(*p);
      if (auto s = en::BindPlanned(*plans_[i], resources_.launch(), resources_.registry(),
                                   "C2 assistant fixture");
          !s)
        return Error(s.error());
    }
    heads_.resize(2 * 262144);
    features_.resize(2 * 2816);
    node_.Run();
    return {};
  }
  en::Status Execute(const std::filesystem::path& output,
                     const std::filesystem::path& incoming_path, bool timed) {
    std::vector<float> fixed;
    std::array<std::int32_t, 6> fixed_anchors{};
    if (!incoming_path.empty()) {
      auto f = Read(incoming_path / "incoming-feature.f32", 3 * 2 * 2816 * sizeof(float));
      auto a = Read(incoming_path / "incoming-anchor.i32", 6 * sizeof(std::int32_t));
      if (!f || !a) return Error("posthoc incoming dimensions differ");
      fixed.resize(3 * 2 * 2816);
      std::memcpy(fixed.data(), f->data(), f->size());
      std::memcpy(fixed_anchors.data(), a->data(), a->size());
      if (!std::ranges::all_of(fixed, [](float x) { return std::isfinite(x); }) ||
          !std::ranges::all_of(fixed_anchors, [](auto x) { return x >= 0 && x < 262144; }))
        return Error("posthoc incoming values invalid");
      for (unsigned owner = 0; owner < 2; ++owner)
        if (std::memcmp(fixed.data() + owner * 2816, owners_[owner].initial.data(),
                        2816 * sizeof(float)) != 0 ||
            fixed_anchors[owner] != owners_[owner].anchor)
          return Error("posthoc initial row differs from authenticated stage zero");
    }
    if (timed && fixed.empty()) return Error("timing requires authenticated fixed incoming rows");
    std::error_code ec;
    if (std::filesystem::exists(output, ec) || !std::filesystem::create_directory(output, ec) || ec)
      return Error("C2 output directory must be new");
    return node_.WithRequest(0, closure_, "C2 immutable assistant replay", [&]() -> en::Status {
      for (const auto& plan : plans_) {
        std::vector<const ggml_tensor*> caches;
        for (const auto& segment : plan->graph.segments)
          for (const auto& [k, v] : segment.caches) {
            caches.push_back(k);
            caches.push_back(v);
          }
        en::Coverage coverage;
        en::CheckCoverage(node_, 0, plan->graph.nodes,
                          {.state = caches, .inputs = plan->graph.inputs}, coverage);
        if (coverage.violations) return Error("C2 catalog coverage: " + coverage.first_violation);
      }
      // Initialize all physical borrowed cache cells, including cells beyond the
      // 256-cell read views. Each pinned upload retires before its buffer is reused.
      for (unsigned owner = 0; owner < 2; ++owner)
        for (unsigned array = 0; array < 4; ++array) {
          const auto& tensor = state_.tensors[(array < 2 ? 28U : 29U) * 2 + array % 2];
          std::memset(witness_, 0, 1048576);
          for (std::uint64_t offset = 0; offset < tensor.bytes; offset += 1048576) {
            const auto bytes = std::min<std::uint64_t>(1048576, tensor.bytes - offset);
            if (auto r = Copy(cache_.base + owner * state_.bytes + tensor.offset + offset, witness_,
                              bytes, jitllm::providers::CopyKind::kHostToDevice);
                !r)
              return r;
          }
          const auto& data = owners_[owner].data[array + 3];
          std::memcpy(witness_, data.data(), data.size());
          if (auto r = Copy(cache_.base + owner * state_.bytes + tensor.offset, witness_,
                            data.size(), jitllm::providers::CopyKind::kHostToDevice);
              !r)
            return r;
        }
      if (auto r = Copy(assistant_.resource_address(assistant_binding_.rope_freqs.index), witness_,
                        256 * sizeof(float), jitllm::providers::CopyKind::kDeviceToHost);
          !r)
        return r;
      if (auto r = md::CheckGemma4RopeFactors(profile_,
                                              std::span(static_cast<const float*>(witness_), 256));
          !r)
        return Error(r.error());
      auto before = Witness();
      if (!before) return Error(before.error());
      std::cout << "physical_cache_before=" << *before << '\n';
      for (const bool joined : {false, true}) {
        const std::string mode = joined ? "joined" : "serial";
        for (const unsigned steps : {1U, 3U}) {
          std::array<std::array<std::string, 2>, 3> expected;
          for (unsigned repeat = 0; repeat < 2; ++repeat) {
            std::vector<float> incoming(2 * 2816);
            std::array<std::int32_t, 2> anchors{};
            for (unsigned owner = 0; owner < 2; ++owner) {
              std::copy(owners_[owner].initial.begin(), owners_[owner].initial.end(),
                        incoming.begin() + owner * 2816);
              anchors[owner] = owners_[owner].anchor;
            }
            for (unsigned step = 0; step < steps; ++step) {
              if (!fixed.empty()) {
                std::copy_n(fixed.begin() + step * 2 * 2816, 2 * 2816, incoming.begin());
                std::copy_n(fixed_anchors.begin() + step * 2, 2, anchors.begin());
              }
              const auto input_anchors = anchors;
              if (auto r = Wave(joined, incoming, anchors); !r) return r;
              const std::array<std::string, 2> hashes{Hash(std::as_bytes(std::span(heads_))),
                                                      Hash(std::as_bytes(std::span(features_)))};
              if (!repeat)
                expected[step] = hashes;
              else if (hashes != expected[step])
                return Error("C2 endogenous repeat differs");
              for (const auto [name, bytes] :
                   {std::pair{"head", std::as_bytes(std::span(heads_))},
                    std::pair{"projection", std::as_bytes(std::span(features_))}}) {
                std::ofstream file(output / (mode + "-chain" + std::to_string(steps) + "-repeat" +
                                             std::to_string(repeat) + "-step" +
                                             std::to_string(step) + "-" + name + ".f32"),
                                   std::ios::binary | std::ios::noreplace);
                if (!file.write(reinterpret_cast<const char*>(bytes.data()),
                                static_cast<std::streamsize>(bytes.size())))
                  return Error("C2 output write");
                file.flush();
                if (!file) return Error("C2 output flush");
              }
              std::cout << "mode=" << mode << " posthoc=" << !fixed.empty() << " chain=" << steps
                        << " repeat=" << repeat << " step=" << step << " P0=64 P1=65"
                        << " input0=" << input_anchors[0] << " input1=" << input_anchors[1]
                        << " next0=" << anchors[0] << " next1=" << anchors[1]
                        << " head_sha256=" << hashes[0] << " projection_sha256=" << hashes[1]
                        << '\n';
              incoming = features_;
            }
            auto after = Witness();
            if (!after) return Error(after.error());
            if (*after != *before) return Error("C2 physical cache changed");
          }
        }
        if (timed) {
          std::vector<float> incoming(2 * 2816);
          std::array<std::int32_t, 2> anchors{};
          auto one = [&](unsigned step) {
            std::copy_n(fixed.begin() + step * 2 * 2816, 2 * 2816, incoming.begin());
            std::copy_n(fixed_anchors.begin() + step * 2, 2, anchors.begin());
            return Wave(joined, incoming, anchors);
          };
          for (unsigned step = 0; step < 3; ++step)
            if (auto r = one(step); !r) return r;
          auto first = Witness();
          if (!first || *first != *before) return Error("C2 paid pre-witness");
          const auto captured = stats_.captured, replayed = stats_.replayed;
          std::array<std::array<std::int32_t, 2>, 32> winners{};
          const auto start = std::chrono::steady_clock::now();
          for (unsigned wave = 0; wave < 32; ++wave) {
            if (auto r = one(wave % 3); !r) return r;
            winners[wave] = anchors;
          }
          const auto seconds =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
          auto last = Witness();
          if (!last || *last != *before) return Error("C2 paid post-witness");
          for (const auto [name, bytes] :
               {std::pair{"paid-winners.i32",
                          std::span<const std::byte>(std::as_bytes(std::span(winners)))},
                std::pair{"paid-last-heads.f32", std::as_bytes(std::span(heads_))},
                std::pair{"paid-last-postprojection.f32", std::as_bytes(std::span(features_))}}) {
            std::ofstream file(output / (mode + "-" + name),
                               std::ios::binary | std::ios::noreplace);
            if (!file.write(reinterpret_cast<const char*>(bytes.data()),
                            static_cast<std::streamsize>(bytes.size())))
              return Error("C2 paid output write");
            file.flush();
            if (!file) return Error("C2 paid output flush");
          }
          std::cout.precision(12);
          std::cout << "paid_mode=" << mode << " waves=32 rows=64 seconds=" << seconds
                    << " captured_delta=" << stats_.captured - captured
                    << " replayed_delta=" << stats_.replayed - replayed
                    << " winner_sha256=" << Hash(std::as_bytes(std::span(winners))) << '\n';
        }
      }
      if (stats_.replayed == 0) return Error("C2 replay witness absent");
      std::cout << "captured=" << stats_.captured << " replayed=" << stats_.replayed << '\n';
      return {};
    });
  }
  std::uint32_t stream() const override { return 0; }
  const jitllm::catalog::Closure& fence_closure() const override { return closure_; }
  std::vector<jitllm::catalog::ExtentId> managed_extents() const override {
    auto ids = target_.extents();
    auto other = assistant_.extents();
    ids.insert(ids.end(), other.begin(), other.end());
    return ids;
  }
  en::Status Release() override {
    for (auto& run : plan_runs_) run.DropGraph();
    for (auto& plan : plans_) plan.reset();
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
  en::Status Copy(std::uint64_t device, void* host, std::uint64_t bytes,
                  jitllm::providers::CopyKind kind) {
    return node_.Job(
        closure_,
        [&](jitllm::providers::NativeStream stream) {
          const bool upload = kind == jitllm::providers::CopyKind::kHostToDevice;
          return jitllm::providers::CopyAsync(stream, upload ? Pointer(device) : host,
                                              upload ? host : Pointer(device), bytes, kind)
                         .ok()
                     ? sc::JobResult::kQueued
                     : sc::JobResult::kUnknown;
        },
        "C2 pinned copy", 0);
  }
  en::Status Wave(bool joined, std::span<const float> incoming,
                  std::array<std::int32_t, 2>& anchors) {
    for (unsigned group = 0; group < (joined ? 1U : 2U); ++group) {
      const auto index = joined ? 2U : group, count = joined ? 2U : 1U;
      auto& plan = *plans_[index];
      const std::array<en::Gemma4AssistantInput, 2> input{
          {{joined ? 0U : group, joined ? 64U : 64U + group, anchors[joined ? 0 : group]},
           {1, 65, anchors[1]}}};
      auto source = en::Gemma4AssistantSources(
          plan.graph, std::span(input).first(count),
          incoming.subspan(joined ? 0 : group * 2816, count * 2816), false, kHostBytes);
      if (!source) return Error(source.error());
      auto staged = runs_.Stage(source->sources, 0);
      if (!staged) return Error(staged.error());
      const auto row = joined ? 0U : group;
      const std::array<en::RunCopy, 2> outputs{
          {{Address(head_) + row * 262144 * sizeof(float), Address(plan.graph.logits->data),
            count * 262144 * sizeof(float)},
           {Address(feature_) + row * 2816 * sizeof(float), Address(plan.graph.next_features->data),
            count * 2816 * sizeof(float)}}};
      auto& run = plan_runs_[index];
      auto posted = node_.Job(
          closure_,
          [&](jitllm::providers::NativeStream stream) {
            auto queued = runs_.Queue(run, *staged, {}, *plan.bound, outputs, run.CaptureDue(true),
                                      stats_, stream);
            if (!queued.result) return sc::JobResult::kUnknown;
            en::Count(stats_, queued.path);
            return sc::JobResult::kQueued;
          },
          "C2 completed assistant wave", 0);
      if (!posted) return posted;
    }
    const auto heads = std::span(static_cast<const float*>(head_), 2 * 262144),
               features = std::span(static_cast<const float*>(feature_), 2 * 2816);
    if (auto r = en::CheckGemma4AssistantOutputs(2, 2816, heads, features); !r) return r;
    std::copy(heads.begin(), heads.end(), heads_.begin());
    std::copy(features.begin(), features.end(), features_.begin());
    for (unsigned owner = 0; owner < 2; ++owner) {
      auto row = heads.subspan(owner * 262144, 262144);
      anchors[owner] =
          static_cast<std::int32_t>(std::max_element(row.begin(), row.end()) - row.begin());
    }
    return {};
  }
  std::expected<std::string, std::string> Witness() {
    jitllm::base::Sha256 hash;
    for (unsigned owner = 0; owner < 2; ++owner)
      for (unsigned array = 0; array < 4; ++array) {
        const auto& tensor = state_.tensors[(array < 2 ? 28U : 29U) * 2 + array % 2];
        const auto& input = owners_[owner].data[array + 3];
        for (std::uint64_t offset = 0; offset < tensor.bytes; offset += 1048576) {
          const auto bytes = std::min<std::uint64_t>(1048576, tensor.bytes - offset);
          if (auto r = Copy(cache_.base + owner * state_.bytes + tensor.offset + offset, witness_,
                            bytes, jitllm::providers::CopyKind::kDeviceToHost);
              !r)
            return Error(r.error());
          const auto data = std::span(static_cast<const std::byte*>(witness_), bytes);
          for (std::size_t i = 0; i < data.size(); ++i) {
            const auto expected = offset + i < input.size() ? input[offset + i] : std::byte{0};
            if (data[i] != expected) return Error("C2 immutable physical cache differs");
          }
          hash.Update(data);
        }
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
  std::array<kg::Gemma4AssistantShape, 3> shapes_;
  std::array<std::unique_ptr<en::Gemma4AssistantPlanned>, 3> plans_;
  en::GraphRuns runs_;
  std::array<en::PlanRuns, 3> plan_runs_;
  en::GraphStats stats_;
  jitllm::catalog::Closure closure_;
  std::array<OwnerInputs, 2> owners_;
  std::vector<float> heads_, features_;
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
  if (argc < 3 || argc > 5 || (argc == 5 && std::string_view(argv[4]) != "time")) {
    std::cerr
        << "usage: jitllm_gemma_assistant_wave_fixture INPUT_ROOT NEW_OUTPUT [INCOMING [time]]\n";
    return 2;
  }
  auto owner = std::make_unique<Lifetime>();
  auto result = owner->node.Open();
  if (result) result = owner->replay.Setup(argv[1]);
  if (result) result = owner->replay.Execute(argv[2], argc >= 4 ? argv[3] : "", argc == 5);
  if (!result) std::cerr << result.error() << '\n';
  auto retired = owner->node.TearDown(owner->entered);
  if (!retired) {
    std::cerr << retired.error() << '\n';
    std::ignore = owner.release();
  }
  return result && retired ? 0 : 1;
}
