// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/qwen_image_runner.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <format>
#include <fstream>
#include <span>
#include <string_view>
#include <tuple>
#include <utility>

#include "artifact/composition.h"
#include "base/sha256.h"
#include "chat/chat.h"
#include "engine/planned.h"
#include "engine/runner_resources.h"
#include "engine/support.h"
#include "execution/registry.h"
#include "kernels/image/gemm.h"
#include "kernels/image/implementations.h"
#include "kernels/image/ops.h"
#include "kernels/image/pipeline.h"
#include "providers/device_runtime.h"
#include "scheduler/commands.h"
#include "scheduler/scheduler.h"
#include "tokenizer/hf.h"
#include "tokenizer/tokenizer.h"

namespace llmp::engine {

namespace {

namespace ja = llmp::artifact;
namespace ki = llmp::kernels::image;
namespace md = llmp::model;
namespace sc = llmp::scheduler;
using catalog::ExtentId;
using catalog::MemoryClass;
using support::Error;
using support::Round;
using support::Seconds;
using Bf16 = std::uint16_t;
using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kExtent = kPagedExtent;
// The least a recorded step is charged at, whatever the device's free memory
// showed across its recording (other work moves it too).
constexpr std::uint64_t kGraphFloorBytes = std::uint64_t{16} << 20U;

Status Checked(providers::DeviceStatus result, std::string_view what) {
  if (!result.ok()) {
    return Error(std::format("{}: {}", what, result.text()));
  }
  return {};
}

constexpr auto kToDevice = providers::CopyKind::kHostToDevice;
constexpr auto kToHost = providers::CopyKind::kDeviceToHost;

template <typename T>
T* At(std::uint64_t address) {
  return reinterpret_cast<T*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::expected<std::vector<Bf16>, std::string> ReadBf16(const std::filesystem::path& p,
                                                       std::size_t count) {
  std::ifstream in(p, std::ios::binary);
  if (!in) {
    return Error(std::format("{}: cannot open", p.string()));
  }
  std::vector<Bf16> out(count);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * 2));
  if (static_cast<std::size_t>(in.gcount()) != count * 2 || in.peek() != EOF) {
    return Error(std::format("{}: not {} BF16 values", p.string(), count));
  }
  return out;
}

struct Component {
  PagedWeights weights;  // its artifact and the groups its phase reads
  std::vector<std::uint32_t> bound;
  std::vector<std::uint64_t> tensor;  // bound tensors' device addresses
  catalog::Closure closure;           // its phase's
};

}  // namespace

struct QwenImageRunner::State {
  State(PagedNode& node, int owner, std::uint32_t stream) : resources(node, owner, stream) {}

  const md::QwenImageProfile& profile = md::QwenImage21();
  std::array<Component, 3> c;  // text encoder, denoiser, VAE
  std::vector<std::int32_t> ids;
  std::size_t drop = 0;
  std::int64_t text = 0;
  std::uint32_t grid = 0;
  std::int64_t image = 0;
  md::QwenImageSchedule schedule;
  std::vector<Bf16> noise0;
  md::RotaryTables rot;
  std::vector<float> freqs;
  std::vector<std::uint64_t> vae_f32_bytes;

  // The plan, bound (D-053).
  std::optional<llmp::execution::Registry> registry;
  std::unique_ptr<ki::QwenImagePipeline> pipeline;

  // Its own memory, the cuBLAS workspace and handle, the staging.
  RunnerResources resources;
  Mapped own_memory;
  std::unique_ptr<ki::LtGemm> lt;  // on the cuBLAS workspace
  ki::DitMemory own;
  std::uint64_t own_bytes = 0;
  std::uint64_t text_bytes = 0, dit_bytes = 0, vae_bytes = 0;
  ki::TextMemory tw;
  ki::DitMemory dw;
  ki::VaeMemory vw;
  std::vector<std::uint64_t> vae_weights;  // BF16 copies in the workspace
  std::byte* in = nullptr;                 // pinned: what a job uploads
  std::uint64_t in_bytes = 0;
  std::byte* out = nullptr;  // pinned: what a job downloads
  std::uint64_t out_bytes = 0;

  // A later step, recorded once (QwenImageOptions::graphs).
  providers::RecordedWork graph;
  // What it holds (driver memory outside the catalog's extents: the drop in
  // the device's free memory across its recording, at least
  // kGraphFloorBytes), charged to the node; its recording's seconds (what
  // recording it again costs); its last replay (planned.h NextPlanUse).
  std::uint64_t graph_bytes = 0;
  std::uint64_t graph_measured = 0;  // the drop itself, before the floor
  double graph_seconds = 0;
  std::uint64_t graph_used = 0;
  memory::ReclaimStamp graph_stamp;  // the reclaim order's at its last replay
  PlanAccount account;

  // The last generation's timings.
  double encode = 0;
  std::vector<double> steps;
  double decode = 0;
  std::uint64_t generations = 0;

  State(const State&) = delete;
  State& operator=(const State&) = delete;
  State(State&&) = delete;
  State& operator=(State&&) = delete;
  ~State() { DropGraph(); }
  // Only once no queued launch of it remains (Release, after the node's
  // jobs are done).
  void DropGraph() { graph.Reset(); }
};

QwenImageRunner::QwenImageRunner(PagedNode& node, const QwenImageOptions& options, int owner,
                                 std::uint32_t stream)
    : node_(node),
      o_(options),
      owner_(owner),
      stream_(stream),
      s_(std::make_unique<State>(node, owner, stream)) {}

QwenImageRunner::~QwenImageRunner() = default;

std::vector<ExtentId> QwenImageRunner::weights() const {
  std::vector<ExtentId> all;
  for (const Component& c : s_->c) {
    all.insert(all.end(), c.weights.extents().begin(), c.weights.extents().end());
  }
  return all;
}

std::uint64_t QwenImageRunner::weight_read_bytes() const {
  std::uint64_t bytes = 0;
  for (const Component& c : s_->c) {
    bytes += c.weights.read_bytes();
  }
  return bytes;
}

Status QwenImageRunner::Setup() {
  State& s = *s_;
  const md::QwenImageProfile& profile = s.profile;

  // The plan, bound against this build's registry.
  auto registry = llmp::execution::Registry::Create(ki::Implementations());
  if (!registry) {
    return Error("the image's registry: " + registry.error().detail);
  }
  s.registry.emplace(std::move(*registry));
  auto choices = ki::QwenImageChoices(o_.plan);
  if (!choices) {
    return Error("the image's plan: " + choices.error());
  }
  auto plan = llmp::execution::Plan::Build(*s.registry, *choices);
  if (!plan) {
    return Error("the image's plan: " + plan.error().detail);
  }
  auto pipeline = ki::QwenImagePipeline::Bind(*s.registry, *plan, profile);
  if (!pipeline) {
    return Error("the image's plan: " + pipeline.error());
  }
  s.pipeline = std::move(*pipeline);

  auto composition = ja::OpenComposition(o_.store / o_.composition);
  if (!composition) {
    return Error("composition: " + composition.error().ToString());
  }
  if (composition->architecture() != profile.pipeline_architecture) {
    return Error("the composition is not a " + std::string(profile.pipeline_architecture));
  }
  const std::array<std::tuple<std::string_view, std::string_view, std::vector<md::QwenImageTensor>>,
                   3>
      roles = {
          {{"text_encoder", profile.text_architecture, md::TextEncoderTensors(profile.text)},
           {"transformer", profile.denoiser_architecture, md::DenoiserTensors(profile.denoiser)},
           {"vae", profile.vae_architecture, md::VaeDecoderTensors(profile.vae)}}};
  for (std::size_t i = 0; i < roles.size(); ++i) {
    const auto& [role, arch, tensors] = roles.at(i);
    const ja::CompositionComponent* found = composition->Find(role);
    if (found == nullptr) {
      return Error(std::format("the composition has no {}", role));
    }
    ja::OpenOptions options;
    options.expected_id = found->artifact;
    Component& c = s.c.at(i);
    if (auto r = c.weights.Open(o_.store / found->artifact, options); !r) {
      return Error(std::format("{}: {}", role, r.error()));
    }
    const ja::Artifact& a = c.weights.artifact();
    if (a.model().architecture != found->architecture) {
      return Error(
          std::format("{}: the artifact's architecture differs from the composition's", role));
    }
    auto bound = md::BindQwenImageComponent(tensors, arch, a);
    if (!bound) {
      return Error(std::format("{}: {}", role, bound.error()));
    }
    c.bound = std::move(*bound);
  }

  // The prompt's tokens (the native tokenizer and renderer, D-088).
  const auto tokenizer_json = composition->Metadata("tokenizer.json");
  if (!tokenizer_json) {
    return Error("the composition keeps no tokenizer.json");
  }
  auto spec = llmp::tokenizer::ReadHfTokenizer(*tokenizer_json);
  if (!spec) {
    return Error("tokenizer: " + spec.error().ToString());
  }
  auto tokenizer = llmp::tokenizer::Tokenizer::Create(std::move(*spec));
  if (!tokenizer) {
    return Error("tokenizer: " + tokenizer.error().ToString());
  }
  auto rendered = llmp::chat::RenderQwenImagePrompt(o_.prompt, *tokenizer);
  if (!rendered) {
    return Error("prompt: " + rendered.error().ToString());
  }
  std::vector<llmp::tokenizer::TokenId> ids;
  if (auto e =
          tokenizer->EncodeMarked(rendered->rendered.text, rendered->rendered.specials, {}, ids);
      !e) {
    return Error("prompt: " + e.error().ToString());
  }
  s.drop = rendered->drop_tokens;
  if (ids.size() <= s.drop || ids.size() > 1024) {
    return Error("a prompt of 1 to 1,024 tokens after the system turn");
  }
  s.ids.assign(ids.begin(), ids.end());
  s.text = static_cast<std::int64_t>(ids.size() - s.drop);
  if (o_.size % 32 != 0 || o_.size < 64 || o_.size > 2048 || o_.steps < 2 || o_.steps > 100) {
    return Error("a size a multiple of 32 in 64-2048, and 2-100 steps");
  }
  s.grid = o_.size / profile.vae.spatial_factor;
  s.image = std::int64_t{s.grid} * s.grid;
  auto schedule = md::QwenImageSchedulerSigmas(profile.scheduler, o_.steps,
                                               static_cast<std::uint32_t>(s.image));
  if (!schedule) {
    return Error(schedule.error());
  }
  s.schedule = std::move(*schedule);
  auto noise = ReadBf16(o_.noise, static_cast<std::size_t>(s.image * profile.denoiser.in_channels));
  if (!noise) {
    return std::unexpected(noise.error());
  }
  s.noise0 = std::move(*noise);
  s.rot = md::QwenImageTextRotary(profile.text, static_cast<std::uint32_t>(s.ids.size()));
  s.freqs =
      md::QwenImageDenoiserRotary(profile.denoiser, static_cast<std::uint32_t>(s.text), s.grid);

  // Each component's weights: the groups its bound tensors are in.
  for (Component& c : s.c) {
    if (!c.weights.opened()) {
      return Error("a component without its artifact");
    }
    const ja::Artifact& a = c.weights.artifact();
    std::vector<GroupPlace> place(a.groups().size(), GroupPlace::kNone);
    for (const std::uint32_t r : c.bound) {
      place[a.resources()[r].group] = GroupPlace::kDevice;
    }
    if (auto r = c.weights.Reserve(node_, place, {}); !r) {
      return r;
    }
  }
  const ja::Artifact& vae = s.c[2].weights.artifact();
  for (const std::uint32_t r : s.c[2].bound) {
    const auto& res = vae.resources()[r];
    if (res.bytes.value() % 4 != 0) {
      return Error(std::format("the VAE's {} is not F32", res.name));
    }
    s.vae_f32_bytes.push_back(res.bytes.value());
  }

  // Working memory: the image's own, and each phase's in the workspace.
  (void)ki::OwnLayout(0, profile.denoiser, s.text, s.image, s.own_bytes);
  (void)ki::TextLayout(0, profile.text, static_cast<std::int64_t>(s.ids.size()), s.text_bytes);
  (void)ki::DitLayout(0, profile.denoiser, s.text + s.image, s.dit_bytes);
  (void)ki::VaeLayout(0, profile.vae, s.grid, s.vae_f32_bytes, true, s.vae_weights, s.vae_bytes);
  work_bytes_ = Round(std::max({s.text_bytes, s.dit_bytes, s.vae_bytes}), kExtent);
  if (auto r = s.resources.Map(s.own_memory, "the image's own memory", s.own_bytes,
                               MemoryClass::kScratch);
      !r) {
    return r;
  }
  std::uint64_t unused = 0;
  s.own = ki::OwnLayout(s.own_memory.base, profile.denoiser, s.text, s.image, unused);

  if (auto r = s.resources.OpenCublas("the image's cuBLAS workspace"); !r) {
    return r;
  }
  // cuBLASLt shares the workspace: both queue on the image's one stream.
  auto lt = ki::LtGemm::Create(s.resources.cublas_workspace().base, s.resources.cublas_bytes());
  if (!lt) {
    return Error("cuBLASLt: " + lt.error());
  }
  s.lt = std::move(*lt);

  // Staging: the largest upload (the first step's rotary tables and latents
  // with the step's sinusoid and dt, or the text's inputs) and download (the
  // decoded image).
  const std::uint64_t text_in = Round(s.ids.size() * 4, 256) + (2 * Round(s.ids.size() * 256, 256));
  const std::uint64_t step_in = Round(s.freqs.size() * 4, 256) + Round(s.noise0.size() * 2, 256) +
                                Round(2 * std::uint64_t{profile.denoiser.timestep_dim} * 2, 256) +
                                256;
  s.in_bytes = std::max<std::uint64_t>({text_in, step_in, 4096});
  s.out_bytes = std::max<std::uint64_t>(
      std::uint64_t{profile.vae.out_channels} * o_.size * o_.size * 2,
      static_cast<std::uint64_t>(s.image * profile.denoiser.out_channels * 2));
  auto in = s.resources.Pinned(s.in_bytes);
  auto out = s.resources.Pinned(s.out_bytes);
  if (!in || !out) {
    return Error("pinned staging for the image");
  }
  s.in = static_cast<std::byte*>(*in);
  s.out = static_cast<std::byte*>(*out);
  return {};
}

Status QwenImageRunner::Register() {
  for (Component& c : s_->c) {
    if (auto r = c.weights.Register(node_, owner_); !r) {
      return r;
    }
  }
  // D-090, for every model: the places stay put for the model's life
  // (never unpinned; the pins go with the scheduler).
  if (auto pinned = node_.scheduler().PinPlaces(weights()); !pinned) {
    return Error(std::format("pinning the image's places: {}", sc::ToString(pinned.error())));
  }
  return {};
}

void QwenImageRunner::ReclaimCandidates(std::uint32_t owner, bool running,
                                        std::vector<memory::ReclaimCandidate>& out) {
  const State& s = *s_;
  if (s.graph.valid() && s.graph_bytes != 0 && s.graph_used < PlanStepStart()) {
    out.push_back({.kind = memory::ReclaimKind::kGraph,
                   .owner = owner,
                   .id = 1,
                   .bytes = s.graph_bytes,
                   .last_use = s.graph_used,
                   .restore_seconds = s.graph_seconds,
                   .running = running});
    memory::SetUse(out.back(), s.graph_stamp);
  }
}

std::uint64_t QwenImageRunner::graph_measured_bytes() const {
  const State& s = *s_;
  return s.graph.valid() ? s.graph_measured : 0;
}

std::uint64_t QwenImageRunner::Reclaim(memory::ReclaimKind kind, std::uint64_t id) {
  State& s = *s_;
  // Between jobs only (nothing in flight replays it); a step under way
  // keeps it.
  if (kind != memory::ReclaimKind::kGraph || id != 1 || !s.graph.valid() ||
      s.graph_used >= PlanStepStart()) {
    return 0;
  }
  s.DropGraph();
  const std::uint64_t bytes = s.graph_bytes;
  s.graph_bytes = 0;
  s.account.Uncharge(bytes);
  return bytes;
}

std::string QwenImageRunner::plan_report() const {
  const State& s = *s_;
  return s.graph_bytes == 0 ? std::string("a denoising step's graph, charged once recorded")
                            : std::format("a denoising step's graph of {:.1f} MiB",
                                          static_cast<double>(s.graph_bytes) / (1U << 20U));
}

Status QwenImageRunner::Bind() {
  State& s = *s_;
  s.account.Bind(
      [this](std::uint64_t bytes, bool required) { return node_.ChargeHost(bytes, required); },
      [this](std::uint64_t bytes) { node_.UnchargeHost(bytes); });
  auto& catalog = node_.catalog();
  // What every phase leases beside its component: the image's own memory,
  // the cuBLAS workspace and the staging, and the activations.
  std::vector<ExtentId> common = s.resources.extents();
  common.insert(common.end(), node_.activations().extents.begin(),
                node_.activations().extents.end());
  std::vector<ExtentId> all = common;
  for (Component& c : s.c) {
    std::vector<ExtentId> mine = c.weights.extents();
    mine.insert(mine.end(), common.begin(), common.end());
    c.closure = catalog.ClosureOfExtents(mine).value();
    all.insert(all.end(), c.weights.extents().begin(), c.weights.extents().end());
    c.tensor.clear();
    for (const std::uint32_t r : c.bound) {
      c.tensor.push_back(c.weights.resource_address(r));
    }
  }
  everything_ = catalog.ClosureOfExtents(all).value();
  fence_ = catalog.ClosureOfExtents(s.own_memory.extents).value();
  const std::uint64_t work = node_.activations().base;
  std::uint64_t bytes = 0;
  s.tw = ki::TextLayout(work, s.profile.text, static_cast<std::int64_t>(s.ids.size()), bytes);
  s.tw.tensors = s.c[0].tensor;
  s.tw.embeds = s.own.embeds;
  s.tw.drop = static_cast<std::int64_t>(s.drop);
  s.dw = ki::DitLayout(work, s.profile.denoiser, s.text + s.image, bytes);
  s.dw.tensors = s.c[1].tensor;
  s.dw.embeds = s.own.embeds;
  s.dw.txt = s.own.txt;
  s.dw.pk = s.own.pk;
  s.dw.pv = s.own.pv;
  s.dw.freqs = s.own.freqs;
  s.dw.latents = s.own.latents;
  s.dw.noise = s.own.noise;
  s.dw.text = s.text;
  s.dw.image = s.image;
  s.vw = ki::VaeLayout(work, s.profile.vae, s.grid, s.vae_f32_bytes, true, s.vae_weights, bytes);
  s.vw.f32 = s.c[2].tensor;
  s.vw.weights = s.vae_weights;
  s.vw.latents = s.own.latents;
  if (std::max({s.text_bytes, s.dit_bytes, s.vae_bytes}) > node_.activations().bytes) {
    return Error("the image's working memory exceeds the workspace");
  }
  return {};
}

// ------------------------------------------------------------------ encode

// The text encoder (the pipeline's Encode): Qwen3-VL's text path over every
// rendered token, the rows after the system turn kept.
Status QwenImageRunner::Encode() {
  State& s = *s_;
  const Component& c = s.c[0];
  const auto rows = static_cast<std::int64_t>(s.ids.size());
  const ki::TextMemory& w = s.tw;
  const ki::Handles handles{.blas = s.resources.cublas().native(), .lt = s.lt.get()};
  const auto started = Clock::now();
  Status ran;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    void* const st = native.handle;
    // The inputs: ids, then the rotary tables.
    std::uint64_t at = 0;
    for (auto [dst, src, bytes] :
         {std::tuple{w.ids, static_cast<const void*>(s.ids.data()), rows * 4},
          {w.cos, s.rot.cos.data(), rows * 256},
          {w.sin, s.rot.sin.data(), rows * 256}}) {
      std::memcpy(s.in + at, src, static_cast<std::size_t>(bytes));
      if (auto r = Checked(providers::CopyAsync(native, At<void>(dst), s.in + at,
                                                static_cast<std::size_t>(bytes), kToDevice),
                           "text inputs");
          !r) {
        ran = r;
        return sc::JobResult::kUnknown;
      }
      at += Round(static_cast<std::uint64_t>(bytes), 256);
    }
    Status r = Checked(providers::FillAsync(native, At<void>(w.bad), 0, 4), "a flag");
    r = r ? s.pipeline->Encode(w, handles, st) : r;
    r = r ? Checked(providers::CopyAsync(native, s.out, At<void>(w.bad), 4, kToHost), "the flag")
          : r;
    if (!r) {
      ran = r;
      return sc::JobResult::kUnknown;
    }
    return sc::JobResult::kQueued;
  };
  const Status posted = node_.Job(c.closure, std::move(job), "the image's encode", stream_);
  if (!ran) {
    return ran;
  }
  if (!posted) {
    return posted;
  }
  std::int32_t flag = 0;
  std::memcpy(&flag, s.out, 4);
  if (flag != 0) {
    return Error("a token id outside the embedding table");
  }
  s.encode = Seconds(Clock::now() - started);
  return {};
}

// ------------------------------------------------------------------ denoise

Status QwenImageRunner::Step(std::uint32_t index, bool hash, std::string* sha) {
  State& s = *s_;
  const md::QwenImageDenoiserProfile& p = s.profile.denoiser;
  const Component& c = s.c[1];
  const bool first = index == 0;
  const auto sinus = md::QwenImageTimestepSinusoid(p, s.schedule.timesteps[index]);
  // dt is a 0-dim F32 tensor times the BF16 noise: PyTorch rounds it to
  // BF16 first (the resident harness's rounding).
  const float dt = ki::EulerStepDt(s.schedule.sigmas[index + 1] - s.schedule.sigmas[index], true);
  const ki::Handles handles{.blas = s.resources.cublas().native(), .lt = s.lt.get()};
  const auto started = Clock::now();
  const PlanStep step;               // the recorded step stays while this one runs
  std::uint64_t recorded_bytes = 0;  // a recording made in this job: what it holds
  std::uint64_t recorded_measured = 0;
  double recorded_seconds = 0;
  Status ran;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    void* const st = native.handle;
    std::uint64_t at = 0;
    const auto upload = [&](std::uint64_t dst, const void* src, std::uint64_t bytes) -> Status {
      std::memcpy(s.in + at, src, bytes);
      auto r = Checked(providers::CopyAsync(native, At<void>(dst), s.in + at, bytes, kToDevice),
                       "an upload");
      at += Round(bytes, 256);
      return r;
    };
    Status r;
    if (first) {
      // The rotary frequencies, the text rows, then the initial latents.
      r = upload(s.dw.freqs, s.freqs.data(), s.freqs.size() * 4);
      r = r ? s.pipeline->TextRows(s.dw, handles, st) : r;
      r = r ? upload(s.dw.latents, s.noise0.data(), s.noise0.size() * 2) : r;
    }
    r = r ? upload(s.dw.sin, sinus.data(), sinus.size() * 2) : r;
    r = r ? upload(s.dw.dt, &dt, sizeof dt) : r;
    if (r && o_.graphs && index >= 2) {
      // Recorded on this job's stream (thread-local: nothing else on this
      // thread uses the device meanwhile), after the second step made
      // every product's descriptors; then replayed.
      if (!s.graph.valid()) {
        const auto recording = Clock::now();
        const std::size_t free_before =
            providers::QueryDeviceMemory().value_or(providers::DeviceMemoryInfo{}).free;
        r = Checked(providers::BeginRecording(native), "capture");
        if (r) {
          const Status queued = s.pipeline->Step(false, s.dw, handles, st);
          auto recorded = providers::EndRecording(native);  // ended even if `queued` failed
          if (!queued) {
            r = queued;  // the recording, if any, is dropped: nothing of it was launched
          } else if (!recorded) {
            r = Checked(recorded.error(), "end capture and instantiate");
          } else {
            s.graph = std::move(*recorded);
            const std::size_t free_after =
                providers::QueryDeviceMemory().value_or(providers::DeviceMemoryInfo{}).free;
            (void)providers::TakeLastError();
            recorded_measured = free_before > free_after ? free_before - free_after : 0;
            recorded_bytes = std::max<std::uint64_t>(recorded_measured, kGraphFloorBytes);
            recorded_seconds = Seconds(Clock::now() - recording);
          }
        }
      }
      r = r ? Checked(providers::Replay(s.graph, native), "graph launch") : r;
    } else {
      r = r ? s.pipeline->Step(first, s.dw, handles, st) : r;
    }
    if (r && hash) {
      r = Checked(
          providers::CopyAsync(native, s.out, At<void>(s.dw.noise),
                               static_cast<std::size_t>(s.image * p.out_channels * 2), kToHost),
          "the noise prediction");
    }
    if (!r) {
      ran = r;
      return sc::JobResult::kUnknown;
    }
    return sc::JobResult::kQueued;
  };
  const Status posted = node_.Job(c.closure, std::move(job), "a denoising step", stream_);
  if (recorded_bytes != 0 && s.graph.valid()) {
    // Recorded in this job (the device lane's thread): charged here, on the
    // driver's, once it completed. Required: it exists already.
    (void)s.account.Charge(recorded_bytes, true);
    s.graph_bytes = recorded_bytes;
    s.graph_measured = recorded_measured;
    s.graph_seconds = recorded_seconds;
  }
  if (s.graph.valid()) {
    s.graph_used = NextPlanUse();
    s.graph_stamp = memory::StampUse();
  }
  if (!ran) {
    return ran;
  }
  if (!posted) {
    return posted;
  }
  s.steps.push_back(Seconds(Clock::now() - started));
  if (hash && sha != nullptr) {
    llmp::base::Sha256 h;
    h.Update(std::span(s.out, static_cast<std::size_t>(s.image * p.out_channels * 2)));
    *sha = llmp::base::ToHex(h.Finish());
  }
  return {};
}

// ------------------------------------------------------------------ decode

// The VAE's weights rounded to BF16 in the workspace (in the layout the
// plan's convolution reads), then the decoder (the pipeline's Decode).
Status QwenImageRunner::Decode(std::string& sha) {
  State& s = *s_;
  const md::QwenImageVaeProfile& p = s.profile.vae;
  const Component& c = s.c[2];
  const ki::VaeMemory& w = s.vw;
  const std::size_t count = std::size_t{p.out_channels} * o_.size * o_.size;
  const ki::Handles handles{.blas = s.resources.cublas().native(), .lt = s.lt.get()};
  const auto started = Clock::now();
  Status ran;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    void* const st = native.handle;
    Status r = s.pipeline->ConvertVae(w, st);
    std::vector<Bf16> stats(std::size_t{2} * 64);
    for (std::size_t ch = 0; ch < 64; ++ch) {
      stats[ch] = md::ToBf16(p.latents_std.at(ch));
      stats[64 + ch] = md::ToBf16(p.latents_mean.at(ch));
    }
    std::memcpy(s.in, stats.data(), stats.size() * 2);
    r = r ? Checked(
                providers::CopyAsync(native, At<void>(w.stats), s.in, stats.size() * 2, kToDevice),
                "latent statistics")
          : r;
    r = r ? s.pipeline->Decode(w, handles, st) : r;
    r = r ? Checked(providers::CopyAsync(native, s.out, At<void>(w.buf.at(md::kVaeX)), count * 2,
                                         kToHost),
                    "the decoded image")
          : r;
    if (!r) {
      ran = r;
      return sc::JobResult::kUnknown;
    }
    return sc::JobResult::kQueued;
  };
  const Status posted = node_.Job(c.closure, std::move(job), "the image's decode", stream_);
  if (!ran) {
    return ran;
  }
  if (!posted) {
    return posted;
  }
  const std::span<const Bf16> decoded(reinterpret_cast<const Bf16*>(s.out), count);
  const auto pixels = md::QwenImagePixels(decoded, p.out_channels, o_.size, o_.size);
  sha = llmp::base::ToHex(llmp::base::Sha256().Update(std::as_bytes(std::span(pixels))).Finish());
  s.decode = Seconds(Clock::now() - started);
  return {};
}

// ------------------------------------------------------------------ the pipeline

Status QwenImageRunner::FirstOutput(std::string& sha) {
  s_->steps.clear();
  if (auto r = Encode(); !r) {
    return r;
  }
  return Step(0, true, &sha);
}

Status QwenImageRunner::Finish(std::string& sha) {
  for (std::uint32_t i = 1; i < o_.steps; ++i) {
    if (auto r = Step(i, false, nullptr); !r) {
      return r;
    }
  }
  if (auto r = Decode(sha); !r) {
    return r;
  }
  ++s_->generations;
  return {};
}

std::vector<std::filesystem::path> QwenImageRunner::data() const {
  std::vector<std::filesystem::path> dirs;
  for (const Component& c : s_->c) {
    if (c.weights.opened()) {
      dirs.push_back(o_.store / c.weights.artifact().id() / "data");
    }
  }
  return dirs;
}

std::string QwenImageRunner::Report() const {
  const State& s = *s_;
  std::vector<double> later(s.steps.begin() + (s.steps.empty() ? 0 : 1), s.steps.end());
  std::ranges::sort(later);
  std::string components;
  const std::array<std::string_view, 3> names = {"text_encoder", "transformer", "vae"};
  for (std::size_t i = 0; i < s.c.size(); ++i) {
    components +=
        std::format(R"({}"{}":{{"extents":{},"read_bytes":{}}})", i == 0 ? "" : ",", names.at(i),
                    s.c.at(i).weights.extents().size(), s.c.at(i).weights.read_bytes());
  }
  return std::format(
      R"({{"components":{{{}}},"plan":"{}","own_bytes":{},"work_bytes":{},"text_rows":{},"generations":{},)"
      R"("graph_measured_bytes":{},"graph_floor_bytes":{},)"
      R"("last":{{"encode":{:.6f},"first_step":{:.6f},"step_median":{:.6f},"decode":{:.6f}}}}})",
      components, s.pipeline ? llmp::base::ToHex(s.pipeline->identity()) : std::string(),
      s.own_bytes, work_bytes_, s.text, s.generations, s.graph_measured, kGraphFloorBytes, s.encode,
      s.steps.empty() ? 0.0 : s.steps.front(), later.empty() ? 0.0 : later[later.size() / 2],
      s.decode);
}

Status QwenImageRunner::Release() {
  State& s = *s_;
  std::vector<std::string> problems;
  // The recorded step first (it names the memory below), then cuBLASLt on
  // the workspace, then the runner's resources.
  s.DropGraph();
  s.account.Uncharge(std::exchange(s.graph_bytes, 0));
  s.lt.reset();
  s.resources.Release(problems);
  for (Component& c : s.c) {
    if (auto r = c.weights.Release(node_.memory()); !r) {
      problems.push_back(r.error());
    }
  }
  return support::Joined(problems);
}

}  // namespace llmp::engine
