// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A fuzzing entry for the importer's container readers (import/), the
// reference decoder of what they describe, spec synthesis/validation/
// normalization/planning (spec/, components/, import/normalize.h, planner/),
// and the v1 prepared-artifact and composition readers (artifact/,
// spec/stored.h):
//   - 0: GGUF container reader;
//   - 1: safetensors shard under a fixed config.json;
//   - 2: config.json and an MLX shard;
//   - 3: config.json and an EXL3 shard;
//   - 4: diffusers model_index.json;
//   - 5: affine tensor parameters and part bytes;
//   - 6: config.json, hf_quant_config.json and a safetensors shard;
//   - 7: fuzzed Checkpoint/Source into spec::Synthesize,
//        components::ValidateSpec, import::Normalize and planner::Plan;
//   - 8: fuzz-mutated v1 artifact and composition directories into
//        artifact::Artifact::Open, artifact::OpenComposition,
//        spec::StoredSpec and spec::SourceOf.
//
// A checkpoint a reader accepts must be consistent (every extent inside
// its file, of its format's part size, overlapping no other; at most one
// part role a tensor) or the entry aborts, and it is turned into its
// mapping source, synthesized/normalized when its architecture matches, and
// planned when valid; its quantized tensors are decoded by the reference from
// the shard's bytes, every part found by its module.
//
// Text seeds beginning with "HEX:" are hex-decoded into binary input bytes so
// seed corpora in the repository remain pure UTF-8 text (D-029, D-071). Built
// as a plain driver that runs each file named on its command line or under
// --corpus=DIR once, plus --iterations=N deterministic mutations when asked;
// to fuzz with libFuzzer, build this file with -DLLMP_LIBFUZZER and clang's
// -fsanitize=fuzzer, which brings its own main. Never run by CTest.

#include <unistd.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "artifact_writer.h"
#include "base/sha256.h"
#include "components/step.h"
#include "execution/registry.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "import/checkpoint.h"
#include "import/error.h"
#include "import/normalize.h"
#include "import/readers/diffusers.h"
#include "import/readers/exl3.h"
#include "import/readers/gguf.h"
#include "import/readers/mlx.h"
#include "import/readers/safetensors.h"
#include "ir/op.h"
#include "kernels/reference/decode.h"
#include "planner/plan.h"
#include "spec/mapping.h"
#include "spec/spec.h"
#include "spec/stored.h"

namespace {

namespace artifact = llmp::artifact;
namespace components = llmp::components;
namespace ex = llmp::execution;
namespace format = llmp::format;
namespace ir = llmp::ir;
namespace planner = llmp::planner;
namespace ref = llmp::kernels::reference;
namespace spec = llmp::spec;
using llmp::import::Checkpoint;
using llmp::import::Error;
using llmp::import::Tensor;

constexpr llmp::import::NormalizationLimits kFuzzLimits{
    .metadata_bytes = 64ULL << 20U,
    .work_units = 4ULL << 30U,
};

void Check(bool ok) {
  if (!ok) {
    std::abort();  // an accepted checkpoint, spec or artifact broke an invariant
  }
}

std::optional<std::uint8_t> HexNibble(char c) {
  if (c >= '0' && c <= '9') {
    return static_cast<std::uint8_t>(c - '0');
  }
  if (c >= 'a' && c <= 'f') {
    return static_cast<std::uint8_t>(10 + (c - 'a'));
  }
  if (c >= 'A' && c <= 'F') {
    return static_cast<std::uint8_t>(10 + (c - 'A'));
  }
  return std::nullopt;
}

std::optional<std::vector<std::uint8_t>> DecodeHexPrefix(std::span<const std::uint8_t> raw) {
  constexpr std::string_view kPrefix = "HEX:";
  if (raw.size() < kPrefix.size()) {
    return std::nullopt;
  }
  const std::string_view head(reinterpret_cast<const char*>(raw.data()), kPrefix.size());
  if (head != kPrefix) {
    return std::nullopt;
  }
  std::vector<std::uint8_t> out;
  out.reserve((raw.size() - kPrefix.size()) / 2);
  std::optional<std::uint8_t> high;
  for (std::size_t i = kPrefix.size(); i < raw.size(); ++i) {
    const auto nibble = HexNibble(static_cast<char>(raw[i]));
    if (!nibble) {
      continue;
    }
    if (!high) {
      high = *nibble;
    } else {
      out.push_back(static_cast<std::uint8_t>((*high << 4U) | *nibble));
      high.reset();
    }
  }
  return out;
}

std::string_view Take(std::span<const std::byte>& rest) {
  if (rest.size() < 2) {
    rest = {};
    return {};
  }
  const std::size_t n =
      std::min<std::size_t>(rest.size() - 2, std::to_integer<std::size_t>(rest[0]) |
                                                 (std::to_integer<std::size_t>(rest[1]) << 8U));
  const std::string_view text(reinterpret_cast<const char*>(rest.data() + 2), n);
  rest = rest.subspan(2 + n);
  return text;
}

std::uint64_t Byte(std::span<const std::byte>& rest) {
  if (rest.empty()) {
    return 0;
  }
  const auto b = std::to_integer<std::uint64_t>(rest[0]);
  rest = rest.subspan(1);
  return b;
}

const ex::Registry& FuzzRegistry() {
  static const ex::Registry kRegistry = [] {
    std::vector<ex::Implementation> impls;
    for (std::uint16_t raw = static_cast<std::uint16_t>(ir::Op::kView);
         raw <= static_cast<std::uint16_t>(ir::Op::kSelect); ++raw) {
      const auto op = static_cast<ir::Op>(raw);
      if (op == ir::Op::kFused) {
        continue;
      }
      const auto key = ir::Primitive(op);
      impls.push_back({
          .name = std::string("fuzz.") + std::string(ir::OpName(op)),
          .operation = key,
          .source = "fuzz",
          .revision = "1",
          .build = "cpu",
          .variant = "any",
          .capability = ex::Capability{.op = key},
      });
    }
    for (const ir::FusedKind fused :
         {ir::FusedKind::kRmsNormMul, ir::FusedKind::kRopeStore, ir::FusedKind::kGroupedStores}) {
      const auto key = ir::Fused(fused);
      impls.push_back({
          .name = std::string("fuzz.fused.") + std::string(ir::FusedKindName(fused)),
          .operation = key,
          .source = "fuzz",
          .revision = "1",
          .build = "cpu",
          .variant = "any",
          .capability =
              ex::Capability{
                  .op = key,
                  .batch =
                      static_cast<std::uint16_t>(fused == ir::FusedKind::kGroupedStores ? 16 : 0),
              },
      });
    }
    auto reg = ex::Registry::Create(std::move(impls));
    Check(reg.has_value());
    return std::move(*reg);
  }();
  return kRegistry;
}

void PlanSpec(const spec::ModelSpec& model, std::span<const spec::SourceTensor> tensors) {
  if (model.stages.size() != 1 || model.stages[0].layers.size() > 4 ||
      model.stages[0].stream.width > 256 || model.stages[0].context == 0) {
    return;
  }
  const std::uint32_t ctx = std::min<std::uint32_t>(model.stages[0].context, 64);
  const auto layout = components::StageLayout(model, 0, ctx, 4);
  if (!layout) {
    return;
  }
  const planner::StepShape shape{
      .phase = planner::Phase::kDecode,
      .segments = {{.rows = 1, .past = 0, .read_cells = 1, .outputs = 1}},
  };
  const std::array<const llmp::model::StateLayoutDerived*, 1> stages = {&*layout};
  const std::array<std::span<const spec::SourceTensor>, 1> artifacts = {tensors};
  const components::Bindings bindings{
      .tensors = tensors,
      .state = &*layout,
      .stages = stages,
      .artifacts = artifacts,
  };
  auto step = components::BuildStep(model, 0, shape, bindings);
  if (!step) {
    return;
  }
  Check(components::CheckStateWrites(*step, 0).has_value());
  const planner::DeviceFacts device{.device_class = "fuzz", .arch = 121};
  const auto planned =
      planner::Plan(std::move(step->graph), FuzzRegistry(), device, {}, {.shape = &shape});
  Check(planned.has_value());
  Check(components::CheckStateWrites(planned->graph, step->state, planned->tensors, 0).has_value());
}

// Every extent inside its file, its part's size under its format, no two
// overlapping.
void ExpectConsistent(const Checkpoint& c) {
  std::vector<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>> spans;  // file, offset, end
  for (const Tensor& t : c.tensors) {
    Check(t.extent.file < c.files.size());
    const std::uint64_t size = c.files[t.extent.file].bytes;
    Check(t.extent.offset <= size && t.extent.bytes <= size - t.extent.offset);
    const auto parts = format::PartsOf(t.format);
    Check(parts.has_value());
    const auto part = parts->Find(t.part);
    Check(part.has_value() && part->bytes == t.extent.bytes);
    if (t.extent.bytes != 0) {
      spans.emplace_back(t.extent.file, t.extent.offset, t.extent.offset + t.extent.bytes);
    }
  }
  std::ranges::sort(spans);
  for (std::size_t i = 1; i < spans.size(); ++i) {
    const auto& [file, offset, end] = spans[i];
    const auto& [previous_file, previous_offset, previous_end] = spans[i - 1];
    Check(file != previous_file || offset >= previous_end);
  }
}

// Each quantized tensor of a one-shard checkpoint decoded from the shard:
// its parts are the tensors of its format with the same name up to the
// last '.', one per role.
void DecodeQuantized(const Checkpoint& c, std::span<const std::byte> shard) {
  std::map<std::string_view, std::vector<const Tensor*>> modules;
  for (const Tensor& t : c.tensors) {
    if (!format::Info(t.format.encoding).quantized) {
      continue;
    }
    const std::string_view name = t.name;
    modules[name.substr(0, name.rfind('.'))].push_back(&t);
  }
  // MLX's and EXL3's modules are named by their parts' suffixes alone
  // (ModelOpt's may share a tensor scale across modules).
  const bool by_suffix = c.container == "mlx" || c.container == "exl3";
  for (const auto& [module, tensors] : modules) {
    const format::TensorFormat& f = tensors.front()->format;
    std::vector<ref::PartView> views;
    bool same = true;
    for (const Tensor* t : tensors) {
      same = same && t->format == f;
      Check(!by_suffix || same);  // a module's parts share their format
      Check(std::ranges::none_of(views, [&](const ref::PartView& v) { return v.role == t->part; }));
      views.push_back({.role = t->part, .bytes = shard.subspan(t->extent.offset, t->extent.bytes)});
    }
    if (!same) {
      continue;
    }
    const auto parts = format::PartsOf(f);
    if (!parts || views.size() != parts->size() || f.shape.ne[1] > 64 || f.shape.ne[0] > 4096) {
      continue;  // incomplete here, or too large to decode in a fuzz step
    }
    std::vector<float> out(f.shape.ne[0] * f.shape.ne[1]);
    static_cast<void>(ref::DecodeRows(f, views, 0, 0, f.shape.ne[1], out));
  }
}

void SynthesizeAndNormalize(const Checkpoint& c) {
  const auto source = llmp::import::SourceOf(c);
  const auto values = llmp::import::ValuesOf(c);
  if (const auto* mapping = spec::FindMapping(source.architecture, source.container);
      mapping != nullptr && mapping->role == spec::StageRole::kTarget) {
    const auto synthesized = spec::Synthesize(*mapping, source, values);
    if (synthesized) {
      if (components::ValidateSpec(*synthesized, source.tensors)) {
        PlanSpec(*synthesized, source.tensors);
      }
    }
  }
  const auto normalized = llmp::import::Normalize(c, kFuzzLimits);
  if (normalized) {
    Check(components::ValidateSpec(normalized->spec(), normalized->canonical().tensors,
                                   components::ValidationPurpose::kImport)
              .has_value());
    if (components::ValidateSpec(normalized->spec(), normalized->canonical().tensors)) {
      PlanSpec(normalized->spec(), normalized->canonical().tensors);
    }
  }
}

void Use(const std::expected<Checkpoint, Error>& c,
         std::optional<std::span<const std::byte>> shard = std::nullopt) {
  if (c) {
    ExpectConsistent(*c);
    if (shard) {
      DecodeQuantized(*c, *shard);
    }
    SynthesizeAndNormalize(*c);
  } else {
    static_cast<void>(c.error().ToString());
  }
}

// An affine tensor from fuzzed parameters, its parts cut from the rest of
// the input (or zeros past its end), decoded whole when valid.
void DecodeAffine(std::span<const std::byte> rest) {
  format::TensorFormat f{.encoding = format::Encoding::kAffine, .layout = format::Layout::kSplit};
  f.params.bits = static_cast<std::uint8_t>(Byte(rest) % 10);
  f.params.group = static_cast<std::uint32_t>(Byte(rest) * 8 % 264);
  f.params.scale = static_cast<format::AffineScale>(Byte(rest) % 5);
  f.params.zero_point = Byte(rest) % 2 == 1;
  const std::uint64_t k = (Byte(rest) + 1) * 8;
  const std::uint64_t n = (Byte(rest) % 8) + 1;
  const std::uint64_t e = (Byte(rest) % 3) + 1;
  f.shape = {.ne = {k, n, e, 1}, .rank = static_cast<std::uint8_t>(e > 1 ? 3 : 2)};
  const auto parts = format::PartsOf(f);
  if (!parts) {
    return;
  }
  std::vector<std::vector<std::byte>> bytes;
  std::vector<ref::PartView> views;
  for (const format::PartExtent& p : parts->parts()) {
    std::vector<std::byte>& b = bytes.emplace_back(p.bytes);
    const std::size_t take = std::min<std::size_t>(b.size(), rest.size());
    std::ranges::copy(rest.first(take), b.begin());
    rest = rest.subspan(take);
  }
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    views.push_back({.role = parts->parts()[i].role, .bytes = bytes[i]});
  }
  std::vector<float> out(k * n);
  for (std::uint64_t m = 0; m < e; ++m) {
    Check(ref::DecodeRows(f, views, m, 0, n, out).has_value());
  }
}

const Checkpoint& BaseQwen2Checkpoint() {
  static const Checkpoint kBase = [] {
    Checkpoint out{
        .container = "gguf",
        .architecture = "qwen2",
        .files = {{.name = "fuzz.gguf", .bytes = 0}},
    };
    out.metadata = {
        {"general.architecture", std::string("qwen2")},
        {"qwen2.attention.head_count", std::int64_t{2}},
        {"qwen2.attention.head_count_kv", std::int64_t{2}},
        {"qwen2.attention.layer_norm_rms_epsilon", static_cast<double>(1e-6F)},
        {"qwen2.block_count", std::int64_t{1}},
        {"qwen2.context_length", std::int64_t{64}},
        {"qwen2.embedding_length", std::int64_t{8}},
        {"qwen2.feed_forward_length", std::int64_t{16}},
        {"qwen2.rope.freq_base", 10000.0},
    };
    std::ranges::sort(out.metadata, {}, &llmp::import::MetaEntry::key);
    struct Desc {
      std::string_view name;
      std::array<std::uint64_t, 2> dims;
      std::uint8_t rank;
      format::Encoding encoding;
    };
    constexpr std::array kTensors{
        Desc{"blk.0.attn_k.bias", {8, 1}, 1, format::Encoding::kF32},
        Desc{"blk.0.attn_k.weight", {8, 8}, 2, format::Encoding::kF16},
        Desc{"blk.0.attn_norm.weight", {8, 1}, 1, format::Encoding::kF32},
        Desc{"blk.0.attn_output.weight", {8, 8}, 2, format::Encoding::kF16},
        Desc{"blk.0.attn_q.bias", {8, 1}, 1, format::Encoding::kF32},
        Desc{"blk.0.attn_q.weight", {8, 8}, 2, format::Encoding::kF16},
        Desc{"blk.0.attn_v.bias", {8, 1}, 1, format::Encoding::kF32},
        Desc{"blk.0.attn_v.weight", {8, 8}, 2, format::Encoding::kF16},
        Desc{"blk.0.ffn_down.weight", {16, 8}, 2, format::Encoding::kF16},
        Desc{"blk.0.ffn_gate.weight", {8, 16}, 2, format::Encoding::kF16},
        Desc{"blk.0.ffn_norm.weight", {8, 1}, 1, format::Encoding::kF32},
        Desc{"blk.0.ffn_up.weight", {8, 16}, 2, format::Encoding::kF16},
        Desc{"output.weight", {8, 16}, 2, format::Encoding::kF16},
        Desc{"output_norm.weight", {8, 1}, 1, format::Encoding::kF32},
        Desc{"token_embd.weight", {8, 16}, 2, format::Encoding::kF16},
    };
    for (const Desc& d : kTensors) {
      const auto shape = format::ShapeOf(std::span(d.dims).first(d.rank));
      Check(shape.has_value());
      const format::TensorFormat f{
          .encoding = d.encoding,
          .layout = format::Layout::kDense,
          .shape = *shape,
      };
      const auto parts = format::PartsOf(f);
      Check(parts.has_value() && parts->size() == 1);
      const std::uint64_t at = (out.files[0].bytes + 31U) / 32U * 32U;
      out.tensors.push_back({
          .name = std::string(d.name),
          .format = f,
          .extent = {.offset = at, .bytes = (*parts)[0].bytes},
          .dtype = format::Info(d.encoding).name,
          .dims = {d.dims.begin(), d.dims.begin() + d.rank},
      });
      out.files[0].bytes = at + (*parts)[0].bytes;
    }
    return out;
  }();
  return kBase;
}

void FuzzNormalizeAndPlan(std::span<const std::byte> rest) {
  Checkpoint c = BaseQwen2Checkpoint();
  if (!rest.empty() && rest[0] != std::byte{'='}) {
    const std::size_t steps = std::min<std::size_t>(8, rest.size() / 2);
    for (std::size_t s = 0; s < steps && !rest.empty(); ++s) {
      const std::uint64_t op = Byte(rest) % 6;
      const std::uint64_t arg = Byte(rest);
      switch (op) {
        case 0: {
          if (!c.metadata.empty()) {
            auto& entry = c.metadata[arg % c.metadata.size()];
            const std::int64_t v = static_cast<std::int64_t>(Byte(rest)) - 4;
            entry.value = v;
          }
          break;
        }
        case 1: {
          if (!c.tensors.empty()) {
            Tensor& t = c.tensors[arg % c.tensors.size()];
            const std::uint64_t d0 = (Byte(rest) % 32) + 1;
            const std::uint64_t d1 = (Byte(rest) % 32) + 1;
            const std::array<std::uint64_t, 2> dims = {d0, d1};
            const std::size_t rank = std::min<std::size_t>(2, t.format.shape.rank);
            if (const auto shape = format::ShapeOf(std::span(dims).first(rank))) {
              format::TensorFormat candidate = t.format;
              candidate.shape = *shape;
              if (const auto parts = format::PartsOf(candidate)) {
                t.format = candidate;
                t.dims.assign(dims.begin(), dims.begin() + rank);
                t.extent.bytes = (*parts)[0].bytes;
              }
            }
          }
          break;
        }
        case 2: {
          if (!c.tensors.empty()) {
            Tensor& t = c.tensors[arg % c.tensors.size()];
            constexpr std::array kEncodings = {
                format::Encoding::kF32,  format::Encoding::kF16,  format::Encoding::kBF16,
                format::Encoding::kQ8_0, format::Encoding::kQ4_K, format::Encoding::kI32,
            };
            const format::Encoding enc = kEncodings[Byte(rest) % kEncodings.size()];
            format::TensorFormat candidate = t.format;
            candidate.encoding = enc;
            candidate.layout =
                format::Info(enc).quantized ? format::Layout::kGgmlBlocks : format::Layout::kDense;
            if (const auto parts = format::PartsOf(candidate)) {
              t.format = candidate;
              t.dtype = format::Info(enc).name;
              t.extent.bytes = (*parts)[0].bytes;
            }
          }
          break;
        }
        case 3: {
          if (!c.tensors.empty()) {
            c.tensors.erase(c.tensors.begin() +
                            static_cast<std::ptrdiff_t>(arg % c.tensors.size()));
          }
          break;
        }
        case 4: {
          if (!c.tensors.empty()) {
            Tensor copy = c.tensors[arg % c.tensors.size()];
            copy.name = std::string(Take(rest));
            c.tensors.push_back(std::move(copy));
          }
          break;
        }
        default: {
          c.metadata.push_back({std::string(Take(rest)), static_cast<std::int64_t>(arg)});
          std::ranges::sort(c.metadata, {}, &llmp::import::MetaEntry::key);
          break;
        }
      }
    }
    std::uint64_t end = 0;
    for (Tensor& t : c.tensors) {
      end = (end + 31U) / 32U * 32U;
      t.extent.offset = end;
      end += t.extent.bytes;
    }
    c.files[0].bytes = end;
  }
  SynthesizeAndNormalize(c);
}

void MutateString(std::string& s, std::span<const std::byte>& rest) {
  if (s.empty() || rest.empty()) {
    return;
  }
  const std::uint64_t kind = Byte(rest) % 3;
  const std::size_t pos = static_cast<std::size_t>(Byte(rest)) % s.size();
  if (kind == 0) {
    s[pos] = static_cast<char>(Byte(rest));
  } else if (kind == 1) {
    const std::string_view chunk = Take(rest);
    s.replace(pos, std::min<std::size_t>(chunk.size(), s.size() - pos), chunk);
  } else {
    s.resize(pos);
  }
}

void WriteRawFile(const std::filesystem::path& path, std::span<const std::byte> bytes) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

void WriteRawFile(const std::filesystem::path& path, std::string_view text) {
  WriteRawFile(path, std::as_bytes(std::span(text.data(), text.size())));
}

class FuzzScratch {
 public:
  FuzzScratch() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): single-threaded fuzz harness reads environment once
    const char* env = std::getenv("LLMP_TEST_SCRATCH");
    const std::filesystem::path base =
        env != nullptr ? std::filesystem::path(env) : std::filesystem::temp_directory_path();
    root_ = base / std::format("llmp-import-fuzz-{}", ::getpid());
    std::error_code ec;
    std::filesystem::remove_all(root_, ec);
    std::filesystem::create_directories(root_, ec);
  }
  ~FuzzScratch() {
    std::error_code ec;
    std::filesystem::remove_all(root_, ec);
  }
  FuzzScratch(const FuzzScratch&) = delete;
  FuzzScratch& operator=(const FuzzScratch&) = delete;
  FuzzScratch(FuzzScratch&&) = delete;
  FuzzScratch& operator=(FuzzScratch&&) = delete;

  const std::filesystem::path& root() const { return root_; }

 private:
  std::filesystem::path root_;
};

const std::filesystem::path& ScratchRoot() {
  static const FuzzScratch kScratch;
  return kScratch.root();
}

struct BaseV1Fixture {
  llmp::test::V1Files artifact_files;
  artifact::Artifact component_artifact;
  spec::Source component_source;
  llmp::test::CompositionFiles composition_files;
};

const BaseV1Fixture& BaseV1() {
  static const BaseV1Fixture kFixture = [] {
    const auto normalized = llmp::import::Normalize(BaseQwen2Checkpoint(), kFuzzLimits);
    Check(normalized.has_value());
    const std::string spec_json = spec::ToJson(normalized->spec());

    llmp::test::V1Group table{.kind = artifact::GroupKind::kTable};
    llmp::test::V1Group layer0{.kind = artifact::GroupKind::kLayer, .layer = 0};
    llmp::test::V1Group head{.kind = artifact::GroupKind::kHead};
    for (const spec::SourceTensor& t : normalized->canonical().tensors) {
      llmp::test::V1Resource r{
          .name = t.name,
          .format = t.format,
          .part = t.part,
          .rows = (t.name == "token_embd.weight" || t.name == "output.weight"),
      };
      if (t.name == "token_embd.weight") {
        table.resources.push_back(std::move(r));
      } else if (t.name.starts_with("blk.0.")) {
        layer0.resources.push_back(std::move(r));
      } else {
        head.resources.push_back(std::move(r));
      }
    }
    const llmp::test::V1Plan plan{
        .architecture = "qwen2",
        .container = "gguf",
        .spec = spec_json,
        .groups = {std::move(table), std::move(layer0), std::move(head)},
    };
    llmp::test::V1Files art_files = llmp::test::Render(plan);
    const std::filesystem::path base_store = ScratchRoot() / "base";
    const std::filesystem::path art_dir = llmp::test::Publish(base_store, art_files);
    auto opened = artifact::Artifact::Open(art_dir);
    Check(opened.has_value());
    auto src = spec::SourceOf(*opened);
    Check(src.has_value());

    const llmp::test::CompositionPlan comp_plan{
        .architecture = "qwen2",
        .components = {{"target", std::string(opened->id()), "qwen2"}},
        .meta = {{"model_index.json", "{\"_class_name\":\"Qwen2\"}\n"}},
        .spec = spec_json,
    };
    llmp::test::CompositionFiles comp_files = llmp::test::Render(comp_plan);
    return BaseV1Fixture{
        .artifact_files = std::move(art_files),
        .component_artifact = std::move(*opened),
        .component_source = std::move(*src),
        .composition_files = std::move(comp_files),
    };
  }();
  return kFixture;
}

void CheckOpenedArtifact(const artifact::Artifact& art) {
  Check(art.format_version() == 1);
  for (const auto& group : art.groups()) {
    Check(group.stored > artifact::Bytes{0} && group.used <= group.stored);
  }
  const auto source = spec::SourceOf(art);
  const auto stored = spec::StoredSpec(art);
  if (source && stored) {
    if (components::ValidateSpec(*stored, source->tensors)) {
      PlanSpec(*stored, source->tensors);
    }
  }
}

void FuzzArtifact(std::span<const std::byte> rest) {
  const auto& base = BaseV1();
  const std::filesystem::path store = ScratchRoot() / "art";
  std::error_code ec;
  std::filesystem::remove_all(store, ec);
  std::filesystem::create_directories(store, ec);

  std::filesystem::path dir;
  if (rest.empty() || rest[0] == std::byte{'='}) {
    dir = llmp::test::Publish(store, base.artifact_files);
    const auto opened = artifact::Artifact::Open(dir);
    Check(opened.has_value());
    CheckOpenedArtifact(*opened);
    std::filesystem::remove_all(store, ec);
    return;
  }

  const std::uint64_t mode = Byte(rest) % 3;
  if (mode == 0) {
    llmp::test::V1Files files = base.artifact_files;
    files.keep_files = (Byte(rest) % 4 == 0);
    if (files.spec && (Byte(rest) % 2 == 0)) {
      MutateString(*files.spec, rest);
    }
    if (!files.shards.empty() && (Byte(rest) % 2 == 0)) {
      MutateString(files.shards[0], rest);
    }
    std::vector<std::byte> index_mut(rest.begin(), rest.begin() + (rest.size() / 2));
    std::vector<std::byte> manifest_mut(rest.begin() + (rest.size() / 2), rest.end());
    const llmp::test::PublishHooks hooks{
        .index =
            [&](std::string& text) {
              std::span<const std::byte> span(index_mut);
              if (!span.empty()) {
                MutateString(text, span);
              }
            },
        .manifest =
            [&](std::string& text) {
              std::span<const std::byte> span(manifest_mut);
              if (!span.empty()) {
                MutateString(text, span);
              }
            },
    };
    dir = llmp::test::Publish(store, files, hooks);
  } else if (mode == 1) {
    dir = llmp::test::Publish(store, base.artifact_files);
    constexpr std::array<std::string_view, 4> kTargets = {
        "manifest.json",
        "index.json",
        "spec.json",
        "data/00000.safetensors",
    };
    const std::filesystem::path target = dir / kTargets[Byte(rest) % kTargets.size()];
    std::ifstream in(target, std::ios::binary);
    std::string bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    MutateString(bytes, rest);
    WriteRawFile(target, bytes);
  } else {
    const std::string_view manifest = Take(rest);
    const std::string_view index = Take(rest);
    const std::string_view spec_text = Take(rest);
    const std::string id = llmp::base::ToHex(llmp::base::Sha256{}.Update(manifest).Finish());
    dir = store / id;
    WriteRawFile(dir / "manifest.json", manifest);
    WriteRawFile(dir / "index.json", index);
    if (!spec_text.empty()) {
      WriteRawFile(dir / "spec.json", spec_text);
    }
    WriteRawFile(dir / "data/00000.safetensors", rest);
  }

  const auto opened = artifact::Artifact::Open(dir);
  if (opened) {
    CheckOpenedArtifact(*opened);
  } else {
    static_cast<void>(opened.error().ToString().size());
  }
  std::filesystem::remove_all(store, ec);
}

void CheckOpenedComposition(const artifact::Composition& comp, const BaseV1Fixture& base) {
  Check(comp.format_version() == 1 || comp.format_version() == 2);
  Check(!comp.components().empty());
  if (comp.components().size() == 1) {
    const std::array<const artifact::Artifact*, 1> comps = {&base.component_artifact};
    const auto stored = spec::StoredSpec(comp, comps);
    if (stored) {
      if (components::ValidateSpec(*stored, base.component_source.tensors)) {
        PlanSpec(*stored, base.component_source.tensors);
      }
    }
  }
}

void FuzzComposition(std::span<const std::byte> rest) {
  const auto& base = BaseV1();
  const std::filesystem::path store = ScratchRoot() / "comp";
  std::error_code ec;
  std::filesystem::remove_all(store, ec);
  std::filesystem::create_directories(store, ec);

  std::filesystem::path dir;
  if (rest.empty() || rest[0] == std::byte{'='}) {
    dir = llmp::test::Publish(store, base.composition_files);
    const auto opened = artifact::OpenComposition(dir);
    Check(opened.has_value());
    CheckOpenedComposition(*opened, base);
    std::filesystem::remove_all(store, ec);
    return;
  }

  const std::uint64_t mode = Byte(rest) % 2;
  if (mode == 0) {
    llmp::test::CompositionFiles files = base.composition_files;
    files.keep_files = (Byte(rest) % 4 == 0);
    if (files.spec && (Byte(rest) % 2 == 0)) {
      MutateString(*files.spec, rest);
    }
    std::vector<std::byte> manifest_mut(rest.begin(), rest.end());
    const llmp::test::PublishHooks hooks{
        .manifest =
            [&](std::string& text) {
              std::span<const std::byte> span(manifest_mut);
              if (!span.empty()) {
                MutateString(text, span);
              }
            },
    };
    dir = llmp::test::Publish(store, files, hooks);
  } else {
    const std::string_view manifest = Take(rest);
    const std::string id = llmp::base::ToHex(llmp::base::Sha256{}.Update(manifest).Finish());
    dir = store / id;
    WriteRawFile(dir / "manifest.json", manifest);
    if (!rest.empty()) {
      WriteRawFile(dir / "spec.json", rest);
    }
  }

  const auto opened = artifact::OpenComposition(dir);
  if (opened) {
    CheckOpenedComposition(*opened, base);
  } else {
    static_cast<void>(opened.error().ToString().size());
  }
  std::filesystem::remove_all(store, ec);
}

void FuzzV1(std::span<const std::byte> rest) {
  if (Byte(rest) % 2 == 0) {
    FuzzArtifact(rest);
  } else {
    FuzzComposition(rest);
  }
}

void RunOneInput(std::span<const std::uint8_t> input) {
  if (input.empty()) {
    return;
  }
  std::span<const std::byte> rest = std::as_bytes(input.subspan(1));
  switch (input[0] % 9) {
    case 0:
      Use(llmp::import::ParseGguf(rest));
      break;
    case 1:
      Use(llmp::import::ParseSafetensors(rest, R"({"model_type":"fuzz"})"), rest);
      break;
    case 2: {
      const std::string_view config = Take(rest);
      Use(llmp::import::ParseMlx(rest, config), rest);
      break;
    }
    case 3: {
      const std::string_view config = Take(rest);
      Use(llmp::import::ParseExl3(rest, config), rest);
      break;
    }
    case 4: {
      const auto index = llmp::import::ParseModelIndex(
          std::string_view(reinterpret_cast<const char*>(rest.data()), rest.size()));
      static_cast<void>(index ? index->roles.size() : index.error().ToString().size());
      break;
    }
    case 5:
      DecodeAffine(rest);
      break;
    case 6: {
      const std::string_view config = Take(rest);
      const std::string_view hf = Take(rest);
      Use(llmp::import::ParseSafetensors(rest, config, hf), rest);
      break;
    }
    case 7:
      FuzzNormalizeAndPlan(rest);
      break;
    default:
      FuzzV1(rest);
      break;
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::span<const std::uint8_t> raw(data, size);
  if (const auto decoded = DecodeHexPrefix(raw)) {
    RunOneInput(*decoded);
  } else {
    RunOneInput(raw);
  }
  return 0;
}

#ifndef LLMP_LIBFUZZER
namespace {

std::vector<std::uint8_t> ReadSeedFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  const std::vector<std::uint8_t> raw{std::istreambuf_iterator<char>(in),
                                      std::istreambuf_iterator<char>()};
  if (const auto decoded = DecodeHexPrefix(raw)) {
    return *decoded;
  }
  return raw;
}

std::vector<std::vector<std::uint8_t>> DefaultSeeds() {
  return {
      {7, static_cast<std::uint8_t>('=')},
      {8, 0, static_cast<std::uint8_t>('=')},
      {8, 1, static_cast<std::uint8_t>('=')},
  };
}

}  // namespace

int main(int argc, char** argv) {
  std::uint64_t iterations = 0;
  std::vector<std::filesystem::path> files;

  for (int i = 1; i < argc; ++i) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    const std::string_view arg(argv[i]);
    if (arg.starts_with("--iterations=")) {
      const std::string_view val = arg.substr(std::string_view("--iterations=").size());
      const auto [ptr, ec] = std::from_chars(val.data(), val.data() + val.size(), iterations);
      if (ec != std::errc{} || ptr != val.data() + val.size()) {
        std::fprintf(stderr, "llmp_import_fuzz: invalid --iterations value: %.*s\n",
                     static_cast<int>(val.size()), val.data());
        return 1;
      }
    } else if (arg.starts_with("--corpus=")) {
      const std::filesystem::path dir(arg.substr(std::string_view("--corpus=").size()));
      std::error_code ec;
      std::vector<std::filesystem::path> entries;
      for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) {
          break;
        }
        if (entry.is_regular_file(ec) && entry.path().extension() != ".license") {
          entries.push_back(entry.path());
        }
      }
      if (ec) {
        std::fprintf(stderr, "llmp_import_fuzz: cannot read corpus directory %s: %s\n", dir.c_str(),
                     ec.message().c_str());
        return 1;
      }
      std::ranges::sort(entries);
      files.insert(files.end(), entries.begin(), entries.end());
    } else if (arg.starts_with("--")) {
      std::fprintf(stderr, "llmp_import_fuzz: unknown option %.*s\n", static_cast<int>(arg.size()),
                   arg.data());
      return 1;
    } else {
      files.emplace_back(arg);
    }
  }

  std::vector<std::vector<std::uint8_t>> seeds;
  seeds.reserve(files.size());
  for (const std::filesystem::path& path : files) {
    std::vector<std::uint8_t> bytes = ReadSeedFile(path);
    RunOneInput(bytes);
    if (!bytes.empty()) {
      seeds.push_back(std::move(bytes));
    }
  }

  if (iterations > 0) {
    if (seeds.empty()) {
      seeds = DefaultSeeds();
      for (const auto& seed : seeds) {
        RunOneInput(seed);
      }
    }
    // NOLINTNEXTLINE(bugprone-random-generator-seed): deterministic fuzz replay
    std::mt19937_64 rng(20261021);
    for (std::uint64_t iter = 0; iter < iterations; ++iter) {
      std::vector<std::uint8_t> mutated = seeds[iter % seeds.size()];
      const std::size_t changes = 1 + static_cast<std::size_t>(rng() % 6);
      for (std::size_t c = 0; c < changes && !mutated.empty(); ++c) {
        const std::uint64_t kind = rng() % 4;
        // Keep the branch selector byte stable most of the time so each seed
        // exercises its target parser/normalizer/artifact path deeply.
        const std::size_t min_idx = (mutated.size() > 1 && (rng() % 4 != 0)) ? 1 : 0;
        const std::size_t idx =
            min_idx + static_cast<std::size_t>(rng() % (mutated.size() - min_idx));
        if (kind == 0) {
          mutated[idx] ^= static_cast<std::uint8_t>(1U << (rng() % 8));
        } else if (kind == 1) {
          mutated[idx] = static_cast<std::uint8_t>(rng() & 0xFFU);
        } else if (kind == 2 && mutated.size() < 65536) {
          mutated.insert(mutated.begin() + static_cast<std::ptrdiff_t>(idx),
                         static_cast<std::uint8_t>(rng() & 0xFFU));
        } else if (mutated.size() > 2) {
          mutated.erase(mutated.begin() + static_cast<std::ptrdiff_t>(idx));
        }
      }
      RunOneInput(mutated);
    }
  }
  return 0;
}
#endif
