// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/component_admission.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "artifact_writer.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "components/state_encoding.h"
#include "components/step.h"
#include "dsv4_gguf_normalization_fixture.h"
#include "expected_error.h"
#include "families/gemma.h"
#include "image_family.h"
#include "import/normalize.h"
#include "import_target_fixture.h"
#include "ir/builder.h"
#include "kernels/common/lower.h"
#include "planner/pattern.h"
#include "qwen2_exl3_normalization_fixture.h"
#include "qwen2_gguf_normalization_fixture.h"
#include "qwen38_fixture.h"
#include "qwen38_gguf_normalization_fixture.h"
#include "runtime/calibration.h"
#include "runtime/component_admission_internal.h"
#include "runtime/numerical_evidence.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec/mappings/qwen38.h"
#include "spec/stored.h"
#include "spec_fixtures.h"
#include "tokenizer_fixtures.h"

namespace {
namespace rt = llmp::runtime;
namespace sp = llmp::spec;
namespace fm = llmp::format;
namespace fam = llmp::test_support::families;

std::array<llmp::artifact::Source, 1> Origins(
    std::string_view name = "gemma-2-2b-it-Q8_0.gguf", std::uint64_t bytes = 2784495456ULL,
    std::string_view hex = "2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe") {
  llmp::base::Sha256Digest digest{};
  for (std::size_t i = 0; i < digest.size(); ++i) {
    unsigned byte = 0;
    const auto read = std::from_chars(hex.data() + 2 * i, hex.data() + 2 * i + 2, byte, 16);
    EXPECT_EQ(read.ec, std::errc{});
    digest[i] = static_cast<std::uint8_t>(byte);
  }
  return {{{std::string(name), llmp::base::Bytes(bytes), digest}}};
}

// Descriptor fixtures explicitly enroll their captured converted/native conventions.
// The production path obtains these records from the authenticated manifest.
void FixturePreprocessing(const sp::ModelSpec& model, sp::Source& source) {
  source.preprocessing.clear();
  for (const auto& fold : model.stages.front().folds) {
    auto names = llmp::components::FoldTensors(model, 0, fold);
    ASSERT_TRUE(names) << names.error();
    source.preprocessing.push_back({.name = fold,
                                    .tensors = std::move(*names),
                                    .source = !std::ranges::contains(source.folds, fold)});
  }
}

class ComponentAdmissionTest : public testing::Test {
 protected:
  sp::Source source = llmp::test_support::Gemma2Source();
  sp::ModelSpec model;
  std::array<llmp::artifact::Source, 1> origins = Origins();
  void SetUp() override {
    const auto values = fam::Gemma2Values(fam::Gemma2_2B());
    ASSERT_TRUE(values);
    auto made = sp::Synthesize(sp::mappings::Gemma2Mapping(), source, *values);
    ASSERT_TRUE(made) << (made ? "" : made.error());
    model = std::move(*made);
    FixturePreprocessing(model, source);
  }
};

// These descriptor-only approvals pin small independent inventories, not
// production checkpoints. The literal digests cover the dense F16 tensor
// and every part of the expert table and affine matrix; they are never
// derived by admission.
struct InventoryCase {
  std::string name;
  std::string_view container;
  fm::TensorFormat format;
  std::string_view digest;
};

InventoryCase ModelOptInventory(std::string name, fm::Encoding encoding, fm::Layout layout,
                                std::uint8_t stacked, std::uint64_t experts,
                                std::string_view digest) {
  return {.name = std::move(name),
          .container = "safetensors",
          .format = {.encoding = encoding,
                     .params = {.tensor_scale = encoding == fm::Encoding::kNvfp4 &&
                                                layout == fm::Layout::kSplit},
                     .layout = layout,
                     .stacked = stacked,
                     .shape = {.ne = {128, 256, experts, 1}, .rank = 3}},
          .digest = digest};
}

InventoryCase MlxInventory(std::uint8_t bits, std::uint32_t group, fm::AffineScale scale,
                           std::string_view digest) {
  const auto scale_name = fm::Info(fm::AffineScaleEncoding(scale)).name;
  return {.name = "Affine" + std::to_string(bits) + "Group" + std::to_string(group) +
                  std::string(scale_name),
          .container = "mlx",
          .format = {.encoding = fm::Encoding::kAffine,
                     .params = {.bits = bits, .group = group, .scale = scale},
                     .layout = fm::Layout::kSplit,
                     .shape = {.ne = {128, 64, 8, 1}, .rank = 3}},
          .digest = digest};
}

class ComponentInventoryTest : public testing::TestWithParam<InventoryCase> {
 protected:
  sp::Source source;
  rt::Approval approval;
  void SetUp() override {
    const auto& c = GetParam();
    source.architecture = "inventory-test";
    source.container = c.container == "mlx" ? "mlx" : "safetensors";
    source.tensors = {
        {.name = "dense.weight",
         .format = {.encoding = fm::Encoding::kF16,
                    .layout = fm::Layout::kDense,
                    .shape = {.ne = {128, 64, 1, 1}, .rank = 2}}},
        {.name = "weight", .format = c.format, .part = fm::PartRole::kCodes, .experts = true},
        {.name = "weight",
         .format = c.format,
         .part = fm::PartRole::kBlockScales,
         .experts = true}};
    if (c.container == "mlx" || c.format.params.tensor_scale)
      source.tensors.push_back(
          {.name = "weight",
           .format = c.format,
           .part = c.container == "mlx" ? fm::PartRole::kBiases : fm::PartRole::kTensorScale,
           .experts = true});
    if (c.container == "mlx") {
      auto matrix = c.format;
      matrix.shape.rank = 2;
      matrix.shape.ne[2] = 1;
      for (const auto part :
           {fm::PartRole::kCodes, fm::PartRole::kBlockScales, fm::PartRole::kBiases})
        source.tensors.push_back({.name = "linear.weight", .format = matrix, .part = part});
    }
    approval = {.name = c.name,
                .architecture = source.architecture,
                .container = source.container,
                .tensors = source.tensors.size(),
                .arrays = 1,
                .inventory = c.digest};
  }
};

TEST_P(ComponentInventoryTest, ExplicitApprovalAcceptsAlternateExpertCountsAndAffineParameters) {
  const auto checked = rt::internal::CheckInventory(source, approval);
  ASSERT_TRUE(checked) << (checked ? "" : checked.error());
  // The complete digest is independent of descriptor ordering.
  std::ranges::reverse(source.tensors);
  EXPECT_TRUE(rt::internal::CheckInventory(source, approval));
}

TEST_P(ComponentInventoryTest, PinnedInventoryRejectsDigestShapeEncodingAndParameterMutations) {
  const auto original = source;
  const auto differs = [](const std::string& error) {
    return error.contains("native tensor inventory differs from the approved checkpoint");
  };
  auto wrong_pin = approval;
  wrong_pin.inventory = "0000000000000000000000000000000000000000000000000000000000000000";
  EXPECT_EQ(llmp::test_support::Failed(rt::internal::CheckInventory(source, wrong_pin), differs),
            true);
  // These mutations remain structurally admissible, so rejection must be
  // by the immutable digest rather than a checkpoint-specific shape rule.
  for (auto& t : source.tensors)
    if (t.experts) ++t.format.shape.ne[2];
  EXPECT_EQ(llmp::test_support::Failed(rt::internal::CheckInventory(source, approval), differs),
            true);
  source = original;
  source.tensors.front().format.encoding = fm::Encoding::kBF16;
  EXPECT_EQ(llmp::test_support::Failed(rt::internal::CheckInventory(source, approval), differs),
            true);
  if (approval.container == "mlx") {
    for (int mutation = 0; mutation != 3; ++mutation) {
      SCOPED_TRACE(mutation);
      source = original;
      for (auto& t : source.tensors) {
        if (!t.experts) continue;
        auto& params = t.format.params;
        if (mutation == 0) params.bits = params.bits == 4 ? 8 : 4;
        if (mutation == 1) params.group = params.group == 32 ? 64 : 32;
        if (mutation == 2)
          params.scale = params.scale == fm::AffineScale::kF16 ? fm::AffineScale::kBF16
                                                               : fm::AffineScale::kF16;
      }
      EXPECT_EQ(llmp::test_support::Failed(rt::internal::CheckInventory(source, approval), differs),
                true);
    }
  }
}

TEST_P(ComponentInventoryTest, GenericStructureRejectsInvalidExpertGeometryAndParts) {
  const auto original = source;
  constexpr std::string_view kStructureError =
      "component artifact admission: native tensor layout, part, parameters or expert geometry "
      "differs";
  for (int mutation = 0; mutation != 5; ++mutation) {
    SCOPED_TRACE(mutation);
    source = original;
    auto& t = source.tensors[1];
    if (mutation == 0) t.format.shape.ne[2] = 0;
    if (mutation == 1) {
      // A valid matrix format cannot describe an expert array.
      t.format.shape.rank = 2;
      t.format.shape.ne[2] = 1;
    }
    if (mutation == 2) t.part = fm::PartRole::kData;
    if (mutation == 3) t.format.stacked = 0;
    if (mutation == 4) {
      if (approval.container == "mlx")
        t.format.params.group = 0;
      else
        t.format.shape.ne[0] = 127;  // not whole NVFP4/MXFP8 blocks
    }
    EXPECT_EQ(llmp::test_support::Failed(rt::internal::CheckInventory(source, approval)),
              kStructureError);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ExplicitApprovals, ComponentInventoryTest,
    testing::Values(
        ModelOptInventory("Nvfp4Split8Experts", fm::Encoding::kNvfp4, fm::Layout::kSplit, 1, 8,
                          "9be42456b6011e75207978d20cb7065d2b4414499819f8b8eeee227275577fd4"),
        ModelOptInventory("Mxfp8Split64Experts", fm::Encoding::kMxfp8, fm::Layout::kSplit, 1, 64,
                          "066262bcff9e44b7db430a54c32a00be1a29e2453d10dab53c3d4380aa191af3"),
        ModelOptInventory("Nvfp4ScaleAtoms7Experts", fm::Encoding::kNvfp4, fm::Layout::kScaleAtoms,
                          1, 7, "b6a40f817de8c0059b2d3418bed55cd2b999c33192c439b0a99183252b17f0dc"),
        ModelOptInventory("Nvfp4StackedScaleAtoms16Experts", fm::Encoding::kNvfp4,
                          fm::Layout::kScaleAtoms, 2, 16,
                          "9d0c0a20b6ccf45a429bc0d30f0702bfd7c4aea9fefc540767b59164cc92802a"),
        MlxInventory(4, 32, fm::AffineScale::kF16,
                     "840ea77db0852ee13534ba17e47f7ac76d95d9b82d84cc9669f72b02b1f0a88b"),
        MlxInventory(4, 32, fm::AffineScale::kBF16,
                     "ae3c027084b41c84ef963cba255a0915dd0fc4102263252a4b15e473938f6d2e"),
        MlxInventory(4, 64, fm::AffineScale::kF16,
                     "7993d555be8ff0fcef2439251441c8f0b9174d956ba9d981152b1d6e967e5c10"),
        MlxInventory(4, 64, fm::AffineScale::kBF16,
                     "cbe6a966652357de6e92f285bb476a18cb529f9668db3b73f09a055818c68da6"),
        MlxInventory(4, 128, fm::AffineScale::kF16,
                     "2e72cc64ca3a96eb63e1667b9d578a70e893454628175ab5bf3bcd5476b29579"),
        MlxInventory(4, 128, fm::AffineScale::kBF16,
                     "c690b1b50094d34adb4de744bbe545f4774958e1ca319003c6b75a9b184187fa"),
        MlxInventory(8, 32, fm::AffineScale::kF16,
                     "424ce8883b291925e7353039ac0460aea7ed1fd10de66a52d3a029660ffef16b"),
        MlxInventory(8, 32, fm::AffineScale::kBF16,
                     "b4cb467d4db0d032090a6d94f18124532ce8df39fd35dc19a963e6a5e2baf1f9"),
        MlxInventory(8, 64, fm::AffineScale::kF16,
                     "ca8e9f80bc32001eddabe8c52bd35461035a8120cfadae1f4d04bb797a9646bb"),
        MlxInventory(8, 64, fm::AffineScale::kBF16,
                     "bdd85d682602bef01d73c7d546327ee32beb3d1ce63660a33e11b274bb12055f"),
        MlxInventory(8, 128, fm::AffineScale::kF16,
                     "48b663ba8aec4b7735c748380365a062d86f9975d1f0f6b26b4f3f61380840b6"),
        MlxInventory(8, 128, fm::AffineScale::kBF16,
                     "1bd4fb54172b709a871634b6f3dfab4f9a31e39645aa32c61b11704a99462ac4"),
        // A format-valid configuration beyond the shared GPU decoder's
        // current capability list: inventory qualification is separate.
        MlxInventory(3, 16, fm::AffineScale::kF32,
                     "fc4ddbe599ecf50d4a373f37455318cf5dd99fd30d30639d71a2d3817f5ea000")),
    [](const testing::TestParamInfo<InventoryCase>& info) { return info.param.name; });

class NumericalOwner final {
 public:
  bool Charge(std::uint64_t bytes) const {
    ++charges;
    if (bytes > available) return false;
    available -= bytes;
    used += bytes;
    return true;
  }
  void Release(std::uint64_t bytes) const {
    used -= bytes;
    available += bytes;
    ++releases;
  }
  mutable std::uint64_t available = UINT64_MAX, used = 0, charges = 0, releases = 0;
};

std::shared_ptr<const llmp::kernels::ConstructionFunder> NumericalFunder(
    const std::shared_ptr<NumericalOwner>& owner) {
  return std::make_shared<const llmp::kernels::ConstructionFunder>(
      [owner](std::uint64_t bytes) { return owner->Charge(bytes); },
      [owner](std::uint64_t bytes) { owner->Release(bytes); });
}

llmp::planner::DeviceFacts NumericalDevice() {
  return {
      .device_class = "sm_121", .arch = 121, .device_model = "NVIDIA GB10", .multiprocessors = 48};
}

// Grants come from component contracts and evidence alone: no approval
// argument exists, and the source's container changes nothing.
TEST_F(ComponentAdmissionTest, NumericalContractsIgnoreContainerOrigin) {
  auto owner = std::make_shared<NumericalOwner>();
  const auto funder = NumericalFunder(owner);
  const auto semantic = model.Digest();
  const auto bind = [&] {
    return rt::AdmittedNumerics(model, source, origins, semantic, {.context = 8192, .slots = 4}, 0,
                                NumericalDevice(), funder);
  };
  auto pinned_facts = bind();
  ASSERT_TRUE(pinned_facts && *pinned_facts);
  source.container = "safetensors";
  auto other_container = bind();
  ASSERT_TRUE(other_container && *other_container);
  EXPECT_EQ((*pinned_facts)->digest(), (*other_container)->digest());
  for (const auto& entry : (*pinned_facts)->entries()) {
    EXPECT_EQ(entry.change, llmp::planner::NumericalChange::kQ8VectorReduction);
    EXPECT_TRUE(entry.exact_dimensions);
    EXPECT_NE(entry.weight_role, UINT16_MAX);
    EXPECT_EQ(entry.semantic, semantic);
  }
}

// Two evidence records grant one product at one encoding and row only where
// the review found (F8, S-F5): the Qwen GGUF vector record and the final
// DeepSeek record share Q6_K and Q8_0 block products at rows 1-4. Both
// grants are kept, in a total order that puts the incumbent recipe's first
// (ProjectNumerics), so a plan attributes the same evidence and status in
// every build (ExpectDefinedAttribution). Any other overlap is new and
// refused here.
// Grants for one site that differ only in their evidence keep a defined
// order: an incumbent recipe's first, then by evidence.
void ExpectDefinedAttribution(std::span<const llmp::planner::NumericalQualification> entries) {
  const auto site = [](const llmp::planner::NumericalQualification& e) {
    return std::tuple(e.change, e.stage, e.component, e.anchor, e.min_rows, e.max_rows, e.max_k,
                      e.max_n, e.weight_role);
  };
  for (std::size_t i = 1; i < entries.size(); ++i) {
    const auto& a = entries[i - 1];
    const auto& b = entries[i];
    if (site(a) != site(b)) continue;
    EXPECT_TRUE(a.incumbent_recipe >= b.incumbent_recipe &&
                (a.incumbent_recipe != b.incumbent_recipe || a.evidence <= b.evidence))
        << "grants for one site in an undefined order at " << i;
  }
}

TEST(NumericalEvidenceTableTest, RecordsOverlapOnlyAtTheKnownBlockProductRows) {
  const auto table = rt::ComponentNumericalEvidenceTable();
  struct Overlap {
    std::size_t first, second;
    fm::Encoding encoding;
    std::uint64_t row;
    bool operator==(const Overlap&) const = default;
  };
  std::vector<Overlap> overlaps;
  for (std::size_t a = 0; a < table.size(); ++a)
    for (std::size_t b = a + 1; b < table.size(); ++b) {
      const auto& x = table[a];
      const auto& y = table[b];
      if (x.pipeline != y.pipeline || x.change != y.change || x.semantic != y.semantic ||
          x.device_class != y.device_class || x.device_model != y.device_model ||
          x.arch != y.arch || x.multiprocessors != y.multiprocessors)
        continue;
      for (const auto encoding : x.encodings) {
        if (!std::ranges::contains(y.encodings, encoding)) continue;
        // A record without listed rows covers every row of its owner bound.
        if (x.rows.empty() || y.rows.empty()) {
          overlaps.push_back({a, b, encoding, 0});
          continue;
        }
        for (const auto row : x.rows)
          if (std::ranges::contains(y.rows, row)) overlaps.push_back({a, b, encoding, row});
      }
    }
  ASSERT_EQ(overlaps.size(), 8U);
  const auto& first = table[overlaps.front().first];
  const auto& second = table[overlaps.front().second];
  EXPECT_EQ(first.semantic, "product.block32_q8");
  EXPECT_EQ(first.report, "docs/experiments/qwen38-gguf/README.md");
  EXPECT_FALSE(first.incumbent_recipe);
  EXPECT_EQ(second.report, "docs/experiments/m36-final/README.md");
  EXPECT_TRUE(second.incumbent_recipe);
  for (const auto& overlap : overlaps) {
    EXPECT_EQ(overlap.first, overlaps.front().first);
    EXPECT_EQ(overlap.second, overlaps.front().second);
    EXPECT_TRUE(overlap.encoding == fm::Encoding::kQ6_K || overlap.encoding == fm::Encoding::kQ8_0);
    EXPECT_GE(overlap.row, 1U);
    EXPECT_LE(overlap.row, 4U);
  }
}

TEST_F(ComponentAdmissionTest, MissingDeviceOrEncodingEvidenceKeepsExactPathWithoutPayment) {
  auto owner = std::make_shared<NumericalOwner>();
  const auto funder = NumericalFunder(owner);
  const auto semantic = model.Digest();
  for (const auto mutation : {0, 1, 2, 3}) {
    auto device = NumericalDevice();
    if (mutation == 0) {
      device.device_class = "sm_86";
      device.arch = 86;
    }
    if (mutation == 1) device.device_model = "another device";
    if (mutation == 2) device.multiprocessors = 1;
    if (mutation == 3) device = {};
    auto facts = rt::AdmittedNumerics(model, source, origins, semantic,
                                      {.context = 8192, .slots = 4}, 0, device, funder);
    ASSERT_TRUE(facts);
    EXPECT_FALSE(*facts);
  }
  EXPECT_EQ(owner->charges, 0U);
  for (auto& tensor : source.tensors)
    if (tensor.format.encoding == fm::Encoding::kQ8_0) tensor.format.encoding = fm::Encoding::kQ2_0;
  auto uncovered =
      rt::AdmittedNumerics(model, source, origins, semantic, {.context = 8192, .slots = 4}, 0,
                           NumericalDevice(), funder);
  ASSERT_TRUE(uncovered);
  EXPECT_FALSE(*uncovered);
  EXPECT_EQ(owner->charges, 0U);
}

TEST_F(ComponentAdmissionTest, NumericalProjectionAuthenticatesAndRetainsPaidStorage) {
  auto owner = std::make_shared<NumericalOwner>();
  auto funder = NumericalFunder(owner);
  const auto semantic = model.Digest();
  const auto bind = [&](std::shared_ptr<const llmp::kernels::ConstructionFunder> payment) {
    return rt::AdmittedNumerics(model, source, origins, semantic, {.context = 8192, .slots = 4}, 0,
                                NumericalDevice(), payment);
  };
  auto wrong = semantic;
  wrong[0] ^= 1;
  EXPECT_FALSE(rt::AdmittedNumerics(model, source, origins, wrong, {.context = 8192, .slots = 4}, 0,
                                    NumericalDevice(), funder));
  const auto original = source.tensors.front();
  source.tensors.front().format.encoding = static_cast<fm::Encoding>(UINT16_MAX);
  EXPECT_FALSE(bind(funder));
  source.tensors.front() = original;
  EXPECT_EQ(owner->charges, 0U);
  EXPECT_FALSE(bind({}));
  owner->available = 0;
  EXPECT_FALSE(bind(funder));
  EXPECT_EQ(owner->used, 0U);
  owner->available = UINT64_MAX;
  auto facts = bind(funder);
  ASSERT_TRUE(facts && *facts);
  EXPECT_GT(owner->used, (*facts)->host_bytes());
  std::weak_ptr<const llmp::kernels::ConstructionFunder> payer = funder;
  funder.reset();
  EXPECT_FALSE(payer.expired());
  facts->reset();
  EXPECT_TRUE(payer.expired());
  EXPECT_EQ(owner->used, 0U);
  EXPECT_EQ(owner->releases, 1U);
}

// The alias-wide Q8_0 KV encoding is applied after admission (component
// build) and touches no numerical site: the encoded spec projects the same
// grants under the admitted identity, which a Q8_0 alias was once refused
// ("numerical semantic identity differs"). Any other change is still refused.
TEST_F(ComponentAdmissionTest, AliasQ8KvEncodingKeepsTheAdmittedNumericalIdentity) {
  auto owner = std::make_shared<NumericalOwner>();
  const auto funder = NumericalFunder(owner);
  const auto semantic = model.Digest();
  const auto bind = [&](const sp::ModelSpec& projected) {
    return rt::AdmittedNumerics(projected, source, origins, semantic,
                                {.context = 8192, .slots = 4}, 0, NumericalDevice(), funder);
  };
  auto encoded = model;
  const auto bytes = llmp::components::KvCacheEncodingBytes(encoded, fm::Encoding::kQ8_0);
  ASSERT_TRUE(bytes) << bytes.error();
  ASSERT_TRUE(llmp::components::ApplyKvCacheEncoding(encoded, fm::Encoding::kQ8_0, *bytes));
  ASSERT_NE(encoded.Digest(), semantic);
  EXPECT_EQ(llmp::components::WithoutKvCacheEncoding(encoded), model);
  auto plain = bind(model);
  auto q8 = bind(encoded);
  ASSERT_TRUE(plain && *plain) << (plain ? "" : plain.error());
  ASSERT_TRUE(q8 && *q8) << (q8 ? "" : q8.error());
  EXPECT_EQ((*plain)->digest(), (*q8)->digest());
  for (const auto& entry : (*q8)->entries()) EXPECT_EQ(entry.semantic, semantic);
  // Another change to the encoded spec is not the admitted spec.
  auto other = encoded;
  other.stages.front().context /= 2;
  auto refused = bind(other);
  ASSERT_FALSE(refused);
  EXPECT_NE(refused.error().find("numerical semantic identity differs"), std::string::npos);
  // An explicit F16 setting is not the admitted spec either.
  for (auto& layer : encoded.stages.front().layers)
    for (auto& component : layer.components)
      for (auto& p : component.params)
        if (p.name == "kv_encoding") p.value = std::string("f16");
  EXPECT_FALSE(bind(encoded));
}

class NumericalBackendOwner final : public llmp::kernels::BackendOwner {
 public:
  NumericalBackendOwner(llmp::kernels::StepBackend table, std::shared_ptr<NumericalOwner> payer)
      : table_(std::move(table)), payer_(std::move(payer)) {}
  const llmp::kernels::StepBackend& backend() const override { return table_; }
  bool Charge(std::uint64_t bytes) const override { return payer_->Charge(bytes); }
  void Release(std::uint64_t bytes) const override { payer_->Release(bytes); }
  std::expected<llmp::execution::BackendDeclaration, std::string> Ready() const override {
    return std::unexpected("admission-only owner has no executable declaration");
  }

 private:
  const llmp::kernels::StepBackend table_;
  const std::shared_ptr<NumericalOwner> payer_;
};

TEST(ComponentAdmissionModelOptTest, ServingNumericsUseOwnedFactsAndPayerInsteadOfTokenFields) {
  const auto fixture = llmp::test_support::qwen38::NativeModelOpt();
  const auto origins = llmp::test_support::qwen38::ModelOptOrigins();
  auto payer = std::make_shared<NumericalOwner>();
  llmp::kernels::StepBackend table{.device = NumericalDevice(),
                                   .construction_funder = NumericalFunder(payer)};
  const auto bind = [&](const llmp::kernels::StepBackend& backend) {
    return rt::AdmittedNumerics(fixture.model, fixture.source, origins, fixture.model.Digest(),
                                {.context = 8192, .slots = 4}, 0, backend);
  };
  const auto absent = bind({});
  ASSERT_TRUE(absent);
  EXPECT_FALSE(*absent);
  auto direct = bind(table);
  ASSERT_TRUE(direct && *direct);
  ASSERT_FALSE((*direct)->entries().empty());
  for (const auto change : {llmp::planner::NumericalChange::kBf16HyperConnectionIntermediates,
                            llmp::planner::NumericalChange::kMxfp8ProductInput,
                            llmp::planner::NumericalChange::kBf16ProductInputAndResult,
                            llmp::planner::NumericalChange::kNvfp4ExpertIntermediates})
    EXPECT_GT(std::ranges::count((*direct)->entries(), change,
                                 &llmp::planner::NumericalQualification::change),
              0);
  llmp::kernels::StepBackend token{.owner = std::make_shared<NumericalBackendOwner>(table, payer)};
  EXPECT_EQ(token.device.arch, 0U);
  EXPECT_TRUE(token.device.device_class.empty());
  EXPECT_FALSE(token.construction_funder);
  auto declared = bind(token);
  ASSERT_TRUE(declared && *declared);
  EXPECT_EQ((*declared)->entries().size(), (*direct)->entries().size());
  EXPECT_EQ((*declared)->digest(), (*direct)->digest());
  auto impostor = std::make_shared<NumericalOwner>();
  impostor->available = 0;
  token.device = {.device_class = "forged", .arch = 86};
  token.construction_funder = NumericalFunder(impostor);
  auto unchanged = bind(token);
  ASSERT_TRUE(unchanged && *unchanged);
  EXPECT_EQ((*unchanged)->digest(), (*direct)->digest());
  EXPECT_EQ(impostor->charges, 0U);
  // Plausible wrapper facts cannot authorize an unqualified owned device.
  auto unknown = table;
  unknown.device = {};
  token.owner = std::make_shared<NumericalBackendOwner>(std::move(unknown), payer);
  token.device = NumericalDevice();
  auto denied = bind(token);
  ASSERT_TRUE(denied);
  EXPECT_FALSE(*denied);
  // Nor can a wrapper's payer replace a missing authoritative payer.
  auto unpaid = table;
  unpaid.construction_funder.reset();
  token.owner = std::make_shared<NumericalBackendOwner>(std::move(unpaid), payer);
  token.construction_funder = table.construction_funder;
  EXPECT_FALSE(bind(token));
}

TEST_F(ComponentAdmissionTest, NativeNumericalContractsUseActualEncodingAcrossContainers) {
  auto fixture = llmp::test_support::qwen38::NativeModelOpt();
  FixturePreprocessing(fixture.model, fixture.source);
  auto opt_origins = llmp::test_support::qwen38::ModelOptOrigins();
  auto owner = std::make_shared<NumericalOwner>();
  const auto funder = NumericalFunder(owner);
  const auto semantic = fixture.model.Digest();
  const auto bind = [&] {
    return rt::AdmittedNumerics(fixture.model, fixture.source, opt_origins, semantic,
                                {.context = 8192, .slots = 4}, 0, NumericalDevice(), funder);
  };
  auto native = bind();
  ASSERT_TRUE(native && *native);
  fixture.source.container = "gguf";
  auto gguf_origin = bind();
  ASSERT_TRUE(gguf_origin && *gguf_origin);
  EXPECT_EQ((*native)->digest(), (*gguf_origin)->digest());
  bool hc = false, prepared = false, experts = false;
  for (const auto& entry : (*native)->entries()) {
    hc |= entry.change == llmp::planner::NumericalChange::kBf16HyperConnectionIntermediates;
    prepared |= entry.change == llmp::planner::NumericalChange::kMxfp8ProductInput;
    experts |= entry.change == llmp::planner::NumericalChange::kNvfp4ExpertIntermediates;
  }
  EXPECT_TRUE(hc);
  EXPECT_TRUE(prepared);
  EXPECT_TRUE(experts);
}

TEST(ComponentAdmissionModelOptTest, GdnMxfp8ResultRoundingReachesTheRegisteredRewrite) {
  namespace ir = llmp::ir;
  namespace pl = llmp::planner;
  namespace co = llmp::components;
  const auto fixture = llmp::test_support::qwen38::NativeModelOpt();
  const auto origins = llmp::test_support::qwen38::ModelOptOrigins();
  const auto instances = co::InstancesOf(fixture.model, 0);
  ASSERT_TRUE(instances);
  const co::Instance* gdn = nullptr;
  for (const auto& layer : instances->layers)
    for (const auto& instance : layer)
      if (!gdn && instance.ref->kind == "attention.gdn") gdn = &instance;
  ASSERT_NE(gdn, nullptr);
  const auto site = std::ranges::find(gdn->def->numerical, "gdn.product_bf16_intermediates",
                                      &co::NumericalSite::semantic);
  ASSERT_NE(site, gdn->def->numerical.end());
  for (const auto role : {"qkv", "gate", "alpha", "beta", "output", "norm"}) {
    for (const auto encoding : {fm::Encoding::kMxfp8, fm::Encoding::kBF16, fm::Encoding::kF16,
                                fm::Encoding::kF32, fm::Encoding::kNvfp4}) {
      EXPECT_EQ(
          co::NumericalSiteApplies(*site, *gdn, ir::Primitive(ir::Op::kMatMul), role, encoding),
          encoding == fm::Encoding::kMxfp8 &&
              (std::string_view(role) == "qkv" || std::string_view(role) == "gate"));
    }
  }
  auto payer = std::make_shared<NumericalOwner>();
  llmp::kernels::StepBackend table{.device = NumericalDevice(),
                                   .construction_funder = NumericalFunder(payer)};
  const llmp::kernels::StepBackend declared{
      .owner = std::make_shared<NumericalBackendOwner>(table, payer)};
  const auto numerical =
      rt::AdmittedNumerics(fixture.model, fixture.source, origins, fixture.model.Digest(),
                           {.context = 8192, .slots = 4}, 0, declared);
  ASSERT_TRUE(numerical && *numerical);
  // This fake implementation only answers availability: authority comes from
  // the real serving projection, never a fixture-created numerical grant.
  auto registry = llmp::execution::Registry::Create({llmp::execution::Implementation{
      .name = "test.gdn.product",
      .operation = ir::Primitive(ir::Op::kMatMul),
      .source = "fake",
      .revision = "r1",
      .build = "cpu",
      .variant = "test",
      .capability = llmp::execution::Capability{.op = ir::Primitive(ir::Op::kMatMul)}}});
  ASSERT_TRUE(registry);
  const auto* pattern = pl::FindPattern("bf16_rows");
  ASSERT_NE(pattern, nullptr);
  // The two observed prefill/scoring row counts, plus domain/ownership refusals.
  for (const std::uint64_t rows : {61U, 997U}) {
    for (const std::string_view role : {"qkv", "gate", "output"}) {
      for (const unsigned refusal : {0U, 1U, 2U, 3U}) {
        SCOPED_TRACE(std::string(role) + ":" + std::to_string(rows) + ":" +
                     std::to_string(refusal));
        const auto binding = std::ranges::find(gdn->ref->weights, role, &sp::WeightBinding::role);
        ASSERT_NE(binding, gdn->ref->weights.end());
        const auto source =
            std::ranges::find(fixture.source.tensors, binding->tensor, &sp::SourceTensor::name);
        ASSERT_NE(source, fixture.source.tensors.end());
        const auto ordinal = co::RoleIndex(*gdn->def, role);
        ASSERT_TRUE(ordinal);
        const auto k = source->format.shape.ne[0], n = source->format.shape.ne[1];
        const auto count = refusal == 3 ? 16385U : rows;  // beyond four owners' 4096 rows
        ir::Builder b;
        b.SetComponent(gdn->index);
        const auto input =
            b.Prepare(b.Input(ir::Dense(fm::Encoding::kF32, ir::ShapeOf({k, count}))),
                      fm::Encoding::kMxfp8, fm::Layout::kScaleAtoms);
        const auto weight = b.Weight(
            source->format, ir::WeightRef{.stage = static_cast<std::uint16_t>(refusal == 1 ? 1 : 0),
                                          .component = gdn->index,
                                          .role = *ordinal,
                                          .layer = gdn->layer});
        const auto product = b.MatMul(weight, input);
        const auto shaped = b.Reshape(product, ir::ShapeOf({n, count}));
        if (role == "gate") {
          const auto heads = b.Reshape(shaped, ir::ShapeOf({128, n / 128, count}));
          const auto values =
              b.Input(ir::Dense(fm::Encoding::kF32, ir::ShapeOf({128, n / 128, count})));
          const auto norm = b.Input(ir::Dense(fm::Encoding::kF32, ir::ShapeOf({128})));
          b.Output(b.Fused(ir::FusedKind::kNormGate, std::array{values, norm, heads},
                           ir::NormGateParams{.eps = 1e-6F, .gate = ir::NormGateFn::kSigmoid})[0]);
        } else {
          // Even the denied output role has a structurally eligible consumer:
          // graph matching alone must not substitute for the component's role.
          b.Output(b.Fused(ir::FusedKind::kGdnHistory, std::array{shaped},
                           ir::GdnHistoryParams{.taps = 3})[0]);
        }
        const auto graph = std::move(b).Finish();
        ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
        pl::Policy policy{.enabled = {"bf16_rows"}, .numerical = *numerical, .active_stage = 0};
        auto device = NumericalDevice();
        if (refusal == 2) device.arch = 86;
        const pl::SelectionView selection(*registry, device, policy, false, false);
        auto result = pl::ApplyPatternTable(std::span(pattern, 1), *graph, selection, nullptr);
        ASSERT_TRUE(result) << (result ? "" : result.error().detail);
        ASSERT_TRUE(result->dropped.empty());
        const bool rounded = role != "output" && refusal == 0;
        EXPECT_EQ(result->applied.size(), rounded ? 1U : 0U);
        for (const auto id : {product, shaped}) {
          const auto mapped = result->tensors.at(id.v);
          ASSERT_TRUE(mapped.valid());
          const auto& output = result->graph.tensor(mapped);
          EXPECT_EQ(output.format.encoding, rounded ? fm::Encoding::kBF16 : fm::Encoding::kF32);
          EXPECT_EQ(output.nb[0], rounded ? 2U : 4U);
          EXPECT_EQ(output.nb[1], n * (rounded ? 2U : 4U));
        }
      }
    }
  }
}

TEST(ComponentAdmissionGgufTest, MixedExpertsAndPleBindOnlyTheirObservedSites) {
  namespace co = llmp::components;
  namespace pl = llmp::planner;
  auto checkpoint = llmp::test_support::Qwen38GgufCheckpoint();
  const auto bound = llmp::import::NormalizationBound(checkpoint);
  ASSERT_TRUE(bound);
  auto normalized =
      llmp::import::Normalize(std::move(checkpoint), {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(normalized) << (normalized ? "" : normalized.error().ToString());
  auto source = normalized->canonical();
  const auto& model = normalized->spec();
  const auto instances = co::InstancesOf(model, 0);
  ASSERT_TRUE(instances);
  auto owner = std::make_shared<NumericalOwner>();
  const auto bind = [&] {
    return rt::AdmittedNumerics(model, source, Origins(), model.Digest(),
                                {.context = 8192, .slots = 4}, 0, NumericalDevice(),
                                NumericalFunder(owner));
  };
  const auto facts = bind();
  ASSERT_TRUE(facts && *facts);
  ExpectDefinedAttribution((*facts)->entries());
  unsigned ple_products = 0, mixed_experts = 0;
  for (const auto& layer : instances->layers) {
    for (const auto& instance : layer) {
      for (const auto& binding : instance.ref->weights) {
        const auto tensor =
            std::ranges::find(source.tensors, binding.tensor, &sp::SourceTensor::name);
        ASSERT_NE(tensor, source.tensors.end());
        const bool ple = instance.ref->kind == "residual.ple" &&
                         (binding.role == "key" || binding.role == "value");
        const bool mixed = instance.ref->kind == "ffn.moe" &&
                           (binding.role == "gate_experts" || binding.role == "up_experts") &&
                           tensor->format.encoding == fm::Encoding::kIQ3_S;
        if (!ple && !mixed) continue;
        if (ple) {
          ++ple_products;
          EXPECT_EQ(tensor->format.encoding, fm::Encoding::kQ8_0);
          const auto& site = instance.def->numerical.front();
          EXPECT_FALSE(co::NumericalSiteApplies(site, instance,
                                                llmp::ir::Primitive(llmp::ir::Op::kMatMul), "conv",
                                                fm::Encoding::kQ8_0));
          EXPECT_FALSE(co::NumericalSiteApplies(site, instance,
                                                llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                                                binding.role, fm::Encoding::kQ6_K));
        } else {
          ++mixed_experts;
        }
        const auto ordinal = co::RoleIndex(*instance.def, binding.role);
        ASSERT_TRUE(ordinal);
        unsigned vector_rows = 0;
        for (const auto& grant : (*facts)->entries()) {
          if (grant.component != instance.index || grant.weight_role != *ordinal ||
              llmp::base::ToHex(grant.evidence) !=
                  "8b9e1298f0e1759cfae0c3f1c792fe79052a1af02c9b9d4e93e9aca137512041")
            continue;
          ++vector_rows;
          EXPECT_EQ(grant.change, pl::NumericalChange::kQ8VectorReduction);
          EXPECT_EQ(grant.anchor,
                    llmp::ir::Primitive(mixed ? llmp::ir::Op::kMatMulId : llmp::ir::Op::kMatMul));
          EXPECT_GE(grant.min_rows, 1U);
          EXPECT_LE(grant.max_rows, 4U);
          EXPECT_EQ(grant.min_rows, grant.max_rows);
          EXPECT_EQ(grant.max_k, tensor->format.shape.ne[0]);
          EXPECT_EQ(grant.max_n, tensor->format.shape.ne[1]);
          EXPECT_TRUE(grant.exact_dimensions);
        }
        EXPECT_EQ(vector_rows, 4U);
      }
    }
  }
  EXPECT_EQ(ple_products, 2U);
  EXPECT_EQ(mixed_experts, 2U);
  // The mixed expert observation cannot authorize a dense IQ3_S product.
  ASSERT_TRUE(instances->output);
  const auto& output = *instances->output;
  ASSERT_FALSE(output.ref->weights.empty());
  const auto head = std::ranges::find(source.tensors, output.ref->weights.front().tensor,
                                      &sp::SourceTensor::name);
  ASSERT_NE(head, source.tensors.end());
  head->format.encoding = fm::Encoding::kIQ3_S;
  const auto dense = bind();
  ASSERT_TRUE(dense && *dense);
  EXPECT_EQ(
      std::ranges::count((*dense)->entries(), output.index, &pl::NumericalQualification::component),
      0U);
}

TEST(ComponentAdmissionImageTest, NumericalContractsRequireDeclaredSiteStageDeviceAndEncoding) {
  auto owner = std::make_shared<NumericalOwner>();
  const auto funder = NumericalFunder(owner);
  const auto& family = llmp::test_support::QwenImage();
  const auto temp_dir =
      std::filesystem::temp_directory_path() / std::format("llmp-test-img-contract-{}", ::getpid());
  llmp::test::CompositionPlan publication;
  publication.components = {{"transformer", std::string(64, 'c'), "QwenImage21Transformer2DModel"}};
  publication.meta = {{"model_index.json", R"({"_class_name":"QwenImage21Pipeline"})"}};
  auto opened = llmp::artifact::OpenComposition(
      llmp::test::Publish(temp_dir, llmp::test::Render(publication)));
  ASSERT_TRUE(opened) << (opened ? "" : opened.error().ToString());
  auto other_publication = publication;
  other_publication.components[0] = {"transformer", std::string(64, 'd'),
                                     "QwenImage21Transformer2DModel"};
  const auto other_temp_dir = temp_dir.string() + "-other";
  auto other_opened = llmp::artifact::OpenComposition(
      llmp::test::Publish(other_temp_dir, llmp::test::Render(other_publication)));
  ASSERT_TRUE(other_opened) << (other_opened ? "" : other_opened.error().ToString());
  ASSERT_EQ(opened->sources().size(), other_opened->sources().size());
  for (std::size_t i = 0; i < opened->sources().size(); ++i) {
    EXPECT_EQ(opened->sources()[i].name, other_opened->sources()[i].name);
    EXPECT_EQ(opened->sources()[i].bytes, other_opened->sources()[i].bytes);
    EXPECT_EQ(opened->sources()[i].sha256, other_opened->sources()[i].sha256);
  }
  EXPECT_NE(opened->id(), other_opened->id());
  sp::Stage stage{.name = "projection", .role = sp::StageRole::kDenoiser, .artifact = 0};
  stage.stream.width = 4096;
  stage.layers.push_back({.components = {{.kind = "projection.spatial",
                                          .version = 1,
                                          .params = {{"in_channels", std::int64_t{4096}},
                                                     {"out_channels", std::int64_t{4096}}},
                                          .weights = {{.role = "weight", .tensor = "w"}}}}});
  sp::ModelSpec generated{.pipeline = sp::PipelineKind::kImage, .stages = {stage}};
  const auto make_graph = [](std::uint16_t weight_stage, std::uint64_t rows = 25,
                             std::uint64_t width = 4096,
                             fm::Encoding encoding = fm::Encoding::kBF16) {
    llmp::ir::Builder b;
    b.SetComponent(1);
    const auto x = b.Input(llmp::ir::Dense(fm::Encoding::kF32, llmp::ir::ShapeOf({width, rows})));
    auto format = llmp::ir::Dense(encoding, llmp::ir::ShapeOf({width, width}));
    if (fm::Info(encoding).quantized) format.layout = fm::Layout::kGgmlBlocks;
    const auto w = b.Weight(format, {.stage = weight_stage, .component = 1, .role = 0});
    b.Output(b.MatMul(w, x));
    return std::move(b).Finish();
  };
  auto graph = make_graph(1);
  ASSERT_TRUE(graph);
  const auto bind = [&](const llmp::planner::DeviceFacts& device) {
    return rt::AdmittedImageNumerics(family.spec, *opened, *graph, generated, 1, device, funder);
  };
  auto unpinned = bind(NumericalDevice());
  ASSERT_TRUE(unpinned && *unpinned);
  auto other_identity = rt::AdmittedImageNumerics(family.spec, *other_opened, *graph, generated, 1,
                                                  NumericalDevice(), funder);
  ASSERT_TRUE(other_identity && *other_identity);
  EXPECT_NE((*other_identity)->digest(), (*unpinned)->digest());
  for (const auto& entry : (*unpinned)->entries()) {
    EXPECT_EQ(entry.stage, 1U);
    EXPECT_EQ(entry.component, 1U);
    EXPECT_EQ(entry.weight_role, 0U);
    EXPECT_TRUE(entry.exact_shape);
  }
  auto sm86 = NumericalDevice();
  sm86.device_class = "sm_86";
  sm86.arch = 86;
  for (const auto& device : {sm86, llmp::planner::DeviceFacts{}}) {
    auto absent = bind(device);
    ASSERT_TRUE(absent);
    EXPECT_FALSE(*absent);
  }
  // Structural eligibility does not depend on the retained benchmark geometry.
  for (const auto [rows, width] : {std::pair{10U, 256U}, std::pair{77U, 4096U}}) {
    generated.stages[0].stream.width = width;
    generated.stages[0].layers[0].components[0].params = {{"in_channels", std::int64_t{width}},
                                                          {"out_channels", std::int64_t{width}}};
    graph = make_graph(1, rows, width);
    ASSERT_TRUE(graph);
    auto other_shape = bind(NumericalDevice());
    ASSERT_TRUE(other_shape && *other_shape);
    for (const auto& entry : (*other_shape)->entries()) {
      EXPECT_EQ(entry.min_rows, rows);
      EXPECT_EQ(entry.max_k, width);
      EXPECT_EQ(entry.max_n, width);
    }
  }
  const auto covered_charges = owner->charges;
  graph = make_graph(1, 25, 4096, fm::Encoding::kF32);
  ASSERT_TRUE(graph);
  auto wrong_encoding = bind(NumericalDevice());
  ASSERT_TRUE(wrong_encoding);
  EXPECT_FALSE(*wrong_encoding);
  for (const auto encoding : {fm::Encoding::kQ4_K, fm::Encoding::kQ6_K}) {
    graph = make_graph(1, 25, 4096, encoding);
    ASSERT_TRUE(graph);
    auto unqualified_quant = bind(NumericalDevice());
    ASSERT_TRUE(unqualified_quant);
    EXPECT_FALSE(*unqualified_quant);
  }
  graph = make_graph(0);
  ASSERT_TRUE(graph);
  auto wrong_stage = bind(NumericalDevice());
  ASSERT_TRUE(wrong_stage);
  EXPECT_FALSE(*wrong_stage);
  graph = make_graph(1);
  // An unrelated component with the same operation and ordinal has no grant.
  generated.stages[0].layers[0].components[0] = {.kind = "norm.rms",
                                                 .version = 1,
                                                 .params = {{"eps", 1e-6}},
                                                 .weights = {{.role = "weight", .tensor = "w"}}};
  auto unrelated = bind(NumericalDevice());
  ASSERT_TRUE(unrelated);
  EXPECT_FALSE(*unrelated);
  EXPECT_EQ(owner->charges, covered_charges);
  std::error_code ec;
  std::filesystem::remove_all(temp_dir, ec);
  std::filesystem::remove_all(other_temp_dir, ec);
}

TEST(ComponentAdmissionImageTest, SpatialProjectionAcceptsValidBlockLinearSource) {
  sp::Stage stage{.name = "flow", .role = sp::StageRole::kDenoiser, .artifact = 0};
  stage.stream.width = 256;
  stage.layers.push_back(
      {.components = {
           {.kind = "projection.spatial",
            .version = 1,
            .params = {{"in_channels", std::int64_t{256}}, {"out_channels", std::int64_t{256}}},
            .weights = {{.role = "weight", .tensor = "flow.weight"}}}}});
  sp::ModelSpec model{.pipeline = sp::PipelineKind::kImage, .stages = {std::move(stage)}};
  fm::TensorFormat q4;
  q4.encoding = fm::Encoding::kQ4_K;
  q4.layout = fm::Layout::kGgmlBlocks;
  q4.shape = *fm::ShapeOf(std::array<std::uint64_t, 2>{256, 256});
  const std::array<sp::SourceTensor, 1> source{
      sp::SourceTensor{.name = "flow.weight", .format = q4, .part = fm::PartRole::kData}};
  const auto checked = llmp::components::ValidateSpec(model, source);
  EXPECT_TRUE(checked.has_value()) << (checked ? "" : checked.error());
}

TEST(ComponentAdmissionImageTest, AttentionTuningNeedsDeviceEvidenceAndActualAlgorithmHeadWidth) {
  auto gb10 = NumericalDevice();
  EXPECT_TRUE(rt::HasImageAttentionEvidence(gb10, 128));
  EXPECT_FALSE(rt::HasImageAttentionEvidence(gb10, 64));
  EXPECT_FALSE(rt::HasImageAttentionEvidence(gb10, 256));
  gb10.arch = 86;
  EXPECT_FALSE(rt::HasImageAttentionEvidence(gb10, 128));
  EXPECT_FALSE(rt::HasImageAttentionEvidence({}, 128));
}

TEST_F(ComponentAdmissionTest, ActualSpecChangesInvalidateCalibrationForTheSameArtifact) {
  rt::CalibrationKey key{.artifact = std::string(64, 'a'),
                         .drafter = {},
                         .device = "test device",
                         .build = "test build",
                         .settings = {}};
  const std::array<std::string_view, 1> artifacts{key.artifact};
  auto digest = sp::BoundDigest(model, artifacts);
  ASSERT_TRUE(digest) << (digest ? "" : digest.error());
  rt::ModelSettings settings;
  settings.bound_spec = *digest;
  key.settings = rt::MeasuredWith(settings);
  const auto record = rt::FormatCalibration(rt::Calibration{}, key);
  EXPECT_TRUE(rt::ParseCalibration(record, key).calibration);
  // Same artifact/build/settings, different real spec semantics and unchanged
  // layout sizes: calibration must still become stale.
  ASSERT_TRUE(model.stages.front().heads.softcap);
  model.stages.front().heads.softcap = *model.stages.front().heads.softcap + 1.0;
  digest = sp::BoundDigest(model, artifacts);
  ASSERT_TRUE(digest) << (digest ? "" : digest.error());
  settings.bound_spec = *digest;
  key.settings = rt::MeasuredWith(settings);
  EXPECT_FALSE(rt::ParseCalibration(record, key).calibration);
}

TEST_F(ComponentAdmissionTest, ApprovesCompleteNativeFormatsAndPreservesTheStoredSpec) {
  const auto before = model.Digest();
  const auto admitted = rt::CheckComponentAdmission(model, source, origins);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
  EXPECT_EQ(model.Digest(), before);
  EXPECT_EQ(model.stages.front().context, 8192U);
  EXPECT_EQ(model.stages.front().folds, std::vector<std::string>{"norm.plus_one"});
  EXPECT_EQ(model.stages.front().embedding->weights.front().tensor,
            model.stages.front().heads.output->weights.front().tensor);
}

TEST_F(ComponentAdmissionTest, FoldReceiptsCoverEveryBoundRoleAndKeepTheirOrigin) {
  ASSERT_EQ(source.preprocessing.size(), 1U);
  ASSERT_GT(source.preprocessing[0].tensors.size(), 1U);
  const auto original = source;
  source.preprocessing[0].tensors.pop_back();
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source = original;
  source.preprocessing[0].tensors.push_back("token_embd.weight");
  std::ranges::sort(source.preprocessing[0].tensors);
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source = original;
  source.preprocessing[0].source = false;
  // A converter claim cannot stand in for the importer's separate fold inventory.
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source.folds = {"norm.plus_one"};
  EXPECT_TRUE(rt::CheckComponentAdmission(model, source, origins));
  source.preprocessing.push_back(original.preprocessing[0]);
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
}

TEST_F(ComponentAdmissionTest, NativeDialectRequiresItsImporterFoldReceipt) {
  source.container = "safetensors";
  source.preprocessing.clear();
  // Native source vectors have no converter-asserted fold. A folded stored
  // stage needs the importer's receipt, independent of the canonical registry pointer.
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source.folds = {"norm.plus_one"};
  FixturePreprocessing(model, source);
  EXPECT_TRUE(rt::CheckComponentAdmission(model, source, origins));
  source.folds = {"norm.times_two"};
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source.folds = {"norm.plus_one"};
  source.container = "mlx";
  EXPECT_TRUE(rt::CheckComponentAdmission(model, source, origins));
  source.preprocessing.clear();
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
}

TEST_F(ComponentAdmissionTest,
       StoredSpecAdmitsUnregisteredArchitectureWhileMalformedOriginsStillFail) {
  origins[0].sha256[0] ^= 1;
  EXPECT_TRUE(rt::CheckComponentAdmission(model, source, origins));
  origins = Origins();
  origins[0].bytes = llmp::base::Bytes(origins[0].bytes.value() - 1);
  EXPECT_TRUE(rt::CheckComponentAdmission(model, source, origins));
  origins[0].bytes = llmp::base::Bytes(0);
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  origins = Origins();
  source.architecture = "unknown_architecture";
  EXPECT_TRUE(rt::CheckComponentAdmission(model, source, origins));
  EXPECT_EQ(rt::FindApproval(source.architecture, origins), nullptr);
  auto policy = rt::ComponentAdmissionPolicy(model, source, origins, {.context = 128, .slots = 2});
  ASSERT_TRUE(policy) << policy.error();
  EXPECT_TRUE(policy->text.empty());
}

TEST_F(ComponentAdmissionTest, StoredContextAndArithmeticDoNotRequireHistoricalApproval) {
  const auto original = model;
  ++model.stages.front().context;
  EXPECT_TRUE(rt::CheckComponentAdmission(model, source, origins));
  model = original;
  model.stages.front().heads.softcap = 29.0;
  EXPECT_TRUE(rt::CheckComponentAdmission(model, source, origins));
  model.stages.front().sampling.temperature = 0.75;
  EXPECT_TRUE(rt::CheckComponentAdmission(model, source, origins));
  model = original;
  model.stages.front().folds.clear();
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  model = original;
  model.stages.front().heads.output->weights.front().tensor = "output.weight";
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  model = original;
  source.folds = {"norm.plus_one"};
  FixturePreprocessing(model, source);
  EXPECT_TRUE(rt::CheckComponentAdmission(model, source, origins));
}

TEST_F(ComponentAdmissionTest, RejectsNativeFormatGeometryAndRoleChanges) {
  const auto original = source;
  auto& format = source.tensors.front().format;
  format.encoding = static_cast<fm::Encoding>(UINT16_MAX);
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source = original;
  source.tensors.front().format.shape.ne[0] += 32;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source = original;
  source.tensors.front().part = fm::PartRole::kCodes;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source = original;
  source.tensors.front().name = "unbound.weight";
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
}

TEST_F(ComponentAdmissionTest, PolicyRechecksTheApprovedSpecBeforeSelectingItsRecipe) {
  auto mapping = sp::mappings::Gemma2Mapping();
  auto rule = rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8192, .slots = 2});
  ASSERT_TRUE(rule) << (rule ? "" : rule.error());
  EXPECT_EQ(rule->text, mapping.policies.front().text);
  // The stored stage context is a bound; the approved envelope (two owners)
  // a default an override passes (D-119).
  EXPECT_FALSE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8193, .slots = 2}));
  EXPECT_TRUE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8192, .slots = 3}));
  model.stages.front().context = 8193;
  EXPECT_TRUE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8192, .slots = 2}));
}

// An approved row's envelope is its settings' default, not a refusal: an
// envelope past it is admitted, its mapping's rule where one holds, else
// the engine's default; within it, an approval whose policy is its
// mapping's still refuses an envelope no rule covers.
TEST_F(ComponentAdmissionTest, AnApprovedEnvelopeIsADefaultNotABound) {
  const rt::Approval* approval = rt::FindApproval(source.architecture, origins);
  ASSERT_NE(approval, nullptr);
  EXPECT_EQ(approval->max_slots, 4U);
  EXPECT_TRUE(rt::WithinApprovedEnvelope(*approval, {.context = 8192, .slots = 4}));
  EXPECT_FALSE(rt::WithinApprovedEnvelope(*approval, {.context = 8192, .slots = 5}));
  const auto past = rt::AdmittedPolicy(model, source, approval, {.context = 4096, .slots = 16});
  ASSERT_TRUE(past) << (past ? "" : past.error());
  EXPECT_FALSE(rt::AdmittedPolicy(model, source, approval, {.context = 8193, .slots = 1}));
  EXPECT_FALSE(rt::AdmittedPolicy(model, source, approval, {.context = 8192, .slots = 0}));
}

TEST(ComponentAdmissionQwenTest, FullNormalizedTargetKeepsItsUntiedHeadAndSourceHash) {
  auto checkpoint = llmp::test_support::Qwen38GgufCheckpoint();
  auto bound = llmp::import::NormalizationBound(checkpoint);
  ASSERT_TRUE(bound);
  auto normalized =
      llmp::import::Normalize(std::move(checkpoint), {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(normalized) << (normalized ? "" : normalized.error().ToString());
  std::array<llmp::artifact::Source, 3> origins{
      Origins("Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf", 10946624,
              "268f81fdedf3149a538f252308927a4d5d1f6e062c178568a51e3b519744f8a8")[0],
      Origins("Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00003.gguf", 49567921344ULL,
              "cfe600b236b88c7fad1613a5ca5e83b9f2beb63cbd44c32b2be50a44747c695f")[0],
      Origins("Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00003.gguf", 32382955968ULL,
              "f1912ba34c79427d2295a58dcb2b732b5931af5bef7a373c60557a57d9ee7250")[0]};
  auto model = normalized->spec();
  auto source = normalized->canonical();
  FixturePreprocessing(model, source);
  const auto admitted = rt::CheckComponentAdmission(model, source, origins);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
  EXPECT_TRUE(rt::OnComponentEngine("qwen4exp"));
  EXPECT_NE(model.stages[0].embedding->weights[0].tensor,
            model.stages[0].heads.output->weights[0].tensor);
  auto policy = rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8192, .slots = 4});
  ASSERT_TRUE(policy) << (policy ? "" : policy.error());
  EXPECT_TRUE(
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262144, .slots = 8}));
  EXPECT_FALSE(
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262145, .slots = 1}));
  for (int mutation = 0; mutation != 7; ++mutation) {
    auto changed_model = model;
    auto changed_source = source;
    auto changed_origins = origins;
    switch (mutation) {
      case 0:
        changed_origins[2].bytes = llmp::base::Bytes(0);
        break;
      case 1:
        changed_origins[2] = changed_origins[1];
        break;
      case 2:
        changed_model.stages[0].context = 0;
        break;
      case 3:
        changed_model.stages[0].folds.clear();
        break;
      case 4:
        changed_model.stages[0].heads.output->weights[0].tensor = "missing.output.weight";
        break;
      case 5:
        changed_source.tensors[0].format.encoding = static_cast<fm::Encoding>(UINT16_MAX);
        break;
      case 6:
        ASSERT_FALSE(changed_model.stages[0].side.empty());
        for (auto& param : changed_model.stages[0].side[0].source.params)
          if (param.name == "multipliers") std::get<std::vector<std::int64_t>>(param.value)[0] += 2;
        break;
    }
    const auto changed =
        rt::CheckComponentAdmission(changed_model, changed_source, changed_origins);
    if (mutation == 6) {
      // A valid stored arithmetic change is not rejected by a historical
      // checkpoint digest; numerical qualification remains a separate step.
      EXPECT_TRUE(changed);
    } else {
      EXPECT_FALSE(changed) << mutation;
    }
  }
}

class NativeGemmaAdmissionTest : public testing::TestWithParam<std::uint32_t> {
 protected:
  sp::Source source;
  sp::ModelSpec model;
  std::array<llmp::artifact::Source, 1> origins;
  void SetUp() override {
    std::vector<sp::SourceValue> values;
    if (GetParam() == 3) {
      source = llmp::test_support::Gemma3Source();
      const auto approved = fam::Gemma3Values(fam::Gemma3_4BQat());
      ASSERT_TRUE(approved);
      values = *approved;
      origins = Origins("gemma-3-4b-it-qat-Q4_0.gguf", 2526080992ULL,
                        "ee91c3e7a4ab95d8c95672f9fcb58bf236b257e9f217966bcf53a5a6df4ab49a");
    } else {
      source = llmp::test_support::Gemma4Source(GetParam());
      // Independent captured GGUF metadata, not the compiled profile values
      // that admission uses as its comparison oracle.
      values = llmp::test_support::Gemma4FixtureValues(GetParam());
      const auto doc = llmp::test_support::gemma4::Fixture(GetParam());
      const auto origin = llmp::test_support::gemma4::Get(doc.root(), "source");
      const auto get = [&](std::string_view key) {
        return llmp::test_support::gemma4::Get(origin, key);
      };
      origins =
          Origins(get("file").string(), static_cast<std::uint64_t>(*get("full_bytes").int64()),
                  get("full_sha256").string());
    }
    auto made = sp::Synthesize(*sp::FindMapping(source.architecture), source, values);
    ASSERT_TRUE(made) << (made ? "" : made.error());
    model = std::move(*made);
    FixturePreprocessing(model, source);
  }
};
TEST_P(NativeGemmaAdmissionTest, CapturedDirectoryAndSpecAreAcceptedWithoutMutation) {
  const auto before = model.Digest();
  const auto admitted = rt::CheckComponentAdmission(model, source, origins);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
  EXPECT_EQ(before, model.Digest());
  std::ranges::reverse(source.tensors);
  EXPECT_TRUE(rt::CheckComponentAdmission(model, source, origins));
  EXPECT_EQ(model.stages.front().folds.empty(), GetParam() != 3);
}
TEST_P(NativeGemmaAdmissionTest, RefusesChangedFormatPlacementExpertIdentityAndArithmetic) {
  const auto saved_source = source;
  auto matrix =
      std::ranges::find_if(source.tensors, [](const auto& t) { return t.format.shape.rank == 2; });
  ASSERT_NE(matrix, source.tensors.end());
  matrix->format.encoding = static_cast<fm::Encoding>(UINT16_MAX);
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source = saved_source;
  source.tensors.front().format.params.tensor_scale = true;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source = saved_source;
  source.tensors.front().experts = !source.tensors.front().experts;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source = saved_source;
  const auto saved_model = model;
  model.stages.front().heads.output->weights.front().tensor = "output.weight";
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  model = saved_model;
  model.stages.front().context = 0;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  model = saved_model;
  model.stages.front().role = sp::StageRole::kDrafter;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  model = saved_model;
  origins.front().sha256.front() ^= 1;
  EXPECT_TRUE(rt::CheckComponentAdmission(model, source, origins));
  origins.front().bytes = llmp::base::Bytes(0);
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
}
TEST_P(NativeGemmaAdmissionTest, ChoosesExactProfileAndEnvelopeRecipeWithoutCheckpointKeys) {
  const bool gemma3 = GetParam() == 3;
  const std::uint32_t context = gemma3 ? 131072 : 8192, slots = gemma3 ? 1 : 4;
  const auto rule =
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = context, .slots = slots});
  ASSERT_TRUE(rule) << (rule ? "" : rule.error());
  EXPECT_EQ(rule->prefill_chunk, gemma3 ? 0U : GetParam() == 26 ? 1024U : 256U);
  EXPECT_EQ(rule->prompts_alone, !gemma3);
  EXPECT_EQ(rule->text,
            gemma3 ? "max_decode_rows 16\n" : "max_decode_rows 16\ndisable owner_padded_reads\n");
  if (gemma3) {
    EXPECT_TRUE(
        rt::ComponentAdmissionPolicy(model, source, origins, {.context = 4096, .slots = 2}));
    // Past the approval's wide context with two owners: an override past
    // its envelope (the recipe's own owner bound is the settings').
    EXPECT_TRUE(
        rt::ComponentAdmissionPolicy(model, source, origins, {.context = 4097, .slots = 2}));
  }
  if (!gemma3) {
    EXPECT_EQ(rule->text, "max_decode_rows 16\ndisable owner_padded_reads\n");
    const auto small =
        rt::ComponentAdmissionPolicy(model, source, origins, {.context = 4096, .slots = 2});
    ASSERT_TRUE(small);
    EXPECT_EQ(small->text, rule->text);
  }
  if (gemma3) {
    EXPECT_EQ(rt::ComponentAdmissionPolicy(model, source, origins,
                                           {.context = context + 1, .slots = slots})
                  .has_value(),
              model.stages.front().context > context);
    EXPECT_TRUE(rt::ComponentAdmissionPolicy(model, source, origins,
                                             {.context = context, .slots = slots + 1}));
  } else {
    // Past the joined recipes' domains, the owners-alone route at M3's
    // bounded admission: twelve scalar owners qualified; context to the
    // trained 262,144, long context not verified. Beyond it, none.
    for (const auto& envelope : {sp::Envelope{.context = context + 1, .slots = slots},
                                 sp::Envelope{.context = context, .slots = slots + 1},
                                 sp::Envelope{.context = 262144, .slots = 12}}) {
      const auto alone = rt::ComponentAdmissionPolicy(model, source, origins, envelope);
      ASSERT_TRUE(alone) << (alone ? "" : alone.error());
      EXPECT_TRUE(alone->owners_alone);
      EXPECT_EQ(alone->text, rule->text);
    }
    EXPECT_FALSE(
        rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262145, .slots = 1}));
    // Past the approved twelve owners: admitted, owners alone.
    const auto more =
        rt::ComponentAdmissionPolicy(model, source, origins, {.context = 4096, .slots = 13});
    ASSERT_TRUE(more) << (more ? "" : more.error());
    EXPECT_TRUE(more->owners_alone);
  }
  EXPECT_FALSE(
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 0, .slots = slots}));
  // A shorter, component-valid stage is a new valid geometry. Refusal is
  // structural, rather than a comparison with a historical layer count.
  model.stages.front().layers.pop_back();
  if (!source.preprocessing.empty())
    EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  FixturePreprocessing(model, source);
  EXPECT_TRUE(rt::CheckComponentAdmission(model, source, origins));
  EXPECT_TRUE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 4096, .slots = 2}));
  ASSERT_FALSE(model.stages.front().layers.front().components.front().weights.empty());
  model.stages.front().layers.front().components.front().weights.front().tensor = "missing";
  EXPECT_FALSE(llmp::components::ValidateSpec(model, source.tensors));
  EXPECT_FALSE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 4096, .slots = 2}));
}
INSTANTIATE_TEST_SUITE_P(ApprovedTargets, NativeGemmaAdmissionTest, testing::Values(3U, 26U, 31U));

TEST(ComponentAdmissionDsv4Test, FullStoredTargetPreservesProfilePolicyAndClosedSourceAuthority) {
  auto checkpoint = llmp::test_support::Dsv4GgufCheckpoint();
  auto bound = llmp::import::NormalizationBound(checkpoint);
  ASSERT_TRUE(bound) << (bound ? "" : bound.error().ToString());
  auto normalized =
      llmp::import::Normalize(std::move(checkpoint), {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(normalized) << (normalized ? "" : normalized.error().ToString());
  std::array<llmp::artifact::Source, 3> origins{
      Origins("DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf", 5257664,
              "de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b")[0],
      Origins("DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf", 49437013568ULL,
              "c8728a298862cd8736782e2e4056e5e3e03c415ba552a38173264494e41ef327")[0],
      Origins("DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf", 47390237120ULL,
              "a4bae0788c025c1b2388754592c98fefa1372981748ac9875039bf6a30e6eeb4")[0]};
  const auto& model = normalized->spec();
  const auto& source = normalized->canonical();
  const auto admitted = rt::CheckComponentAdmission(model, source, origins);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
  // Served on the component engine since c96deafd (K1).
  EXPECT_TRUE(rt::OnComponentEngine("deepseek4"));
  auto policy =
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 1048576, .slots = 8});
  ASSERT_TRUE(policy) << (policy ? "" : policy.error());
  EXPECT_TRUE(policy->text.empty());
  EXPECT_FALSE(
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 1048577, .slots = 1}));
  for (unsigned mutation = 0; mutation != 6; ++mutation) {
    auto changed_model = model;
    auto changed_source = source;
    auto changed_origins = origins;
    if (mutation == 0) changed_origins[2].bytes = llmp::base::Bytes(0);
    if (mutation == 1) changed_origins[2] = changed_origins[1];
    if (mutation == 2)
      changed_source.tensors[0].format.encoding = static_cast<fm::Encoding>(UINT16_MAX);
    if (mutation == 3) changed_model.stages[0].side.pop_back();
    if (mutation == 4)
      changed_model.stages[0].heads.output->weights[0].tensor = "missing.output.weight";
    if (mutation == 5) changed_model.stages[0].folds.push_back("norm.plus_one");
    EXPECT_FALSE(rt::CheckComponentAdmission(changed_model, changed_source, changed_origins))
        << mutation;
  }
}

TEST(ComponentAdmissionQwen2Test, FullStoredTargetPreservesProfilePolicyAndClosedSourceAuthority) {
  auto checkpoint = llmp::test_support::Qwen2GgufCheckpoint();
  auto bound = llmp::import::NormalizationBound(checkpoint);
  ASSERT_TRUE(bound) << (bound ? "" : bound.error().ToString());
  auto normalized =
      llmp::import::Normalize(std::move(checkpoint), {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(normalized) << (normalized ? "" : normalized.error().ToString());
  const auto origins = Origins("qwen2.5-0.5b-instruct-fp16.gguf", 1266425696ULL,
                               "8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc");
  const auto& model = normalized->spec();
  const auto& source = normalized->canonical();
  const auto admitted = rt::CheckComponentAdmission(model, source, origins);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
  EXPECT_TRUE(rt::OnComponentEngine("qwen2"));
  auto policy = rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8192, .slots = 1});
  ASSERT_TRUE(policy) << (policy ? "" : policy.error());
  EXPECT_TRUE(policy->text.contains("max_decode_rows 1"));
  // The stored stage context bounds it; the approval's one owner is a
  // default an override passes.
  EXPECT_EQ(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8193, .slots = 1})
                .has_value(),
            model.stages.front().context >= 8193);
  EXPECT_TRUE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8192, .slots = 2}));
  for (unsigned mutation = 0; mutation != 5; ++mutation) {
    auto changed_model = model;
    auto changed_source = source;
    auto changed_origins = origins;
    if (mutation == 0) changed_origins[0].name.clear();
    if (mutation == 1) changed_origins[0].bytes = llmp::base::Bytes(0);
    if (mutation == 2)
      changed_source.tensors[0].format.encoding = static_cast<fm::Encoding>(UINT16_MAX);
    if (mutation == 3)
      changed_model.stages[0].heads.output->weights[0].tensor = "missing.output.weight";
    if (mutation == 4) changed_model.stages[0].folds.push_back("norm.plus_one");
    EXPECT_FALSE(rt::CheckComponentAdmission(changed_model, changed_source, changed_origins))
        << mutation;
  }
}

TEST(ComponentAdmissionQwen2Exl3Test,
     Approved40And45TargetsPreserveMultipartInventoryAndClosedSourceAuthority) {
  for (const bool rate_4_5 : {false, true}) {
    auto checkpoint = llmp::test_support::Qwen2Exl3Checkpoint(rate_4_5);
    auto bound = llmp::import::NormalizationBound(checkpoint);
    ASSERT_TRUE(bound) << (bound ? "" : bound.error().ToString());
    auto normalized =
        llmp::import::Normalize(std::move(checkpoint), {bound->metadata_bytes, bound->work_units});
    ASSERT_TRUE(normalized) << (normalized ? "" : normalized.error().ToString());
    std::array<llmp::artifact::Source, 5> origins{
        Origins("config.json", rate_4_5 ? 923ULL : 996ULL,
                rate_4_5 ? "dcd945f38ba6e7835c8c281a45c03d9a20759d03afe5761d09def8c62aa11f3c"
                         : "966657b7e8c331ea16127a868eb6d6ff16ad0b16a5ee39d26279e12de0a3205c")[0],
        Origins("generation_config.json", 242ULL,
                "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6")[0],
        Origins("model.safetensors", rate_4_5 ? 611257914ULL : 588951098ULL,
                rate_4_5 ? "94c45f9838a56fbf23f43d0a04c47e187d840dd97f17b4a027a69a20339762bc"
                         : "66b61bd1ca3ebb0fc1cea29148eda8bdaca45d232c6a7e319c6d2835b65c066b")[0],
        Origins("tokenizer.json", 7031645ULL,
                "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539")[0],
        Origins("tokenizer_config.json", 7305ULL,
                "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583")[0]};
    const auto& model = normalized->spec();
    const auto& source = normalized->canonical();
    const auto admitted = rt::CheckComponentAdmission(model, source, origins);
    ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
    auto policy =
        rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8192, .slots = 1});
    ASSERT_TRUE(policy) << (policy ? "" : policy.error());
    EXPECT_TRUE(policy->text.contains("max_decode_rows 1"));
    for (unsigned mutation = 0; mutation != 6; ++mutation) {
      auto changed_model = model;
      auto changed_source = source;
      auto changed_origins = origins;
      if (mutation == 0) changed_origins[0].bytes = llmp::base::Bytes(0);
      if (mutation == 1) changed_origins[1] = changed_origins[0];
      if (mutation == 2) changed_source.container = "unknown";
      if (mutation == 3) changed_source.tensors[2].format.params.k_bits = 6;
      if (mutation == 4) changed_source.tensors[2].part = fm::PartRole::kData;
      if (mutation == 5) changed_model.stages[0].folds.push_back("norm.plus_one");
      EXPECT_FALSE(rt::CheckComponentAdmission(changed_model, changed_source, changed_origins))
          << mutation;
    }
  }
}

TEST(ComponentAdmissionDenseCanaryTest,
     LlamaAndMistralNormalizeAndAdmitAcrossGgufSafetensorsAndExl3) {
  for (std::string_view arch : {"llama", "mistral"}) {
    SCOPED_TRACE(arch);
    EXPECT_TRUE(rt::OnComponentEngine(arch));
    auto checkpoint = llmp::test_support::Qwen2GgufCheckpoint(/*untied=*/false);
    checkpoint.architecture = std::string(arch);
    checkpoint.metadata.clear();
    checkpoint.metadata.push_back({"general.architecture", std::string(arch)});
    checkpoint.metadata.push_back({std::string(arch) + ".attention.head_count", std::int64_t{14}});
    checkpoint.metadata.push_back(
        {std::string(arch) + ".attention.head_count_kv", std::int64_t{2}});
    checkpoint.metadata.push_back(
        {std::string(arch) + ".attention.layer_norm_rms_epsilon", static_cast<double>(1e-5F)});
    checkpoint.metadata.push_back({std::string(arch) + ".block_count", std::int64_t{24}});
    checkpoint.metadata.push_back({std::string(arch) + ".context_length", std::int64_t{8192}});
    checkpoint.metadata.push_back({std::string(arch) + ".embedding_length", std::int64_t{896}});
    checkpoint.metadata.push_back({std::string(arch) + ".feed_forward_length", std::int64_t{4864}});
    checkpoint.metadata.push_back({std::string(arch) + ".rope.freq_base", 500000.0});
    if (arch == "llama") {
      checkpoint.metadata.push_back({"llama.rope.dimension_count", std::int64_t{64}});
    }
    std::ranges::sort(checkpoint.metadata, {}, &llmp::import::MetaEntry::key);
    auto bound = llmp::import::NormalizationBound(checkpoint);
    ASSERT_TRUE(bound) << (bound ? "" : bound.error().ToString());
    auto normalized =
        llmp::import::Normalize(std::move(checkpoint), {bound->metadata_bytes, bound->work_units});
    ASSERT_TRUE(normalized) << (normalized ? "" : normalized.error().ToString());
    const auto origins =
        Origins("unpinned-dense-canary.gguf", 512000000ULL,
                "1111111111111111111111111111111111111111111111111111111111111111");
    const auto& model = normalized->spec();
    const auto& source = normalized->canonical();
    const auto admitted = rt::CheckComponentAdmission(model, source, origins);
    ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
    auto policy =
        rt::ComponentAdmissionPolicy(model, source, origins, {.context = 4096, .slots = 2});
    ASSERT_TRUE(policy) << (policy ? "" : policy.error());
    EXPECT_TRUE(policy->prompts_alone);
    EXPECT_FALSE(policy->owners_alone);
    EXPECT_TRUE(policy->text.contains("max_decode_rows 16"));
    EXPECT_TRUE(policy->text.contains("vector_rows 16"));
    EXPECT_TRUE(policy->text.contains("product dense_pair"));
  }
}

// Serving reads v1 artifacts only (D-118, U2): a v0 artifact is refused at
// open time with unsupported-version, while a v1 artifact passes the
// version gate.
TEST(ComponentAdmissionVersionTest, V0ArtifactsAreRefusedWithTheReimport) {
  const std::filesystem::path scratch =
      std::filesystem::path(testing::TempDir()) / "component-admission-version";
  std::error_code ec;
  std::filesystem::remove_all(scratch, ec);
  std::filesystem::create_directories(scratch, ec);

  llmp::test::V1Plan plan;
  plan.architecture = "gemma2";
  plan.groups = {
      llmp::test::V1Group{.kind = llmp::artifact::GroupKind::kTable,
                          .resources = {{.name = "token_embd.weight",
                                         .format = {.encoding = fm::Encoding::kBF16,
                                                    .layout = fm::Layout::kDense,
                                                    .shape = {.ne = {64, 64, 1, 1}, .rank = 2}}}}}};
  auto v0_rendered = llmp::test::Render(plan);
  v0_rendered.manifest["format"] = llmp::test::Json("jitllm-artifact");
  v0_rendered.manifest["format_version"] = llmp::test::Json(std::int64_t{0});
  const auto v0 = llmp::artifact::Artifact::Open(llmp::test::Publish(scratch / "v0", v0_rendered));
  ASSERT_FALSE(v0.has_value());
  EXPECT_EQ(v0.error().rule, llmp::artifact::Rule::kFormat);

  const auto v1 =
      llmp::artifact::Artifact::Open(llmp::test::Publish(scratch / "v1", llmp::test::Render(plan)));
  ASSERT_TRUE(v1.has_value()) << (v1 ? "" : v1.error().ToString());
  EXPECT_EQ(v1->format_version(), 1U);
  EXPECT_TRUE(rt::RequireServedVersion(*v1).has_value());
}

TEST(ComponentAdmissionDrafterTest, StoredExecutionAdmitsUnknownProvenanceAndChecksProtocol) {
  for (std::uint32_t size : {26U, 31U}) {
    auto target = llmp::test_support::GemmaImportTarget(size);
    auto source = llmp::test_support::Gemma4AssistantSource(size);
    auto values =
        fam::Gemma4AssistantValues(size == 26 ? fam::Gemma4Assistant26() : fam::Gemma4Assistant31(),
                                   size == 26 ? fam::Gemma4_26BA4B() : fam::Gemma4_31B());
    ASSERT_TRUE(values);
    auto composed = sp::ComposeDrafter(target.model, *target.mapping,
                                       sp::mappings::Gemma4AssistantMapping(), source, *values);
    ASSERT_TRUE(composed) << composed.error();
    FixturePreprocessing(target.model, target.source);
    for (const auto& fold : composed->stages.back().folds) {
      auto names = llmp::components::FoldTensors(*composed, 1, fold);
      ASSERT_TRUE(names) << names.error();
      source.preprocessing.push_back({.name = fold,
                                      .tensors = std::move(*names),
                                      .source = !std::ranges::contains(source.folds, fold)});
    }
    auto origins = Origins("unknown-drafter.bin", 1, std::string(64, '1'));
    source.architecture = "unregistered-drafter";
    source.container = "safetensors";
    EXPECT_EQ(sp::FindMapping(source.architecture, source.container), nullptr);
    auto check = rt::CheckComponentDrafterAdmission(*composed, source, origins);
    ASSERT_TRUE(check) << check.error();
    auto valid = rt::CheckComposedAdmission(*composed, target.source, source);
    ASSERT_TRUE(valid) << valid.error();

    auto missing = *composed;
    missing.stages.back().drafting.reset();
    EXPECT_FALSE(rt::CheckComponentDrafterAdmission(missing, source, origins));
    auto wrong = *composed;
    wrong.stages.back().drafting = sp::MtpDraft{};
    EXPECT_FALSE(rt::CheckComposedAdmission(wrong, target.source, source));
    wrong.stages.back().drafting = sp::BlockDraft{5, 0};
    EXPECT_FALSE(rt::CheckComposedAdmission(wrong, target.source, source));
    wrong = *composed;
    wrong.stages.back().reads.clear();
    EXPECT_FALSE(rt::CheckComposedAdmission(wrong, target.source, source));

    auto bad_source = source;
    bad_source.container = "invalid-container";
    EXPECT_FALSE(rt::CheckComponentDrafterAdmission(*composed, bad_source, origins));
    origins[0].bytes = llmp::base::Bytes(0);
    EXPECT_FALSE(rt::CheckComponentDrafterAdmission(*composed, source, origins));
  }
}

TEST(ComponentAdmissionDrafterTest, SelectedHeadsAreIndependentOfAnchoredAndBlockProtocols) {
  const auto check_selected = [](sp::ModelSpec model, sp::Source target, sp::Source drafter) {
    auto& stage = model.stages.back();
    auto& output = *stage.heads.output;
    const auto weight = std::ranges::find(output.weights, "weight", &sp::WeightBinding::role);
    ASSERT_NE(weight, output.weights.end());
    const auto* holder = sp::HoldingStage(model, stage, *weight);
    ASSERT_NE(holder, nullptr);
    const auto& table = holder->artifact == 0 ? target.tensors : drafter.tensors;
    const auto tensor = std::ranges::find(table, weight->tensor, &sp::SourceTensor::name);
    ASSERT_NE(tensor, table.end());
    const auto rows = tensor->format.shape.ne[1];
    std::erase_if(output.params, [](const auto& p) { return p.name == "selected"; });
    output.params.push_back({"selected", true});
    output.weights.push_back({.role = "ids", .tensor = "selected.ids"});
    drafter.tensors.push_back({.name = "selected.ids",
                               .format = {.encoding = fm::Encoding::kI32,
                                          .layout = fm::Layout::kDense,
                                          .shape = {.ne = {1, rows, 1, 1}, .rank = 2}}});
    sp::Canonicalize(model);
    ASSERT_TRUE(sp::DrafterOutput(stage));
    const auto valid = rt::CheckComposedAdmission(model, target, drafter);
    ASSERT_TRUE(valid) << valid.error();
    // These protocols retain their full stored head; vocabulary narrowing is
    // an MTP capability, not an incidental rewrite of every selected component.
    EXPECT_FALSE(sp::ConfigureComposedDraftVocabulary(model, 17));
    for (unsigned mutation = 0; mutation < 4; ++mutation) {
      auto bad = model;
      auto bad_source = drafter;
      auto& head = *bad.stages.back().heads.output;
      auto ids = std::ranges::find(head.weights, "ids", &sp::WeightBinding::role);
      ASSERT_NE(ids, head.weights.end());
      if (mutation == 0) ids->stage = bad.stages.front().name;
      if (mutation == 1) ids->experts = true;
      if (mutation == 2) ids->part = fm::PartRole::kCodes;
      if (mutation == 3) head.weights.push_back(*ids);
      sp::Canonicalize(bad);
      EXPECT_TRUE(!sp::DrafterOutput(bad.stages.back()) ||
                  !rt::CheckComposedAdmission(bad, target, bad_source))
          << mutation;
    }
  };
  auto target = llmp::test_support::GemmaImportTarget(26);
  auto drafter = llmp::test_support::Gemma4AssistantSource(26);
  auto values = fam::Gemma4AssistantValues(fam::Gemma4Assistant26(), fam::Gemma4_26BA4B());
  ASSERT_TRUE(values);
  auto anchored = sp::ComposeDrafter(target.model, *target.mapping,
                                     sp::mappings::Gemma4AssistantMapping(), drafter, *values);
  ASSERT_TRUE(anchored) << anchored.error();
  FixturePreprocessing(target.model, target.source);
  check_selected(*anchored, target.source, drafter);

  target = llmp::test_support::Dsv4ImportTarget();
  const auto& profile = fam::DsparkDeepSeekV4Flash();
  drafter = llmp::test_support::dsv4::SourceOf(
      "dflash", llmp::test_support::dsv4::DrafterResources(profile), sp::mappings::DsparkMapping());
  values = fam::DsparkValues(profile);
  ASSERT_TRUE(values);
  auto block = sp::ComposeDrafter(target.model, *target.mapping, sp::mappings::DsparkMapping(),
                                  drafter, *values);
  ASSERT_TRUE(block) << block.error();
  FixturePreprocessing(target.model, target.source);
  const auto baseline = rt::CheckComposedAdmission(*block, target.source, drafter);
  ASSERT_TRUE(baseline) << baseline.error();
  auto& output = *block->stages.back().heads.output;
  // DSpark's Markov head itself cannot select rows. The protocol can use a
  // different valid head, so test that distinction rather than widening Markov.
  auto markov_selected = *block;
  markov_selected.stages.back().heads.output->params.push_back({"selected", true});
  sp::Canonicalize(markov_selected);
  EXPECT_FALSE(rt::CheckComposedAdmission(markov_selected, target.source, drafter));
  std::erase_if(output.params, [](const auto& p) { return p.name == "markov"; });
  std::erase_if(output.weights, [](const auto& w) { return w.role != "weight"; });
  sp::Canonicalize(*block);
  check_selected(*block, target.source, drafter);
}

TEST(ComponentAdmissionDrafterTest, SelectedMapFundingRequiresTheCompleteDenseDescriptor) {
  llmp::artifact::Resource resource;
  resource.name = "map";
  resource.roles = {"map"};
  resource.part = fm::PartRole::kData;
  resource.format = fm::TensorFormat{.encoding = fm::Encoding::kI32,
                                     .layout = fm::Layout::kDense,
                                     .shape = {.ne = {1, 17, 1, 1}, .rank = 2}};
  resource.bytes = llmp::base::Bytes(68);
  EXPECT_EQ(rt::SelectedHeadStorageBytes(resource), 68U);
  for (unsigned mutation = 0; mutation < 7; ++mutation) {
    auto bad = resource;
    switch (mutation) {
      case 0:
        bad.roles.clear();
        break;
      case 1:
        bad.part = fm::PartRole::kCodes;
        break;
      case 2:
        bad.format->encoding = fm::Encoding::kF32;
        break;
      case 3:
        bad.format->shape.ne[0] = 2;
        break;
      case 4:
        bad.format->shape.ne[1] = UINT64_MAX;
        break;
      case 5:
        bad.format->shape.ne[1] = 0;
        break;
      case 6:
        bad.bytes = llmp::base::Bytes(67);
        break;
    }
    EXPECT_FALSE(rt::SelectedHeadStorageBytes(bad)) << mutation;
  }
}

TEST(ComponentAdmissionModelOptTest, EquivalentScaleDeclarationsHaveTheSameStructuralEligibility) {
  auto fixture = llmp::test_support::qwen38::NativeModelOpt();
  FixturePreprocessing(fixture.model, fixture.source);
  auto legacy = fixture.model;
  auto independent = fixture.model;
  for (auto& layer : legacy.stages[0].layers)
    for (auto& component : layer.components)
      if (component.kind == "ffn.moe") {
        std::erase_if(component.params, [](const auto& param) {
          return param.name == "gate_scale" || param.name == "up_scale" ||
                 param.name == "down_scale";
        });
        auto scales = std::ranges::find(component.params, "projection_scales", &sp::Param::name);
        ASSERT_NE(scales, component.params.end());
        scales->value = true;
      }
  for (auto& layer : independent.stages[0].layers)
    for (auto& component : layer.components)
      if (component.kind == "ffn.moe") {
        std::erase_if(component.params, [](const auto& param) {
          return param.name == "projection_scales" || param.name == "gate_scale" ||
                 param.name == "up_scale" || param.name == "down_scale";
        });
        component.params.push_back({"gate_scale", true});
        component.params.push_back({"up_scale", true});
        component.params.push_back({"down_scale", true});
      }
  sp::Canonicalize(legacy);
  sp::Canonicalize(independent);
  ASSERT_NE(legacy.Digest(), independent.Digest());
  const auto origins = llmp::test_support::qwen38::ModelOptOrigins();
  auto owner = std::make_shared<NumericalOwner>();
  auto funder = NumericalFunder(owner);
  const auto bind = [&](const sp::ModelSpec& model, unsigned slots) {
    return rt::AdmittedNumerics(model, fixture.source, origins, model.Digest(),
                                {.context = 262144, .slots = slots}, 0, NumericalDevice(), funder);
  };
  for (const auto slots : {1U, 4U}) {
    auto old = bind(legacy, slots);
    auto current = bind(independent, slots);
    ASSERT_TRUE(old && *old);
    ASSERT_TRUE(current && *current);
    auto old_entries = std::vector((*old)->entries().begin(), (*old)->entries().end());
    auto new_entries = std::vector((*current)->entries().begin(), (*current)->entries().end());
    for (auto& entry : old_entries) entry.semantic = {};
    for (auto& entry : new_entries) entry.semantic = {};
    EXPECT_EQ(old_entries, new_entries);
    bool hc = false, experts = false;
    for (const auto& grant : (*current)->entries()) {
      if (grant.change == llmp::planner::NumericalChange::kBf16HyperConnectionIntermediates) {
        hc = true;
        EXPECT_EQ(grant.max_k, independent.stages[0].stream.width);
        EXPECT_EQ(grant.max_rows,
                  std::uint64_t{4096} * slots * independent.stages[0].stream.streams);
        EXPECT_TRUE(grant.exact_dimensions);
      }
      experts |= grant.change == llmp::planner::NumericalChange::kNvfp4ExpertIntermediates;
    }
    EXPECT_TRUE(hc);
    EXPECT_TRUE(experts);
  }
  // Equal source receipts cannot grant NVFP4 arithmetic to an F32 expert table.
  for (auto& layer : independent.stages[0].layers)
    for (auto& component : layer.components) {
      // Replace the multipart table with one dense data binding, so this
      // control reaches numerical eligibility with a coherent source/spec.
      std::erase_if(component.weights, [&](const auto& binding) {
        if (binding.role != "down_experts" || binding.part != fm::PartRole::kBlockScales)
          return false;
        std::erase_if(fixture.source.tensors,
                      [&](const auto& tensor) { return tensor.name == binding.tensor; });
        return true;
      });
      for (auto& binding : component.weights)
        if (binding.role == "down_experts") {
          auto tensor =
              std::ranges::find(fixture.source.tensors, binding.tensor, &sp::SourceTensor::name);
          ASSERT_NE(tensor, fixture.source.tensors.end());
          tensor->format = llmp::ir::Dense(fm::Encoding::kF32, tensor->format.shape);
          tensor->part = fm::PartRole::kData;
          binding.part = fm::PartRole::kData;
        }
    }
  sp::Canonicalize(independent);
  const auto admitted = rt::CheckComponentAdmission(independent, fixture.source, origins);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
  auto changed_encoding = bind(independent, 1);
  ASSERT_TRUE(changed_encoding) << (changed_encoding ? "" : changed_encoding.error());
  if (*changed_encoding)
    for (const auto& grant : (*changed_encoding)->entries())
      EXPECT_NE(grant.change, llmp::planner::NumericalChange::kNvfp4ExpertIntermediates);

  // A non-positive row_parameter ("used") on a routed FFN does not underflow
  // when computing numerical evidence rows.
  auto zero_used = fixture.model;
  for (auto& stage : zero_used.stages)
    for (auto& layer : stage.layers)
      for (auto& component : layer.components)
        if (component.kind == "ffn.routed")
          for (auto& param : component.params)
            if (param.name == "used") param.value = std::int64_t{0};
  auto zero_used_numerics = bind(zero_used, 1);
  EXPECT_FALSE(zero_used_numerics);
}

TEST(ComponentAdmissionModelOptTest, GdnFoldClosureIncludesEveryPartButNotItsIndependentNorm) {
  auto fixture = llmp::test_support::qwen38::NativeModelOpt();
  FixturePreprocessing(fixture.model, fixture.source);
  const auto origin = llmp::test_support::qwen38::ModelOptOrigins();
  const auto receipt = std::ranges::find(fixture.source.preprocessing, "gdn.value_heads_tiled",
                                         &sp::FoldReceipt::name);
  ASSERT_NE(receipt, fixture.source.preprocessing.end());
  ASSERT_TRUE(rt::CheckComponentAdmission(fixture.model, fixture.source, origin));
  const auto original = receipt->tensors;
  const auto scale = std::ranges::find_if(receipt->tensors, [&](const auto& name) {
    const auto tensor = std::ranges::find(fixture.source.tensors, name, &sp::SourceTensor::name);
    return tensor != fixture.source.tensors.end() && tensor->part == fm::PartRole::kBlockScales;
  });
  ASSERT_NE(scale, receipt->tensors.end());
  receipt->tensors.erase(scale);
  EXPECT_FALSE(rt::CheckComponentAdmission(fixture.model, fixture.source, origin));
  receipt->tensors = original;
  const auto& components = fixture.model.stages.front().layers.front().components;
  const auto gdn = std::ranges::find(components, "attention.gdn", &sp::ComponentRef::kind);
  ASSERT_NE(gdn, components.end());
  const auto norm = std::ranges::find(gdn->weights, "norm", &sp::WeightBinding::role);
  ASSERT_NE(norm, gdn->weights.end());
  EXPECT_FALSE(std::ranges::contains(receipt->tensors, norm->tensor));
  receipt->tensors.push_back(norm->tensor);
  std::ranges::sort(receipt->tensors);
  EXPECT_FALSE(rt::CheckComponentAdmission(fixture.model, fixture.source, origin));
}

TEST(ComponentAdmissionModelOptTest,
     FullNormalizedTargetPreservesNativeProfilePolicyAndClosedSourceAuthority) {
  auto fixture = llmp::test_support::qwen38::NativeModelOpt();
  FixturePreprocessing(fixture.model, fixture.source);
  auto origins = llmp::test_support::qwen38::ModelOptOrigins();
  const auto& model = fixture.model;
  const auto& source = fixture.source;
  const auto admitted = rt::CheckComponentAdmission(model, source, origins);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());

  auto policy =
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262144, .slots = 8});
  ASSERT_TRUE(policy) << (policy ? "" : policy.error());
  // The native mapping's fast form.
  const auto selected_policies = llmp::spec::PolicyRules(llmp::spec::mappings::Qwen38Mapping(),
                                                         {source.architecture, source.container});
  ASSERT_TRUE(selected_policies) << selected_policies.error();
  ASSERT_FALSE(selected_policies->empty());
  EXPECT_EQ(policy->text, selected_policies->front().text);
  EXPECT_FALSE(
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262145, .slots = 1}));
  EXPECT_FALSE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 0, .slots = 1}));
  EXPECT_FALSE(
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262144, .slots = 0}));

  for (unsigned mutation = 0; mutation != 9; ++mutation) {
    auto changed_model = model;
    auto changed_source = source;
    auto changed_origins = origins;
    if (mutation == 0) changed_origins[0].name.clear();
    if (mutation == 1) changed_origins[0].bytes = llmp::base::Bytes(0);
    if (mutation == 2) changed_source.container = "unknown";
    if (mutation == 3) changed_source.folds.clear();
    if (mutation == 4) changed_model.stages[0].folds.clear();
    if (mutation == 5)
      changed_source.tensors[0].format.encoding = static_cast<fm::Encoding>(UINT16_MAX);
    if (mutation == 6) {
      auto it = std::ranges::find_if(changed_source.tensors, [](const auto& t) {
        return t.format.encoding == fm::Encoding::kNvfp4 && t.format.params.tensor_scale;
      });
      ASSERT_NE(it, changed_source.tensors.end());
      it->format.params.tensor_scale = false;
    }
    if (mutation == 7) {
      changed_model.stages[0].heads.output->weights[0].tensor = "missing.output.weight";
    }
    if (mutation == 8) {
      for (auto& t : changed_source.tensors)
        if (t.experts) t.format.shape.ne[2] = 16;
    }
    EXPECT_FALSE(rt::CheckComponentAdmission(changed_model, changed_source, changed_origins))
        << mutation;
  }

  const std::filesystem::path scratch =
      std::filesystem::path(testing::TempDir()) / "component-admission-modelopt";
  std::error_code ec;
  std::filesystem::remove_all(scratch, ec);
  std::filesystem::create_directories(scratch, ec);

  llmp::test::V1Plan plan;
  plan.architecture = "qwen4exp";
  plan.groups = {
      llmp::test::V1Group{.kind = llmp::artifact::GroupKind::kTable,
                          .resources = {{.name = "token_embd.weight",
                                         .format = {.encoding = fm::Encoding::kBF16,
                                                    .layout = fm::Layout::kDense,
                                                    .shape = {.ne = {64, 64, 1, 1}, .rank = 2}}}}}};
  auto rendered = llmp::test::Render(plan);
  llmp::test::Json::Array source_list;
  for (const auto& o : origins) {
    source_list.emplace_back(
        llmp::test::Json::Object{{"bytes", static_cast<std::int64_t>(o.bytes.value())},
                                 {"name", o.name},
                                 {"sha256", llmp::base::ToHex(o.sha256)}});
  }
  rendered.manifest["source"] = llmp::test::Json(std::move(source_list));
  auto art = llmp::artifact::Artifact::Open(llmp::test::Publish(scratch / "artifact", rendered));
  ASSERT_TRUE(art.has_value()) << (art ? "" : art.error().ToString());
  EXPECT_FALSE(rt::CheckComponentArtifactAdmission(*art));
  EXPECT_FALSE(rt::ReadComponentAdmission(*art));
}

TEST(ComponentAdmissionMlxTest,
     FullNormalizedTargetPreservesNativeProfilePolicyAndClosedSourceAuthority) {
  auto fixture = llmp::test_support::qwen38::NativeMlx();
  FixturePreprocessing(fixture.model, fixture.source);
  auto origins = llmp::test_support::qwen38::MlxOrigins();
  const auto& model = fixture.model;
  const auto& source = fixture.source;
  const auto admitted = rt::CheckComponentAdmission(model, source, origins);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());

  auto policy =
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262144, .slots = 8});
  ASSERT_TRUE(policy) << (policy ? "" : policy.error());
  EXPECT_TRUE(policy->text.empty());
  EXPECT_FALSE(
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262145, .slots = 1}));
  EXPECT_FALSE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 0, .slots = 1}));
  EXPECT_FALSE(
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262144, .slots = 0}));

  for (unsigned mutation = 0; mutation != 12; ++mutation) {
    auto changed_model = model;
    auto changed_source = source;
    auto changed_origins = origins;
    if (mutation == 0) changed_origins[0].name.clear();
    if (mutation == 1) changed_origins[0].bytes = llmp::base::Bytes(0);
    if (mutation == 2) changed_source.container = "unknown";
    if (mutation == 3) changed_source.folds.clear();
    if (mutation == 4) changed_model.stages[0].folds.clear();
    if (mutation == 5)
      changed_source.tensors[0].format.encoding = static_cast<fm::Encoding>(UINT16_MAX);
    if (mutation == 6) {
      auto it = std::ranges::find_if(changed_source.tensors, [](const auto& t) {
        return t.format.encoding == fm::Encoding::kAffine;
      });
      ASSERT_NE(it, changed_source.tensors.end());
      it->format.params.bits = 0;
    }
    if (mutation == 7) {
      changed_model.stages[0].heads.output->weights[0].tensor = "missing.output.weight";
    }
    if (mutation == 8) {
      auto it = std::ranges::find_if(changed_source.tensors, [](const auto& t) {
        return t.format.encoding == fm::Encoding::kAffine;
      });
      ASSERT_NE(it, changed_source.tensors.end());
      it->format.params.zero_point = true;
    }
    if (mutation == 9) {
      for (auto& t : changed_source.tensors)
        if (t.experts) t.format.shape.ne[2] = 16;
    }
    if (mutation == 10 || mutation == 11) {
      for (auto& t : changed_source.tensors) {
        if (t.format.encoding != fm::Encoding::kAffine) continue;
        if (mutation == 10) t.format.params.group = 0;
        if (mutation == 11) t.format.params.scale = static_cast<fm::AffineScale>(UINT8_MAX);
      }
    }
    EXPECT_FALSE(rt::CheckComponentAdmission(changed_model, changed_source, changed_origins))
        << mutation;
  }

  const std::filesystem::path scratch =
      std::filesystem::path(testing::TempDir()) / "component-admission-mlx";
  std::error_code ec;
  std::filesystem::remove_all(scratch, ec);
  std::filesystem::create_directories(scratch, ec);

  llmp::test::V1Plan plan;
  plan.architecture = "qwen4exp";
  plan.groups = {
      llmp::test::V1Group{.kind = llmp::artifact::GroupKind::kTable,
                          .resources = {{.name = "token_embd.weight",
                                         .format = {.encoding = fm::Encoding::kBF16,
                                                    .layout = fm::Layout::kDense,
                                                    .shape = {.ne = {64, 64, 1, 1}, .rank = 2}}}}}};
  auto rendered = llmp::test::Render(plan);
  llmp::test::Json::Array source_list;
  for (const auto& o : origins) {
    source_list.emplace_back(
        llmp::test::Json::Object{{"bytes", static_cast<std::int64_t>(o.bytes.value())},
                                 {"name", o.name},
                                 {"sha256", llmp::base::ToHex(o.sha256)}});
  }
  rendered.manifest["source"] = llmp::test::Json(std::move(source_list));
  auto art = llmp::artifact::Artifact::Open(llmp::test::Publish(scratch / "artifact", rendered));
  ASSERT_TRUE(art.has_value()) << (art ? "" : art.error().ToString());
  EXPECT_FALSE(rt::CheckComponentArtifactAdmission(*art));
  EXPECT_FALSE(rt::ReadComponentAdmission(*art));
}

}  // namespace

// The scale-atom experts' slot (the block-scaled grouped GEMM's fixed
// offsets): the stacked gate-up table's codes and scales, then down's, back
// to back from each expert group's start.
TEST(ComponentAdmissionScaleAtomTest, SlotsHoldTheStackedTableFirstBackToBack) {
  const auto atoms = [](std::uint64_t k, std::uint64_t n, std::uint8_t stacked) {
    return fm::TensorFormat{.encoding = fm::Encoding::kNvfp4,
                            .layout = fm::Layout::kScaleAtoms,
                            .stacked = stacked,
                            .shape = {.ne = {k, n, 1, 1}, .rank = 2}};
  };
  const std::vector<llmp::test::V1Array> gate_up{{.name = "blk.0.ffn_gate_up_exps.weight",
                                                  .format = atoms(64, 256, 2),
                                                  .part = fm::PartRole::kCodes},
                                                 {.name = "blk.0.ffn_gate_up_exps.weight_scale",
                                                  .format = atoms(64, 256, 2),
                                                  .part = fm::PartRole::kBlockScales}};
  const std::vector<llmp::test::V1Array> down{{.name = "blk.0.ffn_down_exps.weight",
                                               .format = atoms(128, 128, 1),
                                               .part = fm::PartRole::kCodes},
                                              {.name = "blk.0.ffn_down_exps.weight_scale",
                                               .format = atoms(128, 128, 1),
                                               .part = fm::PartRole::kBlockScales}};
  const std::filesystem::path scratch =
      std::filesystem::path(testing::TempDir()) / "component-admission-scale-atoms";
  std::error_code ec;
  std::filesystem::remove_all(scratch, ec);
  const auto open = [&](std::string_view name, std::vector<llmp::test::V1Array> arrays) {
    llmp::test::V1Plan plan;
    plan.architecture = "qwen4exp";
    plan.expert_count = 2;
    plan.groups = {llmp::test::V1Group{
        .kind = llmp::artifact::GroupKind::kExpert, .layer = 0, .arrays = std::move(arrays)}};
    return llmp::artifact::Artifact::Open(
        llmp::test::Publish(scratch / name, llmp::test::Render(plan)));
  };
  auto first = gate_up;
  first.insert(first.end(), down.begin(), down.end());
  auto good = open("good", first);
  ASSERT_TRUE(good) << good.error().ToString();
  EXPECT_TRUE(rt::CheckScaleAtomSlots(*good));
  auto last = down;
  last.insert(last.end(), gate_up.begin(), gate_up.end());
  auto swapped = open("swapped", last);
  ASSERT_TRUE(swapped) << swapped.error().ToString();
  EXPECT_FALSE(rt::CheckScaleAtomSlots(*swapped));
  auto gap = first;
  gap[2].offset = 8192 + 1024 + 256;  // down's codes past a gap
  gap[3].offset = 8192 + 1024 + 256 + 8192;
  auto spaced = open("spaced", gap);
  ASSERT_TRUE(spaced) << spaced.error().ToString();
  EXPECT_FALSE(rt::CheckScaleAtomSlots(*spaced));
}

// Stored hash routes (DeepSeek V4's token-hash layers): each side.hash
// table a spec binds is read through bounded direct reads before any routed
// work, and refused unless it is I32 [used, tokens] of consistent bytes
// whose every entry names one of the input's experts.
TEST(ComponentAdmissionHashTableTest, StoredTablesAreReadAndRefusedByShapeBytesAndEntries) {
  constexpr std::int64_t kExperts = 8;
  constexpr std::uint64_t kUsed = 6;
  constexpr std::uint64_t kTokens = 1000;  // 24,000 bytes: several 4 KiB windows
  constexpr std::string_view kTable = "blk.0.ffn_gate_tid2eid.weight";
  // Direct reads need a file system that takes O_DIRECT: the build tree's.
  const char* base = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  const std::filesystem::path scratch =
      (base != nullptr ? std::filesystem::path(base)
                       : std::filesystem::current_path() / "scratch") /
      "component-admission-hash-tables";
  std::error_code ec;
  std::filesystem::remove_all(scratch, ec);
  // A spec whose one stage binds the table as a side.hash input's weight.
  sp::ModelSpec model;
  model.stages.resize(1);
  model.stages[0].side.push_back(
      {.name = "hash",
       .source = {.kind = "side.hash",
                  .params = {{.name = "experts", .value = kExperts},
                             {.name = "used", .value = static_cast<std::int64_t>(kUsed)}},
                  .weights = {{.role = "table", .tensor = std::string(kTable)}}},
       .layers = {0}});
  // A v1 artifact holding a norm and then the table, its entries written.
  const auto publish = [&](std::string_view name, fm::Encoding encoding, std::uint64_t rows,
                           auto entry) -> std::expected<llmp::artifact::Artifact, std::string> {
    llmp::test::V1Plan plan;
    plan.architecture = "deepseek4";
    plan.groups = {llmp::test::V1Group{
        .kind = llmp::artifact::GroupKind::kLayer,
        .layer = 0,
        .resources = {{.name = "blk.0.attn_norm.weight",
                       .format = {.encoding = fm::Encoding::kF32,
                                  .layout = fm::Layout::kDense,
                                  .shape = {.ne = {64, 1, 1, 1}, .rank = 1}}},
                      {.name = std::string(kTable),
                       .format = {.encoding = encoding,
                                  .layout = fm::Layout::kDense,
                                  .shape = {.ne = {rows, kTokens, 1, 1}, .rank = 2}}}}}};
    auto rendered = llmp::test::Render(plan);
    // Where the table's bytes lie in its shard file (the deterministic fill
    // is replaced with the entries).
    auto probe =
        llmp::artifact::Artifact::Open(llmp::test::Publish(scratch / name / "probe", rendered));
    if (!probe) return std::unexpected(probe.error().ToString());
    const auto index = probe->FindResource(kTable);
    if (!index) return std::unexpected("the table is missing");
    const auto place = probe->ResourcePlacement(*index);
    if (!place) return std::unexpected(place.error().ToString());
    const auto& group = probe->groups()[place->group];
    const auto begin = probe->shards()[group.shard].data_offset.value() + group.offset.value() +
                       place->offset.value();
    std::string& file = rendered.shards.at(group.shard);
    for (std::uint64_t i = 0; i * sizeof(std::int32_t) < place->bytes.value(); ++i) {
      const std::int32_t value = entry(i);
      std::memcpy(file.data() + begin + (i * sizeof(value)), &value, sizeof(value));
    }
    auto opened =
        llmp::artifact::Artifact::Open(llmp::test::Publish(scratch / name / "store", rendered));
    if (!opened) return std::unexpected(opened.error().ToString());
    return std::move(*opened);
  };
  const auto valid = [](std::uint64_t i) {
    return static_cast<std::int32_t>((i * 5U) % static_cast<std::uint64_t>(kExperts));
  };
  const auto check = [&](std::string_view name, fm::Encoding encoding, std::uint64_t rows,
                         auto entry) -> std::expected<void, std::string> {
    auto artifact = publish(name, encoding, rows, entry);
    if (!artifact) return std::unexpected("publishing: " + artifact.error());
    return rt::CheckStoredTables(*artifact, model);
  };

  // Every expert in [0, experts): read and accepted.
  const auto good = check("valid", fm::Encoding::kI32, kUsed, valid);
  ASSERT_TRUE(good) << good.error();
  // One entry below zero, one entry naming the expert count: refused.
  const auto negative = check("negative", fm::Encoding::kI32, kUsed,
                              [&](std::uint64_t i) { return i == 4321 ? -1 : valid(i); });
  ASSERT_FALSE(negative);
  EXPECT_NE(negative.error().find("names an invalid expert"), std::string::npos)
      << negative.error();
  const auto past = check("past", fm::Encoding::kI32, kUsed, [&](std::uint64_t i) {
    return i + 1 == kUsed * kTokens ? static_cast<std::int32_t>(kExperts) : valid(i);
  });
  ASSERT_FALSE(past);
  EXPECT_NE(past.error().find("names an invalid expert"), std::string::npos) << past.error();
  // Another encoding of the same shape, its bytes I32's (F32) or half of
  // them (a short table, I16): never read as I32.
  for (const auto encoding : {fm::Encoding::kF32, fm::Encoding::kI16}) {
    const auto other = check(encoding == fm::Encoding::kF32 ? "f32" : "short", encoding, kUsed,
                             [](std::uint64_t) { return 0; });
    ASSERT_FALSE(other);
    EXPECT_NE(other.error().find("not dense I32"), std::string::npos) << other.error();
  }
  // Another shape: rows a token other than the input's `used`.
  const auto shape = check("shape", fm::Encoding::kI32, kUsed + 1, valid);
  ASSERT_FALSE(shape);
  EXPECT_NE(shape.error().find("another shape"), std::string::npos) << shape.error();
  // A spec naming a table the artifact lacks, or no expert count.
  auto artifact = publish("valid-again", fm::Encoding::kI32, kUsed, valid);
  ASSERT_TRUE(artifact) << artifact.error();
  auto missing = model;
  missing.stages[0].side[0].source.weights[0].tensor = "blk.1.ffn_gate_tid2eid.weight";
  EXPECT_FALSE(rt::CheckStoredTables(*artifact, missing));
  auto uncounted = model;
  uncounted.stages[0].side[0].source.params.erase(
      uncounted.stages[0].side[0].source.params.begin());
  EXPECT_FALSE(rt::CheckStoredTables(*artifact, uncounted));
  std::filesystem::remove_all(scratch, ec);
}
