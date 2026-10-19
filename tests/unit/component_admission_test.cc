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
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "artifact_writer.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "dsv4_gguf_normalization_fixture.h"
#include "expected_error.h"
#include "families/gemma.h"
#include "import/normalize.h"
#include "kernels/common/lower.h"
#include "qwen2_exl3_normalization_fixture.h"
#include "qwen2_gguf_normalization_fixture.h"
#include "qwen38_fixture.h"
#include "qwen38_gguf_normalization_fixture.h"
#include "runtime/calibration.h"
#include "runtime/component_admission_internal.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec/mappings/qwen38.h"
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
                     .shape = {.ne = {128, 256, experts, 1}, .rank = 3},
                     .stacked = stacked},
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

TEST_F(ComponentAdmissionTest, NumericalFactsAuthenticateBeforePaymentAndRetainExactRoleSites) {
  const auto* pinned = rt::FindApproval(source.architecture, origins);
  ASSERT_NE(pinned, nullptr);
  ASSERT_TRUE(rt::internal::CheckInventory(source, *pinned));
  constexpr std::array anchors{llmp::ir::Primitive(llmp::ir::Op::kMatMul)};
  constexpr std::array<std::uint64_t, 2> rows{1, 4};
  const std::array domains{rt::NumericalEvidenceDomain{
      .change = llmp::planner::NumericalChange::kQ8VectorReduction,
      .evidence = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      .anchors = anchors,
      .max_context = 8192,
      .max_slots = 4,
      .max_rows = 4,
      .rows = rows}};
  auto approval = *pinned;
  approval.numerical = domains;
  auto owner = std::make_shared<NumericalOwner>();
  auto funder = NumericalFunder(owner);
  const auto semantic = model.Digest();
  const sp::Envelope envelope{.context = 8192, .slots = 4};
  auto bind = [&] {
    return rt::AdmittedNumerics(model, source, origins, &approval, semantic, envelope, 0, funder);
  };
  auto wrong = semantic;
  wrong[0] ^= 1;
  EXPECT_FALSE(rt::AdmittedNumerics(model, source, origins, &approval, wrong, envelope, 0, funder));
  EXPECT_EQ(owner->charges, 0U);
  const auto receipt = origins[0].sha256;
  origins[0].sha256[0] ^= 1;
  EXPECT_FALSE(bind());
  origins[0].sha256 = receipt;
  EXPECT_EQ(owner->charges, 0U);
  const auto encoding = source.tensors[0].format.encoding;
  source.tensors[0].format.encoding =
      encoding == fm::Encoding::kBF16 ? fm::Encoding::kF32 : fm::Encoding::kBF16;
  EXPECT_FALSE(bind());
  source.tensors[0].format.encoding = encoding;
  EXPECT_EQ(owner->charges, 0U);
  auto outside = rt::AdmittedNumerics(model, source, origins, &approval, semantic,
                                      {.context = 8193, .slots = 4}, 0, funder);
  ASSERT_TRUE(outside);
  EXPECT_FALSE(*outside);
  EXPECT_EQ(owner->charges, 0U);
  EXPECT_FALSE(rt::AdmittedNumerics(model, source, origins, &approval, semantic, envelope, 0, {}));
  auto incomplete = std::make_shared<llmp::kernels::ConstructionFunder>();
  incomplete->charge = [owner](std::uint64_t bytes) { return owner->Charge(bytes); };
  EXPECT_FALSE(
      rt::AdmittedNumerics(model, source, origins, &approval, semantic, envelope, 0, incomplete));
  EXPECT_EQ(owner->charges, 0U);
  owner->available = 0;
  EXPECT_FALSE(bind());
  EXPECT_EQ(owner->used, 0U);
  EXPECT_EQ(owner->available, 0U);
  EXPECT_EQ(owner->releases, 0U);
  owner->available = UINT64_MAX;
  auto facts = bind();
  ASSERT_TRUE(facts) << (facts ? "" : facts.error());
  ASSERT_TRUE(*facts);
  ASSERT_FALSE((*facts)->entries().empty());
  const auto used = owner->used;
  EXPECT_GT(used, (*facts)->host_bytes());
  for (const auto& entry : (*facts)->entries()) {
    EXPECT_EQ(entry.min_rows, entry.max_rows);
    EXPECT_TRUE(entry.min_rows == 1 || entry.min_rows == 4);
    EXPECT_TRUE(entry.exact_dimensions);
    EXPECT_TRUE(entry.incumbent_recipe);
    EXPECT_NE(entry.weight_role, UINT16_MAX);
    EXPECT_GT(entry.max_k, 0U);
    EXPECT_GT(entry.max_n, 0U);
    EXPECT_EQ(entry.semantic, semantic);
  }
  const std::weak_ptr<const llmp::kernels::ConstructionFunder> payer = funder;
  funder.reset();
  EXPECT_FALSE(payer.expired());
  auto borrower = *facts;
  facts->reset();
  EXPECT_EQ(owner->used, used);
  borrower.reset();
  EXPECT_TRUE(payer.expired());
  EXPECT_EQ(owner->used, 0U);
  EXPECT_EQ(owner->releases, 1U);
  auto invalid = domains;
  invalid[0].evidence = "not a digest";
  approval.numerical = invalid;
  funder = NumericalFunder(owner);
  const auto attempts = owner->charges;
  EXPECT_FALSE(bind());
  EXPECT_EQ(owner->charges, attempts);
}

TEST_F(ComponentAdmissionTest, Iq2IncumbentDomainProjectsOnlyOneActualRowAndExactBoundRoles) {
  const auto iq2_origins =
      Origins("DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf",
              86720111488ULL, "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0");
  const auto* iq2 = rt::FindApproval("deepseek4", iq2_origins);
  ASSERT_NE(iq2, nullptr);
  ASSERT_EQ(iq2->numerical.size(), 1U);
  const auto& domain = iq2->numerical.front();
  EXPECT_EQ(domain.change, llmp::planner::NumericalChange::kQ8VectorReduction);
  EXPECT_TRUE(domain.incumbent_recipe);
  ASSERT_EQ(domain.rows.size(), 1U);
  EXPECT_EQ(domain.rows.front(), 1U);
  EXPECT_EQ(domain.max_rows, 1U);
  EXPECT_EQ(domain.max_context, 262144U);
  EXPECT_EQ(domain.max_slots, 4U);
  auto changed_origins = iq2_origins;
  changed_origins.front().sha256.front() ^= 1;
  EXPECT_EQ(rt::FindApproval("deepseek4", changed_origins), nullptr);

  // Exercise the same generic projector on this fixture's authenticated
  // source/spec and actual roles. This is not a synthetic IQ2 inventory.
  const auto* pinned = rt::FindApproval(source.architecture, origins);
  ASSERT_NE(pinned, nullptr);
  auto approval = *pinned;
  approval.numerical = iq2->numerical;
  auto owner = std::make_shared<NumericalOwner>();
  const auto funder = NumericalFunder(owner);
  const auto semantic = model.Digest();
  auto facts = rt::AdmittedNumerics(model, source, origins, &approval, semantic,
                                    {.context = 8192, .slots = 4}, 0, funder);
  ASSERT_TRUE(facts) << (facts ? "" : facts.error());
  ASSERT_TRUE(*facts);
  ASSERT_FALSE((*facts)->entries().empty());
  for (const auto& grant : (*facts)->entries()) {
    EXPECT_EQ(grant.min_rows, 1U);
    EXPECT_EQ(grant.max_rows, 1U);  // no multiplication by the configured slots
    EXPECT_TRUE(grant.exact_dimensions);
    EXPECT_TRUE(grant.incumbent_recipe);
    EXPECT_NE(grant.weight_role, UINT16_MAX);
    EXPECT_GT(grant.max_k, 0U);
    if (grant.anchor == llmp::ir::Primitive(llmp::ir::Op::kGlu))
      EXPECT_EQ(grant.max_n, 0U);
    else
      EXPECT_GT(grant.max_n, 0U);
    EXPECT_EQ(grant.semantic, semantic);
  }
  const auto charges = owner->charges;
  source.tensors.front().format.encoding =
      source.tensors.front().format.encoding == fm::Encoding::kF32 ? fm::Encoding::kF16
                                                                   : fm::Encoding::kF32;
  EXPECT_FALSE(rt::AdmittedNumerics(model, source, origins, &approval, semantic,
                                    {.context = 8192, .slots = 4}, 0, funder));
  EXPECT_EQ(owner->charges, charges);  // altered format refuses before payment
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

TEST_F(ComponentAdmissionTest,
       UnpinnedOriginAdmitsValidSpecWhileMalformedOriginOrWrongFamilyFails) {
  origins[0].sha256[0] ^= 1;
  EXPECT_TRUE(rt::CheckComponentAdmission(model, source, origins));
  origins = Origins();
  origins[0].bytes = llmp::base::Bytes(origins[0].bytes.value() - 1);
  EXPECT_TRUE(rt::CheckComponentAdmission(model, source, origins));
  origins[0].bytes = llmp::base::Bytes(0);
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  origins = Origins();
  source.architecture = "gemma3";
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
}

TEST_F(ComponentAdmissionTest, RejectsContextArithmeticFoldsAndUntiedHeadChanges) {
  const auto original = model;
  ++model.stages.front().context;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  model = original;
  model.stages.front().heads.softcap = 29.0;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  model = original;
  model.stages.front().folds.clear();
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  model = original;
  model.stages.front().heads.output->weights.front().tensor = "output.weight";
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  model = original;
  source.folds = {"norm.plus_one"};
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
}

TEST_F(ComponentAdmissionTest, RejectsNativeFormatGeometryAndRoleChanges) {
  const auto original = source;
  auto& format = source.tensors.front().format;
  format.encoding = fm::Encoding::kQ4_0;
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
  EXPECT_FALSE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8192, .slots = 2}));
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
        ++changed_model.stages[0].context;
        break;
      case 3:
        changed_model.stages[0].folds.clear();
        break;
      case 4:
        changed_model.stages[0].heads.output->weights[0].tensor =
            changed_model.stages[0].embedding->weights[0].tensor;
        break;
      case 5:
        changed_source.tensors[0].format.encoding = fm::Encoding::kF16;
        break;
      case 6:
        ASSERT_FALSE(changed_model.stages[0].side.empty());
        for (auto& param : changed_model.stages[0].side[0].source.params)
          if (param.name == "multipliers") std::get<std::vector<std::int64_t>>(param.value)[0] += 2;
        break;
    }
    EXPECT_FALSE(rt::CheckComponentAdmission(changed_model, changed_source, changed_origins))
        << mutation;
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
  matrix->format.encoding =
      matrix->format.encoding == fm::Encoding::kQ8_0 ? fm::Encoding::kQ4_0 : fm::Encoding::kQ8_0;
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
  ++model.stages.front().context;
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
  model.stages.front().layers.pop_back();
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
    if (mutation == 2) changed_source.tensors[0].format.encoding = fm::Encoding::kF16;
    if (mutation == 3) changed_model.stages[0].side.pop_back();
    if (mutation == 4)
      changed_model.stages[0].heads.output->weights[0].tensor =
          changed_model.stages[0].embedding->weights[0].tensor;
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
  EXPECT_FALSE(rt::OnComponentEngine("qwen2"));
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
    if (mutation == 2) changed_source.tensors[0].format.encoding = fm::Encoding::kBF16;
    if (mutation == 3)
      changed_model.stages[0].heads.output->weights[0].tensor =
          changed_model.stages[0].embedding->weights[0].tensor;
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
    std::array<llmp::artifact::Source, 2> origins{
        Origins("config.json", rate_4_5 ? 923ULL : 996ULL,
                rate_4_5 ? "dcd945f38ba6e7835c8c281a45c03d9a20759d03afe5761d09def8c62aa11f3c"
                         : "966657b7e8c331ea16127a868eb6d6ff16ad0b16a5ee39d26279e12de0a3205c")[0],
        Origins("model.safetensors", rate_4_5 ? 611257914ULL : 588951098ULL,
                rate_4_5 ? "94c45f9838a56fbf23f43d0a04c47e187d840dd97f17b4a027a69a20339762bc"
                         : "66b61bd1ca3ebb0fc1cea29148eda8bdaca45d232c6a7e319c6d2835b65c066b")[0]};
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
      if (mutation == 2) changed_source.container = "gguf";
      if (mutation == 3) changed_source.tensors[2].format.params.k_bits = 6;
      if (mutation == 4) changed_source.tensors[2].part = fm::PartRole::kData;
      if (mutation == 5) changed_model.stages[0].folds.push_back("norm.plus_one");
      EXPECT_FALSE(rt::CheckComponentAdmission(changed_model, changed_source, changed_origins))
          << mutation;
    }
  }
}

// Serving reads v1 artifacts only (D-118): a v0 artifact (the reader's
// corpus golden) is refused, as a target and as a drafter, with what to do.
TEST(ComponentAdmissionVersionTest, V0ArtifactsAreRefusedWithTheReimport) {
  const char* configured =
      std::getenv("LLMP_TEST_ARTIFACT_CORPUS");  // NOLINT(concurrency-mt-unsafe)
  const std::filesystem::path corpus =
      configured == nullptr ? std::filesystem::path("artifact-corpus") : configured;
  std::error_code error;
  std::filesystem::path root;
  for (const auto& entry : std::filesystem::directory_iterator(corpus / "golden" / "tiny", error)) {
    root = entry.path();
  }
  ASSERT_FALSE(root.empty()) << "no corpus at " << corpus;
  const auto v0 = llmp::artifact::Artifact::Open(root);
  ASSERT_TRUE(v0.has_value()) << (v0 ? "" : v0.error().ToString());
  ASSERT_EQ(v0->format_version(), 0U);
  const auto reimport = [](const std::string& message) {
    return message.find("re-import its source checkpoint with `llmp import") != std::string::npos;
  };
  const auto served = rt::RequireServedVersion(*v0);
  ASSERT_FALSE(served.has_value());
  EXPECT_TRUE(reimport(served.error())) << served.error();
  const auto admitted = rt::ReadComponentAdmission(*v0);
  ASSERT_FALSE(admitted.has_value());
  EXPECT_NE(admitted.error().find("re-import"), std::string::npos) << admitted.error();
  const auto drafted = rt::ComponentDrafterValues(*v0, *v0);
  ASSERT_FALSE(drafted.has_value());
  EXPECT_TRUE(reimport(drafted.error())) << drafted.error();
}

TEST(ComponentAdmissionDrafterTest, CheckComponentDrafterAdmissionApprovesGemma4Assistants) {
  for (std::uint32_t size : {26U, 31U}) {
    auto source = llmp::test_support::Gemma4AssistantSource(size);
    auto origins =
        size == 26 ? Origins("mtp-gemma-4-26B-A4B-it.gguf", 461766816ULL,
                             "6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b")
                   : Origins("mtp-gemma-4-31B-it.gguf", 514687104ULL,
                             "5ae8b0117bed601e8924c6305bd5b0585de361d51f0e77091bcb4252cf1f27de");
    auto check = rt::CheckComponentDrafterAdmission(source, origins);
    EXPECT_TRUE(check.has_value()) << (check ? "" : check.error());

    // Mutations must fail admission
    auto bad_source = source;
    bad_source.container = "safetensors";
    EXPECT_FALSE(rt::CheckComponentDrafterAdmission(bad_source, origins));

    bad_source = source;
    bad_source.folds = {"some_fold"};
    EXPECT_FALSE(rt::CheckComponentDrafterAdmission(bad_source, origins));

    auto unpinned_origins = origins;
    unpinned_origins[0].sha256[0] ^= 1;
    EXPECT_TRUE(rt::CheckComponentDrafterAdmission(source, unpinned_origins));
    unpinned_origins[0].bytes = llmp::base::Bytes(0);
    EXPECT_FALSE(rt::CheckComponentDrafterAdmission(source, unpinned_origins));
  }
}

TEST(ComponentAdmissionDrafterTest, ComponentDrafterValuesApprovesV1DraftersAndRejectsMismatches) {
  const std::filesystem::path scratch =
      std::filesystem::path(testing::TempDir()) / "component-drafter-values-v1";
  std::error_code ec;
  std::filesystem::remove_all(scratch, ec);
  std::filesystem::create_directories(scratch, ec);

  const auto make_v1 = [&](const std::filesystem::path& path, std::string arch,
                           std::string source_name, std::uint64_t bytes, std::string sha256) {
    llmp::test::V1Plan p;
    p.architecture = std::move(arch);
    p.groups = {llmp::test::V1Group{
        .kind = llmp::artifact::GroupKind::kTable,
        .resources = {{.name = "token_embd.weight",
                       .format = {.encoding = fm::Encoding::kBF16,
                                  .layout = fm::Layout::kDense,
                                  .shape = {.ne = {64, 64, 1, 1}, .rank = 2}}}}}};
    auto rendered = llmp::test::Render(p);
    llmp::test::Json::Array source_list;
    source_list.emplace_back(llmp::test::Json::Object{{"bytes", static_cast<std::int64_t>(bytes)},
                                                      {"name", std::move(source_name)},
                                                      {"sha256", std::move(sha256)}});
    rendered.manifest["source"] = llmp::test::Json(std::move(source_list));
    return llmp::artifact::Artifact::Open(llmp::test::Publish(path, rendered));
  };

  const auto g26_target =
      make_v1(scratch / "g26", "gemma4", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", 16947541728ULL,
              "f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f");
  ASSERT_TRUE(g26_target.has_value()) << (g26_target ? "" : g26_target.error().ToString());

  const auto g31_target =
      make_v1(scratch / "g31", "gemma4", "gemma-4-31B-it-UD-Q4_K_XL.gguf", 18822970304ULL,
              "9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575");
  ASSERT_TRUE(g31_target.has_value()) << (g31_target ? "" : g31_target.error().ToString());

  const auto g26_asst =
      make_v1(scratch / "a26", "gemma4-assistant", "mtp-gemma-4-26B-A4B-it.gguf", 461766816ULL,
              "6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b");
  ASSERT_TRUE(g26_asst.has_value()) << (g26_asst ? "" : g26_asst.error().ToString());

  const auto g31_asst =
      make_v1(scratch / "a31", "gemma4-assistant", "mtp-gemma-4-31B-it.gguf", 514687104ULL,
              "5ae8b0117bed601e8924c6305bd5b0585de361d51f0e77091bcb4252cf1f27de");
  ASSERT_TRUE(g31_asst.has_value()) << (g31_asst ? "" : g31_asst.error().ToString());

  const auto dflash =
      make_v1(scratch / "dflash", "dflash", "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf",
              10896057440ULL, "2c7ac54b0b64a99df1f139a9f1371a00198265e1d6a614b77597d20a655a4249");
  ASSERT_TRUE(dflash.has_value()) << (dflash ? "" : dflash.error().ToString());

  // Matched approved pairs pass their pairing; their values are the stored
  // drafter's kept metadata, which these synthetic artifacts lack.
  auto g26_pair = rt::ComponentDrafterValues(*g26_target, *g26_asst);
  ASSERT_FALSE(g26_pair.has_value());
  EXPECT_NE(g26_pair.error().find("keeps no GGUF metadata"), std::string::npos) << g26_pair.error();

  auto g31_pair = rt::ComponentDrafterValues(*g31_target, *g31_asst);
  ASSERT_FALSE(g31_pair.has_value());
  EXPECT_NE(g31_pair.error().find("keeps no GGUF metadata"), std::string::npos) << g31_pair.error();

  // Pairing qualifications do not replace structural composition. These
  // deliberately incomplete artifacts still fail at their missing metadata.
  for (const auto& pair :
       {std::pair{&*g26_target, &*g31_asst}, std::pair{&*g31_target, &*g26_asst}}) {
    auto values = rt::ComponentDrafterValues(*pair.first, *pair.second);
    ASSERT_FALSE(values);
    EXPECT_NE(values.error().find("keeps no GGUF metadata"), std::string::npos);
  }
  auto arch_mismatch = rt::ComponentDrafterValues(*g26_target, *dflash);
  ASSERT_FALSE(arch_mismatch);
  EXPECT_NE(arch_mismatch.error().find("does not target architecture"), std::string::npos);

  // A pinned drafter's composition: beside its approved target, the one its
  // approval pins (this one is not); beside an unpinned target of its
  // architecture (a re-quant), checked structurally, its own identity
  // pinned by its inventory. Without a drafter stage it is refused.
  const auto requant =
      make_v1(scratch / "g26-requant", "gemma4", "gemma-4-26B-A4B-it-Q8_0.gguf", 27000000000ULL,
              "0000000000000000000000000000000000000000000000000000000000000001");
  ASSERT_TRUE(requant.has_value()) << (requant ? "" : requant.error().ToString());
  ASSERT_EQ(rt::FindApproval("gemma4", requant->sources()), nullptr);
  sp::ModelSpec composed;
  composed.stages.resize(2);
  composed.stages.front().role = sp::StageRole::kTarget;
  composed.stages.back().role = sp::StageRole::kDrafter;
  const auto pinned = rt::CheckComposedAdmission(composed, *g26_target, *g26_asst);
  ASSERT_FALSE(pinned.has_value());
  EXPECT_NE(pinned.error().find("is not the one approved"), std::string::npos) << pinned.error();
  const auto paired = rt::CheckComposedAdmission(composed, *requant, *g26_asst);
  EXPECT_TRUE(paired.has_value()) << (paired ? "" : paired.error());
  sp::ModelSpec alone = composed;
  alone.stages.pop_back();
  EXPECT_FALSE(rt::CheckComposedAdmission(alone, *requant, *g26_asst));
}

TEST(ComponentAdmissionModelOptTest,
     IncumbentHcProjectionUsesActualNormWidthStreamsAndLongContext) {
  auto fixture = llmp::test_support::qwen38::NativeModelOpt();
  auto origins = llmp::test_support::qwen38::ModelOptOrigins();
  const auto* approval = rt::FindApproval(fixture.source.architecture, origins);
  ASSERT_NE(approval, nullptr);
  auto owner = std::make_shared<NumericalOwner>();
  auto funder = NumericalFunder(owner);
  const auto semantic = fixture.model.Digest();
  for (const auto slots : {1U, 4U}) {
    auto facts = rt::AdmittedNumerics(fixture.model, fixture.source, origins, approval, semantic,
                                      {.context = 262144, .slots = slots}, 0, funder);
    ASSERT_TRUE(facts) << (facts ? "" : facts.error());
    ASSERT_TRUE(*facts);
    std::size_t hc = 0, experts = 0;
    const auto& stage = fixture.model.stages.front();
    for (const auto& grant : (*facts)->entries()) {
      if (grant.change == llmp::planner::NumericalChange::kBf16HyperConnectionIntermediates) {
        ++hc;
        EXPECT_EQ(grant.max_k, stage.stream.width);
        EXPECT_EQ(grant.max_n, 0U);
        EXPECT_EQ(grant.max_rows, std::uint64_t{4096} * slots * stage.stream.streams);
        EXPECT_TRUE(grant.exact_dimensions);
        EXPECT_TRUE(grant.incumbent_recipe);
      }
      if (grant.change == llmp::planner::NumericalChange::kNvfp4ExpertIntermediates) {
        ++experts;
        EXPECT_EQ(grant.max_k, stage.stream.width);
        EXPECT_GT(grant.max_rows, std::uint64_t{4096} * slots);
        EXPECT_NE(grant.weight_role, UINT16_MAX);
      }
    }
    EXPECT_GT(hc, 0U);
    EXPECT_GT(experts, 0U);
  }
  EXPECT_EQ(owner->used, 0U);
}

TEST(ComponentAdmissionModelOptTest,
     FullNormalizedTargetPreservesNativeProfilePolicyAndClosedSourceAuthority) {
  auto fixture = llmp::test_support::qwen38::NativeModelOpt();
  auto origins = llmp::test_support::qwen38::ModelOptOrigins();
  const auto& model = fixture.model;
  const auto& source = fixture.source;
  const auto admitted = rt::CheckComponentAdmission(model, source, origins);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());

  auto policy =
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262144, .slots = 8});
  ASSERT_TRUE(policy) << (policy ? "" : policy.error());
  // The native mapping's fast form.
  EXPECT_EQ(policy->text, llmp::spec::mappings::Qwen38NativeMapping().policies.front().text);
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
    if (mutation == 2) changed_source.container = "gguf";
    if (mutation == 3) changed_source.folds.clear();
    if (mutation == 4) changed_model.stages[0].folds.clear();
    if (mutation == 5) changed_source.tensors[0].format.encoding = fm::Encoding::kF16;
    if (mutation == 6) {
      auto it = std::ranges::find_if(changed_source.tensors, [](const auto& t) {
        return t.format.encoding == fm::Encoding::kNvfp4 && t.format.params.tensor_scale;
      });
      ASSERT_NE(it, changed_source.tensors.end());
      it->format.params.tensor_scale = false;
    }
    if (mutation == 7) {
      changed_model.stages[0].heads.output->weights[0].tensor =
          changed_model.stages[0].embedding->weights[0].tensor;
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
    if (mutation == 2) changed_source.container = "safetensors";
    if (mutation == 3) changed_source.folds.clear();
    if (mutation == 4) changed_model.stages[0].folds.clear();
    if (mutation == 5) changed_source.tensors[0].format.encoding = fm::Encoding::kF16;
    if (mutation == 6) {
      auto it = std::ranges::find_if(changed_source.tensors, [](const auto& t) {
        return t.format.encoding == fm::Encoding::kAffine;
      });
      ASSERT_NE(it, changed_source.tensors.end());
      it->format.params.bits = 8;
    }
    if (mutation == 7) {
      changed_model.stages[0].heads.output->weights[0].tensor =
          changed_model.stages[0].embedding->weights[0].tensor;
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
        if (mutation == 10) t.format.params.group = 64;
        if (mutation == 11) t.format.params.scale = fm::AffineScale::kF16;
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
                            .shape = {.ne = {k, n, 1, 1}, .rank = 2},
                            .stacked = stacked};
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
    return rt::CheckStoredHashTables(*artifact, model);
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
  EXPECT_FALSE(rt::CheckStoredHashTables(*artifact, missing));
  auto uncounted = model;
  uncounted.stages[0].side[0].source.params.erase(
      uncounted.stages[0].side[0].source.params.begin());
  EXPECT_FALSE(rt::CheckStoredHashTables(*artifact, uncounted));
  std::filesystem::remove_all(scratch, ec);
}
