// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The registered kernel set (kernels/common/device_backend.h) on the fake
// one, in every profile: registration, the backend made through it, its
// workspace binding and launch, and the composed kernel sources.

#include "kernels/common/device_backend.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "execution/registry.h"
#include "kernels/common/fake_device_backend.h"
#include "kernels/common/selected_operation.h"
#include "providers/fake/fake_device_execution.h"

namespace {

namespace kn = llmp::kernels;
namespace ex = llmp::execution;

// A source with one implementation, which lowers and launches nothing.
class OneSource final : public kn::SelectedSource {
 public:
  explicit OneSource(std::string name) : name_(std::move(name)) {}
  std::vector<ex::Implementation> Implementations() const override {
    ex::Implementation implementation;
    implementation.name = name_;
    return {implementation};
  }
  std::vector<ex::TunedShape> Tuning(std::uint32_t) const override { return {}; }
  bool Owns(std::string_view implementation) const override { return implementation == name_; }
  std::uint64_t persistent_bytes() const override { return 0; }
  std::expected<std::unique_ptr<kn::SelectedCall>, kn::KernelFailure> Lower(
      const ex::Implementation&, const llmp::ir::Node&, std::span<const kn::SelectedOperand>,
      std::span<const kn::SelectedOperand>) const override {
    return std::unexpected(kn::KernelFailure{.detail = "lowers nothing"});
  }
  std::expected<std::unique_ptr<kn::SelectedLaunch>, kn::KernelFailure> Launch(
      int, llmp::providers::DeviceExecution&, llmp::providers::StreamId,
      kn::SelectedWorkspace) const override {
    return std::unexpected(kn::KernelFailure{.detail = "launches nothing"});
  }

 private:
  std::string name_;
};

std::shared_ptr<const kn::SelectedSource> Source(std::string name) {
  return std::make_shared<OneSource>(std::move(name));
}

// The build's sources of a second set, for the refusal of a second registration.
std::expected<std::vector<std::shared_ptr<const kn::SelectedSource>>, std::string> TwoSources(int) {
  return std::vector{Source("other.a"), Source("other.b")};
}

kn::BackendCapabilities FourLanes() { return {.owner_lanes = 4}; }

TEST(DeviceBackendTest, OneKernelSetIsRegisteredAndMakesEveryBackend) {
  // Nothing in this process registered a set before.
  ASSERT_EQ(kn::RegisteredKernelSet(), nullptr);
  llmp::providers::fake::FakeDeviceExecution execution;
  const auto stream = execution.CreateStream();
  ASSERT_TRUE(stream);
  EXPECT_FALSE(kn::CreateDeviceBackend(0, execution, *stream));
  EXPECT_FALSE(kn::ComposeSelectedSources(0, nullptr));
  EXPECT_EQ(kn::DeviceBackendCapabilities().owner_lanes, 0U);

  ASSERT_TRUE(kn::RegisterKernelSet(kn::FakeKernelSet()));
  EXPECT_TRUE(kn::RegisterKernelSet(kn::FakeKernelSet()));  // the same set again
  static const kn::KernelSet kOther{.name = "other",
                                    .capabilities = &FourLanes,
                                    .sources = &TwoSources,
                                    .create = kn::FakeKernelSet().create};
  EXPECT_FALSE(kn::RegisterKernelSet(kOther));
  EXPECT_FALSE(kn::RegisterKernelSet(kn::KernelSet{.name = "incomplete"}));
  EXPECT_EQ(kn::RegisteredKernelSet(), &kn::FakeKernelSet());

  auto backend = kn::CreateDeviceBackend(0, execution, *stream);
  ASSERT_TRUE(backend) << backend.error();
  auto& b = **backend;
  EXPECT_EQ(b.library_workspace_bytes(), kn::kFakeLibraryWorkspaceBytes);
  EXPECT_EQ(b.persistent_bytes(), 0U);
  EXPECT_EQ(b.registry().size(), 0U);
  EXPECT_EQ(b.device().device_class, "fake");
  EXPECT_TRUE(b.FundConstruction([](std::uint64_t) { return true; }, [](std::uint64_t) {}));
  EXPECT_TRUE(b.Declare({}, [](std::uint64_t) { return true; }, [](std::uint64_t) {}));
  // No implementation exists to pin, by name or by the backend's own default.
  const std::array<ex::TunedShape, 1> pin{{{.implementation = "anything", .rows = 1}}};
  EXPECT_FALSE(b.Declare(pin, [](std::uint64_t) { return true; }, [](std::uint64_t) {}));
  EXPECT_FALSE(b.Declare(
      {}, [](std::uint64_t) { return true; }, [](std::uint64_t) {},
      {.image_attention_head_dim = 128}));
  EXPECT_FALSE(b.Launch({}));  // before its workspace is bound
  const std::array<std::uint64_t, 1> bases{4096};
  ASSERT_TRUE(b.BindWorkspaces(bases));
  auto launch = b.Launch({});
  ASSERT_TRUE(launch) << launch.error();
  EXPECT_NE(*launch, nullptr);
  b.ReleaseWorkspaces();
  EXPECT_FALSE(b.Launch({}));
  // Every plan over it is refused.
  EXPECT_TRUE(b.Backend().lower);

  // A fake backend owns no source.
  EXPECT_FALSE(kn::CreateDeviceBackend(0, execution, *stream, {.selected = Source("own")}));

  // The caller's source comes first and alone is preferred; the fake set
  // adds none of its own.
  auto composed = kn::ComposeSelectedSources(0, Source("own"));
  ASSERT_TRUE(composed) << composed.error();
  EXPECT_EQ(composed->preferred, std::vector<std::string>{"own"});
  ASSERT_NE(composed->source, nullptr);
  EXPECT_TRUE(composed->source->Owns("own"));
  EXPECT_EQ(composed->source->Implementations().size(), 1U);
  auto none = kn::ComposeSelectedSources(0, nullptr);
  ASSERT_TRUE(none);
  EXPECT_TRUE(none->preferred.empty());
  EXPECT_TRUE(none->source->Implementations().empty());
}

}  // namespace
