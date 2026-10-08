// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The CUDA provider's device probe: its judgment of what the driver
// reports, on made-up facts, and its handling of a missing or wrong driver
// library. Runs on any host; the probe of a real GB10 is the gpu-labelled
// smoke.doctor test.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "base/report.h"
#include "providers/cuda/cuda_facts.h"
#include "providers/cuda/cuda_probe.h"

namespace {

using llmp::providers::cuda::CudaDeviceFacts;
using llmp::providers::cuda::CudaFacts;
using llmp::providers::cuda::CudaGranularity;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

constexpr std::uint64_t kTwoMiB = 2 << 20;

std::optional<std::string> Value(const llmp::base::Report& report, std::string_view title,
                                 std::string_view key) {
  for (const auto& section : report.sections) {
    if (section.title != title) {
      continue;
    }
    for (const auto& line : section.lines) {
      if (line.key == key) {
        return line.value;
      }
    }
  }
  return std::nullopt;
}

CudaGranularity TwoMiB() {
  CudaGranularity granularity;
  granularity.minimum = kTwoMiB;
  granularity.recommended = kTwoMiB;
  return granularity;
}

// What the probe reads on `spark` (GB10, driver 580.178.04).
CudaDeviceFacts Gb10() {
  CudaDeviceFacts device;
  device.name = "NVIDIA GB10";
  device.major = 12;
  device.minor = 1;
  device.integrated = true;
  device.vmm = true;
  device.host_numa_vmm = true;
  device.gpu_direct_rdma = false;
  device.host_numa_id = 0;
  device.memory_bytes = std::uint64_t{127598832} * 1024;
  device.device_local = TwoMiB();
  device.host_numa = TwoMiB();
  return device;
}

// What it reads on the workstation (RTX 3080 Ti, driver 595.91.07).
CudaDeviceFacts Rtx3080Ti() {
  CudaDeviceFacts device = Gb10();
  device.name = "NVIDIA GeForce RTX 3080 Ti";
  device.major = 8;
  device.minor = 6;
  device.integrated = false;
  device.memory_bytes = std::uint64_t{12163481} * 1024;
  return device;
}

CudaFacts SparkFacts() {
  CudaFacts facts;
  facts.built_version = 13040;
  facts.built_architectures = {121};
  facts.library = "/lib/aarch64-linux-gnu/libcuda.so.1";
  facts.driver_version = 13000;
  facts.nvidia = {.loaded = true, .version = "580.178.04"};
  facts.device_count = 1;
  facts.devices = {Gb10()};
  return facts;
}

// The native x86-64 build on the workstation: code for the GB10 and for
// the discrete sm_86 (D-082).
CudaFacts WorkstationFacts() {
  CudaFacts facts = SparkFacts();
  facts.built_architectures = {121, 86};
  facts.library = "/lib/x86_64-linux-gnu/libcuda.so.1";
  facts.driver_version = 13020;
  facts.nvidia = {.loaded = true, .version = "595.91.07"};
  facts.devices = {Rtx3080Ti()};
  return facts;
}

llmp::base::Report Describe(const CudaFacts& facts) {
  llmp::base::Report report;
  llmp::providers::cuda::DescribeCuda(facts, report);
  return report;
}

TEST(CudaArchitectures, Parse) {
  using llmp::providers::cuda::ParseCudaArchitectures;
  EXPECT_THAT(ParseCudaArchitectures("121-real"), ElementsAre(121));
  EXPECT_THAT(ParseCudaArchitectures("121-real,90-virtual,86"), ElementsAre(121, 86));
  EXPECT_THAT(ParseCudaArchitectures("121-virtual"), IsEmpty());
  EXPECT_THAT(ParseCudaArchitectures("121a-real,121-real"), ElementsAre(121));
  EXPECT_THAT(ParseCudaArchitectures("native,,0,121"), ElementsAre(121));
  EXPECT_THAT(ParseCudaArchitectures(""), IsEmpty());
}

TEST(CudaVersion, Text) {
  using llmp::providers::cuda::CudaVersionText;
  EXPECT_EQ(CudaVersionText(13040), "13.4");
  EXPECT_EQ(CudaVersionText(13000), "13.0");
  EXPECT_EQ(CudaVersionText(12090), "12.9");
}

TEST(DescribeCuda, Spark) {
  const llmp::base::Report report = Describe(SparkFacts());
  EXPECT_THAT(report.problems, IsEmpty());
  EXPECT_THAT(report.warnings, IsEmpty());
  EXPECT_EQ(Value(report, "NVIDIA driver", "kernel module"), "580.178.04");
  EXPECT_EQ(Value(report, "NVIDIA driver", "library"), "/lib/aarch64-linux-gnu/libcuda.so.1");
  EXPECT_EQ(Value(report, "NVIDIA driver", "CUDA driver API"),
            "13.0, older than this build's 13.4: CUDA minor-version compatibility");
  EXPECT_EQ(Value(report, "NVIDIA driver", "CUDA toolkit (this build)"), "13.4");
  EXPECT_EQ(Value(report, "NVIDIA driver", "GPU code (this build)"), "sm_121");
  EXPECT_EQ(Value(report, "NVIDIA driver", "GPUDirect Storage"),
            "nvidia_fs not loaded (no native GDS)");
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GB10", "compute capability"), "12.1 (sm_121)");
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GB10", "class"),
            "unified (integrated: one memory budget with the host)");
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GB10", "memory"), "121.7 GiB");
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GB10", "virtual memory management"), "supported");
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GB10", "device-local backing"),
            "granularity 2 MiB minimum, 2 MiB recommended");
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GB10", "host NUMA node 0 backing"),
            "granularity 2 MiB minimum, 2 MiB recommended");
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GB10", "GPUDirect RDMA"), "not supported");
}

// A discrete GPU the build targets is judged as a GB10 is: VMM, both
// backing classes and the compute mode (D-082).
TEST(DescribeCuda, Workstation) {
  const llmp::base::Report report = Describe(WorkstationFacts());
  EXPECT_THAT(report.problems, IsEmpty());
  EXPECT_THAT(report.warnings, IsEmpty());
  EXPECT_EQ(Value(report, "NVIDIA driver", "GPU code (this build)"), "sm_121, sm_86");
  const std::string_view gpu = "GPU 0: NVIDIA GeForce RTX 3080 Ti";
  EXPECT_EQ(Value(report, gpu, "compute capability"), "8.6 (sm_86)");
  EXPECT_EQ(Value(report, gpu, "class"), "discrete (its own device memory)");
  EXPECT_EQ(Value(report, gpu, "memory"), "11.6 GiB");
  EXPECT_EQ(Value(report, gpu, "device-local backing"),
            "granularity 2 MiB minimum, 2 MiB recommended");
  EXPECT_EQ(Value(report, gpu, "host NUMA node 0 backing"),
            "granularity 2 MiB minimum, 2 MiB recommended");

  CudaFacts facts = WorkstationFacts();
  facts.devices[0].vmm = false;
  facts.devices[0].device_local.reset();
  EXPECT_THAT(Describe(facts).problems,
              ElementsAre("GPU 0 does not report CUDA virtual memory management, which llmpalooza "
                          "requires (D-006)"));
  facts = WorkstationFacts();
  facts.devices[0].host_numa_vmm = false;
  facts.devices[0].host_numa.reset();
  EXPECT_THAT(Describe(facts).problems,
              ElementsAre("GPU 0: no host-backed VMM (the device does not support it), which "
                          "direct file reads need as their landing zone (D-034, D-081)"));
  facts = WorkstationFacts();
  facts.devices[0].device_local =
      CudaGranularity{.minimum = 4 << 20, .recommended = 4 << 20, .error = ""};
  ASSERT_EQ(Describe(facts).problems.size(), 1U);
  EXPECT_THAT(Describe(facts).problems[0], HasSubstr("GPU 0: device-local VMM backing: "));
}

// Llmpalooza uses GPU 0 only (D-082): another GPU, even one the build has code
// for, is reported but not judged, and a host with more than one is warned.
TEST(DescribeCuda, OnlyGpu0IsUsed) {
  CudaFacts facts = WorkstationFacts();
  facts.devices.insert(facts.devices.begin(), Gb10());
  facts.devices[1].ordinal = 1;
  facts.devices[1].host_numa_vmm = false;
  facts.devices[1].host_numa.reset();
  facts.device_count = 2;
  const llmp::base::Report report = Describe(facts);
  const std::string_view other = "GPU 1: NVIDIA GeForce RTX 3080 Ti";
  EXPECT_EQ(Value(report, other, "compute capability"), "8.6 (sm_86)");
  EXPECT_EQ(Value(report, other, "use"), "none: llmpalooza uses GPU 0 only (D-082)");
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GB10", "use"), std::nullopt);
  EXPECT_THAT(report.problems, IsEmpty());
  EXPECT_THAT(report.warnings,
              ElementsAre("the CUDA driver reports 2 GPUs; llmpalooza uses only GPU 0, which "
                          "CUDA_VISIBLE_DEVICES selects (D-082)"));
}

// GPU 0 must be one the build has code for, even when another GPU is.
TEST(DescribeCuda, Gpu0MustBeTargeted) {
  CudaFacts facts = SparkFacts();
  CudaDeviceFacts gb10 = Gb10();
  gb10.ordinal = 1;
  facts.devices = {Rtx3080Ti(), gb10};
  facts.device_count = 2;
  EXPECT_THAT(Describe(facts).problems,
              ElementsAre("GPU 0 is not one this build has code for (sm_121), and llmpalooza uses "
                          "only GPU 0: select one it has code for with CUDA_VISIBLE_DEVICES "
                          "(D-082)"));
}

// The code for each architecture assumes its class of memory: sm_121 the
// GB10's unified budget, anything else a discrete GPU's own.
TEST(DescribeCuda, ClassMustMatchTheTargetedArchitecture) {
  using llmp::providers::cuda::CudaDeviceClass;
  using llmp::providers::cuda::DeviceClassOf;
  using llmp::providers::cuda::TargetClass;
  EXPECT_EQ(TargetClass(121), CudaDeviceClass::kUnified);
  EXPECT_EQ(TargetClass(86), CudaDeviceClass::kDiscrete);
  EXPECT_EQ(TargetClass(120), CudaDeviceClass::kDiscrete);
  EXPECT_EQ(DeviceClassOf(Gb10()), CudaDeviceClass::kUnified);
  EXPECT_EQ(DeviceClassOf(Rtx3080Ti()), CudaDeviceClass::kDiscrete);

  CudaFacts facts = SparkFacts();
  facts.devices[0].integrated = false;
  llmp::base::Report report = Describe(facts);
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GB10", "class"), "discrete (its own device memory)");
  EXPECT_THAT(report.problems,
              ElementsAre("GPU 0 is sm_121 but discrete, and this build's sm_121 code is for the "
                          "GB10's unified memory (D-004)"));

  facts = WorkstationFacts();
  facts.devices[0].integrated = true;
  EXPECT_THAT(Describe(facts).problems,
              ElementsAre("GPU 0 is sm_86 but integrated, and this build targets sm_86 as a "
                          "discrete GPU (D-082)"));

  facts.devices[0].integrated.reset();
  EXPECT_EQ(DeviceClassOf(facts.devices[0]), std::nullopt);
  report = Describe(facts);
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GeForce RTX 3080 Ti", "class"), "unknown");
  EXPECT_THAT(report.problems,
              ElementsAre("GPU 0 does not report whether it is integrated, so whether its memory "
                          "is unified or discrete is unknown (D-082)"));

  // An untargeted GPU's class is reported, not judged.
  facts = SparkFacts();
  CudaDeviceFacts other = Rtx3080Ti();
  other.ordinal = 1;
  other.integrated = true;
  facts.devices.push_back(other);
  EXPECT_THAT(Describe(facts).problems, IsEmpty());
}

TEST(DescribeCuda, SameOrNewerDriver) {
  CudaFacts facts = SparkFacts();
  facts.driver_version = 13040;
  EXPECT_EQ(Value(Describe(facts), "NVIDIA driver", "CUDA driver API"), "13.4");
  facts.driver_version = 14000;
  EXPECT_EQ(Value(Describe(facts), "NVIDIA driver", "CUDA driver API"), "14.0");
  EXPECT_THAT(Describe(facts).problems, IsEmpty());
}

TEST(DescribeCuda, DriverForAnOlderMajorVersion) {
  CudaFacts facts = SparkFacts();
  facts.driver_version = 12090;
  EXPECT_THAT(Describe(facts).problems,
              ElementsAre("the driver supports CUDA 12.9, but this build needs a driver for CUDA "
                          "13"));
}

TEST(DescribeCuda, InitFailed) {
  CudaFacts facts = SparkFacts();
  facts.init_error = "cuInit returned CUDA_ERROR_NO_DEVICE (100)";
  facts.devices.clear();
  EXPECT_THAT(Describe(facts).problems,
              ElementsAre("the CUDA driver did not initialize: cuInit returned "
                          "CUDA_ERROR_NO_DEVICE (100)"));
}

TEST(DescribeCuda, NoDevices) {
  CudaFacts facts = SparkFacts();
  facts.devices.clear();
  facts.device_count = 0;
  EXPECT_THAT(Describe(facts).problems, ElementsAre("the CUDA driver reports no GPU"));
}

TEST(DescribeCuda, NoTargetedDevice) {
  CudaFacts facts = SparkFacts();
  facts.devices = {Rtx3080Ti()};
  const llmp::base::Report report = Describe(facts);
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GeForce RTX 3080 Ti", "compute capability"),
            "8.6 (sm_86, which this build has no code for)");
  EXPECT_THAT(report.problems, ElementsAre("no GPU here is one this build has code for (sm_121)"));
  EXPECT_THAT(report.warnings, IsEmpty());
}

// Only the devices this build targets are judged.
TEST(DescribeCuda, UntargetedDevicesAreNotJudged) {
  CudaFacts facts = SparkFacts();
  CudaDeviceFacts other = Rtx3080Ti();
  other.ordinal = 1;
  other.vmm = false;
  other.device_local.reset();
  other.host_numa.reset();
  facts.devices.push_back(other);
  facts.device_count = 2;
  const llmp::base::Report report = Describe(facts);
  EXPECT_EQ(Value(report, "GPU 1: NVIDIA GeForce RTX 3080 Ti", "virtual memory management"),
            "not supported");
  EXPECT_THAT(report.problems, IsEmpty());
  EXPECT_THAT(report.warnings, ElementsAre(HasSubstr("reports 2 GPUs")));
}

TEST(DescribeCuda, NoVmm) {
  CudaFacts facts = SparkFacts();
  facts.devices[0].vmm = false;
  facts.devices[0].device_local.reset();
  EXPECT_THAT(Describe(facts).problems,
              ElementsAre("GPU 0 does not report CUDA virtual memory management, which llmpalooza "
                          "requires (D-006)"));
  facts.devices[0].vmm.reset();
  EXPECT_EQ(Value(Describe(facts), "GPU 0: NVIDIA GB10", "virtual memory management"), "unknown");
  EXPECT_EQ(Describe(facts).problems.size(), 1U);
}

TEST(DescribeCuda, DeviceGranularityFailed) {
  CudaFacts facts = SparkFacts();
  facts.devices[0].device_local = CudaGranularity{.minimum = 0, .recommended = 0, .error = "x"};
  const llmp::base::Report report = Describe(facts);
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GB10", "device-local backing"), "unknown: x");
  EXPECT_THAT(report.problems, ElementsAre("GPU 0: device-local VMM backing: x"));
}

TEST(DescribeCuda, NoHostBacking) {
  CudaFacts facts = SparkFacts();
  facts.devices[0].host_numa_vmm = false;
  facts.devices[0].host_numa.reset();
  llmp::base::Report report = Describe(facts);
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GB10", "host NUMA node 0 backing"), "not supported");
  EXPECT_THAT(report.warnings, IsEmpty());
  EXPECT_THAT(report.problems,
              ElementsAre("GPU 0: no host-backed VMM (the device does not support it), which "
                          "direct file reads need as their landing zone (D-034, D-081)"));

  facts.devices[0].host_numa_vmm = true;
  facts.devices[0].host_numa_id = -1;
  report = Describe(facts);
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GB10", "host NUMA backing"),
            "supported, but the device has no host NUMA node");
  ASSERT_EQ(report.problems.size(), 1U);
  EXPECT_THAT(report.problems[0], HasSubstr("(the device has no host NUMA node)"));

  facts.devices[0].host_numa_id.reset();
  report = Describe(facts);
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GB10", "host NUMA backing"),
            "supported, but the device's host NUMA node is unknown");
  ASSERT_EQ(report.problems.size(), 1U);
  EXPECT_THAT(report.problems[0], HasSubstr("(the device's host NUMA node is unknown)"));

  facts.devices[0].host_numa_id = 0;
  facts.devices[0].host_numa = CudaGranularity{.minimum = 0, .recommended = 0, .error = "y"};
  report = Describe(facts);
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GB10", "host NUMA node 0 backing"), "unknown: y");
  ASSERT_EQ(report.problems.size(), 1U);
  EXPECT_THAT(report.problems[0], HasSubstr("(y)"));
}

// A driver that does not know the host-NUMA attributes (older than 12.2).
TEST(DescribeCuda, HostBackingUnknown) {
  CudaFacts facts = SparkFacts();
  facts.devices[0].host_numa_vmm.reset();
  facts.devices[0].host_numa_id.reset();
  facts.devices[0].host_numa.reset();
  const llmp::base::Report report = Describe(facts);
  EXPECT_EQ(Value(report, "GPU 0: NVIDIA GB10", "host NUMA backing"), "unknown");
  ASSERT_EQ(report.problems.size(), 1U);
  EXPECT_THAT(report.problems[0], HasSubstr("(support is unknown)"));
}

TEST(DescribeCuda, GranularityCannotBackPagingChunks) {
  for (const auto& [minimum, recommended] : {std::pair<std::uint64_t, std::uint64_t>{0, kTwoMiB},
                                             {3 << 20, 3 << 20},
                                             {4 << 20, 4 << 20},
                                             {kTwoMiB, 0},
                                             {kTwoMiB, 3},
                                             {kTwoMiB, 1 << 20}}) {
    CudaFacts facts = SparkFacts();
    facts.devices[0].device_local =
        CudaGranularity{.minimum = minimum, .recommended = recommended, .error = ""};
    facts.devices[0].host_numa = facts.devices[0].device_local;
    const llmp::base::Report report = Describe(facts);
    // One problem for each backing class.
    ASSERT_EQ(report.problems.size(), 2U) << minimum;
    EXPECT_THAT(report.problems[0], HasSubstr("GPU 0: device-local VMM backing: a minimum of "));
    EXPECT_THAT(report.problems[0], HasSubstr("cannot back 2 MiB paging chunks (D-056)"));
    EXPECT_THAT(report.problems[1], HasSubstr("GPU 0: no host-backed VMM (a minimum of "));
  }
  // Smaller powers of two divide the chunk.
  CudaFacts facts = SparkFacts();
  facts.devices[0].device_local =
      CudaGranularity{.minimum = 64 << 10, .recommended = kTwoMiB, .error = ""};
  EXPECT_THAT(Describe(facts).problems, IsEmpty());
}

TEST(DescribeCuda, UnknownDriverVersion) {
  CudaFacts facts = SparkFacts();
  facts.driver_version = 0;
  const llmp::base::Report report = Describe(facts);
  EXPECT_EQ(Value(report, "NVIDIA driver", "CUDA driver API"), "unknown");
  EXPECT_THAT(report.problems,
              ElementsAre("the driver did not report its CUDA version, so whether it can run this "
                          "build's CUDA 13.4 is unknown"));
}

TEST(DescribeCuda, ComputeMode) {
  using llmp::providers::cuda::CudaComputeMode;
  CudaFacts facts = SparkFacts();
  facts.devices[0].compute_mode = CudaComputeMode::kDefault;
  EXPECT_EQ(Value(Describe(facts), "GPU 0: NVIDIA GB10", "compute mode"), "default");
  facts.devices[0].compute_mode = CudaComputeMode::kExclusiveProcess;
  EXPECT_EQ(Value(Describe(facts), "GPU 0: NVIDIA GB10", "compute mode"), "exclusive process");
  EXPECT_THAT(Describe(facts).problems, IsEmpty());
  facts.devices[0].compute_mode = CudaComputeMode::kProhibited;
  EXPECT_THAT(Describe(facts).problems,
              ElementsAre("GPU 0 is in the prohibited compute mode, so no process can use it"));
  facts.devices[0].compute_mode.reset();
  EXPECT_EQ(Value(Describe(facts), "GPU 0: NVIDIA GB10", "compute mode"), "unknown");
}

// 12.1 is sm_121; a made-up 1.11 is not, and neither is 12.0.
TEST(DescribeCuda, ComputeCapabilityMatchesExactly) {
  CudaFacts facts = SparkFacts();
  facts.devices[0].major = 1;
  facts.devices[0].minor = 11;
  EXPECT_THAT(Describe(facts).problems,
              ElementsAre("no GPU here is one this build has code for (sm_121)"));
  facts.devices[0].major = 12;
  facts.devices[0].minor = 0;
  EXPECT_THAT(Describe(facts).problems,
              ElementsAre("no GPU here is one this build has code for (sm_121)"));
}

TEST(DescribeCuda, TooManyDevices) {
  CudaFacts facts = SparkFacts();
  facts.device_count = 1000;
  EXPECT_THAT(Describe(facts).problems,
              ElementsAre("the CUDA driver reports 1000 GPUs; doctor checks at most 64"));
}

TEST(DescribeCuda, DeviceError) {
  CudaFacts facts = SparkFacts();
  facts.devices[0] = CudaDeviceFacts{};
  facts.devices[0].error = "cuDeviceGet returned CUDA_ERROR_INVALID_DEVICE (101)";
  const llmp::base::Report report = Describe(facts);
  EXPECT_EQ(Value(report, "GPU 0: ", "error"),
            "cuDeviceGet returned CUDA_ERROR_INVALID_DEVICE (101)");
  EXPECT_THAT(report.problems, ElementsAre("no GPU here is one this build has code for (sm_121)"));
}

// What the probe knows without a GPU: this build, and where the loader
// found libcuda.so.1 (the driver, or on hosts without it NVIDIA's stub, whose
// cuInit() fails).
TEST(ProbeCuda, ThisBuild) {
  const CudaFacts facts = llmp::providers::cuda::ProbeCuda("/nonexistent");
  EXPECT_GT(facts.built_version, 0);
  EXPECT_THAT(facts.built_architectures, testing::Not(IsEmpty()));
  EXPECT_THAT(facts.library, HasSubstr("libcuda.so"));
  EXPECT_FALSE(facts.nvidia.loaded);
  EXPECT_FALSE(facts.nvidia_fs.loaded);
}

}  // namespace
