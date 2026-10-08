// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Script the driver's entrypoints to exercise cleanup failures without
// requiring a GPU or deliberately faulting a real CUDA context.

#include "providers/cuda/cuda_device_execution.h"

#include <cuda.h>
#include <gtest/gtest.h>

#include <memory>

#include "expected_error.h"
#include "providers/device_execution.h"

namespace {

using llmp::test_support::FailedCode;

struct Driver {
  CUresult current = CUDA_SUCCESS;
  CUresult destroy_stream = CUDA_SUCCESS;
  CUresult destroy_event = CUDA_SUCCESS;
  int records = 0;
  int event_creates = 0;
  int event_destroys = 0;
};

Driver driver;

}  // namespace

// These executable definitions override the linked driver's symbols. Only
// this test executable sees them; the real GPU tests use the actual driver.
extern "C" {
CUresult CUDAAPI cuInit(unsigned int /*Flags*/) { return CUDA_SUCCESS; }
CUresult CUDAAPI cuDeviceGet(CUdevice* device, int ordinal) {
  *device = ordinal;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice /*dev*/) {
  *pctx = reinterpret_cast<CUcontext>(1);  // NOLINT(performance-no-int-to-ptr)
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice /*dev*/) { return CUDA_SUCCESS; }
CUresult CUDAAPI cuCtxSetCurrent(CUcontext /*ctx*/) { return driver.current; }
CUresult CUDAAPI cuStreamCreate(CUstream* phStream, unsigned int /*Flags*/) {
  *phStream = reinterpret_cast<CUstream>(2);  // NOLINT(performance-no-int-to-ptr)
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamDestroy(CUstream /*hStream*/) { return driver.destroy_stream; }
CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr /*dst*/, CUdeviceptr /*src*/, size_t /*ByteCount*/,
                               CUstream /*hStream*/) {
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamWaitEvent(CUstream /*hStream*/, CUevent /*hEvent*/,
                                   unsigned int /*Flags*/) {
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventCreate(CUevent* phEvent, unsigned int /*Flags*/) {
  ++driver.event_creates;
  *phEvent = reinterpret_cast<CUevent>(3);  // NOLINT(performance-no-int-to-ptr)
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventRecord(CUevent /*hEvent*/, CUstream /*hStream*/) {
  ++driver.records;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventQuery(CUevent /*hEvent*/) { return CUDA_SUCCESS; }
CUresult CUDAAPI cuEventDestroy(CUevent /*hEvent*/) {
  ++driver.event_destroys;
  return driver.destroy_event;
}
CUresult CUDAAPI cuGetErrorName(CUresult /*error*/, const char** pStr) {
  *pStr = "scripted CUDA error";
  return CUDA_SUCCESS;
}
}  // extern "C"

namespace {

using llmp::providers::DeviceExecution;
using llmp::providers::FenceId;
using llmp::providers::FenceState;
using llmp::providers::ProviderError;
using llmp::providers::StreamId;

class CudaExecutionFailureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    driver = Driver{};
    // No pool: each fence makes its event and its release destroys it, as
    // these failures of destruction need.
    auto opened =
        llmp::providers::cuda::OpenDeviceExecution(0, {.events_ahead = 0, .events_kept = 0});
    ASSERT_TRUE(opened.has_value());
    execution_ = std::move(*opened);
    stream_ = execution_->CreateStream().value();
  }

  FenceId CompleteFence() {
    const FenceId fence = execution_->Record(stream_).value();
    EXPECT_EQ(execution_->Query(fence).value(), FenceState::kComplete);
    return fence;
  }

  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
};

// With the pool (the default), events are made when the provider opens and
// kept when a fence is released: fencing makes and destroys none (RE-029:
// cuEventCreate blocks while another thread launches into a full stream).
TEST(CudaExecutionPoolTest, FencesReuseEventsMadeAhead) {
  driver = Driver{};
  {
    auto opened =
        llmp::providers::cuda::OpenDeviceExecution(0, {.events_ahead = 4, .events_kept = 4});
    ASSERT_TRUE(opened.has_value());
    const StreamId stream = (*opened)->CreateStream().value();
    EXPECT_EQ(driver.event_creates, 4);
    for (int i = 0; i < 20; ++i) {
      const FenceId fence = (*opened)->Record(stream).value();
      EXPECT_EQ((*opened)->Query(fence).value(), FenceState::kComplete);
      EXPECT_TRUE((*opened)->Release(fence).has_value());
    }
    EXPECT_EQ(driver.event_creates, 4);
    EXPECT_EQ(driver.event_destroys, 0);
    EXPECT_TRUE((*opened)->DestroyStream(stream).has_value());
  }
  EXPECT_EQ(driver.event_destroys, 4);  // the pool's, when the provider closes
}

TEST_F(CudaExecutionFailureTest, AnUnknownStreamDestructionCannotRecordNewWork) {
  driver.destroy_stream = CUDA_ERROR_UNKNOWN;
  EXPECT_EQ(FailedCode(execution_->DestroyStream(stream_)), ProviderError::kUnknown);
  EXPECT_EQ(FailedCode(execution_->Record(stream_)), ProviderError::kInvalid);
  EXPECT_EQ(driver.records, 0);
}

TEST_F(CudaExecutionFailureTest, ARefusedEventDestructionKeepsTheFenceForRetry) {
  const FenceId fence = CompleteFence();
  driver.destroy_event = CUDA_ERROR_INVALID_CONTEXT;
  EXPECT_EQ(FailedCode(execution_->Release(fence)), ProviderError::kFailed);
  EXPECT_EQ(FailedCode(execution_->DestroyStream(stream_)), ProviderError::kInvalid);
  driver.destroy_event = CUDA_SUCCESS;
  EXPECT_TRUE(execution_->Release(fence).has_value());
  EXPECT_TRUE(execution_->DestroyStream(stream_).has_value());
  EXPECT_EQ(driver.event_destroys, 2);
}

TEST_F(CudaExecutionFailureTest, ARefusedContextChangeDoesNotForgetTheFence) {
  const FenceId fence = CompleteFence();
  driver.current = CUDA_ERROR_INVALID_CONTEXT;
  EXPECT_EQ(FailedCode(execution_->Release(fence)), ProviderError::kFailed);
  EXPECT_EQ(driver.event_destroys, 0);
  driver.current = CUDA_SUCCESS;
  EXPECT_TRUE(execution_->Release(fence).has_value());
  EXPECT_TRUE(execution_->DestroyStream(stream_).has_value());
}

TEST_F(CudaExecutionFailureTest, ASubmissionHandleIsTheStreamAndCountsAsQueuedWork) {
  driver.current = CUDA_ERROR_INVALID_CONTEXT;
  EXPECT_EQ(FailedCode(execution_->Submission(stream_)), ProviderError::kFailed);
  driver.current = CUDA_SUCCESS;
  EXPECT_EQ(execution_->Submission(stream_).value().handle,
            reinterpret_cast<void*>(2));  // NOLINT(performance-no-int-to-ptr)
  EXPECT_EQ(FailedCode(execution_->DestroyStream(stream_)), ProviderError::kInvalid);
  EXPECT_TRUE(execution_->Release(CompleteFence()).has_value());
  EXPECT_TRUE(execution_->DestroyStream(stream_).has_value());
}

TEST_F(CudaExecutionFailureTest, AnUnknownEventDestructionQuarantinesItsHandle) {
  const FenceId fence = CompleteFence();
  driver.destroy_event = CUDA_ERROR_UNKNOWN;
  EXPECT_EQ(FailedCode(execution_->Release(fence)), ProviderError::kUnknown);
  driver.destroy_event = CUDA_SUCCESS;
  EXPECT_EQ(FailedCode(execution_->Release(fence)), ProviderError::kInvalid);
  EXPECT_EQ(FailedCode(execution_->Query(fence)), ProviderError::kInvalid);
  EXPECT_EQ(FailedCode(execution_->Wait(stream_, fence)), ProviderError::kInvalid);
  EXPECT_EQ(FailedCode(execution_->DestroyStream(stream_)), ProviderError::kInvalid);
  EXPECT_EQ(driver.event_destroys, 1);
}

}  // namespace
