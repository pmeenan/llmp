// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Llmpalooza's cuBLAS handle for GGML's cuBLAS matrix multiplication (D-053;
// docs/backend-proof.md#dispatch-and-implementations-d-053). GGML's launch
// context would create a handle on first use, with a cudaMalloc workspace
// of its own. Llmpalooza creates it instead, set up as upstream does
// (common.cuh, ggml_backend_cuda_context::cublas_handle): TF32 tensor-op
// math, bound to one provider stream, with a workspace the caller declared
// and charged. The FP16 gate's recorded plan takes this state as part of
// every cuBLAS call's identity (docs/backend-proof.md#tier-e-exact).
//
// Create refuses unless the loaded cuBLAS is the pinned one (D-076), since
// a system cuBLAS found first on the library path would run other kernels,
// and unless none of cuBLAS's own numerics switches (NVIDIA_TF32_OVERRIDE
// and the CUBLAS* variables other than logging and NVTX) is set. The
// libraries report their version only to the patch level (13.8.0), not the
// build (.4). Create reads the environment, which no other thread may
// change meanwhile (llmpalooza never changes it). The workspace size
// is the caller's: cuBLAS chooses algorithms by it, so the FP16 gate's
// recorded plan fixes it at UpstreamWorkspace's.
//
// A launch context (launch.h) on the same stream borrows the handle, and
// the handle must outlive every context that borrows it; the stream must
// outlive the handle. cuBLAS writes the workspace from the GEMMs it queues,
// so the workspace stays mapped and charged until a fence after the last of
// them has completed. cuBLAS creation and lazy kernel allocations remain in
// the paid node runtime allowance and physical-memory checks. CUDA global
// occupancy on unified memory is not a process allocation census.
//
// Destroying the handle is a blocking call: NVIDIA documents that
// cublasDestroy synchronizes the device, so it belongs where the device is
// quiescent. One handle per stream, used on the device submission lane only.

#ifndef LLMP_KERNELS_GGML_CUBLAS_H_
#define LLMP_KERNELS_GGML_CUBLAS_H_

#include <cstdint>
#include <expected>
#include <memory>

#include "base/bytes.h"
#include "base/sha256.h"
#include "kernels/common/device_backend.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/tensors.h"
#include "providers/device_execution.h"

struct CUctx_st;
struct cublasContext;

namespace llmp::kernels::ggml {

class CublasHandle final : public kernels::StreamLibrary {
 public:
  // The workspace upstream gives each handle on a device of compute
  // capability `cc` (100 × major + 10 × minor): 32 MiB from Hopper on, else
  // 4 MiB.
  static base::Bytes UpstreamWorkspace(int cc);

  // A handle on `device` whose GEMMs queue on `stream`, a CUDA provider's
  // stream, using `workspace`: a 256-byte aligned device range that the
  // caller keeps mapped and charged as launch.h's workspace. Refused, with
  // nothing created, if any step fails.
  static std::expected<std::unique_ptr<CublasHandle>, KernelFailure> Create(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      LaunchContext::Workspace workspace);

  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;
  CublasHandle(CublasHandle&&) = delete;
  CublasHandle& operator=(CublasHandle&&) = delete;
  // Synchronizes the device (above), with the CUDA context the handle was
  // created in made current for the call, as cuBLAS requires, and the
  // caller's restored. No launch context may still borrow it.
  ~CublasHandle();

  // Queried once while the setup context is current, before a required
  // source publishes its compatibility identity. Qualified SDK/install
  // provenance supplies library build lineage; these are actual runtime
  // versions and effective handle settings, not a file-byte attestation.
  std::expected<base::Sha256Digest, KernelFailure> ConfigurationIdentity() const;

  cublasContext* native() const { return handle_; }
  int device() const { return device_; }
  providers::StreamId stream() const { return stream_; }
  providers::NativeStream native_stream() const { return native_stream_; }
  LaunchContext::Workspace workspace() const { return workspace_; }

 private:
  friend class LaunchContext;

  CublasHandle(cublasContext* handle, CUctx_st* owner, int device, providers::StreamId stream,
               providers::NativeStream native_stream, LaunchContext::Workspace workspace);

  cublasContext* handle_;
  CUctx_st* owner_;  // the CUDA context current when it was created
  int device_;
  providers::StreamId stream_;
  providers::NativeStream native_stream_;
  LaunchContext::Workspace workspace_;
  int borrowers_ = 0;  // launch contexts holding it
};

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_CUBLAS_H_
