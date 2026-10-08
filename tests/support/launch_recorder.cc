// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "launch_recorder.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <link.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/check.h"
#include "plan_record.h"

namespace llmp::test_support {
namespace {

thread_local std::vector<Event>* recording = nullptr;
// How deep this thread is in wrapped calls: a wrapped entry point the
// runtime or a library calls from inside another is the outer call's work,
// recorded once, by the outer wrapper.
thread_local int depth = 0;

// Marks a wrapped call for its duration; records only the outermost.
class Outer {
 public:
  Outer() : outermost_(depth++ == 0 && recording != nullptr) {}
  ~Outer() { --depth; }
  Outer(const Outer&) = delete;
  Outer& operator=(const Outer&) = delete;
  Outer(Outer&&) = delete;
  Outer& operator=(Outer&&) = delete;
  bool on() const { return outermost_; }

 private:
  bool outermost_;
};

std::array<unsigned, 3> Dims(dim3 dims) { return {dims.x, dims.y, dims.z}; }

// A kernel's attributes, looked up once per function.
struct Attributes {
  int registers = 0;
  std::size_t static_shared = 0;
  std::size_t local = 0;
};

thread_local std::unordered_map<const void*, Attributes> attributes;

Attributes OfKernel(CUkernel kernel) {
  if (const auto known = attributes.find(kernel); known != attributes.end()) {
    return known->second;
  }
  Attributes found;
  CUdevice device = 0;
  int registers = 0;
  int shared = 0;
  int local = 0;
  if (cuCtxGetDevice(&device) == CUDA_SUCCESS &&
      cuKernelGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel, device) ==
          CUDA_SUCCESS &&
      cuKernelGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel, device) ==
          CUDA_SUCCESS &&
      cuKernelGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel, device) ==
          CUDA_SUCCESS) {
    found = {.registers = registers,
             .static_shared = static_cast<std::size_t>(shared),
             .local = static_cast<std::size_t>(local)};
  } else {
    found.registers = -1;  // unknown, which never matches a record
  }
  return attributes[kernel] = found;
}

Attributes OfFunction(const void* function) {
  if (const auto known = attributes.find(function); known != attributes.end()) {
    return known->second;
  }
  Attributes found;
  cudaFuncAttributes attrs{};
  if (cudaFuncGetAttributes(&attrs, function) == cudaSuccess) {
    found = {.registers = attrs.numRegs,
             .static_shared = attrs.sharedSizeBytes,
             .local = attrs.localSizeBytes};
  } else {
    found.registers = -1;
  }
  return attributes[function] = found;
}

void Kernel(std::string name, std::string_view api, dim3 grid, dim3 block, std::size_t shared,
            const Attributes& attrs, const void* stream) {
  Event event;
  event.kind = EventKind::kKernel;
  event.name = std::move(name);
  event.api = api;
  event.grid = Dims(grid);
  event.block = Dims(block);
  event.shared = shared;
  event.registers = attrs.registers;
  event.static_shared = attrs.static_shared;
  event.local = attrs.local;
  event.stream = stream;
  recording->push_back(std::move(event));
}

std::string_view KindName(cudaMemcpyKind kind) {
  switch (kind) {
    case cudaMemcpyHostToHost:
      return "HtoH";
    case cudaMemcpyHostToDevice:
      return "HtoD";
    case cudaMemcpyDeviceToHost:
      return "DtoH";
    case cudaMemcpyDeviceToDevice:
      return "DtoD";
    case cudaMemcpyDefault:
      return "default";
  }
  return "unknown";
}

void Copy(std::string_view kind, std::size_t bytes, const void* stream) {
  Event event;
  event.kind = EventKind::kCopy;
  event.api = kind;
  event.bytes = bytes;
  event.stream = stream;
  recording->push_back(std::move(event));
}

void Cublas(std::string_view function, int m, int n, int k, int batch) {
  Event event;
  event.kind = EventKind::kCublas;
  event.name = function;
  event.shape = {m, n, k, batch};
  recording->push_back(std::move(event));
}

}  // namespace

Recording::Recording() {
  base::Check(recording == nullptr, "one launch recording at a time on a thread");
  recording = &events_;
}

Recording::~Recording() { recording = nullptr; }

std::vector<Event> Recording::Take() { return std::exchange(events_, {}); }

std::vector<std::pair<std::string, std::string>> LoadedCublas() {
  std::vector<std::pair<std::string, std::string>> found;
  dl_iterate_phdr(
      [](dl_phdr_info* info, std::size_t /*size*/, void* data) {
        const std::string_view path = info->dlpi_name != nullptr ? info->dlpi_name : "";
        const std::string_view name = path.substr(path.rfind('/') + 1);
        if (name.starts_with("libcublas.so") || name.starts_with("libcublasLt.so")) {
          static_cast<std::vector<std::pair<std::string, std::string>>*>(data)->emplace_back(name,
                                                                                             path);
        }
        return 0;
      },
      &found);
  return found;
}

}  // namespace llmp::test_support

// The wrapped entry points. The linker sends every call to the wrapper,
// and __real_ to the library's own.
// NOLINTBEGIN(bugprone-reserved-identifier,misc-use-internal-linkage,readability-identifier-naming)
extern "C" {
cudaError_t __real___cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block, void** args,
                                      std::size_t shared, cudaStream_t stream);
cudaError_t __real_cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* function,
                                       void** args);
cudaError_t __real_cudaLaunchKernel(const void* function, dim3 grid, dim3 block, void** args,
                                    std::size_t shared, cudaStream_t stream);
cudaError_t __real_cudaLaunchCooperativeKernel(const void* function, dim3 grid, dim3 block,
                                               void** args, std::size_t shared,
                                               cudaStream_t stream);
cublasStatus_t __real_cublasLtMatmul(cublasLtHandle_t handle, cublasLtMatmulDesc_t description,
                                     const void* alpha, const void* a,
                                     cublasLtMatrixLayout_t a_layout, const void* b,
                                     cublasLtMatrixLayout_t b_layout, const void* beta,
                                     const void* c, cublasLtMatrixLayout_t c_layout, void* d,
                                     cublasLtMatrixLayout_t d_layout,
                                     const cublasLtMatmulAlgo_t* algorithm, void* workspace,
                                     std::size_t workspace_size, cudaStream_t stream);
cudaError_t __real_cudaMemcpyAsync(void* destination, const void* source, std::size_t count,
                                   cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t __real_cudaMemcpy2DAsync(void* destination, std::size_t destination_pitch,
                                     const void* source, std::size_t source_pitch,
                                     std::size_t width, std::size_t height, cudaMemcpyKind kind,
                                     cudaStream_t stream);
cudaError_t __real_cudaMemsetAsync(void* destination, int value, std::size_t count,
                                   cudaStream_t stream);
CUresult __real_cuMemcpyAsync(CUdeviceptr destination, CUdeviceptr source, std::size_t count,
                              CUstream stream);
cublasStatus_t __real_cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                                     cublasOperation_t transb, int m, int n, int k,
                                     const float* alpha, const float* a, int lda, const float* b,
                                     int ldb, const float* beta, float* c, int ldc);
cublasStatus_t __real_cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                                   cublasOperation_t transb, int m, int n, int k, const void* alpha,
                                   const void* a, cudaDataType a_type, int lda, const void* b,
                                   cudaDataType b_type, int ldb, const void* beta, void* c,
                                   cudaDataType c_type, int ldc, cublasComputeType_t compute,
                                   cublasGemmAlgo_t algo);
cublasStatus_t __real_cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                          cublasOperation_t transb, int m, int n, int k,
                                          const void* alpha, const void* const a[],
                                          cudaDataType a_type, int lda, const void* const b[],
                                          cudaDataType b_type, int ldb, const void* beta,
                                          void* const c[], cudaDataType c_type, int ldc, int batch,
                                          cublasComputeType_t compute, cublasGemmAlgo_t algo);
cublasStatus_t __real_cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const void* alpha, const void* a, cudaDataType a_type, int lda, long long stride_a,
    const void* b, cudaDataType b_type, int ldb, long long stride_b, const void* beta, void* c,
    cudaDataType c_type, int ldc, long long stride_c, int batch, cublasComputeType_t compute,
    cublasGemmAlgo_t algo);

cudaError_t __wrap___cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block, void** args,
                                      std::size_t shared, cudaStream_t stream) {
  const llmp::test_support::Outer outer;
  if (outer.on()) {
    const char* name = nullptr;
    auto* const handle = reinterpret_cast<CUkernel>(kernel);
    if (cuKernelGetName(&name, handle) != CUDA_SUCCESS) {
      name = "(unnamed)";
    }
    llmp::test_support::Kernel(name, "cudaLaunchKernel", grid, block, shared,
                               llmp::test_support::OfKernel(handle), stream);
  }
  return __real___cudaLaunchKernel(kernel, grid, block, args, shared, stream);
}

cudaError_t __wrap_cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* function,
                                       void** args) {
  const llmp::test_support::Outer outer;
  if (outer.on()) {
    const char* name = nullptr;
    if (cudaFuncGetName(&name, function) != cudaSuccess) {
      name = "(unnamed)";
    }
    // The FP16 gate lets only the PDL attribute differ; any other attribute
    // is named in the API, which plan_compare.py refuses.
    std::string_view api = "cudaLaunchKernelExC";
    for (unsigned i = 0; i < config->numAttrs; ++i) {
      const cudaLaunchAttributeID id = config->attrs[i].id;
      if (id != cudaLaunchAttributeIgnore &&
          id != cudaLaunchAttributeProgrammaticStreamSerialization) {
        api = "cudaLaunchKernelExC+attributes";
      }
    }
    llmp::test_support::Kernel(name, api, config->gridDim, config->blockDim,
                               config->dynamicSmemBytes, llmp::test_support::OfFunction(function),
                               config->stream);
  }
  return __real_cudaLaunchKernelExC(config, function, args);
}

// The runtime's C entry points, which llmpalooza's EXL3 launchers call
// directly (kernels/exl3/launch.cc): cooperatively for the GEMM, the
// multi-GEMM and the GEMV, plainly for the reconstruction, Hadamard and
// bias-add kernels.
cudaError_t __wrap_cudaLaunchKernel(const void* function, dim3 grid, dim3 block, void** args,
                                    std::size_t shared, cudaStream_t stream) {
  const llmp::test_support::Outer outer;
  if (outer.on()) {
    const char* name = nullptr;
    if (cudaFuncGetName(&name, function) != cudaSuccess) {
      name = "(unnamed)";
    }
    llmp::test_support::Kernel(name, "cudaLaunchKernel", grid, block, shared,
                               llmp::test_support::OfFunction(function), stream);
  }
  return __real_cudaLaunchKernel(function, grid, block, args, shared, stream);
}

cudaError_t __wrap_cudaLaunchCooperativeKernel(const void* function, dim3 grid, dim3 block,
                                               void** args, std::size_t shared,
                                               cudaStream_t stream) {
  const llmp::test_support::Outer outer;
  if (outer.on()) {
    const char* name = nullptr;
    if (cudaFuncGetName(&name, function) != cudaSuccess) {
      name = "(unnamed)";
    }
    llmp::test_support::Kernel(name, "cudaLaunchCooperativeKernel", grid, block, shared,
                               llmp::test_support::OfFunction(function), stream);
  }
  return __real_cudaLaunchCooperativeKernel(function, grid, block, args, shared, stream);
}

// cuBLASLt's matrix product (the EXL3 reconstruction GEMM, recon_gemm.h),
// with m, n and k as its layouts give them: D is m × n, A m × k (no
// transposes; llmpalooza's calls set none). What it launches is in the nsys
// trace of the run.
cublasStatus_t __wrap_cublasLtMatmul(cublasLtHandle_t handle, cublasLtMatmulDesc_t description,
                                     const void* alpha, const void* a,
                                     cublasLtMatrixLayout_t a_layout, const void* b,
                                     cublasLtMatrixLayout_t b_layout, const void* beta,
                                     const void* c, cublasLtMatrixLayout_t c_layout, void* d,
                                     cublasLtMatrixLayout_t d_layout,
                                     const cublasLtMatmulAlgo_t* algorithm, void* workspace,
                                     std::size_t workspace_size, cudaStream_t stream) {
  const llmp::test_support::Outer outer;
  if (outer.on()) {
    std::uint64_t m = 0;
    std::uint64_t n = 0;
    std::uint64_t k = 0;
    std::size_t written = 0;
    const bool read =
        cublasLtMatrixLayoutGetAttribute(d_layout, CUBLASLT_MATRIX_LAYOUT_ROWS, &m, sizeof(m),
                                         &written) == CUBLAS_STATUS_SUCCESS &&
        cublasLtMatrixLayoutGetAttribute(d_layout, CUBLASLT_MATRIX_LAYOUT_COLS, &n, sizeof(n),
                                         &written) == CUBLAS_STATUS_SUCCESS &&
        cublasLtMatrixLayoutGetAttribute(a_layout, CUBLASLT_MATRIX_LAYOUT_COLS, &k, sizeof(k),
                                         &written) == CUBLAS_STATUS_SUCCESS;
    llmp::test_support::Cublas("cublasLtMatmul", read ? static_cast<int>(m) : -1,
                               read ? static_cast<int>(n) : -1, read ? static_cast<int>(k) : -1, 0);
  }
  return __real_cublasLtMatmul(handle, description, alpha, a, a_layout, b, b_layout, beta, c,
                               c_layout, d, d_layout, algorithm, workspace, workspace_size, stream);
}

cudaError_t __wrap_cudaMemcpyAsync(void* destination, const void* source, std::size_t count,
                                   cudaMemcpyKind kind, cudaStream_t stream) {
  const llmp::test_support::Outer outer;
  if (outer.on()) {
    llmp::test_support::Copy(llmp::test_support::KindName(kind), count, stream);
  }
  return __real_cudaMemcpyAsync(destination, source, count, kind, stream);
}

cudaError_t __wrap_cudaMemcpy2DAsync(void* destination, std::size_t destination_pitch,
                                     const void* source, std::size_t source_pitch,
                                     std::size_t width, std::size_t height, cudaMemcpyKind kind,
                                     cudaStream_t stream) {
  const llmp::test_support::Outer outer;
  if (outer.on()) {
    llmp::test_support::Copy(llmp::test_support::KindName(kind), width * height, stream);
  }
  return __real_cudaMemcpy2DAsync(destination, destination_pitch, source, source_pitch, width,
                                  height, kind, stream);
}

cudaError_t __wrap_cudaMemsetAsync(void* destination, int value, std::size_t count,
                                   cudaStream_t stream) {
  const llmp::test_support::Outer outer;
  if (outer.on()) {
    llmp::test_support::Event event;
    event.kind = llmp::test_support::EventKind::kMemset;
    event.bytes = count;
    event.value = value;
    event.stream = stream;
    llmp::test_support::recording->push_back(std::move(event));
  }
  return __real_cudaMemsetAsync(destination, value, count, stream);
}

CUresult __wrap_cuMemcpyAsync(CUdeviceptr destination, CUdeviceptr source, std::size_t count,
                              CUstream stream) {
  const llmp::test_support::Outer outer;
  if (outer.on()) {
    llmp::test_support::Copy("driver", count, stream);
  }
  return __real_cuMemcpyAsync(destination, source, count, stream);
}

cublasStatus_t __wrap_cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                                     cublasOperation_t transb, int m, int n, int k,
                                     const float* alpha, const float* a, int lda, const float* b,
                                     int ldb, const float* beta, float* c, int ldc) {
  const llmp::test_support::Outer outer;
  if (outer.on()) {
    llmp::test_support::Cublas("cublasSgemm_v2", m, n, k, 0);
  }
  return __real_cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                               ldc);
}

cublasStatus_t __wrap_cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                                   cublasOperation_t transb, int m, int n, int k, const void* alpha,
                                   const void* a, cudaDataType a_type, int lda, const void* b,
                                   cudaDataType b_type, int ldb, const void* beta, void* c,
                                   cudaDataType c_type, int ldc, cublasComputeType_t compute,
                                   cublasGemmAlgo_t algo) {
  const llmp::test_support::Outer outer;
  if (outer.on()) {
    llmp::test_support::Cublas("cublasGemmEx", m, n, k, 0);
  }
  return __real_cublasGemmEx(handle, transa, transb, m, n, k, alpha, a, a_type, lda, b, b_type, ldb,
                             beta, c, c_type, ldc, compute, algo);
}

cublasStatus_t __wrap_cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                          cublasOperation_t transb, int m, int n, int k,
                                          const void* alpha, const void* const a[],
                                          cudaDataType a_type, int lda, const void* const b[],
                                          cudaDataType b_type, int ldb, const void* beta,
                                          void* const c[], cudaDataType c_type, int ldc, int batch,
                                          cublasComputeType_t compute, cublasGemmAlgo_t algo) {
  const llmp::test_support::Outer outer;
  if (outer.on()) {
    llmp::test_support::Cublas("cublasGemmBatchedEx", m, n, k, batch);
  }
  return __real_cublasGemmBatchedEx(handle, transa, transb, m, n, k, alpha, a, a_type, lda, b,
                                    b_type, ldb, beta, c, c_type, ldc, batch, compute, algo);
}

cublasStatus_t __wrap_cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const void* alpha, const void* a, cudaDataType a_type, int lda, long long stride_a,
    const void* b, cudaDataType b_type, int ldb, long long stride_b, const void* beta, void* c,
    cudaDataType c_type, int ldc, long long stride_c, int batch, cublasComputeType_t compute,
    cublasGemmAlgo_t algo) {
  const llmp::test_support::Outer outer;
  if (outer.on()) {
    llmp::test_support::Cublas("cublasGemmStridedBatchedEx", m, n, k, batch);
  }
  return __real_cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, a, a_type, lda,
                                           stride_a, b, b_type, ldb, stride_b, beta, c, c_type, ldc,
                                           stride_c, batch, compute, algo);
}
}
// NOLINTEND(bugprone-reserved-identifier,misc-use-internal-linkage,readability-identifier-naming)
