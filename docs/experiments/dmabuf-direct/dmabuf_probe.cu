// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// Standalone probe for docs/experiments/dmabuf-direct: can a direct read land
// in GPU device memory that the driver exports as a dma-buf and the CPU maps?
// Fatal errors exit; process teardown owns cleanup of whatever is still held.
// Build (on the Spark): see build.sh.
#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/io_uring.h>
#include <linux/memfd.h>
#include <linux/udmabuf.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dirent.h>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
constexpr std::size_t MiB = std::size_t{1} << 20, GiB = std::size_t{1} << 30;
constexpr std::size_t kExtent = 2 * MiB;
constexpr int kThreads = 256;

// ---------------------------------------------------------------- utilities

void require(bool ok, const char* what) {
  if (ok) return;
  std::fprintf(stderr, "FATAL %s (errno=%d %s)\n", what, errno, std::strerror(errno));
  std::exit(1);
}
const char* cu_name(CUresult r) {
  const char* name = nullptr;
  cuGetErrorName(r, &name);
  return name ? name : "unknown";
}
void check(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  std::fprintf(stderr, "FATAL %s: %s\n", what, cu_name(r));
  std::exit(1);
}
void rt_check(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return;
  std::fprintf(stderr, "FATAL %s: %s\n", what, cudaGetErrorName(e));
  std::exit(1);
}
#define CU(x) check((x), #x)
#define RT(x) rt_check((x), #x)

double seconds_since(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}
double us_since(Clock::time_point t) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t).count();
}
double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(p * static_cast<double>(v.size() - 1))];
}
unsigned long long number(const char* s) {
  char* end = nullptr;
  errno = 0;
  auto n = std::strtoull(s, &end, 10);
  require(!errno && end != s && *end == 0, "invalid integer");
  return n;
}
long long meminfo_kib(const char* field) {
  std::ifstream in("/proc/meminfo");
  std::string key, unit;
  long long n = 0;
  while (in >> key >> n >> unit)
    if (key == field) return n;
  return -1;
}
int open_fds() {
  int n = 0;
  DIR* d = opendir("/proc/self/fd");
  if (!d) return -1;
  while (readdir(d)) ++n;
  closedir(d);
  return n - 3;  // ".", "..", and the directory's own fd
}
double cpu_seconds() {
  rusage u{};
  getrusage(RUSAGE_SELF, &u);
  return static_cast<double>(u.ru_utime.tv_sec + u.ru_stime.tv_sec) +
         static_cast<double>(u.ru_utime.tv_usec + u.ru_stime.tv_usec) / 1e6;
}

// The file and every verified buffer hold pattern(word index) at each 32-bit
// word, so the GPU can check any landed range without a second copy.
__host__ __device__ inline std::uint32_t pattern(std::uint64_t index) {
  std::uint32_t x = static_cast<std::uint32_t>(index) ^ static_cast<std::uint32_t>(index >> 32) ^
                    0x9e3779b9U;
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  return x ^ (x >> 16);
}

// Header-line fields of this process's smaps entry that starts at addr.
std::string smaps_info(const void* addr) {
  char start[32];
  std::snprintf(start, sizeof start, "%lx-", reinterpret_cast<unsigned long>(addr));
  std::ifstream in("/proc/self/smaps");
  std::string line, out;
  bool inside = false;
  while (std::getline(in, line)) {
    auto space = line.find(' ');
    bool header = space != std::string::npos && line[space - 1] != ':';
    if (header) {
      if (inside) break;
      inside = line.rfind(start, 0) == 0;
      continue;
    }
    if (!inside) continue;
    for (const char* key : {"Rss:", "KernelPageSize:", "VmFlags:"}) {
      if (line.rfind(key, 0) == 0) {
        std::string v = line.substr(std::strlen(key));
        v.erase(0, v.find_first_not_of(' '));
        while (!v.empty() && v.back() == ' ') v.pop_back();
        std::string k = key;
        k.pop_back();
        out += (out.empty() ? "" : " ") + k + "=[" + v + "]";
      }
    }
  }
  return inside || !out.empty() ? out : "not-found";
}

// ---------------------------------------------------------------- kernels

__device__ std::uint64_t tid() { return static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; }
__device__ std::uint64_t stride() { return static_cast<std::uint64_t>(gridDim.x) * blockDim.x; }

__global__ void fill_kernel(std::uint32_t* d, std::uint64_t words, std::uint64_t first) {
  for (std::uint64_t i = tid(); i < words; i += stride()) d[i] = pattern(first + i);
}
__global__ void verify_kernel(const std::uint32_t* d, std::uint64_t words, std::uint64_t first,
                              unsigned long long* bad) {
  unsigned long long n = 0;
  for (std::uint64_t i = tid(); i < words; i += stride()) n += d[i] != pattern(first + i);
  if (n) atomicAdd(bad, n);
}
// The host-VMM diagnosis's reread (benchmarks/vmm_diag_kernels.cu), unchanged.
__global__ void reread_kernel(const uint4* data, std::uint64_t count, int passes, std::uint32_t* out) {
  std::uint32_t sum = 0;
  for (int pass = 0; pass < passes; ++pass) {
    const std::uint64_t shift = (static_cast<std::uint64_t>(pass) * 7919U * kThreads) % count;
    for (std::uint64_t i = tid(); i < count; i += stride()) {
      std::uint64_t at = i + shift;
      if (at >= count) at -= count;
      const uint4 v = __ldcg(data + at);
      sum += v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  out[tid()] = sum;
}
__global__ void copy_kernel(uint4* to, const uint4* from, std::uint64_t count) {
  for (std::uint64_t i = tid(); i < count; i += stride()) to[i] = from[i];
}
// The diagnosis's other microkernels (benchmarks/vmm_diag_kernels.cu).
__device__ std::uint32_t mix(std::uint64_t index, std::uint32_t seed) {
  auto x = static_cast<std::uint32_t>(index) ^ static_cast<std::uint32_t>(index >> 32U) ^ (seed * 0x9e3779b9U);
  x ^= x >> 16U;
  x *= 0x7feb352dU;
  x ^= x >> 15U;
  x *= 0x846ca68bU;
  return x ^ (x >> 16U);
}
__global__ void scan_vector_kernel(const uint4* data, std::uint64_t count, std::uint32_t* out) {
  std::uint32_t sum = 0;
  for (std::uint64_t i = tid(); i < count; i += stride()) {
    const uint4 v = data[i];
    sum += v.x ^ v.y ^ v.z ^ v.w;
  }
  out[tid()] = sum;
}
__global__ void write_vector_kernel(uint4* data, std::uint64_t count) {
  for (std::uint64_t i = tid(); i < count; i += stride()) {
    const auto x = static_cast<std::uint32_t>(i);
    data[i] = make_uint4(x, x + 1U, x + 2U, x + 3U);
  }
}
__global__ void write_per_block_kernel(std::uint32_t* data) {
  if (threadIdx.x == 0) data[blockIdx.x] = blockIdx.x;
}
__global__ void random_lines_kernel(const std::uint32_t* data, std::uint64_t lines, std::uint64_t window_lines,
                                    int run, int per_warp, std::uint32_t* out) {
  const std::uint64_t warp = tid() / 32U;
  const std::uint32_t lane = threadIdx.x % 32U;
  std::uint32_t sum = 0;
  std::uint64_t region = 0;
  for (int j = 0; j < per_warp; ++j) {
    std::uint64_t line = 0;
    const std::uint64_t r = (static_cast<std::uint64_t>(mix((warp << 20U) + static_cast<std::uint64_t>(j), 17U)) << 32U) |
                            mix((warp << 20U) + static_cast<std::uint64_t>(j), 91U);
    if (window_lines == 0) {
      line = r % lines;
    } else {
      if (j % run == 0) region = (r >> 17U) % (lines / window_lines);
      line = (region * window_lines) + (r % window_lines);
    }
    sum += data[(line * 32U) + lane];
  }
  out[tid()] = sum;
}

// ---------------------------------------------------------------- GPU state

struct Gpu {
  CUdevice dev{};
  CUcontext ctx{};
  int sms{};
  cudaStream_t stream{};
  unsigned long long* bad{};
  std::uint32_t* out{};
} g;

void gpu_init() {
  CU(cuInit(0));
  CU(cuDeviceGet(&g.dev, 0));
  CU(cuDevicePrimaryCtxRetain(&g.ctx, g.dev));
  CU(cuCtxSetCurrent(g.ctx));
  RT(cudaSetDevice(0));
  CU(cuDeviceGetAttribute(&g.sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, g.dev));
  RT(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
  RT(cudaMalloc(&g.bad, sizeof(unsigned long long)));
  RT(cudaMalloc(&g.out, static_cast<std::size_t>(g.sms) * 16 * kThreads * 4));
}
int blocks() { return g.sms * 16; }
unsigned long long gpu_verify(CUdeviceptr p, std::size_t bytes, std::uint64_t first_word) {
  RT(cudaMemsetAsync(g.bad, 0, sizeof *g.bad, g.stream));
  verify_kernel<<<blocks(), kThreads, 0, g.stream>>>(reinterpret_cast<const std::uint32_t*>(p), bytes / 4,
                                                     first_word, g.bad);
  RT(cudaGetLastError());
  unsigned long long bad = 0;
  RT(cudaMemcpyAsync(&bad, g.bad, sizeof bad, cudaMemcpyDeviceToHost, g.stream));
  RT(cudaStreamSynchronize(g.stream));
  return bad;
}
void gpu_fill(CUdeviceptr p, std::size_t bytes, std::uint64_t first_word) {
  fill_kernel<<<blocks(), kThreads, 0, g.stream>>>(reinterpret_cast<std::uint32_t*>(p), bytes / 4, first_word);
  RT(cudaGetLastError());
  RT(cudaStreamSynchronize(g.stream));
}
void gpu_clear(CUdeviceptr p, std::size_t bytes) {
  CU(cuMemsetD8Async(p, 0, bytes, g.stream));
  RT(cudaStreamSynchronize(g.stream));
}
std::uint64_t cpu_verify(const void* p, std::size_t bytes, std::uint64_t first_word) {
  auto* w = static_cast<const volatile std::uint32_t*>(p);
  std::uint64_t bad = 0;
  for (std::size_t i = 0; i < bytes / 4; ++i) bad += w[i] != pattern(first_word + i);
  return bad;
}
void cpu_fill(void* p, std::size_t bytes, std::uint64_t first_word) {
  auto* w = static_cast<std::uint32_t*>(p);
  for (std::size_t i = 0; i < bytes / 4; ++i) w[i] = pattern(first_word + i);
}

// ---------------------------------------------------------------- VMM

struct VmmSpec {
  bool host = false;               // HOST_NUMA 0 instead of the device
  std::size_t handle_bytes = 0;    // 0: one handle for the whole range
  bool cpu_access = false;         // cuMemSetAccess for HOST_NUMA 0 as well
  CUmemAllocationHandleType types = CU_MEM_HANDLE_TYPE_NONE;
  bool rdma = false;               // allocFlags.gpuDirectRDMACapable
};
struct Vmm {
  CUdeviceptr va{};
  std::size_t bytes{};
  std::vector<CUmemGenericAllocationHandle> handles;
  bool mapped = false;
};
CUresult vmm_alloc(Vmm& v, std::size_t bytes, const VmmSpec& s, std::string* step) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = s.types;
  prop.location.type = s.host ? CU_MEM_LOCATION_TYPE_HOST_NUMA : CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = s.host ? 0 : g.dev;
  prop.allocFlags.gpuDirectRDMACapable = s.rdma ? 1 : 0;
  std::size_t hb = s.handle_bytes ? s.handle_bytes : bytes;
  v.bytes = bytes;
  CUresult r = cuMemAddressReserve(&v.va, bytes, 2 * MiB, 0, 0);
  if (r) { *step = "cuMemAddressReserve"; return r; }
  for (std::size_t off = 0; off < bytes; off += hb) {
    CUmemGenericAllocationHandle h{};
    r = cuMemCreate(&h, hb, &prop, 0);
    if (r) { *step = "cuMemCreate"; return r; }
    v.handles.push_back(h);
    r = cuMemMap(v.va + off, hb, 0, h, 0);
    if (r) { *step = "cuMemMap"; return r; }
    v.mapped = true;
  }
  CUmemAccessDesc d[2]{};
  d[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  d[0].location.id = g.dev;
  d[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  d[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  d[1].location.id = 0;
  d[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  r = cuMemSetAccess(v.va, bytes, d, s.cpu_access ? 2 : 1);
  if (r) { *step = "cuMemSetAccess"; return r; }
  return CUDA_SUCCESS;
}
void vmm_free(Vmm& v) {
  if (v.mapped) CU(cuMemUnmap(v.va, v.bytes));
  for (auto h : v.handles) CU(cuMemRelease(h));
  if (v.va) CU(cuMemAddressFree(v.va, v.bytes));
  v = {};
}
Vmm vmm_or_die(std::size_t bytes, const VmmSpec& s) {
  Vmm v;
  std::string step;
  CUresult r = vmm_alloc(v, bytes, s, &step);
  if (r) { std::fprintf(stderr, "FATAL vmm %s: %s\n", step.c_str(), cu_name(r)); std::exit(1); }
  return v;
}

CUresult export_dmabuf(CUdeviceptr p, std::size_t bytes, unsigned long long flags, int* fd) {
  *fd = -1;
  return cuMemGetHandleForAddressRange(fd, p, bytes, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags);
}

// ---------------------------------------------------------------- io_uring

// Minimal io_uring ABI use (as docs/experiments/io-path): one owner of both
// rings, acquire/release on the shared indexes.
struct Ring {
  int fd = -1;
  void* rings{};
  io_uring_sqe* sqes{};
  std::size_t ring_bytes{}, sqe_bytes{};
  unsigned *sqhead{}, *sqtail{}, *sqmask{}, *array{}, *cqhead{}, *cqtail{}, *cqmask{};
  io_uring_cqe* cqes{};
  unsigned pending = 0, in_flight = 0;
  static unsigned acquire(unsigned* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
  static void release(unsigned* p, unsigned v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
  void init(unsigned entries) {
    io_uring_params p{};
    fd = static_cast<int>(syscall(__NR_io_uring_setup, entries, &p));
    require(fd >= 0, "io_uring_setup");
    require(p.features & IORING_FEAT_SINGLE_MMAP, "io_uring needs SINGLE_MMAP");
    ring_bytes = std::max(p.sq_off.array + p.sq_entries * sizeof(unsigned),
                          p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe));
    rings = mmap(nullptr, ring_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, IORING_OFF_SQ_RING);
    require(rings != MAP_FAILED, "ring mmap");
    sqe_bytes = p.sq_entries * sizeof(io_uring_sqe);
    sqes = static_cast<io_uring_sqe*>(mmap(nullptr, sqe_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, IORING_OFF_SQES));
    require(sqes != MAP_FAILED, "sqe mmap");
    auto at = [&](unsigned off) { return reinterpret_cast<unsigned*>(static_cast<char*>(rings) + off); };
    sqhead = at(p.sq_off.head); sqtail = at(p.sq_off.tail); sqmask = at(p.sq_off.ring_mask);
    array = at(p.sq_off.array); cqhead = at(p.cq_off.head); cqtail = at(p.cq_off.tail);
    cqmask = at(p.cq_off.ring_mask);
    cqes = reinterpret_cast<io_uring_cqe*>(static_cast<char*>(rings) + p.cq_off.cqes);
  }
  void prep_read(int file, void* addr, unsigned len, std::uint64_t off, std::uint64_t ud, int fixed = -1) {
    auto tail = acquire(sqtail);
    require(tail - acquire(sqhead) <= *sqmask, "submission queue full");
    auto index = tail & *sqmask;
    auto& s = sqes[index];
    s = {};
    s.opcode = fixed >= 0 ? IORING_OP_READ_FIXED : IORING_OP_READ;
    s.fd = file;
    s.off = off;
    s.addr = reinterpret_cast<std::uint64_t>(addr);
    s.len = len;
    if (fixed >= 0) s.buf_index = static_cast<unsigned short>(fixed);
    s.user_data = ud;
    array[index] = index;
    release(sqtail, tail + 1);
    ++pending;
  }
  void enter(unsigned min_complete) {
    unsigned flags = min_complete ? IORING_ENTER_GETEVENTS : 0;
    if (!pending && !min_complete) return;
    long n;
    do n = syscall(__NR_io_uring_enter, fd, pending, min_complete, flags, nullptr, 0);
    while (n < 0 && errno == EINTR);
    require(n >= 0 && static_cast<unsigned>(n) == pending, "io_uring_enter");
    in_flight += pending;
    pending = 0;
  }
  bool reap(std::uint64_t& ud, int& res) {
    auto head = acquire(cqhead);
    if (head == acquire(cqtail)) return false;
    auto c = cqes[head & *cqmask];
    ud = c.user_data;
    res = c.res;
    release(cqhead, head + 1);
    --in_flight;
    return true;
  }
  int register_buffers(const iovec* v, unsigned n) {
    long r = syscall(__NR_io_uring_register, fd, IORING_REGISTER_BUFFERS, v, n);
    return r < 0 ? -errno : static_cast<int>(r);
  }
  int unregister_buffers() {
    long r = syscall(__NR_io_uring_register, fd, IORING_UNREGISTER_BUFFERS, nullptr, 0);
    return r < 0 ? -errno : static_cast<int>(r);
  }
  void close_ring() {
    close(fd);
    munmap(sqes, sqe_bytes);
    munmap(rings, ring_bytes);
    fd = -1;
  }
};

// One synchronous io_uring read; returns cqe.res (bytes or -errno).
int uring_read_once(Ring& ring, int file, void* addr, unsigned len, std::uint64_t off, int fixed = -1) {
  ring.prep_read(file, addr, len, off, 1, fixed);
  ring.enter(1);
  std::uint64_t ud;
  int res;
  while (!ring.reap(ud, res)) ring.enter(1);
  return res;
}

std::string res_str(long r) {
  if (r >= 0) return std::to_string(r);
  return std::string("-") + std::to_string(-r) + "(" + std::strerror(static_cast<int>(-r)) + ")";
}

// ---------------------------------------------------------------- create

// Writes the pattern file with O_DIRECT (so it is not left in the page cache).
int cmd_create(const char* path, std::size_t gib) {
  int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY | O_DIRECT | O_CLOEXEC, 0644);
  require(fd >= 0, "create file");
  constexpr std::size_t chunk = 64 * MiB;
  void* buf = aligned_alloc(4096, chunk);
  require(buf, "buffer");
  for (std::size_t off = 0; off < gib * GiB; off += chunk) {
    auto* w = static_cast<std::uint32_t*>(buf);
    const std::uint64_t first = off / 4;
    std::vector<std::thread> t;
    for (int k = 0; k < 8; ++k)
      t.emplace_back([&, k] {
        for (std::size_t i = k * (chunk / 32); i < (k + 1) * (chunk / 32); ++i) w[i] = pattern(first + i);
      });
    for (auto& x : t) x.join();
    require(pwrite(fd, buf, chunk, static_cast<off_t>(off)) == static_cast<ssize_t>(chunk), "pwrite");
  }
  require(fsync(fd) == 0 && close(fd) == 0, "fsync/close");
  std::printf("created %s %zu GiB\n", path, gib);
  return 0;
}

// ---------------------------------------------------------------- info

int cmd_info() {
  gpu_init();
  int drv = 0;
  CU(cuDriverGetVersion(&drv));
  char name[256]{};
  CU(cuDeviceGetName(name, sizeof name, g.dev));
  std::printf("device=%s driver_api=%d runtime_build=%d sms=%d\n", name, drv, CUDART_VERSION, g.sms);
  struct A { int id; const char* name; } attrs[] = {
      {CU_DEVICE_ATTRIBUTE_INTEGRATED, "INTEGRATED"},
      {CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, "VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED"},
      {CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, "HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED"},
      {CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, "GPU_DIRECT_RDMA_SUPPORTED"},
      {CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, "GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED"},
      {CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS, "GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS"},
      {CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, "GPU_DIRECT_RDMA_WRITES_ORDERING"},
      {CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, "DMA_BUF_SUPPORTED(124)"},
      {CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED, "HOST_ALLOC_DMA_BUF_SUPPORTED(146)"},
      {152, "DMA_BUF_MMAP_SUPPORTED(152, CUDA 13.4 header)"},
      {CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, "PAGEABLE_MEMORY_ACCESS"},
      {CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, "PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES"},
      {CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, "HOST_NATIVE_ATOMIC_SUPPORTED"},
      {CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, "CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM"},
      {CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, "HOST_NUMA_ID"},
      {CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, "L2_CACHE_SIZE"},
  };
  for (auto a : attrs) {
    int v = -1;
    CUresult r = cuDeviceGetAttribute(&v, static_cast<CUdevice_attribute>(a.id), g.dev);
    std::printf("attr %s = %s\n", a.name, r ? cu_name(r) : std::to_string(v).c_str());
  }
  for (bool host : {false, true}) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = host ? CU_MEM_LOCATION_TYPE_HOST_NUMA : CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = host ? 0 : g.dev;
    std::size_t mn = 0, rec = 0;
    CU(cuMemGetAllocationGranularity(&mn, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    CU(cuMemGetAllocationGranularity(&rec, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    std::printf("granularity %s min=%zu recommended=%zu\n", host ? "host_numa" : "device", mn, rec);
  }
  return 0;
}

// ---------------------------------------------------------------- feasibility

// What each I/O path does when its destination is `m` (a CPU address whose
// GPU view is `gpu`): O_DIRECT pread/pwrite, io_uring READ, registered
// buffers + READ_FIXED, and a buffered pread (a CPU copy from the page cache).
void io_tests(const char* label, void* m, CUdeviceptr gpu, const char* file, const char* dir) {
  const std::uint64_t off = 64 * MiB;
  int dfd = open(file, O_RDONLY | O_DIRECT | O_CLOEXEC);
  int bfd = open(file, O_RDONLY | O_CLOEXEC);
  require(dfd >= 0 && bfd >= 0, "open file");
  auto verify = [&](std::size_t at) {
    return gpu ? gpu_verify(gpu + at, kExtent, off / 4) : cpu_verify(static_cast<char*>(m) + at, kExtent, off / 4);
  };
  auto clear = [&] {
    if (gpu) gpu_clear(gpu, 2 * kExtent); else std::memset(m, 0, 2 * kExtent);
  };
  clear();
  errno = 0;
  long r = pread(dfd, m, kExtent, off);
  long e = r < 0 ? -errno : r;
  std::printf("io %s odirect_pread=%s", label, res_str(e).c_str());
  if (r == static_cast<long>(kExtent)) std::printf(" bad_words=%llu", static_cast<unsigned long long>(verify(0)));
  std::printf("\n");

  {
    int wfd = open(dir, O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
    require(wfd >= 0, "tmpfile");
    errno = 0;
    long w = pwrite(wfd, m, kExtent, 0);
    std::printf("io %s odirect_pwrite_from=%s\n", label, res_str(w < 0 ? -errno : w).c_str());
    close(wfd);
  }

  Ring ring;
  ring.init(8);
  clear();
  int u = uring_read_once(ring, dfd, m, kExtent, off);
  std::printf("io %s uring_read_odirect=%s", label, res_str(u).c_str());
  if (u == static_cast<int>(kExtent)) std::printf(" bad_words=%llu", static_cast<unsigned long long>(verify(0)));
  std::printf("\n");

  iovec v{m, kExtent};
  int reg = ring.register_buffers(&v, 1);
  std::printf("io %s uring_register_buffers=%s", label, res_str(reg).c_str());
  if (reg == 0) {
    clear();
    int f = uring_read_once(ring, dfd, m, kExtent, off, 0);
    std::printf(" read_fixed=%s", res_str(f).c_str());
    if (f == static_cast<int>(kExtent)) std::printf(" bad_words=%llu", static_cast<unsigned long long>(verify(0)));
    ring.unregister_buffers();
  }
  std::printf("\n");
  ring.close_ring();

  clear();
  posix_fadvise(bfd, 0, 0, POSIX_FADV_DONTNEED);
  errno = 0;
  long b = pread(bfd, static_cast<char*>(m) + kExtent, kExtent, off);
  std::printf("io %s buffered_pread=%s", label, res_str(b < 0 ? -errno : b).c_str());
  if (b == static_cast<long>(kExtent)) std::printf(" bad_words=%llu", static_cast<unsigned long long>(verify(kExtent)));
  std::printf("\n");
  posix_fadvise(bfd, 0, 0, POSIX_FADV_DONTNEED);
  close(dfd);
  close(bfd);
}

// Coherence between CPU stores/loads through the mapping and GPU kernels.
void coherence_tests(const char* label, void* m, CUdeviceptr gpu, std::size_t bytes) {
  // 1. GPU writes, CPU reads (after the CPU had the old lines cached).
  cpu_fill(m, bytes, 0);
  (void)cpu_verify(m, bytes, 0);
  gpu_fill(gpu, bytes, 1ULL << 32);
  std::printf("coh %s gpu_write_cpu_read_bad=%llu\n", label,
              static_cast<unsigned long long>(cpu_verify(m, bytes, 1ULL << 32)));
  // 2. CPU writes, GPU reads, with the old lines primed in L2 by a reread.
  gpu_fill(gpu, bytes, 2ULL << 32);
  const std::size_t hot = std::min<std::size_t>(bytes, 4 * MiB);
  reread_kernel<<<blocks(), kThreads, 0, g.stream>>>(reinterpret_cast<const uint4*>(gpu), hot / 16, 8, g.out);
  RT(cudaStreamSynchronize(g.stream));
  cpu_fill(m, bytes, 3ULL << 32);
  std::printf("coh %s cpu_write_gpu_read_l2primed_bad=%llu\n", label,
              static_cast<unsigned long long>(gpu_verify(gpu, bytes, 3ULL << 32)));
  // 3. The dma-buf CPU-access bracket, if the exporter implements it.
}

double bw_run(int threads, std::size_t bytes, const std::function<void(std::size_t, std::size_t)>& fn) {
  std::atomic<int> go{0};
  std::vector<std::thread> t;
  std::atomic<int> ready{0};
  for (int k = 0; k < threads; ++k)
    t.emplace_back([&, k] {
      std::size_t per = bytes / threads;
      ready.fetch_add(1);
      while (!go.load(std::memory_order_acquire)) {}
      fn(k * per, per);
    });
  while (ready.load() != threads) {}
  auto start = Clock::now();
  go.store(1, std::memory_order_release);
  for (auto& x : t) x.join();
  return static_cast<double>(bytes) / seconds_since(start) / 1e9;
}
volatile std::uint64_t sink;
void read_range(const void* p, std::size_t begin, std::size_t len) {
  auto* w = reinterpret_cast<const std::uint64_t*>(static_cast<const char*>(p) + begin);
  std::uint64_t a = 0, b = 0, c = 0, d = 0;
  for (std::size_t i = 0; i < len / 8; i += 4) { a += w[i]; b += w[i + 1]; c += w[i + 2]; d += w[i + 3]; }
  sink = a + b + c + d;
}
struct Stat { double med, lo, hi; };
Stat bw_stat(int threads, std::size_t bytes, int reps, const std::function<void(std::size_t, std::size_t)>& fn) {
  std::vector<double> v;
  for (int i = 0; i < reps + 2; ++i) {
    double x = bw_run(threads, bytes, fn);
    if (i >= 2) v.push_back(x);
  }
  return {pct(v, 0.5), pct(v, 0), pct(v, 1)};
}
// CPU bandwidth through a mapping against ordinary touched memory.
void cpu_bw(const char* label, void* m, std::size_t bytes, int reps) {
  void* src = aligned_alloc(2 * MiB, bytes);
  require(src, "alloc");
  std::memset(src, 1, bytes);
  for (int threads : {1, 8}) {
    auto r = bw_stat(threads, bytes, reps, [&](std::size_t b, std::size_t n) { read_range(m, b, n); });
    auto w = bw_stat(threads, bytes, reps, [&](std::size_t b, std::size_t n) { std::memset(static_cast<char*>(m) + b, 7, n); });
    auto c = bw_stat(threads, bytes, reps, [&](std::size_t b, std::size_t n) {
      std::memcpy(static_cast<char*>(m) + b, static_cast<char*>(src) + b, n);
    });
    std::printf("cpubw %s threads=%d bytes=%zu reps=%d read_gbps=%.2f(%.2f-%.2f) write_gbps=%.2f(%.2f-%.2f) "
                "copy_in_gbps=%.2f(%.2f-%.2f)\n",
                label, threads, bytes, reps, r.med, r.lo, r.hi, w.med, w.lo, w.hi, c.med, c.lo, c.hi);
  }
  free(src);
}

int cmd_feasibility(const char* file, const char* dir) {
  gpu_init();
  const std::size_t bytes = 64 * MiB;
  struct Variant { const char* name; VmmSpec spec; } variants[] = {
      {"dvmm-1h", {false, 0, false, CU_MEM_HANDLE_TYPE_NONE, false}},
      {"dvmm-2m", {false, 2 * MiB, false, CU_MEM_HANDLE_TYPE_NONE, false}},
      {"dvmm-1h-posixfd", {false, 0, false, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, false}},
      {"dvmm-1h-rdma", {false, 0, false, CU_MEM_HANDLE_TYPE_NONE, true}},
      {"hvmm-1h", {true, 0, true, CU_MEM_HANDLE_TYPE_NONE, false}},
      {"hvmm-2m", {true, 2 * MiB, true, CU_MEM_HANDLE_TYPE_NONE, false}},
  };
  bool first_tested = false;
  for (auto& var : variants) {
    Vmm v;
    std::string step;
    CUresult r = vmm_alloc(v, bytes, var.spec, &step);
    if (r) {
      std::printf("variant %s alloc_failed step=%s %s\n", var.name, step.c_str(), cu_name(r));
      vmm_free(v);
      continue;
    }
    for (unsigned long long flags : {0ULL, static_cast<unsigned long long>(CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE)}) {
      int fd = -1;
      r = export_dmabuf(v.va, bytes, flags, &fd);
      std::printf("variant %s export flags=%llu -> %s fd=%d\n", var.name, flags, cu_name(r), fd);
      if (r) continue;
      for (int prot : {PROT_READ | PROT_WRITE, PROT_READ}) {
        errno = 0;
        void* m = mmap(nullptr, bytes, prot, MAP_SHARED, fd, 0);
        int err = errno;
        std::printf("variant %s flags=%llu mmap prot=%s -> %s", var.name, flags, prot == PROT_READ ? "r" : "rw",
                    m == MAP_FAILED ? (std::to_string(err) + "(" + std::strerror(err) + ")").c_str() : "ok");
        if (m == MAP_FAILED) { std::printf("\n"); continue; }
        std::printf(" smaps: %s\n", smaps_info(m).c_str());
        if (prot == (PROT_READ | PROT_WRITE) && flags == 0) {
          gpu_fill(v.va, bytes, 0);
          std::printf("variant %s cpu_reads_gpu_fill_bad=%llu\n", var.name,
                      static_cast<unsigned long long>(cpu_verify(m, bytes, 0)));
          dma_buf_sync sync{DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW};
          errno = 0;
          int s = ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
          std::printf("variant %s dma_buf_sync_start=%s\n", var.name, res_str(s < 0 ? -errno : s).c_str());
          if (s == 0) { sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW; ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync); }
          io_tests(var.name, m, v.va, file, dir);
          coherence_tests(var.name, m, v.va, bytes);
          if (!first_tested && !var.spec.host) {
            first_tested = true;
            cpu_bw("dmabuf-dvmm", m, bytes, 10);
          }
        }
        munmap(m, bytes);
      }
      close(fd);
    }
    // A wide export: 128 and 129 memory handles in one range.
    if (var.spec.handle_bytes == 2 * MiB && !var.spec.host) {
      for (std::size_t n : {128, 129, 4096}) {
        Vmm w;
        VmmSpec s = var.spec;
        r = vmm_alloc(w, n * 2 * MiB, s, &step);
        int fd = -1;
        if (!r) r = export_dmabuf(w.va, n * 2 * MiB, 0, &fd);
        std::printf("variant %s wide_export handles=%zu -> %s", var.name, n, cu_name(r));
        if (!r) {
          void* m = mmap(nullptr, n * 2 * MiB, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
          std::printf(" mmap=%s", m == MAP_FAILED ? std::strerror(errno) : "ok");
          if (m != MAP_FAILED) {
            gpu_fill(w.va, n * 2 * MiB, 0);
            std::printf(" cpu_bad=%llu", static_cast<unsigned long long>(cpu_verify(m, n * 2 * MiB, 0)));
            munmap(m, n * 2 * MiB);
          }
          close(fd);
        }
        std::printf("\n");
        vmm_free(w);
      }
    }
    vmm_free(v);
  }
  // Non-VMM allocations, for comparison.
  {
    CUdeviceptr p{};
    CU(cuMemAlloc(&p, bytes));
    int fd = -1;
    CUresult r = export_dmabuf(p, bytes, 0, &fd);
    std::printf("variant cuMemAlloc export -> %s\n", cu_name(r));
    if (!r) {
      void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      std::printf("variant cuMemAlloc mmap -> %s", m == MAP_FAILED ? std::strerror(errno) : "ok");
      if (m != MAP_FAILED) {
        std::printf(" smaps: %s\n", smaps_info(m).c_str());
        io_tests("cuMemAlloc", m, p, file, dir);
        munmap(m, bytes);
      } else {
        std::printf("\n");
      }
      close(fd);
    }
    CU(cuMemFree(p));
  }
  {
    void* h{};
    CU(cuMemAllocHost(&h, bytes));
    CUdeviceptr p{};
    CU(cuMemHostGetDevicePointer(&p, h, 0));
    int fd = -1;
    CUresult r = export_dmabuf(p, bytes, 0, &fd);
    std::printf("variant cuMemAllocHost export -> %s\n", cu_name(r));
    if (!r) {
      void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      std::printf("variant cuMemAllocHost mmap -> %s", m == MAP_FAILED ? std::strerror(errno) : "ok");
      if (m != MAP_FAILED) {
        std::printf(" smaps: %s\n", smaps_info(m).c_str());
        // The same driver mmap path as a device export would take: do the
        // direct-I/O paths accept an NVIDIA dma-buf mapping on this kernel?
        io_tests("cuMemAllocHost-dmabuf", m, p, file, dir);
        // Is NVIDIA's dma-buf mmap a cached CPU mapping here?
        cpu_bw("cuMemAllocHost-dmabuf", m, bytes, 10);
        munmap(m, bytes);
      } else {
        std::printf("\n");
      }
      close(fd);
    }
    CU(cuMemFreeHost(h));
  }
  // The VMM's own shareable handle (an opaque fd): can it be mapped?
  {
    Vmm v;
    std::string step;
    CUresult r = vmm_alloc(v, bytes, {false, 0, false, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, false}, &step);
    int fd = -1;
    if (!r) r = cuMemExportToShareableHandle(&fd, v.handles[0], CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
    std::printf("variant dvmm-posixfd shareable_export -> %s", cu_name(r));
    if (!r) {
      char link[256]{};
      std::string path = "/proc/self/fd/" + std::to_string(fd);
      if (readlink(path.c_str(), link, sizeof link - 1) < 0) link[0] = 0;
      errno = 0;
      void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      std::printf(" fd_target=%s mmap=%s", link, m == MAP_FAILED ? std::strerror(errno) : "ok");
      if (m != MAP_FAILED) munmap(m, bytes);
      close(fd);
    }
    std::printf("\n");
    vmm_free(v);
  }
  // Ordinary memory, the baseline for the CPU bandwidth figures.
  {
    void* m = aligned_alloc(2 * MiB, bytes);
    std::memset(m, 1, bytes);
    cpu_bw("malloc", m, bytes, 10);
    free(m);
  }
  {
    Vmm v = vmm_or_die(bytes, {true, 0, true, CU_MEM_HANDLE_TYPE_NONE, false});
    std::memset(reinterpret_cast<void*>(v.va), 1, bytes);
    cpu_bw("hvmm", reinterpret_cast<void*>(v.va), bytes, 10);
    vmm_free(v);
  }
  return 0;
}

// ---------------------------------------------------------------- udmabuf

// Ordinary shared memory (a memfd of 4 KiB shmem pages or 2 MiB hugetlb
// pages), exported by the kernel's udmabuf driver and imported into CUDA as
// external memory. The CPU mapping is ordinary memory, so direct reads can
// land in it; the GPU reads the same pages through the imported pointer.
struct UBuf {
  int memfd = -1;
  char* cpu = nullptr;
  std::size_t bytes = 0, chunk = 0;
  std::vector<int> dmafds;
  std::vector<CUexternalMemory> ext;
  std::vector<CUdeviceptr> gpu;
};
std::size_t udmabuf_limit() {
  std::ifstream in("/sys/module/udmabuf/parameters/size_limit_mb");
  std::size_t mb = 0;
  in >> mb;
  return mb * MiB;
}
bool ubuf_alloc(UBuf& u, std::size_t bytes, bool huge, std::size_t chunk, std::string* why) {
  u.bytes = bytes;
  u.chunk = chunk;
  unsigned flags = MFD_ALLOW_SEALING | MFD_CLOEXEC | (huge ? MFD_HUGETLB | MFD_HUGE_2MB : 0U);
  u.memfd = static_cast<int>(syscall(SYS_memfd_create, "llmp-udmabuf", flags));
  if (u.memfd < 0) { *why = std::string("memfd_create ") + std::strerror(errno); return false; }
  if (ftruncate(u.memfd, static_cast<off_t>(bytes)) != 0) { *why = std::string("ftruncate ") + std::strerror(errno); return false; }
  if (fcntl(u.memfd, F_ADD_SEALS, F_SEAL_SHRINK) != 0) { *why = std::string("seal ") + std::strerror(errno); return false; }
  void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, u.memfd, 0);
  if (m == MAP_FAILED) { *why = std::string("mmap ") + std::strerror(errno); return false; }
  u.cpu = static_cast<char*>(m);
  int dev = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
  if (dev < 0) { *why = std::string("/dev/udmabuf ") + std::strerror(errno); return false; }
  for (std::size_t off = 0; off < bytes; off += chunk) {
    const std::size_t n = std::min(chunk, bytes - off);
    udmabuf_create c{static_cast<__u32>(u.memfd), UDMABUF_FLAGS_CLOEXEC, off, n};
    int fd = ioctl(dev, UDMABUF_CREATE, &c);
    if (fd < 0) { *why = std::string("UDMABUF_CREATE ") + std::strerror(errno); close(dev); return false; }
    u.dmafds.push_back(fd);
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC d{};
    d.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD;
    d.handle.fd = fd;
    d.size = n;
    CUexternalMemory e{};
    CUresult r = cuImportExternalMemory(&e, &d);
    if (r) { *why = std::string("cuImportExternalMemory ") + cu_name(r); close(dev); return false; }
    u.ext.push_back(e);
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC b{};
    b.size = n;
    CUdeviceptr p{};
    r = cuExternalMemoryGetMappedBuffer(&p, e, &b);
    if (r) { *why = std::string("cuExternalMemoryGetMappedBuffer ") + cu_name(r); close(dev); return false; }
    u.gpu.push_back(p);
  }
  close(dev);
  return true;
}
// Order: the GPU mapping, the import, the dma-buf, the CPU mapping, the memfd.
void ubuf_free(UBuf& u) {
  for (auto p : u.gpu) CU(cuMemFree(p));
  for (auto e : u.ext) CU(cuDestroyExternalMemory(e));
  for (int fd : u.dmafds) close(fd);
  if (u.cpu) munmap(u.cpu, u.bytes);
  if (u.memfd >= 0) close(u.memfd);
  u = {};
}
UBuf ubuf_or_die(std::size_t bytes, bool huge, std::size_t chunk) {
  UBuf u;
  std::string why;
  if (!ubuf_alloc(u, bytes, huge, chunk, &why)) { std::fprintf(stderr, "FATAL udmabuf: %s\n", why.c_str()); std::exit(1); }
  return u;
}
std::string fd_state(int fd) { return fcntl(fd, F_GETFD) < 0 ? std::string("closed(") + std::strerror(errno) + ")" : "open"; }

// What the import is: support, fd ownership, pointer attributes, whether a
// VMM handle can be imported from the dma-buf, direct reads, coherence.
int cmd_udmabuf(const char* file, bool huge) {
  gpu_init();
  const std::size_t bytes = 64 * MiB;
  std::printf("udmabuf pages=%s size_limit_mb=%zu\n", huge ? "2m-hugetlb" : "4k-shmem", udmabuf_limit() / MiB);
  UBuf u = ubuf_or_die(bytes, huge, bytes);
  std::printf("udmabuf import ok dmabuf_fd_after_import=%s gpu_ptr=0x%llx cpu_ptr=%p smaps: %s\n",
              fd_state(u.dmafds[0]).c_str(), static_cast<unsigned long long>(u.gpu[0]), static_cast<void*>(u.cpu),
              smaps_info(u.cpu).c_str());
  {
    unsigned type = 0;
    CUresult r = cuPointerGetAttribute(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, u.gpu[0]);
    std::printf("udmabuf pointer memory_type=%s (1 host, 2 device, 4 unified)\n",
                r ? cu_name(r) : std::to_string(type).c_str());
    int mapped = 0;
    r = cuPointerGetAttribute(&mapped, CU_POINTER_ATTRIBUTE_MAPPED, u.gpu[0]);
    std::printf("udmabuf pointer mapped=%s\n", r ? cu_name(r) : std::to_string(mapped).c_str());
    CUmemGenericAllocationHandle h{};
    r = cuMemRetainAllocationHandle(&h, reinterpret_cast<void*>(u.gpu[0]));
    std::printf("udmabuf cuMemRetainAllocationHandle -> %s\n", cu_name(r));
    if (!r) cuMemRelease(h);
  }
  // A VMM handle from the dma-buf fd, to map it into a reservation (D-006).
  {
    udmabuf_create c{static_cast<__u32>(u.memfd), UDMABUF_FLAGS_CLOEXEC, 0, 2 * MiB};
    int dev = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
    int fd = ioctl(dev, UDMABUF_CREATE, &c);
    close(dev);
    CUmemGenericAllocationHandle h{};
    CUresult r = cuMemImportFromShareableHandle(&h, reinterpret_cast<void*>(static_cast<std::intptr_t>(fd)),
                                                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    std::printf("udmabuf cuMemImportFromShareableHandle(dmabuf fd) -> %s\n", cu_name(r));
    if (!r) cuMemRelease(h);
    close(fd);
  }
  // Direct I/O into the CPU mapping, checked through the GPU pointer.
  {
    int dfd = open(file, O_RDONLY | O_DIRECT | O_CLOEXEC);
    require(dfd >= 0, "open");
    gpu_clear(u.gpu[0], bytes);
    long n = pread(dfd, u.cpu, kExtent, 64 * MiB);
    std::printf("udmabuf odirect_pread=%s gpu_bad=%llu\n", res_str(n < 0 ? -errno : n).c_str(),
                gpu_verify(u.gpu[0], kExtent, 64 * MiB / 4));
    Ring ring;
    ring.init(8);
    gpu_clear(u.gpu[0], bytes);
    int res = uring_read_once(ring, dfd, u.cpu + kExtent, kExtent, 128 * MiB);
    std::printf("udmabuf uring_read_odirect=%s gpu_bad=%llu\n", res_str(res).c_str(),
                gpu_verify(u.gpu[0] + kExtent, kExtent, 128 * MiB / 4));
    iovec v{u.cpu, bytes};
    int reg = ring.register_buffers(&v, 1);
    std::printf("udmabuf uring_register_buffers=%s", res_str(reg).c_str());
    if (reg == 0) {
      gpu_clear(u.gpu[0], bytes);
      res = uring_read_once(ring, dfd, u.cpu, kExtent, 64 * MiB, 0);
      std::printf(" read_fixed=%s gpu_bad=%llu", res_str(res).c_str(), gpu_verify(u.gpu[0], kExtent, 64 * MiB / 4));
      ring.unregister_buffers();
    }
    std::printf("\n");
    ring.close_ring();
    close(dfd);
  }
  ubuf_free(u);
  return 0;
}

// Coherence of the GPU's view with DMA and CPU writes while the old lines are
// held in L2 (primed by the reread kernel), and of the CPU's view with GPU
// writes. `gpu` and `cpu` are two views of the same bytes.
void coherence(const char* arm, char* cpu, CUdeviceptr gpu, int dfd, int rounds) {
  const std::size_t hot = 4 * MiB;
  unsigned long long dma_bad = 0, cpu_bad = 0, gpu_to_cpu_bad = 0, first_bad = 0;
  for (int i = 0; i < rounds; ++i) {
    const std::uint64_t a = (2 * i) * hot, b = (2 * i + 1) * hot;  // two file offsets
    // Old contents, read by the GPU through L2 (several passes).
    require(pread(dfd, cpu, hot, static_cast<off_t>(a)) == static_cast<ssize_t>(hot), "pread a");
    first_bad += gpu_verify(gpu, hot, a / 4);
    reread_kernel<<<blocks(), kThreads, 0, g.stream>>>(reinterpret_cast<const uint4*>(gpu), hot / 16, 8, g.out);
    RT(cudaStreamSynchronize(g.stream));
    // New contents by DMA (a direct read), then checked by the GPU.
    require(pread(dfd, cpu, hot, static_cast<off_t>(b)) == static_cast<ssize_t>(hot), "pread b");
    dma_bad += gpu_verify(gpu, hot, b / 4);
    // New contents by CPU stores, after another L2 prime.
    reread_kernel<<<blocks(), kThreads, 0, g.stream>>>(reinterpret_cast<const uint4*>(gpu), hot / 16, 8, g.out);
    RT(cudaStreamSynchronize(g.stream));
    cpu_fill(cpu, hot, (7ULL << 32) + i);
    cpu_bad += gpu_verify(gpu, hot, (7ULL << 32) + i);
    // GPU stores, read by the CPU (which had the lines cached).
    gpu_fill(gpu, hot, (9ULL << 32) + i);
    gpu_to_cpu_bad += cpu_verify(cpu, hot, (9ULL << 32) + i);
  }
  std::printf("coherence,%s,rounds=%d,bytes=%zu,initial_bad=%llu,dma_after_l2_prime_bad=%llu,"
              "cpu_after_l2_prime_bad=%llu,gpu_write_cpu_read_bad=%llu\n",
              arm, rounds, hot, first_bad, dma_bad, cpu_bad, gpu_to_cpu_bad);
}

// ---------------------------------------------------------------- micro

// The diagnosis's microkernels over 1 GiB of each kind, in one process:
// device VMM, host VMM, cudaMalloc-equivalent, and the udmabuf import.
int cmd_micro(const char* file, bool huge, int samples, int rotate) {
  gpu_init();
  const std::size_t bytes = GiB;
  struct Arm { std::string name; CUdeviceptr gpu; char* cpu; };
  std::vector<Arm> arms;
  Vmm d = vmm_or_die(bytes, {false, 2 * MiB, false, CU_MEM_HANDLE_TYPE_NONE, false});
  arms.push_back({"dvmm", d.va, nullptr});
  Vmm h = vmm_or_die(bytes, {true, 0, true, CU_MEM_HANDLE_TYPE_NONE, false});
  arms.push_back({"hvmm", h.va, reinterpret_cast<char*>(h.va)});
  CUdeviceptr mal{};
  CU(cuMemAlloc(&mal, bytes));
  arms.push_back({"malloc", mal, nullptr});
  UBuf u = ubuf_or_die(bytes, huge, bytes);
  arms.push_back({huge ? "udmabuf-2m" : "udmabuf-4k", u.gpu[0], u.cpu});
  std::rotate(arms.begin(), arms.begin() + (rotate % static_cast<int>(arms.size())), arms.end());
  for (auto& a : arms) gpu_fill(a.gpu, bytes, 0);
  cudaEvent_t e0, e1;
  RT(cudaEventCreate(&e0));
  RT(cudaEventCreate(&e1));
  constexpr std::size_t kScan = 256 * MiB;
  constexpr int kLines = 64;
  const std::uint64_t warps = static_cast<std::uint64_t>(blocks()) * kThreads / 32;
  const std::uint64_t random_bytes = warps * kLines * 128;
  constexpr unsigned kBlocks = 151936;
  struct Test { const char* name; std::uint64_t useful; };
  const Test tests[] = {{"scan16", kScan},          {"write16", kScan},
                        {"write-per-block", kBlocks * 4ULL}, {"reread-4m", 4 * MiB * 64},
                        {"reread-16m", 16 * MiB * 16}, {"reread-64m", 64 * MiB * 4},
                        {"rand-all", random_bytes},  {"rand-2m", random_bytes},
                        {"rand-8m-span", random_bytes}};
  std::printf("# micro,arm,test,samples,gbps_median,gbps_min,gbps_max,us_median\n");
  for (const auto& t : tests) {
    for (auto& a : arms) {
      std::vector<double> gbps, us;
      for (int i = 0; i < samples + 2; ++i) {
        RT(cudaEventRecord(e0, g.stream));
        const std::string n = t.name;
        if (n == "scan16") scan_vector_kernel<<<blocks(), kThreads, 0, g.stream>>>(reinterpret_cast<const uint4*>(a.gpu), kScan / 16, g.out);
        else if (n == "write16") write_vector_kernel<<<blocks(), kThreads, 0, g.stream>>>(reinterpret_cast<uint4*>(a.gpu + kScan), kScan / 16);
        else if (n == "write-per-block") write_per_block_kernel<<<kBlocks, kThreads, 0, g.stream>>>(reinterpret_cast<std::uint32_t*>(a.gpu + kScan));
        else if (n == "reread-4m") reread_kernel<<<blocks(), kThreads, 0, g.stream>>>(reinterpret_cast<const uint4*>(a.gpu), 4 * MiB / 16, 64, g.out);
        else if (n == "reread-16m") reread_kernel<<<blocks(), kThreads, 0, g.stream>>>(reinterpret_cast<const uint4*>(a.gpu), 16 * MiB / 16, 16, g.out);
        else if (n == "reread-64m") reread_kernel<<<blocks(), kThreads, 0, g.stream>>>(reinterpret_cast<const uint4*>(a.gpu), 64 * MiB / 16, 4, g.out);
        else if (n == "rand-all") random_lines_kernel<<<blocks(), kThreads, 0, g.stream>>>(reinterpret_cast<const std::uint32_t*>(a.gpu), bytes / 128, 0, 1, kLines, g.out);
        else if (n == "rand-2m") random_lines_kernel<<<blocks(), kThreads, 0, g.stream>>>(reinterpret_cast<const std::uint32_t*>(a.gpu), bytes / 128, 2 * MiB / 128, kLines, kLines, g.out);
        else if (n == "rand-8m-span") random_lines_kernel<<<blocks(), kThreads, 0, g.stream>>>(reinterpret_cast<const std::uint32_t*>(a.gpu), 8 * MiB / 128, 0, 1, kLines, g.out);
        RT(cudaGetLastError());
        RT(cudaEventRecord(e1, g.stream));
        RT(cudaEventSynchronize(e1));
        float ms = 0;
        RT(cudaEventElapsedTime(&ms, e0, e1));
        if (i >= 2) { gbps.push_back(static_cast<double>(t.useful) / (ms / 1e3) / 1e9); us.push_back(ms * 1e3); }
      }
      std::printf("micro,%s,%s,%d,%.1f,%.1f,%.1f,%.1f\n", a.name.c_str(), t.name, samples, pct(gbps, .5), pct(gbps, 0),
                  pct(gbps, 1), pct(us, .5));
    }
  }
  // Content check: the reads above left the first 256 MiB unchanged.
  for (auto& a : arms)
    std::printf("micro,%s,content_bad=%llu\n", a.name.c_str(), gpu_verify(a.gpu, kScan, 0));
  int dfd = open(file, O_RDONLY | O_DIRECT | O_CLOEXEC);
  require(dfd >= 0, "open");
  for (auto& a : arms)
    if (a.cpu) coherence(a.name.c_str(), a.cpu, a.gpu, dfd, 20);
  close(dfd);
  ubuf_free(u);
  vmm_free(d);
  vmm_free(h);
  CU(cuMemFree(mal));
  return 0;
}

// ---------------------------------------------------------------- restore

// Modes: hvmm-inplace (D-034 as written), land-ce / land-sm (host-VMM landing
// zone of 2 x depth slots copied into device VMM by the copy engine or an SM
// kernel), udmabuf-4k / udmabuf-2m (direct reads into the CPU mapping of an
// imported udmabuf; the GPU consumes the same pages in place).
int cmd_restore(const char* file, const std::string& mode, unsigned depth, int passes, std::size_t gib) {
  gpu_init();
  const std::size_t bytes = gib * GiB;
  const std::uint64_t extents = bytes / kExtent;
  const bool land = mode == "land-ce" || mode == "land-sm";
  const bool ubuf = mode == "udmabuf-4k" || mode == "udmabuf-2m";
  require(land || ubuf || mode == "hvmm-inplace", "unknown mode");
  // Destination: GPU segments (address, bytes) and the CPU address reads target.
  std::vector<std::pair<CUdeviceptr, std::size_t>> segs;
  char* cpu_dest = nullptr;
  Vmm dest;
  UBuf u;
  if (ubuf) {
    const std::size_t chunk = std::min(bytes, std::max<std::size_t>(udmabuf_limit(), 2 * MiB));
    u = ubuf_or_die(bytes, mode == "udmabuf-2m", chunk);
    for (std::size_t i = 0; i < u.gpu.size(); ++i) segs.emplace_back(u.gpu[i], std::min(chunk, bytes - i * chunk));
    cpu_dest = u.cpu;
  } else {
    dest = mode == "hvmm-inplace" ? vmm_or_die(bytes, {true, 0, true, CU_MEM_HANDLE_TYPE_NONE, false})
                                  : vmm_or_die(bytes, {false, 2 * MiB, false, CU_MEM_HANDLE_TYPE_NONE, false});
    segs.emplace_back(dest.va, bytes);
    cpu_dest = reinterpret_cast<char*>(dest.va);
  }
  const std::size_t nslots = land ? 2 * depth : 0;
  Vmm landing;
  std::vector<void*> slot_ptr(nslots);
  if (land) {
    landing = vmm_or_die(nslots * kExtent, {true, 0, true, CU_MEM_HANDLE_TYPE_NONE, false});
    for (std::size_t s = 0; s < nslots; ++s) slot_ptr[s] = reinterpret_cast<char*>(landing.va) + s * kExtent;
  }
  int fd = open(file, O_DIRECT | O_RDONLY | O_CLOEXEC);
  require(fd >= 0, "open file");
  struct stat st{};
  require(fstat(fd, &st) == 0 && static_cast<std::size_t>(st.st_size) >= bytes, "file too small");
  Ring ring;
  ring.init(64);
  std::vector<cudaEvent_t> copied(std::max<std::size_t>(nslots, 1));
  for (auto& e : copied) RT(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  auto verify_all = [&] {
    unsigned long long bad = 0;
    std::size_t at = 0;
    for (auto [p, n] : segs) { bad += gpu_verify(p, n, at / 4); at += n; }
    return bad;
  };
  std::printf("# restore,mode,depth,pass,gib,gbps,p50_us,p99_us,max_us,cpu_s,bad_words,segments\n");
  for (int pass = 0; pass < passes; ++pass) {
    for (auto [p, n] : segs) gpu_clear(p, n);
    std::vector<Clock::time_point> started(extents);
    std::vector<double> lat;
    lat.reserve(extents);
    std::deque<std::size_t> free_slots;
    for (std::size_t s = 0; s < nslots; ++s) free_slots.push_back(s);
    std::deque<std::pair<std::size_t, std::uint64_t>> copying;
    std::uint64_t next = 0, done = 0;
    const double cpu0 = cpu_seconds();
    const auto start = Clock::now();
    while (done < extents) {
      while (next < extents && ring.in_flight + ring.pending < depth && (nslots == 0 || !free_slots.empty())) {
        std::size_t slot = 0;
        void* target = cpu_dest + next * kExtent;
        if (nslots) { slot = free_slots.front(); free_slots.pop_front(); target = slot_ptr[slot]; }
        started[next] = Clock::now();
        ring.prep_read(fd, target, kExtent, next * kExtent, (next << 8) | slot);
        ++next;
      }
      // Block for a read only when no copy is pending (as the diagnosis did).
      ring.enter(copying.empty() && ring.in_flight + ring.pending > 0 ? 1 : 0);
      std::uint64_t ud;
      int res;
      while (ring.reap(ud, res)) {
        if (res != static_cast<int>(kExtent)) {
          std::printf("restore,%s,read_failed,%s\n", mode.c_str(), res_str(res).c_str());
          return 1;
        }
        const std::uint64_t ext = ud >> 8;
        const std::size_t slot = ud & 0xff;
        if (land) {
          void* to = reinterpret_cast<void*>(dest.va + ext * kExtent);
          if (mode == "land-ce")
            RT(cudaMemcpyAsync(to, slot_ptr[slot], kExtent, cudaMemcpyDefault, g.stream));
          else
            copy_kernel<<<g.sms * 4, kThreads, 0, g.stream>>>(static_cast<uint4*>(to),
                                                             static_cast<const uint4*>(slot_ptr[slot]), kExtent / 16);
          RT(cudaEventRecord(copied[slot], g.stream));
          copying.emplace_back(slot, ext);
          continue;
        }
        lat.push_back(us_since(started[ext]));
        ++done;
      }
      while (!copying.empty() && cudaEventQuery(copied[copying.front().first]) == cudaSuccess) {
        auto [slot, ext] = copying.front();
        copying.pop_front();
        lat.push_back(us_since(started[ext]));
        free_slots.push_back(slot);
        ++done;
      }
    }
    RT(cudaStreamSynchronize(g.stream));
    const double secs = seconds_since(start);
    const double cpu = cpu_seconds() - cpu0;
    const unsigned long long bad = verify_all();
    std::printf("restore,%s,%u,%d,%zu,%.3f,%.0f,%.0f,%.0f,%.2f,%llu,%zu\n", mode.c_str(), depth, pass, gib,
                static_cast<double>(bytes) / secs / 1e9, pct(lat, 0.5), pct(lat, 0.99), pct(lat, 1.0), cpu, bad,
                segs.size());
  }
  // Negative control: one corrupted word must be found. (On the checking
  // stream: a legacy-stream memset does not order with a non-blocking stream.)
  CU(cuMemsetD32Async(segs[0].first + 12345 * 4, 0xdeadbeef, 1, g.stream));
  std::printf("restore,%s,negative_control_bad=%llu\n", mode.c_str(), verify_all());
  ring.close_ring();
  close(fd);
  if (land) vmm_free(landing);
  if (ubuf) ubuf_free(u); else vmm_free(dest);
  return 0;
}

// ---------------------------------------------------------------- cost

void report_mem(const char* when) {
  std::size_t fr = 0, tot = 0;
  CU(cuMemGetInfo(&fr, &tot));
  std::printf("mem %s cu_free_mib=%zu mem_available_mib=%lld shmem_mib=%lld hugepages_free=%lld fds=%d\n", when,
              fr / MiB, meminfo_kib("MemAvailable:") / 1024, meminfo_kib("Shmem:") / 1024,
              meminfo_kib("HugePages_Free:"), open_fds());
}

// Per-extent cost of making one 2 MiB extent GPU-visible this way, teardown
// orders, leaks, and a second process's view of the dma-buf.
int cmd_cost(bool huge, int iterations) {
  gpu_init();
  const std::size_t n = 2 * MiB, extents = 32;
  {
    UBuf base = ubuf_or_die(extents * n, huge, extents * n);  // the memfd and CPU mapping
    int dev = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
    require(dev >= 0, "/dev/udmabuf");
    std::vector<double> tc, ti, tm, tf, td, tx;
    for (int i = 0; i < iterations; ++i) {
      const std::size_t off = (i % extents) * n;
      auto t = Clock::now();
      udmabuf_create c{static_cast<__u32>(base.memfd), UDMABUF_FLAGS_CLOEXEC, off, n};
      int fd = ioctl(dev, UDMABUF_CREATE, &c);
      tc.push_back(us_since(t));
      require(fd >= 0, "UDMABUF_CREATE");
      CUDA_EXTERNAL_MEMORY_HANDLE_DESC d{};
      d.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD;
      d.handle.fd = fd;
      d.size = n;
      CUexternalMemory e{};
      t = Clock::now();
      CU(cuImportExternalMemory(&e, &d));
      ti.push_back(us_since(t));
      CUDA_EXTERNAL_MEMORY_BUFFER_DESC b{};
      b.size = n;
      CUdeviceptr p{};
      t = Clock::now();
      CU(cuExternalMemoryGetMappedBuffer(&p, e, &b));
      tm.push_back(us_since(t));
      if (i == 0) std::printf("cost first_extent_gpu_bad_vs_cpu_fill=%llu\n", (cpu_fill(base.cpu + off, n, 5), gpu_verify(p, n, 5)));
      t = Clock::now();
      CU(cuMemFree(p));
      tf.push_back(us_since(t));
      t = Clock::now();
      CU(cuDestroyExternalMemory(e));
      td.push_back(us_since(t));
      t = Clock::now();
      close(fd);
      tx.push_back(us_since(t));
    }
    close(dev);
    std::printf("cost_us pages=%s n=%d create_p50=%.1f p99=%.1f import_p50=%.1f p99=%.1f map_p50=%.1f p99=%.1f "
                "free_p50=%.1f p99=%.1f destroy_p50=%.1f p99=%.1f close_p50=%.1f p99=%.1f\n",
                huge ? "2m" : "4k", iterations, pct(tc, .5), pct(tc, .99), pct(ti, .5), pct(ti, .99), pct(tm, .5),
                pct(tm, .99), pct(tf, .5), pct(tf, .99), pct(td, .5), pct(td, .99), pct(tx, .5), pct(tx, .99));
    ubuf_free(base);
  }
  // Leak loop: 256 MiB each time, freed in ubuf_free's order.
  report_mem("leak_before");
  for (int i = 0; i < iterations / 5; ++i) {
    UBuf u = ubuf_or_die(256 * MiB, huge, 64 * MiB);
    ubuf_free(u);
  }
  report_mem("leak_after");
  // Teardown with the dma-buf fd and memfd closed first, then the GPU side.
  {
    UBuf u = ubuf_or_die(256 * MiB, huge, 256 * MiB);
    cpu_fill(u.cpu, 256 * MiB, 0);
    for (int fd : u.dmafds) close(fd);
    u.dmafds.clear();
    munmap(u.cpu, u.bytes);
    u.cpu = nullptr;
    close(u.memfd);
    u.memfd = -1;
    report_mem("fds_closed_gpu_alive");
    std::printf("order gpu_reads_after_cpu_side_closed_bad=%llu\n", gpu_verify(u.gpu[0], 256 * MiB, 0));
    ubuf_free(u);
    report_mem("after_gpu_free");
  }
  // Another process (no CUDA in it) writes through its own mapping of the memfd.
  {
    UBuf u = ubuf_or_die(64 * MiB, huge, 64 * MiB);
    gpu_fill(u.gpu[0], 64 * MiB, 0);
    std::fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
      void* m = mmap(nullptr, 64 * MiB, PROT_READ | PROT_WRITE, MAP_SHARED, u.memfd, 0);
      if (m == MAP_FAILED) _exit(3);
      auto bad = cpu_verify(m, 64 * MiB, 0);
      static_cast<std::uint32_t*>(m)[0] = 0xfeedf00d;
      _exit(bad ? 2 : 0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    std::uint32_t first = 0;
    CU(cuMemcpyDtoH(&first, u.gpu[0], 4));
    std::printf("order child_process exit=%d gpu_sees_child_write=%s\n", WIFEXITED(status) ? WEXITSTATUS(status) : -1,
                first == 0xfeedf00d ? "yes" : "no");
    ubuf_free(u);
  }
  report_mem("end");
  return 0;
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const std::string cmd = argc > 1 ? argv[1] : "";
  auto pages = [&](const char* s) {
    const std::string p = s;
    require(p == "4k" || p == "2m", "pages must be 4k or 2m");
    return p == "2m";
  };
  if (cmd == "create" && argc == 4) return cmd_create(argv[2], number(argv[3]));
  if (cmd == "info" && argc == 2) return cmd_info();
  if (cmd == "feasibility" && argc == 4) return cmd_feasibility(argv[2], argv[3]);
  if (cmd == "udmabuf" && argc == 4) return cmd_udmabuf(argv[2], pages(argv[3]));
  if (cmd == "micro" && argc == 6)
    return cmd_micro(argv[2], pages(argv[3]), static_cast<int>(number(argv[4])), static_cast<int>(number(argv[5])));
  if (cmd == "restore" && argc == 7)
    return cmd_restore(argv[2], argv[3], static_cast<unsigned>(number(argv[4])), static_cast<int>(number(argv[5])),
                       number(argv[6]));
  if (cmd == "cost" && argc == 4) return cmd_cost(pages(argv[2]), static_cast<int>(number(argv[3])));
  std::fprintf(stderr,
               "Usage: %s create FILE GIB | info | feasibility FILE DIR | udmabuf FILE PAGES |\n"
               "       micro FILE PAGES SAMPLES ROTATE | restore FILE MODE DEPTH PASSES GIB | cost PAGES ITERATIONS\n"
               "PAGES: 4k (shmem memfd) or 2m (hugetlb memfd)\n"
               "MODE: hvmm-inplace land-ce land-sm udmabuf-4k udmabuf-2m\n",
               argv[0]);
  return 2;
}
