// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// Standalone, bounded M0 probe. Fatal errors exit; process teardown owns cleanup.
#include <cuda.h>
#include <cufile.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
constexpr std::size_t MiB = 1UL << 20, GiB = 1UL << 30;
void require(bool ok, const char* what) {
  if (!ok) { std::fprintf(stderr, "%s (errno=%d %s)\n", what, errno, std::strerror(errno)); std::exit(1); }
}
void check(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* name = nullptr; cuGetErrorName(r, &name);
  std::fprintf(stderr, "%s: %s\n", what, name ? name : "unknown CUDA error"); std::exit(1);
}
#define CUDA(x) check((x), #x)
void cfcheck(CUfileError_t r, const char* what) {
  if (r.err != CU_FILE_SUCCESS) { std::fprintf(stderr, "%s: %d cuda=%d\n", what, r.err, r.cu_err); std::exit(1); }
}
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
unsigned pattern(unsigned long long index) {
  unsigned x = static_cast<unsigned>(index) ^ static_cast<unsigned>(index >> 32) ^ 0x9e3779b9U;
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16);
}
unsigned long long number(const char* s) {
  char* end = nullptr; errno = 0; auto n = std::strtoull(s, &end, 10);
  require(!errno && s[0] != '-' && end != s && *end == 0, "invalid integer"); return n;
}
unsigned long long meminfo(const char* field) {
  std::ifstream in("/proc/meminfo"); std::string key, unit; unsigned long long n = 0;
  while (in >> key >> n >> unit) if (key == field) return n * 1024;
  require(false, "missing meminfo field"); return 0;
}
unsigned long long cached(int fd, std::size_t bytes) {
  void* p = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
  require(p != MAP_FAILED, "cache mmap");
  auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  std::vector<unsigned char> vec((bytes + page - 1) / page);
  require(mincore(p, bytes, vec.data()) == 0, "mincore");
  unsigned long long count = 0; for (auto v : vec) count += (v & 1) != 0;
  require(munmap(p, bytes) == 0, "cache munmap"); return count * page;
}
unsigned long long disk_sectors(bool write) {
  std::ifstream in("/sys/block/nvme0n1/stat"); unsigned long long v = 0;
  for (int i = 0; i <= (write ? 6 : 2); ++i) require(static_cast<bool>(in >> v), "disk stat");
  return v;
}
double cpu_seconds() {
  rusage r{}; require(getrusage(RUSAGE_SELF, &r) == 0, "getrusage");
  return r.ru_utime.tv_sec + r.ru_stime.tv_sec + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e6;
}
void create_file(const char* path, std::size_t bytes) {
  int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_DIRECT | O_CLOEXEC | O_NOFOLLOW, 0600);
  require(fd >= 0, "create exclusive regular test file");
  require(posix_fallocate(fd, 0, bytes) == 0, "fallocate");
  void* p = nullptr; require(posix_memalign(&p, 2 * MiB, 8 * MiB) == 0, "create buffer");
  auto* words = static_cast<unsigned*>(p);
  for (std::size_t off = 0; off < bytes; off += 8 * MiB) {
    for (std::size_t i = 0; i < 8 * MiB / 4; ++i) words[i] = pattern(off / 4 + i);
    require(pwrite(fd, p, 8 * MiB, off) == 8 * MiB, "create pwrite");
  }
  require(fdatasync(fd) == 0, "create fdatasync"); require(close(fd) == 0, "create close");
  free(p); std::printf("created,%zu\n", bytes);
}
struct Vmm {
  CUdeviceptr ptr{}; std::size_t bytes{}, gran{};
  std::vector<CUmemGenericAllocationHandle> handles;
  void init(std::size_t n, bool host = false) {
    CUmemAllocationProp prop{}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = host ? CU_MEM_LOCATION_TYPE_HOST_NUMA : CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    CUDA(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    // Keep device backing independently reclaimable at D-033 granularity.
    bytes = (n + gran - 1) / gran * gran;
    CUDA(cuMemAddressReserve(&ptr, bytes, 0, 0, 0));
    for (std::size_t off = 0; off < bytes; off += gran) {
      CUmemGenericAllocationHandle h{}; CUDA(cuMemCreate(&h, gran, &prop, 0));
      handles.push_back(h); CUDA(cuMemMap(ptr + off, gran, 0, h, 0));
    }
    CUmemAccessDesc access[2]{};
    access[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE; access[0].location.id = 0;
    access[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access[1].location = prop.location; access[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUDA(cuMemSetAccess(ptr, bytes, access, host ? 2 : 1));
  }
  void release() {
    for (std::size_t i = 0; i < handles.size(); ++i) {
      CUDA(cuMemUnmap(ptr + i * gran, gran)); CUDA(cuMemRelease(handles[i]));
    }
    if (ptr) CUDA(cuMemAddressFree(ptr, bytes)); ptr = 0; handles.clear();
  }
};
struct Slot {
  void* host{}; CUdeviceptr gpu{}, scan_output{}; CUstream stream{}; CUevent end{};
  Clock::time_point start; std::size_t last_offset{}; int state{};
  std::vector<double> latency;
};
// Minimal io_uring ABI use; one owner of both rings, acquire/release shared indexes.
struct Ring {
  int fd = -1; void* rings{}; io_uring_sqe* sqes{};
  std::size_t ring_bytes{}, sqe_bytes{};
  unsigned *sqhead{}, *sqtail{}, *sqmask{}, *array{}, *cqhead{}, *cqtail{}, *cqmask{};
  io_uring_cqe* cqes{};
  static unsigned acquire(unsigned* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
  static void release(unsigned* p, unsigned v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
  void init(unsigned depth, const std::vector<Slot>& slots, std::size_t bytes) {
    io_uring_params p{}; fd = syscall(__NR_io_uring_setup, depth, &p); require(fd >= 0, "io_uring_setup");
    require(p.features & IORING_FEAT_SINGLE_MMAP, "io_uring needs SINGLE_MMAP");
    ring_bytes = std::max(p.sq_off.array + p.sq_entries * sizeof(unsigned), p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe));
    rings = mmap(nullptr, ring_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, IORING_OFF_SQ_RING);
    require(rings != MAP_FAILED, "ring mmap");
    sqe_bytes = p.sq_entries * sizeof(io_uring_sqe);
    sqes = static_cast<io_uring_sqe*>(mmap(nullptr, sqe_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, IORING_OFF_SQES));
    require(sqes != MAP_FAILED, "sqe mmap");
    auto at = [&](unsigned off) { return reinterpret_cast<unsigned*>(static_cast<char*>(rings) + off); };
    sqhead = at(p.sq_off.head); sqtail = at(p.sq_off.tail); sqmask = at(p.sq_off.ring_mask); array = at(p.sq_off.array);
    cqhead = at(p.cq_off.head); cqtail = at(p.cq_off.tail); cqmask = at(p.cq_off.ring_mask);
    cqes = reinterpret_cast<io_uring_cqe*>(static_cast<char*>(rings) + p.cq_off.cqes);
    std::vector<iovec> bufs; for (const auto& s : slots) bufs.push_back({s.host, bytes});
    require(syscall(__NR_io_uring_register, fd, IORING_REGISTER_BUFFERS, bufs.data(), bufs.size()) == 0, "register io buffers");
  }
  void submit(int file, Slot& slot, unsigned id, std::size_t bytes, std::size_t off) {
    auto tail = acquire(sqtail); auto index = tail & *sqmask;
    require(tail - acquire(sqhead) <= *sqmask, "submission queue full");
    auto& s = sqes[index]; s = {}; s.opcode = IORING_OP_READ_FIXED; s.fd = file;
    s.off = off; s.addr = reinterpret_cast<unsigned long long>(slot.host); s.len = bytes;
    s.buf_index = id; s.user_data = id; array[index] = index;
    slot.last_offset = off; slot.start = Clock::now(); slot.state = 1;
    release(sqtail, tail + 1);
    int n; do { n = syscall(__NR_io_uring_enter, fd, 1, 0, 0, nullptr, 0); } while (n < 0 && errno == EINTR);
    require(n == 1, "io_uring submit");
  }
  bool reap(unsigned& id, int& result) {
    auto head = acquire(cqhead); if (head == acquire(cqtail)) return false;
    auto c = cqes[head & *cqmask]; id = c.user_data; result = c.res; release(cqhead, head + 1); return true;
  }
  void close_ring() {
    require(syscall(__NR_io_uring_register, fd, IORING_UNREGISTER_BUFFERS, nullptr, 0) == 0, "unregister io buffers");
    require(close(fd) == 0, "ring close"); munmap(sqes, sqe_bytes); munmap(rings, ring_bytes);
  }
};

int main(int argc, char** argv) {
  if (argc == 4 && std::string(argv[1]) == "create") {
    auto gib = number(argv[3]); require(gib >= 1 && gib <= 64, "create size 1..64 GiB");
    create_file(argv[2], gib * GiB); return 0;
  }
  if (argc != 10) {
    std::fprintf(stderr, "Usage: %s CUBIN FILE MODE KiB DEPTH SECONDS CACHE(cold/warm) PRESSURE_GiB LOAD(none/compute/memory)\n", argv[0]); return 2;
  }
  const std::string mode = argv[3], cache = argv[7], load = argv[9];
  require(mode == "buffered" || mode == "direct" || mode == "uring" || mode == "inplace" || mode == "uring_inplace" || mode == "cufile" || mode == "hostvmm" || mode == "uring_hostvmm" || mode == "write" || mode == "hostvmm_write", "unknown mode");
  require(cache == "cold" || cache == "warm" || cache == "resident" || cache == "control" || cache == "sparse", "unknown cache state");
  require(load == "none" || load == "compute" || load == "memory", "unknown load");
  auto kib = number(argv[4]), q = number(argv[5]), duration = number(argv[6]), pressure_gib = number(argv[8]);
  require(kib >= 4 && kib <= 262144 && (kib & (kib - 1)) == 0 && q >= 1 && q <= 32 && duration >= 1 && duration <= 300 && pressure_gib <= 104, "argument out of bounds");
  const auto chunk = kib * 1024;
  const bool hostvmm = mode == "hostvmm" || mode == "uring_hostvmm" || mode == "hostvmm_write";
  const bool inplace = mode == "inplace" || mode == "uring_inplace" || hostvmm;
  const bool uring = mode == "uring" || mode == "uring_inplace" || mode == "uring_hostvmm";
  const bool writing = mode == "write" || mode == "hostvmm_write";
  const bool resident = cache == "resident";
  const bool control = cache == "control";
  const bool sparse = cache == "sparse";
  require(!sparse || (!uring && !writing && chunk <= 2 * MiB), "sparse probe needs synchronous reads <=2MiB");
  require(!control || (!uring && !writing && load != "none"), "control requires background load and synchronous read mode");
  require(!resident || (!uring && !writing && load == "none"), "resident scan requires synchronous read mode and no background load");
  int fd = open(argv[2], (writing ? O_RDWR | O_CREAT | O_EXCL : O_RDONLY) | O_CLOEXEC | O_NOFOLLOW | (mode == "buffered" ? 0 : O_DIRECT), 0600);
  if (fd >= 0 && writing) require(posix_fallocate(fd, 0, 4 * GiB) == 0, "write scratch allocation");
  require(fd >= 0, "open test file"); struct stat st{};
  require(fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_uid == getuid() && st.st_nlink == 1 && (st.st_mode & 077) == 0, "owned private regular test file required");
  const auto file_bytes = static_cast<std::size_t>(st.st_size);
  struct statx sx{};
  require(statx(fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &sx) == 0 && (sx.stx_mask & STATX_DIOALIGN), "direct I/O alignment query");
  std::printf("alignment,mem,%u,offset,%u\n", sx.stx_dio_mem_align, sx.stx_dio_offset_align);
  require(sx.stx_dio_mem_align && sx.stx_dio_offset_align && chunk % sx.stx_dio_offset_align == 0, "unaligned I/O chunk");
  require(file_bytes >= chunk * q && file_bytes <= 64 * GiB && file_bytes % chunk == 0, "invalid file size");
  if (writing) require(file_bytes <= 8 * GiB, "write test limited to 8 GiB existing scratch file");
  CUDA(cuInit(0)); CUdevice dev{}; CUDA(cuDeviceGet(&dev, 0));
  CUcontext context{}; CUDA(cuDevicePrimaryCtxRetain(&context, dev)); CUDA(cuCtxSetCurrent(context));
  for (auto a : {CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED}) {
    int value = -1; auto r = cuDeviceGetAttribute(&value, a, dev); std::printf("attribute,%d,%d,%d\n", a, value, r);
  }
  CUmodule module{}; CUDA(cuModuleLoad(&module, argv[1]));
  CUfunction verify{}, compute{}, scan{}, fill{};
  CUDA(cuModuleGetFunction(&verify, module, "verify")); CUDA(cuModuleGetFunction(&compute, module, "compute")); CUDA(cuModuleGetFunction(&scan, module, "scan")); CUDA(cuModuleGetFunction(&fill, module, "fill"));
  CUdeviceptr errors{}, pressure{}, bgdata{}, bgout{};
  CUDA(cuMemAlloc(&errors, sizeof(unsigned)));
  // Account for both copies, scan outputs, telemetry samples, and driver/OS
  // headroom before a caller-selected pressure allocation can exhaust UMA.
  const auto slot_budget = (std::max<std::size_t>(chunk, 2 * MiB) + chunk) * q;
  const auto probe_budget = slot_budget + (load == "memory" ? GiB : 0) + GiB;
  require(meminfo("MemAvailable:") > pressure_gib * GiB + probe_budget + 12 * GiB,
          "insufficient aggregate memory headroom");
  if (pressure_gib) {
    require(meminfo("MemAvailable:") > pressure_gib * GiB + 12 * GiB, "insufficient memory headroom for pressure test");
    CUDA(cuMemAlloc(&pressure, pressure_gib * GiB)); CUDA(cuMemsetD8(pressure, 0x5a, pressure_gib * GiB)); CUDA(cuCtxSynchronize());
  }
  CUstream bgstream{}; CUDA(cuStreamCreate(&bgstream, CU_STREAM_NON_BLOCKING));
  if (load != "none") {
    CUDA(cuMemAlloc(&bgout, 256 * 256 * sizeof(unsigned)));
    if (load == "memory") { CUDA(cuMemAlloc(&bgdata, GiB)); CUDA(cuMemsetD8(bgdata, 1, GiB)); CUDA(cuCtxSynchronize()); }
  }
  Vmm backing; if (!inplace || hostvmm) {
    backing.init(std::max<std::size_t>(chunk, 2 * MiB) * q, hostvmm);
    std::printf("vmm,granularity,%zu,bytes,%zu,handles,%zu\n", backing.gran, backing.bytes, backing.handles.size());
  }
  std::vector<Slot> slots(q);
  for (std::size_t i = 0; i < q; ++i) {
    auto& s = slots[i];
    if (hostvmm) s.host = reinterpret_cast<void*>(backing.ptr + i * std::max<std::size_t>(chunk, 2 * MiB));
    else { require(posix_memalign(&s.host, 2 * MiB, chunk) == 0, "aligned host buffer"); std::memset(s.host, 0, chunk); }
    if (!inplace) CUDA(cuMemHostRegister(s.host, chunk, CU_MEMHOSTREGISTER_DEVICEMAP));
    s.gpu = inplace ? reinterpret_cast<CUdeviceptr>(s.host) : backing.ptr + i * std::max<std::size_t>(chunk, 2 * MiB);
    CUDA(cuStreamCreate(&s.stream, CU_STREAM_NON_BLOCKING)); CUDA(cuEventCreate(&s.end, CU_EVENT_DISABLE_TIMING));
    if (resident) CUDA(cuMemAlloc(&s.scan_output, 256 * 256 * sizeof(unsigned)));
    s.latency.reserve(static_cast<std::size_t>(duration) * 200000 / q + 1);
  }
  CUfileHandle_t cfh{}; int cufd = -1;
  if (mode == "cufile") {
    // This cuFile release rejects O_NOFOLLOW. Reopen our already validated,
    // still-open descriptor, not the original pathname, to retain identity.
    auto descriptor_path = std::string("/proc/self/fd/") + std::to_string(fd);
    cufd = open(descriptor_path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    struct stat cfstat{};
    require(cufd >= 0 && fstat(cufd, &cfstat) == 0 && cfstat.st_dev == st.st_dev && cfstat.st_ino == st.st_ino, "cuFile reopen identity");
    cfcheck(cuFileDriverOpen(), "cuFileDriverOpen"); CUfileDescr_t descr{}; descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; descr.handle.fd = fd;
    descr.handle.fd = cufd;
    cfcheck(cuFileHandleRegister(&cfh, &descr), "cuFileHandleRegister");
    for (auto& s : slots) cfcheck(cuFileBufRegister(reinterpret_cast<void*>(s.gpu), chunk, 0), "cuFileBufRegister");
  }
  auto transfer = [&](Slot& s, std::size_t off) {
    s.last_offset = off;
    if (mode == "cufile") require(cuFileRead(cfh, reinterpret_cast<void*>(s.gpu), chunk, off, 0) == static_cast<ssize_t>(chunk), "cuFileRead exact length");
    else if (writing) {
      unsigned long long words = chunk / 4, first = off / 4;
      void* args[] = {&s.gpu, &words, &first};
      CUDA(cuLaunchKernel(fill, 256, 1, 1, 256, 1, 1, 0, s.stream, args, nullptr));
      if (!inplace) CUDA(cuMemcpyDtoHAsync(s.host, s.gpu, chunk, s.stream));
      CUDA(cuStreamSynchronize(s.stream));
      require(pwrite(fd, s.host, chunk, off) == static_cast<ssize_t>(chunk), "pwrite exact length");
    } else {
      require(pread(fd, s.host, chunk, off) == static_cast<ssize_t>(chunk), "pread exact length");
      if (!inplace) { CUDA(cuMemcpyHtoDAsync(s.gpu, s.host, chunk, s.stream)); CUDA(cuStreamSynchronize(s.stream)); }
    }
  };
  auto verify_slot = [&](Slot& s, unsigned expected_errors) {
    CUDA(cuMemsetD32Async(errors, 0, 1, s.stream));
    unsigned long long words = chunk / 4, first = s.last_offset / 4;
    void* args[] = {&s.gpu, &words, &first, &errors};
    CUDA(cuLaunchKernel(verify, 256, 1, 1, 256, 1, 1, 0, s.stream, args, nullptr)); CUDA(cuStreamSynchronize(s.stream));
    unsigned bad = 0; CUDA(cuMemcpyDtoH(&bad, errors, sizeof(bad)));
    require(bad == expected_errors, "GPU content verification mismatch");
  };
  // Full-word preflight verification, then a one-word negative control, outside timing.
  if (!writing) {
    for (std::size_t i = 0; i < q; ++i) { transfer(slots[i], i * chunk); verify_slot(slots[i], 0); }
    auto& s = slots[0]; unsigned bad = pattern(s.last_offset / 4) ^ 1U;
    CUDA(cuMemcpyHtoD(s.gpu, &bad, sizeof(bad))); CUDA(cuCtxSynchronize()); verify_slot(s, 1);
    transfer(s, 0); verify_slot(s, 0);
  }
  std::puts("preflight,PASS");
  // Evict only this clean scratch file. Warm runs explicitly populate all of it.
  require(posix_fadvise(fd, 0, file_bytes, POSIX_FADV_DONTNEED) == 0, "file cache discard");
  if (cache == "warm") {
    int warmfd = open(argv[2], O_RDONLY | O_CLOEXEC | O_NOFOLLOW); require(warmfd >= 0, "warm open");
    for (std::size_t off = 0; off < file_bytes; off += chunk)
      require(pread(warmfd, slots[0].host, chunk, off) == static_cast<ssize_t>(chunk), "cache warm read");
    close(warmfd);
  }
  const auto cache_before = cached(fd, file_bytes), available_before = meminfo("MemAvailable:"), disk_before = disk_sectors(writing);
  if (cache == "cold" || sparse) require(cache_before == 0, "scratch file cache is not cold");
  Ring ring; if (uring) ring.init(q, slots, chunk);
  std::atomic<bool> stop_bg{false}; std::atomic<unsigned long long> bg_count{0};
  std::thread bg;
  if (load != "none") bg = std::thread([&] {
    CUDA(cuCtxSetCurrent(context));
    do {
      if (load == "compute") {
        int rounds = 100000; void* args[] = {&bgout, &rounds};
        CUDA(cuLaunchKernel(compute, 192, 1, 1, 256, 1, 1, 0, bgstream, args, nullptr));
      } else {
        unsigned long long words = GiB / 4; void* args[] = {&bgdata, &words, &bgout};
        CUDA(cuLaunchKernel(scan, 256, 1, 1, 256, 1, 1, 0, bgstream, args, nullptr));
      }
      CUDA(cuStreamSynchronize(bgstream)); ++bg_count;
    } while (!stop_bg.load());
  });
  while (load != "none" && bg_count.load() == 0) std::this_thread::yield();
  const auto bg_before = bg_count.load();
  std::atomic<std::size_t> next{0};
  const auto start = Clock::now(); const double cpu_before = cpu_seconds();
  if (control) std::this_thread::sleep_for(std::chrono::seconds(duration));
  else if (uring) {
    std::size_t active = 0;
    do {
      unsigned id = 0; int result = 0;
      while (ring.reap(id, result)) {
        require(id < q && result == static_cast<int>(chunk) && slots[id].state == 1, "invalid read completion");
        auto& s = slots[id];
        if (!inplace) CUDA(cuMemcpyHtoDAsync(s.gpu, s.host, chunk, s.stream));
        CUDA(cuEventRecord(s.end, s.stream)); s.state = 2;
      }
      for (unsigned i = 0; i < q; ++i) {
        auto& s = slots[i];
        if (s.state == 2) {
          auto r = cuEventQuery(s.end);
          if (r == CUDA_SUCCESS) { s.latency.push_back(seconds(s.start) * 1e6); s.state = 0; --active; }
          else require(r == CUDA_ERROR_NOT_READY, "CUDA event query failed");
        }
        if (s.state == 0 && seconds(start) < duration) {
          auto off = (next.fetch_add(chunk)) % file_bytes; ring.submit(fd, s, i, chunk, off); ++active;
        }
      }
      std::this_thread::yield();
    } while (active || seconds(start) < duration);
  } else {
    std::vector<std::thread> workers;
    for (std::size_t i = 0; i < q; ++i) workers.emplace_back([&, i] {
      CUDA(cuCtxSetCurrent(context)); auto& s = slots[i];
      do {
        auto logical = next.fetch_add(chunk);
        // Buffered cold data and write data are traversed at most once.
        if ((writing || (mode == "buffered" && cache == "cold")) && logical >= file_bytes) break;
        if (sparse && logical / chunk >= file_bytes / (2 * MiB)) break;
        if (sparse) logical = logical / chunk * 2 * MiB;
        auto off = logical % file_bytes; auto begin = Clock::now();
        if (resident) {
          unsigned long long words = chunk / 4; void* args[] = {&s.gpu, &words, &s.scan_output};
          CUDA(cuLaunchKernel(scan, 256, 1, 1, 256, 1, 1, 0, s.stream, args, nullptr)); CUDA(cuStreamSynchronize(s.stream));
        } else transfer(s, off);
        s.latency.push_back(seconds(begin) * 1e6);
      } while (seconds(start) < duration);
    });
    for (auto& worker : workers) worker.join();
  }
  const auto elapsed = seconds(start); const auto cpu = cpu_seconds() - cpu_before; const auto bg_after = bg_count.load();
  stop_bg = true; if (bg.joinable()) bg.join();
  auto sync_start = Clock::now(); if (writing) require(fdatasync(fd) == 0, "write fdatasync"); auto sync_seconds = seconds(sync_start);
  const auto disk_after = disk_sectors(writing), cache_after = cached(fd, file_bytes), available_after = meminfo("MemAvailable:");
  if (uring) ring.close_ring();
  std::vector<double> samples;
  for (auto& s : slots) {
    if (!writing) verify_slot(s, 0);
    samples.insert(samples.end(), s.latency.begin(), s.latency.end());
  }
  require(control || !samples.empty(), "no measured transfers"); std::sort(samples.begin(), samples.end());
  auto percentile = [&](double p) { return samples.empty() ? 0.0 : samples[std::min(samples.size() - 1, static_cast<std::size_t>(p * (samples.size() - 1)))]; };
  std::size_t moved = samples.size() * chunk;
  if (writing) {
    // Verify every written word after durability completion, outside timing/counters.
    auto& s = slots[0];
    for (std::size_t off = 0; off < moved; off += chunk) {
      require(pread(fd, s.host, chunk, off) == static_cast<ssize_t>(chunk), "write readback");
      if (!inplace) CUDA(cuMemcpyHtoDAsync(s.gpu, s.host, chunk, s.stream));
      CUDA(cuStreamSynchronize(s.stream));
      s.last_offset = off; verify_slot(s, 0);
    }
  }
  std::printf("result_header,mode,chunk_bytes,depth,seconds,cache,pressure_gib,load,requests,bytes,GBps,p50_us,p95_us,p99_us,max_us,cpu_seconds,cache_before,cache_after,memavailable_before,memavailable_after,disk_bytes,background_completions,background_per_second,fdatasync_seconds\n");
  std::printf("result,%s,%llu,%llu,%.6f,%s,%llu,%s,%zu,%zu,%.6f,%.3f,%.3f,%.3f,%.3f,%.6f,%llu,%llu,%llu,%llu,%llu,%llu,%.6f,%.6f\n", mode.c_str(), chunk, q, elapsed, cache.c_str(), pressure_gib, load.c_str(), samples.size(), moved, moved / elapsed / 1e9, percentile(.5), percentile(.95), percentile(.99), percentile(1), cpu, cache_before, cache_after, available_before, available_after, (disk_after - disk_before) * 512, bg_after - bg_before, (bg_after - bg_before) / elapsed, sync_seconds);
  // Histogram retains latency distribution without printing in the transfer loop.
  std::vector<unsigned long long> hist(10001);
  for (auto x : samples) ++hist[std::min<std::size_t>(10000, static_cast<std::size_t>(x / 10))];
  for (std::size_t i = 0; i < hist.size(); ++i) if (hist[i]) std::printf("hist_us,%zu,%llu\n", i * 10, hist[i]);
  std::puts("verification,PASS");
  if (mode == "cufile") {
    for (auto& s : slots) cfcheck(cuFileBufDeregister(reinterpret_cast<void*>(s.gpu)), "cuFileBufDeregister");
    cuFileHandleDeregister(cfh); cfcheck(cuFileDriverClose(), "cuFileDriverClose");
    require(close(cufd) == 0, "cuFile fd close");
  }
  for (auto& s : slots) {
    CUDA(cuStreamDestroy(s.stream)); CUDA(cuEventDestroy(s.end));
    if (!inplace) CUDA(cuMemHostUnregister(s.host));
    if (!hostvmm) free(s.host);
    if (s.scan_output) CUDA(cuMemFree(s.scan_output));
  }
  backing.release(); if (pressure) CUDA(cuMemFree(pressure));
  if (bgdata) CUDA(cuMemFree(bgdata)); if (bgout) CUDA(cuMemFree(bgout));
  CUDA(cuMemFree(errors)); CUDA(cuStreamDestroy(bgstream)); CUDA(cuModuleUnload(module)); CUDA(cuDevicePrimaryCtxRelease(dev));
  require(close(fd) == 0, "close test file"); std::puts("complete,PASS");
}
