// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// External llama.cpp reference harness for backend-proof P2's memory
// measurements (docs/backend-proof.md, "Memory and workspace"): the FP16
// toolchain bridge run with the census readings and controls, the
// reference side of the coarse memory check (D-085, which ended the census
// rules). It does not implement llmpalooza inference. It links the
// bridge's existing build (census_bridge.sh) and evaluates the trajectories
// of ../backend-proof-p0/fp16_reference.cc twice on one context (the second
// from a cleared cache, as native's second evaluation runs), with readings
// at quiescent points:
//
//   fp16_census MODEL OUTPUT_JSON control|heldout [IDS_FILE]
//
// Readings (each after a fixed settle, below):
// /proc/meminfo's MemAvailable and SUnreclaim and the pages on the per-CPU
// page lists (/proc/zoneinfo), three times back to back; the process's
// RssAnon and RssFile (/proc/self/status); mallinfo2; every other process's
// RssAnon, summed (/proc/*/status); and CLOCK_REALTIME and
// CLOCK_MONOTONIC_RAW, which place the reading on an nsys trace's
// timeline. Steps: process start; after the CUDA context
// (cudaFree(0)); after the backends' initialization; after the model load;
// after the context's creation (KV, compute and output buffers); then, in
// each evaluation, before every chunk and after it, once llama_synchronize
// has returned; then six tail readings with no work between them.
// Every buffer the bridge declares is recorded with each reading: model,
// KV and compute buffers by buffer type (llama_get_memory_breakdown) and
// the output buffer (llama_context::output_reserve's size). The GGML
// pool's committed bytes and the cuBLAS workspace are not visible through
// the API; fp16-plan.json records them.
//
// Controls, three repeats each, read before, while held and after being
// freed, once after the context (followed by a reading of its own, so any
// residue the controls leave is an interval of its own) and once after the
// tail readings: a 64 MiB
// cudaMalloc (cleared), a 64 MiB VMM mapping in 2 MiB extents
// (cuMemCreate, cuMemMap, cuMemSetAccess), and 64 MiB of host memory from
// malloc, written. At the end only, a 64 MiB cudaMallocHost probe is read
// the same way; it is not a control, it shows which counters pinned host
// memory moves.
// The logits' SHA-256 is written too: the harness must reproduce the
// bridge's recorded logits for its counts to be the bridge's, and the
// second evaluation must equal the first bit for bit. Every host page either
// evaluation's logits are kept in is touched before the context is created,
// so no reading interval faults them in (native does the same).
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <malloc.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

// ---------------------------------------------------------------- readings

static uint64_t field_kib(const std::string & text, const char * key) {
    const std::string needle = std::string("\n") + key + ":";
    const size_t at = ("\n" + text).find(needle);
    require(at != std::string::npos, "missing counter");
    const char * p = text.c_str() + at + needle.size() - 1;
    return std::strtoull(p, nullptr, 10) * 1024;
}

static std::string slurp(const char * path) {
    std::ifstream file(path);
    std::stringstream s;
    s << file.rdbuf();
    return s.str();
}

struct Reading {
    std::string step;
    int evaluation = 0;
    int chunk = -1;
    // Three back-to-back reads of the system counters (a list drained
    // between two files' reads tears only one of them).
    std::array<uint64_t, 3> mem_available{}, pcp{}, sunreclaim{};
    uint64_t rss_anon = 0, rss_file = 0;
    uint64_t malloc_arena = 0, malloc_hblkhd = 0, malloc_uordblks = 0;
    uint64_t others_rss_anon = 0, others_processes = 0;
    int64_t settle_from_raw_ns = 0, realtime_ns = 0, raw_ns = 0;
    std::map<std::string, uint64_t> declared;
};

static std::map<std::string, uint64_t> g_declared;

// Each reading waits CENSUS_SETTLE_MS first (2,500 ms by default, more
// than twice vm.stat_interval; the coarse memory check uses 50 ms).
static int settle_ms() {
    const char * env = std::getenv("CENSUS_SETTLE_MS");
    return env ? std::atoi(env) : 2500;
}

// Every other process's RssAnon, summed, and how many there are: memory
// another process takes moves MemAvailable too.
static void others(uint64_t & rss_anon, uint64_t & processes) {
    rss_anon = processes = 0;
    const long self = long(getpid());
    DIR * proc = opendir("/proc");
    require(proc != nullptr, "/proc");
    while (const dirent * entry = readdir(proc)) {
        char * end = nullptr;
        const long pid = std::strtol(entry->d_name, &end, 10);
        if (end == entry->d_name || *end != '\0' || pid == self) continue;
        const std::string status = slurp(("/proc/" + std::string(entry->d_name) + "/status").c_str());
        if (status.empty()) continue;  // exited meanwhile
        ++processes;
        const size_t at = ("\n" + status).find("\nRssAnon:");
        if (at != std::string::npos) rss_anon += std::strtoull(status.c_str() + at + 8, nullptr, 10) * 1024;
    }
    closedir(proc);
}

// The pages on every CPU's per-CPU page lists, in bytes: /proc/zoneinfo's
// `count:` under each zone's pagesets. Freed pages wait there, and
// allocations are served from there, without moving the free-page counter
// that MemAvailable reads, so MemAvailable plus these is what tracks an
// allocation (RE-024).
static uint64_t pcp_bytes() {
    std::ifstream file("/proc/zoneinfo");
    std::string line;
    uint64_t pages = 0;
    while (std::getline(file, line)) {
        const size_t at = line.find("count:");
        if (at != std::string::npos && line.find_first_not_of(" \t") == line.find("count:")) {
            pages += std::strtoull(line.c_str() + at + 6, nullptr, 10);
        }
    }
    return pages * uint64_t(sysconf(_SC_PAGESIZE));
}

static int64_t clock_ns(clockid_t clock) {
    timespec now{};
    clock_gettime(clock, &now);
    return int64_t(now.tv_sec) * 1000000000 + now.tv_nsec;
}

static Reading read(const std::string & step, int evaluation = 0, int chunk = -1) {
    Reading r;
    r.settle_from_raw_ns = clock_ns(CLOCK_MONOTONIC_RAW);
    std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms()));
    r.step = step;
    r.evaluation = evaluation;
    r.chunk = chunk;
    r.realtime_ns = clock_ns(CLOCK_REALTIME);
    r.raw_ns = clock_ns(CLOCK_MONOTONIC_RAW);
    for (size_t i = 0; i < r.mem_available.size(); ++i) {
        const std::string meminfo = slurp("/proc/meminfo");
        r.mem_available[i] = field_kib(meminfo, "MemAvailable");
        r.pcp[i] = pcp_bytes();
        r.sunreclaim[i] = field_kib(meminfo, "SUnreclaim");
    }
    const std::string status = slurp("/proc/self/status");
    r.rss_anon = field_kib(status, "RssAnon");
    r.rss_file = field_kib(status, "RssFile");
    const struct mallinfo2 m = mallinfo2();
    r.malloc_arena = m.arena;
    r.malloc_hblkhd = m.hblkhd;
    r.malloc_uordblks = m.uordblks;
    others(r.others_rss_anon, r.others_processes);
    r.declared = g_declared;
    return r;
}

static std::string triple(const std::array<uint64_t, 3> & v) {
    return "[" + std::to_string(v[0]) + "," + std::to_string(v[1]) + "," + std::to_string(v[2]) + "]";
}

static std::string json(const Reading & r) {
    std::string out = "{\"step\":\"" + r.step + "\",\"evaluation\":" + std::to_string(r.evaluation) +
        ",\"chunk\":" + std::to_string(r.chunk) + ",\"settle_from_raw_ns\":" + std::to_string(r.settle_from_raw_ns) +
        ",\"realtime_ns\":" + std::to_string(r.realtime_ns) + ",\"raw_ns\":" + std::to_string(r.raw_ns) +
        ",\"mem_available\":" + triple(r.mem_available) + ",\"pcp\":" + triple(r.pcp) +
        ",\"sunreclaim\":" + triple(r.sunreclaim) + ",\"rss_anon\":" + std::to_string(r.rss_anon) +
        ",\"rss_file\":" + std::to_string(r.rss_file) + ",\"malloc_arena\":" + std::to_string(r.malloc_arena) +
        ",\"malloc_hblkhd\":" + std::to_string(r.malloc_hblkhd) +
        ",\"malloc_uordblks\":" + std::to_string(r.malloc_uordblks) +
        ",\"others_rss_anon\":" + std::to_string(r.others_rss_anon) +
        ",\"others_processes\":" + std::to_string(r.others_processes) + ",\"declared\":{";
    bool first = true;
    for (const auto & [name, bytes] : r.declared) {
        out += (first ? "\"" : ",\"") + name + "\":" + std::to_string(bytes);
        first = false;
    }
    return out + "}}";
}

// ---------------------------------------------------------------- controls

static constexpr size_t kControl = size_t{64} << 20;
static constexpr int kEvaluations = 2;
// Readings after the last evaluation with no work between them.
static constexpr int kTailReadings = 6;
static constexpr size_t kExtent = size_t{2} << 20;

static void cuda_ok(cudaError_t e, const char * what) { require(e == cudaSuccess, what); }
static void cu_ok(CUresult e, const char * what) { require(e == CUDA_SUCCESS, what); }

struct Control {
    std::string kind;
    int repeat = 0;
    std::string where;
    Reading before, held, after;
};

static std::vector<Control> g_controls;

static void run_controls(const std::string & where, bool probe) {
    for (int repeat = 0; repeat < 3; ++repeat) {
        {
            Control c{"cudaMalloc", repeat, where, read("control"), {}, {}};
            void * p = nullptr;
            cuda_ok(cudaMalloc(&p, kControl), "cudaMalloc control");
            cuda_ok(cudaMemset(p, 1, kControl), "cudaMemset control");
            cuda_ok(cudaDeviceSynchronize(), "sync");
            c.held = read("control");
            cuda_ok(cudaFree(p), "cudaFree control");
            c.after = read("control");
            g_controls.push_back(c);
        }
        {
            Control c{"vmm", repeat, where, read("control"), {}, {}};
            CUmemAllocationProp prop{};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = 0;
            CUdeviceptr base = 0;
            cu_ok(cuMemAddressReserve(&base, kControl, kExtent, 0, 0), "reserve");
            std::vector<CUmemGenericAllocationHandle> handles(kControl / kExtent);
            for (size_t i = 0; i < handles.size(); ++i) {
                cu_ok(cuMemCreate(&handles[i], kExtent, &prop, 0), "create");
                cu_ok(cuMemMap(base + i * kExtent, kExtent, 0, handles[i], 0), "map");
            }
            CUmemAccessDesc access{};
            access.location = prop.location;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            cu_ok(cuMemSetAccess(base, kControl, &access, 1), "access");
            c.held = read("control");
            cu_ok(cuMemUnmap(base, kControl), "unmap");
            for (auto h : handles) cu_ok(cuMemRelease(h), "release");
            cu_ok(cuMemAddressFree(base, kControl), "free");
            c.after = read("control");
            g_controls.push_back(c);
        }
        {
            Control c{"host", repeat, where, read("control"), {}, {}};
            auto * p = static_cast<unsigned char *>(std::malloc(kControl));
            require(p != nullptr, "host control");
            std::memset(p, 1, kControl);
            // The compiler may not elide the allocation or the writes.
            asm volatile("" : : "r"(p) : "memory");
            c.held = read("control");
            asm volatile("" : : "r"(p) : "memory");
            std::free(p);
            c.after = read("control");
            g_controls.push_back(c);
        }
        // The pinned probe runs last and only at the end: the runtime keeps
        // freed pinned memory, so its residue would be charged to what follows.
        if (probe) {
            Control c{"pinned-probe", repeat, where, read("control"), {}, {}};
            void * p = nullptr;
            cuda_ok(cudaMallocHost(&p, kControl), "cudaMallocHost probe");
            std::memset(p, 1, kControl);
            c.held = read("control");
            cuda_ok(cudaFreeHost(p), "cudaFreeHost probe");
            c.after = read("control");
            g_controls.push_back(c);
        }
    }
}

// ---------------------------------------------------------------- trajectories (fp16_reference.cc's)

static constexpr char prompt[] =
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nKeep the label cedar-17. Count: 1, 2, 3. "
    "Text: caf\xc3\xa9, \xe4\xbd\xa0\xe5\xa5\xbd.\n"
    "Code: x = 2 + 3\nWhat label did I give you?<|im_end|>\n"
    "<|im_start|>assistant\nThe label is cedar-17. The sum is 5.";

struct Trajectory {
    std::vector<llama_token> tokens;
    std::vector<int> chunks;
    uint32_t context = 0;
    uint32_t batch = 0;
};

static Trajectory control(const llama_vocab * vocab) {
    Trajectory t;
    t.tokens.resize(512);
    const int n = llama_tokenize(vocab, prompt, sizeof(prompt) - 1, t.tokens.data(), int(t.tokens.size()),
                                 false, true);
    require(n > 40 && n <= 512, "Unexpected token count");
    t.tokens.resize(size_t(n));
    t.chunks.push_back(32);
    t.chunks.insert(t.chunks.end(), size_t(n - 32), 1);
    t.context = 512;
    t.batch = 64;
    return t;
}

static Trajectory heldout(const char * path) {
    std::ifstream file(path, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    require(bytes.size() == 1040 * 8, "Expected 1040 little-endian int64 IDs");
    Trajectory t;
    t.chunks = {16, 17};
    t.chunks.insert(t.chunks.end(), 16, 1);
    t.chunks.push_back(512);
    t.chunks.insert(t.chunks.end(), 16, 1);
    size_t total = 0;
    for (const int c : t.chunks) total += size_t(c);
    for (size_t i = 0; i < total; ++i) {
        int64_t id = 0;
        for (int b = 7; b >= 0; --b) id = (id << 8) | uint8_t(bytes[i * 8 + size_t(b)]);
        require(id >= 0 && id < 151643, "Held-out ID outside the regular vocabulary");
        t.tokens.push_back(llama_token(id));
    }
    t.context = 1024;
    t.batch = 512;
    return t;
}

// SHA-256 (FIPS 180-4), for the logits' identity.
static std::string sha256(const unsigned char * data, size_t size) {
    static constexpr uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::vector<unsigned char> msg(data, data + size);
    const uint64_t bits = uint64_t(size) * 8;
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) msg.push_back(0);
    for (int i = 7; i >= 0; --i) msg.push_back(uint8_t(bits >> (8 * i)));
    for (size_t off = 0; off < msg.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = uint32_t(msg[off + 4 * i]) << 24 | uint32_t(msg[off + 4 * i + 1]) << 16 |
                   uint32_t(msg[off + 4 * i + 2]) << 8 | uint32_t(msg[off + 4 * i + 3]);
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    char out[65];
    for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
    return out;
}

static void declare_breakdown(const llama_context * ctx) {
    for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx)) {
        const std::string name = ggml_backend_buft_name(buft);
        g_declared["model:" + name] = mb.model;
        g_declared["context:" + name] = mb.context;
        g_declared["compute:" + name] = mb.compute;
    }
}

int main(int argc, char ** argv) try {
    std::vector<Reading> readings;
    readings.push_back(read("start"));
    require(argc == 4 || argc == 5, "Usage: fp16_census MODEL OUTPUT_JSON control|heldout [IDS]");
    const std::string which = argv[3];
    require((which == "control" && argc == 4) || (which == "heldout" && argc == 5), "Unknown trajectory");

    cuda_ok(cudaSetDevice(0), "cudaSetDevice");
    cuda_ok(cudaFree(nullptr), "context");
    readings.push_back(read("context"));
    run_controls("after context", false);
    readings.push_back(read("controls"));

    ggml_backend_load_all();
    llama_backend_init();
    readings.push_back(read("backends"));

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp),
                                                                     llama_model_free);
    require(bool(model), "Model load failed");
    const auto * vocab = llama_model_get_vocab(model.get());
    require(llama_vocab_n_tokens(vocab) == 151936, "Unexpected vocabulary size");
    const Trajectory t = which == "control" ? control(vocab) : heldout(argv[4]);
    readings.push_back(read("model"));

    auto cp = llama_context_default_params();
    cp.n_ctx = t.context;
    cp.n_batch = cp.n_ubatch = t.batch;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.offload_kqv = cp.op_offload = true;
    const int n_vocab = 151936;
    // Both evaluations' logits, kept for the hash: reserved and touched
    // before the context, so they add nothing inside a chunk.
    std::vector<std::vector<float>> logits(kEvaluations,
                                           std::vector<float>(t.tokens.size() * size_t(n_vocab), 0.0f));
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    require(bool(ctx), "Context creation failed");
    declare_breakdown(ctx.get());
    // output_reserve: one output row at creation (n_seq_max).
    size_t outputs = 1;
    g_declared["output:CUDA_Host"] = outputs * size_t(n_vocab) * sizeof(float);
    readings.push_back(read("context-created"));

    for (int e = 1; e <= kEvaluations; ++e) {
        // The second evaluation runs from a cleared cache on the same
        // context, as native's does: its buffers are the first's.
        if (e > 1) llama_memory_clear(llama_get_memory(ctx.get()), true);
        int pos = 0;
        int chunk = 0;
        for (const int count : t.chunks) {
            readings.push_back(read("before", e, chunk));
            auto batch = llama_batch_init(count, 0, 1);
            batch.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                batch.token[i] = t.tokens[size_t(pos + i)];
                batch.pos[i] = pos + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = true;
            }
            const int status = llama_decode(ctx.get(), batch);
            llama_synchronize(ctx.get());
            llama_batch_free(batch);
            require(status == 0, "Decode failed");
            for (int i = 0; i < count; ++i) {
                const float * row = llama_get_logits_ith(ctx.get(), i);
                require(row != nullptr, "Missing logits");
                std::memcpy(logits[size_t(e - 1)].data() + size_t(pos + i) * size_t(n_vocab), row,
                            size_t(n_vocab) * sizeof(float));
            }
            outputs = std::max(outputs, size_t(count));
            g_declared["output:CUDA_Host"] = outputs * size_t(n_vocab) * sizeof(float);
            declare_breakdown(ctx.get());
            readings.push_back(read("chunk", e, chunk));
            pos += count;
            ++chunk;
        }
    }
    for (int i = 0; i < kTailReadings; ++i) readings.push_back(read("tail", 0, i));
    run_controls("after the evaluations", true);
    const std::string hash = sha256(reinterpret_cast<const unsigned char *>(logits[0].data()),
                                    logits[0].size() * sizeof(float));
    size_t repeat_differences = 0;
    for (int e = 1; e < kEvaluations; ++e) {
        for (size_t i = 0; i < logits[0].size(); ++i) {
            repeat_differences += std::bit_cast<uint32_t>(logits[0][i]) != std::bit_cast<uint32_t>(logits[size_t(e)][i]);
        }
    }

    std::ofstream out(argv[2]);
    out << "{\"format\":\"llmp-census/2\",\"source\":\"fp16_census (bridge)\",\"trajectory\":\"" << which
        << "\",\"fusion\":" << (std::getenv("GGML_CUDA_DISABLE_FUSION") ? "false" : "true")
        << ",\"settle_ms\":" << settle_ms() << ",\"evaluations\":" << kEvaluations
        << ",\"repeat_bit_differences\":" << repeat_differences
        << ",\"logits_sha256\":\"" << hash << "\",\"chunks\":[";
    for (size_t i = 0; i < t.chunks.size(); ++i) out << (i ? "," : "") << t.chunks[i];
    out << "],\"readings\":[";
    for (size_t i = 0; i < readings.size(); ++i) out << (i ? ",\n" : "\n") << json(readings[i]);
    out << "],\"controls\":[";
    for (size_t i = 0; i < g_controls.size(); ++i) {
        const auto & c = g_controls[i];
        out << (i ? ",\n" : "\n") << "{\"kind\":\"" << c.kind << "\",\"repeat\":" << c.repeat << ",\"where\":\""
            << c.where << "\",\"before\":" << json(c.before) << ",\"held\":" << json(c.held)
            << ",\"after\":" << json(c.after) << "}";
    }
    out << "]}\n";
    out.close();
    require(bool(out), "Output write failed");
    std::printf("%s %s logits %s, repeat differences %zu\n", which.c_str(),
                std::getenv("GGML_CUDA_DISABLE_FUSION") ? "unfused" : "fused", hash.c_str(), repeat_differences);
    ctx.reset();
    model.reset();
    llama_backend_free();
    return repeat_differences == 0 ? 0 : 1;
} catch (const std::exception & error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
