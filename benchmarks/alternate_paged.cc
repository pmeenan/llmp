// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// BP-S3 (docs/backend-proof.md): models using different kernel sources,
// resident and run alternately in one process, with correct accounting of
// shared workspace. The FP16 fixture (GGML's kernels and cuBLAS,
// fp16_runner.h) and an EXL3 fixture (ExLlamaV3's kernels and GGML's,
// exl3_runner.h) on one paged node (tests/support/paged_node.h): one
// catalog domain, one scheduler with its lanes, one landing zone, and one
// workspace (the activations and the GGML pool) that both models' plans
// bind. Each model runs on its own stream, with its own launch contexts.
//
//   llmp_alternate_paged --fp16-artifact DIR --trajectory control|heldout
//                          --tokens FILE --fusion on|off
//                          --exl3-artifact DIR --fixture 4.0bpw|4.5bpw
//                          --arm G|O --plan PLAN.txt --ids FILE
//                          --out DIR [--rounds N] [--prefixes LIST]
//                          [--fp16-expect SHA256] [--exl3-expect DIR]
//
// - The execution budget B is the node's fixed occupancy (the zone, both
//   caches, the workspace, the cuBLAS workspace, the EXL3 lock area, norms
//   and tables, and the pinned staging) plus the larger model's weights
//   plus half the smaller's: never both models' weights at once.
// - Each round evaluates FP16, then EXL3. Before each evaluation the
//   model's whole closure is acquired (test_support::AcquireProgram): the
//   memory module plans the materialization against B, the victims it
//   chooses (the other model's clean weights, least recently used first)
//   are evicted, their backing released, and what is missing is paged in
//   through the zone. The evaluation then runs its trajectory as a
//   standalone run's first evaluation does, each job under a lease.
// - The EXL3 norms and tables are derived once, after its first
//   acquisition; they are pinned and its weights always come back at the
//   same place.
// - Checked (exit 1 on any failure): every evaluation's logits equal that
//   model's first bit for bit, and the given rung-3 hashes (FP16: the
//   logits' SHA-256; EXL3: the data of rung 3's logits-P.prefill.npy and
//   logits-P.suffix.npy for every prefix); every acquisition after the
//   first evicted only the other model's weights, and at least one; the
//   catalog's occupancy never exceeded B; the scratch class was charged
//   exactly the shared workspace, once; and each model's coverage and
//   bounds (its own summary.json and paging.json in DIR/fp16 and
//   DIR/exl3).
// - Reported, loosely (D-085): the catalog's peak occupancy, and the
//   device memory in use (cudaMemGetInfo) at each sample.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <print>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "catalog/catalog.h"
#include "exl3_common.h"
#include "exl3_runner.h"
#include "fp16_runner.h"
#include "paged_node.h"
#include "paged_programs.h"

namespace {

namespace ts = llmp::test_support;
namespace catalog = llmp::catalog;
using llmp::base::Bytes;
using llmp::benchmarks::Exl3Options;
using llmp::benchmarks::Exl3Runner;
using llmp::benchmarks::Fp16Options;
using llmp::benchmarks::Fp16Runner;
using Status = ts::Status;

constexpr int kFp16 = 0;  // owner and stream
constexpr int kExl3 = 1;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

double Seconds(std::chrono::steady_clock::duration d) {
  return std::chrono::duration<double>(d).count();
}

struct Options {
  Fp16Options fp16;
  Exl3Options exl3;
  std::filesystem::path out;
  int rounds = 3;
  std::string fp16_expect;
  std::filesystem::path exl3_expect;
};

std::expected<Options, std::string> Parse(std::span<char*> args) {
  Options o;
  bool fusion_set = false;
  bool arm = false;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view a = args[i];
    if (i + 1 >= args.size()) {
      return Error(std::format("{} needs a value", a));
    }
    const std::string_view v = args[++i];
    if (a == "--fp16-artifact") {
      o.fp16.artifact = v;
    } else if (a == "--trajectory") {
      o.fp16.trajectory = v;
    } else if (a == "--tokens") {
      o.fp16.tokens = v;
    } else if (a == "--fusion") {
      o.fp16.fusion = v == "on";
      fusion_set = v == "on" || v == "off";
    } else if (a == "--exl3-artifact") {
      o.exl3.artifact = v;
    } else if (a == "--fixture") {
      o.exl3.fixture = v;
    } else if (a == "--arm") {
      if (v != "G" && v != "O") {
        return Error("--arm is G or O");
      }
      o.exl3.arm = v == "G" ? llmp::model::Exl3Arm::kG : llmp::model::Exl3Arm::kO;
      arm = true;
    } else if (a == "--plan") {
      o.exl3.plan = v;
    } else if (a == "--ids") {
      o.exl3.ids = v;
    } else if (a == "--prefixes") {
      o.exl3.prefixes.clear();
      std::istringstream list{std::string(v)};
      for (std::string item; std::getline(list, item, ',');) {
        int prefix = 0;
        const auto [end, error] = std::from_chars(item.data(), item.data() + item.size(), prefix);
        if (error != std::errc() || end != item.data() + item.size() || prefix <= 0) {
          return Error("--prefixes takes positive integers");
        }
        o.exl3.prefixes.push_back(prefix);
      }
    } else if (a == "--out") {
      o.out = v;
    } else if (a == "--rounds") {
      if (std::from_chars(v.data(), v.data() + v.size(), o.rounds).ec != std::errc{} ||
          o.rounds < 1 || o.rounds > 16) {
        return Error("--rounds takes a count from 1 to 16");
      }
    } else if (a == "--fp16-expect") {
      o.fp16_expect = v;
    } else if (a == "--exl3-expect") {
      o.exl3_expect = v;
    } else {
      return Error(std::format("unknown argument {}", a));
    }
  }
  if (o.fp16.artifact.empty() || o.fp16.trajectory.empty() || o.fp16.tokens.empty() ||
      !fusion_set || o.exl3.artifact.empty() ||
      (o.exl3.fixture != "4.0bpw" && o.exl3.fixture != "4.5bpw") || !arm || o.exl3.plan.empty() ||
      o.exl3.ids.empty() || o.exl3.prefixes.empty() || o.out.empty()) {
    return Error(
        "usage: llmp_alternate_paged --fp16-artifact DIR --trajectory control|heldout "
        "--tokens FILE --fusion on|off --exl3-artifact DIR --fixture 4.0bpw|4.5bpw --arm G|O "
        "--plan PLAN.txt --ids FILE --out DIR [--rounds N] [--prefixes LIST] "
        "[--fp16-expect SHA256] [--exl3-expect DIR]");
  }
  o.fp16.out = o.out / "fp16";
  o.exl3.out = o.out / "exl3";
  return o;
}

std::string Sha256(std::span<const std::byte> bytes) {
  llmp::base::Sha256 hash;
  hash.Update(bytes);
  return llmp::base::ToHex(hash.Finish());
}

// The SHA-256 of a .npy file's data (format 1.0 or 2.0).
std::expected<std::string, std::string> NpyDataSha256(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  const std::vector<char> bytes{std::istreambuf_iterator<char>(file),
                                std::istreambuf_iterator<char>()};
  if (bytes.size() < 12 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0) {
    return Error(std::format("{} is not a .npy file", path.string()));
  }
  const auto byte = [&](std::size_t i) { return static_cast<std::uint8_t>(bytes[i]); };
  std::size_t data = 0;
  if (byte(6) == 1) {
    data = 10 + (std::size_t{byte(8)} | (std::size_t{byte(9)} << 8U));
  } else {
    data = 12 + (std::size_t{byte(8)} | (std::size_t{byte(9)} << 8U) |
                 (std::size_t{byte(10)} << 16U) | (std::size_t{byte(11)} << 24U));
  }
  if (data > bytes.size()) {
    return Error(std::format("{} is truncated", path.string()));
  }
  return Sha256(std::as_bytes(std::span(bytes)).subspan(data));
}

// One evaluation: which model, what its acquisition did, and its hashes.
struct Evaluation {
  int model = kFp16;
  int round = 0;
  std::uint64_t evicted_other = 0;
  std::uint64_t evicted_own = 0;
  std::uint64_t evicted_else = 0;
  std::uint64_t loaded = 0;
  double acquire_seconds = 0;
  double run_seconds = 0;
  std::map<std::string, std::string> hashes;  // "logits", or "P.prefill"/"P.suffix"
  bool equals_first = true;
  bool equals_expected = true;
};

// A sample of the node's memory: the catalog's occupancy (on the
// scheduler's thread) and the device's memory in use.
struct MemorySample {
  std::uint64_t total = 0;
  std::uint64_t scratch = 0;
  std::uint64_t weights = 0;
  std::uint64_t device_used = 0;
};

class Alternation {
 public:
  explicit Alternation(const Options& options)
      : o_(options),
        node_({.compute_streams = 2,
               .slots = ts::kPagedSlots,
               .inline_lanes = false,
               .coalesce = false}),
        fp16_(node_, o_.fp16, kFp16, kFp16, nullptr, record_),
        exl3_(node_, o_.exl3, kExl3, kExl3) {}

  Status Run();
  Status TearDown() { return node_.TearDown(entered_models_); }

 private:
  Status Sample(std::string_view when);
  Status Write();

  const Options& o_;
  std::string record_;
  ts::PagedNode node_;
  Fp16Runner fp16_;
  Exl3Runner exl3_;
  std::vector<ts::PagedModel*> entered_models_;

  std::set<catalog::ExtentId> fp16_weights_;
  std::set<catalog::ExtentId> exl3_weights_;
  std::uint64_t fixed_ = 0;
  std::uint64_t separate_workspace_ = 0;
  std::uint64_t device_before_ = 0;
  std::vector<Evaluation> evaluations_;
  std::vector<std::pair<std::string, MemorySample>> samples_;
  std::vector<std::vector<float>> fp16_results_;
  std::vector<std::map<int, std::vector<float>>> exl3_results_;
  std::map<std::string, std::string> exl3_expected_;
  std::map<int, std::map<std::string, std::string>> first_hashes_;  // by model
  std::vector<std::string> problems_;
};

std::uint64_t DeviceUsed() {
  std::size_t free = 0;
  std::size_t total = 0;
  if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
    return 0;
  }
  return total - free;
}

Status Alternation::Sample(std::string_view when) {
  MemorySample sample;
  if (auto r = node_.Call(
          [&]() -> Status {
            const catalog::Occupancy occupancy = node_.catalog().OccupancyOf(node_.domain());
            sample.total = occupancy.Total().value();
            sample.scratch =
                occupancy.by_class.at(static_cast<std::size_t>(catalog::MemoryClass::kScratch))
                    .value();
            sample.weights =
                occupancy.by_class.at(static_cast<std::size_t>(catalog::MemoryClass::kWeights))
                    .value();
            return {};
          },
          "sampling the occupancy");
      !r) {
    return r;
  }
  sample.device_used = DeviceUsed();
  if (sample.total > node_.budget().value()) {
    problems_.push_back(std::format("{}: the occupancy {} exceeds the budget {}", when,
                                    sample.total, node_.budget().value()));
  }
  const std::uint64_t workspace = node_.activations().bytes + node_.pool().bytes;
  if (sample.scratch != workspace) {
    problems_.push_back(std::format("{}: the scratch class holds {} bytes, the workspace is {}",
                                    when, sample.scratch, workspace));
  }
  samples_.emplace_back(std::string(when), sample);
  return {};
}

Status Alternation::Run() {
  device_before_ = DeviceUsed();
  if (auto r = node_.Open(); !r) {
    return r;
  }
  entered_models_.push_back(&fp16_);
  if (auto r = fp16_.Setup(); !r) {
    return r;
  }
  entered_models_.push_back(&exl3_);
  if (auto r = exl3_.Setup(); !r) {
    return r;
  }
  // One workspace for both: each part at the larger model's need.
  separate_workspace_ = 0;
  for (const std::uint64_t need : {fp16_.activations_needed(), fp16_.pool_needed(),
                                   exl3_.activations_needed(), exl3_.pool_needed()}) {
    separate_workspace_ += (std::max<std::uint64_t>(need, 1) + ts::kPagedExtent - 1) /
                           ts::kPagedExtent * ts::kPagedExtent;
  }
  if (auto r = node_.MapWorkspace(std::max(fp16_.activations_needed(), exl3_.activations_needed()),
                                  std::max(fp16_.pool_needed(), exl3_.pool_needed()));
      !r) {
    return r;
  }
  // B: what is resident now (nothing of either model's weights), the
  // larger model's weights and half the smaller's.
  fixed_ = node_.catalog().OccupancyOf(node_.domain()).Total().value();
  const std::vector<catalog::ExtentId> fp16_weights = fp16_.weights();
  fp16_weights_.insert(fp16_weights.begin(), fp16_weights.end());
  exl3_weights_.insert(exl3_.weights().begin(), exl3_.weights().end());
  const std::uint64_t a = fp16_weights_.size() * ts::kPagedExtent;
  const std::uint64_t b = exl3_weights_.size() * ts::kPagedExtent;
  const std::uint64_t budget =
      fixed_ + std::max(a, b) +
      (((std::min(a, b) / 2) + ts::kPagedExtent - 1) / ts::kPagedExtent * ts::kPagedExtent);
  if (auto r = node_.Start(Bytes(budget)); !r) {
    return r;
  }
  if (auto r = fp16_.Register(); !r) {
    return r;
  }
  if (auto r = exl3_.Register(); !r) {
    return r;
  }
  if (auto r = fp16_.Bind(); !r) {
    return r;
  }
  if (auto r = exl3_.Bind(); !r) {
    return r;
  }
  node_.Run();
  if (auto r = Sample("start"); !r) {
    return r;
  }
  if (!o_.exl3_expect.empty()) {
    for (const int prefix : o_.exl3.prefixes) {
      for (const std::string_view part : {"prefill", "suffix"}) {
        auto hash = NpyDataSha256(o_.exl3_expect / std::format("logits-{}.{}.npy", prefix, part));
        if (!hash) {
          return std::unexpected(hash.error());
        }
        exl3_expected_[std::format("{}.{}", prefix, part)] = *hash;
      }
    }
  }

  bool derived = false;
  for (int round = 1; round <= o_.rounds; ++round) {
    for (const int model : {kFp16, kExl3}) {
      Evaluation e;
      e.model = model;
      e.round = round;
      const std::string name = model == kFp16 ? "FP16" : "EXL3";
      ts::AcquireReport report;
      auto start = std::chrono::steady_clock::now();
      if (auto r = node_.Acquire(model == kFp16 ? fp16_.everything() : exl3_.everything(), report,
                                 std::format("acquiring {} (round {})", name, round));
          !r) {
        return r;
      }
      e.acquire_seconds = Seconds(std::chrono::steady_clock::now() - start);
      e.loaded = report.loaded;
      const auto& own = model == kFp16 ? fp16_weights_ : exl3_weights_;
      const auto& other = model == kFp16 ? exl3_weights_ : fp16_weights_;
      for (const catalog::ExtentId extent : report.evicted) {
        if (own.contains(extent)) {
          ++e.evicted_own;
        } else if (other.contains(extent)) {
          ++e.evicted_other;
        } else {
          ++e.evicted_else;
        }
      }
      if (e.evicted_own + e.evicted_else != 0) {
        problems_.push_back(
            std::format("acquiring {} (round {}) evicted {} of its own extents "
                        "and {} that are neither model's weights",
                        name, round, e.evicted_own, e.evicted_else));
      }
      // The first acquisition finds room; every later one must evict.
      if ((round != 1 || model != kFp16) && e.evicted_other == 0) {
        problems_.push_back(
            std::format("acquiring {} (round {}) evicted nothing of the other model", name, round));
      }
      if (auto r = Sample(std::format("{} {} acquired", name, round)); !r) {
        return r;
      }
      start = std::chrono::steady_clock::now();
      if (model == kFp16) {
        std::vector<float>& result = fp16_results_.emplace_back();
        if (auto r = fp16_.Evaluate(1, result); !r) {
          return r;
        }
        e.hashes["logits"] = Sha256(std::as_bytes(std::span(result)));
        e.equals_expected = o_.fp16_expect.empty() || e.hashes["logits"] == o_.fp16_expect;
      } else {
        if (!derived) {
          if (auto r = exl3_.Derive(); !r) {
            return r;
          }
          derived = true;
        }
        auto& result = exl3_results_.emplace_back();
        if (auto r = exl3_.Evaluate(1, result); !r) {
          return r;
        }
        for (const auto& [prefix, values] : result) {
          const auto bytes = std::as_bytes(std::span(values));
          const std::size_t prefill =
              static_cast<std::size_t>(prefix) * llmp::model::Qwen25Instruct05BExl3().vocab * 4;
          e.hashes[std::format("{}.prefill", prefix)] = Sha256(bytes.first(prefill));
          e.hashes[std::format("{}.suffix", prefix)] = Sha256(bytes.subspan(prefill));
        }
        e.equals_expected = exl3_expected_.empty() || e.hashes == exl3_expected_;
      }
      // Bit for bit: the hashes are of the logits' bytes.
      const auto first = first_hashes_.try_emplace(model, e.hashes).first;
      e.equals_first = e.hashes == first->second;
      e.run_seconds = Seconds(std::chrono::steady_clock::now() - start);
      if (!e.equals_first || !e.equals_expected) {
        problems_.push_back(std::format("{} (round {}): the logits differ from {}", name, round,
                                        e.equals_expected ? "its first evaluation's" : "rung 3's"));
      }
      if (auto r = Sample(std::format("{} {} evaluated", name, round)); !r) {
        return r;
      }
      evaluations_.push_back(std::move(e));
    }
  }
  return Write();
}

Status Alternation::Write() {
  // Each model's own outputs: logits, coverage and bounds against peaks.
  if (auto r = fp16_.Write(fp16_results_); !r) {
    problems_.push_back("FP16: " + r.error());
  }
  if (auto r = exl3_.Write(exl3_results_); !r) {
    problems_.push_back("EXL3: " + r.error());
  }
  std::string evaluations;
  for (const Evaluation& e : evaluations_) {
    std::string hashes;
    for (const auto& [key, hash] : e.hashes) {
      hashes += std::format(R"({}"{}":"{}")", hashes.empty() ? "" : ",", key, hash);
    }
    evaluations += std::format(
        R"({}{{"model":"{}","round":{},"evicted_other":{},"evicted_own":{},"evicted_else":{},)"
        R"("loaded":{},"acquire_seconds":{:.6f},"run_seconds":{:.6f},"equals_first":{},)"
        R"("equals_expected":{},"hashes":{{{}}}}})",
        evaluations.empty() ? "" : ",\n  ", e.model == kFp16 ? "FP16" : "EXL3", e.round,
        e.evicted_other, e.evicted_own, e.evicted_else, e.loaded, e.acquire_seconds, e.run_seconds,
        e.equals_first ? "true" : "false", e.equals_expected ? "true" : "false", hashes);
  }
  std::string samples;
  std::uint64_t peak_total = 0;
  std::uint64_t peak_device = 0;
  for (const auto& [when, s] : samples_) {
    peak_total = std::max(peak_total, s.total);
    peak_device = std::max(peak_device, s.device_used);
    samples += std::format(
        R"({}{{"when":"{}","occupancy":{},"scratch":{},"weights":{},"device_used":{}}})",
        samples.empty() ? "" : ",\n  ", when, s.total, s.scratch, s.weights, s.device_used);
  }
  std::string problems;
  for (const std::string& problem : problems_) {
    problems += std::format(R"({}"{}")", problems.empty() ? "" : ",", problem);
  }
  std::filesystem::create_directories(o_.out);
  std::ofstream file(o_.out / "alternation.json");
  file << std::format(
      "{{\"rounds\":{},\"budget\":{},\"fixed\":{},\"fp16_weight_extents\":{},"
      "\"exl3_weight_extents\":{},\"workspace\":{{\"activations\":{},\"pool\":{},"
      "\"separate\":{}}},\"fp16_expect\":\"{}\",\"exl3_expect\":\"{}\","
      "\"peak_occupancy\":{},\"device_used_before\":{},\"peak_device_used\":{},"
      "\"problems\":[{}],\n \"evaluations\":[\n  {}],\n \"samples\":[\n  {}]}}\n",
      o_.rounds, node_.budget().value(), fixed_, fp16_weights_.size(), exl3_weights_.size(),
      node_.activations().bytes, node_.pool().bytes, separate_workspace_, o_.fp16_expect,
      o_.exl3_expect.string(), peak_total, device_before_, peak_device, problems, evaluations,
      samples);
  std::println("wrote {}", (o_.out / "alternation.json").string());
  if (!problems_.empty()) {
    std::string all;
    for (const std::string& problem : problems_) {
      all += (all.empty() ? "" : "; ") + problem;
    }
    return Error(all);
  }
  return {};
}

}  // namespace

int main(int argc, char** argv) {
  const auto options = Parse(std::span(argv, static_cast<std::size_t>(argc)));
  if (!options) {
    std::println(stderr, "{}", options.error());
    return 2;
  }
  Status ran;
  {
    Alternation alternation(*options);
    ran = alternation.Run();
    if (auto finished = alternation.TearDown(); !finished) {
      if (!ran) std::println(stderr, "FAILED: {}", ran.error());
      std::println(stderr, "retirement failed: {}", finished.error());
      std::abort();
    }
  }
  if (!ran) {
    std::println(stderr, "FAILED: {}", ran.error());
    return 1;
  }
  std::println("DONE {}", options->out.string());
  return 0;
}
