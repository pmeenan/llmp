// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/json.h"
#include "base/sha256.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/qwen38_runner.h"
#include "engine/support.h"
#include "model/qwen38.h"
#include "qwen38_batch.h"

namespace {
namespace en = llmp::engine;
namespace batch = llmp::benchmarks::qwen_batch;
using en::support::Error;
using Status = en::Status;

struct Options {
  en::Qwen38Options model;
  batch::Requests<std::filesystem::path> inputs;
  std::filesystem::path out;
  bool natural = false;
  std::uint32_t requests = 2;
};

std::expected<Options, std::string> Parse(std::span<char* const> args) {
  if (args.size() != 11 && args.size() != 13 && args.size() != 17) {
    return Error(
        "usage: qwen38_batch --artifact DIR --drafter DIR --input0 FILE --input1 FILE [--input2 "
        "FILE --input3 FILE] --out NEWDIR [--mode natural|natural4]");
  }
  Options o;
  const std::array<std::string_view, 5> two = {"--artifact", "--drafter", "--input0", "--input1",
                                               "--out"};
  const std::array<std::string_view, 7> four = {"--artifact", "--drafter", "--input0", "--input1",
                                                "--input2",   "--input3",  "--out"};
  const auto keys = args.size() == 17 ? std::span<const std::string_view>(four)
                                      : std::span<const std::string_view>(two);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (std::string_view(args[1 + (2 * i)]) != keys[i] || args[2 + (2 * i)][0] == '\0') {
      return Error("C2 proof arguments must use the documented fixed order");
    }
  }
  o.model.artifact = args[2];
  o.model.drafter = args[4];
  o.inputs = {args[6], args[8], {}, {}};
  o.out = args[10];
  if (args.size() == 13) {
    if (std::string_view(args[11]) != "--mode" || std::string_view(args[12]) != "natural") {
      return Error("C2 optional mode must be exactly --mode natural");
    }
    o.natural = true;
  }
  if (args.size() == 17) {
    if (std::string_view(args[15]) != "--mode" || std::string_view(args[16]) != "natural4") {
      return Error("C4 mode must be exactly --mode natural4 with four inputs");
    }
    o.inputs[2] = args[10];
    o.inputs[3] = args[12];
    o.out = args[14];
    o.natural = true;
    o.requests = 4;
  }
  o.model.out = o.out / "state";
  o.model.context = batch::Proof::kContext;
  o.model.max_rows = 4096;
  o.model.draft_rows = 3;
  o.model.draft_vocab = 47172;
  o.model.graphs = false;
  o.model.draft_head_capture = true;
  return o;
}

std::expected<std::vector<std::int32_t>, std::string> Read(const std::filesystem::path& path) {
  static_assert(std::endian::native == std::endian::little);
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec) || ec ||
      std::filesystem::file_size(path, ec) != 8192 * sizeof(std::int32_t) || ec) {
    return Error("C2 packed I32 fixture must contain exactly8192 IDs");
  }
  std::ifstream input(path, std::ios::binary);
  std::vector<std::int32_t> ids(8192);
  input.read(reinterpret_cast<char*>(ids.data()),
             static_cast<std::streamsize>(ids.size() * sizeof(std::int32_t)));
  if (!input || input.gcount() != static_cast<std::streamsize>(ids.size() * sizeof(std::int32_t)) ||
      input.peek() != std::char_traits<char>::eof() || !std::ranges::all_of(ids, [](auto id) {
        return id >= 0 && std::cmp_less(id, llmp::model::Qwen38Flash().vocab);
      })) {
    return Error("C2 packed fixture short/extra bytes or token outside vocabulary");
  }
  return ids;
}

std::string Sha(std::span<const std::int32_t> ids) {
  llmp::base::Sha256 sha;
  sha.Update(std::as_bytes(ids));
  return llmp::base::ToHex(sha.Finish());
}

Status Execute(en::PagedNode& node, en::Qwen38Runner& owner, batch::Proof& proof,
               const Options& options, const batch::Requests<std::vector<std::int32_t>>& inputs,
               std::vector<en::PagedModel*>& entered) {
  if (auto r = node.Open(); !r) {
    return r;
  }
  entered.push_back(&owner);
  if (auto r = owner.Setup(); !r) {
    return r;
  }
  entered.push_back(&proof);
  if (auto r = proof.Setup(); !r) {
    return r;
  }
  if (auto r = node.MapWorkspace(std::max(owner.activations_needed(), proof.activations_needed()),
                                 std::max(owner.pool_needed(), proof.pool_needed()));
      !r) {
    return r;
  }
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total();
  const auto state = llmp::base::Bytes(node.StateCapacity());
  const auto weights = llmp::base::Bytes(owner.weights().size() * en::kPagedExtent);
  // Bounded host control vectors/graph metadata have their own conservative
  // 64MiB per request, separate from cataloged pinned/device allocations.
  const auto host = fixed.Plus(llmp::base::Bytes(std::uint64_t{options.requests} << 26));
  const auto first = host ? host->Plus(state) : std::nullopt;
  const auto budget = first ? first->Plus(weights) : std::nullopt;
  if (!budget || budget->value() > (std::uint64_t{100} << 30)) {
    return Error("C2 explicit execution budget exceeds100GiB or overflows");
  }
  if (auto r = node.Start(*budget); !r) {
    return r;
  }
  if (auto r = owner.Register(); !r) {
    return r;
  }
  if (auto r = proof.Register(); !r) {
    return r;
  }
  if (auto r = owner.Bind(); !r) {
    return r;
  }
  if (auto r = proof.Bind(); !r) {
    return r;
  }
  node.Run();
  std::vector<en::LoadStats> loads;
  if (auto r = node.Load(owner.weights(), "C2 single shared target/drafter residency", loads); !r) {
    return r;
  }
  if (auto r = owner.ReadPleHash(); !r) {
    return r;
  }
  return options.natural ? proof.GenerationControls(inputs, options.out)
                         : proof.Controls(inputs, options.out);
}
}  // namespace

int main(int argc, char** argv) {
  const auto options = Parse({argv, static_cast<std::size_t>(argc)});
  if (!options) {
    std::println(stderr, "{}", options.error());
    return 2;
  }
  batch::Requests<std::vector<std::int32_t>> inputs;
  for (std::size_t s = 0; s < options->requests; ++s) {
    auto ids = Read(options->inputs[s]);
    if (!ids) {
      std::println(stderr, "{}", ids.error());
      return 2;
    }
    inputs[s] = std::move(*ids);
  }
  std::error_code ec;
  if (!std::filesystem::create_directory(options->out, ec) || ec) {
    std::println(stderr, "C2 output must be a fresh directory: {}", ec.message());
    return 2;
  }
  if (!std::filesystem::create_directory(options->model.out, ec) || ec) {
    std::println(stderr, "C2 state directory creation failed: {}", ec.message());
    return 2;
  }
  en::PagedNode node({.compute_streams = 1, .slot_bytes = en::kSlabSlotBytes});
  en::Qwen38Runner owner(node, options->model, 0, 0);
  batch::Proof proof(owner, options->natural, options->requests);
  std::vector<en::PagedModel*> entered;
  const auto start = std::chrono::steady_clock::now();
  auto ran = Execute(node, owner, proof, *options, inputs, entered);
  // Both objects remain alive and at stable addresses throughout cleanup,
  // including failed partial setup. Open can fail before node.memory exists;
  // only models whose Setup was entered may release memory-backed resources.
  // Shared resources release after slot plans when both were entered.
  std::ranges::reverse(entered);
  const auto retired = node.TearDown(entered);
  std::string error;
  if (!ran) {
    error = ran.error();
  }
  if (!retired) {
    error += (error.empty() ? "" : "; ") + retired.error();
  }
  std::string quoted;
  llmp::base::json::AppendQuoted(error, quoted);
  const double seconds = en::support::Seconds(std::chrono::steady_clock::now() - start);
  std::string input_sha;
  for (std::size_t s = 0; s < options->requests; ++s) {
    input_sha += std::format("{}\"{}\"", input_sha.empty() ? "" : ",", Sha(inputs[s]));
  }
  const auto record = std::format(
      R"({{"complete":{},"retired":{},"diagnostic_wall_seconds":{},"input_sha256":[{}],)"
      R"("error":{},"proof":{}}})",
      ran && retired ? "true" : "false", retired ? "true" : "false", seconds, input_sha, quoted,
      proof.receipt());
  std::ofstream output(options->out / "receipt.json", std::ios::binary);
  output << record << '\n';
  output.close();
  if (!output) {
    std::println(stderr, "C2 receipt flush failed");
    if (!retired) {
      std::abort();
    }
    return 1;
  }
  std::println("{}", record);
  // Failed retirement proves no completion. Preserve the error record above,
  // then let the process reclaim captured graphs/plans and pinned resources;
  // their ordinary destructors must not free owners still named by GPU work.
  if (!retired) {
    std::abort();
  }
  return ran && retired ? 0 : 1;
}
