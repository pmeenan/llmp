// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_TESTS_SUPPORT_U4_DIAGNOSTIC_EXPORT_H_
#define LLMP_TESTS_SUPPORT_U4_DIAGNOSTIC_EXPORT_H_

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "base/sha256.h"
#include "engine/decoding.h"
#include "kernels/common/step_program.h"
#include "llm_test_access.h"
#include "runtime/serving.h"

namespace llmp::test_support {
namespace fs = std::filesystem;

struct U4ManualEnrollment {
  fs::path store;
  fs::path export_directory;
  std::string target_id;
  std::string drafter_id;
  std::string model;
  std::string transport;
  std::string expected_target_policy;
  std::string expected_drafter_policy;

  static std::expected<U4ManualEnrollment, std::string> Read(std::string_view expected_model,
                                                             bool needs_drafter,
                                                             bool supports_full_head) {
    const auto required = [](const char* key) -> std::expected<std::string, std::string> {
      const char* value = std::getenv(key);  // NOLINT(concurrency-mt-unsafe)
      if (!value || value[0] == '\0') return std::unexpected(std::format("missing {}", key));
      return std::string(value);
    };
    auto model = required("LLMP_U4_MODEL");
    auto store = required("LLMP_U4_STORE");
    auto target = required("LLMP_U4_TARGET_ID");
    auto transport = required("LLMP_U4_TRANSPORT");
    auto export_dir = required("LLMP_U4_EXPORT_DIR");
    auto target_policy = required("LLMP_U4_EXPECTED_TARGET_POLICY");
    if (!model || !store || !target || !transport || !export_dir || !target_policy)
      return std::unexpected(!model        ? model.error()
                             : !store      ? store.error()
                             : !target     ? target.error()
                             : !transport  ? transport.error()
                             : !export_dir ? export_dir.error()
                                           : target_policy.error());
    if (*model != expected_model)
      return std::unexpected(std::format("LLMP_U4_MODEL must be exactly {}", expected_model));
    if (!fs::path(*store).is_absolute() || !fs::path(*export_dir).is_absolute())
      return std::unexpected("enrolled store and export directory must be absolute paths");
    const auto valid_id = [](std::string_view id) {
      return id.size() == 64 && std::ranges::all_of(id, [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
             });
    };
    if (!valid_id(*target))
      return std::unexpected("LLMP_U4_TARGET_ID must be a lowercase SHA-256 ID");
    std::string drafter;
    std::string drafter_policy;
    if (needs_drafter) {
      auto value = required("LLMP_U4_DRAFTER_ID");
      auto policy = required("LLMP_U4_EXPECTED_DRAFTER_POLICY");
      if (!value) return std::unexpected(value.error());
      if (!policy) return std::unexpected(policy.error());
      if (!valid_id(*value))
        return std::unexpected("LLMP_U4_DRAFTER_ID must be a lowercase SHA-256 ID");
      drafter = std::move(*value);
      drafter_policy = std::move(*policy);
    } else if (std::getenv("LLMP_U4_DRAFTER_ID") != nullptr) {  // NOLINT(concurrency-mt-unsafe)
      return std::unexpected("LLMP_U4_DRAFTER_ID is not valid for this plain-target enrollment");
    }
    if (*transport != "configured-default" && *transport != "full-head")
      return std::unexpected("LLMP_U4_TRANSPORT must be configured-default or full-head");
    if (*transport == "full-head" && !supports_full_head)
      return std::unexpected("full-head transport is unsupported for this enrollment");
    const fs::path store_path(*store);
    std::error_code ec;
    if (!fs::is_directory(store_path / *target, ec) || ec)
      return std::unexpected("enrolled target artifact is absent from the enrolled store");
    if (needs_drafter && (!fs::is_directory(store_path / drafter, ec) || ec))
      return std::unexpected("enrolled drafter artifact is absent from the enrolled store");
    return U4ManualEnrollment{store_path, *export_dir, *target,        std::move(drafter),
                              *model,     *transport,  *target_policy, std::move(drafter_policy)};
  }
};

class U4JsonlFile {
 public:
  U4JsonlFile() = default;
  U4JsonlFile(const U4JsonlFile&) = delete;
  U4JsonlFile& operator=(const U4JsonlFile&) = delete;
  U4JsonlFile(U4JsonlFile&& other) noexcept
      : file_(std::exchange(other.file_, nullptr)),
        bytes_(std::exchange(other.bytes_, 0)),
        failed_(std::exchange(other.failed_, false)) {}
  U4JsonlFile& operator=(U4JsonlFile&& other) noexcept {
    if (this != &other) {
      const bool close_failed = file_ && std::fclose(file_) != 0;
      file_ = std::exchange(other.file_, nullptr);
      bytes_ = std::exchange(other.bytes_, 0);
      failed_ = std::exchange(other.failed_, false) || close_failed;
    }
    return *this;
  }
  ~U4JsonlFile() {
    if (file_) std::fclose(file_);
  }

  static std::expected<U4JsonlFile, std::string> Create(const fs::path& path) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0)
      return std::unexpected(std::format("create {}: {}", path.string(), std::strerror(errno)));
    if (::fchmod(fd, 0600) != 0) {
      const int error = errno;
      ::close(fd);
      return std::unexpected(std::format("protect {}: {}", path.string(), std::strerror(error)));
    }
    std::FILE* file = ::fdopen(fd, "wb");
    if (!file) {
      const int error = errno;
      ::close(fd);
      return std::unexpected(
          std::format("open stream {}: {}", path.string(), std::strerror(error)));
    }
    U4JsonlFile out;
    out.file_ = file;
    return out;
  }

  static std::expected<U4JsonlFile, std::string> AdoptRawForTest(std::FILE* file) {
    if (!file) return std::unexpected("cannot adopt a null diagnostic stream");
    U4JsonlFile out;
    out.file_ = file;
    return out;
  }

  bool Write(std::string_view bytes) {
    if (!file_ || failed_) return false;
    std::size_t at = 0;
    while (at < bytes.size()) {
      const auto n = std::fwrite(bytes.data() + at, 1, bytes.size() - at, file_);
      if (n == 0) {
        failed_ = true;
        return false;
      }
      at += n;
      bytes_ += n;
    }
    return true;
  }

  bool Line(std::string_view line) {
    return line.find('\n') == std::string_view::npos && Write(line) && Write("\n");
  }

  [[nodiscard]] std::FILE* Raw() const { return file_; }
  bool Flush() {
    if (!file_) return false;
    if (std::fflush(file_) != 0 || std::ferror(file_) != 0) failed_ = true;
    return !failed_;
  }

  std::expected<void, std::string> Close() {
    if (!file_) return std::unexpected("diagnostic file is already closed");
    bool failed = failed_ || std::ferror(file_) != 0;
    if (std::fflush(file_) != 0 || std::ferror(file_) != 0) failed = true;
    if (std::fclose(file_) != 0) failed = true;
    file_ = nullptr;
    failed_ = failed;
    if (failed) return std::unexpected("diagnostic file write, flush, or close failed");
    return {};
  }

  [[nodiscard]] std::uint64_t bytes() const { return bytes_; }
  [[nodiscard]] bool failed() const { return failed_; }

 private:
  std::FILE* file_ = nullptr;
  std::uint64_t bytes_ = 0;
  bool failed_ = false;
};

// The caller supplies the exact registered output namespace. It must not
// already exist; a failed or partial run remains there for diagnosis.
class U4DiagnosticExport {
 public:
  static std::expected<U4DiagnosticExport, std::string> Create(fs::path directory) {
    if (directory.empty() || !directory.is_absolute())
      return std::unexpected("diagnostic export directory must be an absolute enrolled path");
    std::error_code ec;
    const auto parent = directory.parent_path();
    const auto parent_status = fs::status(parent, ec);
    if (ec || !fs::is_directory(parent_status))
      return std::unexpected("diagnostic export parent must already exist");
    struct stat parent_info{};
    if (::stat(parent.c_str(), &parent_info) != 0 || parent_info.st_uid != ::geteuid() ||
        (parent_info.st_mode & 0022) != 0)
      return std::unexpected("diagnostic export parent must be owned and not group/world writable");
    if (::mkdir(directory.c_str(), 0700) != 0)
      return std::unexpected(
          std::format("create enrolled export directory: {}", std::strerror(errno)));
    struct stat info{};
    if (::stat(directory.c_str(), &info) != 0 || info.st_uid != ::geteuid() ||
        (info.st_mode & 0777) != 0700)
      return std::unexpected("diagnostic export directory is not private");
    U4DiagnosticExport out;
    out.directory_ = std::move(directory);
    auto status = U4JsonlFile::Create(out.directory_ / "status.jsonl");
    if (!status) return std::unexpected(status.error());
    out.status_.emplace(std::move(*status));
    if (!out.status_->Line("{\"status\":\"incomplete\"}"))
      return std::unexpected("writing initial incomplete status failed");
    return out;
  }

  std::expected<U4JsonlFile, std::string> Open(std::string_view name) const {
    if (name.empty() || name.find('/') != std::string_view::npos || name == "." || name == "..")
      return std::unexpected("invalid diagnostic output filename");
    return U4JsonlFile::Create(directory_ / name);
  }

  [[nodiscard]] const fs::path& directory() const { return directory_; }

  std::expected<void, std::string> Complete() {
    if (!status_) return std::unexpected("diagnostic run status is unavailable");
    if (!status_->Line("{\"status\":\"complete\"}"))
      return std::unexpected("writing complete status failed");
    auto closed = status_->Close();
    if (!closed) return closed;
    status_.reset();
    return {};
  }

 private:
  fs::path directory_;
  std::optional<U4JsonlFile> status_;
};

inline bool U4WriteJsonString(U4JsonlFile& out, std::string_view value);

inline std::expected<void, std::string> U4WriteEnrollment(
    U4DiagnosticExport& export_to, std::string_view model, std::string_view store,
    std::string_view target_id, std::string_view drafter_id, std::string_view transport,
    std::string_view expected_target_policy, std::string_view expected_drafter_policy) {
  auto file = export_to.Open("enrollment.jsonl");
  if (!file) return std::unexpected(file.error());
  auto& out = *file;
  if (!out.Write("{\"type\":\"enrollment\",\"model\":") || !U4WriteJsonString(out, model) ||
      !out.Write(",\"store\":") || !U4WriteJsonString(out, store) ||
      !out.Write(",\"target_id\":") || !U4WriteJsonString(out, target_id) ||
      !out.Write(",\"drafter_id\":") || !U4WriteJsonString(out, drafter_id) ||
      !out.Write(",\"transport\":") || !U4WriteJsonString(out, transport) ||
      !out.Write(",\"expected_target_policy\":") ||
      !U4WriteJsonString(out, expected_target_policy) ||
      !out.Write(",\"expected_drafter_policy\":") ||
      !U4WriteJsonString(out, expected_drafter_policy) || !out.Line("}"))
    return std::unexpected("writing enrollment record failed");
  return out.Close();
}

inline bool U4WriteJsonString(U4JsonlFile& out, std::string_view value) {
  if (!out.Write("\"")) return false;
  for (unsigned char c : value) {
    switch (c) {
      case '"':
        if (!out.Write("\\\"")) return false;
        break;
      case '\\':
        if (!out.Write("\\\\")) return false;
        break;
      case '\n':
        if (!out.Write("\\n")) return false;
        break;
      case '\r':
        if (!out.Write("\\r")) return false;
        break;
      case '\t':
        if (!out.Write("\\t")) return false;
        break;
      default:
        if (c < 0x20) {
          if (!out.Write(std::format("\\u{:04x}", static_cast<unsigned>(c)))) return false;
        } else if (!out.Write(std::string_view(reinterpret_cast<const char*>(&c), 1))) {
          return false;
        }
    }
  }
  return out.Write("\"");
}

inline bool U4WriteIntArray(U4JsonlFile& out, std::span<const std::int32_t> values) {
  if (!out.Write("[")) return false;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i && !out.Write(",")) return false;
    if (!out.Write(std::to_string(values[i]))) return false;
  }
  return out.Write("]");
}

inline bool U4WriteIndexArray(U4JsonlFile& out, std::span<const std::uint32_t> values) {
  if (!out.Write("[")) return false;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i && !out.Write(",")) return false;
    if (!out.Write(std::to_string(values[i]))) return false;
  }
  return out.Write("]");
}

class U4PlanRecordWriter {
 public:
  U4PlanRecordWriter(U4JsonlFile& out, std::string_view phase) : out_(out), phase_(phase) {}
  kernels::PlanRecordSink Sink() {
    return [this](kernels::PlanReportEvent event, std::string_view fragment) {
      if (invalid_) return;
      if (event == kernels::PlanReportEvent::kBegin) {
        if (open_ || !fragment.empty()) {
          invalid_ = true;
          return;
        }
        open_ = true;
        if (!out_.Write("{\"phase\":") || !U4WriteJsonString(out_, phase_) ||
            !out_.Write(",\"record\":"))
          invalid_ = true;
      } else if (event == kernels::PlanReportEvent::kFragment) {
        if (!open_ || fragment.find('\n') != std::string_view::npos || !out_.Write(fragment))
          invalid_ = true;
      } else if (event == kernels::PlanReportEvent::kEnd) {
        if (!open_ || !fragment.empty() || !out_.Line("}")) {
          invalid_ = true;
          return;
        }
        open_ = false;
        ++records_;
      } else {
        invalid_ = true;
      }
    };
  }
  std::expected<std::uint64_t, std::string> Finish() const {
    if (invalid_ || open_ || records_ == 0 || out_.failed())
      return std::unexpected("plan report framing/write failed");
    return records_;
  }

 private:
  U4JsonlFile& out_;
  std::string_view phase_;
  std::uint64_t records_ = 0;
  bool open_ = false;
  bool invalid_ = false;
};

inline std::expected<void, std::string> U4WriteObservation(
    U4DiagnosticExport& export_to, std::string_view model, std::string_view transport,
    const runtime::LlmDecodeObservation& observation) {
  auto file = export_to.Open("decode-observation.jsonl");
  if (!file) return std::unexpected(file.error());
  auto& out = *file;
  if (!out.Write(std::format("{{\"type\":\"summary\",\"model\":{},\"transport\":{},"
                             "\"count\":{},\"capacity\":{},\"incomplete\":{}}}\n",
                             std::format("\"{}\"", model), std::format("\"{}\"", transport),
                             observation.count, observation.limit,
                             observation.incomplete ? "true" : "false")))
    return std::unexpected("writing observation summary failed");
  for (std::size_t i = 0; i < observation.count; ++i) {
    const auto& event = observation.events[i];
    if (!out.Write(std::format(
            "{{\"type\":\"event\",\"index\":{},\"verify\":{},"
            "\"slot\":{},\"ordinal\":{},\"stage\":{},\"drafter\":{},"
            "\"requested_rows\":{},\"actual_rows\":{},\"kind\":{},"
            "\"head_width\":{},\"head_map_sha256\":{},\"raw_output_sha256\":{},"
            "\"head_values_sha256\":{},\"proposal_rows\":",
            i, event.verify ? "true" : "false", event.slot, event.ordinal, event.stage,
            event.drafter, event.requested_rows, event.actual_rows,
            event.kind == runtime::LlmDecodeObservation::Kind::kDeviceTokens ? "\"device-tokens\""
                                                                             : "\"head-rows\"",
            event.head_width, std::format("\"{}\"", event.head_map),
            std::format("\"{}\"", event.raw_output), std::format("\"{}\"", event.head_values))) ||
        !U4WriteIndexArray(out, event.proposal_rows) || !out.Write(",\"input_tokens\":") ||
        !U4WriteIntArray(out, event.input_tokens) || !out.Write(",\"output_tokens\":") ||
        !U4WriteIntArray(out, event.output_tokens) || !out.Write(",\"result_tokens\":") ||
        !U4WriteIntArray(out, event.result_tokens) ||
        !out.Write(std::format(",\"keep\":{}}}\n", event.keep)))
      return std::unexpected("writing observation event failed");
  }
  return out.Close();
}

inline std::expected<std::uint64_t, std::string> U4WritePlanSnapshot(runtime::Llm& model,
                                                                     U4DiagnosticExport& export_to,
                                                                     std::string_view phase) {
  const std::string name = std::format("installed-plans-{}.jsonl", phase);
  auto file = export_to.Open(name);
  if (!file) return std::unexpected(file.error());
  auto& out = *file;
  U4PlanRecordWriter writer(out, phase);
  model.ReportPlans(writer.Sink());
  auto records = writer.Finish();
  if (!records) return std::unexpected(records.error());
  if (!out.Line(std::format("{{\"type\":\"inventory-end\",\"phase\":{},"
                            "\"records\":{},\"bytes_before_end\":{}}}",
                            std::format("\"{}\"", phase), *records, out.bytes())))
    return std::unexpected("writing plan inventory completion marker failed");
  const auto bytes = out.bytes();
  if (auto closed = out.Close(); !closed) return std::unexpected(closed.error());
  return bytes;
}

inline std::string U4Sha256(std::span<const std::byte> bytes) {
  base::Sha256 hash;
  hash.Update(bytes);
  return base::ToHex(hash.Finish());
}

inline std::string U4Sha256Floats(std::span<const float> values) {
  return U4Sha256(std::as_bytes(values));
}

inline std::string U4Sha256Ints(std::span<const std::int32_t> values) {
  return U4Sha256(std::as_bytes(values));
}

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_U4_DIAGNOSTIC_EXPORT_H_
