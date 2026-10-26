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

#include "kernels/common/step_program.h"

// The private JSON-lines export directory and plan-record framing the U4
// diagnostics used (removed 2026-10-26); engine_llm_test.cc's
// U4DiagnosticExporterTest is their only remaining user.
namespace llmp::test_support {
namespace fs = std::filesystem;

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

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_U4_DIAGNOSTIC_EXPORT_H_
