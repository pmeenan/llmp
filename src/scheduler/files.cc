// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "scheduler/files.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <mutex>

#include "base/check.h"

namespace llmp::scheduler {

const FileTable::Entry* FileTable::Find(FileId file) const {
  if (file.generation() != 1 || file.index() >= files_.size()) {
    return nullptr;
  }
  return &files_[file.index()];
}

FileTable::Entry* FileTable::Find(FileId file) {
  if (file.generation() != 1 || file.index() >= files_.size()) {
    return nullptr;
  }
  return &files_[file.index()];
}

FileId FileTable::Add() {
  const std::scoped_lock lock(mutex_);
  files_.emplace_back();
  return {static_cast<std::uint32_t>(files_.size() - 1), 1};
}

bool FileTable::Known(FileId file) const {
  const std::scoped_lock lock(mutex_);
  return Find(file) != nullptr;
}

std::expected<void, FileRefusal> FileTable::Install(FileId file, int fd) {
  if (fd < 0) {
    return std::unexpected(FileRefusal::kInvalid);
  }
  const std::scoped_lock lock(mutex_);
  Entry* entry = Find(file);
  if (entry == nullptr) {
    return std::unexpected(FileRefusal::kUnknown);
  }
  if (entry->fd >= 0) {
    return std::unexpected(FileRefusal::kOpen);
  }
  entry->fd = fd;
  return {};
}

std::expected<int, FileRefusal> FileTable::Take(FileId file) {
  const std::scoped_lock lock(mutex_);
  Entry* entry = Find(file);
  if (entry == nullptr) {
    return std::unexpected(FileRefusal::kUnknown);
  }
  if (entry->reads != 0) {
    return std::unexpected(FileRefusal::kReading);
  }
  const int fd = entry->fd;
  entry->fd = -1;
  return fd;
}

int FileTable::Descriptor(FileId file) const {
  const std::scoped_lock lock(mutex_);
  const Entry* entry = Find(file);
  return entry != nullptr ? entry->fd : -1;
}

std::uint32_t FileTable::Reads(FileId file) const {
  const std::scoped_lock lock(mutex_);
  const Entry* entry = Find(file);
  return entry != nullptr ? entry->reads : 0;
}

std::size_t FileTable::open() const {
  const std::scoped_lock lock(mutex_);
  return static_cast<std::size_t>(
      std::ranges::count_if(files_, [](const Entry& entry) { return entry.fd >= 0; }));
}

int FileTable::BeginRead(FileId file) {
  const std::scoped_lock lock(mutex_);
  Entry* entry = Find(file);
  if (entry == nullptr || entry->fd < 0) {
    return -1;
  }
  ++entry->reads;
  return entry->fd;
}

void FileTable::EndRead(FileId file) {
  const std::scoped_lock lock(mutex_);
  Entry* entry = Find(file);
  base::Check(entry != nullptr && entry->reads > 0, "ending a read its file did not begin");
  --entry->reads;
}

}  // namespace llmp::scheduler
