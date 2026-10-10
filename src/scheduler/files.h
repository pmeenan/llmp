// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The files page sources read by identity (PageSource::file): a model's
// weight shards, whose descriptors are open only while the model is
// active (engine/paged_node.h, PagedNode::SetActive; docs/architecture.md#storage-service).
// A source names the file, not a descriptor, so the descriptor can be
// opened and closed while the source stays registered; the scheduler
// takes the descriptor when a read starts.
//
// The table is shared by the scheduler's thread and the node's driver,
// under a mutex of its own held only for a few loads and stores (never
// across a system call; D-007's rule for the scheduling lock): the driver
// installs a descriptor it opened and takes one back to close, and the
// scheduler begins and ends reads. Completion-aware (docs/async-model.md):
// a read that began on a descriptor counts until the scheduler concludes
// its operation, and while any does, Take refuses, so a descriptor is
// never closed (and its number never reused) under a read that may still
// be submitted or in flight. A read whose operation is quarantined never
// concludes, so its file keeps its descriptor to the process's end.
//
// A read that starts while its file is closed gets descriptor -1: the
// storage provider refuses it (EBADF), nothing is read, and the load fails
// as any failed read does. Files are never removed: a model's weight files
// stay registered for the node's life.

#ifndef LLMP_SCHEDULER_FILES_H_
#define LLMP_SCHEDULER_FILES_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <mutex>
#include <vector>

#include "base/ids.h"

namespace llmp::scheduler {

struct FileTag {
  static constexpr const char* kName = "file";
};
using FileId = base::Id<FileTag>;

enum class FileRefusal : std::uint8_t {
  kUnknown,  // no such file
  kOpen,     // Install: the file has a descriptor already
  kReading,  // Take: reads that began on its descriptor have not concluded
  kInvalid,  // Install: a negative descriptor
};

class FileTable {
 public:
  FileTable() = default;
  FileTable(const FileTable&) = delete;
  FileTable& operator=(const FileTable&) = delete;
  FileTable(FileTable&&) = delete;
  FileTable& operator=(FileTable&&) = delete;
  ~FileTable() = default;

  // Any thread. A new file, closed.
  FileId Add();
  bool Known(FileId file) const;
  // Any thread. Gives a closed file the descriptor `fd`, which stays the
  // caller's to close once Take returns it.
  std::expected<void, FileRefusal> Install(FileId file, int fd);
  // Any thread. The file's descriptor, taken back for its caller to close
  // (-1: it had none); refused (kReading) while a read that began on it has
  // not concluded, leaving it open.
  std::expected<int, FileRefusal> Take(FileId file);
  // Any thread: the descriptor now (-1: closed or unknown), and the reads
  // that began on it and have not concluded.
  int Descriptor(FileId file) const;
  std::uint32_t Reads(FileId file) const;
  std::size_t open() const;  // files with a descriptor

  // The scheduler's: a read about to start on `file` takes its descriptor
  // (-1 when closed or unknown) and, when there is one, counts until
  // EndRead, which the scheduler calls once the read's operation has
  // concluded (its completion proves no further access).
  int BeginRead(FileId file);
  void EndRead(FileId file);

 private:
  struct Entry {
    int fd = -1;
    std::uint32_t reads = 0;
  };
  const Entry* Find(FileId file) const;
  Entry* Find(FileId file);
  mutable std::mutex mutex_;
  std::vector<Entry> files_;  // by index; a FileId's generation is always 1
};

}  // namespace llmp::scheduler

#endif  // LLMP_SCHEDULER_FILES_H_
