// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A diagnostic report, such as `llmp doctor` prints: titled sections of
// facts, the problems that make the check fail, and warnings that do not.
// Each probe adds its own sections and judges its own facts; the program
// that runs the probes formats the result.

#ifndef LLMP_BASE_REPORT_H_
#define LLMP_BASE_REPORT_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace llmp::base {

struct ReportLine {
  std::string key;
  std::string value;
};

struct ReportSection {
  std::string title;
  std::vector<ReportLine> lines;

  void Add(std::string key, std::string value) {
    lines.push_back({.key = std::move(key), .value = std::move(value)});
  }
};

struct Report {
  std::vector<ReportSection> sections;
  // What stops this host from running this build as designed.
  std::vector<std::string> problems;
  // What deserves attention but does not stop it.
  std::vector<std::string> warnings;

  // The new section; the reference lasts until the next AddSection().
  ReportSection& AddSection(std::string title) {
    sections.push_back({.title = std::move(title), .lines = {}});
    return sections.back();
  }
};

// text safe to print as one line of a report or log: control characters
// (C0, DEL and C1), line and paragraph separators, the bidirectional and
// other invisible formatting characters, fillers, variation selectors and
// tag characters that could make text read differently than it is, and bytes
// that are not UTF-8 are escaped (\xNN, \uNNNN); other text, UTF-8
// included, passes through.
std::string Printable(std::string_view text);

// A byte count for people: whole binary units where exact ("2 MiB"),
// otherwise one decimal place ("121.7 GiB"), and plain bytes below 1 KiB.
std::string FormatBytes(std::uint64_t bytes);

}  // namespace llmp::base

#endif  // LLMP_BASE_REPORT_H_
