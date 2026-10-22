// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <istream>

#include "base/work_pulse.h"
#include "runtime/serving.h"

namespace llmp::runtime {

Status ReadContextText(std::istream& input, std::span<char> storage) {
  if (storage.empty()) return std::unexpected("the context text is empty");
  for (std::size_t at = 0; at < storage.size();) {
    if (!base::Pulse()) return std::unexpected("reading the context text was asked to stop");
    const auto count = std::min(storage.size() - at, std::size_t{65536});
    input.read(storage.data() + at, static_cast<std::streamsize>(count));
    if (!input || input.gcount() != static_cast<std::streamsize>(count))
      return std::unexpected("the context text changed length or could not be read");
    at += count;
  }
  if (!base::Pulse()) return std::unexpected("reading the context text was asked to stop");
  char extra = 0;
  if (input.get(extra)) return std::unexpected("the context text grew while it was read");
  if (!input.eof() || input.bad()) return std::unexpected("the context text could not be read");
  return {};
}

bool CompleteContinuationRows(std::span<const float> last, const Generation& reference,
                              const Generation& continued, std::size_t steps, std::size_t width) {
  const auto finite = [width](std::span<const float> row) {
    return row.size() == width &&
           std::ranges::all_of(row, [](float v) { return std::isfinite(v); });
  };
  const auto complete = [&](const Generation& generation) {
    return generation.logits.size() == steps &&
           std::ranges::all_of(generation.logits,
                               [&](const std::vector<float>& row) { return finite(row); });
  };
  return width != 0 && finite(last) && complete(reference) && complete(continued);
}

}  // namespace llmp::runtime
