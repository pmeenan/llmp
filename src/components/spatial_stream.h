// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_COMPONENTS_SPATIAL_STREAM_H_
#define LLMP_COMPONENTS_SPATIAL_STREAM_H_

#include "components/spatial.h"

namespace llmp::components {
// Shared adapters for definitions that reuse their stream arithmetic in a WHCN graph.
// Definitions choose layouts; the compiler never infers them from a component name.
enum class SpatialStreamLayout { kWhcn, kChannelRows, kChannelFirst };
std::expected<SpatialValue, std::string> EmitSpatialStream(SpatialFragment&,
                                                           SpatialStreamLayout input,
                                                           SpatialStreamLayout output,
                                                           bool auxiliary_channel_rows = false);
template <SpatialStreamLayout Input = SpatialStreamLayout::kWhcn,
          SpatialStreamLayout Output = Input, bool AuxiliaryChannelRows = false>
std::expected<SpatialValue, std::string> EmitSpatialStream(SpatialFragment& fragment) {
  return EmitSpatialStream(fragment, Input, Output, AuxiliaryChannelRows);
}

SpatialShape CarrySpatialShape(const ComponentDef&, SpatialShape);
std::vector<double> ToChannelRows(std::span<const double>, SpatialShape);
std::vector<double> FromChannelRows(std::span<const double>, SpatialShape);
// Validate extents, then optionally bind role-ordered reference weights.
std::expected<SpatialShape, std::string> ValidateSpatialReference(SpatialRefIO&);
std::expected<RefIO, std::string> PrepareSpatialReference(SpatialRefIO&);
std::expected<void, std::string> RunSpatialStreamReference(SpatialRefIO&, bool channel_rows,
                                                           bool auxiliary_channel_rows);
template <bool ChannelRows = true, bool AuxiliaryChannelRows = false>
std::expected<void, std::string> RunSpatialStreamReference(SpatialRefIO& io) {
  return RunSpatialStreamReference(io, ChannelRows, AuxiliaryChannelRows);
}
}  // namespace llmp::components
#endif  // LLMP_COMPONENTS_SPATIAL_STREAM_H_
