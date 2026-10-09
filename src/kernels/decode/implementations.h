// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The decode module's entries in the implementation registry (D-053,
// D-108; execution/registry.h), CUDA builds only: every template over
// every decoder, named
//
//   llmp.decode.<template>.<encoding>.<layout>
//
// with template one of vector, vector_glu, matrix, matrix_fma, grouped
// and rows (products.h), and for a GLU over two decoders
// llmp.decode.vector_glu.<gate encoding>.<gate layout>.<up encoding>.<up layout>.
// Each declares its capability: the IR operation (vector, matrix and
// matrix_fma mul_mat; vector_glu fused.mul_mat_glu; grouped mul_mat_id;
// rows get_rows), operand 0 (and a GLU's 1) the weight in its encoding
// and layout, the input F32 dense and a grouped product's or a gather's
// ids I32 dense; rung kNative for BF16 and F16 weights and kOnChipDecode
// for every other encoding (on-chip decode to the device's F32 or BF16/F16
// matrix path, on devices with or without a native kernel for the
// encoding); no expansion; row-invariant and capturable. vector and
// vector_glu take 1 to 16 rows, matrix and matrix_fma 1 to
// kMaxMatrixRows; matrix and grouped need BF16 mma.sync
// (sm_80 on); K is a multiple of the decoder's segment, blocks and scale
// atoms (DecoderKMultiple). Each accepts only the device architectures the
// build has SASS for (CompiledFor), K below 2^30 and N below 2^31.
//
// Each identity covers llmpalooza's code in the module (a digest of every
// file in src/kernels/decode, written at build time, module_digest.cmake),
// the SDK, target, device architectures, build type and sanitizers, and a
// variant naming the kernel, its tiles and its decoder.

#ifndef LLMP_KERNELS_DECODE_IMPLEMENTATIONS_H_
#define LLMP_KERNELS_DECODE_IMPLEMENTATIONS_H_

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "execution/registry.h"
#include "kernels/decode/launch.h"
#include "kernels/decode/operand.h"

namespace llmp::kernels::decode {

std::vector<execution::Implementation> Implementations();

// The digest of the module's own files, generated at build time.
std::string_view ModuleSourcesDigest();

std::string_view TemplateName(Template t);

// What a declaration names: its template and decoders (`up` is `gate`
// except for a GLU over two decoders).
struct Bound {
  Template tmpl = Template::kVector;
  DecoderId gate = DecoderId::kBf16Dense;
  DecoderId up = DecoderId::kBf16Dense;
  bool operator==(const Bound&) const = default;
};
std::string ImplementationName(const Bound& bound);

// The declaration `implementation` names, refused unless this module
// declares it, identity and all.
std::expected<Bound, std::string> Bind(const execution::Implementation& implementation);

// Whether this build has SASS that runs on a device of architecture
// `arch` (CUDA's major · 10 + minor): an `sm_XY` cubin on X.Y up to the
// next major, an arch-specific `sm_XYa` on X.Y only.
bool CompiledFor(std::uint32_t arch);

}  // namespace llmp::kernels::decode

#endif  // LLMP_KERNELS_DECODE_IMPLEMENTATIONS_H_
