// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The importer's parameter folds (D-108 as amended, D-113): a small
// per-channel value a component would otherwise derive at every step from
// its stored one, because its instance's flag asks (components::
// WeightRole::derived: a zero-centered norm's 1 + w, a logarithmic decay's
// -exp(A)), stored derived instead, in F32, computed on the host in double
// and rounded once (FoldBf16). The growth is negligible (vectors of a
// layer's width at most), the run-time work and its rounding go away.
// Chosen by the role each tensor binds, never by architecture: what the
// importer folds is exactly what the synthesized spec's components would
// derive. Bulk weights never widen: only whole dense BF16 vectors (rank 1,
// at most 1 MiB in F32) of one copied source qualify.

#ifndef LLMP_IMPORT_PARAMETER_FOLDS_H_
#define LLMP_IMPORT_PARAMETER_FOLDS_H_

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "import/error.h"
#include "import/indexed_parts.h"
#include "spec/spec.h"

namespace llmp::import {

// A fold an importer applied (spec.h's catalog) and the outputs it folded,
// in output order.
struct AppliedParameterFold {
  std::string fold{};
  std::vector<std::uint32_t> outputs{};
};

// Folds into `outputs` every tensor of `stage` (a spec's stage synthesized
// from these outputs unfolded, a dense output's data part named as it)
// that a component's role derives because its instance's flag is set:
// each such output becomes F32 with its fold (under its head permutation,
// if it has one). A fold covers all the tensors it governs or none (the
// stage then carries it, which clears every governing flag at once): when
// one of them is not a whole dense BF16 vector within the size bound above
// of one copied source, lives in another stage's artifact, or is bound to a
// role that does not derive it too, that fold is not applied. The caller
// records the applied folds in its source's folds (which the mapping's
// ParamRule::unless_fold reads) and in the artifact. Refused for a derived
// role whose fold this importer cannot compute.
std::expected<std::vector<AppliedParameterFold>, Error> FoldDerivedParameters(
    const spec::Stage& stage, std::vector<TensorAssembly>& outputs);

}  // namespace llmp::import

#endif  // LLMP_IMPORT_PARAMETER_FOLDS_H_
