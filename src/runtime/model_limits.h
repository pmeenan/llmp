// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What serving chooses from a model's binding. (A model's context ceiling
// is its settings', model_settings.h RunnerContextCeiling and the trained
// context, checked before the node opens.)

#ifndef LLMP_RUNTIME_MODEL_LIMITS_H_
#define LLMP_RUNTIME_MODEL_LIMITS_H_

namespace llmp::model {
struct Dsv4Binding;
}

namespace llmp::runtime {

// Production frontier selection is measured for Flash's Q4_K head and
// F32 hyperconnection product. Other bound formats keep every head row.
bool Dsv4FrontierHeadForServing(const model::Dsv4Binding& binding);

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_MODEL_LIMITS_H_
