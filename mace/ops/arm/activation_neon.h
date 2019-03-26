// Copyright 2019 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MACE_OPS_ARM_ACTIVATION_NEON_H_
#define MACE_OPS_ARM_ACTIVATION_NEON_H_

#include "mace/core/types.h"

namespace mace {
namespace ops {

void ReluNeon(const float *input, const index_t size, float *output);

void ReluxNeon(const float *input, const float limit,
               const index_t size, float *output);

void LeakyReluNeon(const float *input, const float alpha,
                   const index_t size, float *output);

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_ACTIVATION_NEON_H_
