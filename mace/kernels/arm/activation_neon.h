// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#ifndef MACE_KERNELS_ARM_ACTIVATION_NEON_H_
#define MACE_KERNELS_ARM_ACTIVATION_NEON_H_

#include "mace/core/types.h"

namespace mace {
namespace kernels {

void ReluNeon(const float *input, const index_t size, float *output);

void ReluxNeon(const float *input, const float limit,
               const index_t size, float *output);

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_ARM_ACTIVATION_NEON_H_
