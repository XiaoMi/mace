// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include "micro/ops/nhwc/cmsis_nn/utilities.h"

#include <math.h>

void QuantizeMultiplier(double double_multiplier,
                        int32_t *quantized_multiplier,
                        int32_t *shift) {
  if (double_multiplier == 0.) {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }
  const double q = frexp(double_multiplier, reinterpret_cast<int *>(shift));
  int64_t q_fixed = static_cast<int64_t>(round(q * (1ll << 31)));

  if (q_fixed == (1ll << 31)) {
    q_fixed /= 2;
    ++*shift;
  }

  if (*shift < -31) {
    *shift = 0;
    q_fixed = 0;
  }
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}
