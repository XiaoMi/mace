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


#include "rpc/skel/base_func.h"

#include <HAP_perf.h>

namespace rpc {
namespace skel {

namespace {
// for FillRandomValue
const int32_t kRandM = 1 << 20;
const int32_t kRandA = 9;
const int32_t kRandB = 7;
}

void FillRandomValue(void *buffer, const int32_t buffer_size) {
  uint8_t *mem = static_cast<uint8_t * > (buffer);
  mem[0] = HAP_perf_get_time_us() % 256;
  for (int32_t i = 1; i < buffer_size; ++i) {
    mem[i] = (kRandA * mem[i - 1] + kRandB) % kRandM;
  }
}

}  // namespace skel
}  // namespace rpc

