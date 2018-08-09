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

#include "mace/kernels/arm/conv_2d_neon.h"
#include "mace/kernels/gemm.h"

namespace mace {
namespace kernels {

void Conv2dNeonK1x1S1(const float *input,
                      const float *filter,
                      const index_t batch,
                      const index_t height,
                      const index_t width,
                      const index_t in_channels,
                      const index_t out_channels,
                      float *output) {
  for (index_t b = 0; b < batch; ++b) {
    Gemm(filter, input + b * in_channels * height * width, 1, out_channels,
         in_channels, height * width,
         output + b * out_channels * height * width);
  }
}

}  // namespace kernels
}  // namespace mace
