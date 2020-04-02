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


#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

namespace {
// for FillRandomInput
const int32_t kRandM = 1 << 20;
const int32_t kRandA = 9;
const int32_t kRandB = 7;
}

void PrintDims(const int32_t *dims, const uint32_t dim_size) {
  MACE_ASSERT1(dim_size > 0, "invalide dim size");
  if (dim_size == 1) {
    LOG(INFO) << "[ " << dims[0] << " ]";
  } else if (dim_size == 2) {
    LOG(INFO) << "[ " << dims[0] << ", " << dims[1] << " ]";
  } else if (dim_size == 3) {
    LOG(INFO) << "[ " << dims[0] << ", " << dims[1] << ", " << dims[2] << " ]";
  } else if (dim_size == 4) {
    LOG(INFO) << "[ " << dims[0] << ", " << dims[1]
              << ", " << dims[2] << ", " << dims[3] << " ]";
  } else {
    for (uint32_t i = 0; i < dim_size; ++i) {
      LOG(INFO) << dims[i];
    }
  }
}

void AssertSameDims(const int32_t *x_dims, const uint32_t x_dim_size,
                    const int32_t *y_dims, const uint32_t y_dim_size) {
  if (x_dim_size != y_dim_size) {
    LOG(FATAL) << "invalide dim size. x_dim_size = " << x_dim_size
               << ", y_dim_size = " << y_dim_size;
  }
  for (uint32_t i = 0; i < x_dim_size; ++i) {
    if (x_dims[i] != y_dims[i]) {
      PrintDims(x_dims, x_dim_size);
      PrintDims(y_dims, y_dim_size);
      LOG(FATAL) << "AssertSameDims failed.";
    }
  }
}

void FillRandomInput(void *input, const int32_t shape_size) {
  uint8_t *mem = static_cast<uint8_t * > (input);
  mem[0] = port::api::NowMicros() % 256;
  for (int32_t i = 1; i < shape_size; ++i) {
    mem[i] = (kRandA * mem[i - 1] + kRandB) % kRandM;
  }
}

}  // namespace test
}  // namespace ops
}  // namespace micro


