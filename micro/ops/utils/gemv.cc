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

#include "micro/ops/utils/gemv.h"

#include "micro/base/logging.h"

namespace micro {
namespace ops {

MaceStatus Gemv<mifloat>::Compute(const mifloat *lhs_data,
                                  const mifloat *rhs_data,
                                  const mifloat *bias_data,
                                  const int32_t batch,
                                  const int32_t lhs_height,
                                  const int32_t lhs_width,
                                  const bool lhs_batched,
                                  const bool rhs_batched,
                                  mifloat *output_data) {
  if (lhs_height == 1) {
    for (int32_t b = 0; b < batch; ++b) {
      const int32_t lhs_b_base = static_cast<int32_t>(lhs_batched) * b;
      const int32_t rhs_b_base =
          static_cast<int32_t>(rhs_batched) * b * lhs_width;
      float sum = bias_data != NULL ? bias_data[0] : 0.0f;
      const int32_t lhs_h_base = lhs_b_base * lhs_width;
      for (int32_t w = 0; w < lhs_width; ++w) {
        sum += lhs_data[lhs_h_base + w] * rhs_data[rhs_b_base + w];
      }  // w
      output_data[lhs_b_base] = sum;
    }   // b
  } else if (lhs_height == 2) {
    for (int32_t b = 0; b < batch; ++b) {
      const int32_t lhs_b_base =
          static_cast<int32_t>(lhs_batched) * b * 2;
      const int32_t rhs_b_base =
          static_cast<int32_t>(rhs_batched) * b * lhs_width;

      float sum0 = bias_data != NULL ? bias_data[0] : 0.0f;
      float sum1 = bias_data != NULL ? bias_data[1] : 0.0f;
      const int32_t lhs_h_base0 = lhs_b_base * lhs_width;
      const int32_t lhs_h_base1 = lhs_h_base0 + lhs_width;
      for (int32_t w = 0; w < lhs_width; ++w) {
        float rhs_data_value = rhs_data[rhs_b_base + w];
        sum0 += lhs_data[lhs_h_base0 + w] * rhs_data_value;
        sum1 += lhs_data[lhs_h_base1 + w] * rhs_data_value;
      }  // w
      output_data[lhs_b_base] = sum0;
      output_data[lhs_b_base + 1] = sum1;
    }   // b
  } else if (lhs_height == 3) {
    for (int32_t b = 0; b < batch; ++b) {
      const int32_t lhs_b_base =
          static_cast<int32_t>(lhs_batched) * b * 2;
      const int32_t rhs_b_base =
          static_cast<int32_t>(rhs_batched) * b * lhs_width;

      float sum0 = bias_data != NULL ? bias_data[0] : 0.0f;
      float sum1 = bias_data != NULL ? bias_data[1] : 0.0f;
      float sum2 = bias_data != NULL ? bias_data[2] : 0.0f;
      const int32_t lhs_h_base0 = lhs_b_base * lhs_width;
      const int32_t lhs_h_base1 = lhs_h_base0 + lhs_width;
      const int32_t lhs_h_base2 = lhs_h_base1 + lhs_width;
      for (int32_t w = 0; w < lhs_width; ++w) {
        float rhs_data_value = rhs_data[rhs_b_base + w];
        sum0 += lhs_data[lhs_h_base0 + w] * rhs_data_value;
        sum1 += lhs_data[lhs_h_base1 + w] * rhs_data_value;
        sum2 += lhs_data[lhs_h_base2 + w] * rhs_data_value;
      }  // w
      output_data[lhs_b_base] = sum0;
      output_data[lhs_b_base + 1] = sum1;
      output_data[lhs_b_base + 2] = sum2;
    }   // b
  } else {  // lhs_height >= 4
    int32_t lhs_height_end = lhs_height - 4;
    for (int32_t b = 0; b < batch; ++b) {
      const int32_t lhs_b_base =
          static_cast<int32_t>(lhs_batched) * b * lhs_height;
      const int32_t rhs_b_base =
          static_cast<int32_t>(rhs_batched) * b * lhs_width;
      for (int32_t h = 0; h < lhs_height; h += 4) {
        if (h > lhs_height_end) {
          h = lhs_height_end;
        }
        float sum0 = 0;
        float sum1 = 0;
        float sum2 = 0;
        float sum3 = 0;
        if (bias_data != NULL) {
          sum0 = bias_data[0];
          sum1 = bias_data[1];
          sum2 = bias_data[2];
          sum3 = bias_data[3];
        }
        const int32_t lhs_h_base0 = (lhs_b_base + h) * lhs_width;
        const int32_t lhs_h_base1 = lhs_h_base0 + lhs_width;
        const int32_t lhs_h_base2 = lhs_h_base1 + lhs_width;
        const int32_t lhs_h_base3 = lhs_h_base2 + lhs_width;
        for (int32_t w = 0; w < lhs_width; ++w) {
          float rhs_data_value = rhs_data[rhs_b_base + w];

          sum0 += lhs_data[lhs_h_base0 + w] * rhs_data_value;
          sum1 += lhs_data[lhs_h_base1 + w] * rhs_data_value;
          sum2 += lhs_data[lhs_h_base2 + w] * rhs_data_value;
          sum3 += lhs_data[lhs_h_base3 + w] * rhs_data_value;
        }  // w

        output_data[lhs_b_base + h] = sum0;
        output_data[lhs_b_base + h + 1] = sum1;
        output_data[lhs_b_base + h + 2] = sum2;
        output_data[lhs_b_base + h + 3] = sum3;
      }  // h
    }   // b
  }

  return MACE_SUCCESS;
}

}  // namespace ops
}  // namespace micro
