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

#include "micro/ops/nhwc/conv_2d_ref.h"

#include "micro/base/logging.h"

namespace micro {
namespace ops {

MaceStatus Conv2dRefOp::Compute(int32_t (&output_dims)[4]) {
  const int32_t batch = output_dims[0];
  const int32_t height = output_dims[1];
  const int32_t width = output_dims[2];
  const int32_t channel = output_dims[3];
  const int32_t k_height = filter_dims_[1];
  const int32_t k_width = filter_dims_[2];
  const int32_t k_channel = filter_dims_[3];
  MACE_ASSERT(filter_dims_[0] == channel && input_dims_[3] == k_channel);
  const int32_t in_height = input_dims_[1];
  const int32_t in_width = input_dims_[2];
  const int32_t in_channel = input_dims_[3];

  const int32_t pad_top = padding_sizes_[0] >> 1;
  const int32_t pad_left = padding_sizes_[1] >> 1;

  for (int32_t b = 0; b < batch; ++b) {
    const int32_t batch_base = b * height;
    for (int32_t h = 0; h < height; ++h) {
      const int32_t height_base = (batch_base + h) * width;
      const int32_t in_h = h * strides_[0] - pad_top;
      for (int32_t w = 0; w < width; ++w) {
        const int32_t width_base = (height_base + w) * channel;
        const int32_t in_w = w * strides_[1] - pad_left;
        for (int32_t kb = 0; kb < channel; ++kb) {
          const int32_t o_idx = width_base + kb;
          const int32_t k_batch_base = kb * k_height;
          float output = 0;
          for (int32_t kh = 0; kh < k_height; ++kh) {
            const int32_t in_h_idx = in_h + kh * dilations_[0];
            if (in_h_idx < 0 || in_h_idx >= in_height) {
              continue;
            }
            const int32_t k_height_base = (k_batch_base + kh) * k_width;
            const int32_t in_h_base = in_h_idx * in_width;
            for (int32_t kw = 0; kw < k_width; ++kw) {
              const int32_t in_w_idx = in_w + kw * dilations_[1];
              if (in_w_idx < 0 || in_w_idx >= in_width) {
                continue;
              }
              const int32_t k_width_base = (k_height_base + kw) * k_channel;
              const int32_t in_w_base = (in_h_base + in_w_idx) * in_channel;
              for (int32_t kc = 0; kc < k_channel; ++kc) {
                output += input_[in_w_base + kc] * filter_[k_width_base + kc];
              }  // filter channel
            }  // filter width
          }  // filter height
          output_[o_idx] = output;
        }  // filter batch, output channel
      }  // output width
    }  // output height
  }  // output batch

  return MACE_SUCCESS;
}

}  // namespace ops
}  // namespace micro
