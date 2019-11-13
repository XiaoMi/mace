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

#include "micro/ops/nhwc/pooling_ref.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/framework/scratch_buffer.h"

namespace micro {
namespace ops {

void PoolingRefOp::MaxPooling(const mifloat *input,
                              const int32_t *filter_hw,
                              const int32_t *stride_hw,
                              const int32_t *dilation_hw,
                              const int32_t *pad_hw) {
  const int32_t batch = output_dims_[0];
  const int32_t out_channels = output_dims_[3];
  const int32_t out_height = output_dims_[1];
  const int32_t out_width = output_dims_[2];
  const int32_t in_channels = input_dims_[3];
  const int32_t in_height = input_dims_[1];
  const int32_t in_width = input_dims_[2];

  float *max = ScratchBuffer(engine_config_).GetBuffer<float>(in_channels);
  for (int32_t b = 0; b < batch; ++b) {
    int32_t batch_base = b * out_height;
    int32_t in_b_base = b * in_height;
    for (int32_t h = 0; h < out_height; ++h) {
      int32_t height_base = (batch_base + h) * out_width;
      int32_t inh_addr = h * stride_hw[0] - pad_hw[0];
      for (int32_t w = 0; w < out_width; ++w) {
        int32_t width_base = (height_base + w) * out_channels;
        int32_t inw_addr = w * stride_hw[1] - pad_hw[1];
        for (int32_t c = 0; c < in_channels; ++c) {
          max[c] = base::lowest();
        }
        for (int32_t fh = 0; fh < filter_hw[0]; ++fh) {
          int32_t inh = inh_addr + dilation_hw[0] * fh;
          if (inh < 0 && inh >= in_height) {
            continue;
          }
          int32_t in_h_base = (in_b_base + inh) * in_width;
          for (int32_t fw = 0; fw < filter_hw[1]; ++fw) {
            int32_t inw = inw_addr + dilation_hw[1] * fw;
            int32_t in_w_base = (in_h_base + inw) * in_channels;
            for (int32_t c = 0; c < out_channels; ++c) {
              if (inw >= 0 && inw < in_width) {
                const int32_t input_offset = in_w_base + c;
                float input_value = input[input_offset];
                if (input_value > max[c]) {
                  max[c] = input_value;
                }
              }
            }
          }
        }
        for (int i = 0; i < in_channels; ++i) {
          output_[width_base + i] = max[i];
        }
      }
    }
  }
}

void PoolingRefOp::AvgPooling(const mifloat *input,
                              const int32_t *filter_hw,
                              const int32_t *stride_hw,
                              const int32_t *dilation_hw,
                              const int32_t *pad_hw) {
  const int32_t batch = output_dims_[0];
  const int32_t out_channels = output_dims_[3];
  const int32_t out_height = output_dims_[1];
  const int32_t out_width = output_dims_[2];
  const int32_t in_channels = input_dims_[3];
  const int32_t in_height = input_dims_[1];
  const int32_t in_width = input_dims_[2];

  ScratchBuffer scratch_buffer(engine_config_);
  float *total = scratch_buffer.GetBuffer<float>(in_channels);
  uint32_t *block_size = scratch_buffer.GetBuffer<uint32_t>(in_channels);
  for (int32_t b = 0; b < batch; ++b) {
    int32_t batch_base = b * out_height;
    int32_t in_b_base = b * in_height;
    for (int32_t h = 0; h < out_height; ++h) {
      int32_t height_base = (batch_base + h) * out_width;
      int32_t inh_addr = h * stride_hw[0] - pad_hw[0];
      for (int32_t w = 0; w < out_width; ++w) {
        int32_t width_base = (height_base + w) * out_channels;
        int32_t inw_addr = w * stride_hw[1] - pad_hw[1];
        for (int32_t c = 0; c < out_channels; ++c) {
          total[c] = 0;
          block_size[c] = 0;
        }
        for (int32_t fh = 0; fh < filter_hw[0]; ++fh) {
          int32_t inh = inh_addr + dilation_hw[0] * fh;
          int32_t in_h_base = (in_b_base + inh) * in_width;
          for (int32_t fw = 0; fw < filter_hw[1]; ++fw) {
            int32_t inw = inw_addr + dilation_hw[1] * fw;
            int32_t in_w_base = (in_h_base + inw) * in_channels;
            for (int32_t c = 0; c < out_channels; ++c) {
              if (inh >= 0 && inh < in_height && inw >= 0 && inw < in_width) {
                total[c] += input[in_w_base + c];
                ++block_size[c];
              }
            }
          }
        }
        for (int32_t c = 0; c < out_channels; ++c) {
          output_[width_base + c] = total[c] / block_size[c];
        }
      }
    }
  }
}

}  // namespace ops
}  // namespace micro
