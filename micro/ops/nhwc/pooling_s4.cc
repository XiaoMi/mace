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

#include "micro/ops/nhwc/pooling_s4.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/framework/scratch_buffer.h"

namespace micro {
namespace ops {
void PoolingS4Op::MaxPooling(const mifloat *input,
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
  const int32_t filter_size = filter_hw[0] * filter_hw[1];
  const int32_t filter_size_end = filter_size - 4;

  float *max = ScratchBuffer(engine_config_).GetBuffer<float>(in_channels);
  for (int32_t b = 0; b < batch; ++b) {
    int32_t batch_base = b * out_height;
    int32_t in_b_base = b * in_height;
    for (int32_t h = 0; h < out_height; ++h) {
      int32_t height_base = (batch_base + h) * out_width;
      int32_t inh_base = h * stride_hw[0] - pad_hw[0];
      for (int32_t w = 0; w < out_width; ++w) {
        int32_t width_base = (height_base + w) * out_channels;
        int32_t inw_base = w * stride_hw[1] - pad_hw[1];
        for (int32_t c = 0; c < in_channels; ++c) {
          max[c] = base::lowest();
        }

        for (int32_t s = 0; s < filter_size; s += 4) {
          if (s > filter_size_end) {
            s = filter_size_end;
          }
          const int32_t s1 = s + 1;
          const int32_t s2 = s1 + 1;
          const int32_t s3 = s2 + 1;
          int32_t fh0 = s / filter_hw[1];
          int32_t fh1 = s1 / filter_hw[1];
          int32_t fh2 = s2 / filter_hw[1];
          int32_t fh3 = s3 / filter_hw[1];
          int32_t fw0 = s % filter_hw[1];
          int32_t fw1 = s1 % filter_hw[1];
          int32_t fw2 = s2 % filter_hw[1];
          int32_t fw3 = s3 % filter_hw[1];
          int32_t inh0 = inh_base + dilation_hw[0] * fh0;
          int32_t inh1 = inh_base + dilation_hw[0] * fh1;
          int32_t inh2 = inh_base + dilation_hw[0] * fh2;
          int32_t inh3 = inh_base + dilation_hw[0] * fh3;
          int32_t inw0 = inw_base + dilation_hw[1] * fw0;
          int32_t inw1 = inw_base + dilation_hw[1] * fw1;
          int32_t inw2 = inw_base + dilation_hw[1] * fw2;
          int32_t inw3 = inw_base + dilation_hw[1] * fw3;
          bool valid[4] = {
              inh0 >= 0 && inh0 < in_height && inw0 >= 0 && inw0 < in_width,
              inh1 >= 0 && inh1 < in_height && inw1 >= 0 && inw1 < in_width,
              inh2 >= 0 && inh2 < in_height && inw2 >= 0 && inw2 < in_width,
              inh3 >= 0 && inh3 < in_height && inw3 >= 0 && inw3 < in_width
          };
          int32_t in_w_base0 =
              ((in_b_base + inh0) * in_width + inw0) * in_channels;
          int32_t in_w_base1 =
              ((in_b_base + inh1) * in_width + inw1) * in_channels;
          int32_t in_w_base2 =
              ((in_b_base + inh2) * in_width + inw2) * in_channels;
          int32_t in_w_base3 =
              ((in_b_base + inh3) * in_width + inw3) * in_channels;
          for (int32_t c = 0; c < out_channels; ++c) {
            if (valid[0]) {
              const int32_t input_offset0 = in_w_base0 + c;
              float input_value = input[input_offset0];
              if (input_value > max[c]) {
                max[c] = input_value;
              }
            }
            if (valid[1]) {
              const int32_t input_offset1 = in_w_base1 + c;
              float input_value = input[input_offset1];
              if (input_value > max[c]) {
                max[c] = input_value;
              }
            }
            if (valid[2]) {
              const int32_t input_offset2 = in_w_base2 + c;
              float input_value = input[input_offset2];
              if (input_value > max[c]) {
                max[c] = input_value;
              }
            }
            if (valid[3]) {
              const int32_t input_offset3 = in_w_base3 + c;
              float input_value = input[input_offset3];
              if (input_value > max[c]) {
                max[c] = input_value;
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

void PoolingS4Op::AvgPooling(const mifloat *input,
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
  const int32_t filter_size = filter_hw[0] * filter_hw[1];
  const int32_t filter_size_end = filter_size - 4;

  ScratchBuffer scratch_buffer(engine_config_);
  float *total = scratch_buffer.GetBuffer<float>(in_channels);
  uint32_t *block_size = scratch_buffer.GetBuffer<uint32_t>(in_channels);
  for (int32_t b = 0; b < batch; ++b) {
    int32_t batch_base = b * out_height;
    int32_t in_b_base = b * in_height;
    for (int32_t h = 0; h < out_height; ++h) {
      int32_t height_base = (batch_base + h) * out_width;
      int32_t inh_base = h * stride_hw[0] - pad_hw[0];
      for (int32_t w = 0; w < out_width; ++w) {
        int32_t width_base = (height_base + w) * out_channels;
        int32_t inw_base = w * stride_hw[1] - pad_hw[1];
        for (int32_t c = 0; c < in_channels; ++c) {
          total[c] = 0;
          block_size[c] = 0;
        }

        for (int32_t s = 0; s < filter_size; s += 4) {
          if (s > filter_size_end) {
            s = filter_size_end;
          }
          const int32_t s1 = s + 1;
          const int32_t s2 = s1 + 1;
          const int32_t s3 = s2 + 1;
          int32_t fh0 = s / filter_hw[1];
          int32_t fh1 = s1 / filter_hw[1];
          int32_t fh2 = s2 / filter_hw[1];
          int32_t fh3 = s3 / filter_hw[1];
          int32_t fw0 = s % filter_hw[1];
          int32_t fw1 = s1 % filter_hw[1];
          int32_t fw2 = s2 % filter_hw[1];
          int32_t fw3 = s3 % filter_hw[1];
          int32_t inh0 = inh_base + dilation_hw[0] * fh0;
          int32_t inh1 = inh_base + dilation_hw[0] * fh1;
          int32_t inh2 = inh_base + dilation_hw[0] * fh2;
          int32_t inh3 = inh_base + dilation_hw[0] * fh3;
          int32_t inw0 = inw_base + dilation_hw[1] * fw0;
          int32_t inw1 = inw_base + dilation_hw[1] * fw1;
          int32_t inw2 = inw_base + dilation_hw[1] * fw2;
          int32_t inw3 = inw_base + dilation_hw[1] * fw3;
          bool valid[4] = {
              inh0 >= 0 && inh0 < in_height && inw0 >= 0 && inw0 < in_width,
              inh1 >= 0 && inh1 < in_height && inw1 >= 0 && inw1 < in_width,
              inh2 >= 0 && inh2 < in_height && inw2 >= 0 && inw2 < in_width,
              inh3 >= 0 && inh3 < in_height && inw3 >= 0 && inw3 < in_width
          };
          int32_t in_w_base0 =
              ((in_b_base + inh0) * in_width + inw0) * in_channels;
          int32_t in_w_base1 =
              ((in_b_base + inh1) * in_width + inw1) * in_channels;
          int32_t in_w_base2 =
              ((in_b_base + inh2) * in_width + inw2) * in_channels;
          int32_t in_w_base3 =
              ((in_b_base + inh3) * in_width + inw3) * in_channels;
          int32_t block_num = valid[0] + valid[1] + valid[2] + valid[3];
          for (int32_t c = 0; c < out_channels; ++c) {
            float total_c = 0;
            if (valid[0]) {
              total_c += input[in_w_base0 + c];
            }
            if (valid[1]) {
              total_c += input[in_w_base1 + c];
            }
            if (valid[2]) {
              total_c += input[in_w_base2 + c];
            }
            if (valid[3]) {
              total_c += input[in_w_base3 + c];
            }
            total[c] += total_c;
            block_size[c] += block_num;
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
