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

#include <memory>

#include "mace/ops/arm/base/conv_2d_general.h"
#include "mace/ops/delegator/conv_2d.h"

namespace mace {
namespace ops {
namespace arm {

template<>
MaceStatus Conv2dGeneral<float>::DoCompute(
    const ConvComputeParam &p, const float *filter_data,
    const float *input_data, float *output_data,
    const std::vector<index_t> &filter_shape) {
  const index_t filter_height = filter_shape[2];
  const index_t filter_width = filter_shape[3];
  const index_t filter_size = filter_height * filter_width;

  p.thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t m = start1; m < end1; m += step1) {
        const int stride_h = strides_[0];
        const int stride_w = strides_[1];
        const int dilation_h = dilations_[0];
        const int dilation_w = dilations_[1];
        if (m + 3 < p.out_channels) {
          float *out_ptr0_base =
              output_data + b * p.out_batch_size + m * p.out_image_size;
          float *out_ptr1_base = out_ptr0_base + p.out_image_size;
          float *out_ptr2_base = out_ptr1_base + p.out_image_size;
          float *out_ptr3_base = out_ptr2_base + p.out_image_size;
          for (index_t c = 0; c < p.in_channels; ++c) {
            const float *in_ptr_base =
                input_data + b * p.in_batch_size + c * p.in_image_size;
            const float *filter_ptr0 =
                filter_data + m * p.in_channels * filter_size + c * filter_size;
            const float *filter_ptr1 =
                filter_ptr0 + p.in_channels * filter_size;
            const float *filter_ptr2 =
                filter_ptr1 + p.in_channels * filter_size;
            const float *filter_ptr3 =
                filter_ptr2 + p.in_channels * filter_size;
            for (index_t h = 0; h < p.out_height; ++h) {
              for (index_t w = 0; w + 3 < p.out_width; w += 4) {
                // input offset
                index_t ih = h * stride_h;
                index_t iw = w * stride_w;
                index_t in_offset = ih * p.in_width + iw;
                // output (4 outch x 1 height x 4 width): vo_outch_height
                float vo0[4], vo1[4], vo2[4], vo3[4];
                // load output
                index_t out_offset = h * p.out_width + w;
                for (index_t ow = 0; ow < 4; ++ow) {
                  vo0[ow] = out_ptr0_base[out_offset + ow];
                  vo1[ow] = out_ptr1_base[out_offset + ow];
                  vo2[ow] = out_ptr2_base[out_offset + ow];
                  vo3[ow] = out_ptr3_base[out_offset + ow];
                }
                // calc by row
                for (index_t kh = 0; kh < filter_height; ++kh) {
                  for (index_t kw = 0; kw < filter_width; ++kw) {
                    // outch 0
                    vo0[0] += in_ptr_base[in_offset
                        + kw * dilation_w] * filter_ptr0[kw];
                    vo0[1] += in_ptr_base[in_offset + stride_w
                        + kw * dilation_w] * filter_ptr0[kw];
                    vo0[2] += in_ptr_base[in_offset + 2 * stride_w
                        + kw * dilation_w] * filter_ptr0[kw];
                    vo0[3] += in_ptr_base[in_offset + 3 * stride_w
                        + kw * dilation_w] * filter_ptr0[kw];
                    // outch 1
                    vo1[0] += in_ptr_base[in_offset
                        + kw * dilation_w] * filter_ptr1[kw];
                    vo1[1] += in_ptr_base[in_offset + stride_w
                        + kw * dilation_w] * filter_ptr1[kw];
                    vo1[2] += in_ptr_base[in_offset + 2 * stride_w
                        + kw * dilation_w] * filter_ptr1[kw];
                    vo1[3] += in_ptr_base[in_offset + 3 * stride_w
                        + kw * dilation_w] * filter_ptr1[kw];
                    // outch 2
                    vo2[0] += in_ptr_base[in_offset
                        + kw * dilation_w] * filter_ptr2[kw];
                    vo2[1] += in_ptr_base[in_offset + stride_w
                        + kw * dilation_w] * filter_ptr2[kw];
                    vo2[2] += in_ptr_base[in_offset + 2 * stride_w
                        + kw * dilation_w] * filter_ptr2[kw];
                    vo2[3] += in_ptr_base[in_offset + 3 * stride_w
                        + kw * dilation_w] * filter_ptr2[kw];
                    // outch 3
                    vo3[0] += in_ptr_base[in_offset
                        + kw * dilation_w] * filter_ptr3[kw];
                    vo3[1] += in_ptr_base[in_offset + stride_w
                        + kw * dilation_w] * filter_ptr3[kw];
                    vo3[2] += in_ptr_base[in_offset + 2 * stride_w
                        + kw * dilation_w] * filter_ptr3[kw];
                    vo3[3] += in_ptr_base[in_offset + 3 * stride_w
                        + kw * dilation_w] * filter_ptr3[kw];
                  }  // kw

                  in_offset += dilation_h * p.in_width;
                  filter_ptr0 += filter_width;
                  filter_ptr1 += filter_width;
                  filter_ptr2 += filter_width;
                  filter_ptr3 += filter_width;
                }  // kh

                for (index_t ow = 0; ow < 4; ++ow) {
                  out_ptr0_base[out_offset + ow] = vo0[ow];
                  out_ptr1_base[out_offset + ow] = vo1[ow];
                  out_ptr2_base[out_offset + ow] = vo2[ow];
                  out_ptr3_base[out_offset + ow] = vo3[ow];
                }

                filter_ptr0 -= filter_size;
                filter_ptr1 -= filter_size;
                filter_ptr2 -= filter_size;
                filter_ptr3 -= filter_size;
              }  // w
            }  // h
          }  // c
        } else {
          for (index_t mm = m; mm < p.out_channels; ++mm) {
            float *out_ptr0_base =
                output_data + b * p.out_batch_size + mm * p.out_image_size;
            for (index_t c = 0; c < p.in_channels; ++c) {
              const float *in_ptr_base =
                  input_data + b * p.in_batch_size + c * p.in_image_size;
              const float *filter_ptr0 =
                  filter_data + mm * p.in_channels * filter_size
                      + c * filter_size;

              for (index_t h = 0; h < p.out_height; ++h) {
                for (index_t w = 0; w + 3 < p.out_width; w += 4) {
                  // input offset
                  index_t ih = h * stride_h;
                  index_t iw = w * stride_w;
                  index_t in_offset = ih * p.in_width + iw;
                  // output (1 outch x 1 height x 4 width): vo_outch_height
                  float vo0[4];
                  // load output
                  index_t out_offset = h * p.out_width + w;
                  for (index_t ow = 0; ow < 4; ++ow) {
                    vo0[ow] = out_ptr0_base[out_offset + ow];
                  }

                  // calc by row
                  for (index_t kh = 0; kh < filter_height; ++kh) {
                    for (index_t kw = 0; kw < filter_width; ++kw) {
                      // outch 0
                      vo0[0] += in_ptr_base[in_offset
                          + kw * dilation_w] * filter_ptr0[kw];
                      vo0[1] += in_ptr_base[in_offset + stride_w
                          + kw * dilation_w] * filter_ptr0[kw];
                      vo0[2] += in_ptr_base[in_offset + 2 * stride_w
                          + kw * dilation_w] * filter_ptr0[kw];
                      vo0[3] += in_ptr_base[in_offset + 3 * stride_w
                          + kw * dilation_w] * filter_ptr0[kw];
                    }  // kw

                    in_offset += dilation_h * p.in_width;
                    filter_ptr0 += filter_width;
                  }  // kh

                  for (index_t ow = 0; ow < 4; ++ow) {
                    out_ptr0_base[out_offset + ow] = vo0[ow];
                  }
                  filter_ptr0 -= filter_size;
                }  // w
              }  // h
            }  // c
          }  // mm
        }  // if
      }  // m
    }  // b
  }, 0, p.batch, 1, 0, p.out_channels, 4);

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
