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

#include "mace/ops/arm/fp32/conv_general.h"

#include <memory>

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

MaceStatus Conv2dGeneral::Compute(const OpContext *context,
                                  const Tensor *input,
                                  const Tensor *filter,
                                  Tensor *output) {
  std::unique_ptr<const Tensor> padded_input;
  std::unique_ptr<Tensor> padded_output;

  ResizeOutAndPadInOut(context,
                       input,
                       filter,
                       output,
                       1,
                       4,
                       &padded_input,
                       &padded_output);

  const Tensor *in_tensor = input;
  if (padded_input != nullptr) {
    in_tensor = padded_input.get();
  }
  Tensor *out_tensor = output;
  if (padded_output != nullptr) {
    out_tensor = padded_output.get();
  }
  out_tensor->Clear();

  Tensor::MappingGuard in_guard(input);
  Tensor::MappingGuard filter_guard(filter);
  Tensor::MappingGuard out_guard(output);
  auto filter_data = filter->data<float>();
  auto input_data = in_tensor->data<float>();
  auto output_data = out_tensor->mutable_data<float>();

  auto &in_shape = in_tensor->shape();
  auto &out_shape = out_tensor->shape();
  auto &filter_shape = filter->shape();

  const index_t batch = in_shape[0];
  const index_t in_channels = in_shape[1];
  const index_t in_height = in_shape[2];
  const index_t in_width = in_shape[3];
  const index_t out_channels = out_shape[1];
  const index_t out_height = out_shape[2];
  const index_t out_width = out_shape[3];
  const index_t filter_height = filter_shape[2];
  const index_t filter_width = filter_shape[3];

  const index_t in_image_size = in_height * in_width;
  const index_t out_image_size = out_height * out_width;
  const index_t in_batch_size = in_channels * in_image_size;
  const index_t out_batch_size = out_channels * out_image_size;
  const index_t filter_size = filter_height * filter_width;

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t m = start1; m < end1; m += step1) {
        const int stride_h = strides_[0];
        const int stride_w = strides_[1];
        const int dilation_h = dilations_[0];
        const int dilation_w = dilations_[1];
        if (m + 3 < out_channels) {
          float *out_ptr0_base =
              output_data + b * out_batch_size + m * out_image_size;
          float *out_ptr1_base = out_ptr0_base + out_image_size;
          float *out_ptr2_base = out_ptr1_base + out_image_size;
          float *out_ptr3_base = out_ptr2_base + out_image_size;
          for (index_t c = 0; c < in_channels; ++c) {
            const float *in_ptr_base =
                input_data + b * in_batch_size + c * in_image_size;
            const float *filter_ptr0 =
                filter_data + m * in_channels * filter_size + c * filter_size;
            const float *filter_ptr1 = filter_ptr0 + in_channels * filter_size;
            const float *filter_ptr2 = filter_ptr1 + in_channels * filter_size;
            const float *filter_ptr3 = filter_ptr2 + in_channels * filter_size;
            for (index_t h = 0; h < out_height; ++h) {
              for (index_t w = 0; w + 3 < out_width; w += 4) {
                // input offset
                index_t ih = h * stride_h;
                index_t iw = w * stride_w;
                index_t in_offset = ih * in_width + iw;
                // output (4 outch x 1 height x 4 width): vo_outch_height
                float vo0[4], vo1[4], vo2[4], vo3[4];
                // load output
                index_t out_offset = h * out_width + w;
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

                  in_offset += dilation_h * in_width;
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
          for (index_t mm = m; mm < out_channels; ++mm) {
            float *out_ptr0_base =
                output_data + b * out_batch_size + mm * out_image_size;
            for (index_t c = 0; c < in_channels; ++c) {
              const float *in_ptr_base =
                  input_data + b * in_batch_size + c * in_image_size;
              const float *filter_ptr0 =
                  filter_data + mm * in_channels * filter_size
                      + c * filter_size;

              for (index_t h = 0; h < out_height; ++h) {
                for (index_t w = 0; w + 3 < out_width; w += 4) {
                  // input offset
                  index_t ih = h * stride_h;
                  index_t iw = w * stride_w;
                  index_t in_offset = ih * in_width + iw;
                  // output (1 outch x 1 height x 4 width): vo_outch_height
                  float vo0[4];
                  // load output
                  index_t out_offset = h * out_width + w;
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

                    in_offset += dilation_h * in_width;
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
  }, 0, batch, 1, 0, out_channels, 4);

  UnPadOutput(*out_tensor, output);
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace
