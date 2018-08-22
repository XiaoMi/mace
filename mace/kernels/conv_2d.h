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

#ifndef MACE_KERNELS_CONV_2D_H_
#define MACE_KERNELS_CONV_2D_H_

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/activation.h"
#include "mace/kernels/conv_pool_2d_util.h"
#include "mace/kernels/arm/conv_2d_neon.h"
#include "mace/kernels/arm/conv_winograd.h"
#include "mace/kernels/gemmlowp_util.h"
#include "mace/kernels/quantize.h"
#include "mace/utils/utils.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

struct Conv2dFunctorBase {
  Conv2dFunctorBase(const int *strides,
                    const Padding &padding_type,
                    const std::vector<int> &paddings,
                    const int *dilations,
                    const ActivationType activation,
                    const float relux_max_limit)
    : strides_(strides),
      padding_type_(padding_type),
      paddings_(paddings),
      dilations_(dilations),
      activation_(activation),
      relux_max_limit_(relux_max_limit) {}

  const int *strides_;  // [stride_h, stride_w]
  const Padding padding_type_;
  std::vector<int> paddings_;
  const int *dilations_;  // [dilation_h, dilation_w]
  const ActivationType activation_;
  const float relux_max_limit_;
};

template<DeviceType D, typename T>
struct Conv2dFunctor;

template<>
struct Conv2dFunctor<DeviceType::CPU, float> : Conv2dFunctorBase {
  Conv2dFunctor(const int *strides,
                const Padding &padding_type,
                const std::vector<int> &paddings,
                const int *dilations,
                const ActivationType activation,
                const float relux_max_limit,
                const bool is_filter_transformed,
                ScratchBuffer *scratch)
    : Conv2dFunctorBase(strides,
                        padding_type,
                        paddings,
                        dilations,
                        activation,
                        relux_max_limit),
      is_filter_transformed_(is_filter_transformed),
      scratch_(scratch) {}

  void Conv2dGeneral(const float *input,
                     const float *filter,
                     const index_t *in_shape,
                     const index_t *out_shape,
                     const index_t *filter_shape,
                     const int *stride_hw,
                     const int *dilation_hw,
                     float *output) {
    const index_t in_image_size = in_shape[2] * in_shape[3];
    const index_t out_image_size = out_shape[2] * out_shape[3];
    const index_t in_batch_size = filter_shape[1] * in_image_size;
    const index_t out_batch_size = filter_shape[0] * out_image_size;
    const index_t filter_size = filter_shape[2] * filter_shape[3];

#pragma omp parallel for collapse(2)
    for (index_t b = 0; b < in_shape[0]; b++) {
      for (index_t m = 0; m < filter_shape[0]; m += 4) {
        const index_t in_width = in_shape[3];
        const index_t out_height = out_shape[2];
        const index_t out_width = out_shape[3];
        const index_t out_channels = filter_shape[0];
        const index_t in_channels = filter_shape[1];

        const int stride_h = stride_hw[0];
        const int stride_w = stride_hw[1];
        const int dilation_h = dilation_hw[0];
        const int dilation_w = dilation_hw[1];
        if (m + 3 < out_channels) {
          float *out_ptr0_base =
              output + b * out_batch_size + m * out_image_size;
          float *out_ptr1_base = out_ptr0_base + out_image_size;
          float *out_ptr2_base = out_ptr1_base + out_image_size;
          float *out_ptr3_base = out_ptr2_base + out_image_size;
          for (index_t c = 0; c < in_channels; ++c) {
            const float *in_ptr_base =
                input + b * in_batch_size + c * in_image_size;
            const float *filter_ptr0 =
                filter + m * in_channels * filter_size + c * filter_size;
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
                for (index_t kh = 0; kh < filter_shape[2]; ++kh) {
                  for (index_t kw = 0; kw < filter_shape[3]; ++kw) {
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
                  filter_ptr0 += filter_shape[3];
                  filter_ptr1 += filter_shape[3];
                  filter_ptr2 += filter_shape[3];
                  filter_ptr3 += filter_shape[3];
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
                output + b * out_batch_size + mm * out_image_size;
            for (index_t c = 0; c < in_channels; ++c) {
              const float *in_ptr_base =
                  input + b * in_batch_size + c * in_image_size;
              const float *filter_ptr0 =
                  filter + mm * in_channels * filter_size + c * filter_size;

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
                  for (index_t kh = 0; kh < filter_shape[2]; ++kh) {
                    for (index_t kw = 0; kw < filter_shape[3]; ++kw) {
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
                    filter_ptr0 += filter_shape[3];
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
  }

  MaceStatus operator()(const Tensor *input,   // NCHW
                        const Tensor *filter,  // OIHW
                        const Tensor *bias,
                        Tensor *output,        // NCHW
                        StatsFuture *future) {
    MACE_UNUSED(future);
    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

    std::vector<index_t> filter_shape(4);
    if (is_filter_transformed_) {
      // TOC -> OIHW
      filter_shape[0] = filter->dim(1);
      filter_shape[1] = filter->dim(2);
      filter_shape[2] = filter_shape[3] = 3;
    } else {
      filter_shape = filter->shape();
    }

    std::vector<index_t> output_shape(4);
    std::vector<int> paddings(2);
    if (paddings_.empty()) {
      CalcNCHWPaddingAndOutputSize(input->shape().data(),
                                   filter_shape.data(),
                                   dilations_,
                                   strides_,
                                   padding_type_,
                                   output_shape.data(),
                                   paddings.data());
    } else {
      paddings = paddings_;
      CalcNCHWOutputSize(input->shape().data(),
                         filter_shape.data(),
                         paddings_.data(),
                         dilations_,
                         strides_,
                         RoundType::FLOOR,
                         output_shape.data());
    }
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    index_t batch = output->dim(0);
    index_t channels = output->dim(1);
    index_t height = output->dim(2);
    index_t width = output->dim(3);

    index_t input_batch = input->dim(0);
    index_t input_channels = input->dim(1);
    index_t input_height = input->dim(2);
    index_t input_width = input->dim(3);

    index_t filter_h = filter_shape[2];
    index_t filter_w = filter_shape[3];
    MACE_CHECK(filter_shape[0] == channels, filter_shape[0], " != ", channels);
    MACE_CHECK(filter_shape[1] == input_channels, filter_shape[1], " != ",
               input_channels);

    index_t stride_h = strides_[0];
    index_t stride_w = strides_[1];

    index_t dilation_h = dilations_[0];
    index_t dilation_w = dilations_[1];

    MACE_CHECK(batch == input_batch, "Input/Output batch size mismatch");

    index_t padded_input_height = input_height + paddings[0];
    index_t padded_input_width = input_width + paddings[1];
    index_t extra_input_height = padded_input_height;
    index_t extra_input_width = padded_input_width;
    index_t extra_output_height = height;
    index_t extra_output_width = width;

    int pad_top = paddings[0] >> 1;
    int pad_bottom = paddings[0] - pad_top;
    int pad_left = paddings[1] >> 1;
    int pad_right = paddings[1] - pad_left;

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard filter_guard(filter);
    Tensor::MappingGuard bias_guard(bias);
    Tensor::MappingGuard output_guard(output);

    auto filter_data = filter->data<float>();
    auto bias_data = bias == nullptr ? nullptr : bias->data<float>();
    auto output_data = output->mutable_data<float>();

    std::function<void(const float *input, float *output)> conv_func;

    bool
      use_winograd = is_filter_transformed_ || (filter_h == 3 && filter_w == 3
      && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1
      && input_channels >= 8 && channels >= 8);
    bool use_neon_3x3_s1 = filter_h == 3 && filter_w == 3
      && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1;
    bool use_neon_3x3_s2 = filter_h == 3 && filter_w == 3
      && stride_h == 2 && stride_w == 2 && dilation_h == 1 && dilation_w == 1;
    bool use_neon_1x1_s1 = filter_h == 1 && filter_w == 1
      && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1;
    bool use_neon_5x5_s1 = filter_h == 5 && filter_w == 5
        && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1;
    bool use_neon_1x7_s1 = filter_h == 1 && filter_w == 7
        && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1;
    bool use_neon_7x1_s1 = filter_h == 7 && filter_w == 1
        && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1;
    bool use_neon_7x7_s1 = filter_h == 7 && filter_w == 7
        && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1;
    bool use_neon_7x7_s2 = filter_h == 7 && filter_w == 7
        && stride_h == 2 && stride_w == 2 && dilation_h == 1 && dilation_w == 1;
    bool use_neon_7x7_s3 = filter_h == 7 && filter_w == 7
        && stride_h == 3 && stride_w == 3 && dilation_h == 1 && dilation_w == 1;
    bool use_neon_1x15_s1 = filter_h == 1 && filter_w == 15
        && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1;
    bool use_neon_15x1_s1 = filter_h == 15 && filter_w == 1
        && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1;

    std::vector<index_t> transformed_input_shape;
    std::vector<index_t> transformed_output_shape;
    std::vector<index_t> transformed_filter_shape;

    // When size of input feature map is bigger than 16x16,
    // set winograd out tile size to 6 to get higher performance.
    index_t winograd_out_tile_size = 2;
    if (input_height > 16 && input_width > 16) {
      winograd_out_tile_size = 6;
    }

    if (use_winograd) {
      extra_output_height = RoundUp<index_t>(height, winograd_out_tile_size);
      extra_input_height =
        std::max(padded_input_height, extra_output_height + 2);
      extra_output_width = RoundUp<index_t>(width, winograd_out_tile_size);
      extra_input_width = std::max(padded_input_width, extra_output_width + 2);
      if (extra_input_height != padded_input_height) {
        pad_bottom += (extra_input_height - padded_input_height);
      }
      if (extra_input_width != padded_input_width) {
        pad_right += (extra_input_width - padded_input_width);
      }

      index_t tile_height_count = extra_output_height / winograd_out_tile_size;
      index_t tile_width_count = extra_output_width / winograd_out_tile_size;
      index_t tile_count = tile_height_count * tile_width_count;
      index_t in_tile_area =
        (winograd_out_tile_size + 2) * (winograd_out_tile_size + 2);

      transformed_input_shape.insert(transformed_input_shape.end(),
                                     {in_tile_area, batch, input_channels,
                                      tile_count});
      transformed_output_shape.insert(transformed_output_shape.end(),
                                      {in_tile_area, batch, channels,
                                       tile_count});
      transformed_filter_shape.insert(transformed_filter_shape.end(),
                                      {in_tile_area, channels, input_channels});
    } else {
      index_t tile_h, tile_w;
      if (use_neon_1x1_s1) {
        tile_h = 1;
        tile_w = 1;
      } else if (use_neon_3x3_s1) {
        tile_h = 2;
        tile_w = 4;
      } else if (use_neon_7x1_s1 || use_neon_15x1_s1) {
        tile_h = 4;
        tile_w = 1;
      } else {
        tile_h = 1;
        tile_w = 4;
      }
      extra_output_height = RoundUp<index_t>(height, tile_h);
      extra_input_height =
          std::max(padded_input_height, (extra_output_height - 1) * stride_h
              + (filter_h - 1) * dilation_h + 1);
      extra_output_width = RoundUp<index_t>(width, tile_w);
      extra_input_width =
          std::max(padded_input_width, (extra_output_width - 1) * stride_w
              + (filter_w - 1) * dilation_w + 1);
      if (extra_input_height != padded_input_height) {
        pad_bottom += (extra_input_height - padded_input_height);
      }
      if (extra_input_width != padded_input_width) {
        pad_right += (extra_input_width - padded_input_width);
      }
    }

    // decide scratch size before allocate it
    index_t total_scratch_size = 0;
    index_t transformed_input_size = 0;
    index_t transformed_output_size = 0;
    index_t padded_input_size = 0;
    index_t padded_output_size = 0;
    if (use_winograd) {
      transformed_input_size =
        std::accumulate(transformed_input_shape.begin(),
                        transformed_input_shape.end(),
                        1,
                        std::multiplies<index_t>()) * sizeof(float);
      transformed_output_size =
        std::accumulate(transformed_output_shape.begin(),
                        transformed_output_shape.end(),
                        1,
                        std::multiplies<index_t>()) * sizeof(float);
      total_scratch_size += transformed_input_size + transformed_output_size;
    }
    if (extra_input_height != input_height
      || extra_input_width != input_width) {
      padded_input_size =
        batch * input_channels * (input_height + pad_top + pad_bottom)
          * (input_width + pad_left + pad_right) * sizeof(float) +
            MACE_EXTRA_BUFFER_PAD_SIZE;
      total_scratch_size += padded_input_size;
    }
    if (extra_output_height != height || extra_output_width != width) {
      padded_output_size =
        batch * channels * extra_output_height * extra_output_width
          * sizeof(float);
      total_scratch_size += padded_output_size;
    }
    // Init scratch buffer
    scratch_->Rewind();
    scratch_->GrowSize(total_scratch_size);
    Tensor
      transformed_input(scratch_->Scratch(transformed_input_size), DT_FLOAT);
    Tensor
      transformed_output(scratch_->Scratch(transformed_output_size), DT_FLOAT);
    Tensor padded_input(scratch_->Scratch(padded_input_size), DT_FLOAT);
    Tensor padded_output(scratch_->Scratch(padded_output_size), DT_FLOAT);
    const index_t extra_input_shape[4] =
        {batch, input_channels, extra_input_height, extra_input_width};
    const index_t extra_output_shape[4] =
        {batch, channels, extra_output_height, extra_output_width};

    // make host compiler happy
    MACE_UNUSED(extra_input_shape);
    MACE_UNUSED(extra_output_shape);

    // decide which convolution function to call
    if (use_winograd) {
      transformed_input.Reshape(transformed_input_shape);
      transformed_output.Reshape(transformed_output_shape);
      const float *transformed_filter_ptr;
      if (transformed_filter_.dim_size() == 0) {
        if (is_filter_transformed_) {
          transformed_filter_ptr = filter_data;
        } else {
          MACE_RETURN_IF_ERROR(transformed_filter_.Resize(
              transformed_filter_shape));
          switch (winograd_out_tile_size) {
            case 2:
              TransformFilter4x4(filter_data,
                                 filter_shape[1],
                                 filter_shape[0],
                                 transformed_filter_.mutable_data<float>());
              break;
            case 6:
              TransformFilter8x8(filter_data,
                                 filter_shape[1],
                                 filter_shape[0],
                                 transformed_filter_.mutable_data<float>());
              break;
            default:MACE_NOT_IMPLEMENTED;
          }
          transformed_filter_ptr = transformed_filter_.data<float>();
        }
      } else {
        transformed_filter_ptr = transformed_filter_.data<float>();
      }

      float *transformed_input_data = transformed_input.mutable_data<float>();
      float *transformed_output_data = transformed_output.mutable_data<float>();

      conv_func = [=](const float *pad_input, float *pad_output) {
        WinoGradConv3x3s1(pad_input,
                          transformed_filter_ptr,
                          batch,
                          extra_input_height,
                          extra_input_width,
                          input_channels,
                          channels,
                          winograd_out_tile_size,
                          transformed_input_data,
                          transformed_output_data,
                          pad_output);
      };
    } else if (use_neon_3x3_s1) {
      conv_func = [=](const float *pad_input, float *pad_output) {
        Conv2dNeonK3x3S1(pad_input,
                         filter_data,
                         extra_input_shape,
                         extra_output_shape,
                         pad_output);
      };
    } else if (use_neon_3x3_s2) {
      conv_func = [=](const float *pad_input, float *pad_output) {
        Conv2dNeonK3x3S2(pad_input,
                         filter_data,
                         extra_input_shape,
                         extra_output_shape,
                         pad_output);
      };
    } else if (use_neon_1x1_s1) {
      conv_func = [=](const float *pad_input, float *pad_output) {
        Conv2dNeonK1x1S1(pad_input,
                         filter_data,
                         batch,
                         extra_input_height,
                         extra_input_width,
                         input_channels,
                         channels,
                         pad_output);
      };
    } else if (use_neon_5x5_s1) {
      conv_func = [=](const float *pad_input, float *pad_output) {
        Conv2dNeonK5x5S1(pad_input,
                         filter_data,
                         extra_input_shape,
                         extra_output_shape,
                         pad_output);
      };
    } else if (use_neon_1x7_s1) {
      conv_func = [=](const float *pad_input, float *pad_output) {
        Conv2dNeonK1x7S1(pad_input,
                         filter_data,
                         extra_input_shape,
                         extra_output_shape,
                         pad_output);
      };
    } else if (use_neon_7x1_s1) {
      conv_func = [=](const float *pad_input, float *pad_output) {
        Conv2dNeonK7x1S1(pad_input,
                         filter_data,
                         extra_input_shape,
                         extra_output_shape,
                         pad_output);
      };
    } else if (use_neon_7x7_s1) {
      conv_func = [=](const float *pad_input, float *pad_output) {
        Conv2dNeonK7x7S1(pad_input,
                         filter_data,
                         extra_input_shape,
                         extra_output_shape,
                         pad_output);
      };
    } else if (use_neon_7x7_s2) {
      conv_func = [=](const float *pad_input, float *pad_output) {
        Conv2dNeonK7x7S2(pad_input,
                         filter_data,
                         extra_input_shape,
                         extra_output_shape,
                         pad_output);
      };
    } else if (use_neon_7x7_s3) {
      conv_func = [=](const float *pad_input, float *pad_output) {
        Conv2dNeonK7x7S3(pad_input,
                         filter_data,
                         extra_input_shape,
                         extra_output_shape,
                         pad_output);
      };
    } else if (use_neon_1x15_s1) {
      conv_func = [=](const float *pad_input, float *pad_output) {
        Conv2dNeonK1x15S1(pad_input,
                         filter_data,
                         extra_input_shape,
                         extra_output_shape,
                         pad_output);
      };
    } else if (use_neon_15x1_s1) {
      conv_func = [=](const float *pad_input, float *pad_output) {
        Conv2dNeonK15x1S1(pad_input,
                          filter_data,
                          extra_input_shape,
                          extra_output_shape,
                          pad_output);
      };
    } else {
      conv_func = [=](const float *pad_input, float *pad_output) {
        Conv2dGeneral(pad_input,
                      filter_data,
                      extra_input_shape,
                      extra_output_shape,
                      filter_shape.data(),
                      strides_,
                      dilations_,
                      pad_output);
      };
    }

    // pad input and output
    const Tensor *pad_input_ptr = input;
    if (extra_input_height != input_height
      || extra_input_width != input_width) {
      MACE_RETURN_IF_ERROR(ConstructNCHWInputWithSpecificPadding(input,
                                            pad_top,
                                            pad_bottom,
                                            pad_left,
                                            pad_right,
                                            &padded_input));
      pad_input_ptr = &padded_input;
    }

    // TODO(libin): don't need clear after bias is integrated in each conv
    Tensor *pad_output_ptr = output;
    if (extra_output_height != height || extra_output_width != width) {
      padded_output.Reshape({batch, channels, extra_output_height,
                            extra_output_width});
      padded_output.Clear();
      pad_output_ptr = &padded_output;
    } else if (!use_neon_1x1_s1) {
      output->Clear();
    }

    const float *pad_input_data = pad_input_ptr->data<float>();
    float *pad_output_data = pad_output_ptr->mutable_data<float>();

    conv_func(pad_input_data, pad_output_data);

    // unpack output
    if (extra_output_height != height || extra_output_width != width) {
#pragma omp parallel for collapse(2)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channels; ++c) {
          for (index_t h = 0; h < height; ++h) {
            memcpy(
              output_data + b * channels * height * width + c * height * width
                + h * width,
              pad_output_data
                + b * channels * extra_output_height * extra_output_width
                + c * extra_output_height * extra_output_width
                + h * extra_output_width,
              sizeof(float) * width);
          }
        }
      }
    }

    if (bias_data != nullptr) {
#pragma omp parallel for collapse(2)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channels; ++c) {
          for (index_t i = 0; i < height * width; ++i) {
            output_data[(b * channels + c) * height * width + i] +=
              bias_data[c];
          }
        }
      }
    }

    DoActivation(output_data, output_data, output->size(), activation_,
                 relux_max_limit_);

    return MACE_SUCCESS;
  }

  Tensor transformed_filter_;
  bool is_filter_transformed_;
  ScratchBuffer *scratch_;
};

template<>
struct Conv2dFunctor<DeviceType::CPU, uint8_t> : Conv2dFunctorBase {
  Conv2dFunctor(const int *strides,
                const Padding &padding_type,
                const std::vector<int> &paddings,
                const int *dilations,
                const ActivationType activation,
                const float relux_max_limit,
                const bool is_filter_transformed,
                ScratchBuffer *scratch)
      : Conv2dFunctorBase(strides,
                          padding_type,
                          paddings,
                          dilations,
                          activation,
                          relux_max_limit),
        scratch_(scratch) {
    MACE_UNUSED(is_filter_transformed);
  }

  template <typename T>
  inline void Im2col(
      const T *in_data, const std::vector<index_t> &in_shape,
      const index_t filter_h, const index_t filter_w, const index_t stride_h,
      const index_t stride_w, const T zero_point, const int pad_height,
      const int pad_width, const std::vector<index_t> &out_shape,
      const index_t depth, T* im2col_data) {
    const index_t input_row_size = in_shape[2] * in_shape[3];
    const index_t patch_row_size = filter_w * in_shape[3];

#pragma omp parallel for collapse(3)
    for (index_t b = 0; b < out_shape[0]; ++b) {
      for (index_t h = 0; h < out_shape[1]; ++h) {
        for (index_t w = 0; w < out_shape[2]; ++w) {
          // Reshape a patch of input to column, which is corresponding to
          // a column of output(:, column).
          const index_t ih_begin = h * stride_h - (pad_height >> 1);
          const index_t ih_end = ih_begin + filter_h;
          const index_t iw_begin = w * stride_w - (pad_width >> 1);
          const index_t iw_end = iw_begin + filter_w;
          // gate height and width to separate padding
          const index_t ih_begin_gated = std::max<index_t>(0, ih_begin);
          const index_t ih_end_gated = std::min<index_t>(ih_end, in_shape[1]);
          const index_t iw_begin_gated = std::max<index_t>(0, iw_begin);
          const index_t iw_end_gated = std::min<index_t>(iw_end, in_shape[2]);
          const index_t pad_top = std::max<index_t>(0, -ih_begin);
          const index_t pad_bottom = ih_end - ih_end_gated;
          const index_t pad_left = std::max<index_t>(0, -iw_begin);
          const index_t pad_right = iw_end - iw_end_gated;
          index_t im2col_column_offset =
              ((b * out_shape[1] + h) * out_shape[2] + w) * depth;

          // fill in padding top
          if (pad_top > 0) {
            std::fill_n(im2col_data + im2col_column_offset,
                        pad_top * patch_row_size, zero_point);
          }

          const index_t patch_row_size_gated =
              std::min(filter_w - pad_left,
                       in_shape[2] - iw_begin_gated) * in_shape[3];
          MACE_CHECK(patch_row_size_gated ==
              ((filter_w - (pad_left + pad_right)) * in_shape[3]));
          const index_t pad_left_size = pad_left * in_shape[3];
          const index_t pad_right_size = pad_right * in_shape[3];
          index_t im2col_offset = im2col_column_offset +
              (pad_top * filter_w + pad_left) * in_shape[3];
          index_t in_offset = ((b * in_shape[1] + ih_begin_gated) * in_shape[2]
              + iw_begin_gated) * in_shape[3];

          // fill in effective rows
          for (index_t ih = ih_begin_gated; ih < ih_end_gated; ++ih) {
            // fill in padding left
            if (pad_left > 0) {
              const index_t left_offset = im2col_offset - pad_left_size;
              std::fill_n(im2col_data + left_offset, pad_left_size, zero_point);
            }
            // copy effective data
            std::copy_n(in_data + in_offset, patch_row_size_gated,
                        im2col_data + im2col_offset);
            // fill in padding right
            if (pad_right > 0) {
              const index_t right_offset = im2col_offset + patch_row_size_gated;
              std::fill_n(im2col_data + right_offset, pad_right_size,
                          zero_point);
            }
            in_offset += input_row_size;
            im2col_offset += patch_row_size;
          }

          // fill in padding bottom
          if (pad_bottom > 0) {
            const index_t pad_bottom_size = pad_bottom * patch_row_size;
            const index_t bottom_offset =
                im2col_column_offset + depth - pad_bottom_size;
            std::fill_n(im2col_data + bottom_offset, pad_bottom_size,
                        zero_point);
          }
        }
      }
    }
  }

  MaceStatus operator()(const Tensor *input,   // NHWC
                        const Tensor *filter,  // OHWI
                        const Tensor *bias,
                        Tensor *output,        // NHWC
                        StatsFuture *future) {
    MACE_UNUSED(future);
    MACE_CHECK(dilations_[0] == 1 && dilations_[1] == 1,
               "Quantization convolution does not support dilation > 1 yet.");

    gemmlowp::GemmContext& gemm_context = GetGemmlowpContext();

    std::vector<index_t> output_shape(4);
    std::vector<int> paddings(2);
    if (paddings_.empty()) {
      CalcPaddingAndOutputSize(input->shape().data(),
                               NHWC,
                               filter->shape().data(),
                               OHWI,
                               dilations_,
                               strides_,
                               padding_type_,
                               output_shape.data(),
                               paddings.data());
    } else {
      paddings = paddings_;
      CalcOutputSize(input->shape().data(),
                     NHWC,
                     filter->shape().data(),
                     OHWI,
                     paddings_.data(),
                     dilations_,
                     strides_,
                     RoundType::FLOOR,
                     output_shape.data());
    }
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    index_t batch = output->dim(0);
    index_t height = output->dim(1);
    index_t width = output->dim(2);
    index_t channels = output->dim(3);
    index_t input_batch = input->dim(0);
    index_t input_channels = input->dim(3);
    index_t filter_h = filter->dim(1);
    index_t filter_w = filter->dim(2);
    index_t stride_h = strides_[0];
    index_t stride_w = strides_[1];
    const index_t depth = input_channels * filter_h * filter_w;
    const index_t columns = batch * height * width;

    MACE_CHECK(filter->dim(0) == channels, filter->dim(0), " != ", channels);
    MACE_CHECK(filter->dim(3) == input_channels, filter->dim(3), " != ",
               input_channels);
    MACE_CHECK(batch == input_batch, "Input/Output batch size mismatch");

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard filter_guard(filter);
    Tensor::MappingGuard output_guard(output);

    auto input_data = input->data<uint8_t>();
    auto filter_data = filter->data<uint8_t>();
    auto output_data = output->mutable_data<uint8_t>();

    index_t total_scratch_size = 0;
    index_t zero_bias_size = channels * sizeof(int32_t);
    total_scratch_size += (bias == nullptr ? zero_bias_size : 0);
    index_t im2col_size = depth * columns * sizeof(uint8_t);
    bool im2col_required =
        filter_h != 1 || filter_w != 1 || stride_h != 1 || stride_w != 1;
    total_scratch_size += (im2col_required ? im2col_size : 0);
    scratch_->Rewind();
    scratch_->GrowSize(total_scratch_size);

    std::unique_ptr<Tensor> zero_bias;
    const int32_t *bias_data = nullptr;
    if (bias == nullptr) {
      zero_bias.reset(new Tensor(scratch_->Scratch(zero_bias_size), DT_INT32));
      zero_bias->Reshape({channels});
      zero_bias->Clear();
      bias_data = zero_bias->data<int32_t>();
    } else {
      bias_data = bias->data<int32_t>();
    }

    std::unique_ptr<Tensor> im2col;
    auto gemm_input_data = input_data;
    if (im2col_required) {
      // prepare im2col
      im2col.reset(new Tensor(scratch_->Scratch(im2col_size), DT_UINT8));
      uint8_t *im2col_data = im2col->mutable_data<uint8_t>();
      Im2col(input_data, input->shape(), filter_h, filter_w, stride_h,
             stride_w, static_cast<uint8_t>(input->zero_point()),
             paddings[0], paddings[1], output->shape(), depth, im2col_data);
      gemm_input_data = im2col_data;
    }

    const int gemm_filter_rows = static_cast<int>(channels);
    const int gemm_filter_cols = static_cast<int>(depth);
    const int gemm_input_rows = static_cast<int>(depth);
    const int gemm_input_cols = static_cast<int>(columns);
    const int gemm_output_rows = static_cast<int>(channels);
    const int gemm_output_cols = static_cast<int>(columns);
    gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::RowMajor>
        filter_matrix(filter_data, gemm_filter_rows, gemm_filter_cols);
    gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::ColMajor>
        input_matrix(gemm_input_data, gemm_input_rows, gemm_input_cols);
    gemmlowp::MatrixMap<uint8_t, gemmlowp::MapOrder::ColMajor>
        output_matrix(output_data, gemm_output_rows, gemm_output_cols);

    const auto &output_pipeline = GemmlowpOutputPipeline::Make(
        bias_data, channels, filter->scale(), input->scale(), output->scale(),
        output->zero_point());

    using BitDepthParams = gemmlowp::L8R8WithLhsNonzeroBitDepthParams;
    gemmlowp::GemmWithOutputPipeline<uint8_t, uint8_t, BitDepthParams>(
        &gemm_context, filter_matrix, input_matrix, &output_matrix,
        -filter->zero_point(), -input->zero_point(), output_pipeline);

    return MACE_SUCCESS;
  }

  ScratchBuffer *scratch_;
};

#ifdef MACE_ENABLE_OPENCL
template<typename T>
struct Conv2dFunctor<DeviceType::GPU, T> : Conv2dFunctorBase {
  Conv2dFunctor(const int *strides,
                const Padding &padding_type,
                const std::vector<int> &paddings,
                const int *dilations,
                const ActivationType activation,
                const float relux_max_limit,
                const bool is_filter_transformed,
                ScratchBuffer *scratch)
    : Conv2dFunctorBase(strides,
                        padding_type,
                        paddings,
                        dilations,
                        activation,
                        relux_max_limit) {
    MACE_UNUSED(is_filter_transformed);
    MACE_UNUSED(scratch);
  }

  MaceStatus operator()(const Tensor *input,
                  const Tensor *filter,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_CONV_2D_H_
