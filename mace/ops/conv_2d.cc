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

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/operator.h"
#include "mace/core/tensor.h"
#include "mace/ops/activation.h"
#include "mace/ops/arm/conv_2d_neon.h"
#include "mace/ops/arm/conv_winograd.h"
#include "mace/ops/conv_pool_2d_base.h"
#include "mace/ops/conv_pool_2d_util.h"
#include "mace/utils/utils.h"

#ifdef MACE_ENABLE_QUANTIZE
#include "mace/ops/gemmlowp_util.h"
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/buffer/conv_2d.h"
#include "mace/ops/opencl/image/conv_2d.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template <DeviceType D, class T>
class Conv2dOp;

template <>
class Conv2dOp<DeviceType::CPU, float> : public ConvPool2dOpBase {
 public:
  explicit Conv2dOp(OpConstructContext *context)
      : ConvPool2dOpBase(context),
        activation_(ops::StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation",
                                                  "NOOP"))),
        relux_max_limit_(Operation::GetOptionalArg<float>("max_limit", 0.0f)),
        leakyrelu_coefficient_(Operation::GetOptionalArg<float>(
              "leakyrelu_coefficient", 0.0f)),
        is_filter_transformed_(false) {}

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    std::vector<index_t> filter_shape(4);
    filter_shape = filter->shape();

    std::vector<index_t> output_shape(4);
    std::vector<int> paddings(2);
    if (paddings_.empty()) {
      CalcNCHWPaddingAndOutputSize(input->shape().data(),
                                   filter_shape.data(),
                                   dilations_.data(),
                                   strides_.data(),
                                   padding_type_,
                                   output_shape.data(),
                                   paddings.data());
    } else {
      paddings = paddings_;
      CalcNCHWOutputSize(input->shape().data(),
                         filter_shape.data(),
                         paddings_.data(),
                         dilations_.data(),
                         strides_.data(),
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
        use_winograd = filter_h == 3 && filter_w == 3
        && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1
        && input_channels >= 8 && channels >= 8;
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
    // scratch for sgemm, preoccupy enough buffer
    if (use_neon_1x1_s1) {
      total_scratch_size += (input_batch * input_height * input_width
          * (input_channels + channels))
          * sizeof(float);
    } else if (use_winograd) {
      total_scratch_size += transformed_input_size + transformed_output_size;
    }

    // Init scratch buffer
    ScratchBuffer *scratch = context->device()->scratch_buffer();
    scratch->Rewind();
    scratch->GrowSize(total_scratch_size);
    Tensor
        transformed_input(scratch->Scratch(transformed_input_size), DT_FLOAT);
    Tensor
        transformed_output(scratch->Scratch(transformed_output_size), DT_FLOAT);
    Tensor padded_input(scratch->Scratch(padded_input_size), DT_FLOAT);
    Tensor padded_output(scratch->Scratch(padded_output_size), DT_FLOAT);
    const index_t extra_input_shape[4] =
        {batch, input_channels, extra_input_height, extra_input_width};
    const index_t extra_output_shape[4] =
        {batch, channels, extra_output_height, extra_output_width};

    // make host compiler happy
    MACE_UNUSED(extra_input_shape);
    MACE_UNUSED(extra_output_shape);

    Tensor transformed_filter;

    // decide which convolution function to call
    if (use_winograd) {
      transformed_input.Reshape(transformed_input_shape);
      transformed_output.Reshape(transformed_output_shape);
      const float *transformed_filter_data = nullptr;
      // filter only needs to be transformed once, set transformed_filter_data
      // to null after the first run.
      if (!is_filter_transformed_) {
        transformed_filter.Resize(transformed_filter_shape);
        switch (winograd_out_tile_size) {
          case 2:
            TransformFilter4x4(filter_data,
                               filter_shape[1],
                               filter_shape[0],
                               transformed_filter.mutable_data<float>());
            break;
          case 6:
            TransformFilter8x8(filter_data,
                               filter_shape[1],
                               filter_shape[0],
                               transformed_filter.mutable_data<float>());
            break;
          default:MACE_NOT_IMPLEMENTED;
        }
        transformed_filter_data = transformed_filter.data<float>();
        is_filter_transformed_ = true;
      }

      float *transformed_input_data = transformed_input.mutable_data<float>();
      float *transformed_output_data = transformed_output.mutable_data<float>();

      conv_func = [=](const float *pad_input, float *pad_output) {
        WinoGradConv3x3s1(pad_input,
                          transformed_filter_data,
                          batch,
                          extra_input_height,
                          extra_input_width,
                          input_channels,
                          channels,
                          winograd_out_tile_size,
                          transformed_input_data,
                          transformed_output_data,
                          pad_output,
                          &sgemm_,
                          scratch);
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
                         pad_output,
                         &sgemm_,
                         scratch);
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
                      strides_.data(),
                      dilations_.data(),
                      pad_output);
      };
    }

    // pad input and output
    const Tensor *pad_input_ptr = input;
    if (extra_input_height != input_height
        || extra_input_width != input_width) {
      MACE_RETURN_IF_ERROR(ConstructNCHWInputWithSpecificPadding(
          input, pad_top, pad_bottom, pad_left, pad_right, &padded_input));
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
#pragma omp parallel for collapse(2) schedule(runtime)
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
      const index_t image_size = height * width;
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channels; ++c) {
          float *output_ptr = output_data + (b * channels + c) * image_size;
          const float bias = bias_data[c];
#if defined(MACE_ENABLE_NEON)
          float32x4_t vbias = vdupq_n_f32(bias);
          for (index_t i = 0; i <= image_size - 4; i += 4) {
            float32x4_t v = vld1q_f32(output_ptr + i);
            v = vaddq_f32(v, vbias);
            vst1q_f32(output_ptr + i, v);
          }
          for (index_t i = (image_size >> 2) << 2; i < image_size; ++i) {
            output_ptr[i] += bias;
          }
#else
          for (index_t i = 0; i < image_size; ++i) {
            output_ptr[i] += bias;
          }
#endif
        }
      }
    }

    DoActivation(output_data, output_data, output->size(), activation_,
                 relux_max_limit_, leakyrelu_coefficient_);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
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

#pragma omp parallel for collapse(2) schedule(runtime)
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

 private:
  const ActivationType activation_;
  const float relux_max_limit_;
  const float leakyrelu_coefficient_;
  bool is_filter_transformed_;
  SGemm sgemm_;

 private:
  MACE_OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};


#ifdef MACE_ENABLE_QUANTIZE
template <>
class Conv2dOp<DeviceType::CPU, uint8_t> : public ConvPool2dOpBase {
 public:
  explicit Conv2dOp(OpConstructContext *context)
      : ConvPool2dOpBase(context),
        activation_(ops::StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation",
                                                  "NOOP"))),
        relux_max_limit_(Operation::GetOptionalArg<float>("max_limit", 0.0f)),
        leakyrelu_coefficient_(Operation::GetOptionalArg<float>(
              "leakyrelu_coefficient", 0.0f)) {}

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    MACE_CHECK(dilations_[0] == 1 && dilations_[1] == 1,
               "Quantization convolution does not support dilation > 1 yet.");

    auto gemm_context = context->device()->cpu_runtime()->GetGemmlowpContext();
    MACE_CHECK_NOTNULL(gemm_context);

    std::vector<index_t> output_shape(4);
    std::vector<int> paddings(2);
    if (paddings_.empty()) {
      CalcPaddingAndOutputSize(input->shape().data(),
                               NHWC,
                               filter->shape().data(),
                               OHWI,
                               dilations_.data(),
                               strides_.data(),
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
                     dilations_.data(),
                     strides_.data(),
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

    VLOG(2) << "input scale/zero: " << input->scale() << ", "
            << input->zero_point();
    VLOG(2) << "filter scale/zero: " << filter->scale() << ", "
            << filter->zero_point();
    if (bias) {
      VLOG(2) << "bias scale/zero: " << bias->scale() << ", "
              << bias->zero_point();
    }
    VLOG(2) << "output scale/zero: " << output->scale() << ", "
            << output->zero_point();

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
    ScratchBuffer *scratch = context->device()->scratch_buffer();
    scratch->Rewind();
    scratch->GrowSize(total_scratch_size);

    std::unique_ptr<Tensor> zero_bias;
    const int32_t *bias_data = nullptr;
    if (bias == nullptr) {
      zero_bias.reset(new Tensor(scratch->Scratch(zero_bias_size), DT_INT32));
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
      im2col.reset(new Tensor(scratch->Scratch(im2col_size), DT_UINT8));
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
        gemm_context, filter_matrix, input_matrix, &output_matrix,
        -filter->zero_point(), -input->zero_point(), output_pipeline);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  template <typename T>
  inline void Im2col(
      const T *in_data, const std::vector<index_t> &in_shape,
      const index_t filter_h, const index_t filter_w, const index_t stride_h,
      const index_t stride_w, const T zero_point, const int pad_height,
      const int pad_width, const std::vector<index_t> &out_shape,
      const index_t depth, T* im2col_data) {
    const index_t input_row_size = in_shape[2] * in_shape[3];
    const index_t patch_row_size = filter_w * in_shape[3];

#pragma omp parallel for collapse(3) schedule(runtime)
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

 private:
  const ActivationType activation_;
  const float relux_max_limit_;
  const float leakyrelu_coefficient_;

 private:
  MACE_OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class Conv2dOp<DeviceType::GPU, T> : public ConvPool2dOpBase {
 public:
  explicit Conv2dOp(OpConstructContext *context)
      : ConvPool2dOpBase(context),
        activation_(ops::StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation",
                                                   "NOOP"))),
        relux_max_limit_(Operation::GetOptionalArg<float>("max_limit", 0.0f)),
        leakyrelu_coefficient_(Operation::GetOptionalArg<float>(
              "leakyrelu_coefficient", 0.0f)),
        wino_block_size_(Operation::GetOptionalArg<int>("wino_block_size", 0)) {
    MemoryType mem_type;
    if (context->device()->gpu_runtime()->UseImageMemory()) {
      mem_type = MemoryType::GPU_IMAGE;
      kernel_.reset(new opencl::image::Conv2dKernel<T>);
    } else {
      mem_type = MemoryType::GPU_BUFFER;
      kernel_.reset(new opencl::buffer::Conv2dKernel<T>);
    }
    context->set_output_mem_type(mem_type);
    // Transform filter tensor to target format
    if ((wino_block_size_ == 2 || wino_block_size_ == 4) &&
        (kernel_->CheckUseWinograd(
          context->device()->gpu_runtime()->opencl_runtime(),
          context->workspace()->GetTensor(
              operator_def_->input(1))->shape(),
          std::vector<index_t>(operator_def_->output_shape(0).dims().begin(),
                               operator_def_->output_shape(0).dims().end()),
          strides_.data(),
          dilations_.data(),
          &wino_block_size_))) {
      MACE_CHECK(TransformFilter<T>(
          context, operator_def_.get(), 1,
          OpenCLBufferType::WINOGRAD_FILTER, mem_type, wino_block_size_)
                     == MaceStatus::MACE_SUCCESS);
    } else {
      wino_block_size_ = 0;
      MACE_CHECK(TransformFilter<T>(
          context, operator_def_.get(), 1,
          OpenCLBufferType::CONV2D_FILTER, mem_type)
                     == MaceStatus::MACE_SUCCESS);
    }
    if (operator_def_->input_size() > 2) {
      MACE_CHECK(TransformFilter<T>(
          context, operator_def_.get(), 2, OpenCLBufferType::ARGUMENT, mem_type)
                     == MaceStatus::MACE_SUCCESS);
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);
    return kernel_->Compute(context, input, filter, bias,
                            strides_.data(), padding_type_, paddings_,
                            dilations_.data(), activation_, relux_max_limit_,
                            leakyrelu_coefficient_, wino_block_size_, output);
  }

 private:
  const ActivationType activation_;
  const float relux_max_limit_;
  const float leakyrelu_coefficient_;
  std::unique_ptr<OpenCLConv2dKernel> kernel_;
  int wino_block_size_;

 private:
  MACE_OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
#endif  // MACE_ENABLE_OPENCL


void RegisterConv2D(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Conv2D", Conv2dOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_OP(op_registry, "Conv2D", Conv2dOp,
                   DeviceType::CPU, uint8_t);
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "Conv2D", Conv2dOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "Conv2D", Conv2dOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
