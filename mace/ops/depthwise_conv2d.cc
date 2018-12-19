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
#include <memory>
#include <string>
#include <vector>

#ifdef MACE_ENABLE_QUANTIZE
// We reuse TensorFlow Lite's optimized depthwiseconv_uint8 and parallelized it
// using OpenMP for MACE's quantized depthwise_conv2d.
#include "tensorflow/contrib/lite/kernels/internal/optimized/depthwiseconv_uint8.h"
#endif  // MACE_ENABLE_QUANTIZE

#include "mace/core/future.h"
#include "mace/core/operator.h"
#include "mace/ops/activation.h"
#include "mace/ops/arm/depthwise_conv2d_neon.h"
#include "mace/ops/conv_pool_2d_base.h"
#include "mace/public/mace.h"
#include "mace/utils/quantize.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/buffer/depthwise_conv2d.h"
#include "mace/ops/opencl/image/depthwise_conv2d.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

class DepthwiseConv2dOpBase : public ConvPool2dOpBase {
 public:
  explicit DepthwiseConv2dOpBase(OpConstructContext *context)
      : ConvPool2dOpBase(context),
        activation_(ops::StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation",
                                                  "NOOP"))),
        relux_max_limit_(Operation::GetOptionalArg<float>("max_limit", 0.0f)),
        leakyrelu_coefficient_(Operation::GetOptionalArg<float>(
              "leakyrelu_coefficient", 0.0f)) {}
 protected:
  const ActivationType activation_;
  const float relux_max_limit_;
  const float leakyrelu_coefficient_;
};

template <DeviceType D, class T>
class DepthwiseConv2dOp;

template <>
class DepthwiseConv2dOp<DeviceType::CPU, float> : public DepthwiseConv2dOpBase {
 public:
  explicit DepthwiseConv2dOp(OpConstructContext *context)
      : DepthwiseConv2dOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = nullptr;
    if (this->InputSize() >= 3) {
      bias = this->Input(BIAS);
    }
    Tensor *output = this->Output(OUTPUT);
    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

    std::vector<index_t> output_shape(4);
    std::vector<int> paddings(2);
    std::vector<index_t> filter_shape
        {filter->dim(0) * filter->dim(1), filter->dim(1), filter->dim(2),
         filter->dim(3)};

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
    output->Clear();

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

    int pad_top = paddings[0] >> 1;
    int pad_bottom = paddings[0] - pad_top;
    int pad_left = paddings[1] >> 1;
    int pad_right = paddings[1] - pad_left;

    index_t valid_h_start = pad_top == 0 ? 0 : (pad_top - 1) / stride_h + 1;
    index_t valid_h_stop = pad_bottom == 0
                           ? height
                           : height - ((pad_bottom - 1) / stride_h + 1);
    index_t valid_w_start = pad_left == 0 ? 0 : (pad_left - 1) / stride_w + 1;
    index_t valid_w_stop = pad_right == 0
                           ? width
                           : width - ((pad_right - 1) / stride_w + 1);

    std::function<void(const float *input, float *output)> conv_func;

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard filter_guard(filter);
    Tensor::MappingGuard bias_guard(bias);
    Tensor::MappingGuard output_guard(output);
    auto input_data = input->data<float>();
    auto filter_data = filter->data<float>();
    auto bias_data = bias == nullptr ? nullptr : bias->data<float>();
    auto output_data = output->mutable_data<float>();

    const int pad_hw[2] = {pad_top, pad_left};
    const index_t input_shape[4] =
        {batch, input_channels, input_height, input_width};

    // make host compiler happy
    MACE_UNUSED(pad_hw);
    MACE_UNUSED(input_shape);

    if (filter_h == 3 && filter_w == 3 && stride_h == 1 && stride_w == 1
        && dilation_h == 1 && dilation_w == 1) {
      conv_func = [=](const float *input, float *output) {
        DepthwiseConv2dNeonK3x3S1(input,
                                  filter_data,
                                  input_shape,
                                  output_shape.data(),
                                  pad_hw,
                                  valid_h_start,
                                  valid_h_stop,
                                  valid_w_start,
                                  valid_w_stop,
                                  output);
      };
    } else if (filter_h == 3 && filter_w == 3 && stride_h == 2 && stride_w == 2
        && dilation_h == 1 && dilation_w == 1) {
      conv_func = [=](const float *input, float *output) {
        DepthwiseConv2dNeonK3x3S2(input,
                                  filter_data,
                                  input_shape,
                                  output_shape.data(),
                                  pad_hw,
                                  valid_h_start,
                                  valid_h_stop,
                                  valid_w_start,
                                  valid_w_stop,
                                  output);
      };
    } else {
      conv_func = [=](const float *input, float *output) {
        DepthwiseConv2dGeneral(input,
                               filter_data,
                               input_shape,
                               output_shape.data(),
                               filter_shape.data(),
                               strides_.data(),
                               dilations_.data(),
                               pad_hw,
                               output);
      };
    }

    conv_func(input_data, output_data);

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
                 relux_max_limit_, leakyrelu_coefficient_);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  void DepthwiseConv2dGeneral(const float *input,
                              const float *filter,
                              const index_t *in_shape,
                              const index_t *out_shape,
                              const index_t *filter_shape,
                              const int *stride_hw,
                              const int *dilation_hw,
                              const int *pad_hw,
                              float *output) {
    const index_t multiplier = filter_shape[0] / filter_shape[1];
#pragma omp parallel for collapse(2)
    for (index_t b = 0; b < in_shape[0]; ++b) {
      for (index_t m = 0; m < filter_shape[0]; ++m) {
        for (index_t h = 0; h < out_shape[2]; ++h) {
          for (index_t w = 0; w < out_shape[3]; ++w) {
            const index_t out_channels = filter_shape[0];
            const index_t in_channels = filter_shape[1];
            const index_t filter_height = filter_shape[2];
            const index_t filter_width = filter_shape[3];
            const index_t in_height = in_shape[2];
            const index_t in_width = in_shape[3];
            const index_t out_height = out_shape[2];
            const index_t out_width = out_shape[3];
            index_t out_offset =
                ((b * out_channels + m) * out_height + h) * out_width + w;
            index_t c = m / multiplier;
            index_t o = m % multiplier;
            float sum = 0;
            for (index_t kh = 0; kh < filter_height; ++kh) {
              for (index_t kw = 0; kw < filter_width; ++kw) {
                index_t ih = h * stride_hw[0] + kh * dilation_hw[0] - pad_hw[0];
                index_t iw = w * stride_hw[1] + kw * dilation_hw[1] - pad_hw[1];
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                  index_t in_offset =
                      ((b * in_channels + c) * in_height + ih) * in_width + iw;
                  index_t filter_offset =
                      (((o * in_channels) + c) * filter_height + kh)
                          * filter_width + kw;

                  sum += input[in_offset] * filter[filter_offset];
                }
              }
            }
            output[out_offset] = sum;
          }
        }
      }
    }
  }

 protected:
  MACE_OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

#ifdef MACE_ENABLE_QUANTIZE
template <>
class DepthwiseConv2dOp<DeviceType::CPU, uint8_t>
    : public DepthwiseConv2dOpBase {
 public:
  explicit DepthwiseConv2dOp(OpConstructContext *context)
      : DepthwiseConv2dOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = nullptr;
    if (this->InputSize() >= 3) {
      bias = this->Input(BIAS);
    }
    Tensor *output = this->Output(OUTPUT);
    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

    std::vector<index_t> output_shape(4);
    std::vector<int> paddings(2);

    // reuse OHWI format, only for calculating output
    std::vector<index_t> ohwi_shape{
        filter->dim(2) * filter->dim(3), filter->dim(0), filter->dim(1), 1};
    if (paddings_.empty()) {
      CalcPaddingAndOutputSize(input->shape().data(),
                               NHWC,
                               ohwi_shape.data(),
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
                     ohwi_shape.data(),
                     OHWI,
                     paddings_.data(),
                     dilations_.data(),
                     strides_.data(),
                     RoundType::FLOOR,
                     output_shape.data());
    }

    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    output->Clear();

    MACE_CHECK(output->dim(0) == input->dim(0),
               "Input/Output batch size mismatch");
    MACE_CHECK(filter->dim(2) == input->dim(3), filter->dim(2), " != ",
               input->dim(3));

    index_t out_channels = output_shape[3];
    index_t stride_h = strides_[0];
    index_t stride_w = strides_[1];
    index_t dilation_h = dilations_[0];
    index_t dilation_w = dilations_[1];
    int pad_top = paddings[0] >> 1;
    int pad_left = paddings[1] >> 1;

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard filter_guard(filter);
    Tensor::MappingGuard bias_guard(bias);
    Tensor::MappingGuard output_guard(output);
    auto input_data = input->data<uint8_t>();
    auto filter_data = filter->data<uint8_t>();
    auto output_data = output->mutable_data<uint8_t>();

    if (dilation_h == 1 && dilation_w == 1) {
      std::vector<index_t> bias_shape{out_channels};
      std::unique_ptr<Tensor> zero_bias;
      const int32_t *bias_data = nullptr;
      if (bias == nullptr) {
        zero_bias.reset(
            new Tensor(GetCPUAllocator(), DT_INT32));
        zero_bias->Resize(bias_shape);
        zero_bias->Clear();
        bias_data = zero_bias->data<int32_t>();
      } else {
        bias_data = bias->data<int32_t>();
      }

      int32_t quantized_multiplier;
      int32_t right_shift;
      GetOutputMultiplierAndShift(input->scale(), filter->scale(),
                                  output->scale(), &quantized_multiplier,
                                  &right_shift);
      // 1HWO
      std::vector<index_t> filter_shape{
          1, filter->dim(0), filter->dim(1), filter->dim(2) * filter->dim(3)};

      tflite::optimized_ops::DepthwiseConv(
          input_data, ShapeToTfliteDims(input->shape()), -input->zero_point(),
          filter_data, ShapeToTfliteDims(filter_shape), -filter->zero_point(),
          bias_data, ShapeToTfliteDims(bias_shape), stride_w, stride_h,
          pad_left, pad_top, filter->dim(3), output->zero_point(),
          quantized_multiplier, right_shift, 0, 255, output_data,
          ShapeToTfliteDims(output->shape()));
    } else {
      auto bias_data = bias == nullptr ? nullptr : bias->data<int32_t>();
      float output_multiplier =
          input->scale() * filter->scale() / output->scale();
      const int pad_hw[2] = {pad_top, pad_left};
      DepthwiseConv2dGeneral(
          input_data, filter_data, bias_data, input->shape().data(),
          output_shape.data(), filter->shape().data(), input->zero_point(),
          filter->zero_point(), output->zero_point(), output_multiplier,
          strides_.data(), dilations_.data(), pad_hw, output_data);
    }

    return MaceStatus::MACE_SUCCESS;
  }
 private:
  void DepthwiseConv2dGeneral(const uint8_t *input,
                              const uint8_t *filter,
                              const int32_t *bias,
                              const index_t *in_shape,
                              const index_t *out_shape,
                              const index_t *filter_shape,
                              const int32_t input_zero,
                              const int32_t filter_zero,
                              const int32_t output_zero,
                              const float output_multiplier,
                              const int *stride_hw,
                              const int *dilation_hw,
                              const int *pad_hw,
                              uint8_t *output) {
#pragma omp parallel for collapse(2)
    for (index_t b = 0; b < out_shape[0]; ++b) {
      for (index_t h = 0; h < out_shape[1]; ++h) {
        for (index_t w = 0; w < out_shape[2]; ++w) {
          for (index_t m = 0; m < out_shape[3]; ++m) {
            const index_t filter_height = filter_shape[0];
            const index_t filter_width = filter_shape[1];
            const index_t in_channels = filter_shape[2];
            const index_t depth_multiplier = filter_shape[3];
            const index_t in_height = in_shape[1];
            const index_t in_width = in_shape[2];
            const index_t out_height = out_shape[1];
            const index_t out_width = out_shape[2];
            const index_t out_channels = out_shape[3];
            index_t out_offset =
                ((b * out_height + h) * out_width + w) * out_channels + m;
            index_t c = m / depth_multiplier;
            index_t o = m % depth_multiplier;
            index_t ih_base = h * stride_hw[0] - pad_hw[0];
            index_t iw_base = w * stride_hw[1] - pad_hw[1];
            int32_t sum = 0;
            for (index_t kh = 0; kh < filter_height; ++kh) {
              const index_t ih = ih_base + kh * dilation_hw[0];
              for (index_t kw = 0; kw < filter_width; ++kw) {
                const index_t iw = iw_base + kw * dilation_hw[1];
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                  index_t in_offset =
                      ((b * in_height + ih) * in_width + iw) * in_channels + c;
                  index_t filter_offset =
                      ((kh * filter_width + kw) * in_channels + c)
                          * depth_multiplier + o;

                  sum += (input[in_offset] - input_zero) *
                      (filter[filter_offset] - filter_zero);
                }
              }
            }
            if (bias) {
              sum += bias[m];
            }
            sum = static_cast<int32_t>(std::round(sum * output_multiplier));
            sum += output_zero;
            output[out_offset] =
                static_cast<uint8_t>(std::min(255, std::max(0, sum)));
          }
        }
      }
    }
  }

  inline tflite::Dims<4> ShapeToTfliteDims(const std::vector<index_t> &shape) {
    tflite::Dims<4> d;
    for (int i = 0; i < 4; ++i) {
      int src = static_cast<int>(shape.size() - i - 1);
      if (src >= 0) {
        d.sizes[i] = shape[src];
      } else {
        d.sizes[i] = 1;
      }
    }
    d.strides[0] = 1;
    for (int i = 1; i < 4; i++) {
      d.strides[i] = d.strides[i - 1] * d.sizes[i - 1];
    }
    return d;
  }

 protected:
  MACE_OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class DepthwiseConv2dOp<DeviceType::GPU, T> : public DepthwiseConv2dOpBase {
 public:
  explicit DepthwiseConv2dOp(OpConstructContext *context)
      : DepthwiseConv2dOpBase(context) {
    MemoryType mem_type;
    if (context->device()->gpu_runtime()->UseImageMemory()) {
      mem_type = MemoryType::GPU_IMAGE;
      kernel_.reset(new opencl::image::DepthwiseConv2dKernel<T>);
    } else {
      mem_type = MemoryType::GPU_BUFFER;
      kernel_.reset(new opencl::buffer::DepthwiseConv2dKernel<T>);
    }
    context->set_output_mem_type(mem_type);
    // Transform filter tensor to target format
    MACE_CHECK(TransformFilter<T>(
        context,
        operator_def_.get(),
        1,
        OpenCLBufferType::DW_CONV2D_FILTER,
        mem_type) == MaceStatus::MACE_SUCCESS);
    if (operator_def_->input_size() > 2) {
      MACE_CHECK(TransformFilter<T>(
          context, operator_def_.get(), 2, OpenCLBufferType::ARGUMENT, mem_type)
                     == MaceStatus::MACE_SUCCESS);
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = nullptr;
    if (this->InputSize() >= 3) {
      bias = this->Input(BIAS);
    }
    Tensor *output = this->Output(OUTPUT);
    return kernel_->Compute(context, input, filter, bias,
                            strides_.data(), padding_type_, paddings_,
                            dilations_.data(), activation_, relux_max_limit_,
                            leakyrelu_coefficient_, output);
  }

 private:
  std::unique_ptr<OpenCLDepthwiseConv2dKernel> kernel_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
#endif  // MACE_ENABLE_OPENCL


void RegisterDepthwiseConv2d(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "DepthwiseConv2d",
                   DepthwiseConv2dOp, DeviceType::CPU, float);

#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_OP(op_registry, "DepthwiseConv2d",
                   DepthwiseConv2dOp, DeviceType::CPU, uint8_t);
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "DepthwiseConv2d",
                   DepthwiseConv2dOp, DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "DepthwiseConv2d",
                   DepthwiseConv2dOp, DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
