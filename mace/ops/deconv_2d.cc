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

#include "mace/ops/deconv_2d.h"

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/operator.h"
#include "mace/core/tensor.h"
#include "mace/ops/activation.h"
#include "mace/ops/arm/deconv_2d_neon.h"
#include "mace/ops/conv_pool_2d_util.h"
#include "mace/utils/utils.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/deconv_2d.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

class Deconv2dOpBase : public Operation {
 public:
  explicit Deconv2dOpBase(OpConstructContext *context)
      : Operation(context),
        strides_(Operation::GetRepeatedArgs<int>("strides")),
        padding_type_(static_cast<Padding>(Operation::GetOptionalArg<int>(
            "padding", static_cast<int>(SAME)))),
        paddings_(Operation::GetRepeatedArgs<int>("padding_values")),
        model_type_(static_cast<ops::FrameworkType>(
                        Operation::GetOptionalArg<int>("framework_type", 0))),
        activation_(ops::StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation",
                                                  "NOOP"))),
        relux_max_limit_(Operation::GetOptionalArg<float>("max_limit", 0.0f)) {}


  static void CalcDeconvOutputSize(
      const index_t *input_shape,   // NHWC
      const index_t *filter_shape,  // OIHW
      const int *strides,
      index_t *output_shape,
      const int *padding_size,
      int *input_padding,
      const bool isNCHW = false) {
    MACE_CHECK_NOTNULL(output_shape);
    MACE_CHECK_NOTNULL(padding_size);
    MACE_CHECK_NOTNULL(input_shape);
    MACE_CHECK_NOTNULL(filter_shape);
    MACE_CHECK_NOTNULL(strides);

    const index_t output_channel = filter_shape[0];

    const index_t in_height = isNCHW ? input_shape[2] : input_shape[1];
    const index_t in_width = isNCHW ? input_shape[3] : input_shape[2];

    const index_t kernel_h = filter_shape[2];
    const index_t kernel_w = filter_shape[3];

    input_padding[0] = static_cast<int>((kernel_h -1) * 2 - padding_size[0]);
    input_padding[1] = static_cast<int>((kernel_w -1) * 2 - padding_size[1]);
    input_padding[0] = std::max<int>(0, input_padding[0]);
    input_padding[1] = std::max<int>(0, input_padding[1]);

    index_t out_height =
        (in_height - 1) * strides[0] + kernel_h - padding_size[0];
    index_t out_width =
        (in_width - 1) * strides[1] + kernel_w - padding_size[1];

    output_shape[0] = input_shape[0];
    if (isNCHW) {
      output_shape[1] = output_channel;
      output_shape[2] = out_height;
      output_shape[3] = out_width;
    } else {
      output_shape[1] = out_height;
      output_shape[2] = out_width;
      output_shape[3] = output_channel;
    }
  }

  static void CalcDeconvPaddingAndInputSize(
      const index_t *input_shape,   // NHWC
      const index_t *filter_shape,  // OIHW
      const int *strides,
      Padding padding,
      const index_t *output_shape,
      int *padding_size,
      const bool isNCHW = false) {
    MACE_CHECK_NOTNULL(output_shape);
    MACE_CHECK_NOTNULL(padding_size);
    MACE_CHECK_NOTNULL(input_shape);
    MACE_CHECK_NOTNULL(filter_shape);
    MACE_CHECK_NOTNULL(strides);

    const index_t in_height = isNCHW ? input_shape[2] : input_shape[1];
    const index_t in_width = isNCHW ? input_shape[3] : input_shape[2];

    const index_t out_height = isNCHW ? output_shape[2] : output_shape[1];
    const index_t out_width = isNCHW ? output_shape[3] : output_shape[2];

    const index_t extended_input_height = (in_height - 1) * strides[0] + 1;
    const index_t extended_input_width = (in_width - 1) * strides[1] + 1;

    const index_t filter_h = filter_shape[2];
    const index_t filter_w = filter_shape[3];

    index_t expected_input_height = 0, expected_input_width = 0;

    switch (padding) {
      case VALID:
        expected_input_height =
            (out_height - filter_h + strides[0]) / strides[0];
        expected_input_width =
            (out_width - filter_w + strides[1]) / strides[1];
        break;
      case SAME:
        expected_input_height =
            (out_height + strides[0] - 1) / strides[0];
        expected_input_width =
            (out_width + strides[1] - 1) / strides[1];
        break;
      default:
        MACE_CHECK(false, "Unsupported padding type: ", padding);
    }

    MACE_CHECK(expected_input_height == in_height,
               expected_input_height, "!=", in_height);
    MACE_CHECK(expected_input_width == in_width,
               expected_input_width, "!=", in_width);

    const int p_h = static_cast<int>(out_height +
        filter_h - 1 - extended_input_height);
    const int p_w = static_cast<int>(out_width +
        filter_w - 1 - extended_input_width);

    padding_size[0] = std::max<int>(0, p_h);
    padding_size[1] = std::max<int>(0, p_w);
  }

 protected:
  std::vector<int> strides_;  // [stride_h, stride_w]
  const Padding padding_type_;
  std::vector<int> paddings_;
  const FrameworkType model_type_;
  const ActivationType activation_;
  const float relux_max_limit_;
};

template <DeviceType D, class T>
class Deconv2dOp;

template <>
class Deconv2dOp<DeviceType::CPU, float> : public Deconv2dOpBase {
 public:
  explicit Deconv2dOp(OpConstructContext *context)
      : Deconv2dOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *filter = this->Input(1);
    const Tensor *bias = nullptr;
    const Tensor *output_shape_tensor = nullptr;
    if (model_type_ == ops::CAFFE) {
      bias = this->InputSize() >= 3 ? this->Input(2) : nullptr;
    } else {
      output_shape_tensor =
          this->InputSize() >= 3 ? this->Input(2) : nullptr;
      bias = this->InputSize() >= 4 ? this->Input(3) : nullptr;
    }
    Tensor *output = this->Output(0);

    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

    std::vector<int> paddings(2);
    std::vector<int> out_paddings(2);
    std::vector<index_t> output_shape(4);
    if (model_type_ == FrameworkType::TENSORFLOW) {  // tensorflow
      paddings = std::vector<int>(2, 0);
      MACE_CHECK_NOTNULL(output_shape_tensor);
      MACE_CHECK(output_shape_tensor->size() == 4);
      Tensor::MappingGuard output_shape_mapper(output_shape_tensor);
      auto output_shape_data =
          output_shape_tensor->data<int32_t>();
      output_shape =
          std::vector<index_t>(output_shape_data, output_shape_data + 4);

      const index_t t = output_shape[1];
      output_shape[1] = output_shape[3];
      output_shape[3] = output_shape[2];
      output_shape[2] = t;

      CalcDeconvPaddingAndInputSize(
          input->shape().data(),
          filter->shape().data(),
          strides_.data(), padding_type_,
          output_shape.data(),
          paddings.data(), true);
    } else {  // caffe
      out_paddings = paddings_;
      output_shape = std::vector<index_t>(4, 0);
      CalcDeconvOutputSize(input->shape().data(),
                           filter->shape().data(),
                           strides_.data(),
                           output_shape.data(),
                           out_paddings.data(),
                           paddings.data(),
                           true);
    }
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    index_t kernel_h = filter->dim(2);
    index_t kernel_w = filter->dim(3);
    const index_t *in_shape = input->shape().data();

    MACE_CHECK(filter->dim(0) == output_shape[1], filter->dim(0), " != ",
               output_shape[1]);
    MACE_CHECK(filter->dim(1) == in_shape[1], filter->dim(1), " != ",
               in_shape[1]);
    MACE_CHECK(in_shape[0] == output_shape[0],
               "Input/Output batch size mismatch");
    std::function<void(const float *input,
                       const float *filter,
                       const index_t *in_shape,
                       const index_t *out_shape,
                       float *output)> deconv_func;

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard filter_mapper(filter);
    Tensor::MappingGuard bias_mapper(bias);
    Tensor::MappingGuard output_mapper(output);
    auto input_data = input->data<float>();
    auto filter_data = filter->data<float>();
    auto bias_data = bias == nullptr ? nullptr : bias->data<float>();
    auto output_data = output->mutable_data<float>();

    const index_t padded_out_h = (in_shape[2] - 1) * strides_[0] + kernel_h;
    const index_t padded_out_w = (in_shape[3] - 1) * strides_[1] + kernel_w;
    const index_t pad_h = (padded_out_h - output_shape[2]) / 2;
    const index_t pad_w = (padded_out_w - output_shape[3]) / 2;

    std::vector<index_t> padded_out_shape({output_shape[0], output_shape[1],
                                           padded_out_h, padded_out_w});
    index_t padded_out_size =
        std::accumulate(padded_out_shape.begin(),
                        padded_out_shape.end(),
                        1,
                        std::multiplies<index_t>()) * sizeof(float);
    ScratchBuffer *scratch = context->device()->scratch_buffer();
    scratch->Rewind();
    scratch->GrowSize(padded_out_size);
    Tensor padded_out(scratch->Scratch(padded_out_size), DT_FLOAT);
    padded_out.Reshape(padded_out_shape);
    padded_out.Clear();
    auto *padded_out_data = padded_out.mutable_data<float>();

    bool use_neon_3x3_s1 = kernel_h == kernel_w && kernel_h == 3 &&
        strides_[0] == strides_[1] && strides_[0] == 1;
    bool use_neon_3x3_s2 = kernel_h == kernel_w && kernel_h == 3 &&
        strides_[0] == strides_[1] && strides_[0] == 2;

    bool use_neon_4x4_s1 = kernel_h == kernel_w && kernel_h == 4 &&
        strides_[0] == strides_[1] && strides_[0] == 1;
    bool use_neon_4x4_s2 = kernel_h == kernel_w && kernel_h == 4 &&
        strides_[0] == strides_[1] && strides_[0] == 2;

    if (use_neon_3x3_s1) {
      deconv_func = [=](const float *input,
                        const float *filter,
                        const index_t *in_shape,
                        const index_t *padded_out_shape,
                        float *padded_output) {
        Deconv2dNeonK3x3S1(input,
                           filter,
                           in_shape,
                           padded_out_shape,
                           padded_output);
      };
    } else if (use_neon_3x3_s2) {
      deconv_func = [=](const float *input,
                        const float *filter,
                        const index_t *in_shape,
                        const index_t *padded_out_shape,
                        float *padded_output) {
        Deconv2dNeonK3x3S2(input,
                           filter,
                           in_shape,
                           padded_out_shape,
                           padded_output);
      };
    } else if (use_neon_4x4_s1) {
      deconv_func = [=](const float *input,
                        const float *filter,
                        const index_t *in_shape,
                        const index_t *padded_out_shape,
                        float *padded_output) {
        Deconv2dNeonK4x4S1(input,
                           filter,
                           in_shape,
                           padded_out_shape,
                           padded_output);
      };
    } else if (use_neon_4x4_s2) {
      deconv_func = [=](const float *input,
                        const float *filter,
                        const index_t *in_shape,
                        const index_t *padded_out_shape,
                        float *padded_output) {
        Deconv2dNeonK4x4S2(input,
                           filter,
                           in_shape,
                           padded_out_shape,
                           padded_output);
      };
    } else {
      deconv_func = [=](const float *input,
                        const float *filter,
                        const index_t *in_shape,
                        const index_t *padded_out_shape,
                        float *padded_output) {
        Deconv2dGeneral(input,
                        filter,
                        kernel_h,
                        kernel_w,
                        strides_.data(),
                        in_shape,
                        padded_out_shape,
                        padded_output);
      };
    }

    bool no_pad =
        padded_out_h == output_shape[2] && padded_out_w == output_shape[3];
    float *out_data = no_pad ? output_data : padded_out_data;

    deconv_func(input_data,
                filter_data,
                in_shape,
                padded_out_shape.data(),
                out_data);
    if (!no_pad) {
      CropPadOut<float>(out_data,
                        padded_out_shape.data(),
                        output_shape.data(),
                        pad_h,
                        pad_w,
                        output_data);
    }

    if (bias_data != nullptr) {
      const index_t batch = output_shape[0];
      const index_t channels = output_shape[1];
      const index_t img_size = output_shape[2] * output_shape[3];
#pragma omp parallel for collapse(3)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channels; ++c) {
          for (index_t i = 0; i < img_size; ++i) {
            output_data[(b * channels + c) * img_size + i] +=
                bias_data[c];
          }
        }
      }
    }

    DoActivation<float>(output_data,
                        output_data,
                        output->size(),
                        activation_,
                        relux_max_limit_);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  void Deconv2dGeneral(const float *input,
                       const float *filter,
                       const index_t kernel_h,
                       const index_t kernel_w,
                       const int *strides,
                       const index_t *in_shape,
                       const index_t *out_shape,
                       float *output) {
    const index_t out_height = out_shape[2];
    const index_t out_width = out_shape[3];
    const index_t in_height = in_shape[2];
    const index_t in_width = in_shape[3];
    const index_t out_img_size = out_height * out_width;
    const index_t in_img_size = in_height * in_width;

    const int kernel_size = static_cast<int>(kernel_h * kernel_w);
    std::vector<index_t> index_map(kernel_size, 0);
    for (index_t i = 0; i < kernel_h; ++i) {
      for (index_t j = 0; j < kernel_w; ++j) {
        index_map[i * kernel_w + j] = i * out_width + j;
      }
    }

    const index_t batch = in_shape[0];
    const index_t out_channels = out_shape[1];
    const index_t in_channels = in_shape[1];

#pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
      for (int oc = 0; oc < out_channels; ++oc) {
        float *out_base =
            output + (b * out_channels + oc) * out_img_size;
        for (int i = 0; i < in_height; ++i) {
          for (int j = 0; j < in_width; ++j) {
            const index_t out_offset =
                i * strides[0] * out_width + j * strides[1];
            for (int ic = 0; ic < in_channels; ++ic) {
              const index_t input_idx =
                  (b * in_channels + ic) * in_img_size + i * in_width + j;
              const float val = input[input_idx];
              const index_t kernel_offset =
                  (oc * in_channels + ic) * kernel_size;
              for (int k = 0; k < kernel_size; ++k) {
                const index_t out_idx = out_offset + index_map[k];
                const index_t kernel_idx = kernel_offset + k;
                out_base[out_idx] += val * filter[kernel_idx];
              }
            }
          }
        }
      }
    }
  }
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class Deconv2dOp<DeviceType::GPU, T> : public Deconv2dOpBase {
 public:
  explicit Deconv2dOp(OpConstructContext *context)
      : Deconv2dOpBase(context) {
    if (context->device()->opencl_runtime()->UseImageMemory()) {
      kernel_.reset(new opencl::image::Deconv2dKernel<T>);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *filter = this->Input(1);
    const Tensor *bias = nullptr;
    const Tensor *output_shape_tensor = nullptr;
    if (model_type_ == ops::CAFFE) {
      bias = this->InputSize() >= 3 ? this->Input(2) : nullptr;
    } else {
      output_shape_tensor =
          this->InputSize() >= 3 ? this->Input(2) : nullptr;
      bias = this->InputSize() >= 4 ? this->Input(3) : nullptr;
    }
    Tensor *output = this->Output(0);

    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);
    std::vector<int> paddings(2);
    std::vector<int> out_paddings(2);
    std::vector<index_t> output_shape(4);
    if (model_type_ == FrameworkType::TENSORFLOW) {
      paddings = std::vector<int>(2, 0);
      MACE_CHECK_NOTNULL(output_shape_tensor);
      MACE_CHECK(output_shape_tensor->size() == 4);
      Tensor::MappingGuard output_shape_mapper(output_shape_tensor);
      auto output_shape_data =
          output_shape_tensor->data<int32_t>();
      output_shape =
          std::vector<index_t>(output_shape_data, output_shape_data + 4);
      CalcDeconvPaddingAndInputSize(input->shape().data(),
                                    filter->shape().data(),
                                    strides_.data(),
                                    padding_type_,
                                    output_shape.data(),
                                    paddings.data());
    } else {
      out_paddings = paddings_;
      paddings = std::vector<int>(2, 0);
      output_shape = std::vector<index_t>(4, 0);
      CalcDeconvOutputSize(input->shape().data(),
                           filter->shape().data(),
                           strides_.data(),
                           output_shape.data(),
                           out_paddings.data(),
                           paddings.data());
    }

    return kernel_->Compute(context, input, filter, bias,
                            strides_.data(), paddings.data(), activation_,
                            relux_max_limit_, output_shape, output);
  }

 private:
  std::unique_ptr<OpenCLDeconv2dKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL


void RegisterDeconv2D(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Deconv2D", Deconv2dOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "Deconv2D", Deconv2dOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "Deconv2D", Deconv2dOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
