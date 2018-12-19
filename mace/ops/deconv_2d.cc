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
#include "mace/core/tensor.h"
#include "mace/ops/activation.h"
#include "mace/ops/arm/deconv_2d_neon.h"
#include "mace/utils/utils.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/deconv_2d.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

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

    std::vector<int> in_paddings(2, 0);
    std::vector<int> out_paddings(2, 0);
    std::vector<index_t> out_shape(4, 0);
    std::vector<index_t> padded_out_shape(4, 0);

    if (model_type_ == FrameworkType::TENSORFLOW) {  // tensorflow
      MACE_CHECK_NOTNULL(output_shape_tensor);
      MACE_CHECK(output_shape_tensor->size() == 4);
      Tensor::MappingGuard output_shape_mapper(output_shape_tensor);
      auto output_shape_data =
          output_shape_tensor->data<int32_t>();
      out_shape =
          std::vector<index_t>(output_shape_data, output_shape_data + 4);

      const index_t t = out_shape[1];
      out_shape[1] = out_shape[3];
      out_shape[3] = out_shape[2];
      out_shape[2] = t;

      CalcDeconvShape_TF(
          input->shape().data(),
          filter->shape().data(),
          out_shape.data(),
          strides_.data(),
          1,
          padding_type_,
          in_paddings.data(),
          out_paddings.data(),
          padded_out_shape.data(),
          true);
    } else {  // caffe
      if (!paddings_.empty()) out_paddings = paddings_;
      CalcDeconvShape_Caffe(
          input->shape().data(),
          filter->shape().data(),
          strides_.data(),
          out_paddings.data(),
          1,
          in_paddings.data(),
          out_shape.data(),
          padded_out_shape.data(),
          true);
    }
    MACE_RETURN_IF_ERROR(output->Resize(out_shape));
    output->Clear();
    index_t kernel_h = filter->dim(2);
    index_t kernel_w = filter->dim(3);
    const index_t *in_shape = input->shape().data();

    MACE_CHECK(filter->dim(0) == out_shape[1], filter->dim(0), " != ",
               out_shape[1]);
    MACE_CHECK(filter->dim(1) == in_shape[1], filter->dim(1), " != ",
               in_shape[1]);
    MACE_CHECK(in_shape[0] == out_shape[0],
               "Input/Output batch size mismatch");
    std::function<void(const float *input,
                       const float *filter,
                       const index_t *in_shape,
                       const index_t *output_shape,
                       float *output)> deconv_func;

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard filter_mapper(filter);
    Tensor::MappingGuard bias_mapper(bias);
    Tensor::MappingGuard output_mapper(output);
    auto input_data = input->data<float>();
    auto filter_data = filter->data<float>();
    auto bias_data = bias == nullptr ? nullptr : bias->data<float>();
    auto output_data = output->mutable_data<float>();

    const index_t pad_h = out_paddings[0] / 2;
    const index_t pad_w = out_paddings[1] / 2;

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

    bool use_neon_2x2_s1 = kernel_h == kernel_w && kernel_h == 2 &&
        strides_[0] == strides_[1] && strides_[0] == 1;
    bool use_neon_2x2_s2 = kernel_h == kernel_w && kernel_h == 2 &&
        strides_[0] == strides_[1] && strides_[0] == 2;

    bool use_neon_3x3_s1 = kernel_h == kernel_w && kernel_h == 3 &&
        strides_[0] == strides_[1] && strides_[0] == 1;
    bool use_neon_3x3_s2 = kernel_h == kernel_w && kernel_h == 3 &&
        strides_[0] == strides_[1] && strides_[0] == 2;

    bool use_neon_4x4_s1 = kernel_h == kernel_w && kernel_h == 4 &&
        strides_[0] == strides_[1] && strides_[0] == 1;
    bool use_neon_4x4_s2 = kernel_h == kernel_w && kernel_h == 4 &&
        strides_[0] == strides_[1] && strides_[0] == 2;

    if (use_neon_2x2_s1) {
      deconv_func = [=](const float *input,
                        const float *filter,
                        const index_t *input_shape,
                        const index_t *padded_output_shape,
                        float *padded_output) {
        Deconv2dNeonK2x2S1(input,
                           filter,
                           input_shape,
                           padded_output_shape,
                           padded_output);
      };
    } else if (use_neon_2x2_s2) {
      deconv_func = [=](const float *input,
                        const float *filter,
                        const index_t *input_shape,
                        const index_t *padded_output_shape,
                        float *padded_output) {
        Deconv2dNeonK2x2S2(input,
                           filter,
                           input_shape,
                           padded_output_shape,
                           padded_output);
      };
    } else if (use_neon_3x3_s1) {
      deconv_func = [=](const float *input,
                        const float *filter,
                        const index_t *input_shape,
                        const index_t *padded_output_shape,
                        float *padded_output) {
        Deconv2dNeonK3x3S1(input,
                           filter,
                           input_shape,
                           padded_output_shape,
                           padded_output);
      };
    } else if (use_neon_3x3_s2) {
      deconv_func = [=](const float *input,
                        const float *filter,
                        const index_t *input_shape,
                        const index_t *padded_output_shape,
                        float *padded_output) {
        Deconv2dNeonK3x3S2(input,
                           filter,
                           input_shape,
                           padded_output_shape,
                           padded_output);
      };
    } else if (use_neon_4x4_s1) {
      deconv_func = [=](const float *input,
                        const float *filter,
                        const index_t *input_shape,
                        const index_t *padded_output_shape,
                        float *padded_output) {
        Deconv2dNeonK4x4S1(input,
                           filter,
                           input_shape,
                           padded_output_shape,
                           padded_output);
      };
    } else if (use_neon_4x4_s2) {
      deconv_func = [=](const float *input,
                        const float *filter,
                        const index_t *input_shape,
                        const index_t *padded_output_shape,
                        float *padded_output) {
        Deconv2dNeonK4x4S2(input,
                           filter,
                           input_shape,
                           padded_output_shape,
                           padded_output);
      };
    } else {
      deconv_func = [=](const float *input,
                        const float *filter,
                        const index_t *input_shape,
                        const index_t *padded_output_shape,
                        float *padded_output) {
        Deconv2dGeneral(input,
                        filter,
                        kernel_h,
                        kernel_w,
                        strides_.data(),
                        input_shape,
                        padded_output_shape,
                        padded_output);
      };
    }

    bool no_pad =
        (padded_out_shape[2] == out_shape[2]) &&
            (padded_out_shape[3] == out_shape[3]);
    float *out_data = no_pad ? output_data : padded_out_data;

    deconv_func(input_data,
                filter_data,
                in_shape,
                padded_out_shape.data(),
                out_data);
    if (!no_pad) {
      CropPadOut<float>(out_data,
                        padded_out_shape.data(),
                        out_shape.data(),
                        pad_h,
                        pad_w,
                        output_data);
    }

    if (bias_data != nullptr) {
      const index_t batch = out_shape[0];
      const index_t channels = out_shape[1];
      const index_t img_size = out_shape[2] * out_shape[3];
#pragma omp parallel for collapse(3) schedule(runtime)
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
                        relux_max_limit_,
                        leakyrelu_coefficient_);

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

#pragma omp parallel for collapse(2) schedule(runtime)
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
    MemoryType mem_type = MemoryType::GPU_IMAGE;
    if (context->device()->gpu_runtime()->UseImageMemory()) {
      kernel_.reset(new opencl::image::Deconv2dKernel<T>);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    MACE_CHECK(TransformFilter<T>(
        context, operator_def_.get(), 1,
        OpenCLBufferType::CONV2D_FILTER, mem_type)
                   == MaceStatus::MACE_SUCCESS);
    if (model_type_ == FrameworkType::CAFFE) {
      if (operator_def_->input_size() >= 3) {
        MACE_CHECK(TransformFilter<T>(
            context, operator_def_.get(), 2,
            OpenCLBufferType::ARGUMENT, mem_type) == MaceStatus::MACE_SUCCESS);
      }
    } else {
      if (operator_def_->input_size() >= 4) {
        MACE_CHECK(TransformFilter<T>(
            context,
            operator_def_.get(),
            3,
            OpenCLBufferType::ARGUMENT,
            mem_type) == MaceStatus::MACE_SUCCESS);
      }
      context->SetInputInfo(2, MemoryType::CPU_BUFFER, DataType::DT_INT32);
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

    std::vector<int> in_paddings(2, 0);
    std::vector<index_t> out_shape(4, 0);

    if (model_type_ == FrameworkType::TENSORFLOW) {
      MACE_CHECK_NOTNULL(output_shape_tensor);
      MACE_CHECK(output_shape_tensor->size() == 4);
      Tensor::MappingGuard output_shape_mapper(output_shape_tensor);
      auto output_shape_data =
          output_shape_tensor->data<int32_t>();
      out_shape =
          std::vector<index_t>(output_shape_data, output_shape_data + 4);

      CalcDeconvShape_TF(
          input->shape().data(),
          filter->shape().data(),
          out_shape.data(),
          strides_.data(),
          1,
          padding_type_,
          in_paddings.data(),
          nullptr,
          nullptr);
    } else {
      std::vector<int> out_paddings(2, 0);
      if (!paddings_.empty()) out_paddings = paddings_;
      CalcDeconvShape_Caffe(
          input->shape().data(),
          filter->shape().data(),
          strides_.data(),
          out_paddings.data(),
          1,
          in_paddings.data(),
          out_shape.data(),
          nullptr);
    }

    return kernel_->Compute(context, input, filter, bias,
                            strides_.data(), in_paddings.data(), activation_,
                            relux_max_limit_, leakyrelu_coefficient_,
                            out_shape, output);
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
