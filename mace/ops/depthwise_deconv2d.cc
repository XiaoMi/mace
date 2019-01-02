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
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/ops/arm/depthwise_deconv2d_neon.h"
#include "mace/utils/utils.h"
#include "mace/public/mace.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/depthwise_deconv2d.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template <DeviceType D, class T>
class DepthwiseDeconv2dOp;

template<>
class DepthwiseDeconv2dOp<DeviceType::CPU, float>
    : public Deconv2dOpBase {
 public:
  explicit DepthwiseDeconv2dOp(OpConstructContext *context)
      : Deconv2dOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *filter = this->Input(1);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(2) : nullptr;
    Tensor *output = this->Output(0);

    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

    std::vector<int> out_paddings(2, 0);
    std::vector<index_t> out_shape(4, 0);
    std::vector<index_t> padded_out_shape(4, 0);

    if (!paddings_.empty()) out_paddings = paddings_;
    CalcDeconvShape_Caffe(
        input->shape().data(),
        filter->shape().data(),
        strides_.data(),
        out_paddings.data(),
        group_,
        nullptr,
        out_shape.data(),
        padded_out_shape.data(),
        true);

    MACE_RETURN_IF_ERROR(output->Resize(out_shape));
    output->Clear();
    index_t kernel_h = filter->dim(2);
    index_t kernel_w = filter->dim(3);

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard filter_mapper(filter);
    Tensor::MappingGuard bias_mapper(bias);
    Tensor::MappingGuard output_mapper(output);
    auto input_data = input->data<float>();
    auto filter_data = filter->data<float>();
    auto bias_data = bias == nullptr ? nullptr : bias->data<float>();

    auto output_data = output->mutable_data<float>();

    const index_t pad_left = out_paddings[0] / 2;
    const index_t pad_top = out_paddings[1] / 2;

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

    const index_t in_channels = input->dim(1);
    const index_t out_channels = output->dim(1);

    bool no_pad = paddings_[0] == 0 && paddings_[1] == 0;
    float *out_data = no_pad ? output_data : padded_out_data;

    bool use_neon_3x3_s1 = kernel_h == kernel_w && kernel_h == 3 &&
        strides_[0] == strides_[1] && strides_[0] == 1;
    bool use_neon_3x3_s2 = kernel_h == kernel_w && kernel_h == 3 &&
        strides_[0] == strides_[1] && strides_[0] == 2;
    bool use_neon_4x4_s1 = kernel_h == kernel_w && kernel_h == 4 &&
        strides_[0] == strides_[1] && strides_[0] == 1;
    bool use_neon_4x4_s2 = kernel_h == kernel_w && kernel_h == 4 &&
        strides_[0] == strides_[1] && strides_[0] == 2;

    bool is_depthwise = (group_ == in_channels && group_ == out_channels);

    std::function<void(const float *input,
                       const float *filter,
                       const int group,
                       const index_t *in_shape,
                       const index_t *out_shape,
                       float *output)> kernel_func;

    if (use_neon_3x3_s1) {
      kernel_func = [=](const float *input,
                        const float *filter,
                        const int group,
                        const index_t *in_shape,
                        const index_t *padded_out_shape,
                        float *padded_output) {
        if (is_depthwise) {
          DepthwiseDeconv2dNeonK3x3S1(input,
                                      filter,
                                      in_shape,
                                      padded_out_shape,
                                      padded_output);
        } else {
          GroupDeconv2dNeonK3x3S1(input,
                                  filter,
                                  group,
                                  in_shape,
                                  padded_out_shape,
                                  padded_output);
        }
      };
    } else if (use_neon_3x3_s2) {
      kernel_func = [=](const float *input,
                        const float *filter,
                        const int group,
                        const index_t *in_shape,
                        const index_t *padded_out_shape,
                        float *padded_output) {
        if (is_depthwise) {
          DepthwiseDeconv2dNeonK3x3S2(input,
                                      filter,
                                      in_shape,
                                      padded_out_shape,
                                      padded_output);
        } else {
          GroupDeconv2dNeonK3x3S2(input,
                                  filter,
                                  group,
                                  in_shape,
                                  padded_out_shape,
                                  padded_output);
        }
      };
    } else if (use_neon_4x4_s1) {
      kernel_func = [=](const float *input,
                        const float *filter,
                        const int group,
                        const index_t *in_shape,
                        const index_t *padded_out_shape,
                        float *padded_output) {
        if (is_depthwise) {
          DepthwiseDeconv2dNeonK4x4S1(input,
                                      filter,
                                      in_shape,
                                      padded_out_shape,
                                      padded_output);
        } else {
          GroupDeconv2dNeonK4x4S1(input,
                                  filter,
                                  group,
                                  in_shape,
                                  padded_out_shape,
                                  padded_output);
        }
      };
    } else if (use_neon_4x4_s2) {
      kernel_func = [=](const float *input,
                        const float *filter,
                        const int group,
                        const index_t *in_shape,
                        const index_t *padded_out_shape,
                        float *padded_output) {
        if (is_depthwise) {
          DepthwiseDeconv2dNeonK4x4S2(input,
                                      filter,
                                      in_shape,
                                      padded_out_shape,
                                      padded_output);
        } else {
          GroupDeconv2dNeonK4x4S2(input,
                                  filter,
                                  group,
                                  in_shape,
                                  padded_out_shape,
                                  padded_output);
        }
      };
    } else {
      kernel_func = [=](const float *input,
                        const float *filter,
                        const int group,
                        const index_t *in_shape,
                        const index_t *padded_out_shape,
                        float *padded_output) {
        if (is_depthwise) {
          DepthwiseDeconv2dGeneral(input,
                                   filter,
                                   kernel_h,
                                   kernel_w,
                                   strides_.data(),
                                   in_shape,
                                   padded_out_shape,
                                   padded_output);
        } else {
          GroupDeconv2dGeneral(input,
                               filter,
                               kernel_h,
                               kernel_w,
                               strides_.data(),
                               group,
                               in_shape,
                               padded_out_shape,
                               padded_output);
        }
      };
    }

    kernel_func(input_data,
                filter_data,
                group_,
                input->shape().data(),
                padded_out_shape.data(),
                out_data);


    if (!no_pad) {
      CropPadOut<float>(out_data,
                        padded_out_shape.data(),
                        out_shape.data(),
                        pad_left,
                        pad_top,
                        output_data);
    }

    if (bias_data != nullptr) {
      const index_t batch = out_shape[0];
      const index_t channels = out_shape[1];
      const index_t img_size = out_shape[2] * out_shape[3];
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
                        relux_max_limit_,
                        leakyrelu_coefficient_);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  void DepthwiseDeconv2dGeneral(const float *input,
                                const float *filter,
                                const index_t kernel_h,
                                const index_t kernel_w,
                                const int *strides,
                                const index_t *in_shape,
                                const index_t *out_shape,
                                float *output) {
    const index_t batch = in_shape[0];
    const index_t out_height = out_shape[2];
    const index_t out_width = out_shape[3];

    const index_t channels = in_shape[1];
    const index_t in_height = in_shape[2];
    const index_t in_width = in_shape[3];

    const index_t out_img_size = out_height * out_width;
    const index_t in_img_size = in_height * in_width;

    const int kernel_size = kernel_h * kernel_w;
    std::vector<int> index_map(kernel_size, 0);
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        index_map[i * kernel_w + j] = i * out_width + j;
      }
    }

#pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
      for (int c = 0; c < channels; ++c) {
        float *out_base =
            output + (b * channels + c) * out_img_size;
        for (int i = 0; i < in_height; ++i) {
          for (int j = 0; j < in_width; ++j) {
            const index_t out_offset =
                i * strides[0] * out_width + j * strides[1];
            const index_t input_idx =
                (b * channels + c) * in_img_size + i * in_width + j;
            const float val = input[input_idx];
            const index_t kernel_offset = c * kernel_size;
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

  void GroupDeconv2dGeneral(const float *input,
                            const float *filter,
                            const index_t kernel_h,
                            const index_t kernel_w,
                            const int *strides,
                            const int group,
                            const index_t *in_shape,
                            const index_t *out_shape,
                            float *output) {
    const index_t out_channels = out_shape[1];
    const index_t out_height = out_shape[2];
    const index_t out_width = out_shape[3];

    const index_t in_channels = in_shape[1];
    const index_t in_height = in_shape[2];
    const index_t in_width = in_shape[3];

    MACE_CHECK(in_channels % group == 0 && out_channels % group == 0,
               "invalid input/output channel and group.");

    const index_t out_img_size = out_height * out_width;
    const index_t in_img_size = in_height * in_width;

    const int kernel_size = kernel_h * kernel_w;
    std::vector<int> index_map(kernel_size, 0);
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        index_map[i * kernel_w + j] = i * out_width + j;
      }
    }

    const int in_channels_g = in_channels / group;
    const int out_channels_g = out_channels / group;
#pragma omp parallel for collapse(3)
    for (int b = 0; b < in_shape[0]; ++b) {
      for (int g = 0; g < group; ++g) {
        for (int p = 0; p < out_channels_g; ++p) {
          const index_t out_base =
              ((b * group + g) * out_channels_g + p) * out_img_size;
          for (int i = 0; i < in_height; ++i) {
            for (int j = 0; j < in_width; ++j) {
              const index_t out_offset =
                  i * strides[0] * out_width + j * strides[1];
              for (int q = 0; q < in_channels_g; ++q) {
                const  index_t in_base =
                    ((b * group + g) * in_channels_g + q) * in_img_size;
                const index_t in_offset =
                    in_base + i * in_width + j;
                const float val = input[in_offset];
                const index_t k_offset =
                    ((p * group + g) * in_channels_g + q) * kernel_size;
                for (int k = 0; k < kernel_size; ++k) {
                  const index_t out_idx = out_base + out_offset + index_map[k];
                  const float w = filter[k_offset + k];
                  output[out_idx] += val * w;
                }
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
class DepthwiseDeconv2dOp<DeviceType::GPU, T> : public Deconv2dOpBase {
 public:
  explicit DepthwiseDeconv2dOp(OpConstructContext *context)
      : Deconv2dOpBase(context) {
    MemoryType mem_type = MemoryType::GPU_IMAGE;
    if (context->device()->gpu_runtime()->UseImageMemory()) {
      kernel_.reset(new opencl::image::DepthwiseDeconv2dKernel<T>);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    MACE_CHECK(TransformFilter<T>(
        context, operator_def_.get(), 1,
        OpenCLBufferType::DW_CONV2D_FILTER, mem_type)
                   == MaceStatus::MACE_SUCCESS);
    if (operator_def_->input_size() >= 3) {
      MACE_CHECK(TransformFilter<T>(
          context, operator_def_.get(), 2,
          OpenCLBufferType::ARGUMENT, mem_type) == MaceStatus::MACE_SUCCESS);
    }
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *filter = this->Input(1);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(2) : nullptr;
    Tensor *output = this->Output(0);
    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

    std::vector<int> in_paddings(2, 0);
    std::vector<int> out_paddings(2, 0);
    std::vector<index_t> out_shape(4, 0);

    if (!paddings_.empty()) out_paddings = paddings_;
    CalcDeconvShape_Caffe(input->shape().data(),
                          filter->shape().data(),
                          strides_.data(),
                          out_paddings.data(),
                          group_,
                          in_paddings.data(),
                          out_shape.data(),
                          nullptr);

    return kernel_->Compute(context,
                            input,
                            filter,
                            bias,
                            strides_.data(),
                            in_paddings.data(),
                            group_,
                            activation_,
                            relux_max_limit_,
                            leakyrelu_coefficient_,
                            out_shape,
                            output);
  }

 private:
  std::unique_ptr<OpenCLDepthwiseDeconv2dKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterDepthwiseDeconv2d(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "DepthwiseDeconv2d",
                   DepthwiseDeconv2dOp, DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "DepthwiseDeconv2d",
                   DepthwiseDeconv2dOp, DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "DepthwiseDeconv2d",
                   DepthwiseDeconv2dOp, DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
