// Copyright 2018 The MACE Authors. All Rights Reserved.
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
#include "mace/ops/arm/fp32/deconv_2d_2x2.h"
#include "mace/ops/arm/fp32/deconv_2d_3x3.h"
#include "mace/ops/arm/fp32/deconv_2d_4x4.h"
#include "mace/ops/arm/fp32/deconv_2d_general.h"
#include "mace/ops/arm/fp32/bias_add.h"
#include "mace/ops/arm/fp32/activation.h"
#else
#include "mace/ops/ref/bias_add.h"
#include "mace/ops/ref/activation.h"
#include "mace/ops/ref/deconv_2d.h"
#endif

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/ops/activation.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/utils/memory.h"
#include "mace/utils/math.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/deconv_2d.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template<DeviceType D, class T>
class Deconv2dOp;

template<>
class Deconv2dOp<DeviceType::CPU, float> : public Deconv2dOpBase {
 public:
  explicit Deconv2dOp(OpConstructContext *context)
      : Deconv2dOpBase(context),
        activation_delegator_(activation_,
                              relux_max_limit_,
                              leakyrelu_coefficient_) {}

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *filter = this->Input(1);
    const Tensor *bias = nullptr;
    const Tensor *output_shape_tensor = nullptr;
    if (model_type_ == CAFFE) {
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

#ifdef MACE_ENABLE_NEON
    const index_t kernel_h = filter->dim(2);
    const index_t kernel_w = filter->dim(3);

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

    if (deconv2d_delegator_ == nullptr) {
      if (use_neon_2x2_s1) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK2x2S1>(
            paddings_, padding_type_, model_type_);
      } else if (use_neon_2x2_s2) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK2x2S2>(
            paddings_, padding_type_, model_type_);
      } else if (use_neon_3x3_s1) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK3x3S1>(
            paddings_, padding_type_, model_type_);
      } else if (use_neon_3x3_s2) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK3x3S2>(
            paddings_, padding_type_, model_type_);
      } else if (use_neon_4x4_s1) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK4x4S1>(
            paddings_, padding_type_, model_type_);
      } else if (use_neon_4x4_s2) {
        deconv2d_delegator_ = make_unique<arm::fp32::Deconv2dK4x4S2>(
            paddings_, padding_type_, model_type_);
      } else {
        deconv2d_delegator_ =
            make_unique<arm::fp32::Deconv2dGeneral>(strides_,
                                                    std::vector<int>{1, 1},
                                                    paddings_,
                                                    padding_type_,
                                                    model_type_);
      }
    }
    deconv2d_delegator_->Compute(context,
                                 input,
                                 filter,
                                 output_shape_tensor,
                                 output);
#else
    if (deconv2d_delegator_ == nullptr) {
      deconv2d_delegator_ = make_unique<ref::Deconv2d<float>>(strides_,
                                                              std::vector<int>{
                                                                  1, 1},
                                                              paddings_,
                                                              padding_type_,
                                                              model_type_);
    }
    deconv2d_delegator_->Compute(context,
                                 input,
                                 filter,
                                 output_shape_tensor,
                                 output);

#endif  // MACE_ENABLE_NEON

    bias_add_delegator_.Compute(context, output, bias, output);
    activation_delegator_.Compute(context, output, output);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
#ifdef MACE_ENABLE_NEON
  std::unique_ptr<arm::fp32::Deconv2dBase> deconv2d_delegator_;
  arm::fp32::BiasAdd bias_add_delegator_;
  arm::fp32::Activation activation_delegator_;
#else
  ref::BiasAdd bias_add_delegator_;
  ref::Activation activation_delegator_;
  std::unique_ptr<ref::Deconv2d<float>> deconv2d_delegator_;
#endif  // MACE_ENABLE_NEON
};

#ifdef MACE_ENABLE_OPENCL
template<typename T>
class Deconv2dOp<DeviceType::GPU, T> : public Deconv2dOpBase {
 public:
  explicit Deconv2dOp(OpConstructContext *context)
      : Deconv2dOpBase(context) {
    MemoryType mem_type = MemoryType::GPU_IMAGE;
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::Deconv2dKernel<T>>();
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
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *filter = this->Input(1);
    const Tensor *bias = nullptr;
    const Tensor *output_shape_tensor = nullptr;
    if (model_type_ == CAFFE) {
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

    std::vector<index_t> out_shape;
    if (output_shape_tensor) {
      Tensor::MappingGuard out_shape_guard(output_shape_tensor);
      MACE_CHECK(output_shape_tensor->size() == 4,
                 "output shape should be 4-dims");
      out_shape =
          std::vector<index_t>(output_shape_tensor->data<int32_t>(),
                               output_shape_tensor->data<int32_t>() + 4);
    }
    std::vector<int> in_paddings;
    std::vector<int> out_paddings;

    CalDeconvOutputShapeAndPadSize(input->shape(),
                                   filter->shape(),
                                   strides_,
                                   padding_type_,
                                   paddings_,
                                   1,
                                   &out_shape,
                                   &in_paddings,
                                   &out_paddings,
                                   nullptr,
                                   model_type_,
                                   DataFormat::NHWC);

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
  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("Deconv2D")
          .SetInputMemoryTypeSetter(
              [](OpConditionContext *context) -> void {
                MemoryType mem_type = MemoryType::CPU_BUFFER;
                if (context->device()->device_type() == DeviceType::GPU) {
                  if (context->device()->gpu_runtime()->UseImageMemory()) {
                    mem_type = MemoryType::GPU_IMAGE;
                  } else {
                    MACE_NOT_IMPLEMENTED;
                  }
                  FrameworkType framework_type =
                      static_cast<FrameworkType>(
                        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                            *(context->operator_def()), "framework_type",
                            FrameworkType::TENSORFLOW));
                  if (framework_type == FrameworkType::TENSORFLOW) {
                    context->SetInputInfo(2, MemoryType::CPU_BUFFER,
                                          DataType::DT_INT32);
                  }
                }
                context->set_output_mem_type(mem_type);
              }));
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
