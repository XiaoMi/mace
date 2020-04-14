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

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/core/tensor.h"
#include "mace/ops/activation.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/delegator/activation.h"
#include "mace/ops/delegator/bias_add.h"
#include "mace/ops/delegator/deconv_2d.h"
#include "mace/utils/memory.h"
#include "mace/utils/math.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/deconv_2d.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

namespace {
const std::vector<int> kDeconv2dStrides = {1, 1};
}

template<DeviceType D, class T>
class Deconv2dOp;

template<class T>
class Deconv2dOp<DeviceType::CPU, T> : public Deconv2dOpBase {
 public:
  explicit Deconv2dOp(OpConstructContext *context)
      : Deconv2dOpBase(context),
        activation_delegator_(
            delegator::Activation::Create(
                context->workspace(),
                MACE_DELEGATOR_KEY(Activation, DeviceType::CPU,
                                   T, kCpuImplType),
                delegator::ActivationParam(activation_, relux_max_limit_,
                                           leakyrelu_coefficient_))),
        bias_add_delegator_(delegator::BiasAdd::Create(
            context->workspace(),
            MACE_DELEGATOR_KEY(BiasAdd, DeviceType::CPU, T, kCpuImplType),
            DelegatorParam())) {}

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *filter = this->Input(1);
    const Tensor *bias = nullptr;
    const Tensor *output_shape_tensor = nullptr;
    if (model_type_ == TENSORFLOW) {
      output_shape_tensor =
          this->InputSize() >= 3 ? this->Input(2) : nullptr;
      bias = this->InputSize() >= 4 ? this->Input(3) : nullptr;
    } else {
      bias = this->InputSize() >= 3 ? this->Input(2) : nullptr;
    }
    Tensor *output = this->Output(0);

    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

    if (deconv2d_delegator_ == nullptr) {
      auto tag = MACE_DELEGATOR_KEY(Deconv2d, DeviceType::CPU, T, kCpuImplType);
      if (kCpuImplType == NEON) {
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

        if (use_neon_2x2_s1) {
          tag = MACE_DELEGATOR_KEY_EX(Deconv2d, DeviceType::CPU, T,
                                      kCpuImplType, K2x2S1);
        } else if (use_neon_2x2_s2) {
          tag = MACE_DELEGATOR_KEY_EX(Deconv2d, DeviceType::CPU, T,
                                      kCpuImplType, K2x2S2);
        } else if (use_neon_3x3_s1) {
          tag = MACE_DELEGATOR_KEY_EX(Deconv2d, DeviceType::CPU, T,
                                      kCpuImplType, K3x3S1);
        } else if (use_neon_3x3_s2) {
          tag = MACE_DELEGATOR_KEY_EX(Deconv2d, DeviceType::CPU, T,
                                      kCpuImplType, K3x3S2);
        } else if (use_neon_4x4_s1) {
          tag = MACE_DELEGATOR_KEY_EX(Deconv2d, DeviceType::CPU, T,
                                      kCpuImplType, K4x4S1);
        } else if (use_neon_4x4_s2) {
          tag = MACE_DELEGATOR_KEY_EX(Deconv2d, DeviceType::CPU, T,
                                      kCpuImplType, K4x4S2);
        }
      }
      delegator::Deconv2dParam param(strides_, kDeconv2dStrides, paddings_,
                                     padding_type_, model_type_);
      deconv2d_delegator_ = delegator::Deconv2d::Create(context->workspace(),
                                                        tag, param);
    }

    deconv2d_delegator_->Compute(context, input, filter,
                                 output_shape_tensor, output);
    bias_add_delegator_->Compute(context, output, bias, output);
    activation_delegator_->Compute(context, output, output);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  std::unique_ptr<delegator::Activation> activation_delegator_;
  std::unique_ptr<delegator::BiasAdd> bias_add_delegator_;
  std::unique_ptr<delegator::Deconv2d> deconv2d_delegator_;
};

#ifdef MACE_ENABLE_OPENCL
template<>
class Deconv2dOp<DeviceType::GPU, float> : public Deconv2dOpBase {
 public:
  explicit Deconv2dOp(OpConstructContext *context) : Deconv2dOpBase(context),
      dim_(Operation::GetRepeatedArgs<index_t>("dim")) {
    MemoryType mem_type = MemoryType::GPU_IMAGE;
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::Deconv2dKernel>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    MACE_CHECK(TransformFilter(
        context, operator_def_.get(), 1,
        OpenCLBufferType::CONV2D_FILTER, mem_type)
                   == MaceStatus::MACE_SUCCESS);
    if (model_type_ == FrameworkType::TENSORFLOW) {
      if (operator_def_->input_size() >= 4) {
        MACE_CHECK(TransformFilter(
            context,
            operator_def_.get(),
            3,
            OpenCLBufferType::ARGUMENT,
            mem_type) == MaceStatus::MACE_SUCCESS);
      }
    } else {
      if (operator_def_->input_size() >= 3) {
        MACE_CHECK(TransformFilter(
            context, operator_def_.get(), 2,
            OpenCLBufferType::ARGUMENT, mem_type) == MaceStatus::MACE_SUCCESS);
      }
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *filter = this->Input(1);
    const Tensor *bias = nullptr;
    const Tensor *output_shape_tensor = nullptr;
    if (model_type_ == TENSORFLOW) {
      output_shape_tensor =
          this->InputSize() >= 3 ? this->Input(2) : nullptr;
      bias = this->InputSize() >= 4 ? this->Input(3) : nullptr;
    } else {
      bias = this->InputSize() >= 3 ? this->Input(2) : nullptr;
    }
    Tensor *output = this->Output(0);

    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

    std::vector<index_t> out_shape;
    if (output_shape_tensor) {
      if (dim_.size() < 2) {
        Tensor::MappingGuard out_shape_guard(output_shape_tensor);
        MACE_CHECK(output_shape_tensor->size() == 4,
                   "output shape should be 4-dims");
        out_shape =
            std::vector<index_t>(output_shape_tensor->data<int32_t>(),
                                 output_shape_tensor->data<int32_t>() + 4);
      } else {
        out_shape = dim_;
      }
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
  std::vector<index_t> dim_;
  std::unique_ptr<OpenCLDeconv2dKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterDeconv2D(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "Deconv2D", Deconv2dOp, DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "Deconv2D", Deconv2dOp, DeviceType::CPU);
  MACE_REGISTER_GPU_OP(op_registry, "Deconv2D", Deconv2dOp);
#ifdef MACE_ENABLE_OPENCL
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
                  context->set_output_mem_type(mem_type);
                  FrameworkType framework_type =
                      static_cast<FrameworkType>(
                        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                            *(context->operator_def()), "framework_type",
                            FrameworkType::TENSORFLOW));
                  if (framework_type == FrameworkType::TENSORFLOW) {
                    context->SetInputInfo(2, MemoryType::CPU_BUFFER,
                                          DataType::DT_INT32);
                  }
                } else {
                  context->set_output_mem_type(mem_type);
                }
              }));
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
