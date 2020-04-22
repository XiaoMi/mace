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

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/core/tensor.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/deconv_2d.h"
#include "mace/ops/delegator/activation.h"
#include "mace/ops/delegator/bias_add.h"
#include "mace/ops/delegator/depthwise_deconv_2d.h"
#include "mace/public/mace.h"
#include "mace/utils/math.h"
#include "mace/utils/memory.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/depthwise_deconv2d.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

namespace {
const std::vector<int> kDepthwiseStrides = {1, 1};
}

template<DeviceType D, class T>
class DepthwiseDeconv2dOp;

template<class T>
class DepthwiseDeconv2dOp<DeviceType::CPU, T>
    : public Deconv2dOpBase {
 public:
  explicit DepthwiseDeconv2dOp(OpConstructContext *context)
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
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(2) : nullptr;
    Tensor *output = this->Output(0);

    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

    const index_t in_channels = input->dim(1);
    bool is_depthwise = group_ == in_channels;

    if (depthwise_deconv2d_delegator_ == nullptr) {
      if (kCpuImplType == NEON) {
        const index_t kernel_h = filter->dim(2);
        const index_t kernel_w = filter->dim(3);
        bool use_neon_3x3_s1 = kernel_h == kernel_w && kernel_h == 3 &&
            strides_[0] == strides_[1] && strides_[0] == 1;
        bool use_neon_3x3_s2 = kernel_h == kernel_w && kernel_h == 3 &&
            strides_[0] == strides_[1] && strides_[0] == 2;
        bool use_neon_4x4_s1 = kernel_h == kernel_w && kernel_h == 4 &&
            strides_[0] == strides_[1] && strides_[0] == 1;
        bool use_neon_4x4_s2 = kernel_h == kernel_w && kernel_h == 4 &&
            strides_[0] == strides_[1] && strides_[0] == 2;

        if (is_depthwise) {
          auto tag = MACE_DELEGATOR_KEY(DepthwiseDeconv2d, DeviceType::CPU, T,
                                        kCpuImplType);
          if (use_neon_3x3_s1) {
            tag = MACE_DELEGATOR_KEY_EX(DepthwiseDeconv2d, DeviceType::CPU, T,
                                        kCpuImplType, K3x3S1);
          } else if (use_neon_3x3_s2) {
            tag = MACE_DELEGATOR_KEY_EX(DepthwiseDeconv2d, DeviceType::CPU, T,
                                        kCpuImplType, K3x3S2);
          } else if (use_neon_4x4_s1) {
            tag = MACE_DELEGATOR_KEY_EX(DepthwiseDeconv2d, DeviceType::CPU, T,
                                        kCpuImplType, K4x4S1);
          } else if (use_neon_4x4_s2) {
            tag = MACE_DELEGATOR_KEY_EX(DepthwiseDeconv2d, DeviceType::CPU, T,
                                        kCpuImplType, K4x4S2);
          }
          delegator::DepthwiseDeconv2dParam param(strides_, kDepthwiseStrides,
                                                  paddings_, padding_type_,
                                                  CAFFE, group_);
          depthwise_deconv2d_delegator_ = delegator::DepthwiseDeconv2d::Create(
              context->workspace(), tag, param);
        } else {
          auto tag = MACE_DELEGATOR_KEY(GroupDeconv2d, DeviceType::CPU, T,
                                        kCpuImplType);
          if (use_neon_3x3_s1) {
            tag = MACE_DELEGATOR_KEY_EX(GroupDeconv2d, DeviceType::CPU, T,
                                        kCpuImplType, K3x3S1);
          } else if (use_neon_3x3_s2) {
            tag = MACE_DELEGATOR_KEY_EX(GroupDeconv2d, DeviceType::CPU, T,
                                        kCpuImplType, K3x3S2);
          } else if (use_neon_4x4_s1) {
            tag = MACE_DELEGATOR_KEY_EX(GroupDeconv2d, DeviceType::CPU, T,
                                        kCpuImplType, K4x4S1);
          } else if (use_neon_4x4_s2) {
            tag = MACE_DELEGATOR_KEY_EX(GroupDeconv2d, DeviceType::CPU, T,
                                        kCpuImplType, K4x4S2);
          }
          delegator::GroupDeconv2dParam param(strides_, kDepthwiseStrides,
                                              paddings_, padding_type_,
                                              CAFFE, group_);
          depthwise_deconv2d_delegator_ = delegator::GroupDeconv2d::Create(
              context->workspace(), tag, param);
        }
      }
    }

    depthwise_deconv2d_delegator_->Compute(context, input, filter,
                                           nullptr, output);
    bias_add_delegator_->Compute(context, output, bias, output);
    activation_delegator_->Compute(context, output, output);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  std::unique_ptr<delegator::Activation> activation_delegator_;
  std::unique_ptr<delegator::BiasAdd> bias_add_delegator_;
  std::unique_ptr<delegator::DepthwiseDeconv2d> depthwise_deconv2d_delegator_;
};

#ifdef MACE_ENABLE_OPENCL
template<>
class DepthwiseDeconv2dOp<DeviceType::GPU, float> : public Deconv2dOpBase {
 public:
  explicit DepthwiseDeconv2dOp(OpConstructContext *context)
      : Deconv2dOpBase(context) {
    MemoryType mem_type = MemoryType::GPU_IMAGE;
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::DepthwiseDeconv2dKernel>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    MACE_CHECK(TransformFilter(
        context, operator_def_.get(), 1,
        OpenCLBufferType::DW_CONV2D_FILTER, mem_type)
                   == MaceStatus::MACE_SUCCESS);
    if (operator_def_->input_size() >= 3) {
      MACE_CHECK(TransformFilter(
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

    std::vector<index_t> out_shape;
    std::vector<int> in_paddings;
    std::vector<int> out_paddings;

    CalDeconvOutputShapeAndPadSize(input->shape(),
                                   filter->shape(),
                                   strides_,
                                   padding_type_,
                                   paddings_,
                                   group_,
                                   &out_shape,
                                   &in_paddings,
                                   &out_paddings,
                                   nullptr,
                                   CAFFE,
                                   DataFormat::NHWC);

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

void RegisterDepthwiseDeconv2d(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "DepthwiseDeconv2d",
                   DepthwiseDeconv2dOp, DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "DepthwiseDeconv2d",
                        DepthwiseDeconv2dOp, DeviceType::CPU);

  MACE_REGISTER_GPU_OP(op_registry, "DepthwiseDeconv2d", DepthwiseDeconv2dOp);
}

}  // namespace ops
}  // namespace mace
