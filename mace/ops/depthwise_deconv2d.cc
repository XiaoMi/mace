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
#include "mace/ops/arm/fp32/depthwise_deconv_2d_general.h"
#include "mace/ops/arm/fp32/depthwise_deconv_2d_3x3.h"
#include "mace/ops/arm/fp32/depthwise_deconv_2d_4x4.h"
#include "mace/ops/arm/fp32/bias_add.h"
#include "mace/ops/arm/fp32/activation.h"

#else
#include "mace/ops/ref/depthwise_deconv_2d.h"
#include "mace/ops/ref/bias_add.h"
#include "mace/ops/ref/activation.h"
#endif

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/utils/math.h"
#include "mace/public/mace.h"
#include "mace/utils/memory.h"
#include "mace/ops/common/conv_pool_2d_util.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/depthwise_deconv2d.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template<DeviceType D, class T>
class DepthwiseDeconv2dOp;

template<>
class DepthwiseDeconv2dOp<DeviceType::CPU, float>
    : public Deconv2dOpBase {
 public:
  explicit DepthwiseDeconv2dOp(OpConstructContext *context)
      : Deconv2dOpBase(context),
        activation_delegator_(activation_,
                              relux_max_limit_,
                              leakyrelu_coefficient_) {}

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

#ifdef MACE_ENABLE_NEON
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

    if (deconv2d_delegator_ == nullptr) {
      if (is_depthwise) {
        if (use_neon_3x3_s1) {
          deconv2d_delegator_ = make_unique<arm::fp32::DepthwiseDeconv2dK3x3S1>(
              paddings_, padding_type_, CAFFE);
        } else if (use_neon_3x3_s2) {
          deconv2d_delegator_ = make_unique<arm::fp32::DepthwiseDeconv2dK3x3S2>(
              paddings_, padding_type_, CAFFE);
        } else if (use_neon_4x4_s1) {
          deconv2d_delegator_ = make_unique<arm::fp32::DepthwiseDeconv2dK4x4S1>(
              paddings_, padding_type_, CAFFE);
        } else if (use_neon_4x4_s2) {
          deconv2d_delegator_ = make_unique<arm::fp32::DepthwiseDeconv2dK4x4S2>(
              paddings_, padding_type_, CAFFE);
        } else {
          deconv2d_delegator_ =
              make_unique<arm::fp32::DepthwiseDeconv2dGeneral>(
                  strides_,
                  std::vector<int>{1, 1},
                  paddings_,
                  padding_type_,
                  CAFFE);
        }
      } else {
        if (use_neon_3x3_s1) {
          deconv2d_delegator_ = make_unique<arm::fp32::GroupDeconv2dK3x3S1>(
              paddings_, padding_type_, group_, CAFFE);
        } else if (use_neon_3x3_s2) {
          deconv2d_delegator_ = make_unique<arm::fp32::GroupDeconv2dK3x3S2>(
              paddings_, padding_type_, group_, CAFFE);
        } else if (use_neon_4x4_s1) {
          deconv2d_delegator_ = make_unique<arm::fp32::GroupDeconv2dK4x4S1>(
              paddings_, padding_type_, group_, CAFFE);
        } else if (use_neon_4x4_s2) {
          deconv2d_delegator_ = make_unique<arm::fp32::GroupDeconv2dK4x4S2>(
              paddings_, padding_type_, group_, CAFFE);
        } else {
          deconv2d_delegator_ = make_unique<arm::fp32::GroupDeconv2dGeneral>(
              strides_,
              std::vector<int>{1, 1},
              paddings_,
              padding_type_,
              group_,
              CAFFE);
        }
      }
    }

    deconv2d_delegator_->Compute(context,
                                 input,
                                 filter,
                                 nullptr,
                                 output);
#else
    if (deconv2d_delegator_ == nullptr) {
      if (is_depthwise) {
        deconv2d_delegator_ = make_unique<ref::DepthwiseDeconv2d<float>>(
            strides_,
            std::vector<int>{1, 1},
            paddings_,
            padding_type_,
            CAFFE);
      } else {
        deconv2d_delegator_ = make_unique<ref::GroupDeconv2d<float>>(
            strides_,
            std::vector<int>{1, 1},
            paddings_,
            padding_type_,
            group_,
            CAFFE);
      }
    }
    deconv2d_delegator_->Compute(context,
                                 input,
                                 filter,
                                 nullptr,
                                 output);
#endif

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
  std::unique_ptr<ref::GroupDeconv2d<float>> deconv2d_delegator_;
  ref::BiasAdd bias_add_delegator_;
  ref::Activation activation_delegator_;
#endif  // MACE_ENABLE_NEON
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class DepthwiseDeconv2dOp<DeviceType::GPU, T> : public Deconv2dOpBase {
 public:
  explicit DepthwiseDeconv2dOp(OpConstructContext *context)
      : Deconv2dOpBase(context) {
    MemoryType mem_type = MemoryType::GPU_IMAGE;
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::DepthwiseDeconv2dKernel<T>>();
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
