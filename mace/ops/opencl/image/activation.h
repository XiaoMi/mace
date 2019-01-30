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
#ifndef MACE_OPS_OPENCL_IMAGE_ACTIVATION_H_
#define MACE_OPS_OPENCL_IMAGE_ACTIVATION_H_

#include "mace/ops/opencl/activation.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/common/activation_type.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

template <typename T>
class ActivationKernel : public OpenCLActivationKernel {
 public:
  ActivationKernel(ActivationType type,
                   T relux_max_limit,
                   T leakyrelu_coefficient)
      : activation_(type), relux_max_limit_(relux_max_limit),
        leakyrelu_coefficient_(leakyrelu_coefficient) {}

  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *alpha,
      Tensor *output) override;

 private:
  ActivationType activation_;
  T relux_max_limit_;
  T leakyrelu_coefficient_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
  std::string tuning_key_prefix_;
};

template <typename T>
MaceStatus ActivationKernel<T>::Compute(
    OpContext *context,
    const Tensor *input,
    const Tensor *alpha,
    Tensor *output) {
  const index_t batch = input->dim(0);
  const index_t height = input->dim(1);
  const index_t width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("activation");
    built_options.emplace("-Dactivation=" + kernel_name);
    auto dt = DataTypeToEnum<T>::value;
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
    switch (activation_) {
      case RELU:
        tuning_key_prefix_ = "relu_opencl_kernel";
        built_options.emplace("-DUSE_RELU");
        break;
      case RELUX:
        tuning_key_prefix_ = "relux_opencl_kernel";
        built_options.emplace("-DUSE_RELUX");
        break;
      case PRELU:
        tuning_key_prefix_ = "prelu_opencl_kernel";
        built_options.emplace("-DUSE_PRELU");
        break;
      case TANH:
        tuning_key_prefix_ = "tanh_opencl_kernel";
        built_options.emplace("-DUSE_TANH");
        break;
      case SIGMOID:
        tuning_key_prefix_ = "sigmoid_opencl_kernel";
        built_options.emplace("-DUSE_SIGMOID");
        break;
      case LEAKYRELU:
        tuning_key_prefix_ = "leakyrelu_opencl_kernel";
        built_options.emplace("-DUSE_LEAKYRELU");
        break;
      default:
        LOG(FATAL) << "Unknown activation type: " << activation_;
    }
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("activation", kernel_name,
                                              built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};

  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    int idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    if (activation_ == PRELU) {
      MACE_CHECK_NOTNULL(alpha);
      kernel_.setArg(idx++, *(alpha->opencl_image()));
    }
    kernel_.setArg(idx++, static_cast<float>(relux_max_limit_));
    kernel_.setArg(idx++, static_cast<float>(leakyrelu_coefficient_));
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  std::string tuning_key =
      Concat(tuning_key_prefix_, output->dim(0), output->dim(1), output->dim(2),
             output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_ACTIVATION_H_
