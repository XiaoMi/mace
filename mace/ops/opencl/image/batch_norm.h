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
#ifndef MACE_OPS_OPENCL_IMAGE_BATCH_NORM_H_
#define MACE_OPS_OPENCL_IMAGE_BATCH_NORM_H_

#include "mace/ops/opencl/batch_norm.h"

#include <memory>
#include <vector>
#include <set>
#include <string>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/activation.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

template <typename T>
class BatchNormKernel : public OpenCLBatchNormKernel {
 public:
  BatchNormKernel(
      const float epsilon,
      const ActivationType activation,
      const float relux_max_limit,
      const float leakyrelu_coefficient);
  MaceStatus Compute(OpContext *context,
                     const Tensor *input,
                     const Tensor *scale,
                     const Tensor *offset,
                     const Tensor *mean,
                     const Tensor *var,
                     Tensor *output) override;

 private:
  const float epsilon_;
  const ActivationType activation_;
  const float relux_max_limit_;
  const float leakyrelu_coefficient_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

template <typename T>
BatchNormKernel<T>::BatchNormKernel(const float epsilon,
                                    const ActivationType activation,
                                    const float relux_max_limit,
                                    const float leakyrelu_coefficient)
    : epsilon_(epsilon),
      activation_(activation),
      relux_max_limit_(relux_max_limit),
      leakyrelu_coefficient_(leakyrelu_coefficient) {}

template <typename T>
MaceStatus BatchNormKernel<T>::Compute(
    OpContext *context,
    const Tensor *input,
    const Tensor *scale,
    const Tensor *offset,
    const Tensor *mean,
    const Tensor *var,
    Tensor *output) {
  bool not_folded = (mean != nullptr && var != nullptr);

  const index_t batch = input->dim(0);
  const index_t height = input->dim(1);
  const index_t width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("batch_norm");
    built_options.emplace("-Dbatch_norm=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
    if (!not_folded) {
      built_options.emplace("-DFOLDED_CONSTANT");
    }
    switch (activation_) {
      case NOOP:
        break;
      case RELU:
        built_options.emplace("-DUSE_RELU");
        break;
      case RELUX:
        built_options.emplace("-DUSE_RELUX");
        break;
      case TANH:
        built_options.emplace("-DUSE_TANH");
        break;
      case SIGMOID:
        built_options.emplace("-DUSE_SIGMOID");
        break;
      case LEAKYRELU:
        built_options.emplace("-DUSE_LEAKYRELU");
        break;
      default:
        LOG(FATAL) << "Unknown activation type: " << activation_;
    }

    MACE_RETURN_IF_ERROR(runtime->BuildKernel("batch_norm", kernel_name,
                                              built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, *(scale->opencl_image()));
    kernel_.setArg(idx++, *(offset->opencl_image()));
    if (not_folded) {
      kernel_.setArg(idx++, *(mean->opencl_image()));
      kernel_.setArg(idx++, *(var->opencl_image()));
      kernel_.setArg(idx++, epsilon_);
    }
    kernel_.setArg(idx++, *(output->opencl_image()));
    kernel_.setArg(idx++, relux_max_limit_);
    kernel_.setArg(idx++, leakyrelu_coefficient_);

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  std::string tuning_key =
      Concat("batch_norm_opencl_kernel", activation_, output->dim(0),
             output->dim(1), output->dim(2), output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));
  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_BATCH_NORM_H_
