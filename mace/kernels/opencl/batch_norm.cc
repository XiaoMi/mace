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

#include "mace/kernels/batch_norm.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template <typename T>
MaceStatus BatchNormFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input,
    const Tensor *scale,
    const Tensor *offset,
    const Tensor *mean,
    const Tensor *var,
    const float epsilon,
    Tensor *output,
    StatsFuture *future) {
  MACE_CHECK(folded_constant_ || (mean != nullptr && var != nullptr));

  const index_t batch = input->dim(0);
  const index_t height = input->dim(1);
  const index_t width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};

  auto runtime = context_->device()->opencl_runtime();

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(kernel_error_, context_);
    NON_UNIFORM_WG_CONFIG;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("batch_norm");
    built_options.emplace("-Dbatch_norm=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
    if (folded_constant_) {
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
      default:
        LOG(FATAL) << "Unknown activation type: " << activation_;
    }

    MACE_RETURN_IF_ERROR(runtime->BuildKernel("batch_norm", kernel_name,
                                              built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARG;
    SET_3D_GWS_ARGS(kernel_);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, *(scale->opencl_image()));
    kernel_.setArg(idx++, *(offset->opencl_image()));
    if (!folded_constant_) {
      kernel_.setArg(idx++, *(mean->opencl_image()));
      kernel_.setArg(idx++, *(var->opencl_image()));
      kernel_.setArg(idx++, epsilon);
    }
    kernel_.setArg(idx++, *(output->opencl_image()));
    kernel_.setArg(idx++, relux_max_limit_);

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  std::string tuning_key =
      Concat("batch_norm_opencl_kernel", activation_, output->dim(0),
             output->dim(1), output->dim(2), output->dim(3), folded_constant_);
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, future));
  OUT_OF_RANGE_VALIDATION(kernel_error_);
  return MACE_SUCCESS;
}

template struct BatchNormFunctor<DeviceType::GPU, float>;
template struct BatchNormFunctor<DeviceType::GPU, half>;
}  // namespace kernels
}  // namespace mace
