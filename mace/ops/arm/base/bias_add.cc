// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include "mace/ops/arm/base/bias_add.h"

namespace mace {
namespace ops {
namespace arm {

template<typename T>
MaceStatus BiasAdd<T>::Compute(const OpContext *context,
                               const Tensor *input,
                               const Tensor *bias,
                               Tensor *output,
                               const bool isNCHW) {
  if (input != output) {
    if (bias == nullptr) {
      output->Copy(*input);
    } else {
      MACE_RETURN_IF_ERROR(output->ResizeLike(input));
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard bias_guard(bias);
      Tensor::MappingGuard output_guard(output);
      AddBias(context, input, bias, output, isNCHW);
    }
  } else {
    if (bias != nullptr) {
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard bias_guard(bias);
      AddBias(context, input, bias, output, isNCHW);
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

template<typename T>
void BiasAdd<T>::AddBias(const OpContext *context,
                         const Tensor *input,
                         const Tensor *bias,
                         mace::Tensor *output,
                         const bool isNCHW) {
  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  if (isNCHW) {
    if (bias->dim_size() == 1) {
      AddBiasNCHW<1>(&thread_pool, input, bias, output);
    } else {
      AddBiasNCHW<2>(&thread_pool, input, bias, output);
    }
  } else {
    if (bias->dim_size() == 1) {
      AddBiasNHWC<1>(&thread_pool, input, bias, output);
    } else {
      AddBiasNHWC<2>(&thread_pool, input, bias, output);
    }
  }
}

void RegisterBiasAddDelegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, BiasAdd<float>, DelegatorParam,
      MACE_DELEGATOR_KEY(BiasAdd, DeviceType::CPU, float, ImplType::NEON));
#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_DELEGATOR(
      registry, BiasAdd<uint8_t>, DelegatorParam,
      MACE_DELEGATOR_KEY(BiasAdd, DeviceType::CPU, uint8_t, ImplType::NEON));
#endif  // MACE_ENABLE_QUANTIZE
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
