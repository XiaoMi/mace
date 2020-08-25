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

#include "mace/ops/arm/base/activation.h"

namespace mace {
namespace ops {
namespace arm {

template<typename T>
MaceStatus Activation<T>::Compute(const OpContext *context,
                                  const Tensor *input, Tensor *output) {
  Tensor::MappingGuard input_guard(input);
  if (input != output) {
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    Tensor::MappingGuard output_guard(output);
    DoActivation(context, input, output);
  } else {
    DoActivation(context, input, output);
  }

  return MaceStatus::MACE_SUCCESS;
}

template<typename T>
void Activation<T>::DoActivation(const OpContext *context,
                                 const Tensor *input,
                                 Tensor *output) {
  utils::ThreadPool &thread_pool =
      context->device()->cpu_runtime()->thread_pool();

  switch (type_) {
    case RELU: {
      ActivateRelu(&thread_pool, input, output);
      break;
    }

    case RELUX: {
      ActivateRelux(&thread_pool, input, output);
      break;
    }

    case LEAKYRELU: {
      ActivateLeakyRelu(&thread_pool, input, output);
      break;
    }

    case TANH: {
      ActivateTanh(&thread_pool, input, output);
      break;
    }

    case SIGMOID: {
      ActivateSigmoid(&thread_pool, input, output);
      break;
    }

    case NOOP: {
      break;
    }

    default: {
      MACE_NOT_IMPLEMENTED;
    }
  }
}

void RegisterActivationDelegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, Activation<float>, delegator::ActivationParam,
      MACE_DELEGATOR_KEY(Activation, DeviceType::CPU, float, ImplType::NEON));
#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_DELEGATOR(
      registry, Activation<uint8_t>, delegator::ActivationParam,
      MACE_DELEGATOR_KEY(Activation, DeviceType::CPU, uint8_t, ImplType::NEON));
#endif  // MACE_ENABLE_QUANTIZE
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
