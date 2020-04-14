// Copyright 2019 The MACE Authors. All Rights Reserved.
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

#include "mace/ops/delegator/activation.h"

namespace mace {
namespace ops {
namespace ref {

template<typename T>
class Activation : public delegator::Activation {
 public:
  explicit Activation(const delegator::ActivationParam &param)
      : delegator::Activation(param) {}
  ~Activation() = default;

  MaceStatus Compute(const OpContext *context, const Tensor *input,
                     Tensor *output) override;

 private:
  void DoActivation(const OpContext *context, const Tensor *input,
                    Tensor *output);
};

template<typename T>
MaceStatus Activation<T>::Compute(const OpContext *context,
                                  const Tensor *input,
                                  Tensor *output) {
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
  MACE_UNUSED(context);
  auto input_ptr = input->data<T>();
  auto output_ptr = output->mutable_data<T>();
  const index_t size = input->size();

  switch (type_) {
    case RELU: {
      for (index_t i = 0; i < size; ++i) {
        *output_ptr++ = std::max(0.f, *input_ptr++);
      }

      break;
    }

    case RELUX: {
      for (index_t i = 0; i < size; ++i) {
        *output_ptr++ = std::max(0.f, std::min(limit_, *input_ptr++));
      }

      break;
    }

    case LEAKYRELU: {
      for (index_t i = 0; i < size; ++i) {
        *output_ptr =
            std::max<float>(*input_ptr, 0.f)
                + std::min(*input_ptr, 0.f) * leakyrelu_coefficient_;
        ++input_ptr;
        ++output_ptr;
      }

      break;
    }

    case TANH: {
      for (index_t i = 0; i < size; ++i) {
        *output_ptr++ = std::tanh(*input_ptr++);
      }

      break;
    }

    case SIGMOID: {
      for (index_t i = 0; i < size; ++i) {
        *output_ptr++ = 1 / (1 + std::exp(-(*input_ptr++)));
      }
      break;
    }

    case NOOP:break;

    default:MACE_NOT_IMPLEMENTED;
  }
}

void RegisterActivationDelegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, Activation<float>, delegator::ActivationParam,
      MACE_DELEGATOR_KEY(Activation, DeviceType::CPU, float, ImplType::REF));
  MACE_REGISTER_BF16_DELEGATOR(
      registry, Activation<BFloat16>, delegator::ActivationParam,
      MACE_DELEGATOR_KEY(Activation, DeviceType::CPU, BFloat16, ImplType::REF));
}

}  // namespace ref
}  // namespace ops
}  // namespace mace
