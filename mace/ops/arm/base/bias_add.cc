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
MaceStatus BiasAdd<T>::Compute(const OpContext *context, const Tensor *input,
                               const Tensor *bias, Tensor *output) {
  if (input != output) {
    if (bias == nullptr) {
      output->Copy(*input);
    } else {
      MACE_RETURN_IF_ERROR(output->ResizeLike(input));
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard bias_guard(bias);
      Tensor::MappingGuard output_guard(output);
      AddBias(context, input, bias, output);
    }
  } else {
    if (bias != nullptr) {
      Tensor::MappingGuard input_guard(input);
      Tensor::MappingGuard bias_guard(bias);
      AddBias(context, input, bias, output);
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

template<typename T>
void BiasAdd<T>::AddBias(const OpContext *context, const Tensor *input,
                         const Tensor *bias, mace::Tensor *output) {
  auto input_data = input->data<T>();
  auto bias_data = bias->data<T>();
  auto output_data = output->mutable_data<T>();

  const index_t batch = input->dim(0);
  const index_t channels = input->dim(1);

  const index_t height = input->dim(2);
  const index_t width = input->dim(3);
  const index_t image_size = height * width;

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  if (bias->dim_size() == 1) {
    Add1DimBias(&thread_pool, input_data, bias_data,
                output_data, batch, channels, image_size);
  } else {
    Add2DimsBias(&thread_pool, input_data, bias_data,
                     output_data, batch, channels, image_size);
  }
}

void RegisterBiasAddDelegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, BiasAdd<float>, DelegatorParam,
      MACE_DELEGATOR_KEY(BiasAdd, DeviceType::CPU, float, ImplType::NEON));
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
