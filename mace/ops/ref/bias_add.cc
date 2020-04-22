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

#include "mace/ops/delegator/bias_add.h"

namespace mace {
namespace ops {
namespace ref {

template<typename T>
class BiasAdd : public delegator::BiasAdd {
 public:
  explicit BiasAdd(const DelegatorParam &param) : delegator::BiasAdd(param) {}
  ~BiasAdd() = default;

  MaceStatus Compute(const OpContext *context, const Tensor *input,
                     const Tensor *bias, Tensor *output) override;

 private:
  void AddBias(const OpContext *context, const Tensor *input,
               const Tensor *bias, Tensor *output);
};

template<typename T>
MaceStatus BiasAdd<T>::Compute(const OpContext *context,
                               const Tensor *input,
                               const Tensor *bias,
                               Tensor *output) {
  Tensor::MappingGuard input_guard(input);
  Tensor::MappingGuard bias_guard(bias);
  if (input != output) {
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    if (bias == nullptr) {
      output->Copy(*input);
    } else {
      Tensor::MappingGuard output_guard(output);
      AddBias(context, input, bias, output);
    }
  } else {
    if (bias != nullptr) {
      AddBias(context, input, bias, output);
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

template<typename T>
void BiasAdd<T>::AddBias(const OpContext *context,
                         const Tensor *input,
                         const Tensor *bias,
                         mace::Tensor *output) {
  MACE_UNUSED(context);
  auto input_data = input->data<T>();
  auto bias_data = bias->data<T>();
  auto output_data = output->mutable_data<T>();

  const index_t batch = input->dim(0);
  const index_t channels = input->dim(1);
  const index_t height = output->dim(2);
  const index_t width = output->dim(3);
  const index_t image_size = height * width;

  auto bias_b = bias->dim_size() == 1 ? 0 : bias->shape()[1];
  for (index_t b = 0; b < batch; ++b) {
    for (index_t c = 0; c < channels; ++c) {
      const index_t offset = (b * channels + c) * image_size;
      auto input_ptr = input_data + offset;
      auto output_ptr = output_data + offset;
      const float bias = bias_data[bias_b * channels + c];

      for (index_t i = 0; i < image_size; ++i) {
        (*output_ptr++) = (*input_ptr++) + bias;
      }
    }
  }
}

void RegisterBiasAddDelegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, BiasAdd<float>, DelegatorParam,
      MACE_DELEGATOR_KEY(BiasAdd, DeviceType::CPU, float, ImplType::REF));
  MACE_REGISTER_BF16_DELEGATOR(
      registry, BiasAdd<BFloat16>, DelegatorParam,
      MACE_DELEGATOR_KEY(BiasAdd, DeviceType::CPU, BFloat16, ImplType::REF));
}

}  // namespace ref
}  // namespace ops
}  // namespace mace

