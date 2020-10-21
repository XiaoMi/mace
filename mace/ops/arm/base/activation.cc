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

#include <algorithm>

#include "mace/ops/arm/base/common_neon.h"

namespace mace {
namespace ops {
namespace arm {

extern template void Activation<uint8_t>::ActivateRelu(
    utils::ThreadPool *, const Tensor *, Tensor *);
extern template void Activation<uint8_t>::ActivateRelux(
    utils::ThreadPool *, const Tensor *, Tensor *);
extern template void Activation<uint8_t>::ActivateLeakyRelu(
    utils::ThreadPool *, const Tensor *, Tensor *);
extern template void Activation<uint8_t>::ActivateTanh(
    utils::ThreadPool *, const Tensor *, Tensor *);
extern template void Activation<uint8_t>::ActivateSigmoid(
    utils::ThreadPool *, const Tensor *, Tensor *);

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

    case ELU: {
      ActivateElu(&thread_pool, input, output);
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

template<typename T>
void Activation<T>::ActivateRelu(utils::ThreadPool *thread_pool,
                                 const Tensor *input,
                                 Tensor *output) {
  const auto input_data = input->data<T>();
  auto output_data = output->mutable_data<T>();
  const index_t input_size = input->size();
  const float32x4_t vzero = vdupq_n_f32(0.f);
  const index_t block_count = input_size / 4;
  thread_pool->Compute1D(
      [=](index_t start, index_t end, index_t step) {
        const T *input_ptr = input_data + start * 4;
        T *output_ptr = output_data + start * 4;

        for (index_t i = start; i < end; i += step) {
          float32x4_t v = vld1q(input_ptr);
          v = vmaxq_f32(v, vzero);
          vst1q(output_ptr, v);

          input_ptr += 4;
          output_ptr += 4;
        }
      },
      0, block_count, 1);

  // remain
  for (index_t i = block_count * 4; i < input_size; ++i) {
    output_data[i] = std::max(0.f, input_data[i]);
  }
}

template<typename T>
void Activation<T>::ActivateRelux(utils::ThreadPool *thread_pool,
                                  const Tensor *input,
                                  Tensor *output) {
  const auto input_data = input->data<T>();
  auto output_data = output->mutable_data<T>();
  const index_t input_size = input->size();
  const float32x4_t vzero = vdupq_n_f32(0.f);
  const float32x4_t vlimit = vdupq_n_f32(limit_);
  const index_t block_count = input_size / 4;

  thread_pool->Compute1D(
      [=](index_t start, index_t end, index_t step) {
        auto input_ptr = input_data + start * 4;
        auto output_ptr = output_data + start * 4;

        for (index_t i = start; i < end; i += step) {
          float32x4_t v = vld1q(input_ptr);
          v = vmaxq_f32(v, vzero);
          v = vminq_f32(v, vlimit);
          vst1q(output_ptr, v);

          input_ptr += 4;
          output_ptr += 4;
        }
      },
      0, block_count, 1);

  // remain
  for (index_t i = block_count * 4; i < input_size; ++i) {
    output_data[i] = std::max(0.f, std::min(limit_, input_data[i]));
  }
}

template<typename T>
void Activation<T>::ActivateLeakyRelu(utils::ThreadPool *thread_pool,
                                      const Tensor *input,
                                      Tensor *output) {
  const auto input_data = input->data<T>();
  auto output_data = output->mutable_data<T>();
  const index_t input_size = input->size();
  const float32x4_t vzero = vdupq_n_f32(0.f);
  const float32x4_t valpha = vdupq_n_f32(activation_coefficient_);
  const index_t block_count = input_size / 4;

  thread_pool->Compute1D(
      [=](index_t start, index_t end, index_t step) {
        auto input_ptr = input_data + start * 4;
        auto output_ptr = output_data + start * 4;

        for (index_t i = start; i < end; i += step) {
          float32x4_t v = vld1q(input_ptr);
          float32x4_t u = vminq_f32(v, vzero);
          v = vmaxq_f32(v, vzero);
          v = vmlaq_f32(v, valpha, u);
          vst1q(output_ptr, v);

          input_ptr += 4;
          output_ptr += 4;
        }
      },
      0, block_count, 1);

  // remain
  for (index_t i = block_count * 4; i < input_size; ++i) {
    output_data[i] = std::max(input_data[i], 0.f) +
        std::min(input_data[i], 0.f) * activation_coefficient_;
  }
}

template<typename T>
void Activation<T>::ActivateTanh(utils::ThreadPool *thread_pool,
                                 const Tensor *input,
                                 Tensor *output) {
  const auto input_data = input->data<T>();
  auto output_data = output->mutable_data<T>();
  const index_t input_size = input->size();

  thread_pool->Compute1D(
      [=](index_t start, index_t end, index_t step) {
        for (index_t i = start; i < end; i += step) {
          output_data[i] = std::tanh(input_data[i]);
        }
      },
      0, input_size, 1);
}

template<typename T>
void Activation<T>::ActivateSigmoid(utils::ThreadPool *thread_pool,
                                    const Tensor *input,
                                    Tensor *output) {
  const auto input_data = input->data<T>();
  auto output_data = output->mutable_data<T>();
  const index_t input_size = input->size();

  thread_pool->Compute1D(
      [=](index_t start, index_t end, index_t step) {
        for (index_t i = start; i < end; i += step) {
          output_data[i] = 1 / (1 + std::exp(-(input_data[i])));
        }
      },
      0, input_size, 1);
}

template<typename T>
void Activation<T>::ActivateElu(utils::ThreadPool *thread_pool,
                                const Tensor *input,
                                Tensor *output) {
  const auto *input_data = input->data<T>();
  auto *output_data = output->mutable_data<T>();
  const index_t input_size = input->size();

  thread_pool->Compute1D(
      [=](index_t start, index_t end, index_t step) {
        for (index_t i = start; i < end; i += step) {
          const auto in_val = input_data[i];
          if (in_val < 0) {
            output_data[i] = (std::exp(in_val) - 1) * activation_coefficient_;
          } else {
            output_data[i] = in_val;
          }
        }
      },
      0, input_size, 1);
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

  MACE_REGISTER_BF16_DELEGATOR(
      registry, Activation<BFloat16>, delegator::ActivationParam,
      MACE_DELEGATOR_KEY(Activation, DeviceType::CPU, BFloat16,
                         ImplType::NEON));
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
