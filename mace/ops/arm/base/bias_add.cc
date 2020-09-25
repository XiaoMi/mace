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

#include <functional>
#include <vector>

#include "mace/ops/arm/base/common_neon.h"

namespace mace {
namespace ops {
namespace arm {

extern template void BiasAdd<uint8_t>::AddBiasNCHW<1>(
    utils::ThreadPool *, const Tensor *, const Tensor *, Tensor *);
extern template void BiasAdd<uint8_t>::AddBiasNCHW<2>(
    utils::ThreadPool *, const Tensor *, const Tensor *, Tensor *);
extern template void BiasAdd<uint8_t>::AddBiasNHWC<1>(
    utils::ThreadPool *, const Tensor *, const Tensor *, Tensor *);
extern template void BiasAdd<uint8_t>::AddBiasNHWC<2>(
    utils::ThreadPool *, const Tensor *, const Tensor *, Tensor *);

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

template <typename T>
template <int Dim>
void BiasAdd<T>::AddBiasNCHW(utils::ThreadPool *thread_pool,
                             const Tensor *input,
                             const Tensor *bias,
                             Tensor *output) {
  const auto input_data = input->data<T>();
  const auto bias_data = bias->data<T>();
  auto output_data = output->mutable_data<T>();

  const index_t batch = input->dim(0);
  const index_t channels = input->dim(1);
  const index_t image_size = input->dim(2) * input->dim(3);
  const index_t block_count = image_size / 4;
  const index_t remain = image_size % 4;
  thread_pool->Compute2D(
      [=](index_t start0, index_t end0, index_t step0, index_t start1,
          index_t end1, index_t step1) {
        for (index_t b = start0; b < end0; b += step0) {
          const index_t b_offset = b * channels;
          for (index_t c = start1; c < end1; c += step1) {
            const index_t offset = (b_offset + c) * image_size;
            auto input_ptr = input_data + offset;
            auto output_ptr = output_data + offset;
            const float bias = bias_data[bias_index<Dim>(b_offset, c)];
            float32x4_t vbias = vdupq_n_f32(bias);

            for (index_t i = 0; i < block_count; ++i) {
              float32x4_t v = vld1q(input_ptr);
              v = vaddq_f32(v, vbias);
              vst1q(output_ptr, v);

              input_ptr += 4;
              output_ptr += 4;
            }
            for (index_t i = 0; i < remain; ++i) {
              (*output_ptr++) = (*input_ptr++) + bias;
            }
          }
        }
      },
      0, batch, 1, 0, channels, 1);
}


template <typename T>
template <int Dim>
void BiasAdd<T>::AddBiasNHWC(utils::ThreadPool *thread_pool,
                             const Tensor *input,
                             const Tensor *bias,
                             Tensor *output) {
  const auto input_ptr = input->data<T>();
  const auto bias_ptr = bias->data<T>();
  auto output_ptr = output->mutable_data<T>();

  const std::vector<index_t> &shape = input->shape();
  const index_t channels = *shape.rbegin();
  const auto batch = shape[0];
  if (Dim == 2) {
    MACE_CHECK(batch == bias->shape()[0]);
  }
  const index_t fused_hw = std::accumulate(shape.begin() + 1, shape.end() - 1,
                                           1, std::multiplies<index_t>());
  thread_pool->Compute2D(
      [=](index_t start0, index_t end0, index_t step0, index_t start1,
          index_t end1, index_t step1) {
        for (index_t i = start0; i < end0; i += step0) {
          auto offset = i * fused_hw;
          auto bias_offset = i * channels;
          for (index_t j = start1; j < end1; j += step1) {
            index_t pos = (offset + j) * channels;
            for (index_t c = 0; c < channels; ++c, ++pos) {
              output_ptr[pos] =
                  input_ptr[pos] + bias_ptr[bias_index<Dim>(bias_offset, c)];
            }
          }
        }
      },
      0, batch, 1, 0, fused_hw, 1);
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

  MACE_REGISTER_BF16_DELEGATOR(
      registry, BiasAdd<BFloat16>, DelegatorParam,
      MACE_DELEGATOR_KEY(BiasAdd, DeviceType::CPU, BFloat16, ImplType::NEON));
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
