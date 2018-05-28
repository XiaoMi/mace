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

#ifndef MACE_KERNELS_CHANNEL_SHUFFLE_H_
#define MACE_KERNELS_CHANNEL_SHUFFLE_H_

#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

template<DeviceType D, typename T>
struct ChannelShuffleFunctor {
  explicit ChannelShuffleFunctor(const int groups) : groups_(groups) {}

  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    Tensor::MappingGuard logits_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();

    index_t batch = input->dim(0);
    index_t channels = input->dim(1);
    index_t height = input->dim(2);
    index_t width = input->dim(3);

    index_t image_size = height * width;
    index_t batch_size = channels * image_size;
    index_t channels_per_group = channels / groups_;

#pragma omp parallel for collapse(2)
    for (index_t b = 0; b < batch; ++b) {
      for (index_t c = 0; c < channels; ++c) {
        const T *input_base = input_ptr + b * batch_size;
        T *output_base = output_ptr + b * batch_size;
        index_t g = c % groups_;
        index_t idx = c / groups_;
        for (index_t hw = 0; hw < height * width; ++hw) {
          output_base[c * image_size + hw] = input_base[
              (g * channels_per_group + idx) * image_size + hw];
        }
      }
    }

    return MACE_SUCCESS;
  }

  const int groups_;
};

#ifdef MACE_ENABLE_OPENCL
template<typename T>
struct ChannelShuffleFunctor<DeviceType::GPU, T> {
  explicit ChannelShuffleFunctor(const int groups) : groups_(groups) {}

  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  const int groups_;
  std::vector<index_t> input_shape_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_CHANNEL_SHUFFLE_H_
