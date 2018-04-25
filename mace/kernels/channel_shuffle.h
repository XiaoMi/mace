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

template <DeviceType D, typename T>
struct ChannelShuffleFunctor {
  explicit ChannelShuffleFunctor(const int groups) : groups_(groups) {}

  void operator()(const Tensor *input,
                  Tensor *output,
                  StatsFuture *future) {
    output->ResizeLike(input);

    Tensor::MappingGuard logits_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();

    index_t batch = input->dim(0);
    index_t height = input->dim(1);
    index_t width = input->dim(2);
    index_t channels = input->dim(3);

    index_t bhw_fuse = batch * height * width;
    int channels_per_group = channels / groups_;

#pragma omp parallel for
    for (int bhw = 0; bhw < bhw_fuse; ++bhw) {
      for (int c = 0; c < channels; ++c) {
        index_t channel_base = bhw * channels;
        output_ptr[channel_base + c] =
          input_ptr[channel_base + c % groups_ * channels_per_group
            + c / groups_];
      }
    }
  }

  const int groups_;
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
struct ChannelShuffleFunctor<DeviceType::OPENCL, T> {
  explicit ChannelShuffleFunctor(const int groups) : groups_(groups) {}

  void operator()(const Tensor *input, Tensor *output, StatsFuture *future);

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
