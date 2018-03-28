//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_CHANNEL_SHUFFLE_H_
#define MACE_KERNELS_CHANNEL_SHUFFLE_H_

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

template <typename T>
struct ChannelShuffleFunctor<DeviceType::OPENCL, T> {
  explicit ChannelShuffleFunctor(const int groups) : groups_(groups) {}

  void operator()(const Tensor *input, Tensor *output, StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  bool is_non_uniform_work_groups_supported_;
  const int groups_;
  std::vector<index_t> input_shape_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_CHANNEL_SHUFFLE_H_
