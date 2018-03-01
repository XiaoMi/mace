//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_CHANNEL_SHUFFLE_H_
#define MACE_KERNELS_CHANNEL_SHUFFLE_H_

#include "mace/core/future.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
class ChannelShuffleFunctor {
 public:
  ChannelShuffleFunctor(const int group) : group_(group) {}

  void operator()(const T *input, const index_t *input_shape,
                  T *output, StatsFuture *future) {
    index_t batch = input_shape[0];
    index_t channels = input_shape[1];
    index_t height = input_shape[2];
    index_t width = input_shape[3];

    index_t image_size = height * width;
    int channels_of_group = channels / group_;

    for (int b = 0; b < batch; ++b) {
      for (int c = 0; c < channels_of_group; ++c) {
        for (int g = 0; g < group_; ++g) {
          index_t input_offset =
              (b * channels + g * channels_of_group + c) * image_size;
          index_t output_offset = (b * channels + c * group_ + g) * image_size;
          memcpy(output + output_offset, input + input_offset,
                 image_size * sizeof(T));
        }
      }
    }
  }

 private:
  const int group_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_CHANNEL_SHUFFLE_H_
