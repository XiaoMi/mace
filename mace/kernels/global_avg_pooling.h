//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_GLOBAL_AVG_POOLING_H_
#define MACE_KERNELS_GLOBAL_AVG_POOLING_H_

#include "mace/core/future.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct GlobalAvgPoolingFunctor {
  void operator()(const T *input,
                  const index_t *input_shape,
                  T *output,
                  StatsFuture *future) {
    index_t batch = input_shape[0];
    index_t channels = input_shape[1];
    index_t height = input_shape[2];
    index_t width = input_shape[3];

    index_t image_size = height * width;
    index_t input_offset = 0;
    index_t total_channels = batch * channels;

    for (int c = 0; c < total_channels; ++c) {
      T sum = 0;
      for (int i = 0; i < image_size; ++i) {
        sum += input[input_offset + i];
      }
      output[c] = sum / image_size;
      input_offset += image_size;
    }
  }
};

template <>
void GlobalAvgPoolingFunctor<DeviceType::NEON, float>::operator()(
    const float *input,
    const index_t *input_shape,
    float *output,
    StatsFuture *future);

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_GLOBAL_AVG_POOLING_H_
