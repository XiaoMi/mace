//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/global_avg_pooling.h"
#include <arm_neon.h>

namespace mace {
namespace kernels {

template <>
void GlobalAvgPoolingFunctor<DeviceType::NEON, float>::operator()(
    const float *input, const index_t *input_shape,
    float *output, StatsFuture *future) {
  index_t batch = input_shape[0];
  index_t channels = input_shape[1];
  index_t height = input_shape[2];
  index_t width = input_shape[3];

  index_t image_size = height * width;
  index_t input_offset = 0;
  index_t total_channels = batch * channels;

#pragma omp parallel for
  for (int c = 0; c < total_channels; ++c) {
    const float *inptr = input + c * image_size;
    float sum = 0.0;

    int num_vectors = image_size >> 2;
    int remain = image_size - (num_vectors << 2);

    if (num_vectors > 0) {
      float sum_out[4] = {0.0, 0.0, 0.0, 0.0};

      float32x4_t sum_vector = vld1q_f32(inptr);
      inptr += 4;
      for (int n = 1; n < num_vectors; ++n) {
        float32x4_t vector = vld1q_f32(inptr);
        sum_vector = vaddq_f32(sum_vector, vector);
        inptr += 4;
      }
      vst1q_f32(sum_out, sum_vector);

      sum = sum_out[0] + sum_out[1] + sum_out[2] + sum_out[3];
    }

    for (int i = 0; i < remain; ++i) {
      sum += *inptr;
      ++inptr;
    }
    output[c] = sum / image_size;
  }
};

}  // namespace kernels
}  // namespace mace
