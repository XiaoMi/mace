//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#include "mace/kernels/local_response_norm.h"

namespace mace {
namespace kernels {

void LocalResponseNormFunctor<DeviceType::NEON, float>::operator()(
  const Tensor *input,
  int depth_radius,
  float bias,
  float alpha,
  float beta,
  Tensor *output,
  StatsFuture *future) {
  const index_t batch = input->dim(0);
  const index_t channels = input->dim(1);
  const index_t height = input->dim(2);
  const index_t width = input->dim(3);

  const float *input_ptr = input->data<float>();
  float *output_ptr = output->mutable_data<float>();

  index_t image_size = height * width;
  index_t batch_size = channels * image_size;

#pragma omp parallel for collapse(2)
  for (index_t b = 0; b < batch; ++b) {
    for (index_t c = 0; c < channels; ++c) {
      const int begin_input_c = std::max(static_cast<index_t>(0),
                                         c - depth_radius);
      const int end_input_c = std::min(channels, c + depth_radius + 1);

      index_t pos = b * batch_size;
      for (index_t hw = 0; hw < height * width; ++hw, ++pos) {
        float accum = 0.f;
        for (int input_c = begin_input_c; input_c < end_input_c; ++input_c) {
          const float input_val = input_ptr[pos + input_c * image_size];
          accum += input_val * input_val;
        }
        const float multiplier = std::pow(bias + alpha * accum, -beta);
        output_ptr[pos + c * image_size] =
            input_ptr[pos + c * image_size] * multiplier;
      }
    }
  }
}

}  // namespace kernels
}  // namespace mace
