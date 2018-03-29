//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#include "mace/kernels/batch_norm.h"

namespace mace {
namespace kernels {

void BatchNormFunctor<DeviceType::NEON, float>::operator()(
  const Tensor *input,
  const Tensor *scale,
  const Tensor *offset,
  const Tensor *mean,
  const Tensor *var,
  const float epsilon,
  Tensor *output,
  StatsFuture *future) {
  // Batch normalization in the paper https://arxiv.org/abs/1502.03167 .
  // The calculation formula for inference is
  // Y = \frac{ \scale } { \sqrt{var+\variance_epsilon} } * X +
  //          ( \offset - \frac { \scale * mean } {
  //          \sqrt{var+\variance_epsilon} }
  // new_scale = \frac{ \scale } { \sqrt{var+\variance_epsilon} }
  // new_offset = \offset - mean * common_val;
  // Y = new_scale * X + new_offset;
  const index_t batch = input->dim(0);
  const index_t channels = input->dim(1);
  const index_t height = input->dim(2);
  const index_t width = input->dim(3);

  const float *input_ptr = input->data<float>();
  const float *scale_ptr = scale->data<float>();
  const float *offset_ptr = offset->data<float>();
  float *output_ptr = output->mutable_data<float>();

  std::vector<float> new_scale;
  std::vector<float> new_offset;
  if (!folded_constant_) {
    new_scale.resize(channels);
    new_offset.resize(channels);
    const float *mean_ptr = mean->data<float>();
    const float *var_ptr = var->data<float>();
#pragma omp parallel for
    for (index_t c = 0; c < channels; ++c) {
      new_scale[c] = scale_ptr[c] / std::sqrt(var_ptr[c] + epsilon);
      new_offset[c] = offset_ptr[c] - mean_ptr[c] * new_scale[c];
    }
  }

  const float *scale_data = folded_constant_ ? scale_ptr : new_scale.data();
  const float *offset_data = folded_constant_ ? offset_ptr : new_offset.data();

  index_t channel_size = height * width;
  index_t batch_size = channels * channel_size;

  // NEON is slower, so stick to the trivial implementaion
#pragma omp parallel for collapse(2)
  for (index_t b = 0; b < batch; ++b) {
    for (index_t c = 0; c < channels; ++c) {
      index_t offset = b * batch_size + c * channel_size;
      for (index_t hw = 0; hw < height * width; ++hw) {
        output_ptr[offset + hw] =
          scale_data[c] * input_ptr[offset + hw] + offset_data[c];
      }
    }
  }
  DoActivation(output_ptr, output_ptr, output->size(), activation_,
               relux_max_limit_);
}

}  // namespace kernels
}  // namespace mace
