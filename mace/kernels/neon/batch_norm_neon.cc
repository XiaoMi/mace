//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/batch_norm.h"
#include <arm_neon.h>

namespace mace {
namespace kernels {

template <>
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
  // Y = \frac{ \scale } { \sqrt{var+\epsilon} } * X +
  //          ( \offset - \frac { \scale * mean } { \sqrt{var+\epsilon}
  //          }
  // new_scale = \frac{ \scale } { \sqrt{var+\epsilon} }
  // new_offset = \offset - mean * common_val;
  // Y = new_scale * X + new_offset;
  const index_t n = input->dim(0);
  const index_t sample_size = input->dim(1) * input->dim(2);
  const index_t channel = input->dim(3);

  const float *input_ptr = input->data<float>();
  const float *scale_ptr = scale->data<float>();
  const float *offset_ptr = offset->data<float>();
  const float *mean_ptr = mean->data<float>();
  const float *var_ptr = var->data<float>();
  float *output_ptr = output->mutable_data<float>();

  const index_t ch_blks = channel >> 2;
  const index_t remain_chs = channel - (ch_blks << 2);

  std::vector<float> new_scale(channel);
  std::vector<float> new_offset(channel);

#pragma omp parallel for
  for (index_t c = 0; c < channel; ++c) {
    new_scale[c] = scale_ptr[c] / std::sqrt(var_ptr[c] + epsilon);
    new_offset[c] = offset_ptr[c] - mean_ptr[c] * new_scale[c];
  }

#pragma omp parallel for collapse(2)
  for (index_t i = 0; i < n; ++i) {
    for (index_t j = 0; j < sample_size; ++j) {
      const float *input_sample_ptr = input_ptr + (i * sample_size + j) * channel;
      float *output_sample_ptr = output_ptr + (i * sample_size + j) * channel;
      const float *new_scale_ptr = new_scale.data();
      const float *new_offset_ptr = new_offset.data();
      for (index_t cb = 0; cb < ch_blks; ++cb) {
        float32x4_t new_scale_f = vld1q_f32(new_scale_ptr);
        float32x4_t new_offset_f = vld1q_f32(new_offset_ptr);
        float32x4_t input_f = vld1q_f32(input_sample_ptr);
        float32x4_t output_f = vfmaq_f32(new_offset_f, input_f, new_scale_f);
        vst1q_f32(output_sample_ptr, output_f);

        input_sample_ptr += 4;
        output_sample_ptr += 4;
        new_scale_ptr += 4;
        new_offset_ptr += 4;
      }
      for (index_t c = (ch_blks << 2); c < channel; ++c) {
        *output_sample_ptr = new_scale[c] * *input_sample_ptr + new_offset[c];
        ++output_sample_ptr;
        ++input_sample_ptr;
        ++new_scale_ptr;
        ++new_offset_ptr;
      }
    }
  }
};

}  // namespace kernels
}  // namespace mace
