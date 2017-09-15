//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/batch_norm.h"
#include <arm_neon.h>

namespace mace {
namespace kernels {

template <>
void BatchNormFunctor<DeviceType::NEON, float>::operator()(
    const float* input,
    const float* scale,
    const float* offset,
    const float* mean,
    const float* var,
    const index_t n,
    const index_t channel,
    const index_t sample_size,
    float* output) {
  // Batch normalization in the paper https://arxiv.org/abs/1502.03167 .
  // The calculation formula for inference is
  // Y = \frac{ \scale } { \sqrt{var+\variance_epsilon} } * X +
  //          ( \offset - \frac { \scale * mean } { \sqrt{var+\variance_epsilon}
  //          }
  // new_scale = \frac{ \scale } { \sqrt{var+\variance_epsilon} }
  // new_offset = \offset - mean * common_val;
  // Y = new_scale * X + new_offset;
  float new_scale, new_offset;
  index_t count = sample_size >> 2;
  index_t remain_count = sample_size - (count << 2);
  for (index_t c = 0; c < channel; ++c) {
    new_scale = scale[c] / std::sqrt(var[c] + this->variance_epsilon_);
    new_offset = offset[c] - mean[c] * new_scale;
    index_t pos = c * sample_size;

    float32x4_t new_scale_f = vdupq_n_f32(new_scale);
    float32x4_t new_offset_f = vdupq_n_f32(new_offset);
    for (index_t i = 0; i < n; ++i) {
      const float* input_sample_ptr = input + pos;
      float* output_sample_ptr = output + pos;

      for (index_t j = 0; j < count; ++j) {
        float32x4_t input_f = vld1q_f32(input_sample_ptr);
        float32x4_t output_f = new_offset_f;
        output_f = vfmaq_f32(output_f, input_f, new_scale_f);
        vst1q_f32(output_sample_ptr, output_f);
        input_sample_ptr += 4;
        output_sample_ptr += 4;
      }
      for (index_t j = 0; j < remain_count; ++j) {
        *output_sample_ptr = new_scale * *input_sample_ptr + new_offset;
        ++output_sample_ptr;
        ++input_sample_ptr;
      }
      pos += channel * sample_size;
    }
  }
};

}  // namespace kernels
}  //  namespace mace