//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_BATCH_NORM_H_
#define MACE_KERNELS_BATCH_NORM_H_

#include "mace/core/tensor.h"
#include "mace/proto/mace.pb.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct BatchNormFunctor {
  float variance_epsilon_;

  BatchNormFunctor(const float variance_epsilon)
      : variance_epsilon_(variance_epsilon) {}

  void operator()(const T* input,
                  const T* scale,
                  const T* offset,
                  const T* mean,
                  const T* var,
                  const index_t n,
                  const index_t channel,
                  const index_t sample_size,
                  T* output) {
    // Batch normalization in the paper https://arxiv.org/abs/1502.03167 .
    // The calculation formula for inference is
    // Y = \frac{ \scale } { \sqrt{var+\variance_epsilon} } * X +
    //          ( \offset - \frac { \scale * mean } {
    //          \sqrt{var+\variance_epsilon} }
    // new_scale = \frac{ \scale } { \sqrt{var+\variance_epsilon} }
    // new_offset = \offset - mean * common_val;
    // Y = new_scale * X + new_offset;
    T new_scale, new_offset;
    for (index_t c = 0; c < channel; ++c) {
      new_scale = scale[c] / std::sqrt(var[c] + this->variance_epsilon_);
      new_offset = offset[c] - mean[c] * new_scale;
      index_t pos = c * sample_size;

      for (index_t i = 0; i < n; ++i) {
        const T* input_sample_ptr = input + pos;
        T* output_sample_ptr = output + pos;
        for (index_t j = 0; j < sample_size; ++j) {
          output_sample_ptr[j] = new_scale * input_sample_ptr[j] + new_offset;
        }
        pos += channel * sample_size;
      }
    }
  }
};

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
    float* output);

}  //  namepsace kernels
}  //  namespace mace

#endif  //  MACE_KERNELS_BATCH_NORM_H_
