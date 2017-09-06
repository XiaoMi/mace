//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_BATCH_NORM_H_
#define MACE_KERNELS_BATCH_NORM_H_

#include "mace/core/tensor.h"
#include "mace/proto/mace.pb.h"

namespace mace {
namespace kernels {


template<DeviceType D>
struct BatchNormFunctor {
  void operator()(const float* input,
                  const float* scale,
                  const float* offset,
                  const float* mean,
                  const float* var,
                  const int n,
                  const int channel,
                  const int sample_size,
                  const float variance_epsilon,
                  float* output) ;
};

template<>
struct BatchNormFunctor<DeviceType::CPU> {
  void operator()(const float* input,
                  const float* scale,
                  const float* offset,
                  const float* mean,
                  const float* var,
                  const int n,
                  const int channel,
                  const int sample_size,
                  const float variance_epsilon,
                  float* output) {
    // Batch normalization in the paper https://arxiv.org/abs/1502.03167 .
    // The calculation formula for inference is
    // Y = \frac{ \scale } { \sqrt{var+\variance_epsilon} } * X +
    //          ( \offset - \frac { \scale * mean } { \sqrt{var+\variance_epsilon} }
    // new_scale = \frac{ \scale } { \sqrt{var+\variance_epsilon} }
    // new_offset = \offset - mean * common_val;
    // Y = new_scale * X + new_offset;
    float new_scale, new_offset;
    for (int c = 0; c < channel; ++c) {
      new_scale = scale[c] / std::sqrt(var[c] + variance_epsilon);
      new_offset = offset[c] - mean[c] * new_scale;

      for (int i = 0; i < n; ++i) {
        int pos = i * channel * sample_size + c * sample_size;
        const float* input_sample_ptr = input + pos;
        float* output_sample_ptr = output + pos;
        for (int j = 0; j < sample_size; ++j) {
          output_sample_ptr[j] = new_scale * input_sample_ptr[j] + new_offset;
        }
      }
    }
  }
};

} //  namepsace kernels
} //  namespace mace

#endif //  MACE_KERNELS_BATCH_NORM_H_
