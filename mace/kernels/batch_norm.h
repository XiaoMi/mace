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
struct BatchNormFunctorBase {
  BatchNormFunctorBase(const float variance_epsilon)
          :variance_epsilon_(variance_epsilon){}

  float variance_epsilon_;
};


template<DeviceType D, typename T>
struct BatchNormFunctor : public BatchNormFunctorBase<D, T> {
  BatchNormFunctor(const float variance_epsilon)
          :BatchNormFunctorBase<D, T>(variance_epsilon){}

  void operator()(const T* input,
                  const T* scale,
                  const T* offset,
                  const T* mean,
                  const T* var,
                  const TIndex n,
                  const TIndex channel,
                  const TIndex sample_size,
                  T* output) {
    // Batch normalization in the paper https://arxiv.org/abs/1502.03167 .
    // The calculation formula for inference is
    // Y = \frac{ \scale } { \sqrt{var+\variance_epsilon} } * X +
    //          ( \offset - \frac { \scale * mean } { \sqrt{var+\variance_epsilon} }
    // new_scale = \frac{ \scale } { \sqrt{var+\variance_epsilon} }
    // new_offset = \offset - mean * common_val;
    // Y = new_scale * X + new_offset;
    T new_scale, new_offset;
    for (TIndex c = 0; c < channel; ++c) {
      new_scale = scale[c] / std::sqrt(var[c] + this->variance_epsilon_);
      new_offset = offset[c] - mean[c] * new_scale;
      TIndex pos = c * sample_size;

      for (TIndex i = 0; i < n; ++i) {
        const T* input_sample_ptr = input + pos;
        T* output_sample_ptr = output + pos;
        for (TIndex j = 0; j < sample_size; ++j) {
          output_sample_ptr[j] = new_scale * input_sample_ptr[j] + new_offset;
        }
        pos += channel * sample_size;
      }
    }
  }

};

} //  namepsace kernels
} //  namespace mace

#endif //  MACE_KERNELS_BATCH_NORM_H_
