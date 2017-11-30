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
  void operator()(const Tensor *input,
                  const Tensor *scale,
                  const Tensor *offset,
                  const Tensor *mean,
                  const Tensor *var,
                  const Tensor *epsilon,
                  Tensor *output) {
    // Batch normalization in the paper https://arxiv.org/abs/1502.03167 .
    // The calculation formula for inference is
    // Y = \frac{ \scale } { \sqrt{var+\variance_epsilon} } * X +
    //          ( \offset - \frac { \scale * mean } {
    //          \sqrt{var+\variance_epsilon} }
    // new_scale = \frac{ \scale } { \sqrt{var+\variance_epsilon} }
    // new_offset = \offset - mean * common_val;
    // Y = new_scale * X + new_offset;
    const index_t ch_pixel_size = input->dim(0) * input->dim(1) * input->dim(2);
    const index_t channel = input->dim(3);

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard scale_mapper(scale);
    Tensor::MappingGuard offset_mapper(offset);
    Tensor::MappingGuard mean_mapper(mean);
    Tensor::MappingGuard var_mapper(var);
    Tensor::MappingGuard epsilon_mapper(epsilon);
    Tensor::MappingGuard output_mapper(output);

    const T *input_ptr = input->data<T>();
    const T *scale_ptr = scale->data<T>();
    const T *offset_ptr = offset->data<T>();
    const T *mean_ptr = mean->data<T>();
    const T *var_ptr = var->data<T>();
    const T *epsilon_ptr = epsilon->data<T>();
    T *output_ptr = output->mutable_data<T>();

#pragma omp parallel for
    for (index_t c = 0; c < channel; ++c) {
      T new_scale = scale_ptr[c] / std::sqrt(var_ptr[c] + *epsilon_ptr);
      T new_offset = offset_ptr[c] - mean_ptr[c] * new_scale;
      index_t pos = c;

      for (index_t i = 0; i < ch_pixel_size; ++i) {
        output_ptr[pos] = new_scale * input_ptr[pos] + new_offset;
        pos += channel;
      }
    }
  }
};

template <>
void BatchNormFunctor<DeviceType::NEON, float>::operator()(
    const Tensor *input,
    const Tensor *scale,
    const Tensor *offset,
    const Tensor *mean,
    const Tensor *var,
    const Tensor *epsilon,
    Tensor *output);

template <>
void BatchNormFunctor<DeviceType::OPENCL, float>::operator()(
      const Tensor *input,
      const Tensor *scale,
      const Tensor *offset,
      const Tensor *mean,
      const Tensor *var,
      const Tensor *epsilon,
      Tensor *output);

}  //  namepsace kernels
}  //  namespace mace

#endif  //  MACE_KERNELS_BATCH_NORM_H_
