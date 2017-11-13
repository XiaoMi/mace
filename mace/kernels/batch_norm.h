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
    const index_t n = input->dim(0);
    const index_t channel = input->dim(1);
    const index_t sample_size = input->dim(2) * input->dim(3);

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
      index_t pos = c * sample_size;

      for (index_t i = 0; i < n; ++i) {
        const T *input_sample_ptr = input_ptr + pos;
        T *output_sample_ptr = output_ptr + pos;
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
    const Tensor *input,
    const Tensor *scale,
    const Tensor *offset,
    const Tensor *mean,
    const Tensor *var,
    const Tensor *epsilon,
    Tensor *output);

template <typename T>
struct BatchNormFunctor<DeviceType::OPENCL, T> {
  void operator()(
      const Tensor *input,
      const Tensor *scale,
      const Tensor *offset,
      const Tensor *mean,
      const Tensor *var,
      const Tensor *epsilon,
      Tensor *output);
};

}  //  namepsace kernels
}  //  namespace mace

#endif  //  MACE_KERNELS_BATCH_NORM_H_
