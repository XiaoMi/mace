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
    const index_t batchs = input->dim(0);
    const index_t height = input->dim(1);
    const index_t width = input->dim(2);
    const index_t height_width = height * width;
    const index_t channels = input->dim(3);

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

    vector<T> new_scale(channels);
    vector<T> new_offset(channels);

#pragma omp parallel for
    for (index_t c = 0; c < channels; ++c) {
      new_scale[c] = scale_ptr[c] / std::sqrt(var_ptr[c] + *epsilon_ptr);
      new_offset[c] = offset_ptr[c] - mean_ptr[c] * new_scale[c];
    }

    index_t pos = 0;

#pragma omp parallel for
    for (index_t n = 0; n < batchs; ++n) {
      for (index_t hb = 0; hb < height_width; ++hb) {
        for (index_t c = 0; c < channels; ++c) {
          output_ptr[pos] = new_scale[c] * input_ptr[pos] + new_offset[c];
          ++pos;
        }
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
  void operator()(const Tensor *input,
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
