//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_BATCH_NORM_H_
#define MACE_KERNELS_BATCH_NORM_H_

#include "mace/core/future.h"
#include "mace/core/public/mace.h"
#include "mace/core/tensor.h"
#include "mace/kernels/activation.h"
#include "mace/core/runtime/opencl/cl2_header.h"

namespace mace {
namespace kernels {

struct BatchNormFunctorBase {
  BatchNormFunctorBase(bool folded_constant,
                       const ActivationType activation,
                       const float relux_max_limit,
                       const float prelu_alpha)
      : folded_constant_(folded_constant),
        activation_(activation),
        relux_max_limit_(relux_max_limit),
        prelu_alpha_(prelu_alpha) {}

  const bool folded_constant_;
  const ActivationType activation_;
  const float relux_max_limit_;
  const float prelu_alpha_;
};

template <DeviceType D, typename T>
struct BatchNormFunctor : BatchNormFunctorBase {
  BatchNormFunctor(const bool folded_constant,
                   const ActivationType activation,
                   const float relux_max_limit,
                   const float prelu_alpha)
      : BatchNormFunctorBase(
            folded_constant, activation, relux_max_limit, prelu_alpha) {}

  void operator()(const Tensor *input,
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
    const index_t height = input->dim(1);
    const index_t width = input->dim(2);
    const index_t channels = input->dim(3);

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard scale_mapper(scale);
    Tensor::MappingGuard offset_mapper(offset);
    Tensor::MappingGuard output_mapper(output);

    const T *input_ptr = input->data<T>();
    const T *scale_ptr = scale->data<T>();
    const T *offset_ptr = offset->data<T>();
    T *output_ptr = output->mutable_data<T>();

    vector<T> new_scale;
    vector<T> new_offset;
    if (!folded_constant_) {
      new_scale.resize(channels);
      new_offset.resize(channels);
      Tensor::MappingGuard mean_mapper(mean);
      Tensor::MappingGuard var_mapper(var);
      const T *mean_ptr = mean->data<T>();
      const T *var_ptr = var->data<T>();
#pragma omp parallel for
      for (index_t c = 0; c < channels; ++c) {
        new_scale[c] = scale_ptr[c] / std::sqrt(var_ptr[c] + epsilon);
        new_offset[c] = offset_ptr[c] - mean_ptr[c] * new_scale[c];
      }
    }

    index_t pos = 0;

#pragma omp parallel for
    for (index_t n = 0; n < batch; ++n) {
      for (index_t h = 0; h < height; ++h) {
        for (index_t w = 0; w < width; ++w) {
          for (index_t c = 0; c < channels; ++c) {
            if (folded_constant_) {
              output_ptr[pos] = scale_ptr[c] * input_ptr[pos] + offset_ptr[c];
            } else {
              output_ptr[pos] = new_scale[c] * input_ptr[pos] + new_offset[c];
            }
            ++pos;
          }
        }
      }
    }
    DoActivation(output_ptr, output_ptr, output->NumElements(), activation_,
                 relux_max_limit_, prelu_alpha_);
  }
};

template <>
void BatchNormFunctor<DeviceType::NEON, float>::operator()(const Tensor *input,
                                                           const Tensor *scale,
                                                           const Tensor *offset,
                                                           const Tensor *mean,
                                                           const Tensor *var,
                                                           const float epsilon,
                                                           Tensor *output,
                                                           StatsFuture *future);

template <typename T>
struct BatchNormFunctor<DeviceType::OPENCL, T> : BatchNormFunctorBase {
  BatchNormFunctor(const bool folded_constant,
                   const ActivationType activation,
                   const float relux_max_limit,
                   const float prelu_alpha)
      : BatchNormFunctorBase(
            folded_constant, activation, relux_max_limit, prelu_alpha) {}
  void operator()(const Tensor *input,
                  const Tensor *scale,
                  const Tensor *offset,
                  const Tensor *mean,
                  const Tensor *var,
                  const float epsilon,
                  Tensor *output,
                  StatsFuture *future);
  cl::Kernel kernel_;
};

}  //  namepsace kernels
}  //  namespace mace

#endif  //  MACE_KERNELS_BATCH_NORM_H_
