//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_BATCH_NORM_H_
#define MACE_KERNELS_BATCH_NORM_H_

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "mace/core/future.h"
#include "mace/public/mace.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"
#include "mace/kernels/activation.h"

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

    std::vector<T> new_scale;
    std::vector<T> new_offset;
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

    const T *scale_data = folded_constant_ ? scale_ptr : new_scale.data();
    const T *offset_data = folded_constant_ ? offset_ptr : new_offset.data();

    const int elements = batch * height * width;
    constexpr int c_tile_size = 4;
    const int c_tiles = channels / c_tile_size;
    const index_t remains_start = c_tiles * c_tile_size;

    if (c_tiles > 0) {
#pragma omp parallel for collapse(2)
      for (index_t i = 0; i < elements; ++i) {
        for (int cb = 0; cb < c_tiles; ++cb) {
#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
          static_assert(c_tile_size == 4, "channels tile size must be 4");
          int c = cb * c_tile_size;
          int pos = i * channels + c;

          float32x4_t scales = vld1q_f32(scale_data + c);
          float32x4_t offsets = vld1q_f32(offset_data + c);
          float32x4_t in = vld1q_f32(input_ptr + pos);
          float32x4_t out = vfmaq_f32(offsets, scales, in);
          vst1q_f32(output_ptr + pos, out);
#else
          for (int ci = 0; ci < c_tile_size; ++ci) {
            int c = cb * c_tile_size + ci;
            index_t pos = i * channels + c;
            output_ptr[pos] = scale_data[c] * input_ptr[pos] + offset_data[c];
          }
#endif
        }
      }
    }
    if (remains_start < channels) {
#pragma omp parallel for collapse(2)
      for (index_t i = 0; i < elements; ++i) {
        for (index_t c = remains_start; c < channels; ++c) {
          index_t pos = i * channels + c;
          output_ptr[pos] = scale_data[c] * input_ptr[pos] + offset_data[c];
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

}  // namepsace kernels
}  // namespace mace

#endif  // MACE_KERNELS_BATCH_NORM_H_
