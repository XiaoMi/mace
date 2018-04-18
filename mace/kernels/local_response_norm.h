//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_LOCAL_RESPONSE_NORM_H_
#define MACE_KERNELS_LOCAL_RESPONSE_NORM_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct LocalResponseNormFunctor {
  void operator()(const Tensor *input,
                  int depth_radius,
                  float bias,
                  float alpha,
                  float beta,
                  Tensor *output,
                  StatsFuture *future) {
    const index_t batch = input->dim(0);
    const index_t height = input->dim(1);
    const index_t width = input->dim(2);
    const index_t channels = input->dim(3);

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard output_mapper(output);

    const T *input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();

    const int elements = batch * height * width;

#pragma omp parallel for collapse(2)
    for (index_t i = 0; i < elements; ++i) {
      for (index_t c = 0; c < channels; ++c) {
        const int begin_input_c = std::max(static_cast<index_t>(0),
                                           c - depth_radius);
        const int end_input_c = std::min(channels, c + depth_radius + 1);
        index_t pos = i * channels;
        float accum = 0.f;
        for (int input_c = begin_input_c; input_c < end_input_c; ++input_c) {
          const float input_val = input_ptr[pos + input_c];
          accum += input_val * input_val;
        }
        const float multiplier = std::pow(bias + alpha * accum, -beta);
        output_ptr[pos + c] = input_ptr[pos + c] * multiplier;
      }
    }
  }
};

template <>
struct LocalResponseNormFunctor<DeviceType::NEON, float> {
  void operator()(const Tensor *input,
                  int depth_radius,
                  float bias,
                  float alpha,
                  float beta,
                  Tensor *output,
                  StatsFuture *future);
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_LOCAL_RESPONSE_NORM_H_
