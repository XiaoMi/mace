//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_FULLY_CONNECTED_H_
#define MACE_KERNELS_FULLY_CONNECTED_H_

#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"
#include "mace/kernels/activation.h"
#include "mace/kernels/opencl/helper.h"

namespace mace {
namespace kernels {

struct FullyConnectedBase {
  FullyConnectedBase(const BufferType weight_type,
                     const ActivationType activation,
                     const float relux_max_limit)
      : weight_type_(weight_type),
        activation_(activation),
        relux_max_limit_(relux_max_limit) {}

  const int weight_type_;
  const ActivationType activation_;
  const float relux_max_limit_;
};

template <DeviceType D, typename T>
struct FullyConnectedFunctor : FullyConnectedBase {
  FullyConnectedFunctor(const BufferType weight_type,
                        const ActivationType activation,
                        const float relux_max_limit)
      : FullyConnectedBase(weight_type, activation, relux_max_limit) {}

  void operator()(const Tensor *input,
                  const Tensor *weight,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future) {
    std::vector<index_t> output_shape = {input->dim(0), 1, 1, weight->dim(0)};
    output->Resize(output_shape);
    const index_t N = output->dim(0);
    const index_t input_size = weight->dim(1);
    const index_t output_size = weight->dim(0);
    Tensor::MappingGuard guard_input(input);
    Tensor::MappingGuard guard_weight(weight);
    Tensor::MappingGuard guard_bias(bias);
    Tensor::MappingGuard guard_output(output);
    const T *input_ptr = input->data<T>();
    const T *weight_ptr = weight->data<T>();
    const T *bias_ptr = bias == nullptr ? nullptr : bias->data<T>();
    T *output_ptr = output->mutable_data<T>();

#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
      for (int out_idx = 0; out_idx < output_size; ++out_idx) {
        T sum = 0;
        if (bias_ptr != nullptr) sum = bias_ptr[out_idx];
        index_t input_offset = i * input_size;
        index_t weight_offset = out_idx * input_size;
        for (int in_idx = 0; in_idx < input_size; ++in_idx) {
          sum += input_ptr[input_offset] * weight_ptr[weight_offset];
          input_offset++;
          weight_offset++;
        }
        output_ptr[i * output_size + out_idx] = sum;
      }
    }

    DoActivation(output_ptr, output_ptr, output->size(), activation_,
                 relux_max_limit_);
  }
};

template <>
struct FullyConnectedFunctor<DeviceType::NEON, float> : FullyConnectedBase {
  FullyConnectedFunctor(const BufferType weight_type,
                        const ActivationType activation,
                        const float relux_max_limit)
    : FullyConnectedBase(weight_type, activation, relux_max_limit) {}

  void operator()(const Tensor *input,
                  const Tensor *weight,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future);
};

template <typename T>
struct FullyConnectedFunctor<DeviceType::OPENCL, T> : FullyConnectedBase {
  FullyConnectedFunctor(const BufferType weight_type,
                        const ActivationType activation,
                        const float relux_max_limit)
      : FullyConnectedBase(weight_type, activation, relux_max_limit) {}

  void operator()(const Tensor *input,
                  const Tensor *weight,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future);

  cl::Kernel kernel_;
  std::vector<uint32_t> gws_;
  std::vector<uint32_t> lws_;
  std::vector<index_t> input_shape_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_FULLY_CONNECTED_H_
