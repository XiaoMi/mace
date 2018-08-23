// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MACE_KERNELS_WINOGRAD_TRANSFORM_H_
#define MACE_KERNELS_WINOGRAD_TRANSFORM_H_

#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/activation.h"
#include "mace/kernels/conv_pool_2d_util.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

struct WinogradTransformFunctorBase {
  WinogradTransformFunctorBase(const Padding &padding_type,
                               const std::vector<int> &paddings,
                               const int block_size)
      : strides_({1, 1}),
        dilations_({1, 1}),
        padding_type_(padding_type),
        paddings_(paddings),
        wino_blk_size_(block_size) {}

  const std::vector<int> strides_;    // [stride_h, stride_w]
  const std::vector<int> dilations_;  // [dilation_h, dilation_w]
  Padding padding_type_;
  std::vector<int> paddings_;
  const int wino_blk_size_;
};

template<DeviceType D, typename T>
struct WinogradTransformFunctor : WinogradTransformFunctorBase {
  WinogradTransformFunctor(const Padding &padding_type,
                           const std::vector<int> &paddings,
                           const int block_size)
      : WinogradTransformFunctorBase(padding_type, paddings, block_size) {}

  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(input);
    MACE_UNUSED(output);
    MACE_UNUSED(future);
    MACE_NOT_IMPLEMENTED;
    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template<typename T>
struct WinogradTransformFunctor<DeviceType::GPU, T>
    : WinogradTransformFunctorBase {
  WinogradTransformFunctor(const Padding &padding_type,
                           const std::vector<int> &paddings,
                           const int block_size)
      : WinogradTransformFunctorBase(padding_type, paddings, block_size) {}

  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};
#endif  // MACE_ENABLE_OPENCL

struct WinogradInverseTransformFunctorBase {
  WinogradInverseTransformFunctorBase(const ActivationType activation,
                                      const float relux_max_limit,
                                      const int block_size)
      : wino_blk_size_(block_size),
        activation_(activation),
        relux_max_limit_(relux_max_limit) {}

  const int wino_blk_size_;
  const ActivationType activation_;
  const float relux_max_limit_;
};

template<DeviceType D, typename T>
struct WinogradInverseTransformFunctor : WinogradInverseTransformFunctorBase {
  WinogradInverseTransformFunctor(const ActivationType activation,
                                  const float relux_max_limit,
                                  const int block_size)
      : WinogradInverseTransformFunctorBase(
            activation, relux_max_limit, block_size) {}

  MaceStatus operator()(const std::vector<const Tensor*> &inputs,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(inputs);
    MACE_UNUSED(output);
    MACE_UNUSED(future);
    MACE_NOT_IMPLEMENTED;
    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
struct WinogradInverseTransformFunctor<DeviceType::GPU, T>
    : WinogradInverseTransformFunctorBase {
  WinogradInverseTransformFunctor(const ActivationType activation,
                                  const float relux_max_limit,
                                  const int block_size)
      : WinogradInverseTransformFunctorBase(
            activation, relux_max_limit, block_size) {}

  MaceStatus operator()(const std::vector<const Tensor*> &inputs,
                  Tensor *output,
                  StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_WINOGRAD_TRANSFORM_H_
