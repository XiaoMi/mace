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

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct WinogradTransformFunctor;

#ifdef MACE_ENABLE_OPENCL
class OpenCLWinogradTransformKernel {
 public:
  virtual MaceStatus Compute(
      OpKernelContext *context,
      const Tensor *input,
      Tensor *output,
      StatsFuture *future) = 0;
  MACE_VIRTUAL_EMPTY_DESTRUCTOR(OpenCLWinogradTransformKernel);
};
template<typename T>
struct WinogradTransformFunctor<DeviceType::GPU, T> : OpKernel {
  WinogradTransformFunctor(OpKernelContext *context,
                           const Padding &padding_type,
                           const std::vector<int> &paddings,
                           const int block_size);

  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future);

  std::unique_ptr<OpenCLWinogradTransformKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL


template<DeviceType D, typename T>
struct WinogradInverseTransformFunctor;

#ifdef MACE_ENABLE_OPENCL
class OpenCLWinogradInverseTransformKernel {
 public:
  virtual MaceStatus Compute(
      OpKernelContext *context,
      const std::vector<const Tensor*> &inputs,
      Tensor *output,
      StatsFuture *future) = 0;
  MACE_VIRTUAL_EMPTY_DESTRUCTOR(OpenCLWinogradInverseTransformKernel);
};
template <typename T>
struct WinogradInverseTransformFunctor<DeviceType::GPU, T> : OpKernel {
  WinogradInverseTransformFunctor(OpKernelContext *context,
                                  const ActivationType activation,
                                  const float relux_max_limit,
                                  const int block_size);

  MaceStatus operator()(const std::vector<const Tensor *> &inputs,
                        Tensor *output,
                        StatsFuture *future);

  std::unique_ptr<OpenCLWinogradInverseTransformKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_WINOGRAD_TRANSFORM_H_
