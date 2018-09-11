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

#ifndef MACE_KERNELS_LSTMCELL_H_
#define MACE_KERNELS_LSTMCELL_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"
#include "mace/kernels/kernel.h"

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct LSTMCellFunctor;

template <typename T>
struct LSTMCellFunctor<DeviceType::GPU, T> : OpKernel{
  LSTMCellFunctor(OpKernelContext *context, T forget_bias)
      : OpKernel(context),
        forget_bias_(static_cast<T>(forget_bias)) {}
  MaceStatus operator()(const Tensor *input,
                        const Tensor *pre_output,
                        const Tensor *weight,
                        const Tensor *bias,
                        const Tensor *pre_cell,
                        Tensor *cell,
                        Tensor *output,
                        StatsFuture *future);

  T forget_bias_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_LSTMCELL_H_
