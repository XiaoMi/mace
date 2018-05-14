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

#ifndef MACE_KERNELS_SPACE_TO_BATCH_H_
#define MACE_KERNELS_SPACE_TO_BATCH_H_

#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

struct SpaceToBatchFunctorBase {
  SpaceToBatchFunctorBase(const std::vector<int> &paddings,
                          const std::vector<int> &block_shape,
                          bool b2s)
      : paddings_(paddings.begin(), paddings.end()),
        block_shape_(block_shape.begin(), block_shape.end()),
        b2s_(b2s) {}

  std::vector<int> paddings_;
  std::vector<int> block_shape_;
  bool b2s_;
};

template <DeviceType D, typename T>
struct SpaceToBatchFunctor : SpaceToBatchFunctorBase {
  SpaceToBatchFunctor(const std::vector<int> &paddings,
                      const std::vector<int> &block_shape,
                      bool b2s)
      : SpaceToBatchFunctorBase(paddings, block_shape, b2s) {}

  void operator()(Tensor *space_tensor,
                  const std::vector<index_t> &output_shape,
                  Tensor *batch_tensor,
                  StatsFuture *future) {
    MACE_UNUSED(space_tensor);
    MACE_UNUSED(output_shape);
    MACE_UNUSED(batch_tensor);
    MACE_UNUSED(future);
    MACE_NOT_IMPLEMENTED;
  }
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
struct SpaceToBatchFunctor<DeviceType::GPU, T> : SpaceToBatchFunctorBase {
  SpaceToBatchFunctor(const std::vector<int> &paddings,
                      const std::vector<int> &block_shape,
                      bool b2s)
      : SpaceToBatchFunctorBase(paddings, block_shape, b2s) {}

  void operator()(Tensor *space_tensor,
                  const std::vector<index_t> &output_shape,
                  Tensor *batch_tensor,
                  StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> space_shape_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_SPACE_TO_BATCH_H_
