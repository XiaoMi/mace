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

#ifndef MACE_KERNELS_RESHAPE_H_
#define MACE_KERNELS_RESHAPE_H_

#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct ReshapeFunctor {
  ReshapeFunctor() {}

  MaceStatus operator()(const Tensor *input,
                  const std::vector<index_t> &out_shape,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_UNUSED(future);
    output->ReuseTensorBuffer(*input);
    output->Reshape(out_shape);

    return MACE_SUCCESS;
  }
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_RESHAPE_H_
