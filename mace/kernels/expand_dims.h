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

#ifndef MACE_KERNELS_EXPAND_DIMS_H_
#define MACE_KERNELS_EXPAND_DIMS_H_

#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

struct ExpandDimsBase {
  explicit ExpandDimsBase(int axis) : axis_(axis) {}

  int axis_;
};

template <DeviceType D, typename T>
struct ExpandDimsFunctor;

template <typename T>
struct ExpandDimsFunctor<DeviceType::CPU, T> : ExpandDimsBase {
  explicit ExpandDimsFunctor(int axis) : ExpandDimsBase(axis) {}

  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);

    index_t input_dims_size = input->dim_size();
    if ( axis_ < 0 ) {
      axis_ += input_dims_size + 1;
    }
    MACE_CHECK(axis_ >= 0 && axis_ <= input_dims_size,
               "axis is out of bound: ", axis_);
    const std::vector<index_t> input_shape = input->shape();
    std::vector<index_t> output_shape;
    output_shape.insert(output_shape.end(), input_shape.begin(),
                        input_shape.begin() + axis_);
    output_shape.insert(output_shape.end(), 1);
    output_shape.insert(output_shape.end(), input_shape.begin() + axis_,
                        input_shape.end());

    output->ReuseTensorBuffer(*input);
    output->Reshape(output_shape);

    return MACE_SUCCESS;
  }
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_EXPAND_DIMS_H_
