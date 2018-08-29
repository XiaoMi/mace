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

#ifndef MACE_KERNELS_UNSTACK_H_
#define MACE_KERNELS_UNSTACK_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct UnstackFunctor {
  explicit UnstackFunctor(int axis) : axis_(axis) {}

  MaceStatus operator()(const Tensor *input,
                        const std::vector<Tensor *> &outputs,
                        StatsFuture *future) {
    std::vector<index_t> input_shape = input->shape();
    MACE_CHECK(axis_ >= -(input->dim_size()) && axis_ < input->dim_size(),
               "axis out of bound.");
    if (axis_ < 0) {
      axis_ += input->dim_size();
    }
    MACE_CHECK(outputs.size() == input_shape[axis_],
               "output size not equal input_shape[axis]");

    std::vector<index_t> output_shape = input_shape;
    output_shape.erase(output_shape.begin() + axis_);

    std::vector<T *> output_data(outputs.size(), nullptr);
    for (size_t i = 0; i < input_shape[axis_]; ++i) {
      MACE_RETURN_IF_ERROR(outputs[i]->Resize(output_shape));
      output_data[i] = outputs[i]->mutable_data<T>();
    }
    const T *input_data = input->data<T>();

    index_t high_dim_elem_size =
        std::accumulate(input_shape.begin(), input_shape.begin() + axis_, 1,
                        std::multiplies<index_t>());
    index_t low_dim_elem_size =
        std::accumulate(input_shape.begin() + axis_ + 1, input_shape.end(), 1,
                        std::multiplies<index_t>());

    for (index_t h = 0; h < high_dim_elem_size; ++h) {
      int input_idx = h * input_shape[axis_] * low_dim_elem_size;
      int output_idx = h * low_dim_elem_size;
      for (size_t i = 0; i < input_shape[axis_]; ++i) {
        memcpy(output_data[i] + output_idx, input_data + input_idx,
               sizeof(T) * low_dim_elem_size);
        input_idx += low_dim_elem_size;
      }
    }

    SetFutureDefaultWaitFn(future);
    return MACE_SUCCESS;
  }

  int axis_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_UNSTACK_H_
