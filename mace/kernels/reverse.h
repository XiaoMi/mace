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

#ifndef MACE_KERNELS_REVERSE_H_
#define MACE_KERNELS_REVERSE_H_

#include <functional>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct ReverseFunctor;

template <typename T>
struct ReverseFunctor<DeviceType::CPU, T> {
  MaceStatus operator()(const Tensor *input,
                        const Tensor *axis,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_CHECK(axis->dim_size() == 1, "Only support reverse in one axis now");

    const int32_t *axis_data = axis->data<int32_t>();
    const index_t reverse_dim = *axis_data >= 0 ?
      *axis_data : *axis_data + input->dim_size();
    MACE_CHECK(reverse_dim >= 0 && reverse_dim < input->dim_size(),
               "axis must be in the range [-rank(input), rank(input))");

    const std::vector<index_t> input_shape = input->shape();
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    index_t high_dim_elem_size =
        std::accumulate(input_shape.begin(), input_shape.begin() + reverse_dim,
                        1, std::multiplies<index_t>());
    index_t low_dim_elem_size =
        std::accumulate(input_shape.begin() + reverse_dim + 1,
                        input_shape.end(), 1, std::multiplies<index_t>());

    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

    index_t reverse_size = input_shape[reverse_dim] * low_dim_elem_size;
    for (index_t h = 0; h < high_dim_elem_size; ++h) {
      int input_idx = h * reverse_size;
      int output_idx = input_idx + reverse_size;
      for (index_t i = 0; i < input_shape[reverse_dim]; ++i) {
        output_idx -= low_dim_elem_size;
        memcpy(output_data + output_idx, input_data + input_idx,
               sizeof(T) * low_dim_elem_size);
        input_idx += low_dim_elem_size;
      }
    }

    SetFutureDefaultWaitFn(future);
    return MACE_SUCCESS;
  }
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_REVERSE_H_
