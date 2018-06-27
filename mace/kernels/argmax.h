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

#ifndef MACE_KERNELS_ARGMAX_H_
#define MACE_KERNELS_ARGMAX_H_

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct ArgMaxFunctor {
  MaceStatus operator()(const Tensor *input,
                        const Tensor *axis,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);

    MACE_CHECK(input->dim_size() > 0, "ArgMax input should not be a scalar");
    MACE_CHECK(axis->dim_size() == 0, "Mace argmax only supports scalar axis");
    Tensor::MappingGuard axis_guard(axis);
    int axis_value = axis->data<int32_t>()[0];
    if (axis_value < 0) {
      axis_value += input->dim_size();
    }
    MACE_CHECK(axis_value == input->dim_size() - 1,
               "Mace argmax only supports last dimension as axis");

    std::vector<index_t> output_shape(input->dim_size() - 1);
    for (index_t d = 0; d < input->dim_size() - 1; ++d) {
      output_shape[d] = input->dim(d < axis_value ? d : d + 1);
    }
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    auto input_data = input->data<T>();
    auto output_data = output->mutable_data<int32_t>();

    index_t outer_size = output->size();
    index_t inner_size = input->dim(axis_value);

#pragma omp parallel for
    for (index_t i = 0; i < outer_size; ++i) {
      int idx = 0;
      T max_value = std::numeric_limits<T>::lowest();
      const T *input_ptr = input_data + i * inner_size;
      for (index_t j = 0; j < inner_size; ++j) {
        if (input_ptr[j] > max_value) {
          max_value = input_ptr[j];
          idx = j;
        }
      }
      output_data[i] = idx;
    }

    return MACE_SUCCESS;
  }
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_ARGMAX_H_
