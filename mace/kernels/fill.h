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

#ifndef MACE_KERNELS_FILL_H_
#define MACE_KERNELS_FILL_H_

#include <algorithm>
#include <functional>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"

namespace mace {
namespace kernels {

template <DeviceType D, class T>
struct FillFunctor;

template <>
struct FillFunctor<DeviceType::CPU, float> {
  FillFunctor() {}

  MaceStatus operator()(const Tensor *shape,
                        const Tensor *value,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);

    MACE_CHECK(shape->dim_size() == 1, "Shape must be 1-D");
    const index_t num_dims = shape->dim(0);
    Tensor::MappingGuard shape_guard(shape);
    const int32_t *shape_data = shape->data<int32_t>();

    std::vector<index_t> output_shape;
    for (index_t i = 0; i < num_dims; ++i) {
      MACE_CHECK(shape_data[i] > 0, "Shape must be non-negative: ",
                 shape_data[i]);
      output_shape.push_back(shape_data[i]);
    }

    Tensor::MappingGuard value_guard(value);
    const float *value_data = value->data<float>();

    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    Tensor::MappingGuard output_guard(output);
    float *output_data = output->mutable_data<float>();

    std::fill(output_data, output_data + output->size(), *value_data);

    return MACE_SUCCESS;
  }
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_FILL_H_
