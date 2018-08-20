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

#ifndef MACE_KERNELS_SPLIT_H_
#define MACE_KERNELS_SPLIT_H_

#include <memory>
#include <functional>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/core/types.h"
#include "mace/public/mace.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

struct SplitFunctorBase {
  explicit SplitFunctorBase(const int32_t axis) : axis_(axis) {}

  int32_t axis_;
};

template<DeviceType D, typename T>
struct SplitFunctor : SplitFunctorBase {
  explicit SplitFunctor(const int32_t axis) : SplitFunctorBase(axis) {}

  MaceStatus operator()(const Tensor *input,
                  const std::vector<Tensor *> &output_list,
                  StatsFuture *future) {
    MACE_UNUSED(future);
    const index_t input_channels = input->dim(axis_);
    const size_t outputs_count = output_list.size();
    const index_t output_channels = input_channels / outputs_count;
    std::vector<T *> output_ptrs(output_list.size(), nullptr);
    std::vector<index_t> output_shape(input->shape());
    output_shape[axis_] = output_channels;

    const index_t outer_size = std::accumulate(output_shape.begin(),
                                               output_shape.begin() + axis_,
                                               1,
                                               std::multiplies<index_t>());
    const index_t inner_size = std::accumulate(output_shape.begin() + axis_ + 1,
                                               output_shape.end(),
                                               1,
                                               std::multiplies<index_t>());
    for (size_t i= 0; i < outputs_count; ++i) {
      MACE_RETURN_IF_ERROR(output_list[i]->Resize(output_shape));
      output_ptrs[i] = output_list[i]->mutable_data<T>();
    }
    const T *input_ptr = input->data<T>();

#pragma omp parallel for
    for (int outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
      int input_idx = outer_idx * input_channels * inner_size;
      int output_idx = outer_idx * output_channels * inner_size;
      for (size_t i = 0; i < outputs_count; ++i) {
        if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
          memcpy(output_ptrs[i]+output_idx, input_ptr+input_idx,
                 output_channels * inner_size * sizeof(T));
        } else {
          for (index_t k = 0; k < output_channels * inner_size; ++k) {
            *(output_ptrs[i] + output_idx + k) = *(input_ptr + input_idx + k);
          }
        }
        input_idx += output_channels * inner_size;
      }
    }

    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template<typename T>
struct SplitFunctor<DeviceType::GPU, T> : SplitFunctorBase {
  explicit SplitFunctor(const int32_t axis) : SplitFunctorBase(axis) {}

  MaceStatus operator()(const Tensor *input,
                  const std::vector<Tensor *> &output_list,
                  StatsFuture *future);
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_SPLIT_H_
