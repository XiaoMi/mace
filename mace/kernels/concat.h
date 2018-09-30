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

#ifndef MACE_KERNELS_CONCAT_H_
#define MACE_KERNELS_CONCAT_H_

#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/core/types.h"
#include "mace/kernels/kernel.h"
#include "mace/public/mace.h"
#include "mace/utils/quantize.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct ConcatFunctor : OpKernel {
  ConcatFunctor(OpKernelContext *context, const int32_t axis)
      : OpKernel(context), axis_(axis) {}

  MaceStatus operator()(const std::vector<const Tensor *> &input_list,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);
    const Tensor *input0 = input_list.front();
    const size_t inputs_count = input_list.size();

    std::vector<index_t> output_shape(input0->shape());
    index_t inner_size = 1;
    for (int i = 0; i < axis_; ++i) {
      inner_size *= output_shape[i];
    }
    std::vector<index_t> outer_sizes(inputs_count, 0);
    outer_sizes[0] = input0->size() / inner_size;
    for (size_t i = 1; i < inputs_count; ++i) {
      const Tensor *input = input_list[i];
      MACE_CHECK(input->dim_size() == input0->dim_size(),
                 "Ranks of all input tensors must be same.");
      for (int j = 0; j < input->dim_size(); ++j) {
        if (j == axis_) {
          continue;
        }
        MACE_CHECK(input->dim(j) == input0->dim(j),
                   "Dimensions of inputs should equal except axis.");
      }
      outer_sizes[i] = input->size() / inner_size;
      output_shape[axis_] += input->dim(axis_);
    }
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    T *output_ptr = output->mutable_data<T>();

    std::vector<const T *> input_ptrs(input_list.size(), nullptr);
    for (size_t i = 0; i < inputs_count; ++i) {
      input_ptrs[i] = input_list[i]->data<T>();
    }
    for (int inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
      for (size_t i = 0; i < inputs_count; ++i) {
        if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
          memcpy(output_ptr, input_ptrs[i], outer_sizes[i] * sizeof(T));
          output_ptr += outer_sizes[i];
          input_ptrs[i] += outer_sizes[i];
        } else {
          for (index_t k = 0; k < outer_sizes[i]; ++k) {
            *output_ptr++ = *input_ptrs[i]++;
          }
        }
      }
    }

    return MACE_SUCCESS;
  }

  int32_t axis_;
};

template<>
struct ConcatFunctor<DeviceType::CPU, uint8_t> : OpKernel {
  ConcatFunctor(OpKernelContext *context, const int32_t axis)
      : OpKernel(context), axis_(axis) {}

  MaceStatus operator()(const std::vector<const Tensor *> &input_list,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);
    MACE_CHECK(output->scale() != 0);
    const Tensor *input0 = input_list.front();
    const size_t inputs_count = input_list.size();

    std::vector<index_t> output_shape(input0->shape());
    index_t inner_size = 1;
    for (int i = 0; i < axis_; ++i) {
      inner_size *= output_shape[i];
    }
    std::vector<index_t> outer_sizes(inputs_count, 0);
    outer_sizes[0] = input0->size() / inner_size;
    for (size_t i = 1; i < inputs_count; ++i) {
      const Tensor *input = input_list[i];
      MACE_CHECK(input->dim_size() == input0->dim_size(),
                 "Ranks of all input tensors must be same.");
      for (int j = 0; j < input->dim_size(); ++j) {
        if (j == axis_) {
          continue;
        }
        MACE_CHECK(input->dim(j) == input0->dim(j),
                   "Dimensions of inputs should equal except axis.");
      }
      outer_sizes[i] = input->size() / inner_size;
      output_shape[axis_] += input->dim(axis_);
    }
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    auto output_ptr = output->mutable_data<uint8_t>();

    std::vector<const uint8_t *> input_ptrs(input_list.size(), nullptr);
    for (size_t i = 0; i < inputs_count; ++i) {
      input_ptrs[i] = input_list[i]->data<uint8_t>();
    }

    for (int inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
      for (size_t i = 0; i < inputs_count; ++i) {
        if (input_list[i]->zero_point() == output->zero_point()
            && input_list[i]->scale() == output->scale()) {
          memcpy(output_ptr, input_ptrs[i], outer_sizes[i] * sizeof(uint8_t));
          output_ptr += outer_sizes[i];
          input_ptrs[i] += outer_sizes[i];
        } else {
          const float scale = input_list[i]->scale() / output->scale();
          const float offset =
              -input_list[i]->zero_point() * scale + output->zero_point();
          for (index_t k = 0; k < outer_sizes[i]; ++k) {
            float out = (*input_ptrs[i]) * scale + offset;
            *output_ptr = Saturate<uint8_t>(roundf(out));
            ++output_ptr;
            ++input_ptrs[i];
          }
        }
      }
    }

    return MACE_SUCCESS;
  }

  int32_t axis_;
};

#ifdef MACE_ENABLE_OPENCL
class OpenCLConcatKernel {
 public:
  virtual MaceStatus Compute(
      OpKernelContext *context,
      const std::vector<const Tensor *> &input_list,
      Tensor *output,
      StatsFuture *future) = 0;
  MACE_VIRTUAL_EMPTY_DESTRUCTOR(OpenCLConcatKernel);
};
template <typename T>
struct ConcatFunctor<DeviceType::GPU, T> : OpKernel {
  ConcatFunctor(OpKernelContext *context, const int32_t axis);

  MaceStatus operator()(const std::vector<const Tensor *> &input_list,
                        Tensor *output,
                        StatsFuture *future);

  std::unique_ptr<OpenCLConcatKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_CONCAT_H_
