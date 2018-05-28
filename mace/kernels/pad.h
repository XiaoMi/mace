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

#ifndef MACE_KERNELS_PAD_H_
#define MACE_KERNELS_PAD_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

struct PadFunctorBase {
  PadFunctorBase(const std::vector<int> &paddings,
                 const float constant_value)
      : paddings_(paddings), constant_value_(constant_value) {}

  std::vector<int> paddings_;
  float constant_value_;
};

template<DeviceType D, typename T>
struct PadFunctor : public PadFunctorBase {
  PadFunctor(const std::vector<int> &paddings,
             const float constant_value)
      : PadFunctorBase(paddings, constant_value) {}

  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);
    MACE_CHECK(
        this->paddings_.size() == static_cast<size_t>(input->dim_size()) * 2);
    auto input_shape = input->shape();
    MACE_RETURN_IF_ERROR(output->Resize({input_shape[0] + this->paddings_[0]
                                            + this->paddings_[1],
                                        input_shape[1] + this->paddings_[2]
                                            + this->paddings_[3],
                                        input_shape[2] + this->paddings_[4]
                                            + this->paddings_[5],
                                        input_shape[3] + this->paddings_[6]
                                            + this->paddings_[7]}));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    auto input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();
    std::fill(output_ptr, output_ptr + output->size(), this->constant_value_);

    const index_t batch = input->dim(0);
    const index_t channel = input->dim(1);
    const index_t height = input->dim(2);
    const index_t width = input->dim(3);
#pragma omp parallel for collapse(3)
    for (index_t b = 0; b < batch; ++b) {
      for (index_t c = 0; c < channel; ++c) {
        for (index_t h = 0; h < height; ++h) {
          const index_t in_offset = (((b * channel + c) * height) + h) * width;
          const index_t out_offset = (((b + this->paddings_[0]) * output->dim(1)
              + (c + this->paddings_[2])) * output->dim(2)
              + (h + this->paddings_[4])) * output->dim(3)
              + this->paddings_[6];
          memcpy(output_ptr + out_offset,
                 input_ptr + in_offset,
                 width * sizeof(T));
        }
      }
    }

    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
struct PadFunctor<DeviceType::GPU, T> : PadFunctorBase {
  PadFunctor(const std::vector<int> &paddings,
             const float constant_value)
      : PadFunctorBase(paddings, constant_value) {}

  MaceStatus operator()(const Tensor *input,
                  Tensor *output,
                  StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_PAD_H_
