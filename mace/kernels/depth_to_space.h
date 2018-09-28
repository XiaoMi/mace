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

#ifndef MACE_KERNELS_DEPTH_TO_SPACE_H_
#define MACE_KERNELS_DEPTH_TO_SPACE_H_
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/kernel.h"
#include "mace/public/mace.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

template<DeviceType D, typename T>
struct DepthToSpaceOpFunctor : OpKernel {
  DepthToSpaceOpFunctor(OpKernelContext *context,
                        const int block_size)
      : OpKernel(context), block_size_(block_size) {}
  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);
    const index_t batch_size = input->dim(0);
    const index_t input_depth = input->dim(1);
    const index_t input_height = input->dim(2);
    const index_t input_width = input->dim(3);

    MACE_CHECK(input_depth % (block_size_ * block_size_) == 0,
               "input depth should be dividable by block_size * block_size",
               input_depth);

    const index_t output_depth = input_depth / (block_size_ * block_size_);
    const index_t output_width = input_width * block_size_;
    const index_t output_height = input_height * block_size_;
    std::vector<index_t> output_shape = {batch_size, output_depth,
                                         output_height, output_width};

    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard logits_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();

#pragma omp parallel for
    for (index_t b = 0; b < batch_size; ++b) {
      for (index_t d = 0; d < output_depth; ++d) {
        for (index_t h = 0; h < output_height; ++h) {
          const index_t in_h = h / block_size_;
          const index_t offset_h = (h % block_size_);
          for (int w = 0; w < output_width; ++w) {
            const index_t in_w = w / block_size_;
            const index_t offset_w = w % block_size_;
            const index_t offset_d =
                (offset_h * block_size_ + offset_w) * output_depth;

            const index_t in_d = d + offset_d;
            const index_t o_index =
                ((b * output_depth + d) * output_height + h) * output_width
                    + w;
            const index_t i_index =
                ((b * input_depth + in_d) * input_height + in_h) * input_width
                    + in_w;
            output_ptr[o_index] = input_ptr[i_index];
          }
        }
      }
    }


    return MACE_SUCCESS;
  }

  const int block_size_;
};

#ifdef MACE_ENABLE_OPENCL
class OpenCLDepthToSpaceKernel {
 public:
  virtual MaceStatus Compute(
      OpKernelContext *context,
      const Tensor *input,
      Tensor *output,
      StatsFuture *future) = 0;
  MACE_VIRTUAL_EMPTY_DESTRUCTOR(OpenCLDepthToSpaceKernel);
};
template<typename T>
struct DepthToSpaceOpFunctor<DeviceType::GPU, T> : OpKernel {
  DepthToSpaceOpFunctor(OpKernelContext *context,
                        const int block_size);
  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future);

  std::unique_ptr<OpenCLDepthToSpaceKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_DEPTH_TO_SPACE_H_
