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
#include "mace/public/mace.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

template<DeviceType D, typename T>
struct DepthToSpaceOpFunctor {
  explicit DepthToSpaceOpFunctor(const int block_size, bool d2s)
      : block_size_(block_size), d2s_(d2s) {}
  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);
    const index_t batch_size = input->dim(0);
    const index_t input_depth = input->dim(1);
    const index_t input_height = input->dim(2);
    const index_t input_width = input->dim(3);

    index_t output_depth, output_width, output_height;

    if (d2s_) {
      output_depth = input_depth / (block_size_ * block_size_);
      output_width = input_width * block_size_;
      output_height = input_height * block_size_;
    } else {
      output_depth = input_depth * block_size_ * block_size_;
      output_width = input_width / block_size_;
      output_height = input_height / block_size_;
    }
    std::vector<index_t> output_shape = {batch_size, output_depth,
                                         output_height, output_width};

    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard logits_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();

    if (d2s_) {
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
    } else {
#pragma omp parallel for
      for (index_t b = 0; b < batch_size; ++b) {
        for (index_t d = 0; d < input_depth; ++d) {
          for (index_t h = 0; h < input_height; ++h) {
            const index_t out_h = h / block_size_;
            const index_t offset_h = (h % block_size_);
            for (index_t w = 0; w < input_width; ++w) {
              const index_t out_w = w / block_size_;
              const index_t offset_w = (w % block_size_);
              const index_t offset_d =
                  (offset_h * block_size_ + offset_w) * input_depth;

              const index_t out_d = d + offset_d;
              const index_t o_index =
                  ((b * output_depth + out_d) * output_height + out_h)
                      * output_width + out_w;
              const index_t i_index =
                  ((b * input_depth + d) * input_height + h) * input_width
                      + w;
              output_ptr[o_index] = input_ptr[i_index];
            }
          }
        }
      }
    }

    return MACE_SUCCESS;
  }

  const int block_size_;
  bool d2s_;
};

#ifdef MACE_ENABLE_OPENCL
template<typename T>
struct DepthToSpaceOpFunctor<DeviceType::GPU, T> {
  DepthToSpaceOpFunctor(const int block_size, bool d2s)
      : block_size_(block_size), d2s_(d2s) {}
  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future);

  const int block_size_;
  bool d2s_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_DEPTH_TO_SPACE_H_
