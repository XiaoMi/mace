//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_DEPTH_TO_SPACE_H_
#define MACE_KERNELS_DEPTH_TO_SPACE_H_
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct DepthToSpaceOpFunctor {
  explicit DepthToSpaceOpFunctor(const int block_size, bool d2s)
      : block_size_(block_size), d2s_(d2s) {}
  void operator()(const Tensor *input, Tensor *output, StatsFuture *future) {
    const int batch_size = input->dim(0);
    const int input_height = input->dim(1);
    const int input_width = input->dim(2);
    const int input_depth = input->dim(3);

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
    std::vector<index_t> output_shape = {batch_size, output_height,
                                         output_width, output_depth};

    output->Resize(output_shape);

    Tensor::MappingGuard logits_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();

    if (d2s_) {
#pragma omp parallel for
      for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < output_height; ++h) {
          const int in_h = h / block_size_;
          const int offset_h = (h % block_size_);
          for (int w = 0; w < output_width; ++w) {
            const int in_w = w / block_size_;
            const int offset_w = w % block_size_;
            const int offset_d =
                (offset_h * block_size_ + offset_w) * output_depth;
            for (int d = 0; d < output_depth; ++d) {
              const int in_d = d + offset_d;
              const int o_index =
                  ((b * output_height + h) * output_width + w) * output_depth +
                  d;
              const int i_index =
                  ((b * input_height + in_h) * input_width + in_w) *
                      input_depth +
                  in_d;
              output_ptr[o_index] = input_ptr[i_index];
            }
          }
        }
      }
    } else {
#pragma omp parallel for
      for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < input_height; ++h) {
          const int out_h = h / block_size_;
          const int offset_h = (h % block_size_);
          for (int w = 0; w < input_width; ++w) {
            const int out_w = w / block_size_;
            const int offset_w = (w % block_size_);
            const int offset_d =
                (offset_h * block_size_ + offset_w) * input_depth;
            for (int d = 0; d < input_depth; ++d) {
              const int out_d = d + offset_d;
              const int o_index =
                  ((b * output_height + out_h) * output_width + out_w) *
                      output_depth +
                  out_d;
              const int i_index =
                  ((b * input_height + h) * input_width + w) * input_depth + d;
              output_ptr[o_index] = input_ptr[i_index];
            }
          }
        }
      }
    }
  }

  const int block_size_;
  bool d2s_;
};

template <typename T>
struct DepthToSpaceOpFunctor<DeviceType::OPENCL, T> {
  DepthToSpaceOpFunctor(const int block_size, bool d2s)
      : block_size_(block_size), d2s_(d2s) {}
  void operator()(const Tensor *input, Tensor *output, StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  const int block_size_;
  bool d2s_;
  std::vector<index_t> input_shape_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_DEPTH_TO_SPACE_H_
