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

#ifndef MACE_KERNELS_SPACE_TO_BATCH_H_
#define MACE_KERNELS_SPACE_TO_BATCH_H_

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

struct SpaceToBatchFunctorBase {
  SpaceToBatchFunctorBase(const std::vector<int> &paddings,
                          const std::vector<int> &block_shape,
                          bool b2s)
    : paddings_(paddings.begin(), paddings.end()),
      block_shape_(block_shape.begin(), block_shape.end()),
      b2s_(b2s) {
    MACE_CHECK(
      block_shape.size() == 2 && block_shape[0] > 1 && block_shape[1] > 1,
      "Block's shape should be 1D, and greater than 1");
    MACE_CHECK(paddings.size() == 4, "Paddings' shape should be 2D");
  }

  std::vector<int> paddings_;
  std::vector<int> block_shape_;
  bool b2s_;

 protected:
  void CalculateSpaceToBatchOutputShape(const Tensor *input_tensor,
                                        const DataFormat data_format,
                                        index_t *output_shape) {
    MACE_CHECK(input_tensor->dim_size() == 4, "Input's shape should be 4D");
    index_t batch = input_tensor->dim(0);
    index_t channels = 0;
    index_t height = 0;
    index_t width = 0;
    if (data_format == DataFormat::NHWC) {
      height = input_tensor->dim(1);
      width = input_tensor->dim(2);
      channels = input_tensor->dim(3);
    } else if (data_format == DataFormat::NCHW) {
      height = input_tensor->dim(2);
      width = input_tensor->dim(3);
      channels = input_tensor->dim(1);
    } else {
      MACE_NOT_IMPLEMENTED;
    }

    index_t padded_height = height + paddings_[0] + paddings_[1];
    index_t padded_width = width + paddings_[2] + paddings_[3];
    MACE_CHECK(padded_height % block_shape_[0] == 0, "padded input height",
               padded_height, " is not divisible by block height");
    MACE_CHECK(padded_width % block_shape_[1] == 0, "padded input width",
               padded_height, " is not divisible by block width");

    index_t new_batch = batch * block_shape_[0] * block_shape_[1];
    index_t new_height = padded_height / block_shape_[0];
    index_t new_width = padded_width / block_shape_[1];

    if (data_format == DataFormat::NHWC) {
      output_shape[0] = new_batch;
      output_shape[1] = new_height;
      output_shape[2] = new_width;
      output_shape[3] = channels;
    } else {
      output_shape[0] = new_batch;
      output_shape[1] = channels;
      output_shape[2] = new_height;
      output_shape[3] = new_width;
    }
  }

  void CalculateBatchToSpaceOutputShape(const Tensor *input_tensor,
                                        const DataFormat data_format,
                                        index_t *output_shape) {
    MACE_CHECK(input_tensor->dim_size() == 4, "Input's shape should be 4D");
    index_t batch = input_tensor->dim(0);
    index_t channels = 0;
    index_t height = 0;
    index_t width = 0;
    if (data_format == DataFormat::NHWC) {
      height = input_tensor->dim(1);
      width = input_tensor->dim(2);
      channels = input_tensor->dim(3);
    } else if (data_format == DataFormat::NCHW) {
      height = input_tensor->dim(2);
      width = input_tensor->dim(3);
      channels = input_tensor->dim(1);
    } else {
      MACE_NOT_IMPLEMENTED;
    }

    index_t new_batch = batch / block_shape_[0] / block_shape_[1];
    index_t new_height = height * block_shape_[0] - paddings_[0] - paddings_[1];
    index_t new_width = width * block_shape_[1] - paddings_[2] - paddings_[3];

    if (data_format == DataFormat::NHWC) {
      output_shape[0] = new_batch;
      output_shape[1] = new_height;
      output_shape[2] = new_width;
      output_shape[3] = channels;
    } else {
      output_shape[0] = new_batch;
      output_shape[1] = channels;
      output_shape[2] = new_height;
      output_shape[3] = new_width;
    }
  }
};

template<DeviceType D, typename T>
struct SpaceToBatchFunctor;

template<>
struct SpaceToBatchFunctor<DeviceType::CPU, float> : SpaceToBatchFunctorBase {
  SpaceToBatchFunctor(const std::vector<int> &paddings,
                      const std::vector<int> &block_shape,
                      bool b2s)
    : SpaceToBatchFunctorBase(paddings, block_shape, b2s) {}

  void operator()(Tensor *space_tensor,
                  Tensor *batch_tensor,
                  StatsFuture *future) {
    MACE_UNUSED(future);

    std::vector<index_t> output_shape(4, 0);
    if (b2s_) {
      CalculateBatchToSpaceOutputShape(batch_tensor,
                                       DataFormat::NCHW,
                                       output_shape.data());
      space_tensor->Resize(output_shape);
    } else {
      CalculateSpaceToBatchOutputShape(space_tensor,
                                       DataFormat::NCHW,
                                       output_shape.data());
      batch_tensor->Resize(output_shape);
    }

    Tensor::MappingGuard input_guard(space_tensor);
    Tensor::MappingGuard output_guard(batch_tensor);

    if (b2s_) {
      const float *input_data = batch_tensor->data<float>();
      float *output_data = space_tensor->mutable_data<float>();

      index_t in_height = batch_tensor->dim(2);
      index_t in_width = batch_tensor->dim(3);

      index_t out_batches = space_tensor->dim(0);
      index_t channels = space_tensor->dim(1);
      index_t out_height = space_tensor->dim(2);
      index_t out_width = space_tensor->dim(3);

#pragma omp parallel for collapse(2)
      for (index_t b = 0; b < out_batches; ++b) {
        for (index_t c = 0; c < channels; ++c) {
          for (index_t h = 0; h < out_height; ++h) {
            const index_t in_h = (h + paddings_[0]) / block_shape_[0];
            const index_t tile_h = (h + paddings_[0]) % block_shape_[0];
            for (index_t w = 0; w < out_width; ++w) {
              const index_t in_w = (w + paddings_[2]) / block_shape_[1];
              const index_t tile_w = (w + paddings_[2]) % block_shape_[1];
              const index_t
                in_b = (tile_h * block_shape_[1] + tile_w) * out_batches + b;
              output_data[((b * channels + c) * out_height + h) * out_width
                + w] =
                input_data[
                  ((in_b * channels + c) * in_height + in_h) * in_width
                    + in_w];
            }
          }
        }
      }
    } else {
      const float *input_data = space_tensor->data<float>();
      float *output_data = batch_tensor->mutable_data<float>();

      index_t in_batches = space_tensor->dim(0);
      index_t in_height = space_tensor->dim(2);
      index_t in_width = space_tensor->dim(3);

      index_t out_batches = batch_tensor->dim(0);
      index_t channels = batch_tensor->dim(1);
      index_t out_height = batch_tensor->dim(2);
      index_t out_width = batch_tensor->dim(3);

#pragma omp parallel for collapse(2)
      for (index_t b = 0; b < out_batches; ++b) {
        for (index_t c = 0; c < channels; ++c) {
          const index_t in_b = b % in_batches;
          const index_t tile_h = b / in_batches / block_shape_[1];
          const index_t tile_w = b / in_batches % block_shape_[1];
          for (index_t h = 0; h < out_height; ++h) {
            const index_t in_h = h * block_shape_[0] + tile_h - paddings_[0];
            for (index_t w = 0; w < out_width; ++w) {
              const index_t in_w = w * block_shape_[1] + tile_w - paddings_[2];
              if (in_h >= 0 && in_w >= 0 && in_h < in_height
                && in_w < in_width) {
                output_data[((b * channels + c) * out_height + h) * out_width
                  + w] =
                  input_data[
                    ((in_b * channels + c) * in_height + in_h) * in_width
                      + in_w];
              } else {
                output_data[((b * channels + c) * out_height + h) * out_width
                  + w] = 0;
              }
            }
          }
        }
      }
    }
  }
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
struct SpaceToBatchFunctor<DeviceType::GPU, T> : SpaceToBatchFunctorBase {
  SpaceToBatchFunctor(const std::vector<int> &paddings,
                      const std::vector<int> &block_shape,
                      bool b2s)
      : SpaceToBatchFunctorBase(paddings, block_shape, b2s) {}

  void operator()(Tensor *space_tensor,
                  Tensor *batch_tensor,
                  StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> space_shape_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_SPACE_TO_BATCH_H_
