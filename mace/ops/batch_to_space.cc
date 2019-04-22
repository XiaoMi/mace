// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include <algorithm>
#include <memory>

#include "mace/core/operator.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/batch_to_space.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

class BatchToSpaceOpBase : public Operation {
 public:
  explicit BatchToSpaceOpBase(OpConstructContext *context)
      : Operation(context),
        paddings_(Operation::GetRepeatedArgs<int>("crops", {0, 0, 0, 0})),
        block_shape_(Operation::GetRepeatedArgs<int>("block_shape", {1, 1})) {
    MACE_CHECK(
        block_shape_.size() == 2 && block_shape_[0] > 1 && block_shape_[1] > 1,
        "Block's shape should be 1D, and greater than 1");
    MACE_CHECK(paddings_.size() == 4, "Paddings' shape should be 2D");
  }

 protected:
  std::vector<int> paddings_;
  std::vector<int> block_shape_;

 protected:
  void CalculateBatchToSpaceOutputShape(const Tensor *input_tensor,
                                        const DataFormat data_format,
                                        index_t *output_shape) {
    MACE_CHECK(input_tensor->dim_size() == 4,
               "Input(", input_tensor->name(), ") shape should be 4D");
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

template <DeviceType D, class T>
class BatchToSpaceNDOp;

template <>
class BatchToSpaceNDOp<DeviceType::CPU, float> : public BatchToSpaceOpBase {
 public:
  explicit BatchToSpaceNDOp(OpConstructContext *context)
      : BatchToSpaceOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *batch_tensor = this->Input(0);
    Tensor *space_tensor = this->Output(0);
    std::vector<index_t> output_shape(4, 0);
    CalculateBatchToSpaceOutputShape(batch_tensor,
                                     DataFormat::NCHW,
                                     output_shape.data());
    MACE_RETURN_IF_ERROR(space_tensor->Resize(output_shape));

    Tensor::MappingGuard input_guard(batch_tensor);
    Tensor::MappingGuard output_guard(space_tensor);

    int pad_top = paddings_[0];
    int pad_left = paddings_[2];
    int block_shape_h = block_shape_[0];
    int block_shape_w = block_shape_[1];

    const float *input_data = batch_tensor->data<float>();
    float *output_data = space_tensor->mutable_data<float>();

    index_t in_batches = batch_tensor->dim(0);
    index_t in_height = batch_tensor->dim(2);
    index_t in_width = batch_tensor->dim(3);

    index_t out_batches = space_tensor->dim(0);
    index_t channels = space_tensor->dim(1);
    index_t out_height = space_tensor->dim(2);
    index_t out_width = space_tensor->dim(3);

    // 32k/sizeof(float)/out_width/block_shape
    index_t
        block_h_size =
        std::max(static_cast<index_t>(1), 8 * 1024 / block_shape_w / out_width);

    // make channel outter loop so we can make best use of cache
    for (index_t c = 0; c < channels; ++c) {
      for (index_t block_h = 0; block_h < in_height;
           block_h += block_h_size) {
        for (index_t in_b = 0; in_b < in_batches; ++in_b) {
          const index_t b = in_b % out_batches;
          const index_t tile_index = in_b / out_batches;
          const index_t tile_h = tile_index / block_shape_w;
          const index_t tile_w = tile_index % block_shape_w;
          const index_t valid_h_start = std::max(block_h,
                                                 (pad_top - tile_h
                                                     + block_shape_h - 1)
                                                     / block_shape_h);
          const index_t valid_h_end = std::min(in_height,
                                               std::min(
                                                   block_h + block_h_size,
                                                   (out_height + pad_top
                                                       - tile_h
                                                       + block_shape_h - 1)
                                                       / block_shape_h));
          const index_t valid_w_start = std::max(static_cast<index_t>(0),
                                                 (pad_left - tile_w
                                                     + block_shape_w - 1)
                                                     / block_shape_w);
          const index_t valid_w_end = std::min(in_width,
                                               (out_width + pad_left - tile_w
                                                   + block_shape_w - 1)
                                                   / block_shape_w);
          const float *input_base =
              input_data + (in_b * channels + c) * in_height * in_width;
          float *output_base =
              output_data + (b * channels + c) * out_height * out_width;

          index_t h = valid_h_start * block_shape_h + tile_h - pad_top;
          for (index_t in_h = valid_h_start; in_h < valid_h_end; ++in_h) {
            index_t w = valid_w_start * block_shape_w + tile_w - pad_left;
            for (index_t in_w = valid_w_start; in_w < valid_w_end; ++in_w) {
              output_base[h * out_width + w] =
                  input_base[in_h * in_width + in_w];
              w += block_shape_w;
            }  // w
            h += block_shape_h;
          }  // h
        }  // b
      }  // block_h
    }  // c

    return MaceStatus::MACE_SUCCESS;
  }
};

template <>
class BatchToSpaceNDOp<DeviceType::CPU, uint8_t> : public BatchToSpaceOpBase {
 public:
  explicit BatchToSpaceNDOp(OpConstructContext *context)
      : BatchToSpaceOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *batch_tensor = this->Input(0);
    Tensor *space_tensor = this->Output(0);
    std::vector<index_t> output_shape(4, 0);
    CalculateBatchToSpaceOutputShape(batch_tensor,
                                     DataFormat::NHWC,
                                     output_shape.data());
    MACE_RETURN_IF_ERROR(space_tensor->Resize(output_shape));

    Tensor::MappingGuard input_guard(batch_tensor);
    Tensor::MappingGuard output_guard(space_tensor);

    int pad_top = paddings_[0];
    int pad_left = paddings_[2];
    int block_shape_h = block_shape_[0];
    int block_shape_w = block_shape_[1];

    space_tensor->SetScale(batch_tensor->scale());
    space_tensor->SetZeroPoint(batch_tensor->zero_point());
    const uint8_t *input_data = batch_tensor->data<uint8_t>();
    uint8_t *output_data = space_tensor->mutable_data<uint8_t>();

    index_t in_batches = batch_tensor->dim(0);
    index_t in_height = batch_tensor->dim(1);
    index_t in_width = batch_tensor->dim(2);

    index_t out_batches = space_tensor->dim(0);
    index_t out_height = space_tensor->dim(1);
    index_t out_width = space_tensor->dim(2);
    index_t channels = space_tensor->dim(3);

    for (index_t in_b = 0; in_b < in_batches; ++in_b) {
      const index_t b = in_b % out_batches;
      const index_t tile_index = in_b / out_batches;
      const index_t tile_h = tile_index / block_shape_w;
      const index_t tile_w = tile_index % block_shape_w;
      const index_t valid_h_start = std::max(static_cast<index_t>(0),
                                             (pad_top - tile_h
                                                 + block_shape_h - 1)
                                                 / block_shape_h);
      const index_t valid_h_end = std::min(in_height,
                                           (out_height + pad_top
                                               - tile_h
                                               + block_shape_h - 1)
                                               / block_shape_h);
      const index_t valid_w_start = std::max(static_cast<index_t>(0),
                                             (pad_left - tile_w
                                                 + block_shape_w - 1)
                                                 / block_shape_w);
      const index_t valid_w_end = std::min(in_width,
                                           (out_width + pad_left
                                               - tile_w
                                               + block_shape_w - 1)
                                               / block_shape_w);
      const uint8_t *input_base =
          input_data + in_b * in_height * in_width * channels;
      uint8_t
          *output_base = output_data + b * out_height * out_width * channels;

      index_t h = valid_h_start * block_shape_h + tile_h - pad_top;
      for (index_t in_h = valid_h_start; in_h < valid_h_end; ++in_h) {
        index_t w = valid_w_start * block_shape_w + tile_w - pad_left;
        for (index_t in_w = valid_w_start; in_w < valid_w_end; ++in_w) {
          memcpy(output_base + (h * out_width + w) * channels,
                 input_base + (in_h * in_width + in_w) * channels,
                 channels * sizeof(uint8_t));
          w += block_shape_w;
        }  // w
        h += block_shape_h;
      }  // h
    }  // b

    return MaceStatus::MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class BatchToSpaceNDOp<DeviceType::GPU, T> : public BatchToSpaceOpBase {
 public:
  explicit BatchToSpaceNDOp(OpConstructContext *context)
      : BatchToSpaceOpBase(context) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::BatchToSpaceKernel<T>>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *batch_tensor = this->Input(0);
    Tensor *space_tensor = this->Output(0);
    std::vector<index_t> output_shape(4, 0);
    CalculateBatchToSpaceOutputShape(batch_tensor, DataFormat::NHWC,
                                     output_shape.data());
    return kernel_->Compute(context, batch_tensor, paddings_, block_shape_,
                            output_shape, space_tensor);
  }

 private:
  std::unique_ptr<OpenCLBatchToSpaceKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL


void RegisterBatchToSpaceND(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "BatchToSpaceND",
                   BatchToSpaceNDOp, DeviceType::CPU, float);

  MACE_REGISTER_OP(op_registry, "BatchToSpaceND",
                   BatchToSpaceNDOp, DeviceType::CPU, uint8_t);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "BatchToSpaceND",
                   BatchToSpaceNDOp, DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "BatchToSpaceND",
                   BatchToSpaceNDOp, DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
