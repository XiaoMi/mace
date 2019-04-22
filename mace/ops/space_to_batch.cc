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
#include "mace/ops/opencl/image/space_to_batch.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

class SpaceToBatchOpBase : public Operation {
 public:
  explicit SpaceToBatchOpBase(OpConstructContext *context)
      : Operation(context),
        paddings_(Operation::GetRepeatedArgs<int>("paddings", {0, 0, 0, 0})),
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
};

template <DeviceType D, class T>
class SpaceToBatchNDOp;

template <>
class SpaceToBatchNDOp<DeviceType::CPU, float> : public SpaceToBatchOpBase {
 public:
  explicit SpaceToBatchNDOp(OpConstructContext *context)
      : SpaceToBatchOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *space_tensor = this->Input(0);
    Tensor *batch_tensor = this->Output(0);
    std::vector<index_t> output_shape(4, 0);

    CalculateSpaceToBatchOutputShape(space_tensor,
                                     DataFormat::NCHW,
                                     output_shape.data());
    MACE_RETURN_IF_ERROR(batch_tensor->Resize(output_shape));

    Tensor::MappingGuard input_guard(space_tensor);
    Tensor::MappingGuard output_guard(batch_tensor);

    int pad_top = paddings_[0];
    int pad_left = paddings_[2];
    int block_shape_h = block_shape_[0];
    int block_shape_w = block_shape_[1];

    const float *input_data = space_tensor->data<float>();
    float *output_data = batch_tensor->mutable_data<float>();

    index_t in_batches = space_tensor->dim(0);
    index_t in_height = space_tensor->dim(2);
    index_t in_width = space_tensor->dim(3);

    index_t out_batches = batch_tensor->dim(0);
    index_t channels = batch_tensor->dim(1);
    index_t out_height = batch_tensor->dim(2);
    index_t out_width = batch_tensor->dim(3);

    index_t block_h_size =
        std::max(static_cast<index_t>(1), 8 * 1024 / block_shape_w / in_width);

    // make channel outter loop so we can make best use of cache
    for (index_t c = 0; c < channels; ++c) {
      for (index_t block_h = 0; block_h < out_height;
           block_h += block_h_size) {
        for (index_t b = 0; b < out_batches; ++b) {
          const index_t in_b = b % in_batches;
          const index_t tile_index = b / in_batches;
          const index_t tile_h = tile_index / block_shape_w;
          const index_t tile_w = tile_index % block_shape_w;
          const index_t valid_h_start = std::max(block_h,
                                                 (pad_top - tile_h
                                                     + block_shape_h - 1)
                                                     / block_shape_h);
          const index_t valid_h_end = std::min(out_height,
                                               std::min(
                                                   block_h + block_h_size,
                                                   (in_height + pad_top
                                                       - tile_h
                                                       + block_shape_h - 1)
                                                       / block_shape_h));
          const index_t valid_w_start = std::max(static_cast<index_t>(0),
                                                 (pad_left - tile_w
                                                     + block_shape_w - 1)
                                                     / block_shape_w);
          const index_t valid_w_end = std::min(out_width,
                                               (in_width + pad_left - tile_w
                                                   + block_shape_w - 1)
                                                   / block_shape_w);
          const float *input_base =
              input_data + (in_b * channels + c) * in_height * in_width;
          float *output_base =
              output_data + (b * channels + c) * out_height * out_width;

          memset(output_base + block_h * out_width,
                 0,
                 (valid_h_start - block_h) * out_width * sizeof(float));

          index_t in_h = valid_h_start * block_shape_h + tile_h - pad_top;
          for (index_t h = valid_h_start; h < valid_h_end; ++h) {
            memset(output_base + h * out_width,
                   0,
                   valid_w_start * sizeof(float));

            index_t in_w = valid_w_start * block_shape_w + tile_w - pad_left;
            for (index_t w = valid_w_start; w < valid_w_end; ++w) {
              output_base[h * out_width + w] =
                  input_base[in_h * in_width + in_w];
              in_w += block_shape_w;
            }  // w
            in_h += block_shape_h;

            memset(output_base + h * out_width + valid_w_end,
                   0,
                   (out_width - valid_w_end) * sizeof(float));
          }  // h

          memset(output_base + valid_h_end * out_width,
                 0,
                 (std::min(out_height, block_h + block_h_size) - valid_h_end)
                     * out_width * sizeof(float));
        }  // b
      }  // block_h
    }  // c

    return MaceStatus::MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_QUANTIZE
template <>
class SpaceToBatchNDOp<DeviceType::CPU, uint8_t> : public SpaceToBatchOpBase {
 public:
  explicit SpaceToBatchNDOp(OpConstructContext *context)
      : SpaceToBatchOpBase(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *space_tensor = this->Input(0);
    Tensor *batch_tensor = this->Output(0);
    std::vector<index_t> output_shape(4, 0);

    CalculateSpaceToBatchOutputShape(space_tensor,
                                     DataFormat::NHWC,
                                     output_shape.data());
    MACE_RETURN_IF_ERROR(batch_tensor->Resize(output_shape));
    int zero_point = space_tensor->zero_point();

    Tensor::MappingGuard input_guard(space_tensor);
    Tensor::MappingGuard output_guard(batch_tensor);

    int pad_top = paddings_[0];
    int pad_left = paddings_[2];
    int block_shape_h = block_shape_[0];
    int block_shape_w = block_shape_[1];

    batch_tensor->SetScale(space_tensor->scale());
    batch_tensor->SetZeroPoint(space_tensor->zero_point());
    const uint8_t *input_data = space_tensor->data<uint8_t>();
    uint8_t *output_data = batch_tensor->mutable_data<uint8_t>();

    index_t in_batches = space_tensor->dim(0);
    index_t in_height = space_tensor->dim(1);
    index_t in_width = space_tensor->dim(2);

    index_t out_batches = batch_tensor->dim(0);
    index_t out_height = batch_tensor->dim(1);
    index_t out_width = batch_tensor->dim(2);
    index_t channels = batch_tensor->dim(3);

    for (index_t b = 0; b < out_batches; ++b) {
      const index_t in_b = b % in_batches;
      const index_t tile_index = b / in_batches;
      const index_t tile_h = tile_index / block_shape_w;
      const index_t tile_w = tile_index % block_shape_w;
      const index_t valid_h_start = std::max(static_cast<index_t>(0),
                                             (pad_top - tile_h
                                                 + block_shape_h - 1)
                                                 / block_shape_h);
      const index_t valid_h_end = std::min(out_height,
                                           (in_height + pad_top
                                               - tile_h
                                               + block_shape_h - 1)
                                               / block_shape_h);
      const index_t valid_w_start = std::max(static_cast<index_t>(0),
                                             (pad_left - tile_w
                                                 + block_shape_w - 1)
                                                 / block_shape_w);
      const index_t valid_w_end = std::min(out_width,
                                           (in_width + pad_left - tile_w
                                               + block_shape_w - 1)
                                               / block_shape_w);
      const uint8_t *input_base =
          input_data + in_b * channels * in_height * in_width;
      uint8_t *output_base =
          output_data + b * channels * out_height * out_width;

      memset(output_base,
             zero_point,
             valid_h_start * out_width * channels * sizeof(uint8_t));

      index_t in_h = valid_h_start * block_shape_h + tile_h - pad_top;
      for (index_t h = valid_h_start; h < valid_h_end; ++h) {
        memset(output_base + h * out_width * channels,
               zero_point,
               valid_w_start * channels * sizeof(uint8_t));

        index_t
            in_w = valid_w_start * block_shape_w + tile_w - pad_left;
        for (index_t w = valid_w_start; w < valid_w_end; ++w) {
          memcpy(output_base + (h * out_width + w) * channels,
                 input_base + (in_h * in_width + in_w) * channels,
                 sizeof(uint8_t) * channels);
          in_w += block_shape_w;
        }  // w
        in_h += block_shape_h;

        memset(output_base + (h * out_width + valid_w_end) * channels,
               zero_point,
               (out_width - valid_w_end) * channels * sizeof(uint8_t));
      }  // h

      memset(output_base + valid_h_end * out_width * channels,
             zero_point,
             (out_height - valid_h_end) * out_width * channels
                 * sizeof(uint8_t));
    }  // b

    return MaceStatus::MACE_SUCCESS;
  }
};
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class SpaceToBatchNDOp<DeviceType::GPU, T> : public SpaceToBatchOpBase {
 public:
  explicit SpaceToBatchNDOp(OpConstructContext *context)
      : SpaceToBatchOpBase(context) {
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::SpaceToBatchKernel<T>>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *space_tensor = this->Input(0);
    Tensor *batch_tensor = this->Output(0);
    std::vector<index_t> output_shape(4, 0);
    CalculateSpaceToBatchOutputShape(space_tensor, DataFormat::NHWC,
                                     output_shape.data());
    return kernel_->Compute(context, space_tensor, paddings_, block_shape_,
                            output_shape, batch_tensor);
  }

 private:
  std::unique_ptr<OpenCLSpaceToBatchKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterSpaceToBatchND(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "SpaceToBatchND",
                   SpaceToBatchNDOp, DeviceType::CPU, float);

#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_OP(op_registry, "SpaceToBatchND",
                   SpaceToBatchNDOp, DeviceType::CPU, uint8_t);
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "SpaceToBatchND",
                   SpaceToBatchNDOp, DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "SpaceToBatchND",
                   SpaceToBatchNDOp, DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
