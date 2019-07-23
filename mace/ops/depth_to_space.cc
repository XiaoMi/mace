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

#include <memory>
#include <vector>

#include "mace/core/operator.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/depth_to_space.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

template<DeviceType D, class T>
class DepthToSpaceOp;

template<>
class DepthToSpaceOp<CPU, float> : public Operation {
 public:
  explicit DepthToSpaceOp(OpConstructContext *context)
      : Operation(context),
        block_size_(Operation::GetOptionalArg<int>("block_size", 1)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    MACE_CHECK(input->dim_size() == 4, "input dim should be 4");
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
    const float *input_ptr = input->data<float>();
    float *output_ptr = output->mutable_data<float>();

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

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  const int block_size_;
};

#ifdef MACE_ENABLE_QUANTIZE
template<>
class DepthToSpaceOp<CPU, uint8_t> : public Operation {
 public:
  explicit DepthToSpaceOp(OpConstructContext *context)
      : Operation(context),
        block_size_(Operation::GetOptionalArg<int>("block_size", 1)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    MACE_CHECK(input->dim_size() == 4, "input dim should be 4");
    const index_t batch_size = input->dim(0);
    const index_t input_depth = input->dim(3);
    const index_t input_height = input->dim(1);
    const index_t input_width = input->dim(2);

    MACE_CHECK(input_depth % (block_size_ * block_size_) == 0,
               "input depth should be dividable by block_size * block_size",
               input_depth);

    const index_t output_depth = input_depth / (block_size_ * block_size_);
    const index_t output_width = input_width * block_size_;
    const index_t output_height = input_height * block_size_;
    std::vector<index_t>
        output_shape = {batch_size, output_height, output_width, output_depth};

    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard logits_guard(input);
    Tensor::MappingGuard output_guard(output);
    const uint8_t *input_ptr = input->data<uint8_t>();
    uint8_t *output_ptr = output->mutable_data<uint8_t>();

    for (index_t b = 0; b < batch_size; ++b) {
      for (index_t h = 0; h < output_height; ++h) {
        const index_t in_h = h / block_size_;
        const index_t offset_h = (h % block_size_);
        for (int w = 0; w < output_width; ++w) {
          const index_t in_w = w / block_size_;
          const index_t offset_w = w % block_size_;
          const index_t offset_d =
              (offset_h * block_size_ + offset_w) * output_depth;

          for (index_t d = 0; d < output_depth; ++d) {
            const index_t in_d = d + offset_d;
            const index_t o_index =
                ((b * output_height + h) * output_width + w) * output_depth
                    + d;
            const index_t i_index =
                ((b * input_height + in_h) * input_width + in_w) * input_depth
                    + in_d;
            output_ptr[o_index] = input_ptr[i_index];
          }
        }
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  const int block_size_;
};
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
template<>
class DepthToSpaceOp<DeviceType::GPU, float> : public Operation {
 public:
  explicit DepthToSpaceOp(OpConstructContext *context)
      : Operation(context) {
    int block_size = Operation::GetOptionalArg<int>("block_size", 1);
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::DepthToSpaceKernel>(block_size);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    MACE_CHECK(input->dim_size() == 4, "input dim should be 4");
    return kernel_->Compute(context, input, output);
  }

 private:
  std::unique_ptr<OpenCLDepthToSpaceKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterDepthToSpace(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "DepthToSpace",
                   DepthToSpaceOp, DeviceType::CPU, float);

#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_OP(op_registry, "DepthToSpace",
                   DepthToSpaceOp, DeviceType::CPU, uint8_t);
#endif  // MACE_ENABLE_QUANTIZE

  MACE_REGISTER_GPU_OP(op_registry, "DepthToSpace", DepthToSpaceOp);
}

}  // namespace ops
}  // namespace mace
