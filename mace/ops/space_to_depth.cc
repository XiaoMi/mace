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
#include "mace/ops/opencl/image/space_to_depth.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class SpaceToDepthOp : public Operation {
 public:
  explicit SpaceToDepthOp(OpConstructContext *context)
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

    MACE_CHECK(
        (input_width % block_size_ == 0) && (input_height % block_size_ == 0),
        "input width and height should be dividable by block_size");

    const index_t output_depth = input_depth * block_size_ * block_size_;
    const index_t output_width = input_width / block_size_;
    const index_t output_height = input_height / block_size_;
    std::vector<index_t> output_shape = {batch_size, output_depth,
                                         output_height, output_width};

    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard logits_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();

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
                ((b * input_depth + d) * input_height + h) * input_width + w;
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

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class SpaceToDepthOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit SpaceToDepthOp(OpConstructContext *context)
      : Operation(context) {
    int block_size = Operation::GetOptionalArg<int>("block_size", 1);
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::SpaceToDepthKernel<T>>(block_size);
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
  std::unique_ptr<OpenCLSpaceToDepthKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterSpaceToDepth(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "SpaceToDepth",
                   SpaceToDepthOp, DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "SpaceToDepth",
                   SpaceToDepthOp, DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "SpaceToDepth",
                   SpaceToDepthOp, DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
