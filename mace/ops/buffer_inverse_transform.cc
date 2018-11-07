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

#include <memory>

#include "mace/core/operator.h"
#include "mace/ops/opencl/buffer/buffer_inverse_transform.h"
#include "mace/ops/opencl/image/image_to_buffer.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class BufferInverseTransformOp;

template <typename T>
class BufferInverseTransformOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit BufferInverseTransformOp(OpConstructContext *context)
      : Operation(context),
        wino_blk_size_(Operation::GetOptionalArg<int>("wino_block_size", 2)) {
    if (context->device()->opencl_runtime()->UseImageMemory()) {
      kernel_.reset(new opencl::image::ImageToBuffer<T>);
    } else {
      kernel_.reset(new opencl::buffer::BufferInverseTransform<T>);
    }
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    ops::BufferType type =
        static_cast<ops::BufferType>(Operation::GetOptionalArg<int>(
            "buffer_type", static_cast<int>(ops::CONV2D_FILTER)));

    return kernel_->Compute(context, input, type,
                            wino_blk_size_, output);
  }

 private:
  const int wino_blk_size_;
  std::unique_ptr<OpenCLBufferTransformKernel> kernel_;
};


void RegisterBufferInverseTransform(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "BufferInverseTransform",
                   BufferInverseTransformOp, DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "BufferInverseTransform",
                   BufferInverseTransformOp, DeviceType::GPU, half);
}

}  // namespace ops
}  // namespace mace
