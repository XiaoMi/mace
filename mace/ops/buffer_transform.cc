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
#include "mace/ops/opencl/buffer_transformer.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class BufferTransformOp;

template <typename T>
class BufferTransformOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit BufferTransformOp(OpConstructContext *context)
      : Operation(context),
        wino_blk_size_(Operation::GetOptionalArg<int>("wino_block_size", 2)),
        out_mem_type_(MemoryType::GPU_BUFFER),
        transformer_(nullptr) {
    MemoryType in_mem_type = context->workspace()->GetTensor(
        operator_def_->input(0))->memory_type();
    if (context->device()->opencl_runtime()->UseImageMemory()) {
      out_mem_type_ = MemoryType::GPU_IMAGE;
    }
    transformer_.reset(new OpenCLBufferTransformer<T>(in_mem_type,
                                                      out_mem_type_));
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    ops::BufferType type =
        static_cast<ops::BufferType>(Operation::GetOptionalArg<int>(
            "buffer_type", static_cast<int>(ops::CONV2D_FILTER)));

    return transformer_->Transform(
        context, input, type, wino_blk_size_, out_mem_type_, output);
  }

 private:
  const int wino_blk_size_;
  MemoryType out_mem_type_;
  std::unique_ptr<OpenCLBufferTransformer<T>> transformer_;
};


void RegisterBufferTransform(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "BufferTransform",
                   BufferTransformOp, DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "BufferTransform",
                   BufferTransformOp, DeviceType::GPU, half);
}

}  // namespace ops
}  // namespace mace
