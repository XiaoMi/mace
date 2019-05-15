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
        wino_blk_size_(Operation::GetOptionalArg<int>("wino_block_size", 0)),
        out_mem_type_(static_cast<MemoryType>(Operation::GetOptionalArg<int>(
            "mem_type", static_cast<int>(MemoryType::GPU_IMAGE)))) {}

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    auto type =
        static_cast<OpenCLBufferType>(Operation::GetOptionalArg<int>(
            "buffer_type", static_cast<int>(CONV2D_FILTER)));

    MemoryType in_mem_type = context->workspace()->GetTensor(
        operator_def_->input(0))->memory_type();
    return OpenCLBufferTransformer<T>(in_mem_type, out_mem_type_).Transform(
        context, input, type, out_mem_type_, wino_blk_size_, output);
  }

 private:
  const int wino_blk_size_;
  MemoryType out_mem_type_;
};


void RegisterBufferTransform(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "BufferTransform",
                   BufferTransformOp, DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "BufferTransform",
                   BufferTransformOp, DeviceType::GPU, half);
}

}  // namespace ops
}  // namespace mace
