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

#ifndef MACE_OPS_OPENCL_BUFFER_TRANSFORMER_H_
#define MACE_OPS_OPENCL_BUFFER_TRANSFORMER_H_

#include "mace/core/operator.h"
#include "mace/ops/opencl/common.h"
#include "mace/ops/opencl/image/buffer_to_image.h"
#include "mace/ops/opencl/image/image_to_buffer.h"
#include "mace/ops/opencl/buffer/buffer_transform.h"

namespace mace {
namespace ops {
// Only used for GPU Operation(BufferTransform)
template <typename T>
class OpenCLBufferTransformer {
 public:
  OpenCLBufferTransformer(const MemoryType in_mem_type,
                         const MemoryType out_mem_type) {
    if (out_mem_type == MemoryType::GPU_IMAGE) {
      kernel_.reset(new opencl::image::BufferToImage<T>);
    } else if (in_mem_type == MemoryType::GPU_IMAGE){
      kernel_.reset(new opencl::image::ImageToBuffer<T>);
    } else {
      kernel_.reset(new opencl::buffer::BufferTransform<T>);
    }
  }

  MaceStatus Transform(OpContext *context,
                       const Tensor *input,
                       const BufferType type,
                       const int wino_blk_size,
                       const MemoryType out_mem_type,
                       Tensor *output) {
    Workspace *ws = context->workspace();
    DataType dt = DataTypeToEnum<T>::value;
    MemoryType in_mem_type = input->memory_type();
    if (out_mem_type == MemoryType::GPU_IMAGE ||
        out_mem_type == MemoryType::GPU_BUFFER) {
      if (in_mem_type != MemoryType::CPU_BUFFER) {
        return kernel_->Compute(
            context, input, type, wino_blk_size, output);
      } else {
        // convert to the GPU Buffer with the input's data type.
        Tensor *internal_tensor = ws->CreateTensor(
            InternalTransformedName(input->name()),
            context->device()->allocator(), input->dtype());
        output->Resize(input->shape());
        const uint8_t *input_ptr = input->data<uint8_t>();
        Tensor::MappingGuard guard(internal_tensor);
        uint8_t *internal_ptr = internal_tensor->mutable_data<uint8_t>();
        memcpy(internal_ptr, input_ptr, input->raw_size());
        // convert the internal GPU Buffer to output.
        return kernel_->Compute(
            context, internal_tensor, type, wino_blk_size, output);
      }
    } else {  // out_mem_type == MemoryType::CPU_BUFFER
      // convert to the GPU Buffer with the output's data type.
      Tensor internal_tensor(context->device()->allocator(),
                             dt,
                             false,
                             InternalTransformedName(input->name()));
      MACE_RETURN_IF_ERROR(kernel_->Compute(
          context, input, type, wino_blk_size, &internal_tensor));
      // convert the internal GPU Buffer to output.
      Tensor::MappingGuard guard(&internal_tensor);
      const T *internal_ptr = internal_tensor.data<T>();
      output->Resize(internal_tensor.shape());
      T *output_ptr = output->mutable_data<T>();
      memcpy(output_ptr, internal_ptr, internal_tensor.size() * sizeof(T));
      return MaceStatus::MACE_SUCCESS;
    }
  }

 private:
  std::string InternalTransformedName(const std::string &name) {
    // TODO(liuqi): This may create a conflict.
    const char *postfix = "_mace_identity_internal";
    return name + postfix;
  }

 private:
  std::unique_ptr<OpenCLBufferTransformKernel> kernel_;
};

std::string TransformedName(const std::string &name);

template <typename T>
MaceStatus TransformFilter(
    mace::OpConstructContext *context,
    OperatorDef *op_def,
    const int input_idx,
    const BufferType buffer_type,
    const MemoryType mem_type) {
  const DataType dt = DataTypeToEnum<T>::value;
  OpContext op_context(context->workspace(), context->device());
  Workspace *ws = context->workspace();
  std::string input_name = op_def->input(input_idx);
  Tensor *input = ws->GetTensor(input_name);
  std::string output_name = TransformedName(input_name);
  Tensor *output =
      ws->CreateTensor(output_name, context->device()->allocator(), dt);

  // update the information
  op_def->set_input(input_idx, output_name);
  input->MarkUnused();
  return OpenCLBufferTransformer<T>(input->memory_type(), mem_type).
      Transform(&op_context, input, buffer_type, 0, mem_type, output);
}

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_BUFFER_TRANSFORMER_H_
