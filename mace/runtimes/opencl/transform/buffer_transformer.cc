// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include "mace/runtimes/opencl/transform/buffer_transformer.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mace/core/registry/ops_registry.h"
#include "mace/runtimes/opencl/transform/buffer_to_image.h"
#include "mace/runtimes/opencl/transform/image_to_buffer.h"
#include "mace/utils/memory.h"

namespace mace {

namespace {
std::string TransformedFilterName(const std::string &name) {
  // TODO(liuqi): This may create a conflict.
  const char *postfix = "_mace_identity_transformed";
  return name + postfix;
}

std::string InternalTransformedName(const std::string &name) {
  const char *postfix = "_mace_identity_internal";
  return name + postfix;
}
}  // namespace

namespace runtimes {
namespace opencl {
OpenCLBufferTransformer::OpenCLBufferTransformer(
    const MemoryType in_mem_type, const MemoryType out_mem_type) {
  if (out_mem_type == MemoryType::GPU_IMAGE) {
    kernel_ = make_unique<BufferToImage>();
  } else if (in_mem_type == MemoryType::GPU_IMAGE) {
    kernel_ = make_unique<ImageToBuffer>();
  } else {
    kernel_ = make_unique<BufferTransform>();
  }
}

MaceStatus OpenCLBufferTransformer::Transform(
    OpContext *context, const Tensor *input, const BufferContentType type,
    const MemoryType out_mem_type, const int wino_blk_size, Tensor *output) {
  DataType dt = output->dtype();
  MemoryType in_mem_type = input->memory_type();

  if (out_mem_type == MemoryType::GPU_IMAGE ||
      out_mem_type == MemoryType::GPU_BUFFER) {
    if (in_mem_type != MemoryType::CPU_BUFFER) {
      return kernel_->Compute(
          context, input, type, wino_blk_size, output);
    } else {
      // convert to the GPU Buffer with the input's data type.
      // 1. CPU buffer to GPU Buffer
      auto runtime = context->runtime();
      auto tensor_name = InternalTransformedName(input->name());
      std::unique_ptr<Tensor> inter_tensor =
          make_unique<Tensor>(runtime, input->dtype(), GPU_BUFFER,
                              input->shape(), false, tensor_name, type);
      Tensor *internal_tensor = inter_tensor.get();
      runtime->AllocateBufferForTensor(internal_tensor, RENT_SCRATCH);
      {
        const uint8_t *input_ptr = input->data<uint8_t>();
        // No need to finish the opencl command queue to write to the tensor
        // from CPU, this can accelerate the mapping if using ION buffer.
        Tensor::MappingGuard guard(internal_tensor, false);
        uint8_t *internal_ptr = internal_tensor->mutable_data<uint8_t>();
        memcpy(internal_ptr, input_ptr, input->raw_size());
      }
      // 2. convert the internal GPU Buffer to output.
      return kernel_->Compute(
          context, internal_tensor, type, wino_blk_size, output);
    }
  } else if (out_mem_type == MemoryType::CPU_BUFFER) {
    // 1. convert to the GPU Buffer with the output's data type.
    auto *opencl_runtime = input->GetCurRuntime();
    auto tensor_name = InternalTransformedName(input->name());
    Tensor internal_tensor(opencl_runtime, dt, GPU_BUFFER,
                           input->shape(), false, tensor_name, type);
    opencl_runtime->AllocateBufferForTensor(&internal_tensor, RENT_SCRATCH);
    MACE_RETURN_IF_ERROR(kernel_->Compute(
        context, input, type, wino_blk_size, &internal_tensor));
    // 2. convert the internal GPU Buffer to output.
    Tensor::MappingGuard guard(&internal_tensor);
    const float *internal_ptr = internal_tensor.data<float>();
    output->Resize(internal_tensor.shape());
    float *output_ptr = output->mutable_data<float>();
    memcpy(output_ptr, internal_ptr, internal_tensor.size() * sizeof(float));
    return MaceStatus::MACE_SUCCESS;
  } else {
    LOG(FATAL) << "Unexpected error: " << out_mem_type;
    return MaceStatus::MACE_SUCCESS;
  }
}

}  // namespace opencl
}  // namespace runtimes


MaceStatus TransformFilter(
    mace::OpConstructContext *context,
    OperatorDef *op_def,
    const int input_idx,
    const BufferContentType content_type,
    const MemoryType mem_type,
    const int wino_blk_size) {
  OpContext op_context(context->workspace(), context->runtime());
  Workspace *ws = context->workspace();
  std::string input_name = op_def->input(input_idx);
  Tensor *input = ws->GetTensor(input_name);
  MACE_CHECK(input->is_weight());
  std::string output_name = TransformedFilterName(input_name);
  Tensor *output = ws->GetTensor(output_name);
  if (output == nullptr) {
    auto runtime = context->runtime();
    std::unique_ptr<Tensor> output_tensor =
        make_unique<Tensor>(runtime, input->dtype(), mem_type,
                            input->shape(), false, output_name, content_type);
    output = output_tensor.get();
    output->SetContentType(content_type, wino_blk_size);
    runtime->AllocateBufferForTensor(output, RENT_PRIVATE);
    output_tensor->SetIsWeight(true);
    ws->AddTensor(output_name, std::move(output_tensor));
  }
  // update the information
  op_def->set_input(input_idx, output_name);
  input->MarkUnused();
  return OpenCLBufferTransformer(input->memory_type(), mem_type).
      Transform(&op_context, input, content_type, mem_type, wino_blk_size,
                output);
}

}  // namespace mace
