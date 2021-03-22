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

#include "mace/runtimes/opencl/opencl_runtime.h"

#include <memory>
#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/proto/arg_helper.h"
#include "mace/core/proto/net_def_helper.h"
#include "mace/core/runtime/runtime_registry.h"
#include "mace/core/tensor.h"
#include "mace/proto/mace.pb.h"
#include "mace/runtimes/opencl/core/opencl_context.h"
#include "mace/runtimes/opencl/opencl_image_allocator.h"
#include "mace/utils/memory.h"

namespace mace {

OpenclRuntime::OpenclRuntime(RuntimeContext *runtime_context)
    : Runtime(runtime_context),
      opencl_executor_(nullptr),
      used_memory_type_(MemoryType::GPU_IMAGE) {}

MaceStatus OpenclRuntime::Init(const MaceEngineCfgImpl *engine_config,
                               const MemoryType mem_type) {
  MACE_RETURN_IF_ERROR(CreateOpenclExecutorAndInit(engine_config));
  if (!opencl_executor_->is_opencl_avaliable()) {
    LOG(WARNING) << "The device does not support OpenCL";
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  }

  used_memory_type_ = mem_type;

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus OpenclRuntime::CreateOpenclExecutorAndInit(
    const MaceEngineCfgImpl *engine_config) {
  if (opencl_executor_ == nullptr) {
    opencl_executor_ = make_unique<OpenclExecutor>(
        engine_config->opencl_context()->opencl_cache_storage(),
        engine_config->opencl_context()->opencl_binary_storage(),
        engine_config->opencl_context()->opencl_tuner(),
        engine_config->opencl_context()->opencl_cache_reuse_policy());
    MACE_RETURN_IF_ERROR(opencl_executor_->Init(
        engine_config->gpu_priority_hint(), engine_config->gpu_perf_hint()));
  }
  return MaceStatus::MACE_SUCCESS;
}

RuntimeType OpenclRuntime::GetRuntimeType() {
  return RuntimeType::RT_OPENCL;
}

MemoryType OpenclRuntime::GetBaseMemoryType() {
  return MemoryType::GPU_BUFFER;
}

MemoryType OpenclRuntime::GetUsedMemoryType() {
  return used_memory_type_;
}

void OpenclRuntime::SetUsedMemoryType(MemoryType mem_type) {
  used_memory_type_ = mem_type;
}

std::unique_ptr<Buffer> OpenclRuntime::MakeSliceBuffer(
    const NetDef &net_def, const unsigned char *model_data,
    const index_t model_data_size) {
  MACE_UNUSED(net_def);
  MACE_ASSERT(model_data != nullptr && model_data_size > 0);

  auto max_size = GetOpenclExecutor()->GetDeviceMaxMemAllocSize();
  if (max_size <= static_cast<uint64_t>(model_data_size)) {
    return nullptr;
  }

  MemoryType mem_type = MemoryType::GPU_BUFFER;
  MemoryManager *memory_manager = GetMemoryManager(mem_type);
  auto buffer = make_unique<Buffer>(mem_type, DataType::DT_UINT8,
                                    std::vector<index_t>({model_data_size}),
                                    nullptr);
  buffer->SetBuf(memory_manager->ObtainMemory(*buffer, RENT_PRIVATE));

  this->MapBuffer(buffer.get(), false);
  memcpy(buffer->mutable_data<void>(), model_data, model_data_size);
  this->UnMapBuffer(buffer.get());

  return buffer;
}

DataType OpenclRuntime::GetComputeDataType(const NetDef &net_def,
                                           const ConstTensor &const_tensor) {
  auto is_quantize_model = NetDefHelper::IsQuantizedModel(net_def);
  if (!is_quantize_model && const_tensor.quantized()) {
    if (net_def.data_type() != DataType::DT_FLOAT) {
      return DataType::DT_HALF;
    } else {
      return DataType::DT_FLOAT;
    }
  }

  return Runtime::GetComputeDataType(net_def, const_tensor);
}

OpenclExecutor *OpenclRuntime::GetOpenclExecutor() {
  return opencl_executor_.get();
}

std::vector<index_t> OpenclRuntime::ComputeBufDimFromTensorDim(
    const std::vector<index_t> &dims, const MemoryType mem_type,
    const BufferContentType content_type, const unsigned int content_param) {
  if (mem_type == MemoryType::GPU_IMAGE) {
    std::vector<size_t> image_shape;
    const int wino_blk_size = static_cast<const int>(content_param);
    OpenCLUtil::CalImage2DShape(dims, content_type,
                                &image_shape, wino_blk_size);
    return Buffer::IndexT(image_shape);
  } else if (mem_type == MemoryType::GPU_BUFFER) {
    return Runtime::ComputeBufDimFromTensorDim(dims, mem_type,
                                               content_type, content_param);
  } else {
    MACE_CHECK(false, "invalid tensor mem_type: ", mem_type);
    return {};
  }
}

}  // namespace mace
