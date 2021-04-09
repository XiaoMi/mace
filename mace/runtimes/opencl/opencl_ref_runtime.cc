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

#include "mace/runtimes/opencl/opencl_ref_runtime.h"

#include "mace/core/runtime/runtime_registry.h"

namespace mace {

OpenclRefRuntime::OpenclRefRuntime(RuntimeContext *runtime_context)
    : OpenclRuntime(runtime_context) {}

MaceStatus OpenclRefRuntime::Init(const MaceEngineCfgImpl *engine_config,
                                  const MemoryType mem_type) {
  MACE_RETURN_IF_ERROR(OpenclRuntime::Init(engine_config, mem_type));

  buffer_allocator_ =
      make_unique<OpenclBufferAllocator>(opencl_executor_.get());
  image_allocator_ = make_unique<OpenclImageAllocator>(opencl_executor_.get());
  buffer_manager_ = make_unique<GeneralMemoryManager>(buffer_allocator_.get());
  image_manager_ = make_unique<OpenclImageManager>(image_allocator_.get());

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus OpenclRefRuntime::MapBuffer(Buffer *buffer, bool wait_for_finish) {
  MACE_LATENCY_LOGGER(1, "OpenclRefRuntime Map OpenCL buffer/Image");
  MACE_UNUSED(wait_for_finish);

  void *mapped_ptr = nullptr;
  cl_int error = CL_INVALID_VALUE;
  if (buffer->mem_type == MemoryType::GPU_BUFFER) {
    auto cl_buffer = buffer->mutable_memory<cl::Buffer>();
    auto queue = opencl_executor_->command_queue();
    // TODO(heliangliang) Non-blocking call
    mapped_ptr = queue.enqueueMapBuffer(
        *cl_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
        buffer->offset(), buffer->bytes(), nullptr, nullptr, &error);
  } else if (buffer->mem_type == MemoryType::GPU_IMAGE) {
    MACE_CHECK(buffer->dims.size() == 2,
               "Just support map 2d image, the shape is: ",
               MakeString(buffer->dims));
    auto cl_image = buffer->mutable_memory<cl::Image2D>();
    std::array<size_t, 3> origin = {{0, 0, 0}};
    std::array<size_t, 3> region = {{static_cast<size_t>(buffer->dims[0]),
                                     static_cast<size_t>(buffer->dims[1]), 1}};

    cl::size_type row_pitch = 0;
    cl::size_type slice_pitch = 0;
    mapped_ptr = opencl_executor_->command_queue().enqueueMapImage(
        *cl_image, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, origin, region,
        &row_pitch, &slice_pitch, nullptr, nullptr, &error);
  } else {
    LOG(FATAL) << "invalid MemoryType: " << static_cast<int>(buffer->mem_type);
  }

  if (error != CL_SUCCESS) {
    LOG(FATAL) << "MapBuffer failed, error: " << OpenCLErrorToString(error)
               << ", MemoryType: " << static_cast<int>(buffer->mem_type);
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  } else {
    buffer->SetHost(mapped_ptr);
    return MaceStatus::MACE_SUCCESS;
  }
}

MaceStatus OpenclRefRuntime::UnMapBuffer(Buffer *buffer) {
  MACE_LATENCY_LOGGER(1, "OpenclRefRuntime Unmap OpenCL buffer/Image");
  MACE_CHECK(buffer->mem_type == MemoryType::GPU_BUFFER ||
      buffer->mem_type == MemoryType::GPU_IMAGE);

  auto cl_buffer = buffer->mutable_memory<cl::Buffer>();
  auto queue = opencl_executor_->command_queue();
  cl_int error = queue.enqueueUnmapMemObject(
      *cl_buffer, buffer->mutable_data<void>(), nullptr, nullptr);
  if (error != CL_SUCCESS) {
    LOG(ERROR) << "OpenclRefRuntime Unmap buffer failed, error: "
               << OpenCLErrorToString(error);
  }
  buffer->SetHost(nullptr);
  return MaceStatus::MACE_SUCCESS;
}

MemoryManager *OpenclRefRuntime::GetMemoryManager(MemoryType mem_type) {
  MemoryManager *buffer_manager = nullptr;
  if (mem_type == MemoryType::GPU_BUFFER) {
    buffer_manager = buffer_manager_.get();
  } else if (mem_type == MemoryType::GPU_IMAGE) {
    buffer_manager = image_manager_.get();
  } else {
    MACE_CHECK(false, "OpenclRefRuntime::GetMemoryManagerByMemType",
               "find an invalid mem type:", mem_type);
  }

  return buffer_manager;
}

void RegisterOpenclRefRuntime(RuntimeRegistry *runtime_registry) {
  MACE_UNUSED(runtime_registry);
  MACE_REGISTER_RUNTIME(runtime_registry, RuntimeType::RT_OPENCL,
                        RuntimeSubType::RT_SUB_REF, OpenclRefRuntime);
}

}  // namespace mace
