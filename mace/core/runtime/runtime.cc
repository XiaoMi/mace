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

#include "mace/core/runtime/runtime.h"

#include <functional>
#include <utility>

#include "mace/core/flow/base_flow.h"
#include "mace/core/memory/buffer.h"
#include "mace/core/memory/slice.h"
#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"

namespace mace {

namespace {
enum InterMemState {
  CREATED,   // Buffers are created, but the opencl kernels need to create.
  STABLE,    // Buffers and kernels are created.
  RELEASED,  // Buffers are released.
};

}  // namespace

Runtime::Runtime(RuntimeContext *runtime_context)
    : thread_pool_(runtime_context->thread_pool),
      has_ever_released_inter_mem_(false) {}

Runtime::~Runtime() {}

MaceStatus Runtime::Init(const MaceEngineCfgImpl *engine_config,
                         const MemoryType mem_type) {
  MACE_UNUSED(engine_config);
  MACE_UNUSED(mem_type);
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Runtime::BeforeRun(MaceEngineCfgImpl *config) {
  MACE_UNUSED(config);
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Runtime::AfterRun() {
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Runtime::MapBuffer(Buffer *buffer, bool wait_for_finish) {
  MACE_UNUSED(wait_for_finish);
  buffer->SetHost(buffer->mutable_memory<uint8_t>() + buffer->offset());

  return MaceStatus::MACE_SUCCESS;
}

bool Runtime::CanReuseBuffer(
    const Buffer *buffer, const std::vector<index_t> &shape,
    const BufferContentType content_type, const unsigned int content_param) {
  MACE_UNUSED(content_type);
  MACE_UNUSED(content_param);
  auto size_bytes = std::accumulate(shape.begin(), shape.end(),
                                    1, std::multiplies<index_t>());
  MemoryManager *memory_manager = GetMemoryManager(buffer->mem_type);
  auto real_shape = memory_manager->GetMemoryRealSize(buffer->memory<void>());
  MACE_CHECK(real_shape.size() == 1, "Only support dim 1");
  return (size_bytes <= real_shape[0]);
}

MaceStatus Runtime::UnMapBuffer(Buffer *buffer) {
  MACE_UNUSED(buffer);
  return MaceStatus::MACE_SUCCESS;
}

MemoryType Runtime::GetBaseMemoryType() {
  return MemoryType::CPU_BUFFER;
}

MemoryType Runtime::GetUsedMemoryType() {
  return GetBaseMemoryType();
}

utils::ThreadPool &Runtime::thread_pool() {
  return *thread_pool_;
}

std::unique_ptr<Buffer> Runtime::ObtainBuffer(const MemInfo &info,
                                              BufRentType rent_type) {
  MACE_CHECK(rent_type != BufRentType::RENT_SLICE,
             "you can't obtain a slice buffer");
  MemoryManager *memory_manager = GetMemoryManager(info.mem_type);
  void *ptr = memory_manager->ObtainMemory(info, rent_type);

  auto buffer = make_unique<Buffer>(info, ptr);
  if (info.mem_type == MemoryType::CPU_BUFFER) {
    buffer->SetHost(buffer->mutable_memory<uint8_t>() + buffer->offset());
  }
  return buffer;
}

void Runtime::ReleaseBuffer(Buffer *buffer, BufRentType rent_type) {
  MACE_CHECK(rent_type != BufRentType::RENT_SLICE,
             "you can't release a slice buffer");
  MemoryManager *memory_manager = GetMemoryManager(buffer->mem_type);
  memory_manager->ReleaseMemory(buffer->mutable_memory<void>(), rent_type);
}

void Runtime::ReleaseAllBuffer(BufRentType rent_type, bool del_buf) {
  auto mem_type = GetUsedMemoryType();
  MemoryManager *memory_manager = GetMemoryManager(mem_type);
  memory_manager->ReleaseAllMemory(rent_type, del_buf);

  auto base_mem_type = GetBaseMemoryType();
  if (base_mem_type != mem_type) {
    MemoryManager *memory_manager = GetMemoryManager(base_mem_type);
    memory_manager->ReleaseAllMemory(rent_type, del_buf);
  }
}

std::vector<index_t> Runtime::ComputeBufDimFromTensorDim(
    const std::vector<index_t> &dims, MemoryType mem_type,
    const BufferContentType content_type, const unsigned int content_param) {
  MACE_UNUSED(mem_type);
  MACE_UNUSED(content_type);
  MACE_UNUSED(content_param);
  auto size = std::accumulate(dims.begin(), dims.end(),
                              1, std::multiplies<index_t>());
  return {size};
}

DataType Runtime::GetComputeDataType(const NetDef &net_def,
                                     const ConstTensor &const_tensor) {
  MACE_UNUSED(net_def);
  return const_tensor.data_type();
}

MaceStatus Runtime::AllocateBufferForTensor(
    Tensor *tensor, BufRentType rent_type,
    Buffer *slice_parent, index_t offset_bytes) {
  std::unique_ptr<Buffer> buffer;
  auto data_type = tensor->dtype();

  if (rent_type == BufRentType::RENT_SLICE) {
    MACE_CHECK(slice_parent != nullptr);
    auto mem_type = slice_parent->mem_type;
    auto mem_dims = ComputeBufDimFromTensorDim(tensor->shape(), mem_type,
                                               tensor->content_type_,
                                               tensor->content_param_);
    const index_t bytes = tensor->raw_size();
    MACE_CHECK(offset_bytes + bytes <= slice_parent->bytes());
    buffer.reset(new Slice(mem_type, data_type, mem_dims,
                           slice_parent->mutable_memory<void>(), offset_bytes));
  } else {
    auto mem_type = tensor->memory_type();
    auto mem_dims = ComputeBufDimFromTensorDim(tensor->shape(), mem_type,
                                               tensor->content_type_,
                                               tensor->content_param_);
    MemoryManager *memory_manager = GetMemoryManager(mem_type);
    buffer.reset(new Buffer(mem_type, data_type, mem_dims, nullptr));
    buffer->SetBuf(memory_manager->ObtainMemory(*buffer, rent_type));
  }

  MACE_CHECK(buffer->memory<void>() != nullptr);
  SetBufferToTensor(std::move(buffer), tensor);
  return MaceStatus::MACE_SUCCESS;
}

void Runtime::ReleaseBufferForTensor(Tensor *tensor,
                                     const BufRentType rent_type) {
  if (rent_type != BufRentType::RENT_SLICE) {
    MemoryManager *memory_manager = GetMemoryManager(tensor->buffer_->mem_type);
    memory_manager->ReleaseMemory(tensor->buffer_->mutable_memory<void>(),
                                  rent_type);
  }
  if (rent_type != BufRentType::RENT_SHARE) {
    tensor->buffer_.reset();
  }
}

void Runtime::SetBufferToTensor(
    std::unique_ptr<Buffer> buffer, Tensor *tensor) {
  if (buffer->mem_type == MemoryType::CPU_BUFFER) {
    buffer->SetHost(buffer->mutable_memory<uint8_t>() + buffer->offset());
  }
  tensor->buffer_ = std::move(buffer);
}

void Runtime::ReleaseIntermediateBuffer(const BaseEngine *engine) {
  has_ever_released_inter_mem_ = true;
  auto state = inter_mem_state_map_.at(engine);
  MACE_CHECK(state == InterMemState::CREATED || state == InterMemState::STABLE);
  inter_mem_state_map_[engine] = InterMemState::RELEASED;

  for (auto info : inter_mem_state_map_) {
    if (info.second != InterMemState::RELEASED) {
      return;  // The runtime is shared by another engine, can not release.
    }
  }
  ReleaseAllBuffer(BufRentType::RENT_SHARE, true);
}

void Runtime::OnAllocateIntermediateBuffer(const BaseEngine *engine) {
  MACE_CHECK(inter_mem_state_map_.count(engine) == 0 ||
      inter_mem_state_map_.at(engine) == InterMemState::RELEASED);
  inter_mem_state_map_[engine] = InterMemState::CREATED;
}

void Runtime::OnIntermediateBufferUsed(const BaseEngine *engine) {
  MACE_CHECK(inter_mem_state_map_.count(engine) == 0 ||
      inter_mem_state_map_.at(engine) != InterMemState::RELEASED);
  inter_mem_state_map_[engine] = InterMemState::STABLE;
}

bool Runtime::IntermediateBufferCreated(const BaseEngine *engine) const {
  return inter_mem_state_map_.count(engine) == 0 ||
      inter_mem_state_map_.at(engine) != InterMemState::RELEASED;
}

bool Runtime::IntermediateBufferStable(const OpContext *op_context) const {
  if (!has_ever_released_inter_mem_) {
    return true;  // No ReleaseIntermediateBuffer has been invoked.
  }

  auto engine = op_context->workspace()->GetMaceFlow()->GetMaceEngine();
  MACE_CHECK(inter_mem_state_map_.count(engine) != 0);
  return inter_mem_state_map_.at(engine) == InterMemState::STABLE;
}

}  // namespace mace
