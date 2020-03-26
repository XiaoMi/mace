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

#include "mace/core/allocator.h"

namespace mace {

MaceStatus Allocator::NewImage(const std::vector<size_t> &image_shape,
                               const DataType dt,
                               void **result) {
  MACE_UNUSED(image_shape);
  MACE_UNUSED(dt);
  MACE_UNUSED(result);
  MACE_NOT_IMPLEMENTED;
  return MaceStatus::MACE_SUCCESS;
}

void Allocator::DeleteImage(void *data) {
  MACE_UNUSED(data);
  MACE_NOT_IMPLEMENTED;
}

void *Allocator::Map(void *buffer,
                     size_t offset,
                     size_t nbytes,
                     bool finish_cmd_queue) {
  MACE_UNUSED(nbytes);
  MACE_UNUSED(finish_cmd_queue);
  return reinterpret_cast<char*>(buffer) + offset;
}

void *Allocator::MapImage(void *buffer,
                          const std::vector<size_t> &image_shape,
                          std::vector<size_t> *mapped_image_pitch,
                          bool finish_cmd_queue) {
  MACE_UNUSED(buffer);
  MACE_UNUSED(image_shape);
  MACE_UNUSED(mapped_image_pitch);
  MACE_UNUSED(finish_cmd_queue);
  MACE_NOT_IMPLEMENTED;
  return nullptr;
}

void Allocator::Unmap(void *buffer, void *mapper_ptr) {
  MACE_UNUSED(buffer);
  MACE_UNUSED(mapper_ptr);
}

#ifdef MACE_ENABLE_RPCMEM
Rpcmem *Allocator::rpcmem() {
  MACE_NOT_IMPLEMENTED;
  return nullptr;
}
#endif  // MACE_ENABLE_RPCMEM

Allocator *GetCPUAllocator() {
  static CPUAllocator allocator;
  return &allocator;
}

}  // namespace mace
