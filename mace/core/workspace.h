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

#ifndef MACE_CORE_WORKSPACE_H_
#define MACE_CORE_WORKSPACE_H_

#include <map>
#include <string>
#include <vector>
#include <memory>

#include "mace/core/preallocated_pooled_allocator.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"

namespace mace {

class Workspace {
 public:
  typedef std::map<std::string, std::unique_ptr<Tensor>> TensorMap;

  Workspace();
  ~Workspace() {}

  Tensor *CreateTensor(const std::string &name,
                       Allocator *alloc,
                       DataType type);

  inline bool HasTensor(const std::string &name) const {
    return tensor_map_.find(name) != tensor_map_.end();
  }

  const Tensor *GetTensor(const std::string &name) const;

  Tensor *GetTensor(const std::string &name);

  std::vector<std::string> Tensors() const;

  MaceStatus LoadModelTensor(const NetDef &net_def,
                             DeviceType type,
                             const unsigned char *model_data);

  ScratchBuffer *GetScratchBuffer(DeviceType device_type);

  void RemoveUnusedBuffer();

  void RemoveAndReloadBuffer(const NetDef &net_def,
                             const unsigned char *model_data);

 private:
  MaceStatus CreateOutputTensorBuffer(const NetDef &net_def,
                                      DeviceType device_type);

  TensorMap tensor_map_;

  std::unique_ptr<BufferBase> tensor_buffer_;

  PreallocatedPooledAllocator preallocated_allocator_;

  std::unique_ptr<ScratchBuffer> host_scratch_buffer_;
  bool fused_buffer_;

  MACE_DISABLE_COPY_AND_ASSIGN(Workspace);
};

}  // namespace mace

#endif  // MACE_CORE_WORKSPACE_H_
