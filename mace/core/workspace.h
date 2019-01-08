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

#ifndef MACE_CORE_WORKSPACE_H_
#define MACE_CORE_WORKSPACE_H_

#include <map>
#include <string>
#include <vector>
#include <memory>

#include "mace/core/device.h"
#include "mace/core/preallocated_pooled_allocator.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"

namespace mace {

class MemoryOptimizer;

class Workspace {
 public:
  typedef std::map<std::string, std::unique_ptr<Tensor>> TensorMap;

  Workspace();
  ~Workspace() {}

  Tensor *CreateTensor(const std::string &name,
                       Allocator *alloc,
                       DataType type,
                       bool is_weight = false);

  inline bool HasTensor(const std::string &name) const {
    return tensor_map_.find(name) != tensor_map_.end();
  }

  inline bool diffused_buffer() const {
    return diffused_buffer_;
  }

  const Tensor *GetTensor(const std::string &name) const;

  Tensor *GetTensor(const std::string &name);

  std::vector<std::string> Tensors() const;

  MaceStatus LoadModelTensor(const NetDef &net_def,
                             Device *device,
                             const unsigned char *model_data);

  MaceStatus PreallocateOutputTensor(const NetDef &net_def,
                                     const MemoryOptimizer *mem_optimizer,
                                     Device *device);

  void RemoveUnusedBuffer();

  void RemoveAndReloadBuffer(const NetDef &net_def,
                             const unsigned char *model_data,
                             Allocator *alloc);

  void RemoveTensor(const std::string &name);

 private:
  TensorMap tensor_map_;

  std::unique_ptr<BufferBase> tensor_buffer_;

  PreallocatedPooledAllocator preallocated_allocator_;

  bool diffused_buffer_;

  MACE_DISABLE_COPY_AND_ASSIGN(Workspace);
};

}  // namespace mace

#endif  // MACE_CORE_WORKSPACE_H_
