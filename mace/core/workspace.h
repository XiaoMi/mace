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

#include "mace/core/runtime/runtime.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"

namespace mace {

class BaseFlow;
class OpDelegatorRegistry;

class Workspace {
 public:
  typedef std::map<std::string, std::unique_ptr<Tensor>> TensorMap;

  explicit Workspace(const OpDelegatorRegistry *registry, BaseFlow *flow);
  ~Workspace();

  const BaseFlow *GetMaceFlow() const;

  Tensor *CreateTensor(const std::string &name, Runtime *runtime,
                       DataType type, bool is_weight = false,
                       MemoryType mem_type = MEMORY_NONE,
                       BufferContentType content_type = IN_OUT_CHANNEL);

  inline bool HasTensor(const std::string &name) const {
    return tensor_map_.find(name) != tensor_map_.end();
  }

  inline bool diffused_buffer() const {
    return diffused_buffer_;
  }

  Tensor *GetTensor(const std::string &name) const;
  MaceStatus AddTensor(const std::string &name, std::unique_ptr<Tensor> tensor);

  std::vector<std::string> Tensors() const;

  MaceStatus LoadModelTensor(const NetDef &net_def, Runtime *runtime,
                             const unsigned char *model_data,
                             const index_t model_data_size);

  MaceStatus AddQuantizeInfoForOutputTensor(const NetDef &net_def,
                                            Runtime *runtime);

  void RemoveUnusedBuffer();

  void RemoveAndReloadBuffer(const NetDef &net_def,
                             const unsigned char *model_data,
                             Runtime *runtime);

  void RemoveTensor(const std::string &name);

  const OpDelegatorRegistry *GetDelegatorRegistry() const;

  MaceStatus ReleaseIntermediateBuffer(Runtime **runtimes, size_t size,
                                       Runtime *cpu_runtime);

 private:
  TensorMap tensor_map_;
  std::unique_ptr<Buffer> tensor_buffer_;
  bool diffused_buffer_;

  const OpDelegatorRegistry *op_delegator_registry_;
  BaseFlow *parent_flow_;

  MACE_DISABLE_COPY_AND_ASSIGN(Workspace);
};

}  // namespace mace

#endif  // MACE_CORE_WORKSPACE_H_
