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

#ifndef MACE_CORE_OPS_OP_CONSTRUCT_CONTEXT_H_
#define MACE_CORE_OPS_OP_CONSTRUCT_CONTEXT_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mace/core/arg_helper.h"
#include "mace/core/types.h"
#include "mace/proto/mace.pb.h"

namespace mace {
class Device;
class Workspace;

// memory_optimizer, device
class OpConstructContext {
  typedef std::unordered_map<std::string, std::vector<index_t>> TensorShapeMap;

 public:
  explicit OpConstructContext(Workspace *ws);
  ~OpConstructContext() = default;

  void set_operator_def(std::shared_ptr<OperatorDef> operator_def);

  std::shared_ptr<OperatorDef> operator_def() const {
    return operator_def_;
  }

  Workspace *workspace() const {
    return ws_;
  }

  void set_device(Device *device) {
    device_ = device;
  }

  Device *device() const {
    return device_;
  }
#ifdef MACE_ENABLE_OPENCL
  inline MemoryType GetOpMemoryType() const {
    return static_cast<MemoryType>(
        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
            *operator_def_, OutputMemoryTypeTagName(),
            static_cast<int>(MemoryType::CPU_BUFFER)));
  }
#endif  // MACE_ENABLE_OPENCL

 private:
  std::shared_ptr<OperatorDef> operator_def_;
  Workspace *ws_;
  Device *device_;
};

}  // namespace mace

#endif  // MACE_CORE_OPS_OP_CONSTRUCT_CONTEXT_H_
