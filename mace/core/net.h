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

#ifndef MACE_CORE_NET_H_
#define MACE_CORE_NET_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

#include "mace/core/operator.h"

namespace mace {

class RunMetadata;
class Workspace;
class MemoryOptimizer;

class NetBase {
 public:
  NetBase() noexcept = default;
  virtual ~NetBase() = default;

  virtual MaceStatus Init() = 0;

  virtual MaceStatus Run(RunMetadata *run_metadata = nullptr) = 0;

 protected:
  MACE_DISABLE_COPY_AND_ASSIGN(NetBase);
};

class SerialNet : public NetBase {
 public:
  SerialNet(const OpRegistryBase *op_registry,
            const NetDef *net_def,
            Workspace *ws,
            Device *target_device,
            MemoryOptimizer * mem_optimizer);

  MaceStatus Init() override;

  MaceStatus Run(RunMetadata *run_metadata = nullptr) override;

 protected:
  Workspace *ws_;
  Device *target_device_;
  // CPU is base device.
  std::unique_ptr<Device> cpu_device_;
  std::vector<std::unique_ptr<Operation> > operators_;

  MACE_DISABLE_COPY_AND_ASSIGN(SerialNet);
};

}  // namespace mace

#endif  // MACE_CORE_NET_H_
