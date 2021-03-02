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

#ifndef MACE_CORE_NET_SERIAL_NET_H_
#define MACE_CORE_NET_SERIAL_NET_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

#include "mace/core/ops/operator.h"
#include "mace/core/net/base_net.h"

namespace mace {

class Workspace;
class OpRegistry;

class SerialNet : public BaseNet {
 public:
  SerialNet(const OpRegistry *op_registry,
            const NetDef *net_def,
            Workspace *ws,
            Runtime *target_runtime,
            Runtime *cpu_runtime);
  virtual ~SerialNet();

  MaceStatus Init() override;

  MaceStatus Run(RunMetadata *run_metadata = nullptr) override;

  MaceStatus AllocateIntermediateBuffer() override;

 protected:
  Workspace *ws_;
  Runtime *target_runtime_;
  // CPU is base device.
  Runtime *cpu_runtime_;
  std::vector<std::unique_ptr<Operation>> operators_;

 protected:
  MACE_DISABLE_COPY_AND_ASSIGN(SerialNet);
};

}  // namespace mace

#endif  // MACE_CORE_NET_SERIAL_NET_H_
