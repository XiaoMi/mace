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

#ifndef MACE_CORE_NET_H_
#define MACE_CORE_NET_H_

#include <memory>
#include <string>
#include <vector>

#include "mace/core/operator.h"
#include "mace/public/mace.h"

namespace mace {

class RunMetadata;
class OperatorBase;
class Workspace;

class NetBase {
 public:
  NetBase(const std::shared_ptr<const OperatorRegistryBase> op_registry,
          const std::shared_ptr<const NetDef> net_def,
          Workspace *ws,
          DeviceType type);
  virtual ~NetBase() noexcept {}

  virtual MaceStatus Run(RunMetadata *run_metadata = nullptr) = 0;

  const std::string &Name() const { return name_; }

 protected:
  std::string name_;
  const std::shared_ptr<const OperatorRegistryBase> op_registry_;

  MACE_DISABLE_COPY_AND_ASSIGN(NetBase);
};

class SerialNet : public NetBase {
 public:
  SerialNet(const std::shared_ptr<const OperatorRegistryBase> op_registry,
            const std::shared_ptr<const NetDef> net_def,
            Workspace *ws,
            DeviceType type,
            const NetMode mode = NetMode::NORMAL);

  MaceStatus Run(RunMetadata *run_metadata = nullptr) override;

 protected:
  std::vector<std::unique_ptr<OperatorBase> > operators_;
  DeviceType device_type_;

  MACE_DISABLE_COPY_AND_ASSIGN(SerialNet);
};

std::unique_ptr<NetBase> CreateNet(
    const std::shared_ptr<const OperatorRegistryBase> op_registry,
    const NetDef &net_def,
    Workspace *ws,
    DeviceType type,
    const NetMode mode = NetMode::NORMAL);
std::unique_ptr<NetBase> CreateNet(
    const std::shared_ptr<const OperatorRegistryBase> op_registry,
    const std::shared_ptr<const NetDef> net_def,
    Workspace *ws,
    DeviceType type,
    const NetMode mode = NetMode::NORMAL);

}  // namespace mace

#endif  // MACE_CORE_NET_H_
