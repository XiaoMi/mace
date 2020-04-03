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


#ifndef MACE_CORE_REGISTRY_OP_REGISTRATION_INFO_H_
#define MACE_CORE_REGISTRY_OP_REGISTRATION_INFO_H_

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "mace/core/ops/operator.h"
#include "mace/proto/mace.pb.h"

namespace mace {
class OpConstructContext;
class OpConditionContext;

class OpRegistrationInfo {
 public:
  typedef std::function<std::unique_ptr<Operation>(OpConstructContext *)>
      OpCreator;
  typedef std::function<std::set<DeviceType>(OpConditionContext *)>
      DevicePlacer;
  typedef std::function<void(OpConditionContext *)> MemoryTypeSetter;
  typedef std::function<std::vector<DataFormat>(OpConditionContext *)>
      DataFormatSelector;

  OpRegistrationInfo();

  void AddDevice(DeviceType);

  void Register(const std::string &key, OpCreator creator);

  std::set<DeviceType> devices;
  std::unordered_map<std::string, OpCreator> creators;
  DevicePlacer device_placer;
  MemoryTypeSetter memory_type_setter;
  DataFormatSelector data_format_selector;
};
}  // namespace mace

#endif  // MACE_CORE_REGISTRY_OP_REGISTRATION_INFO_H_
