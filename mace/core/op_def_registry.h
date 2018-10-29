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

#ifndef MACE_CORE_OP_DEF_REGISTRY_H_
#define MACE_CORE_OP_DEF_REGISTRY_H_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mace/proto/mace.pb.h"
#include "mace/public/mace.h"
#include "mace/utils/utils.h"

namespace mace {

// Device placement function
typedef std::function<std::vector<DeviceType>()> DevicePlaceFunc;

struct OpRegistrationInfo {
  OpRegistrationInfo() = default;
  explicit OpRegistrationInfo(const DevicePlaceFunc &func)
      : device_place_func_(func) {}

  DevicePlaceFunc device_place_func_;
};

class OpRegistrationBuilder {
 public:
  explicit OpRegistrationBuilder(const std::string name);

  const std::string name() const;

  OpRegistrationBuilder &SetDevicePlaceFunc(
      std::vector<DeviceType> (*func)());

  void Finalize(OpRegistrationInfo *info) const;
 private:
  std::string name_;
  OpRegistrationInfo info_;
};

class OpDefRegistryBase {
 public:
  typedef std::function<void(OpRegistrationInfo *)> OpRegistrar;
  OpDefRegistryBase() = default;
  virtual ~OpDefRegistryBase() = default;
  void AddRegistrar(const std::string name, const OpRegistrar &registrar);
  MaceStatus Register(const std::string &name);
  MaceStatus Find(const std::string &name, const OpRegistrationInfo **info);

 private:
  std::unordered_map<std::string, OpRegistrar> registrar_;
  std::unordered_map<
      std::string,
      std::unique_ptr<OpRegistrationInfo>> registry_;
  MACE_DISABLE_COPY_AND_ASSIGN(OpDefRegistryBase);
};

void AddOpRegistrar(OpDefRegistryBase *registry,
                    const OpRegistrationBuilder &builder);

#define MACE_REGISTER_OP_DEF(op_def_registry, builder) \
  AddOpRegistrar(op_def_registry, builder)

}  // namespace mace

#endif  // MACE_CORE_OP_DEF_REGISTRY_H_
