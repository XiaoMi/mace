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

#include "mace/core/op_def_registry.h"
#include "mace/utils/logging.h"

namespace mace {

void AddOpRegistrar(OpDefRegistryBase *registry,
                    const OpRegistrationBuilder &builder) {
  registry->AddRegistrar(
      builder.name(),
      [builder](OpRegistrationInfo *info){
        builder.Finalize(info);
      });
}

OpRegistrationBuilder::OpRegistrationBuilder(const std::string name)
    : name_(name) {}

const std::string OpRegistrationBuilder::name() const { return name_; }

OpRegistrationBuilder &OpRegistrationBuilder::SetDevicePlaceFunc(
    std::vector<DeviceType> (*func)()) {
  info_.device_place_func_ = func;
  return *this;
}

void OpRegistrationBuilder::Finalize(OpRegistrationInfo *info) const {
  *info = info_;
}

void OpDefRegistryBase::AddRegistrar(const std::string name,
                                    const OpRegistrar &registrar) {
  registrar_.emplace(name, registrar);
}

MaceStatus OpDefRegistryBase::Register(const std::string &name) {
  VLOG(3) << "Registering operation definition: " << name;
  if (registry_.find(name) != registry_.end()) {
    return MaceStatus::MACE_SUCCESS;
  }
  auto iter = registrar_.find(name);
  if (iter == registrar_.end()) {
    return MaceStatus(MaceStatus::MACE_INVALID_ARGS,
                      "MACE do not support the operation: " + name);
  }
  registry_.emplace(
      name, std::unique_ptr<OpRegistrationInfo>(new OpRegistrationInfo()));
  iter->second(registry_[name].get());
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus OpDefRegistryBase::Find(const std::string &name,
                                  const OpRegistrationInfo **info) {
  auto iter = registry_.find(name);
  if (iter == registry_.end()) {
    *info = nullptr;
    return MaceStatus(MaceStatus::MACE_INVALID_ARGS,
                      "Mace do not support the operation: " + name);
  }
  *info = iter->second.get();
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace mace
