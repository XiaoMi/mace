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


#include "mace/core/runtime/runtime_registry.h"

#include <memory>

namespace mace {

namespace {
unsigned int RuntimeKey(const RuntimeType runtime_type,
                        const RuntimeSubType runtime_sub_type) {
  auto main_type = static_cast<unsigned int>(runtime_type);
  auto sub_type = static_cast<unsigned int>(runtime_sub_type);
  return ((main_type << 16) | sub_type);
}
}  // namespace

RuntimeRegistry::~RuntimeRegistry() {
  VLOG(2) << "Destroy RuntimeRegistry";
}

MaceStatus RuntimeRegistry::Register(const RuntimeType runtime_type,
                                     RuntimeSubType runtime_sub_type,
                                     RuntimeCreator creator) {
  const auto runtime_key = RuntimeKey(runtime_type, runtime_sub_type);
  if (registry_.count(runtime_key) > 0) {
    LOG(FATAL) << "Register a duplicate runtime, main type: " << runtime_type
               << ", sub_type: " << runtime_sub_type;
    return MaceStatus::MACE_RUNTIME_ERROR;
  }

  registry_.emplace(runtime_key, creator);

  return MaceStatus::MACE_SUCCESS;
}

std::unique_ptr<Runtime> RuntimeRegistry::CreateRuntime(
    DeviceType device_type, RuntimeSubType runtime_sub_type,
    RuntimeContext *runtime_context) const {
  return CreateRuntime(static_cast<RuntimeType>(device_type),
                       runtime_sub_type, runtime_context);
}

std::unique_ptr<Runtime> RuntimeRegistry::CreateRuntime(
    const RuntimeType runtime_type, const RuntimeSubType runtime_sub_type,
    RuntimeContext *runtime_context) const {
  const auto runtime_key = RuntimeKey(runtime_type, runtime_sub_type);
  MACE_CHECK(registry_.count(runtime_key) > 0,
             "Current MACE doesn't support the runtime type. runtime_type: ",
             runtime_type, ", runtime_sub_type: ", runtime_sub_type,
             ", perhaps you have specified A type runtime in yml file to"
             " convert model but specified B type runtime in yml file to"
             " run model");

  const RuntimeCreator &creator = registry_.at(runtime_key);
  return creator(runtime_context);
}

}  // namespace mace
