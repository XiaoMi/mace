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

#ifndef MACE_CORE_RUNTIME_RUNTIME_REGISTRY_H_
#define MACE_CORE_RUNTIME_RUNTIME_REGISTRY_H_

#include <memory>
#include <unordered_map>

#include "mace/core/runtime/runtime.h"

namespace mace {

class RuntimeRegistry {
  typedef std::function<std::unique_ptr<Runtime>(RuntimeContext *)>
      RuntimeCreator;

 public:
  RuntimeRegistry() = default;
  ~RuntimeRegistry();

  MaceStatus Register(const RuntimeType runtime_type,
                      const RuntimeSubType runtime_sub_type,
                      RuntimeCreator creator);

  std::unique_ptr<Runtime> CreateRuntime(const DeviceType device_type,
                                         const RuntimeSubType runtime_sub_type,
                                         RuntimeContext *runtime_context) const;
  std::unique_ptr<Runtime> CreateRuntime(const RuntimeType runtime_type,
                                         const RuntimeSubType runtime_sub_type,
                                         RuntimeContext *runtime_context) const;

  template<class DerivedType>
  static std::unique_ptr<Runtime> DefaultCreator(
      RuntimeContext *runtime_context) {
    return make_unique<DerivedType>(runtime_context);
  }

 private:
  std::unordered_map<unsigned int, RuntimeCreator> registry_;
};

// this two function is in mace/runtimes/runtime_registry.cc
void RegisterAllRuntimes(RuntimeRegistry *runtime_registry);
RuntimeSubType SmartGetRuntimeSubType(const RuntimeType runtime_type,
                                      RuntimeContext *runtime_context);
std::unique_ptr<Runtime> SmartCreateRuntime(RuntimeRegistry *runtime_registry,
                                            const RuntimeType runtime_type,
                                            RuntimeContext *runtime_context);

#define MACE_REGISTER_RUNTIME(runtime_registry, runtime_type,             \
                              runtime_sub_type, class_name)               \
  runtime_registry->Register(runtime_type, runtime_sub_type,              \
                             RuntimeRegistry::DefaultCreator<class_name>)

}  // namespace mace

#endif  // MACE_CORE_RUNTIME_RUNTIME_REGISTRY_H_
