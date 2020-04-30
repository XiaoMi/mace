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

#ifndef MACE_CORE_FLOW_FLOW_REGISTRY_H_
#define MACE_CORE_FLOW_FLOW_REGISTRY_H_

#include <memory>
#include <unordered_map>

#include "mace/core/types.h"
#include "mace/public/mace.h"
#include "mace/utils/memory.h"

namespace mace {

class BaseFlow;
struct FlowContext;

class FlowRegistry {
  typedef std::function<std::unique_ptr<BaseFlow>(FlowContext *)> FlowCreator;

 public:
  FlowRegistry() = default;
  ~FlowRegistry();

  MaceStatus Register(const RuntimeType runtime_type,
                      const FlowSubType flow_sub_type,
                      FlowCreator creator);

  std::unique_ptr<BaseFlow> CreateFlow(const RuntimeType runtime_type,
                                       const FlowSubType flow_sub_type,
                                       FlowContext *flow_context) const;

  template<class FlowClass>
  static std::unique_ptr<BaseFlow> DefaultCreator(FlowContext *flow_context) {
    return make_unique<FlowClass>(flow_context);
  }

 private:
  std::unordered_map<unsigned int, FlowCreator> registry_;
};

// this function is in mace/flows/flow_registry.cc
void RegisterAllFlows(FlowRegistry *runtime_registry);

#define MACE_REGISTER_FLOW(flow_registry, runtime_type, \
                           flow_sub_type, class_name)   \
  flow_registry->Register(runtime_type, flow_sub_type,  \
                          FlowRegistry::DefaultCreator<class_name>)

}  // namespace mace

#endif  // MACE_CORE_FLOW_FLOW_REGISTRY_H_
