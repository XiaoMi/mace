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


#include "mace/core/flow/flow_registry.h"

#include <memory>
#include "mace/core/flow/base_flow.h"

namespace mace {

namespace {
unsigned int FlowKey(const RuntimeType runtime_type,
                     const FlowSubType flow_sub_type) {
  auto main_type = static_cast<unsigned int>(runtime_type);
  auto sub_type = static_cast<unsigned int>(flow_sub_type);
  return ((main_type << 16) | sub_type);
}
}  // namespace

FlowRegistry::~FlowRegistry() {
  VLOG(2) << "Destroy FlowRegistry";
}

MaceStatus FlowRegistry::Register(const RuntimeType runtime_type,
                                  FlowSubType flow_sub_type,
                                  FlowCreator creator) {
  const auto flow_key = FlowKey(runtime_type, flow_sub_type);
  if (registry_.count(flow_key) > 0) {
    LOG(FATAL) << "Register a duplicate flow, main type: " << runtime_type
               << ", sub_type: " << flow_sub_type;
    return MaceStatus::MACE_RUNTIME_ERROR;
  }

  registry_.emplace(flow_key, creator);

  return MaceStatus::MACE_SUCCESS;
}

std::unique_ptr<BaseFlow> FlowRegistry::CreateFlow(
    const RuntimeType runtime_type,
    const FlowSubType flow_sub_type, FlowContext *flow_context) const {
  const auto flow_key = FlowKey(runtime_type, flow_sub_type);
  MACE_CHECK(registry_.count(flow_key) > 0);

  const FlowCreator &creator = registry_.at(flow_key);
  return creator(flow_context);
}

}  // namespace mace
