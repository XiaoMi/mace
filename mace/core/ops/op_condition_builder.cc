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

#include "mace/core/ops/op_condition_builder.h"

namespace mace {

OpConditionBuilder::OpConditionBuilder(const std::string &type)
    : type_(type) {}

const std::string OpConditionBuilder::type() const {
  return type_;
}

OpConditionBuilder &OpConditionBuilder::SetDevicePlacerFunc(
    OpRegistrationInfo::DevicePlacer placer) {
  placer_ = placer;
  return *this;
}

OpConditionBuilder &OpConditionBuilder::SetInputMemoryTypeSetter(
    OpRegistrationInfo::MemoryTypeSetter setter) {
  memory_type_setter_ = setter;
  return *this;
}

OpConditionBuilder &OpConditionBuilder::SetInputsDataFormatSelector(
    OpRegistrationInfo::DataFormatSelector selector) {
  data_format_selector_ = selector;
  return *this;
}

void OpConditionBuilder::Finalize(OpRegistrationInfo *info) const {
  if (info != nullptr) {
    if (placer_) {
      info->device_placer = placer_;
    }
    if (memory_type_setter_) {
      info->memory_type_setter = memory_type_setter_;
    }

    if (data_format_selector_) {
      info->data_format_selector = data_format_selector_;
    }
  }
}

}  // namespace mace
