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

#ifndef MACE_CORE_OPS_OP_CONDITION_BUILDER_H_
#define MACE_CORE_OPS_OP_CONDITION_BUILDER_H_

#include <memory>
#include <string>

#include "mace/core/registry/op_registration_info.h"
#include "mace/core/types.h"

namespace mace {
class OpConditionBuilder {
 public:
  explicit OpConditionBuilder(const std::string &type);

  const std::string type() const;

  OpConditionBuilder &SetDevicePlacerFunc(
      OpRegistrationInfo::DevicePlacer placer);

  // If you set input memory type for specified Op,
  // you must call OpConditionContext::set_output_mem_type
  OpConditionBuilder &SetInputMemoryTypeSetter(
      OpRegistrationInfo::MemoryTypeSetter setter);

  OpConditionBuilder &SetInputsDataFormatSelector(
      OpRegistrationInfo::DataFormatSelector selector);

  void Finalize(OpRegistrationInfo *info) const;

 private:
  std::string type_;
  OpRegistrationInfo::DevicePlacer placer_;
  OpRegistrationInfo::MemoryTypeSetter memory_type_setter_;
  OpRegistrationInfo::DataFormatSelector data_format_selector_;
};

}  // namespace mace

#endif  // MACE_CORE_OPS_OP_CONDITION_BUILDER_H_
