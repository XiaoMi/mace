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

#include <sstream>
#include <memory>
#include <string>
#include <vector>

#include "mace/core/operator.h"

namespace mace {

OperatorBase::OperatorBase(const OperatorDef &operator_def, Workspace *ws)
    : operator_ws_(ws),
      operator_def_(std::make_shared<OperatorDef>(operator_def)) {}

OpKeyBuilder::OpKeyBuilder(const char *op_name) : op_name_(op_name) {}

OpKeyBuilder &OpKeyBuilder::Device(DeviceType device) {
  device_type_ = device;
  return *this;
}

OpKeyBuilder &OpKeyBuilder::TypeConstraint(const char *attr_name,
                                           const DataType allowed) {
  type_constraint_[attr_name] = allowed;
  return *this;
}

const std::string OpKeyBuilder::Build() {
  static const std::vector<std::string> type_order = {"T"};
  std::stringstream ss;
  ss << op_name_;
  ss << device_type_;
  for (auto type : type_order) {
    ss << type << "_" << DataTypeToString(type_constraint_[type]);
  }

  return ss.str();
}

OperatorRegistryBase::~OperatorRegistryBase() {}

std::unique_ptr<OperatorBase> OperatorRegistryBase::CreateOperator(
    const OperatorDef &operator_def,
    Workspace *ws,
    DeviceType type,
    const NetMode mode) const {
  const int dtype = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
      operator_def, "T", static_cast<int>(DT_FLOAT));
  const int op_mode_i = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
      operator_def, "mode", static_cast<int>(NetMode::NORMAL));
  const NetMode op_mode = static_cast<NetMode>(op_mode_i);
  if (op_mode == mode) {
    return registry_.Create(
        OpKeyBuilder(operator_def.type().data())
            .Device(type)
            .TypeConstraint("T", static_cast<DataType>(dtype))
            .Build(),
        operator_def, ws);
  } else {
    return nullptr;
  }
}

}  // namespace mace
