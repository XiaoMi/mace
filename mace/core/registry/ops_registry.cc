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

#include "mace/core/registry/ops_registry.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace mace {
namespace {
class OpKeyBuilder {
 public:
  explicit OpKeyBuilder(const std::string &op_name);

  OpKeyBuilder &Device(DeviceType device);

  OpKeyBuilder &TypeConstraint(const char *attr_name,
                               DataType allowed);

  const std::string Build();

 private:
  std::string op_name_;
  DeviceType device_type_;
  std::map<std::string, DataType> type_constraint_;
};

OpKeyBuilder::OpKeyBuilder(const std::string &op_name) : op_name_(op_name) {}

OpKeyBuilder &OpKeyBuilder::Device(DeviceType device) {
  device_type_ = device;
  return *this;
}

OpKeyBuilder &OpKeyBuilder::TypeConstraint(const char *attr_name,
                                           DataType allowed) {
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
}  // namespace

MaceStatus OpRegistry::Register(
    const std::string &op_type,
    const DeviceType device_type,
    const DataType dt,
    OpRegistrationInfo::OpCreator creator) {
  if (registry_.count(op_type) == 0) {
    registry_[op_type] = std::unique_ptr<OpRegistrationInfo>(
        new OpRegistrationInfo);
  }
  registry_[op_type]->AddDevice(device_type);

  std::string op_key = OpKeyBuilder(op_type)
      .Device(device_type)
      .TypeConstraint("T", dt)
      .Build();
  registry_.at(op_type)->Register(op_key, creator);
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus OpRegistry::Register(
    const OpConditionBuilder &builder) {
  std::string op_type = builder.type();
  if (registry_.count(op_type) == 0) {
    registry_[op_type] = std::unique_ptr<OpRegistrationInfo>(
        new OpRegistrationInfo);
  }
  builder.Finalize(registry_[op_type].get());
  return MaceStatus::MACE_SUCCESS;
}

const std::set<DeviceType> OpRegistry::AvailableDevices(
    const std::string &op_type, OpConditionContext *context) const {
  MACE_CHECK(registry_.count(op_type) != 0,
             op_type, " operation is not registered.");

  return registry_.at(op_type)->device_placer(context);
}

void OpRegistry::GetInOutMemoryTypes(
    const std::string &op_type,
    OpConditionContext *context) const {
  MACE_CHECK(registry_.count(op_type) != 0,
             op_type, " operation is not registered. op_type=", op_type);
  return registry_.at(op_type)->memory_type_setter(context);
}

const std::vector<DataFormat> OpRegistry::InputsDataFormat(
    const std::string &op_type,
    OpConditionContext *context) const {
  MACE_CHECK(registry_.count(op_type) != 0,
             op_type, " operation is not registered.");
  return registry_.at(op_type)->data_format_selector(context);
}

std::unique_ptr<Operation> OpRegistry::CreateOperation(
    OpConstructContext *context,
    DeviceType device_type) const {
  auto operator_def = context->operator_def();
  DataType dtype = static_cast<DataType>(
      ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
          *operator_def, "T", static_cast<int>(DT_FLOAT)));
  VLOG(1) << "Creating operator " << operator_def->name() << "("
          << operator_def->type() << "<" << dtype << ">" << ") on "
          << device_type;
  const std::string op_type = context->operator_def()->type();
  MACE_CHECK(registry_.count(op_type) != 0,
             op_type, " operation is not registered.");

  auto key_dtype =
      (device_type == DeviceType::GPU && dtype == DT_HALF) ? DT_FLOAT : dtype;
  std::string key = OpKeyBuilder(op_type)
      .Device(device_type)
      .TypeConstraint("T", key_dtype)
      .Build();
  if (registry_.at(op_type)->creators.count(key) == 0) {
    LOG(FATAL) << "Key not registered: " << key
               << ", op type is: " << operator_def->type();
  }
  return registry_.at(op_type)->creators.at(key)(context);
}

}  // namespace mace
