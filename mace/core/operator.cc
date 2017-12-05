//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"

namespace mace {


OpKeyBuilder::OpKeyBuilder(const char *op_name): op_name_(op_name) {}

OpKeyBuilder &OpKeyBuilder::TypeConstraint(const char *attr_name,
                                           const DataType allowed) {
  type_constraint_[attr_name] = allowed;
  return *this;
}

const std::string OpKeyBuilder::Build() {
  static const std::vector<std::string> type_order = {"T"};
  std::string key = op_name_;
  for (auto type : type_order) {
    key += type + "_" + DataTypeToString(type_constraint_[type]);
  }
  return key;
}

std::map<int32_t, OperatorRegistry *> *gDeviceTypeRegistry() {
  static std::map<int32_t, OperatorRegistry *> g_device_type_registry;
  return &g_device_type_registry;
}

MACE_DEFINE_REGISTRY(CPUOperatorRegistry,
                     OperatorBase,
                     const OperatorDef &,
                     Workspace *);
MACE_REGISTER_DEVICE_TYPE(DeviceType::CPU, CPUOperatorRegistry);

MACE_DEFINE_REGISTRY(NEONOperatorRegistry,
                     OperatorBase,
                     const OperatorDef &,
                     Workspace *);
MACE_REGISTER_DEVICE_TYPE(DeviceType::NEON, NEONOperatorRegistry);

MACE_DEFINE_REGISTRY(OPENCLOperatorRegistry,
                     OperatorBase,
                     const OperatorDef &,
                     Workspace *);
MACE_REGISTER_DEVICE_TYPE(DeviceType::OPENCL, OPENCLOperatorRegistry);

unique_ptr<OperatorBase> CreateOperator(const OperatorDef &operator_def,
                                        Workspace *ws,
                                        DeviceType type,
                                        const NetMode mode) {
  OperatorRegistry *registry = gDeviceTypeRegistry()->at(type);
  const int dtype = ArgumentHelper::GetSingleArgument<OperatorDef, int>(operator_def,
                                                                        "T",
                                                                        static_cast<int>(DT_FLOAT));
  const int op_mode_i= ArgumentHelper::GetSingleArgument<OperatorDef, int>(operator_def,
                                                                        "mode",
                                                                        static_cast<int>(NetMode::NORMAL));
  const NetMode op_mode = static_cast<NetMode>(op_mode_i);
  if (op_mode == mode) {
    return registry->Create(OpKeyBuilder(operator_def.type().data())
                                .TypeConstraint("T", static_cast<DataType>(dtype))
                                .Build(),
                            operator_def,
                            ws);
  } else {
    return nullptr;
  }
}

OperatorBase::OperatorBase(const OperatorDef &operator_def, Workspace *ws)
    : operator_ws_(ws),
      operator_def_(std::make_shared<OperatorDef>(operator_def)) {}

}  // namespace mace
