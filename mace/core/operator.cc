//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"

namespace mace {

std::map<int32_t, OperatorRegistry*>* gDeviceTypeRegistry() {
  static std::map<int32_t, OperatorRegistry*> g_device_type_registry;
  return &g_device_type_registry;
}

MACE_DEFINE_REGISTRY(
    CPUOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
MACE_REGISTER_DEVICE_TYPE(DeviceType::CPU, CPUOperatorRegistry);

MACE_DEFINE_REGISTRY(
    NEONOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
MACE_REGISTER_DEVICE_TYPE(DeviceType::NEON, NEONOperatorRegistry);

unique_ptr<OperatorBase> CreateOperator(
    const OperatorDef& operator_def,
    Workspace* ws,
    DeviceType type) {
  OperatorRegistry* registry = gDeviceTypeRegistry()->at(type);
  return registry->Create(operator_def.type(), operator_def, ws);
}


OperatorBase::OperatorBase(const OperatorDef &operator_def, Workspace *ws)
    : operator_ws_(ws),
      operator_def_(std::make_shared<OperatorDef>(operator_def)) {
}


} // namespace mace
