//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <sstream>

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

std::unique_ptr<OperatorBase> OperatorRegistry::CreateOperator(
    const OperatorDef &operator_def,
    Workspace *ws,
    DeviceType type,
    const NetMode mode) const {
  const int dtype = ArgumentHelper::GetSingleArgument<OperatorDef, int>(
      operator_def, "T", static_cast<int>(DT_FLOAT));
  const int op_mode_i = ArgumentHelper::GetSingleArgument<OperatorDef, int>(
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

extern void Register_Activation(OperatorRegistry *op_registry);
extern void Register_AddN(OperatorRegistry *op_registry);
extern void Register_BatchNorm(OperatorRegistry *op_registry);
extern void Register_BatchToSpaceND(OperatorRegistry *op_registry);
extern void Register_BiasAdd(OperatorRegistry *op_registry);
extern void Register_BufferToImage(OperatorRegistry *op_registry);
extern void Register_ChannelShuffle(OperatorRegistry *op_registry);
extern void Register_Concat(OperatorRegistry *op_registry);
extern void Register_Conv2D(OperatorRegistry *op_registry);
extern void Register_DepthwiseConv2d(OperatorRegistry *op_registry);
extern void Register_FoldedBatchNorm(OperatorRegistry *op_registry);
extern void Register_FusedConv2D(OperatorRegistry *op_registry);
extern void Register_GlobalAvgPooling(OperatorRegistry *op_registry);
extern void Register_ImageToBuffer(OperatorRegistry *op_registry);
extern void Register_Pooling(OperatorRegistry *op_registry);
extern void Register_ResizeBilinear(OperatorRegistry *op_registry);
extern void Register_Softmax(OperatorRegistry *op_registry);
extern void Register_SpaceToBatchND(OperatorRegistry *op_registry);

OperatorRegistry::OperatorRegistry() {
  Register_Activation(this);
  Register_AddN(this);
  Register_BatchNorm(this);
  Register_BatchToSpaceND(this);
  Register_BiasAdd(this);
  Register_BufferToImage(this);
  Register_ChannelShuffle(this);
  Register_Concat(this);
  Register_Conv2D(this);
  Register_DepthwiseConv2d(this);
  Register_FoldedBatchNorm(this);
  Register_FusedConv2D(this);
  Register_GlobalAvgPooling(this);
  Register_ImageToBuffer(this);
  Register_Pooling(this);
  Register_ResizeBilinear(this);
  Register_Softmax(this);
  Register_SpaceToBatchND(this);
}

}  // namespace mace
