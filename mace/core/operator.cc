//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

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

namespace ops {

// Keep in lexicographical order
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
extern void Register_Eltwise(OperatorRegistry *op_registry);
extern void Register_FoldedBatchNorm(OperatorRegistry *op_registry);
extern void Register_FullyConnected(OperatorRegistry *op_registry);
extern void Register_FusedConv2D(OperatorRegistry *op_registry);
extern void Register_GlobalAvgPooling(OperatorRegistry *op_registry);
extern void Register_ImageToBuffer(OperatorRegistry *op_registry);
extern void Register_MatMul(OperatorRegistry *op_registry);
extern void Register_Pooling(OperatorRegistry *op_registry);
extern void Register_Proposal(OperatorRegistry *op_registry);
extern void Register_PSROIAlign(OperatorRegistry *op_registry);
extern void Register_ReOrganize(OperatorRegistry *op_registry);
extern void Register_Reshape(OperatorRegistry *op_registry);
extern void Register_ResizeBilinear(OperatorRegistry *op_registry);
extern void Register_Slice(OperatorRegistry *op_registry);
extern void Register_Softmax(OperatorRegistry *op_registry);
extern void Register_SpaceToBatchND(OperatorRegistry *op_registry);
extern void Register_WinogradInverseTransform(OperatorRegistry *op_registry);
extern void Register_WinogradTransform(OperatorRegistry *op_registry);


}  // namespace ops

OperatorRegistry::OperatorRegistry() {
  // Keep in lexicographical order
  ops::Register_Activation(this);
  ops::Register_AddN(this);
  ops::Register_BatchNorm(this);
  ops::Register_BatchToSpaceND(this);
  ops::Register_BiasAdd(this);
  ops::Register_BufferToImage(this);
  ops::Register_ChannelShuffle(this);
  ops::Register_Concat(this);
  ops::Register_Conv2D(this);
  ops::Register_DepthwiseConv2d(this);
  ops::Register_Eltwise(this);
  ops::Register_FoldedBatchNorm(this);
  ops::Register_FullyConnected(this);
  ops::Register_FusedConv2D(this);
  ops::Register_GlobalAvgPooling(this);
  ops::Register_ImageToBuffer(this);
  ops::Register_MatMul(this);
  ops::Register_Pooling(this);
  ops::Register_Proposal(this);
  ops::Register_PSROIAlign(this);
  ops::Register_ReOrganize(this);
  ops::Register_Reshape(this);
  ops::Register_ResizeBilinear(this);
  ops::Register_Slice(this);
  ops::Register_Softmax(this);
  ops::Register_SpaceToBatchND(this);
  ops::Register_WinogradInverseTransform(this);
  ops::Register_WinogradTransform(this);
}

}  // namespace mace
