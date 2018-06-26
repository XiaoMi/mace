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

std::unique_ptr<OperatorBase> OperatorRegistry::CreateOperator(
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

namespace ops {
// Keep in lexicographical order
extern void Register_Activation(OperatorRegistry *op_registry);
extern void Register_AddN(OperatorRegistry *op_registry);
extern void Register_BatchNorm(OperatorRegistry *op_registry);
extern void Register_BatchToSpaceND(OperatorRegistry *op_registry);
extern void Register_BiasAdd(OperatorRegistry *op_registry);
extern void Register_Cast(OperatorRegistry *op_registry);
extern void Register_ChannelShuffle(OperatorRegistry *op_registry);
extern void Register_Concat(OperatorRegistry *op_registry);
extern void Register_Conv2D(OperatorRegistry *op_registry);
extern void Register_Deconv2D(OperatorRegistry *op_registry);
extern void Register_DepthToSpace(OperatorRegistry *op_registry);
extern void Register_DepthwiseConv2d(OperatorRegistry *op_registry);
extern void Register_Dequantize(OperatorRegistry *op_registry);
extern void Register_Eltwise(OperatorRegistry *op_registry);
extern void Register_FoldedBatchNorm(OperatorRegistry *op_registry);
extern void Register_FullyConnected(OperatorRegistry *op_registry);
extern void Register_Gather(OperatorRegistry *op_registry);
extern void Register_Identity(OperatorRegistry *op_registry);
extern void Register_LocalResponseNorm(OperatorRegistry *op_registry);
extern void Register_MatMul(OperatorRegistry *op_registry);
extern void Register_Pad(OperatorRegistry *op_registry);
extern void Register_Pooling(OperatorRegistry *op_registry);
extern void Register_Proposal(OperatorRegistry *op_registry);
extern void Register_Quantize(OperatorRegistry *op_registry);
extern void Register_ReduceMean(OperatorRegistry *op_registry);
extern void Register_Requantize(OperatorRegistry *op_registry);
extern void Register_Reshape(OperatorRegistry *op_registry);
extern void Register_ResizeBilinear(OperatorRegistry *op_registry);
extern void Register_Shape(OperatorRegistry *op_registry);
extern void Register_Slice(OperatorRegistry *op_registry);
extern void Register_Softmax(OperatorRegistry *op_registry);
extern void Register_Stack(OperatorRegistry *op_registry);
extern void Register_StridedSlice(OperatorRegistry *op_registry);
extern void Register_SpaceToBatchND(OperatorRegistry *op_registry);
extern void Register_SpaceToDepth(OperatorRegistry *op_registry);
extern void Register_Squeeze(OperatorRegistry *op_registry);
extern void Register_Transpose(OperatorRegistry *op_registry);
extern void Register_WinogradInverseTransform(OperatorRegistry *op_registry);
extern void Register_WinogradTransform(OperatorRegistry *op_registry);

#ifdef MACE_ENABLE_OPENCL
extern void Register_BufferToImage(OperatorRegistry *op_registry);
extern void Register_ImageToBuffer(OperatorRegistry *op_registry);
#endif  // MACE_ENABLE_OPENCL
}  // namespace ops

OperatorRegistry::OperatorRegistry() {
  // Keep in lexicographical order
  ops::Register_Activation(this);
  ops::Register_AddN(this);
  ops::Register_BatchNorm(this);
  ops::Register_BatchToSpaceND(this);
  ops::Register_BiasAdd(this);
  ops::Register_Cast(this);
  ops::Register_ChannelShuffle(this);
  ops::Register_Concat(this);
  ops::Register_Conv2D(this);
  ops::Register_Deconv2D(this);
  ops::Register_DepthToSpace(this);
  ops::Register_DepthwiseConv2d(this);
  ops::Register_Dequantize(this);
  ops::Register_Eltwise(this);
  ops::Register_FoldedBatchNorm(this);
  ops::Register_FullyConnected(this);
  ops::Register_Gather(this);
  ops::Register_Identity(this);
  ops::Register_LocalResponseNorm(this);
  ops::Register_MatMul(this);
  ops::Register_Pad(this);
  ops::Register_Pooling(this);
  ops::Register_Proposal(this);
  ops::Register_Quantize(this);
  ops::Register_ReduceMean(this);
  ops::Register_Requantize(this);
  ops::Register_Reshape(this);
  ops::Register_ResizeBilinear(this);
  ops::Register_Shape(this);
  ops::Register_Slice(this);
  ops::Register_Softmax(this);
  ops::Register_Stack(this);
  ops::Register_StridedSlice(this);
  ops::Register_SpaceToBatchND(this);
  ops::Register_SpaceToDepth(this);
  ops::Register_Squeeze(this);
  ops::Register_Transpose(this);
  ops::Register_WinogradInverseTransform(this);
  ops::Register_WinogradTransform(this);

#ifdef MACE_ENABLE_OPENCL
  ops::Register_BufferToImage(this);
  ops::Register_ImageToBuffer(this);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace mace
