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

#include "mace/ops/ops_register.h"

namespace mace {

namespace ops {
// Keep in lexicographical order
extern void Register_Activation(OperatorRegistryBase *op_registry);
extern void Register_AddN(OperatorRegistryBase *op_registry);
extern void Register_ArgMax(OperatorRegistryBase *op_registry);
extern void Register_BatchNorm(OperatorRegistryBase *op_registry);
extern void Register_BatchToSpaceND(OperatorRegistryBase *op_registry);
extern void Register_BiasAdd(OperatorRegistryBase *op_registry);
extern void Register_Cast(OperatorRegistryBase *op_registry);
extern void Register_ChannelShuffle(OperatorRegistryBase *op_registry);
extern void Register_Concat(OperatorRegistryBase *op_registry);
extern void Register_Conv2D(OperatorRegistryBase *op_registry);
extern void Register_Crop(OperatorRegistryBase *op_registry);
extern void Register_Deconv2D(OperatorRegistryBase *op_registry);
extern void Register_DepthToSpace(OperatorRegistryBase *op_registry);
extern void Register_DepthwiseConv2d(OperatorRegistryBase *op_registry);
extern void Register_Dequantize(OperatorRegistryBase *op_registry);
extern void Register_Eltwise(OperatorRegistryBase *op_registry);
extern void Register_Fill(OperatorRegistryBase *op_registry);
extern void Register_FoldedBatchNorm(OperatorRegistryBase *op_registry);
extern void Register_FullyConnected(OperatorRegistryBase *op_registry);
extern void Register_Gather(OperatorRegistryBase *op_registry);
extern void Register_Identity(OperatorRegistryBase *op_registry);
extern void Register_InferConv2dShape(OperatorRegistryBase *op_registry);
extern void Register_LocalResponseNorm(OperatorRegistryBase *op_registry);
extern void Register_MatMul(OperatorRegistryBase *op_registry);
extern void Register_Pad(OperatorRegistryBase *op_registry);
extern void Register_Pooling(OperatorRegistryBase *op_registry);
extern void Register_Proposal(OperatorRegistryBase *op_registry);
extern void Register_Quantize(OperatorRegistryBase *op_registry);
extern void Register_ReduceMean(OperatorRegistryBase *op_registry);
extern void Register_Reshape(OperatorRegistryBase *op_registry);
extern void Register_ResizeBicubic(OperatorRegistryBase *op_registry);
extern void Register_ResizeBilinear(OperatorRegistryBase *op_registry);
extern void Register_ScalarMath(OperatorRegistryBase *op_registry);
extern void Register_Shape(OperatorRegistryBase *op_registry);
extern void Register_Split(OperatorRegistryBase *op_registry);
extern void Register_Softmax(OperatorRegistryBase *op_registry);
extern void Register_Stack(OperatorRegistryBase *op_registry);
extern void Register_StridedSlice(OperatorRegistryBase *op_registry);
extern void Register_SpaceToBatchND(OperatorRegistryBase *op_registry);
extern void Register_SpaceToDepth(OperatorRegistryBase *op_registry);
extern void Register_Squeeze(OperatorRegistryBase *op_registry);
extern void Register_Transpose(OperatorRegistryBase *op_registry);
extern void Register_WinogradInverseTransform(OperatorRegistryBase *op_registry);  // NOLINT(whitespace/line_length)
extern void Register_WinogradTransform(OperatorRegistryBase *op_registry);

#ifdef MACE_ENABLE_OPENCL
extern void Register_BufferToImage(OperatorRegistryBase *op_registry);
extern void Register_ImageToBuffer(OperatorRegistryBase *op_registry);
#endif  // MACE_ENABLE_OPENCL
}  // namespace ops


OperatorRegistry::OperatorRegistry() : OperatorRegistryBase() {
  // Keep in lexicographical order
  ops::Register_Activation(this);
  ops::Register_AddN(this);
  ops::Register_ArgMax(this);
  ops::Register_BatchNorm(this);
  ops::Register_BatchToSpaceND(this);
  ops::Register_BiasAdd(this);
  ops::Register_Cast(this);
  ops::Register_ChannelShuffle(this);
  ops::Register_Concat(this);
  ops::Register_Conv2D(this);
  ops::Register_Crop(this);
  ops::Register_Deconv2D(this);
  ops::Register_DepthToSpace(this);
  ops::Register_DepthwiseConv2d(this);
  ops::Register_Dequantize(this);
  ops::Register_Eltwise(this);
  ops::Register_Fill(this);
  ops::Register_FoldedBatchNorm(this);
  ops::Register_FullyConnected(this);
  ops::Register_Gather(this);
  ops::Register_Identity(this);
  ops::Register_InferConv2dShape(this);
  ops::Register_LocalResponseNorm(this);
  ops::Register_MatMul(this);
  ops::Register_Pad(this);
  ops::Register_Pooling(this);
  ops::Register_Proposal(this);
  ops::Register_Quantize(this);
  ops::Register_ReduceMean(this);
  ops::Register_Reshape(this);
  ops::Register_ResizeBicubic(this);
  ops::Register_ResizeBilinear(this);
  ops::Register_ScalarMath(this);
  ops::Register_Shape(this);
  ops::Register_Split(this);
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
