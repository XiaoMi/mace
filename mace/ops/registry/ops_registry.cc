// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include "mace/ops/registry/ops_registry.h"

namespace mace {

namespace ops {
// Keep in lexicographical order
extern void RegisterActivation(OpRegistryBase *op_registry);
extern void RegisterAddN(OpRegistryBase *op_registry);
extern void RegisterArgMax(OpRegistryBase *op_registry);
extern void RegisterBatchNorm(OpRegistryBase *op_registry);
extern void RegisterBatchToSpaceND(OpRegistryBase *op_registry);
extern void RegisterBiasAdd(OpRegistryBase *op_registry);
extern void RegisterCast(OpRegistryBase *op_registry);
extern void RegisterChannelShuffle(OpRegistryBase *op_registry);
extern void RegisterConcat(OpRegistryBase *op_registry);
extern void RegisterConv2D(OpRegistryBase *op_registry);
extern void RegisterCrop(OpRegistryBase *op_registry);
extern void RegisterCumsum(OpRegistryBase *op_registry);
extern void RegisterDeconv2D(OpRegistryBase *op_registry);
extern void RegisterDepthToSpace(OpRegistryBase *op_registry);
extern void RegisterDepthwiseConv2d(OpRegistryBase *op_registry);
extern void RegisterDepthwiseDeconv2d(OpRegistryBase *op_registry);
extern void RegisterDynamicLSTM(OpRegistryBase *op_registry);
extern void RegisterEltwise(OpRegistryBase *op_registry);
extern void RegisterExpandDims(OpRegistryBase *op_registry);
extern void RegisterExtractPooling(OpRegistryBase *op_registry);
extern void RegisterFill(OpRegistryBase *op_registry);
extern void RegisterFullyConnected(OpRegistryBase *op_registry);
extern void RegisterGather(OpRegistryBase *op_registry);
extern void RegisterIdentity(OpRegistryBase *op_registry);
extern void RegisterDelay(OpRegistryBase *op_registry);
extern void RegisterInferConv2dShape(OpRegistryBase *op_registry);
extern void RegisterKaldiBatchNorm(OpRegistryBase *op_registry);
extern void RegisterLocalResponseNorm(OpRegistryBase *op_registry);
extern void RegisterLSTMNonlinear(OpRegistryBase *op_registry);
extern void RegisterMatMul(OpRegistryBase *op_registry);
extern void RegisterOneHot(OpRegistryBase *op_registry);
extern void RegisterPad(OpRegistryBase *op_registry);
extern void RegisterPadContext(OpRegistryBase *op_registry);
extern void RegisterPNorm(OpRegistryBase *op_registry);
extern void RegisterPooling(OpRegistryBase *op_registry);
extern void RegisterReduce(OpRegistryBase *op_registry);
extern void RegisterPriorBox(OpRegistryBase *op_registry);
extern void RegisterReshape(OpRegistryBase *op_registry);
extern void RegisterResizeBicubic(OpRegistryBase *op_registry);
extern void RegisterResizeBilinear(OpRegistryBase *op_registry);
extern void RegisterResizeNearestNeighbor(OpRegistryBase *op_registry);
extern void RegisterReverse(OpRegistryBase *op_registry);
extern void RegisterScalarMath(OpRegistryBase *op_registry);
extern void RegisterShape(OpRegistryBase *op_registry);
extern void RegisterSlice(OpRegistryBase *op_registry);
extern void RegisterSoftmax(OpRegistryBase *op_registry);
extern void RegisterSpaceToBatchND(OpRegistryBase *op_registry);
extern void RegisterSpaceToDepth(OpRegistryBase *op_registry);
extern void RegisterSplice(OpRegistryBase *op_registry);
extern void RegisterSplit(OpRegistryBase *op_registry);
extern void RegisterSqrDiffMean(OpRegistryBase *op_registry);
extern void RegisterSqueeze(OpRegistryBase *op_registry);
extern void RegisterStack(OpRegistryBase *op_registry);
extern void RegisterStridedSlice(OpRegistryBase *op_registry);
extern void RegisterSumGroup(OpRegistryBase *op_registry);
extern void RegisterTargetRMSNorm(OpRegistryBase *op_registry);
extern void RegisterTranspose(OpRegistryBase *op_registry);
extern void RegisterUnstack(OpRegistryBase *op_registry);

#ifdef MACE_ENABLE_QUANTIZE
extern void RegisterDequantize(OpRegistryBase *op_registry);
extern void RegisterQuantize(OpRegistryBase *op_registry);
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
extern void RegisterBufferTransform(OpRegistryBase *op_registry);
extern void RegisterLSTMCell(OpRegistryBase *op_registry);
#endif  // MACE_ENABLE_OPENCL
}  // namespace ops


OpRegistry::OpRegistry() : OpRegistryBase() {
  // Keep in lexicographical order
  ops::RegisterActivation(this);
  ops::RegisterAddN(this);
  ops::RegisterArgMax(this);
  ops::RegisterBatchNorm(this);
  ops::RegisterBatchToSpaceND(this);
  ops::RegisterBiasAdd(this);
  ops::RegisterCast(this);
  ops::RegisterChannelShuffle(this);
  ops::RegisterConcat(this);
  ops::RegisterConv2D(this);
  ops::RegisterCrop(this);
  ops::RegisterCumsum(this);
  ops::RegisterDeconv2D(this);
  ops::RegisterDepthToSpace(this);
  ops::RegisterDepthwiseConv2d(this);
  ops::RegisterDepthwiseDeconv2d(this);
  ops::RegisterDynamicLSTM(this);
  ops::RegisterEltwise(this);
  ops::RegisterExpandDims(this);
  ops::RegisterExtractPooling(this);
  ops::RegisterFill(this);
  ops::RegisterFullyConnected(this);
  ops::RegisterGather(this);
  ops::RegisterIdentity(this);
  ops::RegisterDelay(this);
  ops::RegisterInferConv2dShape(this);
  ops::RegisterKaldiBatchNorm(this);
  ops::RegisterLocalResponseNorm(this);
  ops::RegisterLSTMNonlinear(this);
  ops::RegisterMatMul(this);
  ops::RegisterOneHot(this);
  ops::RegisterPad(this);
  ops::RegisterPadContext(this);
  ops::RegisterPNorm(this);
  ops::RegisterPooling(this);
  ops::RegisterReduce(this);
  ops::RegisterPriorBox(this);
  ops::RegisterReshape(this);
  ops::RegisterResizeBicubic(this);
  ops::RegisterResizeBilinear(this);
  ops::RegisterResizeNearestNeighbor(this);
  ops::RegisterReverse(this);
  ops::RegisterScalarMath(this);
  ops::RegisterShape(this);
  ops::RegisterSlice(this);
  ops::RegisterSoftmax(this);
  ops::RegisterSpaceToBatchND(this);
  ops::RegisterSpaceToDepth(this);
  ops::RegisterSplice(this);
  ops::RegisterSplit(this);
  ops::RegisterStack(this);
  ops::RegisterStridedSlice(this);
  ops::RegisterSqrDiffMean(this);
  ops::RegisterSqueeze(this);
  ops::RegisterSumGroup(this);
  ops::RegisterTargetRMSNorm(this);
  ops::RegisterTranspose(this);
  ops::RegisterUnstack(this);

#ifdef MACE_ENABLE_QUANTIZE
  ops::RegisterDequantize(this);
  ops::RegisterQuantize(this);
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
  ops::RegisterBufferTransform(this);
  ops::RegisterLSTMCell(this);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace mace
