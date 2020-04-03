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

#include "mace/ops/registry/registry.h"

namespace mace {

namespace ops {
// Keep in lexicographical order
extern void RegisterActivation(OpRegistry *op_registry);
extern void RegisterAddN(OpRegistry *op_registry);
extern void RegisterArgMax(OpRegistry *op_registry);
extern void RegisterBatchNorm(OpRegistry *op_registry);
extern void RegisterBatchToSpaceND(OpRegistry *op_registry);
extern void RegisterBiasAdd(OpRegistry *op_registry);
extern void RegisterCast(OpRegistry *op_registry);
extern void RegisterChannelShuffle(OpRegistry *op_registry);
extern void RegisterConcat(OpRegistry *op_registry);
extern void RegisterConv2D(OpRegistry *op_registry);
extern void RegisterCrop(OpRegistry *op_registry);
extern void RegisterCumsum(OpRegistry *op_registry);
extern void RegisterDeconv2D(OpRegistry *op_registry);
extern void RegisterDepthToSpace(OpRegistry *op_registry);
extern void RegisterDepthwiseConv2d(OpRegistry *op_registry);
extern void RegisterDepthwiseDeconv2d(OpRegistry *op_registry);
extern void RegisterDynamicLSTM(OpRegistry *op_registry);
extern void RegisterEltwise(OpRegistry *op_registry);
extern void RegisterExpandDims(OpRegistry *op_registry);
extern void RegisterExtractPooling(OpRegistry *op_registry);
extern void RegisterFill(OpRegistry *op_registry);
extern void RegisterFullyConnected(OpRegistry *op_registry);
extern void RegisterGather(OpRegistry *op_registry);
extern void RegisterIdentity(OpRegistry *op_registry);
extern void RegisterIfDefined(OpRegistry *op_registry);
extern void RegisterInferConv2dShape(OpRegistry *op_registry);
extern void RegisterKaldiBatchNorm(OpRegistry *op_registry);
extern void RegisterLocalResponseNorm(OpRegistry *op_registry);
extern void RegisterLpNorm(OpRegistry *op_registry);
extern void RegisterLSTMNonlinear(OpRegistry *op_registry);
extern void RegisterMatMul(OpRegistry *op_registry);
extern void RegisterMVNorm(OpRegistry *op_registry);
extern void RegisterOneHot(OpRegistry *op_registry);
extern void RegisterPad(OpRegistry *op_registry);
extern void RegisterPadContext(OpRegistry *op_registry);
extern void RegisterPNorm(OpRegistry *op_registry);
extern void RegisterPooling(OpRegistry *op_registry);
extern void RegisterReduce(OpRegistry *op_registry);
extern void RegisterReplaceIndex(OpRegistry *op_registry);
extern void RegisterPriorBox(OpRegistry *op_registry);
extern void RegisterReshape(OpRegistry *op_registry);
extern void RegisterResizeBicubic(OpRegistry *op_registry);
extern void RegisterResizeBilinear(OpRegistry *op_registry);
extern void RegisterResizeNearestNeighbor(OpRegistry *op_registry);
extern void RegisterReverse(OpRegistry *op_registry);
extern void RegisterScalarMath(OpRegistry *op_registry);
extern void RegisterSelect(OpRegistry *op_registry);
extern void RegisterShape(OpRegistry *op_registry);
extern void RegisterSlice(OpRegistry *op_registry);
extern void RegisterSoftmax(OpRegistry *op_registry);
extern void RegisterSpaceToBatchND(OpRegistry *op_registry);
extern void RegisterSpaceToDepth(OpRegistry *op_registry);
extern void RegisterSplice(OpRegistry *op_registry);
extern void RegisterSplit(OpRegistry *op_registry);
extern void RegisterSqrDiffMean(OpRegistry *op_registry);
extern void RegisterSqueeze(OpRegistry *op_registry);
extern void RegisterStack(OpRegistry *op_registry);
extern void RegisterStridedSlice(OpRegistry *op_registry);
extern void RegisterSubsample(OpRegistry *op_registry);
extern void RegisterSumGroup(OpRegistry *op_registry);
extern void RegisterTargetRMSNorm(OpRegistry *op_registry);
extern void RegisterTile(OpRegistry *op_registry);
extern void RegisterTranspose(OpRegistry *op_registry);
extern void RegisterUnstack(OpRegistry *op_registry);
extern void RegisterUnsqueeze(OpRegistry *op_registry);

#ifdef MACE_ENABLE_QUANTIZE
extern void RegisterDequantize(OpRegistry *op_registry);
extern void RegisterQuantize(OpRegistry *op_registry);
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
extern void RegisterBufferTransform(OpRegistry *op_registry);
extern void RegisterLSTMCell(OpRegistry *op_registry);
#endif  // MACE_ENABLE_OPENCL


void RegisterAllOps(OpRegistry *registry) {
  // Keep in lexicographical order
  ops::RegisterActivation(registry);
  ops::RegisterAddN(registry);
  ops::RegisterArgMax(registry);
  ops::RegisterBatchNorm(registry);
  ops::RegisterBatchToSpaceND(registry);
  ops::RegisterBiasAdd(registry);
  ops::RegisterCast(registry);
  ops::RegisterChannelShuffle(registry);
  ops::RegisterConcat(registry);
  ops::RegisterConv2D(registry);
  ops::RegisterCrop(registry);
  ops::RegisterCumsum(registry);
  ops::RegisterDeconv2D(registry);
  ops::RegisterDepthToSpace(registry);
  ops::RegisterDepthwiseConv2d(registry);
  ops::RegisterDepthwiseDeconv2d(registry);
  ops::RegisterDynamicLSTM(registry);
  ops::RegisterEltwise(registry);
  ops::RegisterExpandDims(registry);
  ops::RegisterExtractPooling(registry);
  ops::RegisterFill(registry);
  ops::RegisterFullyConnected(registry);
  ops::RegisterGather(registry);
  ops::RegisterIdentity(registry);
  ops::RegisterIfDefined(registry);
  ops::RegisterInferConv2dShape(registry);
  ops::RegisterKaldiBatchNorm(registry);
  ops::RegisterLocalResponseNorm(registry);
  ops::RegisterLpNorm(registry);
  ops::RegisterLSTMNonlinear(registry);
  ops::RegisterMatMul(registry);
  ops::RegisterMVNorm(registry);
  ops::RegisterOneHot(registry);
  ops::RegisterPad(registry);
  ops::RegisterPadContext(registry);
  ops::RegisterPNorm(registry);
  ops::RegisterPooling(registry);
  ops::RegisterReduce(registry);
  ops::RegisterReplaceIndex(registry);
  ops::RegisterPriorBox(registry);
  ops::RegisterReshape(registry);
  ops::RegisterResizeBicubic(registry);
  ops::RegisterResizeBilinear(registry);
  ops::RegisterResizeNearestNeighbor(registry);
  ops::RegisterReverse(registry);
  ops::RegisterScalarMath(registry);
  ops::RegisterSelect(registry);
  ops::RegisterShape(registry);
  ops::RegisterSlice(registry);
  ops::RegisterSoftmax(registry);
  ops::RegisterSpaceToBatchND(registry);
  ops::RegisterSpaceToDepth(registry);
  ops::RegisterSplice(registry);
  ops::RegisterSplit(registry);
  ops::RegisterStack(registry);
  ops::RegisterStridedSlice(registry);
  ops::RegisterSqrDiffMean(registry);
  ops::RegisterSqueeze(registry);
  ops::RegisterSubsample(registry);
  ops::RegisterSumGroup(registry);
  ops::RegisterTargetRMSNorm(registry);
  ops::RegisterTile(registry);
  ops::RegisterTranspose(registry);
  ops::RegisterUnstack(registry);
  ops::RegisterUnsqueeze(registry);

#ifdef MACE_ENABLE_QUANTIZE
  ops::RegisterDequantize(registry);
  ops::RegisterQuantize(registry);
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
  ops::RegisterBufferTransform(registry);
  ops::RegisterLSTMCell(registry);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
