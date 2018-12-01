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

#include "mace/ops/ops_registry.h"

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
extern void RegisterDeconv2D(OpRegistryBase *op_registry);
extern void RegisterDepthToSpace(OpRegistryBase *op_registry);
extern void RegisterDepthwiseConv2d(OpRegistryBase *op_registry);
extern void RegisterDepthwiseDeconv2d(OpRegistryBase *op_registry);
extern void RegisterEltwise(OpRegistryBase *op_registry);
extern void RegisterExpandDims(OpRegistryBase *op_registry);
extern void RegisterFill(OpRegistryBase *op_registry);
extern void RegisterFullyConnected(OpRegistryBase *op_registry);
extern void RegisterGather(OpRegistryBase *op_registry);
extern void RegisterIdentity(OpRegistryBase *op_registry);
extern void RegisterInferConv2dShape(OpRegistryBase *op_registry);
extern void RegisterLocalResponseNorm(OpRegistryBase *op_registry);
extern void RegisterMatMul(OpRegistryBase *op_registry);
extern void RegisterPad(OpRegistryBase *op_registry);
extern void RegisterPooling(OpRegistryBase *op_registry);
extern void RegisterReduceMean(OpRegistryBase *op_registry);
extern void RegisterReshape(OpRegistryBase *op_registry);
extern void RegisterResizeBicubic(OpRegistryBase *op_registry);
extern void RegisterResizeBilinear(OpRegistryBase *op_registry);
extern void RegisterReverse(OpRegistryBase *op_registry);
extern void RegisterScalarMath(OpRegistryBase *op_registry);
extern void RegisterShape(OpRegistryBase *op_registry);
extern void RegisterSoftmax(OpRegistryBase *op_registry);
extern void RegisterSpaceToBatchND(OpRegistryBase *op_registry);
extern void RegisterSpaceToDepth(OpRegistryBase *op_registry);
extern void RegisterSplit(OpRegistryBase *op_registry);
extern void RegisterSqrDiffMean(OpRegistryBase *op_registry);
extern void RegisterSqueeze(OpRegistryBase *op_registry);
extern void RegisterStack(OpRegistryBase *op_registry);
extern void RegisterStridedSlice(OpRegistryBase *op_registry);
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
  ops::RegisterDeconv2D(this);
  ops::RegisterDepthToSpace(this);
  ops::RegisterDepthwiseConv2d(this);
  ops::RegisterDepthwiseDeconv2d(this);
  ops::RegisterEltwise(this);
  ops::RegisterExpandDims(this);
  ops::RegisterFill(this);
  ops::RegisterFullyConnected(this);
  ops::RegisterGather(this);
  ops::RegisterIdentity(this);
  ops::RegisterInferConv2dShape(this);
  ops::RegisterLocalResponseNorm(this);
  ops::RegisterMatMul(this);
  ops::RegisterPad(this);
  ops::RegisterPooling(this);
  ops::RegisterReduceMean(this);
  ops::RegisterReshape(this);
  ops::RegisterResizeBicubic(this);
  ops::RegisterResizeBilinear(this);
  ops::RegisterReverse(this);
  ops::RegisterScalarMath(this);
  ops::RegisterShape(this);
  ops::RegisterSoftmax(this);
  ops::RegisterSpaceToBatchND(this);
  ops::RegisterSpaceToDepth(this);
  ops::RegisterSplit(this);
  ops::RegisterStack(this);
  ops::RegisterStridedSlice(this);
  ops::RegisterSqrDiffMean(this);
  ops::RegisterSqueeze(this);
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
