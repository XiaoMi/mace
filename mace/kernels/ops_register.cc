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

#include "mace/kernels/ops_register.h"

namespace mace {

namespace kernels {
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
extern void RegisterDequantize(OpRegistryBase *op_registry);
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
extern void RegisterQuantize(OpRegistryBase *op_registry);
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
#ifdef MACE_ENABLE_OPENCL
extern void RegisterBufferTransform(OpRegistryBase *op_registry);
extern void RegisterBufferInverseTransform(OpRegistryBase *op_registry);
extern void RegisterLSTMCell(OpRegistryBase *op_registry);
extern void RegisterWinogradInverseTransform(OpRegistryBase *op_registry);
extern void RegisterWinogradTransform(OpRegistryBase *op_registry);

#endif  // MACE_ENABLE_OPENCL
}  // namespace kernels


OpRegistry::OpRegistry() : OpRegistryBase() {
  // Keep in lexicographical order
  kernels::RegisterActivation(this);
  kernels::RegisterAddN(this);
  kernels::RegisterArgMax(this);
  kernels::RegisterBatchNorm(this);
  kernels::RegisterBatchToSpaceND(this);
  kernels::RegisterBiasAdd(this);
  kernels::RegisterCast(this);
  kernels::RegisterChannelShuffle(this);
  kernels::RegisterConcat(this);
  kernels::RegisterConv2D(this);
  kernels::RegisterCrop(this);
  kernels::RegisterDeconv2D(this);
  kernels::RegisterDepthToSpace(this);
  kernels::RegisterDepthwiseConv2d(this);
  kernels::RegisterDequantize(this);
  kernels::RegisterEltwise(this);
  kernels::RegisterExpandDims(this);
  kernels::RegisterFill(this);
  kernels::RegisterFullyConnected(this);
  kernels::RegisterGather(this);
  kernels::RegisterIdentity(this);
  kernels::RegisterInferConv2dShape(this);
  kernels::RegisterLocalResponseNorm(this);
  kernels::RegisterMatMul(this);
  kernels::RegisterPad(this);
  kernels::RegisterPooling(this);
  kernels::RegisterQuantize(this);
  kernels::RegisterReduceMean(this);
  kernels::RegisterReshape(this);
  kernels::RegisterResizeBicubic(this);
  kernels::RegisterResizeBilinear(this);
  kernels::RegisterReverse(this);
  kernels::RegisterScalarMath(this);
  kernels::RegisterShape(this);
  kernels::RegisterSoftmax(this);
  kernels::RegisterSpaceToBatchND(this);
  kernels::RegisterSpaceToDepth(this);
  kernels::RegisterSplit(this);
  kernels::RegisterStack(this);
  kernels::RegisterStridedSlice(this);
  kernels::RegisterSqrDiffMean(this);
  kernels::RegisterSqueeze(this);
  kernels::RegisterTranspose(this);
  kernels::RegisterUnstack(this);
#ifdef MACE_ENABLE_OPENCL
  kernels::RegisterBufferTransform(this);
  kernels::RegisterBufferInverseTransform(this);
  kernels::RegisterLSTMCell(this);
  kernels::RegisterWinogradInverseTransform(this);
  kernels::RegisterWinogradTransform(this);

#endif  // MACE_ENABLE_OPENCL
}

}  // namespace mace
