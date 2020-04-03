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

namespace ref {
extern void RegisterActivationDelegator(OpDelegatorRegistry *registry);
extern void RegisterBiasAddDelegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dRefDelegator(OpDelegatorRegistry *registry);
extern void RegisterDeconv2dRefDelegator(OpDelegatorRegistry *registry);
extern void RegisterDepthwiseConv2dRefDelegator(OpDelegatorRegistry *registry);
extern void RegisterDepthwiseDeconv2dRefDelegator(
    OpDelegatorRegistry *registry);
extern void RegisterGemmRefDelegator(OpDelegatorRegistry *registry);
extern void RegisterGemvRefDelegator(OpDelegatorRegistry *registry);

#ifdef MACE_ENABLE_QUANTIZE
namespace q8 {
extern void RegisterEltwiseDelegator(OpDelegatorRegistry *registry);
}  // namespace q8
extern void RegisterGemvUint8RefDelegator(OpDelegatorRegistry *registry);
#endif  // MACE_ENABLE_QUANTIZE
}  // namespace ref

#ifdef MACE_ENABLE_NEON
namespace arm {
namespace fp32 {
extern void RegisterActivationDelegator(OpDelegatorRegistry *registry);
extern void RegisterBiasAddDelegator(OpDelegatorRegistry *registry);

extern void RegisterConv2dK1x1Delegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dK1x7S1Delegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dK7x1S1Delegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dK1x15S1Delegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dK15x1S1Delegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dK3x3S1Delegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dK3x3S2Delegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dK3x3WinogradDelegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dK5x5S1Delegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dK7x7S1Delegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dK7x7S2Delegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dK7x7S3Delegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dGeneralDelegator(OpDelegatorRegistry *registry);

extern void RegisterDeconv2dK2x2S1Delegator(OpDelegatorRegistry *registry);
extern void RegisterDeconv2dK2x2S2Delegator(OpDelegatorRegistry *registry);
extern void RegisterDeconv2dK3x3S1Delegator(OpDelegatorRegistry *registry);
extern void RegisterDeconv2dK3x3S2Delegator(OpDelegatorRegistry *registry);
extern void RegisterDeconv2dK4x4S1Delegator(OpDelegatorRegistry *registry);
extern void RegisterDeconv2dK4x4S2Delegator(OpDelegatorRegistry *registry);
extern void RegisterDeconv2dGeneralDelegator(OpDelegatorRegistry *registry);

extern void RegisterDepthwiseConv2dK3x3S1Delegator(
    OpDelegatorRegistry *registry);
extern void RegisterDepthwiseConv2dK3x3S2Delegator(
    OpDelegatorRegistry *registry);
extern void RegisterDepthwiseDeconv2dK3x3S1Delegator(
    OpDelegatorRegistry *registry);
extern void RegisterDepthwiseDeconv2dK3x3S2Delegator(
    OpDelegatorRegistry *registry);
extern void RegisterGroupDeconv2dK3x3S1Delegator(OpDelegatorRegistry *registry);
extern void RegisterGroupDeconv2dK3x3S2Delegator(OpDelegatorRegistry *registry);
extern void RegisterDepthwiseDeconv2dK4x4S1Delegator(
    OpDelegatorRegistry *registry);
extern void RegisterDepthwiseDeconv2dK4x4S2Delegator(
    OpDelegatorRegistry *registry);
extern void RegisterGroupDeconv2dK4x4S1Delegator(OpDelegatorRegistry *registry);
extern void RegisterGroupDeconv2dK4x4S2Delegator(OpDelegatorRegistry *registry);
extern void RegisterDepthwiseDeconv2dGeneralDelegator(
    OpDelegatorRegistry *registry);
extern void RegisterGroupDeconv2dGeneralDelegator(
    OpDelegatorRegistry *registry);

extern void RegisterGemmDelegator(OpDelegatorRegistry *registry);
extern void RegisterGemvDelegator(OpDelegatorRegistry *registry);
}  // namespace fp32

#ifdef MACE_ENABLE_QUANTIZE
namespace q8 {
extern void RegisterEltwiseDelegator(OpDelegatorRegistry *registry);
extern void RegisterGemvUint8Delegator(OpDelegatorRegistry *registry);
extern void RegisterGemvInt32Delegator(OpDelegatorRegistry *registry);
}  // namespace q8
#endif  // MACE_ENABLE_QUANTIZE

}  // namespace arm
#endif  // MACE_ENABLE_NEON

void RegisterAllOpDelegators(OpDelegatorRegistry *registry) {
  ref::RegisterActivationDelegator(registry);
  ref::RegisterBiasAddDelegator(registry);
  ref::RegisterConv2dRefDelegator(registry);
  ref::RegisterDeconv2dRefDelegator(registry);
  ref::RegisterDepthwiseConv2dRefDelegator(registry);
  ref::RegisterDepthwiseDeconv2dRefDelegator(registry);
  ref::RegisterGemmRefDelegator(registry);
  ref::RegisterGemvRefDelegator(registry);

#ifdef MACE_ENABLE_QUANTIZE
  ref::q8::RegisterEltwiseDelegator(registry);
  ref::RegisterGemvUint8RefDelegator(registry);
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_NEON
  arm::fp32::RegisterActivationDelegator(registry);
  arm::fp32::RegisterBiasAddDelegator(registry);

  arm::fp32::RegisterConv2dK1x1Delegator(registry);
  arm::fp32::RegisterConv2dK1x7S1Delegator(registry);
  arm::fp32::RegisterConv2dK7x1S1Delegator(registry);
  arm::fp32::RegisterConv2dK1x15S1Delegator(registry);
  arm::fp32::RegisterConv2dK15x1S1Delegator(registry);
  arm::fp32::RegisterConv2dK3x3S1Delegator(registry);
  arm::fp32::RegisterConv2dK3x3S2Delegator(registry);
  arm::fp32::RegisterConv2dK3x3WinogradDelegator(registry);
  arm::fp32::RegisterConv2dK5x5S1Delegator(registry);
  arm::fp32::RegisterConv2dK7x7S1Delegator(registry);
  arm::fp32::RegisterConv2dK7x7S2Delegator(registry);
  arm::fp32::RegisterConv2dK7x7S3Delegator(registry);
  arm::fp32::RegisterConv2dGeneralDelegator(registry);

  arm::fp32::RegisterDeconv2dK2x2S1Delegator(registry);
  arm::fp32::RegisterDeconv2dK2x2S2Delegator(registry);
  arm::fp32::RegisterDeconv2dK3x3S1Delegator(registry);
  arm::fp32::RegisterDeconv2dK3x3S2Delegator(registry);
  arm::fp32::RegisterDeconv2dK4x4S1Delegator(registry);
  arm::fp32::RegisterDeconv2dK4x4S2Delegator(registry);
  arm::fp32::RegisterDeconv2dGeneralDelegator(registry);

  arm::fp32::RegisterDepthwiseConv2dK3x3S1Delegator(registry);
  arm::fp32::RegisterDepthwiseConv2dK3x3S2Delegator(registry);
  arm::fp32::RegisterDepthwiseDeconv2dK3x3S1Delegator(registry);
  arm::fp32::RegisterDepthwiseDeconv2dK3x3S2Delegator(registry);
  arm::fp32::RegisterGroupDeconv2dK3x3S1Delegator(registry);
  arm::fp32::RegisterGroupDeconv2dK3x3S2Delegator(registry);
  arm::fp32::RegisterDepthwiseDeconv2dK4x4S1Delegator(registry);
  arm::fp32::RegisterDepthwiseDeconv2dK4x4S2Delegator(registry);
  arm::fp32::RegisterGroupDeconv2dK4x4S1Delegator(registry);
  arm::fp32::RegisterGroupDeconv2dK4x4S2Delegator(registry);
  arm::fp32::RegisterDepthwiseDeconv2dGeneralDelegator(registry);
  arm::fp32::RegisterGroupDeconv2dGeneralDelegator(registry);

  arm::fp32::RegisterGemmDelegator(registry);
  arm::fp32::RegisterGemvDelegator(registry);

#ifdef MACE_ENABLE_QUANTIZE
  arm::q8::RegisterEltwiseDelegator(registry);
  arm::q8::RegisterGemvUint8Delegator(registry);
  arm::q8::RegisterGemvInt32Delegator(registry);
#endif  // MACE_ENABLE_QUANTIZE

#endif  // MACE_ENABLE_NEON
}

}  // namespace ops
}  // namespace mace
