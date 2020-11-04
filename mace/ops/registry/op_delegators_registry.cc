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
extern void RegisterConv2dDelegator(OpDelegatorRegistry *registry);
extern void RegisterDeconv2dDelegator(OpDelegatorRegistry *registry);
extern void RegisterDepthwiseConv2dDelegator(OpDelegatorRegistry *registry);
extern void RegisterDepthwiseDeconv2dDelegator(OpDelegatorRegistry *registry);
extern void RegisterGemmDelegator(OpDelegatorRegistry *registry);
extern void RegisterGemvDelegator(OpDelegatorRegistry *registry);

#ifdef MACE_ENABLE_QUANTIZE
namespace q8 {
extern void RegisterEltwiseDelegator(OpDelegatorRegistry *registry);
extern void RegisterGemvDelegator(OpDelegatorRegistry *registry);
}  // namespace q8
#endif  // MACE_ENABLE_QUANTIZE
}  // namespace ref

#ifdef MACE_ENABLE_NEON
namespace arm {
extern void RegisterConv2dK3x3WinogradDelegator(OpDelegatorRegistry *registry);

extern void RegisterActivationDelegator(OpDelegatorRegistry *registry);
extern void RegisterBiasAddDelegator(OpDelegatorRegistry *registry);

extern void RegisterConv2dK1x1Delegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dK1xNDelegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dK3x3Delegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dK5x5Delegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dK7x7Delegator(OpDelegatorRegistry *registry);
extern void RegisterConv2dGeneralDelegator(OpDelegatorRegistry *registry);

extern void RegisterDeconv2dK2x2Delegator(OpDelegatorRegistry *registry);
extern void RegisterDeconv2dK3x3Delegator(OpDelegatorRegistry *registry);
extern void RegisterDeconv2dK4x4Delegator(OpDelegatorRegistry *registry);
extern void RegisterDeconv2dGeneralDelegator(OpDelegatorRegistry *registry);

extern void RegisterDepthwiseConv2dK3x3Delegator(
    OpDelegatorRegistry *registry);
extern void RegisterDepthwiseDeconv2dK3x3Delegator(
    OpDelegatorRegistry *registry);
extern void RegisterGroupDeconv2dK3x3Delegator(OpDelegatorRegistry *registry);
extern void RegisterDepthwiseDeconv2dK4x4Delegator(
    OpDelegatorRegistry *registry);
extern void RegisterGroupDeconv2dK4x4Delegator(OpDelegatorRegistry *registry);
extern void RegisterDepthwiseDeconv2dGeneralDelegator(
    OpDelegatorRegistry *registry);
extern void RegisterGroupDeconv2dGeneralDelegator(
    OpDelegatorRegistry *registry);

extern void RegisterGemmDelegator(OpDelegatorRegistry *registry);
extern void RegisterGemvDelegator(OpDelegatorRegistry *registry);
#ifdef MACE_ENABLE_FP16
extern void RegisterFP16DepthwiseConv2dK3x3Delegator(
    OpDelegatorRegistry *registry);
extern void RegisterFP16GemmDelegator(OpDelegatorRegistry *registry);
#endif
#ifdef MACE_ENABLE_QUANTIZE
namespace q8 {
extern void RegisterEltwiseDelegator(OpDelegatorRegistry *registry);
extern void RegisterGemvDelegator(OpDelegatorRegistry *registry);
}  // namespace q8
#endif  // MACE_ENABLE_QUANTIZE

}  // namespace arm
#endif  // MACE_ENABLE_NEON

void RegisterAllOpDelegators(OpDelegatorRegistry *registry) {
  ref::RegisterActivationDelegator(registry);
  ref::RegisterBiasAddDelegator(registry);
  ref::RegisterConv2dDelegator(registry);
  ref::RegisterDeconv2dDelegator(registry);
  ref::RegisterDepthwiseConv2dDelegator(registry);
  ref::RegisterDepthwiseDeconv2dDelegator(registry);
  ref::RegisterGemmDelegator(registry);
  ref::RegisterGemvDelegator(registry);
#ifdef MACE_ENABLE_QUANTIZE
  ref::q8::RegisterEltwiseDelegator(registry);
  ref::q8::RegisterGemvDelegator(registry);
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_NEON
  arm::RegisterConv2dK3x3WinogradDelegator(registry);

  arm::RegisterActivationDelegator(registry);
  arm::RegisterBiasAddDelegator(registry);

  arm::RegisterConv2dK1x1Delegator(registry);
  arm::RegisterConv2dK1xNDelegator(registry);
  arm::RegisterConv2dK3x3Delegator(registry);
  arm::RegisterConv2dK5x5Delegator(registry);
  arm::RegisterConv2dK7x7Delegator(registry);
  arm::RegisterConv2dGeneralDelegator(registry);

  arm::RegisterDeconv2dK2x2Delegator(registry);
  arm::RegisterDeconv2dK3x3Delegator(registry);
  arm::RegisterDeconv2dK4x4Delegator(registry);
  arm::RegisterDeconv2dGeneralDelegator(registry);

  arm::RegisterDepthwiseConv2dK3x3Delegator(registry);
  arm::RegisterDepthwiseDeconv2dK3x3Delegator(registry);
  arm::RegisterGroupDeconv2dK3x3Delegator(registry);
  arm::RegisterDepthwiseDeconv2dK4x4Delegator(registry);
  arm::RegisterGroupDeconv2dK4x4Delegator(registry);
  arm::RegisterDepthwiseDeconv2dGeneralDelegator(registry);
  arm::RegisterGroupDeconv2dGeneralDelegator(registry);

  arm::RegisterGemmDelegator(registry);
  arm::RegisterGemvDelegator(registry);
#ifdef MACE_ENABLE_FP16
  arm::RegisterFP16DepthwiseConv2dK3x3Delegator(registry);
  arm::RegisterFP16GemmDelegator(registry);
#endif
#ifdef MACE_ENABLE_QUANTIZE
  arm::q8::RegisterEltwiseDelegator(registry);
  arm::q8::RegisterGemvDelegator(registry);
#endif  // MACE_ENABLE_QUANTIZE

#endif  // MACE_ENABLE_NEON
}

}  // namespace ops
}  // namespace mace
