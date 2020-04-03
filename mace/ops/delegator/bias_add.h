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

#ifndef MACE_OPS_DELEGATOR_BIAS_ADD_H_
#define MACE_OPS_DELEGATOR_BIAS_ADD_H_

#include "mace/core/ops/op_context.h"
#include "mace/core/ops/op_delegator.h"
#include "mace/core/registry/op_delegator_registry.h"

namespace mace {
namespace ops {
namespace delegator {

class BiasAdd : public OpDelegator {
 public:
  explicit BiasAdd(const DelegatorParam &param) : OpDelegator(param) {}
  virtual ~BiasAdd() = default;

  MACE_DEFINE_DELEGATOR_CREATOR(BiasAdd)

  virtual MaceStatus Compute(const OpContext *context,
                             const Tensor *input,
                             const Tensor *bias,
                             Tensor *output) = 0;
};

}  // namespace delegator
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_DELEGATOR_BIAS_ADD_H_
