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

// This implements matrix-vector multiplication described as
// https://github.com/google/gemmlowp/blob/master/todo/fast-gemv.txt

#ifndef MACE_OPS_DELEGATOR_ELTWISE_H_
#define MACE_OPS_DELEGATOR_ELTWISE_H_

#include "mace/core/ops/op_context.h"
#include "mace/core/ops/op_delegator.h"
#include "mace/core/registry/op_delegator_registry.h"
#include "mace/core/types.h"
#include "mace/ops/common/eltwise_type.h"

namespace mace {
namespace ops {
namespace delegator {

struct EltwiseParam : public DelegatorParam {
  explicit EltwiseParam(EltwiseType type)
      : type_(type) {}

  EltwiseType type_;
};

class Eltwise : public OpDelegator {
 public:
  explicit Eltwise(const EltwiseParam &param) : OpDelegator(param),
                                                type_(param.type_) {}
  virtual ~Eltwise() = default;

  MACE_DEFINE_DELEGATOR_CREATOR(Eltwise)

  virtual MaceStatus Compute(const OpContext *context, const Tensor *input0,
                             const Tensor *input1, Tensor *output) = 0;

 protected:
  EltwiseType type_;
};

}  // namespace delegator
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_DELEGATOR_ELTWISE_H_
