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

#ifndef MACE_OPS_ARM_BASE_DECONV_2D_GENERAL_H_
#define MACE_OPS_ARM_BASE_DECONV_2D_GENERAL_H_

#include "mace/ops/arm/base/deconv_2d.h"

// TODO(liutuo): optimize it

namespace mace {
namespace ops {
namespace arm {

template<typename T>
class Deconv2dGeneral : public Deconv2dBase {
 public:
  explicit Deconv2dGeneral(const delegator::Deconv2dParam &param)
      : Deconv2dBase(param, sizeof(T)) {}
  virtual ~Deconv2dGeneral() {}

  MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *output_shape,
      Tensor *output) override;
};

}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_DECONV_2D_GENERAL_H_

