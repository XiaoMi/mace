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

#ifndef MACE_OPS_DECONV_2D_H_
#define MACE_OPS_DECONV_2D_H_

#include <algorithm>
#include <string>
#include <vector>

#include "mace/core/operator.h"
#include "mace/core/types.h"
#include "mace/ops/activation.h"
#include "mace/ops/common/conv_pool_2d_util.h"

namespace mace {
namespace ops {

class Deconv2dOpBase : public Operation {
 public:
  explicit Deconv2dOpBase(OpConstructContext *context)
      : Operation(context),
        strides_(Operation::GetRepeatedArgs<int>("strides")),
        padding_type_(static_cast<Padding>(Operation::GetOptionalArg<int>(
            "padding", static_cast<int>(SAME)))),
        paddings_(Operation::GetRepeatedArgs<int>("padding_values")),
        group_(Operation::GetOptionalArg<int>("group", 1)),
        model_type_(static_cast<FrameworkType>(
                        Operation::GetOptionalArg<int>("framework_type", 0))),
        activation_(ops::StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation",
                                                   "NOOP"))),
        relux_max_limit_(
            Operation::GetOptionalArg<float>("max_limit", 0.0f)),
        leakyrelu_coefficient_(
            Operation::GetOptionalArg<float>("leakyrelu_coefficient", 0.0f)) {}

 protected:
  std::vector<int> strides_;  // [stride_h, stride_w]
  const Padding padding_type_;
  std::vector<int> paddings_;
  const int group_;
  const FrameworkType model_type_;
  const ActivationType activation_;
  const float relux_max_limit_;
  const float leakyrelu_coefficient_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_DECONV_2D_H_
