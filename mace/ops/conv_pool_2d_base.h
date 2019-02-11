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

#ifndef MACE_OPS_CONV_POOL_2D_BASE_H_
#define MACE_OPS_CONV_POOL_2D_BASE_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/ops/common/conv_pool_2d_util.h"

namespace mace {
namespace ops {

class ConvPool2dOpBase : public Operation {
 public:
  explicit ConvPool2dOpBase(OpConstructContext *context)
      : Operation(context),
        strides_(Operation::GetRepeatedArgs<int>("strides")),
        padding_type_(static_cast<Padding>(Operation::GetOptionalArg<int>(
            "padding", static_cast<int>(SAME)))),
        paddings_(Operation::GetRepeatedArgs<int>("padding_values")),
        dilations_(Operation::GetRepeatedArgs<int>("dilations", {1, 1})) {}

 protected:
  std::vector<int> strides_;
  Padding padding_type_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_CONV_POOL_2D_BASE_H_
